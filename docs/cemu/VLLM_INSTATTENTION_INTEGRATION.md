# vLLM 接入 CEMU / InstAttention：设计与实施边界

日期：2026-11-02。状态：首版插件代码已实现，设备链路已通过 CPU QEMU 回归；真实 vLLM + Llama/OPT + CUDA 端到端尚未验证。

实现与运行说明见 [vllm_adapter/README.md](../../tests/cemu/instattention/python/vllm_adapter/README.md)。
以下保留设计依据；实际通过自定义 Worker/ModelRunner 子类对接生命周期和 metadata，无需修改 vLLM 源码。
目前每层 Decode 使用一条变长 batch EXEC，支持固定容量会话内的请求映射/重排，尚无设备 KV slot 回收；仍保留受限容量的兼容 GPU KV 分配。
已补充 OPT-13B 的模型入口、MHA 几何和全选择短测试配置，复用现有 HF 权重；运行说明见适配器 README 的 OPT 章节。

## 1. 目标与源码基线

让 vLLM 提供模型加载、模型主体计算、采样和请求调度；让 CEMU 继续提供设备侧 KV 管理与 SparF Attention。保留已有聚合算子、设备内存缓冲、聚块下放、FTL/NAND 建模及 CUDA 动态库执行机制。

- 本地 vLLM：`/home/lihan/cemu/vllm`，commit `eecf8dbb2b279372733c844a9735262494b3fa9b`，`git describe` 为 `v0.33.0rc0-468-geecf8dbb2b`。这是开发快照，本文件中的接口定位以该提交为准，不代表已经选定最终运行版本。
- 模型：`meta-llama/Llama-3.1-8B` Base，本地目录 `/home/lihan/cemu/models/Meta-Llama-3.1-8B`。
- 已读取正在下载目录中的 `config.json`：32 层、32 query heads、8 KV heads、hidden size 4096，因而 head dimension 128；权重配置为 BF16，RoPE 类型为 `llama3`，最大位置长度 131072。读取到配置不代表完整权重已下载完毕。
- 首版选 FDM KV 路径，以保留 `fdm_kv_aggregate.so` 的真实调用。Native 路径和已有 FlexGen 入口继续作为独立对照。

本地 vLLM 的 `pyproject.toml` 要求 Python `>=3.11,<3.15`，CUDA 依赖文件固定 `torch==2.14.0` 并含 cu13 组件。现有下载环境 `hf-env` 为 Python 3.10，不能直接用于安装这份源码。先核对 Guest GPU 驱动/运行库兼容性，再固定版本并创建独立环境；不要为接入任务直接替换既有 Guest/Host CUDA 环境。

## 2. 保留的设备机制

| 机制 | 当前实现入口（相对 CEMU 根目录） | 接入原则 |
|---|---|---|
| SparF channel 选择、近似打分、token 选择、alpha 与均值补偿 | `hw/femu/csd/sparf_workflow.h` | 继续由设备自治 workflow 推进，保持算法与访问依赖 |
| CUDA exact QK/softmax、PV | `tests/cemu/kernel/sparf_attention_devptr.cu` | 继续编译为 `sparf_attention_devptr.so`，使用设备 GPU 功能计算 |
| 共享库加载和目标选择 | `hw/femu/csd/compute.c`，Python `cemu_flexgen/cemu_device.py` | 保留 load/activate/execute 及 `CUDA_DEVICE_POINTER` target |
| KV 收包及聚合 | `hw/femu/fdm_kv_receive.c`、`fdm_kv_binding.c`；`tests/cemu/kernel/fdm_kv_aggregate.c` | KV 先进入设备 MR，调用原聚合模块，保留接收位图、输入槽背压、分段推进 |
| 设备页池及三种布局 | `hw/femu/fdm_kv_buffer.c`、`fdm_kv_layout.c` | 保留 token-K、channel-K、token-V，组页/聚块在设备侧完成 |
| 块写回、完成回收 | `hw/femu/fdm_kv_writer.c` | 满块按现有规则下放；terminal drain 处理剩余部分，等待 NAND 模型完成后回收 |
| 读缓冲尾部或 NAND 数据 | `hw/femu/fdm_kv_attention.c` | 未下放的 KV 仍可参与 Attention；保留读租约与写回协调 |
| Python/C++/Guest 传输 | `tests/cemu/instattention/guest/{pybind_module.cpp,cemu_client.cpp,io_transport.cpp}` | 复用 `_cemu_client`、MRS、FDM read/write、Execute 和现有传输 |
| 目标计时、冻结及阶段依赖 | `hw/femu/csd/compute.c`、`sparf_workflow.h` 和原存储时间模型 | 功能执行、Host 开销、Guest 时间和目标时间继续分别记录 |

这里有两种不同的 `.so`：

1. `fdm_kv_aggregate.so` 由 C 源码构建，当前绑定要求 `PROGRAM_TARGET_HOST`，在 QEMU 设备实现内操作设备缓冲，不是 CUDA 聚合 kernel。
2. `sparf_attention_devptr.so` 由 NVCC 构建，设备 workflow 的 exact QK/softmax 和 PV 阶段通过设备指针运行在 Host 的 CSD 功能 GPU 上。channel/token 选择等控制及功能逻辑目前还在设备 C 代码中，不能把全部 SparF 描述为一个 CUDA kernel。

`compute.c::load_shared_lib()` 在 QEMU 进程调用 `dlopen`/`dlsym`。因此 Guest 提交的算子路径必须能由物理 Host 上的 QEMU 访问；Guest 的 `_cemu_client.so` 与这些设备算子 `.so` 是不同用途的库。vLLM 放在并列仓库不影响复用。

## 3. vLLM 接入位置

本地 `vllm/model_executor/models/llama.py::LlamaAttention.forward()`：

```text
qkv_proj → split Q/K/V → rotary_emb(positions, Q, K)
         → self.attn(Q, K, V) → o_proj
```

`vllm/model_executor/layers/attention/attention.py` 的 `Attention.forward()` 将输入整形成 token/head/dimension，并经 `unified_attention_with_output()` 调用 `self.impl.forward(...)`。在这里注册 CEMU backend，可复用模型原有的 QKV、Llama 3.1 RoPE、RMSNorm、MLP 和输出头。

拟新增 `CemuAttentionBackend`、`CemuAttentionMetadataBuilder`、`CemuAttentionImpl`，使用本地 `vllm/v1/attention/backends/registry.py::register_backend()`。通过 general plugin 注册，并在 worker 内验证实际选择的 backend。注册函数必须可重复调用，不应在 import/父进程预先打开设备 fd、启动线程或创建 CUDA 上下文。

首版指定常规 V1 GPU runner 路径（本地可用 `VLLM_USE_V2_MODEL_RUNNER=0`），避免将 `gpu_model_runner.py` 的接入逻辑误认为自动覆盖 Model Runner V2。最终选定其他版本时须重新核对该开关和接口。

拟设 `forward_includes_kv_cache_update=True`，由 CEMU backend 明确负责每次新 KV 的唯一一次追加；关闭 KV update bypass/fusion 支持，避免 K/V 被上游融合路径吞掉。仍需完整处理 vLLM 的 cache spec、初始化和 profiling 协议。

## 4. 数据流与完成边界

```text
Guest vLLM worker
  Prefill：QKV + RoPE → GPU dense causal Attention → o_proj/MLP
                     └→ 新 K/V → CEMU Python 会话 → _cemu_client
  Decode：QKV + RoPE → 追加本步 K/V → 提交 Q/mean/workflow
                     → 等待设备输出 → GPU o_proj/MLP
                                      │
                                Guest FDM/驱动/NVMe
                                      │
Host QEMU / CEMU
  MEMORY_WRITE → MR1 输入槽 → 原聚合 .so
                  → 设备页池：token-K / channel-K / token-V
                  → 满块 writer → KV FTL / NAND 模型
  Execute → 读 KV 就绪/租约 → SparF 选择及稀疏读取
          → CUDA .so exact QK/softmax → PV + mean 补偿 → 输出
```

- Prefill 采用经过验证的 GPU dense causal 实现；首版只支持完整 Prefill，不把各 chunk 当成互不相关的 prompt。
- 保存的是 RoPE 后的 K 与原 V，Q 也使用模型已完成 RoPE 的版本；CEMU 不再次应用位置编码。
- 首版可同步等待每层 Prefill 追加 accepted，以便隔离错误；这不等于要求每层完成 NAND drain。后续再恢复有界 Prefill 写线程和跨层重叠。
- Decode 先追加当前位置 KV，再执行包含该 token 的 Attention，输出按原 token 顺序写入 vLLM 提供的 `output` tensor；首版 forward 等待输出可用，不把 FlexGen 的 `PendingAttention` 对象返回给 vLLM。
- APPEND accepted 的含义继承同步/后台聚合模式。异步时是设备拥有并验证输入，尚不代表聚合或 NAND 写回完成。需要该 KV 的 Attention 仍由设备等待就绪，不能提前消费未准备的数据。
- 默认单输入槽和原背压继续生效；Python Future 数量不代表设备已拥有多输入槽。
- 运行结束单独执行 terminal DRAIN，检查 sealed/failed、accepted 前缀和在途写回；生成耗时与包含 drain 的耗时分开报告。DRAIN 会封闭会话，不能每个 token 调用，也不能等同于请求回收/Flash 擦除。

## 5. 必须处理的兼容问题

### 5.1 Attention scale：小范围扩展 Python 客户端

现有 `OptTorchDevice.opt_project()` 已将 Q 乘以 `head_dim**-0.5`，所以 `CemuSparfDevice._prepare_decode()` 打包 workflow 时固定 `scale=1.0`。

vLLM 的 Llama 在 backend 构造时传入 `scale=head_dim**-0.5`，输入 Q 本身没有这一步缩放。拟为设备封装增加显式 `attention_scale`，默认保持 1.0 兼容旧入口；vLLM 传入其 backend 收到的 scale，保持 Q 未预缩放。

现有 workflow ABI 已有 scale，设备近似分数与 CUDA exact QK 都使用它，通常不需要因此改 ABI 或重写 CUDA 算子。必须验证两条计算路径均只缩放一次。

### 5.2 GQA 与数据格式

- Llama 3.1 8B：32 Q heads、8 KV heads；每 4 个 Q heads 共享 1 个 KV head。
- 现有 `CemuSparfDevice` 已区分两种 head 数，并支持相应 mean 映射；底层存储仍按 8 个 KV heads 分配，不能把历史 KV 展开成 32 份后写入。
- 旧 `flexgen_cemu.py::open_backends()` 使用 OPT 的单一 `n_head` 建立多处几何，不能直接拿这个构造函数创建 Llama 会话。新增通用的 `num_query_heads/num_kv_heads/head_dim/layers/dtype/max_tokens` 配置。
- 当前 `SparfGroupLayout` 和 Attention 功能算子支持 FP16/FP32。模型默认 BF16；首版显式使用 FP16，并让原生 vLLM 参考运行也用 FP16。不能把 BF16 原始位模式当 FP16；若保留 BF16 主体并转换 KV，必须单独验证转换误差。
- GQA 的接口能力不等于 Llama 3.1 端到端正确性已经验收。

### 5.3 请求身份、变长输入与现有 session

`CommonAttentionMetadata` 提供 `query_start_loc`、`seq_lens`、`block_table_tensor`、`slot_mapping` 等。行号和物理 block ID 都不能作为永久请求身份。

现有上层按固定 microbatch/stream base、统一 `valid_tokens` 和单调追加组织 KV。第一组正确性测试限制单请求、单 session、一次 generate；接口从开始就保留请求身份、位置、长度和输出切片，单请求不作为最终架构限制。后续需要：

1. 从 runner 的真实请求状态建立 `(request_id, generation) → CEMU slot`，随后映射 layer/head/token。generation 用于同名请求重建，避免旧 Future 写到新请求。
2. 在真实 batch 重排之后按实际 token 顺序构造元数据；不要仅凭 `query_len==1` 判断 Decode，因为单 token prompt 也可能是 Prefill。
3. 在 `GPUModelRunner._update_states()` 相关请求生命周期处对接 finished/aborted/resumed 状态，并在构建实际 Attention metadata 时使用当前行映射。`CommonAttentionMetadata` 本身不提供完整稳定 request ID 生命周期。
4. 变长请求保持整批提交；已扩展 FDM batch 描述，传入每条序列的 KV slot、有效长度与紧凑偏移。不能把全批 max length 当作各请求的有效长度，也不再拆成 B=1 EXEC。
5. 请求结束后正确等待在途 reader/writer，处理取消与失败，补齐可复用 KV 空间的回收机制。关闭文件或 DRAIN 不能证明 NAND 容量已恢复。

当前 `fdm_kv_bind()` 对同一关联 namespace 已存在绑定会拒绝，因此不能为每个请求随意创建独立 FDM buffer。多请求应优先在一个共享绑定中分配稳定 slot，按需扩展现有 microbatch 对齐限制。持续服务和抢占恢复在生命周期机制验收前不开放。

### 5.4 vLLM 的 GPU KV 分配与启动探测

Attention backend 接入与 GPU KV 容量卸载是两个验收项。本地 `Attention.get_kv_cache_spec()` 仍返回完整 KV spec，runner 按它分配 cache；仅让 Decode 忽略 `kv_cache` 不会自动取消 GPU 历史 KV 分配。

首版允许保留原 cache 分配作为功能基线，但实际 Decode 只读 CEMU 历史 KV，不以 GPU cache 回退掩盖失败，也不据此宣称显存收益。随后单独实现并验证：调度逻辑容量由 CEMU 可用容量约束、GPU 只保留所需 Prefill/临时缓冲、cache spec 与 runner 分配一致。不能通过虚报 GPU block 数或伪造零字节缓存绕过一致性检查。

启动 profile/dummy forward 需要单独处理，不能向真实 session 追加虚拟 token。`attn_metadata=None` 是需要处理的场景之一，但不足以识别所有探测；按固定版本核对 profile 路径。无设备副作用的探测行为不计入正确性或性能结果，真实推理不允许静默回退。

## 6. 容量和性能口径

FP16、B=1 时，普通 K+V 的每 token 存储量为：

```text
2 × 32 layers × 8 KV heads × 128 dimensions × 2 bytes = 128 KiB/token
8192 tokens → 1 GiB K+V
```

CEMU 还保存 channel-K 副本，三布局有效载荷理论上为 192 KiB/token，8192 tokens 为 1.5 GiB；这不含组页/块填充、metadata、输入槽、设备缓冲池、写回在途空间及 Attention 暂存。真实 MR 和 NVM 预算必须使用当前布局/分配逻辑核算，不能直接照搬师兄 LMCache/GDS 的“1 GiB KV”设置。

首版从 P1024、小 Decode 次数开始。不要按模型最大上下文 131072 自动分配首个实验的缓冲区。当前 FDM 默认输入槽大小包含 `max_tokens`，也必须与实际实验容量一起检查。

SparF 的 top-r 和 compression ratio 可先保持已有实验参数作功能测试，但稀疏误差需要在新模型上验证。硬件时间模型实现可复用，模型几何和 softmax ratio 查表的覆盖/校准要匹配 Llama 的 workload，不能直接把 OPT-13B 配置或计时结果换名。

Guest GPU 是 vLLM 主体计算资源，Host 的 CSD GPU 是设备功能计算资源。保留现有冻结机制和模型完成时间；单独统计 Python 封装、CPU 中转、系统调用、设备排队、算子实际 Host 时间及目标阶段时间。vLLM 增加的框架开销不能自动归入目标 CSD 成本。

## 7. 代码组织建议

新增包拟放在 `CEMU/tests/cemu/instattention/python/vllm_adapter/`，作为独立可安装插件；保留 `vllm/` 为上游源码仓库，不复制客户端到该仓库。

| 拟新增模块 | 职责 |
|---|---|
| `plugin.py` | 注册 backend、版本/能力检查；无设备 I/O 副作用 |
| `backend.py` | backend/metadata builder/impl，Prefill/Decode 分派和 output 写入 |
| `session.py` | 公共设备配置、FDM store、SparF slots、均值、accepted 前缀与终止 drain |
| `request_state.py` | 稳定请求 ID/generation、行映射、生命周期事件与不支持模式检查 |
| 离线入口及启动配置 | 固定输入、原生/CEMU 两种运行、数值与 token 结果记录 |

复用现有 `cemu_flexgen` 中的设备层；名称中带 flexgen 不代表必须重写。只在发现明确框架耦合时提取公共实现，并由旧入口继续调用。不要依赖导入 FlexGen 的 `framework_attention.py` 来初始化 vLLM。

`_cemu_client` 以新 Guest Python 环境重新编译安装。当前 CPU buffer 传输可作为首版路径；将 CUDA `.so` 继续作为 CEMU 设备程序，与将其链接为 vLLM 的 PyTorch CUDA extension 是不同架构，首版保持前者。

## 8. 实施顺序与验收

1. **环境及原生参考**：完成模型快照检查；核对 Guest Python/GPU/驱动，固定 vLLM 提交；在 Guest 原生 vLLM 上跑 FP16、固定 prompt、greedy、短输出，保存配置和结果。
2. **公共接口适配**：显式 scale、独立 Q/KV head 几何、会话资源工厂；验证旧 OPT 默认行为不变。用不同 scale 和 32Q/8KV 的数值样本检查近似阶段、exact 阶段和均值补偿。
3. **最小 vLLM backend**：单 GPU/单 CSD、单请求、完整 Prefill、单 token Decode、eager；关闭 prefix cache、chunked prefill、推测解码、并行切分、异步调度及尚未验证的融合。初始化时检查实际配置并拒绝不支持模式，而非仅依赖命令行默认值。
4. **真实设备端到端**：使用原 FDM 聚合和 CUDA `.so`，检查设备 program target、实际 Execute、布局读写及 drain。分别跑全选择模式（top-r=head_dim、top-k=全部有效 token）和 SparF；全选择先核对局部 Attention/逐步 logits，SparF 再核对同输入参考与误差，不能要求稀疏模式无条件匹配 dense token。
5. **边界与失败**：FP16 token 页 16-token、channel 组 256-token 边界；已接收但未聚合、未满块 tail、背压、错误传播、无重复 KV 追加、probe 不污染会话。完整模型短跑前先用已有 CPU/qtest 测试覆盖可隔离的设备语义。
6. **扩展阶段**：移除不必要的 GPU 历史 KV 分配；共享 session 多请求、变长、回收与抢占；最后逐项引入分块 Prefill、连续批处理和优化并发。现有 FlexGen 双 slot 不自动等价于 vLLM 调度重叠。

每阶段分别记录源码版本、模型来源、精度、请求负载、CEMU 配置、算子版本和实际启用的聚合模式。结果分清功能通过、数值误差、设备行为覆盖、显存容量和性能，不能以一次文本生成代替所有验收。

## 9. 接入清单：具体在哪接、调用什么

本节将执行适配展开到当前代码位置。行号以第 1 节的 vLLM 提交为准，函数名是后续定位依据。以下均为拟实现接口，不表示已经接通。

### 9.1 调用边界与改动位置

vLLM 决定本轮哪些请求执行多少 token；适配层将调度结果转换成 CEMU 写入和 EXEC；设备自行处理 KV 定位、SparF 选择、算子执行与资源排队。保留原 Scheduler 策略。

| 接入项 | 现有 vLLM 位置（相对 vllm 仓库根目录） | 拟实现方式 | 复用 CEMU 接口 |
|---|---|---|---|
| 注册并选择 CEMU backend | `vllm/v1/attention/backends/registry.py:355` `register_backend()`；`vllm/model_executor/layers/attention/attention.py:356` `get_attn_backend()` | 新插件注册 backend，并显式选择；这些是现有扩展入口，不需要硬改通用 Attention 调用 | worker 内创建会话，加载/激活聚合程序和 Attention 程序，创建两类 MRS |
| 请求创建、完成、取消、恢复 | `vllm/v1/worker/gpu_model_runner.py:1176` `_update_states()` 及 worker 初始化/关闭路径 | 增加受 CEMU 配置控制的生命周期对接，维护稳定请求身份；不在 scheduler 中实现设备算法 | 共享 FDM 绑定、读写在途状态；回收能力单独扩展 |
| 本轮实际请求顺序与元数据 | 同文件 `:2253` `_build_attention_metadata()`；`vllm/v1/attention/backend.py` `AttentionMetadataBuilder.build()` | 在实际重排后映射行到请求，再构造 CEMU metadata；仅靠通用 metadata 不足以获得稳定 request ID | layer、KV slot、追加起点、追加数量、有效前缀、Q/output 切片 |
| Prefill KV 追加 | `attention.py:799` `self.impl.forward(...)` 进入新 `CemuAttentionImpl.forward()` 的 Prefill 分支 | 取当前请求的 RoPE 后 K 和原 V，按当前层/位置追加 | `FDM_KVbuffer.append_batch()` |
| Decode KV 追加 | 同一 `CemuAttentionImpl.forward()` 的 Decode 分支 | 追加本步 K/V，成功后发布 accepted 前缀；只追加一次 | 同一个 `append_batch()` |
| V 均值维护 | 新 backend/session 的成功追加路径 | 按请求身份和 layer 维护 V sum/count，包含刚追加的数据，不跨请求混合 | 提取 `SparfGroupedAttentionBackend._update_mean()` 的数值逻辑 |
| 写 Q、写 mean、准备 EXEC | 新 backend 的 Decode 分支，追加成功后 | 取得独立 Attention slot；snapshot 本次输入和元数据 | `CemuSparfDevice.prepare_decode()`，写 Attention MR0/MR4 |
| 提交 EXEC | 同一 Decode 分支 | 使用准备好的 MRS/程序和 workflow 元数据，等待真实完成 | `run_decode()` 内的一次 execute |
| 读结果与回填 | EXEC 完成后、backend forward 返回前 | 从 MR9 读取，转换到 Guest GPU，将结果放回本轮对应 output 切片 | `collect_decode()`，以及 `output[...] .copy_(...)` |
| GPU KV 分配 | `attention.py:621` `get_kv_cache_spec()` 及 runner 初始化缓存路径 | 首先核对分配协议；后续分离逻辑 KV 容量和 GPU 临时空间 | CEMU 布局预算、设备容量检查；属于单独验收项 |

`llama.py:232` 的 `self.attn(q,k,v)`、`:233` 的 `o_proj` 保持原调用。`attention.py:799` 是实际控制转入点，但实现时通过 backend 注册使 `self.impl` 指向 CEMU 类，不直接在共享函数中无条件替换所有模型的调用。

在 `attention.py:558/:578` 的 KV 更新判断处，CEMU backend 声明 `forward_includes_kv_cache_update=True`，使追加只发生在其 forward 内；不再另写一遍 `do_kv_cache_update()`。真实 compute 才能产生设备副作用，profile/dummy 调用必须识别。

### 9.2 每次计算的适配层描述

拟让每个工作项携带以下逻辑信息，均从真实请求状态和本轮元数据导出：

```text
request_id + request_epoch       # 适配层的请求身份，防止同名请求重建混用
device_session + generation     # 现有设备绑定身份，不等于逐请求 epoch
kv_slot                         # 设备长期保存 KV 的位置
layer_id
append_start, append_count      # 本轮新增 KV 的位置和数量
valid_tokens                    # 本次 Attention 可访问的有效前缀长度
input_slice, output_slice       # 当前 vLLM 扁平 token 数组的位置
phase                           # 由请求进度确定，不能只看 token 数
num_query_heads, num_kv_heads, head_dim, dtype, scale
```

此外，为 EXEC 分配一个短期 `attention_slot`，其中有 Q/mean/output MR。长期 `kv_slot` 和短期 `attention_slot` 必须分开命名和管理：请求的 KV 位置可以不变，而相邻两次 Attention 使用不同执行 slot。

这里传递的是 KV 逻辑身份与有效范围。vLLM GPU 的 block ID 不是 CEMU 的 NAND 地址，不能原样当设备地址传入；设备用自己的 KV 布局/FTL 定位数据。top-r channel、top-k token 集合仍由设备根据 Q 计算。

### 9.3 整批 EXEC 与变长序列描述

新版 FDM workflow 使用 `GROUPED | FDM | RAGGED` 标志，autonomous version 仍为 3。
header 后是原 session/generation binding，再接 `batch_size` 个 20 字节描述：

```text
uint32 slot, valid_tokens, top_k, history_offset, selected_offset
```

偏移是之前序列长度/top-k 的前缀和，尚未乘 Q heads；设备检查这些前缀、各序列边界、
slot 唯一性与 accepted 前缀，按 slot 获取/释放读租约。旧等长 FDM/Native 接口继续接受。

例如本轮顺序 `[C,A,B]`，KV slot 分别为 `[7,2,9]`，长度为 `[513,1025,2049]`：
适配层一次写入 Q/mean，提交一条 B=3 EXEC，描述表保持 `[C,A,B]` 顺序。
设备按实际长度读取 channel K、选择各自 top-k，再紧凑读 K/V。
exact QK 和 PV 各调用一次整批算子，CUDA 分别以 batch × Q heads 分配 blocks。
输出按相同顺序回到 vLLM，不搬动历史 KV，不把短序列补成最长序列。

Attention 算子 ABI v2 的 MR7 是 alpha 数组后接同一批描述；metadata reserved 保存全批
selected token 数（已乘 Q heads）。PV 的输入增加 MR7；v1 的三 MR PV 保持兼容。
需要一起重编译 QEMU 和 Attention `.so`，无需新增 NVMe opcode 或重写 pybind。

vLLM Session 只使用一个常驻执行 MRS，同步等待该层整批完成，去掉逐请求线程队列。
APPEND 仍按请求/层写入；完整 Prefill 仍在 Guest GPU 计算。KV slot 尚不回收，旧等长
runtime model 尚未扩展到 ragged batch，因此当前拒绝非空 runtime_config，不作性能结论。

### 9.4 Prefill 与 Decode 的具体顺序

```text
完整 Prefill（每个请求、每层）：
  获取本层 K/V 与逻辑位置
  → append_batch(start=0, count=prompt_tokens)
  → accepted 后更新 V sum/count
  → GPU dense causal Attention 填入 output
  → 返回 vLLM 的 Wo/MLP

Decode（本轮整批、每层）：
  为各请求 append_batch(start=previous_prefix, count=1)
  → accepted 后更新 V sum/count
  → 复用 batch Attention slot
  → prepare_decode：整批 Q 写 MR0，mean 写 MR4，构造逐序列 metadata
  → 一条 EXEC：设备定位 KV、SparF、整批 CUDA .so
  → collect_decode：MR9 读输出（MR10 是原诊断 trace）
  → 回填 output 对应切片，释放 Attention slot
  → 返回 vLLM 的 Wo/MLP
```

顺序描述的是首版容易验证的同步基线。后续可将 Prefill GPU 计算与有界 KV 写入重叠，或提交多个独立 Decode 后按依赖消费；需保证生产者 GPU 数据已完成、输入 buffer 生命周期有效、Q MR 在完成前不被覆盖，且失败能够传回 vLLM。

分块 Prefill 的**写入**可继续使用 `append_batch(start=chunk_start, count=chunk_tokens)`，上层改为增量均值和前缀检查。但后续 chunk 的 GPU Attention 还需要之前 chunk 的历史 KV；不能只算当前 chunk 的 Q/K/V。必须明确保留 GPU Prefill cache、回读历史 KV，或实现新的设备 Prefill 能力，并在请求完成 Prefill 时处理缓存生命周期。复用 APPEND 不代表分块 Prefill 计算已自动支持。

### 9.5 接入完成的证据

除了生成文本，测试至少要确认：

1. Prefill 和 Decode 的 KV 都经过真实 FDM APPEND/聚合路径，计数无遗漏、无重复。
2. 设备 trace 中存在正确的 session、layer、有效长度及 CUDA program target；历史 KV 读取来自原设备缓冲/存储路径。
3. 使用内容明显不同的多请求，交错长度并改变 batch 顺序，Attention 输出仍与各自参考对应；不能只用相同 prompt 测映射。
4. 相同请求 ID 重建、取消、迟到完成、设备失败不会串写新请求；未实现的回收/恢复能力明确拒绝。
5. 保留旧 OPT 的 scale=1 行为，新 Llama 显式 scale；32Q/8KV、FP16 分页边界及末尾 drain 均验证。

当前实现和验证范围以本文开头及适配器 README 为准；尚未完成真实模型端到端和性能验收。
