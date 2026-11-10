# vLLM → CEMU / InstAttention

实现基线：vLLM `eecf8dbb2b279372733c844a9735262494b3fa9b`。
Guest 安装脚本单独固定 cu129 提交 `77154ca6116ae34cbfe314beef1016bde9b63989`：
已核对该提交的 OPT、Worker hook、backend 注册和 metadata 接口，且官方 cu129 wheel 存在。
原实现基线的 cu129 metadata 返回 404，cu130 wheel 存在；不把新源码与旧二进制混用。
用户提供的 Guest 日志显示驱动为 550.163.01，与 Host 的 575 不同；cu129 环境正在安装，
仍需脚本末尾的真实 CUDA 运算和模型测试验证，接口检查不等于端到端验收。
这是首版实验适配器。设备链路已通过 CPU QEMU qtest，OPT MHA/Llama GQA 的整批设备 CUDA 测试也已通过；真实 vLLM + Llama/OPT + Guest/Host CUDA
端到端尚未运行，不应将 qtest 结果写成完整模型或性能验证结果。

## 调用路径

```text
vLLM LlamaAttention: qkv_proj → RoPE → Attention.forward
vLLM OPTAttention:   qkv_proj → Attention.forward
  → CemuAttentionImpl.forward
    Prefill: Guest GPU Flash SDPA + 新 K/V APPEND
    Decode: 新 K/V APPEND → mean 更新 → Q/mean MR → autonomous EXEC → output MR
  → vLLM o_proj / MLP / 采样
```

- `worker.py`：通过 `Worker._make_model_runner()` 创建 V1 runner 子类；沿用原 scheduler，
  在 batch 重排后读取请求 ID、computed tokens 和 prompt 长度。
- `backend.py`：backend 注册、metadata builder、Attention 实现。包含 KV 更新，
  不再执行 vLLM 的额外 GPU cache 写入。profiling/dummy forward 不追加 KV。
- `session.py`：稳定请求 ID → KV slot；共享 FDM 绑定、一个常驻 batch Attention MRS、逐层 V sum；
  使用原 `_cemu_client` 和 `FDM_KVbuffer`，无逐请求执行队列。
- `cemu_sparf_device.py`：显式传入 Attention scale，默认 1.0 兼容旧 OPT 入口；
  可关闭逐步 trace 回读。新增 ragged batch 描述，CPU/CUDA 算子按逐序列 top-k 和紧凑偏移
  执行整批 exact QK/PV；保留 SparF 算法与旧等长 batch ABI。

Prefill 和 Decode 都按请求、层调用一次 `append_batch`，保存传入 Attention 的 K/V（Llama 的 K 已应用 RoPE；OPT 使用上层位置 embedding）。
设备继续调用 `fdm_kv_aggregate.so` 聚合、维护三布局缓冲、聚块下放和处理 DRAM/NAND 读取。
APPEND 的 `stream_base = kv_slot × layers × kv_heads`；Decode EXEC 携带整批
`(kv_slot, valid_tokens, top_k, history_offset, selected_offset)`，Q/output 行顺序沿用 vLLM。

## 已做的开销控制

1. KV 不在 Guest 维护全历史副本，也不按 Q heads 扩展后追加；均值使用增量 sum/count。
2. 全 worker 共享可复用 pinned K/V/Q/output 缓冲，K/V 一次按层整批搬运；仅 Decode 传 Q。
   D2H 使用同一 CUDA event 等待，不对每个请求调用全设备 synchronize。
3. 每层本轮全部 Decode 一条 EXEC，整批写 Q/mean、整批读结果；单请求使用同一路径。
   channel K、selected K/V 和概率按实际长度紧凑排列，没有最长序列 padding。
   exact QK 和 PV 各一次整批算子调用；CUDA 各一次 kernel launch。
   一个常驻 MRS 跨 token/layer 复用，移除了逐请求调度队列和空闲的第二执行 slot。
   混合 Prefill/Decode 中只把 Decode 子集交给设备，按原行号回填。
4. APPEND 成功后才更新均值；先完成本层新增 KV 的 APPEND，再发本层 EXEC，减少读租约与
   写入交替。后台聚合是否启用仍由 Host 的 `CEMU_FDM_ASYNC_AGGREGATE` 决定。
5. 默认 `read_trace=false`，每次只读 output。mean 的 GQA head 索引在初始化时生成。
6. 沿用 vLLM cache spec，因此仍有兼容性 GPU KV 分配。离线入口按请求数和上下文上限设置
   `kv_cache_memory_bytes`，避免默认占满剩余显存。这部分尚未消除，不能宣称完整 KV 显存卸载。
7. output pinned buffer 按 4096 字节对齐，通过新增 `read_range_into` 直接接收 C++ pread，
   消除原路径的对齐缓冲 → vector → pybind 数组 → NumPy 副本 → staging 复制。
   只有地址对齐、长度为 512 字节倍数时走直接读取；其他调用安全回退到对齐临时缓冲，
   不把 padding 写到调用者数组之外。旧 `read_range`/`read_tensor` API 保留。
8. MHA 的 mean 不再做恒等 head 重排；GQA 仍按设备 ABI 扩展到 Q heads。
   Decode 的 FP64 V sum 原地累加，除法直接输出到已有 FP16 mean 缓冲，保持原精度。
9. Guest C++/pybind 使用 `CMAKE_BUILD_TYPE=Release`。更新到读入现有缓冲区的接口后必须
   重建 `_cemu_client`；安装脚本末尾会检查是否存在 `read_range_into`。

当前 CPU buffer pybind 路径仍有 D2H/H2D、APPEND 序列化、MR 写读等必要复制。
这里没有实现 GPUDirect，也没有承诺总性能不下降；这些边界需在真实 Guest 上测量。

## 首版范围

FP16、Llama/OPT、head_dim=128、单 GPU/单 CSD、完整 Prefill、逐 token Decode、eager。
可在固定容量会话中处理多请求、变长和 batch 换行；`request_capacity` 是整个会话的累计请求
容量，不仅是并发数。完成/取消后不会重用设备 KV slot；重新创建 worker 仅能新建软件会话，
不会回收 NVM 中旧任务的页映射或物理块。容量消耗会跨软件会话累积。
这不是可无限接收请求的在线服务版本。

不支持 prefix caching、chunked prefill、抢占/重算、异步 scheduler、speculative decoding、
TP/PP/DP/CP、LoRA、量化、sleep 或其他模型架构；配置/调度不符合约束会报错。
单 token prompt 根据 prompt/computed 长度判断为 Prefill，不会误走 Decode。

## Guest 安装

先准备上述提交的原生 vLLM 环境并验证 Guest GPU；使用该仓库的 uv / `.venv` 工作流。
该版本要求 Python >=3.11，下载模型的 `hf-env` Python 3.10 不能运行此版本。
以下路径按 Guest 的实际挂载调整：

```bash
VLLM=/home/lihan/cemu/vllm
ADAPTER=/home/lihan/cemu/CEMU/tests/cemu/instattention
uv pip install --python "$VLLM/.venv/bin/python" pybind11 -e "$ADAPTER"
cmake -S "$ADAPTER" -B "$ADAPTER/build-vllm-guest" \
  -DPython3_EXECUTABLE="$VLLM/.venv/bin/python" -DCMAKE_BUILD_TYPE=Release
cmake --build "$ADAPTER/build-vllm-guest" --target _cemu_client -j
export PYTHONPATH="$ADAPTER/build-vllm-guest${PYTHONPATH:+:$PYTHONPATH}"
```

`pyproject.toml` 注册 `vllm.general_plugins:cemu`，worker 子进程也会加载它。
如设置了 `VLLM_PLUGINS` 白名单，需要包含 `cemu`。注册只登记类路径，不打开设备。
配置文件在 `experiments/configs/vllm_llama_cemu.json`：

- `fdm_directory` 必须是 Guest 真实 FDM 挂载下尚不存在的独立会话目录；不要用普通磁盘冒充，
  也不要直接指定挂载点或复用其他会话目录。正常 drain、释放资源后只删除本会话创建的文件及目录；
  初始化、执行、drain 或释放失败时保留现场，不自动删除旧会话残留。
- FDM 文件清理不等于 NVM 状态重置。当前没有会话级 KV 映射/物理块回收；性能对照的每轮
  都要正常关闭 Guest、等待 Host QEMU 退出，再用同配置重新启动 QEMU，获得新 NVM 会话。
  不要在同一 QEMU 实例内直接连续跑 vLLM 和 FlexGen，也不要只重启 Python 或 Guest 系统。
  两个 B8 Guest 测试脚本会在加载权重前通过只读 INFO 检查页/块计数，非零即拒绝测试；
  检查不执行清除、格式化或映射重置。
- `program_path` 与 `aggregate_program` 是 Host QEMU 能访问的两个算子路径，
  分别为 CUDA devptr Attention `.so` 和 HOST 聚合 `.so`，不是 Guest pybind 库路径。
- namespace、MR 容量和 Host 配置须匹配。默认 P+D≤2048、一个请求，用于小规模接通。
- `attention_slots=1`，一个 slot 容纳 `request_capacity` 条 Decode；扩大 batch 要核算 MR 容量。
- 功能配置可以保持 `runtime_config=null`。性能配置同时提供 `runtime_config` 与
  `softmax_ratio_file`，复用此前的 GEMV/filter + Softmax ratio 时间模型，不使用冻结中的
  CUDA 实际耗时。按实际 batch/head/token/top-k 查表；未测量的形状报错，不自动插值或填零。
  此前 ratio 表描述等长 batch，不能据此宣称变长批次的计算时间已校准。
- 此版本需要一起重编译 QEMU 和 `sparf_attention_devptr.so`。旧算子只支持 Attention ABI v1，
  会拒绝新版 batch metadata；聚合算子和 pybind 接口继续复用。

## 离线入口和模型验收

完成本地模型快照后，在有 Guest GPU、FDM 挂载及运行中 CEMU 设备的 Guest 执行：

```bash
"$VLLM/.venv/bin/python" -m vllm_adapter.run \
  --config "$ADAPTER/experiments/configs/vllm_llama_cemu.json" \
  --model /home/lihan/cemu/models/Meta-Llama-3.1-8B \
  --prompt 'The meaning of life is' --max-tokens 8 \
  --output /tmp/llama-cemu.json
```

同一命令加 `--native`，另存输出文件，运行原生 vLLM dense 基线。`--prompt` 可重复；
也支持 `--prompt-token-ids ids.json`，文件内容为 token-ID 数组的数组。
先将配置副本设为 `top_r=128, compression_ratio=1` 做全选择数值/逐步 logits 验证，
再测默认稀疏参数。稀疏结果不要求与 dense 逐 token 相同。

入口强制本地离线模型，检查 incomplete 和索引所列权重分片。结束显式调用 worker 的
`cemu_drain`，先完成同步 EXEC 再 DRAIN，检查 sealed/failed，然后释放资源。JSON 分别记录
生成时间、生成加 drain 时间、兼容 GPU KV 容量、drain 状态、token IDs 和 top-5 logprobs。
首轮时间包括惰性设备资源初始化；性能实验应分离启动成本并采集逐层稳态时间，不能直接
用这条 smoke 命令宣称吞吐收益。

## 用已有 OPT-13B 验证完整 vLLM 接入

完整启动入口为 `femu-scripts/run-vllm-opt-test.sh`：使用迁移后的 normal-boot Guest 镜像，
Guest GPU 默认 PCI `0000:27:00.0`（已绑定 vfio-pci），Host 设备计算 GPU 默认 index 1。
它设置 64 GiB Guest RAM、8 vCPU、1 GiB FDM、4 GiB 新 NVM、262144 页 native I/O 上限，
并将 vLLM 源码只读共享为 `vllmsrc`。这些是功能测试配置，不是经过校准的性能配置。
默认镜像为 `/home/lihan/cemu/cemu-vllm-runtime/guest.qcow2`，运行目录为同级 `runs/`。
旧 `/data` 分区已满，旧 Guest 根分区只有 20 GiB，不能继续用于依赖安装。
先在 Guest 正常 `poweroff`，等待 QEMU 退出，再在 Host 执行：

```bash
cd /home/lihan/cemu/CEMU
sudo bash femu-scripts/migrate-vllm-guest.sh
```

迁移脚本从旧镜像转换成不依赖 backing file 的独立副本，对比逻辑磁盘内容并检查 qcow2，
然后扩虚拟磁盘到 80 GiB；原文件保留，目标存在或源文件仍被使用时拒绝执行。
首次启动新 Guest 后，还须在 Guest 扩分区和 ext4（虚拟磁盘扩容不会自动扩文件系统）：

```bash
command -v growpart >/dev/null || (apt-get update && apt-get install -y cloud-guest-utils)
growpart /dev/vda 1 && resize2fs /dev/vda1
df -hT /
```

Host：

```bash
cd /home/lihan/cemu/CEMU
ninja -C build -j8 qemu-system-x86_64
PATH=/usr/local/cuda-12.4/bin:$PATH make -C tests/cemu/kernel \
  ../build/sparf_attention_devptr.so ../build/fdm_kv_aggregate.so
sudo bash femu-scripts/run-vllm-opt-test.sh
```

迁移源默认为 `/data/lihan/cemu-native-opt/standalone.PnutcI/guest.qcow2`，该路径来自已有
FlexGen 启动脚本；需用户以 sudo 执行离线迁移，不能把临时镜像测试当成实际迁移完成。
需要更换时用 `sudo env CEMU_NATIVE_OS_IMAGE=/实际路径/guest.qcow2 bash ...`。
保持启动终端运行，另一个 Host 终端执行 `ssh -p 2222 root@127.0.0.1`。

Guest 首次安装（需要网络下载 uv/Python/vLLM/PyTorch）：

```bash
mkdir -p /root/CEMU
mountpoint -q /root/CEMU || mount -t 9p -o ro,trans=virtio,version=9p2000.L cemusrc /root/CEMU
apt-get update
apt-get install -y curl ca-certificates git cmake build-essential libnuma1
bash /root/CEMU/femu-scripts/setup-vllm-opt-guest.sh
```

该脚本创建 `/root/vllm-cu129/.venv`（Python 3.12），源码和预编译扩展均固定上述 cu129
提交，保留原 `/root/cemu-gpu-env`。插件从可写副本安装，pybind 在 `/root/cemu-vllm-build`
编译；Host 源码共享保持只读。它挂载 FDM 和模型，不格式化 native NVM。
安装最后检查真实 torch CUDA 运算、vLLM import、backend 注册与 pybind import。
任何一步失败都应先解决，不能绕过检查把后续命令算成已测通。

之后每个 Guest shell：

```bash
source /root/cemu-vllm-env.sh
cd /root
```


可复用 Host `/home/lihan/cemu/models/OPT-13B` 的 Hugging Face checkpoint；已有 Guest
`modelsrc` 挂载对应 `/root/models/OPT-13B`。不要传 `models/flexgen-weights/opt-13b-np`：
那是 FlexGen 的 NumPy 权重目录，当前 vLLM 入口使用原 HF `.bin` 分片。

本地 OPT-13B 为 40 层、40 Q heads、40 KV heads、head_dim=128。vLLM OPTAttention
把未缩放 Q 交给 Attention，scale=1/sqrt(128)，与旧 FlexGen 入口的 scale=1 默认值不同。
OPT-125M 的 head_dim=64，当前 backend 会明确拒绝，不能只改模型名称运行。

已准备 `experiments/configs/vllm_opt13b_cemu.json`：两请求、上下文上限 128、完整 Prefill、
4-token 短生成；top_r=128、compression_ratio=1 先使用全选择数值对照。
配置预留 64 MiB 管理区、512 MiB KV 数据区及额外 Attention MRS，须与 Host FDM 容量匹配。
此配置的 FDM 路径采用已有 Guest 挂载 `/mnt/fdm0`；算子路径仍是 Host 路径。

在已安装该提交 vLLM、已安装插件/pybind、已挂载 FDM 的 Guest 中运行（VLLM 指向 Guest
实际 vLLM 环境，ADAPTER 指向 Guest 的 InstAttention 目录）：

```bash
"$VLLM/.venv/bin/python" -m vllm_adapter.run \
  --config "$ADAPTER/experiments/configs/vllm_opt13b_cemu.json" \
  --model /root/models/OPT-13B \
  --prompt 'The capital of France is' \
  --prompt 'A computer is a machine that' \
  --max-tokens 4 --output /root/cemu-logs/vllm-opt13b-cemu.json
```

同参数加 `--native`，输出另存 `vllm-opt13b-native.json`，获得原生 vLLM dense 对照。
测试入口默认设置 `VLLM_USE_FLASHINFER_SAMPLER=0`，两组均使用 vLLM 内置采样路径。
固定版本会在启动 profile 时预热随机采样，即使实际请求是 greedy；Guest 没有 nvcc，
默认 FlashInfer 采样器可能在这里触发 JIT 编译并失败。此设置仅控制采样器，
不禁用 FlashAttention 或 CEMU Attention；外部显式设置该变量时仍保留用户选择。
两次都从相同 prompt 生成。完整接入证据必须包含实际 vLLM generate 返回、设备 APPEND/EXEC
记录、正确 batch/layer/长度，以及终止 drain 的 sealed=1、failed=0，不能只凭生成文本判断。

对于这组两请求同时 Prefill、各生成 4 tokens 的负载，Prefill 产生首个输出 token，之后有
3 轮 Decode；预期 40×3=120 条 Attention EXEC，每条 B=2，而不是逐请求拆成 240 条。
结果 JSON 的 `drain[0].attention_exec_count` 统计已完成的整批 Attention EXEC，应为 120。
Host 启用 `CEMU_SPARF_AUTONOMOUS_TIMELINE` 可检查真实执行记录（已有启动脚本支持）。
之后再用 top_r=16、compression_ratio=8 验证 SparF 路径；稀疏结果不要求与 dense 相同。

### vLLM 阶段计时

入口默认 `--profile basic`，原生和 CEMU 使用相同的同步 Worker 计时边界。
在上述命令加 `--profile detailed` 可同时记录现有 C++/pybind 分段及 CUDA events；
`--profile off` 关闭适配层计时，用于比较测量开销。不会修改 vLLM 调度器。
详细模式不逐层增加 GPU synchronize，也不增加 APPEND/EXEC 或 MR10 trace 回读；
CUDA events 在执行中查询完成状态，结束报告时收集。计时本身仍有开销，不能当作零成本。

**Freeze 口径：**入口在引擎初始化前将此测试 Guest 的 clocksource 设置为 `hpet`，
初始化后和生成/drain 后再次检查，结果保存在 `guest_clock`。HPET 不可用、写入失败
或运行中变为其他时钟时停止，不能悄悄回退为 `kvm-clock`。只接受带
`cemu_native_opt_smoke=1` 的 Guest，绝不切换宿主机时钟。
这复用原 OPT 测试及既有 freeze 探针确认的 HPET 口径：Python perf_counter 和
C++ steady_clock 读取 Guest 时间，功能冻结期间不推进；**不要再减一次 Host 功能执行时间**。
HPET 读时钟开销较高，A/B 应保持相同计时模式。

结果打印在终端，并保存在同一个 `--output` JSON 中：

| 字段或 metric | 测量范围 |
| --- | --- |
| `initialize_seconds` | LLM 引擎创建、权重加载和启动预热 |
| `generate_seconds` | 整个 `LLM.generate()`，包括请求处理和引擎调度 |
| `timing.phases.prefill/decode/mixed` | 每个同步 batch 从 runner execute 开始，到 sample_tokens 返回 CPU 结果；总时长、步数、平均每步时长 |
| `*.execute_host_wall`, `*.sample_host_wall` | Guest CPU 调用的 HPET 耗时（字段 host 指调用端，不是 QEMU 宿主机）；GPU 可能在 sampling 等待时才完成 |
| `*.attention_host_wall` | 所有层适配器 Attention 调用的累计 Guest HPET 耗时 |
| `*.append_wall`, `*.value_sum_wall` | KV 下放，以及 CPU V 累计值更新 |
| `decode.mean_and_rows_wall`, `decode.metadata_wall` | mean、逐序列长度/slot 与命令元数据准备 |
| `*.d2h_enqueue_wall`, `*.d2h_wait_wall`, `*.h2d_enqueue_wall` | Guest GPU/CPU 数据搬运提交及原有完成等待 |
| `decode.fdm_write_wall`, `decode.execute_wait_wall`, `decode.output_read_wall` | 写 Q/mean、同步 EXEC、结果回读 |
| 详细模式 `*.binding_prepare_wall` 等 | 复用 C++ 计时，区分绑定/缓冲准备、IO 等待、收尾；是调用 wall time，不是 CPU 利用率 |
| `timing.cuda_diagnostics` 中的 `*.cuda_stream` | 独立 GPU 时钟，不随 Guest freeze 停止，可能包含冻结区间；只作诊断，排除在主 metrics 和占比之外 |
| `drain_rpc_seconds`, `terminal_drain.wall` | 最后设备 drain，独立于生成耗时 |

首轮 Prefill 包含懒分配的 staging、FDM 注册和算子初始化，分别有
`staging.allocate_wall`、`session.open_wall`，不要将短 smoke 的平均 Decode 当作稳定吞吐。
混合 Prefill/Decode batch 单独计入 mixed；一个 token 的 prompt 仍属于 Prefill。
步数按 batch 统计，并非请求数；本测试正常应为一个 Prefill step、三个 Decode steps。
这些指标嵌套/重叠，不能直接求和。worker step 不包含其外部全部调度、分词、IPC 开销，
也不等于用户视角 TTFT；未单独测得的剩余耗时不能一概叫作软件开销。

设备明细自动从 `--device-timeline /root/cemu-logs/device-timeline.jsonl` 读取。
启动脚本已开启此 Host 时间线；入口仅读取 generate 开始后新增的记录，并核对
program ID、begin/complete 与已完成 EXEC 数。缺失、截断、失败或数量不符时返回
`device_timing.available=false`，不生成误导性的零耗时。

`device_timing` 的主数据全部使用 **QEMU virtual 时钟，排除功能 freeze**。
设备占比的分母为同一 virtual 时钟上的首条 Decode EXEC admission 到最后一条 ready，
不是完整 generate；分子为设备命令区间并集。approx/qk/pv 使用模型调度的 CU spans，
channel_k/token_k/token_v 使用 virtual 读取区间，并单独报告读算重叠。
冻结期间真实 Host GPU 的功能执行耗时不计入模拟设备 compute；零 compute budget
会明确提示，不能拿冻结中的真实算子时间替代尚未标定的模型预算。
`unattributed_virtual_seconds` 为模型读算区间之外的剩余 virtual 时间，包含排队/分发，
不是纯 CPU 软件成本。Guest 的 EXEC 等待占 worker steps 比例单独使用 Guest HPET。
Host 功能耗时仅保存在 `device_timing.host_diagnostics`，不参与主耗时与占比。
功能调用边界也不是完整的全局 freeze 边界，不能用它们从 Guest 时间中做手工减法。

现阶段环境状态：Guest 独立的 `/root/vllm-cu129/.venv` 已安装完成，
CEMU 插件/pybind 导入与基础 CUDA 检查通过，禁用 FlashInfer 采样器后原生 OPT-13B
已完成两个请求、各四个输出 token 的生成并正常退出。该次旧入口没有记录 HPET/阶段计时，
只证明原生功能路径，正式计时对比应使用新入口重跑两组。CEMU 模型端到端尚待验证。
下载 Llama 权重不是这项验收的前置条件。

## 恢复此前的性能实验与精细计时

小规模 smoke 配置保留用于功能回归。性能实验使用此前的 OPT-13B FP16、B8、P1024、
16 轮 Decode（共 17 个输出 token）、top-r=16、compression=8、balanced 放置、后台异步聚合。
设备复用 `fdm_opt13b_b4_p1024_g1024_lun32.json`：8 channel × 4 LUN、4 KiB page、
11703 ns/page、64 GiB NVM、16 GiB FDM；KV data 8 GiB、meta 128 MiB；Guest 默认 4 vCPU、64 GiB RAM。
11.2 GB/s 仅为理想带宽等效值，不是物理 NAND 时序校准。NVM 继续使用 file-backed SSD 文件。

**调度区别必须保留：**当前 vLLM 是一个 B8 调度批次、一个整批 Attention slot，
不是 FlexGen 的两组 B4 microbatch 双 slot 重叠。没有为了实验修改 vLLM 调度器。
每层每轮仍有 8 次逐请求 APPEND、一次整批 EXEC。若要对照旧 B8/MB4，需明确注明这一差别。

先正常关闭旧 Guest，在 Host 执行：

```bash
cd /home/lihan/cemu/CEMU
make -C build -j4
make -C tests/cemu/kernel ../build/fdm_kv_aggregate.so ../build/sparf_attention_devptr.so
sudo env CEMU_VLLM_TEST_PROFILE=b8-p1024-d16 \
  bash femu-scripts/run-vllm-opt-test.sh
```

此入口使用独立的 `cemu-vllm-runtime/guest.qcow2`，首次环境安装沿用前面的
`setup-vllm-opt-guest.sh`，不要直接在 Host 执行 Guest 脚本。
已安装环境的 Guest 执行：

```bash
bash /root/CEMU/femu-scripts/vllm-opt-guest-test.sh
```

脚本先验证内核 tracepoint、HPET、设备参数及空 FDM，构建 pybind，再测量 B8 所需的
Softmax ratio；可用第一个参数传入已有的匹配 ratio JSON。保存独立的实验配置和输入，
设置 `max_num_batched_tokens=8192`，避免 8×1024 的 Prefill 被调度成多个小批次。
但 token budget 只是容量上限，不保证请求同时进入后台 EngineCore：普通 `generate()`
逐条入队时，较早的请求可能已开始 Decode。本性能配置因此设置 `fixed_batch_size=8`，
通过固定版本 EngineCore 的 `pause_scheduler("keep", False)` 暂停调度、`LLM.enqueue()`
入队全部请求，再 `resume_scheduler()` 和 `wait_for_completion()`；同一客户端 FIFO 保证
恢复发生在这些 ADD 之后。不关闭 EngineCore 多进程、不修改 scheduler 策略、不清除 KV，
也不为未测量 B2/变长形状填零或复用 B8 ratio。Worker 在设备 I/O 之前校验每步实际 B8
及相同 token 范围，异常即停止。本次是同时到达的离线固定批次，不代表在线连续到达。
`request_admission=barrier` 写入结果；入队屏障计入 generate 时间，不计入 Worker Decode 窗口。
功能 smoke 配置不设置 `fixed_batch_size`，仍使用普通 `generate()`。
`max_tokens=1041` 是序列容量，生成 17 tokens；实际追加到 KV 的最后长度仍为 1040。
校验一个 Prefill step、16 个 Decode steps、640 次 EXEC、5440 次 APPEND、完整 drain。
这些是流程/计数检查，不替代完整模型 logits 的正确性对照。

若失败日志出现 `unmeasured Softmax shape vectors=80, tokens=1025`，表示实际 Decode
只有 B2（2×40 heads），并非权重或 FDM 容量失败。更新测试入口与性能配置后使用上述
入队屏障重跑，而不是只补测 B2 来掩盖批次变化。旧失败会话可能留下 FDM 绑定或文件，
建议正常关闭 Guest 并用同配置重启获得新设备会话；不要在设备仍引用 MR 时强删 FDM 文件。
此次修复只涉及 Guest Python/实验配置，无须重编译 QEMU 或算子。

结果目录为 `/root/cemu-logs/vllm-b8-p1024-d16.*`：

- `result.json`：生成、初始化、drain、阶段和设备时间线汇总，以及实际输出 token。
- `trace/phase-intervals.json`：带 step/layer/request-ID/slot/length 的调用区间，Worker 退出前导出。
- `trace/guest.trace`：原有 syscall/FDMFS/iomap/NVMe SQ/CQ 事件，不增加实际 I/O。
- `stack-summary.json`：逐 pwrite、Q/mean、EXEC、结果读取的设备内外时间及并集。
- `decode_stack.broad_stack`：剔除诊断读取后的 Decode 分母；窄口径 syscall 设备外时间，
  以及加入 GPU/CPU 传输提交与等待、打包、mean、参数和绑定包装的宽口径软件时间。
- `read-summary.json`：NAND 读取量、有效读取时间、LUN 分布；后台聚合另列，不能与前台累加。

分析器按每个批次的实际 APPEND 数校验，不再假设一条 APPEND 对应一条 EXEC；
MR10 trace 回读默认关闭，不为计时增加诊断读。各时段取并集并裁剪到真实 Worker Decode
execute/sample 窗口，通过 SQ..完成边界约束 Guest HPET 与 QEMU virtual 的偏移，报告上下界。
不把 Host monotonic/CUDA event 时间直接减进 Guest 时间。设备范围包含其队列/依赖等待，
软件范围包含阻塞/调度与采集扰动，不等于纯 CPU 成本或可优化收益。传输 API 等待也可能
包含上游 GPU 生产时间；剩余范围仍包括模型 GPU 运算、采样、框架和未归因间隙。
Worker 外部的 frontend 调度/IPC 不在该 Decode 分母内，`generate_seconds` 另行保留。

需要检查采集扰动时，在新的同配置 Guest 会话执行
`CEMU_VLLM_STACK_TRACE=0 bash /root/CEMU/femu-scripts/vllm-opt-guest-test.sh /path/to/softmax.json`。
这关闭 Guest 精细采集，保留 basic 阶段计时及 Host 诊断；不是所有计时都关闭的实验。

## 无 GPU 的设备链路回归

使用已构建的 CEMU qtest 二进制与当前 Python 版本的 `_cemu_client`：

```bash
cd /home/lihan/cemu/CEMU
PYTHONPATH=tests/cemu/instattention/python:tests/cemu/instattention/build-host:tests/cemu/instattention \
  /home/lihan/cemu/hf-env/bin/python -m unittest test_vllm_cemu -v
```

测试用临时 NVM 文件和 Unix socket，执行真实 QEMU FDM 接收、聚合 `.so`、SparF workflow 和
CPU Attention 算子；不访问正在运行的 Guest。覆盖非单位 scale、默认 scale、GQA、B2/B4
变长/重排/缩批、两层、token 页/channel 组边界、后台聚合、无 trace 回读、APPEND 计数，
并断言每层每轮一条 EXEC。畸形长度、offset、重复/越界 slot 必须被设备拒绝。
稀疏参考输入使用无 top-r 并列的 FP16 幅值，避免两个合法 tie-break 策略影响比较。

已另用 `CUDA_VISIBLE_DEVICES=0 CEMU_NATIVE_QTEST_CUDA=1` 在宿主机 A100 验证 OPT MHA
全选择及 Llama GQA 稀疏路径的真实 CUDA `.so` 执行。
未覆盖：真实 vLLM worker import/初始化、Guest CUDA Flash SDPA、模型输出和端到端性能。必须完成上面的模型验收后才能把状态更新为完整接通。

源码复查补充：正常 Worker 执行不保证存在 `set_current_vllm_config` 上下文，
因此缓冲容量在 Runtime 初始化时保存，forward 不调用 `get_current_vllm_config()`。
`python/test_vllm_backend.py` 在 CPU 上执行实际适配器 forward 和 Session，测试上下文缺失时的
首次 Prefill、batch 重排、整批单次提交与单请求、混合 Prefill/Decode、probe 无副作用和缓冲复用。
其中 vLLM 基类、CUDA 事件/传输和设备输出使用替身，张量计算与 Session 逻辑真实执行；
这组测试同样不能替代真实 vLLM/CUDA 验证。装有 torch 的环境可运行：

```bash
PYTHONPATH=tests/cemu/instattention/python \
  /path/to/python -m unittest test_vllm_backend -v
```
