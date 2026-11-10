# InstAttention / CEMU 文档导航

当前目标是复现论文行为再评估模拟精度，不以吞吐吻合或一次functional PASS代替行为验收。

vLLM / Llama-3.1-8B 的接入设计见
[vLLM 接入设计与实施边界](../../../docs/cemu/VLLM_INSTATTENTION_INTEGRATION.md)
和 [首版适配器安装与测试](python/vllm_adapter/README.md)
（2026-11-02，插件已实现、CPU QEMU 回归已通过，真实模型/CUDA 端到端待验证；保留现有 SparF、FDM 聚合/缓冲/写回及设备 CUDA 动态库路径）。

当前 native 串行路径的函数入口、Prefill/Decode 数据流、完成条件和计时边界见
[串行执行与数据流](NATIVE_SERIAL_DATAFLOW.md)，用于后续分阶段埋点和论文行为对照。
调用、系统调用、冻结区间及等待边界的图示见
[串行时序图与时间标记](NATIVE_SERIAL_SEQUENCE.md)（含 Prefill、单层 Decode、父 Execute 内部）。
物理硬件阶段、作者 NVMeVirt 模拟的已知边界、当前 CEMU 实现的逐项对应见
[物理阶段对应表](PHYSICAL_STAGE_MAPPING.md)。

2026-11-04 下午关于 KV `pwrite` 提交后返回、设备多块 input、包级 SQ/CQ、
共享完成环与源缓冲区延迟释放的初步方案见
[FDM 异步写入设计讨论记录](FDM_ASYNC_PWRITE_DESIGN_2026-11-04.md)；该方案尚未实现。

## InstAttention FlexGen 接入（2026-10-26）

### FDM 整层追加（2026-10-29）

FDM 客户端不再按 token/head 拆小包。Prefill 和 Decode 每次
`append_batch()` 将当前 layer/microbatch 的 K/V 打成一个包，调用一次 `write_range()`；
ioctl 后端正常完整写入时是一回 `pwrite`，uring 后端是一回 write 提交。
短写/EINTR 仍由现有客户端处理，底层 NVMe 命令仍可能拆分，不等于一条 NVMe 命令。
多个 microbatch 各自提交，不能为了整层合包而打破 FlexGen 的 microbatch 调度。

包头版本 2 保持 88 字节结构和 512 字节包头区；`batch` 是全局起始样本号，
原末尾 reserved 字段改作 `batch_count`。K、V 分别按
`[batch_count, head_count, token_count, head_dim]` 连续排列。设备仍接受版本 1 的单样本小包。
默认输入槽自动取 `max(128 KiB, align512(512 + 2 * microbatch_size * heads * max_tokens * row_bytes))`；
显式给小输入槽会报错，不自动退回分包。输入槽必须完整位于 MR1，且不超过 Linux 单次 write 上限。
OPT-13B B4/P1024 的 Prefill 包是 80 MiB + 512 字节，Decode 包是 80 KiB + 512 字节。
同一微批覆盖 B4、40 层、16 次 Decode 时，追加包为 Prefill 40 次、Decode 640 次，不含 DRAIN。

设备仍直接 DMA 到 MR1，按位图防止重叠，按接收单元计数判断整包完整；不逐分片反复扫描整个大包。
默认同步模式下，收齐的最后到达分片保留原请求，不提前 CQE。聚合每轮处理一个样本的一小段 token，直接索引原输入，
不创建子包/额外 Guest IO，也不复制到新的输入区。进度保存在 MR0，成功部分不重复追加；
轮间释放锁、唤醒 writer，让已满块下放和完成回收继续进行，避免要求页池一次容纳整层。
下一轮使用当前虚拟时刻重新参与完成队列调度，不新增固定聚合 runtime；资源不足仍按原规则重试。
全部聚合成功才释放输入槽并完成最后分片，聚合错误令 buffer 失败，不能重试整个已部分接受的包。

### FDM 后台聚合（可选）

Host 启动时设置 `CEMU_FDM_ASYNC_AGGREGATE=1`，可提前完成 APPEND：整包已进入 MR1、
包头/坐标/接收位图及 token 前缀验证通过，并到达输入写入模型完成时间后，返回最后分片 CQE。
这个返回值表示设备拥有输入数据，**不表示已完成聚合，更不表示 NAND 已持久化**。
默认 `0` 保留原同步基线；native APPEND 路径不受该开关影响。需要重启 QEMU 切换。

每个绑定创建一个聚合线程和一个可复用任务，直接读取原 MR1 输入槽，没有第二份整包复制。
后台仍调用原聚合模块、复用原暂存布局和 NAND writer，不新增每包 EXEC/ioctl，也不新增聚合冻结或虚构 runtime。
无工作或资源不足时用条件变量等待；每个有限大小的聚合片段之后释放锁，给写回和其他设备请求让出执行机会。

FDM 的 READ/WRITE/FILL 先按地址范围判断是否与绑定的 KV metadata/input/暂存数据 MR 相交。
相交时仍取得 KV 互斥锁，原有输入槽接收检查及保护区写入限制不变；不相交的 Q、mean、output 等区域
不再取得聚合线程使用的互斥锁。范围判断和访问期间持有独立的绑定生命周期读锁，绑定/解除使用写锁，
避免访问过程中绑定或 MR 描述被释放、重新分类；后台聚合不持有这把生命周期读写锁。
锁顺序为生命周期锁在前、KV 互斥锁在后。此次缩小锁范围不改变 NAND 建模、聚合布局或 APPEND 返回语义，
也不消除同 SQ 队头阻塞、CUDA mirror 同步等其他等待。重编并重启 QEMU 生效，不需要重编 Guest 内核。

第一版仍是**单输入槽**：后台释放输入槽前，后续命中该槽的 MEMORY_WRITE 暂留 SQ，禁止 DMA 覆盖；
其他 SQ 和 CQ 仍可推进，同一 SQ 的后续命令可能发生队头阻塞。这不是无限缓冲或多包并行聚合。
Attention 若需要已经收齐但尚未聚合的 KV，会在执行线程等待；就绪后在同一锁下取得读租约，
避免提前占用 reader 阻止聚合，也避免未读完被下一次追加改变。未接收的数据仍拒绝访问。
DRAIN 仍等聚合及 NAND 写回全部完成。后台失败为粘性错误，后续读取/DRAIN 报错；
已返回的 APPEND 不会被追溯改成失败。重置会唤醒并终止后台任务及依赖等待。

保持 B8/MB4、双 slot、32 LUN 等效带宽配置的启动方式：

```bash
cd /home/lihan/cemu/CEMU
make -C build -j4
sudo env CEMU_FDM_ASYNC_AGGREGATE=1 bash femu-scripts/run-flexgen-b8-microbatch-test.sh
```

Guest 仍运行 `bash /root/CEMU/femu-scripts/flexgen-b8-microbatch-guest-test.sh`。
启动前关掉使用同一系统盘/GPU 的 VM。此修改不要求重编 Guest 内核或聚合模块。
`launch-settings.txt` 记录 `fdm_async_aggregate=1`，便于确认实际模式。

开启原有 FDM I/O trace 后，前台记录保留 `source=device`；后台单独记录
`source=background_aggregate`、`session/packet_id`、整包 `packet_bytes`、
`accepted_ns/started_ns/finished_ns` 和原聚合阶段统计。这些时间戳使用 CEMU virtual，
`aggregate_host_ns` 仍是 Host 时间，不能混减。后台记录不是另一条 NVMe 完成。
命中繁忙输入槽的 SQ 首次观察时间纳入前台设备区间，并单列 `input_slot_wait_ns`；
不要将输入槽背压算成纯软件开销。分析器单列 `background_aggregation`，要求记录到最终 DRAIN；
后台跨度包含等待，不能与前台 syscall 或阶段耗时相加。
原“未被前台设备区间覆盖”统计也不能直接解释成纯软件耗时：后台聚合可以与它重叠，
同 SQ 未被观察的命令还可能包含队头等待。同步/异步比较应保留这些口径差别。
这次**没有**实现“整包收齐立即返回、后台异步聚合”。NAND 布局、放置策略、冻结机制不变。

管理区 ABI 升为 3，接收状态为 152 字节，buffer 头仍为 376 字节。
需重编 QEMU 和 `fdm_kv_aggregate.so` 并重启 VM，不可对运行中的旧设备热换客户端/模块。
FlexGen 默认数据区从 3840 MiB 调到 3968 MiB，容纳 B4 大输入槽及原页池；
旧命令若显式设置 3840 MiB，需要同步调整。B6/B8 的 8 GiB 数据区仍可用。
下文历史实验中的 128 KiB 输入槽、包数和耗时描述的是改动前的小包配置，不再代表当前默认行为。
计时新增 `aggregate_yield_ns`，单独记录大包分段聚合间的调度等待，不算成 NAND 或空间不足等待。

Host 回归（不需要启动完整 Guest 或直通 GPU）：

```bash
make -C build -j4
make -C tests/cemu/kernel ../build/fdm_kv_aggregate.so
PYTHONPATH=tests/cemu/instattention/python:tests/cemu/instattention \
  python3 -m unittest -v \
  test_instattention_qemu_io.InstattentionQemuIoTest.test_fdm_kv_layout \
  test_fdm_kv_binding test_native_csd_pipeline.FdmCsdPipelineTest \
  test_flexgen_cemu_qtest test_pwrite_stack_trace
```

### 框架来源与接入

框架副本位于 [`third_party/instattention/flexgen`](third_party/instattention/flexgen)，
从 InstAttention 提交 `3b2d4b27fbba2bf700dfffc672db6609f2efb91b` 导入全部 14 个
受版本控制的 Python 文件，不包含源工作区的未提交修改、`__pycache__` 或生成文件。
来源、原始 Git blob 和导入规范化记录见 [`SOURCE.json`](third_party/instattention/SOURCE.json)。
源快照的仓库根目录及 `flexgen` 下没有许可证文件；未将独立的 `ssd-gpu-dma/LICENSE`
套用到框架副本，也未新增许可证声明。

新入口 [`python/flexgen_cemu.py`](python/flexgen_cemu.py) 实际调用此副本的
`OptLM.generate()`，不调用旧 `OptPrefillRunner`/`OptDecodeRunner`。外部 `InstAttention`
克隆保持只读；`framework_import.py` 强制校验导入路径，拒绝已载入的外部同名包。
原 `test_opt_cemu_pipeline.py` 和 runner 保留作对照，不自动切换到新框架。

### 接入边界和数据流

```text
FlexGen OptLM.generate：原 normal / overlap single / overlap multi 循环
  InputEmbed → CemuSelfAttention → 原 MLP → OutputEmbed
    Prefill：GPU dense Attention → GPU Wo/MLP
             K/V → 有界写线程 → 现有 FDM 或 Native store
    Decode：GPU QKV → 等待本 microbatch 的 Prefill 接收完毕
             → append 当前 K/V 并等待 accepted（不是 NAND drain）
             → CemuSparfSlotScheduler → 设备自治 SparF
             → PendingAttention → 消费时等待 → GPU Wo/MLP
  生成结束 → store.drain() → 检查完整 KV 前缀和 sealed/failed
```

- `framework_attention.py` 替换 Attention 的 KV 存取及执行点。FlexGen 的 cache holder
  保存设备 KV 句柄；`load_cache()` 不把历史 KV 搬回 GPU。Decode 的 `store_cache()`
  只清理标记，不能再次追加当前 token。最后一个 Decode 的 KV 也会保存。
- `framework_cache.py` 管理 Future、GPU producer event 和生命周期。Attention 的完成结果
  在 MLP 消费激活时兑现；双 microbatch 可以先提交两份 CSD 工作，再消费结果。共享后端
  不允许 Prefill 追加与本 microbatch 的 Attention 同时在途，所以首轮 Decode 等待其所有
  Prefill 写任务 accepted；不等待 NAND 写回。Prefill 未完成写任务限额为 `2 × microbatch数`。
- FDM 的 `pwrite → 收齐 → 聚合 → accepted`、Native APPEND、三种 KV 布局、后台块写回、
  KV FTL、balanced 放置、SparF 选择/分阶段执行和现有时间预算全部复用。
  本次不改 `hw/femu`、设备算子、Guest 内核或冻结机制；不新增 P2P。
- 公开快照的 `pytorch_backend.py` 内 OPT 调用名对应了不同的 Llama 参数。
  新增独立 `opt_backend.py` 恢复 OPT 所需的 pre-LN、ReLU、Q 缩放和 Wo/LM head 接口，
  不覆盖 Llama 实现。框架中的模型层、microbatch 和三种生成循环保持原样。
- 第一版入口限定 OPT、FP16/FP32、未压缩且常驻 GPU 的权重/激活、单 CSD、等长无 padding
  的 prompt、greedy 固定步数；每次 `generate()` 创建新 session。尚未接 Llama、权重卸载、
  多 CSD 切分、动态 batching、采样/提前停止。保留框架结构不代表这些模式已验收。
- `--decode-steps 16` 对应 FlexGen `max_new_tokens=17`：Prefill 生成 1 个，随后执行 16 次
  Decode。设备接受的 KV 长度是 `prompt_length + 16`，不是再多 1 个。

### 当前验证

CPU 数值测试实际运行 `OptLM.generate()` 的单/双 microbatch、overlap 开/关，逐步 logits
及 token 与旧 OPT runner 的参考结果比较，并验证失败传播、无重复追加和外部导入隔离。
QEMU qtest 用真实 FDM/Native、MRS 和 SparF 设备路径跑两层小 OPT；覆盖单/双 microbatch、
compression=1/8、top-r=128/16、127→130 token 的 FP32 组页边界、最终 drain。
另覆盖 FP16、双 microbatch、compression=8 的 255→258 token 边界。新增 12 项测试通过；
新增测试与既有 FDM/Native/OPT/I/O 回归累计 73 项通过（CPU/qtest，不使用真实 GPU）。
这些不是 Guest Linux/FDMFS 系统调用或真实 OPT-13B GPU 性能验证，后者需另跑。

```bash
cd /home/lihan/cemu/CEMU
make -C build -j4
make -C tests/cemu instattention-flexgen-framework-test instattention-flexgen-cemu-qtest \
  PYTHON=/usr/bin/python3 INSTATTENTION_BUILD_DIR=instattention/build-host
```

除现有 PyTorch/NumPy/CEMU client 外，框架导入需要 `attrs`、`tqdm`。
固定 token ID 测试不需要 transformers；转换 safetensors 时才需要 safetensors。

### 权重准备与 Guest 全流程

FlexGen 使用逐 tensor 的 NumPy 文件，不读取旧 runner 的 checkpoint loader。
先离线转换**同一份本地 HF checkpoint**，不会下载模型或生成假权重：

```bash
cd /root/CEMU/tests/cemu
export PYTHONPATH="/root/native-opt-client:$PWD/instattention/python${PYTHONPATH:+:$PYTHONPATH}"
python3 -m flexgen_adapter.framework_weights \
  --model-dir /root/models/OPT-13B \
  --output-dir /mnt/flexgen-weights
```

`/mnt/flexgen-weights` 必须事先配置成**容量足够且可写的普通磁盘/共享目录**，不是 FDM，
也不是 native KV NVM；不要直接在 20 GiB Guest 根分区生成约 26 GB 的 OPT-13B 副本。
可在 Host 转换再只读共享给 Guest。转换按 shard 处理，保留源目录和文件清单；拒绝覆盖
已有输出目录。失败的半成品没有完整清单，确认后换新目标目录重试。

启动仍使用原 `run-native-opt-test.sh OPT-13B`，profile=`b4-p1024-d16`，
`CEMU_NATIVE_KV_PLACEMENT=balanced`，保持本机已验证的 GPU/镜像/初始化顺序。
Guest 按已有 native-aware mount 流程挂载 FDM，**不要格式化 native NVM**。
新入口不再调用 `native-opt-guest-test.sh` 的旧 runner；完成环境准备后执行：

```bash
(
  set -euo pipefail
  cd /root/CEMU/tests/cemu
  source /root/cemu-gpu-env/bin/activate
  make instattention INSTATTENTION_BUILD_DIR=/root/native-opt-client
  export PYTHONPATH="/root/native-opt-client:$PWD/instattention/python${PYTHONPATH:+:$PYTHONPATH}"
  export CEMU_IO_BACKEND=ioctl
  echo hpet > /sys/devices/system/clocksource/clocksource0/current_clocksource
  test "$(cat /sys/devices/system/clocksource/clocksource0/current_clocksource)" = hpet
  mkdir -p /root/cemu-logs
  output=/root/cemu-logs/flexgen-fdm-$(date +%Y%m%d-%H%M%S)
  ratio=/root/opt13b-b4-p1024-d16.json
  test -s "$ratio"
  python3 instattention/python/flexgen_cemu.py \
    --model-dir /root/models/OPT-13B --weights-dir /mnt/flexgen-weights \
    --storage fdm --batch-size 4 --gpu-batch-size 4 \
    --prompt-length 1024 --decode-steps 16 \
    --csd-target cuda --program /home/lihan/cemu/CEMU/tests/cemu/build/sparf_attention_devptr.so \
    --aggregate-program /home/lihan/cemu/CEMU/tests/cemu/build/fdm_kv_aggregate.so \
    --runtime-config instattention/experiments/configs/opt13b_sparf_1csd.json \
    --softmax-ratio-file "$ratio" \
    --profile-output "$output-profile.csv" --result-output "$output-result.json" \
    2>&1 | tee "$output.log"
)
```

`.so` 是 **Host QEMU 可见路径**；Host 须先编译现有两个算子。FDM 默认 MR0=64 MiB、
MR1=3840 MiB，另需 Attention slots 空间。Native 对照改为 `--storage native`，默认预算
6 GiB 与当前大 profile 对应；smoke 配置须相应减小参数。两条路径不能同时绑定同一 NVM，
先完成 drain/close；严格物理放置对比建议重新启动干净 session。
双 microbatch 用 `--gpu-batch-size 2`，必须另测 vectors=`2×40=80` 的 Softmax ratio；
入口会检查形状覆盖，不能直接沿用只测了 batch=4 的文件。

新入口会立即打印权重文件检查、后端初始化、逐模块权重加载 begin/done、GPU 权重就绪、
生成开始/结束及 final drain 边界。OPT-13B 的 `module=.../82` 包含输入嵌入、40 层各自的
Attention/MLP 和输出头；`layer=1..40` 是便于阅读的 Transformer 层号，不是 82 层模型。
模块 done 表示该模块初始化函数已返回；整体 GPU 就绪消息在原有 `model.sync()` 之后。
进度使用专用 `flexgen.weights` logger，不增加 CUDA 同步，不修改生成循环；加载进度不在
推理计时区间内，阶段边界打印有少量日志开销。已运行的 Python 进程不会获得新日志。

`pipeline.generate_wall` 是生成调用区间，`pipeline.e2e_including_drain_wall` 额外含终止写回。
`framework.prefill_iteration_wall`/`decode_iterations_wall` 来自原 FlexGen 每轮计时；前者
不保证全部异步 Prefill 写入已 accepted，余下等待归入首轮 Decode。
`decode.execute_wait_wall` 保留设备 EXEC 调用边界；主线程等待另记
`framework.attention_result_wait_wall`，不能把两者、写线程累计时间或模型预算相加当端到端时间。
HPET/Guest wall、CUDA event 和设备模型预算仍是不同口径；本次不声称新增校准。

### 软件栈分阶段计时

在 `flexgen_cemu.py` 原命令上增加 `--profile-stack`。默认关闭细分计时，已有粗粒度
profile 不变；开启后仍写入同一个 `--profile-output` CSV，并打印 count、累计、平均、
最大时间及部分阶段的字节数。没有逐包打印、额外设备查询或逐阶段 CUDA synchronize。
新计时属于 Guest Python/C++ 客户端，不改设备、FTL、Linux 内核和 freezing，无需重启 VM。
先在 Guest 重编译绑定；旧绑定会在加载权重前明确报错：

```bash
cd /root/CEMU/tests/cemu
source /root/cemu-gpu-env/bin/activate
make instattention INSTATTENTION_BUILD_DIR=/root/native-opt-client
ctest --test-dir /root/native-opt-client -R '^client_profile$' --output-on-failure
export PYTHONPATH="/root/native-opt-client:$PWD/instattention/python${PYTHONPATH:+:$PYTHONPATH}"
python3 -c 'import _cemu_client; assert _cemu_client.IO_PROFILE_VERSION == 1'
```

`client_profile` 使用普通临时文件和模拟管理/EXEC ioctl，验证数据、对齐、命令、返回值及
错误传播，不代替 Guest 的完整推理回归。正式测试仍用同一模型、prompt、ratio、HPET 和
设备配置，只加 `--profile-stack`，更换输出文件名；可用 `--reference-result` 对照上次 token。

| 范围 | 新指标 | 含义 |
|---|---|---|
| Prefill 写线程 | `prefill.writer.queue_wait_wall` / `worker_wall` | submit 前到任务开始，以及写线程任务执行区间 |
| GPU 数据就绪 | `prefill.producer_wait_wall`、`decode.producer_wait_wall` | 原有 producer event 等待，不增加同步 |
| KV 导出 | `prefill.kv_export_wall`、`decode.kv_export_wall` | 连续化、GPU→CPU、NumPy 转换，含可能的同步等待 |
| KV 存储调用 | `prefill.store_append_wall`、`decode.store_append_wall` | 导出后到 store 返回；FDM/native 都覆盖 |
| FDM 打包 | `prefill/decode.kv_payload_pack_wall`、`fdm_packet_pack_wall` | 切片后的转置/字节化，以及包头和填充构造 |
| KV payload 内部分解 | `prefill/decode.kv_payload_pack.k_copy_wall`、`v_copy_wall`、`join_wall` | K 重排/字节化、V 重排/字节化、K/V 字节串拼接；各自记录处理字节数 |
| V 均值 | `prefill/decode.value_mean_update_wall` | Host 上的均值累计；不归入 I/O |
| Decode 依赖 | `decode.layer_prefill_wait_wall` | 当前层 Prefill Future 等待；全部层等待仍见原 `framework.prefill_accept_wait_wall` |
| Decode 调度 | `decode.attention.queue_wait_wall`、`decode.slot_wait_wall` | Attention executor 排队，以及可用 FDM slot 等待 |
| slot 各线程 | `decode.slot_prepare/slot_execute/slot_collect.queue_wait_wall` 和 `worker_wall` | 准备、EXEC、输出收集各自的排队与处理 |
| Query 准备 | `decode.query_export_wall`、`decode.value_mean_prepare_wall` | Q 导出与当前 value_mean 的生成/拷贝 |
| GPU 工作 | `prefill.gpu_attention`、`decode.qkv/output_to_gpu/wo_residual` | 各有 `.host_wall` / `.cuda_stream`；Prefill Attention 含原 forward 整体 |
| GPU MLP/输出头 | `framework.mlp`、`framework.lm_head` | 同样有两种计时；累计包含 Prefill 和 Decode |

KV payload 三个子区间只在细分计时开启时采集，FlexGen 与 vLLM 共用实现；
vLLM 使用 `--profile detailed`。它们嵌套于原来的 `kv_payload_pack_wall`，不能再与父项
相加，也不改变软件栈占比的统计口径。三段使用四次连续 Guest 时钟读取，记录在操作后
集中写入内存，不逐包输出或增加 GPU 同步；父项还包含临时对象释放和计时记录等开销。
这些仍是 elapsed wall 区间，不是纯 CPU 执行时间；Decode 小包更容易受计时本身扰动。

每个阶段仅保存首个打包输入的 shape、字节 strides、dtype、连续性、转置视图布局及
NumPy 版本，不保存 KV 内容。查看 `CEMU_PROFILE_INTERVALS` 输出中的 `metadata`；
FlexGen 的 CSV 同时生成 `<profile.csv>.metadata.json`，vLLM 的结果文件
`timing.metadata` 也包含该信息。输入布局记录点在 `ascontiguousarray` 之后，
不表示原始 GPU 张量布局或 pinned/NUMA 属性；变长场景中首包不代表所有包。
更新共享 Python 源码后重新启动测试进程即可，本次打包细分无需重编译或重启 VM。

对 FDM 写包、Q/mean 写入、输出/trace 读取，以及 EXEC，再分 C++ 客户端边界。
前缀分别为 `prefill.fdm_write`、`decode.fdm_write`、`drain.fdm_write`、
`attention.query.write`、`attention.value_mean.write`、`attention.output.read`、
`attention.trace.read`、`attention.execute`：

| 后缀 | 统计边界 |
|---|---|
| `.call_wall` | Python 调用客户端到返回；包含以下子项和绑定层返回计时结果的开销 |
| `.binding_prepare_wall` | pybind 内的参数处理；EXEC 包含 Python metadata 拷贝 |
| `.buffer_prepare_wall` | C++ 参数检查、对齐缓冲区分配/清零、写入拷贝或 EXEC 命令构造 |
| `.io_wait_wall` | `pwrite_all` / `pread_all` / ioctl 或 uring 提交并等待；包含内核和设备服务，不是纯软件耗时 |
| `.buffer_finish_wall` | 读完成后 C++ 输出 vector 分配/拷贝；写入/EXEC 当前为零 |
| `.binding_finish_wall` | 绑定层剩余区间，含 GIL 交接、C++ 清理及输出数组转换，不是纯 CPU 执行时间 |
| `.unpack_wall` | 仅 read_tensor：返回 NumPy 数据的 view/reshape/copy |

`.io_wait_wall` 的 bytes 是对齐后的 FDM 传输长度，EXEC 则仅是 metadata 长度；count 是
客户端调用次数，不是内核拆出的 NVMe 命令数。失败的调用保留外层 `.call_wall`，不输出
不完整的 C++ 子阶段；因此失败运行不能当完整性能报告。Native APPEND 本轮只覆盖
`store_append_wall`，未细分 passthrough 内部，不能与 FDM 子阶段直接做相减归因。

Python `perf_counter_ns` 和 C++ `steady_clock` 都读取 Guest 时间；仍需验证/保持 HPET。
排队区间也可能包含线程创建、GIL 和调度，不是独占 CPU 时间。父子指标互相包含，多个
worker 可重叠；不要把所有 CSV 行相加，或用端到端减模型累计量称为“纯软件栈时间”。
CUDA event 会受真实 GPU 推进影响，不能和 Guest 时间互减；设备 timeline 的 Host 时间也单独看。
细分计时本身有开销，应保留关闭开关的基线，尤其短句、小包和 HPET 下不要忽略测量扰动。

单独开启 `--profile-stack` 不会细分 Guest 内核或设备接收/聚合。进一步拆分
`.io_wait_wall` 使用下节可选跟踪；已有设备聚合 Host 累计时间仍不能直接从 Guest 时间扣除。

### FDM pwrite 内核与设备细分

针对 Prefill 的 `prefill.fdm_write.io_wait_wall`，新增
`experiments/pwrite_stack_trace.py`。它不是把同步等待全部算作设备工作：**SQ/CQ、完成通知、
线程唤醒同样属于待核查的 I/O 通路延迟**，但 SQ/CQ 并非文件系统独有，native 命令也会经过。
当前只采集 `--storage fdm` + `CEMU_IO_BACKEND=ioctl` 的同步 `pwrite`，不声称覆盖 uring。

**本功能需要重编译 QEMU 和 linux-cemu，并重启 Guest**，不同于仅开启上一节客户端计时。
内核只增加可关闭的 tracepoint、为 FDMFS 开始事件补文件名/范围；设备只增加时间戳和计数，
不改 FTL、聚合、冻结、背压及完成语义，也没有新增逐包状态查询。所有新跟踪默认关闭。

| 口径 | 报告指标（以 `prefill.` 为前缀） | 边界/用途 |
|---|---|---|
| Guest | `syscall_wall` | `sys_enter_pwrite64` → `sys_exit_pwrite64` |
| Guest | `single.vfs_entry` / `fdm_entry` | 系统调用入口 → FDMFS → iomap DIO 入口 |
| Guest | `single.dio_block_driver` | iomap → NVMe 请求准备完成，含用户页映射、bio/块层、PRP 准备 |
| Guest | `single.sq_submission` | NVMe 准备完成 → 开始拷贝 SQ 命令 |
| Guest | `single.sq_cq_roundtrip` | SQ 拷贝前 → Guest 看见 CQE，含 doorbell、设备、通知和 CQ 路径，**不是纯设备时间** |
| Guest | `single.nvme_completion` | 看见 CQE → NVMe 完成事件，含中间完成处理/可能的延后调度 |
| Guest | `single.bio_wakeup_dio` | NVMe 完成 → iomap 完成，含 bio 完成、唤醒及被调度回来 |
| Guest | `single.fdm_return` / `vfs_return` | iomap 完成 → FDMFS 返回 → 系统调用退出 |
| 设备虚拟时钟 | `device.dispatch` / `memory_lock` / `receive_check` | 取出命令后分派、内存锁等待、输入范围/接收状态检查 |
| 设备虚拟时钟 | `device.dma_copy` / `receive_finish` | PRP 数据搬运，以及 dirty 标记/位图/包头收齐检查 |
| 设备虚拟时钟 | `device.remaining_model_gate` | 接收处理结束后尚未到期的原有 FDM 模型 deadline；不是软件执行耗时 |
| 设备虚拟时钟 | `device.ready_to_check` / `check_to_cq` / `cq_publish` | deadline 后首次聚合检查的调度延迟、检查结束到写 CQ、写 CQ 本身 |
| 设备虚拟时钟 | `device.aggregate_ns` / `check_other_ns` | 聚合函数执行，以及完成检查函数里其余锁、准备、释放等工作 |
| 设备虚拟时钟 | `device.space_retry_ns` / `busy_retry_ns` / `drain_retry_ns` / `model_retry_ns` | 因空间、读 lease、drain、接收 deadline 未满足而重试，直到下次检查的间隔；含队列/poller 延迟 |
| Host | `host.aggregate_ns` | 单独记录聚合实际执行区间，仅用于定位实现成本 |

每项输出次数、累计、平均、最大、P95/P99；`counters` 还统计包数/字节、聚合次数、空间和
lease 重试次数、CQ 满次数。`device.check_span = check_ns + retry_gap`，
`check_ns = aggregate_ns + check_other_ns`，四类重试间隔之和等于 `retry_gap`；这些父子项不能重复相加。
这里不把 Prefill 完成误解为 NAND 全部写回；drain 单列，空间背压仍可能等待后台写回推进。

一条 `pwrite` 只对应一条 NVMe 命令时，九个 `single.*` 区间互不重叠，之和应等于
该调用的 `syscall_wall`。内核拆成多条命令时，只报告 syscall 总时间和 `request.*` 各请求时间，
**不把并行请求的累计时间相加冒充 syscall 的独占分解**。客户端 `.io_wait_wall` 还含
`pwrite_all` 循环/系统调用包装边界，不要求与内核时间精确相等。

跨线程完成通过 `(controller, SQID, CID)` 关联；Guest 包标签记录 session/request ID，
再与设备的 namespace、命令 ID、偏移、长度核对。缺事件、短写、请求失败、关联不唯一、
丢 trace、时钟切换都报 `FAIL/INCOMPLETE`，不输出看似完整的归因。当前 NVMe 请求准备需在
发起同步写的线程内；若块层异步转交导致无法关联，会明确失败而不是按最近时间猜测。

Guest 使用 `trace_clock=mono`，保持已经验证的 HPET；文本 trace 时间戳为微秒精度。
设备用 QEMU virtual，Host 聚合用 Host monotonic；**不将两域时间戳相减，不把等待等同 CPU cycles**。
设备子区间帮助解释 Guest SQ/CQ 往返，尚不能给出严格扣除设备时间后的单一“纯软件时间”。
trace 本身有扰动：内核事件、Guest reader、逐包标签、设备 JSON 写日志都会增加工作；设备
JSON 在 CQ 写入后、该轮通知前输出，也会影响完成通知延迟。应与关闭跟踪的相同配置对照。

Host 构建（只编内核 `.o` 不会更新启动用的 `vmlinux`）：

```bash
make -C /home/lihan/cemu/linux-cemu -j4 vmlinux
make -C /home/lihan/cemu/CEMU/build -j4
```

在已验证的 Host `sudo env ... bash femu-scripts/run-native-opt-test.sh OPT-13B` 命令中增加
`CEMU_NATIVE_IO_TRACE=1`，其余镜像、GPU、初始化顺序、profile、placement 都不变。
默认内核路径为 `/home/lihan/cemu/linux-cemu/vmlinux`；若覆盖了 `CEMU_NATIVE_KERNEL`，须指向新内核。
设备文件写入本次 session 的 `guest-logs/fdm-io-device.jsonl`，通过既有 guestlogs 挂载后
可见于 `/root/cemu-logs/fdm-io-device.jsonl`。该文件不是 `device-timeline.jsonl`。

Guest 完成原有 FDM 挂载、虚拟环境和客户端构建后，保持同一权重、ratio、输入，执行：

```bash
(
  set -euo pipefail
  cd /root/CEMU/tests/cemu
  source /root/cemu-gpu-env/bin/activate
  export PYTHONPATH="/root/native-opt-client:$PWD/instattention/python${PYTHONPATH:+:$PYTHONPATH}"
  export CEMU_IO_BACKEND=ioctl
  echo hpet > /sys/devices/system/clocksource/clocksource0/current_clocksource
  mountpoint -q /sys/kernel/tracing || mount -t tracefs tracefs /sys/kernel/tracing
  test -f /sys/kernel/tracing/events/nvme/nvme_cemu_io/enable
  mkdir -p /root/cemu-logs
  run=$(mktemp -d /root/cemu-logs/pwrite-stack.XXXXXX)
  ratio=/root/opt13b-b4-p1024-d16.json
  test -s "$ratio"
  python3 instattention/experiments/pwrite_stack_trace.py record --output "$run/trace" -- \
    python3 -u instattention/python/flexgen_cemu.py \
      --model-dir /root/models/OPT-13B --weights-dir /mnt/flexgen-weights \
      --storage fdm --batch-size 4 --gpu-batch-size 4 \
      --prompt-length 1024 --decode-steps 16 \
      --csd-target cuda --program /home/lihan/cemu/CEMU/tests/cemu/build/sparf_attention_devptr.so \
      --aggregate-program /home/lihan/cemu/CEMU/tests/cemu/build/fdm_kv_aggregate.so \
      --runtime-config instattention/experiments/configs/opt13b_sparf_1csd.json \
      --softmax-ratio-file "$ratio" --profile-stack \
      --profile-output "$run/profile.csv" --result-output "$run/result.json" \
    2>&1 | tee "$run/run.log"
  python3 instattention/experiments/pwrite_stack_trace.py analyze "$run/trace" \
    --device /root/cemu-logs/fdm-io-device.jsonl \
    2>&1 | tee "$run/analysis.txt"
)
```

录制器创建独立 tracefs instance，不清空其它 trace，子进程退出后停止并删除自己的 instance；
采集期间请勿改时钟或并行跑其它 KV 测试。`trace/capture.json` 保留内核版本、命令、退出状态、
时钟和丢事件统计；`guest.trace`/`packets.jsonl` 是原始记录，`summary.json` 是分析结果。
新 trace 目录必须不存在；如果记录器缺 tracepoint 会在加载权重前停止，先核查 Guest 内核。

本地验证入口：

```bash
make -C tests/cemu instattention-pwrite-stack-test PYTHON=/usr/bin/python3
PYTHONPATH=tests/cemu/instattention/python CEMU_NATIVE_QTEST_CUDA=0 \
  /usr/bin/python3 -m unittest -v \
  test_native_csd_pipeline.FdmCsdPipelineTest.test_fdm_io_trace_packet_identity_and_completion
```

解析器测试覆盖拆命令、跨线程完成、CID 复用、缺事件/失败检测；CPU qtest 覆盖真实设备的
输入分片、聚合、drain 和跟踪包身份。它们不替代带新内核的 Guest OPT-13B 实测。

#### pwrite 设备外区间与阶段覆盖统计

`pwrite_stack_trace.py analyze --account-device` 在原始记录上离线计算，不增加设备探针。
每次 pwrite 先关联它的全部 NVMe 命令，然后对设备区间求并集，不能直接减去逐命令耗时之和。
报告 `rows[].device_accounting` 保留每次写入的结果，
`device_accounting.syscall_by_phase` 分 Prefill/Decode/drain 汇总；比例以总 syscall 时间加权。

提供两种设备边界，不能混为“纯设备计算时间”：

- `request_envelope`：设备看到 SQ 到开始发布 CQE，排除 CQ 发布本身。不能使用 `cq_end`
  作为必然早于 Guest 看到 CQE 的事件，因为 Guest 可能在设备打结束时间戳前已发现完成。
- `service_and_wait`：`memory_enter..max(memory_done,model_ready)` 加
  `first_check..last_check`。包含接收、DMA、模型等待、聚合及重试；第二段也包含 poller
  再检查/聚合让出的间隙，目前无法从累计计数恢复每次间隙的位置，因此不是严格的算子活动区间。
- `guest_outside_pending_nvme`：仅使用 Guest 时钟，syscall 减去 SQ 提交到 CQ 观察区间的并集。
  表示没有这些命令在途的时间，不等于全部驱动/文件系统执行时间。

跨时钟分析要求录制前后都为 HPET，假设 Guest 单调时钟与 QEMU virtual 等速，通过
每条命令的 `Guest SQ <= device SQ`、`device CQ begin <= Guest CQ observed` 约束共同偏移范围。
Guest 文本 trace 预留 2us 量化容差；假设不成立、时钟漂移导致无共同偏移或混入多个 session
时直接报错，不静默扣减。此检查不是硬件时钟校准。输出设备外时间/比例的上下界；
当全部相关设备区间始终落在单次 syscall 内时，区间长度与时钟原点无关，上下界可相同。

录制器同时设置 `CEMU_PROFILE_INTERVALS=.../trace/phase-intervals.json`。
`PhaseProfiler` 复用原有 scope 的起止读钟，区间暂存内存，`write_csv()` 时才输出 JSON；
FlexGen 的阶段边界复用其已有 generate timer 的真实 start/stop，不重建近似边界、不增加同步。
未启用环境变量时不保存区间。默认最多保存 500000 条，可用 `CEMU_PROFILE_INTERVAL_LIMIT`
调整；超限记录 dropped 数，阶段分析拒绝不完整结果，而不是输出偏小占比。

新运行可同时关联同一次 QEMU 的 Attention 时间线：

```bash
python3 /root/CEMU/tests/cemu/instattention/experiments/pwrite_stack_trace.py analyze "$run/trace" \
  --device "$run/fdm-io-device.jsonl" --account-device \
  --attention-device "$run/device-timeline.jsonl" \
  --output "$run/pwrite-stack-summary.json" \
  2>&1 | tee "$run/pwrite-stack-analysis.txt"
```

`flexgen-b8-guest-test.sh` 的 `CEMU_FLEXGEN_STACK_TRACE=1` 模式已自动执行上述分析，B4/B6/B8
共用该入口。已有原始 Guest/device 日志可以只加 `--account-device` 重算 syscall 占比；
旧运行没有 `phase-intervals.json` 时，不能从累计 CSV 恢复阶段交叠，报告明确标为不可用。

阶段结果位于 `device_accounting.pipeline.phases`，按实际 Prefill/Decode/drain 窗口裁剪：

| 字段 | 含义 |
| --- | --- |
| `categories.append_syscalls` | 阶段内 KV 写 syscall 的并集，以及不被设备区间覆盖的部分 |
| `categories.client_preparation` | store append 排除 client I/O call，加追加绑定层在 syscall 外的区间、metadata/value-mean 准备；不是所有 CPU 工作 |
| `categories.main_thread_blocking_wait` | 写队列背压及 Attention future 等待的并集，以及设备覆盖之外的等待 |
| `scopes` | 每个已有计时 scope 在该阶段内的累计时间与去重时间 |
| `foreground_device_complement` | 阶段窗口未被已跟踪前台设备区间覆盖的部分，**不是软件栈占比** |

Prefill worker 若延续到 Decode，相关区间计入实际发生的 Decode 窗口，而不是全部算到 Prefill。
Attention 时间线必须通过完整性、PID、请求数和运行时间范围检查；设备边界采用 started..ready，
包含排队/阶段衔接，不表示 NAND 始终繁忙。Q/结果的普通 FDM 设备 I/O、后台 NAND 活动
尚未完整覆盖，因此 `software_stack_fraction=null`；不能把阶段补集全部算成 CEMU 软件栈。
各类别仍可能相互重叠，不可相加；主线程等待补集也不是已证明的因果关键路径或纯 CPU 时间。

#### Decode 按请求拆分 Future 等待

新增结果位于 `device_accounting.pipeline.decode_request_wait`，控制台输出 `[decode-wait]`。
上述全局设备覆盖统计仍保留，但新的拆分只用**当前 Future 自己的**追加和 Attention
设备区间，避免 slot B 的设备工作掩盖 slot A 的锁等待。两种口径的补集可能不同，
不能要求新结果仍等于之前全局统计的 3.45～3.53 秒。

每次 Decode 关联 layer、microbatch、有效 token 数；请求标识随线程池任务传递到准备、
执行、结果读取线程，并写入对应追加包的诊断记录，不改变设备命令或包头。
EXEC 另记录实际 program ID，离线按 program/layer/tokens 和因果时间边界匹配设备记录；
缺失、重复、歧义、跨请求匹配以及记录丢失均拒绝分析，不按全局发生顺序猜测。

`totals.wait_ns` 是 Decode 窗口内主线程 Future 等待的并集，`outside_min_ns/max_ns`
是其中未被本请求已跟踪设备区间覆盖的上下界；二者之差才是对应的设备覆盖区间。
`rows` 保留逐请求结果。`categories` 按以下优先级分配剩余区间，排除嵌套重复：

| 类别 | 测量范围 |
| --- | --- |
| `lock_wait` | 共享 KV store、backend、设备 slot 锁的获取等待，不包含持锁期间工作 |
| `worker_queue` | Attention/准备/执行/收集任务提交至 worker 开始执行 |
| `producer_or_prefill_wait` | CUDA producer event 或 Prefill Future 等待 |
| `slot_wait` | 获取空闲 Attention slot 的等待 |
| `append_syscall_outside_device` | 本请求 pwrite 中设备已跟踪区间之外的部分 |
| `exec_transport_or_device_queue` | EXEC 等待中设备 started..ready 之外的部分，仍含设备准入/完成通知等 |
| `query_result_trace_io` | Q/mean 写入、结果/trace 读回，尚未扣除这类普通 FDM I/O 的设备工作 |
| `kv_query_export` | K/V/Q 导出，包括可能的 GPU→CPU 拷贝和等待 |
| `packet_packing` | KV payload 和包头/整包组装 |
| `mean_and_metadata` | value mean 更新/准备与命令 metadata 构造 |
| `append_binding_outside_syscall` | 追加绑定调用中已排除 syscall 和上述类别的部分 |
| `return_to_waiter` | callable 主体退出至主线程 result() 返回，含 Future/线程调度及计时本身；不是精确 Future 状态转换时间 |
| `unclassified` | 未分类的软件执行、调度空隙或其他未观测工作，不能直接视作纯软件栈开销 |

各类别的最小值之和等于 `outside_min_ns`，最大值之和等于 `outside_max_ns`；
这些是有时钟偏移不确定性的保守边界，不是总 CPU 时间，也不是整个 Decode 的软件栈占比。
只统计落在主线程等待内的区间；等待之前已经完成的工作不会再加进去。

需要用更新后的 Python 代码重新录制一次，旧日志没有请求标识，不能可靠补算。
继续使用 `flexgen-b8-microbatch-guest-test.sh` 的精细计时流程即可，无需新增开关；
此次只修改 Python 计时/离线分析，不改 QEMU、Guest 内核、NAND、冻结或 I/O 完成语义。
如果当前 VM 已有前述设备/内核探针并共享更新后的源码，无需为加载这些 Python 改动重启。
但这不代表可以无限复用同一个设备会话：FDM 解绑/删文件不回收已写入的 KV 映射页，
新 session ID 也不会清零控制器的累计页数。B8 完整性能对照每轮应先 `poweroff`，
再由 Host 重新运行 `run-flexgen-b8-microbatch-test.sh`，建立 fresh NVM 会话。
仅 Guest `reboot`、清空 FDM 目录或重新挂载都不能替代新建 QEMU 会话。
若出现 `FDM KV writer batch failed ... status=8`，这是 `CEMU_KV_LIMIT`；
在逐页 reserve 路径表示映射页数达到 `max_pages`，不是计时分析失败或主机 DRAM 耗尽。

建议同配置交替跑关闭/开启 trace 的实验：关闭时报告整体性能，开启时解释组成。
内存记录和现有内核/设备 trace 都有扰动，离线求并集没有推理期负担；不能按总扰动比例
机械地修正各阶段时间。本次统计改动无需重编 QEMU/内核，前提是运行的版本已有上述 tracepoint。

#### 整个 Decode 的文件/NVMe 软件路径占比

`--decode-stack` 补齐六类同步调用：KV append pwrite、Q pwrite、value mean pwrite、
Attention EXEC ioctl、output pread、trace pread。精细计时 Guest 脚本已自动启用该分析。
这次需要更新 QEMU **和 Guest 内核**，不是只更新上文的 Python 计时：

```bash
cd /home/lihan/cemu
make -C linux-cemu -j4 vmlinux && make -C CEMU/build -j4
```

旧 VM 先在 Guest 内 `poweroff`，再在 Host 执行以下命令建立 fresh NVM 会话。
脚本直接加载 `linux-cemu/vmlinux`；NVMe 驱动编入内核，不能仅更新模块。

```bash
cd /home/lihan/cemu/CEMU
sudo bash femu-scripts/run-flexgen-b8-microbatch-test.sh
```

进入 Guest 后继续原配置 B8 / microbatch 4 / P1024 / D16 / 双 slot / ioctl / HPET：

```bash
bash /root/CEMU/femu-scripts/flexgen-b8-microbatch-guest-test.sh
```

输出 `[decode-stack]`，详细结果在 `pwrite-stack-summary.json` 的 `decode_stack`：

| 字段 | 含义 |
| --- | --- |
| `decode_window_ns` | 完整 Decode 窗口，不限于主线程 Future 等待 |
| `software_union` | 各 syscall 扣除**自身**设备区间后，裁剪至 Decode 并求并集；给出秒数及占 Decode 的比例上下界 |
| `software_not_overlapped_by_tracked_device` | 上一并集再扣除其他已关联前台设备工作区间；后台 NAND 活动未完整纳入，不能称为关键路径 |
| `by_kind` | 六类调用各自的 syscall 并集、软件区间并集，以及占 syscall/Decode 的比例；类别之间可能重叠，不能相加 |
| `rows` | 逐调用与命令边界，包括 prepared→SQ、CQ observed→complete；EXEC 另列设备准入排队、执行、ready→CQ |

本口径将设备 SQ 取命令至 **CQ 发布开始** 作为设备归属区间，包括设备分发、内部排队、
数据搬运、建模等待和 poller 延迟；CQ 发布、Guest 通知与唤醒留在外侧。
因此统计的是**同步文件/NVMe 路径内、设备归属区间之外的经过时间**，不是纯 CPU 执行时间，
不是所有 FlexGen/Python 软件时间，也不能直接解释为更换软件栈可节省的时间。
双 slot 使用并集去重，而非相加调用耗时，也不是简单地用 Decode 总时间减 NAND 模型时间。

内核额外记录 pread/ioctl 以及 EXEC（compute namespace opcode `0x01`）的 SQ/CQ 边界，
设备 trace 使用 version 2。EXEC 记录中的 `offset`/`bytes` 是命令字的匹配签名，
不是文件偏移或传输字节数。离线结合 controller/qid/cid/nsid/opcode/命令字、线程、
文件名和因果时间关联，并校验 CID 复用、分拆 I/O 字节覆盖和每个 Decode 请求六类调用齐全。
缺失、歧义、时钟不一致、失败 I/O 或 trace 丢失均拒绝给出完整占比。
Guest 脚本在加载权重前检查内核 trace 的 `coverage=2` 标识；旧日志不能补算这项统计。

Decode 分析使用 Guest `SQ copy → nvme_complete_rq` 包住设备 `sq_seen → cq_begin`，
在已有 append 对齐范围内检查所有命令存在共同的时钟偏移，再计算并集。
不再要求更早的 `cq_seen` trace 时间戳作为硬性上界：实测有一条 pread 的 SQ→CQ
仅 4 µs，而设备区间为 26.198 µs；其后 `nvme_complete_rq` 记录完整，命令身份可唯一关联。
该异常保留在 `decode_stack.cq_seen_alignment_anomalies`，控制台也报告数量，
不会跳过请求、扩大固定容差或按日志顺序猜配。若完成边界也不能满足约束，仍拒绝分析。
这只是采用更晚的因果边界，不会把 CQ→complete 变成设备时间，设备归属边界仍为 `cq_begin`。
异常的具体时钟采样原因尚未确定；完成边界一致不等于证明所有时间戳精确，
报告上下界只覆盖对齐偏移不确定性，并非任意时钟采样误差的保证。
已有 version 2 原始日志可离线重算，不需为这个分析器修正重启或重跑推理。

只新增可选探针和离线分析，不改变 pwrite 完成、聚合、EXEC、冻结或 NAND 调度语义。
探针仍会产生扰动，性能结论需要同配置未开启 trace 的对照。

### 固定输入与输出对照

新入口的默认人工 token 序列已对齐旧 runner：`arange % (vocab_size - 3) + 3`，
首 token 从模型 `config.json` 读取 `bos_token_id`，OPT-13B 第一行以 `[2, 4, 5, 6, ...]`
开头。此前 FlexGen 入口使用 `+4`，输入不同，不能拿之前两份日志的 token 直接判错。
这里固定的是**输入**，不会固定、替换或强制生成的输出。

每次新入口运行都会保存 `<result 文件去掉 .json>.prompt.json`，结果 JSON 中另保存完整
`case`（模型配置、prompt IDs、Decode 步数、top-r、compression）和输出 token。
显式输入可以在不加载权重、不打开 CEMU 的情况下准备：

```bash
cd /root/CEMU/tests/cemu
source /root/cemu-gpu-env/bin/activate
export PYTHONPATH="/root/native-opt-client:$PWD/instattention/python${PYTHONPATH:+:$PYTHONPATH}"
python3 -m opt_inputs \
  --model-dir /root/models/OPT-13B --batch-size 4 --prompt-length 1024 --decode-steps 16 \
  --output /root/cemu-logs/inputs/opt13b-b4-p1024-d16.json
```

在原有旧 runner 的 functional SparF 命令中添加：

```bash
--prompt-ids /root/cemu-logs/inputs/opt13b-b4-p1024-d16.json \
--result-output /root/cemu-logs/old-runner-result.json
```

在前述 FlexGen 完整命令中添加以下选项，并给 `--result-output` 使用不同的输出文件：

```bash
--prompt-ids /root/cemu-logs/inputs/opt13b-b4-p1024-d16.json \
--reference-result /root/cemu-logs/old-runner-result.json
```

读取参考文件时先检查输入、模型配置和 SparF 参数；不同或缺少 `case` 会在模型预加载前
拒绝运行。生成结束逐 token 比较，失败时报告第一个不同的 batch/生成位置（0 表示 Prefill
产生的 token），保存实际结果与 profile，并以非零退出；不会把实际输出改成参考输出。
没有 `--reference-result` 时明确打印 `token comparison=not_requested`。

这是**相同输入的 token 回归对照，不是逐步 logits 数值验收**，也不证明参考文件本身正确。
参考和待测必须使用同一份 checkpoint；配置一致不等于权重内容已经做过哈希校验。
旧历史 JSON/日志缺少输入记录时不会自动猜测输入，需要重新导出。严谨性能对照仍须一致的
设备配置、ratio、HPET、预热条件及独立干净 session。

### 真实短句推理

`--prompt-text` 使用模型目录的本地 tokenizer，自动采用真实 token 长度，不补齐到 1024，
不截断；batch>1 时重复同一句。它与 `--prompt-ids` 互斥。文本输入会自动打印续写和完整
文本；重放 IDs 时加 `--decode-text` 即可解码。需要当前 Python 环境安装 `transformers`，
且模型目录已有 tokenizer 文件；代码不联网下载、不执行自定义 tokenizer 代码。

下面在已经启动、挂载 FDM 且已转换权重的 Guest 中执行。仍使用原 SparF/设备路径，
但 B1 和真实短 prompt 的 Softmax 形状不同，**先单独测 ratio，不能复用 B4/P1024 文件**：

```bash
(
  set -euo pipefail
  cd /root/CEMU/tests/cemu
  source /root/cemu-gpu-env/bin/activate
  make instattention INSTATTENTION_BUILD_DIR=/root/native-opt-client
  export PYTHONPATH="/root/native-opt-client:$PWD/instattention/python${PYTHONPATH:+:$PYTHONPATH}"
  export CEMU_IO_BACKEND=ioctl
  python3 -c 'import transformers'
  clock=/sys/devices/system/clocksource/clocksource0/current_clocksource
  echo hpet > "$clock"
  test "$(cat "$clock")" = hpet
  mkdir -p /root/cemu-logs
  output=/root/cemu-logs/flexgen-text-$(date +%Y%m%d-%H%M%S)
  input="$output-prompt.json"
  config=instattention/experiments/configs/opt13b_sparf_1csd.json

  python3 -m opt_inputs \
    --model-dir /root/models/OPT-13B --batch-size 1 --decode-steps 16 \
    --prompt-text 'The capital of France is' --output "$input"
  prompt_length=$(python3 -c 'import json,sys; print(len(json.load(open(sys.argv[1]))[0]))' "$input")

  python3 instattention/experiments/bench_softmax_ratio.py \
    --config "$config" --device cuda:0 --batch-sizes 1 \
    --prompt-length "$prompt_length" --decode-steps 16 --compression-ratio 8 \
    --output "$output-softmax.json" \
    2>&1 | tee "$output-softmax.log"

  python3 instattention/python/flexgen_cemu.py \
    --model-dir /root/models/OPT-13B --weights-dir /mnt/flexgen-weights \
    --storage fdm --batch-size 1 --gpu-batch-size 1 \
    --prompt-ids "$input" --decode-text --decode-steps 16 \
    --csd-target cuda --program /home/lihan/cemu/CEMU/tests/cemu/build/sparf_attention_devptr.so \
    --aggregate-program /home/lihan/cemu/CEMU/tests/cemu/build/fdm_kv_aggregate.so \
    --runtime-config "$config" --softmax-ratio-file "$output-softmax.json" \
    --profile-output "$output-profile.csv" --result-output "$output-result.json" \
    2>&1 | tee "$output.log"
  final_clock=$(cat "$clock")
  printf '[flexgen-clock] final=%s\n' "$final_clock" | tee -a "$output.log"
  test "$final_clock" = hpet
)
```

仍执行固定的 Prefill + 16 次 Decode，不因 EOS 提前停止。文本解码隐藏特殊 token，但 JSON
保留全部 17 个原始生成 IDs。自然语言续写仅用于可读性冒烟测试，不代替准确率/困惑度评估，
也不与 B4/P1024 性能直接对照。文本编码、解码和结果比较均在推理计时区间外。

## FDM KV buffer 收包、聚合、写回与 Attention（2026-10-25）

当前实现 **创建 FDM 管理/数据文件及一个 MRS → 加载并激活一次 Host 聚合模块 → INIT/BIND →
普通 FDM WRITE 分片接收 → 整包 READY → 输入目标时间到达 → 非冻结聚合 → WRITE 完成**，
同时由后台 writer 将成块页提交 KV FTL，目标时间到达后回收页槽；终止包 DRAIN 等待全部写回。
`FDM_KVbuffer.open()` 在 `create_memory_ranges()` 成功后加载 `fdm_kv_aggregate.so`，
随后调用 `CemuClient.bind_fdm_kv_buffer()`；每个包只写 FDM，不另发聚合 EXEC。
模块默认 Host 路径为 `./build/fdm_kv_aggregate.so`（相对于 `tests/cemu`），可通过
`aggregate_program` 指定。下载/激活复用现有 CEMU 接口，不修改 Guest Linux 驱动。
此管理操作固定使用 `NVME_IOCTL_ADMIN_CMD`，不随 `CEMU_IO_BACKEND` 切换。

复用 compute namespace 的 MRS 管理 opcode `0x21`，新增 selector `2`：
`cdw10 = 2 | (rsid << 16)`，`cdw11 = 80`，`cdw12 = 聚合 program ID`，
`cdw13..15` 为零。绑定只接受已激活的 Host shared-library direct program，
其 runtime/runtime_scale 必须为零，不能把 Attention 计算预算套到此软件处理阶段。
80 字节小端描述符为 `FDMKV_BindParams_t`：magic/version、目标 NVM NSID、
写回批量上限、session/generation、输入槽长度，以及模型几何参数。
Python 的 `nvm_namespace_id` 默认 1、`input_bytes` 默认按整层/microbatch 自动计算；后者是
MR1 输入槽容量，不是当前包的实际长度，也不等于 KV 追加已完成。
控制器身份、MR 地址和 NAND channel/block 几何由设备查表获取，不接受 Guest 指针。

- MR0：直接存 `FDMKV_Buffer_t`、收包 bitmap、head/page/domain/writeback 管理数组和数据段表。
- MR1：起始 `input_bytes` 为输入槽；按页对齐后为暂存页池首段；MR2.. 可接续页池。初始化不修改数据段。
- `NvmNamespace.fdm_kv_buffer` 指向 MR0 内的 buffer；
  `MemoryNamespace.fdm_kv_binding` 记录同控制器的目标 NVM、MRS、输入槽偏移和长度。
  `FDMKV_Binding_t` 保存设备侧绑定及 writer 运行态，没有另分配 KV 数据页池。
- 当前要求全部 MR 在同一 FDM namespace；一个 NVM 和一个 FDM namespace 各至多绑定一个 buffer。
  MR 长度受现有 uint32 描述符限制，单 MR 必须小于 4 GiB。
  预算、页大小、重叠范围及 namespace 类型不合法时拒绝绑定，不修改初始化区域。
- 绑定持有 MRS 和 program 的使用引用，普通 EXEC 不可复用该 MRS；绑定期间不能
  停用、覆盖加载或卸载聚合 program。删除 MRS 自动解除绑定。
  控制器 reset/退出在后端释放前清除绑定；reset 后须重新 BIND。
  接收、聚合、后台队列/完成回调和普通 FDM READ 共用 FDM namespace 的互斥锁。
  正常删除在 pending WRITE、在途块、lease，或未失败且仍有暂存页/未收齐包时返回 BUSY；
  先 DRAIN，再删除，不隐式丢弃 accepted 数据。reset/退出属于取消而非排空：先停 poller，
  在 `to_poller/pq` 释放前停止并 join writer、取消内部完成事件，最后解除绑定。

当前自动聚合使用 `ftl_mode=instattention`，即 KV-only NVM；FDM 的普通读写仍保留。
绑定期间该 NVM 拒绝 native KV 数据命令（保留 INFO），避免同时启用另一条缓冲路径；
已有 native tail 时拒绝 FDM BIND。原 native 路径在没有 FDM 绑定时不变。
这是 **CEMU 软件栈用于 LLM 推理的 baseline**，不是要求实现普通 LBA 与 KV 双 FTL 共存。
共存延后。FDM 暂存/Flash 的 Attention 读取适配、Python Prefill/Decode 封包和终止
DRAIN 已接入；小模型 CPU/QEMU qtest 推理已通过，但尚未在 Guest Linux/FDMFS 下
验证真实 OPT-13B 全流程，不据此声称性能或论文精度已验证。
`FemuCtrl.native_kv` 由 `ftl_mode=instattention` 决定，表示控制器采用 KV 专用 FTL，
不表示数据必须来自 `NativeKvBuffer`。因此这条 FDM 路径仍进入 native poller 分支；
命令根据 namespace 分发，FDM WRITE 进入 `memory_write()`，不是 native APPEND。
为保留早期接收回归，`ftl_mode=femu` 仍支持 `cdw12=0` 的只接收诊断绑定，
但它不能自动聚合，不能作为新路径的正常运行配置。

`memory_write()` 在 DMA 前检查目标范围与收包状态。命中输入槽后，仍用原有
`dma_write_prp()` 直接写 MR1，不另分配整包中转区；DMA 成功才更新 MR0 的接收 bitmap
（每 4 字节一位）。支持同一个包的分片乱序到达、包头分片和空洞检查；一个槽只允许一个包
在途，禁止混入另一包、重复/重叠写入，不能用覆盖旧片段的方式重试。

包头区占 512 字节，其中 `FDMKV_AppendHeader_t` 为 88 字节小端结构，其余字节必须为零。
包头携带 session/generation/request_id、batch/layer、head/token 范围及 wire/payload 长度；
版本 2 用末尾字段表示 batch_count，payload 长度必须等于
`2 * batch_count * head_count * token_count * row_bytes`，后接补零至 512 字节的尾部。
版本 1 的 batch_count 隐含为 1，末尾字段必须为零。
输入槽容量与实际 `wire_bytes` 分别校验；包头没有收齐前不解析坐标。这里只验证包格式和
坐标范围，head 的连续追加前缀、页池容量等检查留给聚合步骤。

状态为 `EMPTY → RECEIVING → READY → AGGREGATING → AGGREGATED → EMPTY`。
普通分片 WRITE 仍只确认本片接收；使整包收齐的那条 WRITE 记录为 pending，
保留原请求和字节数，并把完成目标提高到整个包各片目标的最大值。
原 `to_poller/pq` 到期且 CQ 有空间后，调用绑定模块聚合，再返回该 WRITE 的 CQE。
没有新增计算 EXEC、CU 排队、冻结区间或合成 runtime；当前同步处理占用 poller 的
实际开销属于本 baseline 的设备软件实现开销，不能直接称为真实 CSD 的聚合延迟。
普通文件写入可能被拆成多条命令，本协议按覆盖位图判断完整包，而非按 syscall 次数判断。

无效完整包头/尾部、DMA 失败或聚合失败转为 `FAILED`。聚合前缀错误返回 `0x4002`；
页池不足且存在整块待写/在途写回时，保留 READY 包及原 WRITE，在完成队列中延后检查，
不向 Guest 返回失败，也不要求 Guest 重发包。每轮至多尝试一次，下一次检查目标为当前虚拟
时间加 10us；这是本实现的重试策略，不是校准后的硬件延迟。没有后台进展可能时才返回
`NVME_CAP_EXCEEDED | NVME_DNR`（0x4081）。FTL 注册/写入失败使 buffer 失败，不能重试
整个已部分写出的块；后续 WRITE/DRAIN 不得宣称成功。
只有旧的 `cdw12=0` 只接收诊断绑定，收齐时仍返回 0x4001 并保留 READY。

收包记录各次 WRITE 的最大虚拟完成时刻 `max_write_ready_ns`。
READY 表示数据功能上已收齐，不保证模拟时间已经走到该时刻；聚合在完成队列到期后执行。
本阶段不修改 freezing、原有 FDM 传输计时或 NAND 模型。

绑定期间普通 WRITE 不得修改 MR0/暂存页池，FILL 不得覆盖任一保留 MR；其他区域仍走
普通路径。COPY/P2P、其他 MRS 别名或算子写入尚未纳入收包协议，不得用于改写这些保留 MR，
这里不是通用内存访问隔离。MR0 内指针仅供设备进程使用，不是 Guest 可解引用的地址。

Host 上编译 QEMU 后运行（不启动 Guest OS、不访问 GPU/VFIO/已有镜像）：

```bash
make -C build -j4
make -C tests/cemu/kernel ../build/fdm_kv_aggregate.so
make -C tests/cemu instattention-fdm-buffer-write-qtest PYTHON=/usr/bin/python3
```

此目标兼容原 `instattention-fdm-buffer-init-qtest` 和 `instattention-fdm-buffer-receive-qtest`。
测试覆盖 Python open/close/错误回滚、
小规模真实 QEMU 管理及 FDM WRITE 命令、NSID/绑定拒绝条件、MR0 初始化/MR1 保持不变、
乱序/分片包头/空洞/重复/越界/无效头/DMA 失败/READY 保护、解绑/reset 后重绑，
以及旧普通 NVM/FDM 读写、新 KV-only FDM 自动聚合、模块引用保护、输入时间门槛、
pending reset、多包槽复用、token/channel 边界、三种页布局逐字节校验和聚合失败传播。
写回测试还覆盖 1059-token 长序列超过页池容量后的复用、跨 channel 多块在途、deadline 前
保留 MR1/hash、NO_SPACE 保留原 WRITE 且只发布一次 accepted、分片终止 DRAIN、
reset 取消未到期写回、新 session 重绑，以及解绑后按 KV 坐标回读 Flash 的三种布局。
不覆盖 Guest Linux/FDMFS/ioctl 驱动端到端，更不等价于完整推理测试。

### FDM 推理接线与验证边界

`FDM_KVbuffer.append_batch()` 复用 `SparfGroupedAttentionBackend`，把
`[tokens, heads, dim]` 拆为不跨 batch/layer 的连续 head 包，每包 K 在前、V 在后，
每份按 `[head, token, dim]` 排列。每包最多 `2 * token_group_tokens` 个 token，
同时受输入槽容量约束，包头和填充也计入容量。一个输入槽仍只允许一个包在途。
`CemuClient.write_range()` 经原文件写接口（`pwrite` 或选定的 io_uring WRITE）写 MR1，
返回后才更新 Guest accepted 前缀；不存在每包另发聚合 EXEC 的操作。
写失败使客户端会话不可继续使用，不能盲目重试可能已部分写入的包。

Attention EXEC 新增显式 `CEMU_SPARF_WORKFLOW_FDM` 标志，元数据为 112 字节公共头加
24 字节 session/generation/batch_first/NVM NSID。native 的原 128 字节格式保持不变。
设备按当前控制器的绑定表找 FDM buffer，校验身份、几何和每个 head 的 accepted 前缀，
持有读 lease，完成或取消后释放；不是把 `FDMKV_Buffer_t` 强转为 native buffer。

三阶段读取统一进入 `fdm_kv_attention_read()`：

```text
channel/token 索引 → KV 坐标 → 逻辑页键
  ├─ MR0 哈希表命中：直接从 MR1 暂存页取数据，包括 NAND 尚未完成的在途页
  └─ 未命中：按页去重 → bb_native_kv_batch_observed → KV FTL/NAND/后端
  → 拼入 Attention slot 的 FDM MR → 原 SparF 读取 deadline 等待 → 原计算阶段
```

同一阶段中，源选择和暂存复制在 FDM 锁下完成；Flash gather 使用有界临时描述符/读 scratch，
不复制第二份持久 KV 页池。Flash 完成目标仍由现有 NAND 模型计算，并由原状态机等待；
不更改 freezing。诊断保留实际 Flash 页数、去重前字节、有效字节、LUN 和 full-scan 参照。
同 head 的 APPEND 遇到读 lease 时保留原 WRITE 等待，不覆盖页或毒化 buffer；
DRAIN 等待已有 lease，期间不接纳新的读 lease。解绑忙状态不能被误报为成功。

`release_memory_ranges()` 使用标准 NVMe admin passthrough 检查删除结果，因为旧
Guest CEMU DELETE_MRS 包装没有向用户返回设备拒绝状态。未修改 Guest 内核代码；
新路径必须重新编译 `_cemu_client`。关闭先 DRAIN，再检查删除 MRS，最后卸载聚合 program。
失败后如果设备仍返回 BUSY，保留错误，不宣称清理成功；可能需要 reset 后开始新会话。

Host 回归（真实 QEMU 设备，CPU 算子，无 Guest Linux/GPU）：

```bash
make -C build -j4
make -C tests/cemu instattention-fdm-csd-qtest PYTHON=/usr/bin/python3
```

测试覆盖 FP16 的 15/16、253/256/259 边界、GQA、压缩比 1/8、混合暂存/Flash 读取、
终止排空后的纯 Flash 读取、身份拒绝、lease 下 APPEND 等待和 reset；另以合成权重的
2 层小 OPT，batch=2、prompt=127、decode=3，核对 token、logits、hidden states 与
PyTorch reference。这里不是 OPT-13B checkpoint 或论文数据集测试。

完整 Guest pipeline 已支持 `FDM_INTERNALBUF=1`（CLI `--fdm-internalbuf`），
与 `SPARF_NATIVE`/`SPARF_GROUPED` 互斥，必须同时使用 functional/autonomous SparF。
需要显式提供 `FDM_BUFFER_META_BYTES`、`FDM_BUFFER_DATA_BYTES`；聚合模块 Host 路径
可用 `FDM_AGGREGATE_PROGRAM` 指定。先在 Host 编译 `fdm_kv_aggregate.so`，并确认设备
FDM 容量足够容纳 MR0、MR1 和所有 Attention slots。不要沿用 native 路径的独立 DRAM
budget 当作 FDM 可用容量：现有 b4-p1024-d16 配置的 2 GiB FDM 不足以直接承载此页池。
先跑小规模 Guest 功能测试，再另配容量跑大形状；当前不自动增大设备容量，也不开放 NVM LBA。

### FDM OPT-13B 小规模 Guest 验收

`run-native-opt-test.sh` 支持用 `CEMU_NATIVE_CONFIG` 显式选择设备 JSON；不设置时
仍使用原 model/profile 对应配置。该文件会复制到本次会话的 `device-config.json`。
`experiments/configs/fdm_opt13b_smoke.json` 只把原 smoke 的 FDM 从 64 MiB 扩为
256 MiB，NVM 保持 512 MiB，NAND 几何、延迟和计算单元配置不变。

此配置先用于 OPT-13B、batch=GPU batch=1、prompt=32、decode=4、FP16、
channels_per_page=8、write_batch_pages=32。Guest 分配 MR0=4194304 字节（4 MiB）、
MR1=167772160 字节（160 MiB），输入槽仍为 128 KiB。当前初始化需要 32600 个暂存页，
MR1 实际占用 133660672 字节，MR0 管理区实际占用 2214920 字节；另留 Attention slots。
此形状完整三布局共 35200 页（144179200 字节），低于启动器 smoke 的 65536 页上限。
该配置不是 B4/P1024/D16 性能配置，不能直接沿用到大形状。

Host 启动选择 `CEMU_NATIVE_PROFILE=smoke`、`CEMU_NATIVE_CSD_TARGET=cuda`、
`CEMU_NATIVE_CONFIG=/home/lihan/cemu/CEMU/tests/cemu/instattention/experiments/configs/fdm_opt13b_smoke.json`，
并给 `run-native-opt-test.sh` 传 `OPT-13B`，使 Guest RAM 为 64 GiB。
Guest 直接调用 pipeline 的 `--fdm-internalbuf --sparf-autonomous --functional`，
不要调用固定选择 `--sparf-native` 的旧 Guest helper，不格式化或挂载 NVM。
功能验收暂不指定 runtime/ratio，不能以耗时作为校准性能。

### FDM OPT-13B B4/P1024/D16 性能采集

使用 `experiments/configs/fdm_opt13b_b4_p1024_d16.json`。它与原
`femu-scripts/cemu_config_opt13b_sparf_engines2.json` 仅有 FDM 容量差异：2 GiB 改为
4 GiB；96 GiB NVM、8 channel / 6 LUN、256 页/block、NAND 延迟、2 CU / 2 worker /
2 SparF engines 均保持不变。Host 设置 `CEMU_NATIVE_PROFILE=b4-p1024-d16`、
`CEMU_NATIVE_CONFIG` 指向此 JSON、`CEMU_DEVICE_INIT_ORDER=femu-first`，仍启动 OPT-13B。
初始化顺序选项只是本机已成功启动的绕行方式，不代表 CUDA 故障根因已解决。

Guest 使用 `--fdm-internalbuf --sparf-autonomous --functional`，batch=GPU batch=4、
prompt=1024、decode=16、Attention slots=2、top-r=16、compression=8、
channels-per-page=8、write-batch-pages=256；MR0=`67108864`（64 MiB），
MR1=`4026531840`（3840 MiB），输入槽仍为 128 KiB。单个 MR 长度字段为 uint32，
不能把 MR1 配成 4 GiB 或更大。初始化的精确需求为：

- 6400 个 head、972800 个暂存页，MR0 使用 63370896 字节，MR1 使用 3984719872 字节。
- 两个 Attention slots 合计预留 32628736 字节；加上 MR0/MR1 分配共 4126269440 字节，
  小于 4 GiB FDM，要求文件系统没有遗留分配。
- 三布局共 1344000 页（5505024000 字节），启动 profile 的 2097152 页上限足够；
  不能沿用 smoke 的 65536 页上限。

运行前必须加载 `experiments/configs/opt13b_sparf_1csd.json` 的 runtime model，并用
`SoftmaxRatioTable.validate_anchor()` / `require_sparf(4, 40, 1024, 16, 8)` 检查 ratio。
与旧结果对照应复用相同 ratio，不自动挑选文件或静默退回线性估计；每轮复制 runtime 和
ratio JSON 到日志目录。缺少 ratio 时，可在空闲 Guest GPU 单独测量：

```bash
python3 instattention/experiments/bench_softmax_ratio.py \
  --config instattention/experiments/configs/opt13b_sparf_1csd.json \
  --device cuda:0 --batch-sizes 4 --prompt-length 1024 --decode-steps 16 \
  --compression-ratio 8 --output /root/cemu-logs/opt13b-mb4-p1024-d16-softmax.json
```

测量脚本拒绝覆盖已有文件。新 ratio 的结果不能与旧 ratio 运行混作单变量比较；CUDA shape
ratio 仍不是经过 FPGA 校准的计时原型。Guest 固定 HPET，数据传输先固定
`CEMU_IO_BACKEND=ioctl`（FDM 写为 pwrite），避免同时改变传输方式；与 native 对照时也需
固定同一 backend。Host 开 `CEMU_NATIVE_TIMELINE=1` 保留时间线。

预期 640 次 Attention、6656000 个 head-token、33280 个聚合数据包；最后首个 head 的
accepted/token-K/channel-K/token-V 均为 1040，sealed=1、failed=0、暂存页/在途块/
pending_ready_ns 为零。比较 Prefill、Decode、最终排空与端到端时间，权重加载单列；
不要把 worker 累计时间相加，也不要将此短 Decode 工作负载视为论文图 14 的等价实验。
CPU QEMU 容量回归覆盖实际大 MR 绑定、高偏移访问保护、末端 batch/layer/head 的小包
聚合和 drain，不替代完整 Guest CUDA 性能运行。
预期 160 次 Attention 请求，最终三个 committed 前缀均为 36、sealed=1、failed=0，
暂存页、在途块及 pending_ready_ns 归零；数值正确性仍须结合独立参考测试。

容量和真实设备 INIT/BIND（不加载模型、不启动 Guest）的回归入口：
`python3 -m unittest test_fdm_kv_binding.FdmBindingDeviceTest.test_opt13b_smoke_capacity_and_binding`，
需将 `instattention/python` 加入 `PYTHONPATH`。它还验证 64 MiB 数据 MR 被拒绝，以及
配置只改变 FDM 容量，不以修改 NAND 模型来让测试通过。

### FDM OPT-13B B8/P1024/D16 带宽实验

Host 启动脚本新增 `CEMU_NATIVE_PROFILE=b8-p1024-d16`，仅支持 OPT-13B，默认选择
`experiments/configs/fdm_opt13b_b8_p1024_d16.json`，不需要再手动指定 `CEMU_NATIVE_CONFIG`。
此 JSON 相对 B4 FDM 配置只把 FDM namespace 从 4 GiB 增至 16 GiB；96 GiB NVM、
8 channel × 6 LUN × 1 plane、256 页/block、NAND 延迟和计算资源均不变。
放置策略仍由 `CEMU_NATIVE_KV_PLACEMENT` 选择，对比实验显式设置 `balanced`。

| 项目 | B8 容量 |
|---|---|
| Guest RAM / 默认 vCPU | 64 GiB / 4，与 B4 相同 |
| KV 映射页数上限 | 4194304 页，B4 为 2097152 页 |
| Native 独立 buffer 预算上限 | 12 GiB，B4 为 6 GiB；不是额外预分配的 FDM buffer |
| NVM 三布局最终页数 | 2688000 页，11010048000 字节（约 10.25 GiB） |
| 现有算法估算暂存页池 | 1944800 页，含输入槽的数据区至少 7966031872 字节（约 7.42 GiB） |
| Guest FDM 分配 | 管理区 128 MiB、数据区合计 8 GiB，另留两个 Attention slots |

单个 MR 的长度字段仍为 uint32。客户端将 8 GiB 数据区拆为三个独立 FDM 文件/MR，
每段最多 `4 GiB - 64 KiB`，不要求物理相邻。MR0 保存管理数组与段表，MR1 开头仍是
128 KiB 输入槽；其后的暂存页池由页索引解析到 MR1/MR2/MR3。聚合、后台写回和
Attention 的暂存页读取共用该解析，不新增 KV 中转，也不改变分包、domain、FTL 或计时。
设备保护所有绑定段，只有输入槽接受 Guest 写入。管理区 ABI 升为 2/header=376，
需重启新 QEMU、重编聚合 `.so` 和 Guest 客户端；不要把新客户端用于旧的运行中 VM。
单数据 MR 的 B4 路径也受回归覆盖，默认 64 MiB/3840 MiB 分配保持不变。

FDM 容量取 16 GiB 而不是原先计划的 10 GiB：现有 CEMU 直接以 FDM 容量注册 BAR2，
PCI BAR 大小必须是 2 的幂，10 GiB 会在 QEMU 初始化断言失败。这里只修正配置，
不修改 BAR 或 Guest 内核。元数据实际需求约 120.83 MiB，数据页池仍只用 8 GiB 分配。
旧 `native-opt-guest-test.sh` 不支持此 B8 profile，不要用它替代 FlexGen 入口。
完整测试必须 `batch_size=gpu_batch_size=8`，不能用两个 B4 microbatch 代替。
需要重新测量/验证 B8 Softmax ratio，不能直接使用只覆盖 B4 的文件。

停止旧 Guest 后，Host 启动（同一系统盘/GPU 不可被另一 VM 占用）：

```bash
cd /home/lihan/cemu/CEMU
make -C build -j4 &&
make -C tests/cemu/kernel ../build/fdm_kv_aggregate.so ../build/sparf_attention_devptr.so &&
sudo env \
  CEMU_NATIVE_BOOT_MODE=normal \
  CEMU_NATIVE_OS_IMAGE=/data/lihan/cemu-native-opt/standalone.PnutcI/guest.qcow2 \
  CEMU_NATIVE_PROFILE=b8-p1024-d16 \
  CEMU_NATIVE_CSD_TARGET=cuda CEMU_NATIVE_KV_PLACEMENT=balanced \
  CEMU_DEVICE_INIT_ORDER=femu-first \
  CUDA_VISIBLE_DEVICES=1 CEMU_CUDA_DEVICE=0 \
  CEMU_CUDA_LIBRARY_PATH=/home/lihan/cemu/nvidia-575-libs \
  CEMU_NATIVE_TIMELINE=1 CEMU_SPARF_READ_DIAGNOSTICS=1 CEMU_NATIVE_IO_TRACE=0 \
  bash femu-scripts/run-native-opt-test.sh OPT-13B
```

Guest 登录后：

```bash
bash /root/CEMU/femu-scripts/flexgen-b8-guest-test.sh
```

脚本挂载现有 FDM/模型/logsrc，检查空闲容量，不格式化 NVM、不删除残留 FDM 文件。
使用现有 `/root/models/flexgen-weights`，切换并核对 HPET，固定 ioctl/pwrite、128 KiB
输入槽，先在 Guest GPU 测 B8 ratio，再运行完整 FlexGen 推理并分析时间线。
已有合格 B8 ratio 可作为唯一参数传入；脚本复制并验证覆盖范围，不覆写原文件。

同一 B8 容量配置也支持 B6 对照，但须关闭旧 QEMU 并用上述 Host 命令重新启动，
在新的 native NVM 会话中执行（仅 Guest 内 reboot 或清空 FDM 不够）：

```bash
CEMU_FLEXGEN_BATCH_SIZE=6 bash /root/CEMU/femu-scripts/flexgen-b8-guest-test.sh
```

此选项同时设置 batch/microbatch 为 6，重新测量并校验 B6 ratio，按 B6 校验写入计数，
输出到独立 `flexgen-b6-d16.*` 目录；NAND/FDM 容量、8 GiB 数据页池和 128 KiB 输入槽保持不变。
默认仍为 B8，短实验支持 B4/B6/B8，不改变设备实现。比较逐请求 LUN 分布和阶段带宽，
不能仅比较不同 batch 的总时长。

#### B4 的 microbatch 短测试（32 LUN，2026-10-28）

固定总 batch=4、P1024、D16（每条序列输出17个token）、两个Attention slot，
比较 microbatch=4/2/1。默认先拆为两个B2。设备沿用长测的32 LUN、11703 ns页读参数，
而不是旧D16的48 LUN配置。保留同步pwrite/ioctl、128 KiB输入槽、8 GiB暂存池和现有页布局。

关闭上一轮Guest并等待QEMU退出后，在Host启动新的native NVM会话：

```bash
sudo bash /home/lihan/cemu/CEMU/femu-scripts/run-flexgen-b4-microbatch-test.sh
```

默认从 `/data/lihan/cemu-native-opt/standalone.PnutcI/guest.qcow2` 创建测试overlay；
需要更换基础镜像时通过 `CEMU_NATIVE_OS_IMAGE` 指定。启动前可以用
`CEMU_NATIVE_DRY_RUN=1 bash femu-scripts/run-flexgen-b4-microbatch-test.sh` 检查设备配置。
本入口默认关闭额外设备pwrite跟踪，保留Attention时间线和读取诊断。

Guest中执行下面三种之一，每种使用新启动的QEMU会话：

```bash
bash /root/CEMU/femu-scripts/flexgen-b4-microbatch-guest-test.sh 2  # 默认：两个B2
# bash /root/CEMU/femu-scripts/flexgen-b4-microbatch-guest-test.sh 4  # 单个B4参考
# bash /root/CEMU/femu-scripts/flexgen-b4-microbatch-guest-test.sh 1  # 四个B1
```

要复测此前的 syscall / SQ / CQ 细分，仍固定 B4、microbatch=2、D16，
关闭旧 Guest 后使用以下 Host 命令，再在新 Guest 中执行第二条命令：

```bash
# Host
sudo env CEMU_NATIVE_IO_TRACE=1 bash /home/lihan/cemu/CEMU/femu-scripts/run-flexgen-b4-microbatch-test.sh
# Guest
CEMU_FLEXGEN_STACK_TRACE=1 bash /root/CEMU/femu-scripts/flexgen-b4-microbatch-guest-test.sh 2
```

跟踪模式在加载权重前检查内核 tracepoint 和设备跟踪配置；完成后自动保存并分析
`trace/guest.trace`、`trace/packets.jsonl`、`fdm-io-device.jsonl`，生成
`pwrite-stack-summary.json` 和 `pwrite-stack-analysis.txt`，分别给出 Prefill/Decode
的 KV pwrite 软件及设备边界。该分析不覆盖所有 Q/mean/输出 I/O 或 Execute ioctl。
详细跟踪会带来额外开销，对照时注明开关。可通过第二个参数复用上一轮 MB2 的
`softmax.json`，避免重新标定 ratio 引入变化；文件需要在新 Guest 中可访问。

聚合算子内部细分（默认关闭）要求重新编译 QEMU 和 `fdm_kv_aggregate.so`：

```bash
make -C /home/lihan/cemu/CEMU/build -j4 qemu-system-x86_64
make -C /home/lihan/cemu/CEMU/tests/cemu/kernel ../build/fdm_kv_aggregate.so
sudo env CEMU_NATIVE_IO_TRACE=1 CEMU_FDM_AGGREGATE_PROFILE=1 \
  bash /home/lihan/cemu/CEMU/femu-scripts/run-flexgen-b4-microbatch-test.sh
# 新 Guest，仍使用上面的跟踪入口
CEMU_FLEXGEN_STACK_TRACE=1 bash /root/CEMU/femu-scripts/flexgen-b4-microbatch-guest-test.sh 2
```

`pwrite-stack-analysis.txt` 增加 `{prefill,decode}.host.aggregate_detail.*`：
`validate`（参数/布局/输入头校验）、`preflight`（三布局页查找和需求预检）、
`free_list`（空闲页链检查）、`token_k/channel_k/token_v`（各布局实际写入）、
`commit`（accepted和统计提交）、`wrapper`（外层Host计时减去内部阶段之和）。
这些区间在同一 Host 单调时钟下可加，覆盖失败/重试调用；不是虚拟模型预算。
三种布局的时间包含再次查找、分配/清零、复制/转置和ready入队，不能只解释为memcpy。
另有成功调用数、新分配页数、三布局复制总字节数（不含清零）和channel-K元素数。
不会在逐元素memcpy处打点；每个head/layout调用仍有计时开销，比较时注明开关。
不改变FDM存储ABI和写入语义；旧模块缺少profile入口时BIND会拒绝，避免静默漏测。

聚合算子独立 A/B 测试可用 `experiments/bench_kv_aggregate.py`。修改前先保存当前工作树
的 `tests/cemu/kernel/fdm_kv_aggregate.c`，不要用不包含已有修改的旧提交作为基线：

```bash
cd /home/lihan/cemu/CEMU
python3 tests/cemu/instattention/experiments/bench_kv_aggregate.py \
  --baseline /path/to/fdm_kv_aggregate.before.c \
  --output /path/to/new-result-directory --repeats 21
```

两份模块使用相同 `-O2 -march=native` 参数；测试固定一个可用CPU，交替运行A/B，
丢弃首轮，报告各轮每包平均耗时的中位数。用生产初始化、收包及算子代码，
仅算子调用计时，关闭内部profile；FP16/FP32均验证所有元数据/数据逐字节一致。
输入沿用H40/d128/P1024/D16和128KiB分包规则，单独报告Prefill、首步新组、后续15步、
全部16步Decode。测试在B1/L4暂存池中仅处理一个batch的一层，保留历史且无NAND writer；
内存预触页，数据生成、收包、验证不计时。因此它用于比较算子实现，不能当作完整设备或
GPU推理端到端加速比。输出保留源码快照、模块、CSV样本、校验日志和环境/汇总JSON。

可加 `--decode-cache-mib 64`，在每次Decode算子计时之前扫描64MiB的其他内存，
扫描本身不计入算子耗时。它用于观察缓存扰动下的收益，不保证完全冷缓存，也不能
替代实际40层推理交错执行的测量；默认值0保持连续调用测试。

聚合算子对单token、非token页组起点的追加使用已有页快速路径：整包预检时查页并
保存index，复制阶段复用这些index，三种布局的组号、页内偏移和domain按包计算。
临时计划最多2048个页索引（8KiB栈空间）；超出上限、新组和多token包走原通用路径。
索引只在持有buffer锁的本次调用内有效，不跨请求缓存。所有head预检通过后才修改数据，
accepted仍在三种布局全部完成后推进；页分配/清零和pwrite完成语义不变。

已有页Decode路径在暂存页数据超过64MiB时，提前预取接下来4个channel-K页的写入位置；
小缓冲区保持原路径。64MiB是Host实现的经验阈值，不是设备参数或模拟时间预算。
`bench_kv_aggregate.py --interleaved-batches 8` 可交错访问40层、8个batch的Decode页，
分别测首步新页和后续15步。这个测试只构造0..15的位置窗口（与1024..1039组内偏移相同），
没有历史KV、GPU、NAND writer或线程竞争，不能代替完整推理。`--page-ab`改为两边使用
同一baseline算子，仅比较小页/大页建议；必须检查输出的实际AnonHugePages，建议成功不等于
大页实际分配成功。

详细聚合profile开启时，设备原始trace还记录线程CPU时间、minor/major缺页与主动/被动
上下文切换增量。探针位于算子Host计时外侧，但会增加请求外围开销；CPU时间区间略宽于
Host计时区间，不能将两者差值当作精确的调度等待时间。它也不能测量cache/TLB miss。
逐包还记录Host起止时间、线程号、起止CPU编号及batch/layer/head坐标；起止CPU相同
不能证明中途未迁移。启动时保存 `host-execution-context.json`，包含NUMA节点CPU列表、
自动NUMA平衡开关、请求及实际内存策略和启动亲和性。
`run-native-opt-test.sh`（包括B4/B8包装脚本）默认设置
`CEMU_NATIVE_NUMA_POLICY=bind-all`，用 `numactl --membind=all` 启动QEMU，
允许使用所有可用NUMA节点，减少适用内存区域的自动NUMA采样干扰。
该设置不修改Host全局sysctl或CPU亲和性，也不添加预触页、mlock。
启动时验证策略；缺少numactl或策略未生效会报错，不静默回退。
`launch-settings.txt`记录 `host_numa_policy`，上下文JSON记录同一启动前缀下的
`numactl --show`。后续性能对照应使用相同策略，旧B4结果不能直接作为严格对照。
需要恢复默认Linux策略做对照时，在默认内存策略的终端执行：

```bash
sudo env CEMU_NATIVE_NUMA_POLICY=default bash femu-scripts/run-flexgen-b4-microbatch-test.sh
# B8对照入口仍支持 default / bind-all；普通B8入口现已默认bind-all。
sudo bash femu-scripts/run-flexgen-b8-numa-test.sh default
```

Guest脚本自动生成 `aggregate-decode-split.json/txt`，按新页分配/已有页追加及各token位置
拆分Host聚合耗时，并按CPU、已有页请求是否达到250us分组（250us只是诊断筛选阈值）。
完整已有页慢请求保存在JSON中；新运行会检查探针覆盖，使用旧QEMU时明确报错。
旧trace也可离线拆分，只是没有新增的CPU/缺页/调度计数：

```bash
python3 tests/cemu/instattention/experiments/aggregate_decode_trace.py \
  /path/to/fdm-io-device.jsonl --output /path/to/aggregate-decode-split.json
```

脚本按microbatch形状生成并检查Softmax ratio；可用第二个参数传入已匹配的ratio文件。
若已有同输入、同D16的B4参考结果，可设置 `CEMU_FLEXGEN_REFERENCE_RESULT=/path/to/result.json`
比较输出token（不等于完整logits精度验证）。不要使用D1023结果作这个参考。

每轮保存在 `flexgen-b4-mb{4,2,1}-d16.*`。父Execute预期分别为640、1280、2560次；
KV包数、写入字节数仍按总B4校验。`concurrency.json`保存设备实际并发统计；
拆分后若 `max_active_requests < 2`，脚本会报错，不能仅凭microbatch参数宣称已形成流水。
`result.json`另存 `prefill_seconds` 和 `decode_step_seconds`，用于区分首步衔接等待。
统计中的模型阶段累计时间可能重叠，不可当作整轮墙钟时间或直接与Guest耗时相减。

B8与上述B4对照时，Host使用 `femu-scripts/run-flexgen-b8-microbatch-test.sh`，
Guest使用 `femu-scripts/flexgen-b8-microbatch-guest-test.sh`。固定B8/MB4/P1024/D16，
两组microbatch、双slot，父Execute仍为1280次；KV包为66560个。Host显式复用B4的
32-LUN、11703ns页读配置，不能直接使用历史B8默认的48-LUN、62500ns配置比较。
所有设备timeline、NAND读取诊断、pwrite栈和聚合内部profile默认开启，Guest检查
实际设备配置及profile开关。未传ratio文件时重新测量MB4对应的ratio，不复用MB2表。
每轮 `flexgen-b8-mb4-d16.*` 保存原始trace、分段报告、设备配置、算子源码和so及SHA256。
单请求active-LUN数不能解释成整轮同时活跃的LUN数；结合请求级分布与并发统计分析。

当前 drain/close 会释放 FDM buffer，但不回收 NVM 的任务映射和已分配物理页。
B8 写入 2,688,000 页，B6 写入 2,016,000 页；同一会话连续运行会超过
4,194,304 页的映射上限并返回 `CEMU_KV_LIMIT`。每轮容量/放置对照使用新 QEMU 会话。
日志统一放在 `/root/cemu-logs/flexgen-b8-d16.XXXXXX/`，含 `profile.csv`、`result.json`、
`read-summary.json`、`read-loss.json`、`boundaries.json` 与 `bandwidth.txt`。
预期 640 次 Attention、13312000 个 head-token、66560 个 APPEND，最终写 11010048000
字节、accepted/三布局进度均为1040、sealed=1且failed/在途/暂存/待完成均为0。
这些是流程与计数检查，不代替完整模型 logits 正确性验证。

重点对照 B4 的每阶段 `active_luns_mean`、`lun_max_to_mean`、Flash 字节数与 `model_ns`。
`bandwidth.txt` 分别打印 `Flash字节/读阶段模型时间` 和 `Flash字节/Guest Decode时间`，
两种分母不能混用，也不能相加重叠阶段耗时。它们不是实际物理 NAND 总线带宽；现有
NAND 模型以 LUN 页服务排队为主。不要把 B8 原始总时长翻倍直接解释为带宽未改善。

### B4 / 1024 输出 / 32 LUN 带宽等效实验

独立配置 `experiments/configs/fdm_opt13b_b4_p1024_g1024_lun32.json` 使用
8 channel × 4 LUN × 1 plane，4 KiB 页，读延迟 11703 ns：
`32 * 4096 / 11703ns ≈ 11.1999 GB/s`（十进制），约 10.43 GiB/s。
这只是现有 LUN 排队模型的理想吞吐匹配，不是测得的 NAND 页延迟，也没有启用
共享 channel 总线传输建模；不能据此宣称已复现论文硬件。写延迟仍为 65000 ns。
物理几何对应 64 GiB NVM，FDM 仍为 16 GiB，Guest 数据区仍为 8 GiB，输入槽仍为128 KiB。

停止旧 QEMU 后，Host 执行：

```bash
cd /home/lihan/cemu/CEMU
sudo env \
  CEMU_NATIVE_BOOT_MODE=normal \
  CEMU_NATIVE_OS_IMAGE=/data/lihan/cemu-native-opt/standalone.PnutcI/guest.qcow2 \
  CEMU_NATIVE_PROFILE=b4-p1024-g1024-lun32 \
  CEMU_NATIVE_CSD_TARGET=cuda CEMU_NATIVE_KV_PLACEMENT=balanced \
  CEMU_DEVICE_INIT_ORDER=femu-first \
  CUDA_VISIBLE_DEVICES=1 CEMU_CUDA_DEVICE=0 \
  CEMU_CUDA_LIBRARY_PATH=/home/lihan/cemu/nvidia-575-libs \
  CEMU_NATIVE_TIMELINE=1 CEMU_SPARF_READ_DIAGNOSTICS=1 CEMU_NATIVE_IO_TRACE=0 \
  bash femu-scripts/run-native-opt-test.sh OPT-13B
```

在新 Guest 中执行（复用同一测试脚本，不复用旧 NVM 会话）：

```bash
CEMU_FLEXGEN_BATCH_SIZE=4 CEMU_FLEXGEN_DECODE_STEPS=1023 \
  bash /root/CEMU/femu-scripts/flexgen-b8-guest-test.sh
```

Prefill 产生第一个输出，1023 次 Decode 产生剩余输出，总计每条1024个输出，最大KV长度2047。
脚本重新测量/验证完整长度的 ratio（1151种目标形状，可能需要较长时间），统计分析范围为
1025～2047，预期40920次Attention、194400个追加包、13100800个head-token，最终
写2457600页（9.375 GiB）、sealed=1且无失败或未完成页。映射上限为4194304页。
日志在 `/root/cemu-logs/flexgen-b4-d1023.*`；长实验的逐请求 JSON/时间线显著增大，
请预留数 GiB 日志空间。输出分别包含 Decode、generate、包含 drain 的吞吐，不含权重加载。
该配置仅做静态容量、语法和 ratio 形状检查；完整长输出结果须以实际 Guest 运行日志为准。

### 容量回归与启动预检查

本地回归覆盖 B8 真容量绑定、非连续数据 MR 聚合/写回/Attention 数值及既有 Native/FDM
流程；不等同于已完成 Guest A100/OPT-13B B8 性能实测。

可以先在 Host 检查选中的配置，不启动 QEMU、不创建镜像、不访问 GPU，无需 sudo：

```bash
cd /home/lihan/cemu/CEMU
CEMU_NATIVE_PROFILE=b8-p1024-d16 CEMU_NATIVE_DRY_RUN=1 \
  CEMU_NATIVE_CSD_TARGET=cpu CEMU_NATIVE_KV_PLACEMENT=balanced \
  bash femu-scripts/run-native-opt-test.sh OPT-13B
```

dry-run 只打印容量选择并检查配置文件可读，不验证 JSON 内容、镜像、GPU、KVM、
可用内存或磁盘空间。`cpu` 仅用于这次配置检查，不改变后续 CUDA 性能测试的目标。
真实启动仍使用每轮独立的 file-backed `nvm.img`，不是预分配 96 GiB Host 匿名内存。
不自动创建 swap，不停止已有 VM，也不改变 runtime、FTL 放置或冻结机制。

### FDM 原地聚合模块与非冻结调用

这条路径是保留 CEMU 文件/MRS 软件栈的 **baseline**：KV 仍通过 FDM 写入接口进入输入槽，
管理结构和暂存页均使用已分配的 FDM，不以绕开 `pwrite`、删除传输成本或改变 freezing 来提速。
native 路径保留，后续用于行为与开销对照。

新增 `tests/cemu/kernel/fdm_kv_aggregate.c`，编译为 `fdm_kv_aggregate.so`，导出
`fdm_kv_aggregate(struct cemu_args *)`。它不依赖 native buffer，也不另申请 KV 页池。
复用 FDM 自己的布局计算和共享 ABI，布局计算源码已去除 QEMU 专用头依赖。
输入约定：`mr_addr[0]` 为初始化后的 MR0，`mr_addr[1..]` 为数据段，长度必须匹配绑定；
`cparam1=CEMU_FDM_KV_AGGREGATE_COMMAND`，`cparam2` 为设备调度入口取得的当前虚拟时间，
不是 Guest 可任意指定的就绪时间。该时间参数只检查前置传输依赖，不是聚合 runtime。
调用者必须独占该 buffer，保证 MRS/算子生命周期与 Host 数据一致性。

逻辑步骤：

```text
READY + 传输目标时刻已到
  → 校验所有 head 的连续追加前缀、reader、需要的新页和空闲页链
  → 整包资源足够才进入 AGGREGATING
  → 读取 MR1.input 中的 K/V，通过段表直接填入 MR1.. 的暂存页池
  → 更新 MR0 的页目录、有效 token 数和 domain 队列
  → 发布所有 head 的 accepted，进入 AGGREGATED
  → fdm_kv_receive_release() 清收包状态、递增序号
  → 原 WRITE CQE，Guest 可写下一包
```

包内 K、V 各自按 `[head_count, token_count, head_dim]` 排列，一个包对应一个 batch/layer。
token-K/V 页按 token 连续存储；channel-K 页按 `[channels_per_page, channel_group_tokens]`
存储，从同一份 K 转置填充。未满页保留在页目录中；满页只入队一次。
domain 按 batch/layer/kind/NAND-channel 区分，group 序号取模选 NAND-channel，不混淆
embedding channel 与 NAND channel。队列 `count >= block_pages` 表示已具备整块下放条件；
这一步只形成逻辑页键和候选队列，不分配真实 PPA、不提交 NAND、不推进 committed。

返回值：`OK=0`、`INVALID=1`、`BUSY=2`、`NO_SPACE=3`、`WAIT_TIME=4`。
前缀错误、reader 占用、空间不足或传输尚未到期均在修改前返回，管理区/数据段保持不变。
模块成功返回时保留输入包和 bitmap。`fdm_kv_complete_write()` 持锁调用模块并完成
Host dirty 标记及输入槽释放，之后 poller 才返回原 WRITE 的 CQE。
`fdm_kv_receive_release()` 只允许从 AGGREGATED 释放；它清空 bitmap/解析头、
递增 request_id，但不清 KV 暂存页。聚合不额外建模 runtime，不再另等一个聚合 deadline。

本阶段明确选择非冻结软件处理：设备通过已绑定 program 的入口调用模块，不进入
`run_functional_modeling()`。解绑时记录 `FDM KV non-frozen aggregation: calls=..., host_ns=...`，
这是模块调用的 Host monotonic 累计耗时，不包含前置 IO/排队，也不追加到虚拟时间。
其他任务仍可能冻结全局虚拟时钟，因此该 Host 数值不是虚拟时长或校准后的硬件性能。
后台写回独立于聚合模块，见下面的 writer 生命周期。下一步是 KV 专用读取和上层推理接线。

Host 验证命令：

```bash
make -C tests/cemu/kernel ../build/fdm_kv_aggregate.so
make -C tests/cemu instattention-fdm-buffer-aggregate-test PYTHON=/usr/bin/python3
```

测试实际编译并 dlopen 独立 `.so`，使用同一套 FDM 初始化/收包代码，逐字节检查 K 双布局/V、
多 head、多 batch/layer 隔离、FP16/FP32、head_dim=64/128/256、token/channel group 边界、
满页入队/部分页不提前下放、整包原子预检、虚拟就绪时间门槛、输入槽释放和旧包拒绝。
这个独立模块目标没有启动 QEMU；上面的 write-qtest 另行运行真实 QEMU 设备模型。
两者都没有启动 Guest Linux；独立模块测试不包含 NAND，write-qtest 包含设备后台写回，
均不代表新路径的 Attention/完整推理已跑通。

### KV FTL 整块放置 A/B

`ftl_mode=instattention` 新增控制器属性 `native_kv_placement=legacy|balanced`。
默认仍是 `legacy`；Native APPEND 和 FDM buffer 共用此 FTL 选项，无需改 Guest 写入协议。
`femu-scripts/run-native-opt-test.sh` 对应的 Host 环境变量是
`CEMU_NATIVE_KV_PLACEMENT=legacy` 或 `CEMU_NATIVE_KV_PLACEMENT=balanced`，
启动输出、`qemu.log` 和 `guest-logs/launch-settings.txt` 都记录选择。
每次切换必须关闭旧 VM、重新启动 fresh native NVM session；不支持运行时切换或复用旧映射。

- `legacy`：logical domain 的 token-group bucket 固定到物理 NAND channel，
  该 channel 上各 kind 共用 block 分配计数，可能出现 channel-K 只用部分 channel、
  交错 K/V 各用部分 LUN 的现象。
- `balanced`：domain、暂存页池、满块条件及组页顺序全部不变，只在分配**新物理块**时，
  按 task/layer/kind 的独立游标轮转 channel/LUN/plane，各 kind 的初始位置错开。
  batch 共享同一 layer/kind 游标；lane 内 block 编号由 namespace 统一分配，防止不同
  task/domain 占用同一块。已用尽的 lane 会跳过；当前未满块仍可继续写。
- 一块内的所有页仍在同一 channel/LUN/plane/block，page 连续递增；不拆块、不改
  head/token/channel 页格式。读路径查已保存的 KV→PPA 映射，不再从 bucket 推断物理 channel。
- 新游标和 lane 计数属于有界 FTL 元数据，计入映射内存预算，不增加第二份 KV buffer。
  不改 NAND 延迟、freezing、完成队列和上层提交并发度；同一 LUN 的页操作仍串行建模。
  这是确定性放置策略，不是根据当前忙闲动态迁移，也不保证稀疏选择后能用满所有 LUN。

Host 回归（不启动 Guest Linux/VFIO，不使用 GPU）：

```bash
make -C build -j4
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=tests/cemu/instattention/python \
  CEMU_NATIVE_QTEST_CUDA=0 /usr/bin/python3 -m unittest -v \
  test_native_kv_contracts test_instattention_qemu_io \
  test_instattention_native_qtest test_native_csd_pipeline
```

映射测试覆盖两种策略、跨 task/batch/layer/kind 唯一性、整块连续、所有 lane 分布、
读就绪门槛、耗尽和分配失败。真实 NAND adapter 的受控测试保持 32 pages/kind 和
40 ns/page 的测试参数不变，4 channel × 2 LUN 下全读目标间隔由 legacy 640 ns 变为
balanced 160 ns；这是小型调度回归，**不是 OPT-13B 性能结果**。
QEMU qtest 另验证两种策略的 Native/FDM Attention 数值及小模型 Prefill/Decode。
实际 A/B 请保持同一 buffer 路径、硬件配置、ratio、HPET、I/O backend、batch/prompt/decode
和诊断开关，只切换此放置属性；观察 Flash 页数、每 LUN 页数、读取模型时间和端到端时间，
不能把放置更均匀直接当成论文性能复现或时间校准完成。

### FDM writer、NAND 完成和终止 drain

`hw/femu/fdm_kv_writer.c` 是设备侧运行态，不是另一份 native buffer。
MR0 继续持有 head/staged/hash/domain/writeback 元数据，MR1 持有唯一的暂存页池。
Host 侧只新增线程/条件变量、有界页描述符及内部完成事件封装，不分配第二份 KV 数据。

```text
聚合成功 → 唤醒 writer
  → 同一 domain 队列凑够 block_pages，摘取一块页链
  → 按 batch_pages（最多 256）分批，直接借用 MR1 的页指针
  → bb_native_kv_batch → KV 坐标映射/PPA 分配 → 原 NAND 调度及后端写入
  → 取本块各批最大 ready_ns，投递 internal_complete 到 to_poller/pq
  → 目标时刻到达：摘除该块的页目录项，推进连续 committed，归还页槽
```

worker 不等待每块 NAND 期限，可以继续提交其他 domain；不同物理资源的重叠与同一 LUN
串行约束仍由原 NAND 模型决定。等待中的页仍在 MR1/hash 中，不因后端 memcpy 已完成就
提前回收。内部完成没有 Guest CQE，也不再增加 PCIe 时间。共享 ring 满时，poller 自己
产生的 native IO 完成直接转入其 pq，避免等待自身消费 ring 或丢失完成。

写回首次实际提交时注册 KV 映射：FDM `session_id` 完整映射为 FTL 的 64 位 `task`，
geometry/coordinate 逐字段转换，不把 `FDMKV_Buffer_t` 强转为 `NativeKvBuffer`。
映射沿用现有只追加规则；reset/解绑不删除已写映射。写过的 session 不可从 token 0 重绑
重写，后续新绑定应使用新 session；仅递增 generation 也不能代替映射回收。

终止包复用输入槽/收齐协议：`operation=2`，`wire_bytes=512`，`payload_bytes=0`，
batch/layer/head/token 字段全零，session/generation/request_id 正常校验，其余字节补零。
它不调用 APPEND 聚合模块，而是封口 buffer，把尚未满页的有效数据也排入队列，允许尾块
下放。原 DRAIN WRITE 直到所有目标写回完成、`staged_pages=0` 且 `sealed=1` 才返回成功。
之后不能继续 APPEND。这不是 `fsync` 或 SSD 断电持久化模型，只保证模拟 NAND 写回完成。

停止不依赖虚拟时钟推进；取消不推进 committed，也不回滚已经复制的 NVM 后端/映射。
新增路径不改变 freezing，不给聚合或组块另设合成 runtime，NAND 时间仍取原 FTL 接口。

## Native 页组与块聚合（2026-10-17）

当前 native 写回采用两级组织：**一组一页，多个完整页攒成一个物理块后下放**。
例如 4 KiB 页、FP16、d=128、C=8：token-K/V 每组保存 16 token × 128 维；
channel-K 每组保存 256 token × 8 维，同一 head 的 128 维对应 16 个页组。
旧字段 `channel_group_pages=16` 仅保留 ABI 名称，表示同一 token 跨度的页组数量。

```text
APPEND wire buffer → 直接填 token-K / channel-K / token-V 的 DRAM 页
       → 页满后入队（未满页也在页目录中可读）
       → 同一 batch/layer/layout/NAND-channel 的块队列，可混合不同 head
       → 满 pages_per_block → 分批提交 page program → 原生 to_poller / pq 完成队列
       → poller 到块内最大 deadline 回调 → 发布/释放；writer 继续提交后续块
```

`batch_pages` 仅是内部调用上限（1–256），不是聚合门槛。物理块页数取设备 `pgs_per_blk`；
终端 `BUFFER_DRAIN` 是允许未满页补零、未满块刷出的唯一例外，drain 后不可继续 APPEND。
普通 APPEND 成功仍只保证设备接收，Prefill 完成也不保证全部 KV 已写 NAND。

读取用 `(task/generation, batch, layer, head, kind, token, channel)` 定位页键和页内偏移，
再查 DRAM 页目录或原有 PPA 目录。未满页、等待满块的页及等待写完成的页均可参与 Attention。
原始 K/V ring 和后台 `pack_ready_span()` 已移除；APPEND 在锁内先检查所有 head 的前缀、
reader 及整包所需新页数，容量足够才修改页和 accepted，PENDING 不改变任何 head 或页。
`packed[3]` 保留 STATUS ABI，现均等于 accepted，表示三种布局都已直接填入；`committed`
仍表示 NAND 完成的连续 token 前缀。不同页可能乱序完成，不能仅按 committed 判断位置。
读取统计的 `tail_bytes` 表示 DRAM 页命中字节；`tail_copies` 计实际连续页片段 memcpy 次数。

`BUFFER_STATUS` 保留原有 0–87 字节，在 offset 88 起追加 7 个 uint64：
三种 `packed`、`staged_pages`、`write_blocks`、`partial_blocks`、`ready_blocks`。
其中 staged 包含未满页及目标写入尚未完成的页；blocks 计数在目标完成后增长。
offset 144/152 追加 `inflight_blocks` / `peak_inflight_blocks`，计正在准备提交或已提交
但尚未完成的块及其峰值，不能单凭该计数证明 NAND 同时执行。
`INFO` offset 112/120 为物理块页数/本策略版本 2（直接组页）。

块写回不再在 writer 内逐块等待目标时刻。内部完成事件复用原生 NVMe poller 的
`to_poller` 和优先队列，到期仅执行页回收回调，不产生 Guest CQE、不占 CU、不重复添加
PCIe 预算。事件池有界且计入 buffer 预算；完成前页仍可从 DRAM 读取，完成后才能复用。
不同 channel/LUN 可有多个块在途，同一 LUN 的 program 仍受原 NAND 可用时间约束。
停止或 reset 会取消未完成事件，取消不视为成功写回；freezing 和 NAND 时间模型不变。

为避免页池耗尽却无完整块可写的死锁，注册预留各域残块、各 head 未满页及最大整包追加余量；
预算不足返回 LIMIT。运行时空间不足返回 PENDING，等待已有满块或在途块释放页。
OPT-13B `b4-p1024-d16` 的 Host/Guest 启动脚本均已把 native buffer 预算从 1 GiB 调至 6 GiB；
这是模拟器内存预算，不等同于论文设备 DRAM 容量或已校准的 DRAM 时延。
当前仍保留按 batch/layer/layout/NAND-channel 隔离的分配域；没有新增 GC/擦除回收或 DRAM
拷贝时延模型。整块下放也不能据此宣称已测得 WAF=1。

验证包含跨 head 填满同一物理 block、每次仅提交一页、channel 页跨块且部分落盘、
暂存与 NAND 混合读取、terminal drain、冻结时钟取消及 Attention 数值/读取诊断回归。
写入路径现在为 Guest RAM → wire buffer → 最终布局页 → NVM backend，移除了 raw ring
中转和页写入前的 I/O scratch。同步 transfer 只借用页指针，返回后不保留它；页直到目标
NAND deadline 才复用。读取仍保留 scratch，以保证 backend 失败时不覆盖调用者输出。
channel-K 转置前移到 APPEND 处理内，APPEND CQ 前的软件成本会变化；减少复制层数不等于
已经测得端到端加速，也没有增加新的 DRAM/DMA 时延模型。
旧 QEMU 进程不会自动加载改动，需要使用新构建的二进制启动新 session。

## 先读这两份

- [开发历史与保存基线](../../../docs/cemu/DEVELOPMENT_HISTORY.md)：GPU接入、真实模型、公共流水、
  SparF自治、grouped KV、上层/FTL审查，以及已执行测试的证据边界。
- [FTL工作记录](FTL_WORKLOG.md)：从2026-09-21起持续记录独立KV FTL的设计、修改、测试和风险。
  原生KV独立模式已接入初始化/NVMe/FTL worker/CQ，真实QEMU qtest通过；
  Guest `/dev/ng0n1` 小测试已通过，命令见第12节；内部单页/批量IO入口见第15节。
  有界buffer/后台写回已通过隔离QEMU及用户Guest测试，见第16节；
  native稀疏批量gather内部接口及Host回归见第17节；
  新FTL读取接入原有SparF状态机/CPU算子的Host验证见第18节；
  native FDM/MRS/compute、公共OPT入口及收尾回归见第19节。
  小型合成OPT已通过隔离QEMU全链路数值验证，真实Guest GPU/OPT仍待验收。

地址基础回归：在 `tests/cemu` 执行 `make instattention-native-address-test`。
只编译临时C harness，验证坐标/页偏移与边界，不访问设备，不证明FTL物理放置或回收已经实现。

映射/分配回归：同目录执行 `make instattention-native-mapping-test`。
验证任务隔离、PPA放置、假backend字节读回、目标时间可见性和失败回滚；不访问现有实验镜像，
不代表真实设备IO、NAND时序接入或块回收已经完成。调用契约与放置假设见FTL工作记录第10节。

IO回归：`make instattention-native-io-test`，验证临时文件与目标依赖。
生产适配回归：`make instattention-native-qemu-io-test`，需要Host的QEMU构建配置，
直接链接生产FTL/NAND/backend函数，不启动VM、不操作GPU或实验镜像。
目标时间返回后的完成职责和未接入项见FTL工作记录第11节。

真实QEMU协议回归：`make instattention-native-qtest`，使用临时文件及qtest加速器，
不操作现有VM/GPU/镜像。独立Guest启动：`bash femu-scripts/run-native-kv-test.sh`（CEMU根目录）。
这不是OPT实验启动脚本，不要给旧48/96GiB实验镜像直接加`ftl_mode=instattention`。
新Guest缓存测试：`make instattention-native-buffer-device-test NVM_DEVICE=/dev/ng0n1`，
每台新启动的专用测试VM运行一次；不要挂载/格式化native NVM，也不需要FDM/GPU/模型。

Native Attention桥接回归：在Host的`tests/cemu`执行
`make instattention-native-attention-test`。需要已配置的QEMU build、C编译器、NumPy及
可导入的Host版`_cemu_client`；使用临时文件，直接调用生产native FTL及SparF状态机/CPU算子，
不启动OPT、不使用GPU。完整NVMe/FDM/MRS命令链由下面的qtest单独验证。

## 测试入口与清理范围

当前主线保留以下入口（在 `tests/cemu` 执行）：

| 范围 | Make 入口 | 用途 |
| --- | --- | --- |
| Native 存储与设备链路 | `instattention-native-regression-test` | 协议、FTL、tail、Attention 和小型 OPT 回归 |
| OPT 数学正确性 | `instattention-opt-reference-test` | 与 Hugging Face 对照 Prefill/Decode |
| OPT 真实模型全流程 | `instattention-opt-cemu-gpu-test` | dense/SparF/grouped/native，按参数选择路径 |
| TinyLlama 数值对照 | `instattention-tinyllama-reference-test`、`instattention-tinyllama-decode-reference-test` | 分别验证 Prefill 全层 KV/logits 和 Decode |
| TinyLlama 设备全流程 | `instattention-tinyllama-cemu-gpu-test` | 保留旧模型端到端回归 |
| 公共调度 | `instattention-flexgen-prefill-test`、`instattention-flexgen-decode-test`、`instattention-flexgen-attention-async-test` | runner、microbatch、预取及后台写回依赖 |

已删除早期 `test_tinyllama_checkpoint.py`、`test_tinyllama_prefill.py` 和
`test_flexgen_cemu_pipeline.py`，并移除对应 Make 入口。权重缓存/形状检查保留在
`test_hf_checkpoint.py`；真实权重加载、Prefill 和 CEMU Decode 联通由上述 reference
及全流程测试覆盖，不再维护单独的加载/形状冒烟脚本和合成 FlexGen 设备全流程入口。

第二轮撤掉六个旧设备验收脚本及其 Make 入口：

| 删除脚本 | 后续使用的入口/覆盖范围 |
| --- | --- |
| `test_python_bridge.py` | `test_cemu_device.py` 保留 vadd、Python bridge 和 MRS 生命周期检查 |
| `test_attention.py` | `test_cemu_attention_slot_device.py` 保留 dense 数值与双 slot 设备测试；真实模型由全流程入口验证 |
| `test_cemu_attention_phases.py` | 保留 `test_attention_phases.py` 算子数值测试，不再维护独立双 execute Guest 演示 |
| `test_cemu_attention_workflow.py` | 退休早期 dense 六 MR 工作流验收；当前 SparF/native 依赖时序由 stages、timeline 和 native 回归检查 |
| `test_sparf_group_io_device.py` | grouped IO 细项留在 `test_sparf_group_device.py`；Guest 使用 grouped+Attention 综合入口 |
| `test_sparf_group_buffer_device.py` | tail/背压/发布/环形复用细项留在 `test_sparf_group_buffer.py`；Guest 使用 grouped+Attention 综合入口 |

grouped Guest 综合入口为 `instattention-sparf-group-attention-device-test`，native Guest
仍使用 `instattention-native-buffer-device-test` 和 OPT 全流程。这不是声称新入口逐项复刻
所有旧演示（例如旧 dense 工作流的 serial/overlap 对照）；被退休路径的专用设备验收不再维护。
`require_fdmfs` 移至 `device_test_support.py`，避免保留测试依赖已删除的脚本。
`test_cemu_sparf_workflow.py`、`test_sparf_buffer_device.py` 仍由 `test_sparf_stages.py`
动态调用，保留其 staged/autonomous/旧写回回归，不因名字较老而直接删除。

其余底层单元测试暂不按文件名或日期删除：有些仍覆盖支持中的旧后端，有些提供
native/grouped 测试共用的 fixture。`FUNCTIONAL=1` 只验证流程完成，不能替代数值对照，
各入口需要的模型、GPU、Guest 设备或 Host 构建环境仍按其原有要求准备。

## Native 收尾验证

在Host执行，不能在Guest内用Guest版Python扩展替代Host版：

```bash
cd /home/lihan/cemu/CEMU
make -C build -j4
cd tests/cemu
make instattention INSTATTENTION_BUILD_DIR=/tmp/cemu-instattention-host
make instattention-native-regression-test INSTATTENTION_BUILD_DIR=/tmp/cemu-instattention-host
```

需要Host Python的NumPy、PyTorch、pybind11及开发头文件。回归使用临时NVM文件和qtest
Unix socket，不启动KVM Guest、不操作现有VM/GPU/实验镜像。2026-09-23共100项测试通过。
单独复跑设备计算和小型OPT链路：

```bash
PYTHONPATH=/tmp/cemu-instattention-host make instattention-native-csd-qtest
```

该小型OPT测试复用生产Prefill/Decode runner、OPT数学操作、KV adapter、native协议封包及
SparF提交逻辑；仅把Linux ioctl/FDMFS传输换成qtest NVMe命令，算子在CPU上执行。
两层、双batch、prompt=127、Decode=3、FP32、ratio=1，对照PyTorch的token/logits/hidden，
同时验证实际NVM读写和最终DRAIN。它不证明Guest驱动、GPU、异步重叠或目标时间已校准。
FP16/GQA及ratio=8的稀疏数值和边界仍由native Attention回归覆盖。

下一步先用**独立、新建后端**验收Guest OPT-125M，再扩大OPT-13B；native NVM不能mkfs/mount，
不能复用旧grouped的文件池、cookie、模型实验镜像。配置和限制见FTL工作记录第19节。

## Guest io_uring transport (opt-in)

`CEMU_IO_BACKEND=uring` selects the same SQE128/CQE32 NVMe passthrough mechanism
used by `tests/cemu/cemu_benchmark.cpp`. `ioctl` remains the default for regression
comparison. Set this variable before starting Python; do not change it mid-run.
No QEMU, native FTL, SparF algorithm, target runtime, or freezing changes are required.

The transport covers Execute, the C++ client's FDM reads/writes, native KV commands
(including APPEND), and dense staging's `copy_file_range`. Program download,
activation and MRS management remain ioctl operations, as in the benchmark.
Native KV still reads/writes internally: it does not gain an NVM file or switch
back to Guest `copy_file_range`. Other legacy Python `os.pread/pwrite` paths are
not converted by this switch.

One C++ dispatcher owns a depth-64 ring, submits ready requests, and routes CQEs
by request identity. Concurrent callers share this ring. The blocking adapters keep
their existing buffers alive until CQE completion; the dispatcher borrows those
buffers without an additional staging copy and holds duplicated descriptors.
Short reads/writes and NVMe
status/result are preserved. Negative CQE errors fail the request, whereas
positive NVMe status remains available to native backpressure handling.

This integration **retains the existing Python workers, Future interface and
dependency waits**. Synchronous client methods wait on dispatcher promises; this
is not yet a replacement of all Python workers by a CQE-driven runner. The ring
submission/completion mechanism is aligned with CEMU, not the entire benchmark
Job state machine. Native Prefill now has a bounded APPEND window as described
below; other synchronous methods still wait for each command. Speedup is not guaranteed.

### Native Prefill: multiple APPENDs in flight

With `CEMU_IO_BACKEND=uring`, `CEMU_NATIVE_APPEND_DEPTH` controls the native
Prefill APPEND window (default **8**, allowed **1–64**). Depth 1 uses the original
submit/wait loop. ioctl and one-token Decode retain their serial submission path.
The negotiated 128 KiB limit and token-group-sized packetization are unchanged;
this change does not enlarge commands or modify tail/FTL placement/freezing.

The Prefill writer partitions a layer's K/V into disjoint head ranges within
each batch item. It submits independent ranges without waiting for earlier CQEs,
reaps any completed request, and refills the window. Each head's next token
packet waits for its previous packet to be accepted: NVMe completion order is
not assumed. A backpressured packet is retried unchanged; later tokens for those
heads cannot overtake it. One head range alone therefore remains serial.

The asynchronous C++ request owns a payload copy and a duplicated descriptor
until completion. Wait-any releases the Python GIL and uses completion
notification, not Python polling. On permanent failure or timeout, the writer
drains all outstanding handles before failing the outer Future and poisoning
the store. Success of that Future means every APPEND was accepted, not that
background NAND writes have drained. Existing final drain remains necessary.
Old bindings without the asynchronous API fail explicitly, without ioctl fallback.

Observe `client_append_queue_depth`, `client_append_peak_inflight`,
`client_append_commands`, `client_appended_head_tokens` and
`client_append_backpressure_retries` in the final native statistics.
Peak means submitted handles not yet reaped by the writer, **not measured
simultaneous device execution**. The native command worker is still serial;
multiple in-flight commands remove the Guest round-trip barrier, not that worker's
serialization or NAND resource constraints.

In the Guest, rebuild/test using `instattention-io-transport-test` below. Then use
the existing full inference command with the same measured ratio file:

```bash
ratio=/root/opt13b-b4-p1024-d16.json
test -s "$ratio" && CEMU_IO_BACKEND=uring CEMU_NATIVE_APPEND_DEPTH=8 \
  bash /root/CEMU/femu-scripts/native-opt-guest-test.sh OPT-13B b4-p1024-d16 "$ratio"
```

For the serial uring baseline set `CEMU_NATIVE_APPEND_DEPTH=1` on a **fresh
native session**, with the same inputs, ratio, HPET clock and device configuration.
Check functional PASS, accepted-token/write accounting and append retry counts
before comparing Prefill time. Different background-write timing may change
tail/Flash read counts; do not infer improved simulation accuracy from speedup.

Build inside the CEMU Guest with liburing >= 2.2 development files and headers
providing `NVME_URING_CMD_IO`. CMake prints whether support was compiled. A build
without these dependencies retains ioctl support, but selecting uring fails
explicitly, with no fallback. After installing newer headers, clear the cached
probe with `cmake -U CEMU_HAVE_URING -S instattention -B /root/native-opt-client`.
`uring_copy_file_range` uses CEMU's custom opcode 55, not stock Linux's opcode 55;
only use this path with the matching CEMU Guest kernel.

First validate the transport in the Guest (no GPU/FTL changes):

```bash
source /root/cemu-gpu-env/bin/activate
cd /root/CEMU/tests/cemu
make instattention-io-transport-test INSTATTENTION_BUILD_DIR=/root/native-opt-client
```

The C++ test exercises shared-ring concurrent file I/O, asynchronous request
handles, wait-any, descriptor lifetime, short reads, EOF, and isolation after
rejected NVMe commands. An unsupported kernel/sandbox reports
SKIP, **not a device validation PASS**. Python tests use mocks for native command
packing/status, out-of-order APPEND completions, per-head ordering, backpressure,
failure draining and dense copy offsets; neither replaces a real CEMU run.

In a native Guest with FDM mounted, the same executable optionally checks native
INFO payload/status (including 128 queries in eight-request windows) and compares
ioctl/uring FDM + MRS + CPU vadd Execute results:
`/root/native-opt-client/test_io_transport --cemu /absolute/host/path/to/vadd.so`.
The operator path is resolved by the Host CEMU process, not by the Guest.

Then, in an already booted native Guest, run the existing small inference first:

```bash
CEMU_IO_BACKEND=uring bash /root/CEMU/femu-scripts/native-opt-guest-test.sh OPT-125M
```

Use `OPT-13B` instead if that is the only available model. For the existing
paper-sized profile, keep the same ratio/configuration and prepend
`CEMU_IO_BACKEND=uring` to the usual command:

```bash
ratio=/root/opt13b-b4-p1024-d16.json
test -s "$ratio" && CEMU_IO_BACKEND=uring \
  bash /root/CEMU/femu-scripts/native-opt-guest-test.sh OPT-13B b4-p1024-d16 "$ratio"
```

Check `[opt-cemu] Guest I/O backend=uring` in the log. For A/B use the same command
with `CEMU_IO_BACKEND=ioctl` on a fresh native session. Keep geometry, ratio and
inputs identical; functional PASS and matching tokens do not establish timing
calibration or bitwise numerical equivalence.

## Native Guest OPT-125M

`native-opt-guest-test.sh` 默认在模型加载前将 **Guest** clocksource 切换为 `hpet`，
并检查可用性、切换结果及推理前后的实际值。不可用或切换失败则停止，不静默回退。
日志 `[opt-clock]` 记录使用的时钟源；切换保留到手工更改或重启，下次脚本运行会再次
设置。只影响该测试Guest，不修改宿主机、冻结机制、runtime或原有run-csd启动脚本。
如需旧计时口径的诊断对照，可在Guest运行时设置 `CEMU_GUEST_CLOCKSOURCE=kvm-clock`；
日志会警告这种口径可能包含冻结执行耗时，不能和HPET结果混用。

2026-10-10同一KVM Guest探针确认：目标100ms、实际执行1秒时，kvm-clock应用计时
约1100ms，HPET约100ms；两者QEMU功能冻结区间虚拟时间增量均为0。这只验证时钟
冻结语义，不代表直通GPU、异步流水或目标模型已完成校准。HPET读取开销更高，CUDA
event时间不随Guest clocksource改变。直接运行Make/Python不会自动切换时钟；应先
在Guest中检查/设置 `current_clocksource`，不要在宿主机执行时钟切换命令。

这是实际Linux Guest验收入口，不是qtest。模型计算使用Guest直通GPU，CSD使用CPU SparF，
先验证功能，不注入FPGA runtime/Softmax ratio、不用于论文性能对比。
默认batch=1、prompt=32、Decode=4、top-r=16、ratio=8；FP16 OPT-125M的token组为32个token，
因此会跨token组边界，channel-K尾组由最终DRAIN发布。

Host先正常关闭使用同一基础系统盘/VFIO GPU的旧VM，再执行：

```bash
cd /home/lihan/cemu/CEMU
make -C build -j4 &&
make -C tests/cemu/kernel ../build/sparf_attention.so &&
sudo bash femu-scripts/run-native-opt-test.sh
```

默认直通`0000:27:00.0`，必须事先已绑定vfio-pci；脚本不解绑、不重置其他GPU。
每次在`/data/lihan/cemu-native-opt/opt125m.XXXXXX`创建系统盘overlay、新512MiB NVM文件、
Host日志和Guest日志目录；FDM为64MiB，native buffer预算64MiB。基础系统盘不写入，
源码和模型只读共享。不是旧grouped模式；`run-csd.sh`默认仍是旧FTL，只有显式设置
`CEMU_FTL_MODE=instattention`及对应native参数后才能用其测试native路径。

进入`root@...:/#`之后在Guest执行：

```bash
mountpoint -q /proc || mount -t proc proc /proc
mount -o remount,rw /
mkdir -p /root/CEMU
mountpoint -q /root/CEMU || mount -t 9p -o ro,trans=virtio,version=9p2000.L cemusrc /root/CEMU
bash /root/CEMU/femu-scripts/native-opt-guest-test.sh
```

Guest脚本会挂载sys/dev、模型/日志共享及FDM，使用已有`/root/cemu-gpu-env`，
在Guest系统盘内构建Python扩展，再执行真实OPT入口。需要该环境已有PyTorch、NumPy、
pybind11和模型加载依赖；不自动安装驱动或包。算子在Host编译，QEMU工作目录与Guest的
`./build/sparf_attention.so`引用一致，避免把Guest编译的库错误加载进Host。

扩展构建直接使用pybind11头文件和`Python3_add_library`，不依赖系统pybind11的导出
CMake target；避免某些配置把头文件位置解析成不存在的`/include`。头文件先从指定Python
环境查询，再搜索系统include目录，也可显式设置`CEMU_PYBIND11_INCLUDE_DIR`。
Guest脚本设置`CEMU_INSTATTENTION_BUILD_DIR=/root/native-opt-client`，测试优先加载本次
生成的扩展，而不是只读共享目录里的旧`build-guest`副本。

观察`KV storage=native FTL/token-channel addressing`、Prefill完成、48次Attention、
最终DRAIN及`[opt-functional] PASS`。日志沿用的`grouped`字样在这里包含复用adapter的native
路径，应以storage标识为准。functional PASS仅表示真实模型流程完成，不是逐层reference比较。
不要先运行另一套buffer测试占用native会话；每次完整重测需正常关机、重新启动新会话。

退出：`sync; /sbin/poweroff -f`。日志持久保存在Host本次目录的`qemu.log`和`guest-logs/`。
禁止运行旧`cemu-mount.sh`或对native NVM执行mkfs/fio。

### 正常登录模式与 OPT-13B

125M的真实Guest功能流程已经通过。继续13B时可恢复systemd、账号登录和localhost:2222 SSH，
不需要退回旧FTL。启动脚本接受`OPT-13B`，自动配置64GiB Guest RAM和512MiB native buffer预算；
仍共用512MiB NVM/64MiB FDM的小负载配置，batch=1、prompt=32、Decode=4。

先在当前rescue Guest更新其系统盘内的挂载脚本，再关机：

```bash
cp -a /root/cemu-mount.sh /root/cemu-mount.sh.before-native
cp /root/CEMU/femu-scripts/cemu-mount.sh /root/cemu-mount.sh
sync
/sbin/poweroff -f
```

更新后的脚本根据内核命令行`cemu_native_opt_smoke=1`选择安全分支：挂载源码、模型、日志
和FDM，不格式化native NVM、不运行fio、不创建旧的FDM占位文件。没有这个标记时保持旧行为。
必须更新Guest里的副本，单改Host仓库不会替换`/root/cemu-mount.sh`。

Host使用**刚才Guest实际使用的overlay**，不是原始基础镜像。例如本次125M的overlay为
`/data/lihan/cemu-native-opt/opt125m.CTYarJ/guest.qcow2`：

```bash
cd /home/lihan/cemu/CEMU
sudo env CEMU_NATIVE_BOOT_MODE=normal \
  CEMU_NATIVE_OS_IMAGE=/data/lihan/cemu-native-opt/opt125m.CTYarJ/guest.qcow2 \
  bash femu-scripts/run-native-opt-test.sh OPT-13B
```

normal模式不加`init=/bin/bash`，复用指定测试系统盘，保留其环境；每次仍新建native NVM
后端和日志目录，禁止把旧NVM数据当作有重启恢复能力。不能删除这个测试系统盘或它的backing
基础镜像。若实际overlay路径不同，必须替换为启动日志记录的路径，不要自动选择不明镜像。

按原来的账号密码登录Guest后执行：

```bash
bash /root/cemu-mount.sh
bash /root/CEMU/femu-scripts/native-opt-guest-test.sh OPT-13B
```

这次使用真实13B权重、40层，共160次逻辑Attention；仍是Guest GPU+CSD CPU的功能验收，
不加FPGA时间模型。正常模式测试结束使用`poweroff`。首次normal boot若有历史fstab或其他
自启动任务报错，先保留输出检查，不要用mkfs/fio修复native namespace。

### Native OPT-13B：恢复 B4/P1024/D16 观测负载

`CEMU_NATIVE_PROFILE=b4-p1024-d16`使用现有
`femu-scripts/cemu_config_opt13b_sparf_engines2.json`：96GiB文件后端NVM、
8通道/每通道6个LUN、2GiB FDM、2 CU/2计算线程/2 SparF引擎。
native buffer预算为6GiB、页目录上限2097152；不再使用小测试的512MiB NVM。
本次只匹配之前常用的B4/微批4/P1024/D16负载，不是完整论文输出长度实验。
默认保留Guest GPU和CSD CPU后端；CUDA及诊断开关见下一节。CPU结果不能与旧CUDA后端
结果作仅改变FTL的严格性能归因。

容量预估：6400个stream、1040个KV位置，需要1344000个4KiB页（5505024000字节）；
native不再分配原始K/V环形缓存，按物理分配域预留最终布局页池、元数据及读取工作区，预算保持6GiB；
双slot Attention FDM预估32628736字节。文件后端是稀疏文件，Host实际空闲空间仍需
覆盖至少本次约5.13GiB KV数据和日志，并留余量。每次新建后端，不复用已经封存的会话。

先正常关闭当前Guest，然后在Host执行：

```bash
cd /home/lihan/cemu/CEMU
sudo env CEMU_NATIVE_BOOT_MODE=normal \
  CEMU_NATIVE_OS_IMAGE=/data/lihan/cemu-native-opt/opt125m.CTYarJ/guest.qcow2 \
  CEMU_NATIVE_PROFILE=b4-p1024-d16 \
  bash femu-scripts/run-native-opt-test.sh OPT-13B
```

Guest登录后使用已有、覆盖微批4的ratio文件（例中为之前保存的文件）：

```bash
bash /root/cemu-mount.sh
bash /root/CEMU/femu-scripts/native-opt-guest-test.sh \
  OPT-13B b4-p1024-d16 /root/opt13b-b4-p1024-d16.json
```

脚本在注册native会话/加载权重前检查ratio格式、anchor及全部所需shape；缺失或不覆盖时
直接停止，不会退回线性Softmax模型。需要补测时可在同一个尚未注册的Guest内运行：

```bash
source /root/cemu-gpu-env/bin/activate
cd /root/CEMU/tests/cemu
python3 instattention/experiments/bench_softmax_ratio.py \
  --device cuda:0 --batch-sizes 4 --prompt-length 1024 --decode-steps 16 \
  --compression-ratio 8 --output /root/cemu-logs/native-mb4-p1024-d16-ratio.json
```

之后把新文件路径作为Guest脚本第三个参数；若用新ratio，不应把与旧实验的差异全部归因于FTL。
保持`--functional`并输出profile，而不是不支持native会话复用的`--benchmark`。
预期640次逻辑Attention、每个batch生成17个token（Prefill首个加16次Decode）。
关注PASS、terminal drain、sealed/failed、Prefill/Decode/drain/e2e Guest时间以及各CSD modeled项，
不能相加重叠的worker时间，也不能把Guest时钟直接当作Host真实耗时。
设备配置快照保存在Host运行目录，Guest日志同目录保存ratio与runtime配置快照。

### Native CUDA 对照及等待分解

Host启动器支持`CEMU_NATIVE_CSD_TARGET=cpu|cuda`，默认cpu保持小测试兼容。
cuda要求显式设置一个Host GPU的`CUDA_VISIBLE_DEVICES`（单个编号或UUID），内部
`CEMU_CUDA_DEVICE=0`指可见集合内编号；不能把Guest直通GPU的编号当成CSD CUDA设备。
脚本不解绑、不重置GPU、不停止其他用户进程。`CEMU_CUDA_LIBRARY_PATH`仅对该QEMU进程
补充动态库搜索路径；是否需要575兼容目录取决于Host当前加载的驱动。

Native模式在`CEMU_CUDA_SYNC=1`时要求NVM/FDM后端均完成CUDA初始化；否则QEMU立即
拒绝启动，并输出CUDA状态码及错误字符串，不再等到OPT加载后以Execute status 6失败。
成功时打印`Native namespace 1/2: CUDA backend initialized`。独立Python探针成功不代表
另一个QEMU进程也成功；初始化失败的旧进程需要退出后重新启动。此检查不改变legacy模式、
freezing或runtime模型，也不保证后续GPU kernel不会失败。

加载13B前可在Host先运行已有Native CSD qtest的CUDA变体：

```bash
cd /home/lihan/cemu/CEMU
make -C tests/cemu/kernel ../build/sparf_attention_devptr.so
LD_LIBRARY_PATH="/home/lihan/cemu/nvidia-575-libs${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
CUDA_VISIBLE_DEVICES=1 CEMU_NATIVE_QTEST_CUDA=1 \
make -C tests/cemu instattention-native-csd-qtest PYTHON=/usr/bin/python3
```

要求Host Python已安装NumPy/PyTorch，且其依赖库（如CUPTI）可加载；575目录按本机驱动
选择，不是通用固定依赖。此入口使用64MiB隔离QEMU、临时文件NVM和真实CUDA QK/PV算子，
验证KV尾部/Flash读取、稠密/稀疏结果、复位及小型合成OPT生成对照；不启动Guest、不用
VFIO、不加载真实13B权重。默认不设`CEMU_NATIVE_QTEST_CUDA`仍运行CPU回归。CUDA模式
必须显式选择单张可用Host GPU；通过不代表64GiB/KVM/VFIO完整启动或性能校准已通过。

Guest自动根据启动参数选择`--csd-target cuda`及`sparf_attention_devptr.so`，不能只改Guest
命令却忘记Host的`CEMU_CUDA_SYNC=1`。模型推理仍在Guest GPU；CUDA算子用于精确QK/PV，
SparF选择流程仍由原设备工作流处理，并不声称整个SparF都变成GPU kernel。

以下命令保持上一轮native CPU运行的4个vCPU，只切换后端并打开诊断。
旧`run-csd.sh`使用8个vCPU；若与旧grouped路径对比，应双方统一设置为8。
vCPU数量也可能影响冻结交接，不能把同时修改vCPU的差异全归因于CUDA。

```bash
# Guest先正常poweroff；Host确认1号GPU可用后：
cd /home/lihan/cemu/CEMU
make -C tests/cemu/kernel ../build/sparf_attention_devptr.so
sudo env CEMU_NATIVE_BOOT_MODE=normal \
  CEMU_NATIVE_OS_IMAGE=/data/lihan/cemu-native-opt/opt125m.CTYarJ/guest.qcow2 \
  CEMU_NATIVE_PROFILE=b4-p1024-d16 CEMU_NATIVE_VCPUS=4 \
  CEMU_NATIVE_CSD_TARGET=cuda CUDA_VISIBLE_DEVICES=1 CEMU_CUDA_DEVICE=0 \
  CEMU_CUDA_LIBRARY_PATH=/home/lihan/cemu/nvidia-575-libs \
  CEMU_NATIVE_TIMELINE=1 \
  bash femu-scripts/run-native-opt-test.sh OPT-13B
```

Guest运行同一命令，沿用相同ratio，不重新测量它：

```bash
bash /root/cemu-mount.sh
bash /root/CEMU/femu-scripts/native-opt-guest-test.sh \
  OPT-13B b4-p1024-d16 /root/opt13b-b4-p1024-d16.json
```

Host和Guest启动输出均报告CSD后端；`/root/cemu-logs/launch-settings.txt`保存实际启动设置。
如果独立 CUDA 检查与不带 VFIO 的 QEMU 探针通过，但完整启动在 `cudaSetDevice` 失败，
可在 Host 启动命令加入 `CEMU_DEVICE_INIT_ORDER=femu-first` 做单变量对照。
默认 `vfio-first` 保持原启动顺序；两种模式均保留 VFIO GPU、相同 Guest RAM、模型和设备配置，
只交换 FEMU/VFIO 的 `-device` 顺序。脚本记录实际选择，不自动重试或回退 CPU。
该对照通过仅说明初始化顺序与现象相关，不能单独证明驱动失败根因，也不能替代完整推理验证。
`CEMU_NATIVE_TIMELINE=1`保存每次请求开始/完成及阶段时间戳到共享日志目录的
`device-timeline.jsonl`，默认0关闭。需要 native 页/LUN 诊断时，在 Host 的同一条启动命令中
同时设置 `CEMU_NATIVE_TIMELINE=1 CEMU_SPARF_READ_DIAGNOSTICS=1`；默认不采集详细字段。
运行结束后在Guest离线分析（不用再运行模型）：

```bash
python3 /root/CEMU/tests/cemu/instattention/experiments/analyze_sparf_autonomous.py \
  /root/cemu-logs/device-timeline.jsonl \
  --output /root/cemu-logs/native-requests.json \
  --boundary-summary /root/cemu-logs/native-boundaries.json
```

新增boundary摘要不依赖LUN诊断，分别列出Host/virtual父请求时间、engine排队、
读提交→观察到完成→目标ready、计算功能段、功能结束→调度提交→调度完成→观察到完成。
这些是边界区间，不单独等于冻结开销、CUDA kernel时间或Guest ioctl时间；累加值可能重叠。
诊断开关会增加日志开销，最终性能重复测量应关闭或双方保持相同设置。

### Native 读取收益与页/LUN 排队诊断

重新编译 QEMU 后，用原来的 native 启动命令增加上述两个环境变量；已有 QEMU 进程需要
正常关机、重新启动才能使用新代码。Guest 保持 B4/P1024/D16、同一 ratio、HPET、uring
深度等配置不变，运行原来的完整推理命令。无需增加一次 dense 推理，也不会额外发起读取。
结束后在 Guest 执行：

```bash
python3 /root/CEMU/tests/cemu/instattention/experiments/analyze_sparf_autonomous.py \
  /root/cemu-logs/device-timeline.jsonl \
  --latest-session --min-tokens 1025 --max-tokens 1040 \
  --require-read-diagnostics --require-native-read-reference \
  --read-summary /root/cemu-logs/native-read-summary.json \
  --read-loss-summary /root/cemu-logs/native-read-loss.json \
  --boundary-summary /root/cemu-logs/native-boundaries.json
```

- 三个阶段分别记录 `logical_bytes`（来自 Flash 的请求元素字节，可能含 GQA 重复）、
  `tail_bytes`、`aligned_bytes_before_dedup`、实际 `submitted_bytes` / `pages`。
- `unique_flash_bytes` 对同一物理页内的请求区间再去重，避免 GQA 重复请求把页利用率
  算到 100% 以上。`page_utilization = unique_flash_bytes / submitted_bytes`，其倒数
  为 `page_read_amplification`；没有 Flash 读取时为 null，不假报 0 倍。
- `full_scan` 是在同一阶段 buffer 锁内，根据该层/batch 所有 KV heads 的相同有效长度与
  committed/tail 边界计算的全量页数参照。包括末尾部分页；不查询/读取 backend，不调用
  NAND 调度，不改变发布进度。Channel-K 的全量参照只用于本阶段覆盖率，**不计入 dense K/V**。
- `kv_saved_bytes = dense_kv_flash_bytes - selected_kv_flash_bytes`；
  `net_saved_bytes = kv_saved_bytes - channel_k_flash_bytes`，允许负值。
  这是相同存储状态下的读取量参照，不是另一条 dense 执行轨迹或 dense 性能测量；
  K/V 分别使用各自读取阶段的快照，不能认为三个阶段是同一个原子快照。
- `channel_page_totals` / `lun_page_totals` 显示物理分布；`critical_initial_wait_ns`
  表示最终关键 LUN 首次被本阶段读取时已有的等待，剩余关键排队可能包含本阶段后续
  页及期间的其他任务争用。它不是所有页面等待之和。服务时间来自已有 FTL 结果，不重调度。
- 本诊断不改变页面布局、选择算法、runtime、FTL 时序规则或 freezing；会增加真实采集/日志
  开销，因此用于归因，不直接把诊断运行相对旧运行的 wall 变化当作性能改进。

旧日志缺字段时报告 missing；`--require-native-read-reference` 拒绝不完整的 native 参照。
完整 B4/P1024/D16 预期有 640 个 Attention 请求，需核对分析输出的请求数。

native结束统计提供`client_append_commands`、`client_append_attempts`、
`client_append_backpressure_retries`、`client_appended_head_tokens`和
`client_append_batch_max_bytes`；只维护整数计数，不逐命令打印或增加状态ioctl。
新QEMU通过INFO声明批量APPEND能力，客户端自动启用最多128KiB、同batch/layer多head的
有界提交；旧QEMU声明值为0时保持原4KiB接口。需要重启到新QEMU，单独更新Guest源码
不会改变正在运行的设备。native仍不创建三份Guest KV文件。

B4/GB4/P1024/D16、FP16/40heads/head_dim128、128KiB上限下，成功APPEND预计从
819200+102400=921600次降到30720+2560=33280次；背压尝试数不固定，head-token总量仍为
6656000。完整运行应检查功能PASS、sealed=1、failed=0和完整落盘字节5505024000，
不能把命令减少比例当成端到端加速比例。Prefill按token tile轮转各head，而不是先把一个
head的全部1024token提交完。每包先验证所有head，再统一接收；背压时没有部分提交。
成功表示设备buffer已接收，不表示NAND完成。组页、物理放置、Flash目标延迟和freezing/
runtime未调整；队列到达顺序和后台合批数量可以随批量提交改变。详细ABI和验证见
`FTL_WORKLOG.md`第29节。

## 冻结时钟独立验证

`freeze_clock_probe` 不加载模型、不访问 KV/NVM、不使用 GPU 算子，只申请一个 4KiB
FDM MR。固定目标 runtime=100ms，分别在 Host CPU 算子内部等待 0/200/1000ms，
每组测三次，先预热一次。等待发生在原有功能执行冻结区间，而非 Guest Python 中。
诊断不修改 freezing、runtime 计算或调度规则；请在没有其他 Guest 工作负载时运行。

Host 先编译（不会停止正在运行的 VM，正在运行的 VM 必须正常退出后再启动新二进制）：

```bash
cd /home/lihan/cemu/CEMU
make -C tests/cemu/kernel ../build/freeze_clock_probe.so
make -C build -j4
```

Native 模式有意禁止普通算子 Execute；本测试不放宽其安全限制，而是通过独立诊断脚本
运行原生 FEMU 模式。先正常关闭使用该系统盘的 VM，再启动 KVM 对照组：

```bash
sudo env CEMU_CLOCK_ACCEL=kvm CEMU_CLOCK_CPU=qemu64 \
  bash femu-scripts/run-freeze-clock-test.sh \
  /data/lihan/cemu-native-opt/standalone.PnutcI/guest.qcow2
```

该脚本使用4GiB RAM，不启用VFIO/CUDA、不读取OPT；16MiB NVM和8MiB FDM为全新实例，
系统盘写入独立overlay，不改原镜像；测试期间不得从其他VM写入该backing image。
原有run-csd.sh不变。诊断开关 `CEMU_COMPUTE_CLOCK_TRACE=1` 由新脚本传入，默认普通启动
不启用；QEMU日志产生 `CEMU_CLOCK_TRACE` JSON 行。无需重新编译Guest内核。

默认 CPU 型号为 `qemu64`，让 TCG/KVM 使用相同配置；此前探针使用的 `host` 型号
只能在 KVM 下使用。如需复查旧探针，另跑 `CEMU_CLOCK_ACCEL=kvm CEMU_CLOCK_CPU=host`，
不要把它与 TCG 的差异直接归因于加速器。脚本不自动回退加速器，每个会话保存
`environment.txt`，记录配置和 QEMU/kernel/initrd/设备配置/算子 SHA256。

Guest 登录后只挂载源码、日志与FDM；不要执行格式化NVM的脚本：

```bash
mkdir -p /root/CEMU /root/cemu-logs /mnt/fdm0
mountpoint -q /root/CEMU || mount -t 9p -o ro,trans=virtio,version=9p2000.L cemusrc /root/CEMU
mountpoint -q /root/cemu-logs || mount -t 9p -o trans=virtio,version=9p2000.L logsrc /root/cemu-logs
mountpoint -q /mnt/fdm0 || mount -t fdmfs /dev/nvme0m2 /mnt/fdm0
source /root/cemu-gpu-env/bin/activate
cd /root/CEMU/tests/cemu
make instattention-freeze-clock-test \
  INSTATTENTION_BUILD_DIR=/root/native-opt-client \
  CEMU_IO_BACKEND=ioctl \
  PROGRAM=/home/lihan/cemu/CEMU/tests/cemu/build/freeze_clock_probe.so \
  PROFILE_OUTPUT=/root/cemu-logs/freeze-clock-ioctl.json
```

PROGRAM 是 **Host QEMU 所见路径**，不能换成 Guest 的 /root/CEMU 路径。
JSON 文件必须不存在，避免覆盖旧结果。可以将 backend 改为 uring 并换一个输出文件，
无需重启 VM；这个测试没有创建 native KV session。结果读取和程序/MRS 初始化均不计入
Guest Execute 区间。预热一次加九个测量请求，总计十次 Execute。

新探针在同一个 Execute 外同时读取 Python `perf_counter_ns`、C++ `steady_clock`、
`high_resolution_clock` 和 `system_clock`；C++ 读数来自 Guest 扩展，不是 Host。
JSON 同时记录 Python 实现、C++ 时钟别名、Guest clocksource、内核版本、启动参数，
以及20次空测量区间的中位开销（不自动扣除）。各时钟在相邻位置采样，Python 区间
略大于 C++ 区间；只能比较各自差值，不能跨时钟相减。这验证计时 API，而不是复跑
benchmark 的 io_uring/Load/Compute/Store 全流程。

完成 KVM 组并 `poweroff` 后，在 Host 启动 TCG 组，Guest 重复完全相同的命令：

```bash
cd /home/lihan/cemu/CEMU
sudo env CEMU_CLOCK_ACCEL=tcg CEMU_CLOCK_CPU=qemu64 \
  bash femu-scripts/run-freeze-clock-test.sh \
  /data/lihan/cemu-native-opt/standalone.PnutcI/guest.qcow2
```

TCG 启动和 Guest 编译可能较慢。两组均无 VFIO/GPU，不要同时启动，不改 Guest
clocksource、不改 freezing/runtime，也不要把 TCG 的整体速度与 KVM 直接比性能。
探针使用 `read_range_bytes()` 读取结果，不导入 NumPy；默认 qemu64 不具备部分新
NumPy wheel 要求的 X86_V2 指令集。若曾在旧 `read_range()` 处遇到该错误，更新后
重新运行 make 即可编译 Guest 绑定，不需重启或升级 NumPy。失败运行可能留下空
JSON 文件，重试时换一个输出文件名，不覆盖原日志。
每个会话有独立日志目录，所以输出文件名可以相同。两组运行前应检查源码/二进制
未改变、设备配置哈希相同，并记录两个 session 路径。若只有 KVM 中所有应用时钟随
Host 延迟增长，支持加速器/Guest clocksource 差异；若同一 VM 的 Python/C++ 不一致，
先排查 API/库和采样边界；若两组都增长，不能归因于 KVM，需要再对照历史 QEMU。

Host 取本次启动时打印的 `Clock probe session` 目录，分析共享 Guest JSON 与同会话 QEMU 日志：

```bash
session=/data/lihan/cemu-clock-probe/clock.本次目录
sudo python3 /home/lihan/cemu/CEMU/tests/cemu/instattention/experiments/freeze_clock_probe.py analyze \
  --guest-json "$session/guest-logs/freeze-clock-ioctl.json" \
  --host-log "$session/qemu.log"
```

如果 /root/cemu-logs 未挂载到 logsrc，需先手工复制 JSON，不能拿另一会话的日志分析。
分析器检查随机 probe ID、QEMU PID/controller/program、完整的功能执行和调度完成记录，
缺少诊断开关或事件重复时直接报错。每个时钟只对自己的时间戳求差，不跨时钟相减：

- Guest：Python perf_counter_ns() 的 Execute 前后差。
- Host：功能执行包围区间，以及算子回传的内部等待区间。
- CEMU virtual：冻结内前后差，以及设备收到请求到调度器完成的区间；后者不是 Guest 返回边界。
- 同时列出请求 runtime、扣除 freeze_entry 后的有效预算和调度预算，避免误读零预算。

若 Host 功能耗时增加约1秒，Guest Execute也增加约1秒，而冻结内虚拟差接近零、设备虚拟
区间基本不变，结果为 `guest_clock_includes_functional_freeze`；若 Guest 基本不变，
为 `guest_clock_excludes_functional_freeze`。噪声大或设备虚拟区间明显变化返回
`inconclusive`，不是通过。该诊断不校准 InstAttention 性能，也不证明 GPU/freezing 异步精度。

Host 自检（不需要 Guest，不证明真实 KVM clock 行为）：

```bash
python3 tests/cemu/instattention/python/test_freeze_clock_probe.py
```

## KV 打包优化与旧环境恢复

共享 FDM 打包路径使用 `np.ascontiguousarray(array.transpose(1, 0, 2))` 准备 K/V
连续缓冲区，再由 `bytes.join` 直接读取缓冲区生成 payload，避免旧 NumPy 对非连续数组
调用 `tobytes()` 的慢路径，也不再单独生成两份 K/V `bytes`。包头、坐标、K/V 字节顺序、
请求次数及设备行为不变。Decode 转置结果已连续时，`ascontiguousarray` 不复制数据。

详细计时仍使用 `k_copy_wall`、`v_copy_wall`、`join_wall`：前两项现在是连续布局准备，
最后一项是两个缓冲区合并序列化。比较优化前后应看整个 `kv_payload_pack_wall`，不能
将子项含义视为完全相同；布局元数据包含 `packing=contiguous_buffers_join`。

不必为此迁移 Python/CUDA 或重新下载 PyTorch。若旧 vLLM Python 3.12 环境误升级了
NumPy 2.4.6，先停止 vLLM 工作负载，再在 Guest 执行离线恢复：

```bash
bash /root/CEMU/femu-scripts/repair-vllm-numpy-guest.sh
```

脚本依赖 Host 已下载到 `tests/cemu/build/wheels/` 的 NumPy 2.3.5 cp312 x86_64 wheel，
缺失时拒绝修改环境；该构建目录不纳入 Git。脚本保存依赖快照，仅替换旧 vLLM 环境的
NumPy，再运行 `uv pip check` 和 NumPy/Numba/Torch 导入检查。不修改原 FlexGen 环境，
也不使用未完成的 `/root/cemu-shared-cu129`。

在原 vLLM 环境运行无需设备/模型的 CPU 打包 A/B：

```bash
(
  set -euo pipefail
  source /root/cemu-vllm-env.sh
  cd /root/CEMU/tests/cemu
  python instattention/experiments/bench_kv_payload.py --repeats 7 \
    | tee /root/kv-payload-benchmark.jsonl
)
```

测试对比旧实现与当前生产方法，检查逐字节一致性，覆盖 Prefill/Decode 的单 batch 和
B8 包。只测普通 CPU ndarray，不包含 pinned 内存、D2H、文件 I/O、NAND 或完整推理，
因此不能将这里的加速比直接当作端到端加速比。完整实验继续使用原 Guest 测试入口即可；
无需重编译设备或客户端，正在运行的 Python 进程须退出后重新启动以加载代码。

## FlexGen / vLLM 共用 Guest 环境（可选）

两套完整实验可以通过 `CEMU_GUEST_ENV_FILE` 显式选择同一个环境，未设置时仍使用各自原来的环境。
不要仅在终端激活虚拟环境：原测试脚本会选择自己的 Python。自定义环境文件必须导出
`CEMU_CLIENT_BUILD_DIR`，设置相应 Python 的 `PATH` 和客户端/适配器的 `PYTHONPATH`。

已有 vLLM Python 3.12 环境的 `mistral-common` 要求 NumPy `<2.4`，而 Numba 0.65.0 要求
NumPy `<2.5`。独立共享环境使用已有的 Python 3.13 解释器与 NumPy 2.4.6，固定旧 vLLM
修订版以及已安装的 Torch/torchvision/torchaudio、Transformers、tokenizers、Numba 和 pybind11
版本。其他依赖由解析器按 Python 3.13 重新解析；最终版本保存为 `resolved-freeze.txt`。
安装不会升级或修复旧环境；若旧环境此前被改成不兼容的 NumPy，需单独恢复它。

在已经配置好 vLLM 的 Guest 内执行（需要网络和额外磁盘空间）：

```bash
set -o pipefail
bash /root/CEMU/femu-scripts/setup-shared-opt-guest.sh \
  2>&1 | tee /root/cemu-shared-setup.log
```

所有新内容位于 `/root/cemu-shared-cu129`，包括独立 vLLM 克隆、venv、客户端构建目录、
旧环境版本记录、安装约束和自检报告。脚本在完整依赖检查、NumPy/Numba/FlexGen/vLLM/CEMU
导入、PyTorch 和 vLLM CUDA kernel、pinned D2H 数据一致性及框架 CPU 单元测试通过后，才发布
`env.sh`。失败时不会切换测试入口，可保留日志后重跑安装；已发布的环境拒绝原地重装。
这些检查不加载 OPT-13B、不下发 KV，不代表完整模型验收已经通过。

之后分别在新的 QEMU/NVM 会话中执行，不能因 FDM 已清空就连跑两次完整 B8：

```bash
env CEMU_GUEST_ENV_FILE=/root/cemu-shared-cu129/env.sh \
  CEMU_IO_BACKEND=ioctl CEMU_VLLM_STACK_TRACE=1 \
  bash /root/CEMU/femu-scripts/vllm-opt-guest-test.sh /root/cemu-ab-softmax.json
```

```bash
env CEMU_GUEST_ENV_FILE=/root/cemu-shared-cu129/env.sh \
  CEMU_FLEXGEN_BATCH_SIZE=8 CEMU_FLEXGEN_MICROBATCH_SIZE=8 \
  CEMU_FLEXGEN_DECODE_STEPS=16 CEMU_FLEXGEN_STACK_TRACE=1 \
  CEMU_FLEXGEN_DEVICE_PROFILE=b8-p1024-d16 \
  bash /root/CEMU/femu-scripts/flexgen-b8-guest-test.sh /root/cemu-ab-softmax.json
```

`/root/cemu-ab-softmax.json` 指事先保存在 Guest 系统盘上的同一份 B8/P1024/D16 ratio，须用实际路径替换。
两个入口均使用共享目录内同一个 Release 客户端；KV 包格式、打包代码、设备端和 NAND 配置不变。
复测时保存 `/root/cemu-shared-cu129/environment.json` 与 `resolved-freeze.txt`，并检查每轮日志中的
NumPy 版本。恢复旧入口只需不设置 `CEMU_GUEST_ENV_FILE`，不删除原环境。

## 按模式查接口

| 内容 | 文档 |
| --- | --- |
| Dense分阶段接口 | [ATTENTION_PHASES](ATTENTION_PHASES.md) |
| 旧三文件SparF、ratio与观测 | [SPARF_WORKFLOW](SPARF_WORKFLOW.md) |
| 初版自治v3接口 | [SPARF_AUTONOMOUS](SPARF_AUTONOMOUS.md) |
| 旧buffer和spans/compact写回 | [SPARF_BUFFERING](SPARF_BUFFERING.md)、[SPARF_DEVICE_WRITEBACK](SPARF_DEVICE_WRITEBACK.md) |
| 分组页目录、buffer、Attention | [SPARF_GROUP_DEVICE](SPARF_GROUP_DEVICE.md)、[SPARF_GROUP_BUFFER](SPARF_GROUP_BUFFER.md)、[SPARF_GROUP_ATTENTION](SPARF_GROUP_ATTENTION.md) |
| handoff诊断 | [SPARF_HANDOFF_DIAGNOSTICS](SPARF_HANDOFF_DIAGNOSTICS.md) |
| grouped运行手册 | [RUNBOOK](INSTATTENTION_GROUPED_RUNBOOK_2026-09-16.md) |

这些文档按功能阶段编写，不同模式的MR、文件和并发限制不能混用。
最新跨阶段状态以开发历史的保存基线和FTL工作记录为入口；具体参数以
[Makefile](../Makefile)、脚本help和运行时实际配置为准。
旧进展/交接文档的原路径和Git恢复方法已列入开发历史，避免重新创建重复状态文档。
