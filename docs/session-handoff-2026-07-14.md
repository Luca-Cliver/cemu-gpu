# CEMU GPU-CSD 研究会话迁移记录

更新日期：2026-07-14

## 1. 文档用途

本文档用于将当前关于 CEMU、GPU 集成、性能量化和 KV cache/LLM 场景的讨论迁移到后续会话。文档区分以下三类内容：

- 已实现：当前仓库中已经存在的代码和工具。
- 已验证：已经获得的运行结果。
- 研究计划：讨论后形成的方向，但尚未实现。

重要约束：用户已明确要求后续不要在当前环境中编译项目。除非用户再次明确授权，只做代码阅读、静态检查、文档修改和必要的 Git 操作。

## 2. 工作区和仓库

主要仓库：

```text
/home/lihan/cemu/cemu-gpu
```

相关仓库：

```text
/home/lihan/cemu/linux-cemu-gpu
/home/lihan/cemu/ubpf-cemu-gpu
```

相关论文：

```text
tests/KiF_Accelerating_Low-Batch_LLM_Inference_Using_In-Flash_KV_Cache.pdf
tests/InstAttention_In-Storage_Attention_Offloading_for_Cost-Effective_Long-Context_LLM_Inference.pdf
```

此前完整可见会话记录：

```text
/home/lihan/cemu/.codex-old-session.md
```

## 3. 当前研究定位

当前工作的核心不是提出一个已经制造出来的新型 GPU-CSD，而是扩展 CEMU，使其能够快速且准确地模拟 GPU 集成计算存储设备。

两个目标必须分开评价：

1. 模拟得快：降低模拟器为了执行功能模型而花费的宿主机真实时间。
2. 模拟得准：让 Guest 看到的设备延迟、数据通路和执行结果接近目标 GPU-CSD。

GPU 在当前工作中首先是 CSD 功能模型的加速执行后端。物理 GPU 运行 CUDA functional model，并不意味着被模拟设备的运行时间等于物理 CUDA 执行时间。

关键时间定义：

```text
realtime = QEMU/CEMU 在物理宿主机上真正执行功能模型所花的时间
runtime  = CEMU 时间模型向 Guest 呈现的目标设备运行时间
```

理想目标是：

```text
realtime 尽可能小，提升模拟速度
runtime 经过校准，接近目标设备时间
```

## 4. 已实现的 GPU 执行路径

### 4.1 Program target

当前已经加入：

```c
PROGRAM_TARGET_HOST
PROGRAM_TARGET_CUDA_DEVPTR
```

target 已经贯通以下链路：

```text
Guest 用户态程序
  -> ioctl/NVMe program download
  -> Linux CEMU/NVMe driver
  -> CEMU Program.target
  -> HOST 或 CUDA_DEVPTR 执行分支
```

用户态 helper：

```text
prep_shared_library() -> PROGRAM_TARGET_HOST
prep_cuda_lib()       -> PROGRAM_TARGET_CUDA_DEVPTR
```

`cemu_benchmark` 使用 `-u` 或 `--cuda-devptr` 选择 CUDA device-pointer target。

### 4.2 MRS 和 CUDA pointer

CEMU 解析 Memory Range Set 后形成：

```text
mr_addr[]     host/FDM backend pointer
mr_len[]      range length
mr_backend[]  range 所属 backend
mr_dev_addr[] range 对应的 CUDA device pointer
```

CUDA mirror 采用按 range 懒分配方式，而不是为整个 backend 一次性分配显存。

当前主要接口：

```c
backend_cuda_ensure_device_ptr(...)
backend_cuda_sync_ptr(..., true)   // H2D
backend_cuda_sync_ptr(..., false)  // D2H
```

### 4.3 当前 direct CUDA ABI

当前 direct CUDA 路径仍然硬编码为：

```text
mr[0] = input，执行前 H2D
mr[1] = output，执行后 D2H
```

这对 `vadd` 和 `kswitch_proxy` 可以工作，但不能直接承载多输入、读写 KV cache 等复杂 workload。后续需要增加 range role，例如 input、output、read-only、read-write、scratch 和 resident。

### 4.4 当前数据路径

```text
NVM -> FDM/backend host memory -> CUDA device mirror
CUDA device mirror -> FDM/backend host memory -> NVM
```

GPU 显存目前是 FDM range 的 mirror，不是和 NVM/FDM 对等的一等 memory backend。

## 5. 已完成的时间测量

### 5.1 Job E2E 标记

`tests/cemu/cemu_benchmark.cpp` 已加入：

```text
JOB_E2E_START
JOB_E2E_END
```

宿主机通过 SSH 读取 Guest 输出，并用宿主机时钟给每行加时间戳，从而测量测试者实际等待的 wall-clock job E2E。

一个 `iter` 表示当前 benchmark 中一次完整的：

```text
NVM -> FDM input
-> compute
-> FDM output read/copy
```

### 5.2 CEMU backend compute 时间

`hw/femu/csd/compute.c` 输出：

```text
CEMU_COMPUTE: program N, realtime=... ns, runtime=... ns
```

这条日志来自物理宿主机上的 QEMU/FEMU 进程，不来自 Guest benchmark。因此 Guest 内执行 `tee` 只能获得 `Input time`、`Compute time` 等 Guest 输出，不能捕获 `CEMU_COMPUTE`。

### 5.3 测量脚本

`tests/cemu/run_job_e2e.sh` 支持：

```text
--native
--gpu
--log FILE
--compute-log FILE
```

`--compute-log` 必须指向宿主机 QEMU stdout log，不能与 `--log` 指向同一个文件；脚本会覆盖 `--log`。

典型 QEMU log：

```text
~/cemu/CEMU/build/log
```

典型使用方式：

```bash
cd ~/cemu/cemu-gpu/tests/cemu
./run_job_e2e.sh --native --log e2e_native.log \
  --compute-log ~/cemu/CEMU/build/log

./run_job_e2e.sh --gpu --log e2e_gpu.log \
  --compute-log ~/cemu/CEMU/build/log
```

QEMU GPU 模式启动参考：

```bash
cd ~/cemu/CEMU/build
CEMU_CUDA_SYNC=1 CEMU_CUDA_DEVICE=0 ./run-csd.sh
```

## 6. 已验证的性能结果

在 16 MiB、单 job、单并行 chunk 的 `kswitch_proxy` 测试中，排除 `iter=0` 冷启动：

| 路径 | 样本数 | 平均 E2E | P95 | 估算吞吐 |
|---|---:|---:|---:|---:|
| GPU functional model | 13 | 28.59 ms | 30.28 ms | 559.7 MiB/s |
| 原生 CEMU CPU functional model | 31 | 1783.53 ms | 1787.42 ms | 约 9.0 MiB/s |

结果：

```text
simulation speedup = 62.38x
wall-clock latency reduction = 98.4%
```

冷启动：

```text
GPU iter=0    473.55 ms
CPU iter=0   1861.18 ms
```

必须准确表述：该结果证明 GPU 加速了 CEMU functional modeling，不代表目标 GPU-CSD 相比 CPU-CSD 的设备性能提高了 62 倍。

## 7. Guest 内部计时为何容易误解

`cemu_benchmark` 原有的 `Input time`、`Compute time`、`Output time` 使用 Guest 时钟。CEMU 存在冻结/调度时间机制，Guest 时钟不一定覆盖物理宿主机上等待 functional model 的全部时间。

此前观察到：

```text
Guest Compute time 约 10 ms
CEMU backend realtime 可达约 1.76 s
CEMU runtime 约 10 ms
```

因此 Guest `Compute time` 接近 `runtime`，而非 backend `realtime`。测试者体感上的真实等待必须使用宿主机时间戳测量。

当前不再使用 `JOB_STAGE_*` 推导每阶段 wall-clock，因为异步状态切换曾把 compute 等待错误归入下一阶段 input。可靠口径是：

- 宿主机 `JOB_E2E_START/END`：真实 job E2E。
- QEMU `CEMU_COMPUTE realtime`：真实 functional compute 时间。
- Guest Input/Compute/Output：Guest-visible 阶段时间，只用于理解模拟时间，不等同于宿主机 wall-clock breakdown。

## 8. CEMU 的 NVM、FDM 和 Compute 路径

CEMU 属于近存储计算/计算存储的全系统模拟。Guest 运行完整 OS、应用和驱动，CEMU 模拟 NVM、FDM 和 Compute namespace。

NVM 到 FDM 的逻辑路径：

```text
Guest copy_file_range(nvm_fd, fdm_fd)
-> Guest VFS/FDMFS
-> 修改后的 NVMe driver 生成 Memory Copy 命令
-> CEMU Memory Namespace
-> FTL/NAND latency model
-> NVM backend 数据复制到 FDM backend
```

Memory Copy 命令包含 NVM LBA、长度和 FDM destination address。Guest 只提交描述符，数据不需要先进入 Guest 普通内存。

功能实现中，`backend_rw_internal()` 最终通过宿主机 `memmove()` 搬运 NVM/FDM backend 数据；设备时间主要由 `ssd_read()` 等模型产生。

当前注意事项：

- 默认没有完整建模独立的 NAND->FDM 内部带宽节流。
- `CEMU_THROTTLE_INTERNAL` 当前处于关闭状态。
- CUDA H2D 在物理实现上可能经过宿主机 PCIe，但在模拟语义中代表 FDM->CSD GPU 内部互连。
- 不能直接把宿主机 `memmove()` 或 `cudaMemcpy()` wall-clock 当作目标 CSD 内部通路时间。

后续准确 timing model 至少应区分：

```text
Tdevice = TNAND-read
        + TNAND-to-FDM
        + TFDM-to-GPU
        + TGPU-compute
        + TGPU-to-FDM
        + Tqueue/contention
```

## 9. KiF 论文结论

论文：`KiF: Accelerating Low-Batch LLM Inference Using In-Flash KV Cache`。

应用场景：资源受限边缘设备上的低 batch LLM 推理，KV cache 因容量不足溢出到 SSD，并位于 decode critical path。

主要设计：

- 历史 KV 存在 NAND pSLC。
- Q x K^T 和 attention weight x V 在 flash chip 内的专用 GEMV 电路执行。
- Softmax 和暂存 KV 由外部 xPU 处理。
- 新 KV 先在 xPU memory 中聚合到 flash page 大小，再回写。
- KV 按 attention head 分散到多个 flash chip。

论文使用 FlexGen + NVMeVirt，测试 OPT-2.7B、OPT-6.7B、OPT-13B、FP16、batch 最大 20。增加了：

```text
kif_prefill
kif_decode
kif_logit
kif_attend
```

KiF 对本项目的重要提醒：即使消除了外部 PCIe KV 传输，controller-side CSD 仍可能受 flash channel/internal data movement 限制。

## 10. InstAttention 论文结论

论文：`InstAttention: In-Storage Attention Offloading for Cost-Effective Long-Context LLM Inference`。

它比 KiF 更接近当前 CEMU，因为计算位于 SSD controller/CSD，而不是 NAND die 内。

任务划分：

```text
GPU：QKV projection、RoPE、O projection、FFN、prefill
CSD：保存 KV，执行 decode attention
Host：请求管理、调度和传输控制
```

CSD attention 包括：

```text
Logit: Q x K^T
Softmax
Attend: attention weight x V
```

实现：

- DaisyPlus OpenSSD。
- Xilinx MPSoC/FPGA + ARM + 2 GiB CSD DRAM。
- FPGA attention engine，ARM 运行 FTL。
- GPU-CSD P2P DMA。
- 自定义 `config()`、`attend()`、`reclaim()` 命令。
- KV 地址编码 batch、layer、token、head、channel。
- SparF 稀疏 attention 和 flash-aware 双阶段过滤。

软件栈基于 FlexGen，将 `TorchDisk` 扩展为有计算能力的 `TorchDevice`。测试 OPT-13B、OPT-30B、Llama-2-13B，数据集包括 ShareGPT、WikiText-2、SQuAD 和 TriviaQA。

与本项目最相关的是它的模拟校准方法：先在真实 OpenSSD 上测组件，再把延迟写入 NVMeVirt。论文报告 GeMV、Softmax 和 Filter 的虚拟模型与真实硬件吞吐约有 94%--97% 的一致性。

## 11. 论文对本项目贡献边界的影响

KiF 和 InstAttention 已经覆盖：

- SSD-resident KV cache。
- decode attention 下沉。
- GPU/CSD 异构 decoder。
- 自定义 NVMe 命令和 KV layout。
- 真实硬件校准模拟器。

因此不能把“把 KV attention 放到 CSD”本身作为主要新贡献。

当前更合理的核心贡献是：

> 在全系统 CEMU 中加入通用 GPU functional modeling 能力，将功能执行与设备 timing model 解耦，从而更快地模拟复杂 GPU-CSD workload，并为准确校准 NVM、FDM、GPU 计算和内部数据移动提供平台。

与 InstAttention 的弱 FPGA CSD 不同，目标 CSD 内部计算设备是 GPU，因此卸载边界不必固定在 decode attention。未来可以研究：

```text
attention-only offload
vs.
full decoder-layer offload
```

但不建议一开始直接实现完整 decoder layer，因为会同时引入模型权重、MLP、框架和 residency 问题，难以定位故障和测量来源。

## 12. 当前建议的第一项 workload

第一步迁移 `llama2.c` 的单 token dense decode attention，不迁移整个模型，也暂时不迁移 FlexGen/InstAttention。

上游地址：

```text
https://github.com/karpathy/llama2.c
https://github.com/karpathy/llama2.c/blob/master/run.c
```

提取 `transformer()` 中的 attention 逻辑：

```text
Q x K_cache^T
-> scale
-> softmax
-> attention weight x V_cache
-> context
```

第一版配置建议：

```text
dtype = FP32
batch = 1
num_heads = 1
head_dim = 64 或 128
sequence_length = 128 到 8192
```

第一版 MRS ABI：

```text
mr[0] = Q             input
mr[1] = K cache       read-only
mr[2] = V cache       read-only
mr[3] = Context       output
```

Guest driver 参考 `tests/cemu/vadd_example.cpp`，建议新建：

```text
tests/cemu/kv_attention_example.cpp
```

CSD CPU shared library 建议新建：

```text
tests/cemu/kernel/kv_attention.cc
```

第一阶段先将 Q/K/V/Context 放入 FDM，建立 CPU reference，并验证 CEMU CPU shared library 输出。暂时不做 KV append、NVM staging、SparF、FlexGen 或完整模型。

## 13. 建议的实施顺序

### 阶段 A：CPU correctness

1. 从 `llama2.c/run.c` 提取 dense decode attention。
2. 编写独立 CPU reference。
3. 编写 CEMU CPU shared-library 算子。
4. 编写 Guest driver，创建四个 FDM range 并提交 compute。
5. 比较 context 输出，记录最大绝对/相对误差。

### 阶段 B：GPU functional model

1. 保持相同的四 MR ABI。
2. 将 attention 迁移为 CUDA。
3. 将当前固定 `mr[0]` H2D、`mr[1]` D2H 改成 range role 驱动。
4. 只将 Q/K/V 同步为 input/read-only，将 Context 回写为 output。
5. 比较 CPU/CUDA correctness、backend realtime 和 job wall-clock E2E。

### 阶段 C：真实 KV 数据路径

1. 将历史 KV 主副本放入 NVM namespace。
2. 增加 NVM->FDM staging。
3. 建模 FDM->GPU 内部互连。
4. 引入 page/block、flash channel bandwidth、queue 和 contention。
5. 研究分块 attention、online softmax 和双缓冲。

### 阶段 D：卸载边界研究

比较：

```text
InstAttention 风格：只卸载 decode attention
GPU-CSD 风格：卸载完整 decoder layer
```

完整 layer 包括 RMSNorm、QKV projection、RoPE、KV append、attention、Wo、residual、第二次 RMSNorm、SwiGLU MLP 和 residual。

## 14. 后续评估指标

模拟速度：

- Host wall-clock job E2E。
- `CEMU_COMPUTE realtime`。
- 每秒可模拟的 token/job。
- CPU functional model 与 GPU functional model 加速比。
- 冷启动与稳态结果分开报告。

模拟准确性：

- CPU reference 与 CUDA 输出误差。
- `runtime` 与真实硬件、分析模型或论文参数的误差。
- NVM read、NVM->FDM、FDM->GPU、kernel、回传的 latency breakdown。
- 不同 sequence length、head 数、batch 下的趋势误差。
- 队列、并发和 flash channel contention。

应用级指标：

- Per-token decode latency。
- Tokens/s。
- KV cache bytes read/written。
- 外部 PCIe bytes。
- 内部 flash/FDM/GPU traffic。
- Attention-only 与 full-layer offload 的性能和资源权衡。

## 15. 当前开放问题和常见陷阱

1. 当前 CUDA direct path 只支持固定 input/output 角色，必须先解决多 MR 同步语义。
2. `cuda_mirror_valid` 仍是 backend 级状态，复杂多 range workload 需要 range-level valid/dirty/residency。
3. 不能用 Guest `Compute time` 代替物理 wall-clock functional time。
4. 不能用物理宿主机 CUDA `realtime` 直接代表目标 GPU-CSD `runtime`。
5. NVM->FDM 的宿主机 `memmove()` 只完成 functional copy，不代表真实内部带宽。
6. 当前物理 GPU 在语义上是 CSD 内部 GPU，不是 InstAttention 的外部 InstGPU。
7. CEMU 是全系统模拟器，Guest 对应完整 Host system，而不是单独一块 GPU。
8. 第一阶段结果应称为 simulation speedup，避免称为 GPU-CSD device speedup。
9. 正式实验必须确保两条路径使用相同数据量、chunk、并行度、迭代数，并验证 checksum/数值正确性。
10. `iter=0` 包含冷启动，应单独报告或作为 warmup 排除。

## 16. Seminar 暂定材料

题目候选：

```text
面向 GPU 集成计算存储设备的快速功能模拟方法研究
```

短摘要：

> 计算存储通过在存储设备内部处理数据减少主机与设备间的数据搬移，GPU 集成为复杂计算存储功能提供了新的架构空间。现有 CEMU 主要依赖宿主机 CPU 执行功能模型，难以高效支持计算密集型算子及 GPU 集成架构的模拟。本报告设计并实现一种面向 CEMU 的 GPU 加速功能模拟机制，打通用户程序、NVMe 命令、CSD 计算模块与 GPU 设备内存之间的数据路径，为后续开展快速、准确的 GPU-CSD 模拟及 KV cache attention 等场景研究提供基础。

## 17. 最近相关提交

当前仓库最近与本工作直接相关的提交包括：

```text
922ae9571 测试脚本
50f638d10 Support QEMU compute log parsing
445d3513a Fix compute timing awk counter
baa6892c5 Use backend compute timing in E2E script
f7eab9e6b Add per-stage job E2E timing
6704a681c 测试脚本
e03b46fb2 Add job E2E markers to cemu benchmark
0cd1c978b Disable test kernel D2H copies
f35fbe132 Document CUDA devptr integration status
f4d923ce5 Clean up CUDA devptr direct path
aa1c3ecf0 Split compute execution by target
7f185c671 Add CUDA devptr compute target
```

开始下一次会话时，优先阅读：

```text
docs/session-handoff-2026-07-14.md
docs/cemu-gpu-current-status.md
tests/cemu/run_job_e2e.sh
hw/femu/csd/compute.c
hw/femu/backend/backend.c
```
