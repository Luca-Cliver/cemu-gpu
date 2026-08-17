
# CEMU GPU 集成进展记录

## 背景

当前工作是在 CEMU/FEMU 的 CSD 路径里，把 GPU 作为一个新的计算执行后端接入进去，核心目标是加速 CSD 里的计算执行过程，同时让它和原有的存储/内存路径可以协同工作。

目前已经完成了一个可运行的初步版本，但它仍然是“先让 GPU 跑起来”的过渡实现，还没有把 GPU 和 FDM 抹平到完全统一的内存抽象。

### 为什么要做这件事

CEMU 本身是一个 CSD 模拟器，所以把 GPU 接进来，不只是为了“增加一个新功能”，更重要的是看它能不能加速其中最重的计算过程。

这意味着后续不能只看功能是否跑通，还要看它在计算阶段上的收益是否足够明确。最少需要做定量测量，才能判断这条 GPU 路径到底是在加速计算，还是只是把复杂度从一处搬到另一处。

### 需要量化的指标

后续建议重点统计下面几类数据：

- 单次 CSD 任务的总执行时间
- 计算执行阶段的耗时，也就是 `program_execute()` 到 `run_functional_modeling()` 这一段
- backend 侧 host/device 拷贝次数和字节数
- CUDA 初始化和镜像分配开销
- `vadd_example --cuda-devptr` 这类典型 workload 的端到端延迟
- 同一 workload 在“不开 GPU”和“开 GPU”两种模式下的对比

如果后面要写成正式结论，最好再把这些指标拆成“模拟开销”和“计算执行开销”两部分，这样更容易判断 GPU 是不是确实在加速计算过程。

## 已完成的事情

### 1. 接入 CUDA 后端

在 [hw/femu/backend/backend.c](../hw/femu/backend/backend.c) 里加入了 CUDA 运行时适配层，通过 `dlopen` / `dlsym` 动态加载 `libcudart`，并解析出 `cudaSetDevice`、`cudaMalloc`、`cudaFree`、`cudaMemcpy`、`cudaDeviceSynchronize` 等接口。

同时支持通过环境变量控制 CUDA 行为：

- `CEMU_CUDA_SYNC`：是否启用 CUDA 同步
- `CEMU_CUDA_DEVICE`：选择哪块 GPU

### 2. 从“整块镜像”改成“按范围镜像”

最开始的方案是为整个 backend 一次性申请一大块 GPU 显存，但这个方案在实际运行中容易因为显存申请过大而失败。

现在改成了 `CudaMirrorRange` 的按范围懒分配方式：

- 某个 host 内存范围第一次被访问时，才为它分配对应的 device buffer
- 每个范围单独维护 `host_ptr`、`len` 和 `device_ptr`
- 避免一次性申请整个 backend size

这部分的核心逻辑仍然在 [hw/femu/backend/backend.c](../hw/femu/backend/backend.c) 中。

### 3. 在 CSD 侧把内存范围和 backend 关联起来

在 [hw/femu/csd/compute.h](../hw/femu/csd/compute.h) 中，`MemoryRange` 现在多保存了一个 `backend` 指针，这样同一个 MRS 里的每个 range 都能追溯到它所属的 backend。

在 [hw/femu/csd/compute.c](../hw/femu/csd/compute.c) 里，MRS 创建时会：

- 根据 `nsid` 找到对应 namespace
- 通过 `backend_addr()` 得到 host 侧地址
- 记录该 range 归属的 `backend`
- 为执行阶段准备 `mr_addr`、`mr_len`、`mr_backend`

## 术语与 MRS（Memory Range Set）详解

下面这部分面向没有读过代码的人，解释 MRS、MemoryRange、以及文档中频繁出现的几个名字，便于让外部读者快速理解实现细节。

- MRS（Memory Range Set）: 来自 NVMe Compute 规范的概念。在 CEMU 中，MRS 表示一组“内存范围”（MemoryRange），这些范围是计算程序（CSF/CSF-like）在执行时要访问的内存集合。每个 MRS 在 FEMU/CSD 内部用 `rsid` 标识。

- MemoryRange (`struct MemoryRange`): MRS 中的每一项，包含以下主要字段：
	- `nsid`：地址属于哪个 NVMe namespace（用于在模拟控制器中定位正确的 backend）。
	- `sb`：starting byte，range 在对应 namespace 中的起始字节偏移。
	- `len`：range 长度（字节数）。
	- `addr`：在运行时映射到的 host 侧地址（通过 `backend_addr(namespace->backend, sb)` 得到）。
	- `backend`：指向具体后端对象（`SsdBackend *`），便于在执行或同步时找到后端实现（例如 DRAM backend）。

- `mr_addr` / `mr_len`：在 `program_execute()` 准备 job 时，CSD 会把 MRS 转换成一组 `mr_addr[]`（host 指针数组）和 `mr_len[]`（长度数组），供后续调度和运行使用。

- `mr_backend`：与 `mr_addr` 并行的数组，记录每个 `mr_addr` 对应的 `backend`（因为同一个 MRS 里的范围可以来自不同的 namespace/backend）。

- `mr_dev_addr`：GPU 集成后新增的数组，保存每个 `mr_addr` 对应的设备端指针（device pointer），由 `backend_host_to_device()` 返回（若 backend 支持 CUDA 同步/映射）。

生命周期（高层次）

1. MRS 创建：通过 NVMe 的 MRS 管理命令（或在 program_execute 的数据区直接提供 MR 列表），`memory_range_set_management()` 将每个条目解析成 `MemoryRange`，并记录 `addr`（host）和 `backend`。
2. 程序准备执行：`program_execute()` 根据 MRS 填充 `mr_addr`/`mr_len`/`mr_backend`，并在调度前对需要的 range 做 `backend_cuda_sync_ptr(..., to_device=true)`（把 host 数据拷到 device，若 backend 启用了 CUDA）。
3. 设备指针映射：若后端支持 CUDA，`program_execute()` 会调用 `backend_host_to_device(mr_backend[i], mr_addr[i], mr_len[i])` 获取 device 指针并写入 `job->args.mr_dev_addr[i]`，随后把这些 device 指针传给用户态示例程序 / 共享库。
4. 执行结束与回写：kernel 在 GPU 上运行完后，若需要把结果回写到 host/FDM，CSD 会调用 `backend_cuda_sync_ptr(..., to_device=false)` 从 device 拷回 host，然后由 `memory_write`/`backend_rw_internal` 等把数据写回（或由上层 NVMe path 处理）。

当前实现中 MRS 的位置和 GPU 的关系

- MRS 本身仍然以 host 侧地址 (`mr_addr`) 为中心——也就是说，CSD 的主要内存视角是 FDM/host。GPU 通过 `CudaMirrorRange`（后端内部）作“按范围的镜像”，把特定 host range 映射到 device buffer。这种设计使得：
	- CSD 无需在高层频繁区分 host vs device；只要关心 `mr_addr`，backend 层负责把数据搬到 GPU（或从 GPU 回拷）。
	- 但也因此 GPU 并不是被视为“同级的后端”；它更像是 FDM 的镜像层，负责在需要时为某个 range 分配 device 内存并同步数据。

阅读入口（代码位置）

- MRS 创建与管理：`CEMU/hw/femu/csd/compute.c` 中的 `memory_range_set_management()`（查看创建逻辑）和 MRS 初始化片段。
- 程序执行路径：`CEMU/hw/femu/csd/compute.c` 中的 `program_execute()`，这里会准备 `mr_addr`/`mr_len`/`mr_backend`，并填充 `mr_dev_addr`。
- 后端 GPU 逻辑：`CEMU/hw/femu/backend/backend.c` 中的 `backend_cuda_sync_ptr()`、`backend_host_to_device()` 与 `CudaMirrorRange` 管理代码。

如果你需要，我可以把上面的流程画成一幅 ascii 流程图并放到文档里，或者直接把关键代码片段摘出来以代码注释形式放入文档，帮助不看代码的人快速定位源文件和函数。

### 4. 支持 device pointer 直传给示例程序

已经新增了 `vadd_example.cpp` 的 `--cuda-devptr` 路径，示例会拿到 `mr_dev_addr`，直接把设备端指针传给 CUDA kernel，而不是只依赖 host 侧缓冲再间接同步。

这说明 GPU 设备端指针已经真正接入 CEMU 的执行路径。

### 5. 验证通过并已推送仓库

当前版本已经完成过本地验证，`vadd_example --cuda-devptr` 可以跑通并通过数据校验。

相关代码也已经分别推送到两个仓库：

- `git@github.com:Luca-Cliver/cemu-gpu.git`
- `git@github.com:Luca-Cliver/ubpf-cemu-gpu.git`

## 现在的实现状态

当前实现可以理解成：

- FDM 仍然是主路径
- GPU 目前是挂在 backend 之上的镜像层
- CSD 仍然显式维护 `mr_addr`、`mr_dev_addr`、`mr_backend` 这些并行结构

也就是说，现在已经实现了“GPU 可用”，但还没有实现“GPU 和原有计算设备在 CSD 视角下完全抹平”。

**CSD 视角：GPU 作为计算设备及与存储介质的交互**

目标：从 CSD（Compute Service Daemon）层面，把 GPU 描述为“计算设备”（compute device），明确它与 FDM/namespace/backend（即存储介质）之间的交互语义与边界，便于上层不关心物理驻留位置，只关心“访问哪个 range”。

主要观点：

- 在 CSD 语义中，将 GPU 视为一种可选的执行/驻留位置（与 FDM、host memory 同级的逻辑角色），但实现上可以继续由 backend 维护具体的 device mirror（`CudaMirrorRange`）。
- CSD 应暴露“range级别”的访问意图（read/write/atomic/stream）而不是具体的 host/device 操作；backend 根据策略决定是否在 GPU 上驻留、是否直接写入、是否触发迁移。

交互流程（高层，三条常见路径）：

1. 读-执行-回写（当前常见路径）
	- CSD 在 `program_execute()` 准备 `mr_addr`（host）并请求 backend 把数据 H2D（`backend_cuda_sync_ptr(to_device=true)`），得到 `mr_dev_addr` 并传给 kernel。
	- kernel 在 GPU 上计算完成，CSD 根据需要触发 D2H 回拷（`backend_cuda_sync_ptr(to_device=false)`），或让 backend 保持 device 为主并延后回写。

2. 直接设备写入（Direct Device Write，文档中已列方案 A）
	- 当上层或外部数据流直接来自网络/用户态 buffer，并且后端声明支持 `supports_dev_write`，CSD 在 `memory_write()` 层发出“写入 range，偏好 device”意图。
	- backend 实现 `backend_write_device_direct()`：为 range 分配/确认 device buffer，然后将数据写入 device（不写 host），并更新 range 的驻留状态（device-resident）。
	- 只有在后续需要 host 访问时，backend 才回拷（D2H）或者合并到 host 以保证一致性。

3. 主动迁移/按需访问（自动迁移策略）
	- CSD 只报告访问 intent，backend 根据策略（访问频率、带宽/延迟权衡、内存压力）决定是否迁移到 GPU 或回写到 host。

具体接口与数据流（建议实现）：

- CSD → backend：
  - `backend_cuda_sync_ptr(SsdBackend *b, void *host_ptr, size_t len, bool to_device)`（已有）
  - 新增：`backend_write_device_direct(SsdBackend *b, uint64_t sb, const void *buf, size_t len)`（直接写 device）
  - 新增/扩展：range 状态查询 `backend_range_residency(b, sb)` 返回 HOST/DEVICE/UNKNOWN

- backend → CSD：
  - 返回 device pointer（`mr_dev_addr`）供 kernel 使用
  - 返回统计信息（H2D/D2H 字节数、拷贝次数、alloc 时间）以供监控

一致性与策略要点：

- 强一致性模式：写入后立即保证 host 可读（写入后强制 D2H）。代价是额外延迟，适用于要求同步的控制路径。
- 弱/延迟一致性模式：写入或计算后保留 device-resident，直到 host 明确需要再回拷，有利于长链路仅在设备侧连续处理的场景（例如多个 GPU kernels 链式处理）。
- 策略控制：通过配置/环境变量控制默认行为，例如 `CEMU_CUDA_DIRECT_WRITE`、`CEMU_CUDA_RESIDENCY_POLICY=lazy|eager|pinned`。

并发与错误处理：

- 对 range 的 residency/state 变更必须使用后端锁或原子状态机，避免并发写入和回拷冲突。
- 若 device 写入失败（显存不足、cuda error），应提供可回退路径（落回 host-first 并报告错误），并在日志中记录原因。

可观测性（监控与基准）:

- 在每次 H2D/D2H/alloc/free/直接写入 操作处记录轻量 `perf:` 日志行，便于后续解析（这些点应与定量评估计划结合）。
- 后端应导出累计的字节计数与操作次数，供 `tests/cemu` 的 runner 脚本采集并生成 CSV 报表。

示例：`program_execute()` 的简化顺序（CSD 角度）

1. 解析 MRS，构建 `mr` 列表（每项包含 `addr/len/backend`）。
2. 对每个 `mr` 根据后端能力与策略决定：
	- 若需 device 执行并且后端支持 direct-write 或需要 H2D，则调用相应后端接口（`backend_write_device_direct` 或 `backend_cuda_sync_ptr(to_device=true)`），得到 `mr_dev_addr`。
	- 否则以 host pointer 为主，传给用户空间或共享库。
3. 启动 kernel/用户态程序，等待完成。
4. 根据返回值与策略触发 D2H（立即或按需）。

小结：从 CSD 视角，目标是把“访问哪个 range”与“range 在何处驻留”的决策分离出来，让 backend 负责驻留位置与迁移策略。通过新增的直接写入 API 与 residency 状态，CSD 可以更灵活地支持低延迟的 device-first 路径，同时保留 host-first 回退路径以保证稳健性。

## 目前的局限

1. GPU 还不是一个完全对等的 memory backend。
2. 现在的 host/device 指针映射仍然是显式维护的，不是统一内存句柄。
3. `memory_write()`、`memory_fill()`、`memory_copy()` 仍然先落到 backend 的逻辑空间，再额外同步到 GPU。
4. 当前的 `CudaMirrorRange` 更像同步辅助结构，而不是统一资源管理层。

## 后续要做的事情

### 1. 从 CSD 视角统一内存抽象

目标是让 CSD 不再区分“这是 FDM 还是 GPU”，而是只认一种统一 memory/backend 抽象。

建议方向：

- 让 `MemoryRange` 更像统一内存对象，而不是 host 侧地址加 device 镜像的组合
- 收敛 `mr_addr`、`mr_dev_addr`、`mr_backend` 这些并行数组
- 让 backend 自己决定数据驻留位置和迁移策略

### 2. 让 GPU 和 FDM 在语义上对等

后续希望实现的状态是：

- CSD 层只关心“访问哪个 range”
- backend 层决定这个 range 当前在 host、FDM 还是 GPU
- 访问时自动迁移，而不是上层显式同步

### 直接写入 GPU（跳过 FDM）——设计说明

目标：在某些工作流中允许把数据直接写到 GPU 侧，不先落到 FDM/host 再同步，从而减少冗余的内存拷贝与延迟。

要点（灰白、可执行）：

- 最小可行方案（方案 A，推荐起步）：
	- 在现有写路径增加分支：当目标 `MemoryRange` 的 `backend` 声明支持“直接设备写入”能力时，调用后端新接口 `backend_write_device_direct()`。
	- `backend_write_device_direct()` 的职责：确保对应的 `CudaMirrorRange` 已分配（lazy `cudaMalloc`），并直接把用户缓冲区拷到 device buffer（`cudaMemcpy` 或映射写）。不把数据写入 host/FDM。
	- 仅在后端声明并开启（例如通过环境变量 `CEMU_CUDA_DIRECT_WRITE=1`）时启用该分支，默认保持现有 host-first 行为为回退路径。

- 进阶方案（方案 B，重构向长期目标）：
	- 把 `MemoryRange`/MRS 抽象调整为统一的内存句柄，移除或收敛 `mr_addr`/`mr_dev_addr` 并由 backend 返回当前驻留位置（HOST/DEVICE）。
	- CSD 的读/写请求只指定 range，backend 决定并执行迁移或直接访问（更干净但工作量大）。

实现注意事项：

- 一致性策略：写入后若需要 host 访问，由 backend 决定是否立即 D2H 回拷或延后回拷（按访问模式/策略）。
- 并发/锁：range 分配和状态变更需加锁，避免 race。
- 可配置性：通过 `CEMU_CUDA_DIRECT_WRITE` 控制开关，并保留 `CEMU_CUDA_SYNC` 等现有变量。
- 日志：保留关键错误日志，增加可选的 debug 输出以便排查直接写入失败或回退情形。

API 建议（后端侧新增）

- `bool backend->caps.supports_dev_write`：后端声明能力位。
- `int backend_write_device_direct(SsdBackend *b, uint64_t sb, const void *buf, size_t len)`：在后端实现直接写 device 的调用（返回 0 成功，非 0 错误）。

测试与验证：

- 在 `tests/cemu` 增加小用例（例如 `vadd_example --direct-devwrite`），验证写入后 GPU 上数据正确并且 host 并未被写入直到按需回拷。
- 在基准脚本中增加对比：默认模式（host-first）vs 直接写入模式，测 `submit->compute_start`、H2D/D2H 时间、总延迟与字节数。

风险与回退：

- 若设备端写入导致后续 host 读取出现强一致性问题，应回退到 host-first 路径或在写后强制 D2H。保留回退路径以降低风险。
- 该改动牵涉到后端状态管理与同步，要做好测试覆盖（并发、部分回写、错误路径）。

下一步：实现方案 A 的最小 API（`backend_write_device_direct`）并在 `memory_write()` 中加入开关分支，随后添加 `vadd_example` 的直接写入测试。

### 3. 清理临时代码和调试日志

当前代码里还有不少调试日志，用于验证 CUDA 初始化、range 映射和同步行为。后面需要逐步收敛：

- 减少 `femu_log` / `femu_debug` 噪声
- 保留关键错误日志
- 让运行输出更接近可发布版本

### 4. 扩大测试范围

目前验证重点是 `vadd_example --cuda-devptr`。后面还需要补充：

- 更多 range 数量的场景
- indirect / chunked 执行路径
- 读写混合场景
- 更大工作集下的稳定性

### 5. 继续整理对外文档

等接口稳定后，可以把这次 GPU 集成整理成更正式的文档，内容包括：

- 设计目标
- 当前架构
- 使用方式
- 限制和未来方向

## 当前结论

现在这版已经完成了“GPU 初步接入 CEMU 并可运行”的目标，最重要的里程碑是：

- CUDA 后端可以工作
- CSD 能把设备端指针交给程序
- 示例程序已经跑通并验证通过

但从架构上看，它仍然是一个“镜像式过渡方案”，下一步的重点是把 GPU 和原有 FDM/计算路径真正抽象成同一级后端。

### 6. 定量评估与基准测试（新增）

为验证 GPU 是否能真正加速 CSD 的计算阶段，需要设计并执行一套基准测试。建议包括：

- 目标：衡量 GPU 对计算执行阶段的加速比与引入的额外开销（拷贝、初始化、调度）。
- 基本测量指标：总执行时间、计算阶段耗时（`program_execute()`→`run_functional_modeling()`）、host↔device 拷贝字节数与次数、CUDA 初始化与显存分配时间、CPU 侧调度开销。
- 测试算子（初始集合）：`vadd`（已实现）、`memcpy/copy`、`reduce(sum)`、`crc32/hash`。
- 测试算子（建议加入更大、更接近真实负载的算子）：
	- 密集线性代数：`GEMM`（矩阵乘法），尺寸示例：512x512、2048x2048、4096x4096。GEMM 是衡量 GPU 加速收益的基准之一。
	- 卷积：2D 卷积（例如用于 CNN 的 3x3/5x5 卷积），大图尺寸如 2048x2048；可以测小卷积核对内存带宽与计算强度的影响。
	- FFT：一维/二维快速傅里叶变换（大 N，例如 2^20 或 4096x4096），测计算与内存访问的平衡点。
	- Stencil（模板）计算：3D/2D stencil（如 7-point / 27-point），用于测网格类数值计算的内存与计算平衡。
	- 稀疏计算：SpMV（稀疏矩阵-向量乘），用真实稀疏矩阵集合（例如 SuiteSparse）来评估不规则访问的表现。
	- 图计算：PageRank 或 BFS（大规模图），用于评估随机访问 / 并行度有限场景。
	- 数据并行与聚合：排序（radix sort）、分块聚合（group-by / histogram）、大规模 reduce（sum/max）。
	- 压缩/解压缩：LZ4 / zstd 的流式压缩测试，测 CPU vs GPU 在流式处理上的差异。

 这些算子可以覆盖从高度计算密集到高度内存带宽受限、再到不规则访问三类场景，更能反映 GPU 在真实 CSD 工作负载里的收益。
- 输入规模：为每个算子选择小/中/大三档输入（例如 64KB / 1MB / 64MB），并记录缩放趋势。
- 输入规模建议（示例）：
	- 小：64KB ~ 1MB（测固定开销与短任务）
	- 中：1MB ~ 64MB（测典型任务）
	- 大：64MB ~ 数百 MB / 多 GB（测长任务、显存压力与分块策略）
- 运行模式：对每个配置分别测“不开 GPU（baseline）”与“开 GPU（device pointers / mirror）”。
- 输出报告：对每个算子和输入规模产生一张表格，列出各项指标与加速比；并绘制吞吐量/延迟随输入规模的曲线。

实施建议：

- 在 `tests/cemu` 下写一组小脚本，自动化运行不同参数并收集 FEMU/CSd 日志里的时间戳与计数。
- 在 `run` 脚本中导出必要环境变量（`CEMU_CUDA_SYNC`、`CEMU_CUDA_DEVICE`），并保证运行前后清理显存镜像状态（如果需要）。
- 对日志使用小工具（Python 脚本）解析并生成表格/图表，方便对比。

补充测量与工具建议：

- 热身与重复：每个配置先跑若干次热身（如 3 次），再运行 N 次取平均（N=10）。记录均值与标准差。
- 分解时间：至少记录并报告 `submit->compute_start`、`H2D`（host->device）时间、`kernel` 执行时间、`D2H` 时间、`alloc/free` 时间、总时间。
- 可选工具：若可用，使用 `nsys`/`nvprof` 或 `NVIDIA Nsight` 捕获设备端更详细指标（kernel 时间、SM 利用率、全局内存带宽等）。

实验输出格式建议：CSV 每行为一次试验，列包含：算子、输入大小、模式（CPU/GPU）、run_id、submit_to_start_ns、h2d_ns、kernel_ns、d2h_ns、alloc_ns、total_ns、bytes_h2d、bytes_d2h、notes。

完成这项工作后，我们就能用数据判断下一步的架构取舍（继续做“抹平”还是先优化镜像策略）。

完成这项工作后，我们就能用数据判断下一步的架构取舍（继续做“抹平”还是先优化镜像策略）。