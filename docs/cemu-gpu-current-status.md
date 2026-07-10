# CEMU GPU 集成当前状态说明

## 1. 当前阶段的定位

当前工作是在 CEMU/FEMU 的 CSD 路径中接入 GPU，使 CSD 计算程序除了原有 CPU shared library 和 eBPF 执行方式外，还可以通过 CUDA device pointer 在 GPU 上执行。

这一阶段的定位是“CSD 内部 GPU 加速原型”。它已经把 GPU 接入 CEMU 的计算调度路径，但还没有把 GPU 显存做成和 FDM/NVM 对等的一等 memory backend。

当前宏观数据路径是：

```text
NVM -> FDM/host backend memory -> GPU device mirror
GPU device mirror -> FDM/host backend memory -> NVM
```

也就是说：

- FDM 仍然是 CSD 当前的主内存视角。
- GPU 目前作为 FDM/backend 上的 device mirror 使用。
- GPU 显存和 NVM 之间还没有直接 DMA 或 GPUDirect Storage 式数据通路。
- CUDA kernel 的启动仍然由 CPU-side host code 发起，但这个 host code 位于 CEMU/CSD 执行路径内部，而不是由 guest 应用直接管理 GPU。

从 CSD 思想上看，这符合“由存储设备侧管理计算”的方向：guest 应用提交 CSF 和 MRS，CEMU/CSD 根据 program target 调用 GPU 执行。当前还不是最终形态，最终目标是让 GPU memory 成为 CSD 中更原生的驻留位置。

## 2. GPU 是怎么接进来的

GPU 接入点主要在 backend 层和 CSD compute 层。

### 2.1 CUDA runtime 动态加载

在 `hw/femu/backend/backend.c` 中加入 CUDA runtime 适配层。CEMU 不在编译期强依赖 CUDA runtime，而是在运行时通过 `dlopen` / `dlsym` 加载：

- `cudaSetDevice`
- `cudaMalloc`
- `cudaFree`
- `cudaMemcpy`
- `cudaDeviceSynchronize`

这样 CEMU 可以在启用 GPU 路径时动态使用 CUDA。

相关环境变量：

```bash
CEMU_CUDA_SYNC=1
CEMU_CUDA_DEVICE=0
CEMU_CUDA_SYNC_WAIT=1   # 可选调试项
```

含义：

- `CEMU_CUDA_SYNC`：是否启用 CUDA mirror/sync。devptr 路径需要打开。
- `CEMU_CUDA_DEVICE`：选择使用哪块 GPU，默认通常是 0。
- `CEMU_CUDA_SYNC_WAIT`：每次同步后额外调用 `cudaDeviceSynchronize()`，主要用于调试，性能测试时一般不打开。

`femu-scripts/run-csd.sh` 中已经给 `CEMU_CUDA_SYNC` 和 `CEMU_CUDA_DEVICE` 设置了默认值。

### 2.2 按 range 分配 GPU mirror

GPU 显存不是一次性为整个 backend 分配，而是为具体 host memory range 懒分配。

当前 backend 中维护 `CudaMirrorRange`，每个 mirror range 记录：

```text
host_ptr
len
device_ptr
next
```

当某个 MRS range 第一次用于 CUDA devptr 执行时，backend 为它分配对应 device buffer，并返回 device pointer。

当前新增/使用的关键接口：

```c
void *backend_cuda_ensure_device_ptr(SsdBackend *b, void *host_ptr, uint64_t len);
void backend_cuda_sync_ptr(SsdBackend *b, void *ptr, uint64_t len, bool to_device);
```

其中：

- `backend_cuda_ensure_device_ptr()` 只保证 device buffer 存在，并返回 device pointer，不做 H2D。
- `backend_cuda_sync_ptr(..., true)` 负责 H2D。
- `backend_cuda_sync_ptr(..., false)` 负责 D2H。

这个拆分的意义是把“准备 GPU 指针”和“按语义同步数据”分开。`program_execute()` 不再盲目同步所有 range，而是只准备 `mr_dev_addr`；真正 H2D/D2H 由 CUDA target 执行路径控制。

## 3. GPU 是怎么被识别的

当前通过 `Program.target` 显式区分执行目标，而不是靠 so 文件名猜测。

当前 target 包括：

```c
PROGRAM_TARGET_HOST
PROGRAM_TARGET_CUDA_DEVPTR
```

用户态下载程序时通过不同 helper 设置 target：

```c
prep_shared_library(...)  -> PROGRAM_TARGET_HOST
prep_cuda_lib(...)        -> PROGRAM_TARGET_CUDA_DEVPTR
```

target 传递链路是：

```text
tests/cemu 用户态工具
  -> ioctl_download.target
  -> linux-cemu NVMe/CEMU driver
  -> NvmeLoadProgramCmd.target
  -> CEMU Program.target
```

因此，CUDA devptr shared library 必须用 `prep_cuda_lib()` 下载。否则 CEMU 会把它当作普通 HOST shared lib，`mr_dev_addr` 不会准备，CUDA kernel 会看到空 device pointer。

`vadd_example --cuda-devptr` 已经使用 `prep_cuda_lib()`。

`cemu_benchmark` 当前也加入了显式开关：

```bash
-u
--cuda-devptr
```

用于把 benchmark 里的 shared library 下载为 `PROGRAM_TARGET_CUDA_DEVPTR`。

## 4. CUDA devptr 执行路径与数据流

这一节重点说明：数据如何从 CSD 的 MRS/FDM range 进入 GPU，以及 CUDA shared library 如何使用这些 GPU pointer。

整体链路可以概括为：

```text
用户提交 CSF/MRS
  -> CEMU 解析 MRS，得到 FDM/backend host pointer: mr_addr[]
  -> CUDA_DEVPTR target 为每个 range 准备 GPU mirror: mr_dev_addr[]
  -> 执行前按语义 H2D，同步 input range 到 GPU
  -> CUDA shared library 使用 mr_dev_addr[] launch GPU kernel
  -> GPU kernel 写 device output
  -> 执行后 D2H，把 output range 回写到 FDM/backend host memory
  -> 后续 pread 或 FDM -> NVM copy 读取结果
```

也就是说，用户侧仍然只感知 CSD 的 CSF/MRS 接口；GPU pointer 是 CEMU 在设备模型内部根据 MRS 自动形成的，CUDA kernel 通过 `args->mr_dev_addr[]` 使用这些 pointer。

### 4.1 用户提交的是什么

用户态程序并不直接把 GPU pointer 传给 CEMU。用户提交的是：

- 一个计算程序，例如 `kswitch_proxy_devptr.so` 或 `vadd_cuda_devptr.so`。
- 一个 kernel/function 名字，例如 `kswitch_proxy` 或 `vadd`。
- 一个 MRS（Memory Range Set），描述这次计算要访问哪些 FDM range。
- 一组参数，例如 `cparam1`、`cparam2`。

对 CUDA devptr 程序，用户态下载时必须使用 `PROGRAM_TARGET_CUDA_DEVPTR`。在测试程序中对应 `prep_cuda_lib()`，在 `cemu_benchmark` 中对应 `-u/--cuda-devptr`。

### 4.2 MRS 如何变成 host pointer

在 guest 侧，应用通过 FDMFS 文件或 benchmark 中的 FDM fd 构造 MRS。MRS 中的每个条目包含：

- `nsid`：属于哪个 namespace。
- `sb`：range 在 namespace 中的起始偏移。
- `len`：range 长度。

CEMU 收到执行命令后，在 `program_execute()` 中解析 MRS：

```text
MRS entry (nsid, sb, len)
  -> nvme_find_namespace()
  -> backend_addr(namespace->backend, sb)
  -> host pointer: mr_addr[i]
```

同时 CEMU 会记录每个 range 对应的 backend：

```text
mr_backend[i] = namespace->backend
mr_len[i]     = len
```

到这一步，CSD 已经知道了每个 MRS range 在 host backend memory 中的位置，但还没有真正把数据交给 GPU。

### 4.3 host pointer 如何变成 GPU pointer

对于 `PROGRAM_TARGET_CUDA_DEVPTR`，`program_execute()` 会为每个 MRS range 准备一个 GPU 侧 device pointer：

```text
mr_addr[i]
  -> backend_cuda_ensure_device_ptr(mr_backend[i], mr_addr[i], mr_len[i])
  -> mr_dev_addr[i]
```

这里的 `backend_cuda_ensure_device_ptr()` 只做一件事：确保这个 host range 有一个对应的 CUDA device buffer，并返回 device pointer。

它不会做 H2D。这样设计是为了避免在 `program_execute()` 阶段盲目同步所有 range。哪些 range 是 input、哪些 range 是 output，应该由执行路径根据语义决定。

最后，CEMU 把这些数组填入 job 参数：

```text
args.mr_addr      = host pointer array
args.mr_dev_addr  = device pointer array
args.mr_len       = length array
```

这些参数会传给 CUDA shared library。

### 4.4 数据何时真正进入 GPU

当前 direct CUDA devptr 路径约定：

```text
mr[0] = input
mr[1] = output
```

因此真正的数据同步发生在 `run_cuda_devptr_shared_lib()` 中：

```text
before kernel:
  H2D mr[0]

kernel execution:
  read  mr_dev_addr[0]
  write mr_dev_addr[1]

after kernel:
  temporary D2H mr[1]
```

也就是说，输入数据从 FDM/host memory 到 GPU 的路径是：

```text
FDM range mr[0]
  -> host pointer mr_addr[0]
  -> cudaMemcpy H2D
  -> device pointer mr_dev_addr[0]
```

输出数据从 GPU 回到 FDM/host memory 的路径是：

```text
device pointer mr_dev_addr[1]
  -> cudaMemcpy D2H
  -> host pointer mr_addr[1]
  -> FDM range mr[1]
```

当前 D2H 仍然是立即执行的临时策略，目的是先保证 `pread()` 或 FDM -> NVM copy 能读到正确结果。后续会改成 range dirty tracking 后的 lazy D2H。

### 4.5 CUDA shared library 如何使用这些 pointer

CUDA shared library 中的函数接收统一的 `struct cemu_args *args`。对于 devptr kernel，重点使用：

```c
args->mr_dev_addr[i]
args->mr_addr[i]
args->mr_len[i]
args->cparam1
args->cparam2
```

以 `kswitch_proxy_devptr.cu` 为例，direct path 使用：

```c
device_input  = args->mr_dev_addr[0];
device_output = args->mr_dev_addr[1];
host_input    = args->mr_addr[0];
host_output   = args->mr_addr[1];
```

CUDA shared library 里的 CPU-side host wrapper 负责 launch kernel：

```text
CEMU/CSD compute thread
  -> shared library function
     -> CUDA kernel launch
        -> GPU execution
```

这和普通 CUDA 程序一样，GPU kernel 仍然需要 CPU-side host code 发起。区别在于，普通 CUDA 程序由应用自己 launch kernel；当前实现中，launch 发生在 CEMU/CSD 设备模型内部。

### 4.6 执行函数的组织

当前 `hw/femu/csd/compute.c` 中已经把执行逻辑按 target 拆开：

```text
run_functional_modeling()
  -> run_program_by_target()
       -> run_host_program()
       -> run_cuda_devptr_shared_lib()
```

其中：

- `run_functional_modeling()`：负责计时、进入/退出 compute section、统计 runtime。
- `run_program_by_target()`：根据 `Program.target` 分发。
- `run_host_program()`：执行 CPU shared lib 或 eBPF。
- `run_cuda_devptr_shared_lib()`：执行 CUDA devptr shared lib，并负责当前阶段的 H2D/D2H 策略。

indirect CUDA path 暂时不做新的 input/output 角色优化，仍保持更保守的同步策略，后续单独梳理。

## 5. 用户如何使用

### 5.1 启动 CEMU

如果使用 `femu-scripts/run-csd.sh`，默认会设置：

```bash
CEMU_CUDA_SYNC=1
CEMU_CUDA_DEVICE=0
```

如果手动启动，需要确保：

```bash
export CEMU_CUDA_SYNC=1
export CEMU_CUDA_DEVICE=0
```

### 5.2 vadd 示例

`vadd_example` 当前 direct ABI 已统一为：

```text
mr[0] = input,  8192 bytes
mr[1] = output, 4096 bytes
```

原因是 `vadd` 输入为交错数组：

```text
input = [a0, b0, a1, b1, ...]
output[i] = input[2*i] + input[2*i+1]
```

运行 CUDA devptr 版本：

```bash
CEMU_CUDA_SYNC=1 ./build/vadd_example --cuda-devptr
```

### 5.3 cemu_benchmark + kswitch_proxy_devptr

运行 devptr shared library 时需要加 `-u`：

```bash
CEMU_CUDA_SYNC=1 ./build/cemu_benchmark \
  -u \
  -v \
  -l ./build/kswitch_proxy_devptr.so \
  -n kswitch_proxy \
  -e 1.0 \
  -o 1 \
  -p 1 \
  -c 16 \
  -s 1 \
  -d 0
```

如果不加 `-u`，benchmark 会用 `prep_shared_library()`，程序会被下载成 HOST target，CUDA devptr kernel 中的 `args->mr_dev_addr` 会为空。

## 6. 当前已经实现了什么

当前已经完成：

- CUDA runtime 动态加载。
- 通过环境变量启用/选择 CUDA device。
- backend 按 range 懒分配 CUDA mirror。
- CSD 的 `MemoryRange` 记录所属 `SsdBackend`。
- `Program.target` 区分 HOST 和 CUDA_DEVPTR。
- Linux ioctl / NVMe load command 传递 target。
- `program_execute()` 准备 `mr_dev_addr`。
- `run_functional_modeling()` 按 target 拆分执行路径。
- direct CUDA devptr path 采用 `mr[0]=input, mr[1]=output`。
- `vadd_example --cuda-devptr` 可以通过数据校验。
- `cemu_benchmark` 增加 CUDA devptr target 开关。

## 7. 当前还没有实现什么

当前还没有实现：

- GPU 显存和 NVM 之间的直接数据传输。
- GPUDirect Storage 式 NVM <-> GPU VRAM 通路。
- GPU memory 作为 CSD 一等 memory backend。
- range 级完整 dirty/valid 状态机。
- lazy D2H。
- indirect CUDA path 的 input/output/scratch/global range 角色定义。

现在 GPU 仍然是 FDM/backend 的 mirror：

```text
FDM host range
  <-> CUDA device mirror
```

而不是：

```text
NVM <-> GPU VRAM
```

## 8. 和 CSD 思想的关系

当前实现符合 CSD 的第一阶段思想：计算由 CEMU/CSD 路径管理，guest 应用只提交 CSF/MRS，不直接管理 GPU。GPU 被作为 CSD 内部的 compute accelerator 使用。

但它还不是最终的 CSD-GPU memory 架构。当前数据仍然需要经过 FDM/host memory，再同步到 GPU。这意味着它更准确地说是：

```text
CSD-managed GPU acceleration prototype
```

后续目标是把 GPU 从 backend mirror 提升为更原生的 CSD memory/compute resource。

## 9. 下一步计划

建议后续按以下顺序推进：

1. 去掉 CUDA shared library 中默认 D2H，只保留 debug-only D2H。
2. 在 `CudaMirrorRange` 中加入 range 级状态，例如 `device_valid`、`device_dirty`、`host_dirty`。
3. CUDA kernel 写 output 后只标记 device dirty，不立即 D2H。
4. 在 `memory_read()`、FDM -> NVM copy 前统一检查 dirty range，并按需 D2H。
5. direct path 稳定后，再梳理 indirect path 的局部 MRS layout。
6. 最后再考虑 `GPU_MEM backend`，让 GPU memory 从 mirror 变成一等 resource。
