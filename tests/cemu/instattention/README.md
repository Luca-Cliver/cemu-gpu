# InstAttention / CEMU 文档导航

当前目标是复现论文行为再评估模拟精度，不以吞吐吻合或一次functional PASS代替行为验收。

当前 native 串行路径的函数入口、Prefill/Decode 数据流、完成条件和计时边界见
[串行执行与数据流](NATIVE_SERIAL_DATAFLOW.md)，用于后续分阶段埋点和论文行为对照。
调用、系统调用、冻结区间及等待边界的图示见
[串行时序图与时间标记](NATIVE_SERIAL_SEQUENCE.md)（含 Prefill、单层 Decode、父 Execute 内部）。
物理硬件阶段、作者 NVMeVirt 模拟的已知边界、当前 CEMU 实现的逐项对应见
[物理阶段对应表](PHYSICAL_STAGE_MAPPING.md)。

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
