# InstAttention / CEMU 文档导航

当前目标是复现论文行为再评估模拟精度，不以吞吐吻合或一次functional PASS代替行为验收。

当前 native 串行路径的函数入口、Prefill/Decode 数据流、完成条件和计时边界见
[串行执行与数据流](NATIVE_SERIAL_DATAFLOW.md)，用于后续分阶段埋点和论文行为对照。
调用、系统调用、冻结区间及等待边界的图示见
[串行时序图与时间标记](NATIVE_SERIAL_SEQUENCE.md)（含 Prefill、单层 Decode、父 Execute 内部）。
物理硬件阶段、作者 NVMeVirt 模拟的已知边界、当前 CEMU 实现的逐项对应见
[物理阶段对应表](PHYSICAL_STAGE_MAPPING.md)。

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

## Native Guest OPT-125M

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
native buffer预算为1GiB、页目录上限2097152；不再使用小测试的512MiB NVM。
本次只匹配之前常用的B4/微批4/P1024/D16负载，不是完整论文输出长度实验。
默认保留Guest GPU和CSD CPU后端；CUDA及诊断开关见下一节。CPU结果不能与旧CUDA后端
结果作仅改变FTL的严格性能归因。

容量预估：6400个stream、1040个KV位置，需要1344000个4KiB页（5505024000字节）；
native环形K/V缓存本体891289600字节，另有少量元数据及批写scratch，预算1GiB；
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
`CEMU_NATIVE_TIMELINE=1`保存每次请求开始/完成及阶段时间戳到共享日志目录的
`device-timeline.jsonl`，默认0关闭。这里不启用LUN级read diagnostics，因为native回调
尚未提供完整的LUN分布观察量；不能把缺少的字段当成0或让它通过旧LUN诊断验证。
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
