# SparF 单请求设备自治：实验路径 v3

## 当前范围

本路径实现行为对齐计划的第三步 A：一次 Guest execute，由设备保留上下文并推进内部依赖。
它不替换默认路径，不迁移 Guest KV 写回，不修改 SparF 数学、FTL 参数或 runtime 校准。

当前限制：每个计算 namespace 最多一个自治请求，配置要求 `nr_cu=1`，只支持注册的 11-range MRS。
第二个并发自治请求返回忙，不会隐式排队或退回旧路径。多请求共享引擎和论文双内核调度尚未实现。
不能将这个模式当成 Figure 14 的最终性能路径。

## 执行过程

```text
Guest CemuSparfDevice(autonomous=True)
  → 发送 version=3 metadata，一次 execute
  → program_execute：验证、固定 program/MRS、建立父请求
  → 有界设备控制线程：
       选通道、提交 channel-K 读取
       等物理数据返回、等 FTL 目标完成（不占 CU）
       粗分数/Top-k → 内部计算任务完成
       提交 selected-K 读取，等物理/目标完成（不占 CU）
       提交 selected-V 读取，同时执行精确 QK/Softmax
       等 QK 内部计算任务完成，以及 V 的物理/目标完成
       PV/补偿 → 内部计算任务完成
  → 释放上下文与引用，向 Guest 投递唯一完成
```

实现复用 `sparf_workflow.h` 的原有计算代码，在读取和计算边界保存/恢复状态，没有另写一份 SparF。
scratch 数组和读取 ring 在一次请求内复用，不在每次恢复时重新分配。
控制线程按 namespace 延迟创建并复用，不为每层 Attention 反复创建线程。
一个请求同一时刻最多持有一个 Flash 读取请求，因此此版本使用请求私有 completion ring，不假设不同请求按全局顺序返回。

父请求不进入计算 CU 队列。只有粗分数、精确 QK/Softmax 和 PV 三类计算预算进入现有调度器。
内部计算完成只通知设备控制线程，不发送 Guest CQE，不释放父请求的 MR/program。
有效预算为零时不插入占用 CU 的虚假计算任务。

设备控制线程在冻结区间外等待物理返回、目标时刻和内部计算完成；等待使用短暂 sleep，不忙等冻结时钟。
功能分段仍使用现有 `run_functional_modeling()`。`enter_compute_section()`、`leave_compute_section()` 和 wrapper 的冻结扣除逻辑未修改。
分段数量与旧路径不同，因此不承诺新旧 freezing 次数、真实运行时间或最终模拟时间完全相等。

## 数据和生命周期约束

- metadata 已由现有命令入口复制到 Host；数组/extent 不依赖 Guest 后续修改。
- program 的运行引用保持到整个父请求结束，MRS 的 `in_use` 固定到最终完成。
- 自治请求期间不能删除或复用该 MRS；应用也必须保持 FDM 文件和 NVM extent 不变。
- 当前不提供热拔插、VM reset 或命令取消恢复协议；不要通过删除文件/重置设备来取消在途测试。
- 正常计算/读取失败会排空本请求尚未返回的读取，释放缓冲，再返回失败，不发布成功输出。
- Guest buffer 的追加、后台写回、publish 和均值维护保持原有职责；此处没有实现计划第四步。

## 时间与日志边界

目标 Flash 读取完成时间来自现有 FTL；PV 同时受 QK 计算完成和 V 读取完成约束。
父请求不再额外收费整条 workflow runtime。内部计算沿用 wrapper 处理后的有效预算。

MR10 的 `total_model_ns` 继续表示原有单请求阶段合成值，用于兼容与检查，**不是自治路径的端到端完成时延**：它不包含计算资源排队、控制推进和全部应用开销。
当前旧 `CEMU_SPARF_TIMELINE` 的阶段日志不用于 v3：它持有单次栈上 observation，不能跨异步读取保存引用。
v3 改用独立的 `CEMU_SPARF_AUTONOMOUS_TIMELINE`，采集与分析方式见下文；不要混用两个分析器。
`CEMU_COMPUTE_LOG` 的计数是功能分段调用数，不是 Guest attend 次数。

## 已完成的 Host 验证

- QEMU 编译通过；未启动/重启 VM，未运行共享 GPU 实验。
- 生产 C workflow 的确定性假 FTL 测试：v1/v2 回归，v3 FP16/FP32 数值，忙 NAND 下的目标依赖，buffer 前缀/尾部组合。
- 注入 channel-K、selected-K、selected-V 读取失败以及 QK/PV 失败，检查清理。
- Guest 多步检查通过模拟 transport 调用同一 C workflow，对照旧路径、staged 和自治输出。
- v3 时间线分析器覆盖正常依赖、零计算预算、缺失/重复记录、失败请求、CU 重叠、提前执行与截断记录。
- 确定性 C harness 使用生产 workflow 的采集点及生产 JSON 序列化器，与离线分析器联测。harness 中的 CU 调度是假调度，不替代真实 Guest 检查。

这些不是对真实 QEMU 调度器、Guest 驱动和 CUDA 后端的完整集成证明；下面的 Guest 测试仍需执行。

## 小测试命令

Host 的 QEMU 已在本轮重新编译。运行中的 VM 不会自动使用新二进制，需要在方便时正常退出并用原启动脚本重启。
保持 `internal_bandwidth=0`、`nr_cu=1`；不要改 freezing。算子 ABI 没变，可使用已有 SparF 算子库。

Guest：

```bash
source /root/cemu-gpu-env/bin/activate
cd /root/CEMU/tests/cemu
mkdir -p /root/cemu-logs
set -o pipefail
make instattention-sparf-autonomous-device-test \
  NVM_TEST_DIR=/mnt/nvme0 \
  FDM_TEST_DIR=/mnt/fdm0 \
  CSD_TARGET=cuda \
  PROGRAM=./build/sparf_attention_devptr.so \
  TOKEN_COUNT=17 \
  DECODE_STEPS=4 \
  2>&1 | tee /root/cemu-logs/sparf-autonomous-small.log
```

`PROGRAM` 是 QEMU Host 工作目录可解析的算子路径，沿用原来已验证的路径；显式设置它避免 Guest 再编译 CUDA 算子。
预期首先打印 `mode=device-autonomous-v3`，然后检查单请求、旧 staged 回归、多步追加和多个模式的 NumPy 对照。
这里无需加载 OPT-13B，也不会启动其长生成测试。

如需先测 CPU，仅将 `CSD_TARGET=cpu`、`PROGRAM=./build/sparf_attention.so`。
收到忙/无效字段时先核对新 QEMU 是否生效、CU 数量和在途请求，不要放宽数值容差。

## 可选 v3 阶段时间线

默认关闭。开启后按父请求记录 Host 单调时钟与 QEMU 虚拟时钟，两种时钟不相减、不混算。
控制线程记录读取提交、观察到物理数据可用、满足 FTL 目标时刻；调度器记录三段计算的有效预算、提交、完成和实际模拟 CU 占用区间。
阶段记录先保存在内存，只在父请求接收和结束时输出两行 JSON，不在冻结计算段内写日志。
每个计算阶段最多保存 4096 个 CU 区间，超出后标记截断，分析器拒绝给出完整验证通过。

### 采集

在 **Host** 正常退出旧 VM 后，用新 QEMU 和原配置启动；不要为这个测试修改 freezing。
启动脚本已经增加该变量的 sudo 环境传递，保留原来的 GPU 选择和驱动库环境。

```bash
cd /home/lihan/cemu/CEMU/build
mkdir -p /home/lihan/cemu/cemu-logs
trace=/home/lihan/cemu/cemu-logs/sparf-autonomous-$(date +%Y%m%d-%H%M%S).jsonl
printf 'Host timeline: %s\n' "$trace"
CEMU_SPARF_AUTONOMOUS_TIMELINE="$trace" ./run-csd.sh
```

随后在 **Guest** 重跑上面的 `instattention-sparf-autonomous-device-test` 小测试即可，无需加载 OPT 权重。
JSONL 保存在 Host，不是 `/root/cemu-logs`；普通终端日志和旧版时间线不能代替它。
只采集 v3 父请求，测试中的 v1/v2 对照不会混入。

### 离线检查

小测试执行完毕后，在 Host 另一终端分析打印出的实际路径：

```bash
python3 /home/lihan/cemu/CEMU/tests/cemu/instattention/experiments/analyze_sparf_autonomous.py \
  /home/lihan/cemu/cemu-logs/sparf-autonomous-实际时间戳.jsonl \
  --output /home/lihan/cemu/cemu-logs/sparf-autonomous-summary.json
```

检查内容：

- channel-K 就绪后才能粗筛，粗筛完成后才能提交 selected-K。
- selected-K 就绪后才能执行 QK 和提交 selected-V。
- PV 开始不早于 `max(QK 计算完成并被控制线程观察到, V 物理与目标时间均就绪)`。
- 本请求的 CU 计算区间不与自身纯 IO 等待区间重叠，同一 CU 的已采集 v3 请求不能重复占用。
- 开始/完成必须配对；失败、缺失或截断记录不能当作 PASS。

输出 JSON 包含每个请求的 `host_elapsed_ns`、`virtual_elapsed_ns`、`cu_occupied_ns`、
`qk_v_target_overlap_ns` 和 `zero_budget_phases`。

### 解释边界

- `admitted` 是父请求分配和 MR 准备之后的采集起点，`ready` 是设备结束处理、输出诊断日志和投递 Guest CQ 之前的时刻；不是整个 Guest execute 时延。
- 读取的 `observed` 是控制线程观察到物理数据可用的时刻，**不是 FTL 物理完成瞬间**。尤其 V 可能早已返回，但控制线程等完 QK 后才观察它。
- `functional_start/end` 包括对应功能函数及其内部搬运，不是 CUDA event 测得的纯 kernel 时间。
- CU 区间来自现有目标调度器；`qk_v_target_overlap_ns` 是 QK 目标 CU 占用与 V 的提交至 FTL 目标截止区间交集，不证明 Host GPU 和真实 SSD 的物理重叠。
- 有效预算为零会明确记录为未提交 CU；零重叠也可能是有效结果，不能靠强制正值判定流水正确。
- 这里只核验采集到的 v3 请求；不包含旧路径任务的完整 CU 审计。PASS 不代表目标时间已经校准或论文性能已经复现。
- 诊断仍有时间戳、内存记录和文件输出开销；性能实验不设置该环境变量。开关在 QEMU 进程内只初始化一次，修改当前 shell 不会改变正在运行的 VM。

OPT 入口现已提供可选 `SPARF_AUTONOMOUS=1`，并能组合设备写回；默认后端不变。
应用共享 slot 池通过单提交 worker 遵守上述单在途父请求限制，不代表已实现设备多请求流水。
先验证联合小测试，再运行小规模 OPT；命令与限制见 `SPARF_DEVICE_WRITEBACK.md` 的“OPT 应用接入”。
