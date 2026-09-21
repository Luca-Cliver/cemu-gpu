# InstAttention / CEMU 近期开发交接

> 2026-09-16 更新：以下为 09-15 的历史交接。现已增加显式 grouped OPT backend 接入，
> 并修复 FDMFS 零容量统计误判、完整 MR 预算和共享 slot 回归覆盖。
> 页池/stream 上限也已扩展，不能再用本文旧上限判断当前实现。
> 最新命令与剩余边界见 [09-16 runbook](INSTATTENTION_GROUPED_RUNBOOK_2026-09-16.md)。
> 真实 grouped OPT Guest 长流程仍需执行验证，不能由 Host 单测推断通过。

## 1. 目标、当前结论和约束

目标是在 CEMU 上复现 InstAttention 的应用分工、SparF 数据访问、设备调度和存储行为，再比较模拟精度。不是调整参数直到吞吐曲线相同；行为成立后暴露出的差异才有分析意义。

本文按最近开发批次梳理，不把所有未提交 diff 都认定为某一天所写。工作区包含此前实现及未跟踪文件；接手先检查 git status，不要清理或覆盖这些文件。旧设计文档的阶段编号有不同含义，以下以实际功能为准。

**当前最准确的状态：新分组页池 + 设备有界尾部 + 后台整组写回 + 自主 SparF Attention 已通过小规模设备接入测试；真实 OPT Prefill/Decode 还没有切换到这套新存储。**

- 不修改 VM freezing、时钟补偿和原 FTL 目标完成时刻等待。
- 不绕过真实 NVM/FDM/FTL 来获得更快数字。
- 保留公共 Prefill/Decode 调度，不复制两份完整 OPT/TinyLlama runner。
- 新模式保留旧路径用于回归和 A/B，不静默替换默认语义。
- 不自动提交、推送、重启 QEMU、重绑 GPU 或重载宿主机驱动。
- 最新 Guest PASS 来自用户反馈；本次未获得该轮完整 JSONL，不代表已经验证实际重叠比例或论文时间模型。

## 2. 此前已有的基础，不要误认为需要重写

已实现真实 TinyLlama/OPT checkpoint 加载、模型数学路径、公共 Prefill/Decode runner、Guest GPU 与 CSD Attention 分工，以及应用侧 microbatch/slot 调度。OPT-13B 在旧存储路径上已经跑过小规模数值对照和长序列功能实验。

SparF 已有近似选择、精确 Attention 的 CPU/CUDA 算子及时间模型接口。软最大时间曾从 Table I 锚点线性外推，导致严重放大；后来加入本机不同形状 Softmax 实测比值乘论文锚点的方式。这个比值仍是跨硬件外推假设，不是已经验证的 FPGA 时间真值。空的 SOFTMAX_RATIO_FILE 曾导致回退到 legacy 模型，后续必须保存实际启动日志和所用文件。

这些旧路径的 PASS 不能自动证明新的 grouped 路径已经完成 OPT 集成。

## 3. 近期写了什么

### 3.1 自主 SparF 调度与观测

从 Guest 分阶段调用，增加设备 v3 自主父任务：Guest 一次逻辑 Attention，设备推进选择、channel-K 读、近似计算、选中 K/V 读、QK/PV。

- 计算阶段使用原计算调度；I/O 与目标时间等待不占据整个功能计算冻结窗口。
- V 读取先提交，再进行精确 QK；PV 依赖 QK 和 V 两者完成。
- 父任务保存 program/MRS 引用，内部阶段完成不能提前返回 Guest 最终完成。
- 保留原单 namespace 单在途 Attention 父任务限制及 nr_cu=1 约束。两个 slot 不等于两个父任务同时计算。
- 增加 trace 和离线分析，区分 Host 真实区间、目标时间依赖及应用等待。

主要入口：

- `hw/femu/csd/compute.c`：自主任务接收、完成、引用管理及内部 I/O。
- `hw/femu/csd/sparf_workflow.h`：阶段推进。
- `hw/femu/csd/sched.c`、`sched.h`：计算资源调度。
- `hw/femu/csd/sparf_autonomous_trace.h`、`sparf_autonomous_log.h`、`sparf_timeline.h`。
- `tests/cemu/instattention/experiments/analyze_sparf_autonomous.py`、`analyze_sparf_timeline.py`、`analyze_sparf_handoffs.py`。

历史 trace 曾通过 9 请求的 IO/CU 排他和 QK/V→PV 依赖检查。这是旧轮次结果，不是最新 grouped 模型流水的测量。

### 3.2 第一版设备写回：spans / compact

之前 Guest 从 FDM buffer 取回 CPU 后 pwrite；随后增加设备执行写回的路径，以及 compact 几何描述以减少 Guest 逐片描述。

- 设备可以完成内部 NVM/FTL 写入，减少 Guest 数据往返。
- 但旧三文件布局仍有细碎更新、读改写与描述处理，并未因为“下放”就自动变成高效整页写。
- 用户 A/B/C 小实验曾出现设备版显著更慢，促使后续改为新页池、整组发布，而不是删除模拟等待。
- 老路径保留，供行为和性能对照。

入口：`hw/femu/csd/sparf_writeback.h`、`include/hw/femu/sparf_writeback_abi.h`、`python/cemu_flexgen/sparf_device_writeback.py`、`sparf_write_buffer.py`。

### 3.3 G2：设备分组页目录

增加设备拥有的逻辑页目录和独占 NVM 页池。Guest 注册池 extent，设备按 `(stream, layout, group)` 管理 token-K、channel-K、token-V 三份布局。

- 满组写到新页，设备做 channel-K 转置；不再为每个小更新读取旧数据页。
- 选 token / channel 时设备查目录，整页读取、去重，再按请求索引顺序提取。
- 这减少的是应用层旧页读改写；不能宣称 NAND 无 GC、写放大为 1。
- 这是逻辑布局，物理 PPA 仍交给 FEMU FTL；尚非论文跨 head 拼块或指定 channel 放置的完整复现。

入口：

- `include/hw/femu/sparf_group_abi.h`：命令和结果 ABI。
- `hw/femu/csd/sparf_group.h`：页池、目录、PUT/READ。
- `python/cemu_flexgen/sparf_group_layout.py`：布局参考与打包。
- `python/cemu_flexgen/sparf_group_device.py`：Guest 适配。

Guest 已提供 PASS：stream 0/5，token-K/V 和全部 channel-K 正确，稀疏页去重/顺序正确；132 次写、786432 B 写入、843776 B 读取、PUT old-page reads=0。此阶段单独不包含尾部缓存或 Attention。

### 3.4 G3：设备尾部缓存与后台整组写回

新增 APPEND，不再要求 Guest 凑齐并打包完整组。原始 K/V 到设备 FDM 尾部，设备自己积累并后台写回。

- 每个 stream 分别维护 accepted、token-K committed、channel-K committed、token-V committed。
- 有界 K/V 环形尾部；K 在两种 K 布局都提交后才可回收，避免覆盖。
- 一个后台线程轮流服务 stream/layout；写入和等待目标时刻期间不持有尾部锁。
- 满组真实写入并达到目标完成条件后才发布目录，不能提前声明提交。
- 未发布数据从 FDM 尾部读取，已发布数据从 NVM 读取；混合读保持逻辑顺序。
- 空间不足 APPEND 背压；错误 poison 会话并唤醒等待者。
- DRAIN 写完最后不足组的数据并 sealed，属于终结操作，之后禁止 APPEND。
- MRS 跨命令固定，后台退出后才释放；不是后台引用已经关闭的 Guest 请求。

存储 MRS：MR0 输入/输出，MR1 状态，MR2 实际 FDM 尾部与一组写回 scratch。

入口：`hw/femu/csd/sparf_group_buffer.h`、`sparf_group.h`、`compute.c` 中 `sparf_group_transfer`，以及 `SparfGroupedDevice.append/progress/drain/close`。

Guest 已提供 PASS：2 streams、531 tokens/stream，15/16 与 255/256 边界、环形复用、混合读取和终结部分组正确。FDM tail+scratch=344064 B，后台整页写入=950272 B；前 15 tokens 从尾部读取，零 NVM 读取。

### 3.5 G4：将新存储接入已有自主 Attention（最新通过）

这是本轮最后新增部分，不是又写一套 Attention 数学实现。

- workflow v3 增加 GROUPED 标志；112 B header 后附 16 B cookie/stream_base。
- 新模式不再每次传旧三文件 extent 表；首次绑定缓存页池映射。
- 设备直接从新目录与 尾部构造 channel-K、选中 token-K/V 的读取计划。
- 复用旧 `sparf_start_read / sparf_finish_read` 和 CPU/CUDA 算子，保留 V 读取与 QK 的依赖结构。
- 当前 token 也先 APPEND 到新存储；不再从旧 MR5/MR6 拼入当前 K/V。
- 仍由上层提交 Q 和 value_mean；mean 尚未变成设备维护。
- Attention 绑定会话/stream 读取引用；相同 stream 不可同时 APPEND，不能在途 DRAIN/CLOSE。其他 stream 和后台写回仍可推进。
- 正常和错误出口均释放引用，避免失败后永久阻止关闭或追加。

入口：

- `hw/femu/csd/sparf_group_attention.h`：`sparf_group_bind/unbind`、`sparf_start_grouped_read`。
- `hw/femu/csd/sparf_workflow.h`：grouped 分支及自主阶段衔接。
- `hw/femu/csd/compute.c`：metadata 校验、会话接入和完成释放。
- `include/hw/femu/sparf_workflow_abi.h`、`python/cemu_flexgen/sparf_workflow_abi.py`。
- `python/cemu_flexgen/cemu_sparf_device.py`：`grouped_store`、`grouped_stream_base` 参数。

新路径需要独立存储 3-MR 工作区与原 Attention 11-MR 工作区。MR5/6 暂保留容量但不使用其 current K/V；不能把 slot 数当成存储池数量。

## 4. 当前数据流与未接入位置

```text
已经实现并测试的设备小流程：
Guest 原始 K/V → APPEND → 设备 FDM 尾部
                              └→ 后台满组打包 → NVM/FTL → 发布目录
Guest Q + mean + cookie/base → 一次自主 Attention
  → Top-r → channel-K（Flash + 尾部）→ 近似分数/Top-k
  → 选中 token-K（Flash + 尾部）
  → 提交选中 V 读，同时执行精确 QK
  → 两者完成后 PV → 输出
结束 → 等待所有在途任务 → DRAIN → CLOSE

尚待接入：
真实 OPT 公共 Prefill/Decode runner
  → 新 grouped backend 的写入、mean、stream 映射及生命周期
  → 上述已经测试的设备流程
```

关键核查：`python/test_opt_cemu_pipeline.py` 当前仍构造 `SparfAttentionBackend`；`python/flexgen_adapter/sparf_attention.py` 仍创建旧 `SparfKvCacheStore`，使用 cache_paths / 可选旧 buffered_kv。仅修改测试命令不会自动走新分组存储。

保留并复用：

- `python/runtime_common/prefill_runner.py`：公共 Prefill。
- `python/runtime_common/multi_batch_decode.py`：公共 microbatch Decode。
- `python/flexgen_adapter/microbatch.py`：`FlexGenMicrobatchKvWriter`。
- `python/cemu_flexgen/cemu_sparf_slot_scheduler.py`：提交、prepare、execute、collect。

## 5. 必须先解决的容量限制

**现在不能直接把新小测试的容量参数换成 OPT-13B 就跑。**

`include/hw/femu/sparf_group_abi.h` 当前 MAX_PAGES=65536，即 4 KiB 页下页池最多 256 MiB；MAX_STREAMS=4096。设备 FDM 配成 2 GiB、NVM 配成 96 GiB，并不会消除 ABI/实现中的这些限制。

定义 E=元素字节数，P=页字节数，D=head_dim，C=每页 channel 数（不是 SparF compression ratio）：

```text
Gt = P / (D*E)             token-K/V 每组 tokens
Gc = P / (C*E)             channel-K 每组 tokens
Nc = D / C                channel-K 每组页数
streams = layers * batch * kv_heads  （全部放同一个 CSD 的示例）
FDM_tail = streams*(2*Gc+2*Gt)*D*E + Nc*P
```

OPT-13B FP16、D128、P4096、C8：Gt16、Gc256，每 stream 环形尾部 136 KiB，另共用 64 KiB scratch。

| 同一 CSD 的 batch | stream 数 | FDM 尾部，不含其他 MR | T=2048 时三布局有效页数据 |
| --- | ---: | ---: | ---: |
| 1 | 1600 | 212.5 MiB + 64 KiB | 2.34375 GiB |
| 4 | 6400 | 850 MiB + 64 KiB | 9.375 GiB |

三布局容量公式为 `3*layers*batch*kv_heads*T*D*E`；此处 T2048 正好组对齐，仅是容量例子。实际按 prompt+decode_steps 的 KV 语义计算，不把输出 token 数直接当追加次数。未包括目录、工作区、文件系统、非整组填充和 FTL GC 开销；多个 CSD 需按实际分片重算。

后续应先建立容量预检，再扩展页池/stream 限制或设计合理分片/有界共享尾部。不能只删除常量：页序号和 hash 指针编码、guint 数组大小、extent 数量、metadata 大小、Host 内存开销都需审查。目前 grouped Attention 还限定 4 KiB 页及单输出/工作区 64 MiB 上限。

## 6. 下一位模型的实施顺序与验收

### P1：新 backend 接公共运行时，同时解决容量预检

1. 增加显式 grouped 模式，旧路径不变；实现公共 backend 接口，不复制完整 runner。
2. 设计每个 CSD 的 `(microbatch, layer, batch内位置, kv_head)` 唯一 stream 编号；校验尾部 microbatch 和 GQA 映射。
3. Prefill 按有界批块 APPEND；Decode 每层当前 K/V 先 APPEND，再提交 Attention。
4. mean 初期可继续由上层按 stream 增量维护，确保只计一次、生命周期正确、不读回全部历史 V；不要宣称设备已承担 mean。
5. 区分普通等待/flush 与终结 DRAIN。Prefill 后调用公共 flush 不能封存会话导致 Decode 无法 APPEND。
6. 一个 namespace 当前只有一个 grouped 会话；多个 backend/slot 应共享明确的 owner，最终仅排空和关闭一次。
7. `SparfGroupedDevice` 当前单调用者，共享 MR0。并发 APPEND 要锁住完整写 MR→execute→取状态事务，不能只锁 Python bookkeeping；也不要把整段 Attention 锁成全局串行。
8. 共享 Attention slot 的 stream_base 必须随请求稳定保存，不能排队后被另一 microbatch 改掉。
9. 权重加载前检查 pool pages、streams、FDM 与 MR 容量；增加失败、早退、关闭引用测试。

验收：先 OPT-125M 小形状，再具备容量后 OPT-13B；ratio1 再 ratio8；覆盖组边界、多 step、多 microbatch。对照同一输入的旧 SparF/NumPy 路径，不要求稀疏算法等于 dense。FUNCTIONAL PASS 不能代替数值对照。

### P2：在新模型路径上证明流水，不凭 slot 数判断

- 保存新路径 autonomous trace，验证 QK/V→PV 目标依赖，分别报告 Host 区间和目标时间。
- 检查 Prefill 计算/APPEND、Decode GPU QKV/设备 Attention、后台满组写回是否在真实测试中有交叠。
- 必要时增加默认关闭的后台写提交/实际完成/目标完成、背压等待、lease 事件；离线统计，不每个元素打印。
- 原 namespace 单父任务门控仍在。若需多个父任务在途，应另行设计有界队列、资源排他、completion 路由和错误回收，不能简单删除门控。
- 确认共享 CU/FTL 资源的目标排队，不能只把每阶段时间简单相加或用线程存在证明并发。

### P3：补齐与论文的行为差异，再谈精度

- 核对跨 head 拼块、channel-K 布局、NAND/NFC 读粒度、物理放置和 CSD 分片。当前仅完成 head-local 逻辑分组，物理放置仍由原 FTL 决定。
- 核对计算单元、Flash channel/LUN、带宽、CSD 数量、工作负载和 KV 驻留预算。
- 当前 APPEND 与 ATTEND 是两个 Guest 命令；是否合并、mean 是否下放，应按行为和接口开销模型明确处理。
- 目录及部分临时 I/O 元数据仍用 Host RAM，尚未全部纳入模拟目标 DRAM 容量。
- 明确后台写仍每会话单 worker，多个组串行；不能未经证据宣称与论文内部调度完全相同。

### P4：性能与模拟精度实验

- 行为固定后，记录真实 checkpoint、模型配置、batch/microbatch、prompt、新 token 数、top-r、compression、所有运行时校准文件。
- 区分 Host 运行开销、Guest 可见耗时、目标关键路径；worker 累计时间可能交叠，不能求和当端到端。
- 单独核对 Softmax 比值、近似/QK/PV 模型及 Flash 读写目标时间；不靠调参消除差异。
- 明确是否计入预加载、文件预分配、Prefill、最终 drain；使用相同统计边界比较论文。
- 低日志、多次重复；记录池初始化、FTL 预热/GC 状态。保留旧路径 A/B。

## 7. 测试入口、证据与边界

上一实现批次记录：G4 新增 7 项 Host 测试通过，另 107 项既有回归通过；QEMU 编译通过。本次交接没有重新跑这些测试。

- Host G2：`test_sparf_group_layout.py`、`test_sparf_group_device.py` 对应的 make 测试入口以 Makefile 为准。
- Host G3：`test_sparf_group_buffer.py`，可控假 FTL 验证发布、背压、错误与混合读取。
- Host G4：`test_sparf_group_attention.py`，生产 C 存储/桥接/workflow + CPU kernel + 假 FTL + NumPy；包含 V 在途 QK、PV 等 V 的依赖检查。
- Guest G2/G3：用户已贴过完整 PASS 摘要。
- Guest G4：用户最新反馈通过；2 层、batch2、Qheads4/KVheads2、D128、8 MiB 池，15/16/253/256/259/531 tokens × ratio1/8 × 2 层，共24次 Attention。不加载 OPT，诊断 runtime 不是论文校准。

Guest G4 回归命令：

```bash
source /root/cemu-gpu-env/bin/activate
cd /root/CEMU/tests/cemu
make instattention-sparf-group-attention-device-test \
  NVM_TEST_DIR=/mnt/nvme0 \
  FDM_TEST_DIR=/mnt/fdm0 \
  CSD_TARGET=cuda
```

Host G4 回归命令（已有临时扩展目录仍存在时）：

```bash
cd /home/lihan/cemu/CEMU/tests/cemu
PYTHONPATH=/tmp/cemu-opt-integration-build make instattention-sparf-group-attention-test
```

Host Python3.10 与 Guest Python3.13 不同，不要用 Host 构建覆盖 `instattention/build-guest`。生产设备修改可先 `make -C /home/lihan/cemu/CEMU/build -j4`，但运行中的 QEMU 不会自动切换新代码，正常重启由用户安排。

## 8. 环境和交接检查

- 宿主 repo `/home/lihan/cemu/CEMU`，Guest 代码 `/root/CEMU`，Guest 模型 `/root/models/OPT-13B`。
- NVM 是否用 SSD 文件必须看启动配置/日志：`run-csd.sh` 的文件后端依赖 CEMU_NVM_BACKEND_DIR，未设置可能仍走 Host DRAM，不能只凭历史口述判断。
- 曾发生 NVIDIA 575 内核模块与 580 用户库不匹配；使用私有 575 库路径曾恢复。不要修改其他用户驱动；实际 GPU 可用性、VFIO 与环境继承每次独立确认。
- freezing 保持原样；这些存储/工作流改动不是用来移除冻结成本。
- 报错先保留命令、QEMU log、Guest log、trace 和模型配置，不在 /tmp 保存唯一证据。
- 本地论文：`docs/InstAttention_In-Storage_Attention_Offloading_for_Cost-Effective_Long-Context_LLM_Inference.pdf`、`docs/CEMU_ASPLOS_26_DRAFT.pdf`。讨论新的论文行为结论时应重新查证，不把本实现推断当论文事实。

**给接手模型的第一项任务：先读本文和 G4 文档，核定容量与共享生命周期，再接新 grouped backend 到公共 OPT Prefill/Decode；不要继续重复写已通过的底层页目录/缓存，也不要直接开始长时间性能对照。**
