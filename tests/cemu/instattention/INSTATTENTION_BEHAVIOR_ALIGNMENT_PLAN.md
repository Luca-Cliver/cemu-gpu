# InstAttention 行为对齐：设备自治与异步流水设计

2026-09-16 增量：grouped Attention/OPT 路径已接入设备尾部与目录；本轮补充跨 head
APPEND_BATCH、设备有界就绪队列和合批写页，见 [SPARF_GROUP_BUFFER.md](SPARF_GROUP_BUFFER.md)。
不再逐 head 提交 Decode KV、逐组扫描全部 stream。保留真实 FTL 读写、目标 deadline 等待及 freezing。
页数上限合批尚不等于论文的物理块归属/通道放置；一个会话仍只有一个在途写批，
多请求双 Attention engine、回收和完整资源竞争仍需继续实现与验证。
下文各阶段描述为历史计划，不以局部 PASS 或耗时接近作为全部完成的证据。

日期：2026-09-15。状态：完成论文/代码核对和设计草案，尚未实现本文的新设备接口。

后续进展：第三步 A 的实验性单请求推进已实现，见 [SPARF_AUTONOMOUS.md](SPARF_AUTONOMOUS.md)。本文其余接口、设备写回和多请求资源设计仍为计划，不能据此视为已完成。

第四步的 B1 已增加可选的设备内批量写回数据路径，见 [SPARF_DEVICE_WRITEBACK.md](SPARF_DEVICE_WRITEBACK.md)。组触发、published、均值和尾部快照仍由 Guest 管理；B2 会话自治及多请求资源设计尚未实现。

写回 A/B 后的细化：[SPARF_GROUPED_KV_DESIGN.md](SPARF_GROUPED_KV_DESIGN.md)。将页分组布局提前到 B2 之前，先明确 token/channel 不同组容量和独立持久边界；已增加 Host 页布局参考测试，尚未切换设备读写路径。下文历史阶段顺序以该细化文档为准。

G2 进展：[SPARF_GROUP_DEVICE.md](SPARF_GROUP_DEVICE.md) 已增加独立的设备页池/目录、整组打包、FTL 追加写和索引读。旧路径未切换，设备自主凑组与 Attention/OPT 集成尚待完成。

## 1. 目标与不可变边界

目标是在 CEMU 中复现 InstAttention 的计算分工、数据依赖、存储组织和资源竞争，再比较模拟结果与论文结果。不是把运行时间调到与论文相等，也不是仅让一次生成返回 PASS。

本轮只做计划的前两步：行为差异核对、接口与状态机设计。后续实现分批进行，保留当前路径用于回归。

- **不修改 freezing 的进入、退出、时钟、补偿策略，也不绕过 freezing。** 功能计算继续走现有机制。
- 不删除 NVM/FDM 真实数据路径，不把依赖等待改成虚构的完成，不降低正确性验证标准。
- Host CPU buffer 可以作为当前 GPU/存储传输的功能实现；它不等同于论文 P2P。目标传输模型与功能搬运开销分开记录，不能重复收费。
- 保留已有 Prefill/Decode 公共运行时、OPT 权重计算、SparF 数学阶段、FTL 和 CUDA mirror 同步规则。
- 新实现先作为可选路径，不自动改变用户的大实验配置，不重启虚拟机或操作共享 GPU。

## 2. 依据与证据边界

论文：[本地 PDF](../../../docs/InstAttention_In-Storage_Attention_Offloading_for_Cost-Effective_Long-Context_LLM_Inference.pdf)。下文页码指 PDF 页序：

- 第 6–7 页：Algorithm 1、Figure 7/8，SparF 的选列、选 token、精确 Attention 与两个计算内核实例。
- 第 7–8 页：IV-C、Figure 9/11，双 K 布局、页级粗筛/NFC 细筛、通道布局和 DRAM 批量写回。
- 第 8–9 页：IV-D、Figure 10，系统分工、控制接口、逻辑地址和任务回收。

公开上层实现：[flex_opt.py:1034](/home/lihan/cemu/InstAttention/flexgen/flex_opt.py:1034)。必须注意：

1. `i` 是生成步，`k` 是 GPU microbatch，`j` 是运行时模块编号。
2. `j` 不总等于 Transformer 层：`sep_layer` 可将 Attention 和 MLP 拆成两个模块，另有输入/输出模块，见 [flex_opt.py:595](/home/lihan/cemu/InstAttention/flexgen/flex_opt.py:595)。
3. 循环交错提交下一份 Load、当前 Compute、上一份 Store，但也调用 `sync()`；该函数同步磁盘环境和 CUDA，见 [flex_opt.py:793](/home/lihan/cemu/InstAttention/flexgen/flex_opt.py:793)。不能把“无限向后提交、删除所有 barrier”当作论文要求。
4. 这段上层代码不能证明 FPGA 内部每个队列的仲裁策略。论文给出两个相同 Attention Kernel 可按负载服务两类计算，但未据此确定唯一的队列优先级、抢占方式或每次派发粒度。

本文区分：论文明确行为、当前实现行为、待验证的模拟策略。最后一类不能标为已复现。

## 3. 当前已实现什么，差在哪里

| 项目 | 论文行为 | 当前实现与差异 | 验收方向 |
| --- | --- | --- | --- |
| GPU/CSD 分工 | GPU Prefill 和非 Attention Decode，CSD 处理 KV/Attention | 已有真实 OPT 路径；不能因本轮改造复制一份 runner | 同一权重和输入的端到端回归 |
| Prefill 流水 | 每层 KV 传输可与后续计算交错 | 已有异步 KV writer；并非缺失 | 保留写入依赖，实测有可重叠区间 |
| 上层 microbatch | 交错 Load/Compute/Store，并保留依赖同步 | 已有异步提交；仍需逐项对齐模块粒度和权重策略 | 用依赖图而非日志相邻顺序判定 |
| 一次逻辑 attend | 控制器推进内部选取、读取、计算 | v1 已是一条 execute；v2 分四阶段，由 Guest 接力 | 增加设备自治推进，不误称从零实现单命令 |
| SparF 数据流 | Top-r Q → 通道 K → 粗分数 → Top-k → 精确 KV | 已有算法和设备内部读取；细节仍需独立数值测试 | 固定输入验证索引、概率、补偿和输出 |
| 精确 QK 与 V 读取 | 两者可并行，PV 等待二者 | 当前 workflow 已表达 `max(QK, V-read)`，也有先提交 V 读再算 QK | 保留该依赖，并覆盖跨请求资源竞争 |
| 计算资源 | 两个相同 Attention Kernel，可按负载服务两类计算 | Guest slot、功能线程和模拟 CU 不是同一概念 | 明确目标 engine 数和任务粒度后再配置 |
| Decode KV buffer | CSD DRAM 接收增量，后台成组写 Flash | 当前 FDM ring 有界，但后台发布由 Guest Python 执行 | 设备拥有写回与有效数据边界 |
| 双布局与过滤 | 双 K、页粗筛、NFC 细筛、跨通道 group 布局 | 三文件/extent gather 实现了部分语义，不等于完整论文布局 | 单列检查实际页访问、过滤和放置规则 |
| 生命周期 | 定制逻辑地址和任务 reclaim | 当前依赖文件、extent、MR 和关闭清理 | 先定义注册/排空/释放，后补任务感知映射与回收 |
| 时间模型 | Flash 与引擎各自竞争，沿依赖图完成 | 已有 FTL 与阶段 runtime；GPU Softmax ratio 是外推假设 | 不靠吞吐吻合倒推“正确”；保存校准来源 |

### 当前调用链

```text
公共 Decode runner：生成 Q/K/V，追加 KV，提交逻辑 Attention
  → SparF adapter
  → Guest slot scheduler
  → stage 1 execute → Guest 接力 → stage 2 → stage 3 → stage 4
  → 收集输出 → GPU Wo / 残差 / MLP

旁路：Guest buffer worker → token-K/channel-K/token-V → publish
```

关键位置：

- [multi_batch_decode.py:123](python/runtime_common/multi_batch_decode.py#L123)：层和 microbatch 的上层调度。
- [sparf_attention.py:78](python/flexgen_adapter/sparf_attention.py#L78)：追加 KV 和准备设备输入。
- [cemu_sparf_slot_scheduler.py:83](python/cemu_flexgen/cemu_sparf_slot_scheduler.py#L83)：Guest 的阶段接力。
- [sparf_workflow.h:183](../../../hw/femu/csd/sparf_workflow.h#L183)：已有设备内部 SparF 阶段与模型时间合成。
- [sparf_write_buffer.py:80](python/cemu_flexgen/sparf_write_buffer.py#L80)：Guest 有界 ring 与写回调度。
- [sparf_kv_store.py:212](python/cemu_flexgen/sparf_kv_store.py#L212)：文件发布边界。

因此，“已能异步”与“设备行为已完全一致”不是一回事。当前有重叠能力，也仍有 Guest 接力和文件发布产生的空隙；不能把所有空隙都判为错误，有些来自合法数据依赖或资源占用。

## 4. 目标：设备拥有一个可恢复推进的请求

这里的“可恢复”指进程内阶段状态保留，不指掉电恢复。

Guest 提交一次逻辑 Attention 后，设备自己推进内部阶段。父请求保存上下文，但**不在整个生命周期独占一个计算 CU，也不让一个功能线程阻塞等待所有未来模拟时间**。

```text
ADMIT：验证会话/层/版本，固定 Q、KV 可见边界和输出资源
  → SELECT_CHANNELS
  → CHANNEL_K_READ
  → APPROX_SCORE_AND_TOPK
  → SELECTED_K_READ
  → ┬ EXACT_QK_SOFTMAX
    └ SELECTED_V_READ
      → JOIN：两项都满足数据与目标完成条件
      → PV_AND_COMPENSATION
      → OUTPUT_READY → 父请求唯一完成通知

独立后台链：APPEND → 有界 DRAM group → WRITEBACK → PUBLISH
```

Top-r、Top-k、数据筛选等控制工作也需明确归属和耗时，不能因不调用 CUDA 就默认目标成本为零。它们的校准不足应显式记录，而非藏入不明“其他时间”。

### 复用与新增边界

- 复用现有 SparF 计算库的阶段入口，不把完整算法复制进新 runner。
- 复用 FTL 数据访问和延迟模型，不改成只返回估计值。
- 新增持久请求上下文、内部阶段完成分发、KV buffer 所有权和父请求完成汇聚。
- v1/v2 ABI 保留。新语义使用独立版本，不能把旧保留字段静默解释成生命周期指针。
- 先保持既有物理映射，验证自治推进；之后再对齐论文分组和放置。第一阶段不能宣称完整存储行为一致。

## 5. 接口与生命周期草案

以下是概念接口，不是当前已可使用的命令或 CLI 参数。论文的控制接口为 `config/attend/reclaim`；迁移接口不必逐字相同，但职责需对应。

| 概念操作 | 输入/输出 | 约束 |
| --- | --- | --- |
| REGISTER / IMPORT_PREFILL | 模型布局、层/头分区、已发布前缀、受控 extent 范围 | 初始化后设备拥有该会话的访问权 |
| APPEND_KV | 会话、层、token 位置、K/V | 接收完成表示设备已持有有效副本，不表示 Flash 写完 |
| ATTEND | Q、位置、可见 KV 版本、输出目标、runtime 配置版本 | 可与当前 token 追加合并提交；区分接收确认和计算完成 |
| DRAIN | 会话及可选层范围 | 所有请求和后台写回排空，报告失败 |
| RECLAIM / CLOSE | 会话及版本 | 不再接收新工作，待引用释放后归还 buffer、MR 和映射 |

每个请求携带唯一 ID 和会话 generation，避免旧完成事件误作用于复用后的 slot。检查 token 范围、维度、长度乘法溢出、extent 边界、MR 权限和配置版本；不能信任 Guest 提供的裸 Host 指针。

父请求固定或复制必要的 metadata，并引用 program、MR、extent 和 buffer 快照。子请求不能在父对象释放后继续访问这些对象。

现有 [sched.c:328](../../../hw/femu/csd/sched.c#L328) 的普通任务完成分支会投递 Guest 完成并释放资源。内部阶段不能直接套用这个分支：必须有内部完成路径，仅通知父状态机；最终父请求才投递一次 Guest 完成。

## 6. KV buffer：不是把 Python 写回函数搬进 C 就结束

每个会话、层及头/批分区至少维护：

- `accepted`：已由设备接收的连续 token 边界。
- `published`：三种持久布局全部写完并可供后续读取的连续边界。
- 写回中 group、引用计数、失败状态，以及与可见 token 边界对应的 V 均值。

请求看到的数据必须是“不变的已发布前缀 + 被固定引用的未发布尾部”，两者不重不漏。buffer 不能因为开始写回就覆盖；只有写回完成且相关读者释放后才可复用。

当前 `capacity_tokens // 2` 分组是 ring 实现策略，不是论文固定参数。最终 group 容量需由页大小、head dimension、dtype、channel 布局和目标 DRAM 预算决定；跨 head 批写也应纳入。

### 写入一致性

1. APPEND 拒绝越界、缺口和不合法的重复位置；重试不得重复累计 V 均值。
2. 写回成功要覆盖 token-K、channel-K、token-V；任一失败不能推进 published。
3. Flash 后台写与 Attention 读共享目标存储资源，不能作为没有带宽/延迟成本的 memcpy。
4. DRAIN 等待未满 group 的最后一批；关闭时不能丢掉尾部。
5. 错误后保留必要引用直到在途操作排空，不能让已损坏会话继续输出“成功”。

### 与 Guest 文件系统的过渡边界

不能一边让 Guest mmap/page cache 修改 channel-K，一边由设备直接覆盖同一文件的底层块，却不处理一致性。

第一阶段建议：Guest 完成 Prefill 文件发布和 extent 获取后，停止对这些范围的读写与重新分配，把 Decode 的访问权交给设备。文件和 MR 全程固定，验证通过设备读取接口进行。返还给文件系统前必须有明确的缓存一致性方案；仅仅调用一次 flush 不等于失效旧缓存。

如果无法证明该交接安全，应先使用独占的设备数据区域，而不是冒险混用。长远的论文对齐需要任务逻辑地址、两种索引映射和 reclaim 策略；ext4 文件 + FIEMAP 只能作为过渡实现。

## 7. 资源调度与模拟时序

### 三类数量不能混淆

| 数量 | 表示什么 | 不表示什么 |
| --- | --- | --- |
| Guest slots / microbatches | 可在途的请求及数据缓冲 | FPGA 内核个数 |
| CEMU 功能线程 / Host CUDA 执行资源 | 如何算出正确的结果 | 目标设备并行度 |
| 目标 engines / Flash channels / DRAM | 模拟设备资源及竞争 | 无限制实际线程并发 |

论文的两个 Attention Kernel 可以建模为共享引擎池；具体队列策略需要列为模拟假设，并做敏感性验证。不能简单增加 `nr_thread` 后就认定复现了两个 FPGA 内核。

首先明确一个 compute runtime 究竟覆盖单 head、head group，还是整个 microbatch。当前测量的整批 Softmax ratio 不能原封不动套到每个拆分子任务再累加；也不能把整批时间直接除以 2 充当双引擎模型。

### 阶段推进的两个条件

一个后继阶段启动，既需要输入的功能数据有效，也需要前驱的目标完成事件及资源条件满足。FTL 的物理处理完成，不自动等于模拟设备时间已到。

```text
QK_ready = selected_K 的目标完成时刻
QK_start = max(QK_ready, 分配到的目标引擎可用时刻)
QK_end   = QK_start + 本任务定义下的 QK/Softmax runtime
V_end    = FTL 对该 V 读取给出的目标完成时刻
PV_ready = max(QK_end, V_end)
PV_start = max(PV_ready, 分配到的目标引擎可用时刻)
```

上式省略了前面的粗分数阶段，但不能在实际实现中省略。还要计入同一资源上的其他请求、后台写、传输及控制成本。

父请求的完成来自依赖图，**不能先给所有子阶段收费，再给父请求叠加整条 workflow runtime**。当前单请求 `read + approx + read + max(QK,V-read) + PV` 可作无额外竞争基线，不是多请求固定总延迟。

### freezing 完全保留

- 计算子阶段通过现有功能计算封装及 freezing 执行，见 [compute.c:2027](../../../hw/femu/csd/compute.c#L2027)。
- 设备端可先提交 V 读再执行功能 QK；其物理数据就绪和模拟 deadline 分别保存。
- 不在冻结区间忙等“未来虚拟时间到达”。现有机制返回后，由正常目标完成处理继续推进依赖。
- 不改 freeze_entry 扣除方式，不新建一套时钟，也不强行抵消冻结影响。
- 如果复用当前计算 wrapper 会使内部阶段误占资源或错误发完成，调整的是内部 job 类型/完成分发，不是 freezing 本身。

[attention_submit_read:1564](../../../hw/femu/csd/compute.c#L1564) 和 [attention_wait_read:1601](../../../hw/femu/csd/compute.c#L1601) 提供现有读取基础，但当前阻塞等待/预期请求匹配不能直接变成多请求共享队列。新实现需按请求 ID 分发乱序返回，区分功能完成和目标完成。

FTL 内部读写可复用现有能力，不要求上层统一切换成公开 Indirect 模式。仍需核查请求入队时序、内部带宽分支和各层计费，不能同时提前预留未来资源又在真正派发时再次计费。

## 8. 分批实施顺序

### 第三步 A：单请求设备自治，保留现有 KV 存储

- 新增持久父上下文与内部完成类型，先以一个在途请求验证状态迁移。
- 复用现有四阶段算法与数据映射，由设备推进，不再由 Guest 往返四次接力。
- IO 等待不独占计算 CU；只有最后向 Guest 完成。
- 新路径默认关闭，旧 v1/v2 仍可运行。
- 本批不同时修改 buffer、布局、runtime 校准，便于定位回归。

### 第四步 B：设备 buffer 与后台写回

- 实现可见前缀/尾部、buffer 引用、均值版本、成组写回和 DRAIN。
- 先解决文件访问权交接，再移除该路径的 Guest 写回职责。
- 容量不足采用有界背压，不偷偷增加无限 buffer。

### 第五步 C：多请求与目标资源竞争

- 明确 head/group 调度单位，再接入共享 engine 队列和多请求完成汇聚。
- 验证 QK/V 交叠、不同请求的读/算交叠，以及后台写与读竞争。
- 对齐页分组、通道布局、过滤和 reclaim；未覆盖部分明确列在配置报告中。

### 第六步 D：接回 OPT 与论文规模

- 公共 runner 保留，只替换后端职责；保留 CPU buffer 传输的已知差异。
- 小数值回归 → 多步/多 microbatch → 相同输入的两条路径对照 → 长上下文。
- 功能与行为验收后再比较模拟时间，保存论文配置、所有差异和校准假设。

## 9. 验收与观测

以下为待新增测试要求，不是已经执行通过的测试。

| 类别 | 必须覆盖 |
| --- | --- |
| 生命周期 | 唯一完成、乱序子请求、重复事件、取消、关闭时在途 IO、MR/slot 复用 |
| buffer | 无尾部/全尾部/混合、ring 绕回、写回期间读取、失败不发布、最终部分 group |
| 算法 | 固定 Q/K/V 的两次 Softmax、Top-r/Top-k、alpha/mean、相同分数 tie；当前 token 保留规则单独核对 |
| 存储 | 两份 K 和 V 一致，非连续 extent，页边界，实际读页与目标过滤范围，读写共享通道 |
| 模型时间 | 无竞争基线、V 慢/QK 慢、引擎繁忙、后台写争用；不把各 worker 累计时间相加当 wall |
| 应用 | 同一模型/输入/精度下局部算子与端到端对照，多生成步和多个 microbatch，不仅检查 FUNCTIONAL PASS |

调试模式按父请求和阶段记录：会话版本、层/批/头分区、提交、数据就绪、目标 ready/start/end、资源编号、写回可见边界和 Guest 完成。子事件附父 ID，不能仅靠 program 编号配对。

Host monotonic、CEMU 模拟时钟、Guest 计时分别标注。调度器完成记录不是 Guest ioctl 返回时刻，不能把两者相减直接解释成纯算子开销。

正式性能运行关闭逐阶段日志，使用可选计数器和离线分析。不要要求每次都出现正的 overlap：数据量、microbatch 数、资源竞争可能合法地限制重叠；应先检验依赖和可用并行性。

## 10. 进入实现前的明确结论

1. 当前并非“没有异步”或“没有设备内部 Attention”，而是设备自治、写回归属和资源调度尚未完整对齐。
2. 单次 attend 是接口语义，不意味着一个计算任务从读到写全程独占 CU。
3. 将写回搬入设备只是其中一项；映射、批写、过滤、回收和目标资源竞争也需追踪。
4. 论文未公开的精确仲裁策略、GPU ratio 到 FPGA 的外推、当前 token 保留细节，仍属于待核定项，不能用性能接近来证明。
5. 下一次实际编码从第三步 A 开始，只改内部请求生命周期与推进路径，**不动 freezing，不一起改校准参数或大实验默认值**。
