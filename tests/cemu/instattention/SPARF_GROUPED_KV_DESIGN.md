# InstAttention 分组 KV：行为核对与读写设计

日期：2026-09-15。

**G1 初稿交付：设计与 Host 页布局参考测试；当时未接入设备。当前 G2 进展见下一段。**
现有 spans/compact 写回继续作为基线，不自动切换配置，不修改 freezing。

后续更新：G2 独立设备分组页目录、完整组 PUT 和索引 READ 已接入 QEMU/FTL，见 [SPARF_GROUP_DEVICE.md](SPARF_GROUP_DEVICE.md)。Host C 测试已通过，Guest 测试待运行；自动尾部 buffer、Attention 和 OPT 新路径仍未接入。下文保留 G1 时的设计状态说明。

本轮将原计划中较晚的“页分组布局”提前：设备 buffer 的容量、发布边界和读取协议都依赖组大小，
不能先把现有 `capacity // 2` 和单一 `published` 原样移到 C，再称为论文的设备管理。

## 1. 依据、差异与不能推定的细节

论文依据：[InstAttention PDF](../../../docs/InstAttention_In-Storage_Attention_Offloading_for_Cost-Effective_Long-Context_LLM_Inference.pdf)，IV-C、Figures 9–11；上层流水依据 IV-D。

| 事项 | 论文明确内容 | 当前实现 | 本次设计结论 |
| --- | --- | --- | --- |
| 三种数据组织 | token-K、channel-K、token-V | 已有三个文件 | 保留两份 K、一份 V，不增加数学近似 |
| token 页 | FP16、head_dim=128、4 KiB 页对应同 head 的 16 个 token | head 内历史连续，但写回按 ring 的半容量切分 | 组大小由页容量推导，而非 `buffer_tokens / 2` |
| channel 页 | 每页 2–8 个 channel，256–1024 个 token；可根据上下文调整 | 每 channel 预留整段历史，增量散写 | 一页内连续存几个 channel 各自的 token 段 |
| 两级筛选 | 先取含选中数据的整页，再由 NFC 去除不需要的 token/channel | 已有 extent/span gather，但尚未证明完整页放置和过滤链一致 | 记录整页 NAND 访问与过滤后有效字节，不能只按有效元素计 NAND |
| 页放置 | token 组沿 token 维度跨 Flash channel；不同 head 的组聚合进块 | 由现有 FEMU FTL write pointer 分配 PPA | 必须核查实际 PPA；连续 LBA 不保证论文的物理放置 |
| 增量写回 | CSD DRAM group buffer 累积、后台成组追加；跨 head 批写减小 WAF | Guest 凑组，设备对既有文件区间做扇区读改写 | 改为满页写入新分配位置；旧布局必要的 RMW 不能在旧路径直接删掉 |
| 控制归属 | CSD 内部维护 KV、逻辑地址和映射 | Guest 管理 accepted/published、尾部快照、均值和写回触发 | 新路径需要设备会话，Guest 只提交数据和任务 |

仍未由论文/本地公开上层代码确定的内容：某一实验精确采用 2、4 还是 8 channels/page；块拼批等待策略；
尾组强制持久化规则；buffer 预算不足时的仲裁；读写优先级；两个计算引擎的逐请求仲裁。
这些必须作为配置/实现假设公开，不能反向拟合论文吞吐后视为事实。

本地公开 [flex_opt.py:1034](/home/lihan/cemu/InstAttention/flexgen/flex_opt.py:1034) 交错 Load/Compute/Store，
但仍调用 [sync:793](/home/lihan/cemu/InstAttention/flexgen/flex_opt.py:793)。它不能证明 FPGA 内部队列无限并发。
[store_cache:405](/home/lihan/cemu/InstAttention/flexgen/flex_opt.py:405) 在最后生成迭代跳过无需再使用的 KV 写入；
本项目“强制排空并验证所有已接收 KV”的测试口径不同，最终性能对照要单独报告最后一步和 drain，不能偷偷排除。

### 当前代码核对入口

- [kv_layout.py:250](python/cemu_flexgen/kv_layout.py#L250)：当前 head-local token 与整列 channel 文件地址。
- [sparf_kv_store.py:111](python/cemu_flexgen/sparf_kv_store.py#L111)：Guest 按 head pwrite，channel-K 由带步长的 NumPy copy 写入 mmap。
- [sparf_write_buffer.py:19](python/cemu_flexgen/sparf_write_buffer.py#L19)：半容量组、共享发布边界；其 snapshot 仍经 Guest。
- [sparf_writeback_abi.py:17](python/cemu_flexgen/sparf_writeback_abi.py#L17)：spans/compact 表达同一组散写，compact 不改变物理 IO。
- [sparf_writeback.h:145](../../../hw/femu/csd/sparf_writeback.h#L145)：读取旧扇区 → patch → 写回。
- [sparf_workflow.h:226](../../../hw/femu/csd/sparf_workflow.h#L226)：当前三种布局共用一个已发布前缀/尾部协议。
- [ftl.c:185](../../../hw/femu/bbssd/ftl.c#L185)：write pointer 沿 channel、LUN、page 推进，不等于任务感知的 head/group 放置。

## 2. 页布局：先固定数据契约，不固定 FPGA 私有 ABI

参数：`P` 为页字节数，`D` 为每 head 维度，`E` 为每元素字节数，`C` 为每 channel 页的维度数。

```text
Gt = P / (D * E)              token-K/V 每页 token 数
Gc = P / (C * E)              channel-K 每页 token 数
Nc = D / C                    同一 token 窗口的 channel-K 页数
```

首版参考契约要求整除、页为 512 B 倍数、little-endian FP16/FP32；不整除时明确拒绝，不静默截断。
论文的自动组大小选择暂不实现。`C` 必须显式传入，参考测试扫 2/4/8，不能把其中一个称为图中已知配置。

对固定 `(session_generation, layer, batch, kv_head, layout_kind)` 的流：

```text
token-K/V page(group): [Gt, D]，token-major
  group = token // Gt
  offset = ((token % Gt) * D + channel) * E

channel-K page(group, channel_group): [C, Gc]，channel-major
  group = token // Gc
  channel_group = channel // C
  offset = ((channel % C) * Gc + token % Gc) * E
```

参考实现把 channel 页编号编码为 `group * Nc + channel_group`；这是每 head 流的**逻辑页号**，不是 LBA/PPA。
token-K、channel-K、token-V 的页号空间彼此独立。物理追加时由设备维护逻辑页 → 分配位置/有效长度映射，
不能再使用旧版 `channel * max_seq_stride + token * E` 的公式。

页内排列是本项目选用的确定性契约，与论文分组语义相容；不宣称字节级复刻未公开的 FPGA 格式。

### OPT-13B 的容量边界

`D=128, E=2, P=4096`，因此 `Gt=16`。

| C | Gc | Nc | 每层每 batch 的 token-K/V 满组字节（40 heads） |
| --- | --- | --- | --- |
| 2 | 1024 | 64 | 各 160 KiB |
| 4 | 512 | 32 | 各 160 KiB |
| 8 | 256 | 16 | 各 160 KiB |

当前测试配置的 `pgs_per_blk=256`，一个物理块为 1 MiB；这是当前 CEMU 参数，不是论文完整硬件配置已对齐的结论。
一次每 head 16-token 的 K/V 两类页合计 80 页，仅 320 KiB，不能说一次满组自然填满一个块。
需要明确跨哪些 head、时间组或其他流拼批；无足够数据时也不能无限等待整块导致 buffer 死锁。
块级聚合不代表 NAND 一次能 program 整块，底层仍须按页/plane/channel 能力模拟。

若每层每 batch/head 都预留一个未满 channel-K 组及 token-V 组，原始尾部容量约为：
`L * B * H * D * E * (Gc + Gt)`；K 可共享原始副本供两种布局打包，不必常驻两份完整尾部。
OPT-13B `L=40,H=40,B=4` 时，C=2/4/8 分别约 1625/825/425 MiB。
这只算尾部容量，**不含**写回中固定页、打包缓冲、Attention slot、索引、均值及 CUDA mirror；
还未计慢速写回形成的积压。不能据此保证现有 2 GiB FDM 对所有配置足够，也不能沿用 6-token ring。
后续必须在会话注册时计算总预算并拒绝超限，不用无限 Host 内存掩盖 DRAM 不足。

## 3. 三个持久边界，而不是一个 published

每个流维护 `accepted`，以及 `committed_token_k`、`committed_channel_k`、`committed_token_v`。
每个写回组另有有效 token 范围、状态、物理目的地、generation 和读者引用。

例如从空流追加 20 个 token，`C=4`：

```text
accepted = 20
token-K / token-V：16-token 页已写完时 committed 可以是 16
channel-K：512-token 页尚未满，committed 仍为 0
```

token 精读访问 Flash 的 `[0,16)` 和 DRAM 的 `[16,20)`；channel 粗筛仍要访问 DRAM 的 `[0,20)`。
不能因为 token-K 已落盘就释放原始 K；也不能为了等 channel-K 满组而阻止 token-K/V 的后台落盘。
乱序页完成使用完成位图/区间表，连续边界只在没有缺口时推进。

```text
FILLING → SEALED → QUEUED → WRITING → COMMITTED
                                      └→ FAILED
```

- APPEND 完成只表示设备持有稳定副本；accepted 不等于 Flash 完成。
- SEALED 后内容不变。COMMITTED 必须同时满足功能 IO 成功和目标完成时刻。
- 每个 Attention 固定 accepted 版本和各布局可见映射；即使请求中途发生 publish，也不能重复/遗漏 token。
- 原始 K 复用要同时满足两份 K 的持久化需求和所有读者释放；V 独立判断；均值只在 APPEND 成功时累计一次。
- 页打包和在途写回计入有界预算。失败不发布、不释放仍被使用的数据，不把未完成页读成零。

**Prefill 到 Decode：**满组可发布，未满组继续留在设备 buffer；不能把 Prefill 的部分页补零发布后，
又在 Decode 中原地补写而重新引入小 RMW。现有 Prefill 一次 flush 全文件的接口需要适配。

**尾组策略（待实现假设）：**终结会话的 DRAIN 排空所有完整页及部分组；部分页补零，但映射携带有效 token 数。
此终结操作后禁止继续 APPEND。非终结 barrier 只等待已密封组，未满尾部保持 DRAM 可见，不声称全部持久化。
若以后需要中途 checkpoint 后继续追加，应增加版本化新页/COW 协议，不能悄悄重写已发布页。
RECLAIM 等待请求和读者退出再归还资源，不与“忽略最后一步 KV”混淆。

## 4. 追加写与稀疏读必须一起迁移

### 写路径（目标，不是目前已接入代码）

1. REGISTER 传入几何、层/头分区、DRAM 预算和独占 NVM 池；设备登记一次受控映射。
2. Guest 提交 `(session, layer, position, K/V)`，不再每组展开数千物理 span。
3. 设备封组，打包完整页，跨 head 聚合待写页；先分配新位置，再向 FTL 发写请求。
4. 仅新分配、页对齐、完整覆盖的目的页可省略应用侧旧页读取；这不等于关闭 FTL 自身必要的 GC/搬迁。
5. 完成后发布映射，回收不再被引用的 DRAM。新映射不可覆盖其他会话或未完成组的地址。

第一批允许独占、预初始化文件 extent 作为 NVM 容器，但分配与页目录归设备，不允许 Guest mmap 同时更新。
新旧布局必须使用独立版本与临时文件，禁止原地重解释已有 KV 文件。文件容器不是论文无文件系统/任务感知 FTL 的最终实现。

### 读路径

- Top-r：按选中 channel 找到每个 token 窗口的 channel 页；同一阶段内去重页，再过滤页内未选 channel。
- Top-k：只跳过完全没有被选 token 的 token 页；读入后保留选中项并恢复请求顺序。
- token-V 使用同一组选中 token；保留 QK 与 V-read 并行、PV 等二者的依赖。
- 已发布页与未发布尾部从不同数据源取值，必须分别计算覆盖区间；未发布数据不发虚构 NVM 读取。
- NAND 请求按实际整页计量；NFC 过滤后字节与 FDM 写入字节单列，不把少量有效字节当成 NAND 传输量。
- 跨请求不默认缓存去重；若引入缓存必须声明容量、替换和时序，不以“少读了”代替论文行为。

原读 ABI 只有一套前缀/尾部和旧 stride，不能直接套新页目录；新增独立版本/会话访问入口，保留旧版回归。
数学算子可继续消费原来的紧凑 Q/K/V 数组，不必复制 Prefill/Decode runner 或重写 SparF 数学流程。

## 5. 资源、物理放置与计时验收

- 写回与 Attention 读进入同一目标 Flash 资源体系；读和写的优先级是需声明的策略，不默认免费重叠。
- 满页聚合、channel striping、跨 head block 聚合是不同层次；必须记录逻辑组和最终 PPA 的对应关系。
- 当前 FEMU 的分配方式需要专门验证。若不满足 token 维度跨 channel、head 组块聚合，就做受限的任务感知分配；
  不通过改带宽或删时间来掩盖放置差异，也不先承诺完全无需修改 FTL。
- 不修改 freezing、时钟补偿、功能计算包装；IO 功能完成与目标完成双重约束保留。
- Host 控制成本、Guest caller 时间、目标 IO 时间分别记录。Python 打包变快不自动意味着目标设备变快。
- 明确报告写有效字节、NAND 读/写页、应用侧 RMW、FTL GC 搬迁、drain 和端到端时间。
  现有写回的 384 B payload / 18432 B read / 18432 B write 是应用侧扇区搬运诊断，不能直接称为 NAND WAF。

## 6. 本轮可执行参考与下一批边界

新增 [sparf_group_layout.py](python/cemu_flexgen/sparf_group_layout.py)：
纯 Host 页编码/解码和逻辑地址参考，不分配 PPA、不执行 IO、不代表设备已支持分组布局。
测试按完整/部分组编码，检查 token-K/channel-K/token-V 数值和页内地址，覆盖 Prefill=253 到总长=260 的跨组场景。
部分页补零测试只验证终结尾组的字节契约；不代表已实现生产 DRAIN、并发或故障恢复。

```bash
cd /home/lihan/cemu/CEMU/tests/cemu
make instattention-sparf-group-layout-test
```

只需要 Python 和 NumPy，不需要模型、GPU、QEMU 或重新挂载。Guest 也可执行相同 make target。

后续实施门槛：

| 批次 | 交付与验收 | 不得宣称 |
| --- | --- | --- |
| G1（本轮） | 论文/代码差异、页契约、独立地址/编码测试 | 设备分组写回已完成 |
| G2 | 设备受控页池/目录，完整页追加写和两种索引读；小数据经真实 FDM/NVM/FTL 往返，验证页集合/无额外应用 RMW | 仅 Host codec PASS 就替代设备测试 |
| G3 | APPEND、分布局提交边界、有限 buffer、后台写、尾部访问、终结 DRAIN；故障与背压测试 | 只把 Python worker 移进 C 就等于自治 |
| G4 | 实际 PPA 布局和多请求资源竞争验证/修正；接 OPT；保留公共 GPU 调度和计算 | 连续 LBA 等于物理块/通道放置已对齐 |
| G5 | 固定论文配置与公开假设，对比同一 workload 的行为计数和模拟结果 | 强行匹配论文吞吐来校准未知项 |

G2 前先不替换现有大测试：必须有失败不发布、越界/重叠页池拒绝、部分 Prefill 保留、
读写交错、多会话/层/head 隔离、未满组关闭、buffer 预算不足的验收用例，再逐步扩大负载。
