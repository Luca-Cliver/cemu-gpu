# InstAttention 专用 FTL 工作记录

开始日期：2026-09-21。此文件持续追加设计决定、实际改动和验证，不再新建每日FTL交接文档。
背景和既有证据见 [开发历史与基线](../../../docs/cemu/DEVELOPMENT_HISTORY.md)。

## 1. 当前状态

**独立 native KV 读写模式已接入，真实 QEMU qtest 小闭环通过。**
`ftl_mode=instattention` 已成为有效的 FEMU device 属性，默认不设置或 `femu` 仍走旧模式。
本阶段仅允许一个 NVM namespace、新建文件后端和4KiB NAND页；不创建FDM/compute namespace。
Guest 已反馈 `/dev/ng0n1` 客户端 PASS（8 页协议/数据验证）；块设备 ioctl 的旧 nsid=3
重定向不能用于该测试。设备内部单页/合批入口已接到同一映射与 NAND 资源，见第15节。
**有界tail和后台写回已接入并通过隔离QEMU测试，见第16节；尚未把 OPT/SparF 和回收
接到 native 模式，不能直接替换现有大实验配置。**
第8～11节保留此前阶段的历史验收边界；本次接入、测试和命令见第12节。

| 阶段 | 状态 | 交付/验收门槛 |
| --- | --- | --- |
| D0 文档与基线 | 本轮整理完成 | 合并旧记录、保留原文追溯、修复迁移引用 |
| D1 接口及所有权审查 | 首轮完成，接入安全项待落实 | 见第8节；启用模式前必须处理数据搬运和退出顺序 |
| F1 独立原生KV读写 | 真实QEMU测试及用户Guest小测试通过 | namespace初始化、PRP请求、FTL worker、目标deadline完成、reset/unrealize |
| F2 完整生命周期 | 内部IO及buffer/后台写回已接通；reclaim待接入 | 接buffer/后台写回，task reclaim、空间耗尽与失败处理 |
| F3 SparF与模型接入 | 未开始 | 小数值、跨组、多请求、多microbatch，再OPT与论文规模 |

## 2. 基线与操作边界

- CEMU基线：`c5f0710988bb18a52817289ab4eede0d7d2b38d0`。
- linux-cemu基线：`86d7dda112cd22ce0d502295d02e80e0814c872c`。
- 用户反馈已推送；当前Guest二进制/内核与提交的对应关系仍须在设备测试时记录。
- 文档整理批次只改文档；地址/映射批次仅编译/执行临时Host测试，不编译或替换QEMU，
  不启动/停止VM、不执行设备IO、不操作GPU、驱动和实验磁盘。
- IO适配批次已在现有build目录增量编译QEMU，并用临时文件验证生产适配函数；
  没有启动/停止VM、打开实验镜像或操作GPU，原有FTL路由和freezing保持不变。
- 未跟踪配置dump、个人报告、模型、日志和镜像保留；不自动commit/push。
- Git基线不备份Guest镜像或NVM后端。要做可靠镜像备份，应正常停机后离线复制/快照。
- 新模式测试必须使用独立后端文件，禁止复用现有实验镜像或对KV专用空间自动mkfs。
- 保持freezing、算子数值要求和现有runtime参数不变；不顺带改校准数据来掩盖行为差异。

## 3. 论文依据和公开源码边界

依据：[本地InstAttention论文](../../../docs/InstAttention_In-Storage_Attention_Offloading_for_Cost-Effective_Long-Context_LLM_Inference.pdf)，
IV-B/C/D、图8～11（PDF第7～9页，印刷页1516～1518）。

| 论文明确行为 | 新模式需实现/保留 |
| --- | --- |
| 定制KV逻辑地址，驱动替换NVMe DWord10 | 请求表达KV身份，不由Guest文件offset决定KV身份 |
| token-indexed与channel-indexed两套映射 | K双布局、V单布局；设备拥有映射 |
| FP16 D128、4KiB页下16 token/group | 按几何推导，检查整除和边界 |
| 每页2～8 channel、256～1024 tokens | 参数化组大小；固定C8只是一个实验选择 |
| token组沿token维度跨Flash channel | 验证同head不同group的实际PPA分布，而非只看全局channel覆盖 |
| 不同head组优先同block，DRAM积累后后台批写 | 软件合批与物理放置同时满足；仍按NAND页编程、块擦除建模 |
| 页粗筛、NFC细筛 | 记录实际整页读与过滤后字节，不能仅按选中元素收费 |
| 精确QK与选中V读可并行 | 保留PV等待两者的依赖，检查真实数据与目标完成条件 |
| Host任务列表标识过期，空闲/页预算条件下reclaim | Host决定过期，设备定位并回收；不是任意丢弃活跃KV |

论文说明32-bit逻辑地址和字段语义，但图10的Length是回收范围，**不是字段位宽**。
Task字段编码、精确bit分配、block填充等待、混合任务块、队列优先级和具体双引擎仲裁
没有由上述描述完全确定。实现时标注假设，不声称字节级复刻或根据吞吐反向确定。

核对的本地官方仓库为 `../InstAttention`（相对CEMU根目录），提交 `3b2d4b2`：

- `greedy-ftl/src/nvme/nvme_main.c:73`：初始化DDR存储区域。
- `greedy-ftl/src/nvme/nvme_io_cmd.c:110`、`:143`：NVMe read/write按DDR基址加ID×block大小访问。
- 同文件`:81`：forward通过MMIO启动加速器并等待完成。
- `nvme_admin_cmd.c:110`存在ID_format设置，但这不等于已展示完整KV字段解析和映射。
- `InstCSD_hw/instcsd.bit`是bitstream，不能当作可审查RTL。

在这份公开源码中未找到论文完整KV双映射/放置/reclaim实现；这不是“作者未实现”的结论。
复现依据是论文明确行为和可见源码，不把目录名greedy-ftl当作完整代码证据。

## 4. 拟议架构与兼容边界

初始化时按NVM namespace的参数确定策略，不允许每个请求临时切换同一物理页池的所有者。
首版沿用每controller的NVM nsid=1，不承诺任意多NVM namespace路由。
名称为 `femu` 与 `instattention`；本轮选择器是QEMU device属性 `ftl_mode`，不是namespace JSON键。

```text
共享：请求接收/完成、NAND资源与时延、真实数据backend、计算与freezing
    ├─ femu：现有LPN映射、GC、fifo/kv_group实验放置
    └─ instattention：KV身份→group→专用映射→PPA与数据位置、任务生命周期
```

默认必须是旧模式；未知配置报错，不静默fallback。
两套策略不能各自分配同一物理页。多种任务共存先用隔离设备/namespace及明确的资源归属，
不偷偷复制一份NAND资源时钟来消除争用，也不让普通文件系统重写KV专用区域。

复用公共运行时、SparF数学、自治阶段和现有有界buffer；新模式移除KV对Guest pool extent的依赖。
宿主机SSD文件仍可作为真实数据backend，它不是Guest ext4/FIEMAP的替代映射权威。
第一阶段尽量沿用已有控制通道；若需要新metadata必须版本化、检查长度/溢出，不静默改变旧ABI。
是否改Guest内核由接口审查决定，不能提前保证完全不需要。

### 映射与数据一致性

新FTL拥有唯一映射权威。一个稳定page record应明确KV身份、有效token范围、PPA、backend位置、
状态、generation和读者/在途引用；当前核心已记录页身份、有效token数、PPA、状态和目标时间，
但尚无读者lease、回收和控制器退出协调，不能据此启用在线复用。
不能做一张KV→PPA表但仍在别处用旧pool地址随意取数据。
新模式选择由PPA推导backend页偏移，展平顺序与现有 `ppa2pgidx` 相同：
channel→LUN→plane→block→page。地址基础模块已实现范围/溢出检查，但尚无真实backend调用。
回收/复用要维护record和实际页内容；不能只改计时不改真实字节。旧模式继续使用LBA偏移，不迁移旧镜像。

### 生命周期与资源规则

- 初始化完整校验几何、地址乘法溢出、目录/尾部/在途内存预算。
- accepted与三个committed前缀保留；实际IO成功及目标完成后才publish。
- 存在读者、后台写或未完成目标事件时，不释放/复用页、task或MR。
- generation拒绝过期完成事件；错误后中止受影响会话并排空引用，不返回伪成功。
- reclaim只处理已声明过期且安全的范围；混合有效页块不得直接擦除。
- 块隔离/部分范围回收策略需明确；页失效不等于块已经擦除可重用。
- 空间不足采用有界背压或明确错误；不可无限分配Host内存或悄悄交回旧GC。
- erase、读写和后台任务争用同一目标NAND资源；父子请求不重复收费。
- 不在冻结区间等待未来虚拟时间；保持现有freezing及正常完成推进。

## 5. 接口审查清单

下列是已定位的入口，不是已完成全部审查；行号随后续改动可能变化。

| 入口 | 需核定的修改/风险 |
| --- | --- |
| [bb.c](../../../hw/femu/bbssd/bb.c) `bb_init/bb_exit` | 策略创建、失败回滚、线程退出及资源释放；不能仅增加一个选择开关 |
| [ftl.c](../../../hw/femu/bbssd/ftl.c) `ftl_init/ftl_thread` | NAND公共层与旧map/rmap/GC分离，普通读写和后台路径一起覆盖 |
| 同文件 `memory_copy_nvm` 与限速分支 | NVM↔FDM、内部请求及真实backend偏移不能仍落到旧map |
| [compute.c](../../../hw/femu/csd/compute.c) `sparf_group_transfer` | 改接KV请求、身份、完成路由与错误传播，保留页边界切分修复 |
| [sparf_group.h](../../../hw/femu/csd/sparf_group.h) | pool ordinal目录与新映射的边界；不能保留两份可独立漂移的PPA表 |
| [sparf_group_buffer.h](../../../hw/femu/csd/sparf_group_buffer.h) | 合批、publish、尾部引用、失败、关闭和reclaim协调 |
| [sparf_group_attention.h](../../../hw/femu/csd/sparf_group_attention.h) | 稀疏页去重、尾部合并、选择索引顺序和在途lease |
| [sparf_group_device.py](python/cemu_flexgen/sparf_group_device.py) | 新模式注册几何/身份，不再创建Guest文件pool；旧模式保留 |
| [启动脚本](../../../femu-scripts/run-csd.sh)和Guest挂载流程 | 独立backend和模式校验，防止自动mkfs/占位文件污染专用空间 |

现状补充：旧数据后端以LBA偏移访问，PPA用于NAND状态和时序；CLOSE只销毁grouped会话，
未实现task reclaim；DSM也不能直接冒充它。要逐条确定读写、覆盖、回收、重注册行为。
已有环境变量kv_group是进程级选择，新模式若按实例配置不得沿用全局静态bool混淆不同设备。

## 6. 验收顺序与停止条件

每批只改变一个可回归边界。尚未执行下列新模式测试，不把计划写成PASS。

1. 旧模式：无新配置时既有数值、IO计数、资源及失败路径不变，现有Host测试和QEMU构建回归。
2. 原生映射：地址边界/非法字段/溢出、跨task与generation隔离、token-K/channel-K/V一致性。
3. 放置与IO：同head跨channel、跨head块聚合、稀疏页去重、真实数据与PPA对应、整页读写/erase计数。
4. Buffer：15/16、255/256等边界，未满组、写回期间读、ring复用、失败不发布和终结drain。
5. 生命周期：容量耗尽、在途close/reclaim、部分范围有活页、擦除前不可复用、重注册与旧事件拒绝。
6. 设备：隔离镜像上真实NVM/FDM/FTL小测试；mock或C harness不能替代这一层。
7. 算法/应用：grouped Attention数值、同输入SparF参考、短OPT、多microbatch、多step，再论文规模。
8. 观测：Host区间、目标依赖图和Guest计时分开；PV需等QK和V，后台写争用纳入资源时间线。

旧模式回归失败、数据与PPA错位、资源未释放或存在活页误擦时停止扩大负载。
不通过调NAND/runtime、放宽容差、删除等待或关闭freezing让测试“过”。
性能比较在行为验收后进行，不保证新模式更快或必然接近论文曲线。

## 7. 操作日志

### 2026-09-21：文档归并与实现前准备

- 核对CEMU/linux-cemu HEAD及工作树，保留已有三个未跟踪文件。
- 阅读并合并10份历史文档到开发历史，保留模式协议和runbook，修正指向已删除文档的引用。
- 单独建立本工作记录；消除旧文“grouped未接OPT”“只有单父请求”等跨阶段误读。
- 初步定位bb初始化、FTL线程、内部搬运、grouped传输和buffer接口。
- 本批无C/C++/Python/内核实现变更；未运行设备实验、重启VM或修改磁盘、GPU、时延模型。
- 文档验证：检查8份新增/更新文档中的46个本地链接，无新增失效链接；代码围栏配对正确。
  10份删除文档全部列入迁移表，且可从记录的CEMU基线提交恢复。
  `git diff --check`通过，linux-cemu工作树未改动。
- 本次没有重跑280项代码测试或Guest测试；这是文档验证，不是新FTL实现的验收。

后续每批追加：修改文件/接口、默认行为是否变化、所有权和锁规则、执行过的测试、
未执行项、运行配置与日志位置、剩余风险。已有基线数据留在开发历史，不反复复制为新结果。

### 2026-09-21：首轮接口审查与地址基础模块

- 新增 `hw/femu/bbssd/instattention_address.h`：验证布局、解析KV坐标为页key/页内偏移、
  比较完整作用域key、验证PPA并计算物理backend偏移；无线程、分配、IO、时钟或全局状态。
- 新增C harness及Python unittest入口，Make目标 `instattention-native-address-test`。
  本模块没有加入QEMU调用链/meson，不改变旧FTL、Guest ABI或任何默认配置。
- 执行50项Host unittest全部通过：新地址1项（内含6组C断言）、旧布局8项、容量10项、
  grouped device 12项、buffer 19项。新测试以 `-Wall -Wextra -Werror -pedantic` 和UBSan编译。
- 另以 `-O2 -fsanitize=address,undefined` 验证通过；环境下LeakSanitizer报告不支持ptrace，
  该次重跑显式 `ASAN_OPTIONS=detect_leaks=0`，不宣称做过leak检测。
- 旧buffer harness仍有原来的unused static function警告，本批没有修改相关代码。
- 未构建QEMU、未运行Guest/GPU/OPT、未访问NVM镜像；这些结果不是原生FTL读写或回收验收。
- `git diff --check`通过；linux-cemu无变更；无commit/push。

## 8. 首轮审查结果和接入决定

### 8.1 普通读写的数据先于FTL映射发生

`nvme-io.c::nvme_rw` 在解析PRP后直接以 `slba << data_shift` 调用 `backend_rw`。
`ftl_thread` 随后才调用 `ssd_read/ssd_write` 为这些LPN建模。
因此仅替换 `ssd_read/ssd_write` 会出现“模拟读物理页A、实际仍取LBA位置B”的错误。
新模式必须在调用普通 `nvme_rw` 之前分流；不支持的普通块读写、copy或DSM应在触碰backend前拒绝。
初期原生KV命令经版本化控制请求提交给NVM FTL线程，不能把Guest的KV编码强塞进旧slba字段。
具体新命令接入仍待实现；普通模式的read/write流程本批未动。

### 8.2 内部搬运不能漏接

`memory_copy_nvm` 也先按LBA调用 `backend_rw_internal`，再进行NAND建模。
默认路径、internal_bandwidth延迟队列、Direct Memory Copy、Indirect和SparF自主读写都需覆盖。
grouped写回借 `cmd.nsid=3` 和私有completion ring返回，等待目标deadline后才发布目录。
这些 `nsid=2/3` 是现有路由约定，不是可以任意替换成新KV任务编号的字段。

`CEMU_THROTTLE_INTERNAL` 是未开启的另一编译分支，本批没有验证其可用性；
新模式初期不得未经测试宣称支持所有旧实验分支，也不能静默回退为旧LBA访问。
内部helper目前还混用“延迟返回值”和错误码，新接口应分离status和deadline，
失败不可被当作极短成功延迟。这个旧接口问题本批仅记录，未顺带改动所有旧路径。

### 8.3 页表和NAND资源归FTL线程，buffer不是第二个FTL

现有后台buffer线程在锁内打包、锁外transfer，完成后重新锁住并publish。
新方案保持：FTL线程唯一拥有KV动态映射、页/块分配和NAND资源可用时间；
buffer与计算线程只提交带身份的请求、持有返回的稳定record/lease，不直接改页表。
不得在buffer锁内等待FTL再回调同一把锁，不在FTL线程等待Guest或未来模拟时间。
未来pending page需要区分数据写入完成与目标可读时刻，不能在函数返回后立刻当已提交页使用。

一个namespace一个映射权威；task/generation显式分离，stream只用于应用适配，不作全局身份。
接入后注册的layout不可变，旧generation事件不得修改新任务；这些动态生命周期约束尚未编码。

### 8.4 退出路径是启用新模式前的安全门槛

源码中 `ftl_thread` 主循环为 `while (1)`，`bb_exit` 没有stop/join，却释放to_ftl和namespace私有对象。
`nvme_delete_namespace` 更先free_backend再调用namespace exit；`femu_exit`按namespace顺序销毁，
之后才处理poller。group-buffer正常CLOSE中的drain/join不能替代整个controller的有序停机。
这是代码层面的生命周期风险，不是本批已复现一次崩溃；当前进程退出也不能证明hot-unplug安全。

接入新模式前应实现并测试：停止新请求→保留完成处理能力→排空计算/写回/内部IO→停FTL worker→
停完成线程并释放ring→释放映射和backend。错误退出不能无限等无法推进的Guest时钟。
不能只交换free_backend与exit两行就宣称问题解决，也不能一开始全面改写旧调度/冻结。
具体controller停止协调作为F1接入工作单独审查；此处没有提前修复或启用热插拔支持。

### 8.5 初始化、存储隔离和容量

`bb_init` 目前在ring建立前启动FTL线程，靠dataplane_started门控；新初始化应先准备全部资源再启动。
namespace初始化对 `Error **` 的传播/失败回滚也须检查，不把半初始化状态交给worker。
新模式不建立旧maptbl/rmap/line-GC，不能运行旧的前台/后台 `do_gc` 分支。
NAND时序共享现有资源定义；通道传输分支当前关闭属于既有模型配置/实现状态，本批不改时延。

`femu-scripts/cemu-mount.sh` 对未挂载NVM执行mkfs，跳过benchmark占位文件不等于跳过格式化。
原生KV namespace必须在启动、命令与挂载侧显式保护；仅换后端文件名不够。
首版限定独立namespace/镜像、无掉电恢复承诺；后端格式标识、拒绝误用旧镜像和退出策略
完成前不向用户提供原生FTL启动命令。

## 9. 地址基础模块的调用契约

- `cemu_kv_layout_init` 只接受合法、可整页分组的几何；派生容量含三种布局末尾页填充，检查uint64溢出。
- `cemu_kv_resolve` 只接收已由init成功产生、之后保持不变的layout，不能把Guest字节当成layout指针。
  输出完整作用域page key、每head/每kind内的逻辑page_index和byte_offset；它们不是LBA或PPA。
- key包含controller、namespace、task、generation、batch、layer、head、kind、group、channel_group；
  task=0合法，namespace/generation=0拒绝。身份不同不会被页去重合并；注册表的存活检查仍待实现。
- 输出 `token_capacity` 是max_tokens截断后的页容量，不是accepted/committed，不证明数据已写入。
- `cemu_kv_physical_offset` 检查PPA字段与现有bit容量、后端总尺寸和int64大小上限，
  再按物理页编号计算字节偏移。它不选择PPA，不证明跨channel分配策略已经实现。
- 失败不修改输出；结构体是内部C契约，没有作为wire ABI传输，也没有臆造论文的32-bit位宽。
- C测试中的字节后端是小型栈数组，不是实际NVM/FTL；代码无Host动态分配或计时副作用。

Host可独立执行，不需要模型、GPU、Guest桥或QEMU重启：

```bash
cd /home/lihan/cemu/CEMU/tests/cemu
make instattention-native-address-test
```

本节地址模块仍不负责分配；使用这些key的映射/分配核心见第10节。
不能把独立核心PASS解释为“两套FTL已可选择”。

## 10. 2026-09-21：独立页映射与块分配核心

### 10.1 实际实现与调用链

新增 [instattention_mapping.h](../../../hw/femu/bbssd/instattention_mapping.h) 和
[instattention_mapping.c](../../../hw/femu/bbssd/instattention_mapping.c)，
尚未加入QEMU构建或任何现有设备调用链。不是另一份Python模拟器：后续设备请求可调用此C核心，
但目前只由独立Host harness链接验证。

```text
map_create(独占namespace/NAND几何、元数据上限)
  → register(task + generation + KV几何)
  → reserve(页首KV坐标、有效token数)
      → KV key去重 → 分配块/页 → 返回PPA、backend_offset和ticket
  → 调用方完成整页真实写入，并取得同一NAND模型的目标完成时间【设备接入待实现】
  → complete(ticket, IO成功/失败, ready_ns)
  → lookup(KV元素坐标, now_ns)
      → 检查作用域、有效token前缀、IO状态和目标时间 → 页位置 + 页内偏移
```

- create绑定controller/namespace，register复制并验证几何；同task不同generation拒绝，
  重复注册拒绝。不同任务的同坐标不会合并，也不共用块。
- reserve只接受整页起点，token/channel组按地址模块推导；三种kind使用独立逻辑key。
  一张key哈希表承载两个K布局与V布局，不借助旧maptbl、Guest pool ordinal或FIEMAP。
- 页不可覆盖：重复reserve返回EXISTS，不分配第二个PPA，不更改既有记录。
  valid_tokens是页首连续有效前缀，partial页一旦提交不可继续扩展；未满尾部必须由上层buffer维护。
- 每页的ticket在实例生命周期内单调且不复用。complete校验ticket、完整key、PPA、偏移及有效长度，
  旧generation/错误ticket内容和重复完成不能覆盖既有状态。
- IO未完成或now_ns小于ready_ns，lookup返回PENDING；实际IO失败返回IO_FAILED，
  页内超出valid_tokens返回NOT_FOUND。错误不改结果输出，不以零数据冒充成功。
- ready_ns/now_ns由将来的FTL线程传入同一目标时钟域；测试用人工时间点，核心不计算runtime，
  不sleep、不忙等、不改freezing。written_pages表示真实IO成功通知计数，不代表目标时间已经到达。

### 10.2 物理放置规则与假设

当前明确采用以下工程策略，符合论文的token维度条带和head聚合方向，
**不是公开源码已给出的精确PPA公式**，后续设备trace仍须验收：

1. `flash_channel = group_index % NAND_channels`。
   token-K/V的group_index按token组计算；channel-K按其更长的token窗口计算。
   同窗口各head、各channel_group不会因为head编号被条带到不同Flash channel。
   两种K布局的窗口长度不同，不能误认为同token在两种布局必定位于同channel。
2. 活动块域为 `(task/generation, batch, layer, kind, flash_channel)`，不包含head。
   到达顺序相邻的head页优先顺序填同块；块满后换块，不保证超过剩余页数的一组head强行落单块。
   跨layer/kind/task不混块；同域不同token窗口可继续填未满块。
3. 每channel新块的ordinal先轮转LUN，再plane，最后增加block编号。
   页在块内从0递增；每块只被分配一次，不覆盖失败/已写页。
4. PPA按channel→LUN→plane→block→page展平为backend字节偏移，
   数据和时序接入时必须使用这个位置，不能继续用旧文件逻辑offset存字节。

容量不足返回NO_SPACE，不跨到其他channel偷偷改变放置、不触发旧FEMU GC。
因此单channel耗尽或域隔离碎片会先于全盘字节耗尽；这是显式错误，不是已经实现的容量优化。
失败页及其块保留，不能把可能编程过的页当空白页再用。擦除/回收/安全重注册留给生命周期阶段。

### 10.3 内存、线程和退出边界

- create预分配有上限的task/page/domain记录和两张半满上限哈希表，检查size_t乘加溢出与预算。
  reserve/lookup/complete不再动态分配；页目录预期O(1)，task查找为有界线性查找。
  memory_bytes是所申请结构/数组的字节数，不含系统allocator额外开销，也不是FDM数据预算。
- max_pages是元数据记录上限，不是每task的物理空间预留；达到上限返回LIMIT，区别于NO_SPACE。
- 当前约定单一所有者线程，后续应为FTL线程；没有给多线程并发调用伪造线程安全保证。
  注册表不删除task、不允许generation切换；旧事件/引用不能跨destroy后新实例使用。
- destroy只释放独立核心的Host结构。调用者必须已停止访问并排空事件；
  它不代替第8节要求的QEMU线程stop/join、真实IO drain、backend关闭或在线reclaim。
- 尚未接buffer发布前缀、读者lease、GC、持久化恢复、模式选择或命令ABI；旧路径没有改动。

### 10.4 已执行验证

新增 [C harness](guest/test_instattention_mapping.c) 与 [Python驱动](python/test_instattention_mapping.py)，
可执行：

```bash
cd /home/lihan/cemu/CEMU/tests/cemu
make instattention-native-address-test instattention-native-mapping-test
```

- 新映射7项Host unittest通过：身份/任务块隔离、可见性与错误完成、字节读回及放置、
  论文页几何、容量耗尽、非法参数/元数据预算、分配失败回滚。
- 字节测试使用内存数组作为假backend，经返回的PPA偏移写整页、按每个KV元素坐标读回；
  覆盖双K/V、2 batch×2 layer×3 head、512B页35tokens、4KiB页259tokens及C2/4/8。
  4KiB FP16 D128覆盖15/16与255/256；不使用算子或真实NAND。
- 检查物理页不重复、token维度跨channel、首次head组同块、LUN/plane轮转；
  failed页不复用、目标channel耗尽时其他channel仍可分配。
- 链接器包装calloc/free，依次注入6个初始化分配点失败，验证输出不发布、已分配结构全部回滚。
  默认编译启用O2、严格警告和UBSan；另行O1 ASan/UBSan编译并执行全部7个场景也通过。
  独立ASan运行设置detect_leaks=0以避开环境的LeakSanitizer限制，不声称跑过LSan；
  calloc/free计数验证的是本核心分配点的回滚，不替代全进程泄漏检测。
- 地址1项及既有group layout 8、capacity 10、device 12、buffer 19项Host回归通过，
  合计57项。旧buffer harness仍有既有unused-function警告，本批未修改。
- 未构建/替换QEMU，未执行Guest、OPT、GPU、实际NVM读写和论文性能实验。

后续IO适配进展见第11节；读者/在途引用与task失效/块擦除条件仍须落实。
模式选择、请求路由、镜像防误用及有序退出一起验收后，才提供新模式启动命令。

## 11. 2026-09-21：PPA数据IO服务与QEMU适配接口

### 11.1 本轮接到哪，尚未接到哪

新增 [instattention_io.h](../../../hw/femu/bbssd/instattention_io.h) /
[instattention_io.c](../../../hw/femu/bbssd/instattention_io.c)，把映射、整页数据搬运、
NAND目标时间依赖和错误返回组合为可调用服务；不是只改时延、不改数据地址。

生产适配入口 [ftl.c](../../../hw/femu/bbssd/ftl.c) 中的 `bb_native_io_create`：

```text
bb_native_io_create(namespace, metadata limits)
  → cemu_kv_io_create/register
  → write/read
      ├─ KV坐标 → 新映射 → PPA和backend_offset
      ├─ native_page_schedule → 原有ssd_advance_status → 原有LUN资源时间线
      └─ native_page_transfer → backend_rw_internal_checked → 实际backend映射字节
  → 返回实际搬运字节数、元素偏移、开始/目标完成时间
```

以上已加入 [meson.build](../../../hw/femu/meson.build) 并编译进入QEMU。
**bb_init和ftl_thread尚未调用此入口**；普通NVMe命令、SparF和文件pool仍走之前路径。
原因是旧namespace已经有LBA映射/块分配器，直接附着新映射会造成双重分配；
原有线程退出、原生空间拒绝普通LBA写、Guest镜像保护也不能只靠接口库解决。
本轮交付是IO适配层和可执行回归，不是已经可启动的第二套FTL。

### 11.2 整页IO与目标完成语义

- 写：检查页大小/时间/身份 → reserve → 对该PPA预订NAND写时间 → 实际backend写入 →
  complete。重复页不再发IO；调度失败、倒退deadline、数据失败均留下失败页，不发布、不复用。
  失败后已预订的NAND时间不回滚，避免抹掉可能已经发生的设备尝试。
- 新增 `cemu_kv_map_query_written`：只供内部调度查询功能数据已经写好的页及ready_ns；
  它不表示Guest此刻可读。原有lookup仍检查now_ns，未到目标时间返回PENDING。
- 读：查已写映射，`read_start=max(request_start, write_ready)`，再按同一PPA预订NAND读，
  整页从backend读入私有scratch，成功才复制给调用者；调用者按元素偏移过滤。
  因而同LUN的其他任务还会继续排队，不是简单独立地加一次固定读时间。
- 读函数返回的buffer是功能执行结果，**调用者必须等result.ready_ns才发布命令完成/供依赖任务消费**。
  这与CEMU“先算出真实数据、按模拟deadline完成”的分工一致；不能把C函数返回当目标完成。
- 服务不sleep、不忙等、不冻结，也不在冻结期间等待未来目标时间；完成队列接线仍待实现。
- 一个有界页scratch隔离失败读取，不让部分失败数据污染输出；该页和服务结构纳入元数据预算。
  操作按owner线程串行进入，不支持多个线程共用scratch；多请求的目标时间仍可重叠。
- 输入/输出参数是Host可访问buffer，不自动替Guest GPU/FDM做同步；将来上层调用必须准备好源数据、
  维护目的FDM dirty状态和MR生命周期。本轮不绕过这些要求，不宣称P2P已接通。

### 11.3 后端、NAND与防混用

- 新增 [mapped_io.h](../../../hw/femu/backend/mapped_io.h) 与
  [backend.c](../../../hw/femu/backend/backend.c) 的 `backend_rw_internal_checked`。
  检查offset/size/容量，读前执行现有CUDA prepare_host，写后标记host dirty；同步失败显式返回错误。
  **旧void版backend_rw_internal及其调用点没有替换**，不把其历史错误传播问题说成本轮全部修复。
- 实际字节位置取自新映射返回的PPA展平偏移；适用于既有DRAM或mmap文件backend。
  文件路径不由IO服务打开；本轮测试只创建并删除/tmp临时文件，未触碰现有csd0-nvm.img。
  mmap写入成功不等于Host磁盘fsync/掉电持久化，仍沿用现有功能后端语义。
- NAND适配复用 `ssd_advance_status`，没有第二套独立资源时钟，没有修改NAND延迟或打开原先关闭的
  channel传输分支；先验证PPA范围与时间加法溢出，再更新真实NvmNamespace的LUN可用时间。
- factory拒绝已有maptbl/rmap/line管理器的namespace，并只允许同namespace创建一次native实例。
  destroy后也保留used标记，未实现安全擦除前不允许重建映射重新使用相同PPA。
- factory要求调用方已初始化合法NAND拓扑与backend，且由单一FTL所有者调用。
  `bb_native_io_destroy`仅释放组件，**不是**线程stop/join、任务drain或整机退出协议。

### 11.4 验证与未执行项

在Host执行，无需进入Guest或加载OPT：

```bash
cd /home/lihan/cemu/CEMU/tests/cemu
make instattention-native-address-test instattention-native-mapping-test instattention-native-io-test
make instattention-native-qemu-io-test
```

第二个目标需要配置好的QEMU build/compile_commands.json，可用CEMU_QEMU_BUILD_DIR指定位置。
它在临时目录按QEMU实际编译参数编译ftl.c、backend.c和新核心，再链接独立测试程序，
不会启动qemu-system-x86_64；缺少构建配置会明确SKIP，不应算作PASS。

- IO服务5项测试通过：临时文件mmap/pread核对物理位置与字节、写→读依赖、同LUN争用/跨channel并行、
  数据/同步/调度失败、非法输入/预算/页边界；此组NAND callback是mock。
- 生产适配1项通过：直接执行生产 `bb_native_io_create`、`ssd_advance_status` 与checked backend，
  验证旧映射拒绝、单实例与重建拒绝、PPA文件字节、同LUN排队/跨channel独立、时间溢出不推进资源。
  使用受控qemu_clock_get_ns和CUDA关闭的临时namespace；CUDA锁桩若被意外调用立即abort。
  没有用mock替换生产NAND排队函数，但这仍不是FTL线程/Guest/实际GPU测试。
- 时间例：write到达100完成300；read到达110只能从300开始，完成340；
  另channel的write到达110完成310；同LUN的第二个write则排到340之后、完成540。
  这些是小测试纳秒参数，验证依赖算法，不是论文校准测量。
- 地址1项、映射7项及旧group layout 8/capacity 10/device 12/buffer 19项回归一起通过：共63项。
- 新IO服务另行ASan/UBSan运行5个场景通过；detect_leaks=0，未宣称LSan验证。
- `ninja -j2 qemu-system-x86_64`增量构建通过；旧compute.c仍有已有unused-variable警告。
- 未启动VM、未跑OPT、未改linux-cemu、未改freezing、未验证新模式的Guest完成队列与线程关闭。

下一步必须补namespace原生初始化与请求路由、默认旧模式和拒绝混用策略；
并把result.ready_ns接到完成机制，验证停止接收/排空/停止线程/释放backend的生命周期。
在这之前，不给旧实验启动命令加一个无效ftl_mode参数，也不让新FTL写旧镜像。

## 12. 原生 namespace / NVMe 完成闭环

### 12.1 本次接到哪里

```text
Guest native_kv_client.py / QTest模拟Guest命令与内存
  → NVMe IO SQ，REGISTER / INFO / WRITE / READ
  → nvme_process_sq_io → bb_native_submit → to_ftl[1]
  → native_worker → native_execute
  → cemu_kv_io_* → KV身份/group映射 → PPA
  → 原有 ssd_advance_status 的LUN资源时间 + checked文件backend
  → req.expire_time = ready_ns → to_poller[1]
  → 原有PCIe延迟与CQ完成机制，虚拟时间达到deadline后发布CQE
```

- `instattention_native.c`提供真实namespace初始化、独占FTL worker、PRP与版本化KV命令解析。
  只分配NAND channel/LUN资源状态，不初始化旧maptbl/rmap/line管理器，不走Guest LBA/FIEMAP。
- WRITE现在由调用方组织完整4KiB页；READ返回完整页。设备解析token/channel身份并决定PPA。
  **这不是自动把一个token拆成K双布局/V单布局的APPEND接口**，也尚无NFC元素过滤。
- REGISTER最多64个task；映射条目上限为min(namespace页数,65536)，核心内存预算64MiB。
  到界限报错，不覆盖已存在数据，不宣称有GC/reclaim。task身份包含controller/ns/task/generation。
- `ftl_mode`省略或`femu`保持旧路径；未知值报错。native拒绝普通LBA读写、DSM、WRITE_ZEROES、
  FORMAT及旧FEMU私有管理命令，避免普通文件系统绕过KV页池所有权。
- 文件以O_EXCL/O_NOFOLLOW新建；已有文件/符号链接一律拒绝，不能重开旧native文件恢复映射。
  native使用私有统计内存，不复用现有实验的共享统计区，不要求CUDA或GPU。
- PRP方向沿用CEMU命名：`dma_write_prp`从Guest取数据，`dma_read_prp`向Guest返回数据；
  这是NVMe write/read语义。未更改旧DMA函数。

### 12.2 时间与生命周期

- 单个FTL worker串行维护映射，但不同LUN的目标时间仍由原有资源模型独立推进；
  没有用Host线程耗时替代目标NAND延迟，也没有新增/修改VM freezing逻辑。
- 数据拷贝可以先完成，CQ必须等目标完成时间；FLUSH等待此前原生请求的NAND deadline。
  file mmap写入和FLUSH均**不承诺Host fsync/掉电恢复**。本接口用于功能后端与目标时序实验。
- native强制单IO poller和同线程CQ poller；admin队列修改与IO扫描互斥。
  SQ请求池耗尽时停止取命令，CQ满时等待Guest消费；SQ与CQ编号可以不同。
- reset/退出先停止IO接收并join poller，再停止/join FTL worker，再释放环、SQ/CQ映射和backend。
  reset丢弃尚未报告的完成和未执行命令，不等待冻结中的目标时钟；已经功能写入的页保留，
  不重复分配这些PPA。这里不是“所有请求都成功完成”的drain承诺。
- native worker加入RCU读侧保护。新模式补全队列和DBBUF的DMA解除映射、队列复用、
  共享CQ索引及未完成IO时禁止删队列的保护；旧模式不切换到这套线程生命周期。
- reset后可重新建队列/DBBUF并读取已写KV；QEMU进程退出则丢失内存映射，必须换新后端。

### 12.3 ABI v1

定义：`include/hw/femu/native_kv.h`；全部小端，除FLUSH外每次PRP传输4096字节。
这是CEMU过渡测试ABI，**不是论文未公开的32-bit字段分配复刻**，也没有修改linux-cemu。

| 命令 | Opcode | Payload |
| --- | --- | --- |
| REGISTER | 0xc1 | 8个u32：B,L,H,D,T,C,element_bytes,page_bytes，剩余清零 |
| INFO | 0xc2 | 返回magic/version/controller/ns/page、geometry和页统计 |
| WRITE | 0xc5 | 一个完整物理页的内容，valid_tokens标记有效前缀 |
| READ | 0xc6 | 返回坐标所在完整页，非法或未发布坐标拒绝 |
| FLUSH | 0x00 | 无payload，等待此前NAND目标deadline |

DW10低16bit=version1、高16bit=WRITE的valid_tokens；DW11=task；DW12=generation。
DW13低16bit=batch、高16bit=layer；DW14低16bit=head、16..23bit=kind、24..31bit=channel；DW15=token。
kind0=token-K、1=channel-K、2=token-V。REGISTER的坐标字段必须为0；INFO除version外字段为0。
READ可访问页内任意已发布token；WRITE必须页组起点对齐。
INFO偏移0/4/8/12/16/20/24为u32 magic/version/controller/ns/pagebytes/nchannels/luns_per_channel；
偏移32/40/48/56/64/72为u64 backendbytes/reservedpages/writtenpages/failedpages/claimedblocks/deadline。
核心失败通过NVMe失败状态及CQ result中的CemuKvStatus返回；不返回成功但静默丢弃。

### 12.4 已执行验证

Host上：

```bash
cd /home/lihan/cemu/CEMU
ninja -C build -j2 qemu-system-x86_64
make -C tests/cemu instattention-native-qtest
```

qtest需要本地Unix socket权限；使用临时16MiB文件、64MiB QEMU内存及qtest加速器，
没有KVM Guest CPU、没有GPU、没有打开现有OS/NVM镜像。它不是mock NVMe设备：
实际执行本次构建的qemu-system-x86_64、PCI/MMIO/PRP、FTL worker和CQ。

3项真实QEMU测试通过：

1. 8个实际页的token/channel/批次层head隔离与有效前缀，16/256边界；拒绝重复写、过期generation、
   越界、普通LBA命令；暂停推进虚拟时间时CQ不提前完成；带pending请求reset后重建并读回。
2. 16深度CQ的背压、17请求跨wrap、未完成时拒绝删除SQ；目标时钟不走时unrealize能退出。
3. 已有后端内容不变、未知模式/零channel配置拒绝；不会创建无效配置的后端文件。

旧group layout/capacity/device/buffer共49项与native基础/生产适配14项均通过。
尚未运行Linux Guest ioctl测试；未跑OPT或性能对照，未声称中断/KVM Guest已验收。

### 12.5 Guest验证命令

先正常停止占用Guest OS镜像的VM；**不要强杀他人的VM，也不要改GPU绑定**。
Host启动独立测试机：

```bash
cd /home/lihan/cemu/CEMU
bash femu-scripts/run-native-kv-test.sh
```

无/dev/kvm权限时由管理员授权或使用sudo运行此脚本。脚本不直通GPU、不开SSH转发，
使用2GiB Guest RAM、OS qcow2临时overlay、只读代码共享和新16MiB NVM。
OS backing仍采用文件锁；若另一个VM占用，不要绕过锁。脚本只检查语法，尚未实际启动Guest。
镜像、内核路径可用CEMU_NATIVE_BOOT_IMAGE / CEMU_NATIVE_KERNEL / CEMU_NATIVE_INITRD覆盖。

脚本使用 `init=/bin/bash` 跳过旧系统自启动的 cemu-mount/fio；无 systemd、无登录提示，
bash 无 job control 提示正常。Guest内不要运行旧cemu-mount.sh或mkfs/mount NVMe：

```bash
mountpoint -q /proc || mount -t proc proc /proc
mountpoint -q /sys || mount -t sysfs sysfs /sys
mountpoint -q /dev || mount -t devtmpfs devtmpfs /dev
mount -o remount,rw /
mkdir -p /root/CEMU
mount -t 9p -o trans=virtio,version=9p2000.L cemusrc /root/CEMU
cd /root/CEMU/tests/cemu
make instattention-native-device-test NVM_DEVICE=/dev/ng0n1
```

客户端仅用Python标准库和已有NVMe passthrough ioctl，先INFO确认magic才写KV。
Guest探测分区可能因普通READ被拒绝而打印NVMe错误，这是专用命名空间，不是ext4盘。
期望输出`[native-kv] PASS ... 8 physical pages`；这验证FTL请求闭环，不代表Attention正确性。
测试后 `sync; /sbin/poweroff -f` 关闭这台临时测试机。

### 12.6 下一步而不是本次已完成项

1. 先确认上面的Guest测试通过，记录使用的QEMU/内核版本。
2. 为native加入受控FDM/compute资源与内部KV IO提交路径；不能让旧grouped继续拿FIEMAP LPN旁路访问。
3. 迁移有界tail/完整页后台发布：设备维护K双布局与V、pending页和目标deadline，保留QK/V→PV依赖。
4. 实现任务结束/reclaim/块擦除与generation更新，再接SparF小数值和OPT；最后才做论文规模与精度对照。

## 13. 2026-09-22：仅修复我们扩展引入的三项问题

范围不包含原生共享统计 memcpy 长度、原生 FTL/poller 的退出重构、freezing 或 runtime 校准。

- Native 完成队列：满 CQ 的到期请求暂存在本轮局部链表，其他 CQ 继续完成，结束后再放回
  deadline 优先队列。不改 deadline、不重复计算 PCIe，不新增正常路径堆分配。
- Grouped 清理：显式 DRAIN/CLOSE 保留持久化与目标时间等待；底层 free/UNLOAD 改为取消
  未发布写回，不再隐式 drain。后台使用原子取消标志中止未来 deadline 等待，worker 停止后
  不发布 committed；已提交 FTL 请求仍须返回后才释放内存和 MRS 引用。直接卸载未排空
  会话会警告，不能作为持久化完成。未声称修复原生线程退出或强行取消任意底层 IO。
  Grouped 后台请求带私有取消标志，FTL 限速队列允许取出已取消请求并回送失败完成，
  不再等待未来虚拟时间；已在执行的复制先结束。普通请求初始化标志为空，调度行为不变。
- CUDA 复制：旧内部 backend 入口返回 checked IO 状态，D2H 失败不覆盖目标数据；FTL 的
  NVM/FDM 路径把失败记入 req->status，而不是作为纳秒延迟或默认为 NVME_SUCCESS。
  FDM 同步复制的新错误分支也明确设置请求状态并交回完成路径。

新增回归：双 CQ 背压与 deadline、未满尾部取消、冻结目标时钟取消、不提前释放在途
传输、正常 DRAIN 等待和发布，以及真实 backend/FTL 函数的 CUDA 失败注入。
失败注入使用模拟 cudaMemcpy，不访问真实 GPU。QEMU qtest 使用独立临时后端，
不接触现有 Guest、镜像和 GPU。Linux Guest ioctl/OPT 集成仍待用户实际运行。

本批验证：QEMU 增量构建成功；88 项 Host 回归与 4 项真实 QEMU qtest 全部通过；
`git diff --check` 通过。Host grouped Attention 测试使用已有 Python 3.10 `_cemu_client`
扩展完成导入，实际测试后端为隔离 C harness，不冒充 Guest 驱动或真实 CUDA 验证。
构建仍有此前的 unused-function/unused-variable 警告，未顺带调整无关代码。

## 14. Guest 首测入口修正

旧 Guest 初始化在专用测试机运行 fio/挂 FDM/挂模型共享，出现 ENOSPC 和 FDM size=0。
native 配置仅提供 16MiB 专用 NVM，不提供 FDM/模型共享，不能使用旧挂载初始化。
启动脚本改为直接启动 bash，仍使用临时 OS overlay，不改原 OS 镜像启动文件。

当前 linux-cemu 的 `nvme_ioctl()` 把块设备 ioctl 硬编码转到 nsid=3：
ns3 不存在时可发生 NULL Oops；存在时 native 客户端的 nsid=1 被拒绝为 EINVAL。
这发生在 INFO 阶段，不是 KV 映射验证失败；不能通过把命令 nsid 改成 3 解决。
改走 `nvme_ns_chr_ioctl()` 对应的 `/dev/ng0n1`，不修改原生内核。
客户端拒绝非字符设备，Makefile 和文档同步更正，INFO magic 检查仍先于任何写入。
发生 Oops 的 Guest 不继续使用，停止后重新启动独立测试机。新的 Guest 路径仍待实测。

用户随后反馈 `/dev/ng0n1` 的 `[native-kv] PASS ... 8 physical pages`。
其中 opcode=1 错误是协议负向测试主动提交普通 LBA WRITE 并验证被拒绝，不是数据验证失败。

## 15. 设备内部读写与批量提交入口

本批把 Guest 命令和后续设备自治调用收敛到同一入口，而非另建映射/虚拟 NAND 时钟。
`hw/femu/bbssd/instattention_device.c` 提供：

- `bb_native_kv_register`：按 controller/ns/task/generation 注册几何。
- `bb_native_kv_transfer`：按 batch/layer/head/kind/token/channel 读写一页；无 LPN/FIEMAP。
- `bb_native_kv_batch`：最多256页、同task、读或写的批量提交，一次取得 owner 锁；
  每页由原映射分配/查询 PPA，不要求连续LBA或连续物理地址，不把相邻页合并成一次NAND操作。
- `bb_native_kv_snapshot`：一致地取得映射统计及已提交IO的最大目标deadline，供 INFO/FLUSH。

Guest REGISTER/READ/WRITE/INFO/FLUSH 已改用上述入口；内部调用不经过 Guest ioctl、
PRP、外部完成队列，也不重复添加外部PCIe传输时延。旧 `ftl_mode=femu` 的映射/GC不改。

### 数据完成与目标完成是两个条件

接口返回成功表示真实页字节复制及目标 NAND 调度已完成，**不是虚拟时间已到**。
调用者可以提前得到数据，但只可在 `result.ready_ns` 满足后发布模拟完成/committed，
或将它作为下一阶段的目标起点。Guest 仍由原 CQ deadline 路径延迟完成。
接口不推进虚拟时间、不等待目标deadline、不改freezing；锁只覆盖真实操作和共享状态。
同一批的独立页使用同一个请求start，不把上一页ready当下一页start；同LUN竞争仍由
`ssd_advance_status` 串行化，不同channel/LUN允许目标时间重叠。
namespace最大deadline仅供FLUSH，不能拿来串行化所有读写。

每个namespace一把锁保护映射、IO scratch、NAND资源和deadline，防止后续写回线程与
Guest/Attention并发使用原单线程对象。热路径不额外堆分配；batch大小有界。
这是数据/建模临界区，并非声称真实宿主机页复制已经并行，也不是后台写回线程已经迁移。

### 失败和生命周期契约

- batch不是原子事务：首个失败即停，`completed_pages` 为成功前缀；成功页不回滚。
  下一项为失败页，剩余页未提交。调用者不得把整个batch发布为成功。
- 单页/批量结果在实际执行后给出deadline；入参早期拒绝时结果未定义，必须先看status。
  传输失败保留已占用NAND时间，失败页不变成可读成功页；参数拒绝不推进namespace deadline。
- buffer及请求数组由调用者持有至接口返回。返回后没有内部悬挂buffer引用。
- `bb_native_io_destroy` 在同一锁下关闭IO；尚未释放的namespace上的后续入口返回INVALID。
  namespace最终free/mutex_clear前，owner仍必须停止并join所有生产者，不能与新调用并发free。
  当前只有Guest worker；将来接缓存worker时必须先落实其join顺序，不以这把锁替代生命周期管理。

### 验证与未完成项

新 `test_instattention_device_io.c` 链接生产 FTL、backend 和内部入口，使用临时文件：
验证两channel目标重叠、写→读依赖、16线程共享读写与抢写同页、批量上限/部分失败、
失败不覆盖输出、关闭后拒绝，以及参数拒绝不污染deadline。虚拟时钟始终100ns，
写目标设为秒级，接口仍及时返回，证明没有在内部等待虚拟时钟。
测试入口为 `make instattention-native-qemu-io-test`（Host，需QEMU构建配置）。

尚未把有界tail/background worker、FDM/compute、SparF/OPT接入native模式。
下一批将把缓存的三种committed前缀和整页pack接到批量接口，之后再迁移稀疏读及Attention；
不得仅替换旧pool偏移，不能把本批入口测试当作端到端复现完成。

本批实际验证：QEMU 增量构建通过；92项Host回归（含旧grouped/Attention与新内部入口）
和4项隔离QEMU qtest通过；`git diff --check`通过。未操作用户现有VM/GPU/后端镜像，
未修改linux-cemu。构建保留原有unused-variable警告。本批新版二进制未另跑Linux Guest。

## 16. Native 有界缓存与整页后台写回

### 本批新增通路

`instattention_buffer.c/.h` 实现独立native缓存owner，使用第15节的批量接口：

```text
Guest BUFFER_APPEND（KV坐标 + 新K/V）
  -> 有界K/V环形tail -> accepted
  -> 后台worker按FIFO合批、完整页pack
  -> bb_native_kv_batch -> KV映射 -> PPA -> 同一NAND资源/真实backend
  -> 真实传输成功且目标deadline已到 -> 三种committed前缀分别推进
```

保留旧group buffer的环形复用、FIFO队列、K双布局/V单布局和整页pack规则；未直接复用
旧实现的文件pool ordinal目录、FIEMAP计划和额外PPA表。native映射仍是唯一物理地址权威。
旧grouped代码不改，不把旧OPT测试自动切到此路径。

缓存按每个(batch, layer, head)流维护accepted及token-K/channel-K/token-V三个committed。
K容量为两个channel组，V容量为两个token组；只有相应布局的写回到期后才允许覆盖tail。
FP16 D128/C8/4KiB下，token组16个、channel组256个。内存不随Decode步数增加。
容量预算包括tail、队列、stream状态、写scratch、读scratch、页描述及pending描述；
不把宿主线程栈/GLib线程管理开销声称为CSD DRAM容量。当前Guest控制面预算上限64MiB。
这批tail由有界Host内存承载CSD DRAM模型，还没有接入FDM namespace/MRS，也未校准DRAM传输延迟。

### 非阻塞与时序规则

- APPEND空间不足返回PENDING（core=5），不部分接受、不占住Guest命令线程等虚拟时间；
  客户端重新提交相同packet。其他错误不重试。
- 后台worker实际写入后在自己的线程等待目标deadline；等待释放buffer锁，允许tail读取及
  其他流APPEND。沿用旧目标等待的10us检查间隔，并允许stop立即唤醒；没有新增1ms等待下限。
  Guest线程/算子不在这个等待中，freezing机制不改。该轮询仍有宿主调度唤醒误差，非时延校准结论。
- 未committed的数据从tail复制，零NAND读；已committed的数据按KV坐标走新FTL整页读。
  token读返回整行，channel读当前只返回单个元素；稀疏批量页去重/列过滤仍是下一批事项，
  不能用这一诊断读API的逐元素IO次数代表最终SparF读放大或性能。
- 显式DRAIN是终结操作，禁止后续APPEND，残组补零整页写入并登记有效token数。
  DRAIN返回PENDING直至所有布局目标完成，再返回OK。NVMe FLUSH也不能绕过未完成tail返回成功。
- IO失败永久poison该buffer，不发布整批成功；已发生的部分NAND写及资源占用不回滚。
- reset/unrealize先停Guest worker，再取消buffer未来deadline等待并join真实在途复制，
  最后才释放tail、映射和backend。未排空的buffer停机后标为failed，不在reset后假装恢复；
  正常完成必须先显式DRAIN。namespace最终销毁之外不并发free该owner。

### Guest协议扩展（ABI v1新opcode，旧命令不变）

| opcode | 操作 | 数据 |
| --- | --- | --- |
| 0xc9 | BUFFER_REGISTER | 旧32字节geometry + u64预算 + u32 batch_pages，余字节零 |
| 0xcd | BUFFER_APPEND | DWord10高16位count；payload为count行K再count行V，总长不超过4KiB |
| 0xce | BUFFER_READ | 坐标指定token行或channel元素，结果放在4KiB返回包开头 |
| 0xca | BUFFER_STATUS | 指定流，返回11个u64：accepted/三个committed/预算使用/写字节/读字节/批数/pending deadline/sealed/failed |
| 0xcc | BUFFER_DRAIN | 指定task/generation，无数据包，PENDING可重试 |

身份/坐标仍使用旧DWord10～15布局。PENDING使用命令序列状态加core=5，IO_FAILED使用
内部设备错误；错误不是正常成功。对buffer拥有的task拒绝旧raw READ/WRITE，防止绕过tail及失败状态。
STATUS中accepted/committed属于指定流，内存和读写统计属于整个buffer会话。
当前每namespace只允许一个buffer task，其中可含多batch/layer/head；没有CLOSE/reclaim，
同一VM不重复REGISTER新的buffer测试会话，需重新启动隔离测试机。

### 验证

Host C harness链接真实新FTL/backend，覆盖：预算拒绝不注册task、首15token零NVM读写、
目标时钟冻结时committed不推进/APPEND背压但tail可读、两流531token、15/16及255/256边界、
混合Flash/tail读取、ring复用、三布局尾部drain、后台IO失败不发布，以及停止时不等未来虚拟时间。
QEMU qtest新增真实NVMe后台buffer协议和未到期写回时unrealize；临时后端，无现有VM/GPU访问。
两流531token最终写入232整页=950272字节；内存统计420392字节（本机ABI），全程不增长。

Guest待用户验证：停止之前的专用测试VM，Host重新执行
`bash femu-scripts/run-native-kv-test.sh`，按第12.5节挂代码后：

```bash
cd /root/CEMU/tests/cemu
make instattention-native-buffer-device-test NVM_DEVICE=/dev/ng0n1
```

期望`[native-buffer] PASS`。无需GPU、模型、ratio、FDM挂载或mkfs；不使用旧实验启动脚本。
仍未接FDM/计算namespace、自治SparF阶段、稀疏读去重、OPT、任务回收/GC；下一步接稀疏读
与Attention前仍须保留本批读取/背压/目标完成的契约，不能以本小测试代替端到端行为验收。

最终验证：QEMU增量构建通过；95项Host回归及6项隔离QEMU qtest全部通过；
`git diff --check`通过，linux-cemu无改动。没有启动/停止用户的现有VM、操作GPU、
改freezing或复用实验后端。Linux Guest的新buffer命令仍待用户运行；原有unused警告保留。

## 17. Native 稀疏批量读取（2026-09-22）

第16节的Linux Guest buffer测试随后由用户运行并报告PASS：两流531token，
15/16、255/256边界、ring复用及partial drain通过。该版内存统计为420392字节。
本节新增读取工作区后统计会增加，不应再把这个历史数值作为新版本硬编码验收值。

### 接口及数据路径

新增`native_kv_buffer_gather`设备内部接口，位于`instattention_buffer.c/.h`。
输入是最多256个`NativeKvGatherRange { coordinate, tokens }`，每段指定起始KV坐标
和连续token数量；不同段可来自不同batch/layer/head/kind，允许乱序及重复选择。

- token-K/token-V段返回连续完整行，channel必须为0。
- channel-K段返回指定一个channel的连续token元素；多个top-r channel用多段表达。
- 输出按输入段顺序紧密拼接，每段内部token递增；重复选择重复输出，但同一Flash页只读一次。
- 先整体校验accepted、坐标、长度与容量，再按原生KV页key建立去重计划；
  未committed部分留在tail，不会因物理文件已有数据就提前当成Flash完成。
- 已committed部分通过`bb_native_kv_batch(read)`进入唯一native映射/NAND/backend，
  所有页使用相同start_ns，ready_ns取本批资源调度结果的最大值，不线性相加页延迟。
- 全部页读成功才把Flash切片和tail元素写到调用者输出；IO中途失败保持输出不变，
  但已成功读取的物理字节/页计数及失败造成的NAND占用不回滚。

接口返回`NativeKvGatherResult`：ready_ns、实际完成Flash字节/页数、输出字节、tail输出字节。
失败时output_bytes为0；参数/容量预检拒绝不发NAND读、不污染deadline。
实际内存复制完成不等于目标模拟时间已经到达；未来Attention调度必须等ready_ns后
才在目标时间轴上使用结果，不能把Host函数返回当作目标完成。接口不等待虚拟时钟、
不额外收取Guest PCIe时间、不改freezing；tail延迟沿用原模型，本批不声称DRAM校准完成。

### 并发、容量与边界

buffer创建时预留独立读取scratch及页描述/key表，计入原budget；不与后台写scratch共用，
不按Decode步数增长，也不在每次gather时malloc。读取工作区页数等于注册的batch_pages。
每次最多读取batch_pages个不同Flash页，输出字节也不能超过batch_pages × page_bytes。
超限返回LIMIT，不能静默截断或自动改成逐元素读取；未来调度调用方需显式分批。
例如batch_pages=32时，跨64个不同Flash页的top-r选择不能作为一批提交。
该上限是模拟器当前实现约束，不声称是论文硬件的请求粒度。

校验、计划、真实读和scatter持有buffer锁，稳定accepted/committed及tail复用状态；
内部NAND入口再持有namespace IO锁。后台写入口释放IO锁后才取buffer锁，无反向嵌套。
本批未优化真实拷贝期间的buffer锁竞争，不把正确性去重实现声称为最终Host性能设计。
调用者负责ranges/output生命周期，二者不得相互重叠或被并发修改；owner释放仍需先停生产者。

### 本批验证及尚未接入

扩展`test_instattention_device_io.c`，直接链接生产FTL、真实临时文件后端和gather实现：

- 首15token全tail、零Flash读取；后续混合Flash/tail及ring复用。
- channel跨255/256、token跨15/16边界；乱序/重复选择和双stream隔离。
- 7段混合请求合并为7个唯一Flash页，结果逐元素与原始输入一致。
- 非法坐标、未accepted token、错误输出长度、零段/零长度、段数超限、
  唯一页数及输出容量超限都在IO前拒绝。
- 第一个页读成功、第二页backend范围失败时：计入4096读取字节，输出保持哨兵内容；
  恢复后可正常重试。失败/停止的buffer拒绝gather。
- 虚拟时钟在调用期间不推进，接口仍返回未来ready_ns，不在内部阻塞等待。

Host入口仍为`make instattention-native-qemu-io-test`，无需GPU或用户VM。
本批没有新增Guest gather opcode；现有Guest BUFFER_READ仍是诊断单行/单元素接口。
尚未把SparF的top-r/top-k阶段接到此接口，不能据此宣称OPT已经走native稀疏读路径。
下一步需要迁移native模式的计算资源入口及自治Attention读计划，明确分批容量和目标依赖；
旧femu/grouped路径保留，任务回收/GC也仍未完成。

本批实际验证：QEMU增量构建、95项Host回归、6项隔离QEMU qtest及`git diff --check`
全部通过。新buffer固定预算使用559168字节（本机ABI，原参数batch_pages=32），
最终NVM写字节仍为950272。qtest是原Guest协议回归，新增gather的数值/去重/失败验证
由生产C harness执行，不能混称Guest已测过批量gather。未修改linux-cemu、旧FTL路径或
冻结机制，未访问现有VM/GPU/实验镜像；构建仅保留原有unused警告。

## 18. Native FTL → SparF 状态机/算子桥接（2026-09-22）

### 本批接到哪里

新增`hw/femu/bbssd/instattention_attention.c/.h`，把上节gather接到已有
`hw/femu/csd/sparf_workflow.h`的自治状态机，不复制另一套Top-R/Top-K/QK/PV算法。
`include/hw/femu/sparf_storage.h`定义设备内部存储回调；没有绑定回调时，
旧femu/grouped路径仍使用原`sparf_start_grouped_read`。

实际Host回归路径为：

```text
原始K/V → native APPEND → 有界tail/后台group发布 → native映射/NAND/临时文件
Q → 原SparF Top-R → native channel-K gather → 原近似分数/Top-K
  → native token-K gather → 原QK/Softmax
  → native token-V gather → 原PV → output
```

每次Attention先以显式task/generation及batch区间绑定buffer，验证层数、KV头数、
dtype、head_dim及accepted长度。内部复用现有自治header的GROUPED标志表示设备管理读取，
**不把旧group cookie解释为native任务ID，也没有新增Guest可调用的native Attention命令**。
后续Guest协议接入仍需显式区分两种身份，不能仅切换启动配置或复用旧group注册。

读取回调将query head按GQA关系映射到KV head，保留batch/layer和选择索引的输出顺序。
channel-K按列、token-K/V按行调用gather；V返回布局保留原算法的mean槽，由原状态机填入。
因此读取不是从Guest复制整份KV后假装走FTL，测试还把旧current-K/V MR填成NaN，
验证该路径不会误用Guest current输入。value mean仍由调用方提供，本批未迁移它的维护。

### 读取批次与目标依赖

每批最多256个range；超过gather工作区/页数容量时递归二分range或连续token段，
直到适配预分配空间，不逐元素下发Guest命令、不截断选择、不每次malloc。
仅单个gather内去重；跨子批重复Flash页会实际重读并计费，目前未做跨批缓存。
单阶段输出限制64MiB，这是当前实现边界而非论文硬件参数。

同阶段的子批以同一目标start提交，NAND资源状态决定排队和并行，返回最大ready_ns，
不能把每页延迟简单相加。原状态机继续维护：channel-K完成→近似计算→K完成，
QK和V读使用共同目标起点，PV起点取`max(QK完成, V读取完成)`。
生产`SparfRead`增加内部读取结果类型，把回调返回的目标时长交回原等待逻辑。

这里真实Host gather仍同步完成实际拷贝后返回目标deadline；回归证明的是**数值与目标依赖**，
不是Host物理IO/CUDA kernel已经并行，更不是FPGA时延校准。未改变freezing、CU调度器，
没有把Host耗时直接当作目标runtime。回调失败则整阶段失败；若后续子批失败，前面子批
可能已改写输出，状态机不得继续使用；错误路径仍标记Host mirror dirty，禁止返回假成功。

### 在途Attention保护

`native_kv_buffer_acquire/release`为本次layer/batch区间的所有KV头取得读取租约：

- 必须准确匹配已accepted token数，且task/generation、几何参数有效。
- 租约期间相同stream的APPEND返回PENDING，避免后续token使tail环复用；其他stream可继续追加。
- 已排队后台完整group写回仍可继续，并可从tail读取切换到已发布Flash页。
- 存在租约时DRAIN返回PENDING且不提前切换draining状态。
- stop拒绝新租约、停止写回后等待现有租约释放，防止buffer被释放后Attention继续访问。

租约对象必须零初始化、不可复制后重复释放；每条成功、IO失败、算子失败、取消路径都须close。
owner须先取消/收拢Attention，再释放buffer；不能同一线程持有租约又同步stop而不释放。
新增停机测试由另一线程stop，验证确实等到Attention释放才返回。

### 验证范围和后续入口

新增`test_instattention_native_attention.py`和生产C harness，直接链接native映射、
buffer、NAND和文件backend，复用既有自治状态机测试设施及真实CPU SparF算子：

- FP16：15/16、253/256/259/531 token，双batch、双layer，Qheads=4/KVheads=2，ratio=1/8。
- FP32、非零batch起始偏移、全tail/混合Flash-tail/最终drain后全Flash。
- 超容量显式拆批、输出顺序、Flash实际字节/页数、QK/V→PV目标依赖。
- stale generation、错误accepted、同stream APPEND与DRAIN保护、独立stream追加。
- 三种读阶段失败、QK/PV算子失败、多处状态机取消后均释放租约，可重新执行。

FP16数值回归使用绝对值唯一且可精确表示的Q，避免Top-R边界并列时NumPy argpartition与
生产索引tie-break选出不同channel；没有因此放宽容差或改生产选择算法。

**尚未完成：** native模式仍只暴露NVM namespace，未开放FDM/compute/MRS；本批回调绑定
由Host harness完成，Guest命令分发尚未绑定。因此不能声称Transformer已走新FTL，
旧OPT测试仍走旧后端。下一步应接native计算资源生命周期和命令身份，再迁移Python
Prefill APPEND/Decode提交/DRAIN，并做隔离Guest小模型端到端验证；不要直接启动OPT-13B。
同时还需解除小测试的映射/内存规模限制；本批仍未实现GC、任务回收或重启恢复。

最终验证：QEMU增量构建通过；131项Host回归（含新增4项native Attention测试，内部含
多组数值/故障子用例）与6项隔离QEMU协议测试通过；`git diff --check`通过。
既有unused警告未处理。linux-cemu未修改；未操作现有VM、GPU或实验镜像。

## 19. Native 计算入口与功能基线收尾（2026-09-22～23）

本节更新第18节末尾的待接入状态；不把旧grouped的OPT验收结果转记到native模式。

### 昨天接通的部分

- `native_csd_smoke.json`启用NVM(1)、FDM(2)、compute(3)三个namespace；
  `native_kv_smoke.json`仍保留仅NVM的小测试。默认FTL仍是旧FEMU，native须显式选择。
- native计算模式接受program LOAD/ACTIVATE、FDM MRS、EXECUTE；MRS只允许绑定FDM，
  不能把native KV namespace解释成连续LBA内存。
- `CEMU_SPARF_WORKFLOW_NATIVE`显式标记metadata末尾的task/generation/batch身份。
  compute入口先取得native读取租约，再把SparF内部读取交给native gather；
  正常结束、失败及取消均释放租约，不改变原freezing机制。
- 设备reset/stop先收拢在途计算和后台IO，再释放队列及资源。qtest覆盖尚未达到目标deadline
  时的reset、CQ背压、非法MRS与启动保护，而不只覆盖正常结束。
- `NativeKvStore`通过原生KV命令REGISTER/APPEND/STATUS/DRAIN管理生命周期；
  明确检查几何参数、容量、accepted前缀、失败后禁止继续追加，以及单次会话约束。
- `test_opt_cemu_pipeline.py`已增加`--sparf-native`、`--native-kv-device`、
  `--native-kv-budget`，复用公共runner和`SparfGroupedAttentionBackend`，没有复制一套OPT。
  该分支要求functional + autonomous SparF，不能与旧grouped/Guest buffer写回模式混用。

### 今天补齐的功能证据

`test_native_csd_pipeline.py`现包含两条隔离真实QEMU路径：

1. 随机KV经native APPEND进入tail/Flash，FDM Q/mean及MRS送入CPU SparF，读回输出；
   FP16、GQA、ratio=1/8，15/17/259 token；检查trace目标时间与在途reset。
2. 小型合成OPT checkpoint：2层、batch=2、heads=2、head_dim=128、FP32，
   prompt=127、Decode=3。生产Prefill writer实际APPEND，Decode QKV实际提交CSD，
   读回Attention后执行Wo/MLP/LM head，再把生成token送入下一步。
   对照纯PyTorch分支逐步检查logits和hidden（atol/rtol均2e-5）、token完全一致；
   确认6次Attention、130个accepted token、真实NVM读写及最终sealed且failed=0。

第二条使用ratio=1，以便用完整Attention验证模型链路；稀疏ratio=8的独立数值测试仍保留。
FP32对应token group=8、channel group=128，该测试跨越组边界并读到Flash和tail，
不把FP16时的16/256边界硬套到FP32。

测试复用`CemuSparfDevice`的metadata、提交和输出解析；其最底层FDM/ioctl传输换成
qtest的真实NVMe命令。**不是Guest Linux/FDMFS/GPU实测，也不是下载的OPT-125M/13B**。
它补上了公共Transformer流程到native设备的功能证据，不能据此宣称论文复现全部完成。

新增统一入口`make instattention-native-regression-test`，保留旧address/mapping/IO命令
作为契约套件别名。2026-09-23：QEMU增量构建通过；100项回归通过，包含native契约、
IO、故障处理、真实QEMU、OPT公共runner、旧grouped与阶段状态机。测试只使用CPU和临时文件；
linux-cemu无修改。首次Host运行因使用Guest Python 3.13扩展而导入失败，随后改用现有
Host Python 3.10扩展重跑通过；未覆盖Guest扩展文件。

### 保存基线后仍需完成

1. 新建独立native CSD启动配置，按真实小模型核对NVM物理页、页表上限、tail预算和FDM容量，
   验收Guest OPT-125M的Prefill、多步Decode和DRAIN，再上OPT-13B。
   `native_csd_smoke.json`只有16MiB NVM/8MiB FDM，不是论文实验配置。
2. 保持native backend新文件保护；不执行旧`cemu-mount.sh`中的mkfs/fio初始化。
   `/dev/ng0n1`负责KV命令，FDM/compute仍走各自namespace；不能以NVM block节点替代。
3. GC、任务回收、重启恢复仍未完成，当前会话不能无限复用；默认页表上限65536页、
   native buffer预算64MiB，不能直接照搬旧grouped大实验的容量假设。
4. 多microbatch/GPU真实异步、论文配置、目标runtime和时间精度需分别验收。
   CPU qtest功能通过不代表这些指标已对齐；不以A100耗时替代FPGA目标时间。
