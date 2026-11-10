# FDM 同步写入、设备多 input 槽位与常驻聚合设计

日期：2026-11-10。状态：设计与代码修改路线，尚未实现。

本文整理当前讨论，不表示代码已具备多槽位或常驻轮询功能。本次只整理文档，不修改内核、客户端、算子或设备实现。

## 1. 当前阶段的决定

先保留同步 `pwrite → FDMFS → 块层 → NVMe → FDM input` 路径，只改变聚合执行方式和 input 管理方式。

1. 普通成功的 KV pwrite 返回表示整包已经进入设备 input、校验通过且接收模型时间满足，不等待聚合和 NAND 写回。
2. 后台聚合执行部件初始化时启动，常驻检查就绪任务，不再依赖每包条件变量唤醒。
3. input 扩成 N 个固定跨度 W 的槽位，第一版从 N=2 开始；设备独占分配、占用和回收职责。
4. 应用继续向注册的 KV input 文件偏移 0 写完整包，不选择槽位，不增加一个应用侧传输编号。
5. FDMFS 为整次写建立独立上下文，由内核生成 transfer_id，并传递给这次写拆出的所有 NVMe 片段。
6. 设备在 DMA 前按 transfer_id 领取或查找槽位，直接把片段写入该槽位，不增加公共入口到槽位的一次整包拷贝。
7. 同步路径先验收；io_uring 文件写留到后续阶段，但身份、拆分和槽位协议不能只依赖“当前只有一个写请求”。

本阶段不实现特殊的“pwrite 提交即返回”，不新增用户可见包级 SQ/CQ，不让 FDMFS 维护设备槽位空闲表，不修改 native KV APPEND 路径、KV NAND 放置规则、Attention 数学或 NAND 模型参数。

旧文档 [FDM_ASYNC_PWRITE_DESIGN_2026-11-04.md](FDM_ASYNC_PWRITE_DESIGN_2026-11-04.md) 保留为历史记录。后续异步路径是否延迟写完成到聚合结束，应作为显式策略另行决定，不能根据是否使用 io_uring 隐式改变设备协议。

## 2. 三个完成时刻与资源所有权

| 时刻 | 含义 | 可以释放的资源 |
|---|---|---|
| 接收完成 | 整包 DMA 完成，包合法，所有片段模型 deadline 已满足 | 全部子 I/O 完成后，pwrite 返回，应用可释放 Host 源缓冲区 |
| 聚合消费完成 | 聚合写入暂存布局、发布 KV 可读进度，且已结束对 input 的访问 | 设备可以回收对应 input 槽位 |
| NAND 写回完成 | 暂存页的写回模型完成，相关读者和引用允许回收 | 暂存页池资源 |

三个时刻不能混用。写 CQE 不表示 input 槽位空闲；input 空闲也不代表数据已写入 NAND。

同步 pwrite 成功前必须完成该次写的所有子 I/O。可以保留使整包收齐的那条命令，等包校验和接收 deadline 满足后再完成；不必扣住每条片段命令，更不能等待聚合才完成普通 APPEND。

该原始请求一旦归还 poller 并回收，后台不能继续引用它。后台任务需要自己的固定描述符和 trace 快照。槽位复用还要等待关联 DMA、接收回调和需要读取槽位内容的完成路径全部退出；不能只看算子返回。

## 3. 总体流程

```text
应用打包 KV，保留当前源缓冲区
    │ pwrite(input_fd, packet, bytes, 0)
    ▼
FDMFS：识别注册的 KV input 写入
    │ 创建写上下文，生成 transfer_id，记录整包长度
    ▼
iomap：构造多个 bio；每个 bio 带相同传输身份
    ▼
CEMU 块设备／NVMe 驱动：拆分并保留身份，构造带标记的 MEMORY_WRITE
    ▼
设备：用 binding + transfer_id 查找槽位
    ├─ 已分配：接收该包剩余片段
    ├─ 未分配且有空位：领取槽位
    └─ 未分配且无空位：背压，不能覆盖旧槽位
    ▼
直接 DMA 到槽位基址 + 包内偏移
    │ 更新该槽位的片段状态、校验包头、整包和 deadline
    ▼
发布 READY，允许原写请求完成
    ├─ NVMe CQ → bio/DIO 完成 → pwrite 返回
    └─ 常驻聚合部件取任务 → 聚合／发布 KV 就绪 → 设备回收槽位
                                             └─ NAND writer 继续按原路径写回
```

没有应用额外 EXEC 来启动聚合，没有每包槽位查询，也没有“先 DMA 到固定入口再复制到实际槽位”。

## 4. 已核查的现有代码及需要拆除的单槽假设

代码位置以本次检查的源码为准，后续行号会变化，函数名为主要定位依据。

| 现有位置 | 当前行为 | 新方案需要改变什么 |
|---|---|---|
| `linux-cemu/fs/fdmfs/file.c`：`fdmfs_rw_iter()` | 调用 iomap DIO，无写包上下文 | 为注册 input 写创建传输上下文，普通 FDM 不变 |
| `linux-cemu/fs/fdmfs/inode.c`：`fdmfs_iomap_begin()` | `region->off + offset` | 保留逻辑提交入口；不是内核选择真实槽位 |
| `linux-cemu/drivers/nvme/host/cemu.c`：`cemu_bdev_submit_bio()` | 设置 MEMORY 标记并按队列限制拆分 | 身份标记随拆分传播 |
| `linux-cemu/drivers/nvme/host/pci.c`：`nvme_prep_rq()` | 标准读写命令转换成 MEMORY_READ/WRITE | 转换之后填写 KV 专用扩展字段 |
| `CEMU/hw/femu/nvme-io.c`：`nvme_process_sq_io()` | input 聚合忙时停在 SQ head | 按包和空闲槽位判断，不再用全局 aggregate_active 拦截所有 input |
| `CEMU/hw/femu/csd/memory.c`：`memory_write()` | 直接 DMA 到 sb，调用单份 receive | DMA 前解析包身份、槽位映射，使用槽位独立 receive |
| `CEMU/hw/femu/fdm_kv_binding.h` | 一份 pending_write、aggregate_request、aggregate_active | 改为每槽任务、接收完成关联、就绪任务管理 |
| `CEMU/hw/femu/fdm_kv_buffer_abi.h` | 单个 input 指针和 receive 指针 | 槽位数组、容量、每槽状态及独立进度 |
| `CEMU/hw/femu/fdm_kv_receive.c` | next_request_id 在聚合后 release 时递增 | 接收顺序、包身份和聚合顺序分离 |
| `CEMU/hw/femu/fdm_kv_binding.c`：`complete_write_locked()` | 提前 ACK 前要求 head.accepted 等于 token_first | 接收可以领先聚合；不能用聚合进度拒绝合法后继包 |
| 同文件 `aggregate_worker()` | 无任务时 g_cond_wait，每包唤醒 | 常驻执行并检查小型就绪描述符，不扫描 payload |
| `CEMU/tests/cemu/kernel/fdm_kv_aggregate.c` | 单次处理当前 buffer->receive/input，可分轮 | 显式接收槽位上下文，保留每槽分轮进度 |

当前内核的 `__bio_clone()` 已复制 `BIO_NVME_MEMORY_CMD` 标志，这是扩展身份传播时必须继续审查的地方，不代表新增字段会自动复制。

## 5. 传输身份与命令协议

### 5.1 两种编号不能混淆

- `transfer_id`：内核生成，标识一次完整文件写，用来把所有传输片段归到同一设备槽位。
- 包头现有 `request_id`：上层已有的包序号，与 session/generation、KV 坐标一起用于算法顺序和重复包校验。

应用无需新增 transfer_id；也不能直接令它等于包头 request_id，因为内核第一版不应为了给片段编号而读取、修改整个 payload。

建议使用控制器范围的非零 64 位单调 transfer_id，绑定生命周期内绝不重复；回绕失败而不是复用。控制器 reset/rebind 必须先终止旧 I/O。不能用线程 ID、CID 或内核指针代替。若后续允许未完全静止的 epoch 切换，必须扩展携带 epoch 的协议，不能依靠编号碰巧不重复。

### 5.2 每个片段至少携带的信息

| 信息 | 用途 |
|---|---|
| 已注册逻辑入口／binding | 识别属于哪块设备、哪个 KV input 会话 |
| transfer_id | 首片段领取槽位，后续片段命中同一槽位 |
| 包内偏移 | 即便包头未先到，也能确定 DMA 位置 |
| 片段长度 | 本命令 DMA 范围 |
| 整次写入长度 | 提前校验槽位容量、片段边界，并在包头到达后核对 wire_bytes |

不在每个片段前插小包头，不额外 memcpy 全部 KV。字段只进入 NVMe 命令控制信息。

### 5.3 可用字段及推荐布局草案

已检查普通 MEMORY_WRITE 的尾部 `rsvd13[3]`：设备不读取其业务含义，驱动转换时没有为 FDM 专门赋值。它们位于 CDW13～15，共 12 字节。

为了支持包头后到，推荐第一版草案为：CDW13～14 保存 64 位 transfer_id，CDW15 保存 32 位整包长度；sb 仍表示逻辑入口基址加包内偏移，wl 为片段长度。整包长度受 W 和 UINT32_MAX 双重限制，长度计算先用 64 位检查溢出。字段使用明确的小端编码，不能通过非对齐 uint64_t 指针直接存取 CDW13。

这与之前口头提出的“64 位编号 + 32 位版本”不同：此草案把整包长度带到每个片段，模式／版本放在初始化绑定协商中。最终 ABI 需在实现前冻结；当前仅是推荐，不是已实现接口。

绑定必须显式声明 tagged-input 模式，仅对该注册区域的 MEMORY_WRITE 解释扩展字段；绑定不支持该版本则初始化失败，不允许静默退回旧格式。旧格式在 legacy 绑定上继续工作。普通 Q/mean 写、普通 FDM 读和 MEMORY_COPY 不解释这些字段。

重要限制：

- 标准读写构造先写入 dsmgmt/reftag/apptag/appmask，占据相同位置；只能在转换为 FDM 命令之后显式重写这 12 字节，且每次准备／重试都一致。
- MEMORY_COPY 的 rsvd14 在驱动中存有待释放描述符指针，绝不能覆盖。
- 不用 PRP/MPTR 塞包身份，不改变 DMA 数据指针含义，也不增大当前 64 字节命令。
- 注册为 tagged-input 的区域，拒绝无合法标签的命令，避免把普通命令残留值当成 transfer_id。

## 6. 内核代码怎么改

### 6.1 FDMFS 注册与写入口

拟增加专用 input 注册接口：关联 file/inode 的 FDM 区域、控制器、设备绑定代数、逻辑入口、最大包长 W 和协议版本。它不包含设备槽位空闲表，也不在每包时申请设备空间。

现有设备 bind 使用客户端构造的 passthrough 命令，不会自动把“这是 KV input 文件”的信息同步给 FDMFS。需要新增文件注册步骤或一个可核验的联合绑定接口；这是必须补上的初始化链路。

推荐初始化事务：创建文件/MRS → 设备成功建立 tagged 多槽绑定 → 内核验证该绑定并注册 input 文件 → 才允许写入。失败按逆序回滚；禁止用户只提供一个任意 sb 就冒充已注册绑定。详细 ioctl 编号和授权检查待 ABI 定稿。

在 `fdmfs_rw_iter()` 对注册 input 的写入进行专项检查：offset=0、完整包长度非零且不超过 W、满足当前 512 字节提交对齐、绑定有效。普通文件和普通 Q/mean 走原路径。

每次写分配独立上下文，包含 transfer_id、总长度、入口基址、绑定引用和错误状态。第一阶段只允许一个受控的同步包提交者；字段设计和生存期不依赖线程身份，后续扩展异步。

### 6.2 优先复用 iomap 扩展点

现有 `iomap_dio_rw()` 已接受 private 参数，`struct iomap_dio_ops` 已有 submit_io 回调，见 `linux-cemu/include/linux/iomap.h`；`iomap_dio_submit_bio()` 在提交 bio 时调用该回调。

建议用 private 把写上下文传给提交阶段，在 FDMFS 的 submit_io 回调给 bio 附加传输标签，然后走原 submit_bio 链路。第一阶段保持同步 DIO 完成语义，优先不修改通用 iomap 算法。

不能直接占用 `bio->bi_private`：当前它由 iomap 指向 dio，split 后又可能用于完成链关联；也不能覆盖 io_uring 将来使用的 kiocb private。若使用完成回调包装，必须保存原语义，并最终调用原 iomap 完成处理。

### 6.3 标签如何跨拆分存活

推荐先使用 CONFIG_NVME_CEMU 范围内的小型 bio 标签值字段／专用扩展：transfer_id、total_bytes、逻辑入口基址和有效标记。尽量复制标量，避免每个片段单独 malloc 或沿多级 bi_private 反查祖先。

必须覆盖以下路径：

1. `bio_init/reset` 清除标签；不能让对象池复用残留身份。
2. `__bio_clone()` 复制标签；同时保持现有 MEMORY 标志。
3. `bio_split()`、`bio_advance()` 后，使用当前 bi_iter 的地址减逻辑入口得到片段偏移，不能复制一个过期的固定 offset。
4. CEMU `bio_set_dev()` 转发及 NVMe request 构造保留标签。
5. request 部分完成、重试时，用当前请求范围构造命令，身份不能重置。

具体 bio 布局需在实现时确认；本方案要求的是关联正确，不限定一定添加某个现有尚不存在的成员。

### 6.4 合并约束

不同 transfer_id、不同绑定或 tagged/普通 I/O 之间绝不能合并。最保守的第一版可以对 tagged bio 设置 REQ_NOMERGE，并检查 bio→request、request→request 两类合并路径；这会改变命令数量，必须在报告中说明。

后续若要保留同包合并，可增加身份相等且偏移相邻的检查，但不能只修改一个 merge 入口。相关位置是 `linux-cemu/block/blk-merge.c` 的 `blk_rq_merge_ok()`、请求合并检查以及 plug 合并路径。

驱动不能不加检查地从 req->bio 取身份后假设请求中的所有 bio 都一致。

### 6.5 命令构造、完成和失败

`pci.c:nvme_prep_rq()` 的 MEMORY_WRITE 转换分支读取标签，检查范围，编码扩展字段，之后仍走现有 PRP/DMA 映射和 SQ 提交。

不修改现有 NVMe CQ → request → bio → iomap 完成的基本链路，不自行按 NVMe CQE 数量计算整包完成。同步应用在成功返回后可以释放源缓冲区。

一次写必须对应一个完整协议包。部分提交后失败或短写不能通过客户端 pwrite_all 的通用循环以新 transfer_id 续写剩余字节；专用 input 路径应报错并使会话进入可清理的失败状态，待已提交访问终结后回收。无片段提交前的可恢复失败可以重新提交整包。先采用失败即停止会话，暂不实现复杂的跨 syscall 恢复协议。

## 7. 设备多槽位结构与初始化

### 7.1 区分逻辑入口与真实数据池

内核发下来的 sb 仍表示逻辑入口加片段偏移，不再等于 tagged 包真实落点：

```text
packet_offset = command.sb - binding.input_entry_sb
slot = lookup_or_allocate(binding, transfer_id)
destination = slot.data_address + packet_offset
```

普通 FDM 的 sb 含义不变。接收路径必须先识别 tagged 入口、检查所有边界，再改用槽位真实地址进行 DMA；不能先调用原 backend_addr(sb) 写入再移动数据。

物理槽位池、管理区和暂存区都应受到保护，不允许通过普通 MEMORY_WRITE 绕过入口写入内部槽位。CUDA dirty 标记针对真实修改范围；trace 同时保留逻辑入口与物理槽位信息。

### 7.2 拟增加的管理内容

| 结构 | 内容 |
|---|---|
| input 池配置 | 协议版本、N、W、总预留字节、数据段定位、注册代数 |
| 每槽接收描述符 | transfer_id、total_bytes、槽位复用代数、状态、接收／预留位图、包头校验结果、max_ready_ns、在途引用 |
| 每槽聚合描述符 | 已解析包头、包序号、batch/token 分轮游标、依赖、错误和 trace 快照 |
| binding 运行态 | 空闲槽位管理、ID 查找结构、就绪队列、接收完成请求关联、停止状态和统计 |

可共享给算子的状态、槽位表和数据仍放在 FDM MR 布局中；GThread、锁、NvmeRequest 指针和 QEMU 调度对象只留在模拟器运行态，不能当作将来的硬件 ABI。

每槽独立维护当前单槽 ReceiveState 的信息。绝不能仅分配 N 倍数据，却继续复用同一份 received_bitmap、header 或聚合游标。

第一版可要求所有 input 槽位位于第一数据 MR 的可连续寻址区域，验证 N×W 不超过该 MR 的真实容量；之后才扩到跨 MR 槽位池。已有 FDMKV_DataSegment 的暂存页池支持不能被破坏。

初始化需要同时更新 bind ABI、buffer ABI、算子校验、客户端描述符大小和布局测试。数据池起点必须从旧的 align(input_bytes) 改为按整个 input 池预留量计算，审查 `fdm_kv_page_data()` 及所有 input_bytes 的用途，区分“每槽容量”和“全部 input 预留量”。

如果为了扩 input 从固定 data_bytes 内扣空间，暂存页池可用页会减少；必须重算容量，不允许与暂存页重叠或静默缩小页池。

### 7.3 状态及复用条件

```text
FREE → RECEIVING → RECEIVED → READY → AGGREGATING → CONSUMED → FREE
                      │         │           │
                 等deadline   等依赖      分轮／等待暂存空间
```

名称为设计用语，不表示现有枚举已具有这些状态。

- 接收位图只有 DMA 成功后才能置位；缩锁时还需片段预留状态防止并发重复 DMA。
- 第一个到达的片段不必 offset=0；其身份和总长度足以选槽，包头完整后再核对 wire_bytes/session/generation。
- READY 需要整包完整、合法并满足 deadline。发布数据与状态必须具有正确的内存可见性；真实硬件还要考虑 DMA 一致性和缓存维护，不是只加 volatile。
- 进入 READY 就可沿原完成链 ACK，不等待算子被调度。
- FREE 需要 input 消费结束且所有相关访问引用归零，不等待 NAND program。

## 8. 接收进度、算法依赖与常驻聚合

### 8.1 必须拆开的进度

当前 `receive_header()` 要求 request_id 等于 receive->next_request_id，而 `fdm_kv_receive_release()` 在聚合后才递增。多槽位下，这会阻止已经收齐 A 后接收 B。

需要区分：传输身份、按顺序接收的包序号、排队待聚合的 KV 前缀、实际已聚合的 head_states.accepted、已 NAND 写回的 committed。

第一阶段可保持单提交者按包序收齐，并用独立的“下一个接收包序号”验证包头；聚合仍按该顺序推进。接收下一个合法后继包时，不能要求它的 token_first 等于当前聚合 accepted，应与已接收队列的预期前缀核对。相关进度只有完整包验证成功后才发布。

后续多包异步乱序接收需要独立的序号／依赖跟踪，不能把本阶段的单接收顺序假设当成已支持任意乱序提交。传输标签本身不保证算法顺序。

Attention 仍读取实际聚合进度和 lease，不读取“包已经收到”的进度。当前 attention_ready 使用 accepted == valid_tokens，这不应在此次重构中随意改成 >=；旧快照读取、后继 append 和读者保护要继续保持一致。

### 8.2 常驻执行与现有单步算子的关系

“启动一次”不能仅实现为把 g_cond_wait 改成忙循环、仍对每包执行一遍原来的外部算子启动协议，却宣称硬件算子已常驻。

建议保留现有有限步聚合逻辑作为内部 step，新增常驻服务入口：在绑定初始化后进入一次运行循环，读取槽位描述符并调用内部 step，直到显式 stop。单步入口保留给单元测试和 legacy 模式。

若常驻入口放在算子模块，需要一个单独版本的 runtime 接口提供 stop、当前可执行时间／ready 状态以及必要的同步钩子。模拟器负责 NvmeRequest、时钟和锁，算子负责槽位消费和暂存布局。runtime 回调不是设备 FDM 中持久保存的主机函数指针，未来 FPGA 用硬件接口替代它们。

也可以先做设备运行时常驻轮询、模块有限步执行的过渡实现，但必须标明这是过渡，不等同于算子入口只调用一次；最终选型在写代码前确认。

本阶段只设一个聚合执行者，不同时引入多聚合并行。每槽进度独立，共享页池／domain／hash 仍遵守锁和 lease，不能让不同槽位并发修改同一元数据。

### 8.3 轮询的性能边界

- 只检查紧凑的状态／就绪队列，不扫描 payload，不每轮扫描全部元数据或置脏整个 MR。
- 空闲检查不长期持有 fdm_kv_lock 或 native_lock；执行有限步后允许接收、读者和 NAND writer 推进。
- 不在冻结时钟的区间空转；聚合维持现有不额外冻结的计时口径。接收 deadline 到期前不能偷跑。
- 区分硬件式忙轮询与模拟器的自适应 pause/yield/backoff。第一版选择及参数必须记录；退避不是逐包条件变量通知，但会影响响应延迟。
- 移除聚合任务唤醒，不等于删除所有条件变量。SparF 等待者、NAND writer 和关闭路径仍可以使用原有软件通知，不能一并删掉。
- 共享就绪标志、描述符发布和槽位复用必须有同步保证；stop 要能终止空闲和等待依赖状态。

## 9. 背压不能制造队首死锁

当前 `nvme_process_sq_io()` 在忙时不推进 SQ head，直接 break。严格单同步提交者下可以先保留这个行为作为过渡：前一包已经收齐才有后一包，已有任务仍由后台推进。

但为多包并发预留时必须注意：SQ head 是等待空槽的新包 B，而已有槽位的 A 的剩余片段在它后面，若一直不消费 B，就永远收不齐 A，也无法释放槽位。

后续多包接收需要一个有界的待准入描述符机制或等效调度方案，允许暂存新包命令而继续处理已准入包的片段；接管命令不等于完成它，必须保留有效请求和 DMA 资源。还要为已有包片段、普通读写和必要控制命令保留请求额度，不能只把死锁从 SQ 转移到请求池。

本阶段若采用单提交者过渡，必须显式拒绝不支持的并发模式并加入测试，不能声称已经支持 io_uring 多包在途。

## 10. drain、失败与退出

- APPEND 接收成功后发生的聚合错误无法撤回已成功的 pwrite，必须设置会话 sticky error，并让 Attention、后续追加和 drain 报错。
- 错误／取消／超时不能直接释放仍有 DMA 或算子引用的槽位；先终止相关访问，再回收。
- 未完整收齐的失败包需要取消／会话终止机制回收，不能让其永久占据一个槽位。第一版允许中止整个会话，不做隐式无限重试。
- 晚到片段不能命中新代槽位；正常 NVMe retry、已退休 transfer_id 的拒绝策略，以及 reset 后失效策略必须写入协议并测试。
- DRAIN 是屏障，不按普通 APPEND 的“到 input 即完成”解释。等待此前包聚合、传播错误，并按现有用户可见语义完成 NAND drain。
- DRAIN 不能依赖一个永远得不到的 input 额度；同步过渡可等待已有任务释放槽位，失败情况下必须有独立取消／停止路径。
- unbind/reset：停止接管新包 → 终止或排空 I/O → 停止常驻服务 → 解除读者／写回引用 → 释放 MRS 和槽位池。不能先释放 FDM 后 join 执行者。
- 初始化失败必须逆序回滚文件注册、绑定、算子服务和 MRS。

## 11. 按文件划分的实施路线

下表是待修改计划，不表示已修改。

| 阶段 | 文件／目录 | 修改内容及验收点 |
|---|---|---|
| A：协议与注册 | `CEMU/hw/femu/csd/memory.h`、`fdm_kv_buffer_abi.h`；Guest 对应协议定义 | 冻结 tagged MEMORY_WRITE、绑定版本、N/W、编号和长度范围；增加布局／大小校验 |
| B：FDMFS 写上下文 | `linux-cemu/fs/fdmfs/file.c`、`fdmfs.h`、`inode.c`；按需增加局部实现文件 | 文件注册、每写编号、iomap submit_io 标签；普通 FDM 不变 |
| C：标签透传 | `linux-cemu/include/linux/blk_types.h`、`block/bio.c`、`block/blk-merge.c` | 初始化、克隆、拆分、合并、重试的身份正确性；不覆盖 bi_private |
| D：驱动编码 | `linux-cemu/drivers/nvme/host/cemu.c`、`pci.c` | 按当前片段范围填写扩展字段；普通 NVMe／MEMORY_COPY 回归 |
| E：槽位池与接收 | `CEMU/hw/femu/fdm_kv_buffer.[ch]`、`fdm_kv_receive.[ch]`、`fdm_kv_binding.[ch]`、`csd/memory.c` | 独立状态、ID→槽位、DMA 前重定向、整包 ACK、容量与保护检查 |
| F：调度与聚合 | `CEMU/hw/femu/nvme-io.c`、`fdm_kv_aggregate.h`、`tests/cemu/kernel/fdm_kv_aggregate.c`、binding 运行时 | 每槽任务、常驻执行、接收／聚合进度分离、有限步锁范围和退出 |
| G：读者／写回衔接 | `CEMU/hw/femu/fdm_kv_attention.c`、`fdm_kv_writer.c` | 维持 accepted/lease 与原下刷语义，适配通知和 drain，不重写 NAND 放置 |
| H：客户端初始化 | `tests/cemu/instattention/guest/cemu_client.cpp`、`tests/cemu/instattention/guest/cemu_client.h`、`pybind_module.cpp`、`python/cemu_flexgen/fdm_kvbuffer_store.py` | N/W 配置、版本协商、文件注册和回滚、input 专用短写处理；不改变模型计算与 payload 布局 |
| I：测试与启动 | `tests/cemu/instattention/guest/test_fdm_kv_layout.c`、Python 测试及相关启动脚本 | 参数、容量、兼容性、生命周期、实际同步 pwrite 端到端测试 |

实现顺序建议：A → B/C/D 的标签闭环 → E 的 N=1 回归 → N=2 收齐提前 ACK → F 的常驻执行 → G/I 的依赖和异常 → 全流程性能。通过配置开关保留旧单槽路径用于对照。

## 12. 测试与计时验收

### 12.1 功能正确性

1. 一个包单命令／多命令，含超过块层长度限制的大包；每片段标签一致，偏移连续覆盖，不改变 payload 字节。
2. 片段乱序、包头后到、重复／重叠片段、越界、不同长度声明；不误发布 READY。
3. 两槽：A 聚合期间 B DMA；两槽满时 C 背压；A 结束后槽位复用，无数据覆盖。
4. pwrite 返回早于人工延迟的聚合结束，但不早于全部 DMA、包校验及模型 deadline。
5. 接收领先聚合时，合法后继 token 不被 accepted 检查误拒绝；实际聚合顺序正确。
6. SparF 先到仍等待真正 KV 就绪，保留 lease，不读取半更新页；不同 batch/layer 的行为正确。
7. 大小端、命令布局、legacy/tagged 不匹配、MEMORY_COPY 释放路径和普通 Q/mean/read 均回归。
8. 部分提交失败、DMA 失败、ACK 后聚合失败、重复旧 ID、CQ 满、reset/close/drain；无悬空访问、无泄漏、无重复完成。
9. 内核实际 pwrite 测试不可由 qtest 替代：qtest 能验证设备协议，不能证明 bio 标签克隆、合并和 DIO 生命周期正确。

可扩展已有 `test_fdm_kv_binding.py` 和 `test_native_csd_pipeline.py` 中的 input_backpressure、all_fragments_and_model_time、failure_after_ack、reset、lease、segmented_fdm 等测试。第一版并发限制也要有拒绝测试；后续 io_uring 接入前补充跨 SQ、多包交错和资源耗尽死锁测试。

### 12.2 性能口径

先固定我们之前的正式负载和 NAND 配置；每次记录 N、W、暂存页池容量、轮询策略、内核版本、命令标签协议、合并策略以及队列限制。

至少对比旧单槽／旧唤醒、新单槽／常驻、新双槽／常驻，避免把轮询与扩槽收益混为一谈。

记录：pwrite 次数与字节、NVMe 命令数和大小分布、transfer_id/packet_id/slot/复用代数、首片段到达、整包收齐、模型就绪、ACK、聚合开始／结束、槽位释放、无槽等待、聚合等待依赖／暂存页、SparF 等 KV、峰值槽位占用、聚合执行和空闲轮询 CPU 时间。

保持细粒度 trace 可关闭。常驻空转不按每轮写日志；采用累计计数或采样，避免观测本身主导耗时。

验收重点是确认下一包 DMA 与当前包聚合确实可以重叠，且 payload、生成结果与模型时间语义不变。pwrite 变短不代表聚合工作消失；模拟器忙轮询的 CPU 成本不能直接等同 FPGA 成本。

## 13. 开始实现前仍需冻结的细节

1. 注册 ioctl／联合绑定接口，以及如何验证内核文件与设备绑定的对应关系。
2. 12 字节字段最终分配、绑定协商方式、会话 reset 与 transfer_id 退休／重复策略。
3. bio 标签具体存储方式、所有 clone/reset/merge 路径的实现范围。
4. 首版只允许单同步提交者的执行约束；后续有界等待队列与请求额度方案。
5. 常驻模块入口加 runtime 接口，还是先采用设备运行时常驻的过渡实现；轮询退避参数。
6. N/W 的初始值、第一 MR 容量约束、扩 input 后的暂存区预算。

这些未定项不改变本阶段核心决定：同步写在接收完成时返回，槽位由设备管理，内核只携带传输身份，聚合消费在后台持续推进。
