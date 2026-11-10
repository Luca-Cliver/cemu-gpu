# FDM KV 异步写入设计方案

> 2026-11-10 更新：本文保留为历史讨论记录，文中的“已确定”仅指当时的讨论状态。当前阶段改为先实现标准同步 pwrite、设备自主分配多 input 槽位、内核传递写请求身份以及常驻聚合执行。请以 [同步写入与设备多槽位设计](FDM_SYNC_MULTISLOT_DESIGN_2026-11-10.md) 为准。本文的 FDMFS 槽位分配、特殊 pwrite 提前返回及用户包级 SQ/CQ，不是当前阶段的实现要求；两份方案不能混合实现。

讨论日期：2026-11-04 至 2026-11-06。状态：总体方向已确定，接口和异常处理细节仍待设计，尚未实现。本方案不代表当前代码已具备这些行为。

## 1 总体方案（汇报版）

### 1.1 目标与基本思路

保留 `pwrite → VFS → FDMFS → 块层 → NVMe` 软件栈，为专用 KV input 文件增加“提交后返回、后台完成”的写入模式。同时将设备 input 扩成多个固定槽位，让上层计算、下一包 DMA 接收与当前包聚合有机会重叠。

一次正常完整的 KV `pwrite` 提交一个包，对应一个聚合任务。底层可以拆成多个 bio 和 NVMe 命令，聚合算子也可以分多轮处理；新增管理以完整包为单位。

第一版采用一个应用提交者、两个 input 槽位、一个后台聚合线程。应用线程继续构造并提交子 I/O，不新增提交线程。专用模式下 pwrite 返回表示提交已被接管，最终成功或失败由包级 CQ 报告。Q、mean 和普通 FDM 写入保持原行为。

本文 Host 指应用和 Guest 内核这一侧；运行模拟器的物理 Host 是另一层。input 槽位是设备 FDM 中保存原始 KV 包的内存，与聚合后的 KV 暂存页和 NAND block 不同。

### 1.2 整体结构与各层职责

```text
Host 应用：计算并打包 KV；C++ 保留源缓冲区
      │ pwrite                         ↑ 读取共享 CQ，free 后更新 user_ack
      ↓                                │
Host 内核 FDMFS：包级 SQ / CQ + input 槽位空闲表
      │ 领取槽位 i，记录独立请求上下文    ↑ 汇总完成、DIO 收尾、发布包级 CQE
      ↓                                │
现有块层 / NVMe：bio → request → DMA 映射 / PRP → NVMe SQ
      │                                ↑ NVMe CQ → request / bio 完成
      ↓                                │
设备 FDM input：槽位 0、槽位 1……        │
      │ 按 sb 直接定位；分别 DMA 接收     │
      ↓                                │
后台聚合线程：整包就绪 → 聚合写入暂存页 → 交回保留请求，由 poller 发布 CQE
      │
      ├─ 发布 KV 就绪，允许依赖它的 SparF 读取
      └─ ready 暂存页继续由原 NAND writer 写回
```

| 层次 | 职责 |
|---|---|
| C++ | 每包分配并保留源缓冲区；读取包级 CQ、释放缓冲区、确认消费；背压重试和最终 drain |
| FDMFS | 管理包级 SQ/CQ、固定 input 槽位、独立请求上下文；提交后返回，最终发布包级结果 |
| 块层和 NVMe 驱动 | 复用 bio/request、DMA 映射、PRP、命令提交和完成链；约束跨槽位合并 |
| 设备接收路径 | 按 sb 定位已选定的 input 槽位，维护片段位图和模型 deadline |
| 设备聚合线程 | 消费完整且满足依赖的包；写入暂存布局、发布 KV 就绪、结束 input 访问 |
| SparF | 所需 KV 尚未聚合时等待，就绪后继续 Attention |

包级 SQ 是未回收请求的管理结构，不需要另一个线程消费。包级 CQ 映射给 C++ 读取，与 NVMe SQ/CQ 分属不同层次。

### 1.3 初始化与地址约定

初始化时确定 input 起点 `S`、槽位跨度 `W` 和数量 `N`。设备分配 `N × W` 的 FDM 区域；内核建立与其一一对应的槽位空闲表。W 至少容纳最大允许 KV 包及对齐填充。

应用仍向专用 input 文件偏移 0 写包。FDMFS 为本包领取槽位 i，并在本次 I/O 的映射中加入槽位偏移：

```text
片段目标 sb = S + i × W + 包内偏移
```

设备直接按 sb 定位 FDM 后端，并计算槽位编号和包内偏移以维护接收状态。内核不需要设备侧指针，设备也不再为每包另选任意物理块。不同包使用不同槽位，片段交错到达仍可区分。

同时初始化包级 SQ、共享 CQ 和确认游标。设 SQ 容量为 Q，CQ 容量至少为 Q；每个已接管包恰好产生一个最终 CQE，SQ 条目保留到用户确认并由内核回收。因此接管新包时已有空间保留其未来完成记录。input 槽位和 SQ/CQ 额度都必须满足才能接管新包。

### 1.4 一包从提交到回收的完整流程

```text
① C++ 收割共享包级 CQ
   → 按 request_id 释放已完成包的源缓冲区
   → 发布 user_ack
   → 准备新包，保留源缓冲区及其 request_id

② pwrite 进入 FDMFS
   → 根据 user_ack 回收已确认的 CQ 项和对应包级 SQ 条目
   → 检查额度，领取固定 input 槽位 i，登记包级请求
   → 当前线程构造并提交全部子 I/O，各片段使用槽位 i 的地址
   → 返回已接管的提交字节数，不等待设备完成

③ 设备接收
   → 根据 sb 识别槽位 i 和包内偏移，DMA 写入该槽位
   → 更新片段位图，汇总模型 deadline
   → 整包收齐后保留使其收齐的那条原始 NVMe 请求
   → 满足 deadline 和追加依赖后，交给后台聚合线程

④ 设备聚合
   → 将本包 KV 写入暂存布局，更新 accepted 并通知依赖等待者
   → 结束对 input 槽位的相关访问，保存完成路径所需元数据
   → 将保留请求交回原 poller，由其发布 NVMe CQE

⑤ Host 内核完成
   → 原 NVMe 完成链推动 request / bio 完成
   → iomap 引用计数确认提交结束且全部子 I/O 终结
   → 执行 DIO 收尾，再发布包级最终 CQE
   → 成功包的 input 槽位获得复用资格

⑥ C++ 确认与内核回收
   → C++ 读取 CQE，释放对应源缓冲区，更新 user_ack
   → 内核回收该完成项和对应包级 SQ 条目
   → 后续包可以使用释放出的提交额度
```

若子 I/O 在提交引用释放前已经全部完成，由提交路径直接执行 DIO 收尾；否则采用异步完成路径，最后一个 bio 完成后安排 worker 收尾。两条路径汇入同一个包级完成处理，每包只发布一次 CQE，详见 2.3。

pwrite 返回的承诺是子 I/O 已交给现有 I/O 系统。驱动忙时请求仍可能在块层排队，不另外等待全部命令实际进入 NVMe SQ。

### 1.5 背压、依赖和资源释放

无空闲 input 槽位或包级 SQ/CQ 额度时，FDMFS 在接管本包、提交任何片段之前返回 `EAGAIN`。C++ 保留待提交的包和缓冲区，收割 CQ、更新 user_ack 后重试。内核不能在这次 pwrite 中等待一个只能由同一用户线程返回后才能更新的 user_ack。

pwrite 提前返回后，Execute 可能早于 KV 包到达。需要扩展 SparF 现有就绪检查，使其等待对应 KV 聚合完成；设备接收和聚合继续推进，不阻塞 NVMe poller。

| 资源或时刻 | 释放/完成条件 |
|---|---|
| KV pwrite 返回 | 提交被接管；源数据仍必须保留，KV 未必已到设备 |
| input 槽位 | 设备相关访问结束；内核确认本包成功完成后允许复用 |
| DIO 对象及 I/O 资源 | 提交引用和全部子 I/O 引用归零后，按 DIO 完成路径收尾 |
| C++ 源缓冲区 | 上层读到最终包级 CQE，确认访问已终结后释放 |
| 包级 CQ 项和对应 SQ 条目 | C++ 释放源缓冲区并发布 user_ack 后，由内核回收 |
| Execute 完成 | 本次 Attention 完成，与 KV 写包完成分开 |

包级成功完成表示 KV 已聚合到暂存布局，不要求 NAND program 完成。最后一次写入后通过显式 drain 收割所有包级结果、释放剩余源缓冲区并完成内核回收；需要持久写回时再按原语义等待 NAND drain。错误完成也必须先终止相关访问，不能仅凭错误码释放仍在使用的资源。

## 2 讨论中的问题与具体设计

### 2.1 如何区分 KV input 和 Q mean 等普通写入

**问题：** 它们都走 FDM 文件和 MEMORY_WRITE，但只希望 KV input 使用新协议。

**设计：** 复用当前的地址范围识别。上层 fd 对应文件的 FDM 区域，FDMFS 在 [iomap_begin()](/home/lihan/cemu/linux-cemu/fs/fdmfs/inode.c:214) 以 `region->off + offset` 确定目标位置；设备看不到 fd，根据 sb、长度和已注册区域分类。

[memory_write()](/home/lihan/cemu/CEMU/hw/femu/csd/memory.c:89) 已调用 [fdm_kv_write_check()](/home/lihan/cemu/CEMU/hw/femu/fdm_kv_binding.c:582)。普通 Q/mean 写入仍直接按 sb 定位；KV input 使用固定槽位和多块接收流程，保护区及越界检查继续保留。

FDM 没有专用 NVMe SQ/CQ。不同请求共用控制器队列资源，具体 SQ 由 blk-mq 映射决定；不能按某一条 CQ 的所有完成项判断 KV 包完成，必须沿各请求的完成关联处理。

### 2.2 一个 pwrite 拆成多条命令，如何判断整包完成

**问题：** NVMe CQE 是命令粒度；包级 CQE 必须代表整次写入完成，且不能早于整包聚合结束。

**设计：** 设备保留让整包收齐的那条原始 NVMe 请求，聚合完成、input 块可复用后才交回 poller 发 CQE。其他片段可以正常完成。“收齐的那条”不是固定的最高偏移片段，以位图和接收计数判断。

内核复用 [iomap_dio_bio_end_io()](/home/lihan/cemu/linux-cemu/fs/iomap/direct-io.c:159) 的子 I/O 汇总和错误传播，不重新解析、计算原始 NVMe CQE 数。引用计数包含提交阶段的保护引用，避免一部分 I/O 已完成、另一部分尚在构造时提前宣布整包完成。

当前同步 pwrite 会在 [DIO 等待循环](/home/lihan/cemu/linux-cemu/fs/iomap/direct-io.c:714) 睡眠，最后完成时被唤醒。Guest 的普通 NVMe 中断路径 [nvme_irq()](/home/lihan/cemu/linux-cemu/drivers/nvme/host/pci.c:1278) 调用 `nvme_poll_cq()` 消费 CQE；这个函数名不表示有常驻 Guest 轮询线程。设备端 [nvme_poller()](/home/lihan/cemu/CEMU/hw/femu/nvme-io.c:409) 则负责遍历 SQ、处理完成和通知 Guest。

新方案保留底层完成机制，把整次 I/O 的最终结果交给包级 CQ。原 pwrite 线程返回后，即使没有下一次写入，完成路径仍会推进。

### 2.3 提前返回后，请求上下文和收尾由谁负责

**问题：** 不能只把唤醒 waiter 改成写 CQ 标志。同步请求还引用调用栈中的对象，原等待线程还承担收尾。

**设计：** 包级 SQ 条目持有独立分配的请求上下文、文件引用、包标识和完成状态，直到后台不再使用时才释放。不能继续引用 [new_sync_write()](/home/lihan/cemu/linux-cemu/fs/read_write.c:487) 栈上的 `kiocb`、`iov_iter`。

**收尾方式已明确（2026-11-06）：保留 iomap 的直接完成和异步完成两条路径，复用其引用计数判断，不额外设计一套“谁最后完成”的机制，也不强制所有收尾都转交 worker。** 两条路径都在 DIO 收尾后接到同一个包级完成处理逻辑，每个已接管请求恰好发布一个最终 CQE。

`dio->ref` 的变化位置如下：

| 位置 | 操作 | 含义 |
|---|---|---|
| [创建 DIO 对象](/home/lihan/cemu/linux-cemu/fs/iomap/direct-io.c:568) | `atomic_set(&dio->ref, 1)` | 提交阶段持有一个保护引用 |
| [每个 bio 下发前](/home/lihan/cemu/linux-cemu/fs/iomap/direct-io.c:69) | `atomic_inc(&dio->ref)` | 为该 bio 增加一个引用 |
| [提交阶段结束](/home/lihan/cemu/linux-cemu/fs/iomap/direct-io.c:715) | `atomic_dec_and_test(&dio->ref)` | 归还提交引用，判断是否归零 |
| [每个 bio 完成](/home/lihan/cemu/linux-cemu/fs/iomap/direct-io.c:167) | `atomic_dec_and_test(&dio->ref)` | 归还该 bio 的引用，判断是否归零 |

例如只有一个 bio、对应一条 NVMe 命令，在 bio 已加引用而双方都尚未归还时，计数为 2：提交引用 1 个、bio 引用 1 个。NVMe 命令不会再给 `dio->ref` 额外加一个引用。

```text
提交阶段先结束：
  2 → 1：归还提交引用，异步提交路径可返回
  1 → 0：bio 完成，触发异步收尾

bio 先完成：
  2 → 1：归还 bio 引用，提交引用仍保护 dio
  1 → 0：提交阶段结束，由提交路径触发收尾
```

这个初始保护引用防止 bio 很快完成时，提交代码尚在使用的 `dio` 被提前收尾。原子减计数保证只有一方归零；这里的引用计数不是 NVMe 命令计数，也不是包级 SQ/CQ 的槽位占用计数。

新模式使用独立的异步 DIO 上下文，按下面两种结果接入完成处理：

| 提交结束时的情况 | 现有 iomap 行为 | 包级 CQ 的接入方式 |
|---|---|---|
| 提交引用归还后减到 0，子 I/O 已全部完成 | 外层 [iomap_dio_rw()](/home/lihan/cemu/linux-cemu/fs/iomap/direct-io.c:748) 直接调用 `iomap_dio_complete()` 并返回最终结果 | 专用封装将最终结果交给包级完成处理；不能只等 `ki_complete` 回调 |
| 提交引用归还后仍非 0 | 返回内部标识 `-EIOCBQUEUED`，后续最后一个 bio 完成触发异步收尾；采用 [workqueue 路径](/home/lihan/cemu/linux-cemu/fs/iomap/direct-io.c:222) | worker 在 [iomap_dio_complete_work()](/home/lihan/cemu/linux-cemu/fs/iomap/direct-io.c:141) 中先收尾，再通过 `ki_complete` 发布包级完成 |

`-EIOCBQUEUED` 表示内部异步交接，不是上层背压用的 `-EAGAIN`，也不能直接返回给普通同步 pwrite 调用链。专用封装按协议返回已接管的提交字节数；后续最终结果通过包级 CQ 报告。异步完成回调也可能在系统调用真正返回前发生，因此不能用“pwrite 是否已经返回”判断由哪一方完成。

包级请求的身份、资源引用和完成回调必须在首次下发前准备好。两条路径不得各自重复收尾或重复发布 CQE。`iomap_dio_complete()` 释放 DIO 对象之后，包级请求的必要管理信息仍按 SQ/CQ 和 `user_ack` 的回收协议保留。

收尾涉及最终结果、位置、inode DIO 计数、条件性同步和资源释放，不能无条件在硬中断上下文完成。上下文与返回字节数、异步错误的具体接口仍需实现设计。

### 2.4 单个 input 阻止接收和聚合重叠，如何扩充

**问题：** 当前后台聚合模式仍只有一个 input。`aggregate_active` 为真时后续写入暂留 SQ，不能同时 DMA 下一包。

**设计：** 扩充为固定大小整包接收池，由内核领取对应槽位，设备维护各槽位接收状态，每块独立接收、交接和复用。

第一版将当前一个 input 块扩成两个独立块，后续允许配置数量 N。这些块仍属于设备 FDM 内存；它们保存尚未聚合的原始包，与聚合后的 KV 暂存页池、Attention 的 Q/mean/output MR 分开管理。

固定大小方案下，每块容量至少容纳允许提交的最大整包，包含包头和对齐填充。总输入池容量为 `N × 每块容量`，另加各块接收位图和任务元数据。Prefill 大包可能使输入池较大，不能只按 Decode 包大小估算。注册参数、MR 容量校验、设备内存预算和上层额度协商都要随之更新。

内存可以初始化时一次分配、按块划分；每次“分配/释放 input”只是领取或归还块，不要求每包 malloc/free。第一版不同时引入可变大小分配器，减少碎片和容量管理问题。

```text
设备 FDM
├─ input 数据池
│   ├─ 块 0：包 A，聚合线程正在读取
│   └─ 块 1：包 B，设备正在 DMA 接收
├─ KV 暂存页池：保存聚合后的 token-K / token-V / channel-K
└─ Attention 工作区：Q、mean、输出等
```

块数量控制可同时接收/保留多少个包，不等于聚合线程数、NVMe 队列深度或 NAND block 数量。

**每块状态和就绪队列：**

拟议每个输入块保存下列状态；具体结构体和字段命名尚未确定：

| 信息 | 用途 |
|---|---|
| 块地址、容量 | 定位实际 DMA 目标和算子输入 |
| 包身份、session、generation | 将不同命令的片段归入同一个包，防止复用后的旧片段污染 |
| 包头、header_valid | 校验布局、batch/layer/head/token 范围及整包长度 |
| 接收位图、已收单元数 | 防止重复写，判断整包完整 |
| 在途 DMA/片段预留状态 | 缩锁后防止同一范围被重复占用；DMA 完成后才能宣布收到 |
| max_write_ready_ns | 汇总本包所有片段的模型完成时间 |
| 聚合进度 | 保留分轮聚合的 batch/token 游标，避免重复追加 |
| 保留的原始 NVMe 请求 | 聚合终结后交回正常完成路径 |
| 状态及错误 | 决定可接收、可聚合、可回收或失败 |

概念状态流为：

```text
FREE → RECEIVING → RECEIVED → READY → AGGREGATING → FREE
                       │         │          │
                 等模型 deadline │          └─ 分轮推进/资源等待
                                 └─ 等待单个聚合线程取任务
```

这里 RECEIVED/READY 是设计状态，不表示现有代码已有同名枚举。只有字节完整、校验通过且模型 deadline 满足，才能交给聚合消费；模型等待也可以通过任务 deadline 表达，不必增加枚举。

内核维护空闲槽位表，设备维护每槽位的包状态和聚合任务队列。可以用 FIFO，但必须满足同一 KV 流按 token 前缀追加的依赖：包 B 先收齐，不代表它可以越过尚未接收完的前驱包 A。不能无条件按“收齐顺序”聚合。

### 2.5 多包片段交错到达，如何识别归属与回收 input 槽位

**问题：** 不同包若都写相同 sb，设备不能仅凭片段偏移判断归属。NVMe command_id 标识单条命令，包头也不覆盖全部片段。

**选定设计：** 内核领取与设备 input 一一对应的固定槽位，通过已有 sb 表达槽位编号和包内偏移。此前讨论过的“逻辑窗口再映射到任意物理块”方案不再采用。

设 input 区域起点为 S，槽位跨度为 W，数量为 N。每包不超过 W，各槽位按 NVMe/FDM 传输要求对齐：

```text
槽位 0：[S,       S + W)
槽位 1：[S + W,   S + 2W)

本包某片段的目标 sb = S + i × W + 包内偏移
```

应用仍向同一个专用 input 文件的逻辑偏移 0 写包；槽位偏移由 FDMFS 加入。必须将槽位选择放进独立 I/O 上下文并贯穿各段 iomap 映射，不能让并发请求共同改写 inode 的单个 region 起点。

设备先验证地址范围和长度，再定位：

```text
slot_id       = (sb - S) / W
packet_offset = (sb - S) % W
DMA 目标      = backend_addr(sb)
```

例如 A 使用槽位 0、B 使用槽位 1，即使 B 的片段先到达，也只会写槽位 1。每槽位只容纳一个在途包；任意片段都可先到，包头收齐后再校验整包身份和长度。

内核根据自身领取记录和完成通知管理空闲，不需要查询设备指针：

```text
FDMFS 领取 input 槽位 i → 标记被包 A 占用 → 提交各片段
  → 设备按 sb 直接写入槽位 i，接收包 A
  → 聚合 A 完成，结束相关 input 访问
  → poller 发布之前保留的 NVMe CQE
  → 内核确认 A 的全部子 I/O 终结并收尾
  → 包 A 成功完成，槽位 i 获得复用资格，发布包级 CQE
  → 上层消费并确认 CQE 后，内核回收对应 SQ/CQ 记录
```

必须满足的约束：

- 初始化时约定 S、W、N；所有使用同一 input 池的文件描述符共享槽位管理，地址范围不与其他 FDM MR 冲突。
- 单条命令不得跨槽位，块层也不得把不同包合并成跨槽位 request；具体合并限制位置待实现确定。
- 旧包的所有片段和后台访问终结前不能复用槽位。槽位编号不含轮次，不能单靠包头序号判断无包头片段的新旧。取消、超时和 reset 必须先终止或隔离旧访问。
- 失败不能直接视为正常空闲，要根据部分提交、部分聚合和设备状态处理。
- 槽位可复用与 SQ/CQ 记录可覆盖是不同条件。即使 input 已空闲，新包仍须取得包级提交额度；未获 user_ack 的完成记录必须保留。

request_id 与本包、源缓冲区和槽位在请求生命周期内绑定。槽位会复用，不能将 slot_id 单独作为长期唯一请求编号；提交双方如何取得一致 request_id 仍需明确接口，见 2.8。

### 2.6 设备 DMA 聚合和锁怎样改，哪些部分复用

**问题：** 扩大内存还不够，DMA 目标、接收进度和算子上下文必须对应同一个包，接收与聚合也不能被同一把大锁串行化。

**设计一：按 sb 直接定位 DMA，按槽位更新接收状态。**

复用现有 `fdm_kv_write_check()` 的地址分类：Q/mean 等普通 MR 继续直接 DMA；KV input 范围进入多包接收逻辑。内核已选择固定槽位，设备直接按 sb 写入该槽位，不再另外分配或重映射。

```text
收到 MEMORY_WRITE
  → 根据 sb、长度分类
  → 普通 FDM：现有 backend_addr(sb) 路径
  → input：
      解析槽位编号与包内偏移
      → 校验该槽位状态并预留片段范围
      → dma_write_prp(块基址 + 包内偏移, ...)
      → DMA 成功后更新位图、计数和 max_write_ready_ns
      → 收齐后登记聚合任务，通知 worker
```

PRP 遍历和 Guest RAM 到目标缓冲区的搬运机制继续复用。改变的是每包目标 sb 的生成及外围接收状态管理；如果所有包仍使用同一固定 sb 且没有片段归属协议，不能仅靠扩大内存自动实现这一流程。

input 写入的模型 deadline 仍须保留，不能在实际 memcpy 结束后忽略模拟时间就宣布可聚合。

**设计二：一个后台聚合线程按依赖消费任务，复用聚合核心。**

沿用现有后台线程和聚合算子核心，不为每个输入块创建线程。worker 从可执行任务中取包，设置该任务的 input 地址和 receive/progress 上下文，调用原聚合逻辑。

当前算子会访问 `buffer->input`、`buffer->receive` 和 MR 参数，因此需要调整调用上下文或 ABI。不能让接收线程和聚合线程无保护地来回改写同一组全局指针。具体采用每任务视图还是显式传入块描述，需要实现前确定；聚合布局算法可复用，不代表算子入口完全不改。

```text
时间 →
接收： [DMA 包 A] [DMA 包 B........]       [DMA 包 C，复用空闲块]
聚合：           [聚合 A...........] [聚合 B...........]
```

聚合产生的 ready 暂存页继续交给原 NAND writer。聚合完成后即可以归还 input 块，因为数据已进入暂存布局，不必等待 NAND program。遇到暂存页不足时仍需等待 writer；多 input 只增加缓冲和重叠机会，不会消除长期吞吐不足造成的背压。

当前 [complete_write_locked()](/home/lihan/cemu/CEMU/hw/femu/fdm_kv_binding.c:274) 通过请求指针区分交接和实际执行：poller 传原请求，worker 传 `binding->aggregate_request`，后者跳过交接分支。多包版本要把这些全局单任务状态变成每包上下文，保留正常完成请求与内部聚合任务的归属。

当前 async 模式是交接后即可发布写 CQE，新方案改为聚合结束后才终结保留请求。deadline 检查仍是绝对时间检查，重复检查不代表重复等待同一时长。

**设计三：缩小锁范围，按资源状态背压。**

当前 [fdm_kv_input_busy()](/home/lihan/cemu/CEMU/hw/femu/fdm_kv_binding.c:523) 用全局 `aggregate_active` 挡住整个 input 范围。新模式应根据目标槽位是否可接收、片段是否冲突、该槽位是否正在消费等条件决定接收，不能因为任何一包正在聚合就挡住全部 input。

接收线程先短暂持锁校验槽位和预留片段，再进行 DMA，最后发布片段完成；聚合线程独占消费 READY 块。生命周期引用要保证 DMA 期间块和绑定不能被解除或复用。接收块锁不应与整段聚合共用大锁，否则有两块内存仍无法重叠。

暂存页分配、ready 队列、accepted、readers 和 writer 仍需要一致的同步。需要规定锁顺序，并确保等待空间时释放相关锁，让负责释放空间的线程可以运行。

无空闲输入块时施加背压，不覆盖已有包。优先从上层额度限制在途包，设备仍保留容量检查。若继续采用 SQ 头部暂缓取命令，要明确同 SQ 队头阻塞的影响，不能阻塞整个 poller 或已接收包的完成处理。

### 2.7 pwrite 返回后立即 Execute，KV 尚未到达怎么办

**问题：** 新方案下 Execute 可能比 KV 包头还早到达；不能把“写调用返回”解释为“数据已经进入 input 或完成聚合”。

**设计：** 扩展已有 SparF 就绪等待，而不是让 NVMe poller 阻塞等数据。

复用现有 `waiting_attention`、`attention_ready()`、`fdm_kv_attention_wait()` 和聚合后的发布通知。

当前 [fdm_kv_attention_open()](/home/lihan/cemu/CEMU/hw/femu/fdm_kv_attention.c:171) 只允许等待一个已有有效包头、已进入后台聚合、范围匹配的包；接收尚未完成的情况不能直接通过现有校验。新模式必须扩展为可以等待已合法登记但尚在接收或排队的数据，处理 Execute 比包头更早到达的情况。

[SparF 工作线程](/home/lihan/cemu/CEMU/hw/femu/csd/compute.c:2426) 在算法执行前等待，等待时释放共享锁，让接收和聚合继续推进。`accepted` 在数据布局写入后更新；当前就绪条件是精确等于目标 token 前缀，不能未经审查改成大于等于。

需要核对 leases/readers 的取得时机，避免等待者占有资源反过来阻止聚合；写入失败、取消、设备停止时，依赖等待也必须结束。

### 2.8 pwrite 提前返回后，源缓冲区怎么保留和释放

**问题：** 当前 [CemuClient::write_range()](/home/lihan/cemu/CEMU/tests/cemu/instattention/guest/cemu_client.cpp:441) 每次分配 `AlignedBuffer`、复制 Python 包数据，函数返回时析构并 free。提前返回可能使设备 DMA 读取已经被分配器复用的内存。

**设计：** 延长已有缓冲区的生命周期，不要求第一版建立用户缓冲池，也不新增一份内核整包复制。

第一版不要求实现用户缓冲池。继续每包分配现有的 `AlignedBuffer`，由 C++ 的 pending 表保留所有权：

```text
pending[request_id] → 本包 AlignedBuffer

pwrite 提交返回：不 free
上层读到本包最终完成：移除 pending 项 → 析构 → free
```

用户缓冲区在 Guest CPU RAM 中。内核 pin 页面不能禁止应用修改或分配器复用，内核也不能调用用户空间分配器释放 malloc 对象。完成前上层不能修改、释放或复用该缓冲区；错误完成也只有在所有已提交访问终结后才能归还所有权。

不新增正常每包查询 ioctl。C++ 在下一次写入前直接读取共享完成环，释放相应缓冲区。需要设置未完成字节数或请求数量上限，并在最终 drain 回收剩余对象。

**仍待设计：request_id 如何由提交双方一致确定。** 普通 pwrite 没有 user_data 字段；不能仅写一个 pending 表就假定内核能返回相同 ID。需要明确提交序号/包头/input 槽位的关联，并处理短写、EINTR、部分提交和重试。单提交者可简化，但不能掩盖失败后的序号歧义。

### 2.9 不增加每包查询 ioctl，如何通知上层并防止丢完成

**问题：** C++ 必须知道哪些请求已经终结才能 free，但不希望每包增加一次查询 syscall。仅让 write_iter 清 CQ 又会漏掉上层尚未看到的完成项。

**设计：** 把同一套包级 CQ 映射给 C++，用 producer、user_ack、reclaim 协调发布、消费和回收。

包级 CQ 就是 FDMFS 中的那一套完成队列，不新增第二套。初始化时映射给用户空间；包级 SQ 可以只由内核管理。共享区域是 Guest CPU RAM，不是设备 FDM 或原始 NVMe 队列。

| 位置 | 唯一写入方 | 含义 |
|---|---|---|
| producer | 内核完成路径 | 已发布完成项的进度 |
| user_ack | C++ | 已读取并完成源缓冲区释放的进度 |
| reclaim | 内核 write_iter 等回收路径 | 已允许再次覆盖的队列槽位进度 |

三个位置可以一起共享；reclaim 也可以仅放内核。上层不改 CQE 内容，只更新 user_ack。用单调计数及取模定位槽位，具体 ABI、对齐、回绕规则待实现。

必须保留以下顺序：

```text
内核填完整 CQE → release 发布 producer
上层 acquire 读取 producer → 读取完成项 → free 对应缓冲区
  → release 发布 user_ack
内核 acquire 读取 user_ack → 允许回收并覆盖相应槽位
```

单线程应用也存在竞态：上层检查 CQ 后、进入下一次 pwrite 前，可能有新完成项到达。如果 write_iter 无条件清掉它，上层就永远收不到对应完成记录。因此只能回收上层已确认的条目。

**SQ 与 CQ 一起回收。** 内核发布包级 CQE 时不清掉对应 SQ 条目；上层按 request_id 释放源缓冲区并更新 user_ack 后，内核才回收该 CQ 项及对应 SQ 条目。完成可以乱序，必须通过请求身份找到对应 SQ 条目，不能假定 SQ/CQ 相同下标就是同一请求。

CQ 容量至少等于 SQ 容量，每个已接管请求只产生一个最终 CQE。在上述回收规则下，领取 SQ 条目就同时占住了未来完成记录的容量，不必另外增加一套独立预留计数。input 槽位获得复用资格也不能绕过这一提交额度约束。

**队列满时返回背压。** write_iter 先按 user_ack 回收已确认条目；仍无额度时，在接管新包和提交任何片段之前返回 EAGAIN。C++ 保留同一待提交包和源缓冲区，读取 CQ、更新 user_ack 后重试。不能让内核睡眠等待只能由当前用户线程推进的 user_ack。这里的 EAGAIN 表示本次未接管，不适用于已部分下发的包；客户端也需要为专用协议增加明确重试处理。

DIO 收尾释放的是底层 I/O 资源，发生在发布包级 CQE 之前；这里回收的是包级 SQ/CQ 管理记录，发生在用户确认之后，两者不是同一动作。

### 2.10 错误 重置和最后一次写入如何收尾

**问题：** 最后一次写入之后可能没有下一次 pwrite；异常时也可能存在在途 DMA、部分聚合和未领取完成项。

**设计：** 提供最终 drain 和会话终止规则，覆盖全部请求状态，不能依赖下一次写入触发所有回收。

正常完成时先保存待发布的完成信息，解除包到块的关联并标记块可复用，再将保留的原始请求交回 poller。不能在 poller 或 trace 仍引用包头时直接覆盖它；需要复制必要元数据或保留引用。

设备归还的是实际输入块；FDMFS 根据包级完成恢复的是提交额度。这两个动作有因果顺序，但不是同一处内存操作。

收包错误、聚合失败、取消和 reset/unbind 需要终结对应请求并通知依赖者。不能只把块放回空闲列表而留下正在 DMA、聚合或等待的引用；也不能重试已部分推进 accepted 的整个包而重复追加。DRAIN 应等待所有已接受包处理完，再按原语义等待需要的 NAND 写回。

客户端当前支持短写和 EINTR 重试，因此“一包一次 pwrite”是正常提交约定，不是异常路径下的天然保证。实现前需明确专用协议如何区分未接受、部分提交和已接管请求；不能按普通剩余字节重试逻辑盲目另分一个 input 槽位，也不能在失败返回后提前 free 仍被设备访问的源数据。

## 3 实施范围与验证

### 3.1 修改位置

| 位置 | 拟议职责 |
|---|---|
| [fdm_kvbuffer_store.py](/home/lihan/cemu/CEMU/tests/cemu/instattention/python/cemu_flexgen/fdm_kvbuffer_store.py:181) 与客户端封装 | 保持完整包提交，接入提交身份和最终 drain |
| [cemu_client.cpp](/home/lihan/cemu/CEMU/tests/cemu/instattention/guest/cemu_client.cpp:441) | pending 源缓冲区、共享 CQ 收割、user_ack 和错误处理 |
| [FDMFS file.c](/home/lihan/cemu/linux-cemu/fs/fdmfs/file.c:24) | 包级 SQ/CQ、额度、独立上下文、专用提前返回和共享映射 |
| [FDMFS inode.c](/home/lihan/cemu/linux-cemu/fs/fdmfs/inode.c:214) | 将本次请求的固定 input 槽位贯穿 iomap 地址映射 |
| [iomap direct-io.c](/home/lihan/cemu/linux-cemu/fs/iomap/direct-io.c:141) | 优先复用完成汇总和异步收尾；是否需要通用层改动待实现评估 |
| [NVMe pci.c](/home/lihan/cemu/linux-cemu/drivers/nvme/host/pci.c:943) 与块层 | 复用命令/PRP/CQ 完成，落实跨包合并限制所需的最小改动 |

设备侧定位：

| 位置 | 拟议职责 |
|---|---|
| [fdm_kv_binding.h](/home/lihan/cemu/CEMU/hw/femu/fdm_kv_binding.h:11) 及 bind 初始化 | 多槽位池、每包上下文、接收状态/就绪队列、容量和生命周期 |
| [csd/memory.c](/home/lihan/cemu/CEMU/hw/femu/csd/memory.c:89) | input 分类后的 DMA 目标选择、片段接收发布 |
| [fdm_kv_receive.c](/home/lihan/cemu/CEMU/hw/femu/fdm_kv_receive.c:19) | 按块校验、位图、整包收齐、独立进度 |
| [fdm_kv_binding.c](/home/lihan/cemu/CEMU/hw/femu/fdm_kv_binding.c:472) | worker 消费任务、等待资源、释放块、交回原请求 |
| [fdm_kv_aggregate.c](/home/lihan/cemu/CEMU/tests/cemu/kernel/fdm_kv_aggregate.c:521) | 接入当前包视图；复用布局组织和暂存写入核心 |
| [nvme-io.c](/home/lihan/cemu/CEMU/hw/femu/nvme-io.c:55) | 容量背压、保留请求的完成衔接，复用 CQE 写入 |
| [fdm_kv_attention.c](/home/lihan/cemu/CEMU/hw/femu/fdm_kv_attention.c:118) | 多包依赖识别与提前到达的 Execute 等待 |

### 3.2 已确定与待确定的边界

已确定：当前线程提交；内核领取与设备 input 一一对应的固定槽位；包级 SQ/CQ；保留 iomap 两条完成路径；单聚合线程；共享 CQ 和三个游标；user_ack 后一起回收 SQ/CQ 条目；提交前额度不足返回 EAGAIN；C++ 延迟 free；扩展 SparF 等待。

仍待确定的实现细节：

- 专用文件模式、共享映射 ABI、S/W/N 协商及绑定生命周期。
- 异步上下文的文件引用和文件系统写入保护配对，确保原 syscall 返回后仍可安全完成；不能仅延长 kiocb 的存活时间。
- 用户 request_id、包头编号、槽位使用轮次的对应，以及短写/部分提交协议。
- 每任务算子上下文、锁顺序、跨槽位合并限制和 reset 后旧访问隔离。
- 完成环多完成者同步、容量与游标的具体 ABI，以及 EAGAIN 重试和 drain 接口。
- Execute 早于包头时的合法依赖登记、失败传播和取消规则。

### 3.3 验证目标

先以两个设备输入块、一个聚合线程、一个应用提交者验证。覆盖乱序片段/完成、多命令大包、CQ 满、源缓冲区生命周期、Execute 提前到达、聚合失败和最终 drain。

性能验证需确认“下一包 DMA 与当前包聚合”确实重叠，并分别统计提交软件栈、额度等待、DMA、聚合、SparF 等 KV 的时间。提前返回缩短的 syscall 时间，不等于总工作量已经减少。
