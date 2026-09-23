# 在 CEMU 中复现 InstAttention：阶段进展汇报

整理日期：2026-09-22
汇报主线：整体架构 → 真实模型流程 → InstAttention 行为对齐 → KV 存储与 FTL → 异步与时钟 → 验证和下一步。

本报告用于阶段汇报，不替代实现工作记录。重点说明架构和行为复现；异步部分分别介绍新增机制、原生 CEMU 能力和时钟对齐，但不将流水效率或完整并发精度作为当前已验证成果，不宣称已完整复现论文全部实验。

## 一、研究目标与当前定位

我们的目标是在 CEMU 中复现 InstAttention 的计算存储行为，随后在明确工作负载和硬件配置的条件下，对比模拟结果与论文结果，分析模拟精度。

这里的“复现”不是让两份代码完全一致，也不是通过调整 runtime 强行获得相同吞吐，而是首先对齐以下内容：

1. **计算分工**：哪些计算留在应用侧 GPU，哪些计算交给 CSD。
2. **访问流程**：如何根据 Q 选择 channel，再根据近似分数选择 token，读取相应 K/V。
3. **数据组织**：K 的 token/channel 双布局、V 的 token 布局，以及对应的分组方式。
4. **设备职责**：设备管理缓存、映射、内部读取和后台写回，而不是由应用组织所有细节。
5. **完成语义**：真实数据正确可用，同时满足模型规定的完成时间和依赖条件。

目前已经建立真实模型的端到端基线，并逐步把 KV 管理从文件池方案推进到独立的 KV 专用 FTL。新 FTL 与真实模型的集成仍在验证中。

**一句话概括：我们正在从“让模型使用 CEMU”，推进到“让 CEMU 模拟 InstAttention 的设备行为”。**

## 二、整体架构

### 2.1 应用、设备和物理执行环境

```text
Guest：从计算存储架构看，相当于使用 CSD 的应用主机
┌───────────────────────────────────────────────────────────┐
│ OPT checkpoint / tokenizer                                │
│                    ↓                                      │
│ 公共 Prefill / Decode 运行时                               │
│                    ↓                                      │
│ Guest GPU：Prefill、Decode QKV、Wo、MLP、LM head             │
│                    ↕                                      │
│ KV / Attention 后端适配层                                  │
│                    ↓                                      │
│ CEMU 驱动、FDMFS、NVMe 命令                                │
└────────────────────┬──────────────────────────────────────┘
                     ↓
物理 Host：QEMU 中运行的 CEMU 设备模型
┌───────────────────────────────────────────────────────────┐
│ 命令解析与任务管理                                         │
│                    ↓                                      │
│ SparF：选择 channel → 读取 K → 选择 token → 精确 Attention   │
│           ↙                          ↘                    │
│ KV 缓存和存储管理               CPU / CUDA 功能算子         │
│           ↓                                               │
│ FTL：逻辑映射、物理分配、内部读写与 NAND 时间模型             │
│           ↓                                               │
│ 独立 NVM 文件后端：保存真实数据                             │
└───────────────────────────────────────────────────────────┘
```

图中表示总体职责划分。旧 grouped 路径与新 native 路径的集成和验证程度不同，具体状态见第七节。

### 2.2 需要区分的两个 GPU 角色

| 角色 | 职责 | 不应混淆为 |
| --- | --- | --- |
| Guest GPU | 通过直通供 Guest 应用使用，运行大模型主体 | CSD 内部计算部件 |
| Host 上用于 CSD 功能计算的 GPU | 被 QEMU 调用，加速 Attention 等算子的真实结果计算 | 论文 FPGA 的实际执行性能 |

CPU/CUDA 后端负责算出结果；目标 CSD 的计算时间由时间模型描述。Host GPU 执行得快，不自动意味着目标 CSD 也具有相同速度。

同样，文件后端负责保存真实字节，NAND/FTL 模型负责描述目标存储行为。文件放在物理 SSD 上，不等于可以直接把该 SSD 的访问耗时当作目标 Flash 时间。

### 2.3 软件分层

| 层次 | 当前职责 |
| --- | --- |
| 模型适配层 | 解析 TinyLlama/OPT 权重，实现各模型的投影、归一化、位置编码和 FFN |
| 公共运行时 | Prefill、Decode、自回归生成和 microbatch 调度 |
| KV / Attention 适配层 | 将张量和逻辑请求转换为设备接口，管理资源生命周期 |
| CEMU 设备层 | 执行任务、组织内部数据读取和计算、维护存储状态 |
| 数据后端与时间模型 | 保存真实数据，计算目标资源占用和完成时间 |

模型相关数学操作与公共运行流程分离，避免为 TinyLlama 和 OPT 各自维护一整套调度代码。

**关键代码：应用运行时通过后端接口接入设备。** 以下节选来自 [OPT 测试入口](../../tests/cemu/instattention/python/test_opt_cemu_pipeline.py#L1460)：

```python
cemu_runner = make_cemu_decode_runner(
    config,
    loader,
    cemu_backends,
    gpu_batch_size,
    detail_logger,
)
```

`config/loader` 提供模型结构和权重，`cemu_backends` 提供设备 Attention 与 KV 接口。更换存储后端不需要重新复制整个模型运行流程。本文代码均为关键节选，完整校验和生命周期处理见链接源码。

## 三、真实模型的 Prefill 与 Decode 流程

### 3.1 Prefill：产生并保存历史 KV

```text
文本或输入 token
    ↓
Guest GPU 执行各层 Prefill
    ↓
产生每层 K/V，通过 KV 后端提交到 CEMU
    ↓
Guest GPU 完成最终输出计算，得到第一个生成 token
```

这一阶段的意义是把真实模型产生的 KV 引入设备，不再使用孤立的随机矩阵代表整个应用。

### 3.2 Decode：由 CSD 承担 Attention

```text
当前 token 对应的 hidden state
    ↓
Guest GPU：计算本层 Q/K/V
    ↓
提交新增 K/V，以及本层 Attention 请求
    ↓
CEMU：取得当前有效 KV，完成 Attention
    ↓
Guest GPU：执行 Wo、MLP，继续下一层
    ↓
完成全部层后，LM head 产生下一个 token
```

以上描述数据依赖，不表示要求所有阶段完全串行，也不据此宣称当前流水效率已与论文一致。

### 3.3 已建立的真实模型基线

我们从 TinyLlama 开始验证，再接入 OPT-125M 和 OPT-13B，完成了分片权重加载、模型配置解析和真实文本输入支持。

正确性采用分层检查，而不是只看最终能否生成文字：

- 自定义模型流程与 Hugging Face 对比。
- 使用相同 Q/K/V 检查局部 CSD Attention。
- 检查写入存储的 KV 内容。
- 分别检查多层误差、最终 logits 和生成 token。

**已有路径已完成真实 OPT-13B 的端到端运行和多组测试；这些结果不能直接作为新 native FTL 的验收结果。**

## 四、如何对齐 InstAttention 的计算与访问行为

### 4.1 从 Dense Attention 到 SparF

早期基线读取完整历史 KV，再执行 Attention。这可以验证设备计算链路，但还不体现 SparF 的数据选择机制。

后续增加了以下流程：

```text
Q → 选择 Top-r channel
          ↓
读取所选 channel 的历史 K
          ↓
计算近似分数并选择 Top-k token
          ↓
读取选中 token 的 K/V
          ↓
执行精确 Attention 及相应输出处理
```

这是主访问流程的概括，完整算子还维护所需的归一化、修正及中间状态。

关键变化不是简单减少一个矩阵的尺寸，而是：**后续要读哪些数据，由前一阶段的计算结果决定。** 因此设备需要支持“计算产生访问集合，再组织内部读取”。

### 4.2 从 Guest 组织细节到设备管理内部流程

早期 Guest 显式组织较多的数据读取、暂存和执行命令。后续增加设备自治工作流程，由设备在一次逻辑 Attention 请求内部推进各阶段。

**职责变化主要在于：应用提出计算需求，设备负责把需求展开成内部操作。** Guest 仍负责模型层次和 token 迭代，计算并提交 Q、新增 K/V 及必要元数据；设备则根据中间结果选择数据、组织读取并执行后续计算，不要求 Guest 在每个内部阶段之间往返下发命令。

一次自治 Attention 请求在设备侧大致经过以下过程：

1. **建立任务状态**：记录当前请求的参数、执行阶段和所需资源，保持 program/MRS 等资源有效。
2. **按结果组织访问**：选择 channel，读取相应 K；得到近似分数和 token 集合后，再读取所需 K/V。选择由计算逻辑产生，读取由设备存储接口执行，并非算子自行调用 Guest 文件系统。
3. **满足依赖后推进**：设备状态机区分等待 I/O、等待目标时间和等待计算完成，条件满足后继续下一阶段。
4. **返回最终结果**：完成输出后才结束这次 Attention 请求，Guest 随后继续 Wo、MLP 等模型计算。

“一次逻辑请求”不等于“一次 CUDA kernel”或“所有工作都在一个函数里完成”；内部仍可包含多次读取和计算。KV 接收与后台写回也有独立的状态和完成条件，不能把 Attention 返回当作所有 KV 已持久化，最终仍需 DRAIN 确认。

| 对齐维度 | 当前实现方向 |
| --- | --- |
| 计算分工 | 模型主体保留在 Guest GPU，Decode Attention 下放设备 |
| 稀疏访问 | 根据 channel 和 token 选择结果取得相应 K/V |
| 数据布局 | token-K、channel-K、token-V |
| 请求组织 | 应用提交逻辑 Attention，设备组织内部阶段 |
| 缓存与写回 | 设备接受新 KV，积累完整分组并后台写回 |
| 存储寻址 | 从文件池/LBA 逐步转向 KV 专用逻辑坐标 |

本节重点说明功能和职责的对齐。第六节进一步介绍异步机制；其资源调度、实际流水效率和模拟精度仍需分别验收。

## 五、近期重点：KV 布局、设备缓存与专用 FTL

### 5.1 为什么需要保存两种 K 布局

SparF 的两个阶段具有不同访问模式：

- 粗筛时，需要少量 channel 在历史 token 范围内的数据。
- 精确计算时，需要少量 token 的完整 K/V 向量。

因此 K 采用 token 和 channel 两种布局，V 采用 token 布局。它们来自 InstAttention 的设计需求，不作为我们的原创算法。

### 5.2 分组大小如何确定

以当前一种配置为例：页大小 4096 B，FP16，head_dim=128，每页包含 8 个 channel。

| 布局 | 计算方式 | 分组结果 |
| --- | --- | --- |
| token-K/V | 4096 / (128 × 2) | 每页容纳 16 个 token 的一个 head 向量 |
| channel-K | 4096 / (8 × 2) | 每页容纳 8 个 channel、256 个 token |
| 完整 channel-K 窗口 | 128 / 8 | 一个 256-token 窗口对应 16 页 |

这里的 16-token group 针对一个 KV head，不是整个模型所有层、所有 head 的 16 个 token。组大小由几何参数推导，并非所有配置固定为 16/256。

### 5.3 尚未写入 Flash 时如何读取

新增 K/V 先进入设备管理的有界尾部缓存。缓存中的 K 可以保留一份原始排列，在组成不同布局的页时再进行打包。

设备分别维护：

- 已接受的 token 前缀；
- token-K 已发布前缀；
- channel-K 已发布前缀；
- token-V 已发布前缀。

因此，同一时刻 token-K/V 可能已经落盘，而 channel-K 仍在积累。

读取时按照请求坐标和发布状态，从已发布的 Flash 页或者尚有效的尾部缓存取得数据，并恢复请求要求的排列。K 在两种布局都完成且没有在途读者占用之前，不能提前复用对应缓存。

完整分组由后台写回，最终 DRAIN 处理不足一组的尾部。真实 IO 成功和目标完成时刻到达是两个条件，不能只因内存中的数据已经复制完，就提前宣布目标操作完成。

### 5.4 从文件池到独立 KV FTL

旧 grouped 路径：

```text
KV 坐标 → 设备分组目录 → 文件池页号 → extent/LBA → FEMU 页映射
```

这条路径已经实现设备缓存和数据管理，但仍借助 Guest 文件池及通用 LBA 表达存储空间。

新 native 路径：

```text
task / generation / batch / layer / head / kind / group
                         ↓
                KV 专用逻辑地址与映射
                         ↓
                   物理分配与 PPA
                         ↓
                真实数据 IO + NAND 时间模型
```

新模式不依赖 Guest 为 KV 创建文件池和提供 extent，而是把 KV 身份、页映射和放置交给设备管理。原有模式仍然保留，在初始化时选择，不要求所有 CEMU 任务切换到专用 FTL。

**关键代码：新增 KV 使用逻辑坐标提交，而不是文件偏移。** [NativeKvStore.append_batch](../../tests/cemu/instattention/python/cemu_flexgen/native_kv_store.py#L92) 中：

```python
self._call(BUFFER_APPEND, coordinates(self.task, self.generation,
    batch=batch, layer=layer, head=head, token=token + start, valid=count), packet, retry=True)
```

命令携带任务身份、batch/layer/head、token 起点和数量；设备决定如何积累、打包和放置。该片段体现寻址职责的变化，不表示当前 APPEND 的传输粒度或开销已与论文完全一致。

当前可以表述为：**实现了 InstAttention 风格的 KV 专用寻址、布局和设备管理机制。** 不能据此声称完整复刻其所有物理放置、回收和调度策略。

## 六、异步机制：新增流程、原生能力与时钟对齐

本节按三个问题展开：**我们增加了什么异步机制？原生 CEMU 已有什么异步能力？这些异步任务怎样使用同一模拟时间？** 异步提交、实际重叠和准确模拟是三个不同的验收层次。

### 6.1 我们现在做了什么异步

新增机制分布在应用层和设备层，不是单纯增加几个线程。

| 位置 | 已增加的机制 | 依赖与适用边界 |
| --- | --- | --- |
| Prefill | 将 KV 写入交给后台 writer，主流程继续执行后续模型计算 | 在规定的等待点回收前一写入，结束前收拢全部写入；已有日志显示部分写入与计算区间交叠 |
| Decode 应用层 | 按 layer/microbatch 提交 Attention，再回收结果并执行 Wo/MLP | 不同 microbatch 可提供独立工作；同一序列的层间依赖不能跳过 |
| 旧 Dense 设备适配 | 分离 Load/Compute/Store，使用独立 FDM slots 和历史 KV 预取 | 读之前仍需等待相关写入可见，slot 在请求完成前不能被覆盖 |
| SparF 请求提交 | 多 slot 保存独立请求状态，自治模式由设备推进内部阶段 | 稀疏访问依赖当前 Q，不能直接照搬 Dense 的下一层选中 KV 预取 |
| CSD 内部 | 提交选中 V 的读取并执行精确 QK，PV 等待二者满足完成条件 | 目标重叠还取决于资源竞争与实际提交时刻；不能只看线程数或两段时长推断 |
| KV 存储 | APPEND 接受尾部数据，后台 worker 将完整分组写入 Flash | 接受不等于持久化，发布依赖真实 IO 与目标完成时间，最后统一 DRAIN |

此外，公共运行时保留了下一层权重预取接口。但当前部分实验会预加载并缓存全部权重，预取可能只是取得缓存引用，不能把每次 `prefetch` 都说成真实权重搬运与计算重叠。

应用层的典型机会是：microbatch 0 的 Attention 已提交后，继续准备 microbatch 1 的 QKV，再在需要结果时等待。**batch 全部放进一个 microbatch 时，不存在这项跨 microbatch 重叠机会。** 同一序列的下一层 QKV 仍依赖上一层 Attention、Wo/MLP 的输出。

这些能力分阶段实现，旧 Dense、staged SparF、autonomous/grouped 和 native 并非同一条完全相同的执行路径。特别是 native 的真实 OPT 全流程仍在验收，不能把旧路径观察到的重叠直接当作 native 路径的测试结果。Guest 暂停、GPU→CPU 中转和具体资源配置，也会限制实际重叠程度。

主要入口：[Prefill runner](../../tests/cemu/instattention/python/runtime_common/prefill_runner.py#L116)、[多 microbatch Decode](../../tests/cemu/instattention/python/runtime_common/multi_batch_decode.py#L152)、[SparF slot 调度](../../tests/cemu/instattention/python/cemu_flexgen/cemu_sparf_slot_scheduler.py#L1)。

### 6.2 原生 CEMU 的异步是什么样的

**原生 CEMU 并不是只能串行执行。** CEMU 论文 §3.2.3、图5讨论了单 CU 的 chunk 流水，以及多个 CU/CSD 上的并发。其基本处理对象是通用计算任务及数据 chunk，而不是 Transformer 的 layer、microbatch 或 SparF 选择结果。

一个 chunk 包含输入、计算、输出三个阶段。同一 CU 上，目标计算资源一次只能按照调度策略服务相应任务，但一个 chunk 的目标计算区间可以与其他 chunk 的输入、输出区间交叠；多个 CU 或 CSD 则能够进一步提供目标计算并行度。

```text
Chunk 0：输入 → 计算 → 输出
Chunk 1：       输入 → 计算 → 输出
Chunk 2：              输入 → 计算 → 输出
```

以上仅表示流水结构，具体交叠由依赖、资源和时间参数决定，不是测得的时间比例。

实现上需要分清以下层次：

- **命令和完成队列**：请求提交、设备处理和完成通知解耦；上层封装仍可能选择阻塞等待，因此不能说每个 API 都只入队即返回。
- **功能执行与目标调度分离**：功能计算线程产生结果和执行时间信息，调度线程按目标 CU、策略、时间片和执行预算推进任务；两者不是同一个职责。
- **Direct 与 Indirect**：Direct 由应用显式组织数据搬运和执行，多请求仍可并发；Indirect 可让设备依据传输描述组织分 chunk 的输入、计算、输出。Direct 不等于必然串行，Indirect 也不自动保证任意任务充分重叠。
- **多个 CU/CSD**：增加的是目标资源配置；Host 的功能线程数与目标 CU 数不能简单等同。

原有 [计算调度器](../../hw/femu/csd/sched.c#L616) 和 [Indirect 分阶段推进](../../hw/femu/csd/compute.c#L3266) 提供这些通用基础。我们新增的是模型级调度、数据相关的 SparF 状态机以及 KV 后台管理，而不是从零为 CEMU 创造异步能力。

### 6.3 原生 CEMU 如何对齐异步状态下的时钟

这一节讨论 CEMU 的通用机制，不限于 InstAttention：当多个异步 I/O、计算任务、CU 或 CSD 同时工作时，如何保持时间一致。需要区分流水效率与模拟时间语义，本次不把流水效率作为完成成果。

**核心不是各线程拥有一只钟再相互校准，而是同一 QEMU 实例的设备模型共享虚拟时钟，并分别维护任务和资源的时间状态。** 这是 CEMU/QEMU 提供的基础，不作为此次复现工作的原创机制；不同 QEMU 实例不会因此自动同步。

#### 两条时间轴，而不是给每个线程单独计时

| 时间轴 | 用途 | 当前代码依据 |
| --- | --- | --- |
| Host 真实时间 | 衡量功能执行、线程等待和模拟器运行成本 | `CLOCK_MONOTONIC` 等实际计时 |
| 设备模拟时间 | 表达任务起点、资源可用时刻、I/O deadline 和计算完成 | `clock_ns()` 读取 `QEMU_CLOCK_VIRTUAL` |

多个设备任务的目标时间应落在同一模拟时间轴上。不能把一个线程的 Host 时间戳与另一个任务的虚拟 deadline 直接比较，也不能把 CUDA event 的持续时间当作该时间轴上的绝对完成时刻。

**关键代码：共同时间来源。** [clock_ns()](../../hw/femu/nvme.h#L1544) 的有效实现为：

```c
static inline int64_t clock_ns(void)
{
    return qemu_clock_get_ns(QEMU_CLOCK_VIRTUAL);
}
```

#### 依赖与资源共同决定执行时间

对于一个阶段，其目标起点至少受到请求到达、前置依赖完成和所需资源可用的共同限制：

```text
T_start  = max(T_arrival, T_dependencies_ready, T_resource_available)
T_finish = T_start + T_modeled_service
```

这里是建模原则，不是宣称每条现有路径都已精确实现这一简化公式。实际还要考虑调度策略和资源占用；共享一个 CU、Flash LUN 或其他资源的请求，不能仅因分属不同线程就按完全并行处理。

这不仅是应用层依赖：原有 [NAND 模型](../../hw/femu/bbssd/ftl.c#L488) 用 `next_lun_avail_time` 约束同一 LUN 的后续操作；[计算调度器](../../hw/femu/csd/sched.c#L578) 根据 CU 的运行任务、时间片和虚拟时间差累计执行量。多个真实线程不等于多个目标计算资源。

以一般的计算任务 A 和 I/O 任务 B 为例，若后续任务 C 依赖二者：

```text
A 目标完成 ───────┐
                 ├─ 两者均完成，并且所需资源可用 → C
B 目标完成 ───────┘
```

C 不能按两个持续时间简单相加后启动，也不能只等待其中一个。若假设两者同时开始、无其他竞争，A 为 300 us、B 为 500 us、C 为 100 us，则该段目标时间为 600 us，而不是 900 us。该数值仅用于解释依赖，不是实验测量。InstAttention 的 QK/V→PV 是这种一般依赖的一个实例。

#### freezing 期间可以做什么

现有计算段通过暂停 Guest vCPU 和 ticks，将功能计算与目标计算时间分离。此次工作不修改这一机制。

冻结影响的是公共时间轴，而不是某个任务的局部时钟。例如虚拟时间为 100 us、某请求 deadline 为 500 us，此时 Host 用 2 ms 完成功能计算：解冻后虚拟时间从暂停位置继续，deadline 仍为 500 us，不必给所有在途请求分别加减这 2 ms。QEMU 的 [ticks 启停和 offset 调整](../../softmmu/cpu-timers.c#L107) 保持时钟连续。

冻结并不表示所有 Host 线程均停止。已在设备侧的工作可以在不受相关锁阻塞的情况下组织内部 I/O、进行真实数据操作并计算目标 deadline；但模拟时钟没有推进，就不能因为真实操作已经结束而提前宣布目标完成。Guest vCPU 暂停期间，也不能指望 Guest 应用继续下发新的命令。

因此，必须分开判断：

1. 功能数据是否已经正确产生或读回；
2. 模拟时钟是否已经到达目标完成时刻。

**关键代码：结果就绪不等于立即对 Guest 完成。** 原有 [NVMe 完成处理](../../hw/femu/nvme-io.c#L236) 包含：

```c
now = clock_ns();
if (now < req->stat.expire_time) {
    break;
}
```

请求进入完成队列后，仍要满足虚拟 deadline；这体现通用的完成门控，而非仅 SparF 新增的逻辑。

等待未来模拟时刻的逻辑应位于允许时钟继续推进的位置，不能持有冻结区并等待只有解冻后才能到达的 deadline。

#### 多个计算与访存任务并发时，如何统一处理

考虑一个完整场景：设备上同时存在计算任务 A、B，以及 NVM/FDM 读取或搬运任务 R。**它们共用虚拟时间，但由不同机制管理执行和完成；不是每个任务单独冻结，也不是等待所有任务都结束才解冻。**

当前代码中，[计算区入口与出口](../../hw/femu/csd/compute.c#L1496) 用 `compute_active` 记录活动功能计算区，用 `compute_owner` 指定本轮冻结的发起线程，锁和条件变量保护计数与唤醒。与此同时，计算调度器维护各任务的执行预算和 CU 状态，I/O 路径维护资源可用时间及 `expire_time/ready_ns`。

**一次并发执行可以分成四步：**

1. **冻结建立**：A 首先进入功能计算区，成为 owner，暂停 Guest vCPU 和公共时钟。此时 R 若已在设备队列中，其目标完成时间保持不变；Guest 则不能继续执行普通访存或下发新请求。
2. **设备侧继续工作**：已排队的 B 可以进入功能计算区，增加活动计数并共享冻结；R 的 Host worker 在锁和依赖允许时，也可以继续实际读取或搬运。独立 R 不因此加入 `compute_active`，但其模拟时间同样不会前进。
3. **决定何时解冻**：若 A 先算完而 B 尚未退出，A 等待；活动计算区归零后，仍由 owner 恢复时钟。如果最后退出的是 B，则由 B 唤醒 owner。此时不要求独立 R 已达到目标 deadline，否则可能陷入“等时钟前进才能解冻”的循环等待。
4. **分别完成任务**：解冻后，计算按各自的目标预算和 CU 调度继续计时，R 按存储资源和目标完成时刻等待。某个后续任务依赖 A、B、R 中的哪些结果，就等待相应条件，不强制无关任务一起完成。

**关键代码：恢复公共时钟等待的是活动计算区，而非全设备 I/O 清空。** [leave_compute_section](../../hw/femu/csd/compute.c#L1507) 节选如下，外围计数更新、锁和非 owner 唤醒分支省略：

```c
if (qemu_get_thread_id() == compute_owner) {
    while (compute_active != 0) {
        qemu_cond_wait(&compute_cond, &compute_mutex);
    }
    cpu_enable_ticks();
    resume_all_vcpus();
}
```

下面用同一张时间线表示三项工作。横轴为 **Host 真实时间**，假设 R 不受锁阻塞，省略冻结建立和恢复开销：

```text
Host 时间       0 ms       1 ms       2 ms                   5 ms
A（owner）      [ 功能计算────────────] 等待 B ─────────────→ 恢复 VM
B                          [ 功能计算───────────────────────] 唤醒 A
R（独立读取）   [ 实际读取────────────] 数据就绪，目标完成仍需等待
虚拟时间        └────────────────停在 T0────────────────────┘ 继续推进
```

这里 A、B 共享冻结区间，重叠时间不重复扣除。R 在第 2 ms 读回数据，不表示它在模拟时间中也已完成。假设 T0=100 us、R 的目标 deadline=500 us，解冻后仍要等虚拟时间到达 500 us，且数据有效，才能发布完成。若 R 的实际读取反而较慢，则即使 deadline 已到，也不能把未就绪数据当作成功结果返回。

还要注意两点：

- **真实并行不等于目标并行**：A、B 的 Host 功能线程可以重叠，但目标侧是否并行取决于 CU；R 是否排队取决于 LUN、链路等资源。原有 [NAND 模型](../../hw/femu/bbssd/ftl.c#L488)、[FDM 读路径](../../hw/femu/csd/memory.c#L34) 和 [完成门控](../../hw/femu/nvme-io.c#L236) 使用共同时间基准，但并不套用同一种服务时间公式。
- **区分独立 I/O 与计算内部访存**：算子内部必需的访存、搬运属于其功能执行的一部分，真实数据依赖仍须满足；不能在冻结区内等待未来虚拟时刻。真实直通 GPU 等外部硬件也不自动随 Guest 暂停，需单独审查其完成语义。

**总结：活动计算区决定何时解冻，公共虚拟时钟统一时间，各计算和访存任务的资源状态、数据依赖与目标完成条件决定何时对外完成。**

#### 原有时间基础与新增流程的接入

- 原有基础：公共 `clock_ns()`、功能执行/目标调度分离、NAND 资源时间和 NVMe 完成门控。不能把这些通用机制算作此次复现新增。
- 新增自治状态机：`hw/femu/csd/compute.c` 的 `sparf_autonomous_main()`，区分物理读取、目标时间等待和计算调度完成；阶段推进见 `hw/femu/csd/sparf_workflow.h`。
- 新增后台写回：`hw/femu/bbssd/instattention_buffer.c` 在真实写入成功后，等待 `ready_ns`，再更新对应布局的 committed 前缀。

这些机制提供了时间语义的基础，但**共用一个时钟并不足以证明整个异步系统准确**。仍需检查阶段提交是否过晚、Host 调度是否造成额外空隙、资源竞争是否正确计入，以及是否出现提前发布、重复收费或遗漏时间。

尤其是冻结期间，某些 Host 调度代码仍可能运行，但虚拟时间增量为零，其真实耗时不会自然成为模拟开销。若目标设备应承担相应开销，就需要模型表达，不能认为“统一时钟”会自动补齐。真实直通 GPU 等外部硬件也不自动随虚拟时钟停止，需要另行核查完成语义。

汇报时可以概括为：**CEMU 通过公共虚拟时钟、全局 freezing 和目标完成机制提供异步建模基础。我们的工作要在这个基础上正确接入设备流程，并验证依赖、资源竞争和事件发布；当前不把微测试通过等同于端到端异步精度已验证。**

## 七、验证证据与当前边界

| 内容 | 当前进度 | 可以证明什么 |
| --- | --- | --- |
| 真实模型基线 | TinyLlama、OPT-125M、OPT-13B 已有测试结果 | 模型、KV 与 CSD 计算可以组成端到端流程 |
| grouped KV 路径 | 已接公共 OPT 流程并运行 | 设备目录、缓存和 Attention 可以接入真实应用 |
| native KV 协议 | Guest 小测试通过 | KV 坐标寻址、读写及隔离检查有效 |
| native 有界缓存 | Guest 边界测试通过 | 双布局、分组边界、环复用及最终写回有效 |
| native 稀疏读与 SparF | Host 回归通过 | 新存储接口可为实际 SparF 状态机和算子提供数据 |
| native FDM/MRS/compute | 隔离 QEMU 基本命令链已通过 | 新 NVM 路径可以连接设备计算资源和 Attention |
| native 公共 OPT 入口 | 小型合成 OPT 已通过隔离 QEMU 数值验证 | 两层、双 batch、三步 Decode；真实 Guest GPU/OPT-125M/13B 仍待验收 |

需要区分不同层次的 PASS：

- 功能流程完成，不等于与 Hugging Face 全部数值严格一致。
- 数值验证通过，不等于模拟时间已经校准。
- 旧 grouped 路径通过，不等于新 native 路径通过。
- 小规模通过，不等于论文规模的容量、生命周期和资源竞争已验证。

本轮集成仍需完成后续回归和真实 Guest 模型验收。任务回收、GC 和重启恢复也不能列为已完成能力。

## 八、工作价值与下一步

### 8.1 当前工作价值

这项工作不是重新提出 SparF，而是扩展和验证 CEMU 对这类计算存储应用的支持：

1. **真实应用接入**：由真实权重产生 KV，形成完整自回归链路。
2. **设备行为扩展**：支持数据相关选择、内部读取、分阶段计算和后台写回。
3. **专用存储建模**：增加 KV 逻辑地址、双布局和专用映射/分配路径。
4. **可验证性**：分别检查模型数值、存储一致性、边界行为和时间语义。

这些工作为后续研究模拟精度提供基础，而不是当前就宣称模拟精度已经得到验证。

### 8.2 下一步安排

1. 完成 native 集成的安全性和生命周期回归，包括失败、取消、重置及资源释放。
2. 从小规模真实 OPT 开始，验收 native Prefill → KV 存储 → SparF Decode → 最终 DRAIN。
3. 逐步扩大 batch、上下文长度和 Decode 步数，检查存储与缓存预算。
4. 固定行为与配置后，开展时间模型校准、论文实验对照和误差归因。

计算时间、NAND/I/O 时间、Guest 应用计时和物理 Host 实际耗时将分别记录。现有 GPU Softmax 形状比值属于估计方法，不作为已验证的 FPGA 时间模型。

## 九、汇报组织建议

建议使用 9 页 PPT，控制在 10～15 分钟；异步相关内容占两页：

| 页码 | 标题 | 重点 |
| --- | --- | --- |
| 1 | 目标与整体架构 | Guest、CEMU 设备、两类 GPU 的角色 |
| 2 | 真实模型执行流程 | Prefill 产生 KV，CSD 完成 Decode Attention |
| 3 | 对齐 InstAttention 的哪些行为 | 计算分工、SparF 访问流程、设备职责 |
| 4 | KV 双布局与设备缓存 | 为什么分组、未落盘如何读取、何时允许复用 |
| 5 | 从 grouped 到 native FTL | 为什么文件池还不够，新逻辑地址路径如何工作 |
| 6 | 我们新增的异步与原生能力 | 模型/设备新增机制，与原有 chunk/CU 异步分开 |
| 7 | 异步状态下的模拟时间对齐 | 双时间轴、全局 freezing、资源时间和完成门控 |
| 8 | 验证结果与当前边界 | 旧路径结果、新路径结果和未完成项分开 |
| 9 | 下一步 | 完成真实模型迁移，再评价模拟精度 |

### 开场表述

> 我们近期主要在 CEMU 中复现 InstAttention 的计算存储行为。工作先从真实模型接入开始，再逐步补充稀疏 Attention、KV 双布局、设备缓存和专用 FTL。本次重点介绍整体架构以及这些机制如何与目标设计对齐。

### 结束表述

> 目前已有路径完成了真实 OPT 模型的端到端运行，新 KV 专用 FTL 的核心机制也完成了分层验证。接下来先完成新存储路径与真实模型的集成验收，再固定实验配置，对比模拟结果和论文结果，分析差异来自哪里。

## 参考材料

- [开发历史与保存基线](DEVELOPMENT_HISTORY.md)
- [专用 FTL 工作记录](../../tests/cemu/instattention/FTL_WORKLOG.md)：按阶段追加，早期状态描述需结合后续章节阅读。
- [公共 OPT 测试入口](../../tests/cemu/instattention/python/test_opt_cemu_pipeline.py)
- [Native KV 应用适配层](../../tests/cemu/instattention/python/cemu_flexgen/native_kv_store.py)
- [Native FTL 与 Attention 桥接](../../hw/femu/bbssd/instattention_attention.c)
- [隔离 QEMU 计算链路测试](../../tests/cemu/instattention/python/test_native_csd_pipeline.py)
- [InstAttention 本地论文](../InstAttention_In-Storage_Attention_Offloading_for_Cost-Effective_Long-Context_LLM_Inference.pdf)
- [CEMU 本地论文](../CEMU_ASPLOS_26_DRAFT.pdf)：§3.2.3、图4～5介绍 freezing 与 chunk 流水。
