# 物理阶段、InstAttention 模拟与当前 CEMU 的对应表

2026-09-26。范围为当前工作区 native KV + autonomous SparF + 单 microbatch OPT 路径，包含尚未提交的整阶段页去重、批量 APPEND 修改。本文只建立行为对应关系，不给出测量结果，不调整模拟参数。

表中计算目标预算的公式针对加载 `--runtime-config` 的路径；未配置 runtime 模型的 smoke 路径可以使用零目标预算，仍执行功能计算，不能据其耗时评估目标硬件。

“真实物理阶段”以论文设计的 GPU + InstCSD 系统为参照。InstCSD 包含设备 DRAM、控制器/FTL、NAND 通道及 FPGA 计算单元，不是把普通 SSD 当成计算设备。下表按数据依赖排列；后台写回以及明确可重叠的阶段，不要求按行号依次完成。

## 证据边界

- **硬件行为**：论文 §IV、Algorithm 1、Fig.7–11 描述了数据路径和算法；§V-A 描述真实 OpenSSD 部署。
- **作者的模拟方法**：§V-B 明确说先采集真实 InstCSD 的细粒度延迟，再基于 NVMeVirt 在 Host CPU 侧设置相应延迟以反映处理引擎速度。Table I 校验 GeMV、Softmax、Filter，8 条 Flash 通道各 1.4 GB/s，外部 PCIe 4.0×4 最大约 7 GB/s。
- **未知项**：论文没有给出逐阶段模拟实现、CPU 功能执行与目标延迟的结算方式、完整队列/缓冲参数及各个 barrier。公开仓库可见 FPGA bitstream、OpenSSD FTL、FlexGen、P2P DMA 代码，尚未定位到作者修改的 NVMeVirt 源码。因此表中“未披露”表示无法确认，**不表示作者没有模拟这个阶段**；也不能拿上游 NVMeVirt 的默认行为替作者填空。
- **CEMU 实现**：以生产代码为依据，区分真实执行的功能操作和另加的目标时间。详细调用链见 [NATIVE_SERIAL_DATAFLOW.md](NATIVE_SERIAL_DATAFLOW.md)。

论文原文：[本地 HPCA 2025 PDF](../../../docs/InstAttention_In-Storage_Attention_Offloading_for_Cost-Effective_Long-Context_LLM_Inference.pdf)。公开实现：[ChaseLab-PKU/InstAttention](https://github.com/ChaseLab-PKU/InstAttention)。以下 P1–P7 和 C1–C9 的出处列在表后。

## 按当前汇报文档的简版对应

本节按用户提供的“当前实现流程”顺序组织。三列分别说明目标硬件实际做什么、CEMU 如何实现同一功能、作者 NVMeVirt 已披露哪些实现依据。“未披露”不是没有该功能，而是不能确定其具体代码和计时边界。

建议将“假的行为”表述为“功能实现与目标硬件时序模拟”：Guest GPU 模型计算和 Host CPU/CUDA 功能运算都真实执行；模拟的是它们所代表的 CSD 硬件资源、存储行为及完成时间。不是每个软件函数都对应一个独立物理阶段。

| 文档位置 | 真实 InstAttention / InstCSD 行为 | 当前 CEMU 如何对应 | 作者 NVMeVirt 如何对应：证据与未知 |
|---|---|---|---|
| Prefill | GPU 完成本层完整 Attention 和其他模型计算，产生 KV；层间计算与 KV 传送可重叠。[P1] | Guest GPU 实算；单 writer 导出、追加本层 KV，与 GPU 工作部分重叠。 | 模型 GPU 仍是真实 GPU；NVMeVirt 用于虚拟 InstCSD。其层间提交、等待与写完成边界未披露。[P5] |
| KV 写入：导出、打包、ioctl、PRP | GPU 将 KV 经 P2P DMA 传给 CSD，目标是设备接收这些 KV。[P1,P4] | GPU→Guest CPU→打包→PRP→Host wire buffer→tail；这些软件步骤共同实现一次 KV 接收。 | 给出虚拟设备外部 PCIe 4.0×4、最大 7 GB/s 配置；虚拟环境是否采用相同 P2P 路径、包格式和拷贝策略未披露。[P5] |
| tail 接收及 accepted 检查 | Decode 增量 KV 先进入设备 DRAM group buffer，等待凑组。[P3] | Host K/V ring 代替该 buffer；accepted/reader/容量检查维护可见性和背压。 | buffer 容量、ACK 时点、重试协议和是否采用 accepted/committed 机制未披露。 |
| tail→token-K/channel-K/token-V | 设备组织双 K 布局与 V 布局，经 FTL 映射后写 NAND；满组后台写，并利用跨 head 批处理。[P2,P3] | buffer worker 复制/转置，分配 PPA、写 backend；用 NAND deadline 控制 committed 发布。 | 给出 8 条 Flash 通道、每条 1.4 GB/s 的扩展配置；转置、映射、页请求、写回发布的具体实现未披露。[P5] |
| Decode 前半层 | GPU 做 LN/QKV 投影，新 K/V 传入设备。[P1,P4] | Guest GPU 实算；新 K/V 先 APPEND，全部接收后继续本层 Attention。 | GPU 计算由真实 GPU 执行；虚拟 CSD 的逐请求依赖及提交粒度未披露。 |
| Q/mean 上传与 Execute | Q 传入 CSD，控制命令触发设备 Attention；SparF 需要 mean(V) 补偿输入。[P2,P4] | Q/mean 经 FDM pwrite 到 MR0/MR4；父 Execute 引用 MRS 并带 metadata。 | Q 传输/Attention 调用有设计依据；mean 在何处维护、怎样传送、是否单独计时未披露。没有证据说作者也使用 FDM/MRS。 |
| Execute 内选择维度和近似筛选 | FPGA 的选择/计算单元执行 Q top-r、近似 logits、softmax、token top-k 和 alpha 求和。[P2] | Host CPU 实算，近似阶段再提交目标计算预算；top-r/top-k/sum 没有各自独立目标预算。 | §V-B 明确以真实 InstCSD 细粒度延迟校准 Host CPU 上的虚拟引擎；各操作如何拆分、计时和排队未披露。[P5] |
| Execute 内三段 KV 读取 | 根据维度/token 索引查 FTL，读相关 NAND 页，由 NFC 筛出所需元素交给计算引擎。[P2,P3] | 内部 gather 按布局 committed 分流到 backend 或 tail；整阶段页去重、scatter 到 MR1/2/3，并等待读 deadline。 | Flash 通道配置已披露；同页去重、tail 命中、页内过滤流水及每阶段读请求组织未披露。 |
| Execute 内 exact QK / PV | FPGA 执行精确 QK/softmax、PV 和 mean 补偿；V 读取与 QK 计算并行。[P2] | CPU/CUDA 实算后等待 CU 目标预算。native V gather 先于 exact 功能计算，重叠发生在目标时间轴。 | 已披露使用实测引擎延迟校准，Table I 给出主要单元校验；QK/V 重叠及功能耗时与目标延迟的结算实现未披露。[P5] |
| 输出回读 | CSD 将 Attention 输出经 P2P DMA 返回 GPU。[P1,P4] | MR9→Guest CPU→Guest GPU；CUDA target 可能先做 Host GPU→FDM 同步。MR10 trace 是辅助流量。 | 外部链路配置已披露；输出拷贝、完成通知和测量流量的计时方式未披露。 |
| Decode 后半层 | GPU 做 Wo、残差/LN、FFN，继续下一层及下一 token。[P1,P4] | Guest GPU 实算；单 microbatch 同步等待前述结果。 | GPU 部分真实执行，虚拟 CSD 对请求何时完成的定义仍影响等待时间。 |

读表时另保留三个边界：

1. **功能对应不等于数据通路相同**：CPU 中转对应“传送 KV”这个任务，但不等价于 P2P DMA；FDM 对应设备工作缓冲，但文件、pwrite、MRS 是 CEMU 的访问机制。Host CPU/CUDA 对应 FPGA 的数值功能，也不具有同样的硬件时延。
2. **目标时间对应不等于已经校准一致**：CEMU 有 PCIe、NAND/LUN 和 CU 预算模型，但当前 Flash 通道传输时序未启用，选择/归约等没有完整独立预算；作者仅公开总体校准方法，不能直接逐函数核对。
3. **外部 APPEND 合包不等于内部 Flash 批处理**：前者合并同 batch/layer 的 head 数据以减少命令；论文跨 head 的写入批处理面向 NAND 页/块组织。两者都处理多个 head，但属于不同阶段。

原文建议修改：

- “一个 Execute 读取 KV 时查是否已经提交”改为“先确认 KV 已 accepted；读取时按对应布局的 committed 判断：已发布部分读 NAND backend，未发布部分读 tail”。accepted 只表示设备已接收，三种布局的 committed 可以不同。
- “提交一个父 Execute”后注明“一层当前 microbatch 的 Attention 工作单元；这是 CEMU 的命令粒度，尚不能证明等于作者 NVMeVirt 的单命令粒度”。
- 补充“本次运行末尾有 terminal drain；论文未明确同等屏障是否包含在端到端计时中”。

## 逐阶段对应

| ID / 真实物理阶段 | InstAttention 那里如何模拟：已知与未知 | 当前 CEMU 如何模拟：功能执行与目标时间 |
|---|---|---|
| **S00 权重准备与访问**：GPU 算子需要权重位于可访问显存；权重驻留/搬运取决于软件放置策略 | 测试使用真实 A6000、FlexGen 衍生软件；Fig.18/19 有 Weight Access 类别，但该类别的精确计时边界未披露，不能认定等价于模型首次加载。[P5,P6] | `preload_weights()` 在正式 Prefill 前加载全部层并缓存在 Guest GPU；后续 `load_layer()` 命中缓存。GPU 运算仍会读显存权重，但本路径没有每层外部权重 offload。初始化预载位于本次 functional 推理计时之外。[C1] |
| **S01 Prefill GPU 计算**：embedding → 每层 QKV → causal QK/softmax/PV → Wo/FFN → 输出头 | 论文明确 Prefill 全部计算保留在 GPU；实验使用真实 GPU，不是 NVMeVirt 代算模型 GPU 层。[P1,P5] | 在 Guest 直通 GPU 上实际执行 PyTorch 算子。每层 `prefill_attention()` 返回后提交 KV writer；GPU 的 MLP/后续层计算与 writer 存在重叠。[C1] |
| **S02 GPU→CSD 传送 KV**：Prefill 传一段序列；Decode 传当前 token 的 K/V | 硬件设计为映射 GPU VRAM 后进行 P2P DMA，绕过 Host DRAM；虚拟配置给出外部 PCIe 7 GB/s。作者 NVMeVirt 的具体 DMA/copy 路径、传输调度实现未披露。[P1,P4,P5] | Guest GPU `.cpu().numpy()` → Guest CPU 打包 → `/dev/ng0n1` APPEND → PRP copy 到 Host。当前先协商能力，支持时用最多 128 KiB 的 `BUFFER_APPEND_BATCH`，合并同 batch/layer 的多个 head，无逐 head 4 KiB padding；旧单 head/4 KiB 协议为回退。逐包同步提交，成功数据命令由 CQ 计算 PCIe 目标延迟；没有 GPU/CSD P2P。[C2,C8] |
| **S03 设备 DRAM 接收与等待凑组**：新 KV 先进入设备 group buffer，不必立即 program NAND | §IV-C/Fig.11 明确描述 Decode 增量 KV 先缓存、满组后后台写回。虚拟设备是否采用同样 buffer 容量、背压及 ACK 时点，未披露。[P3] | `native_kv_buffer_append_batch()` 先检查整包 head 的前缀、reader 和容量，再真实复制到 Host 堆内存 K/V ring，推进 accepted。APPEND 完成不代表 NAND 完成；容量不足返回 pending，Guest sleep/retry。没有独立的设备 DRAM 时序模型，Host 内存操作实际消耗时间。[C2,C3] |
| **S04 KV 布局整理**：K 保存 token-indexed 和 channel-indexed 两份，V 保存 token-indexed 一份；按页组织数据 | 论文明确双 K 布局和两套索引，token 组为 4 KiB 页；channel 分组可根据输入长度调整。虚拟设备转置、复制和元数据操作怎样计时未披露。[P2,P3] | `pack_batch()` 在 Host CPU 真实复制/转置。FP16、d=128、每页 8 维时，token-K/V 每页 16 token；channel-K 为 256 token×8 维，每个完整维度组 16 页。参数在注册时固定，不是运行中自动调整。[C3] |
| **S05 FTL 映射与多 head 写批次**：逻辑坐标定位物理页，尽量把多个 head 的组放入同一块，沿 token 维度分散到 Flash 通道 | §IV-C 明确坐标索引、多 head 合批和通道条带化；NVMeVirt 侧准确地址放置、批次策略和控制器执行成本未披露。[P3] | native mapping 以任务/代际/batch/layer/head/kind/group 寻址，`group % channels` 选通道；分配域允许不同 head 共用块。worker 一次处理至多 `batch_pages` 页，并不保证每次形成整块写入，也没有等待攒满整块才提交的条件。[C3,C4] |
| **S06 NAND 写入与发布**：数据经过控制器和通道送入 NAND，完成 page program，更新可读状态 | 作者扩展为 8×1.4 GB/s Flash 配置；每类 program 延迟、通道争用、写队列及完成事件实现未披露。[P5] | 真实字节写入 NVM backend；另经 `ssd_advance_status()` 推进 LUN 的 `pg_wr_lat`/排队 deadline。worker 等虚拟时间到达后推进 committed。**通道数据传输计时在 `#if 0` 内，未启用**。backend 字节写完、目标写完成、发布是三个事件。[C3,C4] |
| **S07 Prefill→Decode 的可见性边界**：开始 Decode 时需要前缀 KV 可访问；是否必须全部在 NAND 是额外完成约定 | §IV-C 文字按 Prefill KV 已传输并存入 Flash 后继续 Decode 描述；没有明确给出 NVMeVirt 的 flush/barrier 调用或 tail 是否须清空。[P3] | Prefill 等 Guest writer 完成；native `flush()` 只检查 ready，不 drain。Decode 可以从 committed Flash 页和未发布 tail 合并读取。不能据 Prefill 完成推断历史 KV 全部落 NAND。[C1,C2,C3] |
| **S08 Decode GPU 投影**：当前 token embedding、LN、q/k/v projection | 论文把 qkv Proj. 留在真实 GPU。[P1,P4,P5] | Guest GPU 实际计算 Q/K/V，Q 已乘 `1/sqrt(d)`；先通过 S02/S03 追加当前 K/V，再进入 Attention。[C1] |
| **S09 维护 V 均值**：为 SparF 输出补偿项准备 `mean(V)` | Algorithm 1 将均值作为输入，给出最终 `(1-alpha)*mean(V)`；维护均值的处理器、更新算法、传输位置及模拟成本未明确说明。[P2] | Guest CPU 把 V 转成 float64 累加，记录 accepted token 数；每次生成 dtype 对应的均值并上传 FDM。该 CPU 更新包含在当前 append 计时范围内，是尚不能与作者一一核对的实现选择。[C2,C5] |
| **S10 传 Q / 启动 Attention**：GPU 向设备传 Q，并由控制命令触发引擎 | 硬件数据面为 P2P，控制面用扩展 NVMe 的 `config()/attend()`；虚拟设备的命令粒度、解析和队列时序未披露。[P4,P5] | Q 导出 Guest CPU；Q/mean 分别通过 FDM `pwrite` 写 MR0/MR4，随后一个 Execute 携带 metadata/MRS ID。Host 获取 KV lease 并排队进入 engine；一次父命令覆盖本层全部 B×H vector。FDM 文件/MRS 是 CEMU 的实现层，论文没有同名物理阶段。[C5,C6,C8] |
| **S11 Q top-r 选择**：argtopk 单元按 `abs(q)` 选择重要维度，形成 channel 索引 | 算法/硬件有独立 argtopk；Fig.21 有 Argtopk 类别。作者采用硬件延迟校准的总体方法，但 top-r 在 NVMeVirt 的独立参数和调度未披露。[P2,P5,P6] | Host CPU `qsort` 实际执行；CUDA target 也如此。没有独立 top-r 目标预算；它发生在 channel-K 读取前，现有 approximate functional 区间并不覆盖它。[C6,C7] |
| **S12 channel-K 页级选择与 NAND 读**：根据 top-r 找到需要的页，跳过完全不需要的页，再进行页读取 | SparF Filter-1 / 两级读取的第一步；论文声明 Flash 通道配置，但 NVMeVirt 页请求生成、去重及 DRAM 命中实现未披露。[P2,P3,P5] | native `gather_phase()` 对整个阶段的 Flash 页去重；committed 部分读 backend，未发布部分从 tail 取。Flash 页另累加 LUN `pg_rd_lat`/争用并取最大 deadline；没有完整 Flash 通道传输时序。[C3,C4,C6] |
| **S13 channel-K 页内过滤与输送**：NFC 从读出的整页中筛出所选维度，只把所需元素送进引擎 | SparF Filter-2 由 NFC 硬件执行；Table I 提供 Filter 校准值。每个过滤位置如何分别计时、与 Flash 传输如何流水，未披露。[P2,P5] | Host CPU 按页片段 scatter 到 FDM MR1；功能上确实筛掉无关元素。目标模型仅在 approximate 预算中加入一个 filter 估算，不为每段 scatter 建立独立 NFC 流水/带宽资源。[C3,C6,C7] |
| **S14 近似 logits 与 softmax**：引擎使用 Q 的所选维度及 channel-K 求近似概率，包含范数修正 | Fig.8 的第一个 Attention Kernel；GeMV/Softmax 采用实测延迟校准，Table I 仅给小规模锚点，任意 B/T/r 的外推规则未披露。[P2,P5] | Host CPU C 循环真实计算范数修正、点积、`expf`/归一化；CUDA target 也没有把这段移到 GPU。目标 `approximate_ns = filter + approximate_GeMV + approximate_softmax`，由 FLOPs/吞吐和 softmax 比例表估算，再送 CU 调度。[C6,C7] |
| **S15 token top-k 与 alpha 求和**：argtopk 选 token，sum 单元得到被选概率质量 alpha | Algorithm 1 步骤 10/11，Fig.21 的 Argtopk、Sum；独立虚拟延迟/队列实现未披露。[P2,P6] | Host CPU 排序选 token，并确保当前 token 被选中，再求 alpha。功能上与 S14 同段执行；**没有独立 top-k/sum 目标预算**，不能把三项物理工作等同于三段当前模拟时间。[C6,C7] |
| **S16 所选 token-K 的页级读与页内过滤**：读取完整 d 维 K，页级跳过无关组，NFC 丢弃页内无关 token | Algorithm 1 Filter-3/4；与 channel-K 一样，页级和元素级是两级硬件动作，虚拟实现细节未披露。[P2,P3] | 全阶段页去重 → backend/tail 取数 → Host scatter 到 MR2 `[B×H,k,d]`。LUN 目标时间单列 token-K read；其页内过滤实际执行，但没有另外独立的 filter 目标任务。[C3,C4,C6] |
| **S17 所选 token-V 的读/过滤，与 exact QK 重叠**：V 经 NFC 送向计算单元，尽量在 QK 计算时隐藏读取延迟 | §IV-B/Fig.8 明确描述 Q/K 进入第二 Attention Kernel 时并行加载 V；NVMeVirt 如何实现和结算该重叠未披露。[P2] | 实际 native gather 同步填 MR3，随后调用 exact QK 功能程序。目标读起点与 QK 目标区间允许重叠，再等待较晚完成者。**目标时间重叠不等于真实 Host backend 读与 GPU kernel 并发**。[C6] |
| **S18 exact QK 与 softmax**：用完整 Q、所选 K 求精确概率，再结合 alpha | 第二 Attention Kernel 的 GeMV/Softmax；按真实引擎延迟校准，但该阶段详细 NVMeVirt 代码未披露。[P2,P5] | 调 `.so` 的 exact phase，在 Host CPU 或 Host CSD GPU 真实计算，输出 MR8；目标 `exact_qk_ns = exact_GeMV + exact_softmax`。CUDA mirror 同步是 CEMU 为执行功能程序引入的数据搬运，不能直接当物理 FPGA 成本。[C6,C7,C9] |
| **S19 PV 与 mean 补偿**：概率乘所选 V，加 `(1-alpha)*mean(V)`，形成 Attention 输出 | Algorithm 1 步骤 19，GeMV 等引擎单元完成；平均项相关工作是否独立收费未披露。[P2,P5] | 把 mean 放进 MR3 额外一行，以 `1-alpha` 作为对应概率；CPU/CUDA PV 写 MR9。目标 PV 按含额外一行的 GeMV FLOPs 估算，没有独立“mean 补偿硬件单元”延迟。[C6,C7,C9] |
| **S20 CSD→GPU 返回输出**：输出经设备内部通路及 PCIe DMA 回 GPU | 硬件设计为 P2P 回传输出，外链配置 7 GB/s；虚拟返回数据复制和计时方式未披露。[P1,P4,P5] | 父 Execute 完成后，FDM `pread` → Guest CPU → `torch.as_tensor(...,device=GPU)`。CUDA target 还可能先同步 Host CSD GPU→Host FDM。另读 MR10 trace 属于测量辅助流量，论文算法输出路径没有这一步。[C1,C5,C8,C9] |
| **S21 GPU 后半层与下一 token**：Wo、残差、LN、FFN，逐层继续，最终 LM head/采样 | 论文明确这些留在 GPU，由 Host scheduler 协调。[P1,P4] | Guest GPU 真实执行；单 microbatch 等本层 Attention 完成才继续。之后遍历下一层、下一 Decode step。[C1] |
| **S22 跨阶段调度与并行资源**：GPU、DMA、DRAM/NAND、FPGA 引擎通过队列/依赖关系重叠运行 | 论文描述 Prefill 层间传输流水、两套可调度 Attention Kernel、QK/V 重叠、多 CSD 按 head 分工。NVMeVirt 每个资源的队列、时钟和争用实现未披露。[P1,P2,P4,P5] | 父 workflow 状态机、native 写 worker、engine/CU 调度共同实现。功能段会暂停 Guest vCPU/QEMU ticks，段外等待读 deadline 或计算子任务。Host CPU/CUDA 功能成本、Guest 等待、虚拟时间并存；这种 freeze 方法不能说成作者的方法。两个 slot/engine 也不等价于论文两套 FPGA kernel。[C3,C6] |
| **S23 生成完成 / 剩余 tail 的处理**：达到输出目标后结束生成；是否继续刷出临时 KV 是系统策略，不是每次推理必需的持久化语义 | 论文明确 KV 是临时激活、不要求持久保存；没有明确声明端到端计时包含“把所有未满组尾部写到 Flash”的 barrier。[P4] | functional 流水最后执行 terminal `BUFFER_DRAIN`：剩余组补零写出，等目标写完成并 sealed。此后不能继续 APPEND。它是当前实验收尾约定，不能默认与作者的推理终点相同。[C1,C2,C3] |
| **S24 过期任务回收 / NAND erase**：空闲时回收过期 KV 的块，恢复可用容量 | §IV-D/Fig.10 描述 Host `reclaim()`、FTL 查索引、擦除旧块；NVMeVirt 的触发条件、擦除计时及实验中是否触发未披露。[P4] | 当前 native 主线没有完整任务级 reclaim/块回收循环。关闭 fd 或 terminal drain 不等价于 erase/GC；通用 NAND erase 延迟函数存在也不证明这条生命周期路径已接入。[C2,C4] |

## 当前可以建立、还不能建立的对应关系

**物理功能上已有对应**：GPU 模型层、设备接收 KV、分组缓冲、双 K 布局、FTL 映射、三段稀疏读取、SparF 筛选与计算、输出返回。

**物理成本没有独立对应**：设备 DRAM 接收/访问、布局转置、控制器处理、各处 NFC filter、top-r/top-k/sum、设备内部数据通路。目前有的体现为 Host 功能开销，有的合并进 approximate 目标预算，有的没有独立目标预算。

**明确存在实现差异**：作者硬件的 P2P 与当前 CPU 中转；实际 Flash 通道传输与 CEMU 关闭的通道计时；硬件 NFC 流水与 Host gather/scatter；原生引擎并发与 CEMU 功能执行加虚拟等待；末尾临时 KV 的策略与当前强制 drain。

其中 S03 的“先到设备 DRAM、凑组后后台写 NAND”符合论文硬件设计方向。仍需区分的是组大小、跨 head 合批、背压、完成通知、tail 参与读取，以及 S07/S23 的边界；**不能因为都有 buffer 就判断整个存储时序已经一致**。

## 与论文图中类别的关系

| 论文类别 | 本表对应物理阶段 | 不能直接合并的部分 |
|---|---|---|
| qkvo Proj. | S08 的 QKV、S21 的 Wo | 当前操作 scope 还可能包含 LN/残差；GPU upload 不应算 projection |
| FFN | S21 的 MLP | LM head/采样不自动属于 FFN；论文对这些尾项的归账未细说 |
| Weight Access | S00 的权重访问/搬运 | 当前初始化预载不等价于论文 Decode 内的 Weight Access |
| KV Cache Access | S12、S16、S17 的 KV 获取；写入为 S03–S06 | 论文图中是否包含全部写回/控制开销未明确；Flash、tail、filter 不能仅靠名称归账 |
| qkvo transfer | S02 的 Decode K/V、S10 的 Q、S20 的输出 | 当前额外的 mean、CPU 中转、trace 不能不加区分地视为论文同项 |
| Attention | S11、S14、S15、S18、S19 等引擎工作 | 不应把整个 Execute 的读等待、队列、Host 功能开销一起归入纯计算 |
| 引擎图 Filter / Logit-0 / Argtopk / Sum / Logit / Attend | 大体对应 S13/S16/S17 的过滤、S14、S11/S15、S15、S18、S19 | 这是按操作含义建立的对应，非作者公开的逐段计时 API；Logit/Softmax 的精确统计边界仍不完整 |

Fig.18/19 是 Decode 的归一化分解，Fig.21 是引擎内部归一化分解。两张图的统计范围不同；本表不将它们转换成绝对时间，也不把真实硬件设计中存在的阶段自动当作 NVMeVirt 已披露的独立计时项。

## 来源索引

页码同时列出 PDF 页与论文印刷页，避免版本错位。

| 编号 | 论文依据 |
|---|---|
| P1 | §IV-A，Fig.7，PDF p6 / 印刷 p1515：GPU/CSD 分工、KV 数据流、层级流水 |
| P2 | §IV-B，Algorithm 1、Fig.8，PDF p6–7 / p1515–1516：SparF、NFC 两级过滤、argtopk/sum、QK/V 并行 |
| P3 | §IV-C，Fig.9/11，PDF p7–8 / p1516–1517：双布局、页组、DRAM buffer、跨 head 合批 |
| P4 | §IV-D，Fig.10，PDF p8–9 / p1517–1518：P2P、控制命令、GPU 协作、回收、KV 非持久数据 |
| P5 | §V-A/B、§VI-A、Table I，PDF p9–10 / p1518–1519：硬件部署、NVMeVirt 校准、通道/PCIe 配置、实验设备 |
| P6 | §VI-D，Fig.18/19/21，PDF p12 / p1521：Decode 与引擎归一化分解 |
| P7 | §VI-F、Fig.23，PDF p13–14 / p1522–1523：分组写、batching、写入负担分析；不能据此推导虚拟实现的每个队列参数 |

| 编号 | 当前代码入口 |
|---|---|
| C1 | [test_opt_cemu_pipeline.py](python/test_opt_cemu_pipeline.py)：`main/preload_weights/run_functional_pipeline`；[prefill_runner.py](python/runtime_common/prefill_runner.py)、[decode_runner.py](python/runtime_common/decode_runner.py)、[ops.py](python/opt_runtime/ops.py) |
| C2 | [sparf_attention.py](python/flexgen_adapter/sparf_attention.py)：`_kv/_append/_update_mean/_decode_args`；[native_kv_store.py](python/cemu_flexgen/native_kv_store.py)：`append_batch/flush/drain`；[instattention_native.c](../../../hw/femu/bbssd/instattention_native.c)：接收 APPEND |
| C3 | [instattention_buffer.c](../../../hw/femu/bbssd/instattention_buffer.c)：`native_kv_buffer_append/pack_batch/buffer_worker/native_kv_buffer_gather_phase/native_kv_buffer_drain` |
| C4 | [instattention_mapping.c](../../../hw/femu/bbssd/instattention_mapping.c)、[instattention_io.c](../../../hw/femu/bbssd/instattention_io.c)、[instattention_device.c](../../../hw/femu/bbssd/instattention_device.c)、[ftl.c](../../../hw/femu/bbssd/ftl.c)：坐标/PPA/backend/NAND 目标时序 |
| C5 | [cemu_sparf_device.py](python/cemu_flexgen/cemu_sparf_device.py)：`_prepare_decode/execute_autonomous/_collect_decode`；[cemu_client.cpp](guest/cemu_client.cpp)：FDM read/write、MRS、Execute |
| C6 | [compute.c](../../../hw/femu/csd/compute.c)：`sparf_submit_autonomous/run_functional_modeling/sparf_autonomous_main`；[sparf_workflow.h](../../../hw/femu/csd/sparf_workflow.h)：SparF 功能、读算依赖与目标预算提交 |
| C7 | [runtime_model.py](experiments/runtime_model.py)：`SparfAttentionRuntimeModel.estimate()`；[opt13b_sparf_1csd.json](experiments/configs/opt13b_sparf_1csd.json)：目标硬件参数；运行时可能加载另行校准的 softmax ratio |
| C8 | [memory.c](../../../hw/femu/csd/memory.c)：FDM 命令目标延迟；[nvme-io.c](../../../hw/femu/nvme-io.c)：SQ/CQ、PCIe 模型 |
| C9 | [sparf_attention_devptr.cu](../kernel/sparf_attention_devptr.cu)：exact QK/PV CUDA 功能算子；[backend.c](../../../hw/femu/backend/backend.c)：Host/CUDA mirror 同步 |

当前还不能补齐的核心资料是作者修改后的 NVMeVirt 阶段实现、各阶段时序公式、buffer/ACK/barrier 语义，以及论文统计区间的具体起止事件。这些空白应保留为未知项，不能用我们的代码反推作者的实现。
