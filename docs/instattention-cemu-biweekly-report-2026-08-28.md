# CEMU-GPU × InstAttention 两周进展汇报

日期：2026-08-28

## 一句话结论

InstAttention 的核心判断不是“SSD 适合跑完整 LLM”，而是：长上下文、大 batch 的离线推理中，Decode Attention 同时具有“必须反复读取大体量 KV Cache、计算强度又很低”的特征；把 KV 搬过狭窄 PCIe 到 GPU 会成为瓶颈，而把 Attention 移到 KV 所在的 CSD 内部，可以利用 flash channel 的聚合带宽并避免大规模 KV 外传。

过去两周，我们已经在 CEMU 中把这条思想从单独算子推进到真实 TinyLlama 端到端原型：Guest GPU 完成 Prefill 和非 Attention 算子，KV 落入 CEMU NVM，Decode Attention 通过 NVM→FDM staging、MRS 和 CSD CUDA device-pointer 路径执行，再回到 Guest GPU 完成后续层和 token 生成。同时补齐了多 chunk online softmax、GQA、真实 Hugging Face checkpoint、tokenizer/真实文本入口，以及 range-aware host/device 一致性管理。

但当前结论仍然是“功能路径和数值正确性已打通”，不是“已经复现论文性能”。当前还没有 SparF、论文的 flash-aware 双 K 布局/FTL、GPU-CSD P2P，以及校准到论文硬件参数的完整性能模型。

---

## 第一部分：InstAttention 为什么会得出这个设计结论

### 1. 问题从哪里来

LLM 推理分成 Prefill 和 Decode：

- Prefill 一次处理整段 prompt，矩阵较大，并行度高，GPU 利用率高。
- Decode 每次只生成一个 token。使用 KV Cache 后，不必重算历史 token 的 K/V，但每一步、每一层都要读取全部历史 KV。
- 因而 KV Cache 把 Decode 的复杂度从重复计算降下来，却把问题变成了持续增长的容量和带宽问题。

KV Cache 容量近似为：

```text
2 × layers × tokens × batch × kv_heads × head_dim × bytes_per_element
```

前面的 2 分别代表 K 和 V。它随上下文长度和 batch 线性增长，GPU 显存很快容不下。

#### 具体容量计算

论文正文用 `4bsp` Byte 表示 FP16 KV Cache、用 `2p` Byte 表示 FP16 权重，其中 `b、s` 是 batch 和 sequence length，`p` 是论文用于概括模型规模的记号。汇报时不建议直接拿这个简写做数值代入，因为真实 KV 大小还取决于层数、hidden size 和 KV head 数；直接讲下面的结构公式更严谨。论文据此强调的结论是：权重容量对固定模型不随 batch/context 改变，KV 却随 `batch × sequence length` 线性增长。正文给出的实例是：

- OPT-13B、sequence length 2K、batch 128：模型权重约 24 GB，KV Cache 约 200 GB；
- OPT-175B、相同 sequence/batch 量级：模型权重约 325 GB，KV Cache 可达 2.63 TB。

更通用、也更适合我们实现的逐层公式是：

```text
KV bytes
= 2(K/V) × L × B × S × Hkv × Dh × dtype_bytes
```

其中 `L` 是层数，`B` 是 batch，`S` 是历史 token 数，`Hkv` 是 KV head 数，`Dh` 是 head dimension。MHA 中 `Hkv = Hq`；GQA/MQA 通过减少 `Hkv` 降低 KV 容量，但它仍然随 `B×S` 线性增长。

以 TinyLlama-1.1B 的典型结构举例：22 层、4 个 KV heads、head dimension 64、FP16，则每个 batch、每个 token 的 KV 为：

```text
2 × 22 × 4 × 64 × 2 Byte = 22,528 Byte ≈ 22 KiB/token
```

若 `B=128、S=2048`，仅 KV Cache 就约为：

```text
22,528 × 128 × 2048 ≈ 5.5 GiB
```

模型更大、层数和 KV heads 更多时会迅速增长到数百 GB 乃至 TB。这也解释了为什么论文重点研究大 batch、长上下文的离线推理，而不是小 batch 在线服务。

传统低成本方案把 KV Cache 放到 CPU DRAM 或 SSD。容量成本下降了，但每个 Decode step 又要通过 PCIe 把大量 KV 搬回 GPU；SSD 方案还可能经过文件系统、host buffer 和多层地址转换。于是瓶颈不是 SSD “能不能存下”，而是“存下以后能不能以足够带宽反复送给 GPU”。

#### 具体 Decode 计算量和传输量

对单层、单个 attention head、单个 Decode token，设历史长度为 `S`、head dimension 为 `Dh`：

```text
score = q K^T：约 2 × S × Dh FLOPs
output = softmax(score) V：约 2 × S × Dh FLOPs
合计：约 4 × S × Dh FLOPs
```

若 K/V 为 FP16，至少需要读：

```text
K + V = 2 × S × Dh × 2 Byte = 4 × S × Dh Byte
```

因此忽略较小的 q、score 和 output 流量后：

```text
算术强度 ≈ (4 × S × Dh FLOPs) / (4 × S × Dh Byte)
         ≈ 1 FLOP/Byte
```

这就是论文所谓 Decode Attention 的“1:1 computation-memory-access ratio”。它意味着性能首先由 KV 供给带宽决定，而不是乘加峰值决定。

进一步看，如果仍把 KV 放在 SSD、Attention 放在 GPU，那么每生成一个 token，理论上需要把当前全部历史 K/V 跨 PCIe 传一次。单步 dense Decode 的 KV 流量近似为：

```text
2 × L × B × S × Hkv × Dh × dtype_bytes
```

它恰好与当前有效 KV Cache 容量同阶；生成下一个 token 时还要再读一遍，并且 `S` 又增加 1。因此单请求生成 `T` 个 token 的累计 KV 读取量对上下文长度呈求和增长，而不是只在 Prefill 后搬一次。

### 2. 论文的推理链条

论文的逻辑可以按四步讲：

#### 第一步：先证明 KV offload 确实带来严重性能损失

论文比较了 DeepSpeed 和 FlexGen 在不同 batch、不同存储层级下的吞吐：

- batch 增大但仍处于同一内存层级时，吞吐通常因并行度提升而提高；
- 一旦 KV 或 kernel 被迫换到 SSD，性能陡降；
- 文中报告 DeepSpeed 在 batch size 32 出现 kernel swapping 到 SSD，性能下降 97.01%。

这说明“更大 batch 能提高 GPU 利用率”并不能抵消 SSD offload 的数据搬运代价。

#### 第二步：再用 Decode 延迟分解定位真正瓶颈

论文对 FlexGen Decode 做 breakdown：

- batch 4、8 时 KV 仍可放入 GPU，主要瓶颈是 Weight Access；
- batch 墠大、KV 被 offload 到内存或 SSD 后，KV Cache Access 占比最高达到 98.94%。

所以论文没有笼统地说“LLM 是 memory-bound”，而是进一步定位到：在长上下文、大 batch、KV offload 场景中，Decode 的主要矛盾是 KV 访问。

#### 第三步：用算术强度/roofline 判断“该迁什么计算”

论文分别分析 Prefill 和 Decode 的算子：

- Prefill 的算子虽然在 GPU 上也可能 memory-bound，但其计算量和并行度仍然太高；CSD 算力远弱于 GPU，把 Prefill 移到 CSD 会让计算本身成为新瓶颈。
- Decode 的 QKV projection、O projection、FFN 主要访问权重，并不直接受益于 flash 上的 KV；其计算需求对 CSD 又偏高。
- Decode 的 Logit/Attend，也就是 Attention 主体，接近 GeMV，计算访存比约 1:1；它算得不重，却必须直接读历史 KV，最适合 data-local execution。

论文举了一个直观上界：若 CSD 有 650 GFLOPS 的计算能力，对于约 1 FLOP/Byte 的 Attention，喂满计算单元理论上需要约 650 GB/s 数据带宽，而普通 SSD 的外部 PCIe 只有约 3–10 GB/s。因此把 KV 经 PCIe 送到外部计算设备不合理；应该让计算靠近 flash 数据。

这里的计算是：

```text
所需带宽 = 峰值计算性能 / 算术强度
         = 650 GFLOP/s / 1 FLOP/Byte
         = 650 GB/s
```

需要注意，这个 650 GB/s 是“想完全喂满 650 GFLOPS CSD engine 的理论数据供给需求”，不是论文声称 SSD 真有 650 GB/s。它的用途是说明：在如此低的算术强度下，Attention 极易受带宽限制；仅靠外部 PCIe 搬 KV，不可能发挥计算单元能力。

#### 第四步：CSD 解决了外部总线问题，但又引出两个新问题

CSD 可以利用多 flash channel 的内部聚合带宽，论文给出的现代 SSD 量级是 tens of GB/s，显著高于单盘外部 PCIe 的约 3–7 GB/s。但 CSD 仍远弱于 GPU，并且 flash 的 page 粒度、通道组织、FTL 映射并不天然适合稀疏 Attention。

因此论文最终不是简单地“在 SSD 内做 dense Attention”，而是形成组合设计：

1. 只 offload Decode Attention 和 KV Cache，Prefill、projection、FFN 留在 GPU；
2. 用 SparF 降低 KV 读取量和 CSD 计算量；
3. 用双阶段加载先筛选重要维度/重要 token，再加载完整 K/V；
4. 用 flash-aware KV layout 和专用 FTL 提升 channel 并行性，减少读放大和写放大；
5. GPU 与 CSD 用 PCIe P2P 传输 Q、新 K/V 和 Attention 输出，host 只做控制面；
6. 多 CSD 时按 attention head 分配，利用 head 之间的独立性扩展。

### 3. 论文的动机实验：具体怎么设计、结果是什么

#### 实验 1：不同 batch 和 KV 存储层级如何影响吞吐（Figure 4）

实验问题：扩大 batch 原本能提高 GPU 利用率，但 KV 超出显存后，offload 的代价会不会抵消收益？

实验设置：

- 模型：OPT-13B；
- GPU：NVIDIA A6000，48 GB VRAM；
- 输入长度：1024 tokens；
- 输出长度：1024 tokens；
- 系统：DeepSpeed 和 FlexGen；
- 自变量：batch size 4、8、16、32、64、96；
- 同时记录 KV 所在层级：GPU、host memory、SSD，以及吞吐和 OOM。

对照逻辑：在同一存储层级内增加 batch，观察并行度收益；当 KV 跨到下一存储层级时，观察 offload 引起的断崖式变化。

实验结果：

- DeepSpeed 在 batch 8 和 32 出现明显性能下降；FlexGen 在 batch 8 和 64 出现明显下降；这些转折与 KV 从 GPU→host memory→SSD 的迁移相对应。
- DeepSpeed 不原生支持 SSD KV offload；batch 32 时出现 kernel swapping，从 host memory 换到 SSD，性能下降 97.01%。
- 在同一个 memory tier 内，增大 batch 仍能提升吞吐；真正导致吞吐突降的是跨层级 offload。
- 更大 batch 下还会出现 OOM，说明廉价 SSD 容量并不会自动转化为可用推理吞吐。

由此得到的结论：问题不是 batch 本身，而是 KV 超过高速内存容量后，跨 PCIe/软件栈的数据迁移吞掉了 batching 的收益。

#### 实验 2：Decode 时间到底花在哪里（Figure 5）

实验问题：Figure 4 只能说明 offload 后变慢，还不能证明是 KV 访问而不是权重读取或计算导致的。

实验设置：

- 系统：FlexGen；
- batch size：4、8、16、32、64；
- 将 Decode latency 分为三类：KV Cache Access、Weight Access、Computation；
- 同时标注 KV 实际位于 GPU、host memory 还是 SSD。

实验结果：

- batch 4、8 时 KV 能放进 GPU，主要瓶颈是 Weight Access；
- batch 增大后 KV 先进入 host memory，再进入 SSD，KV Cache Access 占比随之急剧升高；
- 最严重配置中，KV Cache Access 占 Decode latency 的 98.94%，计算本身只剩极小比例。

由此得到的结论：传统 offload 的核心瓶颈已被定位为 Decode 阶段的 KV 数据移动。优化普通 CUDA kernel 或增加少量计算能力无法解决 98.94% 的访问开销。

#### 实验 3：哪些算子适合放到 CSD（Figure 6）

实验问题：既然 KV 在 CSD，是否应该把整个 Prefill 或 Decode 都迁到 CSD？

实验设置：

- 硬件 roofline：NVIDIA A6000 GPU 与 Zynq7045 FPGA-based CSD；
- 阶段：Prefill 和 Decode；
- 算子：QKV Projection、Logit/Attend、O Projection、FFN；
- 横轴：operational intensity（FLOPs/Byte）；
- 纵轴：可达性能；
- 判断每个算子位于 memory-bound 还是 compute-bound 区域，以及是否直接访问 flash 上的 KV。

实验结果和排除过程：

1. Prefill 的 QKV/O projection 和 FFN 计算强度很高，应留在 GPU。
2. Prefill Attention 即便在 GPU roofline 上偏 memory-bound，其总计算规模和并行需求仍会超过弱算力 CSD，迁移后 CSD compute 成为瓶颈。
3. Decode 的 QKV/O projection 和 FFN 在 GPU 上可能呈 memory-bound，但它们访问的是模型权重，不是 flash KV；同时其 operational intensity 已接近 CSD 的算力边界，迁过去既不能利用 KV locality，又会给 CSD 很大计算压力。
4. 只有 Decode Logit/Attend 是低强度 GeMV，约 1 FLOP/Byte，而且直接消费历史 KV。它同时满足“数据就在 CSD”和“CSD 算得动”两个条件。

由此得到的结论：最合理的拆分不是 Prefill/Decode 二分，也不是全模型 offload，而是 operator-level 的最小边界——只把 Decode Attention 和 KV 放到 CSD，projection、FFN、Prefill 留在 GPU。

### 4. 论文提出方案后的实验：怎么验证设计有效

建议将这部分作为“结果证实前面推理”，不必逐图讲，但要交代对照组和结论。

#### 4.1 总体实验平台与对照组

硬件/模型设置：

- NVIDIA A6000 48 GB；Intel Xeon 5320 2.2 GHz；96 GB DDR4；Samsung 980 Pro 2 TB；GPU 与 CPU 使用 PCIe Gen4×16；
- OPT-13B、OPT-30B、Llama-2-13B；OPT 输入/输出均为 1024，Llama-2 输入/输出均为 2048；
- FP16；数据集包括 ShareGPT、WikiText-2、SQuAD、TriviaQA；
- OpenSSD 实测关键部件后，在 NVMeVirt 中构建软件定义 InstCSD；扩展为 8 channels、每 channel 1.4 GB/s，外部 PCIe 4.0×4 最大 7 GB/s。

七个对照系统：

1. DeepSpeed：host-memory KV offload；
2. FlexGen：host memory + SSD offload；
3. FlexGen-GDS：加入 GPUDirect Storage；
4. FlexGen-SparQ：1/8 compression ratio；
5. Recomp：KV 超容量后用 vLLM recomputation；
6. InstA：不使用 SparF 的 dense InstAttention；
7. InstA-SparF：完整方案，默认 1/8 sparsity/compression ratio。

#### 4.2 Accuracy 实验

设计：在 SQuAD、TriviaQA 等任务上比较 SparF、SparQ、H2O、StreamingLLM 和 Local Attention，并扫描 KV compression ratio。

结果：SparF 的精度接近原始 SparQ，且相对 H2O、StreamingLLM、Local Attention 更稳健；论文观察到压缩比达到 1/8 时精度损失仍可忽略，因此把 1/8 作为后续默认配置。

推论：性能提升不能只靠随意丢 KV；SparF 的价值是用 q 的 top-r 维度近似 attention score，再选择 top-k token，以更有针对性的方式减少 flash 读取。

#### 4.3 端到端吞吐实验

设计：在 1 个和 2 个 SSD/CSD、不同 batch、OPT-13B/OPT-30B/Llama-2-13B 上比较上述七类系统，分别评估 dense InstA 和 sparse InstA-SparF。

主要结果：

- 论文摘要报告，在 A6000 上运行 13B 模型时，相比 SSD-based FlexGen，InstAttention 最高提升 11.1×；
- 多 CSD 扩展实验中，在 batch 256 下从 1 个 CSD 扩到 20 个 CSD，dense 和 sparse throughput 分别提升 8.99× 和 7.29×；这是 CSD 数量扩展比，不是相对 FlexGen 的结果；
- 传统 SSD offload 增加第二块 SSD后的收益有限，因为 host filesystem 和 GPU↔SSD 数据路径仍是瓶颈；InstAttention 可以按独立 attention heads 分配给不同 CSD，因此扩展性更好。

汇报时建议把 11.1×作为论文总体 headline；其他倍数必须连同模型、batch、dense/sparse 基线一起说，避免脱离上下文横向比较。

#### 4.4 延迟分解和 SparF engine 实验

设计：分别分解 dense/sparse inference 中 qkvo projection、Attention、FFN，以及 SparF engine 内 Filter、GeMV、Softmax、top-k 等单元开销。

结果：KV access 仍是主要瓶颈，证明 SparF 应继续围绕减少 flash 访问量优化；dual-step loading 让第一阶段按 page/group 粗筛、第二阶段在 NFC 内细筛，从而避免把一个 page 中无用 entry 全部送入计算引擎。

#### 4.5 虚实一致性与硬件资源实验

设计：在真实 Daisyplus OpenSSD 上测一个 OPT-13B attention head、16 tokens 的 GeMV、Softmax、Filter，再把实测吞吐/延迟用于 NVMeVirt；同时在 Zynq7045 上综合硬件模块。

结果：

- GeMV：真实 12.7 GFLOPS、虚拟 13.3 GFLOPS，准确度 95.50%；
- Softmax：真实 14.2 MFLOPS、虚拟 15.1 MFLOPS，准确度 94.00%；
- Filter：真实 1.85 GB/s、虚拟 1.79 GB/s，准确度 96.80%；
- 完整资源占用达到 Zynq7045 的 80.27% LUT、58.92% FF、46.06% BRAM tile、85.33% DSP，说明设计可实现，但 DSP 已较接近上限。

推论：论文性能不是完全凭理想模型估算；关键 kernel 参数来自真实 OpenSSD，再扩展 flash channels 做现代 SSD 量级的系统评估。

#### 4.6 Endurance 和可靠性实验

设计：估算持续追加 KV 对 SSD P/E cycles 的影响，并加入最坏场景 read retry/retention relaxation 分析。

结果：论文认为 KV 是可丢弃的中间数据，专用 FTL 可简化 GC；最坏 read-retry 场景下性能下降最多约 4.7‰。这用于回应“频繁写 KV 是否会让 SSD 不可用”的工程质疑。

### 5. 第一部分最重要的口头总结

可以这样说：

> InstAttention 不是因为 SSD 容量大就把 LLM 搬进 SSD，而是先通过吞吐和 latency breakdown 证明，KV offload 后真正拖慢 Decode 的是 KV 经过 PCIe 的反复搬运；再通过 roofline 证明，只有 Decode Attention 同时满足“强依赖 KV、低算术强度、CSD 算得动”。所以它选择最小而关键的 offload 边界：KV 和 Decode Attention 在 CSD，其余算子在 GPU。SparF 和 flash-aware FTL 则进一步解决 CSD 算力不足及 flash 访问粒度不匹配的问题。

---

## 第二部分：我们在 CEMU 中的复现进展

### 1. 当前实现的总路径

```text
真实文本/token
  -> Guest GPU：TinyLlama Prefill
  -> 每层 K/V 按 layer/token/batch/head 布局写入 CEMU NVM
  -> Guest GPU：Decode 当前 token 的 Q/K/V
  -> 新 K/V 追加到 NVM
  -> 历史 K/V 按 chunk 从 NVM 搬到 FDM staging
  -> 5 个 FDM range 组成 MRS
  -> CEMU/CSD 调度 CUDA Attention shared library
  -> CSD GPU：dense Decode Attention + 跨 chunk online softmax
  -> 输出返回 Guest GPU
  -> Wo、残差、MLP、LM Head
  -> next token 反馈到下一 Decode step
```

当前用了两块逻辑角色不同的 GPU：VFIO 直通的 Guest GPU 负责 Prefill 和非 Attention 部分；宿主 QEMU/CEMU 进程选择的 CSD GPU 负责 CSD Attention。两者当前没有 P2P，数据还会经过 CPU/NumPy、FDM 和 CUDA mirror。

### 2. 两周内四个递进里程碑

#### 里程碑 A：多 chunk Attention 和 FlexGen KV bridge（8 月 17 日）

- 建立 Python→C++ pybind→CEMU ioctl/NVMe command 的控制链路；
- 建立 KV layout、NVM KV store 和 NVM→FDM staging；
- 定义 5-range Attention ABI：Q、K、V、online-softmax state、output；
- 实现行为一致的 CPU dense Attention 和 CUDA device-pointer Attention；
- 使用 online softmax 合并多个 KV chunk，因此 KV 总量可以超过固定 FDM staging 容量；
- 增加 FlexGen 风格 adapter，把 Q/K/V 张量转换成 CEMU 所需布局。

关键意义：不再要求完整历史 KV 同时进入 FDM；这是长上下文容量可扩展的基础。

#### 里程碑 B：FlexGen GPU Prefill 接到 CEMU Attention（8 月 18 日）

- 重建 Llama/FlexGen 风格 Prefill runtime：embedding、RMSNorm、QKV、RoPE、causal attention、Wo、MLP、LM head；
- Prefill 在 Guest GPU 上计算，每层 KV 写入 CEMU NVM；
- Decode Attention 可切换 PyTorch reference backend 或 CEMU backend；
- 逐层比较 Q/K/V、Attention output、hidden state、logits 和 token。

关键意义：从合成 Q/K/V 算子测试推进到完整模型计算边界，能够确认 offload 前后非 Attention 计算保持一致。

#### 里程碑 C：TinyLlama 真实 checkpoint 端到端 Decode（8 月 25 日前）

- 增加 Hugging Face checkpoint loader，支持真实 config、safetensors/权重命名映射和 layer cache；
- 接入 TinyLlama-1.1B 的 22 层 GQA 结构，而不再局限于随机小模型；
- 增加 tokenizer 和真实文本 prompt 入口；
- 完成真实 Prefill、KV 持久化、多层 Decode、CEMU Attention、后续 Wo/MLP/LM head 和 token 反馈；
- 建立多层次 reference：checkpoint 映射、Prefill、纯 PyTorch full reference、Decode reference、CEMU pipeline 分别校验。

关键意义：已经从“模型结构相似的小测试”推进到真实模型配置和真实权重。

#### 里程碑 D：range-aware CUDA mirror 一致性（8 月 25 日）

- 原先 direct CUDA 路径在执行后固定 D2H，Attention 还对 `mr[3]` 做特殊处理；
- 现在每个 CUDA mirror range 独立记录 host/device 有效性和 dirty 状态，并加锁保护；
- `prepare_device()` 只在 device 数据无效时 H2D；
- `prepare_host()` 只在 host 数据无效时 D2H；
- host write、NVM/FDM copy、fill 和 CUDA output 分别标记对应一侧 dirty；
- 去掉“5 个 range 就把 mr[3] 特判为 Attention state”的耦合。

关键意义：跨 chunk 的 state 可以继续留在 device，最终 output 只在 host 真正读取时回传。这既修复一致性，也为减少重复 H2D/D2H 奠定基础。

### 3. 五个 MRS range 和多 chunk 精确合并

| Range | 内容 | 访问角色 |
|---|---|---|
| `mr[0]` | 当前 Decode query | 输入 |
| `mr[1]` | 当前 chunk 的 K | 输入 |
| `mr[2]` | 当前 chunk 的 V | 输入 |
| `mr[3]` | online softmax 状态 | 跨 chunk 读写 |
| `mr[4]` | 最终 Attention output | 输出 |

`mr[3]` 对每个 batch/query head 保存历史最大 score、指数和、加权 V 和。新 chunk 用 online softmax 公式重标定并合并，所以结果等价于对完整历史 KV 一次性做 dense softmax，而不需要保存全部 attention score。

这部分要强调：我们当前的 chunking 是为容量扩展服务，仍是 dense Attention；它不是论文的 SparF 稀疏双阶段加载。

### 4. 当前已经证明什么

- CEMU 的 program download、activate、MRS、execute 可以被 Python 推理 runtime 驱动；
- Prefill 产生的真实 KV 能按确定布局持久化到 NVM；
- Decode 新 K/V 按绝对 token position 追加，不覆盖历史 KV；
- 历史 KV 可按 chunk 搬入有限的 FDM staging；
- CPU/CUDA CSD Attention 与 PyTorch reference 数值对齐；
- MHA、MQA、GQA 的 query head→KV head 映射已覆盖；
- 真实 TinyLlama checkpoint、真实文本/tokenizer、全层 Prefill 和 Decode 路径已接通；
- range-aware dirty/valid 管理避免无条件同步，并去除了 Attention 专属同步特判。

### 5. 哪些仍不能声称“复现完成”

- 当前 CSD kernel 是正确性优先的 dense FP32 kernel；主要循环没有做 FlashAttention/cuBLAS 级优化。
- 尚未实现论文 SparF：top-r query dimension、top-k token、1/8 稀疏压缩等都没有。
- 尚未实现按 token 与按 embedding dimension 保存两份 K 的 flash-aware layout。
- 尚未实现论文的定制 KV Cache FTL、flash channel 调度、轻量 GC。
- 尚未实现 GPU↔CSD PCIe P2P/GPUDirect；当前数据路径存在 host staging 和 CUDA mirror。
- NVM→FDM chunk 当前串行，没有双缓冲或搬运/计算 overlap。
- 当前 CEMU SSD 参数尚未校准为 InstAttention 的 8×1.4 GB/s 内部带宽及 7 GB/s 外部带宽。
- 尚未形成与 FlexGen、FlexGen-GDS、InstA、InstA-SparF 同口径的端到端 throughput/latency 数据。

因此准确表述应是：

> 已经复现了 InstAttention 的核心软件执行边界和 KV-local Decode Attention 数据流，并在 CEMU 中以真实 TinyLlama 跑通功能与数值验证；尚未复现其 SparF、专用 FTL/P2P 硬件数据路径和论文性能。

---

## 第三部分：SSD 和其他带宽为什么这样设置

### 1. 先区分三种带宽

1. **SSD 内部 flash 聚合带宽**：多个 NAND channel/die 并行读取形成，是 CSD 内计算能直接利用的带宽。
2. **SSD/CSD 外部 PCIe 带宽**：设备与 host/GPU 之间传输的上限，受 PCIe generation、lane 数和协议开销影响。
3. **CSD compute bandwidth**：论文或 CEMU benchmark 中用 GB/s 表示某个 kernel 的等效处理吞吐，即 `处理字节数 / 执行时间`；它不是 NAND 或 PCIe 的物理链路带宽。

这三者不能混用。

### 2. InstAttention 的设置

论文的软件定义 InstCSD 设置为：

- 8 个 flash channel；
- 每 channel 1.4 GB/s；
- 内部聚合带宽 11.2 GB/s；
- 外部 PCIe 4.0×4，最大约 7 GB/s。

原因是它想代表 Samsung 980 Pro 一类现代消费级 NVMe SSD/CSD 的现实量级，而不是任意调一个理想值。8×1.4 得到的 11.2 GB/s 是并行 channel 的内部理论/建模聚合值；7 GB/s 对应 PCIe 4.0×4 在编码和协议开销后的常见顺序带宽上限。

这组参数恰好体现论文的核心机会：内部 11.2 GB/s 大于外部 7 GB/s，计算放在设备内可以绕开外部接口。不过 11.2 GB/s 仍远低于 GPU/host memory bandwidth，所以必须配合 SparF 降低 KV 读取和计算量。

另外，论文 testbed 的 GPU 通过 PCIe Gen4×16 连 CPU，host memory 通路文中按约 32 GB/s 讨论；这解释了某些情况下双 CSD 11.2 GB/s 仍低于 GPU-host 带宽。不同数字描述的是不同链路，不矛盾。

### 3. CEMU 论文的设置

CEMU 论文的主要目标是验证“模拟器能不能忠实复刻真实 CSD”，因此参数跟被验证硬件对齐：

- SmartSSD：外部 3 GB/s、内部 2.5 GB/s；
- host SSD：读 3 GB/s、写 2.3 GB/s；
- CEMU 相应带宽配置与 SmartSSD/CSD2000 保持一致；
- Smart-Infinity case study 中，CEMU-aligned compute bandwidth 约 1.25 GB/s；另设 700 GB/s 的 CEMU-extreme，专门展示 CEMU 能模拟超越 host CPU 的未来 accelerator，而不是声称当前 SmartSSD 真有 700 GB/s。

所以 CEMU 论文中的数字首先服务于“硬件对齐与模拟精度”，InstAttention 的数字服务于“现代 CSD 上 attention data locality 的设计评估”。二者实验目的不同。

### 4. 我们当前仓库的实际设置

当前 `build/cemu_config.json`：

- `pcie_bandwidth` 没有显式填写，CEMU 默认值为 4 GB/s；
- `internal_bandwidth = 0`，代码语义是内部搬运不限速；
- NVM 配置为 8 channel、每 channel 6 LUN，并用 page read/write latency 模型描述 NAND 时延；
- CUDA H2D/D2H 使用真实机器上的 `cudaMemcpy`，当前还没有统一纳入论文式 GPU-CSD P2P 带宽模型。

为什么目前这样设：当前阶段的首要目标是打通功能并验证数值，`internal_bandwidth=0` 可避免一个尚未校准的限速模型干扰调试。但代价是当前端到端时间不能用于宣称复现 InstAttention 的性能。

下一步做性能实验时，建议至少提供三组配置：

| 配置 | 外部 PCIe | 内部 flash | 用途 |
|---|---:|---:|---|
| 当前功能配置 | 默认 4 GB/s | 不限速 | 回归与正确性 |
| CEMU-SmartSSD 对齐 | 3 GB/s | 2.5 GB/s | 验证 CEMU 论文口径 |
| InstAttention 对齐 | 7 GB/s | 11.2 GB/s | 复现 InstAttention 设计点 |

同时应把 CUDA copy、NVM→FDM copy、CSD compute 分开计时，避免把“模拟延迟”和“宿主真实执行时间”混成一个结论。

---

## 建议的 12 页汇报结构

1. **题目与本次结论**：为什么是 KV-local Decode Attention；我们已经跑通什么。
2. **背景**：Prefill vs Decode，KV Cache 容量公式和增长趋势。
3. **传统 offload 的矛盾**：容量便宜了，但 KV 每步跨 PCIe。
4. **动机实验 1/2**：Figure 4 和 Figure 5，讲 97.01% 与 98.94%。
5. **动机实验 3**：Figure 6 roofline，为什么只迁 Decode Attention。
6. **InstAttention 设计**：GPU/CSD 分工、SparF、双阶段加载、FTL、P2P。
7. **我们的复现边界**：已复现核心数据流，不等同于完整论文硬件。
8. **当前端到端架构图**：Guest GPU→NVM→FDM/MRS→CSD GPU→Guest GPU。
9. **两周四个里程碑**：multi-chunk、FlexGen prefill、TinyLlama、range-aware sync。
10. **正确性证据**：reference 层级、GQA/全层/多步 token feedback。
11. **带宽参数解释**：InstAttention、CEMU 论文、当前配置三列对比。
12. **差距与下一步**：性能校准→传输 overlap→P2P→SparF/layout/FTL。

## 最后 40 秒总结稿

> InstAttention 的设计结论来自一条比较完整的证据链：先证明 KV offload 会造成近两个数量级的性能损失，再通过 Decode breakdown 把问题定位到 KV access，最后通过 roofline 说明只有 Decode Attention 既依赖 KV 又适合弱算力 CSD。我们过去两周已经在 CEMU 中复现了这条核心执行边界，并从多 chunk KV staging 推进到真实 TinyLlama 的全层 Prefill 和自回归 Decode，同时用 PyTorch reference 做逐层数值校验，最新又补齐了 range 级 host/device 一致性。现在可以明确说软件路径已经打通，但 SparF、定制 FTL、P2P 和论文口径的性能校准还没有完成。下一步重点不是继续堆功能，而是先建立可信的分阶段性能基线，再逐项加入论文优化并量化每一项贡献。
