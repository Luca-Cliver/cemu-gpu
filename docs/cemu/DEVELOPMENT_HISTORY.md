# CEMU GPU / InstAttention 开发历史与基线

整理日期：2026-09-21。本文合并此前分散的 GPU 进展、会话交接、模型接入、
设备自治设计、分组布局和 FTL 审查记录；它是历史总览，不是每次运行的配置文件。
新增 FTL 工作统一记入 [FTL 工作记录](../../tests/cemu/instattention/FTL_WORKLOG.md)。
现行协议、测试入口见 [InstAttention 文档导航](../../tests/cemu/instattention/README.md)。

## 1. 目标、角色与证据口径

目标是在 CEMU 中复现 InstAttention 的计算分工、KV 存储组织、依赖和资源竞争，
再评价模拟速度与准确性；不是调整 runtime 直到吞吐与论文一致。
GPU 功能模型的加速不是目标 FPGA/CSD 的性能提升。CSD 指 computational storage device，
不是早期笔记中的 Compute Service Daemon。

| 角色 | 实际工作 |
| --- | --- |
| 物理 Host | 运行 QEMU/CEMU、文件或 DRAM 数据后端、CSD CPU/CUDA 功能模型 |
| Guest | 运行应用、Linux CEMU 驱动、FDMFS；架构上对应使用 CSD 的主机 |
| Guest GPU | VFIO 直通，运行 Prefill 和 Decode 的 QKV、Wo、MLP、LM head |
| CSD 计算 GPU | 被 QEMU 使用，加速设备功能模型；不是 Guest GPU，也不自动代表目标硬件 |

- `realtime` 是功能执行测量；`runtime` 是目标计算时间，不能互换。
- Guest profile 的 `*_wall` 是 Guest 应用计时，不是物理 Host 等待时间；冻结会影响两者关系。
- workflow 模型累计时间和 worker 累计时间都不是应用端到端时间；阶段可能重叠。
- 功能数据可用、目标完成时刻到达、资源可用是不同条件，不能用线程数或 slots 数证明重叠。
- `FUNCTIONAL PASS` 证明该测试流程完成，不等于 Hugging Face 数值验证、论文精度或时间校准。
- 不改 VM freezing，不绕过真实 NVM/FDM/FTL，不靠放宽容差或删等待消除行为错误。

## 2. 七月：GPU 功能模型和时间测量

### 2.1 已形成的基础接口

CUDA runtime 通过 `dlopen/dlsym` 加载；`Program.target` 区分 HOST 和 CUDA_DEVPTR。
target 从 Guest helper、ioctl、NVMe download 传至 CEMU。`prep_cuda_lib()` 或 benchmark
的 `-u/--cuda-devptr` 必须正确选择 target，不能只根据 `.so` 文件名判断。

MRS 保存 namespace、起始字节和长度。执行层形成 `mr_addr/mr_len/mr_backend/mr_dev_addr`；
backend 对所需范围懒分配 CUDA mirror，算子消费 device pointers。GPU mirror 不是独立
NVM/FDM namespace，也不是已实现 Guest GPU 与 CSD 的 P2P DMA。

入口：[backend.c](../../hw/femu/backend/backend.c)、[compute.c](../../hw/femu/csd/compute.c)、
[vadd 示例](../../tests/cemu/vadd_example.cpp)、[benchmark](../../tests/cemu/cemu_benchmark.cpp)。

七月阶段曾固定 MR0 输入、MR1 输出，并立即 D2H；后续已加入 range/segment dirty tracking、
按需同步和多 MR 算子支持。旧文中的“dirty tracking 尚未实现”“所有任务只有两个 MR”
不再是当前限制。不要将早期 GPU_MEM backend、direct-device-write、驻留策略的提案当作已实现 API。
完整间接模式角色优化、统一显存后端和 P2P 仍需独立审查，不能由 direct 路径测试推断。

### 2.2 历史测量保留

旧 `kswitch_proxy` 实验为 16 MiB、单 job、单并行 chunk，排除 iter=0：

| 功能模型 | 样本 | Host E2E 均值 | P95 | 估算吞吐 |
| --- | ---: | ---: | ---: | ---: |
| GPU | 13 | 28.59 ms | 30.28 ms | 559.7 MiB/s |
| CPU | 31 | 1783.53 ms | 1787.42 ms | 约 9.0 MiB/s |

旧报告计算模拟执行加速约 62.38 倍，冷启动分别 473.55/1861.18 ms。
这是历史记录，整理时没有复跑；不是 InstAttention 性能或目标设备加速比。

[run_job_e2e.sh](../../tests/cemu/run_job_e2e.sh) 用 Host 时间戳包围 Guest
`JOB_E2E_START/END`；`CEMU_COMPUTE` 来自 QEMU 输出。Guest `tee` 不会自动采集 Host 日志。
旧 `JOB_STAGE_*` 的异步归因曾不可靠，不用它推导精确 Host 分阶段时间。
`--log` 与 `--compute-log` 不能指同一文件，前者可能被覆盖。

早期 KiF 调研、full-decoder offload、GEMM/FFT 等 workload 是候选方向，不是当前复现交付。
“KV 下沉”本身是已有工作的设计；本项目应区分功能模拟加速、行为支持和精度验证的贡献。

## 3. 八月至九月初：Dense、真实模型与公共运行时

先通过 Python/pybind/C++ 桥控制 program、MRS、execute，再实现 NVM KV、FDM staging、
CPU/CUDA Dense Attention 和 token 反馈。五个 Dense MR 分别是 Q、K、V、online-softmax
state、output；分 chunk 时状态累计，不是重复生成完整模型的 token。

旧两文件 KV 布局：

```text
token_payload = batch * kv_heads * head_dim * element_bytes
token_stride = align_up(token_payload, 512)
layer_stride = align_up(max_seq_len * token_stride, 4096)
offset = layer * layer_stride + token * token_stride
```

初始随机模型回归曾覆盖 layers=3、batch=2、prompt=5、Decode=4、hidden=32、
Qheads=8/KVheads=2、FP32、staging=9，共 12 次 Attention；两路生成序列相同。
之后接入真实 TinyLlama、OPT-125M 和 OPT-13B，增加 tokenizer、checkpoint 分片/名称兼容、
FP16、权重缓存与预取。08-20 文档中“只有随机权重、FP32、无 tokenizer”的限制已被后续工作替代。

公共调度只在 `python/runtime_common` 中维护，模型 adapter 注入数学操作：

| 层次 | 内容 |
| --- | --- |
| runtime_common | Prefill、Decode、microbatch、预取、自回归和 PyTorch Attention reference |
| tinyllama_runtime | checkpoint、RMSNorm、RoPE、GQA、SwiGLU |
| opt_runtime | checkpoint、LayerNorm、绝对位置编码、带 bias 的 MHA、ReLU FFN |
| flexgen_adapter / cemu_flexgen | 张量布局、KV、设备和 slot 生命周期 |
| guest / tests/cemu/kernel | pybind/ioctl 桥与设备 CPU/CUDA 算子 |

保留“先纯模型对 HF，再接设备”的验证顺序。局部 Attention 要使用同一 Q/K/V 做比较；
多层 FP16 轨迹误差、最终 logits 和生成 token 分开报告，不能把 token 相同等同于逐层精确相同。
旧初版 CUDA 单线程循环的描述不能代表后来并行化的算子。

应用层增加 Prefill KV writer、Load/Compute/Store 分离、独立 FDM slots 和 microbatch 流水。
Guest GPU 转 CPU/NumPy 是当前功能搬运路径，不是 P2P；同步依赖仍必须存在。
追踪、数值验证、benchmark 分开，避免逐层张量保存与打印污染性能测试。

入口：[Makefile](../../tests/cemu/Makefile)、
[OPT 流程](../../tests/cemu/instattention/python/test_opt_cemu_pipeline.py)、
[公共运行时](../../tests/cemu/instattention/python/runtime_common)。
不要复制旧历史中的 8 GiB Guest RAM 启动参数运行 OPT-13B。

## 4. 九月中旬：从 SparF 文件路径到设备自治

### 4.1 旧路径保留的意义

Dense 之外加入 SparF：Top-r query channel → channel-K → 粗分数/Top-k → token-K/V → 精确 Attention。
旧版用 token-K、channel-K、token-V 三文件；有单 execute、Guest staged 接力、有界 buffer 和
spans/compact 设备写回等对照路径。compact 减少描述构造，不自动改变实际散写/RMW。
旧写回微例的 384 B payload、18432 B read、18432 B write 是扇区搬运计数，不能称为 NAND WAF。

设备自治 v3 让一次逻辑 attend 在设备内推进；先发 V 读，再算精确 QK，PV 等待两者。
内部阶段不提前完成 Guest 父请求，program/MRS/slot 引用必须保持到正确的释放点。
旧 9 请求 trace 的依赖检查 PASS 只覆盖该轮，不是当前所有配置的并发证明。

截至保存基线，已存在多 slot 提交和 `sparf_engines` worker，不能沿用旧“仅单父任务、nr_cu=1”
作为所有模式的限制。父请求与 worker/CU 的绑定也不能直接等同论文动态双 Attention Kernel 调度。

### 4.2 分组页目录、尾部与自治 Attention

- G1：明确页布局和参考编码，区分 token/channel 组大小。
- G2：设备目录以 `(session, stream, kind, group)` 管理池内页，整组 PUT、稀疏 READ 去重并恢复顺序。
- G3：APPEND、分布局 committed 边界、有界环形尾部、后台写、混合读取及终结 DRAIN。
- G4：grouped 目录/尾部接入自治 SparF，再接公共 OPT runner；不是新建另一份 Attention 数学实现。

当前 grouped 基线仍由 Guest 分配独占 NVM 文件 pool、初始化并注册 extent；设备拥有后续目录与写回。
它不是每次由 Guest pwrite KV，也还不是论文无文件系统的 KV 专用 FTL。
旧文的 65536 页/4096 streams 上限已经扩展；实际限制以 ABI 和容量预检为准。

## 5. 分组布局与所有权契约

设 P=页字节，D=head_dim，E=元素字节，C=每页 channel 数：

```text
Gt = P / (D * E)       token-K/V 组的 token 数
Gc = P / (C * E)       channel-K 组的 token 数
Nc = D / C            一个 channel-K token 窗口的页数
token offset = ((token % Gt) * D + channel) * E
channel offset = ((channel % C) * Gc + token % Gc) * E
```

页编码要求整除和边界校验。channel 页的逻辑编号 `group * Nc + channel_group` 不是 PPA。
OPT-13B FP16、D128、P4096 时 Gt=16；C=2/4/8 对应 Gc=1024/512/256、Nc=64/32/16。
这是本实现与论文分组语义相容的页内契约，不是声称还原 FPGA 私有字节格式。

每 stream 有 accepted 和 token-K/channel-K/token-V 三条 committed 前缀。
例如 C4、accepted=20：token-K/V 可提交到16，channel-K 仍为0；不能因 token-K 已落盘释放原始 K。
发布要求真实 IO 成功且目标期限到达；发布前尾部仍有效，混合读取不能重复/遗漏 token 或读取错误版本。
K 的复用需要两份 K 布局完成及读者释放，V 独立判断。失败 poison 会话，不静默重试部分 APPEND。
终结 DRAIN 写尾组并 sealed；普通 flush 不封存会话。CLOSE 等待引用退出，但当前不等于 FTL reclaim。

容量不能只算有效数据：

```text
streams = layers * batch * kv_heads
三布局有效数据 = 3 * streams * tokens * D * E
当前环形尾部基数 = streams * (2*Gc + 2*Gt) * D * E
```

还要增加批写 scratch、command/status、所有 Attention MR、文件对齐、目录及在途资源。
OPT-13B B4、T2048、C8 的三布局有效数据为9.375 GiB，尾部基数850 MiB；不是6 GiB pool能容纳的配置。
当前 scratch 随 write_batch_pages 变化，不沿用初版“固定一组64 KiB”的总预算。
FDMFS statfs=0 表示容量未知，应提供扣除已有分配后的可用预算，而非绕过实际容量。

## 6. 上层审计与 FTL 修复

09-17 上层审计补齐：连续 accepted 前缀、Prefill 不重复、提交时复制 Q/mean、同 backend
追加与 Attention 互斥、失败 Future 不被覆盖、部分 APPEND 失败后禁止重试、关闭后不可重新打开。
旧三文件 current KV 在写成功后更新；共享 transport 的 MR0→execute→status 是完整事务。
最终 owner 等待全部 backend 后全会话 drain，不能用单 backend drain 替代全局 barrier。
`min(QK,V)` 统计改为 overlap upper bound，实际交叠需观察区间交集。

09-21 地址路径审查发现并修复：

| 问题 | 修复与边界 |
| --- | --- |
| 65535-sector 分段切穿 NAND 页 | 页对齐起点、4 KiB 页时上限65528 sectors；原32 MiB传输曾让8192唯一页产生8193页操作 |
| 每页 tag 重复扫描全部 pool | plan 排序保留 ordinal，批内排序 O(n log n)、刷新 O(n)，不为整个池再建索引 |
| trace 混淆普通读/KV读和会话 | 加 controller/nsid/cookie/plane、GLib once 初始化；分析按身份区分，兼容旧 trace但警告缺身份 |

修复不改变 freezing、Guest ABI、算子、NAND 参数或目标 runtime。
旧 trace 的1313读中只有553次 KV读：等待合计60133 ns；760次未标记读88940 ns。
全部等待149073 ns不能称为KV等待，更不能称为应用延迟。
物理页复用不自动是GC错误，零复用也不是GC正确性的证明。

当前读写仍经 pool ordinal→extent/LBA→FEMU maptbl/rmap。kv_group 按三种 kind 使用写指针，
还不是 task-aware 放置。真实后端按LBA偏移取数据，PPA描述NAND状态/时间；新模式必须保持
真实字节与物理映射一致。当前 DSM 不提供任务回收，CLOSE 也不擦除该任务的Flash块。

## 7. 09-21 保存基线与验证证据

- CEMU：`c5f0710988bb18a52817289ab4eede0d7d2b38d0`。
- linux-cemu：`86d7dda112cd22ce0d502295d02e80e0814c872c`，FDM memory-range 校验和文件引用修复。
- 用户已反馈推送完成；整理时未重新编译内核，不能保证现有Guest内核来自该提交。
- 历史 Host discovery 从270/272项增至修复后280项通过，QEMU构建通过；保留已有indirect未使用变量警告。
- Host discovery 不执行 Guest 脚本 main，不能替代GPU/NVMe集成验证。
- 用户提供 Guest group-buffer PASS、grouped CUDA Attention 24例PASS，最大绝对误差9.766e-04。
- 早期group-device记录：132次写、786432 B写、843776 B读、PUT旧页读0；不同轮次不能拼成同一实验。
- 后续buffer记录：2 streams×531 tokens，APPEND47次/94 streams、写批34次、最大36页、写950272 B，
  FDM tail+scratch1327104 B；覆盖15/16、255/256、ring复用、混合读和终结尾组。
- 新trace一轮为1352写、0读，kind0/1/2分别404/544/404；不能验证该轮读取等待。

OPT-13B grouped/autonomous functional 的一轮保存结果：B4、microbatch4、P1024、Decode16，
40层×16=640次逻辑Attention，最终drain完成。没有额外microbatch因子，因为该轮只有一个microbatch。

| 项目 | 该轮记录 |
| --- | ---: |
| 权重预加载（单列） | 261.739062 s |
| Guest Prefill | 25.497148 s |
| Guest Decode | 55.970217 s |
| 最终 drain | 1.138732 s |
| E2E including drain | 82.606456 s |
| CSD workflow 模型累计 | 18.150203 s |
| 写批 / 最大页数 | 6339 / 256 |
| APPEND 命令 | 5760 |
| appended_streams / appended_tokens | 512000 / 6656000 |

最后一行按KV stream统计，不是生成token数量。上述一次开启诊断的运行不证明提速、双引擎重叠、
完整HF对齐、GC安全或论文性能复现；本轮文档整理没有重新执行这些测试。

## 8. 剩余差异和运行注意事项

- 新的KV专用地址、物理放置、task/block归属和reclaim尚未实现；见FTL工作记录。
- SparF选择、tie/current-token、补偿需独立数值验证，不能要求稀疏结果无条件等于dense。
- CUDA Softmax shape ratio乘论文锚点是跨硬件外推，不是FPGA校准；不能留空回退legacy而不记录。
- GPU microbatch决定ratio的vectors，prompt/Decode决定长度覆盖；空路径和未测shape应在预加载前发现。
- Guest仍维护V mean，有CPU buffer中转；目标engine仲裁、Flash配置和容量计费仍需逐项对照。
- 公共FlexGen上层有Load/Compute/Store交错和sync；模块j不总等于Transformer层。
- 上游最后生成步可能不再保存无后续用途的KV；本测试强制最终drain，必须单列统计边界。
- 论文数据集、完整生成长度、多CSD head分片、低日志重复测量尚不能由当前短流程PASS代替。

Host/Guest桥需分别编译（历史环境Python3.10/3.13），不要覆盖Guest扩展为Host二进制。
启动时确认模型共享挂载、NVM后端、Guest RAM、FDM可用量；未配置文件后端时可能仍是Host DRAM。
GPU按实际PCI地址和占用确认，不沿用历史编号；不要自动重启宿主机、解绑别人的GPU或重载驱动。
私有575用户库曾解决575内核/580库不匹配，它是环境补救，不是FTL功能。

低日志通过Host的 `CEMU_COMPUTE_LOG=progress`、`CEMU_COMPUTE_LOG_INTERVAL` 或
`CEMU_COMPUTE_LOG=0` 控制，不靠Guest环境改变已运行的QEMU。
保存命令、配置、ratio、Host/Guest日志与trace到持久目录；不把/tmp临时扩展和日志当唯一证据。

## 9. 文档合并来源与原文恢复

以下10份受Git跟踪的旧文已合并；不是删除源码、论文、实验日志或个人未跟踪报告。
历史命令、过期行号和探索性提案不再逐段复制；完整原文仍可从上面的CEMU基线取出。

| 原路径（相对CEMU根目录） | 内容归入本文 |
| --- | --- |
| docs/cemu-gpu-progress.md | 第2节：GPU基础、提案边界 |
| docs/cemu-gpu-current-status.md | 第2节：target、MRS、mirror及已过时限制 |
| docs/session-handoff-2026-07-14.md | 第1/2/8节：时间口径、旧测量、研究与环境边界 |
| tests/cemu/instattention/progress-summary-2026-08-20.md | 第3节：Dense、模型和公共运行时 |
| tests/cemu/instattention/INSTATTENTION_BEHAVIOR_ALIGNMENT_PLAN.md | 第4/5/8节：自治、资源、生命周期 |
| tests/cemu/instattention/SPARF_GROUPED_KV_DESIGN.md | 第5节：双布局、容量和发布契约 |
| tests/cemu/instattention/INSTATTENTION_HANDOFF_2026-09-15.md | 第4/5节：G1～G4演进与错误边界 |
| tests/cemu/instattention/UPPER_LAYER_AUDIT_2026-09-17.md | 第6节：上层审计 |
| tests/cemu/instattention/FTL_ADDRESS_AUDIT_2026-09-21.md | 第6节及FTL工作记录：地址审查、修复 |
| tests/cemu/instattention/BASELINE_2026-09-21.md | 第7节：基线证据 |

只读恢复示例（不改工作树）：

```bash
git show c5f0710988bb18a52817289ab4eede0d7d2b38d0:tests/cemu/instattention/FTL_ADDRESS_AUDIT_2026-09-21.md
```

保留各功能协议说明和grouped runbook，因为它们仍提供可用接口/测试细节；其阶段性描述
只适用于对应模式，跨阶段当前结论以本文基线和FTL工作记录为入口。未跟踪的调研报告不在清理范围。
