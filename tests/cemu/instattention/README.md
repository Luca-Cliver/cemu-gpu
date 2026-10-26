# InstAttention / CEMU 文档导航

当前目标是复现论文行为再评估模拟精度，不以吞吐吻合或一次functional PASS代替行为验收。

当前 native 串行路径的函数入口、Prefill/Decode 数据流、完成条件和计时边界见
[串行执行与数据流](NATIVE_SERIAL_DATAFLOW.md)，用于后续分阶段埋点和论文行为对照。
调用、系统调用、冻结区间及等待边界的图示见
[串行时序图与时间标记](NATIVE_SERIAL_SEQUENCE.md)（含 Prefill、单层 Decode、父 Execute 内部）。
物理硬件阶段、作者 NVMeVirt 模拟的已知边界、当前 CEMU 实现的逐项对应见
[物理阶段对应表](PHYSICAL_STAGE_MAPPING.md)。

## FDM KV buffer 收包、聚合、写回与 Attention（2026-10-25）

当前实现 **创建两个 FDM 文件/MRS → 加载并激活一次 Host 聚合模块 → INIT/BIND →
普通 FDM WRITE 分片接收 → 整包 READY → 输入目标时间到达 → 非冻结聚合 → WRITE 完成**，
同时由后台 writer 将成块页提交 KV FTL，目标时间到达后回收页槽；终止包 DRAIN 等待全部写回。
`FDM_KVbuffer.open()` 在 `create_memory_ranges()` 成功后加载 `fdm_kv_aggregate.so`，
随后调用 `CemuClient.bind_fdm_kv_buffer()`；每个包只写 FDM，不另发聚合 EXEC。
模块默认 Host 路径为 `./build/fdm_kv_aggregate.so`（相对于 `tests/cemu`），可通过
`aggregate_program` 指定。下载/激活复用现有 CEMU 接口，不修改 Guest Linux 驱动。
此管理操作固定使用 `NVME_IOCTL_ADMIN_CMD`，不随 `CEMU_IO_BACKEND` 切换。

复用 compute namespace 的 MRS 管理 opcode `0x21`，新增 selector `2`：
`cdw10 = 2 | (rsid << 16)`，`cdw11 = 80`，`cdw12 = 聚合 program ID`，
`cdw13..15` 为零。绑定只接受已激活的 Host shared-library direct program，
其 runtime/runtime_scale 必须为零，不能把 Attention 计算预算套到此软件处理阶段。
80 字节小端描述符为 `FDMKV_BindParams_t`：magic/version、目标 NVM NSID、
写回批量上限、session/generation、输入槽长度，以及模型几何参数。
Python 的 `nvm_namespace_id` 默认 1、`input_bytes` 默认 128 KiB；后者是
MR1 输入槽容量，不是当前包的实际长度，也不等于 KV 追加已完成。
控制器身份、MR 地址和 NAND channel/block 几何由设备查表获取，不接受 Guest 指针。

- MR0：直接存 `FDMKV_Buffer_t`、收包 bitmap、head/page/domain/writeback 管理数组。
- MR1：起始 `input_bytes` 为输入槽；按页对齐后为暂存页池。初始化不修改 MR1。
- `NvmNamespace.fdm_kv_buffer` 指向 MR0 内的 buffer；
  `MemoryNamespace.fdm_kv_binding` 记录同控制器的目标 NVM、MRS、输入槽偏移和长度。
  `FDMKV_Binding_t` 保存设备侧绑定及 writer 运行态，没有另分配 KV 数据页池。
- 当前要求两个 MR 在同一 FDM namespace；一个 NVM 和一个 FDM namespace 各至多绑定一个 buffer。
  MR 长度受现有 uint32 描述符限制，单 MR 必须小于 4 GiB。
  预算、页大小、重叠范围及 namespace 类型不合法时拒绝绑定，不修改初始化区域。
- 绑定持有 MRS 和 program 的使用引用，普通 EXEC 不可复用该 MRS；绑定期间不能
  停用、覆盖加载或卸载聚合 program。删除 MRS 自动解除绑定。
  控制器 reset/退出在后端释放前清除绑定；reset 后须重新 BIND。
  接收、聚合、后台队列/完成回调和普通 FDM READ 共用 FDM namespace 的互斥锁。
  正常删除在 pending WRITE、在途块、lease，或未失败且仍有暂存页/未收齐包时返回 BUSY；
  先 DRAIN，再删除，不隐式丢弃 accepted 数据。reset/退出属于取消而非排空：先停 poller，
  在 `to_poller/pq` 释放前停止并 join writer、取消内部完成事件，最后解除绑定。

当前自动聚合使用 `ftl_mode=instattention`，即 KV-only NVM；FDM 的普通读写仍保留。
绑定期间该 NVM 拒绝 native KV 数据命令（保留 INFO），避免同时启用另一条缓冲路径；
已有 native tail 时拒绝 FDM BIND。原 native 路径在没有 FDM 绑定时不变。
这是 **CEMU 软件栈用于 LLM 推理的 baseline**，不是要求实现普通 LBA 与 KV 双 FTL 共存。
共存延后。FDM 暂存/Flash 的 Attention 读取适配、Python Prefill/Decode 封包和终止
DRAIN 已接入；小模型 CPU/QEMU qtest 推理已通过，但尚未在 Guest Linux/FDMFS 下
验证真实 OPT-13B 全流程，不据此声称性能或论文精度已验证。
`FemuCtrl.native_kv` 由 `ftl_mode=instattention` 决定，表示控制器采用 KV 专用 FTL，
不表示数据必须来自 `NativeKvBuffer`。因此这条 FDM 路径仍进入 native poller 分支；
命令根据 namespace 分发，FDM WRITE 进入 `memory_write()`，不是 native APPEND。
为保留早期接收回归，`ftl_mode=femu` 仍支持 `cdw12=0` 的只接收诊断绑定，
但它不能自动聚合，不能作为新路径的正常运行配置。

`memory_write()` 在 DMA 前检查目标范围与收包状态。命中输入槽后，仍用原有
`dma_write_prp()` 直接写 MR1，不另分配整包中转区；DMA 成功才更新 MR0 的接收 bitmap
（每 4 字节一位）。支持同一个包的分片乱序到达、包头分片和空洞检查；一个槽只允许一个包
在途，禁止混入另一包、重复/重叠写入，不能用覆盖旧片段的方式重试。

包头区占 512 字节，其中 `FDMKV_AppendHeader_t` 为 88 字节小端结构，其余字节必须为零。
包头携带 session/generation/request_id、batch/layer、head/token 范围及 wire/payload 长度；
payload 长度必须等于 `2 * head_count * token_count * row_bytes`，后接补零至 512 字节的尾部。
输入槽容量与实际 `wire_bytes` 分别校验；包头没有收齐前不解析坐标。这里只验证包格式和
坐标范围，head 的连续追加前缀、页池容量等检查留给聚合步骤。

状态为 `EMPTY → RECEIVING → READY → AGGREGATING → AGGREGATED → EMPTY`。
普通分片 WRITE 仍只确认本片接收；使整包收齐的那条 WRITE 记录为 pending，
保留原请求和字节数，并把完成目标提高到整个包各片目标的最大值。
原 `to_poller/pq` 到期且 CQ 有空间后，调用绑定模块聚合，再返回该 WRITE 的 CQE。
没有新增计算 EXEC、CU 排队、冻结区间或合成 runtime；当前同步处理占用 poller 的
实际开销属于本 baseline 的设备软件实现开销，不能直接称为真实 CSD 的聚合延迟。
普通文件写入可能被拆成多条命令，本协议按覆盖位图判断完整包，而非按 syscall 次数判断。

无效完整包头/尾部、DMA 失败或聚合失败转为 `FAILED`。聚合前缀错误返回 `0x4002`；
页池不足且存在整块待写/在途写回时，保留 READY 包及原 WRITE，在完成队列中延后检查，
不向 Guest 返回失败，也不要求 Guest 重发包。每轮至多尝试一次，下一次检查目标为当前虚拟
时间加 10us；这是本实现的重试策略，不是校准后的硬件延迟。没有后台进展可能时才返回
`NVME_CAP_EXCEEDED | NVME_DNR`（0x4081）。FTL 注册/写入失败使 buffer 失败，不能重试
整个已部分写出的块；后续 WRITE/DRAIN 不得宣称成功。
只有旧的 `cdw12=0` 只接收诊断绑定，收齐时仍返回 0x4001 并保留 READY。

收包记录各次 WRITE 的最大虚拟完成时刻 `max_write_ready_ns`。
READY 表示数据功能上已收齐，不保证模拟时间已经走到该时刻；聚合在完成队列到期后执行。
本阶段不修改 freezing、原有 FDM 传输计时或 NAND 模型。

绑定期间普通 WRITE 不得修改 MR0/暂存页池，FILL 不得覆盖任一保留 MR；其他区域仍走
普通路径。COPY/P2P、其他 MRS 别名或算子写入尚未纳入收包协议，不得用于改写这两个 MR，
这里不是通用内存访问隔离。MR0 内指针仅供设备进程使用，不是 Guest 可解引用的地址。

Host 上编译 QEMU 后运行（不启动 Guest OS、不访问 GPU/VFIO/已有镜像）：

```bash
make -C build -j4
make -C tests/cemu/kernel ../build/fdm_kv_aggregate.so
make -C tests/cemu instattention-fdm-buffer-write-qtest PYTHON=/usr/bin/python3
```

此目标兼容原 `instattention-fdm-buffer-init-qtest` 和 `instattention-fdm-buffer-receive-qtest`。
测试覆盖 Python open/close/错误回滚、
小规模真实 QEMU 管理及 FDM WRITE 命令、NSID/绑定拒绝条件、MR0 初始化/MR1 保持不变、
乱序/分片包头/空洞/重复/越界/无效头/DMA 失败/READY 保护、解绑/reset 后重绑，
以及旧普通 NVM/FDM 读写、新 KV-only FDM 自动聚合、模块引用保护、输入时间门槛、
pending reset、多包槽复用、token/channel 边界、三种页布局逐字节校验和聚合失败传播。
写回测试还覆盖 1059-token 长序列超过页池容量后的复用、跨 channel 多块在途、deadline 前
保留 MR1/hash、NO_SPACE 保留原 WRITE 且只发布一次 accepted、分片终止 DRAIN、
reset 取消未到期写回、新 session 重绑，以及解绑后按 KV 坐标回读 Flash 的三种布局。
不覆盖 Guest Linux/FDMFS/ioctl 驱动端到端，更不等价于完整推理测试。

### FDM 推理接线与验证边界

`FDM_KVbuffer.append_batch()` 复用 `SparfGroupedAttentionBackend`，把
`[tokens, heads, dim]` 拆为不跨 batch/layer 的连续 head 包，每包 K 在前、V 在后，
每份按 `[head, token, dim]` 排列。每包最多 `2 * token_group_tokens` 个 token，
同时受输入槽容量约束，包头和填充也计入容量。一个输入槽仍只允许一个包在途。
`CemuClient.write_range()` 经原文件写接口（`pwrite` 或选定的 io_uring WRITE）写 MR1，
返回后才更新 Guest accepted 前缀；不存在每包另发聚合 EXEC 的操作。
写失败使客户端会话不可继续使用，不能盲目重试可能已部分写入的包。

Attention EXEC 新增显式 `CEMU_SPARF_WORKFLOW_FDM` 标志，元数据为 112 字节公共头加
24 字节 session/generation/batch_first/NVM NSID。native 的原 128 字节格式保持不变。
设备按当前控制器的绑定表找 FDM buffer，校验身份、几何和每个 head 的 accepted 前缀，
持有读 lease，完成或取消后释放；不是把 `FDMKV_Buffer_t` 强转为 native buffer。

三阶段读取统一进入 `fdm_kv_attention_read()`：

```text
channel/token 索引 → KV 坐标 → 逻辑页键
  ├─ MR0 哈希表命中：直接从 MR1 暂存页取数据，包括 NAND 尚未完成的在途页
  └─ 未命中：按页去重 → bb_native_kv_batch_observed → KV FTL/NAND/后端
  → 拼入 Attention slot 的 FDM MR → 原 SparF 读取 deadline 等待 → 原计算阶段
```

同一阶段中，源选择和暂存复制在 FDM 锁下完成；Flash gather 使用有界临时描述符/读 scratch，
不复制第二份持久 KV 页池。Flash 完成目标仍由现有 NAND 模型计算，并由原状态机等待；
不更改 freezing。诊断保留实际 Flash 页数、去重前字节、有效字节、LUN 和 full-scan 参照。
同 head 的 APPEND 遇到读 lease 时保留原 WRITE 等待，不覆盖页或毒化 buffer；
DRAIN 等待已有 lease，期间不接纳新的读 lease。解绑忙状态不能被误报为成功。

`release_memory_ranges()` 使用标准 NVMe admin passthrough 检查删除结果，因为旧
Guest CEMU DELETE_MRS 包装没有向用户返回设备拒绝状态。未修改 Guest 内核代码；
新路径必须重新编译 `_cemu_client`。关闭先 DRAIN，再检查删除 MRS，最后卸载聚合 program。
失败后如果设备仍返回 BUSY，保留错误，不宣称清理成功；可能需要 reset 后开始新会话。

Host 回归（真实 QEMU 设备，CPU 算子，无 Guest Linux/GPU）：

```bash
make -C build -j4
make -C tests/cemu instattention-fdm-csd-qtest PYTHON=/usr/bin/python3
```

测试覆盖 FP16 的 15/16、253/256/259 边界、GQA、压缩比 1/8、混合暂存/Flash 读取、
终止排空后的纯 Flash 读取、身份拒绝、lease 下 APPEND 等待和 reset；另以合成权重的
2 层小 OPT，batch=2、prompt=127、decode=3，核对 token、logits、hidden states 与
PyTorch reference。这里不是 OPT-13B checkpoint 或论文数据集测试。

完整 Guest pipeline 已支持 `FDM_INTERNALBUF=1`（CLI `--fdm-internalbuf`），
与 `SPARF_NATIVE`/`SPARF_GROUPED` 互斥，必须同时使用 functional/autonomous SparF。
需要显式提供 `FDM_BUFFER_META_BYTES`、`FDM_BUFFER_DATA_BYTES`；聚合模块 Host 路径
可用 `FDM_AGGREGATE_PROGRAM` 指定。先在 Host 编译 `fdm_kv_aggregate.so`，并确认设备
FDM 容量足够容纳 MR0、MR1 和所有 Attention slots。不要沿用 native 路径的独立 DRAM
budget 当作 FDM 可用容量：现有 b4-p1024-d16 配置的 2 GiB FDM 不足以直接承载此页池。
先跑小规模 Guest 功能测试，再另配容量跑大形状；当前不自动增大设备容量，也不开放 NVM LBA。

### FDM OPT-13B 小规模 Guest 验收

`run-native-opt-test.sh` 支持用 `CEMU_NATIVE_CONFIG` 显式选择设备 JSON；不设置时
仍使用原 model/profile 对应配置。该文件会复制到本次会话的 `device-config.json`。
`experiments/configs/fdm_opt13b_smoke.json` 只把原 smoke 的 FDM 从 64 MiB 扩为
256 MiB，NVM 保持 512 MiB，NAND 几何、延迟和计算单元配置不变。

此配置先用于 OPT-13B、batch=GPU batch=1、prompt=32、decode=4、FP16、
channels_per_page=8、write_batch_pages=32。Guest 分配 MR0=4194304 字节（4 MiB）、
MR1=167772160 字节（160 MiB），输入槽仍为 128 KiB。当前初始化需要 32600 个暂存页，
MR1 实际占用 133660672 字节，MR0 管理区实际占用 2214920 字节；另留 Attention slots。
此形状完整三布局共 35200 页（144179200 字节），低于启动器 smoke 的 65536 页上限。
该配置不是 B4/P1024/D16 性能配置，不能直接沿用到大形状。

Host 启动选择 `CEMU_NATIVE_PROFILE=smoke`、`CEMU_NATIVE_CSD_TARGET=cuda`、
`CEMU_NATIVE_CONFIG=/home/lihan/cemu/CEMU/tests/cemu/instattention/experiments/configs/fdm_opt13b_smoke.json`，
并给 `run-native-opt-test.sh` 传 `OPT-13B`，使 Guest RAM 为 64 GiB。
Guest 直接调用 pipeline 的 `--fdm-internalbuf --sparf-autonomous --functional`，
不要调用固定选择 `--sparf-native` 的旧 Guest helper，不格式化或挂载 NVM。
功能验收暂不指定 runtime/ratio，不能以耗时作为校准性能。

### FDM OPT-13B B4/P1024/D16 性能采集

使用 `experiments/configs/fdm_opt13b_b4_p1024_d16.json`。它与原
`femu-scripts/cemu_config_opt13b_sparf_engines2.json` 仅有 FDM 容量差异：2 GiB 改为
4 GiB；96 GiB NVM、8 channel / 6 LUN、256 页/block、NAND 延迟、2 CU / 2 worker /
2 SparF engines 均保持不变。Host 设置 `CEMU_NATIVE_PROFILE=b4-p1024-d16`、
`CEMU_NATIVE_CONFIG` 指向此 JSON、`CEMU_DEVICE_INIT_ORDER=femu-first`，仍启动 OPT-13B。
初始化顺序选项只是本机已成功启动的绕行方式，不代表 CUDA 故障根因已解决。

Guest 使用 `--fdm-internalbuf --sparf-autonomous --functional`，batch=GPU batch=4、
prompt=1024、decode=16、Attention slots=2、top-r=16、compression=8、
channels-per-page=8、write-batch-pages=256；MR0=`67108864`（64 MiB），
MR1=`4026531840`（3840 MiB），输入槽仍为 128 KiB。单个 MR 长度字段为 uint32，
不能把 MR1 配成 4 GiB 或更大。初始化的精确需求为：

- 6400 个 head、972800 个暂存页，MR0 使用 63370896 字节，MR1 使用 3984719872 字节。
- 两个 Attention slots 合计预留 32628736 字节；加上 MR0/MR1 分配共 4126269440 字节，
  小于 4 GiB FDM，要求文件系统没有遗留分配。
- 三布局共 1344000 页（5505024000 字节），启动 profile 的 2097152 页上限足够；
  不能沿用 smoke 的 65536 页上限。

运行前必须加载 `experiments/configs/opt13b_sparf_1csd.json` 的 runtime model，并用
`SoftmaxRatioTable.validate_anchor()` / `require_sparf(4, 40, 1024, 16, 8)` 检查 ratio。
与旧结果对照应复用相同 ratio，不自动挑选文件或静默退回线性估计；每轮复制 runtime 和
ratio JSON 到日志目录。缺少 ratio 时，可在空闲 Guest GPU 单独测量：

```bash
python3 instattention/experiments/bench_softmax_ratio.py \
  --config instattention/experiments/configs/opt13b_sparf_1csd.json \
  --device cuda:0 --batch-sizes 4 --prompt-length 1024 --decode-steps 16 \
  --compression-ratio 8 --output /root/cemu-logs/opt13b-mb4-p1024-d16-softmax.json
```

测量脚本拒绝覆盖已有文件。新 ratio 的结果不能与旧 ratio 运行混作单变量比较；CUDA shape
ratio 仍不是经过 FPGA 校准的计时原型。Guest 固定 HPET，数据传输先固定
`CEMU_IO_BACKEND=ioctl`（FDM 写为 pwrite），避免同时改变传输方式；与 native 对照时也需
固定同一 backend。Host 开 `CEMU_NATIVE_TIMELINE=1` 保留时间线。

预期 640 次 Attention、6656000 个 head-token、33280 个聚合数据包；最后首个 head 的
accepted/token-K/channel-K/token-V 均为 1040，sealed=1、failed=0、暂存页/在途块/
pending_ready_ns 为零。比较 Prefill、Decode、最终排空与端到端时间，权重加载单列；
不要把 worker 累计时间相加，也不要将此短 Decode 工作负载视为论文图 14 的等价实验。
CPU QEMU 容量回归覆盖实际大 MR 绑定、高偏移访问保护、末端 batch/layer/head 的小包
聚合和 drain，不替代完整 Guest CUDA 性能运行。
预期 160 次 Attention 请求，最终三个 committed 前缀均为 36、sealed=1、failed=0，
暂存页、在途块及 pending_ready_ns 归零；数值正确性仍须结合独立参考测试。

容量和真实设备 INIT/BIND（不加载模型、不启动 Guest）的回归入口：
`python3 -m unittest test_fdm_kv_binding.FdmBindingDeviceTest.test_opt13b_smoke_capacity_and_binding`，
需将 `instattention/python` 加入 `PYTHONPATH`。它还验证 64 MiB 数据 MR 被拒绝，以及
配置只改变 FDM 容量，不以修改 NAND 模型来让测试通过。

### FDM 原地聚合模块与非冻结调用

这条路径是保留 CEMU 文件/MRS 软件栈的 **baseline**：KV 仍通过 FDM 写入接口进入输入槽，
管理结构和暂存页均使用已分配的 FDM，不以绕开 `pwrite`、删除传输成本或改变 freezing 来提速。
native 路径保留，后续用于行为与开销对照。

新增 `tests/cemu/kernel/fdm_kv_aggregate.c`，编译为 `fdm_kv_aggregate.so`，导出
`fdm_kv_aggregate(struct cemu_args *)`。它不依赖 native buffer，也不另申请 KV 页池。
复用 FDM 自己的布局计算和共享 ABI，布局计算源码已去除 QEMU 专用头依赖。
输入约定：`mr_addr[0]` 为初始化后的 MR0，`mr_addr[1]` 为 MR1，长度必须匹配绑定；
`cparam1=CEMU_FDM_KV_AGGREGATE_COMMAND`，`cparam2` 为设备调度入口取得的当前虚拟时间，
不是 Guest 可任意指定的就绪时间。该时间参数只检查前置传输依赖，不是聚合 runtime。
调用者必须独占该 buffer，保证 MRS/算子生命周期与 Host 数据一致性。

逻辑步骤：

```text
READY + 传输目标时刻已到
  → 校验所有 head 的连续追加前缀、reader、需要的新页和空闲页链
  → 整包资源足够才进入 AGGREGATING
  → 读取 MR1.input 中的 K/V，直接填入 MR1.staged_data
  → 更新 MR0 的页目录、有效 token 数和 domain 队列
  → 发布所有 head 的 accepted，进入 AGGREGATED
  → fdm_kv_receive_release() 清收包状态、递增序号
  → 原 WRITE CQE，Guest 可写下一包
```

包内 K、V 各自按 `[head_count, token_count, head_dim]` 排列，一个包对应一个 batch/layer。
token-K/V 页按 token 连续存储；channel-K 页按 `[channels_per_page, channel_group_tokens]`
存储，从同一份 K 转置填充。未满页保留在页目录中；满页只入队一次。
domain 按 batch/layer/kind/NAND-channel 区分，group 序号取模选 NAND-channel，不混淆
embedding channel 与 NAND channel。队列 `count >= block_pages` 表示已具备整块下放条件；
这一步只形成逻辑页键和候选队列，不分配真实 PPA、不提交 NAND、不推进 committed。

返回值：`OK=0`、`INVALID=1`、`BUSY=2`、`NO_SPACE=3`、`WAIT_TIME=4`。
前缀错误、reader 占用、空间不足或传输尚未到期均在修改前返回，MR0/MR1 保持不变。
模块成功返回时保留输入包和 bitmap。`fdm_kv_complete_write()` 持锁调用模块并完成
Host dirty 标记及输入槽释放，之后 poller 才返回原 WRITE 的 CQE。
`fdm_kv_receive_release()` 只允许从 AGGREGATED 释放；它清空 bitmap/解析头、
递增 request_id，但不清 KV 暂存页。聚合不额外建模 runtime，不再另等一个聚合 deadline。

本阶段明确选择非冻结软件处理：设备通过已绑定 program 的入口调用模块，不进入
`run_functional_modeling()`。解绑时记录 `FDM KV non-frozen aggregation: calls=..., host_ns=...`，
这是模块调用的 Host monotonic 累计耗时，不包含前置 IO/排队，也不追加到虚拟时间。
其他任务仍可能冻结全局虚拟时钟，因此该 Host 数值不是虚拟时长或校准后的硬件性能。
后台写回独立于聚合模块，见下面的 writer 生命周期。下一步是 KV 专用读取和上层推理接线。

Host 验证命令：

```bash
make -C tests/cemu/kernel ../build/fdm_kv_aggregate.so
make -C tests/cemu instattention-fdm-buffer-aggregate-test PYTHON=/usr/bin/python3
```

测试实际编译并 dlopen 独立 `.so`，使用同一套 FDM 初始化/收包代码，逐字节检查 K 双布局/V、
多 head、多 batch/layer 隔离、FP16/FP32、head_dim=64/128/256、token/channel group 边界、
满页入队/部分页不提前下放、整包原子预检、虚拟就绪时间门槛、输入槽释放和旧包拒绝。
这个独立模块目标没有启动 QEMU；上面的 write-qtest 另行运行真实 QEMU 设备模型。
两者都没有启动 Guest Linux；独立模块测试不包含 NAND，write-qtest 包含设备后台写回，
均不代表新路径的 Attention/完整推理已跑通。

### KV FTL 整块放置 A/B

`ftl_mode=instattention` 新增控制器属性 `native_kv_placement=legacy|balanced`。
默认仍是 `legacy`；Native APPEND 和 FDM buffer 共用此 FTL 选项，无需改 Guest 写入协议。
`femu-scripts/run-native-opt-test.sh` 对应的 Host 环境变量是
`CEMU_NATIVE_KV_PLACEMENT=legacy` 或 `CEMU_NATIVE_KV_PLACEMENT=balanced`，
启动输出、`qemu.log` 和 `guest-logs/launch-settings.txt` 都记录选择。
每次切换必须关闭旧 VM、重新启动 fresh native NVM session；不支持运行时切换或复用旧映射。

- `legacy`：logical domain 的 token-group bucket 固定到物理 NAND channel，
  该 channel 上各 kind 共用 block 分配计数，可能出现 channel-K 只用部分 channel、
  交错 K/V 各用部分 LUN 的现象。
- `balanced`：domain、暂存页池、满块条件及组页顺序全部不变，只在分配**新物理块**时，
  按 task/layer/kind 的独立游标轮转 channel/LUN/plane，各 kind 的初始位置错开。
  batch 共享同一 layer/kind 游标；lane 内 block 编号由 namespace 统一分配，防止不同
  task/domain 占用同一块。已用尽的 lane 会跳过；当前未满块仍可继续写。
- 一块内的所有页仍在同一 channel/LUN/plane/block，page 连续递增；不拆块、不改
  head/token/channel 页格式。读路径查已保存的 KV→PPA 映射，不再从 bucket 推断物理 channel。
- 新游标和 lane 计数属于有界 FTL 元数据，计入映射内存预算，不增加第二份 KV buffer。
  不改 NAND 延迟、freezing、完成队列和上层提交并发度；同一 LUN 的页操作仍串行建模。
  这是确定性放置策略，不是根据当前忙闲动态迁移，也不保证稀疏选择后能用满所有 LUN。

Host 回归（不启动 Guest Linux/VFIO，不使用 GPU）：

```bash
make -C build -j4
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=tests/cemu/instattention/python \
  CEMU_NATIVE_QTEST_CUDA=0 /usr/bin/python3 -m unittest -v \
  test_native_kv_contracts test_instattention_qemu_io \
  test_instattention_native_qtest test_native_csd_pipeline
```

映射测试覆盖两种策略、跨 task/batch/layer/kind 唯一性、整块连续、所有 lane 分布、
读就绪门槛、耗尽和分配失败。真实 NAND adapter 的受控测试保持 32 pages/kind 和
40 ns/page 的测试参数不变，4 channel × 2 LUN 下全读目标间隔由 legacy 640 ns 变为
balanced 160 ns；这是小型调度回归，**不是 OPT-13B 性能结果**。
QEMU qtest 另验证两种策略的 Native/FDM Attention 数值及小模型 Prefill/Decode。
实际 A/B 请保持同一 buffer 路径、硬件配置、ratio、HPET、I/O backend、batch/prompt/decode
和诊断开关，只切换此放置属性；观察 Flash 页数、每 LUN 页数、读取模型时间和端到端时间，
不能把放置更均匀直接当成论文性能复现或时间校准完成。

### FDM writer、NAND 完成和终止 drain

`hw/femu/fdm_kv_writer.c` 是设备侧运行态，不是另一份 native buffer。
MR0 继续持有 head/staged/hash/domain/writeback 元数据，MR1 持有唯一的暂存页池。
Host 侧只新增线程/条件变量、有界页描述符及内部完成事件封装，不分配第二份 KV 数据。

```text
聚合成功 → 唤醒 writer
  → 同一 domain 队列凑够 block_pages，摘取一块页链
  → 按 batch_pages（最多 256）分批，直接借用 MR1 的页指针
  → bb_native_kv_batch → KV 坐标映射/PPA 分配 → 原 NAND 调度及后端写入
  → 取本块各批最大 ready_ns，投递 internal_complete 到 to_poller/pq
  → 目标时刻到达：摘除该块的页目录项，推进连续 committed，归还页槽
```

worker 不等待每块 NAND 期限，可以继续提交其他 domain；不同物理资源的重叠与同一 LUN
串行约束仍由原 NAND 模型决定。等待中的页仍在 MR1/hash 中，不因后端 memcpy 已完成就
提前回收。内部完成没有 Guest CQE，也不再增加 PCIe 时间。共享 ring 满时，poller 自己
产生的 native IO 完成直接转入其 pq，避免等待自身消费 ring 或丢失完成。

写回首次实际提交时注册 KV 映射：FDM `session_id` 完整映射为 FTL 的 64 位 `task`，
geometry/coordinate 逐字段转换，不把 `FDMKV_Buffer_t` 强转为 `NativeKvBuffer`。
映射沿用现有只追加规则；reset/解绑不删除已写映射。写过的 session 不可从 token 0 重绑
重写，后续新绑定应使用新 session；仅递增 generation 也不能代替映射回收。

终止包复用输入槽/收齐协议：`operation=2`，`wire_bytes=512`，`payload_bytes=0`，
batch/layer/head/token 字段全零，session/generation/request_id 正常校验，其余字节补零。
它不调用 APPEND 聚合模块，而是封口 buffer，把尚未满页的有效数据也排入队列，允许尾块
下放。原 DRAIN WRITE 直到所有目标写回完成、`staged_pages=0` 且 `sealed=1` 才返回成功。
之后不能继续 APPEND。这不是 `fsync` 或 SSD 断电持久化模型，只保证模拟 NAND 写回完成。

停止不依赖虚拟时钟推进；取消不推进 committed，也不回滚已经复制的 NVM 后端/映射。
新增路径不改变 freezing，不给聚合或组块另设合成 runtime，NAND 时间仍取原 FTL 接口。

## Native 页组与块聚合（2026-10-17）

当前 native 写回采用两级组织：**一组一页，多个完整页攒成一个物理块后下放**。
例如 4 KiB 页、FP16、d=128、C=8：token-K/V 每组保存 16 token × 128 维；
channel-K 每组保存 256 token × 8 维，同一 head 的 128 维对应 16 个页组。
旧字段 `channel_group_pages=16` 仅保留 ABI 名称，表示同一 token 跨度的页组数量。

```text
APPEND wire buffer → 直接填 token-K / channel-K / token-V 的 DRAM 页
       → 页满后入队（未满页也在页目录中可读）
       → 同一 batch/layer/layout/NAND-channel 的块队列，可混合不同 head
       → 满 pages_per_block → 分批提交 page program → 原生 to_poller / pq 完成队列
       → poller 到块内最大 deadline 回调 → 发布/释放；writer 继续提交后续块
```

`batch_pages` 仅是内部调用上限（1–256），不是聚合门槛。物理块页数取设备 `pgs_per_blk`；
终端 `BUFFER_DRAIN` 是允许未满页补零、未满块刷出的唯一例外，drain 后不可继续 APPEND。
普通 APPEND 成功仍只保证设备接收，Prefill 完成也不保证全部 KV 已写 NAND。

读取用 `(task/generation, batch, layer, head, kind, token, channel)` 定位页键和页内偏移，
再查 DRAM 页目录或原有 PPA 目录。未满页、等待满块的页及等待写完成的页均可参与 Attention。
原始 K/V ring 和后台 `pack_ready_span()` 已移除；APPEND 在锁内先检查所有 head 的前缀、
reader 及整包所需新页数，容量足够才修改页和 accepted，PENDING 不改变任何 head 或页。
`packed[3]` 保留 STATUS ABI，现均等于 accepted，表示三种布局都已直接填入；`committed`
仍表示 NAND 完成的连续 token 前缀。不同页可能乱序完成，不能仅按 committed 判断位置。
读取统计的 `tail_bytes` 表示 DRAM 页命中字节；`tail_copies` 计实际连续页片段 memcpy 次数。

`BUFFER_STATUS` 保留原有 0–87 字节，在 offset 88 起追加 7 个 uint64：
三种 `packed`、`staged_pages`、`write_blocks`、`partial_blocks`、`ready_blocks`。
其中 staged 包含未满页及目标写入尚未完成的页；blocks 计数在目标完成后增长。
offset 144/152 追加 `inflight_blocks` / `peak_inflight_blocks`，计正在准备提交或已提交
但尚未完成的块及其峰值，不能单凭该计数证明 NAND 同时执行。
`INFO` offset 112/120 为物理块页数/本策略版本 2（直接组页）。

块写回不再在 writer 内逐块等待目标时刻。内部完成事件复用原生 NVMe poller 的
`to_poller` 和优先队列，到期仅执行页回收回调，不产生 Guest CQE、不占 CU、不重复添加
PCIe 预算。事件池有界且计入 buffer 预算；完成前页仍可从 DRAM 读取，完成后才能复用。
不同 channel/LUN 可有多个块在途，同一 LUN 的 program 仍受原 NAND 可用时间约束。
停止或 reset 会取消未完成事件，取消不视为成功写回；freezing 和 NAND 时间模型不变。

为避免页池耗尽却无完整块可写的死锁，注册预留各域残块、各 head 未满页及最大整包追加余量；
预算不足返回 LIMIT。运行时空间不足返回 PENDING，等待已有满块或在途块释放页。
OPT-13B `b4-p1024-d16` 的 Host/Guest 启动脚本均已把 native buffer 预算从 1 GiB 调至 6 GiB；
这是模拟器内存预算，不等同于论文设备 DRAM 容量或已校准的 DRAM 时延。
当前仍保留按 batch/layer/layout/NAND-channel 隔离的分配域；没有新增 GC/擦除回收或 DRAM
拷贝时延模型。整块下放也不能据此宣称已测得 WAF=1。

验证包含跨 head 填满同一物理 block、每次仅提交一页、channel 页跨块且部分落盘、
暂存与 NAND 混合读取、terminal drain、冻结时钟取消及 Attention 数值/读取诊断回归。
写入路径现在为 Guest RAM → wire buffer → 最终布局页 → NVM backend，移除了 raw ring
中转和页写入前的 I/O scratch。同步 transfer 只借用页指针，返回后不保留它；页直到目标
NAND deadline 才复用。读取仍保留 scratch，以保证 backend 失败时不覆盖调用者输出。
channel-K 转置前移到 APPEND 处理内，APPEND CQ 前的软件成本会变化；减少复制层数不等于
已经测得端到端加速，也没有增加新的 DRAM/DMA 时延模型。
旧 QEMU 进程不会自动加载改动，需要使用新构建的二进制启动新 session。

## 先读这两份

- [开发历史与保存基线](../../../docs/cemu/DEVELOPMENT_HISTORY.md)：GPU接入、真实模型、公共流水、
  SparF自治、grouped KV、上层/FTL审查，以及已执行测试的证据边界。
- [FTL工作记录](FTL_WORKLOG.md)：从2026-09-21起持续记录独立KV FTL的设计、修改、测试和风险。
  原生KV独立模式已接入初始化/NVMe/FTL worker/CQ，真实QEMU qtest通过；
  Guest `/dev/ng0n1` 小测试已通过，命令见第12节；内部单页/批量IO入口见第15节。
  有界buffer/后台写回已通过隔离QEMU及用户Guest测试，见第16节；
  native稀疏批量gather内部接口及Host回归见第17节；
  新FTL读取接入原有SparF状态机/CPU算子的Host验证见第18节；
  native FDM/MRS/compute、公共OPT入口及收尾回归见第19节。
  小型合成OPT已通过隔离QEMU全链路数值验证，真实Guest GPU/OPT仍待验收。

地址基础回归：在 `tests/cemu` 执行 `make instattention-native-address-test`。
只编译临时C harness，验证坐标/页偏移与边界，不访问设备，不证明FTL物理放置或回收已经实现。

映射/分配回归：同目录执行 `make instattention-native-mapping-test`。
验证任务隔离、PPA放置、假backend字节读回、目标时间可见性和失败回滚；不访问现有实验镜像，
不代表真实设备IO、NAND时序接入或块回收已经完成。调用契约与放置假设见FTL工作记录第10节。

IO回归：`make instattention-native-io-test`，验证临时文件与目标依赖。
生产适配回归：`make instattention-native-qemu-io-test`，需要Host的QEMU构建配置，
直接链接生产FTL/NAND/backend函数，不启动VM、不操作GPU或实验镜像。
目标时间返回后的完成职责和未接入项见FTL工作记录第11节。

真实QEMU协议回归：`make instattention-native-qtest`，使用临时文件及qtest加速器，
不操作现有VM/GPU/镜像。独立Guest启动：`bash femu-scripts/run-native-kv-test.sh`（CEMU根目录）。
这不是OPT实验启动脚本，不要给旧48/96GiB实验镜像直接加`ftl_mode=instattention`。
新Guest缓存测试：`make instattention-native-buffer-device-test NVM_DEVICE=/dev/ng0n1`，
每台新启动的专用测试VM运行一次；不要挂载/格式化native NVM，也不需要FDM/GPU/模型。

Native Attention桥接回归：在Host的`tests/cemu`执行
`make instattention-native-attention-test`。需要已配置的QEMU build、C编译器、NumPy及
可导入的Host版`_cemu_client`；使用临时文件，直接调用生产native FTL及SparF状态机/CPU算子，
不启动OPT、不使用GPU。完整NVMe/FDM/MRS命令链由下面的qtest单独验证。

## 测试入口与清理范围

当前主线保留以下入口（在 `tests/cemu` 执行）：

| 范围 | Make 入口 | 用途 |
| --- | --- | --- |
| Native 存储与设备链路 | `instattention-native-regression-test` | 协议、FTL、tail、Attention 和小型 OPT 回归 |
| OPT 数学正确性 | `instattention-opt-reference-test` | 与 Hugging Face 对照 Prefill/Decode |
| OPT 真实模型全流程 | `instattention-opt-cemu-gpu-test` | dense/SparF/grouped/native，按参数选择路径 |
| TinyLlama 数值对照 | `instattention-tinyllama-reference-test`、`instattention-tinyllama-decode-reference-test` | 分别验证 Prefill 全层 KV/logits 和 Decode |
| TinyLlama 设备全流程 | `instattention-tinyllama-cemu-gpu-test` | 保留旧模型端到端回归 |
| 公共调度 | `instattention-flexgen-prefill-test`、`instattention-flexgen-decode-test`、`instattention-flexgen-attention-async-test` | runner、microbatch、预取及后台写回依赖 |

已删除早期 `test_tinyllama_checkpoint.py`、`test_tinyllama_prefill.py` 和
`test_flexgen_cemu_pipeline.py`，并移除对应 Make 入口。权重缓存/形状检查保留在
`test_hf_checkpoint.py`；真实权重加载、Prefill 和 CEMU Decode 联通由上述 reference
及全流程测试覆盖，不再维护单独的加载/形状冒烟脚本和合成 FlexGen 设备全流程入口。

第二轮撤掉六个旧设备验收脚本及其 Make 入口：

| 删除脚本 | 后续使用的入口/覆盖范围 |
| --- | --- |
| `test_python_bridge.py` | `test_cemu_device.py` 保留 vadd、Python bridge 和 MRS 生命周期检查 |
| `test_attention.py` | `test_cemu_attention_slot_device.py` 保留 dense 数值与双 slot 设备测试；真实模型由全流程入口验证 |
| `test_cemu_attention_phases.py` | 保留 `test_attention_phases.py` 算子数值测试，不再维护独立双 execute Guest 演示 |
| `test_cemu_attention_workflow.py` | 退休早期 dense 六 MR 工作流验收；当前 SparF/native 依赖时序由 stages、timeline 和 native 回归检查 |
| `test_sparf_group_io_device.py` | grouped IO 细项留在 `test_sparf_group_device.py`；Guest 使用 grouped+Attention 综合入口 |
| `test_sparf_group_buffer_device.py` | tail/背压/发布/环形复用细项留在 `test_sparf_group_buffer.py`；Guest 使用 grouped+Attention 综合入口 |

grouped Guest 综合入口为 `instattention-sparf-group-attention-device-test`，native Guest
仍使用 `instattention-native-buffer-device-test` 和 OPT 全流程。这不是声称新入口逐项复刻
所有旧演示（例如旧 dense 工作流的 serial/overlap 对照）；被退休路径的专用设备验收不再维护。
`require_fdmfs` 移至 `device_test_support.py`，避免保留测试依赖已删除的脚本。
`test_cemu_sparf_workflow.py`、`test_sparf_buffer_device.py` 仍由 `test_sparf_stages.py`
动态调用，保留其 staged/autonomous/旧写回回归，不因名字较老而直接删除。

其余底层单元测试暂不按文件名或日期删除：有些仍覆盖支持中的旧后端，有些提供
native/grouped 测试共用的 fixture。`FUNCTIONAL=1` 只验证流程完成，不能替代数值对照，
各入口需要的模型、GPU、Guest 设备或 Host 构建环境仍按其原有要求准备。

## Native 收尾验证

在Host执行，不能在Guest内用Guest版Python扩展替代Host版：

```bash
cd /home/lihan/cemu/CEMU
make -C build -j4
cd tests/cemu
make instattention INSTATTENTION_BUILD_DIR=/tmp/cemu-instattention-host
make instattention-native-regression-test INSTATTENTION_BUILD_DIR=/tmp/cemu-instattention-host
```

需要Host Python的NumPy、PyTorch、pybind11及开发头文件。回归使用临时NVM文件和qtest
Unix socket，不启动KVM Guest、不操作现有VM/GPU/实验镜像。2026-09-23共100项测试通过。
单独复跑设备计算和小型OPT链路：

```bash
PYTHONPATH=/tmp/cemu-instattention-host make instattention-native-csd-qtest
```

该小型OPT测试复用生产Prefill/Decode runner、OPT数学操作、KV adapter、native协议封包及
SparF提交逻辑；仅把Linux ioctl/FDMFS传输换成qtest NVMe命令，算子在CPU上执行。
两层、双batch、prompt=127、Decode=3、FP32、ratio=1，对照PyTorch的token/logits/hidden，
同时验证实际NVM读写和最终DRAIN。它不证明Guest驱动、GPU、异步重叠或目标时间已校准。
FP16/GQA及ratio=8的稀疏数值和边界仍由native Attention回归覆盖。

下一步先用**独立、新建后端**验收Guest OPT-125M，再扩大OPT-13B；native NVM不能mkfs/mount，
不能复用旧grouped的文件池、cookie、模型实验镜像。配置和限制见FTL工作记录第19节。

## Guest io_uring transport (opt-in)

`CEMU_IO_BACKEND=uring` selects the same SQE128/CQE32 NVMe passthrough mechanism
used by `tests/cemu/cemu_benchmark.cpp`. `ioctl` remains the default for regression
comparison. Set this variable before starting Python; do not change it mid-run.
No QEMU, native FTL, SparF algorithm, target runtime, or freezing changes are required.

The transport covers Execute, the C++ client's FDM reads/writes, native KV commands
(including APPEND), and dense staging's `copy_file_range`. Program download,
activation and MRS management remain ioctl operations, as in the benchmark.
Native KV still reads/writes internally: it does not gain an NVM file or switch
back to Guest `copy_file_range`. Other legacy Python `os.pread/pwrite` paths are
not converted by this switch.

One C++ dispatcher owns a depth-64 ring, submits ready requests, and routes CQEs
by request identity. Concurrent callers share this ring. The blocking adapters keep
their existing buffers alive until CQE completion; the dispatcher borrows those
buffers without an additional staging copy and holds duplicated descriptors.
Short reads/writes and NVMe
status/result are preserved. Negative CQE errors fail the request, whereas
positive NVMe status remains available to native backpressure handling.

This integration **retains the existing Python workers, Future interface and
dependency waits**. Synchronous client methods wait on dispatcher promises; this
is not yet a replacement of all Python workers by a CQE-driven runner. The ring
submission/completion mechanism is aligned with CEMU, not the entire benchmark
Job state machine. Native Prefill now has a bounded APPEND window as described
below; other synchronous methods still wait for each command. Speedup is not guaranteed.

### Native Prefill: multiple APPENDs in flight

With `CEMU_IO_BACKEND=uring`, `CEMU_NATIVE_APPEND_DEPTH` controls the native
Prefill APPEND window (default **8**, allowed **1–64**). Depth 1 uses the original
submit/wait loop. ioctl and one-token Decode retain their serial submission path.
The negotiated 128 KiB limit and token-group-sized packetization are unchanged;
this change does not enlarge commands or modify tail/FTL placement/freezing.

The Prefill writer partitions a layer's K/V into disjoint head ranges within
each batch item. It submits independent ranges without waiting for earlier CQEs,
reaps any completed request, and refills the window. Each head's next token
packet waits for its previous packet to be accepted: NVMe completion order is
not assumed. A backpressured packet is retried unchanged; later tokens for those
heads cannot overtake it. One head range alone therefore remains serial.

The asynchronous C++ request owns a payload copy and a duplicated descriptor
until completion. Wait-any releases the Python GIL and uses completion
notification, not Python polling. On permanent failure or timeout, the writer
drains all outstanding handles before failing the outer Future and poisoning
the store. Success of that Future means every APPEND was accepted, not that
background NAND writes have drained. Existing final drain remains necessary.
Old bindings without the asynchronous API fail explicitly, without ioctl fallback.

Observe `client_append_queue_depth`, `client_append_peak_inflight`,
`client_append_commands`, `client_appended_head_tokens` and
`client_append_backpressure_retries` in the final native statistics.
Peak means submitted handles not yet reaped by the writer, **not measured
simultaneous device execution**. The native command worker is still serial;
multiple in-flight commands remove the Guest round-trip barrier, not that worker's
serialization or NAND resource constraints.

In the Guest, rebuild/test using `instattention-io-transport-test` below. Then use
the existing full inference command with the same measured ratio file:

```bash
ratio=/root/opt13b-b4-p1024-d16.json
test -s "$ratio" && CEMU_IO_BACKEND=uring CEMU_NATIVE_APPEND_DEPTH=8 \
  bash /root/CEMU/femu-scripts/native-opt-guest-test.sh OPT-13B b4-p1024-d16 "$ratio"
```

For the serial uring baseline set `CEMU_NATIVE_APPEND_DEPTH=1` on a **fresh
native session**, with the same inputs, ratio, HPET clock and device configuration.
Check functional PASS, accepted-token/write accounting and append retry counts
before comparing Prefill time. Different background-write timing may change
tail/Flash read counts; do not infer improved simulation accuracy from speedup.

Build inside the CEMU Guest with liburing >= 2.2 development files and headers
providing `NVME_URING_CMD_IO`. CMake prints whether support was compiled. A build
without these dependencies retains ioctl support, but selecting uring fails
explicitly, with no fallback. After installing newer headers, clear the cached
probe with `cmake -U CEMU_HAVE_URING -S instattention -B /root/native-opt-client`.
`uring_copy_file_range` uses CEMU's custom opcode 55, not stock Linux's opcode 55;
only use this path with the matching CEMU Guest kernel.

First validate the transport in the Guest (no GPU/FTL changes):

```bash
source /root/cemu-gpu-env/bin/activate
cd /root/CEMU/tests/cemu
make instattention-io-transport-test INSTATTENTION_BUILD_DIR=/root/native-opt-client
```

The C++ test exercises shared-ring concurrent file I/O, asynchronous request
handles, wait-any, descriptor lifetime, short reads, EOF, and isolation after
rejected NVMe commands. An unsupported kernel/sandbox reports
SKIP, **not a device validation PASS**. Python tests use mocks for native command
packing/status, out-of-order APPEND completions, per-head ordering, backpressure,
failure draining and dense copy offsets; neither replaces a real CEMU run.

In a native Guest with FDM mounted, the same executable optionally checks native
INFO payload/status (including 128 queries in eight-request windows) and compares
ioctl/uring FDM + MRS + CPU vadd Execute results:
`/root/native-opt-client/test_io_transport --cemu /absolute/host/path/to/vadd.so`.
The operator path is resolved by the Host CEMU process, not by the Guest.

Then, in an already booted native Guest, run the existing small inference first:

```bash
CEMU_IO_BACKEND=uring bash /root/CEMU/femu-scripts/native-opt-guest-test.sh OPT-125M
```

Use `OPT-13B` instead if that is the only available model. For the existing
paper-sized profile, keep the same ratio/configuration and prepend
`CEMU_IO_BACKEND=uring` to the usual command:

```bash
ratio=/root/opt13b-b4-p1024-d16.json
test -s "$ratio" && CEMU_IO_BACKEND=uring \
  bash /root/CEMU/femu-scripts/native-opt-guest-test.sh OPT-13B b4-p1024-d16 "$ratio"
```

Check `[opt-cemu] Guest I/O backend=uring` in the log. For A/B use the same command
with `CEMU_IO_BACKEND=ioctl` on a fresh native session. Keep geometry, ratio and
inputs identical; functional PASS and matching tokens do not establish timing
calibration or bitwise numerical equivalence.

## Native Guest OPT-125M

`native-opt-guest-test.sh` 默认在模型加载前将 **Guest** clocksource 切换为 `hpet`，
并检查可用性、切换结果及推理前后的实际值。不可用或切换失败则停止，不静默回退。
日志 `[opt-clock]` 记录使用的时钟源；切换保留到手工更改或重启，下次脚本运行会再次
设置。只影响该测试Guest，不修改宿主机、冻结机制、runtime或原有run-csd启动脚本。
如需旧计时口径的诊断对照，可在Guest运行时设置 `CEMU_GUEST_CLOCKSOURCE=kvm-clock`；
日志会警告这种口径可能包含冻结执行耗时，不能和HPET结果混用。

2026-10-10同一KVM Guest探针确认：目标100ms、实际执行1秒时，kvm-clock应用计时
约1100ms，HPET约100ms；两者QEMU功能冻结区间虚拟时间增量均为0。这只验证时钟
冻结语义，不代表直通GPU、异步流水或目标模型已完成校准。HPET读取开销更高，CUDA
event时间不随Guest clocksource改变。直接运行Make/Python不会自动切换时钟；应先
在Guest中检查/设置 `current_clocksource`，不要在宿主机执行时钟切换命令。

这是实际Linux Guest验收入口，不是qtest。模型计算使用Guest直通GPU，CSD使用CPU SparF，
先验证功能，不注入FPGA runtime/Softmax ratio、不用于论文性能对比。
默认batch=1、prompt=32、Decode=4、top-r=16、ratio=8；FP16 OPT-125M的token组为32个token，
因此会跨token组边界，channel-K尾组由最终DRAIN发布。

Host先正常关闭使用同一基础系统盘/VFIO GPU的旧VM，再执行：

```bash
cd /home/lihan/cemu/CEMU
make -C build -j4 &&
make -C tests/cemu/kernel ../build/sparf_attention.so &&
sudo bash femu-scripts/run-native-opt-test.sh
```

默认直通`0000:27:00.0`，必须事先已绑定vfio-pci；脚本不解绑、不重置其他GPU。
每次在`/data/lihan/cemu-native-opt/opt125m.XXXXXX`创建系统盘overlay、新512MiB NVM文件、
Host日志和Guest日志目录；FDM为64MiB，native buffer预算64MiB。基础系统盘不写入，
源码和模型只读共享。不是旧grouped模式；`run-csd.sh`默认仍是旧FTL，只有显式设置
`CEMU_FTL_MODE=instattention`及对应native参数后才能用其测试native路径。

进入`root@...:/#`之后在Guest执行：

```bash
mountpoint -q /proc || mount -t proc proc /proc
mount -o remount,rw /
mkdir -p /root/CEMU
mountpoint -q /root/CEMU || mount -t 9p -o ro,trans=virtio,version=9p2000.L cemusrc /root/CEMU
bash /root/CEMU/femu-scripts/native-opt-guest-test.sh
```

Guest脚本会挂载sys/dev、模型/日志共享及FDM，使用已有`/root/cemu-gpu-env`，
在Guest系统盘内构建Python扩展，再执行真实OPT入口。需要该环境已有PyTorch、NumPy、
pybind11和模型加载依赖；不自动安装驱动或包。算子在Host编译，QEMU工作目录与Guest的
`./build/sparf_attention.so`引用一致，避免把Guest编译的库错误加载进Host。

扩展构建直接使用pybind11头文件和`Python3_add_library`，不依赖系统pybind11的导出
CMake target；避免某些配置把头文件位置解析成不存在的`/include`。头文件先从指定Python
环境查询，再搜索系统include目录，也可显式设置`CEMU_PYBIND11_INCLUDE_DIR`。
Guest脚本设置`CEMU_INSTATTENTION_BUILD_DIR=/root/native-opt-client`，测试优先加载本次
生成的扩展，而不是只读共享目录里的旧`build-guest`副本。

观察`KV storage=native FTL/token-channel addressing`、Prefill完成、48次Attention、
最终DRAIN及`[opt-functional] PASS`。日志沿用的`grouped`字样在这里包含复用adapter的native
路径，应以storage标识为准。functional PASS仅表示真实模型流程完成，不是逐层reference比较。
不要先运行另一套buffer测试占用native会话；每次完整重测需正常关机、重新启动新会话。

退出：`sync; /sbin/poweroff -f`。日志持久保存在Host本次目录的`qemu.log`和`guest-logs/`。
禁止运行旧`cemu-mount.sh`或对native NVM执行mkfs/fio。

### 正常登录模式与 OPT-13B

125M的真实Guest功能流程已经通过。继续13B时可恢复systemd、账号登录和localhost:2222 SSH，
不需要退回旧FTL。启动脚本接受`OPT-13B`，自动配置64GiB Guest RAM和512MiB native buffer预算；
仍共用512MiB NVM/64MiB FDM的小负载配置，batch=1、prompt=32、Decode=4。

先在当前rescue Guest更新其系统盘内的挂载脚本，再关机：

```bash
cp -a /root/cemu-mount.sh /root/cemu-mount.sh.before-native
cp /root/CEMU/femu-scripts/cemu-mount.sh /root/cemu-mount.sh
sync
/sbin/poweroff -f
```

更新后的脚本根据内核命令行`cemu_native_opt_smoke=1`选择安全分支：挂载源码、模型、日志
和FDM，不格式化native NVM、不运行fio、不创建旧的FDM占位文件。没有这个标记时保持旧行为。
必须更新Guest里的副本，单改Host仓库不会替换`/root/cemu-mount.sh`。

Host使用**刚才Guest实际使用的overlay**，不是原始基础镜像。例如本次125M的overlay为
`/data/lihan/cemu-native-opt/opt125m.CTYarJ/guest.qcow2`：

```bash
cd /home/lihan/cemu/CEMU
sudo env CEMU_NATIVE_BOOT_MODE=normal \
  CEMU_NATIVE_OS_IMAGE=/data/lihan/cemu-native-opt/opt125m.CTYarJ/guest.qcow2 \
  bash femu-scripts/run-native-opt-test.sh OPT-13B
```

normal模式不加`init=/bin/bash`，复用指定测试系统盘，保留其环境；每次仍新建native NVM
后端和日志目录，禁止把旧NVM数据当作有重启恢复能力。不能删除这个测试系统盘或它的backing
基础镜像。若实际overlay路径不同，必须替换为启动日志记录的路径，不要自动选择不明镜像。

按原来的账号密码登录Guest后执行：

```bash
bash /root/cemu-mount.sh
bash /root/CEMU/femu-scripts/native-opt-guest-test.sh OPT-13B
```

这次使用真实13B权重、40层，共160次逻辑Attention；仍是Guest GPU+CSD CPU的功能验收，
不加FPGA时间模型。正常模式测试结束使用`poweroff`。首次normal boot若有历史fstab或其他
自启动任务报错，先保留输出检查，不要用mkfs/fio修复native namespace。

### Native OPT-13B：恢复 B4/P1024/D16 观测负载

`CEMU_NATIVE_PROFILE=b4-p1024-d16`使用现有
`femu-scripts/cemu_config_opt13b_sparf_engines2.json`：96GiB文件后端NVM、
8通道/每通道6个LUN、2GiB FDM、2 CU/2计算线程/2 SparF引擎。
native buffer预算为6GiB、页目录上限2097152；不再使用小测试的512MiB NVM。
本次只匹配之前常用的B4/微批4/P1024/D16负载，不是完整论文输出长度实验。
默认保留Guest GPU和CSD CPU后端；CUDA及诊断开关见下一节。CPU结果不能与旧CUDA后端
结果作仅改变FTL的严格性能归因。

容量预估：6400个stream、1040个KV位置，需要1344000个4KiB页（5505024000字节）；
native不再分配原始K/V环形缓存，按物理分配域预留最终布局页池、元数据及读取工作区，预算保持6GiB；
双slot Attention FDM预估32628736字节。文件后端是稀疏文件，Host实际空闲空间仍需
覆盖至少本次约5.13GiB KV数据和日志，并留余量。每次新建后端，不复用已经封存的会话。

先正常关闭当前Guest，然后在Host执行：

```bash
cd /home/lihan/cemu/CEMU
sudo env CEMU_NATIVE_BOOT_MODE=normal \
  CEMU_NATIVE_OS_IMAGE=/data/lihan/cemu-native-opt/opt125m.CTYarJ/guest.qcow2 \
  CEMU_NATIVE_PROFILE=b4-p1024-d16 \
  bash femu-scripts/run-native-opt-test.sh OPT-13B
```

Guest登录后使用已有、覆盖微批4的ratio文件（例中为之前保存的文件）：

```bash
bash /root/cemu-mount.sh
bash /root/CEMU/femu-scripts/native-opt-guest-test.sh \
  OPT-13B b4-p1024-d16 /root/opt13b-b4-p1024-d16.json
```

脚本在注册native会话/加载权重前检查ratio格式、anchor及全部所需shape；缺失或不覆盖时
直接停止，不会退回线性Softmax模型。需要补测时可在同一个尚未注册的Guest内运行：

```bash
source /root/cemu-gpu-env/bin/activate
cd /root/CEMU/tests/cemu
python3 instattention/experiments/bench_softmax_ratio.py \
  --device cuda:0 --batch-sizes 4 --prompt-length 1024 --decode-steps 16 \
  --compression-ratio 8 --output /root/cemu-logs/native-mb4-p1024-d16-ratio.json
```

之后把新文件路径作为Guest脚本第三个参数；若用新ratio，不应把与旧实验的差异全部归因于FTL。
保持`--functional`并输出profile，而不是不支持native会话复用的`--benchmark`。
预期640次逻辑Attention、每个batch生成17个token（Prefill首个加16次Decode）。
关注PASS、terminal drain、sealed/failed、Prefill/Decode/drain/e2e Guest时间以及各CSD modeled项，
不能相加重叠的worker时间，也不能把Guest时钟直接当作Host真实耗时。
设备配置快照保存在Host运行目录，Guest日志同目录保存ratio与runtime配置快照。

### Native CUDA 对照及等待分解

Host启动器支持`CEMU_NATIVE_CSD_TARGET=cpu|cuda`，默认cpu保持小测试兼容。
cuda要求显式设置一个Host GPU的`CUDA_VISIBLE_DEVICES`（单个编号或UUID），内部
`CEMU_CUDA_DEVICE=0`指可见集合内编号；不能把Guest直通GPU的编号当成CSD CUDA设备。
脚本不解绑、不重置GPU、不停止其他用户进程。`CEMU_CUDA_LIBRARY_PATH`仅对该QEMU进程
补充动态库搜索路径；是否需要575兼容目录取决于Host当前加载的驱动。

Native模式在`CEMU_CUDA_SYNC=1`时要求NVM/FDM后端均完成CUDA初始化；否则QEMU立即
拒绝启动，并输出CUDA状态码及错误字符串，不再等到OPT加载后以Execute status 6失败。
成功时打印`Native namespace 1/2: CUDA backend initialized`。独立Python探针成功不代表
另一个QEMU进程也成功；初始化失败的旧进程需要退出后重新启动。此检查不改变legacy模式、
freezing或runtime模型，也不保证后续GPU kernel不会失败。

加载13B前可在Host先运行已有Native CSD qtest的CUDA变体：

```bash
cd /home/lihan/cemu/CEMU
make -C tests/cemu/kernel ../build/sparf_attention_devptr.so
LD_LIBRARY_PATH="/home/lihan/cemu/nvidia-575-libs${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
CUDA_VISIBLE_DEVICES=1 CEMU_NATIVE_QTEST_CUDA=1 \
make -C tests/cemu instattention-native-csd-qtest PYTHON=/usr/bin/python3
```

要求Host Python已安装NumPy/PyTorch，且其依赖库（如CUPTI）可加载；575目录按本机驱动
选择，不是通用固定依赖。此入口使用64MiB隔离QEMU、临时文件NVM和真实CUDA QK/PV算子，
验证KV尾部/Flash读取、稠密/稀疏结果、复位及小型合成OPT生成对照；不启动Guest、不用
VFIO、不加载真实13B权重。默认不设`CEMU_NATIVE_QTEST_CUDA`仍运行CPU回归。CUDA模式
必须显式选择单张可用Host GPU；通过不代表64GiB/KVM/VFIO完整启动或性能校准已通过。

Guest自动根据启动参数选择`--csd-target cuda`及`sparf_attention_devptr.so`，不能只改Guest
命令却忘记Host的`CEMU_CUDA_SYNC=1`。模型推理仍在Guest GPU；CUDA算子用于精确QK/PV，
SparF选择流程仍由原设备工作流处理，并不声称整个SparF都变成GPU kernel。

以下命令保持上一轮native CPU运行的4个vCPU，只切换后端并打开诊断。
旧`run-csd.sh`使用8个vCPU；若与旧grouped路径对比，应双方统一设置为8。
vCPU数量也可能影响冻结交接，不能把同时修改vCPU的差异全归因于CUDA。

```bash
# Guest先正常poweroff；Host确认1号GPU可用后：
cd /home/lihan/cemu/CEMU
make -C tests/cemu/kernel ../build/sparf_attention_devptr.so
sudo env CEMU_NATIVE_BOOT_MODE=normal \
  CEMU_NATIVE_OS_IMAGE=/data/lihan/cemu-native-opt/opt125m.CTYarJ/guest.qcow2 \
  CEMU_NATIVE_PROFILE=b4-p1024-d16 CEMU_NATIVE_VCPUS=4 \
  CEMU_NATIVE_CSD_TARGET=cuda CUDA_VISIBLE_DEVICES=1 CEMU_CUDA_DEVICE=0 \
  CEMU_CUDA_LIBRARY_PATH=/home/lihan/cemu/nvidia-575-libs \
  CEMU_NATIVE_TIMELINE=1 \
  bash femu-scripts/run-native-opt-test.sh OPT-13B
```

Guest运行同一命令，沿用相同ratio，不重新测量它：

```bash
bash /root/cemu-mount.sh
bash /root/CEMU/femu-scripts/native-opt-guest-test.sh \
  OPT-13B b4-p1024-d16 /root/opt13b-b4-p1024-d16.json
```

Host和Guest启动输出均报告CSD后端；`/root/cemu-logs/launch-settings.txt`保存实际启动设置。
如果独立 CUDA 检查与不带 VFIO 的 QEMU 探针通过，但完整启动在 `cudaSetDevice` 失败，
可在 Host 启动命令加入 `CEMU_DEVICE_INIT_ORDER=femu-first` 做单变量对照。
默认 `vfio-first` 保持原启动顺序；两种模式均保留 VFIO GPU、相同 Guest RAM、模型和设备配置，
只交换 FEMU/VFIO 的 `-device` 顺序。脚本记录实际选择，不自动重试或回退 CPU。
该对照通过仅说明初始化顺序与现象相关，不能单独证明驱动失败根因，也不能替代完整推理验证。
`CEMU_NATIVE_TIMELINE=1`保存每次请求开始/完成及阶段时间戳到共享日志目录的
`device-timeline.jsonl`，默认0关闭。需要 native 页/LUN 诊断时，在 Host 的同一条启动命令中
同时设置 `CEMU_NATIVE_TIMELINE=1 CEMU_SPARF_READ_DIAGNOSTICS=1`；默认不采集详细字段。
运行结束后在Guest离线分析（不用再运行模型）：

```bash
python3 /root/CEMU/tests/cemu/instattention/experiments/analyze_sparf_autonomous.py \
  /root/cemu-logs/device-timeline.jsonl \
  --output /root/cemu-logs/native-requests.json \
  --boundary-summary /root/cemu-logs/native-boundaries.json
```

新增boundary摘要不依赖LUN诊断，分别列出Host/virtual父请求时间、engine排队、
读提交→观察到完成→目标ready、计算功能段、功能结束→调度提交→调度完成→观察到完成。
这些是边界区间，不单独等于冻结开销、CUDA kernel时间或Guest ioctl时间；累加值可能重叠。
诊断开关会增加日志开销，最终性能重复测量应关闭或双方保持相同设置。

### Native 读取收益与页/LUN 排队诊断

重新编译 QEMU 后，用原来的 native 启动命令增加上述两个环境变量；已有 QEMU 进程需要
正常关机、重新启动才能使用新代码。Guest 保持 B4/P1024/D16、同一 ratio、HPET、uring
深度等配置不变，运行原来的完整推理命令。无需增加一次 dense 推理，也不会额外发起读取。
结束后在 Guest 执行：

```bash
python3 /root/CEMU/tests/cemu/instattention/experiments/analyze_sparf_autonomous.py \
  /root/cemu-logs/device-timeline.jsonl \
  --latest-session --min-tokens 1025 --max-tokens 1040 \
  --require-read-diagnostics --require-native-read-reference \
  --read-summary /root/cemu-logs/native-read-summary.json \
  --read-loss-summary /root/cemu-logs/native-read-loss.json \
  --boundary-summary /root/cemu-logs/native-boundaries.json
```

- 三个阶段分别记录 `logical_bytes`（来自 Flash 的请求元素字节，可能含 GQA 重复）、
  `tail_bytes`、`aligned_bytes_before_dedup`、实际 `submitted_bytes` / `pages`。
- `unique_flash_bytes` 对同一物理页内的请求区间再去重，避免 GQA 重复请求把页利用率
  算到 100% 以上。`page_utilization = unique_flash_bytes / submitted_bytes`，其倒数
  为 `page_read_amplification`；没有 Flash 读取时为 null，不假报 0 倍。
- `full_scan` 是在同一阶段 buffer 锁内，根据该层/batch 所有 KV heads 的相同有效长度与
  committed/tail 边界计算的全量页数参照。包括末尾部分页；不查询/读取 backend，不调用
  NAND 调度，不改变发布进度。Channel-K 的全量参照只用于本阶段覆盖率，**不计入 dense K/V**。
- `kv_saved_bytes = dense_kv_flash_bytes - selected_kv_flash_bytes`；
  `net_saved_bytes = kv_saved_bytes - channel_k_flash_bytes`，允许负值。
  这是相同存储状态下的读取量参照，不是另一条 dense 执行轨迹或 dense 性能测量；
  K/V 分别使用各自读取阶段的快照，不能认为三个阶段是同一个原子快照。
- `channel_page_totals` / `lun_page_totals` 显示物理分布；`critical_initial_wait_ns`
  表示最终关键 LUN 首次被本阶段读取时已有的等待，剩余关键排队可能包含本阶段后续
  页及期间的其他任务争用。它不是所有页面等待之和。服务时间来自已有 FTL 结果，不重调度。
- 本诊断不改变页面布局、选择算法、runtime、FTL 时序规则或 freezing；会增加真实采集/日志
  开销，因此用于归因，不直接把诊断运行相对旧运行的 wall 变化当作性能改进。

旧日志缺字段时报告 missing；`--require-native-read-reference` 拒绝不完整的 native 参照。
完整 B4/P1024/D16 预期有 640 个 Attention 请求，需核对分析输出的请求数。

native结束统计提供`client_append_commands`、`client_append_attempts`、
`client_append_backpressure_retries`、`client_appended_head_tokens`和
`client_append_batch_max_bytes`；只维护整数计数，不逐命令打印或增加状态ioctl。
新QEMU通过INFO声明批量APPEND能力，客户端自动启用最多128KiB、同batch/layer多head的
有界提交；旧QEMU声明值为0时保持原4KiB接口。需要重启到新QEMU，单独更新Guest源码
不会改变正在运行的设备。native仍不创建三份Guest KV文件。

B4/GB4/P1024/D16、FP16/40heads/head_dim128、128KiB上限下，成功APPEND预计从
819200+102400=921600次降到30720+2560=33280次；背压尝试数不固定，head-token总量仍为
6656000。完整运行应检查功能PASS、sealed=1、failed=0和完整落盘字节5505024000，
不能把命令减少比例当成端到端加速比例。Prefill按token tile轮转各head，而不是先把一个
head的全部1024token提交完。每包先验证所有head，再统一接收；背压时没有部分提交。
成功表示设备buffer已接收，不表示NAND完成。组页、物理放置、Flash目标延迟和freezing/
runtime未调整；队列到达顺序和后台合批数量可以随批量提交改变。详细ABI和验证见
`FTL_WORKLOG.md`第29节。

## 冻结时钟独立验证

`freeze_clock_probe` 不加载模型、不访问 KV/NVM、不使用 GPU 算子，只申请一个 4KiB
FDM MR。固定目标 runtime=100ms，分别在 Host CPU 算子内部等待 0/200/1000ms，
每组测三次，先预热一次。等待发生在原有功能执行冻结区间，而非 Guest Python 中。
诊断不修改 freezing、runtime 计算或调度规则；请在没有其他 Guest 工作负载时运行。

Host 先编译（不会停止正在运行的 VM，正在运行的 VM 必须正常退出后再启动新二进制）：

```bash
cd /home/lihan/cemu/CEMU
make -C tests/cemu/kernel ../build/freeze_clock_probe.so
make -C build -j4
```

Native 模式有意禁止普通算子 Execute；本测试不放宽其安全限制，而是通过独立诊断脚本
运行原生 FEMU 模式。先正常关闭使用该系统盘的 VM，再启动 KVM 对照组：

```bash
sudo env CEMU_CLOCK_ACCEL=kvm CEMU_CLOCK_CPU=qemu64 \
  bash femu-scripts/run-freeze-clock-test.sh \
  /data/lihan/cemu-native-opt/standalone.PnutcI/guest.qcow2
```

该脚本使用4GiB RAM，不启用VFIO/CUDA、不读取OPT；16MiB NVM和8MiB FDM为全新实例，
系统盘写入独立overlay，不改原镜像；测试期间不得从其他VM写入该backing image。
原有run-csd.sh不变。诊断开关 `CEMU_COMPUTE_CLOCK_TRACE=1` 由新脚本传入，默认普通启动
不启用；QEMU日志产生 `CEMU_CLOCK_TRACE` JSON 行。无需重新编译Guest内核。

默认 CPU 型号为 `qemu64`，让 TCG/KVM 使用相同配置；此前探针使用的 `host` 型号
只能在 KVM 下使用。如需复查旧探针，另跑 `CEMU_CLOCK_ACCEL=kvm CEMU_CLOCK_CPU=host`，
不要把它与 TCG 的差异直接归因于加速器。脚本不自动回退加速器，每个会话保存
`environment.txt`，记录配置和 QEMU/kernel/initrd/设备配置/算子 SHA256。

Guest 登录后只挂载源码、日志与FDM；不要执行格式化NVM的脚本：

```bash
mkdir -p /root/CEMU /root/cemu-logs /mnt/fdm0
mountpoint -q /root/CEMU || mount -t 9p -o ro,trans=virtio,version=9p2000.L cemusrc /root/CEMU
mountpoint -q /root/cemu-logs || mount -t 9p -o trans=virtio,version=9p2000.L logsrc /root/cemu-logs
mountpoint -q /mnt/fdm0 || mount -t fdmfs /dev/nvme0m2 /mnt/fdm0
source /root/cemu-gpu-env/bin/activate
cd /root/CEMU/tests/cemu
make instattention-freeze-clock-test \
  INSTATTENTION_BUILD_DIR=/root/native-opt-client \
  CEMU_IO_BACKEND=ioctl \
  PROGRAM=/home/lihan/cemu/CEMU/tests/cemu/build/freeze_clock_probe.so \
  PROFILE_OUTPUT=/root/cemu-logs/freeze-clock-ioctl.json
```

PROGRAM 是 **Host QEMU 所见路径**，不能换成 Guest 的 /root/CEMU 路径。
JSON 文件必须不存在，避免覆盖旧结果。可以将 backend 改为 uring 并换一个输出文件，
无需重启 VM；这个测试没有创建 native KV session。结果读取和程序/MRS 初始化均不计入
Guest Execute 区间。预热一次加九个测量请求，总计十次 Execute。

新探针在同一个 Execute 外同时读取 Python `perf_counter_ns`、C++ `steady_clock`、
`high_resolution_clock` 和 `system_clock`；C++ 读数来自 Guest 扩展，不是 Host。
JSON 同时记录 Python 实现、C++ 时钟别名、Guest clocksource、内核版本、启动参数，
以及20次空测量区间的中位开销（不自动扣除）。各时钟在相邻位置采样，Python 区间
略大于 C++ 区间；只能比较各自差值，不能跨时钟相减。这验证计时 API，而不是复跑
benchmark 的 io_uring/Load/Compute/Store 全流程。

完成 KVM 组并 `poweroff` 后，在 Host 启动 TCG 组，Guest 重复完全相同的命令：

```bash
cd /home/lihan/cemu/CEMU
sudo env CEMU_CLOCK_ACCEL=tcg CEMU_CLOCK_CPU=qemu64 \
  bash femu-scripts/run-freeze-clock-test.sh \
  /data/lihan/cemu-native-opt/standalone.PnutcI/guest.qcow2
```

TCG 启动和 Guest 编译可能较慢。两组均无 VFIO/GPU，不要同时启动，不改 Guest
clocksource、不改 freezing/runtime，也不要把 TCG 的整体速度与 KVM 直接比性能。
探针使用 `read_range_bytes()` 读取结果，不导入 NumPy；默认 qemu64 不具备部分新
NumPy wheel 要求的 X86_V2 指令集。若曾在旧 `read_range()` 处遇到该错误，更新后
重新运行 make 即可编译 Guest 绑定，不需重启或升级 NumPy。失败运行可能留下空
JSON 文件，重试时换一个输出文件名，不覆盖原日志。
每个会话有独立日志目录，所以输出文件名可以相同。两组运行前应检查源码/二进制
未改变、设备配置哈希相同，并记录两个 session 路径。若只有 KVM 中所有应用时钟随
Host 延迟增长，支持加速器/Guest clocksource 差异；若同一 VM 的 Python/C++ 不一致，
先排查 API/库和采样边界；若两组都增长，不能归因于 KVM，需要再对照历史 QEMU。

Host 取本次启动时打印的 `Clock probe session` 目录，分析共享 Guest JSON 与同会话 QEMU 日志：

```bash
session=/data/lihan/cemu-clock-probe/clock.本次目录
sudo python3 /home/lihan/cemu/CEMU/tests/cemu/instattention/experiments/freeze_clock_probe.py analyze \
  --guest-json "$session/guest-logs/freeze-clock-ioctl.json" \
  --host-log "$session/qemu.log"
```

如果 /root/cemu-logs 未挂载到 logsrc，需先手工复制 JSON，不能拿另一会话的日志分析。
分析器检查随机 probe ID、QEMU PID/controller/program、完整的功能执行和调度完成记录，
缺少诊断开关或事件重复时直接报错。每个时钟只对自己的时间戳求差，不跨时钟相减：

- Guest：Python perf_counter_ns() 的 Execute 前后差。
- Host：功能执行包围区间，以及算子回传的内部等待区间。
- CEMU virtual：冻结内前后差，以及设备收到请求到调度器完成的区间；后者不是 Guest 返回边界。
- 同时列出请求 runtime、扣除 freeze_entry 后的有效预算和调度预算，避免误读零预算。

若 Host 功能耗时增加约1秒，Guest Execute也增加约1秒，而冻结内虚拟差接近零、设备虚拟
区间基本不变，结果为 `guest_clock_includes_functional_freeze`；若 Guest 基本不变，
为 `guest_clock_excludes_functional_freeze`。噪声大或设备虚拟区间明显变化返回
`inconclusive`，不是通过。该诊断不校准 InstAttention 性能，也不证明 GPU/freezing 异步精度。

Host 自检（不需要 Guest，不证明真实 KVM clock 行为）：

```bash
python3 tests/cemu/instattention/python/test_freeze_clock_probe.py
```

## 按模式查接口

| 内容 | 文档 |
| --- | --- |
| Dense分阶段接口 | [ATTENTION_PHASES](ATTENTION_PHASES.md) |
| 旧三文件SparF、ratio与观测 | [SPARF_WORKFLOW](SPARF_WORKFLOW.md) |
| 初版自治v3接口 | [SPARF_AUTONOMOUS](SPARF_AUTONOMOUS.md) |
| 旧buffer和spans/compact写回 | [SPARF_BUFFERING](SPARF_BUFFERING.md)、[SPARF_DEVICE_WRITEBACK](SPARF_DEVICE_WRITEBACK.md) |
| 分组页目录、buffer、Attention | [SPARF_GROUP_DEVICE](SPARF_GROUP_DEVICE.md)、[SPARF_GROUP_BUFFER](SPARF_GROUP_BUFFER.md)、[SPARF_GROUP_ATTENTION](SPARF_GROUP_ATTENTION.md) |
| handoff诊断 | [SPARF_HANDOFF_DIAGNOSTICS](SPARF_HANDOFF_DIAGNOSTICS.md) |
| grouped运行手册 | [RUNBOOK](INSTATTENTION_GROUPED_RUNBOOK_2026-09-16.md) |

这些文档按功能阶段编写，不同模式的MR、文件和并发限制不能混用。
最新跨阶段状态以开发历史的保存基线和FTL工作记录为入口；具体参数以
[Makefile](../Makefile)、脚本help和运行时实际配置为准。
旧进展/交接文档的原路径和Git恢复方法已列入开发历史，避免重新创建重复状态文档。
