# G2：设备分组页目录、完整页写入与索引读取

本文记录保留的 G2 完整组 PUT 路径。新的 G3 自动 APPEND、FDM 尾部与后台写回见
[SPARF_GROUP_BUFFER.md](SPARF_GROUP_BUFFER.md)；G3 仍未接入 OPT/Attention runner。

状态：已接入 QEMU 命令分发与真实 FTL IO；提供 Host C 单元测试和 Guest 设备测试。
**不是只有 Python 布局参考，也不是已经完成整个 InstAttention/OPT 接入。Guest 测试仍需在新 QEMU 中运行。**

REGISTER 集成修正：最初的命名空间入口错误地检查了 `NvmeNamespace.impl`，但当前配置解析仅初始化 `csi/ops`，
导致正常 NVM 在进入页目录逻辑前返回 `NVME_INVALID_FIELD`（status 2）。已改为检查实际注册的 NVM command set、
参数及 private 状态，仍保留 sector/page 几何校验。Host 页目录测试不覆盖该 QEMU 入口；修正后必须用新二进制重启并重跑 Guest 测试。

## 1. 本次实际改变的路径

```text
Guest REGISTER：一次注册独占、已初始化的 NVM 文件 extent 池和页几何
  → QEMU 保存池及设备拥有的逻辑组目录

Guest PUT：MR0 放原始 [tokens, head_dim] 数据，发送固定 64 B 元数据
  → 设备判断 token-K / channel-K / token-V 布局
  → 从页池游标分配尚未用于 KV 的目的页
  → 设备打包完整页（channel-K 在这里转置，不在 Guest 展开每个 channel 的描述）
  → 复用内部 FTL 写请求，等待功能完成和目标完成
  → 发布 (stream, layout, token_group) → pool_page 映射

Guest READ_TOKENS / READ_CHANNELS：发送逻辑索引
  → 设备查目录、选择并去重整页
  → 真实 FTL 读取 → 等待目标完成 → 页内筛选/恢复索引顺序 → MR0 返回数据

Guest CLOSE / program unload：丢弃设备目录；调用方随后删除自己的临时文件
```

代码入口：

- `include/hw/femu/sparf_group_abi.h`：独立 opcode、版本、命令和结果结构。
- `hw/femu/csd/sparf_group.h`：生产设备逻辑，注册、分配、页目录、打包、稀疏读取。
- `hw/femu/csd/compute.c`：`program_execute → sparf_submit_writeback → sparf_writeback_worker → sparf_group_execute`。
- `python/cemu_flexgen/sparf_group_device.py`：Guest 测试适配器，不参与页目录构造。

token-K/V 每次提交一页，channel-K 每次提交一个 token 窗口覆盖的全部 channel 页。
例如 4 KiB、FP16、D=128、C=8：token 组 16 tokens，channel 组 256 tokens，共 16 页。
Guest PUT 是原始数据和一条几何化命令，不再传每个 channel 对应的物理 span。
这仍是一组/一次命令，不是设备自动凑组；跨 head 拼块的提交策略将在下一批实现。

## 2. 实际 IO 和发布约束

完整组 PUT 只写新分配的位置，不读取旧页：不是在旧散写路径跳过必要 RMW，而是新路径完整覆盖自己的页。
整页读取后才筛选选中项；同一请求重复索引和同页索引只发一次页读取，但输出保留重复项和用户顺序。
原 token/channel 文件布局、spans/compact 写回和原 SparF Attention 均未切换。

继承 `sparf_wb_transfer` 的内部请求、FTL 延迟和完成等待；同 controller、不额外计外部 PCIe。
不调用 Attention 库函数、不占模拟 CU、不修改 freezing。注册/读取/关闭不需要同步未使用的 MR0 输入镜像；
PUT 读取 MR0 前仍走现有 host coherency，读结果写 MR0 后标记相应范围 host dirty。

功能 IO 返回成功且目标时刻已到，才发布目录。IO 失败会 poison 会话：不发布新组、不再接受读写，只允许关闭。
失败写可能已部分到达后端，已分配位置不复用；不承诺回滚/掉电恢复。池空间不足或非法参数则在 IO 前拒绝，不破坏已有目录。

页目录、位置信息不由 Guest 修改。每 namespace 暂时只允许一个分组会话，绑定 program index 和随机 cookie，
其他 program 不得访问它；同一会话可包含多个逻辑 stream。`stream` 由上层唯一标识 layer/batch/head，本测试使用 0、5。
复用同一个 program 的调用者必须独占会话，不得在多个线程同时修改共用 MR0。
组命令与旧设备写回共用 namespace 单在途门控和已有 worker，本版不声称多组写回并行。

卸载 program 前增加 jobs-running 检查，防止在途命令持有目录时卸载；目录随 owner program 卸载释放。
MRS 仅在每次命令期间固定，不在目录里保存跨命令 FDM 指针。CLOSE 是目录释放，不是有尾组缓存的 DRAIN。

## 3. 边界与限制

- 只接受完整组；非对齐起点、部分组、重复逻辑组全部拒绝。尚无自动 APPEND、独立持久边界、未满组读取、背压和终结 DRAIN。
- 这是独立的存储读写入口，尚未让 `sparf_workflow.h` 的 Attention 路径查新目录，也没有改 OPT runner。
- 数据层只做 FP16/FP32 字节重排；Guest 小测试不执行 Top-r/Top-k 或 CUDA Attention，不能当作算法/全模型 PASS。
- 逻辑页号映射到预注册的文件 LBA 池，真正 PPA 仍由现有 FEMU FTL 决定；未实现论文的物理 head/block 聚合或证明跨 channel 放置一致。
- 文件池首次 fallocate、初始化和 fsync 是过渡容器成本。初始化会真实占用 SSD，并使对应 LBA 在 FTL 中已有映射；
  后续 PUT 不做应用侧旧页读取，但不因此宣称 FTL 无页失效、无 GC 或 NAND WAF=1。
- 仅支持 bbssd NVM namespace 1、512 B sector、page 大小匹配配置、`internal_bandwidth=0`。
- 池最多 65536 页、单次 IO 镜像最多 64 MiB、索引最多 4096 个；目录增长受池容量约束。
  目录/临时 IO 镜像暂在 QEMU Host 内存，本批没有把它们全部纳入目标 FDM/DRAM 预算，不能据此做正式容量实验。
- 只用自己创建的临时文件；REGISTER 不能代替 Guest 对文件独占性、已初始化 extent、所属 namespace 的验证。
  文件锁是协作式的，不能防止外部程序或普通 raw IO 改写这些范围。运行期间不能 truncate、重新分配或并发访问该文件。
- 不支持设备 reset、热拔插、进程崩溃后的自动任务回收；若 owner 未卸载且程序退出，会话仍保留，需要正常释放 owner。

## 4. 协议

`cparam1=0x43454d5553475250`，little-endian，metadata header 固定 64 B。
REGISTER 附加 `(sector, sectors)` extent 数组；READ 附加 uint64 索引数组；PUT 和 CLOSE 没有附加数组。
MR0 为 IO 缓冲，MR1 至少 96 B 结果，两者必须不重叠；测试分别分配一个 IO 文件和 4 KiB 状态文件。

| 操作 | 主要输入 | 结果 |
| --- | --- | --- |
| REGISTER=1 | P/D/C/E、cookie、页对齐无重叠 extent 池 | 创建空设备目录，不发 FTL IO |
| PUT=2 | kind、stream、对齐 token 起点、完整组 token 数、MR0 源偏移 | 转换并追加新页，成功发布；旧组不可覆盖 |
| READ_TOKENS=3 | token-K/V、stream、任意顺序 token 索引 | `[index_count, D]` |
| READ_CHANNELS=4 | channel-K、stream、token 起点/长度、channel 索引 | `[index_count, token_count]` |
| CLOSE=5 | owner/cookie | 释放目录，不修改 Flash 数据 |

结果 12 个 uint64：`read_bytes, write_bytes, read_start, read_end, write_start, write_end, ready, copied_bytes,
version, pages_used, groups, output_bytes`。read/write 是整页 IO 字节，output 是过滤后结果字节；
这是每次命令的诊断，不是整个推理模拟时间。PUT 的 read 字段为零，READ 的 write 字段为零，注册/关闭无 IO。

## 5. 测试与命令

Host C 测试直接编译生产 `sparf_group.h`，以确定性假 FTL 回调检查布局字节、读写请求及失败状态，不伪装为 Guest 集成测试。

```bash
cd /home/lihan/cemu/CEMU/tests/cemu
make instattention-sparf-group-test
```

覆盖：两种 dtype、碎片且物理顺序非单调的 extent、重复页去重、索引顺序、跨组读取、stream 隔离、
非法几何/映射/长度/cookie、容量耗尽、重复组、失败不发布、失败后关闭重建、短输出缓冲拒绝。

QEMU 已构建后，需要正常停止旧 VM、以新二进制启动；不要在正在运行的 QEMU 中期待新命令生效。
Guest 不需要更新内核或 CUDA 算子，仅重建已有 Python bridge。此测试只验证 IO，库作为现有 program 注册载体，不执行库函数。

独立 Guest PUT/GET 测试已退休，下文保留当时的测试规模和结果语义。
现在 Host 使用 `make instattention-sparf-group-test` 检查目录/组页/稀疏读写，
Guest 使用 `make instattention-sparf-group-attention-device-test` 做包含 Attention 的综合验收；
后者需要编译并执行算子，资源用量和下面的历史 IO-only 测试不同。

测试分配 2 MiB 私有 NVM 池和约 68 KiB FDM，不加载 OPT。两个 stream 各 512 tokens，验证全部 token-K/V 和所有 channel-K，
再检查跨页/跨组/重复索引稀疏读取。132 次完整组 PUT 共写 192 页（786432 B），PUT 的旧页读取总量应为 0；
这不包含创建文件池的初始化 IO。132 是测试组织的组命令数，不是论文的调度粒度结论。

通过后再推进自动分布局 buffer、设备 Attention 查目录，以及完整 OPT 接入；不能直接用旧 OPT 命令验证这条新路径。
