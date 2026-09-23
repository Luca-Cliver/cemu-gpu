# G3：设备尾部缓存、分布局后台发布与终结排空

后续的设备 Attention 接入已单独实现，见 [SPARF_GROUP_ATTENTION.md](SPARF_GROUP_ATTENTION.md)。
下文包含 G3 存储逻辑及后续跨 head 批量写回。OPT runner 已通过 `SPARF_GROUPED=1` 可选接入。

状态：生产设备逻辑和 QEMU 入口已实现，Host 回归测试通过；用户提供的 Guest 集成日志也已通过。
后续 Attention 桥接及 OPT 接入的基线见 [开发历史](../../../docs/cemu/DEVELOPMENT_HISTORY.md)，新 FTL 待办见 [FTL 工作记录](FTL_WORKLOG.md)。
此路径接在 G2 分组页目录之后，不替换旧 spans/compact、非 grouped SparF 路径，也不修改 freezing。
没有宣称已经完整复现 InstAttention，更不以运行时间接近论文作为本阶段通过条件。

## 1. 命令与数据流

```text
Guest REGISTER_BUFFER：独占 NVM 池、页几何、stream 数、三个不重叠 MR
  MR0：Guest 原始 K/V 输入或索引读取结果
  MR1：状态
  MR2：有界设备 FDM 环形尾部 + 有界后台批写暂存

Prefill / Decode APPEND_BATCH：一次提交多个连续 stream
  → MR0 的 [2, T, streams, D] 原始字节，先 K 后 V
  → 设备复制到 MR2 环形尾部，推进 accepted
  → 有空间即返回；空间不足等待后台释放，不覆盖尚未持久化的尾部

设备后台线程（无需 Guest 再次读取来触发）
  → accepted 更新时将就绪 (stream, layout) 放入去重、有界队列
  → 按配置页数上限合批取出 token-K、channel-K、token-V 组
  → MR2 暂存打包完整页，channel-K 在设备内转置
  → 原内部 NVM/FTL 写路径：真实写入 + 等待目标完成时刻
  → 发布目录，单独推进该布局 committed，唤醒受背压的 APPEND

逻辑 READ_TOKENS / READ_CHANNELS
  → 已提交前缀：查页目录、去重整页、真实 FTL 读、页内筛选
  → 尚未提交尾部：直接从 MR2 拷贝
  → 合并为原索引顺序；写回正在进行的组仍由尾部提供

终结 DRAIN
  → 停止接受新 APPEND；补零写完最后的部分页
  → 等待所有布局 committed == accepted 后 sealed
  → 可继续读取；再次 DRAIN 不产生写 IO
```

这里的 DRAIN 是整个会话的终结操作，**不能在 Prefill 后调用再继续 Decode**。
普通进度观察用 STATUS；不能把未满组提前发布后继续覆盖同一物理页。
CLOSE 自动执行终结排空；后台错误会让 DRAIN/CLOSE 报错，CLOSE 仍清理会话及引用。

## 2. 三条边界、内存与所有权

P=4096、D=128、FP16、C=8 时，token-K/V 为 16 tokens/组，channel-K 为 256 tokens/组。
例如 accepted=253，后台完成满组后可出现：token-K=240、channel-K=0、token-V=240。
这不表示 K 丢失：channel-K 所需原始 K 仍保留在 FDM。

每个 stream 分配 `2*Gc` 个原始 K 和 `2*Gt` 个原始 V，另有所有 stream 共用的 `W*P` 写暂存：

`FDM_tail_bytes = streams * (2*Gc + 2*Gt) * D * E + W * P`

`W` 是写批页数上限，至少容纳一个完整 channel group（`Nc=D/C` 页）。
旧调用不传参数时 `W=Nc`；OPT 与 Guest buffer 测试默认 `W=256`。
2 streams、W=256 时 tail+scratch=1327104 B，另有 MR0=1 MiB、MR1=4 KiB。
K 的回收边界是 `min(token-K committed, channel-K committed)`，V 使用自己的 committed。
这不是逐 token 分配堆内存，也不是把全部历史 KV 常驻 DRAM。
目录、页选择描述和前台读 IO 镜像仍使用有上限的 QEMU Host 内存，尚未全计入目标 DRAM 预算。

G3 会话绑定 program、cookie 和同一个 MRS。注册成功后额外固定 MRS，直到 CLOSE 或 owner unload
排空并等待后台线程退出才释放引用，避免后台继续使用已删除的 MR2。MR0/1/2 必须两两不重叠。
Guest 适配器用锁串行保护共享 MR0 的整个事务；后台任务不依赖 Guest 请求对象的寿命。

## 3. 并发范围与时间语义

一个会话有一个后台写线程，从就绪队列合批；同一批包含多个 stream/layout 的完整页，
一个批次完成后才提交下一批。普通追加不再扫描所有 stream，DRAIN 才遍历全部 stream。
实际 FTL IO/目标时刻等待时不持有尾部锁，因此前台 APPEND 或 READ 可与后台写重叠。
前台 READ 本身仍串行，读取期间不发布后台新目录，保留尾部直到读结束。
已支持跨 head 合批，但页数上限不等于物理 Flash 块所有权或多通道 PPA 放置策略。
当前就绪页不足时直接发送较小批次，不强制等待整块凑满；这避免小请求和背压死锁，
也意味着尚不能声称与论文逐块追加、无前台 GC 的行为完全一致。

后台写不执行算子、不申请 CU、不冻结 VM。它复用旧 `sparf_wb_transfer` 拆出的公共传输函数，
保持原实际 IO、FTL 排队和 `clock_ns() < expire_time` 的等待，不删模拟耗时，不跳过 NVM。
终结 DRAIN 计入测试完整完成边界；APPEND 返回只表示数据接收，不表示已经持久化。

正常 DRAIN/CLOSE 仍排空全部已接受数据并等待目标时间。底层 `sparf_group_free` 则只负责
取消和释放，不再隐式 drain：它停止新写批次、取消未来目标时间的等待，收回已提交 FTL
请求的所有权并 join 后才释放 scratch/MRS 引用；取消批次不发布 committed 前缀。
只对 grouped 后台请求设置取消标志；尚在 FTL 限速等待队列的取消请求可不等未来时钟，
由 FTL 回送取消完成。普通请求没有此标志，保持原有调度条件。
直接 UNLOAD 未排空会话属于放弃未发布 KV，会输出警告，不能当作持久化成功。
取消不强行中断尚未返回的真实 FTL IO，也不修改原生 CEMU 线程退出或 VM freezing。

## 4. ABI 扩展

保留 G2 版本、64 B header 和操作 1～5。新增操作：

| 命令 | 参数与约束 |
| --- | --- |
| REGISTER_BUFFER=6 | 几何及 extent 数组同 REGISTER，header.stream 是 stream 数（1～1048576），要求 3 MR；argument 是写批页数上限，0 使用旧 Nc 页默认值 |
| APPEND=7 | stream、position 必须等于 accepted、count>0、argument 为 MR0 偏移，先 K 后 V |
| DRAIN=8 | 除 version/cookie/op 外全零，终结所有 stream |
| STATUS=9 | stream 选择进度，其他普通参数为零，不隐式排空 |
| APPEND_BATCH=10 | stream 为首 stream，argument 为连续 stream 数；count 为 token 数，position 必须等于每个 stream 的 accepted；MR0 为 [2,T,streams,D] |
| STATS=11 | 除 version/cookie/op 外全零；MR0 返回 5 个 uint64 统计，不隐式排空 |

STATS 返回 `write_batches, max_write_pages, append_commands, appended_streams, appended_tokens`。
写统计只累计成功完成的批次；追加统计只累计成功命令，故失败后不可用它替代 accepted 进度。
Python 先按 token tile，再按连续 head/stream 切分，避免一个 head 的整个 Prefill 占满尾部后才提交其他 head。

buffered 会话不接受 PUT。stream ID 必须在 `[0, stream_count)`。
MR1 先保留 G2 的 12 个 uint64，再追加 8 个 uint64：
`accepted, token_k_committed, channel_k_committed, token_v_committed, sealed, failed, write_bytes, buffer_bytes`。
后三项写入进度由 STATUS/其他命令快照获取；异步后台不会并发覆盖 Guest 正在读的状态 MR。
旧结果中的 IO 时间和字节只代表当前前台操作，不是后台累计耗时。

池耗尽、实际 IO 失败会 poison 会话。已接受的部分数据不承诺回滚；STATUS 返回失败标志，
后续读写被拒绝，阻塞 APPEND 和 DRAIN 被唤醒报错，已分配页不复用。
不提供断电恢复、reset/热拔插或 Guest 崩溃后的自动回收。

## 5. 代码和测试

- `hw/femu/csd/sparf_group_buffer.h`：后台线程、APPEND 背压、分布局提交、排空/生命周期。
- `hw/femu/csd/sparf_group.h`：目录与 Flash/DRAM 混合读取，G2 仍可用。
- `hw/femu/csd/compute.c`：命令入口、真实 FTL 回调、MR2 coherency、跨命令 MRS 引用。
- `python/cemu_flexgen/sparf_group_device.py`：`buffer_streams`、APPEND、STATUS、DRAIN。

Host：

```bash
cd /home/lihan/cemu/CEMU/tests/cemu
make instattention-sparf-group-buffer-test
```

其中新测试编译生产 C 核心并使用可控制的假 FTL：阻塞目标写完成时，确认读取仍取到有效尾部，
而 committed 尚未提前推进；缓冲满时 APPEND 等待，放行后恢复。也测多 stream、环形复用、
15/16、255/256 边界、partial drain、幂等排空、FP32、非法请求、空间不足、IO 失败。
这证明数据和依赖逻辑，不替代 Guest 驱动集成测试或论文时间校准。

Host 正常关闭旧 VM，重新使用编译后的 QEMU 启动；Guest 挂载后：

```bash
source /root/cemu-gpu-env/bin/activate
cd /root/CEMU/tests/cemu
make instattention-sparf-group-buffer-device-test \
  NVM_TEST_DIR=/mnt/nvme0 \
  FDM_TEST_DIR=/mnt/fdm0
```

新 Guest 测试使用独占 2 MiB 池和 2 streams，先 253-token Prefill，再跨 256 边界追加，
最终共 531 tokens/stream 触发环形复用和部分组 DRAIN。校验最终全部 K/V、全部 channel-K，
预期后台写 950272 B；不下载/加载 OPT，不执行 Attention 算子。
Guest 测试不人为暂停 FTL，因而不把每次运行都声称为实际重叠测量。并发能力由 Host 门控测试验证，
之后接 Attention 时再用设备 trace 验证真实流水。

新增 Host 回归覆盖跨 128 heads 的 256 页合批、写完成前尾部读取、失败不发布、
FP16/FP32 批量输入、MR0 token/stream 双维切分和统计 ABI。
这些使用生产 C 逻辑及假 FTL 的测试不证明实机加速，也不证明多请求/双 Attention engine 调度已完成。
