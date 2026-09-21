# KV 专用逻辑地址接入前审查（2026-09-21）

## 结论与范围

本次审查当前 grouped transport → 内部请求 → FEMU FTL 的地址与数据路径，
没有实现新的 KV 地址 ABI，没有修改 freezing、算子、布局或目标时延。
当前实现可以作为实验性基线，但不能认定为没有问题或已实现论文专用 FTL。
初审发现下面三个问题；随后已修复，修复范围及验证见下一节，尚未执行 git add、commit 或 push。
这不是对全部历史未提交变更的完整安全审计。

## 审查后修复

- 连续 sector 段的长度上限按实际 NAND 页边界向下取整；4KiB 页、页对齐起点时
  上限为 65528 sector。读写共用此拆分逻辑，不改变数据覆盖范围。
  Host 回归覆盖 32/64MiB、4/64KiB 页、碎片及非页对齐 sector 起点；
  这仍不是 Guest 大传输的 FTL 实测。
- plan 排序时保留本批 pool ordinal，tag 构建和刷新不再逐页扫描整个 pool。
  复杂度为批内排序 O(batch_pages log batch_pages)，刷新 O(batch_pages)，
  额外索引仅每个选中页一个 guint，不建立整个 pool 的新哈希表。
  回归包含顺序颠倒的 extents、6GiB pool 末端页、写入后 tag 刷新和后台传输 tag 校验。
- trace 增加 controller、nsid、session cookie 和 plane；初始化改为 GLib once，
  移除无同步记录计数器，仍按行输出。cookie 的逐 LPN 数组仅开启 trace 时分配。
  分析器区分全部读、KV 读、未标记读，按设备/namespace/session 分组；
  物理页重复按设备/namespace 统计，不因为 session 不同而漏掉同设备的复用。
  旧 trace 仍能分析，但明确警告缺少完整身份，不能追溯补齐身份。
- 未改 Guest ABI、Linux 内核、算子、freezing、目标 runtime 或 NAND 时延配置。
  新增 cookie 仅用于内部页 tag/trace，不等于已经实现任务地址或 generation/reclaim。

需要重新启动 QEMU 进程加载新设备代码，不需要重启宿主机或重新编译 Guest 内核。
Guest 仍需重跑 grouped I/O、buffer、Attention 三个设备测试，并用新 trace 分析；
通过后再跑相同参数的 OPT 短流程。Host 单测通过不能替代设备集成验证或性能实测。

## 初审问题（修复前）

### 1. 大传输 extent 切分不保持页对齐

位置：`hw/femu/csd/compute.c:2543`，`hw/femu/bbssd/ftl.c:1076`。

`sparf_group_transfer()` 最长合并 65535 个连续 512B sector，然后切成下一段。
4KiB 页包含 8 个 sector，65535 不是 8 的倍数。对一个连续 32MiB 的 plan，
切分为 `(slba=0, nlb=65535)` 和 `(slba=65535, nlb=1)`。
FTL 对每段独立按页建模，LPN 8191 因此被处理两次：8192 个唯一页产生
8193 次页操作。写入还会把第一次分配的边界页立即失效并重新分配。

这是按源码切分规则做的确定性算术复现，未运行 32MiB Guest 设备实测。
当前默认 256 页批次不会触发该边界，但接口允许更大的批次（最高 64MiB）。
模拟出的额外页操作并非论文行为，不能当作设备性能差异。
修复应保持 NAND 页边界，例如 4KiB 配置使用不超过 65528 sector 的段；
不能直接改为 65536，因为现有消费端把解码后的 sector 数保存在 uint16_t。
需要同时覆盖 grouped 读与写的分段边界回归。

### 2. 页 tag 构造存在重复全表扫描

位置：`hw/femu/csd/sparf_group.h:141`、`:160`，
调用位置：`hw/femu/csd/sparf_group_buffer.h:90`、`:133`。

plan 对 sector 排序后，每个页都从 `session->pages[0]` 开始寻找其 ordinal；
随后 refresh tags 再做一遍。查找次数随所选页在 pool 中的位置增长，
最坏单批成本为 O(batch_pages × pool_pages)。后台 worker 在 buffer 锁内
执行这些查找，可能延迟 APPEND 和 Attention 的分组查询。
即使不开 FTL trace，这条 tag 构造路径也会执行。

6GiB/4KiB pool 有 1572864 个页，不能拿小 fixture 的耗时排除该问题。
尚未量化其对既有 OPT wall time 的贡献，不能断言它解释全部变慢。
建议排序前保留 ordinal/tag，或注册时建立 sector→ordinal 索引，
不改变页内容、发布顺序、NAND 页数或模拟时延。

### 3. FTL trace 不是纯 KV 统计，也不具备完整身份隔离

位置：`hw/femu/bbssd/ftl.c:867`、`:921`，
`experiments/analyze_sparf_ftl_trace.py:21`。

写 trace 仅输出有效 KV tag；读 trace 则输出所有 mapped 页，未标记的读为 kind=255。
分析器把所有读纳入 read_wait_ns。现有
`/home/lihan/cemu/cemu-logs/sparf-ftl-kv-group.trace` 实际分布为：

| read kind | 次数 |
| --- | ---: |
| token-K (0) | 171 |
| channel-K (1) | 216 |
| token-V (2) | 166 |
| 未标记 (255) | 760 |

1313 次读中只有 553 次带有效 KV kind。既有 read_wait 汇总不能直接称为 KV 等待。
逐页等待时间之和也不是一次 Attention 的端到端读取时延。

trace 文件使用进程级静态初始化且没有同步；多 FTL 线程首次调用存在竞争。
记录不含 controller/namespace/session 标识，不同设备的相同 PPA 或多个会话的
相同 stream/group 会被分析器合并。physical_page_reuse=0 不是 GC 正确性的证明；
擦除后的合法复用也不能自动当作错误。
建议加入身份、分开 tagged/untagged 读统计，并使文件初始化线程安全。
单 CSD 小例的数值 PASS 不受上述统计口径问题否定。

## 当前已核实的地址及所有权边界

1. Guest 在自己的 NVM 文件上分配 pool、初始化 unwritten extents、fsync 后注册
   extent：`python/cemu_flexgen/sparf_group_device.py:78`。
   后续 grouped APPEND/读写由设备管理，不能再描述为每次 KV 写回都由 Guest pwrite 完成。
2. stream 目前是 microbatch 基址加 `(layer * batch_size + batch) * kv_heads + head`：
   `python/flexgen_adapter/sparf_attention.py:257`。语义由 Guest 持有，FTL 只收到不透明 stream。
3. session cookie 在 grouped 控制和 Attention bind 中检查，目录键是 session 内的
   `(stream, kind, group)`。当前每个 ComputeNamespace 仅一个 grouped session。
   修复后 `include/hw/femu/kv_page.h` 的页 tag 携带 cookie，但不包含 task 几何或 generation。
4. grouped 目录保存 pool ordinal，而非 PPA；读路径通过注册 extents 生成 LBA。
   `maptbl/rmap` 仍是物理映射的唯一权威，覆盖和 GC 都由该 namespace 的 FTL 线程处理。
   因而当前没有第二张独立 PPA 表需要同步，但也没有专用 KV→PPA 映射。
5. 后台写回在 transfer 成功且目标期限到达之后才发布目录及 committed 前缀：
   `hw/femu/csd/sparf_group_buffer.h:139`。发布前数据仍从 tail 读取。
6. 数据后端按 LBA 偏移读写，PPA 用于 NAND 资源/时延和页状态建模：
   `hw/femu/bbssd/ftl.c:1087`。未来专用寻址必须返回匹配的 backend offset 和 PPA，
   不得仅换计时映射而不验证实际内容。
7. CLOSE 会 drain 并销毁 session 目录；不会通知 FTL 按任务失效/擦除页面。
   `hw/femu/bbssd/ftl.c:1327` 的 DSM 分支也只置时延为零。
   这是后续任务生命周期设计缺口，不是现有 CLOSE 已实现 task reclaim 的证据。
8. GC 搬迁按 LPN 保存的 kind 选择写指针，但小例未强制触发 GC、容量耗尽、
   多任务并行或重启恢复。kind lane 不能等同于任务块归属。

## 新地址接口的接入条件

- 先确定 namespace/session-generation/task 的身份边界，以及 batch/layer/head 的
  注册几何；不得将会话局部 stream 直接作为全局 KV key。
- 第一阶段保留一种物理映射权威。可以让专用 KV 索引引用 FTL 拥有的稳定 page record，
  record 同时携带 backend offset 与当前 PPA；不要再建一张可独立漂移的 PPA 副本。
- 普通 NVM 写、覆盖、GC、CLOSE/reclaim 必须定义同一 record 的更新/失效规则。
  KV pool 在会话期间不得被文件系统或其他任务复用。
- 新 ABI 用版本号区分，保留旧路径作回归；32-bit 编码的位宽不能从 Figure 10 的
  Length 数字直接推断。内部宽结构体只代表实现选择，不能称为论文原始二进制格式。
- 测试顺序：编解码和边界 → 同坐标跨 session 隔离 → 双视图读回 → 覆盖/GC 更新 →
  失败未发布/旧 generation 拒绝 → 关闭后重注册 → SparF 数值 → OPT。
- 冻结机制保持不变，普通 NVMe 数据路径不应被实验性 KV 地址模式隐式改变。

## 初审验证（修复前）

Host：

```bash
PYTHONPATH=/tmp/cemu-grouped-review-F19FUK:tests/cemu/instattention/python \
  python3 -m unittest discover -s tests/cemu/instattention/python -p 'test_*.py' -q
ninja -C build qemu-system-x86_64
bash -n femu-scripts/run-csd.sh femu-scripts/run-csd-opt13b-sparf.sh
git diff --check
```

272 个 unittest 通过，QEMU 构建及 shell 语法检查通过。CEMU 与 linux-cemu 的
diff 空白检查通过。临时扩展目录是本机已有 Python 3.10 模块，不能复制为 Guest
Python 3.13 模块。unittest discovery 不执行 Guest 脚本 main；本次没有重启 QEMU、
重跑 Guest/GPU/OPT、强制 GC，亦未编译 linux-cemu 内核。测试通过不覆盖上述发现。

## 修复后验证

同一 Host discovery 命令复测 280 项全部通过（新增 8 项）；后台 buffer C harness
也检查每次传输的 tag、SLBA 和 session cookie。QEMU 重新构建通过，保留已有的
indirect_main 未使用变量警告；两个仓库的 diff 空白检查及启动脚本语法检查通过。
日志：`/tmp/cemu-ftl-fix-tests-20260921.log`、
`/tmp/cemu-ftl-fix-build-20260921.log`。

重新分析旧 trace：全部 1313 次读等待合计 149073ns，其中 553 次 KV 读为
60133ns，760 次未标记读为 88940ns。1737 条旧记录均缺少完整身份，分析器正确警告；
不能用旧 trace 验证新增设备/session 身份，也不能据此声称端到端已加速。

仍未运行 Guest 集成测试、GPU 数值验证、真实多 CSD trace、OPT 或 GC 压力测试。
安全结束旧 VM 并用新 QEMU 启动后，在 Guest 激活原 Python 环境、挂载 NVM/FDM，
按顺序执行：

```bash
cd /root/CEMU/tests/cemu
make instattention-sparf-group-device-test instattention-sparf-group-buffer-device-test \
  NVM_TEST_DIR=/mnt/nvme0 FDM_TEST_DIR=/mnt/fdm0 \
  PROGRAM=./build/sparf_attention.so
make instattention-sparf-group-attention-device-test \
  NVM_TEST_DIR=/mnt/nvme0 FDM_TEST_DIR=/mnt/fdm0 \
  CSD_TARGET=cuda PROGRAM=./build/sparf_attention_devptr.so
```

这些小测试不覆盖 32/64MiB 设备传输或 GC 压力；通过后再复测原 OPT 短流程。
开启新 trace 时需在启动前设置原有的 `CEMU_SPARF_FTL_TRACE`、
`CEMU_SPARF_FTL_TRACE_READS=1` 和 `CEMU_SPARF_FTL_PLACEMENT=kv_group`，
并使用新的 trace 路径，避免覆盖之前的实验记录。

## 保存基线的范围

建议问题修复后保存为 grouped/autonomous + experimental kv_group 基线，而非
InstAttention 专用 FTL 完成版。CEMU 未提交内容不止 ftl.c，还包括新头文件、算子、
Python 适配层、测试、配置及文档；必须成套保存，不能只提交 git diff 可见的 tracked 文件。
排除构建目录、模型、NVM 镜像、运行日志、CSV、临时 config dump 和无关调研文档。

linux-cemu 是独立仓库，另有 `drivers/nvme/host/cemu.c` 与 `fs/fdmfs/file.c`
两处变更；应单独审查/提交，记录匹配版本，不能以 CEMU push 代替内核保存。
本次未暂存或推送任何文件；应在修复的设备回归完成后确认保存范围。
