# InstAttention 上层审计与 FTL 接口边界

范围：当前 OPT grouped/autonomous 路径。检查公共 Prefill/Decode runner、microbatch KV
writer、SparF adapter、共享 slot scheduler、grouped transport 和 Attention 设备入口。
本轮仅修改 Guest Python 和测试，不改变 freezing、NAND 时延、物理页放置或大实验参数。

## 本轮修复

| 问题 | 修复后的行为 |
| --- | --- |
| grouped adapter 不校验本地 accepted 前缀 | 每层只接受一次非空 Prefill，限制 max_seq_len；Decode 必须从 accepted 位置连续追加一个 token；Attention valid_tokens 必须等于该前缀 |
| 排队后 query 可能仍引用调用者数组 | 提交线程复制 query 和 mean，后续调用者修改不影响请求 |
| 排队/执行期间同一 backend 可以推进 KV | 同一 backend 的 append/Attention 互斥；独立 microbatch 的不同 backend 仍可交错 |
| 已完成但失败的 Future 可能被新请求覆盖 | append、同步 decode、异步 submit 都先检查旧结果；失败后该 backend 禁止继续使用 |
| 多块 APPEND_BATCH 只提交部分数据后发生错误 | MR0 写入、execute、status 任一阶段失败都使共享 grouped transport 进入失败状态；禁止重试和后续 Attention 准备，仍允许 CLOSE 清理 |
| close 清除 mean 后在原会话上重新 open | 已关闭的 grouped backend 不允许重新打开；必须创建新的 backend 和存储会话 |
| Attention 输入形状不匹配可能留下旧 FDM 数据 | 写 FDM 前检查 layer、valid_tokens、query 和 mean 形状 |
| 旧三文件 backend 提前更新 current KV | 写入成功后才更新 current KV |
| Guest 把目标时长 min 当作实际重叠 | 改名为 csd_model.qk_v_overlap_upper_bound_ns；实际重叠使用 Host trace 的区间交集 |
| grouped 模式日志误报三组文件 | 输出实际 pool_bytes/streams |

分块 append 不具有跨命令回滚能力。失败后中止会话是明确的失败处理策略，不能从
Guest 原 mean 自动重试。它不改变正常运行中的页数或写回顺序。

## 已核查且保留的调用约束

- Prefill writer 按 microbatch 切分 batch/head，后台提交后在依赖边界等待。
- Decode 在当前 token 的 QKV 完成后，先 APPEND，再以 token_position + 1 提交 Attention。
- grouped stream = microbatch_base + (layer * batch_size + batch) * kv_heads + head。
- flush 是非终止的可见性边界，未满组继续留在设备尾部；运行结束后由 session owner
  先等待全部 backend，再执行一次全会话 drain。单个 backend 的 drain 不应作为跨
  microbatch 的全局 barrier 使用。
- 共享 scheduler 的多个 slot 是隔离的 FDM 工作空间；当前 autonomous compute worker
  仍串行发出 parent execute。它不等于论文的两个 Attention engine。

## 验证

在 Host 执行：

```bash
cd /home/lihan/cemu/CEMU
PYTHONPATH=/tmp/cemu-grouped-review-F19FUK:tests/cemu/instattention/python \
  python3 -m unittest discover -s tests/cemu/instattention/python -p 'test_*.py' -q
ninja -C build qemu-system-x86_64
git diff --check
```

270 个 unittest 通过，QEMU 增量构建通过，diff 空白检查通过。新增用例覆盖前缀校验、
重复 Prefill、排队后的 query 修改、在途 append 拒绝、完成后的异步错误传播、部分
append 后 mean 不推进，以及 MR0/execute/status 失败后禁止重试但允许关闭。
`PYTHONPATH` 中临时目录是本次可用的 Host 扩展位置，换环境需使用对应 build-guest 路径。

unittest discovery 不执行 Guest 脚本的 main()。本轮没有运行 Guest GPU/NVMe 测试，
也没有重跑 B4/P1024/D16 或完整 OPT logits reference。现有 C harness 数值测试通过
只证明当前实现与仓库参考一致，仓库 NumPy 参考不等于独立官方实现。

## 接专用 FTL 前后的工作边界

论文依据为本地 PDF Algorithm 1、IV-C/IV-D、Figure 7–11；公开上层参考为
`InstAttention/flexgen/flex_opt.py`。其中 store_cache 跳过最后生成步的持久缓存写入，
不代表当前 token 可以从 Attention 中移除。当前 CEMU 以 P 个 Prefill token + D 次
Decode Attention 计量工作量，不能直接把 D 当作上游包含 Prefill 输出的 gen_len。

下一次 Guest 验收先运行现有 grouped Attention 小例（包含 15/16、253/256/259 等组边界
和 GQA），再按原 B4/P1024/D16 配置复跑，保持 640 次 Attention、输出 token、append
计数、逻辑/实际读字节可比。确认接口回归后，专用 FTL 可以沿稳定的 stream/group 目录
接入，先做跨通道放置与 task/block 归属，再做 reclaim/后台擦除。

仍不能标为“完全复现 InstAttention”的部分：

1. SparF 数值需要独立核对选取/并列值/current-token 规则，并完成模型级数值验证。
2. value mean 仍由 Guest 增量计算，GPU/CSD 传输仍有 Host 中转，并非论文 P2PDMA。
3. 双 Attention engine 的跨请求资源仲裁尚未实现。
4. task 感知的物理块放置、reclaim、擦除属于待实现的专用 FTL；fifo/kind 只是页分配
   顺序对照，不能作为论文 FTL 完成的证据。

因此这轮完成的是上层接口和错误处理补缺，不是整套系统行为或模拟精度的最终验收。

## 当前 FTL 原型（可选，不等于论文专用 FTL）

在上述审计之后，CEMU 增加了一个显式的 `CEMU_SPARF_FTL_PLACEMENT=kv_group`
对照模式。它只在启用该环境变量时为 NVM LPN 保存 KV kind，并为 token-K、channel-K、
token-V 各自维护 NAND 写指针；因此三类页不会继续共享同一个 FIFO 物理写入游标。若同时
设置 `CEMU_SPARF_FTL_TRACE`，还会保存 stream/group/page-in-group 元数据，使读回路径能够
输出实际 LPN 到 PPA 的映射和每页等待时间。默认 `fifo` 路径不分配这些 KV 元数据，保持
原有 FTL 的内存开销和行为。

该原型的用途是验证“KV 逻辑分组是否能够传递到真实 FTL 读写路径”，而不是宣称已经
复现论文中的地址编码、块回收或 Flash 控制器调度。应使用相同 workload 分别运行
`fifo` 与 `kv_group`，比较 kind/group 的 PPA 分布、读等待和完整 CSD wall time；只有在
这些对照数据稳定后，才继续实现 task-aware reclaim 或更细的通道/块布局。
