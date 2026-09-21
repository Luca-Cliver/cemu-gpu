# G4：分组目录与设备尾部接入 SparF Attention

状态：设备读取入口、原自主调度器、Python 小测试适配器已连接；Host 测试通过，用户已反馈 Guest 新测试通过。
本次反馈未附完整 trace，因此不据此宣称重叠比例或模拟时间已校准。最新交接见 [INSTATTENTION_HANDOFF_2026-09-15.md](INSTATTENTION_HANDOFF_2026-09-15.md)。
没有切换 OPT runner，也没有修改 freezing、计算时间补偿或 FTL 完成时间等待。

## 数据流

```text
Guest APPEND K/V → G3 的设备 FDM 尾部 → 满组后台写回
Guest 提交 Q、value_mean、cookie 和 stream_base（一次 v3 Attention）
  → 设备 Top-r
  → 新目录 channel-K 已提交页 + FDM 未提交尾部 → 近似分数、Top-k
  → 新目录 token-K 已提交页 + FDM 尾部 → 选中 K
  → 发起选中 V 的真实 FTL 读取，同时复制 V 尾部
  → 执行 QK，等待其原有目标计算完成
  → 等待 V 物理读取和目标时刻
  → PV → 输出
```

新路径不通过 Guest READ 命令把历史 K/V 搬回去，再提交 Attention；设备直接查询 G3 目录。
也没有从新存储同步读完 V 才调用旧 Attention：它生成读取计划，交给原来的
`sparf_start_read / sparf_finish_read`，保留 V 在 QK 前提交、PV 等待两者完成的结构。
未提交尾部在生成计划时复制到 Attention 的目的 MR，随后后台可继续发布和回收尾部。
同页选择复用原页缓存去重机制；页池 extent 映射在首次绑定时构建一次，不每次重新扫描全池。

当前必须先 APPEND 当前 token，再提交 Attention；设备不隐式接收新 K/V。
`value_mean` 仍由上层维护并送入 MR4，尚未迁移到设备。这些边界不能说成已经完全复现论文。

## 接口与资源

使用原 11-MR Attention 工作区和独立 G3 的 3-MR 存储工作区，共两套 MRS。
Attention 的 MR5/6 暂保留 ABI 容量但新路径不写入、不读取其 current K/V；当前 token 也从新存储取。
没有新增一份 Prefill/Decode runner。`CemuSparfDevice(grouped_store=..., autonomous=True)`
复用原算子和自主调度过程；未传 grouped_store 时旧行为不变。

workflow version 仍为 3，新增 flags bit 4（GROUPED），不能与 BUFFERED 或 Guest staged 模式混用。
固定 112 B header 后追加 `(uint64 cookie, uint64 stream_base)`，不附旧三文件 extent 表，旧 stride 字段为零。
stream_base 表示当前 Attention 层第一个 KV head 的 stream，映射为：

`stream = stream_base + batch_index * num_kv_heads + kv_head`

Python 适配器默认按 `(layer, batch, kv_head)` 展平，并支持整体 grouped_stream_base 偏移。
GQA 的 q_head → kv_head 分组规则保持不变，读取结果保留选择索引的顺序和重复项。

暂限一个 controller/namespace 内、4 KiB 页、FP16/FP32、`nr_cu=1` 的原 v3 自主路径。
一个 namespace 的 Attention 父任务仍沿用原单在途限制，本次没有增加多父任务并行。
后台分组写回不占这个 Attention 门控，继续使用独立线程和真实 FTL 写请求。

## 生命周期和错误

接收 Attention 时校验 cookie、维度、dtype、stream 边界，并要求所涉及 stream 的 accepted
恰好等于 valid_tokens。获取会话/stream 读取引用，直到全部阶段完成或错误清理结束。
在此期间拒绝对相同 stream 追加、拒绝会话 DRAIN/CLOSE 和 owner unload；其他 stream 可追加。
后台写回不受读取引用阻止，仍可推进 committed；读取计划只使用已发布页和锁内复制的尾部。
引用不是长期持有互斥锁，不会锁着尾部等待整个 Attention。

实际 IO/算子错误会返回失败，并释放 Attention 引用，防止之后无法追加或关闭。
后台写入错误仍由 G3 poison 状态阻止后续访问。终结 DRAIN 后允许只读 Attention，但不再允许 APPEND。

## 测试

Host 新测试编译生产 G3 缓存、分组 Attention 读取桥接和 SparF workflow，运行真实 CPU SparF kernel，
使用确定性假 FTL 与 NumPy SparF 作对照。覆盖：

- 纯尾部读取（Guest current-K/V 特意填 NaN，验证没有偷偷依赖旧输入）；
- 后台写完成被门控时仍能 Attention；
- token/channel 组边界、Flash＋尾部、环形复用、终结后的 Flash 读取；
- ratio=1/8、GQA；旧 kernel 检查器断言 QK 时 V 在途、PV 时 V 已完成；
- cookie/长度错误、读取/算子失败后释放引用、关闭和追加冲突；
- Python 适配器只写 Q 和 mean，不解析旧 KV 文件映射。

```bash
cd /home/lihan/cemu/CEMU/tests/cemu
PYTHONPATH=/tmp/cemu-opt-integration-build make instattention-sparf-group-attention-test
```

新 Guest 测试需要正常重启使用新编译的 QEMU，挂载后执行：

```bash
source /root/cemu-gpu-env/bin/activate
cd /root/CEMU/tests/cemu
make instattention-sparf-group-attention-device-test \
  NVM_TEST_DIR=/mnt/nvme0 \
  FDM_TEST_DIR=/mnt/fdm0 \
  CSD_TARGET=cuda
```

使用自己创建的 8 MiB NVM 池；2 层、batch=2、Q heads=4、KV heads=2、D=128。
在 15/16/253/256/259/531 tokens 上分别测试 ratio=1/8，共 24 次 Attention，再终结排空。
此命令不加载 OPT；测试的三个计算预算是人为设置的调度诊断值，不是论文校准值。
数值 PASS 不代表实际重叠比例或模拟精度通过；运行时若启用原 autonomous JSONL trace，
继续用 `experiments/analyze_sparf_autonomous.py` 检查目标依赖及 QK/V 重叠。

后续：先核定真实模型的页池/stream/FDM 容量，再将 OPT 的写入、value_mean 生命周期和最终 drain 接到这套接口；
另行验证多 microbatch 流水、实际重叠和资源容量。跨 head 拼块及物理 Flash 放置仍未实现。
