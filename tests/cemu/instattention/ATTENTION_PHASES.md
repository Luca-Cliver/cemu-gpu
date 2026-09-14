# Dense Attention 分阶段验证（第一批）

第一批拆分计算，第二批新增实验性设备侧单请求工作流。旧 Dense、多 chunk
online softmax、OPT runner 和 Indirect 路径保持不变。**尚未实现 SparF。**

## 接口与数据

新增 CPU `attention_phases.so`、CUDA `attention_phases_devptr.so`，均导出
`attention_phase(cemu_args *)`。metadata 的 `phase` 选择阶段：

| phase | MR0（输入） | MR1（输入） | MR2（输出） | 计算 |
| --- | --- | --- | --- | --- |
| 1 | Q | K | P | `P = softmax(scale * QKᵀ)` |
| 2 | P | V | output | `output = PV` |

Q、output 为 `[batch, query_heads, head_dim]`，K/V 的有效数据为
`[tokens, batch, kv_heads, head_dim]`，每 token 按 `token_stride` 字节定位，
stride 必须是 512 的倍数。支持 FP32/FP16 和 MHA/GQA/MQA。
相邻 `query_heads / kv_heads` 个 Q head 共享一个 KV head。

P 始终为 FP32 `[batch, query_heads, tokens]`，大小为
`4 * batch * query_heads * tokens` 字节，可复用，不是旧 MR3 的 online-softmax state。
例如 batch=4、heads=40、tokens=2048，需要 1.25 MiB 的 P。
QK 的 MRS 不含 V，避免尚未使用的 V 被计算封装同步或标脏；PV 不接收 Q/K。
MR 输出不可与输入重叠。调用成功返回输出元素数，参数错误返回 -1。

metadata 使用小端 `<9If`，40 字节，顺序为 version、phase、dtype、batch_size、
num_query_heads、num_kv_heads、head_dim、token_count、token_stride、scale。
version=1，dtype=1 表示 FP32，dtype=2 表示 FP16。
Python 对应 `cemu_flexgen/attention_phases_abi.py`。

每次 QK 必须包含本次 softmax 的完整 token 集合；本接口没有跨 chunk 合并语义。
不要把每个 chunk 各自归一化后的输出直接求和。旧 Dense chunk 路径仍然保留。
未来 SparF 可以把选定 K/V 按一致顺序提供给阶段接口，但选择、读取、稀疏修正尚未加入。

## 实现开销

CPU/CUDA 阶段均不在每次调用时申请中间堆内存或显存。
CUDA 每个 batch/head 对应一个 block，QK 按 token 分工，PV 按维度分工；
每 block 使用固定 1 KiB 共享内存做 softmax 归约，P 使用调用方提供的范围。
算子不自行 H2D/D2H，由 CEMU mirror 负责；算子只等待自身线程 CUDA stream 完成。
保留 FP32 softmax 和累加，不承诺逐位匹配 PyTorch FP16 计算路径。

## 测试命令

宿主机，在 `CEMU/tests/cemu`：

```bash
make instattention-attention-phases-test
make -C kernel ../build/attention_phases_devptr.so
```

第一条运行 CPU 数值测试，包括两种精度、MHA/GQA/MQA、单 token、非整齐维度、
跨 block token 数量、40 heads/1025 tokens、稳定 softmax、独立 PV 和无效参数检查。
不需要模型权重、CEMU 或 GPU。仅编译 CUDA 不占用 GPU。
如已获准使用空闲 GPU，也可直接检查 CUDA 算子的数值：

```bash
CUDA_VISIBLE_DEVICES=<空闲GPU编号> python3 instattention/python/test_attention_phases.py \
  --cuda --library build/attention_phases_devptr.so
```

Guest 激活环境后，在 `/root/CEMU/tests/cemu`：

```bash
make instattention-attention-phases-device-test \
  FDM_TEST_DIR=/mnt/fdm0 CSD_TARGET=cuda DTYPE=float16
```

`CSD_TARGET=cpu` 可测 CPU，`DTYPE=float32` 可测另一精度。
算子 `.so` 必须先在宿主机编译，`PROGRAM` 是相对于 QEMU 工作目录的宿主机路径，
规则与旧 Attention 测试相同；Guest 只编译 Python bridge。

Guest 测试创建两个三范围 MRS，共享同一个 P 文件，总共五个 FDM 文件。
连续两轮更新 Q/K/V，每轮两次 Direct execute；QK 后才写入 V，PV 后才读回 P/output，
检查中间 P 与输出。它验证 MRS/mirror 交接，不验证内部 FTL 读取或异步重叠。
两组 MRS 在全部计算结束前都保持存活。

## 设备侧单请求工作流

metadata 带有 K/V 在 NVM namespace 中的物理 extent，一次 Direct execute 使用六个MR：
`Q、K staging、V staging、P、output、trace`。设备先通过已有 FTL 路径读取 K；
默认在提交 V 读取后执行 QK，二者实际工作可以重叠；两者均完成后执行 PV，最后只向
Guest 完成一次请求。`serial` 对照模式在QK完成后才提交V。

当前父请求仍作为一个连续 job 进入原CSD调度器。其目标runtime使用依赖关键路径：
`K_read + max(QK, V_read) + PV`，serial模式为
`K_read + QK + V_read + PV`。Flash读取时延来自原FTL模型，QK/PV时间由metadata分别给出。
实际功能执行期间沿用VM freezing；不会在冻结区等待虚拟时钟到达未来时刻。
这已经避免把可重叠阶段简单相加，但尚未把QK/PV注册成两个独立CU调度job；因此多个
并发逻辑Attention之间的阶段级CU穿插仍是下一轮精化内容。
当前工作流要求配置 `internal_bandwidth=0`。非零内部带宽的原FTL路径要等待虚拟时钟，
而功能算子执行期间虚拟时钟被冻结；CEMU会在提交工作流前拒绝该组合，避免设备侧死锁。

Guest测试命令：

```bash
make instattention-attention-workflow-device-test \
  NVM_TEST_DIR=/mnt/nvme0 FDM_TEST_DIR=/mnt/fdm0 \
  CSD_TARGET=cuda DTYPE=float16 TOKEN_COUNT=257
```

测试使用同一份NVM K/V分别运行overlap和serial模式。每种模式只提交一次Guest
execute，检查P、输出、依赖顺序、关键路径公式，并打印K/QK/V/PV实际时间及QK/V重叠。
trace是Host单调时钟，仅用于验证真实功能时序；模拟时间仍由CEMU scheduler/FTL推进。
