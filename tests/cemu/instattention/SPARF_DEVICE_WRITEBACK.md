# SparF 设备内批量写回：B1 实验路径

## 本次实现与边界

这是行为对齐计划第四步的第一小批：移除后台写回中的 `FDM → Guest 内存 → NVM 文件写入` 往返。
默认路径不变，OPT 大实验未自动切换。不是完整的设备 buffer 会话管理，也不代表论文流水已经复现。

```text
Guest 生成 K/V → 写 FDM ring
Guest 原有后台 worker：凑组、提交一个 WRITEBACK execute（只传描述符）
  → CEMU 固定 program/MRS 引用
  → 设备写回线程：归并目标扇区，通过 FTL 读旧扇区
  → 等物理读完成和读的目标时刻
  → 直接读取 FDM，更新 token-K / channel-K / token-V 扇区内容
  → 通过同一 Controller 的 FTL 写 NVM
  → 等物理写完成和写的目标时刻
  → 返回一次完成，解除本请求引用
Guest：仅成功后推进 published，释放 ring 空间
```

**仍在 Guest 的职责**：接收位置计数、组触发、背压、published 边界、V 均值和 Attention 尾部快照。
尾部快照目前仍会读回 Guest 并放入 Attention slot；本次只去掉写回链的数据往返，不能宣称全部中转消失。
后续 B2 才会迁移 accepted/published、均值、buffer 引用、APPEND/DRAIN 会话协议，之后再接多请求资源调度。

## 设备实现

- 独立命令 `CEMU_SPARF_WRITEBACK_COMMAND`，metadata 版本 1；不挪用旧 SparF 保留字段。
- MR0 是已有 FDM ring，MR1 是 64 字节结果（文件分配 4 KiB）；不复制整块 ring，不创建 CUDA mirror。
- 描述符只含源偏移、目标 NVM 字节地址、长度、条数、源/目标步长。设备检查长度、溢出、边界和资源上限。
- 对相同扇区去重，设备内执行 read-modify-write，保留部分扇区中其他 token/层/通道的数据；支持 ring 绕回及非连续 extent。
- 每次请求最多 65536 个描述符、64 MiB 逻辑复制量、64 MiB 扇区镜像、1048576 个去重前扇区条目。超限显式失败，不隐式退回 Guest 写回。
- 临时扇区镜像使用连续内存；合并连续物理扇区后提交 FTL，避免逐元素发 NVMe 请求。
- 当前保守地读取全部涉及扇区，即使某些扇区会被完全覆盖；以后可以优化，但不能省略必要的读改写成本。
- 每 namespace 最多一个在途写回。写回使用独立、延迟创建并复用的线程，不占目标 CU，可与 Attention 读取/计算竞争同一 Flash 资源。
- Python 同一进程内的多个 microbatch 通过 namespace 锁串行提交写回；其他进程若并发提交会收到忙错误。
- FTL 调用沿用现有内部 NVM 读写路径，同一 Controller 不额外计 PCIe；本版要求 `internal_bandwidth=0`。
- 这是内部 IO 与布局搬运，不调用 Attention 数学算子。不修改、不绕开原计算路径的 freezing、时钟或预算扣除。

FTL 返回失败时不报告成功，Guest 不推进 published；可能已有部分 NVM 数据写入，本协议没有回滚或掉电恢复。
失败会停止该 buffer 的后续提交并保留错误直到排空/关闭。尚不支持热拔插、reset、取消和多进程共享同一批 KV 文件。
FTL/Host 后端自身没有报告出来的错误，不能由这个接口凭空检出。

## 文件所有权交接

只允许本测试独占的临时 KV 文件，不用于他人的文件、普通业务文件或任意未初始化 extent。

1. Guest 完成所有层 Prefill，publish 三种布局。
2. 关闭 channel-K mmap，丢弃清洁文件缓存。
3. 确认文件属于指定 `/dev/nvme0n1`，取得文件锁，通过 FIEMAP 检查映射完整、已初始化、无共享/未知标志和物理重叠。
4. 文件描述符替换成只读 `O_DIRECT`；后续 Guest KV 写函数直接报错。
5. Attention 仍按固定 extent 读取，数值验证用对齐的直接读取，不依赖旧 page cache。
6. 等全部写回和 Attention 完成后关闭。当前不提供交回普通 Guest 写入的协议；关闭后不要重开此 store，对下一次测试创建新会话/文件。

文件锁是协作式锁，不能阻止外部程序无视锁、truncate、移动或修改文件；测试期间必须保持文件、extent 和 FDM 分配不变。
拒绝 unwritten extent 很重要：仅 fallocate 还不代表文件系统会把设备直接写入的块当作有效文件内容。
Prefill 当前会初始化整层布局及 padding；未初始化的层/洞应被拒绝，不能放宽检查凑出 PASS。

## Host 验证

```bash
cd /home/lihan/cemu/CEMU
make -C tests/cemu instattention-sparf-stages-test
ninja -C build -j2 qemu-system-x86_64
```

新增测试直接编译生产 C 写回计划和执行代码，使用确定性假 FTL，覆盖：FP16/FP32、两层、多个 batch/head、ring 绕回、碎片 extent、边界扇区保留、扇区去重、错误 metadata、IO 失败和时间依赖。
Python 回归还检查写回不调用 Guest `_read`/`write_tokens`、完成前不发布、失败停止复用、所有权只交接一次、直接读取和 FIEMAP 拒绝不安全映射。
这不是实际 QEMU/Guest/FTL 集成通过的替代证据。

## Guest 小测试

Host 正常关闭旧 VM 并使用新 QEMU 启动，沿用已验证的 GPU、驱动库和 NVM 后端设置。
无需更新 Linux 内核模块或 CUDA 算子；本次不加载 OPT 权重。

```bash
source /root/cemu-gpu-env/bin/activate
cd /root/CEMU/tests/cemu
mkdir -p /root/cemu-logs
set -o pipefail
make instattention-sparf-device-writeback-test \
  NVM_TEST_DIR=/mnt/nvme0 \
  FDM_TEST_DIR=/mnt/fdm0 \
  CSD_TARGET=cuda \
  PROGRAM=./build/sparf_attention_devptr.so \
  2>&1 | tee /root/cemu-logs/sparf-device-writeback.log
```

`PROGRAM` 指向 Host QEMU 工作目录下已有库。写回本身不调用库函数，Attention 对照仍用该库。
CPU 对照将 `CSD_TARGET=cpu`、`PROGRAM=./build/sparf_attention.so`，其余相同。

测试为两个 microbatch、两层、5-token Prefill、8 次追加，压缩比 1 和 8；设备写回模式每层 ring 容量 6，分组 3，覆盖绕回及最终不足一组的 2-token drain。
每次 Attention 对照 NumPy；最终直接读取 token-K/V 及所有 channel-K 验证，打印最后一组 FTL 读写目标依赖。

结果 MR 字段为 `read_bytes, write_bytes, read_start, read_end, write_start, write_end, ready, copied_bytes`。
时间均为 CEMU 模拟时钟，要求 `read_start ≤ read_end ≤ write_start ≤ write_end ≤ ready`。
这是最后一组写回的诊断，不是整个 Decode 时延，也不是独立 NAND 芯片实测时间。
`CEMU_SPARF_AUTONOMOUS_TIMELINE` 仍只记录 Attention v3，不会自动记录这条 IO 写回命令。
# OPT 应用接入（可选实验路径）

后续布局迁移设计见 [SPARF_GROUPED_KV_DESIGN.md](SPARF_GROUPED_KV_DESIGN.md)。当前 spans/compact 都仍为旧文件布局的扇区读改写；新分组追加布局仅有 Host 参考测试，以下命令不会启用它。

现在可在 `test_opt_cemu_pipeline.py` 中组合启用：

- `SPARF_AUTONOMOUS=1`：v3 单命令 Attention，不能同时设置 `SPARF_STAGED=1`。
- `SPARF_DEVICE_WRITEBACK=1`：FDM buffer 经设备 FTL 写回，要求 `FUNCTIONAL=1` 和至少两个 buffer token。
- `NVM_DEVICE=/dev/nvme0n1`：KV 文件所在 NVM 块设备，必须与 `NVM_TEST_DIR` 对应。

默认路径保持不变。v3 仍要求设备 `nr_cu=1`；本次没有修改设备调度、FTL、算子或 freezing。
共享 slot 调度器允许后台准备下一请求，但通过一个提交 worker 保证当前进程这组 slot 最多一个 v3 parent execute 在途。
不要同时启动另一个自治测试进程。此限制不是论文最终的多请求设备流水，不能据此宣称完整性能复现。
KV 写回使用独立后台路径；最终 `flush()` 等所有 accepted token 发布后才结束。

首次 Prefill flush 时移交文件所有权。为避免 raw device 写入落在 ext4 的 unwritten extent 中却被文件系统读成零，
移交前仅对预分配且未初始化的 extent 写零并 fsync，已有 Prefill 数据不重写；之后重新验证 FIEMAP。
这可能增加一次性的 Prefill drain 时间和实际 SSD 占用，日志打印 `initialized_unwritten_bytes`，不是每层 Decode 的额外初始化。
孔洞、共享 extent、错误的 NVM namespace 仍拒绝移交。设备拥有文件期间不得由其他程序访问或修改这些临时 KV 文件。
buffer 分组、发布边界、均值与 Attention 尾部快照仍由 Guest 管理；移除的是写回的数据回传，不是全部 Guest 中转。

## 联合测试顺序

先在使用已编译设备写回代码的 QEMU 中跑小测试，不加载模型。它覆盖两个 microbatch、两层、八步追加、
环形 buffer 回绕和末尾不足一组的 flush，对照 NumPy Attention 并读取最终 token-K、channel-K、V 校验。

```bash
source /root/cemu-gpu-env/bin/activate
cd /root/CEMU/tests/cemu
mkdir -p /root/cemu-logs
set -o pipefail
make instattention-sparf-autonomous-writeback-test \
  NVM_TEST_DIR=/mnt/nvme0 FDM_TEST_DIR=/mnt/fdm0 \
  CSD_TARGET=cuda PROGRAM=./build/sparf_attention_devptr.so \
  2>&1 | tee /root/cemu-logs/sparf-autonomous-writeback-small.log
```

通过后再接 OPT。先不传 runtime 配置，以免把功能问题与时间模型混在一起；也不设置任何未知的 softmax ratio 路径。

```bash
make instattention-opt-cemu-gpu-test \
  MODEL_DIR=/root/models/OPT-13B \
  NVM_TEST_DIR=/mnt/nvme0 FDM_TEST_DIR=/mnt/fdm0 NVM_DEVICE=/dev/nvme0n1 \
  CSD_TARGET=cuda ATTENTION_MODE=sparf FUNCTIONAL=1 BENCHMARK=0 \
  SPARF_AUTONOMOUS=1 SPARF_STAGED=0 SPARF_DEVICE_WRITEBACK=1 \
  BATCH_SIZE=2 GPU_BATCH_SIZE=1 PROMPT_LENGTH=16 DECODE_STEPS=8 \
  ATTENTION_SLOTS=2 SPARF_KV_BUFFER_TOKENS=6 \
  SPARF_TOP_R=16 SPARF_COMPRESSION_RATIO=8 \
  RUNTIME_CONFIG= SOFTMAX_RATIO_FILE= TRACE_FILE= PROMPT_TEXT= \
  PROFILE_OUTPUT=/root/cemu-logs/opt-v3-writeback-profile.csv \
  2>&1 | tee /root/cemu-logs/opt-v3-writeback.log
```

OPT 的 functional PASS 只证明流程完成；严格 Attention/KV 数据验证看前面的小测试，不把稀疏生成结果当作 dense/HF 等价证明。
本次 Host 单测和模拟 transport 不代替上述真实 Guest 测试，也不代表时间模型已经校准。

## 写回控制开销拆解与紧凑描述

新增 `SPARF_WRITEBACK_FORMAT=compact|spans`（设备写回默认 compact）。这是独立的写回命令 ABI v2，
与 Attention 的 autonomous v3 编号无关。spans 保留 ABI v1 逐 channel 描述，用于同配置对照。

compact 将 channel-K 的“token 数量/步长 + channel 数量/步长”表达成二维矩阵，设备在有界计划中展开。
extent 边界和 ring 回绕仍拆分；非连续映射不会被当成连续地址使用。Guest 不再为每个 channel 计算/打包描述。
原来的源 FDM 数据、三种目标文件布局、扇区集合、读改写顺序和 FTL 请求保持不变。
它减少的是控制描述，不是用省略真实读写来加速，也尚未实现论文的页/group 布局或设备 buffer 会话管理。

一个仅构造 metadata 的 Host 微测（OPT-13B形状，单 microbatch=1，40 heads，128维，连续 extent）得到：

| 写回位置/数量 | spans 大小/描述数 | compact 大小/描述数 | 本机构造中位数 spans / compact |
|---|---|---|---|
| 16 / 3，ring回绕 | 332816 B / 10400 | 11536 B / 240 | 223.35 / 3.66 ms |
| 19 / 3 | 166416 B / 5200 | 5776 B / 120 | 111.93 / 1.85 ms |
| 22 / 2，最终尾部 | 166416 B / 5200 | 5776 B / 120 | 112.32 / 1.84 ms |

每种形状预热一次、测量20次；没有访问 CEMU、GPU、NVM，也没有加载 checkpoint。
这不能等同于 Guest 时间或端到端收益，结果受 Host 环境影响，需以下真实实验验证。

### 新的可选统计

仅设置 `PROFILE_OUTPUT` 时启用。两种描述格式都支持相同统计：

- `kv_writeback.metadata_wall`：Guest 构造描述。
- `kv_writeback.lock_wait_wall`：Guest 等待该 namespace 共享写回锁。
- `kv_writeback.execute_wait_wall`：Guest execute 调用区间。
- `kv_writeback.status_read_wall`：新增的诊断 MR 读取开销（每组一次，仍在共享锁内，避免读到下一组状态）。
- `kv_writeback.device.*_host_ns`：Host 单调时钟测得的计划生成、设备排队、源准备、读服务、读目标等待、数据修补、写服务、写目标等待。
- `kv_writeback.ftl_read_target_ns` / `ftl_write_target_ns`：FTL 请求目标区间，CSV 的 bytes 是实际读/写的扇区字节数。包含资源排队影响，不是纯 NAND 芯片时延。
- `kv_writeback.command_metadata`：命令描述字节累计，时间为0，仅看 bytes。
- `kv_writeback.device_payload`：三种布局实际修补的字节累计，时间为0，仅看 bytes；包含 token-K、channel-K 和 V，而旧 total_wall 的数据量仅计算一份 K/V。

Host 的 read/write service 包括请求构造和等待 FTL 返回；target_wait 是返回后额外等待模拟时钟到达目标时刻的 Host 时间。
它不是 SSD 硬件直接测量值。Host、Guest 和目标时钟各自计算区间，不能跨时钟相减或把所有区间相加。
不设置 PROFILE_OUTPUT 时没有分项设备计时、也没有逐组额外状态读取；原64字节结果仍保持兼容。
开启诊断会增加一定开销，对照时必须两组都开启或都关闭。

### 这轮测试

需要在方便时正常退出旧 VM，再启动本轮重新编译的 QEMU。**不能只更新 Guest Python 就在旧 QEMU 上运行 compact**。
无需更新 Linux 内核模块或 CUDA 算子，不修改 freezing、FTL 参数、目标 runtime。

先在 Guest 跑联合小测试，两种格式都必须通过 Attention、最终 token/channel KV 和 FTL 依赖检查：

```bash
source /root/cemu-gpu-env/bin/activate
cd /root/CEMU/tests/cemu
mkdir -p /root/cemu-logs
set -o pipefail
for format in spans compact; do
  if ! make instattention-sparf-autonomous-writeback-test \
    CSD_TARGET=cuda PROGRAM=./build/sparf_attention_devptr.so \
    NVM_TEST_DIR=/mnt/nvme0 FDM_TEST_DIR=/mnt/fdm0 \
    SPARF_WRITEBACK_FORMAT="$format" \
    PROFILE_OUTPUT="/root/cemu-logs/writeback-$format-small.csv" \
    2>&1 | tee "/root/cemu-logs/writeback-$format-small.log"; then
    break
  fi
done
```

随后复用前面的 OPT 小配置，只增加 `SPARF_WRITEBACK_FORMAT=spans` 或 `compact`，分别保存日志和 CSV。
保持 batch=2、GPU_BATCH_SIZE=1、prompt=16、decode=8、buffer=6、v3和设备写回开启，runtime 配置仍为空。
先检查生成序列，再比较新分项与总耗时。仍存在读写放大时，下一步再独立推进页/group 布局及成组追加，避免同时改变物理负载和控制实现而无法归因。
