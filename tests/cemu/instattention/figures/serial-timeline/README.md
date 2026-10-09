# 横向时间轴（2026-09-29）

本版将完整生成、单层 Decode 和 Execute 实测放大分为三张横图。PNG 可插入汇报文档；SVG 可无损放大。未运行新的性能实验，未修改执行代码。

| 内容 | PNG | SVG |
|---|---|---|
| 完整生成与 Prefill：主线程、KV writer、后台写回 | [图片](01-prefill-overview.png) | [矢量图](01-prefill-overview.svg) |
| 单层 Decode：调用顺序、数据传输、等待范围 | [图片](02-decode-layer.png) | [矢量图](02-decode-layer.svg) |
| 一个历史 Execute：Host / 虚拟时间实际比例 | [图片](03-execute-measured.png) | [矢量图](03-execute-measured.svg) |

## 读图边界

图 01、02 是**结构时间轴**，标注逻辑先后与可能重叠，宽度不表示耗时。图 02 的 τ0…τ10 是拟定的分段边界，不是已有事件；“待测”也不意味着系统没有聚合指标。当前缺少能关联到图 03 同一个请求的完整 Guest 逐请求记录，不能填出端到端实测时间轴。

图 02 从当前层 QKV 调用开始；此前需要取得本层权重。Decode runner 使用 `ModelWeightPrefetcher`，下一层权重可提前加载。每个 token 还包括 embedding、各层外的输出头等；图 01 总览纳入生成流程，但不声称给出了初始化、权重 I/O 与全部 PyTorch 算子的逐项成本。

KV 导出后，`_append_heads()` 在循环内**打一个包、调用一次 ioctl，再打下一个包**。不能画成先打完所有包再依次发送。同包是同层、同 batch 的多个 head。所有包成功后才更新 mean 并继续此层 Attention。

图 01 的 writer 与主线程重叠仅示例一种布局。提交本层写任务后等上一层 writer；下一层 Attention 结束后才进入本层 writer 的 join 点。writer 已完成时 join 可以很短。Prefill 末尾等最后 writer，并不强制所有 tail 写入 NAND；最终 drain 是生成结束后的另一个屏障。

## 图 03 数值来源

- 原日志：`/home/lihan/cemu/native-batch-device-timeline.jsonl`，文件日期 2026-09-26。
- schema：`sparf-autonomous-v1`；session：`906725105082679`；pid：`3063245`；request：`1`。
- layer：`0`；tokens：`1025`；engine：`0`；success：`true`。
- [所选完整请求原文](execute-request-1.json)，[按 Host 时间排序的事件](execute-events.csv)，[各区间数值](execute-timings.csv)。

这是一个历史样本，不是平均值，也不是当前代码重新运行的结果。未将其他日志中的 batch、模型配置、Guest profile 或端到端时间拼接进来。图 02 底部 B=4/H=40/d=128 只是包大小示例，不是由该日志证明的图 03 配置。

| 区间 | Host 实际时间 | QEMU 虚拟时间 |
|---|---:|---:|
| admitted → ready | 183.282234 ms | 82.474129 ms |
| admitted → engine started | 0.074677 ms | 0.075047 ms |
| Approx 功能区间 | 37.462064 ms | 0 ms |
| Exact QK 功能区间 | 9.469244 ms | 0 ms |
| PV 功能区间 | 0.579879 ms | 0 ms |

`admitted → ready` 是 Host 自治请求包络，不是 Guest `decode.execute_wait_wall`：后者还包住 ioctl 控制路径和 CQ 返回等。它也不包含此前 Q/mean 上传和此后输出回读。

读轨道的绿色是 `submitted → observed`，蓝色是 `observed → ready`；它们是状态机的观测窗口。尤其 V 读的 `observed` 在 QK 完成后记录，因此绿色窗口包含 QK 功能执行和模型等待，不能解释成纯 NAND 读或纯 gather 成本。蓝色也可能包含状态机收尾，不能全叫 sleep。

计算轨道紫色是 `functional_start → functional_end`；其后的短蓝色是 `functional_end → observed_done`，包含功能收尾、解冻、CU 提交、模型等待及完成观察。PV 的紫色时间只包住 PV 功能调用，生成 trace 在其后。

虚拟时间读轨道画的是模型起点至 deadline，三个目标长度分别为 22.375000 / 29.187500 / 29.187500 ms。计算轨道画提交至完成区间，可能大于有效计算预算。selected-V 目标与 Exact QK 模型时间重叠，不能直接求和。

`reads[].submitted.virtual_ns` 在代码中使用 `model_cursor`，不一定是与 `submitted.host_ns` 同时取得的时钟采样。两张实测轴各自以 admitted 为零点，不能通过同一 x 坐标或插值构造精确 H↔V 映射。日志未记录每个 freeze enter/leave 边界，也没有单独的 top-r 时间，不能由 H−V 或空白区反推出总冻结时长。

## 代码与重绘

- [Prefill 主线程与 writer join](../../python/runtime_common/prefill_runner.py)
- [Decode 各层、权重预取与输出头](../../python/runtime_common/decode_runner.py)
- [逐包打包与 ioctl 重试](../../python/cemu_flexgen/native_kv_store.py)
- [Q/mean 上传、Execute 和回读](../../python/cemu_flexgen/cemu_sparf_device.py)
- [slot workers 与 future](../../python/cemu_flexgen/cemu_sparf_slot_scheduler.py)
- [Host 冻结、自治循环与完成观察](../../../../../hw/femu/csd/compute.c)
- [读起点、功能计算起止的实际打点](../../../../../hw/femu/csd/sparf_workflow.h)
- [完整原始时序说明](../../NATIVE_SERIAL_SEQUENCE.md)

在本目录运行 `python3 render.py`。脚本优先读取保存的请求 JSON，复用相邻 `serial-sequence/render.py` 的本地 librsvg/cairo 导出函数；不依赖远程渲染服务。
