# InstAttention / CEMU 文档导航

当前目标是复现论文行为再评估模拟精度，不以吞吐吻合或一次functional PASS代替行为验收。

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
