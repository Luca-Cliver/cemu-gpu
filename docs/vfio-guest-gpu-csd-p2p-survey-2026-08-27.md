# VFIO Guest GPU 与 Host GPU/CSD P2P 技术调研

日期：2026-08-27

## 结论摘要

现有技术已经比较成熟地覆盖两类场景：

1. **同一驱动域内的设备 P2P**：Host GPU↔Host SSD/CSD，以及同一 Guest 内多块 VFIO GPU/PCIe 设备之间的 P2P；
2. **GPU↔第三方 PCIe 设备**：GPUDirect RDMA、GPUDirect Storage、SSD↔GPU P2P、SmartSSD 内 NVMe↔FPGA DDR P2P。

但截至本次检索，尚未找到一个公开、通用、受支持的实现，能够让：

```text
VFIO 直通给 Guest、由 Guest NVIDIA driver 管理的 GPU
                         ↕ 直接 PCIe P2P
仍由 Host NVIDIA driver/QEMU 管理的另一块 GPU
```

这种场景困难的根源不是 PCIe 交换机，而是两个设备分属不同的地址空间、IOMMU 安全域和 NVIDIA 驱动域。现有 QEMU 的 `x-nv-gpudirect-clique` 解决的是“多块直通 GPU 位于同一个 Guest”的 P2P 能力声明，并不跨越 Guest/Host 驱动边界。

对 CEMU 最有价值的方向可能是：保留当前真实 CSD 式控制面，由 CEMU 设计一个 **cross-domain P2P bridge / emulated GPUDirect datapath**，将 Guest GPU buffer 的导出、CEMU range/MRS、Host CSD GPU mirror 和模拟时序统一起来。即使第一版仍用功能性 bounce copy，也可以严格模拟 P2P 可见延迟和 residency 状态；后续再替换为 dma-buf、GPUDirect RDMA 或硬件 DMA 数据面。

## 1. 已有技术一：同一 Guest 内多块 VFIO GPU 的 P2P

QEMU 的 `vfio-pci` 已包含 NVIDIA GPUDirect P2P vendor capability：

```text
x-nv-gpudirect-clique=<0..15>
```

QEMU 源码说明，同一个 clique ID 的直通 NVIDIA 设备被声明为可直接 P2P；配置正确性由用户负责。其作用是向 Guest NVIDIA driver 暴露一个虚拟 PCI capability，让 Guest 知道哪些直通 GPU 属于同一 P2P clique。

典型形式：

```text
-device vfio-pci,host=21:00.0,x-nv-gpudirect-clique=1
-device vfio-pci,host=27:00.0,x-nv-gpudirect-clique=1
```

Cloud Hypervisor 和 OpenVMM 也有同类支持。OpenVMM 文档进一步说明：

- 有 ATS 时，直通设备可借助 IOMMU 地址转换进行 P2P；
- 某些平台没有 ATS，需要关闭交换机 ACS 重定向，并保持 Guest BAR 地址与 Host 物理 BAR 地址一致；
- 因此需要 BAR identity mapping/pinning。

这证明“VFIO 设备在虚拟机里做真实 PCIe P2P”不是空想，但已公开实现的基本单元是：

```text
两个 peer 都直通同一个 Guest
```

而不是一个在 Guest、另一个继续由 Host 原生驱动管理。

## 2. 已有技术二：GPUDirect RDMA

NVIDIA GPUDirect RDMA允许第三方 PCIe 设备直接访问 GPU memory。典型 peer 包括 NIC、采集卡和存储控制器。

基本过程：

1. CUDA 应用分配 GPU buffer；
2. NVIDIA driver pin GPU pages；
3. 向第三方设备驱动提供可 DMA 的 page/BAR mapping；
4. 第三方设备 DMA engine 直接读写 GPU memory；
5. 完成后解除 mapping。

关键约束：

- 最好共享同一 PCIe root complex；只有 PCIe switch 的路径最佳；
- GPU BAR1 aperture 和 Resizable BAR 资源必须足够；
- GPUDirect RDMA要求不同设备看到一致的物理地址；传统非 1:1 IOMMU translation 与之不兼容，通常要求关闭 IOMMU 或使用 pass-through/ATS 等平台支持；
- GPU buffer 所属 CUDA VA space 需要由 NVIDIA driver 正确认证和 pinning；一个裸 GPU virtual address 本身不够。

它是实现真正 cross-domain bridge 最值得借鉴的底层接口，但不能直接解决当前问题：Guest里的 CUDA VA 和 pinning 请求由 Guest NVIDIA driver管理，而Host CEMU只能访问Host NVIDIA driver，二者没有现成的page-export协议。

## 3. 已有技术三：GPUDirect Storage、SPIN与SSD↔GPU P2P

GPUDirect Storage通过存储侧或邻近存储的DMA engine直接访问GPU memory，避免CPU DRAM bounce buffer。控制面仍由CPU上的驱动设置，数据面则走PCIe P2P。

SPIN（USENIX ATC 2017）更早研究了SSD与GPU之间的P2P DMA，并将其与page cache、read-ahead、POSIX一致性和虚拟块设备结合。这说明：

- GPU↔SSD P2P并不只是一条硬件copy命令；
- 真正难点包括文件系统语义、page cache、一致性、预读和错误处理；
- 模拟器若只建模`bytes/bandwidth`，可以做性能上界，但要声称完整系统数据面还需描述这些软件语义。

这些工作主要运行在同一个Host OS/driver domain中，没有解决VFIO Guest GPU与Host-owned peer之间的跨域映射。

## 4. 已有技术四：SmartSSD/CSD内部P2P

Samsung/Xilinx SmartSSD的公开架构支持：

```text
NVMe SSD → PCIe P2P → FPGA DDR
```

通过Resizable BAR将FPGA DDR映射到PCIe I/O地址空间，NVMe等第三方设备可以直接DMA到该内存，从而避免Host DRAM。

XRT公开P2P支持也覆盖：

- Alveo DDR/HBM↔另一块Alveo DDR/HBM；
- NVMe等第三方设备↔Alveo DDR/HBM；
- 要求64-bit I/O、大BAR、合适的PCIe拓扑；
- IOMMU会使P2P绕经root complex或明显降低性能。

这对CEMU的启发是：CSD计算设备最好暴露一个稳定的、可被peer DMA寻址的device-memory aperture，而不是让上层传CUDA进程虚拟地址。CEMU当前的FDM/CUDA mirror可演化成逻辑上的P2P BAR/range object。

## 5. 已有技术五：虚拟化环境中的GPU↔第三方设备

有几类接近但不完全相同的工作：

- Janus：将GPU直通Guest，并用SR-IOV NIC VF和P2P DMA让网络数据直接进入GPU memory；说明“Guest GPU与直通/虚拟功能peer设备P2P”可以实现。
- NVIDIA vGPU：在受支持的SR-IOV GPU、Guest OS、NIC和hypervisor组合中支持GPUDirect RDMA/GDS；说明vendor-mediated GPU memory export可以跨虚拟化层实现，但依赖NVIDIA完整软件栈和硬件支持。
- QEMU/Cloud Hypervisor/OpenVMM：支持同一Guest内直通设备P2P、BAR固定映射、ATS/ACS相关配置。

共同特点仍然是：peer访问能力由hypervisor和vendor driver显式协作暴露给Guest；不是Guest GPU可以天然访问任意Host-owned设备。

## 6. 为什么Guest GPU↔Host GPU特别困难

当前CEMU拓扑在物理上有利：`21:00.0`、`22:00.0`、`27:00.0` A100位于同一组Broadcom PEX890xx交换结构下，`27:00.0`绑定VFIO，其余GPU由Host NVIDIA driver管理。

但逻辑上存在四个断点：

1. **驱动所有权**：Host CUDA看不到VFIO GPU，Guest CUDA看不到Host CSD GPU。
2. **地址空间**：Guest CUDA pointer仅在Guest CUDA VA space中有意义；Host无法拿该值调用自己的NVIDIA P2P pinning API。
3. **IOMMU隔离**：VFIO的目标是限制Guest device可访问的地址；允许它任意向Host GPU BAR发TLP会突破默认隔离模型。
4. **同步与生命周期**：即使得到BAR物理地址，还需保证GPU kernel、DMA和buffer free之间的顺序，以及异常/VM退出时撤销映射。

因此，物理PCIe路径可达不等于软件栈已经允许P2P。

## 7. Guest GPU↔CSD计算设备有哪些可借鉴部署

### 7.1 两者都直通同一个Guest

```text
Guest GPU + 物理SmartSSD/CSD/第二GPU → 同一个Guest
```

可以复用VFIO P2P、ATS/BAR identity mapping和Guest内driver协作。这是实现真实P2P最直接的实验方案，但CSD执行控制容易落入Guest，弱化当前“CEMU设备侧启动算子”的自然控制边界。

### 7.2 Guest GPU + vGPU/SR-IOV peer

依赖vendor提供的mediated/SR-IOV GPU memory export和GPUDirect支持，云平台已有先例。但A100、特定GDS版本和full-passthrough/vGPU模式的支持矩阵需逐项核对，不能默认可用。

### 7.3 Guest GPU + Host emulated CSD

这正是CEMU当前场景。虚拟CSD没有独立的物理PCIe DMA requester；所谓P2P最终必须由QEMU执行功能copy，或者由额外真实设备/内核模块代理。因此最合理的是明确称为：

```text
functional copy + modeled P2P datapath
```

而不是物理P2P。

## 8. 对CEMU的建议路线

### 阶段A：可信的模拟P2P

保留当前控制面：

```text
Guest application → NVMe/CSD command → CEMU scheduler → Host CSD GPU
```

新增P2P range和时序模型：

- Guest提交buffer/range descriptor，不直接暴露CSD GPU pointer；
- CEMU内部仍可通过host staging完成正确性copy；
- Guest-visible时间只加入`fixed_latency + bytes / p2p_bw`；
- range状态记录`GUEST_GPU_VALID / CSD_GPU_VALID / STORAGE_VALID`；
- 统计`p2p_bytes、bounce_bytes、modeled_latency、wall_time`；
- 明确区分functional datapath与modeled datapath。

这能回答“若P2P存在，端到端收益是多少”，且符合CEMU的模拟器定位。

### 阶段B：共享pinned/dma-buf中间层

用ivshmem、virtio共享内存或dma-buf降低当前NumPy/文件/FDM多次copy。它仍不是GPU-to-GPU P2P，但可以建立buffer export/import、fence和生命周期协议。

### 阶段C：Guest buffer export协议

研究Guest NVIDIA driver/用户态能否导出dma-buf或GPUDirect元数据，经受控hypercall/virtio通道传给Host bridge；Host kernel module验证并建立peer mapping。此阶段安全和vendor API依赖最大。

### 阶段D：真实硬件peer

将CEMU虚拟CSD替换或旁接真实DMA-capable FPGA/NIC/SmartSSD，验证：

- 同switch P2P；
- BAR mapping；
- IOMMU/ATS/ACS；
- DMA读写Guest GPU buffer；
- 与模拟模型的带宽、延迟和并发一致性。

## 9. 可能的研究价值

谨慎表述应是：本次检索没有发现公开工作完整覆盖“VFIO Guest GPU + Host-controlled GPU accelerator embedded in an emulated CSD”的组合，而不是断言绝对没有。

潜在贡献点包括：

1. 跨Guest/Host驱动域的GPU buffer/range抽象；
2. 保留CSD命令控制面的GPU P2P数据面；
3. functional copy与P2P模拟时序解耦；
4. Host、Guest、storage、CSD GPU四方residency/一致性状态机；
5. 从bounce-copy模拟到真实GPUDirect peer的可替换后端；
6. 在长上下文LLM KV/Attention上的端到端收益和敏感性研究。

最有说服力的论文问题不是“我们让两块GPU通信了”，而是：

> 如何在不把CSD计算设备暴露给Guest应用的前提下，为VFIO Guest GPU提供可验证、可建模、最终可替换为物理GPUDirect的CSD P2P数据通路？

## 10. 主要资料

- NVIDIA GPUDirect RDMA documentation
- NVIDIA GPUDirect Storage Design/Overview Guide
- Linux PCI Peer-to-Peer DMA documentation
- Linux VFIO/IOMMUFD documentation
- QEMU `vfio-pci` NVIDIA GPUDirect clique implementation
- Cloud Hypervisor/OpenVMM VFIO P2P documentation
- Xilinx XRT PCIe P2P documentation
- Samsung/Xilinx SmartSSD Installation and User Guide
- SPIN, USENIX ATC 2017
- Janus, GPU Direct Packet I/O in virtualized environments
- InstAttention/InstInfer, HPCA 2025
