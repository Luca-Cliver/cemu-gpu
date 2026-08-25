# CEMU InstAttention 阶段工作总结

日期：2026-08-20

## 1. 当前工作的定位

这部分工作的目标不是重新实现一套独立的 LLM 推理框架，也不是只做一个 Attention CUDA 算子，而是在 CEMU 全系统 CSD 模拟器中加入一个能够运行 LLM KV Cache 和 Decode Attention 工作负载的完整软件栈。

目前已经形成的路径是：

```text
输入 token
  -> Guest GPU 执行 Llama/FlexGen 风格 Prefill
  -> 按 layer/token/batch/head 布局 KV Cache
  -> K/V 持久化到 CEMU NVM 文件
  -> Decode 新 token 在 Guest GPU 上生成 Q/K/V
  -> 新 K/V 追加到 NVM
  -> 历史 K/V 按 chunk 从 NVM 搬到 FDM
  -> FDM 中的 5 个文件绑定为 CEMU MRS
  -> CSD CUDA 算子执行 Decode Attention
  -> Guest GPU 执行 Wo、残差、MLP 和 LM Head
  -> 生成下一个 token
  -> 将输出 token 反馈给下一步 Decode
```

因此，当前成果可以概括为：**已经在 CEMU 中打通了一个小型、确定性、可校验的 Prefill + 多步自回归 Decode 全流程，其中 Decode Attention 真正通过 CEMU 的程序下载、MRS、调度和 CUDA device-pointer 路径执行。**

## 2. 当前系统中的设备分工

当前实验中有两个不同角色的 GPU：

1. **Guest GPU**
   - 通过 VFIO 直通进入 QEMU Guest。
   - Guest 中的 PyTorch 可以直接使用该 GPU。
   - 负责 Prefill，以及 Decode 中除 Attention 主体以外的计算。

2. **CSD 计算 GPU**
   - 由宿主机上的 QEMU/CEMU 进程使用。
   - 通过 `CEMU_CUDA_DEVICE` 选择。
   - 作为 CEMU CSD 内部计算设备，执行 `dense_attention_devptr.so`。

当前 QEMU 启动参数位于 [`build/run-csd.sh`](../../../build/run-csd.sh)，默认配置包括：

- VFIO GPU：`0000:27:00.0`；
- Guest CPU：8 vCPU；
- Guest RAM：8 GiB；
- CSD CUDA 同步：`CEMU_CUDA_SYNC=1`；
- CSD 计算 GPU：`CEMU_CUDA_DEVICE=0`。

当前还不是 Guest GPU 与 CSD GPU 之间的真正 P2P 路径。Guest PyTorch Tensor 会先转换为 CPU/NumPy 数据，再经过 FDM 文件和 CEMU backend 的 CUDA mirror 进入 CSD GPU。因此目前首先保证的是功能完整性和结果正确性，而不是最终的数据传输性能。

## 3. 目录分层

```text
instattention/
├── CMakeLists.txt
├── guest/
│   ├── cemu_client.h
│   ├── cemu_client.cpp
│   └── pybind_module.cpp
├── python/
│   ├── cemu_flexgen/
│   │   ├── cemu_device.py
│   │   ├── cemu_attention_device.py
│   │   ├── attention_abi.py
│   │   ├── kv_layout.py
│   │   ├── kv_store.py
│   │   └── kv_staging.py
│   ├── flexgen_adapter/
│   │   └── cemu_attention.py
│   ├── flexgen_runtime/
│   │   ├── model_config.py
│   │   ├── weights.py
│   │   ├── embedding.py
│   │   ├── prefill.py
│   │   ├── prefill_runner.py
│   │   ├── decode.py
│   │   ├── decode_runner.py
│   │   ├── generation.py
│   │   ├── reference.py
│   │   ├── mlp.py
│   │   └── output.py
│   └── test_*.py
└── scripts/
    └── build_guest.sh
```

分层原则是：

- `flexgen_runtime`：只负责 Llama/FlexGen 风格张量计算，不依赖 CEMU；
- `cemu_flexgen`：封装 CEMU 设备、NVM、FDM、MRS 和 Attention ABI；
- `flexgen_adapter`：把 FlexGen 风格的 Q/K/V 张量转换成 CEMU 所需布局；
- `guest`：把 Python 调用转换为 CEMU ioctl 和 NVMe passthrough 命令；
- `tests/cemu/kernel`：CSD 内实际加载的 CPU/CUDA `.so` 算子。

这种分层使同一套 Prefill/Decode runtime 可以切换两种 Attention backend：纯 PyTorch reference 或 CEMU Attention。

## 4. 已完成的基础工作

### 4.1 CEMU CUDA device-pointer 执行路径

CEMU 原本主要把 `mr_addr` 作为 host 侧地址传给计算程序。现在增加了 CUDA device-pointer target：

1. `program_execute()` 根据 MRS 获得 `mr_addr`、`mr_len` 和 `mr_backend`；
2. `backend_cuda_ensure_device_ptr()` 为每个 range 懒分配 CUDA mirror；
3. 对应设备指针保存到 `mr_dev_addr`；
4. CUDA `.so` 直接读取 `args->mr_dev_addr[i]`；
5. 执行前完成 H2D，同步后将输出 D2H 回写。

相关实现：

- [`hw/femu/backend/backend.c`](../../../hw/femu/backend/backend.c)：按 range 管理 CUDA mirror；
- [`hw/femu/csd/compute.c`](../../../hw/femu/csd/compute.c)：构造 `mr_dev_addr`，按照 program target 执行；
- [`tests/cemu/kernel/cemu_def.h`](../kernel/cemu_def.h)：算子可见的 `cemu_args`。

Attention 需要跨多个 chunk 保留 online softmax 状态，所以目前 direct CUDA 路径除最后一个输出 range 外，还会将 `mr[3]` 的状态同步回 host。下一次 chunk 执行时，再把它同步到 GPU。

### 4.2 Python 到 CEMU 的桥

[`guest/cemu_client.cpp`](guest/cemu_client.cpp) 把原先 C++ 测试程序中的操作封装成 `CemuClient`：

- `load_program()`：下载 CPU 或 CUDA `.so`；
- `activate_program()`：激活程序；
- `create_memory_ranges()`：打开文件并创建 MRS；
- `write_range()` / `read_range()`：读写 MRS 对应文件；
- `execute()`：构造 NVMe passthrough program-execute 命令；
- `close()`：删除 MRS、停用并卸载程序。

[`guest/pybind_module.cpp`](guest/pybind_module.cpp) 使用 pybind11 将这些方法导出为 `_cemu_client` Python 模块。模块通过 [`CMakeLists.txt`](CMakeLists.txt) 在 Guest 中构建。

在 Python 侧，[`python/cemu_flexgen/cemu_device.py`](python/cemu_flexgen/cemu_device.py) 又封装了一层 `CemuDevice`，使上层代码不再直接处理 ioctl 结构体。

### 4.3 CPU 与 CUDA 基础验证

最初通过 vadd 验证了两条路径：

- host pointer CPU 算子；
- CUDA device-pointer 算子。

这一步证明 Python 可以完成 program download、activate、MRS 创建、execute 和结果读取，也证明 `mr_dev_addr` 能够被 CUDA `.so` 正确使用。

## 5. KV Cache 数据层

### 5.1 KV 逻辑布局

[`python/cemu_flexgen/kv_layout.py`](python/cemu_flexgen/kv_layout.py) 定义了独立于实际文件路径的 KV Cache 逻辑地址：

```text
token payload = batch_size * num_kv_heads * head_dim * element_size
token_stride  = align_up(token payload, 512 B)
layer_stride  = align_up(max_seq_len * token_stride, 4096 B)
file offset   = layer * layer_stride + token * token_stride
```

同一个 token 内部按照下面的顺序保存：

```text
batch 0 / head 0
batch 0 / head 1
...
batch 1 / head 0
batch 1 / head 1
...
```

K 和 V 使用相同的逻辑布局，但保存在不同文件中。布局类支持：

- 根据 `layer/token/batch/head/dimension` 计算偏移；
- 计算一个 token 和一个 layer 的空间占用；
- 根据 FDM staging 容量计算每个 chunk 可容纳的 token 数；
- 将 `[0, valid_tokens)` 自动划分为多个 `KvChunk`。

目前没有实现 InstAttention 中按 token 和 index 保存两份 K Cache 的优化布局；当前只有一份按 token 顺序组织的 K 和一份 V。

### 5.2 NVM 持久化

[`python/cemu_flexgen/kv_store.py`](python/cemu_flexgen/kv_store.py) 使用两个普通文件保存 K/V：

- `k_cache`；
- `v_cache`。

它使用 `posix_fallocate` 预分配布局需要的完整文件空间，并使用 `pwrite/pread` 按逻辑偏移读写单个或多个 token。写入和读取后的形状统一为：

```text
[tokens, batch, kv_heads, head_dim]
```

因此，Prefill 写入的历史 KV 和每步 Decode 新增的 KV 都能通过同一套布局定位。

### 5.3 NVM 到 FDM staging

[`python/cemu_flexgen/kv_staging.py`](python/cemu_flexgen/kv_staging.py) 管理：

- NVM 中完整的 `k_cache`、`v_cache`；
- FDM 中固定容量的 `k_staging_0`、`v_staging_0`。

每次 Decode Attention 只把当前 chunk 对应的 K/V 区间通过 `os.copy_file_range()` 从 NVM 搬到 FDM。这样历史 KV 总量可以大于 FDM staging 容量，不要求整个 KV Cache 同时驻留 FDM。

当前 chunk 是串行处理的：stage 一个 chunk，执行一次算子，再 stage 下一个 chunk。它保证了容量可扩展性，但还没有通过双缓冲、异步搬运或流水线隐藏传输开销。

## 6. CEMU Attention 执行层

### 6.1 五个 MRS range

[`python/cemu_flexgen/cemu_attention_device.py`](python/cemu_flexgen/cemu_attention_device.py) 为 Attention 创建五个 FDM 文件，并按固定顺序绑定到一个 MRS：

| MR | 文件 | 作用 |
|---|---|---|
| `mr[0]` | `attention_query` | 当前 Decode token 的 Q |
| `mr[1]` | `k_staging_0` | 当前 chunk 的 K |
| `mr[2]` | `v_staging_0` | 当前 chunk 的 V |
| `mr[3]` | `attention_state` | 跨 chunk 保存 online softmax 状态 |
| `mr[4]` | `attention_output` | 最终 Attention 输出 |

`mr[3]` 对每个 `[batch, query_head]` 保存：

```text
历史最大 score
历史 exp 分母和
历史加权 V 向量和[head_dim]
```

它不是保存所有 token 的 score，而是保存足以把多个 chunk 精确合并为完整 softmax 的中间状态。

### 6.2 Attention ABI

Python 侧 ABI 位于 [`python/cemu_flexgen/attention_abi.py`](python/cemu_flexgen/attention_abi.py)，C/CUDA 侧结构体位于 [`tests/cemu/kernel/attention_abi.h`](../kernel/attention_abi.h)。

每次 execute 会传递：

- batch size；
- query head 数；
- KV head 数；
- head dimension；
- 当前 chunk token 数；
- token stride；
- `1/sqrt(head_dim)` scale；
- reset/finalize 标志。

第一个 chunk 设置 `reset_state=true`，最后一个 chunk 设置 `finalize=true`。

### 6.3 CPU 和 CUDA Attention 算子

当前有两个行为一致的算子：

- [`tests/cemu/kernel/dense_attention.cc`](../kernel/dense_attention.cc)：CPU functional model；
- [`tests/cemu/kernel/dense_attention_devptr.cu`](../kernel/dense_attention_devptr.cu)：CUDA device-pointer functional model。

对每个 batch 和 query head，算子执行：

1. 根据 query head 映射到对应 KV head；
2. 遍历 chunk 内每个 token，计算 `Q · K * scale`；
3. 计算当前 chunk 的最大 score 和 softmax 分母；
4. 计算当前 chunk 的加权 V；
5. 使用 online softmax 公式与 `mr[3]` 的历史状态合并；
6. 最后一个 chunk 将加权和除以总分母，写入 `mr[4]`。

当前 CUDA 算子首先保证功能正确性。一个 CUDA block 对应一个 `[batch, query_head]`，核心循环主要由 `threadIdx.x == 0` 执行，还没有使用 cuBLAS、FlashAttention 风格并行化或充分利用 block 内线程。

## 7. FlexGen/Llama 风格推理 runtime

这里的“复用 FlexGen”目前是指：根据 FlexGen 的执行边界和张量布局，提取并重建所需的 Llama 推理路径，而不是直接运行未经修改的原始 FlexGen 项目。

### 7.1 模型配置和权重

- [`python/flexgen_runtime/model_config.py`](python/flexgen_runtime/model_config.py)：读取简化的 Llama `config.json`；
- [`python/flexgen_runtime/weights.py`](python/flexgen_runtime/weights.py)：从独立 `.npy` 文件加载 embedding、RMSNorm、Q/K/V/O 和 MLP 权重。

权重被加载到 `FlexGenWeightLoader.device`，当前全流程测试中该设备是 Guest 的 `cuda:0`。

### 7.2 Prefill

[`python/flexgen_runtime/prefill_runner.py`](python/flexgen_runtime/prefill_runner.py) 的执行顺序是：

```text
token_ids
  -> embedding + attention mask + position ids
  -> 对每一层：
       RMSNorm
       Q/K/V projection
       RoPE
       causal self-attention
       Wo + residual
       post-attention RMSNorm
       MLP + residual
       将该层 K/V 写入 NVM
  -> final RMSNorm
  -> LM Head
  -> next_token_ids
```

Prefill 当前全部在 Guest GPU 上完成。每层产生的 KV 按 FlexGen 形式 `[sequence, batch * heads, head_dim]` 输出，再由 adapter 转换为 CEMU 布局后写入 NVM。

### 7.3 单步 Decode

[`python/flexgen_runtime/decode_runner.py`](python/flexgen_runtime/decode_runner.py) 接收形状为 `[batch, 1]` 的当前 token，并对每一层执行：

```text
embedding
  -> input RMSNorm
  -> Q/K/V projection
  -> 对当前绝对 token position 应用 RoPE
  -> 将新 K/V 追加到 NVM
  -> 调用 attention_backend.decode()
  -> Wo + residual
  -> post-attention RMSNorm
  -> MLP + residual
```

所有 layer 完成后，执行 final norm 和 LM Head，使用 argmax 或 sampling 得到下一 token。

其中 [`python/flexgen_runtime/decode.py`](python/flexgen_runtime/decode.py) 将 Attention 前后的 Guest GPU 计算拆成：

- `prepare_flexgen_decode_attention()`：RMSNorm、Q/K/V、RoPE；
- `finish_flexgen_decode_attention()`：接收 CEMU Attention 输出，执行 Wo、残差和后续 RMSNorm。

这种拆分正好把 CSD offload 边界固定在 Attention 主体。

### 7.4 多步自回归 Decode

[`python/flexgen_runtime/generation.py`](python/flexgen_runtime/generation.py) 实现真正的 token 反馈循环：

```text
Prefill next token
  -> Decode step 0
  -> step 0 输出 token 作为 step 1 输入
  -> step 1 输出 token 作为 step 2 输入
  -> ...
```

每一步的 `token_position`、输入 token、Q/K/V、Attention 输出、hidden states、logits 和输出 token 都可以保留下来用于校验。

### 7.5 PyTorch reference backend

[`python/flexgen_runtime/reference.py`](python/flexgen_runtime/reference.py) 在 Guest GPU 上维护一份独立 KV Cache，并使用 `einsum + softmax` 执行 Dense Attention。

`FlexGenDecodeRunner` 不需要知道后端是 PyTorch 还是 CEMU。因此可以用完全相同的权重、token 和非 Attention 计算，分别运行：

- `FlexGenTorchAttentionBackend`；
- `FlexGenAttentionBackend + CemuAttentionDevice`。

这为逐层、逐步比较提供了可靠 reference。

## 8. 当前完整测试的执行流程

最新的完整测试入口是 [`python/test_flexgen_cemu_pipeline.py`](python/test_flexgen_cemu_pipeline.py)。它包含七个阶段：

1. 创建确定性的两层 Llama 风格小模型和随机权重；
2. 在 Guest GPU 上执行完整 Prefill；
3. 验证 Prefill 生成的每层 KV 已正确写入 CEMU NVM；
4. 用纯 PyTorch backend 执行多步自回归 Decode reference；
5. 用相同 token 和权重执行 CEMU Decode 路径；
6. 从 NVM 读回 Decode 新增 K/V，验证追加位置和数据；
7. 对每个 Decode step、每个 layer 比较两条路径的：
   - 输入 token；
   - Q；
   - K；
   - V；
   - Attention 输出；
   - 最终 hidden states；
   - logits；
   - next token。

最近一次通过的配置为：

| 参数 | 当前测试值 |
|---|---:|
| layers | 3 |
| batch size | 2 |
| prompt tokens | 5 |
| decode steps | 4 |
| hidden size | 32 |
| query heads | 8 |
| KV heads | 2 |
| head dimension | 4 |
| vocabulary size | 24 |
| dtype | float32 |
| staging capacity | 9 tokens/chunk |

四步 Decode 中，每一步都会遍历三个 layer。staging 可以容纳完整的 9 个 token，因此每一步、每一层都只执行一个 chunk，总计执行 12 次 CSD Attention。最终两条路径生成相同序列：

```text
batch 0: [16, 2, 8, 10, 8]
batch 1: [8, 10, 8, 10, 8]
```

测试日志最终报告：

```text
PASS: Guest GPU Prefill
   -> 4 autoregressive Decode steps
   -> CEMU Attention
   -> Wo/MLP/LM head
```

上述结果覆盖了 `query_heads=8、KV_heads=2、hidden_size=32` 的普通 GQA 完整 Guest 路径。测试已经参数化 layer、hidden size、Query/KV head 和 staging token 容量；默认 staging 容量覆盖本次完整序列，不再人为限制为 2 token，仍可通过 `STAGING_TOKENS=2` 强制覆盖多 chunk 路径。

## 9. 测试文件分别覆盖什么

| 测试 | 作用 |
|---|---|
| `test_python_bridge.py` | Python→C++→CEMU ioctl 基础链路 |
| `test_cemu_device.py` | `CemuDevice` 高层封装及 CPU/CUDA vadd |
| `test_kv_layout.py` | KV 偏移、对齐、容量和 chunk 划分 |
| `test_kv_store.py` | NVM K/V 文件创建、读写和持久化 |
| `test_kv_staging.py` | NVM→FDM `copy_file_range` 和 chunk 数据校验 |
| `test_prefill.py` | FlexGen KV 布局适配、Prefill 写入和 Decode 追加 |
| `test_attention.py` | 合成 Q/K/V 下的 CPU/CUDA 多 chunk Attention |
| `test_flexgen_prefill_runtime.py` | 单层 Prefill 数学过程、causal mask 和 RoPE |
| `test_flexgen_full_prefill.py` | 多层 Prefill、MLP、LM Head 和每层 KV 输出 |
| `test_flexgen_decode_runtime.py` | 单步和多步 Decode token 反馈，不依赖 CEMU |
| `test_flexgen_cemu_pipeline.py` | 当前最新、覆盖范围最完整的 CEMU 端到端测试 |

## 10. 常用测试命令

在 Guest 中进入 Python 环境：

```bash
source /root/cemu-gpu-env/bin/activate
cd /root/CEMU/tests/cemu
```

构建 Python bridge：

```bash
make instattention PYTHON=/root/cemu-gpu-env/bin/python3
```

测试纯 PyTorch Prefill：

```bash
make instattention-flexgen-prefill-gpu-test
```

测试纯 PyTorch 单步/多步 Decode：

```bash
make instattention-flexgen-decode-gpu-test
```

运行当前最新完整流程：

```bash
make instattention-flexgen-cemu-gpu-test \
  NVM_TEST_DIR=/mnt/nvme0 \
  FDM_TEST_DIR=/mnt/fdm0 \
  DECODE_STEPS=4
```

执行完整流程前，应确保共享目录中已经存在：

```text
tests/cemu/build/dense_attention_devptr.so
```

CUDA 算子通常在宿主机编译：

```bash
make -C tests/cemu/kernel
```

## 11. 当前已经证明的事情

1. Python 应用可以控制 CEMU program 和 MRS，不需要把整个应用改写为 C++。
2. Prefill 生成的真实张量可以按照 CEMU KV 布局持久化到 NVM。
3. Decode 新 K/V 能按绝对 token position 追加，不会覆盖历史 KV。
4. KV 总量超过 FDM staging 容量时，可以通过多 chunk 串行处理。
5. online softmax 状态能够跨 chunk 正确保留和合并。
6. CSD CUDA 算子能够通过 5 个 `mr_dev_addr` 访问 Query、K、V、State 和 Output。
7. 两层 Decode 都真正调用了 CEMU Attention，而不是只处理最后一层。
8. Prefill 输出 token 已经真正作为 Decode 输入，Decode 输出也会反馈给下一步。
9. CEMU 路径的逐层 Q/K/V、Attention、logits 和 token 与 PyTorch reference 一致。

## 12. 当前还没有完成的事情

### 12.1 还没有接入真实模型权重

当前完整测试使用的是临时生成的两层小模型和确定性随机 `.npy` 权重。它验证的是执行流程和接口正确性，不代表已经运行 Llama-7B、OPT-30B 等真实模型。

### 12.2 权重格式仍是测试格式

当前 loader 要求每个权重单独保存成指定文件名的 `.npy`。真实模型通常使用 Hugging Face `safetensors`、分片 checkpoint 和不同参数命名，需要增加转换或直接加载支持。

当前 LM Head 默认复用 embedding 权重，等价于 tied embedding；接入真实模型时需要读取模型配置并正确处理独立 `lm_head.weight`。

### 12.3 GQA 已完成 runtime 和完整 Guest 支持

当前配置、权重 loader、Prefill、Decode 和 PyTorch reference 已支持 `num_attention_heads` 是 `num_key_value_heads` 整数倍的 GQA。K/V 会按照较少的 KV head 数生成和存储，纯 PyTorch Attention 只在计算时临时展开，CSD 算子直接使用紧凑 K/V 并完成 Query head 到 KV head 的映射。

现有 CPU 测试覆盖 `query_heads=4、KV_heads=2`，完整 CEMU Guest pipeline 已分别验证 `query_heads=2、KV_heads=1` 的 MQA 和 `query_heads=8、KV_heads=2` 的普通 GQA。当前完整测试允许通过 Make 变量切换 MHA、GQA 和 MQA。

### 12.4 Attention ABI 当前只支持 float32

真实 LLM 常用 FP16/BF16。当前 NVM KV 文件、Python ABI、CPU 算子和 CUDA 算子统一使用 float32。直接接入大模型会使权重和 KV 容量显著增加，也不能代表实际吞吐。

### 12.5 当前没有 tokenizer 和真实文本输入

测试直接构造 token ID。真实模型需要 tokenizer，将文本转换为 token，并将输出 token 解码回文本。

### 12.6 权重管理尚未面向大模型

当前每个 Prefill/Decode layer 都调用 loader 读取权重；多步 Decode 会重复加载文件。小测试可以接受，但真实模型需要：

- 权重缓存或分层 offload；
- 明确 GPU/Guest RAM/NVM 的容量策略；
- 异步预取和释放；
- 避免每个 token 重复解析权重文件。

### 12.7 QEMU 资源还不适合重负载

当前 Guest 只有 8 GiB RAM。接入真实模型前需要根据模型参数量、dtype、batch、prompt 和 KV Cache 大小重新配置 Guest RAM、NVM backend 容量和 FDM 容量。

### 12.8 当前模拟时间还没有用于完整性能结论

现阶段重点是功能正确。真实文件访问、NVM→FDM 拷贝、Guest GPU 计算、CSD CUDA mirror 同步和 CEMU 建模时间尚未形成统一的端到端性能模型。

## 13. 接入真实模型权重前的建议顺序

### 第一步：确定第一个真实模型

优先选择较小、结构与当前实现接近的 Llama 模型，并明确：

- layer 数；
- hidden size；
- query/KV head 数；
- head dimension；
- intermediate size；
- vocabulary size；
- dtype；
- 是否 tied embedding；
- 是否使用 GQA。

GQA runtime 已经具备，因此可以选择 TinyLlama 这类较小的 GQA 模型，但第一轮仍应使用较短 prompt 和较小 batch。

### 第二步：增加真实权重 loader

建议在 `flexgen_runtime` 中增加独立 loader，而不是破坏现有 `.npy` 测试 loader：

```text
flexgen_runtime/
├── weights.py                 # 保留确定性单元测试 loader
└── hf_weight_loader.py        # 新增真实模型 loader
```

它需要处理：

- Hugging Face config；
- safetensors 分片索引；
- 参数名映射；
- dtype 转换；
- tied/untied LM Head；
- layer 按需加载。

### 第三步：先做纯 Guest GPU reference

先用真实权重运行：

```text
真实 tokenizer
  -> 真实 Prefill
  -> 1 步 Decode
  -> 多步 Decode
```

并与 Hugging Face Transformers 的输出 logits/token 对齐。这样可以先排除权重映射和模型结构问题。

### 第四步：再接回 CEMU Attention

纯 PyTorch 路径对齐后，再复用现有 `FlexGenAttentionBackend`：

```text
真实 Prefill KV
  -> CEMU NVM
  -> FDM chunk
  -> CEMU Attention
  -> 真实模型后续层
```

逐步验证一层、一步 Decode，再扩大到所有层和多步 Decode。

### 第五步：扩大 QEMU 和存储配置

根据模型容量调整：

- QEMU `-m`；
- vCPU 数；
- CEMU NVM backend 是否改为宿主机 SSD 文件后端；
- FDM 容量；
- staging buffer 大小；
- 单次 chunk token 数。

### 第六步：最后再做性能与 InstAttention 优化

功能跑通后再依次加入：

- 分阶段计时与 CEMU 模拟时间校准；
- NVM→FDM 双缓冲和传输/计算重叠；
- GPU/CSD P2P 或更直接的数据路径；
- delayed KV writeback、X-cache；
- InstAttention 的 SparF/Top-k；
- 按 token/index 保存两份 K Cache。

## 14. 当前阶段结论

当前已经越过了“算子能否运行”和“Python 能否调用 CEMU”的验证阶段，也不再是“Prefill 产生 KV 后用一个合成 Query 测 Attention”。现在已经完成：

```text
Prefill 真实预测 token
  -> 多层 Decode
  -> 每层生成并追加新 KV
  -> 每层 CEMU 多 chunk Attention
  -> Wo/MLP/LM Head
  -> 多步 token 反馈
```

下一阶段的核心不再是继续补小型流程，而是把当前确定性小模型替换为真实模型配置、真实 tokenizer 和真实权重，同时保持现有 PyTorch reference 与 CEMU 路径的逐层校验能力。
