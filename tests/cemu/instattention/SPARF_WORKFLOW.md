# CEMU InstAttention SparF workflow

The SparF path follows the dataflow in InstAttention Algorithm 1 while keeping
one Guest `execute` for each logical Decode Attention request.

## Persistent SSD layout

Each GPU microbatch owns three files:

1. `k_token`: K grouped by layer, batch, KV head, token, channel.
2. `k_channel`: the second K copy grouped by layer, batch, KV head, channel, token.
3. `v_token`: V grouped by layer, batch, KV head, token, channel.

Prefill writes both K organizations. Every Decode append updates both K files
and the token-indexed V file. A running V mean is maintained for SparF's
unselected-token compensation term.

## One device request

The Guest writes Q and the running V mean, then submits one SparF workflow:

1. CEMU selects the Top-r absolute query channels.
2. CEMU translates those logical channel ranges through the channel-K extent map.
3. The FTL reads only selected channel data.
4. CEMU computes approximate scores and selects Top-k tokens, always retaining the current token.
5. The FTL reads selected full K vectors from token-K.
6. CEMU submits selected V reads and runs exact QK/Softmax concurrently.
7. PV starts after exact QK and V are both complete.
8. The device applies `alpha * softmax(QK)V + (1-alpha) * mean(V)`.

The modeled critical path is:

```text
channel-K read + approximate selection + token-K read
+ max(exact QK/Softmax, token-V read) + compensated PV
```

Actual CPU/CUDA functional time is separated from this modeled runtime. The
Figure 14 runner derives compute phases from the paper-calibrated CSD model and
uses the CEMU FTL latency for selected SSD reads.

## Validation order

Run these in the Guest after rebuilding/restarting CEMU with
`internal_bandwidth=0`:

```bash
make instattention-sparf-test
make instattention-sparf-kv-test

make instattention-sparf-device-test \
  NVM_TEST_DIR=/mnt/nvme0 \
  FDM_TEST_DIR=/mnt/fdm0 \
  CSD_TARGET=cuda

make instattention-opt-cemu-gpu-test \
  MODEL_DIR=/root/models/OPT-13B \
  NVM_TEST_DIR=/mnt/nvme0 \
  FDM_TEST_DIR=/mnt/fdm0 \
  CSD_TARGET=cuda \
  ATTENTION_MODE=sparf \
  RUNTIME_CONFIG=instattention/experiments/configs/opt13b_sparf_1csd.json \
  FUNCTIONAL=1 \
  BATCH_SIZE=1 \
  PROMPT_LENGTH=128 \
  DECODE_STEPS=4

make instattention-figure14-sparf-functional \
  MODEL_DIR=/root/models/OPT-13B \
  NVM_TEST_DIR=/mnt/nvme0 \
  FDM_TEST_DIR=/mnt/fdm0 \
  CSD_TARGET=cuda \
  BATCH_SIZES=4 \
  OUTPUT_LENGTH=8
```

After the smoke run, remove `OUTPUT_LENGTH=8`, then expand `BATCH_SIZES` one at
a time. The Figure 14 workload uses prompt length 1024 and output length 1024.
The reproduction config currently uses Top-r 16 and the paper's default SparF
compression ratio of 1/8.

## Softmax runtime calibration

The paper's 164 us value is for one attention head and 16 tokens. The runtime
model keeps that value as an anchor and replaces only shape scaling with ratios
measured on the host GPU. Generate the complete OPT-13B decode table first:

```bash
cd /root/CEMU/tests/cemu
make instattention-softmax-ratio-benchmark \
  RUNTIME_CONFIG=instattention/experiments/configs/opt13b_sparf_1csd.json \
  SOFTMAX_DEVICE=cuda:0 GPU_BATCH_SIZE=4 \
  PROMPT_LENGTH=1024 DECODE_STEPS=1023 \
  SPARF_COMPRESSION_RATIO=8 \
  SOFTMAX_RATIO_FILE=/tmp/opt13b-softmax-ratios.json
```

This measures isolated FP32 CUDA Softmax. It does not claim that GPU ratios are
FPGA performance ratios; the paper anchor still sets the absolute CSD time.
The pipeline rejects any shape absent from the report instead of silently using
linear scaling.

Pass the report to Figure 14:

```bash
make instattention-figure14-sparf-functional \
  MODEL_DIR=/root/models/OPT-13B \
  NVM_TEST_DIR=/mnt/nvme0 FDM_TEST_DIR=/mnt/fdm0 \
  CSD_TARGET=cuda BATCH_SIZES=4 \
  SOFTMAX_RATIO_FILE=/tmp/opt13b-softmax-ratios.json
```

Use `OUTPUT_LENGTH=8 DRY_RUN=1` first to inspect the child command. Run a short
functional test, compare the modeled trace with legacy linear scaling, and then
expand the batch-size sweep.

## Guest GPU / CEMU boundary profiling

The current SparF adapter copies Guest CUDA Q/K/V to CPU NumPy arrays, writes
FDM buffers, executes the CEMU request, and copies the output back to the Guest
GPU. VFIO passthrough of the Guest GPU does not make this path GPU-to-CSD P2P.

After updating the Python sources in the Guest, rerun the same short workload:

```bash
cd /root/CEMU/tests/cemu
set -o pipefail
make instattention-opt-cemu-gpu-test \
  MODEL_DIR=/root/models/OPT-13B \
  NVM_TEST_DIR=/mnt/nvme0 FDM_TEST_DIR=/mnt/fdm0 \
  CSD_TARGET=cuda ATTENTION_MODE=sparf \
  RUNTIME_CONFIG=instattention/experiments/configs/opt13b_sparf_1csd.json \
  FUNCTIONAL=1 BENCHMARK= TRACE_FILE= \
  BATCH_SIZE=4 GPU_BATCH_SIZE=4 PROMPT_LENGTH=1024 DECODE_STEPS=16 \
  ATTENTION_SLOTS=2 SPARF_TOP_R=16 SPARF_COMPRESSION_RATIO=8 \
  SOFTMAX_RATIO_FILE=/tmp/opt13b-b4-p1024-d16.json \
  PROFILE_OUTPUT=/tmp/opt13b-sparf-d16-boundary-profile.csv \
  2>&1 | tee /tmp/opt13b-sparf-d16-boundary.log
```

This uses the existing ratio report, which must cover all requested shapes.
Profiling does not require a CEMU rebuild or a different runtime model. Functional
mode suppresses per-layer details and prints the aggregate metrics at the end.

| Metrics | Interpretation |
| --- | --- |
| `pipeline.decode_wall` | Guest end-to-end Decode wall time |
| `decode.qkv.*`, `decode.mlp.*`, `decode.wo_norm_residual.*`, `decode.output_head.*` | GPU operation caller wall time and CUDA stream intervals |
| `decode.kv_to_cpu_wait_wall`, `decode.query_to_cpu_wait_wall` | Q/K/V conversion and GPU-to-CPU copies, including pending GPU work they wait for |
| `decode.output_to_gpu.*` | Output conversion and CPU-to-GPU copy |
| `decode.nvm_append_wall`, `decode.value_mean_wall` | Guest KV store updates and mean preparation |
| `decode.fdm_write_wall`, `decode.metadata_wall`, `decode.output_read_wall`, `decode.trace_read_wall` | Guest FDM and request preparation/readback |
| `decode.execute_wait_wall` | Whole blocking CEMU execute call; includes execution, internal transfer and completion overhead |
| `csd_model.*` | Modeled CSD workflow components, independent of Guest wall measurements |

CUDA events are collected without a synchronization at every phase; outstanding
events are synchronized at the final summary. Event intervals can include stream
idle time and stalls; they are not isolated kernel timings. Instrumentation itself
also adds some overhead. Do not add CUDA times, host times and modeled times, or
subtract the modeled sum from Guest wall time and label the remainder copy time.

For 40 layers and 16 Decode steps with one microbatch, the per-layer Decode
metrics should have 640 samples and the output head should have 16. If execute
wait dominates, further host-side instrumentation is needed to separate CEMU
CUDA execution, mirror copies, and scheduling; this profile alone cannot assign
that entire interval to missing P2P.

`pipeline.prefill_forward_wall` includes KV submissions and waits as well as
Transformer forward work. `prefill.kv_to_cpu_wait_wall` and
`prefill.nvm_write_wall` may overlap the forward pass and each other across
workers. The final KV flush is measured separately.
