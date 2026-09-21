# CEMU InstAttention SparF workflow

An experimental single-request autonomous mode is available as ABI v3; see
[SPARF_AUTONOMOUS.md](SPARF_AUTONOMOUS.md) for constraints and the small Guest test.

The current behavior gaps and proposed device-autonomous pipeline are documented
in [INSTATTENTION_BEHAVIOR_ALIGNMENT_PLAN.md](INSTATTENTION_BEHAVIOR_ALIGNMENT_PLAN.md).
That document is a design plan, not an implemented replacement; freezing remains unchanged.

The optional bounded FDM KV buffer and background publication path is described
in [SPARF_BUFFERING.md](SPARF_BUFFERING.md). It preserves the synchronous baseline
and does not change the hardware runtime configuration.

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

## Staged device workflow and isolated FDM slots

`SPARF_STAGED=1` enables an executable version-2 workflow. This uses the existing
NVMe execute ioctl and metadata transport; no new kernel ioctl or pybind API is
required. Version 1 remains the default comparison path. Version 2 requires the
updated CEMU executable and updated Guest Python sources.

Each staged slot has 12 independent FDM ranges. The extra range stores a workflow
sequence id, the canonical metadata, selected token indices and cumulative trace.
Q/K/V, alpha, probabilities and output stay in that slot's existing ranges. The
CEMU continuation checks stage order, workflow id, shapes and extent maps. No
heap pointer or pending FTL request is retained across NVMe command completion.
SELECT can reset an abandoned/error context. Closing a slot releases its state
with the MRS; Python prevents reuse while the previous request is active.

The actual stages are:

1. `SELECT`: query Top-r, channel-K read, approximate score/Softmax and Top-k.
2. `TOKEN_READ`: selected token-K read and current-K overlay.
3. `EXACT_V`: submit selected token-V read, execute exact QK/Softmax, then join
   the V read and overlay current V/mean compensation. This preserves the
   existing device-internal V-read/QK overlap without exporting Top-k to Guest.
4. `PV`: compensated output calculation.

The slot scheduler reuses `CemuAttentionSharedWorkers`: SELECT/TOKEN_READ use Load,
EXACT_V/PV use Compute, and result collection uses Store. Slots are reserved in
the producer with backpressure, not by blocking the Load worker. This allows one
ready request's load stages to overlap another request's computation. Microbatch
requests are submitted only after their own QKV exists. The multi-batch runner
submits the next microbatch before joining the previous one; layers and generated
tokens retain their original dependencies. `ATTENTION_SLOTS` controls the staged
pool independently of microbatch count. No dense next-layer KV prefetch is used.

The original host `femu-scripts/cemu_config_opt13b.json` has `nr_thread=1`:
even requests submitted on independent Guest workers execute on one CEMU
functional worker. For a separate concurrency experiment, the added
`femu-scripts/cemu_config_opt13b_staged.json` changes only `nr_thread` to 2.
Select it with `CEMU_CONFIG_FILE` in the existing host launch command. It keeps
`nr_cu=1`, the existing FTL and all NAND timings. Staged SparF now routes
SELECT/TOKEN_READ to functional worker 0 and EXACT_V/PV to worker 1 when at
least two workers exist. Legacy commands retain the original scheduling policy.
Two threads permit already queued commands to execute concurrently, but do not
guarantee overlap: the current CEMU functional path pauses Guest vCPUs while it
runs, which can delay subsequent Guest submissions. The timing scheduler still
accounts each command on the original compute-unit model. This configuration
therefore tests functional concurrency, not a completed model of independent
target Load/Compute timelines or a demonstrated end-to-end speedup.

With `BATCH_SIZE=4 GPU_BATCH_SIZE=4` there is only one ready microbatch. Two slots
do not remove its layer dependencies; the staged path is a behavior/diagnostic
comparison and may be slower due to four command submissions per workflow. It
is not proof of extra parallelism within that microbatch. This change does not
implement CSD DRAM group buffering, background block writes or a new FTL. Nor
does it infer an additional group/Softmax pipeline from the paper's phrase
“sequential and pipelined” without a more precise dependency specification.

Timing: each stage charges only its own modeled duration; the final trace is
cumulative and is aggregated once per completed Attention. For 40 layers and 16
Decode steps, one microbatch has 640 attention samples and 2560 execute samples
in staged mode. `decode.stage_select_wall`, `decode.stage_token_read_wall`,
`decode.stage_exact_v_wall`, and `decode.stage_pv_wall` are nested within
`decode.execute_wait_wall`. Do not add these nested metrics together. This does
not create a calibrated end-to-end virtual system clock: the workflow trace
still excludes GPU projections/MLP, KV append and the full Guest control path.

Validation on the development machine (no GPU/Guest/checkpoint required):

```bash
cd /home/lihan/cemu/CEMU/tests/cemu
make instattention-sparf-stages-test
```

The test compiles the production C workflow and CPU SparF kernel against a
small deterministic FTL stub. It checks FP16/FP32 against NumPy, equality with
the monolithic path, slot interleaving, ordering, error recovery and no double
charging. It cannot validate real NAND queues, CUDA mirror behavior or GPU
concurrency. The scheduler tests force Load and Compute to progress independently.

Build CEMU on the host, restart the Guest using the new binary and existing
configuration, and synchronize the changed Guest sources. Then run this small
Guest integration test before loading OPT-13B:

```bash
cd /root/CEMU/tests/cemu
make instattention-sparf-stages-device-test \
  NVM_TEST_DIR=/mnt/nvme0 FDM_TEST_DIR=/mnt/fdm0 CSD_TARGET=cuda
```

This compares the legacy workflow and two independent staged slots with NumPy,
then submits two requests through the shared staged scheduler. No ratio report
or checkpoint is needed. After it passes, use the original OPT profiling command
with `SPARF_STAGED=1` and a new `PROFILE_OUTPUT` filename. Keep the same shape and
ratio report for an A/B comparison. To test microbatch overlap separately, retain
batch 4 but set `GPU_BATCH_SIZE=2`; first generate a matching ratio report for
microbatch 2. Changing microbatch size is a separate experiment, not a like-for-like
latency comparison with the original microbatch-4 run.

The previous `prepare CUDA devptr for mr 0` failure happened during argument
preparation, before `run_sparf_workflow`. That message alone does not establish
that K/V read concurrency or shared slots caused it; the underlying CUDA error
still needs to be captured if it recurs. Independent slots provide ownership
isolation, not a demonstrated fix for that allocation/preparation failure.

## KV visibility and dependent read timing

SparF's raw extent reads bypass the Guest page cache. Ordinary `pwrite` and
`MAP_SHARED` updates alone do not publish appended KV to that path. Both
`decode()` and `submit_decode()` now call `store.publish()` before device work.
On Linux this waits with `fdatasync` on token K, channel K and token V; dirty
MAP_SHARED data is included. Clean calls issue no sync, and a failed sync retains
the dirty state and prevents submission. This establishes a correctness baseline
with the current filesystem/FTL; it does not implement CSD DRAM buffering.

`decode.kv_publish_wall` measures that boundary separately from
`decode.nvm_append_wall` and `decode.execute_wait_wall`. It can increase Guest
wall time because dirty writes must now complete before a raw read. Do not
interpret the previous lower time from buffered writes as equivalent behavior.
Prefill still publishes after the asynchronous KV writer has drained.

Within one CEMU command, the virtual clock is frozen while the FTL advances
NAND availability. Reusing `clock_ns()` for channel K, token K and token V gave
dependent requests effectively the same start time; later read latencies could
include earlier work, and the trace then added that work again. The workflow
now starts a local cursor at command execution time:

- Channel K starts at the cursor.
- Token K starts after channel K completion plus approximate computation.
- Token V and exact QK start together after token K completion.
- PV follows both V and QK, as in the existing `max(V, QK)` model.

A subsequent staged command starts at its own execution time, without adding
previous stages' cumulative durations again. Guest submission gaps remain outside
the workflow sum; actual shared NAND contention may still differ across runs.
This fixes the dependent-read accounting, not the complete system virtual clock.

`make instattention-sparf-stages-test` now includes a busy-NAND/frozen-clock
regression, publication failure/retry tests, and repeated append/read checks.
The Guest `instattention-sparf-stages-device-test` additionally runs four decode
appends by default (`DECODE_STEPS=4`). It reads back the raw channel-K buffer
before current-token overlay and compares both device paths against NumPy on
every step. It tests both the requested sparse ratio and ratio 1, which forces
all historical V entries to be consumed. The same sequence also runs locally
against the production CPU workflow with a simulated transport; real Guest
page-cache/CUDA behavior still requires the Guest test.

After updating and rebuilding the host QEMU, restart it and synchronize Guest
sources. Run the same small staged device test first; it needs no OPT checkpoint
or ratio file. Look for `[sparf-sequence] PASS` as well as `[sparf-stages] PASS`.
Then compare OPT runs with identical ratio/configuration and distinct CSV paths.
The previous functional `PASS` only indicated completion, not numerical agreement
between legacy and staged multi-token generation.

## Measuring actual cross-workflow execution overlap

`CEMU_SPARF_TIMELINE=/absolute/host/path.jsonl` enables optional host and virtual-time diagnostics.
`run-csd.sh` forwards it through its inner sudo. With the variable unset, no
timeline file, per-phase clock reads or formatting are performed. When enabled,
each command collects a bounded set of intervals and per-LUN page counters, then
appends a functional JSON line after leaving the frozen section and a completion
line from the device scheduler. Recording adds overhead, so use it for
behavior validation separately from final performance measurements.

Each line identifies QEMU session, controller, program/slot, workflow id,
layer/token count, stage, actual functional worker, configured worker/CU counts,
success, modeled duration and host monotonic intervals. The command interval
excludes freeze entry/exit waits; those have separate timestamps. Read-wait spans
include host handling; QK/PV spans include mirror preparation and kernel calls.
`v_outstanding` is the request lifetime until the join, **not** measured NAND
service time. No GPU/Guest clock is mixed into these host timestamps.

The original host config has one functional worker and cannot execute two
commands simultaneously. Keep the usual host launch arguments but select:

```bash
CEMU_CONFIG_FILE=/home/lihan/cemu/CEMU/femu-scripts/cemu_config_opt13b_staged.json
CEMU_SPARF_TIMELINE=/tmp/opt13b-overlap.jsonl
```

Pass both as environment assignments in the existing `sudo ... ./run-csd.sh`
command when starting the newly built QEMU. The file is appended; the analyzer
separates QEMU sessions/controllers, but a fresh filename is preferable per
experiment. No Guest restart is done automatically.

Inside the Guest, the helper below creates fresh output paths, generates the
microbatch-2 ratio and runs a serial baseline followed by staged execution:

```bash
cd /root/CEMU/tests/cemu
bash instattention/experiments/run_sparf_overlap_ab.sh
```

It defaults to batch 4, microbatch 2, prompt 1024, decode 4. Use `DECODE_STEPS=16`
to extend the run. Both modes use the same microbatch shape, ratio and KV
publication barrier. Successful execution alone does not establish overlap or
equal generation results; compare both token sequences and analyze the host log.

On the host after completion:

```bash
cd /home/lihan/cemu/CEMU
python3 tests/cemu/instattention/experiments/analyze_sparf_timeline.py \
  /tmp/opt13b-overlap.jsonl --min-tokens 1025 \
  --output /tmp/opt13b-overlap-summary.json \
  --chrome-trace /tmp/opt13b-overlap-trace.json
```

The analyzer reports the union duration where distinct workflows in the same
layer/token have overlapping Load/Compute command intervals, and separately
where channel/token-K read-wait overlaps another workflow's QK. It excludes
same-workflow V/QK nesting, endpoints merely touching, and different controllers
or QEMU sessions. It checks stage ordering and reports incomplete/failed work.
`--require-overlap` returns nonzero if overlap is absent or workflows are
incomplete. The Chrome Trace JSON can be opened in a compatible trace viewer.

`NOT OBSERVED` is a valid diagnostic result, not proof that the detector failed.
Host execution without overlap does not establish serialization on the virtual
time axis: functional execution freezes the virtual clock. Use the virtual
completion/submission records below before changing the submission protocol. Worker routing does
not add modeled CUs, implement independent target resource timelines, or prove a
speedup. CSD DRAM grouping/background writeback remains separate pending work.


## Read-path and virtual-scheduler diagnostics

The same environment variable now enables two clock domains, kept separate:

* Functional records retain the old host intervals, plus a `virtual` object
  recording QEMU virtual time at functional entry/exit. This directly measures
  whether virtual time advanced during host execution; no host duration is
  subtracted to infer virtual time.
* Completion records have `event=completion`, `clock=qemu_virtual_ns`, the same
  session/controller/command identity, request reception, first scheduling and
  device scheduler completion timestamps, charged runtime and final CU. This is
  **not** the Guest ioctl-return or CQ-consumption timestamp. FIFO jobs currently
  include read waits in their charged runtime; do not label these intervals pure
  compute service. First-start to finish is not pure service under preemption.

Each channel-K/token-K/token-V read includes useful logical bytes (before page
rounding), submitted bytes, actual mapped NAND page operations, unmapped pages,
geometry and per-LUN page counts. Counters are updated in the real FTL read path.
A per-read array is allocated only when diagnostics are enabled and released
once that read has completed and its command observation has been written.
There is no new Guest ABI or required GPU kernel rebuild for these counters.

`service_ns` and `queue_ns` are sums across page operations, **not elapsed read
latency**; parallel LUNs and successive page waits overlap. For the page that
finishes latest, the report splits queue time into LUN backlog at its first
access (`critical_initial_wait_ns`) and remaining queue time. With the default
non-interleaved FTL path, remaining wait includes earlier pages of this read on
that LUN. Initial backlog can include earlier simulated reads/writes; it is not
host syscall overhead. `ideal_page_gbps` is channels × LUNs/channel × page bytes /
page-read latency, a full-page upper bound of this model, not achieved useful-KV
bandwidth or the paper's configured channel bandwidth. Duplicate/unmapped pages,
layout imbalance and other traffic can make useful throughput lower.

The analyzer joins functional/completion records even when log lines arrive out
of order. Old logs remain supported but explicitly report these diagnostics as
unavailable. It reports signed gaps between the previous microbatch's device
completion and the next microbatch's reception, within the same layer/token.
Negative gaps establish concurrent requests in flight, **not** NAND/CU service
overlap. It also reports gaps between dependent stages and scheduler queue time.

Start the rebuilt host QEMU with a fresh timeline filename, synchronize Guest
Python sources, then run the existing small test (no checkpoint/ratio needed):

```bash
cd /root/CEMU/tests/cemu
make instattention-sparf-stages-device-test \
  NVM_TEST_DIR=/mnt/nvme0 FDM_TEST_DIR=/mnt/fdm0 CSD_TARGET=cuda DECODE_STEPS=4
```

If that passes, use the same `run_sparf_overlap_ab.sh` as above. It now prints a
compact comparison of the two CSVs after token equality passes. The ratio file
is generated automatically in the durable run directory. To analyze the OPT run
on the host (filtering out the small fixture):

```bash
cd /home/lihan/cemu/CEMU
python3 tests/cemu/instattention/experiments/analyze_sparf_timeline.py \
  /tmp/opt13b-diagnostics.jsonl --min-tokens 1025 --require-model \
  --output /tmp/opt13b-diagnostics-summary.json \
  --virtual-trace /tmp/opt13b-diagnostics-virtual.json \
  --chrome-trace /tmp/opt13b-diagnostics-host.json
```

`--require-model` rejects old/missing/incomplete diagnostics. Use a fresh filename
so older sessions do not cause this check to fail. Virtual and host Chrome traces
are separate artifacts; never compare their absolute timestamps. Logging adds
work to the scheduler, so these runs diagnose behavior, not final performance.

Guest profiling adds `kv_append.pack_wall`, `token_k_pwrite_wall`,
`token_v_pwrite_wall`, `channel_k_mmap_wall` under the `kv_append.` prefix and
`kv_publish.{token_k,channel_k,token_v}_fdatasync_wall`. Append measurements are
aggregated once per append, without per-head profiler locking or logging; no
extra timestamp calls run when profiling is disabled. Byte counts describe
logical data, not filesystem/NAND traffic. Publication metrics include both
prefill and decode calls; their sums are nested within the existing phase
metrics, not additive to them. They time operations including any blocking,
not syscall dispatch overhead in isolation. Data visibility barriers remain.

These changes instrument the existing timing and data paths. Independent NAND/CU
resource scheduling, a CSD DRAM KV buffer and background grouped writeback still
require implementation after the measured bottlenecks are checked. The small
fixture checks correctness; it is not a saturated NAND bandwidth calibration.
