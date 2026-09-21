# Grouped OPT runbook

This run uses the new device-owned grouped KV path. It is opt-in; omit
`SPARF_GROUPED=1` to run the old SparF path for A/B. Do not run it until QEMU
has been rebuilt and the VM has been restarted with that binary.

The preflight now checks the full FDM allocation: raw tail, writeback scratch,
shared command/status MRs and all Attention slots, with 4 KiB file allocation
rounding. Current FDMFS uses `simple_statfs` and reports zero total/free bytes;
that means unknown capacity, not a full device. Pass
`SPARF_GROUP_FDM_BUDGET_BYTES` explicitly in this case. It is the **available
budget for this run**, not the namespace size: subtract existing allocations,
including old mount-script placeholder files. Allocation still enforces the
actual device limit; the budget neither resizes nor bypasses FDM.

The commands below assume the 2 GiB FDM configuration and at least 1 GiB
available for the new run, so use `1073741824` as a conservative budget.
Verify that assumption against the active configuration and existing files.
If FDMFS later reports real free space, both that space and an explicit budget
are enforced. Missing mounts and genuinely full filesystems remain errors.

## 1. Host: rebuild and start CEMU after the normal VM restart

Build the host binary before restarting the VM:

```bash
cd /home/lihan/cemu/CEMU
ninja -C build qemu-system-x86_64
```

Run from the host CEMU build directory:

Confirm GPU 1 is available and the passthrough BDF is still correct first.
If the host still needs the private NVIDIA 575 libraries, pass that library
path explicitly through sudo as shown; do not change the system driver.

```bash
cd /home/lihan/cemu/CEMU/build
mkdir -p /home/lihan/cemu/cemu-logs
sudo env LD_LIBRARY_PATH="/home/lihan/cemu/nvidia-575-libs${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 \
  CEMU_CUDA_SYNC=1 CEMU_CUDA_DEVICE=0 \
  CEMU_COMPUTE_LOG=progress CEMU_COMPUTE_LOG_INTERVAL=100 \
  CEMU_SPARF_AUTONOMOUS_TIMELINE=/home/lihan/cemu/cemu-logs/opt13b-grouped-autonomous.jsonl \
  CEMU_SPARF_READ_DIAGNOSTICS=1 \
  CEMU_CONFIG_FILE=/home/lihan/cemu/CEMU/femu-scripts/cemu_config_opt13b_staged.json \
  CEMU_NVM_BACKEND_DIR=/data/lihan/cemu-nvm-opt13b \
  CEMU_CSD_COUNT=1 CEMU_GUEST_MEMORY=32G \
  CEMU_VFIO_GPU=0000:27:00.0 \
  bash ../femu-scripts/run-csd.sh
```

The launch must keep the existing NVM backend and FDM mounts. The QEMU process
is not restarted by the test command.

## 2. Guest: build and run small regressions first

```bash
source /root/cemu-gpu-env/bin/activate
cd /root/CEMU/tests/cemu
make instattention
make instattention-sparf-group-capacity-test instattention-sparf-grouped-backend-test
make instattention-sparf-group-buffer-device-test \
  NVM_TEST_DIR=/mnt/nvme0 FDM_TEST_DIR=/mnt/fdm0 \
  SPARF_GROUP_WRITE_BATCH_PAGES=256
make instattention-sparf-group-attention-device-test \
  NVM_TEST_DIR=/mnt/nvme0 FDM_TEST_DIR=/mnt/fdm0 CSD_TARGET=cuda
```

These Host-compatible unit tests check unknown/full statfs, complete MR
budgets, microbatch stream isolation and shared scheduler lifetime. They do
not replace the actual Guest test or prove physical overlap.

The grouped OPT path now sends `APPEND_BATCH` across contiguous heads. The
device maintains a bounded ready queue and coalesces ready groups into writes
of at most `SPARF_GROUP_WRITE_BATCH_PAGES` pages (default 256). This increases
the shared scratch from 64 KiB to 1 MiB for FP16 D=128 C=8; the capacity
preflight includes it. No target runtime, freezing, or FTL wait is bypassed.

For B=GPU_BATCH_SIZE=4, 40 layers/heads, P=1024 and D=16 decode steps, the
planned Decode APPEND count drops from 102400 per-head commands to 640 batch
commands. The Attention request count and KV contents are unchanged. With
the 1 MiB command buffer, Prefill uses 5120 batch commands; after terminal
drain the cumulative `append_commands` should be 5760 and `appended_tokens`
should be 6656000 (tokens summed over all head streams). Logs now include
`grouped writeback statistics` with successful write batch count and maximum
pages per batch. These counters are queried once after the measured drain,
not logged per write. `max_write_pages` is a capacity bound, not a promise
that every actual batch fills it.

This is not yet block ownership / PPA placement, multiple concurrent write
batches, or two Attention engines. Do not interpret a faster run as proof of
full paper alignment. Keep the same softmax ratio, input sizes and target
configuration for the comparison; inspect submission counts, KV correctness
and the target timeline before latency. Rebuild/restart QEMU normally before
using the new opcodes; do not restart the physical host for this change.

## 3. Guest: short real-model smoke before the long run

```bash
mkdir -p /root/cemu-logs
set -o pipefail
for ratio in 1 8; do
  make instattention-opt-cemu-gpu-test \
    MODEL_DIR=/root/models/OPT-125M \
    NVM_TEST_DIR=/mnt/nvme0 FDM_TEST_DIR=/mnt/fdm0 \
    CSD_TARGET=cuda ATTENTION_MODE=sparf \
    FUNCTIONAL=1 BENCHMARK=0 TRACE_FILE= PROMPT_TEXT= \
    SPARF_AUTONOMOUS=1 SPARF_GROUPED=1 SPARF_STAGED=0 \
    SPARF_KV_BUFFER_TOKENS=0 SPARF_DEVICE_WRITEBACK=0 \
    RUNTIME_CONFIG= SOFTMAX_RATIO_FILE= \
    BATCH_SIZE=2 GPU_BATCH_SIZE=1 PROMPT_LENGTH=16 DECODE_STEPS=4 \
    ATTENTION_SLOTS=2 SPARF_TOP_R=16 SPARF_COMPRESSION_RATIO="$ratio" \
    SPARF_GROUP_CHANNELS_PER_PAGE=8 SPARF_GROUP_POOL_BYTES=0 \
    SPARF_GROUP_FDM_BUDGET_BYTES=1073741824 \
    PROFILE_OUTPUT="/root/cemu-logs/opt125m-grouped-r${ratio}.csv" \
    2>&1 | tee "/root/cemu-logs/opt125m-grouped-r${ratio}.log" || break
done
```

This uses two independent microbatches and no runtime model, deliberately
isolating functional integration from timing calibration. It exercises the
16-token page boundary. `FUNCTIONAL PASS` means the path completed, not a
full OPT logits/KV numerical comparison; keep the operator/reference tests.

## 4. Guest: longer OPT-13B run and separate microbatch check

Measure the shape-specific ratios first:

```bash
make instattention-softmax-ratio-benchmark \
  RUNTIME_CONFIG=instattention/experiments/configs/opt13b_sparf_1csd.json \
  SOFTMAX_DEVICE=cuda:0 GPU_BATCH_SIZE=4 PROMPT_LENGTH=1024 DECODE_STEPS=16 \
  SPARF_COMPRESSION_RATIO=8 \
  SOFTMAX_RATIO_FILE=/root/cemu-logs/opt13b-b4-p1024-d16.json
```

```bash
make instattention-opt-cemu-gpu-test \
  MODEL_DIR=/root/models/OPT-13B \
  NVM_TEST_DIR=/mnt/nvme0 FDM_TEST_DIR=/mnt/fdm0 \
  CSD_TARGET=cuda ATTENTION_MODE=sparf \
  RUNTIME_CONFIG=instattention/experiments/configs/opt13b_sparf_1csd.json \
  SOFTMAX_RATIO_FILE=/root/cemu-logs/opt13b-b4-p1024-d16.json \
  FUNCTIONAL=1 SPARF_AUTONOMOUS=1 SPARF_GROUPED=1 \
  BENCHMARK=0 TRACE_FILE= PROMPT_TEXT= SPARF_STAGED=0 \
  SPARF_KV_BUFFER_TOKENS=0 SPARF_DEVICE_WRITEBACK=0 \
  BATCH_SIZE=4 GPU_BATCH_SIZE=4 PROMPT_LENGTH=1024 DECODE_STEPS=16 \
  ATTENTION_SLOTS=2 SPARF_TOP_R=16 SPARF_COMPRESSION_RATIO=8 \
  SPARF_GROUP_CHANNELS_PER_PAGE=8 \
  SPARF_GROUP_POOL_BYTES=6442450944 \
  SPARF_GROUP_FDM_BUDGET_BYTES=1073741824 \
  PROFILE_OUTPUT=/root/cemu-logs/opt13b-grouped-d16-profile.csv \
  2>&1 | tee /root/cemu-logs/opt13b-grouped-d16.log
```

`SPARF_GROUP_POOL_BYTES=6442450944` is a 6 GiB private pool; for
`prompt=1024, decode=16` the planner needs about 5.1 GiB of grouped pages and
about 850 MiB of FDM tail plus scratch. The planner runs before checkpoint
weights are loaded and fails early if the selected filesystems are too small.
The final log should contain `grouped KV terminal drain complete`, a modeled CSD
summary, and `PASS`. `pipeline.decode_wall` is the Guest end-to-end Decode
interval; `csd_model.total_model_ns` is the modeled Attention sum and is not an
end-to-end latency.

`BATCH_SIZE=4 GPU_BATCH_SIZE=4` has only one microbatch: it cannot establish
inter-microbatch GPU/CSD overlap. For that check, regenerate the ratio file
with `GPU_BATCH_SIZE=2` (a separate filename), and repeat the model command
with `BATCH_SIZE=4 GPU_BATCH_SIZE=2`, the new ratio file and new log/CSV names.
The shared store capacity is unchanged; per-slot Attention workspaces shrink.
Verify actual trace intervals rather than inferring overlap from two slots.

APPEND_BATCH is one synchronous Guest transaction per tile of contiguous KV
streams/tokens. OPT-13B batch=GPU_BATCH_SIZE=4 now issues 40 APPEND_BATCH
commands per Decode step. With profiling enabled, `prefill.grouped_append_wall` and
`decode.grouped_append_wall` include CPU staging and APPEND/backpressure wait,
aggregated once per layer/microbatch, not per head. They exclude mean updates
and are not NAND-only target times; do not add them to pipeline wall time.

## 5. Optional old-path A/B

Repeat the model command in step 4 with `SPARF_GROUPED=1` removed (and `SPARF_AUTONOMOUS=1` kept) to
compare the legacy file-backed autonomous path. Keep both profile CSVs and the
same ratio file.

Useful checks after either run:

```bash
column -s, -t /root/cemu-logs/opt13b-grouped-d16-profile.csv | less -S
# Run this on the host (the QEMU process writes the timeline on the host):
cd /home/lihan/cemu/CEMU/tests/cemu
python3 instattention/experiments/analyze_sparf_autonomous.py \
  /home/lihan/cemu/cemu-logs/opt13b-grouped-autonomous.jsonl
# --trace-file cannot be combined with FUNCTIONAL.
```

## 6. Validation of the preflight fixes

70 Host regression tests passed after these fixes, including the production
grouped C storage/workflow harness with a fake FTL, the shared-slot scheduler,
capacity checks and OPT entrypoint preflight tests. The Host extension was built
in a separate `/tmp/cemu-grouped-review-*` directory, not over `build-guest`.
No live Guest OPT run was performed during this change. No device ABI, freezing,
FTL timing or GPU binding was changed. The subsequent APPEND_BATCH change is
described in section 2; this paragraph records the earlier preflight validation.

## 7. Grouped read and FTL wait diagnostics

## 8. Grouped write-placement A/B

The grouped write worker supports a placement-only experiment.  The default
`CEMU_SPARF_GROUP_WRITE_ORDER=fifo` preserves the historical ready-queue order.
Set `CEMU_SPARF_GROUP_WRITE_ORDER=kind` on the **Host QEMU launcher** to place
the same selected pages in TOKEN_K, CHANNEL_K, TOKEN_V kind-major order before
the sequential grouped pool is consumed.  Batch size, logical mappings,
contents, FTL latency and freezing are unchanged.  The launcher forwards this
variable through sudo.  A fresh grouped pool is required for each run.

Run the exact same workload twice, changing only this variable:

```bash
# baseline
CEMU_SPARF_GROUP_WRITE_ORDER=fifo ... bash ../femu-scripts/run-csd.sh
# placement experiment
CEMU_SPARF_GROUP_WRITE_ORDER=kind ... bash ../femu-scripts/run-csd.sh
```

Keep separate autonomous trace and Guest profile files.  Compare
`lun_pages`, active-LUN count, maximum-to-mean LUN load, submitted bytes and
target read time.  `FUNCTIONAL PASS` plus equal output checks must hold in both
runs.  This is a causal test of physical page ordering; do not combine it with
runtime, compression, batching or NAND-latency changes.

`CEMU_SPARF_READ_DIAGNOSTICS=1` is opt-in and requires
`CEMU_SPARF_AUTONOMOUS_TIMELINE` to be set when starting the rebuilt **Host
QEMU**. The launch in section 1 enables both. These variables must pass through
sudo; exporting them only in the Guest or attaching them to `make` is ineffective.
Both launcher copies in this checkout forward the switch; use the versioned
`../femu-scripts/run-csd.sh` for reproducibility. No driver/module rebuild is needed.
Do not start a second QEMU against the live image; finish the current experiment
and shut down the Guest normally before starting the rebuilt binary.

First run the small grouped Attention device regression in section 2. Then
repeat **exactly** the B4/GPU_BATCH_SIZE4/P1024/D16 command and ratio file in
section 4. Do not change runtime, geometry, buffer sizes, or microbatch splitting
in this diagnostic comparison. This is a diagnostic run, not a low-overhead
performance benchmark. Disable the new switch (or unset it) for the latter.

On the Host, analyze only the most recent QEMU session and OPT token lengths,
excluding small regression requests that may share the trace file:

```bash
cd /home/lihan/cemu/CEMU/tests/cemu
python3 instattention/experiments/analyze_sparf_autonomous.py \
  /home/lihan/cemu/cemu-logs/opt13b-grouped-autonomous.jsonl \
  --latest-session --min-tokens 1025 --max-tokens 1040 \
  --require-read-diagnostics \
  --output /home/lihan/cemu/cemu-logs/opt13b-grouped-read-requests.json \
  --read-summary /home/lihan/cemu/cemu-logs/opt13b-grouped-read-summary.json
```

The expected complete Attention count is **640 = 40 layers × 16 steps**.
The latest-session filter includes all controllers for that QEMU session; use
a fresh timeline filename for repeated runs with identical lengths in one VM.
Old logs remain analyzable without `--require-read-diagnostics`, but missing
counters are reported as unavailable, never evidence of zero reads.

Each completed grouped autonomous request now carries three optional `io`
objects (`channel_k`, `token_k`, `token_v`). No per-page log writes are added.
Observations live with the suspended autonomous request, not on a worker stack,
and are freed after the physical I/O has completed. With diagnostics disabled,
the FTL observation pointer stays NULL and no per-LUN arrays are allocated.

Field interpretation:

- `requested_bytes = logical_bytes + tail_bytes`: requested selected KV bytes;
  `logical_bytes` is the Flash-bound portion before alignment; `tail_bytes`
  comes directly from the bounded FDM tail. Counts include repeated logical
  requests from query heads (GQA), not just unique KV content. The V-mean row is
  not an NVM read and is excluded. `tail_copies` counts element copies for
  channel-K and head-vector copies for token-K/V, so compare bytes across phases.
- `logical_spans`, `aligned_bytes_before_dedup`, `submitted_bytes`, `ranges`:
  sparse spans, 4-KiB-rounded demand before deduplication, actual submitted read
  buffer bytes after deduplication, and physical extents submitted to the FTL.
  Deduplication is within one read phase, not a cross-request cache. The ratio
  submitted/logical can be below one when repeated requests share a page.
- `pages`/`unmapped_pages`: FTL mapped/unmapped NAND-page visits. They are not
  interchangeable with extent count or the 4-KiB transfer count if NAND geometry
  differs. Any unmapped pages in this initialized workload need investigation.
- `lun_pages` and `lun_initial_wait_ns`: indexed by
  `channel * luns_per_channel + lun`; the analyzer also produces `channel_pages`.
  The first-page wait includes earlier work already occupying that LUN, not
  necessarily this request's own pages. Tail-only phases have empty geometry
  arrays because no FTL request was issued.
- `critical_initial_wait_ns`, `critical_subsequent_queue_ns`, and
  `critical_service_ns` split the longest observed page path. Subsequent wait
  includes serialization after the first visit, possibly other interleaved
  work; it must not automatically be attributed only to this request. The
  active `ssd_advance_status` read path currently models LUN availability; its
  channel-transfer block remains disabled. These observations do not enable or
  replace that model. Per-page service/queue **sums are not elapsed latency**.
- `model_ns` is deadline minus modeled submission time. Host submit→observed
  and observed→ready intervals include emulation/scheduling/waiting, not just
  SSD time. `virtual_deadline_to_ready_ns` measures observed virtual lateness.
  QK/V target overlap is still calculated from actual CU intervals and the V
  read interval, not `min(QK_duration, V_duration)`.

Use these counters to distinguish read amplification, LUN imbalance/backlog,
and Host-side waiting. Neither a long Host wait nor a high page-count sum alone
proves that the simulator's target latency is incorrect. This patch changes no
storage placement, scheduling, target runtime, FTL deadlines, or freezing.
