# SparF handoff/admission diagnostics

## Goal

Match InstAttention's dependencies and resource partitioning before comparing
simulated performance. These changes add observations only: no runtime scaling,
NAND geometry, CU count, queue policy, buffer policy or inference result changes.

InstAttention IV-C describes CSD DRAM buffering followed by background group
writes. IV-D describes GPU/CSD cooperation; the public FlexGen multi-batch loop
issues load/compute/store for different `(layer, microbatch)` coordinates. Neither
justifies treating a Guest worker blocked in a synchronous ioctl as target NAND
service. Our four Direct stages remain an implementation of one logical attend,
not four commands claimed to exist in the paper's hardware.

## Findings from the September 15 run

Input timeline: `cemu-logs/sparf-overlap-20260915-090812.jsonl`, QEMU session
`495191614249710`. Only requests with at least 1024 tokens were analyzed.
Programs 5/6 are staged0, 7/8 staged1, and 9/10 buffered. These IDs are specific
to this run: identify them again from each new run's logs.

All modes complete 320 logical attentions and produce identical token sequences.
The existing general timeline analyzer groups staged1 and buffered together by
session/controller/mode; they must be separated before interpreting mode-specific
overlap or microbatch arrival statistics.

The longest buffered SELECT-to-TOKEN_READ gap is the first request (program 9,
workflow 1, layer 0, valid tokens 1025). Relative to completion of command 1857:

| Event on QEMU virtual clock | Time (ms) |
|---|---:|
| program 9 SELECT completes | 0.000000 |
| program 10 SELECT received (command 1858) | 2.038154 |
| program 10 SELECT first scheduled | 413.086312 |
| program 10 SELECT completes | 416.291288 |
| program 9 TOKEN_READ received (command 1859) | 416.689802 |

Thus 411.048158 ms of the gap intersects the peer's received-to-start interval,
while only 3.204976 ms intersects an observed CU job. This is **not** evidence
for 411 ms of NAND service or CU contention. Before `sched_enqueue_job`,
`program_execute` validates metadata, resolves MR pointers and calls
`backend_cuda_ensure_device_ptr`. A missing mirror performs `cudaMalloc`.
The historical trace does not record when the request enters the scheduling
queue, so cold mirror preparation is a candidate, not a proven cause.

At the Guest level, both SELECT and TOKEN_READ share a one-thread Load executor.
Each executes a blocking ioctl. A peer SELECT can therefore occupy this executor
while the previous request's TOKEN_READ waits in its queue. This is consistent
with the observed command order; the old log cannot split Guest callback/queue
time from ioctl completion delivery time.

Not all gaps are cold admission: the second-largest gap is 93.554432 ms at layer
37, tokens 1028; peer pre-start coverage is only 0.021 ms. Do not extrapolate the
first request's diagnosis to all 320 requests.

Buffered token-V reads use fewer NAND pages (821850 versus 835836 in staged1),
but modeled cumulative read duration rises from 1.017935 s to 1.414846 s.
Critical initial LUN backlog rises from 0.000248 s to 0.421596 s. This is evidence
of increased backlog, not proof that background writes alone caused it. Current
observations do not identify which preceding read/write/GC request owns that
backlog. Read-duration sums and CU-job spans are not exclusive service times.

## New optional device observations

With `CEMU_SPARF_TIMELINE` enabled, each successful functional record additionally
contains `dispatch` timestamps:

- Host monotonic: handler entry, mirror preparation start/end, staged-queue enqueue.
- QEMU virtual: handler entry and staged-queue enqueue.
- Existing completion records supply virtual request receipt, first scheduled
  execution and completion.

The virtual received-to-start interval can now be separated into received-to-
handler, handler-to-enqueue, and enqueue-to-first-scheduled. The latter may still
include scheduler-thread delivery delay, not just modeled resource contention.
Host timestamps separately measure mirror preparation and other dispatch work;
never subtract a Host timestamp from a Guest or virtual timestamp.

Timestamps are captured before publishing the job to the scheduling thread.
Only numeric timestamps are collected on the dispatch path; the existing logger
serializes the additional fields after the frozen functional section. With the
timeline disabled there are no additional dispatch clock reads or output writes.
Enabled tracing still has measurement overhead and is for diagnosis, not a final
performance run. No Guest kernel or CUDA operator ABI changes are required.

## Offline analysis

From the Host workspace, for this existing run:

```bash
python3 CEMU/tests/cemu/instattention/experiments/analyze_sparf_handoffs.py \
  cemu-logs/sparf-overlap-20260915-090812.jsonl \
  --programs 9 10 --top 5 --output /tmp/sparf-handoffs-buffered.json
```

Use `--programs 7 8` for staged1. The analyzer unions overlapping intervals,
separates sessions/controllers, and reports missing admission information as
`null`, never as zero. Its CU coverage covers only the selected observed jobs.
Coverage is a timestamp correlation; it does not by itself establish causation.
Gap sums across requests may overlap.

To collect admission data, rebuild QEMU and restart only the test VM with the same
32 GiB Guest, two functional workers, one CU and a fresh timeline filename.
Rerun only the buffered OPT case with the same workload and existing Softmax
ratio file. A new full three-mode calibration is unnecessary for this diagnosis.
Do not warm away the first request: its admission cost is what we need to locate.
Existing logs cannot reconstruct the new timestamps retrospectively.

Regression command (Host, no QEMU/GPU run):

```bash
make -C CEMU/tests/cemu instattention-sparf-stages-test
```
