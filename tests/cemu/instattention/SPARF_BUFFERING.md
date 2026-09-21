# SparF: bounded KV visibility before Flash writeback

For the September 15 A/B results and optional device admission measurements,
see [SparF handoff diagnostics](SPARF_HANDOFF_DIAGNOSTICS.md).

## Purpose and limits

An opt-in device-internal group writeback path is now available for the small
test; see [B1 device writeback](SPARF_DEVICE_WRITEBACK.md). The description below
remains the default Guest-managed data path. B1 removes writeback data round trips,
but does not yet transfer buffer bookkeeping or suffix snapshots to the device.

This is a behavioral comparison path, not a throughput calibration. It separates
KV acceptance/read visibility from completed Flash publication, following the
DRAM-buffer/background-write distinction in InstAttention IV-C. Existing runtime
parameters, Softmax ratios, NAND geometry and CU count are unchanged.

`SPARF_KV_BUFFER_TOKENS=0` retains the original synchronous append/publish path.
A value of at least 2 enables a bounded per-layer FDM ring in functional mode.
The production CSD workflow reads the published NVM prefix and the buffered suffix.
The suffix is not merely the current token: all uncommitted historical tokens
participate in channel selection, approximate scores, exact QK and PV.

The FDM file is the backing store for uncommitted KV. A single Guest worker per
microbatch implements the background writeback controller, using the existing
NVM filesystem and FTL path. This is **not** a native QEMU-controller writeback
engine. A request snapshots its layer's suffix from FDM and transfers it into the
slot's current-K/current-V ranges; these additional emulation copies are retained
in the I/O/profile measurements. They are not paper P2P transfers.

This change does not implement the paper's block placement/GC policy or independent
target I/O/compute resource scheduling. In particular, two functional workers still
do not mean two modeled CUs. Do not claim the complete paper pipeline or its timing
has been reproduced solely because the buffered test passes.

## Dependencies and storage lifetime

For each layer/microbatch:

1. Prefill writes and publishes its complete prefix before Decode.
2. Decode writes a new K/V pair into a fixed-size FDM ring. Acceptance returns
   after the FDM write, not after Flash publication. No per-request `fdatasync`
   runs on the producer in buffered mode.
3. Half a ring of new tokens triggers a background group write. The worker updates
   token K, channel K and token V, then publishes all three files.
4. Only successful publication advances that layer's committed prefix. Until
   then, its tokens remain pinned in FDM. A full ring applies backpressure.
5. Attention snapshots the committed prefix boundary and the remaining suffix.
   A completed writeback or ring reuse cannot mutate a submitted request snapshot.
6. The CSD excludes buffered tokens from logical NAND spans, overlays their FDM
   values, and keeps the existing `max(QK, V-read)` dependency. Page rounding of
   a prefix read can still physically include unused bytes beyond the boundary.
7. Explicit flush drains partial groups and propagates background errors. A
   failed writeback does not advance the committed boundary or free its capacity.

Per-microbatch ring allocation is:

```
align4096(layers * capacity_tokens * align512(2 * gpu_batch * kv_heads * head_dim * element_bytes))
```

FDMFS uses direct I/O even without `O_DIRECT`. Ring transfers use page-aligned
NumPy storage with `preadv`/`pwritev`, sector-aligned lengths and offsets. The
producer reuses one write buffer under its condition lock; each read owns its
scratch buffer so background writeback and snapshots cannot overwrite each other.
Ordinary `bytes`/`bytearray` passed to `pread`/`pwrite` do not guarantee the required
user-memory alignment and can fail with `EINVAL` on FDMFS.

Each slot additionally holds up to `capacity_tokens` K and V tokens in MR5/MR6.
The number of ranges is unchanged (11 legacy / 12 staged). Slot snapshots are
bounded separately by the configured slot count. The packet uses an optional
buffer-tail flag and trailer; all stages validate the same continuation metadata.
Old QEMU binaries reject this flag, so restart with the rebuilt binary before use.

## Tests

Host-only CPU regressions (no checkpoint, GPU or Guest):

```bash
cd /home/lihan/cemu/CEMU/tests/cemu
make instattention-sparf-stages-test
```

Tests cover full/mixed/empty NVM prefixes, deliberately stale NVM tails, FP16/FP32,
GQA head mapping, ring wraparound, delayed/failed publication, bounded backpressure,
and two-microbatch multistep execution through the actual C workflow. The fake
transport does not establish Guest/CUDA performance or target-time overlap.

After restarting the rebuilt QEMU, first run in the Guest:

```bash
source /root/cemu-gpu-env/bin/activate
cd /root/CEMU/tests/cemu
make instattention-sparf-buffer-device-test \
  NVM_TEST_DIR=/mnt/nvme0 FDM_TEST_DIR=/mnt/fdm0 CSD_TARGET=cuda
```

This uses the existing Host `./build/sparf_attention_devptr.so`, two microbatches,
two layers, eight Decode steps and a four-token ring. Ratios 1 and 8 both compare
against NumPy. Ratio 1 forces all old buffered V entries to be consumed. Final
token K/V and channel K are read back after drain. No OPT weights are loaded.
The CUDA operator itself is unchanged; only QEMU and Guest Python need updating.

Then compare the three OPT functional paths, sharing one measured Softmax table:

```bash
COMPARE_KV_BUFFER=1 SPARF_KV_BUFFER_TOKENS=4 DECODE_STEPS=4 \
  bash instattention/experiments/run_sparf_overlap_ab.sh
```

Results are `staged0`, `staged1`, and `buffered` under the unique run directory.
The original two-case behavior remains the default. The script checks generated
token equality, not full logits equality. Do not infer quality from completion
alone, nor infer stable speedup from one ordered A/B/C run.

Keep Host `CEMU_SPARF_TIMELINE` enabled for this behavior test and analyze the
fresh timeline using the existing analyzer. The zero-runtime scheduler shortcut
now emits a zero-duration service interval at finish rather than a missing start
timestamp; earlier malformed records are not silently rewritten.

## Interpreting timing

Compare `decode.kv_buffer_accept_wall`, `decode.kv_buffer_snapshot_wall`,
`decode.kv_buffer_backpressure_wall`, `kv_writeback.total_wall`, and the existing
execute/stage metrics. Background rows can overlap and must not be added together.

`pipeline.decode_wall` excludes the final drain, as before. The new
`pipeline.decode_and_drain_wall` and `pipeline.e2e_including_drain_wall` explicitly
include that deferred work; they are Guest caller times, not Host wall time or a
calibrated target end-to-end latency. The latter includes Prefill, but not weight
loading. Both intervals also include the small intervening reporting overhead.

The comparison should establish data visibility and actual submission/overlap
behavior first. Any remaining target timing discrepancy is a measurement result,
not a reason to lower modeled runtime or remove required dependencies.

## Batched channel-K append

Decode writeback updates the channel-K mmap with one strided `np.copyto` per
layer/group, rather than one small assignment per batch/head. The source is a
transpose view of the incoming keys; the destination view uses the existing
batch, head, channel, and token strides. No full-layer temporary, extra history
read, padding write, or file-layout change is introduced. Token K/V writes are
unchanged and precede the channel-K copy. Publication still synchronizes all
three files before the write buffer advances its published frontier.

For microbatch 2, 40 KV heads, head dimension 128, and a two-token group, this
replaces 80 assignments of 256 elements with one assignment of 20,480 elements.
NumPy's thresholded GIL-release path can therefore run the larger numeric copy
without holding the GIL (the checked NumPy 2.2 implementation uses a threshold of
500 elements). Small test shapes need not release it. This is not a guarantee
that mmap faults or filesystem waits disappear, nor proof that GIL contention
caused every observed stage gap; verify the deployed NumPy and Guest timings.

The existing `kv_append.channel_k_mmap_wall` metric remains one accumulated row
per writeback group and counts only the logical new K bytes. This optimization
does not alter buffer capacity, flush grouping, CSD commands, SparF selection,
NAND configuration, or the target runtime model. No Guest driver, QEMU, or CUDA
operator rebuild is required for this Python-only change.
