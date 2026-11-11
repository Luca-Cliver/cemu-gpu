#!/bin/bash
set -euo pipefail

export CEMU_FLEXGEN_BATCH_SIZE=${CEMU_FLEXGEN_BATCH_SIZE:-8}
case "$CEMU_FLEXGEN_BATCH_SIZE" in
    4|6|8) ;;
    *) echo 'CEMU_FLEXGEN_BATCH_SIZE must be 4, 6 or 8' >&2; exit 1 ;;
esac
export CEMU_FLEXGEN_MICROBATCH_SIZE=${CEMU_FLEXGEN_MICROBATCH_SIZE:-$CEMU_FLEXGEN_BATCH_SIZE}
if [[ ! "$CEMU_FLEXGEN_MICROBATCH_SIZE" =~ ^[1-9][0-9]*$ ]] ||
   (( CEMU_FLEXGEN_MICROBATCH_SIZE > CEMU_FLEXGEN_BATCH_SIZE ||
      CEMU_FLEXGEN_BATCH_SIZE % CEMU_FLEXGEN_MICROBATCH_SIZE != 0 )); then
    echo 'CEMU_FLEXGEN_MICROBATCH_SIZE must be a positive divisor of the total batch size.' >&2
    exit 1
fi
export CEMU_FLEXGEN_DECODE_STEPS=${CEMU_FLEXGEN_DECODE_STEPS:-16}
expected_profile=b8-p1024-d16
case "$CEMU_FLEXGEN_DECODE_STEPS" in
    16) ;;
    1023)
        if [[ "$CEMU_FLEXGEN_BATCH_SIZE" != 4 ]]; then
            echo 'The 1024-output experiment requires batch=4.' >&2
            exit 1
        fi
        expected_profile=b4-p1024-g1024-lun32
        ;;
    *) echo 'CEMU_FLEXGEN_DECODE_STEPS must be 16 or 1023' >&2; exit 1 ;;
esac
# Device geometry is independent of output length. The short B4 comparison
# explicitly reuses the long run's 32-LUN configuration instead of the old D16 one.
expected_profile=${CEMU_FLEXGEN_DEVICE_PROFILE:-$expected_profile}
case "$expected_profile" in
    b8-p1024-d16)
        [[ "$CEMU_FLEXGEN_DECODE_STEPS" == 16 ]] || exit 1 ;;
    b4-p1024-g1024-lun32)
        [[ "$CEMU_FLEXGEN_BATCH_SIZE" == 4 ]] || exit 1 ;;
    *) echo 'Unsupported CEMU_FLEXGEN_DEVICE_PROFILE.' >&2; exit 1 ;;
esac
maximum=$((1024 + CEMU_FLEXGEN_DECODE_STEPS))
export CEMU_FLEXGEN_FDM_INPUT_SLOTS=${CEMU_FLEXGEN_FDM_INPUT_SLOTS:-0}
case "$CEMU_FLEXGEN_FDM_INPUT_SLOTS" in
    0|1|2) ;;
    *) echo 'CEMU_FLEXGEN_FDM_INPUT_SLOTS must be 0, 1 or 2.' >&2; exit 1 ;;
esac
export CEMU_FLEXGEN_FDM_INPUT_BYTES=$((512 + 2 * CEMU_FLEXGEN_MICROBATCH_SIZE * 40 * maximum * 128 * 2))
extra_input_bytes=0
if [[ "$CEMU_FLEXGEN_FDM_INPUT_SLOTS" == 2 ]]; then
    extra_input_bytes=$(((CEMU_FLEXGEN_FDM_INPUT_BYTES + 4095) / 4096 * 4096))
fi
export CEMU_FLEXGEN_FDM_DATA_BYTES=$((8589934592 + extra_input_bytes))
stack_trace=${CEMU_FLEXGEN_STACK_TRACE:-0}
case "$stack_trace" in
    0|1) ;;
    *) echo 'CEMU_FLEXGEN_STACK_TRACE must be 0 or 1.' >&2; exit 1 ;;
esac

repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
if [[ $(id -u) != 0 || ! -b /dev/nvme0m2 || ! -c /dev/ng0n3 ]]; then
    echo 'Run this script as root inside the configured CEMU Guest, not on the Host.' >&2
    exit 1
fi
mkdir -p /root/models /root/cemu-logs /mnt/fdm0
mountpoint -q /root/models || mount -t 9p -o ro,trans=virtio,version=9p2000.L modelsrc /root/models
mountpoint -q /root/cemu-logs || mount -t 9p -o trans=virtio,version=9p2000.L logsrc /root/cemu-logs
mountpoint -q /mnt/fdm0 || mount -t fdmfs /dev/nvme0m2 /mnt/fdm0
test "$(findmnt -n -o FSTYPE /mnt/fdm0)" = fdmfs
test "$(findmnt -n -o SOURCE /mnt/fdm0)" = /dev/nvme0m2
settings=/root/cemu-logs/launch-settings.txt
for setting in "profile=$expected_profile" csd_target=cuda native_kv_placement=balanced read_diagnostics=1; do
    grep -qxF "$setting" "$settings" || { echo "Wrong VM settings: $setting required" >&2; exit 1; }
done
if [[ "$CEMU_FLEXGEN_FDM_INPUT_SLOTS" != 0 ]]; then
    for setting in fdm_async_aggregate=1 fdm_aggregate_mode=poll; do
        grep -qxF "$setting" "$settings" || { echo "Tagged input requires $setting; restart the VM." >&2; exit 1; }
    done
    mountpoint -q /sys/kernel/tracing || mount -t tracefs tracefs /sys/kernel/tracing
    grep -q 'field:.*transfer_id' /sys/kernel/tracing/events/nvme/nvme_cemu_io/format || {
        echo 'Tagged input requires the rebuilt Guest kernel.' >&2; exit 1;
    }
fi
if [[ -n $(find /mnt/fdm0 -mindepth 1 -maxdepth 1 -print -quit) ]]; then
    echo 'FDM is not empty. Finish the previous job and drain/close it; this script will not delete files.' >&2
    exit 1
fi
if [[ "$stack_trace" == 1 ]]; then
    # Fail before loading weights or creating device KV state.
    mountpoint -q /sys/kernel/tracing || mount -t tracefs tracefs /sys/kernel/tracing
    for event in syscalls/sys_enter_pwrite64 syscalls/sys_exit_pwrite64 syscalls/sys_enter_pread64 syscalls/sys_exit_pread64 syscalls/sys_enter_ioctl syscalls/sys_exit_ioctl fdmfs/fdmfs_rw_begin fdmfs/fdmfs_rw_end iomap/iomap_dio_rw_begin iomap/iomap_dio_complete nvme/nvme_cemu_io nvme/nvme_complete_rq; do
        test -f "/sys/kernel/tracing/events/$event/enable" || { echo "Missing tracepoint: $event" >&2; exit 1; }
    done
    grep -q 'name=' /sys/kernel/tracing/events/fdmfs/fdmfs_rw_begin/format || { echo 'Old FDMFS trace format' >&2; exit 1; }
    grep -q 'coverage=2' /sys/kernel/tracing/events/nvme/nvme_cemu_io/format || { echo 'Rebuild linux-cemu and boot the coverage=2 kernel (FDM read/write and EXEC tracing).' >&2; exit 1; }
    io_trace=$(sed -n 's/^fdm_io_trace=//p' "$settings")
    test -n "$io_trace" || { echo 'Restart Host with CEMU_NATIVE_IO_TRACE=1' >&2; exit 1; }
fi
if [[ -n ${CEMU_GUEST_ENV_FILE:-} ]]; then
    source "$CEMU_GUEST_ENV_FILE"
    : "${CEMU_CLIENT_BUILD_DIR:?Custom Guest environment must set CEMU_CLIENT_BUILD_DIR}"
else
    source /root/cemu-gpu-env/bin/activate
fi
modprobe nvidia_uvm
cd "$repo/tests/cemu"
if [[ -n ${CEMU_GUEST_ENV_FILE:-} ]]; then
    cmake -S instattention -B "$CEMU_CLIENT_BUILD_DIR" \
        -DPython3_EXECUTABLE="$(command -v python3)" -DBUILD_TESTING=OFF -DCMAKE_BUILD_TYPE=Release
    cmake --build "$CEMU_CLIENT_BUILD_DIR" --target _cemu_client -j4
    export PYTHONPATH="$CEMU_CLIENT_BUILD_DIR:$PWD/instattention/python:$PWD/instattention"
else
    make instattention INSTATTENTION_BUILD_DIR=/root/native-opt-client
    export PYTHONPATH="/root/native-opt-client:$PWD/instattention/python:$PWD/instattention${PYTHONPATH:+:$PYTHONPATH}"
fi
export CEMU_IO_BACKEND=ioctl
unset CEMU_PWRITE_PACKET_TRACE
clock=/sys/devices/system/clocksource/clocksource0/current_clocksource
echo hpet > "$clock"
test "$(cat "$clock")" = hpet
python3 - <<'PY'
import attr
import os
from pathlib import Path
import torch
import tqdm
from cemu_flexgen.sparf_group_capacity import grouped_attention_reserve_bytes
from native_kv_client import LinuxTransport, require_fresh_namespace
transport = LinuxTransport('/dev/ng0n1')
try:
    require_fresh_namespace(transport)
finally:
    transport.close()
assert torch.cuda.is_available(), "Guest CUDA unavailable"
space = os.statvfs('/mnt/fdm0')
required = int(os.environ['CEMU_FLEXGEN_FDM_DATA_BYTES']) + 128 * 1024**2 + grouped_attention_reserve_bytes(
    attention_slots=2, batch_size=int(os.environ['CEMU_FLEXGEN_MICROBATCH_SIZE']), query_heads=40, kv_heads=40,
    head_dim=128, element_bytes=2, max_seq_len=1024 + int(os.environ['CEMU_FLEXGEN_DECODE_STEPS']),
    top_r=16, compression_ratio=8)
capacity = int(Path('/sys/class/block/nvme0m2/size').read_text()) * 512
available = space.f_bavail * space.f_frsize if space.f_blocks else capacity
assert available >= required, "insufficient FDM capacity"
if not space.f_blocks:
    print('FDM statfs does not report free space; using empty-filesystem device capacity, BIND remains the final allocation check')
print('Guest GPU:', torch.cuda.get_device_name(0), '; required FDM bytes:', required, flush=True)
PY
run=$(mktemp -d "/root/cemu-logs/flexgen-b${CEMU_FLEXGEN_BATCH_SIZE}-mb${CEMU_FLEXGEN_MICROBATCH_SIZE}-d${CEMU_FLEXGEN_DECODE_STEPS}.XXXXXX")
printf 'B%s log directory: %s\n' "$CEMU_FLEXGEN_BATCH_SIZE" "$run"
exec > >(tee "$run/driver.log") 2>&1
cp -- "$settings" "$run/launch-settings.txt"
if [[ -f /root/cemu-logs/device-config.json ]]; then
    cp -- /root/cemu-logs/device-config.json "$run/device-config.json"
fi
if [[ -f /root/cemu-logs/host-execution-context.json ]]; then
    cp -- /root/cemu-logs/host-execution-context.json "$run/host-execution-context.json"
fi
cp -- "$repo/tests/cemu/kernel/fdm_kv_aggregate.c" "$run/fdm_kv_aggregate.c"
cp -- "$repo/tests/cemu/build/fdm_kv_aggregate.so" "$run/fdm_kv_aggregate.so"
sha256sum "$run/fdm_kv_aggregate.c" "$run/fdm_kv_aggregate.so" > "$run/aggregate-sha256.txt"
printf 'batch=%s\nmicrobatch=%s\nmicrobatches=%s\nprompt=1024\ndecode_steps=%s\noutput_tokens=%s\n' \
    "$CEMU_FLEXGEN_BATCH_SIZE" "$CEMU_FLEXGEN_MICROBATCH_SIZE" \
    "$((CEMU_FLEXGEN_BATCH_SIZE / CEMU_FLEXGEN_MICROBATCH_SIZE))" \
    "$CEMU_FLEXGEN_DECODE_STEPS" "$((CEMU_FLEXGEN_DECODE_STEPS + 1))" \
    > "$run/workload-settings.txt"
printf 'stack_trace=%s\n' "$stack_trace" >> "$run/workload-settings.txt"
printf 'fdm_input_slots=%s\nfdm_input_bytes=%s\nfdm_data_bytes=%s\n' \
    "$CEMU_FLEXGEN_FDM_INPUT_SLOTS" "$CEMU_FLEXGEN_FDM_INPUT_BYTES" \
    "$CEMU_FLEXGEN_FDM_DATA_BYTES" >> "$run/workload-settings.txt"
cp -- instattention/experiments/configs/opt13b_sparf_1csd.json "$run/runtime.json"
ratio=${1:-}
if [[ -n "$ratio" ]]; then
    cp -- "$ratio" "$run/softmax.json"
else
    python3 -u instattention/experiments/bench_softmax_ratio.py \
        --config "$run/runtime.json" --device cuda:0 --batch-sizes "$CEMU_FLEXGEN_MICROBATCH_SIZE" \
        --prompt-length 1024 --decode-steps "$CEMU_FLEXGEN_DECODE_STEPS" --compression-ratio 8 \
        --output "$run/softmax.json"
fi
python3 - "$run" <<'PY'
from pathlib import Path
import os
import sys
from experiments.config import load_experiment_config
from experiments.softmax_ratio import SoftmaxRatioTable
run = Path(sys.argv[1])
ratios = SoftmaxRatioTable.load(run / 'softmax.json')
ratios.validate_anchor(load_experiment_config(run / 'runtime.json').instcsd)
microbatch = int(os.environ['CEMU_FLEXGEN_MICROBATCH_SIZE'])
ratios.require_sparf(microbatch, 40, 1024, int(os.environ['CEMU_FLEXGEN_DECODE_STEPS']), 8)
print(f'Microbatch={microbatch} Softmax ratio validated; FDM input auto-sized for whole-layer packets', flush=True)
PY
program=$(sed -n 's/^program=//p' "$settings")
aggregate=$(dirname -- "$program")/fdm_kv_aggregate.so
reference_args=()
if [[ -n ${CEMU_FLEXGEN_REFERENCE_RESULT:-} ]]; then
    reference_args=(--reference-result "$CEMU_FLEXGEN_REFERENCE_RESULT")
fi
trace_command=()
if [[ "$stack_trace" == 1 ]]; then
    trace_command=(python3 -u instattention/experiments/pwrite_stack_trace.py record --output "$run/trace" --)
fi
"${trace_command[@]}" python3 -u instattention/python/flexgen_cemu.py \
    --model-dir /root/models/OPT-13B --weights-dir /root/models/flexgen-weights \
    --storage fdm --batch-size "$CEMU_FLEXGEN_BATCH_SIZE" --gpu-batch-size "$CEMU_FLEXGEN_MICROBATCH_SIZE" \
    --prompt-length 1024 --decode-steps "$CEMU_FLEXGEN_DECODE_STEPS" --attention-slots 2 \
    --fdm-meta-bytes 134217728 --fdm-data-bytes "$CEMU_FLEXGEN_FDM_DATA_BYTES" \
    --fdm-input-slots "$CEMU_FLEXGEN_FDM_INPUT_SLOTS" --fdm-input-bytes "$CEMU_FLEXGEN_FDM_INPUT_BYTES" \
    --csd-target cuda --program "$program" --aggregate-program "$aggregate" \
    --runtime-config "$run/runtime.json" --softmax-ratio-file "$run/softmax.json" \
    --profile-stack --profile-output "$run/profile.csv" --result-output "$run/result.json" \
    "${reference_args[@]}" \
    2>&1 | tee "$run/run.log"
printf '[flexgen-clock] final=%s\n' "$(cat "$clock")" | tee -a "$run/run.log"
test "$(cat "$clock")" = hpet
cp -- /root/cemu-logs/device-timeline.jsonl "$run/device-timeline.jsonl"
if [[ "$stack_trace" == 1 ]]; then
    cp -- /root/cemu-logs/fdm-io-device.jsonl "$run/fdm-io-device.jsonl"
    python3 instattention/experiments/pwrite_stack_trace.py analyze "$run/trace" \
        --device "$run/fdm-io-device.jsonl" --account-device --decode-stack \
        --attention-device "$run/device-timeline.jsonl" --output "$run/pwrite-stack-summary.json" \
        2>&1 | tee "$run/pwrite-stack-analysis.txt"
    if grep -qxF 'fdm_async_aggregate=1' "$settings"; then
        echo '[aggregate-profile] Background aggregation is recorded separately in fdm-io-device.jsonl; not part of the acknowledged pwrite interval.'
    elif grep -qxF 'aggregate_profile=1' "$settings"; then
        python3 - "$run/pwrite-stack-summary.json" <<'PYAGG'
import json
import sys
report = json.load(open(sys.argv[1]))
for phase in ('prefill', 'decode'):
    assert report['counters'][phase + '.aggregate_profile.calls'] == report['counters'][phase + '.device.aggregate_calls']
    assert phase + '.host.aggregate_detail.channel_k' in report['metrics']
print('[aggregate-profile] Prefill/Decode module timing coverage verified; Host elapsed, including instrumentation overhead.')
PYAGG
        python3 instattention/experiments/aggregate_decode_trace.py "$run/fdm-io-device.jsonl" \
            --require-probe --output "$run/aggregate-decode-split.json" \
            2>&1 | tee "$run/aggregate-decode-split.txt"
    fi
fi
python3 instattention/experiments/analyze_sparf_autonomous.py \
    "$run/device-timeline.jsonl" --latest-session --min-tokens 1025 --max-tokens "$maximum" \
    --require-read-diagnostics --require-native-read-reference \
    --output "$run/read-requests.json" --read-summary "$run/read-summary.json" \
    --read-loss-summary "$run/read-loss.json" --boundary-summary "$run/boundaries.json" \
    2>&1 | tee "$run/analysis.txt"
python3 - "$run" <<'PY' | tee "$run/bandwidth.txt"
import csv
import json
import os
from pathlib import Path
import sys
from experiments.analyze_sparf_autonomous import select_records, summarize_concurrency
run = Path(sys.argv[1])
result = json.loads((run / 'result.json').read_text())
stats = result['statistics']
batch = int(os.environ['CEMU_FLEXGEN_BATCH_SIZE'])
microbatch = int(os.environ['CEMU_FLEXGEN_MICROBATCH_SIZE'])
steps = int(os.environ['CEMU_FLEXGEN_DECODE_STEPS'])
maximum = 1024 + steps
requests = 40 * steps * (batch // microbatch)
assert result['arguments']['batch_size'] == batch
assert result['arguments']['gpu_batch_size'] == microbatch
assert result['arguments']['fdm_input_slots'] == int(os.environ['CEMU_FLEXGEN_FDM_INPUT_SLOTS'])
assert result['arguments']['fdm_input_bytes'] == int(os.environ['CEMU_FLEXGEN_FDM_INPUT_BYTES'])
assert result['arguments']['fdm_data_bytes'] == int(os.environ['CEMU_FLEXGEN_FDM_DATA_BYTES'])
pages_per_head = 2 * ((maximum + 15) // 16) + 16 * ((maximum + 255) // 256)
assert result['attention_submissions'] == requests
assert len(result['tokens']) == batch and all(len(tokens) == steps + 1 for tokens in result['tokens'])
append_requests = 40 * (steps + 1) * (batch // microbatch)
assert stats['client_append_commands'] == stats['append_commands'] == append_requests, (
    f"Expected {append_requests} layer/microbatch append packets, "
    f"client={stats['client_append_commands']}, device={stats['append_commands']}")
assert stats['client_appended_head_tokens'] == 1600 * maximum * batch
assert stats['write_bytes'] == pages_per_head * 1600 * batch * 4096
assert stats['sealed'] == 1 and stats['failed'] == 0
assert all(stats[name] == 0 for name in ('staged_pages', 'inflight_blocks', 'pending_ready_ns'))
assert all(stats[name] == maximum for name in ('accepted_stream0', 'token_k', 'channel_k', 'token_v'))
reads = json.loads((run / 'read-summary.json').read_text())
for phase, values in reads.items():
    assert values['observed_requests'] == requests and values['missing_requests'] == 0
    assert values['unmapped_pages'] == 0
    seconds = values['model_ns'] / 1e9
    bandwidth = values['submitted_bytes'] / seconds / 1024**3 if seconds else 0
    print(f"{phase}: Flash={values['submitted_bytes']} B, modeled={seconds:.6f}s, "
          f"stage_effective={bandwidth:.6f} GiB/s, active_LUN_mean={values['active_luns_mean']:.3f}, "
          f"worst_LUN_max/mean={values['lun_max_to_mean']}")
profiles = {row['phase']: float(row['total_seconds'])
            for row in csv.DictReader((run / 'profile.csv').open())}
flash = sum(values['submitted_bytes'] for values in reads.values())
decode = profiles['framework.decode_iterations_wall']
print(f'Decode Flash={flash} B; HPET elapsed={decode:.6f}s; application={flash / decode / 1024**3:.6f} GiB/s')
print(f'Decode throughput={batch * steps / decode:.6f} token/s; '
      f'generation throughput={batch * (steps + 1) / profiles["pipeline.generate_wall"]:.6f} token/s; '
      f'including drain={batch * (steps + 1) / profiles["pipeline.e2e_including_drain_wall"]:.6f} token/s')
print('Stage rates use modeled intervals, not summed per-page service. Overlapping phases must not be added as elapsed time.')
print('These are simulator effective rates, not measured physical NAND/channel bus bandwidth; token/logits accuracy is not proved by drain checks.')
records = [json.loads(line) for line in (run / 'device-timeline.jsonl').read_text().splitlines() if line.strip()]
records = select_records(records, latest_session=True, min_tokens=1025, max_tokens=maximum)
concurrency = summarize_concurrency(records)
(run / 'concurrency.json').write_text(json.dumps(concurrency, indent=2) + '\n')
print(f'Batch={batch}, microbatch={microbatch}, parents={requests}; device concurrency={concurrency}')
if microbatch < batch:
    assert concurrency['max_active_requests'] >= 2, 'Microbatches split, but no concurrent device parents observed; do not claim a pipeline speedup.'
PY
printf '\nB%s inference and bandwidth diagnostics complete: %s\n' "$CEMU_FLEXGEN_BATCH_SIZE" "$run"
