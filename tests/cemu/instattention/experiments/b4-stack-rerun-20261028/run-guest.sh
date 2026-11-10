#!/bin/bash
set -euo pipefail

export CEMU_FLEXGEN_BATCH_SIZE=4
case "$CEMU_FLEXGEN_BATCH_SIZE" in
    4|6|8) ;;
    *) echo 'CEMU_FLEXGEN_BATCH_SIZE must be 4, 6 or 8' >&2; exit 1 ;;
esac
export CEMU_FLEXGEN_DECODE_STEPS=1023
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
maximum=$((1024 + CEMU_FLEXGEN_DECODE_STEPS))

repo=${CEMU_PROFILE_REPO:-/root/CEMU}
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
if [[ -n $(find /mnt/fdm0 -mindepth 1 -maxdepth 1 -print -quit) ]]; then
    echo 'FDM is not empty. Finish the previous job and drain/close it; this script will not delete files.' >&2
    exit 1
fi
# Validate detailed tracing before loading weights or creating KV state.
mountpoint -q /sys/kernel/tracing || mount -t tracefs tracefs /sys/kernel/tracing
for event in syscalls/sys_enter_pwrite64 syscalls/sys_exit_pwrite64 fdmfs/fdmfs_rw_begin fdmfs/fdmfs_rw_end iomap/iomap_dio_rw_begin iomap/iomap_dio_complete nvme/nvme_cemu_io nvme/nvme_complete_rq; do
    test -f "/sys/kernel/tracing/events/$event/enable" || { echo "Missing tracepoint: $event" >&2; exit 1; }
done
grep -q 'name=' /sys/kernel/tracing/events/fdmfs/fdmfs_rw_begin/format || { echo 'Old FDMFS trace format' >&2; exit 1; }
io_trace=$(sed -n 's/^fdm_io_trace=//p' "$settings")
test -n "$io_trace" || { echo 'Restart Host with CEMU_NATIVE_IO_TRACE=1' >&2; exit 1; }
# Device trace is created lazily by the first memory-write command.
source /root/cemu-gpu-env/bin/activate
modprobe nvidia_uvm
cd "$repo/tests/cemu"
make instattention INSTATTENTION_BUILD_DIR=/root/native-opt-client
export PYTHONPATH="/root/native-opt-client:$PWD/instattention/python:$PWD/instattention${PYTHONPATH:+:$PYTHONPATH}"
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
assert torch.cuda.is_available(), "Guest CUDA unavailable"
space = os.statvfs('/mnt/fdm0')
required = 8 * 1024**3 + 128 * 1024**2 + grouped_attention_reserve_bytes(
    attention_slots=2, batch_size=int(os.environ['CEMU_FLEXGEN_BATCH_SIZE']), query_heads=40, kv_heads=40,
    head_dim=128, element_bytes=2, max_seq_len=1024 + int(os.environ['CEMU_FLEXGEN_DECODE_STEPS']),
    top_r=16, compression_ratio=8)
capacity = int(Path('/sys/class/block/nvme0m2/size').read_text()) * 512
available = space.f_bavail * space.f_frsize if space.f_blocks else capacity
assert available >= required, "insufficient FDM capacity"
if not space.f_blocks:
    print('FDM statfs does not report free space; using empty-filesystem device capacity, BIND remains the final allocation check')
print('Guest GPU:', torch.cuda.get_device_name(0), '; required FDM bytes:', required, flush=True)
PY
run=$(mktemp -d /root/cemu-logs/flexgen-b4-d1023-stack.XXXXXX)
printf 'B%s log directory: %s\n' "$CEMU_FLEXGEN_BATCH_SIZE" "$run"
exec > >(tee "$run/driver.log") 2>&1
cp -- "$settings" "$run/launch-settings.txt"
printf 'batch=%s\nprompt=1024\ndecode_steps=%s\noutput_tokens=%s\n' \
    "$CEMU_FLEXGEN_BATCH_SIZE" "$CEMU_FLEXGEN_DECODE_STEPS" "$((CEMU_FLEXGEN_DECODE_STEPS + 1))" \
    > "$run/workload-settings.txt"
baseline=${1:-/root/cemu-b4-d1023-baseline}
for file in runtime.json softmax.json result.json; do
    test -s "$baseline/$file" || { echo "Missing baseline file: $baseline/$file" >&2; exit 1; }
done
cp -- "$baseline/runtime.json" "$run/runtime.json"
cp -- "$baseline/softmax.json" "$run/softmax.json"
cp -- "$baseline/result.json" "$run/baseline-result.json"
python3 - "$run" <<'PYBASE'
import json
from pathlib import Path
import sys
run = Path(sys.argv[1])
reference = json.loads((run / 'baseline-result.json').read_text())
args = reference['arguments']
assert (args['batch_size'], args['gpu_batch_size'], args['prompt_length'], args['decode_steps']) == (4, 4, 1024, 1023)
assert args['storage'] == 'fdm' and not args['no_overlap']
(run / 'prompt.json').write_text(json.dumps(reference['case']['prompt_ids']) + '\n')
PYBASE
python3 - "$run" <<'PY'
from pathlib import Path
import os
import sys
from experiments.config import load_experiment_config
from experiments.softmax_ratio import SoftmaxRatioTable
run = Path(sys.argv[1])
ratios = SoftmaxRatioTable.load(run / 'softmax.json')
ratios.validate_anchor(load_experiment_config(run / 'runtime.json').instcsd)
batch = int(os.environ['CEMU_FLEXGEN_BATCH_SIZE'])
ratios.require_sparf(batch, 40, 1024, int(os.environ['CEMU_FLEXGEN_DECODE_STEPS']), 8)
print(f'B{batch} Softmax ratio validated; input slot remains 128 KiB', flush=True)
PY
program=$(sed -n 's/^program=//p' "$settings")
aggregate=$(dirname -- "$program")/fdm_kv_aggregate.so
python3 -u instattention/experiments/pwrite_stack_trace.py record --output "$run/trace" -- \
  python3 -u instattention/python/flexgen_cemu.py \
    --model-dir /root/models/OPT-13B --weights-dir /root/models/flexgen-weights \
    --storage fdm --batch-size "$CEMU_FLEXGEN_BATCH_SIZE" --gpu-batch-size "$CEMU_FLEXGEN_BATCH_SIZE" \
    --prompt-length 1024 --decode-steps "$CEMU_FLEXGEN_DECODE_STEPS" --attention-slots 2 \
    --prompt-ids "$run/prompt.json" --reference-result "$run/baseline-result.json" \
    --fdm-meta-bytes 134217728 --fdm-data-bytes 8589934592 \
    --csd-target cuda --program "$program" --aggregate-program "$aggregate" \
    --runtime-config "$run/runtime.json" --softmax-ratio-file "$run/softmax.json" \
    --profile-stack --profile-output "$run/profile.csv" --result-output "$run/result.json" \
    2>&1 | tee "$run/run.log"
printf '[flexgen-clock] final=%s\n' "$(cat "$clock")" | tee -a "$run/run.log"
test "$(cat "$clock")" = hpet
cp -- /root/cemu-logs/fdm-io-device.jsonl "$run/fdm-io-device.jsonl"
cp -- "$repo/tests/cemu/instattention/experiments/configs/fdm_opt13b_b4_p1024_g1024_lun32.json" "$run/expected-device-config.json"
python3 instattention/experiments/pwrite_stack_trace.py analyze "$run/trace" \
    --device "$run/fdm-io-device.jsonl" --output "$run/pwrite-stack-summary.json" \
    2>&1 | tee "$run/pwrite-stack-analysis.txt"
cp -- /root/cemu-logs/device-timeline.jsonl "$run/device-timeline.jsonl"
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
run = Path(sys.argv[1])
result = json.loads((run / 'result.json').read_text())
stats = result['statistics']
batch = int(os.environ['CEMU_FLEXGEN_BATCH_SIZE'])
steps = int(os.environ['CEMU_FLEXGEN_DECODE_STEPS'])
maximum = 1024 + steps
requests = 40 * steps
pages_per_head = 2 * ((maximum + 15) // 16) + 16 * ((maximum + 255) // 256)
assert result['attention_submissions'] == requests
assert len(result['tokens']) == batch and all(len(tokens) == steps + 1 for tokens in result['tokens'])
assert stats['client_append_commands'] == stats['append_commands'] == (7680 + 40 * steps) * batch
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
PY
printf '\nB%s inference and bandwidth diagnostics complete: %s\n' "$CEMU_FLEXGEN_BATCH_SIZE" "$run"
