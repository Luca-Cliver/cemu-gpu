#!/bin/bash
set -euo pipefail

if (( $# > 1 )); then
    echo 'Usage: vllm-opt-guest-test.sh [matching-B8-softmax.json]' >&2
    exit 1
fi
repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
if [[ $(id -u) != 0 || ! -b /dev/nvme0m2 || ! -c /dev/ng0n3 ]]; then
    echo 'Run inside the configured CEMU Guest as root.' >&2
    exit 1
fi
if [[ -n ${CEMU_GUEST_ENV_FILE:-} ]]; then
    source "$CEMU_GUEST_ENV_FILE"
    : "${CEMU_CLIENT_BUILD_DIR:?Custom Guest environment must set CEMU_CLIENT_BUILD_DIR}"
    client_build_dir=$CEMU_CLIENT_BUILD_DIR
else
    source /root/cemu-vllm-env.sh
    client_build_dir=/root/cemu-vllm-build
fi
cd /root
mkdir -p /root/models /root/cemu-logs /mnt/fdm0
mountpoint -q /root/models || mount -t 9p -o ro,trans=virtio,version=9p2000.L modelsrc /root/models
mountpoint -q /root/cemu-logs || mount -t 9p -o trans=virtio,version=9p2000.L logsrc /root/cemu-logs
mountpoint -q /mnt/fdm0 || mount -t fdmfs /dev/nvme0m2 /mnt/fdm0
test "$(findmnt -n -o FSTYPE /mnt/fdm0)" = fdmfs
test "$(findmnt -n -o SOURCE /mnt/fdm0)" = /dev/nvme0m2
if [[ -n $(find /mnt/fdm0 -mindepth 1 -maxdepth 1 -print -quit) ]]; then
    echo 'FDM is not empty; close/drain the previous session. Nothing will be deleted.' >&2
    exit 1
fi
settings=/root/cemu-logs/launch-settings.txt
for setting in profile=b8-p1024-d16 csd_target=cuda native_kv_placement=balanced read_diagnostics=1 fdm_async_aggregate=1; do
    grep -qxF "$setting" "$settings" || { echo "Missing setting: $setting" >&2; exit 1; }
done
trace=${CEMU_VLLM_STACK_TRACE:-1}
export CEMU_VLLM_STACK_TRACE="$trace"
case "$trace" in 0|1) ;; *) echo 'CEMU_VLLM_STACK_TRACE must be 0 or 1' >&2; exit 1 ;; esac
if [[ "$trace" == 1 ]]; then
    mountpoint -q /sys/kernel/tracing || mount -t tracefs tracefs /sys/kernel/tracing
    grep -q 'coverage=2' /sys/kernel/tracing/events/nvme/nvme_cemu_io/format
    test -n "$(sed -n 's/^fdm_io_trace=//p' "$settings")"
fi
cmake -S "$ADAPTER" -B "$client_build_dir" \
    -DPython3_EXECUTABLE="$(command -v python)" -DBUILD_TESTING=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build "$client_build_dir" --target _cemu_client -j4
modprobe nvidia_uvm
clock=/sys/devices/system/clocksource/clocksource0/current_clocksource
echo hpet > "$clock"
test "$(cat "$clock")" = hpet
python - <<'PY'
import json
import os
from pathlib import Path
from cemu_flexgen.sparf_group_capacity import grouped_attention_reserve_bytes
from experiments.pwrite_stack_trace import EVENTS
from native_kv_client import LinuxTransport, require_fresh_namespace
import _cemu_client
import torch
transport = LinuxTransport('/dev/ng0n1')
try:
    require_fresh_namespace(transport)
finally:
    transport.close()
assert getattr(_cemu_client, 'IO_PROFILE_VERSION', 0) == 1, 'Rebuild the Guest client before tracing'
assert torch.cuda.is_available(), 'Guest CUDA unavailable'
config = json.loads(Path('/root/cemu-logs/device-config.json').read_text())
nand = next(ns['parameters'] for ns in config['namespaces'] if ns['type'] == 'nvm')
assert (nand['nchs'], nand['luns_per_ch'], nand['pg_rd_lat']) == (8, 4, 11703)
required = 8 * 1024**3 + 128 * 1024**2 + grouped_attention_reserve_bytes(
    attention_slots=1, batch_size=8, query_heads=40, kv_heads=40, head_dim=128,
    element_bytes=2, max_seq_len=1041, top_r=16, compression_ratio=8)
space = os.statvfs('/mnt/fdm0')
available = space.f_bavail * space.f_frsize if space.f_blocks else int(
    Path('/sys/class/block/nvme0m2/size').read_text()) * 512
assert available >= required, 'Insufficient empty FDM capacity'
if os.environ['CEMU_VLLM_STACK_TRACE'] == '1':
    root = Path('/sys/kernel/tracing/events')
    assert all((root / event / 'enable').exists() for event in EVENTS), 'Missing Guest tracepoints'
print('Guest GPU:', torch.cuda.get_device_name(0), 'required FDM bytes:', required, flush=True)
PY
run=$(mktemp -d /root/cemu-logs/vllm-b8-p1024-d16.XXXXXX)
echo "Experiment directory: $run"
exec > >(tee "$run/driver.log") 2>&1
cp -- "$settings" /root/cemu-logs/device-config.json "$run/"
if [[ -f /root/cemu-logs/host-execution-context.json ]]; then
    cp -- /root/cemu-logs/host-execution-context.json "$run/"
fi
cp -- "$ADAPTER/experiments/configs/opt13b_sparf_1csd.json" "$run/runtime.json"
unset CEMU_PROFILE_INTERVALS CEMU_PWRITE_PACKET_TRACE
if [[ -n ${1:-} ]]; then
    cp -- "$1" "$run/softmax.json"
else
    python -u "$ADAPTER/experiments/bench_softmax_ratio.py" \
        --config "$run/runtime.json" --device cuda:0 --batch-sizes 8 \
        --prompt-length 1024 --decode-steps 16 --compression-ratio 8 \
        --output "$run/softmax.json"
fi
python - "$run" "$ADAPTER" <<'PY'
import json
from pathlib import Path
import sys
from opt_inputs import prepare_prompt
from vllm_adapter.session import Config, load_runtime
root, adapter = map(Path, sys.argv[1:])
config = json.loads((adapter / 'experiments/configs/vllm_opt13b_b8_p1024_d16.json').read_text())
settings = dict(line.split('=', 1) for line in (root / 'launch-settings.txt').read_text().splitlines() if '=' in line)
config.update(runtime_config=str(root / 'runtime.json'), softmax_ratio_file=str(root / 'softmax.json'),
              fdm_directory='/mnt/fdm0/' + root.name, program_path=settings['program'],
              aggregate_program=str(Path(settings['program']).with_name('fdm_kv_aggregate.so')))
load_runtime(Config(**config)).softmax_ratios.require_sparf(8, 40, 1024, 16, 8)
(root / 'adapter.json').write_text(json.dumps(config, indent=2) + '\n')
prompt, _ = prepare_prompt('/root/models/OPT-13B', 8, 1024, 16)
(root / 'prompt.json').write_text(json.dumps(prompt.tolist()) + '\n')
PY
record=()
profile=basic
if [[ "$trace" == 1 ]]; then
    record=(python -u "$ADAPTER/experiments/pwrite_stack_trace.py" record --output "$run/trace" --)
    profile=detailed
fi
"${record[@]}" python -u -m vllm_adapter.run \
    --config "$run/adapter.json" --model /root/models/OPT-13B \
    --prompt-token-ids "$run/prompt.json" --max-tokens 17 --max-num-batched-tokens 8192 \
    --profile "$profile" --output "$run/result.json" \
    2>&1 | tee "$run/run.log"
printf '[vllm-clock] final=%s\n' "$(cat "$clock")" | tee -a "$run/run.log"
test "$(cat "$clock")" = hpet
cp -- /root/cemu-logs/device-timeline.jsonl "$run/device-timeline.jsonl"
if [[ "$trace" == 1 ]]; then
    cp -- /root/cemu-logs/fdm-io-device.jsonl "$run/fdm-io-device.jsonl"
    python "$ADAPTER/experiments/pwrite_stack_trace.py" analyze "$run/trace" \
        --device "$run/fdm-io-device.jsonl" --account-device --decode-stack \
        --attention-device "$run/device-timeline.jsonl" --output "$run/stack-summary.json" \
        2>&1 | tee "$run/stack-analysis.txt"
fi
python "$ADAPTER/experiments/analyze_sparf_autonomous.py" "$run/device-timeline.jsonl" \
    --latest-session --min-tokens 1025 --max-tokens 1040 \
    --require-read-diagnostics --require-native-read-reference \
    --output "$run/read-requests.json" --read-summary "$run/read-summary.json" \
    --read-loss-summary "$run/read-loss.json" --boundary-summary "$run/boundaries.json" \
    2>&1 | tee "$run/analysis.txt"
python - "$run/result.json" <<'PY'
import json
import sys
result = json.load(open(sys.argv[1]))
drain = result['drain'][0]
stats = drain['statistics']
assert result['request_admission'] == 'barrier'
assert len(result['outputs']) == 8 and all(len(row['token_ids']) == 17 for row in result['outputs'])
assert result['timing']['phases']['prefill']['steps'] == 1
assert result['timing']['phases']['decode']['steps'] == 16
assert result['timing']['phases']['mixed']['steps'] == 0
assert drain['attention_exec_count'] == 640
assert stats['append_commands'] == stats['client_append_commands'] == 8 * 40 * 17
assert stats['client_appended_head_tokens'] == 8 * 40 * 40 * 1040
assert stats['sealed'] == 1 and stats['failed'] == 0
assert all(stats[name] == 0 for name in ('staged_pages', 'inflight_blocks', 'pending_ready_ns'))
assert result['device_timing']['available'], result['device_timing']
assert result['device_timing']['compute_union_seconds'] > 0, 'Missing modeled compute budget'
print('Workload/drain/timing coverage passed; this does not certify full-model numerical accuracy.')
PY
printf '\nExperiment complete: %s\n' "$run"
