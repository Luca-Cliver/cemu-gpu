#!/bin/bash
set -euo pipefail

# Fresh session; a B2 softmax table or B4 token reference cannot be reused.
# An optional argument may supply a matching B4-microbatch softmax table.
if (( $# > 1 )); then
    echo 'Usage: flexgen-b8-microbatch-guest-test.sh [matching-softmax.json]' >&2
    exit 1
fi
export CEMU_FLEXGEN_BATCH_SIZE=8 CEMU_FLEXGEN_MICROBATCH_SIZE=4
export CEMU_FLEXGEN_DECODE_STEPS=16 CEMU_FLEXGEN_STACK_TRACE=1
export CEMU_FLEXGEN_DEVICE_PROFILE=b8-p1024-d16
repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
if [[ $(id -u) != 0 || ! -b /dev/nvme0m2 || ! -c /dev/ng0n3 ]]; then
    echo 'Run as root inside the configured CEMU Guest.' >&2
    exit 1
fi
mkdir -p /root/cemu-logs
mountpoint -q /root/cemu-logs || mount -t 9p -o trans=virtio,version=9p2000.L logsrc /root/cemu-logs
# Refuse the historical B8 device or a launch without detailed aggregation timing.
python3 - <<'PY'
import json
from pathlib import Path
root = Path('/root/cemu-logs')
settings = dict(line.split('=', 1) for line in (root / 'launch-settings.txt').read_text().splitlines() if '=' in line)
assert settings['aggregate_profile'] == '1', 'Restart using run-flexgen-b8-microbatch-test.sh'
config = json.loads((root / 'device-config.json').read_text())
nand = next(ns['parameters'] for ns in config['namespaces'] if ns['type'] == 'nvm')
assert (nand['nchs'], nand['luns_per_ch'], nand['pg_rd_lat']) == (8, 4, 11703), 'Expected the same 32-LUN hardware as B4'
PY
exec bash "$repo/femu-scripts/flexgen-b8-guest-test.sh" "$@"
