#!/bin/bash
set -euo pipefail

# One run per fresh CEMU native NVM session. Default: B4 split into two B2s.
# Usage: bash flexgen-b4-microbatch-guest-test.sh [2|4|1] [matching-softmax.json]
microbatch=${1:-2}
case "$microbatch" in
    1|2|4) ;;
    *) echo 'Microbatch must be 1, 2 or 4; total batch stays 4.' >&2; exit 1 ;;
esac
if (( $# > 2 )); then
    echo 'Usage: flexgen-b4-microbatch-guest-test.sh [2|4|1] [matching-softmax.json]' >&2
    exit 1
fi
if (( $# )); then shift; fi
export CEMU_FLEXGEN_BATCH_SIZE=4
export CEMU_FLEXGEN_MICROBATCH_SIZE=$microbatch
export CEMU_FLEXGEN_DECODE_STEPS=16
export CEMU_FLEXGEN_DEVICE_PROFILE=b4-p1024-g1024-lun32
repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
exec bash "$repo/femu-scripts/flexgen-b8-guest-test.sh" "$@"
