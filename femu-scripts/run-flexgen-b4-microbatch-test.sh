#!/bin/bash
set -euo pipefail

repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
export CEMU_NATIVE_BOOT_MODE=normal
export CEMU_NATIVE_OS_IMAGE=${CEMU_NATIVE_OS_IMAGE:-/data/lihan/cemu-native-opt/standalone.PnutcI/guest.qcow2}
export CEMU_NATIVE_PROFILE=b4-p1024-g1024-lun32
export CEMU_NATIVE_CONFIG="$repo/tests/cemu/instattention/experiments/configs/fdm_opt13b_b4_p1024_g1024_lun32.json"
export CEMU_NATIVE_CSD_TARGET=cuda CEMU_NATIVE_KV_PLACEMENT=balanced
export CEMU_DEVICE_INIT_ORDER=femu-first
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}
export CEMU_CUDA_DEVICE=0
export CEMU_CUDA_LIBRARY_PATH=${CEMU_CUDA_LIBRARY_PATH:-/home/lihan/cemu/nvidia-575-libs}
export CEMU_NATIVE_TIMELINE=1 CEMU_SPARF_READ_DIAGNOSTICS=1
# Ordinary runs omit the extra trace; enable it explicitly for stack breakdowns.
export CEMU_NATIVE_IO_TRACE=${CEMU_NATIVE_IO_TRACE:-0}
if [[ ${CEMU_NATIVE_DRY_RUN:-0} != 1 ]]; then
    if [[ $(id -u) != 0 ]]; then
        echo 'Run with sudo on the Host after shutting down the previous CEMU Guest.' >&2
        exit 1
    fi
    if fuser "$CEMU_NATIVE_OS_IMAGE" >/dev/null 2>&1; then
        echo 'Guest image is still in use; shut down the previous Guest first.' >&2
        exit 1
    fi
fi
echo 'Microbatch comparison Guest command: bash /root/CEMU/femu-scripts/flexgen-b4-microbatch-guest-test.sh [2|4|1]; always D16.' >&2
exec bash "$repo/femu-scripts/run-native-opt-test.sh" OPT-13B
