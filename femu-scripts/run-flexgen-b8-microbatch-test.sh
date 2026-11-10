#!/bin/bash
set -euo pipefail

repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
export CEMU_NATIVE_BOOT_MODE=normal
export CEMU_NATIVE_OS_IMAGE=${CEMU_NATIVE_OS_IMAGE:-/data/lihan/cemu-native-opt/standalone.PnutcI/guest.qcow2}
export CEMU_NATIVE_PROFILE=b8-p1024-d16
# Keep the preceding B4 experiment's hardware: 32 LUNs, 11703 ns/page.
# The historical B8 default instead has 48 LUNs and 62500 ns/page.
export CEMU_NATIVE_CONFIG="$repo/tests/cemu/instattention/experiments/configs/fdm_opt13b_b4_p1024_g1024_lun32.json"
export CEMU_NATIVE_CSD_TARGET=cuda CEMU_NATIVE_KV_PLACEMENT=balanced
export CEMU_DEVICE_INIT_ORDER=femu-first
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}
export CEMU_CUDA_DEVICE=0
export CEMU_CUDA_LIBRARY_PATH=${CEMU_CUDA_LIBRARY_PATH:-/home/lihan/cemu/nvidia-575-libs}
export CEMU_NATIVE_TIMELINE=1 CEMU_SPARF_READ_DIAGNOSTICS=1
export CEMU_NATIVE_IO_TRACE=1 CEMU_FDM_AGGREGATE_PROFILE=1
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
echo 'B8 comparison: two B4 microbatches, D16, same 32-LUN device as B4; all traces enabled.' >&2
echo 'Guest: bash /root/CEMU/femu-scripts/flexgen-b8-microbatch-guest-test.sh' >&2
exec bash "$repo/femu-scripts/run-native-opt-test.sh" OPT-13B
