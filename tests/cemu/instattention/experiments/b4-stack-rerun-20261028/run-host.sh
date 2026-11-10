#!/bin/bash
set -euo pipefail
repo=/home/lihan/cemu/CEMU
if [[ $(id -u) != 0 ]]; then
    echo 'Run this Host script with sudo after shutting down the existing CEMU Guest.' >&2
    exit 1
fi
image=/data/lihan/cemu-native-opt/standalone.PnutcI/guest.qcow2
if fuser "$image" >/dev/null 2>&1; then
    echo 'The existing Guest image is still open. Shut down that Guest and wait for QEMU to exit.' >&2
    exit 1
fi
cd "$repo"
exec env \
    CEMU_NATIVE_BOOT_MODE=normal \
    CEMU_NATIVE_OS_IMAGE="$image" \
    CEMU_NATIVE_PROFILE=b4-p1024-g1024-lun32 \
    CEMU_NATIVE_CONFIG="$repo/tests/cemu/instattention/experiments/configs/fdm_opt13b_b4_p1024_g1024_lun32.json" \
    CEMU_NATIVE_CSD_TARGET=cuda CEMU_NATIVE_KV_PLACEMENT=balanced \
    CEMU_DEVICE_INIT_ORDER=femu-first \
    CUDA_VISIBLE_DEVICES=1 CEMU_CUDA_DEVICE=0 \
    CEMU_CUDA_LIBRARY_PATH=/home/lihan/cemu/nvidia-575-libs \
    CEMU_NATIVE_TIMELINE=1 CEMU_SPARF_READ_DIAGNOSTICS=1 CEMU_NATIVE_IO_TRACE=1 \
    bash femu-scripts/run-native-opt-test.sh OPT-13B
