#!/bin/bash
set -euo pipefail

repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
workspace=$(dirname -- "$repo")
export CEMU_NATIVE_BOOT_MODE=normal
export CEMU_NATIVE_OS_IMAGE=${CEMU_NATIVE_OS_IMAGE:-$workspace/cemu-vllm-runtime/guest.qcow2}
export CEMU_NATIVE_TEST_PARENT=${CEMU_NATIVE_TEST_PARENT:-$workspace/cemu-vllm-runtime/runs}
case "${CEMU_VLLM_TEST_PROFILE:-smoke}" in
    smoke)
        guest_vcpus=8
        export CEMU_NATIVE_PROFILE=smoke CEMU_NATIVE_PAGE_LIMIT=262144
        export CEMU_NATIVE_CONFIG="$repo/tests/cemu/instattention/experiments/configs/vllm_opt13b_device_smoke.json"
        export CEMU_FDM_ASYNC_AGGREGATE=0 CEMU_SPARF_READ_DIAGNOSTICS=0
        echo 'vLLM OPT-13B smoke: B2, context <=128, 1 GiB FDM, 4 GiB NVM; functional validation only.'
        ;;
    b8-p1024-d16)
        guest_vcpus=4
        export CEMU_NATIVE_PROFILE=b8-p1024-d16
        export CEMU_NATIVE_CONFIG="$repo/tests/cemu/instattention/experiments/configs/fdm_opt13b_b4_p1024_g1024_lun32.json"
        export CEMU_FDM_ASYNC_AGGREGATE=1 CEMU_SPARF_READ_DIAGNOSTICS=1
        export CEMU_FDM_AGGREGATE_PROFILE=1
        echo 'vLLM OPT-13B experiment: B8/P1024/D16, 32 LUNs, 11703 ns/page, 64 GiB NVM, 16 GiB FDM.'
        echo '11.2 GB/s is an ideal bandwidth-equivalent setting, not calibrated physical NAND.'
        echo 'One scheduled B8 batch, not FlexGen two B4 microbatches. Guest: bash /root/CEMU/femu-scripts/vllm-opt-guest-test.sh'
        ;;
    *) echo 'CEMU_VLLM_TEST_PROFILE must be smoke or b8-p1024-d16' >&2; exit 1 ;;
esac
export CEMU_NATIVE_VLLM_ROOT=${CEMU_NATIVE_VLLM_ROOT:-$workspace/vllm}
export CEMU_NATIVE_CSD_TARGET=cuda CEMU_NATIVE_KV_PLACEMENT=balanced
export CEMU_DEVICE_INIT_ORDER=femu-first
export CEMU_NATIVE_VCPUS=${CEMU_NATIVE_VCPUS:-$guest_vcpus}
export CEMU_VFIO_GPU=${CEMU_VFIO_GPU:-0000:27:00.0}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}
export CEMU_CUDA_DEVICE=0
export CEMU_CUDA_LIBRARY_PATH=${CEMU_CUDA_LIBRARY_PATH:-$workspace/nvidia-575-libs}
export CEMU_NATIVE_TIMELINE=1
export CEMU_NATIVE_IO_TRACE=1
if [[ ${CEMU_NATIVE_DRY_RUN:-0} != 1 ]]; then
    if [[ $(id -u) != 0 ]]; then
        echo 'Run with sudo on the Host.' >&2
        exit 1
    fi
    if [[ ! -f "$CEMU_NATIVE_OS_IMAGE" ]]; then
        echo 'Guest image is missing. After shutting down the old Guest, run: sudo bash femu-scripts/migrate-vllm-guest.sh' >&2
        exit 1
    fi
    if fuser "$CEMU_NATIVE_OS_IMAGE" >/dev/null 2>&1; then
        echo 'Guest image is already in use; shut down that Guest first.' >&2
        exit 1
    fi
fi
echo 'Guest vLLM source share: mount -t 9p -o ro,trans=virtio,version=9p2000.L vllmsrc /mnt/vllm-src'
exec bash "$repo/femu-scripts/run-native-opt-test.sh" OPT-13B
