#!/bin/bash
set -euo pipefail

repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
qemu=${CEMU_QEMU_BINARY:-$repo/build/qemu-system-x86_64}
config=$repo/femu-scripts/cemu_config_opt13b_sparf_engines2.json
visible=${CUDA_VISIBLE_DEVICES:-}
if [[ -z "$visible" || "$visible" == *,* || "$visible" == -1 ]]; then
    echo 'Set CUDA_VISIBLE_DEVICES to one available Host GPU (index or UUID).' >&2
    exit 1
fi
if [[ ! -x "$qemu" || ! -r "$config" ]]; then
    echo 'QEMU binary or device configuration is unavailable.' >&2
    exit 1
fi
cuda_env=(CEMU_CUDA_SYNC=1 CEMU_CUDA_DEVICE=0 CUDA_DEVICE_ORDER=PCI_BUS_ID)
if [[ -n "${CEMU_CUDA_LIBRARY_PATH:-}" ]]; then
    if [[ ! -d "$CEMU_CUDA_LIBRARY_PATH" ]]; then
        echo 'CEMU_CUDA_LIBRARY_PATH must be an existing directory.' >&2
        exit 1
    fi
    cuda_env+=("LD_LIBRARY_PATH=$CEMU_CUDA_LIBRARY_PATH${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}")
fi
directory=$(mktemp -d /tmp/cemu-native-cuda-ab.XXXXXX)
printf 'Probe logs: %s; uid=%s; visible GPU=%s\n' "$directory" "$(id -u)" "$visible"
echo 'No Guest boot, VFIO, existing OS image, RAM preallocation or model weights.'
cp -- "$config" "$directory/device-config.json"
config=$directory/device-config.json
cd "$directory"
failed=0
for case_name in qtest-64M kvm-64M kvm-64G; do
    accelerator=${case_name%-*}
    memory=${case_name#*-}
    cpu_args=()
    if [[ "$accelerator" == kvm ]]; then
        cpu_args=(-cpu host)
    fi
    status=0
    printf 'quit\n' | timeout --kill-after=5s 30s env "${cuda_env[@]}" \
        "$qemu" -machine "q35,accel=$accelerator" "${cpu_args[@]}" \
        -m "$memory" -smp 4 -nodefaults -display none -monitor stdio -S \
        -device "femu,ftl_mode=instattention,config_file=$config,nvm_backend_file=$directory/$case_name.img,native_kv_buffer_budget=1073741824,native_kv_page_limit=2097152" \
        > "$directory/$case_name.log" 2>&1 || status=$?
    printf '\n=== %s: exit=%s ===\n' "$case_name" "$status"
    head -n 12 "$directory/$case_name.log"
    if [[ "$status" == 0 ]] &&
       grep -q 'Native namespace 1: CUDA backend initialized' "$directory/$case_name.log" &&
       grep -q 'Native namespace 2: CUDA backend initialized' "$directory/$case_name.log"; then
        echo "[cuda-probe] $case_name PASS (initialization only)"
    else
        echo "[cuda-probe] $case_name FAIL; inspect $directory/$case_name.log"
        failed=1
    fi
done
echo 'A PASS does not validate CUDA kernels or the full VFIO Guest launch.'
exit "$failed"
