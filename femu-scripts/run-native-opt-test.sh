#!/bin/bash
set -euo pipefail

repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
workspace=$(dirname -- "$repo")
# Keep Host NUMA sampling faults out of device aggregation timing by default.
# Use every allowed memory node; this is not a single-node capacity restriction.
export CEMU_NATIVE_NUMA_POLICY=${CEMU_NATIVE_NUMA_POLICY:-bind-all}
numa_prefix=()
case "$CEMU_NATIVE_NUMA_POLICY" in
    bind-all) numa_prefix=(numactl --membind=all); expected_policy=bind ;;
    default) expected_policy=default ;;
    *) echo 'CEMU_NATIVE_NUMA_POLICY must be bind-all or default.' >&2; exit 1 ;;
esac
command -v numactl >/dev/null || { echo 'numactl is required to verify the Host memory policy.' >&2; exit 1; }
numa_policy_output=$("${numa_prefix[@]}" numactl --show)
if ! [[ $numa_policy_output == "policy: $expected_policy"$'\n'* ]]; then
    printf 'Unexpected Host memory policy; expected %s:\n%s\n' "$expected_policy" "$numa_policy_output" >&2
    exit 1
fi
printf 'Host NUMA policy: %s (verified %s).\n' "$CEMU_NATIVE_NUMA_POLICY" "$expected_policy"
qemu=${CEMU_QEMU_BINARY:-$repo/build/qemu-system-x86_64}
qemu_img=${CEMU_QEMU_IMG:-$repo/build/qemu-img}
kernel=${CEMU_NATIVE_KERNEL:-$workspace/linux-cemu/vmlinux}
initrd=${CEMU_NATIVE_INITRD:-$workspace/dqib_amd64-pc/initrd}
boot_image=${CEMU_NATIVE_BOOT_IMAGE:-$workspace/dqib_amd64-pc/image.qcow2}
models=${CEMU_NATIVE_MODEL_ROOT:-$workspace/models}
vllm_share_args=()
if [[ -n "${CEMU_NATIVE_VLLM_ROOT:-}" ]]; then
    if [[ ! -f "$CEMU_NATIVE_VLLM_ROOT/pyproject.toml" ]]; then
        echo 'CEMU_NATIVE_VLLM_ROOT must point to the vLLM source checkout.' >&2
        exit 1
    fi
    vllm_share_args=(-fsdev "local,path=$CEMU_NATIVE_VLLM_ROOT,security_model=none,readonly=on,id=vllmsrc"
                     -device virtio-9p-pci,fsdev=vllmsrc,mount_tag=vllmsrc)
fi
gpu=${CEMU_VFIO_GPU:-0000:27:00.0}
parent=${CEMU_NATIVE_TEST_PARENT:-/data/lihan/cemu-native-opt}
config=$repo/tests/cemu/instattention/experiments/configs/native_opt125m_smoke.json
csd_target=${CEMU_NATIVE_CSD_TARGET:-cpu}
timeline=${CEMU_NATIVE_TIMELINE:-0}
io_trace=${CEMU_NATIVE_IO_TRACE:-0}
aggregate_profile=${CEMU_FDM_AGGREGATE_PROFILE:-0}
async_aggregate=${CEMU_FDM_ASYNC_AGGREGATE:-0}
if [[ "$async_aggregate" != 0 && "$async_aggregate" != 1 ]]; then
    echo 'CEMU_FDM_ASYNC_AGGREGATE must be 0 or 1' >&2
    exit 1
fi
if [[ "$aggregate_profile" != 0 && "$aggregate_profile" != 1 ]] ||
   [[ "$aggregate_profile" == 1 && "$io_trace" != 1 ]]; then
    echo 'CEMU_FDM_AGGREGATE_PROFILE must be 0 or 1; enabling it requires CEMU_NATIVE_IO_TRACE=1.' >&2
    exit 1
fi
if [[ "$io_trace" != 0 && "$io_trace" != 1 ]]; then
    echo 'CEMU_NATIVE_IO_TRACE must be 0 or 1.' >&2
    exit 1
fi
read_diagnostics=${CEMU_SPARF_READ_DIAGNOSTICS:-0}
kv_placement=${CEMU_NATIVE_KV_PLACEMENT:-legacy}
device_init_order=${CEMU_DEVICE_INIT_ORDER:-vfio-first}
vcpus=${CEMU_NATIVE_VCPUS:-4}
if [[ "$kv_placement" != legacy && "$kv_placement" != balanced ]]; then
    echo 'CEMU_NATIVE_KV_PLACEMENT must be legacy or balanced.' >&2
    exit 1
fi
if [[ "$device_init_order" != vfio-first && "$device_init_order" != femu-first ]]; then
    echo 'CEMU_DEVICE_INIT_ORDER must be vfio-first or femu-first.' >&2
    exit 1
fi
if ! [[ "$vcpus" =~ ^[1-9][0-9]*$ ]]; then
    echo 'CEMU_NATIVE_VCPUS must be a positive integer.' >&2
    exit 1
fi
cuda_env=(CEMU_CUDA_SYNC=0)
case "$csd_target" in
    cpu) operator=$repo/tests/cemu/build/sparf_attention.so ;;
    cuda)
        operator=$repo/tests/cemu/build/sparf_attention_devptr.so
        if [[ -z "${CUDA_VISIBLE_DEVICES:-}" || "$CUDA_VISIBLE_DEVICES" == *,* || "$CUDA_VISIBLE_DEVICES" == -1 ]]; then
            echo 'CUDA requires one explicitly selected Host GPU in CUDA_VISIBLE_DEVICES (index or UUID).' >&2
            exit 1
        fi
        if [[ "${CEMU_CUDA_DEVICE:-0}" != 0 ]]; then
            echo 'CEMU_CUDA_DEVICE must be 0 within the single visible Host GPU.' >&2
            exit 1
        fi
        cuda_env=(CEMU_CUDA_SYNC=1 CEMU_CUDA_DEVICE=0 CUDA_DEVICE_ORDER=PCI_BUS_ID
                  "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES")
        if [[ -n "${CEMU_CUDA_LIBRARY_PATH:-}" ]]; then
            if [[ ! -d "$CEMU_CUDA_LIBRARY_PATH" ]]; then
                echo 'CEMU_CUDA_LIBRARY_PATH is not a directory.' >&2
                exit 1
            fi
            cuda_env+=("LD_LIBRARY_PATH=$CEMU_CUDA_LIBRARY_PATH${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}")
        fi
        ;;
    *) echo 'CEMU_NATIVE_CSD_TARGET must be cpu or cuda.' >&2; exit 1 ;;
esac
if [[ "$timeline" != 0 && "$timeline" != 1 ]]; then
    echo 'CEMU_NATIVE_TIMELINE must be 0 or 1.' >&2
    exit 1
fi
if [[ "$read_diagnostics" != 0 && "$read_diagnostics" != 1 ]]; then
    echo 'CEMU_SPARF_READ_DIAGNOSTICS must be 0 or 1.' >&2
    exit 1
fi
if [[ "$read_diagnostics" == 1 && "$timeline" != 1 ]]; then
    echo 'Read diagnostics require CEMU_NATIVE_TIMELINE=1.' >&2
    exit 1
fi
model=${1:-OPT-125M}
boot_mode=${CEMU_NATIVE_BOOT_MODE:-rescue}
os_image=${CEMU_NATIVE_OS_IMAGE:-}
profile=${CEMU_NATIVE_PROFILE:-smoke}
dry_run=${CEMU_NATIVE_DRY_RUN:-0}
if [[ "$dry_run" != 0 && "$dry_run" != 1 ]]; then
    echo 'CEMU_NATIVE_DRY_RUN must be 0 or 1.' >&2
    exit 1
fi
page_limit=${CEMU_NATIVE_PAGE_LIMIT:-65536}
if ! [[ "$page_limit" =~ ^[1-9][0-9]*$ ]] || (( page_limit > 16777216 )); then
    echo 'CEMU_NATIVE_PAGE_LIMIT must be in 1..16777216.' >&2
    exit 1
fi
case "$model" in
    OPT-125M)
        memory=8G
        buffer_budget=67108864
        weights=("$models/$model/pytorch_model.bin")
        ;;
    OPT-13B)
        memory=64G
        buffer_budget=536870912
        weights=("$models/$model/pytorch_model.bin.index.json"
                 "$models/$model/pytorch_model-00001-of-00003.bin"
                 "$models/$model/pytorch_model-00002-of-00003.bin"
                 "$models/$model/pytorch_model-00003-of-00003.bin")
        ;;
    *) echo 'Usage: bash run-native-opt-test.sh [OPT-125M|OPT-13B]' >&2; exit 1 ;;
esac
case "$profile" in
    smoke) ;;
    b4-p1024-d16)
        if [[ "$model" != OPT-13B ]]; then
            echo 'The b4-p1024-d16 profile requires OPT-13B.' >&2
            exit 1
        fi
        config=$repo/femu-scripts/cemu_config_opt13b_sparf_engines2.json
        # Raw rings plus bounded page staging for block-level native writeback.
        buffer_budget=6442450944
        page_limit=2097152
        ;;
    b8-p1024-d16)
        if [[ "$model" != OPT-13B ]]; then
            echo 'The b8-p1024-d16 profile requires OPT-13B.' >&2
            exit 1
        fi
        config=$repo/tests/cemu/instattention/experiments/configs/fdm_opt13b_b8_p1024_d16.json
        buffer_budget=12884901888
        page_limit=4194304
        ;;
    b4-p1024-g1024-lun32)
        if [[ "$model" != OPT-13B ]]; then
            echo 'The b4-p1024-g1024-lun32 profile requires OPT-13B.' >&2
            exit 1
        fi
        config=$repo/tests/cemu/instattention/experiments/configs/fdm_opt13b_b4_p1024_g1024_lun32.json
        buffer_budget=12884901888
        page_limit=4194304
        ;;
    *) echo 'CEMU_NATIVE_PROFILE must be smoke, b4-p1024-d16, b8-p1024-d16 or b4-p1024-g1024-lun32.' >&2; exit 1 ;;
esac
config=${CEMU_NATIVE_CONFIG:-$config}
if [[ "$profile" == b8-p1024-d16 ]]; then
    echo 'B8 FDM: namespace=16 GiB, Guest metadata=128 MiB, data=8 GiB across multiple MRs; input slot is sized by the Guest client for its KV packets.' >&2
fi
if [[ "$profile" == b4-p1024-g1024-lun32 ]]; then
    echo 'Bandwidth-equivalent experiment: 8 channels x 4 LUNs, 11703 ns/page read; ideal 11.2 GB/s, not calibrated NAND latency or a channel-bus model.' >&2
    echo 'Guest D16 microbatch test: bash /root/CEMU/femu-scripts/flexgen-b4-microbatch-guest-test.sh [2|4|1].' >&2
    echo 'Optional long test: CEMU_FLEXGEN_BATCH_SIZE=4 CEMU_FLEXGEN_DECODE_STEPS=1023 bash /root/CEMU/femu-scripts/flexgen-b8-guest-test.sh; output=1024 tokens.' >&2
fi
if [[ "$dry_run" == 1 ]]; then
    if [[ ! -r "$config" ]]; then
        printf 'Device config not readable: %s\n' "$config" >&2
        exit 1
    fi
    printf '%s\n' "model=$model" "profile=$profile" "device_config=$config" \
        "guest_memory=$memory" "guest_vcpus=$vcpus" \
        "native_buffer_budget=$buffer_budget" "native_page_limit=$page_limit" \
        "native_kv_placement=$kv_placement" "nvm_backend=file"
    echo 'Dry run only: no VM, GPU access, image creation, or hardware/resource validation.'
    exit 0
fi
boot_args=()
network_args=(-nic none)
case "$boot_mode" in
    rescue) boot_args+=(init=/bin/bash) ;;
    normal)
        if [[ ! -f "$os_image" ]]; then
            echo 'Normal boot requires CEMU_NATIVE_OS_IMAGE pointing to the stopped test overlay with the updated /root/cemu-mount.sh.' >&2
            exit 1
        fi
        network_args=(-netdev user,id=net0,hostfwd=tcp:127.0.0.1:2222-:22 -device virtio-net-pci,netdev=net0)
        ;;
    *) echo 'CEMU_NATIVE_BOOT_MODE must be rescue or normal.' >&2; exit 1 ;;
esac

for path in "$qemu" "$qemu_img" "$kernel" "$initrd" "${os_image:-$boot_image}" "$config" "$operator" \
            "$models/$model/config.json" "${weights[@]}"; do
    if [[ ! -r "$path" ]]; then
        printf 'Required file not readable: %s\n' "$path" >&2
        exit 1
    fi
done
if [[ ! -r /dev/kvm || ! -w /dev/kvm ]]; then
    echo 'KVM access required; run this launcher on the physical Host with sudo.' >&2
    exit 1
fi
if ! [[ "$gpu" =~ ^[[:xdigit:]]{4}:[[:xdigit:]]{2}:[[:xdigit:]]{2}\.[0-7]$ ]]; then
    echo 'CEMU_VFIO_GPU must be a PCI address such as 0000:27:00.0.' >&2
    exit 1
fi
driver=$(readlink -f "/sys/bus/pci/devices/$gpu/driver" || true)
if [[ "${driver##*/}" != vfio-pci ]]; then
    printf 'GPU %s is not bound to vfio-pci; refusing to unbind or reset it.\n' "$gpu" >&2
    exit 1
fi

mkdir -p -- "$parent"
directory=$(mktemp -d "$parent/${model,,}.XXXXXX")
mkdir -p "$directory/guest-logs"
"${numa_prefix[@]}" python3 - "$directory/guest-logs/host-execution-context.json" <<'PYHOST'
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
def read(path):
    try:
        return Path(path).read_text().strip()
    except OSError:
        return None
status = read('/proc/self/status') or ''
policy = None
if shutil.which('numactl'):
    probe = subprocess.run(['numactl', '--show'], capture_output=True, text=True)
    policy = {'returncode': probe.returncode, 'stdout': probe.stdout,
              'stderr': probe.stderr}
context = {
    'requested_numa_policy': os.environ['CEMU_NATIVE_NUMA_POLICY'],
    'numa_test_case': os.environ.get('CEMU_NUMA_TEST_CASE'),
    'host_kernel': platform.release(),
    'inherited_numa_policy': policy,
    'numa_balancing': read('/proc/sys/kernel/numa_balancing'),
    'transparent_hugepage': read('/sys/kernel/mm/transparent_hugepage/enabled'),
    'numa_node_cpus': {p.parent.name: read(p) for p in Path('/sys/devices/system/node').glob('node*/cpulist')},
    'launcher_affinity': {line.split(':', 1)[0]: line.split(':', 1)[1].strip()
                          for line in status.splitlines() if line.startswith(('Cpus_allowed_list:', 'Mems_allowed_list:'))},
    'note': 'Probe uses the same NUMA command prefix as QEMU; not a measurement of per-page placement or CPU residency during execution.',
}
expected = 'bind' if context['requested_numa_policy'] == 'bind-all' else 'default'
if not policy or policy['returncode'] or not policy['stdout'].startswith('policy: ' + expected + '\n'):
    raise SystemExit('Requested NUMA policy did not survive into launch probe')
Path(sys.argv[1]).write_text(json.dumps(context, indent=2) + '\n')
PYHOST
cp -- "$config" "$directory/device-config.json"
cp -- "$config" "$directory/guest-logs/device-config.json"
config=$directory/device-config.json
timeline_path=
io_trace_path=
if [[ "$io_trace" == 1 ]]; then
    io_trace_path=$directory/guest-logs/fdm-io-device.jsonl
    printf 'FDM I/O device trace: %s\n' "$io_trace_path"
fi
if [[ "$timeline" == 1 ]]; then
    timeline_path=$directory/guest-logs/device-timeline.jsonl
fi
printf '%s\n' "csd_target=$csd_target" "program=$operator" "profile=$profile" \
    "guest_memory=$memory" "guest_vcpus=$vcpus" "native_buffer_budget=$buffer_budget" "native_page_limit=$page_limit" \
    "timeline=$timeline_path" "fdm_io_trace=$io_trace_path" "aggregate_profile=$aggregate_profile" "fdm_async_aggregate=$async_aggregate" "read_diagnostics=$read_diagnostics" "device_init_order=$device_init_order" \
    "native_kv_placement=$kv_placement" \
    "host_numa_policy=$CEMU_NATIVE_NUMA_POLICY" \
    "cuda_library_path=${CEMU_CUDA_LIBRARY_PATH:-}" \
    "${cuda_env[@]}" > "$directory/guest-logs/launch-settings.txt"
models=$(readlink -f -- "$models")
if [[ -n "$os_image" ]]; then
    os_image=$(readlink -f -- "$os_image")
else
    boot_image=$(readlink -f -- "$boot_image")
    os_image=$directory/guest.qcow2
    "$qemu_img" create -f qcow2 -F qcow2 -b "$boot_image" "$os_image"
fi
printf 'Native OPT test directory: %s\n' "$directory"
printf 'Model: %s; Guest RAM: %s; native KV buffer budget: %s bytes\n' "$model" "$memory" "$buffer_budget"
printf 'Guest vCPUs: %s\n' "$vcpus"
printf 'Profile: %s; native page limit: %s; config snapshot: %s\n' "$profile" "$page_limit" "$config"
printf 'Guest OS image: %s; boot mode: %s\n' "$os_image" "$boot_mode"
printf 'Guest GPU: %s; CSD operator: %s; fresh file-backed native NVM.\n' "$gpu" "$csd_target"
if [[ "$csd_target" == cuda ]]; then
    printf 'CSD Host GPU: CUDA_VISIBLE_DEVICES=%s, local CUDA ordinal=0 (not the Guest GPU).\n' "$CUDA_VISIBLE_DEVICES"
fi
printf 'Device timeline: %s\n' "${timeline_path:-disabled}"
printf 'Native read diagnostics: %s (page/LUN counters; full-scan reference issues no IO).\n' "$read_diagnostics"
printf 'FDM asynchronous aggregation: %s (1: acknowledge input reception before aggregation).\n' "$async_aggregate"
printf 'Device initialization order: %s\n' "$device_init_order"
printf 'Native KV block placement: %s (shared by Native and FDM buffers)\n' "$kv_placement"
echo 'Stop any VM using the OS backing image or this VFIO GPU before starting.'
if [[ "$boot_mode" == normal ]]; then
    echo 'Normal systemd/login boot; use only the updated native-aware cemu-mount.sh. SSH: localhost:2222.'
else
    echo 'Rescue bash boot; do not run the old cemu-mount.sh on native NVM.'
fi
echo 'Do NOT format or mount native NVM as a filesystem.'
echo 'Guest setup and OPT command: tests/cemu/instattention/README.md, Native Guest OPT-125M section.'
echo 'Shutdown: poweroff in normal mode; sync; /sbin/poweroff -f in rescue mode. Each launch creates a fresh native NVM session.'

femu_device="femu,ftl_mode=instattention,config_file=$config,nvm_backend_file=$directory/nvm.img,native_kv_buffer_budget=$buffer_budget,native_kv_page_limit=$page_limit,native_kv_placement=$kv_placement"
vfio_device="vfio-pci,host=$gpu,id=guestgpu0"
early_device=$vfio_device
late_device=$femu_device
if [[ "$device_init_order" == femu-first ]]; then
    early_device=$femu_device
    late_device=$vfio_device
fi

cd "$repo/tests/cemu"
env "${cuda_env[@]}" CEMU_COMPUTE_LOG=progress CEMU_COMPUTE_LOG_INTERVAL=1000 \
    CEMU_FDM_IO_TRACE="$io_trace_path" CEMU_FDM_AGGREGATE_PROFILE="$aggregate_profile" CEMU_FDM_ASYNC_AGGREGATE="$async_aggregate" \
    CEMU_SPARF_AUTONOMOUS_TIMELINE="$timeline_path" CEMU_SPARF_READ_DIAGNOSTICS="$read_diagnostics" \
    "${numa_prefix[@]}" "$qemu" -name "CEMU-native-$model-smoke" \
    -machine q35,accel=kvm -global q35-pcihost.pci-hole64-size=256G \
    -cpu host -smp "$vcpus" -m "$memory" \
    -device "$early_device" \
    -kernel "$kernel" -initrd "$initrd" \
    -drive "file=$os_image,if=virtio,format=qcow2" \
    -device "$late_device" \
    -append "root=LABEL=rootfs console=ttyS0 ${boot_args[*]} cemu_native_opt_smoke=1 cemu_native_model=$model cemu_native_profile=$profile cemu_native_csd_target=$csd_target" \
    -nographic "${network_args[@]}" \
    -fsdev "local,path=$repo,security_model=none,readonly=on,id=cemusrc" \
    -device virtio-9p-pci,fsdev=cemusrc,mount_tag=cemusrc \
    -fsdev "local,path=$models,security_model=none,readonly=on,id=modelsrc" \
    -device virtio-9p-pci,fsdev=modelsrc,mount_tag=modelsrc \
    -fsdev "local,path=$workspace/linux-cemu,security_model=none,readonly=on,id=linuxsrc" \
    -device virtio-9p-pci,fsdev=linuxsrc,mount_tag=linuxsrc \
    -fsdev "local,path=$directory/guest-logs,security_model=none,id=logsrc" \
    -device virtio-9p-pci,fsdev=logsrc,mount_tag=logsrc \
    "${vllm_share_args[@]}" \
    -qmp "unix:$directory/qmp.sock,server=on,wait=off" \
    2>&1 | tee "$directory/qemu.log"
