#!/bin/bash
set -euo pipefail

repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
workspace=$(dirname -- "$repo")
qemu=${CEMU_QEMU_BINARY:-$repo/build/qemu-system-x86_64}
qemu_img=${CEMU_QEMU_IMG:-$repo/build/qemu-img}
kernel=${CEMU_NATIVE_KERNEL:-$workspace/linux-cemu/vmlinux}
initrd=${CEMU_NATIVE_INITRD:-$workspace/dqib_amd64-pc/initrd}
boot_image=${CEMU_NATIVE_BOOT_IMAGE:-$workspace/dqib_amd64-pc/image.qcow2}
models=${CEMU_NATIVE_MODEL_ROOT:-$workspace/models}
gpu=${CEMU_VFIO_GPU:-0000:27:00.0}
parent=${CEMU_NATIVE_TEST_PARENT:-/data/lihan/cemu-native-opt}
config=$repo/tests/cemu/instattention/experiments/configs/native_opt125m_smoke.json
csd_target=${CEMU_NATIVE_CSD_TARGET:-cpu}
timeline=${CEMU_NATIVE_TIMELINE:-0}
vcpus=${CEMU_NATIVE_VCPUS:-4}
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
model=${1:-OPT-125M}
boot_mode=${CEMU_NATIVE_BOOT_MODE:-rescue}
os_image=${CEMU_NATIVE_OS_IMAGE:-}
profile=${CEMU_NATIVE_PROFILE:-smoke}
page_limit=65536
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
        buffer_budget=1073741824
        page_limit=2097152
        ;;
    *) echo 'CEMU_NATIVE_PROFILE must be smoke or b4-p1024-d16.' >&2; exit 1 ;;
esac
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
cp -- "$config" "$directory/device-config.json"
config=$directory/device-config.json
timeline_path=
if [[ "$timeline" == 1 ]]; then
    timeline_path=$directory/guest-logs/device-timeline.jsonl
fi
printf '%s\n' "csd_target=$csd_target" "program=$operator" "profile=$profile" \
    "guest_memory=$memory" "guest_vcpus=$vcpus" "native_buffer_budget=$buffer_budget" "native_page_limit=$page_limit" \
    "timeline=$timeline_path" "cuda_library_path=${CEMU_CUDA_LIBRARY_PATH:-}" \
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
echo 'Stop any VM using the OS backing image or this VFIO GPU before starting.'
if [[ "$boot_mode" == normal ]]; then
    echo 'Normal systemd/login boot; use only the updated native-aware cemu-mount.sh. SSH: localhost:2222.'
else
    echo 'Rescue bash boot; do not run the old cemu-mount.sh on native NVM.'
fi
echo 'Do NOT format or mount native NVM as a filesystem.'
echo 'Guest setup and OPT command: tests/cemu/instattention/README.md, Native Guest OPT-125M section.'
echo 'Shutdown: poweroff in normal mode; sync; /sbin/poweroff -f in rescue mode. Each launch creates a fresh native NVM session.'

cd "$repo/tests/cemu"
env "${cuda_env[@]}" CEMU_COMPUTE_LOG=progress CEMU_COMPUTE_LOG_INTERVAL=1000 \
    CEMU_SPARF_AUTONOMOUS_TIMELINE="$timeline_path" CEMU_SPARF_READ_DIAGNOSTICS=0 \
    "$qemu" -name "CEMU-native-$model-smoke" \
    -machine q35,accel=kvm -global q35-pcihost.pci-hole64-size=256G \
    -cpu host -smp "$vcpus" -m "$memory" \
    -device "vfio-pci,host=$gpu,id=guestgpu0" \
    -kernel "$kernel" -initrd "$initrd" \
    -drive "file=$os_image,if=virtio,format=qcow2" \
    -device "femu,ftl_mode=instattention,config_file=$config,nvm_backend_file=$directory/nvm.img,native_kv_buffer_budget=$buffer_budget,native_kv_page_limit=$page_limit" \
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
    -qmp "unix:$directory/qmp.sock,server=on,wait=off" \
    2>&1 | tee "$directory/qemu.log"
