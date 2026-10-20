#!/bin/bash
set -euo pipefail

repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
workspace=$(dirname -- "$repo")
image=${1:?Usage: bash run-freeze-clock-test.sh /absolute/path/to/stopped-guest.qcow2}
image=$(realpath -- "$image")
qemu=$repo/build/qemu-system-x86_64
qemu_img=$repo/build/qemu-img
kernel=$workspace/linux-cemu/vmlinux
initrd=$workspace/dqib_amd64-pc/initrd
config=$repo/tests/cemu/instattention/experiments/configs/native_csd_smoke.json
parent=${CEMU_CLOCK_TEST_PARENT:-/data/lihan/cemu-clock-probe}
accelerator=${CEMU_CLOCK_ACCEL:-kvm}
cpu=${CEMU_CLOCK_CPU:-qemu64}
case "$accelerator" in
    kvm|tcg) ;;
    *) echo 'CEMU_CLOCK_ACCEL must be kvm or tcg (no fallback).' >&2; exit 1 ;;
esac
if [[ "$accelerator" == tcg && "$cpu" == host ]]; then
    echo 'TCG cannot use -cpu host; use the same CEMU_CLOCK_CPU=qemu64 for both runs.' >&2
    exit 1
fi
for required in "$image" "$qemu" "$qemu_img" "$kernel" "$initrd" "$config" \
    "$repo/tests/cemu/build/freeze_clock_probe.so"; do
    test -r "$required" || { echo "Missing: $required" >&2; exit 1; }
done
mkdir -p "$parent"
session=$(mktemp -d "$parent/clock.XXXXXX")
mkdir "$session/guest-logs"
"$qemu_img" create -q -f qcow2 -F qcow2 -b "$image" "$session/guest.qcow2"
printf 'Clock probe session: %s\n' "$session"
printf 'Accelerator: %s; CPU model: %s; no accelerator fallback.\n' "$accelerator" "$cpu"
{
    printf 'accelerator=%s\ncpu=%s\nbacking_image=%s\n' "$accelerator" "$cpu" "$image"
    uname -a
    sha256sum "$qemu" "$kernel" "$initrd" "$config" "$repo/tests/cemu/build/freeze_clock_probe.so"
} > "$session/environment.txt"
echo 'Stop any VM using the OS backing image before running this test.'
echo 'Legacy FEMU mode, fresh 16MiB NVM, 8MiB FDM, 4GiB Guest RAM, no VFIO or CUDA.'
echo 'OS writes go to a new overlay. Do not modify the backing image while it is in use.'
echo 'No native KV or model inference. Do not run cemu-mount.sh; mount FDM only as documented.'

env CEMU_CUDA_SYNC=0 CEMU_COMPUTE_LOG=off CEMU_COMPUTE_CLOCK_TRACE=1 \
    "$qemu" -name CEMU-freeze-clock-probe -machine "q35,accel=$accelerator" -cpu "$cpu" \
    -smp 4 -m 4G -kernel "$kernel" -initrd "$initrd" \
    -drive "file=$session/guest.qcow2,if=virtio,format=qcow2" \
    -device "femu,ftl_mode=femu,config_file=$config,nvm_backend_file=$session/nvm.img" \
    -append "root=LABEL=rootfs console=ttyS0 cemu_clock_probe=1 cemu_clock_accel=$accelerator cemu_clock_cpu=$cpu" \
    -nic user,model=virtio-net-pci,hostfwd=tcp:127.0.0.1:2222-:22 -nographic \
    -fsdev "local,path=$repo,security_model=none,readonly=on,id=cemusrc" \
    -device virtio-9p-pci,fsdev=cemusrc,mount_tag=cemusrc \
    -fsdev "local,path=$workspace/linux-cemu,security_model=none,readonly=on,id=linuxsrc" \
    -device virtio-9p-pci,fsdev=linuxsrc,mount_tag=linuxsrc \
    -fsdev "local,path=$workspace/models,security_model=none,readonly=on,id=modelsrc" \
    -device virtio-9p-pci,fsdev=modelsrc,mount_tag=modelsrc \
    -fsdev "local,path=$session/guest-logs,security_model=none,id=logsrc" \
    -device virtio-9p-pci,fsdev=logsrc,mount_tag=logsrc \
    2>&1 | tee "$session/qemu.log"
