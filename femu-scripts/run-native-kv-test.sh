#!/bin/bash
set -euo pipefail

repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
workspace=$(dirname -- "$repo")
qemu=${CEMU_QEMU_BINARY:-$repo/build/qemu-system-x86_64}
qemu_img=${CEMU_QEMU_IMG:-$repo/build/qemu-img}
kernel=${CEMU_NATIVE_KERNEL:-$workspace/linux-cemu/vmlinux}
initrd=${CEMU_NATIVE_INITRD:-$workspace/dqib_amd64-pc/initrd}
boot_image=${CEMU_NATIVE_BOOT_IMAGE:-$workspace/dqib_amd64-pc/image.qcow2}
config=$repo/tests/cemu/instattention/experiments/configs/native_kv_smoke.json

for path in "$qemu" "$qemu_img" "$kernel" "$initrd" "$boot_image" "$config"; do
    if [[ ! -r "$path" ]]; then
        printf 'Required file not readable: %s\n' "$path" >&2
        exit 1
    fi
done
if [[ ! -r /dev/kvm || ! -w /dev/kvm ]]; then
    echo 'KVM access required; ask the administrator for access or run this script with sudo.' >&2
    exit 1
fi
directory=$(mktemp -d "${CEMU_NATIVE_TEST_PARENT:-/tmp}/cemu-native-kv.XXXXXX")
boot_image=$(readlink -f -- "$boot_image")
"$qemu_img" create -f qcow2 -F qcow2 -b "$boot_image" "$directory/guest.qcow2"
printf 'Native KV test directory: %s\n' "$directory"
echo 'Uses a new 16MiB NVM file and temporary Guest OS overlay; no GPU passthrough.'
echo 'Stop any VM using the OS backing image first. Do not format or mount the native NVMe namespace.'
echo 'Boots directly into bash; skips old automatic cemu-mount/fio initialization.'
echo 'In Guest: mountpoint -q /proc || mount -t proc proc /proc'
echo 'In Guest: mountpoint -q /sys || mount -t sysfs sysfs /sys'
echo 'In Guest: mountpoint -q /dev || mount -t devtmpfs devtmpfs /dev'
echo 'In Guest: mount -o remount,rw /'
echo 'In Guest: mkdir -p /root/CEMU; mount -t 9p -o trans=virtio,version=9p2000.L cemusrc /root/CEMU'
echo 'Then: cd /root/CEMU/tests/cemu; make instattention-native-device-test NVM_DEVICE=/dev/ng0n1'
echo 'Buffer test (once per VM): make instattention-native-buffer-device-test NVM_DEVICE=/dev/ng0n1'
echo 'After the test: sync; /sbin/poweroff -f (bash is PID 1; no systemd is running).'
cd "$directory"
env CEMU_CUDA_SYNC=0 "$qemu" \
    -name CEMU-native-kv-smoke \
    -machine q35,accel=kvm -cpu host -smp 2 -m 2G \
    -kernel "$kernel" -initrd "$initrd" \
    -drive "file=$directory/guest.qcow2,if=virtio,format=qcow2" \
    -device "femu,ftl_mode=instattention,config_file=$config,nvm_backend_file=$directory/nvm.img" \
    -append 'root=LABEL=rootfs console=ttyS0 init=/bin/bash' -nographic -nic none \
    -fsdev "local,path=$repo,security_model=none,readonly=on,id=cemusrc" \
    -device virtio-9p-pci,fsdev=cemusrc,mount_tag=cemusrc \
    -qmp "unix:$directory/qmp.sock,server=on,wait=off" \
    2>&1 | tee "$directory/qemu.log"
