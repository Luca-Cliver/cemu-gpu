#!/bin/bash

# copy this script to ./build-femu
# this script uses DBIQ image:
#   https://www.giovannimascellani.eu/dqib-debian-quick-image-baker.html
#   https://gitlab.com/api/v4/projects/giomasce%2Fdqib/jobs/artifacts/master/download?job=convert_amd64-pc
#
# use virtfs to mount the tests directory to the VM, this feature
# needs to be enabled when building qemu:
#   sudo apt install libcap-ng-dev libattr-dev
#   ../configure --enable-virtfs
#
# use ssh port 2222 to connect guest:
#   ssh -p 2222 root@localhost
#./auth-client -u lihan2025 -p lihan2025.123 auth
CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
CEMU_CUDA_SYNC="${CEMU_CUDA_SYNC:-1}"
CEMU_CUDA_DEVICE="${CEMU_CUDA_DEVICE:-0}"
CEMU_COMPUTE_LOG="${CEMU_COMPUTE_LOG:-1}"
CEMU_COMPUTE_LOG_INTERVAL="${CEMU_COMPUTE_LOG_INTERVAL:-100}"
CEMU_VFIO_GPU="${CEMU_VFIO_GPU:-0000:27:00.0}"
CEMU_CONFIG_FILE="${CEMU_CONFIG_FILE:-./cemu_config.json}"
CEMU_NVM_BACKEND_DIR="${CEMU_NVM_BACKEND_DIR:-}"
CEMU_CSD_COUNT="${CEMU_CSD_COUNT:-2}"
CEMU_GUEST_MEMORY="${CEMU_GUEST_MEMORY:-8G}"

if ! [[ "$CEMU_CSD_COUNT" =~ ^[1-9][0-9]*$ ]]; then
    echo "CEMU_CSD_COUNT must be a positive integer" >&2
    exit 1
fi

femu_device_args=()
if [[ -n "$CEMU_NVM_BACKEND_DIR" ]]; then
    mkdir -p "$CEMU_NVM_BACKEND_DIR" || exit 1
    echo "CEMU NVM backend: host files under $CEMU_NVM_BACKEND_DIR"
    for ((device_index = 0; device_index < CEMU_CSD_COUNT; device_index++)); do
        femu_device_args+=(
            -device "femu,config_file=$CEMU_CONFIG_FILE,nvm_backend_file=$CEMU_NVM_BACKEND_DIR/csd${device_index}-nvm.img"
        )
    done
else
    echo "CEMU NVM backend: host DRAM"
    for ((device_index = 0; device_index < CEMU_CSD_COUNT; device_index++)); do
        femu_device_args+=(
            -device "femu,config_file=$CEMU_CONFIG_FILE"
        )
    done
fi
echo "CEMU devices: $CEMU_CSD_COUNT, guest memory: $CEMU_GUEST_MEMORY, config: $CEMU_CONFIG_FILE"

sudo CUDA_DEVICE_ORDER="$CUDA_DEVICE_ORDER" \
    CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
    CEMU_CUDA_SYNC="$CEMU_CUDA_SYNC" CEMU_CUDA_DEVICE="$CEMU_CUDA_DEVICE" \
    CEMU_COMPUTE_LOG="$CEMU_COMPUTE_LOG" \
    CEMU_COMPUTE_LOG_INTERVAL="$CEMU_COMPUTE_LOG_INTERVAL" \
    ./x86_64-softmmu/qemu-system-x86_64 \
    -name "CEMU-DBIQ" \
    -machine q35,accel=kvm \
    -global q35-pcihost.pci-hole64-size=256G \
    -cpu host \
    -device vfio-pci,host="$CEMU_VFIO_GPU",id=guestgpu0 \
    -smp 8 \
    -m "$CEMU_GUEST_MEMORY" \
    -kernel ../../linux-cemu/vmlinux \
    -initrd ../../dqib_amd64-pc/initrd \
    -drive file=../../dqib_amd64-pc/image.qcow2,if=virtio,format=qcow2 \
    "${femu_device_args[@]}" \
    -net user,hostfwd=tcp::2222-:22 \
    -net nic,model=virtio \
    -append "root=LABEL=rootfs console=ttyS0" \
    -nographic \
    -fsdev local,path=../,security_model=none,id=cemusrc \
    -device virtio-9p-pci,fsdev=cemusrc,mount_tag=cemusrc \
    -fsdev local,path=../../linux-cemu,security_model=none,id=linuxsrc \
    -device virtio-9p-pci,fsdev=linuxsrc,mount_tag=linuxsrc \
    -fsdev local,path=../../models,security_model=none,id=modelsrc \
    -device virtio-9p-pci,fsdev=modelsrc,mount_tag=modelsrc \
    -qmp unix:./qmp-sock,server,nowait 2>&1 | tee log
