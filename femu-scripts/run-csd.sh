#!/bin/bash

set -o pipefail

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
CEMU_GUEST_CPUS="${CEMU_GUEST_CPUS:-8}"
native_os_explicit="${CEMU_OS_IMAGE:+1}"
CEMU_OS_IMAGE="${CEMU_OS_IMAGE:-../../dqib_amd64-pc/image.qcow2}"
CEMU_FTL_MODE="${CEMU_FTL_MODE:-femu}"
CEMU_DEVICE_INIT_ORDER="${CEMU_DEVICE_INIT_ORDER:-vfio-first}"

if [[ "$CEMU_DEVICE_INIT_ORDER" != vfio-first && "$CEMU_DEVICE_INIT_ORDER" != femu-first ]]; then
    echo "CEMU_DEVICE_INIT_ORDER must be vfio-first or femu-first" >&2
    exit 1
fi

if ! [[ "$CEMU_CSD_COUNT" =~ ^[1-9][0-9]*$ ]]; then
    echo "CEMU_CSD_COUNT must be a positive integer" >&2
    exit 1
fi

if ! [[ "$CEMU_GUEST_CPUS" =~ ^[1-9][0-9]*$ ]]; then
    echo "CEMU_GUEST_CPUS must be a positive integer" >&2
    exit 1
fi

native_device_options=
extra_boot_args=
extra_share_args=()
qemu_log=log
qmp_socket=./qmp-sock
case "$CEMU_FTL_MODE" in
    femu) ;;
    instattention)
        if [[ "$CEMU_CSD_COUNT" != 1 || -z "$CEMU_NVM_BACKEND_DIR" || -z "$native_os_explicit" ||
              ! -r "$CEMU_CONFIG_FILE" || ! -r "$CEMU_OS_IMAGE" ]]; then
            echo "Native mode requires one CSD, a file-backend directory, readable config and explicit CEMU_OS_IMAGE with native-aware cemu-mount.sh." >&2
            exit 1
        fi
        native_budget="${CEMU_NATIVE_KV_BUFFER_BUDGET:-1073741824}"
        native_pages="${CEMU_NATIVE_KV_PAGE_LIMIT:-2097152}"
        native_model="${CEMU_NATIVE_MODEL:-OPT-13B}"
        native_profile="${CEMU_NATIVE_PROFILE:-b4-p1024-d16}"
        if ! [[ "$native_budget" =~ ^[1-9][0-9]*$ && "$native_pages" =~ ^[1-9][0-9]*$ ]] ||
           [[ "$native_model" != OPT-13B && "$native_model" != OPT-125M ]] ||
           [[ "$native_profile" != smoke && "$native_profile" != b4-p1024-d16 ]] ||
           [[ "$native_profile" == b4-p1024-d16 && "$native_model" != OPT-13B ]]; then
            echo "Invalid native buffer/page budget, model or profile." >&2
            exit 1
        fi
        native_target=cuda
        case "${CEMU_CUDA_SYNC,,}" in
            0|false) native_target=cpu ;;
        esac
        if [[ "$native_target" == cuda &&
              ( "$CUDA_VISIBLE_DEVICES" == -1 || "$CUDA_VISIBLE_DEVICES" == *,* || "$CEMU_CUDA_DEVICE" != 0 ) ]]; then
            echo "Native CUDA requires one visible Host GPU and local CEMU_CUDA_DEVICE=0." >&2
            exit 1
        fi
        mkdir -p -- "$CEMU_NVM_BACKEND_DIR" || exit 1
        CEMU_NVM_BACKEND_DIR=$(mktemp -d "$CEMU_NVM_BACKEND_DIR/native.XXXXXX") || exit 1
        CEMU_NVM_BACKEND_DIR=$(cd -- "$CEMU_NVM_BACKEND_DIR" && pwd) || exit 1
        cp -- "$CEMU_CONFIG_FILE" "$CEMU_NVM_BACKEND_DIR/device-config.json" || exit 1
        CEMU_CONFIG_FILE="$CEMU_NVM_BACKEND_DIR/device-config.json"
        mkdir -- "$CEMU_NVM_BACKEND_DIR/guest-logs" || exit 1
        native_device_options=",ftl_mode=instattention,native_kv_buffer_budget=$native_budget,native_kv_page_limit=$native_pages"
        extra_boot_args="cemu_native_opt_smoke=1 cemu_native_model=$native_model cemu_native_profile=$native_profile cemu_native_csd_target=$native_target"
        extra_share_args=(-fsdev "local,path=$CEMU_NVM_BACKEND_DIR/guest-logs,security_model=none,id=logsrc"
                          -device virtio-9p-pci,fsdev=logsrc,mount_tag=logsrc)
        qemu_log="$CEMU_NVM_BACKEND_DIR/qemu.log"
        qmp_socket="$CEMU_NVM_BACKEND_DIR/qmp.sock"
        if [[ "${CEMU_NATIVE_TIMELINE:-0}" == 1 ]]; then
            CEMU_SPARF_AUTONOMOUS_TIMELINE="$CEMU_NVM_BACKEND_DIR/guest-logs/device-timeline.jsonl"
        fi
        printf 'Native session: %s\n' "$CEMU_NVM_BACKEND_DIR"
        printf 'Native OS: %s; CSD target: %s; profile: %s\n' "$CEMU_OS_IMAGE" "$native_target" "$native_profile"
        printf '%s\n' "ftl_mode=instattention" "config=$CEMU_CONFIG_FILE" \
            "os_image=$CEMU_OS_IMAGE" "guest_memory=$CEMU_GUEST_MEMORY" "guest_vcpus=$CEMU_GUEST_CPUS" \
            "csd_target=$native_target" "visible_gpu=$CUDA_VISIBLE_DEVICES" \
            "native_buffer_budget=$native_budget" "native_page_limit=$native_pages" \
            "profile=$native_profile" "device_init_order=$CEMU_DEVICE_INIT_ORDER" \
            "timeline=${CEMU_SPARF_AUTONOMOUS_TIMELINE:-}" \
            > "$CEMU_NVM_BACKEND_DIR/guest-logs/launch-settings.txt" || exit 1
        echo 'Use the native-aware cemu-mount.sh. Do NOT format or mount native NVM.'
        ;;
    *) echo "CEMU_FTL_MODE must be femu or instattention" >&2; exit 1 ;;
esac

femu_device_args=()
if [[ -n "$CEMU_NVM_BACKEND_DIR" ]]; then
    mkdir -p "$CEMU_NVM_BACKEND_DIR" || exit 1
    echo "CEMU NVM backend: host files under $CEMU_NVM_BACKEND_DIR"
    for ((device_index = 0; device_index < CEMU_CSD_COUNT; device_index++)); do
        femu_device_args+=(
            -device "femu,config_file=$CEMU_CONFIG_FILE,nvm_backend_file=$CEMU_NVM_BACKEND_DIR/csd${device_index}-nvm.img$native_device_options"
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
echo "CEMU device initialization order: $CEMU_DEVICE_INIT_ORDER"
early_device_args=(-device "vfio-pci,host=$CEMU_VFIO_GPU,id=guestgpu0")
late_device_args=("${femu_device_args[@]}")
if [[ "$CEMU_DEVICE_INIT_ORDER" == femu-first ]]; then
    early_device_args=("${femu_device_args[@]}" "${early_device_args[@]}")
    late_device_args=()
fi

cuda_library_args=()
if [[ -n "${CEMU_CUDA_LIBRARY_PATH:-}" ]]; then
    if [[ ! -d "$CEMU_CUDA_LIBRARY_PATH" ]]; then
        echo "CEMU_CUDA_LIBRARY_PATH is not a directory: $CEMU_CUDA_LIBRARY_PATH" >&2
        exit 1
    fi
    cuda_library_args+=("LD_LIBRARY_PATH=$CEMU_CUDA_LIBRARY_PATH${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}")
fi

sudo "${cuda_library_args[@]}" CUDA_DEVICE_ORDER="$CUDA_DEVICE_ORDER" \
    CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
    CEMU_CUDA_SYNC="$CEMU_CUDA_SYNC" CEMU_CUDA_DEVICE="$CEMU_CUDA_DEVICE" \
    CEMU_COMPUTE_LOG="$CEMU_COMPUTE_LOG" \
    CEMU_COMPUTE_LOG_INTERVAL="$CEMU_COMPUTE_LOG_INTERVAL" \
    CEMU_SPARF_TIMELINE="${CEMU_SPARF_TIMELINE:-}" \
    CEMU_SPARF_AUTONOMOUS_TIMELINE="${CEMU_SPARF_AUTONOMOUS_TIMELINE:-}" \
    CEMU_SPARF_READ_DIAGNOSTICS="${CEMU_SPARF_READ_DIAGNOSTICS:-0}" \
    CEMU_SPARF_GROUP_WRITE_ORDER="${CEMU_SPARF_GROUP_WRITE_ORDER:-fifo}" \
    CEMU_SPARF_FTL_TRACE="${CEMU_SPARF_FTL_TRACE:-}" \
    CEMU_SPARF_FTL_TRACE_READS="${CEMU_SPARF_FTL_TRACE_READS:-0}" \
    CEMU_SPARF_FTL_PLACEMENT="${CEMU_SPARF_FTL_PLACEMENT:-fifo}" \
    ./x86_64-softmmu/qemu-system-x86_64 \
    -name "CEMU-DBIQ" \
    -machine q35,accel=kvm \
    -global q35-pcihost.pci-hole64-size=256G \
    -cpu host \
    "${early_device_args[@]}" \
    -smp "$CEMU_GUEST_CPUS" \
    -m "$CEMU_GUEST_MEMORY" \
    -kernel ../../linux-cemu/vmlinux \
    -initrd ../../dqib_amd64-pc/initrd \
    -drive "file=$CEMU_OS_IMAGE,if=virtio,format=qcow2" \
    "${late_device_args[@]}" \
    -net user,hostfwd=tcp::2222-:22 \
    -net nic,model=virtio \
    -append "root=LABEL=rootfs console=ttyS0 $extra_boot_args" \
    -nographic \
    -fsdev local,path=../,security_model=none,id=cemusrc \
    -device virtio-9p-pci,fsdev=cemusrc,mount_tag=cemusrc \
    -fsdev local,path=../../linux-cemu,security_model=none,id=linuxsrc \
    -device virtio-9p-pci,fsdev=linuxsrc,mount_tag=linuxsrc \
    -fsdev local,path=../../models,security_model=none,id=modelsrc \
    -device virtio-9p-pci,fsdev=modelsrc,mount_tag=modelsrc \
    "${extra_share_args[@]}" \
    -qmp "unix:$qmp_socket,server,nowait" 2>&1 | tee "$qemu_log"
