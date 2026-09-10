#!/bin/bash

set -euo pipefail

gpu="${CEMU_VFIO_GPU:-0000:27:00.0}"
device_path="/sys/bus/pci/devices/$gpu"

if [[ ! "$gpu" =~ ^[[:xdigit:]]{4}:[[:xdigit:]]{2}:[[:xdigit:]]{2}\.[[:xdigit:]]$ ]]; then
    echo "Invalid PCI address: $gpu" >&2
    exit 1
fi

if [[ ! -d "$device_path" ]]; then
    echo "PCI device does not exist: $gpu" >&2
    exit 1
fi

if [[ "$(<"$device_path/vendor")" != "0x10de" ]]; then
    echo "Refusing to bind non-NVIDIA device: $gpu" >&2
    exit 1
fi

iommu_group_path="$(readlink -f "$device_path/iommu_group" 2>/dev/null || true)"
if [[ -z "$iommu_group_path" || ! -d "$iommu_group_path/devices" ]]; then
    echo "No IOMMU group found for $gpu; check BIOS IOMMU and kernel parameters" >&2
    exit 1
fi

mapfile -t iommu_devices < <(find "$iommu_group_path/devices" -mindepth 1 -maxdepth 1 -printf '%f\n' | sort)
if (( ${#iommu_devices[@]} != 1 )) || [[ "${iommu_devices[0]}" != "$gpu" ]]; then
    echo "Refusing partial VFIO binding; IOMMU group contains:" >&2
    printf '  %s\n' "${iommu_devices[@]}" >&2
    exit 1
fi

current_driver=""
if [[ -L "$device_path/driver" ]]; then
    current_driver="$(basename "$(readlink -f "$device_path/driver")")"
fi

if [[ "$current_driver" == "vfio-pci" ]]; then
    echo "VFIO ready: $gpu is already bound to vfio-pci"
    exit 0
fi

nvidia_information="/proc/driver/nvidia/gpus/$gpu/information"
if [[ "$current_driver" == "nvidia" && -r "$nvidia_information" ]] &&
   command -v fuser >/dev/null 2>&1; then
    device_minor="$(awk -F: '/Device Minor/ { gsub(/[[:space:]]/, "", $2); print $2 }' "$nvidia_information")"
    device_node="/dev/nvidia$device_minor"
    if [[ "$device_minor" =~ ^[0-9]+$ ]] &&
       sudo fuser "$device_node" >/dev/null 2>&1; then
        echo "Refusing to unbind busy GPU $gpu; processes are using $device_node:" >&2
        sudo fuser -v "$device_node" >&2 || true
        exit 1
    fi
fi

echo "Binding $gpu from ${current_driver:-no driver} to vfio-pci"
echo "[1/4] Loading vfio-pci"
sudo modprobe vfio-pci

if [[ -z "$current_driver" && -e "$device_path/reset" ]]; then
    echo "[recovery] Resetting the already-unbound device"
    printf '1\n' | sudo tee "$device_path/reset" >/dev/null
fi

echo "[2/4] Setting driver override"
printf '%s\n' vfio-pci | sudo tee "$device_path/driver_override" >/dev/null

if [[ -n "$current_driver" ]]; then
    echo "[3/4] Unbinding $current_driver (NVIDIA cleanup may take a while)"
    printf '%s\n' "$gpu" | sudo tee "$device_path/driver/unbind" >/dev/null
else
    echo "[3/4] Device is already unbound"
fi

echo "[4/4] Binding vfio-pci"
if ! printf '%s\n' "$gpu" | sudo tee /sys/bus/pci/drivers/vfio-pci/bind >/dev/null ||
   [[ ! -L "$device_path/driver" ]] ||
   [[ "$(basename "$(readlink -f "$device_path/driver")")" != "vfio-pci" ]]; then
    echo "VFIO binding failed; restoring ${current_driver:-automatic driver probing}" >&2
    printf '\n' | sudo tee "$device_path/driver_override" >/dev/null
    printf '%s\n' "$gpu" | sudo tee /sys/bus/pci/drivers_probe >/dev/null || true
    exit 1
fi

echo "VFIO ready: $gpu -> vfio-pci"
