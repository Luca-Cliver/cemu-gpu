#!/bin/bash
# Offline copy: preserve the source and flatten its backing chain onto /home.
set -euo pipefail

repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
workspace=$(dirname -- "$repo")
source_image=${CEMU_VLLM_SOURCE_IMAGE:-/data/lihan/cemu-native-opt/standalone.PnutcI/guest.qcow2}
destination_dir="$workspace/cemu-vllm-runtime"
destination_image="$destination_dir/guest.qcow2"
qemu_img="$repo/build/qemu-img"

if [[ $EUID != 0 ]]; then
    echo 'Run with sudo on the Host after shutting down the Guest.' >&2
    exit 1
fi
for command in fuser flock df python3; do
    command -v "$command" >/dev/null || { echo "Missing command: $command" >&2; exit 1; }
done
[[ -x "$qemu_img" && -f "$source_image" ]] || {
    echo "Missing qemu-img or source image: $source_image" >&2; exit 1;
}
source_image=$(readlink -f -- "$source_image")
if fuser "$source_image" >/dev/null 2>&1; then
    echo 'Source image is still in use. Shut down the Guest and wait for QEMU to exit.' >&2
    exit 1
fi
mkdir -p "$destination_dir"
exec 9> "$destination_dir/migrate.lock"
flock -n 9 || { echo 'Another migration is running.' >&2; exit 1; }
[[ ! -e "$destination_image" && ! -L "$destination_image" ]] || {
    echo "Destination already exists; refusing to overwrite: $destination_image" >&2; exit 1;
}
available=$(df -B1 --output=avail "$destination_dir" | tail -1)
if (( available < 100 * 1024 * 1024 * 1024 )); then
    echo 'Need at least 100 GiB free on the destination filesystem for this 80 GiB Guest and test data.' >&2
    exit 1
fi
source_size=$("$qemu_img" info --output=json "$source_image" |
    python3 -c 'import json,sys; print(json.load(sys.stdin)["virtual-size"])')
if (( source_size > 80 * 1024 * 1024 * 1024 )); then
    echo 'Source is already larger than 80 GiB; refusing to shrink it.' >&2
    exit 1
fi

temporary_dir=$(mktemp -d "$destination_dir/.migrate.XXXXXX")
trap 'rm -f -- "$temporary_dir/guest.qcow2"; rmdir -- "$temporary_dir"' EXIT
temporary_image="$temporary_dir/guest.qcow2"
echo 'Copying the stopped Guest and all backing data into a standalone image...'
"$qemu_img" convert -p -f qcow2 -O qcow2 "$source_image" "$temporary_image"
echo 'Verifying logical disk contents against the source...'
"$qemu_img" compare -p -f qcow2 -F qcow2 "$source_image" "$temporary_image"
"$qemu_img" check -f qcow2 "$temporary_image"
"$qemu_img" resize -f qcow2 "$temporary_image" 80G
# A hard link publishes without overwriting a concurrently created destination.
ln -- "$temporary_image" "$destination_image"
"$qemu_img" info "$destination_image"
printf 'Copied Guest: %s\nOriginal retained: %s\n' "$destination_image" "$source_image"
echo 'Boot with run-vllm-opt-test.sh, then grow /dev/vda1 and its ext4 filesystem inside the Guest.'
