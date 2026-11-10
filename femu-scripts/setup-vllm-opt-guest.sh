#!/bin/bash
# Run once in the Guest launched by run-vllm-opt-test.sh. Requires network access.
set -euo pipefail

if [[ $EUID != 0 ]] || ! grep -qw cemu_native_opt_smoke=1 /proc/cmdline; then
    echo 'Run as root inside the CEMU Guest, not on the Host.' >&2
    exit 1
fi
for command in curl git cmake c++ findmnt; do
    command -v "$command" >/dev/null || {
        echo 'Install Guest prerequisites: apt-get update && apt-get install -y curl ca-certificates git cmake build-essential libnuma1' >&2
        exit 1
    }
done
for directory in /root/CEMU /root/models /root/cemu-logs /mnt/vllm-src /mnt/fdm0; do
    mkdir -p "$directory"
done
for pair in 'cemusrc /root/CEMU' 'modelsrc /root/models' 'vllmsrc /mnt/vllm-src'; do
    read -r tag directory <<< "$pair"
    mountpoint -q "$directory" || mount -t 9p -o ro,trans=virtio,version=9p2000.L "$tag" "$directory"
done
mountpoint -q /root/cemu-logs || mount -t 9p -o trans=virtio,version=9p2000.L logsrc /root/cemu-logs
for device in /dev/nvme0c3 /dev/ng0n3; do
    [[ -c "$device" ]] || { echo "Missing or not a character device: $device" >&2; exit 1; }
done
# The memory namespace is registered with add_disk(), unlike the ioctl nodes.
[[ -b /dev/nvme0m2 ]] || {
    echo 'Missing or not a block device: /dev/nvme0m2' >&2; exit 1;
}
mountpoint -q /mnt/fdm0 || mount -t fdmfs /dev/nvme0m2 /mnt/fdm0
[[ $(findmnt -n -o FSTYPE --target /mnt/fdm0) == fdmfs ]] || {
    echo '/mnt/fdm0 must be an actual FDM filesystem.' >&2; exit 1;
}
nvidia-smi

export PATH="/root/.local/bin:$PATH"
if ! command -v uv >/dev/null; then
    shared_uv=/root/CEMU/tests/cemu/build/tools/uv
    if [[ -f "$shared_uv" ]]; then
        # A Host-downloaded official binary avoids Guest installer network access.
        install -D -m 0755 "$shared_uv" /root/.local/bin/uv
    else
        installer=$(mktemp /tmp/cemu-uv-install.XXXXXX)
        trap 'rm -f "$installer"' EXIT
        curl --connect-timeout 15 --max-time 60 -fLsS https://astral.sh/uv/install.sh -o "$installer"
        sh "$installer"
        rm -f "$installer"
        trap - EXIT
    fi
fi
uv --version
uv_install_args=()
if [[ ${CEMU_SETUP_VERBOSE:-0} == 1 ]]; then
    uv_install_args+=(--verbose)
fi

commit=77154ca6116ae34cbfe314beef1016bde9b63989
checkout=/root/vllm-cu129
if [[ ! -d "$checkout" ]]; then
    # Local clone checks the source Git directory as well as the worktree.
    # Trust only this read-only share, only for the current command.
    git -c safe.directory=/mnt/vllm-src \
        -c safe.directory=/mnt/vllm-src/.git \
        clone --no-hardlinks /mnt/vllm-src "$checkout"
    git -C "$checkout" checkout --detach "$commit"
fi
[[ $(git -C "$checkout" rev-parse HEAD) == "$commit" ]] || {
    echo "$checkout is not the pinned CUDA 12.9 revision; refusing to overwrite it." >&2; exit 1;
}
cd "$checkout"
[[ -x .venv/bin/python ]] || uv venv --python 3.12 .venv
# Use the same revision for Python sources and native extensions. Do not mix
# the current cu130-only checkout with an older cu129 wheel.
VLLM_USE_PRECOMPILED=1 VLLM_MAIN_CUDA_VERSION=12.9 \
VLLM_PRECOMPILED_WHEEL_VARIANT=cu129 VLLM_PRECOMPILED_WHEEL_COMMIT="$commit" \
    uv pip install "${uv_install_args[@]}" --python .venv/bin/python --editable . --torch-backend=cu129

adapter=/root/CEMU/tests/cemu/instattention
package=/root/cemu-vllm-adapter
mkdir -p "$package"
# The shared CEMU checkout is read-only; setuptools needs a writable package.
cp "$adapter/pyproject.toml" "$package/"
cp -a "$adapter/python" "$adapter/experiments" "$package/"
uv pip install "${uv_install_args[@]}" --python .venv/bin/python pybind11 --editable "$package"
cmake -S "$adapter" -B /root/cemu-vllm-build \
    -DPython3_EXECUTABLE="$checkout/.venv/bin/python" -DBUILD_TESTING=OFF \
    -DCMAKE_BUILD_TYPE=Release
cmake --build /root/cemu-vllm-build --target _cemu_client -j8

cat > /root/cemu-vllm-env.sh <<'ENV'
export VLLM=/root/vllm-cu129
export ADAPTER=/root/CEMU/tests/cemu/instattention
export PATH="$VLLM/.venv/bin:/root/.local/bin:$PATH"
export PYTHONPATH="/root/cemu-vllm-build:$ADAPTER/python:$ADAPTER"
export CUDA_VISIBLE_DEVICES=0
export CEMU_IO_BACKEND=ioctl
export VLLM_USE_V2_MODEL_RUNNER=0
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
ENV
source /root/cemu-vllm-env.sh
cd /root
"$VLLM/.venv/bin/python" - <<'PY'
from pathlib import Path
import inspect
import torch
import vllm
import _cemu_client
assert hasattr(_cemu_client.CemuClient, 'read_range_into'), 'Rebuild the CEMU pybind extension'
from vllm_adapter import register
register()
from vllm.v1.attention.backends.registry import AttentionBackendEnum
from vllm_adapter.backend import CemuAttentionBackend, CemuAttentionImpl
assert Path(vllm.__file__).resolve().is_relative_to(Path('/root/vllm-cu129'))
assert not inspect.isabstract(CemuAttentionBackend)
assert not inspect.isabstract(CemuAttentionImpl)
assert AttentionBackendEnum['CEMU'].get_path() == 'vllm_adapter.backend.CemuAttentionBackend'
assert torch.cuda.is_available(), 'Guest CUDA unavailable'
x = torch.ones((32, 32), device='cuda', dtype=torch.float16)
assert (x @ x)[0, 0].item() == 32
print('vLLM:', vllm.__version__, vllm.__file__)
print('Torch:', torch.__version__, 'CUDA:', torch.version.cuda)
print('Guest GPU:', torch.cuda.get_device_name(0))
print('CEMU plugin + pybind imports and Guest CUDA check passed; model test not yet run.')
PY
echo 'Ready. In each new shell: source /root/cemu-vllm-env.sh'
