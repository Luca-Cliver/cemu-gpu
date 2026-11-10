#!/bin/bash
set -euo pipefail

if [[ $EUID != 0 ]] || ! grep -qw cemu_native_opt_smoke=1 /proc/cmdline; then
    echo 'Run as root inside the configured CEMU Guest, not on the Host.' >&2
    exit 1
fi

repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
adapter="$repo/tests/cemu/instattention"
baseline=/root/vllm-cu129
prefix=/root/cemu-shared-cu129
checkout="$prefix/vllm"
venv="$prefix/venv"
build="$prefix/client-build"
package="$prefix/adapter"
commit=77154ca6116ae34cbfe314beef1016bde9b63989
export PATH="/root/.local/bin:$PATH"

for command in uv git cmake c++ flock; do
    command -v "$command" >/dev/null || { echo "Missing prerequisite: $command" >&2; exit 1; }
done
test -x "$baseline/.venv/bin/python"
test -x /root/cemu-gpu-env/bin/python3
test "$(git -C "$baseline" rev-parse HEAD)" = "$commit"
/root/cemu-gpu-env/bin/python3 -c 'import sys; assert sys.version_info[:2] == (3, 13), "Expected existing FlexGen Python 3.13"'
if [[ -e "$prefix" && ! -f "$prefix/.cemu-shared-environment" ]]; then
    echo "Refusing to reuse an unrecognized directory: $prefix" >&2
    exit 1
fi
mkdir -p "$prefix"
exec 9> "$prefix/setup.lock"
flock -n 9 || { echo 'Another shared environment setup is running.' >&2; exit 1; }
touch "$prefix/.cemu-shared-environment"
if [[ -e "$prefix/env.sh" ]]; then
    echo "Shared environment already published: $prefix/env.sh; refusing to modify it." >&2
    exit 1
fi
df -h "$prefix"
uv pip freeze --python "$baseline/.venv/bin/python" > "$prefix/baseline-freeze.txt"
"$baseline/.venv/bin/python" - "$prefix" <<'PY'
from importlib.metadata import version
import json
from pathlib import Path
import sys

root = Path(sys.argv[1])
names = ("torch", "torchvision", "torchaudio", "transformers", "tokenizers", "numba", "pybind11")
versions = {name: version(name) for name in names}
assert versions["numba"] == "0.65.0", "Expected the validated Numba 0.65.0 baseline"
(root / "baseline-versions.json").write_text(json.dumps(versions, indent=2) + "\n")
constraints = ["numpy==2.4.6"] + [f"{name}=={value}" for name, value in versions.items()]
(root / "constraints.txt").write_text("\n".join(constraints) + "\n")
print("Preserving baseline core dependencies:", versions, flush=True)
PY
if [[ ! -d "$checkout" ]]; then
    git clone --no-hardlinks "$baseline" "$checkout"
    git -C "$checkout" checkout --detach "$commit"
fi
test "$(git -C "$checkout" rev-parse HEAD)" = "$commit"
if [[ ! -x "$venv/bin/python" ]]; then
    uv venv --python /root/cemu-gpu-env/bin/python3 "$venv"
fi
"$venv/bin/python" -c 'import sys; assert sys.version_info[:2] == (3, 13)'
mkdir -p "$package"
cp -- "$adapter/pyproject.toml" "$package/"
cp -a -- "$adapter/python" "$adapter/experiments" "$package/"

cd "$checkout"
VLLM_USE_PRECOMPILED=1 VLLM_MAIN_CUDA_VERSION=12.9 \
VLLM_PRECOMPILED_WHEEL_VARIANT=cu129 VLLM_PRECOMPILED_WHEEL_COMMIT="$commit" \
    uv pip install --python "$venv/bin/python" --constraint "$prefix/constraints.txt" \
        --editable "$checkout" --editable "$package" pybind11 attrs tqdm \
        'numpy==2.4.6' --torch-backend=cu129
uv pip check --python "$venv/bin/python"
cmake -S "$adapter" -B "$build" -DPython3_EXECUTABLE="$venv/bin/python" \
    -DBUILD_TESTING=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build "$build" --target _cemu_client -j4

cat > "$prefix/env.pending.sh" <<ENV
export VLLM="$checkout"
export ADAPTER="$adapter"
export CEMU_CLIENT_BUILD_DIR="$build"
export PATH="$venv/bin:/root/.local/bin:\$PATH"
export PYTHONPATH="$build:$adapter/python:$adapter"
export CUDA_VISIBLE_DEVICES=0
export CEMU_IO_BACKEND=ioctl
export VLLM_USE_V2_MODEL_RUNNER=0
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
ENV
source "$prefix/env.pending.sh"
cd /root
"$venv/bin/python" - "$prefix" <<'PY'
from importlib.metadata import version
import inspect
import json
import os
from pathlib import Path
import sys

import numpy as np
import numba
import torch
import vllm
import vllm._C
import _cemu_client
from vllm import _custom_ops
from flexgen_adapter.framework_import import load_framework
from vllm_adapter import register

root = Path(sys.argv[1])
assert np.__version__ == "2.4.6"
assert numba.__version__ == "0.65.0"
assert Path(vllm.__file__).resolve().is_relative_to(Path(os.environ["VLLM"]))
assert Path(_cemu_client.__file__).resolve().is_relative_to(root / "client-build")
assert getattr(_cemu_client, "IO_PROFILE_VERSION", 0) == 1
assert hasattr(_cemu_client.CemuClient, "read_range_into")
framework = load_framework()
register()
from vllm.v1.attention.backends.registry import AttentionBackendEnum
from vllm_adapter.backend import CemuAttentionBackend, CemuAttentionImpl

assert not inspect.isabstract(CemuAttentionBackend)
assert not inspect.isabstract(CemuAttentionImpl)
assert AttentionBackendEnum["CEMU"].get_path() == "vllm_adapter.backend.CemuAttentionBackend"
assert torch.cuda.is_available(), "Guest CUDA unavailable"
matrix = torch.ones((32, 32), device="cuda", dtype=torch.float16)
assert (matrix @ matrix)[0, 0].item() == 32
normalized = torch.empty_like(matrix)
_custom_ops.rms_norm(normalized, matrix, torch.ones(32, device="cuda", dtype=torch.float16), 1e-5)
torch.testing.assert_close(normalized, matrix, atol=1e-3, rtol=1e-3)
array = np.arange(24, dtype=np.float16).reshape(3, 2, 4)
pinned = torch.empty(array.shape, dtype=torch.float16, pin_memory=True)
pinned.copy_(torch.from_numpy(array).cuda(), non_blocking=True)
torch.cuda.synchronize()
assert pinned.numpy().transpose(1, 0, 2).tobytes() == array.transpose(1, 0, 2).tobytes()
report = {
    "python": sys.version, "executable": sys.executable,
    "packages": {name: version(name) for name in
                 ("numpy", "numba", "torch", "torchvision", "torchaudio", "transformers",
                  "tokenizers", "vllm", "mistral-common", "pybind11", "attrs", "tqdm")},
    "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0),
    "vllm_source": vllm.__file__, "flexgen_source": framework.__file__,
    "client": _cemu_client.__file__, "full_model_test": "not_run",
}
(root / "environment.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2), flush=True)
PY
"$venv/bin/python" -m unittest -v test_flexgen_framework test_vllm_runtime
uv pip freeze --python "$venv/bin/python" > "$prefix/resolved-freeze.txt"
mv -- "$prefix/env.pending.sh" "$prefix/env.sh"
printf '\nShared environment ready: %s/env.sh\nOriginal environments were not modified. No NVM/FDM workload was run.\n' "$prefix"
