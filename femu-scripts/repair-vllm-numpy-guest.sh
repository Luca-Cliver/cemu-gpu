#!/bin/bash
set -euo pipefail

if [[ $EUID != 0 ]] || ! grep -qw cemu_native_opt_smoke=1 /proc/cmdline; then
    echo 'Run as root inside the configured CEMU Guest.' >&2
    exit 1
fi
repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
python=/root/vllm-cu129/.venv/bin/python
export PATH="/root/.local/bin:$PATH"
command -v uv >/dev/null
test -x "$python"
"$python" -c 'import sys; assert sys.version_info[:2] == (3, 12)'
if pgrep -f '[v]llm.entrypoints|[v]llm_adapter|[V]LLM::|[E]ngineCore' >/dev/null; then
    echo 'Stop the active vLLM workload before repairing its environment.' >&2
    exit 1
fi
backup=$(mktemp /root/cemu-numpy-repair-before.XXXXXX.txt)
uv pip freeze --python "$python" > "$backup"
printf 'Environment snapshot: %s\n' "$backup"
shopt -s nullglob
wheels=("$repo"/tests/cemu/build/wheels/numpy-2.3.5-cp312-cp312-manylinux*x86_64.whl)
if [[ ${#wheels[@]} != 1 ]]; then
    echo 'Expected one NumPy 2.3.5 cp312 x86_64 wheel in tests/cemu/build/wheels; no changes made.' >&2
    exit 1
fi
uv pip install --python "$python" --offline --no-deps "${wheels[0]}"
uv pip check --python "$python"
"$python" - <<'PY'
import numpy
import numba
import torch

assert numpy.__version__ == "2.3.5"
print("Restored:", "NumPy", numpy.__version__, "Numba", numba.__version__, "Torch", torch.__version__)
PY
echo 'Original vLLM environment repaired; FlexGen and the incomplete shared environment were not modified.'
