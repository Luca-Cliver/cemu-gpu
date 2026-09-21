#!/usr/bin/env bash
# Run inside the Guest. Host timeline recording must be enabled at QEMU start.
set -euo pipefail

test_directory="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$test_directory"
model_directory="${MODEL_DIR:-/root/models/OPT-13B}"
output_root="${RESULT_DIR:-/root/cemu-overlap}"
decode_steps="${DECODE_STEPS:-4}"
runtime_config="${RUNTIME_CONFIG:-instattention/experiments/configs/opt13b_sparf_1csd.json}"
mkdir -p -- "$output_root"
run_directory="$(mktemp -d "$output_root/opt13b-overlap-XXXXXX")"
ratio_file="$run_directory/softmax-mb2.json"
echo "[sparf-overlap] results=$run_directory; batch=4, microbatch=2, decode=$decode_steps"

make instattention-softmax-ratio-benchmark \
  RUNTIME_CONFIG="$runtime_config" SOFTMAX_DEVICE=cuda:0 \
  GPU_BATCH_SIZE=2 PROMPT_LENGTH=1024 DECODE_STEPS="$decode_steps" \
  SPARF_COMPRESSION_RATIO=8 SOFTMAX_RATIO_FILE="$ratio_file" \
  2>&1 | tee -- "$run_directory/ratio.log"

cases=(staged0 staged1)
if [[ "${COMPARE_KV_BUFFER:-0}" == 1 ]]; then
  cases+=(buffered)
fi
for case_name in "${cases[@]}"; do
  staged=1
  buffer_tokens=0
  if [[ "$case_name" == staged0 ]]; then staged=0; fi
  if [[ "$case_name" == buffered ]]; then buffer_tokens="${SPARF_KV_BUFFER_TOKENS:-4}"; fi
  make instattention-opt-cemu-gpu-test \
    MODEL_DIR="$model_directory" \
    NVM_TEST_DIR="${NVM_TEST_DIR:-/mnt/nvme0}" \
    FDM_TEST_DIR="${FDM_TEST_DIR:-/mnt/fdm0}" \
    CSD_TARGET=cuda ATTENTION_MODE=sparf \
    RUNTIME_CONFIG="$runtime_config" SOFTMAX_RATIO_FILE="$ratio_file" \
    FUNCTIONAL=1 BENCHMARK=0 TRACE_FILE= PROMPT_TEXT= \
    BATCH_SIZE=4 GPU_BATCH_SIZE=2 PROMPT_LENGTH=1024 DECODE_STEPS="$decode_steps" \
    ATTENTION_SLOTS=2 SPARF_STAGED="$staged" \
    SPARF_KV_BUFFER_TOKENS="$buffer_tokens" \
    SPARF_TOP_R=16 SPARF_COMPRESSION_RATIO=8 \
    PROFILE_OUTPUT="$run_directory/${case_name}-profile.csv" \
    2>&1 | tee -- "$run_directory/${case_name}.log"
done
python3 - "$run_directory" "${cases[@]}" <<'PY'
import ast
import csv
from pathlib import Path
import re
import sys

root = Path(sys.argv[1])
cases = sys.argv[2:]
sequences = []
for case_name in cases:
    text = (root / f"{case_name}.log").read_text()
    # Serial-console kernel messages can split a token array mid-number.
    text = re.sub(r'\[\s*\d+\.\d+\][^\n]*\n?', '', text)
    match = re.search(r'\[opt-functional\] PASS:[^\n]*tokens=(\[\[.*?\]\])', text)
    if match is None:
        raise SystemExit(f"[sparf-overlap] {case_name}: completed token sequence missing")
    sequences.append(ast.literal_eval(match[1]))
if any(sequence != sequences[0] for sequence in sequences[1:]):
    raise SystemExit("[sparf-overlap] token sequences differ; inspect logs before interpreting overlap")
print("[sparf-overlap] token sequences match across all modes")
profiles = []
for case_name in cases:
    with (root / f"{case_name}-profile.csv").open() as source:
        profiles.append({r["phase"]: float(r["total_seconds"]) for r in csv.DictReader(source)})
print("[sparf-overlap] accumulated seconds (rows can overlap): " + " / ".join(cases))
for name in ("pipeline.decode_wall", "csd_model.total_model_ns",
             "decode.nvm_append_wall", "decode.kv_publish_wall", "decode.execute_wait_wall",
             "kv_append.pack_wall", "kv_append.token_k_pwrite_wall",
             "kv_append.token_v_pwrite_wall", "kv_append.channel_k_mmap_wall",
             "kv_publish.token_k_fdatasync_wall", "kv_publish.channel_k_fdatasync_wall",
             "kv_publish.token_v_fdatasync_wall", "pipeline.decode_and_drain_wall",
             "pipeline.e2e_including_drain_wall", "pipeline.final_flush_wall",
             "decode.kv_buffer_accept_wall", "decode.kv_buffer_backpressure_wall",
             "decode.kv_buffer_snapshot_wall", "kv_writeback.total_wall"):
    values = [f"{p[name]:.6f}" if name in p else "unavailable" for p in profiles]
    print(f"  {name}: {' / '.join(values)}")
PY
echo "[sparf-overlap] completed; results=$run_directory; analyze the host timeline to check actual overlap"
