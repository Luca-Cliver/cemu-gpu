#!/usr/bin/env python3

import argparse
import math
import sys
import tempfile
from pathlib import Path

import numpy as np

PROJECT_DIRECTORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIRECTORY / "build-guest"))
sys.path.insert(0, str(PROJECT_DIRECTORY / "python"))

from cemu_flexgen import CemuSparfDevice, KvLayoutConfig, SparfKvCacheLayout, SparfKvCacheStore, sparf_attention
from test_cemu_attention_phases import require_fdmfs


def main():
    parser = argparse.ArgumentParser(description="Validate one-command device-side InstAttention SparF")
    parser.add_argument("--nvm-dir", default="/mnt/nvme0")
    parser.add_argument("--fdm-dir", default="/mnt/fdm0")
    parser.add_argument("--control", default="/dev/nvme0c3")
    parser.add_argument("--namespace", default="/dev/ng0n3")
    parser.add_argument("--program")
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--dtype", choices=("float16", "float32"), default="float16")
    parser.add_argument("--tokens", type=int, default=129)
    parser.add_argument("--top-r", type=int, default=4)
    parser.add_argument("--compression-ratio", type=int, default=8)
    args = parser.parse_args()
    require_fdmfs(args.fdm_dir)
    dtype = np.dtype(args.dtype)
    layout = SparfKvCacheLayout(KvLayoutConfig(2, args.tokens + 2, 1, 2, 16, dtype))
    random = np.random.default_rng(20260910)
    query = random.normal(size=(1, 4, 16)).astype(dtype) / np.asarray(math.sqrt(16), dtype=dtype)
    keys = random.normal(size=(args.tokens, 1, 2, 16)).astype(dtype)
    values = random.normal(size=keys.shape).astype(dtype)
    top_k = max(1, math.ceil(args.tokens / args.compression_ratio))
    expected = sparf_attention(query, keys, values, min(args.top_r, 16), top_k).output
    program = args.program or ("./build/sparf_attention_devptr.so" if args.cuda else "./build/sparf_attention.so")

    with tempfile.TemporaryDirectory(prefix="cemu-sparf-nvm-", dir=args.nvm_dir) as nvm, \
         tempfile.TemporaryDirectory(prefix="cemu-sparf-fdm-", dir=args.fdm_dir) as fdm:
        paths = (Path(nvm) / "k_token", Path(nvm) / "k_channel", Path(nvm) / "v_token")
        with SparfKvCacheStore(layout, *paths, replace_existing=True) as store:
            store.write_tokens(1, 0, keys, values)
            store.flush()
            with CemuSparfDevice(
                layout, *paths, fdm_directory=fdm, program_path=program,
                query_heads=4, top_r=args.top_r,
                compression_ratio=args.compression_ratio,
                control_path=args.control, namespace_path=args.namespace,
                cuda_target=args.cuda, approximate_runtime_ns=200000,
                exact_qk_runtime_ns=100000, pv_runtime_ns=50000,
            ) as device:
                actual = device.run_decode(
                    query, 1, args.tokens, store.value_mean(1), keys[-1], values[-1]
                )
                tolerance = 2e-3 if dtype == np.float16 else 2e-5
                np.testing.assert_allclose(actual, expected, rtol=tolerance, atol=tolerance)
                trace = device.last_trace
                expected_runtime = (
                    trace.channel_read_model_ns + trace.approximate_model_ns +
                    trace.token_k_read_model_ns +
                    max(trace.exact_qk_model_ns, trace.token_v_read_model_ns) +
                    trace.pv_model_ns
                )
                if trace.total_model_ns != expected_runtime:
                    raise AssertionError("SparF critical-path model is inconsistent")
                print(
                    f"[sparf-workflow] reads channel={trace.channel_read_model_ns}ns, "
                    f"K={trace.token_k_read_model_ns}ns, V={trace.token_v_read_model_ns}ns; "
                    f"compute approximate={trace.approximate_model_ns}ns, "
                    f"QK={trace.exact_qk_model_ns}ns, PV={trace.pv_model_ns}ns, "
                    f"total={trace.total_model_ns}ns",
                    flush=True,
                )
    print("[sparf-workflow] PASS: two K layouts, dynamic Top-r/Top-k, one Guest execute")


if __name__ == "__main__":
    main()
