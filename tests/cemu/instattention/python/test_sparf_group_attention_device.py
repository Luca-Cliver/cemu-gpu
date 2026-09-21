import argparse
from contextlib import ExitStack
import math
from pathlib import Path
import sys
import tempfile

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "build-guest"))

from cemu_flexgen import CemuSparfDevice, KvLayoutConfig, SparfKvCacheLayout, sparf_attention
from cemu_flexgen.sparf_group_device import SparfGroupedDevice
from cemu_flexgen.sparf_group_layout import SparfGroupLayout
from test_cemu_attention_phases import require_fdmfs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nvm-dir", type=Path, default=Path("/mnt/nvme0"))
    parser.add_argument("--fdm-dir", type=Path, default=Path("/mnt/fdm0"))
    parser.add_argument("--nvm-device", default="/dev/nvme0n1")
    parser.add_argument("--program")
    parser.add_argument("--cuda", action="store_true")
    args = parser.parse_args()
    require_fdmfs(args.fdm_dir)
    program = args.program or ("./build/sparf_attention_devptr.so" if args.cuda else "./build/sparf_attention.so")
    config = KvLayoutConfig(2, 531, 2, 2, 128, np.float16)
    layout = SparfKvCacheLayout(config)
    geometry = SparfGroupLayout(128, 8)
    random = np.random.default_rng(917)
    keys = random.normal(size=(2, 531, 2, 2, 128)).astype(np.float16)
    values = random.normal(size=keys.shape).astype(np.float16)
    queries = (random.normal(size=(2, 2, 4, 128)) / math.sqrt(128)).astype(np.float16)
    with ExitStack() as stack:
        nvm = Path(stack.enter_context(tempfile.TemporaryDirectory(prefix="group-attention-pool-", dir=args.nvm_dir)))
        fdm = Path(stack.enter_context(tempfile.TemporaryDirectory(prefix="group-attention-fdm-", dir=args.fdm_dir)))
        store = stack.enter_context(SparfGroupedDevice(
            geometry, nvm / "pool", 8 * 1024 * 1024, fdm, program,
            nvm_path=args.nvm_device, buffer_streams=8,
        ))
        attention = stack.enter_context(CemuSparfDevice(
            layout, nvm / "unused-token-k", nvm / "unused-channel-k", nvm / "unused-token-v",
            fdm, program, query_heads=4, top_r=16, compression_ratio=1,
            cuda_target=args.cuda, autonomous=True, grouped_store=store,
            approximate_runtime_ns=200000, exact_qk_runtime_ns=300000, pv_runtime_ns=100000,
            name_prefix="group_attention",
        ))
        cases = 0
        maximum_error = 0.0
        start = 0
        for end in (15, 16, 253, 256, 259, 531):
            for layer in range(2):
                for batch in range(2):
                    for head in range(2):
                        stream = (layer * 2 + batch) * 2 + head
                        store.append(stream, start, keys[layer, start:end, batch, head], values[layer, start:end, batch, head])
                for ratio in (1, 8):
                    attention.compression_ratio = ratio
                    mean = values[layer, :end].astype(np.float64).mean(axis=0).astype(np.float16)
                    output = attention.run_decode(queries[layer], layer, end, mean)
                    expected = sparf_attention(queries[layer], keys[layer, :end], values[layer, :end],
                                               16, math.ceil(end / ratio)).output
                    np.testing.assert_allclose(output, expected, atol=2e-3, rtol=2e-3)
                    error = float(np.max(np.abs(output.astype(np.float32) - expected)))
                    maximum_error = max(maximum_error, error)
                    trace = attention.last_trace
                    if end == 15:
                        assert trace.channel_read_model_ns == trace.token_k_read_model_ns == trace.token_v_read_model_ns == 0
                    assert trace.total_model_ns == (trace.channel_read_model_ns + trace.approximate_model_ns +
                        trace.token_k_read_model_ns + max(trace.exact_qk_model_ns, trace.token_v_read_model_ns) + trace.pv_model_ns)
                    cases += 1
                    print(f"[sparf-group-attention] layer={layer}, tokens={end}, ratio={ratio}, OK max={error:.3e}, "
                          f"QK_target={trace.exact_qk_model_ns} ns, V_read_target={trace.token_v_read_model_ns} ns", flush=True)
            start = end
        store.drain()
        for stream in range(8):
            progress = store.progress(stream)
            assert [progress[name] for name in ("accepted", "token_k", "channel_k", "token_v")] == [531] * 4
            assert progress["sealed"] and not progress["failed"]
        print(f"[sparf-group-attention] PASS: {cases} numerical cases, 2 layers, batch=2, Qheads=4/KVheads=2, "
              f"ratios=1/8, max={maximum_error:.3e}; grouped Flash+FDM reads -> autonomous SparF -> output; "
              "background drain complete. No OPT or timing calibration; use device trace to verify observed overlap.", flush=True)


if __name__ == "__main__":
    main()
