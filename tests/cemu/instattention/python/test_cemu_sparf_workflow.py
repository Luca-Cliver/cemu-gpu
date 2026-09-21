#!/usr/bin/env python3

import argparse
import math
import sys
import tempfile
from contextlib import ExitStack
from pathlib import Path
from time import perf_counter

import numpy as np

PROJECT_DIRECTORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIRECTORY / "build-guest"))
sys.path.insert(0, str(PROJECT_DIRECTORY / "python"))

from cemu_flexgen import CemuSparfDevice, KvLayoutConfig, SparfKvCacheLayout, SparfKvCacheStore, sparf_attention
from test_cemu_attention_phases import require_fdmfs
from cemu_flexgen.cemu_sparf_slot_scheduler import CemuSparfSlotScheduler


def compare_staged_devices(layout, paths, fdm, program, args, query, keys, values, mean):
    """Interleave two isolated continuations against an independent CPU reference."""
    top_k = max(1, math.ceil(args.tokens / args.compression_ratio))
    queries = (query, -query)
    with ExitStack() as stack:
        devices = [CemuSparfDevice(
            layout, *paths, fdm_directory=fdm, program_path=program,
            query_heads=4, top_r=args.top_r, compression_ratio=args.compression_ratio,
            control_path=args.control, namespace_path=args.namespace,
            cuda_target=args.cuda, staged=True, name_prefix=f"staged_slot{slot}",
            approximate_runtime_ns=200000, exact_qk_runtime_ns=100000, pv_runtime_ns=50000,
        ) for slot in range(2)]
        scheduler = stack.enter_context(CemuSparfSlotScheduler(devices))
        requests = [device.prepare_decode(q, 1, args.tokens, mean, keys[-1], values[-1])
                    for device, q in zip(devices, queries)]
        for _ in range(4):
            for device, request in zip(devices, requests):
                device.execute_stage(request)
        for device, request, q in zip(devices, requests, queries):
            output = device.collect_decode(request)
            expected = sparf_attention(q, keys, values, min(args.top_r, 16), top_k).output
            tolerance = 2e-3 if q.dtype == np.float16 else 2e-5
            np.testing.assert_allclose(output, expected, rtol=tolerance, atol=tolerance)
            if device.decode_count != 1:
                raise AssertionError("stages must count as one attention workflow")
        pending = [scheduler.submit_decode(q, 1, args.tokens, mean, keys[-1], values[-1])
                   for q in queries]
        for future, q in zip(pending, queries):
            expected = sparf_attention(q, keys, values, min(args.top_r, 16), top_k).output
            np.testing.assert_allclose(future.result(), expected, rtol=tolerance, atol=tolerance)
    print("[sparf-stages] PASS: two isolated slots, staged scheduler, NumPy reference",
          flush=True)


def compare_decode_sequence(layout, paths, fdm, program, args, store, keys, values):
    """Publish appended KV, then compare both CSD paths on identical history.

    Ratio 1 also forces every historical V to be consumed, so the test cannot
    pass simply because Top-k happened to omit all previously appended tokens.
    """
    random = np.random.default_rng(914)
    tolerance = 2e-3 if keys.dtype == np.float16 else 2e-5
    for ratio in dict.fromkeys((args.compression_ratio, 1)):
        history_k, history_v = keys.copy(), values.copy()
        store.write_tokens(1, 0, history_k, history_v)
        store.publish()
        with ExitStack() as stack:
            devices = [stack.enter_context(CemuSparfDevice(
                layout, *paths, fdm_directory=fdm, program_path=program,
                query_heads=4, top_r=args.top_r, compression_ratio=ratio,
                control_path=args.control, namespace_path=args.namespace,
                cuda_target=args.cuda, staged=index in (1, 2), autonomous=index == 3,
                name_prefix=f"sequence_r{ratio}_slot{index}",
                approximate_runtime_ns=200000, exact_qk_runtime_ns=100000,
                pv_runtime_ns=50000,
            )) for index in range(4 if args.autonomous else 3)]
            for step in range(args.decode_steps):
                token = history_k.shape[0]
                key = random.normal(size=keys.shape[1:]).astype(keys.dtype)
                value = (random.normal(size=key.shape) + step + 1).astype(keys.dtype)
                history_k = np.concatenate((history_k, key[None]))
                history_v = np.concatenate((history_v, value[None]))
                store.write_token(1, token, key, value)
                started = perf_counter()
                store.publish()
                publish_seconds = perf_counter() - started
                # Distinct magnitudes avoid tie-dependent channel ordering.
                query = np.broadcast_to(np.roll(np.arange(1, 17), step), (1, 4, 16))
                query = (query / 64).astype(keys.dtype)
                count = token + 1
                expected = sparf_attention(query, history_k, history_v,
                    min(args.top_r, 16), max(1, math.ceil(count / ratio)))
                outputs = []
                # Rotate staged slots and alternate which path runs first.
                pair = (devices[0], devices[1 + step % 2])
                if args.autonomous:
                    pair += (devices[3],)
                for device in pair[::(-1 if step % 2 else 1)]:
                    output = device.run_decode(query, 1, count, store.value_mean(1), key, value)
                    np.testing.assert_allclose(output, expected.output,
                                               rtol=tolerance, atol=tolerance)
                    outputs.append(output)
                    # This buffer holds raw channel-K from NVM before the
                    # current-token overlay, so it tests device visibility.
                    channels = device._device.read_tensor(
                        1, (4, min(args.top_r, 16), count), keys.dtype)
                    for head in range(4):
                        want = history_k[:, 0, head // 2, :][:, expected.channel_indices[head]].T
                        np.testing.assert_array_equal(channels[head], want)
                np.testing.assert_allclose(outputs[0], outputs[1], rtol=tolerance, atol=tolerance)
                print(f"[sparf-sequence] step={step + 1}/{args.decode_steps}, "
                      f"ratio={ratio}, tokens={count}, publish_wall={publish_seconds:.6f}s: "
                      f"raw channel-K and {len(pair)} Attention outputs OK", flush=True)
    print("[sparf-sequence] PASS: appended KV visible; legacy/staged match NumPy over multiple steps",
          flush=True)


def main():
    parser = argparse.ArgumentParser(description="Validate one-command device-side InstAttention SparF")
    parser.add_argument("--nvm-dir", default="/mnt/nvme0")
    parser.add_argument("--fdm-dir", default="/mnt/fdm0")
    parser.add_argument("--control", default="/dev/nvme0c3")
    parser.add_argument("--namespace", default="/dev/ng0n3")
    parser.add_argument("--program")
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--compare-staged", action="store_true")
    parser.add_argument("--autonomous", action="store_true")
    parser.add_argument("--dtype", choices=("float16", "float32"), default="float16")
    parser.add_argument("--tokens", type=int, default=129)
    parser.add_argument("--decode-steps", type=int, default=4)
    parser.add_argument("--top-r", type=int, default=4)
    parser.add_argument("--compression-ratio", type=int, default=8)
    args = parser.parse_args()
    print(f"[sparf-workflow] mode={'device-autonomous-v3' if args.autonomous else 'legacy-v1'}", flush=True)
    if args.tokens < 2 or args.decode_steps < 2:
        parser.error("--tokens and --decode-steps must be at least 2")
    require_fdmfs(args.fdm_dir)
    dtype = np.dtype(args.dtype)
    layout = SparfKvCacheLayout(KvLayoutConfig(2, args.tokens + args.decode_steps, 1, 2, 16, dtype))
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
                autonomous=args.autonomous,
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
            if args.compare_staged:
                compare_staged_devices(layout, paths, fdm, program, args,
                                       query, keys, values, store.value_mean(1))
                compare_decode_sequence(layout, paths, fdm, program, args, store, keys, values)
    print("[sparf-workflow] PASS: two K layouts, dynamic Top-r/Top-k, one Guest execute")


if __name__ == "__main__":
    main()
