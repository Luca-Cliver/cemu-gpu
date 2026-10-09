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
from cemu_flexgen.cemu_sparf_slot_scheduler import CemuSparfSlotScheduler
from flexgen_adapter.sparf_attention import SparfAttentionBackend
from cemu_flexgen.sparf_device_writeback import SparfDeviceWriteback
from phase_profiler import PhaseProfiler
from device_test_support import require_fdmfs


def run_case(nvm, fdm, args, ratio, profiler=None):
    dtype = np.dtype(args.dtype)
    capacity = 6 if args.device_writeback else 4
    layout = SparfKvCacheLayout(KvLayoutConfig(2, 13, 1, 2, 16, dtype))
    rng = np.random.default_rng(915)
    paths = [tuple(nvm / f"mb{microbatch}_{name}" for name in ("k", "c", "v"))
             for microbatch in range(2)]
    devices = [CemuSparfDevice(layout, *paths[0], fdm_directory=fdm,
        program_path=args.program, query_heads=4, top_r=4, compression_ratio=ratio,
        cuda_target=args.cuda, staged=not args.autonomous, autonomous=args.autonomous,
        buffer_tokens=capacity, name_prefix=f"slot{slot}", profiler=profiler)
        for slot in range(2)]
    scheduler = CemuSparfSlotScheduler(devices)
    backends = [SparfAttentionBackend(layout, *paths[microbatch], attention_device=scheduler,
        buffer_tokens=capacity, buffer_path=fdm / f"buffer{microbatch}",
        device_writeback=SparfDeviceWriteback(args.program, metadata_format=args.writeback_format)
        if args.device_writeback else None, profiler=profiler)
        for microbatch in range(2)]
    histories = {}
    tolerance = 3e-3 if dtype == np.float16 else 2e-5
    with ExitStack() as stack:
        for backend in backends:
            stack.enter_context(backend)
        for microbatch, backend in enumerate(backends):
            for layer in range(2):
                keys = rng.normal(size=(5, 1, 2, 16)).astype(dtype)
                values = rng.normal(size=keys.shape).astype(dtype)
                backend.write_prefill(layer, keys, values)
                histories[microbatch, layer] = (keys, values)
            backend.flush()
        for token in range(5, 13):
            for layer in range(2):
                requests, expected = [], []
                for microbatch, backend in enumerate(backends):
                    key = rng.normal(size=(1, 1, 2, 16)).astype(dtype)
                    value = rng.normal(size=key.shape).astype(dtype)
                    query = rng.normal(size=(1, 4, 16)).astype(dtype)
                    keys, values = histories[microbatch, layer]
                    keys, values = np.concatenate((keys, key)), np.concatenate((values, value))
                    histories[microbatch, layer] = keys, values
                    backend.append_decode(layer, token, key, value)
                    requests.append(backend.submit_decode(layer, query, token + 1))
                    expected.append(sparf_attention(query, keys, values, 4,
                        math.ceil((token + 1) / ratio)).output)
                for request, reference in zip(requests, expected):
                    np.testing.assert_allclose(request.result(), reference,
                                               rtol=tolerance, atol=tolerance)
        for microbatch, backend in enumerate(backends):
            backend.flush()
            for layer in range(2):
                keys, values = backend.store.read_tokens(layer, 0, 13)
                expected_keys, expected_values = histories[microbatch, layer]
                np.testing.assert_array_equal(keys, expected_keys)
                np.testing.assert_array_equal(values, expected_values)
                for head in range(2):
                    for dimension in range(16):
                        raw = backend.store.store._read_cache(backend.store.store._channel_k_fd,
                            13 * dtype.itemsize, layout.channel_offset(layer, 0, head, dimension))
                        np.testing.assert_array_equal(np.frombuffer(raw, dtype=dtype),
                                                      keys[:, 0, head, dimension])
            if args.device_writeback:
                status = backend.store.device_writeback.status()
                read_bytes, write_bytes, read_start, read_end, write_start, write_end, ready, copied = map(int, status)
                assert read_bytes == write_bytes and copied > 0
                assert read_start <= read_end <= write_start <= write_end <= ready
                print(f"[sparf-buffer] device-writeback mb={microbatch}: read={read_bytes}, write={write_bytes}, "
                      f"payload={copied}, FTL read/write deadlines OK", flush=True)
    print(f"[sparf-buffer] PASS ratio={ratio}, autonomous={args.autonomous}, "
          f"device_writeback={args.device_writeback}, writeback_format={args.writeback_format}: "
          "2 microbatches, 2 layers, 8 Decode steps, "
          "bounded FDM, NumPy Attention, final token/channel KV", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nvm-dir", type=Path, default=Path("/mnt/nvme0"))
    parser.add_argument("--fdm-dir", type=Path, default=Path("/mnt/fdm0"))
    parser.add_argument("--program")
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--device-writeback", action="store_true")
    parser.add_argument("--autonomous", action="store_true")
    parser.add_argument("--writeback-format", choices=("spans", "compact"), default="compact")
    parser.add_argument("--profile-output", type=Path)
    parser.add_argument("--dtype", choices=("float16", "float32"), default="float16")
    args = parser.parse_args()
    args.program = args.program or ("./build/sparf_attention_devptr.so" if args.cuda
                                    else "./build/sparf_attention.so")
    require_fdmfs(args.fdm_dir)
    profiler = PhaseProfiler() if args.profile_output else None
    for ratio in (1, 8):
        with tempfile.TemporaryDirectory(dir=args.nvm_dir, prefix="sparf-buffer-nvm-") as nvm, \
             tempfile.TemporaryDirectory(dir=args.fdm_dir, prefix="sparf-buffer-fdm-") as fdm:
            run_case(Path(nvm), Path(fdm), args, ratio, profiler)
    if profiler is not None:
        for line in profiler.summary():
            print(f"[sparf-buffer-profile] {line}")
        print(f"[sparf-buffer] CSV={profiler.write_csv(args.profile_output)}")


if __name__ == "__main__":
    main()
