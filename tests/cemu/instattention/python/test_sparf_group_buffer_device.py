import argparse
from contextlib import ExitStack
from pathlib import Path
import sys
import tempfile
import time

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "build-guest"))

from cemu_flexgen.sparf_group_layout import SparfGroupLayout
from cemu_flexgen.sparf_group_device import SparfGroupedDevice, TOKEN_K, TOKEN_V
from test_cemu_attention_phases import require_fdmfs


def check(device, stream, keys, values, accepted):
    tokens = sorted(set((0, accepted - 1, max(0, accepted - 2), min(15, accepted - 1))))
    tokens += [tokens[0]]
    for kind, expected in ((TOKEN_K, keys), (TOKEN_V, values)):
        np.testing.assert_array_equal(device.read_tokens(kind, stream, tokens), expected[tokens])
    start = max(0, accepted - 19)
    channels = (127, 0, 7, 0)
    np.testing.assert_array_equal(device.read_channels(stream, start, accepted - start, channels),
                                  keys[start:accepted][:, channels].T)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nvm-dir", type=Path, default=Path("/mnt/nvme0"))
    parser.add_argument("--fdm-dir", type=Path, default=Path("/mnt/fdm0"))
    parser.add_argument("--nvm-device", default="/dev/nvme0n1")
    parser.add_argument("--program", default="./build/sparf_attention_devptr.so")
    parser.add_argument("--write-batch-pages", type=int, default=256)
    args = parser.parse_args()
    require_fdmfs(args.fdm_dir)
    layout = SparfGroupLayout(128, 8)
    random = np.random.default_rng(916)
    keys = random.normal(size=(2, 531, 128)).astype(np.float16)
    values = random.normal(size=keys.shape).astype(np.float16)
    with ExitStack() as stack:
        nvm = Path(stack.enter_context(tempfile.TemporaryDirectory(prefix="group-buffer-pool-", dir=args.nvm_dir)))
        fdm = Path(stack.enter_context(tempfile.TemporaryDirectory(prefix="group-buffer-fdm-", dir=args.fdm_dir)))
        device = stack.enter_context(SparfGroupedDevice(
            layout, nvm / "pool", 2 * 1024 * 1024, fdm, args.program,
            nvm_path=args.nvm_device, buffer_streams=2, write_batch_pages=args.write_batch_pages,
        ))
        device.append_batch(0, 0, keys[:, :15].transpose(1, 0, 2), values[:, :15].transpose(1, 0, 2))
        for stream in range(2):
            check(device, stream, keys[stream], values[stream], 15)
            assert int(device.last_result[0]) == 0
            progress = device.progress(stream)
            assert [progress[name] for name in ("accepted", "token_k", "channel_k", "token_v")] == [15, 0, 0, 0]
        print("[sparf-group-buffer] first 15 tokens: DRAM tail reads correct, zero NVM reads", flush=True)
        device.append_batch(0, 15, keys[:, 15:253].transpose(1, 0, 2), values[:, 15:253].transpose(1, 0, 2))
        deadline = time.monotonic() + 30
        while True:
            progress = device.progress(0)
            if progress["failed"]:
                raise RuntimeError(f"background writeback failed: {progress}")
            if progress["token_k"] == progress["token_v"] == 240:
                break
            if time.monotonic() > deadline:
                raise TimeoutError(f"full groups did not commit: {progress}")
            time.sleep(0.01)
        assert progress["accepted"] == 253 and progress["channel_k"] == 0
        print(f"[sparf-group-buffer] independent publication before Decode: {progress}", flush=True)
        for position in range(253, 267):
            device.append_batch(0, position, keys[:, position:position + 1].transpose(1, 0, 2),
                                values[:, position:position + 1].transpose(1, 0, 2))
            for stream in range(2):
                check(device, stream, keys[stream], values[stream], position + 1)
        device.append_batch(0, 267, keys[:, 267:].transpose(1, 0, 2), values[:, 267:].transpose(1, 0, 2))
        for stream in range(2):
            check(device, stream, keys[stream], values[stream], 531)
        device.drain()
        progress = device.progress(0)
        expected_bytes = 2 * (2 * 34 + 3 * 16) * layout.page_bytes
        assert progress["write_bytes"] == expected_bytes
        for stream in range(2):
            progress = device.progress(stream)
            assert [progress[name] for name in ("accepted", "token_k", "channel_k", "token_v")] == [531] * 4
            assert progress["sealed"] == 1 and progress["failed"] == 0
            for start in range(0, 531, 128):
                stop = min(start + 128, 531)
                for kind, expected in ((TOKEN_K, keys), (TOKEN_V, values)):
                    np.testing.assert_array_equal(device.read_tokens(kind, stream, range(start, stop)), expected[stream, start:stop])
            for first_channel in range(0, 128, 8):
                channels = list(range(first_channel, first_channel + 8))
                np.testing.assert_array_equal(device.read_channels(stream, 0, 531, channels), keys[stream][:, channels].T)
        device.drain()
        assert device.progress(0)["write_bytes"] == expected_bytes
        statistics = device.statistics()
        assert statistics["appended_tokens"] == 2 * 531
        assert statistics["append_commands"] == 47
        assert 0 < statistics["max_write_pages"] <= args.write_batch_pages
        assert statistics["write_batches"] > 0
        print(f"[sparf-group-buffer] statistics: {statistics}", flush=True)
        print(f"[sparf-group-buffer] PASS: batched APPEND, write_batch_pages={args.write_batch_pages}; "
              f"2 streams, 531 tokens/stream, 15/16 and 255/256 boundaries, "
              f"ring reuse, mixed reads, terminal partial groups; FDM tail+scratch={device.tail_bytes} bytes, "
              f"background full-page writes={expected_bytes} bytes; no OPT/Attention integration yet", flush=True)


if __name__ == "__main__":
    main()
