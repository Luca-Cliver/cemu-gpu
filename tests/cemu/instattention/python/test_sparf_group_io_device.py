import argparse
from contextlib import ExitStack
from pathlib import Path
import sys
import tempfile

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "build-guest"))

from cemu_flexgen.sparf_group_layout import SparfGroupLayout
from cemu_flexgen.sparf_group_device import SparfGroupedDevice, TOKEN_K, CHANNEL_K, TOKEN_V
from test_cemu_attention_phases import require_fdmfs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nvm-dir", type=Path, default=Path("/mnt/nvme0"))
    parser.add_argument("--fdm-dir", type=Path, default=Path("/mnt/fdm0"))
    parser.add_argument("--nvm-device", default="/dev/nvme0n1")
    parser.add_argument("--program", default="./build/sparf_attention.so")
    args = parser.parse_args()
    require_fdmfs(args.fdm_dir)
    layout = SparfGroupLayout(128, 8)
    random = np.random.default_rng(915)
    write_bytes = read_bytes = writes = 0
    with ExitStack() as stack:
        nvm = Path(stack.enter_context(tempfile.TemporaryDirectory(prefix="group-pool-", dir=args.nvm_dir)))
        fdm = Path(stack.enter_context(tempfile.TemporaryDirectory(prefix="group-fdm-", dir=args.fdm_dir)))
        device = stack.enter_context(SparfGroupedDevice(
            layout, nvm / "pool", 2 * 1024 * 1024, fdm, args.program, nvm_path=args.nvm_device,
        ))
        for stream in (0, 5):
            keys = random.normal(size=(512, 128)).astype(np.float16)
            values = random.normal(size=keys.shape).astype(np.float16)
            for start in range(0, len(keys), layout.token_group_tokens):
                for kind, source in ((TOKEN_K, keys), (TOKEN_V, values)):
                    result = device.put(kind, stream, start, source[start:start + layout.token_group_tokens])
                    assert int(result[0]) == 0 and int(result[1]) == layout.page_bytes
                    assert int(result[4]) <= int(result[5]) <= int(result[6])
                    write_bytes += int(result[1])
                    writes += 1
            for start in range(0, len(keys), layout.channel_group_tokens):
                result = device.put(CHANNEL_K, stream, start, keys[start:start + layout.channel_group_tokens])
                assert int(result[0]) == 0 and int(result[1]) == layout.channel_group_pages * layout.page_bytes
                assert int(result[4]) <= int(result[5]) <= int(result[6])
                write_bytes += int(result[1])
                writes += 1
            for start in (0, 256):
                for kind, expected in ((TOKEN_K, keys), (TOKEN_V, values)):
                    actual = device.read_tokens(kind, stream, range(start, start + 256))
                    np.testing.assert_array_equal(actual, expected[start:start + 256])
                    assert int(device.last_result[0]) == 16 * layout.page_bytes
                    assert int(device.last_result[1]) == 0
                    read_bytes += int(device.last_result[0])
            tokens = (31, 1, 16, 15, 1, 511)
            np.testing.assert_array_equal(device.read_tokens(TOKEN_K, stream, tokens), keys[list(tokens)])
            assert int(device.last_result[0]) == 3 * layout.page_bytes
            read_bytes += int(device.last_result[0])
            for first_channel in range(0, 128, 8):
                channels = tuple(range(first_channel, first_channel + 8))
                np.testing.assert_array_equal(device.read_channels(stream, 0, 512, channels), keys[:, list(channels)].T)
                assert int(device.last_result[0]) == 2 * layout.page_bytes
                read_bytes += int(device.last_result[0])
            channels = (127, 0, 7, 0)
            np.testing.assert_array_equal(device.read_channels(stream, 253, 8, channels), keys[253:261, list(channels)].T)
            assert int(device.last_result[0]) == 4 * layout.page_bytes
            assert int(device.last_result[2]) <= int(device.last_result[3]) <= int(device.last_result[6])
            read_bytes += int(device.last_result[0])
            print(f"[sparf-group] stream={stream} token-K/V and all channel-K verified; sparse page dedup/order OK", flush=True)
        assert int(device.last_result[9]) == 192
        assert int(device.last_result[10]) == 132
        print(f"[sparf-group] PASS: device directory, full-group packing, NVM/FTL writes={writes}, "
              f"write_bytes={write_bytes}, read_bytes={read_bytes}, PUT old-page reads=0; "
              "no OPT, automatic tail buffering, or Attention kernel is exercised", flush=True)


if __name__ == "__main__":
    main()
