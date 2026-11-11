import argparse
import errno
import json
import os
from pathlib import Path
import tempfile

import numpy as np

from cemu_flexgen.fdm_kvbuffer_store import FDM_KVbuffer
from cemu_flexgen.sparf_group_layout import SparfGroupLayout


def must_reject(function, expected):
    try:
        function()
    except OSError as error:
        if error.errno != expected:
            raise
    else:
        raise AssertionError(f"operation must fail with errno={expected}")


def main():
    parser = argparse.ArgumentParser(description="Guest synchronous tagged-pwrite smoke; requires poll device and new kernel")
    parser.add_argument("--fdm-directory", default="/mnt/fdm0")
    parser.add_argument("--control", default="/dev/nvme0c3")
    parser.add_argument("--namespace", default="/dev/ng0n3")
    parser.add_argument("--aggregate-program", default="./build/fdm_kv_aggregate.so")
    args = parser.parse_args()
    directory = Path(tempfile.mkdtemp(prefix="fdm-two-slot-smoke-", dir=args.fdm_directory))
    print(f"[fdm-slots] files={directory}", flush=True)
    store = FDM_KVbuffer(SparfGroupLayout(128, 8, dtype=np.float16),
        batches=1, layers=1, heads=2, max_tokens=1600, microbatch_size=1,
        fdm_directory=directory, meta_bytes=1024**2, data_bytes=16 * 1024**2,
        input_bytes=2 * 1024**2, input_slots=2, control_path=args.control,
        namespace_path=args.namespace, aggregate_program=args.aggregate_program)
    try:
        store.open()
        for path in (store.meta_path, store.data_path):
            must_reject(lambda: os.unlink(path), errno.EBUSY)
            must_reject(lambda: os.truncate(path, 0), errno.EBUSY)
        descriptor = os.open(store.data_path, os.O_RDWR)
        try:
            must_reject(lambda: os.pwrite(descriptor, bytes(512), 512), errno.EINVAL)
        finally:
            os.close(descriptor)
        keys = (np.arange(1536 * 2 * 128).reshape(1536, 2, 128) % 97).astype(np.float16)
        store.append_batch(0, 0, keys, -keys)
        for token in range(1536, 1552):
            store.append_batch(0, token, keys[:1], -keys[:1])
        store.drain()
        stats = store.statistics()
        if stats["failed"] or not stats["sealed"] or stats["accepted_stream0"] != 1552:
            raise AssertionError(stats)
        if any(stats[name] != 1552 for name in ("token_k", "channel_k", "token_v")):
            raise AssertionError(stats)
        if stats["staged_pages"] or stats["inflight_blocks"]:
            raise AssertionError(stats)
        store.close()
        print("[fdm-slots] PASS: pinned files, tagged synchronous pwrite, reuse and drain; " + json.dumps(stats), flush=True)
    except BaseException:
        print(f"[fdm-slots] FAILED; retained files for diagnosis: {directory}", flush=True)
        raise
    for path in directory.iterdir():
        path.unlink()
    directory.rmdir()


if __name__ == "__main__":
    main()
