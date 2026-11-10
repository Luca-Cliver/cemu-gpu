"""Compare old and current CPU KV packing without opening a CEMU device."""

import argparse
import json
import platform
import statistics
import time

import numpy as np

from cemu_flexgen.fdm_kvbuffer_store import FDM_KVbuffer


def legacy_pack(keys, values):
    return b"".join(array.transpose(1, 0, 2).tobytes() for array in (keys, values))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--prompt", type=int, default=1024)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--heads", type=int, default=40)
    parser.add_argument("--head-dim", type=int, default=128)
    args = parser.parse_args()
    if min(vars(args).values()) <= 0:
        parser.error("All sizes and repeats must be positive")
    store = FDM_KVbuffer.__new__(FDM_KVbuffer)
    methods = {
        "legacy": legacy_pack,
        "contiguous_buffers_join": lambda keys, values: store._pack_payload(keys, values, "prefill", None),
    }
    print(json.dumps({"python": platform.python_version(), "numpy": np.__version__,
                      "source_memory": "ordinary CPU ndarray; no GPU/D2H/device IO",
                      "settings": vars(args)}), flush=True)
    for tokens in dict.fromkeys((args.prompt, 1)):
        for batch in dict.fromkeys((1, args.batch)):
            shape = (tokens, batch * args.heads, args.head_dim)
            keys = np.arange(tokens * batch * args.heads * args.head_dim,
                             dtype=np.uint16).view(np.float16).reshape(shape)
            values = keys.copy()
            expected = legacy_pack(keys, values)
            for method in methods.values():
                if method(keys, values) != expected:
                    raise AssertionError("KV packet bytes changed")
            del expected
            samples = {name: [] for name in methods}
            for iteration in range(args.repeats):
                order = list(methods) if iteration % 2 == 0 else list(reversed(methods))
                for name in order:
                    started = time.perf_counter_ns()
                    payload = methods[name](keys, values)
                    elapsed = time.perf_counter_ns() - started
                    samples[name].append(elapsed / 1e6)
                    del payload
            medians = {name: statistics.median(times) for name, times in samples.items()}
            print(json.dumps({"shape": shape, "payload_bytes": keys.nbytes + values.nbytes,
                              "byte_equal": True, "median_ms": medians,
                              "speedup": medians["legacy"] / medians["contiguous_buffers_join"]}), flush=True)


if __name__ == "__main__":
    main()
