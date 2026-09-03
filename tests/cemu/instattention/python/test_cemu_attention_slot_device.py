#!/usr/bin/env python3

import argparse
import math
import sys
import tempfile
import threading
import time
from pathlib import Path

import numpy as np


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "build-guest"))
sys.path.insert(0, str(PROJECT_DIR / "python"))

CUDA_DEFAULT_SHARED_MEMORY_BYTES = 48 * 1024

from cemu_flexgen import (
    AttentionBufferConfig,
    CemuAttentionDevice,
    CemuAttentionSlotScheduler,
    KvCacheLayout,
    KvCacheStore,
    KvLayoutConfig,
    align_up,
)


class TimelineLogger:
    def __init__(self):
        self._start_ns = time.perf_counter_ns()
        self._lock = threading.Lock()
        self.events = []

    def __call__(self, message):
        elapsed_us = (time.perf_counter_ns() - self._start_ns) / 1000.0
        with self._lock:
            self.events.append((elapsed_us, message))
            print(f"[slot-device][{elapsed_us:12.3f} us] {message}", flush=True)

    def event_index(self, text):
        for index, (_, message) in enumerate(self.events):
            if text in message:
                return index
        return None

    def event_time(self, text):
        for elapsed_us, message in self.events:
            if text in message:
                return elapsed_us
        return None


def positive_integer(value):
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def parse_args():
    parser = argparse.ArgumentParser(
        description="Test two real CEMU Attention MRS slots with shared NVM KV"
    )
    parser.add_argument("--control", default="/dev/nvme0c3")
    parser.add_argument("--namespace", default="/dev/ng0n3")
    parser.add_argument("--nvm-dir", default="/mnt/nvme0")
    parser.add_argument("--fdm-dir", default="/mnt/fdm0")
    parser.add_argument("--program")
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--token-count", type=positive_integer, default=17)
    parser.add_argument(
        "--cuda-shared-memory-bytes",
        type=positive_integer,
        default=CUDA_DEFAULT_SHARED_MEMORY_BYTES,
    )
    parser.add_argument("--require-overlap", action="store_true")
    return parser.parse_args()


def attention_reference(query, keys, values, num_kv_heads, scale):
    batch_size, num_query_heads, _ = query.shape
    token_count = keys.shape[0]
    output = np.empty_like(query)

    for batch in range(batch_size):
        for query_head in range(num_query_heads):
            kv_head = query_head * num_kv_heads // num_query_heads
            scores = np.empty(token_count, dtype=np.float32)
            for token in range(token_count):
                scores[token] = (
                    np.dot(query[batch, query_head], keys[token, batch, kv_head])
                    * scale
                )
            probabilities = np.exp(scores - np.max(scores))
            probabilities /= np.sum(probabilities)
            output[batch, query_head] = np.sum(
                probabilities[:, np.newaxis] * values[:, batch, kv_head],
                axis=0,
            )
    return output


def validate_directory(name, path):
    directory = Path(path)
    if not directory.is_dir():
        raise FileNotFoundError(f"{name} does not exist: {directory}")
    return directory


def make_slot(
    slot_index,
    layout,
    fdm_path,
    staging_bytes,
    query_bytes,
    k_cache_path,
    v_cache_path,
    program_path,
    args,
    timeline,
):
    buffers = AttentionBufferConfig(
        query_bytes=query_bytes,
        staging_bytes=staging_bytes,
        state_bytes=512,
        output_bytes=query_bytes,
        query_path=fdm_path / f"slot_{slot_index}_query",
        k_staging_path=fdm_path / f"slot_{slot_index}_k_staging",
        v_staging_path=fdm_path / f"slot_{slot_index}_v_staging",
        state_path=fdm_path / f"slot_{slot_index}_state",
        output_path=fdm_path / f"slot_{slot_index}_output",
    )
    mode = "cuda" if args.cuda else "cpu"

    return CemuAttentionDevice(
        layout=layout,
        buffers=buffers,
        program_name=f"dense_attention_{mode}_slot_{slot_index}_test",
        program_path=program_path,
        function_name="dense_attention",
        k_cache_path=k_cache_path,
        v_cache_path=v_cache_path,
        control_path=args.control,
        namespace_path=args.namespace,
        cuda_target=args.cuda,
        replace_program=True,
        replace_staging_files=True,
        logger=lambda message: timeline(f"slot={slot_index} {message}"),
    )


def main():
    args = parse_args()
    nvm_root = validate_directory("NVM directory", args.nvm_dir)
    fdm_root = validate_directory("FDM directory", args.fdm_dir)
    program_path = args.program or (
        "./build/dense_attention_devptr.so"
        if args.cuda
        else "./build/dense_attention.so"
    )

    num_layers = 2
    batch_size = 1
    num_query_heads = 4
    num_kv_heads = 2
    head_dim = 8
    token_count = args.token_count
    required_shared_bytes = token_count * np.dtype(np.float32).itemsize
    if args.cuda and required_shared_bytes > args.cuda_shared_memory_bytes:
        max_tokens = args.cuda_shared_memory_bytes // np.dtype(np.float32).itemsize
        raise ValueError(
            f"CUDA Attention requires {required_shared_bytes} shared-memory bytes "
            f"for {token_count} tokens, but the configured per-block limit is "
            f"{args.cuda_shared_memory_bytes}; use --token-count <= {max_tokens}"
        )
    rng = np.random.default_rng(20260902)
    queries = [
        rng.normal(size=(batch_size, num_query_heads, head_dim)).astype(np.float32)
        for _ in range(num_layers)
    ]
    keys = [
        rng.normal(
            size=(token_count, batch_size, num_kv_heads, head_dim)
        ).astype(np.float32)
        for _ in range(num_layers)
    ]
    values = [
        rng.normal(
            size=(token_count, batch_size, num_kv_heads, head_dim)
        ).astype(np.float32)
        for _ in range(num_layers)
    ]

    layout = KvCacheLayout(
        KvLayoutConfig(
            num_layers=num_layers,
            max_seq_len=token_count,
            batch_size=batch_size,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            dtype=np.float32,
        )
    )
    staging_bytes = token_count * layout.token_stride
    query_bytes = align_up(queries[0].nbytes, 512)
    scale = np.float32(1.0 / math.sqrt(head_dim))
    expected = [
        attention_reference(
            queries[layer],
            keys[layer],
            values[layer],
            num_kv_heads,
            scale,
        )
        for layer in range(num_layers)
    ]

    with tempfile.TemporaryDirectory(
        prefix="cemu-slot-nvm-", dir=nvm_root
    ) as nvm_directory, tempfile.TemporaryDirectory(
        prefix="cemu-slot-fdm-", dir=fdm_root
    ) as fdm_directory:
        nvm_path = Path(nvm_directory)
        fdm_path = Path(fdm_directory)
        k_cache_path = nvm_path / "k_cache"
        v_cache_path = nvm_path / "v_cache"

        with KvCacheStore(
            layout,
            k_cache_path,
            v_cache_path,
            replace_existing=True,
        ) as store:
            for layer in range(num_layers):
                store.write_tokens(layer, 0, keys[layer], values[layer])
            store.flush()

        timeline = TimelineLogger()
        slots = tuple(
            make_slot(
                slot_index,
                layout,
                fdm_path,
                staging_bytes,
                query_bytes,
                k_cache_path,
                v_cache_path,
                program_path,
                args,
                timeline,
            )
            for slot_index in range(2)
        )
        scheduler = CemuAttentionSlotScheduler(slots, logger=timeline)

        with scheduler:
            program_ids = tuple(slot.program_id for slot in slots)
            mrs_ids = tuple(slot.memory_range_set_id for slot in slots)
            range_paths = tuple(
                tuple(spec.path for spec in slot.ranges) for slot in slots
            )

            requests = tuple(
                scheduler.submit_decode(
                    queries[layer],
                    layer=layer,
                    valid_tokens=token_count,
                )
                for layer in range(num_layers)
            )
            outputs = scheduler.wait_all()

            for slot_index, slot in enumerate(slots):
                assert slot.memory_range_count == 5
                staged_keys, staged_values = slot.staging.read_staged_tokens(token_count)
                np.testing.assert_array_equal(staged_keys, keys[slot_index])
                np.testing.assert_array_equal(staged_values, values[slot_index])

            assert all(request.done() for request in requests)
            assert program_ids[0] != program_ids[1]
            assert mrs_ids[0] != mrs_ids[1]
            assert not set(range_paths[0]).intersection(range_paths[1])

        for layer, output in enumerate(outputs):
            np.testing.assert_allclose(output, expected[layer], rtol=2e-5, atol=2e-5)

        second_load_start = timeline.event_time("load-start request=1")
        second_load_complete = timeline.event_time("load-complete request=1")
        first_compute_start = timeline.event_time("compute-start request=0")
        first_compute_complete = timeline.event_time("compute-complete request=0")
        event_times = (
            second_load_start,
            second_load_complete,
            first_compute_start,
            first_compute_complete,
        )
        if any(event_time is None for event_time in event_times):
            raise AssertionError("the real slot timeline is missing Load/Compute events")
        overlap_start = max(second_load_start, first_compute_start)
        overlap_end = min(second_load_complete, first_compute_complete)
        overlap_us = max(0.0, overlap_end - overlap_start)
        overlap_observed = overlap_us > 0.0
        mode = "CUDA device-pointer" if args.cuda else "CPU host"
        print(f"[slot-device] mode={mode}, programs={program_ids}, MRS={mrs_ids}")
        print(
            "[slot-device] shared NVM: "
            f"K={k_cache_path}, V={v_cache_path}, layers={num_layers}"
        )
        print(
            "[slot-device] independent FDM ranges: "
            f"slot0={range_paths[0]}, slot1={range_paths[1]}"
        )
        print(
            "[slot-device] one-chunk capacity: "
            f"tokens={token_count}, token_stride={layout.token_stride}, "
            f"staging_bytes={staging_bytes}"
        )
        for layer, output in enumerate(outputs):
            print(
                f"[slot-device] layer={layer} output[:4]="
                f"{output[0, 0, :4].tolist()}"
            )
        print(
            "[slot-device] intervals: "
            f"compute0=[{first_compute_start:.3f}, {first_compute_complete:.3f}] us, "
            f"load1=[{second_load_start:.3f}, {second_load_complete:.3f}] us"
        )
        print(
            "[slot-device] measured Load(request=1) / Compute(request=0) "
            f"overlap={overlap_us:.3f} us, observed={overlap_observed}"
        )
        if args.require_overlap and not overlap_observed:
            raise AssertionError(
                "real CEMU Load/Compute overlap was required but not observed; "
                "increase --token-count and retry"
            )
        print("Real two-slot CEMU Attention test passed")


if __name__ == "__main__":
    main()
