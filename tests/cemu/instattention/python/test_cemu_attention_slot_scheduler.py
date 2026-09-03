#!/usr/bin/env python3

import sys
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "python"))

from cemu_flexgen import (
    CemuAttentionSharedWorkers,
    CemuAttentionSlotScheduler,
    KvCacheLayout,
    KvLayoutConfig,
)


class PipelineControl:
    def __init__(self):
        self.first_compute_started = threading.Event()
        self.release_first_compute = threading.Event()
        self.second_load_complete = threading.Event()
        self.compute_slots = []


class FakeAttentionSlot:
    def __init__(self, slot_index, layout, staging_bytes, control):
        self.slot_index = slot_index
        self.layout = layout
        self.buffers = SimpleNamespace(staging_bytes=staging_bytes)
        self.control = control
        self.is_open = False
        self.staged_chunk = None
        self.query = None

    def open(self):
        self.is_open = True
        return self

    def close(self):
        self.is_open = False

    def stage_chunk(self, chunk):
        if not self.is_open:
            raise RuntimeError("slot is closed")
        self.staged_chunk = chunk
        if self.slot_index == 1:
            self.control.second_load_complete.set()
        return chunk.copy_size

    def execute_staged_chunk(self, query, chunk, metadata):
        if chunk != self.staged_chunk:
            raise RuntimeError("compute did not use the staged chunk")
        self.query = query.copy()
        self.control.compute_slots.append(self.slot_index)
        if self.slot_index == 0:
            self.control.first_compute_started.set()
            if not self.control.release_first_compute.wait(timeout=5):
                raise TimeoutError("test did not release the first compute")
        return 0

    def append_staged_token(self, key, value):
        chunk = self.staged_chunk
        self.staged_chunk = type(chunk)(
            layer=chunk.layer,
            start_token=chunk.start_token,
            token_count=chunk.token_count + 1,
            nvm_offset=chunk.nvm_offset,
            copy_size=chunk.copy_size + self.layout.token_stride,
        )
        return self.staged_chunk

    def collect_output(self, metadata):
        return np.full_like(self.query, self.slot_index + 1)


class CemuAttentionSlotSchedulerTest(unittest.TestCase):
    def setUp(self):
        self.layout = KvCacheLayout(
            KvLayoutConfig(
                num_layers=2,
                max_seq_len=4,
                batch_size=1,
                num_kv_heads=1,
                head_dim=2,
                dtype=np.float32,
            )
        )
        self.control = PipelineControl()
        self.logs = []

    def tearDown(self):
        self.control.release_first_compute.set()

    def test_second_slot_loads_while_first_slot_computes(self):
        slots = self._make_slots(staging_bytes=4 * self.layout.token_stride)
        query = np.arange(4, dtype=np.float32).reshape(1, 2, 2)

        with CemuAttentionSlotScheduler(slots, logger=self.logs.append) as scheduler:
            first = scheduler.submit_decode(query, layer=0, valid_tokens=1)
            self.assertTrue(self.control.first_compute_started.wait(timeout=5))
            second = scheduler.submit_decode(query + 1, layer=1, valid_tokens=1)

            self.assertTrue(self.control.second_load_complete.wait(timeout=5))
            self.assertEqual(self.control.compute_slots, [0])
            self.assertFalse(first.done())
            self.assertFalse(second.done())

            self.control.release_first_compute.set()
            np.testing.assert_array_equal(
                scheduler.wait_request(first, timeout=5),
                np.ones_like(query),
            )
            np.testing.assert_array_equal(
                scheduler.wait_request(second, timeout=5),
                np.full_like(query, 2),
            )
            self.assertEqual(scheduler.wait_all(), ())

        self.assertEqual(self.control.compute_slots, [0, 1])
        first_complete = next(
            index
            for index, message in enumerate(self.logs)
            if "compute-complete request=0" in message
        )
        second_load = next(
            index
            for index, message in enumerate(self.logs)
            if "load-complete request=1" in message
        )
        self.assertLess(second_load, first_complete)

    def test_pipeline_rejects_requests_requiring_multiple_chunks(self):
        slots = self._make_slots(staging_bytes=self.layout.token_stride)
        query = np.zeros((1, 2, 2), dtype=np.float32)

        scheduler = CemuAttentionSlotScheduler(slots).open()
        try:
            request = scheduler.submit_decode(query, layer=0, valid_tokens=2)
            with self.assertRaisesRegex(ValueError, "one KV chunk"):
                request.result(timeout=5)
        finally:
            scheduler.close()

    def test_prefetch_next_layer_overlaps_current_compute(self):
        slots = self._make_slots(staging_bytes=4 * self.layout.token_stride)
        query = np.arange(4, dtype=np.float32).reshape(1, 2, 2)
        token = np.ones((1, 1, 2), dtype=np.float32)

        with CemuAttentionSlotScheduler(slots, logger=self.logs.append) as scheduler:
            first_prefetch = scheduler.prefetch_decode(layer=0, history_tokens=1)
            first = scheduler.submit_prefetched_decode(
                first_prefetch,
                query,
                token,
                token,
            )
            self.assertTrue(self.control.first_compute_started.wait(timeout=5))

            second_prefetch = scheduler.prefetch_decode(layer=1, history_tokens=1)
            second_prefetch.result(timeout=5)
            self.assertTrue(self.control.second_load_complete.is_set())
            self.assertFalse(first.done())

            self.control.release_first_compute.set()
            np.testing.assert_array_equal(
                scheduler.wait_request(first, timeout=5),
                np.ones_like(query),
            )

        first_compute_complete = next(
            index
            for index, message in enumerate(self.logs)
            if "compute-complete request=0" in message
        )
        second_prefetch_complete = next(
            index
            for index, message in enumerate(self.logs)
            if "prefetch-complete request=1" in message
        )
        self.assertLess(second_prefetch_complete, first_compute_complete)

    def test_multiple_schedulers_share_one_compute_stream(self):
        first_control = PipelineControl()
        second_control = PipelineControl()
        workers = CemuAttentionSharedWorkers(logger=self.logs.append)
        first_slots = tuple(
            FakeAttentionSlot(index, self.layout, 4 * self.layout.token_stride, first_control)
            for index in range(2)
        )
        second_slots = tuple(
            FakeAttentionSlot(index, self.layout, 4 * self.layout.token_stride, second_control)
            for index in range(2)
        )
        query = np.arange(4, dtype=np.float32).reshape(1, 2, 2)

        with CemuAttentionSlotScheduler(
            first_slots,
            workers=workers,
        ) as first_scheduler, CemuAttentionSlotScheduler(
            second_slots,
            workers=workers,
        ) as second_scheduler:
            first = first_scheduler.submit_decode(query, layer=0, valid_tokens=1)
            self.assertTrue(first_control.first_compute_started.wait(timeout=5))
            second = second_scheduler.submit_decode(query, layer=1, valid_tokens=1)
            self.assertFalse(second_control.first_compute_started.wait(timeout=0.05))

            first_control.release_first_compute.set()
            first_scheduler.wait_request(first, timeout=5)
            self.assertTrue(second_control.first_compute_started.wait(timeout=5))
            second_control.release_first_compute.set()
            second_scheduler.wait_request(second, timeout=5)

        self.assertFalse(workers.is_open)

    def _make_slots(self, staging_bytes):
        return tuple(
            FakeAttentionSlot(slot_index, self.layout, staging_bytes, self.control)
            for slot_index in range(2)
        )


if __name__ == "__main__":
    unittest.main()
