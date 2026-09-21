import unittest
from concurrent.futures import Future
from threading import Event
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from cemu_flexgen import KvLayoutConfig, SparfKvCacheLayout
from cemu_flexgen.cemu_sparf_slot_scheduler import CemuSparfSlotScheduler
from flexgen_adapter import SparfGroupedAttentionBackend
from phase_profiler import PhaseProfiler


class _Store:
    def __init__(self):
        self.registered = True
        self.calls = []
        self.batch_calls = []
        self.cookie = 7
        self.flushes = 0
        self.drains = 0

    def append(self, stream, position, key, value):
        self.calls.append((stream, position, key.copy(), value.copy()))

    def append_batch(self, stream, position, keys, values):
        self.batch_calls.append((stream, position, keys.shape))
        for index in range(keys.shape[1]):
            self.append(stream + index, position, keys[:, index], values[:, index])

    def flush(self):
        self.flushes += 1

    def drain(self):
        self.drains += 1


class _Attention:
    is_open = False

    def open(self):
        self.is_open = True

    def close(self):
        self.is_open = False

    def run_decode(self, query, layer, valid_tokens, mean, **kwargs):
        self.request = (query, layer, valid_tokens, mean, kwargs)
        return np.zeros_like(query)


class _AsyncAttention(_Attention):
    def __init__(self):
        self.calls = []
        self.futures = []

    def submit_decode(self, query, layer, valid_tokens, mean, **kwargs):
        self.calls.append((query, layer, valid_tokens, mean, kwargs))
        future = Future()
        self.futures.append(future)
        return future


class GroupedBackendTest(unittest.TestCase):
    def test_stream_mapping_mean_and_decode_binding(self):
        layout = SparfKvCacheLayout(KvLayoutConfig(2, 8, 2, 2, 4, np.float16))
        store = _Store()
        attention = _Attention()
        backend = SparfGroupedAttentionBackend(layout, store, attention, stream_base=10)
        backend.open()
        keys = np.arange(2 * 2 * 2 * 4, dtype=np.float16).reshape(2, 4, 4)
        values = keys + 1
        backend.write_prefill(1, keys, values)
        self.assertEqual([call[0] for call in store.calls], [14, 15, 16, 17])
        query = np.zeros((2, 2, 4), np.float16)
        backend.append_decode(1, 2, keys[:1], values[:1])
        self.assertEqual(store.batch_calls, [(14, 0, (2, 4, 4)), (14, 2, (1, 4, 4))])
        backend.decode(1, query, 3)
        self.assertEqual(attention.request[1:3], (1, 3))
        self.assertEqual(attention.request[4]["grouped_stream_base"], 10)
        reshaped = values.reshape(2, 2, 2, 4)
        np.testing.assert_allclose(attention.request[3],
                                   (reshaped.sum(axis=0) + reshaped[0]) / 3)
        backend.close()

    def test_flush_keeps_session_appendable_and_close_does_not_drain_shared_store(self):
        layout = SparfKvCacheLayout(KvLayoutConfig(1, 8, 1, 2, 4, np.float16))
        store = _Store()
        backend = SparfGroupedAttentionBackend(layout, store, _Attention(), stream_base=0)
        keys = np.zeros((1, 2, 4), np.float16)
        with backend:
            backend.write_prefill(0, keys, keys)
            backend.flush()
            self.assertEqual(store.drains, 0)
            backend.append_decode(0, 1, keys, keys)
            backend.drain()
            self.assertEqual(store.drains, 1)
        self.assertTrue(store.registered)
        self.assertEqual(store.drains, 1)

    def test_pending_failure_still_closes_attention(self):
        layout = SparfKvCacheLayout(KvLayoutConfig(1, 8, 1, 2, 4, np.float16))
        attention = _Attention()
        backend = SparfGroupedAttentionBackend(layout, _Store(), attention, stream_base=0)
        backend.open()
        backend._pending = Future()
        backend._pending.set_exception(RuntimeError("injected Attention failure"))
        with self.assertRaisesRegex(RuntimeError, "injected Attention failure"):
            backend.close()
        self.assertFalse(attention.is_open)
        backend.close()

    def test_append_profiling_aggregates_per_layer_not_per_head(self):
        layout = SparfKvCacheLayout(KvLayoutConfig(1, 8, 2, 2, 4, np.float16))
        profiler = PhaseProfiler()
        backend = SparfGroupedAttentionBackend(layout, _Store(), _Attention(), 0, profiler=profiler)
        keys = np.ones((1, 4, 4), np.float16)
        with backend:
            backend.write_prefill(0, keys, keys)
            backend.append_decode(0, 1, keys, keys)
        metrics = {metric.name: metric for metric in profiler.snapshot()}
        self.assertEqual(metrics["prefill.grouped_append_wall"].count, 1)
        self.assertEqual(metrics["decode.grouped_append_wall"].count, 1)

    def test_decode_requires_contiguous_prefix_and_rejects_wrong_token(self):
        layout = SparfKvCacheLayout(KvLayoutConfig(1, 8, 1, 2, 4, np.float16))
        store = _Store()
        attention = _Attention()
        backend = SparfGroupedAttentionBackend(layout, store, attention, 0)
        with backend:
            keys = np.zeros((2, 2, 4), np.float16)
            backend.write_prefill(0, keys, keys)
            with self.assertRaisesRegex(ValueError, "next accepted token 2"):
                backend.append_decode(0, 1, keys[:1], keys[:1])
            with self.assertRaisesRegex(ValueError, "accepted KV prefix 2"):
                backend.decode(0, np.zeros((1, 2, 4), np.float16), 1)
            self.assertEqual(len(store.calls), 2)

    def test_async_submission_snapshots_inputs_and_preserves_completed_failure(self):
        layout = SparfKvCacheLayout(KvLayoutConfig(1, 8, 1, 2, 4, np.float16))
        attention = _AsyncAttention()
        backend = SparfGroupedAttentionBackend(layout, _Store(), attention, 0)
        backend.open()
        keys = np.ones((1, 2, 4), np.float16)
        backend.write_prefill(0, keys, keys)
        query = np.zeros((1, 2, 4), np.float16)
        request = backend.submit_decode(0, query, 1)
        query.fill(7)
        np.testing.assert_array_equal(attention.calls[0][0], np.zeros_like(query))
        attention.futures[0].set_exception(RuntimeError("injected async failure"))
        with self.assertRaisesRegex(RuntimeError, "injected async failure"):
            backend.append_decode(0, 1, keys, keys)
        self.assertIsNone(backend._pending)
        backend.close()

    def test_inflight_append_is_rejected_then_continues_after_completion(self):
        layout = SparfKvCacheLayout(KvLayoutConfig(1, 8, 1, 2, 4, np.float16))
        store, attention = _Store(), _AsyncAttention()
        with SparfGroupedAttentionBackend(layout, store, attention, 0) as backend:
            keys = np.ones((1, 2, 4), np.float16)
            backend.write_prefill(0, keys, keys)
            request = backend.submit_decode(0, keys, 1)
            with self.assertRaisesRegex(RuntimeError, "in flight"):
                backend.append_decode(0, 1, keys, keys)
            self.assertEqual(len(store.batch_calls), 1)
            attention.futures[0].set_result(keys)
            request.result()
            backend.append_decode(0, 1, keys, keys)
            self.assertEqual(backend._mean_count[0], 2)

    def test_completed_failure_is_preserved_by_sync_and_async_decode(self):
        layout = SparfKvCacheLayout(KvLayoutConfig(1, 8, 1, 2, 4, np.float16))
        keys = np.ones((1, 2, 4), np.float16)
        for method in ("decode", "submit_decode"):
            with self.subTest(method=method):
                attention = _AsyncAttention()
                with SparfGroupedAttentionBackend(layout, _Store(), attention, 0) as backend:
                    backend.write_prefill(0, keys, keys)
                    backend.submit_decode(0, keys, 1)
                    attention.futures[0].set_exception(RuntimeError("injected failure"))
                    with self.assertRaisesRegex(RuntimeError, "injected failure"):
                        getattr(backend, method)(0, keys, 1)
                    with self.assertRaisesRegex(RuntimeError, "poisoned"):
                        backend.append_decode(0, 1, keys, keys)
                    self.assertEqual(len(attention.calls), 1)

    def test_validation_and_partial_append_failure_do_not_advance_mean(self):
        layout = SparfKvCacheLayout(KvLayoutConfig(1, 2, 1, 2, 4, np.float16))
        store = _Store()
        keys = np.ones((1, 2, 4), np.float16)
        backend = SparfGroupedAttentionBackend(layout, store, _Attention(), 0)
        with backend:
            with self.assertRaisesRegex(ValueError, "exceeds"):
                backend.write_prefill(0, np.repeat(keys, 3, axis=0), np.repeat(keys, 3, axis=0))
            self.assertFalse(store.batch_calls)
            backend.write_prefill(0, keys, keys)
            with self.assertRaisesRegex(RuntimeError, "already accepted"):
                backend.write_prefill(0, keys, keys)
            with patch.object(store, "append_batch", side_effect=OSError("partial append")):
                with self.assertRaisesRegex(OSError, "partial append"):
                    backend.append_decode(0, 1, keys, keys)
            self.assertEqual(backend._mean_count[0], 1)
            with self.assertRaisesRegex(RuntimeError, "poisoned"):
                backend.decode(0, keys, 1)
        with self.assertRaisesRegex(RuntimeError, "new grouped backend"):
            backend.open()

    def test_shared_slots_preserve_microbatch_binding_and_reference_counts(self):
        entered, prepared = Event(), Event()
        bindings = []

        class Slot:
            staged = False
            autonomous = True

            def __init__(self):
                self.closed = True

            def open(self):
                self.closed = False

            def prepare_decode(self, query, layer, valid_tokens, mean, grouped_stream_base):
                request = SimpleNamespace(query=query, base=grouped_stream_base, mean=mean.copy())
                bindings.append((request.base, request.mean))
                if request.base == 4:
                    if not entered.wait(5):
                        raise AssertionError("first microbatch did not execute")
                    prepared.set()
                return request

            def execute_autonomous(self, request):
                if request.base == 0:
                    entered.set()
                    if not prepared.wait(5):
                        raise AssertionError("second microbatch preparation was serialized")

            def collect_decode(self, request):
                return np.full_like(request.query, request.base)

            def abort_decode(self, request):
                pass

            def close(self):
                self.closed = True

        layout = SparfKvCacheLayout(KvLayoutConfig(2, 8, 1, 2, 4, np.float16))
        store = _Store()
        slots = (Slot(), Slot())
        scheduler = CemuSparfSlotScheduler(slots)
        first = SparfGroupedAttentionBackend(layout, store, scheduler, 0).open()
        second = SparfGroupedAttentionBackend(layout, store, scheduler, 4).open()
        try:
            keys = np.ones((1, 2, 4), np.float16)
            first.write_prefill(1, keys, keys)
            second.write_prefill(1, keys, keys * 2)
            query = np.zeros((1, 4, 4), np.float16)
            first_request = first.submit_decode(1, query, 1)
            second_request = second.submit_decode(1, query, 1)
            np.testing.assert_array_equal(first_request.result(timeout=5), np.zeros_like(query))
            np.testing.assert_array_equal(second_request.result(timeout=5), np.full_like(query, 4))
            self.assertEqual([call[0] for call in store.calls], [2, 3, 6, 7])
            self.assertEqual([binding[0] for binding in bindings], [0, 4])
            np.testing.assert_array_equal(bindings[0][1], np.ones((1, 2, 4)))
            np.testing.assert_array_equal(bindings[1][1], np.full((1, 2, 4), 2))
            first.close()
            self.assertTrue(scheduler.is_open)
            second.close()
            self.assertTrue(all(slot.closed for slot in slots))
            self.assertEqual(store.drains, 0)
        finally:
            prepared.set()
            first.close()
            second.close()


if __name__ == "__main__":
    unittest.main()
