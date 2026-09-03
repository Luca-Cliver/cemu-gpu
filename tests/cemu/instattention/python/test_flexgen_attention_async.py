#!/usr/bin/env python3

import sys
import tempfile
import threading
import unittest
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import numpy as np


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "python"))

from cemu_flexgen import KvCacheLayout, KvLayoutConfig
from flexgen_adapter import FlexGenAttentionBackend, FlexGenMicrobatchKvWriter


class BlockingAttentionDevice:
    def __init__(self):
        self.calls = []
        self.first_started = threading.Event()
        self.release_first = threading.Event()

    def run_decode(self, query, layer, valid_tokens):
        call_index = len(self.calls)
        self.calls.append((layer, valid_tokens, query.copy()))
        if call_index == 0:
            self.first_started.set()
            if not self.release_first.wait(timeout=5):
                raise TimeoutError("test did not release the first Attention request")
        return np.full_like(query, layer + valid_tokens)


class ImmediateAttentionRequest:
    def __init__(self, request_id, output):
        self.request_id = request_id
        self.output = output

    def result(self, timeout=None):
        return self.output


class RecordingAttentionScheduler:
    def __init__(self):
        self.calls = []

    def submit_decode(self, query, layer, valid_tokens):
        request_id = len(self.calls)
        self.calls.append((layer, valid_tokens, query.copy()))
        return ImmediateAttentionRequest(
            request_id,
            np.full_like(query, layer + valid_tokens),
        )


class PendingAttentionRequest:
    def __init__(self, request_id):
        self.request_id = request_id
        self.future = Future()

    def result(self, timeout=None):
        return self.future.result(timeout=timeout)

    def done(self):
        return self.future.done()


class PendingPipelineScheduler:
    def __init__(self):
        self.request = PendingAttentionRequest(0)
        self.store_executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="test-store",
        )
        self.store_futures = []

    def submit_decode(self, query, layer, valid_tokens):
        raise AssertionError("the pipelined path must not call submit_decode()")

    def prefetch_decode(self, layer, history_tokens):
        return SimpleNamespace(
            request_id=0,
            layer=layer,
            history_tokens=history_tokens,
        )

    def submit_prefetched_decode(self, prefetch, query, key, value):
        return self.request

    def wait_request(self, request, timeout=None):
        return request.result(timeout=timeout)

    def submit_store(self, function, *args):
        future = self.store_executor.submit(function, *args)
        self.store_futures.append(future)
        return future

    def close(self):
        self.store_executor.shutdown(wait=True)


class BlockingPrefillBackend:
    def __init__(self, layout):
        self.layout = layout
        self.started = threading.Event()
        self.release = threading.Event()
        self.calls = []

    def write_prefill(self, layer, keys, values):
        self.started.set()
        if not self.release.wait(timeout=5):
            raise TimeoutError("test did not release the Prefill Store")
        self.calls.append((layer, keys.copy(), values.copy()))

    def flush(self):
        pass


class FlexGenAttentionAsyncTest(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory(
            prefix="cemu-attention-async-"
        )
        self.test_directory = Path(self.temporary_directory.name)
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
        self.device = BlockingAttentionDevice()
        self.logs = []

    def tearDown(self):
        self.device.release_first.set()
        self.temporary_directory.cleanup()

    def test_single_worker_serializes_logical_attend_requests(self):
        query = np.arange(4, dtype=np.float32).reshape(1, 2, 2)
        first_key = np.array([[[10.0, 11.0]]], dtype=np.float32)
        first_value = np.array([[[20.0, 21.0]]], dtype=np.float32)
        second_key = np.array([[[30.0, 31.0]]], dtype=np.float32)
        second_value = np.array([[[40.0, 41.0]]], dtype=np.float32)

        with self._new_backend() as backend:
            first = backend.submit_attend(
                decode_step=0,
                layer=0,
                microbatch=0,
                token=0,
                query=query,
                key=first_key,
                value=first_value,
                valid_tokens=1,
            )
            self.assertTrue(self.device.first_started.wait(timeout=5))
            second = backend.submit_attend(
                decode_step=0,
                layer=1,
                microbatch=1,
                token=0,
                query=query + 1,
                key=second_key,
                value=second_value,
                valid_tokens=1,
            )

            self.assertFalse(first.done())
            self.assertFalse(second.done())
            self.assertEqual(len(self.device.calls), 1)
            self.device.release_first.set()

            np.testing.assert_array_equal(first.result(timeout=5), np.ones_like(query))
            np.testing.assert_array_equal(second.result(timeout=5), np.full_like(query, 2))
            stored_first = backend.store.read_token(0, 0)
            stored_second = backend.store.read_token(1, 0)

        np.testing.assert_array_equal(stored_first[0], first_key)
        np.testing.assert_array_equal(stored_first[1], first_value)
        np.testing.assert_array_equal(stored_second[0], second_key)
        np.testing.assert_array_equal(stored_second[1], second_value)
        self.assertEqual(first.request_id, 0)
        self.assertEqual(second.request_id, 1)
        self.assertEqual(
            [(layer, tokens) for layer, tokens, _ in self.device.calls],
            [(0, 1), (1, 1)],
        )
        self.assertTrue(any("attend submit request=0" in line for line in self.logs))
        self.assertTrue(any("attend complete request=1" in line for line in self.logs))

    def test_submit_requires_an_open_backend(self):
        backend = self._new_backend()
        token = np.zeros((1, 1, 2), dtype=np.float32)
        query = np.zeros((1, 2, 2), dtype=np.float32)

        with self.assertRaises(RuntimeError):
            backend.submit_attend(0, 0, 0, 0, query, token, token, 1)

    def test_decode_delegates_to_slot_scheduler(self):
        scheduler = RecordingAttentionScheduler()
        query = np.arange(4, dtype=np.float32).reshape(1, 2, 2)
        backend = FlexGenAttentionBackend(
            layout=self.layout,
            k_cache_path=self.test_directory / "scheduler_k_cache",
            v_cache_path=self.test_directory / "scheduler_v_cache",
            attention_scheduler=scheduler,
            replace_existing=True,
            logger=self.logs.append,
        )

        output = backend.decode(layer=1, query=query, valid_tokens=3)

        self.assertEqual(len(scheduler.calls), 1)
        layer, valid_tokens, normalized_query = scheduler.calls[0]
        self.assertEqual((layer, valid_tokens), (1, 3))
        np.testing.assert_array_equal(normalized_query, query)
        np.testing.assert_array_equal(output, np.full_like(query, 4))
        self.assertTrue(any("decode complete request=0" in line for line in self.logs))

    def test_device_and_scheduler_are_mutually_exclusive(self):
        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            FlexGenAttentionBackend(
                layout=self.layout,
                k_cache_path=self.test_directory / "invalid_k_cache",
                v_cache_path=self.test_directory / "invalid_v_cache",
                attention_device=self.device,
                attention_scheduler=RecordingAttentionScheduler(),
            )

    def test_decode_store_does_not_wait_for_attention_completion(self):
        scheduler = PendingPipelineScheduler()
        logs = []
        backend = FlexGenAttentionBackend(
            layout=self.layout,
            k_cache_path=self.test_directory / "pipeline_k_cache",
            v_cache_path=self.test_directory / "pipeline_v_cache",
            attention_scheduler=scheduler,
            replace_existing=True,
            logger=logs.append,
        )
        query = np.arange(4, dtype=np.float32).reshape(1, 2, 2)
        key = np.array([[[10.0, 11.0]]], dtype=np.float32)
        value = np.array([[[20.0, 21.0]]], dtype=np.float32)

        try:
            with backend:
                prefetch = backend.prefetch_decode(layer=0, history_tokens=1)
                request = backend.submit_prefetched_decode(
                    prefetch,
                    layer=0,
                    token=1,
                    query=query,
                    key=key,
                    value=value,
                    valid_tokens=2,
                )
                scheduler.store_futures[0].result(timeout=5)

                self.assertFalse(request.done())
                stored_key, stored_value = backend.store.read_token(0, 1)
                np.testing.assert_array_equal(stored_key, key)
                np.testing.assert_array_equal(stored_value, value)
                self.assertTrue(any("store-start request=0" in line for line in logs))
                self.assertFalse(any("store-wait request=0" in line for line in logs))

                scheduler.request.future.set_result(np.ones_like(query))
                np.testing.assert_array_equal(
                    backend.wait_decode(request),
                    np.ones_like(query),
                )
        finally:
            if not scheduler.request.done():
                scheduler.request.future.set_result(np.ones_like(query))
            scheduler.close()

    def test_microbatch_writer_splits_prefill_batch_without_copying_heads(self):
        second_layout = KvCacheLayout(
            KvLayoutConfig(
                num_layers=2,
                max_seq_len=4,
                batch_size=1,
                num_kv_heads=1,
                head_dim=2,
                dtype=np.float32,
            )
        )
        backends = tuple(
            FlexGenAttentionBackend(
                layout=second_layout,
                k_cache_path=self.test_directory / f"mb_{microbatch}_k",
                v_cache_path=self.test_directory / f"mb_{microbatch}_v",
                replace_existing=True,
            )
            for microbatch in range(2)
        )
        keys = np.arange(12, dtype=np.float32).reshape(3, 2, 2)
        values = keys + 100

        with backends[0], backends[1]:
            writer = FlexGenMicrobatchKvWriter(backends)
            self.assertEqual(writer.write_prefill(0, keys, values), 3)
            first = backends[0].store.read_tokens(0, 0, 3)
            second = backends[1].store.read_tokens(0, 0, 3)

        np.testing.assert_array_equal(first[0][:, 0, 0], keys[:, 0])
        np.testing.assert_array_equal(first[1][:, 0, 0], values[:, 0])
        np.testing.assert_array_equal(second[0][:, 0, 0], keys[:, 1])
        np.testing.assert_array_equal(second[1][:, 0, 0], values[:, 1])

    def test_microbatch_writer_runs_prefill_store_in_background(self):
        backend = BlockingPrefillBackend(self.layout)
        writer = FlexGenMicrobatchKvWriter((backend,))
        keys = np.arange(6, dtype=np.float32).reshape(3, 1, 2)
        values = keys + 100

        try:
            request = writer.submit_prefill(0, keys, values)
            self.assertTrue(backend.started.wait(timeout=5))
            self.assertFalse(request.future.done())
            backend.release.set()
            self.assertEqual(writer.wait_prefill(request), 3)
        finally:
            backend.release.set()
            writer.close()

        self.assertEqual(len(backend.calls), 1)
        np.testing.assert_array_equal(backend.calls[0][1], keys)
        np.testing.assert_array_equal(backend.calls[0][2], values)

    def _new_backend(self):
        return FlexGenAttentionBackend(
            layout=self.layout,
            k_cache_path=self.test_directory / "k_cache",
            v_cache_path=self.test_directory / "v_cache",
            attention_device=self.device,
            replace_existing=True,
            logger=self.logs.append,
        )


if __name__ == "__main__":
    unittest.main()
