#!/usr/bin/env python3

import tempfile
import unittest
from concurrent.futures import Future
from pathlib import Path
from unittest.mock import patch

import numpy as np

from cemu_flexgen import KvLayoutConfig, SparfKvCacheLayout, SparfKvCacheStore
from flexgen_adapter import SparfAttentionBackend
from phase_profiler import PhaseProfiler


class SparfKvStoreTest(unittest.TestCase):
    def test_batched_channel_append_preserves_padding_and_other_tokens(self):
        for dtype in (np.float16, np.float32):
            with self.subTest(dtype=dtype), tempfile.TemporaryDirectory() as directory:
                layout = SparfKvCacheLayout(KvLayoutConfig(2, 11, 2, 3, 7, dtype))
                paths = [Path(directory) / name for name in ("k", "c", "v")]
                expected = bytearray([0xA5]) * layout.channel_file_size
                with SparfKvCacheStore(layout, *paths) as store:
                    store._channel_map[:] = expected
                    for layer, start, count in ((1, 2, 2), (0, 5, 3), (1, 10, 1)):
                        source = np.arange(count * 2 * 3 * 14, dtype=dtype).reshape(count, 2, 3, 14)
                        keys = source[..., ::2]
                        values = -keys
                        with patch("numpy.copyto", wraps=np.copyto) as copy:
                            store.write_tokens(layer, start, keys, values)
                            self.assertEqual(copy.call_count, 1)
                            self.assertEqual(copy.call_args.kwargs["casting"], "no")
                            self.assertEqual(copy.call_args.args[0].shape, (2, 3, 7, count))
                        for batch in range(2):
                            for head in range(3):
                                for dimension in range(7):
                                    offset = layout.channel_offset(layer, batch, head, dimension, start)
                                    data = keys[:, batch, head, dimension].tobytes()
                                    expected[offset:offset + len(data)] = data
                        actual_keys, actual_values = store.read_tokens(layer, start, count)
                        np.testing.assert_array_equal(actual_keys, keys)
                        np.testing.assert_array_equal(actual_values, values)
                        self.assertEqual(store._channel_map[:], expected)
                    store.publish()
                self.assertEqual(paths[1].read_bytes(), expected)

    def test_opt_group_uses_one_channel_copy(self):
        layout = SparfKvCacheLayout(KvLayoutConfig(1, 17, 2, 40, 128, np.float16))
        keys = (np.arange(20480) % 127).astype(np.float16).reshape(2, 2, 40, 128)
        profiler = PhaseProfiler()
        with tempfile.TemporaryDirectory() as directory:
            paths = [Path(directory) / name for name in ("k", "c", "v")]
            with SparfKvCacheStore(layout, *paths, profiler=profiler) as store:
                with patch("numpy.copyto", wraps=np.copyto) as copy:
                    store.write_tokens(0, 9, keys, -keys)
                    self.assertEqual(copy.call_count, 1)
                    self.assertEqual(copy.call_args.args[0].size, 20480)
                for batch in range(2):
                    for head in range(40):
                        actual = store.read_channels(0, batch, head, range(128), 11)
                        np.testing.assert_array_equal(actual[:, 9:11], keys[:, batch, head].T)
        metrics = {row.name: row for row in profiler.snapshot()}
        self.assertEqual(metrics["kv_append.channel_k_mmap_wall"].count, 1)
        self.assertEqual(metrics["kv_append.channel_k_mmap_wall"].byte_count, keys.nbytes)

    def test_channel_copy_failure_retains_dirty_state_and_can_retry_write(self):
        layout = SparfKvCacheLayout(KvLayoutConfig(1, 9, 1, 2, 8, np.float32))
        keys = np.ones((2, 1, 2, 8), dtype=np.float32)
        with tempfile.TemporaryDirectory() as directory:
            paths = [Path(directory) / name for name in ("k", "c", "v")]
            with SparfKvCacheStore(layout, *paths) as store:
                with patch("numpy.copyto", side_effect=RuntimeError("copy failed")):
                    with self.assertRaisesRegex(RuntimeError, "copy failed"):
                        store.write_tokens(0, 3, keys, -keys)
                self.assertTrue(store._unpublished)
                self.assertEqual(store._value_counts[0], 0)
                store.write_tokens(0, 3, keys, -keys)
                store.publish()
                self.assertFalse(store._unpublished)
                np.testing.assert_array_equal(store.read_channels(0, 0, 1, range(8), 5)[:, 3:5],
                                              keys[:, 0, 1].T)

    def test_append_profiling_preserves_data_and_clean_publish_skips_work(self):
        layout = SparfKvCacheLayout(KvLayoutConfig(1, 9, 2, 3, 8, np.float32))
        key = np.arange(48, dtype=np.float32).reshape(2, 3, 8)
        profiler = PhaseProfiler()
        with tempfile.TemporaryDirectory() as directory:
            outputs = []
            for index, active in enumerate((None, profiler)):
                paths = [Path(directory) / f"{index}-{name}" for name in ("k", "c", "v")]
                with SparfKvCacheStore(layout, *paths, profiler=active) as store:
                    store.write_token(0, 0, key, key + 1)
                    store.publish()
                    store.publish()
                outputs.append([p.read_bytes() for p in paths])
            self.assertEqual(outputs[0], outputs[1])
        metrics = {r.name: r for r in profiler.snapshot()}
        self.assertEqual(metrics["kv_append.token_k_pwrite_wall"].byte_count, key.nbytes)
        self.assertEqual(metrics["kv_append.token_k_pwrite_wall"].count, 1)
        for name in ("token_k", "token_v", "channel_k"):
            self.assertEqual(metrics[f"kv_publish.{name}_fdatasync_wall"].count, 1)

    def test_backend_publishes_appends_before_sync_and_async_device_reads(self):
        """Model raw media separately from the Guest's buffered file contents."""
        layout = SparfKvCacheLayout(KvLayoutConfig(1, 9, 1, 2, 8, np.float32))
        test = self
        media = {}
        expected_k, expected_v = None, None

        class RawDevice:
            is_open = False
            calls = 0

            def open(self):
                self.is_open = True

            def close(self):
                self.is_open = False

            def run_decode(self, query, layer, count, mean, current_k, current_v, *, cache_paths):
                self.calls += 1
                for head in range(2):
                    offset = layout.token_head_offset(layer, 0, head, 0)
                    for path, expected in zip((cache_paths[0], cache_paths[2]),
                                              (expected_k, expected_v)):
                        actual = np.frombuffer(media[path], dtype=np.float32,
                                               count=count * 8, offset=offset).reshape(count, 8)
                        np.testing.assert_array_equal(actual, expected[:, 0, head])
                    for dimension in range(8):
                        actual = np.frombuffer(media[cache_paths[1]], dtype=np.float32,
                            count=count, offset=layout.channel_offset(layer, 0, head, dimension))
                        np.testing.assert_array_equal(actual, expected_k[:, 0, head, dimension])
                np.testing.assert_allclose(mean, expected_v.mean(axis=0), rtol=1e-6)
                return query.copy()

            def submit_decode(self, *args, **kwargs):
                future = Future()
                future.set_result(self.run_decode(*args, **kwargs))
                return future

        with tempfile.TemporaryDirectory() as directory:
            paths = tuple(Path(directory) / name for name in ("k", "c", "v"))
            device = RawDevice()
            with SparfAttentionBackend(layout, *paths, attention_device=device) as backend:
                descriptors = dict(zip((backend.store._token_k_fd, backend.store._channel_k_fd,
                                        backend.store._token_v_fd), paths))

                def publish_file(descriptor):
                    path = descriptors[descriptor]
                    media[path] = path.read_bytes()

                expected_k = np.arange(3 * 16, dtype=np.float32).reshape(3, 1, 2, 8)
                expected_v = expected_k + 100
                query = np.ones((1, 4, 8), dtype=np.float32)
                with patch("os.fdatasync", side_effect=publish_file):
                    backend.write_prefill(0, expected_k, expected_v)
                    backend.flush()
                    for step in range(4):
                        key = expected_k[-1] + 10
                        value = expected_v[-1] + 20
                        token = len(expected_k)
                        expected_k = np.concatenate((expected_k, key[None]))
                        expected_v = np.concatenate((expected_v, value[None]))
                        backend.append_decode(0, token, key[None], value[None])
                        # Sync failure must prevent submission, retain dirtiness
                        # and allow the next attempt to publish all files.
                        calls = device.calls
                        with patch("os.fdatasync", side_effect=OSError("publish failed")):
                            with test.assertRaisesRegex(OSError, "publish failed"):
                                if step % 2:
                                    backend.submit_decode(0, query, token + 1)
                                else:
                                    backend.decode(0, query, token + 1)
                        test.assertEqual(device.calls, calls)
                        if step % 2:
                            backend.submit_decode(0, query, token + 1).result()
                        else:
                            backend.decode(0, query, token + 1)
                    test.assertEqual(device.calls, 4)

    def test_publish_waits_for_all_files_and_retries_after_failure(self):
        layout = SparfKvCacheLayout(KvLayoutConfig(1, 9, 1, 1, 8, np.float32))
        with tempfile.TemporaryDirectory() as directory:
            paths = [Path(directory) / name for name in ("k", "c", "v")]
            with SparfKvCacheStore(layout, *paths) as store:
                key = np.ones((1, 1, 8), dtype=np.float32)
                store.write_token(0, 0, key, key)
                with patch("os.fdatasync", side_effect=[None, OSError("sync failed")]) as sync:
                    with self.assertRaisesRegex(OSError, "sync failed"):
                        store.publish()
                    self.assertEqual(sync.call_count, 2)
                with patch("os.fdatasync") as sync:
                    store.publish()
                    self.assertEqual([call.args[0] for call in sync.call_args_list],
                                     [store._token_k_fd, store._channel_k_fd, store._token_v_fd])
                    store.publish()
                    self.assertEqual(sync.call_count, 3)  # no new writes
                    store.write_token(0, 1, key, key)
                    store.publish()
                    self.assertEqual(sync.call_count, 6)

    def test_two_k_organizations_and_decode_append(self):
        layout = SparfKvCacheLayout(KvLayoutConfig(2, 9, 2, 3, 8, np.float16))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = (root / "k_token", root / "k_channel", root / "v_token")
            keys = np.arange(5 * 2 * 3 * 8, dtype=np.float16).reshape(5, 2, 3, 8)
            values = keys + np.float16(100)
            with SparfKvCacheStore(layout, *paths, replace_existing=True) as store:
                store.write_tokens(1, 0, keys, values)
                key = keys[-1] + np.float16(7)
                value = values[-1] + np.float16(11)
                store.write_token(1, 5, key, value)
                stored_keys, stored_values = store.read_tokens(1, 0, 6)
                np.testing.assert_array_equal(stored_keys[:5], keys)
                np.testing.assert_array_equal(stored_values[:5], values)
                np.testing.assert_array_equal(stored_keys[5], key)
                np.testing.assert_array_equal(stored_values[5], value)
                dimensions = (1, 6, 3)
                channels = store.read_channels(1, 1, 2, dimensions, 6)
                np.testing.assert_array_equal(channels, stored_keys[:, 1, 2, dimensions].T)
                np.testing.assert_allclose(
                    store.value_mean(1), stored_values.mean(axis=0), rtol=1e-3, atol=1e-3
                )
            self.assertEqual(paths[0].stat().st_size, layout.token_file_size)
            self.assertEqual(paths[1].stat().st_size, layout.channel_file_size)
            self.assertEqual(paths[2].stat().st_size, layout.token_file_size)


if __name__ == "__main__":
    unittest.main()
