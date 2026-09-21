import ctypes
import os
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

from cemu_flexgen import KvLayoutConfig, SparfKvCacheLayout, SparfKvCacheStore
from cemu_flexgen.sparf_write_buffer import SparfWriteBuffer


class WriteBufferTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        root = Path(self.directory.name)
        self.layout = SparfKvCacheLayout(KvLayoutConfig(2, 32, 1, 2, 8, np.dtype(np.float32)))
        self.disk = SparfKvCacheStore(self.layout, root / "k", root / "c", root / "v")
        self.buffer = SparfWriteBuffer(self.disk, root / "fdm", 4).open()
        self.addCleanup(self.buffer.close)
        self.prefill = np.arange(3 * 16, dtype=np.float32).reshape(3, 1, 2, 8)
        for layer in range(2):
            self.buffer.write_tokens(layer, 0, self.prefill, -self.prefill)
        self.buffer.flush()

    def test_accept_and_snapshot_do_not_publish_or_modify_nvm(self):
        with patch.object(self.disk, "publish", wraps=self.disk.publish) as publish:
            self.buffer.write_token(0, 3, self.prefill[0], -self.prefill[0])
            keys, values = self.buffer.snapshot(0, 4)
            publish.assert_not_called()
        np.testing.assert_array_equal(keys, self.prefill[:1])
        np.testing.assert_array_equal(values, -self.prefill[:1])
        disk_keys, _ = self.disk.read_tokens(0, 3, 1)
        np.testing.assert_array_equal(disk_keys, 0)
        self.assertEqual(self.buffer.path.stat().st_size, self.buffer.file_bytes)

    def test_full_buffer_backpressure_and_snapshot_during_delayed_publish(self):
        entered, release = threading.Event(), threading.Event()
        original = self.disk.publish

        def slow_publish():
            entered.set()
            if not release.wait(5):
                raise RuntimeError("test publication timed out")
            original()

        with patch.object(self.disk, "publish", side_effect=slow_publish):
            for token in (3, 4):
                self.buffer.write_token(0, token, self.prefill[0] + token, self.prefill[0] - token)
            self.assertTrue(entered.wait(5))
            for token in (5, 6):
                self.buffer.write_token(0, token, self.prefill[0] + token, self.prefill[0] - token)
            keys, _ = self.buffer.snapshot(0, 7)
            self.assertEqual(len(keys), 4)
            with ThreadPoolExecutor(1) as executor:
                pending = executor.submit(self.buffer.write_token, 0, 7, self.prefill[0], self.prefill[0])
                try:
                    self.assertFalse(pending.done())
                finally:
                    release.set()
                pending.result(timeout=5)
        self.buffer.flush()
        self.assertEqual(len(self.buffer.snapshot(0, 8)[0]), 0)
        np.testing.assert_array_equal(keys[0], self.prefill[0] + 3)

    def test_wraparound_group_writeback_and_means_match_synchronous_store(self):
        for token in range(3, 19):
            for layer in range(2):
                self.buffer.write_token(layer, token, self.prefill[0] + token + layer,
                                         self.prefill[0] - token - layer)
        self.buffer.flush()
        for layer in range(2):
            keys, values = self.disk.read_tokens(layer, 0, 19)
            for token in range(3, 19):
                np.testing.assert_array_equal(keys[token], self.prefill[0] + token + layer)
            np.testing.assert_array_equal(self.buffer.value_mean(layer),
                values.astype(np.float64).mean(axis=0).astype(np.float32))
            channel = np.frombuffer(self.disk._pread(self.disk._channel_k_fd, 19 * 4,
                self.layout.channel_offset(layer, 0, 0, 0)), dtype=np.float32)
            np.testing.assert_array_equal(channel, keys[:, 0, 0, 0])

    def test_writeback_failure_keeps_visibility_frontier_and_fails_closed(self):
        failed = threading.Event()

        def fail_publish():
            failed.set()
            raise OSError("injected sync failure")

        with patch.object(self.disk, "publish", side_effect=fail_publish):
            for token in (3, 4):
                self.buffer.write_token(0, token, self.prefill[0], self.prefill[0])
            self.assertTrue(failed.wait(5))
            with self.assertRaisesRegex(RuntimeError, "writeback failed"):
                self.buffer.flush()
            self.assertEqual(self.buffer._published[0], 3)
            with self.assertRaisesRegex(RuntimeError, "writeback failed"):
                self.buffer.close()
        self.assertFalse(self.buffer.is_open)

    def test_gap_and_future_snapshot_rejected(self):
        with self.assertRaisesRegex(ValueError, "contiguous"):
            self.buffer.write_token(0, 5, self.prefill[0], self.prefill[0])
        with self.assertRaisesRegex(ValueError, "latest accepted"):
            self.buffer.snapshot(0, 4)

    def test_fdm_address_alignment_and_short_transfers(self):
        calls = {"read": 0, "write": 0}

        def aligned_io(operation, name):
            def transfer(descriptor, buffers, offset):
                self.assertEqual(descriptor, self.buffer._descriptor)
                self.assertEqual(offset % 512, 0)
                self.assertEqual(len(buffers), 1)
                view = buffers[0]
                self.assertEqual(ctypes.addressof(ctypes.c_char.from_buffer(view)) % 512, 0)
                self.assertEqual(len(view) % 512, 0)
                calls[name] += 1
                return operation(descriptor, [view[:512]], offset)
            return transfer

        with patch("os.preadv", side_effect=aligned_io(os.preadv, "read")), \
             patch("os.pwritev", side_effect=aligned_io(os.pwritev, "write")):
            for token in range(3, 15):
                self.buffer.write_token(0, token, self.prefill[0] + token, self.prefill[0] - token)
                keys, values = self.buffer.snapshot(0, token + 1)
                if len(keys):
                    np.testing.assert_array_equal(keys[-1], self.prefill[0] + token)
                    np.testing.assert_array_equal(values[-1], self.prefill[0] - token)
            self.buffer.flush()
        keys, values = self.disk.read_tokens(0, 3, 12)
        for index, token in enumerate(range(3, 15)):
            np.testing.assert_array_equal(keys[index], self.prefill[0] + token)
            np.testing.assert_array_equal(values[index], self.prefill[0] - token)
        self.assertGreater(calls["read"], 0)
        self.assertEqual(calls["write"], 12)

    def test_fdm_allocation_is_page_aligned_for_odd_capacity(self):
        self.buffer.close()
        self.buffer = SparfWriteBuffer(self.disk, Path(self.directory.name) / "odd-fdm", 3)
        self.addCleanup(self.buffer.close)
        original = os.posix_fallocate

        def aligned_allocation(descriptor, offset, length):
            if descriptor == self.buffer._descriptor:
                self.assertEqual(length % 4096, 0)
            return original(descriptor, offset, length)

        with patch("os.posix_fallocate", side_effect=aligned_allocation):
            self.buffer.open()
        self.assertEqual(self.buffer.file_bytes, 4096)

    def test_fdm_incomplete_transfer_does_not_accept_token(self):
        for returned in (0, 1):
            with patch("os.pwritev", return_value=returned):
                with self.assertRaisesRegex(OSError, "incomplete or unaligned"):
                    self.buffer.write_token(0, 3, self.prefill[0], self.prefill[0])
            self.assertEqual(self.buffer._accepted[0], 3)

    def test_device_writeback_does_not_read_buffer_into_guest_or_write_files(self):
        device = Mock()
        self.buffer.device_writeback = device
        with patch.object(self.buffer, "_read", side_effect=AssertionError("Guest readback")), \
             patch.object(self.disk, "write_tokens", side_effect=AssertionError("Guest file write")):
            self.buffer.write_token(0, 3, self.prefill[0], -self.prefill[0])
            self.buffer.write_token(0, 4, self.prefill[0], -self.prefill[0])
            self.buffer.flush()
        device.writeback.assert_called_once_with(self.buffer, 0, 3, 2)
        self.assertEqual(self.buffer._published[0], 5)

    def test_device_completion_gates_publish_and_failure_stops_reuse(self):
        entered, release = threading.Event(), threading.Event()
        device = Mock()
        def fail_writeback(*args):
            entered.set()
            if not release.wait(5):
                raise TimeoutError("test did not release device")
            raise OSError("device write failed")
        device.writeback.side_effect = fail_writeback
        self.buffer.device_writeback = device
        self.buffer.write_token(0, 3, self.prefill[0], -self.prefill[0])
        self.buffer.write_token(0, 4, self.prefill[0], -self.prefill[0])
        try:
            self.assertTrue(entered.wait(5))
            self.assertEqual(self.buffer._published[0], 3)
        finally:
            release.set()
        with self.assertRaisesRegex(RuntimeError, "writeback failed"):
            self.buffer.flush()
        self.assertEqual(self.buffer._published[0], 3)
        with self.assertRaisesRegex(RuntimeError, "writeback failed"):
            self.buffer.write_token(0, 5, self.prefill[0], -self.prefill[0])
        with self.assertRaisesRegex(RuntimeError, "writeback failed"):
            self.buffer.close()
        device.close.assert_called_once()

    def test_device_handoff_happens_once_after_prefill_publish(self):
        self.buffer.close()
        self.disk = SparfKvCacheStore(self.layout, Path(self.directory.name) / "new-k",
                                     Path(self.directory.name) / "new-c", Path(self.directory.name) / "new-v")
        device = Mock()
        self.buffer = SparfWriteBuffer(self.disk, Path(self.directory.name) / "new-buffer", 4, device).open()
        self.addCleanup(self.buffer.close)
        for layer in range(2):
            self.buffer.write_tokens(layer, 0, self.prefill, -self.prefill)
        def acquire(buffer):
            self.assertFalse(buffer.store._unpublished)
            buffer.store._device_owned = True
        device.acquire.side_effect = acquire
        self.buffer.flush()
        self.buffer.flush()
        device.acquire.assert_called_once_with(self.buffer)
        with self.assertRaisesRegex(RuntimeError, "belong to the CEMU device"):
            self.disk.write_tokens(0, 3, self.prefill[:1], -self.prefill[:1])


if __name__ == "__main__":
    unittest.main()
