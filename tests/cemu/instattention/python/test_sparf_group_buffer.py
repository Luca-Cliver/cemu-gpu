import ctypes
import os
from pathlib import Path
import shlex
import subprocess
import tempfile
import threading
import time
import unittest

import numpy as np

from cemu_flexgen.sparf_group_layout import SparfGroupLayout
from cemu_flexgen.sparf_group_device import (
    APPEND, APPEND_BATCH, DRAIN, STATUS, STATS, CLOSE, PUT, READ_TOKENS, READ_CHANNELS,
    TOKEN_K, CHANNEL_K, TOKEN_V, HEADER, SparfGroupedDevice, command, registration,
)


ROOT = Path(__file__).resolve().parents[4]


class GroupBufferTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.temp.cleanup)
        library = Path(cls.temp.name) / "buffer.so"
        flags = shlex.split(subprocess.check_output(["pkg-config", "--cflags", "--libs", "glib-2.0"], text=True))
        subprocess.run(["cc", "-shared", "-fPIC", "-std=gnu11", "-O1", "-pthread", "-Wall", "-Wextra",
                        "-I" + str(ROOT), "-I" + str(ROOT / "include"),
                        str(ROOT / "tests/cemu/instattention/guest/test_sparf_group_buffer.c"),
                        "-o", str(library), *flags], check=True)
        cls.library = ctypes.CDLL(str(library))
        cls.library.group_buffer_test_new.argtypes = [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_uint64]
        cls.library.group_buffer_test_new.restype = ctypes.c_void_p
        cls.library.group_buffer_test_run.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint32,
                                                     ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p]
        cls.library.group_buffer_test_run.restype = ctypes.c_int
        cls.library.group_buffer_test_gate.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
        cls.library.group_buffer_test_wait_write.argtypes = [ctypes.c_void_p]
        cls.library.group_buffer_test_wait_write.restype = ctypes.c_int
        cls.library.group_buffer_test_free.argtypes = [ctypes.c_void_p]
        cls.library.group_buffer_test_target.argtypes = [ctypes.c_void_p, ctypes.c_int]
        cls.library.group_buffer_test_target_entered.argtypes = [ctypes.c_void_p]
        cls.library.group_buffer_test_target_entered.restype = ctypes.c_int
        cls.library.group_buffer_test_abort.argtypes = [ctypes.c_void_p]
        cls.library.group_buffer_test_abort.restype = ctypes.c_uint64
        cls.library.group_buffer_test_totals.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        cls.library.group_buffer_test_write_stats.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        cls.library.group_buffer_test_lun_pages.argtypes = [ctypes.c_void_p, ctypes.c_uint32,
                                                          ctypes.c_uint32, ctypes.c_uint32, ctypes.c_void_p]

    def setUp(self):
        self.layout = SparfGroupLayout(128, 8)
        self.disk = np.full(4 * 1024 * 1024, 0xA5, np.uint8)
        self.storage = np.full(2 * 1024 * 1024, 0xCC, np.uint8)
        self.context = self.library.group_buffer_test_new(self.disk.ctypes.data, self.disk.nbytes,
                                                         self.storage.ctypes.data, self.storage.nbytes)
        self.addCleanup(self.library.group_buffer_test_free, self.context)
        self.cookie = 7654
        self.keys = np.random.default_rng(915).normal(size=(2, 1051, 128)).astype(np.float16)
        self.values = np.random.default_rng(916).normal(size=self.keys.shape).astype(np.float16)

    def execute(self, metadata, data=b""):
        io = np.zeros(max(131072, len(data)), np.uint8)
        io[:len(data)] = np.frombuffer(data, np.uint8)
        result = np.zeros(20, np.uint64)
        status = self.library.group_buffer_test_run(self.context, metadata, len(metadata), io.ctypes.data,
                                                    io.nbytes, result.ctypes.data)
        return status, io, result

    def register(self, layout=None, streams=2, sectors=4096, batch_pages=0):
        self.layout = layout or self.layout
        self.assertEqual(self.execute(registration(self.layout, ((2048, sectors),), self.cookie, streams, batch_pages))[0], 1)

    def append(self, stream, start, stop):
        data = self.keys[stream, start:stop].tobytes() + self.values[stream, start:stop].tobytes()
        return self.execute(command(APPEND, self.cookie, stream=stream, position=start, count=stop - start), data)

    def append_batch(self, start, stop):
        data = np.stack((self.keys[:, start:stop].transpose(1, 0, 2),
                         self.values[:, start:stop].transpose(1, 0, 2))).tobytes()
        return self.execute(command(APPEND_BATCH, self.cookie, position=start,
                                    count=stop - start, argument=len(self.keys)), data)

    def test_abort_discards_unsealed_tail_without_drain(self):
        self.register()
        self.assertEqual(self.append_batch(0, 15)[0], 1)
        self.assertEqual(self.library.group_buffer_test_abort(self.context), 0)
        totals = np.zeros(2, np.uint64)
        self.library.group_buffer_test_totals(self.context, totals.ctypes.data)
        np.testing.assert_array_equal(totals, [0, 0])

    def test_abort_cancels_frozen_target_without_publishing(self):
        self.register()
        self.library.group_buffer_test_target(self.context, 1)
        self.assertEqual(self.append_batch(0, 16)[0], 1)
        deadline = time.monotonic() + 2
        while not self.library.group_buffer_test_target_entered(self.context):
            self.assertLess(time.monotonic(), deadline)
            time.sleep(0.001)
        published = []
        worker = threading.Thread(target=lambda: published.append(
            self.library.group_buffer_test_abort(self.context)), daemon=True)
        worker.start()
        try:
            worker.join(timeout=2)
            self.assertFalse(worker.is_alive(), "cleanup waited for frozen target time")
            self.assertEqual(published, [0])
        finally:
            self.library.group_buffer_test_target(self.context, 0)
            worker.join(timeout=2)

    def test_abort_waits_for_inflight_transfer_before_freeing(self):
        self.register()
        self.library.group_buffer_test_gate(self.context, 1, 0)
        self.assertEqual(self.append_batch(0, 16)[0], 1)
        self.assertEqual(self.library.group_buffer_test_wait_write(self.context), 1)
        published = []
        worker = threading.Thread(target=lambda: published.append(
            self.library.group_buffer_test_abort(self.context)), daemon=True)
        worker.start()
        try:
            worker.join(timeout=0.05)
            self.assertTrue(worker.is_alive(), "freed storage before transfer returned")
        finally:
            self.library.group_buffer_test_gate(self.context, 0, 0)
            worker.join(timeout=2)
        self.assertFalse(worker.is_alive())
        self.assertEqual(published, [0])

    def test_drain_waits_for_target_and_publishes_normally(self):
        self.register()
        self.library.group_buffer_test_target(self.context, 1)
        self.assertEqual(self.append_batch(0, 16)[0], 1)
        completed = []
        worker = threading.Thread(target=lambda: completed.append(
            self.execute(command(DRAIN, self.cookie))[0]), daemon=True)
        worker.start()
        try:
            worker.join(timeout=0.05)
            self.assertTrue(worker.is_alive())
            self.assertEqual(completed, [])
        finally:
            self.library.group_buffer_test_target(self.context, 0)
            worker.join(timeout=2)
        self.assertFalse(worker.is_alive())
        self.assertEqual(completed, [1])
        for stream in range(2):
            np.testing.assert_array_equal(self.progress(stream)[:6], [16, 16, 16, 16, 1, 0])

    def test_batch_wrap_tail_and_terminal_drain(self):
        self.register(batch_pages=256)
        self.assertEqual(self.append_batch(0, 15)[0], 1)
        for stream in range(2):
            self.check_tokens(stream, [0, 14], 0)
        self.assertEqual(self.append_batch(15, 531)[0], 1)
        for stream in range(2):
            self.check_tokens(stream, [15, 16, 255, 256, 511, 530])
            self.check_channels(stream, 253, 278)
        self.assertEqual(self.execute(command(DRAIN, self.cookie))[0], 1)
        for stream in range(2):
            np.testing.assert_array_equal(self.progress(stream)[:6], [531, 531, 531, 531, 1, 0])
            self.check_channels(stream, 0, 531)

    def test_batch_rejection_does_not_partially_append(self):
        self.register()
        self.assertEqual(self.append(1, 0, 1)[0], 1)
        self.assertEqual(self.append_batch(0, 2)[0], 0)
        self.assertEqual(self.progress(0)[0], 0)
        self.assertEqual(self.progress(1)[0], 1)
        for streams, count in ((0, 1), (3, 1), (2**64 - 1, 1), (1, 0), (2, 1000)):
            self.assertEqual(self.execute(command(APPEND_BATCH, self.cookie, argument=streams, count=count))[0], 0)
        self.assertEqual(self.progress(0)[0], 0)

    def test_batch_write_failure_does_not_publish(self):
        self.register(batch_pages=256)
        self.library.group_buffer_test_gate(self.context, 1, 0)
        result = []
        thread = threading.Thread(target=lambda: result.append(self.append_batch(0, 65)[0]))
        thread.start()
        try:
            self.assertEqual(self.library.group_buffer_test_wait_write(self.context), 1)
            for stream in range(2):
                np.testing.assert_array_equal(self.progress(stream)[:4], [32, 0, 0, 0])
                self.check_tokens(stream, [0, 31], 0)
        finally:
            self.library.group_buffer_test_gate(self.context, 0, 1)
            thread.join(3)
        self.assertFalse(thread.is_alive())
        self.assertEqual(result, [0])
        for stream in range(2):
            np.testing.assert_array_equal(self.progress(stream)[1:4], [0, 0, 0])

    def test_256_page_batch_across_heads_and_publish_after_completion(self):
        self.disk = np.zeros(16 * 1024 * 1024, np.uint8)
        self.storage = np.zeros(32 * 1024 * 1024, np.uint8)
        self.context = self.library.group_buffer_test_new(self.disk.ctypes.data, self.disk.nbytes,
                                                         self.storage.ctypes.data, self.storage.nbytes)
        self.addCleanup(self.library.group_buffer_test_free, self.context)
        self.keys = np.arange(128 * 16 * 128, dtype=np.int32).reshape(128, 16, 128).astype(np.float32)
        self.keys = (self.keys % 1000).astype(np.float16)
        self.values = -self.keys
        self.register(streams=128, sectors=28672, batch_pages=256)
        self.library.group_buffer_test_gate(self.context, 1, 0)
        try:
            self.assertEqual(self.append_batch(0, 16)[0], 1)
            self.assertEqual(self.library.group_buffer_test_wait_write(self.context), 1)
            for stream in (0, 63, 127):
                np.testing.assert_array_equal(self.progress(stream)[:4], [16, 0, 0, 0])
                self.check_tokens(stream, [0, 15], 0)
        finally:
            self.library.group_buffer_test_gate(self.context, 0, 0)
        self.wait_progress(lambda progress: progress[1] == progress[3] == 16)
        statistics = np.zeros(2, np.uint64)
        self.library.group_buffer_test_write_stats(self.context, statistics.ctypes.data)
        np.testing.assert_array_equal(statistics, [1, 256 * 4096])
        status, data, result = self.execute(command(STATS, self.cookie))
        self.assertEqual(status, 1)
        np.testing.assert_array_equal(data[:40].view(np.uint64), [1, 256, 1, 128, 2048])
        for kind in (TOKEN_K, TOKEN_V):
            histogram = np.zeros((8, 6), np.uint64)
            self.library.group_buffer_test_lun_pages(self.context, kind, 8, 6, histogram.ctypes.data)
            self.assertEqual(int(histogram.sum()), 128)
            if os.environ.get("CEMU_SPARF_GROUP_WRITE_ORDER") == "kind":
                np.testing.assert_array_equal(histogram.sum(axis=1), np.full(8, 16))
                self.assertEqual(np.count_nonzero(histogram), 48)
                self.assertEqual(int(histogram.max()), 3)
            else:
                expected = np.zeros(8, np.uint64)
                expected[0 if kind == TOKEN_K else 1::2] = 32
                np.testing.assert_array_equal(histogram.sum(axis=1), expected)
                self.assertEqual(np.count_nonzero(histogram), 24)
                self.assertEqual(int(histogram.max()), 6)
        for stream in (0, 63, 127):
            self.check_tokens(stream, [15, 0])

    def test_registration_rejects_bad_batch_geometry(self):
        for pages in (1, 15, 16385, 2**64 - 1):
            self.assertEqual(self.execute(registration(self.layout, ((2048, 4096),), self.cookie, 2, pages))[0], 0)
        metadata = registration(self.layout, ((2048, 4096),), self.cookie, 2, 256)
        for field in (3, 4, 5, 6):
            header = list(HEADER.unpack_from(metadata))
            header[field] = 0
            self.assertEqual(self.execute(HEADER.pack(*header) + metadata[HEADER.size:])[0], 0)

    def test_batch_float32(self):
        self.keys = self.keys.astype(np.float32)
        self.values = self.values.astype(np.float32)
        self.register(SparfGroupLayout(128, 4, dtype=np.float32), batch_pages=256)
        self.assertEqual(self.append_batch(0, 531)[0], 1)
        self.assertEqual(self.execute(command(DRAIN, self.cookie))[0], 1)
        for stream in range(2):
            self.check_tokens(stream, [0, 7, 8, 255, 256, 530])
            self.check_channels(stream, 253, 278)

    def test_statistics_rejects_extra_fields_and_does_not_flush(self):
        self.register()
        self.assertEqual(self.append_batch(0, 15)[0], 1)
        for field in ("kind", "count", "position", "argument", "stream"):
            self.assertEqual(self.execute(command(STATS, self.cookie, **{field: 1}))[0], 0)
        status, data, _ = self.execute(command(STATS, self.cookie))
        self.assertEqual(status, 1)
        np.testing.assert_array_equal(data[:40].view(np.uint64), [0, 0, 1, 2, 30])
        for stream in range(2):
            np.testing.assert_array_equal(self.progress(stream)[:6], [15, 0, 0, 0, 0, 0])

    def progress(self, stream=0):
        status, _, result = self.execute(command(STATUS, self.cookie, stream=stream))
        self.assertEqual(status, 1)
        return result[12:]

    def wait_progress(self, predicate):
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            progress = self.progress()
            if predicate(progress):
                return progress
            time.sleep(0.001)
        self.fail(f"background progress timed out: {progress}")

    def check_tokens(self, stream, tokens, expected_read=None):
        for kind, expected in ((TOKEN_K, self.keys), (TOKEN_V, self.values)):
            status, data, result = self.execute(command(READ_TOKENS, self.cookie, kind, stream,
                                                        count=len(tokens), indices=tokens))
            self.assertEqual(status, 1)
            actual = data[:int(result[11])].view(expected.dtype).reshape(-1, 128)
            np.testing.assert_array_equal(actual, expected[stream, tokens])
            if expected_read is not None:
                self.assertEqual(result[0], expected_read)

    def check_channels(self, stream, start, count, expected_read=None):
        channels = (127, 0, 7, 0)
        status, data, result = self.execute(command(READ_CHANNELS, self.cookie, CHANNEL_K, stream,
                                                    start, len(channels), count, channels))
        self.assertEqual(status, 1)
        actual = data[:int(result[11])].view(self.keys.dtype).reshape(len(channels), count)
        np.testing.assert_array_equal(actual, self.keys[stream, start:start + count][:, channels].T)
        if expected_read is not None:
            self.assertEqual(result[0], expected_read)

    def test_tail_only_reads_do_not_issue_nvm_reads(self):
        self.register()
        self.assertEqual(self.append(0, 0, 15)[0], 1)
        np.testing.assert_array_equal(self.progress()[:4], [15, 0, 0, 0])
        self.check_tokens(0, [14, 0, 1, 1], 0)
        self.check_channels(0, 0, 15, 0)

    def test_read_and_append_while_target_write_is_pending(self):
        self.register()
        self.library.group_buffer_test_gate(self.context, 1, 0)
        try:
            self.assertEqual(self.append(0, 0, 16)[0], 1)
            self.assertEqual(self.library.group_buffer_test_wait_write(self.context), 1)
            np.testing.assert_array_equal(self.progress()[:4], [16, 0, 0, 0])
            self.check_tokens(0, [15, 0, 7], 0)
            self.assertEqual(self.append(1, 0, 15)[0], 1)
            self.check_channels(1, 0, 15, 0)
        finally:
            self.library.group_buffer_test_gate(self.context, 0, 0)
        self.wait_progress(lambda progress: progress[1] == progress[3] == 16)

    def test_independent_watermarks_and_mixed_prefix_tail(self):
        self.register()
        self.assertEqual(self.append(0, 0, 255)[0], 1)
        progress = self.wait_progress(lambda progress: progress[1] == progress[3] == 240)
        np.testing.assert_array_equal(progress[:4], [255, 240, 0, 240])
        self.check_tokens(0, [254, 0, 239, 240, 0], 8192)
        self.check_channels(0, 237, 18, 0)
        self.assertEqual(self.append(0, 255, 259)[0], 1)
        self.wait_progress(lambda progress: all(progress[index] == 256 for index in (1, 2, 3)))
        self.check_channels(0, 253, 6, 8192)
        self.check_tokens(0, [258, 253, 255, 256, 255], 4096)

    def test_backpressure_unblocks_after_write_completion(self):
        self.register()
        self.library.group_buffer_test_gate(self.context, 1, 0)
        result = []
        thread = threading.Thread(target=lambda: result.append(self.append(0, 0, 65)[0]))
        thread.start()
        try:
            self.assertEqual(self.library.group_buffer_test_wait_write(self.context), 1)
            thread.join(0.05)
            self.assertTrue(thread.is_alive())
            self.assertEqual(self.progress()[0], 32)
        finally:
            self.library.group_buffer_test_gate(self.context, 0, 0)
            thread.join(3)
        self.assertFalse(thread.is_alive())
        self.assertEqual(result, [1])
        self.check_tokens(0, [0, 31, 32, 64])

    def test_prefill_decode_wrap_and_partial_terminal_drain(self):
        self.register()
        for stream in range(2):
            self.assertEqual(self.append(stream, 0, 253)[0], 1)
        for start in range(253, 1051):
            for stream in range(2):
                self.assertEqual(self.append(stream, start, start + 1)[0], 1)
        self.assertEqual(self.execute(command(DRAIN, self.cookie))[0], 1)
        before = self.progress()
        for stream in range(2):
            progress = self.progress(stream)
            np.testing.assert_array_equal(progress[:6], [1051, 1051, 1051, 1051, 1, 0])
            self.check_tokens(stream, [0, 15, 16, 255, 256, 511, 512, 1050])
            for start in range(0, 1051, 128):
                self.check_channels(stream, start, min(128, 1051 - start))
        self.assertEqual(self.execute(command(DRAIN, self.cookie))[0], 1)
        self.assertEqual(before[6], self.progress()[6])
        self.assertEqual(self.append(0, 1050, 1051)[0], 0)
        self.assertEqual(before[6], 2 * (2 * 66 + 5 * 16) * 4096)

    def test_rejects_gaps_duplicates_bounds_and_put_without_mutation(self):
        self.register()
        self.assertEqual(self.append(0, 0, 15)[0], 1)
        before = self.progress().copy()
        self.assertEqual(self.append(0, 0, 15)[0], 0)
        self.assertEqual(self.append(0, 16, 17)[0], 0)
        self.assertEqual(self.execute(command(APPEND, self.cookie, stream=2, count=1), bytes(512))[0], 0)
        self.assertEqual(self.execute(command(APPEND, self.cookie, position=15, count=300))[0], 0)
        self.assertEqual(self.execute(command(PUT, self.cookie, count=16), bytes(4096))[0], 0)
        self.assertEqual(self.execute(command(READ_TOKENS, self.cookie, count=1, indices=(15,)))[0], 0)
        self.assertEqual(self.execute(command(READ_CHANNELS, self.cookie, CHANNEL_K, count=1, argument=16, indices=(0,)))[0], 0)
        np.testing.assert_array_equal(self.progress(), before)

    def test_pool_exhaustion_poison_and_cleanup(self):
        self.register(sectors=8)
        self.assertEqual(self.append(0, 0, 16)[0], 1)
        self.wait_progress(lambda progress: progress[5] == 1)
        self.assertEqual(self.execute(command(DRAIN, self.cookie))[0], 0)
        self.assertEqual(self.append(0, 16, 17)[0], 0)
        self.execute(command(CLOSE, self.cookie))
        self.register()
        self.assertEqual(self.progress()[0], 0)

    def test_write_failure_wakes_blocked_append(self):
        self.register()
        self.library.group_buffer_test_gate(self.context, 0, 1)
        self.assertEqual(self.append(0, 0, 65)[0], 0)
        self.assertEqual(self.progress()[5], 1)
        self.assertEqual(self.execute(command(DRAIN, self.cookie))[0], 0)

    def test_registration_budget_and_float32(self):
        self.assertEqual(self.execute(registration(self.layout, ((2048, 4096),), self.cookie, 4096))[0], 0)
        self.layout = SparfGroupLayout(128, 4, dtype=np.float32)
        self.keys = self.keys.astype(np.float32)
        self.values = self.values.astype(np.float32)
        self.register(self.layout)
        self.assertEqual(self.append(0, 0, 259)[0], 1)
        self.check_channels(0, 253, 6)
        self.assertEqual(self.execute(command(DRAIN, self.cookie))[0], 1)
        self.check_tokens(0, [0, 127, 128, 258])
        totals = np.zeros(2, np.uint64)
        self.library.group_buffer_test_totals(self.context, totals.ctypes.data)
        self.assertEqual(totals[1], (2 * 33 + 2 * 32) * 4096)

    def test_close_drains_without_explicit_drain_and_reopens(self):
        self.register()
        self.assertEqual(self.append(0, 0, 3)[0], 1)
        self.assertEqual(self.execute(command(CLOSE, self.cookie))[0], 1)
        totals = np.zeros(2, np.uint64)
        self.library.group_buffer_test_totals(self.context, totals.ctypes.data)
        np.testing.assert_array_equal(totals, [0, 18 * 4096])
        self.register()
        np.testing.assert_array_equal(self.progress()[:4], [0, 0, 0, 0])

    def test_python_adapter_splits_prefill_and_preserves_incremental_positions(self):
        self.check_python_adapter(False)

    def test_python_batch_adapter_splits_tokens_and_streams(self):
        self.check_python_adapter(True)

    def check_python_adapter(self, batched):
        self.register()
        test = self

        class Client:
            def __init__(self):
                self.payload = b""
                self.data = None
                self.result = None

            def write_tensor(self, index, values):
                test.assertEqual(index, 0)
                self.payload = values.tobytes()

            def execute(self, cparam1, metadata):
                status, self.data, self.result = test.execute(metadata, self.payload)
                return 0 if status else 1

            def read_tensor(self, index, shape, dtype):
                if index == 1:
                    return self.result.copy().reshape(shape)
                count = int(np.prod(shape))
                return self.data[:count * np.dtype(dtype).itemsize].view(dtype).reshape(shape).copy()

        device = SparfGroupedDevice(self.layout, "/unused/pool", 2097152, "/unused/fdm", "/unused/program",
                                     buffer_streams=2)
        device.cookie = self.cookie
        device.device = Client()
        self.assertEqual(device.tail_bytes, 344064)
        if batched:
            device.buffer_bytes = 16 * 2 * 128 * 2
            device.append_batch(0, 0, self.keys[:, :530].transpose(1, 0, 2),
                                self.values[:, :530].transpose(1, 0, 2))
            device.append_batch(0, 530, self.keys[:, 530:531].transpose(1, 0, 2),
                                self.values[:, 530:531].transpose(1, 0, 2))
        else:
            device.append(0, 0, self.keys[0, :530], self.values[0, :530])
            device.append(0, 530, self.keys[0, 530:531], self.values[0, 530:531])
        device.drain()
        progress = device.progress(0)
        self.assertEqual([progress[name] for name in ("accepted", "token_k", "channel_k", "token_v")], [531] * 4)
        np.testing.assert_array_equal(device.read_tokens(TOKEN_K, 0, [0, 255, 530]), self.keys[0, [0, 255, 530]])
        np.testing.assert_array_equal(device.read_channels(0, 253, 8, [127, 0]), self.keys[0, 253:261][:, [127, 0]].T)
        statistics = device.statistics()
        self.assertEqual(statistics["appended_tokens"], 1062 if batched else 531)
        self.assertEqual(statistics["append_commands"], 68 if batched else 2)
        self.assertLessEqual(statistics["max_write_pages"], 16)
        if batched:
            self.check_tokens(1, [0, 15, 255, 530])
            self.check_channels(1, 253, 278)


if __name__ == "__main__":
    unittest.main()
