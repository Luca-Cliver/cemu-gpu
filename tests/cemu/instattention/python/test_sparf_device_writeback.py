import ctypes
import os
from pathlib import Path
import shlex
import struct
import subprocess
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np

from cemu_flexgen import KvLayoutConfig, SparfKvCacheLayout, SparfKvCacheStore
from cemu_flexgen.sparf_write_buffer import SparfWriteBuffer
from cemu_flexgen.sparf_writeback_abi import HEADER, SPAN, MATRIX, pack_writeback
from cemu_flexgen.sparf_device_writeback import SparfDeviceWriteback, initialized_extents, direct_read, initialize_unwritten_extents
from phase_profiler import PhaseProfiler

ROOT = Path(__file__).resolve().parents[4]


class DeviceWritebackTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.temp.cleanup)
        library = Path(cls.temp.name) / "writeback.so"
        glib = shlex.split(subprocess.check_output(["pkg-config", "--cflags", "--libs", "glib-2.0"], text=True))
        subprocess.run(["cc", "-shared", "-fPIC", "-std=gnu11", "-O1", "-I" + str(ROOT),
                        "-I" + str(ROOT / "include"), str(ROOT / "tests/cemu/instattention/guest/test_sparf_writeback.c"),
                        "-o", str(library), *glib], check=True)
        cls.lib = ctypes.CDLL(str(library))
        cls.lib.run_writeback.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_void_p,
            ctypes.c_uint64, ctypes.c_void_p, ctypes.c_uint64, ctypes.c_int, ctypes.c_void_p]
        cls.lib.run_writeback.restype = ctypes.c_int
        cls.lib.run_writeback_diagnostics.argtypes = cls.lib.run_writeback.argtypes + [ctypes.c_void_p]
        cls.lib.run_writeback_diagnostics.restype = ctypes.c_int

    def execute(self, spans, source=None, disk=None, fail=0, metadata=None):
        source = bytes(range(256)) * 32 if source is None else source
        disk = bytearray(b"\xa5" * 8192) if disk is None else disk
        metadata = HEADER.pack(1, len(spans), 0) + b"".join(SPAN.pack(*span) for span in spans) if metadata is None else metadata
        array = (ctypes.c_uint8 * len(disk)).from_buffer(disk)
        status = (ctypes.c_uint64 * 8)()
        result = self.lib.run_writeback(metadata, len(metadata), source, len(source), array, len(disk), fail, status)
        return result, disk, list(status)

    def test_partial_sector_preserves_neighbours_and_deduplicates(self):
        result, disk, status = self.execute([(510, 0, 8, 2, 16, 16), (513, 32, 2, 1, 2, 2)])
        expected = bytearray(b"\xa5" * 8192)
        source = bytes(range(256)) * 32
        expected[510:518], expected[526:534], expected[513:515] = source[:8], source[16:24], source[32:34]
        self.assertEqual(result, 2)
        self.assertEqual(disk, expected)
        self.assertEqual(status[:2], [1024, 1024])
        self.assertEqual(status[2:7], [1000, 1100, 1100, 1600, 1600])

    def test_reject_bad_metadata_before_io(self):
        for span in ((8190, 0, 8, 1, 8, 8), (0, 8190, 8, 1, 8, 8),
                     (0, 0, 8, 1, 4, 8), (0, 0, 8, 0, 8, 8),
                     (2**64-1, 0, 8, 1, 8, 8), (0, 0, 1, 2**32-1, 1, 1)):
            with self.subTest(span=span):
                result, disk, _ = self.execute([span])
                self.assertEqual(result, -1)
                self.assertEqual(disk, b"\xa5" * len(disk))
        for metadata in (b"", HEADER.pack(2, 1, 0), HEADER.pack(1, 0, 0), HEADER.pack(1, 65537, 0)):
            self.assertEqual(self.execute([], metadata=metadata)[0], -1)

    def test_io_errors_do_not_report_success(self):
        for phase in (1, 2):
            result, disk, status = self.execute([(510, 0, 8, 1, 8, 8)], fail=phase)
            self.assertEqual(result, -2)
            self.assertEqual(status[1], 0)
            self.assertEqual(disk, b"\xa5" * len(disk))

    def test_layout_ring_wrap_fragmented_extents_and_both_dtypes(self):
        for dtype in (np.float16, np.float32):
            layout = SparfKvCacheLayout(KvLayoutConfig(2, 20, 2, 2, 16, np.dtype(dtype)))
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                store = SparfKvCacheStore(layout, root / "k", root / "c", root / "v")
                buffer = SparfWriteBuffer(store, root / "buffer", 4)
                source = bytearray(buffer.file_bytes)
                rng = np.random.default_rng(57)
                keys = rng.normal(size=(4, 2, 2, 16)).astype(dtype)
                values = -keys
                for index in range(4):
                    offset = buffer._offset(1, 3 + index)
                    source[offset:offset + buffer.token_bytes] = keys[index].tobytes()
                    source[offset + buffer.token_bytes:offset + 2 * buffer.token_bytes] = values[index].tobytes()
                maps, logical_files, cursor = [], [], 0
                for size in (layout.token_file_size, layout.channel_file_size, layout.token_file_size):
                    mapping = []
                    for offset in range(0, size, 512):
                        mapping.append((cursor // 512, 1))
                        cursor += 1024
                    maps.append(mapping)
                    logical_files.append(bytearray(b"\xa5" * size))
                metadata = pack_writeback(buffer, 1, 3, 4, maps)
                result, disk, _ = self.execute([], bytes(source), bytearray(b"\xa5" * cursor), metadata=metadata)
                compact_metadata = pack_writeback(buffer, 1, 3, 4, maps, compact=True)
                compact_result, compact_disk, _ = self.execute(
                    [], bytes(source), bytearray(b"\xa5" * cursor), metadata=compact_metadata)
                self.assertEqual(compact_result, result)
                self.assertEqual(compact_disk, disk)
                self.assertEqual(result, 2)
                for index in range(4):
                    for batch in range(2):
                        for head in range(2):
                            offset = layout.token_head_offset(1, batch, head, 3 + index)
                            logical_files[0][offset:offset + layout.head_bytes] = keys[index, batch, head].tobytes()
                            logical_files[2][offset:offset + layout.head_bytes] = values[index, batch, head].tobytes()
                            for dimension in range(16):
                                offset = layout.channel_offset(1, batch, head, dimension, 3 + index)
                                logical_files[1][offset:offset + layout.element_size] = keys[index, batch, head, dimension].tobytes()
                for mapping, expected in zip(maps, logical_files):
                    actual = b"".join(disk[slba * 512:(slba + nlb) * 512] for slba, nlb in mapping)
                    self.assertEqual(actual, expected)
                for offset in range(512, cursor, 1024):
                    self.assertEqual(disk[offset:offset + 512], b"\xa5" * 512)

    def test_compact_channel_matrix_preserves_cross_extent_and_token_boundaries(self):
        layout = SparfKvCacheLayout(KvLayoutConfig(1, 300, 1, 1, 8, np.dtype(np.float16)))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            buffer = SparfWriteBuffer(SparfKvCacheStore(layout, root / "k", root / "c", root / "v"),
                                      root / "buffer", 6)
            maps, cursor = [], 0
            for size in (layout.token_file_size, layout.channel_file_size, layout.token_file_size):
                mapping = []
                for offset in range(0, size, 512):
                    mapping.append((cursor // 512, 1))
                    cursor += 1024
                maps.append(mapping)
            source = bytes(range(256)) * (buffer.file_bytes // 256)
            results = [self.execute([], source, bytearray(b"\xa5" * cursor),
                metadata=pack_writeback(buffer, 0, 255, 4, maps, compact=compact))
                for compact in (False, True)]
            self.assertEqual(results[0], results[1])
            self.assertEqual(results[0][0], 2)

    def test_compact_matrix_bounds_and_expansion_limits(self):
        base = (0, 0, 2, 3, 64, 2)
        for tail in ((0, 2, 512, 0), (65537, 0, 0, 0), (2, 8192, 512, 0),
                     (2, 2, 8192, 0), (1, 0, 0, 1)):
            metadata = HEADER.pack(2, 1, 0) + MATRIX.pack(*base, *tail)
            with self.subTest(tail=tail):
                result, disk, _ = self.execute([], metadata=metadata)
                self.assertEqual(result, -1)
                self.assertEqual(disk, b"\xa5" * 8192)
        valid = HEADER.pack(2, 1, 1) + MATRIX.pack(*base, 4, 2, 512, 0)
        result, disk, _ = self.execute([], metadata=valid)
        self.assertEqual(result, 2)
        source = bytes(range(256)) * 32
        for row in range(4):
            for token in range(3):
                self.assertEqual(disk[row * 512 + token * 2:row * 512 + token * 2 + 2],
                                 source[row * 2 + token * 64:row * 2 + token * 64 + 2])

    def test_opt_shape_metadata_is_bounded_by_heads_not_channels(self):
        layout = SparfKvCacheLayout(KvLayoutConfig(40, 24, 1, 40, 128, np.dtype(np.float16)))
        store = SparfKvCacheStore(layout, "unused-k", "unused-c", "unused-v")
        buffer = SparfWriteBuffer(store, "unused-buffer", 6)
        sizes = (layout.token_file_size, layout.channel_file_size, layout.token_file_size)
        maps, cursor = [], 0
        for size in sizes:
            maps.append(((cursor // 512, size // 512),))
            cursor += size
        for start, count in ((16, 3), (19, 3), (22, 2)):
            expanded = pack_writeback(buffer, 0, start, count, maps)
            compact = pack_writeback(buffer, 0, start, count, maps, compact=True)
            self.assertLess(len(compact), len(expanded) // 20)
            self.assertLessEqual(HEADER.unpack_from(compact)[1], 2 * 40 * 3)

    def test_profile_reports_distinct_host_and_target_clocks(self):
        profiler = PhaseProfiler()
        status = [1024, 1024, 100, 200, 200, 400, 400, 64,
                  1, 10, 20, 30, 40, 50, 60, 70, 80, 256, 4]
        SparfDeviceWriteback._record_status(profiler, status)
        metrics = {metric.name: metric for metric in profiler.snapshot()}
        self.assertEqual(metrics["kv_writeback.device.plan_host_ns"].total_ns, 10)
        self.assertEqual(metrics["kv_writeback.ftl_write_target_ns"].total_ns, 200)
        self.assertEqual(metrics["kv_writeback.ftl_read_target_ns"].byte_count, 1024)
        status[8] = 0
        with self.assertRaisesRegex(RuntimeError, "rebuild"):
            SparfDeviceWriteback._record_status(profiler, status)

    def test_production_diagnostics_do_not_change_fake_ftl_deadlines(self):
        for profile in (False, True):
            metadata = HEADER.pack(2, 1, int(profile)) + MATRIX.pack(0, 0, 2, 3, 64, 2, 4, 2, 512, 0)
            disk = ctypes.create_string_buffer(8192)
            result, diagnostics = (ctypes.c_uint64 * 8)(), (ctypes.c_uint64 * 11)()
            returned = self.lib.run_writeback_diagnostics(metadata, len(metadata), bytes(8192),
                8192, disk, 8192, 0, result, diagnostics)
            self.assertEqual(returned, 2)
            self.assertEqual(list(result)[2:7], [1000, 1100, 1100, 1600, 1600])
            if profile:
                self.assertEqual(diagnostics[0], 1)
                self.assertEqual(diagnostics[9], len(metadata))
                self.assertEqual(diagnostics[10], 4)
            else:
                self.assertEqual(list(diagnostics), [0] * 11)

    def test_status_read_is_opt_in_and_command_failure_is_not_hidden(self):
        layout = SparfKvCacheLayout(KvLayoutConfig(1, 8, 1, 1, 8, np.dtype(np.float16)))
        for profile in (False, True):
            buffer = SparfWriteBuffer(SparfKvCacheStore(layout, "unused-k", "unused-c", "unused-v"),
                                      "unused", 6)
            buffer.profiler = PhaseProfiler() if profile else None
            writeback = SparfDeviceWriteback("unused")
            writeback.maps = []
            cursor = 0
            for size in (layout.token_file_size, layout.channel_file_size, layout.token_file_size):
                writeback.maps.append(((cursor // 512, size // 512),))
                cursor += size
            writeback.device = Mock()
            writeback.device.execute.return_value = 0
            writeback.device.read_tensor.return_value = np.array(
                [512, 512, 100, 200, 200, 300, 300, 32, 1, 1, 2, 3, 4, 5, 6, 7, 8, 160, 10], dtype=np.uint64)
            writeback.writeback(buffer, 0, 0, 3)
            metadata = writeback.device.execute.call_args.kwargs["metadata"]
            self.assertEqual(HEADER.unpack_from(metadata)[::2], (2, int(profile)))
            self.assertEqual(writeback.device.read_tensor.call_count, int(profile))
            if profile:
                names = {metric.name for metric in buffer.profiler.snapshot()}
                self.assertTrue({"kv_writeback.metadata_wall", "kv_writeback.lock_wait_wall",
                                 "kv_writeback.execute_wait_wall", "kv_writeback.status_read_wall"} <= names)
            writeback.device.execute.side_effect = RuntimeError("injected execute failure")
            with self.assertRaisesRegex(RuntimeError, "injected"):
                writeback.writeback(buffer, 0, 3, 3)
            self.assertTrue(writeback.failed)

    def test_fiemap_rejects_holes_unwritten_and_shared_extents(self):
        def ioctl(flags, logical=0):
            def fill(descriptor, operation, packet, mutate):
                struct.pack_into("<I", packet, 20, 1)
                struct.pack_into("<QQQQQIIII", packet, 32, logical, 8192, 4096, 0, 0, flags, 0, 0, 0)
            return fill
        with patch("fcntl.ioctl", side_effect=ioctl(1)):
            self.assertEqual(initialized_extents(0, 4096), ((16, 8),))
        for flags, logical in ((0x800, 0), (0x2000, 0), (1, 4096)):
            with patch("fcntl.ioctl", side_effect=ioctl(flags, logical)), self.assertRaises(ValueError):
                initialized_extents(0, 4096)

    def test_direct_verification_read_handles_unaligned_slice(self):
        with tempfile.TemporaryFile() as stream:
            stream.write(bytes(range(256)) * 32)
            stream.flush()
            self.assertEqual(direct_read(stream.fileno(), 13, 4093), (bytes(range(256)) * 32)[4093:4106])

    def test_materialize_only_unwritten_ranges_before_handoff(self):
        extents = [(0, 8192, 4096, 0), (4096, 12288, 8192, 0x801)]
        with tempfile.TemporaryFile() as stream:
            stream.write(b"a" * 4096 + b"b" * 8192)
            stream.flush()
            with patch("cemu_flexgen.sparf_device_writeback.file_extents", return_value=extents):
                self.assertEqual(initialize_unwritten_extents(stream.fileno(), 12288), 8192)
            stream.seek(0)
            self.assertEqual(stream.read(), b"a" * 4096 + bytes(8192))

    def test_initialized_file_needs_no_rewrite(self):
        with patch("cemu_flexgen.sparf_device_writeback.file_extents",
                   return_value=[(0, 8192, 4096, 1)]), patch("os.pwrite") as write:
            self.assertEqual(initialize_unwritten_extents(0, 4096), 0)
            write.assert_not_called()


if __name__ == "__main__":
    unittest.main()
