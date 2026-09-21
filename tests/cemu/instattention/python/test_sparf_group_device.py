import ctypes
from pathlib import Path
import shlex
import subprocess
import tempfile
import unittest

import numpy as np

from cemu_flexgen.sparf_group_layout import SparfGroupLayout
from cemu_flexgen.sparf_group_device import (
    HEADER, REGISTER, PUT, READ_TOKENS, READ_CHANNELS, CLOSE,
    TOKEN_K, CHANNEL_K, TOKEN_V, command, registration,
)


ROOT = Path(__file__).resolve().parents[4]


class SparfGroupDeviceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.temp.cleanup)
        library = Path(cls.temp.name) / "group.so"
        glib = shlex.split(subprocess.check_output(["pkg-config", "--cflags", "--libs", "glib-2.0"], text=True))
        subprocess.run(["cc", "-shared", "-fPIC", "-std=gnu11", "-O1", "-I" + str(ROOT),
                        "-I" + str(ROOT / "include"), str(ROOT / "tests/cemu/instattention/guest/test_sparf_group.c"),
                        "-o", str(library), *glib], check=True)
        cls.library = ctypes.CDLL(str(library))
        cls.library.group_test_new.argtypes = [ctypes.c_void_p, ctypes.c_uint64]
        cls.library.group_test_new.restype = ctypes.c_void_p
        cls.library.group_test_run.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint32,
                                             ctypes.c_void_p, ctypes.c_uint64, ctypes.c_int, ctypes.c_void_p]
        cls.library.group_test_run.restype = ctypes.c_int
        cls.library.group_test_free.argtypes = [ctypes.c_void_p]
        cls.library.group_test_distinct_hashes.argtypes = [
            ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint32, ctypes.c_uint32,
        ]
        cls.library.group_test_distinct_hashes.restype = ctypes.c_uint32
        cls.library.group_test_extents.argtypes = [
            ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_void_p, ctypes.c_void_p,
        ]
        cls.library.group_test_extents.restype = ctypes.c_uint32
        cls.library.group_test_plan_ordinals.argtypes = [ctypes.c_uint32]
        cls.library.group_test_plan_ordinals.restype = ctypes.c_int

    def split_extents(self, sectors, sectors_per_page):
        sectors = np.asarray(sectors, dtype=np.uint64)
        starts = np.zeros(len(sectors), dtype=np.uint64)
        counts = np.zeros(len(sectors), dtype=np.uint32)
        count = self.library.group_test_extents(
            sectors.ctypes.data, len(sectors), sectors_per_page,
            starts.ctypes.data, counts.ctypes.data,
        )
        self.assertNotEqual(count, 2**32 - 1)
        self.assertEqual(int(counts[:count].sum()), len(sectors))
        return [(int(start), int(length)) for start, length in zip(starts[:count], counts[:count])]

    def test_large_transfers_split_on_page_boundaries_without_duplicate_pages(self):
        for page_sectors in (8, 128):
            for byte_count in (32 * 1024**2, 64 * 1024**2):
                with self.subTest(page_sectors=page_sectors, byte_count=byte_count):
                    sectors = np.arange(byte_count // 512, dtype=np.uint64)
                    extents = self.split_extents(sectors, page_sectors)
                    pages = []
                    for start, count in extents:
                        self.assertLessEqual(count, 65535)
                        self.assertEqual(start % page_sectors, 0)
                        self.assertEqual(count % page_sectors, 0)
                        pages.extend(range(start // page_sectors, (start + count) // page_sectors))
                    self.assertEqual(pages, list(range(len(sectors) // page_sectors)))
        self.assertEqual(self.split_extents(np.arange(65536), 8), [(0, 65528), (65528, 8)])

    def test_fragmented_and_partial_sector_transfers_preserve_exact_coverage(self):
        sectors = np.concatenate((np.arange(3, 65544), np.arange(80000, 80019)))
        extents = self.split_extents(sectors, 8)
        self.assertEqual(extents, [(3, 65533), (65536, 8), (80000, 19)])
        reconstructed = [sector for start, count in extents for sector in range(start, start + count)]
        self.assertEqual(reconstructed, sectors.tolist())
        self.assertEqual(self.split_extents([], 8), [])

    def test_plan_tags_follow_ordinals_in_reordered_small_and_6gib_pools(self):
        for pages in (8, 6 * 1024**3 // 4096):
            with self.subTest(pages=pages):
                self.assertEqual(self.library.group_test_plan_ordinals(pages), 1)

    def test_directory_hash_distributes_stream_and_group_pairs(self):
        streams, groups = 1024, 64
        entries = streams * groups * 3
        for stream_base, group_base in ((0, 0), (2**32, 2**32)):
            with self.subTest(stream_base=stream_base, group_base=group_base):
                distinct = self.library.group_test_distinct_hashes(
                    stream_base, group_base, streams, groups,
                )
                self.assertGreater(distinct, entries * 0.99)

    def setUp(self):
        self.disk = np.full(2 * 1024 * 1024, 0xA5, dtype=np.uint8)
        self.context = self.library.group_test_new(self.disk.ctypes.data, self.disk.nbytes)
        self.addCleanup(self.library.group_test_free, self.context)
        self.cookie = 1234
        self.layout = SparfGroupLayout(128, 8)
        self.extents = ((2048, 2048), (8, 1024))

    def execute(self, metadata, data=b"", fail=0, buffer_bytes=131072):
        buffer = np.full(buffer_bytes, 0xCC, dtype=np.uint8)
        buffer[:len(data)] = np.frombuffer(data, dtype=np.uint8)
        result = np.zeros(12, dtype=np.uint64)
        status = self.library.group_test_run(self.context, metadata, len(metadata), buffer.ctypes.data,
                                             buffer.nbytes, fail, result.ctypes.data)
        return status, buffer, result

    def register(self, layout=None, extents=None):
        layout = self.layout if layout is None else layout
        extents = self.extents if extents is None else extents
        self.assertEqual(self.execute(registration(layout, extents, self.cookie))[0], 0)

    def put(self, kind, stream, token, data):
        metadata = command(PUT, self.cookie, kind, stream, token, len(data))
        status, _, result = self.execute(metadata, data.tobytes())
        self.assertEqual(status, 1)
        self.assertEqual(result[0], 0)
        self.assertEqual(result[1], data.nbytes)
        self.assertEqual(result[4] + 500, result[5])
        self.assertEqual(result[5], result[6])
        return result

    def test_device_packing_matches_reference_and_no_rmw(self):
        self.register()
        source = np.random.default_rng(9).normal(size=(256, 128)).astype(np.float16)
        self.put(TOKEN_K, 0, 0, source[:16])
        self.put(CHANNEL_K, 0, 0, source)
        self.put(TOKEN_V, 0, 0, -source[:16])
        expected = self.layout.pack_token_page(source[:16]) + self.layout.pack_channel_group(source) + self.layout.pack_token_page(-source[:16])
        start = self.extents[0][0] * 512
        self.assertEqual(self.disk[start:start + len(expected)].tobytes(), expected)
        self.assertTrue(np.all(self.disk[:start] == 0xA5))
        self.assertTrue(np.all(self.disk[start + len(expected):] == 0xA5))

    def test_sparse_reads_restore_order_deduplicate_pages_and_isolate_streams(self):
        self.register()
        random = np.random.default_rng(11)
        for stream in (0, 91):
            keys = random.normal(size=(512, 128)).astype(np.float16)
            values = random.normal(size=keys.shape).astype(np.float16)
            for start in range(0, 512, 16):
                self.put(TOKEN_K, stream, start, keys[start:start + 16])
                self.put(TOKEN_V, stream, start, values[start:start + 16])
            for start in (0, 256):
                self.put(CHANNEL_K, stream, start, keys[start:start + 256])
            for kind, source in ((TOKEN_K, keys), (TOKEN_V, values)):
                tokens = (31, 1, 16, 15, 1, 511)
                status, data, result = self.execute(command(READ_TOKENS, self.cookie, kind, stream,
                                                           count=len(tokens), indices=tokens))
                self.assertEqual(status, 1)
                self.assertEqual(result[0], 3 * 4096)
                self.assertEqual(result[1], 0)
                self.assertEqual(result[2] + 100, result[3])
                np.testing.assert_array_equal(data[:len(tokens) * 256].view(np.float16).reshape(-1, 128), source[list(tokens)])
            channels = (127, 0, 7, 0)
            status, data, result = self.execute(command(READ_CHANNELS, self.cookie, CHANNEL_K, stream,
                                                       253, len(channels), 8, channels))
            self.assertEqual(status, 1)
            self.assertEqual(result[0], 4 * 4096)
            self.assertEqual(result[11], 8 * len(channels) * 2)
            np.testing.assert_array_equal(data[:int(result[11])].view(np.float16).reshape(len(channels), 8),
                                          keys[253:261, list(channels)].T)

    def test_fragmented_pool_and_float32(self):
        layout = SparfGroupLayout(128, 4, dtype=np.float32)
        extents = ((1024, 8), (8, 512))
        self.register(layout, extents)
        source = np.random.default_rng(4).normal(size=(256, 128)).astype(np.float32)
        self.put(CHANNEL_K, 3, 0, source)
        packed = layout.pack_channel_group(source)
        self.assertEqual(self.disk[1024 * 512:1024 * 512 + 4096].tobytes(), packed[:4096])
        self.assertEqual(self.disk[4096:4096 + len(packed) - 4096].tobytes(), packed[4096:])
        metadata = command(READ_CHANNELS, self.cookie, CHANNEL_K, 3, 0, 2, 256, (0, 127))
        status, data, result = self.execute(metadata)
        self.assertEqual(status, 1)
        np.testing.assert_array_equal(data[:int(result[11])].view(np.float32).reshape(2, 256), source[:, [0, 127]].T)

    def test_reject_unaligned_overlapping_out_of_bounds_and_bad_registration(self):
        for extents in ((), ((1, 8),), ((8, 7),), ((8, 16), (16, 8)), ((4096, 8),), ((2**64 - 8, 16),)):
            with self.subTest(extents=extents):
                self.assertEqual(self.execute(registration(self.layout, extents, self.cookie))[0], -1)
        valid = registration(self.layout, self.extents, self.cookie)
        for metadata in (b"", valid[:-1], valid + b"x"):
            self.assertEqual(self.execute(metadata)[0], -1)
        fields = list(HEADER.unpack_from(valid))
        for field, value in ((0, 2), (3, 0), (4, 3), (5, 3), (6, 512), (11, 0)):
            bad = fields.copy()
            bad[field] = value
            self.assertEqual(self.execute(HEADER.pack(*bad) + valid[HEADER.size:])[0], -1)
        self.assertTrue(np.all(self.disk == 0xA5))
        self.register()
        self.assertEqual(self.execute(valid)[0], -1)

    def test_missing_pages_bad_cookie_and_invalid_requests_do_not_touch_io(self):
        self.register()
        packets = (
            command(READ_TOKENS, self.cookie, TOKEN_K, count=1, indices=(0,)),
            command(PUT, self.cookie + 1, TOKEN_K, count=16),
            command(PUT, self.cookie, TOKEN_K, position=1, count=16),
            command(PUT, self.cookie, TOKEN_K, count=15),
            command(PUT, self.cookie, TOKEN_K, count=16, argument=2**64 - 1),
            command(READ_CHANNELS, self.cookie, CHANNEL_K, count=1, argument=1, indices=(128,)),
            command(READ_CHANNELS, self.cookie, CHANNEL_K, position=2**64 - 1, count=1, argument=2, indices=(0,)),
            command(READ_CHANNELS, self.cookie, CHANNEL_K, count=1, argument=2**64 - 1, indices=(0,)),
            command(READ_TOKENS, self.cookie, CHANNEL_K, count=1, indices=(0,)),
        )
        for metadata in packets:
            with self.subTest(metadata=metadata):
                self.assertEqual(self.execute(metadata)[0], -1)
        self.assertTrue(np.all(self.disk == 0xA5))

    def test_duplicate_and_capacity_exhaustion_preserve_committed_data(self):
        self.register(extents=((8, 8),))
        source = np.ones((16, 128), dtype=np.float16)
        self.put(TOKEN_K, 0, 0, source)
        original = self.disk.copy()
        for token in (0, 16):
            self.assertEqual(self.execute(command(PUT, self.cookie, TOKEN_K, position=token, count=16),
                                          (-source).tobytes())[0], -1)
        np.testing.assert_array_equal(original, self.disk)
        status, data, result = self.execute(command(READ_TOKENS, self.cookie, TOKEN_K, count=1, indices=(0,)))
        self.assertEqual(status, 1)
        self.assertEqual(result[9], 1)
        self.assertEqual(result[10], 1)
        np.testing.assert_array_equal(data[:256].view(np.float16), source[0])

    def test_io_failure_poison_and_close_releases_directory(self):
        self.register()
        metadata = command(PUT, self.cookie, TOKEN_K, count=16)
        status, _, result = self.execute(metadata, fail=1)
        self.assertEqual(status, -1)
        self.assertEqual(result[9], 1)
        self.assertEqual(result[10], 0)
        self.assertEqual(self.execute(metadata)[0], -1)
        self.assertEqual(self.execute(command(CLOSE, self.cookie + 1))[0], -1)
        self.assertEqual(self.execute(command(CLOSE, self.cookie))[0], 0)
        self.cookie += 1
        self.register()
        self.assertEqual(self.execute(command(READ_TOKENS, self.cookie, TOKEN_K, count=1, indices=(0,)))[0], -1)
        self.put(TOKEN_K, 0, 0, np.ones((16, 128), np.float16))
        read = command(READ_TOKENS, self.cookie, TOKEN_K, count=1, indices=(0,))
        self.assertEqual(self.execute(read, fail=1)[0], -1)
        self.assertEqual(self.execute(read)[0], -1)

    def test_short_output_rejected_without_mutation(self):
        self.register()
        self.put(TOKEN_K, 0, 0, np.ones((16, 128), np.float16))
        status, buffer, _ = self.execute(command(READ_TOKENS, self.cookie, TOKEN_K, count=1, indices=(0,)),
                                          buffer_bytes=128)
        self.assertEqual(status, -1)
        self.assertTrue(np.all(buffer == 0xCC))


if __name__ == "__main__":
    unittest.main()
