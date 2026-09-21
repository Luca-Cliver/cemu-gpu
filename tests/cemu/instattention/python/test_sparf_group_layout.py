import unittest

import numpy as np

from cemu_flexgen.sparf_group_layout import SparfGroupLayout


class SparfGroupLayoutTest(unittest.TestCase):
    def test_paper_page_geometry(self):
        for channels, capacity in ((2, 1024), (4, 512), (8, 256)):
            with self.subTest(channels=channels):
                layout = SparfGroupLayout(128, channels)
                self.assertEqual(layout.token_group_tokens, 16)
                self.assertEqual(layout.channel_group_tokens, capacity)
                self.assertEqual(layout.channel_group_pages, 128 // channels)

    def test_geometry_scales_with_dtype_and_page(self):
        self.assertEqual(SparfGroupLayout(128, 4, dtype=np.float32).token_group_tokens, 8)
        self.assertEqual(SparfGroupLayout(128, 4, dtype=np.float32).channel_group_tokens, 256)
        self.assertEqual(SparfGroupLayout(128, 4, page_bytes=8192).token_group_tokens, 32)

    def test_invalid_geometry(self):
        for arguments in (
            dict(head_dim=True, channels_per_page=4),
            dict(head_dim=128, channels_per_page=4.0),
            dict(head_dim=128, channels_per_page=0),
            dict(head_dim=127, channels_per_page=4),
            dict(head_dim=96, channels_per_page=4),
            dict(head_dim=4096, channels_per_page=4),
            dict(head_dim=128, channels_per_page=4, page_bytes=4095),
            dict(head_dim=128, channels_per_page=4, dtype=np.int16),
            dict(head_dim=128, channels_per_page=4, dtype=">f2"),
        ):
            with self.subTest(arguments=arguments), self.assertRaises((ValueError, TypeError)):
                SparfGroupLayout(**arguments)

    def test_full_and_partial_groups_round_trip(self):
        random = np.random.default_rng(17)
        for dtype in (np.float16, np.float32):
            for channels in (2, 4, 8):
                layout = SparfGroupLayout(128, channels, dtype=dtype)
                for kind, capacity in (("token", layout.token_group_tokens),
                                       ("channel", layout.channel_group_tokens)):
                    for count in (1, capacity - 1, capacity):
                        with self.subTest(dtype=dtype, channels=channels, kind=kind, count=count):
                            source = random.normal(size=(count, 256)).astype(dtype)[:, ::2]
                            if kind == "token":
                                data = layout.pack_token_page(source)
                                result = layout.unpack_token_page(data, count)
                                padding = np.frombuffer(data, dtype=dtype).reshape(capacity, 128)[count:]
                            else:
                                data = layout.pack_channel_group(source)
                                result = layout.unpack_channel_group(data, count)
                                padding = np.frombuffer(data, dtype=dtype).reshape(128, capacity)[:, count:]
                            np.testing.assert_array_equal(source, result)
                            self.assertTrue(np.all(padding == 0))

    def test_addresses_match_independent_encoded_bytes(self):
        layout = SparfGroupLayout(128, 8)
        count = layout.channel_group_tokens * 2 + 3
        source = np.random.default_rng(42).normal(size=(count, 128)).astype(np.float16)
        token_data = b"".join(layout.pack_token_page(source[start:start + layout.token_group_tokens])
                              for start in range(0, count, layout.token_group_tokens))
        channel_data = b"".join(layout.pack_channel_group(source[start:start + layout.channel_group_tokens])
                                for start in range(0, count, layout.channel_group_tokens))
        for token in (0, 15, 16, 255, 256, 511, 512, count - 1):
            for channel in range(layout.head_dim):
                expected = source[token, channel].tobytes()
                for address, data in ((layout.token_address(token, channel), token_data),
                                      (layout.channel_address(token, channel), channel_data)):
                    offset = address.page_index * layout.page_bytes + address.byte_offset
                    self.assertEqual(data[offset:offset + 2], expected)

    def test_sparse_page_selection_has_no_duplicate_reads(self):
        layout = SparfGroupLayout(128, 4)
        selected_tokens = (31, 1, 16, 15, 1)
        pages = {layout.token_address(token, 0).page_index for token in selected_tokens}
        self.assertEqual(pages, {0, 1})
        selected_channels = (0, 3, 4, 127, 0)
        pages = {layout.channel_address(token, channel).page_index
                 for token in (0, 511, 512) for channel in selected_channels}
        self.assertEqual(pages, {0, 1, 31, 32, 33, 63})

    def test_multiple_streams_and_prefill_decode_boundary(self):
        layout = SparfGroupLayout(128, 8)
        random = np.random.default_rng(7)
        streams = {}
        for layer in range(2):
            for batch in range(2):
                for head in range(2):
                    keys = random.normal(size=(260, 128)).astype(np.float16)
                    values = random.normal(size=keys.shape).astype(np.float16)
                    streams[layer, batch, head] = (keys, values)
        images = {}
        for stream, (keys, values) in streams.items():
            for kind, source, capacity, pack in (
                ("token_k", keys, layout.token_group_tokens, layout.pack_token_page),
                ("token_v", values, layout.token_group_tokens, layout.pack_token_page),
                ("channel_k", keys, layout.channel_group_tokens, layout.pack_channel_group),
            ):
                pending = np.empty((0, 128), dtype=np.float16)
                image = bytearray()
                for incoming in (source[:253], source[253:256], source[256:]):
                    pending = np.concatenate((pending, incoming))
                    while len(pending) >= capacity:
                        image.extend(pack(pending[:capacity]))
                        pending = pending[capacity:]
                self.assertEqual(len(pending), 4)
                image.extend(pack(pending))
                images[stream, kind] = image
            for token in (15, 16, 252, 253, 255, 256, 259):
                for kind, source, address_fn in (
                    ("token_k", keys, layout.token_address),
                    ("token_v", values, layout.token_address),
                    ("channel_k", keys, layout.channel_address),
                ):
                    address = address_fn(token, 127)
                    offset = address.page_index * layout.page_bytes + address.byte_offset
                    self.assertEqual(images[stream, kind][offset:offset + 2], source[token, 127].tobytes())

    def test_invalid_data_and_coordinates(self):
        layout = SparfGroupLayout(128, 4)
        for source in (np.zeros((1, 128), np.float32), np.zeros((0, 128), np.float16),
                       np.zeros((1, 127), np.float16), np.zeros((17, 128), np.float16)):
            with self.assertRaises(ValueError):
                layout.pack_token_page(source)
        with self.assertRaises(ValueError):
            layout.pack_channel_group(np.zeros((513, 128), np.float16))
        for token, channel in ((-1, 0), (0, -1), (0, 128), (True, 0), (0, 1.5)):
            for method in (layout.token_address, layout.channel_address):
                with self.assertRaises((ValueError, TypeError)):
                    method(token, channel)
        for unpack, size in ((layout.unpack_token_page, 4096),
                             (layout.unpack_channel_group, 32 * 4096)):
            for data, count in ((bytes(size - 1), 1), (bytes(size), 0), (bytes(size), 1025)):
                with self.assertRaises(ValueError):
                    unpack(data, count)


if __name__ == "__main__":
    unittest.main()
