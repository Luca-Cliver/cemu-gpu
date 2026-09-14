#!/usr/bin/env python3

import tempfile
import unittest
from pathlib import Path

import numpy as np

from cemu_flexgen import KvLayoutConfig, SparfKvCacheLayout, SparfKvCacheStore


class SparfKvStoreTest(unittest.TestCase):
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
