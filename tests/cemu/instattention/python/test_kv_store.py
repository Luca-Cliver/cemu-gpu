#!/usr/bin/env python3

import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "python"))

from cemu_flexgen import KvCacheLayout, KvCacheStore, KvLayoutConfig


class KvCacheStoreTest(unittest.TestCase):
    def setUp(self):
        test_root = os.environ.get("CEMU_KV_TEST_DIR") or None
        if test_root is not None and not Path(test_root).is_dir():
            self.fail(f"CEMU_KV_TEST_DIR does not exist: {test_root}")
        self.temporary_directory = tempfile.TemporaryDirectory(
            prefix="cemu-kv-store-",
            dir=test_root,
        )
        self.test_directory = Path(self.temporary_directory.name)
        self.layout = KvCacheLayout(
            KvLayoutConfig(
                num_layers=2,
                max_seq_len=5,
                batch_size=2,
                num_kv_heads=3,
                head_dim=5,
                dtype=np.float16,
            )
        )
        self.k_path = self.test_directory / "k_cache"
        self.v_path = self.test_directory / "v_cache"

    def tearDown(self):
        self.temporary_directory.cleanup()

    def test_open_preallocates_cache_files(self):
        store = self._new_store()

        self.assertFalse(store.is_open)
        with store:
            self.assertTrue(store.is_open)
            self.assertEqual(self.k_path.stat().st_size, self.layout.file_size)
            self.assertEqual(self.v_path.stat().st_size, self.layout.file_size)
        self.assertFalse(store.is_open)

    def test_write_and_read_single_token(self):
        key = self._token_values(10)
        value = self._token_values(100)

        with self._new_store() as store:
            store.write_token(1, 3, key, value)
            stored_key, stored_value = store.read_token(1, 3)

        np.testing.assert_array_equal(stored_key, key)
        np.testing.assert_array_equal(stored_value, value)

        offset = self.layout.token_offset(1, 3)
        with self.k_path.open("rb") as cache_file:
            cache_file.seek(offset)
            raw_token = cache_file.read(self.layout.token_stride)
        self.assertEqual(raw_token[: self.layout.token_bytes], key.tobytes())
        self.assertEqual(
            raw_token[self.layout.token_bytes :],
            bytes(self.layout.token_stride - self.layout.token_bytes),
        )

    def test_write_and_read_multiple_tokens(self):
        keys = np.stack([self._token_values(index * 100) for index in range(3)])
        values = keys + np.float16(7)

        with self._new_store() as store:
            store.write_tokens(0, 1, keys, values)
            stored_keys, stored_values = store.read_tokens(0, 1, 3)

        np.testing.assert_array_equal(stored_keys, keys)
        np.testing.assert_array_equal(stored_values, values)

    def test_data_persists_after_reopen(self):
        key = self._token_values(20)
        value = self._token_values(200)

        with self._new_store() as store:
            store.write_token(0, 4, key, value)
            store.flush()

        with self._new_store() as store:
            stored_key, stored_value = store.read_token(0, 4)

        np.testing.assert_array_equal(stored_key, key)
        np.testing.assert_array_equal(stored_value, value)

    def test_invalid_shape_dtype_and_range(self):
        token = self._token_values(0)

        with self._new_store() as store:
            with self.assertRaises(ValueError):
                store.write_token(0, 0, token.reshape(-1), token)
            with self.assertRaises(TypeError):
                store.write_token(0, 0, token.astype(np.float32), token)
            with self.assertRaises(ValueError):
                store.write_tokens(
                    0,
                    4,
                    np.stack([token, token]),
                    np.stack([token, token]),
                )
            with self.assertRaises(IndexError):
                store.read_token(2, 0)

    def test_incompatible_existing_file_is_rejected(self):
        self.k_path.write_bytes(b"invalid")
        self.v_path.write_bytes(b"invalid")

        with self.assertRaises(ValueError):
            self._new_store().open()

    def _new_store(self):
        return KvCacheStore(self.layout, self.k_path, self.v_path)

    def _token_values(self, start_value):
        element_count = (
            self.layout.config.batch_size
            * self.layout.config.num_kv_heads
            * self.layout.config.head_dim
        )
        return np.arange(
            start_value,
            start_value + element_count,
            dtype=self.layout.config.dtype,
        ).reshape(
            self.layout.config.batch_size,
            self.layout.config.num_kv_heads,
            self.layout.config.head_dim,
        )


if __name__ == "__main__":
    unittest.main()
