#!/usr/bin/env python3

import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "python"))

from cemu_flexgen import (
    KvCacheLayout,
    KvCacheStore,
    KvLayoutConfig,
    KvStagingManager,
)
from flexgen_adapter import FlexGenAttentionBackend


class FakeFlexGenTensor:
    def __init__(self, data):
        self.data = data


class FakeAttentionDevice:
    def __init__(self):
        self.last_call = None

    def run_decode(self, query, layer, valid_tokens):
        self.last_call = (query, layer, valid_tokens)
        return query + np.float32(1)


class FlexGenAttentionBackendTest(unittest.TestCase):
    def setUp(self):
        nvm_root = os.environ.get("CEMU_NVM_TEST_DIR") or None
        fdm_root = os.environ.get("CEMU_FDM_TEST_DIR") or None
        self._validate_test_root("CEMU_NVM_TEST_DIR", nvm_root)
        self._validate_test_root("CEMU_FDM_TEST_DIR", fdm_root)

        self.nvm_directory = tempfile.TemporaryDirectory(
            prefix="cemu-prefill-nvm-",
            dir=nvm_root,
        )
        self.fdm_directory = tempfile.TemporaryDirectory(
            prefix="cemu-prefill-fdm-",
            dir=fdm_root,
        )
        self.nvm_path = Path(self.nvm_directory.name)
        self.fdm_path = Path(self.fdm_directory.name)
        self.layout = KvCacheLayout(
            KvLayoutConfig(
                num_layers=3,
                max_seq_len=7,
                batch_size=2,
                num_kv_heads=3,
                head_dim=4,
                dtype=np.float32,
            )
        )
        self.k_cache_path = self.nvm_path / "k_cache"
        self.v_cache_path = self.nvm_path / "v_cache"

    def tearDown(self):
        self.fdm_directory.cleanup()
        self.nvm_directory.cleanup()

    def test_prefill_writes_multiple_layers_and_stages_all_chunks(self):
        prompt_tokens = 5
        layer_zero_keys = self._flexgen_values(prompt_tokens, 0)
        layer_zero_values = self._flexgen_values(prompt_tokens, 1000)
        layer_two_keys = self._flexgen_values(prompt_tokens, 2000)
        layer_two_values = self._flexgen_values(prompt_tokens, 3000)

        with self._new_backend() as backend:
            written = backend.write_prefill(
                0,
                FakeFlexGenTensor(layer_zero_keys),
                FakeFlexGenTensor(layer_zero_values),
            )
            self.assertEqual(written, prompt_tokens)
            backend.write_prefill(2, layer_two_keys, layer_two_values)
            backend.flush()

        expected_layer_zero_keys = self._expected_cemu_layout(layer_zero_keys)
        expected_layer_zero_values = self._expected_cemu_layout(layer_zero_values)
        expected_layer_two_keys = self._expected_cemu_layout(layer_two_keys)
        expected_layer_two_values = self._expected_cemu_layout(layer_two_values)
        with KvCacheStore(
            self.layout,
            self.k_cache_path,
            self.v_cache_path,
        ) as store:
            stored_layer_zero = store.read_tokens(0, 0, prompt_tokens)
            stored_layer_two = store.read_tokens(2, 0, prompt_tokens)

        np.testing.assert_array_equal(stored_layer_zero[0], expected_layer_zero_keys)
        np.testing.assert_array_equal(stored_layer_zero[1], expected_layer_zero_values)
        np.testing.assert_array_equal(stored_layer_two[0], expected_layer_two_keys)
        np.testing.assert_array_equal(stored_layer_two[1], expected_layer_two_values)

        staging_bytes = 2 * self.layout.token_stride
        manager = KvStagingManager(
            layout=self.layout,
            k_cache_path=self.k_cache_path,
            v_cache_path=self.v_cache_path,
            k_staging_path=self.fdm_path / "k_staging_0",
            v_staging_path=self.fdm_path / "v_staging_0",
            staging_bytes=staging_bytes,
            replace_existing=True,
        )
        chunks = list(
            self.layout.iter_chunks(
                layer=2,
                valid_tokens=prompt_tokens,
                k_staging_bytes=staging_bytes,
                v_staging_bytes=staging_bytes,
            )
        )
        with manager:
            for chunk in chunks:
                manager.stage_chunk(chunk)
                staged_keys, staged_values = manager.read_staged_tokens()
                np.testing.assert_array_equal(
                    staged_keys,
                    expected_layer_two_keys[chunk.start_token : chunk.end_token],
                )
                np.testing.assert_array_equal(
                    staged_values,
                    expected_layer_two_values[chunk.start_token : chunk.end_token],
                )

        print(
            "\n[prefill] FlexGen GPU output -> CEMU NVM -> FDM: "
            f"layers=[0, 2], prompt_tokens={prompt_tokens}, chunks={len(chunks)}"
        )
        print(
            "  FlexGen shape="
            f"{layer_two_keys.shape}, CEMU shape={expected_layer_two_keys.shape}, "
            f"dtype={expected_layer_two_keys.dtype}"
        )
        print(
            "  layer=2 token=0 offset="
            f"{self.layout.token_offset(2, 0)}, "
            f"K[:8]={expected_layer_two_keys[0].reshape(-1)[:8].tolist()}"
        )

    def test_decode_append_uses_logical_token_offset(self):
        prefill_keys = self._flexgen_values(5, 0)
        prefill_values = self._flexgen_values(5, 1000)
        decode_key = self._flexgen_values(1, 4000)
        decode_value = self._flexgen_values(1, 5000)

        with self._new_backend() as backend:
            backend.write_prefill(1, prefill_keys, prefill_values)
            backend.append_decode(1, 5, decode_key, decode_value)
            backend.flush()

        with KvCacheStore(
            self.layout,
            self.k_cache_path,
            self.v_cache_path,
        ) as store:
            stored_key, stored_value = store.read_token(1, 5)

        np.testing.assert_array_equal(
            stored_key,
            self._expected_cemu_layout(decode_key)[0],
        )
        np.testing.assert_array_equal(
            stored_value,
            self._expected_cemu_layout(decode_value)[0],
        )

    def test_decode_converts_flexgen_query_shape(self):
        attention_device = FakeAttentionDevice()
        query = np.arange(2 * 4 * 4, dtype=np.float16).reshape(2 * 4, 1, 4)
        backend = self._new_backend(attention_device=attention_device)

        output = backend.decode(layer=2, query=query, valid_tokens=5)

        normalized_query, layer, valid_tokens = attention_device.last_call
        self.assertEqual(normalized_query.shape, (2, 4, 4))
        self.assertEqual(normalized_query.dtype, np.float32)
        self.assertEqual(layer, 2)
        self.assertEqual(valid_tokens, 5)
        np.testing.assert_array_equal(output, normalized_query + np.float32(1))

    def test_invalid_prefill_shape_is_rejected(self):
        invalid = np.zeros((5, 3, 4), dtype=np.float16)
        with self._new_backend() as backend:
            with self.assertRaises(ValueError):
                backend.write_prefill(0, invalid, invalid)

    def _new_backend(self, attention_device=None):
        return FlexGenAttentionBackend(
            layout=self.layout,
            k_cache_path=self.k_cache_path,
            v_cache_path=self.v_cache_path,
            attention_device=attention_device,
            replace_existing=True,
        )

    def _flexgen_values(self, token_count, start_value):
        element_count = (
            token_count
            * self.layout.config.batch_size
            * self.layout.config.num_kv_heads
            * self.layout.config.head_dim
        )
        return np.arange(
            start_value,
            start_value + element_count,
            dtype=np.float16,
        ).reshape(
            token_count,
            self.layout.config.batch_size * self.layout.config.num_kv_heads,
            self.layout.config.head_dim,
        )

    def _expected_cemu_layout(self, flexgen_array):
        return flexgen_array.reshape(
            flexgen_array.shape[0],
            self.layout.config.batch_size,
            self.layout.config.num_kv_heads,
            self.layout.config.head_dim,
        ).astype(self.layout.config.dtype)

    def _validate_test_root(self, name, path):
        if path is not None and not Path(path).is_dir():
            self.fail(f"{name} does not exist: {path}")


if __name__ == "__main__":
    unittest.main()
