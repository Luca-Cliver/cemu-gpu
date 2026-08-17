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
    KvChunk,
    KvLayoutConfig,
    KvStagingManager,
)


class KvStagingManagerTest(unittest.TestCase):
    def setUp(self):
        nvm_root = os.environ.get("CEMU_NVM_TEST_DIR") or None
        fdm_root = os.environ.get("CEMU_FDM_TEST_DIR") or None
        self._validate_test_root("CEMU_NVM_TEST_DIR", nvm_root)
        self._validate_test_root("CEMU_FDM_TEST_DIR", fdm_root)

        self.nvm_directory = tempfile.TemporaryDirectory(
            prefix="cemu-kv-nvm-",
            dir=nvm_root,
        )
        self.fdm_directory = tempfile.TemporaryDirectory(
            prefix="cemu-kv-fdm-",
            dir=fdm_root,
        )
        self.nvm_path = Path(self.nvm_directory.name)
        self.fdm_path = Path(self.fdm_directory.name)

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
        self.k_cache_path = self.nvm_path / "k_cache"
        self.v_cache_path = self.nvm_path / "v_cache"
        self.k_staging_path = self.fdm_path / "k_staging_0"
        self.v_staging_path = self.fdm_path / "v_staging_0"
        self.staging_bytes = 2 * self.layout.token_stride

        self.keys = np.stack(
            [self._token_values(token * 100) for token in range(5)]
        )
        self.values = self.keys + np.float16(7)
        with KvCacheStore(
            self.layout,
            self.k_cache_path,
            self.v_cache_path,
        ) as store:
            store.write_tokens(1, 0, self.keys, self.values)
            store.flush()

    def tearDown(self):
        self.fdm_directory.cleanup()
        self.nvm_directory.cleanup()

    def test_open_allocates_fdm_staging_files(self):
        manager = self._new_manager()

        self.assertFalse(manager.is_open)
        with manager:
            self.assertTrue(manager.is_open)
            self.assertEqual(manager.token_capacity, 2)
            self.assertEqual(manager.allocation_bytes, 4096)
            self.assertEqual(self.k_staging_path.stat().st_size, 4096)
            self.assertEqual(self.v_staging_path.stat().st_size, 4096)
            print(
                "\n[kv-staging] opened "
                f"NVM K={self.k_cache_path}, V={self.v_cache_path}"
            )
            print(
                "  FDM "
                f"K={self.k_staging_path}, V={self.v_staging_path}, "
                f"capacity={manager.token_capacity} tokens, "
                f"allocation={manager.allocation_bytes} bytes"
            )
        self.assertFalse(manager.is_open)

    def test_stage_every_chunk_and_read_back(self):
        chunks = list(
            self.layout.iter_chunks(
                layer=1,
                valid_tokens=5,
                k_staging_bytes=self.staging_bytes,
                v_staging_bytes=self.staging_bytes,
            )
        )

        with self._new_manager() as manager:
            for chunk_index, chunk in enumerate(chunks):
                copied = manager.stage_chunk(chunk)
                staged_keys, staged_values = manager.read_staged_tokens()
                expected_keys = self.keys[chunk.start_token : chunk.end_token]
                expected_values = self.values[chunk.start_token : chunk.end_token]

                self.assertEqual(copied, chunk.copy_size)
                self.assertEqual(manager.last_chunk, chunk)
                np.testing.assert_array_equal(staged_keys, expected_keys)
                np.testing.assert_array_equal(staged_values, expected_values)

                print(
                    f"\n[kv-staging] chunk {chunk_index}: "
                    f"layer={chunk.layer}, tokens=[{chunk.start_token}, "
                    f"{chunk.end_token}), nvm_offset={chunk.nvm_offset}, "
                    f"copied={copied} bytes"
                )
                print(
                    "  K staged[:8]="
                    f"{staged_keys[0].reshape(-1)[:8].tolist()}"
                )
                print(
                    "  V staged[:8]="
                    f"{staged_values[-1].reshape(-1)[:8].tolist()}"
                )

    def test_invalid_chunk_is_rejected(self):
        invalid_chunk = KvChunk(
            layer=1,
            start_token=0,
            token_count=3,
            nvm_offset=self.layout.token_offset(1, 0),
            copy_size=3 * self.layout.token_stride,
        )

        with self._new_manager() as manager:
            with self.assertRaises(ValueError):
                manager.stage_chunk(invalid_chunk)
            with self.assertRaises(RuntimeError):
                manager.read_staged_tokens()

    def test_wrong_source_file_size_is_rejected(self):
        os.truncate(self.k_cache_path, 512)

        with self.assertRaises(ValueError):
            self._new_manager().open()

    def _new_manager(self):
        return KvStagingManager(
            self.layout,
            self.k_cache_path,
            self.v_cache_path,
            self.k_staging_path,
            self.v_staging_path,
            self.staging_bytes,
        )

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

    def _validate_test_root(self, name, path):
        if path is not None and not Path(path).is_dir():
            self.fail(f"{name} does not exist: {path}")


if __name__ == "__main__":
    unittest.main()
