#!/usr/bin/env python3

import sys
import unittest
from pathlib import Path

import numpy as np


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "python"))

from cemu_flexgen import KvCacheLayout, KvLayoutConfig, align_up


class KvCacheLayoutTest(unittest.TestCase):
    def test_llama_style_layout(self):
        layout = KvCacheLayout(
            KvLayoutConfig(
                num_layers=32,
                max_seq_len=4096,
                batch_size=1,
                num_kv_heads=32,
                head_dim=128,
                dtype=np.float16,
            )
        )

        self.assertEqual(layout.element_size, 2)
        self.assertEqual(layout.head_bytes, 256)
        self.assertEqual(layout.token_bytes, 8192)
        self.assertEqual(layout.token_stride, 8192)
        self.assertEqual(layout.layer_stride, 32 * 1024 * 1024)
        self.assertEqual(layout.file_size, 1024 * 1024 * 1024)

        token_offset = layout.token_offset(3, 100)
        self.assertEqual(token_offset, 3 * layout.layer_stride + 100 * 8192)
        self.assertEqual(
            layout.head_offset(3, 100, 0, 7),
            token_offset + 7 * 256,
        )
        self.assertEqual(
            layout.element_offset(3, 100, 0, 7, 11),
            token_offset + 7 * 256 + 11 * 2,
        )

    def test_token_and_layer_padding(self):
        layout = KvCacheLayout(
            KvLayoutConfig(
                num_layers=2,
                max_seq_len=10,
                batch_size=1,
                num_kv_heads=3,
                head_dim=80,
                dtype=np.float16,
            )
        )

        self.assertEqual(layout.token_bytes, 480)
        self.assertEqual(layout.token_stride, 512)
        self.assertEqual(layout.layer_stride, 8192)
        self.assertEqual(layout.file_size, 16384)
        self.assertEqual(layout.token_offset(1, 0), 8192)
        self.assertEqual(layout.token_offset(1, 9), 8192 + 9 * 512)

    def test_chunk_capacity_from_staging_buffers(self):
        layout = self._small_layout()

        self.assertEqual(layout.tokens_per_chunk(4 * 512, 3 * 512), 3)
        with self.assertRaises(ValueError):
            layout.tokens_per_chunk(256, 512)

    def test_chunk_capacity_from_fdm_budget(self):
        layout = self._small_layout()
        reserved_bytes = 4096
        fdm_budget_bytes = reserved_bytes + 3 * 2 * 2 * layout.token_stride

        self.assertEqual(
            layout.tokens_per_chunk_from_budget(
                fdm_budget_bytes,
                buffer_count=2,
                reserved_bytes=reserved_bytes,
            ),
            3,
        )
        with self.assertRaises(ValueError):
            layout.tokens_per_chunk_from_budget(4096, reserved_bytes=4096)

    def test_iter_chunks(self):
        layout = self._small_layout()
        chunks = list(
            layout.iter_chunks(
                layer=1,
                valid_tokens=8,
                k_staging_bytes=4 * 512,
                v_staging_bytes=3 * 512,
            )
        )

        self.assertEqual([chunk.start_token for chunk in chunks], [0, 3, 6])
        self.assertEqual([chunk.token_count for chunk in chunks], [3, 3, 2])
        self.assertEqual([chunk.end_token for chunk in chunks], [3, 6, 8])
        self.assertEqual(
            [chunk.nvm_offset for chunk in chunks],
            [
                layout.token_offset(1, 0),
                layout.token_offset(1, 3),
                layout.token_offset(1, 6),
            ],
        )
        self.assertEqual(
            [chunk.copy_size for chunk in chunks],
            [3 * 512, 3 * 512, 2 * 512],
        )
        self.assertEqual(layout.chunk_count(8, 3), 3)
        self.assertEqual(layout.chunk_count(0, 3), 0)

    def test_invalid_coordinates(self):
        layout = self._small_layout()

        with self.assertRaises(IndexError):
            layout.token_offset(2, 0)
        with self.assertRaises(IndexError):
            layout.token_offset(0, 10)
        with self.assertRaises(IndexError):
            layout.head_offset(0, 0, 0, 3)
        with self.assertRaises(IndexError):
            layout.element_offset(0, 0, 0, 0, 80)
        with self.assertRaises(ValueError):
            list(layout.iter_chunks(0, 11, 512, 512))

    def test_align_up_and_config_validation(self):
        self.assertEqual(align_up(0, 512), 0)
        self.assertEqual(align_up(513, 512), 1024)
        with self.assertRaises(ValueError):
            align_up(-1, 512)
        with self.assertRaises(ValueError):
            KvLayoutConfig(0, 10, 1, 1, 1)

    @staticmethod
    def _small_layout():
        return KvCacheLayout(
            KvLayoutConfig(
                num_layers=2,
                max_seq_len=10,
                batch_size=1,
                num_kv_heads=3,
                head_dim=80,
                dtype=np.float16,
            )
        )


if __name__ == "__main__":
    unittest.main()
