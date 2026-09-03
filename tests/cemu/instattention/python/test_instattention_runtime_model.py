#!/usr/bin/env python3

import json
import struct
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))
sys.path.insert(0, str(PROJECT_DIR / "python"))

from cemu_flexgen import DenseAttentionMetadata, KvCacheLayout, KvLayoutConfig
from experiments import DenseAttentionRuntimeModel, load_experiment_config


CONFIG_PATH = (
    PROJECT_DIR / "experiments" / "configs" / "opt13b_dense_1csd.json"
)


class InstAttentionRuntimeModelTest(unittest.TestCase):
    def setUp(self):
        self.config = load_experiment_config(CONFIG_PATH)
        self.model = DenseAttentionRuntimeModel(self.config.instcsd)

    def test_opt13b_paper_configuration(self):
        model = self.config.model
        self.assertEqual(model.name, "opt-13b")
        self.assertEqual(model.num_layers, 40)
        self.assertEqual(model.hidden_size, 5120)
        self.assertEqual(model.num_query_heads, 40)
        self.assertEqual(model.num_kv_heads, 40)
        self.assertEqual(model.head_dim, 128)
        self.assertEqual(model.dtype, "float16")
        self.assertEqual(self.config.workload.batch_sizes, (4, 8, 16, 32, 64, 128, 256))

    def test_table1_single_head_anchor(self):
        result = self.model.estimate(1, 1, 128, 16)
        self.assertEqual(result.qk_ns, 323)
        self.assertEqual(result.softmax_ns, 164000)
        self.assertEqual(result.av_ns, 323)
        self.assertEqual(result.total_ns, 164646)

    def test_runtime_scales_with_attention_elements(self):
        base = self.model.estimate(1, 1, 128, 16)
        scaled = self.model.estimate(2, 4, 128, 32)
        self.assertEqual(scaled.qk_ns, 5161)
        self.assertEqual(scaled.softmax_ns, base.softmax_ns * 16)
        self.assertEqual(scaled.av_ns, 5161)

    def test_filter_bandwidth_model(self):
        self.assertEqual(self.model.estimate_filter_ns(0), 0)
        self.assertEqual(self.model.estimate_filter_ns(1850), 1000)

    def test_float16_attention_metadata(self):
        layout = KvCacheLayout(
            KvLayoutConfig(
                num_layers=1,
                max_seq_len=16,
                batch_size=1,
                num_kv_heads=1,
                head_dim=128,
                dtype=np.float16,
            )
        )
        metadata = DenseAttentionMetadata.from_layout(layout, 1, 16)
        packed = struct.unpack("<8IfI", metadata.pack())
        self.assertEqual(metadata.element_size, 2)
        self.assertEqual(packed[1], 2)

    def test_float32_attention_metadata_remains_compatible(self):
        layout = KvCacheLayout(
            KvLayoutConfig(
                num_layers=1,
                max_seq_len=16,
                batch_size=1,
                num_kv_heads=1,
                head_dim=128,
                dtype=np.float32,
            )
        )
        metadata = DenseAttentionMetadata.from_layout(layout, 1, 16)
        packed = struct.unpack("<8IfI", metadata.pack())
        self.assertEqual(metadata.element_size, 4)
        self.assertEqual(packed[1], 1)

    def test_rejects_inconsistent_model_shape(self):
        with CONFIG_PATH.open("r", encoding="utf-8") as config_file:
            raw = json.load(config_file)
        raw["model"]["hidden_size"] = 4096
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "invalid.json"
            path.write_text(json.dumps(raw), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "hidden_size"):
                load_experiment_config(path)


if __name__ == "__main__":
    unittest.main()
