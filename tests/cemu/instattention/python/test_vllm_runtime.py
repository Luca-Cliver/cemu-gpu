import json
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import Mock

import numpy as np

from cemu_flexgen.cemu_sparf_device import CemuSparfDevice
from cemu_flexgen.fdm_kvbuffer_store import FDM_KVbuffer
from cemu_flexgen.kv_layout import KvLayoutConfig, SparfKvCacheLayout
from cemu_flexgen.sparf_group_layout import SparfGroupLayout
from experiments.config import load_experiment_config
from experiments.runtime_model import SparfAttentionRuntimeModel
from experiments.softmax_ratio import IMPLEMENTATION, SoftmaxRatioTable
from vllm_adapter.session import Config, load_runtime


class VllmRuntimeTest(unittest.TestCase):
    def setUp(self):
        self.config_path = Path(__file__).resolve().parents[1] / "experiments/configs/opt13b_sparf_1csd.json"
        self.ratios = dict(schema_version=1, implementation=IMPLEMENTATION,
            anchor=dict(vectors=1, tokens=16, median_us=2.0), measurements=[
                dict(vectors=1, tokens=16, median_us=2.0),
                dict(vectors=320, tokens=1025, median_us=12.0),
                dict(vectors=320, tokens=129, median_us=5.0)])
        self.model = SparfAttentionRuntimeModel(load_experiment_config(self.config_path).instcsd,
                                               SoftmaxRatioTable(self.ratios))
        self.layout = SparfKvCacheLayout(KvLayoutConfig(1, 1041, 8, 40, 128, np.float16))
        self.store = FDM_KVbuffer(SparfGroupLayout(128, 8), 8, 1, 40, 1041, 1,
                                  "unused", 1024**2, 64 * 1024**2)
        self.store.registered = True
        self.device = CemuSparfDevice(self.layout, "unused", "unused", "unused", "unused", "unused",
            40, 16, autonomous=True, grouped_store=self.store, ragged=True, runtime_model=self.model,
            read_trace=False)
        self.device._device = Mock()

    def prepare(self, rows):
        shape = (len(rows), 40, 128)
        return self.device._prepare_batch(np.zeros(shape, np.float16), 0,
                                          np.zeros(shape, np.float16), rows)

    def test_ratio_budget_matches_existing_uniform_model_and_reordered_slots(self):
        rows = [(slot, 1025, 129) for slot in reversed(range(8))]
        request = self.prepare(rows)
        expected = self.model.estimate(8, 40, 128, 1025, 16, 129, 2)
        self.assertEqual(struct.unpack_from("<3I", request.metadata, 8),
                         (expected.approximate_ns, expected.exact_qk_ns, expected.pv_ns))
        self.assertTrue(all(struct.unpack_from("<3I", request.metadata, 8)))
        self.assertEqual(self.device._device.write_tensor.call_count, 2)
        self.assertEqual(struct.unpack_from("<5I", request.metadata, 136)[:3], rows[0])

    def test_unmeasured_batch_or_length_and_ragged_not_silently_zero(self):
        for rows in ([(0, 1025, 129)], [(slot, 1025, 129) for slot in range(2)],
                     [(slot, 1026, 129) for slot in range(8)],
                     [(slot, 1025 + slot, 129) for slot in range(8)]):
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                self.prepare(rows)
        self.device._device.write_tensor.assert_not_called()

    def test_functional_ragged_mode_still_available_without_model(self):
        self.device.runtime_model = None
        request = self.prepare([(0, 17, 3), (2, 33, 5)])
        self.assertEqual(struct.unpack_from("<3I", request.metadata, 8), (0, 0, 0))

    def test_runtime_pair_and_anchor_load(self):
        self.assertIsNone(load_runtime(Config("unused", "unused", "unused")))
        with self.assertRaises(ValueError):
            Config("unused", "unused", "unused", runtime_config=str(self.config_path))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ratios.json"
            path.write_text(json.dumps(self.ratios))
            config = Config("unused", "unused", "unused", runtime_config=str(self.config_path),
                            softmax_ratio_file=str(path))
            self.assertEqual(load_runtime(config).estimate(8, 40, 128, 1025, 16, 129),
                             self.model.estimate(8, 40, 128, 1025, 16, 129))

    def test_budget_bounds_and_trace_flag(self):
        for runtime in (-1, 2**32, True, 0.5):
            with self.assertRaises(ValueError):
                self.store.pack_batch_workflow(self.layout, 0, 40, 16, [(0, 17, 3)],
                                               runtimes=(runtime, 0, 0))
        self.device.read_trace = True
        request = self.prepare([(slot, 1025, 129) for slot in range(8)])
        self.assertEqual(struct.unpack_from("<I", request.metadata, 4)[0] & 1, 1)


if __name__ == "__main__":
    unittest.main()
