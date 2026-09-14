#!/usr/bin/env python3

import copy
from dataclasses import replace
import json
from pathlib import Path
import sys
import tempfile
import unittest


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from experiments.config import load_experiment_config
from experiments.runtime_model import DenseAttentionRuntimeModel, SparfAttentionRuntimeModel
from experiments.softmax_ratio import IMPLEMENTATION, SoftmaxRatioTable, sparf_softmax_shapes


class SoftmaxRatioTest(unittest.TestCase):
    def setUp(self):
        self.device = load_experiment_config(
            PROJECT_DIR / "experiments/configs/opt13b_sparf_1csd.json"
        ).instcsd
        self.report = dict(
            schema_version=1,
            implementation=IMPLEMENTATION,
            anchor=dict(vectors=1, tokens=16, median_us=2.0),
            measurements=[
                dict(vectors=1, tokens=16, median_us=2.0),
                dict(vectors=160, tokens=1025, median_us=10.0),
                dict(vectors=160, tokens=129, median_us=4.0),
            ],
        )

    def test_shapes_cover_every_decode_step_and_both_softmaxes(self):
        shapes = sparf_softmax_shapes([4], 40, 1024, 16, 8)
        self.assertEqual(len(shapes), 18)
        self.assertIn((160, 1025), shapes)
        self.assertIn((160, 1040), shapes)
        self.assertIn((160, 129), shapes)
        self.assertIn((160, 130), shapes)
        self.assertNotIn((160, 1024), shapes)

    def test_shapes_use_microbatch_size_and_deduplicate(self):
        shapes = sparf_softmax_shapes([1, 4, 4], 40, 16, 1, 1)
        self.assertEqual(shapes, [(40, 17), (160, 17)])

    def test_invalid_workload(self):
        for arguments in (([], 40, 8, 1, 8), ([0], 40, 8, 1, 8),
                          ([1], 0, 8, 1, 8), ([1], 40, 8, 0, 8),
                          ([1], 40, 8, 1, 0)):
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                sparf_softmax_shapes(*arguments)

    def test_ratio_is_for_whole_shape_without_another_head_multiplier(self):
        table = SoftmaxRatioTable(self.report)
        self.assertEqual(table.estimate_ns(self.device, 1, 16), 164000)
        self.assertEqual(table.estimate_ns(self.device, 160, 1025), 820000)
        self.assertEqual(table.estimate_ns(self.device, 160, 129), 328000)

    def test_sparf_changes_only_the_two_softmax_terms(self):
        legacy = SparfAttentionRuntimeModel(self.device).estimate(4, 40, 128, 1025, 16, 129)
        measured = SparfAttentionRuntimeModel(
            self.device, SoftmaxRatioTable(self.report),
        ).estimate(4, 40, 128, 1025, 16, 129)
        self.assertEqual(measured.approximate_ns, legacy.approximate_ns - 1681000000 + 820000)
        self.assertEqual(measured.exact_qk_ns, legacy.exact_qk_ns - 211560000 + 328000)
        self.assertEqual(measured.pv_ns, legacy.pv_ns)
        self.assertEqual(measured.pv_ns, 419276)

    def test_dense_optional_ratio_keeps_gemv(self):
        legacy = DenseAttentionRuntimeModel(self.device).estimate(4, 40, 128, 1025)
        measured = DenseAttentionRuntimeModel(
            self.device, SoftmaxRatioTable(self.report),
        ).estimate(4, 40, 128, 1025)
        self.assertEqual(measured.softmax_ns, 820000)
        self.assertEqual(measured.qk_ns, legacy.qk_ns)
        self.assertEqual(measured.av_ns, legacy.av_ns)

    def test_legacy_model_stays_available(self):
        result = SparfAttentionRuntimeModel(self.device).estimate(4, 40, 128, 1025, 16, 129)
        self.assertEqual(result.approximate_ns, 1681612667)
        self.assertEqual(result.exact_qk_ns, 211976051)

    def test_unmeasured_shape_is_not_interpolated_or_scaled(self):
        table = SoftmaxRatioTable(self.report)
        table.require_sparf(4, 40, 1024, 1, 8)
        for arguments in ((4, 40, 1024, 2, 8), (1, 40, 1024, 1, 8)):
            with self.subTest(arguments=arguments), self.assertRaisesRegex(ValueError, "unmeasured"):
                table.require_sparf(*arguments)

    def test_anchor_must_match_runtime_config(self):
        with self.assertRaisesRegex(ValueError, "anchor shape"):
            SparfAttentionRuntimeModel(
                replace(self.device, softmax_anchor_tokens=32), SoftmaxRatioTable(self.report),
            )

    def test_rejects_invalid_measurements(self):
        for value in (0, -1, float("nan"), float("inf"), True, "2"):
            report = copy.deepcopy(self.report)
            report["measurements"][1]["median_us"] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                SoftmaxRatioTable(report)

    def test_rejects_duplicate_shape_and_inconsistent_anchor(self):
        duplicate = copy.deepcopy(self.report)
        duplicate["measurements"].append(duplicate["measurements"][0])
        with self.assertRaisesRegex(ValueError, "duplicate"):
            SoftmaxRatioTable(duplicate)
        inconsistent = copy.deepcopy(self.report)
        inconsistent["anchor"]["median_us"] = 3.0
        with self.assertRaisesRegex(ValueError, "anchor measurement"):
            SoftmaxRatioTable(inconsistent)

    def test_rejects_unknown_implementation(self):
        report = copy.deepcopy(self.report)
        report["implementation"] = "unknown"
        with self.assertRaisesRegex(ValueError, "implementation"):
            SoftmaxRatioTable(report)

    def test_report_roundtrip(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ratios.json"
            path.write_text(json.dumps(self.report), encoding="utf-8")
            table = SoftmaxRatioTable.load(path)
            self.assertEqual(table.ratio(160, 1025), 5.0)


if __name__ == "__main__":
    unittest.main()
