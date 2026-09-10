#!/usr/bin/env python3

import unittest
from pathlib import Path
import sys


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from experiments import load_experiment_config
from experiments.run_figure14_functional import DEFAULT_CONFIG, estimate_capacity


class Figure14FunctionalTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.experiment = load_experiment_config(DEFAULT_CONFIG)

    def test_opt13b_capacity_matches_dense_kv_formula(self):
        capacity = estimate_capacity(
            self.experiment,
            batch_size=4,
            gpu_batch_size=4,
            output_length=1024,
            attention_slots=2,
        )

        self.assertEqual(capacity.gpu_batches, 1)
        self.assertEqual(capacity.sequence_capacity, 2047)
        self.assertEqual(capacity.nvm_bytes, 6_707_609_600)
        self.assertEqual(capacity.fdm_bytes, 335_711_232)
        self.assertEqual(capacity.decode_requests, 40_920)
        self.assertEqual(capacity.decode_staging_bytes, 5_145_575_424_000)

    def test_capacity_scales_with_total_batch_not_fdm_slots(self):
        small = estimate_capacity(
            self.experiment,
            batch_size=4,
            gpu_batch_size=4,
            output_length=1024,
            attention_slots=2,
        )
        large = estimate_capacity(
            self.experiment,
            batch_size=256,
            gpu_batch_size=4,
            output_length=1024,
            attention_slots=2,
        )

        self.assertEqual(large.nvm_bytes, small.nvm_bytes * 64)
        self.assertEqual(large.fdm_bytes, small.fdm_bytes)
        self.assertEqual(large.decode_requests, small.decode_requests * 64)
        self.assertEqual(
            large.decode_staging_bytes,
            small.decode_staging_bytes * 64,
        )

    def test_paper_output_length_runs_one_fewer_decode_iteration(self):
        capacity = estimate_capacity(
            self.experiment,
            batch_size=8,
            gpu_batch_size=4,
            output_length=self.experiment.workload.decode_length,
            attention_slots=2,
        )

        self.assertEqual(
            capacity.sequence_capacity,
            self.experiment.workload.prompt_length
            + self.experiment.workload.decode_length
            - 1,
        )


if __name__ == "__main__":
    unittest.main()
