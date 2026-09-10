#!/usr/bin/env python3

import csv
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "python"))

from phase_profiler import PhaseProfiler, profile_scope


class PhaseProfilerTest(unittest.TestCase):
    def test_aggregates_time_and_bytes(self):
        profiler = PhaseProfiler()
        profiler.record("nvm_to_fdm", 2_000_000_000, 1024**3)
        profiler.record("nvm_to_fdm", 1_000_000_000, 2 * 1024**3)

        metric = profiler.snapshot()[0]
        self.assertEqual(metric.count, 2)
        self.assertEqual(metric.total_ns, 3_000_000_000)
        self.assertEqual(metric.maximum_ns, 2_000_000_000)
        self.assertEqual(metric.byte_count, 3 * 1024**3)
        self.assertAlmostEqual(metric.gib_per_second, 1.0)

    def test_disabled_scope_is_a_no_op(self):
        with profile_scope(None, "unused", 512):
            pass

    def test_writes_csv(self):
        profiler = PhaseProfiler()
        profiler.record("execute", 1000)
        with tempfile.TemporaryDirectory() as directory:
            path = profiler.write_csv(Path(directory) / "profile.csv")
            with path.open(newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
        self.assertEqual(rows[0]["phase"], "execute")
        self.assertEqual(rows[0]["count"], "1")


if __name__ == "__main__":
    unittest.main()
