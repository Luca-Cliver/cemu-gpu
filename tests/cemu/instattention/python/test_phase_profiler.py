#!/usr/bin/env python3

import csv
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "python"))

from phase_profiler import PhaseProfiler, profile_scope, profile_tensor_scope


class PhaseProfilerTest(unittest.TestCase):
    def test_cuda_scope_records_on_current_stream_without_phase_sync(self):
        class Event:
            def __init__(self):
                self.stream = None
                self.waited = False

            def record(self, stream):
                self.stream = stream

            def query(self):
                return False

            def synchronize(self):
                self.waited = True

            def elapsed_time(self, end):
                return 1.25

        profiler = PhaseProfiler()
        start, end = Event(), Event()
        stream = object()
        with patch("torch.cuda.Event", side_effect=[start, end]), patch(
            "torch.cuda.current_stream", return_value=stream
        ):
            with profile_tensor_scope(profiler, "qkv", "cuda:0"):
                self.assertIs(start.stream, stream)
                self.assertIsNone(end.stream)
        self.assertIs(end.stream, stream)
        self.assertFalse(end.waited)
        metrics = {metric.name: metric for metric in profiler.snapshot()}
        self.assertEqual(metrics["qkv.cuda_stream"].total_ns, 1_250_000)
        self.assertEqual(metrics["qkv.host_wall"].count, 1)
        self.assertTrue(end.waited)

    def test_cpu_tensor_scope_and_disabled_scope(self):
        profiler = PhaseProfiler()
        with profile_tensor_scope(profiler, "qkv", "cpu"):
            pass
        self.assertEqual([m.name for m in profiler.snapshot()], ["qkv.host_wall"])
        with profile_tensor_scope(None, "unused", "not-a-device"):
            pass

    def test_deferred_cuda_does_not_wait_until_summary(self):
        class Event:
            ready = False
            waited = False
            def query(self): return self.ready
            def synchronize(self): self.waited = True
            def elapsed_time(self, end): return 2.5
        profiler = PhaseProfiler()
        start, end = Event(), Event()
        profiler.defer_cuda("qkv.cuda_stream", start, end)
        self.assertFalse(end.waited)
        metric, = profiler.snapshot()
        self.assertTrue(end.waited)
        self.assertEqual(metric.total_ns, 2500000)
        self.assertEqual(profiler.snapshot()[0].count, 1)

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
