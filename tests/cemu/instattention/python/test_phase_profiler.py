#!/usr/bin/env python3

import csv
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch
from concurrent.futures import ThreadPoolExecutor
from threading import Event, Lock


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "python"))

from phase_profiler import (PhaseProfiler, profile_client_call, profile_scope, profile_tensor_scope,
                            submit_profiled, profile_request, request_context, profile_lock)


class PhaseProfilerTest(unittest.TestCase):
    def test_metadata_is_first_sample_only_and_written_with_csv_and_intervals(self):
        with tempfile.TemporaryDirectory() as directory:
            intervals = Path(directory) / "intervals.json"
            with patch.dict(os.environ, CEMU_PROFILE_INTERVALS=str(intervals)):
                profiler = PhaseProfiler(stack=True)
            profiler.record_metadata_once("prefill.layout", {"shape": [32, 40, 128]})
            profiler.record_metadata_once("prefill.layout", {"shape": [1, 40, 128]})
            expected = {"prefill.layout": {"shape": [32, 40, 128]}}
            self.assertEqual(profiler.metadata_snapshot(), expected)
            self.assertEqual(profiler.snapshot(), ())
            output = profiler.write_csv(Path(directory) / "profile.csv")
            self.assertEqual(json.loads(Path(str(output) + ".metadata.json").read_text()), expected)
            self.assertEqual(json.loads(intervals.read_text())["metadata"], expected)

    def test_request_context_crosses_pools_and_restores_after_failure(self):
        with patch.dict(os.environ, {"CEMU_PROFILE_INTERVALS": "/tmp/unused-request-intervals"}):
            profiler = PhaseProfiler(stack=True)
        first = dict(id="first", layer=0, microbatch=0, tokens=10)
        second = dict(id="second", layer=0, microbatch=1, tokens=10)
        with ThreadPoolExecutor(max_workers=1) as outer, ThreadPoolExecutor(max_workers=1) as inner:
            def work():
                self.assertEqual(request_context(profiler), first)
                nested = submit_profiled(inner.submit, lambda: request_context(profiler), profiler, "inner")
                self.assertEqual(nested.result(timeout=5), first)
                with profile_request(profiler, second):
                    raise ValueError("restore context")

            with profile_request(profiler, first):
                future = submit_profiled(outer.submit, work, profiler, "outer")
            with self.assertRaisesRegex(ValueError, "restore context"):
                future.result(timeout=5)
            self.assertIsNone(request_context(profiler))
            self.assertIsNone(outer.submit(request_context, profiler).result(timeout=5))
            self.assertIsNone(inner.submit(request_context, profiler).result(timeout=5))
        self.assertEqual(len(profiler._intervals), 4)
        self.assertTrue(all(entry["request"] == first for entry in profiler._intervals))
        self.assertEqual(sum(entry["name"].endswith("queue_wait_wall") for entry in profiler._intervals), 2)

    def test_profile_lock_releases_on_exception_and_traces_acquisition_only(self):
        with patch.dict(os.environ, {"CEMU_PROFILE_INTERVALS": "/tmp/unused-request-intervals"}):
            profiler = PhaseProfiler(stack=True)
        lock = Lock()
        with patch("phase_profiler.time.perf_counter_ns", side_effect=[100, 150]):
            with self.assertRaisesRegex(ValueError, "body"):
                with profile_lock(profiler, "decode.store", lock):
                    self.assertTrue(lock.locked())
                    raise ValueError("body")
        self.assertFalse(lock.locked())
        self.assertEqual(profiler._intervals[0]["end_ns"], 150)
        self.assertEqual(profiler._intervals[0]["name"], "decode.store.lock_wait_wall")
        with profile_lock(None, "disabled", lock):
            self.assertTrue(lock.locked())
        self.assertFalse(lock.locked())

    def test_intervals_are_opt_in_buffered_and_use_existing_clock_reads(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "intervals.json"
            with patch.dict(os.environ, {"CEMU_PROFILE_INTERVALS": str(path)}):
                profiler = PhaseProfiler(stack=True)
            with patch("phase_profiler.time.perf_counter_ns", side_effect=[100, 180]) as clock:
                with profiler.measure("prefill.store_append_wall"):
                    self.assertFalse(path.exists())
            self.assertEqual(clock.call_count, 2)
            self.assertFalse(path.exists())
            profiler.record("duration_only", 50)
            profiler.trace_interval("framework.prefill_iteration_wall", 90, 200)
            profiler.write_csv(Path(directory) / "profile.csv")
            trace = json.loads(path.read_text())
            self.assertEqual(trace["dropped"], 0)
            self.assertEqual(len(trace["intervals"]), 2)
            self.assertEqual(trace["intervals"][0]["start_ns"], 100)
            self.assertEqual(trace["intervals"][0]["end_ns"], 180)
            self.assertTrue(trace["stack"])

    def test_interval_cap_and_disabled_path(self):
        with patch.dict(os.environ, {"CEMU_PROFILE_INTERVALS": "", "CEMU_PROFILE_INTERVAL_LIMIT": "invalid"}):
            profiler = PhaseProfiler()
        profiler.trace_interval("unused", 20, 10)
        self.assertEqual(profiler._intervals, [])
        with patch.dict(os.environ, {"CEMU_PROFILE_INTERVALS": "/tmp/unused-intervals", "CEMU_PROFILE_INTERVAL_LIMIT": "1"}):
            profiler = PhaseProfiler()
        profiler.trace_interval("first", 0, 1)
        profiler.trace_interval("second", 1, 2)
        self.assertEqual(len(profiler._intervals), 1)
        self.assertEqual(profiler._intervals_dropped, 1)
        with self.assertRaises(ValueError):
            profiler.trace_interval("invalid", 2, 1)

    def test_client_breakdown_and_disabled_path(self):
        timings = dict(binding_prepare_wall=10, buffer_prepare_wall=20, io_wait_wall=30,
                       buffer_finish_wall=40, binding_finish_wall=50, io_bytes=512)
        client = Mock(return_value=(17, timings))
        profiler = PhaseProfiler(stack=True)
        self.assertEqual(profile_client_call(profiler, "write", client, 1, b"abc", 0), 17)
        client.assert_called_once_with(1, b"abc", 0, profile=True)
        metrics = {metric.name: metric for metric in profiler.snapshot()}
        self.assertEqual(metrics["write.io_wait_wall"].total_ns, 30)
        self.assertEqual(metrics["write.io_wait_wall"].byte_count, 512)
        self.assertEqual(metrics["write.buffer_prepare_wall"].byte_count, 0)
        for disabled in (None, PhaseProfiler()):
            client.reset_mock()
            client.return_value = 19
            self.assertEqual(profile_client_call(disabled, "write", client, 1, b"abc"), 19)
            client.assert_called_once_with(1, b"abc")
        client.side_effect = OSError("failed write")
        with self.assertRaisesRegex(OSError, "failed write"):
            profile_client_call(profiler, "failed", client)
        metrics = {metric.name: metric for metric in profiler.snapshot()}
        self.assertEqual(metrics["failed.call_wall"].count, 1)
        self.assertNotIn("failed.io_wait_wall", metrics)

    def test_queue_timing_preserves_async_execution_and_failure(self):
        profiler = PhaseProfiler(stack=True)
        release = Event()
        with ThreadPoolExecutor(max_workers=1) as executor:
            blocker = executor.submit(release.wait)
            try:
                result = submit_profiled(executor.submit, lambda value: value + 1,
                                         profiler, "writer", 5)
                self.assertFalse(result.done())
            finally:
                release.set()
            blocker.result(timeout=5)
            self.assertEqual(result.result(timeout=5), 6)
            failure = submit_profiled(executor.submit, Mock(side_effect=ValueError("worker failed")),
                                      profiler, "failed")
            with self.assertRaisesRegex(ValueError, "worker failed"):
                failure.result(timeout=5)
        metrics = {metric.name: metric for metric in profiler.snapshot()}
        for phase in ("writer", "failed"):
            self.assertEqual(metrics[phase + ".queue_wait_wall"].count, 1)
            self.assertEqual(metrics[phase + ".worker_wall"].count, 1)

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
