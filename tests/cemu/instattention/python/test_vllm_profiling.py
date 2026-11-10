"""Timing boundaries and clock accounting; no vLLM/GPU/model download needed."""
from copy import deepcopy
import importlib.util
import os
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

from phase_profiler import PhaseProfiler
from vllm_adapter.profiling import batch_phase, device_report, timing_report, union_ns
from test_sparf_autonomous_timeline import sample
from vllm_adapter.guest_clock import configure_guest_clock, verify_guest_clock


class TimingReportTest(unittest.TestCase):
    def test_packing_metadata_is_preserved_without_affecting_step_totals(self):
        profiler = PhaseProfiler(stack=True)
        profiler.record("decode.step_wall", 100)
        profiler.record("decode.kv_payload_pack_wall", 60)
        for name in ("k_copy_wall", "v_copy_wall", "join_wall"):
            profiler.record("decode.kv_payload_pack." + name, 15)
        profiler.record_metadata_once("decode.kv_payload_layout", {"shape": [1, 40, 128]})
        report = timing_report(profiler)
        self.assertEqual(report["metadata"], profiler.metadata_snapshot())
        self.assertEqual(report["phases"]["decode"]["seconds"], 100 / 1e9)
        self.assertEqual(report["phases"]["decode"]["steps"], 1)

    def test_phase_classifies_one_token_prompt_and_mixed_batch(self):
        self.assertEqual(batch_phase([True]), "prefill")
        self.assertEqual(batch_phase([False, False]), "decode")
        self.assertEqual(batch_phase([False, True]), "mixed")
        self.assertEqual(batch_phase([]), "empty")

    def test_nested_metrics_are_not_added_to_denominator(self):
        p = PhaseProfiler()
        p.record("prefill.step_wall", 100)
        p.record("decode.step_wall", 300)
        p.record("decode.attention_host_wall", 250)
        p.record("decode.execute_wait_wall", 200)
        p.record("terminal_drain.wall", 900)
        report = timing_report(p)
        self.assertEqual(report["exec_wait_percent_of_worker_steps"], 50)
        self.assertEqual(report["phases"]["decode"]["steps"], 1)
        self.assertFalse(timing_report(None)["enabled"])

    def test_device_overlap_uses_union_and_separate_clocks(self):
        first = sample()
        second = deepcopy(first)
        for row in second:
            row["request"] = 2
        unrelated = deepcopy(first)
        for row in unrelated:
            row["program"] = 999
        report = device_report(first + second + unrelated, 3, 2)
        self.assertTrue(report["available"])
        self.assertEqual(report["device_command_union_seconds"], 510 / 1e9)
        self.assertEqual(report["host_diagnostics"]["command_union_seconds"], 1530 / 1e9)
        self.assertEqual(report["virtual_command_sum_seconds"], 1020 / 1e9)
        self.assertEqual(report["device_percent_of_decode_command_window"], 100)
        self.assertEqual(union_ns([(1, 10), (2, 5), (9, 15), (20, 21)]), 15)

    def test_missing_device_completion_not_reported_as_zero(self):
        self.assertFalse(device_report(sample()[:1], 3, 1)["available"])
        self.assertFalse(device_report(sample(), 3, 2)["available"])
        rows = sample()
        rows[1]["success"] = False
        with self.assertRaisesRegex(ValueError, "failed"):
            device_report(rows, 3, 1)

    def test_functional_duration_is_host_time_not_virtual_time(self):
        rows = sample()
        rows[1]["computes"][0]["functional_end"]["host_ns"] += 9
        report = device_report(rows, 3, 1)
        self.assertEqual(report["host_diagnostics"]["functional_union_seconds"], 9 / 1e9)
        self.assertEqual(report["host_diagnostics"]["functional_seconds_by_stage"]["approx"], 9 / 1e9)
        self.assertEqual(report["device_command_union_seconds"], 510 / 1e9)

    def test_host_freeze_growth_does_not_change_primary_timings(self):
        original = sample()
        inflated = deepcopy(original)
        visited = set()
        def grow(value):
            if isinstance(value, (dict, list)):
                if id(value) in visited:
                    return
                visited.add(id(value))
            if isinstance(value, dict):
                for key, child in value.items():
                    if key == "host_ns":
                        value[key] = 1_000_000 + (child - 1_000_000) * 100
                    else:
                        grow(child)
            elif isinstance(value, list):
                for child in value:
                    grow(child)
        grow(inflated)
        before = device_report(original, 3, 1)
        after = device_report(inflated, 3, 1)
        self.assertAlmostEqual(after.pop("host_diagnostics")["command_union_seconds"],
                               before.pop("host_diagnostics")["command_union_seconds"] * 100)
        self.assertEqual(before, after)

    def test_freeze_leak_and_truncated_cu_spans_rejected(self):
        for field, value, message in (("truncated", True, "truncated"),
                                       ("functional_end", {"host_ns": 1000361, "virtual_ns": 121}, "advanced")):
            rows = sample()
            rows[1]["computes"][0][field] = value
            with self.assertRaisesRegex(ValueError, message):
                device_report(rows, 3, 1)

    def test_cuda_metrics_do_not_enter_freeze_excluded_totals(self):
        p = PhaseProfiler(stack=True)
        p.record("decode.step_wall", 100)
        p.record("decode.h2d.cuda_stream", 10000)
        report = timing_report(p)
        self.assertNotIn("decode.h2d.cuda_stream", report["metrics"])
        self.assertIn("decode.h2d.cuda_stream", report["cuda_diagnostics"])
        self.assertEqual(report["phases"]["decode"]["seconds"], 100 / 1e9)


class GuestClockTest(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.cmdline = self.root / "cmdline"
        self.cmdline.write_text("BOOT_IMAGE=kernel cemu_native_opt_smoke=1")
        (self.root / "available_clocksource").write_text("kvm-clock tsc hpet")
        self.current = self.root / "current_clocksource"
        self.current.write_text("kvm-clock")

    def test_select_hpet_and_detect_later_change(self):
        report = configure_guest_clock(cmdline=self.cmdline, directory=self.root)
        self.assertEqual(report["before"], "kvm-clock")
        self.assertEqual(self.current.read_text().strip(), "hpet")
        verify_guest_clock(directory=self.root)
        self.current.write_text("kvm-clock")
        with self.assertRaisesRegex(RuntimeError, "discard"):
            verify_guest_clock(directory=self.root)

    def test_host_clock_never_changed(self):
        self.cmdline.write_text("BOOT_IMAGE=host-kernel")
        with self.assertRaisesRegex(RuntimeError, "Host"):
            configure_guest_clock(cmdline=self.cmdline, directory=self.root)
        self.assertEqual(self.current.read_text(), "kvm-clock")

    def test_no_silent_fallback_when_hpet_missing(self):
        (self.root / "available_clocksource").write_text("kvm-clock tsc")
        with self.assertRaisesRegex(RuntimeError, "unavailable"):
            configure_guest_clock(cmdline=self.cmdline, directory=self.root)
        self.assertEqual(self.current.read_text(), "kvm-clock")


class RunnerTimingTest(unittest.TestCase):
    def test_worker_intervals_include_actual_sample_boundary(self):
        with tempfile.TemporaryDirectory() as directory:
            profiler = self.runner.cemu_profiler
            profiler.interval_path = str(Path(directory) / "phase.json")
            profiler._interval_limit = 100
            self.runner.input_batch.num_computed_tokens_cpu = [1, 5]
            self.runner.execute_model()
            self.assertFalse(any(entry["name"] == "vllm.decode.step_wall" for entry in profiler._intervals))
            self.runner.sample_tokens()
            profiler.write_intervals()
            import json
            report = json.loads(Path(profiler.interval_path).read_text())
            window = next(entry for entry in report["intervals"] if entry["name"] == "vllm.decode.step_wall")
            sample_interval = next(entry for entry in report["intervals"] if entry["name"] == "decode.sample_host_wall")
            self.assertLessEqual(window["start_ns"], sample_interval["start_ns"])
            self.assertGreaterEqual(window["end_ns"], sample_interval["end_ns"])

    def setUp(self):
        # Load extension modules before patch.dict snapshots sys.modules;
        # removing Torch on context exit makes the next import unsafe.
        __import__("torch")
        class Runner:
            def __init__(self):
                self.input_batch = SimpleNamespace(req_ids=["a", "b"],
                    num_computed_tokens_cpu=[0, 0], num_prompt_tokens=[1, 5])

            def _build_attention_metadata(self, **kwargs):
                return "metadata"

            def execute_model(self, fail=False):
                self._build_attention_metadata(num_scheduled_tokens={"a": 1, "b": 5})
                if fail:
                    raise ValueError("model failed")
                return None

            def sample_tokens(self, fail=False):
                if fail:
                    raise ValueError("sampler failed")
                return "completed CPU output"

        runner_module = ModuleType("vllm.v1.worker.gpu_model_runner")
        runner_module.GPUModelRunner = Runner
        worker_module = ModuleType("vllm.v1.worker.gpu_worker")
        worker_module.Worker = object
        name = "vllm_adapter._worker_timing_test"
        spec = importlib.util.spec_from_file_location(name,
            Path(__file__).parent / "vllm_adapter/worker.py")
        module = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {runner_module.__name__: runner_module,
                                     worker_module.__name__: worker_module}):
            spec.loader.exec_module(module)
        with patch.dict(os.environ, CEMU_VLLM_PROFILE="basic"):
            self.runner = module.ProfiledModelRunner()

    def test_counts_after_sampling_excludes_warmup_and_handles_mixed(self):
        r = self.runner
        r._build_attention_metadata(num_scheduled_tokens=None)
        self.assertEqual(timing_report(r.cemu_profiler)["phases"]["prefill"]["steps"], 0)
        for starts, phase in (([0, 0], "prefill"), ([1, 5], "decode"), ([1, 0], "mixed")):
            r.input_batch.num_computed_tokens_cpu = starts
            r.execute_model()
            self.assertEqual(timing_report(r.cemu_profiler)["phases"][phase]["steps"], 0)
            self.assertEqual(r.sample_tokens(), "completed CPU output")
            self.assertEqual(timing_report(r.cemu_profiler)["phases"][phase]["steps"], 1)

    def test_failed_step_is_not_counted_as_completed(self):
        r = self.runner
        with self.assertRaisesRegex(ValueError, "model failed"):
            r.execute_model(fail=True)
        r.execute_model()
        with self.assertRaisesRegex(ValueError, "sampler failed"):
            r.sample_tokens(fail=True)
        self.assertIsNone(r._timing_started)
        self.assertEqual(timing_report(r.cemu_profiler)["phases"]["prefill"]["steps"], 0)


if __name__ == "__main__":
    unittest.main()
