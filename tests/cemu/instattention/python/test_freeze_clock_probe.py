import ctypes
import importlib.util
import contextlib
import io
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch


ROOT = Path(__file__).resolve().parents[4]
SPEC = importlib.util.spec_from_file_location(
    "freeze_clock_probe", Path(__file__).resolve().parents[1] / "experiments/freeze_clock_probe.py")
PROBE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PROBE)


class ProbeArgs(ctypes.Structure):
    _pack_ = 1
    _fields_ = [("numr", ctypes.c_int), ("mr_addr", ctypes.POINTER(ctypes.c_void_p)),
                ("mr_dev_addr", ctypes.POINTER(ctypes.c_void_p)),
                ("mr_len", ctypes.POINTER(ctypes.c_longlong)),
                ("cparam1", ctypes.c_longlong), ("cparam2", ctypes.c_longlong),
                ("data_buffer", ctypes.c_void_p), ("buffer_len", ctypes.c_longlong)]


class FreezeClockProbeTest(unittest.TestCase):
    def fixture(self, leakage=1, frozen=0, virtual_growth=0):
        rows, events = [], []
        for probe_id, delay_ms in enumerate(PROBE.DELAYS_MS, 1):
            delay_ns = delay_ms * 1_000_000
            rows.append(dict(probe_id=probe_id, delay_ms=delay_ms, repetition=0,
                             guest_execute_ns=100_000_000 + leakage * delay_ns,
                             host_operator_ns=delay_ns))
            identity = dict(pid=123, controller=0, program=1, probe_id=probe_id)
            events.append(dict(identity, event="functional", host_start_ns=100,
                               host_end_ns=100 + delay_ns, virtual_start_ns=100,
                               virtual_end_ns=100 + frozen, requested_runtime_ns=100_000_000,
                               effective_runtime_ns=100_000_000, freeze_entry_ns=0))
            events.append(dict(identity, event="completion", received_virtual_ns=100,
                               start_virtual_ns=100,
                               finish_virtual_ns=100 + 100_000_000 + virtual_growth * delay_ns,
                               scheduled_runtime_ns=100_000_000))
        return dict(rows=rows, program=1, runtime_ns=100_000_000), events

    def analyze(self, report, events):
        return PROBE.analyze(report, ["prefix " + PROBE.MARKER + json.dumps(event) for event in events])

    def test_leaking_and_excluding_guest_clocks(self):
        for leakage, verdict in ((1, "guest_clock_includes_functional_freeze"),
                                 (0, "guest_clock_excludes_functional_freeze"),
                                 (0.5, "inconclusive")):
            report, events = self.fixture(leakage=leakage)
            self.assertEqual(self.analyze(report, events)["verdict"], verdict)

    def test_virtual_clock_failure_and_queue_noise(self):
        report, events = self.fixture(frozen=10_000)
        self.assertEqual(self.analyze(report, events)["verdict"], "virtual_clock_advanced_inside_freeze")
        report, events = self.fixture(virtual_growth=1)
        self.assertEqual(self.analyze(report, events)["verdict"], "inconclusive")

    def test_missing_duplicate_and_mismatched_samples(self):
        report, events = self.fixture()
        with self.assertRaisesRegex(ValueError, "missing Host"):
            self.analyze(report, events[:-1])
        with self.assertRaisesRegex(ValueError, "duplicate Host"):
            self.analyze(report, events + events[:1])
        events[1]["pid"] += 1
        with self.assertRaisesRegex(ValueError, "mismatch"):
            self.analyze(report, events)

    def test_runtime_must_stay_fixed(self):
        report, events = self.fixture()
        events[0]["requested_runtime_ns"] += 1
        with self.assertRaisesRegex(ValueError, "runtime changed"):
            self.analyze(report, events)

    def test_zero_or_mismatched_budget_is_inconclusive(self):
        for effective, scheduled in ((0, 0), (100_000_000, 1)):
            report, events = self.fixture()
            events[0]["effective_runtime_ns"] = effective
            events[1]["scheduled_runtime_ns"] = scheduled
            result = self.analyze(report, events)
            self.assertFalse(result["budgets_valid"])
            self.assertEqual(result["verdict"], "inconclusive")

    def test_same_execute_is_bracketed_by_all_clocks(self):
        execute = Mock(return_value=7)
        snapshot = Mock(side_effect=[(10, 20, 30), (110, 220, 330)])
        with patch.object(PROBE.time, "perf_counter_ns", side_effect=[0, 400]):
            status, measurements = PROBE.measure_execute(execute, snapshot)
        execute.assert_called_once_with()
        self.assertEqual(snapshot.call_count, 2)
        self.assertEqual(status, 7)
        self.assertEqual(measurements, dict(guest_execute_ns=400, cpp_steady_ns=100,
                                           cpp_high_resolution_ns=200, cpp_system_ns=300))

    def test_cpp_clock_growth_and_legacy_report(self):
        report, events = self.fixture()
        self.assertEqual(self.analyze(report, events)["cpp_growth_per_host_growth"], {})
        for row in report["rows"]:
            row.update(cpp_steady_ns=row["guest_execute_ns"],
                       cpp_high_resolution_ns=100_000_000, cpp_system_ns=100_000_000)
        result = self.analyze(report, events)
        self.assertEqual(result["cpp_growth_per_host_growth"], dict(
            cpp_steady_ns=1, cpp_high_resolution_ns=0, cpp_system_ns=0))
        self.assertEqual(result["summaries"][-1]["cpp_steady_ns"], 1_100_000_000)

    def test_host_operator_and_invalid_arguments(self):
        with tempfile.TemporaryDirectory() as directory:
            library = Path(directory) / "probe.so"
            subprocess.run(["cc", "-Wall", "-Wextra", "-Werror", "-std=c11", "-shared", "-fPIC",
                            str(ROOT / "tests/cemu/kernel/freeze_clock_probe.c"), "-o", str(library)], check=True)
            kernel = ctypes.CDLL(str(library)).freeze_clock_probe
            kernel.argtypes = [ctypes.POINTER(ProbeArgs)]
            kernel.restype = ctypes.c_longlong
            output = (ctypes.c_uint64 * 512)()
            pointers = (ctypes.c_void_p * 1)(ctypes.addressof(output))
            lengths = (ctypes.c_longlong * 1)(4096)
            args = ProbeArgs(numr=1, mr_addr=pointers, mr_len=lengths, cparam2=42)
            for delay in (0, 20_000_000):
                args.cparam1 = delay
                self.assertEqual(kernel(ctypes.byref(args)), 0)
                self.assertEqual(list(output[:3]), [PROBE.MAGIC, 42, delay])
                self.assertGreaterEqual(output[4] - output[3], delay)
            args.cparam1 = -1
            self.assertEqual(kernel(ctypes.byref(args)), -1)
            args.cparam1 = 2_000_000_001
            self.assertEqual(kernel(ctypes.byref(args)), -1)
            args.cparam1 = 0
            lengths[0] = 8
            self.assertEqual(kernel(ctypes.byref(args)), -1)
            self.assertEqual(kernel(None), -1)

    def test_probe_reads_bytes_without_numpy(self):
        client = Mock(program_id=1)
        client.read_range.side_effect = AssertionError("NumPy read path must not be used")

        def execute(*, cparam1, cparam2, runtime):
            self.assertEqual(runtime, 100_000_000)
            client.read_range_bytes.return_value = struct.pack(
                "<5Q", PROBE.MAGIC, cparam2, cparam1, 100, 100 + cparam1).ljust(4096, b"\0")
            return 0

        client.execute.side_effect = execute
        context = contextlib.nullcontext(client)
        bindings = SimpleNamespace(
            CemuClient=Mock(return_value=context), MemoryRange=Mock(),
            ProgramTarget=SimpleNamespace(HOST=0), guest_clock_info=lambda: {},
            guest_clock_snapshot=lambda: (0, 0, 0))
        support = SimpleNamespace(require_fdmfs=Mock())
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "result.json"
            args = SimpleNamespace(fdm_dir=directory, runtime_ms=100, repetitions=3,
                                   output=str(output), control="unused", namespace="unused",
                                   program="probe.so")
            with patch.dict(sys.modules, {"_cemu_client": bindings, "device_test_support": support,
                                          "numpy": None}), patch.object(sys, "path", sys.path.copy()), \
                    patch.object(Path, "read_text", return_value="test-environment"), \
                    contextlib.redirect_stdout(io.StringIO()):
                PROBE.run_probe(args)
            report = json.loads(output.read_text())
        self.assertEqual(len(report["rows"]), 9)
        self.assertEqual(client.execute.call_count, 10)
        self.assertEqual(client.read_range_bytes.call_count, 9)
        for call in client.read_range_bytes.call_args_list:
            self.assertEqual(call.args, (0, 4096))
        client.read_range.assert_not_called()


if __name__ == "__main__":
    unittest.main()
