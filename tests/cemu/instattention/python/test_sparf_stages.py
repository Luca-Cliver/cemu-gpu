#!/usr/bin/env python3
"""Compile and exercise the actual CEMU workflow with a deterministic fake FTL."""
import ctypes as C
import importlib.util
import json
from pathlib import Path
import shlex
import struct
import subprocess
import tempfile
import unittest

import numpy as np
from cemu_flexgen import KvLayoutConfig, SparfKvCacheLayout, SparfKvCacheStore, sparf_attention
from cemu_flexgen.sparf_workflow_abi import (
    pack_sparf_workflow, stage_metadata, stage_context_bytes, SparfWorkflowTrace,
)

ROOT = Path(__file__).resolve().parents[4]
ERROR = (1 << 64) - 1


class DeviceStagesTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix="sparf-stages-")
        cls.addClassCleanup(cls.temp.cleanup)
        folder = Path(cls.temp.name)
        kernel = folder / "kernel.so"
        harness = folder / "stages.so"
        subprocess.run(["c++", "-std=c++17", "-shared", "-fPIC", "-O2",
                        str(ROOT / "tests/cemu/kernel/sparf_attention.cc"), "-o", str(kernel)], check=True)
        glib = shlex.split(subprocess.check_output(
            ["pkg-config", "--cflags", "--libs", "glib-2.0"], text=True))
        subprocess.run(["cc", "-std=gnu11", "-shared", "-fPIC", "-O1", "-g",
                        "-I" + str(ROOT), "-I" + str(ROOT / "include"),
                        str(ROOT / "tests/cemu/instattention/guest/test_sparf_stages.c"),
                        "-o", str(harness), *glib, "-lm"], check=True)
        cls.kernel = C.CDLL(str(kernel))
        cls.lib = C.CDLL(str(harness))
        cls.lib.configure_kernel.argtypes = [C.c_void_p]
        cls.lib.configure_kernel(cls.kernel.sparf_attention)
        cls.lib.execute_stage.restype = C.c_uint64
        cls.lib.execute_stage.argtypes = [C.POINTER(C.c_void_p), C.POINTER(C.c_longlong),
            C.c_void_p, C.c_int, C.POINTER(C.c_void_p), C.POINTER(C.c_uint64), C.c_int, C.c_int]
        cls.lib.execute_autonomous.restype = C.c_uint64
        cls.lib.execute_autonomous.argtypes = cls.lib.execute_stage.argtypes
        cls.lib.configure_timing.argtypes = [C.c_int, C.c_uint64, C.c_uint64]
        cls.lib.set_virtual_time.argtypes = [C.c_uint64]
        cls.lib.configure_observation.argtypes = [C.c_int]
        cls.lib.configure_autonomous_trace.argtypes = [C.c_int]
        cls.lib.get_autonomous_trace.restype = C.c_char_p
        cls.lib.get_read_observation_json.restype = C.c_char_p
        cls.lib.observed_time.argtypes = [C.c_int, C.c_int]
        cls.lib.observed_time.restype = C.c_uint64
        for name in ("read_start", "read_finish"):
            function = getattr(cls.lib, name)
            function.argtypes = [C.c_int]
            function.restype = C.c_uint64

    def setUp(self):
        self.lib.configure_timing(0, 1000000, 0)
        self.lib.configure_observation(0)
        self.lib.configure_autonomous_trace(0)

    def test_autonomous_production_trace_serialization(self):
        path = ROOT / "tests/cemu/instattention/experiments/analyze_sparf_autonomous.py"
        spec = importlib.util.spec_from_file_location("autonomous_analysis", path)
        analyzer = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(analyzer)
        for dtype in (np.float16, np.float32):
            with self.subTest(dtype=dtype):
                self.lib.configure_autonomous_trace(1)
                self.lib.configure_timing(1, 1000000, 1000800)
                result, runtime = self.execute(self.make_case(49, dtype), autonomous=True)
                self.assertNotEqual(result, ERROR)
                records = [json.loads(line) for line in self.lib.get_autonomous_trace().decode().splitlines()]
                reports = analyzer.analyze(records)
                self.assertEqual(reports[0]["virtual_elapsed_ns"], runtime)
                self.assertEqual(reports[0]["cu_occupied_ns"], 370)
                self.assertEqual(reports[0]["qk_v_target_overlap_ns"], 250)

    def test_page_observation_separates_parallel_service_and_queueing(self):
        self.lib.check_read_observation()
        data = json.loads(self.lib.get_read_observation_json().decode())["io"]
        self.assertEqual(data["lun_pages"], [2, 1])
        self.assertEqual(data["lun_initial_wait_ns"], [100, 0])
        self.assertEqual(data["critical_initial_wait_ns"], 100)
        self.assertEqual(data["critical_queue_ns"], 150)
        self.assertEqual(data["critical_service_ns"], 50)
        self.assertEqual(data["service_ns"], 150)
        self.assertEqual(data["queue_ns"], 250)
        self.assertEqual(data["requested_bytes"], 640)
        self.assertEqual(data["submitted_bytes"], 12288)
        self.assertEqual(data["model_end_ns"] - data["model_start_ns"], 200)

    def test_autonomous_matches_legacy_and_dependency_deadlines(self):
        for dtype in (np.float16, np.float32):
            for busy_until in (0, 1000800):
                with self.subTest(dtype=dtype, busy_until=busy_until):
                    baseline = self.make_case(45, dtype)
                    candidate = self.make_case(45, dtype)
                    self.lib.configure_timing(1, 1000000, busy_until)
                    expected_result, expected_runtime = self.execute(baseline)
                    self.lib.configure_timing(1, 1000000, busy_until)
                    result, runtime = self.execute(candidate, autonomous=True)
                    self.assertEqual(result, expected_result)
                    self.assertEqual(runtime, expected_runtime)
                    np.testing.assert_array_equal(self.output(candidate), self.output(baseline))
                    self.assertGreaterEqual(self.lib.read_start(1), self.lib.read_finish(0) + 50)
                    self.assertGreaterEqual(self.lib.read_start(2), self.lib.read_finish(1))

    def test_autonomous_read_and_compute_errors_release_pending_io(self):
        for read_error, kernel_error in ((1, 0), (2, 0), (3, 0), (0, 1), (0, 2)):
            with self.subTest(read_error=read_error, kernel_error=kernel_error):
                result, _ = self.execute(self.make_case(46), autonomous=True,
                                         read_error=read_error, kernel_error=kernel_error)
                self.assertEqual(result, ERROR)

    def test_autonomous_abort_releases_suspended_context(self):
        for transition in (2, 3, 4, 5, 6, 7, 8):
            with self.subTest(transition=transition):
                result, _ = self.execute(self.make_case(47), autonomous=True,
                                         read_error=-transition)
                self.assertEqual(result, ERROR)

    def test_autonomous_metadata_uses_separate_version(self):
        baseline = self.make_case(48)
        autonomous = self.make_case(48, autonomous=True)
        self.assertEqual(struct.unpack_from("<I", autonomous["metadata"])[0], 3)
        self.assertEqual(baseline["metadata"][4:], autonomous["metadata"][4:])
        with self.assertRaises(ValueError):
            stage_metadata(autonomous["metadata"], 1, 1)

    def make_case(self, seed, dtype=np.float32, autonomous=False):
        layout = SparfKvCacheLayout(KvLayoutConfig(2, 20, 1, 2, 16, np.dtype(dtype)))
        rng = np.random.default_rng(seed)
        q = (rng.normal(size=(1, 4, 16)) / 4).astype(dtype)
        k = rng.normal(size=(17, 1, 2, 16)).astype(dtype)
        v = rng.normal(size=k.shape).astype(dtype)
        with tempfile.TemporaryDirectory() as directory:
            paths = [Path(directory) / name for name in ("k", "c", "v")]
            with SparfKvCacheStore(layout, *paths, replace_existing=True) as store:
                store.write_tokens(1, 0, k, v)
                store.flush()
                mean = store.value_mean(1)[:, [0, 0, 1, 1], :].copy()
                # Fake FTL files are indexed channel-K, token-K, token-V.
                files = [C.create_string_buffer(p.read_bytes()) for p in (paths[1], paths[0], paths[2])]
        metadata = pack_sparf_workflow(layout, 1, 17, 4, 4, 3,
            ((0, layout.token_file_size // 512),), ((0, layout.channel_file_size // 512),),
            ((0, layout.token_file_size // 512),), approximate_runtime_ns=50,
            exact_qk_runtime_ns=250, pv_runtime_ns=70, autonomous=autonomous)
        buffers = [C.create_string_buffer(8192) for _ in range(11)]
        buffers += [C.create_string_buffer(stage_context_bytes(12))]
        for index, array in ((0, q), (4, mean), (5, k[-1]), (6, v[-1])):
            C.memmove(buffers[index], array.ctypes.data, array.nbytes)
        return dict(buffers=buffers, files=files, metadata=metadata, q=q, k=k, v=v)

    def execute(self, case, stage=None, workflow_id=1, read_error=0, kernel_error=0, autonomous=False):
        payload = case["metadata"] if stage is None else stage_metadata(case["metadata"], stage, workflow_id)
        buffers, files = case["buffers"], case["files"]
        addresses = (C.c_void_p * len(buffers))(*(C.addressof(b) for b in buffers))
        sizes = (C.c_longlong * len(buffers))(*(C.sizeof(b) for b in buffers))
        nvm = (C.c_void_p * 3)(*(C.addressof(b) for b in files))
        runtime = C.c_uint64()
        execute = self.lib.execute_autonomous if autonomous else self.lib.execute_stage
        result = execute(addresses, sizes, payload, len(payload), nvm,
                                        C.byref(runtime), read_error, kernel_error)
        return result, runtime.value

    def output(self, case):
        return np.frombuffer(case["buffers"][9], dtype=case["q"].dtype, count=64).reshape(1, 4, 16)

    def test_buffered_history_matches_with_unpublished_nvm_tail(self):
        for dtype in (np.float16, np.float32):
            for published in (0, 13, 16, 17):
                for staged in (False, True, "autonomous"):
                    case = self.make_case(44, dtype)
                    layout = SparfKvCacheLayout(KvLayoutConfig(2, 20, 1, 2, 16, np.dtype(dtype)))
                    count = 17 - published
                    case["metadata"] = pack_sparf_workflow(layout, 1, 17, 4, 4, 17,
                        ((0, layout.token_file_size // 512),),
                        ((0, layout.channel_file_size // 512),),
                        ((0, layout.token_file_size // 512),), buffered_tokens=count)
                    case["buffers"][11] = C.create_string_buffer(stage_context_bytes(4 * 17))
                    for range_index, cache in ((5, case["k"]), (6, case["v"])):
                        source = np.ascontiguousarray(cache[published:] if count else cache[-1:])
                        C.memmove(case["buffers"][range_index], source.ctypes.data, source.nbytes)
                    for token in range(published, 17):
                        for head in range(2):
                            offset = layout.token_head_offset(1, 0, head, token)
                            for file_index in (1, 2):
                                C.memset(C.addressof(case["files"][file_index]) + offset, 0, layout.head_bytes)
                            for dimension in range(16):
                                offset = layout.channel_offset(1, 0, head, dimension, token)
                                C.memset(C.addressof(case["files"][0]) + offset, 0, layout.element_size)
                    for stage in ((1, 2, 3, 4) if staged is True else (None,)):
                        result, _ = self.execute(case, stage, autonomous=staged == "autonomous")
                        self.assertNotEqual(result, ERROR)
                    reference = sparf_attention(case["q"], case["k"], case["v"], 4, 17).output
                    tolerance = 2e-3 if dtype == np.float16 else 2e-5
                    np.testing.assert_allclose(self.output(case), reference, rtol=tolerance, atol=tolerance)
                    if published == 0:
                        trace = SparfWorkflowTrace.unpack(case["buffers"][10])
                        self.assertEqual(trace.channel_read_model_ns, 0)
                        self.assertEqual(trace.token_k_read_model_ns, 0)
                        self.assertEqual(trace.token_v_read_model_ns, 0)

    def test_stages_match_monolithic_and_numpy_with_two_interleaved_slots(self):
        for dtype in (np.float16, np.float32):
            cases = [self.make_case(seed, dtype) for seed in (11, 19)]
            references = []
            for seed, case in zip((11, 19), cases):
                baseline = self.make_case(seed, dtype)
                result, modeled = self.execute(baseline)
                self.assertNotEqual(result, ERROR)
                references.append(self.output(baseline).copy())
                self.assertEqual(modeled, 720)  # 100+50+200+max(250,300)+70
            runtimes = [0, 0]
            for stage in range(1, 5):
                for slot, case in enumerate(cases):
                    result, duration = self.execute(case, stage, workflow_id=slot+1)
                    self.assertNotEqual(result, ERROR)
                    runtimes[slot] += duration
            for case, expected, total in zip(cases, references, runtimes):
                np.testing.assert_array_equal(self.output(case), expected)
                numerical = sparf_attention(case["q"], case["k"], case["v"], 4, 3).output
                tolerance = 2e-3 if dtype == np.float16 else 2e-5
                np.testing.assert_allclose(self.output(case), numerical, rtol=tolerance, atol=tolerance)
                trace = SparfWorkflowTrace.unpack(case["buffers"][10])
                self.assertEqual(total, trace.total_model_ns)
                self.assertEqual(total, 720)

    def test_python_device_drives_c_stages_and_rejects_slot_reuse(self):
        from unittest.mock import Mock, patch
        import importlib
        import sys
        with patch.dict(sys.modules, {"_cemu_client": Mock()}):
            device_module = importlib.import_module("cemu_flexgen.cemu_sparf_device")
        from phase_profiler import PhaseProfiler
        case = self.make_case(21)
        test = self

        class Client:
            def write_tensor(self, index, value):
                C.memmove(case["buffers"][index], value.ctypes.data, value.nbytes)

            def read_tensor(self, index, shape, dtype):
                return np.frombuffer(case["buffers"][index], dtype=dtype,
                                     count=int(np.prod(shape))).reshape(shape).copy()

            def execute(self, *, cparam1, metadata):
                import struct
                fields = struct.unpack_from("<25If2I", metadata)
                version, phase, sequence = fields[0], fields[-2], fields[-1]
                canonical = bytearray(metadata)
                struct.pack_into("<I", canonical, 0, 1)
                struct.pack_into("<2I", canonical, 104, 0, 0)
                case["metadata"] = bytes(canonical)
                result, _ = test.execute(case, phase if version == 2 else None, sequence)
                if result == ERROR:
                    raise RuntimeError("device rejected stage")
                return result

            def close(self):
                pass

        layout = SparfKvCacheLayout(KvLayoutConfig(2, 20, 1, 2, 16, np.dtype(np.float32)))
        profiler = PhaseProfiler()
        device = device_module.CemuSparfDevice(
            layout, "k", "c", "v", fdm_directory="/tmp", program_path="unused",
            query_heads=4, top_r=4, compression_ratio=6, staged=True, profiler=profiler,
            approximate_runtime_ns=50, exact_qk_runtime_ns=250, pv_runtime_ns=70)
        device._device = Client()
        device._resolve_maps = lambda paths: (
            ((0, layout.token_file_size // 512),),
            ((0, layout.channel_file_size // 512),),
            ((0, layout.token_file_size // 512),))
        mean = case["v"].astype(np.float64).mean(axis=0).astype(np.float32)
        inputs = (case["q"], 1, 17, mean, case["k"][-1], case["v"][-1])
        request = device.prepare_decode(*inputs)
        with self.assertRaisesRegex(RuntimeError, "active workflow"):
            device.prepare_decode(*inputs)
        with self.assertRaisesRegex(RuntimeError, "before PV"):
            device.collect_decode(request)
        for _ in range(4):
            device.execute_stage(request)
        output = device.collect_decode(request)
        with self.assertRaisesRegex(RuntimeError, "stale request"):
            device.execute_stage(request)
        np.testing.assert_allclose(output, sparf_attention(case["q"], case["k"], case["v"], 4, 3).output,
                                   rtol=2e-5, atol=2e-5)
        self.assertEqual(device.modeled_runtime_summary(), (720, 1))
        metrics = {m.name: m for m in profiler.snapshot()}
        self.assertEqual(metrics["decode.execute_wait_wall"].count, 4)
        self.assertEqual(metrics["csd_model.total_model_ns"].count, 1)
        request = device.prepare_decode(*inputs)
        device.execute_stage(request)
        device.abort_decode(request)
        np.testing.assert_array_equal(device.run_decode(*inputs), output)
        device.close()

    def test_stage_order_identity_and_metadata_are_checked(self):
        case = self.make_case(13)
        self.assertEqual(self.execute(case, 2)[0], ERROR)
        self.assertNotEqual(self.execute(case, 1)[0], ERROR)
        self.assertEqual(self.execute(case, 3)[0], ERROR)
        self.assertEqual(self.execute(case, 2, workflow_id=2)[0], ERROR)
        original = case["metadata"]
        altered = bytearray(original)
        altered[-8] ^= 1  # different extent map, same shapes
        case["metadata"] = bytes(altered)
        self.assertEqual(self.execute(case, 2)[0], ERROR)
        case["metadata"] = original
        self.assertNotEqual(self.execute(case, 2)[0], ERROR)
        self.assertEqual(self.execute(case, 2)[0], ERROR)  # duplicate

    def test_frozen_clock_does_not_charge_previous_nand_work_again(self):
        origin = 1000000
        for initial_queue in (0, 200):
            expected = 720 + initial_queue
            traces = []
            for staged in (False, True):
                self.lib.configure_timing(1, origin, origin + initial_queue)
                case = self.make_case(13)
                total = 0
                for stage in (range(1, 5) if staged else (None,)):
                    # Guest receives each command completion before submitting
                    # its successor; within a command this clock stays frozen.
                    self.lib.set_virtual_time(origin + total)
                    result, duration = self.execute(case, stage)
                    self.assertNotEqual(result, ERROR)
                    total += duration
                self.assertEqual(total, expected)
                starts = [self.lib.read_start(i) - origin for i in range(3)]
                finishes = [self.lib.read_finish(i) - origin for i in range(3)]
                self.assertEqual(starts, [0, 150 + initial_queue, 350 + initial_queue])
                self.assertEqual(finishes, [100 + initial_queue, 350 + initial_queue,
                                           650 + initial_queue])
                traces.append(SparfWorkflowTrace.unpack(case["buffers"][10]))
            self.assertEqual(traces[0], traces[1])

    def test_guest_gaps_are_not_added_again_to_workflow_compute_time(self):
        origin = 1000000
        self.lib.configure_timing(1, origin, 0)
        case = self.make_case(13)
        total = 0
        for stage in range(1, 5):
            self.lib.set_virtual_time(origin + total + (stage - 1) * 500)
            result, duration = self.execute(case, stage)
            self.assertNotEqual(result, ERROR)
            total += duration
        self.assertEqual(total, 720)
        self.assertEqual([self.lib.read_start(i) - origin for i in range(3)],
                         [0, 650, 1350])

    def test_host_observations_preserve_outputs_and_show_stage_dependencies(self):
        case = self.make_case(13)
        self.execute(case)
        expected = self.output(case).copy()
        self.lib.configure_observation(1)
        total = 0
        expected_phases = {1: (0,), 2: (1,), 3: (2, 3), 4: (4,)}
        for stage in range(1, 5):
            result, duration = self.execute(case, stage)
            self.assertNotEqual(result, ERROR)
            total += duration
            spans = [(self.lib.observed_time(i, 0), self.lib.observed_time(i, 1))
                     for i in range(5)]
            for index, (start, end) in enumerate(spans):
                if index in expected_phases[stage]:
                    self.assertGreater(start, 0)
                    self.assertGreaterEqual(end, start)
                else:
                    self.assertEqual((start, end), (0, 0))
            if stage == 3:
                self.assertLessEqual(spans[2][0], spans[3][0])
                self.assertGreaterEqual(spans[2][1], spans[3][1])
        self.assertEqual(total, 720)
        np.testing.assert_array_equal(self.output(case), expected)

    def test_guest_sequence_check_with_production_cpu_workflow(self):
        """Exercise the Guest regression's data/shape checks without a VM."""
        import importlib
        import io
        import struct
        import sys
        from contextlib import redirect_stdout
        from types import SimpleNamespace
        from unittest.mock import Mock, patch

        with patch.dict(sys.modules, {"_cemu_client": Mock()}):
            device_module = importlib.import_module("cemu_flexgen.cemu_sparf_device")
            guest_test = importlib.import_module("test_cemu_sparf_workflow")
        test = self
        for dtype in (np.float16, np.float32):
            layout = SparfKvCacheLayout(KvLayoutConfig(2, 133, 1, 2, 16, np.dtype(dtype)))
            with tempfile.TemporaryDirectory() as directory:
                paths = tuple(Path(directory) / name for name in ("k", "c", "v"))

                class Transport:
                    def __init__(self, *, ranges, **kwargs):
                        self.buffers = [C.create_string_buffer(spec.size) for spec in ranges]

                    def open(self):
                        return self

                    def close(self):
                        pass

                    def write_tensor(self, index, array):
                        test.assertLessEqual(array.nbytes, C.sizeof(self.buffers[index]))
                        C.memmove(self.buffers[index], array.ctypes.data, array.nbytes)

                    def read_tensor(self, index, shape, dtype):
                        return np.frombuffer(self.buffers[index], dtype=dtype,
                            count=int(np.prod(shape))).reshape(shape).copy()

                    def execute(self, *, cparam1, metadata):
                        fields = struct.unpack_from("<25If2I", metadata)
                        canonical = bytearray(metadata)
                        struct.pack_into("<I", canonical, 0, 1)
                        struct.pack_into("<2I", canonical, 104, 0, 0)
                        case = dict(buffers=self.buffers, metadata=bytes(canonical),
                            files=[C.create_string_buffer(p.read_bytes())
                                   for p in (paths[1], paths[0], paths[2])])
                        result, _ = test.execute(case, fields[-2] if fields[0] == 2 else None,
                                                 fields[-1], autonomous=fields[0] == 3)
                        if result == ERROR:
                            raise RuntimeError("C workflow failed")

                args = SimpleNamespace(top_r=4, compression_ratio=8, decode_steps=4,
                                       control="unused", namespace="unused", cuda=False, autonomous=True)
                rng = np.random.default_rng(20260910)
                keys = rng.normal(size=(129, 1, 2, 16)).astype(dtype)
                values = rng.normal(size=keys.shape).astype(dtype)
                with patch.object(device_module, "CemuDevice", Transport), \
                     patch.object(guest_test, "CemuSparfDevice", device_module.CemuSparfDevice), \
                     patch.object(device_module, "get_file_extents", side_effect=lambda p, n: ((0, n // 512),)), \
                     SparfKvCacheStore(layout, *paths) as store, redirect_stdout(io.StringIO()):
                    guest_test.compare_decode_sequence(layout, paths, directory, "unused",
                                                       args, store, keys, values)

    def test_failed_read_or_kernel_cannot_publish_output_and_select_recovers(self):
        for read_error, kernel_error in ((3, 0), (0, 1)):
            case = self.make_case(17)
            self.execute(case, 1)
            self.execute(case, 2)
            self.assertEqual(self.execute(case, 3, read_error=read_error, kernel_error=kernel_error)[0], ERROR)
            self.assertEqual(self.execute(case, 4)[0], ERROR)
            for stage in range(1, 5):
                self.assertNotEqual(self.execute(case, stage, workflow_id=2)[0], ERROR)

    def test_buffered_guest_driver_with_production_workflow(self):
        import importlib
        import io
        import struct
        import sys
        from contextlib import redirect_stdout
        from threading import Lock
        from types import SimpleNamespace
        from unittest.mock import Mock, patch

        with patch.dict(sys.modules, {"_cemu_client": Mock()}):
            device_module = importlib.import_module("cemu_flexgen.cemu_sparf_device")
            guest_test = importlib.import_module("test_sparf_buffer_device")
        test = self
        execution_lock = Lock()
        registry = {}

        def extents(path, size):
            identifier = next((key for key, value in registry.items() if value == path), None)
            if identifier is None:
                identifier = len(registry) + 1
                registry[identifier] = path
            return ((identifier, size // 512),)

        class Transport:
            def __init__(self, *, ranges, **kwargs):
                self.buffers = [C.create_string_buffer(spec.size) for spec in ranges]

            def open(self):
                return self

            def close(self):
                pass

            def write_tensor(self, index, array):
                test.assertLessEqual(array.nbytes, C.sizeof(self.buffers[index]))
                C.memmove(self.buffers[index], array.ctypes.data, array.nbytes)

            def read_tensor(self, index, shape, dtype):
                return np.frombuffer(self.buffers[index], dtype=dtype,
                    count=int(np.prod(shape))).reshape(shape).copy()

            def execute(self, *, cparam1, metadata):
                with execution_lock:
                    fields = struct.unpack_from("<25If2I", metadata)
                    paths = [registry[struct.unpack_from("<Q", metadata, 112 + index * 16)[0]]
                             for index in range(3)]
                    canonical = bytearray(metadata)
                    struct.pack_into("<I", canonical, 0, 1)
                    struct.pack_into("<2I", canonical, 104, 0, 0)
                    case = dict(buffers=self.buffers, metadata=bytes(canonical),
                        files=[C.create_string_buffer(Path(path).read_bytes())
                               for path in (paths[1], paths[0], paths[2])])
                    result, _ = test.execute(case, fields[-2] if fields[-2] else None,
                                             fields[-1], autonomous=fields[0] == 3)
                    if result == ERROR:
                        raise RuntimeError("buffered C workflow failed")

        for dtype, autonomous in (("float16", False), ("float32", False),
                                  ("float16", True), ("float32", True)):
            for ratio in (1, 8):
                with tempfile.TemporaryDirectory() as directory:
                    root = Path(directory)
                    with patch.object(device_module, "CemuDevice", Transport), \
                         patch.object(guest_test, "CemuSparfDevice", device_module.CemuSparfDevice), \
                         patch.object(device_module, "get_file_extents", side_effect=extents), \
                         redirect_stdout(io.StringIO()):
                        guest_test.run_case(root, root,
                            SimpleNamespace(dtype=dtype, program="unused", cuda=False,
                                            device_writeback=False, autonomous=autonomous,
                                            writeback_format="compact"), ratio)


if __name__ == "__main__":
    unittest.main()
