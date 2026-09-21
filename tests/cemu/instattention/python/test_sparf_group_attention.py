import ctypes as ct
import math
import json
import time
from types import SimpleNamespace
import unittest

import numpy as np

import test_sparf_group_buffer as buffer_tests
import test_sparf_stages as stage_tests
from cemu_flexgen import CemuSparfDevice, KvLayoutConfig, SparfKvCacheLayout, sparf_attention
from cemu_flexgen.sparf_group_device import CLOSE, DRAIN, command
from cemu_flexgen.sparf_group_layout import SparfGroupLayout
from cemu_flexgen.sparf_workflow_abi import pack_grouped_sparf_workflow, SparfWorkflowTrace


class GroupedAttentionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        buffer_tests.GroupBufferTest.setUpClass()
        cls.addClassCleanup(buffer_tests.GroupBufferTest.doClassCleanups)
        stage_tests.DeviceStagesTest.setUpClass()
        cls.addClassCleanup(stage_tests.DeviceStagesTest.doClassCleanups)
        cls.lib = stage_tests.DeviceStagesTest.lib
        cls.lib.configure_grouped_session.argtypes = [ct.c_void_p]
        cls.lib.grouped_test_lease.argtypes = [ct.c_void_p, ct.c_int]
        cls.lib.grouped_test_lease.restype = ct.c_int
        library = buffer_tests.GroupBufferTest.library
        library.group_buffer_test_session.argtypes = [ct.c_void_p]
        library.group_buffer_test_session.restype = ct.c_void_p

    def setUp(self):
        self.store = buffer_tests.GroupBufferTest()
        self.store.setUp()
        self.addCleanup(self.store.doCleanups)
        self.store.register()
        self.lib.configure_grouped_session(self.store.library.group_buffer_test_session(self.store.context))
        self.lib.configure_observation(0)
        self.lib.configure_autonomous_trace(0)
        self.layout = SparfKvCacheLayout(KvLayoutConfig(1, 1051, 1, 2, 128, np.float16))

    def append(self, start, stop):
        self.assertEqual(self.store.append_batch(start, stop)[0], 1)

    def case(self, tokens, ratio=8):
        dtype = self.store.keys.dtype
        query = (np.random.default_rng(tokens).normal(size=(1, 4, 128)) / math.sqrt(128)).astype(dtype)
        keys = self.store.keys[:, :tokens].transpose(1, 0, 2)[:, None]
        values = self.store.values[:, :tokens].transpose(1, 0, 2)[:, None]
        top_k = math.ceil(tokens / ratio)
        mean = values.astype(np.float64).mean(axis=0).astype(dtype)[:, [0, 0, 1, 1]]
        arrays = [query, np.zeros((4, 16, tokens), dtype), np.zeros((4, top_k, 128), dtype),
                  np.zeros((4, top_k + 1, 128), dtype), mean,
                  np.full((1, 2, 128), np.nan, dtype), np.full((1, 2, 128), np.nan, dtype),
                  np.zeros(4, np.float32), np.zeros((4, top_k + 1), np.float32),
                  np.zeros_like(query), np.zeros(72, np.uint8)]
        arrays = [np.ascontiguousarray(array) for array in arrays]
        metadata = pack_grouped_sparf_workflow(self.layout, 0, tokens, 4, 16, top_k, self.store.cookie,
                                               approximate_runtime_ns=50, exact_qk_runtime_ns=250, pv_runtime_ns=70)
        return arrays, metadata, sparf_attention(query, keys, values, 16, top_k).output

    def execute(self, case, read_error=0, kernel_error=0):
        arrays, metadata, expected = case
        pointers = (ct.c_void_p * len(arrays))(*(array.ctypes.data for array in arrays))
        lengths = (ct.c_longlong * len(arrays))(*(array.nbytes for array in arrays))
        files = (ct.c_void_p * 3)(*[self.store.disk.ctypes.data + 2048 * 512] * 3)
        runtime = ct.c_uint64()
        self.lib.configure_timing(1, 1000000, 0)
        result = self.lib.execute_autonomous(pointers, lengths, metadata, len(metadata), files,
                                             ct.byref(runtime), read_error, kernel_error)
        if result != stage_tests.ERROR:
            tolerance = 2e-3 if arrays[0].dtype == np.float16 else 2e-5
            np.testing.assert_allclose(arrays[9], expected, atol=tolerance, rtol=tolerance)
        return result, runtime.value, SparfWorkflowTrace.unpack(arrays[10])

    def test_tail_only_attention_ignores_guest_current_kv(self):
        self.append(0, 15)
        result, runtime, trace = self.execute(self.case(15))
        self.assertNotEqual(result, stage_tests.ERROR)
        self.assertEqual(runtime, 370)
        self.assertEqual((trace.channel_read_model_ns, trace.token_k_read_model_ns, trace.token_v_read_model_ns), (0, 0, 0))

    def test_read_diagnostics_preserve_tail_and_mixed_outputs_and_budgets(self):
        position = 0
        for tokens in (15, 531):
            with self.subTest(tokens=tokens):
                self.append(position, tokens)
                position = tokens
                deadline = time.monotonic() + 3
                while any(self.store.progress(stream)[1:4].tolist() !=
                          [tokens // 16 * 16, tokens // 256 * 256, tokens // 16 * 16]
                          for stream in range(2)):
                    if time.monotonic() > deadline:
                        self.fail("background publication did not settle")
                    time.sleep(0.001)
                self.lib.configure_autonomous_trace(1)
                baseline = self.execute(self.case(tokens))
                self.lib.configure_autonomous_trace(2)
                measured = self.execute(self.case(tokens))
                self.assertEqual(baseline[:2], measured[:2])
                records = [json.loads(line) for line in self.lib.get_autonomous_trace().decode().splitlines()]
                for read in records[1]["reads"]:
                    data = read["io"]
                    self.assertEqual(data["requested_bytes"], data["logical_bytes"] + data["tail_bytes"])
                    self.assertEqual(data["model_start_ns"], read["submitted"]["virtual_ns"])
                    self.assertEqual(data["model_end_ns"], read["deadline_ns"])
                channel = records[1]["reads"][0]["io"]
                self.assertEqual(channel["requested_bytes"], 4 * 16 * tokens * 2)
                self.assertGreater(channel["tail_bytes"], 0)
                self.assertEqual(channel["logical_bytes"] > 0, tokens > 15)

    def test_attention_during_pending_background_write(self):
        self.store.library.group_buffer_test_gate(self.store.context, 1, 0)
        try:
            self.append(0, 16)
            self.assertEqual(self.store.library.group_buffer_test_wait_write(self.store.context), 1)
            self.assertNotEqual(self.execute(self.case(16))[0], stage_tests.ERROR)
            np.testing.assert_array_equal(self.store.progress()[:4], [16, 0, 0, 0])
        finally:
            self.store.library.group_buffer_test_gate(self.store.context, 0, 0)

    def test_mixed_reads_cross_groups_ring_reuse_and_sealed_flash(self):
        position = 0
        for end in (253, 256, 259, 531):
            self.append(position, end)
            for ratio in (1, 8):
                self.assertNotEqual(self.execute(self.case(end, ratio))[0], stage_tests.ERROR)
            position = end
        self.assertEqual(self.store.execute(command(DRAIN, self.store.cookie))[0], 1)
        result, runtime, trace = self.execute(self.case(531))
        self.assertNotEqual(result, stage_tests.ERROR)
        self.assertEqual(runtime, 720)
        self.assertEqual(trace.total_model_ns, 720)
        self.assertEqual(min(trace.exact_qk_model_ns, trace.token_v_read_model_ns), 250)

    def test_lifetime_lease_rejects_overwrite_close_and_drain(self):
        self.append(0, 15)
        case = self.case(15)
        metadata = case[1]
        self.assertEqual(self.lib.grouped_test_lease(metadata, 0), 1)
        try:
            self.assertEqual(self.store.append(0, 15, 16)[0], 0)
            self.assertEqual(self.store.execute(command(CLOSE, self.store.cookie))[0], 0)
            self.assertEqual(self.store.execute(command(DRAIN, self.store.cookie))[0], 0)
            self.assertEqual(self.store.progress()[0], 15)
        finally:
            self.lib.grouped_test_lease(metadata, 1)
        self.append(15, 16)

    def test_stale_cookie_wrong_length_and_failed_phase_release_lease(self):
        self.append(0, 259)
        deadline = time.monotonic() + 3
        while any(self.store.progress(stream)[2] < 256 for stream in range(2)):
            if time.monotonic() > deadline:
                self.fail("channel groups did not commit")
            time.sleep(0.001)
        case = self.case(259)
        metadata = bytearray(case[1])
        metadata[112] ^= 1
        self.assertEqual(self.execute((case[0], bytes(metadata), case[2]))[0], stage_tests.ERROR)
        self.assertEqual(self.execute(self.case(258))[0], stage_tests.ERROR)
        for phase in (1, 2, 3):
            self.assertEqual(self.execute(self.case(259), read_error=phase)[0], stage_tests.ERROR)
        self.assertEqual(self.execute(self.case(259), kernel_error=1)[0], stage_tests.ERROR)
        self.append(259, 260)
        self.assertNotEqual(self.execute(self.case(260))[0], stage_tests.ERROR)

    def test_python_adapter_uses_binding_and_only_writes_query_mean(self):
        self.append(0, 15)
        case = self.case(15)
        writes = []
        test = self

        class Client:
            def write_tensor(self, index, values):
                writes.append(index)
                np.copyto(case[0][index], values)

            def execute(self, *, cparam1, metadata):
                result, _, _ = test.execute((case[0], metadata, case[2]))
                if result == stage_tests.ERROR:
                    raise RuntimeError("grouped workflow failed")
                return result

            def read_tensor(self, index, shape, dtype):
                return case[0][index].copy().reshape(shape)

        store = SimpleNamespace(layout=self.store.layout, buffer_streams=2, registered=True,
                                cookie=self.store.cookie, control_path="/dev/nvme0c3", namespace_path="/dev/ng0n3")
        device = CemuSparfDevice(self.layout, "unused-k", "unused-c", "unused-v", "/tmp", "unused-program",
                                 query_heads=4, top_r=16, compression_ratio=8, autonomous=True, grouped_store=store,
                                 approximate_runtime_ns=50, exact_qk_runtime_ns=250, pv_runtime_ns=70)
        device._device = Client()
        device._resolve_maps = lambda *_: self.fail("grouped Attention must not resolve Guest KV file maps")
        mean = self.store.values[:, :15].astype(np.float64).mean(axis=1).astype(np.float16)[None]
        output = device.run_decode(case[0][0], 0, 15, mean)
        np.testing.assert_allclose(output, case[2], atol=2e-3, rtol=2e-3)
        self.assertEqual(writes, [0, 4])

    def test_float32_and_multi_batch_mapping(self):
        self.assertEqual(self.store.execute(command(CLOSE, self.store.cookie))[0], 1)
        self.store.keys = self.store.keys.astype(np.float32)
        self.store.values = self.store.values.astype(np.float32)
        self.store.register(SparfGroupLayout(128, 4, dtype=np.float32))
        self.lib.configure_grouped_session(self.store.library.group_buffer_test_session(self.store.context))
        self.layout = SparfKvCacheLayout(KvLayoutConfig(1, 1051, 1, 2, 128, np.float32))
        self.append(0, 259)
        case = self.case(259)
        self.assertNotEqual(self.execute(case)[0], stage_tests.ERROR)
        layout = SparfKvCacheLayout(KvLayoutConfig(1, 1051, 2, 1, 128, np.float32))
        metadata = pack_grouped_sparf_workflow(layout, 0, 259, 2, 16, math.ceil(259 / 8), self.store.cookie,
                                               approximate_runtime_ns=50, exact_qk_runtime_ns=250, pv_runtime_ns=70)
        arrays = case[0]
        arrays[0] = arrays[0].reshape(2, 2, 128)
        arrays[9] = arrays[9].reshape(2, 2, 128)
        self.assertNotEqual(self.execute((arrays, metadata, case[2].reshape(2, 2, 128)))[0], stage_tests.ERROR)


if __name__ == "__main__":
    unittest.main()
