import ctypes as ct
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np

import test_instattention_qemu_io as native_tests
import test_sparf_stages as stage_tests
from cemu_flexgen import KvLayoutConfig, SparfKvCacheLayout, sparf_attention
from cemu_flexgen.sparf_workflow_abi import pack_grouped_sparf_workflow, SparfWorkflowTrace


class NativeAttentionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        stage_tests.DeviceStagesTest.setUpClass()
        cls.addClassCleanup(stage_tests.DeviceStagesTest.doClassCleanups)
        cls.workflow = stage_tests.DeviceStagesTest.lib
        cls.workflow.configure_storage.argtypes = [ct.c_void_p, ct.c_void_p]
        temporary = tempfile.TemporaryDirectory(prefix="native-attention-")
        cls.addClassCleanup(temporary.cleanup)
        path = native_tests.InstattentionQemuIoTest().build_sources([
            "hw/femu/bbssd/ftl.c", "hw/femu/backend/backend.c",
            "hw/femu/bbssd/instattention_mapping.c", "hw/femu/bbssd/instattention_io.c",
            "hw/femu/bbssd/instattention_device.c", "hw/femu/bbssd/instattention_buffer.c",
            "hw/femu/bbssd/instattention_attention.c",
            "tests/cemu/instattention/guest/test_instattention_attention.c",
        ], Path(temporary.name), shared=True)
        cls.native = ct.CDLL(str(path))
        signatures = {
            "create": ([ct.c_uint32] * 4, ct.c_void_p),
            "create_budget": ([ct.c_uint32] * 4 + [ct.c_uint64], ct.c_void_p),
            "read": ([ct.c_void_p, ct.c_void_p, ct.c_void_p, ct.c_uint32,
                      ct.c_void_p, ct.c_size_t, ct.c_uint64, ct.c_void_p], ct.c_bool),
            "free": ([ct.c_void_p], None),
            "append": ([ct.c_void_p] + [ct.c_uint32] * 5 + [ct.c_void_p, ct.c_void_p, ct.c_size_t], ct.c_int),
            "append_batch": ([ct.c_void_p] + [ct.c_uint32] * 4 + [ct.c_void_p, ct.c_void_p, ct.c_size_t], ct.c_int),
            "accepted": ([ct.c_void_p, ct.c_uint32, ct.c_uint32], ct.c_uint64),
            "open": ([ct.c_void_p, ct.c_void_p, ct.c_uint32, ct.c_uint32], ct.c_int),
            "close": ([ct.c_void_p], None),
            "stop_leased": ([ct.c_void_p], None),
            "fail": ([ct.c_void_p, ct.c_int], None),
            "time": ([ct.c_uint64], ct.c_uint64),
            "drain": ([ct.c_void_p], ct.c_int),
            "result": ([ct.c_void_p, ct.c_uint32, ct.c_uint32], ct.c_uint64),
        }
        for name, (arguments, result) in signatures.items():
            function = getattr(cls.native, f"native_attention_fixture_{name}")
            function.argtypes = arguments
            function.restype = result

    def create(self, dtype=np.float16, budget=16 * 1024**2):
        self.dtype = np.dtype(dtype)
        self.context = self.native.native_attention_fixture_create_budget(2, 2, 2, self.dtype.itemsize, budget)
        self.assertTrue(self.context)
        self.addCleanup(self.native.native_attention_fixture_free, self.context)
        self.addCleanup(self.workflow.configure_storage, None, None)
        self.workflow.configure_storage(self.native.native_attention_fixture_read, self.context)
        self.workflow.configure_observation(0)
        self.workflow.configure_autonomous_trace(0)
        generator = np.random.default_rng(430)
        self.keys = (generator.normal(size=(2, 2, 2, 600, 128)) / math.sqrt(128)).astype(dtype)
        self.values = generator.normal(size=self.keys.shape).astype(dtype)
        self.accepted = 0

    def append_once(self, batch, layer, head, start, count):
        key = np.ascontiguousarray(self.keys[batch, layer, head, start:start + count])
        value = np.ascontiguousarray(self.values[batch, layer, head, start:start + count])
        return self.native.native_attention_fixture_append(self.context, batch, layer, head,
            start, count, key.ctypes.data, value.ctypes.data, key.nbytes)

    def test_batch_append_atomic_validation_and_backpressure(self):
        self.create()

        def accepted(head, field=0):
            return self.native.native_attention_fixture_accepted(self.context, head, field)

        def append(head, heads, token, count):
            key = np.ascontiguousarray(self.keys[0, 0, head:head + heads, token:token + count])
            value = np.ascontiguousarray(self.values[0, 0, head:head + heads, token:token + count])
            return self.native.native_attention_fixture_append_batch(self.context, head, heads,
                token, count, key.ctypes.data, value.ctypes.data, key.nbytes)

        self.assertEqual(append(0, 1, 0, 16), 0)
        self.assertNotEqual(append(0, 2, 16, 1), 0)
        self.assertEqual([accepted(head) for head in range(2)], [16, 0])
        for attempt in range(100):
            self.native.native_attention_fixture_time(100000)
            if accepted(0, 1) == 16:
                break
        self.assertEqual(accepted(0, 1), 16)
        self.assertEqual(append(0, 1, 16, 16), 0)
        self.assertEqual(append(1, 1, 0, 32), 0)
        self.assertEqual(accepted(1, 1), 0)
        self.assertEqual(append(0, 2, 32, 1), 5)
        self.assertEqual([accepted(head) for head in range(2)], [32, 32])
        for attempt in range(100):
            self.native.native_attention_fixture_time(100000)
            status = append(0, 2, 32, 1)
            if status != 5:
                break
        self.assertEqual(status, 0)
        self.assertEqual([accepted(head) for head in range(2)], [33, 33])
        self.assertNotEqual(append(0, 2, 32, 1), 0)
        self.assertEqual([accepted(head) for head in range(2)], [33, 33])

    def append(self, end):
        for batch in range(2):
            for layer in range(2):
                for head in range(2):
                    for start in range(self.accepted, end, 8):
                        for attempt in range(2000):
                            status = self.append_once(batch, layer, head, start, min(8, end - start))
                            if status != 5:
                                self.assertEqual(status, 0)
                                break
                            self.native.native_attention_fixture_time(100000)
                        else:
                            self.fail("native append did not make progress")
        self.accepted = end

    def case(self, tokens, layer=0, first_batch=0, batches=2, ratio=8):
        layout = SparfKvCacheLayout(KvLayoutConfig(2, 600, batches, 2, 128, self.dtype))
        top_k = math.ceil(tokens / ratio)
        generator = np.random.default_rng(tokens)
        query = np.stack([
            generator.permutation(np.arange(1, 129)) * generator.choice((-1, 1), 128)
            for vector in range(batches * 4)
        ]).reshape(batches, 4, 128).astype(self.dtype) / 1024
        keys = self.keys[first_batch:first_batch + batches, layer, :, :tokens].transpose(2, 0, 1, 3)
        values = self.values[first_batch:first_batch + batches, layer, :, :tokens].transpose(2, 0, 1, 3)
        mean = values.astype(np.float64).mean(axis=0).astype(self.dtype)[:, [0, 0, 1, 1]]
        vectors = batches * 4
        arrays = [query, np.zeros((vectors, 16, tokens), self.dtype),
            np.zeros((vectors, top_k, 128), self.dtype),
            np.zeros((vectors, top_k + 1, 128), self.dtype), mean,
            np.full((batches, 2, 128), np.nan, self.dtype),
            np.full((batches, 2, 128), np.nan, self.dtype),
            np.zeros(vectors, np.float32), np.zeros((vectors, top_k + 1), np.float32),
            np.zeros_like(query), np.zeros(72, np.uint8)]
        arrays = [np.ascontiguousarray(array) for array in arrays]
        metadata = pack_grouped_sparf_workflow(layout, layer, tokens, 4, 16, top_k, 1,
            approximate_runtime_ns=50, exact_qk_runtime_ns=300000, pv_runtime_ns=70)
        expected = sparf_attention(query, keys, values, 16, top_k).output
        return arrays, metadata, expected, first_batch

    def execute(self, case, fail_kind=0, fail_kernel=0, abort=0):
        arrays, metadata, expected, first_batch = case
        self.assertEqual(self.native.native_attention_fixture_open(self.context, metadata, first_batch, 1), 0)
        try:
            self.native.native_attention_fixture_fail(self.context, fail_kind)
            now = self.native.native_attention_fixture_time(0)
            self.workflow.configure_timing(0, now, 0)
            pointers = (ct.c_void_p * len(arrays))(*(array.ctypes.data for array in arrays))
            lengths = (ct.c_longlong * len(arrays))(*(array.nbytes for array in arrays))
            runtime = ct.c_uint64()
            result = self.workflow.execute_autonomous(pointers, lengths, metadata, len(metadata),
                None, ct.byref(runtime), abort, fail_kernel)
            if fail_kind or fail_kernel or abort:
                self.assertEqual(result, stage_tests.ERROR)
                return
            self.assertNotEqual(result, stage_tests.ERROR)
            tolerance = 2e-3 if self.dtype == np.float16 else 2e-5
            np.testing.assert_allclose(arrays[9], expected, rtol=tolerance, atol=tolerance)
            reads = [[self.native.native_attention_fixture_result(self.context, kind, field)
                      for field in range(8)] for kind in range(3)]
            trace = SparfWorkflowTrace.unpack(arrays[10])
            channel, key, value = reads[1], reads[0], reads[2]
            self.assertEqual(key[0], channel[1] + 50)
            self.assertEqual(value[0], key[1])
            self.assertEqual(runtime.value, trace.total_model_ns)
            self.assertEqual(runtime.value, channel[1] - channel[0] + 50 + key[1] - key[0] +
                             max(300000, value[1] - value[0]) + 70)
            for read in reads:
                self.assertEqual(read[2], read[5] * 4096)
                self.assertEqual(read[7], 0)
            return reads
        finally:
            self.native.native_attention_fixture_close(self.context)

    def test_tail_mixed_flash_boundaries_and_split_gather(self):
        self.create()
        for tokens in (15, 16, 253, 256, 259, 531):
            self.append(tokens)
            for layer in (0, 1):
                for ratio in (1, 8):
                    with self.subTest(tokens=tokens, layer=layer, ratio=ratio):
                        reads = self.execute(self.case(tokens, layer, ratio=ratio))
                        if tokens == 15:
                            self.assertTrue(all(read[2] == 0 for read in reads))
                        if tokens == 531:
                            self.assertGreater(reads[1][2], 0)
                            self.assertGreater(reads[1][3], 0)
                            self.assertGreater(reads[1][6], 1)
        self.drain()
        reads = self.execute(self.case(531, layer=1, first_batch=1, batches=1))
        self.assertTrue(all(read[3] == 0 for read in reads))

    def drain(self):
        for attempt in range(2000):
            status = self.native.native_attention_fixture_drain(self.context)
            if status != 5:
                self.assertEqual(status, 0)
                return
            self.native.native_attention_fixture_time(100000)
        self.fail("native drain did not complete")

    def test_fp32_attention(self):
        self.create(np.float32)
        self.append(259)
        for ratio in (1, 8):
            self.execute(self.case(259, ratio=ratio))
        self.drain()
        self.execute(self.case(259, layer=1, first_batch=1, batches=1))

    def read_selected(self, selected, kind, query_heads=4, top_r=16, expect_success=True):
        layout = SparfKvCacheLayout(KvLayoutConfig(2, 600, 1, 2, 128, self.dtype))
        metadata = pack_grouped_sparf_workflow(
            layout, 1, 531, query_heads, top_r, selected.shape[1] if kind != 1 else 129, 1)
        shape = (query_heads, top_r, 531) if kind == 1 else (
            query_heads, selected.shape[1] + (kind == 2), 128)
        output = np.full(shape, np.nan, self.dtype)
        indices = np.ascontiguousarray(selected, dtype=np.uint32)
        self.assertEqual(self.native.native_attention_fixture_open(self.context, metadata, 1, 1), 0)
        try:
            start = self.native.native_attention_fixture_time(1000000000)
            result = ct.create_string_buffer(128)
            success = self.native.native_attention_fixture_read(
                self.context, metadata, indices.ctypes.data, kind,
                output.ctypes.data, output.nbytes, start, result)
            self.assertEqual(success, expect_success)
            fields = [self.native.native_attention_fixture_result(self.context, kind, field)
                      for field in range(8)]
        finally:
            self.native.native_attention_fixture_close(self.context)
        return output, fields

    def test_phase_page_dedup_preserves_gqa_selection_order(self):
        self.create()
        self.append(531)
        self.drain()
        for selected_count in (128, 129, 256, 257):
            for shuffled in (False, True):
                tokens = np.arange(selected_count, dtype=np.uint32)
                if shuffled:
                    tokens = np.concatenate([tokens[offset::16] for offset in range(16)])
                selected = np.stack((tokens, tokens[::-1], tokens, tokens[::-1]))
                for kind, source in ((0, self.keys), (2, self.values)):
                    with self.subTest(selected=selected_count, shuffled=shuffled, kind=kind):
                        output, fields = self.read_selected(selected, kind)
                        expected = np.stack([source[1, 1, head // 2, selected[head]] for head in range(4)])
                        np.testing.assert_array_equal(output[:, :selected_count], expected)
                        if kind == 2:
                            np.testing.assert_array_equal(output[:, -1], 0)
                        pages = 2 * math.ceil(selected_count / 16)
                        self.assertEqual(fields[5], pages)
                        self.assertEqual(fields[2], pages * 4096)
                        self.assertEqual(fields[6], math.ceil(pages / 32))
                        self.assertEqual(fields[1] - fields[0], math.ceil(pages / 4) * 400)
        channels = np.concatenate([np.arange(offset, 64, 8) for offset in range(8)])
        selected = np.stack((channels, channels[::-1], channels, channels[::-1]))
        output, fields = self.read_selected(selected, 1, top_r=64)
        expected = np.stack([self.keys[1, 1, head // 2, :531][:, selected[head]].T for head in range(4)])
        np.testing.assert_array_equal(output, expected)
        self.assertEqual(fields[5], 2 * 8 * 3)
        self.assertEqual(fields[6], 2)
        self.assertEqual(fields[2], fields[5] * 4096)

    def test_phase_plan_budget_and_invalid_selection_fail_without_io(self):
        self.create(budget=1400000)
        self.append(531)
        self.drain()
        selected = np.tile(np.arange(531, dtype=np.uint32), (4, 1))
        output, fields = self.read_selected(selected, 0, expect_success=False)
        self.assertTrue(np.isnan(output).all())
        self.assertEqual(fields[2], 0)
        self.assertEqual(fields[5], 0)
        self.assertEqual(fields[7], 8)
        selected = np.zeros((4, 1), dtype=np.uint32)
        output, fields = self.read_selected(selected, 0)
        self.assertEqual(fields[5], 2)
        np.testing.assert_array_equal(output[:, 0], self.keys[1, 1, [0, 0, 1, 1], 0])
        selected[-1, 0] = 531
        output, fields = self.read_selected(selected, 0, expect_success=False)
        self.assertTrue(np.isnan(output).all())
        self.assertEqual(fields[2], 0)

    def test_phase_backend_failure_releases_plan_and_lease(self):
        self.create()
        self.append(531)
        self.drain()
        case = self.case(531, batches=1)
        self.execute(case, fail_kind=-1)
        self.assertEqual(self.native.native_attention_fixture_result(self.context, 1, 7), 6)
        self.execute(case)

    def test_lease_rejections_and_error_cleanup(self):
        self.create()
        self.append(15)
        case = self.case(15, batches=1)
        metadata = case[1]
        self.assertNotEqual(self.native.native_attention_fixture_open(self.context, metadata, 0, 2), 0)
        self.assertNotEqual(self.native.native_attention_fixture_open(self.context, self.case(14)[1], 0, 1), 0)
        self.assertEqual(self.native.native_attention_fixture_open(self.context, metadata, 0, 1), 0)
        try:
            self.assertEqual(self.append_once(0, 0, 0, 15, 1), 5)
            key = np.ascontiguousarray(self.keys[0, 0, :, 15:16])
            value = np.ascontiguousarray(self.values[0, 0, :, 15:16])
            self.assertEqual(self.native.native_attention_fixture_append_batch(self.context,
                0, 2, 15, 1, key.ctypes.data, value.ctypes.data, key.nbytes), 5)
            for head in range(2):
                self.assertEqual(self.native.native_attention_fixture_accepted(self.context, head, 0), 15)
            self.assertEqual(self.native.native_attention_fixture_drain(self.context), 5)
            self.assertEqual(self.append_once(1, 1, 0, 15, 1), 0)
        finally:
            self.native.native_attention_fixture_close(self.context)
        for kind in (1, 2, 3):
            self.execute(case, fail_kind=kind)
        for phase in (1, 2):
            self.execute(case, fail_kernel=phase)
        for transition in range(2, 9):
            self.execute(case, abort=-transition)
        self.execute(case)
        self.assertEqual(self.append_once(0, 0, 0, 15, 1), 0)

    def test_stop_waits_for_active_attention(self):
        self.create()
        self.append(15)
        metadata = self.case(15)[1]
        self.assertEqual(self.native.native_attention_fixture_open(self.context, metadata, 0, 1), 0)
        self.native.native_attention_fixture_stop_leased(self.context)
        self.assertNotEqual(self.native.native_attention_fixture_open(self.context, metadata, 0, 1), 0)


if __name__ == "__main__":
    unittest.main()
