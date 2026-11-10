"""Exercise adapter semantics with the real QEMU FDM/aggregate/SparF path.

No vLLM/torch/GPU is needed here; this does not replace a real LLM smoke run.
The qtest transport replaces only Guest ioctl plumbing, not the device.
"""

from contextlib import ExitStack
import json
import math
import os
import struct
import tempfile
from threading import RLock
import unittest
from unittest.mock import patch

import numpy as np

from cemu_flexgen import KvLayoutConfig, SparfKvCacheLayout, sparf_attention
from cemu_flexgen.cemu_sparf_device import CemuSparfDevice
import test_native_csd_pipeline as harness_module
from vllm_adapter.session import Config, Session
from phase_profiler import PhaseProfiler
from vllm_adapter.profiling import device_report


class CemuVllmTest(unittest.TestCase):
    def test_fdm_gqa_scale_reordered_requests_and_page_boundaries(self):
        self.run_pipeline(compression=8, top_r=16)

    def test_fdm_all_selected_matches_dense_and_default_scale(self):
        self.run_pipeline(compression=1, top_r=128, scale=1.0)

    def test_fdm_all_selected_llama_scale(self):
        self.run_pipeline(compression=1, top_r=128)

    def test_fdm_async_gqa(self):
        with patch.dict(os.environ, CEMU_FDM_ASYNC_AGGREGATE="1"):
            self.run_pipeline(compression=8, top_r=16)

    def test_ragged_b4_reorder_and_shrink(self):
        self.run_pipeline(compression=8, top_r=16, batch_size=4)

    def test_fdm_opt_mha(self):
        self.run_pipeline(compression=1, top_r=128, kv_heads=4)

    def test_profiling_without_extra_trace_reads(self):
        self.run_pipeline(compression=1, top_r=128, kv_heads=4, profile=True)

    def test_modeled_budgets_reach_device_without_trace_read(self):
        self.run_pipeline(compression=1, top_r=128, kv_heads=4, profile=True, modeled=True)

    def run_pipeline(self, compression, top_r, scale=128**-0.5, batch_size=2, kv_heads=2,
                     profile=False, modeled=False):
        profiler = PhaseProfiler() if profile else None
        layers = 1 if batch_size == 4 or kv_heads == 4 else 2  # Fit the fixture's 3 MiB KV pool.
        with tempfile.TemporaryDirectory(prefix="vllm-cemu-") as directory, ExitStack() as stack:
            harness = harness_module.FdmCsdPipelineTest()
            harness.placement = "balanced"
            device, _ = harness.start_device(directory, stack)
            lock = RLock()

            def locked(fn):
                def call(*args, **kwargs):
                    with lock:
                        return fn(*args, **kwargs)
                return call

            harness.call, harness.memory = locked(harness.call), locked(harness.memory)
            harness.call(0, 0x21, (1 | (1 << 16), 0, 0, 0, 0, 0), 3)
            lengths = [4096, 524288, 524288, 524288, 4096, 4096, 4096, 4096, 16384, 4096, 4096]
            trace_reads, executions = [], []

            class Transport:
                def __init__(self, slot):
                    self.addresses = (np.cumsum([0] + lengths[:-1]) + slot * sum(lengths)).tolist()
                    descriptor = b"".join(struct.pack("<IIQQQ", 2, size, address, 0, 0)
                                          for size, address in zip(lengths, self.addresses))
                    self.rsid = harness.call(0, 0x21, (0, 11, 0, 0, 0, 0), 3, descriptor)

                def write_tensor(self, index, tensor):
                    harness.memory(5, self.addresses[index], tensor.tobytes())

                def read_tensor(self, index, shape, dtype):
                    if index == 10:
                        trace_reads.append(index)
                    data = harness.memory(2, self.addresses[index], math.prod(shape) * np.dtype(dtype).itemsize)
                    return np.frombuffer(data, dtype).reshape(shape).copy()

                def read_tensor_into(self, index, output):
                    np.copyto(output, self.read_tensor(index, output.shape, output.dtype))
                    return output

                def execute(self, cparam1, metadata):
                    executions.append(metadata)
                    harness.call(1, 1, (cparam1 & 0xffffffff, cparam1 >> 32, 0, 0, 0, 0),
                                 3, metadata, command2=1 | (self.rsid << 16), command4=len(metadata))

                def close(self):
                    pass

            store = stack.enter_context(harness.make_store(directory, batch_size, layers, kv_heads, 272, 1, np.float16))
            layout = SparfKvCacheLayout(KvLayoutConfig(layers, 272, batch_size, kv_heads, 128, np.float16))
            slot = CemuSparfDevice(
                layout, "unused", "unused", "unused", directory, "unused", 4, top_r,
                compression_ratio=compression, autonomous=True, grouped_store=store,
                attention_scale=scale, read_trace=False, cuda_target=False, ragged=True,
                approximate_runtime_ns=100000 if modeled else 0,
                exact_qk_runtime_ns=100000 if modeled else 0,
                pv_runtime_ns=100000 if modeled else 0,
                profiler=profiler)
            slot._device = Transport(0)
            stack.callback(slot.close)
            config = Config(directory, "unused", "unused", request_capacity=batch_size,
                            max_tokens=272, compression_ratio=compression)
            session = Session(config, layers, 4, kv_heads, 128, scale, store=store, device=slot,
                              profiler=profiler)
            rng = np.random.default_rng(1729)
            history = {}
            prompts = dict(zip(["short", "long", "medium", "one"][:batch_size], [15, 255, 31, 1]))
            initial = session.plan(list(prompts), [0] * batch_size, list(prompts.values()), list(prompts.values()))
            append_count = batch_size * layers
            for w in initial:
                for layer in range(layers):
                    k = (rng.standard_normal((w.count, kv_heads, 128)) * .2).astype(np.float16)
                    v = rng.standard_normal(k.shape).astype(np.float16)
                    session.append(w, layer, k, v)
                    history[(w.request_id, layer)] = (k, v)
            # Reverse the row order, crossing token-page and channel-group tails.
            for step in range(3):
                names = list(reversed(prompts)) if step < 2 else ["long"]
                work = session.plan(names, [prompts[n] + step for n in names],
                                    [prompts[n] for n in names], [1] * len(names))
                self.assertEqual([w.slot for w in work], [list(prompts).index(n) for n in names])
                append_count += len(work) * layers
                for layer in range(layers):
                    expected, queries = [], []
                    for w in work:
                        k = (rng.standard_normal((1, kv_heads, 128)) * .2).astype(np.float16)
                        v = rng.standard_normal(k.shape).astype(np.float16)
                        session.append(w, layer, k, v)
                        old_k, old_v = history[(w.request_id, layer)]
                        all_k, all_v = np.concatenate((old_k, k)), np.concatenate((old_v, v))
                        history[(w.request_id, layer)] = (all_k, all_v)
                        # Unique FP16 magnitudes avoid unspecified top-r tie
                        # choices between the NumPy and device selectors.
                        q = (np.stack([rng.permutation(np.arange(1, 129))
                                       for _ in range(4)]).reshape(1, 4, 128) / 64)
                        q = (q * rng.choice([-1, 1], size=q.shape)).astype(np.float16)
                        expected.append(sparf_attention(q.astype(np.float32) * scale,
                            all_k[:, None], all_v[:, None], top_r, math.ceil(w.end / compression)).output)
                        queries.append(q)
                    before = len(executions)
                    query_batch = np.concatenate(queries)
                    # Exercise both allocating legacy callers and caller-owned
                    # outputs with real device results after batch reorder/shrink.
                    target = np.empty_like(query_batch) if step != 1 else None
                    result = session.decode_batch(work, layer, query_batch, output=target)
                    if target is not None:
                        self.assertIs(result, target)
                    self.assertEqual(len(executions), before + 1)
                    metadata = executions[-1]
                    self.assertEqual(struct.unpack_from("<I", metadata, 36)[0], len(work))
                    self.assertEqual(len(metadata), 112 + 24 + 20 * len(work))
                    for i, reference in enumerate(expected):
                        np.testing.assert_allclose(result[i:i + 1], reference, atol=3e-3, rtol=3e-3)
                    if step == 0 and layer == 0 and not profile:
                        # Rejected commands must not leak reader leases or poison
                        # subsequent valid APPEND/EXEC. Exercise device validation.
                        mutations = [(136, batch_size), (140, 0), (144, 0), (148, 1), (152, 1),
                                     (156, work[0].slot)]
                        for offset, value in mutations:
                            bad = bytearray(metadata)
                            struct.pack_into("<I", bad, offset, value)
                            harness.call(1, 1, (0x53505246, 0x43454d55, 0, 0, 0, 0),
                                         3, bad, command2=1 | (slot._device.rsid << 16),
                                         command4=len(bad), rejected=True)
                        harness.call(1, 1, (0x53505246, 0x43454d55, 0, 0, 0, 0),
                                     3, metadata[:-1], command2=1 | (slot._device.rsid << 16),
                                     command4=len(metadata) - 1, rejected=True)
            self.assertEqual(len(executions), layers * 3)  # One EXEC per layer per step, including B1.
            self.assertEqual(trace_reads, [])
            if profiler is not None:
                metrics = {m.name: m for m in profiler.snapshot()}
                self.assertEqual(metrics["decode.execute_wait_wall"].count, layers * 3)
                self.assertEqual(metrics["decode.output_read_wall"].count, layers * 3)
                self.assertGreater(metrics["prefill.append_wall"].total_ns, 0)
                records = [json.loads(line) for line in harness.timeline.read_text().splitlines()]
                report = device_report(records, 1, layers * 3)
                self.assertTrue(report["available"], report)
                self.assertGreater(report["device_command_union_seconds"], 0)
                self.assertLessEqual(report["device_percent_of_decode_command_window"], 100)
                if modeled:
                    self.assertGreater(report["compute_union_seconds"], 0)
                    self.assertTrue(all(struct.unpack_from("<3I", metadata, 8) == (100000,) * 3
                                        for metadata in executions))
            self.assertEqual(store._append_commands, append_count)
            with self.assertRaisesRegex(ValueError, "duplicate"):
                session.append(work[0], 0, k, v)
            session.finish(["short"])
            with self.assertRaisesRegex(RuntimeError, "capacity exhausted"):
                session.plan(["short"], [0], [1], [1])
            session.close()
            stats = session.final_statistics
            self.assertEqual(stats["sealed"], 1)
            self.assertEqual(stats["failed"], 0)

    def test_one_token_prompt_and_rejected_chunking(self):
        session = Session(Config("unused", "unused", "unused"), 1, 4, 2, 128, 128**-.5)
        self.assertTrue(session.plan(["one"], [0], [1], [1])[0].prefill)
        with self.assertRaisesRegex(ValueError, "full prefill"):
            session.plan(["chunk"], [0], [8], [4])

    def test_failed_append_cannot_publish_mean_or_be_replayed(self):
        class RejectingStore:
            def append_batch(self, *args):
                raise OSError("APPEND failed")

        session = Session(Config("unused", "unused", "unused"), 1, 4, 2, 128, 128**-.5,
                          store=RejectingStore())
        work = session.plan(["one"], [0], [1], [1])[0]
        kv = np.zeros((1, 2, 128), dtype=np.float16)
        with self.assertRaisesRegex(OSError, "APPEND failed"):
            session.append(work, 0, kv, kv)
        with self.assertRaisesRegex(RuntimeError, "failed"):
            session.append(work, 0, kv, kv)
        with self.assertRaisesRegex(RuntimeError, "failed"):
            session.decode_batch([work], 0, np.zeros((1, 4, 128), dtype=np.float16))


if __name__ == "__main__":
    unittest.main()
