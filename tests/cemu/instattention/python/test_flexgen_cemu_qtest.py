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
import torch

from cemu_flexgen import KvLayoutConfig, SparfKvCacheLayout
from cemu_flexgen.cemu_sparf_device import CemuSparfDevice
from cemu_flexgen.cemu_sparf_slot_scheduler import CemuSparfSlotScheduler
from flexgen_adapter.framework import create_model
from flexgen_adapter.framework_environment import make_environment, make_policy
from flexgen_adapter.framework_import import load_framework
from flexgen_adapter.sparf_attention import SparfGroupedAttentionBackend
import test_native_csd_pipeline as harness_module
from test_flexgen_framework import checkpoint, reference


class FlexGenCemuQtest(unittest.TestCase):
    def test_flexgen_fdm_async_microbatches(self):
        with patch.dict(os.environ, CEMU_FDM_ASYNC_AGGREGATE="1"):
            self.test_flexgen_fdm_half_precision()

    def test_flexgen_fdm_pipeline(self):
        self.run_pipeline(harness_module.FdmCsdPipelineTest)

    def test_flexgen_native_pipeline(self):
        self.run_pipeline(harness_module.NativeCsdPipelineTest)

    def test_flexgen_fdm_microbatches(self):
        self.run_pipeline(harness_module.FdmCsdPipelineTest, microbatches=2)

    def test_flexgen_native_microbatches(self):
        self.run_pipeline(harness_module.NativeCsdPipelineTest, microbatches=2)

    def test_flexgen_fdm_sparse(self):
        self.run_pipeline(harness_module.FdmCsdPipelineTest, microbatches=2, compression=8, top_r=16)

    def test_flexgen_native_sparse(self):
        self.run_pipeline(harness_module.NativeCsdPipelineTest, microbatches=2, compression=8, top_r=16)

    def test_flexgen_fdm_half_precision(self):
        self.run_pipeline(harness_module.FdmCsdPipelineTest, microbatches=2, compression=8, top_r=16,
                          dtype=torch.float16, prompt_length=255, maximum=272)

    def run_pipeline(self, harness_type, microbatches=1, compression=1, top_r=128,
                     dtype=torch.float32, prompt_length=127, maximum=160):
        threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, threads)
        with tempfile.TemporaryDirectory(prefix="flexgen-cemu-") as directory, ExitStack() as stack:
            harness = harness_type()
            harness.placement = "balanced"
            device, offsets = harness.start_device(directory, stack)
            transport_lock = RLock()

            def locked(function):
                def call(*args, **kwargs):
                    with transport_lock:
                        return function(*args, **kwargs)
                return call

            class SerializedDevice:
                def __getattr__(self, name):
                    return getattr(device, name)

                def __call__(self, *args, **kwargs):
                    with transport_lock:
                        return device(*args, **kwargs)

            harness.device = SerializedDevice()
            harness.call = locked(harness.call)
            harness.memory = locked(harness.memory)
            harness.call(0, 0x21, (1 | (1 << 16), 0, 0, 0, 0, 0), 3)
            lengths = [4096, 524288, 524288, 524288, 4096, 4096, 4096, 4096, 16384, 4096, 4096]
            range_sets = []
            for slot in range(2):
                addresses = (np.cumsum([0] + lengths[:-1]) + slot * sum(lengths)).tolist()
                descriptor = b"".join(struct.pack("<IIQQQ", 2, length, address, 0, 0)
                                      for length, address in zip(lengths, addresses))
                rsid = harness.call(0, 0x21, (0, 11, 0, 0, 0, 0), 3, descriptor)
                range_sets.append((rsid, addresses))

            class FdmTransport:
                def __init__(self, rsid, addresses):
                    self.rsid = rsid
                    self.addresses = addresses

                def write_tensor(self, index, tensor):
                    harness.memory(5, self.addresses[index], tensor.tobytes())

                def read_tensor(self, index, shape, dtype):
                    data = harness.memory(2, self.addresses[index], math.prod(shape) * np.dtype(dtype).itemsize)
                    return np.frombuffer(data, dtype).reshape(shape).copy()

                def execute(self, cparam1, metadata):
                    harness.call(1, 1, (cparam1 & 0xffffffff, cparam1 >> 32, 0, 0, 0, 0),
                                 3, metadata, command2=1 | (self.rsid << 16), command4=len(metadata))
                    return 0

                def close(self):
                    pass

            original, config, source, weights = checkpoint(directory, dimension=256, positions=maximum, dtype=dtype)
            prompt = (np.arange(2 * prompt_length).reshape(2, prompt_length) % 28 + 4).astype(np.int64)
            expected, logits = reference(original, source, prompt, 3, compression, top_r)
            batch = 2 // microbatches
            store = stack.enter_context(harness.make_store(directory, 2, 2, 2, maximum, batch, config.dtype))
            layout = SparfKvCacheLayout(KvLayoutConfig(2, maximum, batch, 2, 128, config.dtype))
            slots = []
            for rsid, addresses in range_sets:
                attention = CemuSparfDevice(layout, "unused", "unused", "unused", directory, "unused",
                    query_heads=2, top_r=top_r, compression_ratio=compression, cuda_target=False,
                    autonomous=True, grouped_store=store)
                attention._device = FdmTransport(rsid, addresses)
                slots.append(attention)
            scheduler = CemuSparfSlotScheduler(slots)
            backends = [stack.enter_context(SparfGroupedAttentionBackend(layout, store, scheduler,
                        microbatch * 2 * batch * 2)) for microbatch in range(microbatches)]
            env = make_environment("cpu")
            observed = []
            env.gpu.output_observer = lambda value: observed.append(value.clone())
            model = create_model(config, env, weights, make_policy(batch, microbatches), backends)
            stack.callback(model.delete_all_weights)
            stack.callback(model.cemu_session.close)
            load_framework().timers("generate").reset()
            output = model.generate(prompt, max_new_tokens=4)
            model.cemu_session.drain()
            np.testing.assert_array_equal(output[:, prompt_length:], expected)
            for step, expected_logits in enumerate(logits):
                actual = torch.cat(observed[step * microbatches:(step + 1) * microbatches])
                tolerance = 3e-3 if dtype == torch.float16 else 3e-5
                torch.testing.assert_close(actual, expected_logits, rtol=tolerance, atol=tolerance)
            stats = store.statistics()
            if harness_type is harness_module.FdmCsdPipelineTest:
                self.assertEqual(stats["client_append_commands"], 8 * microbatches)
                self.assertEqual(stats["append_commands"], 8 * microbatches)
            self.assertEqual(stats["accepted_stream0"], prompt_length + 3)
            self.assertEqual(stats["sealed"], 1)
            self.assertEqual(stats["failed"], 0)
            self.assertEqual(sum(slot.decode_count for slot in slots), 6 * microbatches)
            self.assertGreater(stats["read_bytes"], 0)
            self.assertGreater(stats["write_bytes"], 0)
            events = [json.loads(line) for line in harness.timeline.read_text().splitlines()]
            self.assertEqual(sum(event.get("event") == "complete" for event in events), 6 * microbatches)


if __name__ == "__main__":
    unittest.main()
