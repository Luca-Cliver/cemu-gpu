import math
from contextlib import ExitStack
from pathlib import Path
import struct
import subprocess
import tempfile
import time
import unittest

import numpy as np

from test_instattention_native_qtest import NativeQtest, ROOT, DATA, BAR
from cemu_flexgen import KvLayoutConfig, SparfKvCacheLayout, sparf_attention
from cemu_flexgen.native_kv_store import NativeKvStore
from cemu_flexgen.sparf_group_layout import SparfGroupLayout
from cemu_flexgen.sparf_workflow_abi import SPARF_WORKFLOW_COMMAND, SparfWorkflowTrace


class NativeCsdPipelineTest(unittest.TestCase):
    def start_device(self, directory, stack):
        kernel = Path(directory) / "sparf.so"
        subprocess.run(["c++", "-std=c++17", "-shared", "-fPIC", "-O2",
                        str(ROOT / "tests/cemu/kernel/sparf_attention.cc"), "-o", str(kernel)], check=True)
        device = NativeQtest(directory, ROOT / "tests/cemu/instattention/experiments/configs/native_csd_smoke.json")
        stack.callback(device.close)
        self.device = device
        packet = str(kernel).encode() + b"\0sparf_attention\0"
        self.call(1, 0x22, (1 | (2 << 16), len(packet), 0, 0, len(packet), 0), 3, packet)
        self.call(0, 0x23, (1 | (1 << 16), 0, 0, 0, 0, 0), 3)
        lengths = [4096, 1048576, 1048576, 1048576, 4096, 4096, 4096, 4096, 16384, 4096, 4096]
        offsets = np.cumsum([0] + lengths[:-1]).tolist()
        descriptor = b"".join(struct.pack("<IIQQQ", 2, length, offset, 0, 0)
                              for length, offset in zip(lengths, offsets))
        self.call(0, 0x21, (0, 11, 0, 0, 0, 0), 3, descriptor)
        return device, offsets

    def test_native_nvme_fdm_mrs_attention_and_reset(self):
        with tempfile.TemporaryDirectory(prefix="native-csd-") as directory, ExitStack() as stack:
            device, offsets = self.start_device(directory, stack)
            self.call(0, 0x21, (0, 1, 0, 0, 0, 0), 3,
                      struct.pack("<IIQQQ", 1, 4096, 0, 0, 0), rejected=True)
            self.call(1, 1, (0, 0, 0, 0, 0, 0), 2, rejected=True)

            class Transport:
                def __init__(self, path): pass
                def __call__(self, *args, **kwargs): return device(*args, **kwargs)
                def close(self): pass

            store = NativeKvStore(SparfGroupLayout(128, 8, dtype=np.float16),
                                  2, 1, 2, 300, 2, transport_factory=Transport).open()
            stack.callback(store.close)
            layout = SparfKvCacheLayout(KvLayoutConfig(1, 300, 2, 2, 128, np.float16))
            generator = np.random.default_rng(442)
            keys = (generator.normal(size=(259, 4, 128)) / math.sqrt(128)).astype(np.float16)
            values = generator.normal(size=keys.shape).astype(np.float16)
            query = np.stack([generator.permutation(np.arange(1, 129)) for vector in range(8)]).reshape(2, 4, 128).astype(np.float16) / 1024
            previous = 0
            for tokens in (15, 17, 259):
                store.append_batch(0, previous, keys[previous:tokens], values[previous:tokens])
                previous = tokens
                for ratio in (1, 8):
                    top_k = math.ceil(tokens / ratio)
                    shaped_k = keys[:tokens].reshape(tokens, 2, 2, 128)
                    shaped_v = values[:tokens].reshape(tokens, 2, 2, 128)
                    mean = shaped_v.astype(np.float64).mean(axis=0).astype(np.float16)[:, [0, 0, 1, 1]]
                    self.memory(5, offsets[0], query.tobytes())
                    self.memory(5, offsets[4], mean.tobytes())
                    metadata = store.pack_workflow(layout, 0, tokens, 4, 16, top_k, 0,
                        approximate_runtime_ns=1000, exact_qk_runtime_ns=300000, pv_runtime_ns=1000)
                    self.call(1, 1, (SPARF_WORKFLOW_COMMAND & 0xffffffff, SPARF_WORKFLOW_COMMAND >> 32,
                                     0, 0, 0, 0), 3, metadata,
                              command2=1 | (1 << 16), command4=len(metadata))
                    output = np.frombuffer(self.memory(2, offsets[9], query.nbytes), np.float16).reshape(query.shape)
                    expected = sparf_attention(query, shaped_k, shaped_v, 16, top_k).output
                    np.testing.assert_allclose(output, expected, atol=3e-3, rtol=3e-3)
                    trace = SparfWorkflowTrace.unpack(self.memory(2, offsets[10], 72))
                    self.assertGreaterEqual(trace.total_model_ns, 302000)
                    if tokens == 15:
                        self.assertEqual(trace.channel_read_model_ns, 0)
                    if tokens == 259:
                        self.assertGreater(trace.token_k_read_model_ns, 0)
            store.drain()
            stats = store.statistics()
            self.assertEqual(stats["sealed"], 1)
            self.assertGreater(stats["write_bytes"], 0)
            self.assertGreater(stats["read_bytes"], 0)
            metadata = store.pack_workflow(layout, 0, tokens, 4, 16, top_k, 0,
                approximate_runtime_ns=1000000000, exact_qk_runtime_ns=300000, pv_runtime_ns=1000)
            device.write(DATA, metadata)
            device.submit(1, 1, (SPARF_WORKFLOW_COMMAND & 0xffffffff, SPARF_WORKFLOW_COMMAND >> 32,
                                 0, 0, 0, 0), namespace=3,
                          command2=1 | (1 << 16), command4=len(metadata))
            device.command("clock_step 100000")
            time.sleep(0.1)
            self.assertIsNone(device.completion(1))
            device.initialize()
            self.assertEqual(int(device.command(f"readl {BAR + 0x1c:#x}"), 0), 1)
            self.call(0, 0x21, (1 | (1 << 16), 0, 0, 0, 0, 0), 3)
            self.call(0, 0x23, (1, 0, 0, 0, 0, 0), 3)
            self.call(1, 0x22, (1 | (2 << 16) | (1 << 24), 0, 0, 0, 0, 0), 3)

    def test_opt_prefill_decode_through_native_storage_and_attention(self):
        import torch
        from cemu_flexgen.cemu_sparf_device import CemuSparfDevice
        from flexgen_adapter.sparf_attention import SparfGroupedAttentionBackend
        from opt_runtime import (OptCheckpointLoader, OptConfig, OptDecodeRunner,
                                 OptGenerationRunner, OptPrefillRunner, OptTorchAttentionBackend)
        from test_opt_runtime import make_state

        with tempfile.TemporaryDirectory(prefix="native-opt-") as directory, ExitStack() as stack:
            device, offsets = self.start_device(directory, stack)
            test = self

            class Transport:
                def __init__(self, path): pass
                def __call__(self, *args, **kwargs): return device(*args, **kwargs)
                def close(self): pass

            class FdmTransport:
                def write_tensor(self, index, tensor):
                    test.memory(5, offsets[index], tensor.tobytes())

                def read_tensor(self, index, shape, dtype):
                    data = test.memory(2, offsets[index], math.prod(shape) * np.dtype(dtype).itemsize)
                    return np.frombuffer(data, dtype).reshape(shape).copy()

                def execute(self, cparam1, metadata):
                    test.call(1, 1, (cparam1 & 0xffffffff, cparam1 >> 32, 0, 0, 0, 0),
                              3, metadata, command2=1 | (1 << 16), command4=len(metadata))
                    return 0

                def close(self): pass

            config = OptConfig(name="native-opt-smoke", num_hidden_layers=2,
                max_position_embeddings=160, hidden_size=256, num_attention_heads=2,
                ffn_dim=320, vocab_size=32, word_embed_proj_dim=256, dtype=torch.float32)
            torch.save(make_state(config), Path(directory) / "pytorch_model.bin")
            loader = stack.enter_context(OptCheckpointLoader(config, directory, device=torch.device("cpu")))
            store = stack.enter_context(NativeKvStore(SparfGroupLayout(128, 8, dtype=np.float32),
                2, 2, 2, 160, 2, transport_factory=Transport))
            layout = SparfKvCacheLayout(KvLayoutConfig(2, 160, 2, 2, 128, np.float32))
            attention = CemuSparfDevice(layout, "unused", "unused", "unused", directory, "unused",
                query_heads=2, top_r=128, compression_ratio=1, cuda_target=False,
                autonomous=True, grouped_store=store)
            attention._device = FdmTransport()
            backend = stack.enter_context(SparfGroupedAttentionBackend(layout, store, attention, 0))
            prompt = (torch.arange(254).reshape(2, 127) % 28 + 4).long()
            prefill = OptPrefillRunner(config, loader, kv_writer=backend).run(prompt, collect_kv_cache=True)
            reference_backend = OptTorchAttentionBackend(config, prefill.kv_cache)
            expected = OptGenerationRunner(OptDecodeRunner(config, loader, reference_backend)).run(
                prefill.next_token_ids, start_position=127, decode_steps=3, collect_layer_outputs=True)
            actual = OptGenerationRunner(OptDecodeRunner(config, loader, backend)).run(
                prefill.next_token_ids, start_position=127, decode_steps=3, collect_layer_outputs=True)
            torch.testing.assert_close(actual.token_ids, expected.token_ids, rtol=0, atol=0)
            for actual_step, expected_step in zip(actual.steps, expected.steps):
                torch.testing.assert_close(actual_step.decode_result.logits,
                    expected_step.decode_result.logits, rtol=2e-5, atol=2e-5)
                torch.testing.assert_close(actual_step.decode_result.hidden_states,
                    expected_step.decode_result.hidden_states, rtol=2e-5, atol=2e-5)
            self.assertEqual(attention.decode_count, 6)
            backend.drain()
            stats = store.statistics()
            self.assertEqual(stats["accepted_stream0"], 130)
            self.assertEqual(stats["sealed"], 1)
            self.assertEqual(stats["failed"], 0)
            self.assertGreater(stats["read_bytes"], 0)
            self.assertGreater(stats["write_bytes"], 0)

    def call(self, queue, opcode, fields, namespace, packet=None, rejected=False, **kwargs):
        if packet is not None:
            self.device.write(DATA, packet)
        self.device.submit(queue, opcode, fields, namespace=namespace, **kwargs)
        status, result = self.device.wait(queue)
        self.assertEqual(bool(status), rejected, f"opcode={opcode:#x} ns={namespace} status={status:#x}; {self.device.read_log()}")
        return result

    def memory(self, opcode, offset, data):
        size = len(data) if opcode == 5 else data
        output = bytearray()
        for start in range(0, size, 4096):
            count = min(4096, size - start)
            address = offset + start
            self.call(1, opcode, (address & 0xffffffff, address >> 32, count, 0, 0, 0), 2,
                      data[start:start + count] if opcode == 5 else None)
            if opcode == 2:
                output.extend(self.device.read(DATA, count))
        return bytes(output)


if __name__ == "__main__":
    unittest.main()
