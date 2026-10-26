import json
import math
import os
from contextlib import ExitStack
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

import numpy as np

from test_instattention_native_qtest import NativeQtest, ROOT, DATA, BAR
from cemu_flexgen import KvLayoutConfig, SparfKvCacheLayout, sparf_attention
from cemu_flexgen.native_kv_store import NativeKvStore
from cemu_flexgen.sparf_group_layout import SparfGroupLayout
from cemu_flexgen.sparf_workflow_abi import SPARF_WORKFLOW_COMMAND, SparfWorkflowTrace


class NativeCsdPipelineTest(unittest.TestCase):
    def make_store(self, directory, batches, layers, heads, max_tokens, microbatch_size, dtype):
        device = self.device

        class Transport:
            def __init__(self, path): pass
            def __call__(self, *args, **kwargs): return device(*args, **kwargs)
            def close(self): pass

        return NativeKvStore(SparfGroupLayout(128, 8, dtype=dtype), batches, layers,
                             heads, max_tokens, microbatch_size, transport_factory=Transport)

    def start_device(self, directory, stack, read_diagnostics=False):
        cuda = os.environ.get("CEMU_NATIVE_QTEST_CUDA", "0") == "1"
        if cuda:
            kernel = ROOT / "tests/cemu/build/sparf_attention_devptr.so"
            if not kernel.is_file():
                raise FileNotFoundError(f"build CUDA operator first: {kernel}")
        else:
            kernel = Path(directory) / "sparf.so"
            subprocess.run(["c++", "-std=c++17", "-shared", "-fPIC", "-O2",
                            str(ROOT / "tests/cemu/kernel/sparf_attention.cc"), "-o", str(kernel)], check=True)
        self.timeline = Path(directory) / "device-timeline.jsonl"
        with patch.dict(os.environ, {
            "CEMU_SPARF_AUTONOMOUS_TIMELINE": str(self.timeline),
            "CEMU_SPARF_READ_DIAGNOSTICS": str(int(read_diagnostics)),
        }):
            device = NativeQtest(directory, ROOT / "tests/cemu/instattention/experiments/configs/native_csd_smoke.json",
                                 cuda=cuda, placement=getattr(self, "placement", "legacy"))
        stack.callback(device.close)
        self.assertIn(f"Native KV block placement: {getattr(self, 'placement', 'legacy')}",
                      device.read_log())
        self.device = device
        packet = str(kernel).encode() + b"\0sparf_attention\0"
        self.call(1, 0x22, (1 | (2 << 16) | (int(cuda) << 29), len(packet), 0, 0, len(packet), 0), 3, packet)
        self.call(0, 0x23, (1 | (1 << 16), 0, 0, 0, 0, 0), 3)
        lengths = [4096, 1048576, 1048576, 1048576, 4096, 4096, 4096, 4096, 16384, 4096, 4096]
        offsets = np.cumsum([0] + lengths[:-1]).tolist()
        descriptor = b"".join(struct.pack("<IIQQQ", 2, length, offset, 0, 0)
                              for length, offset in zip(lengths, offsets))
        self.call(0, 0x21, (0, 11, 0, 0, 0, 0), 3, descriptor)
        return device, offsets

    def test_balanced_native_attention_and_reset(self):
        with patch.object(self, "placement", "balanced", create=True):
            self.test_native_nvme_fdm_mrs_attention_and_reset()

    def test_balanced_opt_prefill_decode(self):
        with patch.object(self, "placement", "balanced", create=True):
            self.test_opt_prefill_decode_through_native_storage_and_attention()

    def test_native_nvme_fdm_mrs_attention_and_reset(self):
        with tempfile.TemporaryDirectory(prefix="native-csd-") as directory, ExitStack() as stack:
            device, offsets = self.start_device(directory, stack, read_diagnostics=True)
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
            query = (np.stack([generator.permutation(np.arange(1, 129)) for vector in range(8)])
                     .reshape(2, 4, 128) / 1024).astype(np.float16)
            self.assertEqual(query.dtype, layout.config.dtype)
            previous = 0
            for tokens in (15, 17, 259):
                store.append_batch(0, previous, keys[previous:tokens], values[previous:tokens])
                previous = tokens
                if tokens == 259:
                    # Direct assembly no longer stalls on tiny raw rings. Wait
                    # explicitly for 12 full blocks (2 batches x (2 K + 2 V +
                    # 2 channel-K)); leave the last three tokens in DRAM.
                    deadline = time.monotonic() + 5
                    while store.statistics()["write_blocks"] != 12:
                        self.assertLess(time.monotonic(), deadline)
                        device.command("clock_step 1000000")
                        time.sleep(0.001)
                    self.assertGreater(store.statistics()["staged_pages"], 0)
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
            records = [json.loads(line) for line in self.timeline.read_text().splitlines()]
            completed = [record for record in records if record.get("event") == "complete"]
            self.assertEqual(len(completed), 6)
            submitted_bytes = 0
            for record in completed:
                self.assertTrue(record["success"])
                self.assertEqual(len(record["reads"]), 3)
                for read in record["reads"]:
                    self.assertIn("io", read, "native production admission disabled read diagnostics")
                    observation = read["io"]
                    self.assertIn("full_scan", observation)
                    self.assertIn("unique_flash_bytes", observation)
                    self.assertEqual(sum(observation["lun_pages"]), observation["pages"])
                    submitted_bytes += observation["submitted_bytes"]
            self.assertEqual(submitted_bytes, stats["read_bytes"])
            loss_path = Path(directory) / "read-loss.json"
            analysis = subprocess.run([
                sys.executable,
                str(ROOT / "tests/cemu/instattention/experiments/analyze_sparf_autonomous.py"),
                str(self.timeline), "--latest-session", "--require-read-diagnostics",
                "--require-native-read-reference", "--read-loss-summary", str(loss_path),
            ], capture_output=True, text=True)
            self.assertEqual(analysis.returncode, 0, analysis.stdout + analysis.stderr)
            self.assertTrue(loss_path.is_file())
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
            store = stack.enter_context(self.make_store(directory, 2, 2, 2, 160, 2, np.float32))
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

            records = [json.loads(line) for line in self.timeline.read_text().splitlines()]
            completed = [record for record in records if record.get("event") == "complete"]
            self.assertEqual(len(completed), 6)
            for record in completed:
                for read in record["reads"]:
                    self.assertNotIn("io", read)

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


class FdmCsdPipelineTest(unittest.TestCase):
    start_device = NativeCsdPipelineTest.start_device
    call = NativeCsdPipelineTest.call
    memory = NativeCsdPipelineTest.memory
    test_opt_prefill_decode_through_fdm_storage_and_attention = NativeCsdPipelineTest.test_opt_prefill_decode_through_native_storage_and_attention

    def test_balanced_fdm_attention_boundaries_gqa_dedup_and_identity(self):
        with patch.object(self, "placement", "balanced", create=True):
            self.test_fdm_attention_boundaries_gqa_dedup_and_identity()

    def test_balanced_opt_prefill_decode(self):
        with patch.object(self, "placement", "balanced", create=True):
            self.test_opt_prefill_decode_through_fdm_storage_and_attention()

    def test_fdm_attention_boundaries_gqa_dedup_and_identity(self):
        with tempfile.TemporaryDirectory(prefix="fdm-attention-") as directory, ExitStack() as stack:
            device, offsets = self.start_device(directory, stack, read_diagnostics=True)
            store = stack.enter_context(self.make_store(directory, 2, 1, 2, 300, 2, np.float16))
            layout = SparfKvCacheLayout(KvLayoutConfig(1, 300, 2, 2, 128, np.float16))
            generator = np.random.default_rng(442)
            keys = (generator.normal(size=(259, 4, 128)) / math.sqrt(128)).astype(np.float16)
            values = generator.normal(size=keys.shape).astype(np.float16)
            query = (np.stack([generator.permutation(np.arange(1, 129)) for vector in range(8)])
                     .reshape(2, 4, 128) / 1024).astype(np.float16)
            self.assertEqual(query.dtype, layout.config.dtype)
            previous = 0
            for tokens in (15, 16, 253, 256, 259):
                store.append_batch(0, previous, keys[previous:tokens], values[previous:tokens])
                previous = tokens
                if tokens == 259:
                    deadline = time.monotonic() + 5
                    while store.statistics()["inflight_blocks"] or store.statistics()["write_blocks"] < 12:
                        self.assertLess(time.monotonic(), deadline)
                        device.command("clock_step 1000000")
                        time.sleep(0.001)
                for ratio in (1, 8):
                    top_k = math.ceil(tokens / ratio)
                    shaped_k = keys[:tokens].reshape(tokens, 2, 2, 128)
                    shaped_v = values[:tokens].reshape(tokens, 2, 2, 128)
                    mean = shaped_v.astype(np.float64).mean(axis=0).astype(np.float16)[:, [0, 0, 1, 1]]
                    self.memory(5, offsets[0], query.tobytes())
                    self.memory(5, offsets[4], mean.tobytes())
                    metadata = store.pack_workflow(layout, 0, tokens, 4, 16, top_k, 0,
                        approximate_runtime_ns=1000, exact_qk_runtime_ns=300000, pv_runtime_ns=1000)
                    for field, replacement in ((112, store.session_id ^ 1), (120, 2)):
                        invalid = bytearray(metadata)
                        struct.pack_into("<Q", invalid, field, replacement)
                        self.call(1, 1, (SPARF_WORKFLOW_COMMAND & 0xffffffff, SPARF_WORKFLOW_COMMAND >> 32,
                                         0, 0, 0, 0), 3, invalid, rejected=True,
                                  command2=1 | (1 << 16), command4=len(invalid))
                    self.call(1, 1, (SPARF_WORKFLOW_COMMAND & 0xffffffff, SPARF_WORKFLOW_COMMAND >> 32,
                                     0, 0, 0, 0), 3, metadata,
                              command2=1 | (1 << 16), command4=len(metadata))
                    output = np.frombuffer(self.memory(2, offsets[9], query.nbytes), np.float16).reshape(query.shape)
                    expected = sparf_attention(query, shaped_k, shaped_v, 16, top_k).output
                    np.testing.assert_allclose(output, expected, atol=3e-3, rtol=3e-3)
                    if tokens == 15:
                        self.assertEqual(store.statistics()["read_bytes"], 0)
            store.drain()
            before = store.statistics()["read_bytes"]
            self.call(1, 1, (SPARF_WORKFLOW_COMMAND & 0xffffffff, SPARF_WORKFLOW_COMMAND >> 32,
                             0, 0, 0, 0), 3, metadata,
                      command2=1 | (1 << 16), command4=len(metadata))
            output = np.frombuffer(self.memory(2, offsets[9], query.nbytes), np.float16).reshape(query.shape)
            np.testing.assert_allclose(output, expected, atol=3e-3, rtol=3e-3)
            stats = store.statistics()
            self.assertEqual(stats["staged_pages"], 0)
            self.assertEqual(stats["token_k"], 259)
            self.assertEqual(stats["channel_k"], 259)
            self.assertEqual(stats["token_v"], 259)
            self.assertGreater(stats["read_bytes"], before)
            completed = [record for record in map(json.loads, self.timeline.read_text().splitlines())
                         if record.get("event") == "complete"]
            self.assertEqual(len(completed), 11)
            submitted = 0
            for record in completed:
                self.assertTrue(record["success"])
                for read in record["reads"]:
                    observation = read["io"]
                    self.assertIn("full_scan", observation)
                    self.assertEqual(sum(observation["lun_pages"]), observation["pages"])
                    self.assertLessEqual(observation["submitted_bytes"], observation["aligned_bytes_before_dedup"])
                    submitted += observation["submitted_bytes"]
            self.assertEqual(submitted, stats["read_bytes"])
            self.assertTrue(all(read["io"]["tail_bytes"] == 0 for read in completed[-1]["reads"]))

    def test_fdm_attention_lease_defers_append_and_reset_releases_binding(self):
        from test_instattention_native_qtest import IOCQ

        with tempfile.TemporaryDirectory(prefix="fdm-lease-") as directory, ExitStack() as stack:
            device, offsets = self.start_device(directory, stack)
            store = self.make_store(directory, 1, 1, 1, 32, 1, np.float16).open()
            keys = np.ones((15, 1, 128), dtype=np.float16)
            store.append_batch(0, 0, keys, keys)
            layout = SparfKvCacheLayout(KvLayoutConfig(1, 32, 1, 1, 128, np.float16))
            self.memory(5, offsets[0], keys[0].tobytes())
            self.memory(5, offsets[4], keys[0].tobytes())
            metadata = store.pack_workflow(layout, 0, 15, 1, 16, 2, 0,
                                           approximate_runtime_ns=1000000000)
            fields = (SPARF_WORKFLOW_COMMAND & 0xffffffff, SPARF_WORKFLOW_COMMAND >> 32, 0, 0, 0, 0)
            device.write(DATA, metadata)
            device.submit(1, 1, fields, namespace=3, command2=1 | (1 << 16), command4=len(metadata))
            execute_id = device.cid
            time.sleep(0.05)
            leases = struct.unpack_from("<I", self.memory(2, 4 * 1024**2, 512), 192)[0]
            self.assertEqual(leases, 1)
            self.call(0, 0x21, (1 | (store._rsid << 16), 0, 0, 0, 0, 0), 3, rejected=True)
            with patch.object(store._client, "write_range") as packed:
                store.append_batch(0, 15, keys[:1], keys[:1])
            packet = packed.call_args.args[1]
            device.write(DATA, packet)
            device.submit(1, 5, (4 * 1024**2 + 65536, 0, len(packet), 0, 0, 0), namespace=2)
            append_id = device.cid
            device.command("clock_step 100000")
            time.sleep(0.05)
            self.assertIsNone(device.completion(1))
            completed = {}
            deadline = time.monotonic() + 5
            while len(completed) < 2:
                self.assertLess(time.monotonic(), deadline)
                device.command("clock_step 100000000")
                time.sleep(0.001)
                while device.completion(1) is not None:
                    entry = device.read(IOCQ + device.heads[1] * 16, 16)
                    command_id = struct.unpack_from("<H", entry, 12)[0]
                    self.assertNotIn(command_id, completed)
                    completed[command_id] = device.wait(1)[0]
            self.assertEqual(completed, {execute_id: 0, append_id: 0})
            self.assertEqual(store.statistics()["accepted_stream0"], 16)
            self.assertEqual(store.statistics()["failed"], 0)
            metadata = store.pack_workflow(layout, 0, 16, 1, 16, 2, 0,
                                           approximate_runtime_ns=1000000000)
            device.write(DATA, metadata)
            device.submit(1, 1, fields, namespace=3, command2=1 | (1 << 16), command4=len(metadata))
            time.sleep(0.05)
            self.assertEqual(struct.unpack_from("<I", self.memory(2, 4 * 1024**2, 512), 192)[0], 1)
            device.initialize()
            self.assertIsNone(device.completion(1))
            self.assertEqual(struct.unpack_from("<I", self.memory(2, 4 * 1024**2, 512), 176)[0], 0)
            self.call(0, 0x21, (1 | (store._rsid << 16), 0, 0, 0, 0, 0), 3)
            store._client.rsid = None
            store._client.close()
            store._client = None
            store.registered = False
            store.close()

    def make_store(self, directory, batches, layers, heads, max_tokens, microbatch_size, dtype):
        from test_fdm_kv_binding import store_module

        aggregate = Path(directory) / "aggregate.so"
        subprocess.run(["cc", "-shared", "-fPIC", "-O2",
            str(ROOT / "tests/cemu/kernel/fdm_kv_aggregate.c"),
            str(ROOT / "hw/femu/fdm_kv_layout.c"), "-o", str(aggregate)], check=True)
        test = self
        offsets = (4 * 1024**2, 4 * 1024**2 + 65536)

        class Client:
            def __init__(self, *args):
                self.active = False
                self.rsid = None

            def create_memory_ranges(self, ranges):
                descriptor = b"".join(struct.pack("<IIQQQ", 2, spec[2], offset, 0, 0)
                                      for spec, offset in zip(ranges, offsets))
                self.rsid = test.call(0, 0x21, (0, 2, 0, 0, 0, 0), 3, descriptor)
                return self.rsid

            def load_program(self, name, path, entry):
                packet = str(aggregate).encode() + b"\0" + entry.encode() + b"\0"
                test.call(1, 0x22, (2 | (2 << 16), len(packet), 0, 0, len(packet), 0), 3, packet)

            def activate_program(self):
                test.call(0, 0x23, (2 | (1 << 16), 0, 0, 0, 0, 0), 3)
                self.active = True

            def bind_fdm_kv_buffer(self, wire):
                test.call(0, 0x21, (2 | (self.rsid << 16), len(wire), 2, 0, 0, 0), 3, wire)

            def write_range(self, index, payload, offset):
                test.memory(5, offsets[index] + offset, payload)

            def read_range_bytes(self, index, size, offset):
                return test.memory(2, offsets[index] + offset, size)

            def release_memory_ranges(self):
                if self.rsid is not None:
                    test.call(0, 0x21, (1 | (self.rsid << 16), 0, 0, 0, 0, 0), 3)
                    self.rsid = None

            def close(self):
                self.release_memory_ranges()
                if self.active:
                    test.call(0, 0x23, (2, 0, 0, 0, 0, 0), 3)
                    test.call(1, 0x22, (2 | (2 << 16) | (1 << 24), 0, 0, 0, 0, 0), 3)
                    self.active = False

        module = store_module(Client)
        return module.FDM_KVbuffer(SparfGroupLayout(128, 8, dtype=dtype), batches,
            layers, heads, max_tokens, microbatch_size, directory, 65536, 3 * 1024**2,
            aggregate_program=str(aggregate), batch_pages=32)


if __name__ == "__main__":
    unittest.main()
