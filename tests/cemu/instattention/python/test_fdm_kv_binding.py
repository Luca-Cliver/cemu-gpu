import importlib.util
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from native_kv_client import (
    BUFFER_APPEND, BUFFER_APPEND_BATCH, BUFFER_DRAIN, BUFFER_READ, BUFFER_REGISTER,
    BUFFER_STATUS, INFO, READ, REGISTER, VERSION, WRITE, coordinates,
)
from test_instattention_native_qtest import NativeQtest, ROOT, DATA, IOCQ, QEMU


CONFIG = ROOT / "tests/cemu/instattention/experiments/configs/native_csd_smoke.json"


def store_module(client):
    path = Path(__file__).parent / "cemu_flexgen/fdm_kvbuffer_store.py"
    spec = importlib.util.spec_from_file_location("fdm_store_under_test", path)
    module = importlib.util.module_from_spec(spec)
    binding = SimpleNamespace(CemuClient=client, MemoryRange=lambda *args: args)
    with patch.dict(sys.modules, {"_cemu_client": binding}):
        spec.loader.exec_module(module)
    return module


def new_store(module, **kwargs):
    parameters = dict(
        layout=SimpleNamespace(head_dim=128, channels_per_page=8, page_bytes=4096,
                               dtype=SimpleNamespace(itemsize=2)),
        batches=1, layers=1, heads=2, max_tokens=32, microbatch_size=1,
        fdm_directory="/unused/fdm", meta_bytes=65536, data_bytes=2 * 1024 * 1024,
    )
    store = module.FDM_KVbuffer(**(parameters | kwargs))
    store.session_id = 0x46444D01
    return store


class FdmBindingClientTest(unittest.TestCase):
    def test_packet_split_coordinates_layout_and_terminal_drain(self):
        from cemu_flexgen.sparf_group_layout import SparfGroupLayout

        client = Mock()
        module = store_module(Mock(return_value=client))
        store = new_store(module, layout=SparfGroupLayout(128, 8, dtype=np.float16),
                          batches=4, layers=2, heads=3, max_tokens=96, microbatch_size=2,
                          input_bytes=4608)
        store.open()
        keys = np.arange(35 * 6 * 128, dtype=np.float32).reshape(35, 6, 128).astype(np.float16)
        values = -keys
        first_stream = 12 + 6
        store.append_batch(first_stream, 0, keys, values)
        reconstructed_k = np.zeros_like(keys)
        reconstructed_v = np.zeros_like(values)
        for sequence, call in enumerate(client.write_range.call_args_list, 1):
            index, packet, offset = call.args
            fields = module.APPEND_HEADER.unpack_from(packet)
            self.assertEqual((index, offset, fields[3], fields[6]), (1, 0, 1, sequence))
            self.assertEqual(fields[7], len(packet))
            self.assertEqual(len(packet) % 512, 0)
            self.assertLessEqual(len(packet), 4608)
            self.assertEqual(fields[10], 1)
            count, head_count = fields[14], fields[12]
            self.assertLessEqual(count, 32)
            column = (fields[9] - 2) * 3 + fields[11]
            payload = np.frombuffer(packet, np.float16, fields[8] // 2, 512).reshape(2, head_count, count, 128)
            start = fields[13]
            reconstructed_k[start:start + count, column:column + head_count] = payload[0].transpose(1, 0, 2)
            reconstructed_v[start:start + count, column:column + head_count] = payload[1].transpose(1, 0, 2)
            self.assertFalse(any(packet[88:512]))
            self.assertFalse(any(packet[512 + fields[8]:]))
        np.testing.assert_array_equal(reconstructed_k, keys)
        np.testing.assert_array_equal(reconstructed_v, values)
        with self.assertRaises(ValueError):
            store.append_batch(first_stream, 0, keys, values)
        store.drain()
        header = module.APPEND_HEADER.unpack_from(client.write_range.call_args.args[1])
        self.assertEqual(header[3], 2)
        self.assertEqual(header[7], 512)
        self.assertTrue(all(value == 0 for value in header[8:]))
        count = client.write_range.call_count
        store.drain()
        self.assertEqual(client.write_range.call_count, count)
        with self.assertRaises(ValueError):
            store.append_batch(first_stream, 35, keys[:1], values[:1])
        store.close()
        client.release_memory_ranges.assert_called_once()
        client.execute.assert_not_called()

    def test_failed_append_is_not_retried_or_accepted_and_delete_failure_is_visible(self):
        from cemu_flexgen.sparf_group_layout import SparfGroupLayout

        client = Mock()
        store = new_store(store_module(Mock(return_value=client)),
                          layout=SparfGroupLayout(128, 8, dtype=np.float16)).open()
        client.write_range.side_effect = OSError("write failed")
        with self.assertRaisesRegex(OSError, "write failed"):
            store.append_batch(0, 0, np.ones((1, 2, 128)), np.zeros((1, 2, 128)))
        self.assertFalse(store._accepted)
        with self.assertRaises(RuntimeError):
            store.append_batch(0, 0, np.ones((1, 2, 128)), np.zeros((1, 2, 128)))
        self.assertEqual(client.write_range.call_count, 1)
        client.release_memory_ranges.side_effect = RuntimeError("MRS still busy")
        with self.assertRaisesRegex(RuntimeError, "MRS still busy"):
            store.close()
        client.close.assert_not_called()
        self.assertIs(store._client, client)

    def test_open_loads_once_and_binds_without_execute(self):
        client = Mock()
        client.create_memory_ranges.return_value = 7
        module = store_module(Mock(return_value=client))
        store = new_store(module)
        self.assertIs(store.open(), store)
        self.assertIs(store.open(), store)
        self.assertEqual([call[0] for call in client.method_calls],
                         ["create_memory_ranges", "load_program", "activate_program",
                          "bind_fdm_kv_buffer"])
        client.load_program.assert_called_once_with(
            f"fdm_kv_{store.session_id:016x}", "./build/fdm_kv_aggregate.so", "fdm_kv_aggregate")
        client.execute.assert_not_called()
        wire = client.bind_fdm_kv_buffer.call_args.args[0]
        self.assertEqual(len(wire), 80)
        self.assertEqual(struct.unpack("<4I3Q4IQ4I", wire), (
            0x31424B46, 1, 1, 256, store.session_id, 1, 131072,
            1, 1, 2, 128, 32, 8, 2, 4096, 0,
        ))
        store.close()
        store.close()
        client.close.assert_called_once()
        self.assertFalse(store.registered)
        with self.assertRaises(RuntimeError):
            store.open()

    def test_failed_binding_releases_mrs_and_preserves_error(self):
        client = Mock()
        client.bind_fdm_kv_buffer.side_effect = RuntimeError("bind rejected")
        store = new_store(store_module(Mock(return_value=client)))
        with self.assertRaisesRegex(RuntimeError, "bind rejected"):
            store.open()
        client.close.assert_called_once()
        self.assertFalse(store.registered)
        self.assertIsNone(store._client)

    def test_invalid_capacity_and_dimensions_before_io(self):
        factory = Mock()
        module = store_module(factory)
        for parameters in (
            {"batches": 0}, {"heads": 2**32}, {"microbatch_size": 2},
            {"input_bytes": 513}, {"input_bytes": 512},
            {"input_bytes": 4 * 1024 * 1024}, {"data_bytes": 1025},
            {"nvm_namespace_id": 0}, {"batch_pages": False},
            {"meta_bytes": 2**32}, {"data_bytes": 2**32},
        ):
            with self.subTest(parameters=parameters), self.assertRaises(ValueError):
                new_store(module, **parameters)
        factory.assert_not_called()


@unittest.skipUnless(QEMU.exists(), "build qemu-system-x86_64 first")
class FdmBindingDeviceTest(unittest.TestCase):
    def setUp(self):
        self.completions = {}

    def collect_completions(self):
        while self.device.completion(1) is not None:
            entry = self.device.read(IOCQ + self.device.heads[1] * 16, 16)
            command_id = struct.unpack_from("<H", entry, 12)[0]
            self.assertNotIn(command_id, self.completions)
            self.completions[command_id] = self.device.wait(1)

    def wait_completion(self, command_id, step_ns=100000):
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            self.collect_completions()
            if command_id in self.completions:
                return self.completions.pop(command_id)
            self.device.command(f"clock_step {step_ns}")
            time.sleep(0.001)
        self.fail(f"completion timeout cid={command_id}; {self.device.read_log()}")

    def assert_pending(self, command_id):
        self.collect_completions()
        self.assertNotIn(command_id, self.completions, self.device.read_log())

    def call(self, queue, opcode, fields, namespace=3, payload=None, rejected=False,
             expected_status=None, command2=0, command4=0):
        self.device.write(DATA, payload if payload is not None else bytes(4096))
        prp2 = self.payload_prp2(payload)
        self.device.submit(queue, opcode, fields, namespace=namespace, prp2=prp2,
                           command2=command2, command4=command4)
        status, result = (self.device.wait(0) if queue == 0 else
                          self.wait_completion(self.device.cid))
        if expected_status is not None:
            self.assertEqual(status, expected_status, self.device.read_log())
        elif rejected:
            self.assertNotEqual(status, 0)
        else:
            self.assertEqual(status, 0, self.device.read_log())
        return result

    def payload_prp2(self, payload):
        if payload is None or len(payload) <= 4096:
            return 0
        if len(payload) <= 8192:
            return DATA + 4096
        pages = (len(payload) + 4095) // 4096
        address = DATA + 128 * 1024
        self.device.write(address, struct.pack(f"<{pages - 1}Q",
                          *(DATA + page * 4096 for page in range(1, pages))))
        return address

    def create(self, descriptors=None):
        if descriptors is None:
            descriptors = ((2, 65536, 0), (2, 2 * 1024 * 1024, 65536))
        payload = b"".join(struct.pack("<IIQQQ", *descriptor, 0, 0)
                           for descriptor in descriptors)
        return self.call(0, 0x21, (0, len(descriptors), 0, 0, 0, 0), payload=payload)

    def bind(self, rsid, wire=None, rejected=False, program=0):
        self.call(0, 0x21, (2 | (rsid << 16), 80, program, 0, 0, 0),
                  payload=self.wire if wire is None else wire, rejected=rejected)

    def delete(self, rsid):
        self.call(0, 0x21, (1 | (rsid << 16), 0, 0, 0, 0, 0))

    def read_memory(self, offset, length=512):
        self.call(1, 2, (offset, 0, length, 0, 0, 0), namespace=2)
        return self.device.read(DATA, length)

    def test_femu_bind_release_reset_and_ordinary_io(self):
        config = ROOT / "tests/cemu/instattention/experiments/configs/native_csd_smoke.json"
        with tempfile.TemporaryDirectory(prefix="fdm-bind-") as directory:
            self.device = NativeQtest(directory, config, ftl_mode="femu")
            try:
                store = new_store(store_module(Mock()))
                self.wire = store._binding_descriptor()
                rsid = self.create()
                self.bind(0, rejected=True)
                self.bind(1025, rejected=True)
                self.call(0, 0x21, (2 | (rsid << 16), 79, 0, 0, 0, 0),
                          payload=self.wire, rejected=True)
                self.call(0, 0x21, (2 | (rsid << 16), 80, 1, 0, 0, 0),
                          payload=self.wire, rejected=True)
                sentinel = b"X" * 512
                self.call(1, 5, (65536, 0, 512, 0, 0, 0), namespace=2, payload=sentinel)
                self.bind(rsid)
                header = self.read_memory(0)
                self.assertEqual(struct.unpack_from("<2I", header), (1, 360))
                self.assertEqual(struct.unpack_from("<I", header, 12)[0], 1)
                self.assertEqual(struct.unpack_from("<2Q", header, 16), (store.session_id, 1))
                self.assertEqual(struct.unpack_from("<4I", header, 32), (1, 1, 2, 128))
                self.assertEqual(struct.unpack_from("<3I", header, 168), (2, 16, 1))
                self.assertEqual(self.read_memory(65536), sentinel)
                self.bind(rsid, rejected=True)
                other = self.create()
                self.bind(other, rejected=True)
                self.delete(other)
                self.assertEqual(self.read_memory(0), header)

                self.call(1, 1, (0, 0, 0, 0, 0, 0), namespace=1, payload=sentinel)
                self.call(1, 2, (0, 0, 0, 0, 0, 0), namespace=1)
                self.assertEqual(self.device.read(DATA, 512), sentinel)
                self.delete(rsid)
                self.assertEqual(struct.unpack_from("<I", self.read_memory(0), 176)[0], 0)

                for descriptors in (
                    ((2, 512, 0), (2, 2097152, 65536)),
                    ((2, 65536, 0), (2, 131072, 65536)),
                    ((2, 65536, 0), (2, 2097152, 0)),
                    ((2, 65536, 0), (2, 2097152, 8 * 1024 * 1024)),
                    ((2, 65536, 0), (1, 2097152, 65536)),
                ):
                    with self.subTest(descriptors=descriptors):
                        saved = self.read_memory(0)
                        invalid = self.create(descriptors)
                        self.bind(invalid, rejected=True)
                        self.assertEqual(self.read_memory(0), saved)
                        self.delete(invalid)

                rsid = self.create()
                for offset, value in ((0, 0), (4, 2), (8, 0), (8, 2), (8, 4),
                                      (8, 0xFFFFFFFF), (12, 0),
                                      (32, 512), (72, 8192), (76, 1)):
                    wire = bytearray(self.wire)
                    struct.pack_into("<I", wire, offset, value)
                    if wire == self.wire:
                        continue
                    self.bind(rsid, wire, rejected=True)
                for offset in (16, 24):
                    wire = bytearray(self.wire)
                    struct.pack_into("<Q", wire, offset, 0)
                    self.bind(rsid, wire, rejected=True)
                self.bind(rsid)
                self.device.initialize()
                self.assertEqual(struct.unpack_from("<I", self.read_memory(0), 176)[0], 0)
                self.bind(rsid)
                self.delete(rsid)
            finally:
                self.device.close()

    def packet(self, store, **overrides):
        fields = dict(magic=0x31564B46, version=1, header_bytes=512, operation=1,
                      session_id=store.session_id, generation=1, request_id=1,
                      wire_bytes=1536, payload_bytes=1024, batch=0, layer=0,
                      head_first=0, head_count=2, token_first=0, token_count=1, reserved=0)
        fields.update(overrides)
        header = struct.pack("<4I5Q4IQ2I", *fields.values())
        return header.ljust(512, b"\0") + b"K" * 512 + b"V" * 512

    def drain_packet(self, store, next_request_id, **overrides):
        fields = dict(operation=2, request_id=next_request_id, wire_bytes=512, payload_bytes=0,
                      batch=0, layer=0, head_first=0, head_count=0,
                      token_first=0, token_count=0, reserved=0)
        fields.update(overrides)
        return self.packet(store, **fields)[:512]

    def submit_fragment(self, offset, payload):
        self.device.write(DATA, payload)
        self.device.submit(1, 5, (65536 + offset, 0, len(payload), 0, 0, 0),
                           namespace=2, prp2=self.payload_prp2(payload))
        return self.device.cid

    def write_fragment(self, offset, payload, expected_status=0):
        absolute = 65536 + offset
        return self.call(1, 5, (absolute & 0xFFFFFFFF, absolute >> 32, len(payload), 0, 0, 0),
                         namespace=2, payload=payload, expected_status=expected_status)

    def receive_state(self):
        return struct.unpack_from("<2I2Q", self.read_memory(360, 128))

    def test_input_fragments_protection_dma_failure_and_reset(self):
        config = ROOT / "tests/cemu/instattention/experiments/configs/native_csd_smoke.json"
        with tempfile.TemporaryDirectory(prefix="fdm-receive-") as directory:
            self.device = NativeQtest(directory, config, ftl_mode="femu")
            try:
                store = new_store(store_module(Mock()))
                self.wire = store._binding_descriptor()
                rsid = self.create()
                self.bind(rsid)
                packet = self.packet(store)
                self.write_fragment(1024, packet[1024:])
                self.write_fragment(4, packet[4:512])
                self.assertEqual(self.receive_state()[:3], (1, 0, 1))
                before = self.read_memory(65536, 1536)
                metadata = self.read_memory(0)
                for offset, payload in ((-4, b"x" * 8), (131068, b"x" * 8),
                                         (131072, b"x" * 4), (-65536, b"x" * 4)):
                    self.write_fragment(offset, payload, 0x4002)
                self.call(1, 5, (65536, 0, 0, 0, 0, 0), namespace=2, expected_status=0x4002)
                self.write_fragment(4, b"x" * 4, 0x400C)
                for offset in (0, 65536, 196608):
                    self.call(1, 0, (offset, 0, 512, 0, 0, 0), namespace=2,
                              expected_status=0x4002)
                self.assertEqual(self.read_memory(65536, 1536), before)
                self.assertEqual(self.read_memory(0), metadata)
                self.call(1, 5, (0xFFFFFFFC, 0xFFFFFFFF, 8, 0, 0, 0),
                          namespace=2, payload=b"x" * 8, rejected=True)
                self.write_fragment(0, packet[:4])
                self.assertEqual(self.receive_state()[:3], (1, 1, 1))
                self.write_fragment(1536, b"x" * 4, 0x4002)
                self.write_fragment(512, packet[512:1024], 0x4001)
                self.assertEqual(self.receive_state()[:3], (4, 1, 1))
                self.assertGreater(self.receive_state()[3], 0)
                self.assertEqual(self.read_memory(65536, 1536), packet)
                self.write_fragment(0, packet[:512], 0x400C)
                self.assertEqual(self.read_memory(196608, 512), bytes(512))
                self.assertEqual(self.read_memory(264, 88), bytes(88))
                unrelated = 3 * 1024 * 1024
                self.call(1, 5, (unrelated, 0, 512, 0, 0, 0), namespace=2, payload=b"Z" * 512)
                self.assertEqual(self.read_memory(unrelated), b"Z" * 512)
                self.call(1, 0, (unrelated, 0, 512, 0, 0, 0), namespace=2)
                self.assertEqual(self.read_memory(unrelated), bytes(512))
                self.delete(rsid)
                rsid = self.create()
                self.bind(rsid)
                self.device.submit(1, 5, (65536, 0, 512, 0, 0, 0), prp1=0, namespace=2)
                self.assertEqual(self.device.wait(1)[0], 0x4004)
                self.assertEqual(self.receive_state()[:3], (3, 0, 1))
                self.assertEqual(self.read_memory(488, 4), bytes(4))
                self.write_fragment(0, packet[:512], 0x400C)
                self.device.initialize()
                self.bind(rsid)
                self.assertEqual(self.receive_state()[:3], (0, 0, 1))
                self.write_fragment(0, packet[:32])
                self.delete(rsid)
            finally:
                self.device.close()

    def test_invalid_headers_and_namespace_ids(self):
        config = ROOT / "tests/cemu/instattention/experiments/configs/native_csd_smoke.json"
        with tempfile.TemporaryDirectory(prefix="fdm-invalid-header-") as directory:
            self.device = NativeQtest(directory, config, ftl_mode="femu")
            try:
                store = new_store(store_module(Mock()))
                self.wire = store._binding_descriptor()
                for nsid in (0, 4, 0xFFFFFFFF):
                    descriptor = struct.pack("<IIQQQ", nsid, 4096, 0, 0, 0)
                    self.call(0, 0x21, (0, 1, 0, 0, 0, 0), payload=descriptor, rejected=True)
                for overrides in (
                    {"magic": 0}, {"version": 2}, {"header_bytes": 88}, {"operation": 9},
                    {"session_id": 0}, {"generation": 2}, {"request_id": 0}, {"request_id": 2},
                    {"wire_bytes": 1532}, {"wire_bytes": 2048}, {"wire_bytes": 2**64 - 1},
                    {"payload_bytes": 512}, {"payload_bytes": 2**64 - 1},
                    {"batch": 1}, {"layer": 1}, {"head_first": 2}, {"head_count": 0},
                    {"head_first": 1}, {"head_count": 0xFFFFFFFF}, {"token_first": 32},
                    {"token_first": 2**64 - 1}, {"token_count": 0}, {"token_count": 33},
                    {"reserved": 1},
                ):
                    with self.subTest(overrides=overrides):
                        rsid = self.create()
                        self.bind(rsid)
                        self.write_fragment(0, self.packet(store, **overrides)[:512], 0x4002)
                        self.assertEqual(self.receive_state()[0], 3)
                        self.assertEqual(self.read_memory(264, 88), bytes(88))
                        self.delete(rsid)
                for overrides in (
                    {"header_bytes": 88}, {"wire_bytes": 0}, {"wire_bytes": 1024},
                    {"payload_bytes": 512}, {"batch": 1}, {"layer": 1},
                    {"head_first": 1}, {"head_count": 1}, {"token_first": 1},
                    {"token_count": 1}, {"reserved": 1}, {"session_id": store.session_id + 1},
                    {"generation": 2}, {"request_id": 0}, {"request_id": 2},
                ):
                    with self.subTest(drain=overrides):
                        rsid = self.create()
                        self.bind(rsid)
                        self.write_fragment(0, self.drain_packet(store, 1, **overrides), 0x4002)
                        self.assertEqual(self.receive_state()[0], 3)
                        self.assertEqual(self.read_memory(264, 88), bytes(88))
                        self.delete(rsid)
                for extra_payload in (False, True):
                    with self.subTest(drain_extra_payload=extra_payload):
                        rsid = self.create()
                        self.bind(rsid)
                        packet = bytearray(self.drain_packet(store, 1))
                        if extra_payload:
                            self.write_fragment(512, bytes(512))
                        else:
                            packet[88] = 1
                        self.write_fragment(0, packet, 0x4002)
                        self.delete(rsid)
            finally:
                self.device.close()

    def test_native_mode_rejects_binding(self):
        config = ROOT / "tests/cemu/instattention/experiments/configs/native_csd_smoke.json"
        with tempfile.TemporaryDirectory(prefix="fdm-bind-native-") as directory:
            self.device = NativeQtest(directory, config)
            try:
                self.wire = new_store(store_module(Mock()))._binding_descriptor()
                rsid = self.create()
                self.bind(rsid, rejected=True)
                self.delete(rsid)
            finally:
                self.device.close()

    def load_aggregate(self, directory):
        library = Path(directory) / "aggregate.so"
        subprocess.run([
            "cc", "-std=c11", "-shared", "-fPIC", "-O2", "-Wall", "-Wextra", "-Werror",
            str(ROOT / "tests/cemu/kernel/fdm_kv_aggregate.c"),
            str(ROOT / "hw/femu/fdm_kv_layout.c"), "-o", str(library),
        ], check=True, timeout=60)
        payload = str(library).encode() + b"\0fdm_kv_aggregate\0"
        self.call(1, 0x22, (1 | (2 << 16), len(payload), 0, 0, len(payload), 0),
                  payload=payload)
        self.call(0, 0x23, (1 | (1 << 16), 0, 0, 0, 0, 0))

    def test_opt13b_smoke_capacity_and_binding(self):
        from cemu_flexgen.sparf_group_capacity import grouped_attention_reserve_bytes

        config = ROOT / "tests/cemu/instattention/experiments/configs/fdm_opt13b_smoke.json"
        original = json.loads(config.with_name("native_opt125m_smoke.json").read_text())
        original["namespaces"][1]["size"] = "256mb"
        self.assertEqual(json.loads(config.read_text()), original)
        with tempfile.TemporaryDirectory(prefix="fdm-opt13b-capacity-") as directory:
            self.device = NativeQtest(directory, config)
            try:
                self.load_aggregate(directory)
                store = new_store(store_module(Mock()), batches=1, layers=40, heads=40,
                                  max_tokens=36, meta_bytes=4 * 1024**2,
                                  data_bytes=160 * 1024**2, batch_pages=32)
                self.wire = store._binding_descriptor()
                small = self.create(((2, store.meta_bytes, 0),
                                     (2, 64 * 1024**2, store.meta_bytes)))
                self.bind(small, program=1, rejected=True)
                self.delete(small)
                rsid = self.create(((2, store.meta_bytes, 0),
                                    (2, store.data_bytes, store.meta_bytes)))
                self.bind(rsid, program=1)
                header = self.read_memory(0)
                self.assertEqual(struct.unpack_from("<2I", header), (1, 360))
                self.assertEqual(struct.unpack_from("<Q", header, 88)[0], 1600)
                self.assertEqual(struct.unpack_from("<Q", header, 104)[0], 35200)
                self.assertEqual(struct.unpack_from("<I", header, 148)[0], 32600)
                self.assertEqual(struct.unpack_from("<4I", header, 176), (1, 0, 0, 0))
                base = struct.unpack_from("<Q", header, 352)[0] - 360
                writebacks = struct.unpack_from("<I", header, 140)[0]
                meta_used = struct.unpack_from("<Q", header, 232)[0] - base + (writebacks + 1) * 24
                self.assertEqual(meta_used, 2214920)
                self.assertLess(meta_used, store.meta_bytes)
                data_used = store.input_bytes + 32600 * 4096
                self.assertLess(data_used, store.data_bytes)
                slots = grouped_attention_reserve_bytes(
                    attention_slots=2, batch_size=1, query_heads=40, kv_heads=40,
                    head_dim=128, element_bytes=2, max_seq_len=36, top_r=16,
                    compression_ratio=8)
                self.assertLess(store.meta_bytes + store.data_bytes + slots, 256 * 1024**2)
                self.delete(rsid)
            finally:
                self.device.close()

    def test_opt13b_performance_capacity_and_binding(self):
        from cemu_flexgen.sparf_group_capacity import grouped_attention_reserve_bytes

        config = ROOT / "tests/cemu/instattention/experiments/configs/fdm_opt13b_b4_p1024_d16.json"
        original = json.loads((ROOT / "femu-scripts/cemu_config_opt13b_sparf_engines2.json").read_text())
        original["namespaces"][1]["size"] = "4gb"
        self.assertEqual(json.loads(config.read_text()), original)
        with tempfile.TemporaryDirectory(prefix="fdm-opt13b-performance-capacity-") as directory:
            self.device = NativeQtest(directory, config, page_limit=2097152)
            try:
                self.load_aggregate(directory)
                store = new_store(store_module(Mock()), batches=4, layers=40, heads=40,
                                  max_tokens=1040, microbatch_size=4,
                                  meta_bytes=64 * 1024**2, data_bytes=3840 * 1024**2)
                self.wire = store._binding_descriptor()
                small = self.create(((2, store.meta_bytes, 0),
                                     (2, 2 * 1024**3, store.meta_bytes)))
                self.bind(small, program=1, rejected=True)
                self.delete(small)
                rsid = self.create(((2, store.meta_bytes, 0),
                                    (2, store.data_bytes, store.meta_bytes)))
                self.bind(rsid, program=1)
                header = self.read_memory(0)
                self.assertEqual(struct.unpack_from("<2I", header), (1, 360))
                self.assertEqual(struct.unpack_from("<Q", header, 88)[0], 6400)
                self.assertEqual(struct.unpack_from("<Q", header, 104)[0], 1344000)
                self.assertEqual(struct.unpack_from("<I", header, 148)[0], 972800)
                self.assertEqual(struct.unpack_from("<3I", header, 164), (3840, 8, 256))
                self.assertEqual(struct.unpack_from("<4I", header, 176), (1, 0, 0, 0))
                base = struct.unpack_from("<Q", header, 352)[0] - 360
                writebacks = struct.unpack_from("<I", header, 140)[0]
                meta_used = struct.unpack_from("<Q", header, 232)[0] - base + (writebacks + 1) * 24
                self.assertEqual(meta_used, 63370896)
                self.assertLess(meta_used, store.meta_bytes)
                data_used = store.input_bytes + 972800 * 4096
                self.assertEqual(data_used, 3984719872)
                self.assertLess(data_used, store.data_bytes)
                self.assertLess(store.data_bytes, 2**32)
                slots = grouped_attention_reserve_bytes(
                    attention_slots=2, batch_size=4, query_heads=40, kv_heads=40,
                    head_dim=128, element_bytes=2, max_seq_len=1040, top_r=16,
                    compression_ratio=8)
                self.assertEqual(slots, 32628736)
                self.assertLess(store.meta_bytes + store.data_bytes + slots, 4 * 1024**3)
                self.delete(rsid)
                sentinel_offset = store.meta_bytes + data_used - 512
                sentinel = bytes([0x5A]) * 512
                self.call(1, 5, (sentinel_offset, 0, 512, 0, 0, 0), namespace=2, payload=sentinel)
                self.assertEqual(self.read_memory(sentinel_offset), sentinel)
                rsid = self.create(((2, store.meta_bytes, 0),
                                    (2, store.data_bytes, store.meta_bytes)))
                self.bind(rsid, program=1)
                self.call(1, 5, (sentinel_offset, 0, 512, 0, 0, 0), namespace=2,
                          payload=bytes(512), expected_status=0x4002)
                self.assertEqual(self.read_memory(sentinel_offset), sentinel)
                packet = self.packet(store, batch=3, layer=39, head_first=38)
                self.call(1, 5, (store.meta_bytes, 0, len(packet), 0, 0, 0),
                          namespace=2, payload=packet)
                self.call(1, 5, (store.meta_bytes, 0, 512, 0, 0, 0), namespace=2,
                          payload=self.drain_packet(store, 2))
                header = self.read_memory(0)
                self.assertEqual(struct.unpack_from("<2I", header, 184), (1, 0))
                self.assertEqual(struct.unpack_from("<4Q", header, 264), (1, 2, 0, 36 * 4096))
                self.assertEqual(struct.unpack_from("<Q", header, 328)[0], 0)
                self.assertEqual(struct.unpack_from("<Q", header, 344)[0], 0)
                self.delete(rsid)
            finally:
                self.device.close()

    @staticmethod
    def element(head, token, dimension, value=False):
        return ((head + 1) * 1024 + token * 17 + dimension + (32768 if value else 0)) % 65536

    def kv_packet(self, store, request_id, first, count):
        payload = b"".join(struct.pack("<H", self.element(head, token, dimension, value))
                           for value in (False, True) for head in range(store.heads)
                           for token in range(first, first + count) for dimension in range(128))
        header = struct.pack("<4I5Q4IQ2I", 0x31564B46, 1, 512, 1,
                             store.session_id, 1, request_id, 512 + len(payload), len(payload),
                             0, 0, 0, store.heads, first, count, 0)
        return header.ljust(512, b"\0") + payload

    def writer_config(self, directory, **parameters):
        config = json.loads(CONFIG.read_text())
        config["namespaces"][0]["parameters"].update(parameters)
        path = Path(directory) / "writer-config.json"
        path.write_text(json.dumps(config))
        return path

    def writer_directory(self, prefix):
        directory = tempfile.TemporaryDirectory(prefix=prefix)
        self.addCleanup(directory.cleanup)
        return directory.name

    def start_writer(self, directory, config=CONFIG, **parameters):
        self.device = NativeQtest(directory, config)
        self.addCleanup(self.device.close)
        self.load_aggregate(directory)
        store = new_store(store_module(Mock()), **parameters)
        self.wire = store._binding_descriptor()
        rsid = self.create()
        self.bind(rsid, program=1)
        return store, rsid

    def clock(self):
        return int(self.device.command("clock_step 0"))

    def advance_to(self, deadline):
        self.assertGreaterEqual(deadline, self.clock())
        self.device.command(f"clock_set {deadline}")

    def snapshot(self, frozen=False):
        address = DATA + 256 * 1024
        prp_list = DATA + 384 * 1024
        length = 65536
        self.device.write(address, b"\xff" * length)
        self.device.write(prp_list, struct.pack("<15Q", *(
            address + page * 4096 for page in range(1, 16))))
        self.device.submit(1, 2, (0, 0, length, 0, 0, 0), namespace=2,
                           prp1=address, prp2=prp_list)
        command_id = self.device.cid
        if frozen:
            deadline = time.monotonic() + 5
            while self.device.read(address + length - 4, 4) == b"\xff" * 4:
                self.assertLess(time.monotonic(), deadline, self.device.read_log())
                time.sleep(0.001)
        else:
            self.assertEqual(self.wait_completion(command_id), (0, length))
        memory = self.device.read(address, length)
        self.assertEqual(struct.unpack_from("<2I", memory), (1, 360))
        base = struct.unpack_from("<Q", memory, 352)[0] - 360
        head_offset, page_offset, block_offset, hash_offset, writeback_offset = (
            pointer - base for pointer in struct.unpack_from("<5Q", memory, 200))
        pool_pages, free_page, free_pages, hash_size, domains, channels, block_pages = \
            struct.unpack_from("<7I", memory, 148)
        head_count = struct.unpack_from("<Q", memory, 88)[0]
        capacity = struct.unpack_from("<I", memory, 140)[0]
        stats = SimpleNamespace(**dict(zip((
            "append_commands", "appended_head_tokens", "staged_pages", "write_bytes",
            "read_bytes", "write_batches", "write_blocks", "partial_blocks",
            "inflight_blocks", "peak_inflight_blocks", "pending_ready_ns",
        ), struct.unpack_from("<11Q", memory, 264))))
        initialized, draining, sealed, failed = struct.unpack_from("<4I", memory, 176)
        return SimpleNamespace(
            memory=memory, pool_pages=pool_pages, free_page=free_page, free_pages=free_pages,
            channels=channels, block_pages=block_pages, stats=stats,
            initialized=initialized, draining=draining, sealed=sealed, failed=failed,
            receive=struct.unpack_from("<2I2Q", memory, 360),
            heads=[struct.unpack_from("<4Q2I", memory, head_offset + index * 40)
                   for index in range(head_count)],
            pages=[struct.unpack_from("<Q4IQ6I", memory, page_offset + index * 56)
                   for index in range(pool_pages + 1)],
            blocks=[struct.unpack_from("<4I", memory, block_offset + index * 16)
                    for index in range(domains)],
            hashes=struct.unpack_from(f"<{hash_size}I", memory, hash_offset),
            writebacks=[struct.unpack_from("<Q4I", memory, writeback_offset + index * 24)
                        for index in range(1, capacity + 1)],
            data_offset=65536 + struct.unpack_from("<Q", memory, 256)[0] -
                        struct.unpack_from("<Q", memory, 240)[0],
        )

    def await_snapshot(self, predicate):
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            current = self.snapshot()
            self.assertEqual(current.failed, 0, self.device.read_log())
            if predicate(current):
                return current
            time.sleep(0.001)
        self.fail(f"writer state timeout: {vars(current.stats)}; {self.device.read_log()}")

    def verify_resident(self, snapshot):
        resident = set()
        for slot, first in enumerate(snapshot.hashes):
            while first:
                self.assertLessEqual(first, snapshot.pool_pages)
                self.assertNotIn(first, resident)
                resident.add(first)
                page = snapshot.pages[first]
                hashed = (page[0] * 0x9E3779B97F4A7C15) & (2**64 - 1)
                self.assertEqual((hashed ^ (hashed >> 32)) & (len(snapshot.hashes) - 1), slot)
                self.assertGreater(page[8], 0)
                first = page[10]
        free = set()
        first = snapshot.free_page
        while first:
            self.assertLessEqual(first, snapshot.pool_pages)
            self.assertNotIn(first, resident | free)
            free.add(first)
            first = snapshot.pages[first][9]
        self.assertEqual(len(free), snapshot.free_pages)
        self.assertEqual(len(resident), snapshot.stats.staged_pages)
        self.assertEqual(resident | free, set(range(1, snapshot.pool_pages + 1)))
        inflight = set()
        active = [entry for entry in snapshot.writebacks if entry[4]]
        self.assertEqual(len(active), snapshot.stats.inflight_blocks)
        for deadline, first, count, _, _ in active:
            self.assertGreater(deadline, 0)
            for position in range(count):
                self.assertIn(first, resident)
                self.assertNotIn(first, inflight)
                inflight.add(first)
                self.assertEqual(snapshot.pages[first][11], 1)
                first = snapshot.pages[first][9]
            self.assertEqual(first, 0)
        return resident, inflight

    def expected_page(self, head, kind, token, channel, valid):
        capacity = 256 if kind == 1 else 16
        expected = bytearray(4096)
        for position in range(valid):
            for dimension in range(8 if kind == 1 else 128):
                target = dimension * capacity + position if kind == 1 else position * 128 + dimension
                value = self.element(head, token + position,
                                     channel + dimension if kind == 1 else dimension, kind == 2)
                struct.pack_into("<H", expected, 2 * target, value)
        return expected

    def verify_flash(self, store, tokens):
        for head in range(store.heads):
            for kind in (0, 1, 2):
                capacity = 256 if kind == 1 else 16
                for token in range(0, tokens, capacity):
                    for channel in (range(0, 128, 8) if kind == 1 else (0,)):
                        valid = min(capacity, tokens - token)
                        with self.subTest(head=head, kind=kind, token=token, channel=channel):
                            self.call(1, READ, coordinates(store.session_id, head=head, kind=kind,
                                      token=token + valid - 1, channel=channel), namespace=1)
                            self.assertEqual(self.device.read(DATA, 4096),
                                             self.expected_page(head, kind, token, channel, valid))

    def drain(self, store, request_id):
        packet = self.drain_packet(store, request_id)
        self.assertEqual(len(packet), 512)
        self.assertEqual(self.write_fragment(0, packet), 512)
        current = self.snapshot()
        self.assertEqual((current.sealed, current.failed, current.stats.inflight_blocks), (1, 0, 0))
        self.assertEqual(current.free_pages, current.pool_pages)
        self.assertEqual(current.stats.staged_pages, 0)
        self.assertEqual(current.receive[:3], (0, 0, request_id + 1))
        self.verify_resident(current)
        return current

    def verify_staged(self, tokens, heads):
        header = self.read_memory(0)
        meta_base = struct.unpack_from("<Q", header, 352)[0] - 360
        head_offset, staged_offset, blocks_offset = (
            pointer - meta_base for pointer in struct.unpack_from("<3Q", header, 200))
        input_address = struct.unpack_from("<Q", header, 240)[0]
        data_offset = 65536 + struct.unpack_from("<Q", header, 256)[0] - input_address
        pool_pages, _, free_pages = struct.unpack_from("<3I", header, 148)
        self.assertEqual(struct.unpack_from("<3Q", header, 288), (0, 0, 0))
        for head in range(heads):
            self.assertEqual(struct.unpack("<4Q2I", self.read_memory(head_offset + head * 40, 40)),
                             (tokens, 0, 0, 0, 0, 0))
        found = set()
        queued = set()
        for index in range(1, pool_pages + 1):
            page = self.read_memory(staged_offset + index * 56, 56)
            _, batch, layer, head, kind, token, channel, _, valid, _, _, ready = \
                struct.unpack("<Q4IQ6I", page)
            if not valid:
                continue
            self.assertEqual((batch, layer), (0, 0))
            self.assertLess(head, heads)
            self.assertIn(kind, (0, 1, 2))
            capacity = 256 if kind == 1 else 16
            self.assertEqual(token % capacity, 0)
            self.assertEqual(valid, min(capacity, tokens - token))
            self.assertEqual(ready, int(valid == capacity))
            coordinate = (head, kind, token, channel)
            self.assertNotIn(coordinate, found)
            found.add(coordinate)
            if ready:
                queued.add(index)
            expected = bytearray(4096)
            for position in range(valid):
                for dimension in range(8 if kind == 1 else 128):
                    target = dimension * capacity + position if kind == 1 else position * 128 + dimension
                    value = self.element(head, token + position,
                                         channel + dimension if kind == 1 else dimension, kind == 2)
                    struct.pack_into("<H", expected, 2 * target, value)
            self.assertEqual(self.read_memory(data_offset + (index - 1) * 4096, 4096), expected)
        self.assertEqual(len(found), heads * (2 * ((tokens + 15) // 16) + 16 * ((tokens + 255) // 256)))
        self.assertEqual(len(found), pool_pages - free_pages)
        domain_count = struct.unpack_from("<I", header, 164)[0]
        actual_queued = set()
        for domain in range(domain_count):
            first, last, count, _ = struct.unpack("<4I", self.read_memory(blocks_offset + domain * 16, 16))
            previous = 0
            for position in range(count):
                self.assertNotIn(first, actual_queued)
                self.assertIn(first, queued)
                actual_queued.add(first)
                page = self.read_memory(staged_offset + first * 56, 56)
                kind = struct.unpack_from("<I", page, 20)[0]
                token = struct.unpack_from("<Q", page, 24)[0]
                self.assertEqual(domain, kind * 2 + (token // (256 if kind == 1 else 16)) % 2)
                previous, first = first, struct.unpack_from("<I", page, 44)[0]
            self.assertEqual((previous, first), (last, 0))
        self.assertEqual(actual_queued, queued)

    def test_native_write_aggregates_without_execute(self):
        with tempfile.TemporaryDirectory(prefix="fdm-aggregate-device-") as directory:
            config = self.writer_config(directory, pgs_per_blk=256)
            self.device = NativeQtest(directory, config)
            try:
                self.load_aggregate(directory)
                store = new_store(store_module(Mock()), max_tokens=272)
                self.wire = store._binding_descriptor()
                rsid = self.create()
                self.bind(rsid, program=2, rejected=True)
                self.bind(rsid, program=1)
                self.call(0, 0x23, (1, 0, 0, 0, 0, 0), expected_status=0x99)
                self.call(1, 0x22, (1 | (1 << 24), 0, 0, 0, 0, 0), expected_status=0x99)
                self.call(1, 1, (0, 0, 0, 0, 0, 0), namespace=1, rejected=True)
                self.call(1, 2, (0, 0, 0, 0, 0, 0), namespace=1, rejected=True)
                for opcode in (REGISTER, WRITE, READ, BUFFER_REGISTER, BUFFER_APPEND,
                               BUFFER_APPEND_BATCH, BUFFER_READ, BUFFER_STATUS, BUFFER_DRAIN, 0):
                    self.call(1, opcode, coordinates(store.session_id),
                              namespace=1, expected_status=0x400C)
                self.call(1, INFO, (VERSION, 0, 0, 0, 0, 0), namespace=1)
                first = 0
                for request_id, count in enumerate([15, 1] + [32] * 7 + [13, 3, 3], 1):
                    packet = self.kv_packet(store, request_id, first, count)
                    if request_id == 1:
                        self.write_fragment(512, packet[512:])
                        self.write_fragment(4, packet[4:512])
                        self.assertEqual(self.write_fragment(0, packet[:4]), 4)
                    else:
                        self.assertEqual(self.write_fragment(0, packet), len(packet))
                    first += count
                    self.assertEqual(self.receive_state()[:3], (0, 0, request_id + 1))
                    if first in (15, 16, 253, 256, 259):
                        self.verify_staged(first, 2)
                self.call(1, INFO, (VERSION, 0, 0, 0, 0, 0), namespace=1)
                self.assertEqual(struct.unpack_from("<3Q", self.device.read(DATA, 80), 40), (0, 0, 0))
                self.write_fragment(0, packet, 0x4002)
                self.assertEqual(self.receive_state()[0], 3)
                self.verify_staged(259, 2)
                self.delete(rsid)
                self.call(0, 0x23, (1, 0, 0, 0, 0, 0))
                self.call(1, 0x22, (1 | (1 << 24), 0, 0, 0, 0, 0))
                self.assertIn("FDM KV non-frozen aggregation: calls=12", self.device.read_log())
            finally:
                self.device.close()

    def test_native_pending_write_time_gate_and_reset(self):
        config = ROOT / "tests/cemu/instattention/experiments/configs/native_csd_smoke.json"
        with tempfile.TemporaryDirectory(prefix="fdm-aggregate-pending-") as directory:
            self.device = NativeQtest(directory, config)
            try:
                self.load_aggregate(directory)
                store = new_store(store_module(Mock()), heads=7)
                self.wire = store._binding_descriptor()
                rsid = self.create()
                self.bind(rsid, program=1)
                packet = self.kv_packet(store, 1, 0, 32)
                for reset in (True, False):
                    self.device.write(DATA, packet)
                    self.device.submit(1, 5, (65536, 0, len(packet), 0, 0, 0), namespace=2,
                                       prp2=self.payload_prp2(packet))
                    time.sleep(0.02)
                    self.assertIsNone(self.device.completion(1))
                    self.device.submit(1, 2, (0, 0, 512, 0, 0, 0), namespace=2,
                                       prp1=DATA + 256 * 1024)
                    time.sleep(0.02)
                    self.device.command("clock_step 1000")
                    time.sleep(0.02)
                    self.assertEqual(self.device.completion(1), (0, 512))
                    self.assertEqual(self.device.wait(1), (0, 512))
                    header = self.device.read(DATA + 256 * 1024, 512)
                    self.assertEqual(struct.unpack_from("<I", header, 360)[0], 4)
                    self.assertEqual(struct.unpack_from("<Q", header, 264)[0], 0)
                    self.call(0, 0x21, (1 | (rsid << 16), 0, 0, 0, 0, 0), expected_status=0x95)
                    if reset:
                        self.device.initialize()
                        self.assertEqual(struct.unpack_from("<I", self.read_memory(0), 176)[0], 0)
                        self.bind(rsid, program=1)
                    else:
                        self.assertEqual(self.device.wait(1), (0, len(packet)))
                        self.assertEqual(self.receive_state()[:3], (0, 0, 2))
                self.drain(store, 2)
                self.delete(rsid)
            finally:
                self.device.close()

    def test_native_aggregate_failure_does_not_acknowledge_packet(self):
        config = ROOT / "tests/cemu/instattention/experiments/configs/native_csd_smoke.json"
        with tempfile.TemporaryDirectory(prefix="fdm-aggregate-errors-") as directory:
            self.device = NativeQtest(directory, config)
            try:
                self.load_aggregate(directory)
                store = new_store(store_module(Mock()), max_tokens=1024)
                self.wire = store._binding_descriptor()
                rsid = self.create()
                self.bind(rsid, program=1)
                self.write_fragment(0, self.kv_packet(store, 1, 1, 1), 0x4002)
                self.assertEqual(self.receive_state()[0], 3)
                self.assertEqual(self.read_memory(264, 88), bytes(88))
                self.delete(rsid)
            finally:
                self.device.close()

    def test_native_writer_long_sequence_recycles_and_drains_to_flash(self):
        directory = self.writer_directory("fdm-writer-long-")
        store, rsid = self.start_writer(directory, max_tokens=1088)
        initial = self.snapshot()
        self.assertEqual((initial.channels, initial.block_pages), (2, 16))
        tokens = 1059
        for request_id, first in enumerate(range(0, tokens, 32), 1):
            count = min(32, tokens - first)
            packet = self.kv_packet(store, request_id, first, count)
            self.assertEqual(self.write_fragment(0, packet), len(packet))
            current = self.snapshot()
            self.assertEqual(current.failed, 0)
            self.assertEqual(current.stats.append_commands, request_id)
            self.assertEqual(current.stats.appended_head_tokens, (first + count) * store.heads)
            self.assertTrue(all(head[0] == first + count for head in current.heads))
        background = self.await_snapshot(lambda current: current.stats.write_blocks > 0)
        self.assertGreater(background.stats.write_bytes, initial.pool_pages * 4096)
        self.assertLess(background.stats.staged_pages, initial.pool_pages)
        self.assertTrue(all(min(head[1:4]) > 0 for head in background.heads))
        completed = self.drain(store, request_id + 1)
        pages = store.heads * (2 * ((tokens + 15) // 16) + 16 * ((tokens + 255) // 256))
        self.assertGreater(pages, initial.pool_pages * 2)
        self.assertEqual(completed.stats.write_bytes, pages * 4096)
        self.assertGreater(completed.stats.partial_blocks, 0)
        self.assertTrue(all(head[:4] == (tokens,) * 4 for head in completed.heads))
        self.write_fragment(0, self.kv_packet(store, request_id + 2, tokens, 1), 0x400C)
        self.write_fragment(0, self.drain_packet(store, request_id + 2), 0x400C)
        self.delete(rsid)
        self.verify_flash(store, tokens)
        self.call(1, INFO, (VERSION, 0, 0, 0, 0, 0), namespace=1)
        self.assertEqual(struct.unpack_from("<3Q", self.device.read(DATA, 80), 40), (pages, pages, 0))

    def test_native_writer_multichannel_deadline_retains_pages_and_hash(self):
        directory = self.writer_directory("fdm-writer-deadline-")
        config = self.writer_config(directory, pg_wr_lat=1000000000)
        store, rsid = self.start_writer(directory, config, heads=8, max_tokens=128)
        for request_id, first in enumerate(range(0, 64, 16), 1):
            self.write_fragment(0, self.kv_packet(store, request_id, first, 16))
        before = self.await_snapshot(lambda current: current.stats.inflight_blocks == 4 and
                                     current.stats.write_bytes == 64 * 4096 and
                                     all(entry[0] > 0 for entry in current.writebacks if entry[4]))
        resident, inflight = self.verify_resident(before)
        self.assertEqual((len(resident), len(inflight)), (192, 64))
        self.assertEqual(before.stats.write_blocks, 0)
        self.assertEqual(before.stats.partial_blocks, 0)
        self.assertGreaterEqual(before.stats.peak_inflight_blocks, 4)
        self.assertTrue(all(head[:4] == (64, 0, 0, 0) for head in before.heads))
        domains = {before.pages[index][4] * 2 + before.pages[index][5] // 16 % 2
                   for index in inflight}
        self.assertEqual(domains, {0, 1, 4, 5})
        self.call(0, 0x21, (1 | (rsid << 16), 0, 0, 0, 0, 0), expected_status=0x95)
        for index in sorted(inflight):
            page = before.pages[index]
            self.assertEqual(self.read_memory(before.data_offset + (index - 1) * 4096, 4096),
                             self.expected_page(page[3], page[4], page[5], page[6], page[8]))
        deadlines = [entry[0] for entry in before.writebacks if entry[4]]
        frozen_clock = self.clock()
        time.sleep(0.02)
        self.assertEqual(self.clock(), frozen_clock)
        self.assertIsNone(self.device.completion(1))
        self.advance_to(min(deadlines) - 1)
        early = self.snapshot(frozen=True)
        self.assertEqual(self.clock(), min(deadlines) - 1)
        self.assertEqual(self.verify_resident(early), (resident, inflight))
        self.assertEqual(early.free_pages, before.free_pages)
        self.assertEqual(early.heads, before.heads)
        self.assertEqual(early.stats.write_blocks, 0)
        self.advance_to(max(deadlines))
        after = self.await_snapshot(lambda current: current.stats.inflight_blocks == 0)
        self.assertEqual(after.stats.write_blocks, 4)
        self.assertEqual(after.free_pages, before.free_pages + 64)
        self.assertEqual(after.stats.staged_pages, 128)
        self.assertTrue(all(head[:4] == (64, 64, 0, 64) for head in after.heads))
        self.verify_resident(after)
        self.collect_completions()
        self.assertEqual(list(self.completions.values()), [(0, 65536)])
        self.completions.clear()
        self.assertIsNone(self.device.completion(1), "internal writeback must not post a Guest CQE")
        drain_id = self.submit_fragment(0, self.drain_packet(store, 5))
        draining = self.await_snapshot(lambda current: current.draining and
                                       current.stats.write_bytes == 192 * 4096)
        self.assert_pending(drain_id)
        self.advance_to(draining.stats.pending_ready_ns)
        self.assertEqual(self.wait_completion(drain_id), (0, 512))
        self.assertEqual(self.snapshot().sealed, 1)
        self.delete(rsid)
        self.verify_flash(store, 64)

    def test_native_writer_no_space_keeps_ready_and_original_write(self):
        directory = self.writer_directory("fdm-writer-backpressure-")
        config = self.writer_config(directory, pg_wr_lat=1000000000)
        store, rsid = self.start_writer(directory, config, max_tokens=1024)
        for request_id in range(1, 13):
            self.write_fragment(0, self.kv_packet(store, request_id, (request_id - 1) * 32, 32))
        before = self.await_snapshot(lambda current: current.stats.inflight_blocks > 0 and
                                     current.stats.pending_ready_ns > self.clock())
        self.assertLess(before.free_pages, 8)
        packet = self.kv_packet(store, 13, 384, 32)
        command_id = self.submit_fragment(0, packet)
        for attempt in range(3):
            self.device.command("clock_step 10000")
            blocked = self.snapshot()
            self.assertLess(self.clock(), min(entry[0] for entry in before.writebacks if entry[4]))
            self.assert_pending(command_id)
            self.assertEqual(blocked.receive[:3], (4, 1, 13))
            self.assertEqual(blocked.failed, 0)
            self.assertEqual(blocked.heads, before.heads)
            self.assertEqual(blocked.stats.append_commands, 12)
            self.assertEqual(blocked.stats.appended_head_tokens, 768)
            self.assertEqual(blocked.free_pages, before.free_pages)
            self.verify_resident(blocked)
        self.assertEqual(self.read_memory(65536, 512), packet[:512])
        self.call(0, 0x21, (1 | (rsid << 16), 0, 0, 0, 0, 0), expected_status=0x95)
        self.advance_to(before.stats.pending_ready_ns)
        self.assertEqual(self.wait_completion(command_id), (0, len(packet)))
        accepted = self.snapshot()
        self.assertEqual(accepted.receive[:3], (0, 0, 14))
        self.assertEqual(accepted.stats.append_commands, 13)
        self.assertEqual(accepted.stats.appended_head_tokens, 832)
        self.assertTrue(all(head[0] == 416 for head in accepted.heads))
        drain_id = self.submit_fragment(0, self.drain_packet(store, 14))
        draining = self.await_snapshot(lambda current: current.draining and
                                       sum(block[2] for block in current.blocks) == 0 and
                                       current.stats.pending_ready_ns > self.clock())
        self.advance_to(draining.stats.pending_ready_ns)
        self.assertEqual(self.wait_completion(drain_id), (0, 512))
        self.assertEqual(self.snapshot().sealed, 1)
        self.delete(rsid)
        self.verify_flash(store, 416)

    def test_native_drain_partial_fragments_wait_for_sealed(self):
        directory = self.writer_directory("fdm-writer-drain-")
        config = self.writer_config(directory, pg_wr_lat=1000000000)
        store, rsid = self.start_writer(directory, config)
        append = self.kv_packet(store, 1, 0, 3)
        self.write_fragment(0, append[:512])
        self.call(0, 0x21, (1 | (rsid << 16), 0, 0, 0, 0, 0), expected_status=0x95)
        self.write_fragment(512, append[512:])
        before = self.snapshot()
        self.assertEqual((before.stats.write_bytes, before.stats.staged_pages), (0, 36))
        self.call(0, 0x21, (1 | (rsid << 16), 0, 0, 0, 0, 0), expected_status=0x95)
        packet = self.drain_packet(store, 2)
        self.assertEqual(len(packet), 512)
        self.write_fragment(4, packet[4:])
        self.assertEqual(self.receive_state()[:3], (1, 0, 2))
        command_id = self.submit_fragment(0, packet[:4])
        draining = self.await_snapshot(lambda current: current.draining and
                                       current.stats.write_bytes == 36 * 4096 and
                                       current.stats.inflight_blocks == 4)
        self.assert_pending(command_id)
        self.assertEqual(draining.sealed, 0)
        self.assertEqual(draining.free_pages, before.free_pages)
        self.assertEqual(draining.heads, before.heads)
        self.verify_resident(draining)
        self.write_fragment(0, self.kv_packet(store, 3, 3, 1), 0x400C)
        self.call(0, 0x21, (1 | (rsid << 16), 0, 0, 0, 0, 0), expected_status=0x95)
        deadlines = [entry[0] for entry in draining.writebacks if entry[4]]
        self.advance_to(min(deadlines) - 1)
        early = self.snapshot(frozen=True)
        self.assertEqual((early.sealed, early.free_pages), (0, before.free_pages))
        self.assert_pending(command_id)
        self.advance_to(min(deadlines))
        partial = self.await_snapshot(lambda current: current.free_pages > before.free_pages)
        self.assertLess(self.clock(), max(deadlines))
        self.assertGreater(partial.stats.inflight_blocks, 0)
        self.assertLess(partial.free_pages, partial.pool_pages)
        self.assertEqual(partial.sealed, 0)
        self.assert_pending(command_id)
        self.verify_resident(partial)
        self.advance_to(max(deadlines) - 1)
        self.assertEqual(self.snapshot(frozen=True).sealed, 0)
        self.assert_pending(command_id)
        self.advance_to(max(deadlines))
        self.assertEqual(self.wait_completion(command_id), (0, 4))
        completed = self.snapshot()
        self.assertEqual((completed.sealed, completed.failed), (1, 0))
        self.assertEqual(completed.stats.append_commands, 1)
        self.assertEqual((completed.stats.write_blocks, completed.stats.partial_blocks), (2, 2))
        self.assertEqual(completed.free_pages, completed.pool_pages)
        self.assertEqual(completed.receive[:3], (0, 0, 3))
        self.assertTrue(all(head[:4] == (3, 3, 3, 3) for head in completed.heads))
        self.verify_resident(completed)
        self.write_fragment(0, self.drain_packet(store, 3), 0x400C)
        self.delete(rsid)
        self.verify_flash(store, 3)

    def test_native_writer_lazy_registration_and_identity_reuse_fail_closed(self):
        directory = self.writer_directory("fdm-writer-identity-")
        store, rsid = self.start_writer(directory)
        for attempt in range(2):
            self.drain(store, 1)
            self.delete(rsid)
            rsid = self.create()
            self.bind(rsid, program=1)
        self.write_fragment(0, self.kv_packet(store, 1, 0, 3))
        self.assertEqual(self.snapshot().stats.write_bytes, 0)
        self.drain(store, 2)
        self.delete(rsid)
        self.verify_flash(store, 3)
        rsid = self.create()
        self.bind(rsid, program=1)
        packet = self.kv_packet(store, 1, 0, 3)
        self.write_fragment(0, packet[:512] + b"Z" * (len(packet) - 512))
        self.call(1, 5, (65536, 0, 512, 0, 0, 0), namespace=2,
                  payload=self.drain_packet(store, 2), rejected=True)
        failed = self.snapshot()
        self.assertEqual((failed.failed, failed.sealed, failed.receive[0]), (1, 0, 3))
        self.assertEqual(failed.receive[2], 2)
        self.assertEqual((failed.stats.write_bytes, failed.stats.write_blocks), (0, 0))
        self.assertEqual(failed.stats.staged_pages, 36)
        self.assertTrue(all(head[:4] == (3, 0, 0, 0) for head in failed.heads))
        self.write_fragment(0, self.kv_packet(store, 2, 3, 1), 0x400C)
        self.delete(rsid)
        self.verify_flash(store, 3)
        self.device.initialize()
        self.completions.clear()
        store.session_id += 1
        self.wire = store._binding_descriptor()
        rsid = self.create()
        self.bind(rsid, program=1)
        self.write_fragment(0, self.kv_packet(store, 1, 0, 3))
        self.drain(store, 2)
        self.delete(rsid)
        self.verify_flash(store, 3)

    def test_native_writer_reset_cancels_inflight_and_pending_drain(self):
        directory = self.writer_directory("fdm-writer-reset-")
        config = self.writer_config(directory, pg_wr_lat=1000000000)
        store, rsid = self.start_writer(directory, config)
        self.write_fragment(0, self.kv_packet(store, 1, 0, 3))
        command_id = self.submit_fragment(0, self.drain_packet(store, 2))
        before = self.await_snapshot(lambda current: current.stats.write_bytes == 36 * 4096 and
                                     current.stats.inflight_blocks == 4)
        self.assert_pending(command_id)
        self.assertTrue(all(head[:4] == (3, 0, 0, 0) for head in before.heads))
        self.device.initialize()
        self.completions.clear()
        stopped = self.snapshot()
        self.assertEqual((stopped.initialized, stopped.failed, stopped.sealed), (0, 1, 0))
        self.assertEqual(stopped.stats.inflight_blocks, 0)
        self.assertEqual(stopped.stats.pending_ready_ns, 0)
        self.assertEqual(stopped.free_pages, before.free_pages)
        self.assertEqual(stopped.heads, before.heads)
        self.assertEqual((stopped.stats.write_blocks, stopped.stats.partial_blocks), (0, 0))
        store.session_id += 1
        self.wire = store._binding_descriptor()
        self.bind(rsid, program=1)
        fresh = self.snapshot()
        self.assertEqual(fresh.receive[:3], (0, 0, 1))
        self.advance_to(before.stats.pending_ready_ns + 100000)
        time.sleep(0.02)
        self.assertIsNone(self.device.completion(1), "reset must cancel old Guest and internal completions")
        after = self.snapshot()
        self.assertEqual(after.memory, fresh.memory)
        self.write_fragment(0, self.kv_packet(store, 1, 0, 3))
        drain_id = self.submit_fragment(0, self.drain_packet(store, 2))
        draining = self.await_snapshot(lambda current: current.stats.write_bytes == 36 * 4096)
        self.advance_to(draining.stats.pending_ready_ns)
        self.assertEqual(self.wait_completion(drain_id), (0, 512))
        self.assertEqual(self.snapshot().sealed, 1)
        self.delete(rsid)
        self.verify_flash(store, 3)


if __name__ == "__main__":
    unittest.main()
