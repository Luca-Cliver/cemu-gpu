import stat
import struct
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from native_kv_client import (
    BUFFER_APPEND, BUFFER_DRAIN, INFO, LinuxTransport, PAGE_BYTES,
    BUFFER_APPEND_BATCH, APPEND_MAX_BYTES,
    retry_buffer, run_buffer_contract, run_contract,
)


class NativeKVTransportTest(unittest.TestCase):
    def test_non_character_device_rejected_before_ioctl(self):
        for mode in (stat.S_IFBLK, stat.S_IFREG):
            with self.subTest(mode=mode), \
                    patch("native_kv_client.os.open", return_value=42), \
                    patch("native_kv_client.os.fstat", return_value=SimpleNamespace(st_mode=mode)), \
                    patch("native_kv_client.os.close") as close, \
                    patch("native_kv_client.ctypes.CDLL") as library:
                with self.assertRaisesRegex(ValueError, "/dev/ng0n1"):
                    LinuxTransport("/dev/nvme0n1")
                library.assert_not_called()
                close.assert_called_once_with(42)

    def test_character_device_uses_passthrough(self):
        with patch("native_kv_client.os.open", return_value=42), \
                patch("native_kv_client.os.fstat", return_value=SimpleNamespace(st_mode=stat.S_IFCHR)), \
                patch("native_kv_client.os.close") as close, \
                patch("native_kv_client.ctypes.CDLL") as library:
            library.return_value.ioctl.return_value = 0
            transport = LinuxTransport("/dev/ng0n1")
            try:
                status, _, data = transport(INFO, (1, 0, 0, 0, 0, 0))
                self.assertEqual(status, 0)
                self.assertEqual(len(data), PAGE_BYTES)
                command = library.return_value.ioctl.call_args.args[2]._obj
                self.assertEqual(command.nsid, 1)
                self.assertEqual(command.opcode, INFO)
            finally:
                transport.close()
            close.assert_called_once_with(42)

    def test_wrong_controller_header_never_writes(self):
        transport = Mock(return_value=(0, 0, bytes(PAGE_BYTES)))
        with self.assertRaisesRegex(RuntimeError, "refusing writes"):
            run_contract(transport)
        self.assertEqual(transport.call_count, 1)
        self.assertEqual(transport.call_args.args[0], INFO)

    def test_batched_transport_uses_actual_length(self):
        import ctypes
        from native_kv_client import Passthrough
        transport = LinuxTransport.__new__(LinuxTransport)
        transport.descriptor = 42
        transport.libc = Mock()
        payload = bytes(range(256)) * 512

        def ioctl(descriptor, operation, pointer):
            command = ctypes.cast(pointer, ctypes.POINTER(Passthrough)).contents
            self.assertEqual(command.data_len, APPEND_MAX_BYTES)
            self.assertEqual(ctypes.string_at(command.addr, command.data_len), payload)
            return 0

        transport.libc.ioctl.side_effect = ioctl
        transport(BUFFER_APPEND_BATCH, (1, 1, 1, 0, 0, 0), payload)
        for invalid in (None, b"", payload + b"x"):
            with self.assertRaises(ValueError):
                transport(BUFFER_APPEND_BATCH, (1, 1, 1, 0, 0, 0), invalid)

    def test_buffer_wrong_controller_never_writes(self):
        transport = Mock(return_value=(0, 0, bytes(PAGE_BYTES)))
        with self.assertRaisesRegex(RuntimeError, "refusing writes"):
            run_buffer_contract(transport)
        self.assertEqual(transport.call_count, 1)

    def test_pending_retries_identical_packet(self):
        transport = Mock(side_effect=[(12, 5, b""), (0, 0, b"done")])
        with patch("native_kv_client.time.sleep"):
            self.assertEqual(retry_buffer(transport, BUFFER_APPEND, (1,), b"packet"), b"done")
        self.assertEqual(transport.call_args_list[0], transport.call_args_list[1])

    def test_failure_is_not_retried(self):
        transport = Mock(return_value=(6, 6, b""))
        with self.assertRaises(AssertionError):
            retry_buffer(transport, BUFFER_DRAIN, (1,))
        self.assertEqual(transport.call_count, 1)


class NativeStoreTest(unittest.TestCase):
    def setUp(self):
        import numpy as np
        from cemu_flexgen.native_kv_store import NativeKvStore
        from cemu_flexgen.sparf_group_layout import SparfGroupLayout
        self.array_module = np
        data = bytearray(PAGE_BYTES)
        struct.pack_into("<5I", data, 0, 0x31564b4e, 1, 0, 1, PAGE_BYTES)
        struct.pack_into("<3Q", data, 80, 64 * 1024**2, 65536, 1)
        self.transport = Mock(return_value=(0, 0, bytes(data)))
        self.store = NativeKvStore(SparfGroupLayout(128, 8), 4, 2, 2, 300, 2,
                                   transport_factory=lambda path: self.transport).open()
        self.addCleanup(self.store.close)

    def test_microbatch_coordinates_and_payload_tiles(self):
        from native_kv_client import coordinates
        keys = self.array_module.arange(10 * 2 * 128, dtype="float16").reshape(10, 2, 128)
        self.store.append_batch(12, 0, keys, -keys)
        calls = [call.args for call in self.transport.call_args_list if call.args[0] == BUFFER_APPEND]
        self.assertEqual(len(calls), 4)
        stats = self.store.statistics()
        self.assertEqual(stats["client_append_commands"], 4)
        self.assertEqual(stats["client_append_attempts"], 4)
        self.assertEqual(stats["client_append_backpressure_retries"], 0)
        self.assertEqual(stats["client_appended_head_tokens"], 20)
        for column in range(2):
            for part, (start, count) in enumerate(((0, 8), (8, 2))):
                _, fields, packet = calls[column * 2 + part]
                self.assertEqual(fields, coordinates(self.store.task, batch=2, layer=1,
                    head=column, token=start, valid=count))
                expected = keys[start:start + count, column].tobytes() + (-keys[start:start + count, column]).tobytes()
                self.assertEqual(packet, expected.ljust(PAGE_BYTES, b"\0"))
        with self.assertRaisesRegex(ValueError, "accepted prefix"):
            self.store.append_batch(12, 0, keys, keys)

    def test_binding_contains_native_identity_and_batch_origin(self):
        from cemu_flexgen import KvLayoutConfig, SparfKvCacheLayout
        layout = SparfKvCacheLayout(KvLayoutConfig(2, 300, 2, 2, 128, self.array_module.float16))
        metadata = self.store.pack_workflow(layout, 1, 17, 4, 16, 3, 8)
        self.assertTrue(struct.unpack_from("<I", metadata, 4)[0] & 8)
        self.assertEqual(struct.unpack_from("<4I", metadata, len(metadata) - 16),
                         (self.store.task, 1, 2, 0))
        for origin in (-8, 1, 16):
            with self.assertRaises(ValueError):
                self.store.pack_workflow(layout, 1, 17, 4, 16, 3, origin)

    def test_batched_append_coordinates_payload_and_boundaries(self):
        self.store._append_batch_bytes = 8192
        keys = self.array_module.arange(33 * 8 * 128, dtype="float32").reshape(33, 8, 128).astype("float16")
        self.store.append_batch(8, 0, keys, -keys)
        calls = [call.args for call in self.transport.call_args_list if call.args[0] == BUFFER_APPEND_BATCH]
        visited = set()
        for _, fields, packet in calls:
            count, heads, first = fields[0] >> 16, fields[4] >> 16, fields[5]
            batch, layer, head = fields[3] & 65535, fields[3] >> 16, fields[4] & 65535
            self.assertLessEqual(head + heads, self.store.heads)
            self.assertLessEqual(len(packet), 8192)
            column = layer * 4 + (batch - 2) * 2 + head
            expected = b"".join(array[first:first + count, column:column + heads]
                                .transpose(1, 0, 2).tobytes() for array in (keys, -keys))
            self.assertEqual(packet, expected)
            for token in range(first, first + count):
                for local in range(column, column + heads):
                    self.assertNotIn((token, local), visited)
                    visited.add((token, local))
        self.assertEqual(len(visited), 33 * 8)
        self.assertEqual(self.store.statistics()["client_appended_head_tokens"], 33 * 8)

    def test_opt13b_batched_command_count_and_unchanged_payload(self):
        from cemu_flexgen.native_kv_store import NativeKvStore
        from cemu_flexgen.sparf_group_layout import SparfGroupLayout
        data = bytearray(PAGE_BYTES)
        struct.pack_into("<5I", data, 0, 0x31564b4e, 1, 0, 1, PAGE_BYTES)
        struct.pack_into("<4Q", data, 80, 2**30, 2097152, 1, APPEND_MAX_BYTES)
        counts = {"commands": 0, "bytes": 0}

        def transport(opcode, fields, payload=None):
            if opcode == BUFFER_APPEND_BATCH:
                counts["commands"] += 1
                counts["bytes"] += len(payload)
            return 0, 0, data

        connection = Mock(side_effect=transport)
        store = NativeKvStore(SparfGroupLayout(128, 8), 4, 40, 40, 1040, 4,
                              budget=2**30, transport_factory=lambda path: connection).open()
        self.addCleanup(store.close)
        keys = self.array_module.zeros((1024, 160, 128), dtype="float16")
        store.append_batch(0, 0, keys, keys)
        self.assertEqual(counts["commands"], 768)
        for token in range(1024, 1040):
            store.append_batch(0, token, keys[:1], keys[:1])
        self.assertEqual(counts["commands"] * 40, 33280)
        self.assertEqual(counts["bytes"], 1040 * 160 * 128 * 2 * 2)

    def test_batched_retry_and_partial_failure(self):
        self.store._append_batch_bytes = APPEND_MAX_BYTES
        keys = self.array_module.ones((17, 2, 128), dtype="float16")
        self.transport.side_effect = [(12, 5, b""), (0, 0, b""), (6, 6, b"")]
        self.transport.reset_mock()
        with patch("cemu_flexgen.native_kv_store.time.sleep"), self.assertRaises(RuntimeError):
            self.store.append_batch(0, 0, keys, keys)
        self.assertEqual(self.transport.call_args_list[0], self.transport.call_args_list[1])
        self.assertEqual(self.store._accepted, {0: 16, 1: 16})
        with self.assertRaisesRegex(RuntimeError, "failed"):
            self.store.append_batch(0, 16, keys[:1], keys[:1])

    def test_append_statistics_distinguish_backpressure_from_payload(self):
        keys = self.array_module.ones((1, 1, 128), dtype="float16")
        self.transport.side_effect = [(12, 5, b""), (0, 0, b""), (0, 0, bytes(PAGE_BYTES))]
        with patch("cemu_flexgen.native_kv_store.time.sleep"):
            self.store.append_batch(0, 0, keys, keys)
        stats = self.store.statistics()
        self.assertEqual(stats["client_append_attempts"], 2)
        self.assertEqual(stats["client_append_commands"], 1)
        self.assertEqual(stats["client_append_backpressure_retries"], 1)
        self.assertEqual(stats["client_appended_head_tokens"], 1)

    def test_failed_partial_append_poisons_session(self):
        keys = self.array_module.ones((10, 1, 128), dtype="float16")
        self.transport.side_effect = [(0, 0, bytes(PAGE_BYTES)), (6, 6, b"")]
        with self.assertRaisesRegex(RuntimeError, "opcode"):
            self.store.append_batch(0, 0, keys, keys)
        with self.assertRaisesRegex(RuntimeError, "failed"):
            self.store.append_batch(0, 0, keys, keys)

    def test_close_does_not_claim_task_reclamation(self):
        self.store.close()
        with self.assertRaisesRegex(RuntimeError, "fresh controller"):
            self.store.open()


if __name__ == "__main__":
    unittest.main()
