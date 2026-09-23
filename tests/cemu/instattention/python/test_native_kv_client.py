import stat
import struct
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from native_kv_client import (
    BUFFER_APPEND, BUFFER_DRAIN, INFO, LinuxTransport, PAGE_BYTES,
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
