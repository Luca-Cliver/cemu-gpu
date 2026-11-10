import stat
import struct
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from native_kv_client import (
    BUFFER_APPEND, BUFFER_DRAIN, INFO, LinuxTransport, PAGE_BYTES,
    BUFFER_APPEND_BATCH, APPEND_MAX_BYTES,
    require_fresh_namespace, retry_buffer, run_buffer_contract, run_contract,
)


class NativeKVTransportTest(unittest.TestCase):
    def test_fresh_namespace_check_only_issues_info(self):
        data = bytearray(PAGE_BYTES)
        struct.pack_into("<5I", data, 0, 0x31564B4E, 1, 0, 1, PAGE_BYTES)
        transport = Mock(return_value=(0, 0, bytes(data)))
        self.assertEqual(require_fresh_namespace(transport), (0, 1))
        transport.assert_called_once_with(INFO, (1, 0, 0, 0, 0, 0), None)

    def test_used_namespace_rejected_even_after_session_files_are_deleted(self):
        for index, name in enumerate(("reserved_pages", "written_pages", "failed_pages", "claimed_blocks")):
            with self.subTest(counter=name):
                data = bytearray(PAGE_BYTES)
                struct.pack_into("<5I", data, 0, 0x31564B4E, 1, 0, 1, PAGE_BYTES)
                struct.pack_into("<Q", data, 40 + 8 * index, 1)
                transport = Mock(return_value=(0, 0, bytes(data)))
                with self.assertRaisesRegex(RuntimeError, name + "=1.*relaunch Host QEMU"):
                    require_fresh_namespace(transport)
                transport.assert_called_once_with(INFO, (1, 0, 0, 0, 0, 0), None)

    def test_fresh_namespace_check_rejects_invalid_controller_and_info_error(self):
        transport = Mock(return_value=(0, 0, bytes(PAGE_BYTES)))
        with self.assertRaisesRegex(RuntimeError, "refusing writes"):
            require_fresh_namespace(transport)
        transport.assert_called_once_with(INFO, (1, 0, 0, 0, 0, 0), None)
        transport = Mock(return_value=(12, 8, bytes(PAGE_BYTES)))
        with self.assertRaises(AssertionError):
            require_fresh_namespace(transport)
        transport.assert_called_once_with(INFO, (1, 0, 0, 0, 0, 0), None)

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
        transport.backend = "ioctl"
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

    def async_transport(self, outcome=None):
        self.store._append_batch_bytes = 8192
        self.store._append_queue_depth = 3
        self.transport.backend = "uring"
        active_heads = set()
        accepted = {}
        events = []
        requests = []

        def submit(fields, payload):
            count, heads = fields[0] >> 16, fields[4] >> 16
            first_head = fields[4] & 65535
            identities = [(fields[3], head) for head in range(first_head, first_head + heads)]
            for identity in identities:
                self.assertNotIn(identity, active_heads)
                self.assertEqual(accepted.get(identity, 0), fields[5])
            active_heads.update(identities)
            events.append(("submit", fields, payload))

            def complete():
                active_heads.difference_update(identities)
                events.append(("complete", fields, payload))
                status, result = outcome(fields) if outcome else (0, 0)
                if not status:
                    for identity in identities:
                        accepted[identity] = fields[5] + count
                return status, result, b""

            request = SimpleNamespace(result=Mock(side_effect=complete))
            requests.append(request)
            return request

        self.transport.submit_append.side_effect = submit
        self.transport.wait_any.side_effect = lambda active: len(active) - 1
        return events, requests, active_heads, accepted

    def test_async_prefill_window_order_payload_and_refill(self):
        events, requests, active_heads, accepted = self.async_transport()
        keys = self.array_module.arange(33 * 8 * 128, dtype="float32").reshape(33, 8, 128).astype("float16")
        self.store.append_batch(8, 0, keys, -keys)
        self.assertEqual([event[0] for event in events[:4]], ["submit"] * 3 + ["complete"])
        self.assertEqual(events[4][0], "submit")
        self.assertFalse(active_heads)
        self.assertEqual(list(accepted.values()), [33] * 8)
        self.assertEqual(self.store._append_peak_inflight, 3)
        self.assertEqual(self.store._append_commands, 24)
        self.assertEqual(self.store._appended_head_tokens, 33 * 8)
        for request in requests:
            request.result.assert_called_once()
        for event, fields, packet in events:
            if event != "submit":
                continue
            batch, layer, head = fields[3] & 65535, fields[3] >> 16, fields[4] & 65535
            first, count, heads = fields[5], fields[0] >> 16, fields[4] >> 16
            column = layer * 4 + (batch - 2) * 2 + head
            expected = b"".join(array[first:first + count, column:column + heads]
                                .transpose(1, 0, 2).tobytes() for array in (keys, -keys))
            self.assertEqual(packet, expected)
            self.assertLessEqual(len(packet), 8192)
        self.assertEqual(self.store.statistics()["client_append_peak_inflight"], 3)

    def test_async_backpressure_retries_only_unaccepted_packet(self):
        attempts = {}

        def outcome(fields):
            attempts[fields] = attempts.get(fields, 0) + 1
            return (0xC, 5) if fields[5] == 0 and attempts[fields] == 1 else (0, 0)

        _, requests, active_heads, accepted = self.async_transport(outcome)
        keys = self.array_module.ones((33, 8, 128), dtype="float16")
        with patch("cemu_flexgen.native_kv_store.time.sleep"):
            self.store.append_batch(8, 0, keys, keys)
        self.assertEqual(self.store._append_retries, 8)
        self.assertEqual(self.store._append_commands, 24)
        self.assertEqual(self.store._append_attempts, 32)
        self.assertEqual(list(accepted.values()), [33] * 8)
        self.assertFalse(active_heads)
        for request in requests:
            request.result.assert_called_once()

    def test_async_multihead_payload_and_second_append(self):
        events, _, active_heads, accepted = self.async_transport()
        self.store._append_batch_bytes = 16384
        keys = self.array_module.arange(33 * 8 * 128, dtype="float32").reshape(33, 8, 128).astype("float16")
        self.store.append_batch(8, 0, keys, -keys)
        self.store.append_batch(8, 33, keys[:2], -keys[:2])
        self.assertFalse(active_heads)
        self.assertEqual(list(accepted.values()), [35] * 8)
        self.assertEqual(self.store._append_commands, 16)
        self.assertEqual(self.store._appended_head_tokens, 35 * 8)
        for event, fields, packet in events:
            if event != "submit":
                continue
            self.assertEqual(fields[4] >> 16, 2)
            batch, layer = fields[3] & 65535, fields[3] >> 16
            column = layer * 4 + (batch - 2) * 2
            first, count = fields[5], fields[0] >> 16
            start = first if first < 33 else first - 33
            expected = b"".join(array[start:start + count, column:column + 2]
                                .transpose(1, 0, 2).tobytes() for array in (keys, -keys))
            self.assertEqual(packet, expected)
            self.assertLessEqual(len(packet), 16384)

    def test_async_window_validation(self):
        from cemu_flexgen.native_kv_store import NativeKvStore
        for depth in (0, 65, True, 1.5):
            with self.subTest(depth=depth), self.assertRaisesRegex(ValueError, "queue depth"):
                NativeKvStore(self.store.layout, 4, 2, 2, 600, 2, append_queue_depth=depth)

    def test_async_error_drains_requests_and_poisons_store(self):
        _, requests, active_heads, _ = self.async_transport(lambda fields: (2, 7))
        keys = self.array_module.ones((33, 8, 128), dtype="float16")
        with self.assertRaisesRegex(RuntimeError, "status=0x2"):
            self.store.append_batch(8, 0, keys, keys)
        self.assertEqual(len(requests), 3)
        self.assertFalse(active_heads)
        for request in requests:
            request.result.assert_called_once()
        with self.assertRaisesRegex(RuntimeError, "closed or failed"):
            self.store.append_batch(8, 0, keys, keys)

    def test_async_submit_failure_drains_earlier_requests(self):
        _, requests, active_heads, _ = self.async_transport()
        submit = self.transport.submit_append.side_effect

        def fail_second(fields, payload):
            if requests:
                raise OSError("submission failed")
            return submit(fields, payload)

        self.transport.submit_append.side_effect = fail_second
        keys = self.array_module.ones((33, 8, 128), dtype="float16")
        with self.assertRaisesRegex(OSError, "submission failed"):
            self.store.append_batch(8, 0, keys, keys)
        self.assertFalse(active_heads)
        requests[0].result.assert_called_once()

    def test_async_backpressure_timeout_drains(self):
        _, requests, active_heads, _ = self.async_transport(lambda fields: (0xC, 5))
        keys = self.array_module.ones((33, 8, 128), dtype="float16")
        with patch("cemu_flexgen.native_kv_store.time.monotonic", side_effect=[0, 0, 0, 121]):
            with self.assertRaisesRegex(TimeoutError, "backpressure"):
                self.store.append_batch(8, 0, keys, keys)
        self.assertFalse(active_heads)
        for request in requests:
            request.result.assert_called_once()

    def test_decode_and_depth_one_keep_synchronous_submission(self):
        for tokens, depth, backend in ((1, 3, "uring"), (33, 1, "uring"), (33, 3, "ioctl")):
            with self.subTest(tokens=tokens, depth=depth, backend=backend):
                self.store._accepted.clear()
                self.store._append_batch_bytes = 8192
                self.store._append_queue_depth = depth
                self.transport.backend = backend
                keys = self.array_module.ones((tokens, 8, 128), dtype="float16")
                self.store.append_batch(8, 0, keys, keys)
                self.transport.submit_append.assert_not_called()

    def test_async_completion_exception_drains(self):
        def outcome(fields):
            raise OSError("CQE failed")

        _, requests, active_heads, _ = self.async_transport(outcome)
        keys = self.array_module.ones((33, 8, 128), dtype="float16")
        with self.assertRaisesRegex(OSError, "CQE failed"):
            self.store.append_batch(8, 0, keys, keys)
        self.assertFalse(active_heads)
        for request in requests:
            request.result.assert_called_once()

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
