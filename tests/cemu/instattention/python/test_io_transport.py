import importlib
import os
import pkgutil
import stat
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import cemu_io
from cemu_io import copy_file_range, io_backend
from native_kv_client import (
    BUFFER_APPEND_BATCH, BUFFER_DRAIN, INFO, LinuxTransport, PAGE_BYTES, Passthrough,
)
from cemu_flexgen.kv_staging import KvStagingManager


class IoTransportTest(unittest.TestCase):
    def test_pybind_read_into_rejects_unsafe_buffers_before_io(self):
        import numpy as np
        from _cemu_client import CemuClient

        # No NVMe management calls: /dev/null is enough to verify buffer checks
        # run before accessing a memory range or releasing the Python buffer.
        client = CemuClient("/dev/null", "/dev/null")
        self.addCleanup(client.close)
        readonly = np.zeros(8, dtype=np.float16)
        readonly.flags.writeable = False
        for bad in (readonly, np.zeros((2, 4))[:, ::2], np.empty(4, dtype=object)):
            with self.assertRaises((ValueError, BufferError)):
                client.read_range_into(0, bad)

    def test_read_into_reuses_output_and_rejects_unsafe_buffers(self):
        import numpy as np
        from cemu_flexgen.cemu_device import CemuDevice, RangeSpec
        from phase_profiler import PhaseProfiler

        for profiled in (False, True):
            profiler = PhaseProfiler(stack=True) if profiled else None
            device = CemuDevice("test", "./unused.so", "test", [RangeSpec("/unused", 4096)],
                                profiler=profiler)
            client = device._client = Mock()
            timings = dict(binding_prepare_wall=1, buffer_prepare_wall=2, io_wait_wall=3,
                           buffer_finish_wall=0, binding_finish_wall=5, io_bytes=512)

            def read(index, target, offset, **kwargs):
                target.fill(17)
                return (None, timings) if kwargs.get("profile") else None

            client.read_range_into.side_effect = read
            output = np.empty((2, 128), dtype=np.float16)
            self.assertIs(device.read_tensor_into(0, output), output)
            np.testing.assert_array_equal(output, np.full_like(output, 17))
            client.read_range.assert_not_called()
            readonly = output.copy()
            readonly.flags.writeable = False
            for bad in (readonly, output[:, ::2], np.empty(4, dtype=object)):
                with self.assertRaisesRegex(ValueError, "writable C-contiguous numeric"):
                    device.read_tensor_into(0, bad)
            self.assertEqual(client.read_range_into.call_count, 1)
            if profiler:
                metrics = {metric.name: metric for metric in profiler.snapshot()}
                self.assertEqual(metrics["attention.mr0.read.io_wait_wall"].total_ns, 3)
            device.close()

    def test_profiled_attention_client_preserves_data_and_execute_arguments(self):
        import numpy as np
        from cemu_flexgen.cemu_device import CemuDevice, RangeSpec
        from phase_profiler import PhaseProfiler

        profiler = PhaseProfiler(stack=True)
        device = CemuDevice("test", "./unused.so", "test", [RangeSpec("/unused", 4096)],
                            profiler=profiler, profile_range_names={0: "query"})
        client = Mock()
        device._client = client
        timings = dict(binding_prepare_wall=1, buffer_prepare_wall=2, io_wait_wall=3,
                       buffer_finish_wall=4, binding_finish_wall=5, io_bytes=512)
        data = np.arange(8, dtype=np.float32)
        client.write_range.return_value = None, timings
        client.read_range.return_value = data.view(np.uint8), timings
        client.execute.return_value = 17, timings
        device.write_tensor(0, data)
        self.assertEqual(client.write_range.call_args.kwargs, {"profile": True})
        np.testing.assert_array_equal(client.write_range.call_args.args[1], data)
        np.testing.assert_array_equal(device.read_tensor(0, data.shape, data.dtype), data)
        self.assertEqual(device.execute(123, 456, metadata=b"metadata"), 17)
        client.execute.assert_called_once_with(123, 456, 0, 0, 0, b"metadata", profile=True)
        metrics = {metric.name: metric for metric in profiler.snapshot()}
        for name in ("attention.query.write", "attention.query.read", "attention.execute"):
            self.assertEqual(metrics[name + ".io_wait_wall"].total_ns, 3)
            self.assertEqual(metrics[name + ".io_wait_wall"].count, 1)
        device.close()

    def setUp(self):
        self.environment = patch.dict(os.environ, {"CEMU_IO_BACKEND": "ioctl"})
        self.environment.start()
        self.addCleanup(self.environment.stop)

    def test_default_and_invalid_selection(self):
        os.environ.pop("CEMU_IO_BACKEND")
        self.assertEqual(io_backend(), "ioctl")
        os.environ["CEMU_IO_BACKEND"] = "typo"
        with self.assertRaisesRegex(ValueError, "ioctl or uring"):
            io_backend()

    def test_requested_uring_never_falls_back(self):
        os.environ["CEMU_IO_BACKEND"] = "uring"
        importer = SimpleNamespace(import_module=Mock(return_value=SimpleNamespace()))
        with patch.object(cemu_io, "importlib", importer), \
                patch("cemu_io.os.copy_file_range") as copy:
            with self.assertRaisesRegex(RuntimeError, "no ioctl fallback"):
                copy_file_range(1, 2, 512, offset_src=0, offset_dst=0)
            copy.assert_not_called()

    def test_dense_copy_partial_completions_keep_offsets(self):
        os.environ["CEMU_IO_BACKEND"] = "uring"
        module = Mock()
        module.uring_copy_file_range.side_effect = [512, 1024]
        importer = SimpleNamespace(import_module=Mock(return_value=module))
        with patch.object(cemu_io, "importlib", importer), \
                patch("cemu_io.os.copy_file_range") as copy:
            KvStagingManager._copy_file_range_all(3, 4, 1536, 4096, 8192)
            self.assertEqual(module.uring_copy_file_range.call_args_list[0].args,
                             (3, 4, 1536, 4096, 8192))
            self.assertEqual(module.uring_copy_file_range.call_args_list[1].args,
                             (3, 4, 1024, 4608, 8704))
            copy.assert_not_called()

    def test_dense_copy_zero_progress_fails(self):
        with patch("cemu_io.os.copy_file_range", return_value=0):
            with self.assertRaisesRegex(OSError, "no progress"):
                KvStagingManager._copy_file_range_all(3, 4, 512, 0, 0)

    def test_binding_mock_preserves_patch_target_resolution(self):
        original_import = importlib.import_module
        module = Mock()
        importer = SimpleNamespace(import_module=Mock(return_value=module))
        with patch.object(cemu_io, "importlib", importer):
            self.assertIs(cemu_io.uring_bindings(), module)
            self.assertIs(importlib.import_module, original_import)
            self.assertIs(pkgutil.resolve_name("cemu_io.os"), os)
            self.assertIs(pkgutil.resolve_name("native_kv_client.os"), os)
        importer.import_module.assert_called_once_with("_cemu_client")

    def test_native_payload_status_and_drain(self):
        os.environ["CEMU_IO_BACKEND"] = "uring"
        module = Mock()
        captured = []

        def execute(descriptor, packed, payload):
            command = Passthrough.from_buffer_copy(packed)
            captured.append((command, payload))
            self.assertEqual(descriptor, 42)
            self.assertEqual(command.nsid, 1)
            self.assertEqual(command.data_len, len(payload))
            return 2, 17, bytes([9]) * len(payload)

        module.uring_nvme.side_effect = execute
        importer = SimpleNamespace(import_module=Mock(return_value=module))
        with patch.object(cemu_io, "importlib", importer), \
                patch("native_kv_client.os.open", return_value=42) as open_device, \
                patch("native_kv_client.os.fstat", return_value=SimpleNamespace(st_mode=stat.S_IFCHR)), \
                patch("native_kv_client.os.close"), \
                patch("native_kv_client.ctypes.CDLL") as library:
            transport = LinuxTransport("/dev/ng0n1")
            try:
                open_device.assert_called_once_with("/dev/ng0n1", os.O_RDWR | os.O_CLOEXEC)
                self.assertEqual(transport.descriptor, 42)
                payload = bytes(range(256)) * 32
                fields = (1, 2, 3, 4, 5, 6)
                status, result, data = transport(BUFFER_APPEND_BATCH, fields, payload)
                self.assertEqual((status, result, len(data)), (2, 17, len(payload)))
                self.assertEqual(captured[-1][1], payload)
                self.assertEqual(captured[-1][0].cdw15, 6)
                status, result, data = transport(INFO, fields)
                self.assertEqual(data, bytes([9]) * PAGE_BYTES)
                status, result, data = transport(BUFFER_DRAIN, fields)
                self.assertEqual(captured[-1][1], b"")
                self.assertEqual(data, bytes(PAGE_BYTES))
                library.return_value.ioctl.assert_not_called()
                module.uring_nvme.side_effect = OSError("CQE failed")
                with self.assertRaisesRegex(OSError, "CQE failed"):
                    transport(INFO, fields)
                library.return_value.ioctl.assert_not_called()
            finally:
                transport.close()

    def test_async_append_returns_handle_without_waiting(self):
        module = Mock()
        handle = module.uring_nvme_submit.return_value
        transport = LinuxTransport.__new__(LinuxTransport)
        transport.backend = "uring"
        transport.descriptor = 42
        with patch("native_kv_client.uring_bindings", return_value=module):
            payload = bytes(range(256)) * 32
            request = transport.submit_append((1, 2, 3, 4, 5, 6), payload)
            self.assertIs(request, handle)
            descriptor, packed, received = module.uring_nvme_submit.call_args.args
            command = Passthrough.from_buffer_copy(packed)
            self.assertEqual((descriptor, command.opcode, command.nsid, command.data_len),
                             (42, BUFFER_APPEND_BATCH, 1, len(payload)))
            self.assertEqual(received, payload)
            handle.result.assert_not_called()
            transport.wait_any([request])
            module.uring_wait_any.assert_called_once_with([request])
            module.uring_nvme.assert_not_called()

    def test_async_append_requires_new_binding_and_uring(self):
        transport = LinuxTransport.__new__(LinuxTransport)
        transport.backend = "ioctl"
        with self.assertRaisesRegex(RuntimeError, "requires io_uring"):
            transport.submit_append((1, 2, 3, 4, 5, 6), b"x")
        transport.backend = "uring"
        with patch("native_kv_client.uring_bindings", return_value=SimpleNamespace()):
            with self.assertRaisesRegex(RuntimeError, "rebuild.*no ioctl fallback"):
                transport.submit_append((1, 2, 3, 4, 5, 6), b"x")


if __name__ == "__main__":
    unittest.main()
