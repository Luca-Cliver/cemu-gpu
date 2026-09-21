"""Guest transport failures must not allow retrying a partially accepted append."""
import unittest
from unittest.mock import Mock

import numpy as np

from cemu_flexgen.sparf_group_device import SparfGroupedDevice, HEADER, CLOSE
from cemu_flexgen.sparf_group_layout import SparfGroupLayout


class GroupTransportTest(unittest.TestCase):
    def make_device(self):
        device = SparfGroupedDevice(SparfGroupLayout(128, 8), "/unused/pool",
                                    4096, "/unused/fdm", "unused.so", buffer_streams=2)
        device.device = Mock()
        device.registered = True
        device.buffer_bytes = 16 * 2 * 128 * 2  # force both token and stream tiling
        status = np.zeros(20, np.uint64)
        status[8] = 1
        device.device.read_tensor.return_value = status
        device.device.execute.return_value = 0
        return device

    def test_partial_append_failures_block_retry_and_attention_but_allow_close(self):
        keys = np.ones((17, 2, 128), np.float16)
        for failure in ("write", "execute", "status"):
            with self.subTest(failure=failure):
                device = self.make_device()
                client = device.device
                if failure == "write":
                    client.write_tensor.side_effect = [None, OSError("injected write")]
                elif failure == "execute":
                    client.execute.side_effect = [0, RuntimeError("injected execute")]
                else:
                    client.read_tensor.side_effect = [client.read_tensor.return_value,
                                                       OSError("injected status")]
                with self.assertRaisesRegex((OSError, RuntimeError), "injected"):
                    device.append_batch(0, 0, keys, keys)
                commands, writes = client.execute.call_count, client.write_tensor.call_count
                for operation in (lambda: device.append_batch(0, 0, keys, keys),
                                  device.check_ready, device.flush):
                    with self.assertRaisesRegex(RuntimeError, "poisoned"):
                        operation()
                self.assertEqual(client.execute.call_count, commands)
                self.assertEqual(client.write_tensor.call_count, writes)
                client.execute.side_effect = client.read_tensor.side_effect = None
                device.close()
                self.assertEqual(HEADER.unpack_from(client.execute.call_args.kwargs["metadata"])[1], CLOSE)
                client.close.assert_called_once()
                self.assertIsNone(device.device)

    def test_invalid_ranges_fail_before_transport_without_poisoning(self):
        device = self.make_device()
        keys = np.ones((1, 2, 128), np.float16)
        for stream, token in ((-1, 0), (True, 0), (0, 2**64), (0, -1), (0, True)):
            with self.subTest(stream=stream, token=token), self.assertRaises(ValueError):
                device.append_batch(stream, token, keys, keys)
        device.device.write_tensor.assert_not_called()
        device.check_ready()
        device.append_batch(0, 0, keys, keys)


if __name__ == "__main__":
    unittest.main()
