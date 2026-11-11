from contextlib import ExitStack
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

import flexgen_cemu


class FlexgenInputSlotsTest(unittest.TestCase):
    def arguments(self, *extra):
        argv = ["flexgen_cemu", "--model-dir", "model", "--weights-dir", "weights",
                "--program", "program", "--runtime-config", "runtime",
                "--softmax-ratio-file", "ratio", "--profile-output", "profile",
                "--result-output", "result", *extra]
        with patch.object(sys, "argv", argv):
            return flexgen_cemu.arguments()

    def test_legacy_default_and_tagged_options(self):
        args = self.arguments()
        self.assertEqual(args.fdm_input_slots, 0)
        self.assertIsNone(args.fdm_input_bytes)
        args = self.arguments("--fdm-input-slots", "2", "--fdm-input-bytes", "2097152")
        self.assertEqual(args.fdm_input_slots, 2)
        self.assertEqual(args.fdm_input_bytes, 2097152)

    def test_backend_receives_slot_configuration(self):
        args = self.arguments("--fdm-input-slots", "2", "--fdm-input-bytes", "2097152")
        store = SimpleNamespace(input_bytes=2097152, input_slots=2, data_bytes=args.fdm_data_bytes)
        factory = Mock(return_value=store)
        modules = {
            "cemu_flexgen.fdm_kvbuffer_store": SimpleNamespace(FDM_KVbuffer=factory),
            "cemu_flexgen.native_kv_store": SimpleNamespace(NativeKvStore=Mock()),
            "cemu_flexgen.cemu_sparf_device": SimpleNamespace(CemuSparfDevice=Mock()),
            "cemu_flexgen.cemu_sparf_slot_scheduler": SimpleNamespace(CemuSparfSlotScheduler=Mock()),
            "flexgen_adapter.sparf_attention": SimpleNamespace(SparfGroupedAttentionBackend=Mock()),
        }
        config = SimpleNamespace(input_dim=128, n_head=1, num_hidden_layers=1, dtype=np.float16)
        with patch.dict(sys.modules, modules):
            result, _ = flexgen_cemu.open_backends(Mock(spec=ExitStack), args, config, "/unused", None, None)
        self.assertIs(result, store)
        self.assertEqual(factory.call_args.kwargs["input_slots"], 2)
        self.assertEqual(factory.call_args.kwargs["input_bytes"], 2097152)

    def test_guest_budget_preserves_staging_capacity(self):
        script = Path(__file__).resolve().parents[4] / "femu-scripts/flexgen-b8-guest-test.sh"
        prefix = script.read_text().split("stack_trace=", 1)[0]
        for slots in (0, 1, 2):
            with self.subTest(slots=slots):
                result = subprocess.run(["bash", "-c", prefix +
                    '\nprintf "%s %s" "$CEMU_FLEXGEN_FDM_INPUT_BYTES" "$CEMU_FLEXGEN_FDM_DATA_BYTES"'],
                    env=dict(os.environ, CEMU_FLEXGEN_BATCH_SIZE="8", CEMU_FLEXGEN_MICROBATCH_SIZE="4",
                             CEMU_FLEXGEN_DECODE_STEPS="16", CEMU_FLEXGEN_DEVICE_PROFILE="b8-p1024-d16",
                             CEMU_FLEXGEN_FDM_INPUT_SLOTS=str(slots)), capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                width, data = map(int, result.stdout.split())
                self.assertEqual(width, 85197312)
                aligned = lambda size: (size + 4095) // 4096 * 4096
                self.assertGreaterEqual(data - aligned(max(1, slots) * width),
                                        8 * 1024**3 - aligned(width))
                self.assertLess(data + 128 * 1024**2 + 256 * 1024**2, 16 * 1024**3)


if __name__ == "__main__":
    unittest.main()
