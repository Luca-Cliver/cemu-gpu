import sys
import threading
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "build-guest"))

from cemu_flexgen.cemu_sparf_device import CemuSparfDevice
from cemu_flexgen import KvLayoutConfig, SparfKvCacheLayout, align_up
from cemu_flexgen.sparf_group_capacity import grouped_attention_reserve_bytes
from cemu_flexgen.sparf_workflow_abi import SPARF_WORKFLOW_COMMAND
import test_opt_cemu_pipeline as pipeline
from opt_inputs import comparison_case, load_reference, prepare_prompt, write_json


class OptSparfIntegrationTest(unittest.TestCase):
    def arguments(self, *options):
        with patch.object(sys, "argv", ["opt-test", "--model-dir", "/unused", *options]):
            return pipeline.parse_args()

    def test_old_defaults_remain_disabled(self):
        args = self.arguments()
        self.assertFalse(args.sparf_autonomous)
        self.assertFalse(args.sparf_device_writeback)
        self.assertFalse(args.sparf_native)

    def test_fixed_prompt_matches_legacy_generator(self):
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory)
            write_json(model / "config.json", dict(vocab_size=32, bos_token_id=7, pad_token_id=1,
                                                   max_position_embeddings=64))
            prompt, _ = prepare_prompt(model, 4, 11, 16)
            old = pipeline.make_token_ids(SimpleNamespace(vocab_size=32, bos_token_id=7), 4, 11)
            np.testing.assert_array_equal(prompt, old.numpy())

    def test_functional_result_can_be_used_as_framework_reference(self):
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory)
            write_json(model / "config.json", dict(vocab_size=32, bos_token_id=2, pad_token_id=1,
                                                   max_position_embeddings=64))
            args = self.arguments("--attention-mode", "sparf", "--functional", "--decode-steps", "2",
                                  "--result-output", str(model / "result.json"))
            args.model_dir = model
            prompt, _ = prepare_prompt(model, 1, 8, 2)
            prefill = Mock()
            prefill.run.return_value.next_token_ids = pipeline.torch.tensor([[5]])
            generated = pipeline.torch.tensor([[5, 6, 7]])
            with patch.object(pipeline, "OptGenerationRunner") as runner, \
                    patch.object(pipeline, "open_attention_schedulers"), \
                    patch.object(pipeline, "synchronize"), patch.object(pipeline, "log"):
                runner.return_value.run.return_value.token_ids = generated
                pipeline.run_functional_pipeline(args, pipeline.torch.from_numpy(prompt), 8,
                    prefill, Mock(), [], Mock(), [], 2, 1, None, "cpu")
            case = comparison_case(model, prompt, 2, 16, 8)
            np.testing.assert_array_equal(load_reference(args.result_output, case), generated.numpy())

    def test_invalid_modes_fail_before_device_or_checkpoint_access(self):
        cases = (
            (["--sparf-native"], "native KV requires"),
            (["--attention-mode", "sparf", "--functional", "--sparf-autonomous",
              "--sparf-native", "--sparf-grouped"], "native KV requires"),
            (["--sparf-autonomous"], "requires SparF"),
            (["--attention-mode", "sparf", "--sparf-autonomous", "--sparf-staged"], "cannot be combined"),
            (["--attention-mode", "sparf", "--sparf-autonomous", "--benchmark"], "requires --functional"),
            (["--attention-mode", "sparf", "--functional", "--sparf-device-writeback"], "KV buffer"),
            (["--attention-mode", "sparf", "--functional", "--sparf-autonomous",
              "--attention-slots", "1"], "at least two slots"),
        )
        for options, message in cases:
            with self.subTest(options=options), patch.object(pipeline, "parse_args",
                    return_value=self.arguments(*options)), patch.object(pipeline.torch.cuda, "is_available") as cuda:
                with self.assertRaisesRegex(ValueError, message):
                    pipeline.main()
                cuda.assert_not_called()

    def test_combined_mode_reaches_gpu_check(self):
        args = self.arguments("--attention-mode", "sparf", "--functional", "--sparf-autonomous",
                              "--sparf-device-writeback", "--sparf-kv-buffer-tokens", "6")
        with patch.object(pipeline, "parse_args", return_value=args), \
             patch.object(pipeline.torch.cuda, "is_available", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "Guest GPU"):
                pipeline.main()

    def test_native_mode_reaches_gpu_check(self):
        args = self.arguments("--attention-mode", "sparf", "--functional", "--sparf-autonomous", "--sparf-native")
        with patch.object(pipeline, "parse_args", return_value=args), \
             patch.object(pipeline.torch.cuda, "is_available", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "Guest GPU"):
                pipeline.main()

    def test_grouped_fdm_preflight_precedes_checkpoint_and_device_allocation(self):
        config = SimpleNamespace(dtype=pipeline.torch.float16, num_hidden_layers=2,
                                 num_attention_heads=4, num_key_value_heads=4,
                                 head_dim=64, max_seq_len=2048)
        options = ("--attention-mode", "sparf", "--functional", "--sparf-autonomous", "--sparf-grouped")
        cases = (((), "SPARF_GROUP_FDM_BUDGET_BYTES"),
                 (("--sparf-group-fdm-budget-bytes", "1"), "budget is too small"))
        for extra, message in cases:
            with self.subTest(extra=extra), \
                    patch.object(pipeline, "parse_args", return_value=self.arguments(*options, *extra)), \
                    patch.object(pipeline.torch.cuda, "is_available", return_value=True), \
                    patch.object(pipeline, "require_directory", return_value=Path("/unused")), \
                    patch.object(Path, "is_file", return_value=True), \
                    patch.object(pipeline.OptConfig, "from_json", return_value=config), \
                    patch.object(pipeline, "make_token_ids", return_value=None), \
                    patch.object(pipeline, "OptCheckpointLoader") as loader, \
                    patch.object(pipeline, "SparfGroupedDevice") as storage, \
                    patch("cemu_flexgen.sparf_group_capacity.shutil.disk_usage", side_effect=(
                        SimpleNamespace(total=1 << 30, used=0, free=1 << 30),
                        SimpleNamespace(total=0, used=0, free=0),
                    )):
                with self.assertRaisesRegex(ValueError, message):
                    pipeline.main()
                loader.assert_not_called()
                storage.assert_not_called()

    def test_attention_preflight_matches_actual_range_file_allocations(self):
        for batch in (1, 2, 4):
            with self.subTest(batch=batch):
                layout = SparfKvCacheLayout(KvLayoutConfig(2, 19, batch, 2, 64, np.float16))
                device = CemuSparfDevice(
                    layout, "unused-k", "unused-c", "unused-v", "/unused", "unused.so",
                    query_heads=4, top_r=128, compression_ratio=8, autonomous=True,
                )
                actual = 2 * sum(align_up(spec.size, 4096) for spec in device._config["ranges"])
                planned = grouped_attention_reserve_bytes(
                    attention_slots=2, batch_size=batch, query_heads=4, kv_heads=2,
                    head_dim=64, element_bytes=2, max_seq_len=19, top_r=128, compression_ratio=8,
                )
                self.assertEqual(planned, actual)

    def autonomous_device(self):
        device = CemuSparfDevice.__new__(CemuSparfDevice)
        device._lock = threading.RLock()
        device.autonomous = True
        device.profiler = None
        device._device = Mock()
        request = SimpleNamespace(metadata=b"v3-request", completed_stages=0)
        device._active = request
        return device, request

    def test_autonomous_slot_uses_one_execute_and_rejects_repeat(self):
        device, request = self.autonomous_device()
        device.execute_autonomous(request)
        device._device.execute.assert_called_once_with(
            cparam1=SPARF_WORKFLOW_COMMAND, metadata=request.metadata)
        self.assertEqual(request.completed_stages, 4)
        with self.assertRaisesRegex(RuntimeError, "no autonomous"):
            device.execute_autonomous(request)

    def test_autonomous_execute_error_clears_lease(self):
        device, request = self.autonomous_device()
        device._device.execute.side_effect = RuntimeError("injected transport failure")
        with self.assertRaisesRegex(RuntimeError, "injected transport"):
            device.execute_autonomous(request)
        self.assertIsNone(device._active)


if __name__ == "__main__":
    unittest.main()
