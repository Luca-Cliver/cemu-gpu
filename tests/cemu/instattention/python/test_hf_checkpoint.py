import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

from flexgen_runtime import FlexGenLlamaConfig
from flexgen_runtime.hf_checkpoint import FlexGenHfCheckpointLoader


class _FakeSafeTensorHandle:
    def __init__(self, tensors):
        self.tensors = tensors
        self.enter_count = 0
        self.exit_count = 0

    def __enter__(self):
        self.enter_count += 1
        return self

    def __exit__(self, exception_type, exception, traceback):
        self.exit_count += 1

    def keys(self):
        return self.tensors.keys()

    def get_tensor(self, tensor_name):
        return self.tensors[tensor_name]


class _FakeSafeOpen:
    def __init__(self, tensors_by_path):
        self.tensors_by_path = tensors_by_path
        self.open_count = {}
        self.handles = {}

    def __call__(self, path, framework, device):
        self.assert_call_arguments(framework, device)
        resolved_path = str(Path(path).resolve())
        self.open_count[resolved_path] = self.open_count.get(resolved_path, 0) + 1
        handle = _FakeSafeTensorHandle(self.tensors_by_path[resolved_path])
        self.handles[resolved_path] = handle
        return handle

    @staticmethod
    def assert_call_arguments(framework, device):
        if framework != "pt" or device != "cpu":
            raise AssertionError("unexpected safe_open arguments")


class FlexGenHfCheckpointLoaderTest(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.model_directory = Path(self.temporary_directory.name)
        self.config = FlexGenLlamaConfig(
            num_hidden_layers=2,
            hidden_size=8,
            num_attention_heads=4,
            num_key_value_heads=2,
            intermediate_size=12,
            vocab_size=16,
            dtype=torch.float32,
        )

    def tearDown(self):
        self.temporary_directory.cleanup()

    def test_single_file_loads_and_caches_complete_layer(self):
        shard_path = self.model_directory / "model.safetensors"
        shard_path.touch()
        tensors = self._build_tensors()
        fake_safe_open = _FakeSafeOpen({str(shard_path.resolve()): tensors})

        with mock.patch(
            "flexgen_runtime.hf_checkpoint._safe_open",
            fake_safe_open,
        ):
            with FlexGenHfCheckpointLoader(
                self.config,
                self.model_directory,
            ) as loader:
                embedding = loader.load_embedding()
                first = loader.load_layer(0)
                second = loader.load_layer(0)
                final_norm = loader.load_final_norm()
                lm_head = loader.load_lm_head()

                self.assertIs(first, second)
                self.assertEqual(loader.cached_layer_count, 1)
                self.assertEqual(tuple(embedding.shape), (16, 8))
                self.assertEqual(tuple(first.attention.key.shape), (4, 8))
                self.assertEqual(tuple(first.mlp.down.shape), (8, 12))
                self.assertEqual(tuple(final_norm.shape), (8,))
                self.assertEqual(tuple(lm_head.shape), (16, 8))
                self.assertEqual(fake_safe_open.open_count[str(shard_path.resolve())], 1)

            self.assertEqual(
                fake_safe_open.handles[str(shard_path.resolve())].exit_count,
                1,
            )

    def test_sharded_index_opens_each_shard_once(self):
        tensors = self._build_tensors()
        shard_names = ("model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors")
        shard_paths = [self.model_directory / name for name in shard_names]
        for shard_path in shard_paths:
            shard_path.touch()

        weight_map = {}
        tensors_by_path = {}
        for index, shard_path in enumerate(shard_paths):
            shard_tensors = {
                name: tensor
                for position, (name, tensor) in enumerate(tensors.items())
                if position % 2 == index
            }
            tensors_by_path[str(shard_path.resolve())] = shard_tensors
            weight_map.update({name: shard_path.name for name in shard_tensors})
        with (self.model_directory / "model.safetensors.index.json").open(
            "w",
            encoding="utf-8",
        ) as index_file:
            json.dump({"weight_map": weight_map}, index_file)

        fake_safe_open = _FakeSafeOpen(tensors_by_path)
        with mock.patch(
            "flexgen_runtime.hf_checkpoint._safe_open",
            fake_safe_open,
        ):
            with FlexGenHfCheckpointLoader(
                self.config,
                self.model_directory,
            ) as loader:
                loader.load_layer(0)
                loader.load_layer(1)
                loader.load_embedding()
                loader.load_final_norm()
                loader.load_lm_head()

        for shard_path in shard_paths:
            self.assertEqual(fake_safe_open.open_count[str(shard_path.resolve())], 1)

    def test_lm_head_falls_back_to_tied_embedding(self):
        shard_path = self.model_directory / "model.safetensors"
        shard_path.touch()
        tensors = self._build_tensors()
        del tensors["lm_head.weight"]
        fake_safe_open = _FakeSafeOpen({str(shard_path.resolve()): tensors})

        with mock.patch(
            "flexgen_runtime.hf_checkpoint._safe_open",
            fake_safe_open,
        ):
            with FlexGenHfCheckpointLoader(
                self.config,
                self.model_directory,
            ) as loader:
                self.assertIs(loader.load_lm_head(), loader.load_embedding())

    def test_rejects_wrong_tensor_shape(self):
        shard_path = self.model_directory / "model.safetensors"
        shard_path.touch()
        tensors = self._build_tensors()
        tensors["model.layers.0.self_attn.q_proj.weight"] = torch.zeros(7, 8)
        fake_safe_open = _FakeSafeOpen({str(shard_path.resolve()): tensors})

        with mock.patch(
            "flexgen_runtime.hf_checkpoint._safe_open",
            fake_safe_open,
        ):
            with FlexGenHfCheckpointLoader(
                self.config,
                self.model_directory,
            ) as loader:
                with self.assertRaisesRegex(ValueError, "expected \(8, 8\)"):
                    loader.load_layer(0)

    def _build_tensors(self):
        tensors = {
            "model.embed_tokens.weight": torch.arange(16 * 8).reshape(16, 8),
            "model.norm.weight": torch.arange(8),
            "lm_head.weight": torch.arange(16 * 8).reshape(16, 8) + 1000,
        }
        for layer in range(self.config.num_hidden_layers):
            prefix = f"model.layers.{layer}."
            tensors.update(
                {
                    prefix + "self_attn.q_proj.weight": torch.zeros(8, 8) + layer,
                    prefix + "self_attn.k_proj.weight": torch.zeros(4, 8) + layer,
                    prefix + "self_attn.v_proj.weight": torch.zeros(4, 8) + layer,
                    prefix + "self_attn.o_proj.weight": torch.zeros(8, 8) + layer,
                    prefix + "input_layernorm.weight": torch.zeros(8) + layer,
                    prefix + "post_attention_layernorm.weight": torch.zeros(8) + layer,
                    prefix + "mlp.gate_proj.weight": torch.zeros(12, 8) + layer,
                    prefix + "mlp.up_proj.weight": torch.zeros(12, 8) + layer,
                    prefix + "mlp.down_proj.weight": torch.zeros(8, 12) + layer,
                }
            )
        return tensors


if __name__ == "__main__":
    unittest.main()
