import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from opt_runtime import OptCheckpointLoader, OptConfig


def make_config() -> OptConfig:
    return OptConfig(
        name="test-opt",
        num_hidden_layers=2,
        max_position_embeddings=6,
        hidden_size=4,
        num_attention_heads=2,
        ffn_dim=8,
        vocab_size=7,
        word_embed_proj_dim=4,
        dtype=torch.float32,
    )


def make_state(config: OptConfig):
    state = {
        "model.decoder.embed_tokens.weight": torch.arange(
            config.vocab_size * config.hidden_size,
            dtype=config.dtype,
        ).reshape(config.vocab_size, config.hidden_size),
        "model.decoder.embed_positions.weight": torch.zeros(
            config.max_position_embeddings + config.position_offset,
            config.hidden_size,
            dtype=config.dtype,
        ),
        "model.decoder.final_layer_norm.weight": torch.ones(
            config.hidden_size,
            dtype=config.dtype,
        ),
        "model.decoder.final_layer_norm.bias": torch.zeros(
            config.hidden_size,
            dtype=config.dtype,
        ),
    }
    for layer in range(config.num_hidden_layers):
        prefix = f"model.decoder.layers.{layer}."
        attention_prefix = prefix + "self_attn."
        for projection in ("q_proj", "k_proj", "v_proj", "out_proj"):
            state[attention_prefix + projection + ".weight"] = torch.eye(
                config.hidden_size,
                dtype=config.dtype,
            )
            state[attention_prefix + projection + ".bias"] = torch.zeros(
                config.hidden_size,
                dtype=config.dtype,
            )
        for norm in ("self_attn_layer_norm", "final_layer_norm"):
            state[prefix + norm + ".weight"] = torch.ones(
                config.hidden_size,
                dtype=config.dtype,
            )
            state[prefix + norm + ".bias"] = torch.zeros(
                config.hidden_size,
                dtype=config.dtype,
            )
        state[prefix + "fc1.weight"] = torch.zeros(
            config.ffn_dim,
            config.hidden_size,
            dtype=config.dtype,
        )
        state[prefix + "fc1.bias"] = torch.zeros(
            config.ffn_dim,
            dtype=config.dtype,
        )
        state[prefix + "fc2.weight"] = torch.zeros(
            config.hidden_size,
            config.ffn_dim,
            dtype=config.dtype,
        )
        state[prefix + "fc2.bias"] = torch.zeros(
            config.hidden_size,
            dtype=config.dtype,
        )
    return state


class OptCheckpointLoaderTest(unittest.TestCase):
    def test_config_from_json(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(
                json.dumps(
                    {
                        "_name_or_path": "facebook/opt-test",
                        "model_type": "opt",
                        "num_hidden_layers": 2,
                        "max_position_embeddings": 6,
                        "hidden_size": 4,
                        "num_attention_heads": 2,
                        "ffn_dim": 8,
                        "vocab_size": 7,
                        "word_embed_proj_dim": 4,
                        "torch_dtype": "float16",
                    }
                ),
                encoding="utf-8",
            )
            config = OptConfig.from_json(path)

        self.assertEqual(config.name, "facebook/opt-test")
        self.assertEqual(config.head_dim, 2)
        self.assertEqual(config.num_key_value_heads, 2)
        self.assertEqual(config.dtype, torch.float16)

    def test_single_file_load_and_layer_cache(self):
        config = make_config()
        state = make_state(config)
        with tempfile.TemporaryDirectory() as directory:
            model_directory = Path(directory)
            checkpoint = model_directory / "pytorch_model.bin"
            checkpoint.touch()
            with patch.object(
                OptCheckpointLoader,
                "_torch_load",
                return_value=state,
            ) as load_mock:
                with OptCheckpointLoader(
                    config,
                    model_directory,
                    cache_layers=True,
                ) as loader:
                    embedding = loader.load_embedding()
                    layer = loader.load_layer(0)
                    self.assertIs(loader.load_layer(0), layer)
                    self.assertIs(loader.load_lm_head(), embedding.token)
                    self.assertEqual(loader.cached_layer_count, 1)
                    self.assertEqual(loader.cached_shard_count, 1)
                    self.assertEqual(
                        tuple(layer.attention.query.shape),
                        (config.hidden_size, config.hidden_size),
                    )
                    self.assertEqual(
                        tuple(layer.mlp.input.shape),
                        (config.ffn_dim, config.hidden_size),
                    )
                    self.assertEqual(loader.checkpoint_files, (checkpoint.resolve(),))
                load_mock.assert_called_once()

    def test_sharded_index_uses_bounded_cache(self):
        config = make_config()
        state = make_state(config)
        base_state = {
            name: tensor for name, tensor in state.items() if ".layers." not in name
        }
        layer_state = {
            name: tensor for name, tensor in state.items() if ".layers." in name
        }
        weight_map = {
            name: "base.bin" for name in base_state
        }
        weight_map.update({name: "layers.bin" for name in layer_state})

        with tempfile.TemporaryDirectory() as directory:
            model_directory = Path(directory)
            (model_directory / "base.bin").touch()
            (model_directory / "layers.bin").touch()
            (model_directory / "pytorch_model.bin.index.json").write_text(
                json.dumps({"weight_map": weight_map}),
                encoding="utf-8",
            )
            states = {"base.bin": base_state, "layers.bin": layer_state}
            with patch.object(
                OptCheckpointLoader,
                "_torch_load",
                side_effect=lambda path: states[path.name],
            ):
                with OptCheckpointLoader(
                    config,
                    model_directory,
                    max_cached_shards=1,
                ) as loader:
                    loader.load_embedding()
                    self.assertEqual(loader.cached_shard_count, 1)
                    loader.load_layer(1)
                    self.assertEqual(loader.cached_shard_count, 1)
                    self.assertEqual(
                        {path.name for path in loader.checkpoint_files},
                        {"base.bin", "layers.bin"},
                    )

    def test_sharded_index_accepts_decoder_prefix(self):
        config = make_config()
        state = {
            name.removeprefix("model."): tensor
            for name, tensor in make_state(config).items()
        }
        weight_map = {name: "model.bin" for name in state}

        with tempfile.TemporaryDirectory() as directory:
            model_directory = Path(directory)
            (model_directory / "model.bin").touch()
            (model_directory / "pytorch_model.bin.index.json").write_text(
                json.dumps({"weight_map": weight_map}),
                encoding="utf-8",
            )
            with patch.object(
                OptCheckpointLoader,
                "_torch_load",
                return_value=state,
            ):
                with OptCheckpointLoader(config, model_directory) as loader:
                    embedding = loader.load_embedding()
                    layer = loader.load_layer(0)
                    final_norm = loader.load_final_norm()
                    self.assertEqual(
                        tuple(embedding.token.shape),
                        (config.vocab_size, config.hidden_size),
                    )
                    self.assertEqual(
                        tuple(layer.attention.query.shape),
                        (config.hidden_size, config.hidden_size),
                    )
                    self.assertEqual(
                        tuple(final_norm.weight.shape),
                        (config.hidden_size,),
                    )


if __name__ == "__main__":
    unittest.main()
