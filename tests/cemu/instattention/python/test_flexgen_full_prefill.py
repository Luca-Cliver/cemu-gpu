#!/usr/bin/env python3

import argparse
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "python"))

from cemu_flexgen import KvCacheLayout, KvCacheStore, KvLayoutConfig
from flexgen_adapter import FlexGenAttentionBackend
from flexgen_runtime import (
    FlexGenLlamaConfig,
    FlexGenPrefillRunner,
    FlexGenWeightLoader,
)


TEST_DEVICE = torch.device("cpu")


class RecordingKvWriter:
    def __init__(self):
        self.layers = []

    def write_prefill(self, layer, keys, values):
        self.layers.append(
            (
                layer,
                keys.detach().cpu().clone(),
                values.detach().cpu().clone(),
            )
        )


class FlexGenFullPrefillTest(unittest.TestCase):
    def setUp(self):
        self.device = TEST_DEVICE
        self.temporary_directory = tempfile.TemporaryDirectory(
            prefix="flexgen-full-prefill-"
        )
        self.weight_directory = Path(self.temporary_directory.name)
        self.config_path = self.weight_directory / "config.json"
        self.config_values = {
            "num_hidden_layers": 2,
            "hidden_size": 8,
            "num_attention_heads": 2,
            "num_key_value_heads": 2,
            "intermediate_size": 12,
            "vocab_size": 24,
            "pad_token_id": 0,
            "rope_theta": 10000.0,
            "rms_norm_eps": 1e-6,
            "torch_dtype": "float32",
        }
        self.config_path.write_text(json.dumps(self.config_values), encoding="utf-8")
        self.config = FlexGenLlamaConfig.from_json(self.config_path)
        self.rng = np.random.default_rng(20260818)
        self._write_model_weights()

    def tearDown(self):
        self.temporary_directory.cleanup()

    def test_complete_prefill_loads_weights_and_writes_every_layer_cache(self):
        kv_writer = RecordingKvWriter()
        runner = FlexGenPrefillRunner(
            config=self.config,
            weight_loader=FlexGenWeightLoader(
                self.config,
                self.weight_directory,
                device=self.device,
            ),
            kv_writer=kv_writer,
        )
        token_ids = torch.tensor(
            [
                [1, 2, 3, 4, 5],
                [6, 7, 8, 9, 10],
            ],
            dtype=torch.long,
        )

        result = runner.run(token_ids, collect_kv_cache=True)

        self.assertEqual(result.hidden_states.shape, (2, 5, 8))
        self.assertEqual(result.logits.shape, (2, 5, 24))
        self.assertEqual(result.next_token_ids.shape, (2, 1))
        self.assertEqual(len(result.kv_cache), 2)
        self.assertEqual([entry[0] for entry in kv_writer.layers], [0, 1])
        for layer, (_, written_keys, written_values) in enumerate(kv_writer.layers):
            collected_keys, collected_values = result.kv_cache[layer]
            self.assertEqual(written_keys.shape, (5, 4, 4))
            self.assertEqual(written_values.shape, (5, 4, 4))
            torch.testing.assert_close(written_keys, collected_keys.detach().cpu())
            torch.testing.assert_close(written_values, collected_values.detach().cpu())
        self.assertTrue(torch.isfinite(result.hidden_states).all())
        self.assertTrue(torch.isfinite(result.logits).all())

        print(
            "\n[flexgen-prefill] complete path: "
            "tokens -> embedding -> 2 layers -> LM head"
        )
        print(
            "  token_ids="
            f"{tuple(token_ids.shape)}, hidden={tuple(result.hidden_states.shape)}, "
            f"logits={tuple(result.logits.shape)}"
        )
        print(
            "  KV per layer="
            f"{tuple(result.kv_cache[0][0].shape)}, "
            f"next_token_ids={result.next_token_ids.reshape(-1).tolist()}"
        )
        print(f"  execution_device={self.device}")

    def test_prefill_can_avoid_retaining_all_layer_caches(self):
        runner = FlexGenPrefillRunner(
            config=self.config,
            weight_loader=FlexGenWeightLoader(
                self.config,
                self.weight_directory,
                device=self.device,
            ),
        )
        token_ids = torch.tensor([[1, 2, 3]], dtype=torch.long)

        result = runner.run(token_ids)

        self.assertIsNone(result.kv_cache)
        self.assertEqual(result.next_token_ids.shape, (1, 1))

    def test_complete_prefill_writes_real_cemu_kv_store(self):
        layout = KvCacheLayout(
            KvLayoutConfig(
                num_layers=self.config.num_hidden_layers,
                max_seq_len=5,
                batch_size=2,
                num_kv_heads=self.config.num_attention_heads,
                head_dim=self.config.head_dim,
                dtype=np.float32,
            )
        )
        k_cache_path = self.weight_directory / "k_cache"
        v_cache_path = self.weight_directory / "v_cache"
        backend = FlexGenAttentionBackend(
            layout=layout,
            k_cache_path=k_cache_path,
            v_cache_path=v_cache_path,
            replace_existing=True,
        )
        runner = FlexGenPrefillRunner(
            config=self.config,
            weight_loader=FlexGenWeightLoader(
                self.config,
                self.weight_directory,
                device=self.device,
            ),
            kv_writer=backend,
        )
        token_ids = torch.tensor(
            [
                [1, 2, 3, 4, 5],
                [6, 7, 8, 9, 10],
            ],
            dtype=torch.long,
        )

        with backend:
            result = runner.run(token_ids, collect_kv_cache=True)
            backend.flush()

        with KvCacheStore(layout, k_cache_path, v_cache_path) as store:
            for layer in range(self.config.num_hidden_layers):
                stored_keys, stored_values = store.read_tokens(layer, 0, 5)
                expected_keys = (
                    result.kv_cache[layer][0]
                    .detach()
                    .cpu()
                    .reshape(5, 2, 2, 4)
                    .numpy()
                )
                expected_values = (
                    result.kv_cache[layer][1]
                    .detach()
                    .cpu()
                    .reshape(5, 2, 2, 4)
                    .numpy()
                )
                np.testing.assert_array_equal(stored_keys, expected_keys)
                np.testing.assert_array_equal(stored_values, expected_values)

    def test_weight_shape_mismatch_is_rejected(self):
        self._write_weight(
            "layers.0.self_attn.q_proj.weight",
            np.zeros((4, 4), dtype=np.float32),
        )
        loader = FlexGenWeightLoader(self.config, self.weight_directory)

        with self.assertRaises(ValueError):
            loader.load_layer(0)

    def _write_model_weights(self):
        hidden_size = self.config.hidden_size
        intermediate_size = self.config.intermediate_size
        self._write_random_weight(
            "embed_tokens.weight",
            (self.config.vocab_size, hidden_size),
        )
        self._write_random_weight(
            "lm_head.weight",
            (self.config.vocab_size, hidden_size),
        )
        self._write_random_weight("norm.weight", (hidden_size,))

        for layer in range(self.config.num_hidden_layers):
            prefix = f"layers.{layer}."
            for projection in ("q_proj", "k_proj", "v_proj", "o_proj"):
                self._write_random_weight(
                    prefix + f"self_attn.{projection}.weight",
                    (hidden_size, hidden_size),
                )
            self._write_random_weight(
                prefix + "input_layernorm.weight",
                (hidden_size,),
            )
            self._write_random_weight(
                prefix + "post_attention_layernorm.weight",
                (hidden_size,),
            )
            self._write_random_weight(
                prefix + "mlp.gate_proj.weight",
                (intermediate_size, hidden_size),
            )
            self._write_random_weight(
                prefix + "mlp.up_proj.weight",
                (intermediate_size, hidden_size),
            )
            self._write_random_weight(
                prefix + "mlp.down_proj.weight",
                (hidden_size, intermediate_size),
            )

    def _write_random_weight(self, filename, shape):
        values = self.rng.normal(scale=0.05, size=shape).astype(np.float32)
        self._write_weight(filename, values)

    def _write_weight(self, filename, values):
        with (self.weight_directory / filename).open("wb") as weight_file:
            np.save(weight_file, values)


def main():
    global TEST_DEVICE

    parser = argparse.ArgumentParser(description="Test complete FlexGen Prefill")
    parser.add_argument("--cuda", action="store_true")
    args, unittest_args = parser.parse_known_args()
    if args.cuda:
        if not torch.cuda.is_available():
            parser.error("--cuda requested, but PyTorch cannot access a CUDA device")
        TEST_DEVICE = torch.device("cuda:0")
        print(f"[flexgen-full-prefill] device={torch.cuda.get_device_name(0)}")
    else:
        print("[flexgen-full-prefill] device=CPU")
    unittest.main(argv=[sys.argv[0], *unittest_args])


if __name__ == "__main__":
    main()
