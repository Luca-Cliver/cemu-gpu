#!/usr/bin/env python3

import argparse
from concurrent.futures import Future
import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F

from opt_runtime import (
    OptCheckpointLoader,
    OptConfig,
    OptDecodeRunner,
    OptGenerationRunner,
    OptOperations,
    OptMultiBatchPrefillRunner,
    OptPrefillRunner,
    OptTorchAttentionBackend,
)
from runtime_common import partition_kv_cache_by_batch


TEST_DEVICE = torch.device("cpu")


class RecordingMicrobatchWriter:
    def __init__(self, count):
        self.backends = tuple(range(count))
        self.writes = {}

    def submit_prefill_microbatch(self, microbatch, layer, keys, values):
        self.writes[(layer, microbatch)] = (
            keys.detach().clone(),
            values.detach().clone(),
        )
        future = Future()
        future.set_result(keys.shape[0])
        return type(
            "WriteRequest",
            (),
            {"layer": layer, "microbatch": microbatch, "future": future},
        )()

    @staticmethod
    def wait_prefill(request):
        return request.future.result()


def make_config() -> OptConfig:
    return OptConfig(
        name="test-opt-runtime",
        num_hidden_layers=2,
        max_position_embeddings=16,
        hidden_size=8,
        num_attention_heads=2,
        ffn_dim=12,
        vocab_size=24,
        word_embed_proj_dim=8,
        dtype=torch.float32,
    )


def make_state(config: OptConfig):
    generator = torch.Generator().manual_seed(20260903)

    def random_tensor(*shape):
        return torch.randn(*shape, generator=generator, dtype=config.dtype) * 0.05

    state = {
        "model.decoder.embed_tokens.weight": random_tensor(
            config.vocab_size,
            config.hidden_size,
        ),
        "model.decoder.embed_positions.weight": random_tensor(
            config.max_position_embeddings + config.position_offset,
            config.hidden_size,
        ),
        "model.decoder.final_layer_norm.weight": torch.ones(
            config.hidden_size,
            dtype=config.dtype,
        ),
        "model.decoder.final_layer_norm.bias": random_tensor(config.hidden_size),
    }
    for layer in range(config.num_hidden_layers):
        prefix = f"model.decoder.layers.{layer}."
        attention_prefix = prefix + "self_attn."
        for projection in ("q_proj", "k_proj", "v_proj", "out_proj"):
            state[attention_prefix + projection + ".weight"] = random_tensor(
                config.hidden_size,
                config.hidden_size,
            )
            state[attention_prefix + projection + ".bias"] = random_tensor(
                config.hidden_size,
            )
        for norm in ("self_attn_layer_norm", "final_layer_norm"):
            state[prefix + norm + ".weight"] = torch.ones(
                config.hidden_size,
                dtype=config.dtype,
            )
            state[prefix + norm + ".bias"] = random_tensor(config.hidden_size)
        state[prefix + "fc1.weight"] = random_tensor(
            config.ffn_dim,
            config.hidden_size,
        )
        state[prefix + "fc1.bias"] = random_tensor(config.ffn_dim)
        state[prefix + "fc2.weight"] = random_tensor(
            config.hidden_size,
            config.ffn_dim,
        )
        state[prefix + "fc2.bias"] = random_tensor(config.hidden_size)
    return state


class OptRuntimeTest(unittest.TestCase):
    def setUp(self):
        self.config = make_config()
        self.temporary_directory = tempfile.TemporaryDirectory(
            prefix="opt-runtime-"
        )
        self.model_directory = Path(self.temporary_directory.name)
        torch.save(
            make_state(self.config),
            self.model_directory / "pytorch_model.bin",
        )

    def tearDown(self):
        self.temporary_directory.cleanup()

    def test_prefill_and_decode_match_extended_prefill(self):
        loader = OptCheckpointLoader(
            self.config,
            self.model_directory,
            device=TEST_DEVICE,
        )
        prompt = torch.tensor(
            [[2, 4, 5], [2, 6, 7]],
            dtype=torch.long,
        )
        prefill = OptPrefillRunner(self.config, loader).run(
            prompt,
            collect_kv_cache=True,
        )
        extended = OptPrefillRunner(self.config, loader).run(
            torch.cat((prompt, prefill.next_token_ids.cpu()), dim=1),
            collect_kv_cache=True,
        )
        backend = OptTorchAttentionBackend(self.config, prefill.kv_cache)
        decode = OptDecodeRunner(
            self.config,
            loader,
            backend,
        ).run(
            prefill.next_token_ids,
            token_position=prompt.shape[1],
            collect_layer_outputs=True,
        )

        torch.testing.assert_close(
            decode.hidden_states,
            extended.hidden_states[:, -1:, :],
            rtol=1e-5,
            atol=1e-5,
        )
        torch.testing.assert_close(
            decode.logits,
            extended.logits[:, -1:, :],
            rtol=1e-5,
            atol=1e-5,
        )
        torch.testing.assert_close(
            decode.next_token_ids,
            extended.next_token_ids,
        )
        for layer, layer_output in enumerate(decode.layer_outputs):
            cached_keys, cached_values = backend.layer_cache(layer)
            torch.testing.assert_close(cached_keys[-1:], layer_output.key)
            torch.testing.assert_close(cached_values[-1:], layer_output.value)

        print(
            "\n[opt-runtime] shared runner: Prefill next token -> "
            "one-token Decode -> LM head"
        )
        print(
            f"  prompt={tuple(prompt.shape)}, layers={self.config.num_hidden_layers}, "
            f"heads={self.config.num_attention_heads}"
        )
        print(
            f"  K/V={tuple(prefill.kv_cache[0][0].shape)}, "
            f"decode_logits={tuple(decode.logits.shape)}, "
            f"next_tokens={decode.next_token_ids.reshape(-1).tolist()}"
        )

    def test_multi_batch_prefill_matches_full_batch_prefill(self):
        loader = OptCheckpointLoader(
            self.config,
            self.model_directory,
            device=TEST_DEVICE,
        )
        prompt = torch.tensor(
            [[2, 4, 5], [2, 6, 7], [2, 8, 9], [2, 10, 11]],
            dtype=torch.long,
        )
        expected = OptPrefillRunner(self.config, loader).run(
            prompt,
            collect_kv_cache=True,
        )
        writer = RecordingMicrobatchWriter(count=2)
        actual = OptMultiBatchPrefillRunner(
            self.config,
            loader,
            writer,
            gpu_batch_size=2,
        ).run(
            prompt,
            collect_kv_cache=True,
        )

        torch.testing.assert_close(actual.hidden_states, expected.hidden_states)
        torch.testing.assert_close(actual.logits, expected.logits)
        torch.testing.assert_close(actual.next_token_ids, expected.next_token_ids)
        for layer, ((actual_k, actual_v), (expected_k, expected_v)) in enumerate(
            zip(actual.kv_cache, expected.kv_cache)
        ):
            torch.testing.assert_close(actual_k, expected_k)
            torch.testing.assert_close(actual_v, expected_v)
            torch.testing.assert_close(
                torch.cat(
                    [writer.writes[(layer, batch)][0] for batch in range(2)],
                    dim=1,
                ),
                expected_k,
            )

    def test_decode_query_is_scaled_before_attention_backend(self):
        loader = OptCheckpointLoader(
            self.config,
            self.model_directory,
            device=TEST_DEVICE,
        )
        weights = loader.load_layer(0)
        inputs = torch.arange(
            2 * self.config.hidden_size,
            dtype=self.config.dtype,
            device=TEST_DEVICE,
        ).reshape(2, 1, self.config.hidden_size)
        projection = OptOperations(self.config).prepare_decode_attention(
            inputs,
            weights,
            token_position=3,
        )
        normalized = F.layer_norm(
            inputs,
            (self.config.hidden_size,),
            weights.attention.input_norm.weight,
            weights.attention.input_norm.bias,
            self.config.layer_norm_epsilon,
        )
        expected = F.linear(
            normalized,
            weights.attention.query,
            weights.attention.query_bias,
        ) * (self.config.head_dim ** -0.5)
        expected = expected.view(
            2,
            self.config.num_attention_heads,
            self.config.head_dim,
        )

        torch.testing.assert_close(projection.query, expected)

        cache_shape = (
            1,
            2 * self.config.num_key_value_heads,
            self.config.head_dim,
        )
        cache = tuple(
            (
                torch.zeros(cache_shape, dtype=self.config.dtype, device=TEST_DEVICE),
                torch.zeros(cache_shape, dtype=self.config.dtype, device=TEST_DEVICE),
            )
            for _ in range(self.config.num_hidden_layers)
        )
        backend = OptTorchAttentionBackend(self.config, cache)
        self.assertEqual(backend.attention_scale, 1.0)

    def test_attention_backend_can_accumulate_float16_inputs_in_float32(self):
        generator = torch.Generator().manual_seed(20260907)
        cache_shape = (
            3,
            self.config.num_key_value_heads,
            self.config.head_dim,
        )
        keys = torch.randn(cache_shape, generator=generator, dtype=torch.float16)
        values = torch.randn(cache_shape, generator=generator, dtype=torch.float16)
        query = torch.randn(
            1,
            self.config.num_attention_heads,
            self.config.head_dim,
            generator=generator,
            dtype=torch.float16,
        )
        cache = tuple(
            (keys.clone(), values.clone())
            for _ in range(self.config.num_hidden_layers)
        )
        backend = OptTorchAttentionBackend(
            self.config,
            cache,
            accumulation_dtype=torch.float32,
        )

        actual = backend.decode(0, query, valid_tokens=cache_shape[0])
        attention_keys = keys.float().permute(1, 0, 2).unsqueeze(0)
        attention_values = values.float().permute(1, 0, 2).unsqueeze(0)
        scores = torch.einsum("bhd,bhtd->bht", query.float(), attention_keys)
        expected = torch.einsum(
            "bht,bhtd->bhd",
            torch.softmax(scores, dim=-1),
            attention_values,
        ).half()

        self.assertEqual(actual.dtype, torch.float16)
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)

    def test_partition_kv_cache_by_batch_preserves_batch_order(self):
        token_count = 3
        batch_size = 4
        cache_shape = (
            token_count,
            batch_size * self.config.num_key_value_heads,
            self.config.head_dim,
        )
        keys = torch.arange(
            torch.tensor(cache_shape).prod().item(),
            dtype=self.config.dtype,
        ).reshape(cache_shape)
        values = keys + 1000
        cache = tuple(
            (keys.clone(), values.clone())
            for _ in range(self.config.num_hidden_layers)
        )

        partitions = partition_kv_cache_by_batch(
            self.config,
            cache,
            microbatch_size=1,
        )

        self.assertEqual(len(partitions), batch_size)
        for layer in range(self.config.num_hidden_layers):
            torch.testing.assert_close(
                torch.cat([partition[layer][0] for partition in partitions], dim=1),
                cache[layer][0],
            )
            torch.testing.assert_close(
                torch.cat([partition[layer][1] for partition in partitions], dim=1),
                cache[layer][1],
            )

    def test_generation_can_discard_per_step_outputs(self):
        loader = OptCheckpointLoader(
            self.config,
            self.model_directory,
            device=TEST_DEVICE,
        )
        prompt = torch.tensor([[2, 4, 5]], dtype=torch.long)
        prefill = OptPrefillRunner(self.config, loader).run(
            prompt,
            collect_kv_cache=True,
            last_token_only=True,
        )
        generation = OptGenerationRunner(
            OptDecodeRunner(
                self.config,
                loader,
                OptTorchAttentionBackend(self.config, prefill.kv_cache),
            )
        ).run(
            prefill.next_token_ids,
            start_position=prompt.shape[1],
            decode_steps=3,
            collect_layer_outputs=False,
            collect_steps=False,
        )

        self.assertEqual(generation.token_ids.shape, (1, 4))
        self.assertEqual(generation.steps, ())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", action="store_true")
    arguments, unittest_arguments = parser.parse_known_args()
    global TEST_DEVICE
    if arguments.cuda:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        TEST_DEVICE = torch.device("cuda:0")
        print(f"[opt-runtime] device={torch.cuda.get_device_name(TEST_DEVICE)}")
    unittest.main(argv=[__file__, *unittest_arguments])


if __name__ == "__main__":
    main()
