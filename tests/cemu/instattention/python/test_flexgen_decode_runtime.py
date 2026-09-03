#!/usr/bin/env python3

import argparse
import json
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "python"))

from flexgen_runtime import (
    FlexGenDecodeRunner,
    FlexGenGenerationRunner,
    FlexGenLlamaConfig,
    FlexGenMultiBatchDecodeRunner,
    FlexGenPrefillRunner,
    FlexGenTorchAttentionBackend,
    FlexGenWeightLoader,
)


TEST_DEVICE = torch.device("cpu")


class PipelineRequest:
    def __init__(self, layer, output):
        self.layer = layer
        self.output = output


class OrderedPipelineBackend:
    supports_pipelined_decode = True

    def __init__(self):
        self.events = []
        self.first_submit = threading.Event()

    def prefetch_decode(self, layer, history_tokens):
        self.events.append(("prefetch", layer))
        return SimpleNamespace(layer=layer, history_tokens=history_tokens)

    def submit_prefetched_decode(
        self,
        prefetch,
        layer,
        token,
        query,
        key,
        value,
        valid_tokens,
    ):
        self.events.append(("submit", layer))
        if layer == 0:
            self.first_submit.set()
        output = np.zeros(tuple(query.shape), dtype=np.float32)
        return PipelineRequest(layer, output)

    def wait_decode(self, request):
        self.events.append(("wait", request.layer))
        return request.output

    def append_decode(self, layer, token, key, value):
        raise AssertionError("the pipelined path must not use synchronous append")

    def decode(self, layer, query, valid_tokens):
        raise AssertionError("the pipelined path must not use synchronous decode")


class OrderedMicrobatchPipelineBackend:
    supports_pipelined_decode = True

    def __init__(self, microbatch, events):
        self.microbatch = microbatch
        self.events = events

    def prefetch_decode(self, layer, history_tokens):
        self.events.append(("prefetch", layer, self.microbatch))
        return SimpleNamespace(
            layer=layer,
            history_tokens=history_tokens,
            microbatch=self.microbatch,
        )

    def submit_prefetched_decode(
        self,
        prefetch,
        layer,
        token,
        query,
        key,
        value,
        valid_tokens,
    ):
        self.events.append(("submit", layer, self.microbatch))
        return PipelineRequest(
            layer,
            np.zeros(tuple(query.shape), dtype=np.float32),
        )

    def wait_decode(self, request):
        self.events.append(("wait", request.layer, self.microbatch))
        return request.output

    def append_decode(self, layer, token, key, value):
        raise AssertionError("the pipelined path must not use synchronous append")

    def decode(self, layer, query, valid_tokens):
        raise AssertionError("the pipelined path must not use synchronous decode")


class BlockingNextLayerWeightLoader(FlexGenWeightLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.next_layer_started = threading.Event()
        self.release_next_layer = threading.Event()

    def load_layer(self, layer):
        if layer == 1:
            self.next_layer_started.set()
            if not self.release_next_layer.wait(timeout=5):
                raise TimeoutError("test did not release next-layer weight loading")
        return super().load_layer(layer)


class FlexGenDecodeRuntimeTest(unittest.TestCase):
    def setUp(self):
        self.device = TEST_DEVICE
        self.temporary_directory = tempfile.TemporaryDirectory(
            prefix="flexgen-decode-"
        )
        self.weight_directory = Path(self.temporary_directory.name)
        self.config = self._write_test_model()

    def tearDown(self):
        self.temporary_directory.cleanup()

    def test_prefill_token_drives_one_real_decode_step(self):
        loader = FlexGenWeightLoader(
            self.config,
            self.weight_directory,
            device=self.device,
        )
        prompt = torch.tensor(
            [[1, 2, 3], [4, 5, 6]],
            dtype=torch.long,
        )
        prefill = FlexGenPrefillRunner(
            self.config,
            loader,
        ).run(prompt, collect_kv_cache=True)
        extended_prefill = FlexGenPrefillRunner(
            self.config,
            loader,
        ).run(
            torch.cat((prompt, prefill.next_token_ids.cpu()), dim=1),
            collect_kv_cache=True,
        )
        backend = FlexGenTorchAttentionBackend(
            self.config,
            prefill.kv_cache,
        )
        messages = []
        decode = FlexGenDecodeRunner(
            self.config,
            loader,
            backend,
            logger=messages.append,
        ).run(
            prefill.next_token_ids,
            token_position=prompt.shape[1],
            collect_layer_outputs=True,
        )

        self.assertEqual(decode.hidden_states.shape, (2, 1, 8))
        self.assertEqual(decode.logits.shape, (2, 1, 24))
        self.assertEqual(decode.next_token_ids.shape, (2, 1))
        self.assertEqual(len(decode.layer_outputs), 2)
        for layer, layer_output in enumerate(decode.layer_outputs):
            self.assertEqual(layer_output.query.shape, (2, 2, 4))
            self.assertEqual(layer_output.key.shape, (1, 4, 4))
            self.assertEqual(layer_output.value.shape, (1, 4, 4))
            self.assertEqual(layer_output.attention_output.shape, (2, 2, 4))
            cached_keys, cached_values = backend.layer_cache(layer)
            self.assertEqual(cached_keys.shape, (4, 4, 4))
            self.assertEqual(cached_values.shape, (4, 4, 4))
            torch.testing.assert_close(cached_keys[-1:], layer_output.key)
            torch.testing.assert_close(cached_values[-1:], layer_output.value)
            torch.testing.assert_close(
                layer_output.key,
                extended_prefill.kv_cache[layer][0][-1:],
                rtol=1e-5,
                atol=1e-5,
            )
            torch.testing.assert_close(
                layer_output.value,
                extended_prefill.kv_cache[layer][1][-1:],
                rtol=1e-5,
                atol=1e-5,
            )
        torch.testing.assert_close(
            decode.hidden_states,
            extended_prefill.hidden_states[:, -1:, :],
            rtol=1e-5,
            atol=1e-5,
        )
        torch.testing.assert_close(
            decode.logits,
            extended_prefill.logits[:, -1:, :],
            rtol=1e-5,
            atol=1e-5,
        )
        torch.testing.assert_close(
            decode.next_token_ids,
            extended_prefill.next_token_ids,
        )

        self.assertTrue(torch.isfinite(decode.hidden_states).all())
        self.assertTrue(torch.isfinite(decode.logits).all())
        self.assertTrue(any("KV appended at token=3" in line for line in messages))

        print(
            "\n[flexgen-decode] Prefill next token -> one-token Decode -> LM head"
        )
        print(
            f"  input_tokens={prefill.next_token_ids.reshape(-1).tolist()}, "
            f"position={prompt.shape[1]}, layers={self.config.num_hidden_layers}"
        )
        print(
            f"  Q={tuple(decode.layer_outputs[0].query.shape)}, "
            f"appended_K/V={tuple(decode.layer_outputs[0].key.shape)}, "
            f"logits={tuple(decode.logits.shape)}"
        )
        print(
            f"  next_token_ids={decode.next_token_ids.reshape(-1).tolist()}, "
            f"execution_device={self.device}"
        )

    def test_decode_rejects_more_than_one_input_token(self):
        loader = FlexGenWeightLoader(
            self.config,
            self.weight_directory,
            device=self.device,
        )
        backend = FlexGenTorchAttentionBackend(
            self.config,
            tuple(
                (
                    torch.zeros((1, 4, 4), device=self.device),
                    torch.zeros((1, 4, 4), device=self.device),
                )
                for _ in range(self.config.num_hidden_layers)
            ),
        )
        runner = FlexGenDecodeRunner(self.config, loader, backend)

        with self.assertRaises(ValueError):
            runner.run(torch.tensor([[1, 2]]), token_position=1)

    def test_pipelined_decode_prefetches_next_layer_before_waiting(self):
        loader = FlexGenWeightLoader(
            self.config,
            self.weight_directory,
            device=self.device,
        )
        backend = OrderedPipelineBackend()

        result = FlexGenDecodeRunner(
            self.config,
            loader,
            backend,
        ).run(
            torch.tensor([[1], [2]], dtype=torch.long),
            token_position=3,
        )

        self.assertEqual(result.next_token_ids.shape, (2, 1))
        self.assertEqual(
            backend.events,
            [
                ("prefetch", 0),
                ("prefetch", 1),
                ("submit", 0),
                ("wait", 0),
                ("submit", 1),
                ("wait", 1),
            ],
        )

    def test_current_qkv_advances_while_next_layer_weights_load(self):
        loader = BlockingNextLayerWeightLoader(
            self.config,
            self.weight_directory,
            device=self.device,
        )
        backend = OrderedPipelineBackend()
        result = []
        errors = []

        def run_decode():
            try:
                result.append(
                    FlexGenDecodeRunner(
                        self.config,
                        loader,
                        backend,
                    ).run(
                        torch.tensor([[1], [2]], dtype=torch.long),
                        token_position=3,
                    )
                )
            except Exception as error:
                errors.append(error)

        thread = threading.Thread(target=run_decode, name="decode-overlap-test")
        thread.start()
        try:
            self.assertTrue(loader.next_layer_started.wait(timeout=5))
            self.assertTrue(backend.first_submit.wait(timeout=5))
            self.assertTrue(thread.is_alive())
            self.assertIn(("submit", 0), backend.events)
        finally:
            loader.release_next_layer.set()
            thread.join(timeout=5)

        self.assertFalse(thread.is_alive())
        self.assertEqual(errors, [])
        self.assertEqual(len(result), 1)

    def test_multi_batch_pipeline_advances_microbatch_before_waiting(self):
        loader = FlexGenWeightLoader(
            self.config,
            self.weight_directory,
            device=self.device,
        )
        events = []
        backends = tuple(
            OrderedMicrobatchPipelineBackend(microbatch, events)
            for microbatch in range(2)
        )

        result = FlexGenMultiBatchDecodeRunner(
            self.config,
            loader,
            backends,
            gpu_batch_size=1,
        ).run(
            torch.tensor([[1], [2]], dtype=torch.long),
            token_position=3,
            collect_layer_outputs=True,
        )

        self.assertEqual(result.next_token_ids.shape, (2, 1))
        self.assertEqual(len(result.layer_outputs), 2)
        self.assertEqual(
            events,
            [
                ("prefetch", 0, 0),
                ("prefetch", 0, 1),
                ("submit", 0, 0),
                ("prefetch", 1, 0),
                ("submit", 0, 1),
                ("wait", 0, 0),
                ("wait", 0, 1),
                ("prefetch", 1, 1),
                ("submit", 1, 0),
                ("submit", 1, 1),
                ("wait", 1, 0),
                ("wait", 1, 1),
            ],
        )

    def test_multi_batch_decode_matches_full_batch_reference(self):
        loader = FlexGenWeightLoader(
            self.config,
            self.weight_directory,
            device=self.device,
        )
        prompt = torch.tensor(
            [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]],
            dtype=torch.long,
        )
        prefill = FlexGenPrefillRunner(
            self.config,
            loader,
        ).run(prompt, collect_kv_cache=True)
        full_backend = FlexGenTorchAttentionBackend(
            self.config,
            prefill.kv_cache,
        )
        full_result = FlexGenDecodeRunner(
            self.config,
            loader,
            full_backend,
        ).run(
            prefill.next_token_ids,
            token_position=prompt.shape[1],
            collect_layer_outputs=True,
        )

        gpu_batch_size = 2
        microbatch_caches = [[], []]
        for keys, values in prefill.kv_cache:
            token_count = keys.shape[0]
            keys = keys.reshape(
                token_count,
                prompt.shape[0],
                self.config.num_key_value_heads,
                self.config.head_dim,
            )
            values = values.reshape_as(keys)
            for microbatch in range(2):
                batch_start = microbatch * gpu_batch_size
                batch_end = batch_start + gpu_batch_size
                microbatch_caches[microbatch].append(
                    (
                        keys[:, batch_start:batch_end].reshape(
                            token_count,
                            gpu_batch_size * self.config.num_key_value_heads,
                            self.config.head_dim,
                        ),
                        values[:, batch_start:batch_end].reshape(
                            token_count,
                            gpu_batch_size * self.config.num_key_value_heads,
                            self.config.head_dim,
                        ),
                    )
                )
        microbatch_backends = tuple(
            FlexGenTorchAttentionBackend(self.config, cache)
            for cache in microbatch_caches
        )
        multi_result = FlexGenMultiBatchDecodeRunner(
            self.config,
            loader,
            microbatch_backends,
            gpu_batch_size=gpu_batch_size,
        ).run(
            prefill.next_token_ids,
            token_position=prompt.shape[1],
            collect_layer_outputs=True,
        )

        torch.testing.assert_close(multi_result.hidden_states, full_result.hidden_states)
        torch.testing.assert_close(multi_result.logits, full_result.logits)
        torch.testing.assert_close(
            multi_result.next_token_ids,
            full_result.next_token_ids,
        )
        for multi_layer, full_layer in zip(
            multi_result.layer_outputs,
            full_result.layer_outputs,
        ):
            torch.testing.assert_close(multi_layer.query, full_layer.query)
            torch.testing.assert_close(multi_layer.key, full_layer.key)
            torch.testing.assert_close(multi_layer.value, full_layer.value)
            torch.testing.assert_close(
                multi_layer.attention_output,
                full_layer.attention_output,
            )

    def test_gqa_prefill_and_decode_keep_compact_kv_cache(self):
        config = self._write_test_model(
            hidden_size=16,
            num_attention_heads=4,
            num_key_value_heads=2,
            intermediate_size=24,
        )
        loader = FlexGenWeightLoader(
            config,
            self.weight_directory,
            device=self.device,
        )
        prompt = torch.tensor(
            [[1, 2, 3], [4, 5, 6]],
            dtype=torch.long,
        )
        prefill_runner = FlexGenPrefillRunner(config, loader)
        prefill = prefill_runner.run(prompt, collect_kv_cache=True)
        extended_prefill = prefill_runner.run(
            torch.cat((prompt, prefill.next_token_ids.cpu()), dim=1),
            collect_kv_cache=True,
        )
        backend = FlexGenTorchAttentionBackend(config, prefill.kv_cache)
        decode = FlexGenDecodeRunner(config, loader, backend).run(
            prefill.next_token_ids,
            token_position=prompt.shape[1],
            collect_layer_outputs=True,
        )

        self.assertEqual(config.num_key_value_groups, 2)
        self.assertEqual(decode.hidden_states.shape, (2, 1, 16))
        for layer, layer_output in enumerate(decode.layer_outputs):
            self.assertEqual(layer_output.query.shape, (2, 4, 4))
            self.assertEqual(layer_output.key.shape, (1, 4, 4))
            self.assertEqual(layer_output.value.shape, (1, 4, 4))
            self.assertEqual(layer_output.attention_output.shape, (2, 4, 4))
            cached_keys, cached_values = backend.layer_cache(layer)
            self.assertEqual(cached_keys.shape, (4, 4, 4))
            self.assertEqual(cached_values.shape, (4, 4, 4))
            torch.testing.assert_close(
                layer_output.key,
                extended_prefill.kv_cache[layer][0][-1:],
                rtol=1e-5,
                atol=1e-5,
            )
            torch.testing.assert_close(
                layer_output.value,
                extended_prefill.kv_cache[layer][1][-1:],
                rtol=1e-5,
                atol=1e-5,
            )
        torch.testing.assert_close(
            decode.hidden_states,
            extended_prefill.hidden_states[:, -1:, :],
            rtol=1e-5,
            atol=1e-5,
        )
        torch.testing.assert_close(
            decode.logits,
            extended_prefill.logits[:, -1:, :],
            rtol=1e-5,
            atol=1e-5,
        )
        torch.testing.assert_close(
            decode.next_token_ids,
            extended_prefill.next_token_ids,
        )

        print("\n[flexgen-gqa] compact KV Prefill -> one-token Decode")
        print(
            f"  query_heads={config.num_attention_heads}, "
            f"kv_heads={config.num_key_value_heads}, "
            f"groups={config.num_key_value_groups}"
        )
        print(
            f"  Q={tuple(decode.layer_outputs[0].query.shape)}, "
            f"K/V={tuple(decode.layer_outputs[0].key.shape)}, "
            f"attention={tuple(decode.layer_outputs[0].attention_output.shape)}"
        )

    def test_config_rejects_non_divisible_kv_heads(self):
        with self.assertRaises(ValueError):
            FlexGenLlamaConfig(
                num_hidden_layers=1,
                hidden_size=16,
                num_attention_heads=4,
                num_key_value_heads=3,
                intermediate_size=24,
                vocab_size=24,
                dtype=torch.float32,
            )

    def test_multiple_decode_steps_feed_each_output_back_as_input(self):
        loader = FlexGenWeightLoader(
            self.config,
            self.weight_directory,
            device=self.device,
        )
        prompt = torch.tensor(
            [[1, 2, 3], [4, 5, 6]],
            dtype=torch.long,
        )
        prefill_runner = FlexGenPrefillRunner(self.config, loader)
        prefill = prefill_runner.run(prompt, collect_kv_cache=True)
        backend = FlexGenTorchAttentionBackend(self.config, prefill.kv_cache)
        messages = []
        generation = FlexGenGenerationRunner(
            FlexGenDecodeRunner(self.config, loader, backend),
            logger=messages.append,
        ).run(
            prefill.next_token_ids,
            start_position=prompt.shape[1],
            decode_steps=3,
            collect_layer_outputs=True,
        )

        self.assertEqual(generation.token_ids.shape, (2, 4))
        self.assertEqual(len(generation.steps), 3)
        torch.testing.assert_close(
            generation.token_ids[:, :1],
            prefill.next_token_ids,
        )
        torch.testing.assert_close(
            generation.next_token_ids,
            generation.steps[-1].decode_result.next_token_ids,
        )

        for step, generation_step in enumerate(generation.steps):
            self.assertEqual(generation_step.step, step)
            self.assertEqual(
                generation_step.token_position,
                prompt.shape[1] + step,
            )
            torch.testing.assert_close(
                generation_step.input_token_ids,
                generation.token_ids[:, step : step + 1],
            )
            extended_prompt = torch.cat(
                (
                    prompt,
                    generation.token_ids[:, : step + 1].detach().cpu(),
                ),
                dim=1,
            )
            extended_prefill = prefill_runner.run(
                extended_prompt,
                collect_kv_cache=True,
            )
            decode_result = generation_step.decode_result
            torch.testing.assert_close(
                decode_result.hidden_states,
                extended_prefill.hidden_states[:, -1:, :],
                rtol=1e-5,
                atol=1e-5,
            )
            torch.testing.assert_close(
                decode_result.logits,
                extended_prefill.logits[:, -1:, :],
                rtol=1e-5,
                atol=1e-5,
            )
            torch.testing.assert_close(
                decode_result.next_token_ids,
                extended_prefill.next_token_ids,
            )
            for layer, layer_output in enumerate(decode_result.layer_outputs):
                torch.testing.assert_close(
                    layer_output.key,
                    extended_prefill.kv_cache[layer][0][-1:],
                    rtol=1e-5,
                    atol=1e-5,
                )
                torch.testing.assert_close(
                    layer_output.value,
                    extended_prefill.kv_cache[layer][1][-1:],
                    rtol=1e-5,
                    atol=1e-5,
                )

        expected_cache_tokens = prompt.shape[1] + len(generation.steps)
        for layer in range(self.config.num_hidden_layers):
            cached_keys, cached_values = backend.layer_cache(layer)
            self.assertEqual(cached_keys.shape[0], expected_cache_tokens)
            self.assertEqual(cached_values.shape[0], expected_cache_tokens)
        self.assertTrue(any("step=2, position=5" in line for line in messages))

        print("\n[flexgen-generation] autoregressive multi-step Decode")
        print(
            f"  prompt_tokens={prompt.shape[1]}, decode_steps=3, "
            f"token_sequence={generation.token_ids.detach().cpu().tolist()}"
        )
        print(
            f"  cached_tokens_per_layer={expected_cache_tokens}, "
            f"execution_device={self.device}"
        )

    def test_generation_rejects_zero_decode_steps(self):
        loader = FlexGenWeightLoader(
            self.config,
            self.weight_directory,
            device=self.device,
        )
        backend = FlexGenTorchAttentionBackend(
            self.config,
            tuple(
                (
                    torch.zeros((1, 4, 4), device=self.device),
                    torch.zeros((1, 4, 4), device=self.device),
                )
                for _ in range(self.config.num_hidden_layers)
            ),
        )
        runner = FlexGenGenerationRunner(
            FlexGenDecodeRunner(self.config, loader, backend)
        )

        with self.assertRaises(ValueError):
            runner.run(
                torch.tensor([[1]], dtype=torch.long),
                start_position=1,
                decode_steps=0,
            )

    def _write_test_model(
        self,
        hidden_size=8,
        num_attention_heads=2,
        num_key_value_heads=2,
        intermediate_size=12,
    ):
        config_values = {
            "num_hidden_layers": 2,
            "hidden_size": hidden_size,
            "num_attention_heads": num_attention_heads,
            "num_key_value_heads": num_key_value_heads,
            "intermediate_size": intermediate_size,
            "vocab_size": 24,
            "pad_token_id": 0,
            "rope_theta": 10000.0,
            "rms_norm_eps": 1e-6,
            "torch_dtype": "float32",
        }
        config_path = self.weight_directory / "config.json"
        config_path.write_text(json.dumps(config_values), encoding="utf-8")
        config = FlexGenLlamaConfig.from_json(config_path)
        rng = np.random.default_rng(20260819)

        def write_weight(filename, shape):
            values = rng.normal(scale=0.05, size=shape).astype(np.float32)
            with (self.weight_directory / filename).open("wb") as weight_file:
                np.save(weight_file, values)

        write_weight("embed_tokens.weight", (config.vocab_size, config.hidden_size))
        write_weight("norm.weight", (config.hidden_size,))
        for layer in range(config.num_hidden_layers):
            prefix = f"layers.{layer}."
            write_weight(
                prefix + "self_attn.q_proj.weight",
                (config.hidden_size, config.hidden_size),
            )
            kv_hidden_size = config.num_key_value_heads * config.head_dim
            for projection in ("k_proj", "v_proj"):
                write_weight(
                    prefix + f"self_attn.{projection}.weight",
                    (kv_hidden_size, config.hidden_size),
                )
            write_weight(
                prefix + "self_attn.o_proj.weight",
                (config.hidden_size, config.hidden_size),
            )
            write_weight(prefix + "input_layernorm.weight", (config.hidden_size,))
            write_weight(
                prefix + "post_attention_layernorm.weight",
                (config.hidden_size,),
            )
            write_weight(
                prefix + "mlp.gate_proj.weight",
                (config.intermediate_size, config.hidden_size),
            )
            write_weight(
                prefix + "mlp.up_proj.weight",
                (config.intermediate_size, config.hidden_size),
            )
            write_weight(
                prefix + "mlp.down_proj.weight",
                (config.hidden_size, config.intermediate_size),
            )
        return config


def main():
    global TEST_DEVICE

    parser = argparse.ArgumentParser(
        description="Test one-token and multi-step FlexGen Decode"
    )
    parser.add_argument("--cuda", action="store_true")
    args, unittest_args = parser.parse_known_args()
    if args.cuda:
        if not torch.cuda.is_available():
            parser.error("--cuda requested, but PyTorch cannot access a CUDA device")
        TEST_DEVICE = torch.device("cuda:0")
        print(f"[flexgen-decode] device={torch.cuda.get_device_name(0)}")
    else:
        print("[flexgen-decode] device=CPU")
    unittest.main(argv=[sys.argv[0], *unittest_args])


if __name__ == "__main__":
    main()
