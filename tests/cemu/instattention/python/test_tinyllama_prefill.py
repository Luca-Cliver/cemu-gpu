#!/usr/bin/env python3

import argparse
import time
from pathlib import Path

import torch

from flexgen_runtime import (
    FlexGenHfCheckpointLoader,
    FlexGenLlamaConfig,
    FlexGenPrefillRunner,
)


def positive_integer(value):
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--batch-size", type=positive_integer, default=1)
    parser.add_argument("--prompt-length", type=positive_integer, default=8)
    return parser.parse_args()


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def make_token_ids(config, batch_size, prompt_length):
    if config.vocab_size <= 3:
        raise ValueError("TinyLlama vocabulary must contain more than three tokens")
    token_ids = torch.arange(
        batch_size * prompt_length,
        dtype=torch.long,
    ).reshape(batch_size, prompt_length)
    token_ids = token_ids.remainder(config.vocab_size - 3).add(3)
    token_ids[:, 0] = 1
    return token_ids


def validate_result(result, config, batch_size, prompt_length):
    expected_hidden_shape = (batch_size, prompt_length, config.hidden_size)
    expected_logits_shape = (batch_size, prompt_length, config.vocab_size)
    expected_token_shape = (batch_size, 1)
    expected_cache_shape = (
        prompt_length,
        batch_size * config.num_key_value_heads,
        config.head_dim,
    )

    if tuple(result.hidden_states.shape) != expected_hidden_shape:
        raise AssertionError(
            f"hidden states have shape {tuple(result.hidden_states.shape)}, "
            f"expected {expected_hidden_shape}"
        )
    if tuple(result.logits.shape) != expected_logits_shape:
        raise AssertionError(
            f"logits have shape {tuple(result.logits.shape)}, "
            f"expected {expected_logits_shape}"
        )
    if tuple(result.next_token_ids.shape) != expected_token_shape:
        raise AssertionError(
            f"next token IDs have shape {tuple(result.next_token_ids.shape)}, "
            f"expected {expected_token_shape}"
        )
    if result.kv_cache is None:
        raise AssertionError("Prefill did not return a KV cache")
    if len(result.kv_cache) != config.num_hidden_layers:
        raise AssertionError(
            f"KV cache contains {len(result.kv_cache)} layers, "
            f"expected {config.num_hidden_layers}"
        )

    if not torch.isfinite(result.hidden_states).all().item():
        raise AssertionError("hidden states contain NaN or Inf")
    if not torch.isfinite(result.logits).all().item():
        raise AssertionError("logits contain NaN or Inf")
    if result.next_token_ids.min().item() < 0:
        raise AssertionError("next token ID is negative")
    if result.next_token_ids.max().item() >= config.vocab_size:
        raise AssertionError("next token ID exceeds the vocabulary")

    for layer, (keys, values) in enumerate(result.kv_cache):
        if tuple(keys.shape) != expected_cache_shape:
            raise AssertionError(
                f"layer {layer} K has shape {tuple(keys.shape)}, "
                f"expected {expected_cache_shape}"
            )
        if tuple(values.shape) != expected_cache_shape:
            raise AssertionError(
                f"layer {layer} V has shape {tuple(values.shape)}, "
                f"expected {expected_cache_shape}"
            )
        if not torch.isfinite(keys).all().item():
            raise AssertionError(f"layer {layer} K contains NaN or Inf")
        if not torch.isfinite(values).all().item():
            raise AssertionError(f"layer {layer} V contains NaN or Inf")


def main():
    args = parse_args()
    model_directory = args.model_dir.resolve()
    config = FlexGenLlamaConfig.from_json(model_directory / "config.json")
    device = torch.device("cuda:0" if args.cuda else "cpu")
    if args.cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")

    token_ids = make_token_ids(config, args.batch_size, args.prompt_length)
    print(
        "[tinyllama-prefill] "
        f"model={model_directory}, device={device}, dtype={config.dtype}"
    )
    print(
        "[tinyllama-prefill] "
        f"batch={args.batch_size}, prompt_length={args.prompt_length}, "
        f"token_ids={token_ids.tolist()}"
    )
    print(
        "[tinyllama-prefill] "
        f"layers={config.num_hidden_layers}, hidden={config.hidden_size}, "
        f"q_heads={config.num_attention_heads}, kv_heads={config.num_key_value_heads}, "
        f"head_dim={config.head_dim}, vocab={config.vocab_size}"
    )

    with torch.inference_mode():
        with FlexGenHfCheckpointLoader(
            config,
            model_directory,
            device=device,
            cache_layers=True,
        ) as loader:
            print("[tinyllama-prefill] step 1/3: preload checkpoint weights")
            load_start = time.perf_counter()
            loader.load_embedding()
            loader.load_final_norm()
            loader.load_lm_head()
            for layer in range(config.num_hidden_layers):
                print(
                    f"[tinyllama-prefill] load layer={layer + 1}/"
                    f"{config.num_hidden_layers}",
                    flush=True,
                )
                loader.load_layer(layer)
            synchronize(device)
            load_elapsed = time.perf_counter() - load_start

            if loader.cached_layer_count != config.num_hidden_layers:
                raise AssertionError("not all Transformer layers were cached")

            weight_memory = 0
            if device.type == "cuda":
                weight_memory = torch.cuda.memory_allocated(device)
                torch.cuda.reset_peak_memory_stats(device)

            runner = FlexGenPrefillRunner(
                config=config,
                weight_loader=loader,
                logger=lambda message: print(message, flush=True),
            )
            print(
                "[tinyllama-prefill] step 2/3: run full "
                f"{config.num_hidden_layers}-layer Prefill"
            )
            synchronize(device)
            prefill_start = time.perf_counter()
            result = runner.run(
                token_ids,
                collect_kv_cache=True,
            )
            synchronize(device)
            prefill_elapsed = time.perf_counter() - prefill_start

            print("[tinyllama-prefill] step 3/3: validate outputs and KV cache")
            validate_result(
                result,
                config,
                args.batch_size,
                args.prompt_length,
            )
            synchronize(device)

            peak_memory = weight_memory
            if device.type == "cuda":
                peak_memory = torch.cuda.max_memory_allocated(device)

            first_keys, first_values = result.kv_cache[0]
            output_checksum = float(
                result.logits.reshape(-1)[:8].float().sum().cpu()
                + first_keys.reshape(-1)[:8].float().sum().cpu()
                + first_values.reshape(-1)[:8].float().sum().cpu()
            )
            print(
                "[tinyllama-prefill] "
                f"hidden={tuple(result.hidden_states.shape)}, "
                f"logits={tuple(result.logits.shape)}, "
                f"KV/layer={tuple(first_keys.shape)}"
            )
            print(
                "[tinyllama-prefill] "
                f"next_tokens={result.next_token_ids.detach().cpu().reshape(-1).tolist()}, "
                f"checksum={output_checksum:.6f}"
            )
            print(
                "[tinyllama-prefill] "
                f"weight_load={load_elapsed:.6f}s, "
                f"prefill={prefill_elapsed:.6f}s, "
                f"weights_memory={weight_memory / (1024 ** 3):.3f}GiB, "
                f"peak_memory={peak_memory / (1024 ** 3):.3f}GiB"
            )
            print("[tinyllama-prefill] PASS")


if __name__ == "__main__":
    main()
