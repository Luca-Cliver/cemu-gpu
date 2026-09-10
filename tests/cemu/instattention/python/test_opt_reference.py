#!/usr/bin/env python3

import argparse
import gc
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import torch

from opt_runtime import (
    OptCheckpointLoader,
    OptConfig,
    OptDecodeRunner,
    OptGenerationRunner,
    OptPrefillRunner,
    OptTorchAttentionBackend,
)


@dataclass(frozen=True)
class OptGenerationSnapshot:
    prefill_logits: torch.Tensor
    prefill_next_token_ids: torch.Tensor
    prefill_cache: tuple
    generated_token_ids: torch.Tensor
    decode_logits: tuple
    final_cache: tuple
    prefill_elapsed: float
    decode_elapsed: float


def positive_integer(value):
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def nonnegative_float(value):
    parsed = float(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be nonnegative")
    return parsed


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--prompt-text", default="Hello, my name is")
    parser.add_argument("--batch-size", type=positive_integer, default=1)
    parser.add_argument("--decode-steps", type=positive_integer, default=2)
    parser.add_argument("--atol", type=nonnegative_float, default=2e-3)
    parser.add_argument("--rtol", type=nonnegative_float, default=2e-3)
    parser.add_argument("--cuda", action="store_true")
    return parser.parse_args()


def require_transformers():
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as error:
        raise RuntimeError(
            "transformers is required; run: pip install transformers"
        ) from error
    return AutoModelForCausalLM, AutoTokenizer


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def release_device_memory(device):
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        synchronize(device)


def tokenize_prompt(model_directory, prompt_text, batch_size):
    _, auto_tokenizer = require_transformers()
    tokenizer = auto_tokenizer.from_pretrained(
        str(model_directory),
        local_files_only=True,
        trust_remote_code=False,
    )
    encoded = tokenizer(
        prompt_text,
        return_tensors="pt",
        add_special_tokens=True,
    )
    token_ids = encoded.input_ids.repeat(batch_size, 1)
    if token_ids.shape[1] == 0:
        raise ValueError("the prompt produced no tokens")
    return tokenizer, token_ids


def snapshot_cache(cache) -> Tuple[Tuple[torch.Tensor, torch.Tensor], ...]:
    return tuple(
        (keys.detach().cpu().clone(), values.detach().cpu().clone())
        for keys, values in cache
    )


def snapshot_backend_cache(backend, num_layers):
    return snapshot_cache(
        tuple(backend.layer_cache(layer) for layer in range(num_layers))
    )


def preload_custom_weights(loader, config):
    loader.load_embedding()
    loader.load_final_norm()
    loader.load_lm_head()
    for layer in range(config.num_hidden_layers):
        print(
            f"[opt-reference] custom load layer={layer + 1}/"
            f"{config.num_hidden_layers}",
            flush=True,
        )
        loader.load_layer(layer)


def run_custom_generation(config, model_directory, token_ids, decode_steps, device):
    print("[opt-reference] step 1/4: run custom OPT Prefill + Decode")
    with OptCheckpointLoader(
        config,
        model_directory,
        device=device,
        cache_layers=True,
    ) as loader:
        preload_custom_weights(loader, config)
        synchronize(device)

        prefill_runner = OptPrefillRunner(config, loader)
        start = time.perf_counter()
        prefill = prefill_runner.run(token_ids, collect_kv_cache=True)
        synchronize(device)
        prefill_elapsed = time.perf_counter() - start

        prefill_cache = snapshot_cache(prefill.kv_cache)
        backend = OptTorchAttentionBackend(
            config,
            prefill.kv_cache,
            copy_cache=False,
        )
        decode_runner = OptDecodeRunner(config, loader, backend)
        synchronize(device)
        start = time.perf_counter()
        generation = OptGenerationRunner(decode_runner).run(
            prefill.next_token_ids,
            start_position=token_ids.shape[1],
            decode_steps=decode_steps,
        )
        synchronize(device)
        decode_elapsed = time.perf_counter() - start

        snapshot = OptGenerationSnapshot(
            prefill_logits=prefill.logits.detach().cpu(),
            prefill_next_token_ids=prefill.next_token_ids.detach().cpu(),
            prefill_cache=prefill_cache,
            generated_token_ids=generation.token_ids.detach().cpu(),
            decode_logits=tuple(
                step.decode_result.logits.detach().cpu()
                for step in generation.steps
            ),
            final_cache=snapshot_backend_cache(
                backend,
                config.num_hidden_layers,
            ),
            prefill_elapsed=prefill_elapsed,
            decode_elapsed=decode_elapsed,
        )

    del generation, decode_runner, backend, prefill, prefill_runner, loader
    release_device_memory(device)
    return snapshot


def legacy_cache(past_key_values):
    if hasattr(past_key_values, "to_legacy_cache"):
        return past_key_values.to_legacy_cache()
    return tuple(past_key_values)


def convert_huggingface_cache(cache, config):
    cache = legacy_cache(cache)
    if len(cache) != config.num_hidden_layers:
        raise AssertionError(
            f"Hugging Face returned {len(cache)} cache layers, "
            f"expected {config.num_hidden_layers}"
        )

    converted = []
    for layer, layer_cache in enumerate(cache):
        if len(layer_cache) < 2:
            raise AssertionError(f"Hugging Face layer {layer} cache has no K/V pair")
        keys, values = layer_cache[:2]
        if keys.ndim != 4 or values.shape != keys.shape:
            raise AssertionError(f"invalid Hugging Face cache at layer {layer}")
        if keys.shape[1] != config.num_key_value_heads:
            raise AssertionError(f"invalid Hugging Face head count at layer {layer}")
        token_count = keys.shape[2]
        batch_size = keys.shape[0]
        keys = keys.permute(2, 0, 1, 3).reshape(
            token_count,
            batch_size * config.num_key_value_heads,
            config.head_dim,
        )
        values = values.permute(2, 0, 1, 3).reshape(
            token_count,
            batch_size * config.num_key_value_heads,
            config.head_dim,
        )
        converted.append(
            (keys.detach().cpu().clone(), values.detach().cpu().clone())
        )
    return tuple(converted)


def run_huggingface_generation(
    config,
    model_directory,
    token_ids,
    decode_steps,
    device,
):
    print("[opt-reference] step 2/4: run Hugging Face OPT Prefill + Decode")
    auto_model, _ = require_transformers()
    load_start = time.perf_counter()
    model = auto_model.from_pretrained(
        str(model_directory),
        local_files_only=True,
        trust_remote_code=False,
        use_safetensors=False,
        dtype=config.dtype,
        attn_implementation="eager",
    ).to(device)
    model.eval()
    synchronize(device)
    load_elapsed = time.perf_counter() - load_start

    input_ids = token_ids.to(device)
    attention_mask = torch.ones_like(input_ids, dtype=torch.long)
    synchronize(device)
    start = time.perf_counter()
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=True,
        return_dict=True,
    )
    synchronize(device)
    prefill_elapsed = time.perf_counter() - start

    prefill_logits = outputs.logits.detach().cpu()
    current_token_ids = outputs.logits[:, -1, :].argmax(dim=1, keepdim=True)
    prefill_next_token_ids = current_token_ids.detach().cpu()
    prefill_cache = convert_huggingface_cache(outputs.past_key_values, config)
    past_key_values = outputs.past_key_values
    generated = [prefill_next_token_ids]
    decode_logits = []

    synchronize(device)
    start = time.perf_counter()
    for step in range(decode_steps):
        attention_mask = torch.ones(
            (token_ids.shape[0], token_ids.shape[1] + step + 1),
            dtype=torch.long,
            device=device,
        )
        outputs = model(
            input_ids=current_token_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=True,
            return_dict=True,
        )
        step_logits = outputs.logits.detach().cpu()
        current_token_ids = outputs.logits[:, -1, :].argmax(
            dim=1,
            keepdim=True,
        )
        decode_logits.append(step_logits)
        generated.append(current_token_ids.detach().cpu())
        past_key_values = outputs.past_key_values
    synchronize(device)
    decode_elapsed = time.perf_counter() - start

    snapshot = OptGenerationSnapshot(
        prefill_logits=prefill_logits,
        prefill_next_token_ids=prefill_next_token_ids,
        prefill_cache=prefill_cache,
        generated_token_ids=torch.cat(generated, dim=1),
        decode_logits=tuple(decode_logits),
        final_cache=convert_huggingface_cache(past_key_values, config),
        prefill_elapsed=prefill_elapsed,
        decode_elapsed=decode_elapsed,
    )
    del outputs, past_key_values, model, input_ids, attention_mask
    release_device_memory(device)
    print(
        f"[opt-reference] Hugging Face load={load_elapsed:.6f}s, "
        f"prefill={prefill_elapsed:.6f}s, decode={decode_elapsed:.6f}s"
    )
    return snapshot


def compare_tensor(name, actual, expected, atol, rtol):
    if actual.shape != expected.shape:
        raise AssertionError(
            f"{name} shape mismatch: {tuple(actual.shape)} != {tuple(expected.shape)}"
        )
    difference = (actual.float() - expected.float()).abs()
    maximum = difference.max().item()
    mean = difference.mean().item()
    if not torch.allclose(actual, expected, atol=atol, rtol=rtol):
        raise AssertionError(
            f"{name} mismatch: max={maximum:.3e}, mean={mean:.3e}"
        )
    return maximum, mean


def compare_cache(name, actual, expected, atol, rtol):
    if len(actual) != len(expected):
        raise AssertionError(f"{name} layer count mismatch")
    maximum = 0.0
    mean_sum = 0.0
    for layer, ((actual_keys, actual_values), (expected_keys, expected_values)) in enumerate(
        zip(actual, expected)
    ):
        key_maximum, key_mean = compare_tensor(
            f"{name} layer={layer} K",
            actual_keys,
            expected_keys,
            atol,
            rtol,
        )
        value_maximum, value_mean = compare_tensor(
            f"{name} layer={layer} V",
            actual_values,
            expected_values,
            atol,
            rtol,
        )
        maximum = max(maximum, key_maximum, value_maximum)
        mean_sum += key_mean + value_mean
    mean = mean_sum / (2 * len(actual))
    print(
        f"[opt-reference] {name}=OK layers={len(actual)}, "
        f"max={maximum:.3e}, mean={mean:.3e}"
    )


def compare_snapshots(custom, reference, atol, rtol):
    print("[opt-reference] step 3/4: compare Prefill logits and all KV layers")
    prefill_maximum, prefill_mean = compare_tensor(
        "Prefill logits",
        custom.prefill_logits,
        reference.prefill_logits,
        atol,
        rtol,
    )
    if not torch.equal(
        custom.prefill_next_token_ids,
        reference.prefill_next_token_ids,
    ):
        raise AssertionError("Prefill next token mismatch")
    print(
        f"[opt-reference] Prefill logits=OK max={prefill_maximum:.3e}, "
        f"mean={prefill_mean:.3e}, "
        f"next={custom.prefill_next_token_ids.reshape(-1).tolist()}"
    )
    compare_cache(
        "Prefill KV",
        custom.prefill_cache,
        reference.prefill_cache,
        atol,
        rtol,
    )

    print("[opt-reference] step 4/4: compare Decode logits, tokens, and final KV")
    if len(custom.decode_logits) != len(reference.decode_logits):
        raise AssertionError("Decode step count mismatch")
    for step, (custom_logits, reference_logits) in enumerate(
        zip(custom.decode_logits, reference.decode_logits)
    ):
        maximum, mean = compare_tensor(
            f"Decode step={step} logits",
            custom_logits,
            reference_logits,
            atol,
            rtol,
        )
        print(
            f"[opt-reference] Decode step={step} logits=OK "
            f"max={maximum:.3e}, mean={mean:.3e}"
        )
    if not torch.equal(custom.generated_token_ids, reference.generated_token_ids):
        raise AssertionError(
            "generated token sequence mismatch: "
            f"custom={custom.generated_token_ids.tolist()}, "
            f"reference={reference.generated_token_ids.tolist()}"
        )
    compare_cache(
        "Final KV",
        custom.final_cache,
        reference.final_cache,
        atol,
        rtol,
    )


def main():
    args = parse_args()
    model_directory = args.model_dir.resolve()
    config = OptConfig.from_json(model_directory / "config.json")
    device = torch.device("cuda:0" if args.cuda else "cpu")
    if args.cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    tokenizer, token_ids = tokenize_prompt(
        model_directory,
        args.prompt_text,
        args.batch_size,
    )
    if token_ids.shape[1] + args.decode_steps > config.max_seq_len:
        raise ValueError("prompt and Decode steps exceed the OPT context length")

    print(
        f"[opt-reference] model={model_directory}, device={device}, "
        f"dtype={config.dtype}, layers={config.num_hidden_layers}"
    )
    print(
        f"[opt-reference] prompt={args.prompt_text!r}, "
        f"token_ids={token_ids.tolist()}, decode_steps={args.decode_steps}, "
        f"atol={args.atol}, rtol={args.rtol}"
    )

    with torch.inference_mode():
        custom = run_custom_generation(
            config,
            model_directory,
            token_ids,
            args.decode_steps,
            device,
        )
        reference = run_huggingface_generation(
            config,
            model_directory,
            token_ids,
            args.decode_steps,
            device,
        )

    compare_snapshots(custom, reference, args.atol, args.rtol)
    print(
        f"[opt-reference] custom prefill={custom.prefill_elapsed:.6f}s, "
        f"decode={custom.decode_elapsed:.6f}s"
    )
    print(
        "[opt-reference] generated token_ids="
        f"{custom.generated_token_ids.tolist()}, text="
        f"{tokenizer.batch_decode(custom.generated_token_ids, skip_special_tokens=True)}"
    )
    print("[opt-reference] PASS")


if __name__ == "__main__":
    main()
