#!/usr/bin/env python3

import argparse
import gc
import time
from dataclasses import dataclass
from pathlib import Path

import torch

from flexgen_runtime import (
    FlexGenDecodeRunner,
    FlexGenHfCheckpointLoader,
    FlexGenLlamaConfig,
    FlexGenPrefillRunner,
    FlexGenTorchAttentionBackend,
)


@dataclass(frozen=True)
class DecodeSnapshot:
    prefill_token_ids: torch.Tensor
    logits: torch.Tensor
    next_token_ids: torch.Tensor
    appended_kv: tuple
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
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--batch-size", type=positive_integer, default=1)
    parser.add_argument("--prompt-length", type=positive_integer, default=8)
    parser.add_argument("--atol", type=nonnegative_float, default=1e-4)
    parser.add_argument("--rtol", type=nonnegative_float, default=1e-4)
    return parser.parse_args()


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def release_device_memory(device):
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        synchronize(device)


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


def preload_custom_weights(loader, config):
    loader.load_embedding()
    loader.load_final_norm()
    loader.load_lm_head()
    for layer in range(config.num_hidden_layers):
        print(
            f"[tinyllama-decode-reference] custom load layer={layer + 1}/"
            f"{config.num_hidden_layers}",
            flush=True,
        )
        loader.load_layer(layer)


def snapshot_appended_custom_cache(backend, config):
    appended = []
    for layer in range(config.num_hidden_layers):
        keys, values = backend.layer_cache(layer)
        appended.append(
            (
                keys[-1:].detach().cpu(),
                values[-1:].detach().cpu(),
            )
        )
    return tuple(appended)


def run_custom_decode(config, model_directory, token_ids, device):
    print(
        "[tinyllama-decode-reference] step 1/4: "
        "run custom Prefill + one-token Decode"
    )
    with FlexGenHfCheckpointLoader(
        config,
        model_directory,
        device=device,
        cache_layers=True,
    ) as loader:
        preload_custom_weights(loader, config)
        synchronize(device)

        prefill_runner = FlexGenPrefillRunner(config, loader)
        start = time.perf_counter()
        prefill = prefill_runner.run(token_ids, collect_kv_cache=True)
        synchronize(device)
        prefill_elapsed = time.perf_counter() - start

        backend = FlexGenTorchAttentionBackend(
            config,
            prefill.kv_cache,
            copy_cache=False,
        )
        decode_runner = FlexGenDecodeRunner(config, loader, backend)
        synchronize(device)
        start = time.perf_counter()
        decode = decode_runner.run(
            prefill.next_token_ids,
            token_position=token_ids.shape[1],
        )
        synchronize(device)
        decode_elapsed = time.perf_counter() - start

        snapshot = DecodeSnapshot(
            prefill_token_ids=prefill.next_token_ids.detach().cpu(),
            logits=decode.logits.detach().cpu(),
            next_token_ids=decode.next_token_ids.detach().cpu(),
            appended_kv=snapshot_appended_custom_cache(backend, config),
            prefill_elapsed=prefill_elapsed,
            decode_elapsed=decode_elapsed,
        )

    del decode, decode_runner, backend, prefill, prefill_runner, loader
    release_device_memory(device)
    return snapshot


def require_transformers(device):
    try:
        from transformers import AutoModelForCausalLM
    except ImportError as error:
        raise RuntimeError(
            "transformers is required; run: pip install transformers accelerate"
        ) from error
    if device.type == "cuda":
        try:
            import accelerate
        except ImportError as error:
            raise RuntimeError(
                "accelerate is required for direct GPU checkpoint loading"
            ) from error
    return AutoModelForCausalLM


def legacy_cache(past_key_values):
    if hasattr(past_key_values, "to_legacy_cache"):
        return past_key_values.to_legacy_cache()
    return tuple(past_key_values)


def snapshot_appended_huggingface_cache(
    past_key_values,
    config,
    batch_size,
    expected_tokens,
):
    cache = legacy_cache(past_key_values)
    if len(cache) != config.num_hidden_layers:
        raise AssertionError(
            f"Hugging Face returned {len(cache)} cache layers, "
            f"expected {config.num_hidden_layers}"
        )

    expected_shape = (
        batch_size,
        config.num_key_value_heads,
        expected_tokens,
        config.head_dim,
    )
    appended = []
    for layer, layer_cache in enumerate(cache):
        if len(layer_cache) < 2:
            raise AssertionError(f"Hugging Face layer {layer} cache has no K/V pair")
        keys, values = layer_cache[:2]
        if tuple(keys.shape) != expected_shape:
            raise AssertionError(
                f"Hugging Face layer {layer} K has shape {tuple(keys.shape)}, "
                f"expected {expected_shape}"
            )
        if tuple(values.shape) != expected_shape:
            raise AssertionError(
                f"Hugging Face layer {layer} V has shape {tuple(values.shape)}, "
                f"expected {expected_shape}"
            )
        keys = keys[:, :, -1:, :].permute(2, 0, 1, 3).reshape(
            1,
            batch_size * config.num_key_value_heads,
            config.head_dim,
        )
        values = values[:, :, -1:, :].permute(2, 0, 1, 3).reshape(
            1,
            batch_size * config.num_key_value_heads,
            config.head_dim,
        )
        appended.append((keys.detach().cpu(), values.detach().cpu()))
    return tuple(appended)


def run_huggingface_decode(config, model_directory, token_ids, device):
    print(
        "[tinyllama-decode-reference] step 2/4: "
        "run Hugging Face Prefill + one-token Decode"
    )
    auto_model = require_transformers(device)
    load_arguments = {
        "local_files_only": True,
        "use_safetensors": True,
        "trust_remote_code": False,
        "dtype": config.dtype,
        "attn_implementation": "eager",
    }
    if device.type == "cuda":
        load_arguments["device_map"] = {"": str(device)}

    load_start = time.perf_counter()
    model = auto_model.from_pretrained(str(model_directory), **load_arguments)
    if device.type != "cuda":
        model = model.to(device)
    model.eval()
    synchronize(device)
    load_elapsed = time.perf_counter() - load_start

    input_ids = token_ids.to(device)
    prefill_mask = torch.ones_like(input_ids, dtype=torch.long)
    synchronize(device)
    start = time.perf_counter()
    prefill = model(
        input_ids=input_ids,
        attention_mask=prefill_mask,
        use_cache=True,
        return_dict=True,
    )
    synchronize(device)
    prefill_elapsed = time.perf_counter() - start
    prefill_token_ids = prefill.logits[:, -1:, :].argmax(dim=-1)

    decode_mask = torch.ones(
        (token_ids.shape[0], token_ids.shape[1] + 1),
        dtype=torch.long,
        device=device,
    )
    synchronize(device)
    start = time.perf_counter()
    decode = model(
        input_ids=prefill_token_ids,
        attention_mask=decode_mask,
        past_key_values=prefill.past_key_values,
        use_cache=True,
        return_dict=True,
    )
    synchronize(device)
    decode_elapsed = time.perf_counter() - start
    logits = decode.logits.detach().cpu()
    next_token_ids = logits[:, -1:, :].argmax(dim=-1)
    appended_kv = snapshot_appended_huggingface_cache(
        decode.past_key_values,
        config,
        token_ids.shape[0],
        token_ids.shape[1] + 1,
    )
    snapshot = DecodeSnapshot(
        prefill_token_ids=prefill_token_ids.detach().cpu(),
        logits=logits,
        next_token_ids=next_token_ids,
        appended_kv=appended_kv,
        prefill_elapsed=prefill_elapsed,
        decode_elapsed=decode_elapsed,
    )

    del decode, prefill, model, input_ids, prefill_mask, decode_mask
    release_device_memory(device)
    print(
        "[tinyllama-decode-reference] "
        f"Hugging Face load={load_elapsed:.6f}s, "
        f"prefill={prefill_elapsed:.6f}s, decode={decode_elapsed:.6f}s"
    )
    return snapshot


def error_statistics(actual, expected):
    if tuple(actual.shape) != tuple(expected.shape):
        raise AssertionError(
            f"shape mismatch: actual={tuple(actual.shape)}, "
            f"expected={tuple(expected.shape)}"
        )
    difference = (actual.float() - expected.float()).abs()
    return difference.max().item(), difference.mean().item()


def compare_snapshots(custom, reference, atol, rtol):
    print("[tinyllama-decode-reference] step 3/4: compare appended KV")
    mismatches = []
    prefill_tokens_match = torch.equal(
        custom.prefill_token_ids,
        reference.prefill_token_ids,
    )
    print(
        "[tinyllama-decode-reference] "
        f"decode_input custom={custom.prefill_token_ids.reshape(-1).tolist()}, "
        f"reference={reference.prefill_token_ids.reshape(-1).tolist()}, "
        f"status={'OK' if prefill_tokens_match else 'MISMATCH'}"
    )
    if not prefill_tokens_match:
        mismatches.append("decode input token")

    if len(custom.appended_kv) != len(reference.appended_kv):
        raise AssertionError(
            f"cache layer count mismatch: custom={len(custom.appended_kv)}, "
            f"reference={len(reference.appended_kv)}"
        )
    for layer, ((custom_keys, custom_values), (reference_keys, reference_values)) in enumerate(
        zip(custom.appended_kv, reference.appended_kv)
    ):
        key_max, key_mean = error_statistics(custom_keys, reference_keys)
        value_max, value_mean = error_statistics(custom_values, reference_values)
        keys_match = torch.allclose(custom_keys, reference_keys, atol=atol, rtol=rtol)
        values_match = torch.allclose(
            custom_values,
            reference_values,
            atol=atol,
            rtol=rtol,
        )
        status = "OK" if keys_match and values_match else "MISMATCH"
        print(
            f"[tinyllama-decode-reference] layer={layer:02d} {status} "
            f"K(max={key_max:.3e}, mean={key_mean:.3e}) "
            f"V(max={value_max:.3e}, mean={value_mean:.3e})"
        )
        if not keys_match:
            mismatches.append(f"layer {layer} appended K")
        if not values_match:
            mismatches.append(f"layer {layer} appended V")

    print("[tinyllama-decode-reference] step 4/4: compare logits and next token")
    logits_max, logits_mean = error_statistics(custom.logits, reference.logits)
    logits_match = torch.allclose(
        custom.logits,
        reference.logits,
        atol=atol,
        rtol=rtol,
    )
    tokens_match = torch.equal(custom.next_token_ids, reference.next_token_ids)
    print(
        "[tinyllama-decode-reference] "
        f"logits={'OK' if logits_match else 'MISMATCH'} "
        f"max={logits_max:.3e}, mean={logits_mean:.3e}"
    )
    print(
        "[tinyllama-decode-reference] "
        f"custom_tokens={custom.next_token_ids.reshape(-1).tolist()}, "
        f"reference_tokens={reference.next_token_ids.reshape(-1).tolist()}, "
        f"status={'OK' if tokens_match else 'MISMATCH'}"
    )
    if not logits_match:
        mismatches.append("decode logits")
    if not tokens_match:
        mismatches.append("decode next token")
    if mismatches:
        raise AssertionError("reference mismatches: " + ", ".join(mismatches))


def main():
    args = parse_args()
    model_directory = args.model_dir.resolve()
    config = FlexGenLlamaConfig.from_json(model_directory / "config.json")
    device = torch.device("cuda:0" if args.cuda else "cpu")
    if args.cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")

    token_ids = make_token_ids(config, args.batch_size, args.prompt_length)
    print(
        "[tinyllama-decode-reference] "
        f"model={model_directory}, device={device}, dtype={config.dtype}"
    )
    print(
        "[tinyllama-decode-reference] "
        f"token_ids={token_ids.tolist()}, atol={args.atol}, rtol={args.rtol}"
    )

    with torch.inference_mode():
        custom = run_custom_decode(config, model_directory, token_ids, device)
        reference = run_huggingface_decode(
            config,
            model_directory,
            token_ids,
            device,
        )

    compare_snapshots(custom, reference, args.atol, args.rtol)
    print(
        "[tinyllama-decode-reference] "
        f"custom_prefill={custom.prefill_elapsed:.6f}s, "
        f"custom_decode={custom.decode_elapsed:.6f}s, "
        f"reference_prefill={reference.prefill_elapsed:.6f}s, "
        f"reference_decode={reference.decode_elapsed:.6f}s"
    )
    print("[tinyllama-decode-reference] PASS")


if __name__ == "__main__":
    main()
