#!/usr/bin/env python3

import argparse
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "build-guest"))
sys.path.insert(0, str(PROJECT_DIR / "python"))

from cemu_flexgen import (
    AttentionBufferConfig,
    CemuAttentionDevice,
    KvCacheLayout,
    KvCacheStore,
    KvLayoutConfig,
    align_up,
)
from flexgen_adapter import FlexGenAttentionBackend
from flexgen_runtime import (
    FlexGenDecodeRunner,
    FlexGenGenerationRunner,
    FlexGenHfCheckpointLoader,
    FlexGenLlamaConfig,
    FlexGenPrefillRunner,
    FlexGenTorchAttentionBackend,
)


def positive_integer(value):
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def nonnegative_integer(value):
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be a nonnegative integer")
    return parsed


def nonnegative_float(value):
    parsed = float(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be nonnegative")
    return parsed


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run real TinyLlama Prefill on the Guest GPU and validate CEMU CSD "
            "Decode Attention against a PyTorch reference"
        )
    )
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--control", default="/dev/nvme0c3")
    parser.add_argument("--namespace", default="/dev/ng0n3")
    parser.add_argument("--nvm-dir", type=Path, default=Path("/mnt/nvme0"))
    parser.add_argument("--fdm-dir", type=Path, default=Path("/mnt/fdm0"))
    parser.add_argument(
        "--program",
        type=Path,
        help="override the CSD Attention shared library selected by --csd-target",
    )
    parser.add_argument(
        "--csd-target",
        choices=("cpu", "cuda"),
        default="cuda",
        help="functional-model backend used only for CSD Attention",
    )
    parser.add_argument("--batch-size", type=positive_integer, default=1)
    parser.add_argument("--prompt-length", type=positive_integer, default=8)
    parser.add_argument(
        "--prompt-text",
        help=(
            "optional real text prompt encoded with the tokenizer in --model-dir; "
            "the encoded length replaces --prompt-length"
        ),
    )
    parser.add_argument("--decode-steps", type=positive_integer, default=1)
    parser.add_argument("--staging-tokens", type=nonnegative_integer, default=0)
    parser.add_argument("--atol", type=nonnegative_float, default=1e-3)
    parser.add_argument("--rtol", type=nonnegative_float, default=1e-3)
    return parser.parse_args()


def log(message):
    print(message, flush=True)


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def require_directory(name, path):
    directory = path.resolve()
    if not directory.is_dir():
        raise FileNotFoundError(f"{name} does not exist: {directory}")
    return directory


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


def tokenize_prompt(model_directory, prompt_text, batch_size):
    if not prompt_text or not prompt_text.strip():
        raise ValueError("prompt text must contain at least one non-whitespace character")
    try:
        from transformers import AutoTokenizer
    except ImportError as error:
        raise RuntimeError(
            "--prompt-text requires Transformers in the active Python environment"
        ) from error

    tokenizer = AutoTokenizer.from_pretrained(
        model_directory,
        local_files_only=True,
        use_fast=True,
    )
    encoded = tokenizer(
        prompt_text,
        add_special_tokens=True,
        return_tensors="pt",
    )
    token_ids = encoded["input_ids"]
    if token_ids.ndim != 2 or token_ids.shape[0] != 1 or token_ids.shape[1] == 0:
        raise RuntimeError(
            f"tokenizer returned an invalid input_ids shape: {tuple(token_ids.shape)}"
        )
    if batch_size > 1:
        token_ids = token_ids.expand(batch_size, -1).clone()
    return tokenizer, token_ids


def format_token_ids(token_ids, limit=32):
    cpu_token_ids = token_ids.detach().cpu()
    if token_ids.shape[1] <= limit:
        return str(cpu_token_ids.tolist())
    prefixes = cpu_token_ids[:, :limit].tolist()
    return f"{prefixes} ... ({token_ids.shape[1]} tokens per batch)"


def log_generated_text(tokenizer, prompt_token_ids, generated_token_ids):
    prompt_token_ids = prompt_token_ids.detach().cpu()
    generated_token_ids = generated_token_ids.detach().cpu()
    full_token_ids = torch.cat(
        (prompt_token_ids, generated_token_ids),
        dim=1,
    )
    generated_text = tokenizer.batch_decode(
        generated_token_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    full_text = tokenizer.batch_decode(
        full_token_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    for batch, (continuation, complete) in enumerate(zip(generated_text, full_text)):
        log(
            f"[tinyllama-cemu] text batch={batch}, "
            f"continuation={continuation!r}, full={complete!r}"
        )


def preload_weights(loader, config, device):
    log("[tinyllama-cemu] step 1/8: preload real checkpoint weights once")
    start = time.perf_counter()
    loader.load_embedding()
    loader.load_final_norm()
    loader.load_lm_head()
    for layer in range(config.num_hidden_layers):
        log(
            f"[tinyllama-cemu] load layer={layer + 1}/"
            f"{config.num_hidden_layers}"
        )
        loader.load_layer(layer)
    synchronize(device)
    elapsed = time.perf_counter() - start
    if loader.cached_layer_count != config.num_hidden_layers:
        raise AssertionError("not all Transformer layers were cached")
    log(f"[tinyllama-cemu] checkpoint loaded in {elapsed:.6f}s")
    return elapsed


def cache_to_numpy(value, layout, token_count):
    return (
        value.detach()
        .cpu()
        .reshape(
            token_count,
            layout.config.batch_size,
            layout.config.num_kv_heads,
            layout.config.head_dim,
        )
        .numpy()
    )


def tensor_error(actual, expected):
    if tuple(actual.shape) != tuple(expected.shape):
        raise AssertionError(
            f"shape mismatch: actual={tuple(actual.shape)}, "
            f"expected={tuple(expected.shape)}"
        )
    difference = (actual.float() - expected.float()).abs()
    return difference.max().item(), difference.mean().item()


def assert_tensor_close(name, actual, expected, atol, rtol):
    maximum, mean = tensor_error(actual, expected)
    if not torch.allclose(actual, expected, atol=atol, rtol=rtol):
        raise AssertionError(
            f"{name} mismatch: max={maximum:.3e}, mean={mean:.3e}, "
            f"atol={atol}, rtol={rtol}"
        )
    return maximum, mean


def verify_prefill_storage(layout, cache_path, value_path, prefill_result):
    if prefill_result.kv_cache is None:
        raise AssertionError("Prefill did not return a KV cache for validation")
    if len(prefill_result.kv_cache) != layout.config.num_layers:
        raise AssertionError("Prefill KV layer count does not match the CEMU layout")
    layers = sorted({0, layout.config.num_layers - 1})
    token_count = prefill_result.kv_cache[0][0].shape[0]
    with KvCacheStore(layout, cache_path, value_path) as store:
        for layer in layers:
            stored_keys, stored_values = store.read_tokens(layer, 0, token_count)
            expected_keys = cache_to_numpy(
                prefill_result.kv_cache[layer][0],
                layout,
                token_count,
            )
            expected_values = cache_to_numpy(
                prefill_result.kv_cache[layer][1],
                layout,
                token_count,
            )
            np.testing.assert_array_equal(stored_keys, expected_keys)
            np.testing.assert_array_equal(stored_values, expected_values)
            log(
                f"[tinyllama-cemu] NVM Prefill layer={layer:02d} OK, "
                f"K/V={stored_keys.shape}, K[:4]={stored_keys.reshape(-1)[:4].tolist()}"
            )


def verify_decode_storage(
    layout,
    cache_path,
    value_path,
    generation,
    prompt_length,
):
    layers = sorted({0, layout.config.num_layers - 1})
    decode_steps = len(generation.steps)
    with KvCacheStore(layout, cache_path, value_path) as store:
        for layer in layers:
            stored_keys, stored_values = store.read_tokens(
                layer,
                prompt_length,
                decode_steps,
            )
            expected_keys = np.concatenate(
                [
                    cache_to_numpy(
                        step.decode_result.layer_outputs[layer].key,
                        layout,
                        1,
                    )
                    for step in generation.steps
                ],
                axis=0,
            )
            expected_values = np.concatenate(
                [
                    cache_to_numpy(
                        step.decode_result.layer_outputs[layer].value,
                        layout,
                        1,
                    )
                    for step in generation.steps
                ],
                axis=0,
            )
            np.testing.assert_array_equal(stored_keys, expected_keys)
            np.testing.assert_array_equal(stored_values, expected_values)
            log(
                f"[tinyllama-cemu] NVM Decode layer={layer:02d} OK, "
                f"tokens=[{prompt_length}, {prompt_length + decode_steps}), "
                f"K[:4]={stored_keys.reshape(-1)[:4].tolist()}"
            )


def compare_generations(cemu_generation, reference_generation, atol, rtol):
    if len(cemu_generation.steps) != len(reference_generation.steps):
        raise AssertionError("Decode step count mismatch")
    if not torch.equal(cemu_generation.token_ids, reference_generation.token_ids):
        raise AssertionError(
            "generated token sequence mismatch: "
            f"CEMU={cemu_generation.token_ids.detach().cpu().tolist()}, "
            f"reference={reference_generation.token_ids.detach().cpu().tolist()}"
        )

    for cemu_step, reference_step in zip(
        cemu_generation.steps,
        reference_generation.steps,
    ):
        if not torch.equal(cemu_step.input_token_ids, reference_step.input_token_ids):
            raise AssertionError(f"Decode input token mismatch at step {cemu_step.step}")

        cemu_result = cemu_step.decode_result
        reference_result = reference_step.decode_result
        if len(cemu_result.layer_outputs) != len(reference_result.layer_outputs):
            raise AssertionError("Decode layer output count mismatch")

        attention_maximum = 0.0
        for layer, (cemu_layer, reference_layer) in enumerate(
            zip(cemu_result.layer_outputs, reference_result.layer_outputs)
        ):
            assert_tensor_close(
                f"step {cemu_step.step} layer {layer} query",
                cemu_layer.query,
                reference_layer.query,
                atol,
                rtol,
            )
            assert_tensor_close(
                f"step {cemu_step.step} layer {layer} key",
                cemu_layer.key,
                reference_layer.key,
                atol,
                rtol,
            )
            assert_tensor_close(
                f"step {cemu_step.step} layer {layer} value",
                cemu_layer.value,
                reference_layer.value,
                atol,
                rtol,
            )
            maximum, _ = assert_tensor_close(
                f"step {cemu_step.step} layer {layer} attention",
                cemu_layer.attention_output,
                reference_layer.attention_output,
                atol,
                rtol,
            )
            attention_maximum = max(attention_maximum, maximum)

        hidden_maximum, _ = assert_tensor_close(
            f"step {cemu_step.step} hidden states",
            cemu_result.hidden_states,
            reference_result.hidden_states,
            atol,
            rtol,
        )
        logits_maximum, logits_mean = assert_tensor_close(
            f"step {cemu_step.step} logits",
            cemu_result.logits,
            reference_result.logits,
            atol,
            rtol,
        )
        if not torch.equal(cemu_result.next_token_ids, reference_result.next_token_ids):
            raise AssertionError(f"Decode next token mismatch at step {cemu_step.step}")
        log(
            f"[tinyllama-cemu] Decode step={cemu_step.step} OK, "
            f"attention_max={attention_maximum:.3e}, "
            f"hidden_max={hidden_maximum:.3e}, "
            f"logits(max={logits_maximum:.3e}, mean={logits_mean:.3e}), "
            f"next_tokens={cemu_result.next_token_ids.detach().cpu().reshape(-1).tolist()}"
        )


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("PyTorch cannot access the Guest GPU")

    model_directory = require_directory("model directory", args.model_dir)
    nvm_root = require_directory("NVM directory", args.nvm_dir)
    fdm_root = require_directory("FDM directory", args.fdm_dir)
    program_file = args.program or Path(
        "./build/dense_attention_devptr.so"
        if args.csd_target == "cuda"
        else "./build/dense_attention.so"
    )
    if not program_file.is_file():
        raise FileNotFoundError(f"CEMU Attention program does not exist: {program_file}")
    program_reference = str(program_file)
    if not program_file.is_absolute() and not program_reference.startswith("."):
        program_reference = f"./{program_reference}"

    config = FlexGenLlamaConfig.from_json(model_directory / "config.json")
    if config.dtype != torch.float32:
        raise ValueError(
            "the current dense Attention ABI requires a float32 checkpoint; "
            f"the model configuration uses {config.dtype}"
        )
    device = torch.device("cuda:0")
    tokenizer = None
    if args.prompt_text is None:
        token_ids = make_token_ids(config, args.batch_size, args.prompt_length)
        prompt_length = args.prompt_length
    else:
        tokenizer, token_ids = tokenize_prompt(
            model_directory,
            args.prompt_text,
            args.batch_size,
        )
        prompt_length = token_ids.shape[1]
    max_seq_len = prompt_length + args.decode_steps

    layout = KvCacheLayout(
        KvLayoutConfig(
            num_layers=config.num_hidden_layers,
            max_seq_len=max_seq_len,
            batch_size=args.batch_size,
            num_kv_heads=config.num_key_value_heads,
            head_dim=config.head_dim,
            dtype=np.float32,
        )
    )
    staging_tokens = args.staging_tokens or max_seq_len
    if staging_tokens > max_seq_len:
        raise ValueError("staging tokens cannot exceed the configured sequence length")
    staging_bytes = staging_tokens * layout.token_stride
    query_shape = (
        args.batch_size,
        config.num_attention_heads,
        config.head_dim,
    )
    query_elements = int(np.prod(query_shape))

    log(
        f"[tinyllama-cemu] model={model_directory}, device={device}, "
        f"gpu={torch.cuda.get_device_name(device)}, dtype={config.dtype}, "
        f"csd_target={args.csd_target}, csd_program={program_reference}"
    )
    log(
        f"[tinyllama-cemu] layers={config.num_hidden_layers}, "
        f"hidden={config.hidden_size}, q_heads={config.num_attention_heads}, "
        f"kv_heads={config.num_key_value_heads}, head_dim={config.head_dim}, "
        f"batch={args.batch_size}, prompt={prompt_length}, "
        f"decode_steps={args.decode_steps}"
    )
    log(
        f"[tinyllama-cemu] KV token_bytes={layout.token_bytes}, "
        f"token_stride={layout.token_stride}, layer_stride={layout.layer_stride}, "
        f"K_file={layout.file_size}, V_file={layout.file_size}, "
        f"staging_tokens={staging_tokens}, staging_bytes={staging_bytes}"
    )
    if args.prompt_text is not None:
        log(f"[tinyllama-cemu] prompt text={args.prompt_text!r}")
    log(f"[tinyllama-cemu] prompt token_ids={format_token_ids(token_ids)}")

    with tempfile.TemporaryDirectory(
        prefix="tinyllama-cemu-nvm-",
        dir=nvm_root,
    ) as nvm_directory, tempfile.TemporaryDirectory(
        prefix="tinyllama-cemu-fdm-",
        dir=fdm_root,
    ) as fdm_directory:
        nvm_path = Path(nvm_directory)
        fdm_path = Path(fdm_directory)
        k_cache_path = nvm_path / "k_cache"
        v_cache_path = nvm_path / "v_cache"
        buffers = AttentionBufferConfig(
            query_bytes=align_up(query_elements * np.dtype(np.float32).itemsize, 512),
            staging_bytes=staging_bytes,
            state_bytes=align_up(
                args.batch_size
                * config.num_attention_heads
                * (config.head_dim + 2)
                * np.dtype(np.float32).itemsize,
                512,
            ),
            output_bytes=align_up(
                query_elements * np.dtype(np.float32).itemsize,
                512,
            ),
            query_path=fdm_path / "attention_query",
            k_staging_path=fdm_path / "k_staging_0",
            v_staging_path=fdm_path / "v_staging_0",
            state_path=fdm_path / "attention_state",
            output_path=fdm_path / "attention_output",
        )
        attention_device = CemuAttentionDevice(
            layout=layout,
            buffers=buffers,
            program_name=f"tinyllama_cemu_attention_{args.csd_target}",
            program_path=program_reference,
            function_name="dense_attention",
            k_cache_path=k_cache_path,
            v_cache_path=v_cache_path,
            control_path=args.control,
            namespace_path=args.namespace,
            cuda_target=args.csd_target == "cuda",
            replace_program=True,
            replace_staging_files=True,
            logger=log,
        )
        cemu_backend = FlexGenAttentionBackend(
            layout=layout,
            k_cache_path=k_cache_path,
            v_cache_path=v_cache_path,
            attention_device=attention_device,
            replace_existing=True,
            logger=log,
        )

        with torch.inference_mode(), FlexGenHfCheckpointLoader(
            config,
            model_directory,
            device=device,
            cache_layers=True,
        ) as loader, cemu_backend:
            preload_weights(loader, config, device)

            log(
                "[tinyllama-cemu] step 2/8: Guest GPU Prefill and persist all "
                "layer K/V to CEMU NVM"
            )
            prefill_runner = FlexGenPrefillRunner(
                config,
                loader,
                kv_writer=cemu_backend,
                logger=log,
            )
            synchronize(device)
            start = time.perf_counter()
            prefill_result = prefill_runner.run(token_ids, collect_kv_cache=True)
            synchronize(device)
            prefill_elapsed = time.perf_counter() - start
            cemu_backend.flush()
            log(
                f"[tinyllama-cemu] Prefill complete in {prefill_elapsed:.6f}s, "
                f"next_tokens={prefill_result.next_token_ids.detach().cpu().reshape(-1).tolist()}"
            )

            log("[tinyllama-cemu] step 3/8: verify Prefill KV in CEMU NVM")
            verify_prefill_storage(
                layout,
                k_cache_path,
                v_cache_path,
                prefill_result,
            )

            log(
                "[tinyllama-cemu] step 4/8: run pure PyTorch Attention reference "
                "with the same Prefill KV"
            )
            reference_backend = FlexGenTorchAttentionBackend(
                config,
                prefill_result.kv_cache,
                logger=log,
                copy_cache=True,
            )
            reference_runner = FlexGenDecodeRunner(
                config,
                loader,
                reference_backend,
                logger=log,
            )
            synchronize(device)
            start = time.perf_counter()
            reference_generation = FlexGenGenerationRunner(
                reference_runner,
                logger=log,
            ).run(
                prefill_result.next_token_ids,
                start_position=prompt_length,
                decode_steps=args.decode_steps,
                collect_layer_outputs=True,
            )
            synchronize(device)
            reference_elapsed = time.perf_counter() - start
            log(
                f"[tinyllama-cemu] reference Decode={reference_elapsed:.6f}s, "
                f"tokens={reference_generation.token_ids.detach().cpu().tolist()}"
            )

            tokens_per_chunk = layout.tokens_per_chunk(
                staging_bytes,
                staging_bytes,
            )
            for step in range(args.decode_steps):
                valid_tokens = prompt_length + step + 1
                log(
                    f"[tinyllama-cemu] plan step={step}, valid_tokens={valid_tokens}, "
                    f"chunks={layout.chunk_count(valid_tokens, tokens_per_chunk)}, "
                    f"tokens_per_chunk={tokens_per_chunk}"
                )

            log(
                "[tinyllama-cemu] step 5/8: Prefill token -> QKV/RoPE -> append "
                "NVM -> FDM -> 5 MRS -> CSD CUDA Attention"
            )
            cemu_runner = FlexGenDecodeRunner(
                config,
                loader,
                cemu_backend,
                logger=log,
            )
            synchronize(device)
            start = time.perf_counter()
            with attention_device:
                cemu_generation = FlexGenGenerationRunner(
                    cemu_runner,
                    logger=log,
                ).run(
                    prefill_result.next_token_ids,
                    start_position=prompt_length,
                    decode_steps=args.decode_steps,
                    collect_layer_outputs=True,
                )
                synchronize(device)
                if attention_device.memory_range_count != 5:
                    raise AssertionError("CEMU Attention must create exactly five MRS")
            cemu_elapsed = time.perf_counter() - start
            log(
                f"[tinyllama-cemu] CEMU Decode={cemu_elapsed:.6f}s, "
                f"tokens={cemu_generation.token_ids.detach().cpu().tolist()}"
            )

            log("[tinyllama-cemu] step 6/8: verify appended Decode KV in CEMU NVM")
            cemu_backend.flush()
            verify_decode_storage(
                layout,
                k_cache_path,
                v_cache_path,
                cemu_generation,
                prompt_length,
            )

            log(
                "[tinyllama-cemu] step 7/8: compare every CEMU layer with the "
                "PyTorch reference"
            )
            compare_generations(
                cemu_generation,
                reference_generation,
                args.atol,
                args.rtol,
            )

            log("[tinyllama-cemu] step 8/8: report validated token sequence")
            if tokenizer is not None:
                log_generated_text(tokenizer, token_ids, cemu_generation.token_ids)
            log(
                "[tinyllama-cemu] PASS: real TinyLlama checkpoint -> Guest GPU "
                "Prefill -> CEMU NVM/FDM/MRS/CSD Attention -> Guest GPU Wo/MLP/LM "
                f"head; csd_target={args.csd_target}, "
                f"tokens={format_token_ids(cemu_generation.token_ids)}"
            )


if __name__ == "__main__":
    main()
