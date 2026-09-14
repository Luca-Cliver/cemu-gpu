#!/usr/bin/env python3

import argparse
import csv
import os
import statistics
import sys
import tempfile
import threading
import time
from contextlib import ExitStack
from pathlib import Path

import numpy as np
import torch


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "build-guest"))
sys.path.insert(0, str(PROJECT_DIR / "python"))
sys.path.insert(0, str(PROJECT_DIR))

from cemu_flexgen import (
    AttentionBufferConfig,
    CemuAttentionDevice,
    CemuAttentionSlotScheduler,
    CemuAttentionSharedWorkers,
    CemuSparfDevice,
    KvCacheLayout,
    KvCacheStore,
    KvLayoutConfig,
    SparfKvCacheLayout,
    align_up,
)
from flexgen_adapter import FlexGenAttentionBackend, FlexGenMicrobatchKvWriter, SparfAttentionBackend
from opt_runtime import (
    OptDecodeRunner,
    OptGenerationRunner,
    OptMultiBatchPrefillRunner,
    OptCheckpointLoader,
    OptConfig,
    OptMultiBatchDecodeRunner,
    OptPrefillRunner,
    OptTorchAttentionBackend,
)
from experiments import PipelineTrace, SparfAttentionRuntimeModel, load_experiment_config
from experiments.softmax_ratio import SoftmaxRatioTable
from phase_profiler import PhaseProfiler, profile_scope
from runtime_common import partition_kv_cache_by_batch


_LOG_START_NS = time.perf_counter_ns()
_TIMELINE_ENABLED = os.environ.get("CEMU_PIPELINE_TIMELINE", "").lower() in (
    "1",
    "true",
    "yes",
)
_LOG_LOCK = threading.Lock()


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
            "Run real OPT Prefill on the Guest GPU and validate CEMU CSD "
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
    parser.add_argument(
        "--gpu-batch-size",
        type=positive_integer,
        help="sequences handled by each independent GPU microbatch",
    )
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
    parser.add_argument("--attention-slots", type=positive_integer, default=2)
    parser.add_argument("--attention-mode", choices=("dense", "sparf"), default="dense")
    parser.add_argument("--sparf-top-r", type=positive_integer, default=16)
    parser.add_argument("--sparf-compression-ratio", type=positive_integer, default=8)
    parser.add_argument("--runtime-config", type=Path, help="paper hardware config used for CSD runtime modeling")
    parser.add_argument("--softmax-ratio-file", type=Path,
                        help="optional measured CUDA Softmax ratios; requires SparF and --runtime-config")
    parser.add_argument("--atol", type=nonnegative_float, default=2e-3)
    parser.add_argument("--rtol", type=nonnegative_float, default=2e-3)
    parser.add_argument("--qkv-atol", type=nonnegative_float, default=5e-3)
    parser.add_argument("--qkv-rtol", type=nonnegative_float, default=2e-3)
    parser.add_argument("--logits-atol", type=nonnegative_float, default=2e-2)
    parser.add_argument("--logits-rtol", type=nonnegative_float, default=2e-3)
    parser.add_argument(
        "--trace-file",
        type=Path,
        help="record low-overhead Guest pipeline events and overlap analysis as CSV",
    )
    parser.add_argument(
        "--diagnose-errors",
        action="store_true",
        help="print per-layer CEMU/reference error metrics before strict checks",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="skip correctness/reference work and measure the CEMU pipeline only",
    )
    parser.add_argument(
        "--functional",
        action="store_true",
        help=(
            "run the complete OPT Prefill/Decode CEMU path without retaining "
            "reference tensors or per-step Decode outputs"
        ),
    )
    parser.add_argument(
        "--warmup-iterations",
        type=nonnegative_integer,
        default=1,
    )
    parser.add_argument("--iterations", type=positive_integer, default=5)
    parser.add_argument(
        "--benchmark-output",
        type=Path,
        default=PROJECT_DIR / "results" / "opt-cemu-benchmark.csv",
    )
    parser.add_argument(
        "--profile-output",
        type=Path,
        help=(
            "write aggregate phase timing and transferred bytes as CSV; "
            "supported with --functional"
        ),
    )
    return parser.parse_args()


def log(message):
    with _LOG_LOCK:
        if _TIMELINE_ENABLED:
            elapsed_us = (time.perf_counter_ns() - _LOG_START_NS) / 1000.0
            thread_name = threading.current_thread().name
            print(
                f"[pipeline-time][{elapsed_us:14.3f} us][{thread_name}] {message}",
                flush=True,
            )
        else:
            print(message, flush=True)


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def write_profile(profiler, output_path):
    profile_path = profiler.write_csv(output_path)
    log(
        "[opt-profile] accumulated worker times may overlap; compare them with "
        "pipeline.decode_wall instead of adding every row"
    )
    log("[opt-profile] *_wall = Guest caller time; *.cuda_stream = CUDA event interval; "
        "csd_model.* = modeled device time. These are overlapping measurements, not additive. "
        "prefill_forward_wall includes KV work and waits.")
    for line in profiler.summary():
        log(f"[opt-profile] {line}")
    log(f"[opt-profile] CSV={profile_path}")


def make_cemu_decode_runner(
    config,
    loader,
    cemu_backends,
    gpu_batch_size,
    logger,
):
    if len(cemu_backends) > 1:
        return OptMultiBatchDecodeRunner(
            config,
            loader,
            cemu_backends,
            gpu_batch_size=gpu_batch_size,
            logger=logger,
        )
    return OptDecodeRunner(
        config,
        loader,
        cemu_backends[0],
        logger=logger,
    )


def make_torch_reference_decode_runner(
    config,
    loader,
    kv_cache,
    gpu_batch_size,
    logger,
):
    cache_partitions = partition_kv_cache_by_batch(
        config,
        kv_cache,
        gpu_batch_size,
    )
    reference_backends = tuple(
        OptTorchAttentionBackend(
            config,
            partition,
            copy_cache=True,
            accumulation_dtype=torch.float32,
        )
        for partition in cache_partitions
    )
    if len(reference_backends) > 1:
        return OptMultiBatchDecodeRunner(
            config,
            loader,
            reference_backends,
            gpu_batch_size=gpu_batch_size,
            logger=logger,
        )
    return OptDecodeRunner(
        config,
        loader,
        reference_backends[0],
        logger=logger,
    )


def open_attention_schedulers(
    stack,
    attention_schedulers,
    attention_slots,
    num_gpu_batches,
):
    for scheduler in attention_schedulers:
        stack.enter_context(scheduler)
    program_ids = tuple(slot.program_id for slot in attention_slots)
    mrs_ids = tuple(slot.memory_range_set_id for slot in attention_slots)
    if len(set(program_ids)) != len(program_ids):
        raise AssertionError("CEMU Attention slots must use distinct programs")
    if len(set(mrs_ids)) != len(mrs_ids):
        raise AssertionError("CEMU Attention slots must use distinct MRS IDs")
    if any(slot.memory_range_count != 5 for slot in attention_slots):
        raise AssertionError("each CEMU Attention slot must create exactly five ranges")
    log(
        f"[opt-cemu] Attention slots ready: programs={program_ids}, "
        f"MRS={mrs_ids}, ranges={5 * len(attention_slots)}, "
        f"gpu_batches={num_gpu_batches}"
    )


def run_pipeline_benchmark(
    args,
    token_ids,
    prompt_length,
    prefill_runner,
    cemu_runner,
    kv_writer,
    device,
):
    rows = []
    total_iterations = args.warmup_iterations + args.iterations
    generation_runner = OptGenerationRunner(cemu_runner, logger=None)
    log(
        f"[opt-benchmark] warmup={args.warmup_iterations}, "
        f"iterations={args.iterations}; checkpoint loading and CEMU setup excluded"
    )
    for sequence in range(total_iterations):
        measured_iteration = sequence - args.warmup_iterations
        is_warmup = measured_iteration < 0

        synchronize(device)
        total_start = time.perf_counter_ns()
        prefill_start = total_start
        prefill_result = prefill_runner.run(
            token_ids,
            collect_kv_cache=False,
            last_token_only=True,
        )
        kv_writer.flush()
        synchronize(device)
        prefill_end = time.perf_counter_ns()

        generation = generation_runner.run(
            prefill_result.next_token_ids,
            start_position=prompt_length,
            decode_steps=args.decode_steps,
            collect_layer_outputs=False,
            collect_steps=False,
        )
        kv_writer.flush()
        synchronize(device)
        decode_end = time.perf_counter_ns()

        prefill_seconds = (prefill_end - prefill_start) / 1e9
        decode_seconds = (decode_end - prefill_end) / 1e9
        total_seconds = (decode_end - total_start) / 1e9
        generated_tokens = args.batch_size * args.decode_steps
        del generation, prefill_result

        label = "warmup" if is_warmup else f"iteration={measured_iteration}"
        log(
            f"[opt-benchmark] {label}: prefill={prefill_seconds:.6f}s, "
            f"decode={decode_seconds:.6f}s, total={total_seconds:.6f}s"
        )
        if not is_warmup:
            rows.append(
                {
                    "iteration": measured_iteration,
                    "csd_target": args.csd_target,
                    "batch_size": args.batch_size,
                    "gpu_batch_size": args.gpu_batch_size or args.batch_size,
                    "prompt_tokens": prompt_length,
                    "decode_steps": args.decode_steps,
                    "prefill_seconds": prefill_seconds,
                    "decode_seconds": decode_seconds,
                    "total_seconds": total_seconds,
                    "decode_ms_per_step": decode_seconds * 1000.0 / args.decode_steps,
                    "decode_ms_per_token": decode_seconds * 1000.0 / generated_tokens,
                    "decode_tokens_per_second": generated_tokens / decode_seconds,
                }
            )

    output_path = args.benchmark_output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    for metric in ("prefill_seconds", "decode_seconds", "total_seconds"):
        values = [row[metric] for row in rows]
        log(
            f"[opt-benchmark] {metric}: median={statistics.median(values):.6f}s, "
            f"min={min(values):.6f}s, max={max(values):.6f}s"
        )
    throughput = [row["decode_tokens_per_second"] for row in rows]
    log(
        "[opt-benchmark] decode throughput: "
        f"median={statistics.median(throughput):.3f} token/s; CSV={output_path}"
    )


def run_functional_pipeline(
    args,
    token_ids,
    prompt_length,
    prefill_runner,
    cemu_runner,
    cemu_backends,
    kv_writer,
    attention_schedulers,
    attention_slots,
    num_gpu_batches,
    tokenizer,
    device,
    profiler=None,
):
    log(
        "[opt-functional] step 1/3: Guest GPU Prefill and persist K/V to "
        "CEMU NVM"
    )
    with profile_scope(profiler, "pipeline.prefill_wall"):
        with profile_scope(profiler, "pipeline.prefill_forward_wall"):
            prefill_result = prefill_runner.run(
                token_ids,
                collect_kv_cache=False,
                last_token_only=True,
            )
        with profile_scope(profiler, "pipeline.prefill_kv_flush_wall"):
            kv_writer.flush()
        synchronize(device)
    log(
        "[opt-functional] Prefill complete; initial Decode tokens="
        f"{prefill_result.next_token_ids.detach().cpu().reshape(-1).tolist()}"
    )

    log(
        "[opt-functional] step 2/3: autoregressive Decode through "
        "NVM/FDM/MRS/CSD Attention"
    )
    with profile_scope(profiler, "pipeline.decode_wall"):
        with ExitStack() as scheduler_stack:
            open_attention_schedulers(
                scheduler_stack,
                attention_schedulers,
                attention_slots,
                num_gpu_batches,
            )
            generation = OptGenerationRunner(cemu_runner, logger=None).run(
                prefill_result.next_token_ids,
                start_position=prompt_length,
                decode_steps=args.decode_steps,
                collect_layer_outputs=False,
                collect_steps=False,
            )
            synchronize(device)

    modeled = []
    seen_devices = set()
    for backend in cemu_backends:
        csd_device = getattr(backend, "attention_device", None)
        if id(csd_device) in seen_devices:
            continue
        seen_devices.add(id(csd_device))
        summary = getattr(csd_device, "modeled_runtime_summary", None)
        if callable(summary):
            modeled.append(summary())
    if modeled:
        total_ns = sum(item[0] for item in modeled)
        log(
            f"[opt-functional] CSD workflow modeled sum: total={total_ns / 1e9:.6f}s, "
            f"attention_requests={sum(item[1] for item in modeled)}"
        )

    log("[opt-functional] step 3/3: flush persistent Decode K/V")
    with profile_scope(profiler, "pipeline.final_flush_wall"):
        for backend in cemu_backends:
            flush = getattr(backend, "flush", None)
            if callable(flush):
                flush()
    if tokenizer is not None:
        log_generated_text(tokenizer, token_ids, generation.token_ids)
    log(
        "[opt-functional] PASS: real OPT checkpoint -> Guest GPU Prefill -> "
        "CEMU NVM/FDM/MRS/CSD Attention -> Guest GPU Wo/MLP/LM head; "
        f"batch={args.batch_size}, prompt={prompt_length}, "
        f"decode={args.decode_steps}, tokens={format_token_ids(generation.token_ids)}"
    )


def require_directory(name, path):
    directory = path.resolve()
    if not directory.is_dir():
        raise FileNotFoundError(f"{name} does not exist: {directory}")
    return directory


def make_token_ids(config, batch_size, prompt_length):
    if config.vocab_size <= 3:
        raise ValueError("OPT vocabulary must contain more than three tokens")
    token_ids = torch.arange(
        batch_size * prompt_length,
        dtype=torch.long,
    ).reshape(batch_size, prompt_length)
    token_ids = token_ids.remainder(config.vocab_size - 3).add(3)
    token_ids[:, 0] = config.bos_token_id
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


def format_token_ids(token_ids, limit=32, batch_limit=8):
    cpu_token_ids = token_ids.detach().cpu()
    displayed = cpu_token_ids[:batch_limit, :limit].tolist()
    suffixes = []
    if token_ids.shape[0] > batch_limit:
        suffixes.append(f"{token_ids.shape[0]} batches")
    if token_ids.shape[1] > limit:
        suffixes.append(f"{token_ids.shape[1]} tokens per batch")
    if not suffixes:
        return str(displayed)
    return f"{displayed} ... ({', '.join(suffixes)})"


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
            f"[opt-cemu] text batch={batch}, "
            f"continuation={continuation!r}, full={complete!r}"
        )


def preload_weights(loader, config, device, step_label="step 1/8"):
    log(f"[opt-cemu] {step_label}: preload real checkpoint weights once")
    start = time.perf_counter()
    loader.load_embedding()
    loader.load_final_norm()
    loader.load_lm_head()
    for layer in range(config.num_hidden_layers):
        log(
            f"[opt-cemu] load layer={layer + 1}/"
            f"{config.num_hidden_layers}"
        )
        loader.load_layer(layer)
    synchronize(device)
    elapsed = time.perf_counter() - start
    if loader.cached_layer_count != config.num_hidden_layers:
        raise AssertionError("not all Transformer layers were cached")
    log(f"[opt-cemu] checkpoint loaded in {elapsed:.6f}s")
    return elapsed


def cache_to_numpy(
    value,
    layout,
    token_count,
    total_batch_size=None,
    batch_start=0,
):
    total_batch_size = total_batch_size or layout.config.batch_size
    batch_end = batch_start + layout.config.batch_size
    return (
        value.detach()
        .cpu()
        .reshape(
            token_count,
            total_batch_size,
            layout.config.num_kv_heads,
            layout.config.head_dim,
        )[:, batch_start:batch_end]
        .contiguous()
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


def verify_prefill_storage(
    layouts,
    cache_paths,
    prefill_result,
    total_batch_size,
):
    layout = layouts[0]
    if prefill_result.kv_cache is None:
        raise AssertionError("Prefill did not return a KV cache for validation")
    if len(prefill_result.kv_cache) != layout.config.num_layers:
        raise AssertionError("Prefill KV layer count does not match the CEMU layout")
    layers = sorted({0, layout.config.num_layers - 1})
    token_count = prefill_result.kv_cache[0][0].shape[0]
    for microbatch, (microbatch_layout, paths) in enumerate(
        zip(layouts, cache_paths)
    ):
        batch_start = microbatch * microbatch_layout.config.batch_size
        with KvCacheStore(microbatch_layout, *paths) as store:
            for layer in layers:
                stored_keys, stored_values = store.read_tokens(layer, 0, token_count)
                expected_keys = cache_to_numpy(
                    prefill_result.kv_cache[layer][0],
                    microbatch_layout,
                    token_count,
                    total_batch_size,
                    batch_start,
                )
                expected_values = cache_to_numpy(
                    prefill_result.kv_cache[layer][1],
                    microbatch_layout,
                    token_count,
                    total_batch_size,
                    batch_start,
                )
                np.testing.assert_array_equal(stored_keys, expected_keys)
                np.testing.assert_array_equal(stored_values, expected_values)
                log(
                    f"[opt-cemu] NVM Prefill microbatch={microbatch}, "
                    f"layer={layer:02d} OK, K/V={stored_keys.shape}, "
                    f"K[:4]={stored_keys.reshape(-1)[:4].tolist()}"
                )


def verify_decode_storage(
    layouts,
    cache_paths,
    generation,
    prompt_length,
    total_batch_size,
):
    layout = layouts[0]
    layers = sorted({0, layout.config.num_layers - 1})
    decode_steps = len(generation.steps)
    for microbatch, (microbatch_layout, paths) in enumerate(
        zip(layouts, cache_paths)
    ):
        batch_start = microbatch * microbatch_layout.config.batch_size
        with KvCacheStore(microbatch_layout, *paths) as store:
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
                            microbatch_layout,
                            1,
                            total_batch_size,
                            batch_start,
                        )
                        for step in generation.steps
                    ],
                    axis=0,
                )
                expected_values = np.concatenate(
                    [
                        cache_to_numpy(
                            step.decode_result.layer_outputs[layer].value,
                            microbatch_layout,
                            1,
                            total_batch_size,
                            batch_start,
                        )
                        for step in generation.steps
                    ],
                    axis=0,
                )
                np.testing.assert_array_equal(stored_keys, expected_keys)
                np.testing.assert_array_equal(stored_values, expected_values)
                log(
                    f"[opt-cemu] NVM Decode microbatch={microbatch}, "
                    f"layer={layer:02d} OK, tokens=[{prompt_length}, "
                    f"{prompt_length + decode_steps}), "
                    f"K[:4]={stored_keys.reshape(-1)[:4].tolist()}"
                )


def build_cemu_path_attention_references(
    config,
    prefill_kv_cache,
    cemu_generation,
):
    backend = OptTorchAttentionBackend(
        config,
        prefill_kv_cache,
        copy_cache=True,
        accumulation_dtype=torch.float32,
    )
    step_outputs = []
    with torch.no_grad():
        for generation_step in cemu_generation.steps:
            layer_outputs = generation_step.decode_result.layer_outputs
            if layer_outputs is None:
                raise AssertionError(
                    "CEMU Decode must collect layer outputs for local validation"
                )
            local_outputs = []
            for layer, layer_output in enumerate(layer_outputs):
                backend.append_decode(
                    layer,
                    generation_step.token_position,
                    layer_output.key,
                    layer_output.value,
                )
                local_outputs.append(
                    backend.decode(
                        layer,
                        layer_output.query,
                        valid_tokens=generation_step.token_position + 1,
                    )
                )
            step_outputs.append(tuple(local_outputs))
    return tuple(step_outputs)


def compare_generations(
    cemu_generation,
    reference_generation,
    local_attention_references,
    atol,
    rtol,
    logits_atol,
    logits_rtol,
):
    if len(cemu_generation.steps) != len(reference_generation.steps):
        raise AssertionError("Decode step count mismatch")
    if len(cemu_generation.steps) != len(local_attention_references):
        raise AssertionError("Local Attention reference step count mismatch")

    for cemu_step, reference_step, local_step_attention in zip(
        cemu_generation.steps,
        reference_generation.steps,
        local_attention_references,
    ):
        if not torch.equal(cemu_step.input_token_ids, reference_step.input_token_ids):
            raise AssertionError(f"Decode input token mismatch at step {cemu_step.step}")

        cemu_result = cemu_step.decode_result
        reference_result = reference_step.decode_result
        if len(cemu_result.layer_outputs) != len(reference_result.layer_outputs):
            raise AssertionError("Decode layer output count mismatch")
        if len(cemu_result.layer_outputs) != len(local_step_attention):
            raise AssertionError("Local Attention reference layer count mismatch")

        local_attention_maximum = 0.0
        trajectory_qkv_maximum = 0.0
        trajectory_attention_maximum = 0.0
        for layer, (cemu_layer, reference_layer, local_attention) in enumerate(
            zip(
                cemu_result.layer_outputs,
                reference_result.layer_outputs,
                local_step_attention,
            )
        ):
            for actual, expected in (
                (cemu_layer.query, reference_layer.query),
                (cemu_layer.key, reference_layer.key),
                (cemu_layer.value, reference_layer.value),
            ):
                maximum, _ = tensor_error(actual, expected)
                trajectory_qkv_maximum = max(trajectory_qkv_maximum, maximum)
            trajectory_maximum, _ = tensor_error(
                cemu_layer.attention_output,
                reference_layer.attention_output,
            )
            trajectory_attention_maximum = max(
                trajectory_attention_maximum,
                trajectory_maximum,
            )
            maximum, _ = assert_tensor_close(
                f"step {cemu_step.step} layer {layer} local FP32-accumulation Attention",
                cemu_layer.attention_output,
                local_attention,
                atol,
                rtol,
            )
            local_attention_maximum = max(local_attention_maximum, maximum)

        hidden_maximum, hidden_mean = tensor_error(
            cemu_result.hidden_states,
            reference_result.hidden_states,
        )
        logits_maximum, logits_mean = assert_tensor_close(
            f"step {cemu_step.step} logits",
            cemu_result.logits,
            reference_result.logits,
            logits_atol,
            logits_rtol,
        )
        if not torch.equal(cemu_result.next_token_ids, reference_result.next_token_ids):
            raise AssertionError(f"Decode next token mismatch at step {cemu_step.step}")
        log(
            f"[opt-cemu] Decode step={cemu_step.step} OK, "
            f"local_attention_max={local_attention_maximum:.3e}, "
            f"trajectory_qkv_max={trajectory_qkv_maximum:.3e}, "
            f"trajectory_attention_max={trajectory_attention_maximum:.3e}, "
            f"trajectory_hidden(max={hidden_maximum:.3e}, mean={hidden_mean:.3e}), "
            f"logits(max={logits_maximum:.3e}, mean={logits_mean:.3e}), "
            f"next_tokens={cemu_result.next_token_ids.detach().cpu().reshape(-1).tolist()}"
        )

    if not torch.equal(cemu_generation.token_ids, reference_generation.token_ids):
        raise AssertionError(
            "generated token sequence mismatch: "
            f"CEMU={cemu_generation.token_ids.detach().cpu().tolist()}, "
            f"reference={reference_generation.token_ids.detach().cpu().tolist()}"
        )


def diagnose_generations(
    cemu_generation,
    reference_generation,
    local_attention_references,
    atol,
    rtol,
    qkv_atol,
    qkv_rtol,
    logits_atol,
    logits_rtol,
):
    if len(cemu_generation.steps) != len(reference_generation.steps):
        raise AssertionError("Decode step count mismatch")

    for cemu_step, reference_step, local_step_attention in zip(
        cemu_generation.steps,
        reference_generation.steps,
        local_attention_references,
    ):
        cemu_result = cemu_step.decode_result
        reference_result = reference_step.decode_result
        for layer, (cemu_layer, reference_layer, local_attention) in enumerate(
            zip(
                cemu_result.layer_outputs,
                reference_result.layer_outputs,
                local_step_attention,
            )
        ):
            maximum, mean = tensor_error(
                cemu_layer.attention_output,
                local_attention,
            )
            status = "OK" if torch.allclose(
                cemu_layer.attention_output,
                local_attention,
                atol=atol,
                rtol=rtol,
            ) else "DIFF"
            log(
                f"[opt-cemu-diagnostic] step={cemu_step.step}, "
                f"layer={layer}, local_fp32_attention: {status}, "
                f"max={maximum:.3e}, mean={mean:.3e}, "
                f"atol={atol:.3e}, rtol={rtol:.3e}"
            )
            for name, actual, expected in (
                ("trajectory_query", cemu_layer.query, reference_layer.query),
                ("trajectory_key", cemu_layer.key, reference_layer.key),
                ("trajectory_value", cemu_layer.value, reference_layer.value),
                (
                    "trajectory_attention",
                    cemu_layer.attention_output,
                    reference_layer.attention_output,
                ),
            ):
                is_qkv = name in (
                    "trajectory_query",
                    "trajectory_key",
                    "trajectory_value",
                )
                tensor_atol = qkv_atol if is_qkv else atol
                tensor_rtol = qkv_rtol if is_qkv else rtol
                maximum, mean = tensor_error(actual, expected)
                status = "OK" if torch.allclose(
                    actual,
                    expected,
                    atol=tensor_atol,
                    rtol=tensor_rtol,
                ) else "DIFF"
                log(
                    f"[opt-cemu-diagnostic] step={cemu_step.step}, "
                    f"layer={layer}, {name}: {status}, "
                    f"max={maximum:.3e}, mean={mean:.3e}, "
                    f"atol={tensor_atol:.3e}, rtol={tensor_rtol:.3e}"
                )

        for name, actual, expected in (
            ("hidden", cemu_result.hidden_states, reference_result.hidden_states),
            ("logits", cemu_result.logits, reference_result.logits),
        ):
            tensor_atol = logits_atol if name == "logits" else atol
            tensor_rtol = logits_rtol if name == "logits" else rtol
            maximum, mean = tensor_error(actual, expected)
            status = "OK" if torch.allclose(
                actual,
                expected,
                atol=tensor_atol,
                rtol=tensor_rtol,
            ) else "DIFF"
            log(
                f"[opt-cemu-diagnostic] step={cemu_step.step}, {name}: {status}, "
                f"max={maximum:.3e}, mean={mean:.3e}, "
                f"atol={tensor_atol:.3e}, rtol={tensor_rtol:.3e}"
            )


def main():
    args = parse_args()
    selected_modes = sum(
        mode is not None and mode is not False
        for mode in (args.benchmark, args.functional, args.trace_file)
    )
    if selected_modes > 1:
        raise ValueError(
            "--benchmark, --functional, and --trace-file are separate modes"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("PyTorch cannot access the Guest GPU")
    if args.attention_mode == "dense" and args.attention_slots < 2:
        raise ValueError("the slot-scheduled CEMU path requires at least two slots")
    if args.attention_mode == "sparf" and not (args.functional or args.benchmark):
        raise ValueError("SparF OPT integration currently uses --functional or --benchmark; use the device test for strict operator validation")
    if args.profile_output is not None and not args.functional:
        raise ValueError("--profile-output currently requires --functional")

    model_directory = require_directory("model directory", args.model_dir)
    nvm_root = require_directory("NVM directory", args.nvm_dir)
    fdm_root = require_directory("FDM directory", args.fdm_dir)
    default_program = (
        f"./build/{'sparf_attention' if args.attention_mode == 'sparf' else 'dense_attention'}"
        f"{'_devptr' if args.csd_target == 'cuda' else ''}.so"
    )
    program_file = args.program or Path(default_program)
    if not program_file.is_file():
        raise FileNotFoundError(f"CEMU Attention program does not exist: {program_file}")
    program_reference = str(program_file)
    if not program_file.is_absolute() and not program_reference.startswith("."):
        program_reference = f"./{program_reference}"

    config = OptConfig.from_json(model_directory / "config.json")
    if config.dtype == torch.float16:
        layout_dtype = np.dtype(np.float16)
    elif config.dtype == torch.float32:
        layout_dtype = np.dtype(np.float32)
    else:
        raise ValueError(
            "CEMU dense Attention supports only float16 or float32; "
            f"the OPT configuration uses {config.dtype}"
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
    if max_seq_len > config.max_seq_len:
        raise ValueError(
            f"prompt length {prompt_length} + Decode steps {args.decode_steps} "
            f"exceeds OPT capacity {config.max_seq_len}"
        )
    gpu_batch_size = args.gpu_batch_size or args.batch_size
    if gpu_batch_size > args.batch_size:
        raise ValueError("gpu batch size cannot exceed the total batch size")
    if args.batch_size % gpu_batch_size != 0:
        raise ValueError("batch size must be divisible by gpu batch size")
    num_gpu_batches = args.batch_size // gpu_batch_size
    trace = PipelineTrace() if args.trace_file is not None else None
    profiler = PhaseProfiler() if args.profile_output is not None else None
    detail_logger = None if args.benchmark or args.functional else (trace or log)
    softmax_ratios = None
    if args.softmax_ratio_file is not None:
        if args.runtime_config is None or args.attention_mode != "sparf":
            raise ValueError("--softmax-ratio-file requires --attention-mode sparf and --runtime-config")
        softmax_ratios = SoftmaxRatioTable.load(args.softmax_ratio_file)
        softmax_ratios.require_sparf(
            gpu_batch_size, config.num_attention_heads, prompt_length,
            args.decode_steps, args.sparf_compression_ratio,
        )
    sparf_runtime_model = None
    if args.runtime_config is not None and args.attention_mode == "sparf":
        sparf_runtime_model = SparfAttentionRuntimeModel(
            load_experiment_config(args.runtime_config).instcsd,
            softmax_ratios=softmax_ratios,
        )
        print(
            f"[opt-cemu] Softmax timing={'measured CUDA shape ratio (not FPGA-validated)' if softmax_ratios else 'legacy linear anchor scaling'}, "
            f"ratio_file={args.softmax_ratio_file}"
        )

    layout_type = SparfKvCacheLayout if args.attention_mode == "sparf" else KvCacheLayout
    layouts = tuple(
        layout_type(
            KvLayoutConfig(
                num_layers=config.num_hidden_layers,
                max_seq_len=max_seq_len,
                batch_size=gpu_batch_size,
                num_kv_heads=config.num_key_value_heads,
                head_dim=config.head_dim,
                dtype=layout_dtype,
            )
        )
        for _ in range(num_gpu_batches)
    )
    layout = layouts[0]
    staging_tokens = args.staging_tokens or max_seq_len
    if staging_tokens > max_seq_len:
        raise ValueError("staging tokens cannot exceed the configured sequence length")
    if args.attention_mode == "dense" and staging_tokens < max_seq_len:
        raise ValueError(
            "the current double-slot runtime requires one Attention chunk; "
            "use --staging-tokens 0 or at least the full configured sequence length"
        )
    staging_bytes = staging_tokens * layout.token_stride
    query_shape = (
        gpu_batch_size,
        config.num_attention_heads,
        config.head_dim,
    )
    query_elements = int(np.prod(query_shape))

    log(
        f"[opt-cemu] model={model_directory}, device={device}, "
        f"gpu={torch.cuda.get_device_name(device)}, dtype={config.dtype}, "
        f"csd_target={args.csd_target}, attention_mode={args.attention_mode}, "
        f"csd_program={program_reference}"
    )
    log(
        f"[opt-cemu] layers={config.num_hidden_layers}, "
        f"hidden={config.hidden_size}, q_heads={config.num_attention_heads}, "
        f"kv_heads={config.num_key_value_heads}, head_dim={config.head_dim}, "
        f"batch={args.batch_size}, gpu_batch_size={gpu_batch_size}, "
        f"gpu_batches={num_gpu_batches}, prompt={prompt_length}, "
        f"decode_steps={args.decode_steps}, attention_slots={args.attention_slots}"
    )
    if args.attention_mode == "sparf":
        log(
            f"[opt-cemu] SparF K_token={layout.token_file_size}, "
            f"K_channel={layout.channel_file_size}, V_token={layout.token_file_size}, "
            f"files={3 * num_gpu_batches}, top_r={args.sparf_top_r}, "
            f"compression=1/{args.sparf_compression_ratio}"
        )
    else:
        log(
            f"[opt-cemu] KV token_bytes={layout.token_bytes}, "
            f"token_stride={layout.token_stride}, layer_stride={layout.layer_stride}, "
            f"K_file_per_gpu_batch={layout.file_size}, V_file_per_gpu_batch={layout.file_size}, "
            f"KV_files={2 * num_gpu_batches}, staging_tokens={staging_tokens}, staging_bytes={staging_bytes}"
        )
    if args.prompt_text is not None:
        log(f"[opt-cemu] prompt text={args.prompt_text!r}")
    log(f"[opt-cemu] prompt token_ids={format_token_ids(token_ids)}")

    with tempfile.TemporaryDirectory(
        prefix="opt-cemu-nvm-",
        dir=nvm_root,
    ) as nvm_directory, tempfile.TemporaryDirectory(
        prefix="opt-cemu-fdm-",
        dir=fdm_root,
    ) as fdm_directory:
        nvm_path = Path(nvm_directory)
        fdm_path = Path(fdm_directory)
        cache_paths = tuple(
            ((nvm_path / f"k_token_{microbatch}", nvm_path / f"k_channel_{microbatch}", nvm_path / f"v_token_{microbatch}")
             if args.attention_mode == "sparf" else
             (nvm_path / f"k_cache_{microbatch}", nvm_path / f"v_cache_{microbatch}"))
            for microbatch in range(num_gpu_batches)
        )
        query_bytes = align_up(
            query_elements * layout_dtype.itemsize,
            512,
        )
        state_bytes = align_up(
            gpu_batch_size
            * config.num_attention_heads
            * (config.head_dim + 2)
            * np.dtype(np.float32).itemsize,
            512,
        )
        attention_slots = tuple(
                CemuAttentionDevice(
                    layout=layout,
                    buffers=AttentionBufferConfig(
                        query_bytes=query_bytes,
                        staging_bytes=staging_bytes,
                        state_bytes=state_bytes,
                        output_bytes=query_bytes,
                        query_path=fdm_path / f"slot_{slot}_query",
                        k_staging_path=fdm_path
                        / f"slot_{slot}_k_staging",
                        v_staging_path=fdm_path
                        / f"slot_{slot}_v_staging",
                        state_path=fdm_path / f"slot_{slot}_state",
                        output_path=fdm_path / f"slot_{slot}_output",
                    ),
                    program_name=(
                        f"cemu_attention_{args.csd_target}_slot_{slot}"
                    ),
                    program_path=program_reference,
                    function_name="dense_attention",
                    k_cache_path=cache_paths[0][0],
                    v_cache_path=cache_paths[0][1],
                    control_path=args.control,
                    namespace_path=args.namespace,
                    cuda_target=args.csd_target == "cuda",
                    replace_program=True,
                    replace_staging_files=True,
                    logger=detail_logger,
                    attention_scale=1.0,
                    profiler=profiler,
                )
                for slot in range(args.attention_slots)
        ) if args.attention_mode == "dense" else ()
        shared_csd_workers = CemuAttentionSharedWorkers(logger=detail_logger) if attention_slots else None
        attention_scheduler = CemuAttentionSlotScheduler(
                attention_slots,
                workers=shared_csd_workers,
                logger=detail_logger,
                profiler=profiler,
        ) if attention_slots else None
        attention_schedulers = (attention_scheduler,) if attention_scheduler else ()
        if args.attention_mode == "dense":
            cemu_backends = tuple(
            FlexGenAttentionBackend(
                layout=layouts[microbatch],
                k_cache_path=cache_paths[microbatch][0],
                v_cache_path=cache_paths[microbatch][1],
                attention_scheduler=attention_scheduler,
                replace_existing=True,
                logger=detail_logger,
                profiler=profiler,
            )
            for microbatch in range(num_gpu_batches)
            )
        else:
            sparf_device = CemuSparfDevice(
                    layout, *cache_paths[0],
                    fdm_directory=fdm_path,
                    program_path=program_reference,
                    query_heads=config.num_attention_heads,
                    top_r=args.sparf_top_r,
                    compression_ratio=args.sparf_compression_ratio,
                    control_path=args.control,
                    namespace_path=args.namespace,
                    cuda_target=args.csd_target == "cuda",
                    runtime_model=sparf_runtime_model,
                    profiler=profiler,
                    logger=detail_logger,
                    name_prefix="shared",
                )
            cemu_backends = tuple(
                SparfAttentionBackend(
                    layouts[microbatch], *cache_paths[microbatch],
                    attention_device=sparf_device,
                    replace_existing=True, logger=detail_logger, profiler=profiler,
                )
                for microbatch in range(num_gpu_batches)
            )
        kv_writer = FlexGenMicrobatchKvWriter(
            cemu_backends,
            logger=detail_logger,
        )

        with torch.inference_mode(), OptCheckpointLoader(
            config,
            model_directory,
            device=device,
            cache_layers=True,
        ) as loader, ExitStack() as backend_stack, kv_writer:
            for backend in cemu_backends:
                backend_stack.enter_context(backend)
            with profile_scope(profiler, "setup.checkpoint_preload_wall"):
                preload_weights(
                    loader,
                    config,
                    device,
                    step_label=(
                        "setup"
                        if args.benchmark or args.functional
                        else "step 1/8"
                    ),
                )

            prefill_runner = OptPrefillRunner(
                config, loader, kv_writer=kv_writer, logger=detail_logger
            ) if num_gpu_batches == 1 else OptMultiBatchPrefillRunner(
                config,
                loader,
                kv_writer,
                gpu_batch_size,
                logger=detail_logger,
            )
            cemu_runner = make_cemu_decode_runner(
                config,
                loader,
                cemu_backends,
                gpu_batch_size,
                detail_logger,
            )

            if args.benchmark:
                with ExitStack() as scheduler_stack:
                    open_attention_schedulers(
                        scheduler_stack,
                        attention_schedulers,
                        attention_slots,
                        num_gpu_batches,
                    )
                    run_pipeline_benchmark(
                        args,
                        token_ids,
                        prompt_length,
                        prefill_runner,
                        cemu_runner,
                        kv_writer,
                        device,
                    )
                return

            if args.functional:
                run_functional_pipeline(
                    args,
                    token_ids,
                    prompt_length,
                    prefill_runner,
                    cemu_runner,
                    cemu_backends,
                    kv_writer,
                    attention_schedulers,
                    attention_slots,
                    num_gpu_batches,
                    tokenizer,
                    device,
                    profiler,
                )
                if profiler is not None:
                    write_profile(profiler, args.profile_output)
                return

            log(
                "[opt-cemu] step 2/8: Guest GPU Prefill and persist all "
                "layer K/V to CEMU NVM"
            )
            synchronize(device)
            start = time.perf_counter()
            prefill_result = prefill_runner.run(token_ids, collect_kv_cache=True)
            synchronize(device)
            prefill_elapsed = time.perf_counter() - start
            kv_writer.flush()
            log(
                f"[opt-cemu] Prefill complete in {prefill_elapsed:.6f}s, "
                f"next_tokens={prefill_result.next_token_ids.detach().cpu().reshape(-1).tolist()}"
            )

            log("[opt-cemu] step 3/8: verify Prefill KV in CEMU NVM")
            verify_prefill_storage(
                layouts,
                cache_paths,
                prefill_result,
                args.batch_size,
            )

            log(
                "[opt-cemu] step 4/8: run matched PyTorch reference with "
                "the same microbatch partition and FP32 Attention accumulation"
            )
            reference_runner = make_torch_reference_decode_runner(
                config,
                loader,
                prefill_result.kv_cache,
                gpu_batch_size,
                logger=(log if trace is None else None),
            )
            synchronize(device)
            start = time.perf_counter()
            reference_generation = OptGenerationRunner(
                reference_runner,
                logger=(log if trace is None else None),
            ).run(
                prefill_result.next_token_ids,
                start_position=prompt_length,
                decode_steps=args.decode_steps,
                collect_layer_outputs=True,
            )
            synchronize(device)
            reference_elapsed = time.perf_counter() - start
            log(
                f"[opt-cemu] reference Decode={reference_elapsed:.6f}s, "
                f"tokens={reference_generation.token_ids.detach().cpu().tolist()}"
            )

            tokens_per_chunk = layout.tokens_per_chunk(
                staging_bytes,
                staging_bytes,
            )
            for step in range(args.decode_steps):
                valid_tokens = prompt_length + step + 1
                log(
                    f"[opt-cemu] plan step={step}, valid_tokens={valid_tokens}, "
                    f"chunks={layout.chunk_count(valid_tokens, tokens_per_chunk)}, "
                    f"tokens_per_chunk={tokens_per_chunk}"
                )

            log(
                "[opt-cemu] step 5/8: Prefill token -> QKV -> append "
                "NVM -> double-slot FDM/MRS -> CSD Attention"
            )
            synchronize(device)
            start = time.perf_counter()
            with ExitStack() as scheduler_stack:
                open_attention_schedulers(
                    scheduler_stack,
                    attention_schedulers,
                    attention_slots,
                    num_gpu_batches,
                )
                cemu_generation = OptGenerationRunner(
                    cemu_runner,
                    logger=(log if trace is None else None),
                ).run(
                    prefill_result.next_token_ids,
                    start_position=prompt_length,
                    decode_steps=args.decode_steps,
                    collect_layer_outputs=True,
                )
                synchronize(device)
            cemu_elapsed = time.perf_counter() - start
            log(
                f"[opt-cemu] CEMU Decode={cemu_elapsed:.6f}s, "
                f"tokens={cemu_generation.token_ids.detach().cpu().tolist()}"
            )

            log("[opt-cemu] step 6/8: verify appended Decode KV in CEMU NVM")
            kv_writer.flush()
            verify_decode_storage(
                layouts,
                cache_paths,
                cemu_generation,
                prompt_length,
                args.batch_size,
            )

            log(
                "[opt-cemu] step 7/8: validate local CSD Attention and "
                "end-to-end model outputs"
            )
            local_attention_references = build_cemu_path_attention_references(
                config,
                prefill_result.kv_cache,
                cemu_generation,
            )
            if args.diagnose_errors:
                diagnose_generations(
                    cemu_generation,
                    reference_generation,
                    local_attention_references,
                    args.atol,
                    args.rtol,
                    args.qkv_atol,
                    args.qkv_rtol,
                    args.logits_atol,
                    args.logits_rtol,
                )
            compare_generations(
                cemu_generation,
                reference_generation,
                local_attention_references,
                args.atol,
                args.rtol,
                args.logits_atol,
                args.logits_rtol,
            )

            log("[opt-cemu] step 8/8: report validated token sequence")
            if tokenizer is not None:
                log_generated_text(tokenizer, token_ids, cemu_generation.token_ids)
            log(
                "[opt-cemu] PASS: real OPT checkpoint -> Guest GPU "
                "Prefill -> CEMU NVM/FDM/MRS/CSD Attention -> Guest GPU Wo/MLP/LM "
                f"head; csd_target={args.csd_target}, "
                f"tokens={format_token_ids(cemu_generation.token_ids)}"
            )
            if trace is not None:
                event_path, overlap_path = trace.write(args.trace_file)
                log(
                    f"[pipeline-trace] events={len(trace.events)}, "
                    f"event_csv={event_path.resolve()}, "
                    f"overlap_csv={overlap_path.resolve()}"
                )
                for summary_line in trace.summary():
                    log(f"[pipeline-trace] {summary_line}")


if __name__ == "__main__":
    main()
