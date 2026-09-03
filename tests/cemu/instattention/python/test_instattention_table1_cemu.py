#!/usr/bin/env python3

import argparse
import math
import sys
import tempfile
import time
from pathlib import Path

import numpy as np


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "build-guest"))
sys.path.insert(0, str(PROJECT_DIR / "python"))
sys.path.insert(0, str(PROJECT_DIR))

from cemu_flexgen import (
    AttentionBufferConfig,
    CemuAttentionDevice,
    KvCacheLayout,
    KvCacheStore,
    KvLayoutConfig,
    align_up,
)
from experiments import DenseAttentionRuntimeModel, load_experiment_config


DEFAULT_CONFIG = (
    PROJECT_DIR / "experiments" / "configs" / "opt13b_dense_1csd.json"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the InstAttention Table I dense Attention anchor on CEMU"
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--control", default="/dev/nvme0c3")
    parser.add_argument("--namespace", default="/dev/ng0n3")
    parser.add_argument("--nvm-dir", default="/mnt/nvme0")
    parser.add_argument("--fdm-dir", default="/mnt/fdm0")
    parser.add_argument("--program")
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--warmup-iterations", type=int, default=0)
    parser.add_argument("--iterations", type=int, default=1)
    return parser.parse_args()


def validate_directory(name, path):
    directory = Path(path)
    if not directory.is_dir():
        raise FileNotFoundError(f"{name} does not exist: {directory}")
    return directory


def attention_reference(query, keys, values, scale):
    query_fp32 = query.astype(np.float32)
    keys_fp32 = keys.astype(np.float32)
    values_fp32 = values.astype(np.float32)
    scores = np.einsum("bhd,tbhd->bht", query_fp32, keys_fp32) * scale
    probabilities = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
    probabilities /= np.sum(probabilities, axis=-1, keepdims=True)
    output = np.einsum("bht,tbhd->bhd", probabilities, values_fp32)
    return output.astype(query.dtype)


def main():
    args = parse_args()
    if args.warmup_iterations < 0:
        raise ValueError("warmup iterations must be non-negative")
    if args.iterations <= 0:
        raise ValueError("iterations must be positive")
    config = load_experiment_config(args.config)
    runtime_model = DenseAttentionRuntimeModel(config.instcsd)
    nvm_root = validate_directory("NVM directory", args.nvm_dir)
    fdm_root = validate_directory("FDM directory", args.fdm_dir)
    program_path = args.program or (
        "./build/dense_attention_devptr.so"
        if args.cuda
        else "./build/dense_attention.so"
    )

    batch_size = 1
    num_query_heads = 1
    num_kv_heads = 1
    head_dim = config.model.head_dim
    token_count = config.instcsd.softmax_anchor_tokens
    dtype = np.dtype(config.model.dtype)
    scale = np.float32(1.0 / math.sqrt(head_dim))
    rng = np.random.default_rng(20260901)
    query = rng.normal(size=(batch_size, num_query_heads, head_dim)).astype(dtype)
    keys = rng.normal(
        size=(token_count, batch_size, num_kv_heads, head_dim)
    ).astype(dtype)
    values = rng.normal(
        size=(token_count, batch_size, num_kv_heads, head_dim)
    ).astype(dtype)
    expected = attention_reference(query, keys, values, scale)

    layout = KvCacheLayout(
        KvLayoutConfig(
            num_layers=1,
            max_seq_len=token_count,
            batch_size=batch_size,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            dtype=dtype,
        )
    )
    staging_bytes = token_count * layout.token_stride
    query_bytes = align_up(query.nbytes, 512)
    state_bytes = align_up(
        batch_size * num_query_heads * (head_dim + 2) * np.dtype(np.float32).itemsize,
        512,
    )
    modeled_runtime = runtime_model.estimate(
        batch_size=batch_size,
        num_query_heads=num_query_heads,
        head_dim=head_dim,
        token_count=token_count,
    )

    print(
        f"[table1-cemu] config={args.config}, target={'cuda' if args.cuda else 'cpu'}, "
        f"dtype={dtype}"
    )
    print(
        f"[table1-cemu] workload: batch={batch_size}, heads={num_query_heads}, "
        f"tokens={token_count}, head_dim={head_dim}, token_stride={layout.token_stride}"
    )
    print(
        f"[table1-cemu] modeled runtime: QK={modeled_runtime.qk_ns} ns, "
        f"Softmax={modeled_runtime.softmax_ns} ns, AV={modeled_runtime.av_ns} ns, "
        f"total={modeled_runtime.total_ns} ns"
    )

    with tempfile.TemporaryDirectory(
        prefix="table1-cemu-nvm-", dir=nvm_root
    ) as nvm_directory, tempfile.TemporaryDirectory(
        prefix="table1-cemu-fdm-", dir=fdm_root
    ) as fdm_directory:
        nvm_path = Path(nvm_directory)
        fdm_path = Path(fdm_directory)
        k_cache_path = nvm_path / "k_cache"
        v_cache_path = nvm_path / "v_cache"

        with KvCacheStore(
            layout,
            k_cache_path,
            v_cache_path,
            replace_existing=True,
        ) as store:
            store.write_tokens(0, 0, keys, values)
            store.flush()

        buffers = AttentionBufferConfig(
            query_bytes=query_bytes,
            staging_bytes=staging_bytes,
            state_bytes=state_bytes,
            output_bytes=query_bytes,
            query_path=fdm_path / "attention_query",
            k_staging_path=fdm_path / "k_staging_0",
            v_staging_path=fdm_path / "v_staging_0",
            state_path=fdm_path / "attention_state",
            output_path=fdm_path / "attention_output",
        )
        device = CemuAttentionDevice(
            layout=layout,
            buffers=buffers,
            program_name=(
                "table1_dense_attention_cuda"
                if args.cuda
                else "table1_dense_attention_cpu"
            ),
            program_path=program_path,
            function_name="dense_attention",
            k_cache_path=k_cache_path,
            v_cache_path=v_cache_path,
            control_path=args.control,
            namespace_path=args.namespace,
            cuda_target=args.cuda,
            runtime_model=runtime_model,
            replace_program=True,
            replace_staging_files=True,
            logger=print,
        )
        measured_wall_times = []
        with device:
            total_iterations = args.warmup_iterations + args.iterations
            for iteration in range(total_iterations):
                is_warmup = iteration < args.warmup_iterations
                phase = "warmup" if is_warmup else "measured"
                phase_index = (
                    iteration + 1
                    if is_warmup
                    else iteration - args.warmup_iterations + 1
                )
                phase_total = (
                    args.warmup_iterations if is_warmup else args.iterations
                )
                wall_start_ns = time.perf_counter_ns()
                output = device.run_decode(query, layer=0, valid_tokens=token_count)
                guest_wall_ns = time.perf_counter_ns() - wall_start_ns
                np.testing.assert_allclose(
                    output,
                    expected,
                    rtol=(3e-3 if dtype == np.dtype(np.float16) else 2e-5),
                    atol=(3e-3 if dtype == np.dtype(np.float16) else 2e-5),
                )
                if not is_warmup:
                    measured_wall_times.append(guest_wall_ns)
                print(
                    f"[table1-cemu] {phase} iteration={phase_index}/{phase_total}, "
                    f"Guest wall={guest_wall_ns} ns"
                )

    tolerance = 3e-3 if dtype == np.dtype(np.float16) else 2e-5
    np.testing.assert_allclose(output, expected, rtol=tolerance, atol=tolerance)
    print(f"[table1-cemu] query[:4]={query[0, 0, :4].tolist()}")
    print(f"[table1-cemu] output[:4]={output[0, 0, :4].tolist()}")
    print(f"[table1-cemu] reference[:4]={expected[0, 0, :4].tolist()}")
    measured_wall_array = np.asarray(measured_wall_times, dtype=np.int64)
    print(
        "[table1-cemu] measured Guest wall: "
        f"min={int(np.min(measured_wall_array))} ns, "
        f"median={int(np.median(measured_wall_array))} ns, "
        f"mean={int(np.mean(measured_wall_array))} ns, "
        f"max={int(np.max(measured_wall_array))} ns"
    )
    print(
        "[table1-cemu] Guest wall includes NVM->FDM staging, execute command, "
        "and output readback"
    )
    print("[table1-cemu] PASS")


if __name__ == "__main__":
    main()
