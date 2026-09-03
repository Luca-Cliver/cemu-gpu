#!/usr/bin/env python3

import argparse
import math
import sys
import tempfile
from pathlib import Path

import numpy as np


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "build-guest"))
sys.path.insert(0, str(PROJECT_DIR / "python"))

from cemu_flexgen import (
    AttentionBufferConfig,
    CemuAttentionDevice,
    DenseAttentionMetadata,
    KvCacheLayout,
    KvCacheStore,
    KvLayoutConfig,
    align_up,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Test serial multi-chunk dense Attention through CEMU"
    )
    parser.add_argument("--control", default="/dev/nvme0c3")
    parser.add_argument("--namespace", default="/dev/ng0n3")
    parser.add_argument("--nvm-dir", default="/mnt/nvme0")
    parser.add_argument("--fdm-dir", default="/mnt/fdm0")
    parser.add_argument("--program")
    parser.add_argument("--cuda", action="store_true")
    return parser.parse_args()


def attention_reference(query, keys, values, num_kv_heads, scale):
    batch_size, num_query_heads, head_dim = query.shape
    token_count = keys.shape[0]
    output = np.empty_like(query)

    for batch in range(batch_size):
        for query_head in range(num_query_heads):
            kv_head = query_head * num_kv_heads // num_query_heads
            scores = np.empty(token_count, dtype=np.float32)
            for token in range(token_count):
                scores[token] = (
                    np.dot(query[batch, query_head], keys[token, batch, kv_head])
                    * scale
                )
            probabilities = np.exp(scores - np.max(scores))
            probabilities /= np.sum(probabilities)
            output[batch, query_head] = np.sum(
                probabilities[:, np.newaxis] * values[:, batch, kv_head],
                axis=0,
            )
    return output


def validate_directory(name, path):
    directory = Path(path)
    if not directory.is_dir():
        raise FileNotFoundError(f"{name} does not exist: {directory}")
    return directory


def main():
    args = parse_args()
    nvm_root = validate_directory("NVM directory", args.nvm_dir or "/mnt/nvme0")
    fdm_root = validate_directory("FDM directory", args.fdm_dir or "/mnt/fdm0")
    program_path = args.program or (
        "./build/dense_attention_devptr.so"
        if args.cuda
        else "./build/dense_attention.so"
    )

    batch_size = 1
    num_query_heads = 4
    num_kv_heads = 2
    head_dim = 8
    token_count = 17
    rng = np.random.default_rng(20260814)
    query = rng.normal(size=(batch_size, num_query_heads, head_dim)).astype(
        np.float32
    )
    keys = rng.normal(
        size=(token_count, batch_size, num_kv_heads, head_dim)
    ).astype(np.float32)
    values = rng.normal(
        size=(token_count, batch_size, num_kv_heads, head_dim)
    ).astype(np.float32)

    layout = KvCacheLayout(
        KvLayoutConfig(
            num_layers=1,
            max_seq_len=token_count,
            batch_size=batch_size,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            dtype=np.float32,
        )
    )
    staging_bytes = 4096
    chunks = list(
        layout.iter_chunks(
            layer=0,
            valid_tokens=token_count,
            k_staging_bytes=staging_bytes,
            v_staging_bytes=staging_bytes,
        )
    )
    expected = attention_reference(
        query,
        keys,
        values,
        num_kv_heads,
        np.float32(1.0 / math.sqrt(head_dim)),
    )

    with tempfile.TemporaryDirectory(
        prefix="cemu-attention-nvm-", dir=nvm_root
    ) as nvm_directory, tempfile.TemporaryDirectory(
        prefix="cemu-attention-fdm-", dir=fdm_root
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
            query_bytes=align_up(query.nbytes, 512),
            staging_bytes=staging_bytes,
            state_bytes=512,
            output_bytes=align_up(query.nbytes, 512),
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
                "dense_attention_cuda_test" if args.cuda else "dense_attention_cpu_test"
            ),
            program_path=program_path,
            function_name="dense_attention",
            k_cache_path=k_cache_path,
            v_cache_path=v_cache_path,
            control_path=args.control,
            namespace_path=args.namespace,
            cuda_target=args.cuda,
            replace_program=True,
            replace_staging_files=True,
        )

        with device:
            for chunk_index, chunk in enumerate(chunks):
                device.stage_chunk(chunk)
                staged_keys, staged_values = device.staging.read_staged_tokens(
                    chunk.token_count
                )
                np.testing.assert_array_equal(
                    staged_keys,
                    keys[chunk.start_token : chunk.end_token],
                )
                np.testing.assert_array_equal(
                    staged_values,
                    values[chunk.start_token : chunk.end_token],
                )
                metadata = DenseAttentionMetadata.from_layout(
                    layout,
                    num_query_heads=query.shape[1],
                    token_count=chunk.token_count,
                    reset_state=chunk_index == 0,
                    finalize=chunk_index == len(chunks) - 1,
                )
                device.execute_staged_chunk(
                    query,
                    chunk,
                    metadata,
                    write_query=chunk_index == 0,
                )
            output = device.collect_output(metadata)
            assert device.memory_range_count == 5

        np.testing.assert_allclose(output, expected, rtol=2e-5, atol=2e-5)

        mode = "CUDA device-pointer" if args.cuda else "CPU host"
        print(f"[attention] mode={mode}, MRS=5")
        print(
            "[attention] NVM -> FDM: "
            f"tokens={token_count}, token_stride={layout.token_stride}, "
            f"chunks={len(chunks)}, staging_bytes={staging_bytes}"
        )
        print(f"[attention] query[0,0,:4]={query[0, 0, :4].tolist()}")
        print(f"[attention] output[0,0,:4]={output[0, 0, :4].tolist()}")
        print(f"[attention] reference[0,0,:4]={expected[0, 0, :4].tolist()}")
        print("[attention] stages=load(NVM->FDM) -> compute(CEMU) -> collect(output)")
        print("Serial multi-chunk dense Attention test passed")


if __name__ == "__main__":
    main()
