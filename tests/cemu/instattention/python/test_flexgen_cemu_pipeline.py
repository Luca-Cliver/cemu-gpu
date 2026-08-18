#!/usr/bin/env python3

import argparse
import json
import math
import sys
import tempfile
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
    FlexGenLlamaConfig,
    FlexGenPrefillRunner,
    FlexGenWeightLoader,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run GPU FlexGen Prefill followed by CEMU Decode Attention"
    )
    parser.add_argument("--control", default="/dev/nvme0c3")
    parser.add_argument("--namespace", default="/dev/ng0n3")
    parser.add_argument("--nvm-dir", default="/mnt/nvme0")
    parser.add_argument("--fdm-dir", default="/mnt/fdm0")
    parser.add_argument("--program", default="./build/dense_attention_devptr.so")
    return parser.parse_args()


def log(message):
    print(message, flush=True)


def validate_directory(name, path):
    directory = Path(path)
    if not directory.is_dir():
        raise FileNotFoundError(f"{name} does not exist: {directory}")
    return directory


def write_weight(directory, filename, values):
    with (directory / filename).open("wb") as weight_file:
        np.save(weight_file, values)


def create_test_model(directory):
    config_values = {
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
    config_path = directory / "config.json"
    config_path.write_text(json.dumps(config_values), encoding="utf-8")
    config = FlexGenLlamaConfig.from_json(config_path)
    rng = np.random.default_rng(20260818)

    def random_weight(filename, shape):
        values = rng.normal(scale=0.05, size=shape).astype(np.float32)
        write_weight(directory, filename, values)

    random_weight(
        "embed_tokens.weight",
        (config.vocab_size, config.hidden_size),
    )
    random_weight("norm.weight", (config.hidden_size,))
    for layer in range(config.num_hidden_layers):
        prefix = f"layers.{layer}."
        for projection in ("q_proj", "k_proj", "v_proj", "o_proj"):
            random_weight(
                prefix + f"self_attn.{projection}.weight",
                (config.hidden_size, config.hidden_size),
            )
        random_weight(prefix + "input_layernorm.weight", (config.hidden_size,))
        random_weight(
            prefix + "post_attention_layernorm.weight",
            (config.hidden_size,),
        )
        random_weight(
            prefix + "mlp.gate_proj.weight",
            (config.intermediate_size, config.hidden_size),
        )
        random_weight(
            prefix + "mlp.up_proj.weight",
            (config.intermediate_size, config.hidden_size),
        )
        random_weight(
            prefix + "mlp.down_proj.weight",
            (config.hidden_size, config.intermediate_size),
        )
    return config


def attention_reference(query, keys, values, num_kv_heads):
    batch_size, num_query_heads, head_dim = query.shape
    token_count = keys.shape[0]
    scale = np.float32(1.0 / math.sqrt(head_dim))
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


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("PyTorch cannot access the Guest GPU")

    nvm_root = validate_directory("NVM directory", args.nvm_dir)
    fdm_root = validate_directory("FDM directory", args.fdm_dir)
    program_file = Path(args.program)
    if not program_file.is_file():
        raise FileNotFoundError(f"CEMU Attention program does not exist: {program_file}")
    program_reference = args.program
    if not program_file.is_absolute() and not program_reference.startswith("."):
        program_reference = f"./{program_reference}"

    log("[pipeline] step 1/6: prepare deterministic two-layer Llama model")
    log(
        f"[pipeline] CSD operator guest_file={program_file}, "
        f"host_reference={program_reference}"
    )
    with tempfile.TemporaryDirectory(
        prefix="flexgen-cemu-weights-"
    ) as weight_directory, tempfile.TemporaryDirectory(
        prefix="flexgen-cemu-nvm-", dir=nvm_root
    ) as nvm_directory, tempfile.TemporaryDirectory(
        prefix="flexgen-cemu-fdm-", dir=fdm_root
    ) as fdm_directory:
        weight_path = Path(weight_directory)
        nvm_path = Path(nvm_directory)
        fdm_path = Path(fdm_directory)
        config = create_test_model(weight_path)
        prompt_tokens = 5
        selected_layer = config.num_hidden_layers - 1
        token_ids = torch.tensor(
            [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]],
            dtype=torch.long,
        )
        layout = KvCacheLayout(
            KvLayoutConfig(
                num_layers=config.num_hidden_layers,
                max_seq_len=prompt_tokens,
                batch_size=token_ids.shape[0],
                num_kv_heads=config.num_key_value_heads,
                head_dim=config.head_dim,
                dtype=np.float32,
            )
        )
        staging_bytes = 2 * layout.token_stride
        query_shape = (
            layout.config.batch_size,
            config.num_attention_heads,
            config.head_dim,
        )
        k_cache_path = nvm_path / "k_cache"
        v_cache_path = nvm_path / "v_cache"
        buffers = AttentionBufferConfig(
            query_bytes=align_up(np.prod(query_shape) * np.dtype(np.float32).itemsize, 512),
            staging_bytes=staging_bytes,
            state_bytes=512,
            output_bytes=align_up(np.prod(query_shape) * np.dtype(np.float32).itemsize, 512),
            query_path=fdm_path / "attention_query",
            k_staging_path=fdm_path / "k_staging_0",
            v_staging_path=fdm_path / "v_staging_0",
            state_path=fdm_path / "attention_state",
            output_path=fdm_path / "attention_output",
        )
        attention_device = CemuAttentionDevice(
            layout=layout,
            buffers=buffers,
            program_name="flexgen_cemu_pipeline_cuda",
            program_path=program_reference,
            function_name="dense_attention",
            k_cache_path=k_cache_path,
            v_cache_path=v_cache_path,
            control_path=args.control,
            namespace_path=args.namespace,
            cuda_target=True,
            replace_program=True,
            replace_staging_files=True,
            logger=log,
        )
        backend = FlexGenAttentionBackend(
            layout=layout,
            k_cache_path=k_cache_path,
            v_cache_path=v_cache_path,
            attention_device=attention_device,
            replace_existing=True,
            logger=log,
        )
        runner = FlexGenPrefillRunner(
            config=config,
            weight_loader=FlexGenWeightLoader(
                config,
                weight_path,
                device="cuda:0",
            ),
            kv_writer=backend,
            logger=log,
        )

        log(
            "[pipeline] step 2/6: run full Prefill on Guest GPU "
            f"({torch.cuda.get_device_name(0)})"
        )
        with backend:
            result = runner.run(token_ids, collect_kv_cache=True)
            torch.cuda.synchronize()
            backend.flush()

            log("[pipeline] step 3/6: verify GPU-produced KV in CEMU NVM")
            with KvCacheStore(layout, k_cache_path, v_cache_path) as store:
                stored_keys, stored_values = store.read_tokens(
                    selected_layer,
                    0,
                    prompt_tokens,
                )
            expected_keys = (
                result.kv_cache[selected_layer][0]
                .detach()
                .cpu()
                .reshape(
                    prompt_tokens,
                    layout.config.batch_size,
                    layout.config.num_kv_heads,
                    layout.config.head_dim,
                )
                .numpy()
            )
            expected_values = (
                result.kv_cache[selected_layer][1]
                .detach()
                .cpu()
                .reshape(expected_keys.shape)
                .numpy()
            )
            np.testing.assert_array_equal(stored_keys, expected_keys)
            np.testing.assert_array_equal(stored_values, expected_values)
            log(
                f"[pipeline] NVM verified layer={selected_layer}, "
                f"K/V={stored_keys.shape}, K[:4]={stored_keys.reshape(-1)[:4].tolist()}"
            )

            log("[pipeline] step 4/6: prepare one synthetic Decode query")
            query = np.random.default_rng(20260819).normal(size=query_shape).astype(
                np.float32
            )
            expected_output = attention_reference(
                query,
                stored_keys,
                stored_values,
                layout.config.num_kv_heads,
            )
            chunk_count = layout.chunk_count(
                prompt_tokens,
                layout.tokens_per_chunk(staging_bytes, staging_bytes),
            )
            log(
                f"[pipeline] query={query.shape}, chunks={chunk_count}, "
                f"tokens_per_chunk={layout.tokens_per_chunk(staging_bytes, staging_bytes)}"
            )

            log("[pipeline] step 5/6: NVM -> FDM -> 5 MRS -> CSD CUDA Attention")
            with attention_device:
                output = backend.decode(
                    layer=selected_layer,
                    query=query,
                    valid_tokens=prompt_tokens,
                )
                if attention_device.memory_range_count != 5:
                    raise AssertionError("CEMU Attention must create exactly five MRS")

            log("[pipeline] step 6/6: compare CSD output with CPU reference")
            np.testing.assert_allclose(output, expected_output, rtol=2e-5, atol=2e-5)
            log(f"[pipeline] output[0,0]={output[0, 0].tolist()}")
            log(f"[pipeline] reference[0,0]={expected_output[0, 0].tolist()}")

    log("[pipeline] PASS: Guest GPU Prefill -> CEMU NVM/FDM -> CSD GPU Attention")


if __name__ == "__main__":
    main()
