#!/usr/bin/env python3

import argparse
import json
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
    FlexGenDecodeRunner,
    FlexGenGenerationRunner,
    FlexGenLlamaConfig,
    FlexGenPrefillRunner,
    FlexGenTorchAttentionBackend,
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
    parser.add_argument("--decode-steps", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--hidden-size", type=int, default=16)
    parser.add_argument("--num-attention-heads", type=int, default=4)
    parser.add_argument("--num-key-value-heads", type=int, default=2)
    parser.add_argument("--staging-tokens", type=int, default=0)
    args = parser.parse_args()
    if args.decode_steps <= 0:
        parser.error("--decode-steps must be positive")
    if args.num_layers <= 0:
        parser.error("--num-layers must be positive")
    if args.hidden_size <= 0:
        parser.error("--hidden-size must be positive")
    if args.num_attention_heads <= 0:
        parser.error("--num-attention-heads must be positive")
    if args.num_key_value_heads <= 0:
        parser.error("--num-key-value-heads must be positive")
    if args.hidden_size % args.num_attention_heads != 0:
        parser.error("--num-attention-heads must divide --hidden-size")
    if args.num_attention_heads % args.num_key_value_heads != 0:
        parser.error(
            "--num-key-value-heads must divide --num-attention-heads"
        )
    if args.hidden_size // args.num_attention_heads % 2 != 0:
        parser.error("the configured head dimension must be even for RoPE")
    if args.staging_tokens < 0:
        parser.error("--staging-tokens must be non-negative")
    return args


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


def create_test_model(
    directory,
    num_layers,
    hidden_size,
    num_attention_heads,
    num_key_value_heads,
):
    config_values = {
        "num_hidden_layers": num_layers,
        "hidden_size": hidden_size,
        "num_attention_heads": num_attention_heads,
        "num_key_value_heads": num_key_value_heads,
        "intermediate_size": hidden_size * 3 // 2,
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
        for projection in ("q_proj", "o_proj"):
            random_weight(
                prefix + f"self_attn.{projection}.weight",
                (config.hidden_size, config.hidden_size),
            )
        kv_hidden_size = config.num_key_value_heads * config.head_dim
        for projection in ("k_proj", "v_proj"):
            random_weight(
                prefix + f"self_attn.{projection}.weight",
                (kv_hidden_size, config.hidden_size),
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

    log("[pipeline] step 1/7: prepare deterministic configurable Llama model")
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
        config = create_test_model(
            weight_path,
            num_layers=args.num_layers,
            hidden_size=args.hidden_size,
            num_attention_heads=args.num_attention_heads,
            num_key_value_heads=args.num_key_value_heads,
        )
        prompt_tokens = 5
        max_seq_len = prompt_tokens + args.decode_steps
        selected_layer = config.num_hidden_layers - 1
        token_ids = torch.tensor(
            [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]],
            dtype=torch.long,
        )
        layout = KvCacheLayout(
            KvLayoutConfig(
                num_layers=config.num_hidden_layers,
                max_seq_len=max_seq_len,
                batch_size=token_ids.shape[0],
                num_kv_heads=config.num_key_value_heads,
                head_dim=config.head_dim,
                dtype=np.float32,
            )
        )
        staging_tokens = args.staging_tokens or max_seq_len
        staging_bytes = staging_tokens * layout.token_stride
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
            state_bytes=align_up(
                layout.config.batch_size
                * config.num_attention_heads
                * (config.head_dim + 2)
                * np.dtype(np.float32).itemsize,
                512,
            ),
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
        log(
            f"[pipeline] model layers={config.num_hidden_layers}, "
            f"hidden={config.hidden_size}, query_heads={config.num_attention_heads}, "
            f"kv_heads={config.num_key_value_heads}, "
            f"groups={config.num_key_value_groups}"
        )
        log(
            f"[pipeline] FDM staging capacity={staging_tokens} tokens, "
            f"bytes={staging_bytes}, token_stride={layout.token_stride}"
        )
        weight_loader = FlexGenWeightLoader(
            config,
            weight_path,
            device="cuda:0",
        )
        prefill_runner = FlexGenPrefillRunner(
            config=config,
            weight_loader=weight_loader,
            kv_writer=backend,
            logger=log,
        )

        log(
            "[pipeline] step 2/7: run full Prefill on Guest GPU "
            f"({torch.cuda.get_device_name(0)})"
        )
        with backend:
            prefill_result = prefill_runner.run(token_ids, collect_kv_cache=True)
            torch.cuda.synchronize()
            backend.flush()

            log("[pipeline] step 3/7: verify Prefill KV in CEMU NVM")
            with KvCacheStore(layout, k_cache_path, v_cache_path) as store:
                stored_keys, stored_values = store.read_tokens(
                    selected_layer,
                    0,
                    prompt_tokens,
                )
            expected_keys = (
                prefill_result.kv_cache[selected_layer][0]
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
                prefill_result.kv_cache[selected_layer][1]
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

            log(
                "[pipeline] step 4/7: run pure PyTorch autoregressive Decode reference"
            )
            reference_backend = FlexGenTorchAttentionBackend(
                config,
                prefill_result.kv_cache,
                logger=log,
            )
            reference_runner = FlexGenDecodeRunner(
                config,
                weight_loader,
                reference_backend,
                logger=log,
            )
            reference_generation = FlexGenGenerationRunner(
                reference_runner,
                logger=log,
            ).run(
                prefill_result.next_token_ids,
                start_position=prompt_tokens,
                decode_steps=args.decode_steps,
                collect_layer_outputs=True,
            )
            torch.cuda.synchronize()
            log(
                f"[pipeline] Prefill next_tokens="
                f"{prefill_result.next_token_ids.detach().cpu().reshape(-1).tolist()}, "
                f"reference sequence="
                f"{reference_generation.token_ids.detach().cpu().tolist()}"
            )

            tokens_per_chunk = layout.tokens_per_chunk(
                staging_bytes,
                staging_bytes,
            )
            for decode_step in range(args.decode_steps):
                valid_tokens = prompt_tokens + decode_step + 1
                chunk_count = layout.chunk_count(valid_tokens, tokens_per_chunk)
                log(
                    f"[pipeline] Decode step={decode_step}, "
                    f"position={prompt_tokens + decode_step}, "
                    f"valid_tokens={valid_tokens}, chunks={chunk_count}, "
                    f"tokens_per_chunk={tokens_per_chunk}"
                )

            log(
                "[pipeline] step 5/7: autoregressive tokens -> QKV/RoPE -> "
                "NVM -> FDM -> 5 MRS -> CSD CUDA Attention"
            )
            cemu_decode_runner = FlexGenDecodeRunner(
                config,
                weight_loader,
                backend,
                logger=log,
            )
            with attention_device:
                cemu_generation = FlexGenGenerationRunner(
                    cemu_decode_runner,
                    logger=log,
                ).run(
                    prefill_result.next_token_ids,
                    start_position=prompt_tokens,
                    decode_steps=args.decode_steps,
                    collect_layer_outputs=True,
                )
                torch.cuda.synchronize()
                if attention_device.memory_range_count != 5:
                    raise AssertionError("CEMU Attention must create exactly five MRS")

            log("[pipeline] step 6/7: verify all Decode K/V appends in CEMU NVM")
            backend.flush()
            with KvCacheStore(layout, k_cache_path, v_cache_path) as store:
                appended_keys, appended_values = store.read_tokens(
                    selected_layer,
                    prompt_tokens,
                    args.decode_steps,
                )
            expected_appended_keys = np.concatenate(
                [
                    step.decode_result.layer_outputs[selected_layer]
                    .key.detach()
                    .cpu()
                    .reshape(
                        1,
                        layout.config.batch_size,
                        layout.config.num_kv_heads,
                        layout.config.head_dim,
                    )
                    .numpy()
                    for step in cemu_generation.steps
                ],
                axis=0,
            )
            expected_appended_values = np.concatenate(
                [
                    step.decode_result.layer_outputs[selected_layer]
                    .value.detach()
                    .cpu()
                    .reshape(
                        1,
                        layout.config.batch_size,
                        layout.config.num_kv_heads,
                        layout.config.head_dim,
                    )
                    .numpy()
                    for step in cemu_generation.steps
                ],
                axis=0,
            )
            np.testing.assert_array_equal(appended_keys, expected_appended_keys)
            np.testing.assert_array_equal(appended_values, expected_appended_values)
            log(
                f"[pipeline] NVM append verified layer={selected_layer}, "
                f"tokens=[{prompt_tokens}, {max_seq_len}), "
                f"K[:4]={appended_keys.reshape(-1)[:4].tolist()}"
            )

            log(
                "[pipeline] step 7/7: compare every CEMU Decode step with "
                "PyTorch reference"
            )
            torch.testing.assert_close(
                cemu_generation.token_ids,
                reference_generation.token_ids,
            )
            for cemu_step, reference_step in zip(
                cemu_generation.steps,
                reference_generation.steps,
            ):
                torch.testing.assert_close(
                    cemu_step.input_token_ids,
                    reference_step.input_token_ids,
                )
                cemu_result = cemu_step.decode_result
                reference_result = reference_step.decode_result
                for layer, (cemu_layer, reference_layer) in enumerate(
                    zip(cemu_result.layer_outputs, reference_result.layer_outputs)
                ):
                    torch.testing.assert_close(
                        cemu_layer.query,
                        reference_layer.query,
                        rtol=3e-4,
                        atol=3e-4,
                    )
                    torch.testing.assert_close(
                        cemu_layer.key,
                        reference_layer.key,
                        rtol=3e-4,
                        atol=3e-4,
                    )
                    torch.testing.assert_close(
                        cemu_layer.value,
                        reference_layer.value,
                        rtol=3e-4,
                        atol=3e-4,
                    )
                    torch.testing.assert_close(
                        cemu_layer.attention_output,
                        reference_layer.attention_output,
                        rtol=3e-4,
                        atol=3e-4,
                    )
                    log(
                        f"[pipeline] step={cemu_step.step}, layer={layer} "
                        "Q/K/V and Attention match"
                    )
                torch.testing.assert_close(
                    cemu_result.hidden_states,
                    reference_result.hidden_states,
                    rtol=3e-4,
                    atol=3e-4,
                )
                torch.testing.assert_close(
                    cemu_result.logits,
                    reference_result.logits,
                    rtol=3e-4,
                    atol=3e-4,
                )
                torch.testing.assert_close(
                    cemu_result.next_token_ids,
                    reference_result.next_token_ids,
                )
                log(
                    f"[pipeline] step={cemu_step.step} logits match, "
                    f"next_tokens={cemu_result.next_token_ids.detach().cpu().reshape(-1).tolist()}"
                )

    log(
        f"[pipeline] PASS: Guest GPU Prefill -> {args.decode_steps} autoregressive "
        "Decode steps -> CEMU Attention -> Wo/MLP/LM head; token_sequence="
        f"{cemu_generation.token_ids.detach().cpu().tolist()}"
    )


if __name__ == "__main__":
    main()
