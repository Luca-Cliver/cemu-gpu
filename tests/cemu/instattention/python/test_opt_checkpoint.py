import argparse
import time
from pathlib import Path

import torch

from opt_runtime import OptCheckpointLoader, OptConfig


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--all-layers", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    model_directory = args.model_dir.resolve()
    config = OptConfig.from_json(model_directory / "config.json")
    device = torch.device("cuda:0" if args.cuda else "cpu")
    if args.cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")

    print(
        "[opt-checkpoint] "
        f"model={model_directory}, device={device}, dtype={config.dtype}"
    )
    print(
        "[opt-checkpoint] "
        f"layers={config.num_hidden_layers}, hidden={config.hidden_size}, "
        f"heads={config.num_attention_heads}, head_dim={config.head_dim}, "
        f"ffn={config.ffn_dim}, positions={config.max_position_embeddings}, "
        f"vocab={config.vocab_size}"
    )

    start = time.perf_counter()
    with OptCheckpointLoader(
        config,
        model_directory,
        device=device,
        cache_layers=True,
        max_cached_shards=1,
    ) as loader:
        embedding = loader.load_embedding()
        final_norm = loader.load_final_norm()
        lm_head = loader.load_lm_head()
        layer_count = config.num_hidden_layers if args.all_layers else 1
        first_layer = None
        for layer in range(layer_count):
            layer_weights = loader.load_layer(layer)
            if first_layer is None:
                first_layer = layer_weights
        if loader.load_layer(0) is not first_layer:
            raise AssertionError("layer cache did not return the original layer object")

        checksum = float(
            embedding.token.reshape(-1)[:8].float().sum().cpu()
            + embedding.position.reshape(-1)[:8].float().sum().cpu()
            + first_layer.attention.query.reshape(-1)[:8].float().sum().cpu()
            + first_layer.attention.query_bias[:8].float().sum().cpu()
            + first_layer.mlp.input.reshape(-1)[:8].float().sum().cpu()
        )
        elapsed = time.perf_counter() - start
        checkpoint_size = sum(path.stat().st_size for path in loader.checkpoint_files)
        print(
            "[opt-checkpoint] "
            f"checkpoint_files={len(loader.checkpoint_files)}, "
            f"checkpoint_bytes={checkpoint_size}"
        )
        print(
            "[opt-checkpoint] "
            f"token_embedding={tuple(embedding.token.shape)}, "
            f"position_embedding={tuple(embedding.position.shape)}, "
            f"final_norm={tuple(final_norm.weight.shape)}, "
            f"lm_head={tuple(lm_head.shape)}"
        )
        print(
            "[opt-checkpoint] "
            f"layer0.q={tuple(first_layer.attention.query.shape)}, "
            f"layer0.q_bias={tuple(first_layer.attention.query_bias.shape)}, "
            f"layer0.fc1={tuple(first_layer.mlp.input.shape)}, "
            f"cached_layers={loader.cached_layer_count}, "
            f"cached_shards={loader.cached_shard_count}, checksum={checksum:.6f}"
        )
        print(
            "[opt-checkpoint] "
            f"loaded_layers={layer_count}, elapsed={elapsed:.6f}s"
        )


if __name__ == "__main__":
    main()
