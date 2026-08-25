import argparse
import time
from pathlib import Path

import torch

from flexgen_runtime import FlexGenHfCheckpointLoader, FlexGenLlamaConfig


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--all-layers", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    model_directory = args.model_dir.resolve()
    config = FlexGenLlamaConfig.from_json(model_directory / "config.json")
    device = torch.device("cuda:0" if args.cuda else "cpu")
    if args.cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")

    print(
        "[tinyllama-checkpoint] "
        f"model={model_directory}, device={device}, dtype={config.dtype}"
    )
    print(
        "[tinyllama-checkpoint] "
        f"layers={config.num_hidden_layers}, hidden={config.hidden_size}, "
        f"q_heads={config.num_attention_heads}, kv_heads={config.num_key_value_heads}, "
        f"head_dim={config.head_dim}, intermediate={config.intermediate_size}, "
        f"vocab={config.vocab_size}"
    )

    start = time.perf_counter()
    with FlexGenHfCheckpointLoader(
        config,
        model_directory,
        device=device,
        cache_layers=True,
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
            embedding.reshape(-1)[:8].float().sum().cpu()
            + first_layer.attention.query.reshape(-1)[:8].float().sum().cpu()
            + first_layer.mlp.gate.reshape(-1)[:8].float().sum().cpu()
        )
        elapsed = time.perf_counter() - start
        print(
            "[tinyllama-checkpoint] "
            f"embedding={tuple(embedding.shape)}, "
            f"final_norm={tuple(final_norm.shape)}, lm_head={tuple(lm_head.shape)}"
        )
        print(
            "[tinyllama-checkpoint] "
            f"layer0.q={tuple(first_layer.attention.query.shape)}, "
            f"layer0.k={tuple(first_layer.attention.key.shape)}, "
            f"layer0.gate={tuple(first_layer.mlp.gate.shape)}, "
            f"cached_layers={loader.cached_layer_count}, checksum={checksum:.6f}"
        )
        print(
            "[tinyllama-checkpoint] "
            f"loaded_layers={layer_count}, elapsed={elapsed:.6f}s"
        )


if __name__ == "__main__":
    main()
