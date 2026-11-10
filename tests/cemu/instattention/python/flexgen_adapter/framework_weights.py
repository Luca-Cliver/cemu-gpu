import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .framework_import import load_framework


def config_from_json(path, name=None):
    load_framework()
    from flexgen.opt_config import OptConfig

    values = json.loads(Path(path).read_text())
    hidden = int(values["hidden_size"])
    dtype = values.get("torch_dtype", values.get("dtype", "float16"))
    if (values.get("model_type", "opt") != "opt" or
            values.get("word_embed_proj_dim", hidden) != hidden or
            values.get("activation_function", "relu") != "relu" or
            not values.get("do_layer_norm_before", True) or
            not values.get("tie_word_embeddings", True) or
            not values.get("enable_bias", True) or
            not values.get("layer_norm_elementwise_affine", True) or
            values.get("_remove_final_layer_norm", False) or
            dtype not in ("float16", "float32")):
        raise ValueError("supported OPT: pre-LN, ReLU, tied embeddings, no projection, FP16/FP32")
    config = OptConfig(
        name=name or Path(path).resolve().parent.name.lower(),
        num_hidden_layers=int(values["num_hidden_layers"]),
        max_seq_len=int(values["max_position_embeddings"]), hidden_size=hidden,
        n_head=int(values["num_attention_heads"]), input_dim=hidden,
        ffn_embed_dim=int(values["ffn_dim"]), vocab_size=int(values["vocab_size"]),
        layer_norm_eps=float(values.get("layer_norm_eps", 1e-5)),
        pad=int(values.get("pad_token_id", 1)), pad_token_id=int(values.get("pad_token_id", 1)),
        dtype=np.dtype(dtype).type)
    if (min(config.input_dim, config.n_head, config.ffn_embed_dim, config.vocab_size,
            config.num_hidden_layers, config.max_seq_len) <= 0 or hidden % config.n_head or
            not 0 <= config.pad_token_id < config.vocab_size or config.layer_norm_eps <= 0 or
            Path(config.name).name != config.name or "_DUMMY_" in config.name):
        raise ValueError("invalid OPT geometry or model name")
    return config


def weight_shapes(config):
    hidden = config.input_dim
    shapes = {
        "decoder.embed_tokens.weight": (config.vocab_size, hidden),
        "decoder.embed_positions.weight": (config.max_seq_len + 2, hidden),
        "decoder.layer_norm.weight": (hidden,), "decoder.layer_norm.bias": (hidden,),
    }
    for layer in range(config.num_hidden_layers):
        prefix = f"decoder.layers.{layer}."
        for projection in ("q_proj", "k_proj", "v_proj", "out_proj"):
            shapes[prefix + f"self_attn.{projection}.weight"] = (hidden, hidden)
            shapes[prefix + f"self_attn.{projection}.bias"] = (hidden,)
        for norm in ("self_attn_layer_norm", "final_layer_norm"):
            shapes[prefix + norm + ".weight"] = (hidden,)
            shapes[prefix + norm + ".bias"] = (hidden,)
        shapes[prefix + "fc1.weight"] = (config.ffn_embed_dim, hidden)
        shapes[prefix + "fc1.bias"] = (config.ffn_embed_dim,)
        shapes[prefix + "fc2.weight"] = (hidden, config.ffn_embed_dim)
        shapes[prefix + "fc2.bias"] = (hidden,)
    return shapes


def verify_weights(config, directory):
    root = Path(directory) / f"{config.name}-np"
    for name, shape in weight_shapes(config).items():
        array = np.load(root / name, mmap_mode="r", allow_pickle=False)
        if array.shape != shape or array.dtype != np.dtype(config.dtype):
            raise ValueError(f"invalid converted tensor {name}: {array.shape}/{array.dtype}, expected {shape}/{config.dtype}")
    return root


def convert_checkpoint(model_directory, output_directory):
    source = Path(model_directory).resolve()
    config = config_from_json(source / "config.json")
    target = Path(output_directory).resolve() / f"{config.name}-np"
    if target.exists():
        raise FileExistsError(f"refusing to overwrite converted weights: {target}")
    index = next((source / name for name in ("model.safetensors.index.json", "pytorch_model.bin.index.json")
                  if (source / name).is_file()), None)
    if index is not None:
        shards = sorted(set(json.loads(index.read_text())["weight_map"].values()))
    else:
        shards = [next((name for name in ("model.safetensors", "pytorch_model.bin")
                        if (source / name).is_file()), "")]
    if not shards or any(not name or not (source / name).resolve().is_relative_to(source) or
                         not (source / name).is_file() for name in shards):
        raise ValueError("local checkpoint shards are missing or outside the model directory")
    expected = weight_shapes(config)
    written = set()
    target.mkdir(parents=True)
    for shard in shards:
        print(f"[flexgen-weights] converting {shard}", flush=True)
        if shard.endswith(".safetensors"):
            from safetensors import safe_open
            with safe_open(source / shard, framework="pt", device="cpu") as tensors:
                _write_tensors(((key, tensors.get_tensor(key)) for key in tensors.keys()),
                               target, expected, written, config.dtype)
        else:
            tensors = torch.load(source / shard, map_location="cpu", weights_only=True)
            _write_tensors(tensors.items(), target, expected, written, config.dtype)
            del tensors
    if written != set(expected):
        raise ValueError(f"incomplete conversion; missing {sorted(set(expected) - written)}")
    verify_weights(config, output_directory)
    (target / "config.json").write_text((source / "config.json").read_text())
    (target / "source.json").write_text(json.dumps({
        "model_name": config.name, "source": str(source), "shards": shards,
        "tensors": len(written), "dtype": np.dtype(config.dtype).name,
    }, indent=2) + "\n")
    return target


def _write_tensors(tensors, target, expected, written, dtype):
    for name, tensor in tensors:
        name = name.removeprefix("model.").replace("decoder.final_layer_norm.", "decoder.layer_norm.")
        if name not in expected:
            continue
        if name in written or tuple(tensor.shape) != expected[name] or not tensor.is_floating_point():
            raise ValueError(f"invalid or duplicate checkpoint tensor: {name}")
        array = tensor.detach().cpu().to(torch.float32 if dtype == np.float32 else torch.float16).numpy()
        with (target / name).open("wb") as stream:
            np.save(stream, array, allow_pickle=False)
        written.add(name)


def main():
    parser = argparse.ArgumentParser(description="Convert local HF OPT weights for vendored FlexGen (no download)")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    print(convert_checkpoint(args.model_dir, args.output_dir))


if __name__ == "__main__":
    main()
