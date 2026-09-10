"""Hugging Face OPT model configuration."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


_TORCH_DTYPES = {
    "float16": torch.float16,
    "torch.float16": torch.float16,
    "half": torch.float16,
    "float32": torch.float32,
    "torch.float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "torch.bfloat16": torch.bfloat16,
}


@dataclass(frozen=True)
class OptConfig:
    name: str
    num_hidden_layers: int
    max_position_embeddings: int
    hidden_size: int
    num_attention_heads: int
    ffn_dim: int
    vocab_size: int
    word_embed_proj_dim: int
    pad_token_id: int = 1
    bos_token_id: int = 2
    eos_token_id: int = 2
    activation_function: str = "relu"
    layer_norm_epsilon: float = 1e-5
    do_layer_norm_before: bool = True
    tie_word_embeddings: bool = True
    dtype: torch.dtype = torch.float16
    position_offset: int = 2

    def __post_init__(self) -> None:
        for field_name in (
            "num_hidden_layers",
            "max_position_embeddings",
            "hidden_size",
            "num_attention_heads",
            "ffn_dim",
            "vocab_size",
            "word_embed_proj_dim",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{field_name} must be a positive integer")
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError("num_attention_heads must divide hidden_size")
        if self.word_embed_proj_dim != self.hidden_size:
            raise NotImplementedError(
                "OPT models with input/output projection are not supported"
            )
        if self.activation_function != "relu":
            raise NotImplementedError("only the OPT ReLU activation is supported")
        if self.do_layer_norm_before is not True:
            raise NotImplementedError("only pre-LayerNorm OPT models are supported")
        if self.dtype not in (torch.float16, torch.float32, torch.bfloat16):
            raise ValueError("unsupported model dtype")
        if self.layer_norm_epsilon <= 0:
            raise ValueError("layer_norm_epsilon must be positive")
        if self.position_offset < 0:
            raise ValueError("position_offset must be non-negative")

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

    @property
    def num_key_value_heads(self) -> int:
        return self.num_attention_heads

    @property
    def max_seq_len(self) -> int:
        return self.max_position_embeddings

    @classmethod
    def from_json(cls, path: Any):
        config_path = Path(path)
        with config_path.open("r", encoding="utf-8") as config_file:
            values = json.load(config_file)

        if values.get("model_type") != "opt":
            raise ValueError(f"configuration is not an OPT model: {config_path}")
        dtype_name = values.get("torch_dtype", "float16")
        if isinstance(dtype_name, torch.dtype):
            dtype = dtype_name
        else:
            try:
                dtype = _TORCH_DTYPES[str(dtype_name).lower()]
            except KeyError as error:
                raise ValueError(f"unsupported torch dtype: {dtype_name}") from error

        return cls(
            name=str(values.get("_name_or_path", config_path.parent.name)),
            num_hidden_layers=int(values["num_hidden_layers"]),
            max_position_embeddings=int(values["max_position_embeddings"]),
            hidden_size=int(values["hidden_size"]),
            num_attention_heads=int(values["num_attention_heads"]),
            ffn_dim=int(values["ffn_dim"]),
            vocab_size=int(values["vocab_size"]),
            word_embed_proj_dim=int(
                values.get("word_embed_proj_dim", values["hidden_size"])
            ),
            pad_token_id=int(values.get("pad_token_id", 1)),
            bos_token_id=int(values.get("bos_token_id", 2)),
            eos_token_id=int(values.get("eos_token_id", 2)),
            activation_function=str(values.get("activation_function", "relu")),
            layer_norm_epsilon=float(values.get("layer_norm_eps", 1e-5)),
            do_layer_norm_before=bool(values.get("do_layer_norm_before", True)),
            tie_word_embeddings=bool(values.get("tie_word_embeddings", True)),
            dtype=dtype,
        )
