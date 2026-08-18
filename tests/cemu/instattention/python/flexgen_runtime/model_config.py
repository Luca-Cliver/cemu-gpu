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
class FlexGenLlamaConfig:
    num_hidden_layers: int
    hidden_size: int
    num_attention_heads: int
    intermediate_size: int
    vocab_size: int
    pad_token_id: int = 2
    rope_theta: float = 10000.0
    rms_norm_epsilon: float = 1e-6
    dtype: torch.dtype = torch.float16
    num_key_value_heads: int = 0

    def __post_init__(self) -> None:
        for field_name in (
            "num_hidden_layers",
            "hidden_size",
            "num_attention_heads",
            "intermediate_size",
            "vocab_size",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{field_name} must be a positive integer")
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError("num_attention_heads must divide hidden_size")

        num_key_value_heads = self.num_key_value_heads or self.num_attention_heads
        if num_key_value_heads != self.num_attention_heads:
            raise ValueError("the current FlexGen Prefill path supports MHA, not GQA")
        object.__setattr__(self, "num_key_value_heads", num_key_value_heads)
        if self.dtype not in (torch.float16, torch.float32, torch.bfloat16):
            raise ValueError("unsupported model dtype")

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

    @classmethod
    def from_json(cls, path: Any):
        config_path = Path(path)
        with config_path.open("r", encoding="utf-8") as config_file:
            values = json.load(config_file)

        dtype_name = values.get("torch_dtype", "float16")
        if isinstance(dtype_name, torch.dtype):
            dtype = dtype_name
        else:
            try:
                dtype = _TORCH_DTYPES[str(dtype_name).lower()]
            except KeyError as error:
                raise ValueError(f"unsupported torch dtype: {dtype_name}") from error

        pad_token_id = values.get("pad_token_id")
        if pad_token_id is None:
            pad_token_id = 2
        return cls(
            num_hidden_layers=int(values["num_hidden_layers"]),
            hidden_size=int(values["hidden_size"]),
            num_attention_heads=int(values["num_attention_heads"]),
            num_key_value_heads=int(
                values.get("num_key_value_heads", values["num_attention_heads"])
            ),
            intermediate_size=int(values["intermediate_size"]),
            vocab_size=int(values["vocab_size"]),
            pad_token_id=int(pad_token_id),
            rope_theta=float(values.get("rope_theta", 10000.0)),
            rms_norm_epsilon=float(
                values.get("rms_norm_eps", values.get("layer_norm_eps", 1e-6))
            ),
            dtype=dtype,
        )
