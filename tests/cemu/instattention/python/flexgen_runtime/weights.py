from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .model_config import FlexGenLlamaConfig


@dataclass(frozen=True)
class FlexGenAttentionWeights:
    query: torch.Tensor
    key: torch.Tensor
    value: torch.Tensor
    output: torch.Tensor
    input_norm: torch.Tensor
    post_attention_norm: torch.Tensor


@dataclass(frozen=True)
class FlexGenMlpWeights:
    gate: torch.Tensor
    up: torch.Tensor
    down: torch.Tensor


@dataclass(frozen=True)
class FlexGenLayerWeights:
    attention: FlexGenAttentionWeights
    mlp: FlexGenMlpWeights


class FlexGenWeightLoader:
    def __init__(
        self,
        config: FlexGenLlamaConfig,
        weight_directory: Any,
        device: Any = "cpu",
    ):
        if not isinstance(config, FlexGenLlamaConfig):
            raise TypeError("config must be a FlexGenLlamaConfig")
        self.config = config
        self.weight_directory = Path(weight_directory)
        self.device = torch.device(device)
        if not self.weight_directory.is_dir():
            raise FileNotFoundError(
                f"weight directory does not exist: {self.weight_directory}"
            )

    def load_embedding(self) -> torch.Tensor:
        return self._load(
            "embed_tokens.weight",
            (self.config.vocab_size, self.config.hidden_size),
        )

    def load_final_norm(self) -> torch.Tensor:
        return self._load("norm.weight", (self.config.hidden_size,))

    def load_lm_head(self) -> torch.Tensor:
        return self._load(
            "embed_tokens.weight",
            (self.config.vocab_size, self.config.hidden_size),
        )

    def load_layer(self, layer: int) -> FlexGenLayerWeights:
        if not isinstance(layer, int) or isinstance(layer, bool):
            raise TypeError("layer must be an integer")
        if layer < 0 or layer >= self.config.num_hidden_layers:
            raise IndexError("layer is outside the model configuration")

        hidden_size = self.config.hidden_size
        intermediate_size = self.config.intermediate_size
        prefix = f"layers.{layer}."
        attention = FlexGenAttentionWeights(
            query=self._load(
                prefix + "self_attn.q_proj.weight",
                (hidden_size, hidden_size),
            ),
            key=self._load(
                prefix + "self_attn.k_proj.weight",
                (hidden_size, hidden_size),
            ),
            value=self._load(
                prefix + "self_attn.v_proj.weight",
                (hidden_size, hidden_size),
            ),
            output=self._load(
                prefix + "self_attn.o_proj.weight",
                (hidden_size, hidden_size),
            ),
            input_norm=self._load(
                prefix + "input_layernorm.weight",
                (hidden_size,),
            ),
            post_attention_norm=self._load(
                prefix + "post_attention_layernorm.weight",
                (hidden_size,),
            ),
        )
        mlp = FlexGenMlpWeights(
            gate=self._load(
                prefix + "mlp.gate_proj.weight",
                (intermediate_size, hidden_size),
            ),
            up=self._load(
                prefix + "mlp.up_proj.weight",
                (intermediate_size, hidden_size),
            ),
            down=self._load(
                prefix + "mlp.down_proj.weight",
                (hidden_size, intermediate_size),
            ),
        )
        return FlexGenLayerWeights(attention=attention, mlp=mlp)

    def _load(self, filename: str, expected_shape) -> torch.Tensor:
        path = self.weight_directory / filename
        if not path.is_file():
            raise FileNotFoundError(f"model weight does not exist: {path}")
        array = np.load(path, allow_pickle=False)
        if array.shape != expected_shape:
            raise ValueError(
                f"{path} has shape {array.shape}, expected {expected_shape}"
            )
        tensor = torch.from_numpy(array)
        return tensor.to(device=self.device, dtype=self.config.dtype)
