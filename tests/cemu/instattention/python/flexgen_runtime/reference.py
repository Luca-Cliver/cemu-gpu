import math
from typing import Callable, Optional, Sequence, Tuple

import torch

from .model_config import FlexGenLlamaConfig


class FlexGenTorchAttentionBackend:
    def __init__(
        self,
        config: FlexGenLlamaConfig,
        kv_cache: Sequence[Tuple[torch.Tensor, torch.Tensor]],
        logger: Optional[Callable[[str], None]] = None,
        copy_cache: bool = True,
    ):
        if not isinstance(config, FlexGenLlamaConfig):
            raise TypeError("config must be a FlexGenLlamaConfig")
        if len(kv_cache) != config.num_hidden_layers:
            raise ValueError("kv_cache must contain one entry per model layer")
        if logger is not None and not callable(logger):
            raise TypeError("logger must be callable")
        if not isinstance(copy_cache, bool):
            raise TypeError("copy_cache must be a boolean")

        self.config = config
        self.logger = logger
        self._cache = []
        for layer, (keys, values) in enumerate(kv_cache):
            self._validate_cache(layer, keys, values)
            if copy_cache:
                keys = keys.detach().clone()
                values = values.detach().clone()
            else:
                keys = keys.detach()
                values = values.detach()
            self._cache.append((keys, values))

    def append_decode(
        self,
        layer: int,
        token: int,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> None:
        keys, values = self._layer_cache(layer)
        self._validate_cache(layer, key, value, expected_tokens=1)
        if token != keys.shape[0]:
            raise ValueError(
                f"Decode token {token} does not follow the {keys.shape[0]} cached tokens"
            )
        self._cache[layer] = (
            torch.cat((keys, key.to(device=keys.device, dtype=keys.dtype)), dim=0),
            torch.cat(
                (values, value.to(device=values.device, dtype=values.dtype)),
                dim=0,
            ),
        )
        self._log(f"append layer={layer}, token={token}")

    def decode(
        self,
        layer: int,
        query: torch.Tensor,
        valid_tokens: int,
    ) -> torch.Tensor:
        keys, values = self._layer_cache(layer)
        if valid_tokens != keys.shape[0]:
            raise ValueError("valid_tokens does not match the reference KV cache")

        query = torch.as_tensor(query, device=keys.device, dtype=keys.dtype)
        if (
            query.ndim != 3
            or query.shape[1] != self.config.num_attention_heads
            or query.shape[2] != self.config.head_dim
        ):
            raise ValueError(
                "query must have shape [batch, num_attention_heads, head_dim]"
            )

        actual_batch_size = query.shape[0]
        if keys.shape[1] != actual_batch_size * self.config.num_key_value_heads:
            raise ValueError("query batch size does not match the reference KV cache")
        keys = keys.reshape(
            valid_tokens,
            actual_batch_size,
            self.config.num_key_value_heads,
            self.config.head_dim,
        ).permute(1, 2, 0, 3)
        values = values.reshape(
            valid_tokens,
            actual_batch_size,
            self.config.num_key_value_heads,
            self.config.head_dim,
        ).permute(1, 2, 0, 3)
        attention_keys = keys.repeat_interleave(
            self.config.num_key_value_groups,
            dim=1,
        )
        attention_values = values.repeat_interleave(
            self.config.num_key_value_groups,
            dim=1,
        )
        scores = torch.einsum("bhd,bhtd->bht", query, attention_keys)
        scores = scores * (1.0 / math.sqrt(self.config.head_dim))
        probabilities = torch.softmax(scores, dim=-1)
        output = torch.einsum(
            "bht,bhtd->bhd",
            probabilities,
            attention_values,
        )
        self._log(
            f"decode layer={layer}, valid_tokens={valid_tokens}, "
            f"output={tuple(output.shape)}"
        )
        return output

    def layer_cache(self, layer: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self._layer_cache(layer)

    def _layer_cache(self, layer: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if not isinstance(layer, int) or isinstance(layer, bool):
            raise TypeError("layer must be an integer")
        if layer < 0 or layer >= len(self._cache):
            raise IndexError("layer is outside the reference KV cache")
        return self._cache[layer]

    def _validate_cache(
        self,
        layer: int,
        keys: torch.Tensor,
        values: torch.Tensor,
        expected_tokens: Optional[int] = None,
    ) -> None:
        if not isinstance(keys, torch.Tensor) or not isinstance(values, torch.Tensor):
            raise TypeError("reference K/V caches must be torch tensors")
        expected_tail = (
            self.config.num_key_value_heads,
            self.config.head_dim,
        )
        valid_shape = (
            keys.ndim == 3
            and keys.shape[1] % self.config.num_key_value_heads == 0
            and keys.shape[2:] == expected_tail[1:]
        )
        if not valid_shape or values.shape != keys.shape:
            raise ValueError(f"invalid K/V cache shape for layer {layer}")
        if expected_tokens is not None and keys.shape[0] != expected_tokens:
            raise ValueError(f"layer {layer} must contain {expected_tokens} token")
        if keys.device != values.device or keys.dtype != values.dtype:
            raise ValueError("reference K/V caches must share device and dtype")

    def _log(self, message: str) -> None:
        if self.logger is not None:
            self.logger(f"[torch-attention] {message}")
