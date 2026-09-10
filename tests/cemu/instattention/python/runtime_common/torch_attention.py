"""Model-independent PyTorch reference Attention backend."""

import math
from typing import Any, Callable, Optional, Sequence, Tuple

import torch


def partition_kv_cache_by_batch(config, kv_cache, microbatch_size):
    if (
        not isinstance(microbatch_size, int)
        or isinstance(microbatch_size, bool)
        or microbatch_size <= 0
    ):
        raise ValueError("microbatch_size must be a positive integer")
    if len(kv_cache) != config.num_hidden_layers:
        raise ValueError("kv_cache must contain one entry per model layer")

    cache_width = kv_cache[0][0].shape[1]
    if cache_width % config.num_key_value_heads != 0:
        raise ValueError("KV cache width is not divisible by the KV head count")
    total_batch_size = cache_width // config.num_key_value_heads
    if total_batch_size % microbatch_size != 0:
        raise ValueError("KV cache batch size must be divisible by microbatch_size")

    num_microbatches = total_batch_size // microbatch_size
    partitions = [[] for _ in range(num_microbatches)]
    heads_per_microbatch = microbatch_size * config.num_key_value_heads
    for layer, (keys, values) in enumerate(kv_cache):
        if keys.ndim != 3 or values.shape != keys.shape:
            raise ValueError(f"invalid K/V cache shape for layer {layer}")
        if keys.shape[1] != cache_width or keys.shape[2] != config.head_dim:
            raise ValueError(f"inconsistent K/V cache shape for layer {layer}")
        for microbatch in range(num_microbatches):
            start = microbatch * heads_per_microbatch
            end = start + heads_per_microbatch
            partitions[microbatch].append(
                (keys[:, start:end, :], values[:, start:end, :])
            )
    return tuple(tuple(partition) for partition in partitions)


class TorchAttentionBackend:
    def __init__(
        self,
        config: Any,
        kv_cache: Sequence[Tuple[torch.Tensor, torch.Tensor]],
        logger: Optional[Callable[[str], None]] = None,
        copy_cache: bool = True,
        attention_scale: Optional[float] = None,
        accumulation_dtype: Optional[torch.dtype] = None,
    ):
        for field_name in (
            "num_hidden_layers",
            "num_attention_heads",
            "num_key_value_heads",
            "head_dim",
        ):
            value = getattr(config, field_name, None)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise TypeError(f"config must provide a positive {field_name}")
        if config.num_attention_heads % config.num_key_value_heads != 0:
            raise ValueError("num_key_value_heads must divide num_attention_heads")
        if len(kv_cache) != config.num_hidden_layers:
            raise ValueError("kv_cache must contain one entry per model layer")
        if logger is not None and not callable(logger):
            raise TypeError("logger must be callable")
        if not isinstance(copy_cache, bool):
            raise TypeError("copy_cache must be a boolean")
        if attention_scale is None:
            attention_scale = 1.0 / math.sqrt(config.head_dim)
        if not math.isfinite(attention_scale) or attention_scale <= 0:
            raise ValueError("attention_scale must be finite and positive")
        if accumulation_dtype is not None:
            if not isinstance(accumulation_dtype, torch.dtype):
                raise TypeError("accumulation_dtype must be a torch dtype")
            if not torch.empty((), dtype=accumulation_dtype).is_floating_point():
                raise TypeError("accumulation_dtype must be floating point")

        self.config = config
        self.logger = logger
        self.attention_scale = float(attention_scale)
        self.accumulation_dtype = accumulation_dtype
        self._num_key_value_groups = (
            config.num_attention_heads // config.num_key_value_heads
        )
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

        batch_size = query.shape[0]
        if keys.shape[1] != batch_size * self.config.num_key_value_heads:
            raise ValueError("query batch size does not match the reference KV cache")
        output_dtype = query.dtype
        keys = keys.reshape(
            valid_tokens,
            batch_size,
            self.config.num_key_value_heads,
            self.config.head_dim,
        ).permute(1, 2, 0, 3)
        values = values.reshape(
            valid_tokens,
            batch_size,
            self.config.num_key_value_heads,
            self.config.head_dim,
        ).permute(1, 2, 0, 3)
        if self.accumulation_dtype is not None:
            query = query.to(dtype=self.accumulation_dtype)
            keys = keys.to(dtype=self.accumulation_dtype)
            values = values.to(dtype=self.accumulation_dtype)
        attention_keys = keys.repeat_interleave(
            self._num_key_value_groups,
            dim=1,
        )
        attention_values = values.repeat_interleave(
            self._num_key_value_groups,
            dim=1,
        )
        scores = torch.einsum("bhd,bhtd->bht", query, attention_keys)
        scores = scores * self.attention_scale
        probabilities = torch.softmax(scores, dim=-1)
        output = torch.einsum(
            "bht,bhtd->bhd",
            probabilities,
            attention_values,
        )
        if output.dtype != output_dtype:
            output = output.to(dtype=output_dtype)
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
        valid_shape = (
            keys.ndim == 3
            and keys.shape[1] % self.config.num_key_value_heads == 0
            and keys.shape[2] == self.config.head_dim
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
