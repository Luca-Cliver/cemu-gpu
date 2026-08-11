from dataclasses import dataclass
from numbers import Integral
from typing import Any, Iterator

import numpy as np


BLOCK_ALIGNMENT = 512
LAYER_ALIGNMENT = 4096


def align_up(value: int, alignment: int) -> int:
    if not isinstance(value, Integral) or isinstance(value, bool) or value < 0:
        raise ValueError("value must be a non-negative integer")
    if (
        not isinstance(alignment, Integral)
        or isinstance(alignment, bool)
        or alignment <= 0
    ):
        raise ValueError("alignment must be a positive integer")
    return (int(value) + int(alignment) - 1) // int(alignment) * int(alignment)


@dataclass(frozen=True)
class KvLayoutConfig:
    num_layers: int
    max_seq_len: int
    batch_size: int
    num_kv_heads: int
    head_dim: int
    dtype: Any = np.float16

    def __post_init__(self) -> None:
        for field_name in (
            "num_layers",
            "max_seq_len",
            "batch_size",
            "num_kv_heads",
            "head_dim",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, Integral) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{field_name} must be a positive integer")
            object.__setattr__(self, field_name, int(value))

        try:
            normalized_dtype = np.dtype(self.dtype)
        except TypeError as error:
            raise ValueError("dtype must be a valid NumPy dtype") from error
        object.__setattr__(self, "dtype", normalized_dtype)


@dataclass(frozen=True)
class KvChunk:
    layer: int
    start_token: int
    token_count: int
    nvm_offset: int
    copy_size: int

    @property
    def end_token(self) -> int:
        return self.start_token + self.token_count


class KvCacheLayout:
    def __init__(self, config: KvLayoutConfig):
        if not isinstance(config, KvLayoutConfig):
            raise TypeError("config must be a KvLayoutConfig")
        self.config = config

    @property
    def element_size(self) -> int:
        return self.config.dtype.itemsize

    @property
    def head_bytes(self) -> int:
        return self.config.head_dim * self.element_size

    @property
    def token_bytes(self) -> int:
        return self.config.batch_size * self.config.num_kv_heads * self.head_bytes

    @property
    def token_stride(self) -> int:
        return align_up(self.token_bytes, BLOCK_ALIGNMENT)

    @property
    def layer_stride(self) -> int:
        return align_up(
            self.config.max_seq_len * self.token_stride,
            LAYER_ALIGNMENT,
        )

    @property
    def file_size(self) -> int:
        return self.config.num_layers * self.layer_stride

    def token_offset(self, layer: int, token: int) -> int:
        self._validate_index("layer", layer, self.config.num_layers)
        self._validate_index("token", token, self.config.max_seq_len)
        return layer * self.layer_stride + token * self.token_stride

    def head_offset(
        self,
        layer: int,
        token: int,
        batch: int,
        kv_head: int,
    ) -> int:
        self._validate_index("batch", batch, self.config.batch_size)
        self._validate_index("kv_head", kv_head, self.config.num_kv_heads)
        head_index = batch * self.config.num_kv_heads + kv_head
        return self.token_offset(layer, token) + head_index * self.head_bytes

    def element_offset(
        self,
        layer: int,
        token: int,
        batch: int,
        kv_head: int,
        dimension: int,
    ) -> int:
        self._validate_index("dimension", dimension, self.config.head_dim)
        return (
            self.head_offset(layer, token, batch, kv_head)
            + dimension * self.element_size
        )

    def tokens_per_chunk(
        self,
        k_staging_bytes: int,
        v_staging_bytes: int,
    ) -> int:
        self._validate_staging_size("k_staging_bytes", k_staging_bytes)
        self._validate_staging_size("v_staging_bytes", v_staging_bytes)
        capacity = min(
            int(k_staging_bytes) // self.token_stride,
            int(v_staging_bytes) // self.token_stride,
            self.config.max_seq_len,
        )
        if capacity == 0:
            raise ValueError("staging buffers cannot hold one KV token")
        return capacity

    def tokens_per_chunk_from_budget(
        self,
        fdm_budget_bytes: int,
        buffer_count: int = 1,
        reserved_bytes: int = 0,
    ) -> int:
        self._validate_staging_size("fdm_budget_bytes", fdm_budget_bytes)
        if (
            not isinstance(buffer_count, Integral)
            or isinstance(buffer_count, bool)
            or buffer_count <= 0
        ):
            raise ValueError("buffer_count must be a positive integer")
        if (
            not isinstance(reserved_bytes, Integral)
            or isinstance(reserved_bytes, bool)
            or reserved_bytes < 0
            or reserved_bytes % BLOCK_ALIGNMENT != 0
        ):
            raise ValueError(
                "reserved_bytes must be a non-negative multiple of 512"
            )
        if reserved_bytes >= fdm_budget_bytes:
            raise ValueError("reserved bytes leave no space for KV staging")

        bytes_per_token = 2 * int(buffer_count) * self.token_stride
        capacity = min(
            (int(fdm_budget_bytes) - int(reserved_bytes)) // bytes_per_token,
            self.config.max_seq_len,
        )
        if capacity == 0:
            raise ValueError("FDM budget cannot hold one KV token")
        return capacity

    def chunk_count(self, valid_tokens: int, tokens_per_chunk: int) -> int:
        self._validate_valid_tokens(valid_tokens)
        self._validate_chunk_capacity(tokens_per_chunk)
        return (int(valid_tokens) + int(tokens_per_chunk) - 1) // int(
            tokens_per_chunk
        )

    def iter_chunks(
        self,
        layer: int,
        valid_tokens: int,
        k_staging_bytes: int,
        v_staging_bytes: int,
    ) -> Iterator[KvChunk]:
        self._validate_index("layer", layer, self.config.num_layers)
        self._validate_valid_tokens(valid_tokens)
        chunk_capacity = self.tokens_per_chunk(k_staging_bytes, v_staging_bytes)

        for start_token in range(0, int(valid_tokens), chunk_capacity):
            token_count = min(chunk_capacity, int(valid_tokens) - start_token)
            yield KvChunk(
                layer=layer,
                start_token=start_token,
                token_count=token_count,
                nvm_offset=self.token_offset(layer, start_token),
                copy_size=token_count * self.token_stride,
            )

    @staticmethod
    def _validate_index(name: str, value: int, upper_bound: int) -> None:
        if not isinstance(value, Integral) or isinstance(value, bool):
            raise TypeError(f"{name} must be an integer")
        if value < 0 or value >= upper_bound:
            raise IndexError(f"{name} index {value} is out of range")

    def _validate_valid_tokens(self, valid_tokens: int) -> None:
        if not isinstance(valid_tokens, Integral) or isinstance(valid_tokens, bool):
            raise TypeError("valid_tokens must be an integer")
        if valid_tokens < 0 or valid_tokens > self.config.max_seq_len:
            raise ValueError("valid_tokens is outside the configured sequence length")

    @staticmethod
    def _validate_chunk_capacity(tokens_per_chunk: int) -> None:
        if (
            not isinstance(tokens_per_chunk, Integral)
            or isinstance(tokens_per_chunk, bool)
            or tokens_per_chunk <= 0
        ):
            raise ValueError("tokens_per_chunk must be a positive integer")

    @staticmethod
    def _validate_staging_size(name: str, value: int) -> None:
        if (
            not isinstance(value, Integral)
            or isinstance(value, bool)
            or value <= 0
            or value % BLOCK_ALIGNMENT != 0
        ):
            raise ValueError(f"{name} must be a positive multiple of 512")
