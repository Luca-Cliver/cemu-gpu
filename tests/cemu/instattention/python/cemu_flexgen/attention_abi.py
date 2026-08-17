import math
import struct
from dataclasses import dataclass

import numpy as np

from .kv_layout import KvCacheLayout, KvChunk


ATTENTION_ABI_VERSION = 1
ATTENTION_DTYPE_FLOAT32 = 1
ATTENTION_FLAG_RESET_STATE = 1 << 0
ATTENTION_FLAG_FINALIZE = 1 << 1
_ATTENTION_METADATA = struct.Struct("<8IfI")


@dataclass(frozen=True)
class DenseAttentionMetadata:
    batch_size: int
    num_query_heads: int
    num_kv_heads: int
    head_dim: int
    token_count: int
    token_stride: int
    scale: float
    reset_state: bool = True
    finalize: bool = True

    def __post_init__(self) -> None:
        for field_name in (
            "batch_size",
            "num_query_heads",
            "num_kv_heads",
            "head_dim",
            "token_count",
            "token_stride",
        ):
            value = getattr(self, field_name)
            if (
                not isinstance(value, int)
                or isinstance(value, bool)
                or value <= 0
                or value > 0xFFFFFFFF
            ):
                raise ValueError(f"{field_name} must fit in a positive uint32")
        if self.num_query_heads % self.num_kv_heads != 0:
            raise ValueError("num_query_heads must be divisible by num_kv_heads")
        if self.token_stride % 512 != 0:
            raise ValueError("token_stride must be a multiple of 512")
        if not math.isfinite(self.scale) or self.scale <= 0:
            raise ValueError("scale must be finite and positive")
        if not isinstance(self.reset_state, bool):
            raise TypeError("reset_state must be a bool")
        if not isinstance(self.finalize, bool):
            raise TypeError("finalize must be a bool")

    @classmethod
    def from_layout(
        cls,
        layout: KvCacheLayout,
        num_query_heads: int,
        token_count: int,
        scale: float = None,
        reset_state: bool = True,
        finalize: bool = True,
    ):
        if not isinstance(layout, KvCacheLayout):
            raise TypeError("layout must be a KvCacheLayout")
        if layout.config.dtype != np.dtype(np.float32):
            raise TypeError("the first dense Attention ABI supports only float32")
        if scale is None:
            scale = 1.0 / math.sqrt(layout.config.head_dim)
        return cls(
            batch_size=layout.config.batch_size,
            num_query_heads=num_query_heads,
            num_kv_heads=layout.config.num_kv_heads,
            head_dim=layout.config.head_dim,
            token_count=token_count,
            token_stride=layout.token_stride,
            scale=float(scale),
            reset_state=reset_state,
            finalize=finalize,
        )

    @property
    def query_shape(self):
        return (self.batch_size, self.num_query_heads, self.head_dim)

    @property
    def output_shape(self):
        return self.query_shape

    def validate_chunk(self, layout: KvCacheLayout, chunk: KvChunk) -> None:
        if not isinstance(layout, KvCacheLayout):
            raise TypeError("layout must be a KvCacheLayout")
        if not isinstance(chunk, KvChunk):
            raise TypeError("chunk must be a KvChunk")
        if layout.config.dtype != np.dtype(np.float32):
            raise TypeError("the first dense Attention ABI supports only float32")
        expected = (
            layout.config.batch_size,
            layout.config.num_kv_heads,
            layout.config.head_dim,
            layout.token_stride,
            chunk.token_count,
        )
        actual = (
            self.batch_size,
            self.num_kv_heads,
            self.head_dim,
            self.token_stride,
            self.token_count,
        )
        if actual != expected:
            raise ValueError("Attention metadata does not match the KV chunk layout")

    def pack(self) -> bytes:
        return _ATTENTION_METADATA.pack(
            ATTENTION_ABI_VERSION,
            ATTENTION_DTYPE_FLOAT32,
            self.batch_size,
            self.num_query_heads,
            self.num_kv_heads,
            self.head_dim,
            self.token_count,
            self.token_stride,
            self.scale,
            (ATTENTION_FLAG_RESET_STATE if self.reset_state else 0)
            | (ATTENTION_FLAG_FINALIZE if self.finalize else 0),
        )
