import math
import struct
from dataclasses import dataclass

import numpy as np


QK_SOFTMAX = 1
PV = 2
_METADATA = struct.Struct("<9If")


@dataclass(frozen=True)
class AttentionPhaseMetadata:
    batch_size: int
    num_query_heads: int
    num_kv_heads: int
    head_dim: int
    token_count: int
    token_stride: int
    scale: float
    dtype: object = np.float32

    def __post_init__(self):
        for name in (
            "batch_size", "num_query_heads", "num_kv_heads",
            "head_dim", "token_count", "token_stride",
        ):
            value = getattr(self, name)
            if type(value) is not int or not 0 < value <= 0xFFFFFFFF:
                raise ValueError(f"{name} must be a positive uint32")
        dtype = np.dtype(self.dtype)
        if dtype not in (np.dtype(np.float16), np.dtype(np.float32)):
            raise ValueError("phase dtype must be float16 or float32")
        object.__setattr__(self, "dtype", dtype)
        if self.num_query_heads % self.num_kv_heads:
            raise ValueError("query heads must be divisible by KV heads")
        payload = self.batch_size * self.num_kv_heads * self.head_dim * dtype.itemsize
        if self.token_stride % 512 or self.token_stride < payload:
            raise ValueError("token stride must be 512-byte aligned and fit one KV token")
        if not math.isfinite(self.scale) or not 0 < self.scale <= np.finfo(np.float32).max:
            raise ValueError("scale must fit in a finite positive float32")
        scale = struct.unpack("<f", struct.pack("<f", self.scale))[0]
        if scale <= 0:
            raise ValueError("scale underflows float32")
        object.__setattr__(self, "scale", scale)
        if max(self.query_bytes, self.kv_bytes, self.probability_bytes) > 0x7FFFFFFFFFFFFFFF:
            raise ValueError("phase range size exceeds int64")

    @property
    def query_bytes(self):
        return self.batch_size * self.num_query_heads * self.head_dim * self.dtype.itemsize

    @property
    def kv_bytes(self):
        return self.token_count * self.token_stride

    @property
    def probability_bytes(self):
        return self.batch_size * self.num_query_heads * self.token_count * 4

    def pack(self, phase):
        if type(phase) is not int or phase not in (QK_SOFTMAX, PV):
            raise ValueError("unknown Attention phase")
        return _METADATA.pack(
            1, phase, 2 if self.dtype == np.float16 else 1,
            self.batch_size, self.num_query_heads, self.num_kv_heads,
            self.head_dim, self.token_count, self.token_stride, self.scale,
        )
