import math
from dataclasses import dataclass

from .config import InstCsdConfig


@dataclass(frozen=True)
class DenseAttentionRuntimeBreakdown:
    qk_ns: int
    softmax_ns: int
    av_ns: int

    @property
    def total_ns(self) -> int:
        return self.qk_ns + self.softmax_ns + self.av_ns


class DenseAttentionRuntimeModel:
    def __init__(self, device: InstCsdConfig):
        if not isinstance(device, InstCsdConfig):
            raise TypeError("device must be an InstCsdConfig")
        self.device = device

    def estimate(
        self,
        batch_size: int,
        num_query_heads: int,
        head_dim: int,
        token_count: int,
    ) -> DenseAttentionRuntimeBreakdown:
        for name, value in (
            ("batch_size", batch_size),
            ("num_query_heads", num_query_heads),
            ("head_dim", head_dim),
            ("token_count", token_count),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")

        vector_count = batch_size * num_query_heads
        gemv_flops = 2 * vector_count * token_count * head_dim
        gemv_ns = math.ceil(gemv_flops / self.device.gemv_gflops)

        anchor_elements = (
            self.device.softmax_anchor_heads
            * self.device.softmax_anchor_tokens
        )
        softmax_elements = vector_count * token_count
        anchor_flops = (
            self.device.softmax_mflops
            * self.device.softmax_anchor_latency_us
        )
        softmax_flops = anchor_flops * softmax_elements / anchor_elements
        softmax_ns = math.ceil(
            softmax_flops * 1000.0 / self.device.softmax_mflops
        )
        return DenseAttentionRuntimeBreakdown(
            qk_ns=gemv_ns,
            softmax_ns=softmax_ns,
            av_ns=gemv_ns,
        )

    def estimate_filter_ns(self, byte_count: int) -> int:
        if not isinstance(byte_count, int) or isinstance(byte_count, bool):
            raise TypeError("byte_count must be an integer")
        if byte_count < 0:
            raise ValueError("byte_count must be non-negative")
        if byte_count == 0:
            return 0
        return math.ceil(byte_count / self.device.filter_gbps)
