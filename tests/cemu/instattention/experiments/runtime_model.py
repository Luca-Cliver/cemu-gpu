import math
from dataclasses import dataclass

from .config import InstCsdConfig
from .softmax_ratio import SoftmaxRatioTable


@dataclass(frozen=True)
class DenseAttentionRuntimeBreakdown:
    qk_ns: int
    softmax_ns: int
    av_ns: int

    @property
    def total_ns(self) -> int:
        return self.qk_ns + self.softmax_ns + self.av_ns


class DenseAttentionRuntimeModel:
    def __init__(self, device: InstCsdConfig, softmax_ratios=None):
        if not isinstance(device, InstCsdConfig):
            raise TypeError("device must be an InstCsdConfig")
        self.device = device
        if softmax_ratios is not None:
            if not isinstance(softmax_ratios, SoftmaxRatioTable):
                raise TypeError("softmax_ratios must be a SoftmaxRatioTable")
            softmax_ratios.validate_anchor(device)
        self.softmax_ratios = softmax_ratios

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
        if self.softmax_ratios is not None:
            softmax_ns = self.softmax_ratios.estimate_ns(
                self.device, vector_count, token_count,
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


@dataclass(frozen=True)
class SparfAttentionRuntimeBreakdown:
    approximate_ns: int
    exact_qk_ns: int
    pv_ns: int


class SparfAttentionRuntimeModel(DenseAttentionRuntimeModel):
    def estimate(self, batch_size, num_query_heads, head_dim, token_count,
                 top_r, top_k, element_size=2):
        vectors = batch_size * num_query_heads
        approximate_gemv = math.ceil(
            2 * vectors * token_count * top_r / self.device.gemv_gflops
        )
        approximate_filter = self.estimate_filter_ns(
            vectors * (head_dim + token_count) * element_size
        )
        anchor = self.device.softmax_anchor_heads * self.device.softmax_anchor_tokens
        approximate_softmax = math.ceil(
            self.device.softmax_anchor_latency_us * 1000 * vectors * token_count / anchor
        )
        exact_gemv = math.ceil(
            2 * vectors * top_k * head_dim / self.device.gemv_gflops
        )
        pv_gemv = math.ceil(
            2 * vectors * (top_k + 1) * head_dim / self.device.gemv_gflops
        )
        exact_softmax = math.ceil(
            self.device.softmax_anchor_latency_us * 1000 * vectors * top_k / anchor
        )
        if self.softmax_ratios is not None:
            approximate_softmax = self.softmax_ratios.estimate_ns(
                self.device, vectors, token_count,
            )
            exact_softmax = self.softmax_ratios.estimate_ns(
                self.device, vectors, top_k,
            )
        return SparfAttentionRuntimeBreakdown(
            approximate_ns=approximate_filter + approximate_gemv + approximate_softmax,
            exact_qk_ns=exact_gemv + exact_softmax,
            pv_ns=pv_gemv,
        )
