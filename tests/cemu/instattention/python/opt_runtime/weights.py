"""OPT weight containers."""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class OptLayerNormWeights:
    weight: torch.Tensor
    bias: torch.Tensor


@dataclass(frozen=True)
class OptEmbeddingWeights:
    token: torch.Tensor
    position: torch.Tensor


@dataclass(frozen=True)
class OptAttentionWeights:
    query: torch.Tensor
    query_bias: torch.Tensor
    key: torch.Tensor
    key_bias: torch.Tensor
    value: torch.Tensor
    value_bias: torch.Tensor
    output: torch.Tensor
    output_bias: torch.Tensor
    input_norm: OptLayerNormWeights


@dataclass(frozen=True)
class OptMlpWeights:
    input: torch.Tensor
    input_bias: torch.Tensor
    output: torch.Tensor
    output_bias: torch.Tensor
    input_norm: OptLayerNormWeights


@dataclass(frozen=True)
class OptLayerWeights:
    attention: OptAttentionWeights
    mlp: OptMlpWeights
