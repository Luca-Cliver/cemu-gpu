from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .prefill import (
    _apply_rotary_pos_emb,
    _rms_norm,
    _rotary_embedding,
)
from .weights import FlexGenAttentionWeights


@dataclass(frozen=True)
class FlexGenDecodeProjection:
    query: torch.Tensor
    key: torch.Tensor
    value: torch.Tensor


@dataclass(frozen=True)
class FlexGenDecodeAttentionOutput:
    hidden_states: torch.Tensor
    mlp_inputs: torch.Tensor
    query: torch.Tensor
    key: torch.Tensor
    value: torch.Tensor
    attention_output: torch.Tensor


def prepare_flexgen_decode_attention(
    inputs: torch.Tensor,
    weights: FlexGenAttentionWeights,
    num_heads: int,
    num_key_value_heads: int,
    token_position: int,
    rope_theta: float = 10000.0,
    epsilon: float = 1e-6,
) -> FlexGenDecodeProjection:
    if inputs.ndim != 3 or inputs.shape[1] != 1:
        raise ValueError("Decode inputs must have shape [batch, 1, hidden]")
    if not isinstance(weights, FlexGenAttentionWeights):
        raise TypeError("weights must be FlexGenAttentionWeights")
    if not isinstance(token_position, int) or isinstance(token_position, bool):
        raise TypeError("token_position must be an integer")
    if token_position < 0:
        raise ValueError("token_position must be non-negative")

    batch_size, _, hidden_size = inputs.shape
    if num_heads <= 0 or hidden_size % num_heads != 0:
        raise ValueError("num_heads must divide the hidden size")
    if (
        not isinstance(num_key_value_heads, int)
        or isinstance(num_key_value_heads, bool)
        or num_key_value_heads <= 0
        or num_heads % num_key_value_heads != 0
    ):
        raise ValueError("num_key_value_heads must divide num_heads")
    head_dim = hidden_size // num_heads
    if head_dim % 2 != 0:
        raise ValueError("RoPE requires an even head dimension")

    normalized = _rms_norm(weights.input_norm, inputs, epsilon)
    query = F.linear(normalized, weights.query, bias=None)
    key = F.linear(normalized, weights.key, bias=None)
    value = F.linear(normalized, weights.value, bias=None)

    query = query.view(batch_size, 1, num_heads, head_dim).transpose(1, 2)
    key = key.view(
        batch_size,
        1,
        num_key_value_heads,
        head_dim,
    ).transpose(1, 2)
    value = value.view(
        batch_size,
        1,
        num_key_value_heads,
        head_dim,
    ).transpose(1, 2)

    position_ids = torch.full(
        (batch_size, 1),
        token_position,
        dtype=torch.long,
        device=inputs.device,
    )
    cosine, sine = _rotary_embedding(
        head_dim,
        token_position + 1,
        value.dtype,
        inputs.device,
        rope_theta,
    )
    query, key = _apply_rotary_pos_emb(
        query,
        key,
        cosine,
        sine,
        position_ids,
    )

    query = query[:, :, 0, :].contiguous()
    key = key.permute(2, 0, 1, 3).reshape(
        1,
        batch_size * num_key_value_heads,
        head_dim,
    )
    value = value.permute(2, 0, 1, 3).reshape(
        1,
        batch_size * num_key_value_heads,
        head_dim,
    )
    return FlexGenDecodeProjection(query=query, key=key, value=value)


def finish_flexgen_decode_attention(
    inputs: torch.Tensor,
    projection: FlexGenDecodeProjection,
    attention_output,
    weights: FlexGenAttentionWeights,
    epsilon: float = 1e-6,
) -> FlexGenDecodeAttentionOutput:
    if inputs.ndim != 3 or inputs.shape[1] != 1:
        raise ValueError("Decode inputs must have shape [batch, 1, hidden]")
    if not isinstance(projection, FlexGenDecodeProjection):
        raise TypeError("projection must be FlexGenDecodeProjection")
    if not isinstance(weights, FlexGenAttentionWeights):
        raise TypeError("weights must be FlexGenAttentionWeights")

    output = torch.as_tensor(
        attention_output,
        dtype=inputs.dtype,
        device=inputs.device,
    )
    expected_shape = projection.query.shape
    if output.shape != expected_shape:
        raise ValueError(
            f"attention output shape {tuple(output.shape)} does not match "
            f"{tuple(expected_shape)}"
        )

    batch_size = inputs.shape[0]
    output = output.reshape(batch_size, 1, -1)
    output = F.linear(output, weights.output, bias=None)
    hidden_states = output + inputs
    mlp_inputs = _rms_norm(weights.post_attention_norm, hidden_states, epsilon)
    return FlexGenDecodeAttentionOutput(
        hidden_states=hidden_states,
        mlp_inputs=mlp_inputs,
        query=projection.query,
        key=projection.key,
        value=projection.value,
        attention_output=torch.as_tensor(
            attention_output,
            dtype=inputs.dtype,
            device=inputs.device,
        ),
    )
