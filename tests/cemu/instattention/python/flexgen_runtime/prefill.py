import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class FlexGenPrefillOutput:
    hidden_states: torch.Tensor
    keys: torch.Tensor
    values: torch.Tensor


def _rms_norm(
    weight: torch.Tensor,
    hidden_states: torch.Tensor,
    epsilon: float,
) -> torch.Tensor:
    input_dtype = hidden_states.dtype
    normalized = hidden_states.to(torch.float32)
    variance = normalized.pow(2).mean(-1, keepdim=True)
    normalized = normalized * torch.rsqrt(variance + epsilon)
    return weight * normalized.to(input_dtype)


def _rotate_half(tensor: torch.Tensor) -> torch.Tensor:
    first_half = tensor[..., : tensor.shape[-1] // 2]
    second_half = tensor[..., tensor.shape[-1] // 2 :]
    return torch.cat((-second_half, first_half), dim=-1)


def _apply_rotary_pos_emb(
    query: torch.Tensor,
    key: torch.Tensor,
    cosine: torch.Tensor,
    sine: torch.Tensor,
    position_ids: torch.Tensor,
):
    cosine = cosine[position_ids].unsqueeze(1)
    sine = sine[position_ids].unsqueeze(1)
    return (
        query * cosine + _rotate_half(query) * sine,
        key * cosine + _rotate_half(key) * sine,
    )


def _rotary_embedding(
    head_dim: int,
    sequence_length: int,
    dtype: torch.dtype,
    device: torch.device,
    rope_theta: float,
):
    inverse_frequency = 1.0 / (
        rope_theta
        ** (
            torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
            / head_dim
        )
    )
    positions = torch.arange(
        sequence_length,
        dtype=inverse_frequency.dtype,
        device=device,
    )
    frequencies = torch.outer(positions, inverse_frequency)
    embedding = torch.cat((frequencies, frequencies), dim=-1)
    return embedding.cos().to(dtype), embedding.sin().to(dtype)


def run_flexgen_prefill(
    inputs: torch.Tensor,
    attention_mask: torch.Tensor,
    query_weight: torch.Tensor,
    key_weight: torch.Tensor,
    value_weight: torch.Tensor,
    output_weight: torch.Tensor,
    input_norm_weight: torch.Tensor,
    post_attention_norm_weight: torch.Tensor,
    num_heads: int,
    position_ids: Optional[torch.Tensor] = None,
    rope_theta: float = 10000.0,
    epsilon: float = 1e-6,
    use_rotary_embedding: bool = True,
) -> FlexGenPrefillOutput:
    """Run the Prefill path extracted from FlexGen's TorchDevice.mha()."""
    if inputs.ndim != 3:
        raise ValueError("inputs must have shape [batch, sequence, hidden]")
    batch_size, sequence_length, hidden_size = inputs.shape
    if num_heads <= 0 or hidden_size % num_heads != 0:
        raise ValueError("num_heads must divide the hidden size")
    if attention_mask.shape != (batch_size, sequence_length):
        raise ValueError("attention_mask must have shape [batch, sequence]")
    if attention_mask.dtype != torch.bool:
        raise TypeError("attention_mask must use torch.bool")
    if not math.isfinite(float(rope_theta)) or rope_theta <= 0:
        raise ValueError("rope_theta must be finite and positive")

    matrix_shape = (hidden_size, hidden_size)
    for name, weight in (
        ("query_weight", query_weight),
        ("key_weight", key_weight),
        ("value_weight", value_weight),
        ("output_weight", output_weight),
    ):
        if weight.shape != matrix_shape:
            raise ValueError(f"{name} must have shape {matrix_shape}")
    for name, weight in (
        ("input_norm_weight", input_norm_weight),
        ("post_attention_norm_weight", post_attention_norm_weight),
    ):
        if weight.shape != (hidden_size,):
            raise ValueError(f"{name} must have shape ({hidden_size},)")

    head_dim = hidden_size // num_heads
    if use_rotary_embedding and head_dim % 2 != 0:
        raise ValueError("RoPE requires an even head dimension")
    if position_ids is None:
        position_ids = torch.arange(
            sequence_length,
            dtype=torch.long,
            device=inputs.device,
        ).expand(batch_size, -1)
    if position_ids.shape != (batch_size, sequence_length):
        raise ValueError("position_ids must have shape [batch, sequence]")

    hidden = _rms_norm(input_norm_weight, inputs, epsilon)
    scaling = head_dim ** -0.5
    query = F.linear(hidden, query_weight, bias=None) * scaling
    key = F.linear(hidden, key_weight, bias=None)
    value = F.linear(hidden, value_weight, bias=None)

    query = query.view(batch_size, sequence_length, num_heads, head_dim)
    key = key.view(batch_size, sequence_length, num_heads, head_dim)
    value = value.view(batch_size, sequence_length, num_heads, head_dim)
    query = query.permute(0, 2, 1, 3)
    key = key.permute(0, 2, 1, 3)
    value = value.permute(0, 2, 1, 3)

    if use_rotary_embedding:
        cosine, sine = _rotary_embedding(
            head_dim,
            sequence_length,
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

    flexgen_keys = key.permute(2, 0, 1, 3).reshape(
        sequence_length,
        batch_size * num_heads,
        head_dim,
    )
    flexgen_values = value.permute(2, 0, 1, 3).reshape(
        sequence_length,
        batch_size * num_heads,
        head_dim,
    )

    query = query.reshape(batch_size * num_heads, sequence_length, head_dim)
    attention_keys = key.reshape(
        batch_size * num_heads,
        sequence_length,
        head_dim,
    ).transpose(1, 2)
    attention_values = value.reshape(
        batch_size * num_heads,
        sequence_length,
        head_dim,
    )
    attention_weights = torch.bmm(query, attention_keys)

    token_indices = torch.arange(sequence_length, device=inputs.device)
    causal_mask = (token_indices <= token_indices.view(sequence_length, 1)).view(
        1,
        1,
        sequence_length,
        sequence_length,
    )
    combined_mask = attention_mask.view(
        batch_size,
        1,
        1,
        sequence_length,
    ) & causal_mask
    attention_weights = attention_weights.view(
        batch_size,
        num_heads,
        sequence_length,
        sequence_length,
    )
    attention_weights = torch.where(combined_mask, attention_weights, -1e4)
    attention_weights = attention_weights.view(
        batch_size * num_heads,
        sequence_length,
        sequence_length,
    )
    attention_weights = F.softmax(attention_weights, dim=2)

    output = torch.bmm(attention_weights, attention_values).view(
        batch_size,
        num_heads,
        sequence_length,
        head_dim,
    )
    output = output.transpose(1, 2).reshape(
        batch_size,
        sequence_length,
        hidden_size,
    )
    output = F.linear(output, output_weight, bias=None)
    output = output + inputs
    output = _rms_norm(post_attention_norm_weight, output, epsilon)

    return FlexGenPrefillOutput(output, flexgen_keys, flexgen_values)
