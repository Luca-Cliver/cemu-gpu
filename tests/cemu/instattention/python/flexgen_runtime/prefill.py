import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class FlexGenPrefillOutput:
    hidden_states: torch.Tensor
    mlp_inputs: torch.Tensor
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
    num_key_value_heads: Optional[int] = None,
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
    if num_key_value_heads is None:
        num_key_value_heads = num_heads
    if (
        not isinstance(num_key_value_heads, int)
        or isinstance(num_key_value_heads, bool)
        or num_key_value_heads <= 0
        or num_heads % num_key_value_heads != 0
    ):
        raise ValueError("num_key_value_heads must divide num_heads")
    if attention_mask.shape != (batch_size, sequence_length):
        raise ValueError("attention_mask must have shape [batch, sequence]")
    if attention_mask.dtype != torch.bool:
        raise TypeError("attention_mask must use torch.bool")
    if not math.isfinite(float(rope_theta)) or rope_theta <= 0:
        raise ValueError("rope_theta must be finite and positive")

    head_dim = hidden_size // num_heads
    matrix_shape = (hidden_size, hidden_size)
    for name, weight in (
        ("query_weight", query_weight),
        ("output_weight", output_weight),
    ):
        if weight.shape != matrix_shape:
            raise ValueError(f"{name} must have shape {matrix_shape}")
    kv_matrix_shape = (num_key_value_heads * head_dim, hidden_size)
    for name, weight in (
        ("key_weight", key_weight),
        ("value_weight", value_weight),
    ):
        if weight.shape != kv_matrix_shape:
            raise ValueError(f"{name} must have shape {kv_matrix_shape}")
    for name, weight in (
        ("input_norm_weight", input_norm_weight),
        ("post_attention_norm_weight", post_attention_norm_weight),
    ):
        if weight.shape != (hidden_size,):
            raise ValueError(f"{name} must have shape ({hidden_size},)")

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
    key = key.view(
        batch_size,
        sequence_length,
        num_key_value_heads,
        head_dim,
    )
    value = value.view(
        batch_size,
        sequence_length,
        num_key_value_heads,
        head_dim,
    )
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
        batch_size * num_key_value_heads,
        head_dim,
    )
    flexgen_values = value.permute(2, 0, 1, 3).reshape(
        sequence_length,
        batch_size * num_key_value_heads,
        head_dim,
    )

    num_key_value_groups = num_heads // num_key_value_heads
    attention_key = key.repeat_interleave(num_key_value_groups, dim=1)
    attention_value = value.repeat_interleave(num_key_value_groups, dim=1)
    query = query.reshape(batch_size * num_heads, sequence_length, head_dim)
    attention_keys = attention_key.reshape(
        batch_size * num_heads,
        sequence_length,
        head_dim,
    ).transpose(1, 2)
    attention_values = attention_value.reshape(
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
    hidden_states = output + inputs
    mlp_inputs = _rms_norm(post_attention_norm_weight, hidden_states, epsilon)

    return FlexGenPrefillOutput(
        hidden_states=hidden_states,
        mlp_inputs=mlp_inputs,
        keys=flexgen_keys,
        values=flexgen_values,
    )
