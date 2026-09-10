"""OPT tensor operations used by the shared runtime schedulers."""

from dataclasses import dataclass
from typing import Any, Optional, Sequence

import torch
import torch.nn.functional as F

from runtime_common import ModelEmbeddingResult

from .config import OptConfig
from .weights import (
    OptEmbeddingWeights,
    OptLayerNormWeights,
    OptLayerWeights,
    OptMlpWeights,
)


@dataclass(frozen=True)
class OptPrefillOutput:
    hidden_states: torch.Tensor
    mlp_inputs: torch.Tensor
    keys: torch.Tensor
    values: torch.Tensor


@dataclass(frozen=True)
class OptDecodeProjection:
    query: torch.Tensor
    key: torch.Tensor
    value: torch.Tensor


@dataclass(frozen=True)
class OptDecodeAttentionOutput:
    hidden_states: torch.Tensor
    mlp_inputs: torch.Tensor
    query: torch.Tensor
    key: torch.Tensor
    value: torch.Tensor
    attention_output: torch.Tensor


@dataclass(frozen=True)
class OptOutputHeadResult:
    logits: torch.Tensor
    next_token_ids: torch.Tensor


@dataclass(frozen=True)
class OptOperations:
    config: OptConfig

    def embed(
        self,
        token_ids: torch.Tensor,
        embedding_weights: OptEmbeddingWeights,
        token_position: Optional[int] = None,
    ) -> ModelEmbeddingResult:
        if token_ids.ndim != 2:
            raise ValueError("token_ids must have shape [batch, sequence]")
        if token_ids.dtype not in (torch.int32, torch.int64):
            raise TypeError("token_ids must use an integer dtype")
        if not isinstance(embedding_weights, OptEmbeddingWeights):
            raise TypeError("embedding_weights must be OptEmbeddingWeights")

        attention_mask = token_ids.ne(self.config.pad_token_id)
        if token_position is None:
            if token_ids.shape[1] > self.config.max_position_embeddings:
                raise ValueError("sequence exceeds the OPT position embedding")
            mask_values = attention_mask.to(torch.long)
            position_ids = mask_values.cumsum(dim=1) * mask_values - 1
        else:
            if not isinstance(token_position, int) or isinstance(token_position, bool):
                raise TypeError("token_position must be an integer")
            if token_position < 0:
                raise ValueError("token_position must be non-negative")
            if token_position >= self.config.max_position_embeddings:
                raise ValueError("token position exceeds the OPT position embedding")
            position_ids = torch.full_like(token_ids, token_position)

        position_indices = position_ids + self.config.position_offset
        hidden_states = F.embedding(
            token_ids,
            embedding_weights.token,
            padding_idx=self.config.pad_token_id,
        )
        hidden_states = hidden_states + F.embedding(
            position_indices,
            embedding_weights.position,
        )
        return ModelEmbeddingResult(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )

    def prefill_attention(
        self,
        inputs: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        weights: OptLayerWeights,
    ) -> OptPrefillOutput:
        if inputs.ndim != 3:
            raise ValueError("inputs must have shape [batch, sequence, hidden]")
        if not isinstance(weights, OptLayerWeights):
            raise TypeError("weights must be OptLayerWeights")
        attention = weights.attention
        batch_size, sequence_length, hidden_size = inputs.shape
        if attention_mask.shape != (batch_size, sequence_length):
            raise ValueError("attention_mask must have shape [batch, sequence]")
        if position_ids.shape != (batch_size, sequence_length):
            raise ValueError("position_ids must have shape [batch, sequence]")

        normalized = self._layer_norm(inputs, attention.input_norm)
        query = F.linear(normalized, attention.query, attention.query_bias)
        query = query * (self.config.head_dim ** -0.5)
        key = F.linear(normalized, attention.key, attention.key_bias)
        value = F.linear(normalized, attention.value, attention.value_bias)

        query = query.view(
            batch_size,
            sequence_length,
            self.config.num_attention_heads,
            self.config.head_dim,
        ).transpose(1, 2)
        key = key.view(
            batch_size,
            sequence_length,
            self.config.num_attention_heads,
            self.config.head_dim,
        ).transpose(1, 2)
        value = value.view(
            batch_size,
            sequence_length,
            self.config.num_attention_heads,
            self.config.head_dim,
        ).transpose(1, 2)

        flexgen_keys = key.permute(2, 0, 1, 3).reshape(
            sequence_length,
            batch_size * self.config.num_attention_heads,
            self.config.head_dim,
        )
        flexgen_values = value.permute(2, 0, 1, 3).reshape(
            sequence_length,
            batch_size * self.config.num_attention_heads,
            self.config.head_dim,
        )

        query = query.reshape(
            batch_size * self.config.num_attention_heads,
            sequence_length,
            self.config.head_dim,
        )
        attention_keys = key.reshape(
            batch_size * self.config.num_attention_heads,
            sequence_length,
            self.config.head_dim,
        ).transpose(1, 2)
        attention_values = value.reshape(
            batch_size * self.config.num_attention_heads,
            sequence_length,
            self.config.head_dim,
        )
        attention_weights = torch.bmm(query, attention_keys).view(
            batch_size,
            self.config.num_attention_heads,
            sequence_length,
            sequence_length,
        )
        token_indices = torch.arange(sequence_length, device=inputs.device)
        causal_mask = (
            token_indices <= token_indices.view(sequence_length, 1)
        ).view(1, 1, sequence_length, sequence_length)
        combined_mask = attention_mask.view(
            batch_size,
            1,
            1,
            sequence_length,
        ) & causal_mask
        attention_weights = torch.where(combined_mask, attention_weights, -1e4)
        attention_weights = attention_weights.view(
            batch_size * self.config.num_attention_heads,
            sequence_length,
            sequence_length,
        )
        if attention_weights.dtype == torch.float16:
            probabilities = F.softmax(
                attention_weights,
                dim=-1,
                dtype=torch.float32,
            ).to(torch.float16)
        else:
            probabilities = F.softmax(attention_weights, dim=-1)

        output = torch.bmm(probabilities, attention_values).view(
            batch_size,
            self.config.num_attention_heads,
            sequence_length,
            self.config.head_dim,
        )
        output = output.transpose(1, 2).reshape(
            batch_size,
            sequence_length,
            hidden_size,
        )
        output = F.linear(output, attention.output, attention.output_bias)
        hidden_states = inputs + output
        mlp_inputs = self._layer_norm(hidden_states, weights.mlp.input_norm)
        return OptPrefillOutput(
            hidden_states=hidden_states,
            mlp_inputs=mlp_inputs,
            keys=flexgen_keys,
            values=flexgen_values,
        )

    def prepare_decode_attention(
        self,
        inputs: torch.Tensor,
        weights: OptLayerWeights,
        token_position: int,
    ) -> OptDecodeProjection:
        if inputs.ndim != 3 or inputs.shape[1] != 1:
            raise ValueError("Decode inputs must have shape [batch, 1, hidden]")
        if not isinstance(weights, OptLayerWeights):
            raise TypeError("weights must be OptLayerWeights")
        attention = weights.attention
        batch_size = inputs.shape[0]
        normalized = self._layer_norm(inputs, attention.input_norm)
        query = F.linear(normalized, attention.query, attention.query_bias)
        query = query * (self.config.head_dim ** -0.5)
        key = F.linear(normalized, attention.key, attention.key_bias)
        value = F.linear(normalized, attention.value, attention.value_bias)
        query = query.view(
            batch_size,
            self.config.num_attention_heads,
            self.config.head_dim,
        ).contiguous()
        key = key.view(
            batch_size,
            self.config.num_attention_heads,
            self.config.head_dim,
        ).reshape(1, batch_size * self.config.num_attention_heads, -1)
        value = value.view(
            batch_size,
            self.config.num_attention_heads,
            self.config.head_dim,
        ).reshape(1, batch_size * self.config.num_attention_heads, -1)
        return OptDecodeProjection(query=query, key=key, value=value)

    def finish_decode_attention(
        self,
        inputs: torch.Tensor,
        projection: OptDecodeProjection,
        attention_output: Any,
        weights: OptLayerWeights,
    ) -> OptDecodeAttentionOutput:
        attention = weights.attention
        attention_tensor = torch.as_tensor(
            attention_output,
            dtype=inputs.dtype,
            device=inputs.device,
        )
        if attention_tensor.shape != projection.query.shape:
            raise ValueError(
                f"attention output shape {tuple(attention_tensor.shape)} does not match "
                f"{tuple(projection.query.shape)}"
            )
        output = attention_tensor.reshape(inputs.shape[0], 1, self.config.hidden_size)
        output = F.linear(output, attention.output, attention.output_bias)
        hidden_states = inputs + output
        mlp_inputs = self._layer_norm(hidden_states, weights.mlp.input_norm)
        return OptDecodeAttentionOutput(
            hidden_states=hidden_states,
            mlp_inputs=mlp_inputs,
            query=projection.query,
            key=projection.key,
            value=projection.value,
            attention_output=attention_tensor,
        )

    def run_mlp(
        self,
        inputs: torch.Tensor,
        weights: OptMlpWeights,
        residual: torch.Tensor,
    ) -> torch.Tensor:
        if not isinstance(weights, OptMlpWeights):
            raise TypeError("weights must be OptMlpWeights")
        hidden = F.linear(inputs, weights.input, weights.input_bias)
        hidden = F.relu(hidden)
        hidden = F.linear(hidden, weights.output, weights.output_bias)
        return residual + hidden

    def run_output_head(
        self,
        hidden_states: torch.Tensor,
        final_norm: OptLayerNormWeights,
        lm_head: torch.Tensor,
        do_sample: bool,
        temperature: float,
    ) -> OptOutputHeadResult:
        hidden = self._layer_norm(hidden_states, final_norm)
        logits = F.linear(hidden, lm_head, bias=None)
        last_token_logits = logits[:, -1, :]
        if do_sample and temperature >= 1e-5:
            probabilities = torch.softmax(last_token_logits / temperature, dim=-1)
            next_token_ids = torch.multinomial(probabilities, num_samples=1)
        else:
            next_token_ids = last_token_logits.argmax(dim=1, keepdim=True)
        return OptOutputHeadResult(logits=logits, next_token_ids=next_token_ids)

    @staticmethod
    def merge_decode_outputs(
        outputs: Sequence[OptDecodeAttentionOutput],
    ) -> OptDecodeAttentionOutput:
        if any(output is None for output in outputs):
            raise RuntimeError("not every microbatch produced an Attention output")
        return OptDecodeAttentionOutput(
            hidden_states=torch.cat(
                [output.hidden_states for output in outputs],
                dim=0,
            ),
            mlp_inputs=torch.cat(
                [output.mlp_inputs for output in outputs],
                dim=0,
            ),
            query=torch.cat([output.query for output in outputs], dim=0),
            key=torch.cat([output.key for output in outputs], dim=1),
            value=torch.cat([output.value for output in outputs], dim=1),
            attention_output=torch.cat(
                [output.attention_output for output in outputs],
                dim=0,
            ),
        )

    def _layer_norm(
        self,
        hidden_states: torch.Tensor,
        weights: OptLayerNormWeights,
    ) -> torch.Tensor:
        return F.layer_norm(
            hidden_states,
            (self.config.hidden_size,),
            weights.weight,
            weights.bias,
            self.config.layer_norm_epsilon,
        )
