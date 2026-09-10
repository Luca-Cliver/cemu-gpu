"""TinyLlama implementation of the shared model-operation interface."""

from dataclasses import dataclass
from typing import Any, Optional, Sequence

import torch

from runtime_common import ModelEmbeddingResult

from .decode import (
    FlexGenDecodeAttentionOutput,
    finish_flexgen_decode_attention,
    prepare_flexgen_decode_attention,
)
from .embedding import run_flexgen_embedding
from .mlp import run_flexgen_mlp
from .model_config import FlexGenLlamaConfig
from .output import run_flexgen_output_head
from .prefill import run_flexgen_prefill
from .weights import FlexGenLayerWeights


@dataclass(frozen=True)
class TinyLlamaOperations:
    config: FlexGenLlamaConfig

    def embed(
        self,
        token_ids: torch.Tensor,
        embedding_weights: torch.Tensor,
        token_position: Optional[int] = None,
    ) -> ModelEmbeddingResult:
        hidden_states, attention_mask, position_ids = run_flexgen_embedding(
            token_ids,
            embedding_weights,
            self.config.pad_token_id,
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
        weights: FlexGenLayerWeights,
    ):
        attention = weights.attention
        return run_flexgen_prefill(
            inputs=inputs,
            attention_mask=attention_mask,
            query_weight=attention.query,
            key_weight=attention.key,
            value_weight=attention.value,
            output_weight=attention.output,
            input_norm_weight=attention.input_norm,
            post_attention_norm_weight=attention.post_attention_norm,
            num_heads=self.config.num_attention_heads,
            num_key_value_heads=self.config.num_key_value_heads,
            position_ids=position_ids,
            rope_theta=self.config.rope_theta,
            epsilon=self.config.rms_norm_epsilon,
        )

    def prepare_decode_attention(
        self,
        inputs: torch.Tensor,
        weights: FlexGenLayerWeights,
        token_position: int,
    ):
        return prepare_flexgen_decode_attention(
            inputs=inputs,
            weights=weights.attention,
            num_heads=self.config.num_attention_heads,
            num_key_value_heads=self.config.num_key_value_heads,
            token_position=token_position,
            rope_theta=self.config.rope_theta,
            epsilon=self.config.rms_norm_epsilon,
        )

    def finish_decode_attention(
        self,
        inputs: torch.Tensor,
        projection: Any,
        attention_output: Any,
        weights: FlexGenLayerWeights,
    ):
        return finish_flexgen_decode_attention(
            inputs=inputs,
            projection=projection,
            attention_output=attention_output,
            weights=weights.attention,
            epsilon=self.config.rms_norm_epsilon,
        )

    def run_mlp(
        self,
        inputs: torch.Tensor,
        weights: Any,
        residual: torch.Tensor,
    ) -> torch.Tensor:
        return run_flexgen_mlp(inputs, weights, residual=residual)

    def run_output_head(
        self,
        hidden_states: torch.Tensor,
        final_norm: torch.Tensor,
        lm_head: torch.Tensor,
        do_sample: bool,
        temperature: float,
    ):
        return run_flexgen_output_head(
            hidden_states=hidden_states,
            final_norm_weight=final_norm,
            lm_head_weight=lm_head,
            epsilon=self.config.rms_norm_epsilon,
            do_sample=do_sample,
            temperature=temperature,
        )

    @staticmethod
    def merge_decode_outputs(
        outputs: Sequence[FlexGenDecodeAttentionOutput],
    ) -> FlexGenDecodeAttentionOutput:
        if any(output is None for output in outputs):
            raise RuntimeError("not every microbatch produced an Attention output")
        return FlexGenDecodeAttentionOutput(
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
