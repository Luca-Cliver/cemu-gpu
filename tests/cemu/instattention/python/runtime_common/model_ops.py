"""Interfaces between model-specific tensor math and shared scheduling."""

from dataclasses import dataclass
from typing import Any, Optional, Protocol, Sequence

import torch


@dataclass(frozen=True)
class ModelEmbeddingResult:
    hidden_states: torch.Tensor
    attention_mask: torch.Tensor
    position_ids: torch.Tensor


class ModelOperations(Protocol):
    def embed(
        self,
        token_ids: torch.Tensor,
        embedding_weights: Any,
        token_position: Optional[int] = None,
    ) -> ModelEmbeddingResult:
        ...

    def prefill_attention(
        self,
        inputs: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        weights: Any,
    ) -> Any:
        ...

    def prepare_decode_attention(
        self,
        inputs: torch.Tensor,
        weights: Any,
        token_position: int,
    ) -> Any:
        ...

    def finish_decode_attention(
        self,
        inputs: torch.Tensor,
        projection: Any,
        attention_output: Any,
        weights: Any,
    ) -> Any:
        ...

    def run_mlp(
        self,
        inputs: torch.Tensor,
        weights: Any,
        residual: torch.Tensor,
    ) -> torch.Tensor:
        ...

    def run_output_head(
        self,
        hidden_states: torch.Tensor,
        final_norm: Any,
        lm_head: torch.Tensor,
        do_sample: bool,
        temperature: float,
    ) -> Any:
        ...

    def merge_decode_outputs(self, outputs: Sequence[Any]) -> Any:
        ...


def validate_model_operations(operations: Any) -> None:
    required_methods = (
        "embed",
        "prefill_attention",
        "prepare_decode_attention",
        "finish_decode_attention",
        "run_mlp",
        "run_output_head",
        "merge_decode_outputs",
    )
    missing = [
        method_name
        for method_name in required_methods
        if not callable(getattr(operations, method_name, None))
    ]
    if missing:
        raise TypeError(
            "operations must provide: " + ", ".join(required_methods)
        )
