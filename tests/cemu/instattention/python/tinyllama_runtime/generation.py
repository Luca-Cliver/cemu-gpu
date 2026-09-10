"""Compatibility exports for shared autoregressive generation."""

from runtime_common.generation import (
    ModelGenerationResult,
    ModelGenerationRunner,
    ModelGenerationStep,
)

FlexGenGenerationResult = ModelGenerationResult
FlexGenGenerationRunner = ModelGenerationRunner
FlexGenGenerationStep = ModelGenerationStep

__all__ = [
    "FlexGenGenerationResult",
    "FlexGenGenerationRunner",
    "FlexGenGenerationStep",
]
