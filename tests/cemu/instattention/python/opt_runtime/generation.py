"""OPT exports for shared autoregressive generation."""

from runtime_common.generation import (
    ModelGenerationResult,
    ModelGenerationRunner,
    ModelGenerationStep,
)

OptGenerationResult = ModelGenerationResult
OptGenerationRunner = ModelGenerationRunner
OptGenerationStep = ModelGenerationStep

__all__ = [
    "OptGenerationResult",
    "OptGenerationRunner",
    "OptGenerationStep",
]
