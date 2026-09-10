"""TinyLlama adapter for shared Decode orchestration."""

from typing import Any, Callable, Optional

from runtime_common.decode_runner import ModelDecodeResult, ModelDecodeRunner

from .model_config import FlexGenLlamaConfig
from .ops import TinyLlamaOperations
from .weights import FlexGenWeightLoader

FlexGenDecodeResult = ModelDecodeResult


class FlexGenDecodeRunner(ModelDecodeRunner):
    def __init__(
        self,
        config: FlexGenLlamaConfig,
        weight_loader: FlexGenWeightLoader,
        attention_backend: Any,
        logger: Optional[Callable[[str], None]] = None,
    ):
        if not isinstance(config, FlexGenLlamaConfig):
            raise TypeError("config must be a FlexGenLlamaConfig")
        if not isinstance(weight_loader, FlexGenWeightLoader):
            raise TypeError("weight_loader must be a FlexGenWeightLoader")
        super().__init__(
            config,
            weight_loader,
            TinyLlamaOperations(config),
            attention_backend,
            logger=logger,
        )


__all__ = ["FlexGenDecodeResult", "FlexGenDecodeRunner"]
