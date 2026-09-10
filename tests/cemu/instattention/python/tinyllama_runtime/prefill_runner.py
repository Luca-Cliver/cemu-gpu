"""TinyLlama adapter for shared Prefill orchestration."""

from typing import Any, Callable, Optional

from runtime_common.prefill_runner import ModelPrefillResult, ModelPrefillRunner

from .model_config import FlexGenLlamaConfig
from .ops import TinyLlamaOperations
from .weights import FlexGenWeightLoader

FlexGenFullPrefillResult = ModelPrefillResult


class FlexGenPrefillRunner(ModelPrefillRunner):
    def __init__(
        self,
        config: FlexGenLlamaConfig,
        weight_loader: FlexGenWeightLoader,
        kv_writer: Optional[Any] = None,
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
            kv_writer=kv_writer,
            logger=logger,
        )


__all__ = ["FlexGenFullPrefillResult", "FlexGenPrefillRunner"]
