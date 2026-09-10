"""TinyLlama adapter for shared multi-microbatch Decode scheduling."""

from typing import Any, Callable, Optional, Sequence

from runtime_common.multi_batch_decode import ModelMultiBatchDecodeRunner

from .model_config import FlexGenLlamaConfig
from .ops import TinyLlamaOperations
from .weights import FlexGenWeightLoader


class FlexGenMultiBatchDecodeRunner(ModelMultiBatchDecodeRunner):
    def __init__(
        self,
        config: FlexGenLlamaConfig,
        weight_loader: FlexGenWeightLoader,
        attention_backends: Sequence[Any],
        gpu_batch_size: int,
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
            attention_backends,
            gpu_batch_size,
            logger=logger,
        )


__all__ = ["FlexGenMultiBatchDecodeRunner"]
