"""OPT adapter for shared multi-microbatch Decode scheduling."""

from typing import Any, Callable, Optional, Sequence

from runtime_common.multi_batch_decode import ModelMultiBatchDecodeRunner

from .checkpoint import OptCheckpointLoader
from .config import OptConfig
from .ops import OptOperations


class OptMultiBatchDecodeRunner(ModelMultiBatchDecodeRunner):
    def __init__(
        self,
        config: OptConfig,
        weight_loader: OptCheckpointLoader,
        attention_backends: Sequence[Any],
        gpu_batch_size: int,
        logger: Optional[Callable[[str], None]] = None,
    ):
        if not isinstance(config, OptConfig):
            raise TypeError("config must be an OptConfig")
        if not isinstance(weight_loader, OptCheckpointLoader):
            raise TypeError("weight_loader must be an OptCheckpointLoader")
        attention_backends = tuple(attention_backends)
        if not attention_backends:
            raise ValueError("attention_backends must not be empty")
        super().__init__(
            config,
            weight_loader,
            OptOperations(config, profiler=getattr(attention_backends[0], "profiler", None)),
            attention_backends,
            gpu_batch_size,
            logger=logger,
        )


__all__ = ["OptMultiBatchDecodeRunner"]
