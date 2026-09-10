"""OPT adapter for shared multi-microbatch Prefill scheduling."""

from typing import Any, Callable, Optional

from runtime_common.multi_batch_prefill import ModelMultiBatchPrefillRunner

from .checkpoint import OptCheckpointLoader
from .config import OptConfig
from .ops import OptOperations


class OptMultiBatchPrefillRunner(ModelMultiBatchPrefillRunner):
    def __init__(
        self,
        config: OptConfig,
        weight_loader: OptCheckpointLoader,
        kv_writer: Any,
        gpu_batch_size: int,
        logger: Optional[Callable[[str], None]] = None,
    ):
        if not isinstance(config, OptConfig):
            raise TypeError("config must be an OptConfig")
        if not isinstance(weight_loader, OptCheckpointLoader):
            raise TypeError("weight_loader must be an OptCheckpointLoader")
        super().__init__(
            config,
            weight_loader,
            OptOperations(config),
            kv_writer,
            gpu_batch_size,
            logger=logger,
        )


__all__ = ["OptMultiBatchPrefillRunner"]
