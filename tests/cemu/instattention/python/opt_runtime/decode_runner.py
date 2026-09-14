"""OPT adapter for shared Decode orchestration."""

from typing import Any, Callable, Optional

from runtime_common.decode_runner import ModelDecodeResult, ModelDecodeRunner

from .checkpoint import OptCheckpointLoader
from .config import OptConfig
from .ops import OptOperations

OptDecodeResult = ModelDecodeResult


class OptDecodeRunner(ModelDecodeRunner):
    def __init__(
        self,
        config: OptConfig,
        weight_loader: OptCheckpointLoader,
        attention_backend: Any,
        logger: Optional[Callable[[str], None]] = None,
    ):
        if not isinstance(config, OptConfig):
            raise TypeError("config must be an OptConfig")
        if not isinstance(weight_loader, OptCheckpointLoader):
            raise TypeError("weight_loader must be an OptCheckpointLoader")
        super().__init__(
            config,
            weight_loader,
            OptOperations(config, profiler=getattr(attention_backend, "profiler", None)),
            attention_backend,
            logger=logger,
        )


__all__ = ["OptDecodeResult", "OptDecodeRunner"]
