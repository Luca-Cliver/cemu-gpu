"""OPT adapter for shared Prefill orchestration."""

from typing import Any, Callable, Optional

from runtime_common.prefill_runner import ModelPrefillResult, ModelPrefillRunner

from .checkpoint import OptCheckpointLoader
from .config import OptConfig
from .ops import OptOperations

OptPrefillResult = ModelPrefillResult


class OptPrefillRunner(ModelPrefillRunner):
    def __init__(
        self,
        config: OptConfig,
        weight_loader: OptCheckpointLoader,
        kv_writer: Optional[Any] = None,
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
            kv_writer=kv_writer,
            logger=logger,
        )


__all__ = ["OptPrefillResult", "OptPrefillRunner"]
