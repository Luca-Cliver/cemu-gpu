"""OPT compatibility wrapper for the shared PyTorch Attention backend."""

from typing import Callable, Optional, Sequence, Tuple

import torch

from runtime_common import TorchAttentionBackend

from .config import OptConfig


class OptTorchAttentionBackend(TorchAttentionBackend):
    def __init__(
        self,
        config: OptConfig,
        kv_cache: Sequence[Tuple[torch.Tensor, torch.Tensor]],
        logger: Optional[Callable[[str], None]] = None,
        copy_cache: bool = True,
        accumulation_dtype: Optional[torch.dtype] = None,
    ):
        if not isinstance(config, OptConfig):
            raise TypeError("config must be an OptConfig")
        super().__init__(
            config,
            kv_cache,
            logger=logger,
            copy_cache=copy_cache,
            attention_scale=1.0,
            accumulation_dtype=accumulation_dtype,
        )
