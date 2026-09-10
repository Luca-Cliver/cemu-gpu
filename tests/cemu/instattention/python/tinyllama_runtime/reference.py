"""TinyLlama compatibility wrapper for the shared PyTorch Attention backend."""

from typing import Callable, Optional, Sequence, Tuple

import torch

from runtime_common import TorchAttentionBackend

from .model_config import FlexGenLlamaConfig


class FlexGenTorchAttentionBackend(TorchAttentionBackend):
    def __init__(
        self,
        config: FlexGenLlamaConfig,
        kv_cache: Sequence[Tuple[torch.Tensor, torch.Tensor]],
        logger: Optional[Callable[[str], None]] = None,
        copy_cache: bool = True,
    ):
        if not isinstance(config, FlexGenLlamaConfig):
            raise TypeError("config must be a FlexGenLlamaConfig")
        super().__init__(config, kv_cache, logger=logger, copy_cache=copy_cache)
