from .kv_layout import KvCacheLayout, KvChunk, KvLayoutConfig, align_up
from .kv_staging import KvStagingManager
from .kv_store import KvCacheStore
from .attention_abi import DenseAttentionMetadata

__all__ = [
    "AttentionBufferConfig",
    "AttentionRange",
    "CemuAttentionDevice",
    "DenseAttentionMetadata",
    "CemuDevice",
    "RangeSpec",
    "KvCacheLayout",
    "KvCacheStore",
    "KvStagingManager",
    "KvChunk",
    "KvLayoutConfig",
    "align_up",
]


def __getattr__(name):
    if name in ("AttentionBufferConfig", "AttentionRange", "CemuAttentionDevice"):
        from .cemu_attention_device import (
            AttentionBufferConfig,
            AttentionRange,
            CemuAttentionDevice,
        )

        globals()["AttentionBufferConfig"] = AttentionBufferConfig
        globals()["AttentionRange"] = AttentionRange
        globals()["CemuAttentionDevice"] = CemuAttentionDevice
        return globals()[name]
    if name in ("CemuDevice", "RangeSpec"):
        from .cemu_device import CemuDevice, RangeSpec

        globals()["CemuDevice"] = CemuDevice
        globals()["RangeSpec"] = RangeSpec
        return globals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
