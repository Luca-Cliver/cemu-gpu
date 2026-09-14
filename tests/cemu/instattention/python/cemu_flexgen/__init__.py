from .kv_layout import KvCacheLayout, KvChunk, KvLayoutConfig, SparfKvCacheLayout, align_up
from .kv_staging import KvStagingManager
from .kv_store import KvCacheStore
from .sparf_kv_store import SparfKvCacheStore
from .sparf import SparfResult, sparf_attention
from .attention_abi import DenseAttentionMetadata
from .cemu_attention_slot_scheduler import (
    CemuAttentionSharedWorkers,
    CemuAttentionPrefetchRequest,
    CemuAttentionSlotRequest,
    CemuAttentionSlotScheduler,
)

__all__ = [
    "AttentionBufferConfig",
    "AttentionRange",
    "CemuAttentionDevice",
    "CemuAttentionPrefetchRequest",
    "CemuAttentionSharedWorkers",
    "CemuAttentionSlotRequest",
    "CemuAttentionSlotScheduler",
    "DenseAttentionMetadata",
    "CemuDevice",
    "RangeSpec",
    "KvCacheLayout",
    "KvCacheStore",
    "SparfKvCacheLayout",
    "SparfKvCacheStore",
    "SparfResult",
    "sparf_attention",
    "CemuSparfDevice",
    "KvStagingManager",
    "KvChunk",
    "KvLayoutConfig",
    "align_up",
]


def __getattr__(name):
    if name == "CemuSparfDevice":
        from .cemu_sparf_device import CemuSparfDevice

        globals()[name] = CemuSparfDevice
        return CemuSparfDevice
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
