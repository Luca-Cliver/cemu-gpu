from .kv_layout import KvCacheLayout, KvChunk, KvLayoutConfig, align_up
from .kv_store import KvCacheStore

__all__ = [
    "CemuDevice",
    "RangeSpec",
    "KvCacheLayout",
    "KvCacheStore",
    "KvChunk",
    "KvLayoutConfig",
    "align_up",
]


def __getattr__(name):
    if name in ("CemuDevice", "RangeSpec"):
        from .cemu_device import CemuDevice, RangeSpec

        globals()["CemuDevice"] = CemuDevice
        globals()["RangeSpec"] = RangeSpec
        return globals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
