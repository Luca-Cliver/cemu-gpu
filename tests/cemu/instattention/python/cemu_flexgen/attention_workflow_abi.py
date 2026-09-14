import struct
from dataclasses import dataclass

from .attention_phases_abi import AttentionPhaseMetadata, QK_SOFTMAX


ATTENTION_WORKFLOW_COMMAND = 0x43454D5550484153
ATTENTION_WORKFLOW_TRACE = 1
ATTENTION_WORKFLOW_SERIAL = 2
_HEADER = struct.Struct("<8I")
_EXTENT = struct.Struct("<QII")
_TRACE = struct.Struct("<14Q")


def _normalize_extents(name, extents, expected_bytes):
    normalized = []
    total_bytes = 0
    for extent in extents:
        if len(extent) != 2:
            raise ValueError(f"{name} extents must contain (slba, nlb) pairs")
        slba, nlb = extent
        if type(slba) is not int or type(nlb) is not int or slba < 0 or not 0 < nlb <= 65535:
            raise ValueError(f"invalid {name} extent")
        normalized.append((slba, nlb))
        total_bytes += nlb * 512
    if not normalized or len(normalized) > 1024:
        raise ValueError(f"{name} must contain between 1 and 1024 extents")
    if total_bytes != expected_bytes:
        raise ValueError(f"{name} extents cover {total_bytes} bytes, expected {expected_bytes}")
    return tuple(normalized)


def pack_attention_workflow(config, key_extents, value_extents,
                            qk_runtime_ns=0, pv_runtime_ns=0,
                            serial=False, trace=True):
    if not isinstance(config, AttentionPhaseMetadata):
        raise TypeError("config must be AttentionPhaseMetadata")
    for name, value in (("qk_runtime_ns", qk_runtime_ns), ("pv_runtime_ns", pv_runtime_ns)):
        if type(value) is not int or not 0 <= value <= 0xFFFFFFFF:
            raise ValueError(f"{name} must fit uint32")
    keys = _normalize_extents("key", key_extents, config.kv_bytes)
    values = _normalize_extents("value", value_extents, config.kv_bytes)
    flags = (ATTENTION_WORKFLOW_TRACE if trace else 0) | (ATTENTION_WORKFLOW_SERIAL if serial else 0)
    payload = bytearray(_HEADER.pack(
        1, flags, qk_runtime_ns, pv_runtime_ns,
        len(keys), len(values), 0, 0,
    ))
    payload.extend(config.pack(QK_SOFTMAX))
    for slba, nlb in keys + values:
        payload.extend(_EXTENT.pack(slba, nlb, 0))
    return bytes(payload)


@dataclass(frozen=True)
class AttentionWorkflowTrace:
    key_submit_ns: int
    key_ready_ns: int
    value_submit_ns: int
    qk_start_ns: int
    qk_done_ns: int
    value_ready_ns: int
    pv_start_ns: int
    pv_done_ns: int
    finish_ns: int
    key_model_ns: int
    value_model_ns: int
    qk_model_ns: int
    pv_model_ns: int
    total_model_ns: int

    @classmethod
    def unpack(cls, value):
        view = memoryview(value)
        if view.nbytes < _TRACE.size:
            raise ValueError("Attention workflow trace is truncated")
        return cls(*_TRACE.unpack_from(view))

    @property
    def qk_value_actual_overlap_ns(self):
        return max(
            0,
            min(self.qk_done_ns, self.value_ready_ns) -
            max(self.qk_start_ns, self.value_submit_ns),
        )


ATTENTION_WORKFLOW_TRACE_BYTES = _TRACE.size
