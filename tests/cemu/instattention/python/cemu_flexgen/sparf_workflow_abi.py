import struct
from dataclasses import dataclass

import numpy as np

from .kv_layout import SparfKvCacheLayout

SPARF_WORKFLOW_COMMAND = 0x43454D5553505246
SPARF_WORKFLOW_TRACE_BYTES = 9 * 8
_HEADER = struct.Struct("<25If2I")
_EXTENT = struct.Struct("<QII")
_TRACE = struct.Struct("<9Q")


def _extents(name, values, expected_bytes):
    result = tuple((int(slba), int(nlb)) for slba, nlb in values)
    if not result or len(result) > 1024:
        raise ValueError(f"{name} requires 1..1024 extents")
    if sum(nlb * 512 for _, nlb in result) != expected_bytes:
        raise ValueError(f"{name} extents do not cover the complete file")
    return result


def pack_sparf_workflow(layout, layer, valid_tokens, query_heads, top_r, top_k,
                        token_k_extents, channel_k_extents, token_v_extents,
                        approximate_runtime_ns=0, exact_qk_runtime_ns=0,
                        pv_runtime_ns=0, scale=1.0, trace=True):
    if not isinstance(layout, SparfKvCacheLayout):
        raise TypeError("layout must be a SparfKvCacheLayout")
    token_k = _extents("token K", token_k_extents, layout.token_file_size)
    channel_k = _extents("channel K", channel_k_extents, layout.channel_file_size)
    token_v = _extents("token V", token_v_extents, layout.token_file_size)
    dtype = 2 if layout.config.dtype == np.dtype(np.float16) else 1
    values = (
        1, int(bool(trace)), approximate_runtime_ns, exact_qk_runtime_ns,
        pv_runtime_ns, len(token_k), len(channel_k), len(token_v), dtype,
        layout.config.batch_size, query_heads, layout.config.num_kv_heads,
        layout.config.head_dim, layout.config.max_seq_len, valid_tokens, layer,
        top_r, top_k, layout.token_head_stride, layout.token_batch_stride,
        layout.token_layer_stride, layout.channel_stride,
        layout.channel_head_stride, layout.channel_batch_stride,
        layout.channel_layer_stride, float(scale), 0, 0,
    )
    payload = bytearray(_HEADER.pack(*values))
    for slba, nlb in token_k + channel_k + token_v:
        if slba < 0 or not 0 < nlb <= 65535:
            raise ValueError("invalid file extent")
        payload.extend(_EXTENT.pack(slba, nlb, 0))
    return bytes(payload)


@dataclass(frozen=True)
class SparfWorkflowTrace:
    channel_read_model_ns: int
    approximate_model_ns: int
    token_k_read_model_ns: int
    token_v_read_model_ns: int
    exact_qk_model_ns: int
    pv_model_ns: int
    total_model_ns: int
    selected_channels: int
    selected_tokens: int

    @classmethod
    def unpack(cls, value):
        return cls(*_TRACE.unpack_from(memoryview(value)))
