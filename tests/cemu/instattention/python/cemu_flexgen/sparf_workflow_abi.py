import struct
from dataclasses import dataclass

import numpy as np

from .kv_layout import SparfKvCacheLayout

SPARF_WORKFLOW_COMMAND = 0x43454D5553505246
SPARF_WORKFLOW_TRACE_BYTES = 9 * 8
# Workflow stages are separate from CUDA kernel phase numbers.
SPARF_STAGE_SELECT = 1
SPARF_STAGE_TOKEN_READ = 2
SPARF_STAGE_EXACT_V = 3
SPARF_STAGE_PV = 4
SPARF_STAGES = (SPARF_STAGE_SELECT, SPARF_STAGE_TOKEN_READ,
                SPARF_STAGE_EXACT_V, SPARF_STAGE_PV)
SPARF_STAGE_NAMES = ("select", "token_read", "exact_v", "pv")
SPARF_CONTEXT_BYTES = 96
_HEADER = struct.Struct("<25If2I")
_EXTENT = struct.Struct("<QII")
_TRACE = struct.Struct("<9Q")
_BUFFER_TAIL = struct.Struct("<II")
_GROUP_BINDING = struct.Struct("<QQ")


def pack_grouped_sparf_workflow(layout, layer, valid_tokens, query_heads, top_r, top_k,
                                cookie, stream_base=0, approximate_runtime_ns=0,
                                exact_qk_runtime_ns=0, pv_runtime_ns=0, scale=1.0):
    config = layout.config
    if (not 0 <= layer < config.num_layers or not 0 < valid_tokens <= config.max_seq_len or
            query_heads <= 0 or query_heads % config.num_kv_heads or
            not 0 < top_r <= config.head_dim or not 0 < top_k <= valid_tokens or
            not 0 < cookie < 2**64 or stream_base < 0 or not np.isfinite(scale) or scale <= 0 or
            any(type(value) is not int or not 0 <= value <= 0xffffffff for value in
                (approximate_runtime_ns, exact_qk_runtime_ns, pv_runtime_ns))):
        raise ValueError("invalid grouped SparF workflow geometry or binding")
    first_stream = stream_base + layer * config.batch_size * config.num_kv_heads
    values = (3, 5, approximate_runtime_ns, exact_qk_runtime_ns, pv_runtime_ns,
              0, 0, 0, 2 if config.dtype == np.dtype(np.float16) else 1,
              config.batch_size, query_heads, config.num_kv_heads, config.head_dim,
              config.max_seq_len, valid_tokens, layer, top_r, top_k,
              0, 0, 0, 0, 0, 0, 0, float(scale), 0, 0)
    return _HEADER.pack(*values) + _GROUP_BINDING.pack(cookie, first_stream)


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
                        pv_runtime_ns=0, scale=1.0, trace=True, buffered_tokens=None,
                        autonomous=False):
    if not isinstance(layout, SparfKvCacheLayout):
        raise TypeError("layout must be a SparfKvCacheLayout")
    token_k = _extents("token K", token_k_extents, layout.token_file_size)
    channel_k = _extents("channel K", channel_k_extents, layout.channel_file_size)
    token_v = _extents("token V", token_v_extents, layout.token_file_size)
    dtype = 2 if layout.config.dtype == np.dtype(np.float16) else 1
    values = (
        3 if autonomous else 1, int(bool(trace)) | (2 if buffered_tokens is not None else 0), approximate_runtime_ns, exact_qk_runtime_ns,
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
    if buffered_tokens is not None:
        if type(buffered_tokens) is not int or not 0 <= buffered_tokens <= valid_tokens:
            raise ValueError("invalid buffered token count")
        payload.extend(_BUFFER_TAIL.pack(valid_tokens - buffered_tokens, buffered_tokens))
    return bytes(payload)


def stage_metadata(metadata, stage, workflow_id):
    """Encode a continuation using the existing execute metadata buffer."""
    if stage not in SPARF_STAGES:
        raise ValueError("invalid SparF stage")
    if not 0 < workflow_id <= 0xffffffff:
        raise ValueError("workflow id must be a nonzero uint32")
    fields = list(_HEADER.unpack_from(metadata))
    if fields[0] != 1 or fields[-2:] != [0, 0]:
        raise ValueError("expected canonical version-1 workflow metadata")
    fields[0] = 2
    fields[-2:] = [stage, workflow_id]
    return _HEADER.pack(*fields) + metadata[_HEADER.size:]


def stage_context_bytes(selected_tokens):
    # Reserve room for the maximum three extent tables. No CUDA mirror is
    # allocated for this host-only continuation range.
    return SPARF_CONTEXT_BYTES + _HEADER.size + 3 * 1024 * _EXTENT.size + _BUFFER_TAIL.size + selected_tokens * 4


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
