"""Capacity planning for the device-owned grouped SparF store.

The planner is deliberately independent from CEMU handles.  It lets the OPT
driver reject an impossible run before loading a checkpoint or creating large
FDM/NVM mappings.
"""

from dataclasses import dataclass
import shutil

from .kv_layout import align_up
from .sparf_workflow_abi import SPARF_WORKFLOW_TRACE_BYTES


# These bounds keep the C-side page arrays and directory metadata bounded
# while covering the largest documented single-CSD OPT-13B run (9.4 GiB).
GROUP_MAX_PAGES = 1 << 24
GROUP_MAX_STREAMS = 1 << 20
GROUP_IO_BYTES = 1 << 20
GROUP_STATUS_BYTES = 4096
FDM_ALLOCATION_ALIGNMENT = 4096
GROUP_MAX_WORKSPACE_BYTES = 64 << 20


def sparf_attention_range_bytes(*, batch_size, query_heads, kv_heads, head_dim,
                                element_bytes, max_seq_len, top_r,
                                compression_ratio, buffer_tokens=0):
    """Unaligned MR sizes shared by allocation and grouped capacity planning."""
    values = (batch_size, query_heads, kv_heads, head_dim, element_bytes,
              max_seq_len, top_r, compression_ratio)
    if any(type(value) is not int or value <= 0 for value in values):
        raise ValueError("Attention geometry must contain positive integers")
    if query_heads % kv_heads or type(buffer_tokens) is not int or buffer_tokens < 0:
        raise ValueError("invalid Attention head grouping or buffer_tokens")
    vectors = batch_size * query_heads
    maximum_k = (max_seq_len + compression_ratio - 1) // compression_ratio
    current_bytes = max(1, buffer_tokens) * batch_size * kv_heads * head_dim * element_bytes
    return (
        vectors * head_dim * element_bytes,
        vectors * min(top_r, head_dim) * max_seq_len * element_bytes,
        vectors * maximum_k * head_dim * element_bytes,
        vectors * (maximum_k + 1) * head_dim * element_bytes,
        vectors * head_dim * element_bytes,
        current_bytes,
        current_bytes,
        vectors * 4,
        vectors * (maximum_k + 1) * 4,
        vectors * head_dim * element_bytes,
        SPARF_WORKFLOW_TRACE_BYTES,
    )


def grouped_attention_reserve_bytes(*, attention_slots, **geometry):
    if type(attention_slots) is not int or attention_slots <= 0:
        raise ValueError("attention_slots must be a positive integer")
    sizes = sparf_attention_range_bytes(**geometry)
    if (geometry["max_seq_len"] > 65536 or
            geometry["batch_size"] * geometry["query_heads"] > 65536 or
            any(size > GROUP_MAX_WORKSPACE_BYTES for size in sizes)):
        raise ValueError("grouped Attention exceeds device sequence/vector or 64 MiB workspace bounds")
    return attention_slots * sum(align_up(size, FDM_ALLOCATION_ALIGNMENT) for size in sizes)


@dataclass(frozen=True)
class SparfGroupCapacity:
    streams: int
    token_group_tokens: int
    channel_group_tokens: int
    channel_group_pages: int
    fdm_tail_bytes: int
    fdm_required_bytes: int
    pool_bytes: int
    pages: int


def plan_group_capacity(*, num_layers, batch_size, num_kv_heads, head_dim,
                        element_bytes, page_bytes, channels_per_page,
                        max_seq_len, pool_bytes=None, fdm_budget_bytes=None,
                        extra_fdm_bytes=0, write_batch_pages=0):
    values = (num_layers, batch_size, num_kv_heads, head_dim, element_bytes,
              page_bytes, channels_per_page, max_seq_len)
    if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0
           for value in values):
        raise ValueError("grouped capacity parameters must be positive integers")
    if page_bytes % 512 or page_bytes % (head_dim * element_bytes) or \
            page_bytes % (channels_per_page * element_bytes):
        raise ValueError("page geometry does not hold integral token entries")
    if head_dim % channels_per_page:
        raise ValueError("head_dim must be divisible by channels_per_page")
    if element_bytes not in (2, 4) or head_dim > 4096 or page_bytes > 65536:
        raise ValueError("grouped geometry exceeds the device workflow bounds")
    token_group = page_bytes // (head_dim * element_bytes)
    channel_group = page_bytes // (channels_per_page * element_bytes)
    channel_pages = head_dim // channels_per_page
    if (type(write_batch_pages) is not int or write_batch_pages < 0 or
            (write_batch_pages and write_batch_pages < channel_pages) or
            write_batch_pages * page_bytes > GROUP_MAX_WORKSPACE_BYTES):
        raise ValueError("invalid grouped write batch page capacity")
    streams = num_layers * batch_size * num_kv_heads
    pages = streams * (2 * ((max_seq_len + token_group - 1) // token_group) +
                       channel_pages * ((max_seq_len + channel_group - 1) // channel_group))
    if streams > GROUP_MAX_STREAMS or pages > GROUP_MAX_PAGES:
        raise ValueError(
            f"grouped store exceeds ABI bounds: streams={streams}, pages={pages}; "
            "reduce batch/sequence or shard the CSD"
        )
    stride = head_dim * element_bytes
    fdm_tail = streams * (2 * channel_group + 2 * token_group) * stride
    scratch = (write_batch_pages or channel_pages) * page_bytes
    required_fdm = (align_up(fdm_tail + scratch, FDM_ALLOCATION_ALIGNMENT) +
                    GROUP_IO_BYTES + GROUP_STATUS_BYTES)
    if isinstance(extra_fdm_bytes, bool) or not isinstance(extra_fdm_bytes, int) or extra_fdm_bytes < 0:
        raise ValueError("extra_fdm_bytes must be a nonnegative integer")
    required_fdm += extra_fdm_bytes
    if fdm_budget_bytes is not None and (type(fdm_budget_bytes) is not int or fdm_budget_bytes <= 0):
        raise ValueError("fdm_budget_bytes must be a positive integer")
    if fdm_budget_bytes is not None and required_fdm > fdm_budget_bytes:
        raise ValueError(
            f"grouped FDM budget is too small: need {required_fdm} bytes, "
            f"have {fdm_budget_bytes}"
        )
    required_pool = pages * page_bytes
    planned_pool = required_pool if pool_bytes is None else pool_bytes
    if type(planned_pool) is not int or planned_pool <= 0 or planned_pool % page_bytes:
        raise ValueError("grouped pool must contain whole pages")
    if planned_pool < required_pool:
        raise ValueError(
            f"grouped NVM pool is too small: need {required_pool} bytes, "
            f"have {planned_pool}"
        )
    planned_pages = planned_pool // page_bytes
    if planned_pages > GROUP_MAX_PAGES:
        raise ValueError("grouped pool exceeds the page-directory ABI bound")
    return SparfGroupCapacity(streams, token_group, channel_group, channel_pages,
                               fdm_tail, required_fdm, planned_pool, pages)


def check_filesystem_space(path, required_bytes):
    """Raise before allocation if the selected NVM/FDM filesystem is short."""
    if required_bytes < 0:
        raise ValueError("required_bytes must be nonnegative")
    usage = shutil.disk_usage(path)
    if usage.free < required_bytes:
        raise ValueError(
            f"insufficient filesystem space at {path}: need {required_bytes} bytes, "
            f"free {usage.free}"
        )


def check_fdm_space(path, required_bytes, budget_bytes=None):
    """Check allocatable bytes; zero-total statfs means unknown, not empty."""
    if type(required_bytes) is not int or required_bytes <= 0:
        raise ValueError("required_bytes must be a positive integer")
    if budget_bytes is not None and (type(budget_bytes) is not int or budget_bytes <= 0):
        raise ValueError("FDM budget must be a positive integer")
    usage = shutil.disk_usage(path)
    if usage.total == 0:
        if budget_bytes is None:
            raise ValueError(
                f"FDM filesystem at {path} does not report capacity via statfs; "
                "set --sparf-group-fdm-budget-bytes / SPARF_GROUP_FDM_BUDGET_BYTES "
                "to the verified available FDM budget for this run (excluding existing allocations)"
            )
        available = budget_bytes
    else:
        available = usage.free if budget_bytes is None else min(usage.free, budget_bytes)
    if required_bytes > available:
        raise ValueError(
            f"insufficient FDM budget at {path}: need {required_bytes} bytes "
            f"including storage and Attention workspaces, have {available}"
        )
    return available
