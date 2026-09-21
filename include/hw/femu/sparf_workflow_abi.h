#ifndef CEMU_SPARF_WORKFLOW_ABI_H
#define CEMU_SPARF_WORKFLOW_ABI_H

#include <stdint.h>
#include "sparf_attention_abi.h"

#define CEMU_SPARF_WORKFLOW_COMMAND UINT64_C(0x43454d5553505246)
#define CEMU_SPARF_WORKFLOW_VERSION 1U
#define CEMU_SPARF_AUTONOMOUS_VERSION 3U
#define CEMU_SPARF_WORKFLOW_TRACE 1U
#define CEMU_SPARF_WORKFLOW_BUFFERED 2U
#define CEMU_SPARF_WORKFLOW_GROUPED 4U

struct cemu_sparf_group_binding {
    uint64_t cookie;
    uint64_t stream_base;
};
#define CEMU_SPARF_WORKFLOW_MAX_EXTENTS 1024U
#define CEMU_SPARF_WORKFLOW_RANGES 11U

/* Version 2 uses an additional slot-local FDM range for continuation state.
 * reserved[0] is the stage and reserved[1] a nonzero workflow sequence id.
 * No host pointer or pending FTL request survives an execute completion. */
#define CEMU_SPARF_STAGED_VERSION 2U
#define CEMU_SPARF_STAGE_CONTEXT_RANGE 11U
#define CEMU_SPARF_STAGED_RANGES 12U
#define CEMU_SPARF_STAGE_SELECT 1U
#define CEMU_SPARF_STAGE_TOKEN_READ 2U
#define CEMU_SPARF_STAGE_EXACT_V 3U
#define CEMU_SPARF_STAGE_PV 4U
#define CEMU_SPARF_STAGE_ERROR UINT32_MAX
#define CEMU_SPARF_CONTEXT_MAGIC UINT32_C(0x53504632)

struct cemu_sparf_workflow_header {
    uint32_t version;
    uint32_t flags;
    uint32_t approximate_runtime_ns;
    uint32_t exact_qk_runtime_ns;
    uint32_t pv_runtime_ns;
    uint32_t token_k_extents;
    uint32_t channel_k_extents;
    uint32_t token_v_extents;
    uint32_t dtype;
    uint32_t batch_size;
    uint32_t num_query_heads;
    uint32_t num_kv_heads;
    uint32_t head_dim;
    uint32_t max_seq_len;
    uint32_t valid_tokens;
    uint32_t layer;
    uint32_t top_r;
    uint32_t top_k;
    uint32_t token_head_stride;
    uint32_t token_batch_stride;
    uint32_t token_layer_stride;
    uint32_t channel_stride;
    uint32_t channel_head_stride;
    uint32_t channel_batch_stride;
    uint32_t channel_layer_stride;
    float scale;
    uint32_t reserved[2];
};

struct cemu_sparf_workflow_extent {
    uint64_t slba;
    uint32_t nlb;
    uint32_t reserved;
};

struct cemu_sparf_buffer_tail {
    uint32_t published_tokens;
    uint32_t buffered_tokens;
};

struct cemu_sparf_workflow_trace {
    uint64_t channel_read_model_ns;
    uint64_t approximate_model_ns;
    uint64_t token_k_read_model_ns;
    uint64_t token_v_read_model_ns;
    uint64_t exact_qk_model_ns;
    uint64_t pv_model_ns;
    uint64_t total_model_ns;
    uint64_t selected_channels;
    uint64_t selected_tokens;
};

/* Followed by canonical workflow metadata (phase=0), then uint32 token indices.
 * Intermediate tensors remain in this slot's existing FDM ranges. */
struct cemu_sparf_stage_context {
    uint32_t magic;
    uint32_t completed_stage;
    uint32_t workflow_id;
    uint32_t metadata_bytes;
    uint64_t token_count;
    struct cemu_sparf_workflow_trace trace;
};

#if defined(__cplusplus)
static_assert(sizeof(cemu_sparf_stage_context) == 96, "SparF context ABI changed");
#else
_Static_assert(sizeof(struct cemu_sparf_stage_context) == 96,
               "SparF context ABI changed");
#endif

#endif
