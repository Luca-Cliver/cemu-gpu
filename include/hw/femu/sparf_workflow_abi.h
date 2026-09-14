#ifndef CEMU_SPARF_WORKFLOW_ABI_H
#define CEMU_SPARF_WORKFLOW_ABI_H

#include <stdint.h>
#include "sparf_attention_abi.h"

#define CEMU_SPARF_WORKFLOW_COMMAND UINT64_C(0x43454d5553505246)
#define CEMU_SPARF_WORKFLOW_VERSION 1U
#define CEMU_SPARF_WORKFLOW_TRACE 1U
#define CEMU_SPARF_WORKFLOW_MAX_EXTENTS 1024U
#define CEMU_SPARF_WORKFLOW_RANGES 11U

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

#endif
