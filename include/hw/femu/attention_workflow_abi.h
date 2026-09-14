#ifndef CEMU_ATTENTION_WORKFLOW_ABI_H
#define CEMU_ATTENTION_WORKFLOW_ABI_H

#include <stdint.h>
#include "attention_phases_abi.h"

#define CEMU_ATTENTION_WORKFLOW_COMMAND UINT64_C(0x43454d5550484153)
#define CEMU_ATTENTION_WORKFLOW_VERSION 1U
#define CEMU_ATTENTION_WORKFLOW_TRACE 1U
#define CEMU_ATTENTION_WORKFLOW_SERIAL 2U
#define CEMU_ATTENTION_WORKFLOW_MAX_EXTENTS 1024U
#define CEMU_ATTENTION_WORKFLOW_RANGES 6U

struct cemu_attention_workflow_header {
    uint32_t version;
    uint32_t flags;
    uint32_t qk_runtime_ns;
    uint32_t pv_runtime_ns;
    uint32_t key_extents;
    uint32_t value_extents;
    uint32_t reserved[2];
    struct cemu_attention_phase_metadata attention;
};

struct cemu_attention_workflow_extent {
    uint64_t slba;
    uint32_t nlb;
    uint32_t reserved;
};

struct cemu_attention_workflow_trace {
    uint64_t key_submit_ns;
    uint64_t key_ready_ns;
    uint64_t value_submit_ns;
    uint64_t qk_start_ns;
    uint64_t qk_done_ns;
    uint64_t value_ready_ns;
    uint64_t pv_start_ns;
    uint64_t pv_done_ns;
    uint64_t finish_ns;
    uint64_t key_model_ns;
    uint64_t value_model_ns;
    uint64_t qk_model_ns;
    uint64_t pv_model_ns;
    uint64_t total_model_ns;
};

#if defined(__cplusplus)
static_assert(sizeof(cemu_attention_workflow_header) == 72,
              "Attention workflow header ABI changed");
static_assert(sizeof(cemu_attention_workflow_extent) == 16,
              "Attention workflow extent ABI changed");
static_assert(sizeof(cemu_attention_workflow_trace) == 112,
              "Attention workflow trace ABI changed");
#else
_Static_assert(sizeof(struct cemu_attention_workflow_header) == 72,
               "Attention workflow header ABI changed");
_Static_assert(sizeof(struct cemu_attention_workflow_extent) == 16,
               "Attention workflow extent ABI changed");
_Static_assert(sizeof(struct cemu_attention_workflow_trace) == 112,
               "Attention workflow trace ABI changed");
#endif

#endif
