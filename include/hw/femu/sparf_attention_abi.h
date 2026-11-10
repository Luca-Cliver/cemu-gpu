#ifndef CEMU_SPARF_ATTENTION_ABI_H
#define CEMU_SPARF_ATTENTION_ABI_H

#include <stdint.h>

#define CEMU_SPARF_ATTENTION_VERSION 1U
#define CEMU_SPARF_ATTENTION_RAGGED_VERSION 2U
#define CEMU_SPARF_PHASE_EXACT_QK 1U
#define CEMU_SPARF_PHASE_PV 2U
#define CEMU_SPARF_FLOAT32 1U
#define CEMU_SPARF_FLOAT16 2U

/* Packed batch order, independent of persistent KV slot order. Offsets count
 * tokens before this sequence (before multiplication by query heads).
 * Workflow metadata carries these rows; MR7 carries alpha then these rows.
 * Attention v2 reserved holds the total selected tokens over all Q heads. */
struct cemu_sparf_batch_row {
    uint32_t slot;
    uint32_t valid_tokens;
    uint32_t top_k;
    uint32_t history_offset;
    uint32_t selected_offset;
};

struct cemu_sparf_attention_metadata {
    uint32_t version;
    uint32_t phase;
    uint32_t dtype;
    uint32_t batch_size;
    uint32_t num_query_heads;
    uint32_t head_dim;
    uint32_t selected_tokens;
    uint32_t reserved;
    float scale;
};

#if defined(__cplusplus)
static_assert(sizeof(cemu_sparf_batch_row) == 20, "SparF batch row ABI changed");
static_assert(sizeof(cemu_sparf_attention_metadata) == 36,
              "SparF Attention metadata ABI changed");
#else
_Static_assert(sizeof(struct cemu_sparf_batch_row) == 20, "SparF batch row ABI changed");
_Static_assert(sizeof(struct cemu_sparf_attention_metadata) == 36,
               "SparF Attention metadata ABI changed");
#endif

#endif
