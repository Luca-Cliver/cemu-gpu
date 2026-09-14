#ifndef CEMU_ATTENTION_PHASES_ABI_H
#define CEMU_ATTENTION_PHASES_ABI_H

#include <stdint.h>

#define CEMU_ATTENTION_PHASES_VERSION 1U
#define CEMU_ATTENTION_PHASE_QK_SOFTMAX 1U
#define CEMU_ATTENTION_PHASE_PV 2U
#define CEMU_ATTENTION_PHASES_FLOAT32 1U
#define CEMU_ATTENTION_PHASES_FLOAT16 2U

struct cemu_attention_phase_metadata {
    uint32_t version;
    uint32_t phase;
    uint32_t dtype;
    uint32_t batch_size;
    uint32_t num_query_heads;
    uint32_t num_kv_heads;
    uint32_t head_dim;
    uint32_t token_count;
    uint32_t token_stride;
    float scale;
};

#if defined(__cplusplus)
static_assert(sizeof(cemu_attention_phase_metadata) == 40,
              "Attention phase metadata ABI changed");
#else
_Static_assert(sizeof(struct cemu_attention_phase_metadata) == 40,
               "Attention phase metadata ABI changed");
#endif

#endif
