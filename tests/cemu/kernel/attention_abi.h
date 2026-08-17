#ifndef CEMU_ATTENTION_ABI_H
#define CEMU_ATTENTION_ABI_H

#include <stdint.h>

#define CEMU_ATTENTION_ABI_VERSION 1U
#define CEMU_ATTENTION_DTYPE_FLOAT32 1U
#define CEMU_ATTENTION_FLAG_RESET_STATE (1U << 0)
#define CEMU_ATTENTION_FLAG_FINALIZE (1U << 1)

struct cemu_attention_metadata {
    uint32_t version;
    uint32_t dtype;
    uint32_t batch_size;
    uint32_t num_query_heads;
    uint32_t num_kv_heads;
    uint32_t head_dim;
    uint32_t token_count;
    uint32_t token_stride;
    float scale;
    uint32_t flags;
};

#ifdef __cplusplus
static_assert(sizeof(struct cemu_attention_metadata) == 40,
              "unexpected attention metadata size");
#endif

#endif
