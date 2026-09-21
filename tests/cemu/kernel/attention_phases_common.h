#ifndef CEMU_ATTENTION_PHASES_COMMON_H
#define CEMU_ATTENTION_PHASES_COMMON_H

#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>

#include "attention_phases_abi.h"
#include "cemu_def.h"

inline bool phase_multiply(uint64_t *value, uint64_t factor)
{
    if (factor == 0 || *value > static_cast<uint64_t>(INT64_MAX) / factor) {
        return false;
    }
    *value *= factor;
    return true;
}

inline bool validate_attention_phase(const cemu_args *args, bool device,
                                     cemu_attention_phase_metadata *metadata)
{
    if (!args || args->numr != 3 || !args->data_buffer ||
        args->buffer_len != sizeof(*metadata) || !args->mr_len) {
        return false;
    }
    void **addresses = device ? args->mr_dev_addr : args->mr_addr;
    if (!addresses) {
        return false;
    }
    std::memcpy(metadata, args->data_buffer, sizeof(*metadata));
    const auto &config = *metadata;
    if (config.version != CEMU_ATTENTION_PHASES_VERSION ||
        (config.phase != CEMU_ATTENTION_PHASE_QK_SOFTMAX &&
         config.phase != CEMU_ATTENTION_PHASE_PV) ||
        (config.dtype != CEMU_ATTENTION_PHASES_FLOAT32 &&
         config.dtype != CEMU_ATTENTION_PHASES_FLOAT16) ||
        !config.batch_size || !config.num_query_heads || !config.num_kv_heads ||
        !config.head_dim || !config.token_count || !config.token_stride ||
        config.num_query_heads % config.num_kv_heads || config.token_stride % 512 ||
        !std::isfinite(config.scale) || config.scale <= 0.0f) {
        return false;
    }
    const uint64_t element_bytes = config.dtype == CEMU_ATTENTION_PHASES_FLOAT16 ? 2 : 4;
    uint64_t query_bytes = config.batch_size;
    uint64_t token_bytes = config.batch_size;
    uint64_t probability_bytes = config.batch_size;
    uint64_t storage_bytes = config.token_count;
    if (!phase_multiply(&query_bytes, config.num_query_heads) ||
        !phase_multiply(&query_bytes, config.head_dim) ||
        !phase_multiply(&query_bytes, element_bytes) ||
        !phase_multiply(&token_bytes, config.num_kv_heads) ||
        !phase_multiply(&token_bytes, config.head_dim) ||
        !phase_multiply(&token_bytes, element_bytes) ||
        token_bytes > config.token_stride ||
        !phase_multiply(&storage_bytes, config.token_stride) ||
        !phase_multiply(&probability_bytes, config.num_query_heads) ||
        !phase_multiply(&probability_bytes, config.token_count) ||
        !phase_multiply(&probability_bytes, sizeof(float))) {
        return false;
    }
    const bool qk = config.phase == CEMU_ATTENTION_PHASE_QK_SOFTMAX;
    const uint64_t lengths[] = {qk ? query_bytes : probability_bytes,
                                storage_bytes, qk ? probability_bytes : query_bytes};
    for (unsigned int range = 0; range < 3; ++range) {
        const uint64_t alignment = (qk ? range == 2 : range == 0) ? 4 : element_bytes;
        if (!addresses[range] || args->mr_len[range] < 0 ||
            static_cast<uint64_t>(args->mr_len[range]) < lengths[range] ||
            reinterpret_cast<uintptr_t>(addresses[range]) % alignment) {
            return false;
        }
    }
    const uintptr_t output_start = reinterpret_cast<uintptr_t>(addresses[2]);
    if (lengths[2] > UINTPTR_MAX - output_start) {
        return false;
    }
    for (unsigned int range = 0; range < 2; ++range) {
        const uintptr_t input_start = reinterpret_cast<uintptr_t>(addresses[range]);
        if (lengths[range] > UINTPTR_MAX - input_start ||
            (output_start < input_start + lengths[range] &&
             input_start < output_start + lengths[2])) {
            return false;
        }
    }
    return true;
}

#endif
