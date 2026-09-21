#ifndef CEMU_SPARF_ATTENTION_COMMON_H
#define CEMU_SPARF_ATTENTION_COMMON_H

#include <cmath>
#include <cstdint>
#include <cstring>

#include "cemu_def.h"
#include "sparf_attention_abi.h"

inline bool validate_sparf_attention(const cemu_args *args, bool device,
                                     cemu_sparf_attention_metadata *config)
{
    if (!args || !args->data_buffer || args->buffer_len != sizeof(*config) ||
        !args->mr_len || (args->numr != 3 && args->numr != 4)) {
        return false;
    }
    void **addresses = device ? args->mr_dev_addr : args->mr_addr;
    if (!addresses) {
        return false;
    }
    std::memcpy(config, args->data_buffer, sizeof(*config));
    if (config->version != CEMU_SPARF_ATTENTION_VERSION || config->reserved ||
        (config->phase != CEMU_SPARF_PHASE_EXACT_QK &&
         config->phase != CEMU_SPARF_PHASE_PV) ||
        (config->dtype != CEMU_SPARF_FLOAT16 && config->dtype != CEMU_SPARF_FLOAT32) ||
        !config->batch_size || !config->num_query_heads || !config->head_dim ||
        !config->selected_tokens || !std::isfinite(config->scale) ||
        config->scale <= 0.0f) {
        return false;
    }
    const bool exact = config->phase == CEMU_SPARF_PHASE_EXACT_QK;
    if ((exact && args->numr != 4) || (!exact && args->numr != 3)) {
        return false;
    }
    const uint64_t vectors = static_cast<uint64_t>(config->batch_size) * config->num_query_heads;
    const uint64_t element = config->dtype == CEMU_SPARF_FLOAT16 ? 2 : 4;
    const uint64_t q_bytes = vectors * config->head_dim * element;
    const uint64_t probability_bytes = vectors * (config->selected_tokens + 1) * sizeof(float);
    const uint64_t packed_bytes = vectors * config->selected_tokens * config->head_dim * element;
    const uint64_t lengths[] = {
        exact ? q_bytes : probability_bytes,
        exact ? packed_bytes : vectors * (config->selected_tokens + 1) * config->head_dim * element,
        exact ? vectors * sizeof(float) : q_bytes,
        probability_bytes,
    };
    for (int index = 0; index < args->numr; ++index) {
        if (!addresses[index] || args->mr_len[index] < 0 ||
            static_cast<uint64_t>(args->mr_len[index]) < lengths[index]) {
            return false;
        }
    }
    return true;
}

#endif
