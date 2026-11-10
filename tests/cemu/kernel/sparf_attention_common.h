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
    const bool ragged = config->version == CEMU_SPARF_ATTENTION_RAGGED_VERSION;
    if ((!ragged && (config->version != CEMU_SPARF_ATTENTION_VERSION || config->reserved)) ||
        (config->phase != CEMU_SPARF_PHASE_EXACT_QK &&
         config->phase != CEMU_SPARF_PHASE_PV) ||
        (config->dtype != CEMU_SPARF_FLOAT16 && config->dtype != CEMU_SPARF_FLOAT32) ||
        !config->batch_size || !config->num_query_heads || !config->head_dim ||
        !config->selected_tokens || !std::isfinite(config->scale) ||
        config->scale <= 0.0f) {
        return false;
    }
    const bool exact = config->phase == CEMU_SPARF_PHASE_EXACT_QK;
    if ((exact && args->numr != 4) || (!exact && args->numr != (ragged ? 4 : 3))) {
        return false;
    }
    const uint64_t vectors = static_cast<uint64_t>(config->batch_size) * config->num_query_heads;
    const uint64_t element = config->dtype == CEMU_SPARF_FLOAT16 ? 2 : 4;
    const uint64_t q_bytes = vectors * config->head_dim * element;
    const uint64_t selected = ragged ? config->reserved : vectors * config->selected_tokens;
    if (ragged && (selected < vectors || selected > vectors * config->selected_tokens)) return false;
    const uint64_t probability_bytes = (selected + vectors) * sizeof(float);
    const uint64_t packed_bytes = selected * config->head_dim * element;
    const uint64_t alpha_bytes = vectors * sizeof(float) +
        (ragged ? config->batch_size * sizeof(cemu_sparf_batch_row) : 0);
    const uint64_t lengths[] = {
        exact ? q_bytes : probability_bytes,
        exact ? packed_bytes : (selected + vectors) * config->head_dim * element,
        (exact || ragged) ? alpha_bytes : q_bytes,
        exact ? probability_bytes : q_bytes,
    };
    for (int index = 0; index < args->numr; ++index) {
        if (!addresses[index] || args->mr_len[index] < 0 ||
            static_cast<uint64_t>(args->mr_len[index]) < lengths[index]) {
            return false;
        }
    }
    if (ragged) {
        // MR7 is host-generated and retains a host mirror even for devptr jobs.
        if (!args->mr_addr || !args->mr_addr[2]) return false;
        const auto *rows = reinterpret_cast<const cemu_sparf_batch_row *>(
            static_cast<const float *>(args->mr_addr[2]) + vectors);
        uint64_t prefix = 0;
        for (uint32_t b = 0; b < config->batch_size; ++b) {
            if (!rows[b].top_k || rows[b].top_k > config->selected_tokens ||
                rows[b].selected_offset != prefix) return false;
            prefix += rows[b].top_k;
        }
        if (prefix * config->num_query_heads != selected) return false;
    }
    return true;
}

#endif
