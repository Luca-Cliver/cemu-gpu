#include <algorithm>
#include <cstdio>
#include <limits>

#include "sparf_attention_common.h"

namespace {
float load_half(uint16_t value)
{
    const unsigned exponent = (value >> 10) & 31U;
    const unsigned mantissa = value & 1023U;
    float result = exponent == 31 ? std::numeric_limits<float>::infinity()
        : (!exponent ? std::ldexp(static_cast<float>(mantissa), -24)
                     : std::ldexp(static_cast<float>(1024 + mantissa), static_cast<int>(exponent) - 25));
    return value & 0x8000U ? -result : result;
}

uint16_t store_half(float value)
{
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    const uint16_t sign = (bits >> 16) & 0x8000U;
    int exponent = static_cast<int>((bits >> 23) & 255U) - 127;
    if (exponent > 15) return sign | 0x7c00U;
    if (exponent < -25) return sign;
    const uint32_t mantissa = (bits & 0x7fffffU) | 0x800000U;
    const unsigned shift = exponent < -14 ? -exponent - 1 : 13;
    uint32_t rounded = mantissa >> shift;
    const uint32_t remainder = mantissa & ((1U << shift) - 1);
    const uint32_t midpoint = 1U << (shift - 1);
    rounded += remainder > midpoint || (remainder == midpoint && (rounded & 1));
    return sign | (exponent < -14 ? rounded : ((exponent + 14) << 10) + rounded);
}

float load(const void *data, uint64_t index, uint32_t dtype)
{
    return dtype == CEMU_SPARF_FLOAT16
        ? load_half(static_cast<const uint16_t *>(data)[index])
        : static_cast<const float *>(data)[index];
}

void store(void *data, uint64_t index, uint32_t dtype, float value)
{
    if (dtype == CEMU_SPARF_FLOAT16) static_cast<uint16_t *>(data)[index] = store_half(value);
    else static_cast<float *>(data)[index] = value;
}
}

extern "C" long long sparf_attention(cemu_args *args)
{
    cemu_sparf_attention_metadata config;
    if (!validate_sparf_attention(args, false, &config)) {
        std::fprintf(stderr, "sparf_attention: invalid metadata or ranges\n");
        return -1;
    }
    const uint64_t vectors = static_cast<uint64_t>(config.batch_size) * config.num_query_heads;
    const bool ragged = config.version == CEMU_SPARF_ATTENTION_RAGGED_VERSION;
    const auto *rows = ragged ? reinterpret_cast<const cemu_sparf_batch_row *>(
        static_cast<const float *>(args->mr_addr[2]) + vectors) : nullptr;
    if (config.phase == CEMU_SPARF_PHASE_EXACT_QK) {
        const float *alpha = static_cast<const float *>(args->mr_addr[2]);
        float *probability = static_cast<float *>(args->mr_addr[3]);
        for (uint64_t vector = 0; vector < vectors; ++vector) {
            const auto *r = rows ? &rows[vector / config.num_query_heads] : nullptr;
            const uint32_t selected = r ? r->top_k : config.selected_tokens;
            const uint64_t key_base = r ? static_cast<uint64_t>(r->selected_offset) * config.num_query_heads +
                (vector % config.num_query_heads) * selected : vector * selected;
            const uint64_t probability_base = key_base + vector;
            float maximum = -std::numeric_limits<float>::infinity();
            for (uint32_t token = 0; token < selected; ++token) {
                float score = 0.0f;
                for (uint32_t dimension = 0; dimension < config.head_dim; ++dimension) {
                    const uint64_t q = vector * config.head_dim + dimension;
                    const uint64_t k = (key_base + token) * config.head_dim + dimension;
                    score += load(args->mr_addr[0], q, config.dtype) * load(args->mr_addr[1], k, config.dtype);
                }
                probability[probability_base + token] = score * config.scale;
                maximum = std::max(maximum, score * config.scale);
            }
            float denominator = 0.0f;
            for (uint32_t token = 0; token < selected; ++token) {
                float &item = probability[probability_base + token];
                item = std::exp(item - maximum);
                denominator += item;
            }
            for (uint32_t token = 0; token < selected; ++token) {
                probability[probability_base + token] *= alpha[vector] / denominator;
            }
            probability[probability_base + selected] = 1.0f - alpha[vector];
        }
    } else {
        const float *probability = static_cast<const float *>(args->mr_addr[0]);
        for (uint64_t vector = 0; vector < vectors; ++vector) {
            const auto *r = rows ? &rows[vector / config.num_query_heads] : nullptr;
            const uint32_t selected = r ? r->top_k : config.selected_tokens;
            const uint64_t key_base = r ? static_cast<uint64_t>(r->selected_offset) * config.num_query_heads +
                (vector % config.num_query_heads) * selected : vector * selected;
            const uint64_t probability_base = key_base + vector;
            for (uint32_t dimension = 0; dimension < config.head_dim; ++dimension) {
                float result = 0.0f;
                for (uint32_t token = 0; token <= selected; ++token) {
                    const uint64_t index = (probability_base + token) * config.head_dim + dimension;
                    result += probability[probability_base + token] * load(args->mr_addr[1], index, config.dtype);
                }
                store(args->mr_addr[ragged ? 3 : 2], vector * config.head_dim + dimension, config.dtype, result);
            }
        }
    }
    return vectors * config.head_dim;
}
