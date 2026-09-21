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
    if (config.phase == CEMU_SPARF_PHASE_EXACT_QK) {
        const float *alpha = static_cast<const float *>(args->mr_addr[2]);
        float *probability = static_cast<float *>(args->mr_addr[3]);
        for (uint64_t vector = 0; vector < vectors; ++vector) {
            float maximum = -std::numeric_limits<float>::infinity();
            for (uint32_t token = 0; token < config.selected_tokens; ++token) {
                float score = 0.0f;
                for (uint32_t dimension = 0; dimension < config.head_dim; ++dimension) {
                    const uint64_t q = vector * config.head_dim + dimension;
                    const uint64_t k = (vector * config.selected_tokens + token) * config.head_dim + dimension;
                    score += load(args->mr_addr[0], q, config.dtype) * load(args->mr_addr[1], k, config.dtype);
                }
                probability[vector * (config.selected_tokens + 1) + token] = score * config.scale;
                maximum = std::max(maximum, score * config.scale);
            }
            float denominator = 0.0f;
            for (uint32_t token = 0; token < config.selected_tokens; ++token) {
                float &item = probability[vector * (config.selected_tokens + 1) + token];
                item = std::exp(item - maximum);
                denominator += item;
            }
            for (uint32_t token = 0; token < config.selected_tokens; ++token) {
                probability[vector * (config.selected_tokens + 1) + token] *= alpha[vector] / denominator;
            }
            probability[vector * (config.selected_tokens + 1) + config.selected_tokens] = 1.0f - alpha[vector];
        }
    } else {
        const float *probability = static_cast<const float *>(args->mr_addr[0]);
        for (uint64_t vector = 0; vector < vectors; ++vector) {
            for (uint32_t dimension = 0; dimension < config.head_dim; ++dimension) {
                float result = 0.0f;
                for (uint32_t token = 0; token <= config.selected_tokens; ++token) {
                    const uint64_t index = (vector * (config.selected_tokens + 1) + token) * config.head_dim + dimension;
                    result += probability[vector * (config.selected_tokens + 1) + token] * load(args->mr_addr[1], index, config.dtype);
                }
                store(args->mr_addr[2], vector * config.head_dim + dimension, config.dtype, result);
            }
        }
    }
    return vectors * config.head_dim;
}
