#include <algorithm>
#include <cstdio>

#include "attention_phases_common.h"

namespace {

float load_half(uint16_t value)
{
    const float magnitude = (value & 0x7fffU);
    const unsigned int exponent = (value >> 10) & 31U;
    const unsigned int mantissa = value & 1023U;
    float result;
    if (exponent == 31) {
        result = mantissa ? std::numeric_limits<float>::quiet_NaN()
                          : std::numeric_limits<float>::infinity();
    } else if (!exponent) {
        result = std::ldexp(magnitude, -24);
    } else {
        result = std::ldexp(static_cast<float>(1024 + mantissa),
                            static_cast<int>(exponent) - 25);
    }
    return value & 0x8000U ? -result : result;
}

uint16_t store_half(float value)
{
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    const uint16_t sign = (bits >> 16) & 0x8000U;
    const unsigned int source_exponent = (bits >> 23) & 255U;
    const uint32_t fraction = bits & 0x7fffffU;
    if (source_exponent == 255) {
        return sign | (fraction ? 0x7e00U : 0x7c00U);
    }
    const int exponent = static_cast<int>(source_exponent) - 127;
    if (exponent > 15) {
        return sign | 0x7c00U;
    }
    if (exponent < -25) {
        return sign;
    }
    const unsigned int shift = exponent < -14 ? -exponent - 1 : 13;
    const uint32_t mantissa = fraction | 0x800000U;
    uint32_t rounded = mantissa >> shift;
    const uint32_t remainder = mantissa & ((1U << shift) - 1);
    const uint32_t midpoint = 1U << (shift - 1);
    rounded += remainder > midpoint || (remainder == midpoint && (rounded & 1));
    return sign | (exponent < -14 ? rounded : ((exponent + 14) << 10) + rounded);
}

float load(const void *pointer, uint64_t index, uint32_t dtype)
{
    return dtype == CEMU_ATTENTION_PHASES_FLOAT16
        ? load_half(static_cast<const uint16_t *>(pointer)[index])
        : static_cast<const float *>(pointer)[index];
}

}

extern "C" long long attention_phase(cemu_args *args)
{
    cemu_attention_phase_metadata config;
    if (!validate_attention_phase(args, false, &config)) {
        std::fprintf(stderr, "attention_phase: invalid metadata or memory ranges\n");
        return -1;
    }
    const uint64_t vectors = static_cast<uint64_t>(config.batch_size) * config.num_query_heads;
    for (uint64_t vector = 0; vector < vectors; ++vector) {
        const uint64_t batch = vector / config.num_query_heads;
        const uint64_t head = vector % config.num_query_heads;
        const uint64_t kv_head = head / (config.num_query_heads / config.num_kv_heads);
        const uint64_t kv_offset = (batch * config.num_kv_heads + kv_head) * config.head_dim;
        if (config.phase == CEMU_ATTENTION_PHASE_QK_SOFTMAX) {
            float *probabilities = static_cast<float *>(args->mr_addr[2]) + vector * config.token_count;
            float maximum = -std::numeric_limits<float>::infinity();
            for (uint64_t token = 0; token < config.token_count; ++token) {
                const void *keys = static_cast<const uint8_t *>(args->mr_addr[1]) + token * config.token_stride;
                float score = 0.0f;
                for (uint64_t dimension = 0; dimension < config.head_dim; ++dimension) {
                    score += load(args->mr_addr[0], vector * config.head_dim + dimension, config.dtype) *
                             load(keys, kv_offset + dimension, config.dtype);
                }
                probabilities[token] = score * config.scale;
                maximum = std::max(maximum, probabilities[token]);
            }
            float denominator = 0.0f;
            for (uint64_t token = 0; token < config.token_count; ++token) {
                probabilities[token] = std::exp(probabilities[token] - maximum);
                denominator += probabilities[token];
            }
            for (uint64_t token = 0; token < config.token_count; ++token) {
                probabilities[token] /= denominator;
            }
        } else {
            const float *probabilities = static_cast<const float *>(args->mr_addr[0]) + vector * config.token_count;
            for (uint64_t dimension = 0; dimension < config.head_dim; ++dimension) {
                float result = 0.0f;
                for (uint64_t token = 0; token < config.token_count; ++token) {
                    const void *values = static_cast<const uint8_t *>(args->mr_addr[1]) + token * config.token_stride;
                    result += probabilities[token] * load(values, kv_offset + dimension, config.dtype);
                }
                const uint64_t output_index = vector * config.head_dim + dimension;
                if (config.dtype == CEMU_ATTENTION_PHASES_FLOAT16) {
                    static_cast<uint16_t *>(args->mr_addr[2])[output_index] = store_half(result);
                } else {
                    static_cast<float *>(args->mr_addr[2])[output_index] = result;
                }
            }
        }
    }
    return vectors * (config.phase == CEMU_ATTENTION_PHASE_QK_SOFTMAX
                      ? config.token_count : config.head_dim);
}
