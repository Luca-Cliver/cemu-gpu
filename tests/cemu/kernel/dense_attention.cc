#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#include "attention_abi.h"
#include "cemu_def.h"

namespace {

uint32_t element_size(uint32_t dtype)
{
    return dtype == CEMU_ATTENTION_DTYPE_FLOAT16 ? sizeof(uint16_t) : sizeof(float);
}

float half_to_float(uint16_t value)
{
    const uint32_t sign = static_cast<uint32_t>(value & 0x8000U) << 16;
    uint32_t exponent = (value >> 10) & 0x1fU;
    uint32_t mantissa = value & 0x03ffU;
    uint32_t bits = 0;

    if (exponent == 0) {
        if (mantissa == 0) {
            bits = sign;
        } else {
            exponent = 127 - 15 + 1;
            while ((mantissa & 0x0400U) == 0) {
                mantissa <<= 1;
                --exponent;
            }
            mantissa &= 0x03ffU;
            bits = sign | (exponent << 23) | (mantissa << 13);
        }
    } else if (exponent == 0x1fU) {
        bits = sign | 0x7f800000U | (mantissa << 13);
    } else {
        bits = sign | ((exponent + 127 - 15) << 23) | (mantissa << 13);
    }

    float result = 0.0f;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

uint16_t float_to_half(float value)
{
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    const uint16_t sign = static_cast<uint16_t>((bits >> 16) & 0x8000U);
    const uint32_t source_exponent = (bits >> 23) & 0xffU;
    uint32_t mantissa = bits & 0x007fffffU;

    if (source_exponent == 0xffU) {
        if (mantissa == 0) {
            return static_cast<uint16_t>(sign | 0x7c00U);
        }
        return static_cast<uint16_t>(sign | 0x7c00U | (mantissa >> 13) | 1U);
    }

    int exponent = static_cast<int>(source_exponent) - 127 + 15;
    if (exponent >= 31) {
        return static_cast<uint16_t>(sign | 0x7c00U);
    }
    if (exponent <= 0) {
        if (exponent < -10) {
            return sign;
        }
        mantissa |= 0x00800000U;
        const uint32_t shift = static_cast<uint32_t>(14 - exponent);
        const uint32_t rounded = mantissa + (1U << (shift - 1));
        return static_cast<uint16_t>(sign | (rounded >> shift));
    }

    mantissa += 0x00001000U;
    if (mantissa & 0x00800000U) {
        mantissa = 0;
        ++exponent;
        if (exponent >= 31) {
            return static_cast<uint16_t>(sign | 0x7c00U);
        }
    }
    return static_cast<uint16_t>(
        sign | (static_cast<uint16_t>(exponent) << 10) | (mantissa >> 13));
}

float load_value(const void *base, uint64_t index, uint32_t dtype)
{
    if (dtype == CEMU_ATTENTION_DTYPE_FLOAT16) {
        return half_to_float(static_cast<const uint16_t *>(base)[index]);
    }
    return static_cast<const float *>(base)[index];
}

void store_value(void *base, uint64_t index, uint32_t dtype, float value)
{
    if (dtype == CEMU_ATTENTION_DTYPE_FLOAT16) {
        static_cast<uint16_t *>(base)[index] = float_to_half(value);
    } else {
        static_cast<float *>(base)[index] = value;
    }
}

bool load_metadata(const cemu_args *args, cemu_attention_metadata *metadata)
{
    if (!args || !metadata || !args->data_buffer ||
        args->buffer_len < static_cast<long long>(sizeof(*metadata))) {
        std::fprintf(stderr, "dense_attention: missing metadata\n");
        return false;
    }

    std::memcpy(metadata, args->data_buffer, sizeof(*metadata));
    if (metadata->version != CEMU_ATTENTION_ABI_VERSION ||
        (metadata->dtype != CEMU_ATTENTION_DTYPE_FLOAT32 &&
         metadata->dtype != CEMU_ATTENTION_DTYPE_FLOAT16) ||
        metadata->batch_size == 0 || metadata->num_query_heads == 0 ||
        metadata->num_kv_heads == 0 || metadata->head_dim == 0 ||
        metadata->token_count == 0 ||
        metadata->num_query_heads % metadata->num_kv_heads != 0 ||
        metadata->token_stride % 512 != 0 ||
        metadata->token_stride < metadata->batch_size * metadata->num_kv_heads *
                                     metadata->head_dim * element_size(metadata->dtype) ||
        !std::isfinite(metadata->scale) || metadata->scale <= 0.0f ||
        (metadata->flags & ~(CEMU_ATTENTION_FLAG_RESET_STATE |
                             CEMU_ATTENTION_FLAG_FINALIZE)) != 0) {
        std::fprintf(stderr, "dense_attention: invalid metadata\n");
        return false;
    }
    return true;
}

bool validate_ranges(const cemu_args *args, const cemu_attention_metadata &metadata)
{
    if (args->numr != 5 || !args->mr_addr || !args->mr_len) {
        std::fprintf(stderr, "dense_attention: expected five memory ranges\n");
        return false;
    }
    for (int index = 0; index < args->numr; ++index) {
        if (!args->mr_addr[index] || args->mr_len[index] <= 0) {
            std::fprintf(stderr, "dense_attention: invalid memory range %d\n", index);
            return false;
        }
    }

    const uint64_t storage_element_size = element_size(metadata.dtype);
    const uint64_t query_bytes =
        static_cast<uint64_t>(metadata.batch_size) * metadata.num_query_heads *
        metadata.head_dim * storage_element_size;
    const uint64_t token_payload_bytes =
        static_cast<uint64_t>(metadata.batch_size) * metadata.num_kv_heads *
        metadata.head_dim * storage_element_size;
    const uint64_t kv_bytes =
        static_cast<uint64_t>(metadata.token_count - 1) * metadata.token_stride +
        token_payload_bytes;

    if (query_bytes > static_cast<uint64_t>(args->mr_len[0]) ||
        kv_bytes > static_cast<uint64_t>(args->mr_len[1]) ||
        kv_bytes > static_cast<uint64_t>(args->mr_len[2]) ||
        static_cast<uint64_t>(metadata.batch_size) * metadata.num_query_heads *
                (metadata.head_dim + 2) * sizeof(float) >
            static_cast<uint64_t>(args->mr_len[3]) ||
        query_bytes > static_cast<uint64_t>(args->mr_len[4])) {
        std::fprintf(stderr, "dense_attention: memory range is too small\n");
        return false;
    }
    return true;
}

const void *kv_head(const void *base,
                    const cemu_attention_metadata &metadata,
                    uint32_t token,
                    uint32_t batch,
                    uint32_t head)
{
    const uint8_t *token_base = static_cast<const uint8_t *>(base) +
                                static_cast<uint64_t>(token) * metadata.token_stride;
    const uint64_t head_index =
        static_cast<uint64_t>(batch) * metadata.num_kv_heads + head;
    return token_base + head_index * metadata.head_dim * element_size(metadata.dtype);
}

}  // namespace

extern "C" long long dense_attention(struct cemu_args *args)
{
    cemu_attention_metadata metadata = {};
    if (!load_metadata(args, &metadata) || !validate_ranges(args, metadata)) {
        return -1;
    }

    const void *query = args->mr_addr[0];
    const void *key_staging = args->mr_addr[1];
    const void *value_staging = args->mr_addr[2];
    float *state = static_cast<float *>(args->mr_addr[3]);
    void *output = args->mr_addr[4];
    const uint64_t state_stride = metadata.head_dim + 2;
    std::vector<float> scores(metadata.token_count);
    std::vector<float> weighted_sum(metadata.head_dim);

    for (uint32_t batch = 0; batch < metadata.batch_size; ++batch) {
        for (uint32_t query_head = 0;
             query_head < metadata.num_query_heads;
             ++query_head) {
            const uint32_t kv_head_index =
                query_head * metadata.num_kv_heads / metadata.num_query_heads;
            const uint64_t query_index =
                (static_cast<uint64_t>(batch) * metadata.num_query_heads + query_head) *
                metadata.head_dim;
            float *state_record = state +
                (static_cast<uint64_t>(batch) * metadata.num_query_heads +
                 query_head) * state_stride;

            if (metadata.flags & CEMU_ATTENTION_FLAG_RESET_STATE) {
                state_record[0] = -INFINITY;
                state_record[1] = 0.0f;
                for (uint32_t dimension = 0;
                     dimension < metadata.head_dim;
                     ++dimension) {
                    state_record[2 + dimension] = 0.0f;
                }
            }

            float chunk_maximum = -INFINITY;
            for (uint32_t token = 0; token < metadata.token_count; ++token) {
                const void *key_vector =
                    kv_head(key_staging, metadata, token, batch, kv_head_index);
                float score = 0.0f;
                for (uint32_t dimension = 0;
                     dimension < metadata.head_dim;
                     ++dimension) {
                    score += load_value(query, query_index + dimension, metadata.dtype) *
                             load_value(key_vector, dimension, metadata.dtype);
                }
                score *= metadata.scale;
                scores[token] = score;
                chunk_maximum = std::fmax(chunk_maximum, score);
            }

            float chunk_denominator = 0.0f;
            for (uint32_t token = 0; token < metadata.token_count; ++token) {
                scores[token] = std::exp(scores[token] - chunk_maximum);
                chunk_denominator += scores[token];
            }

            const float previous_maximum = state_record[0];
            const float new_maximum = std::fmax(previous_maximum, chunk_maximum);
            const float previous_scale = previous_maximum == -INFINITY
                ? 0.0f
                : std::exp(previous_maximum - new_maximum);
            const float chunk_scale = std::exp(chunk_maximum - new_maximum);
            const float new_denominator =
                state_record[1] * previous_scale + chunk_denominator * chunk_scale;

            for (uint32_t dimension = 0;
                 dimension < metadata.head_dim;
                 ++dimension) {
                float chunk_weighted_sum = 0.0f;
                for (uint32_t token = 0; token < metadata.token_count; ++token) {
                    const void *value_vector =
                        kv_head(value_staging, metadata, token, batch, kv_head_index);
                    chunk_weighted_sum +=
                        scores[token] * load_value(value_vector, dimension, metadata.dtype);
                }
                weighted_sum[dimension] =
                    state_record[2 + dimension] * previous_scale +
                    chunk_weighted_sum * chunk_scale;
            }

            state_record[0] = new_maximum;
            state_record[1] = new_denominator;
            for (uint32_t dimension = 0;
                 dimension < metadata.head_dim;
                 ++dimension) {
                state_record[2 + dimension] = weighted_sum[dimension];
            }

            if (metadata.flags & CEMU_ATTENTION_FLAG_FINALIZE) {
                for (uint32_t dimension = 0;
                     dimension < metadata.head_dim;
                     ++dimension) {
                    store_value(output, query_index + dimension, metadata.dtype,
                                state_record[2 + dimension] / state_record[1]);
                }
            }
        }
    }

    return static_cast<long long>(metadata.batch_size) *
           metadata.num_query_heads * metadata.head_dim;
}
