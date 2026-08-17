#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>

#include "attention_abi.h"
#include "cemu_def.h"

namespace {

bool cuda_ok(cudaError_t error, const char *operation)
{
    if (error == cudaSuccess) {
        return true;
    }
    std::fprintf(stderr, "dense_attention CUDA error at %s: %s\n",
                 operation, cudaGetErrorString(error));
    return false;
}

bool load_metadata(const cemu_args *args, cemu_attention_metadata *metadata)
{
    if (!args || !metadata || !args->data_buffer ||
        args->buffer_len < static_cast<long long>(sizeof(*metadata))) {
        std::fprintf(stderr, "dense_attention(devptr): missing metadata\n");
        return false;
    }
    std::memcpy(metadata, args->data_buffer, sizeof(*metadata));
    if (metadata->version != CEMU_ATTENTION_ABI_VERSION ||
        metadata->dtype != CEMU_ATTENTION_DTYPE_FLOAT32 ||
        metadata->batch_size == 0 || metadata->num_query_heads == 0 ||
        metadata->num_kv_heads == 0 || metadata->head_dim == 0 ||
        metadata->token_count == 0 ||
        metadata->num_query_heads % metadata->num_kv_heads != 0 ||
        metadata->token_stride % 512 != 0 ||
        metadata->token_stride < metadata->batch_size * metadata->num_kv_heads *
                                     metadata->head_dim * sizeof(float) ||
        !std::isfinite(metadata->scale) || metadata->scale <= 0.0f ||
        (metadata->flags & ~(CEMU_ATTENTION_FLAG_RESET_STATE |
                             CEMU_ATTENTION_FLAG_FINALIZE)) != 0 ||
        metadata->batch_size > static_cast<uint32_t>(std::numeric_limits<int>::max()) ||
        metadata->num_query_heads > static_cast<uint32_t>(std::numeric_limits<int>::max())) {
        std::fprintf(stderr, "dense_attention(devptr): invalid metadata\n");
        return false;
    }
    return true;
}

bool validate_ranges(const cemu_args *args, const cemu_attention_metadata &metadata)
{
    if (args->numr != 5 || !args->mr_dev_addr || !args->mr_len) {
        std::fprintf(stderr, "dense_attention(devptr): expected five device ranges\n");
        return false;
    }
    for (int index = 0; index < args->numr; ++index) {
        if (!args->mr_dev_addr[index] || args->mr_len[index] <= 0) {
            std::fprintf(stderr, "dense_attention(devptr): invalid range %d\n", index);
            return false;
        }
    }

    const uint64_t query_bytes =
        static_cast<uint64_t>(metadata.batch_size) * metadata.num_query_heads *
        metadata.head_dim * sizeof(float);
    const uint64_t token_payload_bytes =
        static_cast<uint64_t>(metadata.batch_size) * metadata.num_kv_heads *
        metadata.head_dim * sizeof(float);
    const uint64_t kv_bytes =
        static_cast<uint64_t>(metadata.token_count - 1) * metadata.token_stride +
        token_payload_bytes;
    return query_bytes <= static_cast<uint64_t>(args->mr_len[0]) &&
           kv_bytes <= static_cast<uint64_t>(args->mr_len[1]) &&
           kv_bytes <= static_cast<uint64_t>(args->mr_len[2]) &&
           static_cast<uint64_t>(metadata.batch_size) * metadata.num_query_heads *
                   (metadata.head_dim + 2) * sizeof(float) <=
               static_cast<uint64_t>(args->mr_len[3]) &&
           query_bytes <= static_cast<uint64_t>(args->mr_len[4]);
}

__device__ const float *kv_head(const uint8_t *base,
                                uint32_t token_stride,
                                uint32_t num_kv_heads,
                                uint32_t head_dim,
                                uint32_t token,
                                uint32_t batch,
                                uint32_t head)
{
    const uint8_t *token_base =
        base + static_cast<uint64_t>(token) * token_stride;
    const uint64_t head_index =
        static_cast<uint64_t>(batch) * num_kv_heads + head;
    return reinterpret_cast<const float *>(token_base) + head_index * head_dim;
}

__global__ void dense_attention_kernel(const float *query,
                                       const uint8_t *key_staging,
                                       const uint8_t *value_staging,
                                       float *state,
                                       float *output,
                                       uint32_t batch_size,
                                       uint32_t num_query_heads,
                                       uint32_t num_kv_heads,
                                       uint32_t head_dim,
                                       uint32_t token_count,
                                       uint32_t token_stride,
                                       float scale,
                                       uint32_t flags)
{
    const uint32_t query_vector_index = blockIdx.x;
    const uint32_t batch = query_vector_index / num_query_heads;
    const uint32_t query_head = query_vector_index % num_query_heads;
    if (batch >= batch_size) {
        return;
    }

    const uint32_t kv_head_index =
        query_head * num_kv_heads / num_query_heads;
    const uint64_t query_offset =
        static_cast<uint64_t>(query_vector_index) * head_dim;
    const float *query_vector = query + query_offset;
    extern __shared__ float probabilities[];

    if (threadIdx.x == 0) {
        float chunk_maximum = -INFINITY;
        for (uint32_t token = 0; token < token_count; ++token) {
            const float *key_vector = kv_head(
                key_staging, token_stride, num_kv_heads, head_dim,
                token, batch, kv_head_index);
            float score = 0.0f;
            for (uint32_t dimension = 0; dimension < head_dim; ++dimension) {
                score += query_vector[dimension] * key_vector[dimension];
            }
            score *= scale;
            probabilities[token] = score;
            chunk_maximum = fmaxf(chunk_maximum, score);
        }

        float chunk_denominator = 0.0f;
        for (uint32_t token = 0; token < token_count; ++token) {
            probabilities[token] = expf(probabilities[token] - chunk_maximum);
            chunk_denominator += probabilities[token];
        }

        const uint64_t state_offset =
            static_cast<uint64_t>(query_vector_index) * (head_dim + 2);
        float *state_record = state + state_offset;
        if (flags & CEMU_ATTENTION_FLAG_RESET_STATE) {
            state_record[0] = -INFINITY;
            state_record[1] = 0.0f;
            for (uint32_t dimension = 0; dimension < head_dim; ++dimension) {
                state_record[2 + dimension] = 0.0f;
            }
        }

        const float previous_maximum = state_record[0];
        const float new_maximum = fmaxf(previous_maximum, chunk_maximum);
        const float previous_scale = previous_maximum == -INFINITY
            ? 0.0f
            : expf(previous_maximum - new_maximum);
        const float chunk_scale = expf(chunk_maximum - new_maximum);
        const float new_denominator =
            state_record[1] * previous_scale + chunk_denominator * chunk_scale;

        for (uint32_t dimension = 0; dimension < head_dim; ++dimension) {
            float chunk_weighted_sum = 0.0f;
            for (uint32_t token = 0; token < token_count; ++token) {
                const float *value_vector = kv_head(
                    value_staging, token_stride, num_kv_heads, head_dim,
                    token, batch, kv_head_index);
                chunk_weighted_sum += probabilities[token] * value_vector[dimension];
            }
            state_record[2 + dimension] =
                state_record[2 + dimension] * previous_scale +
                chunk_weighted_sum * chunk_scale;
        }
        state_record[0] = new_maximum;
        state_record[1] = new_denominator;

        if (flags & CEMU_ATTENTION_FLAG_FINALIZE) {
            float *output_vector = output + query_offset;
            for (uint32_t dimension = 0; dimension < head_dim; ++dimension) {
                output_vector[dimension] =
                    state_record[2 + dimension] / state_record[1];
            }
        }
    }
}

}  // namespace

extern "C" long long dense_attention(struct cemu_args *args)
{
    cemu_attention_metadata metadata = {};
    if (!load_metadata(args, &metadata) || !validate_ranges(args, metadata)) {
        return -1;
    }

    int device = 0;
    cudaDeviceProp properties = {};
    if (!cuda_ok(cudaGetDevice(&device), "cudaGetDevice") ||
        !cuda_ok(cudaGetDeviceProperties(&properties, device),
                 "cudaGetDeviceProperties")) {
        return -1;
    }
    const size_t shared_bytes =
        static_cast<size_t>(metadata.token_count) * sizeof(float);
    if (shared_bytes > properties.sharedMemPerBlock) {
        std::fprintf(stderr,
                     "dense_attention(devptr): %zu shared bytes exceed device limit %zu\n",
                     shared_bytes,
                     static_cast<size_t>(properties.sharedMemPerBlock));
        return -1;
    }

    const uint32_t block_count = metadata.batch_size * metadata.num_query_heads;
    constexpr uint32_t threads_per_block = 256;
    dense_attention_kernel<<<block_count, threads_per_block, shared_bytes>>>(
        static_cast<const float *>(args->mr_dev_addr[0]),
        static_cast<const uint8_t *>(args->mr_dev_addr[1]),
        static_cast<const uint8_t *>(args->mr_dev_addr[2]),
        static_cast<float *>(args->mr_dev_addr[3]),
        static_cast<float *>(args->mr_dev_addr[4]),
        metadata.batch_size,
        metadata.num_query_heads,
        metadata.num_kv_heads,
        metadata.head_dim,
        metadata.token_count,
        metadata.token_stride,
        metadata.scale,
        metadata.flags);
    if (!cuda_ok(cudaGetLastError(), "dense_attention_kernel launch") ||
        !cuda_ok(cudaDeviceSynchronize(), "cudaDeviceSynchronize")) {
        return -1;
    }

    return static_cast<long long>(metadata.batch_size) *
           metadata.num_query_heads * metadata.head_dim;
}
