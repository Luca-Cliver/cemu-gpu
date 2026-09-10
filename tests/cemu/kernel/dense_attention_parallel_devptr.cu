#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>

#include "attention_abi.h"
#include "cemu_def.h"

namespace {

constexpr uint32_t kThreadsPerBlock = 256;
constexpr uint32_t kWarpSize = 32;
constexpr uint32_t kMaximumWarps = kThreadsPerBlock / kWarpSize;

uint32_t element_size(uint32_t dtype)
{
    return dtype == CEMU_ATTENTION_DTYPE_FLOAT16 ? sizeof(__half) : sizeof(float);
}

bool cuda_ok(cudaError_t error, const char *operation)
{
    if (error == cudaSuccess) {
        return true;
    }
    std::fprintf(stderr, "dense_attention_parallel CUDA error at %s: %s\n",
                 operation, cudaGetErrorString(error));
    return false;
}

size_t shared_memory_limit()
{
    static const size_t limit = []() {
        int device = 0;
        cudaDeviceProp properties = {};
        if (!cuda_ok(cudaGetDevice(&device), "cudaGetDevice") ||
            !cuda_ok(cudaGetDeviceProperties(&properties, device),
                     "cudaGetDeviceProperties")) {
            return static_cast<size_t>(0);
        }
        std::fprintf(
            stderr,
            "[cemu-cuda-attention-parallel] device=%d name=%s cc=%d.%d "
            "sm_count=%d total_mem=%zuMB shared_mem_per_block=%zuB threads=%u\n",
            device,
            properties.name,
            properties.major,
            properties.minor,
            properties.multiProcessorCount,
            static_cast<size_t>(properties.totalGlobalMem / (1024 * 1024)),
            static_cast<size_t>(properties.sharedMemPerBlock),
            kThreadsPerBlock);
        return static_cast<size_t>(properties.sharedMemPerBlock);
    }();
    return limit;
}

bool load_metadata(const cemu_args *args, cemu_attention_metadata *metadata)
{
    if (!args || !metadata || !args->data_buffer ||
        args->buffer_len < static_cast<long long>(sizeof(*metadata))) {
        std::fprintf(stderr, "dense_attention_parallel(devptr): missing metadata\n");
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
                             CEMU_ATTENTION_FLAG_FINALIZE)) != 0 ||
        metadata->batch_size > static_cast<uint32_t>(std::numeric_limits<int>::max()) ||
        metadata->num_query_heads > static_cast<uint32_t>(std::numeric_limits<int>::max())) {
        std::fprintf(stderr, "dense_attention_parallel(devptr): invalid metadata\n");
        return false;
    }
    return true;
}

bool validate_ranges(const cemu_args *args, const cemu_attention_metadata &metadata)
{
    if (args->numr != 5 || !args->mr_dev_addr || !args->mr_len) {
        std::fprintf(
            stderr,
            "dense_attention_parallel(devptr): expected five device ranges\n");
        return false;
    }
    for (int index = 0; index < args->numr; ++index) {
        if (!args->mr_dev_addr[index] || args->mr_len[index] <= 0) {
            std::fprintf(
                stderr,
                "dense_attention_parallel(devptr): invalid range %d\n",
                index);
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
    return query_bytes <= static_cast<uint64_t>(args->mr_len[0]) &&
           kv_bytes <= static_cast<uint64_t>(args->mr_len[1]) &&
           kv_bytes <= static_cast<uint64_t>(args->mr_len[2]) &&
           static_cast<uint64_t>(metadata.batch_size) * metadata.num_query_heads *
                   (metadata.head_dim + 2) * sizeof(float) <=
               static_cast<uint64_t>(args->mr_len[3]) &&
           query_bytes <= static_cast<uint64_t>(args->mr_len[4]);
}

__device__ float storage_to_float(float value)
{
    return value;
}

__device__ float storage_to_float(__half value)
{
    return __half2float(value);
}

__device__ void store_from_float(float *base, uint64_t index, float value)
{
    base[index] = value;
}

__device__ void store_from_float(__half *base, uint64_t index, float value)
{
    base[index] = __float2half_rn(value);
}

template <typename Storage>
__device__ const Storage *kv_head(const uint8_t *base,
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
    return reinterpret_cast<const Storage *>(token_base) + head_index * head_dim;
}

__device__ float warp_reduce_max(float value)
{
    for (uint32_t offset = kWarpSize / 2; offset > 0; offset /= 2) {
        value = fmaxf(value, __shfl_down_sync(0xffffffffU, value, offset));
    }
    return value;
}

__device__ float warp_reduce_sum(float value)
{
    for (uint32_t offset = kWarpSize / 2; offset > 0; offset /= 2) {
        value += __shfl_down_sync(0xffffffffU, value, offset);
    }
    return value;
}

__device__ float block_reduce_max(float value, float *warp_results)
{
    const uint32_t lane = threadIdx.x % kWarpSize;
    const uint32_t warp = threadIdx.x / kWarpSize;
    value = warp_reduce_max(value);
    if (lane == 0) {
        warp_results[warp] = value;
    }
    __syncthreads();

    if (warp == 0) {
        value = lane < kMaximumWarps ? warp_results[lane] : -INFINITY;
        value = warp_reduce_max(value);
        if (lane == 0) {
            warp_results[0] = value;
        }
    }
    __syncthreads();
    return warp_results[0];
}

__device__ float block_reduce_sum(float value, float *warp_results)
{
    const uint32_t lane = threadIdx.x % kWarpSize;
    const uint32_t warp = threadIdx.x / kWarpSize;
    value = warp_reduce_sum(value);
    if (lane == 0) {
        warp_results[warp] = value;
    }
    __syncthreads();

    if (warp == 0) {
        value = lane < kMaximumWarps ? warp_results[lane] : 0.0f;
        value = warp_reduce_sum(value);
        if (lane == 0) {
            warp_results[0] = value;
        }
    }
    __syncthreads();
    return warp_results[0];
}

template <typename Storage>
__global__ void dense_attention_parallel_kernel(const Storage *query,
                                                const uint8_t *key_staging,
                                                const uint8_t *value_staging,
                                                float *state,
                                                Storage *output,
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
    const Storage *query_vector = query + query_offset;
    const uint64_t state_offset =
        static_cast<uint64_t>(query_vector_index) * (head_dim + 2);
    float *state_record = state + state_offset;

    extern __shared__ float probabilities[];
    __shared__ float warp_results[kMaximumWarps];
    __shared__ float previous_scale;
    __shared__ float chunk_scale;
    __shared__ float output_denominator;

    float local_maximum = -INFINITY;
    for (uint32_t token = threadIdx.x; token < token_count;
         token += blockDim.x) {
        const Storage *key_vector = kv_head<Storage>(
            key_staging,
            token_stride,
            num_kv_heads,
            head_dim,
            token,
            batch,
            kv_head_index);
        float score = 0.0f;
        for (uint32_t dimension = 0; dimension < head_dim; ++dimension) {
            score += storage_to_float(query_vector[dimension]) *
                     storage_to_float(key_vector[dimension]);
        }
        score *= scale;
        probabilities[token] = score;
        local_maximum = fmaxf(local_maximum, score);
    }
    const float chunk_maximum = block_reduce_max(local_maximum, warp_results);

    float local_denominator = 0.0f;
    for (uint32_t token = threadIdx.x; token < token_count;
         token += blockDim.x) {
        const float probability = expf(probabilities[token] - chunk_maximum);
        probabilities[token] = probability;
        local_denominator += probability;
    }
    const float chunk_denominator =
        block_reduce_sum(local_denominator, warp_results);

    if (flags & CEMU_ATTENTION_FLAG_RESET_STATE) {
        if (threadIdx.x == 0) {
            state_record[0] = -INFINITY;
            state_record[1] = 0.0f;
        }
        for (uint32_t dimension = threadIdx.x; dimension < head_dim;
             dimension += blockDim.x) {
            state_record[2 + dimension] = 0.0f;
        }
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        const float previous_maximum = state_record[0];
        const float new_maximum = fmaxf(previous_maximum, chunk_maximum);
        previous_scale = previous_maximum == -INFINITY
            ? 0.0f
            : expf(previous_maximum - new_maximum);
        chunk_scale = expf(chunk_maximum - new_maximum);
        output_denominator =
            state_record[1] * previous_scale + chunk_denominator * chunk_scale;
        state_record[0] = new_maximum;
        state_record[1] = output_denominator;
    }
    __syncthreads();

    for (uint32_t dimension = threadIdx.x; dimension < head_dim;
         dimension += blockDim.x) {
        float chunk_weighted_sum = 0.0f;
        for (uint32_t token = 0; token < token_count; ++token) {
            const Storage *value_vector = kv_head<Storage>(
                value_staging,
                token_stride,
                num_kv_heads,
                head_dim,
                token,
                batch,
                kv_head_index);
            chunk_weighted_sum +=
                probabilities[token] * storage_to_float(value_vector[dimension]);
        }
        const float weighted_sum =
            state_record[2 + dimension] * previous_scale +
            chunk_weighted_sum * chunk_scale;
        state_record[2 + dimension] = weighted_sum;
        if (flags & CEMU_ATTENTION_FLAG_FINALIZE) {
            store_from_float(
                output,
                query_offset + dimension,
                weighted_sum / output_denominator);
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

    const size_t shared_limit = shared_memory_limit();
    if (shared_limit == 0) {
        return -1;
    }
    const size_t shared_bytes =
        static_cast<size_t>(metadata.token_count) * sizeof(float);
    constexpr size_t static_shared_bytes =
        (kMaximumWarps + 4) * sizeof(float);
    if (shared_bytes + static_shared_bytes > shared_limit) {
        std::fprintf(
            stderr,
            "dense_attention_parallel(devptr): %zu shared bytes exceed "
            "device limit %zu\n",
            shared_bytes + static_shared_bytes,
            shared_limit);
        return -1;
    }

    const uint32_t block_count = metadata.batch_size * metadata.num_query_heads;
    if (metadata.dtype == CEMU_ATTENTION_DTYPE_FLOAT16) {
        dense_attention_parallel_kernel<<<
            block_count, kThreadsPerBlock, shared_bytes>>>(
            static_cast<const __half *>(args->mr_dev_addr[0]),
            static_cast<const uint8_t *>(args->mr_dev_addr[1]),
            static_cast<const uint8_t *>(args->mr_dev_addr[2]),
            static_cast<float *>(args->mr_dev_addr[3]),
            static_cast<__half *>(args->mr_dev_addr[4]),
            metadata.batch_size,
            metadata.num_query_heads,
            metadata.num_kv_heads,
            metadata.head_dim,
            metadata.token_count,
            metadata.token_stride,
            metadata.scale,
            metadata.flags);
    } else {
        dense_attention_parallel_kernel<<<
            block_count, kThreadsPerBlock, shared_bytes>>>(
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
    }
    if (!cuda_ok(cudaGetLastError(), "dense_attention_parallel_kernel launch") ||
        !cuda_ok(cudaDeviceSynchronize(), "cudaDeviceSynchronize")) {
        return -1;
    }

    return static_cast<long long>(metadata.batch_size) *
           metadata.num_query_heads * metadata.head_dim;
}
