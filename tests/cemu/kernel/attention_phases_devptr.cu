#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math_constants.h>
#include <cstdio>

#include "attention_phases_common.h"

namespace {

constexpr unsigned int threads_per_block = 256;

template <typename Scalar>
__device__ float read_scalar(Scalar value)
{
    return static_cast<float>(value);
}

template <>
__device__ float read_scalar(__half value)
{
    return __half2float(value);
}

template <bool Maximum>
__device__ float block_reduce(float value, float *scratch)
{
    scratch[threadIdx.x] = value;
    __syncthreads();
    for (unsigned int width = threads_per_block / 2; width; width /= 2) {
        if (threadIdx.x < width) {
            scratch[threadIdx.x] = Maximum
                ? fmaxf(scratch[threadIdx.x], scratch[threadIdx.x + width])
                : scratch[threadIdx.x] + scratch[threadIdx.x + width];
        }
        __syncthreads();
    }
    const float result = scratch[0];
    __syncthreads();
    return result;
}

template <typename Scalar>
__global__ void qk_softmax(const Scalar *query, const uint8_t *keys,
                           float *probability_storage, cemu_attention_phase_metadata config)
{
    __shared__ float scratch[threads_per_block];
    const uint64_t vectors = static_cast<uint64_t>(config.batch_size) * config.num_query_heads;
    for (uint64_t vector = blockIdx.x; vector < vectors; vector += gridDim.x) {
        const uint64_t batch = vector / config.num_query_heads;
        const uint64_t head = vector % config.num_query_heads;
        const uint64_t kv_head = head / (config.num_query_heads / config.num_kv_heads);
        const uint64_t kv_offset = (batch * config.num_kv_heads + kv_head) * config.head_dim;
        float *probabilities = probability_storage + vector * config.token_count;
        float maximum = -CUDART_INF_F;
        for (uint64_t token = threadIdx.x; token < config.token_count; token += blockDim.x) {
            const Scalar *token_keys = reinterpret_cast<const Scalar *>(keys + token * config.token_stride);
            float score = 0.0f;
            for (uint64_t dimension = 0; dimension < config.head_dim; ++dimension) {
                score += read_scalar(query[vector * config.head_dim + dimension]) *
                         read_scalar(token_keys[kv_offset + dimension]);
            }
            probabilities[token] = score * config.scale;
            maximum = fmaxf(maximum, probabilities[token]);
        }
        maximum = block_reduce<true>(maximum, scratch);
        float denominator = 0.0f;
        for (uint64_t token = threadIdx.x; token < config.token_count; token += blockDim.x) {
            const float weight = expf(probabilities[token] - maximum);
            probabilities[token] = weight;
            denominator += weight;
        }
        denominator = block_reduce<false>(denominator, scratch);
        for (uint64_t token = threadIdx.x; token < config.token_count; token += blockDim.x) {
            probabilities[token] /= denominator;
        }
        __syncthreads();
    }
}

template <typename Scalar>
__global__ void probability_value(const float *probability_storage, const uint8_t *values,
                                  Scalar *output, cemu_attention_phase_metadata config)
{
    const uint64_t vectors = static_cast<uint64_t>(config.batch_size) * config.num_query_heads;
    for (uint64_t vector = blockIdx.x; vector < vectors; vector += gridDim.x) {
        const uint64_t batch = vector / config.num_query_heads;
        const uint64_t head = vector % config.num_query_heads;
        const uint64_t kv_head = head / (config.num_query_heads / config.num_kv_heads);
        const uint64_t kv_offset = (batch * config.num_kv_heads + kv_head) * config.head_dim;
        const float *probabilities = probability_storage + vector * config.token_count;
        for (uint64_t dimension = threadIdx.x; dimension < config.head_dim; dimension += blockDim.x) {
            float result = 0.0f;
            for (uint64_t token = 0; token < config.token_count; ++token) {
                const Scalar *token_values = reinterpret_cast<const Scalar *>(values + token * config.token_stride);
                result += probabilities[token] * read_scalar(token_values[kv_offset + dimension]);
            }
            output[vector * config.head_dim + dimension] = static_cast<Scalar>(result);
        }
    }
}

template <typename Scalar>
void launch_phase(cemu_args *args, cemu_attention_phase_metadata config, unsigned int blocks)
{
    if (config.phase == CEMU_ATTENTION_PHASE_QK_SOFTMAX) {
        qk_softmax<<<blocks, threads_per_block, 0, cudaStreamPerThread>>>(
            static_cast<const Scalar *>(args->mr_dev_addr[0]),
            static_cast<const uint8_t *>(args->mr_dev_addr[1]),
            static_cast<float *>(args->mr_dev_addr[2]), config);
    } else {
        probability_value<<<blocks, threads_per_block, 0, cudaStreamPerThread>>>(
            static_cast<const float *>(args->mr_dev_addr[0]),
            static_cast<const uint8_t *>(args->mr_dev_addr[1]),
            static_cast<Scalar *>(args->mr_dev_addr[2]), config);
    }
}

}

extern "C" long long attention_phase(cemu_args *args)
{
    cemu_attention_phase_metadata config;
    if (!validate_attention_phase(args, true, &config)) {
        std::fprintf(stderr, "attention_phase CUDA: invalid metadata or memory ranges\n");
        return -1;
    }
    const uint64_t vectors = static_cast<uint64_t>(config.batch_size) * config.num_query_heads;
    const unsigned int blocks = vectors > 65535 ? 65535 : vectors;
    if (config.dtype == CEMU_ATTENTION_PHASES_FLOAT16) {
        launch_phase<__half>(args, config, blocks);
    } else {
        launch_phase<float>(args, config, blocks);
    }
    cudaError_t status = cudaGetLastError();
    if (status == cudaSuccess) {
        status = cudaStreamSynchronize(cudaStreamPerThread);
    }
    if (status != cudaSuccess) {
        std::fprintf(stderr, "attention_phase CUDA: %s\n", cudaGetErrorString(status));
        return -1;
    }
    return vectors * (config.phase == CEMU_ATTENTION_PHASE_QK_SOFTMAX
                      ? config.token_count : config.head_dim);
}
