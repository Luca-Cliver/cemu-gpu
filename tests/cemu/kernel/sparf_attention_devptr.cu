#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math_constants.h>
#include <cstdio>

#include "sparf_attention_common.h"

namespace {
constexpr unsigned threads = 256;

template <typename Scalar>
__device__ float scalar(Scalar value) { return static_cast<float>(value); }
template <>
__device__ float scalar(__half value) { return __half2float(value); }

template <bool Maximum>
__device__ float reduce(float value, float *scratch)
{
    scratch[threadIdx.x] = value;
    __syncthreads();
    for (unsigned width = threads / 2; width; width /= 2) {
        if (threadIdx.x < width) {
            scratch[threadIdx.x] = Maximum
                ? fmaxf(scratch[threadIdx.x], scratch[threadIdx.x + width])
                : scratch[threadIdx.x] + scratch[threadIdx.x + width];
        }
        __syncthreads();
    }
    return scratch[0];
}

template <typename Scalar>
__global__ void exact_qk(const Scalar *query, const Scalar *keys,
                         const float *alpha, float *probability,
                         cemu_sparf_attention_metadata config)
{
    __shared__ float scratch[threads];
    const uint64_t vectors = static_cast<uint64_t>(config.batch_size) * config.num_query_heads;
    for (uint64_t vector = blockIdx.x; vector < vectors; vector += gridDim.x) {
        const auto *rows = config.version == CEMU_SPARF_ATTENTION_RAGGED_VERSION ?
            reinterpret_cast<const cemu_sparf_batch_row *>(alpha + vectors) : nullptr;
        const auto *r = rows ? &rows[vector / config.num_query_heads] : nullptr;
        const uint32_t selected = r ? r->top_k : config.selected_tokens;
        const uint64_t key_base = r ? static_cast<uint64_t>(r->selected_offset) * config.num_query_heads +
            (vector % config.num_query_heads) * selected : vector * selected;
        const uint64_t probability_base = key_base + vector;
        float maximum = -CUDART_INF_F;
        for (uint32_t token = threadIdx.x; token < selected; token += blockDim.x) {
            float score = 0.0f;
            for (uint32_t dimension = 0; dimension < config.head_dim; ++dimension) {
                score += scalar(query[vector * config.head_dim + dimension]) *
                         scalar(keys[(key_base + token) * config.head_dim + dimension]);
            }
            probability[probability_base + token] = score * config.scale;
            maximum = fmaxf(maximum, score * config.scale);
        }
        maximum = reduce<true>(maximum, scratch);
        float denominator = 0.0f;
        for (uint32_t token = threadIdx.x; token < selected; token += blockDim.x) {
            float &item = probability[probability_base + token];
            item = expf(item - maximum);
            denominator += item;
        }
        denominator = reduce<false>(denominator, scratch);
        for (uint32_t token = threadIdx.x; token < selected; token += blockDim.x) {
            probability[probability_base + token] *= alpha[vector] / denominator;
        }
        if (threadIdx.x == 0) {
            probability[probability_base + selected] = 1.0f - alpha[vector];
        }
        __syncthreads();
    }
}

template <typename Scalar>
__global__ void pv(const float *probability, const Scalar *values, Scalar *output,
                   const float *alpha, cemu_sparf_attention_metadata config)
{
    const uint64_t vectors = static_cast<uint64_t>(config.batch_size) * config.num_query_heads;
    for (uint64_t vector = blockIdx.x; vector < vectors; vector += gridDim.x) {
        const auto *rows = config.version == CEMU_SPARF_ATTENTION_RAGGED_VERSION ?
            reinterpret_cast<const cemu_sparf_batch_row *>(alpha + vectors) : nullptr;
        const auto *r = rows ? &rows[vector / config.num_query_heads] : nullptr;
        const uint32_t selected = r ? r->top_k : config.selected_tokens;
        const uint64_t key_base = r ? static_cast<uint64_t>(r->selected_offset) * config.num_query_heads +
            (vector % config.num_query_heads) * selected : vector * selected;
        const uint64_t probability_base = key_base + vector;
        for (uint32_t dimension = threadIdx.x; dimension < config.head_dim; dimension += blockDim.x) {
            float result = 0.0f;
            for (uint32_t token = 0; token <= selected; ++token) {
                result += probability[probability_base + token] *
                          scalar(values[(probability_base + token) * config.head_dim + dimension]);
            }
            output[vector * config.head_dim + dimension] = static_cast<Scalar>(result);
        }
    }
}

template <typename Scalar>
void launch(cemu_args *args, cemu_sparf_attention_metadata config, unsigned blocks)
{
    if (config.phase == CEMU_SPARF_PHASE_EXACT_QK) {
        exact_qk<<<blocks, threads, 0, cudaStreamPerThread>>>(
            static_cast<const Scalar *>(args->mr_dev_addr[0]),
            static_cast<const Scalar *>(args->mr_dev_addr[1]),
            static_cast<const float *>(args->mr_dev_addr[2]),
            static_cast<float *>(args->mr_dev_addr[3]), config);
    } else {
        pv<<<blocks, threads, 0, cudaStreamPerThread>>>(
            static_cast<const float *>(args->mr_dev_addr[0]),
            static_cast<const Scalar *>(args->mr_dev_addr[1]),
            static_cast<Scalar *>(args->mr_dev_addr[config.version == CEMU_SPARF_ATTENTION_RAGGED_VERSION ? 3 : 2]),
            config.version == CEMU_SPARF_ATTENTION_RAGGED_VERSION ?
                static_cast<const float *>(args->mr_dev_addr[2]) : nullptr, config);
    }
}
}

extern "C" long long sparf_attention(cemu_args *args)
{
    cemu_sparf_attention_metadata config;
    if (!validate_sparf_attention(args, true, &config)) {
        std::fprintf(stderr, "sparf_attention CUDA: invalid metadata or ranges\n");
        return -1;
    }
    const uint64_t vectors = static_cast<uint64_t>(config.batch_size) * config.num_query_heads;
    const unsigned blocks = vectors > 65535 ? 65535 : vectors;
    if (config.dtype == CEMU_SPARF_FLOAT16) launch<__half>(args, config, blocks);
    else launch<float>(args, config, blocks);
    cudaError_t status = cudaGetLastError();
    if (status == cudaSuccess) status = cudaStreamSynchronize(cudaStreamPerThread);
    if (status != cudaSuccess) {
        std::fprintf(stderr, "sparf_attention CUDA: %s\n", cudaGetErrorString(status));
        return -1;
    }
    return vectors * config.head_dim;
}
