#include <cuda_runtime.h>

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "cemu_def.h"

static void cuda_check(cudaError_t err, const char *what)
{
    if (err != cudaSuccess) {
        std::fprintf(stderr, "CUDA error at %s: %s\n", what, cudaGetErrorString(err));
        std::abort();
    }
}

static void log_cuda_device_once()
{
    static std::atomic<bool> logged{false};
    bool expected = false;
    if (!logged.compare_exchange_strong(expected, true)) {
        return;
    }

    int device = -1;
    cuda_check(cudaGetDevice(&device), "cudaGetDevice");

    cudaDeviceProp prop {};
    cuda_check(cudaGetDeviceProperties(&prop, device), "cudaGetDeviceProperties");

    int runtime_version = 0;
    int driver_version = 0;
    cuda_check(cudaRuntimeGetVersion(&runtime_version), "cudaRuntimeGetVersion");
    cuda_check(cudaDriverGetVersion(&driver_version), "cudaDriverGetVersion");

    std::fprintf(stderr,
                 "[cemu-cuda] device=%d name=%s cc=%d.%d sm_count=%d total_mem=%zuMB runtime=%d driver=%d\n",
                 device,
                 prop.name,
                 prop.major,
                 prop.minor,
                 prop.multiProcessorCount,
                 static_cast<size_t>(prop.totalGlobalMem / (1024 * 1024)),
                 runtime_version,
                 driver_version);
}

__global__ void vadd_kernel(const int *input, int *output, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = input[idx * 2] + input[idx * 2 + 1];
    }
}

static long long vadd_gpu(struct cemu_args *args)
{
    log_cuda_device_once();

    int numr = args->numr;
    void **mr_addr = args->mr_addr;
    long long *mr_len = args->mr_len;
    long long size = args->cparam1;
    (void)numr;
    (void)mr_len;
    (void)args->cparam2;
    (void)args->data_buffer;
    (void)args->buffer_len;

    int *host_input = static_cast<int *>(mr_addr[1]);
    int *host_output = static_cast<int *>(mr_addr[0]);
    size_t input_elems = static_cast<size_t>(size) * 2;
    size_t input_bytes = input_elems * sizeof(int);
    size_t output_bytes = static_cast<size_t>(size) * sizeof(int);

    int *device_input = nullptr;
    int *device_output = nullptr;

    cuda_check(cudaMalloc(&device_input, input_bytes), "cudaMalloc(device_input)");
    cuda_check(cudaMalloc(&device_output, output_bytes), "cudaMalloc(device_output)");
    cuda_check(cudaMemcpy(device_input, host_input, input_bytes, cudaMemcpyHostToDevice),
               "cudaMemcpy H2D");

    constexpr int threads_per_block = 256;
    int blocks = static_cast<int>((size + threads_per_block - 1) / threads_per_block);
    vadd_kernel<<<blocks, threads_per_block>>>(device_input, device_output, static_cast<int>(size));
    cuda_check(cudaGetLastError(), "vadd_kernel launch");
    cuda_check(cudaDeviceSynchronize(), "cudaDeviceSynchronize");

    cuda_check(cudaMemcpy(host_output, device_output, output_bytes, cudaMemcpyDeviceToHost),
               "cudaMemcpy D2H");

    cudaFree(device_input);
    cudaFree(device_output);
    return size;
}

extern "C" long long vadd(struct cemu_args *args)
{
    return vadd_gpu(args);
}

extern "C" long long vadd_indirect(struct cemu_args *args)
{
    log_cuda_device_once();

    int numr = args->numr;
    void **mr_addr = args->mr_addr;
    long long *mr_len = args->mr_len;
    long long cparam1 = args->cparam1;
    (void)mr_len;
    (void)args->cparam2;
    (void)args->data_buffer;
    (void)args->buffer_len;

    if (numr < 3) {
        std::fprintf(stderr, "vadd_indirect: expected at least 3 memory ranges, got %d\n", numr);
        return -1;
    }

    int *output = static_cast<int *>(mr_addr[0]);
    int *input = static_cast<int *>(mr_addr[1]);
    int *global_mem = static_cast<int *>(mr_addr[2]);

    long long size = cparam1;
    int pos = global_mem[0];
    int start_loc = global_mem[1];

    if (start_loc > 0 && pos > 0) {
        std::memmove(output, input + start_loc, static_cast<size_t>(pos - start_loc) * sizeof(int));
        pos -= start_loc;
        start_loc = 0;
    }

    size_t in_bytes = static_cast<size_t>(size) * 2 * sizeof(int);
    size_t out_bytes = static_cast<size_t>(size) * sizeof(int);
    int *device_input = nullptr;
    int *device_output = nullptr;

    cuda_check(cudaMalloc(&device_input, in_bytes), "cudaMalloc(device_input)");
    cuda_check(cudaMalloc(&device_output, out_bytes), "cudaMalloc(device_output)");
    cuda_check(cudaMemcpy(device_input, input, in_bytes, cudaMemcpyHostToDevice),
               "cudaMemcpy H2D");

    constexpr int threads_per_block = 256;
    int blocks = static_cast<int>((size + threads_per_block - 1) / threads_per_block);
    vadd_kernel<<<blocks, threads_per_block>>>(device_input, device_output, static_cast<int>(size));
    cuda_check(cudaGetLastError(), "vadd_kernel launch");
    cuda_check(cudaDeviceSynchronize(), "cudaDeviceSynchronize");

    cuda_check(cudaMemcpy(output + pos, device_output, out_bytes, cudaMemcpyDeviceToHost),
               "cudaMemcpy D2H");

    pos += static_cast<int>(size);
    int full_blocks = pos / (512 / sizeof(int));
    if (full_blocks > 0) {
        global_mem[1] = full_blocks * (512 / sizeof(int));
    }
    global_mem[0] = pos;

    cudaFree(device_input);
    cudaFree(device_output);
    return full_blocks;
}