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

    std::fprintf(stderr,
                 "[cemu-cuda-devptr] device=%d name=%s cc=%d.%d sm_count=%d total_mem=%zuMB\n",
                 device,
                 prop.name,
                 prop.major,
                 prop.minor,
                 prop.multiProcessorCount,
                 static_cast<size_t>(prop.totalGlobalMem / (1024 * 1024)));
}

__global__ void vadd_kernel(const int *input, int *output, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = input[idx * 2] + input[idx * 2 + 1];
    }
}

extern "C" long long vadd(struct cemu_args *args)
{
    log_cuda_device_once();

    if (!args || !args->mr_dev_addr || !args->mr_dev_addr[0] || !args->mr_dev_addr[1]) {
        std::fprintf(stderr, "vadd(devptr): missing device pointers\n");
        return -1;
    }

    int size = static_cast<int>(args->cparam1);
    int *device_input = static_cast<int *>(args->mr_dev_addr[0]);
    int *device_output = static_cast<int *>(args->mr_dev_addr[1]);
    int *host_output = static_cast<int *>(args->mr_addr[1]);

    constexpr int threads_per_block = 256;
    int blocks = (size + threads_per_block - 1) / threads_per_block;

    vadd_kernel<<<blocks, threads_per_block>>>(device_input, device_output, size);
    cuda_check(cudaGetLastError(), "vadd_kernel launch");
    cuda_check(cudaDeviceSynchronize(), "cudaDeviceSynchronize");

    if (host_output) {
        // cuda_check(cudaMemcpy(host_output, device_output,
        //                       static_cast<size_t>(size) * sizeof(int),
        //                       cudaMemcpyDeviceToHost),
        //            "cudaMemcpy D2H");
    }

    return size;
}

extern "C" long long vadd_indirect(struct cemu_args *args)
{
    if (!args || !args->mr_dev_addr || !args->mr_dev_addr[0] || !args->mr_dev_addr[1]) {
        std::fprintf(stderr, "vadd_indirect(devptr): missing device pointers\n");
        return -1;
    }

    if (args->numr < 3 || !args->mr_dev_addr[2]) {
        std::fprintf(stderr, "vadd_indirect(devptr): expected at least 3 memory ranges\n");
        return -1;
    }

    int size = static_cast<int>(args->cparam1);
    int *device_output = static_cast<int *>(args->mr_dev_addr[0]);
    int *device_input = static_cast<int *>(args->mr_dev_addr[1]);
    int *device_global = static_cast<int *>(args->mr_dev_addr[2]);
    int *host_output = static_cast<int *>(args->mr_addr[0]);

    int pos = device_global[0];
    int start_loc = device_global[1];
    if (start_loc > 0 && pos > 0 && host_output) {
        std::memmove(host_output, static_cast<int *>(args->mr_addr[1]) + start_loc,
                     static_cast<size_t>(pos - start_loc) * sizeof(int));
        pos -= start_loc;
        start_loc = 0;
    }

    constexpr int threads_per_block = 256;
    int blocks = (size + threads_per_block - 1) / threads_per_block;
    vadd_kernel<<<blocks, threads_per_block>>>(device_input, device_output, size);
    cuda_check(cudaGetLastError(), "vadd_kernel launch");
    cuda_check(cudaDeviceSynchronize(), "cudaDeviceSynchronize");

    if (host_output) {
        cuda_check(cudaMemcpy(host_output + pos, device_output,
                              static_cast<size_t>(size) * sizeof(int),
                              cudaMemcpyDeviceToHost),
                   "cudaMemcpy D2H");
    }

    pos += size;
    int full_blocks = pos / (512 / sizeof(int));
    if (full_blocks > 0) {
        device_global[1] = full_blocks * (512 / sizeof(int));
    }
    device_global[0] = pos;

    return full_blocks;
}
