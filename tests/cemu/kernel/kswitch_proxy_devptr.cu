#include <cuda_runtime.h>

#include <atomic>
#include <cstdio>
#include <cstdlib>

#include "cemu_def.h"

static void compute_ref(const uint32_t *in, uint32_t *out, int n, int rounds)
{
    const uint32_t q = 12289U;
    const uint32_t q_mask = (1U << 14) - 1;
    const uint32_t alpha[8] = {173U, 541U, 947U, 1223U, 1877U, 2311U, 2789U, 3251U};

    for (int i = 0; i < n; ++i) {
        uint64_t acc = (static_cast<uint64_t>(in[i]) + 17U * static_cast<uint64_t>(i)) % q;
        for (int r = 0; r < rounds; ++r) {
            uint64_t x = (acc + static_cast<uint64_t>(r) * 131U) & q_mask;
            uint64_t sum = 0;
            for (int l = 0; l < 8; ++l) {
                uint64_t digit = (x >> l) & 0xF;
                uint64_t mix = (digit + 3U * static_cast<uint64_t>(l) + 5U) % q;
                sum += (mix * alpha[l]) % q;
            }
            acc = (sum + (acc * 8191U) + 127U) % q;
        }
        out[i] = static_cast<uint32_t>(acc);
    }
}

static uint64_t checksum_u32(const uint32_t *data, int n)
{
    const uint8_t *bytes = reinterpret_cast<const uint8_t *>(data);
    const size_t len = static_cast<size_t>(n) * sizeof(uint32_t);
    uint64_t hash = 1469598103934665603ULL;

    for (size_t i = 0; i < len; ++i) {
        hash ^= static_cast<uint64_t>(bytes[i]);
        hash *= 1099511628211ULL;
    }

    return hash;
}

namespace {

__constant__ uint32_t kAlpha[8] = {173U, 541U, 947U, 1223U, 1877U, 2311U, 2789U, 3251U};

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

__device__ __forceinline__ uint64_t mix_round(uint64_t acc, int round)
{
    const uint32_t q = 12289U;
    const uint32_t q_mask = (1U << 14) - 1;

    uint64_t x = (acc + static_cast<uint64_t>(round) * 131U) & q_mask;
    uint64_t sum = 0;

    for (int l = 0; l < 8; ++l) {
        uint64_t digit = (x >> l) & 0xFU;
        uint64_t mix = (digit + 3U * static_cast<uint64_t>(l) + 5U) % q;
        sum += (static_cast<uint64_t>(mix) * kAlpha[l]) % q;
    }

    return (sum + (acc * 8191U) + 127U) % q;
}

__global__ void kswitch_proxy_kernel(const uint32_t *input, uint32_t *output, int n, int rounds)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) {
        return;
    }

    const uint32_t q = 12289U;
    uint64_t acc = (static_cast<uint64_t>(input[idx]) + 17U * static_cast<uint64_t>(idx)) % q;
    for (int r = 0; r < rounds; ++r) {
        acc = mix_round(acc, r);
    }
    output[idx] = static_cast<uint32_t>(acc);
}

} // namespace

extern "C" long long kswitch_proxy(struct cemu_args *args)
{
    log_cuda_device_once();

    if (!args || !args->mr_dev_addr || !args->mr_dev_addr[0] || !args->mr_dev_addr[1]) {
        std::fprintf(stderr, "kswitch_proxy(devptr): missing device pointers\n");
        return -1;
    }

    const int n = static_cast<int>(args->cparam1);
    const int rounds = (args->cparam2 > 0) ? static_cast<int>(args->cparam2) : 24;

    const uint32_t *device_input = static_cast<const uint32_t *>(args->mr_dev_addr[0]);
    uint32_t *device_output = static_cast<uint32_t *>(args->mr_dev_addr[1]);
    uint32_t *host_output = static_cast<uint32_t *>(args->mr_addr[1]);
    const uint32_t *host_input = static_cast<const uint32_t *>(args->mr_addr[0]);

    // std::printf("kswitch_proxy(devptr): host_input_checksum=0x%016llx\n",
    //             static_cast<unsigned long long>(checksum_u32(host_input, n)));

    // /* print pointer addresses to detect aliasing */
    // std::printf("kswitch_proxy(devptr): host_input_ptr=%p host_output_ptr=%p\n", (void *)host_input, (void *)host_output);
    // std::printf("kswitch_proxy(devptr): device_input_ptr=%p device_output_ptr=%p\n", (void *)device_input, (void *)device_output);

    constexpr int threads_per_block = 256;
    const int blocks = (n + threads_per_block - 1) / threads_per_block;

    kswitch_proxy_kernel<<<blocks, threads_per_block>>>(device_input, device_output, n, rounds);
    cuda_check(cudaGetLastError(), "kswitch_proxy_kernel launch");
    cuda_check(cudaDeviceSynchronize(), "cudaDeviceSynchronize");

    if (host_output) {
        // cuda_check(cudaMemcpy(host_output, device_output,
        //                       static_cast<size_t>(n) * sizeof(uint32_t),
        //                       cudaMemcpyDeviceToHost),
        //            "cudaMemcpy D2H");
        // std::printf("kswitch_proxy(devptr): checksum=0x%016llx\n",
        //             static_cast<unsigned long long>(checksum_u32(host_output, n)));

        // /* compute CPU reference and compare */
        // uint32_t *ref = (uint32_t *)malloc(static_cast<size_t>(n) * sizeof(uint32_t));
        // if (ref) {
        //     compute_ref(host_input, ref, n, rounds);
        //     uint64_t ref_cs = checksum_u32(ref, n);
        //     std::printf("kswitch_proxy(devptr): ref_checksum=0x%016llx\n",
        //                 static_cast<unsigned long long>(ref_cs));

        //     int mismatches = 0;
        //     int first_idx = -1;
        //     for (int i = 0; i < n; ++i) {
        //         if (host_output[i] != ref[i]) {
        //             if (mismatches == 0) {
        //                 first_idx = i;
        //                 std::printf("kswitch_proxy(devptr): first_mismatch idx=%d gpu=0x%08x cpu=0x%08x\n",
        //                             i, host_output[i], ref[i]);
        //                 int start = (i - 8) > 0 ? (i - 8) : 0;
        //                 int end = (i + 8) < n ? (i + 8) : (n - 1);
        //                 std::printf("kswitch_proxy(devptr): dumping neighborhood [%d..%d]\n", start, end);
        //                 for (int j = start; j <= end; ++j) {
        //                     std::printf("  idx=%6d in=0x%08x gpu=0x%08x cpu=0x%08x\n",
        //                                 j, host_input[j], host_output[j], ref[j]);
        //                 }
        //             }
        //             ++mismatches;
        //             if (mismatches > 10) break;
        //         }
        //     }
        //     std::printf("kswitch_proxy(devptr): mismatches=%d\n", mismatches);
        //     free(ref);
        // } else {
        //     std::fprintf(stderr, "kswitch_proxy(devptr): malloc failed for ref buffer\n");
        // }
    } else {
        std::fprintf(stderr, "kswitch_proxy(devptr): host output buffer missing, checksum skipped\n");
    }

    return n;
}