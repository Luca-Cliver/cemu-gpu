#include <cstdint>
#include <cstdio>

#include "cemu_def.h"

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

// A compute-heavy proxy for keyswitch-like arithmetic.
// It intentionally amplifies modular multiply-accumulate work on CPU.
extern "C" long long kswitch_proxy(struct cemu_args *args)
{
    if (!args || args->numr < 2 || !args->mr_addr || !args->mr_addr[0] || !args->mr_addr[1]) {
        std::fprintf(stderr, "kswitch_proxy: invalid args\n");
        return -1;
    }

    const int n = static_cast<int>(args->cparam1);
    const int rounds = (args->cparam2 > 0) ? static_cast<int>(args->cparam2) : 24;

    const uint32_t q = 12289U;
    const uint32_t q_mask = (1U << 14) - 1;
    const uint32_t alpha[8] = {173U, 541U, 947U, 1223U, 1877U, 2311U, 2789U, 3251U};

    const uint32_t *in = static_cast<const uint32_t *>(args->mr_addr[0]);
    uint32_t *out = static_cast<uint32_t *>(args->mr_addr[1]);

    std::printf("kswitch_proxy: input_checksum=0x%016llx\n",
                static_cast<unsigned long long>(checksum_u32(in, n)));

    for (int i = 0; i < n; ++i) {
        uint64_t acc = (static_cast<uint64_t>(in[i]) + 17U * static_cast<uint64_t>(i)) % q;

        for (int r = 0; r < rounds; ++r) {
            // Simulate gadget decomposition + relinearization style arithmetic.
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
    std::printf("kswitch_proxy: finished processing %d elements\n", n);
    std::printf("kswitch_proxy: checksum=0x%016llx\n",
                static_cast<unsigned long long>(checksum_u32(out, n)));

    return n;
}
