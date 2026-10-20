#define _POSIX_C_SOURCE 200809L
#include <errno.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include "cemu_def.h"

static uint64_t host_clock_ns(void)
{
    struct timespec timestamp;
    if (clock_gettime(CLOCK_MONOTONIC, &timestamp)) {
        return 0;
    }
    return (uint64_t)timestamp.tv_sec * 1000000000ULL + timestamp.tv_nsec;
}

long long freeze_clock_probe(struct cemu_args *args)
{
    uint64_t result[5];
    struct timespec remaining;
    if (!args || args->numr != 1 || !args->mr_addr || !args->mr_addr[0] ||
        !args->mr_len || args->mr_len[0] < (long long)sizeof(result) ||
        args->cparam1 < 0 || args->cparam1 > 2000000000LL) {
        return -1;
    }
    result[0] = 0x43454d55434c4b31ULL;
    result[1] = args->cparam2;
    result[2] = args->cparam1;
    result[3] = host_clock_ns();
    remaining.tv_sec = args->cparam1 / 1000000000LL;
    remaining.tv_nsec = args->cparam1 % 1000000000LL;
    if (args->cparam1) {
        while (nanosleep(&remaining, &remaining)) {
            if (errno != EINTR) return -1;
        }
    }
    result[4] = host_clock_ns();
    if (!result[3] || result[4] < result[3]) return -1;
    memcpy(args->mr_addr[0], result, sizeof(result));
    return 0;
}
