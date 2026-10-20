/* Optional diagnostics. Counters describe NAND page operations, not elapsed
 * request time: operations on different LUNs can overlap. */
#ifndef FEMU_READ_OBSERVATION_H
#define FEMU_READ_OBSERVATION_H
#include <stdint.h>
#include <stdbool.h>

typedef struct FemuReadObservation {
    uint64_t logical_bytes, submitted_bytes, ranges;
    uint64_t tail_bytes, tail_copies, logical_spans, aligned_bytes_before_dedup;
    uint64_t model_start_ns, model_end_ns;
    uint64_t pages, unmapped_pages, service_ns, queue_ns, max_queue_ns;
    uint64_t critical_queue_ns, critical_service_ns;
    uint64_t critical_initial_wait_ns;
    uint32_t channels, luns_per_channel, page_bytes, page_read_ns;
    uint64_t *lun_pages;
    uint64_t *lun_initial_wait_ns;
    bool full_scan_valid;
    uint64_t full_scan_pages, full_scan_flash_bytes;
    uint64_t full_scan_logical_bytes, full_scan_tail_bytes;
    uint64_t unique_flash_bytes;
} FemuReadObservation;

static inline uint64_t femu_observe_unique_bytes(uint64_t *coverage,
                                                uint32_t offset, uint32_t bytes)
{
    uint64_t added = 0;
    while (bytes) {
        uint32_t bit = offset % 64;
        uint32_t count = bytes < 64 - bit ? bytes : 64 - bit;
        uint64_t mask = count == 64 ? UINT64_MAX : ((UINT64_C(1) << count) - 1) << bit;
        added += __builtin_popcountll(mask & ~coverage[offset / 64]);
        coverage[offset / 64] |= mask;
        offset += count;
        bytes -= count;
    }
    return added;
}

static inline void femu_observe_page_read(FemuReadObservation *obs,
                                         uint32_t lun, uint64_t service,
                                         uint64_t latency)
{
    if (!obs) return;
    uint64_t queue = latency - service;
    if (!obs->lun_pages[lun]) obs->lun_initial_wait_ns[lun] = queue;
    obs->pages++;
    obs->lun_pages[lun]++;
    obs->service_ns += service;
    obs->queue_ns += queue;
    if (queue > obs->max_queue_ns) obs->max_queue_ns = queue;
    if (latency > obs->critical_queue_ns + obs->critical_service_ns) {
        obs->critical_queue_ns = queue;
        obs->critical_service_ns = service;
        obs->critical_initial_wait_ns = obs->lun_initial_wait_ns[lun];
    }
}
#endif
