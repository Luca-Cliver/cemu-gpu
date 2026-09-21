/* Optional diagnostics. Counters describe NAND page operations, not elapsed
 * request time: operations on different LUNs can overlap. */
#ifndef FEMU_READ_OBSERVATION_H
#define FEMU_READ_OBSERVATION_H
#include <stdint.h>

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
} FemuReadObservation;

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
