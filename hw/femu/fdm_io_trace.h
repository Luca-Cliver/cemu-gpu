#ifndef CEMU_FDM_IO_TRACE_H
#define CEMU_FDM_IO_TRACE_H

#include <stdint.h>
#include <stdbool.h>
#include "fdm_kv_aggregate.h"

typedef enum FdmIoRetry {
    FDM_IO_RETRY_NONE,
    FDM_IO_RETRY_SPACE,
    FDM_IO_RETRY_BUSY,
    FDM_IO_RETRY_DRAIN,
    FDM_IO_RETRY_MODEL,
    FDM_IO_RETRY_AGGREGATE,
} FdmIoRetry;

typedef struct FdmIoTrace {
    bool enabled;
    bool input;
    bool async_aggregate;
    bool background;
    uint64_t accepted_ns, started_ns, finished_ns;
    uint64_t input_slot_wait_ns, packet_bytes;
    uint32_t operation;
    uint64_t session, packet_id, token_first;
    uint64_t sq_seen, memory_enter, locked, dma_begin, dma_end, memory_done;
    uint64_t model_ready, first_check, last_check, cq_begin, cq_end;
    uint64_t exec_enter, exec_admitted, exec_started, exec_ready;
    uint64_t check_ns, aggregate_ns, aggregate_host_ns;
    uint64_t checks, aggregate_calls, space_retries, busy_retries, cq_full;
    uint64_t space_retry_ns, busy_retry_ns, drain_retry_ns, model_retry_ns;
    uint64_t aggregate_yield_ns;
    FdmIoRetry retry;
    FDMKV_AggregateProfile aggregate_profile;
    uint64_t aggregate_probe_calls, aggregate_thread_cpu_ns;
    uint64_t aggregate_minor_faults, aggregate_major_faults;
    uint64_t aggregate_voluntary_switches, aggregate_involuntary_switches;
    uint64_t aggregate_first_host_ns, aggregate_last_host_ns, aggregate_cpu_changes;
    int32_t aggregate_tid, aggregate_cpu_begin, aggregate_cpu_end;
    uint32_t aggregate_batch, aggregate_layer, aggregate_head_first, aggregate_head_count;
} FdmIoTrace;

struct NvmeRequest;
void fdm_io_trace_begin(struct NvmeRequest *request);
void fdm_io_trace_end(struct NvmeRequest *request);
void fdm_io_trace_background(struct NvmeRequest *request, uint16_t qid);
uint64_t fdm_io_trace_check_begin(struct NvmeRequest *request);
void fdm_io_trace_check_end(struct NvmeRequest *request, uint64_t start);

#define FDM_IO_STAMP(request, field) do { \
    if ((request)->fdm_io_trace.enabled) \
        (request)->fdm_io_trace.field = clock_ns(); \
} while (0)

#endif
