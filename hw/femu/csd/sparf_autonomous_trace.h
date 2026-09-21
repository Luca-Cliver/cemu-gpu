#ifndef CEMU_SPARF_AUTONOMOUS_TRACE_H
#define CEMU_SPARF_AUTONOMOUS_TRACE_H

#include <glib.h>
#include <stdint.h>
#include <stdbool.h>
#include "hw/femu/read_observation.h"

typedef struct SparfTraceStamp {
    uint64_t host_ns, virtual_ns;
} SparfTraceStamp;

typedef struct SparfCuSpan {
    uint64_t start_ns, end_ns;
    int cu;
} SparfCuSpan;

typedef struct SparfComputeTrace {
    SparfTraceStamp functional_start, functional_end, observed_done;
    uint64_t submitted_ns, completed_ns, effective_ns;
    bool scheduled, truncated;
    GArray *spans;
} SparfComputeTrace;

typedef struct SparfReadTrace {
    SparfTraceStamp submitted, observed, ready;
    uint64_t deadline_ns;
    FemuReadObservation io;
} SparfReadTrace;

typedef struct SparfAutonomousTrace {
    uint64_t request;
    int engine;
    unsigned int engines, compute_units;
    bool read_diagnostics;
    const char *group_write_order;
    SparfTraceStamp admitted, started, ready;
    SparfReadTrace reads[3];
    SparfComputeTrace computes[3];
} SparfAutonomousTrace;

#endif
