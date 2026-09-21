/* Optional host-monotonic observations, separate from modeled NAND time.
 * The including translation unit provides monotonic_ns(). */
#ifndef FEMU_SPARF_TIMELINE_H
#define FEMU_SPARF_TIMELINE_H
#include "hw/femu/read_observation.h"

enum SparfObservedPhase {
    SPARF_OBS_CHANNEL_READ,
    SPARF_OBS_TOKEN_K_READ,
    SPARF_OBS_V_OUTSTANDING,
    SPARF_OBS_QK,
    SPARF_OBS_PV,
    SPARF_OBS_COUNT,
};

typedef struct SparfObservation {
    uint64_t start[SPARF_OBS_COUNT];
    uint64_t end[SPARF_OBS_COUNT];
    uint64_t worker_start, command_start, command_end, worker_end;
    uint64_t virtual_start, virtual_end;
    FemuReadObservation reads[3];
} SparfObservation;

static inline void sparf_observe_begin(SparfObservation *observation, int phase)
{
    if (observation) observation->start[phase] = monotonic_ns();
}

static inline void sparf_observe_end(SparfObservation *observation, int phase)
{
    if (observation) observation->end[phase] = monotonic_ns();
}

/* These are functional workers, not additional modeled compute units. */
static inline int sparf_stage_worker(uint32_t stage, int workers, int fallback)
{
    if (workers < 2 || stage < CEMU_SPARF_STAGE_SELECT || stage > CEMU_SPARF_STAGE_PV)
        return fallback;
    return stage >= CEMU_SPARF_STAGE_EXACT_V ? 1 : 0;
}

#endif
