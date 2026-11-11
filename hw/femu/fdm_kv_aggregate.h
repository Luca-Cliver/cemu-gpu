#ifndef CEMU_FDM_KV_AGGREGATE_H
#define CEMU_FDM_KV_AGGREGATE_H

#include "fdm_kv_buffer_abi.h"

#define CEMU_FDM_KV_AGGREGATE_COMMAND UINT64_C(0x46444d4b56414731)

typedef enum FDMKV_AggregateResult {
    FDM_KV_AGGREGATE_OK,
    FDM_KV_AGGREGATE_INVALID,
    FDM_KV_AGGREGATE_BUSY,
    FDM_KV_AGGREGATE_NO_SPACE,
    FDM_KV_AGGREGATE_WAIT_TIME,
    FDM_KV_AGGREGATE_PROGRESS,
} FDMKV_AggregateResult;

struct cemu_args;
long long fdm_kv_aggregate(struct cemu_args *args);

/* Optional native-module profiling ABI; not part of the FDM buffer/wire ABI.
 * All intervals use the caller's Host monotonic clock, including timer overhead.
 * Keep stage order stable; incompatible changes require a new symbol/version. */
#define FDM_KV_AGGREGATE_PROFILE_STAGES(X) \
    X(validate) X(preflight) X(free_list) X(token_k) X(channel_k) X(token_v) X(commit)
typedef enum FDMKV_AggregateStage {
#define FDM_KV_STAGE_ENUM(name) FDM_KV_AGG_STAGE_##name,
    FDM_KV_AGGREGATE_PROFILE_STAGES(FDM_KV_STAGE_ENUM)
#undef FDM_KV_STAGE_ENUM
    FDM_KV_AGG_STAGE_COUNT
} FDMKV_AggregateStage;
typedef struct FDMKV_AggregateProfile {
    uint64_t host_ns[FDM_KV_AGG_STAGE_COUNT];
    uint64_t calls, successful_calls, new_pages, copied_bytes, channel_elements;
} FDMKV_AggregateProfile;
typedef uint64_t (*FDMKV_ProfileClock)(void);
typedef long long (*FDMKV_ProfileFunction)(void *, FDMKV_AggregateProfile *, FDMKV_ProfileClock);
long long fdm_kv_aggregate_profile_v1(void *args, FDMKV_AggregateProfile *profile,
                                    FDMKV_ProfileClock now_ns);

#define CEMU_FDM_KV_SERVICE_VERSION 1U

typedef enum FDMKV_ServiceResult {
    FDM_KV_SERVICE_STOP,
    FDM_KV_SERVICE_IDLE,
    FDM_KV_SERVICE_PROGRESS,
} FDMKV_ServiceResult;

typedef struct FDMKV_ServiceRuntime {
    uint32_t version;
    uint32_t bytes;
    void *opaque;
    FDMKV_ServiceResult (*step)(void *opaque, FDMKV_ProfileFunction aggregate);
    void (*relax)(void *opaque, FDMKV_ServiceResult result);
} FDMKV_ServiceRuntime;

typedef int (*FDMKV_ServiceFunction)(const FDMKV_ServiceRuntime *runtime);
int fdm_kv_aggregate_service_v1(const FDMKV_ServiceRuntime *runtime);

#endif
