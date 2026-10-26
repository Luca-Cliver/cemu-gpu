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
} FDMKV_AggregateResult;

struct cemu_args;
long long fdm_kv_aggregate(struct cemu_args *args);

#endif
