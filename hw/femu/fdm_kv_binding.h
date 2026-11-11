#ifndef CEMU_FDM_KV_BINDING_H
#define CEMU_FDM_KV_BINDING_H

#include "nvme.h"
#include "csd/compute.h"
#include "fdm_kv_buffer.h"

typedef struct FDMKV_Writer FDMKV_Writer_t;

typedef struct FDMKV_SlotTask {
    NvmeRequest *pending;
    NvmeRequest task;
    uint16_t qid;
    bool background;
} FDMKV_SlotTask_t;

typedef struct FDMKV_Binding {
    NvmeNamespace *nvm_ns;
    NvmeNamespace *fdm_ns;
    MemoryRangeSet *mrs;
    FDMKV_Buffer_t *buffer;
    Program *aggregate_program;
    FDMKV_ProfileFunction aggregate_profile_fn;
    FDMKV_ServiceFunction aggregate_service_fn;
    FDMKV_Writer_t *writer;
    NvmeRequest *pending_write;
    NvmeRequest *aggregate_request;
    FDMKV_SlotTask_t *slot_tasks;
    GThread *aggregate_worker;
    GCond aggregate_wake;
    GList *waiting_attention;
    bool async_aggregate;
    bool aggregate_active;
    bool aggregate_stop;
    uint16_t aggregate_qid;
    uint64_t input_sb;
    uint64_t input_bytes;
    uint64_t aggregate_calls;
    uint64_t aggregate_host_ns;
} FDMKV_Binding_t;

uint16_t fdm_kv_bind(NvmeNamespace *compute_ns, MemoryRangeSet *mrs,
                     const FDMKV_BindSlotsParams_t *wire, uint32_t bytes, Program *program);
bool fdm_kv_unbind(MemoryRangeSet *mrs);
void fdm_kv_bindings_clear(FemuCtrl *ctrl);
bool fdm_kv_complete_write(NvmeRequest *request);
bool fdm_kv_input_busy(FemuCtrl *ctrl, const NvmeCmd *command);
void fdm_kv_async_notify_locked(FDMKV_Binding_t *binding);
void fdm_kv_aggregators_stop(FemuCtrl *ctrl);
bool fdm_kv_range_overlaps(const FDMKV_Binding_t *binding,
                           uint64_t offset, uint64_t bytes);
uint16_t fdm_kv_write_check(const FDMKV_Binding_t *binding,
                            uint64_t offset, uint64_t bytes, bool *input);

#endif
