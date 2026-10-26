#ifndef CEMU_FDM_KV_BINDING_H
#define CEMU_FDM_KV_BINDING_H

#include "nvme.h"
#include "csd/compute.h"
#include "fdm_kv_buffer.h"

typedef struct FDMKV_Writer FDMKV_Writer_t;

typedef struct FDMKV_Binding {
    NvmeNamespace *nvm_ns;
    NvmeNamespace *fdm_ns;
    MemoryRangeSet *mrs;
    FDMKV_Buffer_t *buffer;
    Program *aggregate_program;
    FDMKV_Writer_t *writer;
    NvmeRequest *pending_write;
    uint64_t input_sb;
    uint64_t input_bytes;
    uint64_t aggregate_calls;
    uint64_t aggregate_host_ns;
} FDMKV_Binding_t;

uint16_t fdm_kv_bind(NvmeNamespace *compute_ns, MemoryRangeSet *mrs,
                     const FDMKV_BindParams_t *wire, Program *program);
bool fdm_kv_unbind(MemoryRangeSet *mrs);
void fdm_kv_bindings_clear(FemuCtrl *ctrl);
bool fdm_kv_complete_write(NvmeRequest *request);
bool fdm_kv_range_overlaps(const FDMKV_Binding_t *binding,
                           uint64_t offset, uint64_t bytes);
uint16_t fdm_kv_write_check(const FDMKV_Binding_t *binding,
                            uint64_t offset, uint64_t bytes, bool *input);

#endif
