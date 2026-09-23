#ifndef CEMU_INSTATTENTION_ATTENTION_H
#define CEMU_INSTATTENTION_ATTENTION_H

#include "instattention_buffer.h"
#include "hw/femu/sparf_storage.h"

typedef struct NativeKvAttention {
    NativeKvBufferLease lease;
    struct cemu_sparf_workflow_header header;
} NativeKvAttention;

CemuKvStatus native_kv_attention_open(NativeKvBuffer *buffer,
    const CemuKvIdentity *identity, uint32_t batch_first,
    const struct cemu_sparf_workflow_header *header, NativeKvAttention *attention);
bool native_kv_attention_read(void *opaque,
    const struct cemu_sparf_workflow_header *header, const uint32_t *indices,
    uint32_t kind, void *output, size_t bytes, uint64_t start_ns,
    CemuSparfStorageResult *result);
void native_kv_attention_close(NativeKvAttention *attention);

#endif
