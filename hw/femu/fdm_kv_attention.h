#ifndef CEMU_FDM_KV_ATTENTION_H
#define CEMU_FDM_KV_ATTENTION_H

#include "fdm_kv_binding.h"
#include "bbssd/instattention_address.h"
#include "hw/femu/sparf_storage.h"

typedef struct FDMKV_Attention {
    FDMKV_Binding_t *binding;
    CemuKvIdentity identity;
    CemuKvLayout layout;
    uint32_t batch_first;
    bool ready;
    struct cemu_sparf_batch_row *rows;
    struct cemu_sparf_workflow_header header;
} FDMKV_Attention_t;

bool fdm_kv_attention_open(FemuCtrl *ctrl,
    const struct cemu_sparf_fdm_binding *wire,
    const struct cemu_sparf_workflow_header *header,
    const struct cemu_sparf_batch_row *rows, FDMKV_Attention_t *attention);
bool fdm_kv_attention_read(void *opaque,
    const struct cemu_sparf_workflow_header *header, const uint32_t *indices,
    uint32_t kind, void *output, size_t bytes, uint64_t start_ns,
    CemuSparfStorageResult *result, FemuReadObservation *observation);
void fdm_kv_attention_close(void *opaque);
bool fdm_kv_attention_wait(FDMKV_Attention_t *attention);
bool fdm_kv_attention_publish_locked(FDMKV_Binding_t *binding);

#endif
