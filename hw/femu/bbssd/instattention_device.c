#include "qemu/osdep.h"
#include "bb.h"

static NvmNamespace *native_owner(NvmeNamespace *ns)
{
    NvmNamespace *nns;

    if (!ns || ns->csi != NVME_CSI_NVM || !ns->private) {
        return NULL;
    }
    nns = nvm_ns(ns);
    return nns->namespace == ns && nns->native_io_used ? nns : NULL;
}

CemuKvStatus bb_native_kv_register(NvmeNamespace *ns,
                                  const CemuKvIdentity *identity,
                                  const CemuKvGeometry *geometry)
{
    NvmNamespace *nns = native_owner(ns);
    CemuKvStatus status;

    if (!nns) {
        return CEMU_KV_INVALID;
    }
    g_mutex_lock(&nns->native_io_lock);
    status = cemu_kv_io_register(nns->native_io, identity, geometry);
    g_mutex_unlock(&nns->native_io_lock);
    return status;
}

static CemuKvStatus native_transfer_locked(NvmNamespace *nns,
                                           const CemuKvIdentity *identity,
                                           const CemuKvCoordinate *coordinate,
                                           uint32_t valid_tokens, void *buffer,
                                           size_t bytes, bool write,
                                           uint64_t start_ns,
                                           CemuKvIoResult *result)
{
    CemuKvIoResult candidate = { .start_ns = start_ns, .ready_ns = start_ns };
    CemuKvStatus status;

    if (write) {
        status = cemu_kv_io_write(nns->native_io, identity, coordinate,
                                 valid_tokens, buffer, bytes, start_ns, &candidate);
    } else {
        status = cemu_kv_io_read(nns->native_io, identity, coordinate,
                                buffer, bytes, start_ns, &candidate);
    }
    if (status == CEMU_KV_IO_FAILED) {
        for (int channel = 0; channel < nns->params->nchs; channel++) {
            for (int lun = 0; lun < nns->params->luns_per_ch; lun++) {
                candidate.ready_ns = MAX(candidate.ready_ns,
                    nns->ch[channel].lun[lun].next_lun_avail_time);
            }
        }
    }
    if (status == CEMU_KV_OK || status == CEMU_KV_IO_FAILED) {
        nns->native_deadline = MAX(nns->native_deadline, candidate.ready_ns);
    }
    *result = candidate;
    return status;
}

CemuKvStatus bb_native_kv_transfer(NvmeNamespace *ns,
                                  const CemuKvIdentity *identity,
                                  const CemuKvCoordinate *coordinate,
                                  uint32_t valid_tokens, void *buffer,
                                  size_t bytes, bool write, uint64_t start_ns,
                                  CemuKvIoResult *result)
{
    NvmNamespace *nns = native_owner(ns);
    CemuKvStatus status;

    if (!nns || !result || start_ns > INT64_MAX || (!write && valid_tokens)) {
        return CEMU_KV_INVALID;
    }
    g_mutex_lock(&nns->native_io_lock);
    status = native_transfer_locked(nns, identity, coordinate, valid_tokens,
                                    buffer, bytes, write, start_ns, result);
    g_mutex_unlock(&nns->native_io_lock);
    return status;
}

CemuKvStatus bb_native_kv_batch(NvmeNamespace *ns,
                               const CemuKvIdentity *identity,
                               CemuKvDevicePage *pages, uint32_t count,
                               bool write, uint64_t start_ns,
                               CemuKvDeviceBatchResult *result)
{
    NvmNamespace *nns = native_owner(ns);
    CemuKvDeviceBatchResult candidate = { .ready_ns = start_ns };
    CemuKvStatus status = CEMU_KV_OK;

    if (!nns || !identity || !pages || !result || !count ||
        count > CEMU_NATIVE_INTERNAL_MAX_PAGES || start_ns > INT64_MAX) {
        return CEMU_KV_INVALID;
    }
    for (uint32_t index = 0; index < count; index++) {
        if (!pages[index].buffer || (!write && pages[index].valid_tokens)) {
            return CEMU_KV_INVALID;
        }
    }
    g_mutex_lock(&nns->native_io_lock);
    for (uint32_t index = 0; index < count; index++) {
        CemuKvDevicePage *page = &pages[index];

        status = native_transfer_locked(nns, identity, &page->coordinate,
            page->valid_tokens, page->buffer, page->bytes, write, start_ns, &page->result);
        candidate.ready_ns = MAX(candidate.ready_ns, page->result.ready_ns);
        if (status != CEMU_KV_OK) {
            break;
        }
        candidate.completed_pages++;
        candidate.transferred_bytes += page->result.transferred_bytes;
    }
    *result = candidate;
    g_mutex_unlock(&nns->native_io_lock);
    return status;
}

CemuKvStatus bb_native_kv_snapshot(NvmeNamespace *ns, CemuKvMapStats *stats,
                                  uint64_t *deadline)
{
    NvmNamespace *nns = native_owner(ns);
    CemuKvStatus status = CEMU_KV_INVALID;

    if (!nns || !stats || !deadline) {
        return CEMU_KV_INVALID;
    }
    g_mutex_lock(&nns->native_io_lock);
    if (nns->native_io) {
        *stats = cemu_kv_io_stats(nns->native_io);
        *deadline = nns->native_deadline;
        status = CEMU_KV_OK;
    }
    g_mutex_unlock(&nns->native_io_lock);
    return status;
}
