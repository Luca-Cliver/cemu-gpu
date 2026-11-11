#include "hw/femu/backend/backend.h"
#include "hw/femu/inc/rte_ring.h"
#include "hw/femu/nvme-def.h"
#include "hw/femu/nvme.h"
#include "memory.h"
#include "hw/femu/fdm_kv_binding.h"
#include "hw/femu/fdm_kv_receive.h"

static inline uint16_t check_param(NvmeNamespace *ns, uint64_t sb, uint32_t len, const char *func)
{
    if (sb % 4 || len % 4) {
        femu_err("%s: sb %lu, len %u not dword-aligned!\n", func, sb, len);
        return NVME_INVALID_FIELD;
    }
    if (sb > ns->backend->size || len > ns->backend->size - sb) {
        femu_err("%s: sb (%lu) + fl (%u) > backend size (%lu)!\n", func, sb, len, ns->backend->size);
        return NVME_INVALID_FIELD;
    }
    return NVME_SUCCESS;
}

static bool memory_access_lock(MemoryNamespace *memory, uint64_t offset,
                               uint64_t bytes)
{
    g_rw_lock_reader_lock(&memory->fdm_kv_binding_lock);
    bool protected = fdm_kv_range_overlaps(memory->fdm_kv_binding, offset,
                                         bytes ? bytes : 1);
    if (protected) {
        g_mutex_lock(&memory->fdm_kv_lock);
    }
    return protected;
}

static void memory_access_unlock(MemoryNamespace *memory, bool protected)
{
    if (protected) {
        g_mutex_unlock(&memory->fdm_kv_lock);
    }
    g_rw_lock_reader_unlock(&memory->fdm_kv_binding_lock);
}

static uint16_t memory_read(NvmeNamespace *ns, NvmeCmd *cmd, NvmeRequest *req)
{
    FDM_IO_STAMP(req, memory_enter);
    FemuCtrl *n = ns->ctrl;
    NvmeMemoryReadCmd *read = (NvmeMemoryReadCmd *)cmd;
    uint64_t prp1 = le64_to_cpu(read->prp1);
    uint64_t prp2 = le64_to_cpu(read->prp2);
    uint64_t sb = le64_to_cpu(read->sb);
    uint32_t rl = le32_to_cpu(read->rl);

    femu_debug("memory_read: sb %zu, rl %u, prp1 %zu, prp2 %zu\n", sb, rl, prp1, prp2);

    uint16_t ret = check_param(ns, sb, rl, "memory_read");
    if (ret) {
        return ret;
    }

    MemoryNamespace *memory = ns->private;
    bool protected = memory_access_lock(memory, sb, rl);
    FDM_IO_STAMP(req, locked);
    if (backend_cuda_prepare_host(ns->backend,
                                  backend_addr(ns->backend, sb), rl) != 0) {
        memory_access_unlock(memory, protected);
        return NVME_DNR;
    }
    FDM_IO_STAMP(req, dma_begin);
    if (dma_read_prp(n, backend_addr(ns->backend, sb), rl, prp1, prp2)) {
        femu_err("memory_read: dma_read_prp error\n");
        memory_access_unlock(memory, protected);
        return NVME_DNR;
    }
    FDM_IO_STAMP(req, dma_end);
    memory_access_unlock(memory, protected);
    req->stat.pcie_lat = 0;
    req->stat.data_size = rl;
    req->stat.reqlat += rl / 2;
    req->stat.expire_time += rl / 2;
    if (req->fdm_io_trace.enabled) {
        req->fdm_io_trace.model_ready = req->stat.expire_time;
    }
    FDM_IO_STAMP(req, memory_done);
    req->cqe.n.result = rl;
    femu_debug("memory_read lat: %lu, cmd time: %u\n", clock_ns() - req->stat.stime, req->stat.reqlat);

    return NVME_SUCCESS;
}

static uint16_t memory_write(NvmeNamespace *ns, NvmeCmd *cmd, NvmeRequest *req)
{
    FDM_IO_STAMP(req, memory_enter);
    FemuCtrl *n = ns->ctrl;
    NvmeMemoryWriteCmd *write = (NvmeMemoryWriteCmd *)cmd;
    uint64_t prp1 = le64_to_cpu(write->prp1);
    uint64_t prp2 = le64_to_cpu(write->prp2);
    uint64_t sb = le64_to_cpu(write->sb);
    uint32_t wl = le32_to_cpu(write->wl);

    femu_debug("memory_write: sb %zu, wl %u\n", sb, wl);

    uint16_t ret = check_param(ns, sb, wl, "memory_write");
    if (ret) {
        return ret;
    }

    MemoryNamespace *memory = ns->private;
    bool protected = memory_access_lock(memory, sb, wl);
    FDM_IO_STAMP(req, locked);
    FDMKV_Binding_t *binding = memory->fdm_kv_binding;
    bool input = false;
    uint32_t slot_index = 0;
    uint64_t transfer_id = le32_to_cpu(write->rsvd13[0]) |
                           ((uint64_t)le32_to_cpu(write->rsvd13[1]) << 32);
    uint64_t packet_bytes = le32_to_cpu(write->rsvd13[2]);
    uint8_t *destination = backend_addr(ns->backend, sb);
    ret = fdm_kv_write_check(binding, sb, wl, &input);
    if (ret) {
        goto out;
    }
    if (input && binding->slot_tasks) {
        FDMKV_ReceiveResult acquired = fdm_kv_slot_acquire(binding->buffer,
            transfer_id, packet_bytes, sb - binding->input_sb, wl, &slot_index, &destination);
        if (acquired != FDM_KV_RECEIVE_ACCEPTED) {
            ret = (acquired == FDM_KV_RECEIVE_BUSY ? NVME_CMD_SEQ_ERROR : NVME_INVALID_FIELD) | NVME_DNR;
            goto out;
        }
    } else if (transfer_id || packet_bytes) {
        ret = NVME_INVALID_FIELD | NVME_DNR;
        goto out;
    }
    if (input && req->fdm_io_trace.enabled) {
        req->fdm_io_trace.session = binding->buffer->identity.session_id;
        req->fdm_io_trace.packet_id = binding->slot_tasks ?
            binding->buffer->slots[slot_index].receive->next_request_id :
            binding->buffer->receive->next_request_id;
    }
    FDM_IO_STAMP(req, dma_begin);
    int dma_status = dma_write_prp(n, destination, wl, prp1, prp2);
    FDM_IO_STAMP(req, dma_end);
    if (dma_status) {
        femu_err("memory_write: dma_read_prp error\n");
        if (input) {
            if (binding->slot_tasks) {
                fdm_kv_slot_fail(binding->buffer, slot_index);
            } else {
                fdm_kv_receive_fail(binding->buffer);
            }
            backend_cuda_mark_host_dirty(ns->backend, destination, wl);
        }
        ret = input ? NVME_DATA_TRAS_ERROR | NVME_DNR : NVME_DNR;
        goto out;
    }

    backend_cuda_mark_host_dirty(ns->backend,
                                 destination, wl);

    req->stat.pcie_lat = 0;
    req->stat.data_size = wl;
    req->stat.reqlat += wl / 2;
    req->stat.expire_time += wl / 2;
    req->cqe.n.result = wl;
    if (input) {
        FDMKV_ReceiveResult result = binding->slot_tasks ?
            fdm_kv_slot_record(binding->buffer, slot_index, sb - binding->input_sb,
                               wl, req->stat.expire_time) :
            fdm_kv_receive_record(binding->buffer, sb - binding->input_sb, wl, req->stat.expire_time);
        FDMKV_ReceiveState_t *receive = binding->slot_tasks ?
            binding->buffer->slots[slot_index].receive : binding->buffer->receive;
        if (result == FDM_KV_RECEIVE_READY) {
            if (req->fdm_io_trace.enabled) {
                req->fdm_io_trace.operation = receive->header.operation;
                req->fdm_io_trace.packet_id = receive->header.request_id;
                req->fdm_io_trace.token_first = receive->header.token_first;
            }
            if (binding->aggregate_program) {
                if (binding->slot_tasks) {
                    binding->slot_tasks[slot_index].pending = req;
                } else {
                    binding->pending_write = req;
                }
                req->fdm_kv_pending = true;
                req->stat.expire_time = MAX(req->stat.expire_time,
                    receive->max_write_ready_ns);
            } else {
                femu_err("FDM KV receive-only binding cannot aggregate; write not acknowledged\n");
                ret = NVME_INVALID_OPCODE | NVME_DNR;
            }
        } else if (result != FDM_KV_RECEIVE_ACCEPTED) {
            ret = NVME_INVALID_FIELD | NVME_DNR;
        }
    }
out:
    if (req->fdm_io_trace.enabled) {
        req->fdm_io_trace.input = input;
        req->fdm_io_trace.model_ready = req->stat.expire_time;
    }
    if (input) {
        backend_cuda_mark_host_dirty(ns->backend, binding->buffer, binding->buffer->meta_bytes);
        if (binding->buffer->failed) {
            g_cond_broadcast(&binding->aggregate_wake);
        }
    }
    if (ret) {
        req->cqe.n.result = 0;
    }
    memory_access_unlock(memory, protected);
    FDM_IO_STAMP(req, memory_done);
    return ret;
}

static uint16_t memory_fill(NvmeNamespace *ns, NvmeCmd *cmd, NvmeRequest *req)
{
    NvmeMemoryFillCmd *fill = (NvmeMemoryFillCmd *)cmd;
    uint64_t sb = le64_to_cpu(fill->sb);
    uint32_t fl = le32_to_cpu(fill->fl);

    femu_debug("memory_fill\n");

    uint16_t ret = check_param(ns, sb, fl, "memory_fill");
    if (ret) {
        return ret;
    }

    MemoryNamespace *memory = ns->private;
    bool protected = memory_access_lock(memory, sb, fl);
    if (fdm_kv_range_overlaps(memory->fdm_kv_binding, sb, fl)) {
        memory_access_unlock(memory, protected);
        return NVME_INVALID_FIELD | NVME_DNR;
    }
    backend_fill(ns->backend, sb, fl);
    memory_access_unlock(memory, protected);
    req->stat.pcie_lat = 0;
    req->stat.data_size = fl;
    req->stat.reqlat += fl / 2;
    req->stat.expire_time += fl / 2;
    req->stat.opcode = (cmd->nsid << 4) + (cmd->opcode & 0x0f);
    req->cqe.n.result = fl;
    return NVME_SUCCESS;
}

// called in bbssd/ftl.c

static void *memory_region_backend_addr(uint64_t sdaddr, uint64_t len, FemuCtrl **n, uint16_t *ret)
{
    for (int i = 0; i < nr_csd_memory_region; i++) {
        if (sdaddr >= csd_memory_region[i].dma_addr &&
            sdaddr < csd_memory_region[i].dma_addr + csd_memory_region[i].size) {
            if (sdaddr + len > csd_memory_region[i].dma_addr + csd_memory_region[i].size) {
                *ret = NVME_INVALID_FIELD;
                return NULL;
            }
            *n = csd_memory_region[i].n;
            *ret = 0;
            return (void *)(csd_memory_region[i].backend_addr + sdaddr - csd_memory_region[i].dma_addr);
        }
    }
    *ret = NVME_INVALID_FIELD;
    return NULL;
}

/*
 * copy format 2: nvm -> fdm
 * copy format 3: fdm -> nvm
 * copy format 4: fdm -> fdm
 */
static uint16_t memory_copy(NvmeNamespace *ns, NvmeCmd *cmd, NvmeRequest *req)
{
    FemuCtrl *n = ns->ctrl;
    NvmeMemoryCopyCmd *copy = (NvmeMemoryCopyCmd *)cmd;
    uint64_t len = le64_to_cpu(copy->len);
    uint64_t prp1 = le64_to_cpu(copy->prp1);
    uint64_t prp2 = le64_to_cpu(copy->prp2);
    uint64_t sdaddr = le64_to_cpu(copy->sdaddr);
    NvmeCopyFormat cf[128];
    NvmeCopyFormat *sres = cf;
    uint16_t ret;
    void *data = memory_region_backend_addr(sdaddr, len, &req->mem_ctrl, &ret);

    if (ret) {
        return ret;
    }

    if (req->mem_ctrl != n) {
        femu_debug("memory copy: source and destination belong to different controllers!\n");
    }

    // cdft == 4: fdm -> fdm
    if (copy->cdft < 2 || copy->cdft > 4) {
        femu_err("memory copy: copy descriptor format %u not supported!\n", copy->cdft);
        return NVME_INVALID_FIELD;
    }

    if (copy->cdft == 2 || copy->cdft == 3) {
        // nvm<->fdm
        req->sres = malloc(sizeof(NvmeCopyFormat) * copy->nr);
        req->nr_sres = copy->nr;
        req->is_write = copy->cdft == 3;
        sres = req->sres;
        req->sdaddr = data;
        req->sdaddr_backend = nvme_find_namespace(req->mem_ctrl, 2)->backend;
    }

    // dma read sre
    if (dma_write_prp(n, (void*)sres, sizeof(NvmeCopyFormat) * copy->nr, prp1, prp2)) {
        femu_err("memory copy: dma_write_prp error\n");
        return NVME_DNR;
    }

    if (copy->cdft == 2 || copy->cdft == 3) {
        // nvm<->fdm, process in ftl thread
        femu_debug("memory copy: len %lu, nlb %lu, saddr %lu\n", len, len / 512, sdaddr);
        int rc = femu_ring_enqueue(n->to_ftl[req->poller], (void *)&req, 1);
        if (rc != 1) {
            femu_err("memory_copy enqueue failed, ret=%d\n", rc);
        }
        return 0;
    }

    uint64_t copyed = 0;
    for (int i = 0; i < copy->nr; i++) {
        NvmeNamespace *sns = NULL;
        NvmeCopyFormat4 *sre = &sres[i].cf4;
        uint64_t nbyte = le32_to_cpu(sre->nbyte);
        uint64_t off = le64_to_cpu(sre->saddr);
        uint32_t snsid = le32_to_cpu(sre->snsid);
        if (snsid != 2) {
            femu_err("memory_copy: fdm->fdm snsid != 2!\n");
            return NVME_INVALID_NSID;
        }
        sns = nvme_find_namespace(n, snsid);
        femu_debug("memory_copy fdm->fdm: soff %lu, doff %lu, nbyte %lu, stime %zu\n", off, sdaddr, nbyte, req->stat.stime);
        if (backend_rw_internal(sns->backend, data, off, nbyte, 1)) {
            req->status = NVME_INTERNAL_DEV_ERROR | NVME_DNR;
            return 1;
        }
        data += nbyte;
        copyed += nbyte;
    }

    if (copyed != len) {
        femu_err("copyed %lu, len %lu\n", copyed, len);
        abort();
    }

    // standard hasn't defined the result. use copyed bytes as result
    req->cqe.n.result = copyed;
    req->stat.opcode = 0x21;
    req->stat.pcie_lat = 1;
    req->stat.data_size = copyed;
    req->stat.reqlat += copyed / 4;
    req->stat.expire_time += copyed / 4;
    req->status = NVME_SUCCESS;
    return 1;
}

static int memory_io_cmd(NvmeNamespace *ns, NvmeCmd *cmd, NvmeRequest *req)
{
    switch (cmd->opcode) {
    case NVME_CMD_MEMORY_FILL:
        req->is_write = 1;
        req->status = memory_fill(ns, cmd, req);
        break;
    case NVME_CMD_MEMORY_COPY:
        if (ns->ctrl->native_kv) {
            req->status = NVME_INVALID_OPCODE;
            return 1;
        }
        req->is_write = 1;
        return memory_copy(ns, cmd, req);
    case NVME_CMD_MEMORY_READ:
        req->is_write = 0;
        req->status = memory_read(ns, cmd, req);
        break;
    case NVME_CMD_MEMORY_WRITE:
        req->is_write = 1;
        req->status = memory_write(ns, cmd, req);
        break;
    default:
        femu_err("memory_io_cmd: opcode %u not supported!\n", cmd->opcode);
        req->status = NVME_INVALID_OPCODE;
        return 1;
    }
    // TODO: io command should be async
    return 1;
}

static void memory_init(NvmeNamespace *ns, Error **err)
{
    MemoryNamespace *mns = g_new0(MemoryNamespace, 1);
    g_rw_lock_init(&mns->fdm_kv_binding_lock);
    g_mutex_init(&mns->fdm_kv_lock);
    ns->private = mns;
    // nvme_init_slm(n, ns);
    if (!ns->ctrl->native_kv)
        init_backend(&ns->backend, FEMU_DRAM_BACKEND, NULL, ns->size);
}

static void memory_exit(NvmeNamespace *ns)
{
    MemoryNamespace *memory = ns->private;
    if (memory) {
        g_mutex_clear(&memory->fdm_kv_lock);
        g_rw_lock_clear(&memory->fdm_kv_binding_lock);
        g_free(memory);
        ns->private = NULL;
    }
}

CommandSetOps memory_command_set_ops = {
    .state            = NULL,
    .init             = memory_init,
    .exit             = memory_exit,
    .rw_check_req     = NULL,
    .admin_cmd        = NULL,
    .io_cmd           = memory_io_cmd,
    .get_log          = NULL,
};

nvme_register(NVME_CSI_MEMORY, &memory_command_set_ops);
