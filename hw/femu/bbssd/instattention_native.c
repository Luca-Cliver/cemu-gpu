#include "qemu/osdep.h"
#include "bb.h"
#include "hw/femu/native_kv.h"

static uint16_t native_status(CemuKvStatus status)
{
    switch (status) {
    case CEMU_KV_OK:
        return NVME_SUCCESS;
    case CEMU_KV_PENDING:
        return NVME_CMD_SEQ_ERROR;
    case CEMU_KV_IO_FAILED:
        return NVME_INTERNAL_DEV_ERROR | NVME_DNR;
    default:
        return NVME_INVALID_FIELD | NVME_DNR;
    }
}

static void native_execute(NvmeNamespace *ns, NvmeRequest *req)
{
    NvmNamespace *nns = nvm_ns(ns);
    FemuCtrl *ctrl = ns->ctrl;
    NvmeCmd *cmd = &req->cmd;
    uint32_t fields[6] = {
        le32_to_cpu(cmd->cdw10), le32_to_cpu(cmd->cdw11), le32_to_cpu(cmd->cdw12),
        le32_to_cpu(cmd->cdw13), le32_to_cpu(cmd->cdw14), le32_to_cpu(cmd->cdw15),
    };
    CemuKvIdentity identity = {
        .controller = ctrl->drive_id, .namespace_id = ns->id,
        .task = fields[1], .generation = fields[2],
    };
    CemuKvCoordinate coordinate = {
        .batch = fields[3] & 0xffff, .layer = fields[3] >> 16,
        .head = fields[4] & 0xffff, .kind = (fields[4] >> 16) & 0xff,
        .channel = fields[4] >> 24, .token = fields[5],
    };
    CemuKvIoResult result;
    CemuKvStatus status = CEMU_KV_INVALID;
    uint8_t *buffer = nns->native_buffer;
    uint64_t start = req->stat.stime;
    uint64_t deadline = start;
    bool write = cmd->opcode == CEMU_NATIVE_KV_REGISTER || cmd->opcode == CEMU_NATIVE_KV_WRITE ||
                 cmd->opcode == CEMU_NATIVE_KV_BUFFER_REGISTER || cmd->opcode == CEMU_NATIVE_KV_BUFFER_APPEND;
    bool data = cmd->opcode != CEMU_NATIVE_KV_BUFFER_DRAIN;

    req->status = NVME_INVALID_FIELD | NVME_DNR;
    req->cqe.n.result = cpu_to_le32(CEMU_KV_INVALID);
    req->is_write = write;
    if (cmd->opcode == NVME_CMD_FLUSH) {
        CemuKvMapStats stats;

        if (nns->native_tail) {
            status = native_kv_buffer_drain(nns->native_tail);
            if (status != CEMU_KV_OK) {
                req->cqe.n.result = cpu_to_le32(status);
                req->status = native_status(status);
                return;
            }
        }
        if (bb_native_kv_snapshot(ns, &stats, &deadline) != CEMU_KV_OK) {
            return;
        }
        req->status = NVME_SUCCESS;
        req->cqe.n.result = 0;
        req->stat.expire_time = MAX(start, deadline);
        req->stat.reqlat = req->stat.expire_time - start;
        return;
    }
    if (cmd->fuse || cmd->res1 || cmd->psdt || le64_to_cpu(cmd->res2) ||
        le64_to_cpu(cmd->mptr) ||
        (fields[0] & 0xffff) != CEMU_NATIVE_KV_VERSION ||
        (cmd->opcode != CEMU_NATIVE_KV_WRITE && cmd->opcode != CEMU_NATIVE_KV_BUFFER_APPEND &&
         fields[0] >> 16)) {
        return;
    }
    if (write) {
        req->status = dma_write_prp(ctrl, buffer, CEMU_NATIVE_KV_BYTES,
                                   le64_to_cpu(cmd->dptr.prp1), le64_to_cpu(cmd->dptr.prp2));
        if (req->status) {
            return;
        }
    } else {
        memset(buffer, 0, CEMU_NATIVE_KV_BYTES);
    }
    switch (cmd->opcode) {
    case CEMU_NATIVE_KV_INFO: {
        CemuKvMapStats stats;
        uint64_t pending_deadline;

        if (fields[1] || fields[2] || fields[3] || fields[4] || fields[5]) {
            break;
        }
        status = bb_native_kv_snapshot(ns, &stats, &pending_deadline);
        if (status != CEMU_KV_OK) {
            break;
        }
        stl_le_p(buffer, CEMU_NATIVE_KV_MAGIC);
        stl_le_p(buffer + 4, CEMU_NATIVE_KV_VERSION);
        stl_le_p(buffer + 8, ctrl->drive_id);
        stl_le_p(buffer + 12, ns->id);
        stl_le_p(buffer + 16, CEMU_NATIVE_KV_BYTES);
        stl_le_p(buffer + 20, nns->params->nchs);
        stl_le_p(buffer + 24, nns->params->luns_per_ch);
        stq_le_p(buffer + 32, ns->backend->size);
        stq_le_p(buffer + 40, stats.reserved_pages);
        stq_le_p(buffer + 48, stats.written_pages);
        stq_le_p(buffer + 56, stats.failed_pages);
        stq_le_p(buffer + 64, stats.claimed_blocks);
        stq_le_p(buffer + 72, pending_deadline);
        stq_le_p(buffer + 80, ctrl->native_kv_buffer_budget);
        stq_le_p(buffer + 88, ctrl->native_kv_page_limit);
        stq_le_p(buffer + 96, ctrl->num_namespaces == 3);
        status = CEMU_KV_OK;
        break;
    }
    case CEMU_NATIVE_KV_REGISTER:
    case CEMU_NATIVE_KV_BUFFER_REGISTER: {
        CemuKvGeometry geometry = {
            .batches = ldl_le_p(buffer), .layers = ldl_le_p(buffer + 4),
            .heads = ldl_le_p(buffer + 8), .head_dim = ldl_le_p(buffer + 12),
            .max_tokens = ldl_le_p(buffer + 16),
            .channels_per_page = ldl_le_p(buffer + 20),
            .element_bytes = ldl_le_p(buffer + 24), .page_bytes = ldl_le_p(buffer + 28),
        };
        bool reserved = false;

        bool buffered = cmd->opcode == CEMU_NATIVE_KV_BUFFER_REGISTER;

        for (size_t index = buffered ? 44 : 32; index < CEMU_NATIVE_KV_BYTES; index++) {
            reserved |= buffer[index] != 0;
        }
        if (fields[3] || fields[4] || fields[5] || reserved ||
            geometry.batches > 65536 || geometry.layers > 65536 ||
            geometry.heads > 65536 || geometry.head_dim > 256) {
            break;
        }
        if (buffered) {
            uint64_t budget = ldq_le_p(buffer + 32);
            uint32_t batch_pages = ldl_le_p(buffer + 40);

            if (nns->native_tail) {
                status = CEMU_KV_EXISTS;
            } else if (budget > ctrl->native_kv_buffer_budget || geometry.page_bytes != CEMU_NATIVE_KV_BYTES) {
                status = CEMU_KV_LIMIT;
            } else {
                status = native_kv_buffer_create(ns, &identity, &geometry, budget,
                                                 batch_pages, &nns->native_tail);
            }
        } else {
            status = bb_native_kv_register(ns, &identity, &geometry);
        }
        break;
    }
    case CEMU_NATIVE_KV_WRITE:
        if (native_kv_buffer_matches(nns->native_tail, &identity)) {
            break;
        }
        status = bb_native_kv_transfer(ns, &identity, &coordinate,
                                       fields[0] >> 16, buffer, CEMU_NATIVE_KV_BYTES,
                                       true, start, &result);
        if (status == CEMU_KV_OK || status == CEMU_KV_IO_FAILED) {
            deadline = result.ready_ns;
        }
        break;
    case CEMU_NATIVE_KV_BUFFER_APPEND: {
        uint64_t bytes = (uint64_t)(fields[0] >> 16) * native_kv_buffer_row_bytes(nns->native_tail, false);

        if (!native_kv_buffer_matches(nns->native_tail, &identity) || bytes > CEMU_NATIVE_KV_BYTES / 2) {
            break;
        }
        status = native_kv_buffer_append(nns->native_tail, &coordinate, fields[0] >> 16,
                                         buffer, buffer + bytes, bytes);
        break;
    }
    case CEMU_NATIVE_KV_BUFFER_READ:
        if (!native_kv_buffer_matches(nns->native_tail, &identity)) {
            break;
        }
        status = native_kv_buffer_read(nns->native_tail, &coordinate, buffer,
            native_kv_buffer_row_bytes(nns->native_tail, coordinate.kind == CEMU_KV_PAGE_CHANNEL_K),
            start, &result);
        if (status == CEMU_KV_OK || status == CEMU_KV_IO_FAILED) {
            deadline = result.ready_ns;
        }
        break;
    case CEMU_NATIVE_KV_BUFFER_STATUS: {
        NativeKvBufferStats stats;

        if (!native_kv_buffer_matches(nns->native_tail, &identity) ||
            coordinate.token || coordinate.kind || coordinate.channel) {
            break;
        }
        status = native_kv_buffer_stats(nns->native_tail, &coordinate, &stats);
        if (status != CEMU_KV_OK) {
            break;
        }
        stq_le_p(buffer, stats.accepted);
        for (size_t index = 0; index < 3; index++) {
            stq_le_p(buffer + 8 + index * 8, stats.committed[index]);
        }
        stq_le_p(buffer + 32, stats.memory_bytes);
        stq_le_p(buffer + 40, stats.write_bytes);
        stq_le_p(buffer + 48, stats.read_bytes);
        stq_le_p(buffer + 56, stats.write_batches);
        stq_le_p(buffer + 64, stats.pending_ready_ns);
        stq_le_p(buffer + 72, stats.sealed);
        stq_le_p(buffer + 80, stats.failed);
        break;
    }
    case CEMU_NATIVE_KV_BUFFER_DRAIN:
        if (native_kv_buffer_matches(nns->native_tail, &identity) &&
            !fields[3] && !fields[4] && !fields[5]) {
            status = native_kv_buffer_drain(nns->native_tail);
        }
        break;
    case CEMU_NATIVE_KV_READ:
        if (native_kv_buffer_matches(nns->native_tail, &identity)) {
            break;
        }
        status = bb_native_kv_transfer(ns, &identity, &coordinate,
                                       0, buffer, CEMU_NATIVE_KV_BYTES,
                                       false, start, &result);
        if (status == CEMU_KV_OK || status == CEMU_KV_IO_FAILED) {
            deadline = result.ready_ns;
        }
        break;
    default:
        break;
    }
    req->cqe.n.result = cpu_to_le32(status);
    req->status = native_status(status);
    if (status == CEMU_KV_OK && !write && data) {
        req->status = dma_read_prp(ctrl, buffer, CEMU_NATIVE_KV_BYTES,
                                    le64_to_cpu(cmd->dptr.prp1), le64_to_cpu(cmd->dptr.prp2));
    }
    req->stat.expire_time = deadline;
    req->stat.rw.nand_lat = deadline - start;
    req->stat.reqlat = deadline - start;
    req->stat.data_size = status == CEMU_KV_OK && data ? CEMU_NATIVE_KV_BYTES : 0;
    req->stat.pcie_lat = status == CEMU_KV_OK && data ? 1 : 0;
}

static void *native_worker(void *opaque)
{
    NvmeNamespace *ns = opaque;
    NvmNamespace *nns = nvm_ns(ns);
    NvmeRequest *request;

    rcu_register_thread();
    while (!qatomic_read(&nns->native_stop)) {
        if (femu_ring_dequeue(nns->to_ftl[1], (void *)&request, 1) != 1) {
            g_usleep(10);
            continue;
        }
        rcu_read_lock();
        native_execute(ns, request);
        rcu_read_unlock();
        while (femu_ring_enqueue(ns->ctrl->to_poller[1], (void *)&request, 1) != 1) {
            if (qatomic_read(&nns->native_stop)) {
                goto stopped;
            }
            g_usleep(10);
        }
    }
stopped:
    rcu_unregister_thread();
    return NULL;
}

void bb_native_start(NvmeNamespace *ns)
{
    NvmNamespace *nns = nvm_ns(ns);

    assert(!nns->native_started);
    qatomic_set(&nns->native_stop, false);
    qemu_thread_create(&nns->ftl_thread, "native-kv-ftl", native_worker, ns, QEMU_THREAD_JOINABLE);
    nns->native_started = true;
}

void bb_native_stop(NvmeNamespace *ns)
{
    NvmNamespace *nns = nvm_ns(ns);
    NvmeRequest *request;

    if (!nns) {
        return;
    }
    if (nns->native_started) {
        qatomic_set(&nns->native_stop, true);
        qemu_thread_join(&nns->ftl_thread);
        nns->native_started = false;
    }
    native_kv_buffer_stop(nns->native_tail);
    if (nns->to_ftl && nns->to_ftl[1]) {
        while (femu_ring_dequeue(nns->to_ftl[1], (void *)&request, 1) == 1) {
        }
    }
}

void bb_native_namespace_exit(NvmeNamespace *ns)
{
    NvmNamespace *nns = nvm_ns(ns);

    if (!nns) {
        return;
    }
    bb_native_stop(ns);
    native_kv_buffer_free(nns->native_tail);
    bb_native_io_destroy(ns);
    if (nns->native_io_used) {
        g_mutex_clear(&nns->native_io_lock);
    }
    if (nns->ch) {
        for (int channel = 0; channel < nns->params->nchs; channel++) {
            g_free(nns->ch[channel].lun);
        }
    }
    g_free(nns->ch);
    g_free(nns->native_buffer);
    if (nns->to_ftl) {
        if (nns->to_ftl[1]) {
            femu_ring_free(nns->to_ftl[1]);
        }
        g_free(nns->to_ftl);
    }
    g_free(nns);
    ns->private = NULL;
    ns->ctrl->to_ftl = NULL;
}

void bb_native_namespace_init(NvmeNamespace *ns, Error **errp)
{
    NvmNamespace *nns = g_new0(NvmNamespace, 1);
    CemuKvStatus status;

    ns->private = nns;
    nns->params = ns->params;
    nns->namespace = ns;
    nns->ch = g_try_new0(struct ssd_channel, nns->params->nchs);
    if (!nns->ch) {
        goto fail;
    }
    for (int channel = 0; channel < nns->params->nchs; channel++) {
        nns->ch[channel].lun = g_try_new0(struct nand_lun, nns->params->luns_per_ch);
        if (!nns->ch[channel].lun) {
            goto fail;
        }
    }
    status = bb_native_io_create(ns, 64, MIN(ns->size / CEMU_NATIVE_KV_BYTES, ns->ctrl->native_kv_page_limit),
                                 MAX(64 * MiB, (uint64_t)ns->ctrl->native_kv_page_limit * 256), &nns->native_io);
    if (status != CEMU_KV_OK) {
        goto fail;
    }
    nns->native_buffer = g_try_malloc(CEMU_NATIVE_KV_BYTES);
    nns->to_ftl = g_new0(struct rte_ring *, 2);
    nns->to_ftl[1] = femu_ring_create(FEMU_RING_TYPE_MP_SC, FEMU_MAX_INF_REQS);
    if (!nns->native_buffer || !nns->to_ftl[1]) {
        goto fail;
    }
    ns->ctrl->to_ftl = nns->to_ftl;
    femu_log("CEMU FTL mode=instattention: native KV ABI=1, fresh namespace, no LBA IO\n");
    return;
fail:
    error_setg(errp, "Cannot initialize native KV namespace");
    bb_native_namespace_exit(ns);
}

int bb_native_submit(NvmeNamespace *ns, NvmeCmd *cmd, NvmeRequest *req)
{
    NvmNamespace *nns = nvm_ns(ns);

    switch (cmd->opcode) {
    case CEMU_NATIVE_KV_INFO:
    case CEMU_NATIVE_KV_REGISTER:
    case CEMU_NATIVE_KV_WRITE:
    case CEMU_NATIVE_KV_READ:
    case CEMU_NATIVE_KV_BUFFER_REGISTER:
    case CEMU_NATIVE_KV_BUFFER_APPEND:
    case CEMU_NATIVE_KV_BUFFER_READ:
    case CEMU_NATIVE_KV_BUFFER_STATUS:
    case CEMU_NATIVE_KV_BUFFER_DRAIN:
    case NVME_CMD_FLUSH:
        if (femu_ring_enqueue(nns->to_ftl[1], (void *)&req, 1) == 1) {
            return 0;
        }
        req->status = NVME_INTERNAL_DEV_ERROR | NVME_DNR;
        return 1;
    default:
        req->status = NVME_INVALID_OPCODE | NVME_DNR;
        return 1;
    }
}
