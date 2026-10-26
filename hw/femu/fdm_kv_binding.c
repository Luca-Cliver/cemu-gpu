#include "qemu/osdep.h"
#include "fdm_kv_binding.h"
#include "csd/memory.h"
#include "bbssd/bb.h"
#include "backend/backend.h"
#include "fdm_kv_receive.h"
#include "fdm_kv_aggregate.h"
#include "fdm_kv_writer.h"

static bool valid_range(FemuCtrl *ctrl, const MemoryRange *range,
                        NvmeNamespace **namespace)
{
    if (!range->nsid || range->nsid > ctrl->num_namespaces) {
        return false;
    }
    NvmeNamespace *ns = nvme_find_namespace(ctrl, range->nsid);
    if (!ns || ns->csi != NVME_CSI_MEMORY || !ns->private ||
        !ns->backend || ns->backend->type != FEMU_DRAM_BACKEND ||
        ns->backend->size <= 0 || !ns->backend->logical_space ||
        !range->len || range->sb % 512 || range->len % 512 ||
        range->sb > ns->backend->size ||
        range->len > ns->backend->size - range->sb ||
        range->backend != ns->backend ||
        range->addr != backend_addr(ns->backend, range->sb)) {
        return false;
    }
    *namespace = ns;
    return true;
}

uint16_t fdm_kv_bind(NvmeNamespace *compute_ns, MemoryRangeSet *mrs,
                     const FDMKV_BindParams_t *wire, Program *program)
{
    FemuCtrl *ctrl = compute_ns->ctrl;
    NvmeNamespace *meta_ns, *data_ns;
    if (ctrl->native_kv != (program != NULL) || !mrs || mrs->numr != 2 || !mrs->mr ||
        le32_to_cpu(wire->magic) != CEMU_FDM_KV_BIND_MAGIC ||
        le32_to_cpu(wire->version) != CEMU_FDM_KV_BIND_VERSION) {
        return NVME_INVALID_FIELD;
    }
    if (program && (program->state != PROGRAM_STATE_ACTIVATED ||
        program->type != PROGRAM_TYPE_SHARED_LIB || program->target != PROGRAM_TARGET_HOST ||
        program->is_indirect || !program->shared_lib.jit_fn ||
        program->runtime || program->runtime_scale)) {
        return NVME_INVALID_PROGRAM_DATA;
    }
    if (program && (le32_to_cpu(wire->batch_pages) > CEMU_NATIVE_INTERNAL_MAX_PAGES)) {
        return NVME_INVALID_FIELD;
    }
    if (program && ctrl->native_inflight) {
        return NVME_CMD_SEQ_ERROR;
    }
    if (mrs->in_use || mrs->fdm_kv_binding) {
        return NVME_MR_SET_IN_USE;
    }
    if (!valid_range(ctrl, &mrs->mr[CEMU_FDM_KV_META_MR], &meta_ns) ||
        !valid_range(ctrl, &mrs->mr[CEMU_FDM_KV_DATA_MR], &data_ns) ||
        meta_ns != data_ns) {
        return NVME_INVALID_MEMORY_NS;
    }
    uint32_t nvm_nsid = le32_to_cpu(wire->nvm_nsid);
    if (!nvm_nsid || nvm_nsid > ctrl->num_namespaces) {
        return NVME_INVALID_NSID | NVME_DNR;
    }
    NvmeNamespace *nvm = nvme_find_namespace(ctrl, nvm_nsid);
    if (!nvm || nvm->csi != NVME_CSI_NVM || !nvm->private) {
        return NVME_INVALID_FIELD;
    }
    NvmNamespace *nvm_state = nvm_ns(nvm);
    MemoryNamespace *fdm_state = meta_ns->private;
    NvmParams *nand = nvm_state->params;
    if (!nand || nand->secsz <= 0 || nand->secs_per_pg <= 0 ||
        nand->nchs <= 0 || nand->pgs_per_blk <= 0 ||
        (uint64_t)nand->secsz * nand->secs_per_pg !=
        le32_to_cpu(wire->geometry.page_bytes)) {
        return NVME_INVALID_FIELD;
    }
    g_mutex_lock(&fdm_state->fdm_kv_lock);
    if (nvm_state->native_tail || nvm_state->fdm_kvbuffer_initialized || nvm_state->fdm_kv_buffer ||
        fdm_state->fdm_kv_binding) {
        g_mutex_unlock(&fdm_state->fdm_kv_lock);
        return NVME_MR_SET_IN_USE;
    }
    FDMKV_InitParams_t params = {
        .identity = {
            .controller = ctrl->drive_id,
            .namespace_id = nvm->id,
            .session_id = le64_to_cpu(wire->session_id),
            .generation = le64_to_cpu(wire->generation),
        },
        .geometry = {
            .batches = le32_to_cpu(wire->geometry.batches),
            .layers = le32_to_cpu(wire->geometry.layers),
            .heads = le32_to_cpu(wire->geometry.heads),
            .head_dim = le32_to_cpu(wire->geometry.head_dim),
            .max_tokens = le64_to_cpu(wire->geometry.max_tokens),
            .channels_per_page = le32_to_cpu(wire->geometry.channels_per_page),
            .element_bytes = le32_to_cpu(wire->geometry.element_bytes),
            .page_bytes = le32_to_cpu(wire->geometry.page_bytes),
            .reserved = le32_to_cpu(wire->geometry.reserved),
        },
        .meta_addr = mrs->mr[CEMU_FDM_KV_META_MR].addr,
        .meta_bytes = mrs->mr[CEMU_FDM_KV_META_MR].len,
        .data_addr = mrs->mr[CEMU_FDM_KV_DATA_MR].addr,
        .data_bytes = mrs->mr[CEMU_FDM_KV_DATA_MR].len,
        .input_bytes = le64_to_cpu(wire->input_bytes),
        .nand_channels = nand->nchs,
        .block_pages = nand->pgs_per_blk,
        .batch_pages = le32_to_cpu(wire->batch_pages),
    };
    FDMKV_Binding_t *binding = g_new0(FDMKV_Binding_t, 1);
    if (!fdm_kv_buffer_init(&params, &binding->buffer)) {
        g_free(binding);
        g_mutex_unlock(&fdm_state->fdm_kv_lock);
        return NVME_INVALID_FIELD;
    }
    binding->nvm_ns = nvm;
    binding->fdm_ns = meta_ns;
    binding->mrs = mrs;
    binding->aggregate_program = program;
    binding->input_sb = mrs->mr[CEMU_FDM_KV_DATA_MR].sb;
    binding->input_bytes = params.input_bytes;
    if (program && !fdm_kv_writer_create(binding)) {
        binding->buffer->initialized = 0;
        backend_cuda_mark_host_dirty(meta_ns->backend, params.meta_addr, params.meta_bytes);
        g_free(binding);
        g_mutex_unlock(&fdm_state->fdm_kv_lock);
        return NVME_INTERNAL_DEV_ERROR | NVME_DNR;
    }
    mrs->in_use = 1;
    mrs->fdm_kv_binding = binding;
    fdm_state->fdm_kv_binding = binding;
    nvm_state->fdm_kv_buffer = binding->buffer;
    nvm_state->fdm_kvbuffer_initialized = true;
    ctrl->fdm_kv_bindings = g_list_prepend(ctrl->fdm_kv_bindings, binding);
    if (program) {
        qatomic_inc(&program->jobs_running);
        qatomic_inc(&program->fdm_kv_users);
    }
    backend_cuda_mark_host_dirty(meta_ns->backend, params.meta_addr, params.meta_bytes);
    g_mutex_unlock(&fdm_state->fdm_kv_lock);
    return NVME_SUCCESS;
}

static bool unbind_buffer(MemoryRangeSet *mrs, bool cancel)
{
    FDMKV_Binding_t *binding = mrs->fdm_kv_binding;
    if (!binding) {
        return true;
    }
    NvmNamespace *nvm_state = nvm_ns(binding->nvm_ns);
    MemoryNamespace *fdm_state = binding->fdm_ns->private;
    g_mutex_lock(&fdm_state->fdm_kv_lock);
    if (mrs->in_use != 1 || (!cancel && (binding->pending_write ||
        binding->buffer->stats.inflight_blocks || binding->buffer->leases ||
        (binding->writer && !binding->buffer->failed &&
         (binding->buffer->stats.staged_pages ||
          binding->buffer->receive->state == FDM_KV_RX_RECEIVING))))) {
        g_mutex_unlock(&fdm_state->fdm_kv_lock);
        return false;
    }
    fdm_kv_writer_request_stop(binding);
    g_mutex_unlock(&fdm_state->fdm_kv_lock);
    fdm_kv_writer_join(binding);
    g_mutex_lock(&fdm_state->fdm_kv_lock);
    fdm_kv_writer_free(binding);
    FemuCtrl *ctrl = binding->nvm_ns->ctrl;
    if (binding->pending_write) {
        fdm_kv_receive_fail(binding->buffer);
        binding->pending_write = NULL;
    }
    nvm_state->fdm_kvbuffer_initialized = false;
    nvm_state->fdm_kv_buffer = NULL;
    fdm_state->fdm_kv_binding = NULL;
    binding->buffer->initialized = 0;
    backend_cuda_mark_host_dirty(binding->fdm_ns->backend, binding->buffer,
                                 sizeof(*binding->buffer));
    mrs->fdm_kv_binding = NULL;
    mrs->in_use = 0;
    ctrl->fdm_kv_bindings = g_list_remove(ctrl->fdm_kv_bindings, binding);
    if (binding->aggregate_program) {
        qatomic_dec(&binding->aggregate_program->fdm_kv_users);
        qatomic_dec(&binding->aggregate_program->jobs_running);
        femu_log("FDM KV non-frozen aggregation: calls=%" PRIu64
                 ", host_ns=%" PRIu64 "; no synthetic runtime\n",
                 binding->aggregate_calls, binding->aggregate_host_ns);
    }
    g_free(binding);
    g_mutex_unlock(&fdm_state->fdm_kv_lock);
    return true;
}

bool fdm_kv_unbind(MemoryRangeSet *mrs)
{
    return unbind_buffer(mrs, false);
}

bool fdm_kv_complete_write(NvmeRequest *request)
{
    MemoryNamespace *memory = request->ns->private;
    g_mutex_lock(&memory->fdm_kv_lock);
    FDMKV_Binding_t *binding = memory->fdm_kv_binding;
    uint64_t result = FDM_KV_AGGREGATE_INVALID;
    if (!binding || binding->pending_write != request || !binding->aggregate_program) {
        request->status = NVME_INTERNAL_DEV_ERROR | NVME_DNR;
        goto done;
    }
    uint64_t now = clock_ns();
    if (now < binding->buffer->receive->max_write_ready_ns) {
        request->stat.expire_time = binding->buffer->receive->max_write_ready_ns;
        g_mutex_unlock(&memory->fdm_kv_lock);
        return false;
    }
    if (binding->buffer->failed) {
        request->status = NVME_INTERNAL_DEV_ERROR | NVME_DNR;
        goto finish;
    }
    if (binding->buffer->receive->header.operation == FDM_KV_OP_DRAIN) {
        if (binding->buffer->leases) {
            goto retry;
        }
        if (!fdm_kv_writer_drain(binding)) {
            request->status = NVME_INTERNAL_DEV_ERROR | NVME_DNR;
            goto finish;
        }
        if (!binding->buffer->sealed) {
            goto retry;
        }
        binding->buffer->receive->state = FDM_KV_RX_AGGREGATED;
        result = FDM_KV_AGGREGATE_OK;
        goto release;
    }
    for (uint32_t index = 0; index < binding->mrs->numr; index++) {
        MemoryRange *range = &binding->mrs->mr[index];
        if (backend_cuda_prepare_host(range->backend, range->addr, range->len)) {
            request->status = NVME_INTERNAL_DEV_ERROR | NVME_DNR;
            goto finish;
        }
    }
    struct ubpf_jit_args arguments = {
        .numr = binding->mrs->numr,
        .mr_addr = binding->mrs->mr_addr,
        .mr_len = binding->mrs->mr_len,
        .cparam1 = CEMU_FDM_KV_AGGREGATE_COMMAND,
        .cparam2 = clock_ns(),
    };
    uint64_t host_start = qemu_clock_get_ns(QEMU_CLOCK_REALTIME);
    result = binding->aggregate_program->shared_lib.jit_fn(&arguments);
    binding->aggregate_host_ns += qemu_clock_get_ns(QEMU_CLOCK_REALTIME) - host_start;
    binding->aggregate_calls++;
    if (result == FDM_KV_AGGREGATE_BUSY && binding->buffer->leases &&
        binding->buffer->receive->state == FDM_KV_RX_READY) {
        goto retry;
    }
    if (result == FDM_KV_AGGREGATE_NO_SPACE && fdm_kv_writer_can_progress(binding)) {
        fdm_kv_writer_wake(binding);
        goto retry;
    }
release:
    if (result == FDM_KV_AGGREGATE_OK && fdm_kv_receive_release(binding->buffer)) {
        request->status = NVME_SUCCESS;
        fdm_kv_writer_wake(binding);
    } else {
        request->status = result == FDM_KV_AGGREGATE_NO_SPACE ?
                          NVME_CAP_EXCEEDED | NVME_DNR : NVME_INVALID_FIELD | NVME_DNR;
    }
finish:
    if (request->status) {
        fdm_kv_receive_fail(binding->buffer);
    }
    binding->pending_write = NULL;
    for (uint32_t index = 0; index < binding->mrs->numr; index++) {
        MemoryRange *range = &binding->mrs->mr[index];
        backend_cuda_mark_host_dirty(range->backend, range->addr, range->len);
    }
done:
    request->fdm_kv_pending = false;
    if (request->status) {
        request->cqe.n.result = 0;
    }
    g_mutex_unlock(&memory->fdm_kv_lock);
    return true;
retry:
    request->stat.expire_time = clock_ns() + 10000;
    g_mutex_unlock(&memory->fdm_kv_lock);
    return false;
}

static bool overlaps(uint64_t offset, uint64_t bytes, const MemoryRange *range)
{
    return bytes && (offset >= range->sb ? offset - range->sb < range->len :
                                          range->sb - offset < bytes);
}

bool fdm_kv_range_overlaps(const FDMKV_Binding_t *binding,
                           uint64_t offset, uint64_t bytes)
{
    return binding &&
        (overlaps(offset, bytes, &binding->mrs->mr[CEMU_FDM_KV_META_MR]) ||
         overlaps(offset, bytes, &binding->mrs->mr[CEMU_FDM_KV_DATA_MR]));
}

uint16_t fdm_kv_write_check(const FDMKV_Binding_t *binding,
                            uint64_t offset, uint64_t bytes, bool *input)
{
    *input = false;
    if (!bytes && fdm_kv_range_overlaps(binding, offset, 1)) {
        return NVME_INVALID_FIELD | NVME_DNR;
    }
    if (!fdm_kv_range_overlaps(binding, offset, bytes)) {
        return NVME_SUCCESS;
    }
    if (offset < binding->input_sb || offset - binding->input_sb >= binding->input_bytes ||
        bytes > binding->input_bytes - (offset - binding->input_sb)) {
        return NVME_INVALID_FIELD | NVME_DNR;
    }
    FDMKV_ReceiveResult result = fdm_kv_receive_check(
        binding->buffer, offset - binding->input_sb, bytes);
    if (result == FDM_KV_RECEIVE_BUSY) {
        return NVME_CMD_SEQ_ERROR | NVME_DNR;
    }
    if (result != FDM_KV_RECEIVE_ACCEPTED) {
        return NVME_INVALID_FIELD | NVME_DNR;
    }
    *input = true;
    return NVME_SUCCESS;
}

void fdm_kv_bindings_clear(FemuCtrl *ctrl)
{
    while (ctrl->fdm_kv_bindings) {
        FDMKV_Binding_t *binding = ctrl->fdm_kv_bindings->data;
        bool released = unbind_buffer(binding->mrs, true);
        assert(released);
    }
}
