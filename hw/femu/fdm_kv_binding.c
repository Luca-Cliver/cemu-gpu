#include "qemu/osdep.h"
#include "fdm_kv_binding.h"
#include "csd/memory.h"
#include "bbssd/bb.h"
#include "backend/backend.h"
#include "fdm_kv_receive.h"
#include "fdm_kv_aggregate.h"
#include "fdm_kv_writer.h"
#include "fdm_kv_attention.h"
#include <dlfcn.h>
#include <sys/resource.h>

static void *aggregate_worker(void *opaque);

static uint64_t aggregate_profile_clock(void)
{
    return qemu_clock_get_ns(QEMU_CLOCK_REALTIME);
}

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
                     const FDMKV_BindSlotsParams_t *descriptor, uint32_t bytes, Program *program)
{
    const FDMKV_BindParams_t *wire = &descriptor->base;
    bool multislot = bytes == sizeof(*descriptor) &&
        le32_to_cpu(wire->version) == CEMU_FDM_KV_BIND_SLOTS_VERSION;
    uint32_t input_slots = multislot ? le32_to_cpu(descriptor->input_slots) : 0;
    FemuCtrl *ctrl = compute_ns->ctrl;
    NvmeNamespace *meta_ns, *data_ns;
    if (ctrl->native_kv != (program != NULL) || !mrs || mrs->numr < 2 ||
        mrs->numr > CEMU_FDM_KV_MAX_DATA_MRS + 1 || !mrs->mr ||
        le32_to_cpu(wire->magic) != CEMU_FDM_KV_BIND_MAGIC ||
        (!multislot && (bytes != sizeof(*wire) ||
                       le32_to_cpu(wire->version) != CEMU_FDM_KV_BIND_VERSION)) ||
        (multislot && (!input_slots || input_slots > CEMU_FDM_KV_MAX_INPUT_SLOTS ||
                      le32_to_cpu(descriptor->transfer_protocol) != CEMU_FDM_KV_TRANSFER_PROTOCOL))) {
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
    const char *mode = g_getenv("CEMU_FDM_AGGREGATE_MODE");
    bool polling = g_strcmp0(mode, "poll") == 0;
    if (multislot && !polling) {
        return NVME_INVALID_FIELD;
    }
    FDMKV_ServiceFunction service_fn = NULL;
    if (mode && *mode && !polling && strcmp(mode, "notify")) {
        femu_err("CEMU_FDM_AGGREGATE_MODE must be notify or poll\n");
        return NVME_INVALID_FIELD;
    }
    if (polling) {
        if (!program || g_strcmp0(g_getenv("CEMU_FDM_ASYNC_AGGREGATE"), "1")) {
            femu_err("poll aggregation requires an operator and CEMU_FDM_ASYNC_AGGREGATE=1\n");
            return NVME_INVALID_FIELD;
        }
        service_fn = dlsym(program->shared_lib.so_handle, "fdm_kv_aggregate_service_v1");
        if (!service_fn) {
            femu_err("poll aggregation requires rebuilt fdm_kv_aggregate.so with service v1\n");
            return NVME_INVALID_PROGRAM_DATA;
        }
    }
    FDMKV_ProfileFunction profile_fn = NULL;
    if (program && g_strcmp0(g_getenv("CEMU_FDM_AGGREGATE_PROFILE"), "1") == 0) {
        const char *trace = g_getenv("CEMU_FDM_IO_TRACE");
        profile_fn = dlsym(program->shared_lib.so_handle, "fdm_kv_aggregate_profile_v1");
        if (!trace || !*trace || !profile_fn) {
            femu_err("aggregate profiling requires CEMU_FDM_IO_TRACE and rebuilt fdm_kv_aggregate.so\n");
            return NVME_INVALID_PROGRAM_DATA;
        }
    }
    if (mrs->in_use || mrs->fdm_kv_binding) {
        return NVME_MR_SET_IN_USE;
    }
    if (!valid_range(ctrl, &mrs->mr[CEMU_FDM_KV_META_MR], &meta_ns) ||
        !valid_range(ctrl, &mrs->mr[CEMU_FDM_KV_DATA_MR], &data_ns) ||
        meta_ns != data_ns) {
        return NVME_INVALID_MEMORY_NS;
    }
    FDMKV_DataSegment_t segments[CEMU_FDM_KV_MAX_DATA_MRS] = {0};
    for (uint32_t index = 1; index < mrs->numr; index++) {
        if (!valid_range(ctrl, &mrs->mr[index], &data_ns) || data_ns != meta_ns) {
            return NVME_INVALID_MEMORY_NS;
        }
        segments[index - 1].address = mrs->mr[index].addr;
        segments[index - 1].bytes = mrs->mr[index].len;
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
    g_rw_lock_writer_lock(&fdm_state->fdm_kv_binding_lock);
    g_mutex_lock(&fdm_state->fdm_kv_lock);
    if (nvm_state->native_tail || nvm_state->fdm_kvbuffer_initialized || nvm_state->fdm_kv_buffer ||
        fdm_state->fdm_kv_binding) {
        g_mutex_unlock(&fdm_state->fdm_kv_lock);
        g_rw_lock_writer_unlock(&fdm_state->fdm_kv_binding_lock);
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
        .data_segments = segments,
        .segment_count = mrs->numr - 1,
        .input_bytes = le64_to_cpu(wire->input_bytes),
        .input_slots = input_slots,
        .nand_channels = nand->nchs,
        .block_pages = nand->pgs_per_blk,
        .batch_pages = le32_to_cpu(wire->batch_pages),
    };
    FDMKV_Binding_t *binding = g_new0(FDMKV_Binding_t, 1);
    if (!fdm_kv_buffer_init(&params, &binding->buffer)) {
        g_free(binding);
        g_mutex_unlock(&fdm_state->fdm_kv_lock);
        g_rw_lock_writer_unlock(&fdm_state->fdm_kv_binding_lock);
        return NVME_INVALID_FIELD;
    }
    binding->nvm_ns = nvm;
    binding->fdm_ns = meta_ns;
    binding->mrs = mrs;
    binding->aggregate_program = program;
    binding->aggregate_profile_fn = profile_fn;
    binding->aggregate_service_fn = service_fn;
    binding->input_sb = mrs->mr[CEMU_FDM_KV_DATA_MR].sb;
    binding->input_bytes = params.input_bytes;
    binding->async_aggregate = program && g_strcmp0(g_getenv("CEMU_FDM_ASYNC_AGGREGATE"), "1") == 0;
    g_cond_init(&binding->aggregate_wake);
    if (program && !fdm_kv_writer_create(binding)) {
        binding->buffer->initialized = 0;
        backend_cuda_mark_host_dirty(meta_ns->backend, params.meta_addr, params.meta_bytes);
        g_cond_clear(&binding->aggregate_wake);
        g_free(binding);
        g_mutex_unlock(&fdm_state->fdm_kv_lock);
        g_rw_lock_writer_unlock(&fdm_state->fdm_kv_binding_lock);
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
    if (binding->async_aggregate) {
        if (input_slots) {
            binding->slot_tasks = g_new0(FDMKV_SlotTask_t, input_slots);
        } else {
            binding->aggregate_request = g_new0(NvmeRequest, 1);
        }
        binding->aggregate_worker = g_thread_new("fdm-kv-aggregate", aggregate_worker, binding);
        femu_log("FDM KV aggregation mode=%s, input_slots=%u, idle_poll_us=%u\n",
                 service_fn ? "poll-service-v1" : "notify", input_slots ? input_slots : 1,
                 service_fn ? 10 : 0);
    }
    backend_cuda_mark_host_dirty(meta_ns->backend, params.meta_addr, params.meta_bytes);
    g_mutex_unlock(&fdm_state->fdm_kv_lock);
    g_rw_lock_writer_unlock(&fdm_state->fdm_kv_binding_lock);
    return NVME_SUCCESS;
}

static void select_slot_locked(FDMKV_Binding_t *binding, uint32_t index)
{
    FDMKV_InputSlot_t *slot = &binding->buffer->slots[index];
    FDMKV_SlotTask_t *runtime = &binding->slot_tasks[index];
    binding->buffer->active_slot = index;
    binding->buffer->input = slot->input;
    binding->buffer->receive = slot->receive;
    binding->pending_write = runtime->pending;
    binding->aggregate_request = &runtime->task;
    binding->aggregate_qid = runtime->qid;
}

static void save_slot_locked(FDMKV_Binding_t *binding, uint32_t index)
{
    FDMKV_SlotTask_t *runtime = &binding->slot_tasks[index];
    runtime->pending = binding->pending_write;
    runtime->qid = binding->aggregate_qid;
    runtime->background = runtime->pending == &runtime->task;
    bool active = false;
    for (uint32_t slot = 0; slot < binding->buffer->input_slots; slot++) {
        active |= binding->slot_tasks[slot].background;
    }
    qatomic_set(&binding->aggregate_active, active);
}

static bool slots_busy_locked(const FDMKV_Binding_t *binding)
{
    for (uint32_t index = 0; index < binding->buffer->input_slots; index++) {
        const FDMKV_InputSlot_t *slot = &binding->buffer->slots[index];
        if (binding->slot_tasks[index].pending || slot->dma_active ||
            (!binding->buffer->failed && slot->state != FDM_KV_SLOT_EMPTY)) {
            return true;
        }
    }
    return false;
}

static bool unbind_buffer(MemoryRangeSet *mrs, bool cancel)
{
    FDMKV_Binding_t *binding = mrs->fdm_kv_binding;
    if (!binding) {
        return true;
    }
    NvmNamespace *nvm_state = nvm_ns(binding->nvm_ns);
    MemoryNamespace *fdm_state = binding->fdm_ns->private;
    g_rw_lock_writer_lock(&fdm_state->fdm_kv_binding_lock);
    g_mutex_lock(&fdm_state->fdm_kv_lock);
    if (mrs->in_use != 1 || (!cancel && (slots_busy_locked(binding) ||
        binding->pending_write || binding->aggregate_active ||
        binding->buffer->stats.inflight_blocks || binding->buffer->leases ||
        (binding->writer && !binding->buffer->failed &&
         (binding->buffer->stats.staged_pages ||
          binding->buffer->receive->state == FDM_KV_RX_RECEIVING))))) {
        g_mutex_unlock(&fdm_state->fdm_kv_lock);
        g_rw_lock_writer_unlock(&fdm_state->fdm_kv_binding_lock);
        return false;
    }
    qatomic_set(&binding->aggregate_stop, true);
    g_cond_broadcast(&binding->aggregate_wake);
    fdm_kv_writer_request_stop(binding);
    g_mutex_unlock(&fdm_state->fdm_kv_lock);
    if (binding->aggregate_worker) {
        g_thread_join(binding->aggregate_worker);
        binding->aggregate_worker = NULL;
    }
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
    g_assert(!binding->waiting_attention);
    if (binding->slot_tasks) {
        g_free(binding->slot_tasks);
    } else {
        g_free(binding->aggregate_request);
    }
    g_cond_clear(&binding->aggregate_wake);
    g_free(binding);
    g_mutex_unlock(&fdm_state->fdm_kv_lock);
    g_rw_lock_writer_unlock(&fdm_state->fdm_kv_binding_lock);
    return true;
}

bool fdm_kv_unbind(MemoryRangeSet *mrs)
{
    return unbind_buffer(mrs, false);
}

static bool complete_write_locked(NvmeRequest *request, FDMKV_ProfileFunction service_step)
{
    MemoryNamespace *memory = request->ns->private;
    FDMKV_Binding_t *binding = memory->fdm_kv_binding;
    uint32_t slot_index = binding ? binding->buffer->active_slot : 0;
    uint64_t result = FDM_KV_AGGREGATE_INVALID;
    if (!binding || binding->pending_write != request || !binding->aggregate_program) {
        request->status = NVME_INTERNAL_DEV_ERROR | NVME_DNR;
        goto done;
    }
    uint64_t now = clock_ns();
    if (now < binding->buffer->receive->max_write_ready_ns) {
        if (request->fdm_io_trace.enabled) {
            request->fdm_io_trace.retry = FDM_IO_RETRY_MODEL;
        }
        request->stat.expire_time = binding->buffer->receive->max_write_ready_ns;
        return false;
    }
    if (binding->buffer->failed) {
        request->status = NVME_INTERNAL_DEV_ERROR | NVME_DNR;
        goto finish;
    }
    if (binding->async_aggregate && request != binding->aggregate_request &&
        binding->buffer->receive->header.operation == FDM_KV_OP_APPEND) {
        const FDMKV_AppendHeader_t *header = &binding->buffer->receive->header;
        const FDMKV_Geometry_t *geometry = &binding->buffer->layout.geometry;
        uint32_t batches = header->version == CEMU_FDM_KV_LAYER_PACKET_VERSION ? header->batch_count : 1;
        for (uint32_t batch = header->batch;
             !binding->slot_tasks && batch < header->batch + batches; batch++) {
            for (uint32_t head = header->head_first; head < header->head_first + header->head_count; head++) {
                uint64_t index = ((uint64_t)batch * geometry->layers + header->layer) * geometry->heads + head;
                if (binding->buffer->head_states[index].accepted != header->token_first) {
                    request->status = NVME_INVALID_FIELD | NVME_DNR;
                    goto finish;
                }
            }
        }
        if (binding->slot_tasks && !fdm_kv_slot_publish(binding->buffer, slot_index, now)) {
            request->status = NVME_INVALID_FIELD | NVME_DNR;
            goto finish;
        }
        NvmeRequest *task = binding->aggregate_request;
        memset(task, 0, sizeof(*task));
        task->ns = request->ns;
        task->cmd = request->cmd;
        task->stat = request->stat;
        task->fdm_kv_pending = true;
        task->fdm_io_trace = (FdmIoTrace){
            .enabled = request->fdm_io_trace.enabled, .background = true, .async_aggregate = true,
            .session = binding->buffer->identity.session_id, .packet_id = header->request_id,
            .token_first = header->token_first, .operation = header->operation, .accepted_ns = now,
            .packet_bytes = header->wire_bytes,
        };
        binding->aggregate_qid = request->sq->sqid;
        binding->pending_write = task;
        qatomic_set(&binding->aggregate_active, true);
        request->fdm_kv_pending = false;
        request->fdm_io_trace.async_aggregate = true;
        request->fdm_io_trace.accepted_ns = now;
        if (!binding->aggregate_service_fn) {
            g_cond_signal(&binding->aggregate_wake);
        }
        return true;
    }
    if (binding->buffer->receive->header.operation == FDM_KV_OP_DRAIN) {
        if (binding->buffer->leases) {
            if (request->fdm_io_trace.enabled) {
                request->fdm_io_trace.busy_retries++;
                request->fdm_io_trace.retry = FDM_IO_RETRY_BUSY;
            }
            goto retry;
        }
        if (!fdm_kv_writer_drain(binding)) {
            request->status = NVME_INTERNAL_DEV_ERROR | NVME_DNR;
            goto finish;
        }
        if (!binding->buffer->sealed) {
            if (request->fdm_io_trace.enabled) {
                request->fdm_io_trace.retry = FDM_IO_RETRY_DRAIN;
            }
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
    /* Probe outside the operator's Host timer, only with detailed profiling.
     * CPU time and fault/switch deltas distinguish execution from preemption;
     * they do not identify cache misses or time spent in a particular stage. */
    struct rusage usage_before, usage_after;
    struct timespec cpu_before, cpu_after;
    int cpu_begin = -1;
    if (binding->aggregate_profile_fn && request->fdm_io_trace.enabled) {
        cpu_begin = sched_getcpu();
    }
    bool probe = binding->aggregate_profile_fn && request->fdm_io_trace.enabled &&
                 getrusage(RUSAGE_THREAD, &usage_before) == 0 &&
                 clock_gettime(CLOCK_THREAD_CPUTIME_ID, &cpu_before) == 0;
    uint64_t host_start = qemu_clock_get_ns(QEMU_CLOCK_REALTIME);
    uint64_t aggregate_start = request->fdm_io_trace.enabled ? clock_ns() : 0;
    if (binding->aggregate_profile_fn && request->fdm_io_trace.enabled) {
        FDMKV_ProfileFunction aggregate = service_step ? service_step : binding->aggregate_profile_fn;
        result = aggregate(&arguments,
            &request->fdm_io_trace.aggregate_profile, aggregate_profile_clock);
    } else if (service_step) {
        result = service_step(&arguments, NULL, NULL);
    } else {
        result = binding->aggregate_program->shared_lib.jit_fn(&arguments);
    }
    if (request->fdm_io_trace.enabled) {
        request->fdm_io_trace.aggregate_ns += clock_ns() - aggregate_start;
        request->fdm_io_trace.aggregate_host_ns += qemu_clock_get_ns(QEMU_CLOCK_REALTIME) - host_start;
        request->fdm_io_trace.aggregate_calls++;
    }
    binding->aggregate_host_ns += qemu_clock_get_ns(QEMU_CLOCK_REALTIME) - host_start;
    binding->aggregate_calls++;
    uint64_t host_end = probe ? qemu_clock_get_ns(QEMU_CLOCK_REALTIME) : 0;
    if (probe && clock_gettime(CLOCK_THREAD_CPUTIME_ID, &cpu_after) == 0 &&
        getrusage(RUSAGE_THREAD, &usage_after) == 0) {
        FdmIoTrace *trace = &request->fdm_io_trace;
        int cpu_end = sched_getcpu();
        if (!trace->aggregate_probe_calls) {
            trace->aggregate_first_host_ns = host_start;
            trace->aggregate_cpu_begin = cpu_begin;
            trace->aggregate_tid = qemu_get_thread_id();
            const FDMKV_AppendHeader_t *header = &binding->buffer->receive->header;
            trace->aggregate_batch = header->batch;
            trace->aggregate_layer = header->layer;
            trace->aggregate_head_first = header->head_first;
            trace->aggregate_head_count = header->head_count;
        }
        trace->aggregate_last_host_ns = host_end;
        trace->aggregate_cpu_end = cpu_end;
        trace->aggregate_cpu_changes += cpu_begin >= 0 && cpu_end >= 0 && cpu_begin != cpu_end;
        trace->aggregate_probe_calls++;
        trace->aggregate_thread_cpu_ns +=
            (cpu_after.tv_sec - cpu_before.tv_sec) * INT64_C(1000000000) +
            cpu_after.tv_nsec - cpu_before.tv_nsec;
        trace->aggregate_minor_faults += usage_after.ru_minflt - usage_before.ru_minflt;
        trace->aggregate_major_faults += usage_after.ru_majflt - usage_before.ru_majflt;
        trace->aggregate_voluntary_switches += usage_after.ru_nvcsw - usage_before.ru_nvcsw;
        trace->aggregate_involuntary_switches += usage_after.ru_nivcsw - usage_before.ru_nivcsw;
    }
    if (result == FDM_KV_AGGREGATE_PROGRESS) {
        request->fdm_io_trace.retry = FDM_IO_RETRY_AGGREGATE;
        for (uint32_t index = 0; index < binding->mrs->numr; index++) {
            MemoryRange *range = &binding->mrs->mr[index];
            backend_cuda_mark_host_dirty(range->backend, range->addr, range->len);
        }
        fdm_kv_writer_wake(binding);
        request->stat.expire_time = clock_ns();
        return false;
    }
    if (result == FDM_KV_AGGREGATE_BUSY && binding->buffer->leases &&
        binding->buffer->receive->state == FDM_KV_RX_READY) {
        if (request->fdm_io_trace.enabled) {
            request->fdm_io_trace.busy_retries++;
            request->fdm_io_trace.retry = FDM_IO_RETRY_BUSY;
        }
        goto retry;
    }
    if (result == FDM_KV_AGGREGATE_NO_SPACE && fdm_kv_writer_can_progress(binding)) {
        if (request->fdm_io_trace.enabled) {
            request->fdm_io_trace.space_retries++;
            request->fdm_io_trace.retry = FDM_IO_RETRY_SPACE;
        }
        fdm_kv_writer_wake(binding);
        goto retry;
    }
release:
    if (result == FDM_KV_AGGREGATE_OK &&
        (binding->slot_tasks ? fdm_kv_slot_release(binding->buffer, slot_index) :
                               fdm_kv_receive_release(binding->buffer))) {
        request->status = NVME_SUCCESS;
        fdm_kv_writer_wake(binding);
    } else {
        request->status = result == FDM_KV_AGGREGATE_NO_SPACE ?
                          NVME_CAP_EXCEEDED | NVME_DNR : NVME_INVALID_FIELD | NVME_DNR;
    }
finish:
    if (request->status) {
        if (binding->slot_tasks) {
            fdm_kv_slot_fail(binding->buffer, slot_index);
        } else {
            fdm_kv_receive_fail(binding->buffer);
        }
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
    return true;
retry:
    request->stat.expire_time = clock_ns() + 10000;
    return false;
}

bool fdm_kv_complete_write(NvmeRequest *request)
{
    uint64_t start = fdm_io_trace_check_begin(request);
    MemoryNamespace *memory = request->ns->private;
    g_mutex_lock(&memory->fdm_kv_lock);
    FDMKV_Binding_t *binding = memory->fdm_kv_binding;
    uint32_t index = 0;
    bool done;
    if (binding && binding->slot_tasks) {
        for (; index < binding->buffer->input_slots; index++) {
            if (binding->slot_tasks[index].pending == request) {
                break;
            }
        }
        if (index == binding->buffer->input_slots) {
            request->status = NVME_INTERNAL_DEV_ERROR | NVME_DNR;
            request->cqe.n.result = 0;
            request->fdm_kv_pending = false;
            done = true;
            goto unlock;
        }
        select_slot_locked(binding, index);
        FDMKV_InputSlot_t *slot = &binding->buffer->slots[index];
        if (!binding->buffer->failed && slot->receive->header.operation == FDM_KV_OP_DRAIN) {
            if (slot->receive->header.request_id != binding->buffer->next_aggregate_id ||
                (slot->state == FDM_KV_SLOT_RECEIVED &&
                 !fdm_kv_slot_publish(binding->buffer, index, clock_ns())) ||
                (slot->state == FDM_KV_SLOT_READY && !fdm_kv_slot_begin(binding->buffer, index))) {
                request->stat.expire_time = MAX(clock_ns() + 10000, slot->receive->max_write_ready_ns);
                done = false;
                goto unlock;
            }
        }
    }
    done = complete_write_locked(request, NULL);
    if (binding && binding->slot_tasks) {
        save_slot_locked(binding, index);
        if (binding->buffer->failed) {
            g_cond_broadcast(&binding->aggregate_wake);
        }
    }
unlock:
    g_mutex_unlock(&memory->fdm_kv_lock);
    fdm_io_trace_check_end(request, start);
    return done;
}

void fdm_kv_async_notify_locked(FDMKV_Binding_t *binding)
{
    if (binding->async_aggregate) {
        fdm_kv_attention_publish_locked(binding);
        g_cond_broadcast(&binding->aggregate_wake);
    }
}

static FDMKV_ServiceResult aggregate_step_locked(FDMKV_Binding_t *binding,
                                                 FDMKV_ProfileFunction service_step)
{
    NvmeRequest *task = binding->aggregate_request;
    if (!task->fdm_io_trace.started_ns) {
        task->fdm_io_trace.started_ns = clock_ns();
    }
    uint64_t start = fdm_io_trace_check_begin(task);
    task->fdm_io_trace.retry = FDM_IO_RETRY_NONE;
    bool done = complete_write_locked(task, service_step);
    fdm_io_trace_check_end(task, start);
    if (fdm_kv_attention_publish_locked(binding) || binding->buffer->failed) {
        g_cond_broadcast(&binding->aggregate_wake);
    }
    if (done) {
        if (task->status) {
            femu_err("FDM KV background aggregation failed: session=%" PRIu64
                     ", packet=%" PRIu64 ", status=%u\n",
                     binding->buffer->identity.session_id,
                     binding->buffer->receive->header.request_id, task->status);
        }
        task->fdm_io_trace.finished_ns = clock_ns();
        fdm_io_trace_background(task, binding->aggregate_qid);
        qatomic_set(&binding->aggregate_active, false);
    }
    return done || task->fdm_io_trace.retry == FDM_IO_RETRY_AGGREGATE ?
           FDM_KV_SERVICE_PROGRESS : FDM_KV_SERVICE_IDLE;
}

static FDMKV_ServiceResult aggregate_service_step(void *opaque,
                                                  FDMKV_ProfileFunction service_step)
{
    FDMKV_Binding_t *binding = opaque;
    if (qatomic_read(&binding->aggregate_stop)) {
        return FDM_KV_SERVICE_STOP;
    }
    if (!qatomic_read(&binding->aggregate_active)) {
        return FDM_KV_SERVICE_IDLE;
    }
    MemoryNamespace *memory = binding->fdm_ns->private;
    g_mutex_lock(&memory->fdm_kv_lock);
    FDMKV_ServiceResult result = FDM_KV_SERVICE_IDLE;
    if (binding->aggregate_stop) {
        result = FDM_KV_SERVICE_STOP;
    } else if (!binding->slot_tasks) {
        result = aggregate_step_locked(binding, service_step);
    } else {
        for (uint32_t index = 0; index < binding->buffer->input_slots; index++) {
            FDMKV_InputSlot_t *slot = &binding->buffer->slots[index];
            if (!binding->slot_tasks[index].background ||
                (!binding->buffer->failed &&
                 slot->receive->header.request_id != binding->buffer->next_aggregate_id)) {
                continue;
            }
            if (!binding->buffer->failed && slot->state == FDM_KV_SLOT_READY &&
                !fdm_kv_slot_begin(binding->buffer, index)) {
                fdm_kv_slot_fail(binding->buffer, index);
            }
            select_slot_locked(binding, index);
            result = aggregate_step_locked(binding, service_step);
            save_slot_locked(binding, index);
            break;
        }
    }
    g_mutex_unlock(&memory->fdm_kv_lock);
    return result;
}

static void aggregate_service_relax(void *opaque, FDMKV_ServiceResult result)
{
    if (result == FDM_KV_SERVICE_IDLE) {
        g_usleep(10);
    } else {
        g_thread_yield();
    }
}

static void *aggregate_worker(void *opaque)
{
    FDMKV_Binding_t *binding = opaque;
    MemoryNamespace *memory = binding->fdm_ns->private;
    NvmeRequest *task = binding->aggregate_request;
    if (binding->aggregate_service_fn) {
        FDMKV_ServiceRuntime runtime = {
            .version = CEMU_FDM_KV_SERVICE_VERSION,
            .bytes = sizeof(runtime),
            .opaque = binding,
            .step = aggregate_service_step,
            .relax = aggregate_service_relax,
        };
        int status = binding->aggregate_service_fn(&runtime);
        g_mutex_lock(&memory->fdm_kv_lock);
        if (status || !binding->aggregate_stop) {
            femu_err("FDM KV persistent service terminated unexpectedly: status=%d\n", status);
            fdm_kv_receive_fail(binding->buffer);
            backend_cuda_mark_host_dirty(binding->fdm_ns->backend, binding->buffer,
                                         binding->buffer->meta_bytes);
            fdm_kv_attention_publish_locked(binding);
        }
    } else {
        g_mutex_lock(&memory->fdm_kv_lock);
    }
    while (!binding->aggregate_service_fn && !binding->aggregate_stop) {
        if (!binding->aggregate_active) {
            g_cond_wait(&binding->aggregate_wake, &memory->fdm_kv_lock);
            continue;
        }
        if (aggregate_step_locked(binding, NULL) == FDM_KV_SERVICE_IDLE) {
            g_cond_wait(&binding->aggregate_wake, &memory->fdm_kv_lock);
        }
        g_mutex_unlock(&memory->fdm_kv_lock);
        g_thread_yield();
        g_mutex_lock(&memory->fdm_kv_lock);
    }
    if (binding->slot_tasks) {
        for (uint32_t index = 0; index < binding->buffer->input_slots; index++) {
            FDMKV_SlotTask_t *runtime = &binding->slot_tasks[index];
            if (binding->buffer->slots[index].state != FDM_KV_SLOT_EMPTY) {
                fdm_kv_slot_fail(binding->buffer, index);
            }
            if (runtime->background) {
                runtime->task.status = NVME_INTERNAL_DEV_ERROR | NVME_DNR;
                runtime->task.fdm_io_trace.finished_ns = clock_ns();
                fdm_io_trace_background(&runtime->task, runtime->qid);
                runtime->pending = NULL;
                runtime->background = false;
            }
        }
        binding->pending_write = NULL;
        qatomic_set(&binding->aggregate_active, false);
    } else if (binding->aggregate_active) {
        fdm_kv_receive_fail(binding->buffer);
        binding->pending_write = NULL;
        task->status = NVME_INTERNAL_DEV_ERROR | NVME_DNR;
        task->fdm_io_trace.finished_ns = clock_ns();
        fdm_io_trace_background(task, binding->aggregate_qid);
        qatomic_set(&binding->aggregate_active, false);
    }
    g_cond_broadcast(&binding->aggregate_wake);
    g_mutex_unlock(&memory->fdm_kv_lock);
    return NULL;
}

bool fdm_kv_input_busy(FemuCtrl *ctrl, const NvmeCmd *command)
{
    uint32_t nsid = le32_to_cpu(command->nsid);
    if (command->opcode != NVME_CMD_MEMORY_WRITE || !nsid || nsid > ctrl->num_namespaces) {
        return false;
    }
    NvmeNamespace *ns = nvme_find_namespace(ctrl, nsid);
    if (ns->csi != NVME_CSI_MEMORY) {
        return false;
    }
    MemoryNamespace *memory = ns->private;
    g_rw_lock_reader_lock(&memory->fdm_kv_binding_lock);
    FDMKV_Binding_t *binding = memory->fdm_kv_binding;
    const NvmeMemoryWriteCmd *write = (const void *)command;
    uint64_t offset = le64_to_cpu(write->sb);
    bool busy = binding && qatomic_read(&binding->aggregate_active) &&
        offset >= binding->input_sb && offset - binding->input_sb < binding->input_bytes;
    if (binding && binding->slot_tasks && offset >= binding->input_sb &&
        offset - binding->input_sb < binding->input_bytes) {
        uint64_t transfer = le32_to_cpu(write->rsvd13[0]) |
                            ((uint64_t)le32_to_cpu(write->rsvd13[1]) << 32);
        uint64_t total = le32_to_cpu(write->rsvd13[2]);
        uint64_t length = le32_to_cpu(write->wl);
        g_mutex_lock(&memory->fdm_kv_lock);
        busy = !binding->buffer->failed && !binding->aggregate_stop &&
            !binding->buffer->draining && !binding->buffer->sealed &&
            transfer > binding->buffer->last_transfer_id && total && !(total % 512) &&
            total <= binding->input_bytes && length && !(length % 4) &&
            offset >= binding->input_sb && !((offset - binding->input_sb) % 4) &&
            offset - binding->input_sb < total && length <= total - (offset - binding->input_sb);
        for (uint32_t index = 0; busy && index < binding->buffer->input_slots; index++) {
            const FDMKV_InputSlot_t *slot = &binding->buffer->slots[index];
            if (slot->state == FDM_KV_SLOT_EMPTY || slot->state == FDM_KV_SLOT_RECEIVING ||
                slot->state == FDM_KV_SLOT_RECEIVED ||
                slot->receive->header.operation == FDM_KV_OP_DRAIN) {
                busy = false;
            }
        }
        g_mutex_unlock(&memory->fdm_kv_lock);
    }
    g_rw_lock_reader_unlock(&memory->fdm_kv_binding_lock);
    return busy;
}

void fdm_kv_aggregators_stop(FemuCtrl *ctrl)
{
    for (GList *entry = ctrl->fdm_kv_bindings; entry; entry = entry->next) {
        FDMKV_Binding_t *binding = entry->data;
        MemoryNamespace *memory = binding->fdm_ns->private;
        g_mutex_lock(&memory->fdm_kv_lock);
        qatomic_set(&binding->aggregate_stop, true);
        g_cond_broadcast(&binding->aggregate_wake);
        g_mutex_unlock(&memory->fdm_kv_lock);
    }
    for (GList *entry = ctrl->fdm_kv_bindings; entry; entry = entry->next) {
        FDMKV_Binding_t *binding = entry->data;
        if (binding->aggregate_worker) {
            g_thread_join(binding->aggregate_worker);
            binding->aggregate_worker = NULL;
        }
    }
}

static bool overlaps(uint64_t offset, uint64_t bytes, const MemoryRange *range)
{
    return bytes && (offset >= range->sb ? offset - range->sb < range->len :
                                          range->sb - offset < bytes);
}

bool fdm_kv_range_overlaps(const FDMKV_Binding_t *binding,
                           uint64_t offset, uint64_t bytes)
{
    if (binding) {
        for (uint32_t index = 0; index < binding->mrs->numr; index++) {
            if (overlaps(offset, bytes, &binding->mrs->mr[index])) {
                return true;
            }
        }
    }
    return false;
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
    if (binding->slot_tasks) {
        if (binding->aggregate_stop) {
            return NVME_CMD_SEQ_ERROR | NVME_DNR;
        }
        *input = true;
        return NVME_SUCCESS;
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
