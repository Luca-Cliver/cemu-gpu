#include "qemu/osdep.h"
#include "fdm_kv_writer.h"
#include "fdm_kv_receive.h"
#include "csd/memory.h"
#include "bbssd/bb.h"
#include "backend/backend.h"

typedef struct FDMKV_Completion {
    NvmeRequest request;
    uint32_t index;
} FDMKV_Completion_t;

struct FDMKV_Writer {
    FDMKV_Binding_t *binding;
    GCond wake;
    GThread *thread;
    bool stop;
    bool registered;
    CemuKvIdentity identity;
    CemuKvGeometry geometry;
    CemuKvDevicePage *pages;
    FDMKV_Completion_t *completions;
};

static GMutex *writer_lock(FDMKV_Binding_t *binding)
{
    MemoryNamespace *memory = binding->fdm_ns->private;
    return &memory->fdm_kv_lock;
}

static void mark_metadata(FDMKV_Binding_t *binding)
{
    backend_cuda_mark_host_dirty(binding->fdm_ns->backend, binding->buffer,
                                 binding->buffer->meta_bytes);
}

static uint32_t group_capacity(const FDMKV_Buffer_t *buffer, uint32_t kind)
{
    return kind == FDM_KV_CHANNEL_K ? buffer->layout.channel_group_tokens :
                                     buffer->layout.token_group_tokens;
}

static uint32_t hash_slot(const FDMKV_Buffer_t *buffer, uint64_t key)
{
    uint64_t hash = key * UINT64_C(0x9e3779b97f4a7c15);
    return (hash ^ (hash >> 32)) & (buffer->hash_size - 1);
}

static uint64_t page_key(const FDMKV_Buffer_t *buffer,
                         const FDMKV_Coordinate_t *coordinate)
{
    const FDMKV_Layout_t *layout = &buffer->layout;
    const FDMKV_Geometry_t *geometry = &layout->geometry;
    uint64_t head = ((uint64_t)coordinate->batch * geometry->layers + coordinate->layer) *
                    geometry->heads + coordinate->head;
    uint64_t token_pages = DIV_ROUND_UP(geometry->max_tokens, layout->token_group_tokens);
    uint64_t base = head * layout->pages_per_head;
    if (coordinate->kind == FDM_KV_CHANNEL_K) {
        return base + token_pages + coordinate->token / layout->channel_group_tokens *
               layout->channel_group_pages + coordinate->channel / geometry->channels_per_page;
    }
    return base + (coordinate->kind == FDM_KV_TOKEN_V ?
                   layout->pages_per_head - token_pages : 0) +
           coordinate->token / layout->token_group_tokens;
}

static bool staged_contains(const FDMKV_Buffer_t *buffer, uint64_t key)
{
    for (uint32_t index = buffer->page_hash[hash_slot(buffer, key)]; index;
         index = buffer->staged[index].hash_next) {
        if (buffer->staged[index].key == key) {
            return true;
        }
    }
    return false;
}

static void advance_committed(FDMKV_Buffer_t *buffer,
                               const FDMKV_Coordinate_t *written)
{
    const FDMKV_Geometry_t *geometry = &buffer->layout.geometry;
    uint64_t head_index = ((uint64_t)written->batch * geometry->layers + written->layer) *
                          geometry->heads + written->head;
    FDMKV_HeadState_t *head = &buffer->head_states[head_index];
    uint32_t kind = written->kind;
    uint32_t stripes = kind == FDM_KV_CHANNEL_K ? buffer->layout.channel_group_pages : 1;
    FDMKV_Coordinate_t coordinate = *written;

    while (head->committed[kind] < head->accepted) {
        coordinate.token = head->committed[kind];
        for (uint32_t stripe = 0; stripe < stripes; stripe++) {
            coordinate.channel = stripe * geometry->channels_per_page;
            if (staged_contains(buffer, page_key(buffer, &coordinate))) {
                return;
            }
        }
        head->committed[kind] += MIN((uint64_t)group_capacity(buffer, kind),
                                    head->accepted - head->committed[kind]);
    }
}

static void complete_writeback(NvmeRequest *request, bool cancelled)
{
    FDMKV_Completion_t *completion = container_of(request, FDMKV_Completion_t, request);
    FDMKV_Writer_t *writer = request->internal_owner;
    FDMKV_Binding_t *binding = writer->binding;
    FDMKV_Buffer_t *buffer = binding->buffer;
    FDMKV_Writeback_t *writeback = &buffer->writebacks[completion->index];

    g_mutex_lock(writer_lock(binding));
    assert(writeback->active && buffer->stats.inflight_blocks);
    if (cancelled) {
        fdm_kv_receive_fail(buffer);
    }
    if (!writer->stop && !buffer->failed) {
        for (uint32_t index = writeback->first; index; index = buffer->staged[index].next) {
            FDMKV_StagedPage_t *page = &buffer->staged[index];
            uint32_t *link = &buffer->page_hash[hash_slot(buffer, page->key)];
            while (*link != index) {
                assert(*link);
                link = &buffer->staged[*link].hash_next;
            }
            *link = page->hash_next;
        }
        uint32_t index = writeback->first;
        while (index) {
            uint32_t next = buffer->staged[index].next;
            advance_committed(buffer, &buffer->staged[index].coordinate);
            buffer->staged[index].next = buffer->free_page;
            buffer->free_page = index;
            buffer->free_pages++;
            index = next;
        }
        buffer->stats.staged_pages -= writeback->count;
        if (writeback->count == buffer->block_pages) {
            buffer->stats.write_blocks++;
        } else {
            buffer->stats.partial_blocks++;
        }
    }
    writeback->active = 0;
    writeback->next = buffer->free_writeback;
    buffer->free_writeback = completion->index;
    buffer->stats.inflight_blocks--;
    buffer->stats.pending_ready_ns = 0;
    for (uint32_t index = 1; index <= buffer->writeback_capacity; index++) {
        if (buffer->writebacks[index].active) {
            buffer->stats.pending_ready_ns = MAX(buffer->stats.pending_ready_ns,
                                                  buffer->writebacks[index].deadline_ns);
        }
    }
    if (buffer->draining && !buffer->stats.staged_pages && !buffer->failed) {
        buffer->sealed = 1;
    }
    mark_metadata(binding);
    g_cond_broadcast(&writer->wake);
    fdm_kv_async_notify_locked(binding);
    g_mutex_unlock(writer_lock(binding));
}

static FDMKV_BlockQueue_t *ready_block(FDMKV_Buffer_t *buffer)
{
    for (uint32_t domain = 0; domain < buffer->domains; domain++) {
        if (buffer->blocks[domain].count >= buffer->block_pages ||
            (buffer->draining && buffer->blocks[domain].count)) {
            return &buffer->blocks[domain];
        }
    }
    return NULL;
}

static void *buffer_worker(void *opaque)
{
    FDMKV_Writer_t *writer = opaque;
    FDMKV_Binding_t *binding = writer->binding;
    FDMKV_Buffer_t *buffer = binding->buffer;
    GMutex *lock = writer_lock(binding);

    g_mutex_lock(lock);
    while (!writer->stop) {
        FDMKV_BlockQueue_t *block = ready_block(buffer);
        if (buffer->failed || !block || !buffer->free_writeback) {
            g_cond_wait(&writer->wake, lock);
            continue;
        }
        uint32_t slot = buffer->free_writeback;
        FDMKV_Writeback_t *writeback = &buffer->writebacks[slot];
        FDMKV_Completion_t *completion = &writer->completions[slot];
        buffer->free_writeback = writeback->next;
        *writeback = (FDMKV_Writeback_t){ .first = block->first,
            .count = MIN(block->count, buffer->block_pages), .active = 1 };
        *completion = (FDMKV_Completion_t){ .index = slot };
        completion->request.internal_owner = writer;
        completion->request.internal_complete = complete_writeback;
        uint32_t last = writeback->first;
        for (uint32_t count = 1; count < writeback->count; count++) {
            last = buffer->staged[last].next;
        }
        block->first = buffer->staged[last].next;
        buffer->staged[last].next = 0;
        block->count -= writeback->count;
        if (!block->count) {
            block->last = 0;
        }
        buffer->stats.inflight_blocks++;
        buffer->stats.peak_inflight_blocks = MAX(buffer->stats.peak_inflight_blocks,
                                                buffer->stats.inflight_blocks);
        uint64_t start_ns = clock_ns();
        uint32_t page_index = writeback->first;
        CemuKvStatus status = CEMU_KV_OK;
        if (!writer->registered) {
            g_mutex_unlock(lock);
            status = bb_native_kv_register(binding->nvm_ns, &writer->identity, &writer->geometry);
            g_mutex_lock(lock);
            writer->registered = status == CEMU_KV_OK;
            if (status != CEMU_KV_OK) {
                femu_err("FDM KV writer register failed: task=%" PRIu64 ", status=%u\n",
                         writer->identity.task, status);
                fdm_kv_receive_fail(buffer);
            }
        }
        while (page_index && !writer->stop && !buffer->failed) {
            uint32_t count = 0;
            while (page_index && count < buffer->batch_pages) {
                FDMKV_StagedPage_t *page = &buffer->staged[page_index];
                const FDMKV_Coordinate_t *coordinate = &page->coordinate;
                writer->pages[count++] = (CemuKvDevicePage){
                    .coordinate = { .batch = coordinate->batch, .layer = coordinate->layer,
                        .head = coordinate->head, .kind = coordinate->kind,
                        .token = coordinate->token, .channel = coordinate->channel },
                    .valid_tokens = page->valid_tokens,
                    .buffer = fdm_kv_page_data(buffer, page_index),
                    .bytes = buffer->layout.geometry.page_bytes };
                page_index = page->next;
            }
            CemuKvDeviceBatchResult result = {0};
            g_mutex_unlock(lock);
            status = bb_native_kv_batch(binding->nvm_ns, &writer->identity,
                                        writer->pages, count, true, start_ns, &result);
            g_mutex_lock(lock);
            buffer->stats.write_bytes += result.transferred_bytes;
            buffer->stats.write_batches++;
            writeback->deadline_ns = MAX(writeback->deadline_ns, result.ready_ns);
            buffer->stats.pending_ready_ns = MAX(buffer->stats.pending_ready_ns,
                                                 writeback->deadline_ns);
            if (status != CEMU_KV_OK || result.completed_pages != count) {
                femu_err("FDM KV writer batch failed: task=%" PRIu64
                         ", status=%u, completed=%u/%u\n", writer->identity.task,
                         status, result.completed_pages, count);
                fdm_kv_receive_fail(buffer);
            }
        }
        completion->request.stat.expire_time = writeback->deadline_ns;
        bool failed = writer->stop || buffer->failed;
        mark_metadata(binding);
        g_mutex_unlock(lock);
        if (failed || !nvme_submit_internal_completion(binding->nvm_ns->ctrl,
                                                       &completion->request)) {
            complete_writeback(&completion->request, true);
        }
        g_mutex_lock(lock);
    }
    g_mutex_unlock(lock);
    return NULL;
}

bool fdm_kv_writer_create(FDMKV_Binding_t *binding)
{
    FDMKV_Buffer_t *buffer = binding->buffer;
    const FDMKV_Geometry_t *geometry = &buffer->layout.geometry;
    if (!binding->nvm_ns->ctrl->native_kv ||
        buffer->batch_pages > CEMU_NATIVE_INTERNAL_MAX_PAGES) {
        return false;
    }
    FDMKV_Writer_t *writer = g_try_new0(FDMKV_Writer_t, 1);
    if (!writer) {
        return false;
    }
    writer->binding = binding;
    writer->identity = (CemuKvIdentity){ .controller = buffer->identity.controller,
        .namespace_id = buffer->identity.namespace_id, .task = buffer->identity.session_id,
        .generation = buffer->identity.generation };
    writer->geometry = (CemuKvGeometry){ .batches = geometry->batches,
        .layers = geometry->layers, .heads = geometry->heads, .head_dim = geometry->head_dim,
        .max_tokens = geometry->max_tokens, .channels_per_page = geometry->channels_per_page,
        .element_bytes = geometry->element_bytes, .page_bytes = geometry->page_bytes };
    writer->pages = g_try_new0(CemuKvDevicePage, buffer->batch_pages);
    writer->completions = g_try_new0(FDMKV_Completion_t, (size_t)buffer->writeback_capacity + 1);
    g_cond_init(&writer->wake);
    binding->writer = writer;
    if (writer->pages && writer->completions) {
        GError *error = NULL;
        writer->thread = g_thread_try_new("fdm-kv-writer", buffer_worker, writer, &error);
        g_clear_error(&error);
    }
    if (!writer->thread) {
        fdm_kv_writer_free(binding);
        return false;
    }
    return true;
}

void fdm_kv_writer_wake(FDMKV_Binding_t *binding)
{
    if (binding->writer) {
        g_cond_signal(&binding->writer->wake);
    }
}

bool fdm_kv_writer_can_progress(const FDMKV_Binding_t *binding)
{
    return binding->writer && !binding->writer->stop && !binding->buffer->failed &&
           (binding->buffer->stats.inflight_blocks || ready_block(binding->buffer));
}

bool fdm_kv_writer_drain(FDMKV_Binding_t *binding)
{
    FDMKV_Buffer_t *buffer = binding->buffer;
    if (!binding->writer || binding->writer->stop || buffer->failed || buffer->leases) {
        return false;
    }
    if (!buffer->draining) {
        for (uint32_t slot = 0; slot < buffer->hash_size; slot++) {
            for (uint32_t index = buffer->page_hash[slot]; index;
                 index = buffer->staged[index].hash_next) {
                FDMKV_StagedPage_t *page = &buffer->staged[index];
                if (page->ready) {
                    continue;
                }
                const FDMKV_Coordinate_t *coordinate = &page->coordinate;
                uint32_t domain = ((coordinate->batch * buffer->layout.geometry.layers +
                    coordinate->layer) * CEMU_FDM_KV_KIND_COUNT + coordinate->kind) *
                    buffer->nand_channels +
                    (coordinate->token / group_capacity(buffer, coordinate->kind)) % buffer->nand_channels;
                FDMKV_BlockQueue_t *block = &buffer->blocks[domain];
                page->ready = 1;
                page->next = 0;
                if (block->last) {
                    buffer->staged[block->last].next = index;
                } else {
                    block->first = index;
                }
                block->last = index;
                block->count++;
            }
        }
        buffer->draining = 1;
        buffer->sealed = !buffer->stats.staged_pages;
        mark_metadata(binding);
    }
    fdm_kv_writer_wake(binding);
    return true;
}

void fdm_kv_writer_request_stop(FDMKV_Binding_t *binding)
{
    if (binding->writer) {
        binding->writer->stop = true;
        if (!binding->buffer->sealed) {
            fdm_kv_receive_fail(binding->buffer);
        }
        mark_metadata(binding);
        fdm_kv_writer_wake(binding);
    }
}

void fdm_kv_writer_join(FDMKV_Binding_t *binding)
{
    if (binding->writer && binding->writer->thread) {
        g_thread_join(binding->writer->thread);
        binding->writer->thread = NULL;
    }
}

void fdm_kv_writer_free(FDMKV_Binding_t *binding)
{
    FDMKV_Writer_t *writer = binding->writer;
    if (writer) {
        assert(!writer->thread && !binding->buffer->stats.inflight_blocks);
        g_cond_clear(&writer->wake);
        g_free(writer->pages);
        g_free(writer->completions);
        g_free(writer);
        binding->writer = NULL;
    }
}

void fdm_kv_writers_stop(FemuCtrl *ctrl)
{
    for (GList *entry = ctrl->fdm_kv_bindings; entry; entry = entry->next) {
        FDMKV_Binding_t *binding = entry->data;
        g_mutex_lock(writer_lock(binding));
        fdm_kv_writer_request_stop(binding);
        g_mutex_unlock(writer_lock(binding));
    }
    for (GList *entry = ctrl->fdm_kv_bindings; entry; entry = entry->next) {
        FDMKV_Binding_t *binding = entry->data;
        fdm_kv_writer_join(binding);
        if (binding->writer) {
            nvme_cancel_internal_completions(ctrl, binding->writer);
        }
    }
}
