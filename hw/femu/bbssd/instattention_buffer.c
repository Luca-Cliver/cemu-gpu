#include "qemu/osdep.h"
#include "bb.h"
#include "instattention_buffer.h"

typedef struct NativeKvHeadState {
    uint64_t accepted;
    uint64_t committed[3];
    uint32_t readers;
} NativeKvHeadState;

/* Each entry is exactly one NAND page, including a channel-indexed group. */
typedef struct NativeKvStagedPage {
    uint64_t key;
    CemuKvCoordinate coordinate;
    uint32_t valid_tokens;
    uint32_t next;
    uint32_t hash_next;
    bool ready; /* Full or terminally sealed; queued/inflight pages are immutable. */
} NativeKvStagedPage;

typedef struct NativeKvBlockQueue {
    uint32_t first, last, count;
} NativeKvBlockQueue;

typedef struct NativeKvWriteback {
    NvmeRequest completion;
    uint32_t first, count, next;
    bool active;
} NativeKvWriteback;

struct NativeKvBuffer {
    NvmeNamespace *ns;
    CemuKvIdentity identity;
    CemuKvLayout layout;
    GMutex lock;
    GCond wake;
    GThread *worker;
    NativeKvHeadState *head_states;
    uint32_t append_max_tokens;
    uint32_t row_bytes;
    uint32_t batch_pages;
    uint64_t memory_budget;
    uint8_t *staged_data;
    NativeKvStagedPage *staged;
    NativeKvBlockQueue *blocks;
    uint32_t *page_hash;
    uint32_t *inflight;
    NativeKvWriteback *writebacks;
    uint32_t writeback_capacity, free_writeback;
    uint32_t pool_pages, free_page, free_pages, hash_size;
    uint32_t domains, nand_channels, block_pages;
    uint8_t *read_page;
    uint8_t *gather_data;
    CemuKvDevicePage *gather_pages;
    CemuKvPageKey *gather_keys;
    CemuKvDevicePage *pages;

    NativeKvBufferStats stats;
    bool draining;
    bool stop;
    uint32_t leases;
};

static uint32_t group_capacity(const NativeKvBuffer *buffer, uint32_t kind)
{
    return kind == CEMU_KV_PAGE_CHANNEL_K ? buffer->layout.channel_group_tokens :
                                           buffer->layout.token_group_tokens;
}

static bool kv_head_index(const NativeKvBuffer *buffer, const CemuKvCoordinate *coordinate,
                          uint32_t *head_index)
{
    CemuKvLocation location;

    if (!cemu_kv_resolve(&buffer->layout, &buffer->identity, coordinate, &location)) {
        return false;
    }
    *head_index = ((uint64_t)coordinate->batch * buffer->layout.geometry.layers +
              coordinate->layer) * buffer->layout.geometry.heads + coordinate->head;
    return true;
}

static CemuKvCoordinate kv_head_coordinate(const NativeKvBuffer *buffer, uint32_t head_index)
{
    const CemuKvGeometry *geometry = &buffer->layout.geometry;

    return (CemuKvCoordinate){ .head = head_index % geometry->heads,
        .layer = head_index / geometry->heads % geometry->layers,
        .batch = head_index / geometry->heads / geometry->layers };
}

static uint64_t gather_page_key(const NativeKvBuffer *buffer,
    uint32_t head_index, const CemuKvLocation *location);

static uint32_t page_hash_slot(const NativeKvBuffer *buffer, uint64_t key)
{
    uint64_t hash = key * UINT64_C(0x9e3779b97f4a7c15);
    return (hash ^ (hash >> 32)) & (buffer->hash_size - 1);
}

static uint32_t staged_page(NativeKvBuffer *buffer, uint64_t key)
{
    for (uint32_t p = buffer->page_hash[page_hash_slot(buffer, key)]; p;
         p = buffer->staged[p].hash_next) {
        if (buffer->staged[p].key == key) {
            return p;
        }
    }
    return 0;
}

/* NULL means NAND. Partial/full pages stay readable in DRAM through their deadline.
 * A committed prefix alone cannot classify a page: different NAND channels may
 * flush out of logical token order. Caller holds buffer->lock. */
static uint8_t *dram_element(NativeKvBuffer *buffer, uint32_t head_index,
                            const CemuKvCoordinate *coordinate)
{
    CemuKvLocation location;
    uint32_t page;

    if (!cemu_kv_resolve(&buffer->layout, &buffer->identity, coordinate, &location)) {
        g_assert_not_reached();
    }
    page = staged_page(buffer, gather_page_key(buffer, head_index, &location));
    return page ? buffer->staged_data + (uint64_t)(page - 1) *
                  buffer->layout.geometry.page_bytes + location.byte_offset : NULL;
}

/* Same allocation domain as cemu_kv_map_reserve(): heads share a block, but
 * batch/layer/layout/NAND-channel do not. Embedding channel is NOT NAND channel. */
static uint32_t block_domain(const NativeKvBuffer *buffer,
                             const CemuKvCoordinate *coordinate)
{
    return (((coordinate->batch * buffer->layout.geometry.layers + coordinate->layer) * 3 +
             coordinate->kind) * buffer->nand_channels) +
           (coordinate->token / group_capacity(buffer, coordinate->kind)) % buffer->nand_channels;
}

/* Publish only a complete page (or a sealed partial page during drain).
 * The data remains in the same slot until the block's target deadline. */
static void enqueue_page(NativeKvBuffer *buffer, uint32_t index)
{
    NativeKvStagedPage *entry = &buffer->staged[index];
    NativeKvBlockQueue *block = &buffer->blocks[block_domain(buffer, &entry->coordinate)];

    g_assert(!entry->ready);
    entry->ready = true;
    entry->next = 0;
    if (block->last) {
        buffer->staged[block->last].next = index;
    } else {
        block->first = index;
    }
    block->last = index;
    block->count++;
}

static uint32_t allocate_page(NativeKvBuffer *buffer, const CemuKvCoordinate *coordinate,
                              uint64_t key)
{
    uint32_t index = buffer->free_page;
    uint32_t slot = page_hash_slot(buffer, key);
    NativeKvStagedPage *entry;

    /* APPEND reserves enough slots for the WHOLE command before any mutation. */
    g_assert(index && buffer->free_pages);
    entry = &buffer->staged[index];
    buffer->free_page = entry->next;
    buffer->free_pages--;
    *entry = (NativeKvStagedPage){ .coordinate = *coordinate, .key = key,
                                  .hash_next = buffer->page_hash[slot] };
    buffer->page_hash[slot] = index;
    memset(buffer->staged_data + (uint64_t)(index - 1) * buffer->layout.geometry.page_bytes,
           0, buffer->layout.geometry.page_bytes);
    buffer->stats.staged_pages++;
    return index;
}

/* Visit all page fragments touched by one head/layout. The first pass counts
 * new slots only; the second pass writes directly into final-layout pages.
 * No allocation or validation may fail between these passes under the lock. */
static uint64_t append_layout(NativeKvBuffer *buffer, uint32_t head_index,
    uint32_t kind, uint64_t token, uint32_t count, const uint8_t *input)
{
    const CemuKvGeometry *geometry = &buffer->layout.geometry;
    uint32_t capacity = group_capacity(buffer, kind);
    uint32_t stripes = kind == CEMU_KV_PAGE_CHANNEL_K ? buffer->layout.channel_group_pages : 1;
    uint64_t needed = 0;

    while (count) {
        uint32_t offset = token % capacity;
        uint32_t take = MIN(count, capacity - offset);
        CemuKvCoordinate coordinate = kv_head_coordinate(buffer, head_index);

        coordinate.kind = kind;
        coordinate.token = token - offset;
        for (uint32_t stripe = 0; stripe < stripes; stripe++) {
            CemuKvLocation location;
            uint64_t key;
            uint32_t index;
            NativeKvStagedPage *entry;
            uint8_t *destination;

            coordinate.channel = stripe * geometry->channels_per_page;
            if (!cemu_kv_resolve(&buffer->layout, &buffer->identity, &coordinate, &location)) {
                g_assert_not_reached();
            }
            key = gather_page_key(buffer, head_index, &location);
            index = staged_page(buffer, key);
            if (!input) {
                needed += !index;
                continue;
            }
            if (!index) {
                index = allocate_page(buffer, &coordinate, key);
            }
            entry = &buffer->staged[index];
            g_assert(!entry->ready && entry->valid_tokens == offset);
            destination = buffer->staged_data + (uint64_t)(index - 1) * geometry->page_bytes;
            if (kind != CEMU_KV_PAGE_CHANNEL_K) {
                memcpy(destination + (uint64_t)offset * buffer->row_bytes, input,
                       (size_t)take * buffer->row_bytes);
            } else {
                for (uint32_t channel = 0; channel < geometry->channels_per_page; channel++) {
                    for (uint32_t t = 0; t < take; t++) {
                        memcpy(destination + ((uint64_t)channel * capacity + offset + t) * geometry->element_bytes,
                            input + (uint64_t)t * buffer->row_bytes +
                                    (coordinate.channel + channel) * geometry->element_bytes,
                            geometry->element_bytes);
                    }
                }
            }
            entry->valid_tokens += take;
            if (entry->valid_tokens == capacity) {
                enqueue_page(buffer, index);
            }
        }
        token += take;
        count -= take;
        if (input) {
            input += (size_t)take * buffer->row_bytes;
        }
    }
    return needed;
}

static void advance_committed(NativeKvBuffer *buffer, const CemuKvCoordinate *written)
{
    uint32_t head_index;
    uint32_t kind = written->kind, capacity = group_capacity(buffer, kind);
    uint32_t stripes = kind == CEMU_KV_PAGE_CHANNEL_K ? buffer->layout.channel_group_pages : 1;
    NativeKvHeadState *head;
    CemuKvCoordinate coordinate = *written;

    if (!kv_head_index(buffer, written, &head_index)) {
        g_assert_not_reached();
    }
    head = &buffer->head_states[head_index];
    while (head->committed[kind] < head->accepted) {
        coordinate.token = head->committed[kind];
        for (uint32_t stripe = 0; stripe < stripes; stripe++) {
            coordinate.channel = stripe * buffer->layout.geometry.channels_per_page;
            if (dram_element(buffer, head_index, &coordinate)) {
                return;
            }
        }
        head->committed[kind] += MIN((uint64_t)capacity,
                                    head->accepted - head->committed[kind]);
    }
}

static void complete_writeback(NvmeRequest *request, bool cancelled)
{
    NativeKvWriteback *writeback = container_of(request, NativeKvWriteback, completion);
    NativeKvBuffer *buffer = request->internal_owner;
    uint32_t page;

    g_mutex_lock(&buffer->lock);
    g_assert(writeback->active && buffer->stats.inflight_blocks);
    if (cancelled) {
        buffer->stats.failed = true;
    }
    if (!buffer->stop && !buffer->stats.failed) {
        for (page = writeback->first; page; page = buffer->staged[page].next) {
            NativeKvStagedPage *entry = &buffer->staged[page];
            uint32_t *link = &buffer->page_hash[page_hash_slot(buffer, entry->key)];
            while (*link != page) {
                link = &buffer->staged[*link].hash_next;
            }
            *link = entry->hash_next;
        }
        page = writeback->first;
        while (page) {
            uint32_t next = buffer->staged[page].next;
            advance_committed(buffer, &buffer->staged[page].coordinate);
            buffer->staged[page].next = buffer->free_page;
            buffer->free_page = page;
            buffer->free_pages++;
            page = next;
        }
        buffer->stats.staged_pages -= writeback->count;
        if (writeback->count == buffer->block_pages) {
            buffer->stats.write_blocks++;
        } else {
            buffer->stats.partial_blocks++;
        }
    }
    writeback->active = false;
    writeback->next = buffer->free_writeback;
    buffer->free_writeback = writeback - buffer->writebacks;
    buffer->stats.inflight_blocks--;
    if (buffer->stats.pending_ready_ns == request->stat.expire_time) {
        buffer->stats.pending_ready_ns = 0;
        for (uint32_t index = 1; index <= buffer->writeback_capacity; index++) {
            if (buffer->writebacks[index].active) {
                buffer->stats.pending_ready_ns = MAX(buffer->stats.pending_ready_ns,
                    buffer->writebacks[index].completion.stat.expire_time);
            }
        }
    }
    g_cond_broadcast(&buffer->wake);
    g_mutex_unlock(&buffer->lock);
}

static void *buffer_worker(void *opaque)
{
    NativeKvBuffer *buffer = opaque;

    g_mutex_lock(&buffer->lock);
    while (!buffer->stop) {
        NativeKvBlockQueue *block = NULL;
        NativeKvWriteback *writeback;
        uint64_t deadline = 0, start_ns;
        uint32_t count;

        if (buffer->stats.failed) {
            g_cond_wait(&buffer->wake, &buffer->lock);
            continue;
        }
        for (uint32_t domain = 0; domain < buffer->domains; domain++) {
            if (buffer->blocks[domain].count >= buffer->block_pages) {
                block = &buffer->blocks[domain];
                break;
            }
        }
        /* Terminal drain is the only partial-block/partial-page exception. */
        if (!block && buffer->draining) {
            for (uint32_t domain = 0; domain < buffer->domains; domain++) {
                if (buffer->blocks[domain].count) {
                    block = &buffer->blocks[domain];
                    break;
                }
            }
        }
        if (!block) {
            buffer->stats.sealed = buffer->draining && !buffer->stats.inflight_blocks;
            g_cond_wait(&buffer->wake, &buffer->lock);
            continue;
        }
        if (!buffer->free_writeback) {
            g_cond_wait(&buffer->wake, &buffer->lock);
            continue;
        }
        writeback = &buffer->writebacks[buffer->free_writeback];
        buffer->free_writeback = writeback->next;
        *writeback = (NativeKvWriteback){ .first = block->first };
        count = MIN(block->count, buffer->block_pages);
        writeback->count = count;
        writeback->active = true;
        writeback->completion.internal_owner = buffer;
        writeback->completion.internal_complete = complete_writeback;
        buffer->stats.inflight_blocks++;
        buffer->stats.peak_inflight_blocks = MAX(buffer->stats.peak_inflight_blocks,
                                                buffer->stats.inflight_blocks);
        for (uint32_t i = 0; i < count; i++) {
            buffer->inflight[i] = block->first;
            block->first = buffer->staged[block->first].next;
        }
        buffer->staged[buffer->inflight[count - 1]].next = 0;
        block->count -= count;
        if (!block->count) {
            block->last = 0;
        }
        start_ns = clock_ns();
        for (uint32_t first = 0; first < count; first += buffer->batch_pages) {
            CemuKvDeviceBatchResult result = {0};
            CemuKvStatus status;
            uint32_t chunk = MIN(buffer->batch_pages, count - first);

            for (uint32_t i = 0; i < chunk; i++) {
                uint32_t index = buffer->inflight[first + i];
                NativeKvStagedPage *entry = &buffer->staged[index];
                buffer->pages[i] = (CemuKvDevicePage){ .coordinate = entry->coordinate,
                    .valid_tokens = entry->valid_tokens,
                    .buffer = buffer->staged_data + (uint64_t)(index - 1) * buffer->layout.geometry.page_bytes,
                    .bytes = buffer->layout.geometry.page_bytes };
            }
            g_mutex_unlock(&buffer->lock);
            status = bb_native_kv_batch(buffer->ns, &buffer->identity, buffer->pages,
                                        chunk, true, start_ns, &result);
            g_mutex_lock(&buffer->lock);
            buffer->stats.write_bytes += result.transferred_bytes;
            buffer->stats.write_batches++;
            deadline = MAX(deadline, result.ready_ns);
            writeback->completion.stat.expire_time = deadline;
            buffer->stats.pending_ready_ns = MAX(buffer->stats.pending_ready_ns, deadline);
            if (status != CEMU_KV_OK || result.completed_pages != chunk) {
                buffer->stats.failed = true;
                break;
            }
            if (buffer->stop) {
                break;
            }
        }
        if (buffer->stats.failed || buffer->stop) {
            g_mutex_unlock(&buffer->lock);
            complete_writeback(&writeback->completion, true);
            g_mutex_lock(&buffer->lock);
            continue;
        }
        g_mutex_unlock(&buffer->lock);
        if (!nvme_submit_internal_completion(buffer->ns->ctrl, &writeback->completion)) {
            complete_writeback(&writeback->completion, true);
        }
        g_mutex_lock(&buffer->lock);
    }
    g_mutex_unlock(&buffer->lock);
    return NULL;
}

CemuKvStatus native_kv_buffer_create(NvmeNamespace *ns,
    const CemuKvIdentity *identity, const CemuKvGeometry *geometry,
    uint64_t budget, uint32_t batch_pages, NativeKvBuffer **result)
{
    NativeKvBuffer *buffer;
    CemuKvLayout layout;
    CemuKvStatus status;
    uint64_t required, pool_pages = 0, domains, partial_pages, command_pages;
    uint32_t append_max_tokens;
    uint32_t nand_channels, block_pages, writeback_capacity, hash_size = 1;
    GError *error = NULL;

    if (!ns || !identity || !result || !cemu_kv_layout_init(&layout, geometry) ||
        layout.streams > 65536 || !batch_pages ||
        ns->csi != NVME_CSI_NVM || !ns->private || !nvm_ns(ns)->params ||
        batch_pages > CEMU_NATIVE_INTERNAL_MAX_PAGES) {
        return CEMU_KV_INVALID;
    }
    if (nvm_ns(ns)->params->nchs <= 0 || nvm_ns(ns)->params->pgs_per_blk <= 0) {
        return CEMU_KV_INVALID;
    }
    nand_channels = nvm_ns(ns)->params->nchs;
    block_pages = nvm_ns(ns)->params->pgs_per_blk;
    domains = (uint64_t)geometry->batches * geometry->layers * 3 * nand_channels;
    if (!nand_channels || !block_pages || domains > UINT32_MAX) {
        return CEMU_KV_INVALID;
    }
    /* Bound idle occupancy by domain residuals + per-head partial pages.
     * Keep room for the largest atomic APPEND, so capacity backpressure always
     * has a full/inflight block that can make progress. */
    append_max_tokens = 2 * layout.token_group_tokens;
    for (uint32_t kind = 0; kind < 3; kind++) {
        uint32_t capacity = kind == CEMU_KV_PAGE_CHANNEL_K ?
                            layout.channel_group_tokens : layout.token_group_tokens;
        uint32_t stripes = kind == CEMU_KV_PAGE_CHANNEL_K ? layout.channel_group_pages : 1;
        uint64_t spans = DIV_ROUND_UP(geometry->max_tokens, capacity);
        for (uint32_t channel = 0; channel < nand_channels; channel++) {
            uint64_t pages = (spans / nand_channels + (channel < spans % nand_channels)) *
                             geometry->heads * stripes;
            pool_pages += MIN(pages, (uint64_t)block_pages - 1);
        }
    }
    partial_pages = layout.streams * (2ULL + layout.channel_group_pages);
    command_pages = (uint64_t)geometry->heads *
        (4ULL + DIV_ROUND_UP(append_max_tokens, layout.channel_group_tokens) *
         (uint64_t)layout.channel_group_pages);
    pool_pages = pool_pages * geometry->batches * geometry->layers + partial_pages + command_pages;
    pool_pages = MIN(pool_pages, layout.total_pages);
    if (!pool_pages || pool_pages > UINT32_MAX / 4) {
        return CEMU_KV_LIMIT;
    }
    while (hash_size < pool_pages * 2) {
        hash_size *= 2;
    }
    writeback_capacity = MIN(pool_pages, pool_pages / block_pages + domains + 1);
    required = sizeof(*buffer) + layout.streams * sizeof(NativeKvHeadState) +
        (uint64_t)batch_pages * sizeof(CemuKvDevicePage) +
        (uint64_t)batch_pages * (geometry->page_bytes + sizeof(CemuKvDevicePage) + sizeof(CemuKvPageKey)) +
        geometry->page_bytes + domains * sizeof(NativeKvBlockQueue) +
        (pool_pages + 1) * sizeof(NativeKvStagedPage) + pool_pages * geometry->page_bytes +
        (uint64_t)hash_size * sizeof(uint32_t) + (uint64_t)block_pages * sizeof(uint32_t) +
        ((uint64_t)writeback_capacity + 1) * sizeof(NativeKvWriteback);
    if (required > budget || required > SIZE_MAX) {
        return CEMU_KV_LIMIT;
    }
    buffer = g_try_new0(NativeKvBuffer, 1);
    if (!buffer) {
        return CEMU_KV_NO_MEMORY;
    }
    g_mutex_init(&buffer->lock);
    g_cond_init(&buffer->wake);
    buffer->ns = ns;
    buffer->identity = *identity;
    buffer->layout = layout;
    buffer->batch_pages = batch_pages;
    buffer->memory_budget = budget;
    buffer->pool_pages = pool_pages;
    buffer->hash_size = hash_size;
    buffer->domains = domains;
    buffer->nand_channels = nand_channels;
    buffer->block_pages = block_pages;
    buffer->append_max_tokens = append_max_tokens;
    buffer->row_bytes = geometry->head_dim * geometry->element_bytes;
    buffer->stats.memory_bytes = required;
    buffer->head_states = g_try_new0(NativeKvHeadState, layout.streams);
    buffer->staged_data = g_try_malloc(pool_pages * geometry->page_bytes);
    buffer->staged = g_try_new0(NativeKvStagedPage, pool_pages + 1);
    buffer->blocks = g_try_new0(NativeKvBlockQueue, domains);
    buffer->page_hash = g_try_new0(uint32_t, hash_size);
    buffer->inflight = g_try_new(uint32_t, block_pages);
    buffer->writeback_capacity = writeback_capacity;
    buffer->writebacks = g_try_new0(NativeKvWriteback, (size_t)writeback_capacity + 1);
    buffer->read_page = g_try_malloc(geometry->page_bytes);
    buffer->gather_data = g_try_malloc((uint64_t)batch_pages * geometry->page_bytes);
    buffer->gather_pages = g_try_new(CemuKvDevicePage, batch_pages);
    buffer->gather_keys = g_try_new(CemuKvPageKey, batch_pages);
    buffer->pages = g_try_new(CemuKvDevicePage, batch_pages);
    if (!buffer->head_states || !buffer->staged_data || !buffer->staged || !buffer->blocks ||
        !buffer->page_hash || !buffer->inflight || !buffer->writebacks ||
        !buffer->read_page || !buffer->pages ||
        !buffer->gather_data || !buffer->gather_pages || !buffer->gather_keys) {
        native_kv_buffer_free(buffer);
        return CEMU_KV_NO_MEMORY;
    }
    for (uint32_t i = 1; i <= pool_pages; i++) {
        buffer->staged[i].next = i < pool_pages ? i + 1 : 0;
    }
    buffer->free_page = 1;
    buffer->free_pages = pool_pages;
    for (uint32_t index = 1; index <= writeback_capacity; index++) {
        buffer->writebacks[index].next = index < writeback_capacity ? index + 1 : 0;
    }
    buffer->free_writeback = 1;
    g_mutex_lock(&buffer->lock);
    buffer->worker = g_thread_try_new("native-kv-flush", buffer_worker, buffer, &error);
    status = buffer->worker ? bb_native_kv_register(ns, identity, geometry) : CEMU_KV_NO_MEMORY;
    if (error) {
        g_error_free(error);
    }
    if (status != CEMU_KV_OK) {
        buffer->stop = true;
    }
    g_mutex_unlock(&buffer->lock);
    if (status != CEMU_KV_OK) {
        native_kv_buffer_free(buffer);
        return status;
    }
    *result = buffer;
    return CEMU_KV_OK;
}

bool native_kv_buffer_matches(const NativeKvBuffer *buffer, const CemuKvIdentity *identity)
{
    return buffer && identity && buffer->identity.controller == identity->controller &&
        buffer->identity.namespace_id == identity->namespace_id &&
        buffer->identity.task == identity->task && buffer->identity.generation == identity->generation;
}

size_t native_kv_buffer_row_bytes(const NativeKvBuffer *buffer, bool channel)
{
    return buffer ? (channel ? buffer->layout.geometry.element_bytes : buffer->row_bytes) : 0;
}

CemuKvStatus native_kv_buffer_append(NativeKvBuffer *buffer,
    const CemuKvCoordinate *coordinate, uint32_t count, const void *key,
    const void *value, size_t bytes)
{
    return native_kv_buffer_append_batch(buffer, coordinate, 1, count, key, value, bytes);
}

CemuKvStatus native_kv_buffer_append_batch(NativeKvBuffer *buffer,
    const CemuKvCoordinate *coordinate, uint32_t heads, uint32_t count,
    const void *key, const void *value, size_t bytes)
{
    uint32_t head_index;
    uint64_t needed = 0;
    CemuKvStatus status = CEMU_KV_INVALID;

    if (!buffer || !coordinate || !key || !value || !count || !heads || coordinate->kind || coordinate->channel ||
        !kv_head_index(buffer, coordinate, &head_index) || count > buffer->layout.geometry.max_tokens - coordinate->token ||
        heads > buffer->layout.geometry.heads - coordinate->head ||
        (uint64_t)heads * count * buffer->row_bytes != bytes || count > buffer->append_max_tokens) {
        return CEMU_KV_INVALID;
    }
    g_mutex_lock(&buffer->lock);
    if (buffer->stats.failed) {
        status = CEMU_KV_IO_FAILED;
        goto done;
    }
    if (buffer->stop || buffer->draining) {
        goto done;
    }
    for (uint32_t head = 0; head < heads; head++) {
        if (buffer->head_states[head_index + head].accepted != coordinate->token) {
            goto done;
        }
    }
    for (uint32_t head = 0; head < heads; head++) {
        NativeKvHeadState *head_state = &buffer->head_states[head_index + head];
        if (head_state->readers) {
            status = CEMU_KV_PENDING;
            goto done;
        }
        for (uint32_t kind = 0; kind < 3; kind++) {
            needed += append_layout(buffer, head_index + head, kind,
                                    coordinate->token, count, NULL);
        }
    }
    if (needed > buffer->free_pages) {
        status = CEMU_KV_PENDING;
        goto done;
    }
    for (uint32_t head = 0; head < heads; head++) {
        size_t offset = (size_t)head * count * buffer->row_bytes;
        for (uint32_t kind = 0; kind < 3; kind++) {
            const uint8_t *input = kind == CEMU_KV_PAGE_TOKEN_V ? value : key;
            append_layout(buffer, head_index + head, kind, coordinate->token,
                          count, input + offset);
        }
        buffer->head_states[head_index + head].accepted += count;
    }
    g_cond_signal(&buffer->wake);
    status = CEMU_KV_OK;
done:
    g_mutex_unlock(&buffer->lock);
    return status;
}

CemuKvStatus native_kv_buffer_read(NativeKvBuffer *buffer,
    const CemuKvCoordinate *coordinate, void *output, size_t bytes,
    uint64_t start_ns, CemuKvIoResult *result)
{
    uint32_t head_index;
    NativeKvHeadState *head_state;
    uint8_t *source;
    CemuKvStatus status = CEMU_KV_INVALID;
    CemuKvIoResult candidate = { .start_ns = start_ns, .ready_ns = start_ns };

    if (!buffer || !coordinate || !output || !result || start_ns > INT64_MAX ||
        !kv_head_index(buffer, coordinate, &head_index) ||
        (coordinate->kind == CEMU_KV_PAGE_CHANNEL_K ? bytes != buffer->layout.geometry.element_bytes :
            (bytes != buffer->row_bytes || coordinate->channel))) {
        return CEMU_KV_INVALID;
    }
    g_mutex_lock(&buffer->lock);
    head_state = &buffer->head_states[head_index];
    if (buffer->stats.failed) {
        status = CEMU_KV_IO_FAILED;
    } else if (coordinate->token >= head_state->accepted) {
        status = CEMU_KV_NOT_FOUND;
    } else if ((source = dram_element(buffer, head_index, coordinate))) {
        memcpy(output, source, bytes);
        status = CEMU_KV_OK;
    } else {
        status = bb_native_kv_transfer(buffer->ns, &buffer->identity, coordinate, 0,
            buffer->read_page, buffer->layout.geometry.page_bytes, false, start_ns, &candidate);
        if (status == CEMU_KV_OK) {
            memcpy(output, buffer->read_page + candidate.element_offset, bytes);
            buffer->stats.read_bytes += candidate.transferred_bytes;
        }
    }
    *result = candidate;
    g_mutex_unlock(&buffer->lock);
    return status;
}

static uint32_t gather_page_index(NativeKvBuffer *buffer,
                                  const CemuKvPageKey *key, uint32_t pages)
{
    for (uint32_t index = 0; index < pages; index++) {
        if (cemu_kv_page_key_equal(key, &buffer->gather_keys[index])) {
            return index;
        }
    }
    return pages;
}

static CemuKvStatus gather_ranges(NativeKvBuffer *buffer,
    const NativeKvGatherRange *ranges, uint32_t count, uint8_t *output,
    NativeKvGatherResult *result)
{
    const CemuKvGeometry *geometry = &buffer->layout.geometry;

    for (uint32_t range = 0; range < count; range++) {
        CemuKvCoordinate coordinate = ranges[range].coordinate;
        uint32_t head_index = ((uint64_t)coordinate.batch * geometry->layers + coordinate.layer) *
                          geometry->heads + coordinate.head;
        uint32_t unit = coordinate.kind == CEMU_KV_PAGE_CHANNEL_K ?
                        geometry->element_bytes : buffer->row_bytes;
        uint32_t remaining = ranges[range].tokens;

        while (remaining) {
            CemuKvLocation location;
            uint32_t capacity = group_capacity(buffer, coordinate.kind);
            uint32_t tokens = MIN(remaining, capacity - coordinate.token % capacity);
            uint8_t *dram = dram_element(buffer, head_index, &coordinate);

            if (!dram) {
                uint32_t page;

                if (!cemu_kv_resolve(&buffer->layout, &buffer->identity, &coordinate, &location)) {
                    return CEMU_KV_INVALID;
                }
                page = gather_page_index(buffer, &location.key, result->flash_pages);
                if (!output && page == result->flash_pages) {
                    if (page == buffer->batch_pages) {
                        return CEMU_KV_LIMIT;
                    }
                    buffer->gather_keys[page] = location.key;
                    buffer->gather_pages[page] = (CemuKvDevicePage){
                        .coordinate = coordinate,
                        .buffer = buffer->gather_data + (uint64_t)page * geometry->page_bytes,
                        .bytes = geometry->page_bytes };
                    result->flash_pages++;
                }
                if (output) {
                    g_assert(page < result->flash_pages);
                    memcpy(output, buffer->gather_data + (uint64_t)page * geometry->page_bytes +
                           location.byte_offset, (size_t)tokens * unit);
                }
            } else if (output) {
                memcpy(output, dram, (size_t)tokens * unit);
            } else {
                result->tail_bytes += (uint64_t)tokens * unit;
            }
            if (output) {
                output += (size_t)tokens * unit;
            }
            coordinate.token += tokens;
            remaining -= tokens;
        }
    }
    return CEMU_KV_OK;
}

CemuKvStatus native_kv_buffer_gather(NativeKvBuffer *buffer,
    const NativeKvGatherRange *ranges, uint32_t count, void *output,
    size_t bytes, uint64_t start_ns, NativeKvGatherResult *result)
{
    NativeKvGatherResult candidate = { .ready_ns = start_ns };
    CemuKvDeviceBatchResult batch = { .ready_ns = start_ns };
    CemuKvStatus status = CEMU_KV_INVALID;
    uint64_t required = 0;

    if (!buffer || !ranges || !count || count > CEMU_NATIVE_GATHER_MAX_RANGES ||
        !output || !result || start_ns > INT64_MAX) {
        return CEMU_KV_INVALID;
    }
    *result = candidate;
    g_mutex_lock(&buffer->lock);
    if (buffer->stats.failed) {
        status = CEMU_KV_IO_FAILED;
        goto done;
    }
    for (uint32_t range = 0; range < count; range++) {
        const CemuKvCoordinate *coordinate = &ranges[range].coordinate;
        uint32_t head_index, unit;

        if (!ranges[range].tokens || !kv_head_index(buffer, coordinate, &head_index) ||
            ranges[range].tokens > buffer->layout.geometry.max_tokens - coordinate->token ||
            (coordinate->kind != CEMU_KV_PAGE_CHANNEL_K && coordinate->channel)) {
            goto done;
        }
        if (coordinate->token + ranges[range].tokens > buffer->head_states[head_index].accepted) {
            status = CEMU_KV_NOT_FOUND;
            goto done;
        }
        unit = coordinate->kind == CEMU_KV_PAGE_CHANNEL_K ?
               buffer->layout.geometry.element_bytes : buffer->row_bytes;
        required += (uint64_t)ranges[range].tokens * unit;
    }
    if (required != bytes) {
        goto done;
    }
    if (required > (uint64_t)buffer->batch_pages * buffer->layout.geometry.page_bytes) {
        status = CEMU_KV_LIMIT;
        goto done;
    }
    status = gather_ranges(buffer, ranges, count, NULL, &candidate);
    if (status != CEMU_KV_OK) {
        goto done;
    }
    if (candidate.flash_pages) {
        status = bb_native_kv_batch(buffer->ns, &buffer->identity, buffer->gather_pages,
            candidate.flash_pages, false, start_ns, &batch);
        candidate.ready_ns = batch.ready_ns;
        candidate.transferred_bytes = batch.transferred_bytes;
        buffer->stats.read_bytes += batch.transferred_bytes;
        if (status != CEMU_KV_OK) {
            candidate.flash_pages = batch.completed_pages;
            candidate.tail_bytes = 0;
            *result = candidate;
            goto done;
        }
    }
    status = gather_ranges(buffer, ranges, count, output, &candidate);
    candidate.output_bytes = required;
    *result = candidate;
done:
    g_mutex_unlock(&buffer->lock);
    return status;
}

typedef struct NativeKvReadFragment {
    CemuKvCoordinate coordinate;
    uint64_t page_key;
    uint64_t destination;
    uint32_t offset;
    uint32_t bytes;
    uint32_t next;
    uint32_t last;
    uint32_t next_page;
} NativeKvReadFragment;

static uint64_t gather_page_key(const NativeKvBuffer *buffer,
    uint32_t head_index, const CemuKvLocation *location)
{
    uint64_t token_pages = DIV_ROUND_UP(buffer->layout.geometry.max_tokens,
                                      buffer->layout.token_group_tokens);
    uint64_t channel_pages = buffer->layout.pages_per_stream - 2 * token_pages;
    uint64_t base = (uint64_t)head_index * buffer->layout.pages_per_stream;

    if (location->key.kind == CEMU_KV_PAGE_CHANNEL_K) {
        base += token_pages;
    } else if (location->key.kind == CEMU_KV_PAGE_TOKEN_V) {
        base += token_pages + channel_pages;
    }
    return base + location->page_index;
}

static void gather_full_scan(NativeKvBuffer *buffer,
    const NativeKvBufferLease *lease, uint32_t kind, FemuReadObservation *observation)
{
    uint32_t capacity = group_capacity(buffer, kind);
    uint32_t pages_per_group = kind == CEMU_KV_PAGE_CHANNEL_K ?
                               buffer->layout.channel_group_pages : 1;

    observation->full_scan_valid = true;
    observation->page_bytes = buffer->layout.geometry.page_bytes;
    for (uint32_t batch = lease->batch_first; batch < lease->batch_first + lease->batches; batch++) {
        for (uint32_t head = 0; head < buffer->layout.geometry.heads; head++) {
            uint64_t index = ((uint64_t)batch * buffer->layout.geometry.layers +
                              lease->layer) * buffer->layout.geometry.heads + head;
            CemuKvCoordinate coordinate = kv_head_coordinate(buffer, index);
            uint32_t row_bytes = buffer->row_bytes / pages_per_group;

            coordinate.kind = kind;
            for (coordinate.token = 0; coordinate.token < lease->tokens; coordinate.token += capacity) {
                uint64_t bytes = MIN((uint64_t)capacity, lease->tokens - coordinate.token) * row_bytes;
                for (uint32_t stripe = 0; stripe < pages_per_group; stripe++) {
                    coordinate.channel = stripe * buffer->layout.geometry.channels_per_page;
                    if (dram_element(buffer, index, &coordinate)) {
                        observation->full_scan_tail_bytes += bytes;
                    } else {
                        observation->full_scan_pages++;
                        observation->full_scan_logical_bytes += bytes;
                    }
                }
            }
        }
    }
    observation->full_scan_flash_bytes = observation->full_scan_pages * observation->page_bytes;
}

static uint64_t gather_unique_bytes(const NativeKvReadFragment *fragments,
                                    uint32_t first, uint32_t page_bytes)
{
    uint64_t coverage[65536 / 64];
    uint64_t bytes = 0;

    memset(coverage, 0, page_bytes / 8);
    for (uint32_t index = first; index != UINT32_MAX; index = fragments[index].next) {
        bytes += femu_observe_unique_bytes(coverage, fragments[index].offset, fragments[index].bytes);
    }
    return bytes;
}

CemuKvStatus native_kv_buffer_gather_phase(NativeKvBuffer *buffer,
    NativeKvGatherSource source, void *opaque, uint32_t count, void *output,
    size_t bytes, uint64_t start_ns, NativeKvGatherResult *result,
    const NativeKvBufferLease *lease, uint32_t kind, FemuReadObservation *observation)
{
    NativeKvGatherResult candidate = { .ready_ns = start_ns };
    NativeKvReadFragment *fragments = NULL;
    uint32_t *table = NULL;
    uint64_t required = 0, pieces = 0, plan_bytes = 0, cursor = 0;
    uint32_t table_size = 1, used = 0, first_page = UINT32_MAX, last_page = UINT32_MAX;
    CemuKvStatus status = CEMU_KV_INVALID;

    if (!buffer || !source || !count || !output || !result || start_ns > INT64_MAX ||
        !bytes || bytes > 64ULL * 1024 * 1024 ||
        (observation && (!lease || lease->buffer != buffer || kind > CEMU_KV_PAGE_TOKEN_V))) {
        return CEMU_KV_INVALID;
    }
    *result = candidate;
    g_mutex_lock(&buffer->lock);
    if (buffer->stats.failed) {
        status = CEMU_KV_IO_FAILED;
        goto done;
    }
    for (uint32_t index = 0; index < count; index++) {
        NativeKvGatherRange range;
        uint32_t head_index, capacity, unit;
        uint64_t end;

        source(opaque, index, &range);
        if (!range.tokens || !kv_head_index(buffer, &range.coordinate, &head_index) ||
            range.tokens > buffer->layout.geometry.max_tokens - range.coordinate.token ||
            (range.coordinate.kind != CEMU_KV_PAGE_CHANNEL_K && range.coordinate.channel)) {
            goto done;
        }
        end = range.coordinate.token + range.tokens;
        if (end > buffer->head_states[head_index].accepted) {
            status = CEMU_KV_NOT_FOUND;
            goto done;
        }
        unit = range.coordinate.kind == CEMU_KV_PAGE_CHANNEL_K ?
               buffer->layout.geometry.element_bytes : buffer->row_bytes;
        if ((uint64_t)range.tokens * unit > bytes - required) {
            goto done;
        }
        required += (uint64_t)range.tokens * unit;
        capacity = group_capacity(buffer, range.coordinate.kind);
        while (range.coordinate.token < end) {
            pieces += !dram_element(buffer, head_index, &range.coordinate);
            range.coordinate.token += MIN(end - range.coordinate.token,
                                          capacity - range.coordinate.token % capacity);
        }
    }
    if (required != bytes) {
        goto done;
    }
    if (observation) {
        gather_full_scan(buffer, lease, kind, observation);
        observation->logical_spans = count;
        observation->aligned_bytes_before_dedup = pieces * buffer->layout.geometry.page_bytes;
    }
    if (pieces > UINT32_MAX / 2) {
        status = CEMU_KV_LIMIT;
        goto done;
    }
    while (table_size < pieces * 2) {
        table_size *= 2;
    }
    plan_bytes = pieces ? pieces * sizeof(*fragments) + (uint64_t)table_size * sizeof(*table) : 0;
    if (plan_bytes > buffer->memory_budget - buffer->stats.memory_bytes) {
        status = CEMU_KV_LIMIT;
        goto done;
    }
    if (pieces) {
        fragments = g_try_new(NativeKvReadFragment, pieces);
        table = g_try_new0(uint32_t, table_size);
        if (!fragments || !table) {
            status = CEMU_KV_NO_MEMORY;
            goto done;
        }
    }
    for (uint32_t index = 0; index < count; index++) {
        NativeKvGatherRange range;
        uint32_t head_index, unit, remaining;

        source(opaque, index, &range);
        head_index = ((uint64_t)range.coordinate.batch * buffer->layout.geometry.layers +
                  range.coordinate.layer) * buffer->layout.geometry.heads + range.coordinate.head;
        unit = range.coordinate.kind == CEMU_KV_PAGE_CHANNEL_K ?
               buffer->layout.geometry.element_bytes : buffer->row_bytes;
        remaining = range.tokens;
        while (remaining) {
            CemuKvCoordinate *coordinate = &range.coordinate;
            uint32_t capacity = group_capacity(buffer, coordinate->kind);
            uint32_t tokens = MIN(remaining, capacity - coordinate->token % capacity);

            uint8_t *dram = dram_element(buffer, head_index, coordinate);

            if (!dram) {
                CemuKvLocation location;
                uint64_t key, hash;
                uint32_t slot;

                if (!cemu_kv_resolve(&buffer->layout, &buffer->identity, coordinate, &location)) {
                    goto done;
                }
                key = gather_page_key(buffer, head_index, &location);
                hash = key * UINT64_C(0x9e3779b97f4a7c15);
                slot = (hash ^ (hash >> 32)) & (table_size - 1);
                while (table[slot] && fragments[table[slot] - 1].page_key != key) {
                    slot = (slot + 1) & (table_size - 1);
                }
                g_assert(used < pieces);
                fragments[used] = (NativeKvReadFragment){
                    .coordinate = *coordinate, .page_key = key, .destination = cursor,
                    .offset = location.byte_offset, .bytes = tokens * unit,
                    .next = UINT32_MAX, .last = used, .next_page = UINT32_MAX };
                if (table[slot]) {
                    NativeKvReadFragment *first = &fragments[table[slot] - 1];

                    fragments[first->last].next = used;
                    first->last = used;
                } else {
                    table[slot] = used + 1;
                    if (last_page != UINT32_MAX) {
                        fragments[last_page].next_page = used;
                    } else {
                        first_page = used;
                    }
                    last_page = used;
                }
                used++;
            } else {
                memcpy((uint8_t *)output + cursor, dram, (size_t)tokens * unit);
                candidate.tail_bytes += (uint64_t)tokens * unit;
                if (observation) {
                    observation->tail_copies++;
                }
            }
            cursor += (uint64_t)tokens * unit;
            coordinate->token += tokens;
            remaining -= tokens;
        }
    }
    while (first_page != UINT32_MAX) {
        CemuKvDeviceBatchResult batch = { .ready_ns = start_ns };
        uint32_t page = first_page, pages = 0;

        while (page != UINT32_MAX && pages < buffer->batch_pages) {
            buffer->gather_pages[pages] = (CemuKvDevicePage){
                .coordinate = fragments[page].coordinate,
                .buffer = buffer->gather_data + (uint64_t)pages * buffer->layout.geometry.page_bytes,
                .bytes = buffer->layout.geometry.page_bytes };
            pages++;
            page = fragments[page].next_page;
        }
        status = bb_native_kv_batch_observed(buffer->ns, &buffer->identity, buffer->gather_pages,
                                           pages, false, start_ns, &batch, observation);
        candidate.ready_ns = MAX(candidate.ready_ns, batch.ready_ns);
        candidate.transferred_bytes += batch.transferred_bytes;
        candidate.flash_pages += batch.completed_pages;
        candidate.batches++;
        buffer->stats.read_bytes += batch.transferred_bytes;
        if (status != CEMU_KV_OK) {
            goto done;
        }
        for (uint32_t slot = 0; slot < pages; slot++) {
            if (observation) {
                observation->unique_flash_bytes += gather_unique_bytes(fragments,
                    first_page, buffer->layout.geometry.page_bytes);
            }
            for (uint32_t fragment = first_page; fragment != UINT32_MAX;
                 fragment = fragments[fragment].next) {
                NativeKvReadFragment *span = &fragments[fragment];

                memcpy((uint8_t *)output + span->destination,
                    buffer->gather_pages[slot].buffer + span->offset, span->bytes);
            }
            first_page = fragments[first_page].next_page;
        }
    }
    candidate.output_bytes = required;
    status = CEMU_KV_OK;
done:
    *result = candidate;
    g_free(table);
    g_free(fragments);
    g_mutex_unlock(&buffer->lock);
    return status;
}

CemuKvStatus native_kv_buffer_stats(NativeKvBuffer *buffer,
    const CemuKvCoordinate *coordinate, NativeKvBufferStats *stats)
{
    uint32_t head_index;

    if (!buffer || !stats || !kv_head_index(buffer, coordinate, &head_index)) {
        return CEMU_KV_INVALID;
    }
    g_mutex_lock(&buffer->lock);
    *stats = buffer->stats;
    for (uint32_t domain = 0; domain < buffer->domains; domain++) {
        stats->ready_blocks += buffer->blocks[domain].count / buffer->block_pages;
    }
    stats->accepted = buffer->head_states[head_index].accepted;
    /* Keep the STATUS ABI: all three layouts are now filled at acceptance. */
    for (uint32_t kind = 0; kind < 3; kind++) {
        stats->packed[kind] = stats->accepted;
    }
    memcpy(stats->committed, buffer->head_states[head_index].committed, sizeof(stats->committed));
    g_mutex_unlock(&buffer->lock);
    return CEMU_KV_OK;
}

CemuKvStatus native_kv_buffer_drain(NativeKvBuffer *buffer)
{
    CemuKvStatus status;

    if (!buffer) {
        return CEMU_KV_INVALID;
    }
    g_mutex_lock(&buffer->lock);
    if (buffer->leases && !buffer->stats.failed) {
        g_mutex_unlock(&buffer->lock);
        return CEMU_KV_PENDING;
    }
    if (!buffer->draining && !buffer->stats.failed) {
        buffer->draining = true;
        /* Seal only the current partial page(s) of each layout. Full pages are
         * already queued or inflight; unused bytes were zeroed on allocation. */
        for (uint32_t head_index = 0; head_index < buffer->layout.streams; head_index++) {
            uint64_t accepted = buffer->head_states[head_index].accepted;
            CemuKvCoordinate coordinate = kv_head_coordinate(buffer, head_index);
            for (uint32_t kind = 0; kind < 3; kind++) {
                uint32_t capacity = group_capacity(buffer, kind);
                uint32_t stripes = kind == CEMU_KV_PAGE_CHANNEL_K ? buffer->layout.channel_group_pages : 1;
                if (!(accepted % capacity)) {
                    continue;
                }
                coordinate.kind = kind;
                coordinate.token = accepted - accepted % capacity;
                for (uint32_t stripe = 0; stripe < stripes; stripe++) {
                    CemuKvLocation location;
                    uint32_t page;
                    coordinate.channel = stripe * buffer->layout.geometry.channels_per_page;
                    if (!cemu_kv_resolve(&buffer->layout, &buffer->identity, &coordinate, &location)) {
                        g_assert_not_reached();
                    }
                    page = staged_page(buffer, gather_page_key(buffer, head_index, &location));
                    g_assert(page && buffer->staged[page].valid_tokens == accepted % capacity);
                    enqueue_page(buffer, page);
                }
            }
        }
    }
    g_cond_signal(&buffer->wake);
    status = buffer->stats.failed ? CEMU_KV_IO_FAILED :
        (buffer->stats.sealed ? CEMU_KV_OK : CEMU_KV_PENDING);
    g_mutex_unlock(&buffer->lock);
    return status;
}

CemuKvStatus native_kv_buffer_acquire(NativeKvBuffer *buffer,
    const CemuKvIdentity *identity, uint32_t batch_first, uint32_t batches,
    uint32_t layer, uint64_t tokens, NativeKvBufferLease *lease)
{
    CemuKvStatus status = CEMU_KV_INVALID;
    const CemuKvGeometry *geometry;

    if (!native_kv_buffer_matches(buffer, identity) || !lease || lease->buffer || !batches || !tokens) {
        return CEMU_KV_INVALID;
    }
    geometry = &buffer->layout.geometry;
    if (batch_first >= geometry->batches || batches > geometry->batches - batch_first ||
        layer >= geometry->layers || tokens > geometry->max_tokens) {
        return CEMU_KV_INVALID;
    }
    g_mutex_lock(&buffer->lock);
    if (buffer->stats.failed || buffer->stop) {
        status = CEMU_KV_IO_FAILED;
        goto done;
    }
    if ((buffer->draining && !buffer->stats.sealed) || buffer->leases == UINT32_MAX) {
        status = CEMU_KV_PENDING;
        goto done;
    }
    for (uint32_t batch = batch_first; batch < batch_first + batches; batch++) {
        for (uint32_t head = 0; head < geometry->heads; head++) {
            NativeKvHeadState *head_state = &buffer->head_states[
                ((uint64_t)batch * geometry->layers + layer) * geometry->heads + head];

            if (head_state->accepted != tokens) {
                goto done;
            }
        }
    }
    for (uint32_t batch = batch_first; batch < batch_first + batches; batch++) {
        for (uint32_t head = 0; head < geometry->heads; head++) {
            buffer->head_states[((uint64_t)batch * geometry->layers + layer) * geometry->heads + head].readers++;
        }
    }
    buffer->leases++;
    *lease = (NativeKvBufferLease){ buffer, *geometry, batch_first, batches, layer, tokens };
    status = CEMU_KV_OK;
done:
    g_mutex_unlock(&buffer->lock);
    return status;
}

void native_kv_buffer_release(NativeKvBufferLease *lease)
{
    NativeKvBuffer *buffer;
    const CemuKvGeometry *geometry;

    if (!lease || !lease->buffer) {
        return;
    }
    buffer = lease->buffer;
    geometry = &lease->geometry;
    g_mutex_lock(&buffer->lock);
    for (uint32_t batch = lease->batch_first; batch < lease->batch_first + lease->batches; batch++) {
        for (uint32_t head = 0; head < geometry->heads; head++) {
            NativeKvHeadState *head_state = &buffer->head_states[
                ((uint64_t)batch * geometry->layers + lease->layer) * geometry->heads + head];

            g_assert(head_state->readers);
            head_state->readers--;
        }
    }
    g_assert(buffer->leases);
    buffer->leases--;
    lease->buffer = NULL;
    g_cond_broadcast(&buffer->wake);
    g_mutex_unlock(&buffer->lock);
}

void native_kv_buffer_stop(NativeKvBuffer *buffer)
{
    if (!buffer) {
        return;
    }
    g_mutex_lock(&buffer->lock);
    buffer->stop = true;
    if (!buffer->stats.sealed) {
        buffer->stats.failed = true;
    }
    g_cond_signal(&buffer->wake);
    g_mutex_unlock(&buffer->lock);
    if (buffer->worker) {
        g_thread_join(buffer->worker);
        buffer->worker = NULL;
    }
    nvme_cancel_internal_completions(buffer->ns->ctrl, buffer);
    g_mutex_lock(&buffer->lock);
    while (buffer->leases) {
        g_cond_wait(&buffer->wake, &buffer->lock);
    }
    g_mutex_unlock(&buffer->lock);
}

void native_kv_buffer_free(NativeKvBuffer *buffer)
{
    if (!buffer) {
        return;
    }
    native_kv_buffer_stop(buffer);
    g_mutex_clear(&buffer->lock);
    g_cond_clear(&buffer->wake);
    g_free(buffer->head_states);
    g_free(buffer->staged_data);
    g_free(buffer->staged);
    g_free(buffer->blocks);
    g_free(buffer->page_hash);
    g_free(buffer->inflight);
    g_free(buffer->writebacks);
    g_free(buffer->read_page);
    g_free(buffer->gather_data);
    g_free(buffer->gather_pages);
    g_free(buffer->gather_keys);
    g_free(buffer->pages);
    g_free(buffer);
}
