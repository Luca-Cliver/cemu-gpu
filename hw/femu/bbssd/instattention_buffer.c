#include "qemu/osdep.h"
#include "bb.h"
#include "instattention_buffer.h"

typedef struct NativeKvHeadState {
    uint64_t accepted;
    uint64_t committed[3];
    uint8_t queued;
    uint32_t readers;
} NativeKvHeadState;

typedef struct NativeKvPending {
    uint32_t head_index;
    uint32_t kind;
    uint32_t count;
} NativeKvPending;

struct NativeKvBuffer {
    NvmeNamespace *ns;
    CemuKvIdentity identity;
    CemuKvLayout layout;
    GMutex lock;
    GCond wake;
    GThread *worker;
    NativeKvHeadState *head_states;
    uint32_t *queue;
    uint32_t queue_head;
    uint32_t queue_tail;
    uint32_t queue_count;
    uint32_t key_capacity;
    uint32_t value_capacity;
    uint32_t row_bytes;
    uint32_t batch_pages;
    uint64_t memory_budget;
    uint8_t *tails;
    uint8_t *scratch;
    uint8_t *read_page;
    uint8_t *gather_data;
    CemuKvDevicePage *gather_pages;
    CemuKvPageKey *gather_keys;
    CemuKvDevicePage *pages;
    NativeKvPending *pending;
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

static uint8_t *tail_row(NativeKvBuffer *buffer, uint32_t head_index, uint32_t kind,
                         uint64_t token)
{
    uint64_t base = (uint64_t)head_index * (buffer->key_capacity + buffer->value_capacity);
    uint32_t capacity = buffer->key_capacity;

    if (kind == CEMU_KV_PAGE_TOKEN_V) {
        base += buffer->key_capacity;
        capacity = buffer->value_capacity;
    }
    return buffer->tails + (base + token % capacity) * buffer->row_bytes;
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

static void enqueue_ready(NativeKvBuffer *buffer, uint32_t head_index)
{
    NativeKvHeadState *head_state = &buffer->head_states[head_index];

    for (uint32_t kind = 0; kind < 3; kind++) {
        uint64_t available = head_state->accepted - head_state->committed[kind];

        if (!(head_state->queued & (1U << kind)) && available &&
            (available >= group_capacity(buffer, kind) || buffer->draining)) {
            buffer->queue[buffer->queue_tail] = head_index * 3 + kind;
            buffer->queue_tail = (buffer->queue_tail + 1) % (buffer->layout.streams * 3);
            buffer->queue_count++;
            head_state->queued |= 1U << kind;
        }
    }
}

static uint32_t pack_batch(NativeKvBuffer *buffer, uint32_t *groups)
{
    const CemuKvGeometry *geometry = &buffer->layout.geometry;
    uint32_t total = 0;

    *groups = 0;
    while (buffer->queue_count) {
        uint32_t task = buffer->queue[buffer->queue_head];
        uint32_t head_index = task / 3, kind = task % 3;
        NativeKvHeadState *head_state = &buffer->head_states[head_index];
        uint32_t capacity = group_capacity(buffer, kind);
        uint32_t count = MIN(head_state->accepted - head_state->committed[kind], capacity);
        uint32_t pages = kind == CEMU_KV_PAGE_CHANNEL_K ? buffer->layout.channel_group_pages : 1;

        if (pages > buffer->batch_pages - total) {
            break;
        }
        buffer->pending[(*groups)++] = (NativeKvPending){ head_index, kind, count };
        for (uint32_t page = 0; page < pages; page++) {
            uint8_t *destination = buffer->scratch + (uint64_t)total * geometry->page_bytes;
            CemuKvCoordinate coordinate = kv_head_coordinate(buffer, head_index);

            coordinate.kind = kind;
            coordinate.token = head_state->committed[kind];
            coordinate.channel = page * geometry->channels_per_page;
            memset(destination, 0, geometry->page_bytes);
            for (uint32_t token = 0; token < count; token++) {
                uint8_t *source = tail_row(buffer, head_index, kind, coordinate.token + token);

                if (kind != CEMU_KV_PAGE_CHANNEL_K) {
                    memcpy(destination + (uint64_t)token * buffer->row_bytes, source, buffer->row_bytes);
                } else {
                    for (uint32_t channel = 0; channel < geometry->channels_per_page; channel++) {
                        memcpy(destination + ((uint64_t)channel * capacity + token) * geometry->element_bytes,
                            source + (coordinate.channel + channel) * geometry->element_bytes,
                            geometry->element_bytes);
                    }
                }
            }
            buffer->pages[total++] = (CemuKvDevicePage){ .coordinate = coordinate,
                .valid_tokens = count, .buffer = destination, .bytes = geometry->page_bytes };
        }
        buffer->queue_head = (buffer->queue_head + 1) % (buffer->layout.streams * 3);
        buffer->queue_count--;
    }
    return total;
}

static void *buffer_worker(void *opaque)
{
    NativeKvBuffer *buffer = opaque;

    g_mutex_lock(&buffer->lock);
    while (!buffer->stop) {
        uint32_t groups, count;
        CemuKvDeviceBatchResult result = {0};
        CemuKvStatus status;

        if (buffer->stats.failed || !buffer->queue_count) {
            if (buffer->draining && !buffer->stats.failed) {
                buffer->stats.sealed = true;
            }
            g_cond_wait(&buffer->wake, &buffer->lock);
            continue;
        }
        count = pack_batch(buffer, &groups);
        g_mutex_unlock(&buffer->lock);
        status = bb_native_kv_batch(buffer->ns, &buffer->identity, buffer->pages,
                                    count, true, clock_ns(), &result);
        g_mutex_lock(&buffer->lock);
        buffer->stats.write_bytes += result.transferred_bytes;
        buffer->stats.write_batches++;
        buffer->stats.pending_ready_ns = result.ready_ns;
        if (status != CEMU_KV_OK || result.completed_pages != count) {
            buffer->stats.failed = true;
            continue;
        }
        while (!buffer->stop && (uint64_t)clock_ns() < result.ready_ns) {
            g_cond_wait_until(&buffer->wake, &buffer->lock, g_get_monotonic_time() + 10);
        }
        if (buffer->stop) {
            break;
        }
        for (uint32_t index = 0; index < groups; index++) {
            NativeKvPending *pending = &buffer->pending[index];
            NativeKvHeadState *head_state = &buffer->head_states[pending->head_index];

            head_state->committed[pending->kind] += pending->count;
            head_state->queued &= ~(1U << pending->kind);
            enqueue_ready(buffer, pending->head_index);
        }
        buffer->stats.pending_ready_ns = 0;
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
    uint64_t tails, required, bytes_per_head;
    GError *error = NULL;

    if (!ns || !identity || !result || !cemu_kv_layout_init(&layout, geometry) ||
        layout.streams > 65536 || batch_pages < layout.channel_group_pages ||
        batch_pages > CEMU_NATIVE_INTERNAL_MAX_PAGES) {
        return CEMU_KV_INVALID;
    }
    bytes_per_head = 2ULL * (layout.channel_group_tokens + layout.token_group_tokens) *
                 geometry->head_dim * geometry->element_bytes;
    if (!cemu_kv_multiply(layout.streams, bytes_per_head, &tails)) {
        return CEMU_KV_LIMIT;
    }
    required = sizeof(*buffer) + layout.streams * (sizeof(NativeKvHeadState) + 3 * sizeof(uint32_t)) +
        (uint64_t)batch_pages * (geometry->page_bytes + sizeof(CemuKvDevicePage) + sizeof(NativeKvPending)) +
        (uint64_t)batch_pages * (geometry->page_bytes + sizeof(CemuKvDevicePage) + sizeof(CemuKvPageKey)) +
        geometry->page_bytes;
    if (required > budget || tails > budget - required || tails > SIZE_MAX - required) {
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
    buffer->key_capacity = 2 * layout.channel_group_tokens;
    buffer->value_capacity = 2 * layout.token_group_tokens;
    buffer->row_bytes = geometry->head_dim * geometry->element_bytes;
    buffer->stats.memory_bytes = tails + required;
    buffer->head_states = g_try_new0(NativeKvHeadState, layout.streams);
    buffer->queue = g_try_new(uint32_t, layout.streams * 3);
    buffer->tails = g_try_malloc0(tails);
    buffer->scratch = g_try_malloc((uint64_t)batch_pages * geometry->page_bytes);
    buffer->read_page = g_try_malloc(geometry->page_bytes);
    buffer->gather_data = g_try_malloc((uint64_t)batch_pages * geometry->page_bytes);
    buffer->gather_pages = g_try_new(CemuKvDevicePage, batch_pages);
    buffer->gather_keys = g_try_new(CemuKvPageKey, batch_pages);
    buffer->pages = g_try_new(CemuKvDevicePage, batch_pages);
    buffer->pending = g_try_new(NativeKvPending, batch_pages);
    if (!buffer->head_states || !buffer->queue || !buffer->tails || !buffer->scratch ||
        !buffer->read_page || !buffer->pages || !buffer->pending ||
        !buffer->gather_data || !buffer->gather_pages || !buffer->gather_keys) {
        native_kv_buffer_free(buffer);
        return CEMU_KV_NO_MEMORY;
    }
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
    CemuKvStatus status = CEMU_KV_INVALID;

    if (!buffer || !coordinate || !key || !value || !count || !heads || coordinate->kind || coordinate->channel ||
        !kv_head_index(buffer, coordinate, &head_index) || count > buffer->layout.geometry.max_tokens - coordinate->token ||
        heads > buffer->layout.geometry.heads - coordinate->head ||
        (uint64_t)heads * count * buffer->row_bytes != bytes || count > buffer->value_capacity) {
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
        uint64_t oldest = MIN(head_state->committed[0], head_state->committed[1]);

        if (head_state->readers || count > buffer->key_capacity - (head_state->accepted - oldest) ||
            count > buffer->value_capacity - (head_state->accepted - head_state->committed[2])) {
            status = CEMU_KV_PENDING;
            goto done;
        }
    }
    for (uint32_t head = 0; head < heads; head++) {
        NativeKvHeadState *head_state = &buffer->head_states[head_index + head];

        for (uint32_t token = 0; token < count; token++) {
            uint64_t offset = ((uint64_t)head * count + token) * buffer->row_bytes;

            memcpy(tail_row(buffer, head_index + head, CEMU_KV_PAGE_TOKEN_K, head_state->accepted + token),
                   (const uint8_t *)key + offset, buffer->row_bytes);
            memcpy(tail_row(buffer, head_index + head, CEMU_KV_PAGE_TOKEN_V, head_state->accepted + token),
                   (const uint8_t *)value + offset, buffer->row_bytes);
        }
        head_state->accepted += count;
        enqueue_ready(buffer, head_index + head);
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
    } else if (coordinate->token >= head_state->committed[coordinate->kind]) {
        memcpy(output, tail_row(buffer, head_index, coordinate->kind, coordinate->token) +
            (uint64_t)coordinate->channel * buffer->layout.geometry.element_bytes, bytes);
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
        uint64_t committed = buffer->head_states[head_index].committed[coordinate.kind];
        uint32_t unit = coordinate.kind == CEMU_KV_PAGE_CHANNEL_K ?
                        geometry->element_bytes : buffer->row_bytes;
        uint32_t remaining = ranges[range].tokens;

        while (remaining) {
            CemuKvLocation location;
            uint32_t capacity = group_capacity(buffer, coordinate.kind);
            uint32_t tokens = MIN(remaining, capacity - coordinate.token % capacity);
            bool flash = coordinate.token < committed;

            if (flash) {
                uint32_t page;

                tokens = MIN(tokens, committed - coordinate.token);
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
                for (uint32_t token = 0; token < tokens; token++) {
                    memcpy(output + (size_t)token * unit,
                        tail_row(buffer, head_index, coordinate.kind, coordinate.token + token) +
                        (size_t)coordinate.channel * geometry->element_bytes, unit);
                }
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

CemuKvStatus native_kv_buffer_gather_phase(NativeKvBuffer *buffer,
    NativeKvGatherSource source, void *opaque, uint32_t count, void *output,
    size_t bytes, uint64_t start_ns, NativeKvGatherResult *result)
{
    NativeKvGatherResult candidate = { .ready_ns = start_ns };
    NativeKvReadFragment *fragments = NULL;
    uint32_t *table = NULL;
    uint64_t required = 0, pieces = 0, plan_bytes = 0, cursor = 0;
    uint32_t table_size = 1, used = 0, first_page = UINT32_MAX, last_page = UINT32_MAX;
    CemuKvStatus status = CEMU_KV_INVALID;

    if (!buffer || !source || !count || !output || !result || start_ns > INT64_MAX ||
        !bytes || bytes > 64ULL * 1024 * 1024) {
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
        uint64_t end, committed;

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
        committed = buffer->head_states[head_index].committed[range.coordinate.kind];
        if (range.coordinate.token < committed) {
            capacity = group_capacity(buffer, range.coordinate.kind);
            pieces += (MIN(end, committed) - 1) / capacity - range.coordinate.token / capacity + 1;
        }
    }
    if (required != bytes) {
        goto done;
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
        uint64_t committed;

        source(opaque, index, &range);
        head_index = ((uint64_t)range.coordinate.batch * buffer->layout.geometry.layers +
                  range.coordinate.layer) * buffer->layout.geometry.heads + range.coordinate.head;
        committed = buffer->head_states[head_index].committed[range.coordinate.kind];
        unit = range.coordinate.kind == CEMU_KV_PAGE_CHANNEL_K ?
               buffer->layout.geometry.element_bytes : buffer->row_bytes;
        remaining = range.tokens;
        while (remaining) {
            CemuKvCoordinate *coordinate = &range.coordinate;
            uint32_t tokens = remaining;

            if (coordinate->token < committed) {
                CemuKvLocation location;
                uint32_t capacity = group_capacity(buffer, coordinate->kind);
                uint64_t key, hash;
                uint32_t slot;

                tokens = MIN(tokens, capacity - coordinate->token % capacity);
                tokens = MIN(tokens, committed - coordinate->token);
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
                for (uint32_t token = 0; token < tokens; token++) {
                    memcpy((uint8_t *)output + cursor + (uint64_t)token * unit,
                        tail_row(buffer, head_index, coordinate->kind, coordinate->token + token) +
                        (uint64_t)coordinate->channel * buffer->layout.geometry.element_bytes, unit);
                }
                candidate.tail_bytes += (uint64_t)tokens * unit;
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
        status = bb_native_kv_batch(buffer->ns, &buffer->identity, buffer->gather_pages,
                                   pages, false, start_ns, &batch);
        candidate.ready_ns = MAX(candidate.ready_ns, batch.ready_ns);
        candidate.transferred_bytes += batch.transferred_bytes;
        candidate.flash_pages += batch.completed_pages;
        candidate.batches++;
        buffer->stats.read_bytes += batch.transferred_bytes;
        if (status != CEMU_KV_OK) {
            goto done;
        }
        for (uint32_t slot = 0; slot < pages; slot++) {
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
    stats->accepted = buffer->head_states[head_index].accepted;
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
    buffer->draining = true;
    for (uint32_t head_index = 0; head_index < buffer->layout.streams; head_index++) {
        enqueue_ready(buffer, head_index);
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
    g_free(buffer->queue);
    g_free(buffer->tails);
    g_free(buffer->scratch);
    g_free(buffer->read_page);
    g_free(buffer->gather_data);
    g_free(buffer->gather_pages);
    g_free(buffer->gather_keys);
    g_free(buffer->pages);
    g_free(buffer->pending);
    g_free(buffer);
}
