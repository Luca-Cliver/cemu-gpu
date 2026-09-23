#include "qemu/osdep.h"
#include "bb.h"
#include "instattention_buffer.h"

typedef struct NativeKvStream {
    uint64_t accepted;
    uint64_t committed[3];
    uint8_t queued;
    uint32_t readers;
} NativeKvStream;

typedef struct NativeKvPending {
    uint32_t stream;
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
    NativeKvStream *streams;
    uint32_t *queue;
    uint32_t queue_head;
    uint32_t queue_tail;
    uint32_t queue_count;
    uint32_t key_capacity;
    uint32_t value_capacity;
    uint32_t row_bytes;
    uint32_t batch_pages;
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

static uint8_t *tail_row(NativeKvBuffer *buffer, uint32_t stream, uint32_t kind,
                         uint64_t token)
{
    uint64_t base = (uint64_t)stream * (buffer->key_capacity + buffer->value_capacity);
    uint32_t capacity = buffer->key_capacity;

    if (kind == CEMU_KV_PAGE_TOKEN_V) {
        base += buffer->key_capacity;
        capacity = buffer->value_capacity;
    }
    return buffer->tails + (base + token % capacity) * buffer->row_bytes;
}

static bool stream_index(const NativeKvBuffer *buffer, const CemuKvCoordinate *coordinate,
                          uint32_t *index)
{
    CemuKvLocation location;

    if (!cemu_kv_resolve(&buffer->layout, &buffer->identity, coordinate, &location)) {
        return false;
    }
    *index = ((uint64_t)coordinate->batch * buffer->layout.geometry.layers +
              coordinate->layer) * buffer->layout.geometry.heads + coordinate->head;
    return true;
}

static CemuKvCoordinate stream_coordinate(const NativeKvBuffer *buffer, uint32_t index)
{
    const CemuKvGeometry *geometry = &buffer->layout.geometry;

    return (CemuKvCoordinate){ .head = index % geometry->heads,
        .layer = index / geometry->heads % geometry->layers,
        .batch = index / geometry->heads / geometry->layers };
}

static void enqueue_ready(NativeKvBuffer *buffer, uint32_t index)
{
    NativeKvStream *stream = &buffer->streams[index];

    for (uint32_t kind = 0; kind < 3; kind++) {
        uint64_t available = stream->accepted - stream->committed[kind];

        if (!(stream->queued & (1U << kind)) && available &&
            (available >= group_capacity(buffer, kind) || buffer->draining)) {
            buffer->queue[buffer->queue_tail] = index * 3 + kind;
            buffer->queue_tail = (buffer->queue_tail + 1) % (buffer->layout.streams * 3);
            buffer->queue_count++;
            stream->queued |= 1U << kind;
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
        uint32_t index = task / 3, kind = task % 3;
        NativeKvStream *stream = &buffer->streams[index];
        uint32_t capacity = group_capacity(buffer, kind);
        uint32_t count = MIN(stream->accepted - stream->committed[kind], capacity);
        uint32_t pages = kind == CEMU_KV_PAGE_CHANNEL_K ? buffer->layout.channel_group_pages : 1;

        if (pages > buffer->batch_pages - total) {
            break;
        }
        buffer->pending[(*groups)++] = (NativeKvPending){ index, kind, count };
        for (uint32_t page = 0; page < pages; page++) {
            uint8_t *destination = buffer->scratch + (uint64_t)total * geometry->page_bytes;
            CemuKvCoordinate coordinate = stream_coordinate(buffer, index);

            coordinate.kind = kind;
            coordinate.token = stream->committed[kind];
            coordinate.channel = page * geometry->channels_per_page;
            memset(destination, 0, geometry->page_bytes);
            for (uint32_t token = 0; token < count; token++) {
                uint8_t *source = tail_row(buffer, index, kind, coordinate.token + token);

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
            NativeKvStream *stream = &buffer->streams[pending->stream];

            stream->committed[pending->kind] += pending->count;
            stream->queued &= ~(1U << pending->kind);
            enqueue_ready(buffer, pending->stream);
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
    uint64_t tails, required, per_stream;
    GError *error = NULL;

    if (!ns || !identity || !result || !cemu_kv_layout_init(&layout, geometry) ||
        layout.streams > 65536 || batch_pages < layout.channel_group_pages ||
        batch_pages > CEMU_NATIVE_INTERNAL_MAX_PAGES) {
        return CEMU_KV_INVALID;
    }
    per_stream = 2ULL * (layout.channel_group_tokens + layout.token_group_tokens) *
                 geometry->head_dim * geometry->element_bytes;
    if (!cemu_kv_multiply(layout.streams, per_stream, &tails)) {
        return CEMU_KV_LIMIT;
    }
    required = sizeof(*buffer) + layout.streams * (sizeof(NativeKvStream) + 3 * sizeof(uint32_t)) +
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
    buffer->key_capacity = 2 * layout.channel_group_tokens;
    buffer->value_capacity = 2 * layout.token_group_tokens;
    buffer->row_bytes = geometry->head_dim * geometry->element_bytes;
    buffer->stats.memory_bytes = tails + required;
    buffer->streams = g_try_new0(NativeKvStream, layout.streams);
    buffer->queue = g_try_new(uint32_t, layout.streams * 3);
    buffer->tails = g_try_malloc0(tails);
    buffer->scratch = g_try_malloc((uint64_t)batch_pages * geometry->page_bytes);
    buffer->read_page = g_try_malloc(geometry->page_bytes);
    buffer->gather_data = g_try_malloc((uint64_t)batch_pages * geometry->page_bytes);
    buffer->gather_pages = g_try_new(CemuKvDevicePage, batch_pages);
    buffer->gather_keys = g_try_new(CemuKvPageKey, batch_pages);
    buffer->pages = g_try_new(CemuKvDevicePage, batch_pages);
    buffer->pending = g_try_new(NativeKvPending, batch_pages);
    if (!buffer->streams || !buffer->queue || !buffer->tails || !buffer->scratch ||
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
    uint32_t index;
    NativeKvStream *stream;
    CemuKvStatus status = CEMU_KV_INVALID;

    if (!buffer || !coordinate || !key || !value || !count || coordinate->kind || coordinate->channel ||
        !stream_index(buffer, coordinate, &index) || count > buffer->layout.geometry.max_tokens - coordinate->token ||
        (uint64_t)count * buffer->row_bytes != bytes || count > buffer->value_capacity) {
        return CEMU_KV_INVALID;
    }
    g_mutex_lock(&buffer->lock);
    stream = &buffer->streams[index];
    if (buffer->stats.failed) {
        status = CEMU_KV_IO_FAILED;
    } else if (!buffer->stop && !buffer->draining && coordinate->token == stream->accepted) {
        uint64_t oldest = MIN(stream->committed[0], stream->committed[1]);

        status = CEMU_KV_PENDING;
        if (!stream->readers && count <= buffer->key_capacity - (stream->accepted - oldest) &&
            count <= buffer->value_capacity - (stream->accepted - stream->committed[2])) {
            for (uint32_t token = 0; token < count; token++) {
                memcpy(tail_row(buffer, index, CEMU_KV_PAGE_TOKEN_K, stream->accepted + token),
                    (const uint8_t *)key + (uint64_t)token * buffer->row_bytes, buffer->row_bytes);
                memcpy(tail_row(buffer, index, CEMU_KV_PAGE_TOKEN_V, stream->accepted + token),
                    (const uint8_t *)value + (uint64_t)token * buffer->row_bytes, buffer->row_bytes);
            }
            stream->accepted += count;
            enqueue_ready(buffer, index);
            g_cond_signal(&buffer->wake);
            status = CEMU_KV_OK;
        }
    }
    g_mutex_unlock(&buffer->lock);
    return status;
}

CemuKvStatus native_kv_buffer_read(NativeKvBuffer *buffer,
    const CemuKvCoordinate *coordinate, void *output, size_t bytes,
    uint64_t start_ns, CemuKvIoResult *result)
{
    uint32_t index;
    NativeKvStream *stream;
    CemuKvStatus status = CEMU_KV_INVALID;
    CemuKvIoResult candidate = { .start_ns = start_ns, .ready_ns = start_ns };

    if (!buffer || !coordinate || !output || !result || start_ns > INT64_MAX ||
        !stream_index(buffer, coordinate, &index) ||
        (coordinate->kind == CEMU_KV_PAGE_CHANNEL_K ? bytes != buffer->layout.geometry.element_bytes :
            (bytes != buffer->row_bytes || coordinate->channel))) {
        return CEMU_KV_INVALID;
    }
    g_mutex_lock(&buffer->lock);
    stream = &buffer->streams[index];
    if (buffer->stats.failed) {
        status = CEMU_KV_IO_FAILED;
    } else if (coordinate->token >= stream->accepted) {
        status = CEMU_KV_NOT_FOUND;
    } else if (coordinate->token >= stream->committed[coordinate->kind]) {
        memcpy(output, tail_row(buffer, index, coordinate->kind, coordinate->token) +
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
        uint32_t stream = ((uint64_t)coordinate.batch * geometry->layers + coordinate.layer) *
                          geometry->heads + coordinate.head;
        uint64_t committed = buffer->streams[stream].committed[coordinate.kind];
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
                        tail_row(buffer, stream, coordinate.kind, coordinate.token + token) +
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
        uint32_t stream, unit;

        if (!ranges[range].tokens || !stream_index(buffer, coordinate, &stream) ||
            ranges[range].tokens > buffer->layout.geometry.max_tokens - coordinate->token ||
            (coordinate->kind != CEMU_KV_PAGE_CHANNEL_K && coordinate->channel)) {
            goto done;
        }
        if (coordinate->token + ranges[range].tokens > buffer->streams[stream].accepted) {
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

CemuKvStatus native_kv_buffer_stats(NativeKvBuffer *buffer,
    const CemuKvCoordinate *coordinate, NativeKvBufferStats *stats)
{
    uint32_t index;

    if (!buffer || !stats || !stream_index(buffer, coordinate, &index)) {
        return CEMU_KV_INVALID;
    }
    g_mutex_lock(&buffer->lock);
    *stats = buffer->stats;
    stats->accepted = buffer->streams[index].accepted;
    memcpy(stats->committed, buffer->streams[index].committed, sizeof(stats->committed));
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
    for (uint32_t index = 0; index < buffer->layout.streams; index++) {
        enqueue_ready(buffer, index);
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
            NativeKvStream *stream = &buffer->streams[
                ((uint64_t)batch * geometry->layers + layer) * geometry->heads + head];

            if (stream->accepted != tokens) {
                goto done;
            }
        }
    }
    for (uint32_t batch = batch_first; batch < batch_first + batches; batch++) {
        for (uint32_t head = 0; head < geometry->heads; head++) {
            buffer->streams[((uint64_t)batch * geometry->layers + layer) * geometry->heads + head].readers++;
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
            NativeKvStream *stream = &buffer->streams[
                ((uint64_t)batch * geometry->layers + lease->layer) * geometry->heads + head];

            g_assert(stream->readers);
            stream->readers--;
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
    g_free(buffer->streams);
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
