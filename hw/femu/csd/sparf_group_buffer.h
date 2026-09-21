#ifndef CEMU_SPARF_GROUP_BUFFER_H
#define CEMU_SPARF_GROUP_BUFFER_H

#include "sparf_group.h"

typedef struct SparfGroupPending {
    uint32_t stream, kind, count;
    uint64_t position, page;
} SparfGroupPending;

/* Reassign only pages of an already selected batch.  Keep pending order for
 * publication/re-enqueue, batch boundaries and the transferred sector set.
 * Each channel-K group stays contiguous because readers use base + column. */
static void sparf_group_place_batch(GArray *pending, uint64_t first_page,
                                   uint32_t channel_pages)
{
    uint64_t cursor = first_page;
    for (uint32_t kind = 0; kind < 3; ++kind) {
        for (guint index = 0; index < pending->len; ++index) {
            SparfGroupPending *item = &g_array_index(pending, SparfGroupPending, index);
            if (item->kind != kind) continue;
            item->page = cursor;
            cursor += kind == CEMU_SPARF_GROUP_CHANNEL_K ? channel_pages : 1;
        }
    }
}

static void sparf_group_buffer_enqueue(SparfGroupBuffer *buffer, uint32_t index)
{
    SparfGroupStream *stream = &buffer->streams[index];
    for (uint32_t kind = 0; kind < 3; ++kind) {
        uint32_t capacity = kind == CEMU_SPARF_GROUP_CHANNEL_K ? buffer->key_capacity / 2 : buffer->value_capacity / 2;
        uint64_t available = stream->accepted - stream->committed[kind];
        if (!(stream->queued & (1U << kind)) && available && (available >= capacity || buffer->draining)) {
            buffer->ready[buffer->ready_tail] = index * 3 + kind;
            buffer->ready_tail = (buffer->ready_tail + 1) % (buffer->count * 3);
            ++buffer->ready_count;
            stream->queued |= 1U << kind;
        }
    }
}

static void *sparf_group_buffer_worker(void *opaque)
{
    SparfGroupSession *session = opaque;
    SparfGroupBuffer *buffer = session->buffer;
    uint32_t dimension = session->geometry.head_dim;
    uint32_t element = session->geometry.element_bytes;
    uint32_t channels = session->geometry.channels_per_page;
    uint32_t page_bytes = session->geometry.page_bytes;
    GArray *pending = g_array_sized_new(FALSE, FALSE, sizeof(SparfGroupPending), buffer->batch_pages);
    g_mutex_lock(&buffer->lock);
    while (!buffer->stop) {
        if (session->failed || !buffer->ready_count) {
            if (buffer->draining && !session->failed) buffer->sealed = true;
            g_cond_broadcast(&buffer->wake);
            g_cond_wait(&buffer->wake, &buffer->lock);
            continue;
        }
        g_array_set_size(pending, 0);
        uint32_t total_pages = 0;
        uint64_t first_page = session->cursor;
        while (buffer->ready_count && !session->failed) {
            uint32_t task = buffer->ready[buffer->ready_head];
            uint32_t stream_index = task / 3, kind = task % 3;
            SparfGroupStream *stream = &buffer->streams[stream_index];
            uint32_t capacity = kind == CEMU_SPARF_GROUP_CHANNEL_K ? buffer->key_capacity / 2 : buffer->value_capacity / 2;
            uint64_t position = stream->committed[kind];
            uint64_t available = stream->accepted - position;
            uint32_t count = MIN(available, capacity);
            uint32_t pages = kind == CEMU_SPARF_GROUP_CHANNEL_K ? dimension / channels : 1;
            if (pages > buffer->batch_pages - total_pages) break;
            if (pages > session->pages->len - session->cursor) {
                session->failed = true;
                break;
            }
            SparfGroupPending item = { .stream = stream_index, .kind = kind, .count = count,
                .position = position, .page = session->cursor };
            g_array_append_val(pending, item);
            session->cursor += pages;
            total_pages += pages;
            buffer->ready_head = (buffer->ready_head + 1) % (buffer->count * 3);
            --buffer->ready_count;
        }
        if (session->failed) continue;
        if (buffer->kind_major) sparf_group_place_batch(pending, first_page, dimension / channels);
        GHashTable *selected = g_hash_table_new(g_direct_hash, g_direct_equal);
        for (uint32_t page = 0; page < total_pages; ++page)
            g_hash_table_add(selected, GUINT_TO_POINTER((guint)(first_page + page + 1)));
        SparfWritebackPlan *plan = sparf_group_plan_into(session, selected, buffer->scratch);
        g_hash_table_destroy(selected);
        if (!plan) {
            session->failed = true;
            continue;
        }
        memset(buffer->scratch, 0, (uint64_t)total_pages * page_bytes);
        for (guint index = 0; index < pending->len; ++index) {
            SparfGroupPending item = g_array_index(pending, SparfGroupPending, index);
            SparfGroupStream *stream = &buffer->streams[item.stream];
            uint32_t kind = item.kind;
            uint32_t capacity = kind == CEMU_SPARF_GROUP_CHANNEL_K ? buffer->key_capacity / 2 : buffer->value_capacity / 2;
            uint32_t pages = kind == CEMU_SPARF_GROUP_CHANNEL_K ? dimension / channels : 1;
            for (uint32_t page = 0; page < pages; ++page) {
                uint8_t *destination = sparf_group_page(session, plan, item.page + page);
                for (uint32_t token = 0; token < item.count; ++token) {
                    uint8_t *source = sparf_group_tail(session, stream, kind, item.position + token);
                    if (kind != CEMU_SPARF_GROUP_CHANNEL_K) {
                        memcpy(destination + (uint64_t)token * dimension * element, source, dimension * element);
                    } else {
                        for (uint32_t channel = 0; channel < channels; ++channel) {
                            memcpy(destination + ((uint64_t)channel * capacity + token) * element,
                                   source + (page * channels + channel) * element, element);
                        }
                    }
                }
            }
        }
        for (guint index = 0; index < pending->len; ++index) {
            SparfGroupPending item = g_array_index(pending, SparfGroupPending, index);
            uint32_t pages = item.kind == CEMU_SPARF_GROUP_CHANNEL_K ?
                dimension / channels : 1;
            for (uint32_t page_offset = 0; page_offset < pages; ++page_offset) {
                CemuKvPageTag *tag = &g_array_index(session->page_tags, CemuKvPageTag,
                                                    item.page + page_offset);
                tag->stream = item.stream;
                tag->group = item.position /
                    (item.kind == CEMU_SPARF_GROUP_CHANNEL_K ? buffer->key_capacity / 2 :
                                                                buffer->value_capacity / 2);
                tag->kind = item.kind;
                tag->page_in_group = page_offset;
            }
        }
        if (!sparf_group_refresh_plan_tags(session, plan)) {
            sparf_wb_free(plan);
            session->failed = true;
            continue;
        }
        if (buffer->dirty) buffer->dirty(buffer->opaque, buffer->scratch, (uint64_t)total_pages * page_bytes);
        g_mutex_unlock(&buffer->lock);
        uint64_t start = 0, end = 0;
        bool success = buffer->transfer(buffer->opaque, plan, true, &start, &end);
        sparf_wb_free(plan);
        g_mutex_lock(&buffer->lock);
        if (success) {
            for (guint index = 0; index < pending->len; ++index) {
                SparfGroupPending item = g_array_index(pending, SparfGroupPending, index);
                SparfGroupStream *stream = &buffer->streams[item.stream];
                uint32_t capacity = item.kind == CEMU_SPARF_GROUP_CHANNEL_K ? buffer->key_capacity / 2 : buffer->value_capacity / 2;
                SparfGroupKey *key = g_new0(SparfGroupKey, 1);
                *key = (SparfGroupKey){ .stream = item.stream, .kind = item.kind, .group = item.position / capacity };
                uint64_t *page = g_new(uint64_t, 1);
                *page = item.page;
                g_hash_table_insert(session->directory, key, page);
                stream->committed[item.kind] += item.count;
                stream->queued &= ~(1U << item.kind);
                sparf_group_buffer_enqueue(buffer, item.stream);
            }
            buffer->write_bytes += (uint64_t)total_pages * page_bytes;
            ++buffer->statistics.write_batches;
            buffer->statistics.max_write_pages = MAX(buffer->statistics.max_write_pages, total_pages);
        } else session->failed = true;
        g_cond_broadcast(&buffer->wake);
    }
    g_mutex_unlock(&buffer->lock);
    g_array_free(pending, TRUE);
    return NULL;
}

static bool sparf_group_buffer_drain(SparfGroupSession *session)
{
    SparfGroupBuffer *buffer = session->buffer;
    buffer->draining = true;
    for (uint32_t index = 0; index < buffer->count; ++index) sparf_group_buffer_enqueue(buffer, index);
    g_cond_broadcast(&buffer->wake);
    while (!buffer->sealed && !session->failed) g_cond_wait(&buffer->wake, &buffer->lock);
    return !session->failed;
}

static void sparf_group_buffer_cleanup(SparfGroupSession *session)
{
    SparfGroupBuffer *buffer = session->buffer;
    g_mutex_lock(&buffer->lock);
    sparf_group_buffer_drain(session);
    buffer->stop = true;
    g_cond_broadcast(&buffer->wake);
    g_mutex_unlock(&buffer->lock);
    g_thread_join(buffer->worker);
    if (buffer->release) buffer->release(buffer->opaque);
    g_mutex_clear(&buffer->lock);
    g_cond_clear(&buffer->wake);
    g_free(buffer->streams);
    g_free(buffer->ready);
    g_free(buffer);
    session->buffer = NULL;
}

static SparfGroupSession *sparf_group_buffer_register(const void *metadata, uint32_t length,
    uint64_t nvm_bytes, uint32_t page_bytes, uint8_t *storage, uint64_t storage_bytes,
    SparfWritebackTransfer transfer, void (*dirty)(void *, uint8_t *, uint64_t),
    GDestroyNotify release, void *opaque)
{
    struct cemu_sparf_group_header header;
    if (!metadata || length < sizeof(header) || !storage || !transfer) return NULL;
    memcpy(&header, metadata, sizeof(header));
    if (header.version != CEMU_SPARF_GROUP_VERSION || header.operation != CEMU_SPARF_GROUP_REGISTER_BUFFER ||
        !header.stream || header.stream > CEMU_SPARF_GROUP_MAX_STREAMS) return NULL;
    const char *order = g_getenv("CEMU_SPARF_GROUP_WRITE_ORDER");
    bool kind_major = order && !strcmp(order, "kind");
    if (order && *order && strcmp(order, "fifo") && !kind_major) return NULL;
    uint32_t streams = header.stream;
    uint64_t requested_pages = header.argument;
    header.stream = 0;
    header.argument = 0;
    header.operation = CEMU_SPARF_GROUP_REGISTER;
    SparfGroupSession *session = sparf_group_register(&header, metadata, length, nvm_bytes, page_bytes);
    if (!session) return NULL;
    uint64_t batch_pages = requested_pages ? requested_pages : header.head_dim / header.channels_per_page;
    if (batch_pages < header.head_dim / header.channels_per_page ||
        batch_pages > CEMU_SPARF_WRITEBACK_MAX_BYTES / page_bytes) {
        sparf_group_free(session);
        return NULL;
    }
    uint32_t stride = header.head_dim * header.element_bytes;
    uint32_t key_capacity = 2 * page_bytes / (header.channels_per_page * header.element_bytes);
    uint32_t value_capacity = 2 * page_bytes / stride;
    uint64_t per_stream = (uint64_t)(key_capacity + value_capacity) * stride;
    uint64_t scratch_bytes = batch_pages * page_bytes;
    uint64_t required = streams * per_stream + scratch_bytes;
    if (storage_bytes < required || scratch_bytes > CEMU_SPARF_WRITEBACK_MAX_BYTES) {
        sparf_group_free(session);
        return NULL;
    }
    SparfGroupBuffer *buffer = g_new0(SparfGroupBuffer, 1);
    buffer->streams = g_new0(SparfGroupStream, streams);
    buffer->count = streams;
    buffer->kind_major = kind_major;
    buffer->batch_pages = batch_pages;
    buffer->ready = g_new(uint32_t, streams * 3);
    buffer->key_capacity = key_capacity;
    buffer->value_capacity = value_capacity;
    buffer->bytes = required;
    buffer->scratch = storage + streams * per_stream;
    buffer->transfer = transfer;
    buffer->dirty = dirty;
    buffer->release = release;
    buffer->opaque = opaque;
    for (uint32_t index = 0; index < streams; ++index) {
        buffer->streams[index].key = storage + index * per_stream;
        buffer->streams[index].value = buffer->streams[index].key + (uint64_t)key_capacity * stride;
    }
    g_mutex_init(&buffer->lock);
    g_cond_init(&buffer->wake);
    session->buffer = buffer;
    session->buffer_cleanup = sparf_group_buffer_cleanup;
    buffer->worker = g_thread_new("cemu-group-flush", sparf_group_buffer_worker, session);
    return session;
}

static bool sparf_group_buffer_append(SparfGroupSession *session,
    const struct cemu_sparf_group_header *header, uint32_t length, const uint8_t *input, uint64_t input_bytes)
{
    SparfGroupBuffer *buffer = session->buffer;
    SparfGroupStream *stream = &buffer->streams[header->stream];
    uint32_t stride = session->geometry.head_dim * session->geometry.element_bytes;
    uint64_t bytes = (uint64_t)header->count * stride;
    if (length != sizeof(*header) || header->kind || !header->count || !input ||
        buffer->draining || stream->readers || header->position != stream->accepted ||
        header->position > UINT64_MAX - header->count || header->argument > input_bytes ||
        bytes > (input_bytes - header->argument) / 2) return false;
    uint32_t done = 0;
    while (done < header->count && !session->failed) {
        uint64_t key_oldest = MIN(stream->committed[0], stream->committed[1]);
        uint64_t key_free = buffer->key_capacity - (stream->accepted - key_oldest);
        uint64_t value_free = buffer->value_capacity - (stream->accepted - stream->committed[2]);
        uint64_t count = MIN(MIN(key_free, value_free), header->count - done);
        count = MIN(count, buffer->key_capacity - stream->accepted % buffer->key_capacity);
        count = MIN(count, buffer->value_capacity - stream->accepted % buffer->value_capacity);
        if (!count) {
            g_cond_wait(&buffer->wake, &buffer->lock);
            continue;
        }
        uint8_t *key = sparf_group_tail(session, stream, CEMU_SPARF_GROUP_TOKEN_K, stream->accepted);
        uint8_t *value = sparf_group_tail(session, stream, CEMU_SPARF_GROUP_TOKEN_V, stream->accepted);
        memcpy(key, input + header->argument + (uint64_t)done * stride, count * stride);
        memcpy(value, input + header->argument + bytes + (uint64_t)done * stride, count * stride);
        if (buffer->dirty) {
            buffer->dirty(buffer->opaque, key, count * stride);
            buffer->dirty(buffer->opaque, value, count * stride);
        }
        stream->accepted += count;
        sparf_group_buffer_enqueue(buffer, header->stream);
        done += count;
        g_cond_broadcast(&buffer->wake);
    }
    return !session->failed;
}

static bool sparf_group_buffer_append_batch(SparfGroupSession *session,
    const struct cemu_sparf_group_header *header, uint32_t length, const uint8_t *input, uint64_t input_bytes)
{
    SparfGroupBuffer *buffer = session->buffer;
    uint64_t streams = header->argument;
    uint64_t stride = session->geometry.head_dim * session->geometry.element_bytes;
    if (length != sizeof(*header) || header->kind || !header->count || !input || !streams ||
        streams > buffer->count - header->stream || buffer->draining ||
        header->position > UINT64_MAX - header->count ||
        (uint64_t)header->count > input_bytes / 2 / stride / streams) return false;
    for (uint32_t index = 0; index < streams; ++index) {
        SparfGroupStream *stream = &buffer->streams[header->stream + index];
        if (stream->readers || stream->accepted != header->position) return false;
    }
    uint32_t done = 0;
    while (done < header->count && !session->failed) {
        uint64_t count = header->count - done;
        uint64_t position = header->position + done;
        count = MIN(count, buffer->key_capacity - position % buffer->key_capacity);
        count = MIN(count, buffer->value_capacity - position % buffer->value_capacity);
        for (uint32_t index = 0; index < streams; ++index) {
            SparfGroupStream *stream = &buffer->streams[header->stream + index];
            uint64_t oldest_key = MIN(stream->committed[0], stream->committed[1]);
            count = MIN(count, buffer->key_capacity - (position - oldest_key));
            count = MIN(count, buffer->value_capacity - (position - stream->committed[2]));
        }
        if (!count) {
            g_cond_wait(&buffer->wake, &buffer->lock);
            continue;
        }
        for (uint32_t index = 0; index < streams; ++index) {
            SparfGroupStream *stream = &buffer->streams[header->stream + index];
            uint8_t *key = sparf_group_tail(session, stream, CEMU_SPARF_GROUP_TOKEN_K, position);
            uint8_t *value = sparf_group_tail(session, stream, CEMU_SPARF_GROUP_TOKEN_V, position);
            for (uint32_t token = 0; token < count; ++token) {
                uint64_t offset = ((uint64_t)(done + token) * streams + index) * stride;
                memcpy(key + token * stride, input + offset, stride);
                memcpy(value + token * stride, input + (uint64_t)header->count * streams * stride + offset, stride);
            }
            if (buffer->dirty) {
                buffer->dirty(buffer->opaque, key, count * stride);
                buffer->dirty(buffer->opaque, value, count * stride);
            }
            stream->accepted += count;
            sparf_group_buffer_enqueue(buffer, header->stream + index);
        }
        done += count;
        g_cond_broadcast(&buffer->wake);
    }
    return !session->failed;
}

static bool sparf_group_buffer_execute(SparfGroupSession **session_pointer, const void *metadata, uint32_t length,
    uint8_t *input, uint64_t input_bytes, uint64_t nvm_bytes, uint32_t page_bytes,
    SparfWritebackTransfer transfer, void *opaque, struct cemu_sparf_group_result *result,
    struct cemu_sparf_group_progress *progress)
{
    SparfGroupSession *session = *session_pointer;
    memset(progress, 0, sizeof(*progress));
    memset(result, 0, sizeof(*result));
    result->version = CEMU_SPARF_GROUP_VERSION;
    if (!session || !session->buffer) return sparf_group_execute(session_pointer, metadata, length,
        input, input_bytes, nvm_bytes, page_bytes, transfer, opaque, result);
    struct cemu_sparf_group_header header;
    if (!metadata || length < sizeof(header)) return false;
    memcpy(&header, metadata, sizeof(header));
    if (header.version != CEMU_SPARF_GROUP_VERSION || header.cookie != session->geometry.cookie ||
        header.head_dim || header.channels_per_page || header.element_bytes || header.page_bytes ||
        header.stream >= session->buffer->count) return false;
    SparfGroupBuffer *buffer = session->buffer;
    g_mutex_lock(&buffer->lock);
    bool success = false;
    if (header.operation == CEMU_SPARF_GROUP_STATS) {
        if (length != sizeof(header) || header.kind || header.count || header.position || header.argument ||
            header.stream || !input || input_bytes < sizeof(buffer->statistics)) goto done;
        memcpy(input, &buffer->statistics, sizeof(buffer->statistics));
        result->output_bytes = sizeof(buffer->statistics);
        success = true;
    } else if (header.operation == CEMU_SPARF_GROUP_DRAIN || header.operation == CEMU_SPARF_GROUP_STATUS ||
        header.operation == CEMU_SPARF_GROUP_CLOSE) {
        if (length != sizeof(header) || header.kind || header.count || header.position || header.argument ||
            (header.operation != CEMU_SPARF_GROUP_STATUS && header.stream)) goto done;
        if (header.operation != CEMU_SPARF_GROUP_STATUS && buffer->users) goto done;
        success = header.operation == CEMU_SPARF_GROUP_STATUS || sparf_group_buffer_drain(session);
    } else if (!session->failed) {
        if (header.operation == CEMU_SPARF_GROUP_APPEND) {
            success = sparf_group_buffer_append(session, &header, length, input, input_bytes);
        } else if (header.operation == CEMU_SPARF_GROUP_APPEND_BATCH) {
            success = sparf_group_buffer_append_batch(session, &header, length, input, input_bytes);
        } else success = sparf_group_execute(session_pointer, metadata, length, input, input_bytes,
                                            nvm_bytes, page_bytes, transfer, opaque, result);
        if (success && (header.operation == CEMU_SPARF_GROUP_APPEND || header.operation == CEMU_SPARF_GROUP_APPEND_BATCH)) {
            uint64_t streams = header.operation == CEMU_SPARF_GROUP_APPEND_BATCH ? header.argument : 1;
            ++buffer->statistics.append_commands;
            buffer->statistics.appended_streams += streams;
            buffer->statistics.appended_tokens += streams * header.count;
        }
    }
done:
    progress->accepted = buffer->streams[header.stream].accepted;
    memcpy(progress->committed, buffer->streams[header.stream].committed, sizeof(progress->committed));
    progress->sealed = buffer->sealed;
    progress->failed = session->failed;
    progress->write_bytes = buffer->write_bytes;
    progress->buffer_bytes = buffer->bytes;
    result->pages_used = session->cursor;
    result->groups = g_hash_table_size(session->directory);
    g_cond_broadcast(&buffer->wake);
    bool close = !buffer->users && header.operation == CEMU_SPARF_GROUP_CLOSE && length == sizeof(header) &&
        !header.kind && !header.count && !header.position && !header.argument && !header.stream;
    g_mutex_unlock(&buffer->lock);
    if (close) {
        sparf_group_free(session);
        *session_pointer = NULL;
        return success;
    }
    return success;
}

#endif
