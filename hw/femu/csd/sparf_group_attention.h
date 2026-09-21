#ifndef CEMU_SPARF_GROUP_ATTENTION_H
#define CEMU_SPARF_GROUP_ATTENTION_H

#include "sparf_group_buffer.h"

static bool sparf_group_bind(SparfGroupSession *session,
    const struct cemu_sparf_workflow_header *header, const struct cemu_sparf_group_binding *binding)
{
    if (!session || !session->buffer || binding->cookie != session->geometry.cookie ||
        header->head_dim != session->geometry.head_dim ||
        (header->dtype == CEMU_SPARF_FLOAT16 ? 2 : 4) != session->geometry.element_bytes ||
        session->geometry.page_bytes != 4096) return false;
    uint64_t streams = (uint64_t)header->batch_size * header->num_kv_heads;
    if (!streams || binding->stream_base > session->buffer->count ||
        streams > session->buffer->count - binding->stream_base) return false;
    SparfGroupBuffer *buffer = session->buffer;
    g_mutex_lock(&buffer->lock);
    bool success = !session->failed && (!buffer->draining || buffer->sealed);
    for (uint64_t offset = 0; success && offset < streams; ++offset) {
        success = buffer->streams[binding->stream_base + offset].accepted == header->valid_tokens;
    }
    if (success) {
        if (!session->attention_map) {
            session->attention_map = g_array_new(FALSE, FALSE, sizeof(struct cemu_sparf_workflow_extent));
            for (guint page = 0; page < session->pages->len; ++page) {
                uint64_t sector = g_array_index(session->pages, uint64_t, page);
                GArray *maps = session->attention_map;
                if (maps->len) {
                    struct cemu_sparf_workflow_extent *last = &g_array_index(maps, struct cemu_sparf_workflow_extent, maps->len - 1);
                    if (last->slba + last->nlb == sector && last->nlb <= 65527) {
                        last->nlb += 8;
                        continue;
                    }
                }
                struct cemu_sparf_workflow_extent extent = { .slba = sector, .nlb = 8 };
                g_array_append_val(maps, extent);
            }
        }
        ++buffer->users;
        for (uint64_t offset = 0; offset < streams; ++offset) ++buffer->streams[binding->stream_base + offset].readers;
    }
    g_mutex_unlock(&buffer->lock);
    return success;
}

static void sparf_group_unbind(SparfGroupSession *session,
    const struct cemu_sparf_workflow_header *header, const struct cemu_sparf_group_binding *binding)
{
    SparfGroupBuffer *buffer = session->buffer;
    g_mutex_lock(&buffer->lock);
    --buffer->users;
    for (uint64_t offset = 0; offset < (uint64_t)header->batch_size * header->num_kv_heads; ++offset)
        --buffer->streams[binding->stream_base + offset].readers;
    g_cond_broadcast(&buffer->wake);
    g_mutex_unlock(&buffer->lock);
}

static bool sparf_start_grouped_read(ComputeJob *job, SparfGroupSession *session,
    const struct cemu_sparf_workflow_header *header, const uint32_t *indices, uint32_t kind,
    uint32_t destination_range, struct rte_ring *ring, uint64_t start,
    SparfRead *read, FemuReadObservation *observation)
{
    if (!session || !session->buffer) return false;
    struct cemu_sparf_group_binding binding;
    memcpy(&binding, (uint8_t *)job->args.data_buffer + sizeof(*header), sizeof(binding));
    uint32_t element = session->geometry.element_bytes, dimension = header->head_dim;
    uint32_t page_bytes = session->geometry.page_bytes;
    uint32_t channels = session->geometry.channels_per_page;
    uint32_t capacity = page_bytes / ((kind == CEMU_SPARF_GROUP_CHANNEL_K ? channels : dimension) * element);
    uint32_t ranks = kind == CEMU_SPARF_GROUP_CHANNEL_K ? header->top_r : header->top_k;
    uint64_t vectors = (uint64_t)header->batch_size * header->num_query_heads;
    uint64_t bytes = kind == CEMU_SPARF_GROUP_CHANNEL_K ? vectors * ranks * header->valid_tokens * element :
        vectors * (ranks + (kind == CEMU_SPARF_GROUP_TOKEN_V)) * dimension * element;
    if (bytes > CEMU_SPARF_WRITEBACK_MAX_BYTES || bytes > (uint64_t)job->args.mr_len[destination_range] ||
        backend_cuda_prepare_host(job->mr_backend[destination_range], job->args.mr_addr[destination_range], bytes)) return false;
    uint8_t *destination = job->args.mr_addr[destination_range];
    GArray *spans = g_array_new(FALSE, FALSE, sizeof(SparfLogicalSpan));
    GArray *maps = session->attention_map;
    SparfGroupBuffer *buffer = session->buffer;
    bool success = false;
    g_mutex_lock(&buffer->lock);
    if (session->failed) goto done;
    for (uint64_t vector = 0; vector < vectors; ++vector) {
        uint64_t stream_id = binding.stream_base + vector / header->num_query_heads * header->num_kv_heads +
            (vector % header->num_query_heads) / (header->num_query_heads / header->num_kv_heads);
        SparfGroupStream *stream = &buffer->streams[stream_id];
        for (uint32_t rank = 0; rank < ranks; ++rank) {
            uint32_t chosen = indices[vector * ranks + rank];
            if (kind == CEMU_SPARF_GROUP_CHANNEL_K ? chosen >= dimension : chosen >= header->valid_tokens) goto done;
            uint64_t first = kind == CEMU_SPARF_GROUP_CHANNEL_K ? 0 : chosen;
            uint64_t end = kind == CEMU_SPARF_GROUP_CHANNEL_K ? header->valid_tokens : first + 1;
            for (uint64_t token = first; token < end;) {
                uint64_t output = kind == CEMU_SPARF_GROUP_CHANNEL_K ?
                    ((vector * ranks + rank) * header->valid_tokens + token) * element :
                    (vector * (ranks + (kind == CEMU_SPARF_GROUP_TOKEN_V)) + rank) * dimension * element;
                if (token >= stream->committed[kind]) {
                    uint8_t *source = sparf_group_tail(session, stream, kind, token);
                    if (observation) {
                        observation->tail_bytes += kind == CEMU_SPARF_GROUP_CHANNEL_K ? element : dimension * element;
                        observation->tail_copies++;
                    }
                    memcpy(destination + output, source + (kind == CEMU_SPARF_GROUP_CHANNEL_K ? chosen * element : 0),
                           kind == CEMU_SPARF_GROUP_CHANNEL_K ? element : dimension * element);
                    ++token;
                } else {
                    uint64_t *page = sparf_group_lookup(session, stream_id, kind, token / capacity);
                    if (!page) goto done;
                    uint64_t count = MIN(end - token, capacity - token % capacity);
                    count = MIN(count, stream->committed[kind] - token);
                    uint64_t ordinal = *page + (kind == CEMU_SPARF_GROUP_CHANNEL_K ? chosen / channels : 0);
                    uint64_t within = kind == CEMU_SPARF_GROUP_CHANNEL_K ?
                        ((chosen % channels) * capacity + token % capacity) * element : token % capacity * dimension * element;
                    SparfLogicalSpan span = { .offset = ordinal * page_bytes + within,
                        .size = kind == CEMU_SPARF_GROUP_CHANNEL_K ? count * element : dimension * element,
                        .destination_offset = output };
                    g_array_append_val(spans, span);
                    token += count;
                }
            }
        }
    }
    success = true;
done:
    g_mutex_unlock(&buffer->lock);
    if (success) {
        backend_cuda_mark_host_dirty(job->mr_backend[destination_range], destination, bytes);
        success = sparf_start_read(job, ring, (void *)maps->data, maps->len, spans,
                                  destination_range, start, read, observation);
    } else g_array_free(spans, TRUE);
    return success;
}

#endif
