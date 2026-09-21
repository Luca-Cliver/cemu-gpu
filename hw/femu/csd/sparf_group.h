#ifndef CEMU_SPARF_GROUP_H
#define CEMU_SPARF_GROUP_H

#include "sparf_writeback.h"
#include "hw/femu/sparf_group_abi.h"

typedef struct SparfGroupKey {
    uint64_t stream;
    uint64_t group;
    uint32_t kind;
} SparfGroupKey;

typedef struct SparfGroupStream {
    uint32_t readers;
    uint32_t queued;
    uint64_t accepted;
    uint64_t committed[3];
    uint8_t *key;
    uint8_t *value;
} SparfGroupStream;

typedef struct SparfGroupBuffer {
    uint32_t users;
    GMutex lock;
    GCond wake;
    GThread *worker;
    SparfGroupStream *streams;
    uint32_t count;
    uint32_t *ready;
    uint32_t ready_head, ready_tail, ready_count;
    uint32_t batch_pages;
    uint32_t key_capacity;
    uint32_t value_capacity;
    uint64_t bytes;
    uint64_t write_bytes;
    struct cemu_sparf_group_statistics statistics;
    uint8_t *scratch;
    bool kind_major;
    bool draining;
    bool sealed;
    bool stop;
    SparfWritebackTransfer transfer;
    void (*dirty)(void *, uint8_t *, uint64_t);
    GDestroyNotify release;
    void *opaque;
} SparfGroupBuffer;

typedef struct SparfGroupSession {
    struct cemu_sparf_group_header geometry;
    GArray *pages;
    GArray *page_tags;
    GHashTable *directory;
    GArray *attention_map;
    uint64_t cursor;
    bool failed;
    SparfGroupBuffer *buffer;
    void (*buffer_cleanup)(struct SparfGroupSession *);
} SparfGroupSession;

static guint sparf_group_hash(gconstpointer pointer)
{
    const SparfGroupKey *key = pointer;
    uint64_t mixed = key->stream * UINT64_C(0x9e3779b97f4a7c15) ^
                     key->group * UINT64_C(0xbf58476d1ce4e5b9) ^
                     (uint64_t)key->kind * UINT64_C(0x94d049bb133111eb);
    mixed ^= mixed >> 30;
    mixed *= UINT64_C(0xbf58476d1ce4e5b9);
    mixed ^= mixed >> 27;
    mixed *= UINT64_C(0x94d049bb133111eb);
    mixed ^= mixed >> 31;
    return (guint)(mixed ^ (mixed >> 32));
}

static gboolean sparf_group_equal(gconstpointer left, gconstpointer right)
{
    const SparfGroupKey *first = left, *second = right;
    return first->stream == second->stream && first->group == second->group && first->kind == second->kind;
}

static void sparf_group_free(SparfGroupSession *session)
{
    if (!session) return;
    if (session->buffer_cleanup) session->buffer_cleanup(session);
    g_array_free(session->pages, TRUE);
    if (session->page_tags) g_array_free(session->page_tags, TRUE);
    g_hash_table_destroy(session->directory);
    if (session->attention_map) g_array_free(session->attention_map, TRUE);
    g_free(session);
}

static SparfGroupSession *sparf_group_register(const struct cemu_sparf_group_header *header,
                                              const void *metadata, uint32_t length,
                                              uint64_t nvm_bytes, uint32_t page_bytes)
{
    if (!header->cookie || header->kind || header->stream || header->position || header->argument ||
        !header->count || header->count > CEMU_SPARF_GROUP_MAX_EXTENTS ||
        length != sizeof(*header) + (uint64_t)header->count * sizeof(struct cemu_sparf_group_extent) ||
        header->page_bytes != page_bytes || !page_bytes || page_bytes % 512 || page_bytes > 65536 ||
        !header->head_dim || header->head_dim > 4096 || !header->channels_per_page ||
        header->head_dim % header->channels_per_page ||
        (header->element_bytes != 2 && header->element_bytes != 4) ||
        page_bytes % (header->head_dim * header->element_bytes) ||
        page_bytes % (header->channels_per_page * header->element_bytes)) return NULL;
    SparfGroupSession *session = g_new0(SparfGroupSession, 1);
    session->geometry = *header;
    session->pages = g_array_new(FALSE, FALSE, sizeof(uint64_t));
    session->page_tags = g_array_sized_new(FALSE, TRUE, sizeof(CemuKvPageTag), 0);
    session->directory = g_hash_table_new_full(sparf_group_hash, sparf_group_equal, g_free, g_free);
    uint64_t sectors_per_page = page_bytes / 512;
    for (uint32_t index = 0; index < header->count; ++index) {
        struct cemu_sparf_group_extent extent;
        memcpy(&extent, (const uint8_t *)metadata + sizeof(*header) + index * sizeof(extent), sizeof(extent));
        if (!extent.sectors || extent.sector % sectors_per_page || extent.sectors % sectors_per_page ||
            extent.sector > nvm_bytes / 512 || extent.sectors > nvm_bytes / 512 - extent.sector ||
            extent.sectors / sectors_per_page > CEMU_SPARF_GROUP_MAX_PAGES - session->pages->len) goto invalid;
        for (uint64_t offset = 0; offset < extent.sectors; offset += sectors_per_page) {
            uint64_t sector = extent.sector + offset;
            g_array_append_val(session->pages, sector);
        }
    }
    GArray *sorted = g_array_sized_new(FALSE, FALSE, sizeof(uint64_t), session->pages->len);
    g_array_append_vals(sorted, session->pages->data, session->pages->len);
    g_array_sort(sorted, sparf_wb_compare);
    bool overlap = false;
    for (guint index = 1; index < sorted->len; ++index) {
        if (g_array_index(sorted, uint64_t, index - 1) == g_array_index(sorted, uint64_t, index)) overlap = true;
    }
    g_array_free(sorted, TRUE);
    if (overlap) goto invalid;
    g_array_set_size(session->page_tags, session->pages->len);
    for (guint index = 0; index < session->page_tags->len; ++index) {
        CemuKvPageTag *tag = &g_array_index(session->page_tags, CemuKvPageTag, index);
        tag->kind = UINT32_MAX;
    }
    return session;
invalid:
    sparf_group_free(session);
    return NULL;
}

static bool sparf_group_refresh_plan_tags(SparfGroupSession *session,
                                           SparfWritebackPlan *plan)
{
    if (!plan || !plan->kv_pages) return true;
    if (!plan->kv_page_ordinals || plan->kv_page_ordinals->len != plan->kv_pages->len) return false;
    uint64_t sectors_per_page = session->geometry.page_bytes / 512;
    if (plan->sectors->len != plan->kv_pages->len * sectors_per_page) return false;
    for (guint index = 0; index < plan->kv_pages->len; ++index) {
        CemuKvPageTag *tag = &g_array_index(plan->kv_pages, CemuKvPageTag, index);
        uint64_t sector = g_array_index(plan->sectors, uint64_t, index * sectors_per_page);
        guint page = g_array_index(plan->kv_page_ordinals, guint, index);
        if (page >= session->pages->len ||
            g_array_index(session->pages, uint64_t, page) != sector) return false;
        CemuKvPageTag current = g_array_index(session->page_tags, CemuKvPageTag, page);
        current.slba = sector;
        current.cookie = session->geometry.cookie;
        *tag = current;
    }
    return true;
}

static gint sparf_group_compare_ordinals(gconstpointer left, gconstpointer right,
                                         gpointer opaque)
{
    SparfGroupSession *session = opaque;
    uint64_t first = g_array_index(session->pages, uint64_t, *(const guint *)left);
    uint64_t second = g_array_index(session->pages, uint64_t, *(const guint *)right);
    return (first > second) - (first < second);
}

static SparfWritebackPlan *sparf_group_plan_into(SparfGroupSession *session, GHashTable *selected, uint8_t *image)
{
    uint32_t page_bytes = session->geometry.page_bytes;
    uint64_t bytes = (uint64_t)g_hash_table_size(selected) * page_bytes;
    if (!bytes || bytes > CEMU_SPARF_WRITEBACK_MAX_BYTES) return NULL;
    SparfWritebackPlan *plan = g_new0(SparfWritebackPlan, 1);
    plan->sectors = g_array_new(FALSE, FALSE, sizeof(uint64_t));
    plan->kv_pages = g_array_new(FALSE, FALSE, sizeof(CemuKvPageTag));
    plan->kv_page_ordinals = g_array_new(FALSE, FALSE, sizeof(guint));
    GHashTableIter iterator;
    gpointer pointer;
    g_hash_table_iter_init(&iterator, selected);
    while (g_hash_table_iter_next(&iterator, &pointer, NULL)) {
        guint page = GPOINTER_TO_UINT(pointer) - 1;
        if (page >= session->pages->len) {
            sparf_wb_free(plan);
            return NULL;
        }
        g_array_append_val(plan->kv_page_ordinals, page);
    }
    g_array_sort_with_data(plan->kv_page_ordinals, sparf_group_compare_ordinals, session);
    for (guint index = 0; index < plan->kv_page_ordinals->len; ++index) {
        guint page = g_array_index(plan->kv_page_ordinals, guint, index);
        uint64_t first = g_array_index(session->pages, uint64_t, page);
        for (uint32_t offset = 0; offset < page_bytes / 512; ++offset) {
            uint64_t sector = first + offset;
            g_array_append_val(plan->sectors, sector);
        }
        CemuKvPageTag tag = g_array_index(session->page_tags, CemuKvPageTag, page);
        tag.slba = first;
        tag.cookie = session->geometry.cookie;
        g_array_append_val(plan->kv_pages, tag);
    }
    plan->borrowed_image = image != NULL;
    plan->image = image ? image : g_try_malloc(bytes);
    if (!plan->image) {
        sparf_wb_free(plan);
        return NULL;
    }
    return plan;
}

static SparfWritebackPlan *sparf_group_plan(SparfGroupSession *session, GHashTable *selected)
{
    return sparf_group_plan_into(session, selected, NULL);
}

static uint8_t *sparf_group_tail(SparfGroupSession *session, SparfGroupStream *stream,
                                uint32_t kind, uint64_t token)
{
    SparfGroupBuffer *buffer = session->buffer;
    uint32_t stride = session->geometry.head_dim * session->geometry.element_bytes;
    return kind == CEMU_SPARF_GROUP_TOKEN_V ? stream->value + token % buffer->value_capacity * stride :
                                            stream->key + token % buffer->key_capacity * stride;
}

static uint8_t *sparf_group_page(SparfGroupSession *session, SparfWritebackPlan *plan, uint64_t page)
{
    uint64_t sector = g_array_index(session->pages, uint64_t, page);
    uint64_t *found = bsearch(&sector, plan->sectors->data, plan->sectors->len, sizeof(sector), sparf_wb_compare);
    return plan->image + (found - (uint64_t *)plan->sectors->data) * 512;
}

static uint64_t *sparf_group_lookup(SparfGroupSession *session, uint64_t stream, uint32_t kind, uint64_t group)
{
    SparfGroupKey key = { .stream = stream, .kind = kind, .group = group };
    return g_hash_table_lookup(session->directory, &key);
}

static uint64_t sparf_group_index(const void *metadata, uint32_t index)
{
    uint64_t value;
    memcpy(&value, (const uint8_t *)metadata + sizeof(struct cemu_sparf_group_header) + index * sizeof(value), sizeof(value));
    return value;
}

static bool sparf_group_execute(SparfGroupSession **session_pointer, const void *metadata, uint32_t length,
                                uint8_t *buffer, uint64_t buffer_bytes, uint64_t nvm_bytes, uint32_t page_bytes,
                                SparfWritebackTransfer transfer, void *opaque,
                                struct cemu_sparf_group_result *result)
{
    memset(result, 0, sizeof(*result));
    result->version = CEMU_SPARF_GROUP_VERSION;
    struct cemu_sparf_group_header header;
    if (!metadata || length < sizeof(header)) return false;
    memcpy(&header, metadata, sizeof(header));
    if (header.version != CEMU_SPARF_GROUP_VERSION || !header.cookie) return false;
    if (header.operation == CEMU_SPARF_GROUP_REGISTER) {
        if (*session_pointer) return false;
        *session_pointer = sparf_group_register(&header, metadata, length, nvm_bytes, page_bytes);
        return *session_pointer != NULL;
    }
    SparfGroupSession *session = *session_pointer;
    if (!session || header.cookie != session->geometry.cookie || header.head_dim || header.channels_per_page ||
        header.element_bytes || header.page_bytes) return false;
    if (header.operation == CEMU_SPARF_GROUP_CLOSE) {
        if (length != sizeof(header) || header.count || header.kind || header.stream || header.position || header.argument) return false;
        sparf_group_free(session);
        *session_pointer = NULL;
        return true;
    }
    if (session->failed || !buffer || !header.count || header.kind > CEMU_SPARF_GROUP_TOKEN_V) return false;
    uint32_t dimension = session->geometry.head_dim, element = session->geometry.element_bytes;
    uint32_t channels = session->geometry.channels_per_page;
    uint32_t token_capacity = page_bytes / (dimension * element);
    uint32_t channel_capacity = page_bytes / (channels * element);
    uint32_t channel_pages = dimension / channels;
    if (header.operation != CEMU_SPARF_GROUP_PUT && header.operation != CEMU_SPARF_GROUP_READ_TOKENS &&
        header.operation != CEMU_SPARF_GROUP_READ_CHANNELS) return false;
    bool write = header.operation == CEMU_SPARF_GROUP_PUT;
    SparfGroupStream *stream = NULL;
    if (session->buffer) {
        if (write || header.stream >= session->buffer->count) return false;
        stream = &session->buffer->streams[header.stream];
    }
    uint64_t group = 0, group_pages = 0, output_bytes = 0;
    GHashTable *selected = g_hash_table_new(g_direct_hash, g_direct_equal);
    SparfWritebackPlan *plan = NULL;
    bool success = false;
    if (write) {
        uint32_t capacity = header.kind == CEMU_SPARF_GROUP_CHANNEL_K ? channel_capacity : token_capacity;
        group_pages = header.kind == CEMU_SPARF_GROUP_CHANNEL_K ? channel_pages : 1;
        if (length != sizeof(header) || header.count != capacity || header.position % capacity ||
            header.position > UINT64_MAX - capacity || header.argument > buffer_bytes ||
            (uint64_t)capacity * dimension * element > buffer_bytes - header.argument ||
            group_pages > session->pages->len - session->cursor) goto done;
        group = header.position / capacity;
        if (sparf_group_lookup(session, header.stream, header.kind, group)) goto done;
        for (uint64_t offset = 0; offset < group_pages; ++offset) {
            g_hash_table_add(selected, GUINT_TO_POINTER((guint)(session->cursor + offset + 1)));
        }
    } else {
        if (header.count > CEMU_SPARF_GROUP_MAX_INDICES || length != sizeof(header) + (uint64_t)header.count * 8) goto done;
        if (header.operation == CEMU_SPARF_GROUP_READ_TOKENS) {
            if (header.kind == CEMU_SPARF_GROUP_CHANNEL_K || header.position || header.argument) goto done;
            output_bytes = (uint64_t)header.count * dimension * element;
            for (uint32_t index = 0; index < header.count; ++index) {
                uint64_t token = sparf_group_index(metadata, index);
                if (stream) {
                    if (token >= stream->accepted) goto done;
                    if (token >= stream->committed[header.kind]) continue;
                }
                uint64_t *page = sparf_group_lookup(session, header.stream, header.kind, token / token_capacity);
                if (!page) goto done;
                g_hash_table_add(selected, GUINT_TO_POINTER((guint)(*page + 1)));
            }
        } else {
            if (header.kind != CEMU_SPARF_GROUP_CHANNEL_K || !header.argument ||
                header.argument > CEMU_SPARF_WRITEBACK_MAX_BYTES / element / header.count ||
                header.position > UINT64_MAX - header.argument) goto done;
            output_bytes = header.argument * header.count * element;
            if (stream && header.position + header.argument > stream->accepted) goto done;
            for (uint32_t index = 0; index < header.count; ++index) {
                uint64_t channel = sparf_group_index(metadata, index);
                if (channel >= dimension) goto done;
                uint64_t first = header.position / channel_capacity;
                uint64_t end = header.position + header.argument;
                if (stream) end = MIN(end, stream->committed[CEMU_SPARF_GROUP_CHANNEL_K]);
                if (end <= header.position) continue;
                uint64_t last = (end - 1) / channel_capacity;
                for (uint64_t current = first; current <= last; ++current) {
                    uint64_t *page = sparf_group_lookup(session, header.stream, header.kind, current);
                    if (!page) goto done;
                    g_hash_table_add(selected, GUINT_TO_POINTER((guint)(*page + channel / channels + 1)));
                }
            }
        }
        if (output_bytes > buffer_bytes || output_bytes > CEMU_SPARF_WRITEBACK_MAX_BYTES) goto done;
    }
    if (g_hash_table_size(selected)) {
        plan = sparf_group_plan(session, selected);
        if (!plan) goto done;
    }
    if (write) {
        const uint8_t *source = buffer + header.argument;
        for (uint32_t page = 0; page < group_pages; ++page) {
            uint8_t *destination = sparf_group_page(session, plan, session->cursor + page);
            if (header.kind != CEMU_SPARF_GROUP_CHANNEL_K) {
                memcpy(destination, source, page_bytes);
            } else {
                for (uint32_t channel = 0; channel < channels; ++channel) {
                    for (uint32_t token = 0; token < channel_capacity; ++token) {
                        memcpy(destination + ((uint64_t)channel * channel_capacity + token) * element,
                               source + ((uint64_t)token * dimension + page * channels + channel) * element, element);
                    }
                }
            }
        }
        for (uint32_t page_offset = 0; page_offset < group_pages; ++page_offset) {
            CemuKvPageTag *tag = &g_array_index(session->page_tags, CemuKvPageTag,
                                                session->cursor + page_offset);
            tag->stream = header.stream;
            tag->group = group;
            tag->kind = header.kind;
            tag->page_in_group = page_offset;
        }
        if (!sparf_group_refresh_plan_tags(session, plan)) goto done;
        uint64_t first_page = session->cursor;
        session->cursor += group_pages;
        if (!transfer(opaque, plan, true, &result->io.write_start, &result->io.write_end)) {
            session->failed = true;
            goto done;
        }
        SparfGroupKey *key = g_new0(SparfGroupKey, 1);
        *key = (SparfGroupKey){ .stream = header.stream, .kind = header.kind, .group = group };
        uint64_t *page = g_new(uint64_t, 1);
        *page = first_page;
        g_hash_table_insert(session->directory, key, page);
        result->io.write_bytes = group_pages * page_bytes;
        result->io.copied_bytes = result->io.write_bytes;
    } else {
        if (plan && !transfer(opaque, plan, false, &result->io.read_start, &result->io.read_end)) {
            session->failed = true;
            goto done;
        }
        result->io.read_bytes = plan ? (uint64_t)plan->sectors->len * 512 : 0;
        for (uint32_t index = 0; index < header.count; ++index) {
            uint64_t chosen = sparf_group_index(metadata, index);
            if (header.operation == CEMU_SPARF_GROUP_READ_TOKENS) {
                if (stream && chosen >= stream->committed[header.kind]) {
                    memcpy(buffer + (uint64_t)index * dimension * element,
                           sparf_group_tail(session, stream, header.kind, chosen), dimension * element);
                    continue;
                }
                uint64_t *page = sparf_group_lookup(session, header.stream, header.kind, chosen / token_capacity);
                memcpy(buffer + (uint64_t)index * dimension * element,
                       sparf_group_page(session, plan, *page) + chosen % token_capacity * dimension * element,
                       dimension * element);
            } else {
                for (uint64_t done_tokens = 0; done_tokens < header.argument;) {
                    uint64_t token = header.position + done_tokens;
                    if (stream && token >= stream->committed[CEMU_SPARF_GROUP_CHANNEL_K]) {
                        memcpy(buffer + ((uint64_t)index * header.argument + done_tokens) * element,
                               sparf_group_tail(session, stream, CEMU_SPARF_GROUP_TOKEN_K, token) + chosen * element,
                               element);
                        ++done_tokens;
                        continue;
                    }
                    uint64_t run = MIN(header.argument - done_tokens, channel_capacity - token % channel_capacity);
                    uint64_t *page = sparf_group_lookup(session, header.stream, header.kind, token / channel_capacity);
                    uint8_t *source = sparf_group_page(session, plan, *page + chosen / channels);
                    memcpy(buffer + ((uint64_t)index * header.argument + done_tokens) * element,
                           source + ((chosen % channels) * channel_capacity + token % channel_capacity) * element,
                           run * element);
                    done_tokens += run;
                }
            }
        }
        result->output_bytes = output_bytes;
        result->io.copied_bytes = output_bytes;
    }
    success = true;
done:
    result->pages_used = session->cursor;
    result->groups = g_hash_table_size(session->directory);
    sparf_wb_free(plan);
    g_hash_table_destroy(selected);
    return success;
}

#endif
