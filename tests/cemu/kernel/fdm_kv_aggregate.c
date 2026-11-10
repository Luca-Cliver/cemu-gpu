#include <stdbool.h>
#include <stdint.h>
#include <string.h>
#include "cemu_def.h"
#include "../../../hw/femu/fdm_kv_aggregate.h"
#include "../../../hw/femu/fdm_kv_layout.h"

static bool arena_region(uintptr_t base, uint64_t capacity, uint64_t *used,
                         const void *address, uint64_t count, uint64_t width)
{
    uint64_t padding = (8 - *used % 8) % 8;
    if (!count || count > UINT64_MAX / width || *used > capacity ||
        padding > capacity - *used || count * width > capacity - *used - padding ||
        (uintptr_t)address != base + *used + padding) {
        return false;
    }
    *used += padding + count * width;
    return true;
}

static bool valid_buffer(const struct cemu_args *args, const FDMKV_Buffer_t *buffer)
{
    FDMKV_Layout_t layout;
    uint64_t used = sizeof(*buffer);
    uintptr_t meta = (uintptr_t)args->mr_addr[0];
    uintptr_t data = (uintptr_t)args->mr_addr[1];
    if (buffer->version != CEMU_FDM_KV_BUFFER_VERSION ||
        buffer->header_bytes != sizeof(*buffer) || !buffer->initialized ||
        buffer->meta_bytes != (uint64_t)args->mr_len[0] ||
        meta % 8 || data % 8 || buffer->meta_bytes > UINTPTR_MAX - meta ||
        buffer->segment_count != (uint32_t)args->numr - 1 || buffer->segment_reserved ||
        !fdm_kv_layout_init(&layout, &buffer->layout.geometry) ||
        memcmp(&layout, &buffer->layout, sizeof(layout)) ||
        buffer->row_bytes != (uint64_t)layout.geometry.head_dim * layout.geometry.element_bytes ||
        buffer->append_max_tokens != 2 * layout.token_group_tokens ||
        !buffer->nand_channels || !buffer->block_pages || !buffer->pool_pages ||
        buffer->pool_pages > layout.total_pages ||
        buffer->free_pages > buffer->pool_pages || buffer->free_page > buffer->pool_pages ||
        buffer->domains != (uint64_t)layout.geometry.batches * layout.geometry.layers *
                           CEMU_FDM_KV_KIND_COUNT * buffer->nand_channels ||
        !buffer->hash_size || (buffer->hash_size & (buffer->hash_size - 1)) ||
        !buffer->writeback_capacity || buffer->writeback_capacity > buffer->pool_pages ||
        buffer->input != args->mr_addr[1] || buffer->input_bytes % 512 ||
        buffer->input_bytes < 512 + 2ULL * buffer->row_bytes ||
        buffer->input_bytes > (uint64_t)args->mr_len[1]) {
        return false;
    }
    if (!arena_region(meta, buffer->meta_bytes, &used, buffer->receive, 1, sizeof(*buffer->receive))) {
        return false;
    }
    const FDMKV_ReceiveState_t *receive = buffer->receive;
    uint64_t units = buffer->input_bytes / CEMU_FDM_KV_RECEIVE_UNIT_BYTES;
    if (receive->bitmap_bytes != units / 8 + (units % 8 != 0) ||
        !arena_region(meta, buffer->meta_bytes, &used, receive->received_bitmap, receive->bitmap_bytes, 1) ||
        !arena_region(meta, buffer->meta_bytes, &used, buffer->head_states, layout.head_count, sizeof(*buffer->head_states)) ||
        !arena_region(meta, buffer->meta_bytes, &used, buffer->staged, buffer->pool_pages + 1ULL, sizeof(*buffer->staged)) ||
        !arena_region(meta, buffer->meta_bytes, &used, buffer->blocks, buffer->domains, sizeof(*buffer->blocks)) ||
        !arena_region(meta, buffer->meta_bytes, &used, buffer->page_hash, buffer->hash_size, sizeof(*buffer->page_hash)) ||
        !arena_region(meta, buffer->meta_bytes, &used, buffer->writebacks, buffer->writeback_capacity + 1ULL, sizeof(*buffer->writebacks)) ||
        !arena_region(meta, buffer->meta_bytes, &used, buffer->segments, buffer->segment_count, sizeof(*buffer->segments))) {
        return false;
    }
    uint64_t padding = (layout.geometry.page_bytes - buffer->input_bytes % layout.geometry.page_bytes) %
                       layout.geometry.page_bytes;
    if (padding > (uint64_t)args->mr_len[1] - buffer->input_bytes) {
        return false;
    }
    uint64_t offset = buffer->input_bytes + padding;
    uint64_t pages = 0, total_bytes = 0;
    for (uint32_t index = 0; index < buffer->segment_count; index++) {
        const FDMKV_DataSegment_t *segment = &buffer->segments[index];
        uintptr_t address = (uintptr_t)args->mr_addr[index + 1];
        uint64_t bytes = args->mr_len[index + 1];
        if (!address || address % 8 || !bytes || bytes > UINT32_MAX ||
            bytes > UINTPTR_MAX - address ||
            (meta < address + bytes && address < meta + buffer->meta_bytes) ||
            segment->address != args->mr_addr[index + 1] || segment->bytes != bytes) {
            return false;
        }
        for (uint32_t previous = 0; previous < index; previous++) {
            uintptr_t other = (uintptr_t)buffer->segments[previous].address;
            if (address < other + buffer->segments[previous].bytes && other < address + bytes) {
                return false;
            }
        }
        uint64_t capacity = (bytes - (index ? 0 : offset)) / layout.geometry.page_bytes;
        uint64_t count = capacity < buffer->pool_pages - pages ? capacity : buffer->pool_pages - pages;
        if (segment->first_page != pages + 1 || segment->page_count != count) {
            return false;
        }
        pages += count;
        total_bytes += bytes;
    }
    return (uintptr_t)buffer->staged_data == data + offset &&
           pages == buffer->pool_pages && total_bytes == buffer->data_bytes;
}

static uint32_t hash_slot(const FDMKV_Buffer_t *buffer, uint64_t key)
{
    uint64_t hash = key * UINT64_C(0x9e3779b97f4a7c15);
    return (hash ^ (hash >> 32)) & (buffer->hash_size - 1);
}

static bool find_page(const FDMKV_Buffer_t *buffer, uint64_t key, uint32_t *result)
{
    uint32_t index = buffer->page_hash[hash_slot(buffer, key)];
    uint32_t visited = 0;
    while (index) {
        if (index > buffer->pool_pages || visited++ >= buffer->pool_pages) {
            return false;
        }
        if (buffer->staged[index].key == key) {
            *result = index;
            return true;
        }
        index = buffer->staged[index].hash_next;
    }
    *result = 0;
    return true;
}

static uint64_t page_key(const FDMKV_Buffer_t *buffer, uint64_t head,
                         uint32_t kind, uint64_t group, uint32_t stripe)
{
    uint64_t token_pages = buffer->layout.geometry.max_tokens / buffer->layout.token_group_tokens +
                          (buffer->layout.geometry.max_tokens % buffer->layout.token_group_tokens != 0);
    uint64_t base = head * buffer->layout.pages_per_head;
    if (kind == FDM_KV_CHANNEL_K) {
        return base + token_pages + group * buffer->layout.channel_group_pages + stripe;
    }
    return base + (kind == FDM_KV_TOKEN_V ? buffer->layout.pages_per_head - token_pages : 0) + group;
}

static bool free_next(const FDMKV_Buffer_t *buffer, uint32_t *index)
{
    if (*index > buffer->pool_pages) {
        return false;
    }
    if (*index) {
        *index = buffer->staged[*index].next;
    }
    return *index <= buffer->pool_pages;
}

static bool free_pages_valid(const FDMKV_Buffer_t *buffer, uint64_t needed)
{
    uint32_t index = buffer->free_page, slow = index, fast = index;
    for (uint64_t page = 0; page < needed; page++) {
        uint32_t allocated;
        if (!index || index > buffer->pool_pages ||
            !find_page(buffer, buffer->staged[index].key, &allocated) || allocated == index ||
            !free_next(buffer, &index) || !free_next(buffer, &slow) ||
            !free_next(buffer, &fast) || !free_next(buffer, &fast) || (fast && slow == fast)) {
            return false;
        }
    }
    return true;
}

static uint32_t allocate_page(FDMKV_Buffer_t *buffer, uint64_t key,
                              const FDMKV_Coordinate_t *coordinate)
{
    uint32_t index = buffer->free_page;
    uint32_t slot = hash_slot(buffer, key);
    FDMKV_StagedPage_t *page = &buffer->staged[index];
    buffer->free_page = page->next;
    buffer->free_pages--;
    *page = (FDMKV_StagedPage_t) { .key = key, .coordinate = *coordinate,
                                 .hash_next = buffer->page_hash[slot] };
    buffer->page_hash[slot] = index;
    memset(fdm_kv_page_data(buffer, index),
           0, buffer->layout.geometry.page_bytes);
    buffer->stats.staged_pages++;
    return index;
}

static void enqueue_page(FDMKV_Buffer_t *buffer, uint32_t index, uint32_t domain)
{
    FDMKV_StagedPage_t *page = &buffer->staged[index];
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

/* Keep memcpy's size constant so the compiler emits unaligned-safe scalar
 * loads/stores, rather than calling the generic copy routine per element.
 * Source advances one token row; destination advances within one channel. */
#define DEFINE_CHANNEL_COPY(name, width) \
static inline void name(uint8_t *destination, const uint8_t *input, \
                        uint32_t channels, uint32_t tokens, uint32_t capacity, \
                        uint32_t offset, uint64_t row_bytes) \
{ \
    uint8_t *column = destination + (uint64_t)offset * (width); \
    for (uint32_t channel = 0; channel < channels; channel++) { \
        uint8_t *out = column; \
        const uint8_t *in = input + (uint64_t)channel * (width); \
        for (uint32_t token = 0; token < tokens; token++) { \
            memcpy(out, in, (width)); \
            out += (width); \
            in += row_bytes; \
        } \
        column += (uint64_t)capacity * (width); \
    } \
}

DEFINE_CHANNEL_COPY(copy_channel_k_fp16, 2)
DEFINE_CHANNEL_COPY(copy_channel_k_fp32, 4)
#undef DEFINE_CHANNEL_COPY

static const uint8_t *packet_key(const FDMKV_Buffer_t *buffer, uint32_t relative_head)
{
    const FDMKV_ReceiveState_t *receive = buffer->receive;
    const FDMKV_AppendHeader_t *packet = &receive->header;
    uint64_t vector = (uint64_t)receive->aggregate_batch * packet->head_count + relative_head;
    return buffer->input + CEMU_FDM_KV_PACKET_HEADER_BYTES +
        (vector * packet->token_count + receive->aggregate_tokens) * buffer->row_bytes;
}

static bool visit_layout(FDMKV_Buffer_t *buffer, const FDMKV_AppendHeader_t *header,
                         uint32_t head, uint32_t kind, bool apply, uint64_t *needed)
{
    const FDMKV_Geometry_t *geometry = &buffer->layout.geometry;
    uint64_t head_index = ((uint64_t)header->batch * geometry->layers + header->layer) * geometry->heads + head;
    uint32_t capacity = kind == FDM_KV_CHANNEL_K ? buffer->layout.channel_group_tokens : buffer->layout.token_group_tokens;
    uint32_t stripes = kind == FDM_KV_CHANNEL_K ? buffer->layout.channel_group_pages : 1;
    uint32_t remaining = header->token_count;
    uint64_t token = header->token_first;
    const uint8_t *input = packet_key(buffer, head - header->head_first) +
        (kind == FDM_KV_TOKEN_V ? header->payload_bytes / 2 : 0);
    while (remaining) {
        uint32_t offset = token % capacity;
        uint32_t take = remaining < capacity - offset ? remaining : capacity - offset;
        uint64_t group = token / capacity;
        uint32_t domain = (((uint64_t)header->batch * geometry->layers + header->layer) *
                           CEMU_FDM_KV_KIND_COUNT + kind) * buffer->nand_channels + group % buffer->nand_channels;
        FDMKV_BlockQueue_t *block = &buffer->blocks[domain];
        if (!apply && (block->count > buffer->pool_pages ||
            block->first > buffer->pool_pages || block->last > buffer->pool_pages ||
            (!block->count != !block->first) || (!block->count != !block->last))) {
            return false;
        }
        for (uint32_t stripe = 0; stripe < stripes; stripe++) {
            uint64_t key = page_key(buffer, head_index, kind, group, stripe);
            uint32_t index;
            if (!find_page(buffer, key, &index)) {
                return false;
            }
            if (!apply) {
                if ((!index && offset) || (index && (buffer->staged[index].ready ||
                    buffer->staged[index].valid_tokens != offset))) {
                    return false;
                }
                *needed += !index;
                continue;
            }
            if (!index) {
                FDMKV_Coordinate_t coordinate = { .batch = header->batch, .layer = header->layer,
                    .head = head, .kind = kind, .token = token - offset,
                    .channel = kind == FDM_KV_CHANNEL_K ? stripe * geometry->channels_per_page : 0 };
                index = allocate_page(buffer, key, &coordinate);
            }
            uint8_t *destination = fdm_kv_page_data(buffer, index);
            if (kind != FDM_KV_CHANNEL_K) {
                memcpy(destination + (uint64_t)offset * buffer->row_bytes, input,
                       (uint64_t)take * buffer->row_bytes);
            } else {
                const uint8_t *stripe_input = input + (uint64_t)stripe *
                    geometry->channels_per_page * geometry->element_bytes;
                if (geometry->element_bytes == 2) {
                    copy_channel_k_fp16(destination, stripe_input, geometry->channels_per_page,
                                        take, capacity, offset, buffer->row_bytes);
                } else {
                    copy_channel_k_fp32(destination, stripe_input, geometry->channels_per_page,
                                        take, capacity, offset, buffer->row_bytes);
                }
            }
            buffer->staged[index].valid_tokens += take;
            if (buffer->staged[index].valid_tokens == capacity) {
                enqueue_page(buffer, index, domain);
            }
        }
        token += take;
        remaining -= take;
        input += (uint64_t)take * buffer->row_bytes;
    }
    return true;
}

typedef struct AggregateTimer {
    FDMKV_AggregateProfile *profile;
    FDMKV_ProfileClock now_ns;
    uint64_t last;
    FDMKV_AggregateStage stage;
} AggregateTimer;

static void profile_stage(AggregateTimer *timer, FDMKV_AggregateStage next)
{
    if (timer) {
        uint64_t now = timer->now_ns();
        timer->profile->host_ns[timer->stage] += now - timer->last;
        timer->last = now;
        timer->stage = next;
    }
}

/* Only for one-token appends inside an existing token group. The caller keeps
 * the buffer lock across validation and apply. Save the looked-up page indices
 * on this call's stack; no persistent cache or invalidation protocol is needed.
 * The generic path handles group boundaries and packets exceeding this bound. */
#define DECODE_PLAN_PAGES 2048
static inline void prefetch_channel_write(const FDMKV_Buffer_t *buffer, uint32_t index,
                                         uint32_t offset)
{
    const FDMKV_Geometry_t *geometry = &buffer->layout.geometry;
    uint8_t *column = fdm_kv_page_data(buffer, index) + (uint64_t)offset * geometry->element_bytes;
    for (uint32_t channel = 0; channel < geometry->channels_per_page; channel++) {
        __builtin_prefetch(column, 1, 1);
        column += (uint64_t)buffer->layout.channel_group_tokens * geometry->element_bytes;
    }
}

static long long append_decode_existing(FDMKV_Buffer_t *buffer,
                                        const FDMKV_AppendHeader_t *header,
                                        uint64_t first_head, AggregateTimer *timer)
{
    uint32_t indices[DECODE_PLAN_PAGES], used = 0;
    const FDMKV_Layout_t *layout = &buffer->layout;
    const FDMKV_Geometry_t *geometry = &layout->geometry;
    uint64_t token_pages = geometry->max_tokens / layout->token_group_tokens +
                          (geometry->max_tokens % layout->token_group_tokens != 0);
    uint32_t capacities[3] = { layout->token_group_tokens, layout->channel_group_tokens,
                               layout->token_group_tokens };
    uint32_t stripes[3] = { 1, layout->channel_group_pages, 1 };
    /* Prefetching costs more than it saves for small/hot buffers. Use the
     * bounded lookahead only once staged page storage exceeds 64 MiB.
     * This is a Host implementation heuristic, not a device timing model. */
    bool prefetch = (uint64_t)buffer->stats.staged_pages * geometry->page_bytes >
                    UINT64_C(64) * 1024 * 1024;
    uint32_t offsets[3], domains[3];
    uint64_t key_offsets[3];
    for (uint32_t kind = 0; kind < CEMU_FDM_KV_KIND_COUNT; kind++) {
        uint64_t group = header->token_first / capacities[kind];
        offsets[kind] = header->token_first % capacities[kind];
        key_offsets[kind] = group * stripes[kind] +
            (kind == FDM_KV_CHANNEL_K ? token_pages :
             kind == FDM_KV_TOKEN_V ? layout->pages_per_head - token_pages : 0);
        domains[kind] = (((uint64_t)header->batch * geometry->layers + header->layer) *
            CEMU_FDM_KV_KIND_COUNT + kind) * buffer->nand_channels + group % buffer->nand_channels;
    }
    profile_stage(timer, FDM_KV_AGG_STAGE_preflight);
    for (uint32_t relative = 0; relative < header->head_count; relative++) {
        const FDMKV_HeadState_t *head = &buffer->head_states[first_head + relative];
        if (head->accepted != header->token_first) {
            return FDM_KV_AGGREGATE_INVALID;
        }
        if (head->readers) {
            return FDM_KV_AGGREGATE_BUSY;
        }
        uint64_t base = (first_head + relative) * layout->pages_per_head;
        for (uint32_t kind = 0; kind < CEMU_FDM_KV_KIND_COUNT; kind++) {
            const FDMKV_BlockQueue_t *block = &buffer->blocks[domains[kind]];
            if (block->count > buffer->pool_pages || block->first > buffer->pool_pages ||
                block->last > buffer->pool_pages || (!block->count != !block->first) ||
                (!block->count != !block->last)) {
                return FDM_KV_AGGREGATE_INVALID;
            }
            for (uint32_t stripe = 0; stripe < stripes[kind]; stripe++) {
                uint32_t index;
                if (!find_page(buffer, base + key_offsets[kind] + stripe, &index) || !index ||
                    buffer->staged[index].ready || buffer->staged[index].valid_tokens != offsets[kind]) {
                    return FDM_KV_AGGREGATE_INVALID;
                }
                indices[used++] = index;
            }
        }
    }
    profile_stage(timer, FDM_KV_AGG_STAGE_free_list); /* No new pages needed. */
    buffer->receive->state = FDM_KV_RX_AGGREGATING;
    used = 0;
    for (uint32_t relative = 0; relative < header->head_count; relative++) {
        const uint8_t *key = packet_key(buffer, relative);
        for (uint32_t kind = 0; kind < CEMU_FDM_KV_KIND_COUNT; kind++) {
            profile_stage(timer, kind == FDM_KV_CHANNEL_K ? FDM_KV_AGG_STAGE_channel_k :
                                 kind == FDM_KV_TOKEN_V ? FDM_KV_AGG_STAGE_token_v : FDM_KV_AGG_STAGE_token_k);
            if (prefetch && kind == FDM_KV_CHANNEL_K) {
                for (uint32_t ahead = 0; ahead < 4 && ahead < stripes[kind]; ahead++) {
                    prefetch_channel_write(buffer, indices[used + ahead], offsets[kind]);
                }
            }
            for (uint32_t stripe = 0; stripe < stripes[kind]; stripe++) {
                if (prefetch && kind == FDM_KV_CHANNEL_K && stripe + 4 < stripes[kind]) {
                    prefetch_channel_write(buffer, indices[used + 4], offsets[kind]);
                }
                uint32_t index = indices[used++];
                uint8_t *destination = fdm_kv_page_data(buffer, index);
                if (kind == FDM_KV_CHANNEL_K) {
                    const uint8_t *input = key + (uint64_t)stripe * geometry->channels_per_page * geometry->element_bytes;
                    if (geometry->element_bytes == 2) {
                        copy_channel_k_fp16(destination, input, geometry->channels_per_page,
                                            1, capacities[kind], offsets[kind], buffer->row_bytes);
                    } else {
                        copy_channel_k_fp32(destination, input, geometry->channels_per_page,
                                            1, capacities[kind], offsets[kind], buffer->row_bytes);
                    }
                } else {
                    memcpy(destination + offsets[kind] * buffer->row_bytes,
                           key + (kind == FDM_KV_TOKEN_V ? header->payload_bytes / 2 : 0), buffer->row_bytes);
                }
                if (++buffer->staged[index].valid_tokens == capacities[kind]) {
                    enqueue_page(buffer, index, domains[kind]);
                }
            }
        }
    }
    return FDM_KV_AGGREGATE_OK;
}

static long long aggregate_impl(struct cemu_args *args, AggregateTimer *timer)
{
    if (!args || args->numr < 2 || (uint32_t)args->numr > CEMU_FDM_KV_MAX_DATA_MRS + 1 ||
        !args->mr_addr || !args->mr_len ||
        !args->mr_addr[0] || !args->mr_addr[1] || (uintptr_t)args->mr_addr[0] % 8 ||
        args->mr_len[0] < (long long)sizeof(FDMKV_Buffer_t) || args->mr_len[1] <= 0 ||
        (uint64_t)args->cparam1 != CEMU_FDM_KV_AGGREGATE_COMMAND || args->cparam2 < 0 ||
        args->buffer_len || args->data_buffer) {
        return FDM_KV_AGGREGATE_INVALID;
    }
    FDMKV_Buffer_t *buffer = args->mr_addr[0];
    if (!valid_buffer(args, buffer)) {
        return FDM_KV_AGGREGATE_INVALID;
    }
    FDMKV_ReceiveState_t *receive = buffer->receive;
    const FDMKV_AppendHeader_t *header = &receive->header;
    const FDMKV_Geometry_t *geometry = &buffer->layout.geometry;
    bool layer_packet = header->version == CEMU_FDM_KV_LAYER_PACKET_VERSION;
    uint32_t batches = layer_packet ? header->batch_count : 1;
    if (buffer->failed || buffer->draining || buffer->sealed || receive->state != FDM_KV_RX_READY) {
        return FDM_KV_AGGREGATE_BUSY;
    }
    if (!receive->header_valid || header->magic != CEMU_FDM_KV_PACKET_MAGIC ||
        (!layer_packet && header->version != CEMU_FDM_KV_PACKET_VERSION) || header->operation != FDM_KV_OP_APPEND ||
        header->header_bytes != CEMU_FDM_KV_PACKET_HEADER_BYTES || (!layer_packet && header->reserved) ||
        header->session_id != buffer->identity.session_id || header->generation != buffer->identity.generation ||
        header->request_id != receive->next_request_id || receive->next_request_id == UINT64_MAX ||
        header->batch >= geometry->batches || !batches || batches > geometry->batches - header->batch ||
        header->layer >= geometry->layers ||
        header->head_first >= geometry->heads || !header->head_count ||
        header->head_count > geometry->heads - header->head_first || !header->token_count ||
        (!layer_packet && header->token_count > buffer->append_max_tokens) || header->token_first > geometry->max_tokens ||
        header->token_count > geometry->max_tokens - header->token_first ||
        header->wire_bytes > buffer->input_bytes || header->wire_bytes < CEMU_FDM_KV_PACKET_HEADER_BYTES ||
        header->payload_bytes > header->wire_bytes - CEMU_FDM_KV_PACKET_HEADER_BYTES) {
        return FDM_KV_AGGREGATE_INVALID;
    }
    uint64_t rows = (uint64_t)header->head_count * header->token_count;
    if (rows > UINT64_MAX / batches || receive->aggregate_batch >= batches ||
        receive->aggregate_tokens >= header->token_count ||
        (!layer_packet && (receive->aggregate_batch || receive->aggregate_tokens))) {
        return FDM_KV_AGGREGATE_INVALID;
    }
    rows *= batches;
    if (rows > UINT64_MAX / (2ULL * buffer->row_bytes) ||
        header->payload_bytes != rows * 2 * buffer->row_bytes ||
        buffer->stats.append_commands == UINT64_MAX ||
        rows > UINT64_MAX - buffer->stats.appended_head_tokens) {
        return FDM_KV_AGGREGATE_INVALID;
    }
    if ((uint64_t)args->cparam2 < receive->max_write_ready_ns) {
        return FDM_KV_AGGREGATE_WAIT_TIME;
    }
    FDMKV_AppendHeader_t chunk = *header;
    if (layer_packet) {
        chunk.batch += receive->aggregate_batch;
        chunk.token_first += receive->aggregate_tokens;
        chunk.token_count -= receive->aggregate_tokens;
        uint32_t limit = buffer->append_max_tokens - chunk.token_first % buffer->append_max_tokens;
        if (chunk.token_count > limit) {
            chunk.token_count = limit;
        }
        header = &chunk;
    }
    rows = (uint64_t)header->head_count * header->token_count;
    uint64_t first_head = ((uint64_t)header->batch * geometry->layers + header->layer) * geometry->heads + header->head_first;
    uint64_t needed = 0;
    if (header->token_count == 1 && header->token_first % buffer->layout.token_group_tokens &&
        (uint64_t)header->head_count * (2ULL + buffer->layout.channel_group_pages) <= DECODE_PLAN_PAGES) {
        long long result = append_decode_existing(buffer, header, first_head, timer);
        if (result != FDM_KV_AGGREGATE_OK) {
            return result;
        }
        goto commit;
    }
    profile_stage(timer, FDM_KV_AGG_STAGE_preflight);
    for (uint32_t relative = 0; relative < header->head_count; relative++) {
        if (buffer->head_states[first_head + relative].accepted != header->token_first) {
            return FDM_KV_AGGREGATE_INVALID;
        }
        if (buffer->head_states[first_head + relative].readers) {
            return FDM_KV_AGGREGATE_BUSY;
        }
        for (uint32_t kind = 0; kind < CEMU_FDM_KV_KIND_COUNT; kind++) {
            if (!visit_layout(buffer, header, header->head_first + relative, kind, false, &needed)) {
                return FDM_KV_AGGREGATE_INVALID;
            }
        }
    }
    if (needed > buffer->free_pages) {
        return FDM_KV_AGGREGATE_NO_SPACE;
    }
    profile_stage(timer, FDM_KV_AGG_STAGE_free_list);
    if (!free_pages_valid(buffer, needed)) {
        return FDM_KV_AGGREGATE_INVALID;
    }
    receive->state = FDM_KV_RX_AGGREGATING;
    for (uint32_t relative = 0; relative < header->head_count; relative++) {
        for (uint32_t kind = 0; kind < CEMU_FDM_KV_KIND_COUNT; kind++) {
            FDMKV_AggregateStage stage = kind == FDM_KV_CHANNEL_K ? FDM_KV_AGG_STAGE_channel_k :
                kind == FDM_KV_TOKEN_V ? FDM_KV_AGG_STAGE_token_v : FDM_KV_AGG_STAGE_token_k;
            profile_stage(timer, stage);
            visit_layout(buffer, header, header->head_first + relative, kind, true, &needed);
        }
    }
commit:
    profile_stage(timer, FDM_KV_AGG_STAGE_commit);
    for (uint32_t relative = 0; relative < header->head_count; relative++) {
        buffer->head_states[first_head + relative].accepted += header->token_count;
    }
    buffer->stats.appended_head_tokens += rows;
    if (timer) {
        timer->profile->successful_calls++;
        timer->profile->new_pages += needed;
        timer->profile->copied_bytes += 3 * rows * buffer->row_bytes;
        timer->profile->channel_elements += rows * geometry->head_dim;
    }
    if (layer_packet) {
        receive->aggregate_tokens += header->token_count;
        if (receive->aggregate_tokens == receive->header.token_count) {
            receive->aggregate_tokens = 0;
            receive->aggregate_batch++;
        }
        if (receive->aggregate_batch < batches) {
            receive->state = FDM_KV_RX_READY;
            return FDM_KV_AGGREGATE_PROGRESS;
        }
    }
    buffer->stats.append_commands++;
    receive->state = FDM_KV_RX_AGGREGATED;
    return FDM_KV_AGGREGATE_OK;
}

long long fdm_kv_aggregate(struct cemu_args *args)
{
    return aggregate_impl(args, NULL);
}

long long fdm_kv_aggregate_profile_v1(void *args, FDMKV_AggregateProfile *profile,
                                    FDMKV_ProfileClock now_ns)
{
    if (!profile || !now_ns) {
        return FDM_KV_AGGREGATE_INVALID;
    }
    AggregateTimer timer = { .profile = profile, .now_ns = now_ns,
                            .stage = FDM_KV_AGG_STAGE_validate, .last = now_ns() };
    profile->calls++;
    long long result = aggregate_impl(args, &timer);
    profile_stage(&timer, FDM_KV_AGG_STAGE_commit);
    return result;
}
