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
        buffer->data_bytes != (uint64_t)args->mr_len[1] ||
        meta % 8 || data % 8 || buffer->meta_bytes > UINTPTR_MAX - meta ||
        buffer->data_bytes > UINTPTR_MAX - data ||
        (meta < data + buffer->data_bytes && data < meta + buffer->meta_bytes) ||
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
        buffer->input_bytes > buffer->data_bytes) {
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
        !arena_region(meta, buffer->meta_bytes, &used, buffer->writebacks, buffer->writeback_capacity + 1ULL, sizeof(*buffer->writebacks))) {
        return false;
    }
    uint64_t padding = (layout.geometry.page_bytes - buffer->input_bytes % layout.geometry.page_bytes) %
                       layout.geometry.page_bytes;
    if (padding > buffer->data_bytes - buffer->input_bytes) {
        return false;
    }
    uint64_t offset = buffer->input_bytes + padding;
    return (uintptr_t)buffer->staged_data == data + offset &&
           buffer->pool_pages <= (buffer->data_bytes - offset) / layout.geometry.page_bytes;
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
    memset(buffer->staged_data + (uint64_t)(index - 1) * buffer->layout.geometry.page_bytes,
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

static bool visit_layout(FDMKV_Buffer_t *buffer, const FDMKV_AppendHeader_t *header,
                         uint32_t head, uint32_t kind, bool apply, uint64_t *needed)
{
    const FDMKV_Geometry_t *geometry = &buffer->layout.geometry;
    uint64_t head_index = ((uint64_t)header->batch * geometry->layers + header->layer) * geometry->heads + head;
    uint32_t capacity = kind == FDM_KV_CHANNEL_K ? buffer->layout.channel_group_tokens : buffer->layout.token_group_tokens;
    uint32_t stripes = kind == FDM_KV_CHANNEL_K ? buffer->layout.channel_group_pages : 1;
    uint32_t remaining = header->token_count;
    uint64_t token = header->token_first;
    const uint8_t *input = buffer->input + CEMU_FDM_KV_PACKET_HEADER_BYTES +
        (kind == FDM_KV_TOKEN_V ? header->payload_bytes / 2 : 0) +
        (uint64_t)(head - header->head_first) * header->token_count * buffer->row_bytes;
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
            uint8_t *destination = buffer->staged_data + (uint64_t)(index - 1) * geometry->page_bytes;
            if (kind != FDM_KV_CHANNEL_K) {
                memcpy(destination + (uint64_t)offset * buffer->row_bytes, input,
                       (uint64_t)take * buffer->row_bytes);
            } else {
                for (uint32_t channel = 0; channel < geometry->channels_per_page; channel++) {
                    for (uint32_t element = 0; element < take; element++) {
                        memcpy(destination + ((uint64_t)channel * capacity + offset + element) * geometry->element_bytes,
                               input + (uint64_t)element * buffer->row_bytes +
                               (stripe * geometry->channels_per_page + channel) * geometry->element_bytes,
                               geometry->element_bytes);
                    }
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

long long fdm_kv_aggregate(struct cemu_args *args)
{
    if (!args || args->numr != 2 || !args->mr_addr || !args->mr_len ||
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
    if (buffer->failed || buffer->draining || buffer->sealed || receive->state != FDM_KV_RX_READY) {
        return FDM_KV_AGGREGATE_BUSY;
    }
    if (!receive->header_valid || header->magic != CEMU_FDM_KV_PACKET_MAGIC ||
        header->version != CEMU_FDM_KV_PACKET_VERSION || header->operation != FDM_KV_OP_APPEND ||
        header->header_bytes != CEMU_FDM_KV_PACKET_HEADER_BYTES || header->reserved ||
        header->session_id != buffer->identity.session_id || header->generation != buffer->identity.generation ||
        header->request_id != receive->next_request_id || receive->next_request_id == UINT64_MAX ||
        header->batch >= geometry->batches || header->layer >= geometry->layers ||
        header->head_first >= geometry->heads || !header->head_count ||
        header->head_count > geometry->heads - header->head_first || !header->token_count ||
        header->token_count > buffer->append_max_tokens || header->token_first > geometry->max_tokens ||
        header->token_count > geometry->max_tokens - header->token_first ||
        header->wire_bytes > buffer->input_bytes || header->wire_bytes < CEMU_FDM_KV_PACKET_HEADER_BYTES ||
        header->payload_bytes > header->wire_bytes - CEMU_FDM_KV_PACKET_HEADER_BYTES) {
        return FDM_KV_AGGREGATE_INVALID;
    }
    uint64_t rows = (uint64_t)header->head_count * header->token_count;
    if (rows > UINT64_MAX / (2ULL * buffer->row_bytes) ||
        header->payload_bytes != rows * 2 * buffer->row_bytes ||
        buffer->stats.append_commands == UINT64_MAX ||
        rows > UINT64_MAX - buffer->stats.appended_head_tokens) {
        return FDM_KV_AGGREGATE_INVALID;
    }
    if ((uint64_t)args->cparam2 < receive->max_write_ready_ns) {
        return FDM_KV_AGGREGATE_WAIT_TIME;
    }
    uint64_t first_head = ((uint64_t)header->batch * geometry->layers + header->layer) * geometry->heads + header->head_first;
    uint64_t needed = 0;
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
    if (!free_pages_valid(buffer, needed)) {
        return FDM_KV_AGGREGATE_INVALID;
    }
    receive->state = FDM_KV_RX_AGGREGATING;
    for (uint32_t relative = 0; relative < header->head_count; relative++) {
        for (uint32_t kind = 0; kind < CEMU_FDM_KV_KIND_COUNT; kind++) {
            visit_layout(buffer, header, header->head_first + relative, kind, true, &needed);
        }
    }
    for (uint32_t relative = 0; relative < header->head_count; relative++) {
        buffer->head_states[first_head + relative].accepted += header->token_count;
    }
    buffer->stats.append_commands++;
    buffer->stats.appended_head_tokens += rows;
    receive->state = FDM_KV_RX_AGGREGATED;
    return FDM_KV_AGGREGATE_OK;
}
