#ifndef CEMU_INSTATTENTION_ADDRESS_H
#define CEMU_INSTATTENTION_ADDRESS_H

#include <stdbool.h>
#include <stdint.h>
#include "hw/femu/kv_page.h"

typedef struct CemuKvGeometry {
    uint32_t batches;
    uint32_t layers;
    uint32_t heads;
    uint32_t head_dim;
    uint64_t max_tokens;
    uint32_t channels_per_page;
    uint32_t element_bytes;
    uint32_t page_bytes;
} CemuKvGeometry;

typedef struct CemuKvLayout {
    CemuKvGeometry geometry;
    uint32_t token_group_tokens;
    uint32_t channel_group_tokens;
    uint32_t channel_group_pages;
    uint64_t streams;
    uint64_t pages_per_stream;
    uint64_t total_pages;
} CemuKvLayout;

typedef struct CemuKvIdentity {
    uint32_t controller;
    uint32_t namespace_id;
    uint64_t task;
    uint64_t generation;
} CemuKvIdentity;

typedef struct CemuKvCoordinate {
    uint32_t batch;
    uint32_t layer;
    uint32_t head;
    uint32_t kind;
    uint64_t token;
    uint32_t channel;
} CemuKvCoordinate;

typedef struct CemuKvPageKey {
    CemuKvIdentity identity;
    uint32_t batch;
    uint32_t layer;
    uint32_t head;
    uint32_t kind;
    uint64_t group;
    uint32_t channel_group;
} CemuKvPageKey;

typedef struct CemuKvLocation {
    CemuKvPageKey key;
    uint64_t page_index;
    uint32_t byte_offset;
    uint32_t token_capacity;
} CemuKvLocation;

typedef struct CemuKvNandGeometry {
    uint32_t channels;
    uint32_t luns_per_channel;
    uint32_t planes_per_lun;
    uint32_t blocks_per_plane;
    uint32_t pages_per_block;
    uint32_t page_bytes;
} CemuKvNandGeometry;

typedef struct CemuKvPhysicalPage {
    uint32_t channel;
    uint32_t lun;
    uint32_t plane;
    uint32_t block;
    uint32_t page;
} CemuKvPhysicalPage;

static inline bool cemu_kv_multiply(uint64_t left, uint64_t right,
                                    uint64_t *result)
{
    if (right && left > UINT64_MAX / right) {
        return false;
    }
    *result = left * right;
    return true;
}

static inline bool cemu_kv_page_bytes_valid(uint32_t page_bytes)
{
    return page_bytes >= 512 && page_bytes <= 65536 && page_bytes % 512 == 0;
}

static inline bool cemu_kv_layout_init(CemuKvLayout *layout,
                                      const CemuKvGeometry *geometry)
{
    CemuKvLayout candidate = {0};
    uint64_t row_bytes, channel_bytes, token_pages, channel_pages;

    if (!layout || !geometry || !geometry->batches || !geometry->layers ||
        !geometry->heads || !geometry->head_dim || !geometry->max_tokens ||
        !geometry->channels_per_page ||
        geometry->head_dim % geometry->channels_per_page ||
        (geometry->element_bytes != 2 && geometry->element_bytes != 4) ||
        !cemu_kv_page_bytes_valid(geometry->page_bytes)) {
        return false;
    }
    row_bytes = (uint64_t)geometry->head_dim * geometry->element_bytes;
    channel_bytes = (uint64_t)geometry->channels_per_page * geometry->element_bytes;
    if (geometry->page_bytes % row_bytes || geometry->page_bytes % channel_bytes) {
        return false;
    }
    candidate.geometry = *geometry;
    candidate.token_group_tokens = geometry->page_bytes / row_bytes;
    candidate.channel_group_tokens = geometry->page_bytes / channel_bytes;
    candidate.channel_group_pages = geometry->head_dim / geometry->channels_per_page;
    token_pages = geometry->max_tokens / candidate.token_group_tokens +
                  (geometry->max_tokens % candidate.token_group_tokens != 0);
    channel_pages = geometry->max_tokens / candidate.channel_group_tokens +
                    (geometry->max_tokens % candidate.channel_group_tokens != 0);
    if (!cemu_kv_multiply(channel_pages, candidate.channel_group_pages, &channel_pages) ||
        !cemu_kv_multiply(token_pages, 2, &token_pages) ||
        channel_pages > UINT64_MAX - token_pages) {
        return false;
    }
    candidate.pages_per_stream = token_pages + channel_pages;
    if (!cemu_kv_multiply(geometry->batches, geometry->layers, &candidate.streams) ||
        !cemu_kv_multiply(candidate.streams, geometry->heads, &candidate.streams) ||
        !cemu_kv_multiply(candidate.streams, candidate.pages_per_stream,
                          &candidate.total_pages) ||
        candidate.total_pages > UINT64_MAX / geometry->page_bytes) {
        return false;
    }
    *layout = candidate;
    return true;
}

static inline bool cemu_kv_resolve(const CemuKvLayout *layout,
                                  const CemuKvIdentity *identity,
                                  const CemuKvCoordinate *coordinate,
                                  CemuKvLocation *location)
{
    CemuKvLocation candidate = {0};
    uint64_t first_token, remaining;
    uint32_t capacity;

    if (!layout || !identity || !coordinate || !location ||
        !identity->namespace_id || !identity->generation ||
        !layout->token_group_tokens || !layout->channel_group_tokens ||
        coordinate->kind > CEMU_KV_PAGE_TOKEN_V ||
        coordinate->batch >= layout->geometry.batches ||
        coordinate->layer >= layout->geometry.layers ||
        coordinate->head >= layout->geometry.heads ||
        coordinate->token >= layout->geometry.max_tokens ||
        coordinate->channel >= layout->geometry.head_dim) {
        return false;
    }
    candidate.key.identity = *identity;
    candidate.key.batch = coordinate->batch;
    candidate.key.layer = coordinate->layer;
    candidate.key.head = coordinate->head;
    candidate.key.kind = coordinate->kind;
    capacity = coordinate->kind == CEMU_KV_PAGE_CHANNEL_K ?
               layout->channel_group_tokens : layout->token_group_tokens;
    candidate.key.group = coordinate->token / capacity;
    if (coordinate->kind == CEMU_KV_PAGE_CHANNEL_K) {
        candidate.key.channel_group = coordinate->channel / layout->geometry.channels_per_page;
        candidate.page_index = candidate.key.group * layout->channel_group_pages +
                               candidate.key.channel_group;
        candidate.byte_offset = ((coordinate->channel % layout->geometry.channels_per_page) *
                                 capacity + coordinate->token % capacity) * layout->geometry.element_bytes;
    } else {
        candidate.page_index = candidate.key.group;
        candidate.byte_offset = ((coordinate->token % capacity) * layout->geometry.head_dim +
                                 coordinate->channel) * layout->geometry.element_bytes;
    }
    first_token = candidate.key.group * capacity;
    remaining = layout->geometry.max_tokens - first_token;
    candidate.token_capacity = remaining < capacity ? remaining : capacity;
    *location = candidate;
    return true;
}

static inline bool cemu_kv_page_key_equal(const CemuKvPageKey *left,
                                         const CemuKvPageKey *right)
{
    return left && right &&
           left->identity.controller == right->identity.controller &&
           left->identity.namespace_id == right->identity.namespace_id &&
           left->identity.task == right->identity.task &&
           left->identity.generation == right->identity.generation &&
           left->batch == right->batch && left->layer == right->layer &&
           left->head == right->head && left->kind == right->kind &&
           left->group == right->group && left->channel_group == right->channel_group;
}

static inline bool cemu_kv_physical_offset(const CemuKvNandGeometry *geometry,
                                          const CemuKvPhysicalPage *page,
                                          uint64_t backend_bytes,
                                          uint64_t *offset)
{
    uint64_t total, index;

    if (!geometry || !page || !offset ||
        !geometry->channels || geometry->channels > 128 ||
        !geometry->luns_per_channel || geometry->luns_per_channel > 256 ||
        !geometry->planes_per_lun || geometry->planes_per_lun > 256 ||
        !geometry->blocks_per_plane || geometry->blocks_per_plane > 65536 ||
        !geometry->pages_per_block || geometry->pages_per_block > 65536 ||
        !cemu_kv_page_bytes_valid(geometry->page_bytes) ||
        page->channel >= geometry->channels || page->lun >= geometry->luns_per_channel ||
        page->plane >= geometry->planes_per_lun || page->block >= geometry->blocks_per_plane ||
        page->page >= geometry->pages_per_block) {
        return false;
    }
    if (!cemu_kv_multiply(geometry->channels, geometry->luns_per_channel, &total) ||
        !cemu_kv_multiply(total, geometry->planes_per_lun, &total) ||
        !cemu_kv_multiply(total, geometry->blocks_per_plane, &total) ||
        !cemu_kv_multiply(total, geometry->pages_per_block, &total) ||
        !cemu_kv_multiply(total, geometry->page_bytes, &total) ||
        total > INT64_MAX || total > backend_bytes) {
        return false;
    }
    index = (uint64_t)page->channel * geometry->luns_per_channel + page->lun;
    index = index * geometry->planes_per_lun + page->plane;
    index = index * geometry->blocks_per_plane + page->block;
    index = index * geometry->pages_per_block + page->page;
    *offset = index * geometry->page_bytes;
    return true;
}

#endif
