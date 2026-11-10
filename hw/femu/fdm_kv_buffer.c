#include "qemu/osdep.h"
#include "fdm_kv_buffer.h"
#include "fdm_kv_layout.h"

typedef struct FdmKvArena {
    uint8_t *base;
    uint64_t capacity;
    uint64_t used;
} FdmKvArena;

static void *fdm_kv_take(FdmKvArena *arena, uint64_t count, uint64_t width)
{
    uint64_t padding = (8 - arena->used % 8) % 8;
    uint64_t bytes;

    if (!count || !width || count > UINT64_MAX / width) {
        return NULL;
    }

    bytes = count * width;
    if (arena->used > arena->capacity ||
        padding > arena->capacity - arena->used ||
        bytes > arena->capacity - arena->used - padding) {
        return NULL;
    }

    uint64_t offset = arena->used + padding;
    arena->used = offset + bytes;
    return arena->base + offset;
}

static uint64_t fdm_kv_pool_pages(const FDMKV_Layout_t *layout, uint32_t nand_channels, uint32_t block_pages)
{
    const FDMKV_Geometry_t *geometry = &layout->geometry;
    uint64_t residual = 0;
    uint64_t append_tokens = 2ULL * layout->token_group_tokens;

    for(uint32_t kind = 0; kind < CEMU_FDM_KV_KIND_COUNT; kind++)
    {
        uint32_t capacity = kind == FDM_KV_CHANNEL_K ? layout->channel_group_tokens :
                                                       layout->token_group_tokens;
        uint32_t stripes = kind == FDM_KV_CHANNEL_K ? layout->channel_group_pages : 1;

        uint64_t spans = geometry->max_tokens / capacity +
                         (geometry->max_tokens % capacity != 0);

        for(uint32_t channel = 0; channel < nand_channels; channel++)
        {
            uint64_t pages = (spans / nand_channels + (channel < spans % nand_channels)) * geometry->heads * stripes;
            residual += MIN(pages, (uint64_t)block_pages - 1);
        }
    }

    uint64_t partial = layout->head_count * (2ULL + layout->channel_group_pages);

    uint64_t channel_spans = append_tokens / layout->channel_group_tokens +
                             (append_tokens % layout->channel_group_tokens != 0);

    uint64_t command = (uint64_t)geometry->heads *
                      (4ULL + channel_spans * layout->channel_group_pages);
    uint64_t pages = residual * geometry->batches * geometry->layers +
                     partial + command;

    return MIN(pages, layout->total_pages);
}

bool fdm_kv_buffer_init(const FDMKV_InitParams_t *params, FDMKV_Buffer_t **result)
{
    FDMKV_Buffer_t candidate = {0};
    FDMKV_ReceiveState_t receive = {0};
    FdmKvArena arena;
    FDMKV_Buffer_t *buffer;

    uint64_t row_bytes;
    uint64_t data_padding;
    uint64_t data_offset;
    uint64_t pool_pages;
    uint64_t domains;
    uint64_t writebacks;
    uint64_t units;

    uint32_t hash_size = 1;
    uintptr_t meta_start;
    uintptr_t data_start;
    FDMKV_DataSegment_t segments[CEMU_FDM_KV_MAX_DATA_MRS] = {0};
    uint64_t available_pages = 0;
    uint64_t total_bytes = 0;

    if (!params || !result) {
        return false;
    }

    FDMKV_InitParams_t config = *params;

    if (config.segment_count > CEMU_FDM_KV_MAX_DATA_MRS ||
        (config.segment_count && !config.data_segments)) {
        return false;
    }
    if (config.segment_count) {
        memcpy(segments, config.data_segments, config.segment_count * sizeof(*segments));
        config.data_addr = segments[0].address;
        config.data_bytes = segments[0].bytes;
    } else {
        config.segment_count = 1;
        segments[0].address = config.data_addr;
        segments[0].bytes = config.data_bytes;
    }

    if (!config.meta_addr || !config.data_addr ||
        !config.meta_bytes || !config.data_bytes ||
        config.meta_bytes > SIZE_MAX || config.data_bytes > SIZE_MAX ||
        !config.identity.namespace_id || !config.identity.session_id ||
        !config.identity.generation ||
        !config.nand_channels || config.nand_channels > 128 ||
        !config.block_pages || config.block_pages > 65536 ||
        !config.batch_pages ||
        !fdm_kv_layout_init(&candidate.layout, &config.geometry) ||
        candidate.layout.head_count > 65536) {
        return false;
    }

    meta_start = (uintptr_t)config.meta_addr;
    data_start = (uintptr_t)config.data_addr;

    if (meta_start % 8 || data_start % 8 ||
        config.meta_bytes > UINTPTR_MAX - meta_start ||
        config.data_bytes > UINTPTR_MAX - data_start ||
        (meta_start < data_start + config.data_bytes &&
         data_start < meta_start + config.meta_bytes)) {
        return false;
    }

    row_bytes = (uint64_t)config.geometry.head_dim * config.geometry.element_bytes;

    if(config.input_bytes % 512 ||
       config.input_bytes < CEMU_FDM_KV_PACKET_HEADER_BYTES + 2 * row_bytes ||
       config.input_bytes > config.data_bytes)
    {
        return false;
    }

    data_padding = (config.geometry.page_bytes -
            config.input_bytes % config.geometry.page_bytes) %
            config.geometry.page_bytes;

    if (data_padding > config.data_bytes - config.input_bytes) {
        return false;
    }

    data_offset = config.input_bytes + data_padding;

    for (uint32_t index = 0; index < config.segment_count; index++) {
        uintptr_t address = (uintptr_t)segments[index].address;
        uint64_t bytes = segments[index].bytes;
        if (!address || address % 8 || !bytes || bytes > UINT32_MAX ||
            bytes > UINTPTR_MAX - address || bytes > UINT64_MAX - total_bytes ||
            (meta_start < address + bytes && address < meta_start + config.meta_bytes)) {
            return false;
        }
        for (uint32_t previous = 0; previous < index; previous++) {
            uintptr_t other = (uintptr_t)segments[previous].address;
            if (address < other + segments[previous].bytes && other < address + bytes) {
                return false;
            }
        }
        total_bytes += bytes;
        available_pages += (bytes - (index ? 0 : data_offset)) / config.geometry.page_bytes;
    }

    pool_pages = fdm_kv_pool_pages(&candidate.layout,
                                   config.nand_channels,
                                   config.block_pages);

    domains = (uint64_t)config.geometry.batches * config.geometry.layers *
                CEMU_FDM_KV_KIND_COUNT * config.nand_channels;

    if (!pool_pages || pool_pages > UINT32_MAX / 4 ||
        domains > UINT32_MAX ||
        pool_pages > available_pages) {
        return false;
    }

    while ((uint64_t)hash_size < 2 * pool_pages)
    {
        hash_size *= 2;
    }

    writebacks = MIN(pool_pages, pool_pages / config.block_pages + domains + 1);

    units = config.input_bytes / CEMU_FDM_KV_RECEIVE_UNIT_BYTES;
    receive.bitmap_bytes = units / 8 + (units % 8 != 0);
    receive.state = FDM_KV_RX_EMPTY;
    receive.next_request_id = 1;

    arena = (FdmKvArena){
        .base = config.meta_addr,
        .capacity = config.meta_bytes
    };

    buffer = fdm_kv_take(&arena, 1, sizeof(*buffer));
    candidate.receive = fdm_kv_take(&arena, 1, sizeof(*candidate.receive));

    receive.received_bitmap = fdm_kv_take(
        &arena, receive.bitmap_bytes, 1);

    candidate.head_states = fdm_kv_take(
        &arena, candidate.layout.head_count,
        sizeof(*candidate.head_states));

    candidate.staged = fdm_kv_take(
        &arena, pool_pages + 1, sizeof(*candidate.staged));

    candidate.blocks = fdm_kv_take(
        &arena, domains, sizeof(*candidate.blocks));

    candidate.page_hash = fdm_kv_take(
        &arena, hash_size, sizeof(*candidate.page_hash));

    candidate.writebacks = fdm_kv_take(
        &arena, writebacks + 1, sizeof(*candidate.writebacks));
    candidate.segment_count = config.segment_count;
    candidate.segments = fdm_kv_take(&arena, config.segment_count, sizeof(*candidate.segments));

    if (!buffer || !candidate.receive || !receive.received_bitmap ||
        !candidate.head_states || !candidate.staged || !candidate.blocks ||
        !candidate.page_hash || !candidate.writebacks || !candidate.segments) {
        return false;
    }

    candidate.version = CEMU_FDM_KV_BUFFER_VERSION;
    candidate.header_bytes = sizeof(candidate);
    candidate.identity = config.identity;

    candidate.meta_bytes = config.meta_bytes;
    candidate.data_bytes = total_bytes;

    candidate.append_max_tokens = 2 * candidate.layout.token_group_tokens;
    candidate.row_bytes = row_bytes;
    candidate.batch_pages = config.batch_pages;

    candidate.writeback_capacity = writebacks;
    candidate.free_writeback = 1;

    candidate.pool_pages = pool_pages;
    candidate.free_page = 1;
    candidate.free_pages = pool_pages;

    candidate.hash_size = hash_size;
    candidate.domains = domains;
    candidate.nand_channels = config.nand_channels;
    candidate.block_pages = config.block_pages;

    candidate.input = config.data_addr;
    candidate.input_bytes = config.input_bytes;
    candidate.staged_data = (uint8_t *)config.data_addr + data_offset;

    uint64_t assigned = 0;
    for (uint32_t index = 0; index < config.segment_count; index++) {
        uint64_t capacity = (segments[index].bytes - (index ? 0 : data_offset)) /
                            config.geometry.page_bytes;
        segments[index].first_page = assigned + 1;
        segments[index].page_count = MIN(capacity, pool_pages - assigned);
        assigned += segments[index].page_count;
    }

    memset(config.meta_addr, 0, arena.used);

    *buffer = candidate;
    *buffer->receive = receive;
    memcpy(buffer->segments, segments, config.segment_count * sizeof(*segments));

    for (uint32_t index = 1; index <= buffer->pool_pages; index++) {
        buffer->staged[index].next =
            index < buffer->pool_pages ? index + 1 : 0;
    }

    for (uint32_t index = 1; index <= buffer->writeback_capacity; index++) {
        buffer->writebacks[index].next =
            index < buffer->writeback_capacity ? index + 1 : 0;
    }

    buffer->initialized = 1;
    *result = buffer;
    return true;


}
