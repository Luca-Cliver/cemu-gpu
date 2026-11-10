#include "qemu/osdep.h"
#include "qemu/bswap.h"
#include "hw/femu/fdm_kv_layout.h"
#include "hw/femu/fdm_kv_buffer.h"
#include "hw/femu/fdm_kv_receive.h"
#include "hw/femu/fdm_kv_aggregate.h"
#include "tests/cemu/kernel/cemu_def.h"
#undef assert
#include <assert.h>
#include <dlfcn.h>

static FDMKV_Geometry_t opt_geometry(void)
{
    return (FDMKV_Geometry_t) {
        .batches = 4, .layers = 40, .heads = 40, .head_dim = 128,
        .max_tokens = 1024, .channels_per_page = 8,
        .element_bytes = 2, .page_bytes = 4096,
    };
}

static void test_layout(void)
{
    FDMKV_Geometry_t geometry = opt_geometry();
    FDMKV_Layout_t layout;

    for (uint32_t channels = 2; channels <= 8; channels *= 2) {
        geometry.channels_per_page = channels;
        assert(fdm_kv_layout_init(&layout, &geometry));
        assert(layout.geometry.channels_per_page == channels);
        assert(layout.token_group_tokens == 16);
        assert(layout.channel_group_tokens == 2048 / channels);
        assert(layout.channel_group_pages == 128 / channels);
        assert(layout.head_count == 6400);
        assert(layout.pages_per_head == 192);
        assert(layout.total_pages == UINT64_C(1228800));
        assert(layout.reserved == 0);
    }

    geometry.channels_per_page = 8;
    geometry.element_bytes = 4;
    assert(fdm_kv_layout_init(&layout, &geometry));
    assert(layout.token_group_tokens == 8);
    assert(layout.channel_group_tokens == 128);
    assert(layout.channel_group_pages == 16);
    assert(layout.pages_per_head == 384);
    assert(layout.total_pages == UINT64_C(2457600));

    geometry.element_bytes = 2;
    geometry.page_bytes = 8192;
    assert(fdm_kv_layout_init(&layout, &geometry));
    assert(layout.token_group_tokens == 32);
    assert(layout.channel_group_tokens == 512);
    assert(layout.pages_per_head == 96);
}

static void test_partial_groups(void)
{
    const struct {
        uint64_t tokens;
        uint64_t pages;
    } cases[] = {
        {1, 18}, {15, 18}, {16, 18}, {17, 20},
        {255, 48}, {256, 48}, {257, 66}, {1040, 210},
    };
    FDMKV_Geometry_t geometry = opt_geometry();
    FDMKV_Layout_t layout;

    geometry.batches = geometry.layers = geometry.heads = 1;
    for (size_t index = 0; index < G_N_ELEMENTS(cases); ++index) {
        geometry.max_tokens = cases[index].tokens;
        assert(fdm_kv_layout_init(&layout, &geometry));
        assert(layout.head_count == 1);
        assert(layout.pages_per_head == cases[index].pages);
        assert(layout.total_pages == cases[index].pages);
    }
}

static void test_invalid_layout(void)
{
    FDMKV_Geometry_t geometry = opt_geometry();
    FDMKV_Layout_t layout, saved;

    assert(fdm_kv_layout_init(&layout, &geometry));
    saved = layout;
    assert(!fdm_kv_layout_init(NULL, &geometry));
    assert(!fdm_kv_layout_init(&layout, NULL));
    assert(memcmp(&layout, &saved, sizeof(layout)) == 0);

    for (unsigned int choice = 0; choice < 17; ++choice) {
        geometry = opt_geometry();
        switch (choice) {
        case 0: geometry.batches = 0; break;
        case 1: geometry.layers = 0; break;
        case 2: geometry.heads = 0; break;
        case 3: geometry.head_dim = 0; break;
        case 4: geometry.max_tokens = 0; break;
        case 5: geometry.channels_per_page = 0; break;
        case 6: geometry.head_dim = 127; break;
        case 7: geometry.element_bytes = 8; break;
        case 8: geometry.page_bytes = 256; break;
        case 9: geometry.page_bytes = 4095; break;
        case 10: geometry.head_dim = 4096; break;
        case 11: geometry.reserved = 1; break;
        case 12: geometry.reserved = UINT32_MAX; break;
        case 13:
            geometry.batches = geometry.layers = geometry.heads = UINT32_MAX;
            break;
        case 14: geometry.max_tokens = UINT64_MAX; break;
        case 15: geometry.page_bytes = 131072; break;
        case 16:
            geometry.batches = geometry.layers = geometry.heads = 1;
            geometry.max_tokens = UINT64_C(1) << 55;
            break;
        }
        assert(!fdm_kv_layout_init(&layout, &geometry));
        assert(memcmp(&layout, &saved, sizeof(layout)) == 0);
    }
}

static void assert_bytes(const void *address, size_t bytes, uint8_t expected)
{
    const uint8_t *data = address;

    for (size_t index = 0; index < bytes; index++) {
        assert(data[index] == expected);
    }
}

static FDMKV_InitParams_t buffer_params(void *meta, void *data)
{
    return (FDMKV_InitParams_t) {
        .identity = { .namespace_id = 1, .session_id = 1, .generation = 1 },
        .geometry = {
            .batches = 2, .layers = 2, .heads = 2, .head_dim = 128,
            .max_tokens = 531, .channels_per_page = 8,
            .element_bytes = 2, .page_bytes = 4096,
        },
        .meta_addr = meta, .meta_bytes = 128 * 1024,
        .data_addr = data, .data_bytes = 4 * 1024 * 1024,
        .input_bytes = 64 * 1024 + 512,
        .nand_channels = 2, .block_pages = 4, .batch_pages = 4,
    };
}

static size_t check_buffer(const FDMKV_InitParams_t *params,
                          const FDMKV_Buffer_t *buffer)
{
    const FDMKV_ReceiveState_t *receive = buffer->receive;
    const struct {
        const void *address;
        uint64_t bytes;
    } regions[] = {
        { buffer, sizeof(*buffer) },
        { receive, sizeof(*receive) },
        { receive->received_bitmap, receive->bitmap_bytes },
        { buffer->head_states, buffer->layout.head_count * sizeof(*buffer->head_states) },
        { buffer->staged, (buffer->pool_pages + 1ULL) * sizeof(*buffer->staged) },
        { buffer->blocks, buffer->domains * sizeof(*buffer->blocks) },
        { buffer->page_hash, buffer->hash_size * sizeof(*buffer->page_hash) },
        { buffer->writebacks, (buffer->writeback_capacity + 1ULL) * sizeof(*buffer->writebacks) },
        { buffer->segments, buffer->segment_count * sizeof(*buffer->segments) },
    };
    uintptr_t meta_start = (uintptr_t)params->meta_addr;
    uint64_t used = 0;

    for (size_t index = 0; index < G_N_ELEMENTS(regions); index++) {
        uintptr_t address = (uintptr_t)regions[index].address;
        assert(address % 8 == 0 && address >= meta_start);
        uint64_t offset = address - meta_start;
        assert(offset >= used && offset <= params->meta_bytes);
        assert(regions[index].bytes <= params->meta_bytes - offset);
        used = offset + regions[index].bytes;
    }

    assert(buffer == params->meta_addr);
    assert(buffer->version == CEMU_FDM_KV_BUFFER_VERSION);
    assert(buffer->header_bytes == sizeof(*buffer));
    assert(memcmp(&buffer->identity, &params->identity, sizeof(buffer->identity)) == 0);
    assert(buffer->meta_bytes == params->meta_bytes && buffer->data_bytes == params->data_bytes);
    assert(buffer->row_bytes == params->geometry.head_dim * params->geometry.element_bytes);
    assert(buffer->append_max_tokens == 2 * buffer->layout.token_group_tokens);
    assert(buffer->batch_pages == params->batch_pages);
    assert(buffer->nand_channels == params->nand_channels);
    assert(buffer->block_pages == params->block_pages);
    assert(buffer->domains == params->geometry.batches * params->geometry.layers *
                              CEMU_FDM_KV_KIND_COUNT * params->nand_channels);
    assert(buffer->initialized == 1);
    assert(!buffer->draining && !buffer->sealed && !buffer->failed && !buffer->leases);
    assert(buffer->pool_pages && buffer->pool_pages <= buffer->layout.total_pages);
    assert(buffer->free_pages == buffer->pool_pages && buffer->free_page == 1);
    assert(buffer->hash_size >= 2 * buffer->pool_pages);
    assert((buffer->hash_size & (buffer->hash_size - 1)) == 0);
    assert(buffer->writeback_capacity && buffer->free_writeback == 1);
    assert(receive->state == FDM_KV_RX_EMPTY && !receive->header_valid);
    assert(receive->next_request_id == 1 && !receive->max_write_ready_ns);
    assert(receive->bitmap_bytes == DIV_ROUND_UP(
        params->input_bytes / CEMU_FDM_KV_RECEIVE_UNIT_BYTES, 8));
    assert_bytes(receive->received_bitmap, receive->bitmap_bytes, 0);
    assert_bytes(&receive->header, sizeof(receive->header), 0);
    assert_bytes(buffer->head_states,
                 buffer->layout.head_count * sizeof(*buffer->head_states), 0);
    assert_bytes(buffer->blocks, buffer->domains * sizeof(*buffer->blocks), 0);
    assert_bytes(buffer->page_hash, buffer->hash_size * sizeof(*buffer->page_hash), 0);
    assert_bytes(&buffer->stats, sizeof(buffer->stats), 0);
    assert_bytes(&buffer->staged[0], sizeof(buffer->staged[0]), 0);
    assert_bytes(&buffer->writebacks[0], sizeof(buffer->writebacks[0]), 0);

    for (uint32_t index = 1; index <= buffer->pool_pages; index++) {
        const FDMKV_StagedPage_t *page = &buffer->staged[index];
        assert(page->next == (index < buffer->pool_pages ? index + 1 : 0));
        assert(!page->key && !page->valid_tokens && !page->hash_next && !page->ready);
        assert_bytes(&page->coordinate, sizeof(page->coordinate), 0);
    }
    for (uint32_t index = 1; index <= buffer->writeback_capacity; index++) {
        const FDMKV_Writeback_t *writeback = &buffer->writebacks[index];
        assert(writeback->next == (index < buffer->writeback_capacity ? index + 1 : 0));
        assert(!writeback->deadline_ns && !writeback->first && !writeback->count && !writeback->active);
    }

    uint64_t data_offset = buffer->staged_data - (uint8_t *)params->data_addr;
    assert(buffer->input == params->data_addr && buffer->input_bytes == params->input_bytes);
    assert(data_offset >= params->input_bytes && data_offset <= params->data_bytes);
    assert(data_offset % params->geometry.page_bytes == 0);
    assert(data_offset - params->input_bytes < params->geometry.page_bytes);
    assert((uint64_t)buffer->pool_pages * params->geometry.page_bytes <= params->data_bytes - data_offset);
    assert_bytes((uint8_t *)params->meta_addr + used, params->meta_bytes - used, 0xa5);
    assert_bytes(params->data_addr, params->data_bytes, 0x5a);
    return used;
}

static void test_buffer_init(void)
{
    uint8_t *meta = g_malloc(128 * 1024);
    uint8_t *data = g_malloc(4 * 1024 * 1024);
    FDMKV_InitParams_t params = buffer_params(meta, data);
    FDMKV_Buffer_t *buffer = NULL;
    const uint64_t tokens[] = { 1, 15, 16, 17, 255, 256, 257, 531, 1040 };

    memset(data, 0x5a, params.data_bytes);
    for (uint32_t element_bytes = 2; element_bytes <= 4; element_bytes *= 2) {
        for (size_t index = 0; index < G_N_ELEMENTS(tokens); index++) {
            params.geometry.element_bytes = element_bytes;
            params.geometry.max_tokens = tokens[index];
            memset(meta, 0xa5, params.meta_bytes);
            assert(fdm_kv_buffer_init(&params, &buffer));
            check_buffer(&params, buffer);
        }
    }

    params = buffer_params(meta, data);
    memset(meta, 0xa5, params.meta_bytes);
    assert(fdm_kv_buffer_init(&params, &buffer));
    size_t meta_used = check_buffer(&params, buffer);
    assert(buffer->pool_pages == 256);
    assert(buffer->domains == 24 && buffer->hash_size == 512);
    assert(buffer->writeback_capacity == 89);
    params.meta_bytes = meta_used;
    params.data_bytes = (buffer->staged_data - data) +
                        (uint64_t)buffer->pool_pages * params.geometry.page_bytes;
    memset(meta, 0xa5, params.meta_bytes);
    assert(fdm_kv_buffer_init(&params, &buffer));
    check_buffer(&params, buffer);

    for (unsigned int choice = 0; choice < 28; choice++) {
        params = buffer_params(meta, data);
        memset(meta, 0xa5, params.meta_bytes);
        switch (choice) {
        case 0: params.meta_addr = NULL; break;
        case 1: params.data_addr = NULL; break;
        case 2: params.meta_bytes = 0; break;
        case 3: params.data_bytes = 0; break;
        case 4: params.meta_bytes = meta_used - 1; break;
        case 5: params.data_bytes = 17 * 4096 + 256 * 4096 - 1; break;
        case 6: params.input_bytes = 512; break;
        case 7: params.input_bytes++; break;
        case 8: params.input_bytes = params.data_bytes + 512; break;
        case 9: params.meta_addr = meta + 1; break;
        case 10: params.data_addr = data + 1; break;
        case 11: params.data_addr = meta; break;
        case 12: params.meta_addr = data + 8; break;
        case 13: params.meta_bytes = UINT64_MAX; break;
        case 14: params.data_bytes = UINT64_MAX; break;
        case 15: params.identity.namespace_id = 0; break;
        case 16: params.identity.session_id = 0; break;
        case 17: params.identity.generation = 0; break;
        case 18: params.nand_channels = 0; break;
        case 19: params.nand_channels = 129; break;
        case 20: params.block_pages = 0; break;
        case 21: params.block_pages = 65537; break;
        case 22: params.batch_pages = 0; break;
        case 23: params.geometry.heads = 65537; break;
        case 24: params.geometry.reserved = 1; break;
        case 25: params.geometry.max_tokens = UINT64_MAX; break;
        case 26: params.data_bytes = params.input_bytes; break;
        case 27: params.meta_bytes = sizeof(*buffer); break;
        }
        buffer = (FDMKV_Buffer_t *)meta;
        assert(!fdm_kv_buffer_init(&params, &buffer));
        assert(buffer == (FDMKV_Buffer_t *)meta);
        assert_bytes(meta, 128 * 1024, 0xa5);
        assert_bytes(data, 4 * 1024 * 1024, 0x5a);
    }

    params = buffer_params(meta, data);
    assert(!fdm_kv_buffer_init(NULL, &buffer));
    assert(!fdm_kv_buffer_init(&params, NULL));
    assert_bytes(meta, params.meta_bytes, 0xa5);
    assert_bytes(data, params.data_bytes, 0x5a);
    memcpy(meta, &params, sizeof(params));
    assert(fdm_kv_buffer_init((FDMKV_InitParams_t *)meta, &buffer));
    check_buffer(&params, buffer);
    g_free(data);
    g_free(meta);
}

static uint64_t make_packet(uint8_t *packet, const FDMKV_Buffer_t *buffer,
                            uint32_t heads, uint32_t tokens)
{
    uint64_t payload_bytes = 2ULL * heads * tokens * buffer->row_bytes;
    uint64_t wire_bytes = (512 + payload_bytes + 511) / 512 * 512;
    FDMKV_AppendHeader_t header = {
        .magic = cpu_to_le32(CEMU_FDM_KV_PACKET_MAGIC),
        .version = cpu_to_le32(CEMU_FDM_KV_PACKET_VERSION),
        .header_bytes = cpu_to_le32(CEMU_FDM_KV_PACKET_HEADER_BYTES),
        .operation = cpu_to_le32(FDM_KV_OP_APPEND),
        .session_id = cpu_to_le64(buffer->identity.session_id),
        .generation = cpu_to_le64(buffer->identity.generation),
        .request_id = cpu_to_le64(1),
        .wire_bytes = cpu_to_le64(wire_bytes),
        .payload_bytes = cpu_to_le64(payload_bytes),
        .head_count = cpu_to_le32(heads),
        .token_count = cpu_to_le32(tokens),
    };
    memset(packet, 0, wire_bytes);
    memcpy(packet, &header, sizeof(header));
    memset(packet + 512, 0x3c, payload_bytes);
    return wire_bytes;
}

static FDMKV_ReceiveResult receive_fragment(FDMKV_Buffer_t *buffer,
                                           const uint8_t *packet,
                                           uint64_t offset, uint64_t bytes,
                                           uint64_t ready_ns)
{
    assert(fdm_kv_receive_check(buffer, offset, bytes) == FDM_KV_RECEIVE_ACCEPTED);
    memcpy(buffer->input + offset, packet + offset, bytes);
    return fdm_kv_receive_record(buffer, offset, bytes, ready_ns);
}

static void test_receive(void)
{
    uint8_t *meta = g_malloc(128 * 1024);
    uint8_t *data = g_malloc(4 * 1024 * 1024);
    uint8_t packet[4096];
    FDMKV_InitParams_t params = buffer_params(meta, data);
    FDMKV_Buffer_t *buffer;
    memset(data, 0x5a, params.data_bytes);
    assert(fdm_kv_buffer_init(&params, &buffer));
    uint64_t bytes = make_packet(packet, buffer, 2, 3);
    assert(bytes == 3584);
    assert(fdm_kv_receive_check(buffer, UINT64_MAX - 3, 8) == FDM_KV_RECEIVE_INVALID);
    assert(fdm_kv_receive_check(buffer, 0, 0) == FDM_KV_RECEIVE_INVALID);
    assert(fdm_kv_receive_check(buffer, 1, 4) == FDM_KV_RECEIVE_INVALID);
    assert(fdm_kv_receive_check(buffer, 0, 3) == FDM_KV_RECEIVE_INVALID);
    assert(fdm_kv_receive_check(buffer, params.input_bytes - 4, 8) == FDM_KV_RECEIVE_INVALID);
    assert(receive_fragment(buffer, packet, 1024, bytes - 1024, 9000) == FDM_KV_RECEIVE_ACCEPTED);
    assert(receive_fragment(buffer, packet, 4, 508, 4000) == FDM_KV_RECEIVE_ACCEPTED);
    assert(!buffer->receive->header_valid);
    assert(fdm_kv_receive_check(buffer, 4, 4) == FDM_KV_RECEIVE_BUSY);
    assert(fdm_kv_receive_check(buffer, 0, 8) == FDM_KV_RECEIVE_BUSY);
    assert(receive_fragment(buffer, packet, 0, 4, 5000) == FDM_KV_RECEIVE_ACCEPTED);
    assert(buffer->receive->header_valid);
    assert(buffer->receive->state == FDM_KV_RX_RECEIVING);
    assert(fdm_kv_receive_check(buffer, bytes, 4) == FDM_KV_RECEIVE_INVALID);
    assert(receive_fragment(buffer, packet, 512, 512, 1000) == FDM_KV_RECEIVE_READY);
    assert(buffer->receive->state == FDM_KV_RX_READY);
    assert(buffer->receive->max_write_ready_ns == 9000);
    assert(buffer->receive->next_request_id == 1);
    assert(buffer->receive->header.head_count == 2);
    assert(buffer->receive->header.token_count == 3);
    assert(memcmp(buffer->input, packet, bytes) == 0);
    assert(fdm_kv_receive_check(buffer, 0, 4) == FDM_KV_RECEIVE_BUSY);
    assert_bytes(buffer->staged_data, (uint64_t)buffer->pool_pages * 4096, 0x5a);
    assert_bytes(buffer->head_states, buffer->layout.head_count * sizeof(*buffer->head_states), 0);
    assert_bytes(&buffer->stats, sizeof(buffer->stats), 0);

    assert(fdm_kv_buffer_init(&params, &buffer));
    assert(receive_fragment(buffer, packet, bytes, 4, 1) == FDM_KV_RECEIVE_ACCEPTED);
    assert(receive_fragment(buffer, packet, 0, 512, 2) == FDM_KV_RECEIVE_INVALID);
    assert(buffer->failed && buffer->receive->state == FDM_KV_RX_FAILED);
    assert(fdm_kv_receive_check(buffer, 512, 4) == FDM_KV_RECEIVE_BUSY);

    assert(fdm_kv_buffer_init(&params, &buffer));
    fdm_kv_receive_fail(buffer);
    assert_bytes(buffer->receive->received_bitmap, buffer->receive->bitmap_bytes, 0);
    assert(fdm_kv_receive_check(buffer, 0, 4) == FDM_KV_RECEIVE_BUSY);

    params.geometry.head_dim = 64;
    assert(fdm_kv_buffer_init(&params, &buffer));
    bytes = make_packet(packet, buffer, 1, 1);
    assert(bytes == 1024);
    assert(receive_fragment(buffer, packet, 0, 768, 1) == FDM_KV_RECEIVE_ACCEPTED);
    assert(receive_fragment(buffer, packet, 768, 256, 2) == FDM_KV_RECEIVE_READY);
    assert(fdm_kv_buffer_init(&params, &buffer));
    packet[1020] = 1;
    assert(receive_fragment(buffer, packet, 0, bytes, 1) == FDM_KV_RECEIVE_INVALID);
    assert(buffer->failed);
    g_free(data);
    g_free(meta);
}

typedef long long (*AggregateFunction)(struct cemu_args *args);

static uint32_t sample_value(uint32_t batch, uint32_t layer, uint32_t head,
                              uint64_t token, uint32_t channel, bool value)
{
    return (uint32_t)(token * 131 + channel * 17 + head * 1009 + layer * 3049 +
                      batch * 7919 + (value ? 0x3571 : 0x6813));
}

static uint64_t prepare_aggregate(FDMKV_Buffer_t *buffer, uint32_t batch, uint32_t layer,
                                 uint32_t head_first, uint32_t heads,
                                 uint64_t first, uint32_t tokens)
{
    uint64_t bytes = make_packet(buffer->input, buffer, heads, tokens);
    FDMKV_AppendHeader_t header;
    memcpy(&header, buffer->input, sizeof(header));
    header.request_id = cpu_to_le64(buffer->receive->next_request_id);
    header.batch = cpu_to_le32(batch);
    header.layer = cpu_to_le32(layer);
    header.head_first = cpu_to_le32(head_first);
    header.token_first = cpu_to_le64(first);
    memcpy(buffer->input, &header, sizeof(header));
    const FDMKV_Geometry_t *geometry = &buffer->layout.geometry;
    for (uint32_t kind = 0; kind < 2; kind++) {
        for (uint32_t head = 0; head < heads; head++) {
            for (uint32_t token = 0; token < tokens; token++) {
                for (uint32_t channel = 0; channel < geometry->head_dim; channel++) {
                    uint32_t sample = sample_value(batch, layer, head_first + head, first + token, channel, kind);
                    uint64_t element = ((uint64_t)kind * heads + head) * tokens * geometry->head_dim +
                                       (uint64_t)token * geometry->head_dim + channel;
                    memcpy(buffer->input + 512 + element * geometry->element_bytes, &sample, geometry->element_bytes);
                }
            }
        }
    }
    assert(fdm_kv_receive_record(buffer, 0, bytes, 9000) == FDM_KV_RECEIVE_READY);
    return bytes;
}

static long long invoke_aggregate(AggregateFunction function, const FDMKV_InitParams_t *params,
                                  uint64_t now_ns)
{
    void *addresses[CEMU_FDM_KV_MAX_DATA_MRS + 1] = { params->meta_addr, params->data_addr };
    long long lengths[CEMU_FDM_KV_MAX_DATA_MRS + 1] = { params->meta_bytes, params->data_bytes };
    for (uint32_t index = 0; index < params->segment_count; index++) {
        addresses[index + 1] = params->data_segments[index].address;
        lengths[index + 1] = params->data_segments[index].bytes;
    }
    struct cemu_args args = { .numr = params->segment_count ? params->segment_count + 1 : 2,
        .mr_addr = addresses, .mr_len = lengths,
        .cparam1 = CEMU_FDM_KV_AGGREGATE_COMMAND, .cparam2 = now_ns };
    return function(&args);
}

static void expect_unchanged(AggregateFunction function, const FDMKV_InitParams_t *params,
                             uint64_t now_ns, FDMKV_AggregateResult expected)
{
    void *meta = g_memdup2(params->meta_addr, params->meta_bytes);
    void *data = g_memdup2(params->data_addr, params->data_bytes);
    assert(invoke_aggregate(function, params, now_ns) == expected);
    assert(memcmp(meta, params->meta_addr, params->meta_bytes) == 0);
    assert(memcmp(data, params->data_addr, params->data_bytes) == 0);
    g_free(meta);
    g_free(data);
}

static void check_aggregate_pages(const FDMKV_Buffer_t *buffer)
{
    const FDMKV_Geometry_t *geometry = &buffer->layout.geometry;
    uint64_t count = 0, full = 0;
    uint32_t *domain_counts = g_new0(uint32_t, buffer->domains);
    uint8_t *seen = g_new0(uint8_t, buffer->pool_pages + 1);
    uint8_t *expected = g_malloc(geometry->page_bytes);
    for (uint32_t slot = 0; slot < buffer->hash_size; slot++) {
        uint32_t index = buffer->page_hash[slot];
        while (index) {
            assert(index <= buffer->pool_pages && !seen[index]);
            seen[index] = 1;
            count++;
            const FDMKV_StagedPage_t *page = &buffer->staged[index];
            const FDMKV_Coordinate_t *coordinate = &page->coordinate;
            uint32_t capacity = coordinate->kind == FDM_KV_CHANNEL_K ?
                                buffer->layout.channel_group_tokens : buffer->layout.token_group_tokens;
            assert(page->valid_tokens && page->valid_tokens <= capacity);
            assert(page->ready == (page->valid_tokens == capacity));
            uint64_t head = ((uint64_t)coordinate->batch * geometry->layers + coordinate->layer) * geometry->heads + coordinate->head;
            assert(coordinate->token + page->valid_tokens <= buffer->head_states[head].accepted);
            uint64_t token_pages = DIV_ROUND_UP(geometry->max_tokens, buffer->layout.token_group_tokens);
            uint64_t channel_pages = DIV_ROUND_UP(geometry->max_tokens, buffer->layout.channel_group_tokens) * buffer->layout.channel_group_pages;
            uint64_t local_key = coordinate->token / capacity;
            if (coordinate->kind == FDM_KV_CHANNEL_K) {
                local_key = token_pages + local_key * buffer->layout.channel_group_pages + coordinate->channel / geometry->channels_per_page;
            } else if (coordinate->kind == FDM_KV_TOKEN_V) {
                local_key += token_pages + channel_pages;
            }
            assert(page->key == head * buffer->layout.pages_per_head + local_key);
            memset(expected, 0, geometry->page_bytes);
            uint32_t width = coordinate->kind == FDM_KV_CHANNEL_K ? geometry->channels_per_page : geometry->head_dim;
            for (uint32_t token = 0; token < page->valid_tokens; token++) {
                for (uint32_t channel = 0; channel < width; channel++) {
                    uint32_t sample = sample_value(coordinate->batch, coordinate->layer, coordinate->head,
                        coordinate->token + token, coordinate->channel + channel, coordinate->kind == FDM_KV_TOKEN_V);
                    uint64_t element = coordinate->kind == FDM_KV_CHANNEL_K ?
                        (uint64_t)channel * capacity + token : (uint64_t)token * geometry->head_dim + channel;
                    memcpy(expected + element * geometry->element_bytes, &sample, geometry->element_bytes);
                }
            }
            assert(memcmp(expected, fdm_kv_page_data(buffer, index),
                          geometry->page_bytes) == 0);
            if (page->ready) {
                uint32_t domain = ((coordinate->batch * geometry->layers + coordinate->layer) * 3 +
                    coordinate->kind) * buffer->nand_channels + (coordinate->token / capacity) % buffer->nand_channels;
                domain_counts[domain]++;
                full++;
            }
            index = page->hash_next;
        }
    }
    assert(count == buffer->stats.staged_pages);
    assert(count + buffer->free_pages == buffer->pool_pages);
    uint64_t queued = 0, ready_blocks = 0;
    for (uint32_t domain = 0; domain < buffer->domains; domain++) {
        const FDMKV_BlockQueue_t *block = &buffer->blocks[domain];
        assert(block->count == domain_counts[domain]);
        uint32_t index = block->first, last = 0;
        uint32_t pages = 0;
        while (index) {
            assert(index <= buffer->pool_pages && seen[index] == 1);
            seen[index] = 2;
            assert(buffer->staged[index].ready);
            last = index;
            index = buffer->staged[index].next;
            pages++;
        }
        assert(pages == block->count && last == block->last);
        queued += pages;
        ready_blocks += pages / buffer->block_pages;
    }
    assert(queued == full);
    if (full >= (uint64_t)buffer->domains * buffer->block_pages) {
        assert(ready_blocks);
    }
    uint32_t free_pages = 0;
    for (uint32_t index = buffer->free_page; index; index = buffer->staged[index].next) {
        assert(index <= buffer->pool_pages && !seen[index]);
        seen[index] = 3;
        free_pages++;
    }
    assert(free_pages == buffer->free_pages);
    for (uint64_t head = 0; head < buffer->layout.head_count; head++) {
        assert_bytes(buffer->head_states[head].committed, sizeof(buffer->head_states[head].committed), 0);
    }
    assert(!buffer->stats.write_bytes && !buffer->stats.write_batches && !buffer->stats.write_blocks);
    g_free(expected);
    g_free(seen);
    g_free(domain_counts);
}

static void test_aggregate(AggregateFunction function)
{
    assert(function(NULL) == FDM_KV_AGGREGATE_INVALID);
    struct cemu_args invalid_args = {0};
    assert(function(&invalid_args) == FDM_KV_AGGREGATE_INVALID);
    void *meta = g_malloc(128 * 1024), *data = g_malloc(4 * 1024 * 1024);
    FDMKV_InitParams_t params = buffer_params(meta, data);
    FDMKV_Buffer_t *buffer;
    for (uint32_t variant = 0; variant < 4; variant++) {
        params.geometry.head_dim = variant == 3 ? 256 : (variant == 2 ? 64 : 128);
        params.geometry.element_bytes = variant == 1 ? 4 : 2;
        params.geometry.channels_per_page = variant == 2 ? 4 : 8;
        params.geometry.max_tokens = 4096 / (params.geometry.element_bytes * params.geometry.channels_per_page) + 3;
        memset(data, 0x5a, params.data_bytes);
        assert(fdm_kv_buffer_init(&params, &buffer));
        uint32_t token_group = buffer->layout.token_group_tokens;
        uint32_t channel_group = buffer->layout.channel_group_tokens;
        uint32_t boundaries[] = { token_group - 1, token_group, token_group + 1,
                                  channel_group - 1, channel_group, channel_group + 3 };
        uint32_t first = 0;
        uint64_t packets = 0;
        for (uint32_t boundary = 0; boundary < G_N_ELEMENTS(boundaries); boundary++) {
            while (first < boundaries[boundary]) {
                uint32_t tokens = MIN(boundaries[boundary] - first, buffer->append_max_tokens);
                uint64_t bytes = prepare_aggregate(buffer, 1, 1, 0, 2, first, tokens);
                uint8_t *packet = g_memdup2(buffer->input, bytes);
                assert(!fdm_kv_receive_release(buffer));
                expect_unchanged(function, &params, 8999, FDM_KV_AGGREGATE_WAIT_TIME);
                assert(invoke_aggregate(function, &params, 9000) == FDM_KV_AGGREGATE_OK);
                packets++;
                first += tokens;
                assert(memcmp(packet, buffer->input, bytes) == 0);
                assert(buffer->receive->state == FDM_KV_RX_AGGREGATED);
                assert(buffer->receive->next_request_id == packets);
                assert(buffer->stats.append_commands == packets);
                assert(buffer->stats.appended_head_tokens == 2ULL * first);
                assert(buffer->head_states[6].accepted == first && buffer->head_states[7].accepted == first);
                assert(buffer->head_states[0].accepted == 0);
                expect_unchanged(function, &params, 9000, FDM_KV_AGGREGATE_BUSY);
                assert(fdm_kv_receive_check(buffer, 0, 512) == FDM_KV_RECEIVE_BUSY);
                check_aggregate_pages(buffer);
                assert(fdm_kv_receive_release(buffer));
                assert(buffer->receive->next_request_id == packets + 1);
                assert_bytes(buffer->receive->received_bitmap, buffer->receive->bitmap_bytes, 0);
                assert_bytes(&buffer->receive->header, sizeof(buffer->receive->header), 0);
                assert(!buffer->receive->max_write_ready_ns && !buffer->receive->header_valid);
                assert(memcmp(packet, buffer->input, bytes) == 0);
                assert(!fdm_kv_receive_release(buffer));
                g_free(packet);
            }
        }
        prepare_aggregate(buffer, 0, 0, 1, 1, 0, 1);
        assert(invoke_aggregate(function, &params, 9000) == FDM_KV_AGGREGATE_OK);
        assert(buffer->head_states[1].accepted == 1 && buffer->head_states[0].accepted == 0);
        check_aggregate_pages(buffer);
    }
    params = buffer_params(meta, data);
    assert(fdm_kv_buffer_init(&params, &buffer));
    prepare_aggregate(buffer, 0, 0, 0, 2, 0, 1);
    buffer->head_states[1].accepted = 1;
    expect_unchanged(function, &params, 9000, FDM_KV_AGGREGATE_INVALID);
    buffer->head_states[1].accepted = 0;
    buffer->head_states[1].readers = 1;
    expect_unchanged(function, &params, 9000, FDM_KV_AGGREGATE_BUSY);
    buffer->head_states[1].readers = 0;
    uint32_t free_pages = buffer->free_pages;
    buffer->free_pages = 0;
    expect_unchanged(function, &params, 9000, FDM_KV_AGGREGATE_NO_SPACE);
    buffer->free_pages = free_pages;
    uint32_t saved_next = buffer->staged[buffer->free_page].next;
    buffer->staged[buffer->free_page].next = buffer->pool_pages + 1;
    expect_unchanged(function, &params, 9000, FDM_KV_AGGREGATE_INVALID);
    buffer->staged[buffer->free_page].next = buffer->free_page;
    expect_unchanged(function, &params, 9000, FDM_KV_AGGREGATE_INVALID);
    buffer->staged[buffer->free_page].next = saved_next;
    uint8_t *saved_input = buffer->input;
    buffer->input++;
    expect_unchanged(function, &params, 9000, FDM_KV_AGGREGATE_INVALID);
    buffer->input = saved_input;
    assert(invoke_aggregate(function, &params, 9000) == FDM_KV_AGGREGATE_OK);
    check_aggregate_pages(buffer);
    uint64_t bytes = buffer->receive->header.wire_bytes;
    assert(fdm_kv_receive_release(buffer));
    assert(fdm_kv_receive_record(buffer, 0, bytes, 9000) == FDM_KV_RECEIVE_INVALID);
    assert(buffer->failed);
    g_free(data);
    g_free(meta);
}

static void test_segmented_aggregate(AggregateFunction function)
{
    void *meta = g_malloc(128 * 1024);
    uint8_t *input = g_malloc(128 * 1024);
    uint8_t *tail = g_malloc(2 * 1024 * 1024);
    FDMKV_InitParams_t params = buffer_params(meta, input);
    FDMKV_DataSegment_t segments[] = {
        { .address = input, .bytes = 73728 },
        { .address = tail, .bytes = 2 * 1024 * 1024 },
    };
    params.segment_count = G_N_ELEMENTS(segments);
    params.data_segments = segments;
    FDMKV_Buffer_t *buffer;
    assert(fdm_kv_buffer_init(&params, &buffer));
    assert(buffer->segments[0].page_count == 1);
    assert(fdm_kv_page_data(buffer, 1) == input + 69632);
    assert(fdm_kv_page_data(buffer, 2) == tail);
    assert(!fdm_kv_page_data(buffer, 0));
    assert(!fdm_kv_page_data(buffer, buffer->pool_pages + 1));
    for (uint32_t first = 0; first < 272; first += 16) {
        prepare_aggregate(buffer, 1, 1, 0, 2, first, 16);
        assert(invoke_aggregate(function, &params, 9000) == FDM_KV_AGGREGATE_OK);
        check_aggregate_pages(buffer);
        assert(fdm_kv_receive_release(buffer));
    }
    prepare_aggregate(buffer, 1, 1, 0, 2, 272, 1);
    buffer->segments[1].address++;
    assert(invoke_aggregate(function, &params, 9000) == FDM_KV_AGGREGATE_INVALID);
    buffer->segments[1].address--;
    buffer->segments[1].first_page++;
    assert(invoke_aggregate(function, &params, 9000) == FDM_KV_AGGREGATE_INVALID);
    buffer->segments[1].first_page--;
    assert(invoke_aggregate(function, &params, 9000) == FDM_KV_AGGREGATE_OK);
    check_aggregate_pages(buffer);
    segments[1].address = input;
    assert(!fdm_kv_buffer_init(&params, &buffer));
    segments[1].address = meta;
    assert(!fdm_kv_buffer_init(&params, &buffer));
    segments[1].address = tail;
    segments[1].bytes = 4096;
    assert(!fdm_kv_buffer_init(&params, &buffer));
    segments[1].bytes = UINT64_C(1) << 32;
    assert(!fdm_kv_buffer_init(&params, &buffer));
    g_free(tail);
    g_free(input);
    g_free(meta);
}

static void test_aggregate_plan_boundaries(AggregateFunction function)
{
    void *meta = g_malloc(1024 * 1024), *data = g_malloc(32 * 1024 * 1024);
    FDMKV_InitParams_t params = buffer_params(meta, data);
    params.meta_bytes = 1024 * 1024;
    params.data_bytes = 32 * 1024 * 1024;
    params.input_bytes = 128 * 1024;
    params.geometry.batches = params.geometry.layers = 1;
    params.geometry.max_tokens = 33;
    /* 18 pages/head: 113 heads fit in 2048 cached indices; 114/128
     * exercise bounded-plan fallback, including the existing-page case. */
    const unsigned counts[] = { 113, 114, 128 };
    for (unsigned choice = 0; choice < G_N_ELEMENTS(counts); choice++) {
        params.geometry.heads = counts[choice];
        FDMKV_Buffer_t *buffer;
        memset(data, 0x5a, params.data_bytes);
        assert(fdm_kv_buffer_init(&params, &buffer));
        for (unsigned token = 0; token < 2; token++) {
            prepare_aggregate(buffer, 0, 0, 0, counts[choice], token, 1);
            buffer->head_states[counts[choice] - 1].readers = 1;
            expect_unchanged(function, &params, 9000, FDM_KV_AGGREGATE_BUSY);
            buffer->head_states[counts[choice] - 1].readers = 0;
            if (token) {
                /* Reject a bad page at the end of preflight without writing
                 * any of the earlier heads, on both fast and fallback paths. */
                uint32_t last = 0;
                for (uint32_t index = 1; index <= buffer->pool_pages; index++) {
                    FDMKV_StagedPage_t *page = &buffer->staged[index];
                    if (page->valid_tokens == token &&
                        page->coordinate.head == counts[choice] - 1 &&
                        page->coordinate.kind == FDM_KV_TOKEN_V) {
                        last = index;
                        break;
                    }
                }
                assert(last);
                buffer->staged[last].valid_tokens++;
                expect_unchanged(function, &params, 9000, FDM_KV_AGGREGATE_INVALID);
                buffer->staged[last].valid_tokens--;
                buffer->staged[last].ready = 1;
                expect_unchanged(function, &params, 9000, FDM_KV_AGGREGATE_INVALID);
                buffer->staged[last].ready = 0;
            }
            assert(invoke_aggregate(function, &params, 9000) == FDM_KV_AGGREGATE_OK);
            check_aggregate_pages(buffer);
            assert(fdm_kv_receive_release(buffer));
        }
    }
    g_free(data);
    g_free(meta);
}

static FDMKV_ProfileFunction profile_function;
static uint64_t profile_ticks;

static uint64_t profile_test_clock(void)
{
    return ++profile_ticks;
}

static long long invoke_profiled(struct cemu_args *args)
{
    FDMKV_AggregateProfile profile = {0};
    profile_ticks = 0;
    long long result = profile_function(args, &profile, profile_test_clock);
    uint64_t total = 0;
    for (unsigned stage = 0; stage < FDM_KV_AGG_STAGE_COUNT; stage++) {
        total += profile.host_ns[stage];
    }
    assert(total == profile_ticks - 1);
    assert(profile.calls == 1);
    bool success = result == FDM_KV_AGGREGATE_OK || result == FDM_KV_AGGREGATE_PROGRESS;
    assert(profile.successful_calls == success);
    if (success) {
        assert(profile.copied_bytes && profile.channel_elements);
        for (unsigned stage = 0; stage < FDM_KV_AGG_STAGE_COUNT; stage++) {
            assert(profile.host_ns[stage]);
        }
    }
    return result;
}

static void test_layer_packets(AggregateFunction function)
{
    void *meta = g_malloc(128 * 1024), *data = g_malloc(4 * 1024 * 1024);
    FDMKV_InitParams_t params = buffer_params(meta, data);
    params.input_bytes = 1024 * 1024;
    params.geometry.max_tokens = 300;
    params.block_pages = 256;
    FDMKV_Buffer_t *buffer;
    assert(fdm_kv_buffer_init(&params, &buffer));
    uint32_t first = 0;
    const uint32_t counts[] = {273, 1, 15};
    for (unsigned packet = 0; packet < G_N_ELEMENTS(counts); packet++) {
        uint32_t tokens = counts[packet];
        uint64_t payload = 2ULL * 2 * 2 * tokens * buffer->row_bytes;
        uint64_t bytes = 512 + payload;
        uint8_t *wire = g_malloc0(bytes);
        FDMKV_AppendHeader_t header = {
            .magic = cpu_to_le32(CEMU_FDM_KV_PACKET_MAGIC),
            .version = cpu_to_le32(CEMU_FDM_KV_LAYER_PACKET_VERSION),
            .header_bytes = cpu_to_le32(512), .operation = cpu_to_le32(FDM_KV_OP_APPEND),
            .session_id = cpu_to_le64(params.identity.session_id),
            .generation = cpu_to_le64(params.identity.generation),
            .request_id = cpu_to_le64(packet + 1),
            .wire_bytes = cpu_to_le64(bytes), .payload_bytes = cpu_to_le64(payload),
            .batch = 0, .batch_count = cpu_to_le32(2), .layer = cpu_to_le32(1),
            .head_count = cpu_to_le32(2), .token_first = cpu_to_le64(first),
            .token_count = cpu_to_le32(tokens),
        };
        memcpy(wire, &header, sizeof(header));
        for (uint32_t kind = 0; kind < 2; kind++) {
            for (uint32_t batch = 0; batch < 2; batch++) {
                for (uint32_t head = 0; head < 2; head++) {
                    for (uint32_t token = 0; token < tokens; token++) {
                        for (uint32_t dimension = 0; dimension < 128; dimension++) {
                            uint32_t sample = sample_value(batch, 1, head, first + token, dimension, kind);
                            uint64_t element = (((uint64_t)kind * 4 + batch * 2 + head) * tokens + token) * 128 + dimension;
                            memcpy(wire + 512 + element * 2, &sample, 2);
                        }
                    }
                }
            }
        }
        assert(receive_fragment(buffer, wire, 512, bytes - 512, 9000) == FDM_KV_RECEIVE_ACCEPTED);
        assert(receive_fragment(buffer, wire, 0, 508, 9000) == FDM_KV_RECEIVE_ACCEPTED);
        assert(receive_fragment(buffer, wire, 508, 4, 9000) == FDM_KV_RECEIVE_READY);
        expect_unchanged(function, &params, 8999, FDM_KV_AGGREGATE_WAIT_TIME);
        unsigned calls = 0;
        for (;;) {
            long long result = invoke_aggregate(function, &params, 9000);
            assert(result == FDM_KV_AGGREGATE_OK || result == FDM_KV_AGGREGATE_PROGRESS);
            check_aggregate_pages(buffer);
            assert(++calls < 100);
            if (result == FDM_KV_AGGREGATE_OK) {
                break;
            }
            assert(buffer->stats.append_commands == packet);
            assert(!fdm_kv_receive_release(buffer));
            assert(fdm_kv_receive_check(buffer, 0, 512) == FDM_KV_RECEIVE_BUSY);
            uint32_t batch = buffer->receive->aggregate_batch;
            FDMKV_HeadState_t *head = &buffer->head_states[(batch * 2 + 1) * 2];
            head->readers = 1;
            expect_unchanged(function, &params, 9000, FDM_KV_AGGREGATE_BUSY);
            head->readers = 0;
            if (!packet && calls == 1) {
                uint32_t free_pages = buffer->free_pages;
                buffer->free_pages = 0;
                expect_unchanged(function, &params, 9000, FDM_KV_AGGREGATE_NO_SPACE);
                buffer->free_pages = free_pages;
            }
        }
        first += tokens;
        assert(buffer->stats.append_commands == packet + 1);
        assert(buffer->stats.appended_head_tokens == 4ULL * first);
        for (uint32_t batch = 0; batch < 2; batch++) {
            for (uint32_t head = 0; head < 2; head++) {
                assert(buffer->head_states[(batch * 2 + 1) * 2 + head].accepted == first);
                assert(buffer->head_states[batch * 4 + head].accepted == 0);
            }
        }
        assert(fdm_kv_receive_release(buffer));
        assert(!buffer->receive->aggregate_batch && !buffer->receive->aggregate_tokens);
        assert(!buffer->receive->received_units && !buffer->receive->received_end);
        g_free(wire);
    }
    g_free(data);
    g_free(meta);
}

int main(int argc, char **argv)
{
    test_layout();
    test_partial_groups();
    test_invalid_layout();
    test_buffer_init();
    test_receive();
    puts("FDM KV layout/buffer/receive: initialization, out-of-order fragments, holes, duplicates, padding and READY without aggregation PASS");
    if (argc == 2) {
        void *library = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
        assert(library);
        AggregateFunction function = (AggregateFunction)dlsym(library, "fdm_kv_aggregate");
        assert(function);
        test_aggregate(function);
        test_segmented_aggregate(function);
        test_aggregate_plan_boundaries(function);
        test_layer_packets(function);
        profile_function = (FDMKV_ProfileFunction)dlsym(library, "fdm_kv_aggregate_profile_v1");
        assert(profile_function);
        test_aggregate(invoke_profiled);
        test_segmented_aggregate(invoke_profiled);
        test_aggregate_plan_boundaries(invoke_profiled);
        test_layer_packets(invoke_profiled);
        puts("FDM KV aggregate profiling: same layout/error/atomicity checks and additive Host stage accounting PASS");
        assert(dlclose(library) == 0);
        puts("FDM KV aggregate module: multi-head token-K/channel-K/V, group boundaries, domain queues, atomic preflight, virtual-time gate and input release PASS; no device dispatch/NAND/Attention");
    }
    return 0;
}
