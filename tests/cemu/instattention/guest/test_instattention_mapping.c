#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "hw/femu/bbssd/instattention_mapping.h"

static size_t allocation_attempts;
static size_t fail_allocation;
static size_t live_allocations;

void *__real_calloc(size_t count, size_t width);
void __real_free(void *pointer);

void *__wrap_calloc(size_t count, size_t width)
{
    void *pointer;

    if (++allocation_attempts == fail_allocation) {
        return NULL;
    }
    pointer = __real_calloc(count, width);
    if (pointer) {
        live_allocations++;
    }
    return pointer;
}

void __wrap_free(void *pointer)
{
    if (pointer) {
        assert(live_allocations);
        live_allocations--;
    }
    __real_free(pointer);
}

static const CemuKvIdentity identity = { .controller = 2, .namespace_id = 1,
                                         .task = 0, .generation = 1 };

static CemuKvGeometry geometry(void)
{
    return (CemuKvGeometry){ .batches = 2, .layers = 2, .heads = 3, .head_dim = 128,
                            .max_tokens = 35, .channels_per_page = 8,
                            .element_bytes = 2, .page_bytes = 512 };
}

static CemuKvMapConfig configuration(void)
{
    return (CemuKvMapConfig){
        .controller = 2, .namespace_id = 1,
        .nand = { .channels = 2, .luns_per_channel = 2, .planes_per_lun = 2,
                  .blocks_per_plane = 64, .pages_per_block = 8, .page_bytes = 512 },
        .backend_bytes = 2 * 1024 * 1024,
        .max_tasks = 2, .max_pages = 1024, .memory_budget_bytes = 1024 * 1024,
    };
}

static CemuKvMap *create_map(CemuKvMapConfig config)
{
    CemuKvMap *map = NULL;
    CemuKvGeometry shape = geometry();

    assert(cemu_kv_map_create(&config, &map) == CEMU_KV_OK);
    assert(cemu_kv_map_register(map, &identity, &shape) == CEMU_KV_OK);
    return map;
}

static bool same_block(CemuKvPhysicalPage left, CemuKvPhysicalPage right)
{
    return left.channel == right.channel && left.lun == right.lun &&
           left.plane == right.plane && left.block == right.block;
}

static void test_identity(void)
{
    CemuKvMap *map = create_map(configuration());
    CemuKvGeometry shape = geometry();
    CemuKvIdentity other = identity;
    CemuKvCoordinate start = {0};
    CemuKvPageRef first, second, output;
    uint32_t offset = 123;

    assert(cemu_kv_map_register(map, &identity, &shape) == CEMU_KV_EXISTS);
    other.controller++;
    assert(cemu_kv_map_register(map, &other, &shape) == CEMU_KV_INVALID);
    other = identity;
    other.namespace_id++;
    assert(cemu_kv_map_register(map, &other, &shape) == CEMU_KV_INVALID);
    other = identity;
    other.generation++;
    assert(cemu_kv_map_register(map, &other, &shape) == CEMU_KV_STALE);
    assert(cemu_kv_map_lookup(map, &other, &start, 0, &output, &offset) == CEMU_KV_STALE);
    other.generation = 0;
    assert(cemu_kv_map_register(map, &other, &shape) == CEMU_KV_INVALID);
    other = identity;
    other.task++;
    assert(cemu_kv_map_lookup(map, &other, &start, 0, &output, &offset) == CEMU_KV_NOT_FOUND);
    assert(cemu_kv_map_register(map, &other, &shape) == CEMU_KV_OK);
    assert(cemu_kv_map_reserve(map, &identity, &start, 2, &first) == CEMU_KV_OK);
    assert(cemu_kv_map_reserve(map, &other, &start, 2, &second) == CEMU_KV_OK);
    assert(!same_block(first.physical, second.physical));
    assert(first.backend_offset != second.backend_offset);
    other.task++;
    assert(cemu_kv_map_register(map, &other, &shape) == CEMU_KV_LIMIT);
    assert(cemu_kv_map_stats(map).tasks == 2);
    assert(offset == 123);
    cemu_kv_map_destroy(map);
}

static void test_visibility(void)
{
    CemuKvMap *map = create_map(configuration());
    CemuKvCoordinate start = {0};
    CemuKvPageRef reserved, forged, output;
    unsigned char unchanged[sizeof(output)];
    uint32_t offset = 123;

    memset(&output, 0xa5, sizeof(output));
    memcpy(unchanged, &output, sizeof(output));
    assert(cemu_kv_map_reserve(map, &identity, &start, 1, &reserved) == CEMU_KV_OK);
    assert(cemu_kv_map_reserve(map, &identity, &start, 1, &output) == CEMU_KV_EXISTS);
    assert(cemu_kv_map_lookup(map, &identity, &start, UINT64_MAX, &output, &offset) == CEMU_KV_PENDING);
    forged = reserved;
    forged.key.identity.generation++;
    assert(cemu_kv_map_complete(map, &forged, true, 100) == CEMU_KV_STALE);
    forged = reserved;
    forged.backend_offset++;
    assert(cemu_kv_map_complete(map, &forged, true, 100) == CEMU_KV_INVALID);
    forged = reserved;
    forged.physical.page++;
    assert(cemu_kv_map_complete(map, &forged, true, 100) == CEMU_KV_INVALID);
    forged = reserved;
    forged.valid_tokens++;
    assert(cemu_kv_map_complete(map, &forged, true, 100) == CEMU_KV_INVALID);
    assert(cemu_kv_map_complete(map, &reserved, true, 100) == CEMU_KV_OK);
    assert(cemu_kv_map_complete(map, &reserved, false, 0) == CEMU_KV_EXISTS);
    assert(cemu_kv_map_lookup(map, &identity, &start, 99, &output, &offset) == CEMU_KV_PENDING);
    assert(memcmp(&output, unchanged, sizeof(output)) == 0 && offset == 123);
    assert(cemu_kv_map_lookup(map, &identity, &start, 100, &output, &offset) == CEMU_KV_OK);
    assert(output.ticket == reserved.ticket && offset == 0);
    start.token = 1;
    assert(cemu_kv_map_lookup(map, &identity, &start, 100, &output, &offset) == CEMU_KV_NOT_FOUND);
    start.token = 2;
    assert(cemu_kv_map_reserve(map, &identity, &start, 2, &reserved) == CEMU_KV_OK);
    assert(cemu_kv_map_complete(map, &reserved, false, 0) == CEMU_KV_OK);
    assert(cemu_kv_map_lookup(map, &identity, &start, UINT64_MAX, &output, &offset) == CEMU_KV_IO_FAILED);
    assert(cemu_kv_map_reserve(map, &identity, &start, 2, &output) == CEMU_KV_EXISTS);
    assert(cemu_kv_map_stats(map).reserved_pages == 2);
    assert(cemu_kv_map_stats(map).written_pages == 1);
    assert(cemu_kv_map_stats(map).failed_pages == 1);
    cemu_kv_map_destroy(map);
}

static uint16_t value_at(const CemuKvCoordinate *coordinate)
{
    return (uint16_t)(coordinate->batch * 17011 + coordinate->layer * 7919 +
                     coordinate->head * 1009 + coordinate->token * 131 +
                     coordinate->channel +
                     (coordinate->kind == CEMU_KV_PAGE_TOKEN_V ? 30001 : 0));
}

static void test_data_and_placement(uint32_t page_bytes, uint32_t channels_per_page,
                                    uint64_t tokens)
{
    CemuKvMapConfig config = configuration();
    CemuKvMap *map = NULL;
    CemuKvGeometry shape = geometry();
    CemuKvCoordinate coordinate = {0};
    CemuKvPageRef reference, head_zero;
    unsigned char *backend;
    bool *occupied;
    uint32_t capacity, channel_step, valid, within, channel, offset;
    uint16_t value, actual;
    size_t pages = 0, expected_pages;

    config.backend_bytes *= page_bytes / config.nand.page_bytes;
    config.nand.page_bytes = page_bytes;
    config.max_pages = 2048;
    shape.page_bytes = page_bytes;
    shape.channels_per_page = channels_per_page;
    shape.max_tokens = tokens;
    assert(cemu_kv_map_create(&config, &map) == CEMU_KV_OK);
    assert(cemu_kv_map_register(map, &identity, &shape) == CEMU_KV_OK);
    backend = calloc(1, config.backend_bytes);
    occupied = calloc(config.backend_bytes / shape.page_bytes, sizeof(bool));
    assert(backend && occupied);
    for (coordinate.batch = 0; coordinate.batch < shape.batches; coordinate.batch++) {
        for (coordinate.layer = 0; coordinate.layer < shape.layers; coordinate.layer++) {
            for (coordinate.kind = 0; coordinate.kind < 3; coordinate.kind++) {
                channel_step = coordinate.kind == CEMU_KV_PAGE_CHANNEL_K ?
                               shape.channels_per_page : shape.head_dim;
                capacity = shape.page_bytes / (channel_step * shape.element_bytes);
                for (coordinate.token = 0; coordinate.token < shape.max_tokens;
                     coordinate.token += capacity) {
                    valid = shape.max_tokens - coordinate.token < capacity ?
                            shape.max_tokens - coordinate.token : capacity;
                    for (coordinate.channel = 0; coordinate.channel < shape.head_dim;
                         coordinate.channel += channel_step) {
                        for (coordinate.head = 0; coordinate.head < shape.heads; coordinate.head++) {
                            assert(cemu_kv_map_reserve(map, &identity, &coordinate, valid,
                                                       &reference) == CEMU_KV_OK);
                            assert(reference.physical.channel ==
                                   (coordinate.token / capacity) % config.nand.channels);
                            if (!coordinate.token && !coordinate.channel) {
                                if (!coordinate.head) {
                                    head_zero = reference;
                                } else {
                                    assert(same_block(head_zero.physical, reference.physical));
                                    assert(reference.physical.page == coordinate.head);
                                }
                            }
                            assert(!occupied[reference.backend_offset / shape.page_bytes]);
                            occupied[reference.backend_offset / shape.page_bytes] = true;
                            pages++;
                            for (within = 0; within < valid; within++) {
                                for (channel = 0; channel < channel_step; channel++) {
                                    CemuKvCoordinate element = coordinate;

                                    element.token += within;
                                    element.channel += channel;
                                    offset = coordinate.kind == CEMU_KV_PAGE_CHANNEL_K ?
                                             (channel * capacity + within) * 2 :
                                             (within * shape.head_dim + channel) * 2;
                                    value = value_at(&element);
                                    memcpy(backend + reference.backend_offset + offset, &value, 2);
                                }
                            }
                            assert(cemu_kv_map_complete(map, &reference, true, 50) == CEMU_KV_OK);
                        }
                    }
                }
            }
        }
    }
    for (coordinate.batch = 0; coordinate.batch < shape.batches; coordinate.batch++) {
        for (coordinate.layer = 0; coordinate.layer < shape.layers; coordinate.layer++) {
            for (coordinate.head = 0; coordinate.head < shape.heads; coordinate.head++) {
                for (coordinate.kind = 0; coordinate.kind < 3; coordinate.kind++) {
                    for (coordinate.token = 0; coordinate.token < shape.max_tokens; coordinate.token++) {
                        for (coordinate.channel = 0; coordinate.channel < shape.head_dim; coordinate.channel++) {
                            assert(cemu_kv_map_lookup(map, &identity, &coordinate, 50,
                                                     &reference, &offset) == CEMU_KV_OK);
                            memcpy(&actual, backend + reference.backend_offset + offset, 2);
                            assert(actual == value_at(&coordinate));
                        }
                    }
                }
            }
        }
    }
    capacity = shape.page_bytes / (shape.head_dim * shape.element_bytes);
    expected_pages = 2 * ((shape.max_tokens + capacity - 1) / capacity);
    capacity = shape.page_bytes / (shape.channels_per_page * shape.element_bytes);
    expected_pages += ((shape.max_tokens + capacity - 1) / capacity) *
                      (shape.head_dim / shape.channels_per_page);
    expected_pages *= shape.batches * shape.layers * shape.heads;
    assert(pages == expected_pages && cemu_kv_map_stats(map).written_pages == pages);
    free(occupied);
    free(backend);
    cemu_kv_map_destroy(map);
}

static void test_exhaustion(void)
{
    CemuKvMapConfig config = configuration();
    CemuKvMap *map;
    CemuKvCoordinate start = {0};
    CemuKvPageRef reference, output;
    CemuKvMapStats before, after;
    unsigned ordinal;

    config.nand.blocks_per_plane = 1;
    config.nand.pages_per_block = 1;
    map = create_map(config);
    for (ordinal = 0; ordinal < 4; ordinal++) {
        start.token = ordinal * 4;
        assert(cemu_kv_map_reserve(map, &identity, &start, 2, &reference) == CEMU_KV_OK);
        assert(reference.physical.channel == 0 && reference.physical.block == 0);
        assert(reference.physical.lun == ordinal % 2);
        assert(reference.physical.plane == ordinal / 2);
        assert(cemu_kv_map_complete(map, &reference, ordinal != 0, 0) == CEMU_KV_OK);
    }
    before = cemu_kv_map_stats(map);
    start.token = 16;
    memset(&output, 0x35, sizeof(output));
    memcpy(&reference, &output, sizeof(output));
    assert(cemu_kv_map_reserve(map, &identity, &start, 2, &output) == CEMU_KV_NO_SPACE);
    assert(memcmp(&output, &reference, sizeof(output)) == 0);
    after = cemu_kv_map_stats(map);
    assert(after.reserved_pages == before.reserved_pages &&
           after.claimed_blocks == before.claimed_blocks);
    start.token = 2;
    assert(cemu_kv_map_reserve(map, &identity, &start, 2, &reference) == CEMU_KV_OK);
    assert(reference.physical.channel == 1);
    cemu_kv_map_destroy(map);
    config = configuration();
    config.max_pages = 1;
    map = create_map(config);
    start.token = 0;
    assert(cemu_kv_map_reserve(map, &identity, &start, 2, &reference) == CEMU_KV_OK);
    start.token = 2;
    assert(cemu_kv_map_reserve(map, &identity, &start, 2, &reference) == CEMU_KV_LIMIT);
    assert(cemu_kv_map_stats(map).claimed_blocks == 1);
    cemu_kv_map_destroy(map);
}

static void test_invalid(void)
{
    CemuKvMapConfig config = configuration();
    CemuKvMap *map = NULL;
    CemuKvGeometry shape = geometry();
    CemuKvCoordinate start = {0};
    CemuKvPageRef output = {0};
    CemuKvIdentity other = identity;
    size_t required;

    config.backend_bytes = 1;
    assert(cemu_kv_map_create(&config, &map) == CEMU_KV_INVALID && !map);
    config = configuration();
    config.memory_budget_bytes = 1;
    assert(cemu_kv_map_create(&config, &map) == CEMU_KV_LIMIT && !map);
    config = configuration();
    config.max_pages = SIZE_MAX;
    assert(cemu_kv_map_create(&config, &map) == CEMU_KV_INVALID && !map);
    config = configuration();
    config.max_tasks = SIZE_MAX;
    assert(cemu_kv_map_create(&config, &map) == CEMU_KV_LIMIT && !map);
    config = configuration();
    map = create_map(config);
    required = cemu_kv_map_stats(map).memory_bytes;
    assert(required <= config.memory_budget_bytes);
    assert(cemu_kv_map_reserve(map, &identity, &start, 0, &output) == CEMU_KV_INVALID);
    assert(cemu_kv_map_reserve(map, &identity, &start, 3, &output) == CEMU_KV_INVALID);
    start.token = 1;
    assert(cemu_kv_map_reserve(map, &identity, &start, 1, &output) == CEMU_KV_INVALID);
    start.token = shape.max_tokens;
    assert(cemu_kv_map_reserve(map, &identity, &start, 1, &output) == CEMU_KV_INVALID);
    start.token = 0;
    start.kind = 3;
    assert(cemu_kv_map_reserve(map, &identity, &start, 1, &output) == CEMU_KV_INVALID);
    start.kind = CEMU_KV_PAGE_CHANNEL_K;
    start.channel = 1;
    assert(cemu_kv_map_reserve(map, &identity, &start, 1, &output) == CEMU_KV_INVALID);
    assert(cemu_kv_map_complete(map, &output, true, 0) == CEMU_KV_INVALID);
    other.task++;
    shape.page_bytes *= 2;
    assert(cemu_kv_map_register(map, &other, &shape) == CEMU_KV_INVALID);
    assert(cemu_kv_map_stats(map).reserved_pages == 0);
    assert(cemu_kv_map_stats(map).claimed_blocks == 0);
    cemu_kv_map_destroy(map);
    config.memory_budget_bytes = required;
    assert(cemu_kv_map_create(&config, &map) == CEMU_KV_OK);
    cemu_kv_map_destroy(map);
    map = NULL;
    config.memory_budget_bytes--;
    assert(cemu_kv_map_create(&config, &map) == CEMU_KV_LIMIT && !map);
    cemu_kv_map_destroy(NULL);
}

static void test_allocation_failure(void)
{
    CemuKvMapConfig config = configuration();
    CemuKvMap *map = NULL;
    size_t failure;

    for (failure = 1; failure <= 6; failure++) {
        allocation_attempts = 0;
        fail_allocation = failure;
        assert(cemu_kv_map_create(&config, &map) == CEMU_KV_NO_MEMORY);
        assert(!map && !live_allocations);
    }
    fail_allocation = 0;
    assert(cemu_kv_map_create(&config, &map) == CEMU_KV_OK);
    assert(live_allocations == 6);
    cemu_kv_map_destroy(map);
    assert(!live_allocations);
}

int main(int argc, char **argv)
{
    assert(argc == 2);
    if (strcmp(argv[1], "identity") == 0) {
        test_identity();
    } else if (strcmp(argv[1], "visibility") == 0) {
        test_visibility();
    } else if (strcmp(argv[1], "data") == 0) {
        test_data_and_placement(512, 8, 35);
    } else if (strcmp(argv[1], "paper") == 0) {
        test_data_and_placement(4096, 2, 259);
        test_data_and_placement(4096, 4, 259);
        test_data_and_placement(4096, 8, 259);
    } else if (strcmp(argv[1], "exhaustion") == 0) {
        test_exhaustion();
    } else if (strcmp(argv[1], "invalid") == 0) {
        test_invalid();
    } else if (strcmp(argv[1], "allocation") == 0) {
        test_allocation_failure();
    } else {
        return 2;
    }
    assert(!live_allocations);
    printf("[instattention-mapping] PASS %s; isolated core, no live device IO\n", argv[1]);
    return 0;
}
