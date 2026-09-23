#include <assert.h>
#include <stdio.h>
#include <string.h>
#include "hw/femu/bbssd/instattention_address.h"

static CemuKvGeometry opt_geometry(void)
{
    return (CemuKvGeometry) {
        .batches = 4, .layers = 40, .heads = 40, .head_dim = 128,
        .max_tokens = 2048, .channels_per_page = 8,
        .element_bytes = 2, .page_bytes = 4096,
    };
}

static void test_layout(void)
{
    CemuKvGeometry geometry = opt_geometry();
    CemuKvLayout layout;

    for (uint32_t channels = 2; channels <= 8; channels *= 2) {
        geometry.channels_per_page = channels;
        assert(cemu_kv_layout_init(&layout, &geometry));
        assert(layout.token_group_tokens == 16);
        assert(layout.channel_group_tokens == 2048 / channels);
        assert(layout.channel_group_pages == 128 / channels);
        assert(layout.streams == 6400);
        assert(layout.total_pages * geometry.page_bytes == UINT64_C(10066329600));
    }
    geometry.element_bytes = 4;
    assert(cemu_kv_layout_init(&layout, &geometry));
    assert(layout.token_group_tokens == 8);
    assert(layout.channel_group_tokens == 128);
    assert(layout.total_pages * geometry.page_bytes == UINT64_C(20132659200));
    geometry.max_tokens = 259;
    geometry.element_bytes = 2;
    assert(cemu_kv_layout_init(&layout, &geometry));
    assert(layout.pages_per_stream == 2 * 17 + 2 * 16);
    geometry.page_bytes = 8192;
    assert(cemu_kv_layout_init(&layout, &geometry));
    assert(layout.token_group_tokens == 32);
    assert(layout.channel_group_tokens == 512);
    assert(layout.pages_per_stream == 2 * 9 + 16);
}

static void test_invalid_layout(void)
{
    CemuKvGeometry geometry = opt_geometry();
    CemuKvLayout layout, saved;
    assert(cemu_kv_layout_init(&layout, &geometry));
    saved = layout;
    assert(!cemu_kv_layout_init(NULL, &geometry));
    assert(!cemu_kv_layout_init(&layout, NULL));
    for (unsigned int choice = 0; choice < 13; ++choice) {
        geometry = opt_geometry();
        switch (choice) {
        case 0: geometry.batches = 0; break;
        case 1: geometry.layers = 0; break;
        case 2: geometry.heads = 0; break;
        case 3: geometry.max_tokens = 0; break;
        case 4: geometry.head_dim = 127; break;
        case 5: geometry.channels_per_page = 0; break;
        case 6: geometry.element_bytes = 8; break;
        case 7: geometry.page_bytes = 4095; break;
        case 8: geometry.head_dim = 4096; break;
        case 9: geometry.batches = geometry.layers = geometry.heads = UINT32_MAX; break;
        case 10: geometry.max_tokens = UINT64_MAX; break;
        case 11: geometry.page_bytes = 131072; break;
        case 12:
            geometry.batches = geometry.layers = geometry.heads = 1;
            geometry.max_tokens = UINT64_C(1) << 55;
            break;
        }
        assert(!cemu_kv_layout_init(&layout, &geometry));
        assert(memcmp(&layout, &saved, sizeof(layout)) == 0);
    }
}

static void test_coordinates(void)
{
    CemuKvGeometry geometry = opt_geometry();
    CemuKvLayout layout;
    CemuKvIdentity identity = { .controller = 0, .namespace_id = 1, .task = 0, .generation = 1 };
    CemuKvCoordinate coordinate = { .batch = 3, .layer = 39, .head = 39 };
    CemuKvLocation location;
    const uint64_t tokens[] = {0, 15, 16, 253, 255, 256, 258};

    geometry.max_tokens = 259;
    for (uint32_t channels = 2; channels <= 8; channels *= 2) {
        geometry.channels_per_page = channels;
        for (uint32_t element = 2; element <= 4; element *= 2) {
            geometry.element_bytes = element;
            assert(cemu_kv_layout_init(&layout, &geometry));
            for (size_t position = 0; position < sizeof(tokens) / sizeof(tokens[0]); ++position) {
                coordinate.token = tokens[position];
                for (coordinate.kind = 0; coordinate.kind < 3; ++coordinate.kind) {
                    uint32_t capacity = coordinate.kind == CEMU_KV_PAGE_CHANNEL_K ?
                        4096 / (channels * element) : 4096 / (128 * element);
                    for (coordinate.channel = 0; coordinate.channel < 128; ++coordinate.channel) {
                        assert(cemu_kv_resolve(&layout, &identity, &coordinate, &location));
                        uint64_t linear;
                        if (coordinate.kind == CEMU_KV_PAGE_CHANNEL_K) {
                            linear = (coordinate.token / capacity * 128 + coordinate.channel) *
                                     capacity + coordinate.token % capacity;
                        } else {
                            linear = coordinate.token * 128 + coordinate.channel;
                        }
                        assert(location.page_index * 4096 + location.byte_offset == linear * element);
                        assert(location.byte_offset + element <= 4096);
                        uint64_t remaining = 259 - coordinate.token / capacity * capacity;
                        assert(location.token_capacity == (remaining < capacity ? remaining : capacity));
                    }
                }
            }
        }
    }
}

static void test_identity_and_page_dedup(void)
{
    CemuKvGeometry geometry = opt_geometry();
    CemuKvLayout layout;
    CemuKvIdentity identity = { .namespace_id = 1, .generation = 1 };
    CemuKvCoordinate coordinate = {0};
    CemuKvLocation first, second;
    assert(cemu_kv_layout_init(&layout, &geometry));
    assert(cemu_kv_resolve(&layout, &identity, &coordinate, &first));
    coordinate.token = 15;
    coordinate.channel = 127;
    assert(cemu_kv_resolve(&layout, &identity, &coordinate, &second));
    assert(cemu_kv_page_key_equal(&first.key, &second.key));
    for (unsigned int field = 0; field < 10; ++field) {
        second = first;
        switch (field) {
        case 0: ++second.key.identity.controller; break;
        case 1: ++second.key.identity.namespace_id; break;
        case 2: ++second.key.identity.task; break;
        case 3: ++second.key.identity.generation; break;
        case 4: ++second.key.batch; break;
        case 5: ++second.key.layer; break;
        case 6: ++second.key.head; break;
        case 7: ++second.key.kind; break;
        case 8: ++second.key.group; break;
        case 9: ++second.key.channel_group; break;
        }
        assert(!cemu_kv_page_key_equal(&first.key, &second.key));
    }
    coordinate = (CemuKvCoordinate){ .kind = CEMU_KV_PAGE_CHANNEL_K };
    assert(cemu_kv_resolve(&layout, &identity, &coordinate, &first));
    coordinate.token = 255;
    coordinate.channel = 7;
    assert(cemu_kv_resolve(&layout, &identity, &coordinate, &second));
    assert(cemu_kv_page_key_equal(&first.key, &second.key));
    coordinate.channel = 8;
    assert(cemu_kv_resolve(&layout, &identity, &coordinate, &second));
    assert(!cemu_kv_page_key_equal(&first.key, &second.key));
    coordinate.channel = 0;
    coordinate.token = 256;
    assert(cemu_kv_resolve(&layout, &identity, &coordinate, &second));
    assert(!cemu_kv_page_key_equal(&first.key, &second.key));
}

static void test_invalid_coordinates(void)
{
    CemuKvGeometry geometry = opt_geometry();
    CemuKvLayout layout;
    CemuKvLocation location = {0}, saved = location;
    assert(cemu_kv_layout_init(&layout, &geometry));
    for (unsigned int field = 0; field < 8; ++field) {
        CemuKvIdentity identity = { .namespace_id = 1, .generation = 1 };
        CemuKvCoordinate coordinate = {0};
        switch (field) {
        case 0: coordinate.batch = geometry.batches; break;
        case 1: coordinate.layer = geometry.layers; break;
        case 2: coordinate.head = geometry.heads; break;
        case 3: coordinate.token = geometry.max_tokens; break;
        case 4: coordinate.channel = geometry.head_dim; break;
        case 5: coordinate.kind = UINT32_MAX; break;
        case 6: identity.generation = 0; break;
        case 7: identity.namespace_id = 0; break;
        }
        assert(!cemu_kv_resolve(&layout, &identity, &coordinate, &location));
        assert(memcmp(&location, &saved, sizeof(location)) == 0);
    }
}

static void test_physical_offsets(void)
{
    CemuKvNandGeometry geometry = {2, 2, 2, 2, 4, 512};
    CemuKvPhysicalPage page;
    unsigned char backend[64 * 512] = {0};
    uint64_t expected_index = 0, offset;

    for (page.channel = 0; page.channel < 2; ++page.channel) {
        for (page.lun = 0; page.lun < 2; ++page.lun) {
            for (page.plane = 0; page.plane < 2; ++page.plane) {
                for (page.block = 0; page.block < 2; ++page.block) {
                    for (page.page = 0; page.page < 4; ++page.page) {
                        assert(cemu_kv_physical_offset(&geometry, &page, sizeof(backend), &offset));
                        assert(offset == expected_index * 512);
                        memset(backend + offset, (int)++expected_index, 512);
                    }
                }
            }
        }
    }
    assert(expected_index == 64);
    for (uint64_t index = 0; index < sizeof(backend); ++index) {
        assert(backend[index] == index / 512 + 1);
    }
    page = (CemuKvPhysicalPage){0};
    offset = UINT64_MAX;
    assert(!cemu_kv_physical_offset(&geometry, &page, sizeof(backend) - 1, &offset));
    assert(offset == UINT64_MAX);
    for (unsigned int field = 0; field < 5; ++field) {
        page = (CemuKvPhysicalPage){0};
        switch (field) {
        case 0: page.channel = 2; break;
        case 1: page.lun = 2; break;
        case 2: page.plane = 2; break;
        case 3: page.block = 2; break;
        case 4: page.page = 4; break;
        }
        assert(!cemu_kv_physical_offset(&geometry, &page, sizeof(backend), &offset));
    }
    page = (CemuKvPhysicalPage){0};
    geometry = (CemuKvNandGeometry){128, 256, 256, 65536, 65536, 65536};
    assert(!cemu_kv_physical_offset(&geometry, &page, UINT64_MAX, &offset));
    geometry = (CemuKvNandGeometry){129, 1, 1, 1, 1, 4096};
    assert(!cemu_kv_physical_offset(&geometry, &page, UINT64_MAX, &offset));
    geometry = (CemuKvNandGeometry){128, 256, 256, 4096, 4096, 65536};
    assert(!cemu_kv_physical_offset(&geometry, &page, UINT64_MAX, &offset));
    assert(offset == UINT64_MAX);
}

int main(void)
{
    test_layout();
    test_invalid_layout();
    test_coordinates();
    test_identity_and_page_dedup();
    test_invalid_coordinates();
    test_physical_offsets();
    puts("[instattention-address] PASS: geometry, dual-layout addresses, scoped keys, physical offsets and bounds; no live FTL or device IO");
    return 0;
}
