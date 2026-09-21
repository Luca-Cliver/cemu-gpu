#include "hw/femu/csd/sparf_group.h"

typedef struct GroupTest {
    SparfGroupSession *session;
    uint8_t *nvm;
    uint64_t size;
    uint64_t now;
    uint32_t calls;
    uint32_t fail;
} GroupTest;

uint32_t group_test_extents(const uint64_t *values, uint32_t length,
                            uint32_t sectors_per_page, uint64_t *starts, uint32_t *counts)
{
    GArray *sectors = g_array_sized_new(FALSE, FALSE, sizeof(uint64_t), length);
    g_array_append_vals(sectors, values, length);
    uint32_t extents = 0;
    for (guint index = 0; index < length;) {
        guint count = sparf_wb_extent_sectors(sectors, index, sectors_per_page);
        if (!count) {
            extents = UINT32_MAX;
            break;
        }
        starts[extents] = values[index];
        counts[extents++] = count;
        index += count;
    }
    g_array_free(sectors, TRUE);
    return extents;
}

int group_test_plan_ordinals(uint32_t pages)
{
    if (pages < 8 || pages % 2) return false;
    struct {
        struct cemu_sparf_group_header header;
        struct cemu_sparf_group_extent extents[2];
    } packet = {
        .header = { .cookie = 9876, .count = 2, .page_bytes = 4096,
                    .head_dim = 128, .channels_per_page = 8, .element_bytes = 2 },
        .extents = {
            { .sector = (uint64_t)pages * 4 + 8, .sectors = (uint64_t)pages * 4 },
            { .sector = 8, .sectors = (uint64_t)pages * 4 },
        },
    };
    SparfGroupSession *session = sparf_group_register(&packet.header, &packet, sizeof(packet),
                                                     ((uint64_t)pages + 1) * 4096, 4096);
    if (!session) return false;
    guint ordinals[] = { pages - 1, pages / 2 - 1, pages / 2, 0 };
    GHashTable *selected = g_hash_table_new(g_direct_hash, g_direct_equal);
    for (guint index = 0; index < G_N_ELEMENTS(ordinals); ++index) {
        g_hash_table_add(selected, GUINT_TO_POINTER(ordinals[index] + 1));
    }
    SparfWritebackPlan *plan = sparf_group_plan(session, selected);
    bool success = plan != NULL;
    for (guint index = 0; index < G_N_ELEMENTS(ordinals); ++index) {
        guint page = ordinals[index];
        CemuKvPageTag *tag = &g_array_index(session->page_tags, CemuKvPageTag, page);
        tag->stream = page;
        tag->kind = page % 3;
        tag->group = page / 16;
        tag->page_in_group = page % 16;
    }
    success = success && sparf_group_refresh_plan_tags(session, plan);
    for (guint index = 0; success && index < plan->kv_pages->len; ++index) {
        guint page = g_array_index(plan->kv_page_ordinals, guint, index);
        CemuKvPageTag tag = g_array_index(plan->kv_pages, CemuKvPageTag, index);
        uint64_t sector = g_array_index(plan->sectors, uint64_t, index * 8);
        success = tag.slba == sector && tag.cookie == packet.header.cookie &&
                  tag.stream == page && tag.kind == page % 3 &&
                  tag.group == page / 16 && tag.page_in_group == page % 16 &&
                  sector == g_array_index(session->pages, uint64_t, page);
        if (index && g_array_index(plan->sectors, uint64_t, index * 8 - 1) >= sector) success = false;
    }
    sparf_wb_free(plan);
    g_hash_table_destroy(selected);
    sparf_group_free(session);
    return success;
}

uint32_t group_test_distinct_hashes(uint64_t stream_base, uint64_t group_base,
                                     uint32_t streams, uint32_t groups)
{
    GHashTable *hashes = g_hash_table_new(g_direct_hash, g_direct_equal);
    for (uint32_t stream = 0; stream < streams; ++stream) {
        for (uint32_t group = 0; group < groups; ++group) {
            for (uint32_t kind = 0; kind < 3; ++kind) {
                SparfGroupKey key = { .stream = stream_base + stream,
                    .group = group_base + group, .kind = kind };
                g_hash_table_add(hashes, GUINT_TO_POINTER(sparf_group_hash(&key)));
            }
        }
    }
    uint32_t count = g_hash_table_size(hashes);
    g_hash_table_destroy(hashes);
    return count;
}

static bool group_transfer(void *opaque, SparfWritebackPlan *plan, bool write,
                           uint64_t *start, uint64_t *end)
{
    GroupTest *test = opaque;
    ++test->calls;
    *start = test->now;
    if (test->fail == test->calls) return false;
    if (!plan->kv_pages || plan->kv_pages->len * 8 != plan->sectors->len) return false;
    for (guint index = 0; index < plan->kv_pages->len; ++index) {
        CemuKvPageTag tag = g_array_index(plan->kv_pages, CemuKvPageTag, index);
        if (tag.slba != g_array_index(plan->sectors, uint64_t, index * 8) ||
            tag.cookie != test->session->geometry.cookie ||
            tag.kind > CEMU_KV_PAGE_TOKEN_V) return false;
    }
    for (guint index = 0; index < plan->sectors->len; ++index) {
        uint64_t offset = g_array_index(plan->sectors, uint64_t, index) * 512;
        if (offset > test->size || 512 > test->size - offset) return false;
        uint8_t *disk = test->nvm + offset;
        uint8_t *memory = plan->image + index * 512;
        memcpy(write ? disk : memory, write ? memory : disk, 512);
    }
    test->now += write ? 500 : 100;
    *end = test->now;
    return true;
}

void *group_test_new(uint8_t *nvm, uint64_t size)
{
    GroupTest *test = g_new0(GroupTest, 1);
    test->nvm = nvm;
    test->size = size;
    test->now = 1000;
    return test;
}

int group_test_run(void *opaque, const void *metadata, uint32_t length,
                   uint8_t *buffer, uint64_t buffer_bytes, int fail,
                   struct cemu_sparf_group_result *result)
{
    GroupTest *test = opaque;
    test->calls = 0;
    test->fail = fail;
    bool success = sparf_group_execute(&test->session, metadata, length, buffer, buffer_bytes,
                                       test->size, 4096, group_transfer, test, result);
    result->io.ready = test->now;
    return success ? test->calls : -1;
}

void group_test_free(void *opaque)
{
    GroupTest *test = opaque;
    sparf_group_free(test->session);
    g_free(test);
}
