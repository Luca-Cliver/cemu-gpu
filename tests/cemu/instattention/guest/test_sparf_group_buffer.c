#include "hw/femu/csd/sparf_group_buffer.h"
#include "hw/femu/csd/sparf_wait.h"

static gint target_clock = 1000;

static int64_t buffered_clock(void)
{
    return g_atomic_int_get(&target_clock);
}

typedef struct GroupBufferTest {
    SparfGroupSession *session;
    uint8_t *nvm;
    uint64_t size;
    uint8_t *storage;
    uint64_t storage_bytes;
    GMutex lock;
    GCond wake;
    bool blocked;
    bool entered;
    bool fail;
    gint cancelled;
    gint target_wait;
    gint target_entered;
    uint64_t published_bytes;
    uint64_t read_bytes;
    uint64_t write_bytes;
    uint64_t write_calls;
    uint64_t max_write_bytes;
    uint64_t next_page;
    uint64_t *placement;
} GroupBufferTest;

static bool buffered_transfer(void *opaque, SparfWritebackPlan *plan, bool write,
                               uint64_t *start, uint64_t *end)
{
    GroupBufferTest *test = opaque;
    g_mutex_lock(&test->lock);
    if (write) {
        test->entered = true;
        g_cond_broadcast(&test->wake);
        while (test->blocked) g_cond_wait(&test->wake, &test->lock);
    }
    bool success = !test->fail;
    *start = 1000;
    *end = write ? 1500 : 1100;
    if (!plan->kv_pages || plan->kv_pages->len * 8 != plan->sectors->len) success = false;
    for (guint index = 0; success && index < plan->kv_pages->len; ++index) {
        CemuKvPageTag tag = g_array_index(plan->kv_pages, CemuKvPageTag, index);
        success = tag.slba == g_array_index(plan->sectors, uint64_t, index * 8) &&
                  tag.cookie == test->session->geometry.cookie &&
                  tag.kind <= CEMU_KV_PAGE_TOKEN_V;
    }
    for (guint index = 0; success && index < plan->sectors->len; ++index) {
        uint64_t offset = g_array_index(plan->sectors, uint64_t, index) * 512;
        if (offset > test->size || 512 > test->size - offset) {
            success = false;
            break;
        }
        uint8_t *disk = test->nvm + offset, *memory = plan->image + index * 512;
        memcpy(write ? disk : memory, write ? memory : disk, 512);
        /* Deterministic channel-first allocator, matching the FTL write
         * pointer order (no background traffic or GC in this harness). */
        if (write && offset % 4096 == 0)
            test->placement[offset / 4096] = test->next_page++;
    }
    if (success) {
        if (write) {
            uint64_t bytes = (uint64_t)plan->sectors->len * 512;
            test->write_bytes += bytes;
            ++test->write_calls;
            test->max_write_bytes = MAX(test->max_write_bytes, bytes);
        }
        else test->read_bytes += (uint64_t)plan->sectors->len * 512;
    }
    g_mutex_unlock(&test->lock);
    if (success && write && g_atomic_int_get(&test->target_wait)) {
        g_atomic_int_set(&test->target_entered, 1);
        success = sparf_wait_target(*end, buffered_clock, &test->cancelled);
    }
    return success;
}

static void buffered_cancel(void *opaque)
{
    GroupBufferTest *test = opaque;
    g_atomic_int_set(&test->cancelled, 1);
}

static void buffered_release(void *opaque)
{
    GroupBufferTest *test = opaque;
    test->published_bytes = test->session->buffer->write_bytes;
}

void group_buffer_test_target(void *opaque, int blocked)
{
    GroupBufferTest *test = opaque;
    g_atomic_int_set(&target_clock, blocked ? 1000 : 2000);
    g_atomic_int_set(&test->target_wait, 1);
}

int group_buffer_test_target_entered(void *opaque)
{
    return g_atomic_int_get(&((GroupBufferTest *)opaque)->target_entered);
}

uint64_t group_buffer_test_abort(void *opaque)
{
    GroupBufferTest *test = opaque;
    sparf_group_free(test->session);
    test->session = NULL;
    return test->published_bytes;
}

void *group_buffer_test_new(uint8_t *nvm, uint64_t size, uint8_t *storage, uint64_t storage_bytes)
{
    GroupBufferTest *test = g_new0(GroupBufferTest, 1);
    test->nvm = nvm;
    test->size = size;
    test->storage = storage;
    test->storage_bytes = storage_bytes;
    test->placement = g_new0(uint64_t, (size + 4095) / 4096);
    g_mutex_init(&test->lock);
    g_cond_init(&test->wake);
    return test;
}

void *group_buffer_test_session(void *opaque)
{
    return ((GroupBufferTest *)opaque)->session;
}

void group_buffer_test_gate(void *opaque, int blocked, int fail)
{
    GroupBufferTest *test = opaque;
    g_mutex_lock(&test->lock);
    test->blocked = blocked;
    test->fail = fail;
    test->entered = false;
    g_cond_broadcast(&test->wake);
    g_mutex_unlock(&test->lock);
}

int group_buffer_test_wait_write(void *opaque)
{
    GroupBufferTest *test = opaque;
    gint64 deadline = g_get_monotonic_time() + 2000000;
    g_mutex_lock(&test->lock);
    while (!test->entered && g_cond_wait_until(&test->wake, &test->lock, deadline)) {}
    bool entered = test->entered;
    g_mutex_unlock(&test->lock);
    return entered;
}

int group_buffer_test_run(void *opaque, const void *metadata, uint32_t length,
                           uint8_t *input, uint64_t bytes, uint64_t *output)
{
    GroupBufferTest *test = opaque;
    struct cemu_sparf_group_result result = {0};
    struct cemu_sparf_group_progress progress = {0};
    struct cemu_sparf_group_header header;
    if (length < sizeof(header)) return 0;
    memcpy(&header, metadata, sizeof(header));
    bool success;
    if (header.operation == CEMU_SPARF_GROUP_REGISTER_BUFFER) {
        if (test->session) return 0;
        test->session = sparf_group_buffer_register(metadata, length, test->size, 4096,
            test->storage, test->storage_bytes, buffered_transfer, NULL, buffered_cancel, buffered_release, test);
        success = test->session != NULL;
    } else success = sparf_group_buffer_execute(&test->session, metadata, length, input, bytes,
        test->size, 4096, buffered_transfer, test, &result, &progress);
    memcpy(output, &result, sizeof(result));
    memcpy((uint8_t *)output + sizeof(result), &progress, sizeof(progress));
    return success;
}

void group_buffer_test_totals(void *opaque, uint64_t *output)
{
    GroupBufferTest *test = opaque;
    g_mutex_lock(&test->lock);
    output[0] = test->read_bytes;
    output[1] = test->write_bytes;
    g_mutex_unlock(&test->lock);
}

void group_buffer_test_write_stats(void *opaque, uint64_t *output)
{
    GroupBufferTest *test = opaque;
    g_mutex_lock(&test->lock);
    output[0] = test->write_calls;
    output[1] = test->max_write_bytes;
    g_mutex_unlock(&test->lock);
}

void group_buffer_test_lun_pages(void *opaque, uint32_t kind, uint32_t channels,
                                uint32_t luns, uint64_t *output)
{
    GroupBufferTest *test = opaque;
    SparfGroupSession *session = test->session;
    memset(output, 0, channels * luns * sizeof(*output));
    g_mutex_lock(&session->buffer->lock);
    g_mutex_lock(&test->lock);
    GHashTableIter iter;
    gpointer key, value;
    g_hash_table_iter_init(&iter, session->directory);
    while (g_hash_table_iter_next(&iter, &key, &value)) {
        if (((SparfGroupKey *)key)->kind != kind) continue;
        uint32_t pages = kind == CEMU_SPARF_GROUP_CHANNEL_K ?
            session->geometry.head_dim / session->geometry.channels_per_page : 1;
        for (uint32_t i = 0; i < pages; ++i) {
            uint64_t sector = g_array_index(session->pages, uint64_t, *(uint64_t *)value + i);
            uint64_t ordinal = test->placement[sector / 8];
            ++output[(ordinal % channels) * luns + (ordinal / channels) % luns];
        }
    }
    g_mutex_unlock(&test->lock);
    g_mutex_unlock(&session->buffer->lock);
}

void group_buffer_test_free(void *opaque)
{
    GroupBufferTest *test = opaque;
    group_buffer_test_gate(test, false, test->fail);
    sparf_group_free(test->session);
    g_mutex_clear(&test->lock);
    g_cond_clear(&test->wake);
    g_free(test->placement);
    g_free(test);
}
