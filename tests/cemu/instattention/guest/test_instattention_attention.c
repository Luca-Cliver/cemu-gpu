#define main native_storage_regression_main
#include "test_instattention_device_io.c"
#undef main
#include "hw/femu/bbssd/instattention_attention.h"

typedef struct NativeAttentionFixture {
    Fixture storage;
    NativeKvBuffer *buffer;
    NativeKvAttention attention;
    CemuSparfStorageResult reads[3];
    uint64_t starts[3];
    int fail_kind;
    gint stopped;
} NativeAttentionFixture;

void *native_attention_fixture_create(uint32_t batches, uint32_t layers,
                                      uint32_t heads, uint32_t element)
{
    NativeAttentionFixture *fixture = g_new0(NativeAttentionFixture, 1);
    CemuKvGeometry geometry;

    fixture_open_sized(&fixture->storage, 16, 4096);
    fixture->storage.params.pg_rd_lat = 400;
    fixture->storage.params.pg_wr_lat = 2000;
    fixture->storage.identity.task++;
    geometry = (CemuKvGeometry){ .batches = batches, .layers = layers, .heads = heads,
        .head_dim = 128, .max_tokens = 600, .channels_per_page = 8,
        .element_bytes = element, .page_bytes = 4096 };
    assert(native_kv_buffer_create(&fixture->storage.ns, &fixture->storage.identity,
        &geometry, 16 * MiB, 32, &fixture->buffer) == CEMU_KV_OK);
    return fixture;
}

void native_attention_fixture_free(void *opaque)
{
    NativeAttentionFixture *fixture = opaque;

    native_kv_attention_close(&fixture->attention);
    native_kv_buffer_free(fixture->buffer);
    fixture_close(&fixture->storage);
    g_free(fixture);
}

int native_attention_fixture_append(void *opaque, uint32_t batch, uint32_t layer,
    uint32_t head, uint32_t token, uint32_t count, void *key, void *value, size_t bytes)
{
    NativeAttentionFixture *fixture = opaque;
    CemuKvCoordinate coordinate = { .batch = batch, .layer = layer, .head = head, .token = token };

    return native_kv_buffer_append(fixture->buffer, &coordinate, count, key, value, bytes);
}

int native_attention_fixture_open(void *opaque,
    const struct cemu_sparf_workflow_header *header, uint32_t batch_first, uint32_t generation)
{
    NativeAttentionFixture *fixture = opaque;
    CemuKvIdentity identity = fixture->storage.identity;

    identity.generation = generation;
    fixture->fail_kind = 0;
    memset(fixture->reads, 0, sizeof(fixture->reads));
    memset(fixture->starts, 0, sizeof(fixture->starts));
    return native_kv_attention_open(fixture->buffer, &identity, batch_first, header, &fixture->attention);
}

void native_attention_fixture_close(void *opaque)
{
    NativeAttentionFixture *fixture = opaque;

    native_kv_attention_close(&fixture->attention);
}

static gpointer stop_fixture(gpointer opaque)
{
    NativeAttentionFixture *fixture = opaque;

    native_kv_buffer_stop(fixture->buffer);
    g_atomic_int_set(&fixture->stopped, 1);
    return NULL;
}

void native_attention_fixture_stop_leased(void *opaque)
{
    NativeAttentionFixture *fixture = opaque;
    CemuKvCoordinate coordinate = {0};
    NativeKvBufferStats stats = {0};
    GThread *worker;

    assert(fixture->attention.lease.buffer);
    worker = g_thread_new("stop-leased-buffer", stop_fixture, fixture);
    for (uint32_t attempt = 0; attempt < 1000; attempt++) {
        assert(native_kv_buffer_stats(fixture->buffer, &coordinate, &stats) == CEMU_KV_OK);
        if (stats.failed) {
            break;
        }
        g_usleep(1000);
    }
    assert(stats.failed && !g_atomic_int_get(&fixture->stopped));
    native_kv_attention_close(&fixture->attention);
    g_thread_join(worker);
    assert(g_atomic_int_get(&fixture->stopped));
}

bool native_attention_fixture_read(void *opaque,
    const struct cemu_sparf_workflow_header *header, const uint32_t *indices,
    uint32_t kind, void *output, size_t bytes, uint64_t start_ns, CemuSparfStorageResult *result)
{
    NativeAttentionFixture *fixture = opaque;

    if (fixture->fail_kind == kind + 1) {
        *result = (CemuSparfStorageResult){ .ready_ns = start_ns, .status = CEMU_KV_IO_FAILED };
        return false;
    }
    bool success = native_kv_attention_read(&fixture->attention, header, indices,
        kind, output, bytes, start_ns, result);
    fixture->reads[kind] = *result;
    fixture->starts[kind] = start_ns;
    return success;
}

void native_attention_fixture_fail(void *opaque, int kind)
{
    ((NativeAttentionFixture *)opaque)->fail_kind = kind;
}

uint64_t native_attention_fixture_time(uint64_t advance)
{
    qatomic_set(&virtual_now, qatomic_read(&virtual_now) + advance);
    g_usleep(1000);
    return qatomic_read(&virtual_now);
}

int native_attention_fixture_drain(void *opaque)
{
    return native_kv_buffer_drain(((NativeAttentionFixture *)opaque)->buffer);
}

uint64_t native_attention_fixture_result(void *opaque, uint32_t kind, uint32_t field)
{
    NativeAttentionFixture *fixture = opaque;
    assert(kind < 3);
    CemuSparfStorageResult *result = &fixture->reads[kind];
    uint64_t fields[] = { fixture->starts[kind], result->ready_ns, result->flash_bytes,
        result->tail_bytes, result->output_bytes, result->pages, result->batches, result->status };

    assert(kind < 3 && field < G_N_ELEMENTS(fields));
    return fields[field];
}
