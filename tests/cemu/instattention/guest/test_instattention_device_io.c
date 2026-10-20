#include "qemu/osdep.h"
#include "hw/femu/bbssd/bb.h"
#include <sys/mman.h>

static void unexpected_cuda_lock(QemuMutex *mutex, const char *file, int line)
{
    (void)mutex;
    (void)file;
    (void)line;
    abort();
}

QemuMutexLockFunc qemu_mutex_lock_func = unexpected_cuda_lock;

void qemu_mutex_unlock_impl(QemuMutex *mutex, const char *file, int line)
{
    (void)mutex;
    (void)file;
    (void)line;
    abort();
}

static int64_t virtual_now = 100;

int64_t qemu_clock_get_ns(QEMUClockType type)
{
    assert(type == QEMU_CLOCK_VIRTUAL);
    return qatomic_read(&virtual_now);
}

typedef struct Fixture {
    FemuCtrl controller;
    SsdBackend backend;
    NvmeNamespace ns;
    NvmParams params;
    NvmNamespace nns;
    struct nand_lun luns[2][2];
    struct ssd_channel channels[2];
    CemuKvIdentity identity;
    CemuKvGeometry geometry;
    int descriptor;
    GMutex completion_lock;
    GQueue completions;
    GThread *completion_worker;
    gint completion_stop;
} Fixture;

static gint completion_order(gconstpointer left, gconstpointer right, gpointer opaque)
{
    const NvmeRequest *first = left, *second = right;
    return (first->stat.expire_time > second->stat.expire_time) -
           (first->stat.expire_time < second->stat.expire_time);
}

bool nvme_submit_internal_completion(FemuCtrl *controller, NvmeRequest *request)
{
    Fixture *fixture = container_of(controller, Fixture, controller);
    g_assert(request->internal_complete && request->internal_owner && !request->sq);
    g_mutex_lock(&fixture->completion_lock);
    g_queue_insert_sorted(&fixture->completions, request, completion_order, NULL);
    g_mutex_unlock(&fixture->completion_lock);
    return true;
}

void nvme_cancel_internal_completions(FemuCtrl *controller, void *owner)
{
    Fixture *fixture = container_of(controller, Fixture, controller);
    g_mutex_lock(&fixture->completion_lock);
    for (GList *entry = fixture->completions.head; entry;) {
        GList *next = entry->next;
        NvmeRequest *request = entry->data;
        if (request->internal_owner == owner) {
            g_queue_delete_link(&fixture->completions, entry);
            request->internal_complete(request, true);
        }
        entry = next;
    }
    g_mutex_unlock(&fixture->completion_lock);
}

static void *completion_poller(void *opaque)
{
    Fixture *fixture = opaque;
    while (!qatomic_read(&fixture->completion_stop)) {
        g_mutex_lock(&fixture->completion_lock);
        while (!g_queue_is_empty(&fixture->completions)) {
            NvmeRequest *request = g_queue_peek_head(&fixture->completions);
            if (request->stat.expire_time > qatomic_read(&virtual_now)) {
                break;
            }
            g_queue_pop_head(&fixture->completions);
            request->internal_complete(request, false);
        }
        g_mutex_unlock(&fixture->completion_lock);
        g_usleep(10);
    }
    return NULL;
}

static void fixture_open_sized(Fixture *fixture, uint32_t backend_mib, size_t page_limit)
{
    char path[] = "/tmp/cemu-native-internal-XXXXXX";
    CemuKvIo *io = NULL;

    memset(fixture, 0, sizeof(*fixture));
    qatomic_set(&virtual_now, 100);
    fixture->descriptor = mkstemp(path);
    assert(fixture->descriptor >= 0 && unlink(path) == 0);
    fixture->backend.size = (uint64_t)backend_mib * MiB;
    assert(ftruncate(fixture->descriptor, fixture->backend.size) == 0);
    fixture->backend.logical_space = mmap(NULL, fixture->backend.size,
        PROT_READ | PROT_WRITE, MAP_SHARED, fixture->descriptor, 0);
    assert(fixture->backend.logical_space != MAP_FAILED);
    fixture->controller.drive_id = 1;
    fixture->ns = (NvmeNamespace){ .ctrl = &fixture->controller, .id = 1,
        .csi = NVME_CSI_NVM, .backend = &fixture->backend, .private = &fixture->nns };
    fixture->params = (NvmParams){ .nchs = 2, .luns_per_ch = 2, .pls_per_lun = 1,
        .blks_per_pl = backend_mib * 4, .pgs_per_blk = 16, .secsz = 512, .secs_per_pg = 8,
        .pg_rd_lat = 40000000, .pg_wr_lat = 2000000000 };
    for (int channel = 0; channel < 2; channel++) {
        fixture->channels[channel].lun = fixture->luns[channel];
    }
    fixture->nns.namespace = &fixture->ns;
    fixture->nns.params = &fixture->params;
    fixture->nns.ch = fixture->channels;
    fixture->identity = (CemuKvIdentity){ .controller = 1, .namespace_id = 1,
        .task = 7, .generation = 1 };
    fixture->geometry = (CemuKvGeometry){ .batches = 1, .layers = 1, .heads = 16,
        .head_dim = 128, .max_tokens = 259, .channels_per_page = 8,
        .element_bytes = 2, .page_bytes = 4096 };
    assert(bb_native_io_create(&fixture->ns, 2, page_limit, 8 * MiB, &io) == CEMU_KV_OK);
    assert(bb_native_kv_register(&fixture->ns, &fixture->identity,
                                &fixture->geometry) == CEMU_KV_OK);
    g_mutex_init(&fixture->completion_lock);
    g_queue_init(&fixture->completions);
    fixture->completion_worker = g_thread_new("completion-test", completion_poller, fixture);
}

static void fixture_open(Fixture *fixture)
{
    fixture_open_sized(fixture, 4, 256);
}

static void fixture_close(Fixture *fixture)
{
    qatomic_set(&fixture->completion_stop, true);
    g_thread_join(fixture->completion_worker);
    assert(g_queue_is_empty(&fixture->completions));
    g_mutex_clear(&fixture->completion_lock);
    bb_native_io_destroy(&fixture->ns);
    g_mutex_clear(&fixture->nns.native_io_lock);
    assert(munmap(fixture->backend.logical_space, fixture->backend.size) == 0);
    assert(close(fixture->descriptor) == 0);
}

static void test_dependencies_and_error(void)
{
    Fixture fixture;
    CemuKvCoordinate coordinate = {0};
    CemuKvIoResult first, second, read;
    CemuKvMapStats stats;
    CemuKvIdentity wrong;
    uint64_t deadline;
    uint8_t input[4096], output[4096], disk[4096];
    void *space;

    fixture_open(&fixture);
    memset(input, 0x73, sizeof(input));
    assert(bb_native_kv_transfer(&fixture.ns, &fixture.identity, &coordinate, 16,
        input, sizeof(input), true, 100, &first) == CEMU_KV_OK);
    assert(first.ready_ns == 2000000100);
    coordinate.token = 16;
    assert(bb_native_kv_transfer(&fixture.ns, &fixture.identity, &coordinate, 16,
        input, sizeof(input), true, 100, &second) == CEMU_KV_OK);
    assert(second.ready_ns == first.ready_ns);
    assert(first.page.physical.channel != second.page.physical.channel);
    coordinate.token = 15;
    coordinate.channel = 3;
    assert(bb_native_kv_transfer(&fixture.ns, &fixture.identity, &coordinate, 0,
        output, sizeof(output), false, 100, &read) == CEMU_KV_OK);
    assert(read.start_ns == first.ready_ns && read.ready_ns == 2040000100);
    assert(read.element_offset == (15 * 128 + 3) * 2);
    assert(memcmp(input, output, sizeof(input)) == 0);
    assert(pread(fixture.descriptor, disk, sizeof(disk), first.page.backend_offset) == sizeof(disk));
    assert(memcmp(input, disk, sizeof(disk)) == 0);
    assert(bb_native_kv_snapshot(&fixture.ns, &stats, &deadline) == CEMU_KV_OK);
    assert(stats.written_pages == 2 && deadline == read.ready_ns);
    wrong = fixture.identity;
    wrong.namespace_id++;
    assert(bb_native_kv_transfer(&fixture.ns, &wrong, &coordinate, 0,
        output, sizeof(output), false, INT64_MAX, &read) == CEMU_KV_INVALID);
    assert(bb_native_kv_snapshot(&fixture.ns, &stats, &deadline) == CEMU_KV_OK);
    assert(deadline == 2040000100);
    coordinate = (CemuKvCoordinate){ .token = 32 };
    space = fixture.backend.logical_space;
    fixture.backend.logical_space = NULL;
    assert(bb_native_kv_transfer(&fixture.ns, &fixture.identity, &coordinate, 16,
        input, sizeof(input), true, 100, &second) == CEMU_KV_IO_FAILED);
    assert(second.ready_ns == 4040000100 && second.transferred_bytes == 0);
    fixture.backend.logical_space = space;
    memset(output, 0x29, sizeof(output));
    assert(bb_native_kv_transfer(&fixture.ns, &fixture.identity, &coordinate, 0,
        output, sizeof(output), false, 100, &read) == CEMU_KV_IO_FAILED);
    for (size_t index = 0; index < sizeof(output); index++) {
        assert(output[index] == 0x29);
    }
    assert(bb_native_kv_snapshot(&fixture.ns, &stats, &deadline) == CEMU_KV_OK);
    assert(stats.failed_pages == 1 && stats.written_pages == 2 && deadline == second.ready_ns);
    bb_native_io_destroy(&fixture.ns);
    assert(bb_native_kv_transfer(&fixture.ns, &fixture.identity, &coordinate, 0,
        output, sizeof(output), false, 100, &read) == CEMU_KV_INVALID);
    assert(bb_native_kv_register(&fixture.ns, &fixture.identity,
                                 &fixture.geometry) == CEMU_KV_INVALID);
    assert(bb_native_kv_snapshot(&fixture.ns, &stats, &deadline) == CEMU_KV_INVALID);
    fixture_close(&fixture);
}

typedef struct Worker {
    Fixture *fixture;
    uint32_t head;
    bool duplicate;
    CemuKvStatus status;
} Worker;

static void *transfer_worker(void *opaque)
{
    Worker *worker = opaque;
    Fixture *fixture = worker->fixture;
    uint8_t input[4096], output[4096];
    CemuKvIoResult result;

    memset(input, worker->head + 1, sizeof(input));
    for (uint32_t group = 0; group < 4; group++) {
        CemuKvCoordinate coordinate = { .head = worker->duplicate ? 0 : worker->head,
            .token = group * 16 };
        worker->status = bb_native_kv_transfer(&fixture->ns, &fixture->identity,
            &coordinate, 16, input, sizeof(input), true, 100, &result);
        if (worker->duplicate) {
            return NULL;
        }
        assert(worker->status == CEMU_KV_OK);
        assert(bb_native_kv_transfer(&fixture->ns, &fixture->identity,
            &coordinate, 0, output, sizeof(output), false, 100, &result) == CEMU_KV_OK);
        assert(result.ready_ns > 100 && memcmp(input, output, sizeof(input)) == 0);
    }
    return NULL;
}

static void test_batches(void)
{
    Fixture fixture;
    CemuKvDevicePage pages[4] = {0};
    CemuKvDeviceBatchResult batch;
    CemuKvIoResult read;
    CemuKvMapStats stats;
    uint64_t deadline;
    uint8_t input[4][4096], output[4096];

    fixture_open(&fixture);
    for (uint32_t index = 0; index < 4; index++) {
        memset(input[index], index + 1, sizeof(input[index]));
        pages[index] = (CemuKvDevicePage){ .coordinate = { .token = index * 16 },
            .valid_tokens = 16, .buffer = input[index], .bytes = sizeof(input[index]) };
    }
    assert(bb_native_kv_batch(&fixture.ns, &fixture.identity, pages, 0,
        true, 100, &batch) == CEMU_KV_INVALID);
    assert(bb_native_kv_batch(&fixture.ns, &fixture.identity, pages,
        CEMU_NATIVE_INTERNAL_MAX_PAGES + 1, true, 100, &batch) == CEMU_KV_INVALID);
    pages[2].coordinate = pages[0].coordinate;
    assert(bb_native_kv_batch(&fixture.ns, &fixture.identity, pages, 4,
        true, 100, &batch) == CEMU_KV_EXISTS);
    assert(batch.completed_pages == 2 && batch.transferred_bytes == 8192);
    assert(batch.ready_ns == 2000000100);
    assert(pages[0].result.ready_ns == pages[1].result.ready_ns);
    assert(bb_native_kv_transfer(&fixture.ns, &fixture.identity, &pages[3].coordinate,
        0, output, sizeof(output), false, 100, &read) == CEMU_KV_NOT_FOUND);
    for (uint32_t index = 0; index < 2; index++) {
        pages[index].valid_tokens = 0;
        pages[index].buffer = input[index + 2];
    }
    assert(bb_native_kv_batch(&fixture.ns, &fixture.identity, pages, 2,
        false, 100, &batch) == CEMU_KV_OK);
    assert(batch.completed_pages == 2 && batch.ready_ns == 2040000100);
    assert(memcmp(input[0], input[2], 4096) == 0);
    assert(memcmp(input[1], input[3], 4096) == 0);
    assert(bb_native_kv_snapshot(&fixture.ns, &stats, &deadline) == CEMU_KV_OK);
    assert(stats.written_pages == 2 && stats.reserved_pages == 2);
    fixture_close(&fixture);
}

static void test_read_observation_batches(void)
{
    Fixture fixture;
    CemuKvDevicePage pages[2] = {0};
    CemuKvDeviceBatchResult batch;
    FemuReadObservation observation = {0};
    uint8_t data[2][4096] = {{0}};
    uint64_t coverage[4] = {0};

    assert(femu_observe_unique_bytes(coverage, 3, 130) == 130);
    assert(femu_observe_unique_bytes(coverage, 64, 128) == 59);
    assert(femu_observe_unique_bytes(coverage, 0, 256) == 67);
    assert(femu_observe_unique_bytes(coverage, 0, 256) == 0);
    fixture_open(&fixture);
    for (uint32_t index = 0; index < 2; index++) {
        pages[index] = (CemuKvDevicePage){ .coordinate = { .token = index * 16 },
            .valid_tokens = 16, .buffer = data[index], .bytes = 4096 };
    }
    assert(bb_native_kv_batch(&fixture.ns, &fixture.identity, pages, 2,
                             true, 100, &batch) == CEMU_KV_OK);
    for (uint32_t index = 0; index < 2; index++) {
        pages[index].valid_tokens = 0;
    }
    for (uint32_t iteration = 0; iteration < 2; iteration++) {
        assert(bb_native_kv_batch_observed(&fixture.ns, &fixture.identity, pages, 2,
            false, 100, &batch, &observation) == CEMU_KV_OK);
    }
    assert(observation.pages == 4 && observation.service_ns == 160000000);
    assert(observation.channels == 2 && observation.luns_per_channel == 2);
    assert(observation.lun_pages[0] == 2 && observation.lun_pages[2] == 2);
    assert(observation.lun_pages[1] == 0 && observation.lun_pages[3] == 0);
    assert(observation.critical_initial_wait_ns == 2000000000);
    assert(observation.critical_queue_ns == 2040000000);
    assert(observation.critical_service_ns == 40000000);
    assert(observation.queue_ns == 8080000000ULL);
    assert(batch.ready_ns - 100 == observation.critical_queue_ns + observation.critical_service_ns);
    g_free(observation.lun_pages);
    g_free(observation.lun_initial_wait_ns);
    fixture_close(&fixture);
}

static void test_concurrent_callers(bool duplicate)
{
    Fixture fixture;
    GThread *threads[16];
    Worker workers[16];
    CemuKvMapStats stats;
    uint64_t deadline;
    unsigned int successes = 0;

    fixture_open(&fixture);
    for (uint32_t head = 0; head < 16; head++) {
        workers[head] = (Worker){ .fixture = &fixture, .head = head, .duplicate = duplicate };
        threads[head] = g_thread_new("native-io-test", transfer_worker, &workers[head]);
    }
    for (uint32_t head = 0; head < 16; head++) {
        g_thread_join(threads[head]);
        successes += workers[head].status == CEMU_KV_OK;
        assert(workers[head].status == CEMU_KV_OK || workers[head].status == CEMU_KV_EXISTS);
    }
    assert(successes == (duplicate ? 1 : 16));
    assert(bb_native_kv_snapshot(&fixture.ns, &stats, &deadline) == CEMU_KV_OK);
    assert(stats.written_pages == (duplicate ? 1 : 64));
    assert(stats.failed_pages == 0 && deadline > 100);
    fixture_close(&fixture);
}

static uint16_t token_element(uint32_t stream, uint32_t token, uint32_t channel)
{
    return (uint16_t)(stream * 79 + token * 131 + channel);
}

static CemuKvStatus append_token(NativeKvBuffer *buffer, uint32_t stream, uint32_t token)
{
    uint16_t key[128], value[128];
    CemuKvCoordinate coordinate = { .batch = stream, .token = token };

    for (uint32_t channel = 0; channel < 128; channel++) {
        key[channel] = token_element(stream, token, channel);
        value[channel] = key[channel] ^ 0x1357;
    }
    return native_kv_buffer_append(buffer, &coordinate, 1, key, value, sizeof(key));
}

static void read_token(NativeKvBuffer *buffer, uint32_t stream, uint32_t token)
{
    CemuKvCoordinate coordinate = { .batch = stream, .token = token };
    CemuKvIoResult result;
    uint16_t output[128];

    for (uint32_t kind = 0; kind < 3; kind += 2) {
        coordinate.kind = kind;
        coordinate.channel = 0;
        assert(native_kv_buffer_read(buffer, &coordinate, output, sizeof(output),
                                     qatomic_read(&virtual_now), &result) == CEMU_KV_OK);
        for (uint32_t channel = 0; channel < 128; channel++) {
            uint16_t expected = token_element(stream, token, channel);
            assert(output[channel] == (kind == 2 ? (expected ^ 0x1357) : expected));
        }
    }
    coordinate.kind = 1;
    for (uint32_t channel = 0; channel < 128; channel++) {
        coordinate.channel = channel;
        assert(native_kv_buffer_read(buffer, &coordinate, output, 2,
                                     qatomic_read(&virtual_now), &result) == CEMU_KV_OK);
        assert(output[0] == token_element(stream, token, channel));
    }
}

static NativeKvBuffer *create_buffer(Fixture *fixture)
{
    NativeKvBuffer *buffer = NULL;
    CemuKvIdentity identity = fixture->identity;
    CemuKvGeometry geometry = fixture->geometry;

    identity.task++;
    geometry.batches = 2;
    geometry.heads = 1;
    geometry.max_tokens = 600;
    assert(native_kv_buffer_create(&fixture->ns, &identity, &geometry, 1, 32,
                                   &buffer) == CEMU_KV_LIMIT);
    assert(!buffer);
    /* Old tail-only budget cannot safely hold all partial block domains. */
    assert(native_kv_buffer_create(&fixture->ns, &identity, &geometry, MiB, 32,
                                   &buffer) == CEMU_KV_LIMIT);
    assert(!buffer);
    assert(native_kv_buffer_create(&fixture->ns, &identity, &geometry, 2 * MiB, 32,
                                   &buffer) == CEMU_KV_OK);
    return buffer;
}

static NativeKvBufferStats buffer_stats(NativeKvBuffer *buffer, uint32_t stream)
{
    NativeKvBufferStats stats;
    CemuKvCoordinate coordinate = { .batch = stream };

    assert(native_kv_buffer_stats(buffer, &coordinate, &stats) == CEMU_KV_OK);
    return stats;
}

static void advance_virtual_clock(void)
{
    qatomic_set(&virtual_now, qatomic_read(&virtual_now) + 10000000000LL);
    g_usleep(1000);
}

static NativeKvGatherResult verify_gather(NativeKvBuffer *buffer,
    const NativeKvGatherRange *ranges, uint32_t count)
{
    NativeKvGatherResult result;
    uint16_t output[65536];
    uint64_t bytes = 0, before = buffer_stats(buffer, 0).read_bytes;
    uint32_t position = 0;
    int64_t now = qatomic_read(&virtual_now);

    for (uint32_t range = 0; range < count; range++) {
        bytes += (uint64_t)ranges[range].tokens *
            (ranges[range].coordinate.kind == 1 ? 2 : 256);
    }
    assert(bytes <= sizeof(output));
    assert(native_kv_buffer_gather(buffer, ranges, count, output, bytes, now, &result) == CEMU_KV_OK);
    assert(qatomic_read(&virtual_now) == now && result.ready_ns >= now);
    assert(result.output_bytes == bytes && result.transferred_bytes == result.flash_pages * 4096ULL);
    assert(buffer_stats(buffer, 0).read_bytes - before == result.transferred_bytes);
    for (uint32_t range = 0; range < count; range++) {
        CemuKvCoordinate coordinate = ranges[range].coordinate;
        uint32_t channels = coordinate.kind == 1 ? 1 : 128;

        for (uint32_t token = 0; token < ranges[range].tokens; token++) {
            for (uint32_t channel = 0; channel < channels; channel++) {
                uint16_t expected = token_element(coordinate.batch, coordinate.token + token,
                                                  coordinate.channel + channel);
                assert(output[position++] == (coordinate.kind == 2 ? (expected ^ 0x1357) : expected));
            }
        }
    }
    return result;
}

static void test_gather_rejections(Fixture *fixture, NativeKvBuffer *buffer)
{
    NativeKvGatherRange ranges[33] = {0};
    NativeKvGatherResult result;
    CemuKvMapStats map;
    uint8_t output[136192], sentinel[136192];
    uint64_t before, after, reads;
    CemuKvIdentity identity = fixture->identity;
    CemuKvIoResult page;
    uint8_t scratch[4096];
    uint64_t backend_size = fixture->backend.size;

    memset(output, 0x5c, sizeof(output));
    memcpy(sentinel, output, sizeof(output));
    reads = buffer_stats(buffer, 0).read_bytes;
    assert(bb_native_kv_snapshot(&fixture->ns, &map, &before) == CEMU_KV_OK);
    for (uint32_t index = 0; index < 33; index++) {
        ranges[index] = (NativeKvGatherRange){ .coordinate.token = index * 16, .tokens = 1 };
    }
    assert(native_kv_buffer_gather(buffer, ranges, 33, output, 33 * 256, 100, &result) == CEMU_KV_LIMIT);
    assert(result.output_bytes == 0 && result.transferred_bytes == 0);
    ranges[0].tokens = 531;
    assert(native_kv_buffer_gather(buffer, ranges, 1, output, 531 * 256, 100, &result) == CEMU_KV_LIMIT);
    ranges[0].tokens = 1;
    ranges[1].coordinate.token = 531;
    assert(native_kv_buffer_gather(buffer, ranges, 2, output, 512, 100, &result) == CEMU_KV_NOT_FOUND);
    ranges[1].coordinate.token = UINT64_MAX;
    assert(native_kv_buffer_gather(buffer, ranges, 2, output, 512, 100, &result) == CEMU_KV_INVALID);
    ranges[1].coordinate.token = 16;
    ranges[0].coordinate.channel = 1;
    assert(native_kv_buffer_gather(buffer, ranges, 2, output, 512, 100, &result) == CEMU_KV_INVALID);
    ranges[0].coordinate.channel = 0;
    assert(native_kv_buffer_gather(buffer, ranges, 2, output, 511, 100, &result) == CEMU_KV_INVALID);
    assert(native_kv_buffer_gather(buffer, ranges, 0, output, 0, 100, &result) == CEMU_KV_INVALID);
    assert(native_kv_buffer_gather(buffer, ranges, 257, output, 0, 100, &result) == CEMU_KV_INVALID);
    ranges[0].tokens = 0;
    assert(native_kv_buffer_gather(buffer, ranges, 1, output, 0, 100, &result) == CEMU_KV_INVALID);
    ranges[0].tokens = 1;
    assert(buffer_stats(buffer, 0).read_bytes == reads);
    assert(bb_native_kv_snapshot(&fixture->ns, &map, &after) == CEMU_KV_OK && after == before);
    assert(memcmp(output, sentinel, sizeof(output)) == 0);

    identity.task++;
    assert(bb_native_kv_transfer(&fixture->ns, &identity, &ranges[0].coordinate, 0,
        scratch, sizeof(scratch), false, 100, &page) == CEMU_KV_OK);
    fixture->backend.size = page.page.backend_offset + 4096;
    assert(native_kv_buffer_gather(buffer, ranges, 2, output, 512, 100, &result) == CEMU_KV_IO_FAILED);
    assert(result.transferred_bytes == 4096 && result.flash_pages == 1 && result.output_bytes == 0);
    assert(buffer_stats(buffer, 0).read_bytes == reads + 4096);
    assert(memcmp(output, sentinel, sizeof(output)) == 0);
    fixture->backend.size = backend_size;
    verify_gather(buffer, ranges, 2);
}

static void test_native_buffer(void)
{
    Fixture fixture;
    NativeKvBuffer *buffer;
    NativeKvBufferStats stats;
    uint64_t memory_bytes;

    fixture_open(&fixture);
    buffer = create_buffer(&fixture);
    memory_bytes = buffer_stats(buffer, 0).memory_bytes;
    for (uint32_t token = 0; token < 15; token++) {
        assert(append_token(buffer, 0, token) == CEMU_KV_OK);
    }
    read_token(buffer, 0, 14);
    {
        NativeKvGatherRange ranges[] = {
            { .coordinate = { .token = 14, .kind = 2 }, .tokens = 1 },
            { .coordinate = { .token = 0, .kind = 1, .channel = 7 }, .tokens = 15 },
            { .coordinate = { .token = 0 }, .tokens = 15 },
            { .coordinate = { .token = 14, .kind = 2 }, .tokens = 1 },
        };
        NativeKvGatherResult result = verify_gather(buffer, ranges, G_N_ELEMENTS(ranges));

        assert(result.flash_pages == 0 && result.tail_bytes == result.output_bytes);
    }
    stats = buffer_stats(buffer, 0);
    assert(stats.read_bytes == 0 && stats.write_bytes == 0);
    for (uint32_t token = 15; token < 32; token++) {
        assert(append_token(buffer, 0, token) == CEMU_KV_OK);
    }
    for (uint32_t attempt = 0; attempt < 2000; attempt++) {
        stats = buffer_stats(buffer, 0);
        if (stats.packed[0] == 32 && stats.packed[2] == 32) {
            break;
        }
        g_usleep(1000);
    }
    /* Four full token pages plus sixteen partial channel pages. Neither NAND
     * domain has a full block; partial pages are immediately readable. */
    assert(stats.packed[0] == 32 && stats.packed[2] == 32);
    assert(!stats.pending_ready_ns && !stats.write_bytes && stats.staged_pages == 20);
    assert(stats.committed[0] == 0 && stats.committed[2] == 0);
    assert(append_token(buffer, 0, 32) == CEMU_KV_OK);
    assert(buffer_stats(buffer, 0).accepted == 33);
    read_token(buffer, 0, 0);
    assert(buffer_stats(buffer, 0).read_bytes == 0);
    for (uint32_t stream = 0; stream < 2; stream++) {
        for (uint32_t token = stream ? 0 : 33; token < 531; token++) {
            CemuKvStatus status;
            uint32_t attempts = 0;

            while ((status = append_token(buffer, stream, token)) == CEMU_KV_PENDING) {
                assert(++attempts < 2000);
                advance_virtual_clock();
            }
            assert(status == CEMU_KV_OK);
            if (token == 530) {
                /* Direct pages can absorb the stream without ring backpressure;
                 * explicitly advance target time before expecting NAND hits. */
                for (uint32_t attempt = 0; ; attempt++) {
                    stats = buffer_stats(buffer, stream);
                    assert(attempt < 2000);
                    if (!stats.pending_ready_ns && !stats.ready_blocks) {
                        break;
                    }
                    advance_virtual_clock();
                }
            }
            if (token == 252 || token == 255 || token == 258 || token == 530) {
                read_token(buffer, stream, token);
                read_token(buffer, stream, 0);
                NativeKvGatherRange ranges[] = {
                    { .coordinate = { .batch = stream, .token = token, .kind = 2 }, .tokens = 1 },
                    { .coordinate = { .batch = stream, .token = 0 }, .tokens = 17 },
                    { .coordinate = { .batch = stream, .token = token - 16, .kind = 1, .channel = 127 },
                      .tokens = 17 },
                };
                NativeKvGatherResult result = verify_gather(buffer, ranges, G_N_ELEMENTS(ranges));

                assert(result.tail_bytes);
                if (token == 530) {
                    assert(result.flash_pages);
                }
            }
        }
    }
    for (uint32_t attempts = 0; ; attempts++) {
        CemuKvStatus status = native_kv_buffer_drain(buffer);

        assert(attempts < 2000);
        if (status == CEMU_KV_OK) {
            break;
        }
        assert(status == CEMU_KV_PENDING);
        advance_virtual_clock();
    }
    for (uint32_t stream = 0; stream < 2; stream++) {
        stats = buffer_stats(buffer, stream);
        assert(stats.sealed && !stats.failed && stats.accepted == 531);
        assert(stats.committed[0] == 531 && stats.committed[1] == 531 && stats.committed[2] == 531);
        assert(stats.memory_bytes == memory_bytes && stats.memory_bytes <= 2 * MiB);
        assert(stats.write_bytes == 950272);
        for (uint32_t token = 0; token < 531; token += 15) {
            read_token(buffer, stream, token);
        }
        read_token(buffer, stream, 530);
    }
    assert(append_token(buffer, 0, 531) == CEMU_KV_INVALID);
    {
        NativeKvGatherRange ranges[] = {
            { .coordinate = { .batch = 1, .token = 528, .kind = 2 }, .tokens = 3 },
            { .coordinate = { .token = 250, .kind = 1, .channel = 7 }, .tokens = 20 },
            { .coordinate = { .token = 14 }, .tokens = 4 },
            { .coordinate = { .token = 250, .kind = 1, .channel = 0 }, .tokens = 20 },
            { .coordinate = { .token = 14 }, .tokens = 4 },
            { .coordinate = { .token = 250, .kind = 1, .channel = 8 }, .tokens = 20 },
            { .coordinate = { .batch = 1, .token = 528, .kind = 2 }, .tokens = 3 },
        };
        NativeKvGatherResult result = verify_gather(buffer, ranges, G_N_ELEMENTS(ranges));

        assert(result.flash_pages == 7 && result.tail_bytes == 0);
    }
    test_gather_rejections(&fixture, buffer);
    native_kv_buffer_free(buffer);
    fixture_close(&fixture);
}

/* One page group per head fills one physical block across sixteen heads.
 * batch_pages=1 proves the API chunk limit is independent of block/group size. */
static void test_block_gate_and_cross_head_mapping(void)
{
    Fixture fixture;
    NativeKvBuffer *buffer = NULL;
    CemuKvIdentity identity;
    CemuKvCoordinate coordinate = {0};
    NativeKvBufferStats stats;
    uint16_t payload[16 * 128], output[128];
    uint8_t page_data[4096];

    fixture_open(&fixture);
    identity = fixture.identity;
    identity.task++;
    assert(native_kv_buffer_create(&fixture.ns, &identity, &fixture.geometry,
                                   4 * MiB, 1, &buffer) == CEMU_KV_OK);
    for (uint32_t head = 0; head < 15; head++) {
        coordinate.head = head;
        for (uint32_t i = 0; i < G_N_ELEMENTS(payload); i++) {
            payload[i] = head * 1000 + i;
        }
        assert(native_kv_buffer_append(buffer, &coordinate, 16, payload, payload,
                                      sizeof(payload)) == CEMU_KV_OK);
    }
    coordinate.head = 14;
    for (uint32_t attempt = 0; attempt < 2000; attempt++) {
        assert(native_kv_buffer_stats(buffer, &coordinate, &stats) == CEMU_KV_OK);
        if (stats.staged_pages == 270) {
            break;
        }
        g_usleep(1000);
    }
    assert(stats.staged_pages == 270 && !stats.write_bytes && !stats.write_blocks);
    coordinate.head = 15;
    for (uint32_t i = 0; i < G_N_ELEMENTS(payload); i++) {
        payload[i] = 15000 + i;
    }
    assert(native_kv_buffer_append(buffer, &coordinate, 16, payload, payload,
                                  sizeof(payload)) == CEMU_KV_OK);
    for (uint32_t attempt = 0; attempt < 2000; attempt++) {
        assert(native_kv_buffer_stats(buffer, &coordinate, &stats) == CEMU_KV_OK);
        if (stats.write_batches == 32) {
            break;
        }
        g_usleep(1000);
    }
    assert(stats.write_batches == 32 && stats.pending_ready_ns > qatomic_read(&virtual_now));
    assert(stats.inflight_blocks == 2 && stats.peak_inflight_blocks == 2);
    assert(!stats.committed[0] && !stats.write_blocks);
    /* Backend has bytes, but target deadline is pending: readers still use DRAM. */
    for (uint32_t head = 0; head < 16; head++) {
        CemuKvIoResult result;
        coordinate.head = head;
        assert(native_kv_buffer_read(buffer, &coordinate, output, sizeof(output),
                                    qatomic_read(&virtual_now), &result) == CEMU_KV_OK);
        assert(!result.transferred_bytes);
        for (uint32_t channel = 0; channel < 128; channel++) {
            assert(output[channel] == head * 1000 + channel);
        }
    }
    for (uint32_t attempt = 0; attempt < 2000; attempt++) {
        advance_virtual_clock();
        assert(native_kv_buffer_stats(buffer, &coordinate, &stats) == CEMU_KV_OK);
        if (stats.write_blocks == 2) {
            break;
        }
    }
    assert(stats.write_blocks == 2 && !stats.partial_blocks && stats.staged_pages == 256);
    assert(stats.write_bytes == 32 * 4096 && stats.write_batches == 32);
    for (uint32_t kind = 0; kind <= 2; kind += 2) {
        CemuKvPhysicalPage first = {0};
        uint32_t seen = 0;
        coordinate.kind = kind;
        for (uint32_t head = 0; head < 16; head++) {
            CemuKvIoResult result;
            CemuKvPhysicalPage physical;
            coordinate.head = head;
            assert(bb_native_kv_transfer(&fixture.ns, &identity, &coordinate, 0,
                page_data, sizeof(page_data), false, qatomic_read(&virtual_now), &result) == CEMU_KV_OK);
            physical = result.page.physical;
            if (!head) {
                first = physical;
            }
            assert(physical.channel == first.channel && physical.lun == first.lun &&
                   physical.plane == first.plane && physical.block == first.block);
            assert(physical.page < 16 && !(seen & (1U << physical.page)));
            seen |= 1U << physical.page;
            for (uint32_t i = 0; i < 16 * 128; i++) {
                assert(((uint16_t *)page_data)[i] == head * 1000 + i);
            }
        }
        assert(seen == 0xffff);
    }
    native_kv_buffer_free(buffer);
    fixture_close(&fixture);
}

/* A channel span contains 32 independent page groups (two physical blocks).
 * After the first block completes, those channels must read NAND even though
 * the whole-head channel committed prefix is still zero. */
static void test_channel_pages_across_blocks(void)
{
    Fixture fixture;
    NativeKvBuffer *buffer = NULL;
    CemuKvIdentity identity;
    CemuKvGeometry geometry;
    bool mixed = false;

    fixture_open(&fixture);
    fixture.luns[0][0].next_lun_avail_time = 1000000000000LL;
    identity = fixture.identity;
    identity.task++;
    geometry = fixture.geometry;
    geometry.heads = 1;
    geometry.max_tokens = 512;
    geometry.channels_per_page = 4;
    assert(native_kv_buffer_create(&fixture.ns, &identity, &geometry,
                                   4 * MiB, 1, &buffer) == CEMU_KV_OK);
    for (uint32_t token = 0; token < 512; token++) {
        CemuKvStatus status;
        uint32_t attempts = 0;
        while ((status = append_token(buffer, 0, token)) == CEMU_KV_PENDING) {
            assert(++attempts < 2000);
            advance_virtual_clock();
        }
        assert(status == CEMU_KV_OK);
    }
    for (uint32_t attempt = 0; attempt < 2000; attempt++) {
        CemuKvCoordinate coordinate = { .kind = 1, .token = 300 };
        CemuKvIoResult first, last;
        uint16_t value;

        assert(native_kv_buffer_read(buffer, &coordinate, &value, 2,
            qatomic_read(&virtual_now), &first) == CEMU_KV_OK);
        assert(value == token_element(0, 300, 0));
        coordinate.channel = 127;
        assert(native_kv_buffer_read(buffer, &coordinate, &value, 2,
            qatomic_read(&virtual_now), &last) == CEMU_KV_OK);
        assert(value == token_element(0, 300, 127));
        if (first.transferred_bytes && !last.transferred_bytes) {
            NativeKvGatherRange ranges[] = {
                { .coordinate = { .kind = 1, .token = 250 }, .tokens = 20 },
                { .coordinate = { .kind = 1, .token = 250, .channel = 127 }, .tokens = 20 },
            };
            /* batch_pages=1: one NAND page plus one staged page, in order. */
            NativeKvGatherResult result = verify_gather(buffer, ranges, G_N_ELEMENTS(ranges));
            assert(result.flash_pages == 1 && result.tail_bytes == 40);
            assert(buffer_stats(buffer, 0).committed[1] == 0);
            mixed = true;
            break;
        }
        advance_virtual_clock();
    }
    assert(mixed);
    for (uint32_t attempt = 0; ; attempt++) {
        assert(attempt < 2000);
        if (native_kv_buffer_drain(buffer) == CEMU_KV_OK) {
            break;
        }
        advance_virtual_clock();
    }
    assert(buffer_stats(buffer, 0).committed[1] == 512);
    assert(!buffer_stats(buffer, 0).partial_blocks);
    read_token(buffer, 0, 300);
    native_kv_buffer_free(buffer);
    fixture_close(&fixture);
}

static void fill_heads(uint16_t *payload, uint32_t first, uint32_t count)
{
    for (uint32_t head = 0; head < 16; head++) {
        for (uint32_t token = 0; token < count; token++) {
            for (uint32_t channel = 0; channel < 128; channel++) {
                payload[((size_t)head * count + token) * 128 + channel] =
                    token_element(head, first + token, channel);
            }
        }
    }
}

static void test_parallel_writeback_deadlines(void)
{
    Fixture fixture;
    NativeKvBuffer *buffer = NULL;
    CemuKvIdentity identity;
    CemuKvCoordinate coordinate = {0};
    NativeKvBufferStats stats;
    CemuKvIoResult read;
    uint16_t payload[16 * 32 * 128], output[128];
    uint64_t fast_deadline = 100 + 16ULL * 2000000000;
    uint64_t slow_deadline = 1000000000000ULL + 16ULL * 2000000000;

    fixture_open_sized(&fixture, 16, 4096);
    identity = fixture.identity;
    identity.task++;
    fixture.luns[0][0].next_lun_avail_time = 1000000000000ULL;
    fixture.luns[0][1].next_lun_avail_time = 1000000000000ULL;
    assert(native_kv_buffer_create(&fixture.ns, &identity, &fixture.geometry,
                                   8 * MiB, 1, &buffer) == CEMU_KV_OK);
    fill_heads(payload, 0, 32);
    assert(native_kv_buffer_append_batch(buffer, &coordinate, 16, 32,
        payload, payload, sizeof(payload)) == CEMU_KV_OK);
    memset(payload, 0, sizeof(payload));
    for (uint32_t attempt = 0; ; attempt++) {
        assert(attempt < 2000);
        stats = buffer_stats(buffer, 0);
        if (stats.write_batches == 64 && stats.inflight_blocks == 4) {
            break;
        }
        g_usleep(1000);
    }
    assert(qatomic_read(&virtual_now) == 100 && !stats.write_blocks);
    assert(stats.peak_inflight_blocks == 4 && stats.pending_ready_ns == slow_deadline);
    assert(!stats.committed[0] && !stats.committed[2]);
    coordinate.head = 3;
    coordinate.token = 16;
    assert(native_kv_buffer_read(buffer, &coordinate, output, sizeof(output),
                                 100, &read) == CEMU_KV_OK && !read.transferred_bytes);
    for (uint32_t channel = 0; channel < 128; channel++) {
        assert(output[channel] == token_element(3, 16, channel));
    }
    qatomic_set(&virtual_now, fast_deadline - 1);
    g_usleep(10000);
    assert(!buffer_stats(buffer, 0).write_blocks);
    qatomic_set(&virtual_now, fast_deadline);
    for (uint32_t attempt = 0; ; attempt++) {
        assert(attempt < 2000);
        stats = buffer_stats(buffer, 0);
        if (stats.write_blocks == 2) {
            break;
        }
        g_usleep(1000);
    }
    assert(stats.inflight_blocks == 2 && !stats.committed[0]);
    assert(native_kv_buffer_read(buffer, &coordinate, output, sizeof(output),
        fast_deadline, &read) == CEMU_KV_OK && read.transferred_bytes == 4096);
    coordinate.token = 0;
    assert(native_kv_buffer_read(buffer, &coordinate, output, sizeof(output),
        fast_deadline, &read) == CEMU_KV_OK && !read.transferred_bytes);
    fill_heads(payload, 32, 16);
    coordinate = (CemuKvCoordinate){ .token = 32 };
    assert(native_kv_buffer_append_batch(buffer, &coordinate, 16, 16,
        payload, payload, 16 * 16 * 256) == CEMU_KV_OK);
    for (uint32_t attempt = 0; ; attempt++) {
        assert(attempt < 2000);
        stats = buffer_stats(buffer, 0);
        if (stats.write_batches == 96 && stats.inflight_blocks == 4) {
            break;
        }
        g_usleep(1000);
    }
    assert(stats.pending_ready_ns == slow_deadline + 16ULL * 2000000000);
    qatomic_set(&virtual_now, slow_deadline);
    for (uint32_t attempt = 0; ; attempt++) {
        assert(attempt < 2000);
        stats = buffer_stats(buffer, 0);
        if (stats.write_blocks == 4) {
            break;
        }
        g_usleep(1000);
    }
    assert(stats.committed[0] == 32 && stats.inflight_blocks == 2);
    assert(native_kv_buffer_read(buffer, &coordinate, output, sizeof(output),
        slow_deadline, &read) == CEMU_KV_OK && !read.transferred_bytes);
    qatomic_set(&virtual_now, slow_deadline + 16ULL * 2000000000);
    for (uint32_t attempt = 0; ; attempt++) {
        assert(attempt < 2000);
        stats = buffer_stats(buffer, 0);
        if (stats.write_blocks == 6) {
            break;
        }
        g_usleep(1000);
    }
    assert(stats.committed[0] == 48 && !stats.inflight_blocks && !stats.pending_ready_ns);
    for (uint32_t attempt = 0; native_kv_buffer_drain(buffer) != CEMU_KV_OK; attempt++) {
        assert(attempt < 2000);
        advance_virtual_clock();
    }
    assert(buffer_stats(buffer, 0).sealed);
    native_kv_buffer_free(buffer);
    fixture_close(&fixture);
}

static void test_direct_pages_atomic_capacity(void)
{
    Fixture fixture;
    CemuKvIdentity identity;
    CemuKvGeometry geometry;
    NativeKvBuffer *buffer = NULL;
    NativeKvBufferStats before, after;
    uint16_t payload[16 * 32 * 128], output[128];
    CemuKvCoordinate coordinate = {0};
    uint32_t accepted = 0;
    CemuKvStatus status;

    fixture_open_sized(&fixture, 16, 4096);
    identity = fixture.identity;
    identity.task++;
    geometry = fixture.geometry;
    geometry.max_tokens = 4096;
    assert(native_kv_buffer_create(&fixture.ns, &identity, &geometry,
                                   8 * MiB, 1, &buffer) == CEMU_KV_OK);
    /* Start at an unaligned token offset: the next APPEND touches three token
     * pages per head, while all channel pages are still partial. */
    fill_heads(payload, 0, 15);
    assert(native_kv_buffer_append_batch(buffer, &coordinate, 16, 15, payload,
                                        payload, 16 * 15 * 256) == CEMU_KV_OK);
    accepted = 15;
    while (accepted < 1024) {
        coordinate.token = accepted;
        fill_heads(payload, accepted, 32);
        status = native_kv_buffer_append_batch(buffer, &coordinate, 16, 32,
                                               payload, payload, sizeof(payload));
        if (status == CEMU_KV_PENDING) {
            break;
        }
        assert(status == CEMU_KV_OK);
        accepted += 32;
    }
    assert(accepted < 1024 && status == CEMU_KV_PENDING);
    before = buffer_stats(buffer, 0);
    /* Keep NAND time frozen. Rejected payload must not allocate pages, alter
     * existing partial-page contents, or advance ANY head's accepted prefix. */
    memset(payload, 0x5a, sizeof(payload));
    assert(native_kv_buffer_append_batch(buffer, &coordinate, 16, 32,
        payload, payload, sizeof(payload)) == CEMU_KV_PENDING);
    after = buffer_stats(buffer, 0);
    assert(before.staged_pages == after.staged_pages);
    for (uint32_t head = 0; head < 16; head++) {
        CemuKvCoordinate probe = { .head = head, .token = accepted - 1 };
        CemuKvIoResult result;
        assert(native_kv_buffer_stats(buffer, &probe, &after) == CEMU_KV_OK);
        assert(after.accepted == accepted && after.packed[1] == accepted);
        for (uint32_t kind = 0; kind < 3; kind++) {
            probe.kind = kind;
            probe.channel = kind == 1 ? 127 : 0;
            assert(native_kv_buffer_read(buffer, &probe, output, kind == 1 ? 2 : 256,
                qatomic_read(&virtual_now), &result) == CEMU_KV_OK);
            for (uint32_t c = 0; c < (kind == 1 ? 1 : 128); c++) {
                assert(output[c] == token_element(head, accepted - 1, probe.channel + c));
            }
            probe.token = accepted;
            assert(native_kv_buffer_read(buffer, &probe, output, kind == 1 ? 2 : 256,
                qatomic_read(&virtual_now), &result) == CEMU_KV_NOT_FOUND);
            probe.token = accepted - 1;
        }
    }
    /* Once a block completes, exactly the same command can make progress. */
    fill_heads(payload, accepted, 32);
    for (uint32_t attempt = 0; ; attempt++) {
        assert(attempt < 2000);
        advance_virtual_clock();
        status = native_kv_buffer_append_batch(buffer, &coordinate, 16, 32,
                                               payload, payload, sizeof(payload));
        if (status != CEMU_KV_PENDING) {
            assert(status == CEMU_KV_OK);
            break;
        }
    }
    /* Caller reuses wire data immediately; no borrowed input is retained. */
    memset(payload, 0, sizeof(payload));
    for (uint32_t attempt = 0; ; attempt++) {
        assert(attempt < 2000);
        status = native_kv_buffer_drain(buffer);
        if (status == CEMU_KV_OK) {
            break;
        }
        assert(status == CEMU_KV_PENDING);
        advance_virtual_clock();
    }
    for (uint32_t head = 0; head < 16; head++) {
        for (uint32_t token = 0; token < accepted + 32; token++) {
            CemuKvCoordinate probe = { .head = head, .token = token };
            CemuKvIoResult result;
            for (uint32_t kind = 0; kind < 3; kind++) {
                probe.kind = kind;
                probe.channel = kind == 1 ? 127 : 0;
                assert(native_kv_buffer_read(buffer, &probe, output, kind == 1 ? 2 : 256,
                    qatomic_read(&virtual_now), &result) == CEMU_KV_OK);
                for (uint32_t c = 0; c < (kind == 1 ? 1 : 128); c++) {
                    assert(output[c] == token_element(head, token, probe.channel + c));
                }
            }
        }
    }
    native_kv_buffer_free(buffer);
    fixture_close(&fixture);
}

static void test_buffer_cancel_and_failure(bool fail_transfer)
{
    Fixture fixture;
    NativeKvBuffer *buffer;
    NativeKvBufferStats stats;
    void *space;

    fixture_open(&fixture);
    buffer = create_buffer(&fixture);
    space = fixture.backend.logical_space;
    if (fail_transfer) {
        fixture.backend.logical_space = NULL;
    }
    for (uint32_t token = 0; token < 16; token++) {
        assert(append_token(buffer, 0, token) == CEMU_KV_OK);
    }
    /* A short stream only programs NAND on explicit terminal drain. */
    assert(native_kv_buffer_drain(buffer) == CEMU_KV_PENDING);
    for (uint32_t attempt = 0; attempt < 2000; attempt++) {
        stats = buffer_stats(buffer, 0);
        if (stats.pending_ready_ns || stats.failed) {
            break;
        }
        g_usleep(1000);
    }
    assert(stats.pending_ready_ns || stats.failed);
    if (fail_transfer) {
        assert(stats.failed && native_kv_buffer_drain(buffer) == CEMU_KV_IO_FAILED);
    }
    native_kv_buffer_stop(buffer);
    fixture.backend.logical_space = space;
    stats = buffer_stats(buffer, 0);
    assert(stats.failed && !stats.sealed && !stats.committed[0] && !stats.committed[2]);
    assert(append_token(buffer, 0, 16) == CEMU_KV_IO_FAILED);
    {
        NativeKvGatherRange range = { .tokens = 1 };
        NativeKvGatherResult result;
        uint8_t output[256] = {0};

        assert(native_kv_buffer_gather(buffer, &range, 1, output, sizeof(output),
                                       100, &result) == CEMU_KV_IO_FAILED);
        assert(!result.output_bytes && !result.transferred_bytes);
    }
    native_kv_buffer_free(buffer);
    fixture_close(&fixture);
}

int main(void)
{
    CemuKvMapStats stats;
    uint64_t deadline;
    NvmeNamespace legacy = {0};

    assert(bb_native_kv_snapshot(NULL, &stats, &deadline) == CEMU_KV_INVALID);
    assert(bb_native_kv_snapshot(&legacy, &stats, &deadline) == CEMU_KV_INVALID);
    test_dependencies_and_error();
    test_batches();
    test_read_observation_batches();
    test_concurrent_callers(false);
    test_concurrent_callers(true);
    test_native_buffer();
    test_block_gate_and_cross_head_mapping();
    test_channel_pages_across_blocks();
    test_parallel_writeback_deadlines();
    test_direct_pages_atomic_capacity();
    test_buffer_cancel_and_failure(false);
    test_buffer_cancel_and_failure(true);
    puts("[native-device-io] PASS: shared mapping/NAND/backend, parallel target lanes, "
         "bounded batches/partial failure, 16 callers, duplicate writes, failure deadlines, closed owner; "
         "direct-layout DRAM pages/writeback, 531 tokens x 2 streams, partial drain, "
         "sparse range gather/dedup/order, tail+Flash, bounded preflight, partial read failure; "
         "page/block gate, cross-head PPA, channel pages across blocks, "
         "atomic page-pool backpressure, unaligned batch APPEND, "
         "frozen-clock cancellation; no FDM/Attention integration");
    return 0;
}
