#define _POSIX_C_SOURCE 200809L
#include <assert.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>
#include "hw/femu/bbssd/instattention_io.h"
#include "hw/femu/backend/mapped_io.h"

typedef struct Fixture {
    int descriptor;
    void *space;
    size_t bytes;
    CemuKvIo *io;
    uint64_t available[2][2];
    uint64_t schedule_start;
    CemuKvPhysicalPage scheduled_page;
    size_t transfers;
    size_t schedules;
    size_t prepare_count;
    size_t dirty_count;
    uint64_t last_offset;
    bool fail_io;
    bool fail_sync;
    bool fail_schedule;
    bool bad_deadline;
} Fixture;

static const CemuKvIdentity identity = { .controller = 1, .namespace_id = 1,
                                        .task = 100, .generation = 1 };

static int prepare(void *opaque, void *pointer, uint64_t bytes)
{
    Fixture *fixture = opaque;

    assert(pointer && bytes == 4096);
    fixture->prepare_count++;
    return fixture->fail_sync ? -1 : 0;
}

static void dirty(void *opaque, void *pointer, uint64_t bytes)
{
    Fixture *fixture = opaque;

    assert(pointer && bytes == 4096);
    fixture->dirty_count++;
}

static const CemuMappedIoOps mapped_ops = { .prepare_host = prepare, .mark_host_dirty = dirty };

static int transfer(void *opaque, void *buffer, uint64_t offset, size_t bytes, bool write)
{
    Fixture *fixture = opaque;

    fixture->transfers++;
    fixture->last_offset = offset;
    if (fixture->fail_io) {
        if (!write) {
            memset(buffer, 0x55, bytes / 2);
        }
        return -1;
    }
    return cemu_mapped_io(fixture->space, fixture->bytes, buffer, offset, bytes,
                          write, &mapped_ops, fixture);
}

static bool schedule(void *opaque, const CemuKvPhysicalPage *page, bool write,
                     uint64_t start_ns, uint64_t *ready_ns)
{
    Fixture *fixture = opaque;
    uint64_t *available;
    uint64_t duration = write ? 200 : 40;

    fixture->schedules++;
    fixture->scheduled_page = *page;
    fixture->schedule_start = start_ns;
    if (fixture->fail_schedule) {
        return false;
    }
    if (fixture->bad_deadline) {
        *ready_ns = start_ns - 1;
        return true;
    }
    assert(page->channel < 2 && page->lun < 2);
    available = &fixture->available[page->channel][page->lun];
    start_ns = start_ns > *available ? start_ns : *available;
    if (start_ns > INT64_MAX - duration) {
        return false;
    }
    *available = *ready_ns = start_ns + duration;
    return true;
}

static const CemuKvIoOps io_ops = { .transfer = transfer, .schedule = schedule };

static CemuKvMapConfig configuration(void)
{
    return (CemuKvMapConfig){
        .controller = 1, .namespace_id = 1,
        .nand = { .channels = 2, .luns_per_channel = 2, .planes_per_lun = 1,
                  .blocks_per_plane = 4, .pages_per_block = 4, .page_bytes = 4096 },
        .backend_bytes = 262144, .max_pages = 64, .max_tasks = 2,
        .memory_budget_bytes = 65536,
    };
}

static void open_fixture(Fixture *fixture)
{
    char path[] = "/tmp/cemu-native-io-XXXXXX";
    CemuKvMapConfig config = configuration();
    CemuKvGeometry geometry = {
        .batches = 1, .layers = 1, .heads = 2, .head_dim = 128,
        .max_tokens = 259, .channels_per_page = 8, .element_bytes = 2, .page_bytes = 4096,
    };

    memset(fixture, 0, sizeof(*fixture));
    fixture->bytes = config.backend_bytes;
    fixture->descriptor = mkstemp(path);
    assert(fixture->descriptor >= 0);
    assert(unlink(path) == 0);
    assert(ftruncate(fixture->descriptor, fixture->bytes) == 0);
    fixture->space = mmap(NULL, fixture->bytes, PROT_READ | PROT_WRITE,
                          MAP_SHARED, fixture->descriptor, 0);
    assert(fixture->space != MAP_FAILED);
    assert(cemu_kv_io_create(&config, &io_ops, fixture, &fixture->io) == CEMU_KV_OK);
    assert(cemu_kv_io_register(fixture->io, &identity, &geometry) == CEMU_KV_OK);
}

static void close_fixture(Fixture *fixture)
{
    cemu_kv_io_destroy(fixture->io);
    assert(munmap(fixture->space, fixture->bytes) == 0);
    assert(close(fixture->descriptor) == 0);
}

static void test_roundtrip(void)
{
    Fixture fixture;
    CemuKvCoordinate coordinate = {0};
    CemuKvIoResult written[3], read;
    unsigned char input[4096], output[4096], disk[4096];
    unsigned kind;

    open_fixture(&fixture);
    for (kind = 0; kind < 3; kind++) {
        coordinate.kind = kind;
        memset(input, 0x10 + kind, sizeof(input));
        assert(cemu_kv_io_write(fixture.io, &identity, &coordinate, 16, input,
                                sizeof(input), 100, &written[kind]) == CEMU_KV_OK);
        assert(fixture.last_offset == written[kind].page.backend_offset);
        assert(pread(fixture.descriptor, disk, sizeof(disk), fixture.last_offset) == sizeof(disk));
        assert(memcmp(disk, input, sizeof(input)) == 0);
        assert(written[kind].transferred_bytes == sizeof(input));
        assert(cemu_kv_io_read(fixture.io, &identity, &coordinate, output,
                               sizeof(output), 101, &read) == CEMU_KV_OK);
        assert(memcmp(input, output, sizeof(input)) == 0);
        assert(read.page.ticket == written[kind].page.ticket);
        assert(read.start_ns == written[kind].ready_ns);
        assert(read.ready_ns >= read.start_ns + 40);
        assert(read.transferred_bytes == sizeof(output));
    }
    assert(written[0].page.backend_offset != written[1].page.backend_offset);
    assert(written[1].page.backend_offset != written[2].page.backend_offset);
    assert(written[0].page.backend_offset != written[2].page.backend_offset);
    assert(fixture.dirty_count == 3 && fixture.prepare_count == 3);
    assert(cemu_kv_io_stats(fixture.io).written_pages == 3);
    close_fixture(&fixture);
}

static void test_dependencies(void)
{
    Fixture fixture;
    CemuKvCoordinate coordinate = {0};
    CemuKvIoResult first, second, read;
    unsigned char data[4096] = {0};

    open_fixture(&fixture);
    assert(cemu_kv_io_write(fixture.io, &identity, &coordinate, 16, data,
                            sizeof(data), 100, &first) == CEMU_KV_OK);
    assert(first.ready_ns == 300);
    assert(cemu_kv_io_read(fixture.io, &identity, &coordinate, data,
                           sizeof(data), 110, &read) == CEMU_KV_OK);
    assert(read.start_ns == 300 && read.ready_ns == 340);
    coordinate.token = 16;
    assert(cemu_kv_io_write(fixture.io, &identity, &coordinate, 16, data,
                            sizeof(data), 110, &second) == CEMU_KV_OK);
    assert(second.page.physical.channel == 1 && second.ready_ns == 310);
    coordinate.token = 0;
    coordinate.head = 1;
    assert(cemu_kv_io_write(fixture.io, &identity, &coordinate, 16, data,
                            sizeof(data), 110, &second) == CEMU_KV_OK);
    assert(second.page.physical.channel == 0 && second.ready_ns == 540);
    coordinate.head = 0;
    coordinate.token = 3;
    coordinate.channel = 7;
    assert(cemu_kv_io_read(fixture.io, &identity, &coordinate, data,
                           sizeof(data), 120, &read) == CEMU_KV_OK);
    assert(read.start_ns == 300 && read.ready_ns == 580);
    assert(read.element_offset == (3 * 128 + 7) * 2);
    assert(fixture.scheduled_page.page == first.page.physical.page);
    assert(fixture.scheduled_page.block == first.page.physical.block);
    close_fixture(&fixture);
}

static void test_failures(void)
{
    Fixture fixture;
    CemuKvCoordinate coordinate = {0};
    CemuKvIoResult result;
    unsigned char input[4096], output[4096], unchanged[sizeof(result)];
    size_t schedules, transfers;

    open_fixture(&fixture);
    memset(input, 0x44, sizeof(input));
    memset(&result, 0xa5, sizeof(result));
    memcpy(unchanged, &result, sizeof(result));
    fixture.fail_io = true;
    assert(cemu_kv_io_write(fixture.io, &identity, &coordinate, 16, input,
                            sizeof(input), 100, &result) == CEMU_KV_IO_FAILED);
    assert(memcmp(unchanged, &result, sizeof(result)) == 0);
    fixture.fail_io = false;
    schedules = fixture.schedules;
    transfers = fixture.transfers;
    assert(cemu_kv_io_read(fixture.io, &identity, &coordinate, output,
                           sizeof(output), 1000, &result) == CEMU_KV_IO_FAILED);
    assert(cemu_kv_io_write(fixture.io, &identity, &coordinate, 16, input,
                            sizeof(input), 1000, &result) == CEMU_KV_EXISTS);
    assert(fixture.schedules == schedules && fixture.transfers == transfers);
    coordinate.token = 16;
    fixture.fail_schedule = true;
    assert(cemu_kv_io_write(fixture.io, &identity, &coordinate, 16, input,
                            sizeof(input), 100, &result) == CEMU_KV_IO_FAILED);
    assert(fixture.transfers == transfers);
    fixture.fail_schedule = false;
    coordinate.token = 32;
    fixture.bad_deadline = true;
    assert(cemu_kv_io_write(fixture.io, &identity, &coordinate, 16, input,
                            sizeof(input), 100, &result) == CEMU_KV_IO_FAILED);
    fixture.bad_deadline = false;
    assert(fixture.transfers == transfers && cemu_kv_io_stats(fixture.io).failed_pages == 3);
    coordinate.token = 48;
    assert(cemu_kv_io_write(fixture.io, &identity, &coordinate, 16, input,
                            sizeof(input), 1000, &result) == CEMU_KV_OK);
    fixture.fail_io = true;
    memset(output, 0x33, sizeof(output));
    assert(cemu_kv_io_read(fixture.io, &identity, &coordinate, output,
                           sizeof(output), 1200, &result) == CEMU_KV_IO_FAILED);
    for (size_t index = 0; index < sizeof(output); index++) {
        assert(output[index] == 0x33);
    }
    fixture.fail_io = false;
    fixture.fail_sync = true;
    assert(cemu_kv_io_read(fixture.io, &identity, &coordinate, output,
                           sizeof(output), 1400, &result) == CEMU_KV_IO_FAILED);
    assert(output[0] == 0x33 && output[4095] == 0x33);
    fixture.fail_sync = false;
    assert(cemu_kv_io_read(fixture.io, &identity, &coordinate, output,
                           sizeof(output), 1600, &result) == CEMU_KV_OK);
    assert(memcmp(input, output, sizeof(input)) == 0);
    close_fixture(&fixture);
}

static void test_validation(void)
{
    Fixture fixture;
    CemuKvCoordinate coordinate = {0};
    CemuKvIdentity stale = identity;
    CemuKvIoResult result;
    CemuKvMapConfig config = configuration();
    CemuKvIo *other = NULL;
    unsigned char data[4096] = {0};
    size_t budget;

    open_fixture(&fixture);
    assert(cemu_kv_io_write(fixture.io, &identity, &coordinate, 16, data,
                            sizeof(data) - 1, 100, &result) == CEMU_KV_INVALID);
    assert(cemu_kv_io_write(fixture.io, &identity, &coordinate, 16, data,
                            sizeof(data), UINT64_MAX, &result) == CEMU_KV_INVALID);
    assert(cemu_kv_io_read(fixture.io, &identity, &coordinate, data,
                           sizeof(data), 100, &result) == CEMU_KV_NOT_FOUND);
    stale.generation++;
    assert(cemu_kv_io_write(fixture.io, &stale, &coordinate, 16, data,
                            sizeof(data), 100, &result) == CEMU_KV_STALE);
    assert(!fixture.schedules && !fixture.transfers);
    coordinate.token = 256;
    coordinate.kind = CEMU_KV_PAGE_CHANNEL_K;
    assert(cemu_kv_io_write(fixture.io, &identity, &coordinate, 3, data,
                            sizeof(data), 100, &result) == CEMU_KV_OK);
    coordinate.token = 259;
    assert(cemu_kv_io_read(fixture.io, &identity, &coordinate, data,
                           sizeof(data), 1000, &result) == CEMU_KV_INVALID);
    coordinate.token = 258;
    coordinate.channel = 7;
    assert(cemu_kv_io_read(fixture.io, &identity, &coordinate, data,
                           sizeof(data), 1000, &result) == CEMU_KV_OK);
    assert(result.element_offset == (7 * 256 + 2) * 2);
    budget = cemu_kv_io_stats(fixture.io).memory_bytes;
    assert(budget <= config.memory_budget_bytes);
    config.memory_budget_bytes = budget - 1;
    assert(cemu_kv_io_create(&config, &io_ops, &fixture, &other) == CEMU_KV_LIMIT);
    assert(!other);
    config.memory_budget_bytes++;
    assert(cemu_kv_io_create(&config, &io_ops, &fixture, &other) == CEMU_KV_OK);
    cemu_kv_io_destroy(other);
    close_fixture(&fixture);
}

static void test_backend_bounds(void)
{
    Fixture fixture;
    unsigned char data[4096] = {0};

    open_fixture(&fixture);
    assert(cemu_mapped_io(fixture.space, fixture.bytes, data, fixture.bytes - 4095,
                          sizeof(data), true, &mapped_ops, &fixture) == -EINVAL);
    assert(cemu_mapped_io(fixture.space, fixture.bytes, data, UINT64_MAX,
                          sizeof(data), true, &mapped_ops, &fixture) == -EINVAL);
    assert(cemu_mapped_io(fixture.space, fixture.bytes, data, 0,
                          UINT64_MAX, true, &mapped_ops, &fixture) == -EINVAL);
    assert(!fixture.dirty_count && !fixture.prepare_count);
    assert(cemu_mapped_io(fixture.space, fixture.bytes, data, fixture.bytes - 4096,
                          sizeof(data), true, &mapped_ops, &fixture) == 0);
    assert(fixture.dirty_count == 1);
    fixture.fail_sync = true;
    memset(data, 0x55, sizeof(data));
    assert(cemu_mapped_io(fixture.space, fixture.bytes, data, fixture.bytes - 4096,
                          sizeof(data), false, &mapped_ops, &fixture) == -EIO);
    assert(data[0] == 0x55 && data[4095] == 0x55);
    close_fixture(&fixture);
}

int main(int argc, char **argv)
{
    assert(argc == 2);
    if (strcmp(argv[1], "roundtrip") == 0) {
        test_roundtrip();
    } else if (strcmp(argv[1], "dependencies") == 0) {
        test_dependencies();
    } else if (strcmp(argv[1], "failures") == 0) {
        test_failures();
    } else if (strcmp(argv[1], "validation") == 0) {
        test_validation();
    } else if (strcmp(argv[1], "bounds") == 0) {
        test_backend_bounds();
    } else {
        return 2;
    }
    printf("[instattention-io] PASS %s: temporary mmap file, mock NAND; no live VM\n", argv[1]);
    return 0;
}
