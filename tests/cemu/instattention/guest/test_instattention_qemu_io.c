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
    unexpected_cuda_lock(mutex, file, line);
}

int64_t qemu_clock_get_ns(QEMUClockType type)
{
    assert(type == QEMU_CLOCK_VIRTUAL);
    return 100;
}

int main(void)
{
    char path[] = "/tmp/cemu-native-qemu-XXXXXX";
    int descriptor = mkstemp(path);
    const size_t size = 262144;
    FemuCtrl controller = { .drive_id = 1 };
    SsdBackend backend = { .size = size, .cuda_sync = false };
    NvmeNamespace namespace = { .ctrl = &controller, .id = 1,
                                .csi = NVME_CSI_NVM, .backend = &backend };
    NvmParams params = {
        .nchs = 2, .luns_per_ch = 2, .pls_per_lun = 1, .blks_per_pl = 4,
        .pgs_per_blk = 4, .secsz = 512, .secs_per_pg = 8,
        .pg_rd_lat = 40, .pg_wr_lat = 200,
    };
    struct nand_lun luns[2][2] = {0};
    struct ssd_channel channels[2] = { { .lun = luns[0] }, { .lun = luns[1] } };
    NvmNamespace nns = { .namespace = &namespace, .params = &params, .ch = channels };
    CemuKvIo *io = NULL, *duplicate = NULL;
    CemuKvIdentity identity = { .controller = 1, .namespace_id = 1,
                                .task = 1, .generation = 1 };
    CemuKvGeometry shape = {
        .batches = 1, .layers = 1, .heads = 2, .head_dim = 128,
        .max_tokens = 259, .channels_per_page = 8, .element_bytes = 2, .page_bytes = 4096,
    };
    CemuKvCoordinate coordinate = {0};
    CemuKvIoResult written, read, second;
    unsigned char input[4096], output[4096], disk[4096];
    struct ppa legacy_page = {0};

    assert(descriptor >= 0 && unlink(path) == 0);
    assert(ftruncate(descriptor, size) == 0);
    backend.logical_space = mmap(NULL, size, PROT_READ | PROT_WRITE,
                                 MAP_SHARED, descriptor, 0);
    assert(backend.logical_space != MAP_FAILED);
    namespace.private = &nns;
    nns.maptbl = &legacy_page;
    assert(bb_native_io_create(&namespace, 2, 64, 65536, &io) == CEMU_KV_INVALID);
    assert(!io && !nns.native_io);
    nns.maptbl = NULL;
    assert(bb_native_io_create(&namespace, 2, 64, 65536, &io) == CEMU_KV_OK);
    assert(nns.native_io == io);
    assert(bb_native_io_create(&namespace, 2, 64, 65536, &duplicate) == CEMU_KV_INVALID);
    assert(!duplicate);
    assert(cemu_kv_io_register(io, &identity, &shape) == CEMU_KV_OK);
    memset(input, 0x91, sizeof(input));
    assert(cemu_kv_io_write(io, &identity, &coordinate, 16, input, sizeof(input),
                            100, &written) == CEMU_KV_OK);
    assert(written.ready_ns == 300 && luns[0][0].next_lun_avail_time == 300);
    assert(pread(descriptor, disk, sizeof(disk), written.page.backend_offset) == sizeof(disk));
    assert(memcmp(disk, input, sizeof(input)) == 0);
    assert(cemu_kv_io_read(io, &identity, &coordinate, output, sizeof(output),
                           110, &read) == CEMU_KV_OK);
    assert(read.start_ns == 300 && read.ready_ns == 340);
    assert(luns[0][0].next_lun_avail_time == 340);
    assert(memcmp(input, output, sizeof(input)) == 0);
    coordinate.token = 16;
    assert(cemu_kv_io_write(io, &identity, &coordinate, 16, input, sizeof(input),
                            110, &second) == CEMU_KV_OK);
    assert(second.page.physical.channel == 1 && second.ready_ns == 310);
    assert(luns[1][0].next_lun_avail_time == 310);
    coordinate.token = 0;
    coordinate.head = 1;
    assert(cemu_kv_io_write(io, &identity, &coordinate, 16, input, sizeof(input),
                            110, &second) == CEMU_KV_OK);
    assert(second.page.physical.page == 1 && second.ready_ns == 540);
    assert(luns[0][0].next_lun_avail_time == 540);
    assert(backend_rw_internal_checked(&backend, input, size, sizeof(input), true) == -EINVAL);
    coordinate.token = 32;
    assert(cemu_kv_io_write(io, &identity, &coordinate, 16, input, sizeof(input),
                            INT64_MAX, &second) == CEMU_KV_IO_FAILED);
    assert(luns[0][0].next_lun_avail_time == 540);
    bb_native_io_destroy(&namespace);
    assert(!nns.native_io);
    assert(bb_native_io_create(&namespace, 2, 64, 65536, &duplicate) == CEMU_KV_INVALID);
    g_mutex_clear(&nns.native_io_lock);
    assert(munmap(backend.logical_space, size) == 0 && close(descriptor) == 0);
    puts("[instattention-qemu-io] PASS: production FTL NAND/checked backend adapters; "
         "temporary file, controlled virtual clock, no worker/Guest integration");
    return 0;
}
