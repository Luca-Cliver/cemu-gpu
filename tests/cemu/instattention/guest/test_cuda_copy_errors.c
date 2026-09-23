#include "qemu/osdep.h"
#include "hw/femu/backend/backend.c"
#include "hw/femu/bbssd/ftl.c"

static unsigned copy_calls;
static unsigned lock_depth;

static void test_lock(QemuMutex *mutex, const char *file, int line)
{
    (void)mutex;
    (void)file;
    (void)line;
    assert(lock_depth++ == 0);
}

QemuMutexLockFunc qemu_mutex_lock_func = test_lock;

void qemu_mutex_unlock_impl(QemuMutex *mutex, const char *file, int line)
{
    (void)mutex;
    (void)file;
    (void)line;
    assert(lock_depth-- == 1);
}

int64_t qemu_clock_get_ns(QEMUClockType type)
{
    assert(type == QEMU_CLOCK_VIRTUAL);
    return 100;
}

size_t femu_ring_enqueue(struct rte_ring *ring, void **objects, size_t count)
{
    (void)ring;
    (void)objects;
    (void)count;
    abort();
}

static int failed_copy(void *destination, const void *source, size_t bytes, int kind)
{
    (void)destination;
    (void)source;
    assert(bytes == 4096 && kind == 2);
    ++copy_calls;
    return 999;
}

int main(void)
{
    uint8_t memory[8192], device[4096], output[4096], expected[4096];
    CudaMirrorSegment segment = { .start = 0, .end = 4096, .state = CUDA_RANGE_DEVICE_DIRTY };
    CudaMirrorRange mirror = {
        .host_ptr = memory, .len = 4096, .device_ptr = device,
        .segments = &segment, .nr_segments = 1, .segment_capacity = 1,
    };
    SsdBackend backend = {
        .logical_space = memory, .size = sizeof(memory), .cuda_sync = true, .cuda_mirror = &mirror,
    };
    FemuCtrl controller = {0};
    NvmeNamespace namespaces[2] = {
        { .ctrl = &controller, .id = 1, .backend = &backend },
        { .ctrl = &controller, .id = 2 },
    };
    NvmeCopyFormat extent = {0};
    NvmeRequest request = {
        .ns = &namespaces[1], .mem_ctrl = &controller,
        .sdaddr = output, .sres = &extent, .nr_sres = 1,
    };
    controller.namespaces = namespaces;
    namespaces[0].id_ns.lbaf[0].lbads = 9;
    extent.cf2.snsid = cpu_to_le32(1);
    extent.cf2.nlb = cpu_to_le16(7);
    cuda_api.memcpy = failed_copy;
    memset(memory, 0x51, sizeof(memory));
    memset(output, 0xcc, sizeof(output));
    memcpy(expected, output, sizeof(expected));

    assert(backend_rw_internal(&backend, output, 0, sizeof(output), false) == -EIO);
    assert(memcmp(output, expected, sizeof(output)) == 0);
    assert(segment.state == CUDA_RANGE_DEVICE_DIRTY);
    assert(backend_copy_internal(&backend, 4096, 0, 4096) == -EIO);
    assert(memory[4096] == 0x51);

    assert(memory_copy_nvm(&namespaces[1], &request) == 0);
    assert(request.status == (NVME_INTERNAL_DEV_ERROR | NVME_DNR));
    assert(memcmp(output, expected, sizeof(output)) == 0);
    request.status = NVME_SUCCESS;
    request.is_write = 1;
    request.sdaddr = memory;
    request.sdaddr_backend = &backend;
    assert(memory_copy_nvm(&namespaces[1], &request) == 0);
    assert(request.status == (NVME_INTERNAL_DEV_ERROR | NVME_DNR));
    assert(copy_calls == 4 && lock_depth == 0);

    gint cancelled = 0;
    request.stat.stime = 1000000;
    assert(!memory_copy_pending_ready(&request, 100));
    request.sparf_cancelled = &cancelled;
    assert(!memory_copy_pending_ready(&request, 100));
    g_atomic_int_set(&cancelled, 1);
    assert(memory_copy_pending_ready(&request, 100));
    request.status = NVME_SUCCESS;
    assert(memory_copy_nvm(&namespaces[1], &request) == 0);
    assert(request.status == (NVME_CMD_ABORT_REQ | NVME_DNR));
    assert(request.stat.expire_time == 100);
    assert(copy_calls == 4);
    request.sparf_cancelled = NULL;
    assert(!memory_copy_pending_ready(&request, 100));

    backend.cuda_sync = false;
    assert(backend_rw_internal(&backend, output, 0, sizeof(output), false) == 0);
    assert(memcmp(output, memory, sizeof(output)) == 0);
    assert(backend_copy_internal(&backend, 4096, 0, 4096) == 0);
    assert(copy_calls == 4 && lock_depth == 0);
    assert(backend_rw_internal(&backend, output, sizeof(memory), sizeof(output), true) == -EINVAL);
    puts("[cuda-copy-errors] PASS: injected D2H failures reach backend/FTL callers, "
         "no false success or output overwrite; CUDA-disabled copies unchanged; no GPU used");
    return 0;
}
