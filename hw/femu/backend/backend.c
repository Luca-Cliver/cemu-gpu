#include "./backend.h"
#include "../nvme.h"
#include <dlfcn.h>

typedef int (*cuda_set_decive_fn)(int);
typedef int (*cuda_malloc_fn)(void **, size_t);
typedef int (*cuda_free_fn)(void *);
typedef int (*cuda_memcpy_fn)(void *, const void *, size_t, int);
typedef int (*cuda_synchronize_fn)(void);

typedef struct CudaSyncApi
{
    bool loaded;
    void *module;
    cuda_set_decive_fn set_device;
    cuda_malloc_fn malloc;
    cuda_free_fn free;
    cuda_memcpy_fn memcpy;
    cuda_synchronize_fn synchronize;
} CudaSyncApi;

static CudaSyncApi cuda_api = {0};

typedef struct CudaMirrorRange {
    void *host_ptr;
    uint64_t len;
    void *device_ptr;
    struct CudaMirrorRange *next;
} CudaMirrorRange;

static CudaMirrorRange *cuda_mirror_head(SsdBackend *b)
{
    return (CudaMirrorRange *)b->cuda_mirror;
}

static void cuda_mirror_set_head(SsdBackend *b, CudaMirrorRange *head)
{
    b->cuda_mirror = head;
}

static CudaMirrorRange *cuda_mirror_find(SsdBackend *b, void *host_ptr, uint64_t len)
{
    for (CudaMirrorRange *cur = cuda_mirror_head(b); cur; cur = cur->next) {
        if (cur->host_ptr == host_ptr && cur->len == len) {
            return cur;
        }
    }
    return NULL;
}

static CudaMirrorRange *cuda_mirror_ensure(SsdBackend *b, void *host_ptr, uint64_t len)
{
    CudaMirrorRange *entry = cuda_mirror_find(b, host_ptr, len);
    if (entry) {
        if (entry->len != len) {
            femu_err("CUDA mirror range size mismatch for host %p: have %lu, want %lu\n",
                     host_ptr, entry->len, len);
            return NULL;
        }
        return entry;
    }

    entry = g_malloc0(sizeof(*entry));
    entry->host_ptr = host_ptr;
    entry->len = len;
    if (cuda_api.malloc(&entry->device_ptr, len) != 0) {
        femu_err("Failed to allocate CUDA memory for host %p, len %lu.\n", host_ptr, len);
        g_free(entry);
        return NULL;
    }

    entry->next = cuda_mirror_head(b);
    cuda_mirror_set_head(b, entry);
    return entry;
}

static void *cuda_get_symbol(const char *name)
{
    return cuda_api.module ? dlsym(cuda_api.module, name) : NULL;
}

static void backend_cuda_close_module()
{
    if (!cuda_api.module) {
        return;
    }

    dlclose(cuda_api.module);
    cuda_api.module = NULL;
}

static int backend_cuda_load_module()
{
    const char *names[] = {"libcudart.so", "libcudart.so.11.0", "libcudart.so.12.0"};

    for (size_t i = 0; i < sizeof(names) / sizeof(names[0]); i++) {
        cuda_api.module = dlopen(names[i], RTLD_NOW);
        if (cuda_api.module) {
            return true;
        }
    }
    return false;
}

static int backend_cuda_init_api()
{
    if(cuda_api.loaded)
    {
        return true;
    }

    if (!backend_cuda_load_module()) {
        return false;
    }

    cuda_api.set_device = (cuda_set_decive_fn)cuda_get_symbol("cudaSetDevice");
    cuda_api.malloc = (cuda_malloc_fn)cuda_get_symbol("cudaMalloc");
    cuda_api.free = (cuda_free_fn)cuda_get_symbol("cudaFree");
    cuda_api.memcpy = (cuda_memcpy_fn)cuda_get_symbol("cudaMemcpy");
    cuda_api.synchronize = (cuda_synchronize_fn)cuda_get_symbol("cudaDeviceSynchronize");

    if (!cuda_api.set_device || !cuda_api.malloc || !cuda_api.free || !cuda_api.memcpy || !cuda_api.synchronize) {
        backend_cuda_close_module();
        return false;
    }
    cuda_api.loaded = true;
    return true;
}

int backend_cuda_sync_init(SsdBackend *b)
{
    const char *enable = g_getenv("CEMU_CUDA_SYNC");
    const char *decive_env = g_getenv("CEMU_CUDA_DEVICE");
    int device_id = decive_env ? atoi(decive_env) : 0;
    femu_debug("backend_cuda_sync_init: enable %s, device_id %d\n", enable, device_id);

    if(!enable || !enable[0] || !g_ascii_strcasecmp(enable, "0") || !g_ascii_strcasecmp(enable, "false"))
    {
        return 0;
    }

    if (!backend_cuda_init_api()) {
        femu_err("Failed to load CUDA runtime library or resolve symbols.\n");
        return -1;
    }

    if(cuda_api.set_device && cuda_api.set_device(device_id) != 0) {
        femu_err("Failed to set CUDA device %d.\n", device_id);
        return -1;
    }

    b->cuda_sync = true;
    b->cuda_mirror_valid = false;
    b->cuda_mirror = NULL;
    return 0;
}

void backend_cuda_sync_fini(SsdBackend *b)
{
    if (b && b->cuda_sync && cuda_api.free) {
        CudaMirrorRange *cur = cuda_mirror_head(b);
        while (cur) {
            CudaMirrorRange *next = cur->next;
            if (cur->device_ptr) {
                cuda_api.free(cur->device_ptr);
            }
            g_free(cur);
            cur = next;
        }
        cuda_mirror_set_head(b, NULL);
    }

    if(b)
    {
        b->cuda_sync = false;
        b->cuda_mirror_valid = false;
    }
}

void backend_cuda_sync_ptr(SsdBackend *b, void *ptr, uint64_t len, bool to_device)
{
    uint8_t *host;
    uint8_t *mirror;
    uint64_t offset;
    femu_debug("backend_cuda_sync_ptr: b %p, cuda_sync %d, ptr %p, len %lu, to_device %d\n",
             b, b ? b->cuda_sync : -1, ptr, len, to_device);

    if(!b || !b->cuda_sync || !ptr || len == 0 || !cuda_api.memcpy)
    {
        return;
    }

    host = (uint8_t *)b->logical_space;
    if((uint8_t *)ptr < host)
    {
        femu_err("Pointer %p out of backend logical space range!\n", ptr);
        return;
    }

    offset = (uint64_t)((uint8_t *)ptr - host);
    if(offset >= (uint64_t)b->size || offset + len > (uint64_t)b->size)
    {
        femu_err("Pointer %p with len %lu out of backend logical space range!\n", ptr, len);
        return;
    }

    CudaMirrorRange *entry = NULL;
    if (to_device) {
        /* For host->device we need an allocation/mirror; ensure it exists */
        entry = cuda_mirror_ensure(b, ptr, len);
        if (!entry) {
            return;
        }
    } else {
        /* For device->host only use an existing mirror; don't allocate on read-path */
        entry = cuda_mirror_find(b, ptr, len);
        if (!entry) {
            femu_debug("backend_cuda_sync_ptr: no mirror entry for ptr %p len %lu; skipping device->host copy\n", ptr, len);
            return;
        }
    }

    if(to_device)
    {
        int rc = cuda_api.memcpy(entry->device_ptr, ptr, len, 1 /* host to device */);
        if (rc != 0) {
            femu_err("cudaMemcpy host->device failed: rc=%d, ptr=%p, len=%lu\n", rc, ptr, len);
            return;
        }
        /* mark mirror as containing up-to-date data for this range */
        b->cuda_mirror_valid = true;
    }
    else
    {
        /* Only copy back from device if mirror has valid data. If the mirror
         * hasn't been initialized from the host (no prior host->device copy),
         * skip device->host copy to avoid overwriting valid host memory with
         * uninitialized device contents.
         */
        if (!b->cuda_mirror_valid) {
            femu_debug("backend_cuda_sync_ptr: skipping device->host copy because mirror not valid (ptr %p, len %lu)\n", ptr, len);
        } else {
            int rc = cuda_api.memcpy(ptr, entry->device_ptr, len, 2 /* device to host */);
            if (rc != 0) {
                femu_err("cudaMemcpy device->host failed: rc=%d, ptr=%p, len=%lu\n", rc, ptr, len);
                return;
            }
        }
    }

    /* cudaMemcpy on the default stream is already synchronous.
     * Keep an explicit device sync optional so normal file I/O is not
     * serialized behind a full-device barrier.
     */
    if (cuda_api.synchronize && g_getenv("CEMU_CUDA_SYNC_WAIT"))
    {
        int rc = cuda_api.synchronize();
        if (rc != 0) {
            femu_err("cudaDeviceSynchronize failed: rc=%d, ptr=%p, len=%lu\n", rc, ptr, len);
        }
    }
}

void *backend_host_to_device(SsdBackend *b, void *host_ptr, uint64_t len)
{
    uint8_t *host;
    uint64_t offset;
    femu_debug("backend_host_to_device: b %p, cuda_sync %d, host_ptr %p, len %lu\n",
             b, b ? b->cuda_sync : -1, host_ptr, len);

    if(!b || !b->cuda_sync || !host_ptr)
    {
        return NULL;
    }

    host = (uint8_t *)b->logical_space;
    if((uint8_t *)host_ptr < host)
    {
        femu_err("Pointer %p out of backend logical space range!\n", host_ptr);
        return NULL;
    }

    offset = (uint64_t)((uint8_t *)host_ptr - host);
    if(offset >= (uint64_t)b->size)
    {
        femu_err("Pointer %p out of backend logical space range!\n", host_ptr);
        return NULL;
    }

    CudaMirrorRange *entry = cuda_mirror_find(b, host_ptr, len);
    if (!entry) {
        entry = cuda_mirror_ensure(b, host_ptr, len);
        if (!entry) {
            femu_debug("backend_host_to_device: no CUDA mirror for host_ptr %p len %lu\n", host_ptr, len);
            return NULL;
        }
    }

    return entry->device_ptr;
}

int init_backend(SsdBackend **mbe, BackendType type, char *path, int64_t nbytes)
{
    SsdBackend *b = *mbe = g_malloc0(sizeof(SsdBackend));
    if (b == NULL) {
        femu_err("Failed to allocate memory for ssd backend!\n");
        return -1;
    }

    b->type = type;
    b->size = nbytes;

    switch(b->type) {
    case FEMU_DRAM_BACKEND:
        return init_dram_backend(b);
    default:
        femu_err("Unknown backend type!\n");
        abort();
        return -1;
    }
}

void free_backend(SsdBackend *b)
{
    switch(b->type) {
    case FEMU_DRAM_BACKEND:
        free_dram_backend(b);
        break;
    default:
        femu_err("Unknown backend type!\n");
    }
    g_free(b);
}

int backend_rw(SsdBackend *b, QEMUSGList *qsg, uint64_t *lbal, bool is_write)
{
    int sg_cur_index = 0;
    dma_addr_t sg_cur_byte = 0;
    dma_addr_t cur_addr, cur_len;
    uint64_t mb_oft = lbal[0];
    void *mb = b->logical_space;

    DMADirection dir = DMA_DIRECTION_FROM_DEVICE;

    if (is_write) {
        dir = DMA_DIRECTION_TO_DEVICE;
    }

    while (sg_cur_index < qsg->nsg) {
        cur_addr = qsg->sg[sg_cur_index].base + sg_cur_byte;
        cur_len = qsg->sg[sg_cur_index].len - sg_cur_byte;
        if (dma_memory_rw(qsg->as, cur_addr, mb + mb_oft, cur_len, dir, MEMTXATTRS_UNSPECIFIED)) {
            femu_err("dma_memory_rw error\n");
        }

        sg_cur_byte += cur_len;
        if (sg_cur_byte == qsg->sg[sg_cur_index].len) {
            sg_cur_byte = 0;
            ++sg_cur_index;
        }

        if (b->femu_mode == FEMU_OCSSD_MODE) {
            mb_oft = lbal[sg_cur_index];
        } else if (b->femu_mode == FEMU_BBSSD_MODE ||
                   b->femu_mode == FEMU_NOSSD_MODE ||
                   b->femu_mode == FEMU_ZNSSD_MODE ||
                   b->femu_mode == FEMU_CSD_MODE) {
            mb_oft += cur_len;
        } else {
            assert(0);
        }
    }

    qemu_sglist_destroy(qsg);

    return 0;
}

void backend_rw_internal(SsdBackend *b, void *buf, uint64_t data_offset,
                         uint64_t data_size, int is_write)
{
    const uint64_t copy_threshold = 16UL * 1024UL;
    const uint64_t page_size = 4096;
    const uint64_t page_mask = page_size - 1;

    void *src, *dest;
    if (is_write) {
        src = buf;
        dest = b->logical_space + data_offset;
    } else {
        backend_cuda_sync_ptr(b, b->logical_space + data_offset, data_size, false);
        src = b->logical_space + data_offset;
        dest = buf;
    }
    if (data_size <= copy_threshold) {
        memmove(dest, src, data_size);
    } else {
        uint64_t remain = data_size;
        if (data_offset & page_mask) {
            uint64_t sz = page_size - (data_offset & page_mask);
            memmove(dest, src, sz);
            remain -= sz;
            dest += sz;
            src += sz;
        }
        while (remain > 0) {
            uint64_t sz = remain > copy_threshold ? copy_threshold : remain;
            memmove(dest, src, sz);
            remain -= sz;
            dest += sz;
            src += sz;
        }
    }

    if(is_write) {
        backend_cuda_sync_ptr(b, b->logical_space + data_offset, data_size, true);
    }
}

void *backend_get_ptr(SsdBackend *b, uint64_t offset)
{
    return b->logical_space + offset;
}

void backend_copy_internal(SsdBackend *b, uint64_t doff, uint64_t soff,
                           uint64_t data_size)
{
    void *dest = b->logical_space + doff;
    backend_rw_internal(b, dest, soff, data_size, 0);
}