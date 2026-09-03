#include "./backend.h"
#include "../nvme.h"
#include <dlfcn.h>

typedef int (*cuda_set_decive_fn)(int);
typedef int (*cuda_malloc_fn)(void **, size_t);
typedef int (*cuda_free_fn)(void *);
typedef int (*cuda_memcpy_fn)(void *, const void *, size_t, int);

typedef struct CudaSyncApi
{
    bool loaded;
    void *module;
    cuda_set_decive_fn set_device;
    cuda_malloc_fn malloc;
    cuda_free_fn free;
    cuda_memcpy_fn memcpy;
} CudaSyncApi;

static CudaSyncApi cuda_api = {0};

static bool backend_cuda_valid_range(SsdBackend *b, void *ptr, uint64_t len);

typedef enum CudaRangeState {
    CUDA_RANGE_UNINITIALIZED,
    CUDA_RANGE_CLEAN,
    CUDA_RANGE_HOST_DIRTY,
    CUDA_RANGE_DEVICE_DIRTY,
} CudaRangeState;

typedef struct CudaMirrorSegment {
    uint64_t start;
    uint64_t end;
    CudaRangeState state;
} CudaMirrorSegment;

typedef struct CudaMirrorRange {
    void *host_ptr;
    uint64_t len;
    void *device_ptr;
    CudaMirrorSegment *segments;
    uint32_t nr_segments;
    uint32_t segment_capacity;
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

static bool cuda_ranges_overlap(void *first_ptr, uint64_t first_len,
                                void *second_ptr, uint64_t second_len)
{
    uintptr_t first = (uintptr_t)first_ptr;
    uintptr_t second = (uintptr_t)second_ptr;

    return first <= second ? second - first < first_len
                           : first - second < second_len;
}

static CudaMirrorRange *cuda_mirror_find_exact(SsdBackend *b, void *host_ptr,
                                                uint64_t len)
{
    for (CudaMirrorRange *cur = cuda_mirror_head(b); cur; cur = cur->next) {
        if (cur->host_ptr == host_ptr && cur->len == len) {
            return cur;
        }
    }
    return NULL;
}

static void cuda_mirror_reserve_segments(CudaMirrorRange *entry,
                                         uint32_t capacity)
{
    if (capacity <= entry->segment_capacity) {
        return;
    }

    uint32_t new_capacity = MAX(capacity, entry->segment_capacity * 2);
    entry->segments = g_renew(CudaMirrorSegment, entry->segments,
                              new_capacity);
    entry->segment_capacity = new_capacity;
}

static uint32_t cuda_mirror_split_segment(CudaMirrorRange *entry,
                                          uint64_t offset)
{
    for (uint32_t index = 0; index < entry->nr_segments; index++) {
        CudaMirrorSegment *segment = &entry->segments[index];
        if (offset == segment->start) {
            return index;
        }
        if (offset > segment->start && offset < segment->end) {
            CudaMirrorSegment right = {
                .start = offset,
                .end = segment->end,
                .state = segment->state,
            };

            cuda_mirror_reserve_segments(entry, entry->nr_segments + 1);
            segment = &entry->segments[index];
            memmove(&entry->segments[index + 2],
                    &entry->segments[index + 1],
                    (entry->nr_segments - index - 1) * sizeof(*segment));
            segment->end = offset;
            entry->segments[index + 1] = right;
            entry->nr_segments++;
            return index + 1;
        }
    }
    return entry->nr_segments;
}

static void cuda_mirror_merge_segments(CudaMirrorRange *entry)
{
    uint32_t index = 0;

    while (index + 1 < entry->nr_segments) {
        CudaMirrorSegment *segment = &entry->segments[index];
        CudaMirrorSegment *next = &entry->segments[index + 1];
        if (segment->state == next->state && segment->end == next->start) {
            segment->end = next->end;
            memmove(next, next + 1,
                    (entry->nr_segments - index - 2) * sizeof(*next));
            entry->nr_segments--;
        } else {
            index++;
        }
    }
}

static void cuda_mirror_set_state(CudaMirrorRange *entry, uint64_t start,
                                  uint64_t end, CudaRangeState state)
{
    if (start >= end || end > entry->len) {
        return;
    }

    cuda_mirror_split_segment(entry, start);
    if (end < entry->len) {
        cuda_mirror_split_segment(entry, end);
    }

    for (uint32_t index = 0; index < entry->nr_segments; index++) {
        CudaMirrorSegment *segment = &entry->segments[index];
        if (segment->start >= end) {
            break;
        }
        if (segment->end > start) {
            segment->state = state;
        }
    }
    cuda_mirror_merge_segments(entry);
}

static CudaMirrorRange *cuda_mirror_ensure_locked(SsdBackend *b,
                                                   void *host_ptr,
                                                   uint64_t len)
{
    CudaMirrorRange *entry = cuda_mirror_find_exact(b, host_ptr, len);
    if (entry) {
        return entry;
    }

    for (CudaMirrorRange *cur = cuda_mirror_head(b); cur; cur = cur->next) {
        if (cuda_ranges_overlap(cur->host_ptr, cur->len, host_ptr, len)) {
            femu_err("Overlapping CUDA mirrors are unsupported: existing %p/%lu, requested %p/%lu\n",
                     cur->host_ptr, cur->len, host_ptr, len);
            return NULL;
        }
    }

    entry = g_malloc0(sizeof(*entry));
    entry->host_ptr = host_ptr;
    entry->len = len;
    entry->segment_capacity = 4;
    entry->segments = g_new0(CudaMirrorSegment, entry->segment_capacity);
    entry->nr_segments = 1;
    entry->segments->end = len;
    entry->segments->state = CUDA_RANGE_UNINITIALIZED;
    if (cuda_api.malloc(&entry->device_ptr, len) != 0) {
        femu_err("Failed to allocate CUDA memory for host %p, len %lu.\n", host_ptr, len);
        g_free(entry->segments);
        g_free(entry);
        return NULL;
    }

    entry->next = cuda_mirror_head(b);
    cuda_mirror_set_head(b, entry);
    return entry;
}

static void cuda_mirror_free(CudaMirrorRange *entry)
{
    if (entry->device_ptr) {
        cuda_api.free(entry->device_ptr);
    }
    g_free(entry->segments);
    g_free(entry);
}

static int cuda_mirror_sync_segment(CudaMirrorRange *entry,
                                    CudaMirrorSegment *segment,
                                    bool to_device)
{
    uint8_t *host = (uint8_t *)entry->host_ptr + segment->start;
    uint8_t *device = (uint8_t *)entry->device_ptr + segment->start;
    uint64_t len = segment->end - segment->start;
    int kind = to_device ? 1 : 2;
    int rc = cuda_api.memcpy(to_device ? device : host,
                             to_device ? host : device, len, kind);

    femu_debug("CUDA mirror %s: host=%p, offset=%lu, len=%lu\n",
               to_device ? "H2D" : "D2H", entry->host_ptr,
               segment->start, len);
    if (rc != 0) {
        femu_err("cudaMemcpy %s failed: rc=%d, ptr=%p, offset=%lu, len=%lu\n",
                 to_device ? "host->device" : "device->host", rc,
                 entry->host_ptr, segment->start, len);
        return -1;
    }
    segment->state = CUDA_RANGE_CLEAN;
    return 0;
}

static int cuda_mirror_prepare_locked(SsdBackend *b, void *ptr, uint64_t len,
                                      bool to_device)
{
    uintptr_t request_start = (uintptr_t)ptr;
    uintptr_t request_end = request_start + len;

    for (CudaMirrorRange *entry = cuda_mirror_head(b); entry;
         entry = entry->next) {
        uintptr_t entry_start = (uintptr_t)entry->host_ptr;
        uintptr_t entry_end = entry_start + entry->len;
        uintptr_t overlap_start = MAX(request_start, entry_start);
        uintptr_t overlap_end = MIN(request_end, entry_end);

        if (overlap_start >= overlap_end) {
            continue;
        }

        uint64_t start = overlap_start - entry_start;
        uint64_t end = overlap_end - entry_start;
        cuda_mirror_split_segment(entry, start);
        if (end < entry->len) {
            cuda_mirror_split_segment(entry, end);
        }

        for (uint32_t index = 0; index < entry->nr_segments; index++) {
            CudaMirrorSegment *segment = &entry->segments[index];
            bool needs_sync;

            if (segment->start >= end) {
                break;
            }
            if (segment->end <= start) {
                continue;
            }
            needs_sync = to_device
                ? segment->state == CUDA_RANGE_UNINITIALIZED ||
                      segment->state == CUDA_RANGE_HOST_DIRTY
                : segment->state == CUDA_RANGE_DEVICE_DIRTY;
            if (needs_sync && cuda_mirror_sync_segment(entry, segment,
                                                        to_device) != 0) {
                return -1;
            }
        }
        cuda_mirror_merge_segments(entry);
    }
    return 0;
}

static void cuda_mirror_mark_locked(SsdBackend *b, void *ptr, uint64_t len,
                                    CudaRangeState state)
{
    uintptr_t request_start = (uintptr_t)ptr;
    uintptr_t request_end = request_start + len;

    for (CudaMirrorRange *entry = cuda_mirror_head(b); entry;
         entry = entry->next) {
        uintptr_t entry_start = (uintptr_t)entry->host_ptr;
        uintptr_t entry_end = entry_start + entry->len;
        uintptr_t overlap_start = MAX(request_start, entry_start);
        uintptr_t overlap_end = MIN(request_end, entry_end);

        if (overlap_start < overlap_end) {
            cuda_mirror_set_state(entry, overlap_start - entry_start,
                                  overlap_end - entry_start, state);
        }
    }
}

static void *cuda_get_symbol(const char *name)
{
    return cuda_api.module ? dlsym(cuda_api.module, name) : NULL;
}

static void backend_cuda_close_module(void)
{
    if (!cuda_api.module) {
        return;
    }

    dlclose(cuda_api.module);
    cuda_api.module = NULL;
}

static int backend_cuda_load_module(void)
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

static int backend_cuda_init_api(void)
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

    if (!cuda_api.set_device || !cuda_api.malloc || !cuda_api.free ||
        !cuda_api.memcpy) {
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
    b->cuda_mirror = NULL;
    qemu_mutex_init(&b->cuda_mirror_lock);
    return 0;
}

void backend_cuda_sync_fini(SsdBackend *b)
{
    if (b && b->cuda_sync && cuda_api.free) {
        CudaMirrorRange *cur = cuda_mirror_head(b);
        while (cur) {
            CudaMirrorRange *next = cur->next;
            cuda_mirror_free(cur);
            cur = next;
        }
        cuda_mirror_set_head(b, NULL);
        qemu_mutex_destroy(&b->cuda_mirror_lock);
    }

    if(b)
    {
        b->cuda_sync = false;
    }
}

void *backend_cuda_ensure_device_ptr(SsdBackend *b, void *host_ptr, uint64_t len)
{
    uint8_t *host;
    uint64_t offset;
    femu_debug("backend_cuda_ensure_device_ptr: b %p, cuda_sync %d, host_ptr %p, len %lu\n",
             b, b ? b->cuda_sync : -1, host_ptr, len);

    if(!b || !b->cuda_sync || !host_ptr)
        return NULL;

    host = (uint8_t *)b->logical_space;
    if((uint8_t *)host_ptr < host)
    {
        femu_err("Pointer %p out of backend logical space range!\n", host_ptr);
        return NULL;
    }

    offset = (uint64_t)((uint8_t *)host_ptr - host);

    if(len == 0 ||
       offset >= (uint64_t)b->size ||
       len > (uint64_t)b->size - offset)
    {
        femu_err("Pointer %p with len %lu out of backend logical space range!\n", host_ptr, len);
        return NULL;
    }

    qemu_mutex_lock(&b->cuda_mirror_lock);
    CudaMirrorRange *entry = cuda_mirror_ensure_locked(b, host_ptr, len);
    qemu_mutex_unlock(&b->cuda_mirror_lock);
    if(!entry)
    {
        femu_err("Failed to ensure CUDA mirror for host %p len %lu\n", host_ptr, len);
        return NULL;
    }

    return entry->device_ptr;
}

int backend_cuda_drop_device_ptr(SsdBackend *b, void *host_ptr, uint64_t len)
{
    CudaMirrorRange *previous = NULL;
    CudaMirrorRange *entry;
    int rc;

    if (!b || !b->cuda_sync || !host_ptr || len == 0 || !cuda_api.memcpy) {
        return 0;
    }
    if (!backend_cuda_valid_range(b, host_ptr, len)) {
        return -1;
    }

    qemu_mutex_lock(&b->cuda_mirror_lock);
    entry = cuda_mirror_head(b);
    while (entry && (entry->host_ptr != host_ptr || entry->len != len)) {
        previous = entry;
        entry = entry->next;
    }
    if (!entry) {
        qemu_mutex_unlock(&b->cuda_mirror_lock);
        return 0;
    }

    rc = cuda_mirror_prepare_locked(b, host_ptr, len, false);
    if (rc == 0) {
        if (previous) {
            previous->next = entry->next;
        } else {
            cuda_mirror_set_head(b, entry->next);
        }
        cuda_mirror_free(entry);
        femu_debug("Dropped CUDA mirror for host %p len %lu\n", host_ptr, len);
    }
    qemu_mutex_unlock(&b->cuda_mirror_lock);
    return rc;
}

static bool backend_cuda_valid_range(SsdBackend *b, void *ptr, uint64_t len)
{
    uintptr_t host = (uintptr_t)b->logical_space;
    uintptr_t range = (uintptr_t)ptr;

    if (range < host || len > (uint64_t)b->size ||
        range - host > (uint64_t)b->size - len) {
        femu_err("Pointer %p with len %lu out of backend logical space range!\n",
                 ptr, len);
        return false;
    }
    return true;
}

int backend_cuda_prepare_device(SsdBackend *b, void *ptr, uint64_t len)
{
    int rc;

    if (!b || !b->cuda_sync || !ptr || len == 0 || !cuda_api.memcpy) {
        return 0;
    }
    if (!backend_cuda_valid_range(b, ptr, len)) {
        return -1;
    }

    qemu_mutex_lock(&b->cuda_mirror_lock);
    CudaMirrorRange *entry = cuda_mirror_ensure_locked(b, ptr, len);
    if (!entry) {
        qemu_mutex_unlock(&b->cuda_mirror_lock);
        return -1;
    }
    rc = cuda_mirror_prepare_locked(b, ptr, len, true);
    qemu_mutex_unlock(&b->cuda_mirror_lock);
    return rc;
}

int backend_cuda_prepare_host(SsdBackend *b, void *ptr, uint64_t len)
{
    int rc;

    if (!b || !b->cuda_sync || !ptr || len == 0 || !cuda_api.memcpy) {
        return 0;
    }
    if (!backend_cuda_valid_range(b, ptr, len)) {
        return -1;
    }

    qemu_mutex_lock(&b->cuda_mirror_lock);
    rc = cuda_mirror_prepare_locked(b, ptr, len, false);
    qemu_mutex_unlock(&b->cuda_mirror_lock);
    return rc;
}

void backend_cuda_mark_host_dirty(SsdBackend *b, void *ptr, uint64_t len)
{
    if (!b || !b->cuda_sync || !ptr || len == 0 ||
        !backend_cuda_valid_range(b, ptr, len)) {
        return;
    }

    qemu_mutex_lock(&b->cuda_mirror_lock);
    cuda_mirror_mark_locked(b, ptr, len, CUDA_RANGE_HOST_DIRTY);
    qemu_mutex_unlock(&b->cuda_mirror_lock);
}

void backend_cuda_mark_device_dirty(SsdBackend *b, void *ptr, uint64_t len)
{
    if (!b || !b->cuda_sync || !ptr || len == 0 ||
        !backend_cuda_valid_range(b, ptr, len)) {
        return;
    }

    qemu_mutex_lock(&b->cuda_mirror_lock);
    cuda_mirror_mark_locked(b, ptr, len, CUDA_RANGE_DEVICE_DIRTY);
    qemu_mutex_unlock(&b->cuda_mirror_lock);
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
        if (!is_write &&
            backend_cuda_prepare_host(b, mb + mb_oft, cur_len) != 0) {
            qemu_sglist_destroy(qsg);
            return -1;
        }
        if (dma_memory_rw(qsg->as, cur_addr, mb + mb_oft, cur_len, dir, MEMTXATTRS_UNSPECIFIED)) {
            femu_err("dma_memory_rw error\n");
        }
        if (is_write) {
            backend_cuda_mark_host_dirty(b, mb + mb_oft, cur_len);
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
    void *src, *dest;
    if (is_write) {
        src = buf;
        dest = b->logical_space + data_offset;
    } else {
        if (backend_cuda_prepare_host(b, b->logical_space + data_offset,
                                      data_size) != 0) {
            return;
        }
        src = b->logical_space + data_offset;
        dest = buf;
    }
    memmove(dest, src, data_size);

    if(is_write) {
        backend_cuda_mark_host_dirty(b, b->logical_space + data_offset,
                                     data_size);
    }
}

void *backend_get_ptr(SsdBackend *b, uint64_t offset)
{
    return b->logical_space + offset;
}

void backend_copy_internal(SsdBackend *b, uint64_t doff, uint64_t soff,
                           uint64_t data_size)
{
    if (backend_cuda_prepare_host(b, b->logical_space + soff,
                                  data_size) != 0) {
        return;
    }
    void *dest = b->logical_space + doff;
    memmove(dest, b->logical_space + soff, data_size);
    backend_cuda_mark_host_dirty(b, dest, data_size);
}

void backend_fill(SsdBackend *b, uint64_t offset, uint64_t len)
{
    void *dest = backend_addr(b, offset);

    memset(dest, 0, len);
    backend_cuda_mark_host_dirty(b, dest, len);
}
