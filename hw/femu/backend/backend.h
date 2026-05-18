#ifndef __FEMU_MEM_BACKEND
#define __FEMU_MEM_BACKEND

#include <stdint.h>
#include <stdbool.h>
#include "qemu/osdep.h"

typedef struct QEMUSGList QEMUSGList;

typedef enum BackendType{
    FEMU_DRAM_BACKEND,
} BackendType;

/* SSD Backend */
typedef struct SsdBackend {
    void        *logical_space;
    void        *cuda_mirror;
    int64_t     size; /* in bytes */
    int         femu_mode;
    BackendType type;
    bool        cuda_sync;
    bool        cuda_mirror_valid;
} SsdBackend;

int init_dram_backend(SsdBackend *b);
void free_dram_backend(SsdBackend *b);

int backend_cuda_sync_init(SsdBackend *b);
void backend_cuda_sync_fini(SsdBackend *b);
void backend_cuda_sync_ptr(SsdBackend *b, void *ptr, uint64_t len,
                           bool to_device);

/* Map a host pointer returned by backend_addr() to the corresponding
 * device (GPU) pointer in the backend mirror, or NULL if not available.
 */
void *backend_host_to_device(SsdBackend *b, void *host_ptr, uint64_t len);

int init_backend(SsdBackend **b, BackendType type, char *path, int64_t nbytes);
void free_backend(SsdBackend *b);

int backend_rw(SsdBackend *b, QEMUSGList *qsg, uint64_t *lbal, bool is_write);
void backend_rw_internal(SsdBackend *b, void *buf, uint64_t data_offset,
                           uint64_t data_size, int is_write);
void *backend_get_ptr(SsdBackend *b, uint64_t offset);
void backend_copy_internal(SsdBackend *b, uint64_t doff, uint64_t soff,
                           uint64_t data_size);
static inline void *backend_addr(SsdBackend *b, uint64_t offset)
{
    return b->logical_space + offset;
}
static inline void backend_fill(SsdBackend *b, uint64_t offset, uint64_t len)
{
    memset(backend_addr(b, offset), 0, len);
}

#endif
