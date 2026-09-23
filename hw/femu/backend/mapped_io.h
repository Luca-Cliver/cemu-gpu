#ifndef CEMU_MAPPED_IO_H
#define CEMU_MAPPED_IO_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>
#include <errno.h>

typedef struct CemuMappedIoOps {
    int (*prepare_host)(void *opaque, void *pointer, uint64_t bytes);
    void (*mark_host_dirty)(void *opaque, void *pointer, uint64_t bytes);
} CemuMappedIoOps;

static inline int cemu_mapped_io(void *space, uint64_t capacity, void *buffer,
                                 uint64_t offset, uint64_t bytes, bool write,
                                 const CemuMappedIoOps *ops, void *opaque)
{
    uint8_t *address;

    if (!space || !buffer || capacity > SIZE_MAX || offset > capacity ||
        bytes > capacity - offset || !ops || !ops->prepare_host ||
        !ops->mark_host_dirty) {
        return -EINVAL;
    }
    if (!bytes) {
        return 0;
    }
    address = (uint8_t *)space + offset;
    if (!write && ops->prepare_host(opaque, address, bytes) != 0) {
        return -EIO;
    }
    memmove(write ? address : buffer, write ? buffer : address, bytes);
    if (write) {
        ops->mark_host_dirty(opaque, address, bytes);
    }
    return 0;
}

#endif
