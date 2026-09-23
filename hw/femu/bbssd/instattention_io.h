#ifndef CEMU_INSTATTENTION_IO_H
#define CEMU_INSTATTENTION_IO_H

#include "instattention_mapping.h"

typedef struct CemuKvIoOps {
    int (*transfer)(void *opaque, void *buffer, uint64_t offset,
                    size_t bytes, bool write);
    bool (*schedule)(void *opaque, const CemuKvPhysicalPage *page, bool write,
                     uint64_t start_ns, uint64_t *ready_ns);
} CemuKvIoOps;

typedef struct CemuKvIoResult {
    CemuKvPageRef page;
    uint64_t start_ns;
    uint64_t ready_ns;
    uint32_t element_offset;
    uint32_t transferred_bytes;
} CemuKvIoResult;

typedef struct CemuKvIo CemuKvIo;

CemuKvStatus cemu_kv_io_create(const CemuKvMapConfig *config,
                             const CemuKvIoOps *ops, void *opaque,
                             CemuKvIo **result);
void cemu_kv_io_destroy(CemuKvIo *io);
CemuKvStatus cemu_kv_io_register(CemuKvIo *io, const CemuKvIdentity *identity,
                               const CemuKvGeometry *geometry);
CemuKvStatus cemu_kv_io_write(CemuKvIo *io, const CemuKvIdentity *identity,
                            const CemuKvCoordinate *page_start,
                            uint32_t valid_tokens, const void *buffer,
                            size_t bytes, uint64_t start_ns,
                            CemuKvIoResult *result);
CemuKvStatus cemu_kv_io_read(CemuKvIo *io, const CemuKvIdentity *identity,
                           const CemuKvCoordinate *coordinate, void *buffer,
                           size_t bytes, uint64_t start_ns,
                           CemuKvIoResult *result);
CemuKvMapStats cemu_kv_io_stats(const CemuKvIo *io);

#endif
