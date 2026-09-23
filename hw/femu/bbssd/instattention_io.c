#include "instattention_io.h"
#include <stdlib.h>
#include <string.h>

struct CemuKvIo {
    CemuKvMap *map;
    CemuKvIoOps ops;
    void *opaque;
    void *scratch;
    uint32_t page_bytes;
};

CemuKvStatus cemu_kv_io_create(const CemuKvMapConfig *config,
                             const CemuKvIoOps *ops, void *opaque,
                             CemuKvIo **result)
{
    CemuKvMapConfig bounded;
    CemuKvIo *io;
    CemuKvStatus status;
    size_t overhead;

    if (!config || !ops || !ops->transfer || !ops->schedule || !result ||
        !cemu_kv_page_bytes_valid(config->nand.page_bytes)) {
        return CEMU_KV_INVALID;
    }
    overhead = sizeof(*io) + config->nand.page_bytes;
    if (config->memory_budget_bytes < overhead) {
        return CEMU_KV_LIMIT;
    }
    bounded = *config;
    bounded.memory_budget_bytes -= overhead;
    io = calloc(1, sizeof(*io));
    if (!io) {
        return CEMU_KV_NO_MEMORY;
    }
    status = cemu_kv_map_create(&bounded, &io->map);
    if (status != CEMU_KV_OK) {
        free(io);
        return status;
    }
    io->scratch = malloc(config->nand.page_bytes);
    if (!io->scratch) {
        cemu_kv_io_destroy(io);
        return CEMU_KV_NO_MEMORY;
    }
    io->ops = *ops;
    io->opaque = opaque;
    io->page_bytes = config->nand.page_bytes;
    *result = io;
    return CEMU_KV_OK;
}

void cemu_kv_io_destroy(CemuKvIo *io)
{
    if (io) {
        cemu_kv_map_destroy(io->map);
        free(io->scratch);
        free(io);
    }
}

CemuKvStatus cemu_kv_io_register(CemuKvIo *io, const CemuKvIdentity *identity,
                               const CemuKvGeometry *geometry)
{
    return io ? cemu_kv_map_register(io->map, identity, geometry) : CEMU_KV_INVALID;
}

static bool schedule_page(CemuKvIo *io, bool write, CemuKvIoResult *result)
{
    return io->ops.schedule(io->opaque, &result->page.physical, write,
                            result->start_ns, &result->ready_ns) &&
           result->ready_ns >= result->start_ns && result->ready_ns <= INT64_MAX;
}

CemuKvStatus cemu_kv_io_write(CemuKvIo *io, const CemuKvIdentity *identity,
                            const CemuKvCoordinate *page_start,
                            uint32_t valid_tokens, const void *buffer,
                            size_t bytes, uint64_t start_ns,
                            CemuKvIoResult *result)
{
    CemuKvIoResult candidate = { .start_ns = start_ns };
    CemuKvStatus status;
    bool success;

    if (!io || !buffer || !result || bytes != io->page_bytes || start_ns > INT64_MAX) {
        return CEMU_KV_INVALID;
    }
    status = cemu_kv_map_reserve(io->map, identity, page_start, valid_tokens,
                                &candidate.page);
    if (status != CEMU_KV_OK) {
        return status;
    }
    memcpy(io->scratch, buffer, bytes);
    success = schedule_page(io, true, &candidate);
    if (success) {
        success = io->ops.transfer(io->opaque, io->scratch,
                                   candidate.page.backend_offset, bytes, true) == 0;
    }
    status = cemu_kv_map_complete(io->map, &candidate.page, success,
                                  success ? candidate.ready_ns : 0);
    if (status != CEMU_KV_OK) {
        return status;
    }
    if (!success) {
        return CEMU_KV_IO_FAILED;
    }
    candidate.transferred_bytes = bytes;
    *result = candidate;
    return CEMU_KV_OK;
}

CemuKvStatus cemu_kv_io_read(CemuKvIo *io, const CemuKvIdentity *identity,
                           const CemuKvCoordinate *coordinate, void *buffer,
                           size_t bytes, uint64_t start_ns,
                           CemuKvIoResult *result)
{
    CemuKvIoResult candidate = { .start_ns = start_ns };
    CemuKvStatus status;
    uint64_t written_ready_ns;

    if (!io || !buffer || !result || bytes != io->page_bytes || start_ns > INT64_MAX) {
        return CEMU_KV_INVALID;
    }
    status = cemu_kv_map_query_written(io->map, identity, coordinate,
                                      &candidate.page, &candidate.element_offset,
                                      &written_ready_ns);
    if (status != CEMU_KV_OK) {
        return status;
    }
    if (written_ready_ns > candidate.start_ns) {
        candidate.start_ns = written_ready_ns;
    }
    if (!schedule_page(io, false, &candidate) ||
        io->ops.transfer(io->opaque, io->scratch, candidate.page.backend_offset,
                          bytes, false) != 0) {
        return CEMU_KV_IO_FAILED;
    }
    memcpy(buffer, io->scratch, bytes);
    candidate.transferred_bytes = bytes;
    *result = candidate;
    return CEMU_KV_OK;
}

CemuKvMapStats cemu_kv_io_stats(const CemuKvIo *io)
{
    CemuKvMapStats stats = cemu_kv_map_stats(io ? io->map : NULL);

    if (io) {
        stats.memory_bytes += sizeof(*io) + io->page_bytes;
    }
    return stats;
}
