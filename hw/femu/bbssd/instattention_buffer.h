#ifndef CEMU_INSTATTENTION_BUFFER_H
#define CEMU_INSTATTENTION_BUFFER_H

#include "instattention_io.h"
#include "hw/femu/read_observation.h"

struct NvmeNamespace;
typedef struct NativeKvBuffer NativeKvBuffer;

typedef struct NativeKvBufferLease {
    NativeKvBuffer *buffer;
    CemuKvGeometry geometry;
    uint32_t batch_first;
    uint32_t batches;
    uint32_t layer;
    uint64_t tokens;
} NativeKvBufferLease;

#define CEMU_NATIVE_GATHER_MAX_RANGES 256

typedef struct NativeKvGatherRange {
    CemuKvCoordinate coordinate;
    uint32_t tokens;
} NativeKvGatherRange;

typedef void (*NativeKvGatherSource)(void *opaque, uint32_t index,
                                    NativeKvGatherRange *range);

typedef struct NativeKvGatherResult {
    uint64_t ready_ns;
    uint64_t transferred_bytes;
    uint64_t output_bytes;
    uint64_t tail_bytes;
    uint32_t flash_pages;
    uint32_t batches;
} NativeKvGatherResult;

typedef struct NativeKvBufferStats {
    uint64_t accepted;
    uint64_t committed[3];
    uint64_t memory_bytes;
    uint64_t write_bytes;
    uint64_t read_bytes;
    uint64_t write_batches;
    uint64_t pending_ready_ns;
    uint64_t packed[3]; /* ABI compatibility: equals accepted for each directly filled layout. */
    uint64_t staged_pages; /* Includes partial pages and pages waiting for NAND completion. */
    uint64_t write_blocks;
    uint64_t ready_blocks;
    uint64_t partial_blocks; /* Explicit terminal drain only. */
    uint64_t inflight_blocks;
    uint64_t peak_inflight_blocks;
    bool sealed;
    bool failed;
} NativeKvBufferStats;

CemuKvStatus native_kv_buffer_create(struct NvmeNamespace *ns,
    const CemuKvIdentity *identity, const CemuKvGeometry *geometry,
    uint64_t budget, uint32_t batch_pages, NativeKvBuffer **result);
bool native_kv_buffer_matches(const NativeKvBuffer *buffer, const CemuKvIdentity *identity);
size_t native_kv_buffer_row_bytes(const NativeKvBuffer *buffer, bool channel);
CemuKvStatus native_kv_buffer_append(NativeKvBuffer *buffer,
    const CemuKvCoordinate *coordinate, uint32_t count, const void *key,
    const void *value, size_t bytes);
CemuKvStatus native_kv_buffer_append_batch(NativeKvBuffer *buffer,
    const CemuKvCoordinate *coordinate, uint32_t heads, uint32_t count,
    const void *key, const void *value, size_t bytes);
CemuKvStatus native_kv_buffer_read(NativeKvBuffer *buffer,
    const CemuKvCoordinate *coordinate, void *output, size_t bytes,
    uint64_t start_ns, CemuKvIoResult *result);
CemuKvStatus native_kv_buffer_gather(NativeKvBuffer *buffer,
    const NativeKvGatherRange *ranges, uint32_t count, void *output,
    size_t bytes, uint64_t start_ns, NativeKvGatherResult *result);
CemuKvStatus native_kv_buffer_gather_phase(NativeKvBuffer *buffer,
    NativeKvGatherSource source, void *opaque, uint32_t count, void *output,
    size_t bytes, uint64_t start_ns, NativeKvGatherResult *result,
    const NativeKvBufferLease *lease, uint32_t kind, FemuReadObservation *observation);
CemuKvStatus native_kv_buffer_stats(NativeKvBuffer *buffer,
    const CemuKvCoordinate *coordinate, NativeKvBufferStats *stats);
CemuKvStatus native_kv_buffer_drain(NativeKvBuffer *buffer);
CemuKvStatus native_kv_buffer_acquire(NativeKvBuffer *buffer,
    const CemuKvIdentity *identity, uint32_t batch_first, uint32_t batches,
    uint32_t layer, uint64_t tokens, NativeKvBufferLease *lease);
void native_kv_buffer_release(NativeKvBufferLease *lease);
void native_kv_buffer_stop(NativeKvBuffer *buffer);
void native_kv_buffer_free(NativeKvBuffer *buffer);

#endif
