#ifndef CEMU_INSTATTENTION_MAPPING_H
#define CEMU_INSTATTENTION_MAPPING_H

#include <stddef.h>
#include "instattention_address.h"

typedef enum CemuKvStatus {
    CEMU_KV_OK,
    CEMU_KV_INVALID,
    CEMU_KV_NOT_FOUND,
    CEMU_KV_EXISTS,
    CEMU_KV_STALE,
    CEMU_KV_PENDING,
    CEMU_KV_IO_FAILED,
    CEMU_KV_NO_SPACE,
    CEMU_KV_LIMIT,
    CEMU_KV_NO_MEMORY
} CemuKvStatus;

typedef struct CemuKvMapConfig {
    uint32_t controller;
    uint32_t namespace_id;
    CemuKvNandGeometry nand;
    uint64_t backend_bytes;
    size_t max_tasks;
    size_t max_pages;
    size_t memory_budget_bytes;
} CemuKvMapConfig;

typedef struct CemuKvPageRef {
    CemuKvPageKey key;
    CemuKvPhysicalPage physical;
    uint64_t backend_offset;
    uint64_t ticket;
    uint32_t valid_tokens;
} CemuKvPageRef;

typedef struct CemuKvMapStats {
    size_t tasks;
    size_t reserved_pages;
    size_t written_pages;
    size_t failed_pages;
    uint64_t claimed_blocks;
    size_t memory_bytes;
} CemuKvMapStats;

typedef struct CemuKvMap CemuKvMap;

CemuKvStatus cemu_kv_map_create(const CemuKvMapConfig *config, CemuKvMap **result);
void cemu_kv_map_destroy(CemuKvMap *map);
CemuKvStatus cemu_kv_map_register(CemuKvMap *map,
                                const CemuKvIdentity *identity,
                                const CemuKvGeometry *geometry);
CemuKvStatus cemu_kv_map_reserve(CemuKvMap *map,
                               const CemuKvIdentity *identity,
                               const CemuKvCoordinate *page_start,
                               uint32_t valid_tokens, CemuKvPageRef *result);
CemuKvStatus cemu_kv_map_complete(CemuKvMap *map, const CemuKvPageRef *reference,
                                bool io_success, uint64_t ready_ns);
CemuKvStatus cemu_kv_map_query_written(const CemuKvMap *map,
                                     const CemuKvIdentity *identity,
                                     const CemuKvCoordinate *coordinate,
                                     CemuKvPageRef *result,
                                     uint32_t *byte_offset, uint64_t *ready_ns);
CemuKvStatus cemu_kv_map_lookup(const CemuKvMap *map,
                              const CemuKvIdentity *identity,
                              const CemuKvCoordinate *coordinate,
                              uint64_t now_ns, CemuKvPageRef *result,
                              uint32_t *byte_offset);
CemuKvMapStats cemu_kv_map_stats(const CemuKvMap *map);

#endif
