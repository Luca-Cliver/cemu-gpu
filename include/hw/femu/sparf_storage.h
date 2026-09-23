#ifndef CEMU_SPARF_STORAGE_H
#define CEMU_SPARF_STORAGE_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include "sparf_workflow_abi.h"

typedef struct CemuSparfStorageResult {
    uint64_t ready_ns;
    uint64_t flash_bytes;
    uint64_t tail_bytes;
    uint64_t output_bytes;
    uint64_t pages;
    uint64_t batches;
    uint32_t status;
} CemuSparfStorageResult;

typedef bool (*CemuSparfStorageRead)(void *opaque,
    const struct cemu_sparf_workflow_header *header, const uint32_t *indices,
    uint32_t kind, void *output, size_t bytes, uint64_t start_ns,
    CemuSparfStorageResult *result);

#endif
