#ifndef CEMU_FDM_KV_BUFFER_H
#define CEMU_FDM_KV_BUFFER_H

#include <stdbool.h>
#include "fdm_kv_buffer_abi.h"

typedef struct FDMKV_InitParams
{
    FDMKV_Identity_t identity;
    FDMKV_Geometry_t geometry;

    void *meta_addr;
    uint64_t meta_bytes;

    void *data_addr;
    uint64_t data_bytes;
    const FDMKV_DataSegment_t *data_segments;
    uint32_t segment_count;

    uint64_t input_bytes;

    uint32_t nand_channels;
    uint32_t block_pages;
    uint32_t batch_pages;
} FDMKV_InitParams_t;

bool fdm_kv_buffer_init(const FDMKV_InitParams_t *params, FDMKV_Buffer_t **result);


#endif
