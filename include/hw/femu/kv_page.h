#ifndef CEMU_FEMU_KV_PAGE_H
#define CEMU_FEMU_KV_PAGE_H

#include <stdint.h>

enum CemuKvPageKind {
    CEMU_KV_PAGE_TOKEN_K = 0,
    CEMU_KV_PAGE_CHANNEL_K = 1,
    CEMU_KV_PAGE_TOKEN_V = 2,
};

typedef struct CemuKvPageTag {
    uint64_t slba;
    uint64_t cookie;
    uint64_t stream;
    uint64_t group;
    uint32_t kind;
    uint32_t page_in_group;
} CemuKvPageTag;

#endif
