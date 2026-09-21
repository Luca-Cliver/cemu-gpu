#ifndef CEMU_SPARF_GROUP_ABI_H
#define CEMU_SPARF_GROUP_ABI_H

#include "hw/femu/sparf_writeback_abi.h"

#define CEMU_SPARF_GROUP_COMMAND UINT64_C(0x43454d5553475250)
#define CEMU_SPARF_GROUP_VERSION 1U
#define CEMU_SPARF_GROUP_REGISTER 1U
#define CEMU_SPARF_GROUP_PUT 2U
#define CEMU_SPARF_GROUP_READ_TOKENS 3U
#define CEMU_SPARF_GROUP_READ_CHANNELS 4U
#define CEMU_SPARF_GROUP_CLOSE 5U
#define CEMU_SPARF_GROUP_REGISTER_BUFFER 6U
#define CEMU_SPARF_GROUP_APPEND 7U
#define CEMU_SPARF_GROUP_DRAIN 8U
#define CEMU_SPARF_GROUP_STATUS 9U
#define CEMU_SPARF_GROUP_APPEND_BATCH 10U
#define CEMU_SPARF_GROUP_STATS 11U
#define CEMU_SPARF_GROUP_TOKEN_K 0U
#define CEMU_SPARF_GROUP_CHANNEL_K 1U
#define CEMU_SPARF_GROUP_TOKEN_V 2U
#define CEMU_SPARF_GROUP_MAX_PAGES (1U << 24)
#define CEMU_SPARF_GROUP_MAX_EXTENTS (1U << 16)
#define CEMU_SPARF_GROUP_MAX_INDICES 4096U
#define CEMU_SPARF_GROUP_MAX_STREAMS (1U << 20)

struct cemu_sparf_group_statistics {
    uint64_t write_batches;
    uint64_t max_write_pages;
    uint64_t append_commands;
    uint64_t appended_streams;
    uint64_t appended_tokens;
};

struct cemu_sparf_group_progress {
    uint64_t accepted;
    uint64_t committed[3];
    uint64_t sealed;
    uint64_t failed;
    uint64_t write_bytes;
    uint64_t buffer_bytes;
};

struct cemu_sparf_group_header {
    uint32_t version;
    uint32_t operation;
    uint32_t kind;
    uint32_t head_dim;
    uint32_t channels_per_page;
    uint32_t element_bytes;
    uint32_t page_bytes;
    uint32_t count;
    uint64_t stream;
    uint64_t position;
    uint64_t argument;
    uint64_t cookie;
};

struct cemu_sparf_group_extent {
    uint64_t sector;
    uint64_t sectors;
};

struct cemu_sparf_group_result {
    struct cemu_sparf_writeback_result io;
    uint64_t version;
    uint64_t pages_used;
    uint64_t groups;
    uint64_t output_bytes;
};

#if defined(__cplusplus)
static_assert(sizeof(cemu_sparf_group_header) == 64, "group header ABI changed");
static_assert(sizeof(cemu_sparf_group_extent) == 16, "group extent ABI changed");
static_assert(sizeof(cemu_sparf_group_result) == 96, "group result ABI changed");
static_assert(sizeof(cemu_sparf_group_progress) == 64, "group progress ABI changed");
static_assert(sizeof(cemu_sparf_group_statistics) == 40, "group statistics ABI changed");
#else
_Static_assert(sizeof(struct cemu_sparf_group_header) == 64, "group header ABI changed");
_Static_assert(sizeof(struct cemu_sparf_group_extent) == 16, "group extent ABI changed");
_Static_assert(sizeof(struct cemu_sparf_group_result) == 96, "group result ABI changed");
_Static_assert(sizeof(struct cemu_sparf_group_progress) == 64, "group progress ABI changed");
_Static_assert(sizeof(struct cemu_sparf_group_statistics) == 40, "group statistics ABI changed");
#endif

#endif
