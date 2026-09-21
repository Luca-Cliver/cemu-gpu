#ifndef CEMU_SPARF_WRITEBACK_ABI_H
#define CEMU_SPARF_WRITEBACK_ABI_H

#include <stdint.h>

#define CEMU_SPARF_WRITEBACK_COMMAND UINT64_C(0x43454d555357424b)
#define CEMU_SPARF_WRITEBACK_VERSION 1U
#define CEMU_SPARF_WRITEBACK_MATRIX_VERSION 2U
#define CEMU_SPARF_WRITEBACK_PROFILE UINT64_C(1)
#define CEMU_SPARF_WRITEBACK_MAX_SPANS 65536U
#define CEMU_SPARF_WRITEBACK_MAX_BYTES (64U * 1024U * 1024U)
#define CEMU_SPARF_WRITEBACK_MAX_PIECES 1048576U

struct cemu_sparf_writeback_header {
    uint32_t version;
    uint32_t spans;
    uint64_t reserved;
};

struct cemu_sparf_writeback_span {
    uint64_t destination;
    uint64_t source;
    uint32_t bytes;
    uint32_t count;
    uint32_t source_stride;
    uint32_t destination_stride;
};

struct cemu_sparf_writeback_result {
    uint64_t read_bytes;
    uint64_t write_bytes;
    uint64_t read_start;
    uint64_t read_end;
    uint64_t write_start;
    uint64_t write_end;
    uint64_t ready;
    uint64_t copied_bytes;
};

struct cemu_sparf_writeback_matrix {
    struct cemu_sparf_writeback_span span;
    uint32_t rows;
    uint32_t source_row_stride;
    uint32_t destination_row_stride;
    uint32_t reserved;
};

struct cemu_sparf_writeback_diagnostics {
    uint64_t version;
    uint64_t plan_host_ns;
    uint64_t queue_host_ns;
    uint64_t prepare_host_ns;
    uint64_t read_service_host_ns;
    uint64_t read_target_wait_host_ns;
    uint64_t patch_host_ns;
    uint64_t write_service_host_ns;
    uint64_t write_target_wait_host_ns;
    uint64_t metadata_bytes;
    uint64_t expanded_spans;
};

#if defined(__cplusplus)
static_assert(sizeof(cemu_sparf_writeback_header) == 16, "writeback header ABI changed");
static_assert(sizeof(cemu_sparf_writeback_span) == 32, "writeback span ABI changed");
static_assert(sizeof(cemu_sparf_writeback_result) == 64, "writeback result ABI changed");
static_assert(sizeof(cemu_sparf_writeback_matrix) == 48, "writeback matrix ABI changed");
static_assert(sizeof(cemu_sparf_writeback_diagnostics) == 88, "writeback diagnostics ABI changed");
#else
_Static_assert(sizeof(struct cemu_sparf_writeback_header) == 16, "writeback header ABI changed");
_Static_assert(sizeof(struct cemu_sparf_writeback_span) == 32, "writeback span ABI changed");
_Static_assert(sizeof(struct cemu_sparf_writeback_result) == 64, "writeback result ABI changed");
_Static_assert(sizeof(struct cemu_sparf_writeback_matrix) == 48, "writeback matrix ABI changed");
_Static_assert(sizeof(struct cemu_sparf_writeback_diagnostics) == 88, "writeback diagnostics ABI changed");
#endif

#endif
