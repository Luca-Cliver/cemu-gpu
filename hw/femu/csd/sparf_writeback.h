#ifndef CEMU_SPARF_WRITEBACK_H
#define CEMU_SPARF_WRITEBACK_H

#include <glib.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "hw/femu/sparf_writeback_abi.h"
#include "hw/femu/kv_page.h"

typedef struct SparfWritebackPlan {
    bool borrowed_image;
    GArray *sectors;
    GArray *spans;
    uint8_t *image;
    uint64_t copied_bytes;
    GArray *kv_pages;
    GArray *kv_page_ordinals;
    bool profile;
    struct cemu_sparf_writeback_diagnostics diagnostics;
} SparfWritebackPlan;

static uint64_t sparf_wb_host_ns(void)
{
    struct timespec timestamp;
    clock_gettime(CLOCK_MONOTONIC, &timestamp);
    return (uint64_t)timestamp.tv_sec * UINT64_C(1000000000) + timestamp.tv_nsec;
}

static int sparf_wb_compare(const void *left, const void *right)
{
    uint64_t first = *(const uint64_t *)left;
    uint64_t second = *(const uint64_t *)right;
    return (first > second) - (first < second);
}

static inline guint sparf_wb_extent_sectors(const GArray *sectors, guint index,
                                           uint32_t sectors_per_page)
{
    if (!sectors_per_page || sectors_per_page > UINT16_MAX || index >= sectors->len) return 0;
    uint64_t first = g_array_index(sectors, uint64_t, index);
    guint limit = UINT16_MAX - (first % sectors_per_page + UINT16_MAX) % sectors_per_page;
    guint count = 1;
    while (count < limit && count < sectors->len - index &&
           g_array_index(sectors, uint64_t, index + count) == first + count) ++count;
    return count;
}

static void sparf_wb_free(SparfWritebackPlan *plan)
{
    if (!plan) return;
    if (plan->sectors) g_array_free(plan->sectors, TRUE);
    if (plan->spans) g_array_free(plan->spans, TRUE);
    if (plan->kv_pages) g_array_free(plan->kv_pages, TRUE);
    if (plan->kv_page_ordinals) g_array_free(plan->kv_page_ordinals, TRUE);
    if (!plan->borrowed_image) g_free(plan->image);
    g_free(plan);
}

static SparfWritebackPlan *sparf_wb_plan(const void *metadata, uint32_t length,
                                       uint64_t source_bytes, uint64_t nvm_bytes)
{
    struct cemu_sparf_writeback_header header;
    if (!metadata || length < sizeof(header)) return NULL;
    memcpy(&header, metadata, sizeof(header));
    bool matrix = header.version == CEMU_SPARF_WRITEBACK_MATRIX_VERSION;
    size_t descriptor_size = matrix ? sizeof(struct cemu_sparf_writeback_matrix)
                                    : sizeof(struct cemu_sparf_writeback_span);
    if ((!matrix && header.version != CEMU_SPARF_WRITEBACK_VERSION) ||
        (header.reserved & ~CEMU_SPARF_WRITEBACK_PROFILE) ||
        !header.spans || header.spans > CEMU_SPARF_WRITEBACK_MAX_SPANS ||
        length != sizeof(header) + (uint64_t)header.spans * descriptor_size) return NULL;
    SparfWritebackPlan *plan = g_new0(SparfWritebackPlan, 1);
    plan->profile = (header.reserved & CEMU_SPARF_WRITEBACK_PROFILE) != 0;
    uint64_t started = plan->profile ? sparf_wb_host_ns() : 0;
    plan->sectors = g_array_new(FALSE, FALSE, sizeof(uint64_t));
    plan->spans = g_array_new(FALSE, FALSE, sizeof(struct cemu_sparf_writeback_span));
    for (uint32_t index = 0; index < header.spans; ++index) {
        struct cemu_sparf_writeback_matrix descriptor = { .rows = 1 };
        memcpy(&descriptor, (const uint8_t *)metadata + sizeof(header) + index * descriptor_size, descriptor_size);
        struct cemu_sparf_writeback_span span = descriptor.span;
        if (!descriptor.rows || descriptor.reserved ||
            descriptor.rows > CEMU_SPARF_WRITEBACK_MAX_SPANS - plan->spans->len) goto invalid;
        if (!span.bytes || !span.count || span.source_stride < span.bytes ||
            span.destination_stride < span.bytes) goto invalid;
        uint64_t source_size = (uint64_t)(span.count - 1) * span.source_stride + span.bytes;
        uint64_t destination_size = (uint64_t)(span.count - 1) * span.destination_stride + span.bytes;
        uint64_t copied = (uint64_t)span.count * span.bytes;
        if (span.source > source_bytes || source_size > source_bytes - span.source ||
            span.destination > nvm_bytes || destination_size > nvm_bytes - span.destination ||
            (uint64_t)(descriptor.rows - 1) * descriptor.source_row_stride > source_bytes - span.source - source_size ||
            (uint64_t)(descriptor.rows - 1) * descriptor.destination_row_stride > nvm_bytes - span.destination - destination_size ||
            copied > (CEMU_SPARF_WRITEBACK_MAX_BYTES - plan->copied_bytes) / descriptor.rows) goto invalid;
        plan->copied_bytes += copied * descriptor.rows;
        for (uint32_t row = 0; row < descriptor.rows; ++row) {
            struct cemu_sparf_writeback_span expanded = span;
            expanded.source += (uint64_t)row * descriptor.source_row_stride;
            expanded.destination += (uint64_t)row * descriptor.destination_row_stride;
            g_array_append_val(plan->spans, expanded);
        }
    }
    for (guint index = 0; index < plan->spans->len; ++index) {
        struct cemu_sparf_writeback_span span = g_array_index(plan->spans, struct cemu_sparf_writeback_span, index);
        for (uint32_t item = 0; item < span.count; ++item) {
            uint64_t offset = span.destination + (uint64_t)item * span.destination_stride;
            uint64_t first = offset / 512, last = (offset + span.bytes - 1) / 512;
            if (last - first + 1 > CEMU_SPARF_WRITEBACK_MAX_PIECES - plan->sectors->len) goto invalid;
            for (uint64_t sector = first; sector <= last; ++sector) g_array_append_val(plan->sectors, sector);
        }
    }
    g_array_sort(plan->sectors, sparf_wb_compare);
    guint unique = 0;
    for (guint index = 0; index < plan->sectors->len; ++index) {
        uint64_t sector = g_array_index(plan->sectors, uint64_t, index);
        if (!unique || sector != g_array_index(plan->sectors, uint64_t, unique - 1)) {
            g_array_index(plan->sectors, uint64_t, unique++) = sector;
        }
    }
    g_array_set_size(plan->sectors, unique);
    if ((uint64_t)unique * 512 > CEMU_SPARF_WRITEBACK_MAX_BYTES) goto invalid;
    plan->image = g_try_malloc((size_t)unique * 512);
    if (!plan->image) goto invalid;
    if (plan->profile) {
        plan->diagnostics.version = 1;
        plan->diagnostics.plan_host_ns = sparf_wb_host_ns() - started;
        plan->diagnostics.metadata_bytes = length;
        plan->diagnostics.expanded_spans = plan->spans->len;
    }
    return plan;
invalid:
    sparf_wb_free(plan);
    return NULL;
}

static void sparf_wb_patch(SparfWritebackPlan *plan, const void *metadata, const uint8_t *source)
{
    (void)metadata;
    for (guint index = 0; index < plan->spans->len; ++index) {
        struct cemu_sparf_writeback_span span = g_array_index(plan->spans, struct cemu_sparf_writeback_span, index);
        for (uint32_t item = 0; item < span.count; ++item) {
            uint64_t destination = span.destination + (uint64_t)item * span.destination_stride;
            uint64_t input = span.source + (uint64_t)item * span.source_stride;
            uint32_t remaining = span.bytes;
            while (remaining) {
                uint64_t sector = destination / 512;
                uint64_t *entry = bsearch(&sector, plan->sectors->data, plan->sectors->len,
                                           sizeof(sector), sparf_wb_compare);
                size_t offset = (entry - (uint64_t *)plan->sectors->data) * 512 + destination % 512;
                uint32_t count = MIN(remaining, 512 - destination % 512);
                memcpy(plan->image + offset, source + input, count);
                destination += count;
                input += count;
                remaining -= count;
            }
        }
    }
}

typedef bool (*SparfWritebackTransfer)(void *, SparfWritebackPlan *, bool, uint64_t *, uint64_t *);

static bool sparf_wb_run(SparfWritebackPlan *plan, const void *metadata, const uint8_t *source,
                         SparfWritebackTransfer transfer, void *context,
                         struct cemu_sparf_writeback_result *result)
{
    memset(result, 0, sizeof(*result));
    if (!transfer(context, plan, false, &result->read_start, &result->read_end)) return false;
    result->read_bytes = (uint64_t)plan->sectors->len * 512;
    uint64_t started = plan->profile ? sparf_wb_host_ns() : 0;
    sparf_wb_patch(plan, metadata, source);
    if (plan->profile) plan->diagnostics.patch_host_ns = sparf_wb_host_ns() - started;
    if (!transfer(context, plan, true, &result->write_start, &result->write_end)) return false;
    result->write_bytes = result->read_bytes;
    result->copied_bytes = plan->copied_bytes;
    return true;
}

#endif
