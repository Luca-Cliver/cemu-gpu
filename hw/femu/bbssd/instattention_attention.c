#include "qemu/osdep.h"
#include "qemu/units.h"
#include <math.h>
#include "instattention_attention.h"

CemuKvStatus native_kv_attention_open(NativeKvBuffer *buffer,
    const CemuKvIdentity *identity, uint32_t batch_first,
    const struct cemu_sparf_workflow_header *header, NativeKvAttention *attention)
{
    CemuKvStatus status;
    uint32_t element;

    if (!header || !attention || attention->lease.buffer ||
        header->version != CEMU_SPARF_AUTONOMOUS_VERSION ||
        !(header->flags & CEMU_SPARF_WORKFLOW_GROUPED) ||
        (header->flags & ~(CEMU_SPARF_WORKFLOW_GROUPED | CEMU_SPARF_WORKFLOW_TRACE | CEMU_SPARF_WORKFLOW_NATIVE)) ||
        header->token_k_extents || header->channel_k_extents || header->token_v_extents ||
        header->token_head_stride || header->token_batch_stride || header->token_layer_stride ||
        header->channel_stride || header->channel_head_stride || header->channel_batch_stride ||
        header->channel_layer_stride || header->reserved[0] || header->reserved[1] ||
        !header->batch_size || !header->num_query_heads || !header->num_kv_heads ||
        header->num_query_heads % header->num_kv_heads ||
        (uint64_t)header->batch_size * header->num_query_heads > 65536 ||
        !header->head_dim || header->head_dim > 4096 || !header->max_seq_len ||
        header->max_seq_len > 65536 || !header->valid_tokens ||
        header->valid_tokens > header->max_seq_len ||
        !header->top_r || header->top_r > header->head_dim ||
        !header->top_k || header->top_k > header->valid_tokens ||
        (header->dtype != CEMU_SPARF_FLOAT16 && header->dtype != CEMU_SPARF_FLOAT32) ||
        !isfinite(header->scale) || header->scale <= 0) {
        return CEMU_KV_INVALID;
    }
    status = native_kv_buffer_acquire(buffer, identity, batch_first, header->batch_size,
        header->layer, header->valid_tokens, &attention->lease);
    if (status != CEMU_KV_OK) {
        return status;
    }
    element = header->dtype == CEMU_SPARF_FLOAT16 ? 2 : 4;
    if (attention->lease.geometry.heads != header->num_kv_heads ||
        attention->lease.geometry.head_dim != header->head_dim ||
        attention->lease.geometry.element_bytes != element ||
        attention->lease.geometry.max_tokens < header->max_seq_len) {
        native_kv_buffer_release(&attention->lease);
        return CEMU_KV_INVALID;
    }
    attention->header = *header;
    return CEMU_KV_OK;
}

static uint64_t range_bytes(const NativeKvAttention *attention, const NativeKvGatherRange *range)
{
    return (uint64_t)range->tokens * attention->lease.geometry.element_bytes *
        (range->coordinate.kind == CEMU_KV_PAGE_CHANNEL_K ? 1 : attention->header.head_dim);
}

static CemuKvStatus read_ranges(NativeKvAttention *attention,
    const NativeKvGatherRange *ranges, uint32_t count, uint8_t *output,
    uint64_t start_ns, CemuSparfStorageResult *result)
{
    uint64_t bytes = 0, prefix = 0;
    NativeKvGatherResult gathered = {0};
    CemuKvStatus status;

    for (uint32_t index = 0; index < count; index++) {
        bytes += range_bytes(attention, &ranges[index]);
    }
    status = native_kv_buffer_gather(attention->lease.buffer, ranges, count, output,
                                    bytes, start_ns, &gathered);
    if (status != CEMU_KV_LIMIT) {
        result->ready_ns = MAX(result->ready_ns, gathered.ready_ns);
        result->flash_bytes += gathered.transferred_bytes;
        result->pages += gathered.flash_pages;
        result->tail_bytes += gathered.tail_bytes;
        result->batches++;
        return status;
    }
    if (count > 1) {
        uint32_t first = count / 2;

        for (uint32_t index = 0; index < first; index++) {
            prefix += range_bytes(attention, &ranges[index]);
        }
        status = read_ranges(attention, ranges, first, output, start_ns, result);
        return status == CEMU_KV_OK ? read_ranges(attention, ranges + first,
            count - first, output + prefix, start_ns, result) : status;
    }
    if (ranges[0].tokens > 1) {
        NativeKvGatherRange split[2] = { ranges[0], ranges[0] };

        split[0].tokens /= 2;
        split[1].tokens -= split[0].tokens;
        split[1].coordinate.token += split[0].tokens;
        prefix = range_bytes(attention, &split[0]);
        status = read_ranges(attention, split, 1, output, start_ns, result);
        return status == CEMU_KV_OK ? read_ranges(attention, split + 1, 1,
            output + prefix, start_ns, result) : status;
    }
    return status;
}

bool native_kv_attention_read(void *opaque,
    const struct cemu_sparf_workflow_header *header, const uint32_t *indices,
    uint32_t kind, void *output, size_t bytes, uint64_t start_ns,
    CemuSparfStorageResult *result)
{
    NativeKvAttention *attention = opaque;
    NativeKvGatherRange ranges[CEMU_NATIVE_GATHER_MAX_RANGES];
    uint64_t vectors, required, cursor = 0;
    uint32_t ranks, element, count = 0;

    if (!result) {
        return false;
    }
    *result = (CemuSparfStorageResult){ .ready_ns = start_ns, .status = CEMU_KV_INVALID };
    if (!attention || !attention->lease.buffer || !header || !indices || !output ||
        kind > CEMU_KV_PAGE_TOKEN_V || start_ns > INT64_MAX ||
        memcmp(header, &attention->header, sizeof(*header))) {
        return false;
    }
    vectors = (uint64_t)header->batch_size * header->num_query_heads;
    ranks = kind == CEMU_KV_PAGE_CHANNEL_K ? header->top_r : header->top_k;
    element = attention->lease.geometry.element_bytes;
    required = kind == CEMU_KV_PAGE_CHANNEL_K ? vectors * ranks * header->valid_tokens * element :
        vectors * (ranks + (kind == CEMU_KV_PAGE_TOKEN_V)) * header->head_dim * element;
    if (required != bytes || required > 64 * MiB) {
        return false;
    }
    for (uint64_t index = 0; index < vectors * ranks; index++) {
        if (indices[index] >= (kind == CEMU_KV_PAGE_CHANNEL_K ? header->head_dim : header->valid_tokens)) {
            return false;
        }
    }
    for (uint64_t index = 0; index < vectors * ranks; index++) {
        uint64_t vector = index / ranks;
        CemuKvCoordinate coordinate = {
            .batch = attention->lease.batch_first + vector / header->num_query_heads,
            .layer = header->layer,
            .head = (vector % header->num_query_heads) / (header->num_query_heads / header->num_kv_heads),
            .kind = kind,
            .token = kind == CEMU_KV_PAGE_CHANNEL_K ? 0 : indices[index],
            .channel = kind == CEMU_KV_PAGE_CHANNEL_K ? indices[index] : 0,
        };

        ranges[count++] = (NativeKvGatherRange){ coordinate,
            kind == CEMU_KV_PAGE_CHANNEL_K ? header->valid_tokens : 1 };
        if (count == G_N_ELEMENTS(ranges) || index + 1 == vectors * ranks) {
            CemuKvStatus status = read_ranges(attention, ranges, count,
                (uint8_t *)output + cursor, start_ns, result);

            if (status != CEMU_KV_OK) {
                result->status = status;
                return false;
            }
            for (uint32_t range = 0; range < count; range++) {
                cursor += range_bytes(attention, &ranges[range]);
            }
            count = 0;
        }
    }
    if (kind == CEMU_KV_PAGE_TOKEN_V) {
        uint64_t row = (uint64_t)header->head_dim * element;

        for (uint64_t vector = vectors; vector-- > 0;) {
            memmove((uint8_t *)output + vector * (ranks + 1) * row,
                    (uint8_t *)output + vector * ranks * row, ranks * row);
            memset((uint8_t *)output + (vector * (ranks + 1) + ranks) * row, 0, row);
        }
    }
    result->output_bytes = cursor;
    result->status = CEMU_KV_OK;
    return true;
}

void native_kv_attention_close(NativeKvAttention *attention)
{
    if (attention) {
        native_kv_buffer_release(&attention->lease);
    }
}
