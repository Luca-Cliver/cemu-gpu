#include "fdm_kv_layout.h"

static bool fdm_kv_multiply(uint64_t left, uint64_t right, uint64_t *result)
{
    if (right && left > UINT64_MAX / right) {
        return false;
    }

    *result = left * right;

    return true;
}

bool fdm_kv_layout_init(FDMKV_Layout_t *layout, const FDMKV_Geometry_t *geometry)
{
    FDMKV_Layout_t candidate = {0};
    uint64_t row_bytes;
    uint64_t channel_bytes;
    uint64_t token_pages;
    uint64_t channel_pages;

    if (!layout || !geometry ||
        !geometry->batches || !geometry->layers ||
        !geometry->heads || !geometry->head_dim ||
        !geometry->max_tokens || !geometry->channels_per_page ||
        geometry->reserved ||
        geometry->head_dim % geometry->channels_per_page ||
        (geometry->element_bytes != 2 && geometry->element_bytes != 4) ||
        geometry->page_bytes < 512 || geometry->page_bytes > 65536 ||
        geometry->page_bytes % 512) {
        return false;
    }

    row_bytes = (uint64_t)geometry->head_dim * geometry->element_bytes;
    channel_bytes = (uint64_t)geometry->channels_per_page * geometry->element_bytes;

    if (geometry->page_bytes % row_bytes ||
        geometry->page_bytes % channel_bytes) {
        return false;
    }

    candidate.geometry = *geometry;
    candidate.token_group_tokens = geometry->page_bytes / row_bytes;
    candidate.channel_group_tokens = geometry->page_bytes / channel_bytes;
    candidate.channel_group_pages = geometry->head_dim / geometry->channels_per_page;

    token_pages = geometry->max_tokens / candidate.token_group_tokens +
                  (geometry->max_tokens % candidate.token_group_tokens != 0);
    channel_pages = geometry->max_tokens / candidate.channel_group_tokens +
                    (geometry->max_tokens % candidate.channel_group_tokens != 0);

    if (!fdm_kv_multiply(token_pages, 2, &token_pages) ||
        !fdm_kv_multiply(channel_pages, candidate.channel_group_pages, &channel_pages) ||
        channel_pages > UINT64_MAX - token_pages) {
        return false;
    }

    candidate.pages_per_head = token_pages + channel_pages;

    if (!fdm_kv_multiply(geometry->batches, geometry->layers,
                         &candidate.head_count) ||
        !fdm_kv_multiply(candidate.head_count, geometry->heads,
                         &candidate.head_count) ||
        !fdm_kv_multiply(candidate.head_count, candidate.pages_per_head,
                         &candidate.total_pages) ||
        candidate.total_pages > UINT64_MAX / geometry->page_bytes) {
        return false;
    }

    *layout = candidate;
    return true;
}
