#include "qemu/osdep.h"
#include "qemu/units.h"
#include "fdm_kv_attention.h"
#include "csd/memory.h"
#include "bbssd/bb.h"
#include "backend/backend.h"

typedef struct FDMKV_ReadFragment {
    CemuKvCoordinate coordinate;
    uint64_t key;
    uint32_t offset, bytes, destination;
} FDMKV_ReadFragment_t;

static GMutex *attention_lock(FDMKV_Binding_t *binding)
{
    MemoryNamespace *memory = binding->fdm_ns->private;
    return &memory->fdm_kv_lock;
}

static uint64_t head_index(const FDMKV_Buffer_t *buffer, uint32_t batch,
                            uint32_t layer, uint32_t head)
{
    return ((uint64_t)batch * buffer->layout.geometry.layers + layer) *
           buffer->layout.geometry.heads + head;
}

static uint64_t page_key(const FDMKV_Attention_t *attention,
                         const CemuKvCoordinate *coordinate, const CemuKvLocation *location)
{
    const CemuKvLayout *layout = &attention->layout;
    uint64_t token_pages = DIV_ROUND_UP(layout->geometry.max_tokens, layout->token_group_tokens);
    uint64_t base = head_index(attention->binding->buffer, coordinate->batch,
                               coordinate->layer, coordinate->head) * layout->pages_per_stream;
    if (coordinate->kind == FDM_KV_CHANNEL_K) {
        base += token_pages;
    } else if (coordinate->kind == FDM_KV_TOKEN_V) {
        base += layout->pages_per_stream - token_pages;
    }
    return base + location->page_index;
}

static uint32_t staged_page(const FDMKV_Buffer_t *buffer, uint64_t key)
{
    uint64_t hash = key * UINT64_C(0x9e3779b97f4a7c15);
    uint32_t slot = (hash ^ (hash >> 32)) & (buffer->hash_size - 1);
    for (uint32_t index = buffer->page_hash[slot]; index;
         index = buffer->staged[index].hash_next) {
        if (buffer->staged[index].key == key) {
            return index;
        }
    }
    return 0;
}

bool fdm_kv_attention_open(FemuCtrl *ctrl,
    const struct cemu_sparf_fdm_binding *wire,
    const struct cemu_sparf_workflow_header *header, FDMKV_Attention_t *attention)
{
    if (!wire->nvm_nsid || wire->nvm_nsid > ctrl->num_namespaces) {
        return false;
    }
    NvmeNamespace *nvm = nvme_find_namespace(ctrl, wire->nvm_nsid);
    FDMKV_Binding_t *binding = NULL;
    for (GList *entry = ctrl->fdm_kv_bindings; entry; entry = entry->next) {
        FDMKV_Binding_t *candidate = entry->data;
        if (candidate->nvm_ns == nvm) {
            binding = candidate;
            break;
        }
    }
    if (!binding || !binding->writer || attention->binding) {
        return false;
    }
    FDMKV_Buffer_t *buffer = binding->buffer;
    const FDMKV_Geometry_t *geometry = &buffer->layout.geometry;
    CemuKvGeometry converted = { geometry->batches, geometry->layers, geometry->heads,
        geometry->head_dim, geometry->max_tokens, geometry->channels_per_page,
        geometry->element_bytes, geometry->page_bytes };
    bool valid = false;
    g_mutex_lock(attention_lock(binding));
    if (buffer->failed || qatomic_read(&ctrl->native_stop) ||
        (buffer->draining && !buffer->sealed) || buffer->leases == UINT32_MAX ||
        (binding->pending_write && buffer->receive->header.operation == FDM_KV_OP_DRAIN) ||
        wire->session_id != buffer->identity.session_id ||
        wire->generation != buffer->identity.generation ||
        wire->batch_first >= geometry->batches || !header->batch_size ||
        header->batch_size > geometry->batches - wire->batch_first ||
        header->layer >= geometry->layers || header->head_dim != geometry->head_dim ||
        header->num_kv_heads != geometry->heads || header->max_seq_len > geometry->max_tokens ||
        !header->valid_tokens || header->valid_tokens > header->max_seq_len ||
        geometry->element_bytes != (header->dtype == CEMU_SPARF_FLOAT16 ? 2 : 4) ||
        !cemu_kv_layout_init(&attention->layout, &converted)) {
        goto done;
    }
    for (uint32_t batch = wire->batch_first; batch < wire->batch_first + header->batch_size; batch++) {
        for (uint32_t head = 0; head < geometry->heads; head++) {
            FDMKV_HeadState_t *state = &buffer->head_states[head_index(buffer, batch, header->layer, head)];
            if (state->accepted != header->valid_tokens || state->readers == UINT32_MAX) {
                goto done;
            }
        }
    }
    for (uint32_t batch = wire->batch_first; batch < wire->batch_first + header->batch_size; batch++) {
        for (uint32_t head = 0; head < geometry->heads; head++) {
            buffer->head_states[head_index(buffer, batch, header->layer, head)].readers++;
        }
    }
    buffer->leases++;
    attention->binding = binding;
    attention->batch_first = wire->batch_first;
    attention->identity = (CemuKvIdentity){ ctrl->drive_id, wire->nvm_nsid,
                                            wire->session_id, wire->generation };
    attention->header = *header;
    backend_cuda_mark_host_dirty(binding->fdm_ns->backend, buffer, buffer->meta_bytes);
    valid = true;
done:
    g_mutex_unlock(attention_lock(binding));
    return valid;
}

static void full_scan(FDMKV_Attention_t *attention, uint32_t kind,
                       FemuReadObservation *observation)
{
    const CemuKvLayout *layout = &attention->layout;
    const struct cemu_sparf_workflow_header *header = &attention->header;
    uint32_t capacity = kind == FDM_KV_CHANNEL_K ? layout->channel_group_tokens : layout->token_group_tokens;
    uint32_t stripes = kind == FDM_KV_CHANNEL_K ? layout->channel_group_pages : 1;
    observation->full_scan_valid = true;
    observation->page_bytes = layout->geometry.page_bytes;
    for (uint32_t batch = 0; batch < header->batch_size; batch++) {
        for (uint32_t head = 0; head < header->num_kv_heads; head++) {
            for (uint32_t token = 0; token < header->valid_tokens; token += capacity) {
                uint64_t bytes = (uint64_t)MIN(capacity, header->valid_tokens - token) *
                    layout->geometry.head_dim * layout->geometry.element_bytes / stripes;
                for (uint32_t stripe = 0; stripe < stripes; stripe++) {
                    CemuKvCoordinate coordinate = { attention->batch_first + batch, header->layer,
                        head, kind, token, stripe * layout->geometry.channels_per_page };
                    CemuKvLocation location;
                    bool resolved = cemu_kv_resolve(layout, &attention->identity, &coordinate, &location);
                    g_assert(resolved);
                    if (staged_page(attention->binding->buffer, page_key(attention, &coordinate, &location))) {
                        observation->full_scan_tail_bytes += bytes;
                    } else {
                        observation->full_scan_pages++;
                        observation->full_scan_logical_bytes += bytes;
                    }
                }
            }
        }
    }
    observation->full_scan_flash_bytes = observation->full_scan_pages * observation->page_bytes;
}

static int compare_fragments(const void *left, const void *right)
{
    const FDMKV_ReadFragment_t *first = left, *second = right;
    if (first->key != second->key) {
        return first->key < second->key ? -1 : 1;
    }
    return (first->offset > second->offset) - (first->offset < second->offset);
}

bool fdm_kv_attention_read(void *opaque,
    const struct cemu_sparf_workflow_header *header, const uint32_t *indices,
    uint32_t kind, void *output, size_t bytes, uint64_t start_ns,
    CemuSparfStorageResult *result, FemuReadObservation *observation)
{
    FDMKV_Attention_t *attention = opaque;
    FDMKV_ReadFragment_t *fragments = NULL;
    uint8_t *scratch = NULL;
    CemuKvDevicePage *pages = NULL;
    uint32_t used = 0, cursor = 0;
    bool success = false;
    if (!result) {
        return false;
    }
    *result = (CemuSparfStorageResult){ .ready_ns = start_ns, .status = CEMU_KV_INVALID };
    if (!attention || !attention->binding || !header || !indices || !output ||
        kind > FDM_KV_TOKEN_V || start_ns > INT64_MAX ||
        memcmp(header, &attention->header, sizeof(*header))) {
        return false;
    }
    FDMKV_Binding_t *binding = attention->binding;
    FDMKV_Buffer_t *buffer = binding->buffer;
    uint32_t element = attention->layout.geometry.element_bytes;
    uint32_t row = header->head_dim * element;
    uint32_t ranks = kind == FDM_KV_CHANNEL_K ? header->top_r : header->top_k;
    uint64_t vectors = (uint64_t)header->batch_size * header->num_query_heads;
    uint64_t required = kind == FDM_KV_CHANNEL_K ? vectors * ranks * header->valid_tokens * element :
        vectors * (ranks + (kind == FDM_KV_TOKEN_V)) * row;
    uint32_t capacity = kind == FDM_KV_CHANNEL_K ? attention->layout.channel_group_tokens :
                                                 attention->layout.token_group_tokens;
    uint64_t pieces = vectors * ranks * (kind == FDM_KV_CHANNEL_K ?
        DIV_ROUND_UP(header->valid_tokens, capacity) : 1);
    if (required != bytes || required > 64 * MiB || !pieces ||
        pieces > (64 * MiB) / sizeof(*fragments)) {
        return false;
    }
    for (uint64_t index = 0; index < vectors * ranks; index++) {
        if (indices[index] >= (kind == FDM_KV_CHANNEL_K ? header->head_dim : header->valid_tokens)) {
            return false;
        }
    }
    fragments = g_try_new(FDMKV_ReadFragment_t, pieces);
    scratch = g_try_malloc((uint64_t)buffer->batch_pages * attention->layout.geometry.page_bytes);
    pages = g_try_new(CemuKvDevicePage, buffer->batch_pages);
    if (!fragments || !scratch || !pages) {
        result->status = CEMU_KV_NO_MEMORY;
        goto done;
    }
    g_mutex_lock(attention_lock(binding));
    if (buffer->failed || qatomic_read(&binding->nvm_ns->ctrl->native_stop)) {
        result->status = CEMU_KV_IO_FAILED;
        g_mutex_unlock(attention_lock(binding));
        goto done;
    }
    if (observation) {
        full_scan(attention, kind, observation);
        observation->logical_spans = vectors * ranks;
    }
    for (uint64_t vector = 0; vector < vectors; vector++) {
        for (uint32_t rank = 0; rank < ranks; rank++) {
            CemuKvCoordinate coordinate = { attention->batch_first + vector / header->num_query_heads,
                header->layer, (vector % header->num_query_heads) / (header->num_query_heads / header->num_kv_heads),
                kind, kind == FDM_KV_CHANNEL_K ? 0 : indices[vector * ranks + rank],
                kind == FDM_KV_CHANNEL_K ? indices[vector * ranks + rank] : 0 };
            uint32_t remaining = kind == FDM_KV_CHANNEL_K ? header->valid_tokens : 1;
            while (remaining) {
                CemuKvLocation location;
                bool resolved = cemu_kv_resolve(&attention->layout, &attention->identity, &coordinate, &location);
                g_assert(resolved);
                uint32_t tokens = MIN(remaining, capacity - coordinate.token % capacity);
                uint32_t copied = tokens * (kind == FDM_KV_CHANNEL_K ? element : row);
                uint64_t key = page_key(attention, &coordinate, &location);
                uint32_t staged = staged_page(buffer, key);
                if (staged) {
                    memcpy((uint8_t *)output + cursor, buffer->staged_data +
                        (uint64_t)(staged - 1) * attention->layout.geometry.page_bytes + location.byte_offset, copied);
                    result->tail_bytes += copied;
                    if (observation) {
                        observation->tail_copies++;
                    }
                } else {
                    fragments[used++] = (FDMKV_ReadFragment_t){ coordinate, key, location.byte_offset, copied, cursor };
                }
                cursor += copied;
                coordinate.token += tokens;
                remaining -= tokens;
            }
        }
    }
    g_mutex_unlock(attention_lock(binding));
    if (observation) {
        observation->aligned_bytes_before_dedup = (uint64_t)used * attention->layout.geometry.page_bytes;
    }
    qsort(fragments, used, sizeof(*fragments), compare_fragments);
    for (uint32_t first = 0; first < used;) {
        uint32_t starts[CEMU_NATIVE_INTERNAL_MAX_PAGES + 1];
        uint32_t count = 0, next = first;
        while (next < used && count < buffer->batch_pages) {
            starts[count] = next;
            pages[count] = (CemuKvDevicePage){ .coordinate = fragments[next].coordinate,
                .buffer = scratch + (uint64_t)count * attention->layout.geometry.page_bytes,
                .bytes = attention->layout.geometry.page_bytes };
            uint64_t key = fragments[next].key;
            do { next++; } while (next < used && fragments[next].key == key);
            count++;
        }
        starts[count] = next;
        CemuKvDeviceBatchResult batch = { .ready_ns = start_ns };
        CemuKvStatus status = bb_native_kv_batch_observed(binding->nvm_ns, &attention->identity,
            pages, count, false, start_ns, &batch, observation);
        result->ready_ns = MAX(result->ready_ns, batch.ready_ns);
        result->flash_bytes += batch.transferred_bytes;
        result->pages += batch.completed_pages;
        result->batches++;
        if (status != CEMU_KV_OK) {
            result->status = status;
            goto done;
        }
        for (uint32_t slot = 0; slot < count; slot++) {
            uint32_t covered = 0;
            for (uint32_t index = starts[slot]; index < starts[slot + 1]; index++) {
                FDMKV_ReadFragment_t *fragment = &fragments[index];
                memcpy((uint8_t *)output + fragment->destination,
                       (uint8_t *)pages[slot].buffer + fragment->offset, fragment->bytes);
                uint32_t end = fragment->offset + fragment->bytes;
                if (observation && end > covered) {
                    observation->unique_flash_bytes += end - MAX(covered, fragment->offset);
                }
                covered = MAX(covered, end);
            }
        }
        first = next;
    }
    if (kind == FDM_KV_TOKEN_V) {
        for (uint64_t vector = vectors; vector-- > 0;) {
            memmove((uint8_t *)output + vector * (ranks + 1) * row,
                    (uint8_t *)output + vector * ranks * row, ranks * row);
            memset((uint8_t *)output + (vector * (ranks + 1) + ranks) * row, 0, row);
        }
    }
    result->output_bytes = cursor;
    result->status = CEMU_KV_OK;
    success = true;
done:
    g_mutex_lock(attention_lock(binding));
    buffer->stats.read_bytes += result->flash_bytes;
    backend_cuda_mark_host_dirty(binding->fdm_ns->backend, buffer, buffer->meta_bytes);
    g_mutex_unlock(attention_lock(binding));
    g_free(pages);
    g_free(scratch);
    g_free(fragments);
    return success;
}

void fdm_kv_attention_close(void *opaque)
{
    FDMKV_Attention_t *attention = opaque;
    if (!attention || !attention->binding) {
        return;
    }
    FDMKV_Binding_t *binding = attention->binding;
    FDMKV_Buffer_t *buffer = binding->buffer;
    g_mutex_lock(attention_lock(binding));
    for (uint32_t batch = attention->batch_first; batch < attention->batch_first + attention->header.batch_size; batch++) {
        for (uint32_t head = 0; head < attention->header.num_kv_heads; head++) {
            FDMKV_HeadState_t *state = &buffer->head_states[head_index(buffer, batch, attention->header.layer, head)];
            g_assert(state->readers);
            state->readers--;
        }
    }
    g_assert(buffer->leases);
    buffer->leases--;
    backend_cuda_mark_host_dirty(binding->fdm_ns->backend, buffer, buffer->meta_bytes);
    attention->binding = NULL;
    g_mutex_unlock(attention_lock(binding));
}
