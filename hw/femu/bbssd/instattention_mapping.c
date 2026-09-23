#include "instattention_mapping.h"
#include <stdlib.h>

typedef struct KvTask {
    CemuKvIdentity identity;
    CemuKvLayout layout;
} KvTask;

typedef enum KvPageState {
    KV_RESERVED,
    KV_WRITTEN,
    KV_FAILED
} KvPageState;

typedef struct KvPage {
    CemuKvPageRef reference;
    uint64_t ready_ns;
    KvPageState state;
} KvPage;

typedef struct KvDomain {
    size_t task_index;
    uint32_t batch;
    uint32_t layer;
    uint32_t kind;
    uint32_t channel;
    CemuKvPhysicalPage active;
    uint32_t next_page;
} KvDomain;

struct CemuKvMap {
    CemuKvMapConfig config;
    CemuKvMapStats stats;
    KvTask *tasks;
    KvPage *pages;
    KvDomain *domains;
    size_t *page_hash;
    size_t *domain_hash;
    size_t hash_size;
    size_t domain_count;
    uint64_t next_block[128];
};

static uint64_t hash_field(uint64_t hash, uint64_t value)
{
    value ^= value >> 30;
    value *= UINT64_C(0xbf58476d1ce4e5b9);
    value ^= value >> 27;
    value *= UINT64_C(0x94d049bb133111eb);
    return (hash ^ value ^ (value >> 31)) * UINT64_C(0x100000001b3);
}

static uint64_t page_hash(const CemuKvPageKey *key)
{
    uint64_t hash = hash_field(0, key->identity.controller);

    hash = hash_field(hash, key->identity.namespace_id);
    hash = hash_field(hash, key->identity.task);
    hash = hash_field(hash, key->identity.generation);
    hash = hash_field(hash, key->batch);
    hash = hash_field(hash, key->layer);
    hash = hash_field(hash, key->head);
    hash = hash_field(hash, key->kind);
    hash = hash_field(hash, key->group);
    return hash_field(hash, key->channel_group);
}

static size_t page_slot(const CemuKvMap *map, const CemuKvPageKey *key)
{
    size_t slot = page_hash(key) % map->hash_size;

    while (map->page_hash[slot] &&
           !cemu_kv_page_key_equal(&map->pages[map->page_hash[slot] - 1].reference.key,
                                   key)) {
        slot = (slot + 1) % map->hash_size;
    }
    return slot;
}

static size_t domain_slot(const CemuKvMap *map, const KvDomain *candidate)
{
    uint64_t hash = hash_field(0, candidate->task_index);
    size_t slot;

    hash = hash_field(hash, candidate->batch);
    hash = hash_field(hash, candidate->layer);
    hash = hash_field(hash, candidate->kind);
    slot = hash_field(hash, candidate->channel) % map->hash_size;
    while (map->domain_hash[slot]) {
        const KvDomain *domain = &map->domains[map->domain_hash[slot] - 1];

        if (domain->task_index == candidate->task_index &&
            domain->batch == candidate->batch && domain->layer == candidate->layer &&
            domain->kind == candidate->kind && domain->channel == candidate->channel) {
            break;
        }
        slot = (slot + 1) % map->hash_size;
    }
    return slot;
}

static CemuKvStatus find_task(const CemuKvMap *map, const CemuKvIdentity *identity,
                              size_t *index)
{
    size_t current;

    if (!map || !identity || !identity->generation ||
        identity->controller != map->config.controller ||
        identity->namespace_id != map->config.namespace_id) {
        return CEMU_KV_INVALID;
    }
    for (current = 0; current < map->stats.tasks; current++) {
        if (map->tasks[current].identity.task == identity->task) {
            if (map->tasks[current].identity.generation != identity->generation) {
                return CEMU_KV_STALE;
            }
            *index = current;
            return CEMU_KV_OK;
        }
    }
    return CEMU_KV_NOT_FOUND;
}

static bool add_allocation(size_t count, size_t width, size_t *bytes)
{
    if (count > (SIZE_MAX - *bytes) / width) {
        return false;
    }
    *bytes += count * width;
    return true;
}

CemuKvStatus cemu_kv_map_create(const CemuKvMapConfig *config, CemuKvMap **result)
{
    CemuKvMap *map;
    CemuKvPhysicalPage first = {0};
    uint64_t offset;
    size_t bytes = sizeof(CemuKvMap), hash_size;

    if (!config || !result || !config->namespace_id || !config->max_tasks ||
        !config->max_pages || config->max_pages > SIZE_MAX / 2 ||
        !cemu_kv_physical_offset(&config->nand, &first, config->backend_bytes, &offset)) {
        return CEMU_KV_INVALID;
    }
    hash_size = config->max_pages * 2;
    if (!add_allocation(config->max_tasks, sizeof(KvTask), &bytes) ||
        !add_allocation(config->max_pages, sizeof(KvPage), &bytes) ||
        !add_allocation(config->max_pages, sizeof(KvDomain), &bytes) ||
        !add_allocation(hash_size, sizeof(size_t), &bytes) ||
        !add_allocation(hash_size, sizeof(size_t), &bytes) ||
        bytes > config->memory_budget_bytes) {
        return CEMU_KV_LIMIT;
    }
    map = calloc(1, sizeof(*map));
    if (!map) {
        return CEMU_KV_NO_MEMORY;
    }
    map->config = *config;
    map->hash_size = hash_size;
    map->stats.memory_bytes = bytes;
    map->tasks = calloc(config->max_tasks, sizeof(*map->tasks));
    map->pages = calloc(config->max_pages, sizeof(*map->pages));
    map->domains = calloc(config->max_pages, sizeof(*map->domains));
    map->page_hash = calloc(hash_size, sizeof(*map->page_hash));
    map->domain_hash = calloc(hash_size, sizeof(*map->domain_hash));
    if (!map->tasks || !map->pages || !map->domains ||
        !map->page_hash || !map->domain_hash) {
        cemu_kv_map_destroy(map);
        return CEMU_KV_NO_MEMORY;
    }
    *result = map;
    return CEMU_KV_OK;
}

void cemu_kv_map_destroy(CemuKvMap *map)
{
    if (map) {
        free(map->tasks);
        free(map->pages);
        free(map->domains);
        free(map->page_hash);
        free(map->domain_hash);
        free(map);
    }
}

CemuKvStatus cemu_kv_map_register(CemuKvMap *map,
                                const CemuKvIdentity *identity,
                                const CemuKvGeometry *geometry)
{
    size_t index;
    CemuKvLayout layout;
    CemuKvStatus status = find_task(map, identity, &index);

    if (status != CEMU_KV_NOT_FOUND) {
        return status == CEMU_KV_OK ? CEMU_KV_EXISTS : status;
    }
    if (!cemu_kv_layout_init(&layout, geometry) ||
        geometry->page_bytes != map->config.nand.page_bytes) {
        return CEMU_KV_INVALID;
    }
    if (map->stats.tasks == map->config.max_tasks) {
        return CEMU_KV_LIMIT;
    }
    map->tasks[map->stats.tasks++] = (KvTask){ .identity = *identity, .layout = layout };
    return CEMU_KV_OK;
}

CemuKvStatus cemu_kv_map_reserve(CemuKvMap *map,
                               const CemuKvIdentity *identity,
                               const CemuKvCoordinate *page_start,
                               uint32_t valid_tokens, CemuKvPageRef *result)
{
    size_t task_index, page_entry, domain_entry;
    CemuKvLocation location;
    CemuKvPageRef reference;
    KvDomain candidate, *domain;
    const CemuKvNandGeometry *nand;
    uint64_t ordinal, lanes;
    bool new_block;
    CemuKvStatus status = find_task(map, identity, &task_index);

    if (status != CEMU_KV_OK) {
        return status;
    }
    if (!result || !cemu_kv_resolve(&map->tasks[task_index].layout, identity,
                                   page_start, &location) || location.byte_offset ||
        !valid_tokens || valid_tokens > location.token_capacity) {
        return CEMU_KV_INVALID;
    }
    page_entry = page_slot(map, &location.key);
    if (map->page_hash[page_entry]) {
        return CEMU_KV_EXISTS;
    }
    if (map->stats.reserved_pages == map->config.max_pages) {
        return CEMU_KV_LIMIT;
    }
    nand = &map->config.nand;
    candidate = (KvDomain){
        .task_index = task_index, .batch = location.key.batch,
        .layer = location.key.layer, .kind = location.key.kind,
        .channel = location.key.group % nand->channels,
    };
    domain_entry = domain_slot(map, &candidate);
    domain = map->domain_hash[domain_entry] ?
             &map->domains[map->domain_hash[domain_entry] - 1] : NULL;
    if (domain) {
        candidate = *domain;
    }
    new_block = !domain || candidate.next_page == nand->pages_per_block;
    if (new_block) {
        ordinal = map->next_block[candidate.channel];
        lanes = (uint64_t)nand->luns_per_channel * nand->planes_per_lun;
        if (ordinal >= lanes * nand->blocks_per_plane) {
            return CEMU_KV_NO_SPACE;
        }
        candidate.active = (CemuKvPhysicalPage){
            .channel = candidate.channel,
            .lun = ordinal % nand->luns_per_channel,
            .plane = (ordinal / nand->luns_per_channel) % nand->planes_per_lun,
            .block = ordinal / lanes,
        };
        candidate.next_page = 0;
    }
    candidate.active.page = candidate.next_page;
    reference = (CemuKvPageRef){
        .key = location.key, .physical = candidate.active,
        .ticket = map->stats.reserved_pages + 1, .valid_tokens = valid_tokens,
    };
    if (!cemu_kv_physical_offset(nand, &reference.physical,
                                map->config.backend_bytes, &reference.backend_offset)) {
        return CEMU_KV_INVALID;
    }
    candidate.next_page++;
    if (!domain) {
        domain = &map->domains[map->domain_count++];
        map->domain_hash[domain_entry] = map->domain_count;
    }
    *domain = candidate;
    if (new_block) {
        map->next_block[candidate.channel]++;
        map->stats.claimed_blocks++;
    }
    map->pages[map->stats.reserved_pages] = (KvPage){
        .reference = reference, .state = KV_RESERVED,
    };
    map->page_hash[page_entry] = ++map->stats.reserved_pages;
    *result = reference;
    return CEMU_KV_OK;
}

CemuKvStatus cemu_kv_map_complete(CemuKvMap *map, const CemuKvPageRef *reference,
                                bool io_success, uint64_t ready_ns)
{
    KvPage *page;

    if (!map || !reference || !reference->ticket ||
        reference->ticket > map->stats.reserved_pages) {
        return CEMU_KV_INVALID;
    }
    page = &map->pages[reference->ticket - 1];
    if (!cemu_kv_page_key_equal(&page->reference.key, &reference->key)) {
        return CEMU_KV_STALE;
    }
    if (page->reference.backend_offset != reference->backend_offset ||
        page->reference.valid_tokens != reference->valid_tokens ||
        page->reference.physical.channel != reference->physical.channel ||
        page->reference.physical.lun != reference->physical.lun ||
        page->reference.physical.plane != reference->physical.plane ||
        page->reference.physical.block != reference->physical.block ||
        page->reference.physical.page != reference->physical.page) {
        return CEMU_KV_INVALID;
    }
    if (page->state != KV_RESERVED) {
        return CEMU_KV_EXISTS;
    }
    page->ready_ns = ready_ns;
    page->state = io_success ? KV_WRITTEN : KV_FAILED;
    if (io_success) {
        map->stats.written_pages++;
    } else {
        map->stats.failed_pages++;
    }
    return CEMU_KV_OK;
}

CemuKvStatus cemu_kv_map_query_written(const CemuKvMap *map,
                                     const CemuKvIdentity *identity,
                                     const CemuKvCoordinate *coordinate,
                                     CemuKvPageRef *result,
                                     uint32_t *byte_offset, uint64_t *ready_ns)
{
    size_t task_index, entry;
    CemuKvLocation location;
    const KvPage *page;
    uint32_t capacity;
    CemuKvStatus status = find_task(map, identity, &task_index);

    if (status != CEMU_KV_OK) {
        return status;
    }
    if (!result || !byte_offset || !ready_ns ||
        !cemu_kv_resolve(&map->tasks[task_index].layout, identity, coordinate, &location)) {
        return CEMU_KV_INVALID;
    }
    entry = map->page_hash[page_slot(map, &location.key)];
    if (!entry) {
        return CEMU_KV_NOT_FOUND;
    }
    page = &map->pages[entry - 1];
    capacity = coordinate->kind == CEMU_KV_PAGE_CHANNEL_K ?
               map->tasks[task_index].layout.channel_group_tokens :
               map->tasks[task_index].layout.token_group_tokens;
    if (coordinate->token % capacity >= page->reference.valid_tokens) {
        return CEMU_KV_NOT_FOUND;
    }
    if (page->state == KV_FAILED) {
        return CEMU_KV_IO_FAILED;
    }
    if (page->state == KV_RESERVED) {
        return CEMU_KV_PENDING;
    }
    *result = page->reference;
    *byte_offset = location.byte_offset;
    *ready_ns = page->ready_ns;
    return CEMU_KV_OK;
}

CemuKvStatus cemu_kv_map_lookup(const CemuKvMap *map,
                              const CemuKvIdentity *identity,
                              const CemuKvCoordinate *coordinate,
                              uint64_t now_ns, CemuKvPageRef *result,
                              uint32_t *byte_offset)
{
    CemuKvPageRef candidate;
    uint32_t offset;
    uint64_t ready_ns;
    CemuKvStatus status;

    if (!result || !byte_offset) {
        return CEMU_KV_INVALID;
    }
    status = cemu_kv_map_query_written(map, identity, coordinate, &candidate,
                                      &offset, &ready_ns);
    if (status != CEMU_KV_OK) {
        return status;
    }
    if (now_ns < ready_ns) {
        return CEMU_KV_PENDING;
    }
    *result = candidate;
    *byte_offset = offset;
    return CEMU_KV_OK;
}

CemuKvMapStats cemu_kv_map_stats(const CemuKvMap *map)
{
    return map ? map->stats : (CemuKvMapStats){0};
}
