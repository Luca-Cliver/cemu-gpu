/* Reuse the packet generator and full page/queue validators from the layout
 * test. Only calls into the production shared library are timed. */
int aggregate_layout_test_main(int argc, char **argv);
#define main aggregate_layout_test_main
#include "../guest/test_fdm_kv_layout.c"
#undef main
#include <time.h>
#include <sys/mman.h>

static uint8_t *decode_cache_buffer;
static size_t decode_cache_bytes;
static volatile uint64_t decode_cache_sink;

static void perturb_decode_cache(void)
{
    const volatile uint8_t *bytes = decode_cache_buffer;
    uint64_t sum = 0;
    for (size_t offset = 0; offset < decode_cache_bytes; offset += 64) {
        sum += bytes[offset];
    }
    decode_cache_sink = sum;
}

static uint64_t bench_now(void)
{
    struct timespec stamp;
    assert(clock_gettime(CLOCK_MONOTONIC_RAW, &stamp) == 0);
    return (uint64_t)stamp.tv_sec * 1000000000 + stamp.tv_nsec;
}

static void report_huge_pages(const void *data, size_t bytes, unsigned variant)
{
    FILE *maps = fopen("/proc/self/smaps", "r");
    assert(maps);
    char line[512];
    unsigned long first, last, kib, huge_kib = 0;
    bool selected = false;
    while (fgets(line, sizeof(line), maps)) {
        if (sscanf(line, "%lx-%lx", &first, &last) == 2) {
            selected = first < (uintptr_t)data + bytes && (uintptr_t)data < last;
        } else if (selected && sscanf(line, "AnonHugePages: %lu kB", &kib) == 1) {
            huge_kib += kib;
        }
    }
    fclose(maps);
    fprintf(stderr, "page-ab variant=%u data_bytes=%zu AnonHugePages_KiB=%lu\n", variant, bytes, huge_kib);
}

static uint64_t bench_packet(AggregateFunction function, const FDMKV_InitParams_t *params,
                             FDMKV_Buffer_t *buffer, uint32_t head, uint32_t heads,
                             uint32_t token, uint32_t count)
{
    prepare_aggregate(buffer, 0, 0, head, heads, token, count);
    void *addresses[] = { params->meta_addr, params->data_addr };
    long long lengths[] = { params->meta_bytes, params->data_bytes };
    struct cemu_args args = { .numr = 2, .mr_addr = addresses, .mr_len = lengths,
        .cparam1 = CEMU_FDM_KV_AGGREGATE_COMMAND, .cparam2 = 9000 };
    if (token >= 1024 && decode_cache_bytes) {
        perturb_decode_cache(); /* Excluded from operator time; not a cache flush guarantee. */
    }
    uint64_t start = bench_now();
    long long status = function(&args);
    uint64_t elapsed = bench_now() - start;
    assert(status == FDM_KV_AGGREGATE_OK);
    assert(fdm_kv_receive_release(buffer));
    return elapsed;
}

/* Exercise the live Decode page working set across all 40 layers and B4/B8.
 * This synthetic window is positions 0..15 (same group offsets as 1024..1039),
 * without historical KV, a NAND writer, GPU work or thread contention. */
static void bench_interleaved(AggregateFunction *functions, unsigned repeats, unsigned batches, bool page_ab)
{
    const size_t meta_bytes = 32 * 1024 * 1024;
    const size_t huge = 2 * 1024 * 1024;
    const size_t data_bytes = ((size_t)batches * 40 * 40 * 20 * 4096 + 128 * 1024 + huge - 1) / huge * huge;
    void *meta = g_malloc(meta_bytes), *data = aligned_alloc(huge, data_bytes);
    assert(data);
    void *saved_meta = g_malloc(meta_bytes), *saved_data = g_malloc(data_bytes);
    for (uint32_t element = 2; element <= 4; element += 2) {
        FDMKV_InitParams_t params = {
            .identity = { .namespace_id = 1, .session_id = 1, .generation = 1 },
            .geometry = { .batches = batches, .layers = 40, .heads = 40, .head_dim = 128,
                .max_tokens = 16, .channels_per_page = 8, .element_bytes = element, .page_bytes = 4096 },
            .meta_addr = meta, .meta_bytes = meta_bytes, .data_addr = data, .data_bytes = data_bytes,
            .input_bytes = 128 * 1024, .nand_channels = 8, .block_pages = 256, .batch_pages = 256,
        };
        for (unsigned round = 0; round <= repeats; round++) {
            for (unsigned order = 0; order < 2; order++) {
                unsigned variant = (round + order) % 2;
                if (page_ab) {
                    assert(madvise(data, data_bytes, variant ? MADV_HUGEPAGE : MADV_NOHUGEPAGE) == 0);
                    assert(madvise(data, data_bytes, MADV_DONTNEED) == 0);
                }
                memset(meta, 0xa5, meta_bytes);
                memset(data, 0x5a, data_bytes);
                if (page_ab && !round) {
                    report_huge_pages(data, data_bytes, variant);
                }
                FDMKV_Buffer_t *buffer;
                assert(fdm_kv_buffer_init(&params, &buffer));
                uint64_t times[2] = {0}, pages[2] = {0};
                for (unsigned token = 0; token < 16; token++) {
                    unsigned phase = token != 0;
                    for (unsigned layer = 0; layer < 40; layer++) {
                        for (unsigned batch = 0; batch < batches; batch++) {
                            prepare_aggregate(buffer, batch, layer, 0, 40, token, 1);
                            uint64_t before = buffer->free_pages;
                            if (decode_cache_bytes) {
                                perturb_decode_cache();
                            }
                            uint64_t start = bench_now();
                            assert(invoke_aggregate(functions[page_ab ? 0 : variant], &params, 9000) == FDM_KV_AGGREGATE_OK);
                            times[phase] += bench_now() - start;
                            pages[phase] += before - buffer->free_pages;
                            assert(fdm_kv_receive_release(buffer));
                        }
                    }
                }
                printf("%u,%u,%u,decode_new_group,%u,%llu,%llu\n", element, round, variant, batches * 40,
                       (unsigned long long)times[0], (unsigned long long)pages[0]);
                printf("%u,%u,%u,decode_following,%u,%llu,%llu\n", element, round, variant, batches * 40 * 15,
                       (unsigned long long)times[1], (unsigned long long)pages[1]);
                printf("%u,%u,%u,decode_all,%u,%llu,%llu\n", element, round, variant, batches * 40 * 16,
                       (unsigned long long)(times[0] + times[1]), (unsigned long long)(pages[0] + pages[1]));
                if (!round) {
                    check_aggregate_pages(buffer);
                }
                if (!order) {
                    memcpy(saved_meta, meta, meta_bytes);
                    memcpy(saved_data, data, data_bytes);
                } else {
                    assert(memcmp(saved_meta, meta, meta_bytes) == 0);
                    assert(memcmp(saved_data, data, data_bytes) == 0);
                }
            }
        }
    }
    g_free(saved_data);
    g_free(saved_meta);
    g_free(data);
    g_free(meta);
}

int main(int argc, char **argv)
{
    assert(argc == 7);
    unsigned repeats = strtoul(argv[3], NULL, 10);
    assert(repeats >= 3 && repeats <= 100);
    unsigned cache_mib = strtoul(argv[4], NULL, 10);
    assert(cache_mib <= 256);
    unsigned stream_batches = strtoul(argv[5], NULL, 10);
    assert(stream_batches == 0 || stream_batches == 4 || stream_batches == 8);
    bool page_ab = strtoul(argv[6], NULL, 10) != 0;
    assert(!page_ab || stream_batches);
    decode_cache_bytes = (size_t)cache_mib * 1024 * 1024;
    if (decode_cache_bytes) {
        decode_cache_buffer = g_malloc(decode_cache_bytes);
        memset(decode_cache_buffer, 0x3c, decode_cache_bytes);
    }
    void *libraries[2];
    AggregateFunction functions[2];
    for (unsigned variant = 0; variant < 2; variant++) {
        libraries[variant] = dlopen(argv[variant + 1], RTLD_NOW | RTLD_LOCAL);
        assert(libraries[variant]);
        functions[variant] = (AggregateFunction)dlsym(libraries[variant], "fdm_kv_aggregate");
        assert(functions[variant]);
    }
    puts("element_bytes,round,variant,phase,calls,total_ns,new_pages");
    if (stream_batches) {
        bench_interleaved(functions, repeats, stream_batches, page_ab);
        goto finish;
    }
    const size_t meta_bytes = 8 * 1024 * 1024, data_bytes = 128 * 1024 * 1024;
    void *meta = g_malloc(meta_bytes), *data = g_malloc(data_bytes);
    void *saved_meta = g_malloc(meta_bytes), *saved_data = g_malloc(data_bytes);
    for (uint32_t element = 2; element <= 4; element += 2) {
        FDMKV_InitParams_t params = {
            .identity = { .namespace_id = 1, .session_id = 1, .generation = 1 },
            /* Four layers provide room for one layer's entire history without
             * a writer. Only batch 0/layer 0 is exercised; no NAND simulation. */
            .geometry = { .batches = 1, .layers = 4, .heads = 40, .head_dim = 128,
                .max_tokens = 1040, .channels_per_page = 8, .element_bytes = element,
                .page_bytes = 4096 },
            .meta_addr = meta, .meta_bytes = meta_bytes,
            .data_addr = data, .data_bytes = data_bytes, .input_bytes = 128 * 1024,
            .nand_channels = 8, .block_pages = 256, .batch_pages = 256,
        };
        /* Round 0 warms both implementations; later rounds reverse A/B order. */
        for (unsigned round = 0; round <= repeats; round++) {
            for (unsigned order = 0; order < 2; order++) {
                unsigned variant = (round + order) % 2;
                memset(meta, 0xa5, meta_bytes);
                memset(data, 0x5a, data_bytes); /* Fault pages outside timing. */
                FDMKV_Buffer_t *buffer;
                assert(fdm_kv_buffer_init(&params, &buffer));
                uint32_t tile = 2 * buffer->layout.token_group_tokens;
                uint64_t elapsed = 0, calls = 0, before = buffer->free_pages;
                for (uint32_t token = 0; token < 1024; token += tile) {
                    uint32_t count = MIN(tile, 1024 - token);
                    uint32_t packet_heads = (params.input_bytes - 512) /
                                            (2 * count * buffer->row_bytes);
                    for (uint32_t head = 0; head < 40; head += packet_heads) {
                        elapsed += bench_packet(functions[variant], &params, buffer,
                            head, MIN(packet_heads, 40 - head), token, count);
                        calls++;
                    }
                }
                printf("%u,%u,%u,prefill,%llu,%llu,%llu\n", element, round, variant,
                       (unsigned long long)calls, (unsigned long long)elapsed,
                       (unsigned long long)(before - buffer->free_pages));
                if (!round) {
                    check_aggregate_pages(buffer);
                }
                before = buffer->free_pages;
                uint64_t first = bench_packet(functions[variant], &params, buffer, 0, 40, 1024, 1);
                uint64_t first_pages = before - buffer->free_pages;
                printf("%u,%u,%u,decode_new_group,1,%llu,%llu\n", element, round, variant,
                       (unsigned long long)first, (unsigned long long)(before - buffer->free_pages));
                if (!round) {
                    check_aggregate_pages(buffer);
                }
                before = buffer->free_pages;
                elapsed = 0;
                for (uint32_t token = 1025; token < 1040; token++) {
                    elapsed += bench_packet(functions[variant], &params, buffer, 0, 40, token, 1);
                }
                printf("%u,%u,%u,decode_following,15,%llu,%llu\n", element, round, variant,
                       (unsigned long long)elapsed, (unsigned long long)(before - buffer->free_pages));
                printf("%u,%u,%u,decode_all,16,%llu,%llu\n", element, round, variant,
                       (unsigned long long)(elapsed + first),
                       (unsigned long long)(first_pages + before - buffer->free_pages));
                if (!round) {
                    check_aggregate_pages(buffer);
                }
                /* Same addresses and initial bytes for each A/B pair permit
                 * exact comparison of all metadata, queues, input and pages. */
                if (!order) {
                    memcpy(saved_meta, meta, meta_bytes);
                    memcpy(saved_data, data, data_bytes);
                } else {
                    assert(memcmp(saved_meta, meta, meta_bytes) == 0);
                    assert(memcmp(saved_data, data, data_bytes) == 0);
                }
            }
        }
    }
    g_free(saved_data);
    g_free(saved_meta);
    g_free(data);
    g_free(meta);
finish:
    fprintf(stderr, "PASS: baseline/optimized metadata and data byte-identical for all rounds; full page validators passed.\n");
    g_free(decode_cache_buffer);
    for (unsigned variant = 0; variant < 2; variant++) {
        assert(dlclose(libraries[variant]) == 0);
    }
    return 0;
}
