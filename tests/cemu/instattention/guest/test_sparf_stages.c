/* Run the production workflow with deterministic NVM reads. This tests device
 * stage state and arithmetic, not QEMU scheduling or CUDA performance. */
#include <assert.h>
#include <glib.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include "hw/femu/sparf_workflow_abi.h"

struct ubpf_jit_args {
    int numr;
    void **mr_addr, **mr_dev_addr;
    long long *mr_len;
    long long cparam1, cparam2;
    void *data_buffer;
    long long buffer_len;
} __attribute__((packed));
typedef struct { int unused; } SsdBackend;
typedef struct {
    int target;
    struct { uint64_t (*jit_fn)(struct ubpf_jit_args *); } shared_lib;
} Program;
typedef struct {
    struct ubpf_jit_args args;
    Program *program;
    SsdBackend **mr_backend;
    struct SparfAutonomous *sparf_autonomous;
} ComputeJob;
typedef struct {
    uint64_t offset;
    uint32_t size;
    uint64_t destination_offset;
} SparfLogicalSpan;
typedef struct {
    void *request;
    GArray *spans;
    uint8_t *source, *destination;
    uint32_t range;
    uint64_t latency;
} SparfRead;
struct rte_ring { int pending; };
#define PROGRAM_TARGET_CUDA_DEVPTR 1
#define FEMU_RING_TYPE_MP_SC 0
#define femu_err(...) fprintf(stderr, __VA_ARGS__)
static uint8_t **files;
static struct rte_ring *active_ring;
static int fail_read, fail_kernel;
static bool expected_v_pending;
static bool queue_model;
static uint64_t virtual_now = 1000000, nand_available;
static uint64_t read_starts[3], read_finishes[3];
static uint64_t (*kernel)(struct ubpf_jit_args *);
static int backend_cuda_prepare_host(SsdBackend *b, void *p, uint64_t n) { return 0; }
static int backend_cuda_prepare_device(SsdBackend *b, void *p, uint64_t n) { return 0; }
static void backend_cuda_mark_host_dirty(SsdBackend *b, void *p, uint64_t n) {}
static void backend_cuda_mark_device_dirty(SsdBackend *b, void *p, uint64_t n) {}
static uint64_t clock_ns(void) { return virtual_now; }
static uint64_t monotonic_ns(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000 + ts.tv_nsec;
}
#include "hw/femu/csd/sparf_timeline.h"
static bool observe_enabled;
static SparfObservation observation;

/* Two LUNs can serve pages concurrently. The busy LUN starts with 100 ns
 * backlog, then serves two 50 ns pages; the other serves one 50 ns page. */
void check_read_observation(void)
{
    uint64_t counts[2] = { 0 }, initial[2] = { 0 };
    FemuReadObservation obs = { .lun_pages = counts,
                                .lun_initial_wait_ns = initial };
    femu_observe_page_read(NULL, 0, 50, 150);
    femu_observe_page_read(&obs, 0, 50, 150);
    femu_observe_page_read(&obs, 1, 50, 50);
    femu_observe_page_read(&obs, 0, 50, 200);
    assert(obs.pages == 3 && counts[0] == 2 && counts[1] == 1);
    assert(obs.service_ns == 150 && obs.queue_ns == 250);
    assert(obs.critical_queue_ns == 150 && obs.critical_service_ns == 50);
    assert(obs.critical_initial_wait_ns == 100);
    assert(obs.max_queue_ns == 150);
}
static struct rte_ring *femu_ring_create(int type, int size)
{
    return active_ring = g_new0(struct rte_ring, 1);
}
static void femu_ring_free(struct rte_ring *ring)
{
    assert(ring->pending == 0);
    g_free(ring);
}
static bool sparf_start_read(ComputeJob *job, struct rte_ring *ring,
    const struct cemu_sparf_workflow_extent *map, uint32_t count,
    GArray *spans, uint32_t range, uint64_t start, SparfRead *read, FemuReadObservation *obs)
{
    assert(ring->pending == 0);  /* K is joined before V is issued. */
    *read = (SparfRead) { .request = read, .spans = spans,
        .source = files[range - 1], .destination = job->args.mr_addr[range],
        .range = range };
    if (range == 3) expected_v_pending = spans->len != 0;
    if (obs) {
        obs->model_start_ns = obs->model_end_ns = start;
        obs->logical_spans = spans->len;
        for (guint index = 0; index < spans->len; ++index) {
            SparfLogicalSpan *span = &g_array_index(spans, SparfLogicalSpan, index);
            obs->logical_bytes += span->size;
            obs->aligned_bytes_before_dedup += ((span->offset + span->size + 4095) & ~UINT64_C(4095)) -
                (span->offset & ~UINT64_C(4095));
        }
        obs->submitted_bytes = obs->aligned_bytes_before_dedup;
        obs->ranges = spans->len;
    }
    if (!spans->len) {
        read->request = NULL;
        return true;
    }
    /* A shared busy NAND resource reproduces FTL queueing. Completing a read
     * does NOT advance the frozen virtual clock, as in CEMU functional work. */
    uint64_t finish = (queue_model ? MAX(start, nand_available) : start) + range * 100;
    read_starts[range - 1] = start;
    read_finishes[range - 1] = finish;
    read->latency = finish - start;
    if (obs) obs->model_end_ns = finish;
    if (queue_model) nand_available = finish;
    ring->pending++;
    return true;
}
static void sparf_clear_read(SparfRead *read)
{
    if (read->spans) g_array_free(read->spans, TRUE);
    memset(read, 0, sizeof(*read));
}
static bool sparf_finish_read(SparfRead *read, struct rte_ring *ring, uint64_t *latency)
{
    if (read->spans && !read->spans->len) {
        *latency = 0;
        sparf_clear_read(read);
        return true;
    }
    assert(ring->pending == 1);
    ring->pending--;
    bool success = fail_read != (int)read->range;
    if (success) {
        for (guint i = 0; i < read->spans->len; ++i) {
            SparfLogicalSpan *s = &g_array_index(read->spans, SparfLogicalSpan, i);
            memcpy(read->destination + s->destination_offset, read->source + s->offset, s->size);
        }
        *latency = read->latency;
    }
    sparf_clear_read(read);
    return success;
}
static uint64_t checked_kernel(struct ubpf_jit_args *args)
{
    const struct cemu_sparf_attention_metadata *config = args->data_buffer;
    /* This is an ordering assertion: V is in flight during QK; it must have
     * completed before PV. It does not assert real-time overlap. */
    assert(active_ring->pending == (config->phase == CEMU_SPARF_PHASE_EXACT_QK && expected_v_pending));
    return fail_kernel == 1 || fail_kernel == (int)config->phase ? UINT64_MAX : kernel(args);
}
#include "hw/femu/csd/sparf_group_attention.h"
#include "hw/femu/csd/sparf_workflow.h"
#include "hw/femu/csd/sparf_autonomous_log.h"

static int autonomous_trace_enabled;
static GString *autonomous_trace_json;
static SparfGroupSession *grouped_session;

const char *get_read_observation_json(void)
{
    uint64_t counts[2] = {0}, initial[2] = {0};
    FemuReadObservation read = { .lun_pages = counts, .lun_initial_wait_ns = initial,
        .channels = 2, .luns_per_channel = 1, .page_bytes = 4096, .page_read_ns = 50,
        .logical_bytes = 512, .tail_bytes = 128, .tail_copies = 64, .logical_spans = 4,
        .aligned_bytes_before_dedup = 16384, .submitted_bytes = 12288, .ranges = 3,
        .model_start_ns = 1000, .model_end_ns = 1200 };
    femu_observe_page_read(&read, 0, 50, 150);
    femu_observe_page_read(&read, 1, 50, 50);
    femu_observe_page_read(&read, 0, 50, 200);
    if (autonomous_trace_json) g_string_free(autonomous_trace_json, TRUE);
    autonomous_trace_json = g_string_new("{\"fixture\":true");
    sparf_auto_json_read_io(autonomous_trace_json, &read);
    g_string_append_c(autonomous_trace_json, '}');
    return autonomous_trace_json->str;
}

void configure_grouped_session(void *session)
{
    grouped_session = session;
}

int grouped_test_lease(void *metadata, int release)
{
    struct cemu_sparf_workflow_header *header = metadata;
    struct cemu_sparf_group_binding binding;
    memcpy(&binding, (uint8_t *)header + sizeof(*header), sizeof(binding));
    if (release) {
        sparf_group_unbind(grouped_session, header, &binding);
        return 1;
    }
    return sparf_group_bind(grouped_session, header, &binding);
}

void configure_autonomous_trace(int enabled)
{
    autonomous_trace_enabled = enabled;
    if (autonomous_trace_json) g_string_free(autonomous_trace_json, TRUE);
    autonomous_trace_json = NULL;
}

const char *get_autonomous_trace(void)
{
    return autonomous_trace_json ? autonomous_trace_json->str : NULL;
}

void configure_kernel(void *function) { kernel = function; }
void configure_observation(int enabled) { observe_enabled = enabled; }
uint64_t observed_time(int phase, int end)
{
    return end ? observation.end[phase] : observation.start[phase];
}
void configure_timing(int enabled, uint64_t now, uint64_t busy_until)
{
    queue_model = enabled;
    virtual_now = now;
    nand_available = busy_until;
    memset(read_starts, 0, sizeof(read_starts));
    memset(read_finishes, 0, sizeof(read_finishes));
}
void set_virtual_time(uint64_t now) { virtual_now = now; }
uint64_t read_start(int index) { return read_starts[index]; }
uint64_t read_finish(int index) { return read_finishes[index]; }
uint64_t execute_stage(void **ranges, long long *lengths, void *metadata, int bytes,
                       uint8_t **nvm_files, uint64_t *runtime, int read_error, int kernel_error)
{
    Program program = { .target = 0, .shared_lib.jit_fn = checked_kernel };
    SsdBackend *backends[12] = { 0 };
    ComputeJob job = { .args = { .numr = 12, .mr_addr = ranges,
        .mr_len = lengths, .data_buffer = metadata, .buffer_len = bytes },
        .program = &program, .mr_backend = backends };
    files = nvm_files;
    fail_read = read_error;
    fail_kernel = kernel_error;
    *runtime = 0;
    memset(&observation, 0, sizeof(observation));
    return run_sparf_workflow(&job, runtime, observe_enabled ? &observation : NULL);
}

uint64_t execute_autonomous(void **ranges, long long *lengths, void *metadata, int bytes,
                           uint8_t **nvm_files, uint64_t *runtime, int read_error, int kernel_error)
{
    Program program = { .target = 0, .shared_lib.jit_fn = checked_kernel };
    SsdBackend *backends[12] = { 0 };
    SparfAutonomous state = { 0 };
    struct cemu_sparf_workflow_header *header = metadata;
    struct cemu_sparf_group_binding binding = {0};
    if (header->flags & CEMU_SPARF_WORKFLOW_GROUPED) {
        memcpy(&binding, (uint8_t *)metadata + sizeof(*header), sizeof(binding));
        if (!sparf_group_bind(grouped_session, header, &binding)) return UINT64_MAX;
        state.group_session = grouped_session;
    }
    SparfAutonomousTrace trace = { .request = 1 };
    trace.read_diagnostics = autonomous_trace_enabled > 1;
    if (autonomous_trace_enabled) {
        state.timeline = &trace;
        sparf_trace_stamp(&trace.admitted);
        sparf_trace_stamp(&trace.started);
        for (int index = 0; index < 3; ++index) {
            trace.computes[index].spans = g_array_new(FALSE, FALSE, sizeof(SparfCuSpan));
        }
        autonomous_trace_json = sparf_auto_json(&trace, metadata, 0, 1, true, 0);
    }
    ComputeJob job = { .args = { .numr = 11, .mr_addr = ranges,
        .mr_len = lengths, .data_buffer = metadata, .buffer_len = bytes },
        .program = &program, .mr_backend = backends, .sparf_autonomous = &state };
    files = nvm_files;
    fail_read = read_error;
    fail_kernel = kernel_error;
    uint64_t start = virtual_now, result = 0;
    unsigned transitions = 0, computations = 0;
    while (state.resume != SPARF_RESUME_DONE) {
        assert(++transitions <= 8);
        if (read_error < 0 && transitions == (unsigned)-read_error) {
            state.resume = SPARF_RESUME_DONE;
            result = run_sparf_workflow(&job, runtime, NULL);
            assert(result == UINT64_MAX);
            break;
        }
        if (state.timeline && (state.resume == SPARF_RESUME_VALUE || state.wait == SPARF_WAIT_IO)) {
            int index = state.resume == SPARF_RESUME_CHANNEL ? 0 : state.resume == SPARF_RESUME_KEY ? 1 : 2;
            sparf_trace_stamp(&trace.reads[index].observed);
        }
        if (state.wait == SPARF_WAIT_TIME) {
            virtual_now = MAX(virtual_now, state.deadline);
            if (state.timeline) {
                int index = state.resume == SPARF_RESUME_APPROX ? 0 : state.resume == SPARF_RESUME_EXACT ? 1 : 2;
                sparf_trace_stamp(&trace.reads[index].ready);
            }
        }
        uint64_t compute_runtime = 0;
        result = run_sparf_workflow(&job, &compute_runtime, NULL);
        if (state.timeline && state.wait == SPARF_WAIT_TIME) {
            int index = state.resume == SPARF_RESUME_APPROX ? 0 : state.resume == SPARF_RESUME_EXACT ? 1 : 2;
            trace.reads[index].deadline_ns = state.deadline;
        }
        if (state.wait == SPARF_WAIT_COMPUTE) {
            computations++;
            uint64_t submitted = virtual_now;
            virtual_now += compute_runtime;
            if (state.timeline) {
                int index = state.resume == SPARF_RESUME_KEY_START ? 0 : state.resume == SPARF_RESUME_VALUE ? 1 : 2;
                SparfComputeTrace *compute = &trace.computes[index];
                compute->submitted_ns = submitted;
                compute->completed_ns = virtual_now;
                compute->effective_ns = compute_runtime;
                compute->scheduled = compute_runtime != 0;
                if (compute_runtime) {
                    SparfCuSpan span = { .start_ns = submitted, .end_ns = virtual_now, .cu = 0 };
                    g_array_append_val(compute->spans, span);
                }
                sparf_trace_stamp(&compute->observed_done);
            }
        } else {
            assert(compute_runtime == 0);
        }
        if (result == UINT64_MAX) break;
    }
    assert(state.ring == NULL);
    if (state.group_session) sparf_group_unbind(state.group_session, header, &binding);
    if (result != UINT64_MAX) assert(computations == 3);
    *runtime = virtual_now - start;
    if (state.timeline) {
        sparf_trace_stamp(&trace.ready);
        GString *complete = sparf_auto_json(&trace, metadata, 0, 1, false, result);
        g_string_append_len(autonomous_trace_json, complete->str, complete->len);
        g_string_free(complete, TRUE);
        for (int index = 0; index < 3; ++index) g_array_free(trace.computes[index].spans, TRUE);
    }
    return result;
}
