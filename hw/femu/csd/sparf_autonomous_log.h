#ifndef CEMU_SPARF_AUTONOMOUS_LOG_H
#define CEMU_SPARF_AUTONOMOUS_LOG_H

#include <errno.h>
#include <fcntl.h>
#include <unistd.h>
#include <pthread.h>

static uint64_t sparf_auto_session;
static uint64_t sparf_auto_sequence;
static pthread_mutex_t sparf_auto_log_lock = PTHREAD_MUTEX_INITIALIZER;

static bool sparf_auto_read_diagnostics_enabled(void)
{
    static gsize initialized;
    static bool enabled;
    if (g_once_init_enter(&initialized)) {
        const char *value = getenv("CEMU_SPARF_READ_DIAGNOSTICS");
        enabled = value && !strcmp(value, "1");
        g_once_init_leave(&initialized, 1);
    }
    return enabled;
}

static void sparf_auto_json_read_io(GString *line, const FemuReadObservation *read)
{
    g_string_append_printf(line,
        ",\"io\":{\"requested_bytes\":%llu,\"logical_bytes\":%llu,"
        "\"tail_bytes\":%llu,\"tail_copies\":%llu,\"logical_spans\":%llu,"
        "\"aligned_bytes_before_dedup\":%llu,\"submitted_bytes\":%llu,\"ranges\":%llu,"
        "\"model_start_ns\":%llu,\"model_end_ns\":%llu,"
        "\"pages\":%llu,\"unmapped_pages\":%llu,\"service_ns\":%llu,"
        "\"queue_ns\":%llu,\"max_queue_ns\":%llu,"
        "\"critical_queue_ns\":%llu,\"critical_service_ns\":%llu,"
        "\"critical_initial_wait_ns\":%llu,"
        "\"channels\":%u,\"luns_per_channel\":%u,\"page_bytes\":%u,"
        "\"page_read_ns\":%u,\"lun_pages\":[",
        (unsigned long long)(read->logical_bytes + read->tail_bytes),
        (unsigned long long)read->logical_bytes, (unsigned long long)read->tail_bytes,
        (unsigned long long)read->tail_copies, (unsigned long long)read->logical_spans,
        (unsigned long long)read->aligned_bytes_before_dedup,
        (unsigned long long)read->submitted_bytes, (unsigned long long)read->ranges,
        (unsigned long long)read->model_start_ns, (unsigned long long)read->model_end_ns,
        (unsigned long long)read->pages, (unsigned long long)read->unmapped_pages,
        (unsigned long long)read->service_ns, (unsigned long long)read->queue_ns,
        (unsigned long long)read->max_queue_ns, (unsigned long long)read->critical_queue_ns,
        (unsigned long long)read->critical_service_ns,
        (unsigned long long)read->critical_initial_wait_ns,
        read->channels, read->luns_per_channel, read->page_bytes, read->page_read_ns);
    uint32_t count = read->channels * read->luns_per_channel;
    for (uint32_t index = 0; index < count; ++index) {
        g_string_append_printf(line, "%s%llu", index ? "," : "",
            (unsigned long long)read->lun_pages[index]);
    }
    g_string_append(line, "],\"lun_initial_wait_ns\":[");
    for (uint32_t index = 0; index < count; ++index) {
        g_string_append_printf(line, "%s%llu", index ? "," : "",
            (unsigned long long)read->lun_initial_wait_ns[index]);
    }
    g_string_append(line, "]}");
}

static int sparf_auto_log_fd(void)
{
    static gsize initialized;
    static int descriptor = -1;
    if (g_once_init_enter(&initialized)) {
        const char *path = getenv("CEMU_SPARF_AUTONOMOUS_TIMELINE");
        if (path && *path) {
            descriptor = open(path, O_WRONLY | O_CREAT | O_APPEND | O_CLOEXEC, 0644);
            if (descriptor < 0) femu_err("cannot open autonomous timeline %s: %s\n", path, strerror(errno));
            sparf_auto_session = monotonic_ns();
        }
        g_once_init_leave(&initialized, 1);
    }
    return descriptor;
}

static void sparf_auto_json_stamp(GString *line, const char *name, SparfTraceStamp stamp)
{
    g_string_append_printf(line, ",\"%s\":{\"host_ns\":%llu,\"virtual_ns\":%llu}", name,
        (unsigned long long)stamp.host_ns, (unsigned long long)stamp.virtual_ns);
}

static GString *sparf_auto_json(const SparfAutonomousTrace *trace,
                               const struct cemu_sparf_workflow_header *header,
                               int controller, unsigned program, bool begin, uint64_t result)
{
    static const char *read_names[] = { "channel_k", "token_k", "token_v" };
    static const char *compute_names[] = { "approx", "qk", "pv" };
    GString *line = g_string_new(NULL);
    g_string_append_printf(line,
        "{\"schema\":\"sparf-autonomous-v1\",\"event\":\"%s\",\"pid\":%d,"
        "\"session\":%llu,\"request\":%llu,\"controller\":%d,\"program\":%u,"
        "\"layer\":%u,\"tokens\":%u,\"host_clock\":\"host_monotonic_ns\","
        "\"virtual_clock\":\"qemu_virtual_ns\"",
        begin ? "begin" : "complete", (int)getpid(), (unsigned long long)sparf_auto_session,
        (unsigned long long)trace->request, controller, program, header->layer, header->valid_tokens);
    sparf_auto_json_stamp(line, "admitted", trace->admitted);
    g_string_append_printf(line, ",\"engines\":%u,\"compute_units\":%u",
                           trace->engines, trace->compute_units);
    if (trace->group_write_order)
        g_string_append_printf(line, ",\"group_write_order\":\"%s\"", trace->group_write_order);
    if (!begin) {
        g_string_append_printf(line, ",\"engine\":%d", trace->engine);
        g_string_append_printf(line, ",\"success\":%s", result == UINT64_MAX ? "false" : "true");
        sparf_auto_json_stamp(line, "started", trace->started);
        sparf_auto_json_stamp(line, "ready", trace->ready);
        g_string_append(line, ",\"reads\":[");
        for (int index = 0; index < 3; ++index) {
            const SparfReadTrace *read = &trace->reads[index];
            g_string_append_printf(line, "%s{\"name\":\"%s\",\"deadline_ns\":%llu",
                index ? "," : "", read_names[index], (unsigned long long)read->deadline_ns);
            sparf_auto_json_stamp(line, "submitted", read->submitted);
            sparf_auto_json_stamp(line, "observed", read->observed);
            sparf_auto_json_stamp(line, "ready", read->ready);
            if (trace->read_diagnostics) sparf_auto_json_read_io(line, &read->io);
            g_string_append_c(line, '}');
        }
        g_string_append(line, "],\"computes\":[");
        for (int index = 0; index < 3; ++index) {
            const SparfComputeTrace *compute = &trace->computes[index];
            g_string_append_printf(line,
                "%s{\"name\":\"%s\",\"scheduled\":%s,\"truncated\":%s,"
                "\"effective_ns\":%llu,\"submitted_ns\":%llu,\"completed_ns\":%llu",
                index ? "," : "", compute_names[index], compute->scheduled ? "true" : "false",
                compute->truncated ? "true" : "false", (unsigned long long)compute->effective_ns,
                (unsigned long long)compute->submitted_ns, (unsigned long long)compute->completed_ns);
            sparf_auto_json_stamp(line, "functional_start", compute->functional_start);
            sparf_auto_json_stamp(line, "functional_end", compute->functional_end);
            sparf_auto_json_stamp(line, "observed_done", compute->observed_done);
            g_string_append(line, ",\"cu_spans\":[");
            for (guint span_index = 0; compute->spans && span_index < compute->spans->len; ++span_index) {
                SparfCuSpan span = g_array_index(compute->spans, SparfCuSpan, span_index);
                g_string_append_printf(line, "%s{\"cu\":%d,\"start_ns\":%llu,\"end_ns\":%llu}",
                    span_index ? "," : "", span.cu, (unsigned long long)span.start_ns,
                    (unsigned long long)span.end_ns);
            }
            g_string_append(line, "]}");
        }
        g_string_append_c(line, ']');
    }
    g_string_append(line, "}\n");
    return line;
}

static void sparf_auto_log(SparfAutonomousTrace *trace,
                           const struct cemu_sparf_workflow_header *header,
                           int controller, unsigned program, bool begin, uint64_t result)
{
    pthread_mutex_lock(&sparf_auto_log_lock);
    if (begin) trace->request = ++sparf_auto_sequence;
    GString *line = sparf_auto_json(trace, header, controller, program, begin, result);
    size_t offset = 0;
    while (offset < line->len) {
        ssize_t written = write(sparf_auto_log_fd(), line->str + offset, line->len - offset);
        if (written < 0 && errno == EINTR) continue;
        if (written <= 0) {
            femu_err("autonomous timeline write failed: %s\n", strerror(errno));
            break;
        }
        offset += written;
    }
    g_string_free(line, TRUE);
    pthread_mutex_unlock(&sparf_auto_log_lock);
}

#endif
