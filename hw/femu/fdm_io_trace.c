#include "qemu/osdep.h"
#include "hw/femu/nvme.h"
#include "hw/femu/csd/memory.h"
#include "fdm_io_trace.h"

static GMutex trace_lock;
static FILE *trace_file;
static gsize trace_initialized;

void fdm_io_trace_begin(NvmeRequest *request)
{
    FdmIoTrace *trace = &request->fdm_io_trace;
    FemuCtrl *ctrl = request->sq->ctrl;
    uint32_t nsid = le32_to_cpu(request->cmd.nsid);
    memset(trace, 0, sizeof(*trace));
    if (!nsid || nsid > ctrl->num_namespaces) {
        return;
    }
    enum NvmeCsi csi = ctrl->namespaces[nsid - 1].csi;
    if (!((csi == NVME_CSI_MEMORY &&
           (request->cmd.opcode == NVME_CMD_MEMORY_WRITE ||
            request->cmd.opcode == NVME_CMD_MEMORY_READ)) ||
          (csi == NVME_CSI_COMPUTE &&
           request->cmd.opcode == NVME_CMD_COMPUTE_EXEC))) {
        return;
    }
    if (g_once_init_enter(&trace_initialized)) {
        const char *path = getenv("CEMU_FDM_IO_TRACE");
        if (path && *path) {
            trace_file = fopen(path, "ae");
            if (!trace_file) {
                femu_err("cannot open FDM IO trace %s: %s\n", path, strerror(errno));
            } else {
                setvbuf(trace_file, NULL, _IOLBF, 0);
            }
        }
        g_once_init_leave(&trace_initialized, 1);
    }
    trace->enabled = trace_file != NULL;
    FDM_IO_STAMP(request, sq_seen);
}

uint64_t fdm_io_trace_check_begin(NvmeRequest *request)
{
    FdmIoTrace *trace = &request->fdm_io_trace;
    if (!trace->enabled) {
        return 0;
    }
    uint64_t now = clock_ns();
    if (!trace->checks) {
        trace->first_check = now;
    } else {
        uint64_t elapsed = now - trace->last_check;
        switch (trace->retry) {
        case FDM_IO_RETRY_SPACE: trace->space_retry_ns += elapsed; break;
        case FDM_IO_RETRY_BUSY: trace->busy_retry_ns += elapsed; break;
        case FDM_IO_RETRY_DRAIN: trace->drain_retry_ns += elapsed; break;
        case FDM_IO_RETRY_MODEL: trace->model_retry_ns += elapsed; break;
        case FDM_IO_RETRY_AGGREGATE: trace->aggregate_yield_ns += elapsed; break;
        default: break;
        }
    }
    trace->retry = FDM_IO_RETRY_NONE;
    return now;
}

void fdm_io_trace_check_end(NvmeRequest *request, uint64_t start)
{
    FdmIoTrace *trace = &request->fdm_io_trace;
    if (trace->enabled) {
        trace->last_check = clock_ns();
        trace->check_ns += trace->last_check - start;
        trace->checks++;
    }
}

static void emit_trace(NvmeRequest *request, FemuCtrl *ctrl, uint16_t qid)
{
    FdmIoTrace *trace = &request->fdm_io_trace;
    if (!trace->enabled) {
        return;
    }
    NvmeMemoryWriteCmd *command = (void *)&request->cmd;
    GString *line = g_string_new(NULL);
    g_string_append_printf(line,
        "{\"version\":2,\"opcode\":%u,\"source\":\"%s\",\"pid\":%d,\"controller\":%d,"
        "\"async_aggregate\":%s,\"accepted_ns\":%llu,\"started_ns\":%llu,\"finished_ns\":%llu,"
        "\"qid\":%u,\"cid\":%u,\"nsid\":%u,\"offset\":%llu,\"bytes\":%u,"
        "\"status\":%u,\"input\":%s,\"operation\":%u,\"session\":%llu,\"packet_id\":%llu,"
        "\"token_first\":%llu,\"virtual\":{",
        request->cmd.opcode, trace->background ? "background_aggregate" : "device", getpid(), ctrl->drive_id,
        trace->async_aggregate ? "true" : "false",
        (unsigned long long)trace->accepted_ns, (unsigned long long)trace->started_ns,
        (unsigned long long)trace->finished_ns, qid,
        le16_to_cpu(command->cid), le32_to_cpu(command->nsid),
        (unsigned long long)le64_to_cpu(command->sb), le32_to_cpu(command->wl),
        request->status, trace->input ? "true" : "false", trace->operation,
        (unsigned long long)trace->session,
        (unsigned long long)trace->packet_id, (unsigned long long)trace->token_first);
#define APPEND_FIELD(field) g_string_append_printf(line, "\"" #field "\":%llu,", \
                                                  (unsigned long long)trace->field)
    APPEND_FIELD(sq_seen);
    APPEND_FIELD(memory_enter);
    APPEND_FIELD(locked);
    APPEND_FIELD(dma_begin);
    APPEND_FIELD(dma_end);
    APPEND_FIELD(memory_done);
    APPEND_FIELD(model_ready);
    APPEND_FIELD(first_check);
    APPEND_FIELD(last_check);
    APPEND_FIELD(exec_enter);
    APPEND_FIELD(exec_admitted);
    APPEND_FIELD(exec_started);
    APPEND_FIELD(exec_ready);
    APPEND_FIELD(cq_begin);
    g_string_append_printf(line, "\"cq_end\":%llu},", (unsigned long long)trace->cq_end);
    APPEND_FIELD(check_ns);
    APPEND_FIELD(input_slot_wait_ns);
    APPEND_FIELD(packet_bytes);
    APPEND_FIELD(aggregate_ns);
    APPEND_FIELD(aggregate_host_ns);
    APPEND_FIELD(checks);
    APPEND_FIELD(aggregate_calls);
    APPEND_FIELD(aggregate_probe_calls);
    APPEND_FIELD(aggregate_thread_cpu_ns);
    APPEND_FIELD(aggregate_minor_faults);
    APPEND_FIELD(aggregate_major_faults);
    APPEND_FIELD(aggregate_voluntary_switches);
    APPEND_FIELD(aggregate_involuntary_switches);
    if (trace->aggregate_probe_calls) {
        g_string_append_printf(line,
            "\"aggregate_location\":{\"tid\":%d,\"cpu_begin\":%d,\"cpu_end\":%d,"
            "\"cpu_changes\":%llu,\"host_begin_ns\":%llu,\"host_end_ns\":%llu,"
            "\"batch\":%u,\"layer\":%u,\"head_first\":%u,\"head_count\":%u},",
            trace->aggregate_tid, trace->aggregate_cpu_begin, trace->aggregate_cpu_end,
            (unsigned long long)trace->aggregate_cpu_changes,
            (unsigned long long)trace->aggregate_first_host_ns,
            (unsigned long long)trace->aggregate_last_host_ns,
            trace->aggregate_batch, trace->aggregate_layer,
            trace->aggregate_head_first, trace->aggregate_head_count);
    }
    APPEND_FIELD(space_retries);
    APPEND_FIELD(busy_retries);
    APPEND_FIELD(space_retry_ns);
    APPEND_FIELD(busy_retry_ns);
    APPEND_FIELD(drain_retry_ns);
    APPEND_FIELD(model_retry_ns);
    APPEND_FIELD(aggregate_yield_ns);
    if (trace->aggregate_profile.calls) {
        const FDMKV_AggregateProfile *profile = &trace->aggregate_profile;
        g_string_append(line, "\"aggregate_profile\":{\"version\":1,\"host_ns\":{");
#define PROFILE_STAGE(name) g_string_append_printf(line, "\"" #name "\":%llu,", \
            (unsigned long long)profile->host_ns[FDM_KV_AGG_STAGE_##name]);
        FDM_KV_AGGREGATE_PROFILE_STAGES(PROFILE_STAGE)
#undef PROFILE_STAGE
        g_string_truncate(line, line->len - 1);
        g_string_append(line, "},");
#define PROFILE_COUNT(name) g_string_append_printf(line, "\"" #name "\":%llu,", \
            (unsigned long long)profile->name);
        PROFILE_COUNT(calls)
        PROFILE_COUNT(successful_calls)
        PROFILE_COUNT(new_pages)
        PROFILE_COUNT(copied_bytes)
        PROFILE_COUNT(channel_elements)
#undef PROFILE_COUNT
        g_string_truncate(line, line->len - 1);
        g_string_append(line, "},");
    }
    g_string_append_printf(line, "\"cq_full\":%llu}\n", (unsigned long long)trace->cq_full);
#undef APPEND_FIELD
    g_mutex_lock(&trace_lock);
    if (fwrite(line->str, 1, line->len, trace_file) != line->len) {
        femu_err("FDM IO trace write failed\n");
    }
    g_mutex_unlock(&trace_lock);
    g_string_free(line, true);
    trace->enabled = false;
}

void fdm_io_trace_end(NvmeRequest *request)
{
    emit_trace(request, request->sq->ctrl, request->sq->sqid);
}

void fdm_io_trace_background(NvmeRequest *request, uint16_t qid)
{
    emit_trace(request, request->ns->ctrl, qid);
}
