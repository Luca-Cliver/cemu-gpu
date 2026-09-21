/* Private SparF device workflow implementation.
 * Included after the FTL read helpers in compute.c. Kept separately so the
 * dependency graph can be exercised with a deterministic FTL in unit tests. */
#ifndef FEMU_CSD_SPARF_WORKFLOW_H
#define FEMU_CSD_SPARF_WORKFLOW_H

#include "sparf_autonomous_trace.h"

typedef struct SparfScoreIndex {
    float score;
    uint32_t index;
} SparfScoreIndex;

enum SparfResume {
    SPARF_RESUME_START, SPARF_RESUME_CHANNEL, SPARF_RESUME_APPROX,
    SPARF_RESUME_KEY_START, SPARF_RESUME_KEY, SPARF_RESUME_EXACT,
    SPARF_RESUME_VALUE, SPARF_RESUME_PV, SPARF_RESUME_DONE,
};

enum SparfWait {
    SPARF_WAIT_NONE, SPARF_WAIT_IO, SPARF_WAIT_TIME, SPARF_WAIT_COMPUTE,
};

typedef struct SparfAutonomous {
    SparfGroupSession *group_session;
    enum SparfResume resume;
    enum SparfWait wait;
    uint64_t deadline;
    uint64_t read_start;
    struct rte_ring *ring;
    uint32_t *channels, *tokens;
    float *scores, *query_values, *channel_values, *current_key_values;
    SparfScoreIndex *score_indices, *channel_indices;
    SparfRead channel_read, key_read, value_read;
    struct cemu_sparf_workflow_trace trace;
    uint64_t result;
    uint16_t rsid;
    SparfAutonomousTrace *timeline;
} SparfAutonomous;

static void sparf_trace_stamp(SparfTraceStamp *stamp)
{
    stamp->host_ns = monotonic_ns();
    stamp->virtual_ns = clock_ns();
}

static int sparf_score_index_compare(const void *left, const void *right)
{
    const SparfScoreIndex *first = left;
    const SparfScoreIndex *second = right;

    if (first->score > second->score) {
        return -1;
    }
    if (first->score < second->score) {
        return 1;
    }
    return first->index < second->index ? -1 : first->index != second->index;
}

static float sparf_half_to_float(uint16_t value)
{
    uint32_t sign = (uint32_t)(value & 0x8000U) << 16;
    uint32_t exponent = (value >> 10) & 31U;
    uint32_t mantissa = value & 1023U;
    uint32_t bits;
    if (!exponent) {
        if (!mantissa) {
            bits = sign;
        } else {
            exponent = 127 - 15 + 1;
            while (!(mantissa & 1024U)) {
                mantissa <<= 1;
                --exponent;
            }
            bits = sign | (exponent << 23) | ((mantissa & 1023U) << 13);
        }
    } else if (exponent == 31) {
        bits = sign | 0x7f800000U | (mantissa << 13);
    } else {
        bits = sign | ((exponent + 112) << 23) | (mantissa << 13);
    }
    float result;
    memcpy(&result, &bits, sizeof(result));
    return result;
}

static float sparf_load_scalar(const void *address, uint64_t index, uint32_t dtype)
{
    return dtype == CEMU_SPARF_FLOAT16
        ? sparf_half_to_float(((const uint16_t *)address)[index])
        : ((const float *)address)[index];
}

static uint64_t run_sparf_phase(ComputeJob *job, uint32_t phase,
                                const uint32_t *indices, uint32_t count,
                                struct cemu_sparf_attention_metadata *metadata)
{
    void *host[4] = { 0 };
    void *device[4] = { 0 };
    long long lengths[4] = { 0 };
    SsdBackend *backends[4] = { 0 };
    for (uint32_t index = 0; index < count; ++index) {
        uint32_t source = indices[index];
        host[index] = job->args.mr_addr[source];
        device[index] = job->args.mr_dev_addr ? job->args.mr_dev_addr[source] : NULL;
        lengths[index] = job->args.mr_len[source];
        backends[index] = job->mr_backend[source];
    }
    if (phase == CEMU_SPARF_PHASE_EXACT_QK) {
        uint64_t element = metadata->dtype == CEMU_SPARF_FLOAT16 ? 2 : 4;
        lengths[0] = (uint64_t)metadata->batch_size * metadata->num_query_heads *
                     metadata->head_dim * element;
        lengths[1] = (uint64_t)metadata->batch_size * metadata->num_query_heads *
                     metadata->selected_tokens * metadata->head_dim * element;
        lengths[2] = (uint64_t)metadata->batch_size * metadata->num_query_heads *
                     sizeof(float);
    } else {
        uint64_t element = metadata->dtype == CEMU_SPARF_FLOAT16 ? 2 : 4;
        uint64_t vectors = (uint64_t)metadata->batch_size * metadata->num_query_heads;
        lengths[0] = vectors * (metadata->selected_tokens + 1) * sizeof(float);
        lengths[1] = vectors * (metadata->selected_tokens + 1) *
                     metadata->head_dim * element;
    }
    struct ubpf_jit_args arguments = job->args;
    arguments.numr = count;
    arguments.mr_addr = host;
    arguments.mr_dev_addr = job->args.mr_dev_addr ? device : NULL;
    arguments.mr_len = lengths;
    metadata->phase = phase;
    arguments.data_buffer = metadata;
    arguments.buffer_len = sizeof(*metadata);
    for (uint32_t index = 0; index + 1 < count; ++index) {
        int status = job->program->target == PROGRAM_TARGET_CUDA_DEVPTR
            ? backend_cuda_prepare_device(backends[index], host[index], lengths[index])
            : backend_cuda_prepare_host(backends[index], host[index], lengths[index]);
        if (status != 0) return (uint64_t)-1;
    }
    uint64_t result = job->program->shared_lib.jit_fn(&arguments);
    if (result != (uint64_t)-1) {
        if (job->program->target == PROGRAM_TARGET_CUDA_DEVPTR)
            backend_cuda_mark_device_dirty(backends[count - 1], host[count - 1], lengths[count - 1]);
        else
            backend_cuda_mark_host_dirty(backends[count - 1], host[count - 1], lengths[count - 1]);
    }
    return result;
}

/* Each continuation is stored in its owning MRS, so close/reset frees it with
 * the slot.  A caller must exclusively own the slot until PV completes. */
static struct cemu_sparf_stage_context *sparf_open_stage(
    ComputeJob *job, const struct cemu_sparf_workflow_header *header)
{
    struct cemu_sparf_stage_context *context =
        job->args.mr_addr[CEMU_SPARF_STAGE_CONTEXT_RANGE];
    const uint32_t length = job->args.buffer_len;
    const uint64_t count = (uint64_t)header->batch_size *
                           header->num_query_heads * header->top_k;
    struct cemu_sparf_workflow_header canonical = *header;
    canonical.reserved[0] = 0;
    uint8_t *saved = (uint8_t *)(context + 1);
    if (backend_cuda_prepare_host(job->mr_backend[CEMU_SPARF_STAGE_CONTEXT_RANGE],
                                  context, sizeof(*context) + length + count * 4) != 0) {
        return NULL;
    }
    if (header->reserved[0] == CEMU_SPARF_STAGE_SELECT) {
        memset(context, 0, sizeof(*context));
        context->magic = CEMU_SPARF_CONTEXT_MAGIC;
        context->workflow_id = header->reserved[1];
        context->metadata_bytes = length;
        context->token_count = count;
        memcpy(saved, &canonical, sizeof(canonical));
        memcpy(saved + sizeof(canonical),
               (uint8_t *)job->args.data_buffer + sizeof(canonical),
               length - sizeof(canonical));
    } else if (context->magic != CEMU_SPARF_CONTEXT_MAGIC ||
               context->completed_stage != header->reserved[0] - 1 ||
               context->workflow_id != header->reserved[1] ||
               context->metadata_bytes != length || context->token_count != count ||
               memcmp(saved, &canonical, sizeof(canonical)) ||
               memcmp(saved + sizeof(canonical),
                      (uint8_t *)job->args.data_buffer + sizeof(canonical),
                      length - sizeof(canonical))) {
        femu_err("SparF continuation has stale id, changed metadata or invalid stage order\n");
        return NULL;
    }
    return context;
}

static GArray *sparf_token_spans(const struct cemu_sparf_workflow_header *header,
                                const uint32_t *tokens, bool values, uint32_t published_tokens)
{
    uint32_t element = header->dtype == CEMU_SPARF_FLOAT16 ? 2 : 4;
    uint32_t heads_per_kv = header->num_query_heads / header->num_kv_heads;
    uint64_t vectors = (uint64_t)header->batch_size * header->num_query_heads;
    GArray *spans = g_array_sized_new(FALSE, FALSE, sizeof(SparfLogicalSpan),
                                     vectors * header->top_k);
    for (uint64_t vector = 0; vector < vectors; ++vector) {
        uint32_t batch = vector / header->num_query_heads;
        uint32_t head = (vector % header->num_query_heads) / heads_per_kv;
        for (uint32_t rank = 0; rank < header->top_k; ++rank) {
            if (tokens[vector * header->top_k + rank] >= published_tokens) continue;
            SparfLogicalSpan span = {
                .offset = (uint64_t)header->layer * header->token_layer_stride +
                          (uint64_t)batch * header->token_batch_stride +
                          (uint64_t)head * header->token_head_stride +
                          (uint64_t)tokens[vector * header->top_k + rank] * header->head_dim * element,
                .size = header->head_dim * element,
                .destination_offset = (vector * (header->top_k + values) + rank) *
                                      header->head_dim * element,
            };
            g_array_append_val(spans, span);
        }
    }
    return spans;
}

static uint64_t run_sparf_workflow(ComputeJob *job, uint64_t *modeled_runtime,
                                  SparfObservation *observation)
{
    SparfAutonomous *autonomous = job->sparf_autonomous;
    SparfAutonomousTrace *timeline = autonomous ? autonomous->timeline : NULL;
    FemuReadObservation *read_observations[3];
    for (int index = 0; index < 3; ++index) {
        read_observations[index] = timeline && timeline->read_diagnostics ? &timeline->reads[index].io :
            observation ? &observation->reads[index] : NULL;
    }
    *modeled_runtime = 0;
    struct cemu_sparf_workflow_header header;
    struct cemu_sparf_workflow_trace trace = { 0 };
    memcpy(&header, job->args.data_buffer, sizeof(header));
    bool grouped = header.flags & CEMU_SPARF_WORKFLOW_GROUPED;
    if (grouped && (!autonomous || !autonomous->group_session)) return (uint64_t)-1;
    struct cemu_sparf_buffer_tail tail = { header.valid_tokens, 0 };
    if (header.flags & CEMU_SPARF_WORKFLOW_BUFFERED) {
        if (job->args.buffer_len < sizeof(header) + sizeof(tail)) return (uint64_t)-1;
        memcpy(&tail, (uint8_t *)job->args.data_buffer + job->args.buffer_len - sizeof(tail), sizeof(tail));
        uint64_t elements = (uint64_t)header.batch_size * header.num_kv_heads * header.head_dim;
        uint64_t bytes = elements * (header.dtype == CEMU_SPARF_FLOAT16 ? 2 : 4);
        if (tail.published_tokens > header.valid_tokens ||
            tail.buffered_tokens != header.valid_tokens - tail.published_tokens ||
            (uint64_t)job->args.mr_len[5] < MAX(tail.buffered_tokens, 1) * bytes ||
            (uint64_t)job->args.mr_len[6] < MAX(tail.buffered_tokens, 1) * bytes) return (uint64_t)-1;
        for (uint32_t range = 5; range <= 6 &&
             (!autonomous || autonomous->resume == SPARF_RESUME_START); ++range) {
            if (backend_cuda_prepare_host(job->mr_backend[range], job->args.mr_addr[range],
                                          MAX(tail.buffered_tokens, 1) * bytes) != 0) return (uint64_t)-1;
        }
    }
    uint64_t kv_elements = (uint64_t)header.batch_size * header.num_kv_heads * header.head_dim;
    uint64_t current_offset = tail.buffered_tokens ? (tail.buffered_tokens - 1) * kv_elements : 0;
    const struct cemu_sparf_workflow_extent *maps =
        (const struct cemu_sparf_workflow_extent *)
        ((const uint8_t *)job->args.data_buffer + sizeof(header));
    const struct cemu_sparf_workflow_extent *token_k_map = maps;
    const struct cemu_sparf_workflow_extent *channel_k_map = token_k_map + header.token_k_extents;
    const struct cemu_sparf_workflow_extent *token_v_map = channel_k_map + header.channel_k_extents;
    const uint32_t stage = header.version == CEMU_SPARF_STAGED_VERSION
                         ? header.reserved[0] : 0;
    struct cemu_sparf_stage_context *context = NULL;
    if (stage) {
        context = sparf_open_stage(job, &header);
        if (!context) return (uint64_t)-1;
        trace = context->trace;
    }
    uint64_t previous_runtime = trace.total_model_ns;
    /* QEMU's virtual clock is frozen during functional execution. FTL
     * completion advances NAND availability, not that clock. Advance a local
     * cursor after each dependency, otherwise K/V latency includes already
     * charged earlier reads. A new command starts at its execution time here;
     * do not add the preceding stages' cumulative runtime to it again. */
    uint64_t model_cursor = clock_ns();
    bool select = !stage || stage == CEMU_SPARF_STAGE_SELECT;
    const uint32_t vectors = header.batch_size * header.num_query_heads;
    const uint32_t element = header.dtype == CEMU_SPARF_FLOAT16 ? 2 : 4;
    const uint32_t heads_per_kv = header.num_query_heads / header.num_kv_heads;
    bool resume = autonomous && autonomous->resume != SPARF_RESUME_START;
    struct rte_ring *ring = resume ? autonomous->ring : femu_ring_create(FEMU_RING_TYPE_MP_SC, 8);
    uint32_t *channels = resume ? autonomous->channels : select ? g_new(uint32_t, (uint64_t)vectors * header.top_r) : NULL;
    uint32_t *tokens = resume ? autonomous->tokens : g_new(uint32_t, (uint64_t)vectors * header.top_k);
    float *scores = resume ? autonomous->scores : select ? g_new(float, header.valid_tokens) : NULL;
    SparfScoreIndex *score_indices = resume ? autonomous->score_indices : select ? g_new(SparfScoreIndex, header.valid_tokens) : NULL;
    float *query_values = resume ? autonomous->query_values : select ? g_new(float, (uint64_t)vectors * header.head_dim) : NULL;
    float *channel_values = resume ? autonomous->channel_values : select ? g_new(float,
        (uint64_t)vectors * header.top_r * header.valid_tokens) : NULL;
    float *current_key_values = resume ? autonomous->current_key_values : select && !grouped ? g_new(float,
        (uint64_t)header.batch_size * header.num_kv_heads * header.head_dim) : NULL;
    SparfScoreIndex *channel_indices = resume ? autonomous->channel_indices : select ? g_new(SparfScoreIndex, header.head_dim) : NULL;
    uint64_t result = (uint64_t)-1;
    SparfRead channel_read = { 0 }, key_read = { 0 }, value_read = { 0 };
    struct cemu_sparf_attention_metadata metadata = {
        .version = CEMU_SPARF_ATTENTION_VERSION,
        .phase = CEMU_SPARF_PHASE_EXACT_QK,
        .dtype = header.dtype,
        .batch_size = header.batch_size,
        .num_query_heads = header.num_query_heads,
        .head_dim = header.head_dim,
        .selected_tokens = header.top_k,
        .scale = header.scale,
    };
    if (!ring) goto out;
    uint32_t *saved_tokens = context ? (uint32_t *)((uint8_t *)(context + 1) +
                                                   context->metadata_bytes) : NULL;
    if (resume) {
        channel_read = autonomous->channel_read;
        key_read = autonomous->key_read;
        value_read = autonomous->value_read;
        trace = autonomous->trace;
        switch (autonomous->resume) {
        case SPARF_RESUME_CHANNEL: goto resume_channel;
        case SPARF_RESUME_APPROX: goto resume_approx;
        case SPARF_RESUME_KEY_START: goto resume_key_start;
        case SPARF_RESUME_KEY: goto resume_key;
        case SPARF_RESUME_EXACT: goto resume_exact;
        case SPARF_RESUME_VALUE: goto resume_value;
        case SPARF_RESUME_PV: goto resume_pv;
        default: goto out;
        }
    }
    if (stage && !select) {
        memcpy(tokens, saved_tokens, (uint64_t)vectors * header.top_k * sizeof(*tokens));
        for (uint64_t i = 0; i < (uint64_t)vectors * header.top_k; ++i) {
            if (tokens[i] >= header.valid_tokens) goto out;
        }
    }
    if (select) {
        if (backend_cuda_prepare_host(job->mr_backend[0], job->args.mr_addr[0],
                                               job->args.mr_len[0]) != 0) goto out;

        for (uint64_t index = 0; index < (uint64_t)vectors * header.head_dim; ++index) {
            query_values[index] = sparf_load_scalar(job->args.mr_addr[0], index,
                                                    header.dtype);
        }

        GArray *channel_spans = g_array_new(FALSE, FALSE, sizeof(SparfLogicalSpan));
        for (uint32_t vector = 0; vector < vectors; ++vector) {
            for (uint32_t dimension = 0; dimension < header.head_dim; ++dimension) {
                channel_indices[dimension].score = fabsf(query_values[
                    (uint64_t)vector * header.head_dim + dimension]);
                channel_indices[dimension].index = dimension;
            }
            qsort(channel_indices, header.head_dim, sizeof(*channel_indices),
                  sparf_score_index_compare);
            for (uint32_t rank = 0; rank < header.top_r; ++rank) {
                uint32_t best_index = channel_indices[rank].index;
                channels[(uint64_t)vector * header.top_r + rank] = best_index;
                uint32_t batch = vector / header.num_query_heads;
                uint32_t head = (vector % header.num_query_heads) / heads_per_kv;
                SparfLogicalSpan span = {
                    .offset = (uint64_t)header.layer * header.channel_layer_stride +
                              (uint64_t)batch * header.channel_batch_stride +
                              (uint64_t)head * header.channel_head_stride +
                              (uint64_t)best_index * header.channel_stride,
                    .size = tail.published_tokens * element,
                    .destination_offset = ((uint64_t)vector * header.top_r + rank) *
                                          header.valid_tokens * element,
                };
                if (!grouped && span.size) g_array_append_val(channel_spans, span);
            }
        }
        sparf_observe_begin(observation, SPARF_OBS_CHANNEL_READ);
        if (timeline) {
            sparf_trace_stamp(&timeline->reads[0].submitted);
            timeline->reads[0].submitted.virtual_ns = model_cursor;
        }
        if (grouped) {
            g_array_free(channel_spans, TRUE);
            if (!sparf_start_grouped_read(job, autonomous->group_session, &header, channels,
                CEMU_SPARF_GROUP_CHANNEL_K, 1, ring, model_cursor, &channel_read,
                read_observations[0])) goto out;
        } else if (!sparf_start_read(job, ring, channel_k_map, header.channel_k_extents,
                              channel_spans, 1, model_cursor, &channel_read,
                              read_observations[0])) goto out;
        if (autonomous) {
            autonomous->read_start = model_cursor;
            autonomous->resume = SPARF_RESUME_CHANNEL;
            autonomous->wait = SPARF_WAIT_IO;
            goto suspend;
        }
resume_channel:
        if (!sparf_finish_read(&channel_read, ring, &trace.channel_read_model_ns)) goto out;
        sparf_observe_end(observation, SPARF_OBS_CHANNEL_READ);
        model_cursor += trace.channel_read_model_ns + header.approximate_runtime_ns;
        if (autonomous) {
            autonomous->deadline = autonomous->read_start + trace.channel_read_model_ns;
            autonomous->resume = SPARF_RESUME_APPROX;
            autonomous->wait = SPARF_WAIT_TIME;
            goto suspend;
        }

resume_approx:
        if (timeline) sparf_trace_stamp(&timeline->computes[0].functional_start);
        for (uint64_t index = 0;
             index < (uint64_t)vectors * header.top_r * header.valid_tokens;
             ++index) {
            if (grouped) {
                channel_values[index] = sparf_load_scalar(job->args.mr_addr[1], index, header.dtype);
                continue;
            }
            uint32_t token = index % header.valid_tokens;
            uint32_t vector = index / (header.valid_tokens * header.top_r);
            uint32_t rank = (index / header.valid_tokens) % header.top_r;
            uint32_t batch = vector / header.num_query_heads;
            uint32_t head = (vector % header.num_query_heads) / heads_per_kv;
            uint64_t source = (uint64_t)(token - tail.published_tokens) * kv_elements +
                ((uint64_t)batch * header.num_kv_heads + head) * header.head_dim +
                channels[(uint64_t)vector * header.top_r + rank];
            channel_values[index] = token < tail.published_tokens
                ? sparf_load_scalar(job->args.mr_addr[1], index, header.dtype)
                : sparf_load_scalar(job->args.mr_addr[5], source, header.dtype);
        }
        if (!grouped && backend_cuda_prepare_host(job->mr_backend[5], job->args.mr_addr[5], job->args.mr_len[5]) != 0) goto out;
        for (uint64_t index = 0;
             !grouped && index < (uint64_t)header.batch_size * header.num_kv_heads * header.head_dim;
             ++index) {
            current_key_values[index] = sparf_load_scalar(job->args.mr_addr[5], current_offset + index,
                                                          header.dtype);
        }
        float *alpha = (float *)job->args.mr_addr[7];
        if (backend_cuda_prepare_host(job->mr_backend[7], alpha, vectors * sizeof(float)) != 0) goto out;
        for (uint32_t vector = 0; vector < vectors; ++vector) {
            uint32_t batch = vector / header.num_query_heads;
            uint32_t head = (vector % header.num_query_heads) / heads_per_kv;
            float q_norm = 0.0f, selected_norm = 0.0f, maximum = -INFINITY;
            for (uint32_t dimension = 0; dimension < header.head_dim; ++dimension)
                q_norm += fabsf(query_values[(uint64_t)vector * header.head_dim + dimension]);
            for (uint32_t rank = 0; rank < header.top_r; ++rank)
                selected_norm += fabsf(query_values[(uint64_t)vector * header.head_dim +
                    channels[(uint64_t)vector * header.top_r + rank]]);
            float ratio = q_norm / MAX(selected_norm, 1.17549435e-38f);
            for (uint32_t token = 0; token < header.valid_tokens; ++token) {
                scores[token] = 0.0f;
            }
            for (uint32_t rank = 0; rank < header.top_r; ++rank) {
                uint32_t dimension = channels[(uint64_t)vector * header.top_r + rank];
                float query_value = query_values[(uint64_t)vector * header.head_dim + dimension];
                const float *key_values = &channel_values[
                    ((uint64_t)vector * header.top_r + rank) * header.valid_tokens];
                for (uint32_t token = 0; token + 1 < header.valid_tokens; ++token) {
                    scores[token] += query_value * key_values[token];
                }
                scores[header.valid_tokens - 1] += query_value * (grouped ? key_values[header.valid_tokens - 1] :
                    current_key_values[((uint64_t)batch * header.num_kv_heads + head) * header.head_dim + dimension]);
            }
            for (uint32_t token = 0; token < header.valid_tokens; ++token) {
                scores[token] *= ratio * header.scale;
                maximum = MAX(maximum, scores[token]);
            }
            float denominator = 0.0f;
            for (uint32_t token = 0; token < header.valid_tokens; ++token) {
                scores[token] = expf(scores[token] - maximum);
                denominator += scores[token];
            }
            for (uint32_t token = 0; token < header.valid_tokens; ++token) scores[token] /= denominator;
            for (uint32_t token = 0; token < header.valid_tokens; ++token) {
                score_indices[token].score = scores[token];
                score_indices[token].index = token;
            }
            qsort(score_indices, header.valid_tokens, sizeof(*score_indices),
                  sparf_score_index_compare);
            bool current_selected = false;
            for (uint32_t rank = 0; rank < header.top_k; ++rank) {
                tokens[(uint64_t)vector * header.top_k + rank] = score_indices[rank].index;
                current_selected |= score_indices[rank].index + 1 == header.valid_tokens;
            }
            uint32_t current = header.valid_tokens - 1;
            if (!current_selected) {
                uint32_t weakest = 0;
                for (uint32_t rank = 1; rank < header.top_k; ++rank) {
                    if (scores[tokens[(uint64_t)vector * header.top_k + rank]] <
                        scores[tokens[(uint64_t)vector * header.top_k + weakest]]) {
                        weakest = rank;
                    }
                }
                tokens[(uint64_t)vector * header.top_k + weakest] = current;
            }
            alpha[vector] = 0.0f;
            for (uint32_t rank = 0; rank < header.top_k; ++rank)
                alpha[vector] += scores[tokens[(uint64_t)vector * header.top_k + rank]];
        }
        backend_cuda_mark_host_dirty(job->mr_backend[7], alpha, vectors * sizeof(float));
        trace.approximate_model_ns = header.approximate_runtime_ns;
        if (timeline) sparf_trace_stamp(&timeline->computes[0].functional_end);
        trace.selected_channels = (uint64_t)vectors * header.top_r;
        trace.selected_tokens = (uint64_t)vectors * header.top_k;
        if (autonomous) {
            *modeled_runtime = header.approximate_runtime_ns;
            autonomous->resume = SPARF_RESUME_KEY_START;
            autonomous->wait = SPARF_WAIT_COMPUTE;
            goto suspend;
        }
        if (stage) {
            memcpy(saved_tokens, tokens, trace.selected_tokens * sizeof(*tokens));
            result = 0;
            goto out;
        }
    }

resume_key_start:
    if (!stage || stage == CEMU_SPARF_STAGE_TOKEN_READ) {
        GArray *key_spans = grouped ? NULL : sparf_token_spans(&header, tokens, false, tail.published_tokens);
        sparf_observe_begin(observation, SPARF_OBS_TOKEN_K_READ);
        if (timeline) {
            sparf_trace_stamp(&timeline->reads[1].submitted);
            timeline->reads[1].submitted.virtual_ns = model_cursor;
        }
        if (grouped) {
            if (!sparf_start_grouped_read(job, autonomous->group_session, &header, tokens,
                CEMU_SPARF_GROUP_TOKEN_K, 2, ring, model_cursor, &key_read,
                read_observations[1])) goto out;
        } else if (!sparf_start_read(job, ring, token_k_map, header.token_k_extents,
                              key_spans, 2, model_cursor, &key_read,
                              read_observations[1])) goto out;
        if (autonomous) {
            autonomous->read_start = model_cursor;
            autonomous->resume = SPARF_RESUME_KEY;
            autonomous->wait = SPARF_WAIT_IO;
            goto suspend;
        }
resume_key:
        if (!sparf_finish_read(&key_read, ring, &trace.token_k_read_model_ns)) goto out;
        sparf_observe_end(observation, SPARF_OBS_TOKEN_K_READ);
        model_cursor += trace.token_k_read_model_ns;
        if (autonomous) {
            autonomous->deadline = autonomous->read_start + trace.token_k_read_model_ns;
            autonomous->resume = SPARF_RESUME_EXACT;
            autonomous->wait = SPARF_WAIT_TIME;
            goto suspend;
        }
resume_exact:
        if (backend_cuda_prepare_host(job->mr_backend[2], job->args.mr_addr[2], job->args.mr_len[2]) != 0) goto out;
        for (uint32_t vector = 0; vector < vectors; ++vector) {
            uint32_t batch = vector / header.num_query_heads;
            uint32_t head = (vector % header.num_query_heads) / heads_per_kv;
            for (uint32_t rank = 0; rank < header.top_k; ++rank) {
                uint32_t selected = tokens[(uint64_t)vector * header.top_k + rank];
                if (!grouped && (selected >= tail.published_tokens || selected + 1 == header.valid_tokens)) {
                    uint64_t buffered_offset = selected >= tail.published_tokens
                        ? (uint64_t)(selected - tail.published_tokens) * kv_elements : current_offset;
                    memcpy((uint8_t *)job->args.mr_addr[2] +
                               ((uint64_t)vector * header.top_k + rank) * header.head_dim * element,
                           (uint8_t *)job->args.mr_addr[5] +
                               (buffered_offset + ((uint64_t)batch * header.num_kv_heads + head) * header.head_dim) * element,
                           header.head_dim * element);
                }
            }
        }
        backend_cuda_mark_host_dirty(job->mr_backend[2], job->args.mr_addr[2], job->args.mr_len[2]);
        if (stage) {
            result = 0;
            goto out;
        }
    }

    if (!stage || stage == CEMU_SPARF_STAGE_EXACT_V) {
        GArray *value_spans = grouped ? NULL : sparf_token_spans(&header, tokens, true, tail.published_tokens);
        sparf_observe_begin(observation, SPARF_OBS_V_OUTSTANDING);
        if (timeline) {
            sparf_trace_stamp(&timeline->reads[2].submitted);
            timeline->reads[2].submitted.virtual_ns = model_cursor;
        }
        if (grouped) {
            if (!sparf_start_grouped_read(job, autonomous->group_session, &header, tokens,
                CEMU_SPARF_GROUP_TOKEN_V, 3, ring, model_cursor, &value_read,
                read_observations[2])) goto out;
        } else if (!sparf_start_read(job, ring, token_v_map, header.token_v_extents,
                              value_spans, 3, model_cursor, &value_read,
                              read_observations[2])) goto out;

        const uint32_t exact_ranges[] = { 0, 2, 7, 8 };
        sparf_observe_begin(observation, SPARF_OBS_QK);
        if (timeline) sparf_trace_stamp(&timeline->computes[1].functional_start);
        uint64_t exact_result = run_sparf_phase(job, CEMU_SPARF_PHASE_EXACT_QK,
                                               exact_ranges, 4, &metadata);
        sparf_observe_end(observation, SPARF_OBS_QK);
        if (timeline) sparf_trace_stamp(&timeline->computes[1].functional_end);
        if (autonomous && exact_result != (uint64_t)-1) {
            autonomous->read_start = model_cursor;
            trace.exact_qk_model_ns = header.exact_qk_runtime_ns;
            *modeled_runtime = header.exact_qk_runtime_ns;
            autonomous->resume = SPARF_RESUME_VALUE;
            autonomous->wait = SPARF_WAIT_COMPUTE;
            goto suspend;
        }
        if (exact_result == (uint64_t)-1) goto out;
resume_value:
        if (!sparf_finish_read(&value_read, ring, &trace.token_v_read_model_ns) ||
            (!autonomous && exact_result == (uint64_t)-1)) goto out;
        sparf_observe_end(observation, SPARF_OBS_V_OUTSTANDING);
        if (backend_cuda_prepare_host(job->mr_backend[3], job->args.mr_addr[3],
                                      job->args.mr_len[3]) != 0 ||
            backend_cuda_prepare_host(job->mr_backend[4], job->args.mr_addr[4],
                                      job->args.mr_len[4]) != 0 ||
            (!grouped && backend_cuda_prepare_host(job->mr_backend[6], job->args.mr_addr[6],
                                      job->args.mr_len[6]) != 0)) goto out;
        for (uint32_t vector = 0; vector < vectors; ++vector) {
            uint32_t batch = vector / header.num_query_heads;
            uint32_t head = (vector % header.num_query_heads) / heads_per_kv;
            for (uint32_t rank = 0; rank < header.top_k; ++rank) {
                uint32_t selected = tokens[(uint64_t)vector * header.top_k + rank];
                if (!grouped && (selected >= tail.published_tokens || selected + 1 == header.valid_tokens)) {
                    uint64_t buffered_offset = selected >= tail.published_tokens
                        ? (uint64_t)(selected - tail.published_tokens) * kv_elements : current_offset;
                    memcpy((uint8_t *)job->args.mr_addr[3] +
                               ((uint64_t)vector * (header.top_k + 1) + rank) * header.head_dim * element,
                           (uint8_t *)job->args.mr_addr[6] +
                               (buffered_offset + ((uint64_t)batch * header.num_kv_heads + head) * header.head_dim) * element,
                           header.head_dim * element);
                }
            }
            memcpy((uint8_t *)job->args.mr_addr[3] +
                       ((uint64_t)vector * (header.top_k + 1) + header.top_k) * header.head_dim * element,
                   (uint8_t *)job->args.mr_addr[4] + (uint64_t)vector * header.head_dim * element,
                   header.head_dim * element);
        }
        backend_cuda_mark_host_dirty(job->mr_backend[3], job->args.mr_addr[3], job->args.mr_len[3]);
        trace.exact_qk_model_ns = header.exact_qk_runtime_ns;
        if (autonomous) {
            autonomous->deadline = autonomous->read_start + trace.token_v_read_model_ns;
            autonomous->resume = SPARF_RESUME_PV;
            autonomous->wait = SPARF_WAIT_TIME;
            goto suspend;
        }
        if (stage) {
            result = 0;
            goto out;
        }
    }
resume_pv:;
    const uint32_t pv_ranges[] = { 8, 3, 9 };
    sparf_observe_begin(observation, SPARF_OBS_PV);
    if (timeline) sparf_trace_stamp(&timeline->computes[2].functional_start);
    result = run_sparf_phase(job, CEMU_SPARF_PHASE_PV, pv_ranges, 3, &metadata);
    sparf_observe_end(observation, SPARF_OBS_PV);
    if (timeline) sparf_trace_stamp(&timeline->computes[2].functional_end);
    trace.pv_model_ns = header.pv_runtime_ns;

out:
    if (channel_read.request) sparf_finish_read(&channel_read, ring, &trace.channel_read_model_ns);
    if (key_read.request) sparf_finish_read(&key_read, ring, &trace.token_k_read_model_ns);
    if (value_read.request) sparf_finish_read(&value_read, ring, &trace.token_v_read_model_ns);
    sparf_clear_read(&channel_read);
    sparf_clear_read(&key_read);
    sparf_clear_read(&value_read);
    trace.total_model_ns = trace.channel_read_model_ns + trace.approximate_model_ns +
        trace.token_k_read_model_ns + MAX(trace.exact_qk_model_ns, trace.token_v_read_model_ns) +
        trace.pv_model_ns;
    /* Charge only this stage; the cumulative trace is not a new job runtime. */
    *modeled_runtime = trace.total_model_ns - previous_runtime;
    if (backend_cuda_prepare_host(job->mr_backend[10], job->args.mr_addr[10], sizeof(trace)) == 0) {
        memcpy(job->args.mr_addr[10], &trace, sizeof(trace));
        backend_cuda_mark_host_dirty(job->mr_backend[10], job->args.mr_addr[10], sizeof(trace));
    } else result = (uint64_t)-1;
    if (context) {
        context->trace = trace;
        context->completed_stage = result == (uint64_t)-1 ? CEMU_SPARF_STAGE_ERROR : stage;
        backend_cuda_mark_host_dirty(job->mr_backend[CEMU_SPARF_STAGE_CONTEXT_RANGE],
            context, sizeof(*context) + context->metadata_bytes + context->token_count * 4);
    }
    if (ring) femu_ring_free(ring);
    g_free(channels); g_free(tokens); g_free(scores); g_free(score_indices);
    g_free(query_values); g_free(channel_values); g_free(current_key_values);
    g_free(channel_indices);
    if (autonomous) {
        autonomous->resume = SPARF_RESUME_DONE;
        autonomous->result = result;
        autonomous->wait = result == (uint64_t)-1 ? SPARF_WAIT_NONE : SPARF_WAIT_COMPUTE;
        autonomous->ring = NULL;
        *modeled_runtime = result == (uint64_t)-1 ? 0 : header.pv_runtime_ns;
    }
    return result;

suspend:
    autonomous->ring = ring;
    autonomous->channels = channels;
    autonomous->tokens = tokens;
    autonomous->scores = scores;
    autonomous->score_indices = score_indices;
    autonomous->query_values = query_values;
    autonomous->channel_values = channel_values;
    autonomous->current_key_values = current_key_values;
    autonomous->channel_indices = channel_indices;
    autonomous->channel_read = channel_read;
    autonomous->key_read = key_read;
    autonomous->value_read = value_read;
    autonomous->trace = trace;
    return 0;
}

#endif
