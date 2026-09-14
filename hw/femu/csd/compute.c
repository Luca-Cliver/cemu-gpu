#include <dlfcn.h>
#include <math.h>
#include <pthread.h>
#include <stdlib.h>
#include "compute.h"
#include "memory.h"
#include "qemu/atomic.h"
#include <stdatomic.h>
#include "hw/femu/backend/backend.h"
#include "hw/femu/attention_workflow_abi.h"
#include "hw/femu/sparf_workflow_abi.h"
#include "hw/femu/inc/slab.h"
#include "hw/femu/nvme-def.h"
#include "hw/femu/nvme.h"
#include "hw/femu/param.h"
#include "sched.h"
#include "sysemu/cpus.h"
#include "sysemu/runstate.h"
#include "sysemu/cpu-timers.h"

#define MAX_PIND    1024
#define MAX_RSID    1024

static uint64_t monotonic_ns(void)
{
    struct timespec timestamp;
    clock_gettime(CLOCK_MONOTONIC, &timestamp);
    return (uint64_t)timestamp.tv_sec * 1000000000ULL + timestamp.tv_nsec;
}

static bool multiply_u64(uint64_t *value, uint64_t factor)
{
    if (!factor || *value > INT64_MAX / factor) {
        return false;
    }
    *value *= factor;
    return true;
}

static bool validate_attention_workflow(Program *program, uint64_t command,
                                        uint32_t numr, const long long *mr_len,
                                        const void *data, uint32_t length,
                                        FemuCtrl *controller)
{
    if (command != CEMU_ATTENTION_WORKFLOW_COMMAND) {
        return true;
    }
    if (controller->internal_bandwidth != 0) {
        femu_err("Attention workflow requires internal_bandwidth=0 while "
                 "functional execution freezes the virtual clock\n");
        return false;
    }
    if (program->is_indirect || program->type != PROGRAM_TYPE_SHARED_LIB ||
        (program->target != PROGRAM_TARGET_HOST &&
         program->target != PROGRAM_TARGET_CUDA_DEVPTR) ||
        numr != CEMU_ATTENTION_WORKFLOW_RANGES || !mr_len || !data ||
        length < sizeof(struct cemu_attention_workflow_header)) {
        return false;
    }

    struct cemu_attention_workflow_header header;
    memcpy(&header, data, sizeof(header));
    struct cemu_attention_phase_metadata *config = &header.attention;
    uint64_t extent_count = (uint64_t)header.key_extents + header.value_extents;
    if (header.version != CEMU_ATTENTION_WORKFLOW_VERSION ||
        header.reserved[0] || header.reserved[1] ||
        (header.flags & ~(CEMU_ATTENTION_WORKFLOW_TRACE |
                          CEMU_ATTENTION_WORKFLOW_SERIAL)) ||
        !header.key_extents || !header.value_extents ||
        header.key_extents > CEMU_ATTENTION_WORKFLOW_MAX_EXTENTS ||
        header.value_extents > CEMU_ATTENTION_WORKFLOW_MAX_EXTENTS ||
        length != sizeof(header) + extent_count *
                                  sizeof(struct cemu_attention_workflow_extent) ||
        config->version != CEMU_ATTENTION_PHASES_VERSION ||
        config->phase != CEMU_ATTENTION_PHASE_QK_SOFTMAX ||
        (config->dtype != CEMU_ATTENTION_PHASES_FLOAT16 &&
         config->dtype != CEMU_ATTENTION_PHASES_FLOAT32) ||
        !config->batch_size || !config->num_query_heads ||
        !config->num_kv_heads || !config->head_dim || !config->token_count ||
        !config->token_stride ||
        config->num_query_heads % config->num_kv_heads ||
        config->token_stride % 512 || !isfinite(config->scale) ||
        config->scale <= 0.0f) {
        return false;
    }

    uint64_t element_size = config->dtype == CEMU_ATTENTION_PHASES_FLOAT16 ? 2 : 4;
    uint64_t token_payload = config->batch_size;
    uint64_t query_size = config->batch_size;
    uint64_t storage_size = config->token_count;
    uint64_t probability_size = config->batch_size;
    if (!multiply_u64(&token_payload, config->num_kv_heads) ||
        !multiply_u64(&token_payload, config->head_dim) ||
        !multiply_u64(&token_payload, element_size) ||
        token_payload > config->token_stride ||
        !multiply_u64(&query_size, config->num_query_heads) ||
        !multiply_u64(&query_size, config->head_dim) ||
        !multiply_u64(&query_size, element_size) ||
        !multiply_u64(&storage_size, config->token_stride) ||
        !multiply_u64(&probability_size, config->num_query_heads) ||
        !multiply_u64(&probability_size, config->token_count) ||
        !multiply_u64(&probability_size, sizeof(float))) {
        return false;
    }
    uint64_t required[] = {
        query_size, storage_size, storage_size, probability_size,
        query_size, sizeof(struct cemu_attention_workflow_trace),
    };
    for (uint32_t index = 0; index < CEMU_ATTENTION_WORKFLOW_RANGES; ++index) {
        if (mr_len[index] < 0 || (uint64_t)mr_len[index] < required[index]) {
            return false;
        }
    }

    NvmeNamespace *nvm = nvme_find_namespace(controller, 1);
    if (!nvm || !nvm->backend) {
        return false;
    }
    const uint8_t lba_index = NVME_ID_NS_FLBAS_INDEX(nvm->id_ns.flbas);
    if (nvm->id_ns.lbaf[lba_index].lbads != 9) {
        return false;
    }
    const uint64_t nvm_blocks = nvm->backend->size / 512;
    const uint8_t *extent_data = (const uint8_t *)data + sizeof(header);
    uint64_t accumulated = 0;
    for (uint64_t index = 0; index < extent_count; ++index) {
        struct cemu_attention_workflow_extent extent;
        memcpy(&extent, extent_data + index * sizeof(extent), sizeof(extent));
        if (!extent.nlb || extent.nlb > 65535 || extent.reserved ||
            extent.slba >= nvm_blocks || extent.nlb > nvm_blocks - extent.slba) {
            return false;
        }
        accumulated += (uint64_t)extent.nlb * 512;
        if (index + 1 == header.key_extents || index + 1 == extent_count) {
            if (accumulated != storage_size) {
                return false;
            }
            accumulated = 0;
        }
    }
    return true;
}

static bool validate_sparf_workflow(Program *program, uint64_t command,
                                    uint32_t numr, const long long *mr_len,
                                    const void *data, uint32_t length,
                                    FemuCtrl *controller)
{
    if (command != CEMU_SPARF_WORKFLOW_COMMAND) {
        return true;
    }
    if (controller->internal_bandwidth != 0 || program->is_indirect ||
        program->type != PROGRAM_TYPE_SHARED_LIB ||
        (program->target != PROGRAM_TARGET_HOST &&
         program->target != PROGRAM_TARGET_CUDA_DEVPTR) ||
        numr != CEMU_SPARF_WORKFLOW_RANGES || !mr_len || !data ||
        length < sizeof(struct cemu_sparf_workflow_header)) {
        return false;
    }
    struct cemu_sparf_workflow_header header;
    memcpy(&header, data, sizeof(header));
    uint64_t extents = (uint64_t)header.token_k_extents +
                       header.channel_k_extents + header.token_v_extents;
    if (header.version != CEMU_SPARF_WORKFLOW_VERSION ||
        (header.flags & ~CEMU_SPARF_WORKFLOW_TRACE) ||
        header.reserved[0] || header.reserved[1] ||
        !header.token_k_extents || !header.channel_k_extents ||
        !header.token_v_extents ||
        header.token_k_extents > CEMU_SPARF_WORKFLOW_MAX_EXTENTS ||
        header.channel_k_extents > CEMU_SPARF_WORKFLOW_MAX_EXTENTS ||
        header.token_v_extents > CEMU_SPARF_WORKFLOW_MAX_EXTENTS ||
        length != sizeof(header) + extents * sizeof(struct cemu_sparf_workflow_extent) ||
        (header.dtype != CEMU_SPARF_FLOAT16 && header.dtype != CEMU_SPARF_FLOAT32) ||
        !header.batch_size || !header.num_query_heads || !header.num_kv_heads ||
        !header.head_dim || !header.max_seq_len || !header.valid_tokens ||
        header.valid_tokens > header.max_seq_len ||
        header.num_query_heads % header.num_kv_heads ||
        !header.top_r || header.top_r > header.head_dim ||
        !header.top_k || header.top_k > header.valid_tokens ||
        !header.token_head_stride || !header.token_batch_stride ||
        !header.token_layer_stride || !header.channel_stride ||
        !header.channel_head_stride || !header.channel_batch_stride ||
        !header.channel_layer_stride || !isfinite(header.scale) ||
        header.scale <= 0.0f) {
        return false;
    }
    uint64_t vectors = (uint64_t)header.batch_size * header.num_query_heads;
    uint64_t element = header.dtype == CEMU_SPARF_FLOAT16 ? 2 : 4;
    uint64_t required[] = {
        vectors * header.head_dim * element,
        vectors * header.top_r * header.valid_tokens * element,
        vectors * header.top_k * header.head_dim * element,
        vectors * (header.top_k + 1) * header.head_dim * element,
        vectors * header.head_dim * element,
        (uint64_t)header.batch_size * header.num_kv_heads * header.head_dim * element,
        (uint64_t)header.batch_size * header.num_kv_heads * header.head_dim * element,
        vectors * sizeof(float),
        vectors * (header.top_k + 1) * sizeof(float),
        vectors * header.head_dim * element,
        sizeof(struct cemu_sparf_workflow_trace),
    };
    for (uint32_t index = 0; index < CEMU_SPARF_WORKFLOW_RANGES; ++index) {
        if (mr_len[index] < 0 || (uint64_t)mr_len[index] < required[index]) {
            return false;
        }
    }
    NvmeNamespace *nvm = nvme_find_namespace(controller, 1);
    if (!nvm || !nvm->backend ||
        nvm->id_ns.lbaf[NVME_ID_NS_FLBAS_INDEX(nvm->id_ns.flbas)].lbads != 9) {
        return false;
    }
    const struct cemu_sparf_workflow_extent *map =
        (const struct cemu_sparf_workflow_extent *)((const uint8_t *)data + sizeof(header));
    uint64_t expected[] = {
        ((uint64_t)header.layer + 1) * header.token_layer_stride,
        ((uint64_t)header.layer + 1) * header.channel_layer_stride,
        ((uint64_t)header.layer + 1) * header.token_layer_stride,
    };
    uint32_t counts[] = { header.token_k_extents, header.channel_k_extents,
                          header.token_v_extents };
    uint64_t nvm_blocks = nvm->backend->size / 512;
    uint32_t map_index = 0;
    for (uint32_t file = 0; file < 3; ++file) {
        uint64_t covered = 0;
        for (uint32_t index = 0; index < counts[file]; ++index, ++map_index) {
            if (!map[map_index].nlb || map[map_index].nlb > 65535 ||
                map[map_index].reserved || map[map_index].slba >= nvm_blocks ||
                map[map_index].nlb > nvm_blocks - map[map_index].slba) {
                return false;
            }
            covered += (uint64_t)map[map_index].nlb * 512;
        }
        if (covered < expected[file]) {
            return false;
        }
    }
    return true;
}

static inline ComputeNamespace *compute_ns(NvmeNamespace *ns)
{
    return (ComputeNamespace *)ns->private;
}

struct csd_thread_arg {
    NvmeNamespace *ns;
    int id;
};

static void *compute_thread(void *arg);

/* baseline counters for measuring compute fraction */
static atomic_uint_fast64_t csd_total_compute_ns = 0;
static atomic_uint_fast64_t csd_total_job_ns = 0;
static atomic_uint_fast64_t csd_job_count = 0;
static const int CSD_BASELINE_PRINT_FREQ = 100;

typedef enum CemuComputeLogMode {
    CEMU_COMPUTE_LOG_OFF,
    CEMU_COMPUTE_LOG_PROGRESS,
    CEMU_COMPUTE_LOG_DETAIL,
} CemuComputeLogMode;

static CemuComputeLogMode cemu_compute_log_mode(void)
{
    static gsize initialized;
    static CemuComputeLogMode mode;

    if (g_once_init_enter(&initialized)) {
        const char *value = g_getenv("CEMU_COMPUTE_LOG");

        if (value != NULL &&
            (!value[0] || !g_ascii_strcasecmp(value, "0") ||
             !g_ascii_strcasecmp(value, "false") ||
             !g_ascii_strcasecmp(value, "no") ||
             !g_ascii_strcasecmp(value, "off"))) {
            mode = CEMU_COMPUTE_LOG_OFF;
        } else if (value != NULL &&
                   (!g_ascii_strcasecmp(value, "progress") ||
                    !g_ascii_strcasecmp(value, "summary"))) {
            mode = CEMU_COMPUTE_LOG_PROGRESS;
        } else {
            mode = CEMU_COMPUTE_LOG_DETAIL;
        }
        g_once_init_leave(&initialized, 1);
    }
    return mode;
}

static uint64_t cemu_compute_progress_interval(void)
{
    static gsize initialized;
    static uint64_t interval;

    if (g_once_init_enter(&initialized)) {
        const char *value = g_getenv("CEMU_COMPUTE_LOG_INTERVAL");
        char *end = NULL;

        interval = value ? g_ascii_strtoull(value, &end, 10) : 10000;
        if (interval == 0 || (value && (!end || *end))) {
            interval = 10000;
        }
        g_once_init_leave(&initialized, 1);
    }
    return interval;
}

struct ProgramInitArgs {
    uint32_t nsid;
    uint16_t pind;
};

static void init_program_slab(void *elem, void *arg)
{
    struct ProgramInitArgs *args = (struct ProgramInitArgs *)arg;
    Program *p = (Program *)elem;
    p->nsid = args->nsid;
    p->pind = args->pind++;
    p->state = PROGRAM_STATE_INVALID;
    pthread_mutex_init(&p->lock, NULL);
}

static void init_mrs_slab(void *elem, void *arg)
{
    int *rsid = (int *)arg;
    MemoryRangeSet *mrs = (MemoryRangeSet *)elem;
    mrs->rsid = (*rsid)++;
    mrs->mr = NULL;
    mrs->numr = 0;
    pthread_spin_init(&mrs->lock, 0);
}

static void compute_init(NvmeNamespace *ns, Error **errp)
{
    ComputeParams *param = ns->params;
    ComputeNamespace *cns = malloc(sizeof(ComputeNamespace));
    ns->private = cns;
    cns->ns = ns;
    cns->params = param;

    // program slab init
    struct ProgramInitArgs init_arg;
    init_arg.nsid = ns->id;
    init_arg.pind = 1;
    slab_init(&cns->programs, sizeof(Program), MAX_PIND, init_program_slab, &init_arg);

    // rsid slab init
    int rsid = 1;
    slab_init(&cns->mrs, sizeof(MemoryRangeSet), MAX_RSID, init_mrs_slab, &rsid);

    cns->to_csd = g_malloc0(sizeof(struct rte_ring *) * param->nr_thread);
    cns->compute_threads = g_malloc0(sizeof(QemuThread) * param->nr_thread);
    struct csd_thread_arg* args = g_malloc0(sizeof(struct csd_thread_arg) * param->nr_thread);
    for (int i = 0; i < param->nr_thread; i++) {
        cns->to_csd[i] = femu_ring_create(FEMU_RING_TYPE_SP_SC, FEMU_MAX_INF_REQS);
        char name[32];
        sprintf(name, "FEMU-CSD-Thread-%d", i);
        args[i].ns = ns;
        args[i].id = i;
        qemu_thread_create(&cns->compute_threads[i], name, compute_thread, &args[i],
                           QEMU_THREAD_JOINABLE);
    }

    // init scheduler
    sched_init(ns);
}

static uint16_t memory_range_set_management(NvmeNamespace *ns, NvmeCmd *cmd, NvmeCqe *cqe)
{
    FemuCtrl *n = ns->ctrl;
    NvmeMemoryRangeSetManageCmd *manage = (NvmeMemoryRangeSetManageCmd *)cmd;
    uint16_t rsid = le16_to_cpu(manage->rsid);
    uint32_t sel = manage->sel;
    uint32_t numr = manage->numr;
    uint64_t prp1 = le64_to_cpu(manage->prp1);
    uint64_t prp2 = le64_to_cpu(manage->prp2);

    femu_debug("memory_range_set_management: sel %u, rsid %u, numr %u, prp1 %lx, prp2 %lx\n", sel, rsid, numr, prp1, prp2);

    if (sel == 0) {
        // add memory range set
        // read memory range descriptors
        if (rsid != 0) {
            femu_err("memory_range_set_management: rsid %u != 0 when sel == 0!\n", rsid);
            return NVME_INVALID_FIELD;
        }
        if (numr == 0 || numr >= 128) {
            femu_err("memory_range_set_management: numr %u not supported!\n", numr);
            return NVME_INVALID_FIELD;
        }

        NvmeMemoryRange mr[128];
        if (dma_write_prp(n, (void *)mr, numr * sizeof(NvmeMemoryRange), prp1, prp2)) {
            femu_err("memory_range_set_management: dma_write_prp error\n");
            return NVME_DNR;
        }

        MemoryRangeSet *mrs = slab_alloc(memory_range_set_slab(ns), 1);
        if (mrs == NULL) {
            return NVME_MR_SET_EXCEEDED;
        }
        mrs->in_use = false;
        mrs->numr = numr;
        mrs->mr = g_malloc0(sizeof(MemoryRange) * numr);
        mrs->mr_addr = g_malloc0(sizeof(void *) * numr);
        mrs->mr_len = g_malloc0(sizeof(long long *) * numr);
        for (int i = 0; i < numr; i++) {
            uint32_t nsid = le32_to_cpu(mr[i].nsid);
            uint32_t len = le32_to_cpu(mr[i].len);
            uint64_t sb = le64_to_cpu(mr[i].sb);
            NvmeNamespace *mns = nvme_find_namespace(n, nsid);
            if (mns == NULL) {
                free(mrs->mr);
                slab_free(memory_range_set_slab(ns), mrs, 1);
                return NVME_INVALID_MEMORY_NS;
            }
            mrs->mr[i].addr = backend_addr(mns->backend, sb);
            mrs->mr[i].nsid = nsid;
            mrs->mr[i].len = len;
            mrs->mr[i].sb = sb;
            mrs->mr[i].backend = mns->backend;
            mrs->mr_addr[i] = mrs->mr[i].addr;
            mrs->mr_len[i] = mrs->mr[i].len;
        }

        femu_debug("create mrs rsid %d, mrs %p, numr %d\n", mrs->rsid, mrs, mrs->numr);
        cqe->n.result = mrs->rsid;
    } else if (sel == 1) {
        // remove memory range set
        if (rsid == 0 || rsid > MAX_RSID) {
            femu_err("memory_range_set_management: rsid %u invalid!\n", rsid);
            return NVME_INVALID_MR_SET_ID;
        }
        pthread_spin_lock(&memory_range_set_slab(ns)->lock);
        MemoryRangeSet *mrs = memory_range_set_get(ns, rsid);
        if (mrs == NULL || mrs->numr == 0) {
            pthread_spin_unlock(&memory_range_set_slab(ns)->lock);
            femu_err("memory_range_set_management: rsid %u not found!\n", rsid);
            return NVME_INVALID_FIELD;
        }
        femu_debug("free mrs rsid %d, mrs %p, numr %d\n", rsid, mrs, mrs->numr);
        if (mrs->in_use) {
            pthread_spin_unlock(&memory_range_set_slab(ns)->lock);
            return NVME_MR_SET_IN_USE;
        }
        uint16_t released_numr = mrs->numr;
        mrs->numr = 0;
        pthread_spin_unlock(&memory_range_set_slab(ns)->lock);
        int sync_status = 0;
        for (uint32_t i = 0; i < released_numr; i++) {
            if (backend_cuda_drop_device_ptr(mrs->mr[i].backend,
                                             mrs->mr[i].addr,
                                             mrs->mr[i].len) != 0) {
                sync_status = -1;
            }
        }
        free(mrs->mr);
        free(mrs->mr_addr);
        free(mrs->mr_len);
        slab_free(memory_range_set_slab(ns), mrs, 1);
        if (sync_status != 0) {
            return NVME_DNR;
        }
        cqe->n.result = 0;
    } else {
        femu_err("memory_range_set_management: sel %u not supported!\n", sel);
        return NVME_INVALID_FIELD;
    }

    return NVME_SUCCESS;
}


static uint16_t parse_program(Program *program, char **csf_path, char **csf_name)
{
    int csf_name_len = strlen(program->code);
    if (csf_name_len >= MIN(4096, program->size)) {
        femu_err("load_shared_lib: shared library file name exceeds 4096!\n");
        return NVME_INVALID_PROGRAM_DATA;
    }

    int program_name_len = strlen(program->code + csf_name_len + 1);
    if (program_name_len >= 4096) {
        femu_err("load_shared_lib: shared library program function name exceeds 4096!\n");
        return NVME_INVALID_PROGRAM_DATA;
    }
    if (program_name_len + csf_name_len + 2 > program->size) {
        femu_err("load_shared_lib: shared library file name and program function name format error!\n");
        return NVME_INVALID_PROGRAM_DATA;
    }

    char *path = program->code;
    if (path[0] == '.') {
        // path relative to CEMU/tests/cemu
        g_autofree char *file_dir = g_path_get_dirname(__FILE__);
        g_autofree char *current_dir = realpath(file_dir, NULL);
        g_autofree char *src_dir = g_build_filename(current_dir, "../../../tests/cemu", NULL);
        g_autofree char *tmp_path = g_build_filename(src_dir, path, NULL);
        path = g_strdup(tmp_path);
    }

    *csf_path = path;
    *csf_name = program->code + csf_name_len + 1;
    femu_debug("parse_program: csf_path %s, csf_name %s\n", *csf_path, *csf_name);

    return NVME_SUCCESS;
}

/*
 * user data format for shared library:
 * | so_name | csf_name |
 */
static uint16_t load_shared_lib(Program *program)
{
    program->shared_lib.jit_fn = NULL;

    char *so_name;
    char *program_name;

    uint16_t ret = parse_program(program, &so_name, &program_name);
    if (ret != NVME_SUCCESS) {
        return ret;
    }

    void *dl_handle = NULL;
    femu_debug("load .so: %s\n", so_name);
    dl_handle = dlopen(so_name, RTLD_NOW);
    if (dl_handle == NULL) {
        femu_err("load_shared_lib: dlopen(%s) error: %s\n", so_name, dlerror());
        return NVME_INVALID_PROGRAM_DATA;
    }

    program->shared_lib.jit_fn = dlsym(dl_handle, program_name);
    if (program->shared_lib.jit_fn == NULL) {
        femu_err("load_shared_lib: dlsym(%s) error: %s\n", program_name, dlerror());
        return NVME_INVALID_PROGRAM_DATA;
    }
    program->shared_lib.so_handle = dl_handle;

    femu_debug("load_shared_lib: lib %s, program %s\n", so_name, program_name);

    return NVME_SUCCESS;
}

// user data is eBPF bytecode
static uint16_t load_ubpf(Program *program, int jit)
{
    char *ebpf_path;
    char *ebpf_name;
    ssize_t ret = parse_program(program, &ebpf_path, &ebpf_name);
    if (ret != NVME_SUCCESS) {
        return ret;
    }

    // read eBPF code from file
    int ebpf_fd = open(ebpf_path, O_RDONLY);
    if (ebpf_fd == -1) {
        perror("open");
        exit(1);
    }
    long program_size = lseek(ebpf_fd, 0, SEEK_END);
    lseek(ebpf_fd, 0, SEEK_SET);
    char *ebpf_code = aligned_alloc(4096, program_size);
    ret = read(ebpf_fd, ebpf_code, program_size);
    if (ret != program_size) {
        perror("read");
        exit(1);
    }
    close(ebpf_fd);

    char *errmsg = NULL;
    program->ebpf.jit_fn = NULL;

    program->ebpf.vm = ubpf_create();
    if (program->ebpf.vm == NULL) {
        femu_err("load_ubpf: ubpf_create error\n");
        return NVME_INVALID_PROGRAM_DATA;
    }

    if (ubpf_load_elf(program->ebpf.vm, ebpf_code, program_size, &errmsg)) {
        femu_err("load_ubpf: ubpf_load_elf error: %s\n", errmsg);
        return NVME_INVALID_PROGRAM_DATA;
    }

    if (jit) {
        program->ebpf.jit_fn = ubpf_compile(program->ebpf.vm, &errmsg);
        if (program->ebpf.jit_fn == NULL) {
            femu_err("load_ubpf: ubpf_compile error: %s\n", errmsg);
            return NVME_INVALID_PROGRAM_DATA;
        }
    }

    femu_debug("load_ubpf: downloaded ubpf success\n");
    return NVME_SUCCESS;
}

static uint16_t unload_shared_lib(Program *program)
{
    if (program->shared_lib.so_handle == NULL) {
        femu_err("unload_shared_lib: shared_lib already unloaded!\n");
        return NVME_SUCCESS;
    }
    int ret = dlclose(program->shared_lib.so_handle);
    if (ret != 0) {
        femu_err("unload_shared_lib: dlclose error: %s\n", dlerror());
        return NVME_DNR;
    }
    program->shared_lib.jit_fn = NULL;
    program->shared_lib.so_handle = NULL;
    femu_debug("unload_shared_lib: shared_lib unloaded!\n");
    return NVME_SUCCESS;
}

static uint16_t unload_ubpf(Program *program)
{
    if (program->ebpf.vm == NULL) {
        femu_err("unload_ubpf: ebpf already unloaded!\n");
        return NVME_SUCCESS;
    }
    ubpf_destroy(program->ebpf.vm);
    program->ebpf.vm = NULL;
    program->ebpf.jit_fn = NULL;
    femu_debug("unload_ubpf: ebpf unloaded!\n");
    return NVME_SUCCESS;
}

static uint16_t load_program(NvmeNamespace *ns, NvmeCmd *cmd, NvmeRequest *req)
{
    FemuCtrl *n = ns->ctrl;
    NvmeLoadProgramCmd *load = (NvmeLoadProgramCmd *)cmd;
    int ptype = load->ptype;
    int sel = load->sel;
    int pit = load->pit;
    int jit = load->jit;
    int indirect = load->indirect;
    int target = load->target;
    uint16_t runtime_scale = le16_to_cpu(load->runtime_scale);
    uint32_t runtime = le32_to_cpu(load->runtime);
    uint16_t pind = le16_to_cpu(load->pind);
    uint32_t psize = le32_to_cpu(load->psize);
    uint32_t numb = le32_to_cpu(load->numb);
    uint32_t loff = le32_to_cpu(load->loff);    // load offset
    uint64_t pid = le64_to_cpu(load->pid);
    uint64_t prp1 = le64_to_cpu(load->prp1);
    uint64_t prp2 = le64_to_cpu(load->prp2);

    femu_debug("load_program: sel %d, indirect %d, target %d, runtime %u, runtime_scale %f, ptype %d, pind %d, psize %d, numb %d, loff %d, pid %ld, prp1 %lx, prp2 %lx\n",
               sel, indirect, target, runtime, runtime_scale / 10.0, ptype, pind, psize, numb, loff, pid, prp1, prp2);

    // get program
    if (pind == 0 || pind > MAX_PIND) {
        femu_err("load_program: pind %u invalid!\n", pind);
        return NVME_INVALID_PIND;
    }
    Program *program = program_get(ns, pind);

    pthread_mutex_lock(&program->lock);
    if (sel == 0) {
        // load program
        if (qatomic_load_acquire(&program->jobs_running) != 0) {
            pthread_mutex_unlock(&program->lock);
            femu_err("load_program: program %u is running (jobs_running %d)!\n", pind, qatomic_load_acquire(&program->jobs_running));
            return NVME_PROGRAM_IN_USE;
        }

        if (program->state == PROGRAM_STATE_ACTIVATED ||
            program->state == PROGRAM_STATE_LOADED) {
            femu_debug("load_program: program %u already activated!\n", pind);
        }

        if (loff == 0) {
            // first time load
            if (pit == 1) {
                program->pid = pid;
            }
            if (ptype >= PROGRAM_TYPE_INVALID) {
                pthread_mutex_unlock(&program->lock);
                femu_err("load_program: program %u type %u not supported!\n", pind, ptype);
                return NVME_INVALID_PTYPE;
            }
            if (target >= PROGRAM_TARGET_INVALID) {
                pthread_mutex_unlock(&program->lock);
                femu_err("load_program: program %u target %u not supported!\n", pind, target);
                return NVME_INVALID_FIELD;
            }
            program->size = psize;
            program->type = ptype;
            program->target = target;
            qatomic_set(&program->jobs_running, 0);
            program->code = malloc(psize);
            program->state = PROGRAM_STATE_LOADING;
            program->runtime = runtime;
            program->runtime_scale = runtime_scale / 10.0;
            program->is_indirect = indirect;
        } else {
            // loff != 0
            if (psize != program->size) {
                pthread_mutex_unlock(&program->lock);
                femu_err("load_program: program %u size mismatch!\n", pind);
                return NVME_INVALID_FIELD;
            }
            if (pit == 1 && pid != program->pid) {
                pthread_mutex_unlock(&program->lock);
                femu_err("load_program: program %u pid mismatch!\n", pind);
                return NVME_INVALID_FIELD;
            }
            if (ptype != program->type) {
                pthread_mutex_unlock(&program->lock);
                femu_err("load_program: program %u type mismatch!\n", pind);
                return NVME_INVALID_PTYPE;
            }
        }

        // read program code
        if (dma_write_prp(n, program->code + loff, numb, prp1, prp2)) {
            pthread_mutex_unlock(&program->lock);
            femu_err("load_program: dma_write_prp error\n");
            return NVME_DNR;
        }
        program->load_size += numb;

        int ret = 0;
        if (program->load_size == program->size) {
            // program has transferred, start loading
            switch (program->type) {
            case PROGRAM_TYPE_SHARED_LIB:
                ret = load_shared_lib(program);
                break;
            case PROGRAM_TYPE_EBPF:
                ret = load_ubpf(program, jit);
                break;
            default:
                pthread_mutex_unlock(&program->lock);
                femu_err("load_program: program %u type %u not supported!\n", pind, program->type);
                return NVME_INVALID_PTYPE;
            }
            if (ret) {
                pthread_mutex_unlock(&program->lock);
                femu_err("load_program: load program error\n");
                return ret;
            }
            // TODO: error handle, delete program
            program->state = PROGRAM_STATE_LOADED;
        }
    } else {
        // unload program
        // TODO add jobs_running check
        if (program->state == PROGRAM_STATE_ACTIVATED) {
            femu_err("unload program: program %u is activated!\n", pind);
        }
        program->state = PROGRAM_STATE_INVALID;
        free(program->code);
        if (program->type == PROGRAM_TYPE_SHARED_LIB) {
            unload_shared_lib(program);
        } else if (program->type == PROGRAM_TYPE_EBPF) {
            unload_ubpf(program);
        }
        femu_debug("unload program: program %u unloaded!\n", pind);
    }
    pthread_mutex_unlock(&program->lock);
    return NVME_SUCCESS;
}

static uint16_t program_activation(NvmeNamespace *ns, NvmeCmd *cmd, NvmeCqe *cqe)
{
    NvmeProgramActivationCmd *activation = (NvmeProgramActivationCmd *)cmd;
    uint16_t pind = le16_to_cpu(activation->pind);
    int sel = activation->sel;

    femu_debug("program_activation: sel %d, pind %d\n", sel, pind);

    // get program
    if (pind == 0 || pind > MAX_PIND) {
        femu_err("program_activation: pind %u invalid!\n", pind);
        return NVME_INVALID_PIND;
    }
    Program *program = program_get(ns, pind);

    pthread_mutex_lock(&program->lock);
    if (sel == 0) {
        // deactivate program
        // TODO fix jobs_running when cu>1
        // if (qatomic_load_acquire(&program->jobs_running) > 0) {
        //     pthread_mutex_unlock(&program->lock);
        //     femu_err("program deactivation: program %u is running (%d)!\n", pind, qatomic_load_acquire(&program->jobs_running));
        //     return NVME_PROGRAM_IN_USE;
        // }
        program->state = PROGRAM_STATE_LOADED;
        femu_debug("program deactivation: program %u deactivated!\n", pind);
    } else if (sel == 1) {
        // activate program
        if (program->state != PROGRAM_STATE_LOADED) {
            pthread_mutex_unlock(&program->lock);
            femu_err("program_activation: program %u not loaded!\n", pind);
            return NVME_INVALID_PROGRAM_DATA;
        }
        program->state = PROGRAM_STATE_ACTIVATED;
    } else {
        pthread_mutex_unlock(&program->lock);
        femu_err("program_activation: sel %u not supported!\n", sel);
        return NVME_INVALID_FIELD;
    }
    pthread_mutex_unlock(&program->lock);
    return NVME_SUCCESS;
}

// Do parameter check and preparation for program execution, the actual
// execution is done in run_on_host()
static uint16_t program_execute(NvmeNamespace *ns, NvmeCmd *cmd, NvmeRequest *req)
{
    FemuCtrl *n = ns->ctrl;
    NvmeProgramExecuteCmd *exec = (NvmeProgramExecuteCmd *)cmd;
    uint16_t pind = le16_to_cpu(exec->pind);
    uint16_t rsid = le16_to_cpu(exec->rsid);
    uint32_t numr = le32_to_cpu(exec->numr);
    uint32_t dlen = le32_to_cpu(exec->dlen);
    uint64_t cparam1 = le64_to_cpu(exec->cparam1);
    uint64_t cparam2 = le64_to_cpu(exec->cparam2);
    uint64_t prp1 = le64_to_cpu(exec->prp1);
    uint64_t prp2 = le64_to_cpu(exec->prp2);
    uint32_t chunk_nlb = le32_to_cpu(exec->chunk_nlb) + 1;
    uint32_t runtime = le32_to_cpu(exec->runtime);

    femu_debug("program_execute: stime %lu, pind %u, rsid %u, numr %u, dlen %u, "
               "cparam1 %lx, cparam2 %lx, chunk_nlb %d, group %d, prp1 %lx, prp2 %lx\n",
               req->stat.stime, pind, rsid, numr, dlen, cparam1, cparam2, chunk_nlb,
               le32_to_cpu(exec->group), prp1, prp2);

    // get program
    if (pind == 0 || pind > MAX_PIND) {
        femu_err("program_execute: pind %u invalid!\n", pind);
        return NVME_INVALID_PIND;
    }
    if (numr && dlen < sizeof(NvmeMemoryRange) * numr) {
        femu_err("program_execute: dlen %u less than 32 * numr(%d)!\n", dlen, numr);
        return NVME_INVALID_FIELD;
    }

    Program *program = program_get(ns, pind);
    if (program->state != PROGRAM_STATE_ACTIVATED) {
        femu_err("program_execute: program %u not activated!\n", pind);
        return NVME_PROGRAM_NOT_ACTIVATED;
    }
    if (program->is_indirect && numr == 0) {
        femu_err("program_execute: indirect program %u doesn't provide lba list!\n", pind);
        return NVME_INVALID_FIELD;
    }
    if (program->is_indirect && rsid == 0) {
        femu_err("program_execute: indirect program %u doesn't provide rsid!\n", pind);
        return NVME_INVALID_FIELD;
    }
    // if (program->is_indirect && numr * sizeof(NvmeCopyFormat) < dlen) {
    //     femu_err("program_execute: indirect program %u incorrect dlen or numr!\n", pind);
    //     return NVME_INVALID_FIELD;
    // }
    if (!program->is_indirect && numr == 0 && rsid == 0) {
        femu_err("program_execute: numr %u and rsid %u is zero!\n", numr, rsid);
        return NVME_INVALID_FIELD;
    }
    if (program->is_indirect && dlen < 16) {
        femu_err("program_execute: indirect program %u dlen %u less than 14!\n", pind, dlen);
        return NVME_INVALID_FIELD;
    }

    // read data buffer
    char *data_buffer = NULL;
    if (dlen) {
        data_buffer = malloc(dlen); // free in sched.c:job_finish()
        if (dma_write_prp(n, (void *)data_buffer, dlen, prp1, prp2)) {
            femu_err("program_execute: dma_write_prp error, dlen %u\n", dlen);
            return NVME_DNR;
        }
    }
    req->data_buffer = data_buffer;

    // get all memory range
    void **mr_addr = NULL;
    long long *mr_len = NULL;
    SsdBackend **mr_backend = NULL;
    bool owns_mr_arrays = !program->is_indirect && numr != 0;
    if (program->is_indirect || numr == 0) {
        // memory range set is in rsid
        if (rsid == 0 || rsid > MAX_RSID) {
            femu_err("program_execute: rsid %d invalid!\n", rsid);
            return NVME_INVALID_MR_SET_ID;
        }
        MemoryRangeSet *mrs = memory_range_set_get(ns, rsid);
        mr_addr = mrs->mr_addr;
        mr_len = mrs->mr_len;
        mr_backend = g_malloc0(sizeof(SsdBackend *) * mrs->numr);
        for (uint32_t i = 0; i < mrs->numr; i++) {
            mr_backend[i] = mrs->mr[i].backend;
        }

        if (program->is_indirect) {
            req->mem_ctrl = n;
            req->nr_sres = 0;
            req->sres = NULL;
            req->sdaddr_backend = NULL;

            femu_debug("program_execute: data_buffer %p, dlen %u\n", data_buffer, dlen);
            // parse indirect task from data buffer
            IndirectTask *task = &req->indirect_task;
            task->raw_data_buffer = data_buffer;
            int *raw_task = (int*)data_buffer;
            task->nr_concurrent_chunks = raw_task[0];
            task->destination = raw_task[1];
            task->nr_total_input_cf2 = raw_task[2];
            task->nr_total_output_cf2 = raw_task[3];
            task->chunk_nlb = chunk_nlb;
            femu_debug("program_execute: task->nr_concurrent_chunks %d, task->destination %d, task->chunk_nlb %d, task->nr_total_input_cf2 %d, task->nr_total_output_cf2 %d\n", task->nr_concurrent_chunks, task->destination, task->chunk_nlb, task->nr_total_input_cf2, task->nr_total_output_cf2);

            // check parameters
            if (task->nr_concurrent_chunks == 0) {
                femu_err("program_execute: indirect program %u nr_concurrent_chunks is 0!\n", pind);
                return NVME_INVALID_FIELD;
            }
            if (task->destination != 0 && task->destination != 1) {
                femu_err("program_execute: indirect program %u destination %d not supported!\n", pind, task->destination);
                return NVME_INVALID_FIELD;
            }
            if (task->nr_total_input_cf2 == 0) {
                femu_err("program_execute: indirect program %u nr_total_input_cf2 is 0!\n", pind);
                return NVME_INVALID_FIELD;
            }
            if (task->destination == 1 && task->nr_total_output_cf2 == 0) {
                femu_err("program_execute: indirect program %u destination is 1 but nr_total_output_cf2 is 0!\n", pind);
                return NVME_INVALID_FIELD;
            }
            if (task->chunk_nlb == 0) {
                femu_err("program_execute: indirect program %u chunk_nlb is 0!\n", pind);
                return NVME_INVALID_FIELD;
            }
            if (mrs->numr < task->nr_concurrent_chunks || mrs->numr % task->nr_concurrent_chunks != 0) {
                femu_err("program_execute: indirect program %u memory range set numr %u not divisible by nr_concurrent_chunks %d!\n", pind, mrs->numr, task->nr_concurrent_chunks);
                return NVME_INVALID_FIELD;
            }
            int required_dlen = 16 + task->nr_concurrent_chunks * sizeof(int) + task->nr_total_input_cf2 * sizeof(NvmeCopyFormat);
            if (task->destination == 1) {
                required_dlen += task->nr_concurrent_chunks * sizeof(int) + task->nr_total_output_cf2 * sizeof(NvmeCopyFormat);
            }
            if (dlen < required_dlen) {
                femu_err("program_execute: indirect program %u data buffer (dlen %u) too short, required %d!\n", pind, dlen, required_dlen);
                return NVME_INVALID_FIELD;
            }

            task->nr_input_cf2 = raw_task + 4;
            for (int i = 0; i < task->nr_concurrent_chunks; i++) {
                femu_debug("program_execute: task->nr_input_cf2[%d] %d\n", i, task->nr_input_cf2[i]);
            }
            if (task->destination == 1) {
                // output in NVM
                task->nr_output_cf2 = task->nr_input_cf2 + task->nr_concurrent_chunks;
                data_buffer = (void*)(task->nr_output_cf2 + task->nr_concurrent_chunks);
                for (int i = 0; i < task->nr_concurrent_chunks; i++) {
                    femu_debug("program_execute: task->nr_output_cf2[%d] %d\n", i, task->nr_output_cf2[i]);
                }
            } else {
                // output in FDM
                task->nr_output_cf2 = NULL;
                data_buffer = (void*)(task->nr_input_cf2 + task->nr_concurrent_chunks);
            }
            femu_debug("program_execute: data_buffer %p, dlen %u\n", data_buffer, dlen);
            int total_input_cf2 = 0;
            int total_output_cf2 = 0;
            task->nr_total_nlb = 0;
            task->nr_total_output_nlb = 0;
            task->nr_output_nlb = malloc(sizeof(int) * task->nr_concurrent_chunks);
            task->nr_finished_nlb = malloc(sizeof(int) * task->nr_concurrent_chunks);
            task->nr_finished_output_nlb = malloc(sizeof(int) * task->nr_concurrent_chunks);
            task->iter_finished = malloc(sizeof(int) * task->nr_concurrent_chunks);
            for (int i = 0; i < task->nr_concurrent_chunks; i++) {
                total_input_cf2 += task->nr_input_cf2[i];
                task->nr_total_nlb += task->chunk_nlb;
                task->nr_output_nlb[i] = 0;
                task->nr_finished_nlb[i] = 0;
                task->nr_finished_output_nlb[i] = 0;
                task->iter_finished[i] = 0;
                if (task->destination == 1) {
                    total_output_cf2 += task->nr_output_cf2[i];
                }
            }

            if (total_input_cf2 != task->nr_total_input_cf2) {
                femu_err("program_execute: indirect program %u nr_total_input_cf2 mismatch, got %d, expected %d!\n", pind, total_input_cf2, task->nr_total_input_cf2);
                return NVME_INVALID_FIELD;
            }
            if (task->destination == 1 && total_output_cf2 != task->nr_total_output_cf2) {
                femu_err("program_execute: indirect program %u nr_total_output_cf2 mismatch, got %d, expected %d!\n", pind, total_output_cf2, task->nr_total_output_cf2);
                return NVME_INVALID_FIELD;
            }

            // generate per-chunk sre_iter
            task->iter_in = malloc(sizeof(struct sre_iter) * task->nr_concurrent_chunks);
            task->iter_out = malloc(sizeof(struct sre_iter) * task->nr_concurrent_chunks);
            memset(task->iter_in, 0, sizeof(struct sre_iter) * task->nr_concurrent_chunks);
            memset(task->iter_out, 0, sizeof(struct sre_iter) * task->nr_concurrent_chunks);
            NvmeCopyFormat2 *sres = (NvmeCopyFormat2 *)data_buffer;
            NvmeCopyFormat2 *sres_out = sres + task->nr_total_input_cf2;
            int max_sres_per_chunk = 0;
            for (int i = 0; i < task->nr_concurrent_chunks; i++) {
                task->iter_in[i].sres = sres;
                sres += task->nr_input_cf2[i];
                task->iter_in[i].nr_sres = task->nr_input_cf2[i];
                task->iter_in[i].nlb = task->chunk_nlb;
                max_sres_per_chunk = MAX(max_sres_per_chunk, task->nr_input_cf2[i]);
                if (task->destination == 1) {
                    task->iter_out[i].sres = sres_out;
                    sres_out += task->nr_output_cf2[i];
                    task->iter_out[i].nr_sres = task->nr_output_cf2[i];
                    task->iter_out[i].nlb = task->chunk_nlb;
                    max_sres_per_chunk = MAX(max_sres_per_chunk, task->nr_output_cf2[i]);
                }
            }
            femu_debug("program_execute: data_buffer %p, dlen %u\n", data_buffer, dlen);
            data_buffer += (task->nr_total_input_cf2 + task->nr_total_output_cf2) * sizeof(NvmeCopyFormat);
            dlen -= (data_buffer - task->raw_data_buffer);
            req->data_buffer = data_buffer;

            if (mrs->numr < 2) {
                femu_err("program_execute: indirect program %u memory range set numr %u less than 2!\n", pind, mrs->numr);
                return NVME_INVALID_FIELD;
            }

            req->sdaddr_backend = mr_backend[1];

            // mr[1] is input buffer
            if (chunk_nlb * 512 > mr_len[1]) {
                femu_err("program_execute: indirect program %u input buffer size %lld less than chunk_nlb %d!\n", pind, mr_len[1], chunk_nlb);
                return NVME_INVALID_FIELD;
            }
            req->is_write = 0;
        }
        numr = mrs->numr;
    } else {
        // memory range set is in data buffer
        if (rsid != 0) {
            femu_err("program_execute: rsid %d != 0 when numr !=0!\n", rsid);
            return NVME_INVALID_FIELD;
        }
        if (data_buffer == NULL) {
            return NVME_INVALID_FIELD;
        }
        NvmeMemoryRange *mrs = (NvmeMemoryRange *)data_buffer;
        mr_addr = malloc(sizeof(void *) * numr);
        mr_len = malloc(sizeof(long long) * numr);
        mr_backend = malloc(sizeof(SsdBackend *) * numr);
        for (int i = 0; i < numr; i++) {
            uint32_t nsid = le32_to_cpu(mrs[i].nsid);
            uint32_t len = le32_to_cpu(mrs[i].len);
            uint64_t sb = le64_to_cpu(mrs[i].sb);
            NvmeNamespace *mns = nvme_find_namespace(n, nsid);
            void *addr = backend_addr(mns->backend, sb);
            mr_addr[i] = addr;
            mr_len[i] = len;
            mr_backend[i] = mns->backend;
        }
        data_buffer += sizeof(NvmeMemoryRange) * numr;
        dlen -= sizeof(NvmeMemoryRange) * numr;
    }

    if (!validate_attention_workflow(program, cparam1, numr, mr_len,
                                     dlen ? data_buffer : NULL, dlen, n)) {
        femu_err("program_execute: invalid Attention workflow metadata or ranges\n");
        if (!program->is_indirect) {
            if (owns_mr_arrays) {
                free(mr_addr);
                free(mr_len);
            }
            free(mr_backend);
            free(req->data_buffer);
            req->data_buffer = NULL;
        }
        return NVME_INVALID_FIELD;
    }
    if (!validate_sparf_workflow(program, cparam1, numr, mr_len,
                                dlen ? data_buffer : NULL, dlen, n)) {
        femu_err("program_execute: invalid SparF workflow metadata or ranges\n");
        if (!program->is_indirect) {
            if (owns_mr_arrays) {
                free(mr_addr);
                free(mr_len);
            }
            free(mr_backend);
            free(req->data_buffer);
            req->data_buffer = NULL;
        }
        return NVME_INVALID_FIELD;
    }

    void **mr_dev_addr = NULL;
    if (program->target == PROGRAM_TARGET_CUDA_DEVPTR && numr) {
        femu_debug("program_execute: preparing CUDA devptr args, numr %u\n", numr);
        mr_dev_addr = malloc(sizeof(void *) * numr);
        if (!mr_dev_addr) {
            femu_err("program_execute: mr_dev_addr allocation failed\n");
            return NVME_DNR;
        }
        for (uint32_t i = 0; i < numr; i++) {
            mr_dev_addr[i] = backend_cuda_ensure_device_ptr(mr_backend[i], mr_addr[i], mr_len[i]);
            if (!mr_dev_addr[i]) {
                femu_err("program_execute: failed to prepare CUDA devptr for mr %u, host %p len %lld\n",
                         i, mr_addr[i], mr_len[i]);
                free(mr_dev_addr);
                return NVME_DNR;
            }
            femu_debug("program_execute: mr_dev_addr[%u] = %p for host %p len %lld\n",
                       i, mr_dev_addr[i], mr_addr[i], mr_len[i]);
        }
    }

    sched_alloc_job(compute_ns(req->ns), req);
    ComputeJob *job = req->job;
    job->user_runtime = runtime ? runtime : program->runtime;
    job->program = program;
    job->args.numr = numr;
    job->args.mr_addr = mr_addr;
    job->args.mr_len = mr_len;
    job->args.mr_dev_addr = mr_dev_addr;
    job->owns_mr_arrays = owns_mr_arrays;
    job->owns_mr_dev_addr = mr_dev_addr != NULL;
    job->mr_backend = mr_backend;
    job->args.cparam1 = cparam1;
    job->args.cparam2 = cparam2;
    job->args.data_buffer = dlen ? data_buffer : NULL;
    job->args.buffer_len = dlen;
    if (!program->is_indirect) {
        sched_enqueue_job(compute_ns(req->ns), req);
    }
    int rc = femu_ring_enqueue(compute_ns(req->ns)->to_csd[req->stat.exec.thread], (void *)&req, 1);
    if (rc != 1) {
        femu_err("enqueue failed, ret=%d\n", rc);
    }

    if (!program->is_indirect) {
        qatomic_inc(&program->jobs_running);
    }

    femu_debug("program_execute: completed\n");
    return NVME_SUCCESS;
}

static QemuMutex compute_mutex;  // 计算锁
static QemuCond compute_cond;
static int compute_active = 0;
static int64_t compute_owner = -1;
static void __attribute__((constructor)) init_sync_objects(void) {
    qemu_mutex_init(&compute_mutex);
    qemu_cond_init(&compute_cond);
}
static void __attribute__((destructor)) cleanup_sync_objects(void) {
    qemu_mutex_destroy(&compute_mutex);
    qemu_cond_destroy(&compute_cond);
}
// static void enter_compute_section(void) {
//     qemu_mutex_lock_iothread();
//     cpu_disable_ticks();
//     pause_all_vcpus();
// }
// static void leave_compute_section(void) {
//     cpu_enable_ticks();
//     resume_all_vcpus();
//     qemu_mutex_unlock_iothread();
// }

static void enter_compute_section(void) {
    qemu_mutex_lock(&compute_mutex);
    compute_active++;
    if (compute_active == 1 && compute_owner == -1) {
        qemu_mutex_lock_iothread();  // 只有第一个计算线程加锁
        compute_owner = qemu_get_thread_id();
        cpu_disable_ticks();
        pause_all_vcpus();
    }
    qemu_mutex_unlock(&compute_mutex);
}
static void leave_compute_section(void) {
    qemu_mutex_lock(&compute_mutex);
    compute_active--;
    if (qemu_get_thread_id() == compute_owner) {
        while (compute_active != 0) {
            qemu_cond_wait(&compute_cond, &compute_mutex);
        }
        cpu_enable_ticks();
        resume_all_vcpus();
        qemu_mutex_unlock_iothread();
        compute_owner = -1;
    }
    else if(compute_active == 0)
    {
        qemu_cond_broadcast(&compute_cond);
    }
    qemu_mutex_unlock(&compute_mutex);
}

// static void enter_compute_section(void) {
//     qemu_mutex_lock(&compute_mutex);
//     if (compute_owner == -1) {
//         qemu_mutex_lock_iothread();  // 只有第一个计算线程加锁
//         compute_owner = qemu_get_thread_id();
//         cpu_disable_ticks();
//         pause_all_vcpus();
//     }
//     qemu_mutex_unlock(&compute_mutex);
// }

// static void leave_compute_section(void) {
//     qemu_mutex_lock(&compute_mutex);
//     if (qemu_get_thread_id() == compute_owner) {
//         cpu_enable_ticks();
//         resume_all_vcpus();
//         qemu_mutex_unlock_iothread();
//         compute_owner = -1;
//     }
//     qemu_mutex_unlock(&compute_mutex);
// }

static uint64_t run_host_program(ComputeJob *job)
{
    Program *program = job->program;
    uint64_t res = 0;

    switch (program->type) {
    case PROGRAM_TYPE_SHARED_LIB:
        return program->shared_lib.jit_fn(&job->args);
    case PROGRAM_TYPE_EBPF:
        if (program->ebpf.jit_fn == NULL) {
            femu_debug("running non-jit ebpf...\n");
            if (ubpf_exec(program->ebpf.vm, &job->args, &res) < 0) {
                femu_err("running ebpf: ubpf_exec error\n");
                return (uint64_t)-1;
            }
            return res;
        }

        femu_debug("running jit ebpf...\n");
        return program->ebpf.jit_fn(&job->args);
    default:
        femu_err("run_host_program: program type %u not supported for HOST target\n",
                 program->type);
        return (uint64_t)-1;
    }
}

static uint64_t run_cuda_devptr_shared_lib(ComputeJob *job)
{
    Program *program = job->program;
    uint64_t res = 0;

    if(!job->mr_backend)
    {
        femu_err("run_cuda_devptr_shared_lib: CUDA_DEVPTR target missing mr_backend\n");
        return (uint64_t)-1;
    }

    if (program->type != PROGRAM_TYPE_SHARED_LIB) {
        femu_err("run_cuda_devptr_shared_lib: CUDA_DEVPTR target requires shared library program, type %u\n",
                 program->type);
        return (uint64_t)-1;
    }

    if (!job->args.mr_dev_addr) {
        femu_err("run_cuda_devptr_shared_lib: CUDA_DEVPTR target missing mr_dev_addr\n");
        return (uint64_t)-1;
    }

    if(!program->is_indirect && job->args.numr < 2) {
        femu_err("run_cuda_devptr_shared_lib: direct CUDA_DEVPTR requires input/output ranges\n");
        return (uint64_t)-1;
    }

    if(program->is_indirect) {
        for (uint32_t i = 0; i < job->args.numr; i++) {
            if (backend_cuda_prepare_device(job->mr_backend[i],
                                            job->args.mr_addr[i],
                                            job->args.mr_len[i]) != 0) {
                return (uint64_t)-1;
            }
        }
    } else {
        /* Direct CUDA ABI: mr[0..numr-2] are inputs, mr[numr-1] is output. */
        for (uint32_t i = 0; i + 1 < job->args.numr; i++) {
            if (backend_cuda_prepare_device(job->mr_backend[i],
                                            job->args.mr_addr[i],
                                            job->args.mr_len[i]) != 0) {
                return (uint64_t)-1;
            }
        }
    }

    femu_debug("running CUDA devptr shared lib...\n");
    res = program->shared_lib.jit_fn(&job->args);

    if (res != (uint64_t)-1) {
        uint32_t first_dirty = program->is_indirect ? 0 : 1;
        for (uint32_t i = first_dirty; i < job->args.numr; i++) {
            backend_cuda_mark_device_dirty(job->mr_backend[i],
                                           job->args.mr_addr[i],
                                           job->args.mr_len[i]);
        }
    }

    return res;
}

static uint64_t run_program_by_target(ComputeJob *job)
{
    Program *program = job->program;
    uint64_t res;

    switch (program->target) {
    case PROGRAM_TARGET_HOST:
        if (program->is_indirect) {
            for (uint32_t i = 0; i < job->args.numr; i++) {
                if (backend_cuda_prepare_host(job->mr_backend[i],
                                              job->args.mr_addr[i],
                                              job->args.mr_len[i]) != 0) {
                    return (uint64_t)-1;
                }
            }
        } else {
            for (uint32_t i = 0; i + 1 < job->args.numr; i++) {
                if (backend_cuda_prepare_host(job->mr_backend[i],
                                              job->args.mr_addr[i],
                                              job->args.mr_len[i]) != 0) {
                    return (uint64_t)-1;
                }
            }
        }
        res = run_host_program(job);
        if (res != (uint64_t)-1) {
            uint32_t first_dirty = program->is_indirect ? 0 : 1;
            for (uint32_t i = first_dirty; i < job->args.numr; i++) {
                backend_cuda_mark_host_dirty(job->mr_backend[i],
                                             job->args.mr_addr[i],
                                             job->args.mr_len[i]);
            }
        }
        return res;
    case PROGRAM_TARGET_CUDA_DEVPTR:
        return run_cuda_devptr_shared_lib(job);
    default:
        femu_err("run_program_by_target: program target %u not supported\n",
                 program->target);
        return (uint64_t)-1;
    }
}

static NvmeRequest *attention_submit_read(NvmeRequest *parent,
                                          struct rte_ring *completion_ring,
                                          void *destination,
                                          const struct cemu_attention_workflow_extent *extents,
                                          uint32_t extent_count,
                                          uint64_t model_start_ns)
{
    NvmeRequest *request = g_new0(NvmeRequest, 1);
    request->ns = parent->ns;
    request->cmd.nsid = cpu_to_le32(3);
    request->mem_ctrl = parent->ns->ctrl;
    request->sdaddr = destination;
    request->sdaddr_backend = NULL;
    request->is_write = 0;
    request->status = NVME_INTERNAL_DEV_ERROR | NVME_DNR;
    request->nr_sres = extent_count;
    request->sres = g_new0(NvmeCopyFormat, extent_count);
    request->indirect_task.ring = completion_ring;
    request->stat.stime = model_start_ns;
    request->stat.expire_time = request->stat.stime;
    for (uint32_t index = 0; index < extent_count; ++index) {
        request->sres[index].cf2.snsid = cpu_to_le32(1);
        request->sres[index].cf2.slba = cpu_to_le64(extents[index].slba);
        request->sres[index].cf2.nlb = cpu_to_le16(extents[index].nlb - 1);
    }
    int rc = femu_ring_enqueue(parent->ns->ctrl->to_ftl[1],
                               (void *)&request, 1);
    if (rc != 1) {
        g_free(request->sres);
        g_free(request);
        return NULL;
    }
    return request;
}

static bool attention_wait_read(struct rte_ring *completion_ring,
                                NvmeRequest *expected)
{
    if (!expected) {
        return false;
    }
    NvmeRequest *completed = NULL;
    while (femu_ring_dequeue(completion_ring, (void *)&completed, 1) != 1) {
        usleep(10);
    }
    if (completed != expected) {
        femu_err("Attention workflow received an unexpected FTL completion\n");
        abort();
    }
    if (completed->status != NVME_SUCCESS) {
        femu_err("Attention workflow FTL read failed with status %#x\n",
                 completed->status);
        return false;
    }
    return true;
}

static void attention_free_read(NvmeRequest *request)
{
    if (request) {
        g_free(request->sres);
        g_free(request);
    }
}

typedef struct SparfLogicalSpan {
    uint64_t offset;
    uint32_t size;
    uint64_t destination_offset;
    uint32_t prefix;
    uint32_t read_size;
    uint64_t temporary_offset;
} SparfLogicalSpan;

typedef struct SparfRead {
    NvmeRequest *request;
    GArray *physical_extents;
    GArray *spans;
    uint8_t *temporary;
    uint64_t temporary_size;
    void *destination;
    SsdBackend *destination_backend;
    uint64_t destination_len;
} SparfRead;

typedef struct SparfPageCache {
    uint64_t *keys;
    uint64_t *offsets;
    uint8_t *used;
    uint32_t capacity;
} SparfPageCache;

typedef struct SparfScoreIndex {
    float score;
    uint32_t index;
} SparfScoreIndex;

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

static uint64_t sparf_hash_page(uint64_t value)
{
    value ^= value >> 30;
    value *= UINT64_C(0xbf58476d1ce4e5b9);
    value ^= value >> 27;
    value *= UINT64_C(0x94d049bb133111eb);
    return value ^ (value >> 31);
}

static bool sparf_page_cache_find(const SparfPageCache *cache,
                                  uint64_t key, uint64_t *offset)
{
    uint32_t slot = sparf_hash_page(key) & (cache->capacity - 1);
    for (uint32_t probes = 0; probes < cache->capacity; ++probes) {
        if (!cache->used[slot]) {
            return false;
        }
        if (cache->keys[slot] == key) {
            *offset = cache->offsets[slot];
            return true;
        }
        slot = (slot + 1) & (cache->capacity - 1);
    }
    return false;
}

static bool sparf_page_cache_insert(SparfPageCache *cache,
                                    uint64_t key, uint64_t offset)
{
    uint32_t slot = sparf_hash_page(key) & (cache->capacity - 1);
    for (uint32_t probes = 0; probes < cache->capacity; ++probes) {
        if (!cache->used[slot]) {
            cache->used[slot] = 1;
            cache->keys[slot] = key;
            cache->offsets[slot] = offset;
            return true;
        }
        if (cache->keys[slot] == key) {
            return true;
        }
        slot = (slot + 1) & (cache->capacity - 1);
    }
    return false;
}

static bool sparf_append_physical_range(
    GArray *output, const struct cemu_sparf_workflow_extent *map,
    uint32_t map_count, uint64_t offset, uint64_t size)
{
    uint64_t logical = 0;
    uint64_t end = offset + size;
    for (uint32_t index = 0; index < map_count && offset < end; ++index) {
        uint64_t extent_bytes = (uint64_t)map[index].nlb * 512;
        uint64_t extent_end = logical + extent_bytes;
        if (offset < extent_end && end > logical) {
            uint64_t begin = MAX(offset, logical);
            uint64_t finish = MIN(end, extent_end);
            struct cemu_attention_workflow_extent physical = {
                .slba = map[index].slba + (begin - logical) / 512,
                .nlb = (finish - begin) / 512,
                .reserved = 0,
            };
            if (!physical.nlb || physical.nlb > 65535) {
                return false;
            }
            g_array_append_val(output, physical);
            offset = finish;
        }
        logical = extent_end;
    }
    return offset == end;
}

static bool sparf_start_read(
    ComputeJob *job, struct rte_ring *ring,
    const struct cemu_sparf_workflow_extent *map, uint32_t map_count,
    GArray *requested_spans, uint32_t destination_range,
    uint64_t model_start_ns, SparfRead *read)
{
    memset(read, 0, sizeof(*read));
    read->physical_extents = g_array_new(FALSE, FALSE,
        sizeof(struct cemu_attention_workflow_extent));
    read->spans = requested_spans;
    read->destination = job->args.mr_addr[destination_range];
    read->destination_backend = job->mr_backend[destination_range];
    read->destination_len = 0;
    SparfPageCache pages = { 0 };
    pages.capacity = 1;
    while (pages.capacity < requested_spans->len * 2 + 1) {
        pages.capacity <<= 1;
    }
    pages.keys = g_new(uint64_t, pages.capacity);
    pages.offsets = g_new(uint64_t, pages.capacity);
    pages.used = g_new0(uint8_t, pages.capacity);
    for (guint index = 0; index < requested_spans->len; ++index) {
        SparfLogicalSpan *span = &g_array_index(requested_spans, SparfLogicalSpan, index);
        uint64_t aligned_start = span->offset & ~UINT64_C(4095);
        uint64_t aligned_end = (span->offset + span->size + 4095) & ~UINT64_C(4095);
        span->prefix = span->offset - aligned_start;
        span->read_size = aligned_end - aligned_start;
        uint64_t existing_offset = 0;
        if (span->read_size == 4096 &&
            sparf_page_cache_find(&pages, aligned_start, &existing_offset)) {
            span->temporary_offset = existing_offset;
        } else {
            span->temporary_offset = read->temporary_size;
            if (!sparf_append_physical_range(read->physical_extents, map, map_count,
                                             aligned_start, span->read_size)) {
                g_free(pages.keys);
                g_free(pages.offsets);
                g_free(pages.used);
                return false;
            }
            if (span->read_size == 4096) {
                if (!sparf_page_cache_insert(&pages, aligned_start,
                                             read->temporary_size)) {
                    g_free(pages.keys);
                    g_free(pages.offsets);
                    g_free(pages.used);
                    return false;
                }
            }
            read->temporary_size += span->read_size;
        }
        read->destination_len = MAX(read->destination_len,
                                    span->destination_offset + span->size);
    }
    g_free(pages.keys);
    g_free(pages.offsets);
    g_free(pages.used);
    read->temporary = g_malloc(read->temporary_size);
    read->request = attention_submit_read(
        job->req, ring, read->temporary,
        (const struct cemu_attention_workflow_extent *)read->physical_extents->data,
        read->physical_extents->len, model_start_ns);
    return read->request != NULL;
}

static bool sparf_finish_read(SparfRead *read, struct rte_ring *ring,
                              uint64_t *modeled_ns)
{
    bool success = attention_wait_read(ring, read->request);
    if (success) {
        *modeled_ns = read->request->stat.reqlat;
        if (backend_cuda_prepare_host(read->destination_backend,
                                      read->destination,
                                      read->destination_len) != 0) {
            success = false;
        }
    }
    if (success) {
        for (guint index = 0; index < read->spans->len; ++index) {
            const SparfLogicalSpan *span = &g_array_index(
                read->spans, SparfLogicalSpan, index);
            memcpy((uint8_t *)read->destination + span->destination_offset,
                   read->temporary + span->temporary_offset + span->prefix,
                   span->size);
        }
        backend_cuda_mark_host_dirty(read->destination_backend,
                                     read->destination,
                                     read->destination_len);
    }
    attention_free_read(read->request);
    read->request = NULL;
    g_free(read->temporary);
    read->temporary = NULL;
    if (read->physical_extents) g_array_free(read->physical_extents, TRUE);
    if (read->spans) g_array_free(read->spans, TRUE);
    read->physical_extents = NULL;
    read->spans = NULL;
    return success;
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

static uint64_t run_sparf_workflow(ComputeJob *job, uint64_t *modeled_runtime)
{
    struct cemu_sparf_workflow_header header;
    struct cemu_sparf_workflow_trace trace = { 0 };
    memcpy(&header, job->args.data_buffer, sizeof(header));
    const struct cemu_sparf_workflow_extent *maps =
        (const struct cemu_sparf_workflow_extent *)
        ((const uint8_t *)job->args.data_buffer + sizeof(header));
    const struct cemu_sparf_workflow_extent *token_k_map = maps;
    const struct cemu_sparf_workflow_extent *channel_k_map = token_k_map + header.token_k_extents;
    const struct cemu_sparf_workflow_extent *token_v_map = channel_k_map + header.channel_k_extents;
    const uint32_t vectors = header.batch_size * header.num_query_heads;
    const uint32_t element = header.dtype == CEMU_SPARF_FLOAT16 ? 2 : 4;
    const uint32_t heads_per_kv = header.num_query_heads / header.num_kv_heads;
    struct rte_ring *ring = femu_ring_create(FEMU_RING_TYPE_MP_SC, 8);
    uint32_t *channels = g_new(uint32_t, (uint64_t)vectors * header.top_r);
    uint32_t *tokens = g_new(uint32_t, (uint64_t)vectors * header.top_k);
    float *scores = g_new(float, header.valid_tokens);
    SparfScoreIndex *score_indices = g_new(SparfScoreIndex, header.valid_tokens);
    float *query_values = g_new(float, (uint64_t)vectors * header.head_dim);
    float *channel_values = g_new(float,
        (uint64_t)vectors * header.top_r * header.valid_tokens);
    float *current_key_values = g_new(float,
        (uint64_t)header.batch_size * header.num_kv_heads * header.head_dim);
    SparfScoreIndex *channel_indices = g_new(SparfScoreIndex, header.head_dim);
    uint64_t result = (uint64_t)-1;
    SparfRead channel_read = { 0 }, key_read = { 0 }, value_read = { 0 };
    if (!ring || backend_cuda_prepare_host(job->mr_backend[0], job->args.mr_addr[0],
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
                .size = header.valid_tokens * element,
                .destination_offset = ((uint64_t)vector * header.top_r + rank) *
                                      header.valid_tokens * element,
            };
            g_array_append_val(channel_spans, span);
        }
    }
    if (!sparf_start_read(job, ring, channel_k_map, header.channel_k_extents,
                          channel_spans, 1, clock_ns(), &channel_read) ||
        !sparf_finish_read(&channel_read, ring, &trace.channel_read_model_ns)) goto out;

    for (uint64_t index = 0;
         index < (uint64_t)vectors * header.top_r * header.valid_tokens;
         ++index) {
        channel_values[index] = sparf_load_scalar(job->args.mr_addr[1], index,
                                                  header.dtype);
    }
    if (backend_cuda_prepare_host(job->mr_backend[5], job->args.mr_addr[5], job->args.mr_len[5]) != 0) goto out;
    for (uint64_t index = 0;
         index < (uint64_t)header.batch_size * header.num_kv_heads * header.head_dim;
         ++index) {
        current_key_values[index] = sparf_load_scalar(job->args.mr_addr[5], index,
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
            scores[header.valid_tokens - 1] += query_value * current_key_values[
                ((uint64_t)batch * header.num_kv_heads + head) * header.head_dim + dimension];
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

    GArray *key_spans = g_array_new(FALSE, FALSE, sizeof(SparfLogicalSpan));
    GArray *value_spans = g_array_new(FALSE, FALSE, sizeof(SparfLogicalSpan));
    for (uint32_t vector = 0; vector < vectors; ++vector) {
        uint32_t batch = vector / header.num_query_heads;
        uint32_t head = (vector % header.num_query_heads) / heads_per_kv;
        for (uint32_t rank = 0; rank < header.top_k; ++rank) {
            uint64_t logical = (uint64_t)header.layer * header.token_layer_stride +
                (uint64_t)batch * header.token_batch_stride +
                (uint64_t)head * header.token_head_stride +
                (uint64_t)tokens[(uint64_t)vector * header.top_k + rank] * header.head_dim * element;
            SparfLogicalSpan span = {
                .offset = logical, .size = header.head_dim * element,
                .destination_offset = ((uint64_t)vector * header.top_k + rank) * header.head_dim * element,
            };
            SparfLogicalSpan value_span = span;
            value_span.destination_offset =
                ((uint64_t)vector * (header.top_k + 1) + rank) *
                header.head_dim * element;
            g_array_append_val(key_spans, span);
            g_array_append_val(value_spans, value_span);
        }
    }
    if (!sparf_start_read(job, ring, token_k_map, header.token_k_extents,
                          key_spans, 2, clock_ns(), &key_read) ||
        !sparf_finish_read(&key_read, ring, &trace.token_k_read_model_ns)) goto out;
    if (backend_cuda_prepare_host(job->mr_backend[2], job->args.mr_addr[2], job->args.mr_len[2]) != 0) goto out;
    for (uint32_t vector = 0; vector < vectors; ++vector) {
        uint32_t batch = vector / header.num_query_heads;
        uint32_t head = (vector % header.num_query_heads) / heads_per_kv;
        for (uint32_t rank = 0; rank < header.top_k; ++rank) {
            if (tokens[(uint64_t)vector * header.top_k + rank] + 1 == header.valid_tokens) {
                memcpy((uint8_t *)job->args.mr_addr[2] +
                           ((uint64_t)vector * header.top_k + rank) * header.head_dim * element,
                       (uint8_t *)job->args.mr_addr[5] +
                           ((uint64_t)batch * header.num_kv_heads + head) * header.head_dim * element,
                       header.head_dim * element);
            }
        }
    }
    backend_cuda_mark_host_dirty(job->mr_backend[2], job->args.mr_addr[2], job->args.mr_len[2]);
    if (!sparf_start_read(job, ring, token_v_map, header.token_v_extents,
                          value_spans, 3, clock_ns(), &value_read)) goto out;

    struct cemu_sparf_attention_metadata metadata = {
        .version = CEMU_SPARF_ATTENTION_VERSION,
        .phase = CEMU_SPARF_PHASE_EXACT_QK,
        .dtype = header.dtype,
        .batch_size = header.batch_size,
        .num_query_heads = header.num_query_heads,
        .head_dim = header.head_dim,
        .selected_tokens = header.top_k,
        .reserved = 0,
        .scale = header.scale,
    };
    const uint32_t exact_ranges[] = { 0, 2, 7, 8 };
    result = run_sparf_phase(job, CEMU_SPARF_PHASE_EXACT_QK,
                             exact_ranges, 4, &metadata);
    if (!sparf_finish_read(&value_read, ring, &trace.token_v_read_model_ns) ||
        result == (uint64_t)-1) goto out;
    if (backend_cuda_prepare_host(job->mr_backend[3], job->args.mr_addr[3],
                                  job->args.mr_len[3]) != 0 ||
        backend_cuda_prepare_host(job->mr_backend[4], job->args.mr_addr[4],
                                  job->args.mr_len[4]) != 0 ||
        backend_cuda_prepare_host(job->mr_backend[6], job->args.mr_addr[6],
                                  job->args.mr_len[6]) != 0) goto out;
    for (uint32_t vector = 0; vector < vectors; ++vector) {
        uint32_t batch = vector / header.num_query_heads;
        uint32_t head = (vector % header.num_query_heads) / heads_per_kv;
        for (uint32_t rank = 0; rank < header.top_k; ++rank) {
            if (tokens[(uint64_t)vector * header.top_k + rank] + 1 == header.valid_tokens) {
                memcpy((uint8_t *)job->args.mr_addr[3] +
                           ((uint64_t)vector * (header.top_k + 1) + rank) * header.head_dim * element,
                       (uint8_t *)job->args.mr_addr[6] +
                           ((uint64_t)batch * header.num_kv_heads + head) * header.head_dim * element,
                       header.head_dim * element);
            }
        }
        memcpy((uint8_t *)job->args.mr_addr[3] +
                   ((uint64_t)vector * (header.top_k + 1) + header.top_k) * header.head_dim * element,
               (uint8_t *)job->args.mr_addr[4] + (uint64_t)vector * header.head_dim * element,
               header.head_dim * element);
    }
    backend_cuda_mark_host_dirty(job->mr_backend[3], job->args.mr_addr[3], job->args.mr_len[3]);
    const uint32_t pv_ranges[] = { 8, 3, 9 };
    result = run_sparf_phase(job, CEMU_SPARF_PHASE_PV, pv_ranges, 3, &metadata);

out:
    if (channel_read.request) sparf_finish_read(&channel_read, ring, &trace.channel_read_model_ns);
    if (key_read.request) sparf_finish_read(&key_read, ring, &trace.token_k_read_model_ns);
    if (value_read.request) sparf_finish_read(&value_read, ring, &trace.token_v_read_model_ns);
    trace.approximate_model_ns = header.approximate_runtime_ns;
    trace.exact_qk_model_ns = header.exact_qk_runtime_ns;
    trace.pv_model_ns = header.pv_runtime_ns;
    trace.selected_channels = (uint64_t)vectors * header.top_r;
    trace.selected_tokens = (uint64_t)vectors * header.top_k;
    trace.total_model_ns = trace.channel_read_model_ns + trace.approximate_model_ns +
        trace.token_k_read_model_ns + MAX(trace.exact_qk_model_ns, trace.token_v_read_model_ns) +
        trace.pv_model_ns;
    *modeled_runtime = trace.total_model_ns;
    if (backend_cuda_prepare_host(job->mr_backend[10], job->args.mr_addr[10], sizeof(trace)) == 0) {
        memcpy(job->args.mr_addr[10], &trace, sizeof(trace));
        backend_cuda_mark_host_dirty(job->mr_backend[10], job->args.mr_addr[10], sizeof(trace));
    } else result = (uint64_t)-1;
    if (ring) femu_ring_free(ring);
    g_free(channels); g_free(tokens); g_free(scores); g_free(score_indices);
    g_free(query_values); g_free(channel_values); g_free(current_key_values);
    g_free(channel_indices);
    return result;
}

static uint64_t run_attention_phase_stage(ComputeJob *job, uint32_t phase,
                                          const uint32_t indices[3],
                                          struct cemu_attention_phase_metadata *metadata)
{
    void *host_addresses[3];
    void *device_addresses[3];
    long long lengths[3];
    SsdBackend *backends[3];
    for (uint32_t index = 0; index < 3; ++index) {
        uint32_t source = indices[index];
        host_addresses[index] = job->args.mr_addr[source];
        device_addresses[index] = job->args.mr_dev_addr
                                ? job->args.mr_dev_addr[source] : NULL;
        lengths[index] = job->args.mr_len[source];
        backends[index] = job->mr_backend[source];
    }

    struct ubpf_jit_args arguments = job->args;
    arguments.numr = 3;
    arguments.mr_addr = host_addresses;
    arguments.mr_dev_addr = job->args.mr_dev_addr ? device_addresses : NULL;
    arguments.mr_len = lengths;
    metadata->phase = phase;
    arguments.data_buffer = metadata;
    arguments.buffer_len = sizeof(*metadata);

    if (job->program->target == PROGRAM_TARGET_CUDA_DEVPTR) {
        if (!arguments.mr_dev_addr) {
            return (uint64_t)-1;
        }
        for (uint32_t index = 0; index < 2; ++index) {
            if (backend_cuda_prepare_device(backends[index], host_addresses[index],
                                            lengths[index]) != 0) {
                return (uint64_t)-1;
            }
        }
        uint64_t result = job->program->shared_lib.jit_fn(&arguments);
        if (result != (uint64_t)-1) {
            backend_cuda_mark_device_dirty(backends[2], host_addresses[2], lengths[2]);
        }
        return result;
    }
    if (job->program->target == PROGRAM_TARGET_HOST) {
        for (uint32_t index = 0; index < 2; ++index) {
            if (backend_cuda_prepare_host(backends[index], host_addresses[index],
                                          lengths[index]) != 0) {
                return (uint64_t)-1;
            }
        }
        uint64_t result = job->program->shared_lib.jit_fn(&arguments);
        if (result != (uint64_t)-1) {
            backend_cuda_mark_host_dirty(backends[2], host_addresses[2], lengths[2]);
        }
        return result;
    }
    return (uint64_t)-1;
}

static uint64_t run_attention_workflow(ComputeJob *job, uint64_t *modeled_runtime)
{
    struct cemu_attention_workflow_header header;
    struct cemu_attention_workflow_trace trace = { 0 };
    memcpy(&header, job->args.data_buffer, sizeof(header));
    const struct cemu_attention_workflow_extent *extents =
        (const struct cemu_attention_workflow_extent *)
        ((const uint8_t *)job->args.data_buffer + sizeof(header));
    const bool serial = header.flags & CEMU_ATTENTION_WORKFLOW_SERIAL;
    struct rte_ring *ring = femu_ring_create(FEMU_RING_TYPE_MP_SC, 8);
    NvmeRequest *key_request = NULL;
    NvmeRequest *value_request = NULL;
    uint64_t result = (uint64_t)-1;
    if (!ring) {
        return result;
    }

    trace.key_submit_ns = monotonic_ns();
    uint64_t key_model_start = clock_ns();
    key_request = attention_submit_read(job->req, ring, job->args.mr_addr[1],
                                        extents, header.key_extents,
                                        key_model_start);
    if (!key_request || !attention_wait_read(ring, key_request)) {
        goto out;
    }
    trace.key_ready_ns = monotonic_ns();
    trace.key_model_ns = key_request->stat.reqlat;
    uint64_t middle_model_start = key_model_start + trace.key_model_ns;
    attention_free_read(key_request);
    key_request = NULL;

    if (!serial) {
        trace.value_submit_ns = monotonic_ns();
        value_request = attention_submit_read(
            job->req, ring, job->args.mr_addr[2],
            extents + header.key_extents, header.value_extents,
            middle_model_start);
        if (!value_request) {
            goto out;
        }
    }

    const uint32_t qk_indices[] = { 0, 1, 3 };
    trace.qk_start_ns = monotonic_ns();
    result = run_attention_phase_stage(job, CEMU_ATTENTION_PHASE_QK_SOFTMAX,
                                       qk_indices, &header.attention);
    trace.qk_done_ns = monotonic_ns();
    if (result == (uint64_t)-1) {
        if (value_request) {
            goto wait_value;
        }
        goto out;
    }

    if (serial) {
        trace.value_submit_ns = monotonic_ns();
        value_request = attention_submit_read(
            job->req, ring, job->args.mr_addr[2],
            extents + header.key_extents, header.value_extents,
            middle_model_start + header.qk_runtime_ns);
        if (!value_request) {
            result = (uint64_t)-1;
            goto out;
        }
    }

wait_value:
    if (!attention_wait_read(ring, value_request)) {
        result = (uint64_t)-1;
        goto out;
    }
    trace.value_ready_ns = monotonic_ns();
    trace.value_model_ns = value_request->stat.reqlat;
    attention_free_read(value_request);
    value_request = NULL;
    if (result == (uint64_t)-1) {
        goto out;
    }

    const uint32_t pv_indices[] = { 3, 2, 4 };
    trace.pv_start_ns = monotonic_ns();
    result = run_attention_phase_stage(job, CEMU_ATTENTION_PHASE_PV,
                                       pv_indices, &header.attention);
    trace.pv_done_ns = monotonic_ns();

out:
    if (value_request) {
        attention_wait_read(ring, value_request);
        attention_free_read(value_request);
    }
    attention_free_read(key_request);
    trace.finish_ns = monotonic_ns();
    trace.qk_model_ns = header.qk_runtime_ns;
    trace.pv_model_ns = header.pv_runtime_ns;
    uint64_t middle = serial
        ? (uint64_t)header.qk_runtime_ns + trace.value_model_ns
        : MAX((uint64_t)header.qk_runtime_ns, trace.value_model_ns);
    trace.total_model_ns = trace.key_model_ns + middle + header.pv_runtime_ns;
    *modeled_runtime = trace.total_model_ns;
    if (backend_cuda_prepare_host(job->mr_backend[5], job->args.mr_addr[5],
                                  sizeof(trace)) == 0) {
        memcpy(job->args.mr_addr[5], &trace, sizeof(trace));
        backend_cuda_mark_host_dirty(job->mr_backend[5], job->args.mr_addr[5],
                                     sizeof(trace));
    } else {
        result = (uint64_t)-1;
    }
    femu_ring_free(ring);
    return result;
}

static uint64_t run_functional_modeling(ComputeJob *job)
{
    Program *program = job->program;
    CemuComputeLogMode log_mode = cemu_compute_log_mode();
    bool collect_compute_stats = log_mode != CEMU_COMPUTE_LOG_OFF;
    bool detailed_compute_log = log_mode == CEMU_COMPUTE_LOG_DETAIL;
    uint64_t res = 0;
    uint64_t realtime;
    uint64_t runtime = job->user_runtime;
    const uint64_t requested_runtime = runtime;
    uint64_t freeze_entry_ns = 0;
    uint64_t workflow_runtime = 0;
    bool attention_workflow =
        (uint64_t)job->args.cparam1 == CEMU_ATTENTION_WORKFLOW_COMMAND;
    bool sparf_workflow =
        (uint64_t)job->args.cparam1 == CEMU_SPARF_WORKFLOW_COMMAND;
    struct timespec t0, t1;

    if (detailed_compute_log) {
        clock_gettime(CLOCK_MONOTONIC, &t0);
    }

    femu_debug("run_functional_modeling: program %u, runtime %lu, size %llu\n", program->pind, runtime, job->args.mr_len[0]);

    // execute
    // if (!runtime) {
        struct timespec ts,te;
        struct timespec cs,ce;

        clock_gettime(CLOCK_MONOTONIC, &cs);
        enter_compute_section();
        clock_gettime(CLOCK_MONOTONIC, &ce);
        freeze_entry_ns = (ce.tv_sec-cs.tv_sec)* 1000000000LL + (ce.tv_nsec-cs.tv_nsec);
        if(runtime < freeze_entry_ns)
            runtime = 0;
        else
            runtime -= freeze_entry_ns;

        clock_gettime(CLOCK_MONOTONIC, &ts);
        if (sparf_workflow) {
            res = run_sparf_workflow(job, &workflow_runtime);
        } else if (attention_workflow) {
            res = run_attention_workflow(job, &workflow_runtime);
        } else {
            res = run_program_by_target(job);
        }
        clock_gettime(CLOCK_MONOTONIC, &te);
        realtime = (te.tv_sec-ts.tv_sec)* 1000000000LL + (te.tv_nsec-ts.tv_nsec);
        if (attention_workflow || sparf_workflow) {
            runtime = workflow_runtime > freeze_entry_ns
                    ? workflow_runtime - freeze_entry_ns : 0;
        } else if(!runtime)
        {
            if (program->runtime_scale)
                runtime = realtime * program->runtime_scale;
        }
        leave_compute_section();
    // }

        femu_debug("run_on_host: program %u, runtime %lu, realtime: %lu, size %llu\n", program->pind, runtime, realtime,job->args.mr_len[0]);

        if (detailed_compute_log) {
            printf("CEMU_COMPUTE: program %u, realtime=%lu ns, runtime=%lu ns, "
                   "requested_runtime=%lu ns, freeze_entry=%lu ns\n",
                program->pind, (unsigned long)realtime, (unsigned long)runtime,
                (unsigned long)requested_runtime, (unsigned long)freeze_entry_ns);
        }

        set_sched_runtime(job, runtime);

    if (collect_compute_stats) {
        atomic_fetch_add_explicit(&csd_total_compute_ns,
                                  (uint_fast64_t)realtime,
                                  memory_order_relaxed);
        uint64_t jobs = atomic_fetch_add_explicit(&csd_job_count, 1,
                                                   memory_order_relaxed) + 1;
        if (detailed_compute_log) {
            clock_gettime(CLOCK_MONOTONIC, &t1);
            uint64_t job_ns = (t1.tv_sec - t0.tv_sec) * 1000000000ULL +
                              (t1.tv_nsec - t0.tv_nsec);
            atomic_fetch_add_explicit(&csd_total_job_ns,
                                      (uint_fast64_t)job_ns,
                                      memory_order_relaxed);
        }
        if (detailed_compute_log && jobs % CSD_BASELINE_PRINT_FREQ == 0) {
            uint64_t total_c = atomic_load_explicit(&csd_total_compute_ns,
                                                    memory_order_relaxed);
            uint64_t total_j = atomic_load_explicit(&csd_total_job_ns,
                                                    memory_order_relaxed);
            double frac = total_j ? ((double)total_c / (double)total_j) : 0.0;
            femu_log("CSD baseline: jobs=%lu, compute_ns=%lu, total_ns=%lu, "
                     "compute_fraction=%.4f\n", jobs, total_c, total_j, frac);
        } else if (log_mode == CEMU_COMPUTE_LOG_PROGRESS &&
                   jobs % cemu_compute_progress_interval() == 0) {
            uint64_t total_c = atomic_load_explicit(&csd_total_compute_ns,
                                                    memory_order_relaxed);
            femu_log("CEMU compute progress: jobs=%lu, last_program=%u, "
                     "last_realtime_ns=%lu, avg_realtime_ns=%lu\n",
                     jobs, program->pind, (unsigned long)realtime,
                     (unsigned long)(total_c / jobs));
        }
    }
    return res;
}

static int compute_io_cmd(NvmeNamespace *ns, NvmeCmd *cmd, NvmeRequest *req)
{
    switch (cmd->opcode) {
    case NVME_CMD_COMPUTE_EXEC:
        req->status = program_execute(ns, cmd, req);
        return req->status != NVME_SUCCESS;
    case NVME_CMD_COMPUTE_LOAD:
        req->status = load_program(ns, cmd, req);
        break;
    default:
        femu_err("compute_io_cmd: opcode %u not supported!\n", cmd->opcode);
        req->status = NVME_INVALID_OPCODE;
        break;
    }
    return 1;
}

static uint16_t compute_admin_cmd(NvmeNamespace *ns, NvmeCmd *cmd, NvmeCqe *cqe)
{
    switch (cmd->opcode) {
    case NVME_CMD_COMPUTE_MRS_MGMT:
        return memory_range_set_management(ns, cmd, cqe);
    case NVME_CMD_COMPUTE_ACTIVATE:
        return program_activation(ns, cmd, cqe);
    default:
        femu_err("compute_io_cmd: opcode %u not supported!\n", cmd->opcode);
        return NVME_INVALID_OPCODE;
    }
}

static inline int next_sres(struct sre_iter *iter, NvmeCopyFormat *out_sres)
{
    int nlb = iter->nlb;
    int nr_sres = 0;
    int sre_done = iter->done;
    int pos = iter->pos;
    if (iter->pos >= iter->nr_sres) {
        return 0;
    }
    while (nlb && pos < iter->nr_sres) {
        int n = MIN(nlb, iter->sres[pos].nlb + 1 - sre_done);
        out_sres[nr_sres].cf2.slba = iter->sres[pos].slba + sre_done;
        out_sres[nr_sres].cf2.nlb = n - 1;
        out_sres[nr_sres].cf2.snsid = 1;
        sre_done += n;
        nlb -= n;
        nr_sres++;
        if (sre_done == iter->sres[pos].nlb + 1) {
            pos++;
            sre_done = 0;
        }
    }
    iter->done = sre_done;
    iter->pos = pos;
    return nr_sres;
}

static NvmeRequest *req_dup(NvmeRequest *req)
{
    NvmeRequest *new_req = malloc(sizeof(NvmeRequest));
    memcpy(new_req, req, sizeof(NvmeRequest));
    sched_alloc_job(compute_ns(new_req->ns), new_req);
    memcpy(new_req->job, req->job, sizeof(ComputeJob));
    new_req->job->req = new_req;
    new_req->job->owns_mr_dev_addr = false;
    sched_enqueue_job(compute_ns(new_req->ns), new_req);
    return new_req;
}

static int cmp_pri(pqueue_pri_t next, pqueue_pri_t curr)
{
    return (next > curr);
}

static pqueue_pri_t get_pri(void *a)
{
    return ((NvmeRequest *)a)->stat.expire_time;
}

static void set_pri(void *a, pqueue_pri_t pri)
{
    ((NvmeRequest *)a)->stat.expire_time = pri;
}

static size_t get_pos(void *a)
{
    return ((NvmeRequest *)a)->pos;
}

static void set_pos(void *a, size_t pos)
{
    ((NvmeRequest *)a)->pos = pos;
}

static void *indirect_main(void *arg)
{
    NvmeRequest *indirect_req = arg;
    NvmeProgramExecuteCmd *exec = (NvmeProgramExecuteCmd *)&indirect_req->cmd;
    NvmeNamespace *ns = indirect_req->ns;
    ComputeNamespace *cns = compute_ns(indirect_req->ns);
    FemuCtrl *n = ns->ctrl;
    IndirectTask *task = &indirect_req->indirect_task;
    uint64_t stime = indirect_req->stat.stime;
    bool dest_in_nvm = task->destination == 1;
    struct sre_iter *iter = NULL;

    uint64_t input_time = 0;
    uint64_t output_time = 0;
    uint64_t compute_time = 0;
    uint64_t start = clock_ns();

    int chunk_nlb = task->chunk_nlb;
    int parallel_chunks = task->nr_concurrent_chunks;
    int left_chunks = 0;

    struct rte_ring *ring = femu_ring_create(FEMU_RING_TYPE_MP_SC, 128);
    if (ring == NULL) {
        femu_err("indirect_main: create ring failed!\n");
        abort();
    }

    void **mr_addr = indirect_req->job->args.mr_addr;
    void **mr_dev_addr = indirect_req->job->args.mr_dev_addr;
    long long *mr_len = indirect_req->job->args.mr_len;
    SsdBackend **mr_backend = indirect_req->job->mr_backend;
    int numr_per_chunk = indirect_req->job->args.numr / parallel_chunks;

    femu_debug("indirect main thread start: parallel_chunks %d, chunk_nlb %d\n", parallel_chunks, chunk_nlb);
    femu_debug("indirect main thread mem_ctrl %p\n", indirect_req->mem_ctrl);

    NvmeRequest **reqs = malloc(sizeof(NvmeRequest *) * parallel_chunks);
    for (int i = 0; i < parallel_chunks; i++) {
        reqs[i] = req_dup(indirect_req);
        reqs[i]->indirect_task.chunk_id = i;

        reqs[i]->job->args.mr_addr = mr_addr + numr_per_chunk * i;
        reqs[i]->job->args.mr_dev_addr = mr_dev_addr ? mr_dev_addr + numr_per_chunk * i : NULL;
        reqs[i]->job->args.mr_len = mr_len + numr_per_chunk * i;
        reqs[i]->job->mr_backend = mr_backend ? mr_backend + numr_per_chunk * i : NULL;
        reqs[i]->job->args.numr = numr_per_chunk;
        reqs[i]->indirect_task.ring = ring;
        reqs[i]->indirect_task.stage = 0;

        iter = &task->iter_in[i];
        if (iter->nr_sres == 0) {
            reqs[i]->sres = NULL;
            continue;
        }

        // additional sres for chunk memory copy command
        reqs[i]->sres = malloc(sizeof(NvmeCopyFormat2) * iter->nr_sres);
        // generate next io request
        reqs[i]->nr_sres = next_sres(iter, reqs[i]->sres);
        if (reqs[i]->nr_sres == 0) {
            femu_err("indirect main thread: no more input sres!\n");
            left_chunks--;
        }
        reqs[i]->sdaddr = reqs[i]->job->args.mr_addr[1];
        reqs[i]->mem_ctrl = n;
        reqs[i]->indirect_task.stage = 0;
        reqs[i]->is_write = 0;
        femu_ring_enqueue(n->to_ftl[1], (void *)&reqs[i], 1);
        left_chunks++;
        qatomic_inc(&reqs[i]->job->program->jobs_running);
    }

    pqueue_t *pq = pqueue_init(FEMU_MAX_INF_REQS, cmp_pri, get_pri, set_pri,
                               get_pos, set_pos);
    femu_debug("indirect finished init pqueue\n");

    while (true) {
        // wait io
        NvmeRequest *req;
        uint64_t now = clock_ns();
        if (!femu_ring_empty(ring)) {
            // req has finished ftl
            femu_ring_dequeue(ring, (void *)&req, 1);
            if (req->indirect_task.stage == 0) {
                // input finish
                femu_debug("indirect input io finish, chunk %d, iter %d, stime %lu, expire_time %lu, lat %u\n", req->indirect_task.chunk_id, req->indirect_task.iter_finished[req->indirect_task.chunk_id], req->stat.stime, req->stat.expire_time, req->stat.reqlat);
                input_time += req->stat.reqlat;
                req->indirect_task.stage = 1;
                pqueue_insert(pq, req);
            } else if (req->indirect_task.stage == 1) {
                // compute finish, prepare next io
                femu_debug("indirect compute finish, chunk %d, iter %d, stime %lu, expire_time %lu, lat %u, left_chunks %d\n", req->indirect_task.chunk_id, req->indirect_task.iter_finished[req->indirect_task.chunk_id], req->stat.stime, req->stat.expire_time, req->stat.reqlat, left_chunks);
                compute_time += req->stat.reqlat;
                if (dest_in_nvm && req->cqe.res64 > 0) {
                    // to stage 2
                    req->indirect_task.stage = 2;

                    iter = &req->indirect_task.iter_out[req->indirect_task.chunk_id];
                    iter->nlb = req->cqe.res64;
                    req->nr_sres = next_sres(iter, req->sres);
                    if (req->nr_sres == 0) {
                        // error! no more output sres!
                        left_chunks--;
                        femu_err("indirect main thread: no more output sres!\n");
                    } else {
                        req->sdaddr = req->job->args.mr_addr[0]; // output in mr[0]
                        pqueue_insert(pq, req);
                    }
                } else {
                    // to stage 0
                    iter = &req->indirect_task.iter_in[req->indirect_task.chunk_id];
                    req->nr_sres = next_sres(iter, req->sres);
                    req->indirect_task.stage = 0;

                    if (req->nr_sres == 0) {
                        // no more input, chunk finished
                        left_chunks--;
                    } else {
                        req->sdaddr = req->job->args.mr_addr[1]; // output in mr[0]
                        pqueue_insert(pq, req);
                    }
                    req->indirect_task.iter_finished[req->indirect_task.chunk_id]++;
                }
            } else if (req->indirect_task.stage == 2) {
                // output finish, prepare next io
                femu_debug("indirect output io finish, chunk %d, iter %d, stime %lu, expire_time %lu, lat %u, left_chunks %d\n", req->indirect_task.chunk_id, req->indirect_task.iter_finished[req->indirect_task.chunk_id], req->stat.stime, req->stat.expire_time, req->stat.reqlat, left_chunks);
                struct sre_iter *iter = &req->indirect_task.iter_in[req->indirect_task.chunk_id];
                req->nr_sres = next_sres(iter, req->sres);
                if (req->nr_sres == 0) {
                    // no more input, chunk finished
                    left_chunks--;
                } else {
                    req->sdaddr = req->job->args.mr_addr[1]; // input in mr[1]
                    req->indirect_task.stage = 0;
                    pqueue_insert(pq, req);
                }
                req->indirect_task.iter_finished[req->indirect_task.chunk_id]++;
            }
        }

        // femu_log("indirect pqueue size %zu\n", pqueue_size(pq));
        while ((req = pqueue_peek(pq))) {
            now = clock_ns();
            if (now >= req->stat.expire_time) {
                // io finish
                pqueue_pop(pq);
                if (req->indirect_task.stage == 0) {
                    // input
                    femu_debug("indirect start io, expire_time %lu, now %lu, diff %lu\n", req->stat.expire_time, now, now - req->stat.expire_time);
                    output_time += now - req->stat.stime;
                    req->stat.stime = now;
                    req->stat.expire_time = now;
                    req->stat.reqlat = 0;
                    req->stat.pcie_lat = 0;
                    req->is_write = 0;
                    femu_ring_enqueue(n->to_ftl[1], (void *)&req, 1);
                } else if (req->indirect_task.stage == 1) {
                    // compute
                    sched_enqueue_indirect(cns, req->job);
                    req->stat.stime = now;
                    req->stat.expire_time = now;
                    req->stat.reqlat = 0;
                    req->stat.pcie_lat = 0;
                    uint64_t res = run_functional_modeling(req->job);
                    req->cqe.res64 = res;
                    femu_debug("indirect start compute, expire_time %lu, now %lu, diff %lu, res %lu\n", req->stat.expire_time, now, now - req->stat.expire_time, res);
                } else if (req->indirect_task.stage == 2) {
                    // output
                    femu_debug("indirect start output io, expire_time %lu, now %lu, diff %lu\n", req->stat.expire_time, now, now - req->stat.expire_time);
                    output_time += now - req->stat.stime;
                    req->stat.stime = now;
                    req->stat.expire_time = now;
                    req->stat.reqlat = 0;
                    req->stat.pcie_lat = 0;
                    req->is_write = 1;
                    femu_ring_enqueue(n->to_ftl[1], (void *)&req, 1);
                }
            } else {
                break;
            }
        }

        // femu_log("indirect left_chunks %d\n", left_chunks);
        if (left_chunks == 0)
            break;
    }

    femu_debug("indirect finish\n");
    for (int i = 0; i < parallel_chunks; i++) {
        if (reqs[i]->sres) {
            sched_job_finish_indirect(reqs[i]->job);
            free(reqs[i]->sres);
        }
        free(reqs[i]);
    }
    free(task->raw_data_buffer);
    free(task->nr_output_nlb);
    free(task->nr_finished_nlb);
    free(task->nr_finished_output_nlb);
    free(task->iter_finished);
    free(task->iter_in);
    free(task->iter_out);

    pqueue_free(pq);
    femu_ring_free(ring);
    free(reqs);

    exec->opcode = INDIRECT_THREAD_FINISH;
    indirect_req->stat.stime = stime;
    femu_ring_enqueue(cns->to_csd[0], (void *)&indirect_req, 1);

    uint64_t end = clock_ns();
    femu_debug("indirect: total time %zu, input time %zu, compute time %zu, output time %zu\n", end - start, input_time, compute_time, output_time);

    return NULL;
}

static void *compute_thread(void *arg)
{
    NvmeNamespace *ns = ((struct csd_thread_arg *)arg)->ns;
    ComputeNamespace *cns = compute_ns(ns);
    FemuCtrl *n = ns->ctrl;
    int id = ((struct csd_thread_arg *)arg)->id;

    static Slab indirect_thread_pool;
    if (id == 0)
        slab_init(&indirect_thread_pool, sizeof(QemuThread),
                MAX_INDIRECT_JOBS, NULL, NULL);

    while (!ns->ctrl->dataplane_started) {
        usleep(100000);
    }

    struct rte_ring *to_csd = cns->to_csd[id];

    femu_debug("csd_thread %d start\n", id);

    // int next_poller = 1;

    while (1) {
        if (femu_ring_empty(to_csd))
            continue;

        NvmeRequest *req = NULL;
        size_t rc = femu_ring_dequeue(to_csd, (void *)&req, 1);
        if (rc != 1) {
            printf("FEMU: FTL to_ftl dequeue failed\n");
        }

        ComputeJob *job = req->job;
        NvmeCmd *cmd = &req->cmd;

        uint16_t status = 0;
        switch (cmd->opcode) {
        case NVME_CMD_COMPUTE_EXEC:
            if (job_is_indirect(job)) {
                req->indirect_thread = slab_alloc(&indirect_thread_pool, 1);
                if (req->indirect_thread == NULL) {
                    femu_err("compute_thread: slab_alloc indirect_thread_pool failed\n");
                    break;
                }
                qemu_thread_create(req->indirect_thread, "indirect_scheduler",
                        indirect_main, req, QEMU_THREAD_JOINABLE);
            } else {
                req->cqe.res64 = run_functional_modeling(req->job);
            }
            break;
        case INDIRECT_THREAD_FINISH:
            qemu_thread_join(req->indirect_thread);
            slab_free(&indirect_thread_pool, req->indirect_thread, 1);
            cmd->opcode = NVME_CMD_COMPUTE_EXEC;
            femu_ring_enqueue(n->to_poller[1], (void *)&req, 1);
            break;
        default:
            femu_err("compute_thread: UNKNOWN OPCODE %d\n", cmd->opcode);
            abort();
        }

        req->status = status;
        if (status != NVME_SUCCESS) {
            femu_err("Error IO processed! opcode %d\n", cmd->opcode);
        }

        // femu_debug("csd_thread %d: opcode %d, status %d\n", id, cmd->opcode, status);
    }

    return NULL;
}

static int compute_decode_params(const cJSON *json, NvmeNamespace *ns)
{
    ComputeParams *params = malloc(sizeof(ComputeParams));
    ns->params = params;
    char *algo = NULL;

    DECODE_PARAM(json, "nr_cu", Number, params->nr_cu, 4);
    DECODE_PARAM(json, "nr_thread", Number, params->nr_thread, 4);
    DECODE_PARAM(json, "time_slice", Number, params->csf_sched_option.time_slice, 200000);
    DECODE_PARAM(json, "context_switch_time", Number, params->csf_sched_option.context_switch_time, 200);
    DECODE_PARAM(json, "csf_runtime_scale", Number, params->csf_runtime_scale, 3);
    DECODE_PARAM(json, "csf_sched_option", String, algo, (char *)"rr");
    params->csf_sched_option.algo = strdup(algo);
    DECODE_PARAM(json, "grouped_csf", Number, params->csf_sched_option.take_care_of_group, 0);
    DECODE_PARAM(json, "grouped_csf_prio", Number,
          params->csf_sched_option.take_care_of_group_prio, 0);
    return 0;
}

static int compute_encode_params(cJSON *json, NvmeNamespace *ns)
{
    ComputeParams *params = ns->params;

    ENCODE_PARAM(json, "nr_cu", Number, params->nr_cu);
    ENCODE_PARAM(json, "nr_thread", Number, params->nr_thread);
    ENCODE_PARAM(json, "time_slice", Number, params->csf_sched_option.time_slice);
    ENCODE_PARAM(json, "context_switch_time", Number, params->csf_sched_option.context_switch_time);
    ENCODE_PARAM(json, "csf_runtime_scale", Number, params->csf_runtime_scale);
    ENCODE_PARAM(json, "csf_sched_option", String, params->csf_sched_option.algo);
    ENCODE_PARAM(json, "grouped_csf", Number, params->csf_sched_option.take_care_of_group);
    ENCODE_PARAM(json, "grouped_csf_prio", Number,
          params->csf_sched_option.take_care_of_group_prio);
    return 0;
}

static CommandSetOps compute_ns_ops = {
    .state            = NULL,
    .init             = compute_init,
    .exit             = NULL,
    .rw_check_req     = NULL,
    .admin_cmd        = compute_admin_cmd,
    .io_cmd           = compute_io_cmd,
    .get_log          = NULL,
    .encode_params    = compute_encode_params,
    .decode_params    = compute_decode_params,
};

nvme_register(NVME_CSI_COMPUTE, &compute_ns_ops);
