#include "hw/femu/csd/sparf_writeback.h"

typedef struct WritebackTest {
    uint8_t *nvm;
    int fail;
    int calls;
    uint64_t now;
} WritebackTest;

static bool transfer(void *opaque, SparfWritebackPlan *plan, bool write,
                      uint64_t *start, uint64_t *end)
{
    WritebackTest *test = opaque;
    test->calls++;
    *start = test->now;
    if (test->fail == test->calls) return false;
    for (guint index = 0; index < plan->sectors->len; ++index) {
        uint8_t *disk = test->nvm + g_array_index(plan->sectors, uint64_t, index) * 512;
        uint8_t *memory = plan->image + index * 512;
        memcpy(write ? disk : memory, write ? memory : disk, 512);
    }
    test->now += write ? 500 : 100;
    *end = test->now;
    return true;
}

int run_writeback_diagnostics(const void *metadata, uint32_t length, const uint8_t *source,
                   uint64_t source_bytes, uint8_t *nvm, uint64_t nvm_bytes,
                   int fail, struct cemu_sparf_writeback_result *result,
                   struct cemu_sparf_writeback_diagnostics *diagnostics)
{
    SparfWritebackPlan *plan = sparf_wb_plan(metadata, length, source_bytes, nvm_bytes);
    if (!plan) return -1;
    WritebackTest test = { .nvm = nvm, .fail = fail, .now = 1000 };
    bool success = sparf_wb_run(plan, metadata, source, transfer, &test, result);
    result->ready = test.now;
    if (diagnostics) *diagnostics = plan->diagnostics;
    sparf_wb_free(plan);
    return success ? test.calls : -2;
}

int run_writeback(const void *metadata, uint32_t length, const uint8_t *source,
                   uint64_t source_bytes, uint8_t *nvm, uint64_t nvm_bytes,
                   int fail, struct cemu_sparf_writeback_result *result)
{
    return run_writeback_diagnostics(metadata, length, source, source_bytes, nvm,
                                    nvm_bytes, fail, result, NULL);
}
