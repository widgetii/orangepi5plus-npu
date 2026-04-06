/*
 * openrknn — NPU execution: patch DMA addresses + submit
 *
 * The .rknn regcmd contains 0-based offsets for DMA addresses.
 * We add the appropriate BO DMA base address to each register:
 *   0x1070 SRC_BASE: + input_bo or activation_bo DMA
 *   0x1110 WT_BASE:  + weight_bo DMA
 *   0x4020 DST_BASE: + activation_bo DMA (or output_bo for last op)
 *   0x5020 BS_BASE:  + weight_bo DMA (bias data is within weight BO)
 *   0x0010 PC_BASE:  + weight_bo DMA (chain pointer to next regcmd)
 *
 * SPDX-License-Identifier: MIT
 */
#include "openrknn.h"
#include <stdlib.h>
#include <stdio.h>

/* The .rknn regcmd contains placeholder DMA addresses (zeros) that the
 * RKNN runtime patches during rknn_init based on its BO allocation layout.
 * We cannot reproduce this patching because:
 * 1. The offsets within the weight BO (e.g., bias at +2304) are computed
 *    by the RKNN compiler and not stored explicitly in the .rknn file
 * 2. The activation layout (which tensors at which offsets) is computed
 *    during graph building
 *
 * Solution: let the proxy do rknn_init (which patches the regcmd), then
 * intercept its BO data for our own submit. For now, skip own patching
 * and just log what we would have done.
 */
static void patch_regcmd_addresses(struct orknn_context *ctx)
{
    orknn_log(1, "run: regcmd DMA patching NOT YET IMPLEMENTED "
              "(needs proxy BO intercept)");

    /* Dump patched regcmd for debugging (ORKNN_DUMP_REGCMD=/path) */
    const char *dump_path = getenv("ORKNN_DUMP_REGCMD");
    if (dump_path) {
        FILE *df = fopen(dump_path, "w");
        if (df) {
            /* Dump first task's regcmd entries */
            struct {
                uint32_t f[8];
                uint64_t regcmd_addr;
            } __attribute__((packed)) *tasks = ctx->task_bo.map;

            for (uint32_t t = 0; t < ctx->model.task_count && t < 5; t++) {
                uint32_t amt = tasks[t].f[6]; /* regcfg_amount */
                uint64_t addr = tasks[t].regcmd_addr;
                uint32_t bo_off = (uint32_t)(addr - ctx->weight_bo.dma_addr);
                uint64_t *entries = (uint64_t *)((uint8_t *)ctx->weight_bo.map + bo_off);
                fprintf(df, "=== TASK[%u] addr=0x%lx bo_off=%u amt=%u ===\n",
                        t, (unsigned long)addr, bo_off, amt);
                for (uint32_t e = 0; e < amt + 4; e++) {
                    uint16_t reg = entries[e] & 0xFFFF;
                    uint32_t val = (entries[e] >> 16) & 0xFFFFFFFF;
                    uint16_t tgt = (entries[e] >> 48) & 0xFFFF;
                    fprintf(df, "  [%3u] tgt=0x%04x reg=0x%04x val=0x%08x\n",
                            e, tgt, reg, val);
                }
            }
            fclose(df);
            orknn_log(1, "run: dumped regcmd to %s", dump_path);
        }
    }

    /* Sync patched weight BO to device */
    orknn_bo_sync_to_device(ctx->npu_fd, &ctx->weight_bo);
}

int orknn_own_run(struct orknn_context *ctx, rknn_run_extend *extend)
{
    (void)extend;
    struct orknn_model *m = &ctx->model;

    /* Patch DMA addresses on first run */
    if (!ctx->hw_elapse_time) { /* use as patched flag */
        orknn_log(1, "run: first run, patching DMA addresses...");
        patch_regcmd_addresses(ctx);
        ctx->hw_elapse_time = 1; /* mark as patched */
    }

    /* ORKNN_NO_SUBMIT: skip actual NPU submit (for debugging regcmd) */
    if (getenv("ORKNN_NO_SUBMIT")) {
        orknn_log(1, "run: ORKNN_NO_SUBMIT set, skipping NPU submit");
        return RKNN_SUCC;
    }

    orknn_log(2, "run: submitting %u segments...", m->segment_count);

    /* Submit each segment to the NPU */
    for (uint32_t i = 0; i < m->segment_count; i++) {
        int ret = orknn_npu_submit(ctx->npu_fd, &ctx->task_bo, &m->segments[i]);
        if (ret) {
            orknn_log(0, "run: segment %u submit failed", i);
            return RKNN_ERR_FAIL;
        }
    }

    return RKNN_SUCC;
}
