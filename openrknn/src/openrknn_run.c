/*
 * openrknn — NPU execution: patch DMA addresses + submit
 *
 * The .rknn regcmd stores 0 for all DMA base addresses (SRC, DST, WT, BS).
 * The RKNN runtime computes these during init from the model's BO layout.
 *
 * Our approach: compute BO offsets from the blob assembly order:
 * - BO[1] = assembled weight_data blobs (type=0 are weight/bias, type=6 are
 *   metadata/regcmd/task). Each type=0 blob's offset in BO[1] tells us
 *   where weight/bias data is.
 * - Per-task: WT_BASE and BS_BASE come from the type=0 blob offsets
 * - PC_BASE: add regcmd offset within BO[1] (non-zero template values)
 * - SRC/DST: point to activation BO (with per-operation offsets from the
 *   model's tensor allocation)
 *
 * SPDX-License-Identifier: MIT
 */
#include "openrknn.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

static void patch_regcmd_addresses(struct orknn_context *ctx)
{
    struct orknn_model *m = &ctx->model;
    uint32_t rc_off = (uint32_t)(m->regcmd_data - m->wt_data);
    uint32_t rc_entries = m->regcmd_size / 8;
    uint64_t *rc = (uint64_t *)((uint8_t *)ctx->weight_bo.map + rc_off);

    uint32_t wt_base = (uint32_t)ctx->weight_bo.dma_addr;
    uint32_t act_base = (uint32_t)ctx->activation_bo.dma_addr;
    uint32_t in_base = ctx->input_bos ? (uint32_t)ctx->input_bos[0].dma_addr : 0;
    uint32_t out_base = ctx->output_bos ? (uint32_t)ctx->output_bos[0].dma_addr : 0;

    /* Find weight and bias blob offsets within BO[1].
     * type=0 blobs: first is weight, second is bias.
     * These are stored in m->wt_data at known offsets from blob assembly. */
    uint32_t weight_off = 0; /* offset of weight data in BO[1] */
    uint32_t bias_off = 0;   /* offset of bias data in BO[1] */
    {
        /* Scan for type=0 blobs in the FB weight_data entries.
         * The blob assembly puts them at 64-byte aligned offsets. */
        const uint8_t *fb = m->file_data + ((m->version > 1) ? 0x40 : 0x18);
        uint32_t root = orknn_fb_u32(fb, 0);
        int wt_field = (m->version > 5) ? 20 : 4;
        uint32_t wt_fpos = orknn_fb_field(fb, root, wt_field);
        if (wt_fpos) {
            uint32_t n_entries = orknn_fb_vec_len(fb, wt_fpos);
            uint32_t bo1_off = 0;
            int wt_found = 0, bs_found = 0;
            for (uint32_t i = 1; i < n_entries; i++) {
                uint32_t entry = orknn_fb_vec_at(fb, wt_fpos, i);
                if (!entry) continue;
                uint8_t ttype = fb[entry + 66];
                uint32_t fo0 = orknn_fb_field(fb, entry, 0);
                if (!fo0) continue;
                uint32_t blen;
                orknn_fb_bytes(fb, fo0, &blen);
                if (blen == 0) continue;
                bo1_off = (bo1_off + 63) & ~63u;
                if (ttype == 0 && !wt_found) {
                    weight_off = bo1_off;
                    wt_found = 1;
                } else if (ttype == 0 && !bs_found) {
                    bias_off = bo1_off;
                    bs_found = 1;
                }
                bo1_off += blen;
            }
        }
    }

    orknn_log(1, "run: patching regcmd: wt_bo=0x%x act_bo=0x%x in_bo=0x%x out_bo=0x%x "
              "wt_off=0x%x bs_off=0x%x rc_off=0x%x",
              wt_base, act_base, in_base, out_base, weight_off, bias_off, rc_off);

    /* Parse task BO to determine per-task context (which tasks read input,
     * which write output, activation offsets). */
    struct {
        uint32_t f[8];
        uint64_t regcmd_addr;
    } __attribute__((packed)) *tasks = ctx->task_bo.map;

    /* Track activation write offset: each CONV task writes to a new location.
     * Simple linear allocation starting after input-sized gap. */
    uint32_t act_write_off = 0;

    /* Compute input size in activation (NC1HWC2 padded) */
    uint32_t input_act_size = 0;
    if (m->n_inputs > 0 && m->inputs[0].n_dims == 4) {
        uint32_t C = m->inputs[0].dims[3], H = m->inputs[0].dims[1], W = m->inputs[0].dims[2];
        input_act_size = ((C + 15) / 16) * H * W * 16;
    }
    act_write_off = input_act_size; /* first output goes after input area */

    uint32_t patched = 0;
    uint32_t task_regcmd_start = 0; /* byte offset of current task's regcmd in blob */

    for (uint32_t t = 0; t < m->task_count; t++) {
        uint32_t amt = tasks[t].f[6]; /* regcfg_amount */
        uint32_t enable_mask = tasks[t].f[2];
        uint64_t addr = tasks[t].regcmd_addr;
        uint32_t bo_off = (uint32_t)(addr - ctx->weight_bo.dma_addr);
        uint64_t *entries = (uint64_t *)((uint8_t *)ctx->weight_bo.map + bo_off);
        uint32_t total = amt + 4;

        /* Determine if this is first CONV task (reads from input BO) */
        int is_first_conv = (t == 0 && enable_mask == 0x1d);
        /* Determine if this is an output task (REFORMAT writing to output) */
        int is_output_task = (enable_mask == 0x18);

        for (uint32_t e = 0; e < total; e++) {
            uint16_t reg = entries[e] & 0xFFFF;
            uint32_t val = (entries[e] >> 16) & 0xFFFFFFFF;
            uint32_t new_val = val;
            int do_patch = 0;

            switch (reg) {
            case 0x1070: /* SRC_BASE — input or activation */
                if (val == 0 && is_first_conv)
                    new_val = in_base; /* read from input BO */
                else if (val == 0)
                    new_val = act_base; /* read from activation */
                else
                    new_val = act_base + val;
                do_patch = 1;
                break;

            case 0x1110: /* WT_BASE — weight within BO[1] */
                if (val == 0)
                    new_val = wt_base + weight_off;
                else
                    new_val = wt_base + val;
                do_patch = 1;
                break;

            case 0x4020: /* DST_BASE — activation or output */
                if (val == 0 && is_output_task)
                    new_val = out_base;
                else if (val == 0)
                    new_val = act_base + act_write_off;
                else
                    new_val = act_base + val;
                do_patch = 1;
                break;

            case 0x4110: /* WDMA_BASE */
            case 0x5038: /* RDMA related */
                if (val == 0) {
                    /* These are often 0 meaning "not used" */
                    do_patch = 0;
                } else {
                    new_val = act_base + val;
                    do_patch = 1;
                }
                break;

            case 0x5018: /* RDMA_ACT — reads from previous activation */
                if (val == 0 && t > 0) {
                    /* Read from where previous CONV task wrote */
                    new_val = act_base + act_write_off;
                    do_patch = 1;
                } else if (val != 0) {
                    new_val = act_base + val;
                    do_patch = 1;
                }
                break;

            case 0x5020: /* BS_BASE — bias within BO[1] */
                if (val == 0 && enable_mask == 0x1d) {
                    new_val = wt_base + bias_off;
                    do_patch = 1;
                } else if (val != 0) {
                    new_val = wt_base + val;
                    do_patch = 1;
                }
                /* BS=0 for non-CONV tasks means "no bias" — don't patch */
                break;

            case 0x0010: /* PC_BASE — chain to next task's regcmd */
                if (val != 0) {
                    new_val = wt_base + rc_off + val;
                    do_patch = 1;
                }
                break;

            case 0x6070: /* PC related */
            case 0x701c: /* PC related */
                if (val != 0) {
                    new_val = wt_base + val;
                    do_patch = 1;
                }
                break;
            }

            if (do_patch && new_val != val) {
                entries[e] = (entries[e] & 0xFFFF000000000000ULL) |
                             ((uint64_t)new_val << 16) |
                             (entries[e] & 0xFFFF);
                patched++;
            }
        }
    }

    orknn_log(1, "run: patched %u entries across %u tasks", patched, m->task_count);

    /* Dump patched regcmd for debugging (ORKNN_DUMP_REGCMD=/path) */
    const char *dump_path = getenv("ORKNN_DUMP_REGCMD");
    if (dump_path) {
        FILE *df = fopen(dump_path, "w");
        if (df) {
            for (uint32_t t = 0; t < m->task_count && t < 5; t++) {
                uint32_t amt = tasks[t].f[6];
                uint64_t addr = tasks[t].regcmd_addr;
                uint32_t bo_off = (uint32_t)(addr - ctx->weight_bo.dma_addr);
                uint64_t *entries = (uint64_t *)((uint8_t *)ctx->weight_bo.map + bo_off);
                fprintf(df, "=== TASK[%u] addr=0x%lx bo_off=%u amt=%u em=0x%x ===\n",
                        t, (unsigned long)addr, bo_off, amt, tasks[t].f[2]);
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
    if (!ctx->hw_elapse_time) {
        orknn_log(1, "run: first run, patching DMA addresses...");
        patch_regcmd_addresses(ctx);
        ctx->hw_elapse_time = 1;
    }

    /* ORKNN_NO_SUBMIT: skip actual NPU submit (for debugging regcmd) */
    if (getenv("ORKNN_NO_SUBMIT")) {
        orknn_log(1, "run: ORKNN_NO_SUBMIT set, skipping NPU submit");
        return RKNN_SUCC;
    }

    orknn_log(2, "run: submitting %u segments...", m->segment_count);

    for (uint32_t i = 0; i < m->segment_count; i++) {
        int ret = orknn_npu_submit(ctx->npu_fd, &ctx->task_bo, &m->segments[i]);
        if (ret) {
            orknn_log(0, "run: segment %u submit failed", i);
            return RKNN_ERR_FAIL;
        }
    }

    return RKNN_SUCC;
}
