/*
 * openrknn — NPU execution: patch DMA addresses + submit
 *
 * SPDX-License-Identifier: MIT
 */
#include "openrknn.h"
#include <stdlib.h>
#include <stdio.h>

/* Scan FB weight_data entries to build a map of blob offsets within BO[1].
 * Returns offsets for each type=0 blob (weight/bias data) and type=4/6 blobs. */
struct bo1_blob_info {
    uint32_t offset;
    uint32_t size;
    uint8_t  type;
};

static int scan_blob_offsets(struct orknn_model *m, struct bo1_blob_info *blobs,
                             int max_blobs)
{
    const uint8_t *fb = m->file_data + ((m->version > 1) ? 0x40 : 0x18);
    uint32_t root = orknn_fb_u32(fb, 0);
    int wt_field = (m->version > 5) ? 20 : 4;
    uint32_t wt_fpos = orknn_fb_field(fb, root, wt_field);
    if (!wt_fpos) return 0;

    uint32_t n_entries = orknn_fb_vec_len(fb, wt_fpos);
    uint32_t bo1_off = 0;
    int count = 0;

    for (uint32_t i = 1; i < n_entries && count < max_blobs; i++) {
        uint32_t entry = orknn_fb_vec_at(fb, wt_fpos, i);
        if (!entry) continue;
        uint8_t ttype = fb[entry + 66];
        uint32_t fo0 = orknn_fb_field(fb, entry, 0);
        if (!fo0) continue;
        uint32_t blen;
        orknn_fb_bytes(fb, fo0, &blen);
        if (blen == 0) continue;
        bo1_off = (bo1_off + 63) & ~63u;
        blobs[count].offset = bo1_off;
        blobs[count].size = blen;
        blobs[count].type = ttype;
        count++;
        bo1_off += blen;
    }
    return count;
}

static void patch_regcmd_addresses(struct orknn_context *ctx)
{
    struct orknn_model *m = &ctx->model;
    uint32_t rc_off = (uint32_t)(m->regcmd_data - m->wt_data);

    uint32_t wt_base = (uint32_t)ctx->weight_bo.dma_addr;
    uint32_t act_base = (uint32_t)ctx->activation_bo.dma_addr;
    uint32_t in_base = ctx->input_bos ? (uint32_t)ctx->input_bos[0].dma_addr : 0;
    uint32_t out_base = ctx->output_bos ? (uint32_t)ctx->output_bos[0].dma_addr : 0;

    /* Scan blob offsets to find weight, bias, and other data sections */
    struct bo1_blob_info blobs[128];
    int n_blobs = scan_blob_offsets(m, blobs, 128);

    /* Build per-operation weight/bias offset table.
     *
     * Weight/bias blobs come in pairs within BO[1]. The pairs are ordered
     * in reverse execution order: last-executing op's weights first in BO[1],
     * first-executing op's weights last. We detect pairs by scanning blobs
     * for consecutive (weight, bias) entries and assign to op_idx values
     * found in the task BO.
     *
     * Weight blob: larger, contains kernel data
     * Bias blob: smaller, immediately follows weight, contains bias values
     * Both can be type=0 or type=6.
     */
    struct { uint32_t wt_off; uint32_t bs_off; } op_wt_bs[16];
    int n_ops = 0;

    /* Collect weight+bias pairs from type=0 blobs.
     * Type=0 blobs are weight data, coming in pairs (weight, bias).
     * For models with additional operations beyond the type=0 pairs,
     * type=6 small blobs (not regcmd/task) serve as weight/bias for those ops. */
    /* PC2/PC3 blobs for em=0x60 tasks — detect early so we can skip them in pairing */
    uint32_t pc2_off = 0, pc3_off = 0;
    for (int i = 0; i < n_blobs; i++) {
        if (blobs[i].type == 4 && blobs[i].size <= 4096 && !pc2_off)
            pc2_off = blobs[i].offset;
        if (blobs[i].type == 6 && blobs[i].size <= 4096 &&
            blobs[i].offset != rc_off &&
            blobs[i].size != m->task_data_size && !pc3_off)
            pc3_off = blobs[i].offset;
    }

    struct { uint32_t wt_off; uint32_t bs_off; } pairs[16];
    int n_pairs = 0;

    /* Collect weight+bias pairs by grouping consecutive same-type blobs.
     * Skip: regcmd, task BO, type=4 blobs, PC3 blob, and 1024-byte type=6
     * blobs (per-channel metadata referenced by em=0x60 tasks). */
    for (int i = 0; i < n_blobs - 1 && n_pairs < 16; i++) {
        if (blobs[i].offset == rc_off) continue;
        if (blobs[i].size == m->task_data_size) continue;
        if (blobs[i].type == 4) continue;
        if (blobs[i].offset == pc3_off) continue;
        if (blobs[i].type == 6 && blobs[i].size == 1024) continue; /* per-ch metadata */
        int j = i + 1;
        while (j < n_blobs && (blobs[j].offset == rc_off ||
               blobs[j].size == m->task_data_size ||
               blobs[j].type == 4 || blobs[j].offset == pc3_off ||
               (blobs[j].type == 6 && blobs[j].size == 1024)))
            j++;
        if (j >= n_blobs) break;
        if (blobs[i].type == blobs[j].type) {
            pairs[n_pairs].wt_off = blobs[i].offset;
            pairs[n_pairs].bs_off = blobs[j].offset;
            n_pairs++;
            i = j;
        }
    }

    /* Discover unique op_idx values from CONV tasks (em=0x1d), in order of first appearance */
    struct { uint32_t f[8]; uint64_t regcmd_addr; } __attribute__((packed)) *tasks = ctx->task_bo.map;
    uint32_t op_ids[16];
    int n_op_ids = 0;
    for (uint32_t t = 0; t < m->task_count; t++) {
        uint32_t em = tasks[t].f[2];
        uint32_t op = tasks[t].f[1];
        if (em != 0x1d) continue; /* only CONV tasks have WT/BS */
        int found = 0;
        for (int k = 0; k < n_op_ids; k++)
            if (op_ids[k] == op) { found = 1; break; }
        if (!found && n_op_ids < 16)
            op_ids[n_op_ids++] = op;
    }

    /* Assign pairs to ops: last pair → first op_id, first pair → last op_id.
     * (Blobs stored in reverse execution order.) */
    for (int k = 0; k < n_op_ids && k < n_pairs; k++) {
        int pair_idx = n_pairs - 1 - k; /* reverse */
        op_wt_bs[k].wt_off = pairs[pair_idx].wt_off;
        op_wt_bs[k].bs_off = pairs[pair_idx].bs_off;
        orknn_log(2, "run: op_idx=%u -> wt=0x%x bs=0x%x (pair %d)",
                  op_ids[k], pairs[pair_idx].wt_off, pairs[pair_idx].bs_off, pair_idx);
    }
    n_ops = n_op_ids < n_pairs ? n_op_ids : n_pairs;

    /* Compute activation DST offset for first CONV output.
     * Proxy uses raw NCHW tensor size (H*W*C), NOT NC1HWC2 padded. */
    uint32_t act_dst_off = 0;
    if (m->n_inputs > 0 && m->inputs[0].n_dims == 4) {
        uint32_t H = m->inputs[0].dims[1];
        uint32_t W = m->inputs[0].dims[2];
        uint32_t C = m->inputs[0].dims[3];
        act_dst_off = H * W * C; /* raw NCHW size, e.g., 32*32*3=3072 */
    }

    orknn_log(1, "run: patching: wt=0x%x act=0x%x in=0x%x out=0x%x "
              "rc_off=0x%x act_dst=0x%x n_ops=%d n_pairs=%d",
              wt_base, act_base, in_base, out_base,
              rc_off, act_dst_off, n_ops, n_pairs);

    uint32_t patched = 0;

    /* Track which regcmd offsets we've already patched to avoid
     * double-patching shared regcmd sections (multi-core tasks share regcmd). */
    uint32_t patched_offsets[256];
    int n_patched_offsets = 0;

    for (uint32_t t = 0; t < m->task_count; t++) {
        uint32_t amt = tasks[t].f[6];
        uint32_t enable_mask = tasks[t].f[2];
        uint64_t addr = tasks[t].regcmd_addr;
        uint32_t bo_off = (uint32_t)(addr - ctx->weight_bo.dma_addr);

        /* Skip if we already patched this regcmd section */
        int already_done = 0;
        for (int j = 0; j < n_patched_offsets; j++) {
            if (patched_offsets[j] == bo_off) { already_done = 1; break; }
        }
        if (already_done) continue;
        if (n_patched_offsets < 256) patched_offsets[n_patched_offsets++] = bo_off;

        uint64_t *entries = (uint64_t *)((uint8_t *)ctx->weight_bo.map + bo_off);
        uint32_t total = amt + 4;

        int is_conv = (enable_mask == 0x1d);
        int is_reformat = (enable_mask == 0x18);
        uint32_t op = tasks[t].f[1]; /* op_idx */

        /* Find this task's WT/BS offsets from per-op table */
        uint32_t task_wt_off = 0, task_bs_off = 0;
        for (int k = 0; k < n_ops; k++) {
            if (op_ids[k] == op) {
                task_wt_off = op_wt_bs[k].wt_off;
                task_bs_off = op_wt_bs[k].bs_off;
                break;
            }
        }

        /* CNA input conversion parameters */
        int32_t input_zp = 0;
        if (m->n_inputs > 0) input_zp = m->inputs[0].zp;
        uint16_t cvt_offset = (uint16_t)(input_zp & 0xFFFF);
        uint16_t cvt_scale = 0x4000;
        uint32_t cvt_con0 = 0x000e38e0;
        uint32_t cvt_con1 = ((uint32_t)cvt_scale << 16) | cvt_offset;
        uint32_t cvt_con5 = 0x00000fff;

        for (uint32_t e = 0; e < total; e++) {
            uint16_t reg = entries[e] & 0xFFFF;
            uint32_t val = (entries[e] >> 16) & 0xFFFFFFFF;
            uint32_t new_val = val;
            int do_patch = 0;

            switch (reg) {
            case 0x1070: /* SRC_BASE */
                /* CONV tasks read from input BO (val=0 means input start).
                 * Non-zero val = activation offset. */
                if (val == 0 && is_conv)
                    new_val = in_base;
                else if (val != 0)
                    new_val = act_base + val;
                do_patch = (new_val != val);
                break;

            case 0x1110: /* WT_BASE */
                new_val = wt_base + (val ? val : task_wt_off);
                do_patch = 1;
                break;

            case 0x4020: /* DST_BASE */
                if (is_reformat) {
                    /* REFORMAT tasks write to output BO */
                    new_val = out_base + val;
                } else if (val == 0) {
                    new_val = act_base + act_dst_off;
                } else {
                    new_val = act_base + val;
                }
                do_patch = 1;
                break;

            case 0x5018: /* RDMA_ACT — reads activation */
                if (val == 0 && !is_conv) {
                    /* REFORMAT reads from where CONV wrote: act + act_dst_off */
                    new_val = act_base + act_dst_off;
                    do_patch = 1;
                } else if (val != 0) {
                    /* Non-zero offset: add to act_dst_off (the base where
                     * CONV output starts) for REFORMAT tasks */
                    if (is_reformat)
                        new_val = act_base + act_dst_off + val;
                    else
                        new_val = act_base + val;
                    do_patch = 1;
                }
                break;

            case 0x5020: /* BS_BASE */
                if (val == 0 && is_conv) {
                    new_val = wt_base + task_bs_off;
                    do_patch = 1;
                } else if (val != 0) {
                    new_val = wt_base + val;
                    do_patch = 1;
                }
                break;

            case 0x0010: /* PC_BASE — chain pointer */
                if (val != 0) {
                    new_val = wt_base + rc_off + val;
                    do_patch = 1;
                }
                break;

            case 0x6070: /* PC2 — points to per-channel data in BO[1] */
                if (val != 0) {
                    new_val = wt_base + val;
                    do_patch = 1;
                } else if (enable_mask == 0x60 && pc2_off) {
                    new_val = wt_base + pc2_off;
                    do_patch = 1;
                }
                break;

            case 0x701c: /* PC3 — points to per-channel data in BO[1] */
                if (val != 0) {
                    new_val = wt_base + val;
                    do_patch = 1;
                } else if (enable_mask == 0x60 && pc3_off) {
                    new_val = wt_base + pc3_off;
                    do_patch = 1;
                }
                break;

            case 0x4110: /* WDMA_BASE */
            case 0x5038: /* RDMA related */
                if (val != 0) {
                    new_val = act_base + val;
                    do_patch = 1;
                }
                break;

            /* CNA input conversion registers — runtime-computed from quant params */
            case 0x104c: /* CNA_CVT_CON0 */
                if (is_conv) { new_val = cvt_con0; do_patch = 1; }
                break;
            case 0x1050: /* CNA_CVT_CON1 */
            case 0x1054: /* CNA_CVT_CON2 */
            case 0x1058: /* CNA_CVT_CON3 */
                if (is_conv) { new_val = cvt_con1; do_patch = 1; }
                break;
            case 0x1180: /* CNA_CVT_CON5 */
                if (is_conv) { new_val = cvt_con5; do_patch = 1; }
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

    /* Dump patched regcmd for debugging */
    const char *dump_path = getenv("ORKNN_DUMP_REGCMD");
    if (dump_path) {
        FILE *df = fopen(dump_path, "w");
        if (df) {
            for (uint32_t t = 0; t < m->task_count && t < 10; t++) {
                uint32_t amt = tasks[t].f[6];
                uint64_t addr = tasks[t].regcmd_addr;
                uint32_t bo_off2 = (uint32_t)(addr - ctx->weight_bo.dma_addr);
                uint64_t *ent = (uint64_t *)((uint8_t *)ctx->weight_bo.map + bo_off2);
                fprintf(df, "=== TASK[%u] addr=0x%lx bo_off=%u amt=%u em=0x%x ===\n",
                        t, (unsigned long)addr, bo_off2, amt, tasks[t].f[2]);
                for (uint32_t e2 = 0; e2 < amt + 4; e2++) {
                    uint16_t reg2 = ent[e2] & 0xFFFF;
                    uint32_t val2 = (ent[e2] >> 16) & 0xFFFFFFFF;
                    uint16_t tgt2 = (ent[e2] >> 48) & 0xFFFF;
                    fprintf(df, "  [%3u] tgt=0x%04x reg=0x%04x val=0x%08x\n",
                            e2, tgt2, reg2, val2);
                }
            }
            fclose(df);
            orknn_log(1, "run: dumped regcmd to %s", dump_path);
        }
    }

    /* Dump task BO binary for comparison with proxy */
    const char *task_dump = getenv("ORKNN_DUMP_TASKBO");
    if (task_dump) {
        FILE *tf = fopen(task_dump, "wb");
        if (tf) {
            fwrite(ctx->task_bo.map, 1, m->task_count * 40, tf);
            fclose(tf);
            orknn_log(1, "run: dumped task BO (%u tasks, %u bytes) to %s",
                      m->task_count, m->task_count * 40, task_dump);
        }
    }

    orknn_bo_sync_to_device(ctx->npu_fd, &ctx->weight_bo);
}

int orknn_own_run(struct orknn_context *ctx, rknn_run_extend *extend)
{
    (void)extend;
    struct orknn_model *m = &ctx->model;

    if (!ctx->hw_elapse_time) {
        orknn_log(1, "run: first run, patching DMA addresses...");
        patch_regcmd_addresses(ctx);
        ctx->hw_elapse_time = 1;
    }

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
