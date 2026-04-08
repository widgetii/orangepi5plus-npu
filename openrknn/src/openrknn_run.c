/*
 * openrknn — NPU execution: patch DMA addresses + submit
 *
 * SPDX-License-Identifier: MIT
 */
#include "openrknn.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

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

/* Copy the proxy's fully-patched regcmd into our weight BO, then
 * rebase DMA addresses from proxy BO layout to ours.
 * This handles FC layers and other operations where the template
 * regcmd is completely rewritten by the runtime. */
static int copy_proxy_regcmd(struct orknn_context *ctx)
{
    if (!ctx->real_ctx) return -1;

    struct orknn_proxy *proxy = orknn_proxy_get();
    if (!proxy) return -1;

    /* Run proxy once so it patches everything.
     * Use UINT8 input type (matches bench_rknn convention) so the
     * proxy's CNA_CVT hardware does uint8→int8 conversion. */
    struct orknn_model *m = &ctx->model;
    uint32_t input_size = m->n_inputs > 0 ? m->inputs[0].size : 0;
    if (input_size > 0) {
        rknn_input inp;
        memset(&inp, 0, sizeof(inp));
        inp.index = 0;
        inp.type = RKNN_TENSOR_UINT8;
        inp.fmt = RKNN_TENSOR_NHWC;
        inp.size = input_size;
        inp.buf = calloc(1, input_size);
        proxy->rknn_inputs_set(ctx->real_ctx, 1, &inp);
        free(inp.buf);
    }
    proxy->rknn_run(ctx->real_ctx, NULL);

    /* Now the proxy's BOs have fully-patched data.
     * We need to read the proxy's weight BO. Unfortunately we can't
     * directly access proxy BOs — they're in the proxy's address space.
     *
     * Alternative: use rknn_query to check if the model ran, then
     * read the output. But we actually need the regcmd data...
     *
     * For now: check if an intercept dump exists at /tmp/rknn_dump/ */
    char path[128];
    snprintf(path, sizeof(path), "/tmp/rknn_dump/sub1_bo_001_%uB.bin",
             ctx->weight_bo.size);
    FILE *f = fopen(path, "rb");
    if (!f) {
        /* Try common sizes */
        for (uint32_t sz = ctx->weight_bo.size; sz > 0; sz -= 4096) {
            snprintf(path, sizeof(path), "/tmp/rknn_dump/sub1_bo_001_%uB.bin", sz);
            f = fopen(path, "rb");
            if (f) break;
        }
    }
    if (!f) {
        orknn_log(1, "run: no proxy BO dump found, using template patching");
        return -1;
    }

    /* Read proxy's weight BO */
    uint32_t rc_off = (uint32_t)(m->regcmd_data - m->wt_data);
    uint8_t *proxy_bo = calloc(1, ctx->weight_bo.size);
    size_t nread = fread(proxy_bo, 1, ctx->weight_bo.size, f);
    fclose(f);

    if (nread < rc_off + m->regcmd_size) {
        orknn_log(0, "run: proxy BO dump too small (%zu < %u)", nread, rc_off + m->regcmd_size);
        free(proxy_bo);
        return -1;
    }

    /* Copy the ENTIRE proxy BO[1] into our weight BO.
     * This includes both weight data and regcmd — ensures all runtime-
     * patched register values are correct. Weight data is the same as
     * ours (from the .rknn file), so only the regcmd section differs. */
    uint32_t copy_size = nread < ctx->weight_bo.size ? (uint32_t)nread : ctx->weight_bo.size;
    memcpy(ctx->weight_bo.map, proxy_bo, copy_size);
    orknn_log(1, "run: copied proxy BO[1] (%u bytes)", copy_size);

    /* Now rebase DMA addresses from proxy layout to ours.
     * Read proxy's BO addresses from the dump metadata. */
    char meta_path[128];
    snprintf(meta_path, sizeof(meta_path), "/tmp/rknn_dump/submit_1.txt");
    FILE *mf = fopen(meta_path, "r");
    uint32_t proxy_wt = 0, proxy_act = 0, proxy_in = 0, proxy_out = 0;
    if (mf) {
        char line[256];
        while (fgets(line, sizeof(line), mf)) {
            /* Parse: bo[N] handle=H dma=0xXXXX obj=0xXXXX size=S */
            uint32_t bi, dma, sz;
            if (sscanf(line, "bo[%u] handle=%*u dma=0x%x obj=%*s size=%u", &bi, &dma, &sz) == 3) {
                if (bi == 1) proxy_wt = dma;
                else if (bi == 2) proxy_act = dma;
                else if (bi == 3) proxy_in = dma;
                else if (bi == 4) proxy_out = dma;
            }
        }
        fclose(mf);
    }

    if (!proxy_wt) {
        orknn_log(0, "run: cannot read proxy BO addresses from dump");
        free(proxy_bo);
        return -1;
    }

    uint32_t our_wt = (uint32_t)ctx->weight_bo.dma_addr;
    uint32_t our_act = (uint32_t)ctx->activation_bo.dma_addr;
    uint32_t our_in = ctx->input_bos ? (uint32_t)ctx->input_bos[0].dma_addr : 0;
    uint32_t our_out = ctx->output_bos ? (uint32_t)ctx->output_bos[0].dma_addr : 0;

    orknn_log(1, "run: rebasing DMA: proxy wt=0x%x act=0x%x in=0x%x out=0x%x",
              proxy_wt, proxy_act, proxy_in, proxy_out);
    orknn_log(1, "run:               ours  wt=0x%x act=0x%x in=0x%x out=0x%x",
              our_wt, our_act, our_in, our_out);

    /* Rebase DMA addresses: only touch registers that are actually DMA
     * address registers AND whose values fall in a known proxy BO range. */
    uint64_t *rc = (uint64_t *)((uint8_t *)ctx->weight_bo.map + rc_off);
    uint32_t rc_entries = m->regcmd_size / 8;
    uint32_t rebased = 0;

    /* Read proxy BO sizes from submit metadata */
    uint32_t proxy_bo_sizes[5] = {0};
    {
        FILE *sf = fopen(meta_path, "r");
        if (sf) {
            char line2[256];
            while (fgets(line2, sizeof(line2), sf)) {
                uint32_t bi2, sz2;
                if (sscanf(line2, "bo[%u] handle=%*u dma=%*s obj=%*s size=%u", &bi2, &sz2) == 2)
                    if (bi2 < 5) proxy_bo_sizes[bi2] = sz2;
            }
            fclose(sf);
        }
    }
    uint32_t pw_end = proxy_wt + (proxy_bo_sizes[1] ? proxy_bo_sizes[1] : ctx->weight_bo.size);
    uint32_t pa_end = proxy_act + (proxy_bo_sizes[2] ? proxy_bo_sizes[2] : ctx->activation_bo.size);
    uint32_t pi_end = proxy_in + (proxy_bo_sizes[3] ? proxy_bo_sizes[3] : 0x100000);
    uint32_t po_end = proxy_out + (proxy_bo_sizes[4] ? proxy_bo_sizes[4] : 0x100000);

    /* Known DMA address registers (verified from librocketnpu rnpu_registers.h) */
    #define IS_DMA_REG(r) ( \
        (r) == 0x0010 || /* PC_BASE_ADDRESS */ \
        (r) == 0x1070 || /* CNA_SRC_BASE */ \
        (r) == 0x1110 || /* RDMA_WT_BASE */ \
        (r) == 0x4020 || /* DPU_DST_BASE */ \
        (r) == 0x4110 || /* WDMA_BASE */ \
        (r) == 0x5018 || /* RDMA activation */ \
        (r) == 0x5020 || /* RDMA_BS_BASE */ \
        (r) == 0x5038 || /* RDMA related */ \
        (r) == 0x6070 || /* PC related */ \
        (r) == 0x701c    /* PC related */ \
    )

    for (uint32_t i = 0; i < rc_entries; i++) {
        uint16_t reg = rc[i] & 0xFFFF;
        uint32_t val = (rc[i] >> 16) & 0xFFFFFFFF;
        if (val == 0) continue;
        if (!IS_DMA_REG(reg)) continue;

        uint32_t new_val = val;
        if (val >= proxy_wt && val < pw_end)
            new_val = our_wt + (val - proxy_wt);
        else if (val >= proxy_act && val < pa_end)
            new_val = our_act + (val - proxy_act);
        else if (proxy_in && val >= proxy_in && val < pi_end)
            new_val = our_in + (val - proxy_in);
        else if (proxy_out && val >= proxy_out && val < po_end)
            new_val = our_out + (val - proxy_out);
        else
            continue;

        if (new_val != val) {
            rc[i] = (rc[i] & 0xFFFF000000000000ULL) |
                    ((uint64_t)new_val << 16) | (rc[i] & 0xFFFF);
            rebased++;
        }
    }

    orknn_log(1, "run: rebased %u DMA entries", rebased);

    /* Discover output offsets in the activation BO.
     * The graph memory planner places each output tensor at a specific
     * offset in the activation BO. We find these offsets by:
     * 1. Calling proxy->rknn_outputs_get to get the expected output bytes
     * 2. Reading the proxy's post-run activation BO dump
     * 3. Searching the dump for the output signature (first N bytes) */
    char post_act_path[128];
    uint32_t proxy_act_size = proxy_bo_sizes[2] ? proxy_bo_sizes[2] : ctx->activation_bo.size;
    snprintf(post_act_path, sizeof(post_act_path),
             "/tmp/rknn_dump/post1_bo_002_%uB.bin", proxy_act_size);
    FILE *paf = fopen(post_act_path, "rb");
    uint8_t *proxy_act_data = NULL;
    if (paf) {
        proxy_act_data = calloc(1, proxy_act_size);
        fread(proxy_act_data, 1, proxy_act_size, paf);
        fclose(paf);
    }

    /* Also try to load the bench_rknn golden output file as a fallback
     * signature source. If bench_rknn was run earlier, it saved the
     * canonical proxy output at rknn_golden_<idx>.bin. */
    if (proxy_act_data) {
        /* Get proxy's output via rknn_outputs_get */
        rknn_output proxy_outputs[16];
        memset(proxy_outputs, 0, sizeof(proxy_outputs));
        for (uint32_t i = 0; i < m->n_outputs && i < 16; i++) {
            proxy_outputs[i].index = i;
            proxy_outputs[i].want_float = 0;
        }
        int oret = proxy->rknn_outputs_get(ctx->real_ctx, m->n_outputs, proxy_outputs, NULL);
        orknn_log(1, "run: proxy->rknn_outputs_get returned %d", oret);
        if (oret == 0) {
            for (uint32_t i = 0; i < m->n_outputs && i < 16; i++) {
                const uint8_t *out_bytes = (const uint8_t *)proxy_outputs[i].buf;
                uint32_t out_size = proxy_outputs[i].size;
                if (!out_bytes || out_size == 0) continue;

                /* If the proxy returned all-zp bytes (no anchor), try loading
                 * a bench_rknn golden file which was produced earlier */
                uint8_t zp_check = (uint8_t)(m->outputs[i].zp & 0xFF);
                int all_zp = 1;
                for (uint32_t k = 0; k < out_size && k < 256; k++) {
                    if (out_bytes[k] != zp_check) { all_zp = 0; break; }
                }
                static uint8_t golden_buf[0x100000];
                orknn_log(1, "run: all_zp=%d for output[%u]", all_zp, i);
                if (all_zp) {
                    char golden_path[128];
                    snprintf(golden_path, sizeof(golden_path),
                             "/root/npu-research/librocketnpu/tests/rknn_golden_%u.bin", i);
                    FILE *gf = fopen(golden_path, "rb");
                    orknn_log(1, "run: try golden %s: %s", golden_path, gf ? "opened" : "not found");
                    if (gf) {
                        size_t gread = fread(golden_buf, 1, sizeof(golden_buf), gf);
                        fclose(gf);
                        orknn_log(1, "run: golden read %zu bytes (need %u)", gread, out_size);
                        if (gread == out_size) {
                            out_bytes = golden_buf;
                            orknn_log(1, "run: using golden file for output[%u]", i);
                        }
                    }
                }

                /* Find this output's bytes in the proxy's activation BO.
                 * For quantized outputs the bytes are mostly at zero-point
                 * (e.g., 0x80 for int8 zp=-128). We need to anchor the
                 * search on BYTES THAT ARE DIFFERENT from the zero-point,
                 * otherwise the signature matches many places coincidentally. */
                int found_off = -1;
                uint8_t zp_byte = (uint8_t)(m->outputs[i].zp & 0xFF);

                /* Debug: dump first bytes of proxy output */
                orknn_log(1, "run: output[%u] size=%u zp=0x%02x first16=%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x",
                          i, out_size, zp_byte,
                          out_bytes[0],out_bytes[1],out_bytes[2],out_bytes[3],
                          out_bytes[4],out_bytes[5],out_bytes[6],out_bytes[7],
                          out_bytes[8],out_bytes[9],out_bytes[10],out_bytes[11],
                          out_bytes[12],out_bytes[13],out_bytes[14],out_bytes[15]);

                /* Find the first byte in out_bytes that differs from zp */
                int anchor = -1;
                for (uint32_t k = 0; k < out_size && k < 256; k++) {
                    if (out_bytes[k] != zp_byte) { anchor = (int)k; break; }
                }

                if (anchor >= 0) {
                    /* Build signature: [anchor-8 .. anchor+24], clipped to [0..out_size).
                     * This includes at least one distinctive byte. */
                    uint32_t sig_start = anchor > 8 ? (uint32_t)(anchor - 8) : 0;
                    uint32_t sig_end = anchor + 24;
                    if (sig_end > out_size) sig_end = out_size;
                    uint32_t sig_len = sig_end - sig_start;

                    for (uint32_t off = 0; off + sig_len <= proxy_act_size; off++) {
                        if (memcmp(proxy_act_data + off, out_bytes + sig_start, sig_len) == 0) {
                            found_off = (int)off - (int)sig_start;
                            orknn_log(2, "run: output[%u] sig anchored at byte %d, len=%u, matched at 0x%x (start 0x%x)",
                                      i, anchor, sig_len, off, found_off);
                            break;
                        }
                    }
                } else {
                    /* Output is entirely at zp — can't distinguish. Try first 64 bytes */
                    uint32_t sig_len = out_size < 64 ? out_size : 64;
                    for (uint32_t off = 0; off + sig_len <= proxy_act_size; off++) {
                        if (memcmp(proxy_act_data + off, out_bytes, sig_len) == 0) {
                            found_off = (int)off;
                            break;
                        }
                    }
                }

                if (found_off >= 0) {
                    ctx->act_output_offsets[i] = (uint32_t)found_off;
                    ctx->act_output_valid[i] = 1;
                    orknn_log(1, "run: output[%u] located at act+0x%x (size=%u)",
                              i, found_off, out_size);
                } else {
                    orknn_log(0, "run: output[%u] signature not found in act BO", i);
                }
            }
            proxy->rknn_outputs_release(ctx->real_ctx, m->n_outputs, proxy_outputs);
        }
        free(proxy_act_data);
    } else {
        orknn_log(1, "run: no proxy post-run BO dump at %s", post_act_path);
    }

    /* Dump rebased regcmd for debugging */
    const char *dump_path = getenv("ORKNN_DUMP_REGCMD");
    if (dump_path) {
        FILE *df = fopen(dump_path, "w");
        if (df) {
            struct { uint32_t f[8]; uint64_t regcmd_addr; } __attribute__((packed)) *tsk = ctx->task_bo.map;
            for (uint32_t t = 0; t < m->task_count && t < 10; t++) {
                uint32_t amt = tsk[t].f[6];
                uint64_t addr = tsk[t].regcmd_addr;
                uint32_t bo_off = (uint32_t)(addr - ctx->weight_bo.dma_addr);
                uint64_t *ent = (uint64_t *)((uint8_t *)ctx->weight_bo.map + bo_off);
                fprintf(df, "=== TASK[%u] addr=0x%lx bo_off=%u amt=%u em=0x%x ===\n",
                        t, (unsigned long)addr, bo_off, amt, tsk[t].f[2]);
                for (uint32_t e2 = 0; e2 < amt + 4; e2++) {
                    uint16_t reg2 = ent[e2] & 0xFFFF;
                    uint32_t val2 = (ent[e2] >> 16) & 0xFFFFFFFF;
                    uint16_t tgt2 = (ent[e2] >> 48) & 0xFFFF;
                    fprintf(df, "  [%3u] tgt=0x%04x reg=0x%04x val=0x%08x\n",
                            e2, tgt2, reg2, val2);
                }
            }
            fclose(df);
        }
    }

    /* Task BO regcmd_addr values were already set correctly by
     * orknn_alloc_model_bos — don't rebase them. */
    orknn_bo_sync_to_device(ctx->npu_fd, &ctx->weight_bo);

    free(proxy_bo);
    return 0;
}

static void patch_regcmd_addresses(struct orknn_context *ctx)
{
    /* Try to use proxy's fully-patched regcmd first.
     * This handles FC layers and complex operations correctly. */
    if (copy_proxy_regcmd(ctx) == 0) return;

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
