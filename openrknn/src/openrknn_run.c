/*
 * openrknn — NPU execution: patch DMA addresses + submit
 *
 * SPDX-License-Identifier: MIT
 */
#include "openrknn.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

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

    /* Iterate from i=0 so bo1_off tracks the real BO[1] layout produced
     * by extract_npu_data, which also starts from weight_data[0]. An
     * earlier version started at i=1 which left scan_blob_offsets's
     * offsets shifted vs what's actually in the weight BO, breaking
     * lookups like "blob ending at rc_off". */
    for (uint32_t i = 0; i < n_entries && count < max_blobs; i++) {
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
     * Parse all proxy BOs from the dump metadata.
     * BO[0] = task BO (skip)
     * BO[1] = weight BO
     * BO[2] = activation BO
     * BO[3] = input BO
     * BO[4..n] = output BOs (one per output, in order) */
    char meta_path[128];
    snprintf(meta_path, sizeof(meta_path), "/tmp/rknn_dump/submit_1.txt");

    uint32_t proxy_bo_dma[16] = {0};
    uint32_t proxy_bo_sizes[16] = {0};
    int n_proxy_bos = 0;
    {
        FILE *mf = fopen(meta_path, "r");
        if (mf) {
            char line[256];
            while (fgets(line, sizeof(line), mf)) {
                uint32_t bi, dma, sz;
                if (sscanf(line, "bo[%u] handle=%*u dma=0x%x obj=%*s size=%u",
                           &bi, &dma, &sz) == 3 && bi < 16) {
                    proxy_bo_dma[bi] = dma;
                    proxy_bo_sizes[bi] = sz;
                    if ((int)bi + 1 > n_proxy_bos) n_proxy_bos = bi + 1;
                }
            }
            fclose(mf);
        }
    }

    uint32_t proxy_wt = proxy_bo_dma[1];
    uint32_t proxy_act = proxy_bo_dma[2];
    uint32_t proxy_in = proxy_bo_dma[3];

    if (!proxy_wt) {
        orknn_log(0, "run: cannot read proxy BO addresses from dump");
        free(proxy_bo);
        return -1;
    }

    /* Cache the proxy's BO[3] (first input BO) contents for use by
     * inputs_set. The proxy's W-padding bytes (e.g. 0x80 for DeepLabv3
     * with mean/std=127.5) can't be derived from FB metadata, so we
     * snapshot them from the dump and restore after every inputs_set
     * data write. For W 16-aligned models (MBv1, YOLOv5), the proxy
     * BO[3] is all zeros so the cache copy is a no-op. */
    if (!ctx->proxy_input_cache && proxy_bo_sizes[3] > 0) {
        char ipath[128];
        snprintf(ipath, sizeof(ipath), "/tmp/rknn_dump/sub1_bo_003_%uB.bin",
                 proxy_bo_sizes[3]);
        FILE *f3 = fopen(ipath, "rb");
        if (f3) {
            ctx->proxy_input_cache = malloc(proxy_bo_sizes[3]);
            if (ctx->proxy_input_cache) {
                ctx->proxy_input_cache_size = (uint32_t)fread(
                    ctx->proxy_input_cache, 1, proxy_bo_sizes[3], f3);
                orknn_log(1, "run: cached proxy BO[3] (%u bytes)",
                          ctx->proxy_input_cache_size);
            }
            fclose(f3);
        }
    }

    uint32_t our_wt = (uint32_t)ctx->weight_bo.dma_addr;
    uint32_t our_act = (uint32_t)ctx->activation_bo.dma_addr;
    uint32_t our_in = ctx->input_bos ? (uint32_t)ctx->input_bos[0].dma_addr : 0;

    orknn_log(1, "run: rebasing DMA: proxy_wt=0x%x act=0x%x in=0x%x (%d BOs)",
              proxy_wt, proxy_act, proxy_in, n_proxy_bos);
    orknn_log(1, "run:               ours_wt=0x%x act=0x%x in=0x%x",
              our_wt, our_act, our_in);

    /* Build output BO mapping: proxy BO[4..n] → our output BOs[0..n_outputs).
     * The proxy may have FEWER output BOs than our n_outputs (some are
     * stored in the activation BO instead). Match by index. */
    uint32_t proxy_out_dma[16] = {0}, proxy_out_size[16] = {0};
    uint32_t our_out_dma[16] = {0};
    int n_out_bos = 0;
    for (int b = 4; b < n_proxy_bos && n_out_bos < 16; b++) {
        if (n_out_bos < (int)m->n_outputs && ctx->output_bos) {
            proxy_out_dma[n_out_bos] = proxy_bo_dma[b];
            proxy_out_size[n_out_bos] = proxy_bo_sizes[b];
            our_out_dma[n_out_bos] = (uint32_t)ctx->output_bos[n_out_bos].dma_addr;
            n_out_bos++;
        }
    }
    for (int k = 0; k < n_out_bos; k++) {
        orknn_log(2, "run: output BO[%d] proxy=0x%x->ours=0x%x size=%u",
                  k, proxy_out_dma[k], our_out_dma[k], proxy_out_size[k]);
    }

    /* Rebase DMA addresses: only touch registers that are actually DMA
     * address registers AND whose values fall in a known proxy BO range. */
    uint64_t *rc = (uint64_t *)((uint8_t *)ctx->weight_bo.map + rc_off);
    uint32_t rc_entries = m->regcmd_size / 8;
    uint32_t rebased = 0;

    uint32_t pw_end = proxy_wt + (proxy_bo_sizes[1] ? proxy_bo_sizes[1] : ctx->weight_bo.size);
    uint32_t pa_end = proxy_act + (proxy_bo_sizes[2] ? proxy_bo_sizes[2] : ctx->activation_bo.size);
    uint32_t pi_end = proxy_in + (proxy_bo_sizes[3] ? proxy_bo_sizes[3] : 0x100000);
    /* Proxy task BO (BO[0]) — models like ResNet50 embed per-task CVT
     * scale/offset tables inline in the task BO and reference them via
     * DMA pointers in the regcmd. */
    uint32_t proxy_task = proxy_bo_dma[0];
    uint32_t pt_end = proxy_task + proxy_bo_sizes[0];
    uint32_t our_task = (uint32_t)ctx->task_bo.dma_addr;

    /* Known DMA address registers (verified from librocketnpu rnpu_registers.h
     * plus empirically from ResNet50 regcmd analysis). */
    #define IS_DMA_REG(r) ( \
        (r) == 0x0010 || /* PC_BASE_ADDRESS */ \
        (r) == 0x1070 || /* CNA_SRC_BASE */ \
        (r) == 0x1110 || /* RDMA_WT_BASE */ \
        (r) == 0x1184 || /* CNA per-task CVT table ptr (ResNet50) */ \
        (r) == 0x4020 || /* DPU_DST_BASE */ \
        (r) == 0x4074 || /* DPU per-task data ptr (ResNet50) */ \
        (r) == 0x4080 || /* DPU_OUT_CVT_OFFSET table ptr (ResNet50) */ \
        (r) == 0x4110 || /* WDMA_BASE */ \
        (r) == 0x5018 || /* RDMA activation */ \
        (r) == 0x5020 || /* RDMA_BS_BASE */ \
        (r) == 0x502c || /* RDMA BS extended ptr (ResNet50) */ \
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
        int matched = 0;

        if (val >= proxy_wt && val < pw_end) {
            new_val = our_wt + (val - proxy_wt);
            matched = 1;
        } else if (val >= proxy_act && val < pa_end) {
            new_val = our_act + (val - proxy_act);
            matched = 1;
        } else if (proxy_in && val >= proxy_in && val < pi_end) {
            new_val = our_in + (val - proxy_in);
            matched = 1;
        } else if (proxy_task && val >= proxy_task && val < pt_end) {
            new_val = our_task + (val - proxy_task);
            matched = 1;
        } else {
            /* Check each output BO range */
            for (int k = 0; k < n_out_bos; k++) {
                uint32_t pob = proxy_out_dma[k];
                uint32_t pob_end = pob + proxy_out_size[k];
                if (val >= pob && val < pob_end) {
                    new_val = our_out_dma[k] + (val - pob);
                    matched = 1;
                    break;
                }
            }
        }

        if (!matched) continue;

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
    /* Find the highest post<N>_bo_002 dump — that's the final activation BO
     * state after all submits. Models with multiple submits (YOLO has 18)
     * only have valid output data in the LAST post dump. */
    int max_post = 0;
    for (int n = 1; n <= 64; n++) {
        char test_path[128];
        snprintf(test_path, sizeof(test_path),
                 "/tmp/rknn_dump/post%d_bo_002_%uB.bin", n, proxy_act_size);
        FILE *tf = fopen(test_path, "rb");
        if (tf) { fclose(tf); max_post = n; }
    }
    if (max_post == 0) max_post = 1;
    snprintf(post_act_path, sizeof(post_act_path),
             "/tmp/rknn_dump/post%d_bo_002_%uB.bin", max_post, proxy_act_size);
    orknn_log(1, "run: using post%d act BO dump for output discovery", max_post);
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
                 *
                 * The NPU writes raw NC1HWC2-formatted data to the activation
                 * BO. The user-visible output returned by rknn_outputs_get is
                 * de-tiled (NCHW or NHWC). For 4D outputs with fmt=NCHW, the
                 * byte order differs — we must INVERT the detile transform to
                 * reconstruct the raw NC1HWC2 bytes, then search for those.
                 *
                 * For non-4D outputs (1D, 2D) or fmt=UNDEFINED, the user bytes
                 * appear directly in the activation BO (no transform needed).
                 */
                int found_off = -1;
                uint8_t zp_byte = (uint8_t)(m->outputs[i].zp & 0xFF);
                struct orknn_tensor_info *oti = &m->outputs[i];

                /* Build raw signature buffers. Four candidates:
                 *   nc1hwc2_nchw: NC1HWC2 layout, source interpreted as NCHW
                 *   nc1hwc2_nhwc: NC1HWC2 layout, source interpreted as NHWC
                 *   hwc1c2_nchw:  HWC1C2 layout,  source interpreted as NCHW
                 *   hwc1c2_nhwc:  HWC1C2 layout,  source interpreted as NHWC
                 * We don't trust the model's fmt field because proxies may
                 * return different physical orderings. Try all that fit.
                 * For non-4D outputs we fall back to the raw linear bytes. */
                static uint8_t raw_sig_a[0x200000]; /* HBWCH16+NHWC */
                static uint8_t raw_sig_b[0x200000]; /* HBWCH16+NCHW */
                static uint8_t raw_sig_c[0x200000]; /* NC1HWC2+NHWC */
                static uint8_t raw_sig_d[0x200000]; /* NC1HWC2+NCHW */
                static uint8_t raw_sig_e[0x200000]; /* HWC1C2 +NHWC */
                static uint8_t raw_sig_f[0x200000]; /* HWC1C2 +NCHW */
                uint32_t sig_lens[6] = {0};
                const uint8_t *sig_ptrs[6] = {
                    raw_sig_a, raw_sig_b, raw_sig_c, raw_sig_d, raw_sig_e, raw_sig_f
                };
                /* layout tag stored on successful match:
                 * 0=NC1HWC2, 1=HWC1C2, 3=HBWCH16 */
                uint8_t sig_layouts[6] = {3, 3, 0, 0, 1, 1};
                /* user-output byte order: 0=NHWC (c-minor), 1=NCHW (c-major).
                 * Matches the writes: sig_a=HBWCH16+NHWC, sig_b=HBWCH16+NCHW,
                 * sig_c=NC1HWC2+NHWC, sig_d=NC1HWC2+NCHW, sig_e=HWC1C2+NHWC,
                 * sig_f=HWC1C2+NCHW. */
                uint8_t sig_src_orders[6] = {0, 1, 0, 1, 0, 1};

                if (oti->n_dims == 4) {
                    uint32_t N = oti->dims[0], H = oti->dims[1];
                    uint32_t W = oti->dims[2], C = oti->dims[3];
                    uint32_t c2 = 16;
                    uint32_t C1 = (C + c2 - 1) / c2;
                    uint32_t padC = C1 * c2;
                    uint32_t H_blk = (H + 15) / 16;
                    uint32_t padH = H_blk * 16;
                    uint32_t tile_len = N * C1 * H * W * c2;
                    /* HBWCH16 per-h_blk stride = W*C*16 aligned up to 64 bytes.
                     * YOLOv5 outputs are naturally 64-aligned; DeepLabv3
                     * (W*C*16 = 21840 not aligned) gets +48 padding per h_blk. */
                    uint32_t hbwch16_stride = ((W * C * 16) + 63u) & ~63u;
                    uint32_t hbwch16_len = N * H_blk * hbwch16_stride;

                    if (tile_len <= sizeof(raw_sig_c) &&
                        hbwch16_len <= sizeof(raw_sig_a)) {
                        /* HEAD order: sig_a/b=HBWCH16, sig_c/d=NC1HWC2,
                         * sig_e/f=HWC1C2. HBWCH16 uses hbwch16_len (64-
                         * aligned stride); the others use tile_len. */
                        memset(raw_sig_a, zp_byte, hbwch16_len);
                        memset(raw_sig_b, zp_byte, hbwch16_len);
                        memset(raw_sig_c, zp_byte, tile_len);
                        memset(raw_sig_d, zp_byte, tile_len);
                        memset(raw_sig_e, zp_byte, tile_len);
                        memset(raw_sig_f, zp_byte, tile_len);
                        sig_lens[0] = sig_lens[1] = hbwch16_len;
                        sig_lens[2] = sig_lens[3] = tile_len;
                        sig_lens[4] = sig_lens[5] = tile_len;

                        for (uint32_t n = 0; n < N; n++) {
                            for (uint32_t h = 0; h < H; h++) {
                                uint32_t h_blk = h / 16;
                                uint32_t h_in = h % 16;
                                for (uint32_t w = 0; w < W; w++) {
                                    for (uint32_t c = 0; c < C; c++) {
                                        uint32_t c1 = c / c2;
                                        uint32_t c2_idx = c % c2;
                                        uint32_t nchw_off = ((n * C + c) * H + h) * W + w;
                                        uint32_t nhwc_off = ((n * H + h) * W + w) * C + c;
                                        uint32_t nc1hwc2_off =
                                            ((n * C1 + c1) * H + h) * W * c2 + w * c2 + c2_idx;
                                        uint32_t hwc1c2_off =
                                            ((n * H + h) * W + w) * padC + c;
                                        uint32_t hbwch16_off =
                                            (n * H_blk + h_blk) * hbwch16_stride
                                            + w * C * 16 + c * 16 + h_in;
                                        raw_sig_a[hbwch16_off] = out_bytes[nhwc_off];
                                        raw_sig_b[hbwch16_off] = out_bytes[nchw_off];
                                        raw_sig_c[nc1hwc2_off] = out_bytes[nhwc_off];
                                        raw_sig_d[nc1hwc2_off] = out_bytes[nchw_off];
                                        raw_sig_e[hwc1c2_off]  = out_bytes[nhwc_off];
                                        raw_sig_f[hwc1c2_off]  = out_bytes[nchw_off];
                                    }
                                }
                            }
                        }
                        (void)padH;
                    }
                }

                /* Build list of candidate sigs. */
                const uint8_t *cands[7];
                uint32_t cand_sizes[7];
                uint8_t cand_layouts[7];
                uint8_t cand_src_orders[7];
                int n_cands = 0;
                for (int s = 0; s < 6; s++) {
                    if (sig_lens[s] > 0) {
                        cands[n_cands] = sig_ptrs[s];
                        cand_sizes[n_cands] = sig_lens[s];
                        cand_layouts[n_cands] = sig_layouts[s];
                        cand_src_orders[n_cands] = sig_src_orders[s];
                        n_cands++;
                    }
                }
                /* Always include linear fallback. */
                cands[n_cands] = out_bytes;
                cand_sizes[n_cands] = out_size;
                cand_layouts[n_cands] = 2;
                cand_src_orders[n_cands] = 0;
                n_cands++;

                /* Search outer loop: sig_len (longest → shortest).
                 * Inner loop: each candidate sig.
                 * This picks the sig that matches with the LONGEST unambiguous
                 * byte sequence — avoiding false-positive short matches that
                 * would mis-tag the layout. */
                uint32_t try_lens[] = {4096, 1024, 512, 256, 128, 64, 32, 16};
                int n_try = sizeof(try_lens)/sizeof(try_lens[0]);

                static uint8_t out_bo_buf[16][0x200000]; /* cache per BO */
                static uint32_t out_bo_cached_size[16] = {0};
                /* Load all proxy output BO dumps once */
                for (int b = 4; b < n_proxy_bos; b++) {
                    int ci = b - 4;
                    if (ci >= 16 || out_bo_cached_size[ci] > 0) continue;
                    char obo_path[128];
                    snprintf(obo_path, sizeof(obo_path),
                             "/tmp/rknn_dump/post%d_bo_%03d_%uB.bin",
                             max_post, b, proxy_bo_sizes[b]);
                    FILE *of = fopen(obo_path, "rb");
                    if (!of) continue;
                    size_t sz = proxy_bo_sizes[b] < sizeof(out_bo_buf[0])
                                ? proxy_bo_sizes[b] : sizeof(out_bo_buf[0]);
                    out_bo_cached_size[ci] = (uint32_t)fread(out_bo_buf[ci], 1, sz, of);
                    fclose(of);
                }

                /* Fast path: most proxies lay out output BOs in natural
                 * order (proxy BO[4 + i] → user output[i]). Try this
                 * direct mapping for each candidate layout before doing
                 * the expensive full scan. Critical for models with many
                 * outputs like YOLOv8 (9 outputs) where sig search tends
                 * to collide on tensors with similar value distributions. */
                if (4 + (int)i < n_proxy_bos && oti->n_dims == 4) {
                    int direct_ci = (int)i;
                    if (direct_ci < 16) {
                        uint32_t obo_read = out_bo_cached_size[direct_ci];
                        if (obo_read > 0) {
                            /* Try each 4D-layout candidate at offset 0.
                             * Skip the linear fallback (cl=2): for linear
                             * outputs the data usually lives in the act BO,
                             * not the output BO, so the direct mapping to
                             * the output BO would give zeros. */
                            for (int ci2 = 0; ci2 < n_cands && found_off < 0; ci2++) {
                                const uint8_t *cb2 = cands[ci2];
                                uint32_t cs2 = cand_sizes[ci2];
                                uint8_t cl2 = cand_layouts[ci2];
                                uint8_t so2 = cand_src_orders[ci2];
                                if (cl2 == 2) continue; /* skip linear */
                                uint32_t verify_len = cs2 < 4096 ? cs2 : 4096;
                                if (verify_len < 64) continue;
                                if (verify_len > obo_read) continue;
                                if (memcmp(out_bo_buf[direct_ci], cb2, verify_len) == 0) {
                                    found_off = 0;
                                    ctx->act_output_offsets[i] = 0;
                                    ctx->act_output_valid[i] = 2 + direct_ci;
                                    ctx->act_output_layout[i] = cl2;
                                    ctx->act_output_src_order[i] = so2;
                                    orknn_log(1, "run: output[%u] direct proxy BO[%d] @ 0x0 (verify_len=%u, cand=%d, layout=%u, src=%s)",
                                              i, 4 + direct_ci, verify_len, ci2, cl2,
                                              so2 ? "NCHW" : "NHWC");
                                    break;
                                }
                            }
                        }
                    }
                }

                /* For each sig length (long → short), for each candidate:
                 *   - HBWCH16 layouts (cl=3): search OUTPUT BOs only. Our
                 *     NPU's activation BO doesn't carry this tiled form —
                 *     the true data lives in the dedicated output BO.
                 *   - NC1HWC2/HWC1C2/linear (cl=0/1/2): search ACT BO
                 *     first, then output BOs as fallback. Matches the
                 *     baseline behavior for MBv1/conv/FC models. */
                for (int tl = 0; tl < n_try && found_off < 0; tl++) {
                    uint32_t want_len = try_lens[tl];
                    for (int ci = 0; ci < n_cands && found_off < 0; ci++) {
                        const uint8_t *cb = cands[ci];
                        uint32_t cs = cand_sizes[ci];
                        uint8_t cl = cand_layouts[ci];
                        uint8_t so = cand_src_orders[ci];

                        int anchor = -1;
                        for (uint32_t k = 0; k < cs && k < 4096; k++) {
                            if (cb[k] != zp_byte) { anchor = (int)k; break; }
                        }

                        uint32_t sig_start, sig_len;
                        if (anchor >= 0) {
                            sig_start = anchor > 8 ? (uint32_t)(anchor - 8) : 0;
                            uint32_t sig_end = sig_start + want_len;
                            if (sig_end > cs) sig_end = cs;
                            sig_len = sig_end - sig_start;
                        } else {
                            sig_start = 0;
                            sig_len = want_len < cs ? want_len : cs;
                        }
                        if (sig_len < 4) continue;

                        int search_act_first = (cl != 3);

                        /* Activation BO (if preferred) */
                        if (search_act_first) {
                            for (uint32_t off = 0; off + sig_len <= proxy_act_size; off++) {
                                if (memcmp(proxy_act_data + off, cb + sig_start, sig_len) == 0) {
                                    found_off = (int)off - (int)sig_start;
                                    ctx->act_output_offsets[i] = (uint32_t)found_off;
                                    ctx->act_output_valid[i] = 1;
                                    ctx->act_output_layout[i] = cl;
                                    ctx->act_output_src_order[i] = so;
                                    orknn_log(1, "run: output[%u] ACT BO @ 0x%x (sig_len=%u, cand=%d, layout=%u, src=%s)",
                                              i, found_off, sig_len, ci, cl, so ? "NCHW" : "NHWC");
                                    break;
                                }
                            }
                        }

                        /* Output BOs */
                        if (found_off < 0) {
                            for (int b = 4; b < n_proxy_bos && found_off < 0; b++) {
                                int obo_ci = b - 4;
                                if (obo_ci >= 16) continue;
                                uint32_t obo_read = out_bo_cached_size[obo_ci];
                                if (!obo_read) continue;
                                for (uint32_t off = 0; off + sig_len <= obo_read; off++) {
                                    if (memcmp(out_bo_buf[obo_ci] + off, cb + sig_start, sig_len) == 0) {
                                        found_off = (int)off - (int)sig_start;
                                        if (obo_ci < (int)m->n_outputs) {
                                            ctx->act_output_offsets[i] = (uint32_t)found_off;
                                            ctx->act_output_valid[i] = 2 + obo_ci;
                                            ctx->act_output_layout[i] = cl;
                                            ctx->act_output_src_order[i] = so;
                                            orknn_log(1, "run: output[%u] proxy BO[%d] @ 0x%x -> output_bos[%d] (sig_len=%u, cand=%d, layout=%u, src=%s)",
                                                      i, b, found_off, obo_ci, sig_len, ci, cl, so ? "NCHW" : "NHWC");
                                        }
                                        break;
                                    }
                                }
                            }
                        }
                    }
                }

                if (found_off < 0) {
                    orknn_log(0, "run: output[%u] signature not found anywhere", i);
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
     * This handles FC layers and complex operations correctly.
     * ORKNN_FORCE_TEMPLATE=1 skips the proxy path so the template-patch
     * code below runs unconditionally — used by the phase-0 diff oracle
     * to surface per-register discrepancies against the vendor BO[1]. */
    if (!getenv("ORKNN_FORCE_TEMPLATE") && copy_proxy_regcmd(ctx) == 0) return;

    struct orknn_model *m = &ctx->model;
    uint32_t rc_off = (uint32_t)(m->regcmd_data - m->wt_data);

    uint32_t wt_base = (uint32_t)ctx->weight_bo.dma_addr;
    uint32_t act_base = (uint32_t)ctx->activation_bo.dma_addr;
    uint32_t in_base = ctx->input_bos ? (uint32_t)ctx->input_bos[0].dma_addr : 0;
    uint32_t out_base = ctx->output_bos ? (uint32_t)ctx->output_bos[0].dma_addr : 0;

    /* Scan blob offsets to find weight, bias, and other data sections */
    /* ResNet50 has 190 weight_data entries and the PC LUT blobs sit
     * near the end (indices 186/187), so a 128-entry cap used to cut
     * them off. Use a dynamic allocation sized to the model's actual
     * entry count, with 1024 as a safe upper limit. */
    struct bo1_blob_info blobs[1024];
    int n_blobs = scan_blob_offsets(m, blobs, 1024);

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
    /* PC2/PC3 blobs for em=0x60 tasks.
     *
     * Empirical pattern from ResNet50 (verified via diff oracle): the
     * two shared per-channel correction LUTs are a pair of 1024-byte
     * type=6 blobs whose second blob ends exactly at rc_off (the start
     * of the regcmd blob). PC2 = first of the pair, PC3 = second.
     *
     * For pool-style em=0x60 tasks (MaxPool / AveragePool) the target
     * isn't a weight-BO LUT but an activation tensor — those are
     * handled per-task in the register patch loop using the memory plan,
     * not via these globals. */
    /* Find the PC2/PC3 per-channel LUT pair. In every model we've
     * observed (mobilenet_v1, resnet50, yolov5, yolov8, deeplabv3) the
     * pair is two 1024-byte type=6 blobs sitting near the regcmd blob
     * at the tail of the weight BO.
     *
     * Assignment rule (verified byte-exact against the phase-0 diff
     * oracle on all 4 runtime models that use em=0x60 tasks):
     *
     *   1. Find the TWO 1024-byte type=6 blobs closest to rc_off
     *      (searching backwards from the regcmd blob). Call them
     *      closer (smallest (rc_off - blob_end)) and farther.
     *   2. If closer's end lies within 64 bytes of rc_off (YOLOv5,
     *      YOLOv8, ResNet50: the pair is packed immediately before the
     *      regcmd), then closer = PC3 and farther = PC2.
     *   3. Otherwise (DeepLabv3: the compiler inserted small
     *      per-channel metadata blobs between the PC pair and rc_off),
     *      closer = PC2 and farther = PC3.
     *
     * Semantically PC3 is the per-channel read source and PC2 is the
     * write destination; when the two are packed tight against the
     * regcmd the write-side slot (PC2) comes first and the read-side
     * slot (PC3) is the one abutting rc_off. DeepLabv3 inverts this
     * because the intermediate metadata blobs get allocated after the
     * PC2 write slot but before rc_off. */
    uint32_t pc2_off = 0, pc3_off = 0;
    {
        int best_closer = -1, best_farther = -1;
        uint32_t best_closer_end = 0;
        for (int i = 0; i < n_blobs; i++) {
            if (blobs[i].type != 6 || blobs[i].size != 1024) continue;
            if (blobs[i].size == m->task_data_size) continue;
            uint32_t end = blobs[i].offset + blobs[i].size;
            if (end > rc_off) continue;
            if (best_closer < 0 || end > best_closer_end) {
                best_farther = best_closer;
                best_closer = i;
                best_closer_end = end;
            } else if (best_farther < 0 ||
                       (blobs[i].offset + 1024) >
                       blobs[best_farther].offset + 1024) {
                best_farther = i;
            }
        }
        if (best_closer >= 0 && best_farther >= 0) {
            uint32_t closer_off = blobs[best_closer].offset;
            uint32_t farther_off = blobs[best_farther].offset;
            uint32_t closer_end = closer_off + 1024;
            if (rc_off - closer_end < 64) {
                /* Closer blob ends right at rc_off → it's PC3. */
                pc3_off = closer_off;
                pc2_off = farther_off;
            } else {
                /* Gap between closer and rc_off → closer is PC2. */
                pc2_off = closer_off;
                pc3_off = farther_off;
            }
        }
    }
    orknn_log(1, "run: PC LUT blobs: pc2=0x%x pc3=0x%x rc_off=0x%x",
              pc2_off, pc3_off, rc_off);

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
    /* Dedup tracking for unique regcmd sections. Sized to cover the
     * largest model in our suite (DeepLabv3 = 1858 unique sections).
     * If this ever overflows we'd re-patch sections multiple times,
     * producing garbage values — see the log warning below. */
    #define MAX_PATCHED_OFFSETS 4096
    uint32_t *patched_offsets = calloc(MAX_PATCHED_OFFSETS, sizeof(uint32_t));
    int n_patched_offsets = 0;

    /* Build a list of task-BO sources to iterate. The primary source is
     * ctx->task_bo.map (derived from m->task_data by orknn_alloc_model_bos,
     * already rebased so regcmd_addr is in our weight BO's DMA range).
     * Per-segment per-cycle task BO snapshots in m->segments[s].task_bo_data[c]
     * are VENDOR-addressed and need runtime rebasing. Without including
     * them, CVT patching only covers op_idx tasks in segment 0, and
     * later segments' input-consuming conv tasks stay on template
     * placeholders (breaks MBv1 multi-cycle etc.).
     *
     * For segment sources we read the vendor weight-BO base from
     * /tmp/rknn_dump/submit_1.txt. That's a dump dependency that phase 9
     * will eventually remove along with the per-cycle task BO snapshots. */
    struct task_entry { uint32_t f[8]; uint64_t regcmd_addr; }
        __attribute__((packed));
    const struct task_entry *task_srcs[32];
    uint32_t task_src_counts[32];
    uint64_t task_src_wt_base[32]; /* per-source DMA base for rebasing */
    int n_task_srcs = 0;
    task_srcs[0] = (const struct task_entry *)ctx->task_bo.map;
    task_src_counts[0] = m->task_count;
    task_src_wt_base[0] = ctx->weight_bo.dma_addr;
    n_task_srcs = 1;

    /* Read vendor weight BO base for segment sources. */
    uint64_t vendor_wt_base = 0;
    {
        FILE *mf = fopen("/tmp/rknn_dump/submit_1.txt", "r");
        if (mf) {
            char line[256];
            while (fgets(line, sizeof(line), mf)) {
                if (strncmp(line, "bo[1]", 5) == 0) {
                    char *dp = strstr(line, "dma=0x");
                    if (dp) vendor_wt_base = strtoull(dp + 6, NULL, 16);
                    break;
                }
            }
            fclose(mf);
        }
    }

    for (uint32_t s = 0; s < m->segment_count && n_task_srcs < 32; s++) {
        for (uint32_t c = 0; c < m->segments[s].n_cycles && n_task_srcs < 32; c++) {
            uint8_t *td = m->segments[s].task_bo_data[c];
            uint32_t tsz = m->segments[s].task_bo_size[c];
            if (td && tsz >= 40 && vendor_wt_base) {
                task_srcs[n_task_srcs] = (const struct task_entry *)td;
                task_src_counts[n_task_srcs] = tsz / 40;
                task_src_wt_base[n_task_srcs] = vendor_wt_base;
                n_task_srcs++;
            }
        }
    }

    for (int src = 0; src < n_task_srcs; src++) {
    const struct task_entry *tasks = task_srcs[src];
    uint32_t src_count = task_src_counts[src];
    uint64_t src_wt_base = task_src_wt_base[src];
    /* Per-source sub-task counter: for consecutive REFORMAT tasks with the
     * same op_idx, this counts 0,1,2,... and maps each REFORMAT to a
     * distinct input tensor of the op. Multi-input Concat ops lower to N
     * consecutive REFORMAT tasks where sub-task k reads input_tensors[k]
     * and writes at a running offset within the output tensor. The
     * counter resets whenever op_idx changes (end of the op's REFORMAT
     * group) or the task is not a REFORMAT. */
    uint32_t prev_op_for_sub = UINT32_MAX;
    int prev_em_for_sub = -1;
    uint32_t reformat_sub_idx = 0;
    /* exSoftmax13 em=0x0d sub-task counter. The softmax lowering emits
     * three em=0x0d tasks per op (ReduceMax, rescale, ReduceSum) and
     * each reads CNA_WT_BASE from a different compile-time weight blob.
     * We track this counter separately from reformat_sub_idx (which
     * tracks em=0x18) because both groups live in the same op's task
     * sequence and their sub-indices advance independently. */
    uint32_t prev_op_for_em0d = UINT32_MAX;
    uint32_t em0d_sub_idx = 0;
    for (uint32_t t = 0; t < src_count; t++) {
        uint32_t amt = tasks[t].f[6];
        uint32_t enable_mask = tasks[t].f[2];
        uint64_t addr = tasks[t].regcmd_addr;
        /* Translate vendor-addressed regcmd to an offset within our
         * weight BO. For the primary source this is a no-op (both bases
         * are ctx->weight_bo.dma_addr). */
        uint32_t bo_off = (uint32_t)(addr - src_wt_base);

        /* Skip if we already patched this regcmd section */
        int already_done = 0;
        for (int j = 0; j < n_patched_offsets; j++) {
            if (patched_offsets[j] == bo_off) { already_done = 1; break; }
        }
        if (already_done) continue;
        if (n_patched_offsets < MAX_PATCHED_OFFSETS) {
            patched_offsets[n_patched_offsets++] = bo_off;
        } else {
            /* Overflow: we can't dedupe anymore. Any previously-patched
             * section that reappears here will be re-patched and produce
             * garbage. Log loudly so this doesn't silently corrupt. */
            static int warned = 0;
            if (!warned) {
                orknn_log(0, "run: patched_offsets overflow (>%u unique "
                             "regcmd sections) — results may be wrong",
                          MAX_PATCHED_OFFSETS);
                warned = 1;
            }
        }

        uint64_t *entries = (uint64_t *)((uint8_t *)ctx->weight_bo.map + bo_off);
        uint32_t total = amt + 4;

        int is_conv = (enable_mask == 0x1d);
        int is_reformat = (enable_mask == 0x18);
        uint32_t op = tasks[t].f[1]; /* op_idx */

        /* Update the REFORMAT sub-task counter. Only increments for
         * em=0x18 tasks and only resets when op_idx changes. Non-
         * REFORMAT tasks (em=0x0d softmax continuations) pass through
         * without affecting the counter, so exSoftmax's interleaved
         * 0x18/0x0d sequence still assigns `first em=0x18 task = sub 0`
         * regardless of how many em=0x0d tasks sat in between. */
        (void)prev_em_for_sub;
        if (is_reformat) {
            if (op == prev_op_for_sub) {
                reformat_sub_idx++;
            } else {
                reformat_sub_idx = 0;
            }
            prev_op_for_sub = op;
        } else if (op != prev_op_for_sub) {
            reformat_sub_idx = 0;
            prev_op_for_sub = UINT32_MAX;
        }

        /* exSoftmax em=0x0d sub-task counter — see the declaration
         * above the task loop. Resets per op, counts em=0x0d tasks. */
        int is_em0d = (enable_mask == 0x0d);
        if (is_em0d && op == prev_op_for_em0d) {
            em0d_sub_idx++;
        } else if (is_em0d) {
            em0d_sub_idx = 0;
            prev_op_for_em0d = op;
        } else if (op != prev_op_for_em0d) {
            em0d_sub_idx = 0;
            prev_op_for_em0d = UINT32_MAX;
        }

        /* Find this task's WT/BS offsets from per-op table */
        uint32_t task_wt_off = 0, task_bs_off = 0;
        for (int k = 0; k < n_ops; k++) {
            if (op_ids[k] == op) {
                task_wt_off = op_wt_bs[k].wt_off;
                task_bs_off = op_wt_bs[k].bs_off;
                break;
            }
        }

        /* Phase 4b: resolve per-op tensor offsets from the FB memory
         * plan. Sources:
         *   src_tensor_off   = tensor_offsets[ops[op].input_tensors[0]]
         *                      (primary activation input, in the act BO)
         *   dst_tensor_off   = tensor_offsets[ops[op].output_tensors[0]]
         *                      (activation output)
         *   rdma_tensor_off  = tensor_offsets[ops[op].input_tensors[3]]
         *                      (residual add operand for ConvAdd etc.,
         *                      falls back to dst for plain conv)
         *   wt_bo_off        = wt_blob_offsets[tensor_weight_blob[
         *                                     ops[op].input_tensors[1]]]
         *                      (weight data in BO[1], via f[18] lookup)
         *   bs_bo_off        = same for input_tensors[2] (bias)
         *
         * For input-consuming tasks the template SRC_BASE points at the
         * input BO, not the activation BO — user supplies raw bytes there. */
        uint32_t src_tensor_off = 0;
        uint32_t dst_tensor_off = 0;
        uint32_t rdma_tensor_off = 0;
        uint32_t ew_tensor_off = 0;
        uint32_t op_wt_bo_off = 0;
        uint32_t op_bs_bo_off = 0;
        /* op_in0_bo_off: weight-BO offset of the op's input_tensors[0]
         * blob, if it has one. Used by the InputOperator REFORMAT path
         * whose BS_BASE points at the op's mask blob (input[0]) rather
         * than the conventional bias slot (input[2]). */
        uint32_t op_in0_bo_off = 0;
        int have_op_wt = 0, have_op_bs = 0, have_op_in0 = 0;
        /* Non-zero if the op's output tensor is a subgraph output. When
         * set, REFORMAT tasks target the corresponding output BO rather
         * than the activation BO. sg_out_bo_idx is the index into
         * ctx->output_bos[] for the matching tensor. */
        int dst_is_sg_output = 0;
        int sg_out_bo_idx = -1;
        /* Source tensor: where this op reads its primary input from.
         * Also capture whether that tensor lives in a subgraph output
         * BO (YOLOv8 heads share intermediate feature tensors with the
         * subgraph output list, and subsequent convs need to read from
         * the corresponding output BO rather than the activation BO). */
        int src_is_sg_output = 0;
        int src_sg_bo_idx = -1;
        if (m->ops && op < m->op_count && m->tensor_offsets) {
            const struct orknn_op_info *oi = &m->ops[op];
            if (oi->input_count > 0) {
                /* Concat lowers to one REFORMAT task per input tensor;
                 * the sub-task counter selects which input this task
                 * reads from. All other ops (Conv, Resize, BatchNorm,
                 * etc.) always read from input_tensors[0] regardless
                 * of how many REFORMATs the lowering emits.
                 *
                 * Transpose lowers to a chain of REFORMAT tasks that
                 * stages data through input_tensors[1] as a scratch
                 * tensor: the first sub-task reads from input[0] (real
                 * data) and writes to input[1], and subsequent tasks
                 * read from and write to input[1] repeatedly until the
                 * final task copies the transposed result to the op's
                 * real output. So SRC = input[0] on sub_idx==0 and
                 * input[1] from sub_idx>=1. */
                uint32_t sub = 0;
                if (is_reformat) {
                    if (strcmp(oi->type, "Concat") == 0 &&
                        reformat_sub_idx < oi->input_count) {
                        /* Concat: one sub-task per input tensor. */
                        sub = reformat_sub_idx;
                    } else if ((strcmp(oi->type, "Transpose") == 0 ||
                                strncmp(oi->type, "exSoftmax", 9) == 0) &&
                               oi->input_count >= 2 &&
                               reformat_sub_idx > 0) {
                        /* Transpose and exSoftmax13 share the same
                         * scratch pattern: sub-task 0 reads input[0]
                         * (real data), all subsequent sub-tasks read
                         * from the scratch tensor at input[1]. */
                        sub = 1;
                    }
                } else if (enable_mask == 0x0d &&
                           strncmp(oi->type, "exSoftmax", 9) == 0 &&
                           oi->input_count >= 2) {
                    /* exSoftmax em=0x0d tasks always read from the
                     * scratch tensor (never from the real input). */
                    sub = 1;
                }
                uint32_t tidx = oi->input_tensors[sub];
                if (tidx < m->tensor_count) {
                    src_tensor_off = m->tensor_offsets[tidx];
                    if (m->tensor_is_sg_output &&
                        m->tensor_is_sg_output[tidx]) {
                        src_is_sg_output = 1;
                        if (m->sg_output_tensor_idx) {
                            for (uint32_t oi2 = 0; oi2 < m->n_outputs;
                                 oi2++) {
                                if (m->sg_output_tensor_idx[oi2] == tidx) {
                                    src_sg_bo_idx = (int)oi2;
                                    break;
                                }
                            }
                        }
                    }
                }
            }
            if (oi->output_count > 0) {
                /* Split lowers to one REFORMAT task per output tensor;
                 * the sub-task counter selects which output this task
                 * writes to. Mirrors the Concat rule on the src side. */
                int use_sub_dst = is_reformat &&
                                  strcmp(oi->type, "Split") == 0 &&
                                  reformat_sub_idx < oi->output_count;
                uint32_t dsub = use_sub_dst ? reformat_sub_idx : 0;
                uint32_t tidx = oi->output_tensors[dsub];
                if (tidx < m->tensor_count) {
                    dst_tensor_off = m->tensor_offsets[tidx];
                    if (m->tensor_is_sg_output &&
                        m->tensor_is_sg_output[tidx]) {
                        dst_is_sg_output = 1;
                        if (m->sg_output_tensor_idx) {
                            for (uint32_t oi2 = 0; oi2 < m->n_outputs; oi2++) {
                                if (m->sg_output_tensor_idx[oi2] == tidx) {
                                    sg_out_bo_idx = (int)oi2;
                                    break;
                                }
                            }
                        }
                    }
                }
            }
            /* Scratch-tensor DST override: both Transpose and
             * exSoftmax13 emit multi-REFORMAT / em=0x0d chains that
             * stage intermediate data through input_tensors[1] in the
             * activation BO. Only the final REFORMAT in the chain
             * writes to the real output — we detect that later via a
             * next-task lookahead and flip dst_tensor_off back. The
             * em=0x0d softmax continuation tasks always target the
             * scratch tensor. The output-override is applied after
             * the scratch-final flag is set (see below). */
            int op_uses_scratch =
                (strcmp(oi->type, "Transpose") == 0 ||
                 strncmp(oi->type, "exSoftmax", 9) == 0);
            if (op_uses_scratch &&
                (is_reformat || enable_mask == 0x0d) &&
                oi->input_count >= 2) {
                uint32_t tidx = oi->input_tensors[1];
                if (tidx < m->tensor_count)
                    dst_tensor_off = m->tensor_offsets[tidx];
            }
            (void)op_uses_scratch;
            if (oi->input_count > 3) {
                uint32_t tidx = oi->input_tensors[3];
                if (tidx < m->tensor_count)
                    rdma_tensor_off = m->tensor_offsets[tidx];
            } else {
                rdma_tensor_off = dst_tensor_off;
            }
            /* ElementWise operand: the last input tensor of the op,
             * used as the source for the DPU_RDMA_EW_BASE register in
             * REFORMAT tasks staging data for element-wise Add/Mul/Sub
             * ops (input_count == 2, EW = input[1]) and in conv tasks
             * with a fused residual (input_count == 4, EW = input[3]). */
            if (oi->input_count >= 2) {
                uint32_t tidx = oi->input_tensors[oi->input_count - 1];
                if (tidx < m->tensor_count)
                    ew_tensor_off = m->tensor_offsets[tidx];
            }
            /* input_tensors[0] weight-BO offset: only set if input[0]
             * has a weight blob (InputOperator's mask tensor). Most
             * ops have a data tensor at input[0] with no weight blob,
             * in which case this stays at 0. */
            if (oi->input_count > 0 && m->tensor_weight_blob &&
                m->wt_blob_offsets) {
                uint32_t in0_tidx = oi->input_tensors[0];
                if (in0_tidx < m->tensor_count) {
                    uint32_t blob_idx = m->tensor_weight_blob[in0_tidx];
                    if (blob_idx < m->wt_blob_count) {
                        op_in0_bo_off = m->wt_blob_offsets[blob_idx];
                        have_op_in0 = 1;
                    }
                }
            }
            /* Weight / bias: look up via tensor_weight_blob (FB f[18]).
             * For ops like Resize that reference compiler-generated
             * weight/bias blobs via name-prefix matching, prefer the
             * implicit_{wt,bs}_tidx resolved at parse time over
             * input_tensors[1]/[2] (those slots hold roi/scales tensors
             * for Resize and aren't weight blobs). */
            uint32_t wt_tidx_resolved = UINT32_MAX;
            uint32_t bs_tidx_resolved = UINT32_MAX;
            if (oi->implicit_wt_tidx != UINT32_MAX)
                wt_tidx_resolved = oi->implicit_wt_tidx;
            else if (oi->input_count > 1)
                wt_tidx_resolved = oi->input_tensors[1];
            if (oi->implicit_bs_tidx != UINT32_MAX)
                bs_tidx_resolved = oi->implicit_bs_tidx;
            else if (oi->input_count > 2)
                bs_tidx_resolved = oi->input_tensors[2];
            if (wt_tidx_resolved != UINT32_MAX &&
                m->tensor_weight_blob && m->wt_blob_offsets &&
                wt_tidx_resolved < m->tensor_count) {
                uint32_t blob_idx = m->tensor_weight_blob[wt_tidx_resolved];
                if (blob_idx < m->wt_blob_count) {
                    op_wt_bo_off = m->wt_blob_offsets[blob_idx];
                    have_op_wt = 1;
                }
            }
            if (bs_tidx_resolved != UINT32_MAX &&
                m->tensor_weight_blob && m->wt_blob_offsets &&
                bs_tidx_resolved < m->tensor_count) {
                uint32_t blob_idx = m->tensor_weight_blob[bs_tidx_resolved];
                if (blob_idx < m->wt_blob_count) {
                    op_bs_bo_off = m->wt_blob_offsets[blob_idx];
                    have_op_bs = 1;
                }
            }
        }
        int is_input_consuming_task_for_src =
            (op == m->input_consuming_op_idx);

        /* Phase 1 CVT patching. The template leaves CVT registers as
         * placeholders for the first conv that reads raw user input;
         * the vendor runtime computes them from mean/std/dtype/scale/zp
         * at rknn_run time. We mirror that computation here using the
         * attrs block parsed in openrknn_model.c:parse_fb_attrs().
         *
         * Gate: a task is "input-consuming" iff its op_idx (f[1]) matches
         * model->input_consuming_op_idx, the first op in the FB graph
         * whose primary input tensor is the subgraph input. SRC_BASE==0
         * is NOT a reliable gate — many tasks have that in the template
         * without actually reading user input, and patching them
         * corrupts values the vendor deliberately leaves as placeholders.
         *
         * Formula (validated byte-exact on 4/5 runtime models):
         *   trunc = 14 if trivial pre-processing else 15
         *   scale_hw[c] = round(2^trunc / (std[c] * tensor_scale))
         *   off_hw[c]   = round(-mean[c] * scale_hw[c]/2^trunc + tensor_zp)
         *
         * Special case: dtype=uint8 with trivial mean/std collapses to
         * identity rescale (16384) with offset=-128, centering uint8
         * pixels to int8. YOLOv5 (dtype=int8) is currently unsupported
         * and its tasks stay on the template values. */
        int is_input_consuming = is_conv && m->input_attr_valid
                                 && op == m->input_consuming_op_idx;
        uint32_t cvt_con0 = 0, cvt_con5 = 0;
        uint32_t cvt_con[4] = {0};
        if (is_input_consuming) {
            const float *mean = m->input_attr_mean;
            const float *std  = m->input_attr_std;
            const char  *dt   = m->input_attr_dtype;
            float scale = m->n_inputs > 0 ? m->inputs[0].scale : 0.0078125f;
            int32_t zp  = m->n_inputs > 0 ? m->inputs[0].zp    : 0;

            int trivial_mean_std =
                mean[0] == 0.0f && mean[1] == 0.0f && mean[2] == 0.0f &&
                std[0]  == 1.0f && std[1]  == 1.0f && std[2]  == 1.0f;
            int uniform_mean_zero_std =
                mean[0] == 0.0f && mean[1] == 0.0f && mean[2] == 0.0f &&
                std[0] == std[1] && std[1] == std[2] && std[0] > 0.0f;
            int symmetric_ms =
                mean[0] == mean[1] && mean[1] == mean[2] &&
                std[0]  == std[1]  && std[1]  == std[2]  &&
                mean[0] == std[0];

            int trunc = 14;
            int is_int8_trivial = (strcmp(dt, "int8") == 0 && trivial_mean_std);
            int is_uint8_like_trivial =
                !is_int8_trivial &&
                ((strcmp(dt, "uint8") == 0 && trivial_mean_std) ||
                 uniform_mean_zero_std || symmetric_ms);
            int is_float32 = (strcmp(dt, "float32") == 0);

            if (is_int8_trivial) {
                /* int8 user input with trivial mean/std (YOLOv5 case).
                 * The compiler derives the CVT scale from a rational
                 * approximation of the tensor scale:
                 *
                 *   inv_s  = 1 / tensor_scale
                 *   r_inv  = round(inv_s)
                 *   factor = inv_s / r_inv              (≈ 1.0 for clean
                 *                                        fractions)
                 *   scale_hw = round(2^trunc * factor)
                 *   trunc    = 15
                 *   offset   = 0
                 *
                 * For YOLOv5 (scale=0.01865845, zp=-14):
                 *   inv_s = 53.595, r_inv = 54, factor = 0.9925
                 *   scale_hw = round(32768 * 0.9925) = 32522 = 0x7f0a
                 *
                 * For scales that divide cleanly (e.g. 1/128, 1/255),
                 * r_inv equals inv_s so factor = 1 and scale_hw = 32768.
                 * The YOLOv5 0.9925 is the fractional remainder when the
                 * compiler picked an imperfect integer step count. */
                trunc = 15;
                float inv_s = 1.0f / scale;
                float r_inv = roundf(inv_s);
                if (r_inv == 0.0f) r_inv = 1.0f;
                float factor = inv_s / r_inv;
                int32_t sh = (int32_t)roundf((float)(1 << trunc) * factor);
                if (sh > 32767)  sh = 32767;
                if (sh < -32768) sh = -32768;
                uint32_t packed = ((uint32_t)(sh & 0xFFFF) << 16) | 0;
                for (int c = 0; c < 3; c++) cvt_con[c] = packed;
            } else if (is_uint8_like_trivial) {
                trunc = 14;
                uint16_t off = (uint16_t)((int16_t)-128);
                uint32_t packed = ((uint32_t)16384 << 16) | off;
                for (int c = 0; c < 4; c++) cvt_con[c] = packed;
            } else if (is_float32) {
                trunc = 15;
                int shift = 1 << trunc;
                for (int c = 0; c < 3; c++) {
                    float denom = std[c] * scale;
                    if (denom == 0.0f) denom = 1.0f;
                    /* Use roundf() (round-half-away-from-zero) not the
                     * `+ 0.5f; cast` trick — the latter is wrong for
                     * negative numbers and causes off-by-one errors in
                     * ResNet50's CVT offsets. */
                    int32_t sh = (int32_t)roundf((float)shift / denom);
                    int32_t oh_i = (int32_t)roundf(
                        -mean[c] * (float)sh / (float)shift + (float)zp);
                    if (oh_i < -32768) oh_i = -32768;
                    if (oh_i > 32767)  oh_i = 32767;
                    cvt_con[c] = ((uint32_t)(sh & 0xFFFF) << 16) |
                                 (uint32_t)(oh_i & 0xFFFF);
                }
            } else {
                is_input_consuming = 0;
            }
            if (is_input_consuming) {
                cvt_con0 = ((uint32_t)(trunc & 0x3f) << 4) |
                           ((uint32_t)(trunc & 0x3f) << 10) |
                           ((uint32_t)(trunc & 0x3f) << 16);
                cvt_con5 = 0x00000fff;
            }
        }

        /* W-alignment padding (DeepLabv3 W=513 → W_pad=528).
         *
         * The .rknn template holds the unpadded input width in:
         *   0x1020 CNA_DATA_SIZE0.DATAIN_WIDTH  (bits 16-26)
         *   0x107c CNA_DMA_CON1.LINE_STRIDE     (bits 0-27)
         *   0x1080 CNA_DMA_CON2.SURF_STRIDE     (bits 0-27, = W * H_factor)
         *
         * The NPU requires W aligned to 16, so for non-16-aligned inputs
         * (e.g. 513) the vendor runtime patches these three registers to
         * the padded width (528) before submit. We mirror that here, but
         * only for tasks belonging to the input-consuming op — subsequent
         * layers work on already-padded activation data so their DATA_SIZE
         * values are independent of the input W alignment. */
        uint32_t input_w = 0, input_w_pad = 0;
        int do_wpad = 0;
        if (is_input_consuming && m->n_inputs > 0 &&
            m->inputs[0].n_dims == 4) {
            input_w = m->inputs[0].dims[2];
            input_w_pad = (input_w + 15) & ~15u;
            if (input_w_pad != input_w && input_w > 0)
                do_wpad = 1;
        }

        /* Scratch-chain final-task detection: Transpose and exSoftmax
         * both lower to a chain of REFORMAT (em=0x18) tasks that stage
         * intermediate data through input_tensors[1] and only the last
         * REFORMAT in the group copies the result to the op's real
         * output. Flag the current task as "final" iff it's a REFORMAT
         * AND the next task is different (different op, or next-is-not
         * a REFORMAT) — i.e. we are the tail of a contiguous same-op
         * REFORMAT run. exSoftmax em=0x0d continuation tasks never
         * touch the real output, they always target the scratch. */
        int is_scratch_op = 0;
        int is_scratch_final = 0;
        if (is_reformat && m->ops && op < m->op_count) {
            const char *typ = m->ops[op].type;
            if (strcmp(typ, "Transpose") == 0 ||
                strncmp(typ, "exSoftmax", 9) == 0) {
                is_scratch_op = 1;
                /* Walk forward past trailing non-REFORMAT same-op tasks
                 * (exSoftmax interleaves em=0x18 and em=0x0d; the real
                 * last DMA write is the final em=0x18 before op_idx
                 * changes). */
                int found_later_reformat = 0;
                for (uint32_t s = t + 1; s < src_count; s++) {
                    uint32_t nxt_op = tasks[s].f[1];
                    if (nxt_op != op) break;
                    if (tasks[s].f[2] == 0x18) {
                        found_later_reformat = 1;
                        break;
                    }
                }
                if (!found_later_reformat)
                    is_scratch_final = 1;
            }
        }
        /* Kept as aliases so the existing patch-site code keeps
         * working without a rename pass. */
        int is_transpose = is_scratch_op;
        int is_transpose_final = is_scratch_final;

        /* Final scratch-chain task: flip dst_tensor_off back to the
         * op's real output tensor so the 0x4020 DST_BASE branch writes
         * there instead of the scratch. The earlier override set
         * dst_tensor_off = input[1].f13 for every scratch-op task. */
        if (is_scratch_final && m->ops && op < m->op_count &&
            m->ops[op].output_count > 0) {
            uint32_t out_tidx = m->ops[op].output_tensors[0];
            if (out_tidx < m->tensor_count)
                dst_tensor_off = m->tensor_offsets[out_tidx];
        }

        /* For exSoftmax op 30 em=0x0d, select the CNA_WT blob per
         * sub-index: sub 0 = ReduceMax (softmax_rmax_tidx),
         * sub 1 = rescale (input_tensors[2] — already op_wt_bo_off),
         * sub 2 = ReduceSum (softmax_rsum_tidx). Resolve to a weight
         * BO offset via tensor_weight_blob + wt_blob_offsets. */
        if (enable_mask == 0x0d && m->ops && op < m->op_count &&
            strncmp(m->ops[op].type, "exSoftmax", 9) == 0 &&
            m->tensor_weight_blob && m->wt_blob_offsets) {
            const struct orknn_op_info *oi = &m->ops[op];
            uint32_t pick_tidx = UINT32_MAX;
            if (em0d_sub_idx == 0)
                pick_tidx = oi->softmax_rmax_tidx;
            else if (em0d_sub_idx == 1)
                pick_tidx = oi->input_count > 2 ?
                            oi->input_tensors[2] : UINT32_MAX;
            else if (em0d_sub_idx == 2)
                pick_tidx = oi->softmax_rsum_tidx;
            if (pick_tidx != UINT32_MAX && pick_tidx < m->tensor_count) {
                uint32_t blob_idx = m->tensor_weight_blob[pick_tidx];
                if (blob_idx < m->wt_blob_count) {
                    op_wt_bo_off = m->wt_blob_offsets[blob_idx];
                    have_op_wt = 1;
                }
            }
        }

        for (uint32_t e = 0; e < total; e++) {
            uint16_t reg = entries[e] & 0xFFFF;
            uint32_t val = (entries[e] >> 16) & 0xFFFFFFFF;
            uint32_t new_val = val;
            int do_patch = 0;

            switch (reg) {
            case 0x1070: /* CNA_FEATURE_DATA_ADDR */
                if (is_input_consuming_task_for_src) {
                    /* First conv reads raw user input from input BO.
                     * Template val is the intra-tile offset within the
                     * input BO (usually 0 for the first tile). */
                    new_val = in_base + val;
                    do_patch = 1;
                } else if (is_conv && src_is_sg_output &&
                           src_sg_bo_idx >= 0 &&
                           (uint32_t)src_sg_bo_idx < m->n_outputs) {
                    /* Head conv reads from a tensor that's also a
                     * subgraph output (YOLOv8 detect head). Data is in
                     * the corresponding output BO. */
                    new_val = (uint32_t)ctx->output_bos[src_sg_bo_idx]
                                  .dma_addr + val;
                    do_patch = 1;
                } else if (is_conv) {
                    /* Subsequent convs read from the activation BO at the
                     * op's primary input tensor offset. */
                    new_val = act_base + src_tensor_off + val;
                    do_patch = 1;
                } else if (enable_mask == 0x0d && m->ops &&
                           op < m->op_count &&
                           strncmp(m->ops[op].type, "exSoftmax", 9) == 0) {
                    /* exSoftmax em=0x0d task reads from the scratch
                     * tensor (input_tensors[1]). src_tensor_off was
                     * already set to input[1].f13 above. */
                    new_val = act_base + src_tensor_off + val;
                    do_patch = 1;
                } else if (val != 0) {
                    new_val = act_base + val;
                    do_patch = 1;
                }
                break;

            case 0x1110: /* WT_BASE */
                /* Prefer the per-op weight offset from the FB memory
                 * plan (tensor.f[18] → wt_blob_offsets[]). Fall back to
                 * the old blob-pairing heuristic if the FB lookup didn't
                 * resolve (0 == val uses the fallback value). The
                 * template val is the intra-kernel offset within the
                 * weight blob (always 0 for start-of-kernel reads). */
                if (have_op_wt)
                    new_val = wt_base + op_wt_bo_off + val;
                else
                    new_val = wt_base + (val ? val : task_wt_off);
                do_patch = 1;
                break;

            case 0x4020: /* DST_BASE — output write */
                if (is_reformat && m->ops && op < m->op_count &&
                    strcmp(m->ops[op].type, "InputOperator") == 0) {
                    /* InputOperator REFORMATs pre-process the raw input
                     * buffer in place — both src and dst live in the
                     * input BO (DeepLabv3 task 0 writes at in+0x600). */
                    new_val = in_base + val;
                    do_patch = 1;
                } else if (is_reformat && amt >= 1000) {
                    /* "Sentinel" REFORMAT tasks (amt ~1097, seen in
                     * YOLOv5/v8): large-metadata tasks whose DMA regs
                     * point to wt_base + rc_off (start of the regcmd
                     * blob). These don't perform real DMA writes — the
                     * hardware treats them as chain/marker tasks. */
                    new_val = wt_base + rc_off;
                    do_patch = 1;
                } else if (is_transpose && !is_transpose_final) {
                    /* Intermediate Transpose REFORMAT: write to the
                     * scratch tensor (input[1]) in the activation BO,
                     * ignoring dst_is_sg_output. Only the final task in
                     * the chain writes to the real output. */
                    new_val = act_base + dst_tensor_off + val;
                    do_patch = 1;
                } else if (enable_mask == 0x0d && m->ops &&
                           op < m->op_count &&
                           strncmp(m->ops[op].type, "exSoftmax", 9) == 0) {
                    /* exSoftmax em=0x0d task writes to the scratch
                     * tensor (input_tensors[1]) — never to the op's
                     * real output, those always stay on em=0x18 paths. */
                    new_val = act_base + dst_tensor_off + val;
                    do_patch = 1;
                } else if (dst_is_sg_output && sg_out_bo_idx >= 0 &&
                    (uint32_t)sg_out_bo_idx < m->n_outputs) {
                    /* Writing to a subgraph output tensor: target the
                     * corresponding output BO directly. */
                    new_val = (uint32_t)ctx->output_bos[sg_out_bo_idx]
                                  .dma_addr + val;
                    do_patch = 1;
                } else if (is_reformat) {
                    /* Non-output REFORMAT: writes to an intermediate
                     * activation tensor in act BO. Use memory plan. */
                    new_val = act_base + dst_tensor_off + val;
                    do_patch = 1;
                } else {
                    /* Conv tasks write to the output tensor's allocation
                     * in the activation BO. Template val = intra-tile off. */
                    new_val = act_base + dst_tensor_off + val;
                    do_patch = 1;
                }
                break;

            case 0x5018: /* RDMA_ACT — reads activation for fused
                          * residuals (ConvAdd etc.) or as REFORMAT src */
                if (is_reformat && m->ops && op < m->op_count &&
                    strcmp(m->ops[op].type, "InputOperator") == 0) {
                    new_val = in_base + val;
                    do_patch = 1;
                } else if (is_reformat && amt >= 1000) {
                    /* Sentinel REFORMAT task — see DST_BASE comment. */
                    new_val = wt_base + rc_off;
                    do_patch = 1;
                } else if (is_conv) {
                    /* Only patch if this conv has a fused residual
                     * operand (ConvAdd / ConvReluAdd — input_count >= 4)
                     * OR if the template val is non-zero (explicit
                     * per-task read). Plain Conv leaves RDMA_ACT at 0 in
                     * the template, and the vendor leaves it at 0 too. */
                    if (m->ops && op < m->op_count &&
                        m->ops[op].input_count > 3) {
                        new_val = act_base + rdma_tensor_off + val;
                        do_patch = 1;
                    } else if (val != 0) {
                        new_val = act_base + rdma_tensor_off + val;
                        do_patch = 1;
                    }
                } else if (is_reformat) {
                    new_val = act_base + src_tensor_off + val;
                    do_patch = 1;
                } else if (val != 0) {
                    new_val = act_base + val;
                    do_patch = 1;
                }
                break;

            case 0x5020: /* BS_BASE (bias) */
                if (is_reformat) {
                    /* Only BatchNormalization REFORMATs emit a real
                     * BS_BASE pointer. Everything else — sentinel
                     * lowering, Concat, Conv/ConvEx* REFORMATs —
                     * leaves this at 0; only the corresponding CONV
                     * tasks write the bias pointer.
                     *
                     * For BatchNormalization, BS_BASE points at the
                     * gamma (scale) blob at wt_blob_offsets[input[1]]
                     * — which is `op_wt_bo_off` in our naming (not
                     * `op_bs_bo_off`, which points at beta and feeds
                     * into BN_BASE / 0x502c). Verified byte-exact on
                     * ResNet50 via the phase-0 diff oracle. */
                    if (m->ops && op < m->op_count && have_op_wt &&
                        strncmp(m->ops[op].type, "BatchNormalization",
                                18) == 0) {
                        new_val = wt_base + op_wt_bo_off + val;
                        do_patch = 1;
                    } else if (m->ops && op < m->op_count && have_op_in0 &&
                               strcmp(m->ops[op].type,
                                      "InputOperator") == 0) {
                        /* InputOperator's REFORMAT task stages the raw
                         * input buffer; BS_BASE points at a mask blob
                         * that input_tensors[0] references (DeepLabv3
                         * `sub_7:0_fill_stride_mask`). BN_BASE uses
                         * input[1] which my existing `op_wt_bo_off`
                         * already resolves. */
                        new_val = wt_base + op_in0_bo_off + val;
                        do_patch = 1;
                    } else {
                        new_val = 0;
                        do_patch = 1;
                    }
                } else if (have_op_bs) {
                    new_val = wt_base + op_bs_bo_off + val;
                    do_patch = 1;
                } else if (val == 0 && is_conv) {
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

            /* em=0x60 tasks emit PPU DMA to either a shared per-channel
             * LUT in the weight BO (for fused Conv ops whose template
             * regcmd emits PC2/PC3 as part of the per-channel correction
             * pass) or to activation tensors (MaxPool / AveragePool).
             * The discriminator is the op type string:
             *   - "MaxPool" / "AveragePool" / "GlobalAveragePool" →
             *     act_base + dst/src_tensor_off + val (val = intra-slice
             *     offset baked into the template, e.g. 0xa0 / 0x80).
             *   - Everything else (including Conv fused paths and "Add" /
             *     "Sub" / "Mul") → wt_base + pc2/pc3_off (shared LUT). */
            case 0x6070: /* PC2 — PPU_DST_BASE_ADDR */
                if (enable_mask == 0x60 && m->ops && op < m->op_count &&
                    (strstr(m->ops[op].type, "Pool") != NULL)) {
                    new_val = act_base + dst_tensor_off + val;
                    do_patch = 1;
                } else if (val != 0) {
                    new_val = wt_base + val;
                    do_patch = 1;
                } else if (enable_mask == 0x60 && pc2_off) {
                    new_val = wt_base + pc2_off;
                    do_patch = 1;
                }
                break;

            case 0x701c: /* PC3 — PPU_RDMA_SRC_BASE_ADDR */
                if (enable_mask == 0x60 && m->ops && op < m->op_count &&
                    (strstr(m->ops[op].type, "Pool") != NULL)) {
                    new_val = act_base + src_tensor_off + val;
                    do_patch = 1;
                } else if (val != 0) {
                    new_val = wt_base + val;
                    do_patch = 1;
                } else if (enable_mask == 0x60 && pc3_off) {
                    new_val = wt_base + pc3_off;
                    do_patch = 1;
                }
                break;

            /* 0x4110 is REG_DPU_LUT_LE_START (a 32-bit LUT threshold
             * value, not a DMA address). It's left as the template
             * emitted it. Previously mis-patched here as "WDMA_BASE",
             * which corrupted YOLOv8's sigmoid LUT config. */

            case 0x502c: /* DPU_RDMA_BN_BASE_ADDR — batch-norm beta
                          * (offset) table pointer, emitted for REFORMAT
                          * tasks of BatchNormalization ops. All other
                          * REFORMATs (Conv*, Concat, sentinel) leave
                          * this at 0. Points at wt_blob_offsets[input[2]]
                          * which is `op_bs_bo_off` in our naming — the
                          * beta/bias blob. Paired with BS_BASE (0x5020)
                          * which points at gamma. */
                if (is_reformat && m->ops && op < m->op_count &&
                    strncmp(m->ops[op].type, "BatchNormalization",
                            18) == 0 && have_op_bs) {
                    new_val = wt_base + op_bs_bo_off + val;
                    do_patch = 1;
                } else if (is_reformat && m->ops && op < m->op_count &&
                           have_op_wt &&
                           strcmp(m->ops[op].type,
                                  "InputOperator") == 0) {
                    /* InputOperator's BN_BASE points at the stride/bias
                     * blob referenced by input_tensors[1]. */
                    new_val = wt_base + op_wt_bo_off + val;
                    do_patch = 1;
                } else if (is_reformat) {
                    new_val = 0;
                    do_patch = 1;
                }
                break;

            case 0x5038: /* RDMA_EW_BASE — ElementWise secondary read.
                          * For element-wise binary REFORMATs (Add/Mul/Sub
                          * lowered as a single REFORMAT task with both
                          * operands fed via RDMA_SRC and RDMA_EW) or for
                          * conv tasks with a fused residual, point at the
                          * op's last input tensor in the activation BO.
                          * Concat REFORMATs leave this register at 0
                          * because Concat doesn't have a second operand. */
                if (is_reformat && amt >= 1000) {
                    /* Sentinel REFORMAT: vendor leaves EW at 0. */
                    new_val = 0;
                    do_patch = 1;
                } else if (is_conv && m->ops && op < m->op_count &&
                    m->ops[op].input_count > 3) {
                    new_val = act_base + rdma_tensor_off + val;
                    do_patch = 1;
                } else if (is_reformat && m->ops && op < m->op_count &&
                           ew_tensor_off &&
                           (strcmp(m->ops[op].type, "Add") == 0 ||
                            strcmp(m->ops[op].type, "Sub") == 0 ||
                            strcmp(m->ops[op].type, "Mul") == 0 ||
                            strcmp(m->ops[op].type, "Div") == 0 ||
                            strcmp(m->ops[op].type, "Min") == 0 ||
                            strcmp(m->ops[op].type, "Max") == 0)) {
                    new_val = act_base + ew_tensor_off + val;
                    do_patch = 1;
                } else if (val != 0) {
                    new_val = act_base + rdma_tensor_off + val;
                    do_patch = 1;
                }
                break;

            /* CNA CVT registers — only patched when this task belongs to
             * the input-consuming op (see comment + gate above the entry
             * loop). Values are computed once per task using the parsed
             * attrs block. */
            case 0x104c: /* CNA_CVT_CON0 */
                if (is_input_consuming) { new_val = cvt_con0; do_patch = 1; }
                break;
            case 0x1050: /* CNA_CVT_CON1 (channel 0) */
                if (is_input_consuming) { new_val = cvt_con[0]; do_patch = 1; }
                break;
            case 0x1054: /* CNA_CVT_CON2 (channel 1) */
                if (is_input_consuming) { new_val = cvt_con[1]; do_patch = 1; }
                break;
            case 0x1058: /* CNA_CVT_CON3 (channel 2) */
                if (is_input_consuming) { new_val = cvt_con[2]; do_patch = 1; }
                break;
            /* CNA_CVT_CON4 (0x105c, channel 3) is NOT patched — RGB models
             * leave it as the template placeholder and the vendor runtime
             * doesn't touch it either. */
            case 0x1180: /* CNA_CVT_CON5 */
                if (is_input_consuming) { new_val = cvt_con5; do_patch = 1; }
                break;

            /* W-alignment padding registers — only patched for the
             * input-consuming op when W isn't 16-aligned (DeepLabv3). */
            case 0x1020: /* CNA_DATA_SIZE0 */
                if (do_wpad) {
                    /* Replace DATAIN_WIDTH field (bits 16-26, 11 bits),
                     * preserve the lower 16 bits which hold
                     * DATAIN_HEIGHT + reserved. */
                    uint32_t height_bits = val & 0xFFFF;
                    new_val = height_bits |
                              ((input_w_pad & 0x7FF) << 16);
                    do_patch = (new_val != val);
                }
                break;
            case 0x107c: /* CNA_DMA_CON1 LINE_STRIDE (bits 0-27) */
                if (do_wpad) {
                    new_val = (val & 0xF0000000) | (input_w_pad & 0x0FFFFFFF);
                    do_patch = (new_val != val);
                }
                break;
            case 0x1080: /* CNA_DMA_CON2 SURF_STRIDE (bits 0-27) */
                if (do_wpad && input_w > 0) {
                    /* SURF_STRIDE = LINE_STRIDE * H_factor, so scale by
                     * W_pad/W. Use integer math that avoids precision
                     * loss when the ratio is exact. */
                    uint32_t stride = val & 0x0FFFFFFF;
                    if (stride % input_w == 0) {
                        uint32_t h_factor = stride / input_w;
                        uint32_t new_stride = input_w_pad * h_factor;
                        new_val = (val & 0xF0000000) |
                                  (new_stride & 0x0FFFFFFF);
                        do_patch = (new_val != val);
                    }
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
    }  /* end task loop */
    }  /* end task-source loop */

    orknn_log(1, "run: patched %u entries across %u tasks and %d sources",
              patched, m->task_count, n_task_srcs);

    /* Dump patched regcmd for debugging */
    const char *dump_path = getenv("ORKNN_DUMP_REGCMD");
    if (dump_path) {
        const struct task_entry *dbg_tasks =
            (const struct task_entry *)ctx->task_bo.map;
        FILE *df = fopen(dump_path, "w");
        if (df) {
            for (uint32_t t = 0; t < m->task_count && t < 10; t++) {
                uint32_t amt = dbg_tasks[t].f[6];
                uint64_t addr = dbg_tasks[t].regcmd_addr;
                uint32_t bo_off2 = (uint32_t)(addr - ctx->weight_bo.dma_addr);
                uint64_t *ent = (uint64_t *)((uint8_t *)ctx->weight_bo.map + bo_off2);
                fprintf(df, "=== TASK[%u] addr=0x%lx bo_off=%u amt=%u em=0x%x ===\n",
                        t, (unsigned long)addr, bo_off2, amt, dbg_tasks[t].f[2]);
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

    /* Dump the entire weight BO after patching, for byte-exact diff
     * against the vendor dump at /tmp/rknn_dump/sub1_bo_001_*.bin.
     * Used by tests/diff_regcmd.py to surface per-register template-vs-oracle
     * discrepancies during template-patch development. A companion ".meta"
     * file holds BO DMA bases so the diff tool can rebase DMA-class register
     * values without a manual --template-wt-base flag. */
    const char *bo1_dump = getenv("ORKNN_DUMP_BO1");
    if (bo1_dump) {
        FILE *bf = fopen(bo1_dump, "wb");
        if (bf) {
            fwrite(ctx->weight_bo.map, 1, ctx->weight_bo.size, bf);
            fclose(bf);
            orknn_log(1, "run: dumped weight BO (%u bytes) to %s",
                      ctx->weight_bo.size, bo1_dump);
        } else {
            orknn_log(0, "run: failed to open %s for writing", bo1_dump);
        }
        char meta_path[512];
        snprintf(meta_path, sizeof(meta_path), "%s.meta", bo1_dump);
        FILE *mf = fopen(meta_path, "w");
        if (mf) {
            fprintf(mf, "weight_bo_dma=0x%lx\n",
                    (unsigned long)ctx->weight_bo.dma_addr);
            fprintf(mf, "weight_bo_size=%u\n", ctx->weight_bo.size);
            fprintf(mf, "task_bo_dma=0x%lx\n",
                    (unsigned long)ctx->task_bo.dma_addr);
            fprintf(mf, "activation_bo_dma=0x%lx\n",
                    (unsigned long)ctx->activation_bo.dma_addr);
            fprintf(mf, "activation_bo_size=%u\n", ctx->activation_bo.size);
            for (uint32_t i = 0; i < m->n_inputs; i++) {
                fprintf(mf, "input_bo[%u]_dma=0x%lx size=%u\n", i,
                        (unsigned long)ctx->input_bos[i].dma_addr,
                        ctx->input_bos[i].size);
            }
            for (uint32_t i = 0; i < m->n_outputs; i++) {
                fprintf(mf, "output_bo[%u]_dma=0x%lx size=%u\n", i,
                        (unsigned long)ctx->output_bos[i].dma_addr,
                        ctx->output_bos[i].size);
            }
            fclose(mf);
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
    free(patched_offsets);
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

    uint32_t max_segs = m->segment_count;
    if (getenv("ORKNN_MAX_SEGS")) {
        uint32_t v = (uint32_t)strtoul(getenv("ORKNN_MAX_SEGS"), NULL, 10);
        if (v < max_segs) max_segs = v;
    }
    /* Pick cycle snapshot by run count — cycle[run_count] capped at last
     * available cycle. This matches bench_rknn's warmup + iter 1 pattern:
     * call 0 = warmup, call 1 = iter 1 (saved), etc. */
    for (uint32_t i = 0; i < max_segs; i++) {
        struct orknn_segment *seg = &m->segments[i];
        uint32_t cycle_idx = ctx->run_count;
        if (cycle_idx >= seg->n_cycles)
            cycle_idx = seg->n_cycles > 0 ? seg->n_cycles - 1 : 0;
        uint8_t *seg_task_data = seg->n_cycles > 0
            ? seg->task_bo_data[cycle_idx] : NULL;
        uint32_t seg_task_size = seg->n_cycles > 0
            ? seg->task_bo_size[cycle_idx] : 0;

        if (seg_task_data && seg_task_size <= ctx->task_bo.size) {
            memcpy(ctx->task_bo.map, seg_task_data, seg_task_size);
            /* Rebase regcmd_addr field for every task (offset 32 in each 40-byte task) */
            struct {
                uint32_t f[8];
                uint64_t regcmd_addr;
            } __attribute__((packed)) *ts = ctx->task_bo.map;
            uint32_t n_tasks = seg_task_size / 40;
            /* Proxy weight BO base from dump's submit_1.txt */
            uint64_t proxy_wt_base = 0;
            FILE *mf = fopen("/tmp/rknn_dump/submit_1.txt", "r");
            if (mf) {
                char line[256];
                while (fgets(line, sizeof(line), mf)) {
                    if (strncmp(line, "bo[1]", 5) == 0) {
                        char *dp = strstr(line, "dma=0x");
                        if (dp) proxy_wt_base = strtoull(dp + 6, NULL, 16);
                        break;
                    }
                }
                fclose(mf);
            }
            if (proxy_wt_base) {
                uint64_t our_wt_base = ctx->weight_bo.dma_addr;
                for (uint32_t t = 0; t < n_tasks; t++) {
                    if (ts[t].regcmd_addr >= proxy_wt_base &&
                        ts[t].regcmd_addr < proxy_wt_base + ctx->weight_bo.size) {
                        ts[t].regcmd_addr =
                            our_wt_base + (ts[t].regcmd_addr - proxy_wt_base);
                    }
                }
            }
            orknn_bo_sync_to_device(ctx->npu_fd, &ctx->task_bo);
        }

        int ret = orknn_npu_submit(ctx->npu_fd, &ctx->task_bo, seg);
        if (ret) {
            orknn_log(0, "run: segment %u submit failed", i);
            return RKNN_ERR_FAIL;
        }
    }

    ctx->run_count++;
    return RKNN_SUCC;
}
