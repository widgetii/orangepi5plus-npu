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

/* Derive act_output_offsets / act_output_valid / act_output_layout
 * from FB metadata alone — no /tmp/rknn_dump reads, no sig-search.
 *
 * For each subgraph output tensor X:
 *   1. Walk back through "logical no-op" ops (Reshape/Flatten/Squeeze/
 *      Unsqueeze/Identity — ops whose op_idx never appears in the task
 *      BO, meaning the NPU runtime doesn't execute them and the data
 *      stays wherever the predecessor wrote it).
 *   2. Stop at the first tensor Y whose producer op DOES emit tasks.
 *   3. If Y is itself a subgraph output (Transpose / final conv that
 *      writes directly to the sg output), the template-patch path
 *      routes DST via dst_is_sg_output into `output_bos[k]` at offset
 *      0. Record `valid = 2+k, offset = 0`.
 *   4. Else (Reshape / exSoftmax13 / plain Conv followed by Reshape),
 *      the template-patch path falls through to `act_base + Y.f13 +
 *      val`. Record `valid = 1, offset = Y.f13`.
 *
 * Layout: HBWCH16 (cl=3) for output-BO outputs whose tensor is 4D
 * with non-trivial spatial dims; linear (cl=2) otherwise. src_order
 * follows the output tensor's `fmt` (NHWC/NCHW).
 */
static void discover_outputs_from_fb(struct orknn_context *ctx)
{
    struct orknn_model *m = &ctx->model;
    if (!m->ops || !m->tensor_offsets || !m->sg_output_tensor_idx)
        return;
    if (m->n_outputs == 0 || m->n_outputs > 16) return;

    /* Build a "tensor -> producer op_idx" map. */
    uint32_t *t2op = calloc(m->tensor_count, sizeof(uint32_t));
    if (!t2op) return;
    for (uint32_t i = 0; i < m->tensor_count; i++) t2op[i] = UINT32_MAX;
    for (uint32_t i = 0; i < m->op_count; i++) {
        for (uint32_t k = 0; k < m->ops[i].output_count; k++) {
            uint32_t tidx = m->ops[i].output_tensors[k];
            if (tidx < m->tensor_count && t2op[tidx] == UINT32_MAX)
                t2op[tidx] = i;
        }
    }

    /* "Logical no-op" ops — ops whose output is the same bytes as
     * their input[0], just with a different shape/view. These don't
     * emit NPU tasks, so we walk back through them to find the tensor
     * that actually holds the data at runtime. The list is based on
     * ONNX/TFLite op types we've seen in the five runtime models.
     * Relying on op-type strings is more reliable than counting tasks
     * per op, because the primary task_bo only holds the first
     * segment's tasks and multi-segment models (YOLOv8 head) would
     * otherwise appear to have no tasks for their late ops. */
    #define IS_LOGICAL_NOOP(typ) ( \
        strcmp((typ), "Reshape") == 0 || \
        strcmp((typ), "Flatten") == 0 || \
        strcmp((typ), "Squeeze") == 0 || \
        strcmp((typ), "Unsqueeze") == 0 || \
        strcmp((typ), "Identity") == 0 || \
        strcmp((typ), "OutputOperator") == 0)

    for (uint32_t oi = 0; oi < m->n_outputs; oi++) {
        uint32_t cur = m->sg_output_tensor_idx[oi];
        int hops = 0;
        while (hops++ < 8 && cur < m->tensor_count) {
            uint32_t op_idx = t2op[cur];
            if (op_idx == UINT32_MAX) break;
            const struct orknn_op_info *oi2 = &m->ops[op_idx];
            if (!IS_LOGICAL_NOOP(oi2->type)) break;
            if (oi2->input_count == 0) break;
            cur = oi2->input_tensors[0];
        }
        if (cur >= m->tensor_count) continue;

        int cur_is_sg_output = m->tensor_is_sg_output &&
                               m->tensor_is_sg_output[cur];

        /* Determine layout from the sg output tensor's dimensions. */
        uint8_t layout = 2; /* linear default */
        uint8_t src_order = 0; /* NHWC — the native byte order for
                                * everything the NPU writes, regardless
                                * of the user-visible tensor fmt. */
        if (m->outputs[oi].n_dims == 4 && m->outputs[oi].dims[1] > 1 &&
            m->outputs[oi].dims[2] > 1) {
            /* 4D with spatial dims: YOLO/detection outputs use HBWCH16
             * when they land in a dedicated output BO. */
            layout = cur_is_sg_output ? 3 : 0;
        }

        if (cur_is_sg_output) {
            /* Final write goes to output_bos[oi] at offset 0. */
            ctx->act_output_valid[oi] = (uint8_t)(2 + oi);
            ctx->act_output_offsets[oi] = 0;
        } else {
            /* Final write stayed in the activation BO at cur's f13. */
            ctx->act_output_valid[oi] = 1;
            ctx->act_output_offsets[oi] = m->tensor_offsets[cur];
        }
        ctx->act_output_layout[oi] = layout;
        ctx->act_output_src_order[oi] = src_order;

        orknn_log(1, "run: output[%u] FB-derived: %s @ 0x%x (tensor %u, "
                     "layout=%u, src=%s)",
                  oi,
                  cur_is_sg_output ? "output BO" : "ACT BO",
                  ctx->act_output_offsets[oi], cur, layout,
                  src_order ? "NCHW" : "NHWC");
    }

    free(t2op);
}

static void patch_regcmd_addresses(struct orknn_context *ctx)
{

    struct orknn_model *m = &ctx->model;
    uint32_t rc_off = (uint32_t)(m->regcmd_data - m->wt_data);

    uint32_t wt_base = (uint32_t)ctx->weight_bo.dma_addr;
    uint32_t act_base = (uint32_t)ctx->activation_bo.dma_addr;
    uint32_t in_base = ctx->input_bos ? (uint32_t)ctx->input_bos[0].dma_addr : 0;
    uint32_t out_base = ctx->output_bos ? (uint32_t)ctx->output_bos[0].dma_addr : 0;

    /* Dev: ORKNN_DUMP_BO1_PRE dumps the pre-patch weight BO. Used with
     * tests/diff_regcmd.py to compare vendor's post-init oracle against
     * the raw template we loaded from the .rknn file, helping identify
     * which registers vendor fills in at runtime. */
    const char *pre_dump = getenv("ORKNN_DUMP_BO1_PRE");
    if (pre_dump) {
        FILE *bf = fopen(pre_dump, "wb");
        if (bf) {
            fwrite(ctx->weight_bo.map, 1, ctx->weight_bo.size, bf);
            fclose(bf);
            orknn_log(0, "run: dumped pre-patch weight BO to %s", pre_dump);
        }
    }

    /* Dev: ORKNN_ORACLE_PATCH=/path/to/vendor_bo1.bin ORKNN_ORACLE_WT_BASE=0x...
     *
     * Scouting fallback for models whose patch rules are incomplete.
     * Copies vendor's post-init weight BO verbatim, then rebases every
     * known DMA-bearing register from the vendor base to openrknn's base.
     * This proves whether everything else in the submit/output path
     * works for a new model class, even when patch_regcmd_addresses
     * itself can't produce a byte-exact result. */
    const char *oracle_path = getenv("ORKNN_ORACLE_PATCH");
    const char *oracle_wt_env = getenv("ORKNN_ORACLE_WT_BASE");
    if (oracle_path && oracle_wt_env) {
        uint32_t oracle_wt = (uint32_t)strtoul(oracle_wt_env, NULL, 0);
        /* Also accept per-BO rebase envs so user can adapt bases. */
        const char *oracle_act_env = getenv("ORKNN_ORACLE_ACT_BASE");
        const char *oracle_in_env  = getenv("ORKNN_ORACLE_IN_BASE");
        const char *oracle_out_env = getenv("ORKNN_ORACLE_OUT_BASE");
        uint32_t oracle_act = oracle_act_env ? (uint32_t)strtoul(oracle_act_env, NULL, 0) : 0;
        uint32_t oracle_in  = oracle_in_env  ? (uint32_t)strtoul(oracle_in_env,  NULL, 0) : 0;
        uint32_t oracle_out = oracle_out_env ? (uint32_t)strtoul(oracle_out_env, NULL, 0) : 0;

        FILE *of = fopen(oracle_path, "rb");
        if (of) {
            /* fread directly into an mmap'd DRM BO can behave oddly on
             * some kernels (write-combining / page cache interactions).
             * Read into a malloc buffer first, then memcpy into the BO
             * map — same CPU write pattern openrknn uses everywhere else. */
            uint8_t *stage = malloc(ctx->weight_bo.size);
            size_t n = 0;
            if (stage) {
                n = fread(stage, 1, ctx->weight_bo.size, of);
                memcpy(ctx->weight_bo.map, stage, n);
                free(stage);
            }
            fclose(of);
            orknn_log(0, "run: ORACLE_PATCH loaded %zu bytes from %s "
                      "(oracle_wt=0x%x act=0x%x in=0x%x out=0x%x)",
                      n, oracle_path, oracle_wt, oracle_act, oracle_in, oracle_out);

            /* Walk all tasks and rebase the known DMA registers.
             * Multi-core regions share regcmd sections with the
             * single-core region — if we re-enter the same section
             * we'd read already-rebased values as if they were
             * vendor-bases and apply the offset transform a second
             * time. Dedup via a sorted array of seen offsets. */
            struct task_entry_oracle { uint32_t f[8]; uint64_t regcmd_addr; }
                __attribute__((packed));
            const struct task_entry_oracle *tk =
                (const struct task_entry_oracle *)ctx->task_bo.map;
            uint32_t *rebased_offs = calloc(m->task_count ? m->task_count : 1,
                                            sizeof(uint32_t));
            int n_rebased = 0;
            for (uint32_t t = 0; t < m->task_count; t++) {
                uint32_t amt = tk[t].f[6];
                uint64_t addr = tk[t].regcmd_addr;
                uint32_t bo_off = (uint32_t)(addr - wt_base);
                if (bo_off >= ctx->weight_bo.size) continue;
                /* Skip already-rebased sections. */
                int seen = 0;
                for (int k = 0; k < n_rebased; k++) {
                    if (rebased_offs[k] == bo_off) { seen = 1; break; }
                }
                if (seen) continue;
                rebased_offs[n_rebased++] = bo_off;
                uint64_t *entries = (uint64_t *)
                    ((uint8_t *)ctx->weight_bo.map + bo_off);
                for (uint32_t e = 0; e < amt + 4; e++) {
                    uint16_t reg = entries[e] & 0xFFFF;
                    uint32_t v = (entries[e] >> 16) & 0xFFFFFFFF;
                    uint32_t nv = v;
                    int is_dma = 0;
                    /* Common range-based classifier: try each BO in
                     * turn and rebase if v falls inside. Every check is
                     * BOUNDED on BOTH sides — an unbounded `>=` was
                     * the Phase-0E bug: for value 0xfff92580 (which
                     * sits in the wt range) the old 0x5018 handler
                     * fell through to `v >= oracle_act` and produced a
                     * garbage address because act's upper bound was
                     * missing.
                     *
                     * For the general wt-rel set we check wt first;
                     * for 0x5018/0x701c we check in first (input-
                     * consuming REFORMAT source). Order only matters
                     * when two ranges could logically hold the same
                     * value (they don't on our test models). */
                    uint32_t act_size = ctx->activation_bo.size;
                    uint32_t wt_size  = ctx->weight_bo.size;
                    uint32_t in_size  = ctx->input_bos ?
                                        ctx->input_bos[0].size : 0;
                    uint32_t out_size = ctx->output_bos ?
                                        ctx->output_bos[0].size : 0;
                    int reg_is_wt_first = !(reg == 0x5018 || reg == 0x701c);
                    int try_wt_before_act = reg_is_wt_first;
                    switch (reg) {
                    case 0x1070: case 0x1110: case 0x4020:
                    case 0x5020: case 0x502c: case 0x5038:
                    case 0x6070: case 0x0010:
                    case 0x5018: case 0x701c:
                        if (try_wt_before_act &&
                            v >= oracle_wt && v < oracle_wt + wt_size) {
                            nv = wt_base + (v - oracle_wt);
                            is_dma = 1;
                            break;
                        }
                        if (oracle_act && v >= oracle_act &&
                            v < oracle_act + act_size) {
                            nv = act_base + (v - oracle_act);
                            is_dma = 1;
                            break;
                        }
                        if (!try_wt_before_act &&
                            v >= oracle_wt && v < oracle_wt + wt_size) {
                            nv = wt_base + (v - oracle_wt);
                            is_dma = 1;
                            break;
                        }
                        if (oracle_in && v >= oracle_in &&
                            v < oracle_in + in_size) {
                            nv = in_base + (v - oracle_in);
                            is_dma = 1;
                            break;
                        }
                        if (oracle_out && v >= oracle_out &&
                            v < oracle_out + out_size) {
                            nv = out_base + (v - oracle_out);
                            is_dma = 1;
                            break;
                        }
                        break;
                    }
                    if (is_dma) {
                        /* Preserve bits [48..63] (per-entry metadata)
                         * and reg in bits [0..15]; overwrite only the
                         * 32-bit val payload. Matches the format used
                         * by the regular patch loop at the bottom of
                         * this function. Without this, the NPU PC
                         * engine sees garbage per-entry control bits
                         * and hangs on every submit regardless of
                         * regcmd content. */
                        entries[e] = (entries[e] & 0xFFFF000000000000ULL) |
                                     ((uint64_t)nv << 16) |
                                     (entries[e] & 0xFFFF);
                    }
                }
            }
            orknn_log(0, "run: ORACLE_PATCH rebased DMA registers across %u tasks "
                      "(%d unique regcmd sections)",
                      m->task_count, n_rebased);
            free(rebased_offs);
            /* Debug dump (same paths as the normal path's ORKNN_DUMP_BO1). */
            const char *oracle_bo1_dump = getenv("ORKNN_DUMP_BO1");
            if (oracle_bo1_dump) {
                FILE *bf = fopen(oracle_bo1_dump, "wb");
                if (bf) {
                    fwrite(ctx->weight_bo.map, 1, ctx->weight_bo.size, bf);
                    fclose(bf);
                    orknn_log(0, "run: (oracle) dumped weight BO to %s",
                              oracle_bo1_dump);
                }
            }
            /* Match the end-of-normal-path sync: the regular
             * patch_regcmd_addresses runs `orknn_bo_sync_to_device` on
             * the weight BO just before returning. We just rewrote the
             * BO bytes, so without this sync the NPU will read stale
             * cached data and the first submit hangs. Skipping this
             * was the Phase-0 bug blocking oracle-patch on SmolVLM. */
            orknn_bo_sync_to_device(ctx->npu_fd, &ctx->weight_bo);
            return;
        } else {
            orknn_log(0, "run: ORACLE_PATCH failed to open %s", oracle_path);
        }
    }

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
                /* Gap between closer and rc_off → closer is PC2.
                 * NOTE: this is WRONG for SmolVLM l0_mlp (oracle shows
                 * the assignment should be swapped). But changing it
                 * breaks DeepLabv3 which relies on this convention.
                 * The 20 resulting pc2/pc3 diffs on l0_mlp affect
                 * per-channel correction (em=0x60) quality but don't
                 * crash the NPU. Tracked as a follow-up. */
                pc2_off = closer_off;
                pc3_off = farther_off;
            }
        }
    }
    orknn_log(1, "run: PC LUT blobs: pc2=0x%x pc3=0x%x rc_off=0x%x",
              pc2_off, pc3_off, rc_off);
    /* Dev: ORKNN_DEBUG_BLOBS dumps the full weight-BO blob layout
     * (offset/size/type for every scan_blob_offsets() entry). Useful for
     * matching per-op LUT blobs against op indices when extending the
     * template-patch rules for a new model. */
    if (getenv("ORKNN_DEBUG_BLOBS")) {
        for (int _i = 0; _i < n_blobs; _i++) {
            orknn_log(0, "blob[%3d] off=0x%08x size=%8u type=%u",
                      _i, blobs[_i].offset, blobs[_i].size, blobs[_i].type);
        }
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

    /* Phase 1: build per-op anonymous-blob assignment table for em=0x0d
     * non-softmax tasks. "Anonymous" blobs are entries in scan_blob_offsets
     * that don't appear in wt_blob_offsets (no tensor references them via
     * f[18]). Transpose uses 8192-byte anonymous blobs; exNorm uses 1536B.
     *
     * Assignment order: for each em=0x0d-emitting op in execution order,
     * pop the next anonymous blob of matching size. From oracle analysis
     * (SmolVLM l0_mlp), Transpose ops consume 8192B blobs in reverse
     * scan order — the implementation uses that heuristic. */
    #define MAX_ANON_BLOBS 32
    struct { uint32_t offset; uint32_t size; } anon_blobs[MAX_ANON_BLOBS];
    int n_anon = 0;
    /* Collect "hidden tensor" blob offsets: tensors whose tensor_weight_blob
     * points to a valid blob, but the tensor is NOT in any op's input or
     * output tensor lists. These are per-op LUT blobs (Transpose permutation
     * tables, exNorm auxiliary blobs) that the compiler generates as hidden
     * tensors. They're in the FB tensor table but not in any operator's
     * declared inputs/outputs.
     *
     * Sort ascending by offset so the assignment heuristic (reverse order →
     * Transpose ops in forward execution order) works. */
    {
        /* Build bitmap of tensors referenced by ops */
        uint8_t *in_op = calloc(m->tensor_count + 1, 1);
        if (in_op && m->ops) {
            for (uint32_t oi = 0; oi < m->op_count; oi++) {
                for (uint32_t j = 0; j < m->ops[oi].input_count && j < 8; j++) {
                    uint32_t ti = m->ops[oi].input_tensors[j];
                    if (ti < m->tensor_count) in_op[ti] = 1;
                }
                for (uint32_t j = 0; j < m->ops[oi].output_count && j < 4; j++) {
                    uint32_t ti = m->ops[oi].output_tensors[j];
                    if (ti < m->tensor_count) in_op[ti] = 1;
                }
            }
        }
        if (in_op && m->tensor_weight_blob && m->wt_blob_offsets) {
            for (uint32_t ti = 0; ti < m->tensor_count && n_anon < MAX_ANON_BLOBS; ti++) {
                if (in_op[ti]) continue;
                uint32_t bi = m->tensor_weight_blob[ti];
                if (bi >= m->wt_blob_count) continue;
                uint32_t off = m->wt_blob_offsets[bi];
                if (off >= rc_off) continue; /* skip regcmd + task BO + post-regcmd */
                /* Find size from scan */
                uint32_t bsize = 0;
                for (int s = 0; s < n_blobs; s++) {
                    if (blobs[s].offset == off) { bsize = blobs[s].size; break; }
                }
                if (bsize < 1536) continue;
                anon_blobs[n_anon].offset = off;
                anon_blobs[n_anon].size = bsize;
                n_anon++;
            }
        }
        free(in_op);
        /* Sort ascending by offset */
        for (int a = 0; a < n_anon - 1; a++)
            for (int b = a + 1; b < n_anon; b++)
                if (anon_blobs[a].offset > anon_blobs[b].offset) {
                    uint32_t to = anon_blobs[a].offset, ts = anon_blobs[a].size;
                    anon_blobs[a] = anon_blobs[b];
                    anon_blobs[b].offset = to; anon_blobs[b].size = ts;
                }
    }
    /* Build per-op em=0x0d blob assignment by walking ops in execution
     * order and consuming anonymous blobs of matching size.
     * For each op that emits em=0x0d tasks (identified from the task BO),
     * assign the next available anonymous blob.
     * Heuristic: Transpose ops use 8192B blobs in REVERSE anon order. */
    #define MAX_EM0D_OPS 16
    struct { uint32_t op_idx; uint32_t blob_off; } em0d_blob_assign[MAX_EM0D_OPS];
    int n_em0d_assign = 0;
    if (m->ops && n_anon > 0) {
        /* Find which ops emit em=0x0d tasks */
        uint32_t em0d_ops[MAX_EM0D_OPS];
        int n_em0d_ops = 0;
        for (uint32_t t = 0; t < m->task_count && n_em0d_ops < MAX_EM0D_OPS; t++) {
            uint32_t *tf = (uint32_t *)((uint8_t *)ctx->task_bo.map + t * 40);
            if (tf[2] != 0x0d) continue;
            uint32_t op = tf[1];
            int seen = 0;
            for (int k = 0; k < n_em0d_ops; k++)
                if (em0d_ops[k] == op) { seen = 1; break; }
            if (!seen) em0d_ops[n_em0d_ops++] = op;
        }

        /* Assign anonymous blobs to em=0x0d ops.
         *
         * Heuristic from SmolVLM l0_mlp oracle analysis:
         * 8192-byte hidden-tensor blobs are sorted ascending by offset.
         * The FIRST such blob goes to the LAST Transpose op in execution
         * order; the SECOND blob to the second-to-last, etc. This is the
         * same "reverse" convention used by the Conv weight-pair assignment
         * at the top of this function.
         *
         * Collect Transpose ops (only), then zip with 8k anon blobs. */
        int n_transpose = 0;
        uint32_t transpose_ops[MAX_EM0D_OPS];
        for (int k = 0; k < n_em0d_ops; k++) {
            uint32_t oi = em0d_ops[k];
            if (oi < m->op_count && strstr(m->ops[oi].type, "Transpose"))
                transpose_ops[n_transpose++] = oi;
        }
        /* Collect 8192-byte anonymous blob offsets (already sorted asc) */
        uint32_t anon_8k[MAX_ANON_BLOBS];
        int n_anon_8k = 0;
        for (int a = 0; a < n_anon; a++)
            if (anon_blobs[a].size == 8192 && n_anon_8k < MAX_ANON_BLOBS)
                anon_8k[n_anon_8k++] = anon_blobs[a].offset;

        /* Assign: anon_8k[i] → transpose_ops[n_transpose - 1 - i] */
        for (int i = 0; i < n_transpose && i < n_anon_8k &&
             n_em0d_assign < MAX_EM0D_OPS; i++) {
            uint32_t oi = transpose_ops[n_transpose - 1 - i];
            em0d_blob_assign[n_em0d_assign].op_idx = oi;
            em0d_blob_assign[n_em0d_assign].blob_off = anon_8k[i];
            orknn_log(1, "run: em0d blob assign: op=%u (%s) -> wt+0x%x (anon 8k[%d])",
                      oi, m->ops[oi].type, anon_8k[i], i);
            n_em0d_assign++;
        }
    }

    orknn_log(1, "run: patching: wt=0x%x act=0x%x in=0x%x out=0x%x "
              "rc_off=0x%x act_dst=0x%x n_ops=%d n_pairs=%d anon=%d em0d_assign=%d",
              wt_base, act_base, in_base, out_base,
              rc_off, act_dst_off, n_ops, n_pairs, n_anon, n_em0d_assign);

    /* Dev: dump per-op input_tensors → wt_blob_offsets for transformer
     * patch-rule development (Phase 1 of #80). */
    if (getenv("ORKNN_DEBUG_OP_BLOBS") && m->ops && m->tensor_weight_blob &&
        m->wt_blob_offsets) {
        for (uint32_t oi = 0; oi < m->op_count; oi++) {
            const struct orknn_op_info *op = &m->ops[oi];
            orknn_log(0, "op[%u] type=%s inputs=%u",
                      oi, op->type, op->input_count);
            for (uint32_t j = 0; j < op->input_count && j < 8; j++) {
                uint32_t tidx = op->input_tensors[j];
                uint32_t blob_idx = (tidx < m->tensor_count) ?
                    m->tensor_weight_blob[tidx] : UINT32_MAX;
                uint32_t bo_off = (blob_idx < m->wt_blob_count) ?
                    m->wt_blob_offsets[blob_idx] : UINT32_MAX;
                uint32_t act_off = (tidx < m->tensor_count) ?
                    m->tensor_offsets[tidx] : UINT32_MAX;
                orknn_log(0, "  in[%u] tidx=%u blob=%u "
                          "wt_off=0x%x act_off=0x%x",
                          j, tidx, blob_idx, bo_off, act_off);
            }
        }
    }

    uint32_t patched = 0;

    /* Track which regcmd offsets we've already patched to avoid
     * double-patching shared regcmd sections (multi-core tasks share regcmd).
     * Upper bound on unique sections is m->task_count (at most one distinct
     * regcmd per task). ViT shards hit ~17k tasks; CNN models ~hundreds. */
    uint32_t max_patched_offsets = m->task_count ? m->task_count : 1;
    uint32_t *patched_offsets = calloc(max_patched_offsets, sizeof(uint32_t));
    int n_patched_offsets = 0;

    /* Task-BO source for patching. Previously we also fed per-cycle task
     * BO snapshots from /tmp/rknn_dump back through this loop; those are
     * gone (task 9.4) — the only cycle-to-cycle delta the vendor's
     * snapshots carried was the kernel-owned int_status byte, which the
     * NPU driver rewrites at completion time regardless. The primary
     * task BO at ctx->task_bo.map already has every cross-core replica
     * patched since extract_npu_data copies the raw task blob intact. */
    struct task_entry { uint32_t f[8]; uint64_t regcmd_addr; }
        __attribute__((packed));
    const int n_task_srcs = 1;
    for (int src = 0; src < n_task_srcs; src++) {
    const struct task_entry *tasks = (const struct task_entry *)ctx->task_bo.map;
    uint32_t src_count = m->task_count;
    uint64_t src_wt_base = ctx->weight_bo.dma_addr;
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
    /* Block counter for em=0x0d sequences: increments when a non-em=0x0d
     * task interrupts a consecutive em=0x0d run within the same op. Used
     * by exNorm's auxiliary blob assignment (Phase 1 of #80). */
    uint32_t em0d_block_idx = 0;
    int em0d_in_run = 0; /* true while consecutive em=0x0d tasks of same op */
    /* Hoist debug env lookups out of the per-register hot path. */
    int debug_patch = getenv("ORKNN_DEBUG_PATCH") ? 1 : 0;
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
        if ((uint32_t)n_patched_offsets < max_patched_offsets) {
            patched_offsets[n_patched_offsets++] = bo_off;
        } else {
            /* Overflow: we can't dedupe anymore. Any previously-patched
             * section that reappears here will be re-patched and produce
             * garbage. Log loudly so this doesn't silently corrupt. */
            static int warned = 0;
            if (!warned) {
                orknn_log(0, "run: patched_offsets overflow (>%u unique "
                             "regcmd sections) — results may be wrong",
                          max_patched_offsets);
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
            if (!em0d_in_run) { em0d_block_idx++; em0d_in_run = 1; }
        } else if (is_em0d) {
            em0d_sub_idx = 0;
            em0d_block_idx = 0;
            em0d_in_run = 1;
            prev_op_for_em0d = op;
        } else if (op != prev_op_for_em0d) {
            em0d_sub_idx = 0;
            em0d_block_idx = 0;
            em0d_in_run = 0;
            prev_op_for_em0d = UINT32_MAX;
        } else {
            /* Same op, but not em0d → gap in the em0d run */
            em0d_in_run = 0;
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

            /* Dev: ORKNN_DEBUG_PATCH logs raw pre-patch vals for the
             * main DMA-bearing registers on early tasks. Used when adding
             * patch rules for new op types — pairs with the vendor oracle
             * captured by librocketnpu/tests/intercept_swap.so DUMP_ALL_BOS. */
            if (__builtin_expect(debug_patch != 0, 0) &&
                (reg == 0x1070 || reg == 0x1110 || reg == 0x5018 ||
                 reg == 0x4020) && t < 40) {
                orknn_log(0, "dbg: t=%u op=%u em=0x%x reg=0x%04x raw_val=0x%08x",
                          t, op, enable_mask, reg, val);
            }

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
                } else if (enable_mask == 0x0d) {
                    /* Non-softmax general compute em=0x0d task (observed
                     * on SmolVLM l0_mlp ops 3/4/7 — Transpose, exNorm,
                     * 2nd Transpose). Pattern: src_tensor_off + per-task
                     * stride (val walks through the input activation in
                     * uniform chunks, e.g. 0/0x3000/0x6000/... for
                     * Transpose's 12288-byte tiles). Safe regardless of
                     * op type because src_tensor_off is the primary
                     * input's activation BO offset from the FB memory
                     * plan. */
                    new_val = act_base + src_tensor_off + val;
                    do_patch = 1;
                } else if (val != 0) {
                    new_val = act_base + val;
                    do_patch = 1;
                }
                break;

            case 0x1110: /* WT_BASE */
                /* For em=0x0d non-softmax tasks, use the em0d-specific
                 * blob lookup (Phase 1) instead of the op's default
                 * weight blob. The default (have_op_wt → op_wt_bo_off)
                 * points to input_tensors[1]'s blob which is the CONV
                 * weight for multi-task ops like exNorm — wrong for the
                 * em=0x0d continuation tasks which need auxiliary LUTs
                 * from input_tensors[4+] or anonymous hidden tensors. */
                if (enable_mask == 0x0d && val == 0 &&
                    !(m->ops && op < m->op_count &&
                      strncmp(m->ops[op].type, "exSoftmax", 9) == 0)) {
                    /* em=0x0d non-softmax: look up the anonymous-blob
                     * assignment table built at the top of this function.
                     * This handles Transpose / exNorm / etc. ops whose
                     * LUT blob is NOT referenced by any tensor's f[18]
                     * but IS present in the scan_blob_offsets table as
                     * a type=6 anonymous entry. Phase 1 of #80. */
                    uint32_t em0d_off = 0;
                    for (int k = 0; k < n_em0d_assign; k++) {
                        if (em0d_blob_assign[k].op_idx == op) {
                            em0d_off = em0d_blob_assign[k].blob_off;
                            break;
                        }
                    }
                    /* Also check exNorm: input_tensors[4+] might have
                     * weight blob refs for the alternating sub-tasks.
                     * Use em0d_sub_idx to pick from input[4], [5], ... */
                    if (em0d_off == 0 && m->ops && op < m->op_count) {
                        const struct orknn_op_info *oi = &m->ops[op];
                        uint32_t pick_start = 4; /* skip data/wt/bs/scratch */
                        /* Block-based pick from the END of input_tensors:
                         * exNorm emits em=0x0d tasks in BLOCKS separated
                         * by em=0x1d/0x18 tasks. Each block uses ONE blob
                         * from input_tensors[input_count-1-block_idx].
                         * em0d_block_idx is tracked at task-level above. */
                        uint32_t n_aux = oi->input_count > pick_start ?
                                         oi->input_count - pick_start : 1;
                        uint32_t pick = pick_start +
                                        (em0d_block_idx % n_aux);
                        if (pick < oi->input_count && m->tensor_weight_blob &&
                            m->wt_blob_offsets) {
                            uint32_t tidx = oi->input_tensors[pick];
                            if (tidx < m->tensor_count) {
                                uint32_t bi = m->tensor_weight_blob[tidx];
                                if (bi < m->wt_blob_count)
                                    em0d_off = m->wt_blob_offsets[bi];
                            }
                        }
                    }
                    new_val = wt_base + em0d_off;
                    do_patch = 1;
                } else if (have_op_wt) {
                    new_val = wt_base + op_wt_bo_off + val;
                    do_patch = 1;
                } else {
                    new_val = wt_base + (val ? val : task_wt_off);
                    do_patch = 1;
                }
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
                } else if (enable_mask == 0x0d && m->ops &&
                           op < m->op_count &&
                           m->ops[op].input_count > 3 &&
                           m->tensor_offsets) {
                    /* Non-softmax em=0x0d tasks (exNorm, etc.) write to
                     * a scratch tensor at input_tensors[3], not the op's
                     * output tensor. Oracle analysis: exNorm DST =
                     * act + tensor_offsets[input[3]] (e.g. 0x480000).
                     * openrknn's default dst_tensor_off = output[0]
                     * which is wrong for these intermediate tasks. */
                    uint32_t scratch_tidx = m->ops[op].input_tensors[3];
                    uint32_t scratch_off = (scratch_tidx < m->tensor_count) ?
                        m->tensor_offsets[scratch_tidx] : dst_tensor_off;
                    new_val = act_base + scratch_off + val;
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
                } else if (is_reformat && is_input_consuming_task_for_src &&
                           amt < 1000) {
                    /* When the input-consuming op is itself a REFORMAT
                     * (e.g. SmolVLM l0_mlp op 1 Reshape consumes the
                     * subgraph input directly), the RDMA source is the
                     * input BO, not the activation BO. Without this
                     * branch the task reads zero-filled activation and
                     * the NPU job hangs.
                     *
                     * val is the intra-tile byte offset within the
                     * input BO (0 for task 0, stride*k for subsequent
                     * tile tasks — e.g. 0xff000 for the second tile).
                     *
                     * Exclude sentinel tasks (amt >= 1000): YOLOv8's
                     * first op is also the input-consuming one and has
                     * a sentinel REFORMAT whose 0x5018 should point at
                     * `wt_base + rc_off`, not the input BO. The
                     * sentinel branch below handles that case. */
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
            case 0x6070: /* PPU_DST_BASE_ADDR */
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

            case 0x701c: /* PPU_RDMA_SRC_BASE_ADDR */
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

    /* FB-derived output discovery — sole source of truth now for
     * act_output_offsets / act_output_valid / act_output_layout.
     * The sig-search code in copy_proxy_regcmd's remaining body is
     * dead and gets deleted below in the same commit. */
    discover_outputs_from_fb(ctx);

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
    /* ctx->task_bo was populated once by orknn_alloc_model_bos with the
     * patched task data; each segment just replays a {sc_start, sc_count}
     * slice into the NPU. No per-cycle rewrites needed (see task 9.4).
     *
     * #60: if the user set core_mask to 2 or 3 cores and the model
     * has a compiled multi-core submit plan, dispatch through that
     * plan instead of the single-core segments. The multi-core plan
     * references pre-compiled task BO regions at different offsets;
     * the weight BO was already patched in patch_regcmd_addresses,
     * which iterates the full task BO including those regions. */
    uint32_t mask_popcount = __builtin_popcount(ctx->core_mask & 0x7);
    const struct orknn_multicore_submit *mc_plan = NULL;
    uint32_t mc_count = 0;
    if (mask_popcount == 2 && m->submits_2core) {
        mc_plan = m->submits_2core;
        mc_count = m->n_submits_2core;
    } else if (mask_popcount == 3 && m->submits_3core) {
        mc_plan = m->submits_3core;
        mc_count = m->n_submits_3core;
    }

    if (mc_plan && mc_count > 0) {
        orknn_log(2, "run: multi-core submit (mask=0x%x, %u submits)",
                  ctx->core_mask, mc_count);
        for (uint32_t i = 0; i < mc_count; i++) {
            int ret = orknn_npu_submit_multicore(ctx->npu_fd,
                                                 &ctx->task_bo,
                                                 &mc_plan[i]);
            if (ret) {
                orknn_log(0, "run: multi-core segment %u submit failed", i);
                return RKNN_ERR_FAIL;
            }
        }
    } else {
        /* Dev: ORKNN_SKIP_SEGS="1,2" skips the listed segment indices.
         * Used for Phase-0E bisection of the SmolVLM l0_mlp seg-1 hang
         * (can seg 2 run when seg 1 is skipped?). Does nothing unset. */
        const char *skip_env = getenv("ORKNN_SKIP_SEGS");
        for (uint32_t i = 0; i < max_segs; i++) {
            if (skip_env) {
                char buf[16];
                snprintf(buf, sizeof(buf), "%u", i);
                /* crude substring search: "1,2" or "1" */
                const char *p = skip_env;
                int skip = 0;
                while (*p) {
                    uint32_t v = (uint32_t)strtoul(p, (char **)&p, 10);
                    if (v == i) { skip = 1; break; }
                    if (*p == ',') p++;
                    else break;
                }
                if (skip) {
                    orknn_log(0, "run: ORKNN_SKIP_SEGS skipping seg %u", i);
                    continue;
                }
            }
            struct orknn_segment seg_copy = m->segments[i];
            /* Dev: ORKNN_SEG<N>_TRIM_FIRST=K advances seg N's sc_start
             * by K and reduces sc_count/task_number accordingly. Used
             * to test whether skipping the first K tasks of a specific
             * segment clears a hang on that segment. */
            char env_name[32];
            snprintf(env_name, sizeof(env_name), "ORKNN_SEG%u_TRIM_FIRST", i);
            const char *trim_env = getenv(env_name);
            if (trim_env) {
                uint32_t trim = (uint32_t)strtoul(trim_env, NULL, 10);
                if (trim < seg_copy.sc_count) {
                    orknn_log(0, "run: %s=%u advancing seg %u "
                              "sc_start %u→%u count %u→%u",
                              env_name, trim, i,
                              seg_copy.sc_start, seg_copy.sc_start + trim,
                              seg_copy.sc_count, seg_copy.sc_count - trim);
                    seg_copy.sc_start += trim;
                    seg_copy.sc_count -= trim;
                    seg_copy.task_number -= trim;
                }
            }
            int ret = orknn_npu_submit(ctx->npu_fd, &ctx->task_bo, &seg_copy,
                                       ctx->core_mask);
            if (ret) {
                orknn_log(0, "run: segment %u submit failed", i);
                return RKNN_ERR_FAIL;
            }
        }
    }

    ctx->run_count++;
    return RKNN_SUCC;
}
