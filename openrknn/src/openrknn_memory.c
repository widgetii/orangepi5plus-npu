/*
 * openrknn — Memory management: allocate model BOs
 *
 * SPDX-License-Identifier: MIT
 */
#include "openrknn.h"
#include <stdlib.h>
#include <string.h>
#include <dirent.h>

#define ALIGN_UP(x, a) (((x) + (a) - 1) & ~((a) - 1))

/* Allocate all DMA buffer objects needed for model execution:
 * - weight_bo: weight + regcmd data (from parsed model)
 * - task_bo: RKNPU task descriptors
 * - activation_bo: intermediate tensor scratch space
 * - input_bos[]: one per model input
 * - output_bos[]: one per model output
 */
int orknn_alloc_model_bos(struct orknn_context *ctx)
{
    struct orknn_model *m = &ctx->model;
    int fd = ctx->npu_fd;

    /* If extract_npu_data failed (e.g. MobileSAM: task data is split
     * across many small per-op blobs that our parser can't assemble),
     * m->wt_size == 0. Skip weight/task/activation BO allocation entirely
     * and rely on proxy dispatch for run. Only allocate input/output BOs
     * so query+inputs_set+outputs_get can still use our own metadata. */
    int have_npu_data = (m->wt_size > 0);

    if (have_npu_data) {
        /* Weight + regcmd BO */
        uint32_t wt_size = ALIGN_UP(m->wt_size, 4096);
        if (orknn_bo_create(fd, wt_size, &ctx->weight_bo)) {
            orknn_log(0, "memory: failed to allocate weight BO (%u bytes)", wt_size);
            return -1;
        }
        memcpy(ctx->weight_bo.map, m->wt_data, m->wt_size);
        orknn_bo_sync_to_device(fd, &ctx->weight_bo);
        orknn_log(1, "memory: weight BO: %u bytes, dma=0x%lx",
                  wt_size, (unsigned long)ctx->weight_bo.dma_addr);
    } else {
        orknn_log(1, "memory: skipping weight/task/act BO allocation "
                     "(no NPU data — proxy run path)");
    }
    if (!have_npu_data) goto alloc_io_bos;

    /* Task BO — needs KERNEL_MAPPING (0x8) flag so RKNPU driver can read it.
     * Must be large enough to fit the proxy's per-segment task BO snapshots
     * (which may be larger than our own FB-derived task data). */
    uint32_t task_size = ALIGN_UP(m->task_data_size, 4096);
    for (uint32_t s = 0; s < m->segment_count; s++) {
        for (uint32_t c = 0; c < m->segments[s].n_cycles; c++) {
            if (m->segments[s].task_bo_size[c] > 0) {
                uint32_t need = ALIGN_UP(m->segments[s].task_bo_size[c], 4096);
                if (need > task_size) task_size = need;
            }
        }
    }
    if (task_size < 4096) task_size = 4096;
    if (orknn_bo_create_flags(fd, task_size, 0x40b, &ctx->task_bo)) {
        orknn_log(0, "memory: failed to allocate task BO (%u bytes)", task_size);
        return -1;
    }
    orknn_log(1, "memory: task BO: %u bytes, dma=0x%lx",
              task_size, (unsigned long)ctx->task_bo.dma_addr);

    /* Activation BO (intermediate tensors).
     *
     * Preferred: read the vendor's actual size from /tmp/rknn_dump/submit_1.txt
     * bo[2]. This guarantees parity with the vendor allocation and is what
     * the copy_proxy_regcmd path assumes.
     *
     * Fallback (used when /tmp/rknn_dump is absent — the eventual steady
     * state after phase 9): derive from the template regcmd.
     *
     *   fb_act_size = max(
     *       scan * 4 + largest_io * 2,   // empirical safety margin
     *       1 MB floor)
     *
     * where `scan` is the highest byte offset written to any activation-
     * targeting DMA register in the template (0x4020, 0x1070, 0x5018,
     * 0x4110, 0x5038) excluding sentinel values >= 256MB, and `largest_io`
     * is the biggest input/output tensor native_size. The 4x multiplier
     * was derived empirically from the 5 runtime models in ground_truth
     * (mobilenet_v1, resnet50, yolov5, yolov8, deeplabv3): across all of
     * them, `dump / (scan + largest_io)` is at most 2.95, so 4x gives a
     * safety margin while over-allocating by at most ~2x on the larger
     * models. The ~2x waste is acceptable while we don't have per-op FB
     * metadata that would let us compute the exact compiler budget.
     *
     * Note: m->total_internal_size is deliberately NOT used. The blob
     * extraction that populated it is unreliable (see openrknn_model.c).
     */
    uint32_t dump_act_size = 0;
    {
        FILE *sf = fopen("/tmp/rknn_dump/submit_1.txt", "r");
        if (sf) {
            char line[256];
            while (fgets(line, sizeof(line), sf)) {
                if (strstr(line, "bo[2]")) {
                    char *sp = strstr(line, "size=");
                    if (sp) dump_act_size = (uint32_t)strtoul(sp + 5, NULL, 10);
                }
            }
            fclose(sf);
        }
    }

    /* Template regcmd scan: collect the highest offset touched by any
     * activation-targeting DMA register. Plain uint32_t values near
     * UINT32_MAX (e.g. 0xFFFFC000) appear as compiler sentinels or
     * register-field-pack artifacts — hard-ignore anything above
     * ACT_OFF_SANITY_MAX (256MB). Real activation BOs on RK3588 top out
     * around tens of MB for the largest models we see. */
    #define ACT_OFF_SANITY_MAX 0x10000000u
    uint32_t max_act_off = 0;
    uint32_t scan_sentinels = 0;
    if (m->regcmd_data && m->regcmd_size >= 8) {
        const uint64_t *rc = (const uint64_t *)m->regcmd_data;
        uint32_t n_entries = m->regcmd_size / 8;
        for (uint32_t i = 0; i < n_entries; i++) {
            uint16_t reg = rc[i] & 0xFFFF;
            uint32_t val = (rc[i] >> 16) & 0xFFFFFFFF;
            if (val == 0) continue;
            /* Activation-targeting DMA registers:
             *   0x1070 CNA_FEATURE_DATA_ADDR
             *   0x4020 DPU_DST_BASE_ADDR
             *   0x5018 DPU_RDMA_SRC_BASE_ADDR
             *   0x5038 DPU_RDMA_EW_BASE_ADDR
             * Note 0x4110 is DPU_LUT_LE_START (a LUT value), not DMA. */
            if (reg != 0x4020 && reg != 0x1070 && reg != 0x5018 &&
                reg != 0x5038)
                continue;
            if (val >= ACT_OFF_SANITY_MAX) {
                scan_sentinels++;
                continue;
            }
            if (val > max_act_off) max_act_off = val;
        }
    }
    uint32_t largest_io = 0;
    for (uint32_t i = 0; i < m->n_inputs; i++)
        if (m->inputs[i].native_size > largest_io)
            largest_io = m->inputs[i].native_size;
    for (uint32_t i = 0; i < m->n_outputs; i++)
        if (m->outputs[i].native_size > largest_io)
            largest_io = m->outputs[i].native_size;

    /* FB-derived size: scan*4 + largest_io*2. Validated on all 5 runtime
     * models in ground_truth.json; worst-case over-allocation ~2.2x for
     * YOLO/DeepLabv3, but always covers the vendor allocation. Floor at
     * 1MB so tiny test models still get a usable buffer. */
    uint32_t fb_act_size = max_act_off * 4 + largest_io * 2;
    if (fb_act_size < 1048576) fb_act_size = 1048576;

    uint32_t act_size = dump_act_size ? dump_act_size : fb_act_size;
    act_size = ALIGN_UP(act_size, 4096);
    orknn_log(1, "memory: activation size: dump=%u fb=(scan=%u+%u "
                 "sentinels=%u -> %u) -> %u (%s)",
              dump_act_size, max_act_off, largest_io, scan_sentinels,
              fb_act_size, act_size, dump_act_size ? "dump" : "fb");
    if (orknn_bo_create(fd, act_size, &ctx->activation_bo)) {
        orknn_log(0, "memory: failed to allocate activation BO (%u bytes)", act_size);
        return -1;
    }
    orknn_log(1, "memory: activation BO: %u bytes, dma=0x%lx",
              act_size, (unsigned long)ctx->activation_bo.dma_addr);

alloc_io_bos:
    /* Input BOs — use the model's native_size (set from FB f[12]) which
     * already encodes the correct NC1HWC2 layout for that input, including
     * the actual c2 (3 for RGB inputs, 16 for deeper feature maps) and any
     * W padding. Hardcoding c2=16 here would over-allocate AND mismatch
     * the proxy's regcmd byte offsets, causing wrong reads for non-zero
     * RGB input. */
    ctx->input_bos = calloc(m->n_inputs, sizeof(struct orknn_bo));
    for (uint32_t i = 0; i < m->n_inputs; i++) {
        uint32_t in_size = m->inputs[i].native_size;
        if (in_size == 0 && m->inputs[i].n_dims == 4) {
            /* Fallback: compute conservative c2=16 size if FB had no native_size */
            uint32_t N = m->inputs[i].dims[0], H = m->inputs[i].dims[1];
            uint32_t W = m->inputs[i].dims[2], C = m->inputs[i].dims[3];
            uint32_t c2 = 16;
            in_size = N * ((C + c2 - 1) / c2) * H * W * c2;
        }
        in_size = ALIGN_UP(in_size, 4096);
        if (in_size < 4096) in_size = 4096;
        if (orknn_bo_create(fd, in_size, &ctx->input_bos[i])) {
            orknn_log(0, "memory: failed to allocate input BO[%u] (%u bytes)", i, in_size);
            return -1;
        }
        orknn_log(1, "memory: input BO[%u]: %u bytes, dma=0x%lx",
                  i, in_size, (unsigned long)ctx->input_bos[i].dma_addr);
    }

    /* Read proxy output BO sizes from dump if available */
    uint32_t proxy_out_sizes[16] = {0};
    {
        FILE *sf2 = fopen("/tmp/rknn_dump/submit_1.txt", "r");
        if (sf2) {
            char line[256];
            while (fgets(line, sizeof(line), sf2)) {
                uint32_t bi, sz;
                if (sscanf(line, "bo[%u] handle=%*u dma=%*s obj=%*s size=%u", &bi, &sz) == 2) {
                    if (bi >= 4 && bi < 20)
                        proxy_out_sizes[bi - 4] = sz;
                }
            }
            fclose(sf2);
        }
    }

    /* Output BOs */
    ctx->output_bos = calloc(m->n_outputs, sizeof(struct orknn_bo));
    for (uint32_t i = 0; i < m->n_outputs; i++) {
        uint32_t out_size = ALIGN_UP(m->outputs[i].native_size, 4096);
        /* Use proxy's output BO size if larger */
        if (i < 16 && proxy_out_sizes[i] > out_size)
            out_size = ALIGN_UP(proxy_out_sizes[i], 4096);
        if (out_size < 4096) out_size = 4096;
        if (orknn_bo_create(fd, out_size, &ctx->output_bos[i])) {
            orknn_log(0, "memory: failed to allocate output BO[%u] (%u bytes)", i, out_size);
            return -1;
        }
        orknn_log(1, "memory: output BO[%u]: %u bytes, dma=0x%lx",
                  i, out_size, (unsigned long)ctx->output_bos[i].dma_addr);
    }

    /* Prepare task BO: copy task descriptors and patch regcmd addresses.
     * Task regcmd_addr is an offset within the weight BO; add weight BO's DMA address. */
    memcpy(ctx->task_bo.map, m->task_data, m->task_data_size);

    struct {
        uint32_t f[8];
        uint64_t regcmd_addr;
    } __attribute__((packed)) *tasks = ctx->task_bo.map;

    for (uint32_t t = 0; t < m->task_count; t++) {
        /* regcmd_addr is already offset within weight BO (from model parser) */
        tasks[t].regcmd_addr += ctx->weight_bo.dma_addr;
    }

    orknn_bo_sync_to_device(fd, &ctx->task_bo);

    return 0;
}

/* Stub for public API — currently unused but needed for linking */
rknn_tensor_mem *orknn_own_create_mem(struct orknn_context *ctx, uint32_t size)
{
    (void)ctx; (void)size;
    orknn_log(0, "orknn_own_create_mem: NOT IMPLEMENTED");
    return NULL;
}

int orknn_own_destroy_mem(struct orknn_context *ctx, rknn_tensor_mem *mem)
{
    (void)ctx; (void)mem;
    orknn_log(0, "orknn_own_destroy_mem: NOT IMPLEMENTED");
    return RKNN_ERR_FAIL;
}
