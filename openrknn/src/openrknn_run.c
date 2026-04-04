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

/* Patch regcmd DMA addresses: add BO bases to 0-based offsets.
 * The regcmd in .rknn has offsets relative to each BO's start.
 *
 * From the BO layout investigation (memory/rknn_native_loading.md):
 *   SRC_BASE (0x1070) → activation or input BO
 *   WT_BASE  (0x1110) → weight BO (within the combined wt+rc BO)
 *   DST_BASE (0x4020) → activation BO
 *   WDMA_BASE(0x4110) → activation BO
 *   BS_BASE  (0x5020) → weight BO (bias section)
 *   RDMA act (0x5018) → activation BO
 *   RDMA rel (0x5038) → activation BO
 *   PC_BASE  (0x0010) → weight BO (regcmd chain pointer)
 *   PC rel   (0x6070) → weight BO
 *   PC rel   (0x701c) → weight BO
 */
static void patch_regcmd_addresses(struct orknn_context *ctx)
{
    struct orknn_model *m = &ctx->model;
    uint32_t rc_off = (uint32_t)(m->regcmd_data - m->wt_data);
    uint32_t rc_entries = m->regcmd_size / 8;
    uint64_t *rc = (uint64_t *)((uint8_t *)ctx->weight_bo.map + rc_off);

    uint32_t wt_base  = (uint32_t)ctx->weight_bo.dma_addr;
    uint32_t act_base = (uint32_t)ctx->activation_bo.dma_addr;
    uint32_t in_base  = (uint32_t)ctx->input_bos[0].dma_addr;

    orknn_log(2, "run: patching %u regcmd entries: wt=0x%x act=0x%x in=0x%x",
              rc_entries, wt_base, act_base, in_base);

    uint32_t patched = 0;
    for (uint32_t i = 0; i < rc_entries; i++) {
        uint16_t reg = rc[i] & 0xFFFF;
        uint32_t val = (rc[i] >> 16) & 0xFFFFFFFF;
        uint32_t new_val = val;

        switch (reg) {
        /* Activation/input/output addresses → activation BO base.
         * The activation BO is a single large buffer that holds input,
         * intermediate, and output tensors. All SRC/DST offsets are
         * relative to this BO. */
        case 0x1070: /* CNA_SRC_BASE */
        case 0x4020: /* DPU_DST_BASE */
        case 0x4110: /* WDMA_BASE */
        case 0x5018: /* RDMA activation source */
        case 0x5038: /* RDMA related */
            new_val = act_base + val;
            break;

        /* Weight and bias addresses → weight BO base */
        case 0x1110: /* RDMA_WT_BASE */
        case 0x5020: /* RDMA_BS_BASE */
            new_val = wt_base + val;
            break;

        /* PC chain pointers → weight BO (regcmd is within weight BO) */
        case 0x0010: /* PC_BASE_ADDRESS */
        case 0x6070: /* PC related */
        case 0x701c: /* PC related */
            /* These point to regcmd entries within the weight BO.
             * Add the regcmd section offset + weight BO DMA base. */
            new_val = wt_base + rc_off + val;
            break;

        default:
            continue;
        }

        if (new_val != val) {
            rc[i] = (rc[i] & 0xFFFF000000000000ULL) |
                    ((uint64_t)new_val << 16) |
                    (rc[i] & 0xFFFF);
            patched++;
        }
    }

    orknn_log(1, "run: patched %u/%u regcmd entries", patched, rc_entries);

    /* Sync patched weight BO to device */
    orknn_bo_sync_to_device(ctx->npu_fd, &ctx->weight_bo);
}

int orknn_own_run(struct orknn_context *ctx, rknn_run_extend *extend)
{
    (void)extend;
    struct orknn_model *m = &ctx->model;

    /* Patch DMA addresses on first run */
    static int patched = 0;
    if (!patched) {
        patch_regcmd_addresses(ctx);
        patched = 1;
    }

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
