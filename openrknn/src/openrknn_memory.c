/*
 * openrknn — Memory management: allocate model BOs
 *
 * SPDX-License-Identifier: MIT
 */
#include "openrknn.h"
#include <stdlib.h>
#include <string.h>

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

    /* Task BO — needs KERNEL_MAPPING (0x8) flag so RKNPU driver can read it */
    uint32_t task_size = ALIGN_UP(m->task_data_size, 4096);
    if (task_size < 4096) task_size = 4096;
    if (orknn_bo_create_flags(fd, task_size, 0x40b, &ctx->task_bo)) {
        orknn_log(0, "memory: failed to allocate task BO (%u bytes)", task_size);
        return -1;
    }
    orknn_log(1, "memory: task BO: %u bytes, dma=0x%lx",
              task_size, (unsigned long)ctx->task_bo.dma_addr);

    /* Activation BO (intermediate tensors).
     * Size must cover all intermediate tensor allocations.
     * Heuristic: sum of (input + all output native sizes) * 4 gives
     * enough room for the graph memory planner's allocation. */
    uint32_t act_size = 0;
    for (uint32_t i = 0; i < m->n_inputs; i++)
        act_size += m->inputs[i].native_size;
    for (uint32_t i = 0; i < m->n_outputs; i++)
        act_size += m->outputs[i].native_size;
    act_size *= 4; /* room for intermediate tensors */
    if (m->total_internal_size > act_size)
        act_size = m->total_internal_size;
    if (act_size < 131072)
        act_size = 131072; /* minimum 128KB — must cover all intermediate tensors */
    act_size = ALIGN_UP(act_size, 4096);
    if (orknn_bo_create(fd, act_size, &ctx->activation_bo)) {
        orknn_log(0, "memory: failed to allocate activation BO (%u bytes)", act_size);
        return -1;
    }
    orknn_log(1, "memory: activation BO: %u bytes, dma=0x%lx",
              act_size, (unsigned long)ctx->activation_bo.dma_addr);

    /* Input BOs — need NC1HWC2 padded size, not just native_size.
     * For a 4D NHWC [N,H,W,C] tensor: NC1HWC2 = N*ceil(C/16)*H*W*16. */
    ctx->input_bos = calloc(m->n_inputs, sizeof(struct orknn_bo));
    for (uint32_t i = 0; i < m->n_inputs; i++) {
        uint32_t in_size = m->inputs[i].native_size;
        /* Ensure size covers NC1HWC2 with c2=16 padding */
        if (m->inputs[i].n_dims == 4) {
            uint32_t N = m->inputs[i].dims[0], H = m->inputs[i].dims[1];
            uint32_t W = m->inputs[i].dims[2], C = m->inputs[i].dims[3];
            uint32_t c2 = 16;
            uint32_t nc1hwc2_size = N * ((C + c2 - 1) / c2) * H * W * c2;
            if (nc1hwc2_size > in_size) in_size = nc1hwc2_size;
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

    /* Output BOs */
    ctx->output_bos = calloc(m->n_outputs, sizeof(struct orknn_bo));
    for (uint32_t i = 0; i < m->n_outputs; i++) {
        uint32_t out_size = ALIGN_UP(m->outputs[i].native_size, 4096);
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
