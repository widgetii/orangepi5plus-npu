/*
 * openrknn — Output processing: read NPU output, detile, dequantize
 *
 * The NPU writes output in NC1HWC2 format to the output BO.
 * We need to:
 * 1. Sync output BO from device (cache invalidate)
 * 2. Detile NC1HWC2 → NHWC or NCHW (depending on model format)
 * 3. Optionally dequantize (int8 → float32 if want_float=1)
 * 4. Copy to user buffer or allocate one
 *
 * SPDX-License-Identifier: MIT
 */
#include "openrknn.h"
#include <stdlib.h>
#include <string.h>

int orknn_own_outputs_get(struct orknn_context *ctx, uint32_t n_outputs,
                          rknn_output outputs[], rknn_output_extend *extend)
{
    (void)extend;
    struct orknn_model *m = &ctx->model;

    for (uint32_t i = 0; i < n_outputs; i++) {
        uint32_t idx = outputs[i].index;
        if (idx >= m->n_outputs) return RKNN_ERR_OUTPUT_INVALID;

        struct orknn_tensor_info *ti = &m->outputs[idx];
        struct orknn_bo *bo = &ctx->output_bos[idx];

        /* Sync output BO from device */
        orknn_bo_sync_from_device(ctx->npu_fd, bo);
        /* Also sync activation BO since output often lands there */
        orknn_bo_sync_from_device(ctx->npu_fd, &ctx->activation_bo);

        /* Determine where to read the output from.
         * Discovered during init by signature match against proxy dumps:
         *   valid=0: not discovered, fall back to dedicated output BO
         *   valid=1: read from activation BO at act_output_offsets[idx]
         *   valid=2..N: read from output_bos[valid-2] at offset */
        uint8_t *npu_output_src = (uint8_t *)bo->map;
        if (idx < 16 && ctx->act_output_valid[idx]) {
            uint8_t v = ctx->act_output_valid[idx];
            if (v == 1) {
                /* Read from activation BO */
                orknn_bo_sync_from_device(ctx->npu_fd, &ctx->activation_bo);
                npu_output_src = (uint8_t *)ctx->activation_bo.map + ctx->act_output_offsets[idx];
                orknn_log(2, "outputs_get: reading output[%u] from act+0x%x",
                          idx, ctx->act_output_offsets[idx]);
            } else if (v >= 2 && v - 2 < ctx->model.n_outputs && ctx->output_bos) {
                /* Read from a different output BO */
                int obo_idx = v - 2;
                orknn_bo_sync_from_device(ctx->npu_fd, &ctx->output_bos[obo_idx]);
                npu_output_src = (uint8_t *)ctx->output_bos[obo_idx].map + ctx->act_output_offsets[idx];
                orknn_log(2, "outputs_get: reading output[%u] from output_bos[%d]+0x%x",
                          idx, obo_idx, ctx->act_output_offsets[idx]);
            }
        }

        /* Debug: dump first 32 bytes of output BO */
        if (getenv("ORKNN_DUMP_OUTPUT")) {
            uint8_t *p = (uint8_t *)bo->map;
            fprintf(stderr, "[openrknn] output BO[%u] first 32 bytes:", idx);
            for (int k = 0; k < 32; k++) fprintf(stderr, " %02x", p[k]);
            fprintf(stderr, "\n");
            /* Also dump activation BO around offset 0x700 */
            orknn_bo_sync_from_device(ctx->npu_fd, &ctx->activation_bo);
            uint8_t *act = (uint8_t *)ctx->activation_bo.map;
            fprintf(stderr, "[openrknn] act BO +0x700 first 32:");
            for (int k = 0; k < 32; k++) fprintf(stderr, " %02x", act[0x700 + k]);
            fprintf(stderr, "\n");
        }

        uint32_t out_size;
        if (outputs[i].want_float)
            out_size = ti->n_elems * sizeof(float);
        else
            out_size = ti->size;

        /* Allocate output buffer if not pre-allocated */
        if (!outputs[i].is_prealloc) {
            outputs[i].buf = malloc(out_size);
            if (!outputs[i].buf) return RKNN_ERR_MALLOC_FAIL;
        }
        outputs[i].size = out_size;

        uint8_t *src = npu_output_src;
        uint8_t *dst = (uint8_t *)outputs[i].buf;

        if (ti->n_dims == 4) {
            /* 4D tensor: detile NC1HWC2 → user format (NHWC/NCHW).
             * dims are in NHWC order: [N, H, W, C] */
            uint32_t N = ti->dims[0], H = ti->dims[1];
            uint32_t W = ti->dims[2], C = ti->dims[3];
            uint32_t c2 = 16;
            uint32_t C1 = (C + c2 - 1) / c2;

            if (outputs[i].want_float) {
                /* NC1HWC2 → NHWC float32 with dequantization */
                float *fdst = (float *)dst;
                float scale = ti->scale;
                int32_t zp = ti->zp;

                for (uint32_t n = 0; n < N; n++) {
                    for (uint32_t h = 0; h < H; h++) {
                        for (uint32_t w = 0; w < W; w++) {
                            for (uint32_t c = 0; c < C; c++) {
                                uint32_t c1 = c / c2;
                                uint32_t c2_idx = c % c2;
                                uint32_t src_off = ((n * C1 + c1) * H + h) * W * c2 + w * c2 + c2_idx;
                                uint32_t dst_off = ((n * H + h) * W + w) * C + c;
                                int8_t raw = (int8_t)src[src_off];
                                fdst[dst_off] = ((float)raw - (float)zp) * scale;
                            }
                        }
                    }
                }
            } else if (ti->fmt == RKNN_TENSOR_NCHW) {
                /* NC1HWC2 → NCHW (proxy returns NCHW for NCHW models) */
                for (uint32_t n = 0; n < N; n++) {
                    for (uint32_t c = 0; c < C; c++) {
                        uint32_t c1 = c / c2;
                        uint32_t c2_idx = c % c2;
                        for (uint32_t h = 0; h < H; h++) {
                            for (uint32_t w = 0; w < W; w++) {
                                uint32_t src_off = ((n * C1 + c1) * H + h) * W * c2 + w * c2 + c2_idx;
                                uint32_t dst_off = ((n * C + c) * H + h) * W + w;
                                dst[dst_off] = src[src_off];
                            }
                        }
                    }
                }
            } else {
                /* NC1HWC2 → NHWC */
                for (uint32_t n = 0; n < N; n++) {
                    for (uint32_t h = 0; h < H; h++) {
                        for (uint32_t w = 0; w < W; w++) {
                            for (uint32_t c = 0; c < C; c++) {
                                uint32_t c1 = c / c2;
                                uint32_t c2_idx = c % c2;
                                uint32_t src_off = ((n * C1 + c1) * H + h) * W * c2 + w * c2 + c2_idx;
                                uint32_t dst_off = ((n * H + h) * W + w) * C + c;
                                dst[dst_off] = src[src_off];
                            }
                        }
                    }
                }
            }
        } else {
            /* Non-4D (e.g., 2D [1,1001]): direct copy, trim padding.
             * Native BO may be padded (e.g., 1024 for 1001 elements).
             * Just copy the first n_elems bytes. */
            uint32_t copy_size = ti->n_elems;
            if (outputs[i].want_float) {
                float *fdst = (float *)dst;
                float scale = ti->scale;
                int32_t zp = ti->zp;
                for (uint32_t j = 0; j < copy_size; j++) {
                    int8_t raw = (int8_t)src[j];
                    fdst[j] = ((float)raw - (float)zp) * scale;
                }
            } else {
                memcpy(dst, src, copy_size);
            }
        }

        orknn_log(2, "outputs_get: output[%u] %u bytes%s",
                  idx, out_size, outputs[i].want_float ? " (float)" : "");
    }

    return RKNN_SUCC;
}
