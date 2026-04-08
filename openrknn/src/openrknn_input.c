/*
 * openrknn — Input processing: NHWC→NC1HWC2 layout transform + quantize
 *
 * SPDX-License-Identifier: MIT
 */
#include "openrknn.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

int orknn_own_inputs_set(struct orknn_context *ctx, uint32_t n_inputs,
                         rknn_input inputs[])
{
    struct orknn_model *m = &ctx->model;

    if (n_inputs > m->n_inputs)
        return RKNN_ERR_INPUT_INVALID;

    for (uint32_t i = 0; i < n_inputs; i++) {
        uint32_t idx = inputs[i].index;
        if (idx >= m->n_inputs)
            return RKNN_ERR_INPUT_INVALID;

        struct orknn_tensor_info *ti = &m->inputs[idx];
        struct orknn_bo *bo = &ctx->input_bos[idx];

        if (inputs[i].pass_through) {
            /* Direct copy — user provides data in NPU native format */
            uint32_t copy_size = inputs[i].size;
            if (copy_size > bo->size)
                copy_size = bo->size;
            memcpy(bo->map, inputs[i].buf, copy_size);
        } else if (ti->n_dims == 4 && ti->fmt == RKNN_TENSOR_NHWC) {
            /* NHWC → NC1HWC2 layout transform.
             * Input dims (NHWC order): [N, H, W, C]
             * NPU native (NC1HWC2): [N, ceil(C/16), H, W, 16] */
            uint32_t N = ti->dims[0], H = ti->dims[1];
            uint32_t W = ti->dims[2], C = ti->dims[3];
            uint32_t c2 = 16;
            uint32_t C1 = (C + c2 - 1) / c2;

            const uint8_t *src = inputs[i].buf;
            uint8_t *dst = bo->map;

            /* Pad with zero for padded channels.
             * CNA_CVT handles zp offset, so we use raw 0 as pad. */
            memset(dst, 0, bo->size);

            if (inputs[i].type == ti->type ||
                (inputs[i].type == RKNN_TENSOR_UINT8 && ti->type == RKNN_TENSOR_INT8) ||
                (inputs[i].type == RKNN_TENSOR_INT8 && ti->type == RKNN_TENSOR_UINT8)) {
                /* Same or compatible dtype — just layout transform.
                 * CNA_CVT hardware handles uint8↔int8 conversion via
                 * the offset register, so we pass raw bytes. */
                for (uint32_t n = 0; n < N; n++) {
                    for (uint32_t h = 0; h < H; h++) {
                        for (uint32_t w = 0; w < W; w++) {
                            for (uint32_t c = 0; c < C; c++) {
                                uint32_t c1 = c / c2;
                                uint32_t c2_idx = c % c2;
                                uint32_t src_off = ((n * H + h) * W + w) * C + c;
                                uint32_t dst_off = ((n * C1 + c1) * H + h) * W * c2 + w * c2 + c2_idx;
                                dst[dst_off] = src[src_off];
                            }
                        }
                    }
                }
            } else if (inputs[i].type == RKNN_TENSOR_FLOAT32) {
                /* Float32 input → quantize to int8/uint8 + layout transform */
                const float *fsrc = (const float *)inputs[i].buf;
                float scale = ti->scale;
                int32_t zp = ti->zp;

                for (uint32_t n = 0; n < N; n++) {
                    for (uint32_t h = 0; h < H; h++) {
                        for (uint32_t w = 0; w < W; w++) {
                            for (uint32_t c = 0; c < C; c++) {
                                uint32_t c1 = c / c2;
                                uint32_t c2_idx = c % c2;
                                uint32_t src_off = ((n * H + h) * W + w) * C + c;
                                uint32_t dst_off = ((n * C1 + c1) * H + h) * W * c2 + w * c2 + c2_idx;

                                float val = fsrc[src_off];
                                int32_t q = (int32_t)roundf(val / scale) + zp;
                                if (q < -128) q = -128;
                                if (q > 127) q = 127;
                                dst[dst_off] = (uint8_t)(q & 0xFF);
                            }
                        }
                    }
                }
            } else if (inputs[i].type == RKNN_TENSOR_UINT8 &&
                       ti->type == RKNN_TENSOR_INT8) {
                /* uint8 input → subtract 128 to get int8 + layout transform */
                for (uint32_t n = 0; n < N; n++) {
                    for (uint32_t h = 0; h < H; h++) {
                        for (uint32_t w = 0; w < W; w++) {
                            for (uint32_t c = 0; c < C; c++) {
                                uint32_t c1 = c / c2;
                                uint32_t c2_idx = c % c2;
                                uint32_t src_off = ((n * H + h) * W + w) * C + c;
                                uint32_t dst_off = ((n * C1 + c1) * H + h) * W * c2 + w * c2 + c2_idx;
                                /* RKNN convention: uint8 → int8 = subtract 128 (zp offset) */
                                dst[dst_off] = (uint8_t)(src[src_off] + (uint8_t)ti->zp);
                            }
                        }
                    }
                }
            } else {
                orknn_log(0, "input: unsupported type conversion: user=%d model=%d",
                          inputs[i].type, ti->type);
                return RKNN_ERR_INPUT_INVALID;
            }
        } else {
            /* Non-4D or non-NHWC: direct copy */
            uint32_t copy_size = inputs[i].size;
            if (copy_size > bo->size) copy_size = bo->size;
            memcpy(bo->map, inputs[i].buf, copy_size);
        }

        orknn_bo_sync_to_device(ctx->npu_fd, bo);

        orknn_log(2, "inputs_set: input[%u] %u bytes -> BO dma=0x%lx + act BO",
                  idx, inputs[i].size, (unsigned long)bo->dma_addr);
    }

    return RKNN_SUCC;
}
