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
             * NPU native (NC1HWC2): [N, C1, H, W_pad, c2] where c2 is taken
             * from the model's native_dims (set by FB f[3]). For RGB inputs
             * the proxy uses c2=3 (tight RGB packing), not c2=16 — using
             * c2=16 here would mismatch the proxy's regcmd byte offsets and
             * cause the NPU to read garbage from padding slots. The c2 we
             * pick here MUST match what the regcmd was generated for. */
            uint32_t N = ti->dims[0], H = ti->dims[1];
            uint32_t W = ti->dims[2], C = ti->dims[3];
            uint32_t c2 = (ti->native_n_dims >= 5 && ti->native_dims[4] > 0)
                          ? ti->native_dims[4] : 16;
            uint32_t C1 = (C + c2 - 1) / c2;
            /* W stride may be padded (e.g. DeepLabv3 W=513 → 528).
             * Derive from native_size: native_size = N*C1*H*W_pad*c2 + alignment.
             * If native_size matches an integer W_pad >= W, use it; otherwise
             * fall back to W. */
            uint32_t W_pad = W;
            uint32_t row_count = N * C1 * H * c2;
            if (row_count > 0 && ti->native_size >= row_count * W) {
                uint32_t cand = ti->native_size / row_count;
                if (cand >= W && cand * row_count <= bo->size)
                    W_pad = cand;
            }

            const uint8_t *src = inputs[i].buf;
            uint8_t *dst = bo->map;

            /* Padding byte for the unused W-pad columns (and the C1
             * padding beyond the real channel count). The CNA_CVT
             * hardware applies (pixel * scale_hw - mean[c]*scale_hw +
             * offset) per channel, so a pad byte equal to mean[c]
             * produces zero contribution to the conv sum. We pick
             * mean[0] because pad is written as a single flat byte
             * across all channels; all models we see in the wild use
             * the same mean per channel (ImageNet RGB uses three but
             * DeepLabv3's [127.5,127.5,127.5] and YOLO's [0,0,0] are
             * both channel-uniform). Previously we snapshotted this
             * from the vendor's pre-run BO[3] via proxy_input_cache;
             * computing it from mean[] removes the last /tmp/rknn_dump
             * read from the inputs_set path.
             *
             * For models without W-padding (MBv1/ResNet50/YOLO: W
             * already 16-aligned) this memset is overwritten entirely
             * by the per-pixel copy below and the pad_byte choice
             * doesn't matter. */
            uint8_t pad_byte = 0;
            if (m->input_attr_valid) {
                float m0 = m->input_attr_mean[0];
                if (m0 < 0.0f) m0 = 0.0f;
                if (m0 > 255.0f) m0 = 255.0f;
                pad_byte = (uint8_t)(int)(m0 + 0.5f);
            }
            memset(dst, pad_byte, bo->size);

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
                                uint32_t dst_off = ((n * C1 + c1) * H + h) * W_pad * c2 + w * c2 + c2_idx;
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
                                uint32_t dst_off = ((n * C1 + c1) * H + h) * W_pad * c2 + w * c2 + c2_idx;

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
                                uint32_t dst_off = ((n * C1 + c1) * H + h) * W_pad * c2 + w * c2 + c2_idx;
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

            /* W-pad bytes are now set via the memset(pad_byte) above
             * derived from input_attr_mean. Previously we snapshotted
             * them from the vendor's dump via proxy_input_cache; that
             * code path is gone. */
        } else {
            /* Non-4D or non-NHWC: direct copy */
            uint32_t copy_size = inputs[i].size;
            if (copy_size > bo->size) copy_size = bo->size;
            memcpy(bo->map, inputs[i].buf, copy_size);
        }

        orknn_bo_sync_to_device(ctx->npu_fd, bo);

        /* Unified activation BO: also copy input data into the activation BO
         * at the subgraph input tensor's f[13] offset. The vendor's regcmd
         * template reads input from the activation BO, not a separate BO. */
        if (ctx->unified_act && ctx->activation_bo.map && m->tensor_offsets) {
            /* Find subgraph input tensor index from the input-consuming op */
            uint32_t sg_in_tidx = 0;
            if (m->ops && m->input_consuming_op_idx < m->op_count) {
                const struct orknn_op_info *ico = &m->ops[m->input_consuming_op_idx];
                if (ico->input_count > 0)
                    sg_in_tidx = ico->input_tensors[0];
            }
            uint32_t act_off = (sg_in_tidx < m->tensor_count) ?
                               m->tensor_offsets[sg_in_tidx] : 0;
            uint32_t copy_size = bo->size;
            if (act_off + copy_size <= ctx->activation_bo.size) {
                memcpy((uint8_t *)ctx->activation_bo.map + act_off,
                       bo->map, copy_size);
                orknn_bo_sync_to_device(ctx->npu_fd, &ctx->activation_bo);
                orknn_log(1, "inputs_set: unified copy %u bytes -> act+0x%x",
                          copy_size, act_off);
            }
        }

        orknn_log(2, "inputs_set: input[%u] %u bytes -> BO dma=0x%lx",
                  idx, inputs[i].size, (unsigned long)bo->dma_addr);
    }

    return RKNN_SUCC;
}
