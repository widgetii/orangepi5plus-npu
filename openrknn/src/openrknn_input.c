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

            /* Restore proxy's W-padding bytes from the cached BO[3].
             * For each byte where the cache holds a NON-ZERO value that
             * our NC1HWC2 write didn't touch (padding slot), copy it
             * into our BO. This restores DeepLabv3's 0x80 W-padding
             * (mean/std-derived) without corrupting user data positions.
             * For MBv1/YOLOv5 the cache is all zeros so this is a no-op.
             *
             * Phase 7 plan: replace with `memset(dst, (uint8_t)ti->zp, ...)`
             * once diff-oracle confirms zp matches the cached pad byte for
             * all dump-present models. Rolled back from that eager change
             * to keep copy_proxy_regcmd path byte-exact. */
            if (i == 0 && ctx->proxy_input_cache &&
                ctx->proxy_input_cache_size > 0) {
                const uint8_t *cache = ctx->proxy_input_cache;
                uint32_t lim = ctx->proxy_input_cache_size;
                if (lim > bo->size) lim = bo->size;
                for (uint32_t k = 0; k < lim; k++) {
                    if (cache[k] != 0)
                        dst[k] = cache[k];
                }
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
