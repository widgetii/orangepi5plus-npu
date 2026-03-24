/*
 * Weight/bias preparation — converts TFLite weights to NPU DMA format
 * Extracted from Mesa rkt_coefs.c
 * SPDX-License-Identifier: MIT
 */

#include <stdio.h>
#include <math.h>
#include "rnpu_coefs.h"

/* Check if CONV is depthwise based on TFLite op */
static bool is_depthwise(const struct rnpu_tfl_model *tfl,
                         const struct rnpu_tfl_op *op)
{
   if (op->builtin_code != TFLITE_OP_DEPTHWISE_CONV_2D)
      return false;
   unsigned ic = tfl->tensors[op->inputs[0]].shape[3];
   unsigned oc = tfl->tensors[op->outputs[0]].shape[3];
   return ic > 1 && oc > 1;
}

unsigned rnpu_calc_weight_size(unsigned ww, unsigned wh,
                               unsigned ic, unsigned oc,
                               bool depthwise)
{
   ic = MAX2(ic, FEATURE_ATOMIC_SIZE);
   if (depthwise)
      oc = 1;
   else
      oc = ALIGN_UP(MAX2(oc, WEIGHT_ATOMIC_SIZE), 2);
   return ww * wh * oc * ALIGN_UP(ic, WEIGHT_ATOMIC_SIZE) * 2;
}

unsigned rnpu_fill_weights(const struct rnpu_tfl_model *tfl,
                           const struct rnpu_tfl_op *op,
                           uint8_t *dst, unsigned dst_size)
{
   int wt_idx = op->inputs[1]; /* weight tensor */
   const struct rnpu_tfl_tensor *wt = &tfl->tensors[wt_idx];
   const struct rnpu_tfl_buffer *buf = &tfl->buffers[wt->buffer_index];
   bool dw = is_depthwise(tfl, op);

   unsigned weights_width = wt->shape[1];
   unsigned weights_height = wt->shape[2];
   unsigned input_channels_real = tfl->tensors[op->inputs[0]].shape[3];
   unsigned output_channels_real = tfl->tensors[op->outputs[0]].shape[3];
   unsigned input_channels = MAX2(input_channels_real, FEATURE_ATOMIC_SIZE);
   unsigned output_channels = ALIGN_UP(output_channels_real, 2);
   if (dw) output_channels = 1;
   uint8_t zero_point = (uint8_t)wt->quant.zero_point;

   unsigned input_channel_groups = WEIGHT_ATOMIC_SIZE;
   if (dw) input_channel_groups *= 2;

   unsigned ic1 = DIV_ROUND_UP(input_channels, input_channel_groups);
   unsigned ic2 = MIN2(input_channels, input_channel_groups);

   const uint8_t *w_in = buf->data;
   unsigned n = 0;

   for (int oc1 = 0; oc1 < (int)DIV_ROUND_UP(output_channels, WEIGHT_ATOMIC_SIZE); oc1++) {
      for (int i1 = 0; i1 < (int)ic1; i1++) {
         for (int x = 0; x < (int)weights_width; x++) {
            for (int y = 0; y < (int)weights_height; y++) {
               for (int oc2 = 0; oc2 < (int)MIN2(output_channels, WEIGHT_ATOMIC_SIZE); oc2++) {
                  for (int i2 = 0; i2 < (int)ic2; i2++) {
                     unsigned oc = oc1 * WEIGHT_ATOMIC_SIZE + oc2;
                     unsigned ic = i1 * input_channel_groups + i2;

                     if (output_channels_real > 2 &&
                         oc >= ALIGN_UP(output_channels_real, 2))
                        continue;

                     if (oc >= output_channels_real)
                        dst[n++] = 0x0;
                     else if (ic >= input_channels_real) {
                        if (i2 < 16 || (input_channels_real % 32) > 16)
                           dst[n++] = 0;  /* zero weight for padded channels */
                     } else {
                        unsigned flat = oc * weights_width * weights_height * input_channels_real
                                      + x * weights_height * input_channels_real
                                      + y * input_channels_real + ic;
                        dst[n++] = (int8_t)w_in[flat] - (int8_t)zero_point;
                     }
                  }
               }
            }
         }
      }
   }
   return n;
}

unsigned rnpu_fill_weights_group(const struct rnpu_tfl_model *tfl,
                                 const struct rnpu_tfl_op *op,
                                 const unsigned *channel_indices,
                                 unsigned group_count,
                                 uint8_t *dst, unsigned dst_size)
{
   int wt_idx = op->inputs[1];
   const struct rnpu_tfl_tensor *wt = &tfl->tensors[wt_idx];
   const struct rnpu_tfl_buffer *buf = &tfl->buffers[wt->buffer_index];

   unsigned ww = wt->shape[1];
   unsigned wh = wt->shape[2];
   unsigned ic_real = tfl->tensors[op->inputs[0]].shape[3];
   unsigned ic = MAX2(ic_real, FEATURE_ATOMIC_SIZE);
   uint8_t zp = (uint8_t)wt->quant.zero_point;

   unsigned oc_pad = ALIGN_UP(MAX2(group_count, WEIGHT_ATOMIC_SIZE), 2);
   unsigned ic1 = DIV_ROUND_UP(ic, WEIGHT_ATOMIC_SIZE);
   unsigned ic2 = MIN2(ic, WEIGHT_ATOMIC_SIZE);

   const uint8_t *w_in = buf->data;
   unsigned n = 0;

   for (int o1 = 0; o1 < (int)DIV_ROUND_UP(oc_pad, WEIGHT_ATOMIC_SIZE); o1++) {
      for (int i1 = 0; i1 < (int)ic1; i1++) {
         for (int x = 0; x < (int)ww; x++) {
            for (int y = 0; y < (int)wh; y++) {
               for (int o2 = 0; o2 < (int)MIN2(oc_pad, WEIGHT_ATOMIC_SIZE); o2++) {
                  for (int i2 = 0; i2 < (int)ic2; i2++) {
                     unsigned oc_local = o1 * WEIGHT_ATOMIC_SIZE + o2;
                     unsigned icc = i1 * WEIGHT_ATOMIC_SIZE + i2;

                     if (oc_local >= group_count) {
                        if (oc_local < oc_pad) dst[n++] = 0x0;
                     } else if (icc >= ic_real) {
                        if (i2 < 16 || (ic_real % 32) > 16)
                           dst[n++] = 0;  /* zero weight for padded channels */
                     } else {
                        unsigned oc_global = channel_indices[oc_local];
                        unsigned flat = oc_global * ww * wh * ic_real
                                      + x * wh * ic_real + y * ic_real + icc;
                        dst[n++] = (int8_t)w_in[flat] - (int8_t)zp;
                     }
                  }
               }
            }
         }
      }
   }
   return n;
}

static int32_t calc_bias_correction(const struct rnpu_tfl_model *tfl,
                                    const struct rnpu_tfl_op *op,
                                    unsigned oc)
{
   int wt_idx = op->inputs[1];
   const struct rnpu_tfl_tensor *wt = &tfl->tensors[wt_idx];
   const struct rnpu_tfl_buffer *buf = &tfl->buffers[wt->buffer_index];
   unsigned ic = tfl->tensors[op->inputs[0]].shape[3];
   const struct rnpu_tfl_tensor *it = &tfl->tensors[op->inputs[0]];
   /* Input/output zero points: int8 tensors need +128 for uint8 NPU domain */
   int izp = (it->type == 9) ? (uint8_t)(it->quant.zero_point + 128)
                              : (uint8_t)it->quant.zero_point;
   unsigned ww = wt->shape[1];
   unsigned wh = wt->shape[2];
   /* Weight zero point stays in raw domain (used for weight arithmetic) */
   int wzp = (uint8_t)wt->quant.zero_point;
   bool dw = is_depthwise(tfl, op);
   const uint8_t *w = buf->data;

   /* Compute bias correction: compensates for (izp - 0x80) offset in CNA input.
    * The weight difference (w_q - wzp) must be computed in the TFLite quantized
    * domain, not in NPU offset-binary domain. For uint8 weights (wzp=128):
    * w_q - wzp = w_byte - 128. For int8 weights (wzp=0): w_q - wzp = (int8_t)w_byte. */
   int32_t corr = 0;
   if (dw) {
      for (unsigned x = 0; x < ww; x++)
         for (unsigned y = 0; y < wh; y++) {
            unsigned flat = x * wh * ic + y * ic + oc;
            corr += (w[flat] - wzp) * (izp - 0x80);
         }
   } else {
      for (unsigned x = 0; x < ww; x++)
         for (unsigned y = 0; y < wh; y++)
            for (unsigned i = 0; i < ic; i++) {
               unsigned flat = oc * ww * wh * ic + x * wh * ic + y * ic + i;
               int32_t wd = (int32_t)(int8_t)w[flat] - (int32_t)(int8_t)wzp;
               corr += wd * (izp - 0x80);
            }
   }
   return corr;
}

unsigned rnpu_fill_biases(const struct rnpu_tfl_model *tfl,
                          const struct rnpu_tfl_op *op,
                          unsigned *truncate_bits,
                          uint8_t *dst, unsigned dst_size)
{
   int bias_idx = op->inputs[2];
   const struct rnpu_tfl_tensor *bt = &tfl->tensors[bias_idx];
   const struct rnpu_tfl_buffer *buf = &tfl->buffers[bt->buffer_index];
   const int32_t *biases_in = (const int32_t *)buf->data;
   unsigned oc = tfl->tensors[op->outputs[0]].shape[3];

   int wt_idx = op->inputs[1];
   const struct rnpu_tfl_tensor *wt = &tfl->tensors[wt_idx];

   /* Truncation heuristic (from Mesa — hardcoded scale matching) */
   float w_scale;
   if (wt->quant.scales) {
      w_scale = wt->quant.scales[0];
      for (unsigned i = 1; i < wt->quant.num_scales; i++)
         if (wt->quant.scales[i] > w_scale) w_scale = wt->quant.scales[i];
   } else {
      w_scale = wt->quant.scale;
   }

   *truncate_bits = 0;
   uint32_t sb = fui(w_scale);
   if (sb == 0x3a88323f || sb == 0x3c0060de || sb == 0x3c06022d ||
       sb == 0x3c1642e3 || sb == 0x3c1e3f51 || sb == 0x3c5c8aa8 ||
       sb == 0x3c615e93 || sb == 0x3c7326a2 || sb == 0x3c783013 ||
       sb == 0x3d1748e6 || sb == 0x3d282992 || sb == 0x3d2e87ae ||
       sb == 0x3d77f5f6 || sb == 0x3a9a5956 || sb == 0x3caebc56)
      *truncate_bits = 1;

   uint32_t *biases = (uint32_t *)dst;
   for (unsigned i = 0; i < oc; i++) {
      int32_t corr = calc_bias_correction(tfl, op, i);
      int32_t bv = biases_in[i];
      if (wt->quant.scales && w_scale != 0.0f)
         bv = (int32_t)roundf(bv * (wt->quant.scales[i] / w_scale));
      biases[i] = (bv - corr) / (1 << *truncate_bits);
   }
   return oc * sizeof(uint32_t);
}

unsigned rnpu_fill_biases_group(const struct rnpu_tfl_model *tfl,
                                const struct rnpu_tfl_op *op,
                                const unsigned *channel_indices,
                                unsigned group_count,
                                unsigned *truncate_bits,
                                uint8_t *dst, unsigned dst_size)
{
   int bias_idx = op->inputs[2];
   const struct rnpu_tfl_buffer *buf = &tfl->buffers[tfl->tensors[bias_idx].buffer_index];
   const int32_t *biases_in = (const int32_t *)buf->data;
   *truncate_bits = 0;

   uint32_t *biases = (uint32_t *)dst;
   for (unsigned i = 0; i < group_count; i++) {
      unsigned oc = channel_indices[i];
      int32_t corr = calc_bias_correction(tfl, op, oc);
      biases[i] = biases_in[oc] - corr;
   }
   return group_count * sizeof(uint32_t);
}

int32_t rnpu_compute_bias_scalar(const struct rnpu_tfl_model *tfl,
                                 const struct rnpu_tfl_op *op,
                                 unsigned channel)
{
   int bias_idx = op->inputs[2];
   const struct rnpu_tfl_buffer *buf = &tfl->buffers[tfl->tensors[bias_idx].buffer_index];
   const int32_t *biases_in = (const int32_t *)buf->data;

   int32_t corr = calc_bias_correction(tfl, op, channel);
   return biases_in[channel] - corr;
}

/*
 * Fill BRDMA data for RKNPU per-channel requantization.
 *
 * The RKNN runtime uses BRDMA_DATA_USE=7 which loads both bias (for BS ALU ADD)
 * and MUL scale (for BS MUL) in one DMA transfer. Layout is determined by
 * Phase 1 analysis — initially using Layout A: [bias int32 x oc_pad] [mul int16 x oc_pad].
 *
 * Returns the max conv_scale (input_scale * max_weight_scale / output_scale)
 * which is used for the uniform OUT_CVT_SCALE/SHIFT.
 */
float rnpu_fill_brdma_data(const struct rnpu_tfl_model *tfl,
                            const struct rnpu_tfl_op *op,
                            const struct rnpu_operation *npu_op,
                            uint8_t *dst, unsigned dst_size,
                            unsigned *out_mul_shift)
{
   unsigned oc = tfl->tensors[op->outputs[0]].shape[3];
   unsigned oc_pad = ALIGN_UP(MAX2(oc, 32), 16);
   int wt_idx = op->inputs[1];
   const struct rnpu_tfl_tensor *wt = &tfl->tensors[wt_idx];
   int bias_idx = op->inputs[2];
   const struct rnpu_tfl_buffer *bias_buf = &tfl->buffers[tfl->tensors[bias_idx].buffer_index];
   const int32_t *biases_in = (const int32_t *)bias_buf->data;

   float input_scale = tfl->tensors[op->inputs[0]].quant.scale;
   float output_scale = tfl->tensors[op->outputs[0]].quant.scale;

   /* Find max weight scale */
   float max_ws = wt->quant.scales[0];
   for (unsigned i = 1; i < wt->quant.num_scales; i++)
      if (wt->quant.scales[i] > max_ws) max_ws = wt->quant.scales[i];

   float max_conv_scale = (input_scale * max_ws) / output_scale;

   /* MUL shift from BS_MUL_CFG — RKNN uses shift=14 */
   unsigned mul_shift = 14;

   /* Truncation bits — already computed by compile_weights_and_biases */
   unsigned truncate = npu_op->truncate_bits;

   /* Store mul_shift in op for regcmd generation */
   /* We encode it via the return value and the npu_op structure.
    * For now, store mul_shift in a field we can access from regcmd.
    * Use per_channel_scale_count as a carrier (ugly but functional). */

   /* RKNN BRDMA layout: groups of 8 channels per 64-byte chunk.
    * Each 64-byte chunk:
    *   [0..31]: 8 × int32 bias (32 bytes)
    *   [32..47]: padding (16 bytes, zeros)
    *   [48..63]: 8 × int16 mul_scale (16 bytes)
    *
    * MUL reference is min_ws: MUL[c] = ws[c]/min_ws * 2^mul_shift
    * OUT_CVT uses min_conv_scale = input_scale * min_ws / output_scale
    */
   memset(dst, 0, dst_size);

   unsigned num_groups = oc_pad / 8;
   for (unsigned g = 0; g < num_groups; g++) {
      uint8_t *chunk = dst + g * 64;
      int32_t *bias_grp = (int32_t *)(chunk);         /* offset 0 */
      int16_t *mul_grp = (int16_t *)(chunk + 48);     /* offset 48 (matches RKNN dump) */

      for (unsigned i = 0; i < 8; i++) {
         unsigned c = g * 8 + i;
         if (c < oc) {
            int32_t corr = calc_bias_correction(tfl, op, c);
            /* Bias is NOT rescaled — the MUL unit will scale both the
             * accumulator and bias by ws[c]/max_ws, providing the correct
             * per-channel effective scale. */
            int32_t bv = biases_in[c];
            bias_grp[i] = (bv - corr) / (1 << truncate);

            float ratio = wt->quant.scales[c] / max_ws;
            mul_grp[i] = (int16_t)roundf(ratio * (float)(1 << mul_shift));
         }
      }
   }

   if (out_mul_shift) *out_mul_shift = mul_shift;

   return max_conv_scale;
}

/* Fill BRDMA data for a specific requant group within a per-channel op.
 * Only the channels in this group get their true MUL ratio;
 * all channels outside the group get MUL=0 (zeroed output).
 * group_max_ws is the reference scale for this group's OUT_CVT. */
float rnpu_fill_brdma_data_group(const struct rnpu_tfl_model *tfl,
                                   const struct rnpu_tfl_op *op,
                                   const struct rnpu_operation *npu_op,
                                   const unsigned *sorted_channels,
                                   unsigned group_ch_offset,
                                   unsigned group_size,
                                   float group_max_ws,
                                   uint8_t *dst, unsigned dst_size,
                                   unsigned *out_mul_shift)
{
   unsigned oc = tfl->tensors[op->outputs[0]].shape[3];
   unsigned oc_pad = ALIGN_UP(MAX2(oc, 32), 16);
   int wt_idx = op->inputs[1];
   const struct rnpu_tfl_tensor *wt = &tfl->tensors[wt_idx];
   int bias_idx = op->inputs[2];
   const struct rnpu_tfl_buffer *bias_buf = &tfl->buffers[tfl->tensors[bias_idx].buffer_index];
   const int32_t *biases_in = (const int32_t *)bias_buf->data;
   unsigned truncate = npu_op->truncate_bits;
   unsigned mul_shift = 14;

   /* Build a set of channels in this group for fast lookup */
   bool *in_group = calloc(oc, sizeof(bool));
   for (unsigned i = 0; i < group_size; i++)
      in_group[sorted_channels[group_ch_offset + i]] = true;

   memset(dst, 0, dst_size);

   unsigned num_chunks = oc_pad / 8;
   for (unsigned g = 0; g < num_chunks; g++) {
      uint8_t *chunk = dst + g * 64;
      int32_t *bias_grp = (int32_t *)(chunk);
      int16_t *mul_grp = (int16_t *)(chunk + 48);

      for (unsigned i = 0; i < 8; i++) {
         unsigned c = g * 8 + i;
         if (c < oc) {
            int32_t corr = calc_bias_correction(tfl, op, c);
            /* Bias is NOT rescaled — MUL handles per-channel correction */
            int32_t bv = biases_in[c];
            bias_grp[i] = (bv - corr) / (1 << truncate);

            float ratio = wt->quant.scales[c] / group_max_ws;
            mul_grp[i] = (int16_t)roundf(ratio * (float)(1 << mul_shift));
         }
      }
   }

   free(in_group);
   if (out_mul_shift) *out_mul_shift = mul_shift;
   return (npu_op->input_scale * group_max_ws) / npu_op->output_scale;
}

/* Compute BRDMA buffer size for one operation.
 * RKNN layout: ceil(oc_pad/8) groups × 64 bytes each. */
unsigned rnpu_calc_brdma_size(unsigned oc)
{
   unsigned oc_pad = ALIGN_UP(MAX2(oc, 32), 16);
   return (oc_pad / 8) * 64;
}
