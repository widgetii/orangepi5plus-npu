/*
 * CPU software ops — extracted from Mesa rkt_ml.c
 * SPDX-License-Identifier: MIT
 */

#include "rnpu_sw_ops.h"

static uint8_t *tensor_ptr(struct rnpu_model *m, unsigned idx)
{
   return (uint8_t *)m->activation_bo.map + m->tensors[idx].offset;
}

static void exec_concat(struct rnpu_model *m, struct rnpu_operation *op)
{
   unsigned w = op->output_width, h = op->output_height;
   unsigned out_ch = op->output_channels;
   uint8_t *out = tensor_ptr(m, op->output_tensor);

   if (m->sw_only) {
      unsigned num_pixels = w * h;
      unsigned ch_off = 0;
      for (unsigned i = 0; i < op->sw.concat.input_count; i++) {
         unsigned idx = op->sw.concat.input_tensors[i];
         unsigned ic = op->sw.concat.input_channels_arr[i];
         uint8_t *in = tensor_ptr(m, idx);
         for (unsigned p = 0; p < num_pixels; p++)
            memcpy(out + p * out_ch + ch_off, in + p * ic, ic);
         ch_off += ic;
      }
   } else {
      /* NPU interleaved format — check alignment */
      bool all_aligned = true;
      for (unsigned i = 0; i < op->sw.concat.input_count; i++)
         if (op->sw.concat.input_channels_arr[i] % FEATURE_ATOMIC_SIZE != 0)
            all_aligned = false;

      if (all_aligned) {
         unsigned group_plane = w * h * FEATURE_ATOMIC_SIZE;
         unsigned out_go = 0;
         for (unsigned i = 0; i < op->sw.concat.input_count; i++) {
            unsigned idx = op->sw.concat.input_tensors[i];
            unsigned ic = op->sw.concat.input_channels_arr[i];
            unsigned ig = DIV_ROUND_UP(ic, FEATURE_ATOMIC_SIZE);
            memcpy(out + out_go * group_plane, tensor_ptr(m, idx), ig * group_plane);
            out_go += ig;
         }
      } else {
         unsigned ch_off = 0;
         for (unsigned i = 0; i < op->sw.concat.input_count; i++) {
            unsigned idx = op->sw.concat.input_tensors[i];
            unsigned ic = op->sw.concat.input_channels_arr[i];
            uint8_t *in = tensor_ptr(m, idx);
            for (unsigned c = 0; c < ic; c++) {
               unsigned sg = c / FEATURE_ATOMIC_SIZE, sc = c % FEATURE_ATOMIC_SIZE;
               unsigned dg = (ch_off + c) / FEATURE_ATOMIC_SIZE;
               unsigned dc = (ch_off + c) % FEATURE_ATOMIC_SIZE;
               for (unsigned x = 0; x < w; x++)
                  for (unsigned y = 0; y < h; y++)
                     out[NPU_OFFSET(dg, x, y, w, h) + dc] =
                        in[NPU_OFFSET(sg, x, y, w, h) + sc];
            }
            ch_off += ic;
         }
      }
   }
}

static void exec_max_pool(struct rnpu_model *m, struct rnpu_operation *op)
{
   unsigned in_w = op->input_width, in_h = op->input_height;
   unsigned out_w = op->output_width, out_h = op->output_height;
   unsigned ch = op->input_channels;
   unsigned groups = DIV_ROUND_UP(ch, FEATURE_ATOMIC_SIZE);
   unsigned fw = op->sw.pool.filter_width, fh = op->sw.pool.filter_height;
   unsigned sx = op->sw.pool.stride_x, sy = op->sw.pool.stride_y;
   unsigned pbw = 0, pbh = 0;
   if (op->sw.pool.padding_same) {
      unsigned ptw = (out_w - 1) * sx + fw - in_w;
      unsigned pth = (out_h - 1) * sy + fh - in_h;
      pbw = ptw / 2; pbh = pth / 2;
   }
   uint8_t *in = tensor_ptr(m, op->input_tensor);
   uint8_t *out = tensor_ptr(m, op->output_tensor);

   if (m->sw_only) {
      for (unsigned ox = 0; ox < out_w; ox++)
         for (unsigned oy = 0; oy < out_h; oy++) {
            uint8_t *d = out + (ox * out_h + oy) * ch;
            memset(d, 0x80, ch);
            for (unsigned fx = 0; fx < fh; fx++)
               for (unsigned fy = 0; fy < fw; fy++) {
                  int ix = (int)(ox * sy) - (int)pbw + (int)fx;
                  int iy = (int)(oy * sx) - (int)pbh + (int)fy;
                  if (ix < 0 || ix >= (int)in_w || iy < 0 || iy >= (int)in_h) continue;
                  uint8_t *s = in + (ix * in_h + iy) * ch;
                  for (unsigned c = 0; c < ch; c++)
                     if ((int8_t)s[c] > (int8_t)d[c]) d[c] = s[c];
               }
         }
   } else {
      for (unsigned g = 0; g < groups; g++) {
         unsigned rc = MIN2(FEATURE_ATOMIC_SIZE, ch - g * FEATURE_ATOMIC_SIZE);
         for (unsigned ox = 0; ox < out_w; ox++)
            for (unsigned oy = 0; oy < out_h; oy++) {
               uint8_t *d = out + NPU_OFFSET(g, ox, oy, out_w, out_h);
               memset(d, 0x80, FEATURE_ATOMIC_SIZE);
               for (unsigned fx = 0; fx < fw; fx++)
                  for (unsigned fy = 0; fy < fh; fy++) {
                     int ix = (int)(ox * sx) - (int)pbw + (int)fx;
                     int iy = (int)(oy * sy) - (int)pbh + (int)fy;
                     if (ix < 0 || ix >= (int)in_w || iy < 0 || iy >= (int)in_h) continue;
                     uint8_t *s = in + NPU_OFFSET(g, ix, iy, in_w, in_h);
                     for (unsigned c = 0; c < rc; c++)
                        if ((int8_t)s[c] > (int8_t)d[c]) d[c] = s[c];
                  }
            }
      }
   }
}

static void exec_pad(struct rnpu_model *m, struct rnpu_operation *op)
{
   unsigned in_w = op->input_width, in_h = op->input_height;
   unsigned out_w = op->output_width, out_h = op->output_height;
   unsigned ch = op->input_channels;
   unsigned groups = DIV_ROUND_UP(ch, FEATURE_ATOMIC_SIZE);
   unsigned pbw = op->sw.pad.pad_before_w, pbh = op->sw.pad.pad_before_h;
   uint8_t pv = (uint8_t)((int)op->input_zero_point - 0x80);
   uint8_t *in = tensor_ptr(m, op->input_tensor);
   uint8_t *out = tensor_ptr(m, op->output_tensor);

   if (m->sw_only) {
      for (unsigned ox = 0; ox < out_w; ox++)
         for (unsigned oy = 0; oy < out_h; oy++) {
            int ix = (int)ox - (int)pbw, iy = (int)oy - (int)pbh;
            uint8_t *d = out + (ox * out_h + oy) * ch;
            if (ix >= 0 && ix < (int)in_w && iy >= 0 && iy < (int)in_h)
               memcpy(d, in + (ix * in_h + iy) * ch, ch);
            else
               memset(d, pv, ch);
         }
   } else {
      for (unsigned g = 0; g < groups; g++)
         for (unsigned ox = 0; ox < out_w; ox++)
            for (unsigned oy = 0; oy < out_h; oy++) {
               int ix = (int)ox - (int)pbw, iy = (int)oy - (int)pbh;
               uint8_t *d = out + NPU_OFFSET(g, ox, oy, out_w, out_h);
               if (ix >= 0 && ix < (int)in_w && iy >= 0 && iy < (int)in_h)
                  memcpy(d, in + NPU_OFFSET(g, ix, iy, in_w, in_h), FEATURE_ATOMIC_SIZE);
               else
                  memset(d, pv, FEATURE_ATOMIC_SIZE);
            }
   }
}

static void exec_resize_nearest(struct rnpu_model *m, struct rnpu_operation *op)
{
   unsigned in_w = op->input_width, in_h = op->input_height;
   unsigned out_w = op->output_width, out_h = op->output_height;
   unsigned ch = op->input_channels;
   unsigned groups = DIV_ROUND_UP(ch, FEATURE_ATOMIC_SIZE);
   uint8_t *in = tensor_ptr(m, op->input_tensor);
   uint8_t *out = tensor_ptr(m, op->output_tensor);

   if (m->sw_only) {
      for (unsigned ox = 0; ox < out_w; ox++)
         for (unsigned oy = 0; oy < out_h; oy++) {
            unsigned ix = ox * in_w / out_w, iy = oy * in_h / out_h;
            memcpy(out + (ox * out_h + oy) * ch, in + (ix * in_h + iy) * ch, ch);
         }
   } else {
      for (unsigned g = 0; g < groups; g++)
         for (unsigned ox = 0; ox < out_w; ox++)
            for (unsigned oy = 0; oy < out_h; oy++) {
               unsigned ix = ox * in_w / out_w, iy = oy * in_h / out_h;
               memcpy(out + NPU_OFFSET(g, ox, oy, out_w, out_h),
                      in + NPU_OFFSET(g, ix, iy, in_w, in_h), FEATURE_ATOMIC_SIZE);
            }
   }
}

static void exec_avg_pool(struct rnpu_model *m, struct rnpu_operation *op)
{
   unsigned in_w = op->input_width, in_h = op->input_height;
   unsigned out_w = op->output_width, out_h = op->output_height;
   unsigned ch = op->input_channels;
   unsigned groups = DIV_ROUND_UP(ch, FEATURE_ATOMIC_SIZE);
   unsigned fw = op->sw.pool.filter_width, fh = op->sw.pool.filter_height;
   unsigned sx = op->sw.pool.stride_x, sy = op->sw.pool.stride_y;
   unsigned pbw = 0, pbh = 0;
   if (op->sw.pool.padding_same) {
      unsigned ptw = (out_w - 1) * sx + fw - in_w;
      unsigned pth = (out_h - 1) * sy + fh - in_h;
      pbw = ptw / 2; pbh = pth / 2;
   }
   uint8_t *in = tensor_ptr(m, op->input_tensor);
   uint8_t *out = tensor_ptr(m, op->output_tensor);

   if (m->sw_only) {
      for (unsigned ox = 0; ox < out_w; ox++)
         for (unsigned oy = 0; oy < out_h; oy++) {
            uint8_t *d = out + (ox * out_h + oy) * ch;
            for (unsigned c = 0; c < ch; c++) {
               int32_t sum = 0;
               unsigned count = 0;
               for (unsigned fx = 0; fx < fw; fx++)
                  for (unsigned fy = 0; fy < fh; fy++) {
                     int ix = (int)(ox * sy) - (int)pbw + (int)fx;
                     int iy = (int)(oy * sx) - (int)pbh + (int)fy;
                     if (ix < 0 || ix >= (int)in_w || iy < 0 || iy >= (int)in_h) continue;
                     sum += (int)(int8_t)in[(ix * in_h + iy) * ch + c];
                     count++;
                  }
               int val = count ? (int)roundf((float)sum / count) : 0;
               if (val < -128) val = -128;
               if (val > 127) val = 127;
               d[c] = (uint8_t)(int8_t)val;
            }
         }
   } else {
      for (unsigned g = 0; g < groups; g++) {
         unsigned rc = MIN2(FEATURE_ATOMIC_SIZE, ch - g * FEATURE_ATOMIC_SIZE);
         for (unsigned ox = 0; ox < out_w; ox++)
            for (unsigned oy = 0; oy < out_h; oy++) {
               uint8_t *d = out + NPU_OFFSET(g, ox, oy, out_w, out_h);
               for (unsigned c = 0; c < rc; c++) {
                  int32_t sum = 0;
                  unsigned count = 0;
                  for (unsigned fx = 0; fx < fw; fx++)
                     for (unsigned fy = 0; fy < fh; fy++) {
                        int ix = (int)(ox * sx) - (int)pbw + (int)fx;
                        int iy = (int)(oy * sy) - (int)pbh + (int)fy;
                        if (ix < 0 || ix >= (int)in_w || iy < 0 || iy >= (int)in_h) continue;
                        sum += (int)(int8_t)(in[NPU_OFFSET(g, ix, iy, in_w, in_h) + c]);
                        count++;
                     }
                  int val = count ? (int)roundf((float)sum / count) : 0;
                  if (val < -128) val = -128;
                  if (val > 127) val = 127;
                  d[c] = (uint8_t)(int8_t)val;
               }
               /* Zero-fill padding channels */
               for (unsigned c = rc; c < FEATURE_ATOMIC_SIZE; c++)
                  d[c] = 0;
            }
      }
   }
}

static void exec_reshape(struct rnpu_model *m, struct rnpu_operation *op)
{
   /* Reshape is a data reinterpretation. If dimensions change but the
    * underlying NPU-format layout is compatible, it's a memcpy or no-op.
    * For MBv1: 1×1×1024 (NPU) → 1×1×1024 (NPU) — same layout, different
    * TFLite shape metadata. Just copy if different tensors. */
   if (op->input_tensor == op->output_tensor)
      return;

   uint8_t *in = tensor_ptr(m, op->input_tensor);
   uint8_t *out = tensor_ptr(m, op->output_tensor);
   unsigned in_sz = m->tensors[op->input_tensor].size;
   unsigned out_sz = m->tensors[op->output_tensor].size;
   memcpy(out, in, MIN2(in_sz, out_sz));
}

static void exec_softmax(struct rnpu_model *m, struct rnpu_operation *op)
{
   unsigned w = op->input_width, h = op->input_height;
   unsigned ch = op->input_channels;
   unsigned total = w * h * ch;
   float in_scale = op->sw.softmax.in_scale;
   int in_zp = op->sw.softmax.in_zp;
   float out_scale = op->sw.softmax.out_scale;
   int out_zp = op->sw.softmax.out_zp;
   uint8_t *in = tensor_ptr(m, op->input_tensor);
   uint8_t *out = tensor_ptr(m, op->output_tensor);

   /* Softmax is applied over the entire flattened output (channel dim).
    * Data is in NPU interleaved format unless sw_only. */
   float *exp_vals = malloc(total * sizeof(float));
   float max_val = -1e30f;

   if (m->sw_only) {
      for (unsigned i = 0; i < total; i++) {
         float v = ((int)(int8_t)in[i] - in_zp) * in_scale;
         if (v > max_val) max_val = v;
      }
      float sum = 0;
      for (unsigned i = 0; i < total; i++) {
         float v = ((int)(int8_t)in[i] - in_zp) * in_scale;
         exp_vals[i] = expf(v - max_val);
         sum += exp_vals[i];
      }
      for (unsigned i = 0; i < total; i++) {
         float prob = exp_vals[i] / sum;
         int q = (int)roundf(prob / out_scale) + out_zp;
         if (q < -128) q = -128;
         if (q > 127) q = 127;
         out[i] = (uint8_t)(int8_t)q;
      }
   } else {
      unsigned groups = DIV_ROUND_UP(ch, FEATURE_ATOMIC_SIZE);
      /* NPU format: stored value = TFLite_uint8 - 0x80
       * To read:  tflite_val = (uint8_t)(stored + 0x80)  [unsigned]
       * To write: stored = tflite_val - 0x80 */

      /* First pass: find max for numerical stability */
      for (unsigned g = 0; g < groups; g++) {
         unsigned rc = MIN2(FEATURE_ATOMIC_SIZE, ch - g * FEATURE_ATOMIC_SIZE);
         for (unsigned x = 0; x < w; x++)
            for (unsigned y = 0; y < h; y++) {
               uint8_t *p = in + NPU_OFFSET(g, x, y, w, h);
               for (unsigned c = 0; c < rc; c++) {
                  int tv = (int)(uint8_t)(p[c] + 0x80);
                  float v = (tv - in_zp) * in_scale;
                  if (v > max_val) max_val = v;
               }
            }
      }
      /* Second pass: exp + sum */
      float sum = 0;
      unsigned idx = 0;
      for (unsigned g = 0; g < groups; g++) {
         unsigned rc = MIN2(FEATURE_ATOMIC_SIZE, ch - g * FEATURE_ATOMIC_SIZE);
         for (unsigned x = 0; x < w; x++)
            for (unsigned y = 0; y < h; y++) {
               uint8_t *p = in + NPU_OFFSET(g, x, y, w, h);
               for (unsigned c = 0; c < rc; c++) {
                  int tv = (int)(uint8_t)(p[c] + 0x80);
                  float v = (tv - in_zp) * in_scale;
                  float e = expf(v - max_val);
                  exp_vals[idx++] = e;
                  sum += e;
               }
            }
      }
      /* Third pass: normalize + requantize */
      idx = 0;
      for (unsigned g = 0; g < groups; g++) {
         unsigned rc = MIN2(FEATURE_ATOMIC_SIZE, ch - g * FEATURE_ATOMIC_SIZE);
         for (unsigned x = 0; x < w; x++)
            for (unsigned y = 0; y < h; y++) {
               uint8_t *p = out + NPU_OFFSET(g, x, y, w, h);
               for (unsigned c = 0; c < rc; c++) {
                  float prob = exp_vals[idx++] / sum;
                  int q = (int)roundf(prob / out_scale) + out_zp;
                  if (q < 0) q = 0;
                  if (q > 255) q = 255;
                  p[c] = (uint8_t)(q - 0x80);
               }
               for (unsigned c = rc; c < FEATURE_ATOMIC_SIZE; c++)
                  p[c] = (uint8_t)(out_zp - 0x80);
            }
      }
   }
   free(exp_vals);
}

static void exec_logistic(struct rnpu_model *m, struct rnpu_operation *op)
{
   unsigned w = op->input_width, h = op->input_height, ch = op->input_channels;
   unsigned groups = DIV_ROUND_UP(ch, FEATURE_ATOMIC_SIZE);
   const uint8_t *lut = m->sw_only ? op->sw.logistic.raw_lut : op->sw.logistic.lut;
   unsigned count = m->sw_only ? w * h * ch : groups * w * h * FEATURE_ATOMIC_SIZE;
   uint8_t *in = tensor_ptr(m, op->input_tensor);
   uint8_t *out = tensor_ptr(m, op->output_tensor);
   for (unsigned i = 0; i < count; i++)
      out[i] = lut[in[i]];
}

static void exec_fully_connected(struct rnpu_model *m, struct rnpu_operation *op)
{
   unsigned in_size = op->sw.fc.input_size;
   unsigned out_size = op->sw.fc.output_size;
   const int8_t *weights = op->sw.fc.weights;
   const int32_t *bias = op->sw.fc.bias;
   int in_zp = op->sw.fc.in_zp;
   int w_zp = op->sw.fc.w_zp;
   float in_scale = op->sw.fc.in_scale;
   float out_scale = op->sw.fc.out_scale;
   int out_zp = op->sw.fc.out_zp;
   bool per_channel = op->sw.fc.num_w_scales > 1;

   uint8_t *in = tensor_ptr(m, op->input_tensor);
   uint8_t *out = tensor_ptr(m, op->output_tensor);

   for (unsigned i = 0; i < out_size; i++) {
      int32_t acc = 0;
      for (unsigned j = 0; j < in_size; j++) {
         int32_t iv = (int32_t)(int8_t)in[j] - in_zp;
         int32_t wv = (int32_t)weights[i * in_size + j] - w_zp;
         acc += iv * wv;
      }
      if (bias) acc += bias[i];

      float w_scale = per_channel ? op->sw.fc.w_scales[i] : op->sw.fc.w_scales[0];
      float m_scale = in_scale * w_scale / out_scale;
      int q = (int)roundf((float)acc * m_scale) + out_zp;
      if (q < -128) q = -128;
      if (q > 127) q = 127;
      out[i] = (uint8_t)(int8_t)q;
   }
}

void rnpu_execute_sw_op(struct rnpu_model *m, unsigned op_index)
{
   struct rnpu_operation *op = &m->ops[op_index];
   switch (op->type) {
   case RNPU_OP_CONCAT:         exec_concat(m, op); break;
   case RNPU_OP_MAX_POOL:       exec_max_pool(m, op); break;
   case RNPU_OP_PAD:            exec_pad(m, op); break;
   case RNPU_OP_RESIZE_NEAREST: exec_resize_nearest(m, op); break;
   case RNPU_OP_LOGISTIC:       exec_logistic(m, op); break;
   case RNPU_OP_AVG_POOL:       exec_avg_pool(m, op); break;
   case RNPU_OP_RESHAPE:        exec_reshape(m, op); break;
   case RNPU_OP_SOFTMAX:        exec_softmax(m, op); break;
   case RNPU_OP_FULLY_CONNECTED: exec_fully_connected(m, op); break;
   default: break;
   }
}

void rnpu_apply_per_axis_correction(struct rnpu_model *m,
                                    struct rnpu_operation *op)
{
   if (!op->per_axis_correction) return;
   unsigned w = op->output_width, h = op->output_height;
   unsigned oc = op->output_channels;
   int ozp = (int)(uint8_t)op->output_zero_point - 0x80;
   uint8_t *data = tensor_ptr(m, op->output_tensor);
   unsigned groups = DIV_ROUND_UP(oc, FEATURE_ATOMIC_SIZE);

   for (unsigned g = 0; g < groups; g++) {
      unsigned bc = g * FEATURE_ATOMIC_SIZE;
      unsigned rc = MIN2(FEATURE_ATOMIC_SIZE, oc - bc);
      unsigned go = g * h * w * FEATURE_ATOMIC_SIZE + op->per_channel_group_offset;
      for (unsigned y = 0; y < h; y++)
         for (unsigned x = 0; x < w; x++) {
            uint8_t *px = data + go + y * w * FEATURE_ATOMIC_SIZE + x * FEATURE_ATOMIC_SIZE;
            for (unsigned c = 0; c < rc; c++) {
               float corr = op->per_axis_correction[bc + c];
               if (corr == 1.0f) continue;
               int val = (int)(int8_t)px[c] - ozp;
               val = (int)roundf(val * corr) + ozp;
               if (val < -128) val = -128;
               if (val > 127) val = 127;
               px[c] = (uint8_t)(int8_t)val;
            }
         }
   }
}

void rnpu_scatter_requant_output(struct rnpu_model *m,
                                  struct rnpu_operation *op)
{
   if (op->requant_group_count <= 1) return;

   unsigned w = op->output_width, h = op->output_height;
   unsigned oc = op->output_channels;
   unsigned hw_groups = DIV_ROUND_UP(oc, FEATURE_ATOMIC_SIZE) * 2;
   unsigned full_out_per_group = w * h * hw_groups * FEATURE_ATOMIC_SIZE;
   unsigned px_count = w * h;
   unsigned gp = h * w * FEATURE_ATOMIC_SIZE; /* bytes per NPU channel group plane */
   uint8_t *data = tensor_ptr(m, op->output_tensor);
   /* Final output goes into group 0's position. Scatter from each requant
    * group's output copy, picking only that group's channels. */
   uint8_t *final_buf = malloc(full_out_per_group);
   memset(final_buf, 0, full_out_per_group);

   for (unsigned g = 0; g < op->requant_group_count; g++) {
      unsigned ch_off = op->requant_group_ch_offset[g];
      unsigned gsz = op->requant_group_sizes[g];
      uint8_t *group_out = data + g * full_out_per_group;

      for (unsigned ci = 0; ci < gsz; ci++) {
         unsigned orig_ch = op->group_channel_indices[ch_off + ci];
         unsigned src_g16 = orig_ch / FEATURE_ATOMIC_SIZE;
         unsigned src_c16 = orig_ch % FEATURE_ATOMIC_SIZE;
         for (unsigned p = 0; p < px_count; p++) {
            final_buf[src_g16 * gp + p * FEATURE_ATOMIC_SIZE + src_c16] =
               group_out[src_g16 * gp + p * FEATURE_ATOMIC_SIZE + src_c16];
         }
      }
   }

   memcpy(data, final_buf, full_out_per_group);
   free(final_buf);
}

void rnpu_apply_brdma_correction(struct rnpu_model *m,
                                  struct rnpu_operation *op)
{
   if (!op->use_brdma_per_channel || !op->per_channel_scales)
      return;

   unsigned w = op->output_width, h = op->output_height;
   unsigned oc = op->output_channels;
   int ozp = (int)(uint8_t)op->output_zero_point - 0x80;
   uint8_t *data = tensor_ptr(m, op->output_tensor);
   unsigned groups = DIV_ROUND_UP(oc, FEATURE_ATOMIC_SIZE);

   /* Build per-channel correction factors from MUL quantization residual */
   float *correction = malloc(oc * sizeof(float));

   if (op->requant_group_count > 1) {
      /* Multi-group: each group has its own reference scale and mul_shift */
      for (unsigned g = 0; g < op->requant_group_count; g++) {
         float ref_ws = op->requant_group_max_ws[g];
         unsigned ms = op->requant_mul_shifts[g];
         float md = (float)(1 << ms);
         unsigned ch_off = op->requant_group_ch_offset[g];
         unsigned gsz = op->requant_group_sizes[g];
         for (unsigned i = 0; i < gsz; i++) {
            unsigned orig_ch = op->group_channel_indices[ch_off + i];
            float exact = op->per_channel_scales[orig_ch] / ref_ws;
            float quantized = roundf(exact * md) / md;
            correction[orig_ch] = (quantized > 0) ? exact / quantized : 1.0f;
         }
      }
   } else {
      /* Single group: reference is global max weight scale */
      float ref_ws = op->weights_scale;
      unsigned ms = op->brdma_mul_shift;
      float md = (float)(1 << ms);
      for (unsigned c = 0; c < oc; c++) {
         float exact = op->per_channel_scales[c] / ref_ws;
         float quantized = roundf(exact * md) / md;
         correction[c] = (quantized > 0) ? exact / quantized : 1.0f;
      }
   }

   /* Apply corrections to NPU-format output */
   for (unsigned g = 0; g < groups; g++) {
      unsigned bc = g * FEATURE_ATOMIC_SIZE;
      unsigned rc = MIN2(FEATURE_ATOMIC_SIZE, oc - bc);
      for (unsigned y = 0; y < h; y++)
         for (unsigned x = 0; x < w; x++) {
            uint8_t *px = data + NPU_OFFSET(g, x, y, w, h);
            for (unsigned c = 0; c < rc; c++) {
               float corr = correction[bc + c];
               if (corr == 1.0f) continue;
               int val = (int)(int8_t)px[c] - ozp;
               val = (int)roundf(val * corr) + ozp;
               if (val < -128) val = -128;
               if (val > 127) val = 127;
               px[c] = (uint8_t)(int8_t)val;
            }
         }
   }

   free(correction);
}

void rnpu_compact_unsort_output(struct rnpu_model *m,
                                unsigned first_op,
                                unsigned num_ops)
{
   struct rnpu_operation *op0 = &m->ops[first_op];
   unsigned w = op0->output_width, h = op0->output_height;
   unsigned full_oc = op0->output_tensor_channels;
   unsigned num_groups = DIV_ROUND_UP(full_oc, FEATURE_ATOMIC_SIZE);
   unsigned gp = h * w * FEATURE_ATOMIC_SIZE;
   uint8_t *data = tensor_ptr(m, op0->output_tensor);

   /* Compact: 2x-spaced → contiguous */
   for (unsigned g = 1; g < num_ops; g++)
      memmove(data + g * gp, data + 2 * g * gp, gp);

   /* Un-sort: scatter from sorted to original order */
   unsigned compacted = num_ops * gp;
   uint8_t *copy = malloc(compacted);
   memcpy(copy, data, compacted);
   memset(data, 0, num_groups * gp);

   unsigned px_count = h * w;
   for (unsigned i = 0; i < num_ops; i++) {
      struct rnpu_operation *op = &m->ops[first_op + i];
      unsigned gc = op->output_channels;
      for (unsigned c = 0; c < gc; c++) {
         unsigned orig = op->group_channel_indices[c];
         unsigned sg = i, sc = c;
         unsigned dg = orig / FEATURE_ATOMIC_SIZE, dc = orig % FEATURE_ATOMIC_SIZE;
         for (unsigned p = 0; p < px_count; p++)
            data[dg * gp + p * FEATURE_ATOMIC_SIZE + dc] =
               copy[sg * gp + p * FEATURE_ATOMIC_SIZE + sc];
      }
   }
   free(copy);
}
