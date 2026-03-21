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

void rnpu_execute_sw_op(struct rnpu_model *m, unsigned op_index)
{
   struct rnpu_operation *op = &m->ops[op_index];
   switch (op->type) {
   case RNPU_OP_CONCAT:         exec_concat(m, op); break;
   case RNPU_OP_MAX_POOL:       exec_max_pool(m, op); break;
   case RNPU_OP_PAD:            exec_pad(m, op); break;
   case RNPU_OP_RESIZE_NEAREST: exec_resize_nearest(m, op); break;
   case RNPU_OP_LOGISTIC:       exec_logistic(m, op); break;
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
