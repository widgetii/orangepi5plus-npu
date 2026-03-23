/*
 * Model lifecycle: load, compile, invoke, free
 * Orchestrates TFLite parsing → op lowering → BO allocation → regcmd generation
 * SPDX-License-Identifier: MIT
 */

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <fcntl.h>
#include <unistd.h>

#include "../include/rocketnpu.h"
#include "rnpu_internal.h"
#include "rnpu_tflite.h"
#include "rnpu_coefs.h"
#include "rnpu_task.h"
#include "rnpu_regcmd.h"
#include "rnpu_convert.h"
#include "rnpu_sw_ops.h"

/* ---- Helpers ---- */

static bool tfl_is_depthwise(const struct rnpu_tfl_model *tfl,
                             const struct rnpu_tfl_op *op)
{
   if (op->builtin_code != TFLITE_OP_DEPTHWISE_CONV_2D) return false;
   unsigned ic = tfl->tensors[op->inputs[0]].shape[3];
   unsigned oc = tfl->tensors[op->outputs[0]].shape[3];
   return ic > 1 && oc > 1;
}

/* ---- Op lowering (TFLite → internal ops) ---- */

static void lower_conv(struct rnpu_model *m, const struct rnpu_tfl_op *top,
                        struct rnpu_operation *op)
{
   const struct rnpu_tfl_model *tfl = &m->tfl;
   const struct rnpu_tfl_tensor *it = &tfl->tensors[top->inputs[0]];
   const struct rnpu_tfl_tensor *ot = &tfl->tensors[top->outputs[0]];
   const struct rnpu_tfl_tensor *wt = &tfl->tensors[top->inputs[1]];

   op->type = RNPU_OP_CONV;
   op->depthwise = tfl_is_depthwise(tfl, top);
   op->stride = (top->builtin_code == TFLITE_OP_DEPTHWISE_CONV_2D) ?
                top->opt.dw_conv.stride_w : top->opt.conv.stride_w;
   op->padding_same = (top->builtin_code == TFLITE_OP_DEPTHWISE_CONV_2D) ?
                      (top->opt.dw_conv.padding == 0) : (top->opt.conv.padding == 0);
   op->add_tensor = -1;

   op->input_tensor = top->inputs[0];
   op->input_width = it->shape[1];
   op->input_height = it->shape[2];
   op->input_channels = it->shape[3];
   op->input_zero_point = (uint8_t)it->quant.zero_point;
   op->input_scale = it->quant.scale;

   op->output_tensor = top->outputs[0];
   op->output_width = ot->shape[1];
   op->output_height = ot->shape[2];
   op->output_channels = ot->shape[3];
   op->output_zero_point = (uint8_t)ot->quant.zero_point;
   op->output_scale = ot->quant.scale;

   op->weights_width = wt->shape[1];
   op->weights_height = wt->shape[2];
   op->weights_zero_point = (uint8_t)wt->quant.zero_point;

   /* Per-axis vs per-tensor weight scale */
   if (wt->quant.scales && wt->quant.num_scales > 1) {
      /* Per-axis: use max scale for the HW, corrections applied later */
      float max_s = wt->quant.scales[0];
      for (unsigned i = 1; i < wt->quant.num_scales; i++)
         if (wt->quant.scales[i] > max_s) max_s = wt->quant.scales[i];
      op->weights_scale = max_s;
   } else {
      op->weights_scale = wt->quant.scale;
   }
}

static void lower_sw_op(struct rnpu_model *m, const struct rnpu_tfl_op *top,
                         struct rnpu_operation *op, enum rnpu_op_type type)
{
   const struct rnpu_tfl_model *tfl = &m->tfl;
   const struct rnpu_tfl_tensor *it = &tfl->tensors[top->inputs[0]];
   const struct rnpu_tfl_tensor *ot = &tfl->tensors[top->outputs[0]];

   op->type = type;
   op->add_tensor = -1;
   op->input_tensor = top->inputs[0];
   op->input_width = it->shape[1];
   op->input_height = it->shape[2];
   op->input_channels = it->shape[3];
   op->input_zero_point = (uint8_t)it->quant.zero_point;
   op->input_scale = it->quant.scale;
   op->output_tensor = top->outputs[0];
   op->output_width = ot->shape[1];
   op->output_height = ot->shape[2];
   op->output_channels = ot->shape[3];
   op->output_zero_point = (uint8_t)ot->quant.zero_point;
   op->output_scale = ot->quant.scale;
}

static void lower_concat(struct rnpu_model *m, const struct rnpu_tfl_op *top,
                          struct rnpu_operation *op)
{
   lower_sw_op(m, top, op, RNPU_OP_CONCAT);
   unsigned n = top->input_count;
   op->sw.concat.input_count = n;
   op->sw.concat.input_tensors = calloc(n, sizeof(unsigned));
   op->sw.concat.input_channels_arr = calloc(n, sizeof(unsigned));
   for (unsigned i = 0; i < n; i++) {
      op->sw.concat.input_tensors[i] = top->inputs[i];
      op->sw.concat.input_channels_arr[i] = m->tfl.tensors[top->inputs[i]].shape[3];
   }
}

static void lower_max_pool(struct rnpu_model *m, const struct rnpu_tfl_op *top,
                            struct rnpu_operation *op)
{
   lower_sw_op(m, top, op, RNPU_OP_MAX_POOL);
   op->sw.pool.filter_width = top->opt.pool.filter_w;
   op->sw.pool.filter_height = top->opt.pool.filter_h;
   op->sw.pool.stride_x = top->opt.pool.stride_w;
   op->sw.pool.stride_y = top->opt.pool.stride_h;
   op->sw.pool.padding_same = (top->opt.pool.padding == 0);
}

static void lower_pad(struct rnpu_model *m, const struct rnpu_tfl_op *top,
                       struct rnpu_operation *op)
{
   lower_sw_op(m, top, op, RNPU_OP_PAD);
   /* Pad values from the padding tensor (inputs[1]) */
   if (top->input_count > 1) {
      int pad_idx = top->inputs[1];
      const struct rnpu_tfl_buffer *pb = &m->tfl.buffers[m->tfl.tensors[pad_idx].buffer_index];
      if (pb->data && pb->size >= 16) {
         const int32_t *pv = (const int32_t *)pb->data;
         /* TFLite pad tensor: [4][2] for NHWC → [batch][height][width][channel] */
         op->sw.pad.pad_before_w = pv[2]; /* height before */
         op->sw.pad.pad_after_w = pv[3];
         op->sw.pad.pad_before_h = pv[4]; /* width before */
         op->sw.pad.pad_after_h = pv[5];
      }
   }
}

static void lower_sw_op_auto(struct rnpu_model *m, const struct rnpu_tfl_op *top,
                              struct rnpu_operation *op, enum rnpu_op_type type)
{
   /* Like lower_sw_op but handles 2D tensors (e.g. RESHAPE/SOFTMAX output [1, 1001])
    * by treating them as 1×1×C in NPU format */
   const struct rnpu_tfl_model *tfl = &m->tfl;
   const struct rnpu_tfl_tensor *it = &tfl->tensors[top->inputs[0]];
   const struct rnpu_tfl_tensor *ot = &tfl->tensors[top->outputs[0]];

   op->type = type;
   op->add_tensor = -1;
   op->input_tensor = top->inputs[0];
   if (it->shape_len >= 4) {
      op->input_width = it->shape[1];
      op->input_height = it->shape[2];
      op->input_channels = it->shape[3];
   } else if (it->shape_len == 2) {
      op->input_width = 1;
      op->input_height = 1;
      op->input_channels = it->shape[1];
   }
   op->input_zero_point = (uint8_t)it->quant.zero_point;
   op->input_scale = it->quant.scale;

   op->output_tensor = top->outputs[0];
   if (ot->shape_len >= 4) {
      op->output_width = ot->shape[1];
      op->output_height = ot->shape[2];
      op->output_channels = ot->shape[3];
   } else if (ot->shape_len == 2) {
      op->output_width = 1;
      op->output_height = 1;
      op->output_channels = ot->shape[1];
   }
   op->output_zero_point = (uint8_t)ot->quant.zero_point;
   op->output_scale = ot->quant.scale;
}

static void lower_avg_pool(struct rnpu_model *m, const struct rnpu_tfl_op *top,
                             struct rnpu_operation *op)
{
   lower_sw_op(m, top, op, RNPU_OP_AVG_POOL);
   op->sw.pool.filter_width = top->opt.pool.filter_w;
   op->sw.pool.filter_height = top->opt.pool.filter_h;
   op->sw.pool.stride_x = top->opt.pool.stride_w;
   op->sw.pool.stride_y = top->opt.pool.stride_h;
   op->sw.pool.padding_same = (top->opt.pool.padding == 0);
}

static void lower_reshape(struct rnpu_model *m, const struct rnpu_tfl_op *top,
                            struct rnpu_operation *op)
{
   lower_sw_op_auto(m, top, op, RNPU_OP_RESHAPE);
}

static void lower_softmax(struct rnpu_model *m, const struct rnpu_tfl_op *top,
                            struct rnpu_operation *op)
{
   lower_sw_op_auto(m, top, op, RNPU_OP_SOFTMAX);
   const struct rnpu_tfl_tensor *it = &m->tfl.tensors[top->inputs[0]];
   const struct rnpu_tfl_tensor *ot = &m->tfl.tensors[top->outputs[0]];
   op->sw.softmax.in_scale = it->quant.scale;
   op->sw.softmax.in_zp = it->quant.zero_point;
   op->sw.softmax.out_scale = ot->quant.scale;
   op->sw.softmax.out_zp = ot->quant.zero_point;
}

static void lower_logistic(struct rnpu_model *m, const struct rnpu_tfl_op *top,
                            struct rnpu_operation *op)
{
   lower_sw_op(m, top, op, RNPU_OP_LOGISTIC);
   const struct rnpu_tfl_tensor *it = &m->tfl.tensors[top->inputs[0]];
   const struct rnpu_tfl_tensor *ot = &m->tfl.tensors[top->outputs[0]];
   float in_s = it->quant.scale, out_s = ot->quant.scale;
   int in_zp = it->quant.zero_point, out_zp = ot->quant.zero_point;

   for (int i = 0; i < 256; i++) {
      int tv = (int)(int8_t)((uint8_t)i + 0x80);
      float rv = (tv - in_zp) * in_s;
      float sig = 1.0f / (1.0f + expf(-rv));
      float ov = sig / out_s + out_zp;
      ov = fmaxf(-128.0f, fminf(127.0f, roundf(ov)));
      op->sw.logistic.lut[i] = (uint8_t)((int8_t)(int)ov - 0x80);
   }
   for (int i = 0; i < 256; i++) {
      int tv = (int)(int8_t)(uint8_t)i;
      float rv = (tv - in_zp) * in_s;
      float sig = 1.0f / (1.0f + expf(-rv));
      float ov = sig / out_s + out_zp;
      ov = fmaxf(-128.0f, fminf(127.0f, roundf(ov)));
      op->sw.logistic.raw_lut[i] = (uint8_t)(int8_t)(int)ov;
   }
}

/* ---- Per-axis group decomposition ---- */

static int cmp_scale_idx(const void *a, const void *b, void *ctx)
{
   const float *s = ctx;
   unsigned ia = *(const unsigned *)a, ib = *(const unsigned *)b;
   return (s[ia] > s[ib]) - (s[ia] < s[ib]);
}

static unsigned lower_conv_per_group(struct rnpu_model *m,
                                     const struct rnpu_tfl_op *top,
                                     struct rnpu_operation **ops_out,
                                     unsigned *op_count)
{
   const struct rnpu_tfl_tensor *wt = &m->tfl.tensors[top->inputs[1]];
   const struct rnpu_tfl_tensor *ot = &m->tfl.tensors[top->outputs[0]];
   unsigned full_oc = ot->shape[3];
   unsigned gs = PER_AXIS_GROUP_SIZE;
   unsigned ng = DIV_ROUND_UP(full_oc, gs);

   unsigned *sorted = malloc(full_oc * sizeof(unsigned));
   for (unsigned i = 0; i < full_oc; i++) sorted[i] = i;
   if (gs > 1)
      qsort_r(sorted, full_oc, sizeof(unsigned), cmp_scale_idx,
              (void *)wt->quant.scales);

   /* ops array is pre-allocated with enough space */
   unsigned base = *op_count;

   for (unsigned g = 0; g < ng; g++) {
      struct rnpu_operation *op = &(*ops_out)[base + g];
      memset(op, 0, sizeof(*op));
      lower_conv(m, top, op);

      unsigned gs_start = g * gs;
      unsigned gc = MIN2(gs, full_oc - gs_start);
      op->output_channels = gc;
      op->output_tensor_channels = full_oc;
      op->per_channel_group_offset =
         g * 2 * op->output_height * op->output_width * FEATURE_ATOMIC_SIZE;

      op->group_channel_indices = malloc(gc * sizeof(unsigned));
      for (unsigned i = 0; i < gc; i++)
         op->group_channel_indices[i] = sorted[gs_start + i];

      float gmax = wt->quant.scales[sorted[gs_start]];
      for (unsigned i = 1; i < gc; i++) {
         float s = wt->quant.scales[sorted[gs_start + i]];
         if (s > gmax) gmax = s;
      }
      op->weights_scale = gmax;

      if (gc > 1) {
         op->per_axis_correction = calloc(gc, sizeof(float));
         for (unsigned i = 0; i < gc; i++)
            op->per_axis_correction[i] = wt->quant.scales[sorted[gs_start + i]] / gmax;
      }

      if (gc == 1) {
         op->per_channel_bias = rnpu_compute_bias_scalar(&m->tfl, top, sorted[gs_start]);
      }
   }

   free(sorted);
   *op_count = base + ng;
   return ng;
}

/* ---- Model compilation ---- */

static void allocate_tensors(struct rnpu_model *m)
{
   /* Compute tensor lifetimes: [first_use, last_use] in op index space.
    * Tensors with non-overlapping lifetimes can share activation BO space. */
   unsigned *first_use = calloc(m->tensor_count, sizeof(unsigned));
   unsigned *last_use = calloc(m->tensor_count, sizeof(unsigned));
   for (unsigned i = 0; i < m->tensor_count; i++) {
      first_use[i] = UINT32_MAX;
      last_use[i] = 0;
   }
   for (unsigned i = 0; i < m->op_count; i++) {
      struct rnpu_operation *op = &m->ops[i];
      unsigned it = op->input_tensor;
      unsigned ot = op->output_tensor;
      if (first_use[it] > i) first_use[it] = i;
      if (last_use[it] < i) last_use[it] = i;
      if (first_use[ot] > i) first_use[ot] = i;
      if (last_use[ot] < i) last_use[ot] = i;
      /* Add tensor (element-wise addition input) */
      if (op->add_tensor >= 0 && op->add_tensor < (int)m->tensor_count) {
         unsigned at = op->add_tensor;
         if (first_use[at] > i) first_use[at] = i;
         if (last_use[at] < i) last_use[at] = i;
      }
      /* Concat additional inputs */
      if (op->type == RNPU_OP_CONCAT) {
         for (unsigned j = 0; j < op->sw.concat.input_count; j++) {
            unsigned ci = op->sw.concat.input_tensors[j];
            if (first_use[ci] > i) first_use[ci] = i;
            if (last_use[ci] < i) last_use[ci] = i;
         }
      }
   }

   /* Mark graph output tensors as live until the end */
   for (unsigned i = 0; i < m->tfl.output_count; i++) {
      unsigned ot = m->tfl.graph_outputs[i];
      last_use[ot] = m->op_count;
   }
   /* Mark graph input as live from the start */
   unsigned git = m->tfl.graph_inputs[0];
   first_use[git] = 0;

   /* Greedy offset assignment: for each tensor (in first_use order),
    * find the lowest offset where it doesn't overlap any live tensor. */
   unsigned *order = malloc(m->tensor_count * sizeof(unsigned));
   unsigned active_count = 0;
   for (unsigned i = 0; i < m->tensor_count; i++) {
      if (m->tensors[i].size == 0 || first_use[i] == UINT32_MAX) continue;
      order[active_count++] = i;
   }
   /* Sort by first_use */
   for (unsigned i = 0; i < active_count; i++)
      for (unsigned j = i + 1; j < active_count; j++)
         if (first_use[order[i]] > first_use[order[j]]) {
            unsigned tmp = order[i]; order[i] = order[j]; order[j] = tmp;
         }

   uint32_t total = 0;
   for (unsigned a = 0; a < active_count; a++) {
      unsigned ti = order[a];
      uint32_t sz = ALIGN_UP(m->tensors[ti].size, 64);

      /* Find lowest non-overlapping offset */
      uint32_t best_offset = 0;
      for (unsigned b = 0; b < a; b++) {
         unsigned bj = order[b];
         if (last_use[bj] < first_use[ti] || first_use[bj] > last_use[ti])
            continue; /* no overlap */
         uint32_t bend = m->tensors[bj].offset + ALIGN_UP(m->tensors[bj].size, 64);
         if (bend > best_offset) best_offset = bend;
      }
      m->tensors[ti].offset = best_offset;
      if (best_offset + sz > total) total = best_offset + sz;
   }

   free(order);
   free(first_use);
   free(last_use);

   /* Create activation BO */
   if (total > 0) {
      if (rnpu_bo_create(m->fd, total, &m->activation_bo) < 0) {
         fprintf(stderr, "rnpu: failed to create activation BO (%u bytes)\n", total);
      }
   }
}

static void compile_weights_and_biases(struct rnpu_model *m)
{
   /* First pass: compute total weight and bias sizes */
   uint32_t total_weight = 0, total_bias = 0;
   for (unsigned i = 0; i < m->op_count; i++) {
      struct rnpu_operation *op = &m->ops[i];
      if (op->type != RNPU_OP_CONV) continue;

      unsigned ws = rnpu_calc_weight_size(op->weights_width, op->weights_height,
                                          op->input_channels, op->output_channels,
                                          op->depthwise);
      op->weight_offset = total_weight;
      op->weight_size = ws;
      total_weight += ALIGN_UP(ws, 64);

      unsigned bs = MAX2(op->output_channels, WEIGHT_ATOMIC_SIZE) * sizeof(uint32_t);
      op->bias_offset = total_bias;
      op->bias_size = op->output_channels * sizeof(uint32_t);
      total_bias += ALIGN_UP(bs, 64);
   }

   /* Create BOs */
   if (total_weight > 0)
      rnpu_bo_create(m->fd, total_weight, &m->weight_bo);
   if (total_bias > 0)
      rnpu_bo_create(m->fd, total_bias, &m->bias_bo);

   /* Second pass: fill weight and bias data */
   for (unsigned i = 0; i < m->op_count; i++) {
      struct rnpu_operation *op = &m->ops[i];
      if (op->type != RNPU_OP_CONV) continue;

      /* Find the TFLite op for this operation */
      const struct rnpu_tfl_op *top = NULL;
      for (unsigned j = 0; j < m->tfl.op_count; j++) {
         if (m->tfl.ops[j].builtin_code == TFLITE_OP_CONV_2D ||
             m->tfl.ops[j].builtin_code == TFLITE_OP_DEPTHWISE_CONV_2D) {
            if (m->tfl.ops[j].outputs[0] == (int)op->output_tensor ||
                (op->output_tensor_channels > 0 &&
                 m->tfl.tensors[m->tfl.ops[j].outputs[0]].shape[3] ==
                 (int)op->output_tensor_channels)) {
               top = &m->tfl.ops[j];
               break;
            }
         }
      }
      if (!top) continue;

      uint8_t *wdst = (uint8_t *)m->weight_bo.map + op->weight_offset;
      if (op->group_channel_indices) {
         rnpu_fill_weights_group(&m->tfl, top, op->group_channel_indices,
                                 op->output_channels, wdst, op->weight_size);
      } else {
         rnpu_fill_weights(&m->tfl, top, wdst, op->weight_size);
      }

      if (op->bias_size > 0) {
         uint8_t *bdst = (uint8_t *)m->bias_bo.map + op->bias_offset;
         if (op->group_channel_indices) {
            rnpu_fill_biases_group(&m->tfl, top, op->group_channel_indices,
                                   op->output_channels, &op->truncate_bits,
                                   bdst, op->bias_size);
         } else {
            rnpu_fill_biases(&m->tfl, top, &op->truncate_bits, bdst, op->bias_size);
         }
      }
   }

   /* Flush weight and bias BOs to NPU */
   if (m->weight_bo.handle) rnpu_bo_fini(m->fd, &m->weight_bo);
   if (m->bias_bo.handle) rnpu_bo_fini(m->fd, &m->bias_bo);
}

static void compile_regcmds(struct rnpu_model *m)
{
   /* First pass: split tasks and estimate regcmd sizes */
   uint32_t total_regcmd = 0;
   for (unsigned i = 0; i < m->op_count; i++) {
      struct rnpu_operation *op = &m->ops[i];
      if (op->type != RNPU_OP_CONV) continue;
      rnpu_split_tasks(op);
      /* Each task generates ~110 uint64_t regs, padded to 64 bytes */
      unsigned per_task = ALIGN_UP(140 * sizeof(uint64_t), 64);
      op->regcmd_offset = total_regcmd;
      op->regcmd_size = per_task * op->task_count;
      total_regcmd += op->regcmd_size;
   }

   if (total_regcmd == 0) return;
   rnpu_bo_create(m->fd, total_regcmd, &m->regcmd_bo);

   /* Second pass: generate regcmds */
   for (unsigned i = 0; i < m->op_count; i++) {
      struct rnpu_operation *op = &m->ops[i];
      if (op->type != RNPU_OP_CONV) continue;

      uint8_t *base = (uint8_t *)m->regcmd_bo.map + op->regcmd_offset;
      uint32_t rc_offset = 0;

      for (unsigned t = 0; t < op->task_count; t++) {
         uint64_t *dst = (uint64_t *)(base + rc_offset);
         unsigned count = rnpu_fill_regcmd(m, op, dst, 120, t);
         unsigned size_bytes = count * sizeof(uint64_t);

         struct rnpu_split_task *task = &op->tasks[t];
         task->regcfg_amount = count;
         task->regcfg_addr = (uint32_t)(m->regcmd_bo.dma_addr +
                                        op->regcmd_offset + rc_offset);
         /* Chain pointers patched in second pass below */

         rc_offset += ALIGN_UP(size_bytes, 64);
      }

      /* Patch chain pointers between tasks */
      rc_offset = 0;
      for (unsigned t = 0; t < op->task_count - 1; t++) {
         struct rnpu_split_task *task = &op->tasks[t];
         struct rnpu_split_task *next = &op->tasks[t + 1];
         uint64_t *regs = (uint64_t *)(base + rc_offset);
         unsigned count = task->regcfg_amount;

         uint64_t *chain_addr = &regs[count - 4];
         uint64_t *chain_count = &regs[count - 3];

         *chain_addr |= (uint64_t)next->regcfg_addr << 16;

         unsigned regs_to_fetch = next->regcfg_amount - 4;
         regs_to_fetch = ALIGN_UP(regs_to_fetch / 2, 2);
         *chain_count |= (uint64_t)regs_to_fetch << 16;

         rc_offset += ALIGN_UP(count * sizeof(uint64_t), 64);
      }
   }



   rnpu_bo_fini(m->fd, &m->regcmd_bo);
}

static void build_execution_plan(struct rnpu_model *m)
{
   /* Count segments */
   unsigned ns = 0;
   bool prev_hw = false;
   for (unsigned i = 0; i < m->op_count; i++) {
      bool hw = (m->ops[i].type == RNPU_OP_CONV);
      if (hw) { if (!prev_hw) ns++; }
      else ns++;
      prev_hw = hw;
   }
   m->segments = calloc(ns, sizeof(struct rnpu_exec_segment));
   m->segment_count = ns;

   /* Count HW resources */
   unsigned total_hw_ops = 0, total_tasks = 0;
   for (unsigned i = 0; i < m->op_count; i++) {
      if (m->ops[i].type == RNPU_OP_CONV) {
         total_hw_ops++;
         total_tasks += m->ops[i].task_count;
      }
   }

   /* One job per CONV operation */
   if (total_hw_ops > 0) {
      m->jobs = calloc(total_hw_ops, sizeof(struct drm_rocket_job));
      m->hw_tasks = calloc(total_tasks, sizeof(struct drm_rocket_task));
      m->in_handles = calloc(total_hw_ops * 3, sizeof(uint32_t));
      m->out_handles = calloc(total_hw_ops, sizeof(uint32_t));
   }

   unsigned ji = 0, ti = 0;
   for (unsigned i = 0; i < m->op_count; i++) {
      if (m->ops[i].type != RNPU_OP_CONV) continue;
      struct rnpu_operation *op = &m->ops[i];
      unsigned first_task = ti;
      for (unsigned t = 0; t < op->task_count; t++) {
         m->hw_tasks[ti].regcmd = op->tasks[t].regcfg_addr;
         m->hw_tasks[ti].regcmd_count = op->tasks[t].regcfg_amount;
         ti++;
      }
      unsigned hi = ji * 3;
      m->in_handles[hi] = m->weight_bo.handle;
      m->in_handles[hi + 1] = m->regcmd_bo.handle;
      m->in_handles[hi + 2] = m->bias_bo.handle ? m->bias_bo.handle : m->weight_bo.handle;
      m->out_handles[ji] = m->activation_bo.handle;
      struct drm_rocket_job *job = &m->jobs[ji];
      job->task_struct_size = sizeof(struct drm_rocket_task);
      job->tasks = (uint64_t)(uintptr_t)&m->hw_tasks[first_task];
      job->task_count = op->task_count;
      job->in_bo_handles = (uint64_t)(uintptr_t)&m->in_handles[hi];
      job->in_bo_handle_count = 3;
      job->out_bo_handles = (uint64_t)(uintptr_t)&m->out_handles[ji];
      job->out_bo_handle_count = 1;
      ji++;
   }
   m->job_count = total_hw_ops;

   /* Build segments */
   unsigned si = 0;
   for (unsigned i = 0; i < m->op_count;) {
      if (m->ops[i].type == RNPU_OP_CONV) {
         struct rnpu_exec_segment *seg = &m->segments[si++];
         seg->is_hw = true;
         seg->first_op = i;
         seg->op_count = 0;
         while (i < m->op_count && m->ops[i].type == RNPU_OP_CONV) {
            seg->op_count++;
            i++;
         }
         seg->job_count = seg->op_count;
      } else {
         struct rnpu_exec_segment *seg = &m->segments[si++];
         seg->is_hw = false;
         seg->first_op = i;
         seg->op_count = 1;
         seg->job_count = 0;
         i++;
      }
   }

   /* Detect sw_only */
   m->sw_only = (total_hw_ops == 0);
}

/* ---- Public API implementation ---- */

int rnpu_open(const char *device)
{
   int fd;
   if (device) {
      fd = open(device, O_RDWR);
      if (fd < 0) {
         fprintf(stderr, "rnpu: cannot open %s\n", device);
         return -1;
      }
      if (strstr(device, "accel"))
         rnpu_active_driver = RNPU_DRIVER_ROCKET;
      else
         rnpu_active_driver = RNPU_DRIVER_RKNPU;
      goto found;
   }

   /* Auto-detect: try Rocket first, then scan DRM cards/render nodes for RKNPU */
   fd = open("/dev/accel/accel0", O_RDWR);
   if (fd >= 0) {
      rnpu_active_driver = RNPU_DRIVER_ROCKET;
      goto found;
   }
   for (int i = 0; i < 8; i++) {
      char sysfs[128], driver[64], devpath[32];
      /* Check both card and renderD nodes */
      int found_rknpu = 0;
      for (int pass = 0; pass < 2; pass++) {
         if (pass == 0) {
            snprintf(sysfs, sizeof(sysfs),
                     "/sys/class/drm/card%d/device/uevent", i);
            snprintf(devpath, sizeof(devpath), "/dev/dri/card%d", i);
         } else {
            snprintf(sysfs, sizeof(sysfs),
                     "/sys/class/drm/renderD%d/device/uevent", 128 + i);
            snprintf(devpath, sizeof(devpath), "/dev/dri/renderD%d", 128 + i);
         }
         FILE *uf = fopen(sysfs, "r");
         if (!uf) continue;
         int is_rknpu = 0;
         while (fgets(driver, sizeof(driver), uf)) {
            if (strncmp(driver, "DRIVER=RKNPU", 12) == 0 ||
                strncmp(driver, "DRIVER=rknpu", 12) == 0) {
               is_rknpu = 1;
               break;
            }
         }
         fclose(uf);
         if (!is_rknpu) continue;
         fd = open(devpath, O_RDWR);
         if (fd >= 0) {
            rnpu_active_driver = RNPU_DRIVER_RKNPU;
            fprintf(stderr, "rnpu: opened %s\n", devpath);
            found_rknpu = 1;
            break;
         }
      }
      if (found_rknpu) goto found;
   }
   fprintf(stderr, "rnpu: no NPU device found\n");
   return -1;

found:
   fprintf(stderr, "rnpu: using %s driver\n",
           rnpu_active_driver == RNPU_DRIVER_ROCKET ? "Rocket" : "RKNPU");
   return fd;
}

void rnpu_close(int fd)
{
   if (fd >= 0) close(fd);
}

rnpu_model_t *rnpu_model_load(int fd, const char *tflite_path)
{
   struct rnpu_model *m = calloc(1, sizeof(*m));
   m->fd = fd;

   /* Parse TFLite */
   if (rnpu_tflite_parse(tflite_path, &m->tfl) < 0) {
      free(m);
      return NULL;
   }

   fprintf(stderr, "rnpu: parsed %s: %u tensors, %u ops, %u inputs, %u outputs\n",
           tflite_path, m->tfl.tensor_count, m->tfl.op_count,
           m->tfl.input_count, m->tfl.output_count);

   /* Create tensor metadata */
   m->tensor_count = m->tfl.tensor_count;
   m->tensors = calloc(m->tensor_count, sizeof(struct rnpu_npu_tensor));
   for (unsigned i = 0; i < m->tensor_count; i++) {
      const struct rnpu_tfl_tensor *t = &m->tfl.tensors[i];
      if (t->shape_len >= 4) {
         m->tensors[i].width = t->shape[1];
         m->tensors[i].height = t->shape[2];
         m->tensors[i].channels = t->shape[3];
      } else if (t->shape_len == 2) {
         m->tensors[i].width = 1;
         m->tensors[i].height = 1;
         m->tensors[i].channels = t->shape[1];
      }
      m->tensors[i].scale = t->quant.scale;
      m->tensors[i].zero_point = t->quant.zero_point;
   }

   /* Pre-scan to compute total internal ops needed */
   unsigned total_ops = 0;
   for (unsigned i = 0; i < m->tfl.op_count; i++) {
      const struct rnpu_tfl_op *top = &m->tfl.ops[i];
      if (top->builtin_code == TFLITE_OP_CONV_2D ||
          top->builtin_code == TFLITE_OP_DEPTHWISE_CONV_2D) {
         const struct rnpu_tfl_tensor *wt = &m->tfl.tensors[top->inputs[1]];
         if (wt->quant.scales && wt->quant.num_scales > 1) {
            const struct rnpu_tfl_tensor *ot = &m->tfl.tensors[top->outputs[0]];
            total_ops += DIV_ROUND_UP(ot->shape[3], PER_AXIS_GROUP_SIZE);
            continue;
         }
      }
      total_ops++;
   }

   /* Lower TFLite ops to internal ops */
   m->op_count = 0;
   m->ops = calloc(total_ops, sizeof(struct rnpu_operation));

   for (unsigned i = 0; i < m->tfl.op_count; i++) {
      const struct rnpu_tfl_op *top = &m->tfl.ops[i];
      struct rnpu_operation *op = &m->ops[m->op_count];
      memset(op, 0, sizeof(*op));
      op->add_tensor = -1;

      switch (top->builtin_code) {
      case TFLITE_OP_CONV_2D:
      case TFLITE_OP_DEPTHWISE_CONV_2D: {
         const struct rnpu_tfl_tensor *wt = &m->tfl.tensors[top->inputs[1]];
         if (wt->quant.scales && wt->quant.num_scales > 1) {
            lower_conv_per_group(m, top, &m->ops, &m->op_count);
         } else {
            lower_conv(m, top, op);
            m->op_count++;
         }
         break;
      }
      case TFLITE_OP_ADD: {
         /* Fuse into preceding conv */
         /* Find producer of each input and fuse */
         if (m->op_count > 0) {
            struct rnpu_operation *prev = &m->ops[m->op_count - 1];
            if (prev->type == RNPU_OP_CONV) {
               prev->output_tensor = top->outputs[0];
               prev->addition_input = true;
               /* Find the other input (not produced by prev) */
               for (int j = 0; j < top->input_count; j++) {
                  if (top->inputs[j] != (int)prev->output_tensor) {
                     prev->add_tensor = top->inputs[j];
                     const struct rnpu_tfl_tensor *at = &m->tfl.tensors[top->inputs[j]];
                     prev->addition_offset = 0x80 - at->quant.zero_point;
                     prev->addition_scale = at->quant.scale;
                     break;
                  }
               }
            }
         }
         break;
      }
      case TFLITE_OP_CONCATENATION:
         lower_concat(m, top, op);
         m->op_count++;
         break;
      case TFLITE_OP_MAX_POOL_2D:
         lower_max_pool(m, top, op);
         m->op_count++;
         break;
      case TFLITE_OP_PAD:
         lower_pad(m, top, op);
         m->op_count++;
         break;
      case TFLITE_OP_RESIZE_NEAREST_NEIGHBOR:
         lower_sw_op(m, top, op, RNPU_OP_RESIZE_NEAREST);
         m->op_count++;
         break;
      case TFLITE_OP_LOGISTIC:
         lower_logistic(m, top, op);
         m->op_count++;
         break;
      case TFLITE_OP_AVERAGE_POOL_2D:
         lower_avg_pool(m, top, op);
         m->op_count++;
         break;
      case TFLITE_OP_RESHAPE:
         lower_reshape(m, top, op);
         m->op_count++;
         break;
      case TFLITE_OP_SOFTMAX:
         lower_softmax(m, top, op);
         m->op_count++;
         break;
      default:
         fprintf(stderr, "rnpu: unsupported op %d\n", top->builtin_code);
         break;
      }
   }

   fprintf(stderr, "rnpu: lowered to %u internal ops\n", m->op_count);

   /* Compute tensor sizes */
   for (unsigned i = 0; i < m->op_count; i++) {
      struct rnpu_operation *op = &m->ops[i];
      unsigned isz = rnpu_calc_npu_tensor_size(op->input_width, op->input_height,
                                                op->input_channels);
      if (m->tensors[op->input_tensor].size < isz)
         m->tensors[op->input_tensor].size = isz;

      unsigned osz = rnpu_calc_raw_output_size(op->output_width, op->output_height,
                                               op->output_channels,
                                               op->output_tensor_channels);
      if (m->tensors[op->output_tensor].size < osz)
         m->tensors[op->output_tensor].size = osz;

      /* Concat additional inputs */
      if (op->type == RNPU_OP_CONCAT) {
         for (unsigned j = 0; j < op->sw.concat.input_count; j++) {
            unsigned idx = op->sw.concat.input_tensors[j];
            unsigned sz = rnpu_calc_npu_tensor_size(op->input_width, op->input_height,
                                                    op->sw.concat.input_channels_arr[j]);
            if (m->tensors[idx].size < sz) m->tensors[idx].size = sz;
         }
      }
   }

   /* Allocate activation BO */
   allocate_tensors(m);

   /* Compile weights and biases into shared BOs */
   compile_weights_and_biases(m);

   /* Generate register commands */
   compile_regcmds(m);

   /* Build execution plan */
   build_execution_plan(m);

   /* Store graph I/O info */
   if (m->op_count > 0) {
      m->graph_input_tensor = m->ops[0].input_tensor;
   }

   /* For output tensors: if the TFLite graph output points to a tensor
    * we didn't allocate (because unsupported ops like SOFTMAX/RESHAPE
    * sit between the last CONV and the graph output), use the last
    * lowered op's output tensor instead. */
   m->output_count = m->tfl.output_count;
   m->graph_output_tensors = calloc(m->output_count, sizeof(unsigned));
   for (unsigned i = 0; i < m->output_count; i++) {
      unsigned ti = m->tfl.graph_outputs[i];
      if (ti < m->tensor_count && m->tensors[ti].size > 0) {
         m->graph_output_tensors[i] = ti;
      } else {
         /* Fallback: use last op's output */
         m->graph_output_tensors[i] = m->ops[m->op_count - 1].output_tensor;
         fprintf(stderr, "rnpu: output %u remapped to tensor %u (last op output)\n",
                 i, m->graph_output_tensors[i]);
      }
   }

   unsigned hw_ops = 0;
   for (unsigned i = 0; i < m->op_count; i++)
      if (m->ops[i].type == RNPU_OP_CONV) hw_ops++;
   fprintf(stderr, "rnpu: ready — %u HW ops (%u jobs), %u SW ops, %u BOs "
           "(W=%uK B=%uK R=%uK A=%uK)\n",
           hw_ops, m->job_count, m->op_count - hw_ops,
           (m->weight_bo.handle ? 1u : 0u) + (m->bias_bo.handle ? 1u : 0u) +
           (m->regcmd_bo.handle ? 1u : 0u) + (m->activation_bo.handle ? 1u : 0u),
           m->weight_bo.size / 1024, m->bias_bo.size / 1024,
           m->regcmd_bo.size / 1024, m->activation_bo.size / 1024);

   return m;
}

int rnpu_invoke(rnpu_model_t *m, const void *input, size_t input_size)
{
   /* Convert input to NPU format */
   struct rnpu_operation *first = &m->ops[0];
   uint8_t *act = (uint8_t *)m->activation_bo.map;

   if (m->sw_only) {
      unsigned total = first->input_width * first->input_height * first->input_channels;
      memcpy(act + m->tensors[m->graph_input_tensor].offset, input, total);
   } else {
      rnpu_convert_input(act + m->tensors[m->graph_input_tensor].offset,
                         input,
                         first->input_width, first->input_height,
                         first->input_channels, first->input_zero_point);
   }

   /* Flush input to NPU */
   rnpu_bo_fini(m->fd, &m->activation_bo);

   /* Execute segments — one job per CONV operation */
   unsigned hw_job_idx = 0;
   for (unsigned s = 0; s < m->segment_count; s++) {
      struct rnpu_exec_segment *seg = &m->segments[s];
      if (seg->is_hw) {
         /* Submit all jobs in this segment */
         int ret = rnpu_submit(m->fd, &m->jobs[hw_job_idx], seg->job_count);
         if (ret) {
            fprintf(stderr, "rnpu: segment submit failed (%u jobs)\n", seg->job_count);
            return ret;
         }

         /* Wait for completion */
         rnpu_bo_prep(m->fd, &m->activation_bo);

         /* Apply per-axis corrections */
         for (unsigned j = seg->first_op; j < seg->first_op + seg->op_count; j++)
            rnpu_apply_per_axis_correction(m, &m->ops[j]);

         /* Compact/unsort per-group outputs */
         for (unsigned j = seg->first_op; j < seg->first_op + seg->op_count; j++) {
            if (m->ops[j].output_tensor_channels == 0) continue;
            unsigned first_g = j;
            unsigned out_idx = m->ops[j].output_tensor;
            while (j + 1 < seg->first_op + seg->op_count &&
                   m->ops[j + 1].output_tensor == out_idx &&
                   m->ops[j + 1].output_tensor_channels > 0)
               j++;
            rnpu_compact_unsort_output(m, first_g, j - first_g + 1);
         }

         /* Flush corrections back if there are more segments */
         if (s + 1 < m->segment_count)
            rnpu_bo_fini(m->fd, &m->activation_bo);

         hw_job_idx += seg->job_count;
      } else {
         rnpu_execute_sw_op(m, seg->first_op);
         if (s + 1 < m->segment_count && m->segments[s+1].is_hw)
            rnpu_bo_fini(m->fd, &m->activation_bo);
      }
   }

   /* Final prep for output reading */
   rnpu_bo_prep(m->fd, &m->activation_bo);
   return 0;
}

int rnpu_get_output(rnpu_model_t *m, int idx, void *out, size_t max_size)
{
   if (idx < 0 || idx >= (int)m->output_count) return -1;
   unsigned ti = m->graph_output_tensors[idx];
   struct rnpu_npu_tensor *t = &m->tensors[ti];
   unsigned nhwc_size = t->width * t->height * t->channels;
   if (max_size < nhwc_size) return -1;

   uint8_t *npu = (uint8_t *)m->activation_bo.map + t->offset;

   if (m->sw_only) {
      memcpy(out, npu, nhwc_size);
   } else {
      rnpu_convert_output(out, npu, t->width, t->height, t->channels);
   }
   return nhwc_size;
}

int rnpu_get_tensor(rnpu_model_t *m, int tensor_idx, void *out, size_t max_size,
                    int *w, int *h, int *c)
{
   if (tensor_idx < 0 || tensor_idx >= (int)m->tensor_count) return -1;
   struct rnpu_npu_tensor *t = &m->tensors[tensor_idx];
   if (t->size == 0) return 0;
   *w = t->width; *h = t->height; *c = t->channels;
   unsigned nhwc_size = t->width * t->height * t->channels;
   if (max_size < nhwc_size) return -1;
   uint8_t *npu = (uint8_t *)m->activation_bo.map + t->offset;
   if (m->sw_only)
      memcpy(out, npu, nhwc_size);
   else
      rnpu_convert_output(out, npu, t->width, t->height, t->channels);
   return nhwc_size;
}

int rnpu_get_output_dims(rnpu_model_t *m, int idx, int *w, int *h, int *c)
{
   if (idx < 0 || idx >= (int)m->output_count) return -1;
   unsigned ti = m->graph_output_tensors[idx];
   *w = m->tensors[ti].width;
   *h = m->tensors[ti].height;
   *c = m->tensors[ti].channels;
   return 0;
}

int rnpu_get_input_dims(rnpu_model_t *m, int *w, int *h, int *c)
{
   struct rnpu_npu_tensor *t = &m->tensors[m->graph_input_tensor];
   *w = t->width;
   *h = t->height;
   *c = t->channels;
   return 0;
}

int rnpu_output_count(rnpu_model_t *m)
{
   return m->output_count;
}

void rnpu_model_free(rnpu_model_t *m)
{
   if (!m) return;

   /* Free operations */
   for (unsigned i = 0; i < m->op_count; i++) {
      struct rnpu_operation *op = &m->ops[i];
      free(op->per_axis_correction);
      free(op->group_channel_indices);
      free(op->tasks);
      if (op->type == RNPU_OP_CONCAT) {
         free(op->sw.concat.input_tensors);
         free(op->sw.concat.input_channels_arr);
      }
   }
   free(m->ops);
   free(m->tensors);
   free(m->segments);
   free(m->jobs);
   free(m->hw_tasks);
   free(m->in_handles);
   free(m->out_handles);
   free(m->graph_output_tensors);

   /* Wait for all pending NPU work on every BO before destroying.
    * Without this, GEM_CLOSE races with the DRM scheduler's fence
    * tracking, corrupting IOMMU state on the next NPU use. */
   rnpu_bo_prep(m->fd, &m->activation_bo);
   rnpu_bo_prep(m->fd, &m->weight_bo);
   rnpu_bo_prep(m->fd, &m->bias_bo);
   rnpu_bo_prep(m->fd, &m->regcmd_bo);

   /* Destroy BOs */
   rnpu_bo_destroy(m->fd, &m->weight_bo);
   rnpu_bo_destroy(m->fd, &m->bias_bo);
   rnpu_bo_destroy(m->fd, &m->regcmd_bo);
   rnpu_bo_destroy(m->fd, &m->activation_bo);

   /* Free TFLite model */
   rnpu_tflite_free(&m->tfl);

   free(m);
}
