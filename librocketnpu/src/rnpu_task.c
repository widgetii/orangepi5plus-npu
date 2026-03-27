/*
 * CBUF task splitting — extracted from Mesa rkt_task.c
 * SPDX-License-Identifier: MIT
 */

#include <stdio.h>
#include "rnpu_task.h"

static unsigned calc_entries_per_slice(struct rnpu_operation *op)
{
   unsigned bpe = sizeof(uint8_t);
   unsigned atomics_per_entry = CBUF_ENTRY_SIZE / FEATURE_ATOMIC_SIZE;
   unsigned total_c = DIV_ROUND_UP(op->input_channels * bpe, FEATURE_ATOMIC_SIZE);
   unsigned last_c = total_c % atomics_per_entry;
   unsigned int_c = (total_c / atomics_per_entry) * op->input_width;
   unsigned frac_c = (last_c == 3) ? op->input_width
      : DIV_ROUND_UP(last_c * op->input_width, atomics_per_entry);
   return int_c + frac_c;
}

static unsigned calc_input_banks(struct rnpu_operation *op)
{
   return DIV_ROUND_UP(calc_entries_per_slice(op) * op->input_height,
                       CBUF_ENTRIES_PER_BANK);
}

static unsigned calc_weights_banks(struct rnpu_operation *op)
{
   unsigned bytes = op->weights_width * op->weights_height *
                    op->input_channels * sizeof(uint8_t);
   if (!op->depthwise) bytes *= op->output_channels;
   unsigned banks = DIV_ROUND_UP(DIV_ROUND_UP(bytes, CBUF_ENTRY_SIZE),
                                 CBUF_ENTRIES_PER_BANK);
   return banks + 1;
}

static unsigned calc_line_stride(unsigned width)
{
   return width * ATOMIC_K_SIZE * sizeof(uint8_t);
}

static void calc_explicit_padding(const struct rnpu_operation *op,
                                  unsigned *pt, unsigned *pb,
                                  unsigned *pl, unsigned *pr)
{
   if (op->padding_same && op->weights_width > 1) {
      unsigned pw = MAX2((op->output_width - 1) * op->stride +
                         op->weights_width - op->input_width, 0);
      unsigned ph = MAX2((op->output_height - 1) * op->stride +
                         op->weights_height - op->input_height, 0);
      *pl = ph / 2;
      *pr = ph - *pl;
      *pt = pw / 2;
      *pb = pw - *pt;
   } else {
      *pl = *pr = *pt = *pb = 0;
   }
}

static void fill_task(struct rnpu_operation *op, struct rnpu_split_task *task)
{
   task->stride_x = op->stride;
   task->stride_y = op->stride;
   task->input_width = op->input_width;
   if (task->input_width == 8 && (op->addition_input || op->add_tensor != -1))
      task->input_width *= 2;
   task->input_height = op->input_height;
   task->input_channels = ALIGN_UP(MAX2(op->input_channels, FEATURE_ATOMIC_SIZE),
                                   FEATURE_ATOMIC_SIZE);
   task->input_channels_real = op->input_channels;
   task->input_zero_point = op->input_zero_point;
   task->input_scale = op->input_scale;
   task->output_width = op->output_width;
   task->output_height = op->output_height;
   task->output_channels_real = op->output_channels;

   task->output_channels = ALIGN_UP(MAX2(op->output_channels, 32), 32);
   if (op->depthwise) {
      if (task->output_channels_real <= 32)
         task->output_channels *= 2;
      task->output_channels = ALIGN_UP(task->output_channels, 64);
   }

   task->output_zero_point = op->output_zero_point;
   task->output_int8 = op->output_int8;
   task->output_scale = op->output_scale;

   if (task->input_channels_real == 1 &&
       (task->output_channels_real > 1 ||
        op->addition_input || op->add_tensor != -1)) {
      task->input_width = MAX2(task->input_width, FEATURE_ATOMIC_SIZE);
      task->input_line_stride =
         MAX2(calc_line_stride(op->input_width) / FEATURE_ATOMIC_SIZE,
              FEATURE_ATOMIC_SIZE);
      if (op->input_channels == 32 && op->input_width == 80) {
         task->input_line_stride *= 4;
         task->input_surface_stride = (float)task->input_line_stride *
                                      (((float)task->input_height / 4) - 1);
      } else {
         task->input_surface_stride =
            (float)task->input_line_stride * (((float)task->input_height) - 1);
      }
   } else {
      task->input_line_stride = calc_line_stride(op->input_width) / 4;
      task->input_surface_stride =
         (float)task->input_line_stride * (((float)task->input_height / 4) - 1);
   }

   if (task->input_width == 8 && (op->addition_input || op->add_tensor != -1)) {
      task->input_line_stride /= 2;
      task->input_surface_stride = 112;
   }

   int ols = calc_line_stride(op->output_width);
   task->output_surface_stride = ols * task->output_height / FEATURE_ATOMIC_SIZE;

   if (task->input_channels_real == 1)
      task->input_data_entries = task->input_width * task->input_height;
   else if (task->input_width == 40 && task->input_channels_real == 40)
      task->input_data_entries = 40;
   else
      task->input_data_entries = DIV_ROUND_UP(
         task->input_width * 2 *
         DIV_ROUND_UP(task->input_channels_real, FEATURE_ATOMIC_SIZE), 8);

   task->weights_width = op->weights_width;
   task->weights_height = op->weights_height;
   task->weights_zero_point = op->weights_zero_point;
   task->weights_int8 = op->weights_int8;
   task->weights_scale = op->weights_scale;

   if (op->depthwise)
      task->weights_kernels = 1;
   else if (op->output_channels == 1 && op->output_tensor_channels > 0)
      task->weights_kernels = 2;
   else
      task->weights_kernels = ALIGN_UP(MAX2(op->output_channels, 32), 2);

   task->surfaces_per_row = task->output_width * task->output_height * 2;
   if (op->depthwise) task->surfaces_per_row *= 2;
}

static void replicate_tasks_for_requant(struct rnpu_operation *op,
                                        unsigned spatial_count)
{
   unsigned ng = op->requant_group_count;
   if (ng <= 1) return;

   unsigned total = spatial_count * ng;
   struct rnpu_split_task *new_tasks = calloc(total, sizeof(struct rnpu_split_task));

   for (unsigned g = 0; g < ng; g++) {
      for (unsigned s = 0; s < spatial_count; s++) {
         unsigned idx = g * spatial_count + s;
         new_tasks[idx] = op->tasks[s]; /* copy spatial task */
         new_tasks[idx].num = idx;
         new_tasks[idx].weights_scale = op->requant_group_max_ws[g];
         new_tasks[idx].requant_group_idx = g;
         new_tasks[idx].brdma_group_offset = op->requant_brdma_offsets[g];
         /* output_offset for this requant group: shifted by group_idx * full_output_size */
         unsigned full_out_per_group = op->output_width * op->output_height *
            DIV_ROUND_UP(op->output_channels, FEATURE_ATOMIC_SIZE) * 2 * FEATURE_ATOMIC_SIZE;
         new_tasks[idx].output_offset = op->tasks[s].output_offset +
                                         g * full_out_per_group;
      }
   }

   free(op->tasks);
   op->tasks = new_tasks;
   op->task_count = total;
}

void rnpu_split_tasks(struct rnpu_operation *op)
{
   unsigned entries_per_slice = calc_entries_per_slice(op);
   unsigned input_banks_req = calc_input_banks(op);
   unsigned weights_banks_req = calc_weights_banks(op);
   unsigned avail_wb = weights_banks_req;
   unsigned avail_ib = CBUF_BANKS - weights_banks_req;
   unsigned pt, pb, pl, pr;
   calc_explicit_padding(op, &pt, &pb, &pl, &pr);

   if (weights_banks_req + 1 < CBUF_BANKS)
      op->reuse_weights_cbuf = true;
   else {
      op->reuse_weights_cbuf = false;
      avail_ib = 7;
      avail_wb = CBUF_BANKS - avail_ib;
   }

   if (input_banks_req <= avail_ib) {
      op->task_count = 1;
      op->tasks = calloc(1, sizeof(struct rnpu_split_task));
      struct rnpu_split_task *t = &op->tasks[0];
      t->num = 0;
      fill_task(op, t);
      t->input_banks = input_banks_req;
      t->weights_banks = CBUF_BANKS - t->input_banks;
      t->input_height = op->input_height;
      t->pad_top = pt; t->pad_bottom = pb;
      t->pad_left = pl; t->pad_right = pr;
      t->atomic_count = t->output_width * t->output_height;

      replicate_tasks_for_requant(op, 1);
      return;
   }

   /* Multiple tasks needed.
    * max_tasks estimate: ceil(ih / effective_stride_per_task) + safety.
    * For strided convs, effective output rows per task = avail_slices / stride,
    * which means more tasks are needed. Use generous safety margin. */
   unsigned avail_slices = (CBUF_ENTRIES_PER_BANK * avail_ib) / entries_per_slice;
   if (avail_slices == 0) avail_slices = 1;
   unsigned effective_advance = (avail_slices > op->weights_height) ?
      avail_slices - op->weights_height + op->stride : op->stride;
   unsigned max_tasks = DIV_ROUND_UP(op->input_height, effective_advance) + 4;
   op->tasks = calloc(max_tasks, sizeof(struct rnpu_split_task));
   op->task_count = 0;

   struct rnpu_split_task *t = &op->tasks[op->task_count++];
   t->num = 0;
   fill_task(op, t);
   t->input_banks = avail_ib;
   t->weights_banks = avail_wb;
   t->top_slice = 0;
   t->bottom_slice = avail_slices - 1;
   t->pad_top = pt; t->pad_left = pl; t->pad_right = pr;

   for (unsigned slice = op->weights_height - pt - 1; slice < op->input_height;) {
      struct rnpu_split_task *prev = &op->tasks[op->task_count - 1];
      while (slice <= prev->bottom_slice) slice += op->stride;
      if (slice > prev->bottom_slice) slice -= op->stride;

      if (op->task_count >= max_tasks) {
         fprintf(stderr, "rnpu: task split overflow: task_count=%u >= max=%u "
                 "(iw=%u ih=%u ic=%u oc=%u stride=%u eps=%u avail_sl=%u)\n",
                 op->task_count, max_tasks, op->input_width, op->input_height,
                 op->input_channels, op->output_channels, op->stride,
                 entries_per_slice, avail_slices);
         break;
      }
      t = &op->tasks[op->task_count++];
      memset(t, 0, sizeof(*t));
      t->num = op->task_count - 1;
      fill_task(op, t);
      t->top_slice = MIN2(slice, prev->bottom_slice) -
                     (op->weights_height - 1) + op->stride;
      t->bottom_slice = t->top_slice + avail_slices - 1;
      t->pad_left = pl; t->pad_right = pr;
      t->input_banks = avail_ib;
      t->weights_banks = avail_wb;

      if (t->bottom_slice >= op->input_height - 1) {
         t->bottom_slice = op->input_height - 1;
         t->pad_bottom = pb;
         break;
      }
      slice = t->top_slice + op->weights_height - 1;
   }

   /* Trim last task if it's out of range or too small for the kernel.
    * When trimming, extend the previous task to cover the remaining rows. */
   while (op->task_count > 1) {
      struct rnpu_split_task *last = &op->tasks[op->task_count - 1];
      unsigned last_ih = last->bottom_slice - last->top_slice + 1;
      if (last->top_slice >= op->input_height ||
          last->bottom_slice >= op->input_height + pb ||
          last_ih + last->pad_top + last->pad_bottom < op->weights_height) {
         /* Extend previous task to cover these rows */
         if (op->task_count >= 2) {
            struct rnpu_split_task *prev = &op->tasks[op->task_count - 2];
            prev->bottom_slice = MIN2(last->bottom_slice, op->input_height - 1);
            prev->pad_bottom = last->pad_bottom;
         }
         op->task_count--;
      } else
         break;
   }

   /* Overlap slices */
   for (unsigned i = 1; i < op->task_count; i++) {
      struct rnpu_split_task *prev = &op->tasks[i - 1];
      struct rnpu_split_task *cur = &op->tasks[i];
      if (prev->bottom_slice >= cur->top_slice) {
         cur->num_overlap_slices = prev->bottom_slice - cur->top_slice + 1;
         prev->num_retain_slices = cur->num_overlap_slices;
      }
   }

   /* Per-task dimensions */
   unsigned output_h_done = 0;
   for (unsigned i = 0; i < op->task_count; i++) {
      t = &op->tasks[i];

      unsigned s = t->top_slice + (op->weights_height - 1) - t->pad_top;
      t->convolutions = 0;
      while (s <= t->bottom_slice + t->pad_bottom) {
         s += op->stride;
         t->convolutions++;
      }
      t->bottom_slice = MIN2(t->bottom_slice, op->input_height - 1);
      t->input_height = t->bottom_slice - t->top_slice + 1;
      t->output_width = (t->input_width + t->pad_left + t->pad_right -
                         op->weights_width) / op->stride + 1;
      t->output_height = (t->input_height + t->pad_top + t->pad_bottom -
                          op->weights_height) / op->stride + 1;
      t->atomic_count = t->output_width * t->output_height;
      t->input_offset = calc_line_stride(op->input_width) * t->top_slice;
      t->output_offset = calc_line_stride(op->output_width) * output_h_done;
      output_h_done += t->output_height;
   }

   unsigned spatial_count = op->task_count;
   replicate_tasks_for_requant(op, spatial_count);
}
