/*
 * Copyright (c) 2024 Tomeu Vizoso <tomeu@tomeuvizoso.net>
 * SPDX-License-Identifier: MIT
 */

#include "pipe/p_state.h"
#include "util/macros.h"
#include "util/u_dynarray.h"
#include "util/u_inlines.h"

#include <xf86drm.h>
#include <math.h>

#include "drm-uapi/rocket_accel.h"

#include "rkt_coefs.h"
#include "rkt_ml.h"
#include "rkt_regcmd.h"
#include "rkt_task.h"

#ifdef __aarch64__
#include <arm_neon.h>
#endif

/* NPU tensor interleaved format offset (x-major, matching input conversion).
 * w = dims[1], h = dims[2], x ranges over w, y ranges over h.
 * Note: NPU HW output is y-major; read_outputs handles this. SW ops use
 * x-major for NPU-format tensors. For sw_only subgraphs, flat NHWC is used
 * instead (no interleaving, no NPU_OFFSET).
 */
#define NPU_OFFSET(g, x, y, w, h) \
   ((g) * (w) * (h) * FEATURE_ATOMIC_SIZE + \
    (x) * (h) * FEATURE_ATOMIC_SIZE + \
    (y) * FEATURE_ATOMIC_SIZE)

void
rkt_dump_buffer(const uint8_t *ptr, char *name, int operation_nr,
                int suboperation_nr, int offset, unsigned size)
{
   char buffer[255];

   snprintf(buffer, sizeof(buffer), "mesa-%s-%03u-%03u.bin", name, operation_nr,
            suboperation_nr);

   FILE *f = fopen(buffer, "wb");
   assert(f);
   fwrite(ptr + offset, 1, size, f);
   if (ferror(f)) {
      DBG("Error in writing to file: %s\n", strerror(errno));
   }
   fflush(f);
   fclose(f);
}

static void
create_tensor(struct rkt_ml_subgraph *subgraph, unsigned idx,
              unsigned size)
{
   struct pipe_context *context = subgraph->base.context;
   struct pipe_resource **tensors = util_dynarray_begin(&subgraph->tensors);

   assert(idx < util_dynarray_num_elements(&subgraph->tensors,
                                           struct pipe_resource *));

   struct pipe_resource *res = tensors[idx];

   if (res != NULL) {
      /* Per-group ops may request smaller size than the full tensor */
      assert(size <= pipe_buffer_size(res));
      return;
   }

   res = pipe_buffer_create(context->screen, 0, PIPE_USAGE_DEFAULT, size);
   tensors[idx] = res;
}

struct rkt_resource *
rkt_get_tensor(struct rkt_ml_subgraph *subgraph,
               unsigned idx)
{
   return rkt_resource(
      *util_dynarray_element(&subgraph->tensors, struct pipe_resource *, idx));
}

bool
rkt_is_depthwise(const struct pipe_ml_operation *poperation)
{
   unsigned input_channels = poperation->input_tensors[0]->dims[3];
   unsigned output_channels = poperation->output_tensors[0]->dims[3];

   return poperation->conv.depthwise && input_channels > 1 &&
          output_channels > 1;
}

static unsigned
calc_raw_output_size(struct rkt_operation *operation)
{
   /* For per-group ops, use the full channel count for tensor sizing */
   unsigned oc = operation->output_tensor_channels > 0
      ? operation->output_tensor_channels : operation->output_channels;
   unsigned output_channels_1 =
      DIV_ROUND_UP(oc, FEATURE_ATOMIC_SIZE) * 2;
   unsigned output_channels_2 = FEATURE_ATOMIC_SIZE;

   return operation->output_width * operation->output_height *
          output_channels_1 * output_channels_2;
}

static unsigned
calc_npu_tensor_size(unsigned width, unsigned height, unsigned channels)
{
   unsigned groups = DIV_ROUND_UP(channels, FEATURE_ATOMIC_SIZE) * 2;
   return width * height * groups * FEATURE_ATOMIC_SIZE;
}

static void
compile_operation(struct rkt_ml_subgraph *subgraph,
                  struct rkt_operation *operation)
{
   struct pipe_context *pcontext = subgraph->base.context;
   unsigned regcfg_total_size = 0;
   struct util_dynarray *regcfgs;
   struct pipe_transfer *transfer = NULL;
   unsigned num_tasks =
      util_dynarray_num_elements(&operation->tasks, struct split_task);

   regcfgs = calloc(num_tasks, sizeof(struct util_dynarray));

   for (int i = 0; i < num_tasks; i++) {
      regcfgs[i] = UTIL_DYNARRAY_INIT;
      rkt_fill_regcmd(subgraph, operation, &regcfgs[i], i);

      unsigned size =
         util_dynarray_num_elements(&regcfgs[i], uint64_t) * sizeof(uint64_t);
      regcfg_total_size += align(size, 64);
   }

   operation->regcmd = pipe_buffer_create(pcontext->screen, 0,
                                          PIPE_USAGE_DEFAULT, regcfg_total_size);
   uint8_t *regcmd =
      pipe_buffer_map(pcontext, operation->regcmd, PIPE_MAP_WRITE, &transfer);

   unsigned regcmd_offset = 0;
   for (int i = 0; i < num_tasks; i++) {
      unsigned size = util_dynarray_num_elements(&regcfgs[i], uint64_t);
      struct split_task *task =
         util_dynarray_element(&operation->tasks, struct split_task, i);

      if (i < num_tasks - 1) {
         /* Patch next address and amount of regs to fetch, positions are relative
          * to end */
         unsigned reg_count = util_dynarray_num_elements(&regcfgs[i], uint64_t);
         uint64_t *next_address_reg =
            util_dynarray_element(&regcfgs[i], uint64_t, reg_count - 4);
         uint64_t *reg_count_reg =
            util_dynarray_element(&regcfgs[i], uint64_t, reg_count - 3);

         uint64_t addr = rkt_resource(operation->regcmd)->phys_addr +
                         regcmd_offset + align(size * sizeof(uint64_t), 64);
         *next_address_reg |= addr << 16;

         unsigned regs_to_fetch =
            util_dynarray_num_elements(&regcfgs[i + 1], uint64_t);
         regs_to_fetch -= 4;
         regs_to_fetch = align(regs_to_fetch / 2, 2);
         *reg_count_reg |= regs_to_fetch << 16;
      }

      memcpy(regcmd + regcmd_offset, util_dynarray_begin(&regcfgs[i]),
             size * sizeof(uint64_t));
      util_dynarray_fini(&regcfgs[i]);

      task->regcfg_amount = size;
      task->regcfg_addr =
         rkt_resource(operation->regcmd)->phys_addr + regcmd_offset;

      if (DBG_ENABLED(ROCKET_DBG_DUMP_BOS))
         rkt_dump_buffer(regcmd, "regcmd", 0, i, regcmd_offset,
                         (size + 4) * sizeof(uint64_t));

      regcmd_offset += align(size * sizeof(uint64_t), 64);
   }

   pipe_buffer_unmap(pcontext, transfer);

   for (int i = 0; i < num_tasks; i++) {
      util_dynarray_fini(&regcfgs[i]);
   }

   free(regcfgs);

}

/*
 * Chain the last task of operation N to the first task of operation N+1.
 * This allows hardware task chaining: the NPU's PC unit follows the
 * regcmd chain without CPU intervention between tasks.
 */
static void
chain_operations(struct rkt_ml_subgraph *subgraph)
{
   struct pipe_context *pcontext = subgraph->base.context;
   struct rkt_operation *ops = util_dynarray_begin(&subgraph->operations);
   unsigned num_ops = util_dynarray_num_elements(&subgraph->operations,
                                                  struct rkt_operation);

   for (unsigned i = 0; i < num_ops - 1; i++) {
      struct rkt_operation *op = &ops[i];
      struct rkt_operation *next_op = &ops[i + 1];

      unsigned num_tasks = util_dynarray_num_elements(&op->tasks, struct split_task);
      struct split_task *last_task =
         util_dynarray_element(&op->tasks, struct split_task, num_tasks - 1);
      struct split_task *first_task_next =
         util_dynarray_element(&next_op->tasks, struct split_task, 0);

      /* Map the regcmd BO to patch the chain pointers */
      struct pipe_transfer *transfer;
      uint8_t *regcmd = pipe_buffer_map(pcontext, op->regcmd,
                                        PIPE_MAP_WRITE, &transfer);

      /* Find offset of last task within the regcmd buffer */
      uint64_t bo_phys = rkt_resource(op->regcmd)->phys_addr;
      unsigned last_task_offset = last_task->regcfg_addr - bo_phys;

      /* The chain fields are at positions (regcfg_amount - 4) and (regcfg_amount - 3)
       * from the start of the task's regcmd data */
      uint64_t *chain_base_addr = (uint64_t *)(regcmd + last_task_offset +
                                   (last_task->regcfg_amount - 4) * sizeof(uint64_t));
      uint64_t *chain_reg_count = (uint64_t *)(regcmd + last_task_offset +
                                   (last_task->regcfg_amount - 3) * sizeof(uint64_t));

      /* Set next task address */
      uint64_t next_addr = first_task_next->regcfg_addr;
      *chain_base_addr |= next_addr << 16;

      /* Set register count for next task */
      unsigned regs_to_fetch = first_task_next->regcfg_amount;
      regs_to_fetch -= 4;
      regs_to_fetch = align(regs_to_fetch / 2, 2);
      *chain_reg_count |= regs_to_fetch << 16;

      pipe_buffer_unmap(pcontext, transfer);
   }
}

/*
 * Build execution plan with segments for mixed HW/SW execution.
 * Consecutive HW (convolution) ops are grouped into a single submit.
 * Each SW op is its own segment executed on the CPU.
 */
static void
build_execution_plan(struct rkt_ml_subgraph *subgraph)
{
   struct rkt_operation *ops = util_dynarray_begin(&subgraph->operations);
   unsigned num_ops = util_dynarray_num_elements(&subgraph->operations,
                                                  struct rkt_operation);

   /* Count segments */
   unsigned num_segments = 0;
   bool prev_hw = false;
   for (unsigned i = 0; i < num_ops; i++) {
      bool is_hw = (ops[i].type == RKT_OP_CONVOLUTION);
      if (is_hw) {
         if (!prev_hw) num_segments++;
      } else {
         num_segments++;
      }
      prev_hw = is_hw;
   }

   subgraph->exec_segments = calloc(num_segments, sizeof(struct rkt_exec_segment));
   subgraph->exec_segment_count = num_segments;

   /* Count total HW ops and tasks */
   unsigned total_hw_ops = 0;
   unsigned total_tasks = 0;
   for (unsigned i = 0; i < num_ops; i++) {
      if (ops[i].type == RKT_OP_CONVOLUTION) {
         total_hw_ops++;
         total_tasks += util_dynarray_num_elements(&ops[i].tasks, struct split_task);
      }
   }

   /* Allocate HW job arrays */
   if (total_hw_ops > 0) {
      subgraph->cached_jobs = calloc(total_hw_ops, sizeof(struct drm_rocket_job));
      subgraph->cached_tasks = calloc(total_tasks, sizeof(struct drm_rocket_task));
      subgraph->cached_in_handles = calloc(total_hw_ops, sizeof(uint32_t *));
      subgraph->cached_out_handles_arr = calloc(total_hw_ops, sizeof(uint32_t *));
   }
   subgraph->cached_job_count = total_hw_ops;
   subgraph->cached_task_count = total_tasks;

   /* Fill tasks and jobs for HW ops */
   unsigned hw_op_idx = 0;
   unsigned task_offset = 0;
   for (unsigned i = 0; i < num_ops; i++) {
      if (ops[i].type != RKT_OP_CONVOLUTION) continue;

      struct rkt_operation *op = &ops[i];
      unsigned num_tasks = util_dynarray_num_elements(&op->tasks, struct split_task);

      util_dynarray_foreach(&op->tasks, struct split_task, task) {
         subgraph->cached_tasks[task_offset].regcmd = task->regcfg_addr;
         subgraph->cached_tasks[task_offset].regcmd_count = task->regcfg_amount;
         task_offset++;
      }

      unsigned num_inputs = op->add_tensor != -1 ? 2 : 1;
      subgraph->cached_in_handles[hw_op_idx] = calloc(num_inputs, sizeof(uint32_t));
      subgraph->cached_in_handles[hw_op_idx][0] =
         rkt_get_tensor(subgraph, op->input_index)->handle;
      if (op->add_tensor != -1)
         subgraph->cached_in_handles[hw_op_idx][1] =
            rkt_get_tensor(subgraph, op->add_tensor)->handle;

      subgraph->cached_out_handles_arr[hw_op_idx] = malloc(sizeof(uint32_t));
      subgraph->cached_out_handles_arr[hw_op_idx][0] =
         rkt_get_tensor(subgraph, op->output_index)->handle;

      struct drm_rocket_job *job = &subgraph->cached_jobs[hw_op_idx];
      job->task_struct_size = sizeof(struct drm_rocket_task);
      job->tasks = (uint64_t)(uintptr_t)&subgraph->cached_tasks[task_offset - num_tasks];
      job->task_count = num_tasks;
      job->in_bo_handles = (uint64_t)(uintptr_t)subgraph->cached_in_handles[hw_op_idx];
      job->in_bo_handle_count = num_inputs;
      job->out_bo_handles = (uint64_t)(uintptr_t)subgraph->cached_out_handles_arr[hw_op_idx];
      job->out_bo_handle_count = 1;

      hw_op_idx++;
   }

   /* Build segments */
   unsigned seg_idx = 0;
   unsigned hw_job_offset = 0;
   unsigned i = 0;
   while (i < num_ops) {
      if (ops[i].type == RKT_OP_CONVOLUTION) {
         struct rkt_exec_segment *seg = &subgraph->exec_segments[seg_idx++];
         seg->is_hw = true;
         seg->first_op = i;
         seg->op_count = 0;
         unsigned first_job = hw_job_offset;
         while (i < num_ops && ops[i].type == RKT_OP_CONVOLUTION) {
            seg->op_count++;
            hw_job_offset++;
            i++;
         }
         seg->submit.job_struct_size = sizeof(struct drm_rocket_job);
         seg->submit.jobs = (uint64_t)(uintptr_t)&subgraph->cached_jobs[first_job];
         seg->submit.job_count = seg->op_count;
         seg->submit.reserved = 0;
      } else {
         struct rkt_exec_segment *seg = &subgraph->exec_segments[seg_idx++];
         seg->is_hw = false;
         seg->first_op = i;
         seg->op_count = 1;
         i++;
      }
   }

   /* Save graph input index and sw_only flag */
   struct rkt_operation *first_op = util_dynarray_begin(&subgraph->operations);
   subgraph->graph_input_index = first_op->input_index;
   subgraph->sw_only = (total_hw_ops == 0);
}

static void
fill_conv_common(struct rkt_operation *operation,
                 const struct pipe_ml_operation *poperation)
{
   operation->type = RKT_OP_CONVOLUTION;
   operation->tasks = UTIL_DYNARRAY_INIT;

   operation->depthwise = rkt_is_depthwise(poperation);
   operation->padding_same = poperation->conv.padding_same;
   operation->stride = poperation->conv.stride_x;

   operation->input_index = poperation->input_tensors[0]->index;
   operation->input_width = poperation->input_tensors[0]->dims[1];
   operation->input_height = poperation->input_tensors[0]->dims[2];
   operation->input_channels = poperation->input_tensors[0]->dims[3];
   operation->input_zero_point = poperation->input_tensors[0]->zero_point;
   operation->input_scale = poperation->input_tensors[0]->scale;

   operation->output_index = poperation->output_tensors[0]->index;
   operation->output_width = poperation->output_tensors[0]->dims[1];
   operation->output_height = poperation->output_tensors[0]->dims[2];
   operation->output_zero_point = poperation->output_tensors[0]->zero_point;
   operation->output_scale = poperation->output_tensors[0]->scale;

   operation->weights_width = poperation->conv.weight_tensor->dims[1];
   operation->weights_height = poperation->conv.weight_tensor->dims[2];
   operation->weights_zero_point = poperation->conv.weight_tensor->zero_point;
}

/*
 * Lower a per-axis CONV into per-group operations, one per 16-channel group.
 * Each group uses max(weight_scale) within its 16 channels, keeping the
 * within-group ratio small (typically < 2x). Returns the number of operations
 * appended to subgraph->operations.
 *
 * This mirrors the RKNN vendor driver approach (per-channel task decomposition)
 * but uses groups of 16 for efficiency since that matches FEATURE_ATOMIC_SIZE.
 */
static unsigned
lower_convolution_per_group(struct rkt_ml_subgraph *subgraph,
                            const struct pipe_ml_operation *poperation)
{
   struct pipe_tensor *wt = poperation->conv.weight_tensor;
   unsigned full_oc = poperation->output_tensors[0]->dims[3];
   unsigned num_groups = DIV_ROUND_UP(full_oc, FEATURE_ATOMIC_SIZE);
   unsigned ow = poperation->output_tensors[0]->dims[1];
   unsigned oh = poperation->output_tensors[0]->dims[2];

   for (unsigned g = 0; g < num_groups; g++) {
      struct rkt_operation group_op = {0};
      group_op.add_tensor = -1;
      fill_conv_common(&group_op, poperation);

      unsigned group_start = g * FEATURE_ATOMIC_SIZE;
      unsigned group_count = MIN2(FEATURE_ATOMIC_SIZE, full_oc - group_start);

      group_op.output_channels = group_count;
      group_op.output_tensor_channels = full_oc;
      /* Each per-group op writes 32 channels (16 real + 16 padding due to
       * the hardware minimum alignment). So hardware groups are spaced 2x
       * apart: group g occupies hardware groups 2g and 2g+1. */
      group_op.per_channel_group_offset =
         g * 2 * oh * ow * FEATURE_ATOMIC_SIZE;

      /* Use max scale within this group — keeps within-group ratio small */
      float group_max = wt->scales[group_start];
      for (unsigned i = 1; i < group_count; i++)
         if (wt->scales[group_start + i] > group_max)
            group_max = wt->scales[group_start + i];
      group_op.weights_scale = group_max;

      /* Within-group correction factors (all close to 1.0) */
      group_op.per_axis_correction = calloc(group_count, sizeof(float));
      for (unsigned i = 0; i < group_count; i++)
         group_op.per_axis_correction[i] =
            wt->scales[group_start + i] / group_max;

      group_op.weights = rkt_fill_weights_group(subgraph, poperation,
                                                 group_start, group_count);
      group_op.biases = rkt_fill_biases_group(subgraph, poperation,
                                               group_start, group_count,
                                               &group_op.truncate_bits);

      util_dynarray_append(&subgraph->operations, group_op);
   }

   return num_groups;
}

static void
lower_convolution(struct rkt_ml_subgraph *subgraph,
                  const struct pipe_ml_operation *poperation,
                  struct rkt_operation *operation)
{
   fill_conv_common(operation, poperation);
   operation->output_channels = poperation->output_tensors[0]->dims[3];

   struct pipe_tensor *wt = poperation->conv.weight_tensor;
   if (wt->scales != NULL) {
      /* Per-axis weights: handled by lower_convolution_per_group instead.
       * This path should not be reached for per-axis ops. */
      assert(!"per-axis should use lower_convolution_per_group");
   }

   operation->weights_scale = poperation->conv.weight_tensor->scale;
   operation->weights = rkt_fill_weights(subgraph, poperation);
   operation->biases =
      rkt_fill_biases(subgraph, poperation, &operation->truncate_bits);
}

static void
lower_concatenation(const struct pipe_ml_operation *poperation,
                    struct rkt_operation *operation)
{
   operation->type = RKT_OP_CONCATENATION;

   operation->input_index = poperation->input_tensors[0]->index;
   operation->input_width = poperation->input_tensors[0]->dims[1];
   operation->input_height = poperation->input_tensors[0]->dims[2];
   operation->input_channels = poperation->input_tensors[0]->dims[3];
   operation->input_zero_point = poperation->input_tensors[0]->zero_point;
   operation->input_scale = poperation->input_tensors[0]->scale;

   operation->output_index = poperation->output_tensors[0]->index;
   operation->output_width = poperation->output_tensors[0]->dims[1];
   operation->output_height = poperation->output_tensors[0]->dims[2];
   operation->output_channels = poperation->output_tensors[0]->dims[3];
   operation->output_zero_point = poperation->output_tensors[0]->zero_point;
   operation->output_scale = poperation->output_tensors[0]->scale;

   unsigned n = poperation->input_count;
   operation->sw.concat.input_count = n;
   operation->sw.concat.input_indices = calloc(n, sizeof(unsigned));
   operation->sw.concat.input_channels_arr = calloc(n, sizeof(unsigned));

   for (unsigned i = 0; i < n; i++) {
      operation->sw.concat.input_indices[i] = poperation->input_tensors[i]->index;
      operation->sw.concat.input_channels_arr[i] = poperation->input_tensors[i]->dims[3];
   }
}

static void
lower_max_pool(const struct pipe_ml_operation *poperation,
               struct rkt_operation *operation)
{
   operation->type = RKT_OP_MAX_POOL_2D;

   operation->input_index = poperation->input_tensors[0]->index;
   operation->input_width = poperation->input_tensors[0]->dims[1];
   operation->input_height = poperation->input_tensors[0]->dims[2];
   operation->input_channels = poperation->input_tensors[0]->dims[3];
   operation->input_zero_point = poperation->input_tensors[0]->zero_point;
   operation->input_scale = poperation->input_tensors[0]->scale;

   operation->output_index = poperation->output_tensors[0]->index;
   operation->output_width = poperation->output_tensors[0]->dims[1];
   operation->output_height = poperation->output_tensors[0]->dims[2];
   operation->output_channels = poperation->output_tensors[0]->dims[3];
   operation->output_zero_point = poperation->output_tensors[0]->zero_point;
   operation->output_scale = poperation->output_tensors[0]->scale;

   operation->sw.pool.filter_width = poperation->pooling.filter_width;
   operation->sw.pool.filter_height = poperation->pooling.filter_height;
   operation->sw.pool.stride_x = poperation->pooling.stride_x;
   operation->sw.pool.stride_y = poperation->pooling.stride_y;
   operation->sw.pool.padding_same = poperation->pooling.padding_same;
}

static void
lower_pad(const struct pipe_ml_operation *poperation,
          struct rkt_operation *operation)
{
   operation->type = RKT_OP_PAD;

   operation->input_index = poperation->input_tensors[0]->index;
   operation->input_width = poperation->input_tensors[0]->dims[1];
   operation->input_height = poperation->input_tensors[0]->dims[2];
   operation->input_channels = poperation->input_tensors[0]->dims[3];
   operation->input_zero_point = poperation->input_tensors[0]->zero_point;
   operation->input_scale = poperation->input_tensors[0]->scale;

   operation->output_index = poperation->output_tensors[0]->index;
   operation->output_width = poperation->output_tensors[0]->dims[1];
   operation->output_height = poperation->output_tensors[0]->dims[2];
   operation->output_channels = poperation->output_tensors[0]->dims[3];
   operation->output_zero_point = poperation->output_tensors[0]->zero_point;
   operation->output_scale = poperation->output_tensors[0]->scale;

   /* pipe pad naming: before_x/after_x = height (dims[1]) padding,
    * before_y/after_y = width (dims[2]) padding */
   operation->sw.pad.pad_before_w = poperation->pad.before_x;
   operation->sw.pad.pad_after_w = poperation->pad.after_x;
   operation->sw.pad.pad_before_h = poperation->pad.before_y;
   operation->sw.pad.pad_after_h = poperation->pad.after_y;
}

static void
lower_resize_nearest(const struct pipe_ml_operation *poperation,
                     struct rkt_operation *operation)
{
   operation->type = RKT_OP_RESIZE_NEAREST;

   operation->input_index = poperation->input_tensors[0]->index;
   operation->input_width = poperation->input_tensors[0]->dims[1];
   operation->input_height = poperation->input_tensors[0]->dims[2];
   operation->input_channels = poperation->input_tensors[0]->dims[3];
   operation->input_zero_point = poperation->input_tensors[0]->zero_point;
   operation->input_scale = poperation->input_tensors[0]->scale;

   operation->output_index = poperation->output_tensors[0]->index;
   operation->output_width = poperation->output_tensors[0]->dims[1];
   operation->output_height = poperation->output_tensors[0]->dims[2];
   operation->output_channels = poperation->output_tensors[0]->dims[3];
   operation->output_zero_point = poperation->output_tensors[0]->zero_point;
   operation->output_scale = poperation->output_tensors[0]->scale;
}

static void
lower_logistic(const struct pipe_ml_operation *poperation,
               struct rkt_operation *operation)
{
   operation->type = RKT_OP_LOGISTIC;

   operation->input_index = poperation->input_tensors[0]->index;
   operation->input_width = poperation->input_tensors[0]->dims[1];
   operation->input_height = poperation->input_tensors[0]->dims[2];
   operation->input_channels = poperation->input_tensors[0]->dims[3];
   operation->input_zero_point = poperation->input_tensors[0]->zero_point;
   operation->input_scale = poperation->input_tensors[0]->scale;

   operation->output_index = poperation->output_tensors[0]->index;
   operation->output_width = poperation->output_tensors[0]->dims[1];
   operation->output_height = poperation->output_tensors[0]->dims[2];
   operation->output_channels = poperation->output_tensors[0]->dims[3];
   operation->output_zero_point = poperation->output_tensors[0]->zero_point;
   operation->output_scale = poperation->output_tensors[0]->scale;

   /* Build 256-entry sigmoid LUT: NPU input byte -> NPU output byte */
   float in_scale = poperation->input_tensors[0]->scale;
   int in_zp = poperation->input_tensors[0]->zero_point;
   float out_scale = poperation->output_tensors[0]->scale;
   int out_zp = poperation->output_tensors[0]->zero_point;

   for (int i = 0; i < 256; i++) {
      /* i is the raw NPU byte. Convert to TFLite int8 value:
       * npu_byte = (uint8_t)tfl_int8 - 0x80, so tfl_int8 = (int8_t)(i + 0x80) */
      int tfl_val = (int)(int8_t)((uint8_t)i + 0x80);
      float real_val = (tfl_val - in_zp) * in_scale;
      float sigmoid = 1.0f / (1.0f + expf(-real_val));
      float out_tfl_val = sigmoid / out_scale + out_zp;
      out_tfl_val = fmaxf(-128.0f, fminf(127.0f, roundf(out_tfl_val)));
      /* Convert back to NPU format: npu = (uint8_t)tfl_int8 - 0x80 */
      operation->sw.logistic.lut[i] = (uint8_t)((int8_t)(int)out_tfl_val - 0x80);
   }

   /* Build raw LUT for sw_only: raw int8 byte → raw int8 byte */
   for (int i = 0; i < 256; i++) {
      int tfl_val = (int)(int8_t)(uint8_t)i;
      float real_val = (tfl_val - in_zp) * in_scale;
      float sigmoid = 1.0f / (1.0f + expf(-real_val));
      float out_val = sigmoid / out_scale + out_zp;
      out_val = fmaxf(-128.0f, fminf(127.0f, roundf(out_val)));
      operation->sw.logistic.raw_lut[i] = (uint8_t)(int8_t)(int)out_val;
   }
}

/* ======== Software op execution ======== */

static void
execute_concatenation(struct pipe_context *pcontext,
                      struct rkt_ml_subgraph *subgraph,
                      struct rkt_operation *op)
{
   unsigned w = op->output_width;
   unsigned h = op->output_height;
   unsigned out_channels = op->output_channels;
   unsigned out_groups = DIV_ROUND_UP(out_channels, FEATURE_ATOMIC_SIZE);
   uint8_t out_zp = op->output_zero_point;
   float out_scale = op->output_scale;

   struct pipe_transfer *out_transfer = NULL;
   struct rkt_resource *out_res = rkt_get_tensor(subgraph, op->output_index);
   uint8_t *out_data = pipe_buffer_map(pcontext, &out_res->base,
                                       PIPE_MAP_WRITE, &out_transfer);

   if (subgraph->sw_only) {
      /* Flat NHWC concat: for each pixel, append channels from each input */
      unsigned num_pixels = w * h;
      for (unsigned i = 0; i < op->sw.concat.input_count; i++) {
         unsigned in_idx = op->sw.concat.input_indices[i];
         unsigned in_ch = op->sw.concat.input_channels_arr[i];

         struct pipe_transfer *in_transfer = NULL;
         struct rkt_resource *in_res = rkt_get_tensor(subgraph, in_idx);
         uint8_t *in_data = pipe_buffer_map(pcontext, &in_res->base,
                                            PIPE_MAP_READ, &in_transfer);

         /* Compute channel offset in output for this input */
         unsigned ch_off = 0;
         for (unsigned j = 0; j < i; j++)
            ch_off += op->sw.concat.input_channels_arr[j];

         for (unsigned p = 0; p < num_pixels; p++) {
            memcpy(out_data + p * out_channels + ch_off,
                   in_data + p * in_ch,
                   in_ch);
         }

         pipe_buffer_unmap(pcontext, in_transfer);
      }
   } else {
      /* NPU interleaved format paths */
      bool all_aligned = true;
      for (unsigned i = 0; i < op->sw.concat.input_count; i++) {
         if (op->sw.concat.input_channels_arr[i] % FEATURE_ATOMIC_SIZE != 0)
            all_aligned = false;
      }
      if (all_aligned) {
      /* Fast path: memcpy whole groups from each input */
      unsigned out_group_offset = 0;
      unsigned group_plane_size = w * h * FEATURE_ATOMIC_SIZE;

      for (unsigned i = 0; i < op->sw.concat.input_count; i++) {
         unsigned in_idx = op->sw.concat.input_indices[i];
         unsigned in_ch = op->sw.concat.input_channels_arr[i];
         unsigned in_groups = DIV_ROUND_UP(in_ch, FEATURE_ATOMIC_SIZE);

         struct pipe_transfer *in_transfer = NULL;
         struct rkt_resource *in_res = rkt_get_tensor(subgraph, in_idx);
         uint8_t *in_data = pipe_buffer_map(pcontext, &in_res->base,
                                            PIPE_MAP_READ, &in_transfer);

         memcpy(out_data + out_group_offset * group_plane_size,
                in_data,
                in_groups * group_plane_size);

         pipe_buffer_unmap(pcontext, in_transfer);
         out_group_offset += in_groups;
      }
   } else {
      /* Slow path: per-element copy with channel remapping */
      unsigned channel_offset = 0;

      for (unsigned i = 0; i < op->sw.concat.input_count; i++) {
         unsigned in_idx = op->sw.concat.input_indices[i];
         unsigned in_ch = op->sw.concat.input_channels_arr[i];

         struct pipe_transfer *in_transfer = NULL;
         struct rkt_resource *in_res = rkt_get_tensor(subgraph, in_idx);
         uint8_t *in_data = pipe_buffer_map(pcontext, &in_res->base,
                                            PIPE_MAP_READ, &in_transfer);

         for (unsigned c = 0; c < in_ch; c++) {
            unsigned src_g = c / FEATURE_ATOMIC_SIZE;
            unsigned src_c = c % FEATURE_ATOMIC_SIZE;
            unsigned dst_g = (channel_offset + c) / FEATURE_ATOMIC_SIZE;
            unsigned dst_c = (channel_offset + c) % FEATURE_ATOMIC_SIZE;

            for (unsigned x = 0; x < w; x++) {
               for (unsigned y = 0; y < h; y++) {
                  out_data[NPU_OFFSET(dst_g, x, y, w, h) + dst_c] =
                     in_data[NPU_OFFSET(src_g, x, y, w, h) + src_c];
               }
            }
         }

         pipe_buffer_unmap(pcontext, in_transfer);
         channel_offset += in_ch;
      }
      }
   }

   pipe_buffer_unmap(pcontext, out_transfer);
}

static void
execute_max_pool(struct pipe_context *pcontext,
                 struct rkt_ml_subgraph *subgraph,
                 struct rkt_operation *op)
{
   unsigned in_w = op->input_width;
   unsigned in_h = op->input_height;
   unsigned out_w = op->output_width;
   unsigned out_h = op->output_height;
   unsigned channels = op->input_channels;
   unsigned groups = DIV_ROUND_UP(channels, FEATURE_ATOMIC_SIZE);

   unsigned fw = op->sw.pool.filter_width;
   unsigned fh = op->sw.pool.filter_height;
   unsigned sx = op->sw.pool.stride_x;
   unsigned sy = op->sw.pool.stride_y;

   /* Compute padding for padding_same.
    * For sw_only (flat NHWC): ox=H-axis uses fh/sy, oy=W-axis uses fw/sx.
    * For NPU format: ox uses fw/sx, oy uses fh/sy. */
   unsigned pad_w_before = 0, pad_h_before = 0;
   if (op->sw.pool.padding_same) {
      if (subgraph->sw_only) {
         unsigned pad_w_total = (out_w - 1) * sy + fh - in_w;
         unsigned pad_h_total = (out_h - 1) * sx + fw - in_h;
         pad_w_before = pad_w_total / 2;
         pad_h_before = pad_h_total / 2;
      } else {
         unsigned pad_w_total = (out_w - 1) * sx + fw - in_w;
         unsigned pad_h_total = (out_h - 1) * sy + fh - in_h;
         pad_w_before = pad_w_total / 2;
         pad_h_before = pad_h_total / 2;
      }
   }

   struct pipe_transfer *in_transfer = NULL;
   struct rkt_resource *in_res = rkt_get_tensor(subgraph, op->input_index);
   uint8_t *in_data = pipe_buffer_map(pcontext, &in_res->base,
                                      PIPE_MAP_READ, &in_transfer);

   struct pipe_transfer *out_transfer = NULL;
   struct rkt_resource *out_res = rkt_get_tensor(subgraph, op->output_index);
   uint8_t *out_data = pipe_buffer_map(pcontext, &out_res->base,
                                       PIPE_MAP_WRITE, &out_transfer);

   if (subgraph->sw_only) {
      /* Flat NHWC with raw TFLite values (no 0x80 bias).
       * data[x * in_h * C + y * C + c] where x=dims[1]=TFLite H,
       * y=dims[2]=TFLite W. Use fh/sy for ox-axis (TFLite H). */
      for (unsigned ox = 0; ox < out_w; ox++) {
         for (unsigned oy = 0; oy < out_h; oy++) {
            uint8_t *dst = out_data + (ox * out_h + oy) * channels;
            /* Init with minimum int8 value for max pooling */
            memset(dst, 0x80, channels);

            for (unsigned fx = 0; fx < fh; fx++) {
               for (unsigned fy = 0; fy < fw; fy++) {
                  int ix = (int)(ox * sy) - (int)pad_w_before + (int)fx;
                  int iy = (int)(oy * sx) - (int)pad_h_before + (int)fy;

                  if (ix < 0 || ix >= (int)in_w || iy < 0 || iy >= (int)in_h)
                     continue;

                  uint8_t *src = in_data + (ix * in_h + iy) * channels;
                  for (unsigned c = 0; c < channels; c++) {
                     /* Raw int8 comparison — no bias inversion */
                     if ((int8_t)src[c] > (int8_t)dst[c])
                        dst[c] = src[c];
                  }
               }
            }
         }
      }
   } else {
      /* NPU interleaved format (for mixed HW/SW subgraphs) */
      for (unsigned g = 0; g < groups; g++) {
         unsigned real_c = MIN2(FEATURE_ATOMIC_SIZE, channels - g * FEATURE_ATOMIC_SIZE);

         for (unsigned ox = 0; ox < out_w; ox++) {
            for (unsigned oy = 0; oy < out_h; oy++) {
               uint8_t *dst = out_data + NPU_OFFSET(g, ox, oy, out_w, out_h);

               memset(dst, 0x80, FEATURE_ATOMIC_SIZE);

               for (unsigned fx = 0; fx < fw; fx++) {
                  for (unsigned fy = 0; fy < fh; fy++) {
                     int ix = (int)(ox * sx) - (int)pad_w_before + (int)fx;
                     int iy = (int)(oy * sy) - (int)pad_h_before + (int)fy;

                     if (ix < 0 || ix >= (int)in_w || iy < 0 || iy >= (int)in_h)
                        continue;

                     uint8_t *src = in_data + NPU_OFFSET(g, ix, iy, in_w, in_h);
                     for (unsigned c = 0; c < real_c; c++) {
                        if ((int8_t)src[c] > (int8_t)dst[c])
                           dst[c] = src[c];
                     }
                  }
               }
            }
         }
      }
   }

   pipe_buffer_unmap(pcontext, in_transfer);
   pipe_buffer_unmap(pcontext, out_transfer);
}

static void
execute_pad(struct pipe_context *pcontext,
            struct rkt_ml_subgraph *subgraph,
            struct rkt_operation *op)
{
   unsigned in_w = op->input_width;
   unsigned in_h = op->input_height;
   unsigned out_w = op->output_width;
   unsigned out_h = op->output_height;
   unsigned channels = op->input_channels;
   unsigned groups = DIV_ROUND_UP(channels, FEATURE_ATOMIC_SIZE);

   unsigned pb_w = op->sw.pad.pad_before_w;
   unsigned pb_h = op->sw.pad.pad_before_h;

   /* Zero-point in NPU format */
   uint8_t pad_val = (uint8_t)((int)op->input_zero_point - 0x80);

   struct pipe_transfer *in_transfer = NULL;
   struct rkt_resource *in_res = rkt_get_tensor(subgraph, op->input_index);
   uint8_t *in_data = pipe_buffer_map(pcontext, &in_res->base,
                                      PIPE_MAP_READ, &in_transfer);

   struct pipe_transfer *out_transfer = NULL;
   struct rkt_resource *out_res = rkt_get_tensor(subgraph, op->output_index);
   uint8_t *out_data = pipe_buffer_map(pcontext, &out_res->base,
                                       PIPE_MAP_WRITE, &out_transfer);

   /* Read x-major, write y-major (see execute_max_pool comment) */
   if (subgraph->sw_only) {
      /* Flat NHWC */
      for (unsigned ox = 0; ox < out_w; ox++) {
         for (unsigned oy = 0; oy < out_h; oy++) {
            int ix = (int)ox - (int)pb_w;
            int iy = (int)oy - (int)pb_h;
            uint8_t *dst = out_data + (ox * out_h + oy) * channels;
            if (ix >= 0 && ix < (int)in_w && iy >= 0 && iy < (int)in_h) {
               uint8_t *src = in_data + (ix * in_h + iy) * channels;
               memcpy(dst, src, channels);
            } else {
               memset(dst, pad_val, channels);
            }
         }
      }
   } else {
      for (unsigned g = 0; g < groups; g++) {
         for (unsigned ox = 0; ox < out_w; ox++) {
            for (unsigned oy = 0; oy < out_h; oy++) {
               int ix = (int)ox - (int)pb_w;
               int iy = (int)oy - (int)pb_h;
               uint8_t *dst = out_data + NPU_OFFSET(g, ox, oy, out_w, out_h);
               if (ix >= 0 && ix < (int)in_w && iy >= 0 && iy < (int)in_h) {
                  uint8_t *src = in_data + NPU_OFFSET(g, ix, iy, in_w, in_h);
                  memcpy(dst, src, FEATURE_ATOMIC_SIZE);
               } else {
                  memset(dst, pad_val, FEATURE_ATOMIC_SIZE);
               }
            }
         }
      }
   }

   pipe_buffer_unmap(pcontext, in_transfer);
   pipe_buffer_unmap(pcontext, out_transfer);
}

static void
execute_resize_nearest(struct pipe_context *pcontext,
                       struct rkt_ml_subgraph *subgraph,
                       struct rkt_operation *op)
{
   unsigned in_w = op->input_width;
   unsigned in_h = op->input_height;
   unsigned out_w = op->output_width;
   unsigned out_h = op->output_height;
   unsigned channels = op->input_channels;
   unsigned groups = DIV_ROUND_UP(channels, FEATURE_ATOMIC_SIZE);

   struct pipe_transfer *in_transfer = NULL;
   struct rkt_resource *in_res = rkt_get_tensor(subgraph, op->input_index);
   uint8_t *in_data = pipe_buffer_map(pcontext, &in_res->base,
                                      PIPE_MAP_READ, &in_transfer);

   struct pipe_transfer *out_transfer = NULL;
   struct rkt_resource *out_res = rkt_get_tensor(subgraph, op->output_index);
   uint8_t *out_data = pipe_buffer_map(pcontext, &out_res->base,
                                       PIPE_MAP_WRITE, &out_transfer);

   if (subgraph->sw_only) {
      /* Flat NHWC */
      for (unsigned ox = 0; ox < out_w; ox++) {
         for (unsigned oy = 0; oy < out_h; oy++) {
            unsigned ix = ox * in_w / out_w;
            unsigned iy = oy * in_h / out_h;
            uint8_t *src = in_data + (ix * in_h + iy) * channels;
            uint8_t *dst = out_data + (ox * out_h + oy) * channels;
            memcpy(dst, src, channels);
         }
      }
   } else {
      for (unsigned g = 0; g < groups; g++) {
         for (unsigned ox = 0; ox < out_w; ox++) {
            for (unsigned oy = 0; oy < out_h; oy++) {
               unsigned ix = ox * in_w / out_w;
               unsigned iy = oy * in_h / out_h;
               uint8_t *src = in_data + NPU_OFFSET(g, ix, iy, in_w, in_h);
               uint8_t *dst = out_data + NPU_OFFSET(g, ox, oy, out_w, out_h);
               memcpy(dst, src, FEATURE_ATOMIC_SIZE);
            }
         }
      }
   }

   pipe_buffer_unmap(pcontext, in_transfer);
   pipe_buffer_unmap(pcontext, out_transfer);
}

static void
execute_logistic(struct pipe_context *pcontext,
                 struct rkt_ml_subgraph *subgraph,
                 struct rkt_operation *op)
{
   unsigned w = op->input_width;
   unsigned h = op->input_height;
   unsigned channels = op->input_channels;
   unsigned groups = DIV_ROUND_UP(channels, FEATURE_ATOMIC_SIZE);
   unsigned total_size = groups * w * h * FEATURE_ATOMIC_SIZE;

   struct pipe_transfer *in_transfer = NULL;
   struct rkt_resource *in_res = rkt_get_tensor(subgraph, op->input_index);
   uint8_t *in_data = pipe_buffer_map(pcontext, &in_res->base,
                                      PIPE_MAP_READ, &in_transfer);

   struct pipe_transfer *out_transfer = NULL;
   struct rkt_resource *out_res = rkt_get_tensor(subgraph, op->output_index);
   uint8_t *out_data = pipe_buffer_map(pcontext, &out_res->base,
                                       PIPE_MAP_WRITE, &out_transfer);

   const uint8_t *lut = subgraph->sw_only ?
      op->sw.logistic.raw_lut : op->sw.logistic.lut;
   unsigned count = subgraph->sw_only ?
      w * h * channels : total_size;

   for (unsigned i = 0; i < count; i++)
      out_data[i] = lut[in_data[i]];

   pipe_buffer_unmap(pcontext, in_transfer);
   pipe_buffer_unmap(pcontext, out_transfer);
}

/*
 * Apply per-channel scale correction after NPU CONV for per-axis quantized weights.
 *
 * The NPU computes: npu_out = round(acc * S0/So) + ozp
 *   where S0 = input_scale * weight_scale[0] (first channel's scale)
 * The correct result: correct = round(acc * Sc/So) + ozp
 *   where Sc = input_scale * weight_scale[oc]
 *
 * Correction: correct = round((npu_out - ozp) * weight_scale[oc]/weight_scale[0]) + ozp
 *
 * The output tensor is in NPU interleaved format:
 *   [group][y=height][x=width][FEATURE_ATOMIC_SIZE=16]
 * Each group of 16 channels gets the same correction factor applied to all
 * spatial positions within that channel.
 */
static void
apply_per_axis_correction(struct pipe_context *pcontext,
                          struct rkt_ml_subgraph *subgraph,
                          struct rkt_operation *op)
{
   if (op->per_axis_correction == NULL)
      return;

   unsigned w = op->output_width;
   unsigned h = op->output_height;
   unsigned oc = op->output_channels;
   int ozp_npu = (int)(uint8_t)op->output_zero_point - 0x80;

   struct pipe_transfer *transfer = NULL;
   struct rkt_resource *res = rkt_get_tensor(subgraph, op->output_index);
   uint8_t *data = pipe_buffer_map(pcontext, &res->base,
                                   PIPE_MAP_READ_WRITE, &transfer);

   /* Iterate in the NPU output layout order (y-major, matching read_outputs).
    * For per-group ops, per_channel_group_offset positions us at the right
    * group within the full output tensor. */
   unsigned groups = DIV_ROUND_UP(oc, FEATURE_ATOMIC_SIZE);
   for (unsigned g = 0; g < groups; g++) {
      unsigned base_c = g * FEATURE_ATOMIC_SIZE;
      unsigned real_c = MIN2(FEATURE_ATOMIC_SIZE, oc - base_c);
      unsigned group_off = g * h * w * FEATURE_ATOMIC_SIZE +
                           op->per_channel_group_offset;

      for (unsigned y = 0; y < h; y++) {
         for (unsigned x = 0; x < w; x++) {
            uint8_t *pixel = data + group_off +
               y * w * FEATURE_ATOMIC_SIZE + x * FEATURE_ATOMIC_SIZE;
            for (unsigned c = 0; c < real_c; c++) {
               float corr = op->per_axis_correction[base_c + c];
               if (corr == 1.0f)
                  continue;
               int val = (int)(int8_t)pixel[c] - ozp_npu;
               val = (int)roundf(val * corr) + ozp_npu;
               if (val < -128) val = -128;
               if (val > 127) val = 127;
               pixel[c] = (uint8_t)(int8_t)val;
            }
         }
      }
   }

   pipe_buffer_unmap(pcontext, transfer);
}

/*
 * Compact per-group output: move data from sparse 2x-spaced hardware groups
 * to consecutive groups. After per-group CONV ops, each group's real data is
 * at hardware group 2*g. This moves it to hardware group g so read_outputs
 * sees the standard contiguous layout.
 */
static void
compact_per_group_output(struct pipe_context *pcontext,
                         struct rkt_ml_subgraph *subgraph,
                         struct rkt_operation *op)
{
   if (op->output_tensor_channels == 0)
      return;

   unsigned w = op->output_width;
   unsigned h = op->output_height;
   unsigned full_oc = op->output_tensor_channels;
   unsigned num_groups = DIV_ROUND_UP(full_oc, FEATURE_ATOMIC_SIZE);
   unsigned group_size = h * w * FEATURE_ATOMIC_SIZE;

   struct pipe_transfer *transfer = NULL;
   struct rkt_resource *res = rkt_get_tensor(subgraph, op->output_index);
   uint8_t *data = pipe_buffer_map(pcontext, &res->base,
                                   PIPE_MAP_READ_WRITE, &transfer);

   /* Group g's data is at offset 2*g*group_size, needs to move to g*group_size.
    * Process forward — source is always ahead of destination so no overlap. */
   for (unsigned g = 1; g < num_groups; g++) {
      memmove(data + g * group_size, data + 2 * g * group_size, group_size);
   }

   pipe_buffer_unmap(pcontext, transfer);
}

static void
execute_sw_op(struct pipe_context *pcontext,
              struct rkt_ml_subgraph *subgraph,
              unsigned op_index)
{
   struct rkt_operation *op =
      util_dynarray_element(&subgraph->operations, struct rkt_operation, op_index);

   DBG("Executing SW op %d (type %d)\n", op_index, op->type);

   switch (op->type) {
   case RKT_OP_CONCATENATION:
      execute_concatenation(pcontext, subgraph, op);
      break;
   case RKT_OP_MAX_POOL_2D:
      execute_max_pool(pcontext, subgraph, op);
      break;
   case RKT_OP_PAD:
      execute_pad(pcontext, subgraph, op);
      break;
   case RKT_OP_RESIZE_NEAREST:
      execute_resize_nearest(pcontext, subgraph, op);
      break;
   case RKT_OP_LOGISTIC:
      execute_logistic(pcontext, subgraph, op);
      break;
   default:
      UNREACHABLE("Not a software op");
   }

}

/* ======== End software op execution ======== */

static struct rkt_operation *
find_first_consumer(struct rkt_ml_subgraph *subgraph, unsigned tensor_index)
{
   util_dynarray_foreach (&subgraph->operations, struct rkt_operation,
                          operation) {
      if (operation->input_index == tensor_index)
         return operation;
      if (operation->type == RKT_OP_CONCATENATION) {
         for (unsigned j = 0; j < operation->sw.concat.input_count; j++) {
            if (operation->sw.concat.input_indices[j] == tensor_index)
               return operation;
         }
      }
   }

   return NULL;
}

static struct rkt_operation *
find_producer(struct rkt_ml_subgraph *subgraph,
              unsigned tensor_index)
{
   util_dynarray_foreach (&subgraph->operations, struct rkt_operation,
                          operation) {
      if (operation->output_index == tensor_index)
         return operation;
   }

   return NULL;
}

static unsigned
count_tensors(const struct pipe_ml_operation *poperations,
              unsigned count)
{
   unsigned tensor_count = 0;

   for (unsigned i = 0; i < count; i++) {
      const struct pipe_ml_operation *poperation = &poperations[i];
      tensor_count = MAX2(tensor_count, poperation->input_tensors[0]->index);
      tensor_count = MAX2(tensor_count, poperation->output_tensors[0]->index);
      switch (poperation->type) {
      case PIPE_ML_OPERATION_TYPE_CONVOLUTION:
         tensor_count = MAX2(tensor_count, poperation->conv.weight_tensor->index);
         tensor_count = MAX2(tensor_count, poperation->conv.bias_tensor->index);
         break;
      case PIPE_ML_OPERATION_TYPE_ADD:
         tensor_count = MAX2(tensor_count, poperation->input_tensors[1]->index);
         break;
      case PIPE_ML_OPERATION_TYPE_CONCATENATION:
         for (unsigned j = 1; j < poperation->input_count; j++)
            tensor_count = MAX2(tensor_count, poperation->input_tensors[j]->index);
         break;
      case PIPE_ML_OPERATION_TYPE_POOLING:
      case PIPE_ML_OPERATION_TYPE_PAD:
      case PIPE_ML_OPERATION_TYPE_RESIZE:
      case PIPE_ML_OPERATION_TYPE_LOGISTIC:
         break;
      default:
         DBG("poperation->type %d\n", poperation->type);
         UNREACHABLE("Unsupported ML operation type");
      }
   }

   return tensor_count + 1;
}

static bool
tensor_quantization_supported(struct pipe_tensor *tensor)
{
   /*
    * Per-axis quantization not supported, for details see:
    * https://ai.google.dev/edge/litert/models/quantization_spec#per-axis_vs_per-tensor
    */
   return tensor->scales == NULL && tensor->zero_points == NULL;
}

static bool
is_quantized_feature_tensor(struct pipe_tensor *tensor)
{
   /* Must be a quantized tensor (non-zero scale) with 4D NHWC shape */
   return tensor->scale != 0.0f && tensor->dims[1] > 0 && tensor->dims[2] > 0;
}

bool
rkt_ml_operation_supported(struct pipe_context *pcontext,
                           const struct pipe_ml_operation *operation)
{
   bool supported = false;

   switch (operation->type) {
   case PIPE_ML_OPERATION_TYPE_CONVOLUTION: {
      struct pipe_tensor *input_tensor = operation->input_tensors[0];
      struct pipe_tensor *weight_tensor = operation->conv.weight_tensor;
      struct pipe_tensor *bias_tensor = operation->conv.bias_tensor;
      struct pipe_tensor *output_tensor = operation->output_tensors[0];

      /* Per-axis weights are supported via post-CONV scale correction.
       * Only require per-tensor quantization on input/output tensors. */
      if (tensor_quantization_supported(input_tensor) &&
          tensor_quantization_supported(output_tensor) &&
          operation->conv.dilation_width_factor == 1 &&
          operation->conv.dilation_height_factor == 1)
         supported = true;

      break;
   }
   case PIPE_ML_OPERATION_TYPE_ADD:
      supported = operation->input_tensors[0]->resource == NULL &&
                  operation->input_tensors[1]->resource == NULL;
      break;
   case PIPE_ML_OPERATION_TYPE_CONCATENATION: {
      int axis = operation->conc.axis;
      if (axis < 0) axis += 4;
      if (axis != 3) break;
      if (!is_quantized_feature_tensor(operation->output_tensors[0])) break;
      supported = tensor_quantization_supported(operation->output_tensors[0]);
      for (unsigned i = 0; i < operation->input_count && supported; i++) {
         if (!tensor_quantization_supported(operation->input_tensors[i]) ||
             !is_quantized_feature_tensor(operation->input_tensors[i]))
            supported = false;
      }
      break;
   }
   case PIPE_ML_OPERATION_TYPE_POOLING:
      supported = (operation->pooling.type == PIPE_ML_POOLING_TYPE_MAX) &&
                  is_quantized_feature_tensor(operation->input_tensors[0]) &&
                  is_quantized_feature_tensor(operation->output_tensors[0]) &&
                  tensor_quantization_supported(operation->input_tensors[0]) &&
                  tensor_quantization_supported(operation->output_tensors[0]);
      break;
   case PIPE_ML_OPERATION_TYPE_PAD:
      supported = is_quantized_feature_tensor(operation->input_tensors[0]) &&
                  is_quantized_feature_tensor(operation->output_tensors[0]) &&
                  tensor_quantization_supported(operation->input_tensors[0]) &&
                  tensor_quantization_supported(operation->output_tensors[0]) &&
                  operation->pad.before_z == 0 && operation->pad.after_z == 0;
      break;
   case PIPE_ML_OPERATION_TYPE_RESIZE:
      supported = is_quantized_feature_tensor(operation->input_tensors[0]) &&
                  is_quantized_feature_tensor(operation->output_tensors[0]) &&
                  tensor_quantization_supported(operation->input_tensors[0]) &&
                  tensor_quantization_supported(operation->output_tensors[0]);
      break;
   case PIPE_ML_OPERATION_TYPE_LOGISTIC:
      supported = is_quantized_feature_tensor(operation->input_tensors[0]) &&
                  is_quantized_feature_tensor(operation->output_tensors[0]) &&
                  tensor_quantization_supported(operation->input_tensors[0]) &&
                  tensor_quantization_supported(operation->output_tensors[0]);
      break;
   default:
      supported = false;
   }

   return supported;
}

struct pipe_ml_subgraph *
rkt_ml_subgraph_create(struct pipe_context *pcontext,
                       const struct pipe_ml_operation *poperations,
                       unsigned count)
{
   struct rkt_ml_subgraph *subgraph;
   unsigned tensor_count;

   subgraph = calloc(1, sizeof(*subgraph));
   subgraph->base.context = pcontext;

   tensor_count = count_tensors(poperations, count);
   subgraph->tensors = UTIL_DYNARRAY_INIT;
   subgraph->operations = UTIL_DYNARRAY_INIT;
   if (!util_dynarray_resize(&subgraph->tensors, struct pipe_resource *,
                             tensor_count))
      return NULL;
   memset(util_dynarray_begin(&subgraph->tensors), 0, subgraph->tensors.size);

   /* Lower */
   for (int i = 0; i < count; i++) {
      struct rkt_operation operation = {0};
      operation.add_tensor = -1;

      switch (poperations[i].type) {
      case PIPE_ML_OPERATION_TYPE_CONVOLUTION:
         if (poperations[i].conv.weight_tensor->scales != NULL) {
            /* Per-axis weights: decompose into per-group operations */
            lower_convolution_per_group(subgraph, &poperations[i]);
            /* Don't append 'operation' — per_group already appended N ops */
         } else {
            lower_convolution(subgraph, &poperations[i], &operation);
            util_dynarray_append(&subgraph->operations, operation);
         }
         break;
      case PIPE_ML_OPERATION_TYPE_ADD: {
         /* Fuse tensor addition into convolution*/
         struct rkt_operation *input_op_1 =
            find_producer(subgraph, poperations[i].input_tensors[1]->index);
         struct rkt_operation *input_op_2 =
            find_producer(subgraph, poperations[i].input_tensors[0]->index);

         assert(input_op_1);
         assert(input_op_2);

         if (input_op_1 == NULL) {
            /* Graph input */
            input_op_2->add_tensor = poperations[i].input_tensors[1]->index;
         } else {
            input_op_1->addition_input = true;
            input_op_2->add_tensor = input_op_1->output_index;
         }

         input_op_2->output_index = poperations[i].output_tensors[0]->index;
         input_op_2->addition_offset =
            0x80 - poperations[i].input_tensors[1]->zero_point;
         input_op_2->addition_scale = poperations[i].input_tensors[1]->scale;

         break;
      }
      case PIPE_ML_OPERATION_TYPE_CONCATENATION:
         lower_concatenation(&poperations[i], &operation);
         util_dynarray_append(&subgraph->operations, operation);
         break;
      case PIPE_ML_OPERATION_TYPE_POOLING:
         lower_max_pool(&poperations[i], &operation);
         util_dynarray_append(&subgraph->operations, operation);
         break;
      case PIPE_ML_OPERATION_TYPE_PAD:
         lower_pad(&poperations[i], &operation);
         util_dynarray_append(&subgraph->operations, operation);
         break;
      case PIPE_ML_OPERATION_TYPE_RESIZE:
         lower_resize_nearest(&poperations[i], &operation);
         util_dynarray_append(&subgraph->operations, operation);
         break;
      case PIPE_ML_OPERATION_TYPE_LOGISTIC:
         lower_logistic(&poperations[i], &operation);
         util_dynarray_append(&subgraph->operations, operation);
         break;
      default:
         DBG("poperation->type %d\n", poperations[i].type);
         UNREACHABLE("Unsupported ML operation type");
      }
   }

   /* Create input tensors */
   util_dynarray_foreach (&subgraph->operations, struct rkt_operation,
                          operation) {
      unsigned input_channels_1 =
         DIV_ROUND_UP(operation->input_channels, FEATURE_ATOMIC_SIZE) * 2;
      unsigned input_channels_2 = FEATURE_ATOMIC_SIZE;
      unsigned input_size = operation->input_width * operation->input_height *
                            input_channels_1 * input_channels_2;

      create_tensor(subgraph, operation->input_index, input_size);

      /* For concat, also create tensors for additional inputs */
      if (operation->type == RKT_OP_CONCATENATION) {
         for (unsigned j = 0; j < operation->sw.concat.input_count; j++) {
            unsigned idx = operation->sw.concat.input_indices[j];
            unsigned ch = operation->sw.concat.input_channels_arr[j];
            unsigned sz = calc_npu_tensor_size(operation->input_width,
                                               operation->input_height, ch);
            create_tensor(subgraph, idx, sz);
         }
      }
   }

   /* Create output tensors */
   util_dynarray_foreach (&subgraph->operations, struct rkt_operation,
                          operation) {
      struct rkt_resource *res =
         rkt_get_tensor(subgraph, operation->output_index);
      if (res != NULL)
         continue;

      create_tensor(subgraph, operation->output_index,
                    calc_raw_output_size(operation));
   }

   /* Mark intermediate and output tensors as device-resident (written by NPU) */
   util_dynarray_foreach (&subgraph->operations, struct rkt_operation,
                          operation) {
      /* Don't mark graph input tensor - CPU writes it every invoke */
   }

   struct rkt_operation *first_op = util_dynarray_begin(&subgraph->operations);

   /* Compile HW ops only */
   util_dynarray_foreach (&subgraph->operations, struct rkt_operation,
                          operation) {
      if (operation->type != RKT_OP_CONVOLUTION)
         continue;
      rkt_split_tasks(subgraph, operation);
      compile_operation(subgraph, operation);
   }

   /* Build execution plan with mixed HW/SW segments */
   build_execution_plan(subgraph);

   return &subgraph->base;
}

void
rkt_ml_subgraph_invoke(struct pipe_context *pcontext,
                       struct pipe_ml_subgraph *psubgraph,
                       unsigned inputs_count, unsigned input_idxs[],
                       void *inputs[], bool is_signed[])
{
   struct rkt_screen *screen = rkt_screen(pcontext->screen);
   struct rkt_ml_subgraph *subgraph = (struct rkt_ml_subgraph *)(psubgraph);
   int ret;

   DBG("Processing input\n");

   for (int i = 0; i < inputs_count; i++) {
      struct rkt_operation *operation =
         find_first_consumer(subgraph, input_idxs[i]);
      struct pipe_resource *input =
         &rkt_get_tensor(subgraph, input_idxs[i])->base;
      unsigned input_channels = operation->input_channels;
      unsigned output_channels = operation->output_channels;

      /* Write to the actual tensor being provided, not necessarily
       * operation->input_index (which may differ for multi-input ops
       * like CONCAT where each graph input maps to a different tensor) */
      struct rkt_resource *input_tensor =
         rkt_get_tensor(subgraph, input_idxs[i]);
      if (subgraph->sw_only) {
         /* SW-only subgraph: copy flat NHWC bytes as-is, no bias, no
          * interleaving. SW ops work on raw TFLite values directly. */
         struct pipe_transfer *transfer_out;
         uint8_t *map = pipe_buffer_map(pcontext, &input_tensor->base,
                                        PIPE_MAP_WRITE, &transfer_out);
         unsigned total = input_channels * operation->input_width *
                          operation->input_height;
         memcpy(map, inputs[i], total);
         pipe_buffer_unmap(pcontext, transfer_out);
      } else if (output_channels == 1 && input_channels == 1 &&
          !operation->addition_input && (operation->add_tensor == -1)) {
         pipe_buffer_copy(pcontext, &input_tensor->base, input, 0, 0,
                          pipe_buffer_size(input));
      } else {
         unsigned input_width = operation->input_width;
         unsigned input_height = operation->input_height;
         unsigned zero_point = operation->input_zero_point;
         struct pipe_transfer *transfer_out;
         uint8_t(*input_in)[input_height][input_channels] = inputs[i];
         uint8_t *map = pipe_buffer_map(pcontext, &input_tensor->base,
                                        PIPE_MAP_WRITE, &transfer_out);

         DBG("Converting data\n");

         if (input_channels == 1) {
            unsigned n = 0;
            for (int x = 0; x < input_width; x++) {
               for (int y = 0; y < MAX2(input_height, FEATURE_ATOMIC_SIZE); y++) {
                  if (y < input_height)
                     map[n++] = input_in[x][y][0];
                  else
                     map[n++] = zero_point;
               }
            }
#ifdef __aarch64__
         } else if (input_channels == 3) {
            /*
             * Fast path for 3-channel RGB input (e.g. 224x224x3).
             * NPU format: for each pixel, write 3 channels + 13 padding bytes
             * (FEATURE_ATOMIC_SIZE=16), with 0x80 bias subtraction.
             */
            uint8_t pad_val = (uint8_t)(zero_point - 0x80);
            uint8x16_t pad_vec = vdupq_n_u8(pad_val);
            uint8x16_t bias = vdupq_n_u8(0x80);
            unsigned n = 0;

            for (int x = 0; x < input_width; x++) {
               int y = 0;
               /* Process 8 pixels at a time with NEON */
               for (; y + 7 < input_height; y += 8) {
                  for (int p = 0; p < 8; p++) {
                     /* Store pad first, then overwrite first 3 bytes */
                     vst1q_u8(map + n, pad_vec);
                     map[n + 0] = input_in[x][y + p][0] - 0x80;
                     map[n + 1] = input_in[x][y + p][1] - 0x80;
                     map[n + 2] = input_in[x][y + p][2] - 0x80;
                     n += FEATURE_ATOMIC_SIZE;
                  }
               }
               /* Remaining pixels */
               for (; y < input_height; y++) {
                  vst1q_u8(map + n, pad_vec);
                  map[n + 0] = input_in[x][y][0] - 0x80;
                  map[n + 1] = input_in[x][y][1] - 0x80;
                  map[n + 2] = input_in[x][y][2] - 0x80;
                  n += FEATURE_ATOMIC_SIZE;
               }
            }
#endif
         } else {
            unsigned n = 0;
            for (int u = 0; u < DIV_ROUND_UP(input_channels, FEATURE_ATOMIC_SIZE);
                 u++) {
               for (int x = 0; x < input_width; x++) {
                  for (int y = 0; y < input_height; y++) {
                     unsigned base_channel = u * FEATURE_ATOMIC_SIZE;
                     unsigned real_channels = MIN2(FEATURE_ATOMIC_SIZE,
                                                   input_channels - base_channel);
                     for (int c = 0; c < real_channels; c++) {
                        map[n++] = input_in[x][y][base_channel + c] - 0x80;
                     }
                     /* Pad remaining with zero_point - 0x80 */
                     uint8_t pad = zero_point - 0x80;
                     for (int c = real_channels; c < FEATURE_ATOMIC_SIZE; c++) {
                        map[n++] = pad;
                     }
                  }
               }
            }
         }

         if (DBG_ENABLED(ROCKET_DBG_DUMP_BOS))
            rkt_dump_buffer(map, "input", 0, 0, 0,
                            rkt_get_tensor(subgraph, input_idxs[i])->bo_size);

         DBG("Converted data\n");

         pipe_buffer_unmap(pcontext, transfer_out);
      }
   }
   DBG("Processed input\n");

   DBG("Submitting graph\n");

   /* Execute segments: HW batches submitted to NPU, SW ops on CPU */
   for (unsigned s = 0; s < subgraph->exec_segment_count; s++) {
      struct rkt_exec_segment *seg = &subgraph->exec_segments[s];
      if (seg->is_hw) {
         ret = drmIoctl(screen->fd, DRM_IOCTL_ROCKET_SUBMIT, &seg->submit);
         assert(ret == 0);

         /* Apply per-axis scale correction to CONV outputs that used per-axis weights */
         struct rkt_operation *ops_arr = util_dynarray_begin(&subgraph->operations);
         for (unsigned j = seg->first_op; j < seg->first_op + seg->op_count; j++)
            apply_per_axis_correction(pcontext, subgraph, &ops_arr[j]);

         /* Compact per-group outputs: move from sparse 2x-spaced layout
          * to contiguous layout. Only run once per output tensor (on the
          * last group for each tensor). */
         for (unsigned j = seg->first_op; j < seg->first_op + seg->op_count; j++) {
            struct rkt_operation *op = &ops_arr[j];
            if (op->output_tensor_channels == 0)
               continue;
            /* Check if this is the last group for this output tensor */
            bool is_last = (j + 1 >= seg->first_op + seg->op_count) ||
                           ops_arr[j + 1].output_index != op->output_index;
            if (is_last)
               compact_per_group_output(pcontext, subgraph, op);
         }
      } else {
         execute_sw_op(pcontext, subgraph, seg->first_op);
      }
   }

   DBG("Submitted graph\n");
}

void
rkt_ml_subgraph_read_outputs(struct pipe_context *pcontext,
                             struct pipe_ml_subgraph *psubgraph,
                             unsigned outputs_count,
                             unsigned output_idxs[], void *outputs[],
                             bool is_signed[])
{
   struct rkt_ml_subgraph *subgraph = (struct rkt_ml_subgraph *)(psubgraph);

   DBG("Processing output\n");

   for (int i = 0; i < outputs_count; i++) {

      struct rkt_operation *operation = find_producer(subgraph, output_idxs[i]);
      struct rkt_resource *output_tensor =
         rkt_get_tensor(subgraph, output_idxs[i]);
      struct pipe_transfer *transfer = NULL;
      uint8_t *raw_output;

      DBG("Before pipe_buffer_map\n");
      raw_output = pipe_buffer_map(pcontext, &output_tensor->base, PIPE_MAP_READ,
                                   &transfer);
      DBG("After pipe_buffer_map\n");

      DBG("Converting data\n");

      unsigned ow = operation->output_width;
      unsigned oh = operation->output_height;
      unsigned oc_total = operation->output_channels;
      unsigned groups = DIV_ROUND_UP(oc_total, FEATURE_ATOMIC_SIZE);

      uint8_t(*output_out)[ow][oc_total] = (void *)outputs[i];

      if (subgraph->sw_only) {
         /* SW-only: data is flat NHWC, no bias — just copy */
         memcpy(outputs[i], raw_output, ow * oh * oc_total);
      } else if (groups == 1) {
         /* Common case: output_channels <= 16 (final classification layer).
          * NPU layout is [1][oh][ow][16], output is [oh][ow][oc_total].
          */
         uint8_t *src = raw_output;
         for (int y = 0; y < oh; y++) {
            for (int x = 0; x < ow; x++) {
#ifdef __aarch64__
               if (oc_total == FEATURE_ATOMIC_SIZE) {
                  uint8x16_t v = vld1q_u8(src);
                  v = vaddq_u8(v, vdupq_n_u8(0x80));
                  vst1q_u8(&output_out[y][x][0], v);
               } else {
                  for (int c = 0; c < oc_total; c++)
                     output_out[y][x][c] = src[c] + 0x80;
               }
#else
               for (int c = 0; c < oc_total; c++)
                  output_out[y][x][c] = src[c] + 0x80;
#endif
               src += FEATURE_ATOMIC_SIZE;
            }
         }
      } else {
         /* Multi-group case: scatter channels from each group.
          * Read sequentially from NPU, scatter-write to output.
          * NPU layout: each group at stride oh*ow*FEATURE_ATOMIC_SIZE.
          * (calc_raw_output_size allocates 2x groups for HW alignment)
          */
         for (unsigned g = 0; g < groups; g++) {
            unsigned base_c = g * FEATURE_ATOMIC_SIZE;
            unsigned real_c = MIN2(FEATURE_ATOMIC_SIZE, oc_total - base_c);
            uint8_t *group_base = raw_output + g * oh * ow * FEATURE_ATOMIC_SIZE;

            for (int y = 0; y < oh; y++) {
               for (int x = 0; x < ow; x++) {
                  uint8_t *src = group_base + (y * ow + x) * FEATURE_ATOMIC_SIZE;
                  for (int c = 0; c < real_c; c++)
                     output_out[y][x][base_c + c] = src[c] + 0x80;
               }
            }
         }
      }

      if (DBG_ENABLED(ROCKET_DBG_DUMP_BOS))
         rkt_dump_buffer(raw_output, "output", 0, 0, 0, output_tensor->bo_size);

      DBG("Converted data\n");

      pipe_buffer_unmap(pcontext, transfer);
   }

   DBG("Processed output\n");
}

static void
free_operation(struct rkt_operation *operation)
{
   free(operation->per_axis_correction);
   if (operation->type == RKT_OP_CONVOLUTION) {
      util_dynarray_fini(&operation->tasks);
      pipe_resource_reference(&operation->regcmd, NULL);
      pipe_resource_reference(&operation->weights, NULL);
      pipe_resource_reference(&operation->biases, NULL);
   } else if (operation->type == RKT_OP_CONCATENATION) {
      free(operation->sw.concat.input_indices);
      free(operation->sw.concat.input_channels_arr);
   }
}

void
rkt_ml_subgraph_destroy(struct pipe_context *context,
                        struct pipe_ml_subgraph *psubgraph)
{
   struct rkt_ml_subgraph *subgraph = (struct rkt_ml_subgraph *)(psubgraph);

   free(subgraph->cached_tasks);
   for (unsigned i = 0; i < subgraph->cached_job_count; i++) {
      free(subgraph->cached_in_handles[i]);
      free(subgraph->cached_out_handles_arr[i]);
   }
   free(subgraph->cached_in_handles);
   free(subgraph->cached_out_handles_arr);
   free(subgraph->cached_jobs);
   free(subgraph->exec_segments);

   util_dynarray_foreach (&subgraph->operations, struct rkt_operation, operation)
      free_operation(operation);
   util_dynarray_fini(&subgraph->operations);

   util_dynarray_foreach (&subgraph->tensors, struct pipe_resource *, tensor)
      if (tensor)
         pipe_resource_reference(tensor, NULL);
   util_dynarray_fini(&subgraph->tensors);

   free(subgraph);
}
