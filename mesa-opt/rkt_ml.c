/*
 * Copyright (c) 2024 Tomeu Vizoso <tomeu@tomeuvizoso.net>
 * SPDX-License-Identifier: MIT
 */

#include "pipe/p_state.h"
#include "util/macros.h"
#include "util/u_dynarray.h"
#include "util/u_inlines.h"

#include <xf86drm.h>

#include "drm-uapi/rocket_accel.h"

#include "rkt_coefs.h"
#include "rkt_ml.h"
#include "rkt_regcmd.h"
#include "rkt_task.h"

#ifdef __aarch64__
#include <arm_neon.h>
#endif

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
      assert(size == pipe_buffer_size(res));
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
   unsigned output_channels_1 =
      DIV_ROUND_UP(operation->output_channels, FEATURE_ATOMIC_SIZE) * 2;
   unsigned output_channels_2 = FEATURE_ATOMIC_SIZE;

   return operation->output_width * operation->output_height *
          output_channels_1 * output_channels_2;
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

   /* Mark regcmd, weights, biases as device-resident */
   rkt_resource(operation->regcmd)->device_resident = true;
   rkt_resource(operation->weights)->device_resident = true;
   rkt_resource(operation->biases)->device_resident = true;
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
 * Build cached per-operation job structures.
 * Each operation gets one job with all its tasks (batched submission).
 * This avoids per-invoke malloc/free while keeping multi-core parallelism.
 */
static void
build_cached_submit(struct rkt_ml_subgraph *subgraph)
{
   unsigned num_ops = util_dynarray_num_elements(&subgraph->operations,
                                                  struct rkt_operation);

   /* Count total tasks for allocation */
   unsigned total_tasks = 0;
   util_dynarray_foreach(&subgraph->operations, struct rkt_operation, op) {
      total_tasks += util_dynarray_num_elements(&op->tasks, struct split_task);
   }

   subgraph->cached_tasks = calloc(total_tasks, sizeof(struct drm_rocket_task));
   subgraph->cached_task_count = total_tasks;

   /* Allocate per-operation job array and handle arrays */
   subgraph->cached_job_count = num_ops;
   subgraph->cached_jobs = calloc(num_ops, sizeof(struct drm_rocket_job));
   subgraph->cached_in_handles = calloc(num_ops, sizeof(uint32_t *));
   subgraph->cached_out_handles_arr = calloc(num_ops, sizeof(uint32_t *));

   /* Fill task array and build per-operation jobs */
   unsigned task_offset = 0;
   unsigned op_idx = 0;
   util_dynarray_foreach(&subgraph->operations, struct rkt_operation, op) {
      unsigned num_tasks = util_dynarray_num_elements(&op->tasks, struct split_task);

      /* Fill tasks for this operation */
      util_dynarray_foreach(&op->tasks, struct split_task, task) {
         subgraph->cached_tasks[task_offset].regcmd = task->regcfg_addr;
         subgraph->cached_tasks[task_offset].regcmd_count = task->regcfg_amount;
         task_offset++;
      }

      /* Allocate BO handle arrays */
      unsigned num_inputs = op->add_tensor != -1 ? 2 : 1;
      subgraph->cached_in_handles[op_idx] = calloc(num_inputs, sizeof(uint32_t));
      subgraph->cached_in_handles[op_idx][0] =
         rkt_get_tensor(subgraph, op->input_index)->handle;
      if (op->add_tensor != -1)
         subgraph->cached_in_handles[op_idx][1] =
            rkt_get_tensor(subgraph, op->add_tensor)->handle;

      subgraph->cached_out_handles_arr[op_idx] = malloc(sizeof(uint32_t));
      subgraph->cached_out_handles_arr[op_idx][0] =
         rkt_get_tensor(subgraph, op->output_index)->handle;

      /* Build job struct */
      struct drm_rocket_job *job = &subgraph->cached_jobs[op_idx];
      job->task_struct_size = sizeof(struct drm_rocket_task);
      job->tasks = (uint64_t)(uintptr_t)&subgraph->cached_tasks[task_offset - num_tasks];
      job->task_count = num_tasks;
      job->in_bo_handles = (uint64_t)(uintptr_t)subgraph->cached_in_handles[op_idx];
      job->in_bo_handle_count = num_inputs;
      job->out_bo_handles = (uint64_t)(uintptr_t)subgraph->cached_out_handles_arr[op_idx];
      job->out_bo_handle_count = 1;

      op_idx++;
   }

   /* Save graph input index for later */
   struct rkt_operation *first_op = util_dynarray_begin(&subgraph->operations);
   subgraph->graph_input_index = first_op->input_index;

   /* Build submit struct */
   subgraph->cached_submit.job_struct_size = sizeof(struct drm_rocket_job);
   subgraph->cached_submit.jobs = (uint64_t)(uintptr_t)subgraph->cached_jobs;
   subgraph->cached_submit.job_count = num_ops;
   subgraph->cached_submit.reserved = 0;

   subgraph->submit_built = true;
}

static void
lower_convolution(struct rkt_ml_subgraph *subgraph,
                  const struct pipe_ml_operation *poperation,
                  struct rkt_operation *operation)
{
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
   operation->output_channels = poperation->output_tensors[0]->dims[3];
   operation->output_zero_point = poperation->output_tensors[0]->zero_point;
   operation->output_scale = poperation->output_tensors[0]->scale;

   operation->weights_width = poperation->conv.weight_tensor->dims[1];
   operation->weights_height = poperation->conv.weight_tensor->dims[2];
   operation->weights_zero_point = poperation->conv.weight_tensor->zero_point;
   operation->weights_scale = poperation->conv.weight_tensor->scale;

   operation->weights = rkt_fill_weights(subgraph, poperation);
   operation->biases =
      rkt_fill_biases(subgraph, poperation, &operation->truncate_bits);
}

static struct rkt_operation *
find_first_consumer(struct rkt_ml_subgraph *subgraph, unsigned tensor_index)
{
   util_dynarray_foreach (&subgraph->operations, struct rkt_operation,
                          operation) {
      if (operation->input_index == tensor_index)
         return operation;
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

      // Dilation and per-axis quantization not yet implemented
      if (tensor_quantization_supported(input_tensor) &&
          tensor_quantization_supported(weight_tensor) &&
          tensor_quantization_supported(bias_tensor) &&
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
         lower_convolution(subgraph, &poperations[i], &operation);
         util_dynarray_append(&subgraph->operations, operation);
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

   /* Mark graph input tensor as cpu_write_only (skip PREP_BO) */
   struct rkt_operation *first_op = util_dynarray_begin(&subgraph->operations);
   rkt_get_tensor(subgraph, first_op->input_index)->cpu_write_only = true;

   /* Compile */
   util_dynarray_foreach (&subgraph->operations, struct rkt_operation,
                          operation) {
      rkt_split_tasks(subgraph, operation);
      compile_operation(subgraph, operation);
   }

   /* Pre-build per-operation job structures for cached submit */
   build_cached_submit(subgraph);

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

      struct rkt_resource *input_tensor =
         rkt_get_tensor(subgraph, operation->input_index);
      if (output_channels == 1 && input_channels == 1 &&
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

   /* Use pre-built per-operation jobs (zero malloc per invoke) */
   ret = drmIoctl(screen->fd, DRM_IOCTL_ROCKET_SUBMIT, &subgraph->cached_submit);
   assert(ret == 0);

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

      /*
       * Reordered output conversion: iterate in (g, y, x, c) order
       * for sequential reads from NPU's interleaved format.
       * NPU layout: [group][height][width][FEATURE_ATOMIC_SIZE]
       */
      if (groups == 1) {
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
   util_dynarray_fini(&operation->tasks);
   pipe_resource_reference(&operation->regcmd, NULL);
   pipe_resource_reference(&operation->weights, NULL);
   pipe_resource_reference(&operation->biases, NULL);
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

   util_dynarray_foreach (&subgraph->operations, struct rkt_operation, operation)
      free_operation(operation);
   util_dynarray_fini(&subgraph->operations);

   util_dynarray_foreach (&subgraph->tensors, struct pipe_resource *, tensor)
      if (tensor)
         pipe_resource_reference(tensor, NULL);
   util_dynarray_fini(&subgraph->tensors);

   free(subgraph);
}
