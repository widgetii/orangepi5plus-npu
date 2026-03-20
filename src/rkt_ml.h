/*
 * Copyright (c) 2024 Tomeu Vizoso <tomeu@tomeuvizoso.net>
 * SPDX-License-Identifier: MIT
 */

#ifndef RKT_ML_H
#define RKT_ML_H

#include <util/u_dynarray.h>

#include "rkt_device.h"
#include "drm-uapi/rocket_accel.h"

// http://nvdla.org/hw/v1/ias/unit_description.html#convolution-buffer
#define CBUF_BANK_SIZE        32768
#define CBUF_BANKS            12
#define CBUF_ENTRIES_PER_BANK 256
#define CBUF_ENTRY_SIZE       (CBUF_BANK_SIZE / CBUF_ENTRIES_PER_BANK)
#define FEATURE_ATOMIC_SIZE   16
#define WEIGHT_ATOMIC_SIZE    32
#define ATOMIC_K_SIZE         16

enum rkt_op_type {
   RKT_OP_CONVOLUTION,
   RKT_OP_CONCATENATION,
   RKT_OP_MAX_POOL_2D,
   RKT_OP_PAD,
   RKT_OP_RESIZE_NEAREST,
   RKT_OP_LOGISTIC,
};

struct split_task {
   unsigned num;

   unsigned top_slice;
   unsigned bottom_slice;
   unsigned num_overlap_slices;
   unsigned num_retain_slices;
   unsigned convolutions;

   unsigned pad_top;
   unsigned pad_bottom;
   unsigned pad_left;
   unsigned pad_right;

   unsigned stride_x;
   unsigned stride_y;

   unsigned input_width;
   unsigned input_height;
   unsigned input_channels;
   unsigned input_channels_real;
   unsigned input_zero_point;
   float input_scale;
   unsigned input_data_entries;
   int input_line_stride;
   int input_surface_stride;
   unsigned input_offset;

   unsigned output_width;
   unsigned output_height;
   unsigned output_channels;
   unsigned output_channels_real;
   unsigned output_zero_point;
   float output_scale;
   int output_surface_stride;
   unsigned output_offset;

   unsigned weights_width;
   unsigned weights_height;
   unsigned weights_kernels;
   unsigned weights_zero_point;
   float weights_scale;

   unsigned input_banks;
   unsigned weights_banks;

   unsigned atomic_count;
   unsigned surfaces_per_row;

   unsigned regcfg_amount;
   uint32_t regcfg_addr;
};

struct rkt_operation {
   enum rkt_op_type type;

   struct pipe_resource *regcmd;
   struct pipe_resource *weights;
   struct pipe_resource *biases;

   bool depthwise;
   bool reuse_weights_cbuf;
   unsigned truncate_bits;
   bool padding_same;
   unsigned stride;

   bool addition_input;
   int addition_offset;
   float addition_scale;

   unsigned input_index;
   unsigned input_width;
   unsigned input_height;
   unsigned input_channels;
   uint8_t input_zero_point;
   float input_scale;

   unsigned output_index;
   unsigned output_width;
   unsigned output_height;
   unsigned output_channels;
   uint8_t output_zero_point;
   float output_scale;

   unsigned weights_width;
   unsigned weights_height;
   uint8_t weights_zero_point;
   float weights_scale;

   int add_tensor;

   /* Per-axis scale correction: weight_scale[oc] / weight_scale[0].
    * NULL for per-tensor quantized weights (no correction needed). */
   float *per_axis_correction;

   struct util_dynarray tasks; /* struct split_task */

   /* Software op parameters */
   union {
      struct {
         unsigned *input_indices;
         unsigned *input_channels_arr;
         unsigned input_count;
      } concat;
      struct {
         unsigned filter_width;
         unsigned filter_height;
         unsigned stride_x;
         unsigned stride_y;
         bool padding_same;
      } pool;
      struct {
         unsigned pad_before_w;
         unsigned pad_after_w;
         unsigned pad_before_h;
         unsigned pad_after_h;
      } pad;
      struct {
         uint8_t lut[256];     /* NPU byte → NPU byte (for mixed HW/SW) */
         uint8_t raw_lut[256]; /* raw int8 byte → raw int8 byte (for sw_only) */
      } logistic;
   } sw;
};

struct rkt_exec_segment {
   bool is_hw;
   unsigned first_op;
   unsigned op_count;
   struct drm_rocket_submit submit;
};

struct rkt_ml_subgraph {
   struct pipe_ml_subgraph base;

   struct util_dynarray operations; /* rkt_operation */
   struct util_dynarray tensors;    /* pipe_resource* */

   /* Execution segments for mixed HW/SW execution */
   struct rkt_exec_segment *exec_segments;
   unsigned exec_segment_count;

   /* Pre-built HW job data */
   struct drm_rocket_job *cached_jobs;
   struct drm_rocket_task *cached_tasks;
   uint32_t **cached_in_handles;
   uint32_t **cached_out_handles_arr;
   unsigned cached_job_count;
   unsigned cached_task_count;
   unsigned graph_input_index;
   bool sw_only; /* No HW ops — use flat NHWC instead of NPU interleaved */
};

bool
rkt_ml_operation_supported(struct pipe_context *pcontext, const struct pipe_ml_operation *operation);

struct pipe_ml_subgraph *
rkt_ml_subgraph_create(struct pipe_context *pcontext,
                       const struct pipe_ml_operation *poperations,
                       unsigned count);

void rkt_ml_subgraph_invoke(struct pipe_context *pcontext,
                            struct pipe_ml_subgraph *psubgraph,
                            unsigned inputs_count, unsigned input_idxs[],
                            void *inputs[], bool is_signed[]);

void rkt_ml_subgraph_read_outputs(struct pipe_context *pcontext,
                                  struct pipe_ml_subgraph *psubgraph,
                                  unsigned outputs_count,
                                  unsigned output_idxs[], void *outputs[],
                                  bool is_signed[]);

void rkt_ml_subgraph_destroy(struct pipe_context *context,
                             struct pipe_ml_subgraph *psubgraph);

struct rkt_resource *rkt_get_tensor(struct rkt_ml_subgraph *subgraph,
                                    unsigned idx);

bool rkt_is_depthwise(const struct pipe_ml_operation *poperation);

void rkt_dump_buffer(const uint8_t *ptr, char *name, int operation_nr,
                     int suboperation_nr, int offset, unsigned size);

#endif /* RKT_ML_H */
