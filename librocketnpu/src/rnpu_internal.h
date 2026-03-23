/*
 * Internal types for librocketnpu
 * SPDX-License-Identifier: MIT
 */

#ifndef RNPU_INTERNAL_H
#define RNPU_INTERNAL_H

#include <stdint.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* DRM structs — must match kernel UAPI */
struct drm_rocket_task {
   uint32_t regcmd;
   uint32_t regcmd_count;
};

struct drm_rocket_job {
   uint64_t tasks;
   uint64_t in_bo_handles;
   uint64_t out_bo_handles;
   uint32_t task_count;
   uint32_t task_struct_size;
   uint32_t in_bo_handle_count;
   uint32_t out_bo_handle_count;
};

struct drm_rocket_submit {
   uint64_t jobs;
   uint32_t job_count;
   uint32_t job_struct_size;
   uint64_t reserved;
};

#include "rnpu_drm.h"

/* NPU hardware constants (NVDLA-derived) */
#ifndef CBUF_BANK_SIZE
#define CBUF_BANK_SIZE        32768
#endif
#ifndef CBUF_BANKS
#define CBUF_BANKS            12
#endif
#ifndef CBUF_ENTRIES_PER_BANK
#define CBUF_ENTRIES_PER_BANK 256
#endif
#ifndef CBUF_ENTRY_SIZE
#define CBUF_ENTRY_SIZE       (CBUF_BANK_SIZE / CBUF_ENTRIES_PER_BANK)
#endif
#ifndef FEATURE_ATOMIC_SIZE
#define FEATURE_ATOMIC_SIZE   16
#endif
#ifndef WEIGHT_ATOMIC_SIZE
#define WEIGHT_ATOMIC_SIZE    32
#endif
#ifndef ATOMIC_K_SIZE
#define ATOMIC_K_SIZE         16
#endif

#define PER_AXIS_GROUP_SIZE   1

#ifndef ALIGN_UP
#define ALIGN_UP(x, a)  (((x) + (a) - 1) & ~((a) - 1))
#endif
#ifndef DIV_ROUND_UP
#define DIV_ROUND_UP(n, d) (((n) + (d) - 1) / (d))
#endif
#ifndef MIN2
#define MIN2(a, b) ((a) < (b) ? (a) : (b))
#endif
#ifndef MAX2
#define MAX2(a, b) ((a) > (b) ? (a) : (b))
#endif

/* NPU tensor offset: interleaved format (x-major) */
#define NPU_OFFSET(g, x, y, w, h) \
   ((g) * (w) * (h) * FEATURE_ATOMIC_SIZE + \
    (x) * (h) * FEATURE_ATOMIC_SIZE + \
    (y) * FEATURE_ATOMIC_SIZE)

/* IEEE754 float-to-uint32 reinterpret (for scale extraction) */
static inline uint32_t fui(float f) {
   union { float f; uint32_t u; } u;
   u.f = f;
   return u.u;
}

/* ---- TFLite parsed representation ---- */

struct rnpu_tfl_quant {
   float *scales;
   int64_t *zero_points;
   float scale;
   int32_t zero_point;
   unsigned num_scales;
};

struct rnpu_tfl_tensor {
   int32_t shape[4];
   int shape_len;
   int type;
   unsigned buffer_index;
   struct rnpu_tfl_quant quant;
};

enum rnpu_tfl_opcode {
   TFLITE_OP_ADD = 0,
   TFLITE_OP_CONCATENATION = 2,
   TFLITE_OP_CONV_2D = 3,
   TFLITE_OP_DEPTHWISE_CONV_2D = 4,
   TFLITE_OP_AVERAGE_POOL_2D = 1,
   TFLITE_OP_LOGISTIC = 14,
   TFLITE_OP_MAX_POOL_2D = 17,
   TFLITE_OP_RESHAPE = 22,
   TFLITE_OP_SOFTMAX = 25,
   TFLITE_OP_PAD = 34,
   TFLITE_OP_RESIZE_NEAREST_NEIGHBOR = 97,
};

struct rnpu_tfl_op {
   int builtin_code;
   int *inputs;
   int *outputs;
   int input_count;
   int output_count;
   union {
      struct { int padding; int stride_w, stride_h; int dilation_w, dilation_h; } conv;
      struct { int padding; int stride_w, stride_h; int depth_multiplier;
               int dilation_w, dilation_h; } dw_conv;
      struct { int padding; int stride_w, stride_h; int filter_w, filter_h; } pool;
      struct { int axis; } concat;
   } opt;
};

struct rnpu_tfl_buffer {
   const uint8_t *data;
   uint32_t size;
};

struct rnpu_tfl_model {
   struct rnpu_tfl_tensor *tensors;
   unsigned tensor_count;
   struct rnpu_tfl_op *ops;
   unsigned op_count;
   struct rnpu_tfl_buffer *buffers;
   unsigned buffer_count;
   int *graph_inputs;
   int *graph_outputs;
   unsigned input_count;
   unsigned output_count;
   uint8_t *file_data;
   size_t file_size;
};

/* ---- NPU internal representation ---- */

enum rnpu_op_type {
   RNPU_OP_CONV,
   RNPU_OP_CONCAT,
   RNPU_OP_MAX_POOL,
   RNPU_OP_PAD,
   RNPU_OP_RESIZE_NEAREST,
   RNPU_OP_LOGISTIC,
   RNPU_OP_AVG_POOL,
   RNPU_OP_RESHAPE,
   RNPU_OP_SOFTMAX,
};

struct rnpu_split_task {
   unsigned num;
   unsigned top_slice, bottom_slice;
   unsigned num_overlap_slices, num_retain_slices;
   unsigned convolutions;
   unsigned pad_top, pad_bottom, pad_left, pad_right;
   unsigned stride_x, stride_y;
   unsigned input_width, input_height, input_channels, input_channels_real;
   unsigned input_zero_point;
   float input_scale;
   unsigned input_data_entries;
   int input_line_stride, input_surface_stride;
   unsigned input_offset;
   unsigned output_width, output_height, output_channels, output_channels_real;
   unsigned output_zero_point;
   float output_scale;
   int output_surface_stride;
   unsigned output_offset;
   unsigned weights_width, weights_height, weights_kernels;
   unsigned weights_zero_point;
   float weights_scale;
   unsigned input_banks, weights_banks;
   unsigned atomic_count, surfaces_per_row;
   unsigned regcfg_amount;
   uint32_t regcfg_addr;
};

struct rnpu_operation {
   enum rnpu_op_type type;
   bool depthwise;
   bool reuse_weights_cbuf;
   unsigned truncate_bits;
   bool padding_same;
   unsigned stride;
   bool addition_input;
   int addition_offset;
   float addition_scale;

   unsigned input_tensor;
   unsigned output_tensor;
   int add_tensor;

   unsigned input_width, input_height, input_channels;
   uint8_t input_zero_point;
   float input_scale;

   unsigned output_width, output_height, output_channels;
   uint8_t output_zero_point;
   float output_scale;

   unsigned weights_width, weights_height;
   uint8_t weights_zero_point;
   float weights_scale;

   /* Per-axis quantization */
   float *per_axis_correction;
   unsigned output_tensor_channels;
   unsigned per_channel_group_offset;
   unsigned *group_channel_indices;
   int32_t per_channel_bias;

   /* Offsets into shared BOs */
   uint32_t weight_offset;
   uint32_t weight_size;
   uint32_t bias_offset;
   uint32_t bias_size;
   uint32_t regcmd_offset;
   uint32_t regcmd_size;

   struct rnpu_split_task *tasks;
   unsigned task_count;

   /* SW op parameters */
   union {
      struct {
         unsigned *input_tensors;
         unsigned *input_channels_arr;
         unsigned input_count;
      } concat;
      struct {
         unsigned filter_width, filter_height;
         unsigned stride_x, stride_y;
         bool padding_same;
      } pool;
      struct {
         unsigned pad_before_w, pad_after_w;
         unsigned pad_before_h, pad_after_h;
      } pad;
      struct {
         uint8_t lut[256];
         uint8_t raw_lut[256];
      } logistic;
      struct {
         float in_scale;
         int in_zp;
         float out_scale;
         int out_zp;
      } softmax;
   } sw;
};

struct rnpu_npu_tensor {
   unsigned width, height, channels;
   float scale;
   int32_t zero_point;
   uint32_t offset;  /* byte offset in activation_bo */
   uint32_t size;
};

struct rnpu_exec_segment {
   bool is_hw;
   unsigned first_op;
   unsigned op_count;
   unsigned job_count;  /* number of merged jobs in this HW segment */
};

struct rnpu_model {
   int fd;

   /* Shared BOs — all pre-allocated at load */
   struct rnpu_bo weight_bo;
   struct rnpu_bo bias_bo;
   struct rnpu_bo regcmd_bo;
   struct rnpu_bo activation_bo;

   /* Tensors */
   struct rnpu_npu_tensor *tensors;
   unsigned tensor_count;

   /* Operations */
   struct rnpu_operation *ops;
   unsigned op_count;

   /* Execution plan */
   struct rnpu_exec_segment *segments;
   unsigned segment_count;

   /* Pre-built submit data */
   struct drm_rocket_job *jobs;
   struct drm_rocket_task *hw_tasks;
   uint32_t *in_handles;   /* flat array: per-job input handles */
   uint32_t *out_handles;  /* flat array: per-job output handles */
   unsigned job_count;

   /* I/O info */
   unsigned graph_input_tensor;
   unsigned *graph_output_tensors;
   unsigned output_count;
   bool sw_only;

   /* TFLite model data (kept alive for weight references) */
   struct rnpu_tfl_model tfl;
};

#endif /* RNPU_INTERNAL_H */
