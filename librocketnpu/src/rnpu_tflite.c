/*
 * TFLite FlatBuffer parser — reads .tflite binary directly
 * Supports schema v3 (stable since TFLite 2.0).
 * SPDX-License-Identifier: MIT
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>

#include "rnpu_tflite.h"

/* FlatBuffer primitives */
static inline uint32_t fb_u32(const uint8_t *b, uint32_t off) {
   return b[off] | (b[off+1]<<8) | (b[off+2]<<16) | (b[off+3]<<24);
}
static inline int32_t fb_i32(const uint8_t *b, uint32_t off) {
   return (int32_t)fb_u32(b, off);
}
static inline uint16_t fb_u16(const uint8_t *b, uint32_t off) {
   return b[off] | (b[off+1]<<8);
}
static inline int64_t fb_i64(const uint8_t *b, uint32_t off) {
   uint64_t lo = fb_u32(b, off);
   uint64_t hi = fb_u32(b, off + 4);
   return (int64_t)(lo | (hi << 32));
}
static inline float fb_f32(const uint8_t *b, uint32_t off) {
   union { uint32_t u; float f; } v;
   v.u = fb_u32(b, off);
   return v.f;
}

/* Dereference: follow a uoffset_t at 'off' */
static inline uint32_t fb_deref(const uint8_t *b, uint32_t off) {
   return off + fb_u32(b, off);
}

/* Get absolute offset of field 'field' in table at 'table'. Returns 0 if absent. */
static uint32_t fb_field(const uint8_t *b, uint32_t table, int field) {
   int32_t vt_off = fb_i32(b, table);
   uint32_t vt = table - vt_off;
   uint16_t vt_size = fb_u16(b, vt);
   uint16_t byte_off = 4 + field * 2;
   if (byte_off >= vt_size) return 0;
   uint16_t f_off = fb_u16(b, vt + byte_off);
   if (f_off == 0) return 0;
   return table + f_off;
}

/* Vector length at absolute vector position */
static uint32_t fb_vec_len(const uint8_t *b, uint32_t vec) {
   return fb_u32(b, vec);
}

/* BuiltinOptions type indices (from TFLite schema) */
#define OPT_CONV2D                1
#define OPT_DEPTHWISE_CONV2D      2
#define OPT_CONCATENATION         3
#define OPT_ADD                   18
#define OPT_POOL2D                5
#define OPT_POOL2D_COMPAT         22
#define OPT_PAD                   28
#define OPT_RESIZE_NEAREST        42

static void parse_quant(const uint8_t *b, uint32_t qtable, struct rnpu_tfl_quant *q)
{
   memset(q, 0, sizeof(*q));
   if (!qtable) return;

   /* field 2: scale (vector of float) */
   uint32_t f = fb_field(b, qtable, 2);
   if (f) {
      uint32_t vec = fb_deref(b, f);
      unsigned n = fb_vec_len(b, vec);
      if (n == 1) {
         q->scale = fb_f32(b, vec + 4);
      } else if (n > 1) {
         q->scales = calloc(n, sizeof(float));
         q->num_scales = n;
         for (unsigned i = 0; i < n; i++)
            q->scales[i] = fb_f32(b, vec + 4 + i * 4);
         q->scale = q->scales[0]; /* fallback */
      }
   }

   /* field 3: zero_point (vector of int64) */
   f = fb_field(b, qtable, 3);
   if (f) {
      uint32_t vec = fb_deref(b, f);
      unsigned n = fb_vec_len(b, vec);
      if (n == 1) {
         q->zero_point = (int32_t)fb_i64(b, vec + 4);
      } else if (n > 1) {
         q->zero_points = calloc(n, sizeof(int64_t));
         for (unsigned i = 0; i < n; i++)
            q->zero_points[i] = fb_i64(b, vec + 4 + i * 8);
         q->zero_point = (int32_t)q->zero_points[0];
      }
   }
}

static int parse_tensor(const uint8_t *b, uint32_t ttable, struct rnpu_tfl_tensor *t)
{
   memset(t, 0, sizeof(*t));

   /* field 0: shape (vector of int32) */
   uint32_t f = fb_field(b, ttable, 0);
   if (f) {
      uint32_t vec = fb_deref(b, f);
      t->shape_len = MIN2(fb_vec_len(b, vec), 4);
      for (int i = 0; i < t->shape_len; i++)
         t->shape[i] = fb_i32(b, vec + 4 + i * 4);
   }

   /* field 1: type (int8 enum: 0=FLOAT32, 3=UINT8, 9=INT8, 2=INT32) */
   f = fb_field(b, ttable, 1);
   if (f) t->type = b[f];

   /* field 2: buffer index */
   f = fb_field(b, ttable, 2);
   if (f) t->buffer_index = fb_u32(b, f);

   /* field 4: quantization */
   f = fb_field(b, ttable, 4);
   if (f) {
      uint32_t qt = fb_deref(b, f);
      parse_quant(b, qt, &t->quant);
   }

   return 0;
}

static void parse_conv_options(const uint8_t *b, uint32_t ot, struct rnpu_tfl_op *op)
{
   uint32_t f;
   f = fb_field(b, ot, 0); op->opt.conv.padding = f ? b[f] : 0; /* default SAME=0 */
   f = fb_field(b, ot, 1); op->opt.conv.stride_w = f ? fb_i32(b, f) : 1;
   f = fb_field(b, ot, 2); op->opt.conv.stride_h = f ? fb_i32(b, f) : 1;
   f = fb_field(b, ot, 4); op->opt.conv.dilation_w = f ? fb_i32(b, f) : 1;
   f = fb_field(b, ot, 5); op->opt.conv.dilation_h = f ? fb_i32(b, f) : 1;
}

static void parse_dw_conv_options(const uint8_t *b, uint32_t ot, struct rnpu_tfl_op *op)
{
   uint32_t f;
   f = fb_field(b, ot, 0); op->opt.dw_conv.padding = f ? b[f] : 0; /* default SAME=0 */
   f = fb_field(b, ot, 1); op->opt.dw_conv.stride_w = f ? fb_i32(b, f) : 1;
   f = fb_field(b, ot, 2); op->opt.dw_conv.stride_h = f ? fb_i32(b, f) : 1;
   f = fb_field(b, ot, 3); op->opt.dw_conv.depth_multiplier = f ? fb_i32(b, f) : 1;
   f = fb_field(b, ot, 5); op->opt.dw_conv.dilation_w = f ? fb_i32(b, f) : 1;
   f = fb_field(b, ot, 6); op->opt.dw_conv.dilation_h = f ? fb_i32(b, f) : 1;
}

static void parse_pool_options(const uint8_t *b, uint32_t ot, struct rnpu_tfl_op *op)
{
   uint32_t f;
   f = fb_field(b, ot, 0); op->opt.pool.padding = f ? b[f] : 0; /* default SAME=0 */
   f = fb_field(b, ot, 1); op->opt.pool.stride_w = f ? fb_i32(b, f) : 1;
   f = fb_field(b, ot, 2); op->opt.pool.stride_h = f ? fb_i32(b, f) : 1;
   f = fb_field(b, ot, 3); op->opt.pool.filter_w = f ? fb_i32(b, f) : 1;
   f = fb_field(b, ot, 4); op->opt.pool.filter_h = f ? fb_i32(b, f) : 1;
}

static void parse_concat_options(const uint8_t *b, uint32_t ot, struct rnpu_tfl_op *op)
{
   uint32_t f;
   f = fb_field(b, ot, 0); op->opt.concat.axis = f ? fb_i32(b, f) : 0;
}

static int parse_operator(const uint8_t *b, uint32_t otable,
                          const int *opcode_map, unsigned opcode_count,
                          struct rnpu_tfl_op *op)
{
   memset(op, 0, sizeof(*op));

   /* field 0: opcode_index */
   uint32_t f = fb_field(b, otable, 0);
   unsigned oci = f ? fb_u32(b, f) : 0;
   op->builtin_code = (oci < opcode_count) ? opcode_map[oci] : -1;

   /* field 1: inputs */
   f = fb_field(b, otable, 1);
   if (f) {
      uint32_t vec = fb_deref(b, f);
      op->input_count = fb_vec_len(b, vec);
      op->inputs = calloc(op->input_count, sizeof(int));
      for (int i = 0; i < op->input_count; i++)
         op->inputs[i] = fb_i32(b, vec + 4 + i * 4);
   }

   /* field 2: outputs */
   f = fb_field(b, otable, 2);
   if (f) {
      uint32_t vec = fb_deref(b, f);
      op->output_count = fb_vec_len(b, vec);
      op->outputs = calloc(op->output_count, sizeof(int));
      for (int i = 0; i < op->output_count; i++)
         op->outputs[i] = fb_i32(b, vec + 4 + i * 4);
   }

   /* field 3: builtin_options_type, field 4: builtin_options */
   uint32_t ot_type_f = fb_field(b, otable, 3);
   uint32_t ot_f = fb_field(b, otable, 4);
   if (ot_type_f && ot_f) {
      uint8_t ot_type = b[ot_type_f];
      uint32_t ot = fb_deref(b, ot_f);
      switch (ot_type) {
      case OPT_CONV2D: parse_conv_options(b, ot, op); break;
      case OPT_DEPTHWISE_CONV2D: parse_dw_conv_options(b, ot, op); break;
      case OPT_POOL2D: case OPT_POOL2D_COMPAT: parse_pool_options(b, ot, op); break;
      case OPT_CONCATENATION: parse_concat_options(b, ot, op); break;
      default: break;
      }
   }

   return 0;
}

int rnpu_tflite_parse(const char *path, struct rnpu_tfl_model *model)
{
   memset(model, 0, sizeof(*model));

   FILE *f = fopen(path, "rb");
   if (!f) {
      fprintf(stderr, "rnpu: cannot open %s: %s\n", path, strerror(errno));
      return -1;
   }
   fseek(f, 0, SEEK_END);
   model->file_size = ftell(f);
   fseek(f, 0, SEEK_SET);
   model->file_data = malloc(model->file_size);
   if (fread(model->file_data, 1, model->file_size, f) != model->file_size) {
      fclose(f);
      free(model->file_data);
      return -1;
   }
   fclose(f);

   const uint8_t *b = model->file_data;

   /* Root table = Model */
   uint32_t root = fb_deref(b, 0);

   /* field 1: operator_codes — build opcode map */
   int *opcode_map = NULL;
   unsigned opcode_count = 0;
   {
      uint32_t fld = fb_field(b, root, 1);
      if (fld) {
         uint32_t vec = fb_deref(b, fld);
         opcode_count = fb_vec_len(b, vec);
         opcode_map = calloc(opcode_count, sizeof(int));
         for (unsigned i = 0; i < opcode_count; i++) {
            uint32_t oc = fb_deref(b, vec + 4 + i * 4);
            /* field 3: builtin_code (int32, schema v3+) */
            uint32_t ff = fb_field(b, oc, 3);
            if (ff) {
               opcode_map[i] = fb_i32(b, ff);
            } else {
               /* field 0: deprecated_builtin_code (int8) */
               ff = fb_field(b, oc, 0);
               opcode_map[i] = ff ? (int)(int8_t)b[ff] : 0;
            }
         }
      }
   }

   /* field 4: buffers */
   {
      uint32_t fld = fb_field(b, root, 4);
      if (fld) {
         uint32_t vec = fb_deref(b, fld);
         model->buffer_count = fb_vec_len(b, vec);
         model->buffers = calloc(model->buffer_count, sizeof(struct rnpu_tfl_buffer));
         for (unsigned i = 0; i < model->buffer_count; i++) {
            uint32_t bt = fb_deref(b, vec + 4 + i * 4);
            uint32_t df = fb_field(b, bt, 0);
            if (df) {
               uint32_t dv = fb_deref(b, df);
               model->buffers[i].size = fb_vec_len(b, dv);
               model->buffers[i].data = b + dv + 4;
            }
         }
      }
   }

   /* field 2: subgraphs — parse first subgraph only */
   {
      uint32_t fld = fb_field(b, root, 2);
      if (!fld) { free(opcode_map); return -1; }
      uint32_t vec = fb_deref(b, fld);
      if (fb_vec_len(b, vec) == 0) { free(opcode_map); return -1; }
      uint32_t sg = fb_deref(b, vec + 4);

      /* field 0: tensors */
      uint32_t tf = fb_field(b, sg, 0);
      if (tf) {
         uint32_t tv = fb_deref(b, tf);
         model->tensor_count = fb_vec_len(b, tv);
         model->tensors = calloc(model->tensor_count, sizeof(struct rnpu_tfl_tensor));
         for (unsigned i = 0; i < model->tensor_count; i++) {
            uint32_t tt = fb_deref(b, tv + 4 + i * 4);
            parse_tensor(b, tt, &model->tensors[i]);
         }
      }

      /* field 1: inputs */
      tf = fb_field(b, sg, 1);
      if (tf) {
         uint32_t iv = fb_deref(b, tf);
         model->input_count = fb_vec_len(b, iv);
         model->graph_inputs = calloc(model->input_count, sizeof(int));
         for (unsigned i = 0; i < model->input_count; i++)
            model->graph_inputs[i] = fb_i32(b, iv + 4 + i * 4);
      }

      /* field 2: outputs */
      tf = fb_field(b, sg, 2);
      if (tf) {
         uint32_t ov = fb_deref(b, tf);
         model->output_count = fb_vec_len(b, ov);
         model->graph_outputs = calloc(model->output_count, sizeof(int));
         for (unsigned i = 0; i < model->output_count; i++)
            model->graph_outputs[i] = fb_i32(b, ov + 4 + i * 4);
      }

      /* field 3: operators */
      tf = fb_field(b, sg, 3);
      if (tf) {
         uint32_t ov = fb_deref(b, tf);
         model->op_count = fb_vec_len(b, ov);
         model->ops = calloc(model->op_count, sizeof(struct rnpu_tfl_op));
         for (unsigned i = 0; i < model->op_count; i++) {
            uint32_t ot = fb_deref(b, ov + 4 + i * 4);
            parse_operator(b, ot, opcode_map, opcode_count, &model->ops[i]);
         }
      }
   }

   free(opcode_map);
   return 0;
}

void rnpu_tflite_free(struct rnpu_tfl_model *model)
{
   for (unsigned i = 0; i < model->tensor_count; i++) {
      free(model->tensors[i].quant.scales);
      free(model->tensors[i].quant.zero_points);
   }
   free(model->tensors);
   for (unsigned i = 0; i < model->op_count; i++) {
      free(model->ops[i].inputs);
      free(model->ops[i].outputs);
   }
   free(model->ops);
   free(model->buffers);
   free(model->graph_inputs);
   free(model->graph_outputs);
   free(model->file_data);
}
