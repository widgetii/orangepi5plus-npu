/*
 * ONNX model parser for librocketnpu
 *
 * Parses quantized ONNX graphs (with QuantizeLinear/DequantizeLinear nodes)
 * into the same rnpu_tfl_model struct used by the TFLite path, enabling
 * the existing rnpu_model pipeline to work unchanged.
 *
 * Primary target: check0_base_optimize.onnx from RKNN toolkit, which preserves
 * per-channel quantization annotations as explicit Q/DQ nodes.
 *
 * SPDX-License-Identifier: MIT
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>

#include "rnpu_onnx.h"
#include "onnx.pb-c.h"

/* ---- Name→index hash map (simple open-addressing) ---- */

#define MAP_CAPACITY 4096

struct name_entry {
   const char *name;
   unsigned index;
};

struct name_map {
   struct name_entry entries[MAP_CAPACITY];
   unsigned count;
};

static unsigned name_hash(const char *s)
{
   unsigned h = 5381;
   while (*s)
      h = h * 33 + (unsigned char)*s++;
   return h;
}

static void map_put(struct name_map *m, const char *name, unsigned index)
{
   unsigned h = name_hash(name) % MAP_CAPACITY;
   for (unsigned i = 0; i < MAP_CAPACITY; i++) {
      unsigned slot = (h + i) % MAP_CAPACITY;
      if (!m->entries[slot].name) {
         m->entries[slot].name = name;
         m->entries[slot].index = index;
         m->count++;
         return;
      }
   }
   fprintf(stderr, "rnpu_onnx: name map full\n");
}

/* Returns index, or (unsigned)-1 if not found */
static unsigned map_get(const struct name_map *m, const char *name)
{
   unsigned h = name_hash(name) % MAP_CAPACITY;
   for (unsigned i = 0; i < MAP_CAPACITY; i++) {
      unsigned slot = (h + i) % MAP_CAPACITY;
      if (!m->entries[slot].name)
         return (unsigned)-1;
      if (strcmp(m->entries[slot].name, name) == 0)
         return m->entries[slot].index;
   }
   return (unsigned)-1;
}

/* ---- Quantization info extracted from DequantizeLinear/QuantizeLinear ---- */

struct quant_info {
   const char *tensor_name;  /* the output (for DQ) or input (for Q) tensor */
   float *scales;            /* per-channel or per-tensor */
   int64_t *zero_points;
   unsigned num_scales;
   float scale;              /* scalar (per-tensor or first per-channel) */
   int32_t zero_point;
};

#define MAX_QUANT_ENTRIES 1024

/* ---- Attribute helpers ---- */

static const Onnx__AttributeProto *find_attr(const Onnx__NodeProto *node,
                                              const char *name)
{
   for (size_t i = 0; i < node->n_attribute; i++) {
      if (strcmp(node->attribute[i]->name, name) == 0)
         return node->attribute[i];
   }
   return NULL;
}

static int64_t attr_int(const Onnx__NodeProto *node, const char *name, int64_t def)
{
   const Onnx__AttributeProto *a = find_attr(node, name);
   if (!a) return def;
   return a->i;
}

static int attr_ints(const Onnx__NodeProto *node, const char *name,
                     int64_t *out, unsigned max_n)
{
   const Onnx__AttributeProto *a = find_attr(node, name);
   if (!a || a->n_ints == 0) return 0;
   unsigned n = a->n_ints < max_n ? a->n_ints : max_n;
   for (unsigned i = 0; i < n; i++)
      out[i] = a->ints[i];
   return n;
}

/* ---- TensorProto data extraction ---- */

static const uint8_t *tensor_raw_data(const Onnx__TensorProto *t, size_t *out_len)
{
   if (t->raw_data.len > 0) {
      *out_len = t->raw_data.len;
      return t->raw_data.data;
   }
   *out_len = 0;
   return NULL;
}

static float *tensor_float_data(const Onnx__TensorProto *t, unsigned *out_count)
{
   if (t->raw_data.len > 0) {
      *out_count = t->raw_data.len / sizeof(float);
      return (float *)t->raw_data.data;  /* little-endian, same as host */
   }
   if (t->n_float_data > 0) {
      *out_count = t->n_float_data;
      return t->float_data;
   }
   *out_count = 0;
   return NULL;
}

static int64_t tensor_int64_scalar(const Onnx__TensorProto *t)
{
   if (t->raw_data.len >= 8)
      return *(int64_t *)t->raw_data.data;
   if (t->n_int64_data > 0)
      return t->int64_data[0];
   /* int8 zero points may be stored as int32 */
   if (t->n_int32_data > 0)
      return t->int32_data[0];
   if (t->raw_data.len >= 1)
      return (int8_t)t->raw_data.data[0];
   return 0;
}

/* ---- ONNX op_type → TFLite builtin_code mapping ---- */

static int onnx_to_tflite_opcode(const char *op_type)
{
   if (strcmp(op_type, "Conv") == 0) return TFLITE_OP_CONV_2D;
   if (strcmp(op_type, "Add") == 0) return TFLITE_OP_ADD;
   if (strcmp(op_type, "Concat") == 0) return TFLITE_OP_CONCATENATION;
   if (strcmp(op_type, "MaxPool") == 0) return TFLITE_OP_MAX_POOL_2D;
   if (strcmp(op_type, "AveragePool") == 0) return TFLITE_OP_AVERAGE_POOL_2D;
   if (strcmp(op_type, "Pad") == 0) return TFLITE_OP_PAD;
   if (strcmp(op_type, "Resize") == 0) return TFLITE_OP_RESIZE_NEAREST_NEIGHBOR;
   if (strcmp(op_type, "Sigmoid") == 0) return TFLITE_OP_LOGISTIC;
   if (strcmp(op_type, "Reshape") == 0) return TFLITE_OP_RESHAPE;
   if (strcmp(op_type, "Softmax") == 0) return TFLITE_OP_SOFTMAX;
   return -1;
}

/* ---- Main parser ---- */

int rnpu_onnx_parse(const char *path, struct rnpu_tfl_model *model)
{
   memset(model, 0, sizeof(*model));

   /* Read file */
   FILE *f = fopen(path, "rb");
   if (!f) {
      fprintf(stderr, "rnpu_onnx: cannot open %s: %s\n", path, strerror(errno));
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

   /* Parse protobuf */
   Onnx__ModelProto *mp = onnx__model_proto__unpack(NULL, model->file_size,
                                                     model->file_data);
   if (!mp) {
      fprintf(stderr, "rnpu_onnx: failed to parse protobuf\n");
      free(model->file_data);
      return -1;
   }

   Onnx__GraphProto *graph = mp->graph;
   if (!graph) {
      fprintf(stderr, "rnpu_onnx: no graph in model\n");
      onnx__model_proto__free_unpacked(mp, NULL);
      free(model->file_data);
      return -1;
   }

   printf("rnpu_onnx: loaded %s — %zu nodes, %zu initializers\n",
          path, graph->n_node, graph->n_initializer);

   /* Build initializer name→index map */
   struct name_map *init_map = calloc(1, sizeof(struct name_map));
   for (size_t i = 0; i < graph->n_initializer; i++) {
      if (graph->initializer[i]->name)
         map_put(init_map, graph->initializer[i]->name, i);
   }

   /* Pass 1: Extract quant info from DequantizeLinear/QuantizeLinear nodes.
    * These nodes encode: tensor_name → (scale, zero_point).
    * DQ: inputs[0]=quantized, inputs[1]=scale, inputs[2]=zp → output=dequantized
    * Q:  inputs[0]=float, inputs[1]=scale, inputs[2]=zp → output=quantized */

   struct quant_info *quant_table = calloc(MAX_QUANT_ENTRIES, sizeof(struct quant_info));
   unsigned quant_count = 0;

   /* Also build a map: output_name → node_index for resolving tensor producers */
   struct name_map *output_map = calloc(1, sizeof(struct name_map));
   for (size_t i = 0; i < graph->n_node; i++) {
      Onnx__NodeProto *node = graph->node[i];
      for (size_t j = 0; j < node->n_output; j++)
         map_put(output_map, node->output[j], i);
   }

   for (size_t i = 0; i < graph->n_node; i++) {
      Onnx__NodeProto *node = graph->node[i];
      bool is_dq = strcmp(node->op_type, "DequantizeLinear") == 0;
      bool is_q  = strcmp(node->op_type, "QuantizeLinear") == 0;
      if (!is_dq && !is_q) continue;
      if (node->n_input < 2 || node->n_output < 1) continue;
      if (quant_count >= MAX_QUANT_ENTRIES) continue;

      struct quant_info *qi = &quant_table[quant_count];

      /* For DQ: the output name is what downstream nodes reference */
      /* For Q: the output name is the quantized tensor */
      qi->tensor_name = node->output[0];

      /* Scale tensor */
      unsigned scale_idx = map_get(init_map, node->input[1]);
      if (scale_idx != (unsigned)-1) {
         const Onnx__TensorProto *st = graph->initializer[scale_idx];
         unsigned n;
         float *sd = tensor_float_data(st, &n);
         if (sd && n > 0) {
            qi->scale = sd[0];
            qi->num_scales = n;
            if (n > 1) {
               qi->scales = calloc(n, sizeof(float));
               memcpy(qi->scales, sd, n * sizeof(float));
            }
         }
      }

      /* Zero point tensor (optional, input[2]) */
      if (node->n_input >= 3) {
         unsigned zp_idx = map_get(init_map, node->input[2]);
         if (zp_idx != (unsigned)-1) {
            const Onnx__TensorProto *zt = graph->initializer[zp_idx];
            qi->zero_point = (int32_t)tensor_int64_scalar(zt);
            if (qi->num_scales > 1) {
               qi->zero_points = calloc(qi->num_scales, sizeof(int64_t));
               if (zt->raw_data.len > 0) {
                  /* int8 zero points: 1 byte each */
                  for (unsigned k = 0; k < qi->num_scales && k < zt->raw_data.len; k++)
                     qi->zero_points[k] = (int8_t)zt->raw_data.data[k];
               } else if (zt->n_int32_data > 0) {
                  for (unsigned k = 0; k < qi->num_scales && k < zt->n_int32_data; k++)
                     qi->zero_points[k] = zt->int32_data[k];
               }
            }
         }
      }

      quant_count++;
   }

   /* Build quant name→index map */
   struct name_map *quant_map = calloc(1, sizeof(struct name_map));
   for (unsigned i = 0; i < quant_count; i++)
      map_put(quant_map, quant_table[i].tensor_name, i);

   printf("rnpu_onnx: extracted %u quant annotations\n", quant_count);

   /* Pass 2: Count meaningful ops (Conv, Add, Concat, etc.) and tensors.
    * We skip DequantizeLinear/QuantizeLinear (they're metadata, not compute).
    * Also skip exDataConvert (RKNN internal). */

   unsigned n_ops = 0;
   for (size_t i = 0; i < graph->n_node; i++) {
      const char *ot = graph->node[i]->op_type;
      if (strcmp(ot, "DequantizeLinear") == 0) continue;
      if (strcmp(ot, "QuantizeLinear") == 0) continue;
      if (strcmp(ot, "exDataConvert") == 0) continue;
      if (onnx_to_tflite_opcode(ot) < 0) {
         printf("rnpu_onnx: skipping unsupported op: %s\n", ot);
         continue;
      }
      n_ops++;
   }

   /* We need a flat tensor array indexed by integer, like TFLite.
    * Strategy: assign each unique ONNX tensor name an integer index.
    * Collect all tensor names from node inputs/outputs. */

   struct name_map *tensor_map = calloc(1, sizeof(struct name_map));
   unsigned next_tensor_idx = 0;

   /* Register graph inputs first */
   for (size_t i = 0; i < graph->n_input; i++) {
      if (graph->input[i]->name &&
          map_get(tensor_map, graph->input[i]->name) == (unsigned)-1) {
         map_put(tensor_map, graph->input[i]->name, next_tensor_idx++);
      }
   }

   /* Register all node outputs as tensors (skip Q/DQ intermediates — use their
    * input/output names directly) */
   for (size_t i = 0; i < graph->n_node; i++) {
      Onnx__NodeProto *node = graph->node[i];
      for (size_t j = 0; j < node->n_output; j++) {
         if (map_get(tensor_map, node->output[j]) == (unsigned)-1)
            map_put(tensor_map, node->output[j], next_tensor_idx++);
      }
      for (size_t j = 0; j < node->n_input; j++) {
         if (node->input[j] && map_get(tensor_map, node->input[j]) == (unsigned)-1)
            map_put(tensor_map, node->input[j], next_tensor_idx++);
      }
   }

   unsigned total_tensors = next_tensor_idx;

   /* Allocate model arrays */
   model->tensor_count = total_tensors;
   model->tensors = calloc(total_tensors, sizeof(struct rnpu_tfl_tensor));

   /* We'll over-allocate buffers — one per initializer plus some slack */
   model->buffer_count = graph->n_initializer + 1;
   model->buffers = calloc(model->buffer_count, sizeof(struct rnpu_tfl_buffer));

   /* Fill buffer data from initializers */
   for (size_t i = 0; i < graph->n_initializer; i++) {
      const Onnx__TensorProto *tp = graph->initializer[i];
      size_t len;
      const uint8_t *data = tensor_raw_data(tp, &len);
      if (data) {
         model->buffers[i + 1].data = data;
         model->buffers[i + 1].size = len;
      }
   }

   /* Fill tensor metadata. For tensors that are initializers, set buffer_index.
    * For activation tensors, try to get shape from graph input/output ValueInfoProto. */

   /* Set shapes from initializer tensors */
   for (size_t i = 0; i < graph->n_initializer; i++) {
      const Onnx__TensorProto *tp = graph->initializer[i];
      if (!tp->name) continue;
      unsigned ti = map_get(tensor_map, tp->name);
      if (ti == (unsigned)-1) continue;

      struct rnpu_tfl_tensor *t = &model->tensors[ti];
      t->buffer_index = i + 1;  /* buffer 0 is empty */
      t->shape_len = tp->n_dims < 4 ? tp->n_dims : 4;
      for (int d = 0; d < t->shape_len; d++)
         t->shape[d] = tp->dims[d];

      switch (tp->data_type) {
      case ONNX__TENSOR_PROTO__DATA_TYPE__FLOAT: t->type = 0; break;  /* FLOAT32 */
      case ONNX__TENSOR_PROTO__DATA_TYPE__INT8:  t->type = 9; break;
      case ONNX__TENSOR_PROTO__DATA_TYPE__INT32: t->type = 2; break;
      default: t->type = tp->data_type; break;
      }
   }

   /* Set shapes from graph inputs ValueInfoProto */
   for (size_t i = 0; i < graph->n_input; i++) {
      Onnx__ValueInfoProto *vi = graph->input[i];
      if (!vi->name) continue;
      unsigned ti = map_get(tensor_map, vi->name);
      if (ti == (unsigned)-1) continue;

      struct rnpu_tfl_tensor *t = &model->tensors[ti];
      if (vi->type && vi->type->tensor_type && vi->type->tensor_type->shape) {
         Onnx__TensorShapeProto *sh = vi->type->tensor_type->shape;
         t->shape_len = sh->n_dim < 4 ? sh->n_dim : 4;
         for (int d = 0; d < t->shape_len; d++) {
            if (sh->dim[d]->value_case ==
                ONNX__TENSOR_SHAPE_PROTO__DIMENSION__VALUE_DIM_VALUE)
               t->shape[d] = sh->dim[d]->dim_value;
            else
               t->shape[d] = 1;  /* symbolic dim → assume 1 (batch) */
         }
      }
   }

   /* Set shapes from graph outputs ValueInfoProto */
   for (size_t i = 0; i < graph->n_output; i++) {
      Onnx__ValueInfoProto *vi = graph->output[i];
      if (!vi->name) continue;
      unsigned ti = map_get(tensor_map, vi->name);
      if (ti == (unsigned)-1) continue;

      struct rnpu_tfl_tensor *t = &model->tensors[ti];
      if (vi->type && vi->type->tensor_type && vi->type->tensor_type->shape) {
         Onnx__TensorShapeProto *sh = vi->type->tensor_type->shape;
         t->shape_len = sh->n_dim < 4 ? sh->n_dim : 4;
         for (int d = 0; d < t->shape_len; d++) {
            if (sh->dim[d]->value_case ==
                ONNX__TENSOR_SHAPE_PROTO__DIMENSION__VALUE_DIM_VALUE)
               t->shape[d] = sh->dim[d]->dim_value;
            else
               t->shape[d] = 1;
         }
      }
   }

   /* Apply quantization annotations to tensors.
    * Walk Q/DQ nodes: for each, find the tensor they annotate and set quant params. */
   for (unsigned i = 0; i < quant_count; i++) {
      struct quant_info *qi = &quant_table[i];

      /* The Q/DQ output tensor carries the quant annotation */
      unsigned ti = map_get(tensor_map, qi->tensor_name);
      if (ti == (unsigned)-1) continue;

      struct rnpu_tfl_quant *q = &model->tensors[ti].quant;
      q->scale = qi->scale;
      q->zero_point = qi->zero_point;
      if (qi->num_scales > 1) {
         q->num_scales = qi->num_scales;
         q->scales = calloc(qi->num_scales, sizeof(float));
         memcpy(q->scales, qi->scales, qi->num_scales * sizeof(float));
         if (qi->zero_points) {
            q->zero_points = calloc(qi->num_scales, sizeof(int64_t));
            memcpy(q->zero_points, qi->zero_points, qi->num_scales * sizeof(int64_t));
         }
      }
   }

   /* Propagate shapes and buffer data through DQ/Q nodes.
    * DQ output tensor should inherit shape/buffer from its input[0] (quantized data).
    * This ensures Conv weight inputs (which reference DQ outputs) have correct shapes. */
   for (size_t i = 0; i < graph->n_node; i++) {
      Onnx__NodeProto *node = graph->node[i];
      bool is_dq = strcmp(node->op_type, "DequantizeLinear") == 0;
      bool is_q  = strcmp(node->op_type, "QuantizeLinear") == 0;
      if (!is_dq && !is_q) continue;
      if (node->n_input < 1 || node->n_output < 1) continue;

      unsigned src_ti = map_get(tensor_map, node->input[0]);
      unsigned dst_ti = map_get(tensor_map, node->output[0]);
      if (src_ti == (unsigned)-1 || dst_ti == (unsigned)-1) continue;

      struct rnpu_tfl_tensor *src = &model->tensors[src_ti];
      struct rnpu_tfl_tensor *dst = &model->tensors[dst_ti];

      /* Copy shape if dst has none */
      if (dst->shape_len == 0 && src->shape_len > 0) {
         dst->shape_len = src->shape_len;
         memcpy(dst->shape, src->shape, sizeof(dst->shape));
      }

      /* Copy buffer reference if dst has none (DQ: weight data flows through) */
      if (dst->buffer_index == 0 && src->buffer_index > 0)
         dst->buffer_index = src->buffer_index;

      /* Copy type */
      if (dst->type == 0 && src->type != 0)
         dst->type = src->type;

      /* For Q nodes: propagate quant params BACK to input tensor.
       * This ensures Conv output tensors (which feed Q nodes) have quant params.
       * Q: input[0]=float_tensor, input[1]=scale, input[2]=zp → output=quantized */
      if (is_q && dst->quant.scale > 0 && src->quant.scale == 0) {
         src->quant.scale = dst->quant.scale;
         src->quant.zero_point = dst->quant.zero_point;
         if (dst->quant.num_scales > 1 && !src->quant.scales) {
            src->quant.num_scales = dst->quant.num_scales;
            src->quant.scales = calloc(dst->quant.num_scales, sizeof(float));
            memcpy(src->quant.scales, dst->quant.scales,
                   dst->quant.num_scales * sizeof(float));
            if (dst->quant.zero_points) {
               src->quant.zero_points = calloc(dst->quant.num_scales, sizeof(int64_t));
               memcpy(src->quant.zero_points, dst->quant.zero_points,
                      dst->quant.num_scales * sizeof(int64_t));
            }
         }
      }

      /* For DQ nodes: propagate quant params to output too (for activation tensors).
       * DQ: input[0]=quantized → output=dequantized (but carries the quant annotation) */
      if (is_dq && dst->quant.scale > 0 && src->quant.scale == 0) {
         src->quant.scale = dst->quant.scale;
         src->quant.zero_point = dst->quant.zero_point;
      }
   }

   /* Propagate quant params through passthrough nodes (Relu, Add, Sigmoid, etc.)
    * in reverse topological order. If a node's output has quant params but its
    * input doesn't, propagate back. This handles Conv→Relu→Q chains. */
   for (int pass = 0; pass < 3; pass++) {  /* multiple passes for longer chains */
      for (size_t i = graph->n_node; i > 0; i--) {
         Onnx__NodeProto *node = graph->node[i - 1];
         if (node->n_input < 1 || node->n_output < 1) continue;
         /* Skip Q/DQ — already handled */
         if (strcmp(node->op_type, "DequantizeLinear") == 0) continue;
         if (strcmp(node->op_type, "QuantizeLinear") == 0) continue;

         unsigned out_ti = map_get(tensor_map, node->output[0]);
         if (out_ti == (unsigned)-1) continue;
         struct rnpu_tfl_tensor *out_t = &model->tensors[out_ti];
         if (out_t->quant.scale == 0) continue;  /* output has no quant → skip */

         /* Propagate to input[0] (activation input of the op) */
         unsigned in_ti = map_get(tensor_map, node->input[0]);
         if (in_ti == (unsigned)-1) continue;
         struct rnpu_tfl_tensor *in_t = &model->tensors[in_ti];
         if (in_t->quant.scale == 0) {
            in_t->quant.scale = out_t->quant.scale;
            in_t->quant.zero_point = out_t->quant.zero_point;
         }
      }
   }

   /* Pass 3: Build ops array.
    * ONNX uses NCHW; TFLite uses NHWC. We need to convert shapes.
    * ONNX Conv inputs: [input, weight, bias?]
    * ONNX Conv weight shape: [OC, IC/group, KH, KW] (NCHW)
    * TFLite Conv weight shape: [OC, KH, KW, IC] (OHWI)
    * TFLite activation shape: [N, H, W, C] (NHWC)
    *
    * For the rnpu_model pipeline, tensor shapes must be NHWC.
    * We convert ONNX NCHW shapes to NHWC here. */

   model->op_count = n_ops;
   model->ops = calloc(n_ops, sizeof(struct rnpu_tfl_op));

   unsigned op_idx = 0;
   for (size_t i = 0; i < graph->n_node; i++) {
      Onnx__NodeProto *node = graph->node[i];
      int tfl_code = onnx_to_tflite_opcode(node->op_type);
      if (tfl_code < 0) continue;  /* skip Q/DQ, unsupported */

      struct rnpu_tfl_op *op = &model->ops[op_idx++];
      op->builtin_code = tfl_code;

      /* Resolve input tensor indices.
       * For Conv: ONNX inputs may reference DQ output names.
       * We need to find the actual activation/weight tensor. */
      op->input_count = node->n_input;
      op->inputs = calloc(node->n_input, sizeof(int));
      for (size_t j = 0; j < node->n_input; j++) {
         if (!node->input[j] || node->input[j][0] == '\0') {
            op->inputs[j] = -1;  /* optional input not present */
            continue;
         }
         unsigned ti = map_get(tensor_map, node->input[j]);
         op->inputs[j] = (ti != (unsigned)-1) ? (int)ti : -1;
      }

      op->output_count = node->n_output;
      op->outputs = calloc(node->n_output, sizeof(int));
      for (size_t j = 0; j < node->n_output; j++) {
         unsigned ti = map_get(tensor_map, node->output[j]);
         op->outputs[j] = (ti != (unsigned)-1) ? (int)ti : -1;
      }

      /* Parse Conv-specific attributes */
      if (tfl_code == TFLITE_OP_CONV_2D || tfl_code == TFLITE_OP_DEPTHWISE_CONV_2D) {
         int64_t strides[2] = {1, 1};
         attr_ints(node, "strides", strides, 2);

         int64_t pads[4] = {0, 0, 0, 0};  /* [top, left, bottom, right] */
         int n_pads = attr_ints(node, "pads", pads, 4);

         int64_t dilations[2] = {1, 1};
         attr_ints(node, "dilations", dilations, 2);

         int64_t group = attr_int(node, "group", 1);

         /* Auto-pad attribute */
         const Onnx__AttributeProto *ap = find_attr(node, "auto_pad");
         int padding = 1;  /* default: VALID */
         if (ap && ap->s.len > 0) {
            if (strncmp((char *)ap->s.data, "SAME", 4) == 0)
               padding = 0;  /* SAME */
         }
         if (n_pads > 0 && (pads[0] > 0 || pads[1] > 0))
            padding = 0;  /* explicit padding = SAME-like */

         /* Check if depthwise: group == input_channels */
         unsigned weight_ti = (unsigned)op->inputs[1];
         if (weight_ti < total_tensors && model->tensors[weight_ti].shape_len >= 4) {
            struct rnpu_tfl_tensor *wt = &model->tensors[weight_ti];
            /* ONNX weight: [OC, IC/group, KH, KW] */
            unsigned oc = wt->shape[0];
            unsigned ic_per_group = wt->shape[1];
            if (group > 1 && group == (int64_t)oc && ic_per_group == 1) {
               /* Depthwise convolution */
               op->builtin_code = TFLITE_OP_DEPTHWISE_CONV_2D;
               op->opt.dw_conv.padding = padding;
               op->opt.dw_conv.stride_w = strides[1];
               op->opt.dw_conv.stride_h = strides[0];
               op->opt.dw_conv.depth_multiplier = 1;
               op->opt.dw_conv.dilation_w = dilations[1];
               op->opt.dw_conv.dilation_h = dilations[0];

               /* Convert weight shape from [OC,1,KH,KW] to TFLite [1,KH,KW,OC] */
               int32_t kh = wt->shape[2], kw = wt->shape[3];
               wt->shape[0] = 1;
               wt->shape[1] = kh;
               wt->shape[2] = kw;
               wt->shape[3] = oc;
            } else {
               op->opt.conv.padding = padding;
               op->opt.conv.stride_w = strides[1];
               op->opt.conv.stride_h = strides[0];
               op->opt.conv.dilation_w = dilations[1];
               op->opt.conv.dilation_h = dilations[0];

               /* Convert weight shape from [OC,IC,KH,KW] to TFLite [OC,KH,KW,IC] */
               unsigned ic = wt->shape[1] * group;
               int32_t kh = wt->shape[2], kw = wt->shape[3];
               wt->shape[0] = oc;
               wt->shape[1] = kh;
               wt->shape[2] = kw;
               wt->shape[3] = ic;
            }
         }
      }

      /* Parse Pool attributes */
      if (tfl_code == TFLITE_OP_MAX_POOL_2D || tfl_code == TFLITE_OP_AVERAGE_POOL_2D) {
         int64_t kernel[2] = {1, 1}, strides[2] = {1, 1};
         attr_ints(node, "kernel_shape", kernel, 2);
         attr_ints(node, "strides", strides, 2);

         const Onnx__AttributeProto *ap = find_attr(node, "auto_pad");
         int padding = 1;
         if (ap && ap->s.len > 0 && strncmp((char *)ap->s.data, "SAME", 4) == 0)
            padding = 0;

         op->opt.pool.padding = padding;
         op->opt.pool.stride_w = strides[1];
         op->opt.pool.stride_h = strides[0];
         op->opt.pool.filter_w = kernel[1];
         op->opt.pool.filter_h = kernel[0];
      }

      /* Parse Concat axis */
      if (tfl_code == TFLITE_OP_CONCATENATION) {
         int64_t axis = attr_int(node, "axis", 0);
         /* ONNX NCHW axis=1 (channels) → TFLite NHWC axis=3 */
         if (axis == 1) axis = 3;
         else if (axis == 2) axis = 1;
         else if (axis == 3) axis = 2;
         op->opt.concat.axis = axis;
      }
   }

   /* Convert activation tensor shapes from NCHW to NHWC.
    * We do this for all 4D tensors that are NOT weight tensors (already converted).
    * Weight tensors were already converted above in the Conv handling. */
   for (unsigned i = 0; i < total_tensors; i++) {
      struct rnpu_tfl_tensor *t = &model->tensors[i];
      if (t->shape_len != 4) continue;
      /* Skip if this is a weight/bias initializer (already converted or not activation) */
      if (t->buffer_index > 0) continue;  /* has initializer data → skip */

      /* NCHW → NHWC: [N,C,H,W] → [N,H,W,C] */
      int32_t n = t->shape[0], c = t->shape[1], h = t->shape[2], w = t->shape[3];
      t->shape[0] = n;
      t->shape[1] = h;
      t->shape[2] = w;
      t->shape[3] = c;
   }

   /* Graph inputs/outputs */
   model->input_count = graph->n_input;
   model->graph_inputs = calloc(graph->n_input, sizeof(int));
   for (size_t i = 0; i < graph->n_input; i++) {
      if (graph->input[i]->name) {
         unsigned ti = map_get(tensor_map, graph->input[i]->name);
         model->graph_inputs[i] = (ti != (unsigned)-1) ? (int)ti : 0;
      }
   }

   /* Filter graph inputs: ONNX lists initializers as graph inputs too.
    * We only want actual activation inputs (those without initializer data). */
   unsigned real_inputs = 0;
   for (unsigned i = 0; i < model->input_count; i++) {
      unsigned ti = model->graph_inputs[i];
      if (ti < total_tensors && model->tensors[ti].buffer_index == 0)
         model->graph_inputs[real_inputs++] = model->graph_inputs[i];
   }
   model->input_count = real_inputs;

   model->output_count = graph->n_output;
   model->graph_outputs = calloc(graph->n_output, sizeof(int));
   for (size_t i = 0; i < graph->n_output; i++) {
      if (graph->output[i]->name) {
         unsigned ti = map_get(tensor_map, graph->output[i]->name);
         model->graph_outputs[i] = (ti != (unsigned)-1) ? (int)ti : 0;
      }
   }

   printf("rnpu_onnx: %u ops, %u tensors, %u inputs, %u outputs\n",
          model->op_count, model->tensor_count,
          model->input_count, model->output_count);

   /* Print summary of Conv ops */
   unsigned conv_count = 0;
   for (unsigned i = 0; i < model->op_count; i++) {
      if (model->ops[i].builtin_code == TFLITE_OP_CONV_2D ||
          model->ops[i].builtin_code == TFLITE_OP_DEPTHWISE_CONV_2D) {
         conv_count++;
         if (conv_count <= 5 || conv_count == model->op_count) {
            struct rnpu_tfl_op *op = &model->ops[i];
            unsigned wi = (unsigned)op->inputs[1];
            if (wi < total_tensors) {
               struct rnpu_tfl_tensor *wt = &model->tensors[wi];
               printf("  Conv[%u]: weight=[%d,%d,%d,%d]",
                      conv_count - 1, wt->shape[0], wt->shape[1],
                      wt->shape[2], wt->shape[3]);
               if (wt->quant.num_scales > 1)
                  printf(" per-axis(%u scales)", wt->quant.num_scales);
               else if (wt->quant.scale > 0)
                  printf(" scale=%.6f", wt->quant.scale);
               printf("\n");
            }
         } else if (conv_count == 6) {
            printf("  ... (more convs)\n");
         }
      }
   }
   printf("rnpu_onnx: total %u Conv ops\n", conv_count);

   /* Cleanup protobuf tree (but keep file_data alive — buffers point into it) */
   /* NOTE: We can't free the protobuf tree yet because buffer data pointers
    * reference raw_data inside TensorProto nodes. We need to copy the weight data
    * into standalone buffers first. */

   /* Copy initializer data to standalone buffers so we can free the protobuf tree */
   for (unsigned i = 0; i + 1 < model->buffer_count; i++) {
      if (model->buffers[i + 1].data && model->buffers[i + 1].size > 0) {
         uint8_t *copy = malloc(model->buffers[i + 1].size);
         memcpy(copy, model->buffers[i + 1].data, model->buffers[i + 1].size);
         model->buffers[i + 1].data = copy;
      }
   }

   /* Free quant table scale copies */
   for (unsigned i = 0; i < quant_count; i++)
      free(quant_table[i].scales);
   /* Don't free zero_points — they were set into model->tensors[].quant.zero_points */

   onnx__model_proto__free_unpacked(mp, NULL);
   free(init_map);
   free(output_map);
   free(quant_map);
   free(quant_table);
   free(tensor_map);

   return 0;
}
