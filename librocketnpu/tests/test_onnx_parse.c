/*
 * Test ONNX parser — parse an ONNX model and print extracted ops/tensors.
 * Usage: ./test_onnx_parse <model.onnx>
 * SPDX-License-Identifier: MIT
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "rnpu_internal.h"
#include "rnpu_onnx.h"

int main(int argc, char **argv)
{
   if (argc < 2) {
      fprintf(stderr, "Usage: %s <model.onnx>\n", argv[0]);
      return 1;
   }

   struct rnpu_tfl_model model;
   int ret = rnpu_onnx_parse(argv[1], &model);
   if (ret < 0) {
      fprintf(stderr, "Failed to parse %s\n", argv[1]);
      return 1;
   }

   printf("\n=== Parsed Model Summary ===\n");
   printf("Tensors: %u\n", model.tensor_count);
   printf("Ops:     %u\n", model.op_count);
   printf("Inputs:  %u\n", model.input_count);
   printf("Outputs: %u\n", model.output_count);
   printf("Buffers: %u\n", model.buffer_count);

   /* Print graph inputs */
   printf("\nGraph inputs:\n");
   for (unsigned i = 0; i < model.input_count; i++) {
      unsigned ti = model.graph_inputs[i];
      struct rnpu_tfl_tensor *t = &model.tensors[ti];
      printf("  [%u] tensor %u: shape=[%d,%d,%d,%d] type=%d",
             i, ti, t->shape[0], t->shape[1], t->shape[2], t->shape[3], t->type);
      if (t->quant.scale > 0)
         printf(" scale=%.6f zp=%d", t->quant.scale, t->quant.zero_point);
      printf("\n");
   }

   /* Print graph outputs */
   printf("\nGraph outputs:\n");
   for (unsigned i = 0; i < model.output_count; i++) {
      unsigned ti = model.graph_outputs[i];
      struct rnpu_tfl_tensor *t = &model.tensors[ti];
      printf("  [%u] tensor %u: shape=[%d,%d,%d,%d] type=%d",
             i, ti, t->shape[0], t->shape[1], t->shape[2], t->shape[3], t->type);
      if (t->quant.scale > 0)
         printf(" scale=%.6f zp=%d", t->quant.scale, t->quant.zero_point);
      printf("\n");
   }

   /* Print all ops */
   printf("\nOps:\n");
   unsigned conv_idx = 0;
   for (unsigned i = 0; i < model.op_count; i++) {
      struct rnpu_tfl_op *op = &model.ops[i];
      const char *name = "?";
      switch (op->builtin_code) {
      case TFLITE_OP_CONV_2D: name = "CONV_2D"; break;
      case TFLITE_OP_DEPTHWISE_CONV_2D: name = "DW_CONV_2D"; break;
      case TFLITE_OP_ADD: name = "ADD"; break;
      case TFLITE_OP_CONCATENATION: name = "CONCAT"; break;
      case TFLITE_OP_MAX_POOL_2D: name = "MAX_POOL"; break;
      case TFLITE_OP_AVERAGE_POOL_2D: name = "AVG_POOL"; break;
      case TFLITE_OP_PAD: name = "PAD"; break;
      case TFLITE_OP_RESIZE_NEAREST_NEIGHBOR: name = "RESIZE"; break;
      case TFLITE_OP_LOGISTIC: name = "SIGMOID"; break;
      case TFLITE_OP_RESHAPE: name = "RESHAPE"; break;
      case TFLITE_OP_SOFTMAX: name = "SOFTMAX"; break;
      }

      printf("  [%3u] %-12s in=[", i, name);
      for (int j = 0; j < op->input_count; j++)
         printf("%s%d", j ? "," : "", op->inputs[j]);
      printf("] out=[");
      for (int j = 0; j < op->output_count; j++)
         printf("%s%d", j ? "," : "", op->outputs[j]);
      printf("]");

      /* For conv: print weight shape and quant */
      if (op->builtin_code == TFLITE_OP_CONV_2D ||
          op->builtin_code == TFLITE_OP_DEPTHWISE_CONV_2D) {
         printf(" (conv#%u)", conv_idx++);
         if (op->input_count >= 2 && op->inputs[1] >= 0 &&
             (unsigned)op->inputs[1] < model.tensor_count) {
            struct rnpu_tfl_tensor *wt = &model.tensors[op->inputs[1]];
            printf(" w=[%d,%d,%d,%d]", wt->shape[0], wt->shape[1],
                   wt->shape[2], wt->shape[3]);
            if (wt->quant.num_scales > 1)
               printf(" per-axis(%u)", wt->quant.num_scales);
            else if (wt->quant.scale > 0)
               printf(" ws=%.6f", wt->quant.scale);

            /* Check if weight data is available */
            if (wt->buffer_index > 0 && wt->buffer_index < model.buffer_count) {
               const struct rnpu_tfl_buffer *buf = &model.buffers[wt->buffer_index];
               if (buf->data && buf->size > 0)
                  printf(" data=%uB", buf->size);
               else
                  printf(" NO_DATA");
            }
         }

         /* Output tensor quant */
         if (op->output_count >= 1 && op->outputs[0] >= 0 &&
             (unsigned)op->outputs[0] < model.tensor_count) {
            struct rnpu_tfl_tensor *ot = &model.tensors[op->outputs[0]];
            if (ot->quant.scale > 0)
               printf(" os=%.6f ozp=%d", ot->quant.scale, ot->quant.zero_point);
         }

         /* Conv stride */
         if (op->builtin_code == TFLITE_OP_CONV_2D)
            printf(" s=%dx%d", op->opt.conv.stride_h, op->opt.conv.stride_w);
         else
            printf(" s=%dx%d dm=%d", op->opt.dw_conv.stride_h,
                   op->opt.dw_conv.stride_w, op->opt.dw_conv.depth_multiplier);
      }

      printf("\n");
   }

   printf("\nTotal: %u ops (%u conv)\n", model.op_count, conv_idx);

   /* Cleanup — note: rnpu_tflite_free works for our struct too, but we need to
    * also free the copied buffer data */
   for (unsigned i = 0; i < model.tensor_count; i++) {
      free(model.tensors[i].quant.scales);
      free(model.tensors[i].quant.zero_points);
   }
   free(model.tensors);
   for (unsigned i = 0; i < model.op_count; i++) {
      free(model.ops[i].inputs);
      free(model.ops[i].outputs);
   }
   free(model.ops);
   /* Free copied buffer data */
   for (unsigned i = 1; i < model.buffer_count; i++)
      free((void *)model.buffers[i].data);
   free(model.buffers);
   free(model.graph_inputs);
   free(model.graph_outputs);
   free(model.file_data);

   return 0;
}
