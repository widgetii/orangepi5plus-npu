/*
 * RKNN NPU benchmark for MobileNetV1
 * Uses RKNN C API (librknnrt.so) on vendor kernel.
 * Accepts .tflite or .rknn model files.
 *
 * Build: gcc -O2 -I$RKNN_API/include -o bench_rknn bench_rknn.c -lrknnrt -lm
 * Run:   ./bench_rknn mobilenet_v1_1.0_224_quant.tflite [num_runs]
 *
 * SPDX-License-Identifier: MIT
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "rknn_api.h"

static double now_ms(void) {
   struct timespec ts;
   clock_gettime(CLOCK_MONOTONIC, &ts);
   return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

int main(int argc, char **argv)
{
   if (argc < 2) {
      fprintf(stderr, "Usage: %s model.tflite [num_runs]\n", argv[0]);
      return 1;
   }
   int num_runs = argc > 2 ? atoi(argv[2]) : 200;

   /* Load model file */
   FILE *f = fopen(argv[1], "rb");
   if (!f) { perror("fopen"); return 1; }
   fseek(f, 0, SEEK_END);
   uint32_t model_size = ftell(f);
   fseek(f, 0, SEEK_SET);
   void *model_data = malloc(model_size);
   fread(model_data, 1, model_size, f);
   fclose(f);

   /* Init RKNN */
   rknn_context ctx;
   double t0 = now_ms();
   int ret = rknn_init(&ctx, model_data, model_size, 0, NULL);
   double t_init = now_ms() - t0;
   if (ret < 0) {
      fprintf(stderr, "rknn_init failed: %d\n", ret);
      return 1;
   }
   printf("RKNN init: %.1f ms\n", t_init);

   /* Query input/output */
   rknn_input_output_num io_num;
   rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
   printf("Inputs: %u, Outputs: %u\n", io_num.n_input, io_num.n_output);

   rknn_tensor_attr input_attr;
   memset(&input_attr, 0, sizeof(input_attr));
   input_attr.index = 0;
   rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &input_attr, sizeof(input_attr));
   printf("Input: %ux%ux%ux%u, type=%s, fmt=%s\n",
          input_attr.dims[0], input_attr.dims[1],
          input_attr.dims[2], input_attr.dims[3],
          get_type_string(input_attr.type),
          get_format_string(input_attr.fmt));

   rknn_tensor_attr output_attr;
   memset(&output_attr, 0, sizeof(output_attr));
   output_attr.index = 0;
   rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &output_attr, sizeof(output_attr));
   printf("Output: %ux%ux%ux%u, type=%s\n",
          output_attr.dims[0], output_attr.dims[1],
          output_attr.dims[2], output_attr.dims[3],
          get_type_string(output_attr.type));

   /* Prepare input (zeros) */
   unsigned input_size = input_attr.dims[1] * input_attr.dims[2] * input_attr.dims[3];
   uint8_t *input_data = calloc(1, input_size);

   rknn_input inputs[1];
   memset(inputs, 0, sizeof(inputs));
   inputs[0].index = 0;
   inputs[0].type = RKNN_TENSOR_UINT8;
   inputs[0].fmt = RKNN_TENSOR_NHWC;
   inputs[0].buf = input_data;
   inputs[0].size = input_size;
   inputs[0].pass_through = 0;

   /* Query all output attrs */
   rknn_tensor_attr out_attrs[16];
   for (unsigned i = 0; i < io_num.n_output && i < 16; i++) {
      memset(&out_attrs[i], 0, sizeof(out_attrs[i]));
      out_attrs[i].index = i;
      rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &out_attrs[i], sizeof(out_attrs[i]));
      printf("Output %u: %ux%ux%ux%u, type=%s\n", i,
             out_attrs[i].dims[0], out_attrs[i].dims[1],
             out_attrs[i].dims[2], out_attrs[i].dims[3],
             get_type_string(out_attrs[i].type));
   }

   /* Warmup */
   printf("Warmup...\n");
   rknn_inputs_set(ctx, 1, inputs);
   rknn_run(ctx, NULL);
   rknn_output outputs[16];
   memset(outputs, 0, sizeof(outputs));
   for (unsigned i = 0; i < io_num.n_output && i < 16; i++) {
      outputs[i].index = i;
      outputs[i].want_float = 0;
   }
   rknn_outputs_get(ctx, io_num.n_output, outputs, NULL);
   rknn_outputs_release(ctx, io_num.n_output, outputs);

   /* Benchmark */
   printf("Running %d iterations...\n", num_runs);
   double total = 0, min_ms = 1e9;
   for (int i = 0; i < num_runs; i++) {
      rknn_inputs_set(ctx, 1, inputs);

      double start = now_ms();
      rknn_run(ctx, NULL);
      double dt = now_ms() - start;

      memset(outputs, 0, sizeof(outputs));
      for (unsigned j = 0; j < io_num.n_output && j < 16; j++) {
         outputs[j].index = j;
         outputs[j].want_float = 0;
      }
      rknn_outputs_get(ctx, io_num.n_output, outputs, NULL);

      /* Save golden output on last iteration */
      if (i == num_runs - 1) {
         for (unsigned j = 0; j < io_num.n_output && j < 16; j++) {
            char fname[64];
            snprintf(fname, sizeof(fname), "rknn_golden_%u.bin", j);
            FILE *gf = fopen(fname, "wb");
            if (gf) {
               fwrite(outputs[j].buf, 1, outputs[j].size, gf);
               fclose(gf);
               printf("Saved %s (%u bytes)\n", fname, outputs[j].size);
            }
         }
      }

      rknn_outputs_release(ctx, io_num.n_output, outputs);
      total += dt;
      if (dt < min_ms) min_ms = dt;
   }
   printf("Latency: avg=%.2f ms, min=%.2f ms, FPS=%.1f\n",
          total / num_runs, min_ms, 1000.0 * num_runs / total);

   free(input_data);
   free(model_data);
   rknn_destroy(ctx);
   return 0;
}
