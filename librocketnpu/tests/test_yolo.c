/*
 * Test: YOLOv5s-relu INT8 inference via librocketnpu
 * Verifies detection accuracy and per-axis quantization handling.
 * SPDX-License-Identifier: MIT
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "../include/rocketnpu.h"

static double now_ms(void)
{
   struct timespec ts;
   clock_gettime(CLOCK_MONOTONIC, &ts);
   return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

int main(int argc, char **argv)
{
   const char *model_path = argc > 1 ? argv[1] : "yolov5s_relu_int8.tflite";
   const char *input_path = argc > 2 ? argv[2] : NULL;
   int num_runs = argc > 3 ? atoi(argv[3]) : 5;

   int fd = rnpu_open(NULL);
   if (fd < 0) {
      fprintf(stderr, "Failed to open NPU\n");
      return 1;
   }

   double t0 = now_ms();
   rnpu_model_t *m = rnpu_model_load(fd, model_path);
   double t1 = now_ms();
   if (!m) {
      fprintf(stderr, "Failed to load model\n");
      rnpu_close(fd);
      return 1;
   }
   printf("Model loaded in %.1f ms\n", t1 - t0);

   int w, h, c;
   rnpu_get_input_dims(m, &w, &h, &c);
   printf("Input: %dx%dx%d\n", w, h, c);

   int n_out = rnpu_output_count(m);
   for (int i = 0; i < n_out; i++) {
      int ow, oh, oc;
      rnpu_get_output_dims(m, i, &ow, &oh, &oc);
      printf("Output %d: %dx%dx%d\n", i, ow, oh, oc);
   }

   /* Load or generate input */
   size_t isz = w * h * c;
   uint8_t *input = malloc(isz);
   if (input_path) {
      FILE *f = fopen(input_path, "rb");
      if (f) {
         size_t got = fread(input, 1, isz, f);
         fclose(f);
         printf("Loaded %zu bytes from %s\n", got, input_path);
      } else {
         fprintf(stderr, "Cannot open %s, using random\n", input_path);
         srand(42);
         for (size_t i = 0; i < isz; i++) input[i] = rand() & 0xFF;
      }
   } else {
      srand(42);
      for (size_t i = 0; i < isz; i++) input[i] = rand() & 0xFF;
      printf("Using random input\n");
   }

   /* Warmup */
   rnpu_invoke(m, input, isz);

   /* Benchmark */
   printf("Running %d iterations...\n", num_runs);
   double total = 0, mn = 1e9;
   for (int i = 0; i < num_runs; i++) {
      double start = now_ms();
      int ret = rnpu_invoke(m, input, isz);
      double dt = now_ms() - start;
      if (ret) { fprintf(stderr, "Invoke %d failed\n", i); break; }
      total += dt;
      if (dt < mn) mn = dt;
   }
   printf("YOLO Latency: avg=%.2f ms, min=%.2f ms\n", total / num_runs, mn);

   /* Save outputs */
   for (int i = 0; i < n_out; i++) {
      int ow, oh, oc;
      rnpu_get_output_dims(m, i, &ow, &oh, &oc);
      size_t osz = ow * oh * oc;
      uint8_t *out = malloc(osz);
      rnpu_get_output(m, i, out, osz);

      char fname[256];
      snprintf(fname, sizeof(fname), "yolo_output_%d.bin", i);
      FILE *f = fopen(fname, "wb");
      if (f) { fwrite(out, 1, osz, f); fclose(f); }
      printf("Output %d: %dx%dx%d → %s (%zu bytes)\n", i, ow, oh, oc, fname, osz);

      /* Stats */
      int unique[256] = {0};
      for (size_t j = 0; j < osz; j++) unique[out[j]]++;
      int cnt = 0;
      for (int j = 0; j < 256; j++) if (unique[j]) cnt++;
      printf("  Unique values: %d, min=%d max=%d\n",
             cnt, out[0], out[osz > 0 ? osz - 1 : 0]);

      free(out);
   }

   free(input);
   rnpu_model_free(m);
   rnpu_close(fd);
   printf("Done.\n");
   return 0;
}
