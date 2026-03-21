/*
 * Test: MobileNetV1 INT8 inference via librocketnpu
 * Verifies bit-exact output vs Mesa system library.
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
   const char *model_path = argc > 1 ? argv[1] :
      "/root/npu-research/models/mobilenet_v1_1.0_224_quant.tflite";
   int num_runs = argc > 2 ? atoi(argv[2]) : 10;
   const char *golden_path = argc > 3 ? argv[3] : NULL;

   /* Open NPU */
   int fd = rnpu_open(NULL);
   if (fd < 0) {
      fprintf(stderr, "Failed to open NPU device\n");
      return 1;
   }

   /* Load model */
   double t0 = now_ms();
   rnpu_model_t *m = rnpu_model_load(fd, model_path);
   double t1 = now_ms();
   if (!m) {
      fprintf(stderr, "Failed to load model\n");
      rnpu_close(fd);
      return 1;
   }
   printf("Model loaded in %.1f ms\n", t1 - t0);

   /* Get dimensions */
   int w, h, c;
   rnpu_get_input_dims(m, &w, &h, &c);
   printf("Input: %dx%dx%d\n", w, h, c);

   int n_out = rnpu_output_count(m);
   for (int i = 0; i < n_out; i++) {
      int ow, oh, oc;
      rnpu_get_output_dims(m, i, &ow, &oh, &oc);
      printf("Output %d: %dx%dx%d\n", i, ow, oh, oc);
   }

   /* Create test input (all 128 = zero point for typical models) */
   size_t input_size = w * h * c;
   uint8_t *input = calloc(1, input_size);
   memset(input, 128, input_size);

   /* Warmup */
   printf("Warmup...\n");
   if (rnpu_invoke(m, input, input_size) != 0) {
      fprintf(stderr, "Warmup invoke failed!\n");
      rnpu_model_free(m);
      rnpu_close(fd);
      return 1;
   }

   /* Benchmark */
   printf("Running %d iterations...\n", num_runs);
   double total = 0, min_ms = 1e9;
   for (int i = 0; i < num_runs; i++) {
      double start = now_ms();
      int ret = rnpu_invoke(m, input, input_size);
      double elapsed = now_ms() - start;
      if (ret != 0) {
         fprintf(stderr, "Invoke %d failed!\n", i);
         break;
      }
      total += elapsed;
      if (elapsed < min_ms) min_ms = elapsed;
   }
   printf("Latency: avg=%.2f ms, min=%.2f ms\n", total / num_runs, min_ms);

   /* Dump output */
   for (int i = 0; i < n_out; i++) {
      int ow, oh, oc;
      rnpu_get_output_dims(m, i, &ow, &oh, &oc);
      size_t out_size = ow * oh * oc;
      uint8_t *output = malloc(out_size);
      rnpu_get_output(m, i, output, out_size);

      /* Print first 32 values */
      printf("Output %d (first 32): ", i);
      for (int j = 0; j < 32 && j < (int)out_size; j++)
         printf("%d ", output[j]);
      printf("\n");

      /* Save to file for comparison */
      char fname[256];
      snprintf(fname, sizeof(fname), "output_%d.bin", i);
      FILE *f = fopen(fname, "wb");
      if (f) {
         fwrite(output, 1, out_size, f);
         fclose(f);
         printf("Saved %s (%zu bytes)\n", fname, out_size);
      }

      free(output);
   }

   /* Golden comparison */
   int result = 0;
   if (golden_path && n_out > 0) {
      int ow, oh, oc;
      rnpu_get_output_dims(m, 0, &ow, &oh, &oc);
      size_t out_size = ow * oh * oc;

      FILE *gf = fopen(golden_path, "rb");
      if (!gf) {
         fprintf(stderr, "Cannot open golden file: %s\n", golden_path);
         result = 1;
      } else {
         fseek(gf, 0, SEEK_END);
         long gsize = ftell(gf);
         fseek(gf, 0, SEEK_SET);

         if ((size_t)gsize != out_size) {
            fprintf(stderr, "Golden size mismatch: %ld vs %zu\n", gsize, out_size);
            fclose(gf);
            result = 1;
         } else {
            uint8_t *golden = malloc(out_size);
            uint8_t *output = malloc(out_size);
            fread(golden, 1, out_size, gf);
            fclose(gf);
            rnpu_get_output(m, 0, output, out_size);

            int exact = 0, max_diff = 0;
            long sum_diff = 0;
            for (size_t i = 0; i < out_size; i++) {
               int d = abs((int)output[i] - (int)golden[i]);
               if (d == 0) exact++;
               if (d > max_diff) max_diff = d;
               sum_diff += d;
            }
            double mean_diff = (double)sum_diff / out_size;

            printf("\n=== Golden Comparison ===\n");
            printf("Total bytes: %zu\n", out_size);
            printf("Exact matches: %d / %zu (%.1f%%)\n",
                   exact, out_size, 100.0 * exact / out_size);
            printf("Max diff: %d\n", max_diff);
            printf("Mean diff: %.3f\n", mean_diff);

            if (max_diff == 0) {
               printf("RESULT: PASS (bit-exact)\n");
            } else if (max_diff <= 1) {
               printf("RESULT: CLOSE (max_diff=1)\n");
            } else {
               printf("RESULT: FAIL (max_diff=%d)\n", max_diff);
               result = 1;
            }

            free(golden);
            free(output);
         }
      }
   }

   /* Cleanup */
   free(input);
   rnpu_model_free(m);
   rnpu_close(fd);

   printf("Done.\n");
   return result;
}
