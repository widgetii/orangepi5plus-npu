/*
 * Test: Fully Connected (Linear) INT8 inference via librocketnpu
 *
 * Usage: ./test_fc [model.tflite] [golden.bin] [input.bin]
 *
 * SPDX-License-Identifier: MIT
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../include/rocketnpu.h"

int main(int argc, char **argv)
{
   const char *model_path = argc > 1 ? argv[1] : "fc_model_int8.tflite";
   const char *golden_path = (argc > 2 && argv[2][0]) ? argv[2] : NULL;
   const char *input_path = argc > 3 ? argv[3] : NULL;

   int fd = rnpu_open(NULL);
   if (fd < 0) {
      fprintf(stderr, "Failed to open NPU device\n");
      return 1;
   }

   rnpu_model_t *m = rnpu_model_load(fd, model_path);
   if (!m) {
      fprintf(stderr, "Failed to load model: %s\n", model_path);
      rnpu_close(fd);
      return 1;
   }

   int w, h, c;
   rnpu_get_input_dims(m, &w, &h, &c);
   printf("Input: %dx%dx%d\n", w, h, c);

   int n_out = rnpu_output_count(m);
   for (int i = 0; i < n_out; i++) {
      int ow, oh, oc;
      rnpu_get_output_dims(m, i, &ow, &oh, &oc);
      printf("Output %d: %dx%dx%d\n", i, ow, oh, oc);
   }

   size_t input_size = w * h * c;
   uint8_t *input = calloc(1, input_size);

   if (input_path) {
      FILE *inf = fopen(input_path, "rb");
      if (!inf) {
         fprintf(stderr, "Cannot open input: %s\n", input_path);
         rnpu_model_free(m); rnpu_close(fd);
         return 1;
      }
      fread(input, 1, input_size, inf);
      fclose(inf);
      printf("Loaded input: %s\n", input_path);
   } else {
      memset(input, 128, input_size);
      printf("Using default input (all 128)\n");
   }

   if (rnpu_invoke(m, input, input_size) != 0) {
      fprintf(stderr, "Invoke failed!\n");
      rnpu_model_free(m); rnpu_close(fd);
      return 1;
   }

   /* Dump all outputs */
   for (int i = 0; i < n_out; i++) {
      int ow, oh, oc;
      rnpu_get_output_dims(m, i, &ow, &oh, &oc);
      size_t out_size = ow * oh * oc;
      uint8_t *output = malloc(out_size);
      rnpu_get_output(m, i, output, out_size);

      printf("Output %d values: ", i);
      for (size_t j = 0; j < out_size; j++)
         printf("%d ", (int)(int8_t)output[j]);
      printf("\n");

      char fname[256];
      snprintf(fname, sizeof(fname), "fc_npu_output_%d.bin", i);
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
         fprintf(stderr, "Cannot open golden: %s\n", golden_path);
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
            int d = abs((int)(int8_t)output[i] - (int)(int8_t)golden[i]);
            if (d == 0) exact++;
            if (d > max_diff) max_diff = d;
            sum_diff += d;
         }

         printf("\n=== Golden Comparison ===\n");
         printf("Total bytes: %zu\n", out_size);
         printf("Exact: %d / %zu (%.1f%%)\n", exact, out_size, 100.0 * exact / out_size);
         printf("Max diff: %d\n", max_diff);
         printf("Mean diff: %.3f\n", (double)sum_diff / out_size);

         if (max_diff == 0)
            printf("RESULT: PASS (bit-exact)\n");
         else if (max_diff <= 1)
            printf("RESULT: PASS (max_diff=%d, within HW tolerance)\n", max_diff);
         else {
            printf("RESULT: FAIL (max_diff=%d)\n", max_diff);
            result = 1;
         }

         free(golden);
         free(output);
      }
   }

   free(input);
   rnpu_model_free(m);
   rnpu_close(fd);
   printf("Done.\n");
   return result;
}
