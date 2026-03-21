/*
 * Debug test: runs model and checks intermediate tensor values.
 * Uses the safe API (rnpu_get_tensor) instead of raw buffer access.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "../include/rocketnpu.h"

static double now_ms(void) {
   struct timespec ts;
   clock_gettime(CLOCK_MONOTONIC, &ts);
   return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

int main(int argc, char **argv)
{
   if (argc < 2) {
      fprintf(stderr, "Usage: %s model.tflite [input.bin]\n", argv[0]);
      return 1;
   }
   int fd = rnpu_open(NULL);
   rnpu_model_t *m = rnpu_model_load(fd, argv[1]);
   if (!m) return 1;

   int w, h, c;
   rnpu_get_input_dims(m, &w, &h, &c);
   size_t isz = w * h * c;
   uint8_t *input = malloc(isz);

   if (argc > 2) {
      FILE *f = fopen(argv[2], "rb");
      if (f) { fread(input, 1, isz, f); fclose(f); }
   } else {
      srand(42);
      for (size_t i = 0; i < isz; i++) input[i] = rand() & 0xFF;
   }

   double t0 = now_ms();
   int ret = rnpu_invoke(m, input, isz);
   double dt = now_ms() - t0;
   printf("Invoke: ret=%d, %.2f ms\n", ret, dt);

   /* Check key intermediate tensors.
    * For MBv1: op 0 outputs tensor 7 (112x112x32).
    * Try tensors 1, 7, 33, 37 which are early conv outputs. */
   int check_tensors[] = {7, 33, 37, 1};
   for (int i = 0; i < 4; i++) {
      int tw, th, tc;
      uint8_t buf[1024*1024];
      int got = rnpu_get_tensor(m, check_tensors[i], buf, sizeof(buf), &tw, &th, &tc);
      if (got <= 0) {
         printf("Tensor %d: empty or error (%d)\n", check_tensors[i], got);
         continue;
      }
      int unique[256] = {0};
      for (int j = 0; j < got; j++) unique[buf[j]]++;
      int cnt = 0;
      for (int j = 0; j < 256; j++) if (unique[j]) cnt++;
      printf("Tensor %3d (%3dx%3dx%3d): %3d unique vals, first 8: ",
             check_tensors[i], tw, th, tc, cnt);
      for (int j = 0; j < 8 && j < got; j++) printf("%d ", buf[j]);
      printf("\n");
   }

   /* Also save the graph outputs */
   int n = rnpu_output_count(m);
   for (int i = 0; i < n; i++) {
      int ow, oh, oc;
      rnpu_get_output_dims(m, i, &ow, &oh, &oc);
      size_t osz = ow * oh * oc;
      uint8_t *out = malloc(osz);
      rnpu_get_output(m, i, out, osz);
      int unique[256] = {0};
      for (size_t j = 0; j < osz; j++) unique[out[j]]++;
      int cnt = 0; for (int j = 0; j < 256; j++) if (unique[j]) cnt++;
      printf("Output %d (%dx%dx%d): %d unique vals, first 8: ",
             i, ow, oh, oc, cnt);
      for (int j = 0; j < 8 && j < (int)osz; j++) printf("%d ", out[j]);
      printf("\n");
      char fn[64]; snprintf(fn, 64, "npu_out_%d.bin", i);
      FILE *f = fopen(fn, "wb"); fwrite(out, 1, osz, f); fclose(f);
      free(out);
   }

   free(input);
   rnpu_model_free(m);
   rnpu_close(fd);
   return 0;
}
