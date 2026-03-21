/*
 * Test different batch sizes: submit 1, 2, 5, all jobs to find minimum
 * batch that makes the NPU execute successfully.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../include/rocketnpu.h"
#include "../src/rnpu_internal.h"

int main(int argc, char **argv)
{
   const char *model = argc > 1 ? argv[1] :
      "/root/npu-research/zero2pro_NPU_example/mobilenet_v1_1.0_224_quant.tflite";
   int fd = rnpu_open(NULL);
   rnpu_model_t *m = rnpu_model_load(fd, model);
   if (!m) return 1;

   int w, h, c;
   rnpu_get_input_dims(m, &w, &h, &c);
   size_t isz = w * h * c;
   uint8_t *input = malloc(isz);
   srand(42);
   for (size_t i = 0; i < isz; i++) input[i] = rand() & 0xFF;

   int batch_sizes[] = {1, 2, 3, 5, 10, 28};
   for (int bi = 0; bi < 6; bi++) {
      unsigned n = batch_sizes[bi];
      if (n > m->job_count) n = m->job_count;

      /* Reset activation buffer */
      memset(m->activation_bo.map, 0, m->activation_bo.size);
      extern void rnpu_convert_input(uint8_t *, const void *, unsigned,
                                     unsigned, unsigned, uint8_t);
      rnpu_convert_input(
         (uint8_t *)m->activation_bo.map +
            m->tensors[m->graph_input_tensor].offset,
         input, w, h, c, m->ops[0].input_zero_point);
      rnpu_bo_fini(fd, &m->activation_bo);

      int ret = rnpu_submit(fd, &m->jobs[0], n);
      rnpu_bo_prep(fd, &m->activation_bo);

      /* Check op0 output */
      uint8_t *t7 = (uint8_t *)m->activation_bo.map + m->tensors[7].offset;
      int nz = 0;
      for (int i = 0; i < 100000; i++)
         if (t7[i] != 0) nz++;

      printf("batch=%2u: submit=%d op0_nz=%d\n", n, ret, nz);
      fflush(stdout);
   }

   free(input);
   rnpu_model_free(m);
   rnpu_close(fd);
   return 0;
}
