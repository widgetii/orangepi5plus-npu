/*
 * Full debug: model load + invoke + check ALL intermediate tensors
 * + dump op 1 regcmd details. ALL in one process to avoid BO cleanup races.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "../include/rocketnpu.h"
#include "../src/rnpu_internal.h"

static double now_ms(void) {
   struct timespec ts;
   clock_gettime(CLOCK_MONOTONIC, &ts);
   return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

int main(int argc, char **argv)
{
   const char *model = argc > 1 ? argv[1] :
      "/root/npu-research/zero2pro_NPU_example/mobilenet_v1_1.0_224_quant.tflite";

   int fd = rnpu_open(NULL);
   rnpu_model_t *m = rnpu_model_load(fd, model);
   if (!m) return 1;

   /* Print all op details */
   printf("\n=== %u operations ===\n", m->op_count);
   for (unsigned i = 0; i < m->op_count && i < 5; i++) {
      struct rnpu_operation *op = &m->ops[i];
      printf("Op%u: dw=%d stride=%d in=t%u(%ux%ux%u) out=t%u(%ux%ux%u) tasks=%u\n",
             i, op->depthwise, op->stride,
             op->input_tensor, op->input_width, op->input_height, op->input_channels,
             op->output_tensor, op->output_width, op->output_height, op->output_channels,
             op->task_count);
   }

   /* Dump op 1 regcmd key registers */
   if (m->op_count > 1) {
      struct rnpu_operation *op1 = &m->ops[1];
      uint64_t *rc = (uint64_t *)((uint8_t *)m->regcmd_bo.map + op1->regcmd_offset);
      unsigned cnt = op1->tasks[0].regcfg_amount;
      uint64_t act = m->activation_bo.dma_addr;
      uint64_t wt = m->weight_bo.dma_addr;
      uint64_t bias = m->bias_bo.dma_addr;
      printf("\n=== Op1 regcmd (%u entries) ===\n", cnt);
      for (unsigned i = 0; i < cnt; i++) {
         uint32_t r = rc[i] & 0xFFFF, v = (rc[i] >> 16) & 0xFFFFFFFF;
         if (r == 0x1070) printf("[%u] FEAT_ADDR=0x%x exp=0x%x\n", i, v,
            (uint32_t)(act + m->tensors[op1->input_tensor].offset));
         if (r == 0x1110) printf("[%u] WT_ADDR=0x%x exp=0x%x\n", i, v,
            (uint32_t)(wt + op1->weight_offset));
         if (r == 0x4020) printf("[%u] DST_ADDR=0x%x exp=0x%x\n", i, v,
            (uint32_t)(act + m->tensors[op1->output_tensor].offset));
         if (r == 0x5020) printf("[%u] BIAS_ADDR=0x%x exp=0x%x\n", i, v,
            (uint32_t)(bias + op1->bias_offset));
         if (r == 0x100c) printf("[%u] CONV_CON1=0x%08x\n", i, v);
         if (r == 0x3014) printf("[%u] MISC_CFG=0x%08x\n", i, v);
         if (r == 0x1020) printf("[%u] DATA_SIZE0=0x%08x\n", i, v);
         if (r == 0x1024) printf("[%u] DATA_SIZE1=0x%08x\n", i, v);
         if (r == 0x1028) printf("[%u] DATA_SIZE2=0x%08x\n", i, v);
         if (r == 0x102c) printf("[%u] DATA_SIZE3=0x%08x\n", i, v);
         if (r == 0x1038) printf("[%u] WT_SIZE2=0x%08x\n", i, v);
         if (r == 0x1040) printf("[%u] CBUF_CON0=0x%08x\n", i, v);
      }
   }

   /* Invoke with random data */
   int w, h, c;
   rnpu_get_input_dims(m, &w, &h, &c);
   size_t isz = w * h * c;
   uint8_t *input = malloc(isz);
   srand(42);
   for (size_t i = 0; i < isz; i++) input[i] = rand() & 0xFF;

   printf("\n=== Invoke ===\n");
   double t0 = now_ms();
   int ret = rnpu_invoke(m, input, isz);
   double dt = now_ms() - t0;
   printf("ret=%d %.2f ms\n", ret, dt);

   /* Check ALL tensors that have data */
   printf("\n=== Tensor check ===\n");
   uint8_t *buf = malloc(1024 * 1024);
   for (unsigned i = 0; i < m->op_count && i < 10; i++) {
      unsigned ti = m->ops[i].output_tensor;
      int tw, th, tc;
      int got = rnpu_get_tensor(m, ti, buf, 1024 * 1024, &tw, &th, &tc);
      if (got <= 0) continue;
      int unique[256] = {0};
      for (int j = 0; j < got; j++) unique[buf[j]]++;
      int cnt = 0;
      for (int j = 0; j < 256; j++) if (unique[j]) cnt++;
      printf("Op%u→t%u (%dx%dx%d): %d unique, first 4: %d %d %d %d\n",
             i, ti, tw, th, tc, cnt, buf[0], buf[1], buf[2], buf[3]);
   }
   free(buf);

   printf("\n=== Done, freeing model ===\n");
   free(input);
   rnpu_model_free(m);
   printf("Model freed OK\n");
   rnpu_close(fd);
   printf("Device closed OK\n");
   return 0;
}
