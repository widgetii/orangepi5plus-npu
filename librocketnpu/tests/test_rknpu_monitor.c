/*
 * Monitor NPU PC registers via /dev/mem in a tight loop.
 * Run this in background while submitting a job.
 * Usage: ./test_rknpu_monitor [duration_seconds]
 */
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <time.h>

int main(int argc, char **argv)
{
   int duration = argc > 1 ? atoi(argv[1]) : 5;
   int memfd = open("/dev/mem", O_RDONLY | O_SYNC);
   if (memfd < 0) { perror("open /dev/mem"); return 1; }

   volatile uint32_t *pc = mmap(NULL, 0x100, PROT_READ, MAP_SHARED, memfd, 0xfdab0000);
   if (pc == MAP_FAILED) { perror("mmap PC"); return 1; }

   volatile uint32_t *iommu = mmap(NULL, 0x100, PROT_READ, MAP_SHARED, memfd, 0xfdab9000);
   if (iommu == MAP_FAILED) { perror("mmap IOMMU"); return 1; }

   uint32_t prev_data_addr = 0;
   uint32_t prev_int_raw = 0;
   struct timespec start, now;
   clock_gettime(CLOCK_MONOTONIC, &start);

   printf("Monitoring NPU PC @ 0xfdab0000 for %d seconds...\n", duration);
   int sample = 0;
   while (1) {
      clock_gettime(CLOCK_MONOTONIC, &now);
      double elapsed = (now.tv_sec - start.tv_sec) + (now.tv_nsec - start.tv_nsec) / 1e9;
      if (elapsed > duration) break;

      uint32_t data_addr = pc[0x10/4];
      uint32_t data_amount = pc[0x14/4];
      uint32_t int_mask = pc[0x20/4];
      uint32_t int_raw = pc[0x2c/4];
      uint32_t task_ctrl = pc[0x30/4];
      uint32_t dma_base = pc[0x34/4];
      uint32_t op_en = pc[0x08/4];
      uint32_t iommu_dte = iommu[0x00/4];
      uint32_t iommu_status = iommu[0x04/4];
      uint32_t iommu_pf = iommu[0x0c/4];

      if (data_addr != prev_data_addr || int_raw != prev_int_raw || (sample % 10000 == 0)) {
         printf("[%.3fs] PC_DATA_ADDR=0x%08x AMOUNT=0x%x INT_MASK=0x%08x RAW=0x%08x "
                "TASK_CTRL=0x%08x DMA_BASE=0x%x OP_EN=%u "
                "IOMMU_DTE=0x%08x STATUS=0x%x PF=0x%x\n",
                elapsed, data_addr, data_amount, int_mask, int_raw,
                task_ctrl, dma_base, op_en,
                iommu_dte, iommu_status, iommu_pf);
         prev_data_addr = data_addr;
         prev_int_raw = int_raw;
      }
      sample++;
   }
   printf("Total samples: %d\n", sample);

   munmap((void *)pc, 0x100);
   munmap((void *)iommu, 0x100);
   close(memfd);
   return 0;
}
