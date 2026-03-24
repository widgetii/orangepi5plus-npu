/*
 * Minimal RKNPU submit test — matches RKNN's exact pattern
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <errno.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/wait.h>
#include <stdint.h>
#include <drm/drm.h>

struct rknpu_mem_create {
   uint32_t handle, flags;
   uint64_t size, obj_addr, dma_addr, sram_size;
   int32_t iommu_domain_id;
   uint32_t core_mask;
};
struct rknpu_mem_map { uint32_t handle, reserved; uint64_t offset; };
struct rknpu_mem_destroy { uint32_t handle, reserved; uint64_t obj_addr; };
struct rknpu_task {
   uint32_t flags, op_idx, enable_mask, int_mask, int_clear, int_status;
   uint32_t regcfg_amount, regcfg_offset;
   uint64_t regcmd_addr;
} __attribute__((packed));
struct rknpu_subcore_task { uint32_t task_start, task_number; };
struct rknpu_submit {
   uint32_t flags, timeout, task_start, task_number, task_counter;
   int32_t priority;
   uint64_t task_obj_addr;
   uint32_t iommu_domain_id, reserved;
   uint64_t task_base_addr;
   int64_t hw_elapse_time;
   uint32_t core_mask;
   int32_t fence_fd;
   struct rknpu_subcore_task subcore_task[5];
};
struct rknpu_action { uint32_t flags, value; };
struct rknpu_mem_sync { uint32_t flags, reserved; uint64_t obj_addr, offset, size; };

#define RKNPU_MEM_CACHEABLE      (1 << 1)
#define RKNPU_MEM_KERNEL_MAPPING (1 << 3)

#define DRM_IOCTL_RKNPU_ACTION      _IOWR(DRM_IOCTL_BASE, DRM_COMMAND_BASE + 0x00, struct rknpu_action)
#define DRM_IOCTL_RKNPU_SUBMIT      _IOWR(DRM_IOCTL_BASE, DRM_COMMAND_BASE + 0x01, struct rknpu_submit)
#define DRM_IOCTL_RKNPU_MEM_CREATE  _IOWR(DRM_IOCTL_BASE, DRM_COMMAND_BASE + 0x02, struct rknpu_mem_create)
#define DRM_IOCTL_RKNPU_MEM_MAP     _IOWR(DRM_IOCTL_BASE, DRM_COMMAND_BASE + 0x03, struct rknpu_mem_map)
#define DRM_IOCTL_RKNPU_MEM_DESTROY _IOWR(DRM_IOCTL_BASE, DRM_COMMAND_BASE + 0x04, struct rknpu_mem_destroy)
#define DRM_IOCTL_RKNPU_MEM_SYNC    _IOWR(DRM_IOCTL_BASE, DRM_COMMAND_BASE + 0x05, struct rknpu_mem_sync)

struct bo_info {
   uint32_t handle;
   uint64_t dma_addr, obj_addr;
   void *map;
   uint32_t size;
};

static int create_bo(int fd, uint32_t size, uint32_t flags, struct bo_info *bo) {
   struct rknpu_mem_create mc = { .size = size, .flags = flags };
   if (ioctl(fd, DRM_IOCTL_RKNPU_MEM_CREATE, &mc)) return -1;
   bo->handle = mc.handle; bo->dma_addr = mc.dma_addr; bo->obj_addr = mc.obj_addr; bo->size = size;
   struct rknpu_mem_map mm = { .handle = mc.handle };
   if (ioctl(fd, DRM_IOCTL_RKNPU_MEM_MAP, &mm)) return -1;
   bo->map = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, mm.offset);
   return (bo->map == MAP_FAILED) ? -1 : 0;
}

static void sync_bo(int fd, struct bo_info *bo, int to_device) {
   struct rknpu_mem_sync s = { .flags = to_device ? 1 : 2, .obj_addr = bo->obj_addr, .size = bo->size };
   ioctl(fd, DRM_IOCTL_RKNPU_MEM_SYNC, &s);
}

int main(int argc, char **argv)
{
   int fd = open("/dev/dri/card1", O_RDWR);
   if (fd < 0) { perror("open"); return 1; }

   /* Match RKNN: query version, HW version, IOMMU */
   struct drm_version ver = {0};
   ioctl(fd, DRM_IOCTL_VERSION, &ver);
   struct rknpu_action act = { .flags = 0 }; /* GET_HW_VERSION */
   ioctl(fd, DRM_IOCTL_RKNPU_ACTION, &act);
   printf("HW ver: 0x%x\n", act.value);
   act.flags = 18; /* GET_IOMMU_EN */
   ioctl(fd, DRM_IOCTL_RKNPU_ACTION, &act);
   printf("IOMMU: %d\n", act.value);

   /* Load regcmd */
   const char *regcmd_file = (argc > 1) ? argv[1] : NULL;
   uint8_t regcmd_data[4096] = {0};
   uint32_t regcmd_len = 16 * 8;
   if (regcmd_file) {
      FILE *f = fopen(regcmd_file, "rb");
      if (f) { regcmd_len = fread(regcmd_data, 1, 4096, f); fclose(f); }
   }
   uint32_t regcmd_entries = regcmd_len / 8;
   printf("Regcmd: %u bytes (%u entries)\n", regcmd_len, regcmd_entries);

   /* Create REGCMD BO (CACHEABLE, like RKNN) */
   struct bo_info regcmd_bo;
   if (create_bo(fd, 4096, RKNPU_MEM_CACHEABLE, &regcmd_bo)) {
      fprintf(stderr, "Failed to create regcmd BO\n"); return 1;
   }
   memcpy(regcmd_bo.map, regcmd_data, regcmd_len);
   sync_bo(fd, &regcmd_bo, 1); /* SYNC_TO_DEVICE */
   printf("Regcmd BO: dma=0x%llx\n", (unsigned long long)regcmd_bo.dma_addr);

   /* Create TASK BO (KERNEL_MAPPING + CACHEABLE) */
   struct bo_info task_bo;
   uint32_t task_bo_size = 4096; /* room for many tasks */
   if (create_bo(fd, task_bo_size, RKNPU_MEM_KERNEL_MAPPING | RKNPU_MEM_CACHEABLE, &task_bo)) {
      fprintf(stderr, "Failed to create task BO\n"); return 1;
   }
   printf("Task BO: dma=0x%llx obj=0x%llx\n",
          (unsigned long long)task_bo.dma_addr, (unsigned long long)task_bo.obj_addr);

   /* Fill task struct */
   memset(task_bo.map, 0, task_bo_size);
   struct rknpu_task *task = (struct rknpu_task *)task_bo.map;
   uint32_t regcfg_amount = regcmd_entries > 8 ? regcmd_entries - 8 : regcmd_entries;
   task->enable_mask = 0xf;
   task->int_mask = 0x300;
   task->int_clear = 0x1ffff;
   task->regcfg_amount = regcfg_amount;
   task->regcmd_addr = regcmd_bo.dma_addr;
   printf("Task: regcmd_addr=0x%llx regcfg_amount=%u\n",
          (unsigned long long)task->regcmd_addr, regcfg_amount);

   /* Sync task BO */
   sync_bo(fd, &task_bo, 1);

   /* Submit */
   struct rknpu_submit submit = {
      .flags = 0x5, /* PC | PINGPONG */
      .timeout = 2000,
      .task_number = 1,
      .task_obj_addr = task_bo.obj_addr,
      .core_mask = 0, /* auto */
      .fence_fd = -1,
      .subcore_task = { {0,1}, {0,1}, {0,1}, {0,1}, {0,1} },
   };

   printf("Submitting...\n");
   int ret = ioctl(fd, DRM_IOCTL_RKNPU_SUBMIT, &submit);
   if (ret) {
      fprintf(stderr, "SUBMIT failed: %s\n", strerror(errno));
   } else {
      printf("SUBMIT OK! hw_time=%lld us\n", (long long)submit.hw_elapse_time);
   }

   /* Verify task readback */
   struct rknpu_task *rb = (struct rknpu_task *)task_bo.map;
   printf("Task readback: int_status=0x%x (kernel writes completion status here)\n", rb->int_status);

   /* Cleanup */
   munmap(regcmd_bo.map, regcmd_bo.size);
   munmap(task_bo.map, task_bo.size);
   struct rknpu_mem_destroy md1 = { .handle = regcmd_bo.handle, .obj_addr = regcmd_bo.obj_addr };
   ioctl(fd, DRM_IOCTL_RKNPU_MEM_DESTROY, &md1);
   struct rknpu_mem_destroy md2 = { .handle = task_bo.handle, .obj_addr = task_bo.obj_addr };
   ioctl(fd, DRM_IOCTL_RKNPU_MEM_DESTROY, &md2);
   close(fd);
   return ret ? 1 : 0;
}
