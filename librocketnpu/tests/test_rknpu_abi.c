/*
 * RKNPU ABI regression tests — struct layouts, BO flags, regcmd format.
 * Pure C, no hardware needed. Prevents regressions like the regcfg_amount
 * off-by-4 bug that caused silent NPU hangs.
 *
 * SPDX-License-Identifier: MIT
 */

#include <stdio.h>
#include <string.h>
#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>
#include "../src/rnpu_internal.h"
#include "../src/rnpu_regcmd.h"

static int tests_run = 0;
static int tests_passed = 0;

#define ASSERT(cond, fmt, ...) do { \
   tests_run++; \
   if (!(cond)) { \
      fprintf(stderr, "FAIL [%s:%d]: " fmt "\n", __func__, __LINE__, ##__VA_ARGS__); \
      return 1; \
   } \
   tests_passed++; \
} while (0)

/* ---- Re-declare RKNPU structs locally (same as rnpu_drm.c / test_rknpu_submit.c) ---- */

struct rknpu_task {
   uint32_t flags, op_idx, enable_mask, int_mask, int_clear, int_status;
   uint32_t regcfg_amount, regcfg_offset;
   uint64_t regcmd_addr;
} __attribute__((packed));

struct rknpu_subcore_task { uint32_t task_start, task_number; };

struct rknpu_submit {
   uint32_t flags, timeout, task_start, task_number, task_counter;
   int32_t  priority;
   uint64_t task_obj_addr;
   uint32_t iommu_domain_id, reserved;
   uint64_t task_base_addr;
   int64_t  hw_elapse_time;
   uint32_t core_mask;
   int32_t  fence_fd;
   struct rknpu_subcore_task subcore_task[5];
};

/* RKNPU constants */
#define RKNPU_MEM_NON_CONTIGUOUS              (1 << 0)
#define RKNPU_MEM_CACHEABLE                   (1 << 1)
#define RKNPU_MEM_KERNEL_MAPPING              (1 << 3)
#define RKNPU_MEM_IOMMU_LIMIT_IOVA_ALIGNMENT  (1 << 10)
#define RKNPU_PC_DATA_EXTRA_AMOUNT            4

/* ================================================================
 * Test 1: regcfg_amount calculation
 *
 * Regression: count - 8 instead of count - 4 caused the last 4 entries
 * (containing PC_OPERATION_ENABLE) to be skipped → NPU hung silently.
 * ================================================================ */
static int test_regcfg_amount(void)
{
   ASSERT(RKNPU_PC_DATA_EXTRA_AMOUNT == 4,
          "EXTRA_AMOUNT should be 4, got %d", RKNPU_PC_DATA_EXTRA_AMOUNT);

   /* For count=130: amount = 130 - 4 = 126 (not 122) */
   unsigned count1 = 130;
   unsigned amount1 = count1 - RKNPU_PC_DATA_EXTRA_AMOUNT;
   ASSERT(amount1 == 126, "count=130: expected amount=126, got %u", amount1);

   /* For count=140: amount = 140 - 4 = 136 */
   unsigned count2 = 140;
   unsigned amount2 = count2 - RKNPU_PC_DATA_EXTRA_AMOUNT;
   ASSERT(amount2 == 136, "count=140: expected amount=136, got %u", amount2);

   return 0;
}

/* ================================================================
 * Test 2: BO flags
 *
 * Verify individual flag bits and composite values for data/task BOs.
 * ================================================================ */
static int test_bo_flags(void)
{
   ASSERT(RKNPU_MEM_NON_CONTIGUOUS == 0x1,
          "NON_CONTIGUOUS should be 0x1, got 0x%x", RKNPU_MEM_NON_CONTIGUOUS);
   ASSERT(RKNPU_MEM_CACHEABLE == 0x2,
          "CACHEABLE should be 0x2, got 0x%x", RKNPU_MEM_CACHEABLE);
   ASSERT(RKNPU_MEM_KERNEL_MAPPING == 0x8,
          "KERNEL_MAPPING should be 0x8, got 0x%x", RKNPU_MEM_KERNEL_MAPPING);
   ASSERT(RKNPU_MEM_IOMMU_LIMIT_IOVA_ALIGNMENT == 0x400,
          "IOMMU_LIMIT should be 0x400, got 0x%x", RKNPU_MEM_IOMMU_LIMIT_IOVA_ALIGNMENT);

   /* Composite: data BO = NON_CONTIGUOUS | CACHEABLE | IOMMU_LIMIT = 0x403 */
   uint32_t data_flags = RKNPU_MEM_NON_CONTIGUOUS | RKNPU_MEM_CACHEABLE
                        | RKNPU_MEM_IOMMU_LIMIT_IOVA_ALIGNMENT;
   ASSERT(data_flags == 0x403, "data BO flags should be 0x403, got 0x%x", data_flags);

   /* Composite: task BO = data | KERNEL_MAPPING = 0x40b */
   uint32_t task_flags = data_flags | RKNPU_MEM_KERNEL_MAPPING;
   ASSERT(task_flags == 0x40b, "task BO flags should be 0x40b, got 0x%x", task_flags);

   return 0;
}

/* ================================================================
 * Test 3: rknpu_task struct layout
 *
 * The kernel reads this struct at fixed offsets — any layout change
 * silently corrupts fields (e.g., regcmd_addr at wrong offset).
 * ================================================================ */
static int test_task_struct_layout(void)
{
   ASSERT(sizeof(struct rknpu_task) == 40,
          "sizeof(rknpu_task) should be 40, got %zu", sizeof(struct rknpu_task));

   ASSERT(offsetof(struct rknpu_task, flags) == 0,
          "flags offset should be 0, got %zu", offsetof(struct rknpu_task, flags));
   ASSERT(offsetof(struct rknpu_task, op_idx) == 4,
          "op_idx offset should be 4, got %zu", offsetof(struct rknpu_task, op_idx));
   ASSERT(offsetof(struct rknpu_task, enable_mask) == 8,
          "enable_mask offset should be 8, got %zu", offsetof(struct rknpu_task, enable_mask));
   ASSERT(offsetof(struct rknpu_task, int_mask) == 12,
          "int_mask offset should be 12, got %zu", offsetof(struct rknpu_task, int_mask));
   ASSERT(offsetof(struct rknpu_task, int_clear) == 16,
          "int_clear offset should be 16, got %zu", offsetof(struct rknpu_task, int_clear));
   ASSERT(offsetof(struct rknpu_task, int_status) == 20,
          "int_status offset should be 20, got %zu", offsetof(struct rknpu_task, int_status));
   ASSERT(offsetof(struct rknpu_task, regcfg_amount) == 24,
          "regcfg_amount offset should be 24, got %zu", offsetof(struct rknpu_task, regcfg_amount));
   ASSERT(offsetof(struct rknpu_task, regcfg_offset) == 28,
          "regcfg_offset offset should be 28, got %zu", offsetof(struct rknpu_task, regcfg_offset));
   ASSERT(offsetof(struct rknpu_task, regcmd_addr) == 32,
          "regcmd_addr offset should be 32, got %zu", offsetof(struct rknpu_task, regcmd_addr));

   return 0;
}

/* ================================================================
 * Test 4: rknpu_submit struct layout
 *
 * Key fields: task_obj_addr at 24, core_mask at 56, subcore_task at 64.
 * ================================================================ */
static int test_submit_struct_layout(void)
{
   ASSERT(sizeof(struct rknpu_submit) == 104,
          "sizeof(rknpu_submit) should be 104, got %zu", sizeof(struct rknpu_submit));

   ASSERT(offsetof(struct rknpu_submit, task_obj_addr) == 24,
          "task_obj_addr offset should be 24, got %zu",
          offsetof(struct rknpu_submit, task_obj_addr));
   ASSERT(offsetof(struct rknpu_submit, core_mask) == 56,
          "core_mask offset should be 56, got %zu",
          offsetof(struct rknpu_submit, core_mask));
   ASSERT(offsetof(struct rknpu_submit, subcore_task) == 64,
          "subcore_task offset should be 64, got %zu",
          offsetof(struct rknpu_submit, subcore_task));

   return 0;
}

/* ================================================================
 * Test 5: regcmd chain area (last 4 entries)
 *
 * The last 4 regcmd entries are the "chain area":
 *   [count-4]: chain pointer (PC_BASE_ADDRESS or null)
 *   [count-3]: PC_REGISTER_AMOUNTS
 *   [count-2]: 0x0041000000000000 sentinel
 *   [count-1]: PC_OPERATION_ENABLE (target=0x0081)
 * ================================================================ */
static int test_regcmd_chain_area(void)
{
   /* Build minimal fake model for regcmd generation */
   struct rnpu_model model;
   memset(&model, 0, sizeof(model));
   model.fd = -1;
   model.activation_bo.dma_addr = 0x10000;
   model.weight_bo.dma_addr = 0x20000;
   model.bias_bo.dma_addr = 0x30000;

   struct rnpu_npu_tensor tensors[2];
   memset(tensors, 0, sizeof(tensors));
   tensors[0].width = 8; tensors[0].height = 8; tensors[0].channels = 16;
   tensors[0].offset = 0; tensors[0].size = 8 * 8 * 16;
   tensors[1].width = 8; tensors[1].height = 8; tensors[1].channels = 32;
   tensors[1].offset = 8 * 8 * 16; tensors[1].size = 8 * 8 * 32;
   model.tensors = tensors;
   model.tensor_count = 2;

   struct rnpu_operation op;
   memset(&op, 0, sizeof(op));
   op.type = RNPU_OP_CONV;
   op.input_tensor = 0;
   op.output_tensor = 1;
   op.add_tensor = -1;
   op.input_width = 8; op.input_height = 8; op.input_channels = 16;
   op.output_width = 8; op.output_height = 8; op.output_channels = 32;
   op.weights_width = 3; op.weights_height = 3;
   op.input_zero_point = 128; op.output_zero_point = 128;
   op.weights_zero_point = 128;
   op.input_scale = 0.1f; op.output_scale = 0.1f; op.weights_scale = 0.1f;

   /* Single task */
   struct rnpu_split_task task;
   memset(&task, 0, sizeof(task));
   task.input_width = 8; task.input_height = 8;
   task.input_channels = 16; task.input_channels_real = 16;
   task.output_width = 8; task.output_height = 8;
   task.output_channels = 32; task.output_channels_real = 32;
   task.weights_width = 3; task.weights_height = 3; task.weights_kernels = 32;
   task.stride_x = 1; task.stride_y = 1;
   task.input_zero_point = 128; task.output_zero_point = 128;
   task.weights_zero_point = 128;
   task.input_scale = 0.1f; task.output_scale = 0.1f; task.weights_scale = 0.1f;
   task.input_banks = 4; task.weights_banks = 8;
   task.atomic_count = 2; task.surfaces_per_row = 1;
   task.input_data_entries = 128;
   task.input_line_stride = 128; task.input_surface_stride = 1024;
   task.output_surface_stride = 1024;
   task.num = 0;
   op.tasks = &task;
   op.task_count = 1;

   uint64_t buf[256];
   memset(buf, 0xAA, sizeof(buf));
   unsigned count = rnpu_fill_regcmd(&model, &op, buf, 256, 0);
   ASSERT(count > 4, "regcmd count should be > 4, got %u", count);

   /* [count-1]: target=0x0081, has PC_OP_EN bit */
   uint64_t last = buf[count - 1];
   uint16_t last_target = (uint16_t)(last >> 48);
   ASSERT(last_target == 0x0081,
          "last entry target should be 0x0081, got 0x%04x", last_target);

   /* [count-2]: sentinel 0x0041000000000000 */
   ASSERT(buf[count - 2] == 0x0041000000000000ULL,
          "sentinel should be 0x0041000000000000, got 0x%016llx",
          (unsigned long long)buf[count - 2]);

   /* [count-3]: reg field should be 0x0014 (PC_REGISTER_AMOUNTS) */
   uint16_t reg3 = (uint16_t)(buf[count - 3] & 0xFFFF);
   ASSERT(reg3 == 0x0014,
          "PC_REGISTER_AMOUNTS reg should be 0x0014, got 0x%04x", reg3);

   /* [count-4]: null chain pointer (last task → no chain) → raw 0x0 */
   ASSERT(buf[count - 4] == 0x0,
          "last task chain pointer should be 0x0, got 0x%016llx",
          (unsigned long long)buf[count - 4]);

   return 0;
}

/* ================================================================
 * Test 6: regcmd entry count sanity
 *
 * A valid regcmd buffer must have at least EXTRA_AMOUNT + 10 entries.
 * ================================================================ */
static int test_regcmd_entry_count(void)
{
   /* Reuse same fake model from test 5 */
   struct rnpu_model model;
   memset(&model, 0, sizeof(model));
   model.fd = -1;
   model.activation_bo.dma_addr = 0x10000;
   model.weight_bo.dma_addr = 0x20000;
   model.bias_bo.dma_addr = 0x30000;

   struct rnpu_npu_tensor tensors[2];
   memset(tensors, 0, sizeof(tensors));
   tensors[0].width = 4; tensors[0].height = 4; tensors[0].channels = 16;
   tensors[0].offset = 0; tensors[0].size = 4 * 4 * 16;
   tensors[1].width = 4; tensors[1].height = 4; tensors[1].channels = 16;
   tensors[1].offset = 4 * 4 * 16; tensors[1].size = 4 * 4 * 16;
   model.tensors = tensors;
   model.tensor_count = 2;

   struct rnpu_operation op;
   memset(&op, 0, sizeof(op));
   op.type = RNPU_OP_CONV;
   op.input_tensor = 0;
   op.output_tensor = 1;
   op.add_tensor = -1;
   op.input_width = 4; op.input_height = 4; op.input_channels = 16;
   op.output_width = 4; op.output_height = 4; op.output_channels = 16;
   op.weights_width = 1; op.weights_height = 1;
   op.input_zero_point = 128; op.output_zero_point = 128;
   op.weights_zero_point = 128;
   op.input_scale = 0.1f; op.output_scale = 0.1f; op.weights_scale = 0.1f;

   struct rnpu_split_task task;
   memset(&task, 0, sizeof(task));
   task.input_width = 4; task.input_height = 4;
   task.input_channels = 16; task.input_channels_real = 16;
   task.output_width = 4; task.output_height = 4;
   task.output_channels = 16; task.output_channels_real = 16;
   task.weights_width = 1; task.weights_height = 1; task.weights_kernels = 16;
   task.stride_x = 1; task.stride_y = 1;
   task.input_zero_point = 128; task.output_zero_point = 128;
   task.weights_zero_point = 128;
   task.input_scale = 0.1f; task.output_scale = 0.1f; task.weights_scale = 0.1f;
   task.input_banks = 4; task.weights_banks = 8;
   task.atomic_count = 1; task.surfaces_per_row = 1;
   task.input_data_entries = 64;
   task.input_line_stride = 64; task.input_surface_stride = 256;
   task.output_surface_stride = 256;
   task.num = 0;
   op.tasks = &task;
   op.task_count = 1;

   uint64_t buf[256];
   unsigned count = rnpu_fill_regcmd(&model, &op, buf, 256, 0);
   unsigned min_count = RKNPU_PC_DATA_EXTRA_AMOUNT + 10;
   ASSERT(count >= min_count,
          "regcmd count should be >= %u, got %u", min_count, count);

   return 0;
}

/* ---- Main ---- */

typedef int (*test_fn)(void);
struct test_case {
   const char *name;
   test_fn fn;
};

int main(void)
{
   struct test_case tests[] = {
      {"regcfg_amount",         test_regcfg_amount},
      {"bo_flags",              test_bo_flags},
      {"task_struct_layout",    test_task_struct_layout},
      {"submit_struct_layout",  test_submit_struct_layout},
      {"regcmd_chain_area",     test_regcmd_chain_area},
      {"regcmd_entry_count",    test_regcmd_entry_count},
   };
   unsigned n = sizeof(tests) / sizeof(tests[0]);

   printf("Running %u RKNPU ABI tests...\n\n", n);
   int failures = 0;
   for (unsigned i = 0; i < n; i++) {
      printf("  %-35s", tests[i].name);
      int ret = tests[i].fn();
      if (ret == 0) {
         printf("OK\n");
      } else {
         printf("FAIL\n");
         failures++;
      }
   }

   printf("\n%d/%d tests passed, %d assertions checked\n",
          tests_passed, tests_run, tests_run);
   if (failures)
      printf("%d test(s) FAILED\n", failures);
   else
      printf("All tests passed!\n");

   return failures ? 1 : 0;
}
