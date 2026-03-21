/*
 * Rockchip IOMMU model for QEMU
 *
 * Emulates the Rockchip IOMMU used by the RK3588 NPU. Supports both
 * v1 (rockchip,iommu — mainline) and v2 (rockchip,iommu-v2 — vendor)
 * page table formats via DTE_ADDR-based page table walk.
 *
 * SPDX-License-Identifier: GPL-2.0-or-later
 */

#ifndef HW_MISC_ROCKCHIP_IOMMU_H
#define HW_MISC_ROCKCHIP_IOMMU_H

#include "hw/sysbus.h"
#include "qom/object.h"

#define TYPE_ROCKCHIP_IOMMU "rockchip-iommu"
OBJECT_DECLARE_SIMPLE_TYPE(RockchipIOMMUState, ROCKCHIP_IOMMU)

/* Register offsets */
#define RK_IOMMU_DTE_ADDR    0x00
#define RK_IOMMU_STATUS      0x04
#define RK_IOMMU_COMMAND     0x08
#define RK_IOMMU_PAGE_FAULT  0x0C
#define RK_IOMMU_ZAP_ONE     0x10
#define RK_IOMMU_INT_RAWSTAT 0x14
#define RK_IOMMU_INT_CLEAR   0x18
#define RK_IOMMU_INT_MASK    0x1C
#define RK_IOMMU_INT_STATUS  0x20
#define RK_IOMMU_AUTO_GATING 0x24

/* Status bits */
#define RK_IOMMU_STATUS_PAGING_ENABLED   (1 << 0)
#define RK_IOMMU_STATUS_STALL_ACTIVE     (1 << 2)
#define RK_IOMMU_STATUS_STALL_NOT_ACTIVE (1 << 3)
#define RK_IOMMU_STATUS_IDLE             (1 << 7)

/* Commands */
#define RK_IOMMU_CMD_ENABLE_PAGING   0
#define RK_IOMMU_CMD_DISABLE_PAGING  1
#define RK_IOMMU_CMD_ENABLE_STALL    2
#define RK_IOMMU_CMD_DISABLE_STALL   3
#define RK_IOMMU_CMD_ZAP_CACHE       4
#define RK_IOMMU_CMD_PAGE_FAULT_DONE 5
#define RK_IOMMU_CMD_FORCE_RESET     6

/* Number of MMIO instances (vendor kernel maps 4 MMU regions) */
#define RK_IOMMU_NUM_INSTANCES 4
#define RK_IOMMU_INSTANCE_SIZE 0x100

/* Per-instance register state */
typedef struct RkIOMMUInstance {
    MemoryRegion iomem;
    uint32_t dte_addr;         /* staging register (written by MMIO) */
    uint32_t active_dte_addr;  /* latched on ENABLE_PAGING */
    uint32_t status;
    uint32_t int_mask;
    uint32_t int_rawstat;
    uint32_t auto_gating;
    bool paging_enabled;
} RkIOMMUInstance;

struct RockchipIOMMUState {
    SysBusDevice parent_obj;

    RkIOMMUInstance instances[RK_IOMMU_NUM_INSTANCES];
    qemu_irq irq[3];

    /* Guest physical address space for page table reads */
    AddressSpace *dma_as;
};

/*
 * Translate an IOVA to a guest physical address using the page table.
 * Supports both v1 and v2 entry formats (auto-detected from entry bits).
 * Returns the GPA, or falls back to identity mapping if translation fails.
 */
hwaddr rk_iommu_translate(RockchipIOMMUState *s, uint32_t iova);

/*
 * Direct instance-level MMIO access (for NPU device to forward IOMMU
 * register accesses that fall within the NPU's 0x10000 region).
 */
uint64_t rk_iommu_instance_read(RkIOMMUInstance *inst, hwaddr addr,
                                 unsigned size);
void rk_iommu_instance_write(RkIOMMUInstance *inst, hwaddr addr,
                              uint64_t val, unsigned size);

#endif /* HW_MISC_ROCKCHIP_IOMMU_H */
