/*
 * Rockchip IOMMU v2 model for QEMU
 *
 * Minimal emulation for the vendor rknpu driver. The Rockchip IOMMU uses
 * a two-level page table (DTE -> PTE -> page). The vendor kernel programs
 * DTE_ADDR with the physical address of the directory table, then issues
 * ENABLE_PAGING. We walk the page table to translate IOVAs for NPU DMA.
 *
 * v2 page table entry format:
 *   phys = (entry & 0xFFFFF000) | ((entry & 0xFF0) << 28)
 *   This encodes up to 40-bit physical addresses.
 *
 * SPDX-License-Identifier: GPL-2.0-or-later
 */

#include "qemu/osdep.h"
#include "qemu/log.h"
#include "hw/misc/rockchip-iommu.h"
#include "hw/irq.h"
#include "hw/qdev-properties.h"
#include "system/address-spaces.h"
#include "migration/vmstate.h"

/* Page table constants */
#define RK_IOMMU_PAGE_SIZE   0x1000
#define RK_IOMMU_PAGE_MASK   0xFFFFF000ULL
#define RK_IOMMU_DTE_COUNT   1024  /* 10-bit DTE index: IOVA[31:22] */
#define RK_IOMMU_PTE_COUNT   1024  /* 10-bit PTE index: IOVA[21:12] */

/* Page table entry: bit 0 = present */
#define RK_IOMMU_ENTRY_PRESENT  (1 << 0)

/*
 * Decode a page table entry to a physical address.
 *
 * Rockchip IOMMU has two formats:
 *   v1 DTE: phys = entry & 0xFFFFFC00  (1KB-aligned page table base)
 *   v1 PTE: phys = entry & 0xFFFFF000  (4KB page)
 *   v2 DTE: phys = (entry & 0xFFFFF000) | ((entry >> 4) & 0xFF) << 32
 *   v2 PTE: same as v2 DTE
 *
 * For systems with <=4GB RAM (our case), bits 39:32 are always 0, so
 * v2 high-bit encoding is irrelevant. We use v1 masks which are wider
 * and correctly decode both v1 and v2 entries in the <4GB range.
 */
static hwaddr rk_iommu_decode_dte(uint32_t entry)
{
    return (hwaddr)(entry & 0xFFFFFC00ULL);
}

static hwaddr rk_iommu_decode_pte(uint32_t entry)
{
    return (hwaddr)(entry & 0xFFFFF000ULL);
}

hwaddr rk_iommu_translate(RockchipIOMMUState *s, uint32_t iova)
{
    /*
     * Find the most recently activated page table.
     * The kernel programs all instances with the same DTE, but may
     * transiently enable a default domain before the real one.
     * Use the last instance (highest index) since the kernel writes
     * all 4 instances sequentially — the last one has the final DTE.
     */
    uint32_t dte_addr = 0;
    for (int i = RK_IOMMU_NUM_INSTANCES - 1; i >= 0; i--) {
        if (s->instances[i].paging_enabled && s->instances[i].active_dte_addr) {
            dte_addr = s->instances[i].active_dte_addr;
            break;
        }
    }

    if (!dte_addr) {
        return (hwaddr)iova;
    }

    /* Strip valid/present bit from DTE_ADDR register value.
     * DTE_ADDR is page-aligned with bit 0 = valid flag. */
    hwaddr dt_base = (hwaddr)(dte_addr & ~1U);

    /* DTE index from IOVA[31:22] */
    uint32_t dte_idx = (iova >> 22) & 0x3FF;
    uint32_t offset = iova & 0xFFF;

    /* Read DTE */
    uint32_t dte;
    address_space_read(s->dma_as, dt_base + dte_idx * 4,
                       MEMTXATTRS_UNSPECIFIED, &dte, sizeof(dte));
    dte = le32_to_cpu(dte);

    if (!(dte & RK_IOMMU_ENTRY_PRESENT)) {
        qemu_log_mask(LOG_GUEST_ERROR,
                      "rockchip-iommu: DTE[%u] not present for IOVA 0x%08x\n",
                      dte_idx, iova);
        return (hwaddr)iova;
    }

    /* Decode page table base from DTE (v1: 1KB-aligned) */
    hwaddr pt_base = rk_iommu_decode_dte(dte);

    /* PTE index from IOVA[21:12] */
    uint32_t pte_idx = (iova >> 12) & 0x3FF;

    /* Read PTE */
    uint32_t pte;
    address_space_read(s->dma_as, pt_base + pte_idx * 4,
                       MEMTXATTRS_UNSPECIFIED, &pte, sizeof(pte));
    pte = le32_to_cpu(pte);

    if (!(pte & RK_IOMMU_ENTRY_PRESENT)) {
        qemu_log_mask(LOG_GUEST_ERROR,
                      "rockchip-iommu: PTE[%u] not present for IOVA 0x%08x "
                      "(DTE=0x%08x pt_base=0x%lx)\n",
                      pte_idx, iova, dte, (unsigned long)pt_base);
        return (hwaddr)iova;
    }

    /* Decode page physical address from PTE (4KB page) */
    hwaddr page_phys = rk_iommu_decode_pte(pte);
    return page_phys + offset;
}

/* ======================================================================
 * MMIO read/write handlers (per-instance)
 * ====================================================================== */

uint64_t rk_iommu_instance_read(RkIOMMUInstance *inst, hwaddr addr,
                                 unsigned size)
{

    switch (addr) {
    case RK_IOMMU_DTE_ADDR:
        return inst->dte_addr;
    case RK_IOMMU_STATUS:
        return inst->status;
    case RK_IOMMU_INT_MASK:
        return inst->int_mask;
    case RK_IOMMU_INT_RAWSTAT:
        return inst->int_rawstat;
    case RK_IOMMU_INT_STATUS:
        return inst->int_rawstat & inst->int_mask;
    case RK_IOMMU_AUTO_GATING:
        return inst->auto_gating;
    case RK_IOMMU_PAGE_FAULT:
        return 0;
    default:
        return 0;
    }
}

void rk_iommu_instance_write(RkIOMMUInstance *inst, hwaddr addr,
                              uint64_t val, unsigned size)
{

    switch (addr) {
    case RK_IOMMU_DTE_ADDR:
        inst->dte_addr = (uint32_t)val;
        qemu_log_mask(LOG_UNIMP,
                      "rockchip-iommu: DTE_ADDR = 0x%08x\n",
                      inst->dte_addr);
        break;

    case RK_IOMMU_COMMAND:
        switch ((uint32_t)val) {
        case RK_IOMMU_CMD_ENABLE_PAGING:
            inst->paging_enabled = true;
            inst->active_dte_addr = inst->dte_addr;
            inst->status |= RK_IOMMU_STATUS_PAGING_ENABLED;
            qemu_log_mask(LOG_UNIMP,
                          "rockchip-iommu: ENABLE_PAGING (DTE=0x%08x)\n",
                          inst->active_dte_addr);
            break;
        case RK_IOMMU_CMD_DISABLE_PAGING:
            inst->paging_enabled = false;
            inst->active_dte_addr = 0;
            inst->status &= ~RK_IOMMU_STATUS_PAGING_ENABLED;
            break;
        case RK_IOMMU_CMD_ENABLE_STALL:
            inst->status &= ~RK_IOMMU_STATUS_STALL_NOT_ACTIVE;
            inst->status |= RK_IOMMU_STATUS_STALL_ACTIVE;
            break;
        case RK_IOMMU_CMD_DISABLE_STALL:
            inst->status |= RK_IOMMU_STATUS_STALL_NOT_ACTIVE;
            inst->status &= ~RK_IOMMU_STATUS_STALL_ACTIVE;
            break;
        case RK_IOMMU_CMD_ZAP_CACHE:
            /* No cache to zap */
            break;
        case RK_IOMMU_CMD_FORCE_RESET:
            inst->dte_addr = 0;
            inst->active_dte_addr = 0;
            inst->paging_enabled = false;
            inst->status = RK_IOMMU_STATUS_IDLE |
                           RK_IOMMU_STATUS_STALL_NOT_ACTIVE |
                           RK_IOMMU_STATUS_STALL_ACTIVE;
            inst->int_mask = 0;
            inst->int_rawstat = 0;
            break;
        default:
            break;
        }
        break;

    case RK_IOMMU_INT_MASK:
        inst->int_mask = (uint32_t)val;
        break;

    case RK_IOMMU_INT_CLEAR:
        inst->int_rawstat &= ~(uint32_t)val;
        break;

    case RK_IOMMU_AUTO_GATING:
        inst->auto_gating = (uint32_t)val;
        break;

    default:
        break;
    }
}

static uint64_t rk_iommu_read(void *opaque, hwaddr addr, unsigned size)
{
    return rk_iommu_instance_read(opaque, addr, size);
}

static void rk_iommu_write(void *opaque, hwaddr addr, uint64_t val,
                            unsigned size)
{
    rk_iommu_instance_write(opaque, addr, val, size);
}

static const MemoryRegionOps rk_iommu_ops = {
    .read = rk_iommu_read,
    .write = rk_iommu_write,
    .endianness = DEVICE_LITTLE_ENDIAN,
    .impl = { .min_access_size = 4, .max_access_size = 4 },
};

/* ======================================================================
 * Device lifecycle
 * ====================================================================== */

static void rk_iommu_realize(DeviceState *dev, Error **errp)
{
    RockchipIOMMUState *s = ROCKCHIP_IOMMU(dev);
    SysBusDevice *sbd = SYS_BUS_DEVICE(dev);

    s->dma_as = &address_space_memory;

    for (int i = 0; i < RK_IOMMU_NUM_INSTANCES; i++) {
        RkIOMMUInstance *inst = &s->instances[i];
        char name[32];
        snprintf(name, sizeof(name), "rockchip-iommu-%d", i);
        inst->status = RK_IOMMU_STATUS_IDLE |
                       RK_IOMMU_STATUS_STALL_NOT_ACTIVE |
                       RK_IOMMU_STATUS_STALL_ACTIVE;
        memory_region_init_io(&inst->iomem, OBJECT(dev), &rk_iommu_ops,
                              inst, name, RK_IOMMU_INSTANCE_SIZE);
        sysbus_init_mmio(sbd, &inst->iomem);
    }

    for (int i = 0; i < 3; i++) {
        sysbus_init_irq(sbd, &s->irq[i]);
    }
}

static void rk_iommu_reset(DeviceState *dev)
{
    RockchipIOMMUState *s = ROCKCHIP_IOMMU(dev);

    for (int i = 0; i < RK_IOMMU_NUM_INSTANCES; i++) {
        RkIOMMUInstance *inst = &s->instances[i];
        inst->dte_addr = 0;
        inst->active_dte_addr = 0;
        inst->status = RK_IOMMU_STATUS_IDLE |
                       RK_IOMMU_STATUS_STALL_NOT_ACTIVE |
                       RK_IOMMU_STATUS_STALL_ACTIVE;
        inst->int_mask = 0;
        inst->int_rawstat = 0;
        inst->auto_gating = 0;
        inst->paging_enabled = false;
    }
}

static const VMStateDescription vmstate_rk_iommu_instance = {
    .name = "rockchip-iommu-instance",
    .version_id = 1,
    .minimum_version_id = 1,
    .fields = (const VMStateField[]) {
        VMSTATE_UINT32(dte_addr, RkIOMMUInstance),
        VMSTATE_UINT32(active_dte_addr, RkIOMMUInstance),
        VMSTATE_UINT32(status, RkIOMMUInstance),
        VMSTATE_UINT32(int_mask, RkIOMMUInstance),
        VMSTATE_UINT32(int_rawstat, RkIOMMUInstance),
        VMSTATE_UINT32(auto_gating, RkIOMMUInstance),
        VMSTATE_BOOL(paging_enabled, RkIOMMUInstance),
        VMSTATE_END_OF_LIST()
    }
};

static const VMStateDescription vmstate_rk_iommu = {
    .name = "rockchip-iommu",
    .version_id = 1,
    .minimum_version_id = 1,
    .fields = (const VMStateField[]) {
        VMSTATE_STRUCT_ARRAY(instances, RockchipIOMMUState,
                             RK_IOMMU_NUM_INSTANCES, 1,
                             vmstate_rk_iommu_instance, RkIOMMUInstance),
        VMSTATE_END_OF_LIST()
    }
};

static void rk_iommu_class_init(ObjectClass *klass, const void *data)
{
    DeviceClass *dc = DEVICE_CLASS(klass);
    dc->realize = rk_iommu_realize;
    device_class_set_legacy_reset(dc, rk_iommu_reset);
    dc->vmsd = &vmstate_rk_iommu;
    dc->desc = "Rockchip IOMMU v2 (page table walk for vendor rknpu driver)";
}

static const TypeInfo rk_iommu_info = {
    .name = TYPE_ROCKCHIP_IOMMU,
    .parent = TYPE_SYS_BUS_DEVICE,
    .instance_size = sizeof(RockchipIOMMUState),
    .class_init = rk_iommu_class_init,
};

static void rk_iommu_register(void)
{
    type_register_static(&rk_iommu_info);
}

type_init(rk_iommu_register)
