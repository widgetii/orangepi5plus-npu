/*
 * Rockchip RK3588 machine type for QEMU
 *
 * Minimal RK3588 SoC emulation: 4x Cortex-A55, GICv3, UART2, stub CRU/PMU,
 * 3x Rocket NPU cores. Boots mainline Linux 6.18+ kernels for NPU CI testing.
 *
 * SPDX-License-Identifier: GPL-2.0-or-later
 */

#include "qemu/osdep.h"
#include "qemu/units.h"
#include "qemu/error-report.h"
#include "qapi/error.h"
#include "qobject/qlist.h"
#include "system/address-spaces.h"
#include "system/system.h"
#include "hw/arm/boot.h"
#include "hw/arm/rk3588.h"
#include "hw/arm/machines-qom.h"
#include "hw/boards.h"
#include "hw/char/serial-mm.h"
#include "hw/intc/arm_gicv3.h"
#include "hw/loader.h"
#include "hw/misc/unimp.h"
#include "hw/qdev-properties.h"
#include "hw/sysbus.h"
#include "system/device_tree.h"
#include "target/arm/cpu.h"
#include "target/arm/gtimer.h"

/* NPU device model */
#include "hw/misc/rockchip-npu.h"
#include "hw/misc/rockchip-iommu.h"

#define GIC_NUM_SPI 192

/*
 * RK3588 CRU stub — returns PLL lock status to prevent vendor kernel hangs.
 *
 * The vendor kernel's rockchip-clk driver reads PLL status registers and
 * spins until the lock bit (bit 6) is set. The unimplemented_device stub
 * returns 0, causing an infinite loop. This stub returns lock=1 for all
 * PLL status reads and stores/returns other register writes.
 *
 * RK3588 CRU PLL layout: each PLL has 8 regs (0x20 bytes).
 *   PLL0 at 0x000, PLL1 at 0x020, ..., PLL_STATUS at offset 0x18 within PLL
 *   Lock bit = bit 6 of PLL_STATUS.
 *
 * Additionally covers: PHPGRF, PMUGRF, SYS_GRF, IOC, etc. via additional stubs.
 */
#define CRU_STUB_SIZE     0x10000
#define CRU_STUB_REGS     (CRU_STUB_SIZE / 4)

typedef struct RK3588CRUStub {
    MemoryRegion iomem;
    uint32_t regs[CRU_STUB_REGS];
} RK3588CRUStub;

static uint64_t rk3588_cru_read(void *opaque, hwaddr addr, unsigned size)
{
    RK3588CRUStub *s = opaque;
    uint32_t idx = addr / 4;

    if (idx < CRU_STUB_REGS) {
        /*
         * PLL status registers: offset 0x18 within each 0x20 PLL block.
         * Return lock bit (bit 6) set for all PLLs.
         * PLLs at: 0x000-0x1E0 (up to 16 PLLs at 0x20 spacing).
         */
        if (addr < 0x200 && (addr & 0x1F) == 0x18) {
            return s->regs[idx] | (1 << 6);  /* PLL lock = 1 */
        }
        return s->regs[idx];
    }
    return 0;
}

static void rk3588_cru_write(void *opaque, hwaddr addr, uint64_t val,
                              unsigned size)
{
    RK3588CRUStub *s = opaque;
    uint32_t idx = addr / 4;
    if (idx < CRU_STUB_REGS) {
        s->regs[idx] = (uint32_t)val;
    }
}

static const MemoryRegionOps rk3588_cru_ops = {
    .read = rk3588_cru_read,
    .write = rk3588_cru_write,
    .endianness = DEVICE_LITTLE_ENDIAN,
    .impl = { .min_access_size = 4, .max_access_size = 4 },
};

static struct arm_boot_info rk3588_binfo;
static int rk3588_npu_num_cores = 1;  /* set from NPU device property */
static int rk3588_npu_driver_mode = 0; /* 0=rocket, 1=rknpu */

/*
 * Generate a minimal device tree for the RK3588 machine.
 */
static void *rk3588_create_dtb(MachineState *ms, int *fdt_size)
{
    void *fdt;
    char node[128];
    int gic_phandle;
    uint64_t npu_bases[] = {
        RK3588_NPU_CORE0_BASE,
        RK3588_NPU_CORE1_BASE,
        RK3588_NPU_CORE2_BASE,
    };
    uint32_t npu_irqs[] = {
        RK3588_NPU_CORE0_IRQ,
        RK3588_NPU_CORE1_IRQ,
        RK3588_NPU_CORE2_IRQ,
    };

    fdt = create_device_tree(fdt_size);
    if (!fdt) {
        error_report("create_device_tree() failed");
        exit(1);
    }

    qemu_fdt_setprop_string(fdt, "/", "compatible",
                            "rockchip,rk3588-orangepi-5-plus");
    qemu_fdt_setprop_string(fdt, "/", "model",
                            "QEMU RK3588 (Orange Pi 5 Plus)");
    qemu_fdt_setprop_cell(fdt, "/", "#address-cells", 2);
    qemu_fdt_setprop_cell(fdt, "/", "#size-cells", 2);

    /* /chosen */
    qemu_fdt_add_subnode(fdt, "/chosen");
    qemu_fdt_setprop_string(fdt, "/chosen", "stdout-path",
                            "/serial@feb50000");

    /* /memory */
    qemu_fdt_add_subnode(fdt, "/memory");
    qemu_fdt_setprop_string(fdt, "/memory", "device_type", "memory");
    uint64_t ram_size = ms->ram_size;
    uint64_t low_size = MIN(ram_size, 0x100000000ULL - RK3588_RAM_BASE);
    if (ram_size > low_size) {
        uint64_t high_size = ram_size - low_size;
        uint64_t mem_reg[4] = {
            cpu_to_be64(RK3588_RAM_BASE), cpu_to_be64(low_size),
            cpu_to_be64(RK3588_RAM_HIGH_BASE), cpu_to_be64(high_size),
        };
        qemu_fdt_setprop(fdt, "/memory", "reg", mem_reg, sizeof(mem_reg));
    } else {
        uint64_t mem_reg[2] = {
            cpu_to_be64(RK3588_RAM_BASE), cpu_to_be64(low_size),
        };
        qemu_fdt_setprop(fdt, "/memory", "reg", mem_reg, sizeof(mem_reg));
    }

    /* /cpus */
    qemu_fdt_add_subnode(fdt, "/cpus");
    qemu_fdt_setprop_cell(fdt, "/cpus", "#address-cells", 1);
    qemu_fdt_setprop_cell(fdt, "/cpus", "#size-cells", 0);

    for (int i = 0; i < RK3588_NUM_CPUS; i++) {
        snprintf(node, sizeof(node), "/cpus/cpu@%d", i);
        qemu_fdt_add_subnode(fdt, node);
        qemu_fdt_setprop_string(fdt, node, "compatible", "arm,cortex-a55");
        qemu_fdt_setprop_string(fdt, node, "device_type", "cpu");
        qemu_fdt_setprop_cell(fdt, node, "reg", i);
        qemu_fdt_setprop_string(fdt, node, "enable-method", "psci");
    }

    /* PSCI */
    qemu_fdt_add_subnode(fdt, "/psci");
    qemu_fdt_setprop_string(fdt, "/psci", "compatible", "arm,psci-1.0");
    qemu_fdt_setprop_string(fdt, "/psci", "method",
                            (rk3588_npu_driver_mode == NPU_DRIVER_MODE_RKNPU)
                            ? "smc" : "hvc");

    /* Timer */
    qemu_fdt_add_subnode(fdt, "/timer");
    qemu_fdt_setprop_string(fdt, "/timer", "compatible", "arm,armv8-timer");
    uint32_t timer_irq[12] = {
        cpu_to_be32(1), cpu_to_be32(13), cpu_to_be32(0xf04),
        cpu_to_be32(1), cpu_to_be32(14), cpu_to_be32(0xf04),
        cpu_to_be32(1), cpu_to_be32(11), cpu_to_be32(0xf04),
        cpu_to_be32(1), cpu_to_be32(10), cpu_to_be32(0xf04),
    };
    qemu_fdt_setprop(fdt, "/timer", "interrupts",
                     timer_irq, sizeof(timer_irq));

    /* GICv3 */
    gic_phandle = qemu_fdt_alloc_phandle(fdt);
    qemu_fdt_add_subnode(fdt, "/intc");
    qemu_fdt_setprop_string(fdt, "/intc", "compatible", "arm,gic-v3");
    qemu_fdt_setprop_cell(fdt, "/intc", "#interrupt-cells", 3);
    qemu_fdt_setprop(fdt, "/intc", "interrupt-controller", NULL, 0);
    qemu_fdt_setprop_cell(fdt, "/intc", "phandle", gic_phandle);
    qemu_fdt_setprop_cell(fdt, "/intc", "#address-cells", 2);
    qemu_fdt_setprop_cell(fdt, "/intc", "#size-cells", 2);
    qemu_fdt_setprop(fdt, "/intc", "ranges", NULL, 0);
    uint64_t gic_reg[4] = {
        cpu_to_be64(RK3588_GIC_DIST_BASE), cpu_to_be64(0x10000),
        cpu_to_be64(RK3588_GIC_REDIST_BASE), cpu_to_be64(0x80000),
    };
    qemu_fdt_setprop(fdt, "/intc", "reg", gic_reg, sizeof(gic_reg));
    qemu_fdt_setprop_cell(fdt, "/", "interrupt-parent", gic_phandle);

    /* UART2 */
    snprintf(node, sizeof(node), "/serial@%" PRIx64,
             (uint64_t)RK3588_UART2_BASE);
    qemu_fdt_add_subnode(fdt, node);
    qemu_fdt_setprop_string(fdt, node, "compatible", "ns16550a");
    uint64_t uart_reg[2] = {
        cpu_to_be64(RK3588_UART2_BASE), cpu_to_be64(RK3588_UART2_SIZE),
    };
    qemu_fdt_setprop(fdt, node, "reg", uart_reg, sizeof(uart_reg));
    uint32_t uart_irq[3] = {
        cpu_to_be32(0), cpu_to_be32(RK3588_UART2_IRQ), cpu_to_be32(4),
    };
    qemu_fdt_setprop(fdt, node, "interrupts", uart_irq, sizeof(uart_irq));
    qemu_fdt_setprop_cell(fdt, node, "clock-frequency",
                          (rk3588_npu_driver_mode == NPU_DRIVER_MODE_RKNPU)
                          ? 24000000 : 1843200);
    qemu_fdt_setprop_cell(fdt, node, "reg-shift", 2);
    qemu_fdt_setprop_cell(fdt, node, "reg-io-width", 4);

    /* Fixed clock for NPU (1 GHz dummy) */
    int clk_phandle = qemu_fdt_alloc_phandle(fdt);
    qemu_fdt_add_subnode(fdt, "/npu-clk");
    qemu_fdt_setprop_string(fdt, "/npu-clk", "compatible", "fixed-clock");
    qemu_fdt_setprop_cell(fdt, "/npu-clk", "#clock-cells", 0);
    qemu_fdt_setprop_cell(fdt, "/npu-clk", "clock-frequency", 1000000000);
    qemu_fdt_setprop_cell(fdt, "/npu-clk", "phandle", clk_phandle);

    /* Dummy reset controller (qemu_reset.ko module) */
    int rst_phandle = qemu_fdt_alloc_phandle(fdt);
    qemu_fdt_add_subnode(fdt, "/npu-reset");
    qemu_fdt_setprop_string(fdt, "/npu-reset", "compatible",
                            "qemu,reset-dummy");
    qemu_fdt_setprop_cell(fdt, "/npu-reset", "#reset-cells", 1);
    qemu_fdt_setprop_cell(fdt, "/npu-reset", "phandle", rst_phandle);

    if (rk3588_npu_driver_mode == NPU_DRIVER_MODE_RKNPU) {
        /* ============================================================
         * RKNPU mode: single NPU node + Rockchip IOMMU node
         * ============================================================ */

        /* Rockchip IOMMU node */
        int rk_iommu_phandle = qemu_fdt_alloc_phandle(fdt);
        qemu_fdt_add_subnode(fdt, "/iommu@fdab9000");
        qemu_fdt_setprop_string(fdt, "/iommu@fdab9000", "compatible",
                                "rockchip,iommu-v2");
        uint64_t iommu_reg[8] = {
            cpu_to_be64(RK3588_NPU_IOMMU0_BASE), cpu_to_be64(0x100),
            cpu_to_be64(RK3588_NPU_IOMMU1_BASE), cpu_to_be64(0x100),
            cpu_to_be64(RK3588_NPU_IOMMU2_BASE), cpu_to_be64(0x100),
            cpu_to_be64(RK3588_NPU_IOMMU3_BASE), cpu_to_be64(0x100),
        };
        qemu_fdt_setprop(fdt, "/iommu@fdab9000", "reg",
                         iommu_reg, sizeof(iommu_reg));
        uint32_t iommu_irqs[9] = {
            cpu_to_be32(0), cpu_to_be32(RK3588_NPU_CORE0_IRQ),
            cpu_to_be32(4),
            cpu_to_be32(0), cpu_to_be32(RK3588_NPU_CORE1_IRQ),
            cpu_to_be32(4),
            cpu_to_be32(0), cpu_to_be32(RK3588_NPU_CORE2_IRQ),
            cpu_to_be32(4),
        };
        qemu_fdt_setprop(fdt, "/iommu@fdab9000", "interrupts",
                         iommu_irqs, sizeof(iommu_irqs));
        qemu_fdt_setprop_cell(fdt, "/iommu@fdab9000", "#iommu-cells", 0);
        /* 6 clocks for IOMMU */
        uint32_t iommu_clks[6];
        for (int i = 0; i < 6; i++)
            iommu_clks[i] = cpu_to_be32(clk_phandle);
        qemu_fdt_setprop(fdt, "/iommu@fdab9000", "clocks",
                         iommu_clks, sizeof(iommu_clks));
        {
            char *clk_names[] = {
                (char *)"aclk0", (char *)"aclk1", (char *)"aclk2",
                (char *)"hclk0", (char *)"hclk1", (char *)"hclk2",
            };
            qemu_fdt_setprop_string_array(fdt, "/iommu@fdab9000",
                                          "clock-names", clk_names, 6);
        }
        qemu_fdt_setprop_string(fdt, "/iommu@fdab9000", "status", "okay");
        qemu_fdt_setprop_cell(fdt, "/iommu@fdab9000", "phandle",
                              rk_iommu_phandle);

        /* Single NPU node with 3 reg regions */
        qemu_fdt_add_subnode(fdt, "/npu@fdab0000");
        qemu_fdt_setprop_string(fdt, "/npu@fdab0000", "compatible",
                                "rockchip,rk3588-rknpu");
        uint64_t npu_reg[6] = {
            cpu_to_be64(npu_bases[0]), cpu_to_be64(0x10000),
            cpu_to_be64(npu_bases[1]), cpu_to_be64(0x10000),
            cpu_to_be64(npu_bases[2]), cpu_to_be64(0x10000),
        };
        qemu_fdt_setprop(fdt, "/npu@fdab0000", "reg",
                         npu_reg, sizeof(npu_reg));
        uint32_t npu_irqs_prop[9] = {
            cpu_to_be32(0), cpu_to_be32(RK3588_NPU_CORE0_IRQ),
            cpu_to_be32(4),
            cpu_to_be32(0), cpu_to_be32(RK3588_NPU_CORE1_IRQ),
            cpu_to_be32(4),
            cpu_to_be32(0), cpu_to_be32(RK3588_NPU_CORE2_IRQ),
            cpu_to_be32(4),
        };
        qemu_fdt_setprop(fdt, "/npu@fdab0000", "interrupts",
                         npu_irqs_prop, sizeof(npu_irqs_prop));
        {
            char *irq_names[] = {
                (char *)"npu0_irq", (char *)"npu1_irq", (char *)"npu2_irq",
            };
            qemu_fdt_setprop_string_array(fdt, "/npu@fdab0000",
                                          "interrupt-names", irq_names, 3);
        }
        /* 8 clocks */
        uint32_t npu_clks[8];
        for (int i = 0; i < 8; i++)
            npu_clks[i] = cpu_to_be32(clk_phandle);
        qemu_fdt_setprop(fdt, "/npu@fdab0000", "clocks",
                         npu_clks, sizeof(npu_clks));
        {
            char *clk_names[] = {
                (char *)"clk_npu", (char *)"aclk0", (char *)"aclk1",
                (char *)"aclk2", (char *)"hclk0", (char *)"hclk1",
                (char *)"hclk2", (char *)"pclk",
            };
            qemu_fdt_setprop_string_array(fdt, "/npu@fdab0000",
                                          "clock-names", clk_names, 8);
        }
        /* 6 resets */
        uint32_t npu_rsts[12] = {
            cpu_to_be32(rst_phandle), cpu_to_be32(0),
            cpu_to_be32(rst_phandle), cpu_to_be32(1),
            cpu_to_be32(rst_phandle), cpu_to_be32(2),
            cpu_to_be32(rst_phandle), cpu_to_be32(3),
            cpu_to_be32(rst_phandle), cpu_to_be32(4),
            cpu_to_be32(rst_phandle), cpu_to_be32(5),
        };
        qemu_fdt_setprop(fdt, "/npu@fdab0000", "resets",
                         npu_rsts, sizeof(npu_rsts));
        {
            char *rst_names[] = {
                (char *)"srst_a0", (char *)"srst_a1", (char *)"srst_a2",
                (char *)"srst_h0", (char *)"srst_h1", (char *)"srst_h2",
            };
            qemu_fdt_setprop_string_array(fdt, "/npu@fdab0000",
                                          "reset-names", rst_names, 6);
        }
        /* IOMMU reference */
        uint32_t iommu_ref = cpu_to_be32(rk_iommu_phandle);
        qemu_fdt_setprop(fdt, "/npu@fdab0000", "iommus",
                         &iommu_ref, sizeof(iommu_ref));

    } else {
        /* ============================================================
         * Rocket mode: per-core NPU nodes + QEMU IOMMU mailbox
         * ============================================================ */

        /* Dummy IOMMU (qemu_iommu.ko module) with mailbox for IOVA→GPA */
        int iommu_phandle = qemu_fdt_alloc_phandle(fdt);
        snprintf(node, sizeof(node), "/qemu-iommu@%" PRIx64,
                 (uint64_t)NPU_IOMMU_BASE);
        qemu_fdt_add_subnode(fdt, node);
        qemu_fdt_setprop_string(fdt, node, "compatible",
                                "qemu,iommu-dummy");
        uint64_t iommu_reg[2] = {
            cpu_to_be64(NPU_IOMMU_BASE), cpu_to_be64(0x1000),
        };
        qemu_fdt_setprop(fdt, node, "reg", iommu_reg, sizeof(iommu_reg));
        qemu_fdt_setprop_cell(fdt, node, "#iommu-cells", 0);
        qemu_fdt_setprop_cell(fdt, node, "phandle", iommu_phandle);

        /* NPU cores */
        for (int i = 0; i < rk3588_npu_num_cores; i++) {
            snprintf(node, sizeof(node), "/npu@%" PRIx64,
                     (uint64_t)npu_bases[i]);
            qemu_fdt_add_subnode(fdt, node);
            qemu_fdt_setprop_string(fdt, node, "compatible",
                                    "rockchip,rk3588-rknn-core");
            uint64_t npu_reg[6] = {
                cpu_to_be64(npu_bases[i] + 0x0000), cpu_to_be64(0x1000),
                cpu_to_be64(npu_bases[i] + 0x1000), cpu_to_be64(0x1000),
                cpu_to_be64(npu_bases[i] + 0x3000), cpu_to_be64(0x1000),
            };
            qemu_fdt_setprop(fdt, node, "reg", npu_reg, sizeof(npu_reg));
            {
                char *reg_names[] = {
                    (char *)"pc", (char *)"cna", (char *)"core",
                };
                qemu_fdt_setprop_string_array(fdt, node, "reg-names",
                                              reg_names, 3);
            }
            uint32_t npu_irq_prop[3] = {
                cpu_to_be32(0), cpu_to_be32(npu_irqs[i]), cpu_to_be32(4),
            };
            qemu_fdt_setprop(fdt, node, "interrupts",
                             npu_irq_prop, sizeof(npu_irq_prop));
            uint32_t clk_refs[4] = {
                cpu_to_be32(clk_phandle), cpu_to_be32(clk_phandle),
                cpu_to_be32(clk_phandle), cpu_to_be32(clk_phandle),
            };
            qemu_fdt_setprop(fdt, node, "clocks",
                             clk_refs, sizeof(clk_refs));
            {
                char *clk_names[] = {
                    (char *)"aclk", (char *)"hclk",
                    (char *)"npu", (char *)"pclk",
                };
                qemu_fdt_setprop_string_array(fdt, node, "clock-names",
                                              clk_names, 4);
            }
            uint32_t rst_refs[4] = {
                cpu_to_be32(rst_phandle), cpu_to_be32(i * 2),
                cpu_to_be32(rst_phandle), cpu_to_be32(i * 2 + 1),
            };
            qemu_fdt_setprop(fdt, node, "resets",
                             rst_refs, sizeof(rst_refs));
            {
                char *rst_names[] = {
                    (char *)"srst_a", (char *)"srst_h",
                };
                qemu_fdt_setprop_string_array(fdt, node, "reset-names",
                                              rst_names, 2);
            }
            uint32_t iommu_ref = cpu_to_be32(iommu_phandle);
            qemu_fdt_setprop(fdt, node, "iommus",
                             &iommu_ref, sizeof(iommu_ref));
        }
    }

    return fdt;
}

static MachineState *rk3588_ms; /* saved for get_dtb callback */

static void *rk3588_get_dtb(const struct arm_boot_info *info, int *size)
{
    return rk3588_create_dtb(rk3588_ms, size);
}

static void rk3588_init(MachineState *ms)
{
    MemoryRegion *sysmem = get_system_memory();
    DeviceState *gicdev;
    SysBusDevice *gicbusdev;
    Object *cpuobj[RK3588_NUM_CPUS];
    uint64_t ram_size = ms->ram_size;

    rk3588_ms = ms;
    uint64_t low_size = MIN(ram_size, 0x100000000ULL - RK3588_RAM_BASE);
    MemoryRegion *lowram = g_new(MemoryRegion, 1);

    /* RAM */
    memory_region_init_ram(lowram, NULL, "rk3588.lowram", low_size,
                           &error_fatal);
    memory_region_add_subregion(sysmem, RK3588_RAM_BASE, lowram);

    if (ram_size > low_size) {
        uint64_t high_size = ram_size - low_size;
        MemoryRegion *highram = g_new(MemoryRegion, 1);
        memory_region_init_ram(highram, NULL, "rk3588.highram", high_size,
                               &error_fatal);
        memory_region_add_subregion(sysmem, RK3588_RAM_HIGH_BASE, highram);
    }

    /* CPUs */
    for (int i = 0; i < ms->smp.cpus; i++) {
        cpuobj[i] = object_new(ms->cpu_type);
        if (object_property_find(cpuobj[i], "has_el3")) {
            /* In rknpu mode, enable EL3 so SMC from vendor drivers
             * (e.g., rockchip_drm SIP calls) get handled by QEMU's
             * built-in PSCI/SMC stub instead of causing UNDEF. */
            bool el3 = (rk3588_npu_driver_mode == NPU_DRIVER_MODE_RKNPU);
            object_property_set_bool(cpuobj[i], "has_el3", el3,
                                     &error_fatal);
        }
        if (object_property_find(cpuobj[i], "has_el2")) {
            object_property_set_bool(cpuobj[i], "has_el2", false,
                                     &error_fatal);
        }
        /* Secondary CPUs start powered off, woken by PSCI CPU_ON */
        if (i > 0) {
            object_property_set_bool(cpuobj[i], "start-powered-off", true,
                                     &error_fatal);
        }
        qdev_realize(DEVICE(cpuobj[i]), NULL, &error_fatal);
    }

    /* GICv3 */
    gicdev = qdev_new(gicv3_class_name());
    qdev_prop_set_uint32(gicdev, "num-cpu", ms->smp.cpus);
    qdev_prop_set_uint32(gicdev, "num-irq", GIC_NUM_SPI + 32);
    qdev_prop_set_uint32(gicdev, "revision", 3);
    qdev_prop_set_bit(gicdev, "has-security-extensions", false);
    {
        QList *redist_region_count = qlist_new();
        qlist_append_int(redist_region_count, ms->smp.cpus);
        qdev_prop_set_array(gicdev, "redist-region-count",
                            redist_region_count);
    }
    gicbusdev = SYS_BUS_DEVICE(gicdev);
    sysbus_realize_and_unref(gicbusdev, &error_fatal);
    sysbus_mmio_map(gicbusdev, 0, RK3588_GIC_DIST_BASE);
    sysbus_mmio_map(gicbusdev, 1, RK3588_GIC_REDIST_BASE);

    for (int i = 0; i < ms->smp.cpus; i++) {
        DeviceState *cpudev = DEVICE(cpuobj[i]);
        /* PPI inputs: start after SPIs, then 32 per CPU */
        int intidbase = GIC_NUM_SPI + i * 32;

        /* GIC outputs → CPU interrupt inputs */
        sysbus_connect_irq(gicbusdev, i,
                           qdev_get_gpio_in(cpudev, ARM_CPU_IRQ));
        sysbus_connect_irq(gicbusdev, i + RK3588_NUM_CPUS,
                           qdev_get_gpio_in(cpudev, ARM_CPU_FIQ));
        sysbus_connect_irq(gicbusdev, i + 2 * RK3588_NUM_CPUS,
                           qdev_get_gpio_in(cpudev, ARM_CPU_VIRQ));
        sysbus_connect_irq(gicbusdev, i + 3 * RK3588_NUM_CPUS,
                           qdev_get_gpio_in(cpudev, ARM_CPU_VFIQ));

        /* Timer PPIs → GIC (PPI IDs from ARM BSA spec) */
        qdev_connect_gpio_out(cpudev, GTIMER_PHYS,
                              qdev_get_gpio_in(gicdev, intidbase + 30));
        qdev_connect_gpio_out(cpudev, GTIMER_VIRT,
                              qdev_get_gpio_in(gicdev, intidbase + 27));
        qdev_connect_gpio_out(cpudev, GTIMER_HYP,
                              qdev_get_gpio_in(gicdev, intidbase + 26));
        qdev_connect_gpio_out(cpudev, GTIMER_SEC,
                              qdev_get_gpio_in(gicdev, intidbase + 29));
    }

    /* UART2 — use 24MHz clock for vendor kernel compatibility (1500000 baud) */
    {
        int uart_freq = (rk3588_npu_driver_mode == NPU_DRIVER_MODE_RKNPU)
                        ? 24000000 : 1843200;
        serial_mm_init(sysmem, RK3588_UART2_BASE, 2,
                       qdev_get_gpio_in(gicdev, RK3588_UART2_IRQ),
                       uart_freq, serial_hd(0), DEVICE_LITTLE_ENDIAN);
        /*
         * DW 8250 in the vendor kernel accesses registers beyond the
         * standard 8 (offsets 0x7c, 0x80, 0x88, etc). serial_mm only
         * handles the first 32 bytes. Cover the rest of the page to
         * prevent synchronous external aborts from unmapped MMIO.
         */
        create_unimplemented_device("rk3588.uart2-dw",
                                    RK3588_UART2_BASE + 0x20,
                                    RK3588_UART2_SIZE - 0x20);
    }

    /* CRU stub with PLL lock bits (prevents vendor kernel CRU driver hang) */
    {
        static RK3588CRUStub cru_stub;
        memset(&cru_stub, 0, sizeof(cru_stub));
        memory_region_init_io(&cru_stub.iomem, NULL, &rk3588_cru_ops,
                              &cru_stub, "rk3588.cru", RK3588_CRU_SIZE);
        memory_region_add_subregion(sysmem, RK3588_CRU_BASE,
                                    &cru_stub.iomem);
    }
    /* PMU CRU also has PLLs — use same approach */
    {
        static RK3588CRUStub pmu_cru_stub;
        memset(&pmu_cru_stub, 0, sizeof(pmu_cru_stub));
        memory_region_init_io(&pmu_cru_stub.iomem, NULL, &rk3588_cru_ops,
                              &pmu_cru_stub, "rk3588.pmu-cru",
                              RK3588_PMU_SIZE);
        memory_region_add_subregion(sysmem, RK3588_PMU_BASE,
                                    &pmu_cru_stub.iomem);
    }
    /* Additional GRF/IOC stubs for vendor kernel built-in drivers */
    create_unimplemented_device("rk3588.sys-grf",    0xfd58c000, 0x1000);
    create_unimplemented_device("rk3588.php-grf",    0xfd5b0000, 0x1000);
    create_unimplemented_device("rk3588.pmu1-grf",   0xfd5a0000, 0x1000);
    create_unimplemented_device("rk3588.ioc",        0xfd5f0000, 0x10000);
    create_unimplemented_device("rk3588.pmu0-grf",   0xfd588000, 0x1000);
    create_unimplemented_device("rk3588.bigcore0-grf", 0xfd590000, 0x1000);
    create_unimplemented_device("rk3588.bigcore1-grf", 0xfd592000, 0x1000);
    create_unimplemented_device("rk3588.litcore-grf",  0xfd594000, 0x1000);
    create_unimplemented_device("rk3588.pipe-phy-grf", 0xfd5b8000, 0x1000);
    create_unimplemented_device("rk3588.usb-grf",    0xfd5ac000, 0x4000);
    create_unimplemented_device("rk3588.pmu",        0xfdd90000, 0x1000);

    /* NPU */
    {
        uint64_t npu_bases[] = {
            RK3588_NPU_CORE0_BASE,
            RK3588_NPU_CORE1_BASE,
            RK3588_NPU_CORE2_BASE,
        };
        uint32_t npu_irqs[] = {
            RK3588_NPU_CORE0_IRQ,
            RK3588_NPU_CORE1_IRQ,
            RK3588_NPU_CORE2_IRQ,
        };

        DeviceState *npu = qdev_new(TYPE_ROCKCHIP_NPU);
        SysBusDevice *npubus = SYS_BUS_DEVICE(npu);
        RockchipNPUState *npu_s = ROCKCHIP_NPU(npu);

        rk3588_npu_driver_mode = npu_s->driver_mode;

        /* In rknpu mode, force 3 cores (single NPU node with 3 reg regions) */
        if (npu_s->driver_mode == NPU_DRIVER_MODE_RKNPU) {
            npu_s->num_cores = 3;
        }
        rk3588_npu_num_cores = npu_s->num_cores;

        if (npu_s->driver_mode == NPU_DRIVER_MODE_RKNPU) {
            /*
             * Create Rockchip IOMMU state.
             * IOMMU registers are embedded within NPU core MMIO regions
             * (offsets 0x9000/0xa000), so we don't map them separately.
             * The NPU read/write handlers forward accesses to the IOMMU.
             */
            DeviceState *iommu_dev = qdev_new(TYPE_ROCKCHIP_IOMMU);
            sysbus_realize_and_unref(SYS_BUS_DEVICE(iommu_dev),
                                     &error_fatal);
            npu_s->rk_iommu = ROCKCHIP_IOMMU(iommu_dev);
        }

        sysbus_realize_and_unref(npubus, &error_fatal);

        /* In rknpu mode, all 3 cores are always mapped (single NPU node) */
        int cores_to_map = (npu_s->driver_mode == NPU_DRIVER_MODE_RKNPU)
                           ? 3 : rk3588_npu_num_cores;
        for (int i = 0; i < cores_to_map; i++) {
            sysbus_mmio_map(npubus, i, npu_bases[i]);
            sysbus_connect_irq(npubus, i,
                               qdev_get_gpio_in(gicdev, npu_irqs[i]));
        }

        /* IOMMU mailbox — only in Rocket mode */
        if (npu_s->driver_mode == NPU_DRIVER_MODE_ROCKET) {
            sysbus_mmio_map(npubus, rk3588_npu_num_cores, NPU_IOMMU_BASE);
        }
    }

    /* Boot */
    rk3588_binfo.ram_size = ram_size;
    rk3588_binfo.loader_start = RK3588_RAM_BASE;
    rk3588_binfo.board_id = -1;
    rk3588_binfo.psci_conduit = (rk3588_npu_driver_mode == NPU_DRIVER_MODE_RKNPU)
                                ? QEMU_PSCI_CONDUIT_SMC
                                : QEMU_PSCI_CONDUIT_HVC;
    if (ms->dtb) {
        rk3588_binfo.dtb_filename = ms->dtb;
    } else {
        rk3588_binfo.get_dtb = rk3588_get_dtb;
    }
    arm_load_kernel(ARM_CPU(cpuobj[0]), ms, &rk3588_binfo);
}

static void rk3588_machine_class_init(MachineClass *mc)
{
    mc->desc = "Rockchip RK3588 (Orange Pi 5 Plus) with Rocket NPU";
    mc->init = rk3588_init;
    mc->default_cpus = RK3588_NUM_CPUS;
    mc->min_cpus = 1;
    mc->max_cpus = RK3588_NUM_CPUS;
    mc->default_cpu_type = ARM_CPU_TYPE_NAME("cortex-a55");
    mc->default_ram_size = 4 * GiB;
    mc->default_ram_id = "rk3588.ram";
    mc->no_cdrom = true;
}

DEFINE_MACHINE_ARM("orangepi5plus", rk3588_machine_class_init)
