/*
 * Rockchip RK3588 machine type for QEMU
 *
 * RK3588 SoC emulation: 4x Cortex-A55 (EL3), GICv3, DW UART2, CRU/PMU
 * stubs, Rockchip IOMMU v2, 3x NPU cores with INT8 convolution engine.
 *
 * Boots both mainline Linux (6.18+ with Rocket driver) and vendor kernel
 * (6.1.115-vendor-rk35xx with rknpu driver). Both drivers program the same
 * NPU hardware registers — the DTB uses the vendor rknpu binding which the
 * Rocket driver also matches via rockchip,rk3588-rknn-core fallback.
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

#include "hw/misc/rockchip-npu.h"
#include "hw/misc/rockchip-iommu.h"

#define GIC_NUM_SPI 256

/*
 * CRU stub — stores register values for the Rockchip CRU clock/reset
 * controller driver. Returns PLL lock bit (bit 15) for PLL status registers
 * so the kernel's rockchip-clk driver doesn't spin waiting for PLL lock.
 *
 * The mainline kernel's rk3588-cru driver uses the CRU to register both
 * clocks and reset controllers. By emulating enough of the CRU, we get
 * a built-in reset controller and eliminate the custom qemu_reset.ko module.
 */
#define CRU_STUB_REGS  (RK3588_CRU_SIZE / 4)

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
         * PLL lock status: each PLL has 8 registers (0x20 spacing).
         * The status register is at offset +0x18 within each PLL block.
         * Bit 15 = PLL lock. PLLs span the first 0x200 bytes of the CRU.
         */
        if (addr < 0x200 && (addr & 0x1F) == 0x18) {
            return s->regs[idx] | (1 << 15);
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

static void *rk3588_create_dtb(MachineState *ms, int *fdt_size)
{
    void *fdt;
    char node[128];
    uint64_t npu_bases[] = {
        RK3588_NPU_CORE0_BASE,
        RK3588_NPU_CORE1_BASE,
        RK3588_NPU_CORE2_BASE,
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
    uint64_t ram_size = MIN(ms->ram_size,
                            RK3588_RAM_LOW_TOP - RK3588_RAM_BASE);
    uint64_t mem_reg[2] = {
        cpu_to_be64(RK3588_RAM_BASE), cpu_to_be64(ram_size),
    };
    qemu_fdt_setprop(fdt, "/memory", "reg", mem_reg, sizeof(mem_reg));

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

    /* PSCI — SMC conduit (EL3 enabled for vendor kernel SIP calls) */
    qemu_fdt_add_subnode(fdt, "/psci");
    qemu_fdt_setprop_string(fdt, "/psci", "compatible", "arm,psci-1.0");
    qemu_fdt_setprop_string(fdt, "/psci", "method", "smc");

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
    int gic_phandle = qemu_fdt_alloc_phandle(fdt);
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

    /* UART2 (24 MHz clock for 1500000 baud) */
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
    qemu_fdt_setprop_cell(fdt, node, "clock-frequency", 24000000);
    qemu_fdt_setprop_cell(fdt, node, "reg-shift", 2);
    qemu_fdt_setprop_cell(fdt, node, "reg-io-width", 4);

    /* Virtio-MMIO transports
     *
     * These provide virtio-blk (root disk) and virtio-net (networking).
     * The real RK3588 GMAC (Synopsys DWMAC 4.20a) cannot be emulated via
     * QEMU's NPCM GMAC model — NPCM implements DWMAC3 registers while
     * stmmac expects DWMAC4 DMA at different offsets. Virtio-net is the
     * practical solution for QEMU networking.
     */
    for (int i = 0; i < RK3588_NUM_VIRTIO; i++) {
        hwaddr base = RK3588_VIRTIO_BASE + i * RK3588_VIRTIO_SIZE;
        snprintf(node, sizeof(node), "/virtio_mmio@%" PRIx64, (uint64_t)base);
        qemu_fdt_add_subnode(fdt, node);
        qemu_fdt_setprop_string(fdt, node, "compatible", "virtio,mmio");
        uint64_t vio_reg[2] = {
            cpu_to_be64(base), cpu_to_be64(RK3588_VIRTIO_SIZE),
        };
        qemu_fdt_setprop(fdt, node, "reg", vio_reg, sizeof(vio_reg));
        uint32_t vio_irq[3] = {
            cpu_to_be32(0), cpu_to_be32(RK3588_VIRTIO_IRQ + i),
            cpu_to_be32(1),
        };
        qemu_fdt_setprop(fdt, node, "interrupts", vio_irq, sizeof(vio_irq));
    }

    /* Fixed clock for NPU (1 GHz dummy) */
    int clk_phandle = qemu_fdt_alloc_phandle(fdt);
    qemu_fdt_add_subnode(fdt, "/npu-clk");
    qemu_fdt_setprop_string(fdt, "/npu-clk", "compatible", "fixed-clock");
    qemu_fdt_setprop_cell(fdt, "/npu-clk", "#clock-cells", 0);
    qemu_fdt_setprop_cell(fdt, "/npu-clk", "clock-frequency", 1000000000);
    qemu_fdt_setprop_cell(fdt, "/npu-clk", "phandle", clk_phandle);

    /*
     * System GRF (General Register File) as syscon.
     * The CRU driver looks up "rockchip,grf" for PLL lock status.
     * Our unimplemented_device MMIO stub handles reads (returns 0).
     */
    int grf_phandle = qemu_fdt_alloc_phandle(fdt);
    qemu_fdt_add_subnode(fdt, "/syscon@fd58c000");
    {
        char *grf_compat[] = {
            (char *)"rockchip,rk3588-sys-grf",
            (char *)"syscon",
        };
        qemu_fdt_setprop_string_array(fdt, "/syscon@fd58c000", "compatible",
                                      grf_compat, 2);
    }
    uint64_t grf_reg[2] = {
        cpu_to_be64(0xfd58c000ULL), cpu_to_be64(0x1000),
    };
    qemu_fdt_setprop(fdt, "/syscon@fd58c000", "reg",
                     grf_reg, sizeof(grf_reg));
    qemu_fdt_setprop_cell(fdt, "/syscon@fd58c000", "phandle", grf_phandle);

    /*
     * CRU (Clock and Reset Unit) — the kernel's built-in rockchip-clk
     * driver probes this node and registers both clocks and the reset
     * controller. This eliminates the need for qemu_reset.ko.
     *
     * With all PLL registers at zero + lock bit set, PLLs report as
     * "slow mode" (24 MHz) — no division-by-zero, just low clock rates.
     * The soft reset registers at CRU+0xA00 are backed by our read/write
     * CRU stub, so assert/deassert works correctly.
     */
    int cru_phandle = qemu_fdt_alloc_phandle(fdt);
    qemu_fdt_add_subnode(fdt, "/clock-controller@fd7c0000");
    qemu_fdt_setprop_string(fdt, "/clock-controller@fd7c0000",
                            "compatible", "rockchip,rk3588-cru");
    uint64_t cru_reg[2] = {
        cpu_to_be64(RK3588_CRU_BASE), cpu_to_be64(RK3588_CRU_SIZE),
    };
    qemu_fdt_setprop(fdt, "/clock-controller@fd7c0000", "reg",
                     cru_reg, sizeof(cru_reg));
    qemu_fdt_setprop_cell(fdt, "/clock-controller@fd7c0000",
                          "#clock-cells", 1);
    qemu_fdt_setprop_cell(fdt, "/clock-controller@fd7c0000",
                          "#reset-cells", 1);
    qemu_fdt_setprop_cell(fdt, "/clock-controller@fd7c0000",
                          "phandle", cru_phandle);
    {
        uint32_t grf_ref = cpu_to_be32(grf_phandle);
        qemu_fdt_setprop(fdt, "/clock-controller@fd7c0000", "rockchip,grf",
                         &grf_ref, sizeof(grf_ref));
    }

    /* Rockchip IOMMU */
    int iommu_phandle = qemu_fdt_alloc_phandle(fdt);
    qemu_fdt_add_subnode(fdt, "/iommu@fdab9000");
    {
        char *iommu_compat[] = {
            (char *)"rockchip,iommu-v2",
            (char *)"rockchip,iommu",
        };
        qemu_fdt_setprop_string_array(fdt, "/iommu@fdab9000", "compatible",
                                      iommu_compat, 2);
    }
    uint64_t iommu_reg[8] = {
        cpu_to_be64(RK3588_NPU_IOMMU0_BASE), cpu_to_be64(0x100),
        cpu_to_be64(RK3588_NPU_IOMMU1_BASE), cpu_to_be64(0x100),
        cpu_to_be64(RK3588_NPU_IOMMU2_BASE), cpu_to_be64(0x100),
        cpu_to_be64(RK3588_NPU_IOMMU3_BASE), cpu_to_be64(0x100),
    };
    qemu_fdt_setprop(fdt, "/iommu@fdab9000", "reg",
                     iommu_reg, sizeof(iommu_reg));
    uint32_t iommu_irqs[9] = {
        cpu_to_be32(0), cpu_to_be32(RK3588_NPU_CORE0_IRQ), cpu_to_be32(4),
        cpu_to_be32(0), cpu_to_be32(RK3588_NPU_CORE1_IRQ), cpu_to_be32(4),
        cpu_to_be32(0), cpu_to_be32(RK3588_NPU_CORE2_IRQ), cpu_to_be32(4),
    };
    qemu_fdt_setprop(fdt, "/iommu@fdab9000", "interrupts",
                     iommu_irqs, sizeof(iommu_irqs));
    qemu_fdt_setprop_cell(fdt, "/iommu@fdab9000", "#iommu-cells", 0);
    /* The mainline rockchip-iommu driver uses devm_clk_get(dev, "aclk")
     * and devm_clk_get(dev, "iface"). Provide two named clocks. */
    uint32_t iommu_clks[2] = {
        cpu_to_be32(clk_phandle), cpu_to_be32(clk_phandle),
    };
    qemu_fdt_setprop(fdt, "/iommu@fdab9000", "clocks",
                     iommu_clks, sizeof(iommu_clks));
    {
        char *clk_names[] = {
            (char *)"aclk", (char *)"iface",
        };
        qemu_fdt_setprop_string_array(fdt, "/iommu@fdab9000",
                                      "clock-names", clk_names, 2);
    }
    qemu_fdt_setprop_string(fdt, "/iommu@fdab9000", "status", "okay");
    qemu_fdt_setprop_cell(fdt, "/iommu@fdab9000", "phandle", iommu_phandle);

    /* NPU — single node for vendor rknpu driver */
    qemu_fdt_add_subnode(fdt, "/npu@fdab0000");
    qemu_fdt_setprop_string(fdt, "/npu@fdab0000", "compatible",
                            "rockchip,rk3588-rknpu");
    uint64_t npu_reg[6] = {
        cpu_to_be64(npu_bases[0]), cpu_to_be64(0x10000),
        cpu_to_be64(npu_bases[1]), cpu_to_be64(0x10000),
        cpu_to_be64(npu_bases[2]), cpu_to_be64(0x10000),
    };
    qemu_fdt_setprop(fdt, "/npu@fdab0000", "reg", npu_reg, sizeof(npu_reg));
    uint32_t npu_irqs[9] = {
        cpu_to_be32(0), cpu_to_be32(RK3588_NPU_CORE0_IRQ), cpu_to_be32(4),
        cpu_to_be32(0), cpu_to_be32(RK3588_NPU_CORE1_IRQ), cpu_to_be32(4),
        cpu_to_be32(0), cpu_to_be32(RK3588_NPU_CORE2_IRQ), cpu_to_be32(4),
    };
    qemu_fdt_setprop(fdt, "/npu@fdab0000", "interrupts",
                     npu_irqs, sizeof(npu_irqs));
    {
        char *irq_names[] = {
            (char *)"npu0_irq", (char *)"npu1_irq", (char *)"npu2_irq",
        };
        qemu_fdt_setprop_string_array(fdt, "/npu@fdab0000",
                                      "interrupt-names", irq_names, 3);
    }
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
    /* Reset IDs from dt-bindings/reset/rockchip,rk3588-cru.h */
    uint32_t npu_rsts[12] = {
        cpu_to_be32(cru_phandle), cpu_to_be32(272),  /* SRST_A_RKNN0 */
        cpu_to_be32(cru_phandle), cpu_to_be32(250),  /* SRST_A_RKNN1 */
        cpu_to_be32(cru_phandle), cpu_to_be32(254),  /* SRST_A_RKNN2 */
        cpu_to_be32(cru_phandle), cpu_to_be32(274),  /* SRST_H_RKNN0 */
        cpu_to_be32(cru_phandle), cpu_to_be32(252),  /* SRST_H_RKNN1 */
        cpu_to_be32(cru_phandle), cpu_to_be32(256),  /* SRST_H_RKNN2 */
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
    uint32_t iommu_ref = cpu_to_be32(iommu_phandle);
    qemu_fdt_setprop(fdt, "/npu@fdab0000", "iommus",
                     &iommu_ref, sizeof(iommu_ref));

    /*
     * Per-core NPU nodes for the upstream Rocket driver.
     *
     * These use the same MMIO addresses as the rknpu node above but with
     * smaller sub-regions (pc/cna/core). The rknpu node is disabled here
     * (status=disabled) so that only one set of nodes is active at a time:
     * - Default DTB: rknpu node disabled, per-core nodes active → Rocket
     * - Vendor kernel: pass -dtb with only the rknpu node (or rknpu driver
     *   matches first and ignores the per-core nodes)
     *
     * On the real RK3588, the upstream DTS has only per-core nodes and the
     * vendor DTS has only the single rknpu node — they never coexist.
     * We include both but disable the rknpu node by default since the
     * Rocket driver needs reg-names and per-core IOMMU groups.
     */
    /* Both rknpu and per-core nodes are enabled. The vendor kernel matches
     * the rknpu node; the Rocket driver matches the per-core nodes. Region
     * overlaps cause non-fatal "can't request region" warnings. */

    for (int i = 0; i < RK3588_NPU_NUM_CORES; i++) {
        snprintf(node, sizeof(node), "/npu-core@%" PRIx64,
                 (uint64_t)npu_bases[i]);
        qemu_fdt_add_subnode(fdt, node);
        qemu_fdt_setprop_string(fdt, node, "compatible",
                                "rockchip,rk3588-rknn-core");
        /* Only enable the last core (fdad0000) — the Rocket driver probes
         * in reverse order so this becomes core 0. Disabling others avoids
         * DRM scheduler timeouts from QEMU's synchronous execution model. */
        if (i < RK3588_NPU_NUM_CORES - 1) {
            qemu_fdt_setprop_string(fdt, node, "status", "disabled");
        }
        uint64_t core_reg[6] = {
            cpu_to_be64(npu_bases[i] + 0x0000), cpu_to_be64(0x1000),
            cpu_to_be64(npu_bases[i] + 0x1000), cpu_to_be64(0x1000),
            cpu_to_be64(npu_bases[i] + 0x3000), cpu_to_be64(0x1000),
        };
        qemu_fdt_setprop(fdt, node, "reg", core_reg, sizeof(core_reg));
        {
            char *reg_names[] = {
                (char *)"pc", (char *)"cna", (char *)"core",
            };
            qemu_fdt_setprop_string_array(fdt, node, "reg-names",
                                          reg_names, 3);
        }
        uint32_t core_irq[3] = {
            cpu_to_be32(0),
            cpu_to_be32(RK3588_NPU_CORE0_IRQ + i),
            cpu_to_be32(4),
        };
        qemu_fdt_setprop(fdt, node, "interrupts",
                         core_irq, sizeof(core_irq));
        uint32_t core_clks[4] = {
            cpu_to_be32(clk_phandle), cpu_to_be32(clk_phandle),
            cpu_to_be32(clk_phandle), cpu_to_be32(clk_phandle),
        };
        qemu_fdt_setprop(fdt, node, "clocks",
                         core_clks, sizeof(core_clks));
        {
            char *core_clk_names[] = {
                (char *)"aclk", (char *)"hclk",
                (char *)"npu", (char *)"pclk",
            };
            qemu_fdt_setprop_string_array(fdt, node, "clock-names",
                                          core_clk_names, 4);
        }
        /* SRST_A_RKNN{0,1,2} and SRST_H_RKNN{0,1,2} from rk3588-cru.h */
        static const uint32_t srst_a_rknn[] = { 272, 250, 254 };
        static const uint32_t srst_h_rknn[] = { 274, 252, 256 };
        uint32_t core_rsts[4] = {
            cpu_to_be32(cru_phandle), cpu_to_be32(srst_a_rknn[i]),
            cpu_to_be32(cru_phandle), cpu_to_be32(srst_h_rknn[i]),
        };
        qemu_fdt_setprop(fdt, node, "resets",
                         core_rsts, sizeof(core_rsts));
        {
            char *core_rst_names[] = {
                (char *)"srst_a", (char *)"srst_h",
            };
            qemu_fdt_setprop_string_array(fdt, node, "reset-names",
                                          core_rst_names, 2);
        }
        qemu_fdt_setprop(fdt, node, "iommus",
                         &iommu_ref, sizeof(iommu_ref));
    }

    return fdt;
}

static MachineState *rk3588_ms;

static void *rk3588_get_dtb(const struct arm_boot_info *info, int *size)
{
    return rk3588_create_dtb(rk3588_ms, size);
}

/* No-op: RAM is capped below the MMIO hole, so arm_boot.c's single-region
 * /memory layout is always correct. Callback kept for future use. */
static void rk3588_modify_dtb(const struct arm_boot_info *info, void *fdt)
{
    (void)info;
    (void)fdt;
}

static void rk3588_init(MachineState *ms)
{
    MemoryRegion *sysmem = get_system_memory();
    DeviceState *gicdev;
    SysBusDevice *gicbusdev;
    Object *cpuobj[RK3588_NUM_CPUS];
    uint64_t ram_size = ms->ram_size;

    rk3588_ms = ms;
    uint64_t max_low = RK3588_RAM_LOW_TOP - RK3588_RAM_BASE;
    if (ram_size > max_low) {
        warn_report("Capping RAM from %" PRIu64 " MiB to %" PRIu64
                    " MiB (rockchip-iommu DTE_ADDR is 32-bit; high memory "
                    "causes IOMMU page table corruption)",
                    ram_size / MiB, max_low / MiB);
        ram_size = max_low;
    }
    MemoryRegion *lowram = g_new(MemoryRegion, 1);

    /* RAM — single contiguous region below MMIO hole */
    memory_region_init_ram(lowram, NULL, "rk3588.lowram", ram_size,
                           &error_fatal);
    memory_region_add_subregion(sysmem, RK3588_RAM_BASE, lowram);

    /* CPUs — EL3 enabled for vendor kernel SMC SIP calls */
    for (int i = 0; i < ms->smp.cpus; i++) {
        cpuobj[i] = object_new(ms->cpu_type);
        if (object_property_find(cpuobj[i], "has_el3")) {
            object_property_set_bool(cpuobj[i], "has_el3", true,
                                     &error_fatal);
        }
        if (object_property_find(cpuobj[i], "has_el2")) {
            object_property_set_bool(cpuobj[i], "has_el2", false,
                                     &error_fatal);
        }
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
        int intidbase = GIC_NUM_SPI + i * 32;

        sysbus_connect_irq(gicbusdev, i,
                           qdev_get_gpio_in(cpudev, ARM_CPU_IRQ));
        sysbus_connect_irq(gicbusdev, i + RK3588_NUM_CPUS,
                           qdev_get_gpio_in(cpudev, ARM_CPU_FIQ));
        sysbus_connect_irq(gicbusdev, i + 2 * RK3588_NUM_CPUS,
                           qdev_get_gpio_in(cpudev, ARM_CPU_VIRQ));
        sysbus_connect_irq(gicbusdev, i + 3 * RK3588_NUM_CPUS,
                           qdev_get_gpio_in(cpudev, ARM_CPU_VFIQ));

        qdev_connect_gpio_out(cpudev, GTIMER_PHYS,
                              qdev_get_gpio_in(gicdev, intidbase + 30));
        qdev_connect_gpio_out(cpudev, GTIMER_VIRT,
                              qdev_get_gpio_in(gicdev, intidbase + 27));
        qdev_connect_gpio_out(cpudev, GTIMER_HYP,
                              qdev_get_gpio_in(gicdev, intidbase + 26));
        qdev_connect_gpio_out(cpudev, GTIMER_SEC,
                              qdev_get_gpio_in(gicdev, intidbase + 29));
    }

    /* UART2 (24 MHz, DW 8250 gap fill for vendor kernel) */
    serial_mm_init(sysmem, RK3588_UART2_BASE, 2,
                   qdev_get_gpio_in(gicdev, RK3588_UART2_IRQ),
                   24000000, serial_hd(0), DEVICE_LITTLE_ENDIAN);
    create_unimplemented_device("rk3588.uart2-dw",
                                RK3588_UART2_BASE + 0x20,
                                RK3588_UART2_SIZE - 0x20);

    /* Virtio-MMIO transports */
    for (int i = 0; i < RK3588_NUM_VIRTIO; i++) {
        hwaddr base = RK3588_VIRTIO_BASE + i * RK3588_VIRTIO_SIZE;
        sysbus_create_simple("virtio-mmio", base,
                             qdev_get_gpio_in(gicdev,
                                              RK3588_VIRTIO_IRQ + i));
    }

    /* CRU stub with PLL lock bits */
    {
        static RK3588CRUStub cru_stub;
        memset(&cru_stub, 0, sizeof(cru_stub));
        memory_region_init_io(&cru_stub.iomem, NULL, &rk3588_cru_ops,
                              &cru_stub, "rk3588.cru", RK3588_CRU_SIZE);
        memory_region_add_subregion(sysmem, RK3588_CRU_BASE,
                                    &cru_stub.iomem);
    }
    {
        static RK3588CRUStub pmu_cru_stub;
        memset(&pmu_cru_stub, 0, sizeof(pmu_cru_stub));
        memory_region_init_io(&pmu_cru_stub.iomem, NULL, &rk3588_cru_ops,
                              &pmu_cru_stub, "rk3588.pmu-cru",
                              RK3588_PMU_SIZE);
        memory_region_add_subregion(sysmem, RK3588_PMU_BASE,
                                    &pmu_cru_stub.iomem);
    }

    /* GRF/IOC stubs for vendor kernel built-in drivers */
    create_unimplemented_device("rk3588.sys-grf",      0xfd58c000, 0x1000);
    create_unimplemented_device("rk3588.pmu0-grf",     0xfd588000, 0x1000);
    create_unimplemented_device("rk3588.bigcore0-grf", 0xfd590000, 0x1000);
    create_unimplemented_device("rk3588.bigcore1-grf", 0xfd592000, 0x1000);
    create_unimplemented_device("rk3588.litcore-grf",  0xfd594000, 0x1000);
    create_unimplemented_device("rk3588.pmu1-grf",     0xfd5a0000, 0x1000);
    create_unimplemented_device("rk3588.usb-grf",      0xfd5ac000, 0x4000);
    create_unimplemented_device("rk3588.php-grf",      0xfd5b0000, 0x1000);
    create_unimplemented_device("rk3588.pipe-phy-grf", 0xfd5b8000, 0x1000);
    create_unimplemented_device("rk3588.ioc",          0xfd5f0000, 0x10000);
    create_unimplemented_device("rk3588.pmu",          0xfdd90000, 0x1000);

    /* NPU + Rockchip IOMMU */
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

        DeviceState *iommu_dev = qdev_new(TYPE_ROCKCHIP_IOMMU);
        sysbus_realize_and_unref(SYS_BUS_DEVICE(iommu_dev), &error_fatal);

        DeviceState *npu = qdev_new(TYPE_ROCKCHIP_NPU);
        RockchipNPUState *npu_s = ROCKCHIP_NPU(npu);
        npu_s->rk_iommu = ROCKCHIP_IOMMU(iommu_dev);

        SysBusDevice *npubus = SYS_BUS_DEVICE(npu);
        sysbus_realize_and_unref(npubus, &error_fatal);

        for (int i = 0; i < RK3588_NPU_NUM_CORES; i++) {
            sysbus_mmio_map(npubus, i, npu_bases[i]);
            sysbus_connect_irq(npubus, i,
                               qdev_get_gpio_in(gicdev, npu_irqs[i]));
        }
    }

    /* Boot */
    rk3588_binfo.ram_size = ram_size;
    rk3588_binfo.loader_start = RK3588_RAM_BASE;
    rk3588_binfo.board_id = -1;
    rk3588_binfo.psci_conduit = QEMU_PSCI_CONDUIT_SMC;
    rk3588_binfo.modify_dtb = rk3588_modify_dtb;
    if (ms->dtb) {
        rk3588_binfo.dtb_filename = ms->dtb;
    } else {
        rk3588_binfo.get_dtb = rk3588_get_dtb;
    }
    arm_load_kernel(ARM_CPU(cpuobj[0]), ms, &rk3588_binfo);
}

static void rk3588_machine_class_init(MachineClass *mc)
{
    mc->desc = "Rockchip RK3588 (Orange Pi 5 Plus) with NPU";
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
