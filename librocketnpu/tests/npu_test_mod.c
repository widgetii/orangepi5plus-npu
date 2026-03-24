/*
 * NPU flush-all-BOs test: walks IOMMU page table to find ALL mapped pages,
 * flushes their caches, then re-pulses OP_EN.
 */
#include <linux/module.h>
#include <linux/io.h>
#include <linux/delay.h>

static inline void dc_civac_range(void *start, unsigned long len) {
	unsigned long addr = (unsigned long)start & ~63UL;
	unsigned long end = (unsigned long)start + len;
	for (; addr < end; addr += 64)
		asm volatile("dc civac, %0" :: "r"(addr) : "memory");
	asm volatile("dsb sy" ::: "memory");
}

static inline phys_addr_t rk_addr_v2(u32 dte) {
	u64 d = dte;
	d = ((d & GENMASK(7,4)) << 32) | ((d & GENMASK(11,8)) << 24) | (d & 0xfffff000);
	return (phys_addr_t)d;
}

static int flush_all = 1;
module_param(flush_all, int, 0);

static int __init npu_test_init(void)
{
	void __iomem *base, *iommu_regs;
	u32 raw;
	int i, j, total_flushed = 0;

	base = ioremap(0xfdab0000, 0x10000);
	iommu_regs = ioremap(0xfdab9000, 0x100);
	if (!base || !iommu_regs) return -ENOMEM;

	pr_info("npu_test: PC_DATA_ADDR=0x%08x INT_RAW=0x%08x STATUS=0x%08x\n",
		readl(base + 0x10), readl(base + 0x2c), readl(base + 0x3c));

	if (flush_all) {
		/* Walk ALL IOMMU pages and flush their dcache */
		u32 dte_reg = readl(iommu_regs + 0x0);
		u32 *l1 = (u32 *)phys_to_virt(rk_addr_v2(dte_reg));

		for (i = 0; i < 1024; i++) {
			if (!(l1[i] & 1)) continue;
			u32 *l2 = (u32 *)phys_to_virt(rk_addr_v2(l1[i]));
			for (j = 0; j < 1024; j++) {
				if (!(l2[j] & 1)) continue;
				phys_addr_t pg = rk_addr_v2(l2[j]);
				void *va = phys_to_virt(pg);
				dc_civac_range(va, 4096);
				total_flushed++;
			}
		}
		pr_info("npu_test: Flushed dcache for %d IOMMU-mapped pages\n", total_flushed);
	}

	/* Re-pulse OP_EN */
	pr_info("npu_test: Pulsing OP_EN...\n");
	writel(0x1, base + 0x08);
	writel(0x0, base + 0x08);

	for (i = 0; i < 100; i++) {
		raw = readl(base + 0x2c);
		if (raw & 0x300) {
			pr_info("npu_test: TASK DONE at %dms! RAW=0x%08x STATUS=0x%08x\n",
				i, raw, readl(base + 0x3c));
			writel(0x1ffff, base + 0x24);
			break;
		}
		if (i < 5 || raw != 0xc0000000)
			pr_info("npu_test: [%dms] RAW=0x%08x STATUS=0x%08x\n",
				i, raw, readl(base + 0x3c));
		msleep(1);
	}
	if (i >= 100) {
		pr_info("npu_test: TIMEOUT RAW=0x%08x STATUS=0x%08x\n",
			readl(base + 0x2c), readl(base + 0x3c));
		pr_info("npu_test: IOMMU PF=0x%08x\n", readl(iommu_regs + 0xc));
	}

	/* Check IRQ count */
	pr_info("npu_test: CNA SRC=0x%08x WT=0x%08x DST=0x%08x\n",
		readl(base + 0x1070), readl(base + 0x1110), readl(base + 0x4020));

	iounmap(iommu_regs);
	iounmap(base);
	return -ENODEV;
}

static void __exit npu_test_exit(void) {}
module_init(npu_test_init);
module_exit(npu_test_exit);
MODULE_LICENSE("GPL");
