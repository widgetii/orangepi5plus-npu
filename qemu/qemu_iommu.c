// SPDX-License-Identifier: GPL-2.0
/*
 * Dummy IOMMU for QEMU RK3588 NPU emulation.
 *
 * Provides a real IOMMU for the Rocket NPU driver, with identity-like
 * behavior. IOVA→GPA mappings are forwarded to a QEMU MMIO mailbox so
 * the QEMU NPU model can resolve DMA addresses in regcmds.
 *
 * Mailbox registers (at DT "reg" address):
 *   0x00: IOVA  (write page-aligned IOVA)
 *   0x04: PHYS  (write GPA → adds mapping)
 *   0x08: UNMAP (write IOVA → removes mapping)
 */
#include <linux/module.h>
#include <linux/of.h>
#include <linux/of_iommu.h>
#include <linux/platform_device.h>
#include <linux/iommu.h>
#include <linux/io.h>

struct qemu_iommu {
	struct iommu_device iommu;
	struct device *dev;
	void __iomem *mailbox;  /* QEMU MMIO mailbox for IOVA→GPA */
};

struct qemu_iommu_domain {
	struct iommu_domain domain;
	struct qemu_iommu *qi;  /* back-pointer for mailbox access */
};

static struct qemu_iommu *global_qiommu;

static struct iommu_domain *qemu_iommu_domain_alloc_paging(struct device *dev)
{
	struct qemu_iommu_domain *qdom;

	qdom = kzalloc(sizeof(*qdom), GFP_KERNEL);
	if (!qdom)
		return ERR_PTR(-ENOMEM);

	qdom->domain.geometry.aperture_start = 0;
	qdom->domain.geometry.aperture_end = (1ULL << 32) - 1;
	qdom->domain.geometry.force_aperture = true;
	qdom->domain.pgsize_bitmap = PAGE_SIZE;
	qdom->qi = global_qiommu;

	return &qdom->domain;
}

static void qemu_iommu_domain_free(struct iommu_domain *domain)
{
	kfree(container_of(domain, struct qemu_iommu_domain, domain));
}

static int qemu_iommu_attach_dev(struct iommu_domain *domain, struct device *dev)
{
	return 0;
}

static int qemu_iommu_map_pages(struct iommu_domain *domain, unsigned long iova,
				phys_addr_t paddr, size_t pgsize,
				size_t pgcount, int prot, gfp_t gfp,
				size_t *mapped)
{
	struct qemu_iommu_domain *qdom =
		container_of(domain, struct qemu_iommu_domain, domain);

	/* Notify QEMU of each page mapping via the mailbox */
	if (qdom->qi && qdom->qi->mailbox) {
		for (size_t i = 0; i < pgcount; i++) {
			writel((u32)(iova + i * pgsize), qdom->qi->mailbox + 0x00);
			writel((u32)(paddr + i * pgsize), qdom->qi->mailbox + 0x04);
		}
	}

	if (mapped)
		*mapped = pgsize * pgcount;
	return 0;
}

static size_t qemu_iommu_unmap_pages(struct iommu_domain *domain,
				     unsigned long iova, size_t pgsize,
				     size_t pgcount,
				     struct iommu_iotlb_gather *gather)
{
	struct qemu_iommu_domain *qdom =
		container_of(domain, struct qemu_iommu_domain, domain);

	if (qdom->qi && qdom->qi->mailbox) {
		for (size_t i = 0; i < pgcount; i++)
			writel((u32)(iova + i * pgsize), qdom->qi->mailbox + 0x08);
	}

	return pgsize * pgcount;
}

static phys_addr_t qemu_iommu_iova_to_phys(struct iommu_domain *domain,
					    dma_addr_t iova)
{
	return (phys_addr_t)iova;  /* Fallback; real translation in QEMU */
}

static struct iommu_device *qemu_iommu_probe_device(struct device *dev)
{
	if (!global_qiommu)
		return ERR_PTR(-ENODEV);
	return &global_qiommu->iommu;
}

static struct iommu_group *qemu_iommu_device_group(struct device *dev)
{
	return generic_single_device_group(dev);
}

static int qemu_iommu_of_xlate(struct device *dev,
				const struct of_phandle_args *args)
{
	return 0;
}

static const struct iommu_domain_ops qemu_iommu_domain_ops = {
	.attach_dev	= qemu_iommu_attach_dev,
	.map_pages	= qemu_iommu_map_pages,
	.unmap_pages	= qemu_iommu_unmap_pages,
	.iova_to_phys	= qemu_iommu_iova_to_phys,
	.free		= qemu_iommu_domain_free,
};

static struct iommu_domain qemu_identity_domain = {
	.type = IOMMU_DOMAIN_IDENTITY,
	.ops = &qemu_iommu_domain_ops,
};

static const struct iommu_ops qemu_iommu_ops = {
	.owner			= THIS_MODULE,
	.identity_domain	= &qemu_identity_domain,
	.domain_alloc_paging	= qemu_iommu_domain_alloc_paging,
	.probe_device		= qemu_iommu_probe_device,
	.device_group		= qemu_iommu_device_group,
	.of_xlate		= qemu_iommu_of_xlate,
	.default_domain_ops	= &qemu_iommu_domain_ops,
};

static int qemu_iommu_probe(struct platform_device *pdev)
{
	struct qemu_iommu *qiommu;
	struct resource *res;
	int ret;

	qiommu = devm_kzalloc(&pdev->dev, sizeof(*qiommu), GFP_KERNEL);
	if (!qiommu)
		return -ENOMEM;

	qiommu->dev = &pdev->dev;

	/* Map the QEMU mailbox MMIO region */
	res = platform_get_resource(pdev, IORESOURCE_MEM, 0);
	if (res) {
		qiommu->mailbox = devm_ioremap_resource(&pdev->dev, res);
		if (IS_ERR(qiommu->mailbox)) {
			dev_warn(&pdev->dev, "mailbox ioremap failed, "
				 "IOVA translation will use identity\n");
			qiommu->mailbox = NULL;
		}
	}

	platform_set_drvdata(pdev, qiommu);

	ret = iommu_device_sysfs_add(&qiommu->iommu, &pdev->dev, NULL,
				     "qemu-iommu");
	if (ret)
		return ret;

	ret = iommu_device_register(&qiommu->iommu, &qemu_iommu_ops,
				    &pdev->dev);
	if (ret) {
		dev_err(&pdev->dev, "iommu_device_register failed: %d\n", ret);
		iommu_device_sysfs_remove(&qiommu->iommu);
		return ret;
	}

	global_qiommu = qiommu;
	dev_info(&pdev->dev, "QEMU dummy IOMMU registered (mailbox=%s)\n",
		 qiommu->mailbox ? "yes" : "no");
	return 0;
}

static const struct of_device_id qemu_iommu_of_ids[] = {
	{ .compatible = "qemu,iommu-dummy" },
	{}
};
MODULE_DEVICE_TABLE(of, qemu_iommu_of_ids);

static struct platform_driver qemu_iommu_driver = {
	.probe = qemu_iommu_probe,
	.driver = {
		.name = "qemu-iommu",
		.of_match_table = qemu_iommu_of_ids,
	},
};
module_platform_driver(qemu_iommu_driver);
MODULE_DESCRIPTION("QEMU dummy IOMMU with mailbox for Rocket NPU");
MODULE_LICENSE("GPL");
