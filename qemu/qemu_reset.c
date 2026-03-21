// SPDX-License-Identifier: GPL-2.0
// Dummy reset controller for QEMU — all reset ops are no-ops
#include <linux/module.h>
#include <linux/of.h>
#include <linux/platform_device.h>
#include <linux/reset-controller.h>

static int qemu_reset_assert(struct reset_controller_dev *rcdev, unsigned long id) { return 0; }
static int qemu_reset_deassert(struct reset_controller_dev *rcdev, unsigned long id) { return 0; }
static int qemu_reset_status(struct reset_controller_dev *rcdev, unsigned long id) { return 0; }

static const struct reset_control_ops qemu_reset_ops = {
    .assert   = qemu_reset_assert,
    .deassert = qemu_reset_deassert,
    .status   = qemu_reset_status,
};

static int qemu_reset_probe(struct platform_device *pdev)
{
    struct reset_controller_dev *rcdev;
    rcdev = devm_kzalloc(&pdev->dev, sizeof(*rcdev), GFP_KERNEL);
    if (!rcdev) return -ENOMEM;
    rcdev->ops = &qemu_reset_ops;
    rcdev->of_node = pdev->dev.of_node;
    rcdev->nr_resets = 1024;
    return devm_reset_controller_register(&pdev->dev, rcdev);
}

static const struct of_device_id qemu_reset_ids[] = {
    { .compatible = "qemu,reset-dummy" },
    {}
};
MODULE_DEVICE_TABLE(of, qemu_reset_ids);

static struct platform_driver qemu_reset_driver = {
    .probe = qemu_reset_probe,
    .driver = { .name = "qemu-reset", .of_match_table = qemu_reset_ids },
};
module_platform_driver(qemu_reset_driver);
MODULE_LICENSE("GPL");
