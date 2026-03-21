/*
 * Rockchip RK3588 SoC — QEMU machine type constants
 *
 * SPDX-License-Identifier: GPL-2.0-or-later
 */

#ifndef HW_ARM_RK3588_H
#define HW_ARM_RK3588_H

/* CPU configuration */
#define RK3588_NUM_CPUS     4

/* Memory map */
#define RK3588_RAM_BASE       0x00200000ULL
#define RK3588_CRU_BASE       0xfd7c0000ULL
#define RK3588_CRU_SIZE       0x00010000ULL
#define RK3588_PMU_BASE       0xfd8d8000ULL
#define RK3588_PMU_SIZE       0x00010000ULL
#define RK3588_GIC_DIST_BASE  0xfe600000ULL
#define RK3588_GIC_REDIST_BASE 0xfe680000ULL
#define RK3588_UART2_BASE     0xfeb50000ULL
#define RK3588_UART2_SIZE     0x00001000ULL

/* NPU core MMIO base addresses */
#define RK3588_NPU_NUM_CORES  3
#define RK3588_NPU_CORE0_BASE 0xfdab0000ULL
#define RK3588_NPU_CORE1_BASE 0xfdac0000ULL
#define RK3588_NPU_CORE2_BASE 0xfdad0000ULL

/* Interrupts (SPI numbers) */
#define RK3588_UART2_IRQ      148
#define RK3588_NPU_CORE0_IRQ  110
#define RK3588_NPU_CORE1_IRQ  111
#define RK3588_NPU_CORE2_IRQ  112

/* NPU IOMMU mailbox (QEMU-specific, for IOVA→GPA translation) */
#define NPU_IOMMU_BASE       0xfdaf0000ULL

/* High RAM (above 4GB) */
#define RK3588_RAM_HIGH_BASE  0x100000000ULL

#endif /* HW_ARM_RK3588_H */
