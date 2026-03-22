# Bug: rockchip-iommu allocates page table directory with GFP_KERNEL

## Summary

The mainline `rockchip-iommu` driver allocates its page table directory
with `get_zeroed_page(GFP_KERNEL)`, which can return a page above 4GB on
systems with high memory. The 32-bit `DTE_ADDR` register truncates the
address, causing all IOMMU translations to fail.

## Affected code

File: `drivers/iommu/rockchip-iommu.c`

```c
static int rk_iommu_domain_alloc_paging(struct device *dev)
{
    ...
    domain->dt = (u32 *)get_zeroed_page(GFP_KERNEL);   // <-- bug
    domain->dt_dma = dma_map_single(dev, domain->dt,
                                    IOMMU_DTE_SIZE, DMA_TO_DEVICE);
    ...
}
```

Then later:

```c
rk_iommu_write(iommu->bases[i], RK_MMU_DTE_ADDR, domain->dt_dma);
```

Where `rk_iommu_write` calls `writel(value, base + offset)` — a 32-bit
write that truncates `dt_dma` to its low 32 bits.

## Proposed fix

```c
- domain->dt = (u32 *)get_zeroed_page(GFP_KERNEL);
+ domain->dt = (u32 *)get_zeroed_page(GFP_DMA32);
```

`GFP_DMA32` guarantees the page is below 4GB, fitting the 32-bit register.

The same fix may be needed for `rk_mk_dte` / `rk_mk_dte_v2` which encode
page table second-level bases — those also use `get_zeroed_page(GFP_KERNEL)`
in `rk_iommu_zap_iova_first` and related paths.

## Reproduction

Requires a system with a Normal zone (RAM above 4GB). On RK3588 boards
with ≤4GB this never triggers because all RAM is in the DMA zone.

### QEMU reproduction (deterministic)

```sh
# Build QEMU with RK3588 NPU emulator (see qemu-boot/README.md)

# FAILS — Normal zone exists, IOMMU directory lands at >4GB:
qemu-system-aarch64 -M orangepi5plus -m 4G -nographic -smp 4 \
  -kernel Image-6.18 -initrd initrd.gz \
  -append "console=ttyS0,1500000 earlycon panic=10"
# Result: conv tests 0/6, IOMMU reads garbage page table

# PASSES — mem= eliminates Normal zone, proving the root cause:
qemu-system-aarch64 -M orangepi5plus -m 4G -nographic -smp 4 \
  -kernel Image-6.18 -initrd initrd.gz \
  -append "console=ttyS0,1500000 earlycon panic=10 mem=3838M"
# Result: conv tests 6/6, MobileNetV1 bit-exact
```

The failure threshold is ~3840 MiB — the point where low DRAM (below the
SoC MMIO region at 0xF0000000) is exhausted and a Normal zone appears.

### On real hardware

Likely affects RK3588 boards with 8GB or 16GB RAM (Orange Pi 5 Plus 16GB,
Rock 5B 16GB, etc.) under memory pressure that forces `GFP_KERNEL` to
allocate from the Normal zone. May manifest as intermittent NPU failures
or IOMMU faults.

## Impact

- All DMA through the Rockchip IOMMU returns wrong physical addresses
- NPU convolutions read/write garbage memory
- Affects both Rocket (mainline) and RKNPU (vendor) kernel drivers

## Workarounds

1. **QEMU emulator**: RAM capped at ~3838 MiB (our current approach)
2. **Real hardware**: `mem=3838M` kernel parameter (limits usable RAM)
3. **Real hardware**: `iommu.passthrough=1` (bypasses IOMMU entirely —
   only works if all DMA buffers are in the 32-bit address range)

## Status

- Discovered: 2026-03-22
- Upstream report: not yet filed
- Kernel versions tested: 6.18.10-current-rockchip64 (mainline)
