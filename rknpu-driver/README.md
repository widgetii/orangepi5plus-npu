# RKNPU Driver Fork (Instrumented)

Forked from [armbian/linux-rockchip](https://github.com/armbian/linux-rockchip/tree/rk-6.1-rkr5/drivers/rknpu) branch `rk-6.1-rkr5`.

## Purpose

Instrumented RKNPU kernel driver for tracing all NPU data flow:
register writes, DMA content, task submissions, IRQ completion.
Used to achieve byte-identical output between librocketnpu and RKNN.

## Building

Build as an **out-of-tree module** on the Orange Pi 5 Plus (vendor kernel):

```bash
cd rknpu-driver
make    # uses /lib/modules/$(uname -r)/build
```

Produces `rknpu_trace.ko`.

## IMPORTANT: Do NOT rebuild the full vendor kernel

The Armbian vendor kernel `6.1.115-vendor-rk35xx` is built from the
`rk-6.1-rkr5` branch (base 6.1.99) with **hundreds of Armbian patches**
that bump the version to 6.1.115 and modify the ABI. Attempting to
rebuild the full kernel from the upstream branch source causes:

1. **ABI mismatch**: Source is 6.1.99, running kernel is 6.1.115.
   Even with `SUBLEVEL = 115` in the Makefile, internal structures
   differ. `modules_install` overwrites working modules with
   incompatible ones.

2. **Network failure on boot**: The Realtek r8125 2.5G ethernet driver
   and other critical modules fail to load with the wrong ABI, leaving
   the board unreachable via SSH.

3. **Recovery requires UART or SD card**: The board boots but has no
   network. Must recover via U-Boot serial console or backup SD card.

### What works

- **Out-of-tree module build** against installed kernel headers
  (`/usr/src/linux-headers-6.1.115-vendor-rk35xx`) — this compiles
  the RKNPU driver as a loadable module without touching the rest
  of the kernel.

### What doesn't work

- `make Image && cp Image /boot/` — kernel boots but modules are
  ABI-incompatible, network down
- `make modules_install` — overwrites working modules with broken ones
- Unbinding the built-in RKNPU driver from userspace
  (`echo fdab0000.npu > /sys/bus/platform/drivers/RKNPU/unbind`) —
  causes kernel panic due to IOMMU teardown races

## Loading the module

The stock RKNPU driver is **built-in** (`CONFIG_ROCKCHIP_RKNPU=y`),
not a module. To replace it with our instrumented version:

**TODO**: Implement safe driver swap mechanism. Options under investigation:
- `driver_override` sysfs approach (set override before unbind)
- Kernel shim module that does atomic unbind + rebind
- Devicetree overlay with different compatible string

## Files

Downloaded from upstream `rk-6.1-rkr5` branch, with tracing additions:

| File | Description |
|------|-------------|
| `rknpu_drv.c` | Device probe, power management (devfreq/debugger stubbed) |
| `rknpu_job.c` | Job submission, IRQ handlers, REG_WRITE tracing |
| `rknpu_gem.c` | DRM GEM memory management |
| `rknpu_fence.c` | DMA fence support |
| `rknpu_iommu.c` | IOMMU domain management (dma_cookie stubbed) |
| `rknpu_mem.c` | Memory allocation helpers |
| `rknpu_mm.c` | Memory manager |
| `rknpu_reset.c` | Hardware reset |
| `rknpu_trace.c` | Tracing infrastructure (debugfs + relay channel) |
| `include/rknpu_trace.h` | Trace record definitions |
| `Makefile` | Out-of-tree module build |
