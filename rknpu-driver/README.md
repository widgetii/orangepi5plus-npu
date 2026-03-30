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

## Tracing the built-in driver with kprobes (RECOMMENDED)

No module swap needed. The vendor kernel has ftrace and kprobes enabled,
allowing us to trace all 92 RKNPU kernel functions from the built-in driver.

### Setup

```bash
# Set up kprobes on submit and IRQ handlers
# rknpu_submit struct offsets: flags(+0) timeout(+4) task_start(+8)
# task_number(+12) core_mask(+56). x1 = pointer to struct.
echo 'p:rknpu_sub __rknpu_submit_ioctl flags=+0(%x1):u32 timeout=+4(%x1):u32 task_start=+8(%x1):u32 task_num=+12(%x1):u32 core_mask=+56(%x1):u32' > /sys/kernel/debug/tracing/kprobe_events
echo 'p:rknpu_irq0 rknpu_core0_irq_handler' >> /sys/kernel/debug/tracing/kprobe_events

# Enable probes
echo 1 > /sys/kernel/debug/tracing/events/kprobes/enable
echo > /sys/kernel/debug/tracing/trace
echo 1 > /sys/kernel/debug/tracing/tracing_on
```

### Running a trace

```bash
# Clear buffer, run workload, read trace
echo > /sys/kernel/debug/tracing/trace
./test_mobilenet /root/npu-research/mobilenet_v1.tflite 1
cat /sys/kernel/debug/tracing/trace | grep rknpu_sub
```

### Example output

```
test_mobilenet-2810 [004] d.... 191.345131: rknpu_sub: flags=5 timeout=6000 task_start=0 task_num=3 core_mask=0
test_mobilenet-2810 [004] d.... 191.345481: rknpu_sub: flags=5 timeout=6000 task_start=0 task_num=2 core_mask=0
```

### Tracing RKNN for comparison

```bash
echo > /sys/kernel/debug/tracing/trace
LD_PRELOAD=/tmp/intercept_swap.so DUMP_REGCMD=0 python3 run_rknn_intercept.py
cat /sys/kernel/debug/tracing/trace | grep rknpu_sub
```

RKNN submit pattern (YOLO):
```
flags=5 task_start=0   task_num=591  core_mask=1   # HW segment 1 (591 chained tasks)
flags=1 task_start=197 task_num=12   core_mask=1   # SW segment (no pingpong)
flags=5 task_start=201 task_num=42   core_mask=1   # HW segment 2
flags=1 task_start=215 task_num=9    core_mask=1   # SW segment
flags=5 task_start=218 task_num=21   core_mask=1   # HW segment 3
flags=1 task_start=225 task_num=6    core_mask=1   # SW segment
```

### Cleanup

```bash
echo 0 > /sys/kernel/debug/tracing/events/kprobes/enable
echo > /sys/kernel/debug/tracing/kprobe_events
```

### Available RKNPU functions (92 total)

Key functions for tracing:
- `__rknpu_submit_ioctl` — submit entry point (args: dev, rknpu_submit*)
- `rknpu_job_next` — job dispatch to hardware
- `rknpu_job_schedule` — job scheduling
- `rknpu_core{0,1,2}_irq_handler` — IRQ completion per core
- `__rknpu_gem_create_ioctl` — BO allocation
- `__rknpu_gem_sync_ioctl` — cache sync

Full list: `grep rknpu /sys/kernel/debug/tracing/available_filter_functions`

## Loading the instrumented module

The stock RKNPU driver is **built-in** (`CONFIG_ROCKCHIP_RKNPU=y`).
To replace it with our instrumented module:

### One-time setup

Add `initcall_blacklist=rknpu_init` to kernel boot args:

```bash
# Edit /boot/armbianEnv.txt, prepend to extraargs:
extraargs=initcall_blacklist=rknpu_init cma=256M panic=10 ...
```

Reboot. The built-in RKNPU driver won't probe (dmesg shows
`initcall rknpu_init blacklisted`). The NPU device is unbound.

### Loading our module

```bash
insmod /root/npu-research/rknpu-driver/rknpu_trace.ko
# Verify: /dev/dri/card1 appears, dmesg shows RKNPU_TRACE
```

### Restoring the built-in driver

Remove `initcall_blacklist=rknpu_init` from `/boot/armbianEnv.txt`
and reboot.

### Why unbind doesn't work

Direct unbind (`echo fdab0000.npu > .../RKNPU/unbind`) crashes the
kernel due to IOMMU teardown + power domain cascade. The
`initcall_blacklist` approach avoids this by preventing the built-in
from ever probing.

### Build symlink fix

If `modules_install` was run from the wrong kernel tree, fix the
build symlink:
```bash
rm /lib/modules/6.1.115-vendor-rk35xx/build
ln -s /usr/src/linux-headers-6.1.115-vendor-rk35xx /lib/modules/6.1.115-vendor-rk35xx/build
```

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
