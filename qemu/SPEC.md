# QEMU RK3588 with Functional Rocket NPU — Design Specification

## Overview

A QEMU machine type (`orangepi5plus`) that emulates the RK3588 SoC with a software Rocket NPU model. Boots mainline Linux 6.18+ kernels, probes the upstream Rocket DRM/accel driver, and executes INT8 convolution jobs entirely in software. Enables CI testing of the kernel driver and Mesa/libteflon/librocketnpu userspace without physical hardware.

Built against QEMU v10.2.0. Total implementation: ~2700 lines across 13 files.

## Architecture

### Components

```
┌─────────────────────────────────────────────────┐
│                    QEMU Host                     │
│                                                  │
│  ┌──────────────────────────────────────────┐   │
│  │ rk3588.c — Machine Type                  │   │
│  │  4x Cortex-A55, GICv3, UART, DTB gen     │   │
│  └──────────────┬───────────────────────────┘   │
│                 │                                │
│  ┌──────────────▼───────────────────────────┐   │
│  │ rockchip-npu.c — NPU Device Model        │   │
│  │  3 cores × (MMIO + regcmd parser + conv) │   │
│  │  IOMMU mailbox for IOVA→GPA translation  │   │
│  └──────────────────────────────────────────┘   │
│                                                  │
├─────────────────────────────────────────────────┤
│                   Guest Kernel                   │
│                                                  │
│  ┌─────────────┐ ┌────────────┐ ┌────────────┐ │
│  │ rocket.ko   │ │ qemu_      │ │ qemu_      │ │
│  │ (upstream)  │ │ iommu.ko   │ │ reset.ko   │ │
│  └─────────────┘ └────────────┘ └────────────┘ │
│                                                  │
├─────────────────────────────────────────────────┤
│                   Guest Userspace                │
│  libteflon / librocketnpu / bench_teflon.py     │
└─────────────────────────────────────────────────┘
```

### File Structure

| File | Lines | Role |
|------|-------|------|
| `hw/arm/rk3588.c` | 401 | Machine type: CPUs, GICv3, UART, DTB generation, device wiring |
| `hw/misc/rockchip-npu.c` | 1171 | NPU device: MMIO handlers, regcmd parser, INT8 conv engine, SDP pipeline, IOMMU mailbox |
| `hw/misc/rockchip-npu.h` | 281 | NPU register offsets, RocketConvTask struct, tensor offset macros |
| `include/hw/arm/rk3588.h` | 42 | Memory map constants, IRQ numbers |
| `qemu_iommu.c` | 205 | Kernel module: dummy IOMMU with MMIO mailbox for IOVA→GPA forwarding |
| `qemu_reset.c` | 40 | Kernel module: dummy reset controller |
| `tests/npu_test.c` | 444 | Userspace test: CREATE_BO, fill regcmd, SUBMIT, verify output |
| `hw/arm/Kconfig.rk3588` | 10 | Kconfig additions for ORANGEPI5PLUS |
| `hw/misc/Kconfig.rockchip_npu` | 3 | Kconfig additions for ROCKCHIP_NPU |

## Machine Type: orangepi5plus

### Memory Map

| Device | Address | QEMU Model | Notes |
|--------|---------|------------|-------|
| DDR (low) | 0x00200000 | RAM | Default 4GB, configurable |
| CRU stub | 0xfd7c0000 | `create_unimplemented_device` | Returns 0, prevents clock driver crash |
| PMU stub | 0xfd8d8000 | `create_unimplemented_device` | Returns 0 |
| NPU core 0 | 0xfdab0000 | `rockchip-npu` (MMIO) | PC+CNA+Core+DPU+RDMA regions |
| NPU core 1 | 0xfdac0000 | `rockchip-npu` (MMIO) | |
| NPU core 2 | 0xfdad0000 | `rockchip-npu` (MMIO) | |
| IOMMU mailbox | 0xfdaf0000 | `rockchip-npu` (MMIO) | QEMU-specific, IOVA→GPA forwarding |
| GICv3 GICD | 0xfe600000 | `arm_gicv3` | 192 SPIs |
| GICv3 GICR | 0xfe680000 | `arm_gicv3` | Per-CPU redistributors |
| UART2 | 0xfeb50000 | `serial_mm` (16550A) | Console, `ns16550a` compatible |
| DDR (high) | 0x100000000 | RAM | Above 4GB |

### CPU Configuration

- 4× Cortex-A55 (homogeneous, skip big.LITTLE — not needed for NPU CI)
- EL3/EL2 disabled (no TrustZone, no hypervisor)
- Secondary CPUs start powered off, woken via PSCI HVC
- ARM generic timer at 62.5 MHz (built into CPU model)

### DTB Generation

The machine generates a complete DTB programmatically via `get_dtb` callback (no external DTB needed). Key nodes:

- `/cpus` — 4 CPU nodes with `enable-method = "psci"`
- `/psci` — `arm,psci-1.0` with `method = "hvc"`
- `/timer` — `arm,armv8-timer` with standard PPI numbers
- `/intc` — `arm,gic-v3` with GICD/GICR reg entries
- `/serial@feb50000` — `ns16550a` (not `snps,dw-apb-uart` — kernel earlycon requires this)
- `/npu-clk` — `fixed-clock` at 1 GHz for NPU clock references
- `/npu-reset` — `qemu,reset-dummy` with unique per-core reset cell IDs
- `/qemu-iommu@fdaf0000` — `qemu,iommu-dummy` with reg for mailbox MMIO
- `/npu@fdab0000` etc. — `rockchip,rk3588-rknn-core` with reg-names, clocks, resets, iommus

### GIC Wiring

```
Timer PPI → GIC:  intidbase = GIC_NUM_SPI + cpu_index * 32
                  GTIMER_PHYS → intidbase + 30  (ARCH_TIMER_NS_EL1_IRQ)
                  GTIMER_VIRT → intidbase + 27
                  GTIMER_HYP  → intidbase + 26
                  GTIMER_SEC  → intidbase + 29

GIC → CPU:        sysbus_connect_irq(gic, i,            cpu ARM_CPU_IRQ)
                  sysbus_connect_irq(gic, i + N,        cpu ARM_CPU_FIQ)
                  sysbus_connect_irq(gic, i + 2N,       cpu ARM_CPU_VIRQ)
                  sysbus_connect_irq(gic, i + 3N,       cpu ARM_CPU_VFIQ)

NPU SPI:          Core 0 → SPI 110, Core 1 → SPI 111, Core 2 → SPI 112
UART SPI:         UART2 → SPI 148
```

### Boot Flow

1. `arm_load_kernel()` places boot stub at RAM_BASE (0x200000)
2. Stub loads DTB address into x0, jumps to kernel at RAM_BASE + 0x200000
3. Kernel decompresses, sets up MMU, initializes GICv3 and timer
4. PSCI wakes secondary CPUs
5. Full kernel init completes (~1.6s in QEMU TCG)

**Required kernel cmdline**: `console=ttyS0 earlycon iommu.passthrough=1 rdinit=/init`

The `iommu.passthrough=1` is needed because the dummy IOMMU module registers late (as a loadable module), and without passthrough mode the DMA default domain setup fails.

## NPU Device Model

### MMIO Regions

Each NPU core has a single 24KB MMIO region (`NPU_REGION_SIZE = 0x6000`) covering:

| Offset | Block | Description |
|--------|-------|-------------|
| 0x0000 | PC | Program Controller — job control, IRQ, task chaining |
| 0x1000 | CNA | Convolution engine — input/weight config, DMA addresses |
| 0x3000 | CORE | Core control — output dimensions, misc config |
| 0x4000 | DPU | Data Processing Unit — output DMA, quantization, bias/BN/EW |
| 0x5000 | RDMA | Read DMA — bias/BN data source addresses |

### Job Execution Flow

When the kernel writes `PC_OPERATION_ENABLE = 1`:

```
1. Read PC_BASE_ADDRESS (IOVA of regcmd buffer)
2. Read PC_REGISTER_AMOUNTS (encoded count)
3. Decode count: actual_entries = (encoded + 1) * 2
4. Read regcmd buffer via IOVA→GPA translation (npu_dma_read)
5. Parse regcmd entries → build RocketConvTask
6. Execute INT8 convolution in software
7. Write output tensor via IOVA→GPA translation (npu_dma_write)
8. Follow task chain pointers (if multi-task job)
9. Set pc_irq_raw_status |= DPU_0 | DPU_1 (0x0300)
10. Raise IRQ if (raw_status & irq_mask) != 0
```

Execution is **synchronous** within the MMIO write handler — deterministic, no threading.

### Regcmd Format

Each regcmd entry is a packed `uint64_t`:
```
bits [15:0]  = register offset (0x1xxx=CNA, 0x3xxx=Core, 0x4xxx=DPU, 0x5xxx=RDMA)
bits [47:16] = 32-bit register value
bits [63:48] = target block selector (0x0201=CNA, 0x0801=Core, 0x1001=DPU, 0x2001=RDMA)
```

Special entries:
- Target `0x0041` — pre-operation control marker (ignored by parser)
- Target `0x0081` — operation enable trigger (ignored by parser)
- Entry `0x0` — null/padding (ignored)

The last 4 entries of each task are: chain pointer (PC_BASE_ADDRESS), chain count (PC_REGISTER_AMOUNTS), `0x0041` marker, `0x0081` enable.

### Register Encoding Details

**CNA registers store RAW values** (Mesa emits without adjustment):
- `DATA_SIZE0` (0x1020): HEIGHT[10:0], WIDTH[26:16]
- `DATA_SIZE1` (0x1024): CHANNEL[15:0] (raw), CHANNEL_REAL[29:16] (count-1)
- `WEIGHT_SIZE2` (0x1038): KERNELS[13:0], HEIGHT[20:16], WIDTH[28:24]
- `CONV_CON3` (0x1014): STRIDE_X[2:0], STRIDE_Y[5:3]
- `PAD_CON0` (0x1068): PAD_TOP[3:0], PAD_LEFT[7:4]
- `PAD_CON1` (0x1184): pad_value (signed int8 in lowest byte)

**CORE/DPU registers store COUNT-1**:
- `DATAOUT_SIZE_0` (0x3014): WIDTH[15:0]+1, HEIGHT[31:16]+1
- `DATAOUT_SIZE_1` (0x3018): CHANNEL[12:0]+1
- `DATA_CUBE_CHANNEL` (0x403c): CHANNEL[12:0]+1, ORIG_CHANNEL[28:16]+1

**DPU DST_SURF_STRIDE** (0x4024): value shifted left by 4 in register, parser shifts right.

**DPU additional parsed registers** (stored in RocketConvTask for future use):
- `FEATURE_MODE_CFG` (0x400c): OUTPUT_MODE, BURST_LEN, CONV_MODE mirror
- `DATA_FORMAT` (0x4010): BS_MUL_SHIFT_VALUE_NEG bits
- `BS_OW_CFG` (0x4050): SIZE_E field (3 for depthwise, 1 for standard)
- `WDMA_SIZE_0` (0x4058), `WDMA_SIZE_1` (0x405c): output DMA channel counts
- `EW_RELUX_CMP` (0x407c): EW ReLUx clamp value
- `SURFACE_ADD` (0x40c0): surfaces per row for output addressing

**RDMA additional parsed registers**:
- `SRC_BASE_ADDR` (0x5018), `NRDMA_CFG` (0x5028), `BN_BASE_ADDR` (0x502c)
- `FEATURE_MODE_CFG` (0x5044), `WEIGHT` (0x5068)

### INT8 Convolution Engine

```c
for (oy = 0..out_h)
  for (ox = 0..out_w)
    for (oc = 0..out_c)
      // CACC: accumulate
      acc = 0
      for (ky, kx, ic): acc += in_val * w_val
      // CACC truncation (NVDLA round-half-to-even)
      acc = nvdla_truncate(acc, truncate_bits)
      // SDP BS stage: ALU(bias) → MUL → ReLU/ReLUx
      acc = apply_sdp_x_stage(acc, bs_cfg, bias[oc], bs_mul_cfg, bs_relux_cmp)
      // SDP BN stage: ALU → MUL → ReLU/ReLUx
      acc = apply_sdp_x_stage(acc, bn_cfg, bn_alu_cfg, bn_mul_cfg, bn_relux_cmp)
      // SDP EW stage: element-wise ALU(max/min/add) → CVT → ReLU/ReLUx
      if (!ew_bypass): acc = ew_alu(acc, ew_src[oc]) → ew_cvt → ew_relu
      // OUT_CVT: requantization (NVDLA 64-bit shift-right rounding)
      result = clamp((acc * scale) >> shift + offset, -128, 127)
      output[og][oy][ox][oc%16] = result
```

**Tensor layouts** (must match NPU hardware exactly):
- **Input**: x-major interleaved `[group][x][y][c16]` — `offset = g*W*H*16 + x*H*16 + y*16`
- **Output**: y-major interleaved `[group][y][x][c16]` — `offset = g*W*H*16 + y*W*16 + x*16`
- **Weights**: packed `[oc1][ic1][kx][ky][oc2][ic2]` with `WEIGHT_ATOMIC_SIZE=32`, `FEATURE_ATOMIC_SIZE=16`

**Weight reading**: 6-level nested index with `oc1=oc/32, oc2=oc%32, ic1=ic/ic_group, ic2=ic%ic_group` where `ic_group=32` (normal) or `64` (depthwise).

### IRQ Model

The kernel's Rocket driver IRQ flow:
1. Before job: writes `INTERRUPT_MASK = DPU_0 | DPU_1` (0x300) — **enables** these bits
2. Writes `INTERRUPT_CLEAR = DPU_0 | DPU_1` — clears pending
3. Writes `OPERATION_ENABLE = 1` — starts job
4. QEMU executes convolution, sets `raw_status |= 0x300`
5. QEMU computes `status = raw_status & mask` — **mask=1 means enabled** (not inverted!)
6. If status != 0, raises IRQ line
7. Kernel ISR reads `INTERRUPT_RAW_STATUS`, checks for DPU_0/DPU_1
8. Kernel thread writes `OPERATION_ENABLE = 0`, `INTERRUPT_CLEAR = 0x1ffff`
9. QEMU clears raw_status bits, lowers IRQ

Key finding: the interrupt mask polarity is `status = raw & mask` (1=enabled), NOT `raw & ~mask`.

### IOMMU Mailbox

The Rocket NPU driver allocates IOVAs for BOs via its own `drm_mm` allocator within an IOMMU paging domain. These IOVAs (0x0, 0x1000, 0x2000...) don't correspond to guest physical addresses. The QEMU NPU model needs to translate IOVAs to GPAs to read/write BO data.

Solution: a **mailbox MMIO region** at `0xfdaf0000` where the kernel IOMMU module writes IOVA→GPA mappings:

| Offset | Register | Description |
|--------|----------|-------------|
| 0x00 | IOVA | Write page-aligned IOVA (first) |
| 0x04 | PHYS | Write GPA (triggers mapping addition) |
| 0x08 | UNMAP | Write IOVA to remove mapping |
| 0x0c | COUNT | Read: number of active mappings |

The kernel module's `map_pages` callback:
```c
writel((u32)(iova + i * pgsize), mailbox + 0x00);
writel((u32)(paddr + i * pgsize), mailbox + 0x04);
```

The QEMU NPU model's DMA read/write:
```c
hwaddr gpa = npu_iova_to_gpa(s, iova);  // linear scan of mailbox table
address_space_read(s->dma_as, gpa, ...);
```

Multi-page reads are handled by translating each page boundary crossing.

## Kernel Helper Modules

### qemu_iommu.ko

A minimal IOMMU driver (`qemu,iommu-dummy` compatible) that:
- Registers with the kernel IOMMU subsystem (`.owner = THIS_MODULE` required for modules)
- Provides `domain_alloc_paging` returning domains with 32-bit aperture and `PAGE_SIZE` pgsize_bitmap
- Forwards `map_pages`/`unmap_pages` to the QEMU MMIO mailbox
- Uses `generic_single_device_group` for device grouping
- Identity domain with `IOMMU_DOMAIN_IDENTITY` type and ops

### qemu_reset.ko

A 40-line reset controller (`qemu,reset-dummy` compatible) with no-op `assert`/`deassert`/`status` ops. Each NPU core gets unique reset cell IDs (`i*2`, `i*2+1`) to avoid exclusive lock conflicts.

## Verified Test Results

### Boot (Phase 1)
```
Machine model: QEMU RK3588 (Orange Pi 5 Plus)
earlycon: ns16550a0 at MMIO32 0x00000000feb50000
psci: PSCIv1.1 detected in firmware.
GICv3: 192 SPIs implemented
GICv3: CPU0..3: found redistributor regions
arch_timer: cp15 timer running at 62.50MHz (virt)
smp: Brought up 1 node, 4 CPUs
```

### NPU Probe (Phase 2)
```
qemu-iommu fdaf0000.qemu-iommu: QEMU dummy IOMMU registered (mailbox=yes)
[drm] Initialized rocket 0.0.0 for rknn on minor 0
rocket fdad0000.npu: Rockchip NPU core 0 version: 65538
rocket fdac0000.npu: Rockchip NPU core 1 version: 65538
rocket fdab0000.npu: Rockchip NPU core 2 version: 65538
/dev/accel/accel0
```

### NPU Job Submission (Phase 3)
```
Opened /dev/accel/accel0 (fd=3)
Input  BO: handle=1 dma=0x0
Output BO: handle=2 dma=0x1000
Weight BO: handle=3 dma=0x2000
Bias   BO: handle=4 dma=0x3000
Regcmd BO: handle=5 dma=0x4000
Submitting NPU job...          → kernel DRM scheduler dispatches to core
conv 1x1x16 -> 1x1x16          → QEMU parses regcmd, executes convolution
Output: [0]=10, [2]=10, ...     → correct values from identity convolution
```

## Build Instructions

### QEMU

```bash
git clone --depth 1 --branch v10.2.0 https://gitlab.com/qemu-project/qemu.git qemu-src
cd qemu-src

# Copy source files
cp $PROJECT/qemu/hw/arm/rk3588.c hw/arm/
cp $PROJECT/qemu/hw/misc/rockchip-npu.{c,h} hw/misc/
cp $PROJECT/qemu/include/hw/arm/rk3588.h include/hw/arm/

# Patch build system
cat >> hw/arm/Kconfig < $PROJECT/qemu/hw/arm/Kconfig.rk3588
cat >> hw/misc/Kconfig < $PROJECT/qemu/hw/misc/Kconfig.rockchip_npu
# Add to hw/arm/meson.build before the final line:
#   arm_common_ss.add(when: ['CONFIG_ORANGEPI5PLUS', 'TARGET_AARCH64'], if_true: files('rk3588.c'))
# Add to hw/misc/meson.build:
#   system_ss.add(when: 'CONFIG_ROCKCHIP_NPU', if_true: files('rockchip-npu.c'))

mkdir build && cd build
../configure --target-list=aarch64-softmmu --disable-docs
make -j$(nproc)
```

### Kernel Modules (on aarch64 board or cross-compile)

```bash
# qemu_iommu.ko
echo 'obj-m := qemu_iommu.o' > Makefile
make -C /lib/modules/$(uname -r)/build M=$(pwd) modules

# qemu_reset.ko
echo 'obj-m := qemu_reset.o' > Makefile
make -C /lib/modules/$(uname -r)/build M=$(pwd) modules
```

### Initramfs

```bash
mkdir -p rootfs/{bin,sbin,dev,proc,sys,lib/modules/$(uname -r)/kernel/drivers/{accel/rocket,gpu/drm/scheduler}}
cp busybox rootfs/bin/ && cd rootfs/bin && for c in sh ls cat mount insmod sleep; do ln -s busybox $c; done
cp rocket.ko drm_shmem_helper.ko gpu-sched.ko qemu_iommu.ko qemu_reset.ko rootfs/lib/modules/.../
# Create init script that loads modules in order:
#   qemu_iommu → qemu_reset → sleep 1 → drm_shmem_helper → gpu-sched → rocket
cd rootfs && find . | cpio -o -H newc | gzip > ../initrd.gz
```

### Run

```bash
qemu-system-aarch64 -M orangepi5plus -smp 4 -m 1G \
  -kernel Image-6.18 -initrd initrd.gz \
  -append "console=ttyS0 earlycon rdinit=/init iommu.passthrough=1" \
  -nographic
```

## Design Decisions and Findings

### Why ns16550a, not snps,dw-apb-uart
The kernel's `earlycon` subsystem maps `uart8250,mmio32,<addr>` to the 8250 driver which matches `ns16550a`. The `snps,dw-apb-uart` compatible requires a different clock setup and the QEMU `serial_mm` device doesn't emulate the DW-specific registers. Using `ns16550a` gives working earlycon output from the first instruction.

### Why not use the RK3588 CRU for clocks/resets
The `rockchip,rk3588-cru` driver (CONFIG_CLK_RK3588=y, built-in) probes very early and reads PLL configuration registers to derive clock parents. With QEMU's `create_unimplemented_device` returning 0 for all reads, the PLL mux parent selection dereferences null pointers (`clk_mux_get_parent` crash). A fixed-clock + dummy reset module is the correct approach.

### Why IOMMU mailbox instead of SMMUv3
QEMU's ARM SMMUv3 model is tightly coupled to PCI (`primary-bus` property required). The Rocket NPU is a platform device. Writing a full Rockchip IOMMU model would be hundreds of lines for something that just needs identity-like mapping. The mailbox approach (4 MMIO registers, ~60 lines of QEMU code + ~30 lines of kernel module code) solves the problem with minimal complexity.

### Why iommu.passthrough=1
The dummy IOMMU module registers as a loadable module (late), not a builtin. Without `iommu.passthrough=1`, the kernel's IOMMU DMA default domain setup runs before the dummy IOMMU exists, causing `iommu_setup_default_domain` to fail with "IOMMU driver was not able to establish FW requested direct mapping". Passthrough mode skips this check.

### IRQ mask polarity
The NPU's `INTERRUPT_MASK` register uses mask=1 to **enable** interrupts (not mask them). `INTERRUPT_STATUS = INTERRUPT_RAW_STATUS & INTERRUPT_MASK`. The kernel sets `MASK = DPU_0 | DPU_1` before starting a job. This was initially implemented inverted (`& ~mask`) causing all jobs to time out.

### REGISTER_AMOUNTS encoding
The kernel encodes the regcmd entry count as `PC_DATA_AMOUNT = (regcmd_count + 1) / 2 - 1` where `regcmd_count` is the number of `uint64_t` entries. To recover: `entries = (PC_DATA_AMOUNT + 1) * 2`. This encoding is specific to the Rocket kernel driver and differs from a simple byte or word count.

## Known Limitations

1. **Weight packing**: The test program's identity weight matrix only produces correct values for even output channels due to WEIGHT_ATOMIC_SIZE=32 interleaving. Full models (MobileNetV1, YOLO) use Mesa's `rkt_fill_weights()` which handles this correctly.

2. **Per-channel quantization**: The convolution engine supports per-channel mode (RKNN-style BS bypass with `BS_CFG=0x13F`, bias from `BS_ALU_CFG` register, `BN_CFG=0x12` for ReLU). Not yet tested end-to-end in QEMU.

3. **BN per-channel DMA**: The BN stage supports scalar ALU/MUL operands from registers but not DMA-sourced per-channel operands via NRDMA. RKNN uses per-channel BN for ReLU — not needed for Mesa/librocketnpu workloads.

4. **No virtio-blk**: The machine doesn't add virtio devices. For rootfs, use initramfs or add virtio-mmio support.

5. **IOMMU as module**: Requires `iommu.passthrough=1` kernel cmdline. A built-in IOMMU driver would remove this requirement but needs kernel recompilation.

6. **Single-threaded**: NPU execution is synchronous in the MMIO write handler. Multi-core parallelism (kernel distributing jobs across 3 cores) works but executes sequentially in QEMU.
