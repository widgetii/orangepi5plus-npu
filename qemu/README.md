# QEMU RK3588 with Rocket NPU

QEMU machine type for the Orange Pi 5 Plus (RK3588) with functional Rocket NPU emulation.

## Building

Apply these files to a QEMU source tree:

```bash
git clone https://gitlab.com/qemu-project/qemu.git
cd qemu
git checkout v9.2.0  # or latest stable

# Copy source files
cp $OPIS/qemu/hw/arm/rk3588.c hw/arm/
cp $OPIS/qemu/hw/misc/rockchip-npu.c hw/misc/
cp $OPIS/qemu/hw/misc/rockchip-npu.h hw/misc/
cp $OPIS/qemu/include/hw/arm/rk3588.h include/hw/arm/
cp $OPIS/qemu/tests/qtest/rockchip-npu-test.c tests/qtest/

# Patch build system (add lines from .rk3588 / .rockchip_npu files)
# hw/arm/Kconfig: add ORANGEPI5PLUS block
# hw/arm/meson.build: add rk3588.c
# hw/misc/Kconfig: add ROCKCHIP_NPU block
# hw/misc/meson.build: add rockchip-npu.c

# Build
mkdir build && cd build
../configure --target-list=aarch64-softmmu
make -j$(nproc)
```

## Running

### Boot to shell
```bash
qemu-system-aarch64 -M orangepi5plus -smp 4 -m 4G \
  -kernel Image -append "console=ttyS0 root=/dev/vda rw" \
  -drive file=rootfs.ext4,format=raw,if=virtio \
  -nographic
```

### NPU CI test
```bash
timeout 120 qemu-system-aarch64 -M orangepi5plus -smp 4 -m 4G \
  -kernel Image -append "console=ttyS0 root=/dev/vda rw" \
  -drive file=ci-rootfs.qcow2,format=qcow2,if=virtio \
  -nographic -monitor none
```

Inside guest:
```bash
dmesg | grep -c "rocket.*core"  # Should print "3"
ls /dev/accel/accel0             # NPU device node
python3 bench_teflon.py mobilenet_v1_int8.tflite --verify golden.bin
```

## Architecture

### Phase 1: Bootable Machine (rk3588.c)
- 4x Cortex-A55 CPUs (homogeneous, no big.LITTLE)
- GICv3 interrupt controller
- UART2 (16550A) for serial console
- CRU/PMU stub devices (return 0, prevent kernel panics)
- Programmatic DTB generation (or accept external -dtb)

### Phase 2: NPU Stub (rockchip-npu.c MMIO)
- 3 NPU cores at 0xfdab0000, 0xfdac0000, 0xfdad0000
- PC_VERSION returns valid probe value
- IRQ on SPI 110/111/112
- All register writes stored and readable

### Phase 3: Functional NPU (rockchip-npu.c convolution engine)
- Parses regcmd buffers (packed 64-bit register write commands)
- Executes INT8 convolution with correct tensor layouts
- Supports standard convolution and depthwise convolution
- Supports per-channel quantization (RKNN-style BS bypass)
- Multi-task chaining via regcmd chain pointers
- Synchronous execution (deterministic, no threading)

### Tensor Layouts
- **Input**: x-major interleaved `[group][x][y][c16]` — NPU_OFFSET(g,x,y,w,h)
- **Output**: y-major interleaved `[group][y][x][c16]` — different from input!
- **Weights**: packed `[oc1][ic1][kx][ky][oc2][ic2]` with WEIGHT_ATOMIC_SIZE=32

### Regcmd Format
Each entry is `uint64_t`:
- bits [15:0] = register offset (0x1xxx=CNA, 0x3xxx=Core, 0x4xxx=DPU, etc.)
- bits [47:16] = register value
- bits [63:48] = target block selector

## Files

| File | Lines | Description |
|------|-------|-------------|
| `hw/arm/rk3588.c` | ~260 | Machine type, DTB gen, device wiring |
| `hw/misc/rockchip-npu.c` | ~700 | NPU: MMIO, regcmd parser, INT8 conv engine |
| `hw/misc/rockchip-npu.h` | ~200 | Register defs, task struct, offset macros |
| `include/hw/arm/rk3588.h` | ~55 | SoC state, memory map constants |
| `tests/qtest/rockchip-npu-test.c` | ~100 | QTest unit tests |
