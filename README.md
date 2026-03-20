# RK3588 NPU Research — Orange Pi 5 Plus

Research into the RK3588's 6 TOPS Neural Processing Unit, comparing the proprietary
RKNN stack against the open-source Rocket driver (mainline Linux + Mesa).

## Board Setup

| | |
|---|---|
| Board | Orange Pi 5 Plus (RK3588, 16GB LPDDR4X, 233GB eMMC) |
| OS | Armbian 25.11.1 Noble (Ubuntu 24.04) |
| IP | `10.216.128.51` (DHCP) |
| SSH | `ssh root@10.216.128.51` (password: `orangepi`, key auth configured) |
| Serial | UART2 at 1500000 baud (read-only, 0xfeb50000) |

### Installed Kernels

| Kernel | Branch | NPU Driver | Use Case |
|--------|--------|-----------|----------|
| 6.18.10-current-rockchip64 | mainline | Rocket (open-source) | Reverse engineering, Teflon/TFLite |
| 6.1.115-vendor-rk35xx | vendor | RKNPU 0.9.8 (proprietary) | RKNN benchmarks, production perf |

### NPU Software Stack

| Component | Path / Package |
|-----------|---------------|
| librknnrt 2.3.2 | `/usr/lib/librknnrt.so` |
| librkllmrt | `/usr/lib/librkllmrt.so` |
| Mesa Teflon delegate 26.0.2 | `/usr/lib/teflon/libteflon.so` (Kisak PPA) |
| rknn-toolkit-lite2 2.3.2 | `/root/npu-research/venv` (Python 3.12) |
| ai-edge-litert 2.1.3 | `/root/npu-research/venv` |
| rknn-toolkit2 repo | `/root/npu-research/rknn-toolkit2` |
| rknn-llm repo | `/root/npu-research/rknn-llm` |
| rknn_benchmark (native) | `.../rknn_benchmark/build/rknn_benchmark` |

## Switching Kernels

U-Boot cannot follow symlinks — you must copy the actual files to `/boot/Image`,
`/boot/uInitrd`, and `/boot/dtb/`.

### Switch to mainline (Rocket NPU)

```bash
ssh root@10.216.128.51
rm /boot/Image /boot/uInitrd && rm -rf /boot/dtb
cp /boot/vmlinuz-6.18.10-current-rockchip64 /boot/Image
cp /boot/uInitrd-6.18.10-current-rockchip64 /boot/uInitrd
cp -a /boot/dtb-6.18.10-current-rockchip64 /boot/dtb
sync && reboot
```

### Switch to vendor (RKNN proprietary)

```bash
ssh root@10.216.128.51
rm /boot/Image /boot/uInitrd && rm -rf /boot/dtb
cp /boot/vmlinuz-6.1.115-vendor-rk35xx /boot/Image
cp /boot/uInitrd-6.1.115-vendor-rk35xx /boot/uInitrd
cp -a /boot/dtb-6.1.115-vendor-rk35xx /boot/dtb
sync && reboot
```

**Note:** Do NOT insert an SD card — U-Boot prefers SD over eMMC and will boot
the wrong image.

## Running Inference

### Rocket / Teflon (mainline kernel)

```bash
cd /root/npu-research/zero2pro_NPU_example
source /root/npu-research/venv/bin/activate

# CPU only
python3 classification.py -i grace_hopper.bmp \
  -m mobilenet_v1_1.0_224_quant.tflite \
  -l labels_mobilenet_quant_v1_224.txt

# NPU via Teflon delegate
python3 classification.py -i grace_hopper.bmp \
  -m mobilenet_v1_1.0_224_quant.tflite \
  -l labels_mobilenet_quant_v1_224.txt \
  -e /usr/lib/teflon/libteflon.so
```

### RKNN (vendor kernel)

```bash
cd /root/npu-research/rknn-toolkit2/rknn-toolkit-lite2/examples/resnet18
source /root/npu-research/venv/bin/activate
python3 test.py
```

### C Benchmark (vendor kernel)

```bash
BENCH=/root/npu-research/rknn-toolkit2/rknpu2/examples/rknn_benchmark/build/rknn_benchmark
MODEL=/root/npu-research/rknn-toolkit2/rknpu2/examples/rknn_mobilenet_demo/model/RK3588/mobilenet_v1.rknn

# Single core
$BENCH $MODEL "" 100 1

# All 3 cores
$BENCH $MODEL "" 100 7
```

## Benchmark Results

### Proprietary RKNN (vendor kernel 6.1, driver 0.9.8)

| Model | 1 Core | 3 Cores |
|-------|--------|---------|
| MobileNetV1 224x224 | 2.6ms / 384 FPS | 1.6ms / 622 FPS |
| YOLOv5s 640x640 | 16.7ms / 60 FPS | 9.5ms / 106 FPS |

### Open-source Rocket (mainline 6.18, Mesa 26.1.0-devel, single core)

| Model | CPU-only | Rocket NPU (optimized) |
|-------|----------|----------------------|
| MobileNetV1 224x224 | 68ms | **10.2ms** |
| SSD MobileNetV1 | 89ms | **19.8ms** |
| YOLOv5s-relu 640x640 | 142ms | NPU timeout (unsupported conv configs) |
| YOLOv8n 640x640 | 86ms | NPU timeout (unsupported conv configs) |

### Vendor vs Open-source (same YOLOv5s-relu model, single core)

| | RKNN (vendor) | Rocket (open-source) | Gap |
|--|------|--------|-----|
| MobileNetV1 224 | 2.6ms | 10.2ms | 3.9x |
| YOLOv5s 640 | 16.7ms | **Not functional** | — |

YOLO models need 13+ ops beyond CONV_2D/ADD. Patch 0004 adds 5 software ops
(CONCATENATION, MAX_POOL_2D, PAD, RESIZE_NEAREST, LOGISTIC) that run on the CPU,
eliminating format conversion at graph splits. All test models produce bit-exact output.
Patch 0005 fixes an upstream INT8 regression in Mesa git HEAD.

The Teflon delegate per-axis quantization assertion crash is fixed in patch 0003.

## NPU Hardware Architecture

The RK3588 NPU has **3 independent cores**, each with:

| Offset | Unit | Function |
|--------|------|----------|
| +0x0000 | PC (Frontend) | DMA engine that writes register values to CNA |
| +0x1000 | CNA | Convolution Neural Accelerator (MAC operations) |
| +0x3000 | Core | Power, clock, interrupt control |

Base addresses: Core 0 = `0xfdab0000`, Core 1 = `0xfdac0000`, Core 2 = `0xfdad0000`

The NPU is **register-programmed** — there is no instruction set. User-space builds
command buffers containing (register, value) pairs. The frontend (PC) unit reads these
via DMA and writes them to the CNA, configuring each convolution operation.

## Useful Commands

```bash
# NPU load (vendor kernel)
cat /sys/kernel/debug/rknpu/load

# Rocket driver status (mainline kernel)
dmesg | grep rocket
ls /dev/accel/accel0

# Trace IOCTLs during inference
strace -e trace=ioctl -e raw=ioctl -f -o /tmp/trace.log <command>

# Ftrace rocket driver functions
echo "function_graph" > /sys/kernel/debug/tracing/current_tracer
echo "rocket_*" > /sys/kernel/debug/tracing/set_ftrace_filter
echo 1 > /sys/kernel/debug/tracing/tracing_on
# ... run inference ...
echo 0 > /sys/kernel/debug/tracing/tracing_on
cat /sys/kernel/debug/tracing/trace
```

## Known Issues

- **Upstream Mesa INT8 regression (FIXED by patch 0005)** — Mesa git HEAD per-task job
  splitting breaks INT8 models. Fixed by batching all tasks per operation.

- **Vendor DTB has UART2 disabled** — the original Armbian vendor DTB ships with
  `serial@feb50000` set to `status = "disabled"`. The DTB on this board has been
  patched (`status = "okay"` + `chosen/stdout-path`). Mainline DTB is fine.

- **devmem2 bus error on mainline** — `CONFIG_STRICT_DEVMEM=y` prevents direct MMIO
  access to regions claimed by the Rocket driver. Use ftrace instead.

- **Armbian mirror flaky** — `fi.mirror.armbian.de` sometimes times out. The source
  is at `/etc/apt/sources.list.d/armbian.sources` (URI: `http://apt.armbian.com`).

## Research Documents

- [`optimization_report.md`](optimization_report.md) — Detailed analysis of all Rocket driver optimizations (12% latency reduction) and SW ML ops
- [`patches/README.md`](patches/README.md) — How to apply the optimization patches (0003 perf + 0004 ML ops)
- [`rocket_ioctl_analysis.md`](rocket_ioctl_analysis.md) — Decoded IOCTL protocols for both Rocket and RKNPU drivers
- [`npu_research_report.md`](npu_research_report.md) — Full research report: architecture, ftrace analysis, driver comparison

## Future Research Ideas

- Decode RKNPU_SUBMIT struct contents to understand the pre-compiled execution plan
- Implement multi-core support in Rocket (hardware supports it, driver doesn't yet)
- Profile cache sync overhead with perf to quantify its share of the performance gap
- Test RKNN-LLM with quantized language models on the NPU
- Compare power consumption between single-core and 3-core NPU operation
- Trace CNA register values to reverse-engineer convolution configuration parameters
