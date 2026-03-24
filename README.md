# RK3588 NPU Research — Orange Pi 5 Plus

[![CI](https://github.com/widgetii/orangepi5plus-npu/actions/workflows/ci.yml/badge.svg)](https://github.com/widgetii/orangepi5plus-npu/actions/workflows/ci.yml)

Open-source reverse engineering of the Rockchip RK3588 Neural Processing Unit.
This project includes a standalone NPU driver (`librocketnpu`), Mesa Gallium
optimization patches, a QEMU emulator for the NPU hardware, and research into
the proprietary RKNN register programming model.

## Project Components

### librocketnpu — Standalone Open-Source NPU Driver

A zero-dependency C library that drives the RK3588 NPU directly via DRM IOCTLs,
without Mesa or the proprietary RKNN stack.

```
librocketnpu/
  src/
    rnpu_tflite.c    # TFLite FlatBuffer parser (zero deps)
    rnpu_onnx.c      # ONNX protobuf parser (protobuf-c)
    rnpu_model.c     # Graph analysis, per-channel grouping, scheduling
    rnpu_task.c      # CBUF bank allocation, spatial tiling
    rnpu_coefs.c     # Weight/bias quantization formatting
    rnpu_regcmd.c    # Hardware register command generation
    rnpu_drm.c       # DRM IOCTL submission (CREATE_BO, SUBMIT)
    rnpu_rknn.c      # RKNN binary parser (BRDMA extraction)
    rnpu_sw_ops.c    # CPU fallback: concat, maxpool, pad, resize, sigmoid, softmax
    rnpu_convert.c   # NPU tensor format conversion (NHWC <-> NCHW interleaved)
  include/
    rocketnpu.h      # Public C API
  tests/
    test_sw_ops.c    # 27 unit tests (CPU-only, no hardware)
    test_rknpu_abi.c # 29 ABI regression tests (CPU-only)
    test_onnx_parse.c # ONNX parser validation
  docs/
    compiler_architecture.md  # Research: open-source NPU compiler design
```

**Features:**
- Loads TFLite INT8 models directly — no conversion step
- ONNX frontend (protobuf-c) for RKNN-toolkit graph consumption
- Dual-driver: supports both Rocket (mainline) and RKNPU (vendor) kernel drivers
- Per-channel quantization via scale-sorted grouping + BRDMA DMA
- Multi-task batching for cross-operation chaining
- 56 unit tests, CI on GitHub Actions

**Build:**
```bash
# On board (aarch64, Armbian)
apt install libdrm-dev
cd librocketnpu && make

# With ONNX support
apt install libprotobuf-c-dev
make test_onnx_parse

# Run tests (no hardware needed)
make test
```

### Mesa Optimization Patches

Performance patches for the upstream Rocket Gallium driver:

| Patch | Description | Impact |
|-------|-------------|--------|
| `0003` | BO pool, cache sync reduction, NEON I/O conversion, cached submit | 12% avg latency reduction |
| `0004` | SW ops: concat, maxpool, pad, resize, logistic | YOLO mixed HW/SW execution |
| `0005` | Fix INT8 regression: batch tasks per operation | Correctness fix for upstream |

### QEMU NPU Emulator

Full-system emulation of the RK3588 NPU for development without hardware:
- Boots unmodified Armbian disk images
- Emulates CRU (Clock Reset Unit) for kernel driver probe
- NPU MMIO register model (PC, CNA, Core units)
- IOMMU stub for DMA address translation

## Benchmark Results

### MobileNetV1 224x224 INT8 (single core)

| Stack | Latency | Status |
|-------|---------|--------|
| RKNN proprietary (vendor kernel) | **2.6ms** | Bit-exact |
| librocketnpu (vendor kernel, BRDMA) | **10.2ms** | Bit-exact (max_diff=0) |
| Mesa Rocket + patches (mainline kernel) | **10.2ms** | Bit-exact |
| Mesa Rocket stock (mainline kernel) | 11.6ms | Bit-exact |

### YOLOv5s-relu 640x640 INT8

| Stack | Latency | Accuracy vs RKNN golden |
|-------|---------|------------------------|
| RKNN proprietary (3 cores) | **9.5ms** | Reference |
| RKNN simulator (x86, ONNX Runtime) | N/A | ~0.2 mean diff |
| librocketnpu (vendor, per-channel groups) | **292ms** | ~18-25 mean diff |

The accuracy gap is due to per-channel quantization hardware limitations — the
NVDLA-derived CNA applies one requantization scale per operation, while YOLO's
per-axis weights need per-channel scaling. librocketnpu approximates this with
scale-sorted channel grouping and BRDMA MUL correction.

## NPU Hardware Architecture

The RK3588 NPU has **3 independent cores** (6 TOPS total), each with:

| Offset | Unit | Function |
|--------|------|----------|
| +0x0000 | PC (Frontend) | DMA engine: reads register command buffers, writes to CNA |
| +0x1000 | CNA | Convolution Neural Accelerator (INT8 MAC array) |
| +0x2000 | DPU + RDMA | Data Processing: bias, batch norm, element-wise, output quantization |
| +0x3000 | Core | Power, clock, interrupt control |

The NPU is **register-programmed** — no instruction set. Each "instruction" is a
`(register_address, value)` pair packed into 64-bit entries, DMA'd from memory by
the PC unit. A typical convolution requires ~130 register writes across CNA, DPU,
and RDMA units.

## Research Documents

| Document | Description |
|----------|-------------|
| [`librocketnpu/docs/compiler_architecture.md`](librocketnpu/docs/compiler_architecture.md) | Open-source NPU compiler design (ONNC, TVM, MLIR comparison) |
| [`optimization_report.md`](optimization_report.md) | Mesa Rocket driver optimizations (12% latency reduction) |
| [`rocket_ioctl_analysis.md`](rocket_ioctl_analysis.md) | Decoded IOCTL protocols for Rocket and RKNPU drivers |
| [`npu_research_report.md`](npu_research_report.md) | Full research report: architecture, ftrace, driver comparison |
| [`per_axis_quantization_research.md`](per_axis_quantization_research.md) | Per-channel quantization hardware investigation |

## Board Setup

| | |
|---|---|
| Board | Orange Pi 5 Plus (RK3588, 16GB LPDDR4X, 233GB eMMC) |
| OS | Armbian 25.11.1 Noble (Ubuntu 24.04) |
| Kernels | 6.18.10-current-rockchip64 (mainline) / 6.1.115-vendor-rk35xx (vendor) |

## License

librocketnpu: MIT. Research documents and patches: as noted per file.
