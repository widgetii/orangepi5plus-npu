# RK3588 NPU for Custom Heavy-Math Operations

## Research Question

Can the RK3588 NPU be repurposed beyond ML inference for custom heavy-math operations — specifically bicubic and lanczos image upscaling/downscaling?

## NPU Hardware Architecture

The RK3588 NPU is not just a convolution engine. Each of the three cores contains three pipelined processing units:

| Unit | Purpose | Operations |
|------|---------|------------|
| **CNA** (Convolution Neural Accelerator) | MAC engine | Convolution (direct, dilated, depthwise), matrix multiply. 1024 INT8 or 512 FP16 MACs/cycle |
| **DPU** (Data Processing Unit) | Post-processing | Element-wise add/mul, ReLU, PReLU, Sigmoid, batch norm |
| **PPU** (Planar Processing Unit) | Pooling | Max/min/average pooling |

Data flows through a fixed pipeline: `Input → CNA → DPU → PPU → Output`. Each unit can be bypassed but the pipeline itself cannot be rearranged or extended.

The architecture derives from NVIDIA's NVDLA (Deep Learning Accelerator) design, adapted by Rockchip. Like NVDLA, the NPU is register-programmed — a frontend DMA engine (the "PC" unit) writes pre-computed register values to the CNA/DPU/PPU hardware. There is no instruction set architecture (ISA) and no ability to define custom compute kernels.

**Sources:** Rocket kernel driver (`drivers/accel/rocket/`), [jas-hacks.blogspot.com](http://jas-hacks.blogspot.com/2024/02/rk3588-reverse-engineering-rknn.html), [mtx512/rk3588-npu](https://github.com/mtx512/rk3588-npu), [NVDLA hardware architecture](https://nvdla.org/hw/v1/hwarch.html)

## Fundamental Limitation: No Custom Programmability

The NPU cannot run arbitrary custom operations. Key constraints:

- **No ISA** — register-programmed via DMA'd command buffers, not instruction-programmable
- **Fixed data path** — CNA→DPU→PPU pipeline with no branching or looping
- **32KB L1 SRAM scratchpad** — operations exceeding this cause hardware failures
- **No public register documentation** — Rockchip TRM Chapter 36 lists registers but lacks practical programming guidance
- **High per-inference overhead** — 634 IOCTLs, 10 submits, 140 jobs, 210 HW submits per inference via Rocket

Reverse engineers investigating custom compute on this NPU explicitly concluded it is not feasible and recommended OpenCL on the Mali GPU instead ([jas-hacks blog](http://jas-hacks.blogspot.com/2024/02/rk3588-reverse-engineering-rknn.html)).

## Can Bicubic/Lanczos Be Expressed as Convolution?

**Mathematically, yes.** Both bicubic and lanczos resampling compute weighted sums of neighboring pixels — the same fundamental operation as spatial convolution.

| Algorithm | Kernel Size | As Convolution? |
|-----------|------------|-----------------|
| Bicubic | 4x4 | Yes — outer product of 1D cubic polynomials |
| Lanczos2 | 4x4 | Yes — windowed sinc function |
| Lanczos3 | 6x6 | Yes — larger windowed sinc |

**Downscaling** maps naturally to strided convolution with fixed weights — a standard NPU operation.

**Upscaling** has critical problems on this hardware:

1. **Per-pixel varying kernels**: Upscaling requires evaluating the interpolation kernel at sub-pixel offsets. Each output pixel needs a *different* set of weights depending on its fractional position. A standard convolution applies the *same* kernel everywhere — this doesn't match the upscaling requirement.

2. **No transposed convolution in Rocket**: Upscaling via learned methods typically uses transposed (deconvolution) or sub-pixel convolution. The open-source Rocket/Teflon stack doesn't currently support these operations.

3. **Input layout constraints**: The NPU expects NC1HWC2 tensor format with strict alignment. Classical image data would need reshaping.

4. **IOCTL overhead**: For a single fixed convolution, the 634-IOCTL overhead of Rocket makes it slower than CPU NEON for simple kernels.

**Sources:** [Wikipedia: Bicubic interpolation](https://en.wikipedia.org/wiki/Bicubic_interpolation), [Wikipedia: Lanczos resampling](https://en.wikipedia.org/wiki/Lanczos_resampling), [Mesa Teflon docs](https://docs.mesa3d.org/teflon.html)

## NPU Feasibility Assessment

| Scenario | Feasibility | Notes |
|----------|-------------|-------|
| Downscaling (bicubic) | Partially feasible | Strided conv with fixed weights works, but IOCTL overhead may negate gains vs CPU NEON |
| Upscaling (bicubic) | Not feasible | Requires per-pixel varying kernels; no transposed conv support |
| Lanczos (any direction) | Not feasible | Same per-pixel kernel issue for upscaling; downscaling theoretically possible but impractical |
| Arbitrary custom math | Not feasible | No ISA, fixed-function pipeline |

## Practical Alternatives on the Orange Pi 5 Plus

### RGA Hardware Scaling (Best for Bicubic)

The RK3588 includes dedicated Rockchip Graphics Accelerator cores — 1x RGA2-Enhance and 2x RGA3 — purpose-built for image operations including scaling.

| Feature | RGA2-Enhance | RGA3 (x2 cores) |
|---------|-------------|------------------|
| Upscale interpolation | Bilinear + Bicubic | Bicubic only |
| Downscale method | Bilinear + Average | Average only |
| Scale range | 1/16x to 16x | 1/8x to 8x |
| Max input resolution | 8192x8192 | 8176x8176 |
| CSC precision | 8-bit | 10-bit |
| Lanczos support | No | No |

Access via `librga` (im2d API). Three total cores available for throughput-parallel scaling.

### Mali G610 GPU via OpenCL (Best for Lanczos)

The Mali G610 MP4 GPU supports OpenCL and can run arbitrary compute shaders. This is the **only hardware-accelerated option for lanczos** on this board. Custom OpenCL kernels can implement any interpolation algorithm with full control over per-pixel weight computation.

### ML Super-Resolution on NPU (Best Perceptual Quality)

Lightweight super-resolution models like ESPCN (3-4 conv layers, CVPR 2016) and FSRCNN run well on the NPU and produce better perceptual quality than classical algorithms. These models use standard convolution layers that map directly to the CNA hardware, followed by a sub-pixel shuffle layer (which can be done on CPU with negligible overhead).

From our benchmarks, MobileNetV1 runs at 81 FPS on a single NPU core via Rocket — lightweight super-resolution models with fewer parameters would achieve similar or better throughput, enabling real-time HD upscaling.

**Source:** [ESPCN paper (CVPR 2016)](https://arxiv.org/abs/1609.05158)

### Comparison Summary

| Approach | Hardware | Performance | Quality | Lanczos? |
|----------|----------|-------------|---------|----------|
| OpenCV resize | CPU (NEON) | Good for single images | Deterministic | Yes |
| RGA2/RGA3 | Dedicated HW | Hardware-accelerated, 3 cores | Bicubic/bilinear | No |
| OpenCL kernel | Mali GPU | Good parallelism | Any algorithm | Yes |
| ESPCN/FSRCNN | NPU | Real-time HD possible | Better than classical | N/A (learned) |

## Conclusions

| Question | Answer |
|----------|--------|
| Can the NPU do custom math? | **No** — fixed-function pipeline, no ISA |
| Can bicubic/lanczos be expressed as convolution? | **Yes mathematically**, but upscaling requires per-pixel varying kernels |
| Is downscaling feasible on NPU? | **Partially** — strided convolution works, but IOCTL overhead may negate gains |
| Best approach for bicubic scaling? | **RGA2/RGA3** — dedicated HW with bicubic support, 3 cores available |
| Best approach for lanczos? | **Mali GPU (OpenCL)** — only HW-accelerated option; or CPU NEON |
| Is there value in NPU for non-ML? | **Very limited** — the hardware is purpose-built for CNN inference |

The RK3588 NPU is architecturally unsuitable for custom heavy-math operations. For image scaling on the Orange Pi 5 Plus, the RGA hardware (bicubic) and Mali GPU (lanczos/custom) are the correct tools. The NPU's value for image quality improvement lies in running ML super-resolution models, which paradoxically produce better results than the classical algorithms we were trying to implement.
