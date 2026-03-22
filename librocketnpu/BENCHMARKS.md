# librocketnpu Performance Benchmarks

## MobileNetV1 INT8 (224×224×3, 1001 classes)

### Test Conditions
- Board: Orange Pi 5 Plus (RK3588, 16GB RAM)
- CPU/NPU governor: `performance` (fixed max frequency)
- Iterations: 200 (after 1 warmup invoke)
- Input: constant uint8 value 128 (zero-point)
- Measurement: wall-clock time per `rnpu_invoke()` / `rknn_run()` call,
  includes input conversion, all 28 HW conv ops, 3 SW ops (avg_pool, reshape,
  softmax), and output conversion
- Single-core NPU (core_mask=0, auto-selected by kernel)

### Results

| Variant                   | Kernel          | Driver | Avg (ms) | Min (ms) |
|---------------------------|-----------------|--------|----------|----------|
| RKNN runtime (librknnrt)  | 6.1 vendor      | RKNPU  | **2.18** | **1.98** |
| librocketnpu              | 6.1 vendor      | RKNPU  | 6.01     | 5.49     |
| Custom Mesa (round 3 opt) | 6.18 mainline   | Rocket | 9.77     | 9.22     |
| librocketnpu              | 6.18 mainline   | Rocket | 11.16    | 10.53    |
| System Mesa 26.0.2        | 6.18 mainline   | Rocket | 11.18    | 10.68    |

### Analysis

**RKNPU vendor driver is ~1.9× faster than Rocket mainline** for the same
librocketnpu code (6.01 vs 11.16 ms). This is likely due to:
- Higher default NPU clock on vendor kernel (1 GHz vs potentially lower on mainline)
- Vendor kernel's optimized IOMMU and power domain management
- Reduced per-submit overhead in the RKNPU DRM driver vs Rocket's DRM scheduler

**RKNN runtime is 2.8× faster than librocketnpu** on the same vendor kernel
(2.18 vs 6.01 ms). RKNN's advantages:
- Multi-core NPU dispatch (3 cores in parallel) — librocketnpu uses single core
- Per-channel weight decomposition with optimal OUT_CVT scaling per channel
- Batched multi-task submit (591 tasks in one ioctl vs 1 per operation)
- Weight data pre-arranged in hardware-native format (no runtime conversion)

### Optimization Opportunities for librocketnpu

To close the 2.8× gap with RKNN:

1. **Multi-core dispatch** (~2-3× speedup potential): Submit jobs to all 3 NPU
   cores in parallel. The RKNPU kernel auto-schedules with `core_mask=0`, but
   we currently submit sequentially. Pipelining ops across cores would nearly
   triple throughput.

2. **Batched task submit** (~10-20% speedup): Submit all 28 conv operations as
   a single ioctl with 28+ tasks instead of 28 separate ioctls. Reduces
   kernel entry/exit and IOMMU domain refcount overhead.

3. **Pre-built task BO** (~5% speedup): Allocate one task BO at model load
   time and reuse across invocations, instead of creating/destroying per submit.

4. **Per-channel regcmd decomposition** (accuracy improvement): Split each
   conv into per-output-channel tasks with individual OUT_CVT scales, matching
   RKNN's approach. This would also fix the max_diff=18 vs Rocket golden.
