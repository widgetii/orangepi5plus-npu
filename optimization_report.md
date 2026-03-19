# Rocket NPU Driver Optimization Report

## Hardware
- Board: Orange Pi 5 Plus (RK3588)
- NPU: 3-core Rockchip RKNN (CNA/DPU architecture, NVDLA-derived)
- Kernel: Linux 6.18.10-current-rockchip64 (mainline)
- Mesa: 26.1.0-devel (custom build with optimizations)

## Implemented Optimizations (Mesa Patches)

### Phase 1: Buffer Pooling
Pool GEM BOs in `rkt_screen` to avoid CREATE_BO/GEM_CLOSE per invocation.
- Best-fit allocation within 2x size, pool cap 256 entries
- Files: `rkt_device.h`, `rkt_device.c`

### Phase 2: Cache Sync Reduction
Skip PREP_BO/FINI_BO for write-once BOs (weights, biases, regcmds).
- `device_resident` flag on `rkt_resource`
- `persistent_map` keeps mmap alive across invocations
- Files: `rkt_device.h`, `rkt_device.c`, `rkt_ml.c`

### Phase 3: Batched Submission + Cached Submit
Consolidate tasks per operation into single jobs, pre-allocate submit structures.
- Eliminates malloc/free per invoke for job/task arrays
- Files: `rkt_ml.h`, `rkt_ml.c`

## Results

### IOCTL Reduction (per inference, steady-state)
| IOCTL | Before | After |
|-------|--------|-------|
| CREATE_BO | 171 | 0 |
| PREP_BO | 229 | 4 |
| FINI_BO | 86 | 2 |
| SUBMIT | 2 | 2 |
| GEM_CLOSE | 114 | 0 |
| **Total** | **757** | **8** |

### Latency (CPU governor=performance, 100 runs)
| Model | System Mesa | Custom Mesa | CPU | Speedup |
|-------|------------|-------------|-----|---------|
| MobileNetV1 224 INT8 | 12.51ms | **11.12ms** | 68.0ms | 6.1x |
| MobileNetV2 224 INT8 | - | **13.36ms** | 39.8ms | 3.0x |
| SSD MobileNetV1 INT8 | - | **22.07ms** | 89.5ms | 4.1x |

### Correctness
Custom Mesa produces **bit-exact identical** output to system Mesa (max_diff=0).
NPU vs CPU difference (max 7-61 quant steps) is inherent to NPU hardware quantization.

## Architecture Analysis

### Why RKNN is Still 4.3x Faster (2.6ms vs 11.1ms)
1. **Single-submit compiled graph**: RKNN precompiles the entire model into a single command stream. Rocket rebuilds commands each invocation.
2. **No data format conversion**: RKNN stores tensors in HW-native format. Rocket converts NHWC uint8 <-> interleaved int8 per invocation (~1-2ms CPU overhead).
3. **Multi-core utilization**: RKNN uses all 3 cores simultaneously. Rocket uses 1 core per operation.
4. **IOMMU overhead**: Kernel attaches/detaches IOMMU per job (~0.3ms per attach on ARM64).

### Remaining Optimization Paths

**Kernel-side (requires module rebuild):**
- Skip IOMMU attach/detach when consecutive jobs share same domain
- Batch pm_runtime get/put per submit instead of per job
- Estimated: 15-25% improvement

**Mesa-side:**
- Pre-compute and cache input/output data conversion in subgraph_create
- Support AVGPOOL/RESHAPE as software ops (needs HW<->SW tensor format conversion)
- Estimated: 5-10% improvement

**Stretch goals:**
- SRAM utilization (intermediate activations in on-chip SRAM)
- Multi-core parallelism (split output channels across cores)
- Estimated: 1.5-2.5x throughput improvement

## Known Issues
- EfficientNet-Lite0 crashes Rocket driver (unsupported ops trigger kernel panic)
- NPU job timeout corrupts IOMMU state, can cascade to kernel memory corruption
- Boot args `panic=10 panic_on_oops=1` required for auto-recovery

## Files
- Mesa patch: `patches/0001-rocket-buffer-pool-cache-sync-batched-submit.patch`
- Benchmark suite: `benchmark/scripts/`
- Baseline results: `benchmark/results/baseline/performance.json`
