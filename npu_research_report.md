# RK3588 NPU Deep Research Report

## 1. IOCTL Protocol Analysis (Rocket Driver)

### IOCTL Commands
| Hex | Name | Count | Direction | Size | Purpose |
|-----|------|-------|-----------|------|---------|
| 0xc0186440 | ROCKET_CREATE_BO | 171 | RW | 24B | Allocate GPU buffer object |
| 0x40106442 | ROCKET_PREP_BO | 245 | W | 16B | Prepare BO for CPU access (cache sync) |
| 0x40086443 | ROCKET_FINI_BO | 94 | W | 8B | Finalize BO after CPU writes |
| 0x40186441 | ROCKET_SUBMIT | 10 | W | 24B | Submit jobs to NPU |
| 0x40086409 | GEM_CLOSE | 114 | W | 8B | Free buffer objects |

### Inference Flow (MobileNetV1, single run)
Total: 634 IOCTLs, 10 SUBMIT calls

1. **Setup phase**: 171x CREATE_BO (weights, activations, register cmd buffers)
2. **Data upload**: 246x PREP_BO + 94x FINI_BO (CPU writes data to BOs)
3. **Execution**: 10x SUBMIT → 140 jobs → 210 HW submits (tasks within jobs)
4. **Cleanup**: 114x GEM_CLOSE

## 2. Kernel Driver Execution Flow (ftrace)

### Function Call Counts (single inference)
```
311 rocket_iommu_domain_get/put    — IOMMU domain management
280 rocket_acquire_object_fences   — Dependency tracking
246 rocket_ioctl_prep_bo           — Cache sync for CPU access
210 rocket_job_irq_handler         — NPU completion interrupts
210 rocket_job_hw_submit           — Register writes to NPU
171 rocket_ioctl_create_bo         — Buffer allocation
140 rocket_job_run                 — Job scheduler dispatch
140 rocket_job_cleanup             — Post-completion cleanup
 94 rocket_ioctl_fini_bo           — Cache sync for NPU access
 10 rocket_ioctl_submit            — User-space submit IOCTLs
```

### Job Scheduling
- 10 SUBMIT calls → 140 jobs → 210 HW submits (avg 14 jobs/submit, 1.5 tasks/job)
- Jobs scheduled on CPU 1 (kernel worker), IRQs handled on CPU 0
- Each hw_submit takes ~2.5µs (register writes to NPU frontend)
- IRQ handler takes ~2µs, threaded handler ~18µs (includes next job dispatch)

### Multi-Core NPU Usage
Currently **single-core only** — Rocket driver uses one NPU core (core 0).
The IRQ handler thread chains job submissions: each completion triggers next submit.
Pattern: submit → IRQ → thread (submit next + cleanup previous) → IRQ → ...

## 3. NPU Hardware Architecture (from DTB + dmesg)

### Memory Map
| Address | Size | Name | Description |
|---------|------|------|-------------|
| 0xfdab0000 | 4K | PC | Program Counter / Frontend (Core 0) |
| 0xfdab1000 | 4K | CNA | Convolution Neural Accelerator (Core 0) |
| 0xfdab3000 | 4K | Core | Core control registers (Core 0) |
| 0xfdac0000-0xfdac3000 | — | Core 1 | Same layout as Core 0 |
| 0xfdad0000-0xfdad3000 | — | Core 2 | Same layout as Core 0 |

### Register Regions (per core)
- **PC (0x000)**: Frontend unit — receives command buffers, dispatches to CNA
- **CNA (0x1000)**: Convolution engine — performs MAC operations
- **Core (0x3000)**: Power/clock/interrupt control

### Clocks
- aclk: AXI clock
- hclk: AHB clock
- npu: Core clock (assigned 200MHz = 0xbebc200)
- pclk: APB clock

### Power
- vdd_npu_s0: 550-950mV regulator
- Power domains: nputop, npu1, npu2 (per-core power gating)

## 4. Proprietary vs Open-Source Comparison

### Performance (MobileNetV1 224x224, single core)
| Metric | RKNN (proprietary) | Rocket (open-source) | Ratio |
|--------|-------------------|---------------------|-------|
| Latency | 2.6ms | 12.4ms | 4.8x |
| FPS | 384 | 81 | 4.7x |

### Architecture Differences
| Aspect | RKNN Driver | Rocket Driver |
|--------|-------------|---------------|
| Kernel interface | DRM render node (/dev/dri/renderD129) | DRM accel node (/dev/accel/accel0) |
| IOCTL style | Custom RKNN IOCTLs | Standard DRM GEM + 4 custom IOCTLs |
| Multi-core | Yes (1/2/3 cores, bitmask) | Single core only (currently) |
| Model format | .rknn (proprietary) | .tflite via Mesa Teflon delegate |
| Compilation | Offline (rknn-toolkit) | JIT in Mesa Gallium driver |
| Buffer mgmt | Internal DMA pool | DRM GEM objects + IOMMU |
| Job submission | Direct register writes | gpu_sched framework |

### Why Rocket is Slower
1. **No multi-core**: RKNN can use all 3 cores (3x throughput potential)
2. **JIT compilation**: Mesa compiles layer-by-layer vs RKNN's pre-compiled graphs
3. **Limited op coverage**: Only convolution, addition, ReLU implemented
4. **No hardware-specific optimizations**: RKNN uses tiled memory layout, DMA chaining
5. **Extra cache sync overhead**: 340 PREP/FINI_BO calls per inference (cache maintenance)

## 5. NPU Command Submission Protocol

The Rocket driver reveals the RK3588 NPU command submission mechanism:

1. **Register Command Buffers**: User-space writes NPU register values into DMA buffers
2. **Frontend Unit (PC)**: Receives buffer DMA address + command count
3. **Execution**: Frontend writes registers to CNA unit sequentially
4. **Completion**: IRQ fires when all commands in a task are processed
5. **Chaining**: Multiple tasks in a job execute sequentially; jobs can have dependencies

Each "task" is essentially a list of (register_address, value) pairs that configure the CNA
for one convolution operation. The frontend (PC unit) reads these and writes them to CNA registers.

## 6. Key Findings

1. The NPU is a register-programmed accelerator — no instruction set, just register writes
2. The frontend (PC) is a simple DMA engine that writes register values to the CNA
3. Each convolution layer = 1 task = set of register writes to configure CNA + trigger
4. The CNA performs the actual MAC operations using configured weights/activations
5. Three independent cores share the same design but can run different layers in parallel
6. The proprietary RKNN stack's performance advantage comes primarily from multi-core scheduling and pre-compiled register command sequences
