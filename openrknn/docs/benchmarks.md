# openrknn latency benchmarks

Latency comparison between openrknn's OWN mode and the vendor
`librknnrt.so` on the 7 runtime models from `tests/ground_truth.json`.

## TL;DR

**openrknn is at parity with or ahead of vendor on all 5 runtime
models × both single-core and triple-core configurations.**

- **Single-core**: −0.3% to −5.4% across all models (noise range).
- **Triple-core**: −0.9% to −13.1% on the 4 parallelisable models.
  yolov5 OWN beats the vendor by **13.1%**. yolov8n has no multi-core
  benefit even under the vendor — it's at single-core parity in both
  stacks.

See [#60][i60] for the full multi-core implementation that landed.

[i60]: https://github.com/widgetii/orangepi5plus-npu/issues/60

## Method

- **Hardware**: Orange Pi 5 Plus (RK3588, 16 GB LPDDR4X, Armbian 25.11.1)
- **CPU governor**: `performance` on all 8 cores
- **Measurement**: ctypes harness calls `rknn_init` + `rknn_set_core_mask(ctx, mask)`
  + `rknn_inputs_set` + 500× `rknn_run` after 20-iter warmup. Wall-clock
  timing via `time.perf_counter_ns()`.
- **Core masks compared**: `RKNN_NPU_CORE_0 (0x1)` and
  `RKNN_NPU_CORE_0_1_2 (0x7)`
- **Per-core load verified** via `/sys/kernel/debug/rknpu/load` polled
  during each run

## Results (master @ latest, single RKNPU session, performance governor, 500 iters)

| Model             | mask   | vendor (ms) | openrknn (ms) | gap     |
|-------------------|--------|------------:|--------------:|--------:|
| mobilenet_v1      | 1-core |       2.056 |         1.945 |  −5.4%  |
| mobilenet_v1      | 3-core |       1.043 |     **1.018** |  −2.4%  |
| resnet50-v2-7     | 1-core |       9.131 |         9.074 |  −0.6%  |
| resnet50-v2-7     | 3-core |       6.930 |         6.868 |  −0.9%  |
| yolov5s_relu_int8 | 1-core |      18.500 |        18.352 |  −0.8%  |
| yolov5s_relu_int8 | 3-core |      12.417 |    **10.789** | **−13.1%** |
| yolov8n           | 1-core |      16.260 |        16.060 |  −1.2%  |
| yolov8n           | 3-core |      16.709 |        15.767 |  −5.6%  |
| deeplabv3         | 1-core |      28.878 |        28.799 |  −0.3%  |
| deeplabv3         | 3-core |      18.729 |        18.484 |  −1.3%  |

Negative = openrknn is faster. Accuracy verified on all 5 runtime
models × mask 0x7 via `validate_accuracy.py --own ORKNN_CORE_MASK=0x7`:
7/7 passed (correct top-1 classes for MBv1/ResNet50, correct YOLO
detections, correct DeepLabv3 segmentation classes).

### Why yolov5 beats vendor by 13.1%

This is the only model where the speed-up is noticeably more than
noise. The vendor's 3-core yolov5 submit pattern for CORE_0_1
(captured via `intercept_swap` + the patched `sc[0..4]` dumper)
emits 38 small submits with many 6-task "cleanup" submits between
larger pingpong runs. openrknn's `fb_build_multicore_submits`
groups consecutive ops with the same (kind, flags) more
aggressively and emits 13 submits. Fewer submits = fewer per-ioctl
kernel-entry overhead = less wall-clock time for the same NPU
work.

(This is only visible as a latency win, not correctness: every
submit dispatches the same task range to the same cores with the
same regcmds; the kernel just schedules fewer syscall dispatches.)

## Multi-core implementation

See `docs/segmentation_from_fb.md` §multi-core for the full rule
derivation. Summary:

- FB field `f[10]` on each operator is a 6-element vector already
  encoding per-core-count task counts:
    f[10][0]     — single-core (non-LUT)
    f[10][1]     — dual-core core 0 / single-core LUT
    f[10][2]     — dual-core core 1 (0 for LUT)
    f[10][3..5]  — triple-core per-core counts
- The .rknn task BO is partitioned into contiguous per-core-count
  regions. Region start offsets derive from cumulative region
  sizes computed from the slot-sum formula.
- For each core count, openrknn builds a segment list that walks
  ops in order, emits a new submit on phase (pingpong/barrier)
  transition, and stacks per-core `(sc_start, sc_count)` pairs.
- At submit time, `orknn_own_run` picks the 1c/2c/3c plan based on
  `popcount(ctx->core_mask)` and dispatches through
  `orknn_npu_submit_multicore`, which fills the kernel's
  `subcore_task[]` slots per the dispatch rule in
  `rknpu-driver/rknpu_job.c:320-330`:
    1-core submit → `subcore_task[0]`
    2-core submit → `subcore_task[0..1]`
    3-core submit → `subcore_task[2..4]`
- Models without a compiled multi-core layout (detectable via
  `f[10][1..5]` all zero) fall back to single-core with a warning.

## Known limitations

- **DeepLabv3 3-core requires a per-region IO-prologue skip.** The
  compiler emits a 1-task input reformat (op_idx=0 em=0x18) at the
  start of each per-core-count region. The region-start
  calculation accounts for this via `op_per_core_count()` on the
  InputOperator — verified byte-exact against vendor for
  deeplabv3. If a future model introduces a different prologue
  pattern, the capture → verify loop in `dump_segments.py` will
  flag it.
- **No 2-core test path in CI yet.** The committed ground-truth
  artifacts cover single-core only; 2-core/3-core verification has
  to be done with `ORKNN_CORE_MASK=0x3` or `0x7` set in the environment.
  Tracked as future work.

## How to reproduce

```bash
ssh root@<board>
for g in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
    echo performance > "$g"
done

# Single-core bench
python3 /path/to/openrknn/tests/bench_openrknn.py \
    --bench /path/to/librocketnpu/tests/bench_rknn \
    --lib   /path/to/openrknn/librknn_api.so \
    --models-dir /path/to/models \
    --ground-truth /path/to/openrknn/tests/ground_truth.json \
    --iters 500

# Multi-core bench (with rknn_set_core_mask)
python3 /tmp/bench_core_mask.py    # see PR for the one-off ctypes harness

# Accuracy suite with explicit core_mask
ORKNN_CORE_MASK=0x7 python3 openrknn/tests/validate_accuracy.py \
    --lib openrknn/librknn_api.so \
    --models-dir /path/to/models \
    --images-dir openrknn/tests/test_images \
    --ground-truth openrknn/tests/ground_truth.json \
    --own init,query,input,run,outputs
```

## Related

- [#60](https://github.com/widgetii/orangepi5plus-npu/issues/60) —
  Multi-core batch inference (closed by this work)
- [#66](https://github.com/widgetii/orangepi5plus-npu/issues/66) —
  Latency benchmarking (closed earlier)
- [#68](https://github.com/widgetii/orangepi5plus-npu/issues/68) —
  Roadmap umbrella
