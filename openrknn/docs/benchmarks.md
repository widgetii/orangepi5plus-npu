# openrknn latency benchmarks

Latency comparison between openrknn's OWN mode and the vendor
`librknnrt.so` on the 7 runtime models from `tests/ground_truth.json`.

## TL;DR

- **Single-core: parity with vendor on all 5 runtime models** (within
  noise, −0.2% to −1.5% on average latency). openrknn's from-scratch
  submit path doesn't add any measurable per-run overhead.
- **`rknn_set_core_mask` is now honoured for single-core values.**
  Users can pick Core 0 (0x1), Core 1 (0x2), or Core 2 (0x4)
  explicitly for load balancing across concurrent processes.
- **Multi-core values (0x3 / 0x5 / 0x6 / 0x7) fall back to Core 0
  with a warning.** True multi-core parallelism requires parsing
  the .rknn's pre-compiled per-core-count task BO layouts, which
  openrknn doesn't yet do — see [#60][i60] and the "Multi-core
  roadmap" section below.

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

The earlier `tests/bench_openrknn.py` script (which shells out to
`librocketnpu/tests/bench_rknn`) is useful for quick sanity checks but
doesn't set `core_mask` explicitly and has measurement artefacts on the
shortest model (mobilenet_v1 first-iter warmup leaks into the average).
The numbers below come from the ctypes harness, which bypasses that.

## Single-core (RKNN_NPU_CORE_0): parity

| Model             | vendor (ms) | openrknn (ms) | gap     |
|-------------------|------------:|--------------:|--------:|
| mobilenet_v1      |    1.98     |     1.95      |  −1.5%  |
| resnet50          |    9.13     |     9.08      |  −0.5%  |
| yolov5s_relu_int8 |   18.61     |    18.47      |  −0.7%  |
| yolov8n           |   16.29     |    16.09      |  −1.3%  |
| deeplabv3         |   28.85     |    28.79      |  −0.2%  |

Negative means openrknn is faster; all five are within measurement noise.
`/sys/kernel/debug/rknpu/load` shows **Core 0 only** active for both
paths (0% on Core 1/2).

**Takeaway:** openrknn's FB-derived segment submit path has no extra
overhead vs the vendor's runtime loop. The per-run cost is dominated by
the NPU-bound compute and the cache-sync ioctls, which both paths do
identically.

## Triple-core (RKNN_NPU_CORE_0_1_2): gap

| Model             | vendor (ms) | openrknn (ms) | gap       | single→triple speedup (vendor) |
|-------------------|------------:|--------------:|----------:|-------------------------------:|
| mobilenet_v1      |    **1.04** |     1.95      |  **+87%** |  1.90× |
| resnet50          |    **6.91** |     9.08      |  **+31%** |  1.32× |
| yolov5s_relu_int8 |   **12.58** |    18.50      |  **+47%** |  1.48× |
| yolov8n           |   16.35     |    16.03      |   −2.0%   |  1.00× *(model doesn't parallelise)* |
| deeplabv3         |   **18.95** |    28.79      |  **+52%** |  1.52× |

Positive means openrknn is slower. On every model the vendor can
parallelise, openrknn gives up 1.31×–1.87× because it silently falls
back to single-core. yolov8n is the only model where even the vendor
sees no speedup — its compiled segments don't distribute across cores,
so openrknn's single-core execution is just as fast.

**Takeaway:** the multi-core gap is the single biggest practical
limitation of openrknn today. Any application that calls
`rknn_set_core_mask(RKNN_NPU_CORE_0_1_2)` is silently getting
single-core execution.

## Multi-core roadmap

### Phase 1: honour single-core masks *(landed, this file)*

`rknn_set_core_mask` now works for `RKNN_NPU_CORE_0` (0x1),
`RKNN_NPU_CORE_1` (0x2), and `RKNN_NPU_CORE_2` (0x4). The kernel
reads `subcore_task[core_index]` to find the task range for the
active core, and we populate slots [0..2] with the same
`(sc_start, sc_count)` so any of the three works. Hardware verified
via `/sys/kernel/debug/rknpu/load` — mask=0x2 lights up Core 1 at
93-99%, Core 0/2 at 0%, etc.

Multi-core masks (0x3, 0x5, 0x6, 0x7) log a one-time warning and
fall back to Core 0. Set `ORKNN_ALLOW_MULTICORE=1` to force the
submit through anyway — expect hangs or wrong outputs.

### Phase 2: honest multi-core parallelism *(#60 — not landed)*

The hard scope. Investigation during the bench work revealed that
the RKNN toolkit pre-compiles **separate per-core-count task BO
regions** into the .rknn file. For mobilenet_v1 the layout is:

| Region | Size | Purpose |
|--------|------|---------|
| tasks 0–50    | 51 tasks | single-core execution |
| tasks 51–131  | 81 tasks | dual-core execution (core 0 at 51–82, core 1 at 99–130, + cleanup) |
| tasks 133–236 | 104 tasks | triple-core execution (core 0 at 133–160, core 1 at 177–204, core 2 at 207–234, + cleanup) |
| tasks 237–306 | 70 tasks | second-cycle warmup/iter replicas |

Each region has its own pre-computed regcmds with different
task_start offsets per core. The vendor reads `subcore_task[core_index + 2]`
for 3-core mode (slots [2..4]) and `subcore_task[core_index]` for 1/2-core
(slots [0..2]) — this is verified in the kernel driver at
`rknpu-driver/rknpu_job.c:320-330` (`rknpu_get_task_number` +
`rknpu_core_index` functions).

openrknn's `fb_build_segments` derives only the single-core region
from the FB operator graph. Adding multi-core support requires:

1. Reverse-engineering how the RKNN toolkit encodes per-core-count
   regions in the task BO (or in an auxiliary FB field we don't
   yet parse)
2. Extending `parse_fb_operators` / `fb_build_segments` to produce
   N segment lists, one per supported core count
3. Runtime dispatch in `orknn_own_run` that picks the right segment
   list based on `ctx->core_mask` at submit time

Issue [#60][i60] tracks this; reproduction of the vendor's exact
submit pattern is captured in the task 10.1 artifacts at the end of
this PR's commit message.

### Gap that remains

Until Phase 2 lands, openrknn gives up **1.31×–1.87× on 4 of 5
runtime models** when the caller asks for multi-core:

| Model      | vendor 1c | vendor 3c | openrknn (1c fallback) | gap    |
|------------|----------:|----------:|-----------------------:|-------:|
| MBv1       |  1.98 ms  |  1.04 ms  |                1.95 ms | +87%   |
| ResNet50   |  9.13 ms  |  6.91 ms  |                9.08 ms | +31%   |
| YOLOv5     | 18.61 ms  | 12.58 ms  |               18.47 ms | +47%   |
| YOLOv8     | 16.29 ms  | 16.35 ms  |               16.09 ms |  −1%   |
| DeepLabv3  | 28.85 ms  | 18.95 ms  |               28.79 ms | +52%   |

YOLOv8 is the only model where even the vendor sees no benefit from
multi-core (its compiled pattern doesn't parallelise), so openrknn's
fallback is already at parity.

## How to reproduce

The numbers above came from a one-off ctypes harness (not committed —
it's too specific to this diagnosis). To reproduce:

```bash
ssh root@<board>
for g in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
    echo performance > "$g"
done

# The committed single-core-only benchmark (uses bench_rknn):
python3 /path/to/openrknn/tests/bench_openrknn.py \
    --bench /path/to/librocketnpu/tests/bench_rknn \
    --lib   /path/to/openrknn/librknn_api.so \
    --models-dir /path/to/models \
    --ground-truth /path/to/openrknn/tests/ground_truth.json \
    --iters 500

# Monitor per-core load while it runs:
watch -n 0.3 cat /sys/kernel/debug/rknpu/load
```

For the full triple-core comparison, you need to link a benchmark
program against librknn_api.so, call `rknn_set_core_mask(ctx, 0x7)`
before the run loop, and compare vendor vs openrknn explicitly. That
harness will be committed alongside the fix for [#60][i60].

## Side-finds from this benchmark

1. **deeplabv3 query mismatch** — openrknn's `rknn_query(RKNN_QUERY_INPUT_ATTR)`
   result for deeplabv3 differs from the vendor's by one byte at offset
   360. Dims/scale/zp/name all match, just one flag byte. Cosmetic but
   should be fixed; file a separate issue before it confuses someone.
2. **`bench_openrknn.py` first-iter leakage** — the earlier benchmark
   reported −32% for mobilenet_v1 because the vendor's first few
   iterations at cold NPU state ran ~3 ms each and weren't fully
   washed out by bench_rknn's internal warmup. The ctypes harness with
   a 20-iter explicit warmup eliminates this. The committed
   `bench_openrknn.py` inherits the issue but it only matters for
   models under ~3 ms; larger models are unaffected.

## Related

- [#60][i60] — Multi-core batch inference (this is the fix)
- [#66](https://github.com/widgetii/orangepi5plus-npu/issues/66) —
  Tracking issue for this benchmark
- [#68](https://github.com/widgetii/orangepi5plus-npu/issues/68) —
  Roadmap umbrella
