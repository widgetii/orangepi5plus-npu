# openrknn latency benchmarks

Latency comparison between openrknn's OWN mode and the vendor
`librknnrt.so` on the 7 runtime models from `tests/ground_truth.json`.

## TL;DR

- **Single-core: parity with vendor on all 5 runtime models** (within
  noise, −0.2% to −1.5% on average latency). openrknn's from-scratch
  submit path doesn't add any measurable per-run overhead.
- **Multi-core: openrknn loses 1.31×–1.87× on 4/5 models** because
  `orknn_npu_submit` in `openrknn_drm.c` hardcodes `core_mask=0x0` and
  silently ignores whatever the user sets via `rknn_set_core_mask`. The
  mask is stored in `ctx->core_mask` but never plumbed into the
  ioctl descriptor. Tracked by [#60][i60].

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

## Why openrknn ignores the user's core_mask

`rknn_set_core_mask` stores the requested mask in `ctx->core_mask`
(see `openrknn_api.c:322`), but `orknn_npu_submit` in
`openrknn_drm.c:270` hardcodes the ioctl descriptor's `core_mask`
field to `0x0`:

```c
struct rknpu_submit sub = {
    .flags = seg->flags | RKNPU_JOB_BLOCK,
    .task_start = seg->sc_start,
    .task_number = seg->task_number,
    .task_obj_addr = task_bo->obj_addr,
    .core_mask = 0x0,   /* <-- ignores ctx->core_mask */
    .subcore_task = {
        { seg->sc_start, seg->sc_count },
        { seg->sc_start, seg->sc_count },
        { seg->sc_start, seg->sc_count },
        { seg->sc_start, seg->sc_count },
        { seg->sc_start, seg->sc_count },
    },
};
```

Wiring `ctx->core_mask` through to the submit isn't enough on its own:
for a real multi-core speedup the segment needs `task_number =
sc_count × active_cores` (the cross-core replica count) and each of
the 5 `subcore_task` entries needs a core-specific `(task_start,
task_count)` slice. The .rknn task BO already contains those replicas
— we just need to map them. See [#60][i60] for the full design sketch.

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
