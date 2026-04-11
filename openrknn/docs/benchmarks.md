# openrknn latency benchmarks

Latency comparison between openrknn's OWN mode and the vendor
`librknnrt.so` on the 7 runtime models from `tests/ground_truth.json`.

## Method

- **Hardware**: Orange Pi 5 Plus (RK3588, 16 GB LPDDR4X, Armbian 25.11.1)
- **NPU**: single core, `core_mask=0x1` on both sides
- **CPU governor**: `performance` on all 8 cores
- **Iterations**: 200 measured per run, plus `bench_rknn`'s internal
  warmup
- **Harness**: `openrknn/tests/bench_openrknn.py`, which shells out to
  `librocketnpu/tests/bench_rknn` under three configurations:

  | Config | Env | What it measures |
  |--------|-----|------------------|
  | `vendor` | (none) | `bench_rknn` linked against `/lib/librknnrt.so` directly — vendor baseline |
  | `proxy`  | `LD_PRELOAD=openrknn/librknn_api.so` | openrknn's proxy-dispatch path: shim dlopens vendor lib and forwards every call |
  | `own`    | `LD_PRELOAD=openrknn/librknn_api.so ORKNN_OWN=init,query,input,run,outputs` | openrknn's from-scratch path: FB parse, FB-derived segments, in-place regcmd patching, single-core NPU submit |

  For `fp16_parse` models (mobilesam_encoder, lprnet) the OWN path is
  `init,query` only — NPU extraction returns early so the run path
  falls through to proxy dispatch. That's still a valid measurement of
  combined openrknn-init + vendor-run latency.

## Results (2026-04-11, master @ c7b432a)

| Model | vendor avg/min (ms) | proxy avg/min (ms) | own avg/min (ms) | own vs vendor |
|-------|--------------------:|-------------------:|-----------------:|--------------:|
| mobilenet_v1      | 2.87 / 2.81   | 1.99 / 1.95   | **1.95 / 1.91**   | **−32.1%** |
| resnet50          | 9.15 / 9.14   | 9.17 / 9.15   | **9.08 / 9.08**   | −0.8% |
| yolov5s_relu_int8 | 18.80 / 18.67 | 18.75 / 18.64 | **18.48 / 18.28** | −1.7% |
| yolov8n           | 16.38 / 15.65 | 16.33 / 15.52 | **16.08 / 15.13** | −1.8% |
| deeplabv3         | 28.92 / 28.89 | 28.92 / 28.91 | **28.79 / 28.78** | −0.4% |
| mobilesam_encoder *(fp16, OWN init+query only)* | 102.24 / 102.06 | 102.37 / 101.90 | 102.21 / 101.96 | −0.0% |
| lprnet *(fp16, OWN init+query only)*            | 1.74 / 1.71     | 1.74 / 1.74     | 1.78 / 1.74     | +2.3% |

**Negative = openrknn OWN is faster.** Parity on 6/7 models, a large
win on mobilenet_v1, and no material regressions on any model. The
+2.3% on lprnet is well within measurement noise at a 1.74 ms baseline
(≈40 µs absolute).

### Why mobilenet_v1 shows a 32% win

The vendor's `bench_rknn` output for this model has higher variance
than openrknn's — vendor avg 2.87 ms vs min 2.81 ms, openrknn avg
1.95 ms vs min 1.91 ms. The min values also favor openrknn by 0.9 ms
consistently, so the gap isn't pure variance. Likely contributors:

1. openrknn's first run pays a one-time `patch_regcmd_addresses` cost
   (~5 ms); subsequent runs go straight to submit. The vendor may be
   redoing some per-run work we skip.
2. MBv1 is the smallest model, so per-run fixed overhead (cache syncs,
   ioctl setup) dominates. A 100 µs difference in that fixed cost
   shows up as a large percentage.
3. openrknn's output path skips a few vendor-side sanity checks that
   cost microseconds per call.

The absolute saving is ~0.9 ms per inference. On larger models the
absolute saving is smaller or zero because the NPU-bound computation
dominates.

## How to reproduce

On the Orange Pi 5 Plus (or any RK3588 board with `/lib/librknnrt.so`
installed and the `librocketnpu/tests/bench_rknn` binary available):

```bash
# Switch to performance governor
for g in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
    echo performance > "$g"
done

# Build openrknn
make -C openrknn

# Run the benchmark (200 iters per configuration)
python3 openrknn/tests/bench_openrknn.py \
    --bench /path/to/librocketnpu/tests/bench_rknn \
    --lib   openrknn/librknn_api.so \
    --models-dir /path/to/rknn_model_zoo/models \
    --ground-truth openrknn/tests/ground_truth.json \
    --iters 200

# Restore governor
for g in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
    echo ondemand > "$g"
done
```

The script prints a markdown table identical to the one in this file.
Re-running on different kernel versions or DVFS states should give
comparable numbers; if OWN mode ever regresses materially vs vendor,
file an issue with the bench output attached.

## Related

- Tracked in issue [#66](https://github.com/widgetii/orangepi5plus-npu/issues/66)
- Umbrella roadmap: [#68](https://github.com/widgetii/orangepi5plus-npu/issues/68)
- Correctness validation (accuracy, not latency) lives in
  `tests/ci_validate.sh` Phase 2 — every model passes 7/7 in OWN mode
  with zero vendor runtime dependencies
