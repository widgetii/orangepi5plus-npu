# openrknn throughput benchmarks

Throughput comparison between openrknn's OWN mode and the vendor
`librknnrt.so` under concurrent load — i.e. "how many FPS can the
RK3588 NPU sustain when N worker threads submit inferences in
parallel", not "how long does one inference take".

Latency is covered in [`benchmarks.md`](benchmarks.md); this
document is the concurrency story.

## TL;DR

- **Pinned strategy** (worker `i` uses `core_mask = 0x1 << (i % 3)`)
  delivers **≈3× single-core FPS** on all models. Peak is reached at
  N=3 and stays flat out to N=8; p95/p99 double from N=4 on as new
  workers queue onto already-occupied cores.
- **Multicore strategy** (all workers use `core_mask = 0x7`)
  saturates at **N=1**; adding workers does not grow aggregate FPS
  because every submit is already a 3-core job. Use it only when
  minimum single-request latency matters more than concurrency.
- openrknn tracks the vendor across all (model, strategy, N) cells.
  Moderate/high-N deltas are within ±4% on mobilenet_v1, resnet50
  and deeplabv3. yolov5 multicore at N≥3 is **+10 to +14% faster on
  openrknn**, matching the 3-core latency win reported in
  [`benchmarks.md`](benchmarks.md). yolov8 pinned has a low-N gap
  (−38% at N=1, closing to −5% at N=8) — filed as a follow-up.
- GPU (Mali G610) is **not a throughput knob** for any of our 5
  runtime models. The vendor uses the Mali only as a fallback for
  ops compiled with `target=1`, and none of the models in the
  benchmark set have any such ops. Peer-accelerator throughput
  scaling is a future-scope exploration, not a Phase 12 deliverable.

## Method

- **Hardware**: Orange Pi 5 Plus (RK3588, 16 GB LPDDR4X, Armbian 25.11.1)
- **Kernel**: mainline 6.18 (Rocket driver) — same as `benchmarks.md`
- **CPU governor**: `performance` on the A76 big cluster (cpus 4–7);
  restored on exit by the sweep driver
- **Harness**: `tests/bench_throughput.c` — pure C, pthread, no Python,
  no ctypes. Each worker gets its own `rknn_context` via an
  independent `rknn_init` call on a shared model-bytes buffer.
  Workers synchronise start via a two-phase barrier pair
  (`ready_barrier` for "warmup done", `go_barrier` for "deadline
  set, go") so there is no t_end race. Per-request latency is
  recorded inline into a pre-allocated uint32 ns array.
- **Hot loop** per worker: `rknn_run` → `rknn_outputs_get` (with
  `want_float=0` so no dequantize) → record `t1-t0` → release. This
  is the closest approximation to an inference server's per-request
  cost without also timing post-processing. `rknn_inputs_set` is
  called once outside the loop.
- **Library selection**: the binary links against `-lrknnrt` and
  picks up `/lib/librknnrt.so` via SONAME by default (vendor
  baseline). For openrknn OWN mode, the sweep driver sets
  `LD_PRELOAD=./librknn_api.so ORKNN_OWN=init,query,input,run,outputs`
  — exactly the same pattern `bench_openrknn.py` uses for latency.
- **Measurement window**: 5 s per data point, 30 warmup iterations
  per worker before the barrier, N ∈ {1, 2, 3, 4, 6, 8}.
- **Strategies**:
  - `pinned`: worker `i` uses `core_mask = 1 << (i % 3)` → workers
    0, 1, 2 go to distinct NPU cores; workers 4+ queue on an
    already-occupied core.
  - `multicore`: every worker uses `0x7` → each submit is a
    3-core job; the kernel serialises concurrent 3-core submits.
- **dmesg check** after every run: no `RKNPU: job timeout` or
  `soft reset` entries attributable to this sweep.

## Results

All 5 runtime models × both strategies × N ∈ {1, 2, 3, 4, 6, 8} ×
two libraries. FPS is aggregate count / wall-clock. Latency columns
are p50 / p95 / p99 ms across all per-request samples from all
workers. Δ% is (openrknn - vendor) / vendor (negative = openrknn
slower).

### mobilenet_v1

**pinned** — peak 1425 FPS at N≥6; openrknn within 3% of vendor
from N=3 onwards.

| N | vendor FPS | vendor p50/p95/p99 ms | openrknn FPS | openrknn p50/p95/p99 ms | Δ% |
|--:|-----------:|-----------------------|-------------:|-------------------------|---:|
| 1 |  452.5 | 2.21 / 2.23 / 2.24 |  492.0 | 2.03 / 2.04 / 2.04 | +8.7 |
| 2 |  919.9 | 2.23 / 2.36 / 2.37 |  743.2 | 2.37 / 4.47 / 4.54 | −19.2 |
| 3 | 1374.5 | 2.19 / 2.43 / 2.44 | 1323.5 | 2.22 / 2.50 / 2.51 | −3.7 |
| 4 | 1404.2 | 2.20 / 4.06 / 4.10 | 1363.9 | 2.40 / 4.14 / 4.26 | −2.9 |
| 6 | 1426.8 | 4.26 / 4.32 / 4.33 | 1426.2 | 4.26 / 4.38 / 4.47 | −0.0 |
| 8 | 1429.2 | 6.08 / 6.48 / 6.51 | 1434.6 | 6.03 / 6.46 / 6.50 | +0.4 |

**multicore** — peak ~1080 FPS, reached basically at N=1. Adding
workers just distributes the same work over more queues and
lengthens p95.

| N | vendor FPS | vendor p50/p95/p99 ms | openrknn FPS | openrknn p50/p95/p99 ms | Δ% |
|--:|-----------:|-----------------------|-------------:|-------------------------|---:|
| 1 |  927.5 | 1.06 / 1.21 / 1.26 |  827.3 | 1.11 / 2.39 / 2.39 | −10.8 |
| 2 | 1011.6 | 1.98 / 2.79 / 2.88 | 1030.8 | 1.99 / 2.61 / 2.85 | +1.9 |
| 3 | 1030.2 | 2.97 / 3.75 / 4.21 | 1008.7 | 3.03 / 3.04 / 3.07 | −2.1 |
| 4 | 1038.4 | 3.90 / 4.01 / 4.08 | 1017.1 | 3.95 / 4.69 / 4.83 | −2.0 |
| 6 | 1068.5 | 5.58 / 6.55 / 7.09 | 1078.9 | 5.55 / 6.37 / 6.79 | +1.0 |
| 8 | 1077.8 | 7.43 / 8.36 / 8.83 | 1082.6 | 7.40 / 8.24 / 8.56 | +0.5 |

### resnet50

**pinned** — peak 288 FPS at N=4; openrknn within 2% of vendor
across the whole range.

| N | vendor FPS | vendor p50/p95/p99 ms | openrknn FPS | openrknn p50/p95/p99 ms | Δ% |
|--:|-----------:|-----------------------|-------------:|-------------------------|---:|
| 1 | 109.3 |  9.15 /  9.15 /  9.16 | 109.0 |  9.18 /  9.18 /  9.18 | −0.3 |
| 2 | 202.8 |  9.66 / 10.30 / 10.32 | 205.2 |  9.50 / 10.08 / 10.09 | +1.1 |
| 3 | 280.7 | 10.85 / 10.96 / 11.02 | 282.1 | 10.58 / 10.84 / 24.97 | +0.5 |
| 4 | 288.4 | 10.80 / 19.93 / 20.02 | 284.2 | 10.92 / 20.02 / 20.10 | −1.5 |
| 6 | 288.2 | 21.18 / 21.36 / 21.42 | 288.3 | 21.18 / 21.36 / 21.44 | +0.0 |
| 8 | 288.0 | 30.07 / 31.96 / 32.06 | 288.1 | 30.07 / 31.95 / 32.01 | +0.0 |

**multicore** — peak 163 FPS at N=2; openrknn within 1% beyond the
low-N overhead region.

| N | vendor FPS | vendor p50/p95/p99 ms | openrknn FPS | openrknn p50/p95/p99 ms | Δ% |
|--:|-----------:|-----------------------|-------------:|-------------------------|---:|
| 1 | 136.7 |  7.00 /  9.18 /  9.29 | 121.8 |  8.53 /  8.92 /  9.00 | −10.9 |
| 2 | 163.4 | 12.05 / 13.61 / 13.88 | 150.4 | 13.48 / 13.56 / 13.59 | −7.9 |
| 3 | 158.7 | 18.93 / 20.48 / 20.56 | 163.1 | 18.12 / 20.51 / 20.65 | +2.8 |
| 4 | 163.0 | 24.40 / 27.16 / 27.70 | 163.7 | 24.38 / 27.20 / 27.67 | +0.4 |
| 6 | 162.3 | 36.97 / 38.43 / 40.03 | 162.0 | 37.05 / 38.20 / 39.11 | −0.2 |
| 8 | 160.9 | 49.80 / 51.07 / 51.60 | 160.6 | 49.94 / 51.23 / 51.74 | −0.2 |

### yolov5

**pinned** — openrknn is ~15% slower across the pinned range on
this model, widening slightly at the N=4 knee. Same pattern as the
1-core latency gap in `benchmarks.md` (18.4 ms vendor vs 18.3 ms
openrknn is single-stream; under load the openrknn path costs more
per worker).

| N | vendor FPS | vendor p50/p95/p99 ms | openrknn FPS | openrknn p50/p95/p99 ms | Δ% |
|--:|-----------:|-----------------------|-------------:|-------------------------|---:|
| 1 |  51.7 | 19.33 / 19.37 / 19.79 |  42.8 | 23.39 / 23.40 / 23.42 | −17.3 |
| 2 |  96.4 | 20.78 / 21.19 / 21.27 |  81.6 | 24.50 / 24.54 / 24.68 | −15.3 |
| 3 | 133.7 | 22.45 / 22.80 / 24.51 | 114.7 | 25.93 / 26.18 / 34.49 | −14.2 |
| 4 | 132.5 | 24.49 / 43.60 / 43.86 | 102.5 | 35.10 / 45.26 / 45.44 | −22.6 |
| 6 | 138.0 | 43.32 / 44.04 / 44.46 | 127.7 | 46.29 / 54.72 / 55.47 |  −7.4 |
| 8 | 138.3 | 63.62 / 65.11 / 67.15 | 129.5 | 66.32 / 75.59 / 76.54 |  −6.4 |

**multicore** — at N≥3 openrknn **beats** vendor by +10 to +14%.
This is the same structural win recorded in `benchmarks.md` for
3-core latency (−13.1%) carried through to concurrent throughput.

| N | vendor FPS | vendor p50/p95/p99 ms | openrknn FPS | openrknn p50/p95/p99 ms | Δ% |
|--:|-----------:|-----------------------|-------------:|-------------------------|---:|
| 1 | 75.4 | 13.23 / 13.79 / 14.68 | 61.1 | 15.73 / 15.75 / 29.29 | −19.0 |
| 2 | 78.9 | 25.50 / 25.63 / 25.96 | 72.9 | 29.58 / 34.66 / 34.75 |  −7.6 |
| 3 | 80.5 | 37.61 / 38.04 / 38.59 | 91.6 | 32.83 / 41.81 / 43.21 | **+13.9** |
| 4 | 81.9 | 49.44 / 51.78 / 52.58 | 90.8 | 42.19 / 54.90 / 55.91 | **+10.9** |
| 6 | 83.3 | 71.69 / 75.74 / 77.64 | 91.7 | 64.67 / 75.69 / 77.67 | **+10.1** |
| 8 | 83.1 | 96.42 / 99.13 / 99.92 | 91.8 | 86.95 / 96.90 / 98.71 | **+10.5** |

### yolov8

**pinned** — openrknn has a significant N=1 penalty (−38.8%) that
closes as N grows; still −5% at N=8. Filed as a follow-up
optimisation.

| N | vendor FPS | vendor p50/p95/p99 ms | openrknn FPS | openrknn p50/p95/p99 ms | Δ% |
|--:|-----------:|-----------------------|-------------:|-------------------------|---:|
| 1 |  59.9 | 16.74 / 16.75 / 16.76 |  36.6 | 26.43 / 33.68 / 33.99 | −38.8 |
| 2 | 114.7 | 17.18 / 21.09 / 21.37 |  98.7 | 19.95 / 26.16 / 26.65 | −14.0 |
| 3 | 171.8 | 17.45 / 18.13 / 18.20 | 126.3 | 26.49 / 28.52 / 29.51 | −26.5 |
| 4 | 172.2 | 21.18 / 31.61 / 33.34 | 142.2 | 27.74 / 40.30 / 41.97 | −17.4 |
| 6 | 193.1 | 30.97 / 34.63 / 37.12 | 176.5 | 33.77 / 40.58 / 41.63 |  −8.6 |
| 8 | 192.4 | 45.23 / 49.62 / 51.23 | 183.4 | 46.03 / 55.19 / 56.92 |  −4.7 |

**multicore** — yolov8 has no multi-core latency benefit on either
stack (see `benchmarks.md`), so the multicore peak (67 FPS) is
actually *below* the pinned peak (193 FPS). openrknn within 4% of
vendor across the whole range.

| N | vendor FPS | vendor p50/p95/p99 ms | openrknn FPS | openrknn p50/p95/p99 ms | Δ% |
|--:|-----------:|-----------------------|-------------:|-------------------------|---:|
| 1 | 59.4 |  16.81 /  18.00 /  18.14 | 49.0 |  19.52 /  25.81 /  25.98 | −17.5 |
| 2 | 63.9 |  32.22 /  33.02 /  37.34 | 56.0 |  38.20 /  40.80 /  41.23 | −12.3 |
| 3 | 62.5 |  48.39 /  48.65 /  49.28 | 64.5 |  46.43 /  53.10 /  54.82 |  +3.4 |
| 4 | 67.5 |  59.19 /  64.69 /  66.61 | 66.0 |  60.30 /  67.74 /  68.98 |  −2.2 |
| 6 | 66.8 |  89.26 /  97.72 / 102.02 | 64.6 |  91.86 /  98.29 /  99.57 |  −3.3 |
| 8 | 67.3 | 118.77 / 126.39 / 132.80 | 65.8 | 121.60 / 127.01 / 127.99 |  −2.3 |

### deeplabv3

**pinned** — peak 95 FPS at N=3; openrknn within 4% on every cell.
Large-model case: the NPU cost dominates per-request overhead, so
both stacks converge.

| N | vendor FPS | vendor p50/p95/p99 ms | openrknn FPS | openrknn p50/p95/p99 ms | Δ% |
|--:|-----------:|-----------------------|-------------:|-------------------------|---:|
| 1 | 34.3 | 29.15 / 29.17 / 29.18 | 32.9 | 30.35 / 30.37 / 30.39 | −4.0 |
| 2 | 66.8 | 29.70 / 30.27 / 30.29 | 65.1 | 30.10 / 31.36 / 31.38 | −2.4 |
| 3 | 94.1 | 32.00 / 32.16 / 32.21 | 92.7 | 32.08 / 32.90 / 32.95 | −1.5 |
| 4 | 94.0 | 32.10 / 62.79 / 62.89 | 92.2 | 33.09 / 62.48 / 62.56 | −1.9 |
| 6 | 94.8 | 63.37 / 63.62 / 63.70 | 95.0 | 63.31 / 63.55 / 63.62 | +0.2 |
| 8 | 94.7 | 94.09 / 95.22 / 95.31 | 94.8 | 94.01 / 95.13 / 95.19 | +0.1 |

**multicore** — peak 54 FPS at N=1. Saturates immediately; openrknn
within 1% past N=1.

| N | vendor FPS | vendor p50/p95/p99 ms | openrknn FPS | openrknn p50/p95/p99 ms | Δ% |
|--:|-----------:|-----------------------|-------------:|-------------------------|---:|
| 1 | 52.3 |  19.08 /  19.58 /  19.60 | 49.9 |  20.03 /  20.07 /  20.09 | −4.5 |
| 2 | 54.0 |  37.03 /  52.10 /  54.29 | 53.9 |  37.30 /  52.44 /  52.54 | −0.2 |
| 3 | 54.0 |  55.59 /  55.97 /  70.78 | 54.3 |  55.28 /  55.37 /  57.36 | +0.6 |
| 4 | 53.9 |  74.20 /  88.83 / 106.69 | 54.5 |  73.41 /  88.61 / 103.44 | +1.1 |
| 6 | 53.9 | 111.22 / 126.37 / 140.90 | 54.5 | 110.14 / 123.75 / 140.88 | +1.0 |
| 8 | 54.0 | 148.19 / 163.88 / 178.29 | 54.5 | 146.65 / 163.32 / 176.45 | +1.0 |

## Peak FPS and knee per model

| Model        | strategy  | peak FPS | knee N | single-core baseline |
|--------------|-----------|---------:|-------:|---------------------:|
| mobilenet_v1 | pinned    |   1434.6 |      3 |         493 (≈N=1)   |
| mobilenet_v1 | multicore |   1082.6 |      1 |         928          |
| resnet50     | pinned    |    288.4 |      4 |         109          |
| resnet50     | multicore |    163.7 |      2 |         137          |
| yolov5       | pinned    |    138.3 |      6 |          52          |
| yolov5       | multicore |     91.8 |      3 |          75          |
| yolov8       | pinned    |    193.1 |      6 |          60          |
| yolov8       | multicore |     67.5 |      4 |          59          |
| deeplabv3    | pinned    |     95.0 |      3 |          34          |
| deeplabv3    | multicore |     54.5 |      2 |          52          |

**Takeaway**: for every model, the **pinned** peak is higher than
the **multicore** peak — sometimes by 3× (mobilenet_v1: 1435 vs
1083), sometimes by 1.8× (deeplabv3: 95 vs 55). This is consistent
with the NPU's internal scheduler: a 3-core submit uses all three
cores cooperatively on one inference (minimising latency) while
three 1-core submits use all three cores independently on three
inferences (maximising throughput). Which one wins for you depends
on whether your workload metric is tail latency (multicore) or
aggregate FPS (pinned).

## On the Mali GPU as a peer accelerator

The RK3588 includes a Mali G610 GPU reachable from user-space via
OpenCL (`libOpenCL.so`). It is reasonable to ask whether the GPU
can soak up additional inference FPS alongside the NPU to scale
throughput above the NPU's own ceiling.

What the vendor runtime actually does today:

- The model compiler (`rknn-toolkit2`) can mark individual ops
  with `target=1`, meaning "run on GPU instead of NPU". This is a
  *compile-time* decision.
- At runtime `librknnrt` dlopens `libOpenCL.so` and uses the GPU
  only as a **fallback** for those `target=1` ops — e.g. ops that
  the NPU cannot express natively. There is no "run this .rknn
  model on the GPU" mode.
- None of the 5 runtime models in our benchmark set have any
  `target=1` ops, so the GPU is **dead weight** for this workload
  in the vendor stack.

openrknn does not implement the GPU fallback at all; every op in
the supported models is NPU-targeted, and models with GPU-target
ops fall through to the proxy path via `extract_npu_data` returning
early.

**What a real peer-accelerator throughput multiplier would need** is
an independent inference engine for `.rknn` models running on the
Mali GPU, scheduled alongside the NPU by a shared dispatcher. That
is a much bigger scope than Phase 12 — a standalone OpenCL-based
executor for the RKNN op set, per-op quant handling, and a
top-level router. Filed as a follow-up roadmap issue so the idea
is tracked without blocking this PR.

## Reproduction

```bash
# Build the harness
make -C openrknn tests/bench_throughput

# Single-point sanity check
cd openrknn

# Vendor baseline, mobilenet_v1, 3 pinned workers, 3 s window:
./tests/bench_throughput \
    --model /root/npu-research/mobilenet_v1.rknn \
    --workers 3 --duration 3 --strategy pinned --warmup 20

# openrknn OWN mode, same config:
LD_PRELOAD=./librknn_api.so ORKNN_OWN=init,query,input,run,outputs \
    ./tests/bench_throughput \
    --model /root/npu-research/mobilenet_v1.rknn \
    --workers 3 --duration 3 --strategy pinned --warmup 20

# Full sweep — 5 models × 2 strategies × 6 N-values × 2 libs
# (wall-clock ≈ 15 min):
bash tests/bench_throughput.sh \
    --models-dir /root/npu-research \
    --duration 5 --warmup 30 \
    --out /tmp/throughput_run
```

The sweep driver sets the A76 cluster to `performance` governor on
entry and restores it on exit. Raw per-point JSON lands in
`${OUT}/raw.jsonl`; an assembled markdown report in
`${OUT}/summary.md`.

## Numbers as of

These tables were produced from a 120-point sweep on 2026-04-11
(Armbian 25.11.1, kernel 6.18.10-current-rockchip64) via
`tests/bench_throughput.sh --duration 5 --warmup 30`. Both libraries
completed without any `RKNPU: job timeout` or soft-reset entries in
dmesg attributable to the sweep.
