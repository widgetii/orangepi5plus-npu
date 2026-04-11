#!/usr/bin/env python3
"""
bench_openrknn — latency benchmark for openrknn vs vendor librknnrt.so.

Runs `librocketnpu/tests/bench_rknn` (which calls the RKNN API in a loop
and prints Latency: avg/min) under three configurations per model:

  1. vendor   — bench_rknn straight, SONAME-resolved /lib/librknnrt.so
  2. proxy    — LD_PRELOAD=openrknn/librknn_api.so, no ORKNN_OWN set
                → openrknn's proxy-dispatch path, dlopens the vendor
                  lib under the hood
  3. own      — LD_PRELOAD=openrknn/librknn_api.so,
                ORKNN_OWN=init,query,input,run,outputs
                → openrknn's from-scratch path, no vendor runtime code
                  involved in inference (for runtime models). fp16_parse
                  models still fall through to proxy internally because
                  extract_npu_data returns early.

Output: a markdown table suitable for pasting into docs/benchmarks.md.

Example:
  python3 tests/bench_openrknn.py \\
      --bench /root/npu-research/librocketnpu/tests/bench_rknn \\
      --lib    /root/npu-research/openrknn/librknn_api.so \\
      --models-dir /root/npu-research \\
      --ground-truth tests/ground_truth.json \\
      --iters 100

Expects the performance CPU governor to be enabled by the caller.
"""
import argparse
import json
import os
import re
import subprocess
import sys


# Matches `Latency: avg=2.83 ms, min=2.83 ms, FPS=353.5`
LATENCY_RE = re.compile(
    r"Latency: avg=([0-9.]+)\s*ms, min=([0-9.]+)\s*ms, FPS=([0-9.]+)"
)


def run_bench(bench, model_path, iters, env=None):
    """Invoke bench_rknn and return (avg_ms, min_ms, fps) or None on failure."""
    full_env = os.environ.copy()
    if env:
        full_env.update(env)
    try:
        proc = subprocess.run(
            [bench, model_path, str(iters)],
            capture_output=True, env=full_env, timeout=600,
        )
    except subprocess.TimeoutExpired:
        return None
    if proc.returncode != 0:
        return None
    text = proc.stdout.decode(errors="replace") + proc.stderr.decode(errors="replace")
    m = LATENCY_RE.search(text)
    if not m:
        return None
    return float(m.group(1)), float(m.group(2)), float(m.group(3))


def bench_model(bench, lib, model_path, iters, fp16_parse=False):
    """Run all three configurations for a single model.

    For fp16_parse models (mobilesam_encoder, lprnet) OWN mode is only
    init+query — extract_npu_data returns early so the run path has no
    segments to submit. We still exercise OWN init/query and let the
    run path fall through to proxy via orknn_run; setting
    ORKNN_OWN=init,query matches the validate_accuracy.py pattern and
    measures the combined openrknn-init + vendor-run latency.
    """
    own_flags = ("init,query" if fp16_parse
                 else "init,query,input,run,outputs")
    configs = [
        ("vendor", None),
        ("proxy",  {"LD_PRELOAD": lib}),
        ("own",    {"LD_PRELOAD": lib, "ORKNN_OWN": own_flags}),
    ]
    results = {}
    for name, env in configs:
        r = run_bench(bench, model_path, iters, env=env)
        results[name] = r
    return results


def format_latency(r):
    if r is None:
        return "**FAIL**"
    avg, mn, fps = r
    return f"{avg:.2f} / {mn:.2f}"


def format_delta(own, vendor):
    """Return '(+5.2%)' style comparison string, or '—' if either is None."""
    if own is None or vendor is None:
        return "—"
    own_avg = own[0]
    vendor_avg = vendor[0]
    if vendor_avg <= 0:
        return "—"
    pct = (own_avg - vendor_avg) / vendor_avg * 100
    sign = "+" if pct >= 0 else ""
    return f"{sign}{pct:.1f}%"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", required=True,
                    help="Path to librocketnpu/tests/bench_rknn")
    ap.add_argument("--lib", required=True,
                    help="Path to openrknn/librknn_api.so")
    ap.add_argument("--models-dir", required=True)
    ap.add_argument("--ground-truth", required=True)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--only", default="",
                    help="Comma-separated model names to limit the run")
    args = ap.parse_args()

    gt = json.load(open(args.ground_truth))
    only = set(args.only.split(",")) if args.only else None

    print(f"# openrknn latency benchmark")
    print()
    print(f"Hardware: Orange Pi 5 Plus (RK3588, single NPU core)")
    print(f"Iterations per run: {args.iters} (plus bench_rknn's internal warmup)")
    print(f"CPU governor: (caller-controlled; recommended: `performance`)")
    print()
    print("| Model | vendor avg/min (ms) | proxy avg/min (ms) "
          "| own avg/min (ms) | own vs vendor |")
    print("|-------|---------------------|--------------------"
          "|------------------|---------------|")

    all_results = {}
    for name, cfg in gt.items():
        if only and name not in only:
            continue
        model_path = os.path.join(args.models_dir, cfg["model"])
        if not os.path.exists(model_path):
            print(f"| {name} | *(model missing)* | — | — | — |")
            continue
        print(f"...benching {name}", file=sys.stderr, flush=True)
        fp16 = cfg.get("type") == "fp16_parse"
        r = bench_model(args.bench, args.lib, model_path, args.iters,
                        fp16_parse=fp16)
        all_results[name] = r
        tag = "" if not fp16 else " *(fp16, OWN init+query only)*"
        print(f"| {name}{tag} | {format_latency(r['vendor'])} "
              f"| {format_latency(r['proxy'])} "
              f"| {format_latency(r['own'])} "
              f"| {format_delta(r['own'], r['vendor'])} |")

    # Summary
    print()
    print("**Interpretation:** `own vs vendor` is the avg-latency gap "
          "of openrknn OWN mode relative to the vendor baseline. "
          "Positive means openrknn is slower. Negative means faster.")

    # Exit non-zero if any run failed
    for name, r in all_results.items():
        if any(v is None for v in r.values()):
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
