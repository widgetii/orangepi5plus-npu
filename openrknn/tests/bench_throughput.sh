#!/usr/bin/env bash
# bench_throughput.sh — drive bench_throughput across (lib, model,
# strategy, N) combinations and assemble markdown tables + raw JSON.
#
# Usage:
#   bash tests/bench_throughput.sh \
#       --models-dir /root/npu-research \
#       [--duration 5] [--warmup 30] [--out /tmp/throughput_xxx]
#
# The script sweeps the 5 runtime models from ground_truth.json
# (classification/detection/segmentation — skips fp16-only parse
# entries), both libraries (vendor default, openrknn OWN via
# LD_PRELOAD), both strategies (pinned, multicore), and
# N ∈ {1,2,3,4,6,8}. Per combination it runs bench_throughput --json
# and records the single JSON row emitted.
#
# Output layout:
#   $OUT/
#     raw.jsonl                # one JSON object per run
#     table_<model>_<strat>.md # per-model/strategy markdown table
#     summary.md               # concatenated tables + key numbers
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OPENRKNN_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

MODELS_DIR="/root/npu-research"
DURATION=5
WARMUP=30
OUT="/tmp/throughput_$(date +%Y%m%dT%H%M%S)"
WORKERS=(1 2 3 4 6 8)

while [[ $# -gt 0 ]]; do
    case "$1" in
        --models-dir) MODELS_DIR="$2"; shift 2 ;;
        --duration)   DURATION="$2";   shift 2 ;;
        --warmup)     WARMUP="$2";     shift 2 ;;
        --out)        OUT="$2";        shift 2 ;;
        -h|--help)
            sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "unknown arg: $1" >&2; exit 1 ;;
    esac
done

BENCH="$OPENRKNN_DIR/tests/bench_throughput"
SHIM="$OPENRKNN_DIR/librknn_api.so"

if [[ ! -x "$BENCH" ]]; then
    echo "missing $BENCH — run 'make tests/bench_throughput' first" >&2
    exit 1
fi
if [[ ! -f "$SHIM" ]]; then
    echo "missing $SHIM — run 'make' first" >&2
    exit 1
fi

# Runtime models only (skip fp16_parse entries that can't actually run).
MODELS=(mobilenet_v1 resnet50 yolov5 yolov8 deeplabv3)
declare -A MODEL_FILE=(
    [mobilenet_v1]="mobilenet_v1.rknn"
    [resnet50]="resnet50-v2-7.rknn"
    [yolov5]="yolov5s_relu_int8.rknn"
    [yolov8]="yolov8.rknn"
    [deeplabv3]="deeplabv3.rknn"
)

mkdir -p "$OUT"
RAW="$OUT/raw.jsonl"
: > "$RAW"

# Set performance governor on A76 cluster (cpus 4-7); restore on exit.
ORIG_GOV=""
if [[ -r /sys/devices/system/cpu/cpu4/cpufreq/scaling_governor ]]; then
    ORIG_GOV="$(cat /sys/devices/system/cpu/cpu4/cpufreq/scaling_governor)"
    for c in 4 5 6 7; do
        echo performance > /sys/devices/system/cpu/cpu$c/cpufreq/scaling_governor 2>/dev/null || true
    done
fi
restore_gov() {
    if [[ -n "$ORIG_GOV" ]]; then
        for c in 4 5 6 7; do
            echo "$ORIG_GOV" > /sys/devices/system/cpu/cpu$c/cpufreq/scaling_governor 2>/dev/null || true
        done
    fi
}
trap restore_gov EXIT

run_one() {
    local lib="$1" model="$2" strat="$3" n="$4"
    local model_path="$MODELS_DIR/${MODEL_FILE[$model]}"
    local label="${lib}"
    local out
    if [[ "$lib" == "vendor" ]]; then
        out=$("$BENCH" --model "$model_path" --workers "$n" \
                       --duration "$DURATION" --warmup "$WARMUP" \
                       --strategy "$strat" --json --label "$label" 2>/dev/null) || return 1
    else
        out=$(LD_PRELOAD="$SHIM" ORKNN_OWN=init,query,input,run,outputs \
              "$BENCH" --model "$model_path" --workers "$n" \
                       --duration "$DURATION" --warmup "$WARMUP" \
                       --strategy "$strat" --json --label "$label" 2>/dev/null) || return 1
    fi
    # Insert the model shortname into the JSON so post-processing can
    # group by it without re-parsing the model path.
    out="${out/\{/\{\"model_name\":\"$model\",}"
    printf '%s\n' "$out" >> "$RAW"
}

TOTAL=$(( ${#MODELS[@]} * 2 * 2 * ${#WORKERS[@]} ))
COUNTER=0

for model in "${MODELS[@]}"; do
    for strat in pinned multicore; do
        for n in "${WORKERS[@]}"; do
            for lib in vendor openrknn; do
                COUNTER=$((COUNTER + 1))
                printf '[%d/%d] %s / %s / %s / N=%d ... ' \
                    "$COUNTER" "$TOTAL" "$lib" "$model" "$strat" "$n"
                if run_one "$lib" "$model" "$strat" "$n"; then
                    echo ok
                else
                    echo FAIL
                fi
            done
        done
    done
done

# Assemble markdown tables. One table per (model, strategy),
# interleaving vendor / openrknn rows for each N.
python3 - <<'PY' "$RAW" "$OUT"
import json, sys, os, collections

raw_path, out_dir = sys.argv[1], sys.argv[2]
rows = []
with open(raw_path) as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))

def key(r): return (r["model_name"], r["strategy"], r["workers"], r["label"])
by_key = {key(r): r for r in rows}
models     = sorted({r["model_name"] for r in rows})
strategies = ["pinned", "multicore"]
ns         = sorted({r["workers"] for r in rows})

summary = []
for model in models:
    summary.append(f"## {model}\n")
    for strat in strategies:
        summary.append(f"### {strat}\n")
        summary.append("| N | vendor FPS | vendor p50/p95/p99 ms | openrknn FPS | openrknn p50/p95/p99 ms | Δ% |")
        summary.append("|--:|-----------:|-----------------------|-------------:|-------------------------|---:|")
        for n in ns:
            v = by_key.get((model, strat, n, "vendor"))
            o = by_key.get((model, strat, n, "openrknn"))
            if not v or not o:
                continue
            delta = (o["fps"] - v["fps"]) / v["fps"] * 100.0 if v["fps"] else 0.0
            summary.append(
                f"| {n} | {v['fps']:.1f} | {v['p50_ms']:.2f} / {v['p95_ms']:.2f} / {v['p99_ms']:.2f} | "
                f"{o['fps']:.1f} | {o['p50_ms']:.2f} / {o['p95_ms']:.2f} / {o['p99_ms']:.2f} | "
                f"{delta:+.1f} |"
            )
        summary.append("")

out_md = os.path.join(out_dir, "summary.md")
with open(out_md, "w") as f:
    f.write("\n".join(summary) + "\n")
print(f"wrote {out_md}")
PY

echo
echo "Done. Raw JSON: $RAW"
echo "Summary:       $OUT/summary.md"
