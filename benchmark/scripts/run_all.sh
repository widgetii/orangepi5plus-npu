#!/bin/bash
# Full benchmark regression suite for Rocket NPU optimization.
# Usage: ./run_all.sh [phase_name] [--baseline-only]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BENCH_DIR="$(dirname "$SCRIPT_DIR")"
RESULTS_DIR="$BENCH_DIR/results"
BASELINE_DIR="$RESULTS_DIR/baseline"

PHASE="${1:-baseline}"
BASELINE_ONLY="${2:-}"
PHASE_DIR="$RESULTS_DIR/$PHASE"

# Paths
MODEL_DIR="$BENCH_DIR/models"
DATASET_DIR="$BENCH_DIR/datasets"
DELEGATE_LIB="/usr/lib/aarch64-linux-gnu/libteflon_delegate.so"
DELEGATE_LOCAL="/usr/local/lib/aarch64-linux-gnu/libteflon_delegate.so"

# Use local build if available, else system
if [ -f "$DELEGATE_LOCAL" ]; then
    DELEGATE="$DELEGATE_LOCAL"
elif [ -f "$DELEGATE_LIB" ]; then
    DELEGATE="$DELEGATE_LIB"
else
    echo "ERROR: No Teflon delegate found"
    echo "  Checked: $DELEGATE_LOCAL"
    echo "  Checked: $DELEGATE_LIB"
    exit 1
fi

echo "=== Rocket NPU Benchmark Suite ==="
echo "Phase:    $PHASE"
echo "Delegate: $DELEGATE"
echo "Date:     $(date -Iseconds)"
echo ""

mkdir -p "$PHASE_DIR"

# Lock CPU governor to performance for reproducible results
echo "Setting CPU governor to performance..."
for cpu in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
    echo performance > "$cpu" 2>/dev/null || true
done

# Record system info
cat > "$PHASE_DIR/system_info.json" << SYSEOF
{
    "kernel": "$(uname -r)",
    "date": "$(date -Iseconds)",
    "phase": "$PHASE",
    "delegate": "$DELEGATE",
    "mesa_version": "$(pkg-config --modversion libdrm 2>/dev/null || echo unknown)",
    "cpu_governor": "$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null || echo unknown)",
    "npu_irqs_before": $(grep -c 'rocket\|rknpu' /proc/interrupts 2>/dev/null || echo 0)
}
SYSEOF

PERF_JSON="$PHASE_DIR/performance.json"
ACCURACY_JSON="$PHASE_DIR/accuracy.json"

# Initialize empty JSON arrays
echo "[]" > "$PERF_JSON"
echo "[]" > "$ACCURACY_JSON"

run_bench() {
    local model_path="$1"
    local model_name
    model_name=$(basename "$model_path" .tflite)

    if [ ! -f "$model_path" ]; then
        echo "  SKIP: $model_path not found"
        return
    fi

    echo "--- $model_name ---"

    # CPU baseline
    echo "  CPU..."
    python3 "$SCRIPT_DIR/bench_cpu.py" "$model_path" \
        -n 100 -w 10 -l "$PHASE" -o "$PERF_JSON" 2>&1 | grep -E "avg|FPS" || true

    # Rocket NPU
    echo "  Rocket NPU..."
    python3 "$SCRIPT_DIR/bench_rocket.py" "$model_path" \
        -d "$DELEGATE" -n 100 -w 10 --validate --count-ioctls \
        -l "$PHASE" -o "$PERF_JSON" 2>&1 | grep -E "avg|FPS|Correct|IOCTL" || true

    echo ""
}

echo "=== Classification models ==="
for model in "$MODEL_DIR"/classification/*.tflite; do
    [ -f "$model" ] && run_bench "$model"
done

echo "=== Detection models ==="
for model in "$MODEL_DIR"/detection/*.tflite; do
    [ -f "$model" ] && run_bench "$model"
done

echo "=== Segmentation models ==="
for model in "$MODEL_DIR"/segmentation/*.tflite; do
    [ -f "$model" ] && run_bench "$model"
done

# Also benchmark existing models in npu-research dir
echo "=== Existing models ==="
for model in /root/npu-research/zero2pro_NPU_example/*.tflite; do
    [ -f "$model" ] && run_bench "$model"
done

# NPU IRQ count after
NPU_IRQS_AFTER=$(grep -c 'rocket\|rknpu' /proc/interrupts 2>/dev/null || echo 0)
echo "NPU IRQs during benchmark: ~$NPU_IRQS_AFTER"

# Compare with baseline if not baseline run
if [ "$PHASE" != "baseline" ] && [ -f "$BASELINE_DIR/performance.json" ]; then
    echo ""
    echo "=== Regression check ==="
    python3 "$SCRIPT_DIR/compare_results.py" compare \
        "$BASELINE_DIR/performance.json" "$PERF_JSON" || {
        echo "REGRESSION DETECTED!"
        exit 1
    }
fi

echo ""
echo "=== Summary ==="
python3 "$SCRIPT_DIR/compare_results.py" summary "$PHASE_DIR"

echo ""
echo "Results saved to: $PHASE_DIR"
echo "Done."
