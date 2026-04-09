#!/bin/bash
# openrknn CI validation — runs on the Orange Pi 5 Plus self-hosted runner.
#
# Builds openrknn natively, runs the real-image validation suite against
# both the vendor librknnrt.so (baseline) and our librknn_api.so (OWN path).
#
# Requires:
#   - /root/npu-research/<model>.rknn for each model in ground_truth.json
#   - /root/npu-research/librocketnpu/tests/{bench_rknn,intercept_swap.so}
#     (used by --populate-dumps to seed /tmp/rknn_dump per model)
#   - /lib/librknnrt.so (vendor lib)
#   - python3 + PIL (pillow)
#
# Exit codes: 0 = all pass, 1 = test failure, 2 = setup error
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
OPENRKNN_DIR="$REPO_ROOT/openrknn"
MODEL_DIR="/root/npu-research"
BENCH_DIR="/root/npu-research/librocketnpu/tests"
IMAGES_DIR="$SCRIPT_DIR/test_images"
GT="$SCRIPT_DIR/ground_truth.json"

echo "=== openrknn CI validation ==="
echo "Host: $(hostname)"
echo "Kernel: $(uname -r)"
echo "Repo: $REPO_ROOT"

# --- Setup sanity checks ---

if [ ! -f "/lib/librknnrt.so" ]; then
    echo "ERROR: /lib/librknnrt.so not found (vendor library required)"
    exit 2
fi

if [ ! -d "$IMAGES_DIR" ] || [ -z "$(ls -A "$IMAGES_DIR" 2>/dev/null)" ]; then
    echo "ERROR: test images missing at $IMAGES_DIR"
    exit 2
fi

if [ ! -f "$GT" ]; then
    echo "ERROR: ground_truth.json missing at $GT"
    exit 2
fi

# Parse model filenames from ground_truth.json (simple grep, avoids jq dep)
MISSING=0
while read -r model; do
    [ -z "$model" ] && continue
    if [ ! -f "$MODEL_DIR/$model" ]; then
        echo "ERROR: missing model: $MODEL_DIR/$model"
        MISSING=1
    fi
done < <(python3 -c "
import json
with open('$GT') as f:
    gt = json.load(f)
for entry in gt.values():
    print(entry['model'])
")
if [ "$MISSING" -ne 0 ]; then
    exit 2
fi

if [ ! -x "$BENCH_DIR/bench_rknn" ] || [ ! -f "$BENCH_DIR/intercept_swap.so" ]; then
    echo "ERROR: bench_rknn or intercept_swap.so missing at $BENCH_DIR"
    echo "  (needed for --populate-dumps — the OWN run path reads /tmp/rknn_dump/)"
    exit 2
fi

python3 -c "from PIL import Image" 2>/dev/null || {
    echo "ERROR: python3 PIL (pillow) not installed on board"
    echo "  Install with: pip3 install --break-system-packages pillow"
    exit 2
}

# --- Build openrknn ---

echo ""
echo "=== Building openrknn ==="
make -C "$OPENRKNN_DIR" clean >/dev/null 2>&1 || true
make -C "$OPENRKNN_DIR"
LIB="$OPENRKNN_DIR/librknn_api.so"
[ -f "$LIB" ] || { echo "ERROR: build did not produce $LIB"; exit 2; }

# Get list of model names from ground_truth.json
MODELS=$(python3 -c "import json; print(' '.join(json.load(open('$GT')).keys()))")

# Helper: run validate_accuracy.py for a single model. Returns 0 on pass,
# non-zero on failure. Segfaults in one model don't abort the suite —
# each model runs in its own Python process for isolation.
run_one() {
    local lib="$1"; local name="$2"; shift 2
    python3 "$SCRIPT_DIR/validate_accuracy.py" \
        --lib "$lib" \
        --models-dir "$MODEL_DIR" \
        --images-dir "$IMAGES_DIR" \
        --ground-truth "$GT" \
        --only "$name" \
        "$@" 2>&1 | tail -15
    return ${PIPESTATUS[0]}
}

run_phase() {
    local phase_name="$1"; local lib="$2"; shift 2
    echo ""
    echo "=== $phase_name ==="
    local n_pass=0 n_fail=0 n_total=0
    local failures=""
    for name in $MODELS; do
        n_total=$((n_total + 1))
        echo ""
        echo "--- $name ---"
        if run_one "$lib" "$name" "$@"; then
            n_pass=$((n_pass + 1))
        else
            n_fail=$((n_fail + 1))
            failures="$failures $name"
        fi
    done
    echo ""
    echo "  Phase result: $n_pass/$n_total passed"
    if [ -n "$failures" ]; then
        echo "  Failed:$failures"
        return 1
    fi
    return 0
}

# --- Phase 1: vendor baseline ---
# Ensures the models + test images produce the expected ground-truth
# classes. If this fails, the ground_truth.json is wrong or a model is
# corrupt — not an openrknn regression.

overall_rc=0
run_phase "Phase 1: vendor librknnrt.so baseline" /lib/librknnrt.so || overall_rc=1

# --- Phase 2: openrknn OWN path ---
# Exercises every openrknn code path: init, query, inputs_set (UINT8
# NHWC → NC1HWC2 with correct c2), run (regcmd rebase + NPU submit),
# outputs_get (detile + dequantize with want_float=1).
# --populate-dumps runs bench_rknn first per model to seed /tmp/rknn_dump
# which openrknn's copy_proxy_regcmd reads.

run_phase "Phase 2: openrknn OWN path (full pipeline)" "$LIB" \
    --populate-dumps --bench-dir "$BENCH_DIR" \
    --own init,query,input,run,outputs || overall_rc=1

# --- Phase 3: openrknn proxy-dispatch path ---
# The default user-facing mode — LD_PRELOAD=./librknn_api.so without
# any ORKNN_OWN env var. This exercises the proxy delegation path which
# existing applications use when upgrading to openrknn.

run_phase "Phase 3: openrknn proxy-dispatch path" "$LIB" || overall_rc=1

echo ""
if [ "$overall_rc" -eq 0 ]; then
    echo "=== All openrknn CI validation tests passed ==="
else
    echo "=== openrknn CI validation FAILED (see phase results above) ==="
fi
exit "$overall_rc"
