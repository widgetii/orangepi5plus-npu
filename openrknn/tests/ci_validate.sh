#!/bin/bash
# openrknn CI validation — runs on the Orange Pi 5 Plus self-hosted runner.
#
# Phase 2 is the authoritative openrknn test: it runs the OWN path purely
# from the .rknn FlatBuffer on a freshly wiped /tmp/rknn_dump, with
# /lib/librknnrt.so not even required. Phases 1, 2.5, and 3 exist for
# cross-reference against the vendor stack and are skipped gracefully if
# the vendor library or bench_rknn/intercept_swap.so tooling is absent.
#
# Requires (mandatory):
#   - /root/npu-research/<model>.rknn for each model in ground_truth.json
#   - python3 + PIL (pillow)
#
# Optional (enables extra phases):
#   - /lib/librknnrt.so                                 — Phase 1, 3
#   - /root/npu-research/librocketnpu/tests/{bench_rknn,
#     intercept_swap.so}                                — Phase 2.5
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

# Capability probes for optional phases. Each is a boolean (true/false).
HAVE_VENDOR_LIB=false
HAVE_BENCH_TOOLS=false
[ -f "/lib/librknnrt.so" ] && HAVE_VENDOR_LIB=true
[ -x "$BENCH_DIR/bench_rknn" ] && [ -f "$BENCH_DIR/intercept_swap.so" ] \
    && HAVE_BENCH_TOOLS=true

echo "=== openrknn CI validation ==="
echo "Host: $(hostname)"
echo "Kernel: $(uname -r)"
echo "Repo: $REPO_ROOT"
echo "Vendor lib:  $HAVE_VENDOR_LIB"
echo "Bench tools: $HAVE_BENCH_TOOLS"

# --- Mandatory setup sanity checks ---

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

overall_rc=0

# --- Phase 1: vendor baseline ---
# Ensures the models + test images produce the expected ground-truth
# classes. If this fails, the ground_truth.json is wrong or a model is
# corrupt — not an openrknn regression. Optional: skipped when the
# vendor library isn't installed.

if $HAVE_VENDOR_LIB; then
    run_phase "Phase 1: vendor librknnrt.so baseline" /lib/librknnrt.so \
        || overall_rc=1
else
    echo ""
    echo "=== Phase 1: vendor librknnrt.so baseline (SKIPPED — no vendor lib) ==="
fi

# --- Phase 2: openrknn OWN path (no vendor deps) ---
# The authoritative openrknn test. Wipes /tmp/rknn_dump first so any
# accidental dump reads would fail loudly; runs the OWN path purely
# from the .rknn FlatBuffer. Does NOT invoke bench_rknn or touch the
# vendor library. Exercises: init, query, inputs_set (UINT8 NHWC →
# NC1HWC2), run (regcmd rebase + NPU submit), outputs_get (detile +
# dequantize with want_float=1).

if [ -e /tmp/rknn_dump ]; then
    find /tmp/rknn_dump -mindepth 1 -delete 2>/dev/null || true
    sync
fi

run_phase "Phase 2: openrknn OWN path (no vendor deps)" "$LIB" \
    --own init,query,input,run,outputs || overall_rc=1

# --- Phase 2.5: template-patch byte-exact diff vs vendor oracle ---
# Runs each model with ORKNN_DUMP_BO1 set, which dumps the patched weight BO
# after the first run. diff_regcmd.py then diffs it against the vendor-patched
# BO[1] captured in /tmp/rknn_dump/sub1_bo_001_*.bin. Non-DMA register deltas
# are real template-patch bugs.
#
# Gating: the phase fails overall_rc if any model OUTSIDE the expected-fail
# allowlist shows register deltas. Allowlisted models log their diff count
# and continue; see the comment on DIFF_ALLOWLIST for the list and why each
# entry is there.

# Models allowed to have template-patch regcmd diffs. Empty right now —
# all five runtime models in ground_truth.json are byte-exact against the
# vendor oracle. Keeping the hook here for future models; add an entry
# and document the root-cause op when a new allowlist is needed.
DIFF_ALLOWLIST=""

is_allowlisted() {
    case " $DIFF_ALLOWLIST " in
        *" $1 "*) return 0 ;;
        *) return 1 ;;
    esac
}

if ! $HAVE_BENCH_TOOLS || ! $HAVE_VENDOR_LIB; then
    echo ""
    echo "=== Phase 2.5: template-patch byte-exact diff (SKIPPED) ==="
    if ! $HAVE_VENDOR_LIB; then
        echo "  reason: /lib/librknnrt.so not installed"
    fi
    if ! $HAVE_BENCH_TOOLS; then
        echo "  reason: bench_rknn / intercept_swap.so not found in $BENCH_DIR"
    fi
    SKIP_PHASE_2_5=true
else
    SKIP_PHASE_2_5=false
fi

if ! $SKIP_PHASE_2_5; then

DIFF_DIR=/tmp/orknn_diff
mkdir -p "$DIFF_DIR"
echo ""
echo "=== Phase 2.5: template-patch byte-exact diff ==="
n_clean=0; n_dirty=0; dirty_names=""
n_allowlisted_dirty=0; allowlisted_names=""
for name in $MODELS; do
    bo1_out="$DIFF_DIR/${name}_bo1.bin"
    rm -f "$bo1_out"

    # Repopulate the vendor dump for this model (Phase 2 wiped it between
    # models for isolation). We need both sub1_bo_001 (oracle) and
    # sub1_bo_000 (task BO) for the diff.
    python3 -c "
import sys; sys.path.insert(0, '$SCRIPT_DIR')
from validate_accuracy import populate_intercept_dump
populate_intercept_dump('$MODEL_DIR/$(python3 -c "import json; print(json.load(open('$GT'))['$name']['model'])")', '$BENCH_DIR')
" 2>/dev/null || { echo "--- $name: dump populate failed (skip) ---"; continue; }

    # Run validate_accuracy on just this model with ORKNN_DUMP_BO1 writing
    # the per-model patched weight BO for diff_regcmd.py to consume.
    ORKNN_DUMP_BO1="$bo1_out" \
    python3 "$SCRIPT_DIR/validate_accuracy.py" \
        --lib "$LIB" \
        --models-dir "$MODEL_DIR" \
        --images-dir "$IMAGES_DIR" \
        --ground-truth "$GT" \
        --only "$name" \
        --own init,query,input,run,outputs >/dev/null 2>&1 || true

    if [ ! -s "$bo1_out" ]; then
        if [ "${name}" = "mobilesam_encoder" ] || [ "${name}" = "lprnet" ]; then
            # fp16_parse models don't go through the own NPU run path, so
            # there's no weight BO to dump and nothing to diff.
            continue
        fi
        echo "--- $name: template run produced no BO1 dump (skip) ---"
        continue
    fi

    oracle_bo1=$(ls /tmp/rknn_dump/sub1_bo_001_*.bin 2>/dev/null | head -1)
    task_bo=$(ls /tmp/rknn_dump/sub1_bo_000_*.bin 2>/dev/null | head -1)
    if [ -z "$oracle_bo1" ] || [ -z "$task_bo" ]; then
        echo "--- $name: vendor dump missing (skip) ---"
        continue
    fi

    echo ""
    echo "--- $name ---"
    if python3 "$SCRIPT_DIR/diff_regcmd.py" \
        --oracle "$oracle_bo1" \
        --template "$bo1_out" \
        --task-bo "$task_bo" 2>&1 | tail -30; then
        n_clean=$((n_clean + 1))
        echo "  $name: BYTE-EXACT"
    else
        if is_allowlisted "$name"; then
            n_allowlisted_dirty=$((n_allowlisted_dirty + 1))
            allowlisted_names="$allowlisted_names $name"
            echo "  $name: HAS DIFFS (allowlisted, not gating)"
        else
            n_dirty=$((n_dirty + 1))
            dirty_names="$dirty_names $name"
            echo "  $name: HAS DIFFS (gating failure)"
        fi
    fi
done
echo ""
echo "  Phase 2.5 result: $n_clean clean, $n_dirty gating, $n_allowlisted_dirty allowlisted"
if [ -n "$dirty_names" ]; then
    echo "  Template-patch regressions:$dirty_names"
    overall_rc=1
fi
if [ -n "$allowlisted_names" ]; then
    echo "  Template-patch known diffs:$allowlisted_names"
fi

fi  # SKIP_PHASE_2_5

# --- Phase 3: openrknn proxy-dispatch path ---
# The default user-facing mode — LD_PRELOAD=./librknn_api.so without
# any ORKNN_OWN env var. This exercises the proxy delegation path which
# existing applications use when upgrading to openrknn. Needs the vendor
# library since proxy delegation dlopens it.

if $HAVE_VENDOR_LIB; then
    run_phase "Phase 3: openrknn proxy-dispatch path" "$LIB" || overall_rc=1
else
    echo ""
    echo "=== Phase 3: openrknn proxy-dispatch path (SKIPPED — no vendor lib) ==="
fi

echo ""
if [ "$overall_rc" -eq 0 ]; then
    echo "=== All openrknn CI validation tests passed ==="
else
    echo "=== openrknn CI validation FAILED (see phase results above) ==="
fi
exit "$overall_rc"
