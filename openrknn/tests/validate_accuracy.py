#!/usr/bin/env python3
"""
Real-world accuracy validation for openrknn.

Loads test images, runs inference via the RKNN C API (through ctypes),
applies model-specific post-processing (softmax, NMS, argmax), and
validates that the semantic output matches known-good ground truth.

Usage:
  # With openrknn (must have intercept dump in /tmp/rknn_dump/):
  python3 validate_accuracy.py --lib ./librknn_api.so

  # With vendor librknnrt (baseline):
  python3 validate_accuracy.py --lib /lib/librknnrt.so

  # Single model:
  python3 validate_accuracy.py --lib ./librknn_api.so --only mobilenet_v1

  # Compare vendor vs openrknn:
  python3 validate_accuracy.py --compare
"""
import argparse, ctypes, json, os, struct, subprocess, sys
import numpy as np
from pathlib import Path

# ── ctypes RKNN API wrapper ─────────────────────────────────────────────

# Enum values
RKNN_TENSOR_UINT8 = 3
RKNN_TENSOR_NHWC = 1
RKNN_QUERY_IN_OUT_NUM = 0
RKNN_QUERY_INPUT_ATTR = 1
RKNN_QUERY_OUTPUT_ATTR = 2

class rknn_input_output_num(ctypes.Structure):
    _fields_ = [("n_input", ctypes.c_uint32), ("n_output", ctypes.c_uint32)]

class rknn_tensor_attr(ctypes.Structure):
    _fields_ = [
        ("index", ctypes.c_uint32),
        ("n_dims", ctypes.c_uint32),
        ("dims", ctypes.c_uint32 * 16),
        ("name", ctypes.c_char * 256),
        ("n_elems", ctypes.c_uint32),
        ("size", ctypes.c_uint32),
        ("fmt", ctypes.c_int32),
        ("type", ctypes.c_int32),
        ("qnt_type", ctypes.c_int32),
        ("fl", ctypes.c_int8),
        ("_pad0", ctypes.c_uint8 * 3),
        ("zp", ctypes.c_int32),
        ("scale", ctypes.c_float),
        ("w_stride", ctypes.c_uint32),
        ("size_with_stride", ctypes.c_uint32),
        ("pass_through", ctypes.c_uint8),
        ("_pad1", ctypes.c_uint8 * 3),
        ("h_stride", ctypes.c_uint32),
    ]

class rknn_input_st(ctypes.Structure):
    _fields_ = [
        ("index", ctypes.c_uint32),
        ("_pad0", ctypes.c_uint32),  # alignment for pointer
        ("buf", ctypes.c_void_p),
        ("size", ctypes.c_uint32),
        ("pass_through", ctypes.c_uint8),
        ("_pad1", ctypes.c_uint8 * 3),
        ("type", ctypes.c_int32),
        ("fmt", ctypes.c_int32),
    ]

class rknn_output_st(ctypes.Structure):
    _fields_ = [
        ("want_float", ctypes.c_uint8),
        ("is_prealloc", ctypes.c_uint8),
        ("_pad0", ctypes.c_uint8 * 2),
        ("index", ctypes.c_uint32),
        ("buf", ctypes.c_void_p),
        ("size", ctypes.c_uint32),
        ("_pad1", ctypes.c_uint32),  # alignment
    ]

class RKNNLib:
    """Thin ctypes wrapper around librknn_api.so / librknnrt.so."""

    def __init__(self, lib_path):
        self.lib = ctypes.CDLL(lib_path)
        self.lib_path = lib_path

    def init(self, model_path):
        data = open(model_path, "rb").read()
        buf = ctypes.create_string_buffer(data)
        ctx = ctypes.c_uint64(0)
        ret = self.lib.rknn_init(
            ctypes.byref(ctx), buf, ctypes.c_uint32(len(data)),
            ctypes.c_uint32(0), None
        )
        if ret != 0:
            raise RuntimeError(f"rknn_init failed: {ret}")
        # Honour optional ORKNN_CORE_MASK override — used by CI tests
        # that parametrize over single-core vs multi-core execution.
        cm = os.environ.get("ORKNN_CORE_MASK")
        if cm:
            mask_val = int(cm, 0)
            self.lib.rknn_set_core_mask(ctx, ctypes.c_uint32(mask_val))
        return ctx

    def query_io_num(self, ctx):
        info = rknn_input_output_num()
        ret = self.lib.rknn_query(
            ctx, ctypes.c_int32(RKNN_QUERY_IN_OUT_NUM),
            ctypes.byref(info), ctypes.c_uint32(ctypes.sizeof(info))
        )
        if ret != 0:
            raise RuntimeError(f"rknn_query IN_OUT_NUM failed: {ret}")
        return info.n_input, info.n_output

    def query_attr(self, ctx, cmd, idx):
        attr = rknn_tensor_attr()
        attr.index = idx
        ret = self.lib.rknn_query(
            ctx, ctypes.c_int32(cmd),
            ctypes.byref(attr), ctypes.c_uint32(ctypes.sizeof(attr))
        )
        if ret != 0:
            raise RuntimeError(f"rknn_query cmd={cmd} idx={idx} failed: {ret}")
        dims = [attr.dims[i] for i in range(attr.n_dims)]
        return {
            "dims": dims, "n_dims": attr.n_dims, "n_elems": attr.n_elems,
            "size": attr.size, "fmt": attr.fmt, "type": attr.type,
            "qnt_type": attr.qnt_type, "scale": attr.scale, "zp": attr.zp,
            "name": attr.name.decode("utf-8", errors="replace").rstrip("\x00"),
        }

    def run_inference(self, ctx, input_uint8, n_outputs, want_float=True):
        """Set input, run, get outputs. Returns list of numpy arrays."""
        # Set input
        inp = rknn_input_st()
        inp.index = 0
        inp.buf = input_uint8.ctypes.data
        inp.size = input_uint8.nbytes
        inp.pass_through = 0
        inp.type = RKNN_TENSOR_UINT8
        inp.fmt = RKNN_TENSOR_NHWC
        ret = self.lib.rknn_inputs_set(ctx, ctypes.c_uint32(1), ctypes.byref(inp))
        if ret != 0:
            raise RuntimeError(f"rknn_inputs_set failed: {ret}")

        # Run
        ret = self.lib.rknn_run(ctx, None)
        if ret != 0:
            raise RuntimeError(f"rknn_run failed: {ret}")

        # Get outputs
        outputs_arr = (rknn_output_st * n_outputs)()
        for i in range(n_outputs):
            outputs_arr[i].want_float = 1 if want_float else 0
            outputs_arr[i].is_prealloc = 0
            outputs_arr[i].index = i
        ret = self.lib.rknn_outputs_get(
            ctx, ctypes.c_uint32(n_outputs), outputs_arr, None
        )
        if ret != 0:
            raise RuntimeError(f"rknn_outputs_get failed: {ret}")

        results = []
        for i in range(n_outputs):
            sz = outputs_arr[i].size
            buf = outputs_arr[i].buf
            if not buf or sz == 0:
                # openrknn returned an empty buffer for this output (e.g.,
                # sig search failed to find a valid offset). Return zeros
                # so the caller can still run post-processing without
                # segfaulting on a NULL dereference.
                if want_float:
                    results.append(np.zeros(sz // 4, dtype=np.float32))
                else:
                    results.append(np.zeros(sz, dtype=np.uint8))
                continue
            if want_float:
                arr = np.ctypeslib.as_array(
                    (ctypes.c_float * (sz // 4)).from_address(buf)
                ).copy()
            else:
                arr = np.ctypeslib.as_array(
                    (ctypes.c_uint8 * sz).from_address(buf)
                ).copy()
            results.append(arr)

        self.lib.rknn_outputs_release(
            ctx, ctypes.c_uint32(n_outputs), outputs_arr
        )
        return results

    def destroy(self, ctx):
        self.lib.rknn_destroy(ctx)


# ── Image preprocessing ─────────────────────────────────────────────────

def load_image(path, target_size):
    """Load image, resize to target_size, return uint8 NHWC [1, H, W, 3]."""
    # Try PIL first, fall back to raw numpy
    try:
        from PIL import Image
        img = Image.open(path).convert("RGB")
        img = img.resize((target_size[1], target_size[0]), Image.BILINEAR)
        arr = np.array(img, dtype=np.uint8)
    except ImportError:
        # Fallback: use cv2
        import cv2
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (target_size[1], target_size[0]))
        arr = img.astype(np.uint8)
    return arr.reshape(1, target_size[0], target_size[1], 3)


# ── Validation logic ────────────────────────────────────────────────────

def validate_classification(lib, ctx, config, images_dir, models_dir):
    from postprocess import classify_top_k
    img_path = os.path.join(images_dir, config["image"])
    if not os.path.exists(img_path):
        return "SKIP", f"image not found: {img_path}"

    model_path = os.path.join(models_dir, config["model"])
    input_img = load_image(img_path, config["input_size"])

    n_in, n_out = lib.query_io_num(ctx)
    out_attr = lib.query_attr(ctx, RKNN_QUERY_OUTPUT_ATTR, 0)

    outputs = lib.run_inference(ctx, input_img.flatten(), n_out, want_float=True)
    result = classify_top_k(outputs[0])

    expect = config["expect"]
    top1 = result["top1_class"]
    top5 = result["classes"]
    score = result["top1_score"]

    details = f"top1={top1} score={score:.3f} top5={top5}"

    # Check: top1 in expected range, or any expected class in top5
    if top1 in expect.get("top1_class_in", []):
        return "PASS", details
    for c in expect.get("top5_must_include", []):
        if c in top5:
            return "PASS", details + f" (expected {c} in top5)"
    for c in expect.get("top5_must_include_any", []):
        if c in top5:
            return "PASS", details + f" (expected {c} in top5)"
    return "FAIL", details


def validate_detection(lib, ctx, config, images_dir, models_dir, det_type):
    from postprocess import yolov5_postprocess, yolov8_postprocess, COCO_CLASSES
    img_path = os.path.join(images_dir, config["image"])
    if not os.path.exists(img_path):
        return "SKIP", f"image not found: {img_path}"

    input_img = load_image(img_path, config["input_size"])
    n_in, n_out = lib.query_io_num(ctx)

    # Query output shapes for reshape
    out_attrs = [lib.query_attr(ctx, RKNN_QUERY_OUTPUT_ATTR, i) for i in range(n_out)]
    outputs = lib.run_inference(ctx, input_img.flatten(), n_out, want_float=True)

    # Reshape outputs to their declared dims
    reshaped = []
    for i, out in enumerate(outputs):
        dims = out_attrs[i]["dims"]
        try:
            reshaped.append(out.reshape(dims))
        except ValueError:
            reshaped.append(out)  # keep flat if reshape fails

    if det_type == "detection_v5":
        boxes, classes, scores = yolov5_postprocess(
            reshaped, config["input_size"])
    else:
        boxes, classes, scores = yolov8_postprocess(
            reshaped, config["input_size"])

    expect = config["expect"]
    detected_names = set()
    det_strs = []
    for j in range(min(len(classes), 10)):
        name = COCO_CLASSES[classes[j]] if classes[j] < len(COCO_CLASSES) else f"cls{classes[j]}"
        detected_names.add(name)
        det_strs.append(f"{name}({scores[j]:.2f})")

    n_det = len(classes)
    details = f"{n_det} detections: {' '.join(det_strs[:5])}"

    required = set(expect.get("required_classes", []))
    min_det = expect.get("min_detections", 1)

    if n_det < min_det:
        return "FAIL", details + f" (need >={min_det})"
    missing = required - detected_names
    if missing:
        return "FAIL", details + f" (missing: {missing})"
    return "PASS", details


def validate_segmentation(lib, ctx, config, images_dir, models_dir):
    from postprocess import deeplabv3_postprocess
    img_path = os.path.join(images_dir, config["image"])
    if not os.path.exists(img_path):
        return "SKIP", f"image not found: {img_path}"

    input_img = load_image(img_path, config["input_size"])
    n_in, n_out = lib.query_io_num(ctx)
    out_attr = lib.query_attr(ctx, RKNN_QUERY_OUTPUT_ATTR, 0)

    outputs = lib.run_inference(ctx, input_img.flatten(), n_out, want_float=True)
    dims = out_attr["dims"]
    out_reshaped = outputs[0].reshape(dims)

    seg_map = deeplabv3_postprocess(out_reshaped, config["input_size"])
    unique, counts = np.unique(seg_map, return_counts=True)
    total = seg_map.size
    class_fracs = {int(u): float(c / total) for u, c in zip(unique, counts)}

    expect = config["expect"]
    details_parts = []
    for u in sorted(class_fracs, key=lambda x: -class_fracs[x])[:5]:
        details_parts.append(f"cls{u}({class_fracs[u]*100:.1f}%)")
    details = "classes: " + " ".join(details_parts)

    # Check expected classes present
    for c in expect.get("classes_present", []):
        if c not in class_fracs:
            return "FAIL", details + f" (class {c} not found)"

    # Check minimum pixel fractions
    for c_str, min_frac in expect.get("min_pixel_fraction", {}).items():
        c = int(c_str)
        if class_fracs.get(c, 0) < min_frac:
            return "FAIL", details + f" (class {c} frac {class_fracs.get(c,0):.3f} < {min_frac})"

    # Check minimum number of distinct classes
    min_classes = expect.get("min_classes", 0)
    if len(class_fracs) < min_classes:
        return "FAIL", details + f" (only {len(class_fracs)} classes, need >={min_classes})"

    return "PASS", details


def validate_fp16_parse(lib, ctx, config):
    n_in, n_out = lib.query_io_num(ctx)
    expect = config["expect"]
    in_attr = lib.query_attr(ctx, RKNN_QUERY_INPUT_ATTR, 0)

    details_parts = []
    tp = in_attr["type"]
    tp_names = {0: "FP32", 1: "FP16", 2: "INT8", 3: "UINT8"}
    details_parts.append(f"type={tp_names.get(tp, tp)}")
    details_parts.append(f"dims={'x'.join(str(d) for d in in_attr['dims'])}")

    if "input_type" in expect and in_attr["type"] != expect["input_type"]:
        return "FAIL", " ".join(details_parts) + f" (expected type={expect['input_type']})"

    return "PASS", " ".join(details_parts)


# ── Main ─────────────────────────────────────────────────────────────────

DUMP_DIR = "/tmp/rknn_dump"


def _clean_dump_dir(path=DUMP_DIR):
    """Fully clean path so bench_rknn starts with an empty directory.

    Replaces the previous `sh -c "rm -rf /tmp/rknn_dump/* 2>/dev/null"` which
    had two failure modes: (1) the shell glob doesn't expand if the dir is
    empty or missing, leaving stale files if the listing is racy, and
    (2) errors are silently swallowed, so a leftover file from a prior CI
    run could survive into the next model's intercept dump and mislead
    openrknn's sig search — the exact cause of the intermittent
    Phase 2 mobilenet_v1 failure on master CI.

    This version:
      * Creates the directory if missing (mkdir -p).
      * Walks the top-level entries with os.scandir, removes each file
        directly (os.remove) and each sub-tree with shutil.rmtree.
      * Follows the symlink for path itself (self-hosted runner points
        /tmp/rknn_dump at /root/rknn_dump to avoid filling tmpfs), but
        does NOT follow symlinks for entries — removes the link itself.
      * Forces the filesystem to flush the unlinks with os.sync() so a
        subsequent bench_rknn + readback on the same dir can't see stale
        dirents.
      * Verifies the dir is empty after cleanup and raises if not.
    """
    os.makedirs(path, exist_ok=True)
    import shutil
    # Resolve symlink so os.scandir walks the real target, not the link.
    real = os.path.realpath(path)
    for entry in os.scandir(real):
        try:
            if entry.is_dir(follow_symlinks=False):
                shutil.rmtree(entry.path)
            else:
                os.remove(entry.path)
        except FileNotFoundError:
            pass
    os.sync()
    leftover = os.listdir(real)
    if leftover:
        raise RuntimeError(
            f"_clean_dump_dir: {real} still has {len(leftover)} entries "
            f"after cleanup (first 5: {leftover[:5]})"
        )


def populate_intercept_dump(model_path, bench_dir):
    """Run bench_rknn under intercept_swap.so to populate /tmp/rknn_dump/.
    Required for openrknn 'own' run path which reads pre-patched regcmd
    from the intercept dump."""
    bench = os.path.join(bench_dir, "bench_rknn")
    intercept = os.path.join(bench_dir, "intercept_swap.so")
    if not os.path.exists(bench) or not os.path.exists(intercept):
        return False
    env = os.environ.copy()
    env["LD_PRELOAD"] = intercept
    env["DUMP_ALL_BOS"] = "1"
    # Fully clean /tmp/rknn_dump before regenerating. A previous CI run
    # may have left a post{N}_bo_*.bin for a different model with a
    # HIGHER submit index than the current model produces — openrknn
    # picks post{max} during output discovery and would sig-search
    # against garbage, false-matching on the wrong BO and returning
    # stale bytes from our never-written output_bos[].
    _clean_dump_dir()
    proc = subprocess.run(
        [bench, model_path, "1"],
        env=env, capture_output=True, cwd=bench_dir, timeout=600
    )
    if proc.returncode != 0:
        return False
    # Sanity check: at least one post*_bo_*.bin exists for this model.
    # If bench_rknn silently produced no dumps, openrknn would see
    # whatever was there before (which our clean just wiped, so now
    # nothing) and fail with a cleaner error than sig-search garbage.
    posts = [f for f in os.listdir(os.path.realpath(DUMP_DIR))
             if f.startswith("post") and f.endswith(".bin")]
    if not posts:
        return False
    return True


def run_validation(lib_path, models_dir, images_dir, ground_truth,
                   only=None, populate_dumps=False, bench_dir=None):
    lib = RKNNLib(lib_path)
    results = []

    for name, config in ground_truth.items():
        if only and name != only:
            continue

        model_path = os.path.join(models_dir, config["model"])
        if not os.path.exists(model_path):
            results.append((name, config.get("image", "—"),
                           config["type"], "SKIP", f"model not found"))
            continue

        # Populate intercept dump for openrknn's 'own' run path
        if populate_dumps and bench_dir:
            populate_intercept_dump(model_path, bench_dir)

        try:
            ctx = lib.init(model_path)
        except RuntimeError as e:
            results.append((name, config.get("image", "—"),
                           config["type"], "FAIL", str(e)))
            continue

        try:
            t = config["type"]
            if t == "classification":
                status, details = validate_classification(
                    lib, ctx, config, images_dir, models_dir)
            elif t in ("detection_v5", "detection_v8"):
                status, details = validate_detection(
                    lib, ctx, config, images_dir, models_dir, t)
            elif t == "segmentation":
                status, details = validate_segmentation(
                    lib, ctx, config, images_dir, models_dir)
            elif t == "fp16_parse":
                status, details = validate_fp16_parse(lib, ctx, config)
            else:
                status, details = "SKIP", f"unknown type: {t}"
        except Exception as e:
            status, details = "FAIL", f"exception: {e}"

        results.append((name, config.get("image", "—"), config["type"],
                        status, details))
        lib.destroy(ctx)

    return results


def print_results(results, lib_path):
    print(f"\n{'=' * 72}")
    print(f"  OpenRKNN Accuracy Validation")
    print(f"  Library: {lib_path}")
    print(f"{'=' * 72}\n")

    hdr = f"{'Model':<22} {'Image':<18} {'Type':<16} {'Result':<6} Details"
    print(hdr)
    print("─" * 72)

    n_pass = n_fail = n_skip = 0
    for name, image, typ, status, details in results:
        mark = {"PASS": "\033[32mPASS\033[0m",
                "FAIL": "\033[31mFAIL\033[0m",
                "SKIP": "\033[33mSKIP\033[0m"}.get(status, status)
        print(f"{name:<22} {image:<18} {typ:<16} {mark:<15} {details}")
        if status == "PASS": n_pass += 1
        elif status == "FAIL": n_fail += 1
        else: n_skip += 1

    print(f"\n{n_pass} passed, {n_fail} failed, {n_skip} skipped "
          f"out of {len(results)} tests\n")
    return 0 if n_fail == 0 else 1


def main():
    parser = argparse.ArgumentParser(description="OpenRKNN accuracy validation")
    parser.add_argument("--lib", default="./librknn_api.so",
                       help="Path to librknn_api.so or librknnrt.so")
    parser.add_argument("--models-dir", default="/root/npu-research",
                       help="Directory containing .rknn model files")
    parser.add_argument("--images-dir", default=None,
                       help="Directory containing test images")
    parser.add_argument("--ground-truth", default=None,
                       help="Path to ground_truth.json")
    parser.add_argument("--only", default=None,
                       help="Run only this model name")
    parser.add_argument("--compare", action="store_true",
                       help="Compare vendor vs openrknn")
    parser.add_argument("--populate-dumps", action="store_true",
                       help="Run bench_rknn to populate /tmp/rknn_dump before each model "
                            "(needed for openrknn 'own' run path)")
    parser.add_argument("--bench-dir", default="/root/npu-research/librocketnpu/tests",
                       help="Directory containing bench_rknn + intercept_swap.so")
    parser.add_argument("--own", default=None,
                       help="Set ORKNN_OWN env var (e.g. 'init,query,input,run,outputs')")
    args = parser.parse_args()

    if args.own:
        os.environ["ORKNN_OWN"] = args.own

    # Find ground truth
    script_dir = Path(__file__).parent
    gt_path = args.ground_truth or str(script_dir / "ground_truth.json")
    with open(gt_path) as f:
        ground_truth = json.load(f)

    # Find images
    images_dir = args.images_dir
    if not images_dir:
        candidates = [
            str(script_dir / "test_images"),
            "/root/npu-research/test_images",
            str(script_dir.parent / "tests" / "test_images"),
        ]
        for c in candidates:
            if os.path.isdir(c):
                images_dir = c
                break
        if not images_dir:
            images_dir = str(script_dir / "test_images")

    if args.compare:
        print("=== Vendor library (baseline) ===")
        vendor_results = run_validation(
            "/lib/librknnrt.so", args.models_dir, images_dir, ground_truth, args.only)
        rc1 = print_results(vendor_results, "/lib/librknnrt.so")

        print("\n=== OpenRKNN library ===")
        os.environ["ORKNN_OWN"] = args.own or "init,query,input,run,outputs"
        own_results = run_validation(
            args.lib, args.models_dir, images_dir, ground_truth, args.only,
            populate_dumps=True, bench_dir=args.bench_dir)
        rc2 = print_results(own_results, args.lib)
        return max(rc1, rc2)

    results = run_validation(args.lib, args.models_dir, images_dir,
                            ground_truth, args.only,
                            populate_dumps=args.populate_dumps,
                            bench_dir=args.bench_dir)
    return print_results(results, args.lib)


if __name__ == "__main__":
    sys.exit(main())
