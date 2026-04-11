#!/usr/bin/env python3
"""
dump_segments — Capture segmentation ground truth for a .rknn model.

For each model this tool produces a JSON artifact containing:

  * vendor_submits — the per-submit {flags, sc_start, sc_count, task_number,
    task_obj_addr} captured by intercept_swap.so in /tmp/rknn_dump/submit_*.txt
  * task_bo — the full task BO contents (one record per 40-byte task) as
    captured by intercept_swap.so in sub1_bo_000_*.bin, so we can see which
    tasks each submit references
  * fb_ops — the operator graph as parsed directly from the .rknn FlatBuffer,
    including every non-empty vtable slot per operator so Task 9.2 can
    identify which FB field holds the per-op task count and submit boundary

Usage:
  python3 dump_segments.py \\
    --bench-dir /root/npu-research/librocketnpu/tests \\
    --model /root/npu-research/mobilenet_v1.rknn \\
    --out /tmp/gt/mobilenet_v1.json

  python3 dump_segments.py --all \\
    --bench-dir /root/npu-research/librocketnpu/tests \\
    --models-dir /root/npu-research \\
    --ground-truth openrknn/tests/ground_truth.json \\
    --out-dir /tmp/gt/

Task 9.1: measurement-first. This tool does not derive any segmentation
rule — it only records what the vendor does. Task 9.2 consumes the output.
"""
import argparse
import json
import os
import pathlib
import re
import shutil
import struct
import subprocess
import sys
from pathlib import Path


DUMP_DIR = "/tmp/rknn_dump"


# ── FlatBuffer helpers (mirrors openrknn/src/openrknn_flatbuf.c) ────────

def u16(b, p): return struct.unpack_from("<H", b, p)[0]
def u32(b, p): return struct.unpack_from("<I", b, p)[0]
def i32(b, p): return struct.unpack_from("<i", b, p)[0]
def u64(b, p): return struct.unpack_from("<Q", b, p)[0]


def fb_follow(b, p):
    return p + u32(b, p)


def fb_vtable(b, table):
    vt = table - i32(b, table)
    vt_size = u16(b, vt)
    offs = []
    fo = 4
    while fo < vt_size:
        offs.append(u16(b, vt + fo))
        fo += 2
    return vt, vt_size, offs


def fb_field(b, table, field):
    _, _, offs = fb_vtable(b, table)
    if field >= len(offs):
        return 0
    return table + offs[field] if offs[field] else 0


def fb_vec_len(b, fpos):
    return u32(b, fb_follow(b, fpos))


def fb_vec_at(b, fpos, idx):
    vec = fb_follow(b, fpos)
    if idx >= u32(b, vec):
        return 0
    return fb_follow(b, vec + 4 + idx * 4)


def fb_string(b, fpos, maxlen=256):
    try:
        s = fb_follow(b, fpos)
        n = u32(b, s)
        if n > maxlen:
            return None
        return bytes(b[s + 4 : s + 4 + n]).decode("utf-8", errors="replace")
    except Exception:
        return None


def classify_field(b, fb_size, fpos):
    """Try to identify what a vtable slot contains.

    Returns a dict with whatever interpretations look structurally valid:
      scalar_u8, scalar_u32 — always present (inline bytes at fpos)
      vec_u32  — if it looks like a vector of u32 scalars (length + values)
      str      — if it looks like a string (printable ASCII)
      n_tables — if it looks like a vector of table offsets
    """
    out = {"fpos": fpos}
    if fpos < fb_size:
        out["u8"] = b[fpos]
    if fpos + 4 <= fb_size:
        out["u32"] = u32(b, fpos)
        # Try follow + length
        rel = out["u32"]
        if 0 < rel < fb_size - 4:
            target = fpos + rel
            if target + 4 <= fb_size:
                n = u32(b, target)
                # String?
                if 0 < n < 256 and target + 4 + n <= fb_size:
                    payload = bytes(b[target + 4 : target + 4 + n])
                    if all(0x20 <= c <= 0x7e for c in payload):
                        out["str"] = payload.decode("ascii")
                # Vector of u32?
                if 0 < n < 4096 and target + 4 + n * 4 <= fb_size:
                    vals = [u32(b, target + 4 + i * 4) for i in range(min(n, 32))]
                    out["vec_u32_len"] = n
                    out["vec_u32"] = vals
                # Vector of tables? (first element's follow lands on a
                # valid vtable)
                if 0 < n < 4096 and target + 4 + n * 4 <= fb_size:
                    try:
                        first_off = u32(b, target + 4)
                        if 0 < first_off < fb_size:
                            tbl = target + 4 + first_off
                            if 0 < tbl < fb_size - 4:
                                vt = tbl - i32(b, tbl)
                                if 0 <= vt < fb_size - 4:
                                    vts = u16(b, vt)
                                    if 4 <= vts <= 512:
                                        out["maybe_tables_n"] = n
                    except Exception:
                        pass
    return out


# ── Parsers for /tmp/rknn_dump files ────────────────────────────────────

def parse_submit_txt(path):
    """Parse a submit_N.txt file into a dict."""
    text = open(path).read()
    m = re.search(
        r"submit=(\d+)\s+flags=0x([0-9a-fA-F]+)\s+tasks=(\d+)\s+"
        r"task_obj=0x([0-9a-fA-F]+)\s+core_mask=0x([0-9a-fA-F]+)\s+"
        r"sc_start=(\d+)\s+sc_count=(\d+)",
        text,
    )
    if not m:
        return None
    return {
        "submit_idx": int(m.group(1)),
        "flags": int(m.group(2), 16),
        "tasks": int(m.group(3)),  # == task_number
        "task_obj": int(m.group(4), 16),
        "core_mask": int(m.group(5), 16),
        "sc_start": int(m.group(6)),
        "sc_count": int(m.group(7)),
    }


def parse_task_bo(path):
    """Parse a sub*_bo_000_*.bin file into a list of task records.

    struct layout (from librocketnpu/src/rnpu_drm.c:112):
       0: flags u32
       4: op_idx u32
       8: enable_mask u32
      12: int_mask u32
      16: int_clear u32
      20: int_status u32
      24: regcfg_amount u32
      28: regcfg_offset u32
      32: regcmd_addr u64
      Total: 40 bytes
    """
    data = open(path, "rb").read()
    tasks = []
    for i in range(len(data) // 40):
        o = i * 40
        (flags, op_idx, em, _im, _ic, _is,
         rcg_amt, rcg_off) = struct.unpack_from("<IIIIIIII", data, o)
        rcmd_addr = u64(data, o + 32)
        tasks.append({
            "idx": i,
            "flags": flags,
            "op_idx": op_idx,
            "enable_mask": em,
            "regcfg_amount": rcg_amt,
            "regcfg_offset": rcg_off,
            "regcmd_addr": rcmd_addr,
        })
    return tasks


# ── Parser for the .rknn FlatBuffer operator table ──────────────────────

def parse_fb_ops(rknn_path):
    """Return (n_tensors, ops[]) for a .rknn model.

    Each op is a dict with idx, type, inputs, outputs, and a vtable dump
    containing all non-empty slots. This gives Task 9.2 enough material to
    find the per-op task-count field.
    """
    data = open(rknn_path, "rb").read()
    if len(data) < 0x50 or data[:4] != b"RKNN":
        raise RuntimeError(f"not an RKNN file: {rknn_path}")
    version = u64(data, 8)
    config_start = 0x40 if version > 1 else 0x18
    fb = memoryview(data)[config_start:]
    fb_size = len(fb)

    root = u32(fb, 0)
    f2_root = fb_field(fb, root, 2)
    sg = fb_vec_at(fb, f2_root, 0)
    tensors_fpos = fb_field(fb, sg, 0)
    ops_fpos = fb_field(fb, sg, 1)
    n_tensors = fb_vec_len(fb, tensors_fpos) if tensors_fpos else 0
    n_ops = fb_vec_len(fb, ops_fpos) if ops_fpos else 0

    ops = []
    for i in range(n_ops):
        op = fb_vec_at(fb, ops_fpos, i)
        if not op:
            continue
        entry = {"idx": i}

        # Type string (f[1] per parse_fb_operators)
        f1 = fb_field(fb, op, 1)
        entry["type"] = fb_string(fb, f1) if f1 else None

        # Inputs (f[4]) and outputs (f[5])
        for f, key in [(4, "inputs"), (5, "outputs")]:
            fp = fb_field(fb, op, f)
            if not fp:
                entry[key] = []
                continue
            vec = fb_follow(fb, fp)
            n = u32(fb, vec)
            if n > 256:
                entry[key] = []
                continue
            entry[key] = [u32(fb, vec + 4 + k * 4) for k in range(n)]

        # Full vtable dump — every non-empty slot
        _, vt_size, offs = fb_vtable(fb, op)
        fields = {}
        for f, off in enumerate(offs):
            if off == 0:
                continue
            fields[f] = classify_field(fb, fb_size, op + off)
        entry["vtable_size"] = vt_size
        entry["fields"] = fields
        ops.append(entry)

    return {
        "version": version,
        "n_tensors": n_tensors,
        "n_ops": n_ops,
        "ops": ops,
    }


# ── /tmp/rknn_dump lifecycle ────────────────────────────────────────────

def clean_dump_dir():
    # Same hardening as validate_accuracy.py: clean ONCE, synchronously.
    if os.path.exists(DUMP_DIR):
        for name in os.listdir(DUMP_DIR):
            path = os.path.join(DUMP_DIR, name)
            try:
                if os.path.isdir(path):
                    shutil.rmtree(path)
                else:
                    os.remove(path)
            except Exception:
                pass


def populate_dump(bench_dir, model_path):
    """Run bench_rknn under intercept_swap.so with DUMP_ALL_BOS=1 to
    capture the vendor's submits into /tmp/rknn_dump/."""
    bench = os.path.join(bench_dir, "bench_rknn")
    intercept = os.path.join(bench_dir, "intercept_swap.so")
    if not os.path.exists(bench) or not os.path.exists(intercept):
        raise RuntimeError(f"missing {bench} or {intercept}")
    clean_dump_dir()
    env = os.environ.copy()
    env["LD_PRELOAD"] = intercept
    env["DUMP_ALL_BOS"] = "1"
    proc = subprocess.run(
        [bench, model_path, "1"],
        env=env, cwd=bench_dir,
        capture_output=True, timeout=600,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"bench_rknn failed: rc={proc.returncode} "
            f"stderr={proc.stderr.decode(errors='replace')[:500]}"
        )


def collect_vendor_submits():
    """Read every submit_*.txt currently in DUMP_DIR in numeric order."""
    submits = []
    files = sorted(
        Path(DUMP_DIR).glob("submit_*.txt"),
        key=lambda p: int(re.search(r"submit_(\d+)", p.name).group(1)),
    )
    for f in files:
        s = parse_submit_txt(f)
        if s:
            submits.append(s)
    return submits


def collect_task_bos():
    """Read every sub*_bo_000_*.bin file. Returns dict {submit_idx: tasks[]}."""
    out = {}
    for f in sorted(Path(DUMP_DIR).glob("sub*_bo_000_*.bin")):
        m = re.match(r"sub(\d+)_bo_000_", f.name)
        if not m:
            continue
        out[int(m.group(1))] = parse_task_bo(f)
    return out


# ── Artifact builder ────────────────────────────────────────────────────

def _task_key(t):
    """Key that identifies a task independently of kernel-owned fields.
    We only care about fields set at compile time — the kernel may rewrite
    int_status between submits but that's execution state, not schedule."""
    return (t["flags"], t["op_idx"], t["enable_mask"],
            t["regcfg_amount"], t["regcfg_offset"], t["regcmd_addr"])


def _relativize_task_bo(tasks):
    """Rewrite regcmd_addr as an offset from the smallest regcmd_addr so
    the artifact is independent of kernel DMA allocations (which shift
    every run)."""
    if not tasks:
        return tasks
    base = min(t["regcmd_addr"] for t in tasks if t["regcmd_addr"] != 0)
    for t in tasks:
        t["regcmd_offset"] = (t["regcmd_addr"] - base) if t["regcmd_addr"] else 0
        del t["regcmd_addr"]
    return tasks


def build_artifact(model_path, bench_dir):
    populate_dump(bench_dir, model_path)
    submits = collect_vendor_submits()
    task_bos = collect_task_bos()
    fb = parse_fb_ops(model_path)

    # Task BO is identical across submits apart from kernel-owned status
    # fields. Store only the first snapshot and cross-check that all
    # other snapshots match on the compile-time fields.
    base_idx = min(task_bos.keys())
    base = task_bos[base_idx]
    divergent = []
    for idx, tb in task_bos.items():
        if idx == base_idx:
            continue
        if len(tb) != len(base):
            divergent.append([idx, "len-mismatch"])
            continue
        for i, (a, b) in enumerate(zip(tb, base)):
            if _task_key(a) != _task_key(b):
                divergent.append([idx, f"task[{i}] differs"])
                break

    # Strip non-deterministic DMA addresses so the artifact hash is stable
    # across runs. What we care about for segmentation analysis is the
    # relative position of each regcmd within the weight BO, not the DMA
    # absolute address.
    base = _relativize_task_bo(base)
    for s in submits:
        del s["task_obj"]

    return {
        "model": os.path.basename(model_path),
        "fb": fb,
        "vendor_submits": submits,
        "task_bo": base,          # Only one snapshot needed
        "task_bo_divergent_submits": divergent,
    }


# ── CLI ─────────────────────────────────────────────────────────────────

# ── Task 9.2: FB-derived segmentation rule ─────────────────────────────
#
# See openrknn/docs/segmentation_from_fb.md for the full derivation. The
# rule is verified byte-exact against the vendor dump for all 5 runtime
# models via `dump_segments.py --verify`.

LUT_OPS = {"ConvSigmoid", "ConvExSwish", "exSoftmax13"}
IO_OPS = {"InputOperator", "OutputOperator"}


def task_count_for_op(op):
    """Return the true per-op task count from f[10].

    f[10] is a 6-element uint32 vector. For activation-fused ops
    (ConvSigmoid, ConvExSwish, exSoftmax13, …) f[10][2] is 0 and
    f[10][1] is the total including the LUT-load pre-task. For all
    other ops f[10][0] is the total and f[10][2] is non-zero.
    """
    f10 = op["fields"].get("10", {}).get("vec_u32", [])
    if not f10:
        return 0
    if len(f10) < 3:
        return f10[0]
    return f10[1] if f10[2] == 0 else f10[0]


def derive_segments(fb_ops):
    """Reproduce the vendor's per-cycle submit list from the FB op table."""
    out = []
    cur_idx = 0
    cur = None
    for op in fb_ops:
        tc = task_count_for_op(op)
        if op["type"] in IO_OPS:
            if cur is not None:
                out.append(cur); cur = None
            cur_idx += tc
            continue
        if tc == 0:
            continue
        seg_flags = 0x1 if op["type"] in LUT_OPS else 0x5
        if cur is None or cur["flags"] != seg_flags:
            if cur is not None:
                out.append(cur)
            cur = {"start": cur_idx, "count": 0, "flags": seg_flags}
        cur["count"] += tc
        cur_idx += tc
    if cur is not None:
        out.append(cur)
    return out


def verify_artifact(path):
    """Compare derive_segments() against the vendor submit plan stored
    in a ground-truth artifact. Returns (ok, details)."""
    a = json.load(open(path))
    seen = set()
    vendor_list = []
    for s in a["vendor_submits"]:
        k = (s["flags"], s["sc_start"], s["sc_count"])
        if k in seen:
            continue
        seen.add(k)
        vendor_list.append(k)
    predicted = derive_segments(a["fb"]["ops"])
    predicted_list = [(p["flags"], p["start"], p["count"]) for p in predicted]
    ok = vendor_list == predicted_list
    return ok, {
        "vendor": vendor_list,
        "predicted": predicted_list,
        "model": a["model"],
    }


def cmd_verify(args):
    targets = []
    p = pathlib.Path(args.verify)
    if p.is_dir():
        targets = sorted(p.glob("*.json"))
    else:
        targets = [p]
    failures = 0
    for t in targets:
        ok, det = verify_artifact(t)
        if ok:
            print(f"OK   {det['model']}: {len(det['vendor'])} segments match")
        else:
            print(f"FAIL {det['model']}:")
            for i, (v, q) in enumerate(zip(det["vendor"], det["predicted"])):
                if v != q:
                    print(f"     seg[{i}] vendor={v} predicted={q}")
                    break
            if len(det["vendor"]) != len(det["predicted"]):
                print(f"     vendor count={len(det['vendor'])} "
                      f"predicted count={len(det['predicted'])}")
            failures += 1
    return 1 if failures else 0


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_mutually_exclusive_group(required=True)
    sub.add_argument("--bench-dir",
                     help="dir containing bench_rknn + intercept_swap.so "
                          "(required for capture mode)")
    sub.add_argument("--verify", metavar="PATH",
                     help="verify FB-derived segmentation against one "
                          "artifact file or a directory of them; exits "
                          "non-zero on mismatch")
    group = ap.add_mutually_exclusive_group()
    group.add_argument("--model", help="single .rknn file (capture mode)")
    group.add_argument("--all", action="store_true",
                       help="run over all models in --ground-truth")
    ap.add_argument("--models-dir", help="for --all")
    ap.add_argument("--ground-truth", help="for --all")
    ap.add_argument("--out", help="output JSON path (--model mode)")
    ap.add_argument("--out-dir", help="output dir (--all mode)")
    args = ap.parse_args()

    if args.verify:
        return cmd_verify(args)

    if args.model:
        if not args.out:
            ap.error("--out required with --model")
        art = build_artifact(args.model, args.bench_dir)
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(art, f, indent=2, sort_keys=True)
        print(f"wrote {args.out}")
        return 0

    # --all mode
    if not args.out_dir or not args.ground_truth or not args.models_dir:
        ap.error("--out-dir, --ground-truth, --models-dir all required with --all")
    os.makedirs(args.out_dir, exist_ok=True)
    gt = json.load(open(args.ground_truth))
    failed = []
    for name, cfg in gt.items():
        if cfg.get("type") == "fp16_parse":
            continue  # fp16 models don't run on the NPU path
        model_path = os.path.join(args.models_dir, cfg["model"])
        if not os.path.exists(model_path):
            print(f"SKIP {name}: {model_path} missing")
            continue
        out = os.path.join(args.out_dir, f"{name}.json")
        try:
            art = build_artifact(model_path, args.bench_dir)
            with open(out, "w") as f:
                json.dump(art, f, indent=2, sort_keys=True)
            print(f"OK   {name} → {out} "
                  f"(submits={len(art['vendor_submits'])}, "
                  f"ops={art['fb']['n_ops']})")
        except Exception as e:
            print(f"FAIL {name}: {e}")
            failed.append(name)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
