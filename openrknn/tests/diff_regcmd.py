#!/usr/bin/env python3
"""Compare two weight BO snapshots (template-patched vs vendor oracle) at the
per-task per-register level.

This is the phase-0 diff oracle for the template-patch development work. It
reads an "oracle" weight BO (populated by the vendor librknnrt.so via the
intercept_swap.so dump at /tmp/rknn_dump/sub1_bo_001_*.bin) and a "template"
weight BO (produced by openrknn with ORKNN_FORCE_TEMPLATE=1 ORKNN_DUMP_BO1=...
set), walks the task BO to find each task's regcmd slice, decodes every
(reg, val) pair, and prints the deltas.

DMA-address registers are expected to differ (the two BOs live at different
DMA addresses); the tool flags them separately. Any *non-DMA* register
difference is a real template-path bug.

Usage:
    diff_regcmd.py \\
        --oracle /tmp/rknn_dump/sub1_bo_001_8388608B.bin \\
        --template /tmp/template_bo1.bin \\
        --task-bo  /tmp/rknn_dump/sub1_bo_000_4096B.bin \\
        [--oracle-wt-base 0x4d000000] [--template-wt-base 0x...] \\
        [--max-tasks 8]

Exit code is nonzero if any non-DMA difference is found.
"""
import argparse
import struct
import sys

# Registers that encode DMA addresses. Their values are expected to differ
# between oracle and template if the two BOs have different DMA bases — the
# tool subtracts the respective base before comparing.
#
# Register names below come from librocketnpu/src/rnpu_registers.h (the
# Mesa-generated authoritative reference). Several addresses that looked
# like DMA pointers at first glance turned out to be plain 32-bit values
# (CNA_PAD_CON1, DPU_EW_CVT_OFFSET_VALUE, DPU_OUT_CVT_OFFSET,
# DPU_LUT_LE_START); those are not in this map and fall through to the
# non-DMA branch.
DMA_REGS = {
    0x0010: "PC_BASE_ADDRESS",           # REG_PC_BASE_ADDRESS
    0x1070: "CNA_FEATURE_DATA_ADDR",     # REG_CNA_FEATURE_DATA_ADDR
    0x1110: "CNA_DCOMP_ADDR0",           # REG_CNA_DCOMP_ADDR0 (weight base)
    0x4020: "DPU_DST_BASE_ADDR",         # REG_DPU_DST_BASE_ADDR
    0x5018: "DPU_RDMA_SRC_BASE_ADDR",    # REG_DPU_RDMA_RDMA_SRC_BASE_ADDR
    0x5020: "DPU_RDMA_BS_BASE_ADDR",     # REG_DPU_RDMA_RDMA_BS_BASE_ADDR
    0x502c: "DPU_RDMA_BN_BASE_ADDR",     # REG_DPU_RDMA_RDMA_BN_BASE_ADDR
    0x5038: "DPU_RDMA_EW_BASE_ADDR",     # REG_DPU_RDMA_RDMA_EW_BASE_ADDR
    0x6070: "PPU_DST_BASE_ADDR",         # REG_PPU_DST_BASE_ADDR
    0x701c: "PPU_RDMA_SRC_BASE_ADDR",    # REG_PPU_RDMA_RDMA_SRC_BASE_ADDR
}

# Human-readable labels for a handful of often-interesting non-DMA registers.
KNOWN_REGS = {
    0x104c: "CNA_CVT_CON0",
    0x1050: "CNA_CVT_CON1",
    0x1054: "CNA_CVT_CON2",
    0x1058: "CNA_CVT_CON3",
    0x105c: "CNA_CVT_CON4",
    0x1180: "CNA_CVT_CON5",
    0x100c: "CNA_CONV_CON1",
    0x1010: "CNA_CONV_CON2",
    0x1014: "CNA_CONV_CON3",
    0x1020: "CNA_DATA_SIZE0",
    0x1024: "CNA_DATA_SIZE1",
    0x1028: "CNA_DATA_SIZE2",
    0x102c: "CNA_DATA_SIZE3",
    0x1030: "CNA_WEIGHT_SIZE0",
    0x1034: "CNA_WEIGHT_SIZE1",
    0x1038: "CNA_WEIGHT_SIZE2",
    0x107c: "CNA_DMA_CON1",
    0x1080: "CNA_DMA_CON2",
    0x1184: "CNA_PAD_CON1",
    0x4014: "DPU_DATA_CUBE_WIDTH",
    0x4018: "DPU_DATA_CUBE_HEIGHT",
    0x401c: "DPU_DATA_CUBE_CHANNEL",
    0x4024: "DPU_DST_SURF_STRIDE",
    0x4040: "DPU_BS_CFG",
    0x4048: "DPU_BS_MUL_CFG",
    0x4050: "DPU_BN_CFG",
    0x4070: "DPU_OUT_CVT_SCALE",
    0x4074: "DPU_EW_CVT_OFFSET_VALUE",
    0x4078: "DPU_OUT_CVT_SHIFT",
    0x407c: "DPU_OUT_CVT_OFFSET",
    0x4080: "DPU_EW_CVT_SCALE_VALUE",
    0x4110: "DPU_LUT_LE_START",
    0x4114: "DPU_LUT_LE_END",
    0x4118: "DPU_LUT_LO_START",
    0x411c: "DPU_LUT_LO_END",
    0x5000: "RDMA_DATA_CUBE_WIDTH",
    0x5004: "RDMA_DATA_CUBE_HEIGHT",
    0x5008: "RDMA_DATA_CUBE_CHANNEL",
    0x5010: "RDMA_SRC_LINE_STRIDE",
    0x5014: "RDMA_SRC_SURF_STRIDE",
}


def reg_label(reg):
    if reg in DMA_REGS:
        return f"{DMA_REGS[reg]}(DMA)"
    if reg in KNOWN_REGS:
        return KNOWN_REGS[reg]
    return f"0x{reg:04x}"


TASK_STRUCT = "<IIIIIIIIQ"  # 40 bytes
TASK_FIELDS = ["flags", "op_idx", "enable_mask", "int_mask",
               "int_clear", "int_status", "regcfg_amount",
               "regcfg_offset", "regcmd_addr"]
TASK_SIZE = 40


def parse_tasks(task_bo_bytes):
    tasks = []
    n = len(task_bo_bytes) // TASK_SIZE
    for i in range(n):
        fields = struct.unpack_from(TASK_STRUCT, task_bo_bytes, i * TASK_SIZE)
        task = dict(zip(TASK_FIELDS, fields))
        task["index"] = i
        tasks.append(task)
    # Drop trailing zero tasks (task BO is allocated to 4K).
    while tasks and tasks[-1]["enable_mask"] == 0 and tasks[-1]["regcmd_addr"] == 0:
        tasks.pop()
    return tasks


def decode_entry(entry):
    reg = entry & 0xFFFF
    val = (entry >> 16) & 0xFFFFFFFF
    tgt = (entry >> 48) & 0xFFFF
    return reg, val, tgt


def read_regcmd_slice(bo_bytes, base_dma, wt_base, regcmd_addr, n_entries):
    """Read a task's regcmd slice from a weight BO.

    ``regcmd_addr`` is an absolute DMA address (from the task BO's regcmd_addr
    field). ``wt_base`` is that particular weight BO's DMA base. The slice
    offset within the BO is (regcmd_addr - wt_base).
    """
    off = regcmd_addr - wt_base
    if off < 0 or off + n_entries * 8 > len(bo_bytes):
        return None
    return list(struct.unpack_from(f"<{n_entries}Q", bo_bytes, off))


def classify_dma(val, bases):
    """Given a DMA-register value, return (bo_tag, relative_offset) if it
    falls inside one of the known BO ranges, else (None, val).
    ``bases`` maps tag -> (dma_base, size).

    Uses 64-bit math to avoid 32-bit wraparound when a BO ends past 4 GB
    in virtual address space (which happens for ResNet50 whose weight BO
    sits around 0xfe6b2000 and extends through 0x100002800)."""
    v = val & 0xFFFFFFFF
    for tag, (base, size) in bases.items():
        if base == 0:
            continue
        lo = base & 0xFFFFFFFF
        # Raw 64-bit range check
        if base <= v < base + max(size, 1):
            return tag, v - lo
        # If base+size overflows 32-bit, also check the "below-wrap" half
        if base + max(size, 1) > 0x100000000 and v < (base + max(size, 1)) & 0xFFFFFFFF:
            return tag, (v + 0x100000000) - lo
    return None, v


def diff_task(task, oracle_bo, template_bo,
              oracle_bases, template_bases, em_filter):
    """Diff a single task's regcmd. Returns (dma_diffs, nondma_diffs)."""
    if em_filter and task["enable_mask"] not in em_filter:
        return [], []

    n_entries = task["regcfg_amount"] + 4
    oracle_wt = oracle_bases["wt"][0]
    template_wt = template_bases["wt"][0]

    oracle_slice = read_regcmd_slice(
        oracle_bo, 0, oracle_wt, task["regcmd_addr"], n_entries)

    tmpl_regcmd_addr = template_wt + (task["regcmd_addr"] - oracle_wt)
    tmpl_slice = read_regcmd_slice(
        template_bo, 0, template_wt, tmpl_regcmd_addr, n_entries)

    dma_diffs = []
    nondma_diffs = []

    if oracle_slice is None or tmpl_slice is None:
        return dma_diffs, [(task, -1, 0, 0, 0, "slice out of range")]

    for e, (o, t) in enumerate(zip(oracle_slice, tmpl_slice)):
        if o == t:
            continue
        oreg, oval, otgt = decode_entry(o)
        treg, tval, ttgt = decode_entry(t)
        if oreg != treg or otgt != ttgt:
            nondma_diffs.append((task, e, oreg, oval, tval,
                                 f"register/target mismatch: oracle="
                                 f"tgt=0x{otgt:04x} reg=0x{oreg:04x}, "
                                 f"template=tgt=0x{ttgt:04x} reg=0x{treg:04x}"))
            continue
        if oreg in DMA_REGS:
            # Classify each value by which BO it points into, then compare
            # the relative offsets. If the classifications and offsets
            # match, the only difference is the BO base delta — benign.
            # Otherwise it's a real DMA mismatch (pointing at the wrong
            # BO or wrong offset within the BO).
            o_tag, o_rel = classify_dma(oval, oracle_bases)
            t_tag, t_rel = classify_dma(tval, template_bases)
            if o_tag == t_tag and o_rel == t_rel and o_tag is not None:
                continue  # benign base delta
            note = (f"oracle=({o_tag or '???'}+0x{o_rel & 0xFFFFFFFF:x}) "
                    f"template=({t_tag or '???'}+0x{t_rel & 0xFFFFFFFF:x})")
            dma_diffs.append((task, e, oreg, oval, tval, note))
        else:
            nondma_diffs.append((task, e, oreg, oval, tval, ""))

    return dma_diffs, nondma_diffs


def parse_em_filter(s):
    if not s:
        return None
    return {int(x, 0) for x in s.split(",")}


def parse_submit_txt(path):
    """Parse /tmp/rknn_dump/submit_N.txt. Returns dict: bo_idx -> (dma, size)."""
    import re
    bos = {}
    try:
        txt = open(path).read()
    except OSError:
        return bos
    for line in txt.splitlines():
        m = re.match(
            r"bo\[(\d+)\]\s+handle=\S+\s+dma=0x([0-9a-fA-F]+)\s+obj=\S+\s+size=(\d+)",
            line)
        if m:
            bos[int(m.group(1))] = (int(m.group(2), 16), int(m.group(3)))
    return bos


def bases_from_submit_txt(path):
    """Return the BO-base dict expected by classify_dma(), built from an
    intercept submit_N.txt file."""
    bos = parse_submit_txt(path)
    bases = {}
    if 0 in bos: bases["task"] = bos[0]
    if 1 in bos: bases["wt"] = bos[1]
    if 2 in bos: bases["act"] = bos[2]
    if 3 in bos: bases["in"] = bos[3]
    for i in range(4, 20):
        if i in bos:
            bases[f"out{i-4}"] = bos[i]
    return bases


def parse_meta_sidecar(path):
    """Parse the .meta sidecar written by openrknn_run.c when ORKNN_DUMP_BO1
    is set. Returns the same bases dict shape as bases_from_submit_txt."""
    bases = {}
    try:
        txt = open(path).read()
    except OSError:
        return bases
    def _int(s): return int(s, 0)
    for line in txt.splitlines():
        line = line.strip()
        if line.startswith("weight_bo_dma="):
            dma = _int(line.split("=", 1)[1])
            bases.setdefault("wt", [dma, 0])
        elif line.startswith("weight_bo_size="):
            bases.setdefault("wt", [0, 0])
            bases["wt"] = (bases["wt"][0], _int(line.split("=", 1)[1]))
        elif line.startswith("task_bo_dma="):
            bases.setdefault("task", (_int(line.split("=", 1)[1]), 4096))
        elif line.startswith("activation_bo_dma="):
            bases.setdefault("act", [_int(line.split("=", 1)[1]), 0])
        elif line.startswith("activation_bo_size="):
            prev = bases.get("act", [0, 0])
            bases["act"] = (prev[0], _int(line.split("=", 1)[1]))
        elif line.startswith("input_bo["):
            idx = int(line[len("input_bo["):line.index("]")])
            dma_s = line.split("dma=", 1)[1].split()[0]
            sz_s = line.split("size=", 1)[1]
            tag = "in" if idx == 0 else f"in{idx}"
            bases[tag] = (_int(dma_s), _int(sz_s))
        elif line.startswith("output_bo["):
            idx = int(line[len("output_bo["):line.index("]")])
            dma_s = line.split("dma=", 1)[1].split()[0]
            sz_s = line.split("size=", 1)[1]
            bases[f"out{idx}"] = (_int(dma_s), _int(sz_s))
    # Normalize any tuple-tuple leftovers.
    for k, v in list(bases.items()):
        if isinstance(v, list):
            bases[k] = tuple(v)
    return bases


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oracle", required=True,
                    help="Vendor weight BO (sub1_bo_001_*.bin from /tmp/rknn_dump)")
    ap.add_argument("--template", required=True,
                    help="openrknn weight BO dumped via ORKNN_DUMP_BO1")
    ap.add_argument("--task-bo", required=True,
                    help="Task BO (sub1_bo_000_*.bin from /tmp/rknn_dump)")
    ap.add_argument("--submit-txt", default="/tmp/rknn_dump/submit_1.txt",
                    help="Vendor intercept submit_N.txt (for oracle BO bases)")
    ap.add_argument("--template-meta", default=None,
                    help="Template .meta sidecar path "
                         "(default: <template>.meta)")
    ap.add_argument("--max-tasks", type=int, default=0,
                    help="Limit diff to first N tasks (0 = all)")
    ap.add_argument("--em-filter", default="",
                    help="Comma-separated list of enable_mask values to include "
                         "(e.g. 0x1d,0x60). Default: all.")
    ap.add_argument("--show-dma", action="store_true",
                    help="Also print per-DMA-register deltas (normally "
                         "summarized only)")
    args = ap.parse_args()

    oracle_bo = open(args.oracle, "rb").read()
    template_bo = open(args.template, "rb").read()
    task_bo = open(args.task_bo, "rb").read()

    oracle_bases = bases_from_submit_txt(args.submit_txt)
    if "wt" not in oracle_bases:
        print(f"ERROR: could not read bo[1] base from {args.submit_txt}",
              file=sys.stderr)
        return 2

    meta_path = args.template_meta or (args.template + ".meta")
    template_bases = parse_meta_sidecar(meta_path)
    if "wt" not in template_bases:
        print(f"ERROR: could not read weight_bo_dma from {meta_path}",
              file=sys.stderr)
        return 2

    em_filter = parse_em_filter(args.em_filter)

    tasks = parse_tasks(task_bo)
    if args.max_tasks:
        tasks = tasks[:args.max_tasks]

    print(f"oracle    : {args.oracle} ({len(oracle_bo)} bytes, "
          f"wt=0x{oracle_bases['wt'][0]:x})")
    print(f"template  : {args.template} ({len(template_bo)} bytes, "
          f"wt=0x{template_bases['wt'][0]:x})")
    print(f"task bo   : {args.task_bo} ({len(tasks)} tasks)")
    print(f"oracle bases  : "
          + " ".join(f"{k}=0x{v[0]:x}" for k, v in oracle_bases.items()))
    print(f"template bases: "
          + " ".join(f"{k}=0x{v[0]:x}" for k, v in template_bases.items()))
    if em_filter:
        print(f"em filter : {{{', '.join(f'0x{e:02x}' for e in em_filter)}}}")
    print()

    total_dma = 0
    total_nondma = 0
    seen_regcmd = set()

    for task in tasks:
        if task["regcmd_addr"] in seen_regcmd:
            continue
        seen_regcmd.add(task["regcmd_addr"])

        dma_diffs, nondma_diffs = diff_task(
            task, oracle_bo, template_bo,
            oracle_bases, template_bases, em_filter)

        if not dma_diffs and not nondma_diffs:
            continue

        hdr = (f"=== task[{task['index']:3d}] op={task['op_idx']:3d} "
               f"em=0x{task['enable_mask']:02x} "
               f"amt={task['regcfg_amount']} "
               f"regcmd=0x{task['regcmd_addr']:x} ===")
        print(hdr)

        for (_, e, reg, oval, tval, note) in nondma_diffs:
            label = reg_label(reg)
            print(f"  NONDMA [{e:3d}] {label:30s} oracle=0x{oval:08x} "
                  f"template=0x{tval:08x}  {note}")
        total_nondma += len(nondma_diffs)

        if args.show_dma:
            for (_, e, reg, oval, tval, note) in dma_diffs:
                label = reg_label(reg)
                print(f"  DMA    [{e:3d}] {label:30s} oracle=0x{oval:08x} "
                      f"template=0x{tval:08x}  {note}")
        total_dma += len(dma_diffs)

        print()

    print(f"summary: {total_nondma} non-DMA diffs, {total_dma} DMA-class diffs, "
          f"across {len(seen_regcmd)} unique regcmd sections")

    # Both non-DMA diffs AND DMA-class mismatches (where classify_dma found
    # different BO tags or different relative offsets) are real bugs.
    # Only "benign base delta" diffs are filtered out in diff_task(), so
    # everything surfaced here is a legitimate mismatch.
    return 1 if (total_nondma > 0 or total_dma > 0) else 0


if __name__ == "__main__":
    sys.exit(main())
