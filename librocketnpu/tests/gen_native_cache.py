#!/usr/bin/env python3
"""
Generate a .rknn_cache binary from intercept dump files.

The cache contains everything needed to replay RKNN's HW execution:
- Weight+regcmd BO data (from sub1_bo_001_*)
- Task array metadata (regcmd offsets, amounts)
- Original DMA addresses for patching

Usage: python3 gen_native_cache.py /tmp/rknn_dump model.rknn_cache
"""

import struct
import sys
import os
import glob

MAGIC = b'RNCA'  # Rocket NPU Cache
VERSION = 4  # v4: segment table with sc_start/sc_count from intercept

def main():
    dump_dir = sys.argv[1] if len(sys.argv) > 1 else "/tmp/rknn_dump"
    out_path = sys.argv[2] if len(sys.argv) > 2 else "model.rknn_cache"

    # Parse submit_1.txt for BO info
    submit_path = os.path.join(dump_dir, "submit_1.txt")
    bo_info = []
    submit_flags = 0
    submit_tasks = 0
    with open(submit_path) as f:
        for line in f:
            if line.startswith("submit="):
                parts = line.split()
                for p in parts:
                    if p.startswith("flags="): submit_flags = int(p.split("=")[1], 16)
                    if p.startswith("tasks="): submit_tasks = int(p.split("=")[1])
            elif line.startswith("bo["):
                parts = line.split()
                idx = int(parts[0].split("[")[1].split("]")[0])
                dma = size = 0
                for p in parts:
                    if "dma=" in p: dma = int(p.split("=")[1], 16)
                    if "size=" in p: size = int(p.split("=")[1])
                bo_info.append((idx, dma, size))

    print(f"Submit: flags=0x{submit_flags:x}, tasks={submit_tasks}")
    print(f"BOs: {len(bo_info)}")
    for idx, dma, size in bo_info:
        print(f"  BO[{idx}]: dma=0x{dma:x} size={size}")

    # Find weight+regcmd BO (BO[1], largest non-activation BO)
    # Parse tasks to find which BO contains regcmd addresses
    tasks_path = os.path.join(dump_dir, "sub1_tasks.txt")
    tasks = []
    with open(tasks_path) as f:
        for line in f:
            if not line.startswith("task["):
                continue
            parts = line.split()
            rc_addr = int([p for p in parts if "regcmd_addr=" in p][0].split("=")[1], 16)
            rc_amt = int([p for p in parts if "regcfg_amount=" in p][0].split("=")[1])
            tasks.append((rc_addr, rc_amt))

    # Identify which BO is the weight+regcmd BO (contains regcmd addresses)
    wt_rc_bo_idx = -1
    for rc_addr, _ in tasks[:1]:
        for idx, dma, size in bo_info:
            if dma <= rc_addr < dma + size:
                wt_rc_bo_idx = idx
                break

    if wt_rc_bo_idx < 0:
        print("ERROR: Cannot find weight+regcmd BO")
        sys.exit(1)

    wt_rc_dma = bo_info[wt_rc_bo_idx][1]
    wt_rc_size = bo_info[wt_rc_bo_idx][2]
    print(f"\nWeight+regcmd BO: [{wt_rc_bo_idx}] dma=0x{wt_rc_dma:x} size={wt_rc_size}")

    # Load weight+regcmd BO data
    wt_rc_file = glob.glob(os.path.join(dump_dir, f"sub1_bo_{wt_rc_bo_idx:03d}_*B.bin"))[0]
    wt_rc_data = open(wt_rc_file, "rb").read()
    assert len(wt_rc_data) == wt_rc_size

    # Find regcmd start offset within BO
    rc_start = tasks[0][0] - wt_rc_dma
    print(f"Regcmd starts at BO offset 0x{rc_start:x}")
    print(f"Weight region: 0..0x{rc_start:x} ({rc_start} bytes)")
    print(f"Regcmd region: 0x{rc_start:x}..end ({wt_rc_size - rc_start} bytes)")

    # Identify activation BO (all zeros, second largest)
    # and input BO (matches known input sizes)
    act_bo_idx = -1
    for idx, dma, size in bo_info:
        if idx == wt_rc_bo_idx:
            continue
        bo_file = glob.glob(os.path.join(dump_dir, f"sub1_bo_{idx:03d}_*B.bin"))[0]
        bo_data = open(bo_file, "rb").read()
        if all(b == 0 for b in bo_data[:4096]) and size > 100000:
            act_bo_idx = idx
            break

    # Build cache file
    # Header: magic(4) version(4) bo_count(4) task_count(4)
    #         submit_flags(4) pad(4)
    #         wt_rc_bo_idx(4) wt_rc_size(4) wt_rc_dma(8)
    #         rc_start_offset(4) pad(4)
    # Per-BO: dma(8) size(8)
    # Per-task: rc_offset_in_bo(4) rc_amount(4)
    # Data: wt_rc_data (full BO[1])

    n_bos = len(bo_info)
    n_tasks = len(tasks)

    header = struct.pack("<4sIIIII IIQ II",
        MAGIC, VERSION, n_bos, n_tasks,
        submit_flags, 0,  # pad
        wt_rc_bo_idx, wt_rc_size, wt_rc_dma,
        rc_start, 0)  # pad

    bo_table = b""
    for idx, dma, size in bo_info:
        bo_table += struct.pack("<QQ", dma, size)

    # Parse raw task BO to get enable_mask per task
    task_bo_file = glob.glob(os.path.join(dump_dir, f"sub1_bo_000_*B.bin"))[0]
    task_bo_data = open(task_bo_file, "rb").read()
    TASK_SIZE = 40  # struct rknpu_task: 8*u32 + 1*u64
    enable_masks = []
    op_indices = []
    for t in range(len(tasks)):
        off = t * TASK_SIZE
        if off + TASK_SIZE <= len(task_bo_data):
            fields = struct.unpack_from("<IIIIII II Q", task_bo_data, off)
            enable_masks.append(fields[2])  # enable_mask
            op_indices.append(fields[1])    # op_idx
        else:
            enable_masks.append(0x1d)
            op_indices.append(0)

    task_table = b""
    for i, (rc_addr, rc_amt) in enumerate(tasks):
        rc_off = rc_addr - wt_rc_dma
        task_table += struct.pack("<IIII", rc_off, rc_amt, enable_masks[i], op_indices[i])

    # Raw task BO: first n_tasks * 40 bytes
    raw_task_bo = task_bo_data[:n_tasks * TASK_SIZE]

    # Parse submit segments from intercept stderr log.
    # The intercept logs: SWAP: SUBMIT[N] flags=0xF tasks=T ... sc[0]={start,count}
    # We need to find these lines — they're in stderr during the capture.
    # Parse from submit_*.txt files which contain the first line of each submit.
    # But sc values are only in the SWAP: SUBMIT lines (stderr), not in submit_*.txt.
    # So we parse from the submit_*.txt which has 'submit=N flags=... tasks=...'
    # and extract sc from a separate source.
    #
    # Better approach: parse sc directly from the raw task BO + submit metadata.
    # The submit_*.txt has task_obj and task counts. The sc values from intercept are:
    # stored in the SUBMIT log lines. Let me check if submit_*.txt has them.

    # Actually, let's just hardcode the extraction from intercept stderr.
    # The gen script should be run right after the intercept capture.
    # Parse all SWAP: SUBMIT lines from a log file if available.
    segments = []
    for sub_id in range(1, 50):
        sub_path = os.path.join(dump_dir, f"submit_{sub_id}.txt")
        if not os.path.exists(sub_path):
            break
        with open(sub_path) as sf:
            line = sf.readline().strip()
        parts = line.split()
        flags = int([p for p in parts if p.startswith("flags=")][0].split("=")[1], 16)
        task_num = int([p for p in parts if p.startswith("tasks=")][0].split("=")[1])
        # Parse sc_start and sc_count if present (v4 intercept format)
        sc_start = sc_count = 0
        for p in parts:
            if p.startswith("sc_start="): sc_start = int(p.split("=")[1])
            if p.startswith("sc_count="): sc_count = int(p.split("=")[1])
        segments.append({"flags": flags, "task_number": task_num,
                         "sc_start": sc_start, "sc_count": sc_count})

    for i, seg in enumerate(segments):
        print(f"  Segment {i+1}: flags=0x{seg['flags']:x} sc={{{seg['sc_start']},{seg['sc_count']}}} task_number={seg['task_number']}")

    # Encode segments: flags(4) + sc_start(4) + sc_count(4) + task_number(4) per segment
    seg_data = b""
    for seg in segments:
        seg_data += struct.pack("<IIII", seg["flags"], seg["sc_start"], seg["sc_count"], seg["task_number"])
    n_segments = len(segments)

    with open(out_path, "wb") as f:
        f.write(header)
        f.write(bo_table)
        f.write(task_table)
        f.write(wt_rc_data)
        f.write(raw_task_bo)
        # v3: segment table
        f.write(struct.pack("<I", n_segments))
        f.write(seg_data)

    total = len(header) + len(bo_table) + len(task_table) + len(wt_rc_data) + len(raw_task_bo) + 4 + len(seg_data)
    print(f"\nWrote {out_path}: {total} bytes")
    print(f"  Header: {len(header)} bytes")
    print(f"  BO table: {len(bo_table)} bytes ({n_bos} entries)")
    print(f"  Task table: {len(task_table)} bytes ({n_tasks} entries)")
    print(f"  BO data: {len(wt_rc_data)} bytes")
    print(f"  Task BO: {len(raw_task_bo)} bytes ({n_tasks} tasks × {TASK_SIZE}B)")
    print(f"  Segments: {n_segments}")

if __name__ == "__main__":
    main()
