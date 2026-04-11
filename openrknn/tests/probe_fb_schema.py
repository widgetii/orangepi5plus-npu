#!/usr/bin/env python3
"""Probe the .rknn FlatBuffer schema to find where the operator graph lives.

Phase 3 exploration tool. Walks the root table, the first subgraph, and all
the fields below, reporting the vtable structure so we can identify which
field holds the operator vector.

Usage: probe_fb_schema.py /root/npu-research/mobilenet_v1.rknn [--subgraph 0]
"""
import argparse
import struct
import sys


def u8(b, p): return b[p]
def u16(b, p): return struct.unpack_from("<H", b, p)[0]
def u32(b, p): return struct.unpack_from("<I", b, p)[0]
def i32(b, p): return struct.unpack_from("<i", b, p)[0]
def u64(b, p): return struct.unpack_from("<Q", b, p)[0]


def fb_follow(b, p):
    return p + u32(b, p)


def fb_vtable(b, table):
    """Return (vt_start, vt_size, [field_offsets]) for a table."""
    vt = table - i32(b, table)
    vt_size = u16(b, vt)
    tbl_size = u16(b, vt + 2)
    field_offs = []
    fo = 4
    while fo < vt_size:
        off = u16(b, vt + fo)
        field_offs.append(off)
        fo += 2
    return vt, vt_size, tbl_size, field_offs


def fb_field(b, table, field):
    vt, vt_size, _, offs = fb_vtable(b, table)
    if field >= len(offs):
        return 0
    return table + offs[field] if offs[field] else 0


def fb_vec_len(b, fpos):
    vec = fb_follow(b, fpos)
    return u32(b, vec)


def fb_vec_at(b, fpos, idx):
    """For a vector of table offsets."""
    vec = fb_follow(b, fpos)
    count = u32(b, vec)
    if idx >= count:
        return 0
    elem_pos = vec + 4 + idx * 4
    return fb_follow(b, elem_pos)


def fb_string(b, fpos, maxlen=64):
    try:
        s = fb_follow(b, fpos)
        n = u32(b, s)
        if n > 256:
            return None
        return bytes(b[s+4:s+4+n]).decode("utf-8", errors="replace")
    except Exception:
        return None


def classify_vec(b, fpos, fb_size):
    """Try to classify a vector: scalar (u32/i32/float) or table/string."""
    try:
        vec = fb_follow(b, fpos)
        if vec >= fb_size - 4:
            return "out-of-bounds"
        n = u32(b, vec)
        if n > 1000000:
            return f"bogus-count={n}"
        if n == 0:
            return "empty"

        # Scalar 4-byte elements: check if first element looks like an offset
        # to a table or a plain int
        first = u32(b, vec + 4)

        # If first interpreted as offset from (vec+4) lands inside fb, and
        # the target looks like a table (has a valid vtable), it's a vec
        # of tables.
        if first < fb_size:
            tbl_candidate = vec + 4 + first
            if 0 < tbl_candidate < fb_size - 8:
                try:
                    vt = tbl_candidate - i32(b, tbl_candidate)
                    if 0 <= vt < fb_size - 4:
                        vt_size = u16(b, vt)
                        if 4 <= vt_size <= 256:
                            return f"vec-of-tables[{n}] (first elem vt_size={vt_size})"
                except Exception:
                    pass

        # Otherwise: probably a scalar vector
        if n <= 16:
            vals = [u32(b, vec + 4 + i * 4) for i in range(n)]
            return f"vec-u32[{n}]={vals}"
        return f"vec-u32[{n}] sample={u32(b, vec + 4)},{u32(b, vec + 8) if n > 1 else 0}"
    except Exception as e:
        return f"error: {e}"


def dump_table(b, table, label, fb_size, depth=0):
    indent = "  " * depth
    print(f"{indent}{label} @ 0x{table:x}")
    vt, vt_size, tbl_size, field_offs = fb_vtable(b, table)
    print(f"{indent}  vtable @ 0x{vt:x} vt_size={vt_size} tbl_size={tbl_size} "
          f"n_fields={len(field_offs)}")
    for i, off in enumerate(field_offs):
        if off == 0:
            continue
        fpos = table + off
        # Try to infer what kind of field this is.
        # If it looks like a vec offset (follow → count), try to classify.
        info = ""
        # scalar quick-reads
        try:
            scalar_u8 = u8(b, fpos) if fpos < fb_size else 0
            scalar_u32 = u32(b, fpos) if fpos + 4 <= fb_size else 0
        except Exception:
            scalar_u8 = scalar_u32 = 0
        info = f"u8={scalar_u8} u32=0x{scalar_u32:x}"

        # Try classifying as vec/string
        cls = classify_vec(b, fpos, fb_size)
        # Try as string too
        s = fb_string(b, fpos)
        if s is not None and all(0x20 <= ord(c) < 0x7f or c in "\t\n" for c in s):
            if s:
                info += f"  str={s!r}"

        print(f"{indent}  f[{i:2d}] off={off} abs=0x{fpos:x}  {info}")
        print(f"{indent}        {cls}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("rknn_file")
    ap.add_argument("--subgraph", type=int, default=0)
    ap.add_argument("--dump-op", type=int, default=None,
                    help="If set, dump the Nth operator's fields")
    ap.add_argument("--op-field", type=int, default=1,
                    help="Which subgraph field holds the operator vector "
                         "(default: 1)")
    args = ap.parse_args()

    data = open(args.rknn_file, "rb").read()
    print(f"file: {args.rknn_file} ({len(data)} bytes)")
    # Header
    version = u64(data, 8)
    config_start = 0x40 if version > 1 else 0x18
    print(f"version={version} config_start=0x{config_start:x}")

    fb = memoryview(data)[config_start:]
    fb_size = len(fb)

    root = u32(fb, 0)
    print(f"\n=== ROOT TABLE @ 0x{root:x} ===")
    dump_table(fb, root, "root", fb_size)

    # Root field 2 = subgraphs vector (from existing code)
    f2_root = fb_field(fb, root, 2)
    if not f2_root:
        print("no subgraphs field")
        return

    n_sg = fb_vec_len(fb, f2_root)
    print(f"\n{n_sg} subgraph(s)")

    sg = fb_vec_at(fb, f2_root, args.subgraph)
    if not sg:
        return

    print(f"\n=== SUBGRAPH[{args.subgraph}] @ 0x{sg:x} ===")
    dump_table(fb, sg, f"subgraph[{args.subgraph}]", fb_size)

    # For every field of the subgraph that looks like a vec-of-tables,
    # dump the first element's table structure.
    _, _, _, sg_field_offs = fb_vtable(fb, sg)
    for i, off in enumerate(sg_field_offs):
        if off == 0:
            continue
        fpos = sg + off
        cls = classify_vec(fb, fpos, fb_size)
        if "vec-of-tables" not in cls:
            continue
        try:
            vec = fb_follow(fb, fpos)
            n = u32(fb, vec)
            if n == 0 or n > 100000:
                continue
            first = fb_vec_at(fb, fpos, 0)
            print(f"\n  --- subgraph.f[{i}] first element table @ 0x{first:x} "
                  f"(vec of {n} tables) ---")
            dump_table(fb, first, f"sg.f[{i}][0]", fb_size, depth=1)

            if args.dump_op is not None and i == args.op_field:
                target = fb_vec_at(fb, fpos, args.dump_op)
                if target:
                    print(f"\n  === OPERATOR[{args.dump_op}] in sg.f[{i}] "
                          f"@ 0x{target:x} ===")
                    dump_table(fb, target, f"op[{args.dump_op}]", fb_size, depth=1)
                    # For each vec-of-tables field in the op, dump element 0
                    _, _, _, op_field_offs = fb_vtable(fb, target)
                    for oi, ooff in enumerate(op_field_offs):
                        if ooff == 0:
                            continue
                        ofpos = target + ooff
                        ocls = classify_vec(fb, ofpos, fb_size)
                        if "vec-of-tables" not in ocls:
                            continue
                        try:
                            first = fb_vec_at(fb, ofpos, 0)
                            if first:
                                print(f"\n    --- op.f[{oi}][0] @ 0x{first:x} ---")
                                dump_table(fb, first, f"op.f[{oi}][0]",
                                           fb_size, depth=2)
                        except Exception:
                            pass
        except Exception as e:
            print(f"  error: {e}")


if __name__ == "__main__":
    main()
