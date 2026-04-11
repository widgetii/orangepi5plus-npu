# FB-Derived Segmentation Rule

> Task 9.2 deliverable. Canonical source for how openrknn builds the
> `model->segments[]` array from the .rknn FlatBuffer, replacing the
> former read of `/tmp/rknn_dump/submit_*.txt`.

## Runtime context

The vendor `librknnrt.so` run loop (`sub_31FD20`, decompile lines
642601–644830) walks the op graph sequentially and accumulates task
counts from consecutive NPU nodes. Each non-NPU node, or any op that
requires a different submit mode, forces a flush — the accumulated task
range becomes one `ioctl(0xC0687201)` submit and the accumulator resets.

Each submit carries four numbers:

- `flags`         — 0x5 (`PC | PINGPONG`) for pipelined multi-task chains,
                   0x1 (`PC` only) for single-task barriers
- `sc_start`      — task index into the weight BO at which this submit starts
- `sc_count`      — number of tasks in this submit
- `task_number`   — `sc_count × active_core_count` (multi-core replication)

The vendor's choice of boundaries and flags is a direct function of the
.rknn operator graph — there is no runtime-only state involved. Our job
is to reproduce the same `(flags, sc_start, sc_count)` tuples from the
FlatBuffer.

## The key FlatBuffer fields

A subgraph's operators live in `sg.f[1]` (vector of Operator tables).
Per-op fields we care about for segmentation:

| Field | Type            | Meaning                           | Written to runtime node at |
| ----- | --------------- | --------------------------------- | -------------------------- |
| f[1]  | string          | Op type name (`"Conv"`, …)        | `node+8`                  |
| f[4]  | vec<uint32>     | Input tensor indices              | `node+80`                 |
| f[5]  | vec<uint32>     | Output tensor indices             | `node+104`                |
| f[10] | vec<uint32>(6)  | Task counts (see below)           | `node+488..+496`          |
| f[11] | vec<uint32>(6)  | Sub-counts (unused for segmentation) | `node+464..+472`      |
| f[12] | vec<uint32>(9)  | Tensor sizes (bytes)              | `node+512..+540`          |

The `f[10]` vector is the authoritative source for per-op task count.
`librknnrt.so` reads `f[10][0]` into `node+488` and uses it as the task
count for the run loop's accumulator (see `sub_2DC018` line 563864 in
`librknnrt.so.c`, which is the v>5 operator extraction function; the
matching decompile fragment is:

```c
v358 = (unsigned __int16 *)((char *)v20 - *v20);   /* vtable ptr */
if (v358[12])                                       /* f[10] offset */
    *(_DWORD *)(v28 + 488) = *(int *)(                /* node+488 */
        (char *)v20 + v358[12]                        /* follow to vec */
      + *(unsigned int *)((char *)v20 + v358[12])
      + 4);                                           /* vec[0] */
```

). Note that `vtable[12]` is **field 10**: the first four half-words of the
vtable are (`vt_size`, `tbl_size`, `field[0]_off`, `field[1]_off`), so
`vtable[12]` = field 10 offset.

## Per-op task count rule

Empirically (verified byte-exact against the vendor dump for all five
runtime models), the correct per-op task count is:

```
task_count(op) = f[10][1]  if f[10][2] == 0
                 f[10][0]  otherwise
```

`f[10][2] == 0` identifies ops with an activation LUT (`ConvSigmoid`,
`ConvExSwish`, `exSoftmax13`, …) that have an implicit pre-task to load
the LUT. For these ops `f[10][0]` is the CONV-only count and `f[10][1]`
is the total including the LUT load. For all other ops `f[10][0]` is the
total and `f[10][2]` holds a sub-count that is always non-zero.

For `InputOperator` and `OutputOperator`, task_count() may return 0 (the
usual case) or a non-zero value (DeepLabv3's InputOperator reformat,
which reserves 1 task slot but is never submitted).

## Per-op segment-flags rule

```
flags(op) = 0x1  if op.type in {"ConvSigmoid", "ConvExSwish", "exSoftmax13"}
            0x5  otherwise
```

This is the empirical LUT-activation set. Task 9.2's verification
(`--verify-segments`) tests that the resulting segment list matches the
vendor byte-exact across all five runtime models; if a new model
introduces a LUT-activation op type (e.g. `ConvExMish`, `ConvExHardSwish`),
add it to this set.

## The segmentation algorithm

```
cur_task_idx = 0
current_seg = None
segments = []

for op in subgraph.operators:
    tc = task_count(op)

    if op.type in {"InputOperator", "OutputOperator"}:
        # IO ops reserve task-BO slots (e.g. DeepLabv3's input reformat
        # at task[0]) but are never submitted. Flush the current segment
        # and advance cur_task_idx past their slots without contributing
        # to any segment.
        if current_seg is not None:
            segments.append(current_seg)
            current_seg = None
        cur_task_idx += tc
        continue

    if tc == 0:
        continue

    seg_flags = 0x1 if op.type in LUT_OPS else 0x5

    if current_seg is None or current_seg.flags != seg_flags:
        if current_seg is not None:
            segments.append(current_seg)
        current_seg = {start: cur_task_idx, count: 0, flags: seg_flags}

    current_seg.count += tc
    cur_task_idx += tc

if current_seg is not None:
    segments.append(current_seg)
```

The resulting list is byte-exact against the vendor's `submit_*.txt`
contents for all five models in
`openrknn/tests/segmentation_ground_truth/`.

## `task_number` from `sc_count`

The vendor's submit descriptor carries `task_number = sc_count ×
active_core_count` — a multi-core replica count. For RK3588 with
core_mask populated in `task->core_mask` the kernel expects
`task_number = 3 × sc_count`; for single-core submits it's
`task_number = sc_count`.

Our openrknn run path uses `core_mask = 0x1` (single core) because we
allocate one activation BO and don't partition IO. Task 9.3 sets
`task_number = sc_count` unconditionally and the kernel driver treats
the sub-core descriptors as replicas.

## What happens if the rule breaks for a new model

Add the new op type to either `LUT_OPS` or the IO set above, depending
on whether it takes flags=0x1 and whether it contributes executable tasks.
Task 9.2's `dump_segments.py --verify` catches any mismatch at the
artifact level before `orknn_own_init` fails at runtime.
