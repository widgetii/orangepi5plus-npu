# FP16 Transformer (ViT) OWN-path Support — Status and Next Steps

## TL;DR

openrknn's `ORKNN_OWN=...run...` path currently supports **INT8 CNN models only**
(mobilenet_v1, resnet50, yolov5/v8, deeplabv3). For FP16 transformer models —
specifically SmolVLM's 24 per-layer shards (SigLIP vision encoder, exported
with rknn-toolkit2 2.3.2 via the sandwich/nano-tiled pattern from
[poad42/smolvlm_rk3588_full_npu_native](https://github.com/poad42/smolvlm_rk3588_full_npu_native))
— the OWN path:

1. **Parser and segmenter both work fine.** The 24 shards are recognised,
   ops classified, memory plan built, multi-core submits assembled.
2. **Memory allocation is correct.** All five BOs (task/weight/activation/
   input/output) get allocated with the right sizes and flags (`0x40b` for
   task, `0x403` for the rest), matching what vendor's `librknnrt.so` does.
3. **`patch_regcmd_addresses` is incomplete for transformer op lowerings.**
   Specifically it misses register-level rules for `em=0x0d` (non-softmax
   general compute) and a few DMA-bearing registers that transformer shards
   populate but CNN shards leave at zero.
4. **A deeper problem exists below the patch layer.** Even when we load
   vendor's post-init weight BO verbatim and rebase every known DMA register
   (`ORKNN_ORACLE_PATCH` env — see below), the submit still times out on
   segment 0. Something else — cache sync, iommu domain, memory layout, or a
   kernel-side quirk we haven't identified — also contributes.

Earlier messaging in the parent conversation ("openrknn already runs
mobilesam_encoder at 2145 FPS") was wrong: mobilesam_encoder and lprnet bail
out of `extract_npu_data` at init-time with `regcmd(0) or taskbo(0) not
found` and silently fall through to the vendor proxy. **openrknn has never
actually run any FP16 transformer end-to-end in OWN mode.**

Read this doc end-to-end before starting any ViT-on-openrknn work.

---

## 1. What SmolVLM shards contain

Example: `l0_mlp.rknn` (one layer, unfused). Op types as parsed from the FB:

| idx | type        | inputs       | notes                                |
|-----|-------------|--------------|--------------------------------------|
| 0   | InputOperator | —          | no tasks                             |
| 1   | Reshape     | [12, 10, 13] | **consumes subgraph input**          |
| 2   | Mul         | [14, 8]      | sandwich ×10 scale                   |
| 3   | Transpose   | [15]         | 128 em=0x0d tasks, 12 KB stride      |
| 4   | exNorm      | [16,2,3,17,31,30] | LayerNorm with 4 aux blob refs  |
| 5   | ConvExSwish | [18, 6, 4]   | FFN expand (fused swish)             |
| 6   | Conv        | [19, 7, 5]   | FFN project                          |
| 7   | Transpose   | [20]         | output transpose                     |
| 8   | Add         | [15, 21]     | residual                             |
| 9   | Mul         | [22, 9]      | sandwich ÷10 descale                 |
| 10  | Reshape     | [23, 11, 24] | output reshape                       |
| 11  | OutputOperator | [25]      | no tasks                             |

Total 1183 tasks in the task BO across 3 FB-derived segments:

```
seg[0] flags=0x5 sc=[0..+171]   task_num=171   (pingpong)
seg[1] flags=0x1 sc=[171..+15]  task_num=15    (barrier)
seg[2] flags=0x5 sc=[186..+152] task_num=152   (pingpong)
```

Plus multi-core submit regions `[338..733]` (2-core) and `[733..1183]`
(3-core). The FB segmenter in `openrknn_model.c:fb_build_segments()` is
**already correct** for this shard — no changes needed there.

## 2. Enable-mask distribution

For `l0_mlp` the task-BO `enable_mask` histogram is:

| em     | count | meaning                                          |
|--------|-------|--------------------------------------------------|
| 0x0d   | 750   | **General CNA compute** (Transpose / exNorm / LUT LUTs) |
| 0x18   | 66    | REFORMAT (Reshape / Transpose staging / Concat)  |
| 0x1d   | 357   | Conv (standard CNN path — already works)         |
| 0x60   | 10    | PPU (pool-style or per-channel correction)       |

For `l0_attn.rknn` (unfused, uses `exMatMul`+`Softmax`+`Mul`): 19676 tasks,
mostly em=0x0d with a handful of em=0x60. Even bigger problem.

**Key correction to earlier belief**: `em=0x0d` does NOT mean "exSoftmax13
continuation task only". openrknn's `patch_regcmd_addresses` had a comment
saying so, but that's only true for CNN models which happen not to use
em=0x0d for anything else. In transformer models, em=0x0d is the generic
"CNA compute task without the full Conv data path" encoding and is used by
Transpose, exNorm, Mul, and anything else whose lowering walks data through
the CNA engine with a per-task LUT/coefficient blob.

## 3. Vendor's patching pattern (diffed against oracle)

The `.rknn` file's raw weight_data contains regcmd sections with DMA-bearing
registers set to **zero placeholders**. Vendor's runtime fills them in at
first-run time based on per-op metadata. openrknn must do the same.

Running vendor on l0_mlp and diffing the raw BO against vendor's post-init
BO shows vendor touches exactly these registers:

| Reg    | Name                       | What vendor puts there                   |
|--------|----------------------------|------------------------------------------|
| 0x0010 | PC_BASE_ADDRESS            | next task's regcmd section (chain ptr)  |
| 0x1070 | CNA_FEATURE_DATA_ADDR      | `act_base + src_tensor_off + tile_offset`|
| 0x1110 | CNA_DCOMP_ADDR0            | `wt_base + op_aux_blob_offset`          |
| 0x4020 | DPU_DST_BASE_ADDR          | `act_base + dst_tensor_off + tile_offset`|
| 0x4048 | DPU_BS_MUL_CFG             | occasional wt-rel pointer (LUT-ish)     |
| 0x5018 | DPU_RDMA_SRC_BASE_ADDR     | `in_base + val` (input-consuming op)    |
| 0x5020 | DPU_RDMA_BS_BASE_ADDR      | `wt_base + bias_blob`                   |
| 0x502c | DPU_RDMA_BN_BASE_ADDR      | `wt_base + bn_blob`                     |
| 0x5038 | DPU_RDMA_EW_BASE_ADDR      | `act_base + ...`                        |
| 0x504c | DPU_RDMA_EW variant        | `wt_base + LUT`                         |
| 0x6070 | PPU_DST_BASE_ADDR          | `wt_base + pc2_off` (em=0x60)           |
| 0x701c | PPU_RDMA_SRC_BASE_ADDR     | `wt_base + pc3_off` (em=0x60)           |

The **first pre-patch value** at each of these slots is 0 across all tasks
except a few that encode the intra-tile offset.

### 3.1 op-to-auxiliary-blob mapping (the blocker)

For `l0_mlp`, vendor's 0x1110 CNA_DCOMP_ADDR0 points to these weight-BO
offsets, grouped by op:

| op | type          | enable_mask | CNA_WT_BASE offset | Notes              |
|----|---------------|-------------|--------------------|--------------------|
| 3  | Transpose     | 0x0d        | wt + 0x910180      | 8192-byte type=6 blob (blob[17]) |
| 4  | exNorm        | 0x0d        | wt + 0x90f580 / 0x90fb80 | Two 1536-byte blobs (blob[15]/[16]) |
| 4  | exNorm        | 0x1d        | wt + 0x909180 / 0x90c580 | Two 12288-byte blobs (blob[12]/[14]) |
| 5  | ConvExSwish   | 0x1d        | wt + 0x007080      | 4 MB FC weight (blob[5]) — standard Conv path |
| 6  | Conv          | 0x1d        | wt + 0x487080      | 4 MB FC weight (blob[6]) — standard Conv path |
| 7  | Transpose     | 0x0d        | wt + 0x907180      | 8192-byte type=6 blob (blob[11]) |

The `em=0x1d` (standard Conv) offsets resolve correctly via openrknn's
existing `tensor_weight_blob[input_tensors[1/2]]` lookup — those work.

The **em=0x0d offsets do not**. These LUT/coefficient blobs are not
referenced by the op's `input_tensors[]` list at all. The op's FB record
has a 70-u32 auxiliary metadata field (FB field 9) that almost certainly
encodes these offsets, but its schema is undocumented. exNorm is even more
hairy because it references 4+ distinct blobs per op.

Without this mapping openrknn fills 0x1110 with `wt_base + 0`, the NPU's
CNA engine reads garbage as a permutation/LUT table, and segment 0 hangs
immediately.

**Two possible approaches** to unblock this:

1. **Decode FB field 9 schema**: reverse-engineer what the 70 u32s encode
   per op type. Probably contains `{blob_tidx, stride, count, ...}` in some
   form. Use the diff oracle (`diff_regcmd.py`) to validate each op type
   as rules are added.
2. **Scan-then-assign heuristic**: observe that all em=0x0d LUT blobs are
   `type=6` with specific sizes, and each op's lowering uses a deterministic
   blob assignment order. For l0_mlp the pattern holds — but it may break
   for multi-head attention shards. Fragile; use only as a temporary
   bootstrap.

## 4. The deeper blocker (ORACLE_PATCH mode)

I added a scouting `ORKNN_ORACLE_PATCH` mode
(`openrknn_run.c:patch_regcmd_addresses`) that **bypasses patching
entirely**: it loads vendor's post-init weight BO byte-for-byte, then walks
every task's regcmd slice and rebases all known DMA-bearing registers from
vendor's BO bases to openrknn's own.

```
ORKNN_ORACLE_PATCH=/tmp/rknn_dump/sub1_bo_001_9928704B.bin \
ORKNN_ORACLE_WT_BASE=0xff67c000 \
ORKNN_ORACLE_ACT_BASE=0xfdb6b000 \
ORKNN_ORACLE_IN_BASE=0xfd9eb000 \
ORKNN_ORACLE_OUT_BASE=0xfd86b000 \
  LD_PRELOAD=.../librknn_api.so ORKNN_OWN=... bench_throughput --model ...
```

Once rebased, the weight BO is byte-exact with what vendor's runtime set up.
All 12 known DMA registers are rebased to the right openrknn BO offsets.
**This STILL hangs on segment 0 submit.**

That tells us *weight-BO content alone is not sufficient* — vendor does
something else during init or submit that we haven't matched. Candidates
(not yet verified; investigation ended when the night ran out):

- **IOMMU domain**: vendor passes `iommu_domain_id=0` (matches openrknn,
  confirmed via extended intercept_swap logging). Ruled out.
- **task_base_addr**: vendor passes `0` (matches openrknn). Ruled out.
- **task_obj_addr**: vendor and openrknn both pass valid kernel gem object
  pointers, different but both valid. Probably fine.
- **Cache sync ordering**: openrknn does `MEM_SYNC(TO_DEVICE)` after both
  initial load and post-patch. Vendor likely does the same. Probably fine.
- **Memory allocation layout**: vendor's activation BO is 28 MB
  (`3 × 9.3 MB`), openrknn allocates 9.3 MB (single copy). Max observed
  regcmd activation offset is 7.9 MB, fits in both — but the NPU kernel
  module may require contiguous allocation of all cores' slices even for
  single-core submits. **High priority to test**: allocate activation BO
  at 3× size and see if oracle-patch mode stops hanging.
- **Some register class we haven't identified as DMA**: ran a full scan
  of the oracle BO for all values in any BO range. Found `0x4048` and
  `0x504c` — those are now in the oracle-patch rebase list but hang
  persists.

The fastest way to narrow this down is probably a **single-task test**:
isolate task 0 (op=1 Reshape, em=0x18, just reads from input BO) and submit
it alone. If that one task hangs too, the problem is outside regcmd
content. If it succeeds, recur into subsequent tasks to find the first
failure.

## 5. What landed this session

All committable changes live in `openrknn_run.c` / `openrknn_drm.c`:

- **`MAX_PATCHED_OFFSETS` dynamic sizing**
  (`openrknn_run.c:~351`). The old hard cap of 4096 was sized for DeepLabv3
  (1858 sections). SmolVLM shards hit 17k unique sections and corrupted
  past-the-cap patches with random rewrites. Replaced with
  `max_patched_offsets = m->task_count` (strict upper bound). Strict
  bugfix, no known regression on CNN models (verified on mobilenet_v1,
  yolo, deeplabv3, mobilesam_encoder). **Safe to upstream standalone.**

- **em=0x18 input-consuming Reshape rule**
  (`openrknn_run.c` 0x5018 handler). When the op that consumes the
  subgraph input is itself a REFORMAT (not a Conv — e.g. SmolVLM's op-1
  Reshape), `DPU_RDMA_SRC_BASE_ADDR` must point at the input BO, not the
  activation BO. Added an `is_reformat && is_input_consuming_task_for_src`
  branch. Verified via `diff_regcmd.py`: task 0/1 DMA-class diff
  disappears after this fix.

- **em=0x0d non-softmax CNA_FEATURE_DATA_ADDR rule**
  (`openrknn_run.c` 0x1070 handler). The old fallback `else if (val != 0)
  { new_val = act_base + val }` misses tasks whose raw val is 0 but whose
  semantic offset is 0 (Transpose tile 0, etc). Added an `enable_mask ==
  0x0d` branch that always resolves from `act_base + src_tensor_off + val`.
  Verified via diff: the 0x1070 slot is patched correctly for
  Transpose/exNorm tasks on l0_mlp.

- **Dev scaffolding** (sticky — keep for future ViT work):
  - `ORKNN_DEBUG_BLOBS=1` — dump the full weight-BO blob layout at run
    start.
  - `ORKNN_DEBUG_PATCH=1` — log raw pre-patch values for 0x1070/0x1110/
    0x5018/0x4020 on the first 40 tasks (lookup hoisted out of the hot
    path).
  - `ORKNN_DUMP_BO1_PRE=/path` — dump the weight BO *before* patching
    runs. Pairs with `tests/diff_regcmd.py` to identify which registers
    vendor fills in that openrknn leaves alone.
  - `ORKNN_ORACLE_PATCH=/path ORKNN_ORACLE_{WT,ACT,IN,OUT}_BASE=0x...` —
    bypass patching and load vendor's post-init BO verbatim (with DMA
    rebase). Useful for isolating patch bugs from everything-else-bugs.

No CNN model behavior changed. The new branches only fire on
REFORMAT-with-input-consuming-op or em=0x0d tasks, neither of which appears
in the existing CNN test suite. Verified smoke: mobilenet_v1 runs fine.

## 6. Next-step playbook for whoever picks this up

In order of increasing difficulty:

### Step A — resolve the ORACLE_PATCH hang

This is the biggest open question. Without this we don't even know if the
patch rules are the only missing piece. Concrete experiments:

1. **Activation BO size**: bump openrknn's activation allocation to match
   vendor's (3× the single-core size, or outright 28 MB for SmolVLM shards).
   See `openrknn_memory.c:~135` — the activation BO size computation.
2. **Single-task isolation**: add an `ORKNN_MAX_TASKS=N` env var that
   truncates the first submit to N tasks. Find the first N that hangs;
   that N-1'th task is the first corrupt one.
3. **Compare submit ioctl arg bytes**: hex-dump openrknn's `struct
   rknpu_submit` and vendor's (via extended intercept_swap) byte-for-byte.
   If anything differs beyond `task_obj_addr`, that's the bug.
4. **Per-segment sync**: try adding an explicit `MEM_SYNC(TO_DEVICE)` for
   task BO, weight BO, and activation BO immediately before each submit
   ioctl (not just at init time).
5. **IOMMU attach state**: there's a kernel issue where re-attach on the
   same IOMMU group between runs adds latency/hangs. Our vendor kernel
   has an out-of-tree patch for this
   (`patches/kernel/0001-rocket-iommu-attach-caching*.patch`). Check the
   vendor kernel's rknpu driver version loaded on the board — if IOMMU
   caching isn't in, that could be the culprit for transformer shards
   specifically.

### Step B — decode FB field 9 for em=0x0d ops

Once ORACLE_PATCH runs clean, the remaining work is teaching
`patch_regcmd_addresses` to compute the op-specific LUT offsets without
the oracle. This means:

1. Pick ONE op type (start with Transpose — simplest, single LUT reference).
2. Dump field 9 for all Transpose ops in l0_mlp, l0_attn, and compare.
   Look for common patterns: a u32 that encodes the blob tensor index,
   or a byte-offset that maps to wt_blob_offsets[].
3. Write the extraction code in `parse_fb_operators()` and store the
   resolved blob offset in `struct orknn_op_info`.
4. Add a case to the 0x1110 switch: `if (enable_mask == 0x0d && op_type
   == "Transpose") { new_val = wt_base + op_aux_blob_off + val; do_patch
   = 1; }`.
5. Validate via `diff_regcmd.py`: task 3-N (Transpose) should show zero
   non-DMA diffs AND zero DMA-class diffs.
6. Repeat for exNorm (harder — multiple blobs per op), Mul (unknown
   structure), etc.

Target progression: l0_mlp byte-exact → l0_mlp end-to-end correct output →
l0_attn byte-exact → all 24 shards → full SmolVLM vision encoder under
`validate.py` with cosine > 0.99 vs vendor.

### Step C — lift FB extractor for mobilesam / lprnet

The FP16 models already in the CI suite (`mobilesam_encoder`, `lprnet`)
fail much earlier — at `extract_npu_data()` with "regcmd(0) or taskbo(0)
not found". Their `.rknn` files use a different FB field layout that
openrknn's extract code doesn't recognise. Once SmolVLM shards run, it
makes sense to also generalise `extract_npu_data` so mobilesam/lprnet can
go through the OWN path too. Expected small-ish change: `openrknn_model.c`
walks weight_data vector searching for the regcmd/taskbo blobs, just needs
additional heuristics.

## 7. Tooling references

- **Oracle capture**: `librocketnpu/tests/intercept_swap.c` LD_PRELOAD
  (`DUMP_ALL_BOS=1`). Produces `/tmp/rknn_dump/sub1_bo_{000..004}.bin` +
  `submit_1.txt`. Pre-run BO state (valid for diffing). Also logs submit
  parameters: `task_obj_addr`, `core_mask`, `task_base_addr`,
  `iommu_domain_id`, full `subcore_task[0..2]` ranges. Rebuild on the board
  with `gcc -shared -fPIC -o intercept_swap.so intercept_swap.c -ldl`.
- **Template dump**: `ORKNN_DUMP_BO1=/path`,
  `ORKNN_DUMP_BO1_PRE=/path`,
  `ORKNN_DUMP_TASKBO=/path`.
- **Diff tool**: `openrknn/tests/diff_regcmd.py` with `--oracle`,
  `--template`, `--task-bo`, `--submit-txt` flags. Surfaces non-DMA
  register differences (template-patch bugs) and flags DMA-class
  mismatches separately. Supports `--em-filter 0x1d,0x18,0x0d`.
- **Raw task BO decoder**: the Python scripts in this doc use
  `struct.unpack_from('<8I Q', task_bo, i*40)` for the 40-byte
  `rknpu_task` layout. Fields are `{flags, op_idx, enable_mask, int_mask,
  int_clear, int_status, regcfg_amount, regcfg_offset, regcmd_addr}`.

## 8. Reference data from this session

For `l0_mlp.rknn` (use as a known-good bisection target):

```
weight BO size:   9928704 bytes (0x974000)
task BO size:     47320 bytes (used) / 49152 allocated
regcmd blob:      offset 0x916580, size 352448
task BO blob:     offset 0x96c640, size 47320
FB segments:      [0..171) flags=0x5, [171..186) flags=0x1, [186..338) flags=0x5
total tasks:      1183 (covers single-core + 2-core + 3-core regions)
op count:         12 (including InputOperator + OutputOperator)
tensor count:     38
weight blob count: 23
```

Blob layout (from `ORKNN_DEBUG_BLOBS=1`):

```
blob[ 0] off=0x00000000 size=    8776 type=0   # weight A
blob[ 1] off=0x00002280 size=    1536 type=0   # bias A
blob[ 2] off=0x00002880 size=    3072 type=4
blob[ 3] off=0x00003480 size=   12288 type=6   # layernorm gamma / 12KB LUT
blob[ 4] off=0x00006480 size=    3072 type=6
blob[ 5] off=0x00007080 size= 4718592 type=6   # FFN weight 1 (ConvExSwish)
blob[ 6] off=0x00487080 size= 4718592 type=6   # FFN weight 2 (Conv)
blob[ 7] off=0x00907080 size=       4 type=6   # scalar
blob[ 8] off=0x009070c0 size=       4 type=6   # scalar
blob[ 9] off=0x00907100 size=      32 type=6
blob[10] off=0x00907140 size=      24 type=6
blob[11] off=0x00907180 size=    8192 type=6   # Transpose LUT (op 7)
blob[12] off=0x00909180 size=   12288 type=6   # exNorm weight
blob[13] off=0x0090c180 size=    1024 type=6   # pc3 (per-channel correction)
blob[14] off=0x0090c580 size=   12288 type=6   # exNorm bias
blob[15] off=0x0090f580 size=    1536 type=6   # exNorm LUT 1
blob[16] off=0x0090fb80 size=    1536 type=6   # exNorm LUT 2
blob[17] off=0x00910180 size=    8192 type=6   # Transpose LUT (op 3)
blob[18] off=0x00912180 size=    1024 type=6   # pc2 (per-channel correction)
blob[19] off=0x00912580 size=    8192 type=6
blob[20] off=0x00914580 size=    8192 type=6
blob[21] off=0x00916580 size=  352448 type=6   # regcmd blob
blob[22] off=0x0096c640 size=   47320 type=6   # task BO blob
```

Two unexplained 8192-byte LUTs at blobs 19 and 20 — probably bonus Transpose
LUTs used by the multi-core regions we didn't diff in detail.

## 9. Session notes / things NOT to re-litigate

- **task_number = 3 * seg**: don't. Kernel's
  `rknpu_get_task_number()` in `drivers/rknpu/rknpu_job.c` returns the
  per-core subcore task count, not the `args->task_number` field directly
  (for `use_core_num == 1 || 2`). Setting `task_number` to 3× just causes
  a 2× perf regression on CNN models without changing correctness.
- **iommu_domain_id / task_base_addr nonzero**: vendor passes both as 0 on
  RK3588 for the shards we tested. Confirmed via extended intercept_swap
  logging.
- **RKNN_FLAG_COLLECT_MODEL_INFO_ONLY shortcut**: doesn't help — this flag
  skips delegate_init entirely; we still need the run path.
