# openrknn — RKNN API reimplementation without vendor binaries

A from-scratch implementation of the Rockchip RKNN user-space API
(`librknn_api.so`) that runs `.rknn` models on the RK3588 NPU without
`librknnrt.so` or any other vendor binary at runtime.

Drop-in compatible with apps that link against the vendor library: same
ABI, same symbol names, same behavior on every public call. When the
vendor library is installed it is still used as a fallback for features
openrknn hasn't implemented yet (zero-copy I/O, custom ops, dynamic
shapes — see the [roadmap](#roadmap)). When it is absent, openrknn's
own code path takes every inference end-to-end.

Part of the larger [orangepi5plus-npu][repo] reverse-engineering
project. See #68 for the live roadmap and progress tracking.

[repo]: https://github.com/widgetii/orangepi5plus-npu

## What actually works

On a freshly booted Orange Pi 5 Plus with `/lib/librknnrt.so` renamed
out of the way and `/tmp/rknn_dump` removed entirely, openrknn runs
all seven models in the CI suite:

| Model | Input | Task | Result |
|-------|-------|------|--------|
| `mobilenet_v1` | 224×224 UINT8 | ImageNet classification | top-1 = 156, score 0.930 |
| `resnet50-v2-7` | 224×224 UINT8 | ImageNet classification | top-1 = 155 |
| `yolov5s_relu_int8` | 640×640 UINT8 | COCO detection | 4 person detections |
| `yolov8n` | 640×640 UINT8 | COCO detection | 4 person + 1 bus |
| `deeplabv3` | 513×513 UINT8 | semantic segmentation | class 0/2/15 regions |
| `mobilesam_encoder` | 448×448 FP16 | image encoder (init + query only) | PASS |
| `lprnet` | 94×24 FP16 | license plate net (init + query only) | PASS |

All seven run under `ci_validate.sh` Phase 2, which is the "no vendor
deps" gate: every call goes through openrknn's own init/parse/patch/
submit path, and `grep -rn "/tmp/rknn_dump" src/` returns zero matches.

## Usage

openrknn has two dispatch modes:

**Proxy mode** (default) is an LD_PRELOAD-compatible shim. It dlopens
`librknnrt.so` and forwards every public API call, with optional
interception hooks. Existing applications upgrade transparently:

```bash
LD_PRELOAD=./librknn_api.so ./your_rknn_app
```

**OWN mode** is the from-scratch path. The `ORKNN_OWN` env var lists
which API calls should run through openrknn instead of being proxied:

```bash
# Full OWN pipeline — no vendor code involved in inference
ORKNN_OWN=init,query,input,run,outputs \
    LD_PRELOAD=./librknn_api.so ./your_rknn_app

# Mixed — OWN init/query, vendor run (useful for debugging)
ORKNN_OWN=init,query \
    LD_PRELOAD=./librknn_api.so ./your_rknn_app
```

Each comma-separated token turns on one openrknn code path:

| Token | Replaces |
|-------|----------|
| `init` | `rknn_init` — parse the .rknn FlatBuffer, extract weights / regcmd / task BO |
| `query` | `rknn_query` — answer tensor attribute queries from parsed FB metadata |
| `input` | `rknn_inputs_set` — NHWC→NC1HWC2 layout transform and DMA sync |
| `run` | `rknn_run` — DMA-address patching, per-segment NPU submit |
| `outputs` | `rknn_outputs_get` — NC1HWC2→NHWC detile, optional dequantize |

All other API calls (custom ops, dynamic shapes, zero-copy memory,
multi-core batch) fall through to the vendor library when it is
present. See the [roadmap](#roadmap) for the in-progress OWN-mode
coverage of those.

## Build

Native on the Orange Pi 5 Plus (aarch64, Armbian 25.11.1 Noble):

```bash
apt install libdrm-dev python3-pil
make -C openrknn           # produces openrknn/librknn_api.so
```

No external dependencies beyond `libdrm`. The build is a handful of
`.c` files linked into a single shared library that exposes the RKNN
ABI.

## Run the test suite

The test suite is run on-board against the real NPU (no QEMU / CPU
emulation). `ci_validate.sh` builds openrknn and runs four phases:

```bash
cd openrknn && bash tests/ci_validate.sh
```

| Phase | What it tests | Vendor lib required? |
|-------|---------------|---------------------|
| Phase 1 | vendor `librknnrt.so` baseline — sanity check on models + images | yes (skipped if absent) |
| Phase 2 | openrknn OWN path — the authoritative test | **no** |
| Phase 2.5 | byte-exact regcmd diff: patched weight BO vs vendor oracle | yes (skipped if absent) |
| Phase 3 | openrknn proxy-dispatch path — LD_PRELOAD without `ORKNN_OWN` | yes (skipped if absent) |

Phase 2 is the one that proves openrknn is self-sufficient. Run it
standalone on a clean board to confirm:

```bash
mv /lib/librknnrt.so /lib/librknnrt.so.hidden
rm -rf /tmp/rknn_dump
bash openrknn/tests/ci_validate.sh          # Phase 2 should pass 7/7
mv /lib/librknnrt.so.hidden /lib/librknnrt.so
```

Expected output when the vendor library is hidden:

```
=== Phase 1: vendor librknnrt.so baseline (SKIPPED — no vendor lib) ===
=== Phase 2: openrknn OWN path (no vendor deps) ===
  Phase result: 7/7 passed
=== Phase 2.5: template-patch byte-exact diff (SKIPPED) ===
=== Phase 3: openrknn proxy-dispatch path (SKIPPED — no vendor lib) ===
=== All openrknn CI validation tests passed ===
```

## Source layout

```
openrknn/
├── src/
│   ├── openrknn.h            Internal context + struct definitions
│   ├── openrknn_api.c        Public ABI entry points (rknn_init, _run, ...)
│   ├── openrknn_proxy.c      dlopen(librknnrt.so) + function table
│   ├── openrknn_flatbuf.c    Minimal FlatBuffer reader (~100 lines)
│   ├── openrknn_model.c      .rknn parser: header, JSON config, FB ops,
│   │                         tensor memory plan, task BO extraction,
│   │                         fb_build_segments — see docs/
│   ├── openrknn_drm.c        DRM IOCTL submission to /dev/dri/renderD129
│   ├── openrknn_memory.c     Weight/task/activation/input/output BO alloc
│   ├── openrknn_input.c      NHWC→NC1HWC2 layout transform, input CVT
│   ├── openrknn_run.c        DMA-address patching, per-segment submit,
│   │                         InputOperator CVT register fixup
│   ├── openrknn_output.c     NC1HWC2→NHWC detile, dequantize
│   └── openrknn_query.c      rknn_query cases (IN_OUT_NUM, INPUT_ATTR,
│                             OUTPUT_ATTR, SDK_VERSION, ...)
├── include/
│   └── rknn_api.h            Public header matching the vendor ABI
├── docs/
│   ├── README.md (this file)
│   └── segmentation_from_fb.md   How fb_build_segments turns the FB
│                                  operator graph into RKNPU submit
│                                  segments (canonical reference)
├── tests/
│   ├── ci_validate.sh            4-phase CI harness
│   ├── validate_accuracy.py      Per-model accuracy validation (ctypes
│   │                             wrapper around the RKNN API)
│   ├── ground_truth.json         Per-model expected outputs + test images
│   ├── test_images/              Input images for the 5 runtime models
│   ├── dump_segments.py          Capture vendor submit plan + FB op
│   │                             table; --verify reproduces the rule
│   ├── segmentation_ground_truth/  Per-model JSON artifacts used by
│   │                                --verify (committed, idempotent)
│   ├── diff_regcmd.py            Byte-exact weight-BO diff vs vendor
│   │                             oracle (drives Phase 2.5)
│   ├── probe_fb_schema.py        Interactive FlatBuffer schema walker
│   └── postprocess.py            Model-specific post-processing for
│                                 classification, detection, segmentation
├── Makefile
└── librknn_api.so (build output, .gitignored on fresh clones)
```

## How openrknn parses a .rknn file

This is a quick tour for contributors who want to jump in. See
`src/openrknn_model.c:orknn_own_init` for the full implementation.

1. **Validate the container.** Raw RKNN files start with the 4-byte
   magic `"RKNN"`. Version-dependent header layout sets the FlatBuffer
   start offset (`0x40` for v>1).
2. **Parse the legacy JSON config.** A jsonxx-serialized block before
   the FlatBuffer carries `target_platform`, `input_num`, connection
   info, and preliminary tensor metadata. `json_*` helpers in
   `openrknn_model.c` handle this without a full JSON parser.
3. **Parse the FlatBuffer root.** Subgraphs live in root field 2.
   `sg.f[0]` is the tensor vector, `sg.f[1]` is the operator vector.
4. **Extract NPU data** (`extract_npu_data`). Walks the weight_data
   vector (field 4 for v≤5, field 20 for v>5), collects byte blobs,
   assembles them into a contiguous BO[1] layout. Finds the task BO
   (the blob containing 40-byte `rknpu_task` records with valid
   enable_mask values) and the regcmd blob.
5. **Extract tensor metadata** (`extract_fb_tensors`). Per-tensor
   scale, zero-point, format, native shape, element type.
6. **Parse operator graph** (`parse_fb_operators`). Per-op:
   - `f[1]` → type string (`"Conv"`, `"ConvSigmoid"`, …)
   - `f[4]` → input tensor indices
   - `f[5]` → output tensor indices
   - `f[10]` → task-count vector; `[1]` if `[2]==0` else `[0]`
     (captures activation-LUT pre-tasks for Sigmoid/Swish/Softmax)
7. **Parse tensor memory plan** (`parse_fb_tensor_offsets`).
   Per-tensor byte offset into the activation BO (`f[13]`) and
   weight blob index (`f[18]`).
8. **Build submit segments** (`fb_build_segments`). Walks the op
   list and groups ops by submit-flag class (`0x5` pingpong for
   most, `0x1` barrier for LUT-activation ops). The resulting
   segment list is byte-exact against the vendor's `submit_*.txt`
   capture — see [`docs/segmentation_from_fb.md`][seg] for the
   full derivation.
9. **Allocate DMA buffers** (`orknn_alloc_model_bos`). Weight,
   task, activation, input, and output BOs via `orknn_bo_create`
   against `/dev/dri/renderD129`.
10. **Copy task data, rebase regcmd pointers.** The `.rknn`'s
    regcmd\_addr fields are self-relative offsets; we add our
    weight BO's DMA base.

On the first `rknn_run`, `patch_regcmd_addresses` walks every task,
finds the DMA-bearing register writes in its regcmd slice, and
rebases them from the compiler's placeholder values to the BO DMA
addresses we actually allocated.

[seg]: docs/segmentation_from_fb.md

## Roadmap

Phase 9 landed on 2026-04-11 — the OWN path is now self-sufficient.
Further work is tracked in individual GitHub issues:

| # | Priority | Feature |
|---|---|---|
| [#58][i58] | High | Zero-copy I/O via `rknn_create_mem` / `rknn_set_io_mem` |
| [#59][i59] | High | DMA-BUF import (`rknn_create_mem_from_fd` / `_from_phys`) |
| [#60][i60] | Medium | Multi-core batch inference (`rknn_set_batch_core_num`) |
| [#61][i61] | Medium | Dynamic-shape models (`rknn_set_input_shape(s)`) |
| [#62][i62] | Medium | Custom operators (`rknn_register_custom_ops`) |
| [#63][i63] | Medium | RNN/LSTM operator support |
| [#64][i64] | Low | Encrypted `.rknn` container (CYPTKNNR) |
| [#65][i65] | Low | Tighten activation BO sizing (remove 1.2x–2.1x slack) |
| [#66][i66] | Medium, help wanted | Latency benchmarking vs vendor |
| [#67][i67] | Low, help wanted | CI matrix job: Phase 2 with vendor lib hidden |

[i58]: https://github.com/widgetii/orangepi5plus-npu/issues/58
[i59]: https://github.com/widgetii/orangepi5plus-npu/issues/59
[i60]: https://github.com/widgetii/orangepi5plus-npu/issues/60
[i61]: https://github.com/widgetii/orangepi5plus-npu/issues/61
[i62]: https://github.com/widgetii/orangepi5plus-npu/issues/62
[i63]: https://github.com/widgetii/orangepi5plus-npu/issues/63
[i64]: https://github.com/widgetii/orangepi5plus-npu/issues/64
[i65]: https://github.com/widgetii/orangepi5plus-npu/issues/65
[i66]: https://github.com/widgetii/orangepi5plus-npu/issues/66
[i67]: https://github.com/widgetii/orangepi5plus-npu/issues/67

The umbrella [#68][i68] tracks overall progress and lists every
completed development phase (0–9) with commit hashes.

[i68]: https://github.com/widgetii/orangepi5plus-npu/issues/68

## Contributing

**Adding a new runtime model:**

1. Drop the `.rknn` file on the CI runner at `/root/npu-research/`
2. Add an entry to `tests/ground_truth.json` with the model path,
   test image, and expected output
3. Capture a ground-truth segmentation artifact on the board:
   ```bash
   python3 tests/dump_segments.py --all \
       --bench-dir /root/npu-research/librocketnpu/tests \
       --models-dir /root/npu-research \
       --ground-truth tests/ground_truth.json \
       --out-dir tests/segmentation_ground_truth/
   ```
4. Verify the FB-derived rule still matches:
   ```bash
   python3 tests/dump_segments.py --verify tests/segmentation_ground_truth/
   ```
5. Run `bash tests/ci_validate.sh` — Phase 2 should pass on the new
   model. If it doesn't, the model introduces an op type that isn't
   yet in `LUT_OPS` or `IO_OPS` in `src/openrknn_model.c`. Add it
   there and update [`docs/segmentation_from_fb.md`][seg].

**Filing a bug:**

Attach the full `tests/ci_validate.sh` output. Phase 2 failures are
real openrknn regressions; Phase 2.5 failures are template-patch
regressions that `diff_regcmd.py` can localize to specific
(task, register) pairs.

**Reverse-engineering references:**

The vendor runtime has been extensively reverse-engineered. When
investigating how something works on the vendor side:

- `~/projects/ida/librknnrt/docs/*.md` — per-subsystem writeups
  (graph-executor, model-parsing, memory-management, execution-engine,
  device-driver, rknn-model-format, …)
- `~/projects/ida/librknnrt/librknnrt.so.c` — full Hex-Rays decompile,
  useful for cross-referencing FB-field-to-runtime-offset mappings
  (`sub_2DC018` is the v>5 operator extraction function that writes
  node+488 = f[10][0])

## License

MIT. See the SPDX headers in individual source files.
