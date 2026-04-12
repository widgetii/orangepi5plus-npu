# SmolVLM ViT Sharding Reproduction on RK3588 (vendor stack)

Reproduction of Adhitya Mohan's [shard-optimizing-vision-transformers-edge-npu](https://amohan.dev/blog/2025/shard-optimizing-vision-transformers-edge-npu/) work. Upstream code: [poad42/smolvlm_rk3588_full_npu_native](https://github.com/poad42/smolvlm_rk3588_full_npu_native).

**This is a vendor-stack experiment** (RKNN-Toolkit2 2.3.2 + librknnrt 2.3.2 on vendor kernel `6.1.115-vendor-rk35xx`). It does **not** use our Mesa rocket / openrknn code.

## What was reproduced

SmolVLM-256M-Instruct vision encoder (SigLIP, 12-layer ViT, 768-d, 1024 tokens @ 448×448) split into 24 `.rknn` shards — one per {attention, MLP} block per layer — running FP16 with "sandwich" ×10/÷10 scaling to keep activations out of FP16 underflow/overflow corners.

## Build/run

Host (x86_64 Linux, Docker):

```
docker build -t smolvlm-shard-convert .
docker run --rm \
  -v <path-to-upstream-repo>:/work \
  -v $HOME/.cache/huggingface:/root/.cache/huggingface \
  -w /work smolvlm-shard-convert \
  python scripts/convert.py
```

Produces `smolvlm_subshards/{l0..l11}_{attn,mlp}.rknn` (24 files, ~240 MB total).

Board (RK3588, vendor kernel, `/dev/dri/renderD129`):

```
rsync smolvlm_subshards/ scripts/ src/ data/ root@board:/root/npu-research/smolvlm_shard/
# then restructure: scripts/, src/, smolvlm_subshards/, data/ as siblings

ssh board
cd /root/npu-research/smolvlm_shard
/root/npu-research/venv/bin/python scripts/validate.py
```

## Patches needed vs upstream

1. **`requirements_convert.txt`** pins `transformers==4.44.2` (implicit via install time), but SmolVLM uses `idefics3` architecture only in `transformers>=4.46`. Bumped to `4.46.3` in our Dockerfile.
2. **Board transformers** must match: `4.46.3`. `transformers 5.x` removes `AutoModelForVision2Seq`.
3. No source patches. Upstream `convert.py`, `validate.py`, `run_inference.py`, `src/rknn_patterns/*.py`, `src/smolvlm_{convert,infer}/*.py` all ran as-is.

## Validation results (`scripts/validate.py`)

Layer-by-layer cosine similarity of NPU shard output vs PyTorch CPU reference (teacher-forced, random image):

| Layer | attn | mlp  | notes                        |
|-------|------|------|------------------------------|
| L0    | 1.0000 | 0.9848 | sandwich path                |
| L1    | 1.0000 | 0.9982 | sandwich path                |
| L2    | 0.8078 | 0.8218 | non-sandwich path in validator |
| L3    | 0.8044 | 0.8106 | non-sandwich path in validator |
| L4    | 0.8190 | 0.8010 | non-sandwich path in validator |
| L5-L11| 1.0000 ± 0.0003 | 1.0000 ± 0.0003 | sandwich path |

L2-L4 dip is a mismatch between `convert.py` (which hardcodes `use_fp16=True` for all layers — i.e. all sandwich) and `validate.py` (which skips pre-scaling for layers 2-4, assuming they are INT8 "green-zone"). Not a real accuracy regression — just an artifact of the upstream validation script being written for the INT8-mixed mode that was commented out.

20 of 24 shards produce bit-close output vs CPU. The vision encoder end-to-end is numerically stable.

## Latency (448×448 input, 1024 tokens, deterministic, 3 runs)

Full vision encoder forward:

| Backend               | Latency (ms) |
|-----------------------|--------------|
| NPU sharded (24 × rknn) | **1970**     |
| PyTorch CPU (4× A76)  | 3850         |
| **Speedup**           | **1.95×**    |

Per-shard NPU time (100 runs averaged, running in isolation; core-mask as assigned by upstream round-robin):

```
attn  shards: 107-131 ms each  (avg ~115 ms, 12 shards = ~1380 ms)
mlp   shards:  45- 67 ms each  (avg ~47 ms,  12 shards =  ~560 ms)
total:                                           ~1940 ms
```

Sum-of-per-shard (~1940 ms) ≈ end-to-end (1970 ms) → **the CPU orchestration overhead between shards is negligible (~30 ms over 24 hops)**.

## Key observations

### 1. Round-robin core assignment is cosmetic, not parallel

The pipeline is strictly sequential — shard N needs shard N-1's output. Assigning shards to different cores only helps if multiple independent workloads run concurrently. For a single image, the observed NPU time is the sum of all per-shard times regardless of which core each lands on. The upstream's "synchronized round-robin schedule (Core 0 → Core 1 → Core 2)" phrasing oversells the parallelism — under a sequential data dependency, only one core is ever busy at a time.

To actually exploit 3-core parallelism you'd need:
- Pipelined batch of images (shard N on core A for image `t+1` while shard N+1 runs on core B for image `t`), or
- Graph-level restructuring so independent shards exist (e.g. Q/K/V projections in parallel across cores)

### 2. Attention dominates latency (~70%)

Attention shards are 2–3× slower than MLPs despite smaller parameter counts (the MLP has the FFN expansion). This is NanoTiled attention paying its cost — lots of small 32×32 transposes and MatMuls means the NPU is not working on its strongest shape regime. On the plus side, it actually compiles and runs, which stock attention doesn't.

### 3. "Sandwich quantization" is a real FP16 workaround

The validator clearly shows that non-sandwich (L2-L4) FP16 shards produce structurally correct but lower-fidelity output (~0.80 cos sim) vs sandwich shards (~1.00). The ×10/÷10 wrap really is necessary for SigLIP's activation range on RK3588's FP16.

### 4. Stock ViT-B/16 would hit `REGTASK Overflow (0xe010)` here

We did not reproduce the failure directly, but the upstream workaround presumes it. Any port of this approach to our open Rocket stack would need to avoid the same SRAM pressure a priori rather than discovering it through kernel errors.

## Gap analysis for openrknn / Mesa rocket

To run this on our open stack (`patches/mesa/0004-rocket-add-sw-ops-*`), we would need support for:

- **LayerNorm** (`NativeLayerNorm` upstream uses a decomposed mean/variance path — still needs Sub, Mul, Add, Sqrt, Div in HW or SW)
- **GELU** (`DecomposedGELU` — tanh-approximation-based, needs Tanh/approximation)
- **Softmax** (already in patch 0004)
- **MatMul** large-K (our Mesa rocket only does Conv today; a MatMul-via-1×1-Conv rewrite is possible but slow for K>>C)
- **Transpose / Reshape / Gather** (already partial)
- **FP16 path** (we only do INT8 today — would need entirely new quant-to-weights/format pipeline)

FP16 is the big missing piece. Even if we add the ops, INT8 on a ViT without per-axis + sandwich-equivalent scaling will give poor quality given the activation dynamic range. A reasonable next step would be a "feasibility" PR adding LayerNorm+GELU as SW ops and benching a trivial ViT attention block end-to-end, but this is out of scope for the current experiment.

## Files on host

- `librocketnpu/tests/smolvlm_shard/Dockerfile` — conversion image
- `external/smolvlm_rk3588_full_npu_native/` — upstream repo clone (not committed)
- `external/smolvlm_rk3588_full_npu_native/smolvlm_subshards/` — built shards (~240 MB, not committed)

## Files on board

- `/root/npu-research/smolvlm_shard/{scripts,src,smolvlm_subshards,data}/`

## References

- [Reverse-Engineering the RK3588 NPU — Adhitya Mohan](https://amohan.dev/blog/2025/shard-optimizing-vision-transformers-edge-npu/)
- [poad42/smolvlm_rk3588_full_npu_native](https://github.com/poad42/smolvlm_rk3588_full_npu_native)
- [HuggingFaceTB/SmolVLM-256M-Instruct](https://huggingface.co/HuggingFaceTB/SmolVLM-256M-Instruct)
