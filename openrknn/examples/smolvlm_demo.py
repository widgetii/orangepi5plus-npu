#!/usr/bin/env python3
"""
SmolVLM vision-language demo on Orange Pi 5+ NPU — powered by openrknn.

Runs the SmolVLM-256M vision encoder (24 FP16 transformer shards) on the
RK3588 NPU via openrknn, then feeds embeddings to the language model on
CPU to generate a natural-language image description.

Usage:
    python3 smolvlm_demo.py --image photo.jpg
    python3 smolvlm_demo.py --video clip.mp4 --frame 100
    python3 smolvlm_demo.py --image photo.jpg --compare
    python3 smolvlm_demo.py --image photo.jpg --prompt "What animals are in this image?"

Requirements (Orange Pi 5+ with vendor kernel 6.1.x):
    source /root/npu-research/venv/bin/activate
    # numpy, torch, transformers, pillow must be installed
    # SmolVLM model downloads on first run (~500 MB)
"""

import argparse
import ctypes
import os
import subprocess
import sys
import time

import numpy as np

# ---------------------------------------------------------------------------
# ctypes RKNN API definitions (from openrknn/tests/validate_accuracy.py)
# ---------------------------------------------------------------------------

RKNN_TENSOR_FLOAT16 = 1
RKNN_TENSOR_FLOAT32 = 0
RKNN_TENSOR_NHWC = 1
RKNN_QUERY_IN_OUT_NUM = 0
RKNN_QUERY_INPUT_ATTR = 4  # native input attr


class rknn_input_output_num(ctypes.Structure):
    _fields_ = [("n_input", ctypes.c_uint32), ("n_output", ctypes.c_uint32)]


class rknn_tensor_attr(ctypes.Structure):
    _fields_ = [
        ("index", ctypes.c_uint32), ("n_dims", ctypes.c_uint32),
        ("dims", ctypes.c_uint32 * 16), ("name", ctypes.c_char * 256),
        ("n_elems", ctypes.c_uint32), ("size", ctypes.c_uint32),
        ("fmt", ctypes.c_int32), ("type", ctypes.c_int32),
        ("qnt_type", ctypes.c_int32), ("fl", ctypes.c_int8),
        ("_pad0", ctypes.c_uint8 * 3), ("zp", ctypes.c_int32),
        ("scale", ctypes.c_float), ("w_stride", ctypes.c_uint32),
        ("size_with_stride", ctypes.c_uint32),
        ("pass_through", ctypes.c_uint8), ("_pad1", ctypes.c_uint8 * 3),
        ("h_stride", ctypes.c_uint32),
    ]


class rknn_input_st(ctypes.Structure):
    _fields_ = [
        ("index", ctypes.c_uint32), ("_pad0", ctypes.c_uint32),
        ("buf", ctypes.c_void_p), ("size", ctypes.c_uint32),
        ("pass_through", ctypes.c_uint8), ("_pad1", ctypes.c_uint8 * 3),
        ("type", ctypes.c_int32), ("fmt", ctypes.c_int32),
    ]


class rknn_output_st(ctypes.Structure):
    _fields_ = [
        ("want_float", ctypes.c_uint8), ("is_prealloc", ctypes.c_uint8),
        ("_pad0", ctypes.c_uint8 * 2), ("index", ctypes.c_uint32),
        ("buf", ctypes.c_void_p), ("size", ctypes.c_uint32),
        ("_pad1", ctypes.c_uint32),
    ]


# ---------------------------------------------------------------------------
# OpenRKNN shard runner (ctypes, OWN mode)
# ---------------------------------------------------------------------------

class OpenRKNNShard:
    """Run a single .rknn FP16 transformer shard via openrknn's C API."""

    def __init__(self, lib, model_path):
        self.lib = lib
        self.name = os.path.basename(model_path)
        self.ctx = ctypes.c_void_p()
        with open(model_path, "rb") as f:
            self._data = f.read()
        self._buf = ctypes.create_string_buffer(self._data)
        rc = lib.rknn_init(ctypes.byref(self.ctx), self._buf,
                           len(self._data), 0, None)
        if rc != 0:
            raise RuntimeError("rknn_init failed for %s: %d" % (self.name, rc))

        io = rknn_input_output_num()
        lib.rknn_query(self.ctx, RKNN_QUERY_IN_OUT_NUM,
                       ctypes.byref(io), ctypes.sizeof(io))
        self.n_output = io.n_output

        # Get input size
        attr = rknn_tensor_attr()
        attr.index = 0
        lib.rknn_query(self.ctx, RKNN_QUERY_INPUT_ATTR,
                       ctypes.byref(attr), ctypes.sizeof(attr))
        self.input_size = attr.size
        self.input_fmt = attr.fmt

    def run(self, x_fp32):
        """Run on [1, 1024, 768] FP32 input. Returns FP32 [1, 1024, 768]."""
        x_fp16 = x_fp32.astype(np.float16)
        inp_buf = ctypes.create_string_buffer(x_fp16.tobytes())
        inp = rknn_input_st()
        inp.index = 0
        inp.buf = ctypes.cast(inp_buf, ctypes.c_void_p)
        inp.size = x_fp16.nbytes
        inp.pass_through = 1
        inp.type = RKNN_TENSOR_FLOAT16
        inp.fmt = self.input_fmt

        self.lib.rknn_inputs_set(self.ctx, 1, (rknn_input_st * 1)(inp))
        self.lib.rknn_run(self.ctx, None)

        outs = (rknn_output_st * self.n_output)()
        self.lib.rknn_outputs_get(self.ctx, self.n_output, outs, None)
        out_data = ctypes.string_at(outs[0].buf, outs[0].size)
        self.lib.rknn_outputs_release(self.ctx, self.n_output, outs)

        # openrknn detiles FP16 3D output automatically (PR #94)
        result = np.frombuffer(out_data, dtype=np.float16).reshape(1, 1024, 768)
        return result.astype(np.float32)

    def destroy(self):
        self.lib.rknn_destroy(self.ctx)


# ---------------------------------------------------------------------------
# rknnlite2 shard runner (vendor, for comparison)
# ---------------------------------------------------------------------------

class VendorShard:
    """Run a single .rknn shard via vendor rknnlite2 Python API."""

    def __init__(self, model_path):
        from rknnlite.api import RKNNLite
        self.name = os.path.basename(model_path)
        self.rknn = RKNNLite(verbose=False)
        self.rknn.load_rknn(model_path)
        self.rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_AUTO)

    def run(self, x_fp32):
        if x_fp32.shape[1] < 1024:
            pad = np.zeros((1, 1024 - x_fp32.shape[1], 768), dtype=np.float32)
            x_fp32 = np.concatenate([x_fp32, pad], axis=1)
        elif x_fp32.shape[1] > 1024:
            x_fp32 = x_fp32[:, :1024, :]
        return self.rknn.inference(inputs=[x_fp32])[0]

    def destroy(self):
        self.rknn.release()


# ---------------------------------------------------------------------------
# Vision encoder (chains 24 shards with sandwich scaling)
# ---------------------------------------------------------------------------

class NPUVisionEncoder:
    """SigLIP vision encoder running 24 FP16 shards on the RK3588 NPU."""

    def __init__(self, shards, label="openrknn"):
        self.shards = shards
        self.label = label
        self.last_elapsed_ms = 0

    def __call__(self, inputs_embeds):
        x = inputs_embeds.copy()
        x *= 0.1  # sandwich pre-scale

        t0 = time.monotonic()
        for shard in self.shards:
            x = shard.run(x)
        elapsed = time.monotonic() - t0
        self.last_elapsed_ms = elapsed * 1000

        x *= 10.0  # sandwich post-scale
        return x


# ---------------------------------------------------------------------------
# HuggingFace model wrapper
# ---------------------------------------------------------------------------

def load_shards(shard_dir, lib=None, backend="openrknn"):
    """Load 24 .rknn shards. Returns list of shard runners."""
    shards = []
    for i in range(12):
        for kind in ["attn", "mlp"]:
            path = os.path.join(shard_dir, "l%d_%s.rknn" % (i, kind))
            if not os.path.exists(path):
                print("ERROR: missing shard: %s" % path)
                sys.exit(1)
            if backend == "openrknn":
                shards.append(OpenRKNNShard(lib, path))
            else:
                shards.append(VendorShard(path))
    return shards


def extract_frame(video_path, frame_num):
    """Extract a single frame from video using ffmpeg."""
    from PIL import Image
    import io
    cmd = [
        "ffmpeg", "-i", video_path,
        "-vf", "select=eq(n\\,%d)" % frame_num,
        "-vframes", "1", "-f", "image2pipe",
        "-pix_fmt", "rgb24", "-vcodec", "rawvideo", "-"
    ]
    # First get dimensions
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height", "-of", "csv=p=0",
         video_path],
        capture_output=True, text=True
    )
    w, h = [int(x) for x in probe.stdout.strip().split(",")]

    result = subprocess.run(cmd, capture_output=True, timeout=30)
    if result.returncode != 0:
        print("ERROR: ffmpeg failed: %s" % result.stderr.decode()[:200])
        sys.exit(1)

    raw = np.frombuffer(result.stdout, dtype=np.uint8).reshape(h, w, 3)
    return Image.fromarray(raw)


def run_demo(args):
    import torch
    from PIL import Image
    from transformers import AutoProcessor, AutoModelForVision2Seq
    from transformers.modeling_outputs import BaseModelOutput

    MODEL_ID = "HuggingFaceTB/SmolVLM-256M-Instruct"
    RESOLUTION = args.resolution

    # -- Banner --
    print("=" * 72)
    print("  SmolVLM on Orange Pi 5+ NPU -- powered by openrknn")
    print("=" * 72)
    print()

    # -- Load image --
    if args.video:
        print("Extracting frame %d from %s..." % (args.frame, args.video))
        image = extract_frame(args.video, args.frame)
        image_label = "%s (frame %d)" % (os.path.basename(args.video), args.frame)
    else:
        image = Image.open(args.image).convert("RGB")
        image_label = os.path.basename(args.image)
    print("Image: %s (%dx%d)" % (image_label, image.width, image.height))
    print("Prompt: \"%s\"" % args.prompt)
    print()

    # -- Load HuggingFace model --
    print("Loading SmolVLM model...", end=" ", flush=True)
    try:
        processor = AutoProcessor.from_pretrained(MODEL_ID)
        model = AutoModelForVision2Seq.from_pretrained(MODEL_ID)
    except OSError:
        print("\nERROR: Model not cached and no internet. Run once with internet:")
        print("  python3 -c \"from transformers import AutoProcessor, AutoModelForVision2Seq; "
              "AutoProcessor.from_pretrained('%s'); "
              "AutoModelForVision2Seq.from_pretrained('%s')\"" % (MODEL_ID, MODEL_ID))
        sys.exit(1)
    print("done")

    # -- Load NPU shards --
    print("Loading 24 NPU shards...", end=" ", flush=True)

    # openrknn OWN mode
    os.environ["ORKNN_OWN"] = "init,query,input,run,outputs"
    lib_path = args.lib or os.path.join(
        os.path.dirname(__file__), "..", "librknn_api.so")
    lib = ctypes.CDLL(lib_path)
    own_shards = load_shards(args.shard_dir, lib, "openrknn")
    own_encoder = NPUVisionEncoder(own_shards, "openrknn")
    print("done")

    # -- Prepare input --
    messages = [{"role": "user", "content": [
        {"type": "image"},
        {"type": "text", "text": args.prompt},
    ]}]
    prompt_text = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=prompt_text, images=[image], return_tensors="pt",
                       size={"height": RESOLUTION, "width": RESOLUTION})

    # -- Run vision encoder on NPU --
    class NPUEncoder(torch.nn.Module):
        def __init__(self, encoder_obj, original_config):
            super().__init__()
            self.config = original_config
            self.enc = encoder_obj

        def forward(self, inputs_embeds, attention_mask=None, **kwargs):
            x_np = inputs_embeds.detach().numpy().astype(np.float32)
            result = self.enc(x_np)
            return BaseModelOutput(last_hidden_state=torch.from_numpy(result))

    original_encoder = model.model.vision_model.encoder
    model.model.vision_model.encoder = NPUEncoder(own_encoder, original_encoder.config)

    print("Running vision encoder on NPU...", end=" ", flush=True)
    t_vision_start = time.monotonic()

    pixel_values = inputs["pixel_values"]
    if pixel_values.ndim == 5:
        pixel_values = pixel_values.squeeze(0)
    with torch.no_grad():
        vision_outputs = model.model.vision_model(
            pixel_values=pixel_values, output_hidden_states=True)
    t_vision = (time.monotonic() - t_vision_start) * 1000
    print("%.1fs" % (t_vision / 1000))

    # -- Run language model on CPU --
    print("Running language model on CPU...", end=" ", flush=True)
    t_lm_start = time.monotonic()
    with torch.no_grad():
        out_ids = model.generate(
            **inputs, max_new_tokens=args.max_tokens,
            do_sample=False, repetition_penalty=1.1)
    t_lm = (time.monotonic() - t_lm_start) * 1000
    print("%.1fs" % (t_lm / 1000))
    print()

    generated = processor.decode(out_ids[0], skip_special_tokens=True)
    # Extract just the assistant response (after the prompt)
    if "Assistant:" in generated:
        generated = generated.split("Assistant:")[-1].strip()

    # -- Print results --
    print("-" * 72)
    print(generated)
    print("-" * 72)
    print()
    print("  Vision encoder (NPU):  %6.0f ms  (24 shards, %.1f ms avg)" % (
        own_encoder.last_elapsed_ms,
        own_encoder.last_elapsed_ms / 24))
    print("  Language model (CPU):  %6.0f ms  (%d tokens)" % (
        t_lm, args.max_tokens))
    print("  Total:                 %6.0f ms" % (t_vision + t_lm))

    # -- Comparison mode --
    if args.compare:
        print()
        print("=" * 72)
        print("  Comparison: openrknn vs vendor rknnlite2")
        print("=" * 72)

        # Clean up OWN env for vendor run
        del os.environ["ORKNN_OWN"]

        print("Loading vendor rknnlite2 shards...", end=" ", flush=True)
        try:
            vendor_shards = load_shards(args.shard_dir, None, "vendor")
        except ImportError:
            print("\nrknnlite2 not installed -- skipping comparison")
            return
        vendor_encoder = NPUVisionEncoder(vendor_shards, "vendor")
        print("done")

        print("Running vendor vision encoder...", end=" ", flush=True)
        model.model.vision_model.encoder = NPUEncoder(vendor_encoder, original_encoder.config)
        t_vendor_start = time.monotonic()
        with torch.no_grad():
            model.model.vision_model(
                pixel_values=pixel_values, output_hidden_states=True)
        t_vendor = (time.monotonic() - t_vendor_start) * 1000
        print("%.1fs" % (t_vendor / 1000))

        for s in vendor_shards:
            s.destroy()

        delta = (own_encoder.last_elapsed_ms - vendor_encoder.last_elapsed_ms) / vendor_encoder.last_elapsed_ms * 100
        print()
        print("  %-26s %10s %10s %8s" % ("", "openrknn", "vendor", "delta"))
        print("  " + "-" * 58)
        print("  %-26s %8.0f ms %8.0f ms %+6.1f%%" % (
            "Vision encoder (NPU)",
            own_encoder.last_elapsed_ms,
            vendor_encoder.last_elapsed_ms,
            delta))
        print()
        if delta < 0:
            print("  openrknn is %.1f%% faster than vendor rknnlite2" % abs(delta))
        else:
            print("  vendor is %.1f%% faster" % delta)

    # Cleanup
    for s in own_shards:
        s.destroy()

    print()
    print("=" * 72)


def main():
    default_shard_dir = "/root/npu-research/smolvlm_shard/smolvlm_subshards_fused"
    default_image = os.path.join(os.path.dirname(__file__), "..", "tests",
                                  "test_images", "dog_224x224.jpg")

    parser = argparse.ArgumentParser(
        description="SmolVLM vision-language demo on Orange Pi 5+ NPU")
    parser.add_argument("--image", default=default_image,
                        help="Path to input image (default: test dog image)")
    parser.add_argument("--video", help="Extract frame from video instead")
    parser.add_argument("--frame", type=int, default=100,
                        help="Frame number to extract (default: 100)")
    parser.add_argument("--shard-dir", default=default_shard_dir,
                        help="Path to 24 .rknn shard files")
    parser.add_argument("--lib", help="Path to openrknn librknn_api.so")
    parser.add_argument("--prompt", default="Describe this image in detail.",
                        help="Question about the image")
    parser.add_argument("--max-tokens", type=int, default=150,
                        help="Max tokens to generate (default: 150)")
    parser.add_argument("--resolution", type=int, default=308,
                        help="Vision encoder input resolution (default: 308)")
    parser.add_argument("--compare", action="store_true",
                        help="Compare openrknn vs vendor rknnlite2 timing")
    args = parser.parse_args()

    if args.video and not os.path.exists(args.video):
        print("ERROR: video not found: %s" % args.video)
        sys.exit(1)
    if not args.video and not os.path.exists(args.image):
        print("ERROR: image not found: %s" % args.image)
        sys.exit(1)
    if not os.path.isdir(args.shard_dir):
        print("ERROR: shard directory not found: %s" % args.shard_dir)
        sys.exit(1)

    run_demo(args)


if __name__ == "__main__":
    main()
