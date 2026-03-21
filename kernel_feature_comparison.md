# RK3588 Kernel Feature Comparison: Mainline (6.18+) vs Vendor (6.1)

Last updated: 2026-03-21

Reference for the Orange Pi 5 Plus board. Covers hardware support status across
mainline Linux and Rockchip's vendor BSP kernel.

---

## Summary Table

| Subsystem | Mainline 6.18+ | Vendor 6.1 |
|---|---|---|
| CPU / SMP / DVFS | Full (since 6.11) | Full |
| GPU (OpenGL ES 3.1) | Full via Panthor (since 6.10) | Mali blob (higher raw perf, X11 only) |
| GPU Vulkan 1.4 | PanVK conformant (Mesa 25+) | Proprietary only |
| GPU OpenCL | Experimental (Rusticl) | Proprietary OpenCL |
| H.264 decode | Upstream (7.0) | Full via MPP |
| H.265/HEVC decode | Upstream (7.0) | Full via MPP |
| AV1 decode | Upstream (hantro) | Full via MPP |
| VP9 decode | Not yet | Full via MPP |
| Video encode (H.264/H.265) | **NOT upstream** | Full via MPP |
| JPEG encode | Linux 6.12 | Full |
| ISP / Camera (rkisp3) | **NOT upstream** | Full (RKCIF + RKISP3) |
| HDMI output (single) | Linux 6.13 (4K@30Hz max) | 4K@120Hz |
| HDMI output (dual) | Linux 6.15 | Full dual |
| HDMI audio | Linux 6.15 | Full |
| HDMI input (RX) | Linux 6.15 | Full |
| HDMI CEC | Not yet | Supported |
| MIPI DSI | Linux 6.15 (driver merged, DT pending) | Full |
| DisplayPort (USB-C) | Patch in review, not merged | Full |
| VOP2 | Full | Full |
| PCIe 3.0 / NVMe | Linux 6.7+ (ASPM quirks) | Full |
| USB3 (all 3 controllers) | Linux 6.10 | Full |
| USB-C DP Alt Mode | Not yet (USBDP PHY done) | Full |
| USB-PD | U-Boot 2025.01 | Full |
| GMAC 1GbE | Linux 6.3 | Full |
| RTL8125 2.5GbE (PCIe) | Linux 6.7+ | Full |
| WiFi (M.2 AP6275P) | Partial (Armbian patches) | Full |
| Bluetooth | Unreliable on mainline | Full |
| RGA2 (2D accel) | Linux 6.12 | Full via librga |
| RGA3 (2D accel) | Patch in review | Full via librga |
| I2S / TDM audio | Upstream | Full |
| NPU | Rocket driver (upstream) | Proprietary RKNPU |
| RNG | Linux 6.15 | Full |
| eMMC / SD | Full | Full |
| SATA (via PCIe) | Full | Full |

---

## 1. GPU — Mali-G610 MP4

### Mainline: Panthor + PanVK

The **Panthor** DRM driver for Mali Valhall CSF-based GPUs landed in Linux 6.10.

- **OpenGL ES 3.1** conformant (Mesa 24.1+)
- Desktop OpenGL 3.1 supported; 3.3 blocked by lack of geometry shader HW
- Transaction elimination and incremental rendering enabled by default in Mesa 25.0
- **Vulkan 1.1** conformant from Mesa 25.0.2
- **Vulkan 1.2** conformant ~6 weeks after 1.1
- **Vulkan 1.4** conformance achieved at FOSDEM 2026
- Rusticl OpenCL: experimental, behind env-var gate

### Vendor: Mali proprietary blob

- Higher raw GPU utilization in some workloads
- X11-only, closed-source, unmaintained for upstream kernels
- No Vulkan unless using deprecated Panfork workaround
- Dead-end path — Panthor/PanVK is clearly superior for features, Wayland, and long-term viability

---

## 2. VPU — Video Decode / Encode

### Decode — Mainline

| Codec | Driver | Merged | Notes |
|---|---|---|---|
| H.264 (AVC) | RKVDEC2 (VDPU381) | Linux 7.0 | 17-patch series by Collabora |
| H.265 (HEVC) | RKVDEC2 (VDPU381) | Linux 7.0 | New V4L2 HEVC UAPI controls |
| AV1 | Hantro (Verisilicon) | Earlier | Separate HW block |
| MPEG2, VP8 | Hantro G1 (VDPU121) | Earlier | Up to 1920x1088 |
| VP9 | RKVDEC2 | Planned | Not yet |
| AVS2 | RKVDEC2 | — | HW supports it, no driver |

RK3588 has two VDPU381 cores for up to 8K@30; currently only single-core is
used. Multi-core scheduling is being worked on. GStreamer 1.28 has merged
support; FFmpeg has preliminary patches.

### Parallel Video Processing

All VPU blocks (VDPU121, 2x VDPU381, VPU981, 5x VEPU121, VEPU580/721) are
**physically independent hardware** with separate MMIO register sets, clock
domains, IRQ lines, and DMA engines. They can all operate simultaneously at
full declared throughput — e.g. decoding H.265 + AV1 + encoding H.264 + JPEG
all at once.

The practical bottleneck is **DRAM bandwidth** (~25.6 GB/s LPDDR4X or ~51.2 GB/s
LPDDR5), shared across CPU, GPU, NPU, all VPU blocks, and display. A single
8K@30 HEVC decode already consumes significant bandwidth, so running everything
at max simultaneously would saturate the bus.

On vendor kernel, Rockchip MPP (Media Process Platform) is the userspace
scheduler that routes work across all codec blocks. On mainline, each block is
exposed as a separate `/dev/video*` V4L2 device, and userspace manages
parallelism directly.

### VOD Transcoding Use Case

VEPU580 (H.265) and VEPU721 (H.264) are independent hardware blocks with
separate MMIO, clocks, and IRQs. A VOD transcoding pipeline can run both
encoders simultaneously:

```
Source file
    |
    v
VDPU381 (HW decode) --> shared DRAM buffer
    |                         |
    +-- RGA3 (HW scale) --> VEPU580 (H.265) --> HLS/DASH .265 segments
    |
    +-- RGA2 (HW scale) --> VEPU721 (H.264) --> HLS/DASH .264 segments
```

Approximate per-encoder throughput (from Rockchip specs):

| Resolution | Encode speed | Time per rendition (2hr movie) |
|---|---|---|
| 4K@30 | ~1x realtime | ~2 hours |
| 1080p@30 | ~4x realtime | ~30 min |
| 720p@30 | ~6-8x realtime | ~15-20 min |
| 480p@30 | ~10x+ realtime | ~12 min |

Each encoder processes one stream at a time, so the ABR ladder is sequential
within each codec but **parallel across codecs**. A 4-rendition ladder for a
2-hour movie takes ~3 hours per codec — but both run simultaneously, so
wall-clock time is ~3 hours for complete H.264 + H.265 ladders (vs ~6 hours
sequential). Decode and scaling are essentially free (VDPU381 and RGA are much
faster than encode).

Constraints:
- **Memory bandwidth** — simultaneous decode + 2x encode + 2x scale is heavy
  at 4K; fine at 1080p and below
- **Encode quality** — Hantro H2 hardware encoders produce lower quality-per-bit
  than x265/x264 software. Acceptable for cost-effective mass transcoding
  (IPTV, budget streaming), not for premium VOD (Netflix-quality)
- **B-frame support** — Hantro H2 likely has limited B-frame support vs software
  encoders, reducing compression efficiency
- **Software stack** — requires vendor kernel + MPP; no mainline encoder driver.
  FFmpeg + MPP integration exists but isn't upstream

At ~10W TDP, RK3588 could serve as a low-power transcoding node doing both
H.264 + H.265 ABR ladders in roughly realtime for 4K content, or 3-4x realtime
for 1080p. A rack of these could compete with a single Xeon for bulk
transcoding at a fraction of the power budget, trading encode quality for
power efficiency.

### Encode — NOT upstream

- **JPEG encode** (VEPU121): merged in Linux 6.12
- **H.264 / H.265 video encode**: NOT upstream. Requires vendor kernel + MPP library
- Vendor kernel: Rockchip MPP handles H.264/H.265 encode up to 8K@30 via proprietary ioctl API (not V4L2)

---

## 3. ISP — Camera / Image Signal Processor

**NOT upstream for RK3588.**

- `rkisp1` in mainline covers RK3288/RK3399 era — different HW
- RK3588 has **ISP3** (rkisp3), substantially different, no upstream driver
- Camera pipeline: RKCIF (MIPI CSI-2) feeds RKISP3 or directly to user
- VICAP: 1 DVP + 6 MIPI CSI-2 receivers. RK3568 VICAP driver was in review (v4, Feb 2025); RK3588 variant not yet addressed
- libcamera integration described as needing full IQ algorithm rewrite

Vendor kernel 6.1: Full RKCIF + RKISP3 + 3A server. Supports multi-camera.
Not expected upstream before late 2026 at earliest.

---

## 4. Display

### HDMI

- Linux 6.9: HDMI PHY initial support
- **Linux 6.13**: HDMI0 first usable
- **Linux 6.15**: HDMI1 added; HDMI audio merged; HDMI RX (capture) merged
- Max resolution: **4K@30Hz** — 4K@60Hz requires HDMI 2.0 high TMDS clock ratio + scrambling, not yet implemented
- HDMI CEC: planned, status unclear

### DisplayPort over USB-C

- Two DW DPTX controllers (DP 1.4a, 4 lanes, up to 8.1 Gbps)
- v5 patch series posted July 2025, not merged
- Tested at 1080p and 4K@60 YCbCr 4:2:0
- USBDP PHY upstreamed in Linux 6.10 for USB3; DP Alt Mode DRM bridge not merged

### MIPI DSI

- Driver merged in Linux 6.15; device-tree changes still pending

### VOP2

- 4 video ports: VP0/1/2 up to 4096x2160, VP3 up to 2048x1080
- Basic VOP2 support merged by Andy Yan (Rockchip)

### Vendor kernel

- Full dual HDMI 2.1 (4K@120Hz), MIPI DSI, DP via USB-C, HDMI input — all working
- All 4 VOP2 video ports configured

---

## 5. PCIe

- Functional since **Linux 6.7**
- PCIe 3.0 x4 (NVMe), PCIe 3.0 x2, and PCIe 2.0 lanes all work
- Known issue: ASPM compatibility problems with some NVMe drives on 6.10+ — `pcie_aspm=off` workaround may be needed
- NVMe boot via SPI U-Boot supported

---

## 6. USB

| Controller | PHY | Mainline Since |
|---|---|---|
| USB3 #1 (DWC3) | Separate PHY | Linux 6.8 |
| USB3 #2 (DWC3) | USBDP PHY0 | Linux 6.10 |
| USB3 #3 (DWC3) | USBDP PHY1 | Linux 6.10 |
| USB2 (multiple) | USB2 PHY | Earlier |
| USB-C DP Alt Mode | USBDP PHY | PHY 6.10, DRM bridge NOT YET |
| USB-PD | — | U-Boot 2025.01 |

All USB3 ports functional in Linux 6.10+.

---

## 7. Ethernet

- **Native GMAC 1GbE**: upstream since Linux 6.3 via `dwmac-rk` (stmmac)
- **RTL8125 2.5GbE** (PCIe-attached): functional since Linux 6.7+ once PCIe landed
- Orange Pi 5 Plus: dual RTL8125 2.5GbE — both ports work in mainline

---

## 8. RGA — 2D Accelerator

RK3588 has 1x RGA2 + 2x RGA3 cores.

- **RGA2**: merged in Linux 6.12 via V4L2. Scaling, rotation, alpha blending, color fill, YUV422/420
- **RGA3**: patch series under review, post-6.12. Scaling, format conversion, uses rockchip-iommu-v2
- GStreamer `v4l2convert` can leverage RGA2 for HW pixel format conversion
- Vendor kernel: full RGA2 + RGA3 via proprietary librga interface

---

## 9. NPU — Neural Processing Unit

- **Mainline**: Rocket DRM/accel driver by Tomeu Vizoso (Collabora), upstream 2024
  - Mesa Teflon (libteflon.so) provides TFLite inference API
  - 3 cores, register-programmed (no ISA), INT8 convolution
  - MobileNetV1 ~10ms, per-axis quantization via per-group decomposition
- **Vendor**: Proprietary RKNPU driver + RKNN SDK
  - Per-output-channel task decomposition (all 227 YOLO tasks have total_ch=1)
  - Full INT8/INT16/FP16 support, mature quantization

---

## 10. Audio

- **I2S/TDM**: upstream, RK3588 compatible string in rockchip I2S driver
- **HDMI audio**: merged in Linux 6.15 (Collabora, Detlev Casanova)
- **HDMI CEC**: planned, status uncertain
- **Analog/headphone**: depends on board-specific codec (ES8316 etc.), generally functional
- Vendor kernel: full HDMI audio, I2S, SPDIF, PDM microphone

---

## 11. WiFi / Bluetooth (Orange Pi 5 Plus)

No onboard WiFi/BT — M.2 E-Key slot for optional modules (AP6275P / BCM4375).

- **Mainline**: brcmfmac has partial support; requires Armbian community patches. BT unreliable
- **Vendor kernel**: full WiFi 6 + BT 5.0 support with BSP drivers

---

## Prioritized Upstreaming Gaps

Sorted by impact for typical SBC/developer use cases:

1. **Video encode (H.264/H.265)** — No upstream path. Blocks all recording/streaming use cases. Requires RKVENC V4L2 driver or equivalent
2. **HDMI 4K@60Hz** — High TMDS clock ratio + scrambling needed. Most users with 4K monitors are affected
3. **DisplayPort over USB-C** — Patch series exists (v5), needs merge. Blocks USB-C display output
4. **ISP / Camera (rkisp3)** — Full rewrite needed. Blocks all camera use cases
5. **VP9 decode** — Common web video codec. RKVDEC2 driver exists, just needs VP9 support added
6. **RGA3** — Patch in review. RGA2 alone is functional but slower
7. **WiFi/BT** — Board-specific, only affects boards with M.2 WiFi modules
8. **HDMI CEC** — Nice-to-have for media center use cases
9. **USB-C DP Alt Mode** — USBDP PHY done, DRM bridge side pending
10. **Video encode VP9/AV1** — Future consideration, HW may not support these

### What works well on mainline today

For a developer/server/NAS/NPU research workload, mainline 6.18+ is fully
usable: HDMI display (4K@30), HDMI audio, GPU with Vulkan 1.4, all USB ports,
PCIe + NVMe, dual 2.5GbE, NPU, eMMC/SD. The vendor kernel is only needed for
camera, video encode, 4K@60 HDMI, or DP output.

---

## IP Core Provenance — Third-Party Licensed Blocks

The RK3588 is heavily built from licensed third-party IP. Understanding which
vendor designed each block clarifies where upstream drivers already exist (shared
with other SoCs) and where Rockchip-proprietary reverse engineering is needed.

### IP Provenance Summary

| IP Block | Vendor | Product Name | Shared With |
|---|---|---|---|
| CPU | ARM | Cortex-A76 (x4) + Cortex-A55 (x4) | Universal |
| GPU | ARM | Mali-G610 MP4 (Valhall 2nd gen) | Various ARM licensees |
| GIC | ARM | GICv3 | Universal ARM SoCs |
| NPU | **Rockchip** (in-house) | 3rd-gen RKNPU | Only Rockchip SoCs |
| Video Dec (VDPU121) | VeriSilicon/Hantro | Hantro G1 | RK3399/3568, NXP i.MX |
| Video Dec (VDPU381) | VeriSilicon/Hantro | G2-successor | RK3576 (VDPU383), RK356x (VDPU346) |
| Video Dec (VPU981) | VeriSilicon/Hantro | Hantro AV1 decoder | Future non-RK SoCs expected |
| Video Enc (VEPU121) | VeriSilicon/Hantro | Hantro H1 family (JPEG) | RK3568 (identical) |
| Video Enc (VEPU580/721) | VeriSilicon/Hantro | Hantro H2 (H.264/H.265) | Various Rockchip SoCs |
| ISP 3.0 | Rockchip (evolved from shared ancestor with VeriSilicon ISP8000) | rkisp3 | ISP8000Nano in NXP i.MX8MP |
| RGA2/RGA3 | **Rockchip** (in-house) | RGA2-Enhance, RGA3 | Only Rockchip SoCs |
| DDR Controller | Synopsys DWC | uMCTL2/uPCTL2 | Universal |
| DDR PHY | Synopsys DWC | LPDDR5/4X PHY + PUB | Universal |
| USB 3.x | Synopsys DWC | DWC3 | Universal |
| USB 2.0 | Synopsys DWC | DWC2 | Universal |
| PCIe | Synopsys DWC | DWC PCIe Gen3 | NXP, Amlogic, Allwinner |
| Ethernet MAC | Synopsys DWC | DWMAC4/5 (stmmac) | ST, NXP, Amlogic, Allwinner |
| HDMI 2.1 TX | Synopsys DWC | DW HDMI QP TX | Newer ARM SoCs |
| DisplayPort TX | Synopsys DWC | DW DPTX (DP 1.4a) | Various |
| MIPI DSI2 | Synopsys DWC | DW MIPI DSI2 | RK3576, others |
| SDMMC/SDIO | Synopsys DWC | DW-MSHC | All Rockchip SoCs |
| eMMC | Synopsys DWC | DWCMSHC (SDHCI) | Qualcomm, NXP |
| UART | Synopsys DWC | DW APB UART (8250) | Intel, Qualcomm, NXP |
| HDMI/eDP PHY | Samsung (licensed) | HDMI/eDP TX Combo PHY | Samsung Exynos |
| USBDP Combo PHY | Samsung (licensed) | USB3+DP Alt Mode PHY | Samsung Exynos |
| I2C | **Rockchip** (in-house) | rk3x I2C | Only Rockchip SoCs |
| SPI | **Rockchip** (in-house) | Rockchip SPI | Only Rockchip SoCs |
| I2S/TDM | **Rockchip** (in-house) | Rockchip I2S-TDM | Only Rockchip SoCs |
| VOP2 | **Rockchip** (in-house) | Video Output Processor 2 | Only Rockchip SoCs |

### Key Observations

1. **Synopsys DesignWare dominates** the interface IP: DDR, USB, PCIe, Ethernet,
   HDMI, DP, MIPI DSI, SDMMC, eMMC, UART. This is why so many mainline Linux
   drivers "just work" — the DWC framework drivers are shared across dozens of SoCs.

2. **VeriSilicon/Hantro supplies all video codec IP.** The kernel driver directory
   is literally `drivers/media/platform/verisilicon/`. The VPU981 AV1 decoder is
   explicitly described in kernel patches as "not a Rockchip hardware design" that
   "will likely start appearing on non-RK SoCs."

3. **Samsung provides two critical PHY blocks** (USBDP combo PHY and HDMI/eDP PHY)
   — an often-overlooked third-party dependency.

4. **Rockchip in-house IP:** NPU (explicitly confirmed), RGA, VOP2, I2C, SPI,
   I2S/TDM. These are the blocks requiring Rockchip-specific reverse engineering
   or vendor cooperation for upstream drivers.

5. **ISP lineage is ambiguous.** The RK3399 ISP1 definitively shares ancestry with
   VeriSilicon ISP8000Nano (used in NXP i.MX8MP — same `rkisp1` driver works for
   both). Whether RK3588's ISP3 continues from this lineage or has diverged is not
   publicly confirmed, but the shared ancestor is established.

6. **DDR training firmware** (Synopsys PUB blob) is the **only closed-source binary**
   remaining in the RK3588 bootchain.

---

## Key Contributors

Collabora is the primary driver of mainline RK3588 support: Sebastian Reichel,
Detlev Casanova, Cristian Ciocaltea, Heiko Stübner, Tomeu Vizoso. Rockchip
itself has largely stepped back from upstream contributions.

## Sources

- [Collabora: PanVK Vulkan 1.2 conformance](https://www.collabora.com/news-and-blog/news-and-events/panvk-reaches-vulkan-12-conformance-on-mali-g610.html)
- [Collabora: Mesa 25 PanVK production quality](https://www.collabora.com/news-and-blog/news-and-events/mesa-25-panvk-moves-towards-production-quality.html)
- [CNX Software: RK3588 H.264/H.265 mainline](https://www.cnx-software.com/2026/02/27/rockchip-rk3588-rk3576-h-264-and-h-265-video-decoders-mainline-linux/)
- [Collabora: RK3588 upstream progress](https://www.collabora.com/news-and-blog/news-and-events/rockchip-rk3588-upstream-support-progress-future-plans.html)
- [Collabora: Mainline Rockchip year in review (March 2026)](https://www.collabora.com/news-and-blog/blog/2026/03/02/running-mainline-linux-u-boot-and-mesa-on-rockchip-a-year-in-review/)
- [CNX Software: RK3588 mainline status 2025](https://www.cnx-software.com/2024/12/21/rockchip-rk3588-mainline-linux-support-current-status-and-future-work-for-2025/)
- [LWN: RK3588 DisplayPort Controller](https://lwn.net/Articles/1031560/)
- [LWN: RGA3 support](https://lwn.net/Articles/1041152/)
- [ChipEstimate: VeriSilicon Hantro G2 Video Decoder IP](https://www.chipestimate.com/VeriSilicon-Introduces-Hantro-G2-Video-Decoder-IP-with-HEVC-and-VP9-Support/Semiconductor-IP-Core/news/21441)
- [Business Wire: VeriSilicon Hantro H2 HEVC Encoder IP](https://www.businesswire.com/news/home/20140714005068/en/VeriSilicon-Introduces-Hantro-H2-HEVC-Encoder-Semiconductor-IP)
- [LWN: AV1 stateless decoder for RK3588 (VPU981)](https://lwn.net/Articles/930790/)
- [Patchwork: verisilicon AV1 decoder on RK3588](https://patchwork.kernel.org/project/linux-media/patch/20230412115652.403949-13-benjamin.gaignard@collabora.com/)
- [Synopsys: DWC DDR Universal uMCTL2](https://www.synopsys.com/dw/ipdir.php?ds=dwc_ddr_universal_umctl2)
- [LWN: RK3588 USBDP PHY support](https://lwn.net/Articles/961692/)
- [kernel.org: rkisp1 driver documentation](https://docs.kernel.org/admin-guide/media/rkisp1.html)
