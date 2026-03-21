/*
 * Minimal Rocket NPU job submission test for QEMU.
 * Tests: CREATE_BO, mmap, SUBMIT, PREP_BO (wait for completion).
 *
 * Runs a trivial 1x1 convolution: 1 input channel, 1 output channel,
 * 1x1 filter, 1x1 spatial. Input=42, weight=1, bias=0 → output≈42.
 */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <errno.h>
#include <time.h>

/* DRM base */
/* DRM uses type 'd' (0x64), commands start at 0x40 */
#define DRM_IOCTL_BASE 'd'
#define DRM_COMMAND_BASE 0x40

/* Rocket ioctls */
#define DRM_ROCKET_CREATE_BO 0x00
#define DRM_ROCKET_SUBMIT    0x01
#define DRM_ROCKET_PREP_BO   0x02
#define DRM_ROCKET_FINI_BO   0x03

struct drm_rocket_create_bo {
    uint32_t size;
    uint32_t handle;
    uint64_t dma_address;
    uint64_t offset;
};

struct drm_rocket_prep_bo {
    uint32_t handle;
    uint32_t reserved;
    int64_t timeout_ns;
};

struct drm_rocket_fini_bo {
    uint32_t handle;
    uint32_t reserved;
};

struct drm_rocket_task {
    uint32_t regcmd;
    uint32_t regcmd_count;
};

struct drm_rocket_job {
    uint64_t tasks;
    uint64_t in_bo_handles;
    uint64_t out_bo_handles;
    uint32_t task_count;
    uint32_t task_struct_size;
    uint32_t in_bo_handle_count;
    uint32_t out_bo_handle_count;
};

struct drm_rocket_submit {
    uint64_t jobs;
    uint32_t job_count;
    uint32_t job_struct_size;
    uint64_t reserved;
};

#define DRM_IOCTL_ROCKET_CREATE_BO _IOWR(DRM_IOCTL_BASE, DRM_COMMAND_BASE + DRM_ROCKET_CREATE_BO, struct drm_rocket_create_bo)
#define DRM_IOCTL_ROCKET_SUBMIT    _IOW(DRM_IOCTL_BASE, DRM_COMMAND_BASE + DRM_ROCKET_SUBMIT, struct drm_rocket_submit)
#define DRM_IOCTL_ROCKET_PREP_BO   _IOW(DRM_IOCTL_BASE, DRM_COMMAND_BASE + DRM_ROCKET_PREP_BO, struct drm_rocket_prep_bo)
#define DRM_IOCTL_ROCKET_FINI_BO   _IOW(DRM_IOCTL_BASE, DRM_COMMAND_BASE + DRM_ROCKET_FINI_BO, struct drm_rocket_fini_bo)

/* NPU register targets (bits [63:48] of regcmd entry) */
#define TARGET_CNA   0x0201
#define TARGET_CORE  0x0801
#define TARGET_DPU   0x1001
#define TARGET_RDMA  0x2001
#define TARGET_PC    0x0101

static uint64_t emit(uint16_t target, uint16_t reg, uint32_t value)
{
    return ((uint64_t)target << 48) | ((uint64_t)value << 16) | (uint64_t)reg;
}

/* Build a minimal regcmd for 1x1x16 → 1x1x16 conv, stride=1, no padding */
static unsigned build_regcmd(uint64_t *buf, uint32_t in_addr, uint32_t wt_addr,
                             uint32_t out_addr, uint32_t bias_addr)
{
    uint64_t *p = buf;

    /* CNA registers */
    *p++ = emit(TARGET_CNA, 0x1040, 0x00000b01);  /* CBUF_CON0: weight_bank=11, data_bank=1 */
    *p++ = emit(TARGET_CNA, 0x1100, 0);           /* DCOMP_REGNUM */
    *p++ = emit(TARGET_CNA, 0x1104, 0);           /* DCOMP_CTRL */
    *p++ = emit(TARGET_CNA, 0x100c, 0);           /* CONV_CON1: normal conv */

    /* DPU/RDMA S_POINTER */
    *p++ = emit(TARGET_DPU, 0x4004, 0x00070007);
    *p++ = emit(TARGET_RDMA, 0x5004, 0x00070007);

    *p++ = emit(TARGET_CNA, 0x100c, 0);           /* CONV_CON1 again */
    *p++ = emit(TARGET_CNA, 0x1010, 52);          /* CONV_CON2: feature_grains */
    *p++ = emit(TARGET_CNA, 0x1014, 0);           /* CONV_CON3: stride=1,1 */

    /* Input: 1x1, 16 channels. CNA stores raw values. */
    *p++ = emit(TARGET_CNA, 0x1020, 0x00010001);  /* DATA_SIZE0: h=1, w=1 (raw) */
    /* DATA_SIZE1: CHANNEL[15:0]=16(raw), CHANNEL_REAL[29:16]=15(count-1) */
    *p++ = emit(TARGET_CNA, 0x1024, (15 << 16) | 16);
    *p++ = emit(TARGET_CNA, 0x1028, 0x00000001);  /* DATA_SIZE2: out_w=1 */
    *p++ = emit(TARGET_CNA, 0x102c, 0x00000001);  /* DATA_SIZE3: atomics=1 */

    /* Weights: 1x1 filter, 32 kernels. CNA stores raw values. */
    *p++ = emit(TARGET_CNA, 0x1030, 16 * 32);     /* WEIGHT_SIZE0 */
    *p++ = emit(TARGET_CNA, 0x1034, 16);           /* WEIGHT_SIZE1 */
    /* WEIGHT_SIZE2: KERNELS[13:0]=32, HEIGHT[20:16]=1, WIDTH[28:24]=1 */
    *p++ = emit(TARGET_CNA, 0x1038, (1 << 24) | (1 << 16) | 32);

    *p++ = emit(TARGET_CNA, 0x1040, 0x00000b01);  /* CBUF_CON0 again */
    *p++ = emit(TARGET_CNA, 0x1044, 0x00ff0000);  /* CBUF_CON1: data_entries */

    /* CVT: signed bypass */
    *p++ = emit(TARGET_CNA, 0x1048, 0x00000030);  /* CVT_CON0: DATA_SIGN=1, CVT_TYPE=1, CVT_BYPASS=1 */
    *p++ = emit(TARGET_CNA, 0x104c, 0x00010000);  /* CVT_CON1: scale0=1 */
    *p++ = emit(TARGET_CNA, 0x1050, 0x00010000);  /* CVT_CON2: scale1=1 */
    *p++ = emit(TARGET_CNA, 0x1054, 0x00010000);  /* CVT_CON3: scale2=1 */
    *p++ = emit(TARGET_CNA, 0x1058, 0x00010000);  /* CVT_CON4: scale3=1 */

    *p++ = emit(TARGET_CNA, 0x105c, 0);           /* FC_CON0 */
    *p++ = emit(TARGET_CNA, 0x1060, 0);           /* FC_CON1 */
    *p++ = emit(TARGET_CNA, 0x1068, 0);           /* PAD_CON0: no padding */

    /* Input DMA address */
    *p++ = emit(TARGET_CNA, 0x1070, in_addr);
    *p++ = emit(TARGET_CNA, 0x1074, 0);           /* FC_CON2 */
    *p++ = emit(TARGET_CNA, 0x1078, 0x0f0f0000);  /* DMA_CON0: burst_len */
    *p++ = emit(TARGET_CNA, 0x107c, 16);          /* DMA_CON1: line_stride */
    *p++ = emit(TARGET_CNA, 0x1080, 16);          /* DMA_CON2: surf_stride */

    *p++ = emit(TARGET_CNA, 0x1084, 0x00000000);  /* FC_DATA_SIZE0: w=1(0), h=1(0) */
    *p++ = emit(TARGET_CNA, 0x1088, 0x000f0000);  /* FC_DATA_SIZE1: ch=16(0xf) */
    *p++ = emit(TARGET_CNA, 0x1104, 0);           /* DCOMP_CTRL */
    *p++ = emit(TARGET_CNA, 0x1100, 0);           /* DCOMP_REGNUM */

    /* Weight DMA address */
    *p++ = emit(TARGET_CNA, 0x1110, wt_addr);

    /* DCOMP_AMOUNT0..15 = 0 */
    for (int i = 0; i < 16; i++)
        *p++ = emit(TARGET_CNA, 0x1114 + i * 4, 0);

    *p++ = emit(TARGET_CNA, 0x10a0, 0);           /* CVT_CON5 */
    *p++ = emit(TARGET_CNA, 0x1184, 0xffffff80);  /* PAD_CON1: pad_value */

    /* CORE registers */
    *p++ = emit(TARGET_CORE, 0x3010, 0x00000001);  /* MISC_CFG: QD_EN=1 */
    *p++ = emit(TARGET_CORE, 0x3014, 0x00000000);  /* DATAOUT_SIZE_0: w=0+1=1, h=0+1=1 */
    *p++ = emit(TARGET_CORE, 0x3018, 0x0000000f);  /* DATAOUT_SIZE_1: ch=15+1=16 */
    *p++ = emit(TARGET_CORE, 0x301c, 1);           /* CLIP_TRUNCATE: 1 bit */
    *p++ = emit(0x0801, 0x3030, 0);                /* raw CORE reg */

    /* DPU registers */
    *p++ = emit(TARGET_DPU, 0x400c, 0x0000020f);  /* FEAT_MODE_CFG: burst=15, out_mode=2 */
    *p++ = emit(TARGET_DPU, 0x4010, 0);           /* DATA_FORMAT */
    *p++ = emit(TARGET_DPU, 0x4014, 0);           /* OFFSET_PEND */

    /* Output DMA address */
    *p++ = emit(TARGET_DPU, 0x4020, out_addr);
    *p++ = emit(TARGET_DPU, 0x4024, 0x00000010);  /* DST_SURF_STRIDE: 1*1=1 << 4 */
    *p++ = emit(TARGET_DPU, 0x4030, 0);           /* DATA_CUBE_WIDTH: 0+1=1 */
    *p++ = emit(TARGET_DPU, 0x4034, 0);           /* DATA_CUBE_HEIGHT: 0+1=1 */
    *p++ = emit(TARGET_DPU, 0x4038, 0);           /* DATA_CUBE_NOTCH */
    *p++ = emit(TARGET_DPU, 0x403c, 0x000f000f);  /* DATA_CUBE_CHANNEL: ch=15+1=16, orig=15+1=16 */

    /* BS stage: ALU_ALGO=2, ALU_SRC=1, RELU_BYPASS=1, MUL_BYPASS=1 */
    *p++ = emit(TARGET_DPU, 0x4040, 0x00020150);  /* BS_CFG */
    *p++ = emit(TARGET_DPU, 0x4044, 0);           /* BS_ALU_CFG */
    *p++ = emit(TARGET_DPU, 0x4048, 0);           /* BS_MUL_CFG */
    *p++ = emit(TARGET_DPU, 0x404c, 0);           /* BS_RELUX_CMP */
    *p++ = emit(TARGET_DPU, 0x4050, 0x00010101);  /* BS_OW_CFG */
    *p++ = emit(TARGET_DPU, 0x4054, 0);           /* BS_OW_OP: wt_zp=0 */

    *p++ = emit(TARGET_DPU, 0x4058, 0x0000000f);  /* WDMA_SIZE_0: ch=15+1=16 */
    *p++ = emit(TARGET_DPU, 0x405c, 0);           /* WDMA_SIZE_1: w=0+1=1, h=0+1=1 */

    /* BN: all bypass */
    *p++ = emit(TARGET_DPU, 0x4060, 0x0000001f);
    *p++ = emit(TARGET_DPU, 0x4064, 0);
    *p++ = emit(TARGET_DPU, 0x4068, 0);
    *p++ = emit(TARGET_DPU, 0x406c, 0);

    /* EW: all bypass */
    *p++ = emit(TARGET_DPU, 0x4070, 0x00000383);
    *p++ = emit(TARGET_DPU, 0x4074, 0);
    *p++ = emit(TARGET_DPU, 0x4078, 0x00010000);  /* EW_CVT_SCALE: scale=1 */
    *p++ = emit(TARGET_DPU, 0x407c, 0);

    /* OUT_CVT: scale=16384(0x4000), shift=14, offset=0 → identity */
    *p++ = emit(TARGET_DPU, 0x4080, 0);           /* OUT_CVT_OFFSET: 0 */
    *p++ = emit(TARGET_DPU, 0x4084, 0x00004000);  /* OUT_CVT_SCALE: 16384 */
    *p++ = emit(TARGET_DPU, 0x4088, 14);          /* OUT_CVT_SHIFT: 14 */

    /* EW_OP values */
    for (int i = 0; i < 8; i++)
        *p++ = emit(TARGET_DPU, 0x4090 + i * 4, 0);

    *p++ = emit(TARGET_DPU, 0x40c0, 1);           /* SURFACE_ADD */
    *p++ = emit(0x1001, 0x40c4, 0);               /* raw DPU reg */

    /* LUT registers (all zero) */
    *p++ = emit(TARGET_DPU, 0x40b4, 0);
    *p++ = emit(TARGET_DPU, 0x40b8, 0);
    *p++ = emit(TARGET_DPU, 0x40bc, 0);
    *p++ = emit(TARGET_DPU, 0x40c0, 0);
    for (int i = 0; i < 8; i++)
        *p++ = emit(TARGET_DPU, 0x40c8 + i * 4, 0);

    /* RDMA registers */
    *p++ = emit(TARGET_RDMA, 0x5008, 0);          /* RDMA_DATA_CUBE_WIDTH */
    *p++ = emit(TARGET_RDMA, 0x500c, 0);          /* RDMA_DATA_CUBE_HEIGHT */
    *p++ = emit(TARGET_RDMA, 0x5010, 0x0000000f); /* RDMA_DATA_CUBE_CHANNEL: 15+1=16 */
    *p++ = emit(TARGET_RDMA, 0x5014, 0);          /* RDMA_SRC_BASE_ADDR */

    /* BRDMA: DATA_USE=1 (bias from DMA) */
    *p++ = emit(TARGET_RDMA, 0x501c, 0x00000002);
    *p++ = emit(TARGET_RDMA, 0x5020, bias_addr);  /* BS_BASE_ADDR */
    *p++ = emit(TARGET_RDMA, 0x5028, 0);          /* NRDMA_CFG */
    *p++ = emit(TARGET_RDMA, 0x502c, 0);          /* BN_BASE_ADDR */
    *p++ = emit(TARGET_RDMA, 0x5034, 0x00000001); /* ERDMA_CFG: disable */
    *p++ = emit(TARGET_RDMA, 0x5038, 0);          /* EW_BASE_ADDR */
    *p++ = emit(TARGET_RDMA, 0x5040, 0);          /* EW_SURF_STRIDE */
    *p++ = emit(TARGET_RDMA, 0x5000, 0x000f0010); /* FEAT_MODE_CFG: burst=15, MRDMA_DISABLE=1 */
    *p++ = emit(TARGET_RDMA, 0x5038, 0);          /* SRC_DMA_CFG */
    *p++ = emit(TARGET_RDMA, 0x503c, 0);          /* SURF_NOTCH */
    *p++ = emit(TARGET_RDMA, 0x5040, 0);          /* PAD_CFG */
    *p++ = emit(TARGET_RDMA, 0x5044, 0x01010101); /* WEIGHT */
    *p++ = emit(TARGET_RDMA, 0x5048, 0);          /* EW_SURF_NOTCH */

    /* Chain pointer: none (single task) */
    *p++ = 0;  /* null chain entry */
    *p++ = emit(TARGET_PC, 0x0014, 0);            /* PC_REGISTER_AMOUNTS: 0 */

    /* Control: activate */
    *p++ = 0x0041000000000000ULL;
    *p++ = emit(0x0081, 0x0008, 0x0000001d);      /* OP_ENABLE */

    return (unsigned)(p - buf);
}

static int create_bo(int fd, uint32_t size, uint32_t *handle, uint64_t *dma_addr, uint64_t *offset)
{
    struct drm_rocket_create_bo req = { .size = size };
    int ret = ioctl(fd, DRM_IOCTL_ROCKET_CREATE_BO, &req);
    if (ret) { perror("CREATE_BO"); return -1; }
    *handle = req.handle;
    *dma_addr = req.dma_address;
    *offset = req.offset;
    return 0;
}

int main(void)
{
    int fd = open("/dev/accel/accel0", O_RDWR);
    if (fd < 0) { perror("open accel0"); return 1; }
    printf("Opened /dev/accel/accel0 (fd=%d)\n", fd);

    /* Create BOs */
    uint32_t in_h, out_h, wt_h, bias_h, regcmd_h;
    uint64_t in_dma, out_dma, wt_dma, bias_dma, regcmd_dma;
    uint64_t in_off, out_off, wt_off, bias_off, regcmd_off;

    /* Input: 16 bytes (1x1x16 in NPU interleaved format) */
    if (create_bo(fd, 4096, &in_h, &in_dma, &in_off)) return 1;
    printf("Input  BO: handle=%u dma=0x%llx\n", in_h, (unsigned long long)in_dma);

    /* Output: 16 bytes */
    if (create_bo(fd, 4096, &out_h, &out_dma, &out_off)) return 1;
    printf("Output BO: handle=%u dma=0x%llx\n", out_h, (unsigned long long)out_dma);

    /* Weights: 32*16 = 512 bytes (padded) */
    if (create_bo(fd, 4096, &wt_h, &wt_dma, &wt_off)) return 1;
    printf("Weight BO: handle=%u dma=0x%llx\n", wt_h, (unsigned long long)wt_dma);

    /* Bias: 16 * 4 = 64 bytes */
    if (create_bo(fd, 4096, &bias_h, &bias_dma, &bias_off)) return 1;
    printf("Bias   BO: handle=%u dma=0x%llx\n", bias_h, (unsigned long long)bias_dma);

    /* Regcmd: room for 256 entries */
    if (create_bo(fd, 4096, &regcmd_h, &regcmd_dma, &regcmd_off)) return 1;
    printf("Regcmd BO: handle=%u dma=0x%llx\n", regcmd_h, (unsigned long long)regcmd_dma);

    /* Map BOs */
    uint8_t *in_buf = mmap(NULL, 4096, PROT_READ|PROT_WRITE, MAP_SHARED, fd, in_off);
    uint8_t *out_buf = mmap(NULL, 4096, PROT_READ|PROT_WRITE, MAP_SHARED, fd, out_off);
    uint8_t *wt_buf = mmap(NULL, 4096, PROT_READ|PROT_WRITE, MAP_SHARED, fd, wt_off);
    int32_t *bias_buf = (int32_t *)mmap(NULL, 4096, PROT_READ|PROT_WRITE, MAP_SHARED, fd, bias_off);
    uint64_t *regcmd_buf = (uint64_t *)mmap(NULL, 4096, PROT_READ|PROT_WRITE, MAP_SHARED, fd, regcmd_off);

    if (in_buf == MAP_FAILED || out_buf == MAP_FAILED || wt_buf == MAP_FAILED ||
        bias_buf == MAP_FAILED || regcmd_buf == MAP_FAILED) {
        perror("mmap"); return 1;
    }

    /*
     * Fill input: 16 channels.
     * Channel 0: value = 10 (basic identity test)
     * Channel 1: value = -3 (NVDLA rounding test: -3 >> 1 should be -2, not -1)
     * Channels 2-15: value = 10
     */
    memset(in_buf, 0, 4096);
    in_buf[0] = 10;
    in_buf[1] = (uint8_t)(int8_t)(-3);
    for (int c = 2; c < 16; c++)
        in_buf[c] = 10;

    /* Fill weights: identity-like (w[oc][0][0][ic] = 1 if oc==ic, else 0)
     * Weight layout: [oc1][ic1][kx][ky][oc2][ic2] with WEIGHT_ATOMIC_SIZE=32
     * For 1x1 filter, 16 output, 16 input: oc1=0, ic1=0, kx=0, ky=0
     * ic2_count = MIN(MAX(in_ch, FEATURE_ATOMIC_SIZE), WEIGHT_ATOMIC_SIZE)
     *           = MIN(MAX(16, 16), 32) = 16
     * offset = oc2 * ic2_count + ic2 = oc * 16 + ic
     */
    memset(wt_buf, 0, 4096);
    for (int oc = 0; oc < 16; oc++) {
        wt_buf[oc * 16 + oc] = 1;  /* w[oc][oc] = 1 (signed) */
    }

    /* Fill bias: all zeros */
    memset(bias_buf, 0, 4096);

    /* Clear output */
    memset(out_buf, 0xff, 4096);

    /* Fill regcmd */
    unsigned regcmd_count = build_regcmd(regcmd_buf,
        (uint32_t)in_dma, (uint32_t)wt_dma,
        (uint32_t)out_dma, (uint32_t)bias_dma);
    printf("Regcmd: %u entries (%u bytes)\n", regcmd_count, regcmd_count * 8);

    /* Sync BOs for device access */
    struct drm_rocket_fini_bo fini;
    fini.handle = in_h; fini.reserved = 0; ioctl(fd, DRM_IOCTL_ROCKET_FINI_BO, &fini);
    fini.handle = wt_h; ioctl(fd, DRM_IOCTL_ROCKET_FINI_BO, &fini);
    fini.handle = bias_h; ioctl(fd, DRM_IOCTL_ROCKET_FINI_BO, &fini);
    fini.handle = regcmd_h; ioctl(fd, DRM_IOCTL_ROCKET_FINI_BO, &fini);

    /* Build submit structures */
    struct drm_rocket_task task = {
        .regcmd = (uint32_t)regcmd_dma,
        .regcmd_count = regcmd_count * 2,  /* count in uint32_t units (entries*2) */
    };

    uint32_t in_handles[] = { in_h, wt_h, bias_h, regcmd_h };
    uint32_t out_handles[] = { out_h };

    struct drm_rocket_job job = {
        .tasks = (uint64_t)(uintptr_t)&task,
        .in_bo_handles = (uint64_t)(uintptr_t)in_handles,
        .out_bo_handles = (uint64_t)(uintptr_t)out_handles,
        .task_count = 1,
        .task_struct_size = sizeof(struct drm_rocket_task),
        .in_bo_handle_count = 4,
        .out_bo_handle_count = 1,
    };

    struct drm_rocket_submit submit = {
        .jobs = (uint64_t)(uintptr_t)&job,
        .job_count = 1,
        .job_struct_size = sizeof(struct drm_rocket_job),
    };

    printf("Submitting NPU job...\n");
    int ret = ioctl(fd, DRM_IOCTL_ROCKET_SUBMIT, &submit);
    if (ret) {
        printf("SUBMIT failed: %s (errno=%d)\n", strerror(errno), errno);
        /* Continue anyway to check output */
    } else {
        printf("SUBMIT succeeded!\n");
    }

    /* Wait for completion by PREP_BO on output */
    struct drm_rocket_prep_bo prep = {
        .handle = out_h,
        .timeout_ns = 5000000000LL,  /* 5 seconds */
    };
    printf("Waiting for NPU completion...\n");
    ret = ioctl(fd, DRM_IOCTL_ROCKET_PREP_BO, &prep);
    if (ret) {
        printf("PREP_BO (wait) failed: %s\n", strerror(errno));
    } else {
        printf("NPU job completed!\n");
    }

    /* Read output */
    printf("Output (first 16 bytes, y-major interleaved):\n");
    for (int i = 0; i < 16; i++)
        printf("  [%2d] = %d (0x%02x)\n", i, (int8_t)out_buf[i], out_buf[i]);

    /*
     * Check results with CLIP_TRUNCATE=1:
     *
     * Channel 0: input=10, acc=10, nvdla_truncate(10,1):
     *   guide=0, result = 10>>1 = 5. OUT_CVT(5*16384>>14) = 5.
     *
     * Channel 1: input=-3, acc=-3, nvdla_truncate(-3,1):
     *   sign=1, guide=1, sticky=0, round_up = 1 & (0|0) = 0
     *   result = (-3>>1) + 0 = -2. OUT_CVT(-2*16384>>14) = -2.
     *   (Simple round-half-up would give (-3+1)>>1 = -1 — WRONG)
     *
     * Channels 2-15: same as channel 0, expected 5.
     */
    int8_t expected[16];
    expected[0] = 5;    /* 10 >> 1 = 5 */
    expected[1] = -2;   /* -3 >> 1 = -2 (NVDLA rounding) */
    for (int i = 2; i < 16; i++) expected[i] = 5;

    int pass = 1;
    for (int i = 0; i < 16; i++) {
        int8_t val = (int8_t)out_buf[i];
        if (val != expected[i]) {
            printf("  MISMATCH ch[%d]: got %d, expected %d\n",
                   i, val, expected[i]);
            pass = 0;
        }
    }

    if (pass) {
        printf("PASS: Identity conv with NVDLA truncation correct\n");
        printf("  ch[0]=5 (10>>1), ch[1]=-2 (-3>>1 NVDLA rounding), ch[2-15]=5\n");
    } else {
        printf("FAIL: Output mismatch — check convolution engine\n");
    }

    /* Cleanup */
    munmap(in_buf, 4096);
    munmap(out_buf, 4096);
    munmap(wt_buf, 4096);
    munmap((void *)bias_buf, 4096);
    munmap((void *)regcmd_buf, 4096);
    close(fd);

    return pass ? 0 : 1;
}
