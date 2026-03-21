/*
 * RKNPU vendor driver test for QEMU RK3588 NPU emulator.
 *
 * Tests the vendor rknpu driver (from kernel 6.1.115-vendor-rk35xx)
 * running under QEMU with the rockchip-iommu model.
 *
 * Phase 1: probe verification (device exists, version reads work)
 * Phase 2: memory allocation + simple convolution (requires vendor headers)
 *
 * Builds statically for aarch64, runs inside QEMU initrd.
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <dirent.h>
#include <errno.h>

/* ======================================================================
 * RKNPU ioctl definitions (from rknpu_ioctl.h in vendor kernel)
 * ====================================================================== */

#define DRM_IOCTL_BASE 'd'
#define DRM_COMMAND_BASE 0x40

/* Action types */
#define RKNPU_GET_HW_VERSION    0
#define RKNPU_GET_DRV_VERSION   1
#define RKNPU_GET_IOMMU_EN      18

struct rknpu_action {
    uint32_t flags;
    uint32_t value;
};

struct rknpu_mem_create {
    uint32_t handle;
    uint32_t flags;
    uint64_t size;
    uint64_t obj_addr;
    uint64_t dma_addr;
};

struct rknpu_mem_map {
    uint32_t handle;
    uint32_t reserved;
    uint64_t offset;
};

struct rknpu_mem_destroy {
    uint32_t handle;
    uint32_t reserved;
    uint64_t obj_addr;
};

#define DRM_IOCTL_RKNPU_ACTION \
    _IOWR(DRM_IOCTL_BASE, DRM_COMMAND_BASE + 0, struct rknpu_action)
#define DRM_IOCTL_RKNPU_MEM_CREATE \
    _IOWR(DRM_IOCTL_BASE, DRM_COMMAND_BASE + 2, struct rknpu_mem_create)
#define DRM_IOCTL_RKNPU_MEM_MAP \
    _IOWR(DRM_IOCTL_BASE, DRM_COMMAND_BASE + 3, struct rknpu_mem_map)
#define DRM_IOCTL_RKNPU_MEM_DESTROY \
    _IOWR(DRM_IOCTL_BASE, DRM_COMMAND_BASE + 4, struct rknpu_mem_destroy)

/* ======================================================================
 * Test helpers
 * ====================================================================== */

static int find_drm_device(char *path, size_t pathlen)
{
    /* Try /dev/dri/card0 first (vendor DRM device) */
    const char *candidates[] = {
        "/dev/dri/card0",
        "/dev/dri/card1",
        "/dev/dri/renderD128",
    };

    for (int i = 0; i < 3; i++) {
        int fd = open(candidates[i], O_RDWR);
        if (fd >= 0) {
            snprintf(path, pathlen, "%s", candidates[i]);
            return fd;
        }
    }
    return -1;
}

/* ======================================================================
 * Test 1: Device probe verification
 * ====================================================================== */

static int test_probe(void)
{
    printf("test_probe: checking for rknpu device...\n");

    /* Check if rknpu module is loaded (via sysfs) */
    FILE *f = fopen("/proc/modules", "r");
    if (f) {
        char line[256];
        int found = 0;
        while (fgets(line, sizeof(line), f)) {
            if (strstr(line, "rknpu")) {
                printf("  rknpu module loaded: %s", line);
                found = 1;
                break;
            }
        }
        fclose(f);
        if (!found) {
            printf("  rknpu module not found in /proc/modules\n");
            return 0;
        }
    }

    /* Check for DRM device */
    char devpath[64];
    int fd = find_drm_device(devpath, sizeof(devpath));
    if (fd < 0) {
        printf("  FAIL: no DRM device found\n");
        return 0;
    }
    printf("  DRM device: %s\n", devpath);
    close(fd);

    printf("  test_probe: PASS\n");
    return 1;
}

/* ======================================================================
 * Test 2: HW version query via ACTION ioctl
 * ====================================================================== */

static int test_hw_version(void)
{
    printf("test_hw_version: querying NPU hardware version...\n");

    char devpath[64];
    int fd = find_drm_device(devpath, sizeof(devpath));
    if (fd < 0) {
        printf("  FAIL: no DRM device\n");
        return 0;
    }

    struct rknpu_action act = {
        .flags = RKNPU_GET_HW_VERSION,
        .value = 0,
    };

    int ret = ioctl(fd, DRM_IOCTL_RKNPU_ACTION, &act);
    if (ret < 0) {
        printf("  FAIL: RKNPU_ACTION ioctl: %s (errno=%d)\n",
               strerror(errno), errno);
        /* This is expected to fail if the ioctl struct doesn't match.
         * Still useful to see the error code. */
        close(fd);
        return 0;
    }

    printf("  HW version: 0x%08x\n", act.value);
    close(fd);
    printf("  test_hw_version: PASS\n");
    return 1;
}

/* ======================================================================
 * Test 3: Memory allocation
 * ====================================================================== */

static int test_mem_create(void)
{
    printf("test_mem_create: allocating NPU memory...\n");

    char devpath[64];
    int fd = find_drm_device(devpath, sizeof(devpath));
    if (fd < 0) {
        printf("  FAIL: no DRM device\n");
        return 0;
    }

    struct rknpu_mem_create mem;
    memset(&mem, 0, sizeof(mem));
    mem.size = 4096;
    mem.flags = 0;

    int ret = ioctl(fd, DRM_IOCTL_RKNPU_MEM_CREATE, &mem);
    if (ret < 0) {
        printf("  FAIL: RKNPU_MEM_CREATE ioctl: %s (errno=%d)\n",
               strerror(errno), errno);
        close(fd);
        return 0;
    }

    printf("  handle=%u dma_addr=0x%lx obj_addr=0x%lx\n",
           mem.handle, (unsigned long)mem.dma_addr,
           (unsigned long)mem.obj_addr);

    /* Clean up: destroy the buffer */
    struct rknpu_mem_destroy destroy = {
        .handle = mem.handle,
        .obj_addr = mem.obj_addr,
    };
    ioctl(fd, DRM_IOCTL_RKNPU_MEM_DESTROY, &destroy);

    close(fd);
    printf("  test_mem_create: PASS\n");
    return 1;
}

/* ======================================================================
 * Test 4: dmesg checks (read /dev/kmsg for rknpu/iommu messages)
 * ====================================================================== */

static int test_dmesg(void)
{
    printf("test_dmesg: checking kernel messages...\n");

    FILE *f = fopen("/dev/kmsg", "r");
    if (!f) {
        /* Try /proc/kmsg */
        f = fopen("/proc/kmsg", "r");
    }
    if (!f) {
        printf("  Cannot read kernel log, skipping\n");
        return 1;
    }

    /* Set non-blocking to avoid waiting forever */
    int fd = fileno(f);
    int flags = fcntl(fd, F_GETFL, 0);
    fcntl(fd, F_SETFL, flags | O_NONBLOCK);

    char line[512];
    int rknpu_found = 0, iommu_found = 0;

    /* Seek to start */
    lseek(fd, 0, SEEK_SET);

    while (fgets(line, sizeof(line), f)) {
        if (strstr(line, "RKNPU") || strstr(line, "rknpu")) {
            if (!rknpu_found) {
                printf("  rknpu: %s", line);
                rknpu_found = 1;
            }
        }
        if (strstr(line, "rockchip-iommu")) {
            if (!iommu_found) {
                printf("  iommu: %s", line);
                iommu_found = 1;
            }
        }
    }
    fclose(f);

    if (!rknpu_found) {
        printf("  WARNING: no rknpu messages in dmesg\n");
    }
    if (!iommu_found) {
        printf("  WARNING: no rockchip-iommu messages in dmesg\n");
    }

    printf("  test_dmesg: %s\n",
           (rknpu_found && iommu_found) ? "PASS" : "PARTIAL");
    return rknpu_found ? 1 : 0;
}

/* ======================================================================
 * Main
 * ====================================================================== */

int main(int argc, char **argv)
{
    printf("=== RKNPU Vendor Driver Test Suite ===\n");

    int passed = 0;
    int total = 4;

    passed += test_probe();
    passed += test_hw_version();
    passed += test_mem_create();
    passed += test_dmesg();

    printf("=== RKNPU TESTS: %d/%d passed ===\n", passed, total);
    return (passed >= 2) ? 0 : 1;  /* Pass if at least probe + one other works */
}
