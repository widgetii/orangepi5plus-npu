# Out-of-tree build for RKNPU driver (instrumented fork)
# Build on the board: make
# Or cross-compile: make ARCH=arm64 CROSS_COMPILE=aarch64-linux-gnu-

KVER ?= $(shell uname -r)
KDIR ?= /lib/modules/$(KVER)/build

# Module name: rknpu_trace (different from built-in rknpu)
obj-m := rknpu_trace.o

rknpu_trace-y := rknpu_drv.o rknpu_job.o rknpu_gem.o rknpu_fence.o \
                 rknpu_iommu.o rknpu_mem.o rknpu_mm.o rknpu_reset.o

ccflags-y := -I$(src)/include \
             -DCONFIG_ROCKCHIP_RKNPU_DRM_GEM \
             -DCONFIG_ROCKCHIP_RKNPU_FENCE \
             -DCONFIG_ROCKCHIP_RKNPU_IOMMU \
             -DCONFIG_NO_GKI

all:
	$(MAKE) -C $(KDIR) M=$(PWD) modules

clean:
	$(MAKE) -C $(KDIR) M=$(PWD) clean

install:
	$(MAKE) -C $(KDIR) M=$(PWD) modules_install

.PHONY: all clean install
