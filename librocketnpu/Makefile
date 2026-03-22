CC ?= gcc
CFLAGS = -O2 -Wall -Wno-unused-function -I/usr/include/libdrm -Iinclude -Isrc
LDFLAGS = -lm -ldrm

# Auto-detect aarch64 for NEON
ARCH := $(shell uname -m)
ifeq ($(ARCH),aarch64)
  CFLAGS += -march=armv8-a
endif

SRCS = src/rnpu_drm.c \
       src/rnpu_tflite.c \
       src/rnpu_coefs.c \
       src/rnpu_task.c \
       src/rnpu_regcmd.c \
       src/rnpu_convert.c \
       src/rnpu_sw_ops.c \
       src/rnpu_model.c

OBJS = $(SRCS:.c=.o)

.PHONY: all clean test

all: librocketnpu.so test_mobilenet test_yolo test_debug test_sw_ops test_rknpu_abi

librocketnpu.so: $(OBJS)
	$(CC) -shared -o $@ $^ $(LDFLAGS)

%.o: %.c
	$(CC) $(CFLAGS) -fPIC -c -o $@ $<

test_mobilenet: tests/test_mobilenet.c librocketnpu.so
	$(CC) $(CFLAGS) -o $@ $< -L. -lrocketnpu -Wl,-rpath,'$$ORIGIN' $(LDFLAGS)

test_yolo: tests/test_yolo.c librocketnpu.so
	$(CC) $(CFLAGS) -o $@ $< -L. -lrocketnpu -Wl,-rpath,'$$ORIGIN' $(LDFLAGS)

test_debug: tests/test_debug.c librocketnpu.so
	$(CC) $(CFLAGS) -o $@ $< -L. -lrocketnpu -Wl,-rpath,'$$ORIGIN' $(LDFLAGS)

test_sw_ops: tests/test_sw_ops.c librocketnpu.so
	$(CC) $(CFLAGS) -o $@ $< -L. -lrocketnpu -Wl,-rpath,'$$ORIGIN' $(LDFLAGS)

test_rknpu_abi: tests/test_rknpu_abi.c librocketnpu.so
	$(CC) $(CFLAGS) -o $@ $< -L. -lrocketnpu -Wl,-rpath,'$$ORIGIN' $(LDFLAGS)

test: test_sw_ops test_rknpu_abi
	@LD_LIBRARY_PATH=. ./test_sw_ops
	@LD_LIBRARY_PATH=. ./test_rknpu_abi
	@echo "Run: LD_LIBRARY_PATH=. ./test_mobilenet <model.tflite>"

clean:
	rm -f $(OBJS) librocketnpu.so test_mobilenet test_yolo test_debug test_rknpu_abi
