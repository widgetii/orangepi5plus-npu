CC ?= gcc
CFLAGS = -O2 -Wall -Wextra -fPIC -Iinclude -Isrc
LDFLAGS = -shared -ldl -lm

SRCS = src/openrknn_api.c \
       src/openrknn_proxy.c \
       src/openrknn_model.c \
       src/openrknn_flatbuf.c \
       src/openrknn_query.c \
       src/openrknn_drm.c \
       src/openrknn_memory.c \
       src/openrknn_input.c \
       src/openrknn_run.c \
       src/openrknn_output.c

OBJS = $(SRCS:.c=.o)

all: librknn_api.so

librknn_api.so: $(OBJS)
	$(CC) $(LDFLAGS) -o $@ $^

src/%.o: src/%.c src/openrknn.h include/rknn_api.h
	$(CC) $(CFLAGS) -c -o $@ $<

clean:
	rm -f $(OBJS) librknn_api.so test_proxy

# Test programs
test_proxy: tests/test_proxy.c librknn_api.so
	$(CC) -O2 -Wall -Iinclude -o $@ $< -L. -lrknn_api -Wl,-rpath,.

.PHONY: all clean
