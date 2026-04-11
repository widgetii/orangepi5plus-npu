/*
 * demo_deeplabv3 — real-world segmentation demo using openrknn.
 *
 * Reads a video from stdin (or via ffmpeg subprocess), runs deeplabv3
 * INT8 inference on every frame using openrknn (no vendor librknnrt
 * symbols are linked in — the binary depends only on
 * openrknn/librknn_api.so), composites a colored class-mask overlay,
 * and writes the result back through ffmpeg into an MP4 file.
 *
 * Pipeline:
 *   ffmpeg(decode) → input_q  →  N inference workers  → output_q
 *                                                            ↓
 *                                                  reorder + composite
 *                                                            ↓
 *                                                      ffmpeg(encode)
 *
 * Each worker has its own rknn_context and (in pinned mode) is bound
 * to a single NPU core via rknn_set_core_mask. The output thread is
 * single-threaded so frames land in the encoder in source order.
 *
 * Reports two FPS numbers at the end:
 *   - inference-only FPS  (sum of worker counts / wall-clock window)
 *   - end-to-end FPS      (frames written to encoder / wall-clock)
 *
 * Build: see openrknn/Makefile rule for `tests/demo_deeplabv3`.
 *
 * SPDX-License-Identifier: MIT
 */
#define _GNU_SOURCE
#include "rknn_api.h"
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <sys/wait.h>
#include <errno.h>

#define DLAB_W   513
#define DLAB_H   513
#define DLAB_OW  65
#define DLAB_OH  65
#define DLAB_OC  21

#define MAX_WORKERS 8

/* High-contrast palette tuned for visibility on real-world video.
 * Index 0 (background) is left untouched by composite_overlay so its
 * value here is unused. Person (class 15) is the most common hit on
 * the demo clip, hence the vivid magenta. */
static const uint8_t pascal_palette[DLAB_OC][3] = {
    {  0,   0,   0}, /* background — unused */
    {255,   0,   0}, /* aeroplane    — red */
    {255, 255,   0}, /* bicycle      — yellow */
    {  0, 255,   0}, /* bird         — green */
    {  0, 255, 255}, /* boat         — cyan */
    {  0, 200, 255}, /* bottle       — sky blue */
    {255, 128,   0}, /* bus          — orange */
    {255, 200,   0}, /* car          — amber */
    {180, 100,  20}, /* cat          — brown */
    {  0, 220, 180}, /* chair        — teal */
    {200, 160,  60}, /* cow          — tan */
    {200,  80, 200}, /* diningtable  — purple */
    {220,  60,  60}, /* dog          — coral */
    {120,  60, 200}, /* horse        — violet */
    {  0,  80, 255}, /* motorbike    — blue */
    {255,   0, 200}, /* person       — magenta */
    {  0, 200,  60}, /* pottedplant  — leaf green */
    {220, 160, 200}, /* sheep        — pink */
    {180, 100, 100}, /* sofa         — dusty red */
    {100, 200, 200}, /* train        — pale cyan */
    {200, 200,   0}, /* tvmonitor    — olive */
};

static uint64_t now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}

/* ----------------------------------------------------------------------
 * Bounded thread-safe queue of void* pointers.
 * -------------------------------------------------------------------- */
struct queue {
    void           **slots;
    int              cap;
    int              head;
    int              tail;
    int              size;
    pthread_mutex_t  mtx;
    pthread_cond_t   not_full;
    pthread_cond_t   not_empty;
};

static void queue_init(struct queue *q, int cap) {
    q->slots = calloc((size_t)cap, sizeof(void *));
    q->cap = cap;
    q->head = q->tail = q->size = 0;
    pthread_mutex_init(&q->mtx, NULL);
    pthread_cond_init(&q->not_full, NULL);
    pthread_cond_init(&q->not_empty, NULL);
}

static void queue_push(struct queue *q, void *item) {
    pthread_mutex_lock(&q->mtx);
    while (q->size == q->cap)
        pthread_cond_wait(&q->not_full, &q->mtx);
    q->slots[q->tail] = item;
    q->tail = (q->tail + 1) % q->cap;
    q->size++;
    pthread_cond_signal(&q->not_empty);
    pthread_mutex_unlock(&q->mtx);
}

static void *queue_pop(struct queue *q) {
    pthread_mutex_lock(&q->mtx);
    while (q->size == 0)
        pthread_cond_wait(&q->not_empty, &q->mtx);
    void *item = q->slots[q->head];
    q->head = (q->head + 1) % q->cap;
    q->size--;
    pthread_cond_signal(&q->not_full);
    pthread_mutex_unlock(&q->mtx);
    return item;
}

/* ----------------------------------------------------------------------
 * Per-frame work packet.
 * -------------------------------------------------------------------- */
struct frame_pkt {
    int      frame_id;          /* monotonic, 0..N-1 */
    int      sentinel;          /* 1 = end-of-stream marker */
    int      width;
    int      height;
    uint8_t *rgb;               /* W*H*3 bytes, source RGB24 */
    uint8_t *cls;               /* W*H bytes, per-pixel class id (after worker) */
    uint64_t worker_lat_ns;     /* per-frame inference latency for stats */
};

static struct frame_pkt *frame_alloc(int w, int h) {
    struct frame_pkt *p = calloc(1, sizeof(*p));
    p->width = w;
    p->height = h;
    p->rgb = malloc((size_t)w * (size_t)h * 3);
    p->cls = malloc((size_t)w * (size_t)h);
    return p;
}

static void frame_free(struct frame_pkt *p) {
    if (!p) return;
    free(p->rgb);
    free(p->cls);
    free(p);
}

/* ----------------------------------------------------------------------
 * Tiling strategy.
 *
 * A 1280×720 frame letterboxed into 513×513 wastes ~44% of the input
 * tensor on gray padding, leaving only 513×288 of real pixels. That
 * 2.5× vertical downsample crushes small objects (distant cars, the
 * top row of a parking lot) below the model's 65×65 output grid —
 * classes end up smeared across 1-2 cells and argmax flips trivially
 * frame-to-frame → flickering.
 *
 * Fix: for wide frames, run the model twice on two overlapping
 * square crops (tiles). Each tile covers the full frame height and
 * half the width + overlap:
 *
 *   tile 0: src rect (0,             0, min(h,w/2+ov), h)
 *   tile 1: src rect (w - tile_size, 0, tile_size,    h)
 *
 * Both tiles get the full 513×513 inference budget → objects are
 * ~2× larger in the input, giving ~4× more argmax cells to land in.
 *
 * For portrait or near-square frames (aspect < 1.25) one tile is
 * enough; letterbox fallback is used there.
 * -------------------------------------------------------------------- */

#define MAX_TILES 2

struct box { int x0, y0, w, h; };

struct tile_plan {
    int        n_tiles;
    struct box src[MAX_TILES];   /* source rect each tile samples from */
    struct box content[MAX_TILES]; /* content area inside 513×513 canvas */
};

static struct tile_plan plan_tiles(int sw, int sh)
{
    struct tile_plan tp;
    memset(&tp, 0, sizeof(tp));
    double aspect = (double)sw / sh;

    if (aspect > 1.25 && sh <= sw) {
        /* Wide: two horizontal square tiles with ≥20% overlap. */
        int tile_size = sh;  /* square, full height */
        if (tile_size > sw) tile_size = sw;
        tp.n_tiles = 2;
        tp.src[0]     = (struct box){ 0,           0, tile_size, tile_size };
        tp.src[1]     = (struct box){ sw - tile_size, 0, tile_size, tile_size };
        tp.content[0] = (struct box){ 0, 0, DLAB_W, DLAB_H };
        tp.content[1] = (struct box){ 0, 0, DLAB_W, DLAB_H };
    } else if (aspect < 0.8 && sw <= sh) {
        /* Tall: two vertical tiles. */
        int tile_size = sw;
        if (tile_size > sh) tile_size = sh;
        tp.n_tiles = 2;
        tp.src[0]     = (struct box){ 0, 0,             tile_size, tile_size };
        tp.src[1]     = (struct box){ 0, sh - tile_size, tile_size, tile_size };
        tp.content[0] = (struct box){ 0, 0, DLAB_W, DLAB_H };
        tp.content[1] = (struct box){ 0, 0, DLAB_W, DLAB_H };
    } else {
        /* Near-square: single letterboxed tile. */
        double scale = (double)DLAB_W / (sw > sh ? sw : sh);
        int w_used = (int)(sw * scale + 0.5);
        int h_used = (int)(sh * scale + 0.5);
        if (w_used > DLAB_W) w_used = DLAB_W;
        if (h_used > DLAB_H) h_used = DLAB_H;
        int x0 = (DLAB_W - w_used) / 2;
        int y0 = (DLAB_H - h_used) / 2;
        tp.n_tiles = 1;
        tp.src[0]     = (struct box){ 0, 0, sw, sh };
        tp.content[0] = (struct box){ x0, y0, w_used, h_used };
    }
    return tp;
}

/* Bilinear resize from an arbitrary src rectangle (src_rect inside a
 * frame of (sw,sh)) into a (dw,dh) sub-region of dst at offset
 * (dx0,dy0). Outside that sub-region is untouched. */
static void resize_region_bilinear(const uint8_t *src, int sw, int sh,
                                   struct box src_rect,
                                   uint8_t *dst, int dst_stride_bytes,
                                   int dx0, int dy0, int dw, int dh)
{
    (void)sh;
    for (int dy = 0; dy < dh; dy++) {
        int64_t fy = (int64_t)dy * src_rect.h * 65536 / dh;
        int iy = (int)(fy >> 16);
        int ry = (int)(fy & 0xFFFF);
        int iy0 = src_rect.y0 + iy;
        int iy1 = iy + 1 < src_rect.h ? iy0 + 1 : iy0;
        const uint8_t *row0 = src + (size_t)iy0 * sw * 3;
        const uint8_t *row1 = src + (size_t)iy1 * sw * 3;
        uint8_t *drow = dst + ((size_t)(dy0 + dy) * (dst_stride_bytes / 3) + dx0) * 3;
        for (int dx = 0; dx < dw; dx++) {
            int64_t fxp = (int64_t)dx * src_rect.w * 65536 / dw;
            int ix = (int)(fxp >> 16);
            int rx = (int)(fxp & 0xFFFF);
            int ix0 = src_rect.x0 + ix;
            int ix1 = ix + 1 < src_rect.w ? ix0 + 1 : ix0;
            const uint8_t *p00 = row0 + ix0 * 3;
            const uint8_t *p01 = row0 + ix1 * 3;
            const uint8_t *p10 = row1 + ix0 * 3;
            const uint8_t *p11 = row1 + ix1 * 3;
            int wx1 = rx, wx0 = 65536 - rx;
            int wy1 = ry, wy0 = 65536 - ry;
            for (int c = 0; c < 3; c++) {
                int64_t top = (int64_t)p00[c] * wx0 + (int64_t)p01[c] * wx1;
                int64_t bot = (int64_t)p10[c] * wx0 + (int64_t)p11[c] * wx1;
                int64_t v = (top * wy0 + bot * wy1) >> 32;
                if (v < 0) v = 0;
                if (v > 255) v = 255;
                drow[dx * 3 + c] = (uint8_t)v;
            }
        }
    }
}

/* Prepare the 513×513 input buffer for one tile:
 * - Fill with neutral gray (for the letterbox case)
 * - Bilinearly resize the tile's source rectangle into content area. */
static void prepare_tile_input(const uint8_t *src, int sw, int sh,
                               const struct tile_plan *tp, int t,
                               uint8_t *dst_513)
{
    memset(dst_513, 128, (size_t)DLAB_W * DLAB_H * 3);
    resize_region_bilinear(src, sw, sh, tp->src[t],
                           dst_513, DLAB_W * 3,
                           tp->content[t].x0, tp->content[t].y0,
                           tp->content[t].w, tp->content[t].h);
}

/* ----------------------------------------------------------------------
 * DeepLabv3 input quantization.
 *
 * Vendor reports input as INT8, scale=0.007843 (~1/127.5), zp=0. The
 * model expects [-1,1]-normalized values: x_norm = (rgb - 127.5)/127.5.
 * Quantized: x_int8 = round(x_norm / scale + zp) ≈ rgb - 128, clipped.
 * -------------------------------------------------------------------- */
static void quantize_input(const uint8_t *rgb_513, int8_t *out_513)
{
    const int n = DLAB_W * DLAB_H * 3;
    for (int i = 0; i < n; i++) {
        int v = (int)rgb_513[i] - 128;
        if (v < -128) v = -128;
        if (v >  127) v =  127;
        out_513[i] = (int8_t)v;
    }
}

/* ----------------------------------------------------------------------
 * Multi-tile logit merge + bilinear upsample + argmax.
 *
 * For each tile t the model output is a 65×65×21 int8 logit volume.
 * We:
 *
 *   1. Bilinearly upsample each tile's logits in place to a per-tile
 *      "HI grid" of HI_RES×HI_RES×21 int16 (smoother boundaries,
 *      stable across frames).
 *   2. Argmax at HI_RES×HI_RES to get a HI_RES×HI_RES class index map
 *      per tile.
 *   3. For every destination pixel in the full frame, map into
 *      whichever tile covers that source pixel; in the overlap
 *      region, prefer the tile whose center is closer (so the seam
 *      falls between two equally-good predictions).
 *
 * openrknn returns deeplabv3 output as plain NHWC [1, 65, 65, 21]
 * int8 (src_order=NHWC per openrknn_run.c). Position (y,x,c) is at
 * offset (y*W + x)*C + c.
 *
 * HI_RES is 130 = 2× the model output grid. Each HI cell covers
 * ~2.8 tile-source pixels at 720-tile input; combined with nearest
 * upsample to the full frame that gives ~5.5-pixel effective cell
 * size in a 720×720 tile, vs ~11-pixel cells before. Bilinear in
 * logit space means the class boundaries no longer jump by whole
 * cells between frames.
 * -------------------------------------------------------------------- */
#define HI_RES 130

/* Bilinearly interpolate one 65×65×21 NHWC logit volume into a
 * HI_RES×HI_RES int16 buffer. dst layout is [y][x][c]. The 21-class
 * int16 vector is then argmax-reduced in place to a uint8 class map
 * stored in dst_cls. */
static void upsample_and_argmax_tile(const int8_t *src_nhwc,
                                     uint8_t *dst_cls)
{
    /* Scratch: one interpolated row of logits so we don't need the
     * full HI×HI×21 buffer. */
    int16_t row_logits[HI_RES * DLAB_OC];

    for (int dy = 0; dy < HI_RES; dy++) {
        /* Source y in the 65 grid. */
        int64_t fy = (int64_t)dy * (DLAB_OH - 1) * 65536 / (HI_RES - 1);
        int iy = (int)(fy >> 16);
        int ry = (int)(fy & 0xFFFF);
        int iy1 = iy + 1 < DLAB_OH ? iy + 1 : iy;
        const int8_t *row0 = src_nhwc + (size_t)iy  * DLAB_OW * DLAB_OC;
        const int8_t *row1 = src_nhwc + (size_t)iy1 * DLAB_OW * DLAB_OC;
        int wy1 = ry, wy0 = 65536 - ry;

        for (int dx = 0; dx < HI_RES; dx++) {
            int64_t fx = (int64_t)dx * (DLAB_OW - 1) * 65536 / (HI_RES - 1);
            int ix = (int)(fx >> 16);
            int rx = (int)(fx & 0xFFFF);
            int ix1 = ix + 1 < DLAB_OW ? ix + 1 : ix;
            int wx1 = rx, wx0 = 65536 - rx;

            const int8_t *p00 = row0 + (size_t)ix  * DLAB_OC;
            const int8_t *p01 = row0 + (size_t)ix1 * DLAB_OC;
            const int8_t *p10 = row1 + (size_t)ix  * DLAB_OC;
            const int8_t *p11 = row1 + (size_t)ix1 * DLAB_OC;
            int16_t *out = row_logits + dx * DLAB_OC;
            for (int c = 0; c < DLAB_OC; c++) {
                /* Bilinear in int math: weights sum to 2^32, so we
                 * scale down by 2^24 to fit in int16 while keeping
                 * enough dynamic range for argmax. */
                int64_t top = (int64_t)p00[c] * wx0 + (int64_t)p01[c] * wx1;
                int64_t bot = (int64_t)p10[c] * wx0 + (int64_t)p11[c] * wx1;
                int64_t v = (top * wy0 + bot * wy1) >> 24;
                if (v >  32767) v =  32767;
                if (v < -32768) v = -32768;
                out[c] = (int16_t)v;
            }
        }

        /* Argmax this HI row into dst_cls. */
        uint8_t *dst_row = dst_cls + (size_t)dy * HI_RES;
        for (int dx = 0; dx < HI_RES; dx++) {
            const int16_t *p = row_logits + dx * DLAB_OC;
            int best = 0;
            int16_t best_v = p[0];
            for (int c = 1; c < DLAB_OC; c++) {
                if (p[c] > best_v) { best_v = p[c]; best = c; }
            }
            dst_row[dx] = (uint8_t)best;
        }
    }
}

/* Stitch per-tile HI class maps into the full-frame class map by
 * reverse-mapping each destination pixel through the tile plan.
 *
 * In the overlap region both tiles have a vote; we pick whichever
 * tile's center is closer (this places the seam right between them).
 */
static void stitch_tiles_to_frame(const uint8_t tile_cls[MAX_TILES][HI_RES * HI_RES],
                                  const struct tile_plan *tp,
                                  int out_w, int out_h, uint8_t *cls)
{
    /* Precompute per-tile horizontal span centers for overlap
     * arbitration (wide-tile case). */
    double centers[MAX_TILES];
    for (int t = 0; t < tp->n_tiles; t++)
        centers[t] = tp->src[t].x0 + tp->src[t].w * 0.5;

    for (int dy = 0; dy < out_h; dy++) {
        uint8_t *drow = cls + (size_t)dy * out_w;
        for (int dx = 0; dx < out_w; dx++) {
            /* Find tile(s) that cover this source pixel, pick best. */
            int best_t = -1;
            double best_dist = 1e18;
            for (int t = 0; t < tp->n_tiles; t++) {
                const struct box *b = &tp->src[t];
                if (dx < b->x0 || dx >= b->x0 + b->w) continue;
                if (dy < b->y0 || dy >= b->y0 + b->h) continue;
                /* How close is this pixel to the tile's x center? */
                double d = dx - centers[t];
                if (d < 0) d = -d;
                if (d < best_dist) { best_dist = d; best_t = t; }
            }
            if (best_t < 0) { drow[dx] = 0; continue; }
            /* Map (dx,dy) → tile-local (tx,ty) → content-area
             * fraction → HI grid. */
            const struct box *b  = &tp->src[best_t];
            const struct box *ca = &tp->content[best_t];
            double tx = (double)(dx - b->x0) / b->w;  /* 0..1 */
            double ty = (double)(dy - b->y0) / b->h;
            /* Content area spans [ca->x0 .. ca->x0+ca->w] in the
             * 513 canvas; the HI grid covers the whole 513 canvas,
             * but content may be a sub-rect (letterbox). */
            double cx = (ca->x0 + tx * ca->w) / DLAB_W;  /* 0..1 on 513 */
            double cy = (ca->y0 + ty * ca->h) / DLAB_H;
            int gx = (int)(cx * HI_RES);
            int gy = (int)(cy * HI_RES);
            if (gx < 0) gx = 0;
            if (gx >= HI_RES) gx = HI_RES - 1;
            if (gy < 0) gy = 0;
            if (gy >= HI_RES) gy = HI_RES - 1;
            drow[dx] = tile_cls[best_t][gy * HI_RES + gx];
        }
    }
}

/* ----------------------------------------------------------------------
 * Composite: blend the per-pixel class colour over the source RGB
 * frame at alpha=0.55 for non-background pixels. Background (class 0)
 * is left untouched. Alpha is integer fixed-point: 141/256 source +
 * 115/256 colour matches alpha≈0.45; we use 115/256+141/256 reversed
 * here for stronger overlay = alpha 0.55.
 * -------------------------------------------------------------------- */
static void composite_overlay(uint8_t *rgb, const uint8_t *cls, int w, int h)
{
    const int n = w * h;
    for (int i = 0; i < n; i++) {
        uint8_t c = cls[i];
        if (c == 0 || c >= DLAB_OC) continue;
        const uint8_t *col = pascal_palette[c];
        uint8_t *p = rgb + i * 3;
        /* alpha = 141/256 ≈ 0.55 */
        p[0] = (uint8_t)((p[0] * 115 + col[0] * 141) >> 8);
        p[1] = (uint8_t)((p[1] * 115 + col[1] * 141) >> 8);
        p[2] = (uint8_t)((p[2] * 115 + col[2] * 141) >> 8);
    }
}

/* ----------------------------------------------------------------------
 * Worker thread: pop frame_pkt → resize → quantize → rknn_run →
 * argmax+upsample → push to output queue.
 * -------------------------------------------------------------------- */
struct worker_arg {
    int               idx;
    rknn_context      ctx;
    uint32_t          n_outputs;
    struct queue     *in_q;
    struct queue     *out_q;
    /* stats */
    uint64_t          frames;
    uint64_t          total_lat_ns;
};

static void *worker_thread(void *arg)
{
    struct worker_arg *w = arg;
    uint8_t *resized = malloc(DLAB_W * DLAB_H * 3);
    int8_t  *qbuf    = malloc(DLAB_W * DLAB_H * 3);
    /* Per-tile HI-res class maps. */
    uint8_t (*tile_cls)[HI_RES * HI_RES] =
        malloc(sizeof(uint8_t[MAX_TILES][HI_RES * HI_RES]));
    if (!resized || !qbuf || !tile_cls) return NULL;

    rknn_input in;
    memset(&in, 0, sizeof(in));
    in.index = 0;
    in.buf = qbuf;
    in.size = DLAB_W * DLAB_H * 3;
    in.pass_through = 0;
    in.type = RKNN_TENSOR_INT8;
    in.fmt  = RKNN_TENSOR_NHWC;

    rknn_output outs[8];

    while (1) {
        struct frame_pkt *p = queue_pop(w->in_q);
        if (p->sentinel) {
            queue_push(w->out_q, p);
            break;
        }

        uint64_t t0 = now_ns();

        struct tile_plan tp = plan_tiles(p->width, p->height);

        int fail = 0;
        for (int t = 0; t < tp.n_tiles && !fail; t++) {
            prepare_tile_input(p->rgb, p->width, p->height, &tp, t, resized);
            quantize_input(resized, qbuf);

            if (rknn_inputs_set(w->ctx, 1, &in) != 0) { fail = 1; break; }
            if (rknn_run(w->ctx, NULL) != 0)          { fail = 1; break; }

            memset(outs, 0, sizeof(outs));
            for (uint32_t i = 0; i < w->n_outputs; i++) {
                outs[i].index = i;
                outs[i].want_float = 0;
                outs[i].is_prealloc = 0;
            }
            if (rknn_outputs_get(w->ctx, w->n_outputs, outs, NULL) != 0) {
                fail = 1; break;
            }

            upsample_and_argmax_tile((const int8_t *)outs[0].buf,
                                     tile_cls[t]);

            rknn_outputs_release(w->ctx, w->n_outputs, outs);
        }

        if (fail) {
            fprintf(stderr, "worker[%d]: inference failed on frame %d\n",
                    w->idx, p->frame_id);
            p->sentinel = 1;
            queue_push(w->out_q, p);
            break;
        }

        stitch_tiles_to_frame((const uint8_t (*)[HI_RES * HI_RES])tile_cls,
                              &tp, p->width, p->height, p->cls);

        uint64_t t1 = now_ns();
        p->worker_lat_ns = t1 - t0;
        w->total_lat_ns += (t1 - t0);
        w->frames++;

        queue_push(w->out_q, p);
    }

    free(qbuf);
    free(resized);
    free(tile_cls);
    return NULL;
}

/* ----------------------------------------------------------------------
 * Output thread: pop frame_pkts (in arbitrary order) into a small
 * reorder buffer indexed by frame_id, then drain in source order
 * to the encoder pipe.
 * -------------------------------------------------------------------- */
struct output_arg {
    struct queue *out_q;
    FILE         *enc_pipe;
    int           total_frames;
    int           workers_remaining;
    /* stats */
    uint64_t      frames_written;
    uint64_t      first_write_ns;
    uint64_t      last_write_ns;
};

#define REORDER_CAP 256

static void *output_thread(void *arg)
{
    struct output_arg *o = arg;
    struct frame_pkt **reorder = calloc(REORDER_CAP, sizeof(struct frame_pkt *));
    int next_id = 0;
    int sentinels_seen = 0;

    while (1) {
        struct frame_pkt *p = queue_pop(o->out_q);
        if (p->sentinel) {
            sentinels_seen++;
            frame_free(p);
            if (sentinels_seen >= o->workers_remaining) break;
            continue;
        }

        int slot = p->frame_id % REORDER_CAP;
        reorder[slot] = p;

        /* Drain any contiguous prefix starting at next_id. */
        while (reorder[next_id % REORDER_CAP] &&
               reorder[next_id % REORDER_CAP]->frame_id == next_id) {
            struct frame_pkt *q = reorder[next_id % REORDER_CAP];
            reorder[next_id % REORDER_CAP] = NULL;

            composite_overlay(q->rgb, q->cls, q->width, q->height);
            size_t n = (size_t)q->width * q->height * 3;
            if (fwrite(q->rgb, 1, n, o->enc_pipe) != n) {
                fprintf(stderr, "output: write to encoder failed\n");
            }

            uint64_t t = now_ns();
            if (o->frames_written == 0) o->first_write_ns = t;
            o->last_write_ns = t;
            o->frames_written++;

            frame_free(q);
            next_id++;
        }
    }

    /* Anything still buffered (shouldn't be, but flush to keep sane). */
    for (int i = 0; i < REORDER_CAP; i++)
        if (reorder[i]) frame_free(reorder[i]);
    free(reorder);
    return NULL;
}

/* ----------------------------------------------------------------------
 * main: spin up ffmpeg decode/encode subprocesses, allocate worker
 * pool, drive frames through the pipeline.
 * -------------------------------------------------------------------- */
static FILE *spawn_decoder(const char *path, int *out_w, int *out_h, int *out_n)
{
    /* Probe with ffprobe first. */
    char cmd[1024];
    snprintf(cmd, sizeof(cmd),
        "ffprobe -v error -select_streams v:0 "
        "-show_entries stream=width,height,nb_frames "
        "-of csv=p=0 '%s'", path);
    FILE *pp = popen(cmd, "r");
    if (!pp) return NULL;
    int w = 0, h = 0, n = 0;
    if (fscanf(pp, "%d,%d,%d", &w, &h, &n) != 3) {
        pclose(pp);
        return NULL;
    }
    pclose(pp);
    *out_w = w; *out_h = h; *out_n = n;

    snprintf(cmd, sizeof(cmd),
        "ffmpeg -hide_banner -loglevel error -i '%s' "
        "-f rawvideo -pix_fmt rgb24 -", path);
    return popen(cmd, "r");
}

static FILE *spawn_encoder(const char *path, int w, int h, int fps)
{
    /* libx264 with yuv420p needs even-sized frames; an extra
     * `crop=trunc(iw/2)*2:trunc(ih/2)*2` filter on the encoder side
     * trims one row/column when needed without modifying the C
     * pipeline. */
    char cmd[1024];
    snprintf(cmd, sizeof(cmd),
        "ffmpeg -hide_banner -loglevel error -y "
        "-f rawvideo -pix_fmt rgb24 -s %dx%d -r %d -i - "
        "-vf 'crop=trunc(iw/2)*2:trunc(ih/2)*2' "
        "-c:v libx264 -preset ultrafast -pix_fmt yuv420p -crf 23 "
        "'%s'", w, h, fps, path);
    return popen(cmd, "w");
}

int main(int argc, char **argv)
{
    const char *model_path = "/root/npu-research/deeplabv3.rknn";
    const char *in_path  = NULL;
    const char *out_path = "demo_out.mp4";
    int n_workers = 3;
    int strategy_pinned = 1;
    int max_frames = 0;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--model") && i + 1 < argc) model_path = argv[++i];
        else if (!strcmp(argv[i], "--in") && i + 1 < argc) in_path = argv[++i];
        else if (!strcmp(argv[i], "--out") && i + 1 < argc) out_path = argv[++i];
        else if (!strcmp(argv[i], "--workers") && i + 1 < argc) n_workers = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--strategy") && i + 1 < argc)
            strategy_pinned = !strcmp(argv[++i], "pinned");
        else if (!strcmp(argv[i], "--max-frames") && i + 1 < argc)
            max_frames = atoi(argv[++i]);
        else { fprintf(stderr, "unknown arg: %s\n", argv[i]); return 1; }
    }
    if (!in_path) {
        fprintf(stderr,
            "usage: %s --in IN.mp4 [--out OUT.mp4] [--model PATH] "
            "[--workers N] [--strategy pinned|multicore] [--max-frames N]\n",
            argv[0]);
        return 1;
    }
    if (n_workers < 1 || n_workers > MAX_WORKERS) {
        fprintf(stderr, "workers must be 1..%d\n", MAX_WORKERS);
        return 1;
    }

    int W = 0, H = 0, N = 0;
    FILE *dec = spawn_decoder(in_path, &W, &H, &N);
    if (!dec) { fprintf(stderr, "spawn_decoder failed\n"); return 1; }
    if (max_frames > 0 && max_frames < N) N = max_frames;
    fprintf(stderr, "demo_deeplabv3: %s — %dx%d, %d frames, %d workers (%s)\n",
            in_path, W, H, N, n_workers,
            strategy_pinned ? "pinned" : "multicore");

    FILE *enc = spawn_encoder(out_path, W, H, 30);
    if (!enc) { fprintf(stderr, "spawn_encoder failed\n"); pclose(dec); return 1; }

    /* Load model bytes once. */
    FILE *mf = fopen(model_path, "rb");
    if (!mf) { perror("fopen model"); return 1; }
    fseek(mf, 0, SEEK_END);
    long msz = ftell(mf);
    fseek(mf, 0, SEEK_SET);
    void *model_buf = malloc((size_t)msz);
    if (fread(model_buf, 1, (size_t)msz, mf) != (size_t)msz) {
        fprintf(stderr, "fread model truncated\n"); return 1;
    }
    fclose(mf);

    /* Per-worker contexts. */
    struct worker_arg ws[MAX_WORKERS];
    pthread_t tids[MAX_WORKERS];
    rknn_input_output_num io_num;
    memset(&io_num, 0, sizeof(io_num));

    for (int i = 0; i < n_workers; i++) {
        memset(&ws[i], 0, sizeof(ws[i]));
        ws[i].idx = i;
        if (rknn_init(&ws[i].ctx, model_buf, (uint32_t)msz, 0, NULL) != 0) {
            fprintf(stderr, "worker[%d]: rknn_init failed\n", i);
            return 1;
        }
        uint32_t mask = strategy_pinned ? (1u << (i % 3)) : 0x7u;
        rknn_set_core_mask(ws[i].ctx, (rknn_core_mask)mask);
        if (i == 0)
            rknn_query(ws[i].ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
        ws[i].n_outputs = io_num.n_output;
    }

    struct queue in_q, out_q;
    queue_init(&in_q, 16);     /* small input queue keeps memory bounded */
    queue_init(&out_q, 64);

    /* Start workers. */
    for (int i = 0; i < n_workers; i++) {
        ws[i].in_q  = &in_q;
        ws[i].out_q = &out_q;
        pthread_create(&tids[i], NULL, worker_thread, &ws[i]);
    }

    /* Output thread. */
    struct output_arg oarg;
    memset(&oarg, 0, sizeof(oarg));
    oarg.out_q = &out_q;
    oarg.enc_pipe = enc;
    oarg.total_frames = N;
    oarg.workers_remaining = n_workers;
    pthread_t out_tid;
    pthread_create(&out_tid, NULL, output_thread, &oarg);

    /* Main loop: read raw RGB frames from decoder, push to in_q. */
    uint64_t t_start = now_ns();
    int frame_id = 0;
    while (frame_id < N) {
        struct frame_pkt *p = frame_alloc(W, H);
        size_t bytes = (size_t)W * H * 3;
        size_t got = fread(p->rgb, 1, bytes, dec);
        if (got != bytes) {
            frame_free(p);
            fprintf(stderr, "decoder EOF after %d frames\n", frame_id);
            break;
        }
        p->frame_id = frame_id++;
        queue_push(&in_q, p);
    }

    /* Send sentinels — one per worker. */
    for (int i = 0; i < n_workers; i++) {
        struct frame_pkt *s = calloc(1, sizeof(*s));
        s->sentinel = 1;
        queue_push(&in_q, s);
    }

    /* Join. */
    for (int i = 0; i < n_workers; i++) pthread_join(tids[i], NULL);
    pthread_join(out_tid, NULL);

    uint64_t t_end = now_ns();
    double wall_s = (double)(t_end - t_start) / 1e9;

    /* Stats. */
    uint64_t total_inf = 0;
    uint64_t total_lat = 0;
    for (int i = 0; i < n_workers; i++) {
        total_inf += ws[i].frames;
        total_lat += ws[i].total_lat_ns;
    }
    double end_to_end_fps = oarg.frames_written / wall_s;
    double inf_only_fps   = total_inf > 0 && total_lat > 0
        ? ((double)total_inf * n_workers) / ((double)total_lat / 1e9)
        : 0.0;
    double mean_lat_ms = total_inf > 0
        ? (double)total_lat / total_inf / 1e6
        : 0.0;

    fprintf(stderr,
            "\n=== demo_deeplabv3 results ===\n"
            "input:        %s (%dx%d, %d frames)\n"
            "output:       %s\n"
            "workers:      %d (%s)\n"
            "wall-clock:   %.2f s\n"
            "frames out:   %llu\n"
            "end-to-end:   %.1f FPS\n"
            "per-worker mean inference latency: %.2f ms\n"
            "aggregate inference throughput:    %.1f FPS\n"
            "==================================\n",
            in_path, W, H, N, out_path, n_workers,
            strategy_pinned ? "pinned" : "multicore",
            wall_s,
            (unsigned long long)oarg.frames_written,
            end_to_end_fps,
            mean_lat_ms,
            inf_only_fps);

    /* Cleanup. */
    for (int i = 0; i < n_workers; i++) rknn_destroy(ws[i].ctx);
    pclose(dec);
    pclose(enc);
    free(model_buf);
    return 0;
}
