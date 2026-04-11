/*
 * bench_throughput — pure-C NPU throughput benchmark for openrknn vs
 * vendor librknnrt.so. Spawns N pthreads, each with its own
 * rknn_context, barriers them, measures aggregate FPS and per-request
 * latency distribution over a fixed wall-clock window.
 *
 * No Python. No ctypes. The only per-call overhead above the RKNN
 * runtime itself is a clock_gettime()+comparison+array write per
 * iteration, so the measurement ceiling is close to the real hardware
 * limit.
 *
 * Usage:
 *   bench_throughput --model PATH [options]
 *
 * Options:
 *   --model PATH           path to .rknn file (required)
 *   --workers N            number of concurrent worker threads (default 1)
 *   --duration SECONDS     measurement window duration (default 5)
 *   --warmup N             per-worker warmup iterations before barrier
 *                          (default 20)
 *   --strategy NAME        pinned | multicore (default multicore)
 *                            pinned:    worker i uses core_mask 0x1 << (i%3)
 *                            multicore: all workers use 0x7 (all cores)
 *   --json                 emit JSON instead of a markdown row
 *   --label STR            free-form label included in output (e.g. "vendor")
 *
 * Library selection: default SONAME resolution picks /lib/librknnrt.so.
 * Override with LD_PRELOAD=/path/to/openrknn/librknn_api.so and
 * ORKNN_OWN=init,query,input,run,outputs for the openrknn OWN path.
 *
 * Exit codes:
 *   0 — success
 *   1 — argument / init failure
 *   2 — watchdog tripped (a worker didn't respond within duration + 30s)
 *
 * SPDX-License-Identifier: MIT
 */
#define _GNU_SOURCE
#include "rknn_api.h"
#include <pthread.h>
#include <time.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <errno.h>

/* Per-worker latency array size cap. At 400 kFPS × 5 s = 2 M
 * inferences — far above anything we'll measure. 8 MB/worker, 64 MB
 * total at N=8 on a 16 GB board. */
#define MAX_LAT_PER_WORKER (2u * 1024u * 1024u)

/* Maximum number of worker threads the harness supports at once. */
#define MAX_WORKERS 16

struct worker {
    int                 idx;
    rknn_context        ctx;
    uint32_t            core_mask;
    void               *input_buf;    /* zero-filled, owned by main */
    uint32_t            input_size;
    rknn_tensor_type    input_type;
    rknn_tensor_format  input_fmt;
    uint32_t            n_outputs;
    uint32_t            warmup_iters;

    /* Shared: absolute wall-clock end of the measurement window. */
    uint64_t            t_end_ns;
    pthread_barrier_t  *ready_barrier;   /* workers signal "warmup done" */
    pthread_barrier_t  *go_barrier;      /* main releases workers after
                                          * writing t_end_ns */

    /* Results */
    uint64_t            count;
    uint32_t            n_lat;
    uint32_t           *lat_ns;
};

static uint64_t now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}

static void *worker_thread(void *arg) {
    struct worker *w = arg;

    /* Bind input once. rknn_inputs_set copies into DMA; later iters
     * only need rknn_run + rknn_outputs_get/release. */
    rknn_input in;
    memset(&in, 0, sizeof(in));
    in.index = 0;
    in.buf = w->input_buf;
    in.size = w->input_size;
    in.pass_through = 0;
    in.type = w->input_type;
    in.fmt = w->input_fmt;
    int ret = rknn_inputs_set(w->ctx, 1, &in);
    if (ret != 0) {
        fprintf(stderr, "worker[%d]: rknn_inputs_set failed: %d\n", w->idx, ret);
        return NULL;
    }

    /* Pre-build the output slots for rknn_outputs_get. want_float=0
     * keeps the hot path at "NPU + DMA copy" without adding dequant
     * cost. Each iteration calls outputs_get + outputs_release with
     * these. */
    rknn_output outs[16];
    if (w->n_outputs > 16) {
        fprintf(stderr, "worker[%d]: too many outputs (%u)\n",
                w->idx, w->n_outputs);
        return NULL;
    }
    memset(outs, 0, sizeof(outs));
    for (uint32_t i = 0; i < w->n_outputs; i++) {
        outs[i].index = i;
        outs[i].want_float = 0;
        outs[i].is_prealloc = 0;
    }

    /* Warmup: amortize openrknn first-run patching and vendor caching
     * BEFORE we hit the start barrier. */
    for (uint32_t i = 0; i < w->warmup_iters; i++) {
        if (rknn_run(w->ctx, NULL) != 0) {
            fprintf(stderr, "worker[%d]: warmup rknn_run failed\n", w->idx);
            return NULL;
        }
        if (rknn_outputs_get(w->ctx, w->n_outputs, outs, NULL) != 0) {
            fprintf(stderr, "worker[%d]: warmup outputs_get failed\n", w->idx);
            return NULL;
        }
        rknn_outputs_release(w->ctx, w->n_outputs, outs);
    }

    /* Two-phase synchronized start:
     *   ready_barrier: worker signals warmup is done. Main waits for
     *                  all workers here, then writes t_end_ns.
     *   go_barrier:    main releases all workers simultaneously, with
     *                  t_end_ns guaranteed-set before any worker sees
     *                  it. Avoids the race where a worker would read
     *                  t_end_ns==0 and exit immediately. */
    pthread_barrier_wait(w->ready_barrier);
    pthread_barrier_wait(w->go_barrier);

    while (1) {
        uint64_t t0 = now_ns();
        if (t0 >= w->t_end_ns) break;

        if (rknn_run(w->ctx, NULL) != 0) {
            fprintf(stderr, "worker[%d]: rknn_run failed after %llu iters\n",
                    w->idx, (unsigned long long)w->count);
            break;
        }
        if (rknn_outputs_get(w->ctx, w->n_outputs, outs, NULL) != 0) {
            fprintf(stderr, "worker[%d]: outputs_get failed after %llu iters\n",
                    w->idx, (unsigned long long)w->count);
            break;
        }
        uint64_t t1 = now_ns();
        rknn_outputs_release(w->ctx, w->n_outputs, outs);

        uint64_t dt = t1 - t0;
        if (w->n_lat < MAX_LAT_PER_WORKER)
            w->lat_ns[w->n_lat++] = (dt > UINT32_MAX) ? UINT32_MAX : (uint32_t)dt;
        w->count++;
    }
    return NULL;
}

static int cmp_u32(const void *a, const void *b) {
    uint32_t x = *(const uint32_t *)a, y = *(const uint32_t *)b;
    return (x > y) - (x < y);
}

static void usage(const char *argv0) {
    fprintf(stderr,
        "Usage: %s --model PATH [options]\n"
        "  --model PATH          .rknn file (required)\n"
        "  --workers N           concurrent workers (default 1, max %d)\n"
        "  --duration SECONDS    measurement window (default 5)\n"
        "  --warmup N            warmup iters per worker (default 20)\n"
        "  --strategy NAME       pinned | multicore (default multicore)\n"
        "  --json                emit JSON instead of markdown row\n"
        "  --label STR           free-form label\n",
        argv0, MAX_WORKERS);
}

int main(int argc, char **argv) {
    const char *model_path = NULL;
    int workers = 1;
    double duration_sec = 5.0;
    uint32_t warmup = 20;
    const char *strategy = "multicore";
    int emit_json = 0;
    const char *label = "";

    for (int i = 1; i < argc; i++) {
        const char *a = argv[i];
        if (!strcmp(a, "--model") && i + 1 < argc)      model_path = argv[++i];
        else if (!strcmp(a, "--workers") && i + 1 < argc) workers = atoi(argv[++i]);
        else if (!strcmp(a, "--duration") && i + 1 < argc) duration_sec = atof(argv[++i]);
        else if (!strcmp(a, "--warmup") && i + 1 < argc) warmup = (uint32_t)atoi(argv[++i]);
        else if (!strcmp(a, "--strategy") && i + 1 < argc) strategy = argv[++i];
        else if (!strcmp(a, "--json"))                   emit_json = 1;
        else if (!strcmp(a, "--label") && i + 1 < argc) label = argv[++i];
        else if (!strcmp(a, "-h") || !strcmp(a, "--help")) { usage(argv[0]); return 0; }
        else { fprintf(stderr, "unknown arg: %s\n", a); usage(argv[0]); return 1; }
    }
    if (!model_path) { usage(argv[0]); return 1; }
    if (workers < 1 || workers > MAX_WORKERS) {
        fprintf(stderr, "workers must be 1..%d\n", MAX_WORKERS);
        return 1;
    }

    int is_pinned = !strcmp(strategy, "pinned");
    int is_multi  = !strcmp(strategy, "multicore");
    if (!is_pinned && !is_multi) {
        fprintf(stderr, "strategy must be 'pinned' or 'multicore'\n");
        return 1;
    }

    /* Load model file once into memory; each worker's rknn_init gets
     * its own copy via the rknn_init(data, size) API which copies
     * into a per-context buffer. */
    FILE *f = fopen(model_path, "rb");
    if (!f) { perror("fopen model"); return 1; }
    fseek(f, 0, SEEK_END);
    uint32_t model_size = (uint32_t)ftell(f);
    fseek(f, 0, SEEK_SET);
    void *model_data = malloc(model_size);
    if (!model_data) { fclose(f); return 1; }
    if (fread(model_data, 1, model_size, f) != model_size) {
        fprintf(stderr, "fread: truncated read\n");
        free(model_data); fclose(f); return 1;
    }
    fclose(f);

    /* Per-worker state. */
    struct worker *ws = calloc((size_t)workers, sizeof(*ws));
    pthread_t *tids = calloc((size_t)workers, sizeof(*tids));
    if (!ws || !tids) { free(model_data); return 1; }

    pthread_barrier_t ready_barrier, go_barrier;
    /* +1 because main thread participates in both barriers. */
    pthread_barrier_init(&ready_barrier, NULL, (unsigned)workers + 1);
    pthread_barrier_init(&go_barrier,    NULL, (unsigned)workers + 1);

    rknn_input_output_num io_num;
    rknn_tensor_attr in_attr;
    rknn_tensor_attr out_attrs[16];
    memset(&in_attr, 0, sizeof(in_attr));
    memset(out_attrs, 0, sizeof(out_attrs));

    /* Initialise every worker's context. Each rknn_init loads a fresh
     * copy of the model — matches how a real inference server scales
     * per-client. */
    for (int w = 0; w < workers; w++) {
        int ret = rknn_init(&ws[w].ctx, model_data, model_size, 0, NULL);
        if (ret != 0) {
            fprintf(stderr, "worker[%d]: rknn_init failed: %d\n", w, ret);
            return 1;
        }

        uint32_t mask = is_pinned ? (1u << (w % 3)) : 0x7u;
        ws[w].core_mask = mask;
        if (rknn_set_core_mask(ws[w].ctx, (rknn_core_mask)mask) != 0)
            fprintf(stderr, "worker[%d]: rknn_set_core_mask(0x%x) failed\n",
                    w, mask);

        /* Query once on worker 0 to size the input/output buffers —
         * they're identical across workers since they all share the
         * same model. */
        if (w == 0) {
            rknn_query(ws[w].ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
            in_attr.index = 0;
            rknn_query(ws[w].ctx, RKNN_QUERY_INPUT_ATTR, &in_attr, sizeof(in_attr));
            for (uint32_t o = 0; o < io_num.n_output && o < 16; o++) {
                out_attrs[o].index = o;
                rknn_query(ws[w].ctx, RKNN_QUERY_OUTPUT_ATTR,
                           &out_attrs[o], sizeof(out_attrs[o]));
            }
        }

        ws[w].idx           = w;
        ws[w].input_size    = in_attr.size;
        ws[w].input_type    = in_attr.type;
        ws[w].input_fmt     = in_attr.fmt;
        ws[w].n_outputs     = io_num.n_output;
        ws[w].warmup_iters  = warmup;
        ws[w].ready_barrier = &ready_barrier;
        ws[w].go_barrier    = &go_barrier;
        ws[w].lat_ns        = malloc(MAX_LAT_PER_WORKER * sizeof(uint32_t));
        if (!ws[w].lat_ns) {
            fprintf(stderr, "worker[%d]: out of memory for lat buffer\n", w);
            return 1;
        }
    }

    /* Shared zero input buffer. All workers reference the same bytes
     * — rknn_inputs_set copies into each context's own DMA buffer, so
     * there's no contention. */
    uint8_t *input_buf = calloc(1, in_attr.size ? in_attr.size : 1);
    for (int w = 0; w < workers; w++) ws[w].input_buf = input_buf;

    /* Spawn worker threads — they will warm up, then block at the
     * barrier waiting for us. */
    for (int w = 0; w < workers; w++) {
        if (pthread_create(&tids[w], NULL, worker_thread, &ws[w]) != 0) {
            fprintf(stderr, "pthread_create failed for worker %d\n", w);
            return 1;
        }
    }

    /* Two-phase start. ready_barrier: wait for all workers to finish
     * warmup. At that point every worker is blocked on go_barrier, so
     * writing t_end_ns is race-free. go_barrier: release everyone
     * simultaneously with the deadline already set. */
    pthread_barrier_wait(&ready_barrier);
    uint64_t t_window_start = now_ns();
    uint64_t deadline = t_window_start + (uint64_t)(duration_sec * 1e9);
    for (int w = 0; w < workers; w++) ws[w].t_end_ns = deadline;
    pthread_barrier_wait(&go_barrier);

    /* Wait for workers to finish. Simple join — the deadline
     * mechanism gives each worker an internal exit condition, so
     * they'll return within a few milliseconds of `deadline`. No
     * watchdog in v1; dmesg inspection catches NPU hangs externally. */
    for (int w = 0; w < workers; w++) pthread_join(tids[w], NULL);
    uint64_t t_window_end = now_ns();

    double elapsed = (double)(t_window_end - t_window_start) / 1e9;

    /* Aggregate. */
    uint64_t total_count = 0;
    uint64_t total_lat = 0;
    for (int w = 0; w < workers; w++) {
        total_count += ws[w].count;
        total_lat   += ws[w].n_lat;
    }
    double fps = elapsed > 0 ? (double)total_count / elapsed : 0.0;

    /* Merge per-worker latencies into one sorted buffer and pick
     * percentiles. Allocate max-worst-case size. */
    uint32_t *all_lat = NULL;
    uint64_t n_all = 0;
    double p50 = 0, p95 = 0, p99 = 0;
    if (total_lat > 0) {
        all_lat = malloc(total_lat * sizeof(uint32_t));
        if (all_lat) {
            for (int w = 0; w < workers; w++) {
                memcpy(all_lat + n_all, ws[w].lat_ns,
                       ws[w].n_lat * sizeof(uint32_t));
                n_all += ws[w].n_lat;
            }
            qsort(all_lat, n_all, sizeof(uint32_t), cmp_u32);
            p50 = (double)all_lat[(n_all * 50) / 100] / 1e6;
            p95 = (double)all_lat[(n_all * 95) / 100] / 1e6;
            p99 = (double)all_lat[(n_all * 99) / 100] / 1e6;
        }
    }

    if (emit_json) {
        printf("{\"label\":\"%s\",\"model\":\"%s\",\"strategy\":\"%s\","
               "\"workers\":%d,\"duration_s\":%.3f,\"count\":%llu,"
               "\"fps\":%.2f,\"p50_ms\":%.3f,\"p95_ms\":%.3f,\"p99_ms\":%.3f}\n",
               label, model_path, strategy, workers, elapsed,
               (unsigned long long)total_count, fps, p50, p95, p99);
    } else {
        printf("| %s | %s | %d | %.3fs | %llu | %.1f | %.3f | %.3f | %.3f |\n",
               label[0] ? label : strategy, strategy, workers, elapsed,
               (unsigned long long)total_count, fps, p50, p95, p99);
    }

    /* Cleanup. */
    if (all_lat) free(all_lat);
    for (int w = 0; w < workers; w++) {
        free(ws[w].lat_ns);
        rknn_destroy(ws[w].ctx);
    }
    pthread_barrier_destroy(&ready_barrier);
    pthread_barrier_destroy(&go_barrier);
    free(input_buf);
    free(ws);
    free(tids);
    free(model_data);
    return 0;
}
