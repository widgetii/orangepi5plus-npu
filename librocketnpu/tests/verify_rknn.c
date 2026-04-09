/*
 * verify_rknn.c — Step-by-step comparison: RKNN golden vs librocketnpu
 *
 * Loads golden data from /tmp/verify_golden/ (captured by verify_intercept.so),
 * runs librocketnpu on the same model, compares at each step.
 *
 * Usage:
 *   ./verify_rknn conv_int8.tflite /tmp/verify_golden
 *
 * Build:
 *   gcc -o verify_rknn tests/verify_rknn.c -Iinclude -Isrc -L. -lrocketnpu -ldl -Wl,-rpath,.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <dirent.h>
#include <dlfcn.h>
#include "rocketnpu.h"
#include "rnpu_internal.h"
#include "rnpu_drm.h"

/* rknpu_task struct (40 bytes, matches kernel definition) */
struct rknpu_task {
    uint32_t flags;
    uint32_t op_idx;
    uint32_t enable_mask;
    uint32_t int_mask;
    uint32_t int_clear;
    uint32_t int_status;
    uint32_t regcfg_amount;
    uint32_t regcfg_offset;
    uint64_t regcmd_addr;
};

/* ---- Golden data structures ---- */

#define MAX_GOLDEN_BOS 64

struct golden_bo {
    uint64_t dma;
    uint64_t size;
    uint32_t flags;
    uint8_t *pre_submit;   /* BO content before submit */
    uint8_t *post_submit;  /* BO content after submit */
};

struct golden {
    struct golden_bo bos[MAX_GOLDEN_BOS];
    int bo_count;
    char dir[256];
};

/* ---- Utility ---- */

static uint8_t *load_file(const char *path, size_t *out_size) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    uint8_t *buf = malloc(sz);
    fread(buf, 1, sz, f);
    fclose(f);
    if (out_size) *out_size = sz;
    return buf;
}

static int compare_bytes(const char *label, const uint8_t *a, const uint8_t *b,
                         size_t size, int verbose) {
    int diffs = 0;
    int first_diff = -1;
    for (size_t i = 0; i < size; i++) {
        if (a[i] != b[i]) {
            if (first_diff < 0) first_diff = i;
            diffs++;
        }
    }
    if (diffs == 0) {
        printf("  %s: %zu bytes IDENTICAL\n", label, size);
    } else {
        printf("  %s: %d/%zu bytes DIFFER (%.1f%% match)\n",
               label, diffs, size, 100.0 * (size - diffs) / size);
        if (verbose && first_diff >= 0) {
            printf("    first diff at offset %d (0x%x):\n", first_diff, first_diff);
            int ctx_start = first_diff > 8 ? first_diff - 8 : 0;
            int ctx_end = first_diff + 24 < (int)size ? first_diff + 24 : (int)size;
            printf("      golden:");
            for (int i = ctx_start; i < ctx_end; i++)
                printf(" %02x", a[i]);
            printf("\n      ours:  ");
            for (int i = ctx_start; i < ctx_end; i++)
                printf(" %02x", b[i]);
            printf("\n");
        }
    }
    return diffs;
}

/* Compare regcmd entries with DMA normalization */
static int compare_regcmd(const uint8_t *golden_bo, const uint8_t *our_bo,
                          uint32_t rc_offset, uint32_t rc_amount,
                          uint64_t golden_bases[], uint64_t our_bases[],
                          uint64_t bo_sizes[], int n_bos, int task_idx) {
    int diffs = 0;
    uint64_t *g_entries = (uint64_t *)(golden_bo + rc_offset);
    uint64_t *o_entries = (uint64_t *)(our_bo + rc_offset);

    for (unsigned e = 0; e < rc_amount + 4; e++) {
        uint16_t g_reg = g_entries[e] & 0xFFFF;
        uint32_t g_val = (g_entries[e] >> 16) & 0xFFFFFFFF;
        uint16_t o_reg = o_entries[e] & 0xFFFF;
        uint32_t o_val = (o_entries[e] >> 16) & 0xFFFFFFFF;

        if (g_reg != o_reg) {
            printf("    T%d [%3u] REG MISMATCH: golden=0x%04x ours=0x%04x\n",
                   task_idx, e, g_reg, o_reg);
            diffs++;
            continue;
        }

        if (g_val == o_val) continue;

        /* Normalize DMA addresses */
        uint32_t g_norm = g_val, o_norm = o_val;
        for (int b = 0; b < n_bos; b++) {
            uint32_t g_off = (g_val - (uint32_t)golden_bases[b]) & 0xFFFFFFFF;
            uint32_t o_off = (o_val - (uint32_t)our_bases[b]) & 0xFFFFFFFF;
            if (g_off < bo_sizes[b]) { g_norm = g_off; break; }
            if (o_off < bo_sizes[b]) { o_norm = o_off; break; }
        }

        if (g_norm != o_norm) {
            printf("    T%d [%3u] 0x%04x: golden=0x%08x ours=0x%08x (norm: %u vs %u)\n",
                   task_idx, e, g_reg, g_val, o_val, g_norm, o_norm);
            diffs++;
        }
    }
    return diffs;
}

/* ---- Load golden data ---- */

static int load_golden(struct golden *g, const char *dir) {
    memset(g, 0, sizeof(*g));
    strncpy(g->dir, dir, sizeof(g->dir) - 1);

    /* Parse meta.txt for BO info */
    char path[512];
    snprintf(path, sizeof(path), "%s/meta.txt", dir);
    FILE *f = fopen(path, "r");
    if (!f) { fprintf(stderr, "Cannot open %s\n", path); return -1; }

    char line[512];
    while (fgets(line, sizeof(line), f)) {
        if (strncmp(line, "bo_count=", 9) == 0)
            g->bo_count = atoi(line + 9);
        int idx;
        unsigned handle;
        unsigned long long dma, obj, sz;
        unsigned flags;
        if (sscanf(line, "bo[%d] handle=%u dma=0x%llx obj=0x%llx size=%llu flags=0x%x",
                   &idx, &handle, &dma, &obj, &sz, &flags) == 6 && idx < MAX_GOLDEN_BOS) {
            g->bos[idx].dma = dma;
            g->bos[idx].size = sz;
            g->bos[idx].flags = flags;
        }
    }
    fclose(f);

    /* Load pre/post submit BO dumps */
    for (int i = 0; i < g->bo_count; i++) {
        snprintf(path, sizeof(path), "%s/pre_submit_1_bo_%03d_%lluB.bin",
                 dir, i, (unsigned long long)g->bos[i].size);
        g->bos[i].pre_submit = load_file(path, NULL);

        snprintf(path, sizeof(path), "%s/post_submit_1_bo_%03d_%lluB.bin",
                 dir, i, (unsigned long long)g->bos[i].size);
        g->bos[i].post_submit = load_file(path, NULL);
    }

    printf("Loaded golden: %d BOs from %s\n", g->bo_count, dir);
    return 0;
}

/* ---- Main ---- */

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s model.tflite /tmp/verify_golden\n", argv[0]);
        return 1;
    }
    const char *tflite_path = argv[1];
    const char *golden_dir = argv[2];

    /* Generate deterministic input (LCG, matches compare_rknn_c.c) */
    uint8_t input[3072];
    uint32_t s = 42;
    for (int i = 0; i < 3072; i++) {
        s = s * 1103515245 + 12345;
        input[i] = (s >> 16) & 0xFF;
    }

    /* Load golden data */
    struct golden golden;
    if (load_golden(&golden, golden_dir) != 0) return 1;

    /* Load model via librocketnpu (uses .rknn_cache) */
    int fd = rnpu_open(NULL);
    rnpu_model_t *m = rnpu_model_load(fd, tflite_path);
    if (!m) { fprintf(stderr, "Model load failed\n"); return 1; }

    printf("\n=== Step 1: Weight BO comparison ===\n");
    /* Identify golden BOs by DMA address match (IOMMU deterministic) */
    int g_wt = -1, g_act = -1;
    for (int i = 0; i < golden.bo_count; i++) {
        if (golden.bos[i].dma == m->weight_bo.dma_addr) g_wt = i;
        else if (m->activation_bo.handle && golden.bos[i].dma == m->activation_bo.dma_addr) g_act = i;
    }
    if (g_wt < 0) g_wt = 1; /* fallback */
    if (g_act < 0) g_act = 2;
    printf("  Golden BO mapping: weight=BO[%d](dma=0x%llx) activation=BO[%d](dma=0x%llx)\n",
           g_wt, (unsigned long long)(g_wt >= 0 ? golden.bos[g_wt].dma : 0),
           g_act, (unsigned long long)(g_act >= 0 ? golden.bos[g_act].dma : 0));

    if (g_wt >= 0 && golden.bos[g_wt].pre_submit && m->weight_bo.map) {
        size_t wt_sz = golden.bos[g_wt].size < m->weight_bo.size ?
                       golden.bos[g_wt].size : m->weight_bo.size;
        compare_bytes("weight+regcmd BO", golden.bos[g_wt].pre_submit,
                      m->weight_bo.map, wt_sz, 1);
    }

    printf("\n=== Step 2: Task descriptors ===\n");
    if (m->native_task_bo_data) {
        /* Find golden task BO */
        int g_task = -1;
        for (int i = 0; i < golden.bo_count; i++)
            if (golden.bos[i].flags & 0x8) { g_task = i; break; }

        if (g_task >= 0 && golden.bos[g_task].pre_submit) {
            struct rknpu_task *g_tasks = (struct rknpu_task *)golden.bos[g_task].pre_submit;
            struct rknpu_task *o_tasks = (struct rknpu_task *)m->native_task_bo_data;
            int n_tasks = m->native_task_bo_size / 40;
            int g_n_tasks = golden.bos[g_task].size / 40;
            if (n_tasks > g_n_tasks) n_tasks = g_n_tasks;
            printf("  Task count: golden=%d ours=%d\n", g_n_tasks, n_tasks);

            for (int t = 0; t < n_tasks; t++) {
                int match = (g_tasks[t].op_idx == o_tasks[t].op_idx &&
                             g_tasks[t].enable_mask == o_tasks[t].enable_mask &&
                             g_tasks[t].regcfg_amount == o_tasks[t].regcfg_amount);
                uint32_t g_rc_off = (uint32_t)(g_tasks[t].regcmd_addr - golden.bos[g_wt].dma);
                uint32_t o_rc_off = (uint32_t)(o_tasks[t].regcmd_addr - m->weight_bo.dma_addr);
                if (g_rc_off != o_rc_off) match = 0;

                printf("  T%d: op=%u en=0x%x amt=%u rc_off=0x%x %s",
                       t, o_tasks[t].op_idx, o_tasks[t].enable_mask,
                       o_tasks[t].regcfg_amount, o_rc_off,
                       match ? "MATCH" : "DIFFER");
                if (!match)
                    printf(" (golden: op=%u en=0x%x amt=%u rc_off=0x%x)",
                           g_tasks[t].op_idx, g_tasks[t].enable_mask,
                           g_tasks[t].regcfg_amount, g_rc_off);
                printf("\n");
            }
        }
    }

    printf("\n=== Step 3: Regcmd entry comparison ===\n");
    if (g_wt >= 0 && golden.bos[g_wt].pre_submit && m->weight_bo.map &&
        m->native_task_bo_data) {
        struct rknpu_task *g_tasks = NULL;
        for (int i = 0; i < golden.bo_count; i++)
            if (golden.bos[i].flags & 0x8) {
                g_tasks = (struct rknpu_task *)golden.bos[i].pre_submit;
                break;
            }

        struct rknpu_task *o_tasks = (struct rknpu_task *)m->native_task_bo_data;
        int n_tasks = m->native_task_bo_size / 40;

        /* Build DMA base arrays for normalization */
        uint64_t g_bases[MAX_GOLDEN_BOS], o_bases[MAX_GOLDEN_BOS], sizes[MAX_GOLDEN_BOS];
        /* Our BOs: weight, activation, input, output(s) */
        int n_bos = 0;
        g_bases[n_bos] = golden.bos[g_wt].dma; o_bases[n_bos] = m->weight_bo.dma_addr;
        sizes[n_bos] = golden.bos[g_wt].size; n_bos++;
        if (g_act >= 0) {
            g_bases[n_bos] = golden.bos[g_act].dma; o_bases[n_bos] = m->activation_bo.dma_addr;
            sizes[n_bos] = golden.bos[g_act].size; n_bos++;
        }
        if (m->native_input_bo.handle) {
            /* Find golden input BO by size match */
            for (int i = 0; i < golden.bo_count; i++) {
                if ((golden.bos[i].flags & 0x8) || i == g_wt || i == g_act) continue;
                if (golden.bos[i].size == m->native_input_bo.size) {
                    g_bases[n_bos] = golden.bos[i].dma;
                    o_bases[n_bos] = m->native_input_bo.dma_addr;
                    sizes[n_bos] = golden.bos[i].size;
                    n_bos++;
                    break;
                }
            }
        }
        for (unsigned oi = 0; oi < m->native_output_bo_count; oi++) {
            for (int i = 0; i < golden.bo_count; i++) {
                if ((golden.bos[i].flags & 0x8) || i == g_wt || i == g_act) continue;
                if (golden.bos[i].size == m->native_output_bos[oi].size) {
                    g_bases[n_bos] = golden.bos[i].dma;
                    o_bases[n_bos] = m->native_output_bos[oi].dma_addr;
                    sizes[n_bos] = golden.bos[i].size;
                    n_bos++;
                    break;
                }
            }
        }

        int total_diffs = 0;
        for (int t = 0; t < n_tasks && g_tasks; t++) {
            uint32_t g_rc_off = (uint32_t)(g_tasks[t].regcmd_addr - golden.bos[g_wt].dma);
            uint32_t rc_amt = g_tasks[t].regcfg_amount;
            int d = compare_regcmd(golden.bos[g_wt].pre_submit, m->weight_bo.map,
                                   g_rc_off, rc_amt, g_bases, o_bases, sizes, n_bos, t);
            total_diffs += d;
        }
        printf("  Total regcmd diffs: %d\n", total_diffs);
    }

    printf("\n=== Step 4: Input + Submit + Post-inference comparison ===\n");
    rnpu_invoke(m, input, 3072);

    /* Compare post-submit BOs */
    for (int i = 0; i < golden.bo_count; i++) {
        if (!golden.bos[i].post_submit) continue;

        /* Find our matching BO */
        uint8_t *our_map = NULL;
        uint32_t our_size = 0;
        const char *label = "unknown";

        if (i == g_wt && m->weight_bo.map) {
            our_map = m->weight_bo.map; our_size = m->weight_bo.size; label = "weight";
        } else if (i == g_act && m->activation_bo.map) {
            rnpu_bo_prep(fd, &m->activation_bo);
            our_map = m->activation_bo.map; our_size = m->activation_bo.size; label = "activation";
        } else {
            /* Check output BOs */
            for (unsigned oi = 0; oi < m->native_output_bo_count; oi++) {
                if (golden.bos[i].size == m->native_output_bos[oi].size) {
                    rnpu_bo_prep(fd, &m->native_output_bos[oi]);
                    our_map = m->native_output_bos[oi].map;
                    our_size = m->native_output_bos[oi].size;
                    label = "output";
                    break;
                }
            }
            if (!our_map && m->native_input_bo.handle &&
                golden.bos[i].size == m->native_input_bo.size) {
                our_map = m->native_input_bo.map;
                our_size = m->native_input_bo.size;
                label = "input";
            }
        }

        if (our_map) {
            size_t cmp_sz = golden.bos[i].size < our_size ?
                            golden.bos[i].size : our_size;
            char buf[64];
            snprintf(buf, sizeof(buf), "post_submit BO[%d] (%s)", i, label);
            compare_bytes(buf, golden.bos[i].post_submit, our_map, cmp_sz, 1);
        }
    }

    printf("\n=== Step 5: Output comparison ===\n");
    uint8_t raw_out[65536];
    int raw_sz = rnpu_get_output_raw(m, 0, raw_out, sizeof(raw_out));
    if (raw_sz > 0) {
        /* Find golden output BO (smallest non-task, non-weight, non-activation, non-input) */
        for (int i = 0; i < golden.bo_count; i++) {
            if (!golden.bos[i].post_submit) continue;
            if (i == g_wt || i == g_act) continue;
            if (golden.bos[i].flags & 0x8) continue;
            if ((int)golden.bos[i].size == raw_sz) {
                compare_bytes("raw output BO", golden.bos[i].post_submit, raw_out, raw_sz, 1);
                /* Also show first 8 values as int8 */
                int8_t *gi = (int8_t *)golden.bos[i].post_submit;
                int8_t *oi = (int8_t *)raw_out;
                printf("    golden first8: [%d,%d,%d,%d,%d,%d,%d,%d]\n",
                       gi[0],gi[1],gi[2],gi[3],gi[4],gi[5],gi[6],gi[7]);
                printf("    ours   first8: [%d,%d,%d,%d,%d,%d,%d,%d]\n",
                       oi[0],oi[1],oi[2],oi[3],oi[4],oi[5],oi[6],oi[7]);
                break;
            }
        }
    }

    printf("\n=== Done ===\n");
    return 0;
}
