#ifndef _RKNPU_TRACE_H
#define _RKNPU_TRACE_H

extern int rknpu_trace_enabled;

static inline int rknpu_trace_init(void)
{
    rknpu_trace_enabled = 0;
    pr_info("rknpu_trace: initialized (echo 1 > /sys/module/rknpu_trace/parameters/trace_enabled)\n");
    return 0;
}

static inline void rknpu_trace_exit(void)
{
    pr_info("rknpu_trace: removed\n");
}

#endif
