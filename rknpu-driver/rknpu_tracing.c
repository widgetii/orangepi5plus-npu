#include <linux/module.h>

int rknpu_trace_enabled = 0;
module_param(rknpu_trace_enabled, int, 0644);
MODULE_PARM_DESC(trace_enabled, "Enable RKNPU register tracing via ftrace");

EXPORT_SYMBOL(rknpu_trace_enabled);
