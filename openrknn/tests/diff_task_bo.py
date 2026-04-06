#!/usr/bin/env python3
"""Compare our task BO binary with the proxy's, field by field.

Usage: python3 diff_task_bo.py /tmp/our_task_bo.bin /tmp/rknn_dump/sub1_bo_000_4096B.bin

Task struct (40 bytes):
  +0:  uint32 flags
  +4:  uint32 op_idx
  +8:  uint32 enable_mask
  +12: uint32 int_mask
  +16: uint32 int_clear
  +20: uint32 int_status
  +24: uint32 regcfg_amount
  +28: uint32 regcfg_offset
  +32: uint64 regcmd_addr
"""
import struct, sys

fields = ['flags', 'op_idx', 'enable_mask', 'int_mask', 'int_clear',
          'int_status', 'regcfg_amount', 'regcfg_offset', 'regcmd_addr']

def parse_tasks(data, max_tasks=20):
    tasks = []
    for t in range(min(max_tasks, len(data) // 40)):
        f = struct.unpack_from('<IIIIIIIIQ', data, t * 40)
        tasks.append(dict(zip(fields, f)))
    return tasks

our_data = open(sys.argv[1], 'rb').read()
prx_data = open(sys.argv[2], 'rb').read()

our_tasks = parse_tasks(our_data)
prx_tasks = parse_tasks(prx_data)

# For regcmd_addr: translate our BO addresses to proxy's for comparison
# (filled in by user based on actual DMA addresses)

n = min(len(our_tasks), len(prx_tasks))
match = 0; diff = 0
for t in range(n):
    for f in fields:
        ov = our_tasks[t][f]
        pv = prx_tasks[t][f]
        if f == 'regcmd_addr':
            # Different BO addresses, compare offset within BO instead
            # Our weight_bo DMA and proxy weight_bo DMA differ
            # Just show both
            if ov != pv:
                print(f'  task[{t}] {f:15s} our=0x{ov:016x} prx=0x{pv:016x} (different BOs)')
                diff += 1
            else:
                match += 1
        elif ov != pv:
            print(f'  task[{t}] {f:15s} our=0x{ov:08x} prx=0x{pv:08x}')
            diff += 1
        else:
            match += 1

print(f'\nTotal: {match} match, {diff} diff ({n} tasks)')
