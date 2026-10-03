# Device allocations past 4 GiB on the B70 — 2026-09-30

The first run with `--expert-cache auto` allocated one 27 GiB device buffer for the expert slots. Zeroing it with a single memset never finished, and the runtime's watchdog stopped the process after 120 s.

[large_alloc_probe.cpp](large_alloc_probe.cpp) ([output](large_alloc_probe.txt)) allocates N GiB of device USM, then runs one of three operations on it:

- one memset (mode 0)
- memsets of 1 GiB each (mode 1)
- a device-to-device memcpy into a second allocation (mode 2)

A kernel then writes the last byte, and the probe reads it back.

| Case | Result |
| --- | --- |
| 3 GiB, one memset | 7.5 ms |
| 5 GiB, one memset | does not complete (stopped by the 30 s timeout) |
| 5 GiB and 27 GiB, memsets of 1 GiB | 8.8 ms and 42.9 ms |
| 5 GiB, one memcpy | completes |
| a kernel writing the last byte of 5 GiB and 27 GiB buffers | reads back correctly |

`max_mem_alloc_size` is 30.30 GiB, and allocations of that size succeed. Only a single memset of more than 4 GiB fails.
`strata::gpu::memset` and `memset_async` (src/core/gpu.cpp) therefore issue 1 GiB pieces, as does `DeviceArena`'s poison fill.
`UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS=1` did not change the outcome of the 5 GiB memset.

Build: `icpx -fsycl -O2 large_alloc_probe.cpp`; arguments `GiB mode`.
