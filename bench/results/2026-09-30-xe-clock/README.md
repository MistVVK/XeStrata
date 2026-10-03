# A device clock for the verify window's stage profiler — 2026-09-30

`STRATA_VERIFY_PROFILE` has the verify window write a timestamp between its stages (`gpu_stamp`). The stamps are recorded in the window's graph and written on every replay. CUDA read `%globaltimer` in a one-thread kernel. The port read SYCL's device-scope clock, and with the profile on, the first stamp threw during the window capture:

```
strata generate: verify: window capture: gpu_stamp: the device has no device-scope clock
```

These probes look for another way to write a device timestamp from a recorded graph on the B70:

| Probe | What it tries | Result ([output](clock_probe.txt), [output](gts_probe.txt), [output](tag_graph.txt)) |
| --- | --- | --- |
| [clock_probe.cpp](clock_probe.cpp) | the `sycl_ext_oneapi_clock` scopes | sub-group clock only: `clock_work_group 0 clock_device 0`. A sub-group clock is defined only within one sub-group, so stamps from different kernels cannot be compared |
| [gts_probe.cpp](gts_probe.cpp) | `zeCommandListAppendWriteGlobalTimestamp` through `ext_codeplay_enqueue_native_command`, eagerly and in a recorded graph | the first, eager submission fails with `UR_RESULT_ERROR_UNSUPPORTED_FEATURE`, under both Level Zero adapters (`SYCL_UR_USE_LEVEL_ZERO_V2=0` and `1`) |
| [tag_graph.cpp](tag_graph.cpp) | `submit_profiling_tag` while recording a graph | "Profiling information is unavailable for events returned from a submission to a queue in the recording state." |

None of them gives a stamp in a replayed graph. The verify window now checks for the device-scope clock when it starts (`gpu_stamp_available`). Without one, it prints that the stage profile is off and captures the window as usual. The stage profile is therefore not available on the B70.
SYCL does not define the device clock's rate, so where the clock exists the report is in millions of ticks, not milliseconds.

Build: `icpx -fsycl -O2 PROBE.cpp` (add `-lze_loader` for `gts_probe.cpp`).
