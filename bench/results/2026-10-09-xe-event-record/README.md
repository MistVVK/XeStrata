<!--
SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
SPDX-License-Identifier: LGPL-3.0-or-later
-->
# Event records as timestamps or barriers — 2026-10-09

On the B70, upstream's SYCL port read a 4K prompt faster than XeStrata (1,418 against 1,215 tok/s, [the 0.1.40 record](../2026-10-09-upstream-0140/README.md#against-upstreams-sycl-port-sycl-engine-0140-sycl)).
The prompt path's timing showed the GPU idle at each layer's routing sync: about 19 ms a layer with the default streamed ring (384 slots), 3 ms with `STRATA_PREFILL_RING=16`.
Timestamps taken around the sync (the events' `command_submit`, the host's submission in the device's clock) put the time between the GPU finishing the routing read-back and the host submitting the next work at 905 ms for the 48 layers with 384 slots and 151 ms with 16; the host spent it in `urEventWait`, polling in the driver.

`gpu::event_record` marked a queue's end with `submit_profiling_tag`, a timestamp.
The prompt path records one for each streamed expert (its copy done, its slot used) and holds the events of the whole ring.
On the B70 each timestamp whose event is still held makes the next wait take about 65 us longer; a barrier (`ext_oneapi_submit_barrier`) costs nothing there.

## Probe

[event_record_probe.cpp](event_record_probe.cpp): 384 rounds of a single-task kernel and a mark on an in-order queue, the marks' events held until after the wait or dropped at once, then a barrier's wait for the queue; medians of 5.
Built with intel/llvm (the contrib-llvm compiler, `-fsycl-targets=spir64,nvidia_gpu_sm_89`; for the RX 9060 XT the free build's, `amdgcn-amd-amdhsa --offload-arch=gfx1200`).

| Device (backend) | Mark | Events | Submit | Wait |
| --- | --- | --- | --- | --- |
| B70 (Level Zero v2) | timestamp | held | 10.27 ms | 31.50 ms |
| | timestamp | dropped | 5.17 ms | 0.09 ms |
| | barrier | held | 0.90 ms | 0.29 ms |
| | barrier | dropped | 0.82 ms | 0.06 ms |
| RTX 4070 (CUDA) | timestamp | held | 1.48 ms | 1.81 ms |
| | timestamp | dropped | 1.54 ms | 1.75 ms |
| | barrier | held | 0.81 ms | 0.00 ms |
| | barrier | dropped | 0.82 ms | 0.00 ms |
| RX 9060 XT (HIP) | timestamp | held | 1.77 ms | 4.09 ms |
| | timestamp | dropped | 1.81 ms | 4.04 ms |
| | barrier | held | 1.45 ms | 4.42 ms |
| | barrier | dropped | 1.47 ms | 4.39 ms |

On the RX 9060 XT the wait is the 384 kernels themselves, whichever the mark.

## The choice

Events used for timing (`event_elapsed_ms`: the prompt and stage timings, the CPU share's measurement, the PCIe probe) are created with `event_create(&e, true)` and stay timestamps.
Every other event is a barrier where the runtime measured held timestamps to slow its waits: `Runtime` times the wait after 64 held timestamps and after 64 barriers on a queue of its own when it is made, and takes barriers when the timestamps cost more than 20 us each (`STRATA_EVENT_MARK=tag` or `barrier` overrides).
At start-up the 64 held timestamps' wait was 6,084 us on the B70 (barriers 2 us) and 169 us on the RTX 4070 (barriers 1 us): barriers on the B70, timestamps on the RTX 4070.

## The engine

IQ2_XS, `--expert-cache auto --spec 4 --spec-min-p 0.5 --mtp --max-context 8192 --kv int8 --greedy --prefill auto --vram-reserve-mib 700`, the contrib-llvm build; 4,095 random token ids (as the 0.1.40 record) with 16 out, and an 18-token chat with 256 out.
Three interleaved rounds; the first run of a new binary builds its kernels (JIT) and is left out.

| B70 | Before | After |
| --- | --- | --- |
| Prompt, 4,094 tokens | 1249.9, 1241.7 tok/s | 1598.2, 1597.4 tok/s |
| Time to the first token | 4.20, 4.22 s | 3.49, 3.49 s |
| Decode, 256 tokens | 67.77, 67.62 tok/s | 68.37, 68.35 tok/s |

The tokens are the same before and after.
Against upstream's SYCL port on the same card the prompt is now 1,598 against 1,418 tok/s and the first token 3.49 against 5.59 s.

On the RTX 4070 barriers read the prompt 1% slower than timestamps (759.3, 759.7, 759.5 tok/s with timestamps; 750.1, 756.3, 748.9 with barriers; 758.3, 752.0, 742.8 with the event of the queue's last command, `ext_oneapi_get_last_event`), the same tokens; with the measured choice it keeps timestamps (760.0 tok/s).
Its decode varies with the drafts accepted from run to run (the answers differ between runs of the same binary) and was not compared.

## RX 9060 XT

In the AMD container (Ryzen 7 5700X, ROCm 7.1.1), the free build, Coder IQ1_M (`--expert-cache auto --prefill auto --spec 4 --spec-min-p 0.5 --mtp`, 8K context), 1,016- and 3,561-token prompts with 256 greedy tokens out, two interleaved rounds; before is `b111f001`.
The probe gives the timestamps no cost there, so the runtime should keep them (its choice was not printed); the engine reads the same as with the choice forced to timestamps (`STRATA_EVENT_MARK=tag`):

| | Prompt, 1K | Prompt, 4K | Decode, 4K |
| --- | --- | --- | --- |
| Before | 708.7, 702.5 tok/s | 749.2, 747.4 tok/s | 47.82, 47.77 tok/s |
| After | 702.4, 703.0 tok/s | 749.1, 748.4 tok/s | 47.81, 47.80 tok/s |
| After, `STRATA_EVENT_MARK=tag` | 704.0, 702.7 tok/s | 749.0, 748.7 tok/s | 47.82, 47.76 tok/s |

The 4K answers are the same in all six runs.
The 1K answers differ from run to run in all three, before as well: a chunk below 2,048 tokens hands some experts to the CPU in the share it measures (`STRATA_PREFILL_CPU_SHARE=auto`), and the CPU rounds differently.
