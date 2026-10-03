# Host/GPU doorbell on the B70 — 2026-09-30

The engine overlaps the CPU expert pool with GPU work through a doorbell.
A kernel publishes the routing to host memory and rings a flag there.
The host polls the flag, runs the pool, writes the result and a second flag, and a spinning kernel waits for that flag.
The port specification (section 7.2) asks for the visibility and progress of this pattern to be shown on Xe before it is carried over.

| Probe | What it checks | Result |
| --- | --- | --- |
| [doorbell_probe.cpp](doorbell_probe.cpp) ([output](doorbell_probe.txt)) | A kernel stores a host-USM ring with a system-scope atomic and spins on a host-USM answer. The host polls the ring while the kernel runs, writes a 256-float payload, then the answer. A following kernel reads the payload. Tested by direct launch and by SYCL graph replay | 2,000 direct round trips and 2,000 graph replays, no payload error. Launch to ring seen: median 7.0 µs (direct), 8.6 µs (graph). Answer to done: median 5.7 µs |
| [doorbell_probe2.cpp](doorbell_probe2.cpp) ([output](doorbell_probe2.txt)) | The ways a spinning kernel can observe the host's answer | Mode 1, a volatile load followed by a system-scope acquire fence: 2,000/2,000, worst 33.5 µs. Mode 3, a device-USM flag written by a host-to-device copy on a second queue: 2,000/2,000, about 560 µs each |
| [doorbell_kernels_check.cpp](doorbell_kernels_check.cpp) ([output](doorbell_kernels_check.txt)) | The library's `doorbell_publish`, `doorbell_wait`, `copy_from_mapped` and `copy_i32_from_mapped` through the session's protocol, on an engine stream, directly and replayed from a SYCL graph | 400 rounds (200 direct, 200 graph replays): payload and answer correct every time |

Two readings do not work and must not be used for a GPU-side wait:

- A system-scope `atomic_ref` load alone (mode 0) saw the answer in some runs and spun forever in others, including the first version of `doorbell_probe.cpp`.
- A shared-USM flag (mode 2) was never observed.

A spinning kernel was not stopped by the xe driver's job timeout (5 s) during these hangs. The ported `doorbell_wait` spins without a bound, as the CUDA one does; a host that never answers is reported by the runtime's 120 s watchdog at the next host wait.
The GPU-to-host direction (a system-scope atomic store the host polls with a volatile load) was observed in every run.

Build: `icpx -fsycl -O2 -std=c++20 <probe>.cpp`; `doorbell_probe2` takes the mode as its argument.
