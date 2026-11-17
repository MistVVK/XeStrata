# A/B harnesses and calculators for single investigations (10 to 30 October 2026)

| File | Date | What it was for |
| --- | --- | --- |
| `h2d_bench.cu` | 10 Oct | What a stream of expert-sized host-to-device copies costs end to end (issue, copy, host function, sync). `nvcc -O2 -o h2d_bench research/tools/ab-harnesses-2026-10/h2d_bench.cu`. |
| `quant_bound.py` | 12 Oct | Decode bounds of a quant from its real per-layer formats (the output of `tools/quant_formats.py --json`). |
| `idle_gpu_ab.py` | 20 Oct | Task 2507-0: how idle are the GPUs, and does an existing switch close the gap? Interleaved A/B of whole engine runs with a per-card utilisation sampler. Measurement only. |
| `arc/ab.py`, `arc/b.py`, `arc/kl.py` | 30 Oct | Interleaved A/B of engine arms on the Intel Arc test box, one engine load per arm per round; and KL between first-token logit dumps. `ab.py` expects the repository copied to `<work dir>/src` on that box. |

Maintained Arc tooling is in [sycl/](../../../../engine/sycl) and [docs/INTEL_ARC.md](../../../../docs/INTEL_ARC.md).
