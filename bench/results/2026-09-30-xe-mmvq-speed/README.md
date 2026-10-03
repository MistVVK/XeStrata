# MMVQ speed on the B70 — 2026-09-30

Decode-side matrix-vector products at the IQ2_XS model's shapes, after the numerical port ([i-quant results](../2026-09-30-xe-iq/README.md)).
This is the baseline the stage-5 optimization starts from, not a tuned result.
The B70 is on PCIe 4.0 x8, which does not affect these on-device timings.

## Memory bandwidth reference

[bandwidth_dp4a.cpp](bandwidth_dp4a.cpp) reads 1 GiB of device memory filled with xorshift data at **590 GB/s** ([output](bandwidth-and-dp4a.txt); first repeat 548 GB/s).
Filled with a constant, the same read and a device-to-device copy reported 1.4 TB/s and 1.9 TB/s, above the card's memory interface: the GPU compresses such data, so any Xe bandwidth benchmark must use incompressible contents.
The MMVQ runs below use random weight bytes.

## dp4a

The same probe times a dependent chain of 4096 integer dot products per work-item.
The byte loop that first stood in for CUDA's `__dp4a` ran at about 1,230 G dot/s; SPIR-V's `SDotKHR` (SPV_KHR_integer_dot_product) at about 5,180 G dot/s, 4.2 times faster, with the saturating form at the same rate.
`src/kernels/xe/cuda_intrinsics.hpp` now uses the non-saturating form, matching `__dp4a`; `iq_parity` still reports 0 failures and the random-block MMVQ check printed identical values with either implementation.

## MMVQ

[mmvq_speed.cpp](mmvq_speed.cpp) times back-to-back launches on the in-order compute queue from the host, so launch cost is included as decode pays it; each figure is the median of five loops (20 calls for the output head, 400 otherwise) after 10 warm-up calls.
The byte-loop column is one run of the earlier build ([output](byte-loop-dp4a.txt)); the SPIR-V column is three runs ([1](spirv-dp4a-1.txt), [2](spirv-dp4a-2.txt), [3](spirv-dp4a-3.txt)).
GB/s counts weight bytes only.

| Shape | ncols | MB | byte-loop µs | SPIR-V dp4a µs (median of 3 runs, range) | GB/s | share of 590 GB/s |
| --- | --- | --- | --- | --- | --- | --- |
| output head        IQ4_XS | 1 | 337.72 | 2346.8 | 2124.9 (2124.0–2125.3) | 158.9 | 27% |
| output head        IQ4_XS | 4 | 337.72 | 3774.6 | 3271.1 (3270.6–3271.4) | 103.2 | 17% |
| attn_qkv           IQ4_XS | 1 | 13.93 | 107.9 | 99.2 (99.1–99.2) | 140.4 | 24% |
| attn_qkv           IQ4_XS | 4 | 13.93 | 171.3 | 141.2 (141.1–141.2) | 98.7 | 17% |
| ssm_out            IQ4_XS | 1 | 8.36 | 58.5 | 51.8 (51.8–51.8) | 161.4 | 27% |
| ssm_out            IQ4_XS | 4 | 8.36 | 149.2 | 60.9 (60.9–60.9) | 137.3 | 23% |
| attn_gate          IQ3_S | 1 | 6.76 | 45.6 | 42.3 (42.3–42.4) | 159.8 | 27% |
| attn_gate          IQ3_S | 4 | 6.76 | 166.9 | 155.5 (155.5–155.6) | 43.5 | 7% |
| shexp gate         IQ3_S | 1 | 0.70 | 6.7 | 6.6 (6.5–6.6) | 106.1 | 18% |
| shexp gate         IQ3_S | 4 | 0.70 | 22.1 | 20.8 (20.8–20.8) | 33.7 | 6% |
| shexp up           IQ4_XS | 1 | 0.87 | 11.5 | 10.6 (10.6–10.7) | 82.1 | 14% |
| shexp up           IQ4_XS | 4 | 0.87 | 21.6 | 18.8 (18.8–18.8) | 46.3 | 8% |
| shexp down         IQ4_NL | 1 | 0.92 | 21.4 | 20.1 (20.1–20.1) | 45.8 | 8% |
| shexp down         IQ4_NL | 4 | 0.92 | 33.0 | 27.5 (27.5–27.5) | 33.5 | 6% |
| expert gate        IQ2_S | 1 | 0.52 | 6.1 | 5.5 (5.5–5.5) | 94.5 | 16% |
| expert gate        IQ2_S | 4 | 0.52 | 19.7 | 17.1 (17.1–17.1) | 30.4 | 5% |
| expert gate        IQ2_XXS | 1 | 0.42 | 6.8 | 6.0 (6.0–6.0) | 70.0 | 12% |
| expert gate        IQ2_XXS | 4 | 0.42 | 22.0 | 18.5 (18.5–18.5) | 22.7 | 4% |
| expert gate        IQ1_M | 1 | 0.36 | 4.3 | 3.4 (3.4–3.5) | 105.9 | 18% |
| expert gate        IQ1_M | 4 | 0.36 | 11.2 | 7.2 (7.2–7.2) | 50.0 | 8% |
| expert down        Q2_0 | 1 | 0.46 | 19.4 | 17.5 (17.5–17.5) | 26.3 | 4% |
| expert down        Q2_0 | 4 | 0.46 | 32.1 | 26.1 (26.1–26.1) | 17.6 | 3% |

The large matrices reach about 27% of the read bandwidth, and the per-expert matrices (under 1 MB) take 3.4–26 µs per call, so work-group layout rather than arithmetic now bounds them.
The layout is still CUDA's: one 128-lane work-group per row (four rows when K is small), and in the small-K expert down projection (K = 640) only 20 of 128 lanes of a row have blocks to read.
Changing it belongs to stage 5 and must keep each column bitwise equal to a single-column call.
