<!--
SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
SPDX-License-Identifier: LGPL-3.0-or-later
-->
# The products without XMX through DP4a — 2026-10-02

Without the matrix engines (a GPU or a SYCL runtime that reports none, as intel/llvm 6.2's for the B70: [the free build's record](../2026-10-02-free-build/README.md)), the prompt path's products ran in FP32 on the vector engines at 6 TFLOP/s, and the prompt path was 3.1 times slower than with XMX.
They now quantize both sides to int8 in blocks of 32 values with a float scale each while the tiles load (W as q8_0, X as q8_1 without its sum) and multiply with DP4a (`src/kernels/xe/dp4a_gemm.cpp`).

- Machine and runs as in [the free build's record](../2026-10-02-free-build/README.md) ([free_ab.sh](../2026-10-02-free-build/free_ab.sh): icpx with XMX, the same binary with `STRATA_NO_XMX=1`, and the free build with dpclang 6.2, which has no XMX for the B70), at `cd0e784` with this change. Logs in [runs/](runs/)
- The kernel timings come from scratch probes (event profiling), not kept in the repository

## The kernel

FP16 inputs, B70, TOPS:

| Shape (T x N x K) | FP32 (before) | int8 copies made beforehand, product alone | Quantized while loading (now) |
| --- | --- | --- | --- |
| 8192 x 10240 x 2560 (attn_qkv) | 5.9 | 23.5 (128 x 64 tiles) | 18.7 |
| 8192 x 2560 x 6144 (ssm_out) | 6.0 | 22.4 | 19.9 |
| 8192 x 320 x 10240 (hc down) | 6.1 | 14.4 | 18.8 (64 x 64) |
| 160 x 1280 x 2560 (an expert's gate/up) | 3.2 | 10.3 (64 x 64) | 10.1 (64 x 64) |

- Quantizing while loading needs no int8 copies in memory, and the copies' own quantization (0.7–2.4 ms for X at these shapes) is not in the middle column
- The first version read the input type from a run-time flag in the inner loop (BF16 or FP16) and ran at 6 TOPS; as a template parameter it runs at the speed above
- 128 x 128 tiles (8 x 8 outputs a work-item) spill and run at 0.3 TOPS
- Short products stay slow: 128 x 320 x 10240 at 0.36 TOPS, 128 x 2560 x 2560 at 2.3 (few tiles, a long K loop); split-K would help short prompts

### Checks

Against the same arithmetic on the CPU (blocks of 32, scale = largest magnitude / 127, round, integer dot, times the scales), every output agrees except where a value lands exactly on a half (−63.5: the CPU's `std::round` gives −64, the GPU −63); one such value changes the whole column it belongs to, so the largest difference over a 64 x 64 x 2560 product was 0.05 on outputs of about 50.
Against FP64 of the FP16 inputs the relative error is 0.71–0.85% (the quantization).

The same probe on the integrated UHD 770 (Xe-LP, 32 compute units, i915 driver), at shapes small enough for i915's time limit (a product of seconds was reset): its outputs are bit-identical to the B70's, at 0.22–0.37 TOPS.
The whole engine does not start on it: pinning the 322 MiB embedding table in host memory fails (`runs/igpu-short.txt`), which is not about these products.

## The engine

| Build | Prompt, 26,292 tokens (tok/s) | Decode with MTP (tok/s) |
| --- | --- | --- |
| icpx, XMX | 1015 / 1017 | 59.1 / 59.1 |
| icpx, `STRATA_NO_XMX=1`: DP4a | 639 / 637 | 59.0 / 59.0 |
| free, dpclang 6.2: DP4a | 686 / 649 | 47.9 / 59.3 |
| (before) icpx, `STRATA_NO_XMX=1`: FP32 | 324 / 324 | — |

Without XMX the prompt path is now 1.5–1.6 times slower than with it, not 3.1. The free build's first decode run compiled the MTP kernels.
The time breakdown without XMX (`STRATA_PREFILL_TIMING`, [FP32](runs/nx-timing-fp32.txt) against an earlier DP4a build, [DP4a](runs/nx-timing.txt)) put the experts' products at 26 s and the dense projections at about 36 s of 89 before.

### The output

The two DP4a builds (icpx and dpclang) generate the same tokens.
Against the XMX build, the long prompt's 24 tokens agree to token 12; there XMX chose token 846 by 2.25 over 248045, and DP4a chose 248045 by 0.88. The logits of the first 12 rows differ by at most 2.3–2.6, as much as the FP32 path's and intel/llvm 7.1.1's differ from icpx's, but those kept every token: the int8 quantization moves the answer more than another compiler's rounding does.
Quantizing the i-quant experts' weights again to int8 adds to llama.cpp's own error (its MMQ keeps the weights' codes); a native MMQ would remove that part.
