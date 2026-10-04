<!--
SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
SPDX-License-Identifier: LGPL-3.0-or-later
-->
# The prompt path's matrix products without oneMKL — 2026-10-02

The prompt path ran its projections through oneMKL's `gemm`, which is not free software, so the engine could not go into Debian main.
They now run on XeStrata's own XMX kernels (`src/kernels/xe/xmx_gemm.cpp`, SYCL `joint_matrix`), and the engine no longer links oneMKL (`ldd`).
The routed experts, which oneMKL multiplied one call per expert, are multiplied G at a time in one launch (`--prefill-experts G`, default 16).

- Machine and runs as in [the new machine's record](../2026-10-02-new-machine/README.md): `build/xe-portable`, IQ3_S, the 26,292-token prompt, `--prefill auto --kv int8 --max-context 32768 --expert-cache 10000 --suffix-draft 0`, 4 tokens generated; [xmx_ab.sh](xmx_ab.sh) alternates the oneMKL build (`a6c8eea`) and this one. Logs in [runs/](runs/), paths written as `<repo>`, `<data>`, `<record>`
- The kernel timings come from scratch probes (event profiling, or host time for back-to-back launches), not kept in the repository

## Result

| Build | Prompt (tok/s), two alternating rounds | Decode, 4 tokens (tok/s) |
| --- | --- | --- |
| oneMKL (`a6c8eea`) | 1053.5 / 1051.9 | 19.4 / 19.4 |
| XMX kernels, G = 16 | 1020.5 / 1015.7 | 19.2 / 19.0 |

The prompt path is 3.3% slower than with oneMKL; the user chose to take the change, so the engine is free enough for Debian main, and win the speed back afterwards.

- Output: after the 26,292-token prompt the 24 generated tokens are the same as oneMKL's (`xmx-chk-*`); the logits differ by at most 3.1 (a different order of the FP32 sums over 26k tokens). Two runs of the new build give identical logits
- CTest (31 tests; `expert_multi_test` skipped without AVX-512), `iq_parity` and `native_expert_parity` on the IQ3_S file pass
- VRAM: the prompt path's region grows by 16 x 9.4 MiB of dequantized experts less the 2 the old ring held and oneMKL's 32 MiB workspace, about +118 MiB, lent by the expert cache

## The kernel

A sub-group computes a 32 x 64 tile of Y as 4 x 4 accumulators of 8 x 16, K advances 32 at a time, and the work-group's next A and B tiles are prefetched into the cache, split among its sub-groups (the scheme of intel/llvm's `joint_matrix_bf16_fill_k_cache` test).
Large register file (`grf_size<256>`), 16-wide sub-groups.

Dense products, 8192 rows, FP16, µs (TFLOP/s):

| Shape (N x K) | oneMKL | First version | Work-groups in strips of 2 row tiles |
| --- | --- | --- | --- |
| 10240 x 2560 (attn_qkv) | 2703 (159) | 4086 (105) | 2835-2954 (145-152) |
| 2560 x 6144 (ssm_out) | 1546 (167) | 1643 (157) | 1583-1658 (155-163) |
| 12288 x 2560 (attn_q) | 3264 (158) | 4896 (105) | 3604-3675 (140-143) |

- The strips: work-groups numbered so the ones running at once share their A and their B tiles in the cache. Without them a wide product reads W once per row tile and runs at 105 TFLOP/s
- W is the B operand read column-major straight from its rows. The VNNI-packed layout the XMX engines take natively ran at the same speed (142-147 against 143-153 TFLOP/s for the wide products; 61 against 56 for the experts' gate/up), and dequantizers writing the packed layout were 2-3 times slower than row-major ones (local-memory staging, a lane per row, 4 rows a lane: 20-35 µs against 11 µs an expert's gate/up), so nothing is packed
- The bounds-checked column-major load lost the device; the B tiles past the last row of W are read from the last 16 rows instead and their columns not stored
- A 32 x 64 tile of `joint_matrix` 32 x 64 (one accumulator) spilled and ran at 1-3 TFLOP/s; a K step of 64 cannot be prefetched (64 bytes a row at most); prefetch distance 1-2 steps is best (3-4 slower)
- The kernels are functors whose properties set the register file. A functor wrapping the kernel lambda passed the captures as one by-value struct (`arg_byvalue` in the kernel's info) instead of global pointers, with the same machine code, and ran 8-30% slower (ssm_out 2.45 ms against 1.72-1.77)
- The work-group tile depends on the shape: narrow products (the router, the indexer, the hyper-connection's down projection N 320 K 10240) take narrow tiles or one covering all their columns, so X is read once (hc down 1162 → 650 µs, oneMKL 604)
- N under 16 (the hyper-connection's inject N 4 K 10240, the shared expert's gate N 1) goes to a plain kernel, a sub-group a row computing all its outputs (1.7 ms a chunk when a sub-group took one output)
- Below about 1024 rows a product takes at least 40 µs (the K loop's latency): alpha (N 48 K 2560) at 512 rows 38 µs against oneMKL's 6. Split-K would help short prompts; not done

## The experts

One layer's products as the prompt path runs them (512 experts of 80-240 rows; gate/up N 1280 K 2560, then down N 2560 K 640), host-timed back to back:

| | ms a layer | TFLOP/s |
| --- | --- | --- |
| oneMKL, one call per expert | 16.2 | 49.7 |
| Grouped, G = 1 | 31.8 | 25.3 |
| G = 4 | 16.6 | 48.4 |
| G = 8 | 12.9 | 62.3 |
| G = 16 | 10.6 | 76.1 |
| G = 32 | 9.4 | 85.4 |

In the engine the gain mostly goes: the old path dequantized an expert into one of two 9.4 MiB slots and multiplied it at once, while the weights were still in the L2 cache; G experts' dequantized weights do not fit, so they go to VRAM and are read back.
The prompt rate by G (one run each, before the last kernel changes): G = 2 999, G = 8 1013, G = 16 1017 tok/s (`xmx-g*`).
With the dense products on oneMKL and the experts grouped (a temporary build, `x-hyb-*`): 1026 against 998 with both on the XMX kernels.
The next step is a kernel that dequantizes inside the product, so the dequantized weights are never written.

`STRATA_PREFILL_TIMING` does not compare the two paths: its event between the products of each expert cost oneMKL's path (76k experts) about 2 s that the grouped path does not pay, so the breakdown showed the experts' products at 5.9 s against 3.1 while the runs without it differ the other way.
