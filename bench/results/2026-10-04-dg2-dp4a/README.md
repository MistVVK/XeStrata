<!--
SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
SPDX-License-Identifier: LGPL-3.0-or-later
-->
# The prompt path's products on the Arc A series (Xe-HPG) — 2026-10-04

On the Arc A380 the prompt path's products through DP4a (`src/kernels/xe/dp4a_gemm.cpp`) took 2 seconds for a 4096 x 2560 x 2560 product, which the B70 does in 2.6 ms.
The cause was the 128 x 64 tile (8 x 4 outputs a work-item): its 64 accumulators a work-item spilled on the 8-wide EUs of Xe-HPG.
The 128 x 64 tile is now taken only where the GPU reports an EU SIMD width of 16 or more (`ext_intel_gpu_eu_simd_width`: 16 on the B70, 8 on the A380 and the UHD 770), and the A380 takes the 64 x 64 tile, 100 times faster.

- Machine: an Arc A380 (6 GB, 128 compute units) with a Ryzen 5 3600 on Debian 13 (Linux 6.12), the engine's code in a throwaway incus container of Ubuntu 26.04 (Intel's compute-runtime 26.05 from Ubuntu, the A380 passed through), built on the development machine with intel/llvm 7.1.1
- The timings come from scratch probes (`xmx_gemm` with `STRATA_NO_XMX=1`, the best of 3 after a warm-up, host-timed), not kept in the repository

## The tile and the sub-group size

FP16, TOPS:

| Shape (T x N x K) | Tile | B70, 16-wide sub-groups | B70, 32-wide | A380, 16-wide | A380, 32-wide |
| --- | --- | --- | --- | --- | --- |
| 4096 x 2560 x 2560 | 128 x 64 | 19.5 | 0.70 | 0.026 | 0.024 |
| 512 x 2560 x 2560 | 128 x 64 | 14.7 | 11.4 | 0.027 | 0.024 |
| 256 x 2560 x 2560 | 64 x 64 | 10.9 | 14.4 | 2.50 | 0.026 |
| 64 x 2560 x 2560 | 64 x 64 | 6.8 | 4.8 | 1.86 | 0.028 |

- The tile follows the row count: 128 x 64 from `fill_rows(1024)` rows on (1,024 on the B70, 512 on the A380), 64 x 64 below
- `STRATA_SUB_GROUP_32=1` takes 32-wide sub-groups where 16 is listed too; on the A380 every shape spilled at 32, and on the B70 the 128 x 64 tile did (the 64 x 64 one did not)
- The engine takes 16 where the GPU lists it (`narrow_sub_group`), so the A380 needed only the tile changed

## With the change

| Shape | B70 | A380 |
| --- | --- | --- |
| 4096 x 2560 x 2560 | 3.6–5.7 ms (128 x 64, as before) | 2,033 ms → 20.1 ms (2.67 TOPS) |
| 512 x 2560 x 2560 | 0.6–1.0 ms | 248 ms → 2.7 ms |

On the B70 the code it runs is the same; alternating runs of the build before and after this change (intel/llvm 7.1.1) spread over 3.6–5.8 ms either way.

## Against mma_gemm on the A380

`mma_gemm` (`src/kernels/xe/mma_gemm.cpp`, joint_matrix's portable API) ran the A380's matrix engines in their 8 x 8 x 16 shape at about 1 TFLOP/s, against 2.4–2.7 TOPS for DP4a with this change (two alternating rounds, the same within 3%):

| Product | mma_gemm (8 x 8 x 16, 8 x 8 sub-groups) | DP4a |
| --- | --- | --- |
| 4096 x 2560 x 2560 | 52 ms | 20 ms |
| 4096 x 10240 x 2560 | 217 ms | 87–90 ms |
| 4096 x 2560 x 6144 | 130 ms | 54–56 ms |
| 512 x 2560 x 2560 | 6.7 ms | 2.8 ms |
| 16 experts of 160 rows, 1280 x 2560 | 21.7 ms | 8.2 ms |
| 16 experts of 160 rows, 2560 x 640 | 10.3 ms | 3.9 ms |

DP4a quantizes both sides to int8 and is less exact (relative error against FP64 0.53%, against 0.0001% for `mma_gemm`), but was faster then, so `mma_gemm`'s 8 x 8 x 16 shape was taken out until it ran faster.
Its work-group size mattered (2 x 2 sub-groups 125 ms, 4 x 4 79 ms, 8 x 8 53 ms for the first product), and 4 x 4 tiles a sub-group spilled (99 ms).

## mma_gemm made faster

`mma_gemm` read W (B, column-major) straight from global memory in every sub-group, and copied only X into local memory.
Each step below was measured on the 4096 x 2560 x 2560 product, the outputs the same bits throughout:

| Step | A380 |
| --- | --- |
| As above (W from global memory, 8 x 8 sub-groups) | 52 ms |
| W into local memory too, column-major, 16 bytes a work-item | 116 ms |
| W transposed into local memory (B row-major) | 33 ms |
| W in Intel's VNNI-packed layout in local memory | 23 ms |
| The packed stores of neighbouring lanes side by side | 21 ms |
| Two buffers (the next step loaded while this one is multiplied) | 20 ms |
| 4 x 4 tiles a sub-group in the large register file (4 x 8 sub-groups) | 16 ms |

- K steps of 64 instead of 32 were no faster (23.0 ms against 23.2)
- Tiles beyond 2 x 2 a sub-group spilled in the default 128 registers (832–1,696 bytes in IGC's assembly, 84–97 ms): an 8 x 8 FP32 accumulator takes 8 of Xe-HPG's 32-byte registers.
  In the large register file (`grf_size<256>`, which the A380 builds) 4 x 4 tiles fit: with 4 x 8 sub-groups 16.1 ms, 4 x 4 17.0, 8 x 4 16.7, 4 x 2 24.8, 2 x 4 23.4; 8 x 8 sub-groups do not launch with it
- In the assembly of the 2 x 2 tiles, the four `dpas.8x8` a K step of 16 came with 32 loads from local memory of 32 bytes each; 4 x 4 tiles share each load among twice the products
- The experts' groups of 160 rows take a shorter work-group tile (2 x 8 sub-groups of 2 x 2 tiles, 32 rows) than a product: 7.0 ms for 16 experts of 1280 x 2560, against 9.2 with 8 x 8 sub-groups; none of the large-register layouts was faster for them (7.2–15 ms)

Against DP4a, two alternating rounds (ms):

| Product | mma_gemm | DP4a |
| --- | --- | --- |
| 4096 x 2560 x 2560 | 16.0–16.3 | 20.0–20.1 |
| 4096 x 10240 x 2560 | 66.2–68.6 | 88.9–89.5 |
| 4096 x 2560 x 6144 | 37.6–37.7 | 53.7 |
| 512 x 2560 x 2560 | 2.1 | 2.9 |
| 16 experts of 160 rows, 1280 x 2560 | 7.0–7.1 | 8.0–8.3 |
| 16 experts of 160 rows, 2560 x 640 | 3.5 | 3.9 |

So the Arc A series takes `mma_gemm` again: 20–30% faster than DP4a for a product, 10–15% for the experts, and exact to 0.0001%.
A GPU that does not build the large register file takes the groups' layout for a product too (checked on the A380 by forcing it: the same outputs).
The same code on the B70 with `STRATA_MMA=1` (Xe2's 8 x 16 x 16, untuned 2 x 2 sub-groups) ran 15–24 T/s against 9–15 before (XMX's own kernels 120–170, so the B70 keeps them),
and on an RTX 4070 (16 x 16 x 16, untuned) 16–27 T/s with a relative error of 0.0004%.

## Not checked

- The whole engine on the A380 (a model loaded, an answer)
- The other Arc A cards; their EU is 8 wide as the A380's (Xe-HPG), which is what the choice reads
