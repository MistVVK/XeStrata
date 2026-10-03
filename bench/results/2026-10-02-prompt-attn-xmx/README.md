<!--
SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
SPDX-License-Identifier: LGPL-3.0-or-later
-->
# The prompt attention on the matrix engines — 2026-10-02

The prompt path's attention over the selected cells (`qsa_prompt_attn_batch`, Strata's tensor-core kernel on CUDA) ran on Xe through its FP32 fallback.
It now has a port to joint_matrix (`src/kernels/xe/qsa_prompt_attn.cpp`): 8 x 16 x 16 FP16 tiles with FP32 accumulation, q and p as FP16 hi + lo pairs so the products keep about FP32's precision, an online softmax over chunks of 16 cells.
A device or runtime without XMX for FP16 refuses it and keeps the FP32 kernel.

- Machine as in [the free build's record](../2026-10-02-free-build/README.md): Intel Arc Pro B70 (PCIe Gen5 x16), icpx 2026.1, at `adc4bf6` with this change
- The kernel numbers come from `qsa_prompt_attn_parity` (ported from the CUDA build in this change), the engine's from [pa_ab.sh](pa_ab.sh) (the scratch `run.sh` it calls is not kept): IQ3_S, the 26,292-token prompt, int8 KV, 24 tokens, two alternating rounds after a warm-up. Logs in [runs/](runs/)

## The kernel

`qsa_prompt_attn_parity`: ms per chunk of queries, the FP32 kernel against the new one; both against FP64 of the same inputs.

| Case | FP32 kernel (ms) | XMX (ms) | Speed-up | vs FP64 as the test prints it, FP32 / XMX (outputs up to about 3.6) |
| --- | --- | --- | --- | --- |
| int8 KV, 32,768 cells, 2,048 queries | 36.6 | 32.8 | 1.12x | 2.2e-6 / 3.2e-6 |
| fp16 KV, 32,768 cells, 2,048 queries | 32.9 | 24.2 | 1.36x | 1.2e-6 / 2.9e-6 |
| int8 KV, 1,500 cells, 1,500 queries | 12.2 | 9.0 | 1.35x | 1.6e-6 / 2.3e-6 |
| int8 KV, 2,100 cells, 256 queries | 5.9 | 5.2 | 1.12x | 1.7e-6 / 3.3e-6 |

Repeated runs move the speed-ups by about 0.1 (the first measurement of the 2,100-cell case was 0.97x).

What it took, and what did not help:

- **The hi + lo split.** Written as `x - (float) half(x)`, the compiler took the round trip for exact and the low half came out zero: the products kept FP16's 11 bits and the outputs were off by 1e-3. The high half now goes back to FP32 through its bits (`f32_from_f16`)
- **Scaling the accumulators.** `joint_matrix_apply` with each element's row and column cost more than all the chunk's products. An 8 x 16 accumulator gives each lane one column with element i in row i, so the rescale goes by element order (the parity test would catch a different layout), and it is skipped when no row's maximum moved: a row's reference maximum moves only when a score passes it by more than 8 (in log2), as in FlashAttention-3, and V's unit only when a larger V scale arrives
- **Chunks of 32 cells** kept twice the local memory and ran at half the speed; 16 stays
- **K and V packed for the B operand (VNNI) in local memory** were slower than loading them row- and column-major

## The engine

| Build | Prompt, 26,292 tokens (tok/s) |
| --- | --- |
| FP32 prompt attention | 1012.8 / 1018.3 |
| XMX prompt attention | 1046.7 / 1049.2 |

3.2% faster. Decode does not use this kernel (23.7–23.9 tok/s in both, without MTP).

### The output

Each build repeats its own logits bit for bit across runs.
The two builds generate the same tokens up to token 17 of 24: there the FP32 build chose 7248 by 0.42 and the XMX build chose 248048 by 0.13.
Before that the logits differ by 0.39–0.87, and 1.5–3.5 on rows 11–14; the DP4a and intel/llvm builds differ from icpx's by up to 2.4–2.7 ([the DP4a record](../2026-10-02-dp4a/README.md)).
Both kernels are within 3.3e-6 of FP64 on outputs up to about 3.6, far below these differences, so they come from the order of the sums carried through the layers, not from one kernel being the less accurate (`unverified` beyond this test's inputs).
