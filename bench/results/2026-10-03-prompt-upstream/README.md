<!--
SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
SPDX-License-Identifier: LGPL-3.0-or-later
-->
# Prompt: upstream's changes after 0.1.24 and the fused expert kernel on the B70 — 2026-10-03

Upstream Strata's prompt-path changes after the fork point (0.1.24) ported to SYCL, and XeStrata's fused expert kernel, measured on the B70.
Machine as in [the new machine's record](../2026-10-02-new-machine/README.md); IQ3_S, `--expert-cache 10000`, greedy.
The long prompt: 26,292 tokens, `--prefill auto` (chunks of 8192, 8192, 8192, 1716), `--kv int8`, 24 tokens generated, logits dumped.
Each comparison alternates, two rounds after a warm-up; logs in [runs/](runs/).

## Kept: 1,039 to 1,081 tok/s (+4.0%)

| Build | Round 1 (tok/s) | Round 2 |
| --- | --- | --- |
| before (stage C) | 1038.49 | 1040.30 |
| F-1 and F-2 | 1054.98 | 1052.59 |
| F-1, F-2 and the pipelined GDN recurrence | 1081.05 | 1081.56 |

[i_ab.sh](i_ab.sh); "F-1 and F-2" is the last row's binary with `STRATA_GDN_PIPELINE=0`.
The logits are bitwise the same in all six runs.

- **F-1 and F-2 (upstream 882bb6d, b046845), +1.4%.**
  The hyper-connection read no longer writes the normalized rows in FP32: the norm writes each row's scale (`gr_norm_rs`) and the mix recomputes the rows from R (`gr_mix_r`).
  The hyper-connection write also computes the next half's norm (`gr_write_norm_rs`) when nothing else touches R in between: not after a stage's last half, not before layer 1's PLE block, not under a control vector.
- **The GDN recurrence's loads pipelined (upstream ac6aad4, 04f4a53), +2.6%.**
  The column-split kernel loads the next token's q and k rows, v, gate and beta into registers while the current token computes.
  The arithmetic and its order are unchanged. `STRATA_GDN_PIPELINE=0` loads each token's inputs when it starts, as before.

The same binary on the B70 made to look like a small card (`STRATA_VRAM_LIMIT_MIB=8192 STRATA_MAX_ALLOC_MIB=4096 STRATA_NO_XMX=1`, IQ2_XS) starts, answers (17.6 tok/s with MTP) and reads the long prompt (574 tok/s).

## Kept: `--prefill auto` up to 32768-token chunks (upstream ae652b2), +43%

Each chunk streams again every expert it routes to, so the long prompt in 8192-token chunks streamed nearly every expert four times.
`--prefill auto` may now pick 32768 or 16384 too, under the same lend rule (a share of the cache's slots) and never past `--max-context`.
Upstream made these sizes opt-in (3e31eea) because other chunks rounded differently on CUDA; on the B70 the logits are bitwise the same, so XeStrata takes them by default.
`STRATA_PREFILL_AUTO_MAX=8192` restores the old ceiling.
[r_ab.sh](r_ab.sh), two rounds:

| Prompt | auto up to 8192 (tok/s) | auto up to 32768 | Experts streamed |
| --- | --- | --- | --- |
| 26,292 tokens | 1083.76 / 1076.29 (8192) | 1552.10 / 1550.80 (26368) | 50,356 to 17,728 |
| its first 12,000 | 978.39 / 975.08 (8192) | 1308.61 / 1304.47 (12032) | 27,740 to 14,680 |

The lent slots take 0.36 s to refill after the long prompt, against 0.14 s (counted in the rates above).
With `--mtp` the long prompt reads at 1537.57 tok/s, with the same drafts accepted (89 of 120).
On the small-card limits the lend rule still picks 6144 tokens, as before (574 tok/s).

## Kept: the draft layer's prompt pass in batches (upstream c6c6594, E-9)

With `--mtp`, the draft layer needs its own K/V for every prompt cell its window can reach.
The drafter computed them itself, four rows per graph launch: 2.23 s of the long prompt.
The prompt path now computes them after each chunk, in batches of thousands of rows, with its own kernels and its idle scratch (`Prefill::draft_kv`).
The products are FP16 matrix products of the dequantized Q8_0 weights, where the drafter quantizes the activations to Q8_1 (upstream uses a Q8_1 x Q8_0 MMQ kernel, which XeStrata does not have).
So the drafts may differ; the target model's computation does not change.
`STRATA_MTP_BATCH=0` keeps the drafter's own pass.

The long prompt with `--mtp <data>/mtp/rt`, then 128 tokens ([m_ab.sh](m_ab.sh)):

| Pass | Prefill (tok/s) | MTP prompt (ms) | Drafts accepted | Decode (tok/s) |
| --- | --- | --- | --- | --- |
| batched | 1076.29 / 1074.74 | 53.1 / 53.0 | 89 of 120 | 61.04 / 61.03 |
| the drafter's own | 984.42 / 986.86 | 2232.1 / 2230.1 | 91 of 117 | 62.23 / 62.94 |

Prefill +9.2%.
The drafts accepted on two more prompts ([runs/](runs/), `m-acc-*`), batched against the drafter's own pass:

| Prompt | MTP prompt (ms) | Drafts accepted |
| --- | --- | --- |
| the long prompt's first 6,133 tokens (`mid_ids`) | 13.0 against 538.9 | 78 of 150 against 76 of 156 |
| its first 12,000 tokens | 24.3 against 1028.6 | 76 of 153 against 75 of 156 |

Acceptance is neither better nor worse.
On the long prompt the two passes generated the same first 115 tokens, then different ones.
That comes from the verify windows, not from the target's state: the logits are bitwise the same up to position 29, where the drafts first differ.
After that the windows take other shapes, and the logits differ by up to about 1.
The drafter's own pass with only the window shapes changed (`--spec-min-p 0.9`, [runs/m-dump-ownminp.txt](runs/m-dump-ownminp.txt)) differs from it as much: up to 1 from position 1 on, and a different token from position 91.

On the small-card limits (no XMX: the products run through DP4a), IQ2_XS, the long prompt and 32 tokens: the MTP prompt took 157.7 ms against 2233.2, with the same drafts accepted (8 of 69).

## Kept: the fused kernel for the cache's small experts (F8)

At 1,500 tokens an expert has about 29 rows, and dequantizing its 9.4 MiB to FP16 cost more than its products (0.8 s of the phase times).
The fused kernel (`iq_gemm_grouped_f16`, below under "Not kept" as first used) now takes the experts that sit in the VRAM cache and have at most 256 rows: they are laid out first and multiplied 32 a launch straight from their GGUF blocks in the cache; the others go as before.
Its outputs are those of dequantizing and multiplying, so the logits do not change.
`STRATA_PREFILL_FUSED=0` dequantizes every expert; a device without XMX never takes it.
[f8_ab.sh](f8_ab.sh), two rounds:

| Prompt | Every expert dequantized (tok/s) | Small resident experts fused |
| --- | --- | --- |
| first 1,500 tokens | 803.31 / 801.57 | 833.90 / 832.35 |
| first 6,133 tokens | 1007.39 / 1008.34 | 1016.60 / 1017.63 |
| 26,292 tokens | 1550.18 / 1555.21 | 1549.82 / 1547.91 |

+3.8% and +0.9%, and the same at 26K (its experts have more rows); the logits are bitwise the same in every pair.

## Opt-in: the BF16 projections' remainder products (STRATA_PREFILL_BF16X2, upstream 61638c1, 4e0592b)

The prompt path rounds the activations of the BF16-weight projections (router, indexer, SSM alpha/beta, shared gate, PLE key/value, hyper-connection) to BF16; decode reads them in FP32.
`STRATA_PREFILL_BF16X2=2` adds each activation's BF16 remainder as a second product summed into the first (all but the hyper-connection); `=1` the hyper-connection's too; `0` (the default, as upstream) is off.
The matrix products gained an `accumulate` form for it ([x2_ab.sh](x2_ab.sh), [kl_first.py](kl_first.py)).

- **The default is the same bits.** The 26,292-token prompt (IQ3_S, int8 KV) against the binary before the change: logits bitwise the same, 1,512.60 / 1,541.16 tok/s before and 1,525.65 / 1,544.84 after ([runs](runs/x2-long-bf16x2-1.txt)).
  Without the matrix engines (`STRATA_NO_XMX=1`, the DP4a path) and on the small configuration the logits are bitwise the same too.
- **What it took.** As a run-time flag inside the XMX kernel the default products ran six times slower (245.69 / 254.66 tok/s), so the sum is a template argument with one tile shape of its own.
  An accumulating DP4a kernel beside the others (a run-time flag, then a template argument) moved the last bits of the default DP4a products (first-token KL 5.2e-3 on the 1,500-token prompt), so without XMX the remainder product goes through `gemm_rows` instead.
- **The products.** A probe ([acc_gemm.cpp](acc_gemm.cpp)) compared the summed product with the two products summed on the host, at the prompt path's BF16 shapes and 1, 18, 100 and 1,500 rows: within 4.0e-6 relative with XMX and 3.9e-7 without ([runs](runs/acc_gemm.txt), [without XMX](runs/acc_gemm_noxmx.txt)).
- **The numbers it changes.** First-token KL against mode 1 (the most precise):

  | Prompt | Mode 0 (default) | Mode 2 |
  | --- | --- | --- |
  | 19-token chat, IQ2_XS | 1.1e-7 | 4.7e-7 |
  | 19-token Japanese, IQ2_XS | 3.1e-10 | 1.5e-10 |
  | 1,500 tokens, IQ2_XS | 5.1e-3 | 2.4e-2 |
  | 26,292 tokens, IQ3_S | 5.0e-5 | 1.5e-4 |

  The first token is the same in every case. Mode 2 is not between mode 0 and mode 1 here; upstream's IQ2_XS runs found it closer to mode 1 on two of three short prompts.
  The engine has no decode-path reference for a native pack's prompt: its token loop starts the verify window at the first position after the batched part, so `--prefill-until` skips the rest of the prompt instead of running it token by token.
  Which mode is nearer decode's FP32 activations is `unverified`.
- **Speed.** The long prompt, one run each: mode 0 1,503.63 tok/s, mode 2 1,535.83, mode 1 1,431.53. The small configuration's 1,500-token prompt: mode 0 299.93, mode 2 288.40.

## Not kept

- **Experts streamed from 1024-token chunks (upstream 8acd17c, was 2048).**
  On the long prompt's first 1,500 tokens (one chunk; [h_ab.sh](h_ab.sh)), the prefill rate fell from 805.36 / 804.51 to 437.94 / 432.28 tok/s.
  At that size every expert is streamed, including the ones the routing does not pick: 12,454 experts against 5,947 ([h2_ab.sh](h2_ab.sh), with the phase times).
  Upstream measured the gain on a 12 GB card, which holds a smaller cache; the threshold stays 2048.
- **The grouping tables in host USM (upstream fe609ce).**
  The ids, slots, sources and bounds were read and written in place by kernels instead of being copied behind the expert stream.
  1080.54 / 1080.89 tok/s against 1077.62 / 1080.70 with the copies ([h_ab.sh](h_ab.sh), arms `hnofused` and `hcopy`): the same.
- **The fused expert kernel (`iq_gemm_grouped_f16`).**
  It multiplies an expert group's rows by weights still in GGUF blocks, dequantizing each tile in local memory: the same bits as dequantizing and then multiplying.
  It was used for the groups streamed through the big ring.
  On its own (16 experts, 80–240 rows each) it ran gate/up at 0.96–1.52 times and down at 1.6–1.8 times the speed of dequantizing and multiplying.
  Each work-group of 256 rows decodes its slice of the weights again, so with more rows than that it was slower: IQ3_S gate/up at 1000 rows an expert, 0.54 times.
  Used for every streamed group, the long prompt took 1024.67 / 1022.76 tok/s against 1060.90 / 1062.85 without it (`STRATA_PREFILL_FUSED=0`; [f2_ab.sh](f2_ab.sh)).
  Phase times (`STRATA_PREFILL_TIMING`, [runs/f3-time-fused.txt](runs/f3-time-fused.txt), [runs/f3-time-dq.txt](runs/f3-time-dq.txt)): dequantizing went from 4.07 to 2.41 s, while the two products went from 2.69 to 5.31 s.
  Used only for groups of at most 256 rows an expert, it was neutral: 1082.35 / 1080.69 tok/s against 1080.54 / 1080.89.
  Not committed.
- **The AVX-2 kernels' prefetch of the expert rows (upstream fa6310b)**, measured where the CPU's experts set the pace: the small-card limits above, IQ2_XS, a 96-token MTP answer, three rounds ([pf_ab.sh](pf_ab.sh)).
  On: 17.64 / 16.58 / 16.68 tok/s. Off: 16.42 / 16.66 / 16.77. Within the noise.
- **The QSA block scores, one block a work-item.**
  After the 32768-token chunks, the phase times put the block scores at about 0.77 s of the long prompt and the top-k at 0.24 s.
  The score kernel gives each (query, block) a sub-group: each lane multiplies 4 of the 128 dims, and an xor butterfly sums the lanes, once per indexer head.
  A kernel with one block a work-item and the query in local memory did the same products and the butterfly's sums in the same pairs, so the scores and the logits were bitwise the same.
  It was slower: 1533.23 / 1532.77 tok/s against 1551.11 / 1552.60 ([o_ab.sh](o_ab.sh)); each work-item's key reads no longer coalesce. Not committed.
- **The prompt attention without the low halves.** A probe dropped the low FP16 halves of q and p (one product each instead of two) in `qsa_prompt_attn`.
  The kernel ran only 1.05-1.07 times as fast (32,768 cells, 2,048 queries: 33.5 to 31.9 ms), and its error against FP64 grew from 3e-6 to 3.6e-3.
  The matrix products are not what limits it; the per-chunk gathers and barriers are. Not committed.
- **The prompt attention with two queries a work-group.**
  Neighbouring queries select mostly the same cells: a tile of 16 queries reads 19.5% of the cells the 16 read one at a time (`STRATA_SEL_OVERLAP`).
  A version of `qsa_prompt_attn` took two queries a work-group, walked the union of their selections 16 cells at a time (each entry placed by its index plus the other list's entries below it that its own list lacks) and masked, row by row, the cells a query had not selected.
  Its outputs matched the one-query kernel to the last bits on probes with equal, disjoint and partly shared selections, and `qsa_prompt_attn_parity` passed.
  The long prompt read at 897.29 tok/s against 1547.70 (the attention phase several times longer), with half the local memory staging q one query at a time; with the windows read from global memory inside the walk, 671 tok/s. Not committed.
- **Split-K for the products with few tiles (F9), not written.**
  At 1,536 rows the engine's XMX GEMM leaves the GPU part idle on the narrow products: `alpha` (N 48, K 2560) took 37.8 us at its best tile against oneMKL's 11.1, and the hyper-connection down projection (N 320, K 10240) 181.8 us (the scratch probe `xg_shapes`, kernel times from SYCL events).
  The prompt path runs the down projection 96 times a chunk and `alpha` / `beta` 72 times: about 18 ms and 3 ms of a 1,500-token prompt's 1.86 s.
  Halving them would save about 12 ms (0.7%), and at 26K the products have tiles enough. Not worth a second kernel.
- **The prompt attention with the next chunk's gather in flight.** `qsa_prompt_attn` loaded the next 16 cells' K/V pieces (or only their row indices and scales) into registers before computing the current chunk, and stored them to local memory at the top of the next pass (the same bits).
  `qsa_prompt_attn_parity`, ms per chunk of queries against the kernel as committed: int8 KV, 32,768 cells, 2,048 queries 37.4 to 41.7 (0.90x) with the K/V pieces, 36.3 to 40.8 (0.89x) with the indices only; fp16 KV 0.90x and 1.06x; 1,500 cells 1.04x and 1.07x. Not committed.

## The prompt attention, again: +12% on the long prompt

Measured in `qsa_prompt_attn_parity` with parts of the kernel cut out (a probe build, not kept): of 33.3 ms a chunk (int8, 2,048 queries over 32K cells), gathering K and V took about 10 ms, q.k 5 ms, p.v 3 ms, and the rest (row lookups, scales, softmax, splitting p, barriers) about 15 ms.
Three changes ([1623fc6](../../../src/kernels/xe/qsa_prompt_attn.cpp)), each measured alone and kept:

- The large register file (`grf_size<256>`): 30.3 to 21.1 ms. The kernel runs only where `xmx_available()`, which requires the mode.
- int8 codes to FP16 with `vec<int8_t, 8>::convert<half>` instead of CUDA's mantissa trick two at a time: 33.3 to 29.9 ms with the default register file.
- Splitting p into hi and lo with each lane owning one cell (the chunk's 16 cells are the sub-group's 16 lanes): 21.1 to 19.5 ms with the large register file.

| Case | Before (ms) | After (ms) |
| --- | --- | --- |
| int8, 32K cells, 2,048 queries | 33.3 | 18.9 |
| fp16, 32K cells, 2,048 queries | 24.6 | 16.0 |
| int8, 1,500 cells, 1,500 queries | 9.1 | 5.2 |
| int8, 2,100 cells, 256 queries | 5.3 | 2.7 |

The error against FP64 and against the FP32 kernel is the same number as before in all four cases.
In the engine, IQ3_S, the 26,292-token prompt (`--prefill auto`, 32K chunks), `--expert-cache 10000`, two rounds alternating with the commit before: 1542 / 1547 to 1729 / 1753 tok/s, the logits bitwise the same ([runs](runs/), `qpa-*`).
The 32-cell chunk is still slower with the large register file (37 ms against 19).
The free build with intel/llvm 7 gives the same parity times. On the small configuration (8 GiB, 4 GiB allocations) IQ2_XS read a 6,134-token prompt at 986 tok/s with XMX and 583 without (`qpa-small*`).
