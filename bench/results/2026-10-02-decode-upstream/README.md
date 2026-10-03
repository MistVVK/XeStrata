<!--
SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
SPDX-License-Identifier: LGPL-3.0-or-later
-->
# Decode: upstream's changes after 0.1.24 on the B70 — 2026-10-02

Upstream Strata changes after the fork point (0.1.24) ported to SYCL and measured on the B70.
Machine as in [the new machine's record](../2026-10-02-new-machine/README.md); IQ3_S and IQ2_XS, `--expert-cache 10000`, greedy.
MTP runs: `--mtp <data>/mtp/rt`, the 20-token `port_ids` prompt, 128 tokens.
"No drafts" runs: the `--spec 4` window without a drafter, the 19-token `chat_ids` prompt, 96 tokens.
Each pair alternates, two rounds after a warm-up ([d_ab.sh](d_ab.sh), [e_ab.sh](e_ab.sh)); logs in [runs/](runs/).

## D1: the i-quant weights decoded once for every column (upstream 8ec94aa)

The i-quant MMVQ and the GPU's grouped experts called a format's dot once per column (a token of the verify window) or entry.
Each call re-read the weight words and redid the grid lookups and the sign handling.
Each dot is now split into a weight side (`load`) and an activation side (`apply`).
A block is loaded once and applied to every column: dense rows take 1, 2, 4 or 8 columns a pass, the grouped experts 4 entries.
A one-column dot is `apply(load(...))` too, so every column is bitwise what it was.

| Model | MTP before (tok/s) | MTP after | No drafts before | No drafts after |
| --- | --- | --- | --- | --- |
| IQ2_XS | 67.64 / 67.64 | 69.97 / 68.27 | 25.20 / 24.68 | 25.34 / 25.17 |
| IQ3_S | 59.02 / 59.21 | 58.80 / 60.35 | 24.14 / 24.14 | 23.95 / 24.12 |

IQ2_XS +2.2% with MTP; its dense matrices are mostly i-quants. IQ3_S's are mostly Q6_K (`native_mmvq`), which this does not touch.
The logits are bitwise the same in all four cases.

## D2: the verify window's commit not waited for (upstream 42b4299, a454dbb)

The commit graph's host wait (about 0.5 ms a round) is gone on a single GPU.
The next window follows on the same queue, and the drafter reads nothing the commit writes.
Everything that touches the session from another queue or the host first waits on an event recorded after the commit.
`STRATA_COMMIT_SYNC=1` keeps the wait.

IQ3_S, MTP: 61.22 / 61.14 tok/s, against 59.69 / 60.11 with `STRATA_COMMIT_SYNC=1` (+2.0%).
These runs are a later binary with D1, D2 and the two changes below (the switches turn one change off each).

Not kept:

- **QSA block scores read once for all of a window's queries (upstream a20f3b5).** 61.22 / 61.24 tok/s with it off (`STRATA_SCORES_MULTI=0`), the same as on; reverted
- **The AVX-2 kernels' software prefetch of the expert rows (upstream fa6310b).** 61.00 / 61.11 with it off (`STRATA_IQ_PREFETCH=0`): within the noise here, where decode waits on the GPU. To be measured where the CPU's experts set the pace (a smaller card)

## D8: the sampled path's top_k split over the GPU (upstream 47d6894) — 2026-10-03

With a temperature, the sampler gave a row one work-group, which took the top_k list by k argmaxes over the whole vocabulary, checking each logit against the ids already taken.
Stage 1 now takes the first k of each block of 4,096 logits on a work-group of its own; stage 2 takes the row's first k from the blocks' lists, then draws as before.
The first k of the selection order (penalised value descending, id ascending; NaN and -inf never taken) lie within the first k of each block, so the list and the token are the same.
`STRATA_OLD_SAMPLER=1` keeps the one-group kernel.

IQ3_S, MTP, 128 tokens on `port_ids`, `--temperature 0.7 --top-k 40 --top-p 0.95 --seed 7` ([p_ab.sh](p_ab.sh)):

| Sampler | Decode (tok/s) | Drafts accepted |
| --- | --- | --- |
| split | 65.86 / 65.87 | 82 of 141 |
| one group | 51.98 / 52.39 | 82 of 141 |
| greedy (for scale) | 60.99 / 60.13 | 80 of 150 |

+26%, the same generated tokens.
[sampler_paths.cpp](sampler_paths.cpp) samples 60 cases of 4 rows at the model's 248,320 logits (ties, NaN, -inf, penalties, k from 1 to 64, fewer candidates than k); run once with `STRATA_OLD_SAMPLER=1` and once without, the 240 tokens are the same.

## D5: the multi-token hyper-connection read's norm split by stream (upstream dbb1c23) — 2026-10-03

The verify windows' and the drafter's hyper-connection read normalized each token on one work-group: 4 groups for 4 tokens.
It now takes a group per token and stream; a work-item adds its stream's elements in `norm_step`'s order and the partials in the same order, so the outputs are bitwise the same.
`STRATA_HC_SPLIT=0` keeps one group per token.
MTP, 128 tokens on `port_ids` ([hc_ab.sh](hc_ab.sh)):

| Model | One group per token (tok/s) | Per token and stream |
| --- | --- | --- |
| IQ3_S | 61.47 / 61.20 | 62.85 / 62.94 |
| IQ2_XS | 70.94 / 70.38 | 72.88 / 72.06 |

+2.6% and +2.5%; the logits are bitwise the same.
Upstream's other step (the down projection's activations staged by cp.async, dbb1c23's "staged") has no counterpart here: the SYCL down kernel already gives each (row, token) a sub-group and stages nothing.

## D6: where a verify window's GPU time goes — 2026-10-03

VTune stalls the engine, and the B70 has no device-scope clock for the window's own stage stamps.
A scratch build ran the window as host-launched segments (the segmented verify window of the UHD 770, forced with `STRATA_VERIFY_SEGMENTED=1`), cut at every stage stamp, and timed each segment from the host (about 0.01 ms of each is the sync).
IQ3_S, MTP, 128 tokens, 4-token windows; ms per window, summed over the layers:

| Stage | 36 GDN layers | 12 QSA layers |
| --- | --- | --- |
| attention half's hyper-connection read (stamp 1) | 4.15 | 0.70 |
| GDN projections, conv, recurrence, norm (2-6) | 7.71 | - |
| QSA projections, indexer, selection, attention (7-14) | - | 3.59 |
| out projection, FFN half's read (16) | 2.16 | 0.67 |
| router and the ring (17) | 3.30 | 1.05 |
| shared expert (18) | 2.34 | 0.78 |
| the VRAM experts (20) | 6.71 | 2.13 |
| the rest (19, 21-24) | 2.90 | 0.97 |

The VRAM experts take about 0.19 ms a layer, about twice their weights' traffic at the card's bandwidth.
The hyper-connection reads take about 7.7 ms a window; their weights (13 MB a read) would need about 0.02 ms each.

Tried on the read and not kept: the down projection with one sub-group per row for every token, so each weight row is read once instead of once per token (the same bits).
It was slower: IQ3_S 60.89 / 60.97 / 60.67 tok/s against 62.72 / 62.78 / 62.71, IQ2_XS 70.31 / 70.00 / 69.16 against 72.22 / 72.24 / 70.93 ([hcd_ab.sh](hcd_ab.sh)).
The re-reads come from the cache; a quarter of the sub-groups leaves the GPU idler.

Tried on the VRAM experts and not kept: their SwiGLU and its q8_1 quantization in one launch instead of two (the same bits).
IQ3_S 62.81 / 62.86 / 62.32 tok/s against 62.84 / 61.75 / 62.69, IQ2_XS 69.31 / 72.22 / 72.29 against 72.28 / 72.01 / 71.88 ([sq_ab.sh](sq_ab.sh)): the same.

## E: the draft subset with Chinese, Japanese and Korean (upstream 6e153c9)

The draft head's token subset went from the English and code subset (40,525 tokens, now `data/draft_vocab_en.bin`) to one with every Han, kana, Hangul and CJK punctuation token (106,299, `data/draft_vocab.bin`).
`setup.sh --draft-vocab cjk|en|cyrillic` chooses one; cjk is the default.

IQ3_S, MTP, 128 tokens; a Japanese prompt (「空が青い理由を三文で説明してください。」, 19 tokens) and the English `chat_ids` one:

| Prompt | English subset (tok/s; drafts accepted) | CJK subset |
| --- | --- | --- |
| Japanese | 45.54 / 45.04; 76 of 156 | 53.38 / 54.94; 82 of 135 |
| English | 47.44 / 48.97; 75 of 156 | 47.94 / 47.68; 75 of 156 |

Japanese answers +19.6%, English ones -0.3% (noise: the same drafts were accepted).
The draft head grows from 81.2 to 212.9 MiB of VRAM, and drafting from 4.55 to 5.37 ms a round.
The expert cache now reserves the head's VRAM before it is sized (upstream b981f63); with `--expert-cache auto` the cache takes that much less.
The generated tokens are the same with either subset (the first 40 of both prompts compared).
The logits' last bits differ, as the verify windows take other shapes.

## Coupled draft sampling (upstream 1643965, 8979c32): not kept — 2026-10-03

Under sampling the MTP draft layer proposes its argmax. Upstream's coupled mode samples the draft instead, with the target's chain (top_k, top_p, min_p, temperature, penalties) and the same Philox draw the verify window will use for the row that checks it (counter = the draft's cell + 1).
It was ported to the SYCL sampler (the split sampler's stage 1 with the widest list, then a merge that reads the request's parameters and the counter from device memory, inside the drafter's captured graphs) behind `--coupled-draft`. The host-side arithmetic test passed (907,257 checks).

IQ2_XS, `--expert-cache 10000`, `--adapt-every 0`, seed 42, top_k 20, top_p 0.95, up to 256 tokens ending at the end of turn, the four prompts of [the acceptance record](../2026-10-03-mtp-accept/README.md), summed:

| `--spec-min-p`, temperature | Argmax drafts (tok/s, accepted) | Coupled drafts (tok/s, accepted) |
| --- | --- | --- |
| 0.5, 0.7 | 60.94, 0.736 | 58.15, 0.724 |
| 0.5, 1.0 | 56.01, 0.662 | 56.11, 0.712 |
| 0, 0.7 | 53.45, 0.578 | 52.83, 0.566 |
| 0, 1.0 | 54.44, 0.554 | 53.43, 0.555 |

The coupling works: shifting its counter off the verifying row's (by 1, -1 or 1000) lowers the acceptance at temperature 1.0 from 0.555 to 0.533-0.557.
It just gains nothing here, as the draft layer's distribution is too far from the target's for a shared draw to agree more often than the argmax does; with `--spec-min-p 0.5` the chains also stop sooner, as a sampled draft's probability is lower than the argmax's.
Not committed; the patch is kept outside the repository (`records/2026-10-02-new-machine/patches/coupled-draft.patch`).
Runs with `--expert-cache auto` differed between repeats only because the cache's size followed the free VRAM.

## The hyper-connection read's up step, four rows a sub-group: not kept — 2026-10-03

`fused_gr_read_multi` alone (a probe at the model's geometry, 4 tokens, B70): about 38 µs a read with the weights in the caches, of which the norm takes about 5, the down projection 11-14 and the up projection about 20 (measured by leaving steps out).
The up projection takes one 640-byte row of `w_up` a sub-group at a time and reduces every token's sum over 32 lanes.
A version taking four rows a sub-group, eight lanes a row, cut the up step from 20 to 15 µs with the weights in the caches, but made it slower with them evicted (256 MiB written between reads): 32 to 35-36 µs. Reading the four rows' 2,560 bytes with consecutive lanes and regrouping them in local memory did not help (42 µs evicted).
In the engine, MTP, `--expert-cache 10000`, 128 tokens: IQ2_XS 70.0-71.4 to 73.3-73.8 tok/s (+3%), IQ3_S 60.8-61.3 to 55.2-60.9 (six rounds; -0.8% without the first round's 55.2).
Not committed, as one model got slower; the patch is kept outside the repository (`records/2026-10-02-new-machine/patches/hc-up-four-rows.patch`).
A down step with the tokens' rows in local memory and each weight row read once was twice as slow as the engine's (25 against 12 µs at 4 tokens), as was its one-row-a-sub-group form before.

## The DP4a prompt path's accuracy — 2026-10-03

On a device without the matrix engines (or a SYCL runtime that reports none) the prompt path's products go through `dp4a_gemm`, which quantizes both X and W to int8 in blocks of 32.
Against llama.cpp at the pinned commit on the CPU (`ref_logits`, as in `bench/results/2026-09-30-xe-iq2xs`), IQ2_XS, the first generated token's logits:

| Prompt | KL(llama.cpp ‖ XMX) | KL(llama.cpp ‖ DP4a) | Argmax (llama.cpp, XMX, DP4a) |
| --- | --- | --- | --- |
| 19-token chat | 2.0e-07 | 3.1e-07 | the same |
| 19-token Japanese | 2.6e-09 | 3.5e-08 | the same |
| 1,500 tokens (the prompt path) | 9.8e-03 | 0.23 | DP4a differs |

So the DP4a path is not just rounded differently: re-quantizing the weights (already FP16 / BF16) to int8 adds an error the 48 layers build up.
Splitting W into two int8 planes (the second at 1/127 of the scale, so W keeps about 14 bits; X stays int8, as llama.cpp's activations do) gives KL 6.1e-03 and llama.cpp's argmax, at 16-23% of that path's prompt speed (IQ2_XS, `STRATA_NO_XMX=1`: 6K tokens 583 to 487 tok/s, 26K 714 to 550; with 8 x 4 outputs a work-item it spilled and ran at 43 tok/s, so 4 x 4).
The owner chose not to take it (2026-10-03): the patch is kept outside the repository (`records/2026-10-02-new-machine/patches/dp4a-weight-split.patch`).
Short prompts, which go through the decode path, are not affected.
