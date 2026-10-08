<!--
SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
SPDX-License-Identifier: LGPL-3.0-or-later
-->
# Upstream 0.1.40's speed paths on the B70 and the RTX 4070 — 2026-10-09

The merge of upstream Strata 0.1.39 to 0.1.40.2 brought paths meant to be faster.
Each was measured against the same binary without it, on the development machine: an Intel Arc Pro B70 (32 GB), an RTX 4070 (12 GB, PCIe x4), a Core i7-14700 and a Samsung 990 Pro.
A path is kept when it is faster on the B70 or on the RTX 4070 without making the other slower; one that is faster on only one of them is chosen at run time from what the device reports, or stays opt-in.

The model is the IQ2_XS pack with the stock draft layer (`--spec 4 --spec-min-p 0.5`, `--kv int8`), greedy.
The first items were measured with the contrib-icpx build (icpx), the rest with the contrib build (intel/llvm with CUDA), which runs on both cards.
The arms alternate; a run that overlapped a build is left out.

- [perf.py](perf.py): three fresh documents of about 4,945 tokens, 256 tokens generated after each, then three short chats of 256 tokens; the medians of the prompt speed, the decode after the long prompt and the short decode.
- [perfdec.py](perfdec.py): decode only, six chats of 512 tokens after a warm-up (about ±0.3% from run to run).

## Not kept

| Path | Result |
| --- | --- |
| Unbuffered reads on Linux (2d929a28, 805088d5, 066094c8) | `--mmap-experts --resident-budget-gib 8`, the page cache emptied before each run: decode 9.9-19.1 against 21.7-39.7 tok/s with the OS cache (also at 22 and 14 GB of RAM) |
| BF16 GEMV four rows a group (c637ed5a), as a default | within the spread on the B70; chosen per device at run time instead (below) |
| MMQ groups sorted by rows for all-resident layers (c6a0d403) | prompt 1140.0, 1134.6 against 1132.9, 1134.9 tok/s; the condition never happens on the RTX 4070 |
| combine with float4 (822251be) | +5% / +12% (B70), +21% / +12% (RTX 4070) kernel time at 1 / 4 tokens |
| fused_gr's up with T fixed (104a8485) | 36.3 -> 36.0 / 46.8 -> 46.4 µs (B70), 41.3 -> 41.5 / 54.7 -> 55.0 µs (RTX 4070) at T=4 / 6 |
| kv_stream resolve without the zero atomic (1690a4a3) | 3.42 -> 3.44 µs (B70), 6.752 -> 6.760 µs (RTX 4070) |
| The GDN step's prefetch with six to four barriers (ae3b249f), always | 11.0 -> 17.1 µs on the B70; chosen per device instead (below) |
| A small format's one-time unpacking (2715a752) | no difference on either card |
| Attention scores one cell a work-item (8b30b1d5) | 54.3 / 68.8 -> 70.3 / 298.8 µs (B70), 36.2 / 89.5 -> 72.2 / 198.0 µs (RTX 4070), T=1 / 4 |
| HC read-out's up writing q8_1 (22abb92d) | 41.00 -> 43.81 µs (B70), 43.81 -> 43.96 µs (RTX 4070) |
| Draft acceptance by probability (`STRATA_SPEC_PROB`, 33356f72) | sampled decode 79.20 (B70), 50.25 (RTX 4070) against 80.25 and 51.05 tok/s |

## Kept

Kernel times are the median of back-to-back launches in a probe, old -> new.

| Path | B70 | RTX 4070 |
| --- | --- | --- |
| Sigmoid and scale fused (822251be), 1 / 4 tokens | 2.94 / 2.93 -> 1.47 / 1.47 µs | 3.83 / 3.87 -> 1.61 / 1.61 µs |
| gr_write_multi (104a8485), T=2 / 4 / 6 | 3.10 / 6.07 / 9.08 -> 1.56 / 1.67 / 1.68 µs | 4.33 / 8.62 / 12.92 -> 2.17 / 2.16 / 2.17 µs |
| BF16 MMVF with the rows fixed, up 320 -> 10240 (104a8485), T=2 / 4 / 6 | 48.4 / 50.3 / 88.7 -> 29.7 / 47.4 / 66.0 µs | 36.5 / 36.9 / 65.0 -> 20.7 / 36.8 / 53.5 µs |
| The same, down 10240 -> 320, T=2 / 3 / 4 / 5 | 15.8 / 21.7 / 25.3 / 33.8 -> 9.2 / 11.1 / 13.0 / 14.7 µs | 12.7-13.7 / 15.3-16.5 / 18.0-19.3 / 25.7 -> 10.4 / 14.3 / 20.5 / 19.9 µs |
| The same, inject 10240 -> 4, T=2 / 4 / 6 | 8.4 / 12.0 / 17.3 -> 4.0 / 4.7 / 5.4 µs | 3.6 / 4.1 / 8.7 -> 2.9 / 3.8 / 6.3 µs |
| The MTP router, top-10 and combine in one (868f2ef0), T=2 / 4 / 5 | 27.95 / 55.68 / 69.70 -> 15.16 / 17.97 / 19.11 µs | 19.93 / 36.86 / 45.98 -> 10.82 / 12.31 / 13.68 µs |
| gdn_ab_multi with T fixed (ae3b249f), T=2 / 4 / 6 | 5.28 / 8.87 / 12.43 -> 3.98 / 5.80 / 8.53 µs | 6.2-6.7 / 10.6-10.7 / 15.0 -> 5.89 / 9.64 / 14.01 µs |
| resident_plan's scan (fe4de5de), T=2 / 4 / 8 | 4.4-6.2 / 6.97 / 12.24 -> 4.0-4.1 / 5.84 / 9.80 µs | the same |
| i-quant MMVQ by the column count (2715a752), IQ3_S 3 / 5 / 7 columns | 34.5 / 40.3 / 58.9 -> 20.7 / 25.4 / 32.9 µs | 19.7 / 25.1 / 36.1 -> 13.7 / 19.0 / 24.2 µs |
| quantize_q8_0_scaled one block a sub-group (3281ac32), n=2560 / 20480 / 40960 | 4.5-5.3 / 4.14 / 4.54 -> 2.2-2.7 / 2.05-2.08 / 2.11 µs | 3.2-3.4 / 4.9-5.3 / 8.3-9.0 -> 1.49 / 1.59 / 1.73 µs |
| SwiGLU and Q8_0 fused (3281ac32), 10 / 40 / 80 entries | 3.03 / 2.95 / 3.22 -> 2.35 / 2.41 / 2.78 µs | 3.95 / 3.66 / 4.17 -> 1.97 / 2.25 / 2.78 µs |
| The last mixer's fused read-out (edc592b1), T=2 / 4 / 6 | 69.8 / 139.6 / 209.2 -> 25.7 / 35.8 / 46.7 µs | 62.9 / 118.2 / 177.3 -> 25.8 / 39.7 / 54.1 µs |
| copy_from_mapped_multi (4a5713fb) | 15.5 -> 8.5 µs | 13.0 -> 6.0 µs |
| The router and the shared gate in one (b08cf3e1), 4 rows | 9.97 -> 7.56 µs | 8.95 -> 7.17 µs |
| The shared gate and up in one (be7889ed), Q6_K rows, 4 columns | 11.63 -> 7.58 µs | 9.47 -> 7.44 µs |
| SwiGLU and q8_1 in one, shared expert, 4 rows | 31.3 -> 27.6 µs | 18.2 -> 17.4 µs |
| The GDN norm writing q8_1 (22abb92d), 4 tokens | 14.73 -> 13.82 µs | 10.01 -> 9.66 µs |
| The GDN split (e3c3d6ba), 4 / 6 / 2 tokens, from 3 tokens on | 11.0 / 14.8 / 7.3 -> 10.3 / 13.0 / 7.5 µs | 9.9 / 13.5 / 6.2 -> 9.4 / 12.1 / 6.7 µs |
| row_top_prob in eight parts (3e4aa1d6), 248,320 / 40,525 words, from 65,536 words | 38.96 -> 38.84 / 7.60 -> 7.85 µs | 15.12 -> 10.24 / 3.98 -> 3.52 µs |
| The GDN step's prefetch (ae3b249f), chosen per device | 11.0 -> 17.1 µs (not taken) | 9.16 -> 8.55 µs (taken) |
| BF16 GEMV four rows a group (c637ed5a), chosen per device | 22.30 -> 27.56 µs (not taken) | 22.83 -> 15.61 µs (taken) |
| The shared expert on a branch of its own (4a5713fb), perfdec | 77.17 -> 76.99 tok/s | 58.96 -> 59.67 tok/s |
| The PLE rows in one batch (d92c9feb), perfdec | 77.05 -> 78.05 tok/s | 59.54 -> 59.78 tok/s |
| A one-token window committing itself (352cad8f), perfdec | 77.12 -> 77.69 tok/s | 58.28 -> 58.68 tok/s |
| K/V and index appended in one (44ffa86c), perfdec | 76.52, 76.62, 76.65, 76.12 -> 76.66, 77.00, 76.46, 76.74 tok/s | |
| The MTP catch-up without the rejected rows (3fd0460b), perfdec | 76.77, 76.75, 76.40 -> 76.75, 76.90, 77.01, 76.67 tok/s | |

The CPU kernels for IQ2_S and IQ2_XXS against Q2_0 (a7336175 and others), one thread on a P-core, ms per expert at 1 / 2 / 4 tokens: IQ2_S 0.242 / 0.360 / 0.488 -> 0.242 / 0.253 / 0.382, IQ2_XXS 0.203 / 0.254 / 0.355 -> 0.203 / 0.252 / 0.344.

## Defaults and choices

| Path | B70 | RTX 4070 | Kept as |
| --- | --- | --- | --- |
| `STRATA_PREFILL_CPU_SHARE` (a short chunk's least-routed experts on the CPU pool), prompt tok/s at 262 / 914 tokens | 319.2 / 736.3 -> 383.1 / 828.0 | 109.9 / 275.9 -> 266.1 / 457.8 | `auto`, the default |
| `--kv-grow` (the K/V growing into the expert cache's VRAM), perf.py prompt / after / short | 1133.5 / 100.50 / 77.80 -> 681.9 / 96.2 / 70.4 | 791.6 / 73.60 / 58.50 -> 802.4 / 76.50 / 61.40 | on where the device's mapping granularity is 2 MiB or more (the RTX 4070; the B70's is 64 KiB) |
| `--adapt-async` (the resident RAM mode's swaps between windows), perf.py after / short | 96.70, 98.20 / 75.80, 76.00 -> 96.50, 96.80 / 73.80, 75.30 | 71.10, 68.80 / 53.00, 54.70 -> 79.80, 81.30 / 60.60, 61.90 | the default in the resident RAM mode |
| `--batch-mtp` (each batch slot verifies an MTP proposal), 4 slots x 200 tokens, tok/s from the first batch token | 102.9, 102.1 -> 107.8, 107.9 | 52.0, 45.4 -> 51.6, 50.6 | the default with `--batch` and `--mtp`; the same tokens |
| `--lookup-chain 3`, rewrite / ordinary chats | 109.79, 110.30, 108.80 / 70.58, 71.00, 71.76 -> 129.80, 128.60, 129.07 / 68.49, 68.64, 67.73 | | opt-in |
| `STRATA_ADAPT_LAG=2`, perfdec | 77.85, 75.59, 76.10 -> 76.82, 76.69, 77.13 | 59.19, 58.94, 59.02 -> 64.24, 64.16, 64.24 | opt-in |
| `--expert-cache-per-layer` on a native pack (818ec1c6, a fix), perfdec | 77.61, 75.66, 76.32 -> 74.96, 75.62, 75.59 | 58.81, 58.83, 57.41 -> 54.27, 54.43, 54.58 | opt-in, as before |
| HC read-out in Q8 (`STRATA_HC_Q8`, fe6c5260), UD-Q4_K_XL perfdec | 44.97, 42.06 -> 45.43, 45.27 | 22.39, 22.40 -> 21.92, 21.76 | opt-in |
| Coupled drafts with Gumbel picks (`STRATA_SPEC_GUMBEL`, 6381df34), sampled decode (temperature 1.0, top_p 0.95, top_k 20) | 77.50 -> 72.75 | 49.80 -> 52.05 | opt-in |
| `STRATA_EXCHANGE_ROTATE=1` (1e6cec80, e57c9072), perf.py after / short | 96.10, 96.70 / 74.60, 73.30 -> 95.80, 95.20 / 72.30, 75.20 | 79.10, 81.20 / 60.00, 57.70 -> 81.00, 82.70 / 59.70, 60.00 | opt-in; inactive on a native pack (its experts differ in size by layer) |

With `--kv-grow` on the B70 the K/V alone in virtual memory read the prompt at 1291.5 tok/s and the expert cache alone at 643.5: the slowdown comes from the expert cache in mapped virtual memory, its cause not found.

## `--pipeline-windows 2` (B70 + RTX 4070)

The B70 runs layers 0 to K-1, the RTX 4070 the rest; 32K context, perf.py.

| Split | Serial | `--pipeline-windows 2` |
| --- | --- | --- |
| K=24, after the long prompt | 115.6, 119.8 tok/s | 145.8, 125.8 tok/s |
| K=24, short chats | 87.9, 88.2 tok/s | 90.6, 87.4 tok/s |
| K=47 (auto), after the long prompt | 105.9 tok/s | 87.7 tok/s |
| K=47 (auto), short chats | 81.3 tok/s | 59.3 tok/s |

With auto's split the RTX 4070 holds one layer, so there is nothing to overlap: it stays opt-in.
K=24 serial is also faster than auto's K=47 serial; auto's estimate does not match what this pair measures (not investigated).
With `STRATA_IQ_MT_MIN=1 --pcie-frac 0 --adapt-every 0` and the same expert caches the eight greedy answers of a fixed set were identical to the serial loop's.

## Against upstream's SYCL port (`sycl/`, engine 0.1.40-sycl)

Upstream 0.1.40.2's own Intel port, built from its `sycl/` with oneAPI's icpx, AOT for the B70 (`-DSTRATA_SYCL_AOT=bmg-g31`), against XeStrata's contrib-icpx build, on the B70 alone.
Both run the engine directly: IQ2_XS, `--expert-cache auto --spec 4 --spec-min-p 0.5 --mtp --max-context 8192 --kv int8 --greedy`, XeStrata with its defaults (`--prefill auto --vram-reserve-mib 700`), the port as its docs/INTEL.md runs a model whose experts do not all fit in VRAM (`STRATA_VERIFY_NO_HOST=1 --stream-experts --prefill 4096 --vram-reserve-mib 1024 --ple-io ram`; without `STRATA_VERIFY_NO_HOST=1` its first verify window timed out at layer 4 and the device was lost).
An 18-token chat with 256 tokens out, and 4,095 random token ids (Python `random.seed(1)`, `randint(1000, 150000)`) with 16 out; two interleaved rounds after a first one, all three alike.

| | XeStrata | upstream's port |
| --- | --- | --- |
| decode, chat, 256 tokens | 70.84, 70.84, 70.84 tok/s | 2.18, 2.18, 2.19 tok/s |
| prompt, 4,094 tokens | 1215.9, 1213.1, 1216.4 tok/s | 1418.4, 1417.8, 1421.5 tok/s |
| time to the first token, 4,095 tokens | 4.19, 4.20, 4.19 s | 5.59, 5.59, 5.58 s |

XeStrata with `--prefill 4096` read the prompt at the same 1214.8 and 1216.2 tok/s.
The port held 16,298 experts in VRAM and mirrored the 8,278 others in pinned host memory; its decode took about 2.5 s per verify window (47 windows, 211 of 214 drafts accepted).
