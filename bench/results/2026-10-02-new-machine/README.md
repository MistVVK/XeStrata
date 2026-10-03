# The new development machine — 2026-10-02

The B70 moved to a new machine on 2026-10-01: an i7-14700 (8 performance and 12 efficient cores, AVX2, no AVX-512) on an ASUS PRIME B760-PLUS D4, DDR4-3200 in two channels (about 35.5 GB/s read), the B70 at PCIe Gen5 x16 (the engine's probe reads 42.4 GB/s host to device), and the model files on a CPU-attached Gen4 x4 NVMe.
The records dated 2026-09-30 come from the previous machine (Ryzen 7 3800XT, the B70 at Gen4 x8, about 14.3 GB/s), so their speeds are not comparable with these. This record measures again what the move affects.

- Engine: `build/xe-portable` (the default `STRATA_PORTABLE=ON`), at `a568d1c`; `build/xe` (`-march=native`) where noted
- Runs go through [run.sh](run.sh): greedy, `--spec 4 --adapt-every 0 --prefill 512`, the expert cache `auto` unless `CACHE` fixes it. Each run is a new process; the first run after a build was discarded (JIT)
- Prompts: [port_ids.txt](port_ids.txt) (20 tokens, 128 generated) and the old records' chat prompt [chat_ids.txt](chat_ids.txt) (19 tokens, 96 generated); the 26,292-token prompt is the one `bench/results/2026-09-30-xe-iq2xs/long_prompt.py` writes
- Paths are written as `<repo>`, `<data>` (the data folder) and `<record>` (where the runs wrote); every log is in [runs/](runs/)

## Outputs that change from run to run

On 2026-10-01 two runs with the same settings gave different tokens (IQ2_XS with MTP from token 45, IQ3_S without drafts from token 77), and the CPU's 19 worker threads were suspected.

- With the cache size fixed, the runs repeat bit for bit: IQ3_XXS without drafts at `--expert-cache 12000` gave the same 128 tokens 9 times out of 9 (`fx-nd-*`), and at 12287 6 times out of 6 (`fy-nd-*`). IQ2_XS with MTP and IQ3_S without drafts, 3 runs each on both builds, with and without `--dump-logits`, were identical (`p-*`, `n-*`, `pp-*`)
- The usual cause is the cache size: `auto` sizes the cache from the free VRAM at start, which moves by a few MiB, so the number of resident experts moves by one or two slots (16,454 against 16,452 in `x3-nd-*`). A different set of experts runs on the CPU, whose arithmetic differs from the GPU's in the last bits. In `x3-nd-1` against `x3-nd-2` the logits first differ at row 26 of the generation (largest difference 0.32) and the tokens at 53. Over the 16 model and mode pairs of the decode baseline, every pair with a different output but one also had a different slot count
- The one exception: IQ3_XXS without drafts, `base-iq3_xxs-nodraft-1` against `-2`, the same 16,454 slots and the same log apart from the load speed, but different tokens. The later runs with 16,454 slots (`x3-nd-1`, `fy-nd-1` to `-5`) all gave run 2's tokens; the cause of run 1's is `unverified`
- The number of CPU worker threads does not change the output (7, 11, 15 and 19 workers gave the same tokens, `w*-iq3-nd-*`)
- To compare outputs between runs, fix the cache size (`--expert-cache N`), as the records with "fixed residency" already did

## CPU worker threads

The pool's default is one worker per physical core: 19 workers and the host thread on this CPU, against 7 on the old one. IQ3_S without drafts, 128 tokens, two rounds alternating (`w*-iq3-nd-*`):

| Workers | Decode (tok/s) | CPU pool (ms/round) | Waiting for the GPU (ms/round) |
| --- | --- | --- | --- |
| 19 (default) | 15.02 / 15.13 | 6.93 / 6.84 | 52.0 / 51.7 |
| 15 | — (the start failed, see below) / 15.20 | 7.09 | 51.1 |
| 11 | 15.13 / 15.18 | 7.37 / 7.18 | 51.1 / 51.0 |
| 7 | 15.17 / 15.17 | 6.80 / 6.90 | 51.3 / 51.3 |

The CPU's part takes about 7 ms of a 60 ms round whatever the count, and the round waits on the GPU, so the default stays. The efficient cores do not hold the pool back measurably here.

## Decode on every model

128 tokens on `port_ids.txt` (`base-*`), and the old records' 96 tokens on the chat prompt (`chat-*`), two rounds each. The old machine's numbers are those records' `spec-nodraft.txt` and `spec-mtp.txt` (chat prompt, 96 tokens; IQ2_XS: its README's adaptive runs).

| Model | No drafts, 128 tokens | MTP, 128 tokens | No drafts, chat 96 | MTP, chat 96 | Old machine, chat 96 (no drafts / MTP) |
| --- | --- | --- | --- | --- | --- |
| IQ2_XS | 17.05 / 16.97 | 44.14 / 44.14 | 17.16 / 17.05 | 42.98 / 43.93 | 16.7 (suffix) / 29.8 |
| IQ3_XXS | 16.19 / 15.93 | 43.41 / 38.00 | 16.21 / 16.21 | 40.12 / 38.05 | 14.84 / 25.84 |
| IQ3_S | 15.19 / 15.04 | 33.67 / 33.87 | 15.05 / 15.07 | 33.42 / 33.55 | 14.17 / 29.03 |
| Q2_0 | 17.33 / 17.34 | 49.35 / 49.47 | 17.41 / 17.42 | 41.35 / 41.31 | 16.28 / 26.96 |
| Coder IQ1_M | 14.16 / 14.15 | 33.76 / 33.74 | 14.13 / 14.12 | 33.35 / 33.88 | 9.10 / 33.09 |
| Swift IQ2_XS | 17.05 / 17.09 | 43.16 / 43.20 | 17.21 / 17.21 | 28.83 / 28.69 | 5.73 / 18.42 |
| Swift IQ3_XXS | 16.15 / 15.80 | 39.47 / 39.50 | 16.25 / 16.00 | 38.29 / 38.74 | 14.64 / 32.39 |
| Swift Q2_0 | 17.33 / 17.32 | 43.20 / 43.21 | 17.39 / 17.40 | 43.55 / 42.68 | 16.89 / 41.18 |

Without drafts every model now runs at 14–17 tok/s and waits on the GPU (the CPU pool takes 0.6–6.6 ms of each round). The old Coder (9.10) and Swift IQ2_XS (5.73) runs were the slow ones of [the slowdown record](../2026-09-30-xe-swift-iq2xs/README.md), not a fair base.

## The long prompt

The old records' resident run: 26,292 tokens, 24 generated, `--prefill auto --max-context 32768 --kv int8` with MTP (`long-*`), one run each.

| Model | Prefill (tok/s) | Old machine |
| --- | --- | --- |
| IQ2_XS | 1,016 | 929 |
| IQ3_XXS | 939 | 812 |
| IQ3_S | 867 | 744 |
| Q2_0 | 1,108 | 877 |
| Coder IQ1_M | 1,235 | 1,110 |
| Swift IQ2_XS | 1,025 | 857 |
| Swift IQ3_XXS | 940 | 788 |
| Swift Q2_0 | 1,115 | 1,010 |

With `STRATA_PREFILL_TIMING=1` (`long-timing-*`; the mode's own accounting is the "host grouping" share, not real work): the GPU waited on copies for 0.5% of its timeline on IQ2_XS (2.3% on the old machine) and 3.2% on IQ3_S (4.6%). Dequantizing takes 11–14%, the expert GEMMs 9–18%, the prompt attention 14% (IQ2_XS); these are the next phase's targets.

## The expert arena and the PCIe share

The comparison of `records/2026-09-30-pcie-compare` again: the arena as one mapping (the default; GPU kernels cannot read it, so nothing goes over PCIe) against one host USM block per layer (`STRATA_ARENA_HOST_USM=1`), MTP with suffix drafts off, `--expert-cache 10000`, 128 tokens, two rounds alternating ([pcie.sh](pcie.sh), `pc-*`).

| Arena, mode, share | IQ2_XS (tok/s) | IQ3_S (tok/s) | Experts per layer over PCIe (IQ2_XS / IQ3_S) |
| --- | --- | --- | --- |
| mapping (default) | 45.20 / 44.47 | 34.89 / 34.91 | 0 / 0 |
| host USM, share 0 | 44.04 / 45.05 | 34.87 / 34.91 | 0 / 0 |
| host USM, kernel, 0.15 | 45.91 / 45.87 | 35.72 / 36.26 | 0.25 / 0.07 |
| host USM, kernel, 0.30 | 42.07 / 42.04 | 34.36 / 34.39 | 0.88 / 0.35 |
| host USM, kernel, 0.55 | 42.13 / 41.77 | 33.72 / 33.44 | 1.94 / 1.15 |
| host USM, direct, 0.30 | 42.39 / 42.50 | 34.33 / 34.31 | 0.88 / 0.35 |
| host USM, DMA, 0.30 | 24.88 / 24.83 | 26.94 / 26.64 | 0.88 / 0.35 |

- The best case, the kernel mode at 0.15, is 2–4% faster than the default; every larger share is slower, and DMA is far slower. On the old machine the same comparison gave +5% (IQ2_XS) and +1% (IQ3_S) at 0.15
- Why: the CPU's part of a round fell from about 25 ms (old machine) to 8–9 ms, while the GPU's is 43–50 ms. Moving experts from the CPU to the GPU over PCIe adds to the side that is already the longer one (the wait grows from 43 to 51 ms on IQ2_XS at 0.55)
- Host USM reading on the CPU is no longer slower than the mapping at share 0 (pool 8.6–9.3 ms against 8.8–9.1 ms on IQ2_XS)
- Each setting repeats its own tokens; different shares give different tokens (a different CPU/GPU split, as above)
- The engine's default share for native packs is 0.55 (from the probe), which this measurement does not support

## The output head's VRAM refusal

The refusal of `bench/results/2026-09-30-xe-iq3xxs` (the output head's upload refused at start) happened 3 times in about 120 starts: `p-iq2-mtp-1`, `w15-iq3-nd-1`, `fx-nd-1`. With the logging added in 383d827: 322–521 MB refused while the device reported 27.0–27.3 GiB free and the process held 3.4–3.9 GiB in about 306 blocks, so it is not a lack of VRAM. Every time, as in the first record, it is the first device allocation after the arena's mapping was registered for device copies. The next start with the same settings ran. Cause `unverified`.

## The grouped expert GEMM prototype

The prototype of `records/2026-09-30-prefill-gemm` (`grouped.cpp`, the same build and shapes: 16 experts, 2,904 rows), two runs ([gemm/grouped.txt](gemm/grouped.txt)):

| Shape | oneMKL, one call per expert | Grouped XMX, one launch | Old machine |
| --- | --- | --- | --- |
| gate/up, N 1280 K 2560 | 44.3 / 43.2 TFLOP/s | 62.0 / 62.2 TFLOP/s | 45.5 → 58.9 |
| down, N 2560 K 640 | 36.8 / 37.3 TFLOP/s | 51.4 / 49.8 TFLOP/s | 34.0 → 52.0 |

The GPU-only probe gives the same picture as before. The separate oneMKL probe ([gemm/gemm_group.txt](gemm/gemm_group.txt)) timed the per-expert down calls at 59.8 TFLOP/s, so how much the grouped kernel gains on the down shape depends on how the baseline is timed and is `unverified` until the kernel runs inside the engine.
