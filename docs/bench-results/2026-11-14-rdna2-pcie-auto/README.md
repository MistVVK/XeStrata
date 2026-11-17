# PcieAuto on gfx103x: the decode miss split from the CPU pool's speed (#4007)

Question: on an RX 6000 card, is the share refined from the CPU pool (`STRATA_PCIE_FRAC_DEFAULT=auto`, PcieAuto) faster than the
start-up share, which only looks at the link? Same build in both arms; the only difference is that variable.

## Result

Paired change, `auto` against unset, ten interleaved pairs per machine (median, range, pairs where `auto` was faster):

| Machine | Share (start -> refined) | 4K decode | 32K decode | Prompt 4K / 32K |
| --- | --- | --- | --- | --- |
| RX 6900 XT, Windows 11 | 0.35 -> 0.20 (9 starts), 0.25 (1) | **+15.1%** (+12.4 to +18.0), 10/10 | **+13.3%** (+8.8 to +18.6), 10/10 | +0.2% / 0.0% |
| RX 6900 XT, Linux | 0.39 -> 0.30 -> 0.25 | **+4.9%** (+3.0 to +5.4), 10/10 | **+1.0%** (+0.3 to +3.5), 10/10 | -0.1% / 0.0% |
| RX 6700 XT (gfx1031), Linux container, probe 2.8 GB/s (maintainers' check, 2026-11-16, see below) | 0.08, no change logged | -0.2% (-0.2 to +0.04), 1/3 | not run | 0.0% / not run |

Absolute medians (tok/s, unset -> auto): Windows decode 38.0 -> 43.8 at 4K and 39.0 -> 44.1 at 32K, prompts 460 / 469 in both
arms; Linux decode 45.0 -> 47.1 and 47.7 -> 48.1, prompts 490 / 519-520. 0 stalls in all 40 starts. The weakest Windows pair (+8.8%)
is the one start whose refinement stopped at 0.25.

A third machine, from the issue (kulight, RX 6700 XT gfx1031, Ryzen 7 5800X, PCIe 4.0 x16, Linux, 0.2.13, IQ2_XS, DFlash2):
0.55 -> 0.35, decode +4.9% at 4K (9/10) and +3.4% at 32K (10/10), prompts unchanged
([comment](https://github.com/Niko1221/Strata/issues/4007#issuecomment-6879926995)).

## Setup

| | Windows | Linux |
| --- | --- | --- |
| Card | RX 6900 XT 16 GB, PCIe 4.0 x8 (probe 12.6 GB/s), display on the iGPU | RX 6900 XT 16 GB, PCIe 4.0 x8 (probe 14.1 GB/s), one of two cards (`gpu: [1]`) |
| Host | i9-10900K, 64 GB DDR4-3600, Windows 11 | Ryzen 5 5600X, 128 GB DDR4-3200, Ubuntu 26.04, kernel 7.0 |
| Engine | release 0.2.13 package `windows-x86_64-hip-rdna2`, engine 0.1.59, unmodified (installed by `strata update`) | main 13eedf552 built from source (gfx1030, PREFILL_MMQ / Q6K_EXPERTS / MMQ_KQUANTS on), ROCm 10.0 |
| Server | `serve/server.py` from v0.2.13 | `serve/server.py` from 13eedf552 |

`generate.cpp` has no PcieAuto difference between v0.2.13 and 13eedf552.

Both: Qwen3.8-Flash-Next IQ3_S pack, `--expert-cache auto --prefill auto --spec 3 --spec-min-p 0.6` with the MTP drafter (`--mtp-q4 head`),
`--max-context 131072 --kv int8 --kv-resident 32768 --vram-reserve-mib 700 --pipeline-windows 2 --conversation-cache-mib 0`,
parallel 1, temperature 0. Linux also had `--ple-io ram`; Windows did not (with 64 GB the 27 GB PLE table left too little RAM for the
47 GB expert arena, so it was dropped in both Windows arms). No other `STRATA_*` variable was set.

## Method

Each arm is a fresh engine start: a warm-up, then synthetic prompts of 4,096 and 32,768 tokens (a local script, not BENCHMARK_PROTOCOL.md), two runs each, 256
output tokens, a nonce per request (nothing cached). Decode = 256 / (elapsed - time to first token) on the client, median of the two
runs; prompt = the engine's rate. Order A B B A, five times (20 starts per machine). An `auto` start counts only if its log has the
`PCIe share: the CPU pool takes ...` line (20 of 20 did); an unset start only has the probe line.

Outputs were not compared token by token (each request has its own nonce). PcieAuto moves missed experts between the GPU lane and
the CPU lane, which round differently, so output changes in the last bits, as with the gfx12 default.

## Files

- `windows-rx6900xt.csv`, `linux-rx6900xt.csv`: per start, decode and prompt rates, the final share, stalls.

## Maintainers' check on an RX 6700 XT (2026-11-16)

The engine of this change (f-1116-amd 3f83e96a1, engine 0.1.61, rx6000 build), RX 6700 XT 12 GB (gfx1031) in a Linux container on a Ryzen AI 9 HX 470 host
(ROCm 10.1), Qwen3.8-Flash-Next IQ2_XS pack, no drafter, `--kv int8 --max-context 8192`, one 4,000-token prompt and 128 output tokens per start. The new
default against `STRATA_PCIE_FRAC_DEFAULT=old`, three interleaved pairs (A B, B A, A B), the share left to the engine (`--pcie-frac -1`).

| | default | `=old` |
| --- | ---: | ---: |
| decode tok/s (128 tokens), per start | 19.43 / 19.41 / 19.40 | 19.42 / 19.45 / 19.44 |
| prompt tok/s, per start | 207.2 / 207.0 / 207.2 | 206.8 / 207.5 / 207.0 |

The link probe reads 2.8 GB/s on this card, so the start-up share is 0.08, already at the low end; no `PCIe share:` line was printed, and the 128 tokens and the
first-token logits were identical in all six starts. So on a slow link the change does nothing, and it costs nothing measurable (decode -0.2% to +0.04%).

