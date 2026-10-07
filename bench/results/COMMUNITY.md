# Community bench results, index

One row per community folder in this directory (ported for 0.1.40.2 from the community bench PRs, authors credited in the commits). Raw dumps, logs and binaries were trimmed from the folders; each trimmed folder has a `TRIMMED.md` listing what was left out. The README in each folder has the full method.

How to read the table:

- **Prompt** is prompt (prefill) tok/s; **Output** is decode tok/s. Cells with `/` give the 4K / 32K / 128K runs unless the cell says otherwise.
- Numbers are the submitter's own medians, taken from each folder's README. Different machines, prompts and engine versions: compare rows with care. Our own numbers for the same engine are in the dated reports above (`2026-09-29-speed-0126` and others).
- Strata is the engine version the submitter ran.
- PR is the original pull request (closed by the 2026-10-06 history rewrite; the work is in 0.1.40.2).

| Folder | Hardware | Model | Prompt tok/s | Output tok/s | Strata | PR |
|---|---|---|---|---|---|---|
| [2026-10-01-0.1.31-release](2026-10-01-0.1.31-release/) | RX 7900 XTX 24 GB, Ryzen 7 7700X | Coder IQ1_M | 545 to 900 median (0.1.30 to 0.1.31) | 16-34 to 17-65 | 0.1.30 / 0.1.31 | #404 |
| [2026-10-01-community-2x-rtx-pro-4500](2026-10-01-community-2x-rtx-pro-4500/) | 2x RTX PRO 4500 32 GB, Threadripper 7960X | Swift IQ3_XXS | 1,428 / 2,454 / 2,785 | 115 / 120 / 93 | 0.1.30 | #418 |
| [2026-10-02-community-2x-rtx-pro-4500-engine-0.1.36](2026-10-02-community-2x-rtx-pro-4500-engine-0.1.36/) | 2x RTX PRO 4500 32 GB, Threadripper 7960X | Swift IQ3_XXS | 2,960 / 5,065 / 5,762 | 127 / 130 / 105 | 0.1.36 | #418 |
| [2026-10-01-community-2x-titan-rtx](2026-10-01-community-2x-titan-rtx/) | 2x TITAN RTX 24 GB (NVLink), Xeon E5-2696 v4 | IQ3_S, 262K | 865 / 1,612 / 1,785 | 68 / 60 / 62 | 0.1.31 | #389 |
| [2026-10-01-community-rtx-pro-4500](2026-10-01-community-rtx-pro-4500/) | RTX PRO 4500 32 GB, Threadripper PRO 3975WX | UD-Q4_K_XL / UD-IQ4_XS / IQ3_S | see README | UD-Q4_K_XL 61.5 / 72.1 / 70.2; UD-IQ4_XS 82.0 / 95.8 / 91.4; IQ3_S 106 / 120 / 117 (2.7K / 64K / 128K) | 0.1.31 | #417 |
| [2026-10-01-community-rtx-5090-5950x](2026-10-01-community-rtx-5090-5950x/) | RTX 5090, Ryzen 9 5950X (AVX2), 96 GB | UD-Q4_K_XL, IQ3_S, Swift IQ3_XXS | UD-Q4_K_XL 14.7K: 2,161 (0.1.31) | UD-Q4_K_XL 14.7K: 88.3 | 0.1.31 to 0.1.39 | #433 |
| [2026-10-01-ctx-ladder](2026-10-01-ctx-ladder/) | RTX 4070 Ti SUPER 16 GB (docker) | IQ2_XS | 167 / 128 / 133 (SSE prefill, 32K / 64K / 128K) | 97 / 84 / 74 (32K / 64K / 128K, KV q4_0 at 128K) | see README | #995 |
| [2026-10-05-community-rtx-4070-ti-super](2026-10-05-community-rtx-4070-ti-super/) | RTX 4070 Ti SUPER 16 GB, Ryzen 7 9800X3D | IQ2_XS / IQ3_XXS | cold prefill in README (about 2,500 at 32K-256K) | IQ2_XS 97.9 / 94.3 / 90.0 / 82.4; IQ3_XXS 76 / 71 / 70 / 56 (32K / 64K / 128K / 256K) | 0.1.39 (6f32ec0) | #995 |
| [2026-10-01-iq4xs-64gb](2026-10-01-iq4xs-64gb/) | RTX 5060 Ti 16 GB (PCIe 4.0 x8), Ryzen 7 5700X3D, 64 GB | UD-IQ4_XS (60.9 GiB of experts) | n/a (ms per draft round) | 33.9 pinned arena, 16.9-39 mmap variants | 0.1.31 / 0.1.32 | #416 |
| [2026-10-02-community-2x5060ti](2026-10-02-community-2x5060ti/) | 2x RTX 5060 Ti 16 GB (PCIe gen3), i9-9900KF | IQ3_S | 1,108 (20.9K cold) | 56.4 (20.9K); 62.9 short | 0.1.34 | #483 |
| [2026-10-02-community-rtx-5060-ti-x2](2026-10-02-community-rtx-5060-ti-x2/) | 2x RTX 5060 Ti 16 GB, Xeon E5-2690 v4 | IQ3_XXS, 262K | 808 / 1,931 / 2,199 | 77.5 / 72.7 / 72.4 | 0.1.35 | #508 |
| [2026-10-05-community-rtx-5060-ti-x2](2026-10-05-community-rtx-5060-ti-x2/) | 2x RTX 5060 Ti 16 GB, Xeon E5-2690 v4 | IQ3_XXS, 262K (rerun) | 911 / 2,087 / 2,440 | 84.8 / 86.5 / 83.8 | 0.1.39 | #508 |
| [2026-10-02-community-rtx-5090-iq3s](2026-10-02-community-rtx-5090-iq3s/) | RTX 5090, Ryzen 9 9950X3D, 96 GB | IQ3_S, 262K | 2,473 / 4,168 / 4,450 (4K / 32K / 131K) | 141.8 / 142.2 / 127.2 | 0.1.33 | #440 |
| [2026-10-02-community-rtx2080ti-11gb](2026-10-02-community-rtx2080ti-11gb/) | RTX 2080 Ti 11 GB, Threadripper 3960X, 128 GB | IQ3_S, 262K | 497 / 637 / 566 | 39.6 / 38.0 / 34.1 | 0.1.33 | #469 |
| [2026-10-02-community-rtx4000ada-layer-split](2026-10-02-community-rtx4000ada-layer-split/) | 2x RTX 4000 Ada (+ 1x SFF Ada), EPYC 7313P | IQ3_S | 2,110 (2 GPU) vs 779 (3 GPU), 16.6K | 73 vs 74 median | 0.1.35 | #521 |
| [2026-10-02-community-rtx5090-1m-agent](2026-10-02-community-rtx5090-1m-agent/) | RTX 5090, Ryzen 9 7950X | IQ3_S, 1M context (YaRN) | about 2,500 / 3,300 / 3,400 (4K / 32K / 128K) | 94-105 | 0.1.31 | #466 |
| [2026-10-02-community-rx9070xt-windows](2026-10-02-community-rx9070xt-windows/) | RX 9070 XT 16 GB, Windows 11 | IQ3_S | about 600 | 45-46 (37 first run) | 0.1.35 (AMD engine) | #499 |
| [2026-10-03-ashley-2080ti](2026-10-03-ashley-2080ti/) | RTX 2080 Ti, i9-9900KF, Windows 11 | IQ3_XXS | 525 at 32K | 33-38 (1K to 128K) | 0.1.38 | #698 |
| [2026-10-03-dev-5090](2026-10-03-dev-5090/) | RTX 5090, Ryzen 9 9950X3D | IQ3_S (0.1.34 / 0.1.38 / `STRATA_PF_FUSED=1`) | 4K: 4,308 / 4,846 / 5,361 | 4K: 178 / 193 / 189; 128K: 144 / 152 / 165 | 0.1.34, 0.1.38 | #698 |
| [2026-10-03-community-r9700-windows](2026-10-03-community-r9700-windows/) | Radeon AI PRO R9700 (gfx1201), Ryzen 9 9950X, Windows 11 | IQ2_XS | 856 (4K) / 1,182 (32K) / 1,232 (131K) / 1,130 (248K) | 99.8 (4K) / 97.8 / 89.9 / 87.5 | 0.1.38 | #618 |
| [2026-10-03-community-rtx4090-iq3s-140k-code](2026-10-03-community-rtx4090-iq3s-140k-code/) | RTX 4090, Ryzen 9 7950X | IQ3_S, 140K, code prompts | 2,059 / 2,948 / 3,048 | 77.5 / 97.2 / 84.4 | 0.1.38 | #624 |
| [2026-10-03-community-rtx4090-iq3s-140k-ru](2026-10-03-community-rtx4090-iq3s-140k-ru/) | RTX 4090, Ryzen 9 7950X | IQ3_S, 140K, Russian prose | 2,084 / 2,991 / 3,078 | 76.1 / 102.5 / 102.2 | 0.1.38 | #624 |
| [2026-10-03-community-rtx4090-iq3xxs-200k-code](2026-10-03-community-rtx4090-iq3xxs-200k-code/) | RTX 4090, Ryzen 9 7950X | IQ3_XXS, 200K, code prompts | 1,844 / 2,976 / 2,962 | 118 / 128 / 108 | 0.1.38 | #624 |
| [2026-10-03-community-rtx4090-iq3xxs-200k-ru](2026-10-03-community-rtx4090-iq3xxs-200k-ru/) | RTX 4090, Ryzen 9 7950X | IQ3_XXS, 200K, Russian prose | 1,824 / 2,912 / 2,984 | 94.2 / 85.6 / 90.3 | 0.1.38 | #624 |
| [2026-10-04-community-rtx4090-iq3xxs-200k-code](2026-10-04-community-rtx4090-iq3xxs-200k-code/) | RTX 4090, Ryzen 9 7950X | IQ3_XXS, 200K, code prompts | 1,966 / 2,974 / 3,085 | 124 / 135 / 120 | 0.1.39 | #624 |
| [2026-10-04-community-rtx4090-iq3xxs-200k-ru](2026-10-04-community-rtx4090-iq3xxs-200k-ru/) | RTX 4090, Ryzen 9 7950X | IQ3_XXS, 200K, Russian prose | 1,908 / 3,128 / 3,126 | 81.1 / 90.5 / 87.8 | 0.1.39 | #624 |
| [2026-10-03-rx6900xt-coder](2026-10-03-rx6900xt-coder/) | RX 6900 XT, Threadripper 3990X (Windows) | Coder IQ1_M | 313 (4K) / 350 (32K) | 45.3 (4K) with 31 workers vs 11.6 with 63 | 0.1.38 | #628 |
| [2026-10-03-windows-groups](2026-10-03-windows-groups/) | RX 6900 XT, Threadripper 3990X (Windows) | Coder IQ1_M | see README | 12.7 to 46.1 (short), 13.1 to 47.4 (4K) with processor groups kept | 0.1.38 + fix | #628 |
| [2026-10-03-community-rx7900xtx-rocm102-nightly](2026-10-03-community-rx7900xtx-rocm102-nightly/) | RX 7900 XTX 24 GB, ROCm 7.1.1 vs 10.2 nightly | IQ3_S | 921 vs 1,535 (without / with matching hipBLASLt) | 62.0 vs 61.0 | 0.1.38 | #745 |
| [2026-10-04-community-rx7900xtx-hipblaslt-100500](2026-10-04-community-rx7900xtx-hipblaslt-100500/) | RX 7900 XTX 24 GB, ROCm 10.2 nightly | IQ3_S, 128K | 926 to 1,687 (gfx1100 table) | 62.3 | 0.1.38 lineage | #745 |
| [2026-10-03-community-v100](2026-10-03-community-v100/) | Tesla V100-PCIE 32 GB, 2x Xeon E5-2696 v3, Windows | Flash-Next (see README) | see README | 16.5 to 33.5 after calibration; 22.0 prose | 0.1.38, updated for 0.1.39 | #707 |
| [2026-10-03-community-z840-rtx3090ti](2026-10-03-community-z840-rtx3090ti/) | RTX 3090 Ti, 2x Xeon E5-2699 v3 (HP Z840) | IQ3_S, 262K | 1,239-1,304 (about 5K) | 90-99 (NUMA variants) | 0.1.38 | #674 |
| [2026-10-04-community-2x-rtx3060](2026-10-04-community-2x-rtx3060/) | 2x RTX 3060 12 GB, Xeon E5-2678 v3 | IQ3_S | 603 / 1,156 / 1,323 (3.2K / 24.8K / 98.6K) | 43.6 / 42.5 / 40.0 | 0.1.37 (source) | #721 |
| [2026-10-04-community-rx-7900-gre](2026-10-04-community-rx-7900-gre/) | RX 7900 GRE 16 GB, Ryzen 7 5700X3D | IQ2_XS / IQ3_XXS / Coder IQ1_M | 1,140-1,216 (8.8K) | 65.3 / 59.1 / 54.7 | 0.1.38 | #740 |
| [2026-10-04-community-mi50](2026-10-04-community-mi50/) | MI50 (gfx906), i5-12400F | IQ2_XS | about 330 / 330 / 305 | 36.0 / 34.0 / 33.5 | 0.1.38 | #758 |
| [2026-10-04-community-rtx-4090-laptop](2026-10-04-community-rtx-4090-laptop/) | RTX 4090 Laptop, i9-13900HX | IQ3_S | 1,345 / 2,506 / 2,386 | 51.2 / 49.9 / 48.1 | 0.1.38 | #757 |
| [2026-10-04-community-rtx-5090-9700x](2026-10-04-community-rtx-5090-9700x/) | RTX 5090, Ryzen 7 9700X, Windows 11 | IQ3_XXS | 3,775 (median, fresh) | 171.7 | 0.1.39 | #777 |
| [2026-10-05-community-rtx-5090-9700x-iq3s](2026-10-05-community-rtx-5090-9700x-iq3s/) | RTX 5090, Ryzen 7 9700X, Windows 11 | IQ3_S | 3,072 (median, fresh) | 164.3 | 0.1.39 | #811 |
| [2026-10-04-community-rtx-4080s-iq3s](2026-10-04-community-rtx-4080s-iq3s/) | RTX 4080 SUPER, i7-13790F, Windows 11 | IQ3_S | cold 2,535 (pool 4) vs 2,486 (pool 15) | 95 (4 workers) vs 69-78 (15 workers) mean | 0.1.39 | #780 |
| [2026-10-04-community-rx-6800-windows](2026-10-04-community-rx-6800-windows/) | RX 6800, Windows | Flash-Next (see README) | 326 (7.9K) / 311 (30.3K) | 42.6 to 56.0 English (0.1.39 vs `STRATA_SH_STREAM=0`) | 0.1.38 / 0.1.39 | #815 |
| [2026-10-04-community-v100-16gb-ram](2026-10-04-community-v100-16gb-ram/) | Tesla V100-PCIE 32 GB (x4 link), i7-13700KF, 16 GB RAM | Coder IQ1_M | 1,233 / 1,580 / 1,394 | 69.3 / 68.6 / 67.1 | 0.1.39 | #823 |
| [2026-10-04-community-rtx-5070ti-windows](2026-10-04-community-rtx-5070ti-windows/) | RTX 5070 Ti, Ryzen 7 9800X3D, Windows 11 | IQ3_S | 3,012 (5.2K) / 3,487 (41K) / 3,523 (131K) / 3,442 (164K) | 75.8 / 74.7 / 82.8 / 79.6 | 0.1.39 | #832 |
| [2026-10-04-community-4090-32gb-ram-512k](2026-10-04-community-4090-32gb-ram-512k/) | RTX 4090 24 GB, 32 GB RAM | IQ2_XS, 512K | 3,411 (477K prompt) | 122.0 at depth | 0.1.39 | #834 |
| [2026-10-04-community-v100-rtx4070-split](2026-10-04-community-v100-rtx4070-split/) | Tesla V100 32 GB + RTX 4070 12 GB, 16 GB RAM | IQ2_XS | 985 / 1,535 / 1,533 | 79.1 / 83.8 / 77.1 | 0.1.39 | #850 |
| [2026-10-04-community-4090-agents-vs-dense-27b](2026-10-04-community-4090-agents-vs-dense-27b/) | RTX 4090 24 GB, 32 GB RAM | IQ2_XS vs a dense 27B (coding-agent task) | about 3,800 (70-110K re-reads) | 173-185 | 0.1.39 + 4 PRs | #882 |
| [2026-10-05-community-v100-sxm2](2026-10-05-community-v100-sxm2/) | Tesla V100-SXM2 32 GB, Windows (CUDA 12.6) | Q2_0 / UD-IQ4_XS / third quant (see README) | Q2_0: 1,268 (8K) | Q2_0 76.4; UD-IQ4_XS 26.3 | 0.1.39 (6f32ec0) | #902 |
| [2026-10-05-community-e5-2673v3-rtx-4060ti](2026-10-05-community-e5-2673v3-rtx-4060ti/) | RTX 4060 Ti 16 GB, Xeon E5-2673 v3 (DDR3, PCIe 3.0 x8) | Q2_0 / IQ3_S (two arms each) | 715 (Q2_0) / 439 (IQ3_S), 3.5K | Q2_0 48.1 to 55.7; IQ3_S 30.5 to 35.0 | 0.1.39 | #913 |
| [2026-10-06-community-rtx5070ti-5900x](2026-10-06-community-rtx5070ti-5900x/) | RTX 5070 Ti, Ryzen 9 5900X, Windows 10 | IQ2_XS / IQ3_XXS / IQ3_S | IQ2_XS 1,992 / 2,817; IQ3_XXS 1,653 / 2,482; IQ3_S 1,439 / 2,261 | IQ2_XS 102 / 101; IQ3_XXS 72.5 / 73.7; IQ3_S 62.9 / 66.1 | 0.1.39 | #993 |
| [2026-10-06-community-rtx-5060ti](2026-10-06-community-rtx-5060ti/) | RTX 5060 Ti 16 GB, Ryzen 9 7945HX | IQ3_S | 1,158 / 1,225 / 1,176 | 56.6 / 60.2 / 57.2 | 0.1.35 (d9ab843) | #1016 |
| [2026-10-05-community-rx-6700-xt](2026-10-05-community-rx-6700-xt/) | RX 6700 XT 12 GB (gfx1031), Ryzen 7 5700G | Flash-Next (see README) | 302 / 309 / 298 | 32.1 / 30.7 / 30.2 | 0.1.39 | #1027 |
| [2026-10-05-community-2x-p40](2026-10-05-community-2x-p40/) | 2x Tesla P40 22 GB, 2x Xeon E5-2698 v4 | IQ2_XS | 353 (4K) / 501 (16K), both cards; 348 / 388 one card | 34.8 both cards, 19.6 one card | 0.1.39 | #1028 |
| [2026-10-05-community-2x-rtx-4060ti](2026-10-05-community-2x-rtx-4060ti/) | 2x RTX 4060 Ti 16 GB, Threadripper PRO 3975WX | UD-IQ4_XS | 880 / 1,917 / 2,221 | 54.4 / 56.6 / 54.7 | 0.1.39 | #1062 |
| [2026-10-06-community-rx-6800m](2026-10-06-community-rx-6800m/) | RX 6800M (gfx1031), Windows, self-built HIP | Flash-Next quants (see README) | 50-114 | 9-27 | 0.1.40 | #1078 |
| [2026-10-05-community-2x-rx-6900xt](2026-10-05-community-2x-rx-6900xt/) | 2x RX 6900 XT 16 GB (gfx1030), Ryzen 5 5600X, 128 GB | IQ3_S, 131K | one card stock 457 / 474 / 459, with #835 + #849 + #854 868 / 1,100 / 1,023; layer split with them 853 / 1,616 / 1,823 | 40-43 one card, 57-61 layer split | 0.1.39 stock and with #835, #849, #854 | #927 |
| [2026-10-05-community-gfx1151](2026-10-05-community-gfx1151/) | Radeon 8060S (gfx1151), Ryzen AI Max+ 395, 128 GB unified memory | Q2_0, UD-IQ4_XS | 250.9 to 528.9 with the hipBLASLt table (matched synthetic prompts) | 45.6 to 46.8 (shared-expert stream on) | 0.1.39 + #895, #820 | #917 |
| [2026-10-05-community-arc-b65](2026-10-05-community-arc-b65/) | Intel Arc Pro B65 32 GB (SYCL), Core i5-12600K, 128 GB | IQ2_XS, 8K | 307-359 (512 and 7,000-token tasks) | 36-49 by task; medians of the five tasks 40.4-41.3 / 39.8-40.9 | 0.1.40 source + 6 local patches | #955 |

Notes:

- #927 ran seven configurations; the table shows the one-card stock and PR sets and the layer split with the PRs (#835 is opt-in in 0.1.40, so those are not the release default), and its README has the helper-mode rows. #917 is 0.1.39 plus then-unmerged #895 and #820, and 0.1.40 turned the shared-expert stream off on HIP (#826). #955's patches 3, 5 and 6 are source changes in 0.1.40.2.
- #1016 ran its engine at the 0.1.35 commit and the table shows the numbers as submitted.
- #995 also changed the canonical `2026-09-29-speed-0126` report (matrix.json rewritten, README extended with another host's rows); that part was not ported, only the two new folders.
- #777 and #811 are the same machine on two quants, kept as two folders.
- The earlier community folders (`2026-09-30-community-rtx-5090`, `2026-10-03-community-2x-mi50`, `2026-10-04-community-2x-arc-pro-b60`) are listed in `docs/COMMUNITY_BENCHMARKS.md`.
| [2026-10-06-community-rtxpro-v0140](2026-10-06-community-rtxpro-v0140/) | RTX PRO 6000 | Q4 / Q8 (MTP and ngram runs) | see the README | see the README | 0.1.40 | #1137 |
| [2026-10-06-community-2x-rx-7900-gre](2026-10-06-community-2x-rx-7900-gre/) | 2x RX 7900 GRE (gfx1100) | layer split, #848 / #854 | see the README | see the README | 0.1.40 | #1140 |
| [2026-10-06-community-2x-p100](2026-10-06-community-2x-p100/) | 2x Tesla P100 16 GB | Flash-Next IQ3_S, 128K | see the README | see the README | 0.1.40 | #1157 |
| [2026-10-06-community-rtx4090-opt-ins](2026-10-06-community-rtx4090-opt-ins/) | RTX 4090 | IQ3_S 140K / IQ3_XXS 200K opt-ins and tool-call hotfix | see the README | see the README | 0.1.38 to 0.1.40 | #1158 |
| [2026-10-06-community-rtx4090-toolcall-hotfix](2026-10-06-community-rtx4090-toolcall-hotfix/) | RTX 4090 | IQ3_S 140K / IQ3_XXS 200K opt-ins and tool-call hotfix | see the README | see the README | 0.1.38 to 0.1.40 | #1158 |
| [2026-10-06-q8-prefill8192-rtxpro](2026-10-06-q8-prefill8192-rtxpro/) | RTX PRO 6000 | Q8, --prefill 8192 | see the README | see the README | 0.1.40 | #1171 |
| [2026-10-06-community-rx-7900-xtx](2026-10-06-community-rx-7900-xtx/) | RX 7900 XTX, Windows 11 | IQ3_S | see the README | see the README | 0.1.40 | #1173 |
| [2026-10-06-community-4x-tesla-p100](2026-10-06-community-4x-tesla-p100/) | 4x Tesla P100 16 GB (CUDA 12 engine) | IQ3_XXS | see the README | see the README | 0.1.40 | #1192 |
| [2026-10-06-community-rtx3090-egpu-64gb](2026-10-06-community-rtx3090-egpu-64gb/) | RTX 3090 eGPU, 64 GB RAM | IQ2_XS vs IQ3_XXS | see the README | see the README | 0.1.40 | #1199 |
| [2026-10-06-community-r9-2x-rtx-2080-ti](2026-10-06-community-r9-2x-rtx-2080-ti/) | 2x RTX 2080 Ti (R9) | see README | see the README | see the README | 0.1.40 | #1225 |
| [2026-10-06-community-2x-rtx-pro-4500-engine-0.1.40.1](2026-10-06-community-2x-rtx-pro-4500-engine-0.1.40.1/) | 2x RTX PRO 4500 Blackwell | Swift IQ3_XXS | see the README | see the README | 0.1.40.1 | #1226 |
| [2026-10-05-community-rtx5080-9900x3d](2026-10-05-community-rtx5080-9900x3d/) | RTX 5080, Ryzen 9 9900X3D | Coder IQ1_M and others | see the README | see the README | 0.1.40 | #1243 |
| [2026-10-07-community-rtx-5090-laptop-ud-iq4xs](2026-10-07-community-rtx-5090-laptop-ud-iq4xs/) | RTX 5090 Laptop 24 GB | UD-IQ4_XS (first NVIDIA measurement) | see the README | see the README | 0.1.40.1 | #1245 |
| [2026-10-06-community-rtx-5090-laptop](2026-10-06-community-rtx-5090-laptop/) | RTX 5090 Laptop 24 GB, Windows 11 | see README | see the README | see the README | 0.1.40.1 | #1263 |
| [2026-10-06-community-2x-rx-6900xt-0.1.40.1](2026-10-06-community-2x-rx-6900xt-0.1.40.1/) | 2x RX 6900 XT (gfx1030), Ryzen 5 5600X | IQ3_S, 131K | see the README | see the README | 0.1.40.1 | #1270 |
