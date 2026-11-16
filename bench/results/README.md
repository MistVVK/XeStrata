# Benchmark results

Raw results and short reports from measurement runs, one folder per run, named by date. They stay here as they were written: each states
the Strata version, hardware and method of its own day, and none is a statement about a later release. Current numbers are in the
[docs](../../docs/README.md) and the [README](../../README.md); how to measure is in [BENCHMARK_PROTOCOL.md](../../docs/BENCHMARK_PROTOCOL.md).

This page is built by `python tools/bench_results_index.py` (`--check` fails when a folder is missing from it). 143 folders and 3 loose files, 2188 files in all.


## 2026-09

| Folder | What | Files |
| --- | --- | ---: |
| [2026-09-24-final](2026-09-24-final/) | Output (decode) tok/s - estimated | 4 |
| [2026-09-24-vision](2026-09-24-vision/) | (no Markdown page) | 2 |
| [2026-09-27-cache-parity](2026-09-27-cache-parity/) | 2026-09-27: expert cache on vs off (issue #23) | 1 |
| [2026-09-27-esp](2026-09-27-esp/) | 2026-09-27: the experimental-speed-projection control vector | 5 |
| [2026-09-27-kv-q4](2026-09-27-kv-q4/) | 2026-09-27: Q4_0 KV with Hadamard rotation (PR #21, --kv q4_0) | 1 |
| [2026-09-28-coder](2026-09-28-coder/) | The GSQ-RCO Coder (256 of 512 experts) on Strata | 2 |
| [2026-09-28-prefill-speed](2026-09-28-prefill-speed/) | Faster prompt reading (engine 0.1.13) | 1 |
| [2026-09-28-rope-scaling](2026-09-28-rope-scaling/) | Rope scaling: needle recall past the trained 262K | 10 |
| [2026-09-28-speed-0114](2026-09-28-speed-0114/) | Speed with engine 0.1.14 | 3 |
| [2026-09-29-layer-split](2026-09-29-layer-split/) | Layer split across GPUs (2026-09-29) | 35 |
| [2026-09-29-layer-split-limits](2026-09-29-layer-split-limits/) | Layer split: the features that were off across GPUs (2026-09-29) | 15 |
| [2026-09-29-rtx3090-epyc-milan](2026-09-29-rtx3090-epyc-milan/) | Preliminary native RTX 3090 / EPYC 7453 results | 60 |
| [2026-09-29-speed-0122](2026-09-29-speed-0122/) | Speed with engine 0.1.22 | 2 |
| [2026-09-29-speed-0126](2026-09-29-speed-0126/) | Speed with engine 0.1.26 | 2 |
| [2026-09-30-community-rtx-5090](2026-09-30-community-rtx-5090/) | Community benchmark: RTX 5090, Core Ultra 9 285K | 18 |
| [2026-09-30-tc-gemv-sm75](2026-09-30-tc-gemv-sm75/) | BF16 GEMV on Turing tensor cores (sm_75): measured negative | 2 |

## 2026-10

| Folder | What | Files |
| --- | --- | ---: |
| [2026-10-01-0.1.31-release](2026-10-01-0.1.31-release/) | Strata 0.1.31 vs 0.1.30 — RX 7900 XTX (gfx1100), 2026-10-01 | 3 |
| [2026-10-01-community-2x-rtx-pro-4500](2026-10-01-community-2x-rtx-pro-4500/) | Community benchmark: 2x RTX PRO 4500 Blackwell, Swift IQ3_XXS | 10 |
| [2026-10-01-community-2x-titan-rtx](2026-10-01-community-2x-titan-rtx/) | Community benchmark: 2x TITAN RTX (sm_75, NVLink), Xeon E5-2696 v4 | 9 |
| [2026-10-01-community-rtx-5090-5950x](2026-10-01-community-rtx-5090-5950x/) | Community benchmark on RTX 5090 + Ryzen 9 5950X (AVX2): UD-Q4_K_XL, IQ3_S and Swift IQ3_XXS, engines 0.1.31 to 0.1.39 | 89 |
| [2026-10-01-community-rtx-pro-4500](2026-10-01-community-rtx-pro-4500/) | Community benchmark: RTX PRO 4500 Blackwell, Threadripper PRO 3975WX | 174 |
| [2026-10-01-ctx-ladder](2026-10-01-ctx-ladder/) | 2026-10-01 — Strata IQ2_XS context-ladder TPS on RTX 4070 Ti SUPER | 14 |
| [2026-10-01-iq4xs-64gb](2026-10-01-iq4xs-64gb/) | 2026-10-01: IQ4_XS on a 64 GB PC — pinned arena vs experts.bin mmap vs the RAM budget | 1 |
| [2026-10-02-community-2x-rtx-pro-4500-engine-0.1.36](2026-10-02-community-2x-rtx-pro-4500-engine-0.1.36/) | Community benchmark on 2x RTX PRO 4500 Blackwell, Swift IQ3_XXS, engine 0.1.36 | 15 |
| [2026-10-02-community-2x5060ti](2026-10-02-community-2x5060ti/) | Community benchmark on 2× RTX 5060 Ti 16 GB (PCIe gen3) | 4 |
| [2026-10-02-community-rtx-5060-ti-x2](2026-10-02-community-rtx-5060-ti-x2/) | Community benchmark: 2x RTX 5060 Ti (16 GB), Xeon E5-2690 v4 | 10 |
| [2026-10-02-community-rtx-5090-iq3s](2026-10-02-community-rtx-5090-iq3s/) | Community-format benchmark: RTX 5090, Ryzen 9 9950X3D, IQ3_S at 262K | 8 |
| [2026-10-02-community-rtx2080ti-11gb](2026-10-02-community-rtx2080ti-11gb/) | Community benchmark on RTX 2080 Ti 11 GB | 29 |
| [2026-10-02-community-rtx4000ada-layer-split](2026-10-02-community-rtx4000ada-layer-split/) | Community benchmark: RTX 4000 Ada 2-GPU vs 3-GPU layer split | 6 |
| [2026-10-02-community-rtx5090-1m-agent](2026-10-02-community-rtx5090-1m-agent/) | Community benchmark: RTX 5090, Ryzen 9 7950X, 1M-context agent workload | 5 |
| [2026-10-02-community-rx9070xt-windows](2026-10-02-community-rx9070xt-windows/) | Community results: RX 9070 XT 16 GB on Windows 11, the ready-made AMD engine (0.1.35), IQ3_S | 5 |
| [2026-10-02-gdn-chunk](2026-10-02-gdn-chunk/) | v100/gdn-chunk: chunked GDN prompt recurrence and Volta recurrence variants (2026-10-02) | 1 |
| [2026-10-03-ashley-2080ti](2026-10-03-ashley-2080ti/) | Ashley-PC: RTX 2080 Ti / i9-9900KF, Qwen3.8-Flash-Next IQ3_XXS | 17 |
| [2026-10-03-community-2x-mi50](2026-10-03-community-2x-mi50/) | Community benchmark: 2x AMD Instinct MI50 16 GB (gfx906), Xeon E5-2666 v3 | 8 |
| [2026-10-03-community-r9700-windows](2026-10-03-community-r9700-windows/) | Community benchmark: Radeon AI PRO R9700 (gfx1201), Windows 11, Ryzen 9 9950X | 21 |
| [2026-10-03-community-rtx4090-iq3s-140k-code](2026-10-03-community-rtx4090-iq3s-140k-code/) | Community benchmark: NVIDIA GeForce RTX 4090 | 4 |
| [2026-10-03-community-rtx4090-iq3s-140k-ru](2026-10-03-community-rtx4090-iq3s-140k-ru/) | Community benchmark: NVIDIA GeForce RTX 4090 | 4 |
| [2026-10-03-community-rtx4090-iq3xxs-200k-code](2026-10-03-community-rtx4090-iq3xxs-200k-code/) | Community benchmark: NVIDIA GeForce RTX 4090 | 5 |
| [2026-10-03-community-rtx4090-iq3xxs-200k-ru](2026-10-03-community-rtx4090-iq3xxs-200k-ru/) | Community benchmark: NVIDIA GeForce RTX 4090 | 5 |
| [2026-10-03-community-rx7900xtx-rocm102-nightly](2026-10-03-community-rx7900xtx-rocm102-nightly/) | Community benchmark on AMD Radeon RX 7900 XTX 24 GB — ROCm 7.1.1 vs ROCm 10.2 nightly (serve path, Qwen3.8-Flash-Next IQ3_S) | 4 |
| [2026-10-03-community-v100](2026-10-03-community-v100/) | 2026-10-03: Tesla V100-PCIE-32GB, dual Xeon E5-2696 v3, 128 GB RAM (Windows) | 1 |
| [2026-10-03-community-z840-rtx3090ti](2026-10-03-community-z840-rtx3090ti/) | Community benchmark on RTX 3090 Ti + dual Xeon E5-2699 v3 (HP Z840) | 3 |
| [2026-10-03-dev-5090](2026-10-03-dev-5090/) | IQ3_S single-GPU results, RTX 5090 / Ryzen 9 9950X3D | 50 |
| [2026-10-03-rx6900xt-coder](2026-10-03-rx6900xt-coder/) | Coder IQ1_M on RX 6900 XT and Threadripper 3990X | 52 |
| [2026-10-03-v100-kv-int8-read](2026-10-03-v100-kv-int8-read/) | V100 / Swift IQ2_XS / 32k / --kv int8: the KV read path | 1 |
| [2026-10-03-v100-prompt-attn](2026-10-03-v100-prompt-attn/) | V100 (sm_70): a tensor-core QSA prompt attention (2026-10-03) | 1 |
| [2026-10-03-windows-groups](2026-10-03-windows-groups/) | Threadripper 3990X / RX 6900 XT: Windows processor-group benchmark | 66 |
| [2026-10-04-community-2x-arc-pro-b60](2026-10-04-community-2x-arc-pro-b60/) | Community benchmark on 2x Intel Arc Pro B60 | 14 |
| [2026-10-04-community-2x-rtx3060](2026-10-04-community-2x-rtx3060/) | Community benchmark: 2× RTX 3060 12 GB (Xeon E5-2678 v3), Qwen3.8-Flash-Next IQ3_S | 30 |
| [2026-10-04-community-4090-32gb-ram-512k](2026-10-04-community-4090-32gb-ram-512k/) | Community benchmark: RTX 4090 24 GB with 32 GB of system RAM, 512K context (IQ2_XS) | 5 |
| [2026-10-04-community-4090-agents-vs-dense-27b](2026-10-04-community-4090-agents-vs-dense-27b/) | Community report: Flash-Next IQ2_XS on Strata vs a dense 27B, as coding agents on one RTX 4090 + 32 GB RAM | 7 |
| [2026-10-04-community-mi50](2026-10-04-community-mi50/) | Community benchmark: AMD MI50 (gfx906), Intel i5-12400F | 13 |
| [2026-10-04-community-rtx-4080s-iq3s](2026-10-04-community-rtx-4080s-iq3s/) | Community benchmark: RTX 4080 SUPER (32 GB), Core i7-13790F | 7 |
| [2026-10-04-community-rtx-4090-laptop](2026-10-04-community-rtx-4090-laptop/) | Community benchmark: RTX 4090 Laptop GPU, Core i9-13900HX | 15 |
| [2026-10-04-community-rtx-5070ti-windows](2026-10-04-community-rtx-5070ti-windows/) | Community benchmark: RTX 5070 Ti, Ryzen 7 9800X3D, Windows 11 | 6 |
| [2026-10-04-community-rtx-5090-9700x](2026-10-04-community-rtx-5090-9700x/) | Community benchmark: RTX 5090, Ryzen 7 9700X | 21 |
| [2026-10-04-community-rtx4090-iq3xxs-200k-code](2026-10-04-community-rtx4090-iq3xxs-200k-code/) | Community benchmark: NVIDIA GeForce RTX 4090 | 5 |
| [2026-10-04-community-rtx4090-iq3xxs-200k-ru](2026-10-04-community-rtx4090-iq3xxs-200k-ru/) | Community benchmark: NVIDIA GeForce RTX 4090 | 5 |
| [2026-10-04-community-rx-6800-windows](2026-10-04-community-rx-6800-windows/) | Community benchmark: RX 6800 (Windows), 0.1.39 against 0.1.38 | 10 |
| [2026-10-04-community-rx-7900-gre](2026-10-04-community-rx-7900-gre/) | Community benchmark: RX 7900 GRE (gfx1100, 16 GB), Ryzen 7 5700X3D | 34 |
| [2026-10-04-community-rx7900xtx-hipblaslt-100500](2026-10-04-community-rx7900xtx-hipblaslt-100500/) | Community benchmark on AMD Radeon RX 7900 XTX 24 GB — follow-up: a gfx1100 hipBLASLt 100500 table, the 128K prefill confound resolved | 6 |
| [2026-10-04-community-v100-16gb-ram](2026-10-04-community-v100-16gb-ram/) | Community benchmark on Tesla V100-PCIE-32GB with 16 GB of system RAM | 6 |
| [2026-10-04-community-v100-rtx4070-split](2026-10-04-community-v100-rtx4070-split/) | Community benchmark: Tesla V100 32 GB + RTX 4070 12 GB layer split, 16 GB of system RAM | 10 |
| [2026-10-04-rdna2-fp16-prompt](2026-10-04-rdna2-fp16-prompt/) | RX 6900 XT (gfx1030): the prompt path's 16-bit GEMMs in FP16 in and out (2026-10-04) | 1 |
| [2026-10-04-rdna2-helper-pcie-share](2026-10-04-rdna2-helper-pcie-share/) | 2x RX 6900 XT (gfx1030): a helper GPU's experts and the PCIe share (2026-10-04) | 2 |
| [2026-10-04-rdna2-pre75-attention](2026-10-04-rdna2-pre75-attention/) | RX 6900 XT (gfx1030): PR #540's attention kernel with 8 cells per step and DPP lane exchanges (2026-10-04) | 2 |
| [2026-10-04-windows-mapped-release](2026-10-04-windows-mapped-release/) | Windows mapped expert-page release | 3 |
| [2026-10-05-community-2x-p40](2026-10-05-community-2x-p40/) | Community benchmark: 2x Tesla P40 22 GB (Pascal, sm_61), 2x Xeon E5-2698 v4 | 12 |
| [2026-10-05-community-2x-rtx-4060ti](2026-10-05-community-2x-rtx-4060ti/) | Community benchmark: 2x RTX 4060 Ti 16 GB, Threadripper PRO 3975WX | 20 |
| [2026-10-05-community-2x-rx-6900xt](2026-10-05-community-2x-rx-6900xt/) | Community benchmark: 2x AMD Radeon RX 6900 XT 16 GB (gfx1030), Ryzen 5 5600X, 128 GB RAM | 67 |
| [2026-10-05-community-arc-b65](2026-10-05-community-arc-b65/) | Community benchmark: Intel Arc Pro B65, PCIe Gen4 x16 | 45 |
| [2026-10-05-community-e5-2673v3-rtx-4060ti](2026-10-05-community-e5-2673v3-rtx-4060ti/) | Community benchmark: RTX 4060 Ti 16 GB, Xeon E5-2673 v3 (AVX2), DDR3, PCIe 3.0 x8 | 6 |
| [2026-10-05-community-gfx1151](2026-10-05-community-gfx1151/) | Community report: Radeon 8060S (gfx1151), Ryzen AI Max+ 395 | 17 |
| [2026-10-05-community-rtx-4070-ti-super](2026-10-05-community-rtx-4070-ti-super/) | Community benchmark: RTX 4070 Ti SUPER, Ryzen 7 9800X3D | 6 |
| [2026-10-05-community-rtx-5060-ti-x2](2026-10-05-community-rtx-5060-ti-x2/) | Community benchmark (rerun): 2x RTX 5060 Ti (16 GB), Xeon E5-2690 v4 | 10 |
| [2026-10-05-community-rtx-5090-9700x-iq3s](2026-10-05-community-rtx-5090-9700x-iq3s/) | Community benchmark: RTX 5090, Ryzen 7 9700X, IQ3_S | 11 |
| [2026-10-05-community-rtx5080-9900x3d](2026-10-05-community-rtx5080-9900x3d/) | Community benchmark on RTX 5080 + Ryzen 9 9900X3D | 14 |
| [2026-10-05-community-rx-6700-xt](2026-10-05-community-rx-6700-xt/) | Community benchmark: AMD Radeon RX 6700 XT 12 GB (gfx1031), Ryzen 7 5700G | 16 |
| [2026-10-05-community-v100-sxm2](2026-10-05-community-v100-sxm2/) | Community benchmark: Tesla V100-SXM2-32GB on Windows (CUDA 12.6, MSVC), three quantizations | 2 |
| [2026-10-05-rdna2-rocblas-table](2026-10-05-rdna2-rocblas-table/) | RX 6900 XT (gfx1030): rocBLAS's kernel choice for the FP16 prompt GEMMs, and a solution table (2026-10-05) | 4 |
| [2026-10-06-community-2x-p100](2026-10-06-community-2x-p100/) | Community benchmark: 2x Tesla P100 PCIe 16GB, Xeon E5-2699 v4 | 12 |
| [2026-10-06-community-2x-rtx-pro-4500-engine-0.1.40.1](2026-10-06-community-2x-rtx-pro-4500-engine-0.1.40.1/) | Community benchmark on 2x RTX PRO 4500 Blackwell, Swift IQ3_XXS, engine 0.1.40.1 | 19 |
| [2026-10-06-community-2x-rx-6900xt-0.1.40.1](2026-10-06-community-2x-rx-6900xt-0.1.40.1/) | Community benchmark: 2x AMD Radeon RX 6900 XT 16 GB (gfx1030), Ryzen 5 5600X, 128 GB RAM - Strata 0.1.40.1 | 49 |
| [2026-10-06-community-2x-rx-7900-gre](2026-10-06-community-2x-rx-7900-gre/) | Community benchmark: 2x AMD Radeon RX 7900 GRE 16 GiB (gfx1100), Ryzen 7 5700X3D | 21 |
| [2026-10-06-community-4x-tesla-p100](2026-10-06-community-4x-tesla-p100/) | Community benchmark on 4x Tesla P100 16 GB (Pascal, sm_60) | 6 |
| [2026-10-06-community-r9-2x-rtx-2080-ti](2026-10-06-community-r9-2x-rtx-2080-ti/) | Community benchmark: 2x NVIDIA GeForce RTX 2080 Ti, Ryzen 9 5950X | 9 |
| [2026-10-06-community-rtx-5060ti](2026-10-06-community-rtx-5060ti/) | Community benchmark: RTX 5060 Ti 16 GB, Ryzen 9 7945HX | 11 |
| [2026-10-06-community-rtx-5090-laptop](2026-10-06-community-rtx-5090-laptop/) | Community benchmark: RTX 5090 Laptop GPU (24 GB), Ryzen 9 9955HX3D | 14 |
| [2026-10-06-community-rtx3090-egpu-64gb](2026-10-06-community-rtx3090-egpu-64gb/) | Community benchmark — RTX 3090 in Thunderbolt eGPU, 64 GB RAM — 2026-10-06 | 5 |
| [2026-10-06-community-rtx4090-opt-ins](2026-10-06-community-rtx4090-opt-ins/) | What two opt-ins cost on a single-GPU RTX 4090: STRATA_KV_PREFETCH=1, and "parallel": 2 with --batch-mtp | 8 |
| [2026-10-06-community-rtx4090-toolcall-hotfix](2026-10-06-community-rtx4090-toolcall-hotfix/) | What 0.1.40.1 does with a <tool_call> the model writes, and with requests waiting for a dead engine | 7 |
| [2026-10-06-community-rtx5070ti-5900x](2026-10-06-community-rtx5070ti-5900x/) | Community benchmark: RTX 5070 Ti, Ryzen 9 5900X, 64 GB DDR4-3600 (Windows 10) | 6 |
| [2026-10-06-community-rtxpro-v0140](2026-10-06-community-rtxpro-v0140/) | Community results: Strata 0.1.39 versus 0.1.40 on RTX PRO 6000 | 8 |
| [2026-10-06-community-rx-6800m](2026-10-06-community-rx-6800m/) | Community benchmark on AMD Radeon RX 6800M (gfx1031) | 12 |
| [2026-10-06-community-rx-7900-xtx](2026-10-06-community-rx-7900-xtx/) | Community benchmark on RX 7900 XTX (Windows 11) | 4 |
| [2026-10-06-llamacpp-0.6.0](2026-10-06-llamacpp-0.6.0/) | Strata vs llama.cpp v0.6.0 head-to-head (2026-10-06) | 25 |
| [2026-10-06-q8-prefill8192-rtxpro](2026-10-06-q8-prefill8192-rtxpro/) | Q8 prefill: 6.7x faster with 8,192-token chunks | 6 |
| [2026-10-07-community-2x-rtx-4000](2026-10-07-community-2x-rtx-4000/) | Community benchmark: 2x NVIDIA Quadro RTX 4000 8 GB, Xeon E5-2620 v3, 64 GB RAM | 1 |
| [2026-10-07-community-arc-b65-v01402](2026-10-07-community-arc-b65-v01402/) | Arc Pro B65: Strata v0.1.40.2 on PCIe Gen4 | 28 |
| [2026-10-07-community-gfx1150](2026-10-07-community-gfx1150/) | Radeon 890M (gfx1150): a hipBLASLt table, and the gfx1151 exact switches (2026-10-07) | 4 |
| [2026-10-07-community-rtx-5080-docker-wsl2](2026-10-07-community-rtx-5080-docker-wsl2/) | Community benchmark: RTX 5080 in Docker Desktop on Windows (WSL2) | 35 |
| [2026-10-07-community-rtx-5090-laptop-ud-iq4xs](2026-10-07-community-rtx-5090-laptop-ud-iq4xs/) | Community benchmark: RTX 5090 Laptop, Core Ultra 9 275HX, 188 GB RAM — Unsloth UD-IQ4_XS (first NVIDIA measurement) | 3 |
| [2026-10-07-community-rtx5090-pro4000-helper](2026-10-07-community-rtx5090-pro4000-helper/) | Community benchmark: RTX 5090 + RTX PRO 4000 Blackwell, UD-Q4_K_XL on the helper-cache config, engine 0.1.40.3 | 5 |
| [2026-10-07-community-v100-p100-helper](2026-10-07-community-v100-p100-helper/) | Community benchmark: Tesla V100-PCIE-32GB + Tesla P100-PCIE-16GB, P100 as expert-helper card (mixed Volta/Pascal) | 7 |
| [2026-10-07-community-xtx-6800xt-1m-helper](2026-10-07-community-xtx-6800xt-1m-helper/) | RX 7900 XTX + RX 6800 XT helper cache, Xeon E5-2696 v4, 1M context | 22 |
| [2026-10-07-rdna2-qsa-select-simt](2026-10-07-rdna2-qsa-select-simt/) | RX 6900 XT (gfx1030): the FP32 tiled QSA block scorer on HIP (2026-10-07) | 1 |
| [2026-10-07-v100-decode-kernels](2026-10-07-v100-decode-kernels/) | V100 (sm_70): three decode kernels that were built for other cards (2026-10-07) | 3 |
| [2026-10-07-v100-prompt-experts](2026-10-07-v100-prompt-experts/) | V100 (sm_70): the prompt's experts on FP16 tensor cores (2026-10-07) | 2 |
| [2026-10-08-community-2x-rtx3080-20gb-ud-q4-k-xl-512k-pool](2026-10-08-community-2x-rtx3080-20gb-ud-q4-k-xl-512k-pool/) | Community benchmark: 2x RTX 3080 20 GB, Xeon E5-2696 v4, UD-Q4_K_XL, 512k-token shared KV pool | 32 |
| [2026-10-08-community-2x-titan-rtx-0.1.40.3](2026-10-08-community-2x-titan-rtx-0.1.40.3/) | Community benchmark: 2x TITAN RTX (sm_75, NVLink), Xeon E5-2696 v4 | 10 |
| [2026-10-08-community-r9700-linux](2026-10-08-community-r9700-linux/) | Single R9700 Flash Next speed and short prefill measurements | 30 |
| [2026-10-08-community-rtx-5090-laptop-engine-0.1.40.3](2026-10-08-community-rtx-5090-laptop-engine-0.1.40.3/) | Community benchmark: RTX 5090 Laptop GPU (24 GB), engine 0.1.40.3 | 13 |
| [2026-10-08-community-rtx4070-iq2-xs](2026-10-08-community-rtx4070-iq2-xs/) | Community benchmark: Flash-Next IQ2_XS on RTX 4070 12 GB / 32 GB DDR4 | 6 |
| [2026-10-08-community-rtx4090-iq3xxs-200k-code](2026-10-08-community-rtx4090-iq3xxs-200k-code/) | Community benchmark: NVIDIA GeForce RTX 4090 | 8 |
| [2026-10-08-community-rtx4090-iq3xxs-200k-ru](2026-10-08-community-rtx4090-iq3xxs-200k-ru/) | Community benchmark: NVIDIA GeForce RTX 4090 (Russian prompts) | 8 |
| [2026-10-08-community-v100-sm70-table](2026-10-08-community-v100-sm70-table/) | Community benchmark on Tesla V100-SXM2-32GB (sm_70): the opt-in Volta decode kernels | 3 |
| [2026-10-08-gfx906-qsa-query-swizzle](2026-10-08-gfx906-qsa-query-swizzle/) | QSA query LDS swizzle qualification | 3 |
| [2026-10-09-community-2x-rx-6900xt-0.1.41](2026-10-09-community-2x-rx-6900xt-0.1.41/) | Community benchmark: 2x AMD Radeon RX 6900 XT 16 GB (gfx1030), Ryzen 5 5600X, 128 GB RAM - Strata 0.1.41: one card, layer split, expert helper, and... | 47 |
| [2026-10-09-community-4x-rx-7900-xt](2026-10-09-community-4x-rx-7900-xt/) | Community benchmark on 4x AMD Radeon RX 7900 XT | 14 |
| [2026-10-09-community-gfx1151-ud-iq4xs-0.1.41](2026-10-09-community-gfx1151-ud-iq4xs-0.1.41/) | Community benchmark: EVO-X2, Ryzen AI Max+ 395 / Radeon 8060S (gfx1151), 128 GB UMA | 51 |
| [2026-10-09-community-p4-prefill-roofline](2026-10-09-community-p4-prefill-roofline/) | Tesla P4 prefill: transfer reuse and chunk sizing | 22 |
| [2026-10-09-community-radeon-pro-vii-16gb](2026-10-09-community-radeon-pro-vii-16gb/) | Community benchmark: AMD Radeon Pro VII 16 GB (gfx906), Xeon E5-2666 v3 | 37 |
| [2026-10-09-community-rtx-3060-0141](2026-10-09-community-rtx-3060-0141/) | Community benchmark: RTX 3060 12 GB, Strata 0.1.41 | 26 |
| [2026-10-09-community-rtx-5060-ti-e5-2680-v4](2026-10-09-community-rtx-5060-ti-e5-2680-v4/) | Community benchmark: RTX 5060 Ti 16 GB + Xeon E5-2680 v4, Strata 0.1.41 | 12 |
| [2026-10-09-community-rtx-5060ti-wsl2-iq3_s-0141](2026-10-09-community-rtx-5060ti-wsl2-iq3_s-0141/) | Community benchmark on RTX 5060 Ti 16 GB (WSL2) + GSQ-RCO IQ3_S | 8 |
| [2026-10-09-community-rtx-5080-4060ti-0.1.41](2026-10-09-community-rtx-5080-4060ti-0.1.41/) | Community benchmark on RTX 5080 16 GB + RTX 4060 Ti 8 GB (helper expert cache), Ryzen 9 9950X3D, 96 GB | 10 |
| [2026-10-09-community-rtx-5080-w7-2475x-ud-iq4xs-786k](2026-10-09-community-rtx-5080-w7-2475x-ud-iq4xs-786k/) | Community benchmark: RTX 5080 16 GB, Xeon w7-2475X, 128 GB DDR5, Unsloth UD-IQ4_XS at a 786,432-token context (YaRN x3) | 12 |
| [2026-10-09-community-rtx5090-9800x3d-swift-1m](2026-10-09-community-rtx5090-9800x3d-swift-1m/) | Community benchmark: RTX 5090 / 9800X3D, Swift IQ3_XXS, 1M context | 67 |
| [2026-10-09-community-strix-halo-laptop-70w](2026-10-09-community-strix-halo-laptop-70w/) | Community benchmark on Radeon 8060S (Strix Halo laptop, 70 W) | 2 |
| [2026-10-09-rope-yarn-long-context](2026-10-09-rope-yarn-long-context/) | Existing RoPE and YaRN at 512K and 1M | 6 |
| [2026-10-10-community-2x-p40-iq3_s](2026-10-10-community-2x-p40-iq3_s/) | 2026-10-10 — community 2× Tesla P40, Qwen3.8-Flash-Next IQ3_S, 131K context | 2 |
| [2026-10-10-community-rx-9060-xt](2026-10-10-community-rx-9060-xt/) | Community benchmark on RX 9060 XT with IQ2_XS and IQ3_XXS | 70 |
| [2026-10-10-community-rx-9070](2026-10-10-community-rx-9070/) | Community benchmark on AMD Radeon RX 9070 16 GB (gfx1201), Windows 11, Ryzen 7 5800X3D | 8 |
| [2026-10-11-community-rtx3090-5600x-arena-async](2026-10-11-community-rtx3090-5600x-arena-async/) | --adapt-async 1 in the arena mode — community report, RTX 3090 + Ryzen 5 5600X (Windows 11) | 7 |
| [2026-10-11-community-rtx3090-5600x-arena-pin-by-need](2026-10-11-community-rtx3090-5600x-arena-pin-by-need/) | Pin by need (STRATA_PIN_BY_NEED=1) — community report, RTX 3090 + Ryzen 5 5600X (Windows 11) | 8 |
| [2026-10-11-gfx1100-qsa-scores](2026-10-11-gfx1100-qsa-scores/) | Exact gfx1100 QSA scorer: measured prefill benefit, opt-in only | 7 |
| [2026-10-11-rtx-5090-callback-affinity](2026-10-11-rtx-5090-callback-affinity/) | RTX 5090: adaptive-overlap callback affinity (#1975) | 1 |
| [2026-10-12-community-gfx1103](2026-10-12-community-gfx1103/) | Radeon 780M (gfx1103) on ROCm 10.1: dot4 + WMMA + a hipBLASLt table (2026-10-12) | 23 |
| [2026-10-12-hc-persist-gfx1030](2026-10-12-hc-persist-gfx1030/) | STRATA_HC_PERSIST=1: the verifier's hyper-connection read as one persistent kernel (gfx1030) | 2 |
| [2026-10-12-p40-sm61-kernels](2026-10-12-p40-sm61-kernels/) | Tesla P40 (sm_61): wide dequantization, the dense GEMMs as cublasSgemm (opt-in), expert mode 8 and a matvec table (2026-10-12) | 16 |
| [2026-10-20-0.2.1-bench](2026-10-20-0.2.1-bench/) | First real strata bench run on the released 0.2.1 (terminal 0.2.1, engine 0.1.46 cuda12 Linux packages) | 43 |
| [2026-10-20-0.2.1-idle-gpu](2026-10-20-0.2.1-idle-gpu/) | The idle GPUs (proposal 2507-0): what the existing switches do on NVIDIA | 15 |

## 2026-11

| Folder | What | Files |
| --- | --- | ---: |
| [2026-11-14-rdna2-pcie-auto](2026-11-14-rdna2-pcie-auto/) | PcieAuto on gfx103x: the decode miss split from the CPU pool's speed (#4007) | 3 |

## Loose files

| File |
| --- |
| [COMMUNITY.md](COMMUNITY.md) |
| [exchange-rotation-ab.json](exchange-rotation-ab.json) |
| [exchange-rotation-token-ids.json](exchange-rotation-token-ids.json) |
