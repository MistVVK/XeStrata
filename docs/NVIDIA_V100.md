# NVIDIA Tesla V100 / Titan V (Volta, sm_70)

This page is for people running Strata on a V100 or Titan V. Volta uses the experimental CUDA 12 engine, which current releases
provide as an engine component through the app installer or the Terminal ([INSTALL.md](INSTALL.md)). CUDA 13 cannot run Volta.
How setup selects the CUDA 12 engine and the driver requirements are in [OLDER_GPUS.md](OLDER_GPUS.md). Speeds below were
reported on the named machines. They are not a guarantee for another V100 setup.

> **On this page:** [Quick start](#quick-start) · [What runs on Volta](#what-runs-on-volta) · [Opt-in switches](#opt-in-switches) ·
> [Checking the kernels](#checking-and-benchmarking-the-volta-kernels) · [Limits](#limits) · [History](#history--older-measurements)

## Quick start

1. Install the app or Terminal for your system ([INSTALL.md](INSTALL.md)). Setup picks the CUDA 12 engine when the model runs on
   a Volta card.
2. To choose it by hand from a source checkout: `START-HERE.bat --setup --gpu 0 --cuda 12` (Linux: `./setup.sh`).

### Build from source

CUDA 13 dropped Volta, so the build needs a CUDA 12.x toolkit (12.8 was used) and a host GCC that toolkit accepts.

```sh
cmake -S engine -B build -DSTRATA_ENABLE_CUDA=ON -DSTRATA_BUILD_TESTS=OFF \
      -DCMAKE_CUDA_ARCHITECTURES=70 -DSTRATA_EXPERIMENTAL_SM60=ON
cmake --build build --target strata -j
```

A source checkout can fetch or build its CUDA 12 engine; `--build` requests compilation with the toolkit (see
[OLDER_GPUS.md](OLDER_GPUS.md)). For a PC with a V100 and a newer card, build for both architectures
(`-DCMAKE_CUDA_ARCHITECTURES="70;86"`). A sm_70-only build has no code for a card such as a Quadro RTX 4000 (sm_75); hide it with
`CUDA_VISIBLE_DEVICES`.

## What runs on Volta

The flag lowers the floor (CMake and the runtime device check) and replaces two intrinsics Volta lacks. Which kernel runs is
decided per architecture when the engine is built.

| Part of the prompt path | On sm_70 |
| --- | --- |
| MoE experts | MMQ by default. Opt-in `STRATA_PF_WMMA=1` (only when every visible card is compute capability 7 or newer): `gemm_iq_f16_grouped`, a group of experts per launch on the FP16 tensor cores. FP16 activations instead of MMQ's q8_1, so answers differ. |
| Dense projections (dequantized weights) | FP16 tensor-core GEMMs (cuBLAS / CUTLASS `s884`) |
| BF16 projections (hyper-connection, router, indexer) | converted to FP16 and run on the FP16 tensor cores (`STRATA_BF16_TC=0`: cuBLAS BF16, an FP32 SIMT kernel on Volta) |
| QSA attention for decode and verify windows | the #540 kernel, fewer shuffles, bit-exact (`STRATA_ATTN_PRE75=0`: the one other cards run) |
| QSA prompt attention, int8 / FP16 / K8V4 KV | `prompt_attn_v70_kernel` on `mma.m8n8k4` (`STRATA_PROMPT_ATTN_OLD=1`: the decode kernel, one query at a time) |
| QSA prompt attention, Q4_0 KV | the decode kernel |
| QSA block scores | the warp kernel (the tensor-core scorer needs sm_80) |

| Part of a decode or verify window | On sm_70 |
| --- | --- |
| Routed experts in VRAM | expert mode 8 (codebook grid and the group's activations in shared memory, SwiGLU fused). On by default via `STRATA_SM70_TABLE=1`. |
| Dense 2-4 column GEMVs and the head | the interleaved `native_mmvq_il` with a Volta rows table (`STRATA_SM70_TABLE=1`), and decoupled 6-token launch (`STRATA_B6_MMVQ_ROWS=1`). Opt-in `STRATA_Q8_INPLACE=1` converts dense `Q8_0` matrices in-place (`+0.0 MiB` VRAM) to split planes with bit-exact prefill dequantization and `6` blocks/SM on Volta. |
| Hyper-connection read | coalesced 4-block `Q8_0` down projection (`STRATA_GR_Q8_DOWN_MODE=2`), split mid (`STRATA_GR_Q8_UP_COLS=16`), and pre-normalized up projection (`STRATA_GR_Q8_UP_MODE=2`). Bit-exact to the unfused `Q8_0` HC kernels. |

## Default `sm_70` table & `volta-forever` detection

Every single-GPU or uniform-`sm_70` run enables 21 bit-exact switches by default (`STRATA_SM70_PERF=0` or `STRATA_DEFAULTS=026` turns the table off; any individual switch you set yourself wins):

- `STRATA_ADAPT_OVERLAP=1`, `STRATA_SM70_TABLE=1`, `STRATA_B6_MMVQ_ROWS=1`, `STRATA_TSUM=1`, `STRATA_SMALL_UNR=3`, `STRATA_GR_NORM_FAST=1`, `STRATA_SIDE_MIXER=2`, `STRATA_ARGMAX_ROWS_SM70=1`, `STRATA_LFUSE=1`, `STRATA_GDN_SPLIT=1`, `STRATA_SH_STREAM=1`, `STRATA_Q80_ROWS1=1`, `STRATA_NO_MMVQ_ROWS2=1`, `STRATA_CPU_IQ_ALL_EXACT1=1`, `STRATA_GR_Q8_DOWN_MODE=2`, `STRATA_GR_Q8_UP_COLS=16`, `STRATA_GR_Q8_UP_MODE=2`, `STRATA_SCORES_HALVE=1`, `STRATA_SCORES_THREAD=1`, `STRATA_TOPK_SAMPLE=1`, `STRATA_SELECT_LONG=1`.

When running under CUDA 13 via [nbald/volta-forever](https://github.com/nbald/volta-forever), Strata probes `volta_forever_version` in `libcuda.so.1`, remaps emulated `7.5` reporting back to Volta `(7, 0)` for defaults and kernel selection, and keeps `CUDA_MODULE_LOADING=LAZY` so cuBLAS initialization does not exhaust heap memory.

## Opt-in switches

These switches are opt-in (either multi-GPU topology-specific or non-bitwise FP rounding):

| Switch | What it does |
| --- | --- |
| `STRATA_Q8_INPLACE=1` | Packs eligible dense `Q8_0` matrices (with `n_out >= STRATA_Q8_INPLACE_MIN_OUT`, default `640`) in-place into split int8/fp16 planes at zero extra VRAM (`q8_sm60.cuh`, `6` blocks/SM on Volta) and bit-exact prefill dequantization. Changes FP accumulation order (`~1e-7` relative RMS). |
| `STRATA_PEER_HEAD_SPLIT=1` | Splits the `Q6_K` output LM head 50/50 across GPU 0 and peer GPU 1 over NVLink P2P (`cudaMemcpyDefault`), cutting LM head time and freeing ~248.7 MiB on GPU 0 for additional resident experts. |
| `STRATA_NUMA_ARENA=1` | On dual-socket Linux hosts using the resident expert arena (`4 KiB` pages), first-touches Slice 0 and Slice 1 of each expert's `[gate \| up \| down]` roles on NUMA nodes 0 and 1 before `cudaHostRegister`, overriding any process-wide `MPOL_INTERLEAVE` (`numactl --interleave=all`). |
| `STRATA_FUSE_DECODE=1` | The decode GEMV / MoE fusions of #1097. Bitwise the unfused kernels. |
| `STRATA_PF_WMMA=1` | The FP16 tensor-core prompt experts; also takes IQ4_NL down projections. |
| `STRATA_PF_WMMA2=1` | A second prompt-expert kernel for IQ2_S gate/up and IQ4_NL down (#2356). |
| `STRATA_ARGMAX_ROWS_SM70=1` | The multi-block greedy pick on sm_7x (#2357), bitwise the one-block pick. |
| `STRATA_GR_NORM_FAST=1` | The register-split hyper-connection norm (#2358). |
| `STRATA_GR_V70=1` | The token-split hyper-connection read: one warp per row and token for the down and up projections. Bitwise the default read; each card checks it at start-up against the default read at 1-8 tokens and keeps the default read if any output differs. |
| `STRATA_Q8_GATHER_WIDE=0` | Turns off the 8-wide int8 KV gather (#1095) that the experimental build uses below compute capability 7.5. |
| `STRATA_GDN_REC_WARP=2` | The chain-split DeltaNet recurrence of the prompt (#1098). Changes the FP32 summation order, so the default stays the pipelined kernel. `1` is the bit-exact sync-free variant; `STRATA_GDN_CHUNK=1` the chunked path. |
| `STRATA_Q8_V70=1` | The dense Q8_0 GEMV of 2-4 columns with its block loop unrolled (`n_in` a multiple of 1024). Bitwise the default kernel. Not taken with `STRATA_TSUM=1`. |

Volta's tensor cores take FP16 only. A model's BF16 weights are converted, and a weight outside FP16's range would saturate (none
did in the check in the history section).

### Controlled dual V100-SXM2 (NVLink) qualification

Measured on dual Tesla V100-SXM2-16GB (`sm_70`, NVLink `NV6`) locked at `200 W / 1020 MHz SM / 877 MHz HBM2` (`ecc=0`, zero throttle events across all samples), running `Qwen3.8-Flash-Next` (`512x56B`, `UD-IQ2_XS-v31-Q8`, `--kv int8 --peer-device 1 --peer-reserve-mib 1600 --draft dflash2 --spec 6`, `STRATA_ROUTE_TAIL_SKIP=0`, `STRATA_PF_PEER_STREAM=0`, `STRATA_P2P_DECODE_SCATTER=1`) over 2 interleaved rounds (`A->B->C->D` and `D->C->B->A`) × 3 seeded prompts (`64K` 3-needle retrieval `256` tokens, C++20 lock-free ring buffer `384` tokens, technical prose `384` tokens = `6` decode samples per primary arm; `1` round `64K` for `STRATA_PF_WMMA=1`):

| Arm | Configuration | GPU 0 / 1 slots | 64K prefill (tok/s) | 64K TTFT (s) | Decode median, n=6 (tok/s) | 64K / Code / Prose decode (tok/s) | Window median (ms/w) | Parity & 64K needles |
| --- | --- | ---: | ---: | ---: | ---: | --- | ---: | --- |
| `base` | `origin/main` (`STRATA_PF_WMMA=0`) | 5357 / 8587 | 1,450.0 | 42.64 | 68.05 | 68.05 / 74.0 / 63.8 | 41.53 | baseline (`3/3` needles) |
| `stageb_default` | zero manual Stage B flags (19-switch `sm_70` default) | 5357 / 8587 | 1,524.8 (+5.16%) | 40.56 | 84.60 (+24.32%) | 84.6 / 91.8 / 78.3 | 33.46 (-8.07) | **bit-exact SHA on all 3 prompts** (`3/3`) |
| `stageb_optin` | `+ STRATA_PEER_HEAD_SPLIT=1 + STRATA_Q8_INPLACE=1` | 5508 / 8430 | 1,522.7 (+5.01%) | 40.61 | 90.65 (+33.21%) | 90.65 / 99.3 / 85.05 | 33.05 (-8.48) | opt-in FP sum order + `+151` GPU0 slots (`3/3`) |
| `stageb_serving` | `+ draft_vocab_en.bin --draft-conf 0.45` | 5589 / 8425 | 1,522.7 (+5.01%) | 40.61 | 92.75 (+36.30%) | 92.75 / 97.85 / 87.0 | 31.39 (-10.15) | self-deterministic across rounds (`3/3`) |
| `stageb_pf_wmma` | `stageb_serving` + `STRATA_PF_WMMA=1` (`n=1`, 64K only) | 5589 / 8425 | 1,644.1 (+13.39%) | 37.62 | 89.50 (+31.52%) | 89.5 / — / — | 37.65 | FP16 TC MoE prefill (`3/3`) |

## Checking and benchmarking the Volta kernels

```sh
cmake -S engine -B build -DSTRATA_PARITY_PROMPT_ATTN=ON ... && cmake --build build --target qsa_prompt_attn_parity
CUDA_VISIBLE_DEVICES=0 ./build/qsa_prompt_attn_parity 32768 2048 5     # synthetic, no model; exit 0 = PASS
```

It compares the tensor-core attention with the FP32 kernel and an FP64 host reference and times both. The Q4_0 cases are skipped
below sm_80.

## Limits

- A layer split over two V100 was not measured with this kernel. The cards work in turn on a single request, so a split adds
  expert-cache room, not prompt speed.
- Pascal (sm_60) is covered by the same flag. Community runs exist for it; this page's own kernel measurements are V100 only.

## History / older measurements

**Prompt attention kernel.** Measured on: one V100-PCIE-32GB, i9-7960X, Unsloth UD-IQ4_XS, int8 KV, MTP `--spec 2`, random-word
prompts, prompt read speed in tok/s with and without the Volta kernel (`STRATA_PROMPT_ATTN_OLD=1`). Decode was 38-51 tok/s in
every run and is not affected. Report: [docs/bench-results/2026-10-03-v100-prompt-attn](bench-results/2026-10-03-v100-prompt-attn/README.md).

| Prompt tokens | old attention path | `prompt_attn_v70_kernel` |
| --- | ---: | ---: |
| 7,194 | 1,010 | 1,164 |
| 28,650 | 1,063 | 1,251 |
| 114,338 | 973 | 1,123 |

Against an FP64 reference the kernel's error is about 2-3e-6 (the FP32 kernel it replaces: 2e-6). 15 of 15 needles were found.
Greedy output was identical on the 28,650- and 114,338-token prompts and differed late in the answer on the 7,194-token one.

**Other measurements**

| Measured on | Result |
| --- | --- |
| V100-SXM2, contributor's measurement | decode kernels (`STRATA_SM70_TABLE=1`): GPU work per verify window 22.8 -> 21.2 ms ([report](bench-results/2026-10-07-v100-decode-kernels/README.md)) |
| V100, `STRATA_PF_WMMA=1` | prompt +10% against MMQ ([report](bench-results/2026-10-07-v100-prompt-experts/README.md)) |
| V100, `STRATA_PF_WMMA2=1` | prompt +2.5-4.3% |
| V100, `STRATA_ARGMAX_ROWS_SM70=1` / `STRATA_GR_NORM_FAST=1` | decode +0.7-0.8% / +0.5-0.6% |
| V100, `STRATA_Q8_GATHER_WIDE=0` | +11% on the gather kernel |
| V100, `STRATA_GDN_REC_WARP=2` | +6-9% on the recurrence |
| V100, `STRATA_FUSE_DECODE=1` | 4-16 us per token saved, not visible end to end |
| Tesla P100 alone and with an RTX 2070 SUPER as expert helper (#1069) | 30.5 tok/s alone, 35.8-38.4 with the helper |
| Two Tesla P40, layer split (#1028) | [report](bench-results/2026-10-05-community-2x-p40/README.md) |
| V100 + P100 pair (#1079) | +6-11% decode from engine 0.1.39 to 0.1.40 with the P100 as helper cache |
