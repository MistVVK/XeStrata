# Strix Halo (Ryzen AI Max, gfx1151)

This page is for people with a Ryzen AI Max "Strix Halo" PC (Radeon 8060S / 8050S, RDNA 3.5, `gfx1151`, one pool of LPDDR5X
shared by the CPU and the GPU). Strata runs there through the same HIP backend as the Radeon cards in [AMD_HIP.md](AMD_HIP.md).
**Status: experimental.**

> **On this page:** [Quick start](#quick-start) · [What the engine does differently](#what-the-engine-does-differently) ·
> [1. Toolchain](#1-the-toolchain-no-root-needed) · [2. Engine](#2-the-engine) · [3. Tuning table](#3-the-prompt-table-and-running) ·
> [4. Defaults](#4-what-is-on-by-default-on-gfx1151) · [5. Opt-in](#5-what-is-not-on-by-default-it-changes-bits) ·
> [5b. Desktop and RAM](#5b-the-desktop-the-ram-and-a-discrete-card-beside-the-strix-halo-linux) · [6. Measured](#6-measured) ·
> [History](#history--older-measurements)

## Quick start

Use the [current release](https://github.com/Niko1221/Strata/releases/latest). You do not need to build ROCm for a packaged install.

| System | What to install | Engine variant |
| --- | --- | --- |
| Linux | `Strata-0.2.18-linux-x86_64.AppImage` (it downloads the engine on first start), or `strata launch` | `linux-x86_64-hip-halo` |
| Windows | `Strata-0.2.18-windows-x64-setup.exe` (it downloads the engine on first start), or `strata launch` | `windows-x86_64-hip-rdna3` |

Packaged engines carry the matching hipBLASLt tuning table for gfx1151 and select it on their own. Setup recognizes `gfx1151` and
PCI ID `1002:1586`; other integrated Radeon chips have separate support limits. Unified memory is counted once. Setup does not
change BIOS or kernel settings. The manual Linux toolchain below is for reproduction and builds, not a prerequisite for the app. WSL is not supported: the AMD engines hang there.

On Windows, [STRIX_HALO_WINDOWS.md](STRIX_HALO_WINDOWS.md) has the speeds measured on 0.2.7 with Qwen IQ3_S, how memory is used and the traps.

## What the engine does differently

- **Unified memory.** The GPU's "VRAM" is the GTT pool of the same RAM. `device_free_bytes()` (the figure `--expert-cache auto`
  and the memory plan use) counts what Linux can give back (`MemAvailable` less 6 GiB for the system; `STRATA_UMA_HEADROOM_GIB`
  changes the 6) for any integrated GPU on Linux, a DGX Spark included.
- **gfx11 matrix cores.** The prompt attention, the block scorer, the prompt GEMMs and the prompt experts have WMMA kernels for
  gfx1100 / 1101 / 1102 / 1103 / 1150 / 1151. A gfx11 part outside that list (gfx1152) takes the portable path.
- **The hipBLASLt table.** `terminal/tools/hip/gfx1151-hipblaslt-100401.txt` is for ROCm 7.14.1's hipBLASLt (`100401`), including the prompt
  shapes at `--prefill 16384`. Setup's own ROCm for gfx1151 (the 7.14.0a20260608 wheels, hipBLASLt 1.4.0) takes
  `terminal/tools/hip/gfx1151-hipblaslt-100400.txt`, calibrated against that library.
- **The expert cache and the RAM budget share the RAM.** For a model with a RAM budget (UD-IQ4_XS, UD-Q4_K_XL) setup writes a number
  for `--expert-cache` (the RAM less 24 GB less the budget), not `auto`, because `auto` counts all memory the OS can give back at
  start and can take the room the OS needs. The number is in the config's `args`. An engine run by hand should pass one too.
- **Windows.** The engine does not register the expert arena with the GPU driver. The arena stays in RAM, held by the
  working-set lock, and the expert cache gets its normal slots. `STRATA_ARENA_PIN_GIB=N` (or `0`) or `STRATA_UMA_REGISTER=1`
  registers it. Linux registers the whole arena by default.

## 1. The toolchain (no root needed)

ROCm 7.14.1 for gfx1151 from AMD's TheRock builds, extracted into a folder of your own:

```sh
mkdir -p ~/rocm && cd ~/rocm
curl -LO https://repo.amd.com/rocm/tarball-multi-arch/therock-dist-linux-gfx1151-7.14.1.tar.gz   # 1.7 GB
sha256sum therock-dist-linux-gfx1151-7.14.1.tar.gz   # c40e8f2bd6630a7d11557c762b99c6fa8afb04c9bd0e51ed1675ee1ca24afb00
mkdir install && tar -xzf therock-dist-linux-gfx1151-7.14.1.tar.gz -C install
export SDK=$HOME/rocm/install
export ROCM_PATH=$SDK HIP_PATH=$SDK HIP_PLATFORM=amd
export PATH=$SDK/bin:$SDK/lib/llvm/bin:$PATH
export LD_LIBRARY_PATH=$SDK/lib:$SDK/lib/rocm_sysdeps/lib:$SDK/lib/llvm/lib:${LD_LIBRARY_PATH:-}
```

You also need CMake 3.24+, Ninja, git and a C++ host toolchain with libstdc++ headers
(`sudo apt install build-essential cmake ninja-build git`). If the system has no compiler, point ROCm's clang at a user-space gcc
with `--gcc-toolchain=<dir>/usr` in `CMAKE_HIP_FLAGS`.

The GPU side needs the kernel's amdgpu driver and `/dev/kfd` access (your user in the `render` and `video` groups). To let the GPU
use most of a 128 GB machine, the maintainers' box boots with `ttm.pages_limit=29360128 ttm.page_pool_size=29360128` (a 112 GiB GTT;
the default is about half of the RAM) and `amd_iommu=off`. Both are host settings the engine does not need to run, only to have the room.

## 2. The engine

The engine build needs no llama.cpp checkout.

```sh
git clone https://github.com/Niko1221/Strata.git && cd Strata
cmake -S engine -B build-halo -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DSTRATA_ENABLE_HIP=ON -DSTRATA_ENABLE_CUDA=OFF \
  -DCMAKE_HIP_ARCHITECTURES=gfx1151 \
  -DCMAKE_HIP_COMPILER=$SDK/lib/llvm/bin/clang++ -DCMAKE_HIP_COMPILER_ROCM_ROOT=$SDK \
  "-DCMAKE_PREFIX_PATH=$SDK;$SDK/lib/rocm_sysdeps;$SDK/lib/llvm" \
  "-DCMAKE_HIP_FLAGS=--rocm-path=$SDK --rocm-device-lib-path=$SDK/lib/llvm/amdgcn/bitcode"
cmake --build build-halo --target strata -j 8        # about 25 minutes the first time
```

The build warns that `gfx1151` is "unvalidated". The engine is built without llama.cpp's MMQ kernels: the prompt experts run on
Strata's own kernels where they cover a layer and on the FP16 path elsewhere. To test:
`cmake --build build-halo -j 8` builds the tests and `ctest --test-dir build-halo` runs them. `hip_prefill_hcd_exact_parity` and
`hip_prefill_hipblaslt_gemm` need the table (next section) and skip without it.

## 3. The prompt table, and running

A packaged engine finds its table itself. For a hand-built engine, hand it the table, because without it the prompt's dense matrix
products use plain hipBLAS and are slower:

```sh
export STRATA_HIPBLASLT_TUNING=$PWD/tools/hip/gfx1151-hipblaslt-100401.txt
```

The engine accepts a table only for the architecture and hipBLASLt version named on its second line
(`STRATA_HIPBLASLT_TUNING_V1 gfx1151 100401`: ROCm 7.14.1) and says so at start-up (`prefill gemm: hipBLASLt tuning enabled (90 rows,
gfx1151, version 100401)`). Then run `strata` with the same arguments as on any other card (the model packs, `--serve`, the settings
of [DETAILS.md](DETAILS.md)). The numbers below used packs of the Unsloth UD-IQ4_XS or UD-Q4_K_XL GGUFs.

## 4. What is on by default on gfx1151

At start the engine looks at GPU 0. On a gfx1151 it sets the switches below unless you set them yourself.
`STRATA_GFX1151_DEFAULTS=0` turns the whole table off, and the start-up line names what was set. Every one of them gave
byte-identical output on the maintainers' box (same token ids at 4K / 32K / 64K and on four models, plus a bitwise harness per
kernel): they change speed, not answers. Nothing here applies to CUDA or to another AMD architecture.

| Switch | What it does |
| --- | --- |
| `STRATA_GDN_HEAD`, `STRATA_GDN_PP=2`, `STRATA_GDN_CONVL2`, `STRATA_GDN_NOY` | The prompt's gated-delta-net recurrence: four lanes per column, two columns per lane, convolution and norms fused. |
| `STRATA_HCD_EXACT` | The hyper-connection down projection (N 320, K 10240) in a WMMA kernel that reproduces hipBLASLt solution 1176 / 1177's summation order (`hip_prefill_hcd_exact_parity` checks it bit for bit). It runs only where the table makes hipBLASLt pick one of those two, and says once if it cannot. |
| `STRATA_CVEC_FUSE`, `STRATA_PF_PAD` | A steered layer's write, control vector and next norm in one pass; padded GEMM row strides. |
| `STRATA_TOPK_REG_MIN_BLOCKS=4096` | The prompt's QSA top-k uses the register kernel from 16K cells of context instead of 28K. The selected cells are identical (`qsa_topk_parity`). |
| `STRATA_Q8_PACKED`, `STRATA_Q6_PACKED`, `STRATA_MMVF_ROWS`, `STRATA_ATTN_LANECELL` | Decode layouts and kernels: packed Q8_0 / Q6_K weights, 4-row BF16 GEMV, one-cell-per-thread attention scores. |
| `STRATA_EXPERT_V2`, `STRATA_TSUM`, `STRATA_LFUSE` | The grouped decode experts (IQ3_S / IQ4_NL), single-butterfly warp sums, fewer launches around the shared expert and the KV append. |
| `STRATA_GDN_SPLIT`, `STRATA_QFUSE`, `STRATA_PLE_BATCH` | The GDN step over four blocks per head, activation images written by their producers, the PLE key / value projections of a verify window at once. |
| `STRATA_SH_STREAM=1` | The shared expert on a second stream beside the routed experts. Off by default on other AMD cards, but it pays on Linux gfx1151 (decode +1.8% UD-IQ4_XS, +6.7% UD-Q4_K_XL at 8K). **Off by default on Windows gfx1151 since 0.2.7:** it measured 6.9% slower there with the same token ids ([STRIX_HALO_WINDOWS.md](STRIX_HALO_WINDOWS.md)). `STRATA_SH_STREAM=1` turns it back on. |

Two further defaults change bits. `STRATA_HC_Q8` reads the hyper-connection weights as 8-bit (`STRATA_HC_Q8=0` restores the BF16
read), and on Windows the prompt is read in chunks of 8192 tokens (`STRATA_GFX1151_CHUNK=0` keeps 6144). About `STRATA_HC_Q8`: it
halves the hyper-connection weight bytes (13.29 -> 7.15 MB per read) and gave +8.4% decode on `json_extract`, at the cost of answers
that are not bit-identical to the BF16 read. How far they move: [AMD_HIP.md](AMD_HIP.md) ("Strix Halo: 8-bit hyper-connection weights").

## 5. What is not on by default (it changes bits)

These are faster on gfx1151 and round differently from the default path (a different summation order or a half-precision
intermediate), so they are opt-in. The maintainers gate them with KL against the default. `STRATA_PF_SWITCH_MIN_T=4096` keeps
short prompts on the default numerics, since these pay on long ones. The maintainers' fast configuration sets all of them:

```sh
export STRATA_PF_FUSED=1        # the prompt experts on the matrix cores (IQ2 / IQ3 / IQ4_XS gate/up, Q2_0 / IQ4_NL / Q8_0 down)
export STRATA_PF_GEMM=1         # the prompt's FP16 projections on the 128 x 256 WMMA GEMM
export STRATA_HC_UPMIX=1        # the hyper-connection up projection with its epilogue
export STRATA_PA_FAST=1         # the prompt attention with single FP16 q and p
export STRATA_HIP_WMMA=1        # the prompt attention on matrix cores
export STRATA_SELECT_WMMA=1     # the block scorer on matrix cores
export STRATA_PF_SWITCH_MIN_T=4096
export STRATA_PREFILL_STREAM_MIN=128   # prompt reads of 128-1,023 tokens on the fused experts too
```

`STRATA_PF_GEMM`, `STRATA_HC_UPMIX`, `STRATA_PA_FAST`, `STRATA_HIP_WMMA` and `STRATA_SELECT_WMMA` together are one opt-in on an
integrated Radeon: `STRATA_IGPU_PROMPT_SET=1` ([AMD_HIP.md](AMD_HIP.md), "Integrated GPUs"), measured on a Radeon 780M only, where
it is the default. `STRATA_PF_SWITCH_MIN_T` and `STRATA_PREFILL_STREAM_MIN` stay opt-in as well. `STRATA_PF_FUSED` is on by itself on an integrated Radeon under Linux (measured on the 780M only; the Strix Halo takes it by the same rule and was not run there); `STRATA_PF_FUSED=0` keeps MMQ in a build that has it (the release builds of 0.2.16 and later have none: `=0` there runs the prompt experts on the native int8 kernels, #6251).

**On Windows the Strix Halo is the exception** (engine 0.1.52 and later): `gfx1151` takes the matrix-core set by itself -
`STRATA_HIP_WMMA`, `STRATA_PF_GEMM`, `STRATA_PA_FAST`, `STRATA_HC_UPMIX` and `STRATA_PF_FUSED`, without `STRATA_SELECT_WMMA`
(x1.003 there) - and the prompt keeps its own buffers instead of lending expert-cache slots. Measured on Qwen3.8-Flash-Next
IQ3_S, prompts of 4K to 64K: x1.59 to x1.72 over 0.2.7. The whole block is off with `STRATA_GFX1151_DEFAULTS=0`; the set with
`STRATA_IGPU_PROMPT_SET=0`, the buffers with `STRATA_PREFILL_BORROW=1`
([STRIX_HALO_WINDOWS.md](STRIX_HALO_WINDOWS.md#4-what-is-on-by-default)). Linux `gfx1151` keeps the opt-in above until it is
measured there.

| Switch | What it does |
| --- | --- |
| `STRATA_PREFILL_STREAM_MIN=N` | `STRATA_PF_FUSED=1` reaches only chunks of N tokens or more (1,024 by default, the floor measured on a discrete card). On Windows a UD-Q4_K_XL set reads pieces below 1,024 tokens on the fused kernels' staged entry by itself (pieces of 96 to 1,000 tokens read 3.8 to 5.2 times sooner than on the old path; `STRATA_PF_STAGED_FO=0` gives the long-prompt path from 64 tokens instead, 3.6 to 4.5 times sooner); other sets keep 1,024, and a smaller N is opt-in. |
| `STRATA_PF_FUSED_KQ=1` | A UD-Q4_K_XL or other K-quant pack takes the fused experts only with this as well. |
| `STRATA_PF_F16=1` (gfx11 only) | Runs the covered prompt experts on the F16 matrix cores instead of the int8 ones. It rounds differently from the int8 kernels. |
| `STRATA_REFILL_PARTS=1` (exact; HIP and CUDA, a GGUF read in place) | Refills the expert-cache slots a prompt borrowed straight from the GGUF mapping instead of through an assembled 4 MB copy per expert. Only gfx1151 has measured it. |
| `--mtp-window 8192` | Not Strix-specific: the draft layer attends to the last 8,192 cells. |
| `--spec`, `--lookup-chain`, `--mtp-q4` | See [DETAILS.md](DETAILS.md). |

## 5b. The desktop, the RAM, and a discrete card beside the Strix Halo (Linux)

The Strix Halo chip usually draws your desktop, and its graphics memory is your RAM. The engine takes both into account.

- **One memory, counted once.** The automatic expert cache size is the smaller of what the GPU reports free and what the OS reports
  available, less a reserve for the desktop (the larger of 8 GiB and 8% of the RAM; the start-up line says so when it changes the
  size). `STRATA_UMA_RESERVE_GIB=N` sets the reserve (`0` = nothing kept back). `--expert-cache N` is never changed.
- **A card of its own beside the chip** (for example a Radeon AI PRO R9700). That card is the recommended Main card and the chip
  stays Off in the card roles. If you name the chip in a layer split anyway (`--gpus 0,1`, or the config's `"gpu": [0, 1]`), the
  split runs as asked, the chip takes the first layers and never the last stage, the picture reader starts on the card of its own,
  and the start prints one plain line about it. `"gpu_order": "as_given"` and a manual `layer_split` keep your order.
  `setup --gpus all` leaves the chip out when two cards of their own are present; name it by number to use it.
- **The CPU.** On Linux with built-in Radeon graphics the engine starts at nice +5, so the desktop keeps its share of the CPU.
  `STRATA_ENGINE_NICE=0` turns it off, `STRATA_ENGINE_NICE=N` sets another value.

Not known yet: how much of the screen freeze a long compute job on the same chip causes, and whether registering the whole expert
arena with the driver (the Linux default on gfx1151) hurts when the GTT is small. `STRATA_ARENA_PIN_GIB=none` runs the unregistered
arena on any OS. Reports with `amdgpu_top` and `dmesg` are welcome.

## 6. Measured

**Measured on:** a Strix Halo PC (gfx1151, Windows), IQ3_S model, Strata 0.2.3 (engine 0.1.48), 3 pairs against 0.2.2, from
[release-notes-0.2.3.md](release-notes-0.2.3.md). Answer speed +6% to +19% (tool calls +19%, code +18%, 64K prompt +13%, 8K prompt
+12%, 4K prompt +8%, JSON +6%, agent chat +9%). Prompt speed +5% to +14% (8K +14%, 64K +5%, 4K unchanged because it is one chunk).
Prompt speed at 4K, 8K and 64K is 685, 783 and 749 tokens per second. This is specific evidence, not validation of every APU or
quantization.

The Linux numbers from the maintainers' Ryzen AI Max+ 395 (128 GB, Ubuntu 26.04, kernel 7.0, ROCm 7.14.1) are in the history
section below.

## History / older measurements

<a id="measured"></a>

**Measured on:** Ryzen AI Max+ 395, 128 GB, gfx1151, Linux, Strata 0.1.40 code, the maintainers' fast configuration of section 5 on
top of the defaults of section 4, `--spec 4`, `--mtp`, `--lookup-chain 3`, `--mtp-q4 all`, `--prefill 16384`, int8 KV, all experts
in unified memory. Prompt tok/s is (prompt tokens - 1) / prompt time. Medians of 3 runs, one fresh process per run, two engines
interleaved, every pair printed the same token ids. "Before" is the maintainers' earlier engine branch with the same switches set
by hand.

| Model | Context | Prompt before -> now | Output before -> now |
| --- | --- | --- | --- |
| UD-IQ4_XS | 8K | 1,212 -> 1,293 (+6.7%) | 51.7 -> 53.8 (+4.0%) |
| UD-IQ4_XS | 128K | 1,241 -> 1,320 (+6.4%) | 49.3 -> 51.4 (+4.2%) |
| UD-Q4_K_XL | 8K | 1,136 -> 1,169 (+2.9%) | 50.6 -> 52.6 (+3.9%) |
| UD-Q4_K_XL | 128K | 1,220 -> 1,248 (+2.4%) | 41.6 -> 43.3 (+3.9%) |
| IQ3_S | 8K | 1,202 -> 1,242 (+3.4%) | 56.2 -> 59.7 (+6.3%) |
| IQ3_S | 128K | 1,254 -> 1,312 (+4.6%) | 40.6 -> 42.8 (+5.5%) |
| IQ3_XXS | 8K | 1,237 -> 1,238 (+0.1%) | 48.8 -> 51.5 (+5.6%) |
| IQ3_XXS | 128K | 1,259 -> 1,252 (-0.6%) | 48.4 -> 50.9 (+5.1%) |

UD-IQ4_XS at 4K / 32K / 64K (5-6 interleaved pairs): prompt +9.2% / +6.0% / +5.3%, output +5.0% / +3.9% / +4.6%.

**Same box, merged 0.1.40 code, no environment switches (arch defaults only).** 4 interleaved pairs per cell, medians. "j" is the
build of the table above, "merged" is the 0.1.40 release code. Every run printed the same token ids. Prompt numbers at 8K swing by
more than the differences because other jobs shared the box, so read them as unchanged.

| Model | Context | Prompt j -> merged | Output j -> merged |
| --- | --- | --- | --- |
| UD-IQ4_XS | 8K | 1,260 -> 1,159 | 53.94 -> 53.14 (-1.5%) |
| UD-IQ4_XS | 128K | 1,317 -> 1,299 (-1.4%) | 51.38 -> 50.72 (-1.3%) |
| UD-Q4_K_XL | 8K | 1,163 -> 1,104 | 52.66 -> 50.47 (-4.2%) |
| UD-Q4_K_XL | 128K | 1,252 -> 1,211 (-3.3%) | 43.28 -> 42.64 (-1.5%) |
| IQ3_S | 8K | 1,283 -> 1,265 (-1.4%) | 59.81 -> 59.10 (-1.2%) |
| IQ3_S | 128K | 1,311 -> 1,310 (-0.1%) | 42.74 -> 42.09 (-1.5%) |
| IQ3_XXS | 8K | 1,127 -> 1,212 (+7.6%) | 51.64 -> 50.95 (-1.3%) |
| IQ3_XXS | 128K | 1,252 -> 1,250 (-0.1%) | 50.88 -> 50.22 (-1.3%) |

With the shared-expert stream (`STRATA_SH_STREAM=1`, a default on Linux; Windows leaves it off since 0.2.7) at 8K, 4 pairs: UD-IQ4_XS output 53.1 -> 54.05 (+1.8%),
UD-Q4_K_XL 49.5 -> 52.8 (+6.7%), prompt unchanged.

**Other switch measurements** (Measured on: the same Ryzen AI Max+ 395)

| Switch | Result |
| --- | --- |
| `STRATA_HC_Q8` (IQ3_S, 2026-10-21, MTP 4 drafts, min-p 0.5) | `json_extract` 53.55 -> 58.03 tok/s (+8.4%); 94 of 250 output tokens differ from the BF16 read (first at 60); cold read 93.8 -> 20.3 us (142 -> 352 GB/s) |
| `STRATA_PF_F16=1`, one layer, 16,384 tokens | Q4_K / Q5_1 85.3 -> 53.5 ms; IQ3_S / IQ4_NL gate/up 11.65 -> 9.87 ms and down 6.11 -> 4.71 ms |
| `STRATA_PF_F16=1`, first-token KL against no switches, 12 prompts of 1.5K-24K tokens, top-1 same in 12 of 12 | UD-Q4_K_XL mean 0.0068 (median 0.0021) against 0.0031 (0.0024) without it; UD-IQ4_XS 0.0085 (0.0036) against 0.0068 (0.0032) |
| `STRATA_REFILL_PARTS=1`, UD-Q4_K_XL and UD-IQ4_XS | 4K-8K prompt +2.3% to +3.5%, 32K +1.1% |
| `--mtp-window 8192` | 64K output +3.5%, 32K -0.4% |
| 12-turn agent replay, `STRATA_PREFILL_STREAM_MIN=128`, C2T8, `--no-prefill-borrow` | 3.4 -> 2.1 s per turn |
| Defaults off, IQ3_XXS 8K | prompt 8% lower (1,137) |

