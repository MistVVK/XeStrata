<!--
SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
-->
# XeStrata - the details

English | [日本語](DETAILS.ja.md)

The technical side of XeStrata: the measured speeds, which model to choose, setup's settings, the API, images and how the engine works.
New here? Start with the [README](../README.md): it has everything you need to install and use it.
How the engine runs on Intel GPUs, and how that was checked, is in [XE.md](XE.md).
The environment variables here are `XESTRATA_*`; upstream's names, `STRATA_*`, are read too (`XESTRATA_*` wins when both are set).
The Japanese version ([DETAILS.ja.md](DETAILS.ja.md)) is the original; this is its translation.

> **On this page:** [Speed](#speed-measured) · [Which model](#which-model) · [What you need](#what-you-need) ·
> [Install and start](#install-and-start) · [Sharing the GPU](#sharing-the-gpu-with-other-programs) ·
> [API](#using-it) · [Web search](#web-search) · [MCP tools](#tools-from-mcp-servers) · [Images](#images) · [Troubleshooting](#troubleshooting) ·
> [How it works](#how-it-works)

---

## Speed (measured)

Every number here was measured on an Arc Pro B70 (32 GB).
The machine: an i7-14700 (AVX2, no AVX-512), DDR4-3200 in two channels (about 35.5 GB/s read), the B70 at PCIe Gen5 x16,
and the model files on a CPU-attached Gen4 x4 NVMe.
The RTX numbers in Strata's (CUDA) README do not apply to XeStrata.

The tables below are commit `58dbbe5` with the settings setup writes (images off), one code-agent prompt per length and 256 generated tokens.
MTP speculative decoding is on.
"262K" is the model's full context window (a 259,942-token prompt).
One run per cell.

### Prefill (tokens/s)

| Model | 1K | 4K | 32K | 64K | 128K | 262K |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **Q2_0** | 726 | 1,116 | 1,834 | 1,532 | 1,393 | 1,122 |
| **IQ2_XS** | 685 | 1,063 | 1,818 | 1,524 | 1,388 | 1,095 |
| **IQ3_XXS** | 577 | 862 | 1,759 | 1,488 | 1,356 | 1,099 |
| **IQ3_S** | 450 | 740 | 1,718 | 1,458 | 1,333 | 1,081 |
| **Coder** | 936 | 1,573 | 2,015 | 1,659 | 1,499 | 1,188 |

While a short prompt (a chunk below 2,048 tokens) is read, the decode CPU pool, idle then, computes the experts outside VRAM that few of its tokens route to (opt-in upstream, the default here: `XESTRATA_PREFILL_CPU_SHARE=auto`). It measures the CPU's and the GPU's time per expert and hands over the share at which both end together. With `XESTRATA_PREFILL_CPU_GATE=1` it also times neighbouring layers with and without the share, and shares only while the layers that share take less time per expert outside VRAM (upstream f2a3b946, #1282: on one PC the busy CPU slowed the host that issues the streams, and every share lost). The layers tried without the share cost more than that saved here, 1.3% of a 1K prompt's speed on the B70 and 4.4% on the RTX 4070, so it is off by default. `XESTRATA_DBG_CPU_GATE=1` prints those readings. The default leaves it off when the GPU cache holds every expert (nothing streams) and when the host is short of memory (less free RAM than the share's buffers and 3 GiB, or file pages of the experts it would read that the free RAM cannot keep) (upstream eaaa237e, 04dbe561; `XESTRATA_PREFILL_CPU_SHARE=auto` turns it on anyway). A buffer of the share that cannot be had ends the share instead of failing the request. The CPU computes in its own activation format, so an answer can differ a little from the GPU-only one. The measured share moved a little from run to run, and the experts the CPU took with it, so two greedy runs with the same settings parted after some tokens. Setup's warm-up therefore measures the share once per PC and model and writes it into the run config (`--prefill-cpu-share`): a prompt is read the same way every run. `--prefill-cpu-share auto` measures it in every prompt as before. The decode's PCIe share also changes from run to run, but it does not change the answer where the experts compute alike on the CPU and the GPU (i-quant, IQ4_XS or Q2_0 gate/up and IQ4_NL or Q2_0 down: the four Qwen3.8 files); to compare two runs' answers exactly on a model with experts of other types, add `--pcie-frac 0` as well. `XESTRATA_PREFILL_CPU_SHARE=0` keeps every expert on the GPU, `0.4` and the like a fixed share. 262- / 914-token prompts went from 319.2 to 383.1 / 736.3 to 828.0 tokens/s on the B70 and from 109.9 to 266.1 / 275.9 to 457.8 tokens/s on the RTX 4070 (one run each).

With a helper GPU (`--expert-cache-device1`, or the helper `--layer-split auto` chose), in the same short chunks the experts not in VRAM that the helper holds and few tokens route to are computed by the helper from its own VRAM. Their tokens' rows go to it through host memory registered on both GPUs, and the results come back the same way. How many tokens an expert may have is where moving its rows over the helper's link and back takes less time than streaming the expert to the first GPU, from both links as measured at the start (19 for a B70 and an RTX 4070 on x4). `XESTRATA_PREFILL_HELPER_ROWS=n` sets it, `0` turns it off. The helper computes with its int8 products, so the reply can differ slightly from the first GPU alone's (the first token's KL on a 1K prompt 0.011; the CPU share at 0.5: 0.020). On a B70 and an RTX 4070 a 1K prompt went from 693 to 735 tokens/s.

The prefill has four opt-in switches, set in the engine's environment. None of them changes a bit of the answers.

- `XESTRATA_PLE_COMPACT=1`: the chunk's PLE rows go up to the GPU still quantized (IQ4_NL) and are dequantized there, about a seventh of the bytes over the host link. Without VRAM for that buffer the rows are dequantized on the host as before.
- `XESTRATA_PLE_COPY_STREAM=1` (with `XESTRATA_PLE_COMPACT=1`): those rows go up on a stream of their own instead of the compute stream.
- `XESTRATA_EMBED_VRAM=1`: the native embedding table in VRAM instead of mapped host memory. Without room for the table plus 512 MiB it stays in host memory, and the log says so.
- `XESTRATA_DEQUANT_SIDE=1`: the dense weights' dequantization to FP16 runs on a side stream, overlapping the product before it. It takes a second scratch buffer.

`XESTRATA_GDN_TC=1` (opt-in, changes the reply, upstream 8c2a84694) runs the DeltaNet recurrence of prompt chunks of 128 tokens or more as a scan over blocks of 32 tokens on the matrix engines (Intel's XMX, NVIDIA's tensor cores, RDNA3 and RDNA4's WMMA). Its products take the values split in two FP16 parts and add in FP32, so the rounding differs from the default FP32 recurrence. A GPU without matrix engines, and `XESTRATA_NO_XMX=1`, keeps the default recurrence. It takes about 60 MiB of VRAM for its scratch.

An opt-in that reads the late layers of a long prompt approximately for a faster prefill (`XESTRATA_KVA=1`, it changes the answers) is in [KVA](KVA.md).

### Decode (tokens/s)

| Model | 1K | 4K | 32K | 64K | 128K | 262K |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **Q2_0** | 78.5 | 84.2 | 72.0 | 67.4 | 63.1 | 56.6 |
| **IQ2_XS** | 86.0 | 96.4 | 81.8 | 73.5 | 76.9 | 62.7 |
| **IQ3_XXS** | 70.7 | 71.9 | 77.0 | 70.1 | 69.4 | 59.1 |
| **IQ3_S** | 77.1 | 73.9 | 72.5 | 68.8 | 63.2 | 50.3 |
| **Coder** | 72.8 | 73.8 | 71.0 | 69.3 | 69.7 | 59.1 |

Decode speed depends on the text as well.
Speculative decoding runs faster when more of the drafted tokens are accepted, so a difference of a few percent between neighbouring cells means nothing
(IQ2_XS is faster at 128K than at 64K because it accepted 0.768 of its drafts there against 0.683).

`--lookup-chain K` (default 0, off) checks, after the draft layer's guesses, up to K more tokens that followed the same
text earlier in the conversation (upstream 2312c83b).
It speeds up handing back a given text nearly as it was, and slows ordinary chat a little
(B70, IQ2_XS, K=3: rewriting a document 109.6 → 129.2, ordinary chat 71.1 → 68.3 tokens/s).
`--lookup-chain-min M` is the match it needs before adding any (default 3 tokens).

In the server's decode, a missed expert (not in VRAM) that a token routes at rank 7 or lower (the 8th to 10th of the
top 10, the smallest weights) contributes zero to that token, and one that every token of a verify window routes that
low is neither read nor computed (`XESTRATA_ROUTE_TAIL_SKIP`, default 7, upstream b4e3407b). The other experts keep their
weights (no renormalisation), so the answers differ slightly. Upstream zeroes it only when no other token of the window
needs the expert, so a token's output depended on the window's other rows; here it is decided for each token.
It does nothing when every expert is in VRAM (nothing is skipped) and with `--batch`; on a layer split each stage skips
its own. With a helper cache (`--expert-cache-device1..3`) the helper GPU computes the experts it holds, and only those
on neither GPU are skipped.
It is not on by default either when 90% or more of the experts are in VRAM (upstream's #1884: the misses are few and the
skip costs more draft acceptance than it saves; a V100 32 GB with Coder IQ1_M at 94.8% decoded 1-2% slower);
`XESTRATA_TAIL_SKIP_MAX_CACHED=<percent>` moves that line (101: always on).
`XESTRATA_ROUTE_TAIL_SKIP=0` turns it off and 1-9 sets the rank; the start prints one line.

Measured (IQ3_XXS):

| | B70 | RTX 4070 | RX 9060 XT |
| --- | --- | --- | --- |
| Mean KL of the decode path (floor: the same setting twice) | 0.0027 (0.0000) | 0.0053 (0.0009) | 0.0049 (0.0012) |
| Top-1 agreement | 99.1% | 97.8% | 98.4% |
| HumanEval (82) and arithmetic tasks (30) passed, off / on | 75, 25 / 76, 24 | 77, 25 / 79, 25 | 77, 24 / 77, 24 |
| Serve decode, off to on (1K and 32K, greedy and temperature 0.7) | -1.0 to +12.7% | +7.5 to +24.5% | +6.3 to +55.4% |

On a GPU whose cache is small `XESTRATA_ROUTE_TAIL_SKIP=4` (opt-in) decodes faster still; its answers drift 3 to 4 times
as far as rank 7's, so it is not the default. Rank 4 against rank 7 (IQ3_XXS, the speeds two rounds each):

| | B70 (VRAM limited to 12 GiB) | RTX 4070 | RX 9060 XT |
| --- | --- | --- | --- |
| Everyday use (context 262144), output | +13 to +26% | +14 to +21% | +14 to +42% |
| Serve decode at 32K (greedy and temperature 0.7) | +6 to +11% | greedy +34%, temperature 0.7 -9% from the texts' lengths | (its cache holds 25% at 32K: rank 7) |
| Mean KL (rank 7 to 4) | 0.0072 to 0.0293 | 0.0066 to 0.0226 | 0.0051 to 0.0187 |
| Top-1 agreement (rank 7 to 4) | 97.6% to 95.2% | 97.3% to 95.5% | 98.1% to 96.7% |

On the RTX 4070 (its cache holds 15%) long agent turns (23 turns from 2.5K to 100K tokens, temperature 1) decode at
80.7 tok/s instead of 67.3 at rank 7, and half of HumanEval (82) and the arithmetic tasks (30) pass 103 at rank 7 and
102 at 4.

`XESTRATA_ROUTE_RESIDENT=<margin>` (upstream 5df35dcb, experimental, opt-in) swaps a verify window's missed experts at
ranks 6-9 of the top 10 for resident ones whose logit is within the margin (`XESTRATA_ROUTE_RESIDENT_RANKS=lo-hi`).
`XESTRATA_ROUTE_PRIOR=<lambda>` (upstream 2de9ae06) replaces the router's missed experts in its top 10 with resident
ones whose logit plus lambda beats theirs (the weights stay the router's).
It is off by default. Unlike the tail skip, which drops a missed expert unread, it adds experts the router did not pick,
and the text suffers: on the RTX 4070 (its cache holds 15%) long agent turns decode 28% faster, but the NLL against the
exact routing rises 8.2%, two tool names came out broken, and 99 of the 112 tasks pass (103 without it; upstream's #4402 also finds
repetition and weaker results in long agent sessions).
When it is on, the expert adaptation still counts the router's own picks (`XESTRATA_ROUTE_PRIOR_COUNT=picked`: the
replaced ones), and once serve's loop guard finds a loop the rest of that reply does without it. Both change the answers.

### Small cards and GPUs without XMX

With the B70 made to look like a small card (`XESTRATA_VRAM_LIMIT_MIB=8192 XESTRATA_MAX_ALLOC_MIB=4096 XESTRATA_NO_XMX=1`: 8 GB of VRAM, no XMX),
IQ2_XS writes 17.6 tokens/s with MTP and reads the 26K-token prompt at 574 tokens/s.
On a GPU without XMX the prefill path's products run through DP4a, so prefill takes about 1.6 times as long.

The prefill path's half-output buffer shares the embedding buffer, which is dead after the first hyper-connection
broadcast (upstream 199fd1d5, #1454): T x 2560 floats less VRAM per chunk (320 MiB at 32,768 tokens).
The planner still counts those bytes by default, so the chunk and the cache slots the prefill path borrows stay as
they were, and so do the prefill's bits.
`XESTRATA_EMB_REUSE_ACCOUNT=1` lets the planner use them: a chunk that VRAM limits grows (upstream 55dee8ed, RTX 3060,
IQ3_XXS: 6,400 to 6,656 tokens, prefill about 3.9% faster).
A different chunk changes the prefill's rounding.

### AMD APUs

On an AMD APU (Ryzen AI Max "Strix Halo" and the like) the CPU and the GPU share one memory (upstream c4a329fe).
The GPU's part is the BIOS carve-out (which the OS does not see, so it is room beside the RAM) and the shared GTT pool.
The GTT pool is the RAM itself, about half of it by the kernel's default.
Setup counts the carve-out as room for experts (the low-RAM mode) and the carve-out plus the shared pool for the recommended context.
It tells Strix Halo (gfx1151) apart by name, names the Radeon 8060S / 8050S / 8040S, and recommends UD-IQ4_XS from about 80 GB of memory.
The kernel's boot options `ttm.pages_limit` and `ttm.page_pool_size` (in 4 KiB pages) give the GPU more of the shared memory.
Setup changes no host setting.
The engine counts an integrated GPU's free memory as the smaller of what its runtime reports and what Linux can give back (`MemAvailable`) less 6 GiB (upstream 004b15cd; `XESTRATA_UMA_HEADROOM_GIB` changes the 6).
For the models that keep a RAM budget of their experts (Unsloth's), setup writes the expert cache as a number, not `auto` (upstream #1715).
The cache and the budget draw on the same RAM, so `auto`, sized from the memory free at start, would count what the budget has not taken yet and leave the OS nothing.
The number is the RAM less 24 GB and the budget (and a KV cache streamed to RAM), in slots; `--expert-cache N` in the config's `args` sets another.

### Where the KV cache lives, and its precision

**KV streaming**: at a context of 64K or more, setup keeps the context's KV cache in RAM.
Only the part the attention reads stays in VRAM (`--kv-resident 32768`), so more experts fit on the GPU.
The attention reads exactly the same values; only where the KV lives changes.
It costs about 13.7 KB of RAM per context token, about 1.7 GB at 128K.

**The elastic K/V** (`--kv-grow` or `XESTRATA_KV_GROW=1`, upstream 9e3dbfe7): the whole KV cache in VRAM, taking VRAM only for the cells the requests reach.
The expert cache holds the rest, giving slots to the K/V as the context grows and taking them back after a short request.
Both live in virtual memory (`sycl_ext_oneapi_virtual_mem`) and neither address moves, so the captured graphs stay valid.
It needs one GPU with virtual memory, an expert profile, no KV streaming and every expert in RAM; otherwise (and with `--batch` or `--vram-elastic`) the engine says so and stays off.
`XESTRATA_KV_GROW_HOLD` keeps the longest parked conversation's cells in the K/V until it comes back (`=1` always, `=0` never).
Unless given, the device's reported virtual-memory granularity decides, as for `--kv-grow` itself: kept at 2 MiB or more.
Where the pages are smaller (the B70's 64 KiB) the cache in virtual memory is slower there anyway, so holding measured slower (faster on the RTX 4070, slower on the B70, 2026-10-09).
At a 128K context, int8, IQ2_XS and a 4,945-token prompt (one run each) the result differs by card:

| | prefill (tokens/s) | decode after it | short chat decode | slots |
| --- | ---: | ---: | ---: | ---: |
| RTX 4070, KV streaming (default) | 791.6 | 73.60 | 58.50 | 4,102 |
| RTX 4070, the whole K/V in VRAM | 770.2 | 66.70 | 56.70 | 3,149 |
| RTX 4070, `--kv-grow` | 802.4 | 76.50 | 61.40 | 4,219 |
| B70, KV streaming (default) | 1133.5 | 100.50 | 77.80 | 19,192 |
| B70, the whole K/V in VRAM | 1245.9 | 101.00 | 78.60 | 18,347 |
| B70, `--kv-grow` | 681.9 | 98.90 | 73.40 | 19,303 |

On the B70 the slow side is the cache in virtual memory: with the K/V alone in it prefill ran at 1291.5 and the decode after it at 105.10, faster than the whole K/V in VRAM.
Unless asked, the device's reported virtual-memory granularity decides: on at 2 MiB or more, off below (the B70 reports 64 KiB).
`--kv-grow` and `--no-kv-grow` (`XESTRATA_KV_GROW=1` and `0`) take precedence over it.

**KV precision**: above an 8K context setup asks for the KV cache's precision (`--kv`).

- `int8` (the default): 8 bits
- `q4_0`: a Hadamard rotation before 4-bit rounding, half the memory.
  Measurably less precise on long documents (upstream measured perplexity 8–12% worse; needle tests still pass)
- `k8v4`: 8-bit keys and 4-bit values, about three quarters of the memory.
  It streams its KV like the other formats (about 10.6 KB of RAM per context token)

## Which model

### How setup asks and recommends

setup (`./setup.sh`, the packages' `xestrata`, the launcher's window) asks for the model, its version (the original, Swift 1.5, the Coder, or whose copy with the refusals removed), its quantization (whose files) and the size, in that order.
A level with one choice is not asked.
On the command line, choose each level with `--base`, `--variant` and `--quantizer`, or all of them with `--family` (`--base qwen3.8-flash-next --variant sc117` is `--family sc117`).
`--model` is the size.

The model chosen first is the first of Qwen3.8-Flash-Next's Swift 1.5 in SC117's copy (where its IQ3_XXS fits), Gyro-Coder-L (where its experts all fit the RAM), Ornith-1.5, Qwen3.6-35B-A3B, Qwen3.5-9B and Qwen3.5-2B that this PC runs (Qwen3.5-2B on a PC without a GPU).
The size is recommended by quality:

- ISTA-DASLab's GSQ-RCO files (the original, Swift 1.5, the Coder, SC117's and huihui-ai's versions): IQ3_XXS; where this PC does not hold it, the best quality it holds (in the low-RAM mode too).
- The files that run on a RAM budget (Unsloth's, Gyro, AtomicChat's AD): the best quality whose experts all fit the RAM; never an experimental one.
- The MoE models other than Qwen3.8-Flash-Next (Ornith-1.5, llm-jp-4.1, Qwen3.6-35B-A3B, GLM-5.3-Flash): the best quality the RAM and VRAM together hold.
- Qwen3.5 (9B, 2B, 0.8B): the largest the VRAM holds. On a PC without a GPU, UD-Q5_K_XL for the 2B and Q8_0 for the 0.8B.
- The models other than Qwen3.8-Flash-Next up to Q8_0, never their unquantized BF16 weights (BF16 answers as Q8_0 does, in twice the memory).

When none fits, the smallest.

### Qwen3.8-Flash-Next's sizes

Every model is a compression of [Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next).
The original's four sizes are [ISTA-DASLab's GSQ-RCO quantizations](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF).

| Size | Download | RAM it uses | Speed | Quality |
| --- | ---: | ---: | --- | --- |
| **Q2_0** | 66 GB | ~34 GB experts + ~6 GB | fastest | good |
| **IQ2_XS** | 68 GB | ~36 GB experts + ~6 GB | close to Q2_0 | a bit better |
| **IQ3_XXS** | 76 GB | ~43 GB experts + ~6 GB | slower (more CPU work) | very good |
| **IQ3_S** | 84 GB | ~50 GB experts + ~6 GB | slowest | best |

With 64 GB of RAM all of them fit (IQ3_S on a PC with little else running).
With 48 GB, Q2_0 and IQ2_XS fit.
With 32 GB, choose the Coder (below).
With less RAM than that, a GPU with enough VRAM still runs them in the low-RAM mode ([below](#less-ram-than-the-model-the-low-ram-mode)).

### Or: the Coder (half the experts, for code)

**[Qwen3.8-Flash-Next GSQ-RCO Coder](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF)** is
ISTA-DASLab's expert-pruned release.
256 of each layer's 512 experts are kept (still 10 active per token), chosen with RCO on code, agentic and vision calibration data.
Its authors report 91.3% of the full model's SWE-bench Verified and 98.7% of LiveCodeBench v6.

It comes in one size, named IQ1_M for its 1.89 bits per *original* parameter.
The kept experts are stored like IQ3_S (IQ2_S–IQ4_XS gate/up, IQ4_NL and Q2_0 down).
Shard 1 is 29.6 GB (experts: 23 GB of RAM), so it runs on **32 GB of RAM**.
Its shard 2 and its image encoder are the original's files: with the original installed, setup downloads only shard 1.
XeStrata ships its expert profile (`data/expert-profile-coder.bin`):
the shipped ranking mapped onto the kept experts through the release's `rco-allocation.txt`.
Images work.
The experimental speed projection loaded and ran on it upstream (it was made for the full model).

```sh
./setup.sh --setup --family coder
```

### Or: Swift 1.5 (a fine-tune that thinks shorter)

setup's first question also offers **[Swift 1.5](https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF)**,
UkisAI's fine-tune of Qwen3.8-Flash-Next, trained to reach the answer with much less thinking.
Its authors report 63% fewer thinking tokens, answers 1.8× sooner, and under 1% accuracy loss.
The same architecture; four sizes (Q2_0, IQ2_XS, IQ3_XXS, IQ3_S); its own image encoder.
Its authors recommend **IQ2_XS** (their Q2_0 is marked experimental).
Its license is the Swift Open License 1.0: read it on the model page.

```sh
./setup.sh --setup --family swift --model IQ2_XS
```

### Or: Unsloth's Dynamic v3.1 (UD-IQ1_M to UD-IQ4_XS)

**[Unsloth's Dynamic v3.1](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF)** (2026-11-04) are 1- to 4-bit versions of the original model (upstream #621's UD-IQ4_XS is now this version too).
The experts' gate and up are IQ2_XXS to IQ4_XS and their downs Q2_0, IQ4_NL and Q5_0; the n-gram table is in the last shard (IQ4_NL).

| Size | Download | Experts | RAM that holds them all |
| --- | ---: | ---: | ---: |
| UD-IQ1_M | 68.0 GB | 34.8 GB | about 59 GB |
| UD-Q2_0 | 70.5 GB | 37.4 GB | about 61 GB |
| UD-IQ2_XS | 75.8 GB | 42.3 GB | about 66 GB |
| UD-Q2_K_XL | 79.6 GB | 46.0 GB | about 70 GB |
| UD-IQ3_XXS | 83.6 GB | 50.3 GB | about 74 GB |
| UD-Q3_K_XL | 89.5 GB | 56.2 GB | about 80 GB |
| UD-IQ4_XS | 93.7 GB | 60.4 GB | about 84 GB |

The dense side and the RAM budget are as UD-Q4_K_XL's below (`--resident-budget-gib N`; setup chooses the RAM less 24 GB).
With less RAM the experts that do not fit are read from the SSD while it answers.
setup recommends the largest size whose experts all fit the RAM (UD-Q4_K_XL aside).
setup asks about images (on by default); the image encoder is the original model's.

```sh
./setup.sh --setup --family unsloth --model UD-IQ3_XXS
```

On an Arc Pro B70 (32 GB) with an i7-14700 and 96 GB of RAM (a 55 GiB RAM budget), 96 tokens of a short chat decoded at 32.5 / 32.8 tokens/s, and the answer read correctly.

### Or: Unsloth's UD-Q4_K_XL (EXPERIMENTAL)

**[Unsloth's UD-Q4_K_XL](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF)** is a 4-bit file of the original model in four shards (111 GB).
Its experts are Q4_K, Q5_K, Q5_1 and Q8_0 blocks, 77 GB of them: more than a 64 GB PC holds.
setup downloads the shards from a pinned revision, checks their SHA-256 once, and packs the model with `--compat-bf16`.
That turns its Q8_0 hyper-connection projections into BF16, the form the engine reads.
The verify window's hyper-connection read (the final mixer's before the head too) can read the GGUF's Q8_0 values themselves (upstream fe6c5260; a BF16 pack's are quantized to Q8_0 at load; 0.62 GiB more VRAM, and the answers change).
At load the engine times the read in its BF16 and Q8_0 forms and keeps the Q8_0 one only when it is faster (`XESTRATA_HC_Q8=1`: always, `=0`: never).
On one GPU, once the Q8_0 read is chosen the layers' BF16 copies leave the VRAM (upstream b60aab1b; 1.17 GiB with IQ3_XXS) and the expert cache gets their room; the prompt path turns the Q8_0 values back into BF16 in a scratch as it reads them.
A layer split cannot drop the BF16, so there a GPU whose Q8_0 copies would take more than 5% of the free VRAM stays BF16.
The Q8_0 read that also writes the next projections' q8_1, 32 columns a work-group (upstream 0a379a1f), is timed too and taken where it is faster (`XESTRATA_HC_Q8_FUSE=0/1` decides; the same answers).
IQ3_XXS's 32K decode: B70 +3.5%, RTX 4070 the same (prefill +1%), RX 9060 XT +0.6% (there with the q8_1 written in the read).
The engine reads the experts from the GGUF files in place.
It keeps a RAM budget of them, the most-used first after the ones the GPU holds (`--resident-budget-gib N`).
When the budget has room left, the slots the prompt path borrows keep their experts in RAM too (the engine prints `the prompt path's lend region: N of its M slots keep their experts in RAM too`; upstream f2e142f4, #1683); before, a budget always left them out, and every prompt read the lent slots' experts from the files again.
setup chooses the RAM less 24 GB for N, less the KV cache too when that lives in RAM:
36 GiB on a 64 GB PC with a 128K context.
The rest come from the SSD while it answers.
Images can be chosen, with a warning that this file has not been run with them (upstream #967; reported working upstream).

```sh
./setup.sh --setup --family unsloth
```

Measured on an Arc Pro B70 (32 GB) and an i7-14700 with 96 GB of RAM, with setup's configuration, 64 generated tokens of a short chat,
every run from a cold OS cache:

| | Decode tokens/s |
| --- | ---: |
| RAM budget 66 GiB (setup's choice for this PC): every expert the GPU does not hold in RAM | 24.3 / 23.5 |
| RAM budget 36 GiB in a 62 GiB memory cgroup (setup's choice for a 64 GB PC) | 17.2 / 17.2 |
| No RAM budget (`--mmap-experts`): every expert the GPU does not hold read through the OS file cache | 10.0–11.1 |

The answers were the same tokens in every run.

### Or: Unsloth's UD-Q5_K_XL, UD-Q6_K_XL and Q8_0 (EXPERIMENTAL)

Unsloth's 5-, 6- and 8-bit files can be chosen too (six shards each, upstream 0.1.44).
Their experts' gate and up are Q5_K, Q6_K and Q8_0, their downs Q8_0, which the engine's expert kernels read as they are.
XeStrata has not run these files, so setup marks them experimental.
The RAM is handled as for UD-Q4_K_XL: with less RAM than below, a RAM budget of the experts is kept in RAM and the rest read from the SSD.

| Size | Download | Experts | RAM, about |
| --- | ---: | ---: | ---: |
| UD-Q5_K_XL | 158 GB | 98 GB | 108 GB |
| UD-Q6_K_XL | 169 GB | 109 GB | 118 GB |
| Q8_0 | 188 GB | 128 GB | 136 GB |

```sh
./setup.sh --setup --family unsloth --model UD-Q5_K_XL
```

### Or: Gyro-S, Gyro-M and Gyro-L

**[Gyro](https://huggingface.co/agentionai/Qwen3.8-Flash-Next-Gyro-GGUF)** is agentionai's quantization of the original model (upstream ba64156fd).
Its experts are rotated by Hadamard matrices and stored in trellis formats (TQ2_T, TQK6, TQK7; TQP5, TQP6 and TQP7 for Gyro-L).
In RAM alone, Gyro-S needs 36 GB, Gyro-M 43 GB and Gyro-L 59 GB. With less RAM it runs in the [low-RAM mode](#less-ram-than-the-model-the-low-ram-mode), where the experts the VRAM holds are not kept in RAM, so 38 GB, 45 GB and 61 GB of RAM and VRAM together are enough.
The files hold the n-gram table too (read from the SSD), so the downloads are 58 GB, 92 GB and 108 GB.
Gyro-M is about IQ3_XXS's quality with about 20% less GPU memory.
Gyro-L is the best of the Gyro sizes, and decodes slower than Gyro-M.
Gyro-S is the smallest, of a lower quality than the 2- and 3-bit sizes, and Q2_0 decodes faster.
Images work too. The model's license is Qwen3.8-Flash-Next's, the Qwen Community License 1.0.

```sh
./setup.sh --setup --family gyro --model Gyro-M
```

### Or: Gyro-Coder-L and Gyro-Coder

**[Gyro-Coder](https://huggingface.co/agentionai/Qwen3.8-Flash-Next-Gyro-Coder-GGUF)** is agentionai's version for code.
Each layer keeps 256 of its 512 experts, the ones code needs (a token still uses 10), rotated by Hadamard matrices and stored in trellis formats as Gyro's are.
Gyro-Coder-L's experts are TQP5, TQP6 and TQP7, which set two weights a step (about 3.2 bits a weight); Gyro-Coder's are TQ2_T (about 2.1 bits).
In RAM alone, Gyro-Coder-L needs 35 GB and Gyro-Coder 26 GB. As for Gyro, with less RAM it runs in the [low-RAM mode](#less-ram-than-the-model-the-low-ram-mode), and 37 GB and 28 GB of RAM and VRAM together are enough.
The files hold the n-gram table too (read from the SSD), so the downloads are 84 GB and 76 GB.
The two keep the same experts, so XeStrata ships one expert profile for them (`data/expert-profile-gyro-coder.bin`), made from the routing of answers to coding requests.
It is weaker outside code. Images work too (with the original model's image encoder). The model's license is the Qwen Community License 1.0.

```sh
./setup.sh --setup --family gyro-coder --model Gyro-Coder-L
```

### Or: huihui-ai's abliterated versions

**[huihui-ai's abliterated versions](https://huggingface.co/huihui-ai/Huihui-Qwen3.8-Flash-Next-abliterated-GGUF)** have the refusal direction removed from the model (abliteration).
They refuse far less, and their safety filtering is weaker, so check what they write.

- `--family huihui-swift`: Swift 1.5's IQ2_XS and IQ3_XXS, under the Swift Open License 1.0.
- `--family huihui-unsloth`: Unsloth's UD-IQ4_XS (the three shards from before Dynamic v3.1), UD-Q4_K_XL and Q8_0, under the Qwen Community License 1.0.

The files' names, shards and sizes are the originals'; the weights differ. The picture readers are the originals'.
The experimental speed projection does not work with them.

```sh
./setup.sh --setup --family huihui-swift --model IQ2_XS
```

### Or: SC117's abliterated versions

**[SC117's abliterated versions](https://huggingface.co/SC117/Qwen3.8-Flash-Next-GSQ-RCO-abliterated-GGUF)** are ISTA-DASLab's four sizes (Q2_0, IQ2_XS, IQ3_XXS, IQ3_S) with the weights of OrcaRouter's abliterated version put in (`--family sc117`).
What was put in is each layer's `ssm_out`, `attn_output` and the shared and routed experts' down; the rest is ISTA-DASLab's files as they are.
They refuse far less, and their safety filtering is weaker, so check what they write.

The projections put in are Q8_0, so the first shard is 0.2 to 0.4 GB larger than the original's.
The second shard (the n-gram table) is the original's file, shared with the original when that is installed.
The picture reader and the MTP draft head are the original model's.
The experimental speed projection does not work with them. The license is the Qwen Community License 1.0.

Swift 1.5's four sizes have the same done to them ([SC117's abliterated versions of Swift 1.5](https://huggingface.co/SC117/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-abliterated-GGUF), `--family sc117-swift`).
The projections put in keep the abliterated version's types (IQ4_XS and the like), and the second shard is a file of its own for each size.
The picture reader is Swift 1.5's, and the license is the Swift Open License 1.0.

```sh
./setup.sh --setup --family sc117 --model IQ3_XXS
./setup.sh --setup --family sc117-swift --model IQ3_XXS
```

### Or: models other than Qwen3.8-Flash-Next (Ornith-1.5, llm-jp-4.1, Qwen3.6-35B-A3B, Qwen3.5-9B, GLM-5.3-Flash)

setup's first question also offers four MoE models built differently from Qwen3.8-Flash-Next, and the dense
Qwen3.5 models of 9B, 2B and 0.8B.
Each is read from its GGUF file as it is (GLM-5.3-Flash's in three or four parts).
The MoE models keep the most used experts in the GPU's VRAM and the rest in RAM (the ones VRAM does not hold are
handled as for Qwen3.8-Flash-Next).
Qwen3.5's 9B, 2B and 0.8B keep every weight in the GPU's VRAM; 2B and 0.8B also run on the CPU alone, on a PC without
a GPU ([a PC without a GPU](#a-pc-without-a-gpu)).

- **[Ornith-1.5-35B-A3B](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B-GGUF)** (Ornith AI, MIT): a Qwen3.5-family
  MoE of 35B with 3B active per token, context up to 256K. The official Q4_K_M to BF16, and pfeifferj's GSQ-RCO 3 and
  3.5 bits (a [community reproduction](https://huggingface.co/pfeifferj/Ornith-1.5-35B-A3B-GSQ-RCO-GGUF), not
  ISTA-DASLab's).
  dealignai's version with its refusals removed,
  [Ornith-1.5-35B-A3B-UNCENSORED](https://huggingface.co/dealignai/Ornith-1.5-35B-A3B-UNCENSORED-GGUF) (MIT, Q4_K_M, Q5_K_M,
  Q6_K and Q8_0, with the tensor types of the same official versions), can be chosen too
  (`--family ornith-uncensored`). It refuses far less, and its safety filtering is reduced, so check what it writes.
- **[llm-jp-4.1-32b-a3b-thinking](https://huggingface.co/llm-jp/llm-jp-4.1-32b-a3b-thinking-gguf)** (LLM-jp,
  Apache-2.0): a thinking model of 32B with 3B active per token, in Japanese and English, context up to 256K.
- **[Qwen3.6-35B-A3B](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-MTP-GGUF)** (Qwen, Apache-2.0): a thinking
  Qwen3.5-family MoE of 35B with 3B active per token, context up to 256K. The GGUFs are Unsloth's UD-Q4_K_XL,
  UD-Q6_K_XL and Q8_0 with the MTP layer.
  llmfan46's version with its refusals removed by Heretic and MPOA, the MTP layer kept, quantized by SC117 with APEX, [Qwen3.6-35B-A3B-uncensored-heretic-Native-MTP-Preserved-APEX](https://huggingface.co/SC117/Qwen3.6-35B-A3B-uncensored-heretic-Native-MTP-Preserved-APEX-GGUF) (Apache-2.0; APEX-I-Quality, APEX-I-Balanced, APEX-I-Compact and APEX-I-MINI, 14.3 to 26.0 GB), can be chosen too (`--family qwen36-uncensored`). It refuses far less and its safety filtering is weaker, so check what it writes.
- **[Qwen3.5-9B](https://huggingface.co/unsloth/Qwen3.5-9B-GGUF)** (Qwen, Apache-2.0): a dense thinking model of 9B
  that reads images too, context up to 256K. The GGUFs are Unsloth's Q4_K_M, Q6_K, Q8_0 and BF16.
  HauhauCS's version with its refusals removed,
  [Qwen3.5-9B-Uncensored-HauhauCS-Aggressive](https://huggingface.co/HauhauCS/Qwen3.5-9B-Uncensored-HauhauCS-Aggressive)
  (Apache-2.0, the same four sizes), can be chosen too (`--family qwen35-uncensored`). It refuses far less, and its
  safety filtering is reduced, so check what it writes.
- **[Qwen3.5-2B](https://huggingface.co/unsloth/Qwen3.5-2B-GGUF), [Qwen3.5-0.8B](https://huggingface.co/unsloth/Qwen3.5-0.8B-GGUF)**
  (Qwen, Apache-2.0): small dense thinking models built as the 9B, that read images too, context up to 256K.
  For a small GPU or a PC without one; their answers fall short of the 9B's (and the 0.8B's of the 2B's).
  The GGUFs are Unsloth's: Q4_K_M, UD-Q5_K_XL, Q6_K, Q8_0 and BF16 for the 2B, Q4_K_M, Q6_K, Q8_0 and BF16 for the
  0.8B (`--family qwen35-2b`, `--family qwen35-0.8b`). setup recommends the largest up to Q8_0 that the GPU holds
  (without a GPU, UD-Q5_K_XL for the 2B and Q8_0 for the 0.8B).
- **[GLM-5.3-Flash](https://huggingface.co/unsloth/GLM-5.3-Flash-GGUF)** (Z.ai, MIT): a thinking MoE of 320B with 18B
  active per token; setup offers contexts up to 256K. Its attention is KDA (linear attention) and DSA (attention over
  the tokens it selects), with four residual streams (mHC). The GGUFs are Unsloth's UD-IQ1_S, UD-IQ3_XXS and UD-IQ4_XS.

| Model | Size | Download | RAM, about | VRAM |
| --- | --- | ---: | ---: | ---: |
| Ornith-1.5 | **Q4_K_M** | 21.7 GB | 30 GB | 4 GB and up |
| | Q5_K_M | 25.3 GB | 34 GB | 6 GB and up |
| | **Q6_K** | 29.2 GB | 38 GB | 6 GB and up |
| | Q8_0 | 37.8 GB | 46 GB | 6 GB and up |
| | **BF16** | 71.1 GB | 80 GB | 8 GB and up |
| | **GSQ-RCO-3BIT** | 13.0 GB | 21 GB | 4 GB and up |
| | GSQ-RCO-3.5BIT | 15.2 GB | 24 GB | 6 GB and up |
| Ornith-1.5 (refusals removed) | Q4_K_M | 21.7 GB | 30 GB | 4 GB and up |
| | Q5_K_M | 25.3 GB | 34 GB | 6 GB and up |
| | Q6_K | 29.2 GB | 38 GB | 6 GB and up |
| | Q8_0 | 37.8 GB | 46 GB | 6 GB and up |
| llm-jp-4.1 | **Q4_K_M** | 21.5 GB | 30 GB | 4 GB and up |
| | BF16 | 64.3 GB | 73 GB | 8 GB and up |
| Qwen3.6-35B-A3B | **UD-Q4_K_XL** | 22.9 GB | 31 GB | 4 GB and up |
| | **UD-Q6_K_XL** | 32.6 GB | 41 GB | 6 GB and up |
| | Q8_0 | 37.8 GB | 46 GB | 6 GB and up |
| Qwen3.5-9B | Q4_K_M | 5.7 GB | 8 GB | 8 GB and up |
| | Q6_K | 7.5 GB | 8 GB | 10 GB and up |
| | Q8_0 | 9.5 GB | 8 GB | 12 GB and up |
| | BF16 | 17.9 GB | 8 GB | 20 GB and up |
| Qwen3.5-2B | Q4_K_M | 1.3 GB | 8 GB | 3 GB and up |
| | **UD-Q5_K_XL** | 1.5 GB | 8 GB | 3 GB and up |
| | Q6_K | 1.6 GB | 8 GB | 3 GB and up |
| | Q8_0 | 2.0 GB | 8 GB | 4 GB and up |
| | BF16 | 3.8 GB | 8 GB | 6 GB and up |
| Qwen3.5-0.8B | Q4_K_M | 0.5 GB | 8 GB | 2 GB and up |
| | Q6_K | 0.6 GB | 8 GB | 2 GB and up |
| | **Q8_0** | 0.8 GB | 8 GB | 3 GB and up |
| | BF16 | 1.5 GB | 8 GB | 3 GB and up |
| GLM-5.3-Flash | **UD-IQ1_S** | 93.1 GB | 101 GB | 12 GB and up |
| | **UD-IQ3_XXS** | 120.4 GB | 128 GB | 12 GB and up |
| | UD-IQ4_XS | 156.8 GB | 165 GB | 12 GB and up |

The RAM is about the download's size plus 8 GB (for GLM-5.3-Flash, RAM and VRAM together).
With less, the experts that RAM and VRAM do not hold are read from the model's files, which is slower.
Qwen3.5's VRAM is its weights and, at a 32K context, its KV cache (a token's: 32 KB for the 9B, 12 KB for the 2B and
the 0.8B) and about 1 GB more (the same for the version without refusals).
setup recommends the largest version up to Q8_0 the GPU's VRAM holds and warns of a choice it does not; weights that do not fit
the VRAM are not streamed from RAM yet.
The sizes in bold are the files whose answers this release compared with llama.cpp's.
The others are made of the same tensor types, checked on those files and in the kernels' tests.

```sh
./setup.sh --setup --family ornith --model Q4_K_M
./setup.sh --setup --family ornith-uncensored --model Q4_K_M
./setup.sh --setup --family llmjp
./setup.sh --setup --family qwen36 --model UD-Q4_K_XL
./setup.sh --setup --family qwen36-uncensored --model APEX-I-Quality
./setup.sh --setup --family qwen35 --model Q4_K_M
./setup.sh --setup --family qwen35-uncensored --model Q8_0
./setup.sh --setup --family glm53 --model UD-IQ3_XXS
```

The API works, with the thinking kept apart from the answer, streaming and tool calls.
Ornith's official versions (Q4_K_M to BF16) and the ones with its refusals removed, Qwen3.6 and GLM-5.3-Flash draft up to two tokens ahead with the model's MTP layer and check them in a window (speculative decoding; the GSQ-RCO versions and llm-jp have no MTP layer).
Requests with a temperature take it too: each row of a window draws from a random number its position fixes, and a draft is kept only where it equals that draw, so the text is the one without drafts (Qwen3.6-35B-A3B on the B70, 400 tokens at temperature 0.8: decode 92.9 -> 131.7 tokens/s).
With two or more GPUs chosen by `setup --gpus`, the layers are split across them (the first GPU runs the first layers,
the next the later ones, the head and the MTP layer; the points follow each GPU's share of the free memory), and the
experts in VRAM are all the GPUs' together.
`setup`'s `--kv`, `--parallel`, `--replicas` and `--resident-budget-gib N` (N GiB of the experts the GPU does not hold in RAM, the rest read from the file) go into the config as for Qwen3.8-Flash-Next (`--kv`'s `q4_0` and `k8v4` for the models with heads of 256, `fp16` and `int8` for llm-jp, `fp16` alone for GLM-5.3-Flash; `--parallel` not for GLM-5.3-Flash nor with images).
Ornith-1.5, Qwen3.6-35B-A3B and Qwen3.5's 9B, 2B and 0.8B read images (2B and 0.8B on a GPU only; setup asks as for
Qwen3.8-Flash-Next, and takes the GPU reader on a GPU of 24 GB and more, the CPU reader below that).
Each picture reader (mmproj) is its publisher's: Ornith AI's BF16 for Ornith-1.5 (pfeifferj's GSQ-RCO files use it too),
dealignai's for its version without refusals, Unsloth's BF16 for Qwen3.6-35B-A3B and Qwen3.5's 9B, 2B and 0.8B (each repository's own), and SC117's F16 for
Qwen3.6-35B-A3B's version without refusals.
Not yet: llm-jp-4.1's and GLM-5.3-Flash's images, and the options made for Qwen3.8-Flash-Next such as the low-RAM modes (GLM-5.3-Flash's KV cache is FP16 only for now).
When RAM is short of all the experts, the engine keeps the ones VRAM does not hold in RAM as far as it has room and
reads the rest from the model file through the OS file cache (slower; setup warns of it when the RAM and the VRAM
together are short).
`XESTRATA_RAM_ADAPT=<n>` (above) makes the RAM copy follow the conversation then, on one GPU.
`XESTRATA_DISK_SKIP_FROM=R` (opt-in, changes the answer, after Project Maya's #109): an expert held neither in VRAM nor in the RAM copy, read from the file, contributes zero to the tokens that route it at rank R or lower, and one that every token of the window routes that low is not read (the SSD is not waited for). GLM-5.3-Flash UD-IQ3_XXS on the B70 with its RAM held to 56 GB decoded 4.33 -> 5.11 tokens/s with R=6 (the same 128 tokens) and 5.93 with R=4 (parting at the 13th token).

The engine's options (added to the settings' `args`):

| Option | Meaning |
| --- | --- |
| `--max-context N` | the KV cache's tokens (setup picks it from the GPU's VRAM) |
| `--kv-ram on\|off` | GLM-5.3-Flash: the DSA layers' latent cells (1 KB a position) in RAM instead of VRAM, which goes to the experts (on by default); a decode window copies the positions each token selected into VRAM, a prompt chunk each layer's cells up to it. With 128K positions the B70's VRAM tier went from 2,150 to 2,322 experts, the RTX 4070 could start at 128K at all, and at 64K its prompt went from 3.3 to 7.1 tokens/s |
| `--kv fp16\|int8\|q4_0\|k8v4` | the attention layers' KV cache in FP16 (the default) or as int8 codes with an FP16 scale per 64 values: half the VRAM, the answers differ slightly. `q4_0` keeps K and V as Q4_0 after the Hadamard rotation (about 28% of FP16), `k8v4` K in int8 and V as rotated Q4_0 (about 40%); these two for the models with heads of 256 (Ornith, Qwen3.6, Qwen3.5-9B). Qwen3.5-9B's 32K prefill on the B70: FP16 20.6 s, `q4_0` 20.4 s, `k8v4` 19.8 s (none of them for GLM-5.3-Flash) |
| `--kv-grow`, `--no-kv-grow` | with `--serve` on one GPU, the KV cache in virtual memory, taking VRAM only for the cells the requests reach: the VRAM tier of experts (in `--vram-elastic`'s segments) gives it up before a long request and takes it back after a short one. On by default, as Qwen3.8-Flash-Next's, where the GPU's virtual memory pages are 2 MiB or more (the RTX 4070, not the B70). On the RTX 4070, Qwen3.6-35B-A3B at a 128K context started with 4,182 slots instead of 2,870 and its 32K prefill went from 32.9 s to 29.0 s (not with the batch slots) |
| `--control-vector-scaled FILE:SCALE[,...]`, `--control-vector FILE`, `--control-vector-layer-range A B`, `--cvec-mode`, `--cvec-dir` | a control vector in llama.cpp's format on the residual stream after each layer's FFN (Qwen3.8-Flash-Next's meanings). With `--serve` a request switches it (`experimental_speed_projection`), and a switch reads the conversations again (not for GLM-5.3-Flash yet) |
| `--expert-profile PATH` | fill the experts in VRAM in this profile's order at start |
| `--expert-profile-save PATH` | write what this run learned as a profile |
| `--adapt-every N` | how often the experts in VRAM are swapped: at every 4N-th token of a request; where a layer rounds differently on the CPU and the GPU the windows are cut there, so the text does not depend on the drafting. Qwen3.8's four files decode to the same bits on the CPU and the GPU and are not cut, but serve's prompts from the second turn on round with the swaps' timing, so the text can differ from run to run (`XESTRATA_ADAPT_CUT=1` cuts them) (the batch slots, `--pipeline-windows` and `--adapt-async`: every N windows) (default 4; 0: no swaps). Without it, when the CPU pool took 30% or more of the checks' time at the 32nd output token, the swaps come every 4 tokens; not when the experts the GPU does not hold are read from the files (`--mmap-experts`; with the models other than Qwen3.8-Flash-Next, also when the RAM is short of all the experts and some of them are read) |
| `--adapt-swaps N` | the most experts one swap moves (default 96; the one right after a prompt moves up to 2,048, those the prompt used most, and fills every free slot with any expert it used) |
| `--adapt-mib-per-token N` | the swaps copy at most N MiB for each token read since the last one (default 48; 0: no bound) |
| `--adapt-min-gain F` | the bar a swap must clear: the decayed routing count of the expert coming in against the one going out (default 1.5) |
| `--adapt-async 0\|1` | 1 (the default): a swap's copies run beside the next windows, and the experts it brings in are used three windows on; 0: the swap waits for its copies (as Qwen3.8-Flash-Next's `--adapt-async 0`) |
| `--mmap-experts` | read the experts from the model file instead of loading them into RAM (setup writes it with `--low-ram on` or `mmap`) |
| `--expert-cache-device1 N` (and `2`, `3`), `--expert-cache-remote-placement stripe\|layer` | N experts on a helper GPU (from the first GPU no stage of a layer split runs on), which computes the rows routed to them instead of the CPU (Qwen3.8-Flash-Next's meanings); filled in the profile's order, then expert by expert across the layers. Qwen3.6-35B-A3B with 3,000 slots on the B70 and 6,000 experts on an RTX 4070 decoded 166.8 tokens/s instead of 100.0 |
| `--shared-expert-arena FILE` | share the experts' RAM copy with the other engines that name the same FILE (Linux: the first to load fills it, the rest map it; Qwen3.8-Flash-Next's; not with `--mmap-experts` or the RAM tier) |
| `--resident-experts`, `--resident-budget-gib N`, `--resident-cpu-experts` | read the experts in place and copy those no VRAM tier holds into RAM in the profile's order, as far as the RAM allows (`--resident-budget-gib`: up to N GiB); `--resident-cpu-experts` keeps that copy in ordinary RAM, not page-locked (Qwen3.8-Flash-Next's names) |
| `--no-pcie-split` | the CPU computes every expert VRAM does not hold |
| `--pcie-frac F` | that share of the experts VRAM does not hold read by the GPU over PCIe in every window, instead of the measured split (a request's `strata_tune` `pcie_frac` too) |
| `--vram-elastic` | with `--serve` on one GPU: the experts' VRAM tier in segments (`--vram-segment-mib`, default 512), given back before an image job and taken again after it (`--image-gpu-policy shrink`) |
| `--prefill-cpu-share S` | the CPU pool's share of the experts outside VRAM in prefill (0-1; the default `auto` takes it from the times measured at start; 0: all on the GPU). The CPU rounds differently from the GPU, and which experts it takes follows which are in VRAM, a tier that `--expert-cache auto` sizes from the free VRAM at start: a greedy answer to the same question can differ a little from run to run. With 0 it is the same each time |
| `--pool-workers N` | the CPU threads that compute experts (default one fewer than the physical cores, at most 32; in a Linux container with a CPU quota (cgroup), at most one fewer than the quota) |
| `--pool-cores LIST`, `--aux-cpus SET`, `--pool-tasks N`, `--no-host-worker`, `--memory-limit-mib N` | the CPUs of the expert workers, the CPUs of the threads that are neither workers nor the host (a request's `strata_tune` `aux_cpus` too), the pool's tasks a phase, the host thread kept out of the drain, the RAM the parking and saving of conversations counts (Qwen3.8-Flash-Next's meanings) |
| `--prompt-chunk N` | the most tokens prefill computes at once (default 4096, or a larger chunk the VRAM tier of experts lends, up to 32768, its buffers within 85% of the slots; 0: none at once) |
| `--max-t N` | the decode window's most tokens (default 8) |
| `--spec N` | the most tokens of a window checking the MTP layer's drafts (default 3: two drafts; 0: off) |
| `--spec-window all\|auto\|min-p` | how many of the drafts a window checks: `all` (the default) every draft made; `auto`, as Qwen3.8-Flash-Next does, while the drafts' probabilities and the measured window times say a draft raises the tokens per ms; `min-p` while a draft is at least `--spec-min-p P` likely |
| `--draft-vocab PATH` | the MTP layer drafts only tokens of this subset (`data/draft_vocab*.bin`), reading less of the head (setup writes it into Qwen3.6's settings, after its `--draft-vocab` choice) |
| `--coupled-draft`, `--no-coupled-draft` | under sampling, the MTP layer's drafts are drawn too, with the request's chain (temperature, top_k, top_p, min_p) and the random number of the window row that checks each (Qwen3.8-Flash-Next's; off by default) |
| `--suffix-draft N`, `--lookup-chain K`, `--lookup-chain-min M` | with `--serve`, drafts from an earlier repeat of the conversation's text (a match of its last N tokens or more), checked in a window where they go past the MTP layer's drafts (also on a model without that layer); `--lookup-chain` adds up to K such tokens after the MTP's drafts (off by default) |
| `--layer-split K[,K2..]` / `auto` | split the layers across the GPUs: the first runs the layers before K, the next those from K on, ... (`auto`: by each GPU's share of the free memory; setup's `--gpus` writes it) |
| `--split-device D[,D2..]` | the later GPUs (engine numbers or PCI addresses; by default the other discrete GPUs) |
| `--pipeline-windows N` | with a layer split, 1 (the default) pipelines the windows a prompt is read through over the stages (an earlier stage computes the next window while a later one computes this one; the rows and their order are the serial ones, so the state is too); 2, with two stages and the MTP layer, the decode too: the first stage computes a guess of the next window while the last checks this one (when this one holds whole and its next token is the draft layer's, the guess is the next window; else the first stage goes back), only where the chance that it holds (the draft layer's probability, scaled by the share of the guesses so far that held) times the first stage's time it saves exceeds the time a guess beside a window adds to it, each time measured over the windows; until they are measured, where that chance is 0.5 or more (Qwen3.8-Flash-Next's default is 0.2). The windows are sized by the serial rule (`--spec-window`). The text is that of the serial windows with the experts in the same places (2 keeps the second verifier's VRAM out of the experts, so their places change, and a text at temperature can change with the CPU's and the GPU's rounding). Qwen3.6-35B-A3B on a B70 + RTX 4070 split at layer 20: a copy reply 187.2 -> 247.2 tokens/s, a story at temperature 0.8 134.5 and 135.7 -> 133.0 and 134.1. Qwen3.8-Flash-Next on the same pair split at layer 24 (the experts not swapped): a story, greedy, serial 65.9 -> 67.2 and 67.5 tokens/s, at temperature 0.8 60.7 -> 60.9 and 64.0 (57.6 and 57.9, 54.7 and 55.1 when `--spec` sized the windows). 0: serial. Qwen3.6-35B-A3B on a B70 + RTX 4070 split at layer 20, 819 tokens with `--prompt-chunk 0`: prefill 108.5 -> 139.3 tokens/s, the same answer. A long prompt read in chunks has its chunks pipelined over the stages the same way (13,506 tokens on the same split: prefill 1,030.8 -> 1,357.9 tokens/s, the same answer) |
| `--vram-reserve-later-mib N` | with a layer split, the VRAM left free on the later GPUs (by default as `--vram-reserve-mib`) |
| `--vision` | with `--serve`, take requests with images (Qwen3.5-9B and the qwen35moe models; setup writes it when images are chosen) |
| `--device cpu`, `--cpu-threads N` | run a dense Qwen3.5 model (qwen35) on the CPU alone ([a PC without a GPU](#a-pc-without-a-gpu); without a GPU that SYCL can use it runs on the CPU unasked). `--cpu-threads` is the number of threads (default: physical cores, subject to the CPU quota) |
| `--batch N` (`"parallel": N`) | with `--serve`, decode up to N conversations together ([BATCHING](BATCHING.md#the-other-models-the-plain-engine); not GLM-5.3-Flash nor `--vision` yet) |
| `--prompt-cache N`, `--prompt-cache-every N`, `--prompt-cache-root N`, `--turn-token ID`, `--conversation-save-checkpoints N` | with `--serve`, the conversation checkpoints of a model with a recurrent state (Ornith, Qwen3.6, Qwen3.5-9B). A recurrent state cannot go back to an earlier position, so a request that takes a conversation up short of its end (the next turn after a reply rendered otherwise, an edited message, a new chat on the same system prompt) reads all of it again without one. The meanings and defaults are Qwen3.8-Flash-Next's ([the conversation cache](#the-conversation-cache)), and a request's `pin` and `"strata_checkpoint": false` apply. A checkpoint is the recurrent state only (about 60 MB for Qwen3.6); the K/V of the positions below it are the conversation's. `--turn-token` defaults to the vocabulary's `<|im_start|>`, else `<|start|>`. A model of attention layers only (llm-jp, GLM-5.3-Flash) needs none: the K/V up to any position serve |
| `--tail-role-token ID`, `--prompt-cache-tail`, `--prompt-cache-spread`, `--conversation-cache-min-tokens N`, `--conversation-cache-park-gain N`, `--short-read N`, `--session-save-reclaim` | `--serve`'s policies of the conversation cache and the checkpoints (Qwen3.8-Flash-Next's meanings); `--short-read` defaults to `--max-t` |
| `--rope-scaling none\|linear\|yarn`, `--rope-scale F`, `--yarn-orig-ctx N`, `--yarn-ext-factor F`, `--yarn-attn-factor F`, `--yarn-beta-fast F`, `--yarn-beta-slow F` | context extension (the YaRN knobs as llama.cpp's) |

#### A PC without a GPU

When setup finds no GPU, it offers Qwen3.5-2B and 0.8B, which run on the CPU alone (the 2B by default).
The engine runs with `--device cpu`: the weights repacked as int8 and computed on the CPU. It needs AVX2, and takes
AVX-VNNI, AVX-VNNI-INT8 or AVX-512 VNNI where the CPU has them.
On a CPU with AVX2 alone the weights are kept in seven bits (faster; the answers differ slightly.
`XESTRATA_I8_GEMM_WEIGHT_BITS=8` keeps eight).

The RAM is the int8 weights (about 2.4 GB for the 2B, 0.9 GB for the 0.8B), the KV cache in FP32 (24 KB a token, taken
for the whole context at the start) and about 2 GB more: at a 32K context about 5 GB for the 2B and 4 GB for the 0.8B,
at 256K about 11 GB and 9 GB. setup recommends the longest context up to 32K that the RAM holds.

On the CPU there are no images to read (a request with one is answered that the CPU reads text only), no decode of
several conversations together (`--parallel`), no choice of the KV cache's precision (`--kv`: FP32 only) and no image
generation.
The conversation cache, the checkpoints and the conversation save work as on a GPU.

On an i7-14700 (AVX-VNNI), reading a prompt of 1,692 tokens and giving a short answer took about 1.1 s with the 0.8B
and 2.1 s with the 2B. A long prompt is slow: 31,000 tokens took the 2B about two and a half minutes.

### Less RAM than the model: the low-RAM mode

Every model but Unsloth's sizes (huihui-ai's versions of them too) keeps all of its experts in RAM, and the GPU holds a copy of the most-used ones.
When the experts do not fit the RAM (with about 10 GB left for the rest), setup chooses the low-RAM mode (`--low-ram auto`, the default).
The fewer experts the GPU holds, the more are read from the files, and the slower it is.
The engine then reads the experts from the model files instead of copying them all into RAM:
a native pack's from the GGUF files in place, the AVX-512 Q2_0 pack's from its `experts.bin`.
An engine started by hand with neither `--resident-experts` nor `--mmap-experts` starts as with `--resident-experts`, and says so in one line, when the free RAM cannot hold all the experts and 8 GiB more (issue #10: the whole set swapped while it loaded, and on an Intel GPU, once moved into the GPU driver's memory, the OS could not take the RAM back and the host stopped).
`XESTRATA_ARENA_LOW_RAM=keep` loads the whole set anyway.
With every expert in RAM, the move into the GPU driver's memory (host USM) for the PCIe share takes a layer only while the RAM has room for it and `XESTRATA_HOST_USM_RESERVE_MIB` (default 1024) more; otherwise nothing moves and the CPU computes the missed experts.

- If the experts the GPU does not hold fit the RAM, they are copied into RAM once at start (`--resident-experts`).
  The copy is page-locked only when the RAM left free after it is more than 8 GiB and what is still read from the model files (the experts neither on the GPU nor in the copy) together (issue #11: a B70 with 31 GB of RAM and 17.5 GiB page-locked stalled the GPU's copies while the files were read again).
  The engine says at start which it chose, with the numbers. `XESTRATA_RESIDENT_PIN=0` never page-locks it, `1` always does.
  The server then moves the adaptive tier's swaps (exchanges between the RAM copy and the GPU cache) on a helper
  thread between verify windows (`--adapt-async`, opt-in upstream in b437b8ea, the default here).
  No window waits for a whole round of swaps.
  On an RTX 4070, swapping about 95 experts a round, decoding ran 14-15% faster (the B70 within the run-to-run spread).
  Which window first computes a swapped-in expert on the GPU depends on when its copy lands, so an answer can differ a little from run to run.
  `--adapt-async 0` restores the tier in which a window waits.
  Without page-locked exchange buffers, and with the pipelined groups of `--batch-groups`, the waiting tier runs (beside the `--batch` slots it moves on between their windows, upstream 495dc436).
  `--adapt-min-gain F` (default 1.5) is the bar a swap must clear (the decayed routing count of the expert coming in against the one going out); higher means fewer swaps a round. Both tiers use it.
  With `--pipeline-windows 2` ([MULTIGPU](MULTIGPU.md)) the decode loop queues each card's copies while that card has no window in flight, and the copies into the evicted slots and the moves into RAM wait until the windows that were in flight when they were decided have completed (upstream 13eee452; `XESTRATA_PIPELINE_ADAPT_ASYNC=0`: the waiting tier).
  `STRATA_EXCHANGE_ROTATE=1` (opt-in, upstream 1e6cec80) hands the exchange buffer's ownership over at a swap instead of
  copying the evicted expert into the RAM copy (the same tokens, fewer host copies).
  It works only with a fully page-locked RAM copy; a native pack whose blobs differ by layer (IQ3_XXS has nine sizes) keeps its spare buffers per size (upstream fa1d1d1e).
  That takes more page-locked RAM (1.20 GiB with IQ3_XXS and `--adapt-swaps 80`). On the B70, IQ3_XXS, 4K, `--resident-experts --adapt-every 1 --adapt-swaps 80` decoded +1.4% (73.2 -> 74.3 tok/s) with the same tokens.
- `XESTRATA_RAM_ADAPT=<n>` (opt-in, `--serve`, one GPU, upstream d7298883, 2b13a4493): the RAM copy follows the conversation too.
  Without it the copy is chosen once, at start, so when the RAM cannot hold every expert the GPU does not, an expert outside the copy is read from the files (the SSD) every time the conversation routes to it.
  With it, before a decode window and while no adaptive round is in flight, experts the windows keep routing to (the adaptive tier's decayed count, 2 or more) take the RAM places of experts of their own layer that are not routed to (below 0.5): up to 4 pairs per layer, the largest difference first.
  The incoming expert was usually read for the window before; when its blob still sits in that read's stage buffer it is copied from there and no file is read. `n` caps only the pairs that need a file read. The log counts the experts moved and how many came out of a stage buffer.
  The RAM stays the same size; the routing, the weights and the VRAM tier's rule are untouched. It needs the adaptive tier (`--adapt-every`), and does nothing with `STRATA_EXCHANGE_ROTATE=1`, a layer split, a helper GPU or `--batch`.
  Upstream measured decode +16.5% on an RTX 3060 12 GB with 32 GB of RAM (IQ2_XS). On the B70, IQ3_XXS with a 6 GiB RAM budget, serve at 32K was equal: the file reads come from the OS file cache there.
- Otherwise they are read through the OS file cache (`--mmap-experts`), which re-reads them from the SSD as it runs: slower.
  Reading from the GGUF files, the prefill copies the experts with the SSD profile (many threads, a deep ring). When
  90% or more of the experts' pages are already in the page cache at the first prefill (Linux: a sample checked with
  `mincore`), it takes the RAM profile (a few threads) instead and says so in its log (upstream 83a7dbc0, #1353).
  `XESTRATA_STAGER_SSD=1` / `0` forces either.
  On Linux, reading the GGUF files in place, the decode reads as follows (upstream 149).
  An expert whose pages are all in the page cache is computed from the files' mapping, not copied (`XESTRATA_COLD_OVERLAP=0` copies it).
  The others are read with pread without read-ahead; the next layer's router predicts the experts of the next two layers, and I/O threads read them ahead (`XESTRATA_IO_PREFETCH=0` turns this off, `XESTRATA_LOOKAHEAD=0` the prediction alone).
  The read-ahead runs only while the page cache has room for 55% of the experts and what it reads is used (`XESTRATA_IO_PF_GATE=0` reads ahead always).
  An experts.bin pack has its pages read ahead by the I/O threads the same way while the page cache has room for 55% of its experts.
  Reading from the files, the adaptive tier's period is not shortened (`--adapt-every`), and the check's guesses are cut at the `--spec-min-p` floor (`--spec-window auto` sizes them by the measured times).
  On the RX 9060 XT with 32 GiB of RAM (IQ3_XXS) the decode went from 11.6 to 26.0 tokens/s at 1K and from 11.2 to 16.8 at 32K; on the B70 from 46.9 to 56.5 at 1K, and on the RTX 4070 from 13.3 to 34.8 at 1K.

`--low-ram on|off|resident|mmap` overrides the choice, and `./setup.sh --check` shows what each size would use.

The engine locks the experts it keeps in RAM (`mlock`) so the OS does not swap them out:
of an unlocked 40 GiB copy, 3–4.6 GiB went to swap and decoding ran at half speed.
That needs a memlock limit (`ulimit -l`) as large as the copy; setup warns when the limit is small.
Raise it with `memlock` in `/etc/security/limits.conf` or `DefaultLimitMEMLOCK` in systemd's `user.conf`, then log in again.

With a native pack whose experts the GPU mostly holds, `XESTRATA_ARENA_MMAP=1` in the engine's environment (opt-in, upstream #640) maps the experts read-only from the pack's `experts.bin`, not locked.
The first start writes the arena to `experts.bin` (only when the drive has its size and 2 GiB more free); later starts map it.
The pages of the experts in VRAM are handed back to the OS (`XESTRATA_ARENA_RELEASE=0` keeps them) and the rest are read ahead.
The GPU does not read the mapping directly: run with `--pcie-frac 0`.
With the Coder on the B70 (a 23.4 GiB arena, all in VRAM) it handed 23.38 GiB back, gave the same output, decoded as fast (28.3 tok/s) and ran prefill a little slower (54 to 48 tok/s).

The loads at start (the weights, the native dense matrices, the GPU cache's fill from the profile, the experts kept in RAM, the MTP draft files) ask the kernel to read ahead before they read (madvise / posix_fadvise WILLNEED in 128 KiB steps, upstream cb33c0ba).
The drive then sees a deep queue instead of one page fault at a time.
Upstream measured it on a Gen3 NVMe (RTX 5090, 32 GB, Q2_0 kept in RAM at 262K): ready in 70 s instead of ~920 s.
`XESTRATA_READ_AHEAD=0` turns it off; `XESTRATA_FILL_AHEAD=N` sets how many pairs the cache fill asks for ahead (default 256).

The processor's own graphics share the RAM, so they never take the low-RAM mode.
setup's estimate of how much of a model the GPU holds is upstream's (its VRAM less ~5 GB).

## What you need

The GPU, CPU, RAM and disk are in the [README](../README.md#what-you-need).
This adds only what the README leaves out.

- **Packages**: with the deb and rpm packages ([README](../README.md#install)), apt or dnf installs what is needed.
  From the source, an Intel GPU needs its runtime (Level Zero), an AMD GPU ROCm (HIP, rocBLAS, hipBLASLt and others),
  an NVIDIA GPU its driver and the CUDA toolkit, and an Intel GPU in the default contrib-llvm build oneMKL.
  setup can build the SYCL compiler (intel/llvm).
  setup installs no system packages, so install them first.
  What Ubuntu 26.04 and Fedora 44 need is in [BUILD.md](BUILD.md#packages), and `./setup.sh --packages` prints the one-line command for this PC.
- **The GPU device**: your user must be able to open `/dev/dri/renderD*` (and `/dev/kfd` for an AMD GPU; the `render` group).
- **Disk**: about 60–110 GB for the model and about 6 GB for the MTP layer (1 GB more with images).
  **The original model's Q2_0 on an AVX-512 CPU** also writes a one-time copy of its experts (about 40 GB) for the fast CPU kernel.
  An NVMe SSD is strongly recommended.
- **Time**: the first run takes a while to download and to build the engine.
  With `--intel-llvm-build`, building intel/llvm takes another 13–20 minutes.

With the packages, the first start writes these, and nothing in the program's folders (`/usr`):

- `~/.local/share/xestrata`: the model files (`models/`, `packs/`, `mtp/`, 70–120 GB)
- `~/.config/xestrata`: the settings (`settings.json`) and the model configs (`xestrata-<model>.json`)
- `~/.local/state/xestrata`: the logs

`xestrata --remove-data` lists them with their sizes and deletes them when you type yes.
Removing the packages leaves them.

Installed from the source, the first start installs:

- in this XeStrata folder: `.venv/` (the Python environment), `engine/` (the built engine), `third_party/main/llama.cpp`,
  and with `--intel-llvm-build` `.tools/intel-llvm/`
- in **`XeStrata-data` next to this folder**: the model files (`models/`, `packs/`, `mtp/`, 70–120 GB)
- in `~/.config/xestrata` and `~/.local/state/xestrata`: as with the packages, the settings, the model configs and the logs. Every copy of XeStrata of the same user uses these settings.
  Model configs an earlier setup wrote into the XeStrata folder are moved to `~/.config/xestrata` by the next setup (when a config of the same name is already there, that one is used and setup says it did not move it).

The model files go next to the folder so that a new copy of XeStrata, unpacked elsewhere, finds them and sets itself up the same way.
The place is remembered per user (`~/.config/xestrata/settings.json`); `--data-dir` chooses another.
To move the files later, move the folder yourself and run `./setup.sh --setup --data-dir <new place>` once:
setup finds the files there and writes the configs again.
Given a new place with `--data-dir` while model files are elsewhere, setup leaves them where they are and uses them there, and the new place takes the new downloads (it asks whether to move them when it can ask).
`--move-data` moves them into the new place; `--set-default-data-dir` remembers the new place without moving anything.

---

## Install and start

### Run `./setup.sh`

With the packages, run `xestrata` instead of `./setup.sh`: the same options, and no engine is built.

**The first time** it asks a few questions and does the rest:

1. **Which model?** Qwen3.8-Flash-Next (the original), Swift 1.5, the Coder, or Unsloth's (UD-Q2_0 to UD-IQ4_XS, the experimental UD-Q4_K_XL and others).
1. **Which size?** It recommends one for your RAM.
1. **How much context?** 8K to 256K tokens; it recommends one for your GPU.
   384K and 512K are offered too, as experiments: past the trained length (262,144) setup adds RoPE scaling.
1. **KV precision** (above 8K): see [above](#where-the-kv-cache-lives-and-its-precision).
1. **Images?** (see [Images](#images)).
1. **The translation model**: auto by default ([below](#translation-optional)).
1. **The experimental speed projection**: off by default ([below](#experimental-speed-projection-experimental-off-by-default)).

Then it chooses the SYCL compiler and builds the engine ([BUILD.md](BUILD.md#the-sycl-compiler)),
downloads and prepares the model, runs it once to compile its GPU code, and **starts the model**.
The model is 60–110 GB, so the first start takes a while; an interrupted download continues where it stopped.

Your browser opens `http://127.0.0.1:8095`.
The app has three tabs:

- **Chat**: streaming answers; the model's thinking (folded away once it answers); code with a copy button;
  pictures when images are on; sampling and thinking-level settings.
  Chats stay in your browser.
- **Monitor**: what the model is doing (prefill, with progress, or decode, at how many tokens/s);
  GPU load, VRAM, temperature, power and PCIe traffic; CPU, RAM and disk; the context in use; the last requests.
  With the model split across several GPUs, or the images on a GPU of their own, tabs above the GPU cards switch
  between all the GPUs together and each one, and their marks say what each GPU does (text, images, helper, its share
  of the layers, prefill / decode).
  With the images on a GPU of their own, the text's state and the image job's are on lines of their own.
  Its conversation cache card shows how many prefill tokens the cache gave back (in the last request and since the
  start) and, with `--conversation-cache-mib`, the parked conversations and their memory, and how many were parked,
  restored and evicted (upstream #596; the same is under `"conversation_cache"` in `/metrics`).
- **About**: the model and engine settings, and the addresses to connect other apps.

`http://127.0.0.1:8095/?q=your question` opens it with a new chat already asking.
The API for your apps is at `http://127.0.0.1:8095/v1`.

**Every time after that**, `./setup.sh` just starts the model.
Loading 23–50 GB into RAM takes a while; nothing is downloaded again.
Closing the window stops the model.
With more than one model installed, it asks which one to start; `run-<model>.sh` starts a model directly.
Running setup again for the same model (another context, say) rewrites the keys setup writes (`exe`, `args`, `port`, `gpu_pci`, `vision`, ...) and keeps the ones you added (`sampling`, `mcp_servers`, `cors_origins`, ...) and a `host` and `api_key` this run does not give.
The earlier file is kept as `xestrata-<model>.json.bak`.
For Qwen3.8-Flash-Next the size question marks the installed size `(installed)` and makes it the default, and `--yes` takes it: a repeat run does not move the model to the recommended size and download it.
Engine options you added to `"args"` by hand are not carried over: setup names them, and you add them again (upstream #629).

```text
./setup.sh --setup                          install another model, or change context / images
./setup.sh --model IQ2_XS --context 32768 --vision yes --yes     no questions
./setup.sh --gguf-dir /data/models/IQ2_XS   use GGUF files you already have
./setup.sh --data-dir /data/XeStrata-data   keep the model files somewhere else
./setup.sh --port 8081                      another port
./setup.sh --gpu 1                          another GPU (numbered as --check lists them; the default has the most VRAM)
./setup.sh --gpus 1,2                       spread the layers over two or more GPUs (remembered; docs/MULTIGPU.md)
./setup.sh --vram-reserve-mib 2048          leave 2 GB of VRAM free for other programs (remembered)
./setup.sh --setup --ram-reserve-gib 16     leave 16 GiB of RAM free for other programs (remembered)
./setup.sh --ple-io ram                     keep the n-gram table in RAM (remembered; direct reads it from the SSD, auto lets setup choose)
./setup.sh --no-browser                     do not open the chat page when the model is ready (remembered; --browser undoes it)
./setup.sh --calibrate                      tune the engine for this PC (about 15-30 minutes), then start
./setup.sh --check                          only check this PC
```

**Updating**: run `./update.sh`.
For a git checkout it runs `git pull`, then updates the engine, the Python packages and the model settings (it does not start the model).
It compiles the engine again when its source changed; if that compile fails, it says so and keeps the engine you had.

**New releases**: the server asks GitHub for MistVVK/XeStrata's latest release at the start and every six hours after; when it is newer than this copy, it prints a line in its window and shows it on the chat page's About tab.
About's "Check now" and `./setup.sh --check-update` (`xestrata --check-update` with a package) check at once, at any time.
`XESTRATA_UPDATE_CHECK=0` in the environment turns the automatic checks off ("Check now" and `--check-update` still work).

### Leaving VRAM for other programs

XeStrata fills the GPU's free VRAM with experts (the expert cache) and leaves `--vram-reserve-mib` MiB free: by default a 32nd of the device's memory, between 700 MiB and 768 MiB (700 on a 16 GiB card, 768 past 24 GiB). The image engine and SeedVR2 size their own room by the same rule (`core::vram_room`).
For a game, a 3D program or another model beside it, leave more (upstream #493).

- `./setup.sh --vram-reserve-mib 2048` writes it into the model's `xestrata-<model>.json` and starts it.
- At setup (`--setup --vram-reserve-mib 2048`) it goes into the new config.
- By hand: add `"--vram-reserve-mib", "2048"` to the config's `"args"` list and restart.

The expert cache is then that much smaller, so answers can be a little slower.

The amount can also change while it runs (upstream #533).
A config with image generation or a GPU encoder holds the expert cache in pieces of 512 MiB (`"vram_segment_mib"` changes it; `--vram-elastic`). Other configs do the same with `"vram_elastic": true`, and `"vram_elastic": false` always keeps one allocation. Held in pieces, the cache read the prompt and wrote the answer as fast on the B70, the RTX 4070 and the RX 9060 XT.
Between requests, `POST /v1/vram` with `{"reserve_mib": 6000}` gives the cache's last pieces back to the GPU until that much is free, and `{"reserve_mib": null}` goes back to the amount the start left.
A running request finishes first.
The experts of the pieces given back are computed on the CPU: the same answers, slower.
`GET /v1/status` shows the current size under `"vram"`.
A reserve that cannot be met (the cache at its smallest size, or no VRAM from the driver for the next segment) still answers `200`: compare the answer's `vram_free_mib` with your `reserve_mib` (upstream #1034).
While a request runs or waits, it waits up to 300 s, then answers `409`.
Where the GPU's runtime has no virtual memory it is off and the cache stays as it started (Intel's Level Zero and NVIDIA's CUDA have it; AMD's HIP has it with XeStrata's intel/llvm patch `hip-virtual-mem`).
At start the engine gives its last piece back and checks that the GPU's free memory grows. A runtime where it does not (ROCm 7.1 and 7.2's HIP keep an unmapped piece's VRAM in the process; ROCm 10.1 gives it back) refuses to free VRAM, and `GET /v1/status`'s `"vram"` and INFO's `vram_elastic` say 0. The server then never picks `shrink` for an image job, and starts a GPU encoder before the model.

### Leaving RAM for other programs

With `--ram-reserve-gib N` at setup, XeStrata runs leaving N GiB of RAM free for other programs.
Without it, nothing changes.

- Setup chooses how the model runs (all its experts in RAM, or the low-RAM mode), how many of its experts stay in RAM, the recommended size and context, and the image models' sizes as if the RAM were N GiB smaller.
  When the experts no longer fit the RAM, the rest are read from the SSD while it answers, which is slower.
- The text engine (`--ram-reserve-mib`) and the image engine keep at least N GiB beside the experts and layers they copy into RAM and the conversations they keep there (otherwise 8 GiB, and 2.5 GiB for the conversations).
- When leaving N would not leave the least RAM the model runs in, setup says so and installs it with that least (less than N is then left free).
- It is kept in the model's config. To change it, set up again with `--setup --ram-reserve-gib` (set up again without it, nothing is left free any more).
- XeStrata keeps it by itself; it is not a limit the OS enforces. The OS's cache of the model files read from the SSD does not count: the OS gives it back when another program needs the RAM.

### Model files downloaded by hand, or from a mirror

setup's step 5 prints the folder it expects them in (`XeStrata-data/models/<SIZE>/`; upstream #495).
Put them there with their original names, or point setup at them with `--gguf-dir`.
To see what a file really holds before using or downloading it, `./setup.sh --inspect <file>` (upstream #911) reads only its headers (a few MB, also over the network).
It shows every weight group's real bits per weight and storage types, and whether the file is one setup installs (under any name), stored like one of them, or not one XeStrata runs (another architecture, or experts in a format no kernel reads).
It takes a file, a folder, a URL, `ms:owner/repo` (ModelScope) or `hf:owner/repo`, optionally followed by a variant name.
A name says little: Unsloth's `UD-IQ3_XXS` (Dynamic v3.1) stores its routed experts at 3.33 bits per weight, `UD-Q4_K_XL` at 5.10, the original `Q2_0` at 2.25 and `IQ3_S` at 3.33.
To download from a Hugging Face mirror, set `HF_ENDPOINT` (e.g. `HF_ENDPOINT=https://hf-mirror.com ./setup.sh`):
the pinned revisions and the checks are the same, and the MTP tensors come from the same host.
Every model file setup downloads has a fixed size and SHA-256 (upstream e193161de).
When the server's size is not the pinned one, setup stops before the download; a downloaded file that does not match its SHA-256 is deleted and setup stops.
When the pinned revision is gone from the repository, setup stops (for testing, `XESTRATA_ALLOW_UNPINNED=1` takes the current files without the check).

**Fetching the files yourself (the `hf` command)**: setup downloads over 16 connections (`XESTRATA_DL_CONNECTIONS` 1 to 32), resumes a stopped download and checks every file's SHA-256 against the pinned value.
Measured by upstream from two computers (a home PC and a server in China), anonymous downloads from Hugging Face ran at 16 to 19 MB/s whatever the connection count (one stream 7 MB/s, 16 connections 17, 32 connections 19).
Hugging Face's own `hf` command with its Xet backend gave 16 to 24 MB/s anonymously in the same test, so it is not faster by itself.
Hugging Face limits anonymous requests per IP address and says a login (a token of your own account) raises the limits, so if the download is slow you can fetch the file with the `hf` command, optionally after `hf auth login`, and give the folder to setup:

```bash
uvx hf auth login            # optional: your own Hugging Face account; XeStrata does not read or send this token
uvx hf download agentionai/Qwen3.8-Flash-Next-Gyro-GGUF Qwen3.8-Flash-Next-Gyro-S-TQ1_0.gguf \
    --revision 1ada08422e404aca51ab9df665f089f34c419c6b --local-dir /data/gyro-s
./setup.sh --gguf-dir /data/gyro-s       # Windows: xestrata-cli --gguf-dir D:\gyro-s
```

Use the file names and the revision setup shows for your model (the revision above is the one this version pins for Gyro-S: another version of XeStrata may pin another).
Keep the file's original name. Setup finds the files in `--gguf-dir`, checks their SHA-256 and does not download them again; a file that does not match the published one is reported and, for files in your own folder, never deleted.
Without `uvx`, `pip install -U huggingface_hub` gives the same `hf` command.

**From ModelScope** (mainland China, upstream #908): every repository setup downloads from (the GSQ-RCO models, Swift 1.5, the Coder, Unsloth's files, and the original checkpoint the MTP draft layer is taken from) is on [ModelScope](https://www.modelscope.cn) under the same name, with the same files.
Setup never switches to it by itself (a download that fails says how).
Choose it with `--source modelscope` (`XESTRATA_SOURCE=modelscope`).
ModelScope serves a repository's current files (no pinned revision), so each file is checked against its pinned SHA-256 (a file without one against the SHA-256 ModelScope publishes for it), and the MTP tensors against the pinned checkpoint's own hashes.
The published SHA-256 is ModelScope's own statement: a changed upstream repository would carry its own hash.
Setup says so when you choose it.

When a file in `--gguf-dir` is not under its published name (renamed, or taken from somewhere else), setup checks its size and SHA-256 against the published file's.
It stops when they differ, so another model with a size's name in its own (a fine-tune's `IQ3_XXS`) is not installed as that size.
Such a model goes in with `--gguf`, below.

### Installing from your own GGUF files (`--gguf`)

A model setup does not list can be installed from GGUF files you have; nothing is downloaded.

```sh
xestrata --setup --gguf ~/models/My-Model-Q4_K_M-00001-of-00003.gguf --name my-model
```

- **What goes in**: Qwen3.8-Flash-Next (`qwen4exp`), and models of the other models' architectures (`qwen35moe`: Ornith-1.5, Qwen3.6-35B-A3B; `qwen35`: Qwen3.5-9B, 2B and 0.8B; `qwen3moe`: llm-jp-4.1; `glm5-next`: GLM-5.3-Flash).
- **What to give**: a `.gguf` file (any one of a split model's, all of them in the same folder), or a folder with one model only. Without `--name`, the file's name is used.
- **What is checked**: setup reads the headers and asks the engine whether it runs them (`xestrata-engine --check-model <file>`, which opens no GPU).
  When the headers cannot be read (a file cut short too), the architecture is another, or the experts or the PLE table are stored in a type the engine's kernels do not take, setup stops with the reason for each layer and makes nothing.
- **What is made**: the config (`xestrata-custom-<name>.json`) and the pack, under the name `custom-<name>`: apart from setup's sizes, so it never writes over an official one.
  A start, `--bench` and `--calibrate` take `--model custom-<name>`.
  `--setup --model custom-<name>` sets it up again from the same files; another model's files under the same name are refused.
- **Qwen3.8-Flash-Next**: the MTP draft layer and the picture reader are the other sizes' (downloaded when they are not here; a `mmproj-Qwen3.8-Flash-Next-BF16.gguf` beside the model is used).
  The parts stored as Q8_0 and the like become the BF16 the engine reads (`--compat-bf16`).
- Only the types are checked against the engine: the answers are as good as the file.

### Tuning for your PC (`--calibrate`)

These engine settings depend on the PC more than on the model; the calibration measures the three other than the
window length:

- the share of the experts missing from VRAM that are copied to the GPU instead of computed by the CPU (`--pcie-frac`).
  Without it the engine splits them by an expert's time on each side, measured at start
  ([XE](XE.md#splitting-the-missed-experts-between-the-cpu-and-pcie)); a fixed share is kept only when it is more than 3% faster;
- how sure the draft layer must be to add another guess to a check, which is not measured:
  by default (`--spec-window auto`) the engine adds a guess while the output it is expected to add, over the time of a
  check of that size measured since the start, raises the output per ms (the expectation is the draft layer's
  probability, corrected by the share actually accepted); a request that passes `spec_min_p` runs the `--spec-min-p`
  floor instead, so the calibration passes none; when the experts the GPU does not hold are read from the files
  (`--mmap-experts`) and `--spec-window` is not given, the `--spec-min-p` floor cuts them (a check's time there swings
  with the SSD's reads, and the wider ones were priced too high);
- how many CPU threads compute experts (`--pool-workers`): on CPUs with efficiency cores, fewer can be faster;
  it also tries one fewer than the P-cores, and one fewer than one socket's cores on a 2-socket PC;
  on a CPU with more efficiency cores than performance cores, setup writes the performance cores (but the host's
  one) and half the efficiency cores into the config before any tuning (upstream #642);
- how busily the VRAM expert cache follows the conversation (`--adapt-every`, `--adapt-swaps`, `--adapt-decay`, upstream #907):
  a PC whose CPU reads the missed experts slowly (DDR3, or DDR4 in few channels) gains from swapping more.
  Upstream measured 160 swaps every window 7.9% faster than the default on a Xeon E5-2673 v3 with DDR3 and an RTX 4060 Ti on PCIe 3.0 x8.
  This step measures 512-token answers.

The defaults were measured by upstream on a Ryzen 5 7600 with an RTX 5070.
Some have been measured again on the B70 ([XE.md](XE.md#upstream-settings-not-remeasured-on-xe)).
The engine and setup fit them from what they measure, so this is usually not needed. They are measured only when you
run `./setup.sh --calibrate` (it asks first whether to measure anyway; `--yes` skips the question).
It measures the decode speed with each setting and keeps one only when it is more than 3% faster.
The result is remembered per PC and model in the settings file, so updates keep it.
Measuring the worker count needs a fresh engine, so the model is loaded more than once, and the PC can stop responding for a minute or two each time (upstream #1197).
When it finishes it starts the model (not with `--no-start`): do not start it a second time.

`--pool-tasks N` sets how many row tasks each batched CPU expert phase (gate/up, then down) is cut into (upstream #949).
It is neither the number of threads nor the tasks per expert.
The default `0` keeps three per participating thread (the host included when it works too).
Values `1` to `4096` are allowed, capped by the phase's row count.
More tasks can even out the cores, but cost more scheduling:
compare against `0` with the same worker count and workload.
It changes neither the kernels, the phase barriers nor the PCIe placement; `--calibrate` does not tune it yet.
For the server, add `"--pool-tasks", "192"` (for example) to the `args` of its configuration and restart it.

The CPU pool puts one worker on each physical core (never on an SMT sibling) and keeps the first core for the host (the thread that waits for the GPU).
The start-up log says where they sit in one line (`CPU workers on logical processors 2,4,...,30 (15 workers, one per physical core; host on 0)`).
Where the OS does not say which logical processors share a core (some containers), the line says so.
The workers are named `xs-pool-N` and the host `xs-host` (the other helper threads too, `xs-adapt` and so on), so Linux's `ps -L -o tid,psr,comm` and the Windows tools that read thread names tell them apart.
`--pool-cores LIST` (experimental, off by default, upstream 7a5599581) pins the workers to exactly these logical processors, in this order.
`XESTRATA_POOL_CORES=LIST` or `"pool_cores": "2,4,6,8"` in the configuration sets it too.
The list is numbers, ranges and `first-last:step` (`0,2,4,6`, `2-14:2`), numbered as Task Manager and `lscpu` show them.
The worker count is the list's length; `--pool-workers N` takes its first N entries.
Nothing is refused; the engine warns when two entries are the two threads of one core, when an entry is the host's logical processor, and when one is a processor this process may not use (left out). With nothing usable left the pool keeps its default placement, and a list that cannot be read is said and ignored.
The host keeps the first core, so leave it out of the list. It moves threads only and does not change the answers.
It applies to the Qwen3.8-Flash-Next engines; the other MoE models (Ornith, llm-jp, Qwen3.6, GLM-5.3-Flash) keep the default placement.

`--aux-cpus` decides which CPUs run the threads that are neither CPU pool workers nor the host thread (Linux, `off` by
default, upstream 91f1e85d): the expert cache's refills, the prefill copies, the file reads, the GPU runtime's threads
and the like.
`auto` takes the SMT siblings of the host's core, then the CPUs of physical cores with no worker. It never takes a CPU
that shares a core with a worker, and does nothing when none is spare.
A list such as `24,26` is taken as given, minus the host's and the workers' CPUs.
It moves threads only and does not change the answers.
`XESTRATA_AUX_CPUS` sets it too, and the API's `"strata_tune": {"aux_cpus": 1}` (`0` puts them back) switches it per
request. `XESTRATA_AUX_STAGER=0` leaves the prefill copy threads where they were started.
On an RX 9060 XT with a Ryzen 7 5700X (serve, IQ3_XXS, five interleaved rounds) `auto` sped up the 1K prefill and the
wait before a 32K follow-up, but the 32K prefill read 3.2% slower, so the default stays `off`.

### Chat in the terminal

```sh
.venv/bin/python chat.py
```

### The window (the launcher)

The window (Windows' `xestrata.exe`, Linux's `xestrata gui` and `./setup.sh gui`) is an entry to setup: setup.py does
all the work.
Its fields (the KV cache's precision, web search, the start's settings and the rest) are drawn from the options setup's
`--describe` returns (setup.py's `OPTIONS`), so an option added to setup is in the window too.
A field's text is in the window's language where its catalogs have it, else in setup's English.

- **Language**: Windows' display language; on Linux the first of `LC_ALL`, `LC_MESSAGES` and `LANG` that is set.
  `--lang ja` chooses one.
- **Linux's screen**: it runs on X11, through XWayland under Wayland. Its scale is `Xft.dpi` or
  `AVALONIA_GLOBAL_SCALE_FACTOR` (for example `AVALONIA_GLOBAL_SCALE_FACTOR=1.5 xestrata gui`).
  Japanese and the other languages typed through an input method take IBus or Fcitx. Characters the theme's font lacks
  are drawn with Noto Sans CJK (which the package recommends).
  Where there is no screen (ssh alone, a server) it does not open, and names the terminal's `xestrata --setup` and the
  browser instead.
- **Drawing**: a window that closes without a trace draws without the GPU with `--software-render`.
- **Log**: the errors that stop the window go to `launcher.log` (Windows: `%LOCALAPPDATA%\xestrata\state`, Linux:
  `~/.local/state/xestrata`).
- **A model already running**: when the model Start starts is already served on its port (the systemd service, for
  example), the window connects to it. It does not stop a server it did not start, and says how to stop it.

---

## Sharing the GPU with other programs

By default the model stays loaded until you close XeStrata.
On a PC that also games, renders or runs another model server, three server options give the VRAM back.
All are off by default, and each is also a key in `xestrata-<model>.json`.

| Option | Config key | What it does |
| --- | --- | --- |
| `--idle-unload 600` | `"idle_unload_s": 600` | unload the model after 600 s without requests; the next request loads it again |
| `--min-free-vram-mib 11000` | `"min_free_vram_mib": 11000` | load an unloaded model only when that much VRAM is free on its GPU (read through Level Zero's sysman on an Intel GPU, NVML on an NVIDIA one; it waits up to 15 s for memory being given back), else answer **503** "the GPU is in use by another program" |
| `--before-load "cmd"` | `"before_load": "cmd"` or `["cmd", "arg"]` | a command run before the model is loaded again, e.g. one that unloads another server's model |
| | `"before_load_timeout_s": 120` | how long that command may run (default 120 s). When it exits non-zero or times out (a peer that is not running makes `curl` exit 7), the free VRAM decides: with `min_free_vram_mib` set and not enough free, the model is **not** loaded and the request gets **503** with `Retry-After`; with room (or no limit set) it loads, with a warning in the log. `"before_load_strict": true` always refuses on a failure; `"before_load_ignore_failure": true` always loads |

- **Unloading and loading**: `POST /unload` unloads it now (`409` while a request is running; `POST /unload?wait=1` waits for the running turn first, up to 300 s, or `?wait=<seconds>`, so a second server can really free the card) and `POST /load` loads it ahead of a request.
  Both take `Content-Type: application/json` (e.g. `curl -X POST -H "Content-Type: application/json" localhost:8095/unload`).
- **State**: `/health` says `"loaded"`, `/v1/models` lists an unloaded model as `unloaded` (like llama.cpp's router),
  `/props` sets `is_sleeping`, and the Monitor shows the state.
- **What unloading does**: it ends the engine process, and the image encoder when images are on (that is started again first, as at a start).
  So their VRAM and RAM go straight back.
  The model files stay in the OS file cache, so loading again takes seconds while that RAM is not needed elsewhere.

`serve/server.py --engine xestrata-engine --config xestrata-<model>.json --lazy` (or `"lazy_load": true` in that config) starts the HTTP API without starting the engine.
The first generation request loads it the same way, including `before_load` and `min_free_vram_mib`.
This is text-only: a vision configuration with lazy startup is refused.
`POST /v1/load` and `/v1/unload` are JSON forms of the same controls for integrations.
They accept `{}` or `{"model":"<configured model>"}` and return the model's status.
They need the API key (when one is set), `application/json`, and no foreign browser Origin.
They answer **409** while a request is running or queued and **404** for an unknown model.
`/api/health` is the same as `/health`; `/v1/status` reports `loaded` and `auto_load`.

### Keeping what the expert cache learned across restarts

A start fills the GPU's expert cache from the shipped profile, and the adaptive tier (`--adapt-every`) then moves in the experts your requests use (upstream #477).
`--adapt-mib-per-token N` caps a round's copies at N MiB for each window since the last round (Qwen3.8-Flash-Next: off by default; the other models' engine bounds them at 48 MiB a token).
With `"expert_profile_save": "expert-profile-learned.bin"` in `xestrata-<model>.json` the engine saves that as a profile:
the experts in VRAM first, then the routing it counted since the start, then the shipped order.
It saves on a clean exit and every 10 minutes between requests (`"expert_profile_save_every": 5` for another interval, `0` for exit only).
It writes a temporary file and renames it, so a crash never leaves half a file.
The next start begins from it instead of the config's `--expert-profile` when it is a profile of the same model.
A relative path is in the XeStrata folder; one file per model.
A profile per project works the same way: point the key at another file.
The file is a fingerprint of what you used the model for: keep it on your PC.
Without the key it is saved as `expert-profile-<model name>.bin` (in the engine's working folder) when the config has an `--expert-profile`; `"expert_profile_save": false` counts and writes nothing.
A model unloaded for an image job comes back from the learned profile too, and the experts it lets go of in RAM before the unload (`ARENA DROP`) are the last ones of this profile.
A setup run again keeps the key, and `--expert-profile` (a file that still exists) and `--expert-profile-save` added to `"args"` by hand (upstream #775).

An expert the adaptive tier swaps in is used on the GPU from the second window, so no window waits for its copy
(upstream #764; the answers are the same run to run with the same settings).
Decode went from 59.1 to 64.2 tokens/s on an RTX 4070 and from 76.5 to 76.9 on the B70 (IQ2_XS).
`XESTRATA_ADAPT_LAG=1` in the engine's environment makes the next window wait for the copy and use it.

### When the draft head does not fit

The MTP draft layer's head covers a token subset (`./setup.sh --draft-vocab cjk|cyrillic|fr|en`; upstream #474).
`fr` is the English/code subset plus the tokens of French text, so French answers draft more (a head of about 151 MiB; upstream #597).
The default includes Chinese, Japanese and Korean and takes up to about 348 MiB of VRAM.
When the start stops with "the draft head does not fit", the engine says how much the head needs, how much VRAM is free and which smaller subset fits,
and the server's start error repeats it.
setup suggests `--draft-vocab en` on cards under 14 GB; it is only a suggestion, nothing changes unless you pass it.

The draft head's product is timed both ways on the GPU at start, and the faster one is taken:
the head as it is, or a copy of its rows packed to 2 bits (a minimum and a step for each 128 values) that scores them all roughly,
the best 16 of them computed again from the head itself.
Either way only the drafts change, never the answer (the model's own head checks the drafts).
The 2-bit copy takes about 80 MiB (a token subset) to 180 MiB (every token) of VRAM.
On the B70 a draft step's head for Qwen3.8-Flash-Next went from 556 to 292 microseconds, a round's drafting about 0.9 ms shorter.
The MTP layers of the other models (GLM-5.3-Flash, Ornith-1.5, Qwen3.6-35B-A3B) do the same.
`XESTRATA_DRAFT_INT2=0` keeps the head as it is, `1` takes the 2-bit copy without timing.

The server also runs without `--mtp` (upstream 3216d271).
The drafts then come from the suffix and prompt lookup only (one token a round when neither fires), and every token is still verified against the model.
The draft layer's VRAM (about 0.7-1 GiB with its head) goes to the expert cache: 1,000 to 1,678 slots in upstream's comparison on an 8 GB card.
Parked and saved conversations carry the draft layer's K/V, so without `--mtp` the conversation cache's parking, `--conversation-save` and the session files are off.

A native Qwen3.8-Flash-Next pack (one whose experts are read from the GGUF files) takes `--spec` 2 at the smallest: the engine refuses to start with `--spec 0` or `--spec 1` (`it needs --native SHARD1, --spec T (T >= 2) and --prefill CHUNK`).

---

## Using it

The server listens on `http://127.0.0.1:8095` (change it with `--port` in setup, or in the run script).
To switch between it and other model servers behind one address, see [LLAMA_SWAP.md](LLAMA_SWAP.md).

| API | Endpoint |
| --- | --- |
| OpenAI Chat Completions (stream and non-stream, tools) | `POST /v1/chat/completions` |
| Anthropic Messages (stream and non-stream, tools) | `POST /v1/messages` |
| OpenAI Responses (stream and non-stream, tools; stateless, [below](#the-responses-api-and-codex-cli)) | `POST /v1/responses` |
| Model list / health | `GET /v1/models`, `GET /models`, `GET /v1/models/{id}`, `GET /health` |
| Model properties | `GET /props` (also accepts `?model=<loaded-model-id>`) |
| What the model is doing right now | `GET /status`, `GET /slots` (a single slot, busy or idle) |
| Everything the Monitor tab shows (engine, live state, last requests, hardware) | `GET /metrics` |
| The same for Prometheus, with vLLM's metric names (`Accept: text/plain` or `?format=prometheus`) | `GET /metrics` |
| The totals and what is running, for Prometheus with llama-server's metric names | `GET /metrics/prometheus` |
| The MCP servers, their state and tools ([below](#tools-from-mcp-servers)) | `GET /mcp` |
| Translation: DeepL's API ([below](#translation-optional)) | `POST /v2/translate`, `GET /v2/languages` |
| Translation: LibreTranslate's API | `POST /translate`, `POST /detect`, `GET /languages`, `GET /frontend/settings` |

`/models` and `/v1/models` list the model with its context limit and input modalities.
It is listed before it is loaded and while it starts too, with `status` `loading` (`unloaded` when it was unloaded for idleness).
OpenAI's `created` (the server's start time) and `owned_by` (`xestrata`) are included.
`GET /v1/models/{id}` returns one model; an unknown id is a 404 (`model_not_found`).
`/props` gives the original chat template, the context limit, the configured generation defaults (shared settings take precedence),
the model path and the engine version where available.
Context means the full engine context, not the resident KV window.
`n_predict: -1` means no fixed output cap.
Unconfigured sampling fields are left out.
`autoload` has no effect; an unknown `model` returns 404.
These metadata endpoints and `/slots` require the API key when one is configured.
They do not load, unload or restart models.

```bash
curl http://127.0.0.1:8095/v1/chat/completions -H "Content-Type: application/json" -d '{
  "model": "xestrata", "messages": [{"role": "user", "content": "Write a haiku about GPUs."}], "max_tokens": 512 }'
```

```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:8095/v1", api_key="none")
r = client.chat.completions.create(model="xestrata", messages=[{"role": "user", "content": "Hello!"}])
print(r.choices[0].message.content)
```

### Thinking levels

The model thinks before it answers.
The thinking is streamed as `reasoning_content` and `reasoning` (the same text; Anthropic: `thinking` blocks).
Choose how much per request, from none, low, medium and high:
in the chat page (the "Thinking" menu), in `chat.py` (`/think low`), or over the API.

| API | How |
| --- | --- |
| OpenAI | `"reasoning_effort": "none" \| "low" \| "medium" \| "high"` (also `"reasoning": {"effort": ...}`, or `"chat_template_kwargs": {"enable_thinking": false}`) |
| Anthropic | `"output_config": {"effort": "low" \| "medium" \| "high"}`, `"thinking": {"type": "disabled"}`, `"thinking": {"type": "enabled", "budget_tokens": N}` (under 2K = low, under 8K = medium, more = high), or `"thinking": {"type": "adaptive"}` (the model's default level, or `effort`) |

Without a setting the model uses its own default, **high**.
`"reasoning_effort": "low"` (or `none`, `medium`, `high`) in the run config sets the level of every request that names none (upstream #1641).
A `--reasoning-effort` in `args` is moved there by the server, with a note.
`none` answers at once (fastest); `low` keeps the thinking short.
The levels are instructions the model was trained with, not a hard token limit:
on easy questions all three think briefly, on hard ones `high` thinks longest and is most accurate.

- **A hard thinking budget (opt-in).** `"reasoning_budget_tokens": N` (or llama.cpp's `"thinking_budget_tokens": N`, which clients written for llama-server send) in a request (OpenAI or Anthropic) caps the thinking at N tokens.
  When it gets there the server ends it with a short wrap-up line and `</think>`, and the model answers from there.
  The engine continues from what it already holds, so nothing is read again.
  The wrap-up is part of the thinking the client sees and counts as output tokens.
  `"reasoning_budget_tokens": N` in `xestrata-<model>.json` sets it for every request; a request's own value wins, and `0` means no budget.
  Off by default; Anthropic's `"thinking": {"budget_tokens": N}` chooses the level, as above, and is also the budget (a `reasoning_budget_tokens` in the same request wins).
- **max_tokens counts the thinking, so the answer keeps its room.** A client that caps the whole reply (the desktop agent's harness sends `max_completion_tokens: 8192` on every request) and a task the model thinks long about used to be cut off in the middle of the thinking, with no answer.
  A request with thinking on, no budget of its own and `max_tokens` of 1,024 or more gets one: the thinking is wrapped up when only a quarter of the cap (at least 512 tokens) is left, and the model answers in that room (8,192: thinking up to 6,144, answer 2,048).
  Requests without `max_tokens` (the rest of the context) get a budget of three quarters of it.
  `"answer_room": false` in `xestrata-<model>.json` turns it off.
  Qwen recommends up to 262,144 tokens of reasoning and 131,072 of answer: a client should not cap the reply far below what the context allows.
- **Changing the effort without reading the prompt again (opt-in).** The effort's instruction is the first thing in the prompt, so a request that only changes the effort (an agent's "think harder" switch, `none` for a quick tool step) reads the whole conversation again.
  `"effort_position": "end"` in `xestrata-<model>.json` renders every request's prompt start as the default effort's and puts a `low` / `medium` effort in a short system turn right before the answer (no thinking: the empty thinking block, as always).
  The engine keeps its conversation checkpoint in front of that turn (`--tail-role-token`; the server checks that the engine knows it), so the next request reuses the conversation whatever its effort.
  The default (`"start"`) prompt is unchanged.
  The model sees a level that is not the default in another place than it was trained with.
- **A reply that falls into a loop is steered out of it (on by default).** When a reply starts repeating one block of tokens (8 to 2,048 tokens) back to back, the server sees it and breaks it (the loop guard of [Project Maya](https://github.com/mw00/project-maya), transcribed).
  A block under 32 tokens must come 4 times, a longer one 3 times, 96 tokens in all at least; in the thinking 5 and 4 times, 192 tokens. A table's rows, an array or similar code lines are not one block repeated as it is, so they do not count.
  Until a loop is seen the output is the same; the stream holds back at most 64 tokens of a tail that could be a loop's repeats.
  The first time, the repeats not sent yet are dropped and the reply goes on from what was kept with a short repetition penalty (1.15 over 2 x the period, at least 128 tokens) for the next 2 x the period + 128 tokens, then with the request's own sampling.
  A second loop in the answer (or a fourth in all) ends it (`finish_reason` `"stop"`). In the thinking the first two get the penalty and the third closes the thinking, so the model answers (Harmony's thinking is never closed: the penalty only).
  A loop that touches a tool call is never cut: calls that repeat stay whole, and a loop inside one call only gets the penalty.
  The server says so in its window and counts it in `/metrics` under `loop_guard` (in the Prometheus form, `strata:loop_guard_*_total`).
  `"loop_guard": false` in `xestrata-<model>.json` or `XESTRATA_LOOP_GUARD=0` turns it off; a request's `"loop_guard": true` / `false` changes it for that request.
- **Other languages' characters kept out of the answer (optional).** A request's `"script_guard"` set to the answer's language (`ja`, `en`, `ko`, `zh-cn` or `zh-tw`) keeps other languages' characters out of it.
  `ja` bans the Han characters of GB2312 (Simplified Chinese's character set) that JIS X 0213 does not have (这, 们, 说, 时 and the like; not the ones Japanese writes too, such as 没 and 来), all of Hangul, and the letters only Vietnamese writes (ả, ệ, ơ, ư, đ and the like; not the á, à and é pinyin and French share).
  `en` (English and the other languages that write no CJKV) bans every Han character, the kana, Hangul, Bopomofo, the CJK punctuation, symbols and full-width forms, and the letters only Vietnamese writes.
  `ko` bans the kana and the Han characters KS X 1001 (Korean's character set) does not have (这, 国 and the like; not 國 or 學).
  `zh-cn` and `zh-tw` ban the kana, Hangul and the forms only Japanese writes (in JIS X 0208 and in neither GB2312 nor Big5: 気, 駅, 込 and the like), and `zh-cn` the traditional forms (in Big5, not in GB2312: 這, 說, 國), `zh-tw` the simplified ones (in GB2312, not in Big5: 这, 说, 国).
  Every language but `en` also bans the letters only Vietnamese writes.
  The tokens that hold one get logit_bias -100 while the answer is written. The thinking is left alone (where it ends, the server goes on in a pass with the bias).
  A character the conversation (system, user, assistant, tool results) already holds is not banned, so a Chinese, Korean or Vietnamese text pasted into it can be quoted or translated; `{"lang": "ja", "allow_context": false}` (or `en`) bans those too.
  A character spelt in byte tokens slips past the tokens' bias. The engine follows the bytes of a character the reply has begun and keeps every row of a speculative window from taking a token that would complete a banned one there. The server writes a table for each vocabulary and language once (`script-tables` in the state folder) and hands it to the engine, which reads the banned tokens from it too, so no long bias goes with each request.
  In `--pipeline-windows 2`'s windows, and with an older engine that cannot read the table, the server drops such a character before it is sent and has the model write that place again with the first of those tokens banned (up to 8 times an answer; past that, or when part of it was already sent, it is let through and the server's window says so in a line).
  Speculative decoding works with the guard as without it. The same question three times each on the B70 decoded at 97.9 and 97.7 tokens/s on Qwen3.8-Flash-Next and at 141.9 and 144 tokens/s on Ornith-1.5-35B-A3B (Ornith had been at 106.5 tokens/s, the guard turning speculative decoding off).
  A client that cannot add a field to the body sends the header `X-XeStrata-Script-Guard: ja` instead (the body's `script_guard` wins).
  A request that names none gets, in this order, the web app's Chat settings shared with other apps ("Use for other apps too"), `XESTRATA_SCRIPT_GUARD=ja`, `"script_guard": "ja"` in `xestrata-<model>.json`, or `"script_guard"` in the settings every model shares (`settings.json`); the two files can also be changed in the web app's Settings. Without any of them it is off, and a request's `"script_guard": "off"` turns it off for that request.
  Setup asks whether to use it the first time it runs interactively and with `--setup`, and writes the answer to `settings.json` (`--script-guard ja|en|ko|zh-cn|zh-tw|off` decides without asking). The answer it offers is `ja` when the system's language is Japanese, `ko` for Korean, `zh-cn` for Chinese (`zh-tw` in Taiwan and Hong Kong), `off` for Vietnamese, and `en` otherwise.
  XeStrata's measurements (`xestrata --setup --bench`, the measuring tools in `tools/`, the release's and the upstream syncs' comparisons) send `"script_guard": "off"` with every request, so this setting does not touch them.
  The web app's Chat settings start at "Japanese" when the page is in Japanese, "Korean" in Korean, "Chinese (Simplified)" in Chinese and "English etc. (non-CJKV)" in any other language; turn it off to have the model write a language whose characters it bans.
  An engine that takes no logit_bias (an older engine, or `"parallel"` batch slots) does not apply it, and the server's window says so in a line.
  `script_guard` in `GET /props` is the value a request that names none gets; `script_guard` in `/metrics` (Prometheus: `strata:script_guard_*_total`) counts the requests it applied to, the characters written again and the ones let through.
- **A reply stuck on one token is ended.** When a reply repeats the same token 256 times in a row, the server ends it there with `finish_reason` `"length"` and says so in its window:
  a model in a loop, or a broken state that answers one token forever (upstream's #606 saw 36,689 tokens of `!`).
  `"repeat_stop_tokens": N` in `xestrata-<model>.json` sets the run length; `0` turns it off (for a request that really wants one token many times).
- **Non-finite logits end the request and drop the cached state (on by default).** Rarely a window's logits become NaN and the reply degenerates into token 0 (`!`) repeated (upstream's #606, #879).
  The poison sits in the conversation's K/V and recurrent state, so every later turn that reuses it stays broken.
  Every window of `--serve` (decode, a layer split's `--pipeline-windows`, the prompt's short tail, batch slots) checks its head logits on the GPU beside the greedy pick (one small kernel into page-locked flags, read after the sync the window already does).
  A flagged window emits none of its tokens; the engine prints a line with the position and ends the request with `DONE ... nonfinite` (a slot: `BDONE <slot> <n> nonfinite`).
  The next request first drops the live session, every checkpoint, every parked conversation and every idle slot, and reads its prompt from token 0 (a slot still running finishes its reply and is not kept after it).
  The server sends a request that has sent nothing yet again once, transparently; a reply that had started ends as `"length"`; a second fire on the retry ends the request with an error (503 before an answer, an error event in a stream).
  When the flagged window was one that reads the prompt's short tail (the part after the last turn boundary, read through the windows, `--short-read`), the retry reads its whole prompt through the batched path instead (upstream #3465: a reporter's prompts failed in that tail every time and passed with `--short-read 0`); the engine prints a line saying so.
  The kernel only writes flags, so an answer without a NaN is the one the engine gives with `XESTRATA_NAN_GUARD=0` (no kernel recorded).
  Debug switch: `XESTRATA_NAN_INJECT=<n>[,<n>...]` reports the n-th window the engine checks as non-finite.
- **Anthropic requests that don't ask for thinking.** A `/v1/messages` request with no `"thinking"`, effort or budget is rendered without thinking, by Anthropic's own rule.
  Claude Code's short helper calls (a session title in a few dozen tokens) need it.
  `"anthropic_thinking": "model"` in `xestrata-<model>.json` makes such a request think as the model's template does.
  `POST /v1/messages/count_tokens` renders and tokenizes a request as `/v1/messages` would, without generating output tokens. A picture counts as the image tokens it becomes when read, so one not read yet is read by the image encoder (a picture read once is kept, for later requests too; an encoder an idle unload stopped is brought back by loading the model, as the request would). Pictures in tool results also wake the encoder before they can be omitted as unreadable. A refusal to load is a request error, not an omitted picture.

### Streaming and connecting

- **Streaming.** With `"stream": true` everything arrives as it is made: the thinking, the answer, and tool calls
  (the tool's name first, then its arguments piece by piece, as OpenAI and Anthropic do).
  While the model reads a long prompt the stream sends keep-alive messages, so agents do not time out.
  The server window prints progress every 15 s, and `GET /status` says what it is doing (`reading the prompt`, `answering`, tokens so far).
  Closing the connection or pressing stop in your app really stops the model, so the next request starts at once.
- **Which tool calls come back as calls.** A call counts only when it names a tool the request declared.
  A call for another name, or any call in a Chat Completions or Anthropic request that declared no tools, is returned as the text it was written as, so a client never runs a tool it did not offer (the Responses route too, but for a Codex compaction request, which shows its conversation's tools in the prompt and still gets its calls back; an explicit `tool_choice: "none"` forbids calls even there).
  A namespaced tool (`mcp__filesystem.write_file`) is that call when the model writes `mcp__filesystem__write_file`, `mcp__filesystem=write_file`, `mcp__filesystem_write_file`, or the bare member (`write_file`) and exactly one declared tool has that member.
  A name with `__` in it (`mcp__playwright__browser_navigate`) is also that call when the model writes every `__` as `_` (`mcp_playwright_browser_navigate`), unless two declared tools collapse to the same spelling.
  Names are never guessed: `bash` is not `shell`, and a reordered name (`server1_label_values` for `label_values_mcp_server1`) is not the declared one.
  A call whose envelope drifted by a token, `<=NAME>`, `<function:NAME>` or `<NAME>` instead of `<function=NAME>`, is read as the call when NAME is a declared tool.
  So are the forms next to the template's (a call that opens with `<parameter=NAME>`, JSON inside `<tool_call>`, a `<function=NAME>` block without `<tool_call>`, a second call inside one `<tool_call>`) when the call is a declared tool's (the default since upstream 0.2.8).
  On 1,462 agent turns 98.3% read as intended with it and 92.9% without.
  `"tool_call_recovery": false` in `xestrata-<model>.json` turns it off.
  A well-formed call of a name the request did not declare can come back as a tool call instead, with `"undeclared_tool_calls": "call"` in `xestrata-<model>.json` (or `XESTRATA_UNDECLARED_TOOL_CALLS=call`; the config wins): it is for a client with its own policy for unknown names (it answers the model with the tool names it has). The default, `"text"`, stays the text the model wrote, with `strata_rejected_tool_calls` beside it (upstream #4622).
  A `<tool_call>` block that still does not read as a call is counted in the server window.
- **What the parser repairs, and what it leaves.** A string argument that quotes a whole tool call, or repeats its own `</parameter><parameter=NAME>`, stays exact (upstream #4078).
  A call whose `</function>` was left out is read as the call (upstream #4427), in the answer and inside the thinking.
  An argument the tool declares as an array or an object, or declares nothing for, is read as written when it has a slip in it: a trailing or doubled comma, single quotes or `True`/`False`/`None`, a bare string between two members of an object (`"status": "pending", "in_progress", "pending": false`), brackets left open at the end.
  The server window says `N tool call argument(s) repaired` and `/metrics/prometheus` counts them in `strata:tool_args_repaired_total`.
  Text that is not an array or an object, a string cut short and a string-typed parameter are never touched.
  After the thinking has ended at a closing think tag, a second one at the start of a line in the answer is dropped (the model quoted the tag while thinking, upstream #4077); the call that follows is kept.
  The reasoning between the two went out as the answer already, because a stream cannot take text back.
- **A call that came back as text because its name was not declared.** The text stays exactly as the model wrote it and `finish_reason` stays `stop` (`end_turn` on `/v1/messages`).
  Next to it the reply carries a list, only when there is something in it: `"strata_rejected_tool_calls": [{"name": "bash", "reason": "undeclared"}]` (at the top level of a Chat Completions reply, in its last streamed chunk, and in the `message_delta` event and the message of `/v1/messages`; up to five names).
  A client that wants to recover can answer the model with the tool names it does have and ask again; XeStrata never runs or forwards a call for a tool the request did not declare.
  The server window prints the same names, and `/metrics/prometheus` counts the blocks in `strata:tool_calls_unread_total`.
  A block that is malformed (not an undeclared name) is counted there as well, but has no name to list.
- **Anthropic's tool search on `/v1/messages`** (upstream #4663). A tool with `"defer_loading": true` is declared but not listed in the prompt; a `tool_reference` block in a `tool_result` (a tool search's result) puts that tool's definition into the prompt where it stands, in the form the tool list uses, so loading a tool leaves the prompt's start, and the prompt cache, unchanged.
  Any other reference is left out, as before (a tool whose MCP server is gone must not fail the session).
  A request with a deferred tool holds back more tools than it declares (Claude Code declares ToolSearch and a `DeferredToolPlaceholder` and names its MCP tools in the conversation), so there a call for an undeclared name is returned as a `tool_use`: Claude Code runs it, where text ended the turn.
- **Chat apps.** Any app with an "OpenAI-compatible" provider works: base URL `http://127.0.0.1:8095/v1`, any API key.
- **OpenCode** (upstream #543). A starting point for `opencode.jsonc` (in your project, or `~/.config/opencode/`).
  The field names are OpenCode's, so check its config docs if your version differs:

  ```jsonc
  {
    "$schema": "https://opencode.ai/config.json",
    "provider": {
      "xestrata": {
        "npm": "@ai-sdk/openai-compatible",
        "name": "XeStrata (local)",
        "options": { "baseURL": "http://127.0.0.1:8095/v1", "apiKey": "none" },  // or your api_key
        "models": {
          "xestrata": {
            "name": "Qwen3.8-Flash-Next (XeStrata)",
            // context: what you chose in setup; output: what one reply may use (prompt + output must fit)
            "limit": { "context": 262144, "output": 32768 },
            "options": { "reasoningEffort": "high" },                  // sent as reasoning_effort
            "variants": {                                              // switch between them in OpenCode
              "low": { "reasoningEffort": "low" },
              "medium": { "reasoningEffort": "medium" },
              "none": { "reasoningEffort": "none" }
            }
          }
        }
      }
    },
    "model": "xestrata/xestrata"
  }
  ```

  Set `limit.context` to the context you chose in setup: OpenCode compacts the conversation before it gets there.
  Keep `limit.output` well under it: a request whose prompt plus `max_tokens` runs past the context has its `max_tokens` shortened to the room left (see **Context** below).
  For a hard cap on the thinking, add `"reasoning_budget_tokens": N` to `xestrata-<model>.json` (see above).
  OpenCode writes an `<env>` block (working directory, today's date) near the start of its system message, ahead of ~40K characters of instructions and tool definitions that are the same in every session.
  The prefix cache only reuses a prompt's start, so every new folder and every new day reads ~14K tokens again.
  `"env_to_user": true` in `xestrata-<model>.json` moves that block to the front of the first user message, so the system turn is the same for every session (opt-in, upstream d8a49433).
  Measured by upstream (RTX 5070 Ti Laptop, a new session in a second folder): the first request took 1.1 s instead of 11.9 s, and the model still answered the folder and date correctly.
- **Claude Code**: set `ANTHROPIC_BASE_URL=http://127.0.0.1:8095` and `ANTHROPIC_MODEL` to a Claude model name it knows.
  Claude Code refuses names it doesn't know; XeStrata ignores the name.
  Add any `ANTHROPIC_AUTH_TOKEN` (or your `api_key`, if you set one).
- **Codex CLI**: see [the Responses API](#the-responses-api-and-codex-cli) below.
- **Mainplane**: see [Mainplane](#mainplane) below.
- **Context.** Chosen in setup (8K–262K).
  Requests longer than that are refused, never silently cut.
  A request whose `max_tokens` would run past the context is answered with `max_tokens` shortened to the room left (agents such as Claude Code ask for 32,000 on every request); the thinking budget follows the shortened cap (upstream f31b6efd6).
  For the strict llama.cpp behaviour (400), set `"fit_max_tokens": false` in `xestrata-<model>.json` or pass `--no-fit-max-tokens` to `serve/server.py`.
  A prompt that leaves no room at all is always refused.
- **Model names.** Setup names a model by the levels of its questions: the model, the version (when it is not the original), the quantization and the size; SC117's Swift 1.5 in GSQ-RCO's IQ3_XXS, for example, is `qwen3.8-flash-next-swift-sc117-gsq-rco-iq3_xxs`.
  The `id` in `/v1/models` and the `model` of the replies are this name.
  A model installed before 2026-11-09 takes it when setup is run again, and its earlier name (`qwen3.8-flash-next-iq3_xxs` and the like) stays an alias.
- **Model aliases.** `"aliases": ["qwen", "local-model"]` in `xestrata-<model>.json` lists the model under those names too in `/v1/models`
  (each with its own `id`, and in the model's `aliases`), like llama-server's `--alias`.
  A request naming one is answered under that name.
  Any other name is still served.
- **A model menu for a host that swaps models (upstream 717e7e8).** On a host that runs one model at a time on a GPU and swaps them, `"model_switcher"` in `xestrata-<model>.json` names the RPC that does the swap (a URL taking `{"mode": m}`); the web page's header then has a menu of the models served on this port.
  Choosing one makes the server ask the RPC for the swap, and the page waits for `/health` to name the new model, then reloads.
  Only XeStrata's own page can ask for a swap (JSON, as for `/settings`).
- **Model settings in the web page (upstream #564).** The About tab's Model settings card shows and changes a few keys of `xestrata-<model>.json`:
  the `sampling` defaults (temperature, top_p, top_k, min_p), `reasoning_budget_tokens`, `fit_max_tokens`, `anthropic_thinking`, `effort_position`, `aliases`, `idle_unload_s`, `lazy_load`, `engine_silence_s`, `api_monitor`, `debug_chat_log`, `debug_chat_log_mib`, `open_browser` and `--vram-reserve-mib`.
  An empty field removes the key (its default).
  Every other key of the file stays as it is, the earlier file is kept as `xestrata-<model>.json.bak`, and the model uses the change from its next start.
  Only XeStrata's own page can save (JSON, with the API key when one is set, as for the Chat settings); the network, key, MCP and program keys are not editable there.
- **Settings for every model.** A few keys can be written, besides a model's `xestrata-<model>.json`, into the user's settings (`~/.config/xestrata/settings.json`, on Windows `%APPDATA%\xestrata\settings.json`), where they apply to every model.
  These are the web search's `search_engine`, `search_url`, `deepseek_api_key` and `tinyfish_api_key` ([Web search](#web-search)), and the debug chat log's `debug_chat_log` and `debug_chat_log_mib` ([Debug chat log](#debug-chat-log)); the two API keys are held in settings.json only.
  A model's config that has the key wins, and an empty value (`""`) turns the key off for that model alone.
  The Model settings card gives such a key an "Every model" field, and shows the value in use and where it comes from; `off` in the model's field turns it off for that model alone.
  `POST /config` with `{"set": {...}, "scope": "global"}` changes settings.json, without `scope` (or with `"model"`) the model's config; settings.json's other keys (setup's) stay as they are.
- **From other devices on your network.** The server listens on your PC only (`127.0.0.1`) unless you say otherwise:
  run `./setup.sh --host 0.0.0.0 --api-key some-long-secret` (kept in the model's config for the next starts), or add `"host": "0.0.0.0"` and `"api_key": "..."` to `xestrata-<model>.json`.
  The server window then prints this PC's addresses (`from other devices: http://192.168.x.x:8095/`).
  For IPv6 use `--host ::` (or `"host": "::"`, all IPv6 interfaces) and put the address in brackets in URLs, for example `http://[2001:db8::7]:8095/v1`.
  Open that on the other device, or use `.../v1` as an API base URL.
- **From the internet.** Put a tunnel in front of it, for example
  [cloudflared](https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/do-more-with-tunnels/trycloudflare/):
  `cloudflared tunnel --url http://127.0.0.1:8095`.
  **Set a key first**, or anyone with the link can use your PC:
  add `"api_key": "some-long-secret"` to `xestrata-<model>.json` (or set the `XESTRATA_API_KEY` environment variable); clients then send it as their API key.
  Several keys, separated by commas, are all accepted (`"key1,key2"`, as llama.cpp does; `--api-key key1,key2` likewise).
  An API key that is given but empty refuses to start the server (it would switch authentication off).
  To serve on the network without a key on purpose, write `"api_key": "none"` (or pass `--no-api-key`): the server starts and warns once that it has no key (and, with MCP tools, that the tools can read and change files on this PC).
  Streamed answers carry `X-Accel-Buffering: no`, so nginx-style proxies pass each token on at once.
  The web app's settings and MCP tools only answer XeStrata's own page.
  When you open it through a proxy or tunnel whose address differs, add that address, e.g. `"trusted_origins": ["https://strata.example.com"]`.
  With the key set, any `Host` name reaches the server (see Host names below).
- **From web apps in a browser (CORS).** Off by default.
  `"cors_origins": ["https://chat.example.com"]` lets pages of those origins call `/v1/*` from the browser (Open WebUI's direct connections, browser extensions).
  `["*"]` lets any page do it: only sensible with an API key.
  It never opens `/settings`, `/unload` or the MCP tools.
- **Host names (DNS rebinding).** A web page of another site can point its own name at `127.0.0.1` and then reach this server as if it were its own.
  So without an API key the server answers only requests whose `Host` is a name it knows.
  With a key the check is off: such a page cannot send the key, and tunnels and proxies that pass their own name on keep working.
  The names it knows, on any port:
  `localhost` (and `*.localhost`), any IP address (`127.0.0.1`, `[::1]`, `192.168.x.x`, ...), the address it listens on,
  and, when it listens beyond this PC (`0.0.0.0` or a LAN address), this PC's name (`mypc`, `mypc.local`) and the names a container reaches this PC by (Podman's `host.containers.internal`, Docker's `host.docker.internal`).
  Others get **403** naming the setting, and the server window prints one line for each.
  To reach it under another name (a reverse proxy that keeps the name, a tunnel, a DNS name on your network, another container's name for it), add the name:
  `"allowed_hosts": ["strata.example.com"]` in `xestrata-<model>.json` or `XESTRATA_ALLOWED_HOSTS=strata.example.com` (comma-separated).
  `".example.com"` allows that name and every name below it, and `["*"]` turns the check off (so does setting `api_key`).
  The hosts of `trusted_origins` count as allowed.
  Requests without a `Host` header (HTTP/1.0 clients) pass.
- **Web pages without an API key.** Without `api_key`, a `POST` to `/v1/*` that carries an `Origin` header (a browser page sent it) is answered only for:
  XeStrata's own page, pages on `localhost` or an allowed host name (any port), the origins in `trusted_origins` or `cors_origins`,
  and browser extensions and desktop apps (`chrome-extension://`, `moz-extension://`, `app://`: no web site can send those).
  The body must be JSON.
  Any other page, and `Origin: null`, gets **403**.
  Clients that send no `Origin` (curl, the OpenAI and Anthropic SDKs, other servers) are not affected.
  With an API key, the key decides.
  `POST /unload`, `POST /load` and `POST /config` take `Content-Type: application/json` from XeStrata's own page (or no `Origin`), like `/settings`.
- **One line per request.** `XESTRATA_REQUEST_LINES=1` in the server's environment prints each finished request's numbers from the engine's log
  (`request prompt P cached C output O prompt_read R ms total S ms prefill X tok/s decode Y tok/s`),
  for a supervisor that only sees the server's output.

### The conversation cache

A request that continues a chat reads only the part after what the engine already holds:
the live session, or one of the checkpoints it keeps in RAM (up to 6, about 118 MB each).
Checkpoints are taken at the start of each new assistant turn and every 16K prompt tokens.
A checkpoint is used only when the prompt starts with exactly its tokens and pictures.
The checkpoint at the start of the assistant turn is copied in one batch into a pinned staging buffer, and written into ordinary RAM after the reply (on the RTX 4070 a continuing request waits 30-40 ms less each time; `XESTRATA_CKPT_STAGE=0` writes it before the reply).

The oldest checkpoint is kept for good.
In practice it is the end of the system prompt, which every chat of the same client shares.
The rest rotate by least recent use, so a **new** chat that shares that prefix starts reading after it instead of from token 0.
A prompt read from the start is also checkpointed at the end of its system prompt when that is 2,048 tokens or more (upstream PR #62 and #65),
so the root exists for agent clients with long system prompts and tool lists.
Engine options: `--prompt-cache N` (0 = off), `--prompt-cache-every N`, `--prompt-cache-root N` (0 = no system-prompt checkpoint), `--turn-token ID`.
`--prompt-cache-every` does not set the size of the pieces a prompt is read in: the checkpoint every N tokens is taken at a chunk boundary of the prefill, and the read goes on.
What splits a read is the end of the system prompt (`--prompt-cache-root`), the start of the last message, the `XESTRATA_CACHE_MESSAGE_BOUNDARY` boundary and the `"strata_prefix"` boundary (the chunk size is `--prefill`).
`--prompt-cache-every 0` takes neither the checkpoint every N tokens nor the `--prompt-cache-tail` one, so a client that rewrote an earlier message reads again from a checkpoint that much farther back.
A reply is checkpointed too, every `--reply-checkpoint-every N` tokens (default 512, 0 = off; not with Qwen3.8-Flash-Next's `--pipeline-windows 2`), the newest kept.
A next request that leaves the reply near its end reads from there. After a thinking budget's wrap-up the server goes on with the reply's text encoded whole, and the next turn comes in that form, so the last token can differ from the one the model wrote (Qwen3.6-35B-A3B on a B70, a reply with a thinking budget of 150: its continuation read 158 tokens again before and about 23 after, the next turn 236 and 18).

A one-shot request that no later request continues (a classification call, a probe) can send `"strata_checkpoint": false` in its body.
It takes no checkpoint at the start of its last turn nor every 16K tokens, and its session is neither kept nor parked for a next request.
It still starts from a checkpoint it matches, and still takes the system-prompt root when that reaches `--prompt-cache-root`.
Without the field, or with `true`, nothing changes.

For many questions about one long document a request can say where the document ends (`"strata_prefix"`, opt-in, upstream 13d14a49).
It takes `{"messages": 1}` (the first message is the document), `{"message": 0, "chars": 210000}` (the first 210,000 characters of message 0, for a client that sends the document and the question in one message) or `{"tokens": 12345}`.
The engine reads the prompt in two parts there and pins the checkpoint at the boundary.
A pinned checkpoint is never evicted, a parked conversation holding it stays parked, and `SAVE` keeps it.
When a later question resumes from it, the question it leaves is not parked.
The next question starts at the end of the document, so any number of questions cost one read of it.
On the B70 with IQ2_XS, three questions after a 1,791-token prompt: the second and third reused 1,766 tokens and read only 19-20.
The field is checked against the prompt's own ids (the prefix is always the longest common start), a prefix that cannot be marked is said in the server's log and ignored, and a malformed field is a 400.
One pinned prefix at a time per engine (a new one replaces it); it needs `--prompt-cache 3` or more.
`tools/research_run.py` runs a document and a question list against a server with and without the field.
The engine's own key is `pin=N` on the `GEN` line.

Two options are off by default (both on one GPU).

- `--prompt-cache-tail` (an engine argument): one more checkpoint at a chunk boundary within one chunk of the prompt's end.
  A continuation that branches between the every-16K checkpoints reads less again. This checkpoint is evicted first and pushes out no other.
- `--prompt-cache-spread` (an engine argument, opt-in, upstream a7d4b754): the checkpoints kept spread back from the conversation's end.
  Least recent use keeps the newest: after a few short turns all 6 sit within the last 1-2K tokens, and an agent that rewrites its history a few thousand tokens back matches none of them and reads again from the system prompt.
  With the flag the checkpoint that leaves is the one whose removal leaves the smallest gap for its distance from the newest, so the kept points thin out geometrically (after 200 turns of 1,000 tokens: 0, 1K, 3K, 8K, 43K and 200K before the end), and an edit d tokens back reads about d again.
  The same number of checkpoints and RAM; the system-prompt root, a pinned prefix and the newest point are never dropped.
- `XESTRATA_CACHE_MESSAGE_BOUNDARY=1` (the engine's environment): a checkpoint before the last user message (at the end of the turn before it).
  An agent that keeps a long history and replaces only the last message does not read the history again.
  It applies to a history of 8,192 tokens or more with a last message of at most 1,024; it splits the prompt's chunks and costs a fresh request the time of the snapshot.

### Several conversations

Add `--conversation-cache-mib 8192 --conversation-cache-slots 4` to the engine arguments (the config's `args`) to park up to four conversations in at most 8 GiB of RAM.
When a request continues another conversation than the one the engine holds, the engine first copies the held one aside
(its K/V cache, running state and checkpoints) and puts back a parked one whose tokens start the request, instead of reading it again.
A chat and an agent's subagent can then alternate without re-reading each other.

- Requests still run one at a time.
  Nothing tells conversations apart but their tokens, pictures and control-vector setting.
- Without `--conversation-cache-mib` the budget is sized from the RAM free once the model is loaded (upstream #3493): 40% of what is left above `--conversation-cache-min-free-mib`, at most 8192, in steps of 256, and off when that is under 1024; the start log says which in one line. `--conversation-cache-mib 0` turns it off, and so does `--prompt-cache 0`; with Qwen3.8-Flash-Next, so does running without `--mtp`.
  Parked conversations are written to the disk as well, as [the next section](#keeping-parked-conversations-on-the-disk) says, so they serve after the RAM cache pushed them out and after a restart.
- Before each copy the engine checks that `--conversation-cache-min-free-mib N` (default 2560) of RAM stays free (the figure is read as below).
  If it does not, it evicts the oldest parked conversations and checks again (upstream 940011da).
  If that is still not enough, or the copy does not fit the budget, it skips parking and reads the prompt as before.
  To fit the budget, too, the oldest parked conversation is dropped first.
- A new conversation that starts with a checkpoint inside a parked one (subagents that share a system prompt and tool list, a compacted history) borrows the K/V up to there and the checkpoint, and the parked conversation stays where it is (the engine log says `borrowed`; upstream c9e571e0 and 2b296987, #1164).
  A match on a parked conversation's live state or its deepest checkpoint is that conversation going on and moves it into the session as before; so does a match that reaches a pinned shared prefix (`pin=N`).
  When the outgoing conversation fits only in the parked one's room, it keeps its place and the parked one is moved in whole, as before.
- `--conversation-cache-min-tokens N` (default 0 = every one) parks only conversations of at least N tokens (upstream 624070c5).
  A parked conversation takes one of the slots whatever its length, so a client that puts short side requests between the turns of a long conversation filled the slots with them and evicted the long one.
- `--conversation-cache-park-gain N` (opt-in, default 0 = park every turn) parks a conversation again only after it advanced N tokens past the copy it would supersede (upstream d9cb22d2, #2265).
  A turn that grew by a few hundred tokens keeps the older copy instead of writing a whole snapshot that the next turn drops; a resume from the older copy then reads up to N tokens again.
- A copy that fails to go back stops the engine instead of letting it answer from half a state.
- After a restore, the K/V pages a conversation has not changed are reused for its next copy (`XESTRATA_SNAPSHOT_FULL_CAPTURE=1` turns that off).
  The new pages get room for the next turns of at most an eighth of each buffer (and at most 16 MiB), which the budget counts like the rest of the copy (upstream 53f72571).

**How the available RAM is read** (upstream daa76552): the RAM checked before parking a conversation and before saving or restoring a conversation file is the smallest of:

1. `MemAvailable`
1. The room under the memory limits of the engine's cgroup and of every cgroup above it (v2 `memory.max` and `memory.high`, v1 `memory.limit_in_bytes`), less what is in use.
   The clean inactive file cache (`inactive_file` in `memory.stat`) is not counted as in use.
   A cgroup file that cannot be read makes the guard warn once and use `MemAvailable` alone without `--memory-limit-mib`; with it, the figure is unknown and nothing is parked.
1. `--memory-limit-mib N` (or `XESTRATA_MEMORY_LIMIT_MIB=N`; the flag wins, 0 = no limit) less what the engine's container uses now.
   For a container that cannot see its own limit: the total, in MiB, it may use.

Started with `--serve`, the engine prints this breakdown in one `memory guard:` line.

**When the expert arena does not fit in the RAM that is free** (upstream #4051): the engine checks the RAM before it reads the experts into the arena.
A memory limit below the arena (a container) stops the start with the numbers.
Less free RAM than the arena (an engine that was still exiting, another program) is waited for up to 20 s, and the log says so (`XESTRATA_ARENA_RAM_WAIT_S` changes the seconds, `0` loads at once).
If it is still short, the load goes on with a warning that says how much, never a refusal.
Below half of the arena the warning adds that the whole PC can stall in memory reclaim, and that a systemd `MemoryHigh=` below the RAM keeps one program from taking all of it.

Measured on the B70 with IQ2_XS: a 970-token conversation (8K context, FP16 KV) is 251 MiB parked, copied out in 70–104 ms and back in 21–23 ms.
Its answers and state match a run without parking byte for byte:
with FP16, INT8 (also with KV streaming), Q4_0 and K8V4 KV, and with INT8 on the small configuration (8 GB, no XMX).
With parking off, the engine gives the same tokens and state as the engine before the change.

### Conversation cache copies on a slow host link

Where the card sits behind a slow host link, the conversation cache's copies are on the path of the next request.
A checkpoint copies the DeltaNet state (118 MB on Qwen3.8-Flash-Next) from the card to RAM, and parking a conversation copies its K/V and its checkpoints before the next prompt can start.
These variables (the engine's environment) move the copies card-to-card in VRAM and onto background threads (upstream #3779).
The answers and the parked images are the same bytes.
Under `--serve`, when the link to the card probes below 12 GB/s and none of the three is set, `XESTRATA_CKPT_ASYNC`, `XESTRATA_CKPT_GATHER` and `XESTRATA_PARK_ASYNC` are set to 1 by themselves (the start log says so). Otherwise they are off, and setting any of them to 0 keeps all three as set.

- `XESTRATA_CKPT_ASYNC=1`: a checkpoint's state is copied into a VRAM stage and moved to RAM by a background thread (the checkpoint at the start of the assistant turn into its pinned staging buffer). A reader waits only for the copy it needs. The stage is the size of the state, allocated at the first save; if it cannot be had, the save copies as before.
- `XESTRATA_CKPT_VRAM_SLOTS=N` (0-16, with `XESTRATA_CKPT_ASYNC=1`): the last N saved states also stay in VRAM (N x 118 MB), and a restore from one copies card-to-card.
- `XESTRATA_CKPT_GATHER=1`: a checkpoint's 37 small copies are gathered into one buffer and fetched with one copy.
- `XESTRATA_PARK_ASYNC=1`: a park copies the K/V into a VRAM stage and moves it to RAM in the background. The checkpoints the request drops move into the parked image instead of being copied.
- `XESTRATA_PARK_VRAM_MIB=N` (with `XESTRATA_PARK_ASYNC=1`): a parked conversation's K/V also stays in VRAM slots (64 MiB granules, up to N MiB in all), and switching back restores it card-to-card. When the K/V needs the VRAM to grow, the slots go first.
- `XESTRATA_PARK_TRACE=1`: diagnostic, the save and park steps' waits and timings on stderr.

The VRAM stages and slots are taken only while a sixteenth of the card's memory stays free beside them.

### Keeping parked conversations on the disk

Setup writes `"conversation_save": "<models folder>/conversations/<model name>"` into each model's config.
The engine also writes its parked conversations to that folder, and puts one back from there when a request continues a conversation the RAM cache pushed out, or one from before a restart.
`"conversation_save": false` in the config turns it off; running setup again keeps that value and the other `conversation_save_*` keys.
Nothing is written while the previous section's conversation cache is off (when the RAM left is too little, for example).
A relative path is from the XeStrata folder.

- A conversation is written when the RAM cache pushes it out, and at a normal exit (the parked ones and the one the engine holds).
  With the models other than Qwen3.8-Flash-Next, a conversation too large for the RAM cache is written then as well.
  If the engine crashes, a conversation not yet written is lost.
- For each request, the engine looks for a file that matches more of the request's start than the conversation it holds and the parked ones, and reads it back if there is one.
  When reading it would leave less RAM free than `--conversation-cache-min-free-mib`, the prompt is read instead.
- With Qwen3.8-Flash-Next, each conversation is written with up to `"conversation_save_checkpoints"` (default 2) checkpoints:
  the start of the last turn (what the next turn of the same chat resumes from), the end of the system prompt (where a new chat of the same client starts),
  then the most recently used.
  A conversation takes about 118 MB × (1 + the checkpoints), plus about 14 KB of KV per context token (INT8).
- When the folder holds more than `"conversation_save_mib"` (default 32768), the files used longest ago are deleted first.
  A file not used for `"conversation_save_hours"` (default 48) hours is deleted too, at start and every 10 minutes between requests.
  Reading a file back counts as using it.
- A file is not written when it would leave less disk space free than `--session-min-free-mib` (default 4096, the same floor as SAVE's).
- Writing a conversation deletes the file of the same conversation a turn back.
- A file is used only by an engine with the same model files, engine version and settings (KV precision, context length, MTP, control vectors, and so on).
  Other files are left alone until their age or the budget removes them; the budget counts every file in the folder, so setup gives each model a folder of its own.
  A damaged file is deleted and the prompt is read as before.
- Only the owner can read the files (a folder the engine makes is 0700, the files 0600).
  They hold values computed from the conversations: keep them on this PC.
- `"conversation_save_compress": true` writes the floating-point parts of a file (the DeltaNet states, the indexer rows, FP16 KV, GLM-5.3-Flash's latent rows) compressed with c-blosc2's ZSTD at level 1 (off by default).
  A file gets about 10% smaller with INT8 KV and about 15% with FP16 KV. KV of 1-byte codes (INT8, Q4_0, K8V4) hardly compresses and is written as it is.
  It is available only when the build found libblosc2-dev (Debian / Ubuntu); with the key and a build without it, the engine does not start.
  A compressed file is read by any engine built with libblosc2, with or without the key; an engine without it deletes the file and reads the prompt again.
- As engine arguments: `--conversation-save DIR`, `--conversation-save-mib N`, `--conversation-save-hours N`, `--conversation-save-checkpoints N`, `--conversation-save-compress`, `--session-min-free-mib N`.

Measured on the B70 with IQ2_XS, a conversation of 9,183 tokens (INT8 KV, two checkpoints) took 471 MiB.
After a restart it was read back from the CPU-attached NVMe in 0.28 s, and only the 7 new tokens were read (without the file, the 9,167 tokens are read again in 7.9 s).
The answer matches a run without the restart, also on the small configuration (8 GB, no XMX).
Writing a conversation of 300-360 MiB takes 0.10-0.16 s; when the RAM cache pushes one out, the next request waits that long.
Compressed, the same conversation takes 421 MiB, 0.30 s to read back and 0.23 s to write.

### Saving a conversation to a file and restoring it (opt-in)

The conversation the engine holds can be saved to a named file and restored later, also after a restart.
A request that continues the restored conversation does not read its long prompt again.
Started with `--slot-save-path DIR` (`"slot_save_path"` in the config), the server takes llama-server's slot API save and restore, for slot 0 only.
The file format is XeStrata's (Strata's), not llama.cpp's. There is no erase action.

```bash
curl -X POST "http://127.0.0.1:8095/slots/0?action=save"    -H "Content-Type: application/json" -d '{"filename": "chat1.bin"}'
# {"id_slot": 0, "filename": "chat1.bin", "n_saved": 4996, "n_written": 312432956, "timings": {"save_ms": 200.7}}
curl -X POST "http://127.0.0.1:8095/slots/0?action=restore" -H "Content-Type: application/json" -d '{"filename": "chat1.bin"}'
# {"id_slot": 0, "filename": "chat1.bin", "n_restored": 4996, "n_read": 312432956, "timings": {"restore_ms": 162.3}}
```

- DIR becomes an absolute path at start (`--slot-save-path` from the server's working folder, `"slot_save_path"` from the config's `"cwd"`), created with mode 0700 when missing.
  The files hold the conversation's tokens and state and are created with mode 0600. Nothing deletes them.
- `filename` is a file name inside DIR only: no path, drive, `:`, Windows device name, control character, leading `.`, or trailing `.` or space.
- Before a save, the disk must have room for the new file plus `--session-min-free-mib` (an engine argument; default 4096, 0 = no check).
- `--session-save-reclaim` (an engine argument the server adds by default, `"session_save_reclaim": false` in the config leaves it out; upstream 73ed39ae): a save that fails its RAM check first releases host caches that can be rebuilt, then checks again.
  In this order: the retained K/V copies, the oldest unpinned parked conversations, the unpinned checkpoints other than the one written to the file.
  Pinned prefixes and the checkpoint written stay. The server gives the released RAM up to 30 seconds to show in the figures before it tries again.
- A file is restored only by the same engine version with the same model files and settings (KV precision, context length, MTP, control vector and so on).
  A file that does not match, or a damaged one, is refused, and the engine's conversation stays as it was.
- Only the checkpoint at the start of the last turn is saved: a continuation that edits an earlier turn reads again from there.
- Not with parallel requests (`"parallel"`) or a layer split over several GPUs (`501`, and the engine refuses).
- The request must be `Content-Type: application/json` (else `415`); from a browser page only XeStrata's own page and trusted origins.
- Status codes: `501` without a save folder or with parallel requests; `400` for another slot, action or name, or a file the engine refused; `404` for a restore of a missing file;
  `503` while the model is not loaded or the RAM is short; `507` when the disk has no room; `500` for any other I/O failure and when the engine ended.
  An engine that ended is started again by the next request.

Measured on the B70 with IQ2_XS, the file of a 4,996-token conversation (INT8 KV) took 312 MB, 0.20 s to save and 0.16 s to restore.
Restored after a restart, the continuation was answered in 0.31 s without reading the 4,996 tokens again (8 s to read them all), and the answer matched a run without the save.
The same held on the small configuration (8 GB, no XMX) and with the icpx build.

### Current limits and sampling

- One request at a time (several at once only with `"parallel"`, [BATCHING](BATCHING.md)), and one conversation's history in the KV cache at a time.
  Switching between two chats re-reads the part where they diverge unless parking is on; the shared prefix, such as the system prompt, is reused.
- Images only when set up with them ([below](#images)); no video.
- **Temperature, top_p, top_k, min_p and seed** are honored per request (OpenAI and Anthropic fields).
  With the default adaptive expert tier, a sampled result is not reproducible run to run;
  for seed-reproducible output add `--adapt-every 100000` (static residency) to the engine arguments.
- The run config's optional `sampling` block sets the defaults for requests that leave the fields out
  (`"sampling": {"temperature": 1.0, "top_p": 0.95, "top_k": 20}`).
  A request's own fields always win.
  With no block at all, a request without sampling keys gets the sampling the model card recommends (upstream #2685).
  It is picked by the thinking state: Qwen3.8-Flash-Next (Swift and Gyro too) thinking temperature 1.0, top_p 0.95, top_k 20; not thinking temperature 0.7, top_p 0.8, top_k 20, presence_penalty 1.5; Qwen3.6-35B-A3B and Qwen3.5-9B thinking 1.0 / 0.95 / 20 with presence_penalty 1.5, not thinking 0.7 / 0.8 / 20 with 1.5; HauhauCS's Qwen3.5-9B thinking 0.6 / 0.95 / 20, not thinking 0.7 / 0.8 / 20.
  Ornith-1.5 takes its card's set for general tasks, 0.6, 0.95 and 20 (with or without thinking), dealignai's
  Ornith-1.5 its card's 1.0, 0.95 and 20 (with or without thinking), GLM-5.3-Flash its generation_config's temperature
  1.0 and top_p 0.95.
  llm-jp's card names no sampling, so it is not in the table and decodes greedy as before.
  The order of precedence: the request, the web page's shared Chat settings, the config's `sampling` block, the card.
  Greedy is only for who asks for it: `temperature: 0` in the request, or `"sampling": {"temperature": 0}` in the config.
  The web page sends no sampling of its own until you change its Chat settings (it showed and sent 0.6 before); an old stored 0.6 with the default top_p and top_k counts as not set.
  Chat settings takes a min_p too; empty, it is the server's default (the config's `sampling.min_p`).
- The penalties (`presence_penalty`, `frequency_penalty`, `repetition_penalty`) ride the same path.
  `penalty_last_n` caps how many recent tokens they count over, 64 by default when any penalty is set.
  llama.cpp's names `repeat_penalty` and `repeat_last_n` work too, in the config's `sampling` block and in a request (upstream #1819).
  They apply to every token speculative decoding checks at once, exactly as if it decoded one token at a time.
  The draft layer guesses without penalties, so more of its guesses are rejected and requests with penalties run slower.
  The penalty window (`penalty_last_n`, 64 by default when any penalty is set) is the last tokens of prompt and reply together.
  `XESTRATA_PENALTY_PROMPT=0` counts only the tokens the current reply has written (upstream #4590): the prompt, including earlier assistant turns in it, is not counted, so the numbers or a quoted earlier answer in a new task are not penalised.
  Both engines (Qwen3.8-Flash-Next and plain) count this way.
- `top_k` keeps at most 64 candidates: `0` ("off") or anything above 64 uses all 64.
- A sampling field sent as a numeric string (`"0.7"`) is read and used; a value that cannot be read as a number, or of another type (`true`), is a 400 that names the field (upstream #1663).
- **logit_bias**: `"logit_bias"` in `/v1/chat/completions` (`{"<token id>": -100 to 100}`, or a list of `[id, bias]` pairs) is added before the penalties and filters (upstream dcc6f756).
  -100 (`false` in a pair list) keeps the token out. Not with parallel requests (`"parallel"`).
  On the models other than Qwen3.8-Flash-Next too, a request with a logit_bias decodes in speculative windows (the bias is the same for every row).
- `XESTRATA_SPEC_GUMBEL=1` (opt-in, upstream 6381df34) picks a sampled request's token by Gumbel-max.
  The noise is drawn per token id instead of along the probability-sorted list, so the tokens follow the same distribution while `--coupled-draft` drafts agree more often with the target's pick.
  With `--coupled-draft`, temperature 1.0 / top_p 0.95 / top_k 20 and eight 200-token story and code answers, three runs each, the median decode went from 51.05 to 52.05 tokens/s (+2.0%) on an RTX 4070 and from 80.25 to 72.75 tokens/s (-9%) on the B70.
  The drafts kept on the 4070 rose from 64.1% to 70.8%.
  The result differs by card: measure on yours before turning it on.
- **logprobs:** `"logprobs": true` with `"top_logprobs": K` (0–20) on `/v1/chat/completions` returns each token's log-probability and the K most likely candidates in `choices[].logprobs` (streamed too; from upstream's Intel port, 4ba35fd).
  The tokens, joined, spell the reply exactly (a character split across two tokens lands in the one that finishes it).
  `"allowed_token_ids": [ids]` keeps only those ids in `top_logprobs`, most likely first, renormalised to sum to 1; the chosen token is one of them when it was named (vLLM's field, but vLLM masks generation with it, while here it filters what is reported, so a token outside the named ids keeps its own `logprob` and an empty list).
  An id past the model's vocabulary is a 400.
  A token whose probability could not be read gets no entry, and the reply goes on.
  A `json_schema` reply carries them only when its tokens spell the JSON returned.
  On a server started with `--parallel`, only a `max_tokens` 1 request gets them, and a longer reply is a 400: a batch slot's `BT` lines carry none, and entries for a prefix only could not be told from the model's own answer.
  The values are the model's own, before sampling, temperature and penalties.
  Tokens written while it thinks are listed under `reasoning_content`; the stop token is not listed.
  On the B70 with IQ2_XS the answer was the same with and without it.

### JSON response formats

`POST /v1/chat/completions` accepts `response_format: {"type":"json_object"}` or
`{"type":"json_schema","json_schema":{"name":"answer","strict":true,"schema":{"type":"object","properties":{"answer":{"type":"integer"}},"required":["answer"],"additionalProperties":false}}}`.
The schema must describe an object at its root.
Local `#` references work; remote references are refused.
`json_schema` is checked with the Python package `jsonschema` (setup installs it into `.venv`, the deb and rpm packages depend on the distribution's python3-jsonschema, and the Windows 7z carries it).
Without it a `strict: true` request (`json_schema`'s `strict: true`, Messages' `output_config.format`, a `strict: true` tool) cannot be checked and is a 400, and another `json_schema` answer is only checked to be one JSON object; the server says so once, and `xestrata --setup` says how to install it.

When the engine can constrain its decoding (not with `--batch-groups` 2 or more), **the answer is written under the schema's grammar** and checked after it as well. With tools, only the calls' arguments have a grammar, and the answer is checked after it is written.
The thinking is not constrained: only the JSON answer after it.
Otherwise the schema is in the prompt and the server checks the answer; `STRATA_GRAMMAR=0` in the environment turns the constraint off too.
`strict` is true, false or null (as if not given); another type is a 400.
A tool's `strict: true` (Chat Completions, Responses, Messages; on Responses a namespace's and `additional_tools`' tools too) and a `custom` tool's regex grammar constrain the call's arguments the same way (models with Qwen's call form); text and the other tools' calls are not constrained.
A call of a `strict: true` tool has its arguments checked against the schema before it is sent, whether or not it was constrained: it is not sent while it is written, but whole once it is; one that does not match, or holds a number JSON cannot (NaN, Infinity, one too large), is a `structured_output_failed`, as an answer is.
In Qwen's call form, a string parameter's body is its value as written. Type restoration intersects sibling constraints and `allOf`, unions the branches of `anyOf` and `oneOf`, and resolves local JSON Pointer references from the tool's entire `parameters` schema. These are type hints, not a replacement for schema validation.
For a string `enum` or `const`, JSON string quotes are removed only when the raw body is not a candidate but its decoded string is. If both are candidates, the raw body is kept. Parameters needing that decision are buffered until their end even in a stream.
One generation is made per request, with no hidden retry.
Successful responses contain a validated JSON object.
Malformed JSON, duplicate keys, non-finite numbers, schema violations and incomplete generations return **502** with `error.code: structured_output_failed`;
invalid request schemas return **400**.
JSON formats combined with tools/MCP are refused explicitly.
Without `response_format`, ordinary text and tool behavior stays the same.
Structured SSE buffers the answer while sending keep-alive comments, emits the content only after validation, then usage/timings and `[DONE]`.
A failure emits an SSE error and `[DONE]` without invalid content.
Math answers often hold LaTeX such as `\alpha`, which is not a valid JSON escape.
Add `"repair_latex_escapes": true` to the `response_format` (or to `text.format` on `/v1/responses`) and, only after the strict parse failed, the server doubles the backslashes that are not valid escapes and checks the answer again.
Valid output is never touched.
`\b`, `\f`, `\n`, `\r`, `\t` and `\uXXXX` are valid JSON, so they stay: a `\beta` beside a `\alpha` comes back with a backspace.
Off by default.
`/v1/status.structured_output` advertises the formats, the validation method and the buffered streaming.

### The web page (Chat, Images, Monitor, About)

Open the server's address in a browser.
Everything in it runs from the server's own files: no CDN, no outside address.
Chats, settings and the API key stay in the browser.

**Chat settings** (the button next to the message box) holds the sampling and thinking settings and the switches below.
Every switch below is **off by default**; with all of them off the page behaves as it did before.

- **Render math.** Equations written as `\( ... \)`, `\[ ... \]`, `$x$` or `$$ ... $$` are drawn as formulas by KaTeX 0.19.1.
  KaTeX is in `third_party/main/katex` (MIT; `PROVENANCE.json` there lists the package and a hash of every file), and the server hands it out at `/web/vendor/katex/`.
  The text of a message is never changed: copying, the Markdown export and the next request all carry what the model wrote.
  Code, prices (`$5 and $10`) and a formula still being written stay text. A formula KaTeX cannot read stays as its source.
  A formula can only draw: it cannot make a link, load a picture or run anything (`\href`, `\url`, `\includegraphics` and `\htmlClass` are shown as their own names).
- **Saved chats list.** Keeps every chat in a list on the left, with search over titles and text, rename, delete (with Undo) and days as headings.
  The chats live in this browser's IndexedDB; a picture or file is kept by name only. Nothing goes to the server.
  In a private window, or when the browser's storage is full, the list stays in memory and the page says so once.
  Turning it on keeps the chat on the screen as the first one in the list; turning it off keeps that chat as the single chat.
  While an answer is being written, another chat can be opened or a new one started: the answer goes on in its own chat, which the list marks with a dot.
  When the browser is closed, the server writes the answer on and keeps it, and the same browser gets it in its chat when the page is opened again. The server lets an answer go once it is delivered, and an undelivered one after 12 hours; a restart of the server loses them.
- **Edit and rewind messages.** Under each message: **Regenerate** (a new answer to the same question), **Edit answer** (correct its text; the old thinking and tool calls go and no request is sent) and **Rewind to here**.
  Rewind asks what you want.
  On your message, **Edit and send again here** hides it and the messages after it and puts it in the box; on an answer, **Continue from here** hides the messages after it.
  **Branch from here** (needs the saved chats list) starts a new chat from the messages up to that point and leaves this chat as it is; the branch keeps a name such as "Branch of ..." and links back to the original.
  An **Undo** bar above the box brings back what was hidden, until your next question, another chat or a reload.
  A picture or file that was only kept by name is not sent again: the page asks you to attach it again.
- **When you send while it is answering.** Send works while an answer is written: your message goes into the chat's queue.
  *Send after* (the default): when the answer is done, the queue goes one message after another.
  *Interrupt*: the answer stops where it is, what was written stays and is marked interrupted, and your message goes at once; a tool that is running finishes first.
  Each waiting message has **Take back** (back to the box: cancelled) and **Send now** (the answer stops and it goes first). Stop keeps the waiting messages for you to send yourself.
  The queue is each chat's own: it stays with its chat while another is open, and goes on when that chat is opened again after its answer ended.
- **Go on past the context** (on by default). One chat goes on past the context with no summary, as [Mainplane](#mainplane)'s sessions do.
  Within a segment the earlier messages go back as they were sent, so the prompt only grows and each turn reads only what it added.
  Whether an answer's thinking goes back with it is decided once, when the answer ends: it does while more than half the segment is left, and where dropping it would read again more (the answer and its tools' results) than the thinking holds.
  A message that would take the segment past its size (estimated from the last answer and the message, or refused by the server as too long) starts a new segment.
  A new segment's prompt is a note on the segments and the tools, this chat's memory, the last two exchanges word for word, and the message.
  The earlier segments stay in the chat, and the model finds and reads them back with tools the page runs (`history_search`, `history_read`); what must carry on it writes to the memory with `memory_write`, which every segment starts with.
  **Segment size** makes the segments smaller (empty: the context less the answer's room), and **This chat's memory** shows the memory and lets you correct it.
  The page draws a rule where a segment starts and offers no edit controls on an earlier segment's messages. Since the chat stays in the browser's storage, the page asks the browser not to clear it (`navigator.storage.persist()`).

An answer's ` ```svg ` or ` ```html ` block gets a **Preview** button.
The code is drawn in a frame that is cut off from the page: an empty sandbox (no scripts, forms, popups or downloads, and no access to the page, its storage or the server), a policy inside the frame that allows no address at all, and a page header (`Content-Security-Policy: frame-src 'none'`) that keeps a link or a redirect in the model's HTML from loading a web page into it.
An svg is shown as a picture.
For an HTML page there is also a warning button, **Run scripts**: after you confirm, that one block runs with scripts, still with no way to the page and no network, until you close the preview.
It can still use your CPU, and it is code the model wrote.

**Load / Unload** in About loads the model or frees the GPU (`POST /load`, `POST /unload`).
It asks before unloading and is off while an answer is being written or an image is made (an image made on a GPU of
its own leaves it on).
The light or dark theme is switched in About > Settings.

**Language.** About > Settings lists the page's languages: English, German, French, Spanish, Japanese, Korean, Portuguese, Russian and Chinese.
The default, **Automatic**, takes the first of the browser's languages (`navigator.languages`) the page has (the region dropped when the whole tag is not one: `ja-JP` is `ja`, `pt-BR` is `pt`), else English.
A language you pick is kept in this browser; picking Automatic forgets it.
The Chat view, its settings, the header and the Images view are translated; Monitor and About are still mostly English.
A language is a file `serve/web/locales/<lang>.json`; the English text lives in the code next to its key.
`python tools/web_i18n.py` checks every file for the same keys and placeholders (`--write` rewrites `en.json`).
`/?lang=de` shows a language for one visit.

**Text size.** About > Settings' "Text size" makes the page's text Small, Standard, Large or Larger (0.9, 1, 1.15 and 1.3 times the standard). The choice is kept in this browser. The browser's own zoom makes everything larger, the layout too.

### API request monitor

Off by default, since it keeps prompts and answers in memory.
Turn it on with `"api_monitor": true` in `xestrata-<model>.json` (or `serve/server.py --api-monitor`);
otherwise nothing is recorded and the endpoints below answer 404.
Apart from it, the server keeps in its process memory the text and tokens of its last few prompts, whatever `"api_monitor"` says, to encode the next turn faster; no endpoint shows them, and they are gone when the server stops.

`/api-monitor` shows the model state, load/unload controls, active and queued requests,
their request bodies, output, reasoning and non-stream response bodies, with the queue, load, first-token and decode times apart.
`GET /api/requests` returns compact summaries and `GET /api/requests?id=<id>` one retained request; both use the API key check.
It keeps the newest **100 requests in memory** until the server restarts, at most **262,144 characters per field** (with visible truncation flags).
Headers are not recorded, and the monitor's key is kept in the tab's session storage.
Treat the history as sensitive when the server is reachable from a network: set an API key.

`GET /metrics` also lists each recent request's speculative drafts, `drafts_offered` and `drafts_accepted` (`null` when the engine did not report them),
and their sums since the server started in `totals`.
Its `hit_rate` is the VRAM share of the experts looked up while answering: experts the GPU reads over PCIe (`--pcie-frac`) are not in it, so a higher `--pcie-frac` raises it even when decoding gets slower.
`pcie_share` is their share of all routed experts; the server log and the Monitor tab show it beside the hit rate (upstream #588).
The Monitor tab's list of recent requests also shows the prefill speed (`Prefill t/s`): what the cache already held is not read again, so it is the fresh tokens (`prompt_tokens` minus `reused`) over the prefill time (`prompt_ms`), and a request that read no new token shows no number.

---

### Debug chat log

A log for looking into a conversation afterwards. It writes prompts, answers and pictures to the disk, so it is off by default.
`"debug_chat_log": true` in `xestrata-<model>.json` or `~/.config/xestrata/settings.json` (or `serve/server.py --debug-chat-log`) turns it on, and the start log says so.
It goes to the state folder's `debug-chat-log/` (`~/.local/state/xestrata/debug-chat-log/` on Linux):

- `requests-<date>.jsonl`: one line a chat request: its body, the prompt the engine read (the text after the chat template, each picture's image tokens as one `<|image_pad|>x<count> [picture <key>]`), the server's own tool calls and the results the model read, the answer (reasoning, text, tool calls), usage and timings, an error.
- `img/`: the pictures a request sent and the ones a tool made, one file each, named by their hash. A data URL in a request's body becomes `{"picture": "img/..."}`, naming the file.
- `chats/<chat id>.json`: the web page's own record of a chat, as the browser keeps it (its pictures, files and the copies of the images made included). While the log is on, the page sends a chat each time it saves it.

When the folder grows past `"debug_chat_log_mib"` (1024 by default), the oldest days' lines and the oldest chats go first, then the pictures nothing names any more.
A write that fails is said once in the server's log, and the request goes on.
Set an API key when others can reach the server.

### Chat Completions and Messages fields

Errors come in the API's own shape: on OpenAI's routes `{"error": {"message", "type", "param", "code"}}` (`param` and `code` `null` when there are none), on Anthropic's `{"type": "error", "error": {"type", "message"}}`.
An unknown route is a 404 (`Invalid URL`).
A refusal while the engine starts (and the like) is a 503 on OpenAI's routes and a 529 (`overloaded_error`) on Anthropic's, both with `Retry-After`.

**Files**: Chat Completions' `file`, Responses' `input_file` and Messages' `document` (a PDF, `base64` or `url`) are read.
A PDF goes in as its text page by page and, when the model reads pictures (a setup with vision), each page as a picture, in page order.
A text file (`text/*`, JSON, XML, YAML, ...; UTF-8) goes in as its text.
A file is read up to 32 MiB and a PDF up to 100 pages; another format, an encrypted or broken PDF, and a `file_id` (the Files API) are a 400.
On Linux a PDF is read with poppler's `pdftotext` and `pdftoppm`, on Windows with pypdf and pypdfium2, which the 7z carries. On Linux without poppler a PDF is a 400; `xestrata --setup` says how to install it.

**Chat Completions**:

- A `custom` tool in `tools` is shown to the model as a tool that takes one `input` string; its calls come back as `{"type": "custom", "custom": {"name", "input"}}`. A regex grammar (`format`) [constrains](#json-response-formats) its argument; a Lark grammar is only described to the model.
- `tool_choice` takes `none`, `auto`, `required`, a named tool and `allowed_tools` (the tools shown are those it lists; `mode: required` forces a call). A name that is not one of the tools, or an unknown shape, is a 400. A named tool gets one call of that tool back; another tool's call written in the thinking or after it stays text (Messages and Responses alike).
- With `parallel_tool_calls: false` the reply ends when its first call is complete, and that call is returned.
- `n` (1 to 128) generates each choice in its own run; with a `seed`, choice k gets `seed + k`. Streamed, every chunk has the same `id`.
- `stream_options.include_usage` adds the usage chunk with `choices: []`; the finishing chunk keeps its usage too (as llama.cpp's does).
- The usage has `completion_tokens_details.reasoning_tokens`, and responses carry a `system_fingerprint` (made from the engine's version and the model).
- The old `role: function` and an assistant's `function_call` are read as a tool's result and call. A request with the old `functions` and `function_call` is read as `tools` and `tool_choice` (one call at a time), and answered in the old shape (`function_call`, finish reason `function_call`); they cannot be used with `tools` or `tool_choice`. A null field, and `function_call: "none"` without `functions`, are read as nothing asked.
- A `user` or `system` message's `name` is written before what it says, followed by a colon. An assistant's `refusal` is read as its text when it has no content.
- An image with `image_url` `detail: "low"` is read with its longer side brought down to 512.
- `top_logprobs` is 0 to 20 and only with `logprobs: true`. `input_audio` and audio output (`audio` in `modalities`, or `audio`) are a 400 (the model reads and writes no audio).
- `max_tokens` and `max_completion_tokens` are whole numbers (2.9 is a 400).

**Messages**:

- `max_tokens` is required, a whole number of 1 or more.
- Tool results are paired with their calls by `tool_use_id`; returned `tool_use` ids start with `toolu_`. `disable_parallel_tool_use` allows one call at most. A `tool_choice` name that is not one of the tools, or an unknown shape, is a 400.
- `output_config.format` (and `output_format`) `json_schema` checks the answer against the schema before it is returned, constrained and checked as the [JSON response formats](#json-response-formats) are; an answer that does not match is a 502 (an error event in a stream).
- A `tool_result` with `is_error: true` reaches the model with `Error:` before it.
- A text `document` (`source` `text` or `content`) and a `search_result` are read as text with their titles. `citations.enabled: true` is a 400.
- Thinking blocks carry a `signature` (`signature_delta` in a stream): the thinking's text in base64; thinking sent back that differs from it is a 400. A `redacted_thinking` block XeStrata made is read, others are dropped.
- The stream's keep-alive is `event: ping`. `"thinking": {"type": "adaptive"}` leaves the level to the model.
- The usage in both `message_start` and `message_delta` splits the prompt into `cache_read_input_tokens` (what the conversation cache held) and `cache_creation_input_tokens` (what was read). `cache_control` breakpoints do nothing: prefixes are cached on their own.
- A reply stopped by the context limit has `stop_reason` `model_context_window_exceeded`; one stopped by `max_tokens`, `max_tokens`.
- In a non-streaming reply, a tool call the output ended inside comes back as a text block with the arguments it wrote.

### The Responses API and Codex CLI

`POST /v1/responses` speaks OpenAI's newer Responses API, which Codex CLI uses (it no longer
speaks Chat Completions). It runs on the same path as `/v1/chat/completions`, so the thinking levels, the thinking
budget, the conversation cache and the same API key, Host and Origin checks apply.

It is **stateless**: nothing is stored, so the client sends the whole conversation in `input` every time (Codex does,
with `store: false`). `previous_response_id`, `conversation`, `background` and the retrieve/delete/cancel endpoints
are refused with an error that says so.

| Request | What XeStrata does |
| --- | --- |
| `input` as a string, or as items | `message` items (`user`, `assistant`, `system`, `developer`; text and images), `reasoning`, `function_call`, `function_call_output`, `custom_tool_call(_output)`, `input_file` (the files [above](#chat-completions-and-messages-fields)), an earlier answer's `image_generation_call` (the image is kept, so a following `edit_image` can change it). Hosted tools' items (`web_search_call`, `mcp_call`, ...) go into the conversation as text |
| `instructions` | The system message (with leading `developer` messages; later ones become user messages, as on the chat path) |
| `tools` | `function` tools, `namespace` tools (the model sees `namespace.name`; calls come back with `namespace` and `name`), `custom` tools (one free-form `input` string). Hosted tools (`web_search`, `file_search`, ...) are left out: the model cannot run them. A `function` tool without `strict` (or with null) is strict when its parameters meet strict mode's rules as they are (every object `additionalProperties: false`, every property `required`) and jsonschema is there, else non-strict; the response's `tools` say which (OpenAI's Responses default). The check visits schema positions only, not JSON values such as `default`, `enum` and `examples`. Unresolved references do not justify automatic strict mode; explicit `strict: true` is validated separately. |
| `tool_choice` | `"none"` hides the tools. `required`, a named tool (`custom` and `namespace` ones too) and `allowed_tools` work as on Chat Completions. `{"type": "image_generation"}` forces a call of the image tools (the request's other tools stay available, as on OpenAI's; the name kept to the image tools by a grammar where the engine takes one). A name that is not one of the tools, or an unknown shape, is a 400 |
| `parallel_tool_calls: false` | The reply ends when its first call is complete |
| `max_tool_calls` | The cap on calls of hosted tools (here, the image tools); calls past it are not run. Not applied to `function` tools |
| `truncation: "auto"` | When the conversation does not fit, the oldest items go (`instructions` stay; a call goes with its results). A conversation that still does not fit is a 400 |
| `text.verbosity` | `low` and `high` add one sentence asking for a short or a detailed answer after the system message's own text |
| `include` `message.output_text.logprobs`, `top_logprobs` | `output_text` carries `logprobs` |
| `reasoning.effort` | `none`/`minimal`, `low`, `medium`, `high`/`xhigh`; without it the config's `reasoning_effort`, else the model's default (high) |
| `max_output_tokens` | The output cap (thinking included). Running out ends the response `incomplete` (`max_output_tokens`). `0` and `-1` (or leaving it out) mean no cap, as for `max_tokens`; anything below `-1` is a 400 |
| `text.format` | `json_schema` and `json_object` use the [JSON response formats](#json-response-formats) (constrained and checked) |
| `temperature`, `top_p`, `reasoning_budget_tokens`, ... | As on the chat path |

The final event (`response.completed` or `response.incomplete`) and a non-streaming reply carry the same top-level `timings` object as `/v1/chat/completions`.

The model's thinking comes back as a `reasoning` output item with `reasoning_text` content (streamed as
`response.reasoning_text.delta`). This model writes no separate summaries: when `reasoning.summary` (`auto`, `concise`,
`detailed`) asks for one, the thinking's text also goes into `summary`, with the `response.reasoning_summary_*` events;
otherwise `summary` is empty. A reasoning item sent back with only a `summary` readable has that summary read as the
thinking. With
`"include": ["reasoning.encrypted_content"]` the item also carries `encrypted_content`: an opaque string (base64, not
encrypted; the client already holds the text). Send the reasoning items back with the rest of the conversation, as
Codex does: their thinking goes back into the prompt, so it matches what the model wrote and the conversation cache
is reused.

With `"stream": true` the events are the official ones, in order: `response.created`, `response.in_progress`, then
for each output item `response.output_item.added`, its deltas (`response.reasoning_text.delta`,
`response.output_text.delta`, `response.function_call_arguments.delta`), its `...done` events and
`response.output_item.done`, and at the end `response.completed`, `response.incomplete` or `response.failed`.
While a long prompt is read, a `response.in_progress` event goes out every 15 s, which Codex counts as activity.
Errors before the answer starts have the Responses form (`{"error": {"message", "type", "param", "code"}}`); after
it started they arrive as an `error` event and `response.failed`.

**Codex CLI.** In `~/.codex/config.toml`:

```toml
model = "xestrata"                      # any name; XeStrata answers with its model
model_provider = "xestrata"
model_context_window = 32768            # the context you chose in setup: Codex compacts before it gets there
show_raw_agent_reasoning = true         # show the model's thinking (it writes no summaries)
# model_reasoning_effort = "medium"     # none, low, medium or high; default: the model's (high)

[model_providers.xestrata]
name = "XeStrata (local)"
base_url = "http://127.0.0.1:8095/v1"
wire_api = "responses"
stream_idle_timeout_ms = 600000         # a first, long prompt can take minutes to read
# env_key = "XESTRATA_API_KEY"          # only if the server has an api_key: the variable holding it
```

Then run `codex` (or `codex exec "..."`) as usual. Codex warns `Model metadata for ... not found` for a local model
name; that is expected.

### Mainplane

[Mainplane](https://github.com/mainplane-ai/mainplane) is an agent harness that never summarizes a conversation (no compaction).
The context grows up to `context_limit`, and a session past it is carried on in a new session, which reads what it needs from the old one.
XeStrata takes it through the `openai-chat` shape: add it to the `providers` of the harness's config (`/var/lib/mainplane-server/config.json` on Linux).

```json
"providers": {"xestrata": {"shape": "openai-chat", "url": "http://127.0.0.1:8095/v1", "key": "local"}}
```

Mainplane wants a `key`, so write anything; XeStrata reads it only when the config has an `api_key` (then that value).
Make a session like this:

```sh
echo '{"model":"xestrata/xestrata","tools":"patch","context_limit":245000,"workers":[{"name":"admin"}],"params":{"max_tokens":16384}}' | mainplane new
```

- Anything goes after the `/` of `model`: XeStrata answers with its own model.
- Keep `context_limit` at or under the context chosen in setup less the answer (`max_tokens`).
  A prompt and `max_tokens` past the context get `max_tokens` cut down, but a prompt past the context alone is refused with a 400 (never truncated), and Mainplane fails that session.
- Mainplane sends the thinking back (`reasoning_content`). XeStrata keeps the earlier turns' thinking in the prompt (it gives Qwen3.6-35B-A3B's template `preserve_thinking` by default, and adds the same switch to Qwen3.5-9B's template, which has none, as it loads it), so each step reads only what it added.
  `"preserve_thinking": false` in a request's `chat_template_kwargs` drops the earlier turns' thinking as the template does.
  llm-jp-4.1's template (the Harmony format) never writes the earlier turns' thinking, so each step reads from where that thinking was on (the earlier call and the tool's result).
- Mainplane sends each session's `prompt_cache_key`. With [parallel requests](BATCHING.md) (`"parallel"`), when no slot holds the prompt's start, the slot the same session used last is taken first, and the other sessions' conversations stay.
- Going back and forth between many long sessions pushes conversations out of the RAM cache; they are [on the disk as well](#keeping-parked-conversations-on-the-disk), so they come back without being read again.
- Mainplane gives up on a response after 300 s without headers or bytes. While a long prompt is read, XeStrata sends an SSE comment every 10 s.

## Web search

XeStrata lets a request that asks for it search the web.
The model calls the `web_search` (search) and `fetch_page` (read a page's text) tools, and XeStrata runs them and returns the results.
`search_engine` chooses the search engine: `searxng` (the default) is a SearXNG-compatible search server, `deepseek` is DeepSeek's search API, `tinyfish` is TinyFish's search API.

**The search server** (`searxng`): anything that answers `GET /search?q=...&format=json`.

- [SearXNG](https://github.com/searxng/searxng): add `json` to `search.formats` in its `settings.yml` (without it, it answers 403).
- A server that returns the pages' text too (`include_raw_content`): XeStrata uses that text, and reads a page itself when there is none.

**DeepSeek's search API** (`deepseek`): DeepSeek's [Search API](https://api-docs.deepseek.com/guides/search), with your own DeepSeek API key.
DeepSeek bills USD 0.75 per 1,000 searches that find something (failures, empty results and timeouts are not billed). The search words go to DeepSeek.
The key is `deepseek_api_key` in settings.json (one for every model, in a file only you can read). The web page's Model settings can only write it and never shows it ("Clear" removes it).

**TinyFish's search API** (`tinyfish`): TinyFish's [Search API](https://docs.tinyfish.ai/search-api), with your own TinyFish API key.
The texts of the pages found come with the same key from TinyFish's [Fetch API](https://docs.tinyfish.ai/fetch-api), in one request for all of them (when that fails, the snippets answer alone).
It is free up to a daily allowance (12,000 searches and 1,000 pages, reset at 00:00 UTC); a search takes as many pages of it as it found (3, 5 or 8 by `search_context_size`). The search words and the addresses of the pages found go to TinyFish.
The key is `tinyfish_api_key` in settings.json, kept as DeepSeek's is.
A query is at most 128 characters; anything longer is cut. The pages' text comes with the search results, and `fetch_page` is read by XeStrata itself.

**Setting it up with setup**: `xestrata --setup` asks once (and at each `--setup`) whether to use web search (`deepseek` for DeepSeek, `tinyfish` for TinyFish, `no`). The default is no, and without an answer nothing is done.
With `deepseek` and `tinyfish` it asks for the API key without showing it (`--web-search deepseek`, `--web-search tinyfish`; the launcher's window's Web search choice and key field).
`--web-search no` takes it out. A `search_url` you wrote yourself is left as it is.

**Setting it yourself**: write the search server's address in `search_url` (for example `http://127.0.0.1:8888`).

- For every model: `"search_url": "http://127.0.0.1:8888"` in `~/.config/xestrata/settings.json` (on Windows `%APPDATA%\xestrata\settings.json`; for DeepSeek `"search_engine": "deepseek"` and `"deepseek_api_key"`, for TinyFish `"search_engine": "tinyfish"` and `"tinyfish_api_key"`).
- For one model: the same key in `xestrata-<model>.json`. `"search_url": ""` turns the search off for that model alone.
- The web page: the About tab's Model settings ([Settings for every model](#using-it)).

It is used from the next start. Without it nothing is searched, and the requests below are answered as before.

**Requests** (only those that ask for it search):

| API | How to ask | The answer |
| --- | --- | --- |
| Chat Completions | `web_search_options` (`search_context_size` `low`, `medium` or `high`: the number of results and how much page text) | `url_citation` in `message.annotations` for the addresses the answer writes |
| Responses | the hosted tool `web_search` in `tools` (or `web_search_preview`; `filters.allowed_domains`, `search_context_size`) | `web_search_call` items (the query and the pages found in `action`), and `url_citation` in `output_text` |
| Anthropic Messages | the server tool `web_search_20250305` (`max_uses`, `allowed_domains`, `blocked_domains`); `web_fetch_20250910` to read pages | `server_tool_use` and `web_search_tool_result` (`web_fetch_tool_result`) blocks, `usage.server_tool_use` |

The web page's chat has a "Search: on / off" switch in its Chat settings; on sends `web_search_options`. The pages found are listed under the answer.

**Safeguards**: `fetch_page` reads public addresses only. It checks the address the name resolves to, connects to the address it checked, and checks every redirect the same way; pages on this PC or its network are not read.
One result holds at most 24,000 characters, a page is read up to 2 MiB and for 15 seconds. When the search server does not answer or finds nothing, the model is told so and the answer goes on.
**The search's language and region** (TinyFish only): the country of the request's `user_location` (`web_search_options.user_location.approximate.country` in Chat Completions, the tool's `user_location.country` in Responses and Anthropic) goes to TinyFish as the region, and TinyFish chooses the language from it.
Without `user_location`, the first language of the request's `Accept-Language` (with its region when it names one) goes. In the web page's chat, that is the browser's language setting.
Without either, nothing goes, and TinyFish searches with its defaults (US, English). The SearXNG-compatible servers and DeepSeek are not told (DeepSeek's language control, `lang:`, keeps only pages in that language).

## Tools from MCP servers

The chat page can give the model tools from [MCP](https://modelcontextprotocol.io) servers, as LM Studio and Claude Desktop do:
reading your files, fetching web pages, searching, anything an MCP server offers.
List the servers in `xestrata-<model>.json` under `"mcp_servers"`.
It has the same shape as Claude Desktop's `mcpServers` block, which you can also paste as it is (key `"mcpServers"`):

```json
"mcp_servers": {
  "files": {"command": "npx", "args": ["-y", "@modelcontextprotocol/server-filesystem", "/home/me/notes"]},
  "search": {"url": "http://127.0.0.1:3000/mcp", "headers": {"Authorization": "Bearer ..."}}
},
"mcp": {"timeout_s": 60, "max_result_chars": 20000, "max_rounds": 8}
```

Or keep them in their own file and start the server with `--mcp-config path/to/claude_desktop_config.json` (a file with an `mcpServers` block);
add it to the `serve/server.py` line of your run script.
Restart XeStrata after a change.

- **A program** (`command`, `args`, optional `env` and `cwd`) is started by XeStrata and spoken to over its stdin/stdout.
  `npx`, `uvx`, `python` and friends are found on `PATH` as usual (Node.js is needed for `npx` servers).
  **An address** (`url`, optional `headers`) uses MCP's Streamable HTTP transport (the older SSE-only transport is not supported).
  `"disabled": true` leaves an entry out.
- The servers start with XeStrata, in the background.
  The server window says what each one offers (`MCP server 'files': 14 tools (...)`), or why it did not start.
  Its tools are then left out and the chat works without them.
  The Monitor tab lists them, and Chat settings has **Use tools from MCP servers** (on by default).
  A server that stops later is started again at its next call.
- In the chat each call shows as a small block (tool, arguments, result).
  The model reads the result and goes on, up to `max_rounds` calls in a row per answer.
  At `max_rounds` it answers once more, with no tools, from the results it already has.
  A tool that fails or takes longer than `timeout_s` (default 60 s) gives the model an `error: ...` result instead of ending the chat.
  Results longer than `max_result_chars` (default 20,000 characters) are cut, with a note, before the model reads them.
  Stop stops a running tool too.
- Only the chat page uses them.
  API clients (omp, Claude Code, the OpenAI and Anthropic SDKs) see the API exactly as before and keep their own tools.
  A request to `/v1/chat/completions` opts in with `"strata_mcp": true` (it then gets `strata_mcp` tool events in the stream).
- Pictures a tool returns (PNG, JPEG, WebP) show in the chat with `"mcp": {"image_results": true}` (off by default).
  They are kept by their SHA-256 in `<config name>.tool-media/` beside the run configuration, and the chat reads them from `/tool-media/<name>`, behind the API key.
  At most four a call, about 18 MB each.
  With a picture reader (`--vision`) the model reads them too; without one it reads the text only, as before.

**Security.** MCP tools run on your PC with your user's rights, and **the model decides when to call them**,
also because of what it reads (a web page or a file can contain instructions).
Give a filesystem server only the folders it needs, prefer read-only tools, and don't add servers you don't trust.
The tools can only be used from the chat page itself (a request with another site's Origin or without a JSON content type is refused).
If XeStrata is reachable from other devices, set an API key.

---

## Skills in the web chat

The chat page takes skills in [Agent Skills](https://agentskills.io)' form (a folder's `SKILL.md`: `name` and `description` at its top, the skill below them).
The model is given their names and descriptions alone, and reads a body with the `load_skill` tool only when a request needs it; a conversation that reads none carries none.

- Chat settings' drawer, **Skills**: add one (written there, or a `SKILL.md`, or a zip that holds one, imported), edit, delete, switch on and off.
  **Use the skills added here** off gives the model none of them.
- They are kept in the server's data folder, `skills/`: the same from every browser (setup writes `"skills_dir"` into the config). Which are on is kept in the same folder's `.state.json`.
- XeStrata's own skills (the image tools', `serve/skills/`) are in the same list and read with the same `load_skill`. They can be switched off, not edited or deleted, and a skill of the same name cannot be added.
- An added skill's body reaches the model marked as not XeStrata's.
- Skills are added, edited, deleted and switched only from the chat page itself (a request from another site's origin, or not JSON, is refused), with the API key where one is set.
- A `/v1/chat/completions` request opts in to the added skills with `"xestrata_skills": true` (the web page sends it while one is on).

---

## Manage XeStrata from your AI assistant (MCP server)

`tools/strata_mcp.py` is an MCP server for Claude Code, Claude Desktop, Cursor, VS Code, Codex and other assistants.
Once it is added, you can ask your assistant "install XeStrata for this PC", "start XeStrata" or "is XeStrata running?".
In Claude Code, add it with:

```bash
claude mcp add xestrata -- python3 /path/to/XeStrata/tools/strata_mcp.py
```

Other clients take the same command (`python3` with the script's path) in their MCP settings, as a stdio server.

It has eight tools: status (the running model, what is installed, the hardware, a recommended size), the model list,
install, start, stop, logs, a speed test, and connection settings for other apps.

Install runs `setup.py` with `--yes --no-start` in the background.
Before it downloads anything, it shows the plan and waits for your OK.
Start and stop work like the run scripts and the server's own unload.
No tool takes a shell command or a free path: every argument is checked against setup's own choices.
The only path (install's `data_dir`) must be the data folder, a folder inside the XeStrata folder,
or a new folder named `XeStrata...` outside the system folders.
The MCP server only ends processes it started itself (it keeps their process id and start time in `.strata-mcp/`).
It uses only Python's standard library, so it works before `.venv` exists.

This is the opposite direction from [Tools from MCP servers](#tools-from-mcp-servers) above, where the model calls *your* MCP tools.

---

## Images

The model has an image encoder:
[`mmproj-Qwen3.8-Flash-Next-BF16.gguf`](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF)
(0.9 GB, a 27-layer ViT plus the projector into the language model).
Ornith-1.5, Qwen3.6-35B-A3B and Qwen3.5-9B use their publishers' mmproj files (0.9 GB;
[other models](#or-models-other-than-qwen38-flash-next-ornith-15-llm-jp-41-qwen36-35b-a3b-qwen35-9b-glm-53-flash)), chosen
and set as this section describes.
It is **optional**, and setup's "Images?" defaults to yes for a model that reads pictures (upstream 91baead75).
To leave it out, say no or pass `--vision no`; to add it later, run setup again with `--vision gpu` (or `--vision cpu`).
Yes (`--vision yes`) means the GPU encoder on a card of 24 GB and more and the CPU encoder on a smaller one: the GPU encoder keeps about 1.4 GB of VRAM free, which cost 3-5% of text speed on 16 GB cards (upstream #2406's measurement). `--vision gpu` keeps the GPU encoder on a small card too.
setup downloads the mmproj and adds the picture reader (`xestrata-vision`: XeStrata's own vision encoder, built with the engine, on the GPU, or on the CPU for a small card and when it cannot start on the GPU) to your start script.
Nothing else changes.

| Encoder on | Speed | Image tokens per picture |
| --- | --- | --- |
| **GPU** (recommended) | 30–45× faster than the CPU encoder | 1,024 (small and wide pictures scaled up to 1,024) |
| CPU | slow | scaled down to about 300 |

With images on, setup passes `--vram-reserve-mib 700` to the engine.
That is the same value as the text-only default, but the engine then no longer lowers the reserve on a small card
([XE.md](XE.md#the-vram-reserve-on-a-small-card)).
The GPU encoder's own VRAM can leave the expert cache smaller and the text a little slower.

**The encoder only while there are pictures** (upstream ca41a6b9): with the expert cache held in pieces (`vram_elastic`), the GPU encoder does not start with the server by default, so the expert cache keeps all of the VRAM. `"elastic": false` in the `"vision"` section starts it with the model, as before.
The first picture's request makes the cache keep `"elastic_mib"` (default 2048) free on top of the reserve and starts the encoder there (a few seconds).
If the cache cannot shrink (the resident low-RAM mode, an engine without `--vram-elastic`, a layer split) the encoder starts anyway and the log says why; on a layer split the server says at start that the encoder's VRAM has to be kept free with `--vram-reserve-mib`.
An encoder that fails to start (out of VRAM) ends the request with a `503`; the next picture tries again.
After `"elastic_idle_s"` (default 120) without a picture it stops between requests and the cache takes the VRAM back.
A `POST /v1/vram` while the encoder runs stacks the reserve on top of the encoder's share; an unload stops the encoder too.
With `"vram_elastic": false` the server says so and starts the encoder with the model, as before.
When the engine says it cannot give pieces back (`vram_elastic` 0, above), the first picture starts the encoder first and loads the model again, and from then on the encoder starts with the model.

The GPU encoder's (XeStrata's own vision encoder) build, its precision, and the order in which the server falls back to the CPU are in [XE.md](XE.md#images).
On the GPU encoder a picture becomes about 1,024 tokens of the context; on the CPU encoder about 300.
The same picture sent again, as chat apps do on every turn, is encoded only once.

**More image tokens:** `--vision-tokens N` at setup (`./setup.sh --setup --vision cpu --vision-tokens 768`) sets the most tokens a picture becomes.
It is `"max_tokens"` in the `"vision"` section of `xestrata-<model>.json`, which you can also edit by hand.
More tokens keep more detail (small text, charts, screenshots) and take longer to encode, on the CPU most of all.
A setup run again keeps the value for the same encoder place (GPU or CPU).

**A minimum of image tokens** (upstream #767): the encoder scales a small picture up to at least `"min_tokens"` tokens (mtmd's `image_min_tokens`).
The default is 1,024 on the GPU encoder and no minimum on the CPU encoder.
With few tokens small text such as a receipt's is misread (an 800x400 receipt at 325 tokens gave a wrong number and total, at 1,024 the right ones; llama.cpp's mtmd recommends at least 1,024 for Qwen-VL models too).
On the GPU 1,024 takes no more VRAM and at most 1.5 s a picture on the B70.
To read small text accurately on the CPU too, write `"min_tokens": 1024` and `"max_tokens": 1024` by hand in the `"vision"` section of `xestrata-<model>.json`.
Encoding then takes 8.9–9.9 s a picture (a Ryzen 7 5700X, 8 threads; 2.4–2.7 s at the default of about 300 tokens).
A setup run again keeps the values.

**A Q8_0 encoder** (upstream #625): `"mmproj"` in the `"vision"` section can point to another mmproj file of this model, for example a Q8_0 one (`llama-quantize mmproj-Qwen3.8-Flash-Next-BF16.gguf mmproj-Qwen3.8-Flash-Next-Q8_0.gguf Q8_0` makes one).
The file is smaller (590 MiB against 865 MiB for the BF16 one), but the encoder converts the weights when it loads them, to FP16 on the GPU and to BF16 on the CPU, so it takes the same memory and time as with the BF16 file.
Setup downloads the BF16 file, and a setup run again keeps a file of your own that still exists.

### Sending a picture

**Terminal chat:** type `/image <path to a picture>`, press Enter, then type your question.

```text
you> /image /home/me/Pictures/receipt.jpg
(picture attached: receipt.jpg - now type your question)
you> What is the total on this receipt?
```

**OpenAI API** (an `image_url` part: a `data:` URL, an `http(s)://` URL or a local file path):

```python
import base64
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:8095/v1", api_key="none")
img = base64.b64encode(open("photo.jpg", "rb").read()).decode()
r = client.chat.completions.create(model="xestrata", messages=[{"role": "user", "content": [
    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img}"}},
    {"type": "text", "text": "What is in this picture?"}]}])
print(r.choices[0].message.content)
```

**Anthropic API:** an `image` block with a `base64` (or `url`) source, as usual.

JPEG, PNG, BMP, GIF, WebP, TIFF and AVIF work (the last ones are converted to PNG first; agents such as omp send WebP).
An `http(s)://` URL that points to the server's own PC or its LAN (a loopback, link-local or private address), or redirects there, is not fetched. To serve pictures from the LAN, start the server with `XESTRATA_IMAGE_URL_PRIVATE=1`.
Chat apps with image upload work the same way.

**How it works inside:** the encoder turns the picture into rows of the same width as the model's word embeddings.
XeStrata puts them where the prompt has `<|image_pad|>` tokens and gives each one its 2-D position (row and column in the picture);
the model uses interleaved M-RoPE.

## Image generation (optional)

How to use it, its options and the API's fields are listed in [IMAGE](IMAGE.md); this section also explains how it works inside.

`./setup.sh --images qwen-image-turbo` sets up image generation with Qwen-Image-2.1-Turbo beside the text model (upstream 0.1.44).
Images are made by XeStrata's own image engine, `xestrata-image`, which setup builds with the text engine: the same SYCL kernels, on every maker's GPU.
The image attention runs through oneDNN on Intel GPUs and aotriton on AMD GPUs (both in the packages), through cuDNN on NVIDIA GPUs where it is installed (contrib-llvm; `xestrata --setup` says how), and through XeStrata's own kernel elsewhere. Where the GPU's math library has int8 products (Intel and NVIDIA in the contrib-llvm mode, AMD), the image models' products take int8 weights and activations.
Setup downloads the image model, its VAE, its text encoder (Qwen3-VL-8B) and, for the edits, its vision encoder (Qwen3-VL's mmproj), and adds an `"images"` section to the config.
The image model is Q8_0 by default. The engine streams from RAM the layers that do not fit in VRAM, so the GPU's memory does not decide the size.
When the RAM has room (8 GiB and the GPU's memory left beside the layer), a streamed layer is copied into page-locked RAM the first time it streams, and copied to the GPU from there after that. A layer the GPU rewrites before use (Qwen-Image's int8, Mage-Flow's FP16) is kept as rewritten, so from the second time it is not rewritten either. Qwen-Image at a 5 GiB budget sampled from the second image in 3.5 s instead of 4.6 on the B70 and 7.4 s instead of 8.1 on the RX 9060 XT, and its text encoder took 0.27 s instead of 0.66 on the B70 (the first image is slower by the time the RAM takes). Where the GPU copies from the file's pages fast (the RX 9060 XT), a layer that is not rewritten is copied from there, not page-locked. When the image engine lets its cache go, it lets this RAM go too. `XESTRATA_RESIDENT_PIN=0` never page-locks it, `1` always does.
Its text encoder is Q8_0 by default too. The RAM decides both: when the four files (the vision encoder for the edits among them) and the engine's own (about 20.2 GB with both at Q8_0) do not fit the RAM the text model leaves, setup first lowers the text encoder (Q6_K, Q5_K, Q4_K), and only when that does not fit either takes the largest image model that fits (Q6_K, Q4_K_M), with the largest text encoder that fits beside it.
With `--image-ssd-offload on` it keeps both sizes chosen and reads what does not fit from the SSD (slower).
With `--images qwen-image-turbo-uncensored` the text encoder is AtomicChat's Qwen3-VL-8B with its refusal direction removed, its size chosen the same way; the image model, the VAE and the vision encoder stay the same. Its authors measured no change in what it can draw, only a small overall shift in the pictures.
When setup lowers it, setup, the server at start and the web page (the Images tab) say so.
With the same seed it makes the image stable-diffusion.cpp makes with the model card's command (within an LPIPS of 0.03).
With `--images qwen-image-base` the images are made by Qwen-Image-2.1, the model Turbo was made from (`qwen-image-base-uncensored` takes AtomicChat's text encoder above). Only the image model is another file: the VAE, the text encoder and the vision encoder are Turbo's.
It runs 40 steps by default (the default of Qwen's diffusers pipeline, without guidance), about 5 times as long an image as Turbo; a request's `steps` (2 to 100) changes them. The steps' sigmas follow diffusers' scheduler, shifted by the image's size.
The weights of Qwen-Image-2.1-Turbo and Qwen-Image-2.1 are under the Qwen Research License, not a free software license: read it before any commercial use.

With `--images mage-flow-turbo` the images are made by Microsoft's Mage-Flow-Turbo (4B, 4 steps). `mage-flow` (the model tuned with reinforcement learning, 20 steps) and `mage-flow-base` (the model before it, 30 steps) take CFG (scale 5 by default). `--images magetrail` makes them with RicemanT's MageTrail, Mage-Flow tuned on Danbooru and e621 images (30 steps, CFG 5, no edits).
setup downloads the image models for generation and for edits (Comfy-Org's bf16 safetensors), the VAE, the text encoder (Qwen3-VL-4B) and the vision encoder for the edits (Qwen3-VL-4B's mmproj). The text encoder's size is chosen as for Qwen-Image-2.1; the image model comes in one size.
When the image model's blocks all fit in VRAM with the text encoder's layers, the engine times multiplying them 16-bit (FP16) as they are and in 8 bits on the GPU and takes the faster (16-bit on a GPU without int8 products); when they do not, it turns them into 8 bits as it loads them and streams the blocks that do not fit from RAM (`--image-dit-q8 on` always 8 bits, `off` always 16-bit). Where the GPU's math library has int8 products (Intel and NVIDIA in the contrib-llvm mode, AMD), 8 bits are the weights rotated by Hadamard matrices in groups of 256 and made int8 with a scale for each row, and the activations rotated alike and made int8 for each token (the int8_convrot form; the time modulations stay Q8_0); elsewhere Q8_0, turned back into 16 bits for each product.
With `-uncensored` the text encoder is huihui-ai's Qwen3-VL-4B with its refusal direction removed (noctrex's GGUF).
The API takes `steps` (each model's count by default), the CFG scale as `true_cfg_scale` or `guidance_scale` (1: no CFG) and `negative_prompt` (one space by default). It has no transparent backgrounds (masks and strength as `/v1/images/edits` below).
Edits (`POST /v1/images/edits`, the Images tab of the web page, the chat's `edit_image`) are made with the edit image model, which the engine switches to. An edit takes up to 3 reference images (as the edit models were trained), in 4 steps by default with Turbo and 30 with the other two. The image model not in use stays in RAM in 8 bits (`--image-dit-keep off` lets it go and reads it from the file again at each switch); a switch takes a few seconds.
Mage-Flow's weights are under MIT, but Microsoft's README says the models are for research and not meant for products or services. XeStrata adds no content filter and no watermark for Mage-Flow.

`--images magetrail`'s MageTrail is Mage-Flow's DiT reshaped to 12 blocks and tuned, each block's modulation split into a projection all the blocks share (3072 to 256, once a step) and a small one of the block's own (bluvoll's mageflow-lowrank-modulation-v1). The engine reads this form as it is and keeps the modulation's weights FP32, as the format asks.
Its VAE is Black Forest Labs' Flux2 VAE, which its author uses. Mage-Flow's latents are in the Flux2 VAE's latent space from the start (Mage-VAE was made to share it), so the DiT's output goes to it as it is. Setup takes the bf16 copy Black Forest Labs puts under Apache 2.0 with FLUX.2-klein-4B (the same VAE in MageTrail's repository is FLUX.2-dev's file, under its non-commercial license).
The Flux2 VAE works in about 4.3 GB at 1024x1024. Where that does not fit in VRAM, the stages up to the deepest one that fits (the middle's attention among them) run on the whole image and the large ones after them in tiles of the latent overlapping by 4 tokens (64 pixels), blended linearly across the overlap; the tiled stages' group norms take each tile's statistics, so the colors may shift slightly.

With `--images hunyuan-image-3-distil` the images are made by Tencent's HunyuanImage 3.0 Instruct-Distil (an MoE model, 80B, 8 steps, guidance 2.5). It has no text encoder: its DiT reads the prompt's ids itself, so the server builds them with the model's own `tokenizer.json` (a byte-level BPE): `<|startoftext|>`, the system prompt (`en_unified.txt`'s 5,674 characters, which setup reads out of `system_prompt.py`), `User:` and the prompt, `Assistant:` (a space after the colon in both), `<answer>`, `<boi>`, `<img_size_1024>`, `<img_ratio_i>`, `<timestep>`, `<guidance>`, `<timestep_r>`, `<img>` (one for each 16x16 pixels) for the image block, `<eoi>`. The engine reads those ids as they are, so running `xestrata-image` by hand takes the same shape of sequence in `--ids`.
The size snaps to the model's own 37 resolution buckets (built from 1024, walked 64 pixels a step, every side rounded to 16 pixels): the one whose shape is nearest the one asked for. The table has no square below 1024, so 512x512 becomes 1024x1024.
With the size `auto`, or none, the model chooses it: the sequence up to the generated image's block's `<boi> <img_size_1024>` goes through the 32 layers as text, and its last row through the final norm and the output rows gives the logits of the 37 aspect-ratio tokens, of which the largest is taken (the reference implementation's `img_ratio`, greedy). Those weights are only in the bf16 shards, so setup takes them out by range requests (311 KB). Against Tencent's own modules run on the CPU in FP32, the engine chose the same bucket for three sequences.
The guidance is one token given to the time embedding, not two passes as CFG is (the distillation folded CFG into the model); the steps' sigmas the engine builds from the step count (shift 3.0). The way Tencent runs it by default (`bot_task: think_recaption`: the model generates up to 2048 tokens of reasoning and a rewritten prompt before it draws) is not used: it draws from the prompt at once.
The weights are Tencent's two quantizations (w4a8, 46.9 GB, and int8_convrot, 81.8 GB; Tencent publishes no bf16). w4a8 is a single checkpoint, a step about 2.5 times as fast as int8's, which comes nearer the bf16 weights' images.
Edits (up to 3 reference images) are the model's own way: a reference is cut to the bucket whose shape is nearest (the same values as PIL's LANCZOS), made a latent by the VAE's encoder (a sample of the posterior with the seed, as the reference implementation draws), and made features by SigLIP2 (the model's own `vision_model`, read from its last bf16 shard; naflex, up to 1024 patches) and the aligner into 4096 dimensions, and stands before the prompt as the reference's block (`<boi> <img_size> <img_ratio> <timestep>`, the VAE grid's `<img>` tokens, `<joint_img_sep>`, 1024 `<img>` tokens, `<eoi>`). Inside the block the attention is full, and the VAE grid and the SigLIP2 grid each have 2D rope positions. These rows do not change between a request's steps, so they are computed once with the text rows. Masks and strength work as the other models', on the latent [32][th][tw] kept on the host (the image through the VAE's encoder at the reference's crop, the posterior's mean).
The engine keeps the four dense matrices, the VAE and the entry and exit (about 6.7 GB) in VRAM as FP16; the 32 layers' expert banks (1.36 GB a layer in w4a8, 43.6 GB in all) stay quantized, the first layers the budget holds resident and the rest streamed from the RAM through two slots at every step, each expert turned back to FP16 on the device as it is multiplied. The router's softmax and top-k run on the device as well (in FP32, ranking as the reference does), and where the VAE's decode does not fit in the VRAM it is tiled as the reference tiles it.
On the B70's w4a8 a step of the sampling takes 3.7 s at 256x256 (16x16 tokens) and 6.1 s at 1024x1024 (64x64 tokens). The entry and exit (patch_embed and final_layer) take 0.074 s of a step, moved onto the GPU from the host's FP32, which was 30 s a step at 16x16 tokens.
Against Tencent's own modules run on the CPU in FP32, from the same initial latent and 8 steps, the B70's w4a8 comes to a PSNR of 35.67 dB (2.83/255 on average) and the RTX 4070's contrib-llvm build to 33.97 dB (3.29/255), XeStrata's bar for that comparison being 30 dB.
HunyuanImage 3.0's weights are under the Tencent Hunyuan Community License Agreement, which does not apply in the European Union, the United Kingdom or South Korea ([Tencent's license](https://github.com/Tencent-Hunyuan/HunyuanImage-3.0/blob/main/LICENSE)).

The server makes images on `POST /v1/images/generations` (OpenAI's images API, plus `seed` and `steps`), and the web page gets an Images tab.
The image API is vLLM-Omni compatible (OpenAI compatible), vLLM-Omni's extra fields included: `num_inference_steps` is `steps` (Turbo keeps its 8 steps and ignores a number of steps asked for); `true_cfg_scale` above 1 with a `negative_prompt` puts classifier-free guidance on Qwen-Image-2.1 and Turbo (diffusers' Qwen-Image-2.1 formula; with the prompt's and the negative prompt's both computed, it takes about twice as long at 1024x1024 and 1.5 times at 512x512). Without either, there is no guidance, and `guidance_scale` has no use with Qwen-Image-2.1 and Turbo.
`output_format` is `png` (the default), `jpeg`, `webp` or `jxl`, with `output_compression` (0 to 100, default 100) the quality jpeg, webp and jxl are written at (jxl's 100 is lossless). `jxl` (JPEG XL) is there only with libjxl's cjxl (BSD-3-Clause), and `/health`'s `image_defaults` lists it in `formats`. The deb and rpm packages recommend it (Ubuntu's libjxl-tools, Fedora's libjxl-utils), and the Podman image and the Windows 7z carry it; run from the source, install that package. Lossless images are written at cjxl's effort 3, the others at its default 7. A JPEG XL input (a reference image, img2img's, a tool's, or a picture a chat's tool made before) is read as PNG by djxl, of the same packages; without djxl, that input alone is refused with the reason.
`n` goes up to `max_n` (default 16); a number whose time, estimated from the speed the server measured (a step's seconds and the rest's, for each size; above the largest size measured a step's time grows as the pixels to the power 1.5), is over `budget_s` (default 600 seconds) is refused (400) with the estimate. Before the server has made an image, 4 at most. While several images are made, a request waiting for the GPU's turn gets it after each image (not under `unload`). `quality` sets Qwen-Image-2.1's steps (`low` 20, `medium` and `standard` 30, `high`, `hd`, `xhigh`, `max` and `auto` the default 40; `steps`, when given, wins); Turbo keeps its 8, and Mage-Flow does not use it. The answer's `usage` counts the image engine's tokens: the prompt's, and one for each 16x16 pixels of the reference images and of the images made.
`background` `transparent` wraps the prompt in the form Qwen recommends for it and makes an image with a transparent background (a PNG, webp or jxl with alpha; not with jpeg); `opaque` flattens the alpha on white, and `auto` (the default) keeps what the model made. An edit's reference images reach the image model with their alpha.
With `stream: true` the answer comes as SSE, as OpenAI's does. `partial_images` (0 to 3, default 0) makes the sampling's predicted result (x0 = x − σ v) into an image with the VAE and sends up to that many of them an image as `image_generation.partial_image` events (`image_edit.partial_image` for an edit; none after the last step, so fewer with few steps). Then each image comes as an `image_generation.completed` event (`b64_json` and `usage`, plus its `seed` and the `xestrata-engine` figures). An event's `size` is the image's, the model's choice for `size: auto` included. A multipart edit's numbers (`n`, `seed`, `steps`, `num_inference_steps`, `output_compression`, `partial_images`, `guidance_scale`, `true_cfg_scale`) and `stream` are read as numbers and a boolean, as a JSON request's are. The noise is drawn once at the start, so the partial images do not change the final one; they add the VAE's time. HunyuanImage 3.0 makes no partial images and sends the final image alone. A request without `stream` ignores `partial_images`. The web page sends `partial_images: 2` with "Show partial images" (off by default).
`POST /v1/images/edits` (OpenAI's multipart form with `image` or `image[]`, or JSON with `images: [{"image_url": ...}]`) makes an image from reference images (up to 10; PNG, JPEG, GIF and BMP as they are, other formats such as WebP turned into PNG first); with a `mask` (as OpenAI's: a PNG with an alpha channel, the first image's size) only its transparent part is drawn again and the rest stays the first image's pixels (each step puts the latent outside the mask back to the image's, and the end blends in its pixels). Each reference image is read at about 1024x1024 pixels in its own shape, whatever the size made (as diffusers' Qwen-Image-2.1; Mage-Flow's picture reader reads them with the long side at 384 pixels or less, its VAE at the size made). More reference images take longer and more memory (on the B70 at 1024x1024, about 46 s with 3 and 4 minutes with 10): the K and V cache of their tokens (about 2 GB for one at 1024x1024) goes, as far as it does not fit in VRAM, to pinned RAM and is copied in at each step, and an image for which even that leaves the VRAM or the RAM short is refused (400) before it starts. An edit can change them, or take the people and things in them as its subject. `size` `auto` takes the first reference's shape at about 1024x1024 pixels. On the web page, an image attached in the Images tab makes an edit.
Every model takes a mask and a `strength` (above 0 and at most 1; `src/image/img_start.hpp`). As diffusers' img2img, a strength skips the first floor(n (1 − strength)) of an n-step schedule and starts at the next sigma s from the image's latent z0 and the seed's noise, (1 − s) z0 + s noise (strength 1 is the generation or the edit as before); the server's Python and the engine's C++ count the skipped steps in doubles, so they agree. The image is the first reference image with a model that edits, and the request's one image with a model that does not (MageTrail, Mage-Flow without its edit model: img2img, its strength by default 0.87 (no more than 1 − 1/steps), and 1 with a mask alone), made the posterior's mean latent by the VAE's encoder (the Flux2 VAE's for MageTrail).
A conversation can make images too: `/v1/chat/completions` with `"xestrata_images": true` (the web page sends it) gives the model the tools `generate_image` and `edit_image`, and when it calls one the server makes the image and shows it there; `edit_image` changes the conversation's last image (one attached or one made in it). With a model that edits, the pictures attached to the user's last message go with it as reference images after the image edited (image 1: they are image 2, 3 and on), so an edit can take from them ("the face from image 2"). With several image models, the tools take a `model` (the installed models' names, with what each is good for), and the model chooses the one the user's words ask for (none: the default; `edit_image` those that edit first, img2img with one that does not, and a `strength` too). `"xestrata_images": {"model": "<name>"}` makes the tools use that model alone; with `"force": true` as well (the web page, while its "Image" switch left of the send button is on), the reply opens a call of one of the image tools (after its thinking, if it thinks; the name kept to the image tools by a grammar where the engine takes one; the request's own `tools` stay available; not together with a `tool_choice`); the web page sets it with "Images" under the message box. With `"edit": {"image": <a data: URL or a /v1/images/files/ link>, "mask": {"image_url": <a data: URL>}}`, `edit_image` changes that image where the mask is transparent and nowhere else (the mask is fitted to the image's size); with `"layout": {"size": "<width>x<height>", "items": [{"label", "caption", "text", "box": [x0, y0, x1, y1]}]}` (`box` in shares of the sides), the boxes become MOSS-Image's layout prompt (its JSON form): with them alone, the tool is `generate_image` alone, which makes a new image by them at that size (the picture they were drawn on is the canvas' background only and does not go into the image); with a mask too, `edit_image` redraws the painted part by them. The image model is a MOSS-Image one whether the choice was auto or another model, and without a MOSS-Image model the request gets a 400. On the web page they are drawn in the preview that a click on an attached picture or an image made in the chat opens ("Mask" and "Boxes"), and "Use in the next message" puts them on the next message. On `/v1/responses`, OpenAI's hosted `image_generation` tool does the same, and the image comes back as an `image_generation_call` item (its `result` the image in base64, with its `size`, `quality`, `background`, `output_format` and `revised_prompt`). The tool's `size`, `quality`, `output_format`, `output_compression` and `background` are used for the images it makes (`size` `auto`: the size the model asks for). Its `model`, when it names one of the installed image models, makes the tool use that one alone (OpenAI's `gpt-image-1` and the like: the default model, or the one the words choose). Its `input_image_mask` (`image_url` a data: or http(s) URL; no `file_id`) is the mask of its edits (the shape of `/v1/images/edits`' `mask`). In a `stream` request, the tool's `partial_images` (1 to 3) sends partial images as `response.image_generation_call.partial_image` events (`partial_image_b64` and `partial_image_index`). `moderation` and `input_fidelity` are accepted and ignored.
Each side is 256 to 2048 pixels. A size whose sides are not multiples of 16 (1920x1080) is made at the sides rounded up (1920x1088) and its middle cut out; the web page says so.
The image engine starts with the first image and stays; it is a process apart from the server, stays when the text model is changed, and ends 30 minutes (the config's `daemon_idle_s`) after the last server leaves: its kernels are prepared once, and the same prompt's text states and the layers it holds in VRAM are kept for the next image.
When the `"images"` section's `"gpu_pci"` (setup's `--image-gpu`) names a GPU that none of the server's text engines takes (the server's GPU, a layer split's, the replicas' and a split serve's cards), the images are made on it. Its jobs take that GPU's turn alone and not the service's queue, so the chat is answered meanwhile and nothing is done to the text model (the policy `separate`); only while MOSS-Image's text encoder runs on the chat's engine does a job take the queue. With a helper GPU (`--expert-cache-device1..3` names it by its device number) or an environment variable that narrows the GPUs (`ONEAPI_DEVICE_SELECTOR`, `ZE_AFFINITY_MASK`, `HIP_VISIBLE_DEVICES`), the text model's GPU is used. When it names the second of the text model's two GPUs, the engine gets `--split-roles helper` and makes that GPU a helper cache or leaves it; a helper is closed with `HELPER off` before each image job and opened again with `HELPER on` after it ([IMAGE](IMAGE.md)).
Image jobs run one at a time on the GPU. Before a job the engine says what the image needs in VRAM (the least, and the DiT whole with the VAE in one piece), and what the text model does meanwhile is `--image-gpu-policy`: `keep` (stays as it is), `shrink` (gives VRAM back through `--vram-elastic`), `unload` (its conversation saved, unloaded, restored at the next start), or `auto` (the default: `keep` when everything fits, else `shrink` when the engine can, else `keep` with the layers streamed when the least fits, else `unload`).
Under `unload`, a single engine whose images run on its own GPU gets a shared arena from the server when there is an image model (`--shared-expert-arena`; on Linux a memfd, on Windows a page-file-backed `section:<name>`), and its experts stay there while it is off. When the RAM the image needs (`host_mib`, the image model's files, 2 GiB, `--ram-reserve-mib`) is short even with what the unload frees, the server sends the engine `ARENA DROP <MiB> <file>`, and the engine marks that much of its experts dropped in the arena's header, from the end of the `--expert-profile` ranking. Once the engine has ended the server gives their pages back (a punched hole on Linux, `MEM_RESET` on Windows), and the next start reads those alone. Any other unload gives the whole arena back. It is not given with `--mmap-experts`, the RAM tier or `"keep_arena": false`.
After a job the image engine keeps its layers only under `keep` or `separate` with VRAM to spare; otherwise it lets go of what it holds (idle, the process alone, about 200 MiB).
`xestrata images --config <config>` (on Windows `xestrata-cli images`, from the source `tools/image/install.py`) adds it to an existing install.

With several image models installed (`--images qwen-image-turbo,magetrail`), the config's `"images"` is the default model and `"image_models"` the others, chosen by a request's `model` (on the web page, by its choice of model); no name, or an unknown one, is the default model. The chat's image tools use the default model.
On one GPU, image jobs run one at a time, whichever server they come to: a job holds the GPU's lock file in the temp folder (the GPU the engine's environment names) until it ends, and another server's job waits for it. The job that has the turn reads the list of the GPU's image engines (in the same folder) and ends every one but its own model's first (waiting until it has given its VRAM back); a server whose engine was ended starts it again with its next job. Servers of the same model share one engine. Only the image engines are counted: another server's text engine's VRAM is not.
The web page's Images tab shows each model's defaults (`/health`'s `image_models`) in its fields, and sends a negative prompt, the steps, the CFG strength, a transparent background, a mask (painted over the first image) and a strength; an image attached is img2img with a model without edits.

### The image engine's LoRA adapters

The image engine not only reads the LoRAs that are distributed but applies them to the DiT's products with each image (`include/strata/image/image_lora.hpp`, `src/image/image_lora.cpp`). Qwen-Image-2.1 (Turbo and the base model), MOSS-Image, Mage-Flow and MageTrail take them. HunyuanImage 3.0's DiT has no adapter path and refuses a `LORA:` word.

- **Reading:** only a safetensors' header JSON and a GGUF's metadata and tensor infos are read. A key has its prefix (`model.diffusion_model.`, `diffusion_model.`, `transformer.`, `lora_unet_`, `lycoris_`) and its role's suffix (`.lora_A`, `.lora_B`, `.lora_down`, `.lora_up`, `.lokr_w1`, `.lokr_w1_a`, `.hada_w1_a`, `.oft_blocks`, `.a1`, `.diff`, `.set_weight`, `.w_norm`, `.diff_b`, `.alpha`, `.dora_scale` and the like) taken off, longest first, and is the model's weight's name. sd-scripts' `lora_unet_` and the LyCORIS form's `lycoris_` write that name with every dot an underscore, so neither side rewrites the file's underscores blindly: the listing matches the name against the model's own names (`serve/loras.py`'s `_KOHYA` and `kohya_name`, from the same TARGETS table) and the engine does the same against its target slots (`image_lora.cpp`'s `lora_key_names`). So a name holding underscores of its own - `attn.to_q`, `img_mlp.net.0.proj`, `proj.ffn_up_gate` - reads in both spellings.
- **Forms:** `kind_of` tells lora, lokr, loha, glora, oft and diff apart from the roles. A file with two forms on one weight is refused when read (a loader that keeps one patch a key would drop one without a word). A file with a text encoder's key (`lora_te_`, `text_encoder.`, `te.`) is refused when bound.
- **Strength:** alpha is `<key>.alpha`'s tensor, else the metadata's `lora_alpha`, else 1.0; a weight's strength is `the request's strength × alpha / rank`. A file with `use_rslora` is refused (alpha / √rank is another formula). PEFT's `rank_pattern` is not read: alpha alone. A strength of 0 binds nothing, and a set of all zeros runs the path with no adapter (the image is the same byte for byte).
- **Binding:** a new set binds the DiT's weights again (`QiDit::set_loras`, `MossDit::set_loras`, `MfDit::set_loras`). Each weight `W` has a `LoraLayer` and adds its parts after its product (`product()`: the caller is unchanged).
- **The forms' paths:**
   - LoRA, LoCon: A and B in FP16, `B (A x)` computed from `x16` and added; alpha folds into the part's strength.
   - LoKr: `u = x16 · w2^T`, then the `KronRows` kernel adds w1's rows (a fused weight's row range is cut by that mapping).
   - DoRA: DoRA's weight decomposition, `c = dora_scale / (|W|'s rows + eps)`, as the base product's rows `1 + s (c − 1)` and the delta's rows `c`. The row norms are taken on the host from the file's weights, and only when a DoRA is bound (`dora_norms`; Qwen-Image-2.1's block formats other than Q8_0 are turned back to FP32 first by the GPU's dequantization the products use). An alpha of 0 still applies the base's row factors. The column form is refused.
   - A whole difference: the difference in FP16, added to the product. `.set_weight` scales the base's rows by `1 − s` and interpolates (at a strength of 1 the same as the difference set as it is).
   - LoHa: the delta is made on the host when bound (64-row tiles, threads, FP32 sums) and added in FP16. The file holds the factors only, so the delta is not kept in the `LoraFile`.
   - A bias's delta: added at the place the product adds the weight's own bias (`add_bias`, after the row scale `rs` and a DoRA's `c`); alpha is not applied, only the request's strength. A weight that adds no bias refuses it.
   - OFT, BOFT: the product's output is turned (`rotate`): `Q = blocks − blocks^T`, `Q`'s Frobenius norm capped by the file's alpha, `r = (I + Q)(I − Q)^−1`. An OFT is kept transposed (the product takes r^T W) and a BOFT crosses rows stage by stage; the strength interpolates once.
   - GLoRA: `b1 b2` goes through the delta's path as a plain LoRA's pair, and `W (a1 a2 x)` is computed by the weight's product once more (`add_glora`, a band of 128 tokens, `kLoraGloraRows`). The four tensors keep the file's orientation, and a square weight's rows and columns choose between the two readings (`lora_glora_for_weight`); the strength and the row scale fold into the input `v`.
- **Quantization and the host's copies:** in a W8A8 layer the weight's product stays int8, and `defer` is cut for the products an adapter takes (an int32 sum takes no float). A MOSS-Image block an adapter takes makes its input as 16-bit rows (`rows16`), which the delta's input needs, not by the shortcut that makes the int8 rows alone. Q8_0's weights are as before, and the host's carried weights (`take_host`/`reuse`) are used as they are.
- **Budget:** A/B, a DoRA's coefficients, a LoHa's delta, an OFT's rotation and a GLoRA's band buffers count in the VRAM's fixed part (`lora_bytes()`, `lora_gpu_bytes()`). `use_loras` binds before `plan()`, and `need()` counts `lora_gpu_bytes(loras, T_act)` since it runs before the bind. An image whose budget they do not fit is refused with the VRAM it needs.
- **Caches:** a new set makes Qwen-Image-2.1 drop and rebuild the prefix's cache (the text's rows and K and V), and MOSS-Image drop its context (the rows the text embeddings went through); the same set keeps them. Mage-Flow makes its context with each request, so it has none to drop.
- **The engine's protocol:** the server gives `LORA:<strength>:<path>` on its IMG and NEED lines, repeated (a path with spaces and the like is `%XX`; `xestrata-image` used directly is the same). An adapter's file is read once in the process (`Loras`) and kept through `RELEASE ALL` (only the VRAM is given back, and the next request binds it again from the host's copy). With the same set, `set_loras` does nothing.
- **serve:** the listing (`serve/loras.py`) writes the DiT's target tables a second time (`TARGETS`, `_BIASED`), lists only the files and forms the model applies, and puts the others in `skipped` with their reasons. `TARGETS` against the C++ `build_lora_targets` is checked name for name on the real Qwen (264 to 264). A request's `lora` field and a prompt's tags resolve their names here; a tag's unknown name is taken out and warned of. A file the engine refuses when binding (rsLoRA, MageTrail's low-rank modulation, a text encoder's weight and the like) fails before an image is made, with the key and the reason.
- **How it was checked:** on a small synthetic model, one step with a LoRA applied against one step of the same model with ΔW baked in gives a cosine of 0.9999 or more (Qwen-Image-2.1 0.9999999, MOSS-Image 0.9999709, Mage-Flow 0.9999955; in 8 bits 0.9999076 or more; the small configuration without XMX comes lower from the delta's FP16 rounding, and is checked by the bytes and the budget). Each form's formula matches the form's weight, built in torch from the file's tensors and applied to the base weight, within rel 2.3e-6 at a strength of 1 (OFT and `set_weight` interpolate once, so between 0 and 1 they differ from scaling the built weight's change). On the real Mage-Flow's two DiTs, each form's random adapter changes the image at a strength of 1, lies between at 0.5, and gives the bytes of no adapter at 0. There are also the bytes of an image with no adapter, the engine's end to end, the API and the web page's tests.
- **Limits:** text encoder LoRAs are not applied. An OFT on Qwen-Image-2.1's `modulation.1` is refused (the DiT writes that weight `C` rows at a time, and a turn turns a part's rows whole).

---

## Translation (optional)

`./setup.sh --translate auto` sets up a translation model apart from the text model (Preferred Networks' PLaMo 3 Translate: the 2B or the 0.8B by the RAM, the 8B with `--translate 8b`).
The server translates through DeepL's API (`/v2/translate`), LibreTranslate's API (`/translate`, `/detect`) and the OpenAI-compatible API with the translation model's name as `model`, and the web page shows a Translate tab.
The translation engine starts with the first translation and shares each GPU's VRAM ledger with the image engine (`--translate-gpu` chooses where it runs).
Choosing a model, where it runs and the APIs' fields: [TRANSLATE.md](TRANSLATE.md).

---

## Experimental speed projection (EXPERIMENTAL, off by default)

**This is an experiment, not a finished feature.**
It ships with XeStrata but stays off unless you turn it on.

A 480 KB control vector for Qwen3.8-Flash-Next (`third_party/nonfree/experimental-speed-projection/`, see its README).
It is under the Qwen Community License, so it is kept apart from the rest, and XeStrata runs without it.
After each of layers 4–44 the engine removes one direction from every hyper-connection stream of the residual:
`h -= (h . v) v`, one unit vector `v` per layer, exactly as llama.cpp does with the package's `--cvec-mode project` patches.

**What it changes.** The vector's own package describes it as a **refusal-direction projection**:
with it the model declines far fewer requests (it reports 1 of 50 vs 50 of 50 on its test set).
Removing refusals removes a safety behaviour: you are responsible for what the model writes with it on.
It also shifts ordinary answers a little (measured below).
It is not an optimization in the engine: on the same text it costs 0.2–0.4% per token.
What a chat's tokens/s does with it on depends on the text the model writes (length, repetition, how well the drafts land),
so measure it on your own prompts; the Monitor marks every request ESP or stock.

**Turning it on (at setup).** `./setup.sh --setup` asks "Turn on the experimental speed projection?" (default: no),
or pass `--experimental-speed-projection on` (`off`, or a path to another vector GGUF).
When it is chosen, setup recommends SC117's and huihui-ai's abliterated versions instead, which have the refusals removed in their weights ([above](#or-sc117s-abliterated-versions)), and asks whether to use the projection anyway (no by default); with `--experimental-speed-projection on --yes` it is turned on without asking.
The deb and rpm packages do not carry the vector: with it turned on, setup downloads the vector and its license from the installed version's tag on GitHub into `experimental-speed-projection/` in the data folder.
Only for the original Qwen3.8-Flash-Next (the Coder and Unsloth's files included), not Swift 1.5 or huihui-ai's and SC117's abliterated copies.
It writes these engine flags (llama.cpp's) into `xestrata-<model>.json`:

```text
--control-vector-scaled <XeStrata>/third_party/nonfree/experimental-speed-projection/Qwen3.8-Flash-Next-experimental-speed-projection.gguf:1.0
--control-vector-layer-range 4 44 --cvec-mode project --cvec-dir per-layer
```

The engine log then says `control vector mode = project, dir = per-layer, layers 4..44 (41 steered)`, and the web app's About tab lists it.
(`--cvec-mode add` is llama.cpp's stock additive mode, for additive vectors.)

**Per request.** A loaded vector is on for every request unless it says otherwise.
The web app's Chat settings has a switch, and the API takes `"experimental_speed_projection": false` in the request body (OpenAI and Anthropic);
a config default goes in `"sampling": {"experimental_speed_projection": false}`.
Switching drops the conversation cache once, since the model state was computed the other way.
Switched off, the output is token-for-token the stock model's.

**Measured upstream** (CUDA; Q2_0, fixed experts, 2,557 teacher-forced tokens of code, a document and a chat):
the top-1 token changes at 10% of positions, the mean KL from the stock model is 0.063 nats (max 4.1),
and perplexity is +15% on code, +2.3% on the document and +0.4% on the chat.
Details: upstream's `bench/results/2026-09-27-esp/`.
On the B70, `cvec_parity` checks the control-vector kernel ([XE.md](XE.md#validation-inventory)).

---

## Troubleshooting

| Symptom | What to do |
| --- | --- |
| `no Intel GPU on the xe or i915 driver, no AMD GPU on amdgpu and no NVIDIA GPU on NVIDIA's driver found` | The Intel GPU is not on the xe or i915 driver, the AMD GPU not on amdgpu, and the NVIDIA GPU not on NVIDIA's; check the driver with `lspci -k`. For a discrete card, enable Above 4G Decoding and Re-Size BAR and disable CSM in the BIOS. |
| `Resizable BAR looks off` or a warning that CPU-visible VRAM is small | Copies to the GPU can be slower. On AMD GPUs, work can also appear stalled for a long time, and LoRA training can fail. Enable Above 4G Decoding and Resizable BAR in the BIOS. In the observed apparent stall, it is not known whether the small window was the cause. |
| `an NVIDIA GPU needs the contrib build` | `--license free` is chosen. NVIDIA GPUs run in the contrib-llvm build only: run setup again with `--license contrib`. |
| `NVIDIA's CUDA toolkit is missing` | The CUDA toolkit that makes the NVIDIA GPU's code is not installed. Install it and run setup again (Ubuntu: `sudo apt install nvidia-cuda-toolkit`). |
| `... hands the Intel GPU's dense matrix products to oneMKL, which is not installed` | The contrib-llvm build hands the Intel GPU's dense matrix products to oneMKL, which is missing. Install it ([BUILD.md](BUILD.md#packages)), or run setup again with `--license free`. |
| `no access to the GPU` | Your user cannot open `/dev/dri/renderD*`: `sudo usermod -aG render $USER`, then log in again. |
| `no access to the GPU's compute device (/dev/kfd)` | Your user cannot open `/dev/kfd`, which AMD GPUs compute through: `sudo usermod -aG render $USER`, then log in again. |
| `... the engine needs an RDNA2 or later AMD GPU` | AMD GPUs older than RDNA2 do not run it. |
| `... rocBLAS ... has no code for gfx...`, `... has no code for gfx...` | The distribution's rocBLAS (in a package, its engine) has no code for that GPU. On Ubuntu 26.04, it may run on Fedora 44 ([README](../README.md#amd-gpus)). |
| `building for the AMD GPU needs ROCm's ...` | ROCm's packages that build the AMD GPU's code are missing. Install the ones named and run setup again (see also `./setup.sh --packages`). |
| `... has no HIP target (AMD GPUs)` | intel/llvm was built before ROCm was installed. Let setup build it again when it asks, or run `python3 tools/intel_llvm_build.py --rebuild`. |
| `no progress for 60 s ... reading the prompt` on Linux, and the stall report says `threads in uninterruptible sleep (state D): 16 ...` (in `blk_io_schedule` and the like: waits for the drive) | The engine waits for the drive, not a deadlock: the n-gram table is read at random (`--ple-io direct`), which a rotational disk cannot keep up with (upstream #605). The engine warns at start when the table is on one; `--ple-io ram` (it needs RAM for the table; `./setup.sh --ple-io ram`, in the packages `xestrata --ple-io ram`, remembers it) or the model on an SSD fixes it. Setup adds `--ple-io ram` itself on a rotational disk when the RAM holds the table (on Linux on an SSD too, when the RAM holds the model, the table and 16 GB more). |
| `the SYCL runtime of ... lists no GPU` | The GPU's Level Zero driver (`libze-intel-gpu1`, Intel's compute-runtime) is missing or too old for it; install a newer one ([BUILD.md](BUILD.md#packages)). |
| `... gives this GPU no XMX` | The compiler gives the GPU no XMX; choose from setup's options. It runs without XMX, with prefill taking about 1.6 times as long ([BUILD.md](BUILD.md#the-sycl-compiler)). |
| `no SYCL compiler for the engine` | There is no SYCL compiler: install the distribution's DPC++, or pass `--intel-llvm-build`. |
| `alloc_device: ... refused` at start | A driver state that refuses device allocations while the VRAM is free (Intel's [compute-runtime #916](https://github.com/intel/compute-runtime/issues/916); compute-runtime 26.22 works around an xe kernel bug fixed in Linux 7.1 - on XeStrata's side, [#34](https://github.com/MistVVK/XeStrata/issues/34). On the B70 it happened 3 times in about 120 starts). The engine retries the allocation in pieces and starts with what it gets. While the card is in that state, only a reboot brings it back. On Ubuntu 24.04/26.04, update the driver from the [graphics PPA](BUILD.md#packages) ([XE.md](XE.md#the-output-heads-vram-refusal)). |
| setup, the server or the web page says "lacks the fix of the xe driver's CVE-2026-68264" | The running kernel's xe driver lacks the fix of CVE-2026-68264 (drm/xe/pt: Reset current_op in xe_pt_update_ops_init()): a GPU memory bind that is retried can write past its array and crash the driver. When an Intel GPU runs on the xe driver, Ubuntu's kernel build is compared with the fixed ones on [Ubuntu's CVE page](https://ubuntu.com/security/CVE-2026-68264) (7.0.0-38.38 and later for 26.04's `linux` and 24.04's `linux-hwe-7.0`), Debian asks `debsecan` and Arch `arch-audit` (other distributions are not checked). Update the kernel and reboot. A container (Podman, incus) runs on the host's kernel, so update the host's. |
| The start log says `IGC's GEP simplification is off` | Intel Graphics Compiler (IGC) 2.27 to 2.30 (Ubuntu 26.04's 2.28.4 among them) miscompile some reads for Xe2 and later GPUs (the Arc B-series, Lunar Lake, Panther Lake): the GPU reads past its buffer and stops, or the answers are wrong. On Linux, when the installed IGC is one of those and there is an Xe2 or later GPU, the engine sets `IGC_EnableGEPSimplification=0` when it starts, which turns that optimization off (on the B70 the decode of a 1K prompt is 5 to 10% slower, 32K the same). A value you set yourself, `IGC_EnableGEPSimplification=1`, is kept. For full speed, use Fedora 44 (its IGC is 2.32 or later), or [install with Podman](../README.md#podman) (Fedora 44 inside). On Ubuntu 24.04/26.04 there is also the way of moving to the HWE kernel with the [graphics PPA](BUILD.md#packages) (IGC alone cannot be updated: the PPA installs the driver with it, and that driver is for the HWE kernel). |
| setup says `the IOMMU translates the GPU's memory accesses` | The IOMMU translates the GPU's memory accesses. Each start shows tens of GB of RAM to the GPU, and the kernel then allocates an I/O address for every page: a start soon after the engine stopped can copy to the GPU tens of times slower (`pre-filled ... MB/s` in the start log). Starting Linux with `iommu=pt` avoids it (the PC's own devices reach memory directly; devices passed to a virtual machine keep the IOMMU). Run the commands setup shows and restart the PC. |
| setup says `memory compaction may move the expert arena under the Intel GPU` | Memory compaction can move pages of the expert arena registered with the Intel GPU, and each time the GPU's driver maps the whole arena again. With the IOMMU translating, the GPU's copies can stall or fail with a copy-engine error (`Engine memory CAT error`, #42). The engine locks the arena in RAM, so with `vm.compact_unevictable_allowed` at 0 compaction leaves it alone. Run the commands setup shows (no restart needed). The memlock limit has to allow the arena too. |
| Python or the build tools could not be installed | Install what it names, then run it again. Everything already done is kept and skipped. |
| `port 8095 is already in use` | XeStrata is already running (look for its window), or another program uses the port: `./setup.sh --port 8081`. |
| The first start takes minutes | It is reading 23–50 GB into RAM; the second start is faster while the files are in the OS cache. |
| The PC freezes for a few minutes at the start | Normal, most of all the first time (the server window says when it happens): the engine loads the experts into RAM, registers part of it for the GPU and sizes the expert cache. Wait; don't close the window. Still frozen after 10 minutes: restart the PC, close other programs, try again, or pick a smaller size. |
| The start takes many minutes while the kernel is busy compacting memory (`kcompactd`) | With transparent huge pages on `madvise`, the engine asks for `MADV_HUGEPAGE` on its arena, and on fragmented RAM every fault can then compact memory (upstream #771). `XESTRATA_NO_ARENA_THP=1` stops the request (with THP on `always` the engine does not make it at all). |
| The start error says `the engine process ended: ...` | The system ended the engine process, and the error says how: `killed by signal 4 (SIGILL, Illegal instruction)` on Linux, `exit code 0xC000001D: illegal instruction` on Windows, and so on. An illegal instruction means the engine ran an instruction the CPU cannot execute; on Linux the error lists which of AVX, AVX2, FMA and F16C the CPU lacks, or says the CPU has them all, which is a bug in the engine. Open an issue with the CPU model. |
| `the engine stopped unexpectedly (exit code ...)` | The engine process ended mid-answer, usually out of RAM (Linux ends the biggest program: `journalctl -k \| grep -i -E 'killed process\|out of memory'`). The next request starts it again by itself. If it repeats, close other programs or pick a smaller size. The server also warns at start when the model's experts leave less than ~6 GB of RAM for everything else. |
| On Linux the engine ended with signal 9, but RAM was enough | A real-time CPU limit (RLIMIT_RTTIME) on the desktop session lets the kernel end a program with signal 9 and no message. The audio system (PipeWire) sets it to 0 when rtkit is not installed, and every program started from the desktop inherits it. The server says so when it sees the limit, at start and when the engine ends with signal 9. Check with `ulimit -R`; install rtkit, or start outside the session's limits (`systemd-run --user --collect -p LimitRTTIME=infinity -- xestrata`). |
| Slow decode, disk light busy | Not enough free RAM: close other programs, or choose Q2_0 / IQ2_XS. |
| `prompt ... exceeds the context` | The request is longer than the context you chose: run setup again with a bigger `--context`. |
| An earlier reply that came back empty is gone from the chat history | The server leaves out finished assistant turns that have no text, so the next reply does not copy them (turns with pictures or tool calls and the last message stay). `XESTRATA_KEEP_EMPTY_TURNS=1` in the server's environment keeps them. |
| Slower than expected | A monitor plugged into the GPU and other GPU programs take VRAM from the expert cache; RAM below its rated speed (check XMP in the BIOS) slows the CPU half. Check setup's `SYCL compiler: ...` line for a build without XMX. |
| `this server was started without the vision encoder` | The model was set up for text only: run setup again with `--vision gpu`. |
| A picture is refused or `cannot read the image` | The file is not a picture Pillow can open (JPEG, PNG, WebP, GIF, BMP, TIFF, AVIF work). |
| Pictures are slow | The encoder runs on the CPU: run setup again with `--vision gpu`. |
| A request never finishes (`no progress for ... s`) | When the engine stops making progress, the request ends with an error (after 1 minute) and the next request starts the engine again. The log says where it stopped and what every CPU pool thread and the GPU handshake were doing. If you see it, please open an issue with that report. (`XESTRATA_WATCHDOG_S` sets the time in seconds; 0 turns it off.) |
| `Xe runtime: waiting for ... for ... s`, then `GPU wait exceeded the hang limit` | Work sent to the GPU had not finished after 120 s, and the engine ended itself (the next request starts it again). The xe driver ends a stuck job within 10 s, so on Linux with an Intel GPU this shows only when the GPU stays stuck. To wait longer for a long job on another driver, set `XESTRATA_GPU_WAIT_LIMIT_S` (seconds; 0 turns it off). If you see it, please open an issue with the log. |
| The engine stopped in the middle of a request (a toast in the web app, "Engine stopped" in the Monitor's state) | An exception such as a kernel launch the GPU runtime refused (device memory or resources short) or a GPU that stopped answering. The engine writes one line with what it was doing (the stage and the prompt chunk), the SYCL and Unified Runtime error codes, the VRAM free and whether the GPU still answers (Level Zero's `zeDeviceGetStatus`), ends, and the next request starts it again (issue #10). On Level Zero `level_zero backend failed with error: 40` stands for device memory and for resources alike. If the GPU is "lost", please open an issue with the xe lines and the PCIe AER lines of `sudo dmesg`. |
| `the engine said nothing for ... s during the request` | The engine and the server lost step (the engine waits for its next command, the server for the request's end; GPU at 0%, nothing in the log; upstream #481). The server ends the engine after 300 s without a line from it during a request, the request ends with an error and the next request starts the engine again. `"engine_silence_s": 600` in `xestrata-<model>.json` sets the time (0 = wait forever, as before). |
| `the engine said nothing for ... s and used no CPU or disk in that time (frozen ...)` | A quicker version of the check above (upstream #1317): an engine that prints nothing and uses no CPU or disk for 90 s is taken as stuck, ended, and started again by the next request. One whose GPU is busy (10% or more) or that is silent but working is not ended, only reported once. `XESTRATA_ENGINE_STALL_S` sets the time (0 = off). While a prompt layer reads the file tier again (low RAM), the engine's own watchdog also waits up to 10 times as long (`XESTRATA_WATCHDOG_IO_S`, upstream #1407). |
| Anything else | Follow the `log` field in the model configuration actually in use. Usually it is in `~/.local/state/xestrata` (for a source installation set up by an earlier setup, possibly in the XeStrata folder). |

The [recovery and diagnosis instructions](DIAGNOSE.md) are for LLM agents that can use your terminal.
They distinguish source and deb/rpm installations and ask for the cause and evidence whether the agent fixes the problem or gets stuck.
The report format brings together reproduction steps, logs, code investigation and changes for developers.
When opening an issue, include the report and necessary log excerpts after removing secrets and personal information.

---

## How it works

![memory tiers](paper/tiers.svg)

- **GPU (VRAM):** attention and DeltaNet mixers, the gated-residual weights, routers, shared experts, the output head, the MTP draft layer,
  the KV cache (from 64K only its most-read part; the rest streams from RAM),
  and an **expert cache** that fills the rest of VRAM with the most-used experts.
  The cache adapts to the conversation while you chat.
- **RAM:** all 24,576 experts, in one large mapping registered for device copies.
  The CPU computes the experts that are not on the GPU **in place**, at the same time as the GPU works on the cached ones
  (AVX-512 / AVX2 kernels, ggml's for the i-quants; on a CPU with AVX-VNNI the Q2_0 and IQ4_NL dots use it,
  `XESTRATA_NO_AVXVNNI=1` goes back to AVX2).
- **SSD:** the 28.8 GB n-gram table, read a few rows per token through the OS cache.
- **Speculation:** the model's own MTP layer drafts up to 3 tokens; one pass over all 48 layers checks them.
  On the B70 a check advances 2.1–3.4 tokens on average.
  When the reply repeats the context (code edits, quoted text), **prompt lookup** drafts up to 5 tokens from the earlier copy,
  but only where its measured acceptance and cost say it pays.
  The drafts are checked like the MTP's, so the output is the same.
- **Prefill** runs in chunks of up to 32,768 tokens, with the experts streamed to the GPU over PCIe.
  The matrix products run on XMX on Intel GPUs, on WMMA on AMD GPUs (RDNA3 or later) and on tensor cores on NVIDIA
  GPUs, and the dense ones go through oneMath: to oneMKL or cuBLAS in the contrib-llvm build, and to
  rocBLAS and hipBLASLt on AMD GPUs.

Strata's design, measurements and bottlenecks are in upstream's paper, **[docs/paper/Strata-Paper.pdf](paper/Strata-Paper.pdf)** (measured on CUDA).

---

## Credits and licenses

XeStrata's own code is under the [LGPL-3.0-or-later](../COPYING.LESSER), and XeStrata as a combined work is distributed under it.
The code from Strata, from ggml / llama.cpp and from Project Maya (all MIT) stays under MIT within it; their copyright and permission notices are in [NOTICE](../NOTICE).
The exceptions stay under their own licenses: the patches to intel/llvm and oneDNN, the changes to oneMath, the copies in `third_party/main/` (`ggml-common.h`, `ggml-common-tq.h`, `quantscope.py`, `stb_image.h`, Outfit, KaTeX) and the patch to AOTriton, and `third_party/nonfree/` (below and [third_party/main/README.md](../third_party/main/README.md)).
The model files are not part of it; their licenses apply to them (below).
The web app's About > Model terms shows the full terms of the models it runs now (setup fetches them with the models), and About > Licenses those of XeStrata and of the parts its build carries or is built with (the API: `GET /v1/licenses/models` and `GET /v1/licenses/software`).

- **The original software**: [Strata](https://github.com/Niko1221/Strata) by Niko1221 and the Strata contributors.
- **Models**: [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) by the Qwen team;
  quantizations: [ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF).
  The Coder: [ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF)
  (Apache-2.0 per its card); its support in Strata came from @pjgmobile's PR #54.
  Swift 1.5: [ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF](https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF) by UkisAI.
  UD-IQ1_M to UD-Q4_K_XL (Dynamic v3.1), UD-Q5_K_XL, UD-Q6_K_XL and Q8_0: [unsloth/Qwen3.8-Flash-Next-GGUF](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF) by Unsloth.
  Gyro: [agentionai/Qwen3.8-Flash-Next-Gyro-GGUF](https://huggingface.co/agentionai/Qwen3.8-Flash-Next-Gyro-GGUF) by agentionai;
  Gyro-Coder: [agentionai/Qwen3.8-Flash-Next-Gyro-Coder-GGUF](https://huggingface.co/agentionai/Qwen3.8-Flash-Next-Gyro-Coder-GGUF), also by agentionai.
  The abliterated versions: [huihui-ai/Huihui-Qwen3.8-Flash-Next-abliterated-GGUF](https://huggingface.co/huihui-ai/Huihui-Qwen3.8-Flash-Next-abliterated-GGUF) by huihui-ai;
  the GSQ-RCO abliterated versions: [SC117/Qwen3.8-Flash-Next-GSQ-RCO-abliterated-GGUF](https://huggingface.co/SC117/Qwen3.8-Flash-Next-GSQ-RCO-abliterated-GGUF) by SC117 (the weights with the refusals removed are OrcaRouter's);
  Swift 1.5's GSQ-RCO abliterated versions: [SC117/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-abliterated-GGUF](https://huggingface.co/SC117/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-abliterated-GGUF) by SC117.
  The other models: [Ornith-1.5](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B-GGUF) by Ornith AI (the GSQ-RCO ones pfeifferj's [reproduction](https://huggingface.co/pfeifferj/Ornith-1.5-35B-A3B-GSQ-RCO-GGUF), the version with its refusals removed dealignai's [Ornith-1.5-35B-A3B-UNCENSORED](https://huggingface.co/dealignai/Ornith-1.5-35B-A3B-UNCENSORED-GGUF)), [llm-jp-4.1](https://huggingface.co/llm-jp/llm-jp-4.1-32b-a3b-thinking-gguf) by LLM-jp, [Qwen3.6-35B-A3B](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-MTP-GGUF) by Qwen (the version with its refusals removed llmfan46's, as SC117's [APEX GGUFs](https://huggingface.co/SC117/Qwen3.6-35B-A3B-uncensored-heretic-Native-MTP-Preserved-APEX-GGUF)) and [GLM-5.3-Flash](https://huggingface.co/unsloth/GLM-5.3-Flash-GGUF) by Z.ai (both GGUFs by Unsloth), [Qwen3.5-9B](https://huggingface.co/unsloth/Qwen3.5-9B-GGUF), [Qwen3.5-2B](https://huggingface.co/unsloth/Qwen3.5-2B-GGUF) and [Qwen3.5-0.8B](https://huggingface.co/unsloth/Qwen3.5-0.8B-GGUF) by Qwen (GGUFs by Unsloth) and [Qwen3.5-9B-Uncensored-HauhauCS-Aggressive](https://huggingface.co/HauhauCS/Qwen3.5-9B-Uncensored-HauhauCS-Aggressive) by HauhauCS.
  The image models: Qwen's Qwen-Image-2.1 and Qwen-Image-2.1-Turbo (GGUFs by Unsloth), Microsoft's Mage-Flow (Comfy-Org's safetensors), RicemanT's MageTrail with Black Forest Labs' Flux2 VAE (FLUX.2-klein-4B's), Tencent's HunyuanImage 3.0 (quantized as single files by PedroMarinhoDev), OpenMOSS's MOSS-Image with its edit adapters (chinoll's Moss-Image-Edit, as a single file by PedroMarinhoDev; akhaliq's MOSS-Image-Edit-HQ armV5 and armV6), the text encoders, Qwen3-VL (Unsloth's, AtomicChat's, and huihui-ai's by noctrex), and the parts' fix's detectors, Bingsu's adetailer (`face_yolov8m`, `hand_yolov8s`, `person_yolov8m-seg`, trained with Ultralytics YOLOv8: Apache 2.0 by their publisher; Ultralytics holds models trained with YOLO to be under AGPL-3.0 by default), and the upscalers, Xintao Wang, Liangbin Xie, Chao Dong and Ying Shan's RealESRGAN x4plus (BSD-3-Clause; the official release's file, fetched from lllyasviel/Annotators) and ByteDance Seed's SeedVR2 7B, 7B sharp and 3B with its VAE (Apache-2.0; Comfy-Org's repackaging), the composition control's alibaba-pai Qwen-Image-2.1 Fun ControlNet Union (Qwen Research License) and darask0's MOSS-Image color and depth adapter and 4-step adapter (Apache-2.0), and the networks that make the control maps, Depth Anything V2 Small (Apache-2.0), DWPose (Apache-2.0), Megvii's YOLOX-L (Apache-2.0), ControlNet 1.1's HED (Apache-2.0), Caroline Chan's line art (MIT) and NAVER's M-LSD (Apache-2.0) ([IMAGE](IMAGE.md#licenses)).
  Their licenses apply to the weights.
- **[llama.cpp / ggml](https://github.com/ggml-org/llama.cpp)** (MIT): the i-quant formats, the GPU dot products and dequantizers transcribed in `src/kernels/xe/`,
  the CPU backend linked for the i-quant experts,
  and `gguf-py` used by the tools. See `third_party/main/ggml/LICENSE`.
- **[oneMath](https://github.com/uxlfoundation/oneMath)** (Apache-2.0): the layer that hands the dense matrix products to oneMKL (Intel) and cuBLAS (NVIDIA) in the contrib-llvm mode, and to rocBLAS and hipBLASLt (AMD) in the free and contrib-llvm modes.
  It is not in the repository: CMake fetches XeStrata's fork ([MistVVK/oneMath](https://github.com/MistVVK/oneMath), oneMath with XeStrata's changes: cuBLAS's BF16 product, hipBLASLt in the rocBLAS backend), under oneMath's license. See `third_party/main/oneMath/LICENSE`.
- **[intel/llvm](https://github.com/intel/llvm)**'s DPC++ (Apache-2.0 WITH LLVM-exception): the free and contrib-llvm modes' compiler and SYCL runtime.
  It is not in the repository: `tools/intel_llvm_build.py` fetches the release and builds it,
  with XeStrata's fixes from `third_party/main/intel-llvm/patches/`, under intel/llvm's license. See `third_party/main/intel-llvm/LICENSE.TXT`.
- **The image engine's parts**: for the attention, [oneDNN](https://github.com/uxlfoundation/oneDNN) (Apache-2.0, Intel GPUs), [AOTriton](https://github.com/ROCm/aotriton) (MIT, AMD GPUs; its kernels can carry parts of aiter (MIT), FlyDSL (Apache-2.0) and Triton (MIT)) and [cuDNN Frontend](https://github.com/NVIDIA/cudnn-frontend) (Apache-2.0, parts MIT, NVIDIA GPUs in the contrib-llvm mode); for the int8 products on NVIDIA GPUs in the contrib-llvm mode, [CUTLASS](https://github.com/NVIDIA/cutlass) (BSD-3-Clause), on AMD GPUs [comfy-kitchen](https://github.com/Comfy-Org/comfy-kitchen) (Apache-2.0); for reading reference images, [stb_image](https://github.com/nothings/stb) (MIT or the Unlicense). On Windows AOTriton links [XZ Utils](https://tukaani.org/xz/)' liblzma (0BSD).
- **[quantscope](https://github.com/1872183316/quantscope)** (MIT or Apache-2.0): what `setup.py --inspect` reads a GGUF's contents with.
- **[Project Maya](https://github.com/mw00/project-maya)** (MIT, by Peasant Smith and the Project Maya contributors): the server's loop guard (`serve/loop_guard.py`) is transcribed from it.
- **[ROCm's device libraries](https://github.com/ROCm/llvm-project/tree/f30ae3e6b680ad23dcf522e668c6d4c5c90838d0/amd/device-libs)** (University of Illinois/NCSA, by AMD): ocml, ockl and the others, which the code for AMD GPUs links.
  They are not in the repository: `tools/intel_llvm_build.py` fetches them from a release of ROCm's llvm-project and builds them with intel/llvm. See `third_party/main/rocm-device-libs/LICENSE.TXT`.
  The target features their build is given are transcribed from the `debian/rules` of Debian's [rocm-llvm](https://salsa.debian.org/rocm-team/rocm-llvm) package (Expat; its authors in NOTICE).
- **Ideas** (references in the paper) from:
   - [Splash](https://github.com/incoai/splash), by Inco AI and the Splash contributors;
   - [ninfer](https://github.com/Neroued/ninfer), by Neroued;
   - [HyperQwen](https://github.com/syv-ai/HyperQwen), by syv.ai and the HyperQwen contributors;
   - [Project Maya](https://github.com/mw00/project-maya), by Peasant Smith and the Project Maya contributors (how many
     of the experts missing from VRAM the CPU computes and how many are read over PCIe,
     [XE](XE.md#splitting-the-missed-experts-between-the-cpu-and-pcie));
   - [nextsycl](https://github.com/maxfridbe/nextsycl), by maxfridbe (which kernels the decode of the Qwen3.5 family's
     models merges; the code is written anew).
   - The YOLOv8 detector (`src/image/yolo_detect.cpp`): its layers and shapes are built from the tensors' names and
     shapes in the weights' file and the model configuration (its layer list) the file carries. The `.pt` weights are
     converted to safetensors by `tools/image/yolo_pt.py` (no torch: a reader that builds only the types it allows).
   - RealESRGAN x4plus (`src/upscale/esrgan.cpp`): the network (RRDBNet) is written from Real-ESRGAN's paper,
     [BasicSR](https://github.com/XPixelGroup/BasicSR)'s (Apache-2.0) RRDBNet's order of layers, and the tensors'
     names and shapes in the weights' file. The tiles overlap by 32 pixels around each, and the four times larger
     image is Lanczos-resized to the size asked for. The `.pth` weights are
     converted to safetensors by `tools/image/esrgan.py` (no torch: `yolo_pt.py`'s reader). The code is written
     without copying theirs.
   - SeedVR2 (`src/upscale/seedvr2_dit.cpp`, `seedvr2.cpp`; its VAE `src/image/mf_vae.cpp`'s LDM path): the 7B's DiT
     is written from ByteDance Seed's [SeedVR](https://github.com/ByteDance-Seed/SeedVR) (Apache-2.0) `models/dit`,
     the 3B's from its `models/dit_v2` (the windows, the joint attention of the text and a window, the RoPE, the
     modulation, the time embedding) and Comfy-Org's weights' names and shapes; the 3B checked against
     [vLLM-Omni](https://github.com/vllm-project/vllm-omni)'s (Apache-2.0) `vllm_omni/diffusion/models/seedvr2/`. The
     one step adds the noise, takes the condition latent times 0.9152 and takes the velocity off; int8_convrot's
     rotation is a regular Hadamard matrix. The code is written without copying theirs.
   - The upscale's color corrections (`src/image/img_color.cpp`): their names (`lab`, `adain`, `wavelet`, `none`) and
     the steps of `wavelet` (five dilated blurs) and `lab` (`wavelet`'s, then each CIE Lab channel matched rank for
     rank, the lightness kept 0.8) follow [vLLM-Omni](https://github.com/vllm-project/vllm-omni)'s (Apache-2.0) SeedVR2
     color corrections; the ranks are matched by a histogram of 65,536 bins, not a sort. The CIE Lab of the masked
     edit's color match (`color_match`) follows kornia's (Apache-2.0) CIE Lab conversion. The code is written without
     copying theirs.
   - SAM 3's detector (`src/image/sam3.cpp`, `include/strata/image/sam3.hpp`): written again for the engine after
     [vLLM-Omni](https://github.com/vllm-project/vllm-omni)'s (Apache-2.0) port, `vllm_omni/diffusion/models/sam3/`
     at 15bb15d (LGPL-3.0-or-later AND Apache-2.0, `NOTICE`), and checked against transformers' Sam3Model. CLIP's
     BPE that reads the words (`tools/strata_tokenizer.py`'s `ClipTokenizer`) is written from its `tokenizer.json`'s
     shape.
   - The video segmentation's rules (`src/image/sam3_video.cpp`, `include/strata/image/sam3_video.hpp`): after
     [vLLM-Omni](https://github.com/vllm-project/vllm-omni)'s (Apache-2.0) `vllm_omni/diffusion/models/sam3/sam3_video.py`
     at 2b610b3 (its association, hot start, occlusion suppression, delayed outputs and several words; LGPL-3.0-or-later
     AND Apache-2.0, `NOTICE`).
   - Florence-2 (`src/image/florence2.cpp`, `serve/image_regions.py`): written from Microsoft's
     [Florence-2](https://huggingface.co/microsoft/Florence-2-large) (MIT) `modeling_florence2.py` (DaViT, BART) and
     `processing_florence2.py` (the task prompts, the added tokens, reading `<loc_*>` and the polygons), its beam search
     after [transformers](https://github.com/huggingface/transformers)' (Apache-2.0) generation `_beam_search`, and
     checked against transformers' Florence2ForConditionalGeneration (florence-community's weights).
   - BiRefNet (`src/image/birefnet.cpp`; the foreground's colors `src/image/aux_kernels.cpp`'s `aux_fg_*`): written
     from Peng Zheng's [BiRefNet](https://github.com/ZhengPeng7/BiRefNet) (MIT) `birefnet.py` and `image_proc.py`'s
     `refine_foreground` (the GPU version of Photoroom's
     [fast-foreground-estimation](https://github.com/Photoroom/fast-foreground-estimation) blur fusion).
   - The composition control (`src/image/qi_control.cpp`, `src/image/cnet_*.cpp`, `serve/image_controls.py`):
     Qwen-Image-2.1's Fun ControlNet Union branch and how its condition is made (16 control blocks beside every second
     block of the model; 129 channels: the control picture's latent, a mark of what is kept, and the latent of the base
     picture with what is redrawn grey) were written from alibaba-pai's model card,
     [VideoX-Fun](https://github.com/aigc-apps/VideoX-Fun)'s (Apache-2.0) `QwenImage21ControlPipeline` and
     `QwenImage21ControlTransformer2DModel`, and the weights' names and shapes.
     The networks that make the control maps were written from their papers (Canny 1986, Xie and Tu's HED, Chan,
     Durand and Isola's Informative Drawings, Gu et al.'s M-LSD, Yang et al.'s Depth Anything V2 and Ranftl et al.'s
     DPT, Ge et al.'s YOLOX, Jiang et al.'s RTMPose and Yang et al.'s DWPose) and from the weights' names and shapes.
     To match what the control models were trained on, the skeletons' colors, stroke widths and hand and face points
     follow IDEA-Research's [DWPose](https://github.com/IDEA-Research/DWPose) (Apache-2.0) drawing in
     `ControlNet-v1-1-nightly/annotator/dwpose/util.py` (its colors and how the body, hands and face are drawn), M-LSD's
     segments are read as its authors' [mlsd](https://github.com/navervision/mlsd) (Apache-2.0) reads them (the
     centres' 3x3 peaks, the 200 strongest, a score and a length to keep one), and the margin of a person's box given
     to DWPose (1.25 times) is RTMPose's in [MMPose](https://github.com/open-mmlab/mmpose)'s (Apache-2.0)
     configurations. MOSS-Image's color and depth were
     written from darask0's model card and training code (`training/moss_control.py`, Apache-2.0), and MOSS-Image's 4
     steps (`src/image/moss_pipeline.cpp`'s `few_steps`, `moss_dit.cpp`'s `set_time_path`) from the sampler's formula
     in the model card of darask0's 4-step adapter and from its weights' names and shapes.
     comfyui_controlnet_aux (Fannovel16, Apache-2.0) was only compared with, by its outputs and speed; none of
     its code is copied.
   - [ComfyUI-Detail-Daemon](https://github.com/Jonseed/ComfyUI-Detail-Daemon), by Jonseed (MIT), after
     [sd-webui-detail-daemon](https://github.com/muerrilla/sd-webui-detail-daemon), by muerrilla (MIT): Detail
     Daemon's schedule (`make_detail_daemon_schedule`), its lookup and interpolation by sigma (`get_dd_schedule`),
     and DetailDaemonSamplerNode's way of changing only the sigma given to the model, with its defaults
     (`src/image/guidance.cpp`'s `detail_daemon_schedule` and `Guidance::model_sigma`). Written without copying its
     code.
- **The web app's font**: [Outfit](https://github.com/Outfitio/Outfit-Fonts) (SIL Open Font License 1.1, see `third_party/main/outfit/OFL.txt`;
  without it the app uses the system font). The chat draws math with [KaTeX](https://katex.org/) 0.19.1 (MIT, see
  `third_party/main/katex/LICENSE`). Its Monitor tab started from @code-martin's dashboard idea (PR #22).
  The chat page's wide-window layout, svg/html preview, math, Load/Unload, saved chats, edit and rewind came from upstream's PRs #3176 (@mnott), #1628 (@thoives), #2972 and #2974 (@CC-David-CC), #3550 (@sakuumomo), #1787 (@Scotty-Hudson) and #2615 (@uppet), and the rewind choice and sending while answering from issues #3823 and #4004 (@xvcezrqpbxtpzblrt).
- **`third_party/nonfree/`**: Qwen Community License 1.0, not free; XeStrata runs without it.
  It holds the experimental speed projection's vector, made from the model's activations, and the original model's chat template (see its README).
