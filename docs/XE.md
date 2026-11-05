<!--
SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
SPDX-License-Identifier: LGPL-3.0-or-later
-->
# XeStrata on Intel GPUs

English | [日本語](XE.ja.md)

This document records how XeStrata's engine runs on Intel GPUs and how that was checked: what the engine requires of the GPU, and the ported arithmetic and its validation.
How to use it is in the [README](../README.md) and the [details](DETAILS.md); how to build it, what setup does and the packages it needs, in [BUILD](BUILD.md).
The Japanese version ([XE.ja.md](XE.ja.md)) is the original; this is its translation.

XeStrata xe0.2.8-1 runs Strata's engine on Intel GPUs through Level Zero and SYCL; its version is the upstream version it has integrated and the count of XeStrata's releases on it, from 1, joined by a hyphen (xe0.2.8-1; on an upstream pre-release such as the beta, the deb and rpm packages take "~" for its "-": xe0.2.0-beta-1 is 0.2.0~beta-1, a version before the release 0.2.0-1, which upgrades it; releases before 2026-10-13 added a number from the second one on: xe0.1.40.2.1).
It is ported from Strata 0.1.24 (`3ce2523c2823687de5372be3af58534f56cbf286`) and carries part of the changes up to 0.1.39 (`6f32ec07`)
([Integration through Strata 0.1.38](#integration-through-strata-0138)).
The same SYCL code is compiled with intel/llvm for NVIDIA GPUs as well (the contrib-llvm build, [Build and run](BUILD.md#build-and-run)).
Upstream's CUDA build is retired and its sources are removed.
Upstream Strata (`3ce2523`) keeps them, and each Xe source names the CUDA file it ports in a comment.

It is developed and measured on an Intel Arc Pro B70 (Xe2, 32 GB); the speeds in the records are that card's unless they say otherwise.
On NVIDIA it is checked on an RTX 4070 and an RTX 3070.
The engine chooses its paths from what the GPU reports (AGENTS.md, the first rule).
The prefill path's matrix products use a kernel of their own on Xe2's XMX (xmx_gemm), a joint_matrix product that takes the reported tile shape on the Arc A series' XMX and NVIDIA's tensor cores (mma_gemm), the same product through the WMMA instructions on RDNA3 and RDNA4, and the DP4a paths on a GPU without matrix engines.
The experts' products in int8 (their quantized blocks as they are, iq_mmq) run on Intel's XMX (Xe2, Xe-HPG), NVIDIA's tensor cores (Turing on) and the WMMA of RDNA3 and RDNA4, and take DP4a only on a GPU without matrix engines.
Volta, whose matrix engines have no int8, takes the FP16 path.
Qwen3.8's prompt keeps the FP16 path on a GPU that runs xmx_gemm (faster there on the B70).

A smaller card is checked by making the B70 look like one.
`STRATA_VRAM_LIMIT_MIB` caps the VRAM the engine sees, `STRATA_MAX_ALLOC_MIB` the largest allocation it assumes, and
`STRATA_NO_XMX=1` takes the paths without the matrix engines.
The processor's own graphics (a UHD 770 on the development machine) is a second, real configuration for checks.

## Integration through Strata 0.2.8

The changes of upstream Strata 0.2.7 to 0.2.8 (`870aedd9`) are carried into Xe.
Most of upstream's changes in this range are the app, the Terminal, the Mac, DGX Spark and DeepSeek V4.1-Flash (left for later work), which are not taken.

What is taken:

- Route prior (`STRATA_ROUTE_PRIOR`) is no longer on by default. In 0.2.7 XeStrata set it to 1.5 by itself on one GPU whose expert cache held less than 20% of the expert pairs; upstream took its default back because in long agent sessions it repeats itself and its results are weaker, and XeStrata made it opt-in again (`STRATA_ROUTE_PRIOR=1.5`). On a GPU with a small cache the decode is slower than in the previous version: in everyday use at a context of 262144, 20-30% on the RTX 4070 and 10-20% on the RX 9060 XT (whose cache holds 16% there). On a GPU with a small cache `STRATA_ROUTE_TAIL_SKIP=4` (opt-in, [details](DETAILS.md)), which keeps the text better than the route prior, makes everyday use write 13 to 42% faster than rank 7.
- Serve's tool calls: damaged arguments, a stray closing think tag and a missing `</function>` no longer cost the call; a value that quotes a whole call, or repeats an argument name, is read as part of the value. This repair (`tool_call_recovery`) is on by default; `"tool_call_recovery": false` turns it off.
- The conversation cache: parking an idle batch slot no longer drops the main session's copy and the shared prefix's checkpoint.
- The engine: it waits up to 20 s for the RAM the expert arena needs before it loads; under the PLE table's RAM policy (`STRATA_PLE_AUTO_RAM=1`) the pinned table is given back when it leaves the conversation cache too little RAM; a stop on a CPU without AVX-512 names the cause, a pack without `native_experts.txt`, not the CPU.
- Layer split: on exactly two Xe2 GPUs `--pipeline-windows 2` is the default (as upstream; `--pipeline-windows 0` turns it off, [MULTIGPU](MULTIGPU.md)).
- Setup: an unused APU with a large BIOS carve-out is reported with the RAM the OS loses and how to fix it.

Compared with upstream v0.2.8 on the three makers' GPUs (IQ3_XXS: generate at 1K and 32K, serve at 32K, everyday use).
The B70 reads prompts 2.1 to 2.9 times and decodes 35 to 54% faster than upstream's `sycl/` port through generate; through serve, prompts 2.1 times and decode 65 to 66% faster. In everyday use the time to the first token, the prompt and the output are all faster.
The RTX 4070 reads prompts 17 to 74% and decodes 38 to 40% faster through generate; through serve, prompts 74 to 75% and decode 12 to 25% faster, and everyday use is faster in every measure.
The RX 9060 XT reads prompts 2.1 to 2.7 times and decodes 0 to 5% faster through generate; through serve, prompts 2.5 times and the greedy decode 18% faster. At temperature 0.7 the decode is 7% slower, because XeStrata writes a longer answer; through generate at 32K, on the same text, the two are level. In everyday use the time to the first token and the prompt are faster, and the output is 1 to 7% slower.

Not taken: DeepSeek V4.1-Flash, the app and the Terminal, the Mac, DGX Spark and upstream's `sycl/` items.
These are not taken either:

- HIP's default of verify and batch windows of up to 16 rows: it helps only batches of more than eight slots in one group under a layer split, and it is held.
- The prompt attention's lane cells and query batch: XeStrata's prompt attention is its own matrix kernel.
- The automatic expert cache allowing for what the load allocates after it: XeStrata has no warning for it, and it would only make the cache smaller.
- Setup recommending `--spec 2` for two Xe2 GPUs.

## Integration through Strata 0.2.7

The changes of upstream Strata 0.2.6 to 0.2.7 (`5e8a4cca`) are carried into Xe.
Most of upstream's changes in this range are DeepSeek V4.1-Flash (left for later work), the app, the Terminal, Docker and the Mac, which are not taken.

What is taken:

- The conversation cache: retained K/V is reused only for the conversation it was read for; an idle batch slot's decoded conversation is parked before an admission overwrites it; a checkpoint whose copy runs out of memory is skipped and the request goes on.
- Under `--serve`, when the link to the GPU probes below 12 GB/s, the conversation cache's copies move to background threads and VRAM stages (`STRATA_CKPT_ASYNC`, `STRATA_CKPT_GATHER`, `STRATA_PARK_ASYNC`; opt-in upstream). On the RTX 4070 (6.4 GB/s) a follow-up waits 201 ms instead of 226; on the B70 (28 GB/s) it is slower, so they stay off there.
- Route prior: `STRATA_ROUTE_PRIOR=1.5` turns it on (opt-in; it changes the answers). Upstream made it the default on NVIDIA cards of 13 GiB or less in 0.2.7, and XeStrata set it by itself on one GPU whose expert cache held less than 20% of the model's experts; in long agent sessions it repeats itself and its results are weaker, so in 0.2.8 upstream and XeStrata took it out of the defaults. The serve path decodes 6-77% faster on the RTX 4070 (15%) and 15-80% on the B70 limited to 12 GiB (11%), and 2-18% slower on the RX 9060 XT (25%).
- Serve: OpenAI `logprobs`, `top_logprobs` and `allowed_token_ids`; timings and `max_output_tokens` of 0 and -1 on `/v1/responses`; IPv6 bind addresses; tool results in the order of their calls; a call to an undeclared name named in the reply; Codex's namespace spellings; LaTeX-safe recovery of structured output (opt-in); 503 when the picture encoder cannot start; one final reply without tools at the MCP tool-round limit; MCP tool pictures (opt-in); split serving (prefill and decode on different GPUs, opt-in).
- The web page: the chat history redone (saved chats, edit and rewind, branches, sending while answering), nine languages (the browser's language by default), KaTeX math (opt-in, `third_party/main/katex/`), svg and html previews, Load and Unload buttons, wide windows, a Timeout badge for the queue.
- Setup and tools: the `--rope-scale` check, the published size when the download length cannot be probed, caps on the worker count and the picture encoder's threads, calibration dropping only a candidate whose engine died, bounded reads in `gguf_reader.py`, and NVIDIA GPUs older than Pascal told why they cannot be used.
- Engine fixes: no hang after a failed PLE read, no per-part allocations after the prefill's region cannot be had, the `--adapt-decay` range, `STRATA_TRACE` times, and the decode timing line's parts.
- Taken as opt-in (not faster on XeStrata's three makers' GPUs): the PLE table resident in RAM (`STRATA_PLE_AUTO_RAM=1`; on the B70 a 32K prompt no longer finished), swiglu fused with the q8 quantization (`STRATA_SWIGLU_QUANT=1`), the layer-major prefill (`STRATA_PREFILL_LAYER_MAJOR=1`), the DeltaNet recurrence on matrix units (`STRATA_GDN_TC=1`, changes the answers), KVA (`STRATA_KVA=1`, changes the answers, [KVA](KVA.md)), `STRATA_EMBED_VRAM`, `STRATA_PLE_COMPACT`, `STRATA_PLE_COPY_STREAM`, `STRATA_DEQUANT_SIDE`, the pipelined batch groups' `STRATA_PIPED_MTP` and `STRATA_PIPE_ADAPT_EVERY`, the `STRATA_RAM_ADAPT` change and GP100's `STRATA_DOWN_WS`.
- Two NVIDIA cards run `--batch` in pipelined groups again (one serial group with `STRATA_BATCH_GROUPS_NV_SERIAL=1`).

Compared with upstream v0.2.7 on the three makers' GPUs (the B70 and the RTX 4070: generate for every type, and the IQ3_XXS serve path and everyday use; the RX 9060 XT: IQ3_XXS and Gyro-M).
On IQ3_XXS through generate, the B70 reads prompts 1.8 to 3.0 times and decodes 36 to 64% faster than upstream's `sycl/` port; on the other types at 32K, prompts 1.6 to 1.9 times and decode from 25% to 3.2 times faster (UD-Q4_K_XL's prompt was not measured right: the run came just after a start with the file out of the page cache).
The RTX 4070 reads IQ3_XXS prompts 16 to 74% and decodes 0 to 23% faster, and on the other types at 32K prompts 8 to 43% and decode 1 to 24% faster. The RX 9060 XT reads IQ3_XXS prompts 2.6 to 2.8 times and decodes 3 to 20% faster.
Through serve, prompts are read 1.8 to 2.9 times faster on the B70, 1.17 to 1.73 times on the RTX 4070 and 1.8 to 2.6 times on the RX 9060 XT. The serve decode is 33 to 64% faster on the B70, -4 to +12% on the RX 9060 XT and -14 to +41% on the RTX 4070 (the slower cells are one run each at temperature 0.7 and 32K). In everyday use (chat-ja, chat-en and the agent), all three GPUs are faster than upstream in the time to the first token, the prompt and the output.

Not taken: DeepSeek V4.1-Flash, the app and the Terminal, Docker, the Mac, upstream's `sycl/`,
gfx12's default decode miss split (`--pcie-frac 0.15`: prompts 4-30% slower on the RX 9060 XT, and XeStrata splits by the measured link),
the gated speculative proposal (`STRATA_SPEC_PROB`: its probabilistic acceptance was not taken in 0.1.40),
CUDA's per-architecture tables and fused kernels (the sm_80 and sm_89 MMVQ-IL tables, the wide hyper-connection read, the fused IQ prompt experts),
GP100's weight-stationary MMQ and 4-bit expert format (experimental and large), and the hipBLASLt and `STRATA_SH_STREAM` entries.

## Integration through Strata 0.2.6

The changes of upstream Strata 0.2.2 to 0.2.6 (`4b9e05a2`) are carried into Xe.
Most of upstream's changes in this range are for DeepSeek V4.1-Flash (left for later work), the app, the terminal and the Mac, and are not carried in.

What was carried in:

- serve: without `--conversation-cache-mib`, the conversation cache is sized from the free RAM (both Qwen3.8-Flash-Next and PlainModel). A chat request with no messages is a 400. A one-token reply gives no decode speed. The tool-call aliases and the edges of the wrappers follow upstream, and a fenced tool call left unclosed is read too. An image a tool returned wakes the parked image encoder. The image token limit has a default, and an image above it is refused.
- Web page: an API key rewritten there is used at once, even when the browser refuses to store it. Long words, wide tables and the model settings rows stay inside a narrow screen.
- setup: `--kv fp16` is taken. The record of the packages it installed keeps the OS that wrote it. Build tools in the virtualenv come first, and an old cmake counts as none. A download that gets no data for 2 minutes says so, with how far it got and what to do next.
- Speculative decoding: the cost-aware improvements to the `--spec-window auto` window (a cost table per context length, outlier rounds counted as a fraction, looking one or two drafts ahead before keeping a draft) are opt-in with `STRATA_SPEC_COST=1`; PlainModel's speculative decoding uses the same part. They are not the default because the decode of a 1K prompt is 1.4 to 4.7% slower on the B70 and the RTX 4070 (1.3 to 3.2% faster at 32K).

What XeStrata changed on its side during this integration:

- What Qwen3.8-Flash-Next and PlainModel (GLM, Ornith, llm-jp, Qwen3.6) do alike is now one part both call: the option parsing, the DONE line, the expert tier's swap rules and its filling from the profile, the automatic interval when `--adapt-every` is not given, the PCIe split's measurement, the `--prefill auto` chunk choice, the prompt's candidates for experts on the CPU, and the MTP draft head over a vocabulary subset. llm-jp-4.1 does not shorten the swap interval by itself (it is slower when it does). A prompt chunk PlainModel picks from the VRAM tier is taken only when it is larger than 4096.
- PlainModel takes `--vram-elastic`, `--low-ram on|mmap`, `--kv int8` (not GLM), `--prefill-cpu-share`, `STRATA_RAM_ADAPT`, `--adapt-min-gain`, `STRATA_FETCH_THREADS` and `--spec-window` as well. GLM-5.3-Flash keeps the DSA latent in RAM and gives the freed VRAM to the experts (`--kv-ram`, on by default).
- The MTP draft head takes a 2-bit coarse product and a re-pick of the top 16 where that is faster on the GPU (both Qwen3.8-Flash-Next and PlainModel).
- `STRATA_DISK_SKIP_FROM` skips experts read from the SSD by rank (opt-in, from Project Maya's #109).
- The copies of the image layers streamed in go through a relay per slot and a multi-threaded memcpy.
- From 49152 tokens of context, the MTP draft head at 2 bits uses its coarse products only to pick the 16 candidates, and takes the draft probabilities from the 16 taken again (the coarse error of the other ~100,000 tokens lowered the probabilities, so the drafts stopped early at long contexts: the RTX 4070's 64K serve, first turn, 56.5 -> 71.7 tok/s). Shorter contexts keep the coarse values (the rest at a probability of 0 made the probabilities too high there: 5% slower on the RX 9060 XT at 32K).
- serve: with the conversation cache sized from the free RAM by default, every next turn of a chat parked the live conversation into RAM (the client drops the last reply's thinking from the history, so the turn read as a switch). The next turn of the same conversation no longer parks it; a switch to another conversation does, as before (both Qwen3.8-Flash-Next and PlainModel; 80 to 140 ms a turn off the wait on the B70, 120 to 370 ms on the RX 9060 XT).
- The three places that time paths at start-up to choose one no longer take the slower one on a noisy sample or short VRAM. The two forms of the DeltaNet recurrence are timed on the chunk's first 1024 tokens at most (for a 16384-token chunk the timing's copy did not fit in VRAM, which left the 5x slower form on: the RTX 4070's 32K serve prompt 1,687 -> 1,958 tok/s). The experts' products take FP16 only where it is 3% faster than int8. The hyper-connection read times its three forms in turn, three times.

Compared with upstream v0.2.6 on the three makers' GPUs (the B70: generate for IQ3_XXS, Q2_0, IQ2_XS, IQ3_S and UD-Q4_K_XL; the RTX 4070: the IQ3_XXS serve path; the RX 9060 XT: IQ3_XXS through generate and serve).
On IQ3_XXS through generate, the B70 reads prompts 1.8 to 2.7 times and decodes 24 to 48% faster than upstream's `sycl/` port; on the other types at 32K, prompts 1.7 to 2.5 times and decode from 12% to 2.9 times faster.
The RX 9060 XT reads prompts 2.1 to 2.8 times and decodes 8 to 10% faster.
Through serve, prompts are read 1.16 to 1.75 times faster on the RTX 4070 and 1.9 to 2.6 times on the RX 9060 XT. The serve decode is 0.2 to 52% faster on the RTX 4070 and 1 to 24% on the RX 9060 XT. In everyday use (chat-ja, chat-en and the agent), all three GPUs are faster than upstream in the time to the first token, the prompt and the output.

Not carried in: the DeepSeek V4.1-Flash support, the app and the terminal, the Mac, upstream's `sycl/`, the DFlash2 drafter,
the AVX-512 rows of the CPU i-quants (what reaches the main model is one IQ3 token only, on top of the DeepSeek CPU lane's set; there is no AVX-512 machine to measure them on),
the new trellis types (TQK5X, TQK4X and others: experimental, with no model released in them), the gfx1151 HIP experiments,
the start-up swaps of speculative decoding (upstream made them opt-in again), choosing the asynchronous tier by measurement (on XeStrata's tier, asynchronous was not faster in the RTX 4070's serve),
and the hipBLASLt tuning tables (XeStrata's AMD products go through oneMath's rocBLAS).

## Integration through Strata 0.2.2

The changes of upstream Strata 0.2.1.1 to 0.2.2 (`011c9e60`) are carried into Xe.
Most of upstream's changes in this range are for the app, the terminal, the Mac and upstream's `sycl/`, and are not carried in (for the same reasons as 0.2.0-beta).

Carried in:

- The CPU pool: the start-up says where its workers sit, `--pool-cores` (experimental) places them, and the engine's threads carry names; serve passes the setting `pool_cores` to the engine.
- Speculative decoding: the rule for when it starts is gathered in `spec_loop_runs`, and `STRATA_SPEC_WATCH` watches for stalls.
- serve: when a short remaining prefill window gives non-finite logits, the next request reads the whole prompt at once; an engine that cannot start gets a 503 with its reason instead of a dropped connection; `"api_key": "none"` and `--no-api-key` choose no key.
- serve: Anthropic's usage reports the tokens kept for the next request as `cache_creation_input_tokens`; llama.cpp's `thinking_budget_tokens` is read as the thinking limit too; a tool call whose fence is one token off is read as a call, and a call of an undeclared tool comes back as text; the done line's speed comes from the engine's decode time.
- setup: a download whose SHA-256 does not match is not deleted, only the differing pieces are fetched again, and the check of a large file shows its progress; the user's files under `--gguf-dir` are never deleted; free space stops the install only when it cannot fit; folders in the trash are not remembered as an earlier install.
- iq_pack: an experts.bin write that stopped halfway keeps the finished layers and goes on from the next.

Fixed on XeStrata's side during this integration:

- Regressions with intel/llvm v7.2.0: the XMX product and the grouped product lose `joint_matrix_prefetch` (IGC 2.28 cannot handle v7.2.0's prefetch; the B70's 1K prefill goes from 468 to 751 tok/s); AMD's libspirv gains GroupNonUniform's shuffles (the patch `hip-group-shuffle`); iq_mmq declares its work-group size by attribute, which ends the register spills on AMD (the RX 9060 XT's 32K prefill goes from 1,113 to 1,250 tok/s).
- Speculative decoding: the round that builds a window's graph is left out of the measure the window sizes are chosen by; the time of a size not yet measured is estimated from the nearest smaller measured size; Gyro's trellis layers do not cut a window where the VRAM swaps experts (Gyro-M's decode on the B70 at 1K goes from 48.0 to 54.2 tok/s); a window size without a graph shrinks to one that fits when the VRAM is short (the B70 at 262K no longer fails).

Compared with upstream v0.2.2 on the three makers' GPUs (generate with IQ3_XXS and Gyro-M on all three, Q2_0, IQ2_XS, IQ3_S and UD-Q4_K_XL on the B70 and the RTX 4070; IQ3_XXS's serve and everyday use).
In IQ3_XXS's generate, the B70 is 1.8 to 3.0 times as fast as upstream's `sycl/` build in prefill and 12 to 61% faster in decode.
The RTX 4070 is 1.02 to 1.16 times as fast in prefill and 25 to 37% faster in decode; the RX 9060 XT is 2.1 to 2.7 times as fast in prefill and 2 to 7% faster in decode.
In Gyro-M's generate, the B70 is 1.3 to 2.0 times as fast in prefill and 8 to 11% faster in decode; the RX 9060 XT is 1.3 to 2.3 times as fast in prefill and 1 to 2% faster in decode.
The RTX 4070 is 1.08 to 1.12 times as fast in prefill; its decode is even at 32K, and at 1K 3% slower when it follows upstream's text, 17% faster on XeStrata's own text.

Not carried in: the CPU pool split by NUMA node (XeStrata's pool has no groundwork for nodes), the app and the terminal, the Mac, upstream's `sycl/`,
CUDA's Gyro decode with the trellis codebook in shared memory and its Pascal kernels (tried in XeStrata's SYCL kernels: 17 to 19% slower on the RX 9060 XT, and no faster on the B70 or the RTX 4070), and HIP's fixes for gfx11 and gfx1151 alone.

## Integration through Strata 0.2.1.1

The changes of upstream Strata 0.2.0-beta to 0.2.1.1 (`9ae93f7b`) are carried into Xe.
Most of upstream's changes in this range are for the new app, the terminal, Docker and the Mac, and are not carried in (for the same reasons as 0.2.0-beta).

Carried in:

- setup: Meteor Lake's and Arrow Lake-H's LP-E cores count as E-cores; the low-RAM fit counts the draft layer's VRAM (about 1 GB).
- Downloads: 16 connections by default, four neighbouring pieces per request, a 0-0 range asked of a server whose HEAD gives no length; the MTP fetch goes on over a flaky link and stops with setup.
- setup: a new `--data-dir` moves no model files by itself and keeps finding the old place (`--move-data`, `--set-default-data-dir`).
- serve: an engine ended by signal 9 names a finite RLIMIT_RTTIME (PipeWire without rtkit) as the cause.
- Windows: `strata.exe`, `strata-image.exe` and `strata-device.exe` carry a UTF-8 manifest, so paths with letters outside the ANSI code page arrive intact.
- `--batch`: a layer split over two NVIDIA GPUs runs the default as one serial group (+12 to 18% in upstream's measurement).

Compared with upstream v0.2.1.1 on the RX 9060 XT (IQ3_XXS's generate): 2.1 to 2.7 times as fast in prefill, 3 to 7% faster in decode.
Upstream's engine changes do not touch the other GPUs' measured paths (one GPU, the experts in RAM), so the B70 and the RTX 4070 keep the 0.2.0-beta comparison.

Not carried in: the app and the terminal, the Docker distribution, the Mac, upstream's `sycl/`, HIP's integrated-GPU defaults, hipBLASLt tables and the order the GPU queues are opened in (XeStrata's RX 9060 XT prompts stay within 1.5% from start to start: the two speeds upstream fixed do not show), the experts.bin packs' I/O read-ahead and file tier (XeStrata lacks their groundwork), and the Intel GPU picture reader (XeStrata has its Vulkan reader).

## Integration through Strata 0.2.0-beta

The changes of upstream Strata 0.1.45 to 0.2.0-beta (`3d1fb480`) are carried into Xe.
When the release 0.2.0 is out, what it adds is carried in as well.

The main ones:

- Gyro-S and Gyro-M (agentionai): the trellis quantizations (TQ2_T, TQK6, TQK7) and the experts' Hadamard rotation, computed on the CPU (AVX2, AVX-512) and the GPU (decode and prompt), and listed by setup. The pack version is xs2.
- setup: the model files it downloads are pinned by size and SHA-256; large files are downloaded over several connections; the build tools (cmake, ninja) are apart from the run-time requirements; `--ple-io ram` also on an SSD when the RAM has room; pictures on by default for the models that read them.
- serve: a `max_tokens` that does not fit in the context is shortened to the room left instead of answered with 400 (`--no-fit-max-tokens` refuses it as before).
- The CPU pool's default workers are at most 32 and the container's CPU quota; the cgroup RAM check counts the clean page cache as reclaimable.
- The adaptive tier's swapped-in experts are used from the second window by default (`STRATA_ADAPT_LAG=2`, decode +8.6% on the RTX 4070).
- The picture reader (strata-vision) reads only the text model's vocabulary and header (so a file of Gyro's types opens).

Fixed on XeStrata's side meanwhile:

- The windows are cut at the positions of the VRAM experts' swaps only for a model with a layer that the CPU and the GPU round differently (`STRATA_ADAPT_CUT=0|1`). Every layer of Qwen3.8's four files gives the same bits on the CPU and the GPU, so their windows are not cut: on one text, decode 59.3 -> 61.3 tok/s on the RTX 4070 and 80.5 -> 86.9 on the B70.

Compared with upstream v0.2.0-beta on the three makers' GPUs (generate on five model files, serve, everyday use, image generation).
In IQ3_XXS's generate, the B70 is 1.7 to 3.0 times as fast in prefill as upstream's `sycl/` port and 18 to 38% faster in decode;
the RTX 4070 1.02 to 1.14 times in prefill and 23 to 34% faster in decode; the RX 9060 XT 2.1 to 2.7 times in prefill and equal to 8% faster in decode.
Image generation (1024x1024, 8 steps) takes XeStrata 7.6 s on the B70, 13.9 s on the RTX 4070 and 22.5 s on the RX 9060 XT, upstream 52.0 s, 26.6 s and 73.7 s.

Not carried in: upstream's new app (`apps/`) and terminal (`strata_cli`, `strata_control`), the changes to distribution (installers, signed delivery, engine part updates), NVFP4, upstream's `sycl/`,
the CUDA- and HIP-only speed paths (the sm_120 switches and others), `--kv-grow` on HIP's virtual memory (intel/llvm's HIP adapter lacks the virtual memory extension),
the low-RAM form that reads the GGUF in place (experts used without a copy, I/O read-ahead), and upstream's test tools and records.

## Integration through Strata 0.1.45

The changes of upstream Strata 0.1.44.1 to 0.1.45 (`477ac13c`) are carried into Xe.

The main ones:

- The prompt's MoE products read the experts resident in VRAM from their cache slots directly (B70, 1K: prefill +13%). The input is quantized once a token and spread to its k rows.
- The verify window's QSA block scores read a block's keys once for all of the window's rows (RTX 4070, 32K: decode +1.3%).
- The experts' decode keeps the activations in work-group local memory, the default on every GPU it fits (B70: +1%).
- The QSA top-k picks its radix digits with a cooperative scan; a layer whose experts are all resident groups its MMQ experts by their rows, most first.
- A layer split's prefill hands a chunk's rows to the next GPU in eight pieces, so the two GPUs' copies overlap (`STRATA_HANDOFF_PIPE=0`: one copy).
- The conversation cache: a new conversation that starts at another one's mid-conversation checkpoint borrows it (#1164); a conversation that grew little is not parked again (`--conversation-cache-park-gain`); checkpoints are kept spaced back from a conversation's end (`--prompt-cache-spread`).
- The resident RAM mode: with room left in the budget, the experts of the slots the prompt path borrows are kept in RAM too (#1683).
- `--layer-split auto`: when two cards leave a small stage, it picks again by the ratio of their room (#2644).
- serve: the thinking limit (`max_tokens` and Anthropic's `budget_tokens`), the model card's sampling for a request without one, the prompt tokenized from the last shared prefix (#1852), the next turn sent to the slot that holds the conversation's checkpoint (#2518), short prompts admitted first (`STRATA_ADMIT_SHORT_FIRST`, opt-in), parser fixes.
- Carried as opt-in: `STRATA_EXCHANGE_ROTATE`, `STRATA_RAM_ADAPT`, `STRATA_SPLIT_UNIFORM`.

Fixed on XeStrata's side during the integration:

- The boundary between the experts' int8 and FP16 products is timed up to 1024 rows an expert. Timed to 256 only, it read IQ3_S's and UD-Q4_K_XL's 32K chunks with the int8 products (B70, IQ3_S, 32K prefill 1,896 -> 2,425 tok/s).
- The choice of the big prompt chunks no longer counts the RAM copy it has already made a second time (B70, UD-Q4_K_XL, 32K prefill 1,760 -> 2,412 tok/s).
- A layer split's later stages no longer hold a flat 1 GiB out of their expert caches, only their verify windows and, on the last stage, the draft head that binds after the cache (B70 + RTX 4070 at 10 GB: the second card's cache +560-700 slots, prefill +4-5%).

Compared with upstream v0.1.45 on the three makers' GPUs (generate on five model files, serve, everyday use, image generation); the B70 is now compared with upstream's `sycl/` port too.
On the RTX 4070, generate's prefill is the same or faster on all five files (IQ3_XXS at 1K and IQ2_XS 17% faster, UD-Q4_K_XL 1%), and decode 13-30% faster.
On the RX 9060 XT, prefill is 2.1-3.2 times as fast, and IQ3_XXS's decode, 2.5-4.7% slower in 0.1.44.1, is now the same to 7% faster.
On the B70, prefill is 1.6-2.5 times as fast as upstream's `sycl/` port and decode 15-95% faster (UD-Q4_K_XL is not compared: the `sycl/` port's answers on it are broken).
Image generation (1024x1024, 8 steps) takes XeStrata 7.6 s on the B70, 13.2 s on the RTX 4070 and 22.3 s on the RX 9060 XT, upstream's sd-cli 52 s, 25.5 s and 73.5 s.
Against the XeStrata before the integration on the same text, decode is as fast on the B70 and the RTX 4070, and the B70's prefill at 1K 35% faster.

Not carried: upstream's `sycl/`, the batch work (more than 8 slots on a layer split, the slots' GPU-to-GPU copies, the admission opt-ins, the one-launch rows),
the Turing (sm_75) speed paths, `--prefill-pipe` (B70 + RTX 4070: prefill 38.6% slower in the smaller chunks), `STRATA_SPLIT_OVERLAP` (XeStrata's stages share no memory word between the GPUs),
the CUDA opt-in paths without a measurement, and upstream's test tools and records.

## Integration through Strata 0.1.44.1

The changes of upstream Strata 0.1.43.1 to 0.1.44.1 (`296b59d2`) are carried into Xe.
From this run on, Windows changes are taken too (except the HIP parts, which XeStrata's Windows build does not have).

The main ones:

- Qwen3.6-35B-A3B (Unsloth's UD-Q4_K_XL, UD-Q6_K_XL and Q8_0 with the MTP layer) is read on the same plain path as
  Ornith-1.5. Upstream gave Qwen3.6 an engine of its own for CUDA; in XeStrata the plain path reads the GDN's F32
  alpha and beta and the MTP layer's BF16 routers. Its answers were compared with llama.cpp's and it runs on the three
  makers' GPUs. Right after a prompt the VRAM tier is filled with the experts that prompt used most (upstream's q36
  idea), which makes decode 20-40% faster (Ornith and llm-jp too).
- Image generation (Qwen-Image-2.1-Turbo, `POST /v1/images/generations`): instead of upstream's stable-diffusion.cpp,
  XeStrata's own SYCL image engine, `strata-image` (the DiT, the VAE and the text encoder; attention on the matrix
  units; resident beside the server, streaming the layers that do not fit in VRAM).
- Unsloth's files: the K-quant experts' weights are decoded once per grouped call (KqSplit, the same bits; B70
  UD-Q4_K_XL decode +7%). setup offers UD-Q5_K_XL, UD-Q6_K_XL and Q8_0 as well (experimental).
- The commit graph does every GDN and QSA layer in one launch (`STRATA_COMMIT_BATCH`, on by default).
- The layer split's automatic placement (among near ties, the one that leaves the tightest card the most room; a slow
  link's miss cost only from a 4x gap), and the VRAM check at start covers the later cards too.
- The 16384-token prompt chunk only where the RAM covers its loan. The arena's MADV_HUGEPAGE is not asked for when
  free 2 MiB blocks cover less than half of it (#2126).
- serve: a long read gives way to a short request at its next chunk (#656), opencode's `<env>` block moved into the
  first user message (`"env_to_user"`, opt-in), the image encoder only while there are pictures (`"elastic"`, opt-in),
  an engine stopped inside the GPU runtime's set-up started again, how a crashed engine ended, the Responses API's
  `reasoning_effort`, pictures matched by their bytes in the conversation cache (#2228), and the stdin reader's QUIT
  race.
- The quantization yardstick (`tools/quant_quality.py`, `tools/quant_formats.py`).
- `STRATA_HC_Q8` (the hyper-connection down/up read in Q8_0; a BF16 pack's quantized at load): both forms are timed
  at load and the faster kept; with Q8_0 on one GPU the BF16 copies leave the VRAM, and on AMD the read also writes
  the next projections' q8_1 (B70 decode +3.5%; an approximation that changes the output).
- Ported and kept opt-in: `STRATA_MTP_PRIO` and `STRATA_COPY_PRIO`, `STRATA_POOL_UMWAIT`, `STRATA_PCIE_MIN1`, `--adapt-async 1` in the arena
  mode, `STRATA_DENSE_F32`.

Fixed on XeStrata's side during this integration:

- A failed GPU query is told from an unfinished one, and a wait ends with an error (issue #12). The "RAM is tight"
  warning counts the experts actually placed in RAM (issue #11).
- WSL2: setup not finding the GPU, a failure in the plain path's PCIe split, and a prompt buffer too large (the three
  points of a report).
- A container to install and run it with Podman (podman compose).

Compared with upstream v0.1.44.1 on the three makers' GPUs (generate on five model files, serve, everyday use):
on the RTX 4070 generate's decode is 8-21% faster on all five files and the prefill as fast or faster (IQ3_XXS at 1K
16%, IQ2_XS 15%). On the RX 9060 XT the prefill is 2.1-3.2x, but IQ3_XXS's decode is 2.5-4.7% slower: upstream reads
the hyper-connection down/up projections in Q8_0 by default on gfx12 (+4.4% when turned off and on in upstream).
XeStrata times it at load and takes Q8_0 too, but gains only 0.6% on the RX 9060 XT. The B70 is not compared
with upstream; on the same text it is as fast as XeStrata before this integration.

Not ported: the Mac (Metal), upstream's `sycl/`, the NVFP4 experts (on GPUs without FP4 matrix units their prompts
are slower than the IQ formats', and their quality below UD-Q4_K_XL's), upstream's CUDA-only Qwen3.6 engine, gfx1103
in the gfx11 WMMA kernels (held with the RDNA2 and 3 items), `--prefill-head-alias` and the arena's pin by need (they
do not fit XeStrata's USM and arena), and upstream's test tools and records.

## Integration through Strata 0.1.43.1

The changes of upstream Strata 0.1.43 to 0.1.43.1 (`a0116b27`) are carried into Xe, except the Windows implementation.
They were ported one at a time.

The main ones:

- Asynchronous host-to-device copies from pageable RAM go through a pinned slot, or wait until they are sent (upstream #1996).
  They were the draft K/V's tokens, MTP's prompt records and the image embeddings' rows.
- PLE: a slow direct read only warns and no longer switches to mmap (`STRATA_PLE_AUTO_MMAP=1` for 0.1.42's switch).
  `--ple-io mmap`'s prompt reads ask for the next 4096 rows ahead (upstream #2040, #1984, #1971).
- The watchdog: a file tier busy within the last 10 s keeps its I/O allowance. Parking a conversation copies 1 MiB at a time so the watchdog does not stop it,
  and the stall report names an MTP draft wait (upstream #2049, #2065). `STRATA_TRACE=2` waits for the device at each startup mark.
- serve: a refused BGEN slot tries another slot (#1994), a failing before_load hook answers 503 with Retry-After and `/unload?wait=1` is taken (#1997),
  a quoted `</tool_call>` in a message is text (#2026), a call in a Markdown fence inside `<tool_call>` is read (#2037),
  and a client that leaves while its pictures are prepared stops the preparation (#1999).
- Layer split with `--batch`: the default groups no longer give way to `--batch-mtp` (#2038; `STRATA_BATCH_MTP_SERIAL=1` for the old form).
- setup's `--calibrate` tunes the model of `--model` and `--family` (#2036). The benchmark reports only the GPUs the engine uses (#1985, #2062),
  and the server it starts runs without `--expert-profile-save` (#2088).

Fixed on XeStrata's side meanwhile:

- Prompt path: the FP16 and int8 experts' rows and groups share one place, so a model with both paths keeps its chunk.
  RTX 4070, UD-Q4_K_XL at 32K: 460 → 617 tok/s (upstream: 620).
- Small VRAM's prefill: the hyper-connection block's rows live in the attention and MoE work's region (`STRATA_PREFILL_HC_IN_REGION=0` for their own place),
  and the FP16 experts' input rows in the upper half of their down outputs' rows (`STRATA_PREFILL_XS_IN_DM=0`).
  IQ3_XXS at 64K on an RTX 4070 limited to 8 GB: 291 → 428 tok/s.
- With `--spec` and an expert cache of no slots, the engine stops before reading the prompt.
  setup warns of a context whose KV cache and expert cache fit in neither VRAM nor RAM, and names a `--context` or `--kv q4_0` that fits.

Measured against upstream v0.1.43.1 on the three makers' GPUs (generate with five model files, the serve path, everyday use).
On the RTX 4070, IQ3_XXS's decode is 11 to 20% faster from 1K to 262K and the prefill equal or faster.
On the RX 9060 XT, the prefill is 2.1 to 3.1 times and the decode 5 to 12% faster.
On an 8 GB card (both engines run with 8 GB of VRAM left), the prefill is 270 against 194 tok/s on the RTX 4070 at 64K and 713 against 306 tok/s on the RX 9060 XT at 32K.

Not carried: the Windows implementation, upstream's `sycl/`, `STRATA_ADAPT_OVERLAP` and its callback thread's tuning (a feature XeStrata does not have),
telling an adaptive stage's landing by a stream-ordered copy instead of a host function (XeStrata raises its flags in a host function only with `--pcie-mode dma`, not on its default path),
a smaller chunk that would lend more slots (#2090; XeStrata's ring is monotonic in the chunk), and upstream's test tools and records (`bench/results`).

## Integration through Strata 0.1.43

The changes of upstream Strata 0.1.42 to 0.1.43 (`7695859a`) are carried into Xe, except the Windows implementation.
They were ported one at a time; each speed idea was A/B-measured on one GPU and taken when it was faster or no slower.

The main ones:

- A guard on the windows whose logits are NaN or infinite: the request ends and the state it would reuse is dropped (upstream #879, on by default; `STRATA_NAN_GUARD=0` turns it off).
- Prompt path: the host copy of the K/V written 4 bytes a lane (`STRATA_KV_WIDE_HOST=0` for the old form; on the B70 at 64K, 1,974 → 2,399 tok/s),
  the expert copies on the GPU's copy-only engine (Level Zero, `STRATA_COPY_ENGINE=0`),
  an expert's gate/up and down dequantized in one launch (`STRATA_PF_DQ_FUSED=0`; +9% on the B70's UD-Q4_K_XL at 32K),
  the chunk's step records and token ids sent from page-locked buffers (upstream #1712).
- Decode: the expert hits folded into the multi-row combine (`STRATA_FUSE_HITS=0`), Q3_K's GEMV unpacking once a block,
  the route-resident swap one sub-group a token (inside the opt-in feature, +3.9% on the B70).
- serve's tail skip off by default when 90% or more of the experts are in VRAM (upstream #1884, `STRATA_TAIL_SKIP_MAX_CACHED`).
- serve's reports (time and tokens of a request that moved between the solo path and a batch slot, `timings.prompt_n`; upstream #1869 and #1870),
  `STRATA_DROP_CACHE_ON_EXIT`, the file-cache hint when RAM is short, `STRATA_TRACE`'s VRAM marks.
- `xestrata --setup --bench-original-compat`, which runs upstream's benchmark mode (`strata_bench.py`) as it measures ([BENCH](BENCH.md)).

Fixed on XeStrata's side during this integration:

- The split of the missed experts between the CPU and PCIe refines the CPU's times from the decode ([Splitting the missed experts between the CPU and PCIe](#splitting-the-missed-experts-between-the-cpu-and-pcie)).
- When the RAM cannot hold the whole expert arena, the engine starts as with `--resident-experts` instead of loading it, and moves layers into host USM only while the RAM has room for them (issue #10, [DETAILS](DETAILS.md)).
- An engine that stops on an exception says in one line what it was doing, the SYCL and Unified Runtime codes, the VRAM free and whether the GPU still answers, and the web app shows it (issue #10).
- The web app's GPU readings agree across the three makers (Intel's temperature is the GPU's own, a layer split's other cards, product names, the PCIe path's limit).
- serve writes the checkpoint at the start of the assistant turn into RAM after the reply ([DETAILS](DETAILS.md)). On the RTX 4070 a continuing request waits 222 instead of 288 ms at 1K and 253 instead of 317 ms at 64K.

Measured against upstream v0.1.43 on the three makers' GPUs (generate with five model files, serve, everyday use): XeStrata is equal or faster in every measure.
On the RX 9060 XT, IQ3_XXS decodes 9-13% faster from 1K to 262K (25-45% faster than xe0.1.42) and reads the prompt 2.9-3.5 times as fast.
The RTX 4070's serve follow-ups wait 222 and 253 ms, against upstream's 240 and 282 ms.

Not taken: the Windows implementation, upstream's `sycl/`, the HIP and CUDA kernels' tuning tables and the paths for GPUs there is no machine for (Pascal's rows table, gfx12's fused int8 prompt experts, gfx12's decode HC norm and folded flag waits, HIP layer-split defaults),
the mixed expert profile upstream made for `sycl/` (generate's 32K decode: B70 −2.0%, RTX 4070 +6.2%, RX 9060 XT −5.3%),
what has nothing to act on in XeStrata (`STRATA_ADAPT_OVERLAP`, PcieAuto's share, skipping the GEMM operand staging, refilling lent slots by DMA), and upstream's records (`bench/results`).

## Integration through Strata 0.1.42

The changes of upstream Strata 0.1.40.2 to 0.1.42 (`61b3fb5d`, 0.1.41 included) are carried into Xe, except the Windows implementation.
They were carried one by one by hand.

The main ones:

- On serve's decode, `STRATA_ROUTE_TAIL_SKIP=7` (default): a missed expert a token routes at rank 7 or lower contributes zero to that token (decided for each token, not for the window), and one that every token of a verify window routes that low is neither read nor computed. Upstream keeps it off on HIP; XeStrata measured the KL, a task check and the speed again on the three makers' GPUs and turned it on for all of them.
- `STRATA_ROUTE_RESIDENT` (opt-in) and `STRATA_ROUTE_PRIOR` (opt-in), which lean a verify window's routing towards the experts in VRAM; they change the answers.
- The prefill CPU share's extension (experts in RAM that is not page-locked as well) and the rule that leaves it off by default when every expert is in VRAM or memory is short.
- `--aux-cpus` (opt-in), `--ple-io direct` falling back to mmap when it measures slow, the stager's RAM profile, the embedding buffer's reuse.
- The memory guard (the cgroup limit and `--memory-limit-mib`), `--conversation-cache-min-tokens`, `--session-save-reclaim`, `logit_bias`, `strata --version`.
- serve's changes (metrics in Prometheus' format, the Monitor's conversation-cache card, AMD GPU readings, the sampling fields, the tool list's order and more) and setup's (how the config is written, `--gguf-dir`, rotational disks, a VRAM reserve on a card that also drives a display, the worker count on hybrid CPUs and more).

Measured against upstream v0.1.42 on the three makers' GPUs (generate with five model files, serve, everyday use):
the B70 and the RTX 4070 match or beat upstream in every cell; the RX 9060 XT does in every cell but generate's decode at
32K (36.9 against 44.6 tok/s) and serve's wait for a follow-up at 1K (526 against 478 ms), and reads prompts 2.2 to 2.4
times as fast at every length. Its decode moves by about a third from run to run in the same setting (upstream's too),
for reasons not yet known.

Found during the final comparison and fixed on XeStrata's side:

- When page-locking the RAM copy of the experts is refused as one allocation (the B70's Level Zero takes 30 GiB at most), it is taken in parts cut between experts (`STRATA_PIN_PARTS=0`: as before). UD-Q4_K_XL at 32K on the B70: prefill +17%, decode +14%.
- The measurement behind the CPU/PCIe split of the missed experts holds still from start to start ([Splitting the missed experts between the CPU and PCIe](#splitting-the-missed-experts-between-the-cpu-and-pcie)).
- serve's short reads keep each path's readings with where the read started and choose from the readings near it; a batched read is tried only up to 16K. On the RTX 4070 at 64K the fourth turn's follow-up went from 505 to 304 ms.

Of the paths meant to be faster, those faster on the B70, the RTX 4070 or the RX 9060 XT and slower on none are carried.
The CPU share's auto gate (upstream f2a3b946) slowed 1K prompts on the B70 and the RTX 4070, so it is opt-in (`STRATA_PREFILL_CPU_GATE=1`); the Q8_0 dequantization four threads a block (upstream e910e410) was faster on neither, so it is left out.
Not carried: the Windows implementation, upstream's `sycl/` (measured against on the same B70), the HIP and CUDA kernels' tuning tables, some paths for GPUs there is no machine for (those for RDNA2 only), and the multi-GPU changes (carried together in work of their own).

## Integration through Strata 0.1.40

The changes of upstream Strata 0.1.39 to 0.1.40.2 (`e8ca9afd`) are carried into Xe, except the Windows implementation.
Of the AMD (HIP) changes, those that apply to XeStrata's AMD path (the same SYCL kernels) are carried: an AMD APU's unified memory (setup, and the engine's free memory of an integrated GPU) and the fused RMSNorm + RoPE, which was meant for every GPU.
Upstream's HIP-only speed switches were measured on and off on an RX 9060 XT (gfx1200); those for gfx11 (RDNA3) and gfx103x (RDNA2) only are not carried, with no such card to try them on.
Upstream's `sycl/` (another port to Intel GPUs) is not taken in: it is measured against on the same B70.
On NVIDIA GPUs the same kernels run in the contrib-llvm build ([BUILD](BUILD.md)).

The main things carried:

- Conversations saved to a file and restored (session files), `ckpt=0` for a request that takes no checkpoint, `pin=N` to keep a shared beginning, and `tools/research_run.py`.
- The PLE table is read in the format the GGUF has it: IQ4_NL (the default table), Q4_0, Q5_0, Q5_1, Q8_0, FP8 (E4M3 with a scale) or BF16.
  Upstream measured the mean per-row error against the BF16 table: Q8_0 0.53%, FP8 2.64%, Q5_1 3.78%, Q5_0 4.25%, IQ4_NL 7.60%, Q4_0 8.55%.
- GPU kernels for the gate/up of native Q4_0, Q4_1, Q5_0 and Q6_K experts; the CPU experts' AVX-2 i-quant and Q2_0 kernels.
- Fused kernels with fewer launches in the verify windows and the prefill path (#783 and others).
- `STRATA_PREFILL_CPU_SHARE=auto` (the default): a short chunk's least-routed experts are computed by the idle CPU pool.
- `--kv-grow`: the K/V takes VRAM only for the cells a request has reached, and the expert cache uses the rest.
  It is on by default where the device's virtual memory mapping granularity is 2 MiB or more (on for the RTX 4070; the B70's is 64 KiB, off).
- `--adapt-async` (the resident mode's default), `--batch-mtp` (the default with `--batch` and `--mtp`), `--pipeline-windows` (the default on a split of two Xe2 GPUs, opt-in otherwise, [MULTIGPU](MULTIGPU.md)) and `--lookup-chain` (opt-in).
- Setup's additions (`--inspect`, `--source modelscope`, a 200K context and more) and the server's changes.

Of the paths meant to be faster, those faster on the B70 or the RTX 4070 without being slower on the other are carried.
One faster on only one of them is chosen at run time from what the device reports, or is opt-in.

The integration builds in the free (intel/llvm), contrib-icpx (icpx) and contrib-llvm (intel/llvm with CUDA) modes, and all 71 CTest cases pass in each (`expert_multi_test` skips on a CPU without AVX-512).
The AVX-512 path passes `expert_multi_test` under Intel SDE (`-icx`).
`expert_parity --selftest` and `pool_test --selftest` need the canonical pack and cannot run under SDE (they pass on AVX2).
On the B70 the icpx and free builds completed all eight answers (thinking on and off), six of them identical.
Prefill 1,148 tok/s, decode after it 98.6 tok/s, short chats 77.3 tok/s (icpx).
The configuration with 8 GiB, 4 GiB, no XMX and six CPU workers, and the RTX 4070, completed the answers as well.
On the UHD 770 the free and icpx builds started and gave a short answer.

## Integration through Strata 0.1.38

Of upstream Strata's changes up to 0.1.38 (`99f3dbd0b21d1401b3769e0c0d963913607f380b`), the single-GPU Linux ones are carried into Xe.
The upstream history is kept as a merge.
The CUDA, HIP and Windows implementations are not carried; of the multi-GPU ones, only the layer split is ([MULTIGPU](MULTIGPU.md)).

Carried:

- `--coupled-draft`: MTP drafts are sampled with the target model's sampling chain (off by default; `--no-coupled-draft` turns it off explicitly).
  This includes the draft vocabulary map and the penalty history.
  A server config can set `coupled_draft`; engine arguments take precedence.
- `--resident-budget-gib`, `--resident-experts` and `--resident-cpu-experts` for the file expert tier.
- `--shared-expert-arena PATH`: completed expert weights shared between Linux processes.
  The data is flushed before completion is published under a file lock.
  The reader verifies the pack's identity before reusing it.
- Native PLE keys accept Q2_0, IQ3_XXS, IQ4_XS and Q8_0.
  A BF16 PLE key in the pack is used when there is one.
- Native RoPE follows the request's scaling configuration.
- MTP downloads verify the pinned checkpoint and refuse invalid ranges.

The paths meant to be faster are not carried: measured on the B70, none was faster.

| Path | On the B70 |
| --- | --- |
| Quantized expert products in the prefill path (`STRATA_PREFILL_MMQ`) | 0.34x of the default XMX FP16 path |
| Fused MoE (`STRATA_PF_FUSED`) | 0.25x of the default path |
| GDN key-head sharing (`STRATA_GDN_KEYHEAD`) | 0.54x of the default path |
| Split hyper-connection read (`STRATA_GR_V3`) | Decode about 18% slower |
| P-core / E-core placement of the CPU pool (`--pool-affinity`) | Decode 1–4% slower |
| AVX2 multi-token K-quant products (`STRATA_KQ256`) | Within the spread |
| Software prefetch in the AVX2 IQ kernels | Within the spread |
| Routing-aware prefetch of the file tier (`STRATA_LOOKAHEAD`) | Decode about 2% slower |

The quantized products and the fused MoE compute on the decode kernels' DP4a dots, without XMX.
A GPU with XMX therefore runs prefill faster on the default path.

The integration builds in both modes, free (dpclang++) and nonfree (icpx), and both pass the 52 CTest cases.
The default configuration's output (IQ2_XS, nonfree) was compared with the pre-integration build (`bae99bd`).
32 tokens from an 18-token chat were the same in both of two runs.
16 tokens from a 4,095-token prompt were the same in all three pre-integration runs.
After the integration, two of three runs matched them and one differed from the first token ([Why outputs change from run to run](#why-outputs-change-from-run-to-run)).
Under Intel SDE (`-icx`), the AVX-512 paths pass `expert_multi_test` and `native_expert_parity --synthetic` (iq3_xxs/iq4_nl, q4_K/q5_1).
`expert_parity` and `pool_test` were skipped: there is no canonical pack.
The integration commit checked that, with the 8 GiB limit, no XMX and two CPU workers, the same 16 tokens come out with and without `--coupled-draft`.
On the UHD 770 the free build answered with a two-token MTP window.
The nonfree build's four-token window stopped inside the Intel runtime.

## Runtime contract

What the engine requires of the GPU and the host, and the rules it keeps.

- **BIOS**: a discrete card needs Above 4G Decoding and Re-Size BAR enabled and CSM disabled.
  Otherwise the B70's BARs stayed unassigned and the xe driver did not bind.
- **Choosing the GPU**: the runtime drives one GPU (several with a layer split, [MULTIGPU](MULTIGPU.md#gpu-numbers)): an Intel GPU through Level Zero, an NVIDIA GPU through the CUDA backend (not OpenCL, which lists the same Intel GPUs again).
  It is chosen by what it reports, never by its maker or device ID.
  It needs 32-wide sub-groups, FP16, and device and host USM, and the executable must carry code for it (an NVIDIA GPU with a build without `STRATA_CUDA_ARCHS` is refused).
  `STRATA_GPU_PCI` (setup writes it) names the card by PCI address.
  Without it, a discrete card is taken before the processor's own graphics (Level Zero's integrated flag; the other backends' GPUs count as discrete cards),
  then the one with the most memory, then the most compute units (whose size differs between makers).
  There is no CPU fallback.
  A GPU without what it needs is refused with the reason.
- **Queues**: one context owns an in-order compute queue and an in-order transfer queue.
  `copy_async` returns an event; `compute_after` inserts the cross-queue dependency.
  For readback, pass the compute completion event to `copy_async` and wait before reading the host buffer.
  The existing parity harness takes this path with separate host USM staging allocations.
  `create_stream` makes further in-order queues on the same context, for the work CUDA put on its own streams.
  `destroy_stream` drains and releases one, and `finish()` drains them all.
  A kernel's `void* stream` is resolved by `Runtime::stream`:
  an engine stream, the compute queue, or null (the compute queue, with synchronous completion where CUDA synchronized); any other pointer is refused.
  Work on different streams is ordered only through events, as with CUDA's non-blocking streams.
- **Allocations**: device allocations are 4096-byte aligned; bump allocations validate capacity without wrapping.
- **The expert arena**: `PinnedArena` is one 2 MB-aligned anonymous mapping with `MADV_HUGEPAGE`.
  It is registered with `prepare_for_device_copy` before the loader fills it; a failed mapping or registration throws.
  Host USM is not used for it:
  Level Zero refuses a single USM allocation above `max_mem_alloc_size` (32.5 GB on the B70), and every model's arena but the Coder's is larger.
  The registration is split into pieces no larger than the device's largest allocation, ending at layer starts.
  Past that size the UHD 770 (4 GiB) registered without an error and copied wrong bytes.
  `registered_bytes` is the whole capacity. No hugetlbfs pool is used; transparent huge pages back the arena.
  The registration needs no memlock limit (the B70 registered a 47.46 GiB arena under an 8 MiB limit).
  Level Zero's registration (an xe or i915 userptr) follows the pages instead of pinning them, so when memory compaction moves one page, the driver maps the whole registration again for the GPU at the next copy (#42).
  The filled arena is therefore locked with `mlock` (`locked_bytes`) where `vm.compact_unevictable_allowed` is 0, else pinned with io_uring's fixed buffers.
  A shared arena (memfd) is moved even when pinned, so only `mlock` with `vm.compact_unevictable_allowed=0` holds it. Both need a memlock limit as large as the arena.
- **Following the device**: kernels take what the device reports.
  A work-group of 1024 work-items only where the device takes one (the UHD 770 takes 512).
  FP64 only where the device has it; on the UHD 770 and the Arc A series those kernels sum in FP32.
- **The verify window's wait**: the verify window's GPU waits for the CPU's experts by spinning on a flag in host memory.
  That needs a running kernel to see the host's writes.
  The verifier checks that at start (`doorbell_visible`).
  Where it fails (the UHD 770), each window runs as segments the host launches one after another, with no waiting kernel: the same results, slower.
  `STRATA_VERIFY_SEGMENTED=1` / `0` forces either.
  A window whose verifier layers have every expert in VRAM runs as another graph that plans the experts on the GPU and never waits for the host (upstream cfd3b72; no segments either).
  On the B70 the Coder IQ1_M went from 28.0 to 28.5 tok/s with the same logits.
  `STRATA_VERIFY_RESIDENT_GRAPH=0` turns it off.
- **Lifetimes**: every source, destination and reader stays alive until its final consumer completes.
  A copy-complete event alone does not allow reuse while a kernel still reads the destination.
  Arena destruction drains every owned queue before freeing USM or releasing the registration.
  USM pointers are used in their allocation context, and their dependencies are the application's responsibility
  (see the [Khronos USM reference](https://github.com/KhronosGroup/SYCL_Reference/blob/main/source/iface/usm_basic_concept.rst)).
- **Errors and waits**: synchronous SYCL failures propagate to the CLI; asynchronous errors are kept and checked at waits.
  Waits block in the driver; a watchdog thread reports the named wait and ends the process when one lasts 120 s.
  The xe driver resets a job after `job_timeout_ms` (5 s here), so a healthy wait returns well inside that limit.
  A hang or an unsafe teardown failure ends the process without unwinding through live GPU allocations.
  Driver job timeouts are not modified; CTest uses an outer 60-second timeout.

The parity harness also keeps its small embedding-plus-scale capture/replay check through the experimental [oneAPI graph extension](https://github.com/intel/llvm/blob/sycl/sycl/doc/extensions/experimental/sycl_ext_oneapi_graph.asciidoc).
This does not validate a whole-token graph or a CPU/GPU doorbell protocol.

## Implemented arithmetic

### Elementwise operations

The Xe library exports the embedding gather `embed_rows` (`iq_kernels.hpp`), `scale_inplace` and `f32_to_f16_bulk`.
The other declarations in `elementwise.hpp` are not ported and not provided by the library.
These kernels and the parity oracle are compiled without fast math and without multiply/add contraction.

All device code is built with `-foffload-fp32-prec-div -foffload-fp32-prec-sqrt`.
CUDA divides and takes square roots with IEEE rounding by default; SYCL device code does not.
`quantize_act_parity` sees the one-ulp difference in Q8_K's `1/iscale`.

| Operation | Reference and unchanged gate |
| --- | --- |
| Packed embedding row | Independent scalar unpacker; separate multiply/add rounding; bitwise equality, signed zero, guards; 108 row cases plus captured executions |
| GDN gate | CPU double softplus with the `x > 20` branch; relative L1 ≤ `1e-6`, negative gates |
| SiLU | CPU double formula; relative L1 ≤ `1e-7`; Xe uses a stable FP32 formula |
| Scaling | CPU FP32 multiplication; bitwise equality |
| FP32 → FP16 | Shared integer conversion; bitwise agreement |
| RMS norm | CPU FP64 sum/mean, epsilon, and optional weight; relative L1 < `1e-6`, finite output, original null-weight check |

No test cases were added; fixture inputs, CPU expected values and tolerances are unchanged.
`quantize_act_parity`, which also carries NumPy-derived FP16 boundary checks, passes on the B70.
New cases require approval.

### i-quants and MMVQ

All of `iq_kernels.hpp` is ported (`src/kernels/xe/iq_kernels.cpp`):
the q8_1 quantizer, dequantization of IQ1_M, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, IQ4_NL, IQ4_XS, Q3_K and Q2_0, their MMVQ,
embedding rows, the prefill path's gate/up expansion, and the grouped native experts.
A warp is a sub-group of 32; its sums use CUDA's butterfly order, and CUDA's byte intrinsics are written out.
SwiGLU in the grouped experts uses the precise exponential where CUDA used `__expf`.

All of `native_mmvq.hpp` is ported too (`src/kernels/xe/native_mmvq.cpp`):
the llama.cpp MMVQ adapters for Q2_0, Q3_K, Q4_K, Q5_K, Q6_K, Q4_0, Q5_0, Q8_0, IQ4_NL and IQ4_XS (1–8 columns, in the exact and upstream layouts), and the dispatcher.
The dispatcher sends the other i-quant types to `iq_mmvq`, as the CUDA one does.
One column also runs the multi-column kernel in the CUDA single-column layout; there are no separate single-column kernels.
The CUDA file kept the two bitwise equal as well.
Calls require an explicit runtime stream and enqueue without waiting, as the CUDA contract required a non-null stream.

The i-quant measurements show all ten formats dequantizing real model rows exactly as gguf-py does (Q2_0: the definition mirrored from ggml).
Their MMVQ is within `iq_parity`'s `2e-2` bound (4.2e-3–5.8e-3), and `iq_parity` reports 0 failures.
Q6_K, Q8_0, Q4_K, Q5_K, Q4_0 and Q5_0 MMVQ are within `2e-2` of FP64 on random blocks.
Columns 2–8 are bitwise equal to single-column calls in the exact layout.

### Other decode kernels

- `src/kernels/xe/rope.cpp`: `rope.hpp`, `native_rope.hpp` and the M-RoPE table of `mrope.hpp` (one pointer: one Xe device)
- `src/kernels/xe/router.cpp`: `router_top10.hpp` and `native_router.hpp`
- `src/kernels/xe/bf16_gemv.cpp`: `bf16_gemv.hpp`, including the ggml mmvf kernels
- `quantize_act.cpp`, `qsa.cpp`, `kv_q8.cpp`, `kv_q4.cpp`, `qsa_decode_attn.cpp`, `kv_stream.cpp`: the headers of the same names.
  The QSA entry points throw `DeviceError` where CUDA exited

The KV host pools (`KvHostPools`) are read and written by kernels, so on Xe they must be host USM.
Memory registered only for device copies is not kernel-addressable.
`native_rope` and `native_router` use precise `pow`/`cos`/`sin`/`exp` where CUDA used fast math.
The decode-kernel checks record `rope_parity`, `router_top10_parity` and `bf16_gemv_parity` passing; all are registered in CTest.

## Models on the B70

Each model went through the same checks (the procedure):

- `native_expert_parity` on layers covering every expert type pairing, and `dequant_bf16_test`
- a short prompt, and 96 greedy tokens against llama.cpp at the pinned commit on the CPU
- the three draft modes with fixed residency
- the server (streaming, a cancel, a three-turn conversation)
- a 26,293-token prompt with and without KV streaming

The tensor inventory pins the files.

| Model | Result | Against llama.cpp: tokens before the first split (the reference's top-two gap there) | Draft modes |
| --- | --- | --- | --- |
| Qwen3.8-Flash-Next IQ2_XS | runs | 96 of 96 without drafts | the same tokens |
| Qwen3.8-Flash-Next IQ3_XXS | runs | 6 (0.090) | differ; see below |
| Qwen3.8-Flash-Next IQ3_S | runs | 61 (0.051) | no drafts = suffix; MTP differs at the same near-tie |
| Qwen3.8-Flash-Next Q2_0 | runs (the native pack; the canonical AVX-512 pack is not run on this CPU) | 47 (0.050) | the same tokens |
| Qwen3.8-Flash-Next Coder IQ1_M | runs | 20 (0.238) | the same tokens |
| Swift 1.5 IQ2_XS | runs | 7 (0.151) | no drafts = suffix; MTP differs |
| Swift 1.5 IQ3_XXS | runs | 62 (0.058, three candidates) | differ (the cache size also differed between runs) |
| Swift 1.5 Q2_0 | runs (note) | 25 (0.273) | no drafts = suffix; MTP differs |

Note: in Swift 1.5 Q2_0, layer 13's expert tensors span both shards.
The pack index names a shard per tensor.
Packs from before 2026-10-02 say v4, newer ones XeStrata's own xs1.
xs1 is numbered apart from upstream's v&lt;N&gt;, so neither is taken for the other.

Every split from llama.cpp comes at a near-tie of the reference; after it the comparison says nothing.
The native path writes `--dump-logits` rows from the last prompt token on (its prompt is read batched, without the head).
The records above were taken before that and compare tokens only.

Where the draft modes differ, there are two causes:

- The MTP runs had fewer cached experts (the drafter takes VRAM), so some experts ran on the CPU instead of the GPU.
  From 2026-10-17 the CPU and the GPU (Intel, NVIDIA, AMD) compute the experts whose gate and up rows are i-quants (IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S), IQ4_XS or Q2_0 and whose down rows are IQ4_NL or Q2_0 (every layer of the four Qwen3.8 files) alike, and an expert gives the same bits on either (`canon_math.hpp`, `canon_expert.cpp`).
  The SwiGLU and the int8 scale are built from additions, multiplications and fmas only, the gate/up activations take one scale for each 256 values, and the integer sums of what shares a scale add first, then go into the GPU's lanes with fmas in the GPU's order.
  The answer stays the same when the window, the PCIe split or the tier changes move an expert between the CPU and the GPU.
  K-quant and BF16 experts (Unsloth's files, Ornith, llm-jp) are not covered.
  `STRATA_CANON=0` takes the CPU's former kernels (for comparisons; the GPU's stay, so the CPU's and the GPU's results then differ).
- On Qwen IQ3_XXS, the suffix run differed from the no-draft run even with the same cache.
  The CPU took its multi-token expert kernels (i-quant gate/up and IQ4_NL down) from two tokens on and ggml's single-token dot for one, so a token's rows were summed in another order alone than in a window.
  From 2026-10-16 the CPU computes these and Q4_K, Q5_K and Q6_K with the multi-token kernels from one token on, and Q2_K, Q3_K and Q8_0 with ggml's dot at every token count. A token's rows are the same alone and in a window, and with the same cache the draft modes give bit-identical logits (the GPU side of a multi-token window added no difference before either).
  `STRATA_IQ_MT_MIN=N` takes the multi-token kernels from N tokens as before (for comparisons).

### Why outputs change from run to run

Which experts run on the CPU follows the cache size, and that changes the output's last bits.
`auto` sizes the cache from the free VRAM, so free VRAM moving between starts moves it by a slot or two and can change the output.
Rounding it down to 64 slots gave every start on a machine the same cache, but it slowed the RTX 4070's decode by 3-4%, so it is not rounded.
Setup's warm-up instead writes the VRAM free at its start, less 32 MiB, into the run config (`--vram-free-mib`): the engine does not count more free at its start than that, so a start with a little more free VRAM gets the same cache (on the B70 it moved by about 20 MiB between starts, 15 slots). A start with less free gets a smaller cache.
When the same output is needed (comparisons across machines or settings too), fix it with `--expert-cache N`.

### The output head's VRAM refusal

Sometimes the upload of the output head is refused VRAM at start.
About 27 GiB is free then, and it is always the first device allocation after the arena is registered for device copies.
The same settings then start normally.
On the development machine it happened 3 times in about 120 starts.

### The VRAM reserve on a small card

On a small card (upstream #496), the default 700 MiB reserve can leave the cache too few slots to lend the prefill path a 256-token chunk.
The engine then lowers the reserve to what leaves that many:
down to 300 MiB, and not below a reserve given with `--vram-reserve-mib`.
When even that is not enough, the start stops and says how many MiB are short and what makes room.
Checked with `STRATA_VRAM_LIMIT_MIB`:
at 4,608 MiB (IQ2_XS, 32K, MTP) the reserve became 482 MiB, leaving the 368 slots it needs;
at 4,096 MiB the start stopped 330 MiB short.

## Splitting the missed experts between the CPU and PCIe

In a verify window the experts missing from VRAM are computed by the CPU or read over PCIe and computed by the GPU.
How many go each way is chosen from times measured at start (after [Project Maya](https://github.com/mw00/project-maya)'s CPU lane; the code is written anew).
Before, it was a share from the link's speed alone (0.55 from 20 GB/s, below that in proportion).

- **Measured**: the CPU pool's time for an expert (64 taken in turn, past the CPU's cache), the window's copy kernel (`fetch_blobs`) reading one over PCIe,
  the window's expert kernel computing one from VRAM, and the memory's rate as the CPU reads experts.
- **Sharing the memory**: the CPU and PCIe read the same memory; when their rates together pass the memory's, both are taken as slowed by that factor.
  On the B70 that gives 0.13 ms an expert over PCIe (0.066 alone), as the decode's waits work out.
  The memory's rate is the fastest of five readings 100 ms apart, taken once the kernels' preload thread is done, and in generate before the prompt is read.
  A single reading beside other work came out up to 3x low, and on the RX 9060 XT some starts then gave the CPU every miss, the 64K decode at 21 tok/s (38-41 once steady).
- **The split**: for each group of a window, the GPU computes its VRAM hits, then (beside them where the window's fetch branch runs) reads and computes the PCIe experts,
  and the CPU computes the rest; the number read over PCIe makes the later side earliest.
  Experts go over PCIe only when that is at least 10% earlier than the CPU taking them all.
- **Refined from the decode**: the CPU's time at start is measured a token at a time, and the window's pool calls can run slower (1.27-1.29x on the RX 9060 XT, 3.5x beside a busy host).
  The decode's calls are set against the time the split expected for them, and when they are more than 15% off, the CPU's times are scaled by the ratio (after 512 calls with experts on the CPU, then 1,024 and 2,048 more, three updates at most; as upstream's PcieAuto refines its share).
  On the RX 9060 XT, IQ3_XXS decodes at 52.2 instead of 50.3 tok/s at 1K, 51.3 instead of 48.5 at 64K and 44.7 instead of 42.5 at 262K; on the B70 and the RTX 4070, whose times at start match the decode's, nothing changes. `STRATA_PCIE_SPLIT_REFINE=0` keeps the times taken at start.
- **The arena**: on Level Zero, GPU kernels cannot read the arena registered for device copies.
  Only when the split reads experts over PCIe, the arena moves into host USM a layer at a time (about 7.5 s for IQ3_XXS's 40 GB). `STRATA_ARENA_HOST_USM=1` puts it in host USM from the start.
- **The wait before READY**: the engine waits, before it says READY, until the kernel has stopped moving host pages (`pgmigrate_success` in `/proc/vmstat` below 10,000 pages a second for 2 seconds, at most 30 seconds; Linux only).
  The compaction right after the load can move the registered arena's pages as well, so it is over before the first request. `STRATA_SETTLE_S` sets the most seconds, 0 no wait.
- **The share stays**: with `--pcie-frac` (also as `--calibrate` saved it), over a layer split, with `--pcie-mode dma` or `direct`, and when the times cannot be measured.

The RTX 4070 (PCIe 4.0 x4) now computes every miss on the CPU: IQ3_XXS 32K 49.9 -> 57-58.6 tok/s, UD-Q4_K_XL 32K 22.0 -> 25.7.
The B70 keeps every miss on the CPU, as before; the RX 9060 XT reads 1.0 instead of 4.7 experts a layer over PCIe; neither changes speed.

## Images

Images are available only when chosen in setup (off by default).
When they are, the image encoder (`tools/vision`: llama.cpp's mtmd with the model's mmproj) runs on the GPU by default (`./setup.sh --vision yes` or `gpu`).
When it cannot, the CPU encoder takes over; `--vision cpu` uses the CPU encoder alone.
The CPU encoder reads the pictures checked as llama.cpp does, on grids of both shapes.
Every model that runs was checked with images:
two grid shapes, two images in one request, and a conversation that sends the image again and reuses its cache.

The GPU encoder is the Vulkan build (ggml-vulkan, `-DSTRATA_VISION_VULKAN=ON`).
It takes the GPU by its PCI address (`strata-vision --gpu-pci`, from the config's `vision.gpu_pci`):
with Vulkan the first GPU can be the processor's graphics or another card.
On Intel GPUs its driver is Mesa, free software.
It encodes 7–9× faster than the CPU encoder (8 threads); its embeddings differ from the CPU encoder's by 2.4–3.1% (relative L2) and repeat bit for bit.
(The SYCL build of the contrib-icpx mode, ggml-sycl with oneMKL, was removed with that mode on 2026-10-15.)

The server starts the encoders in this order until one is ready:

1. the GPU encoder
1. the CPU encoder (the config's `vision.fallback`)

The console and `/v1/status` (`vision.encoder`, `vision.fallback`) say which one serves and why the ones before it did not start.
When none starts, or the serving encoder stops later, the server keeps serving text and `vision.error` says why.

## The IQ2_XS file's tensors

The frozen model inventory records the Hugging Face revision and the SHA-256 of each complete GGUF header.
The tensor CSV records every tensor's actual type, shape, offset and byte size.
Only headers were fetched for that inventory; a header prefix is not a usable model shard.
Generation with the complete model, the prefill path, MTP and prompt-lookup drafts, the server, and a 26k-token prompt with KV streaming were checked
with IQ2_XS too.
Greedy tokens were compared with llama.cpp on the CPU.

The `IQ2_XS` distribution at the recorded revision holds 1,224 tensors in 12 storage types, with **no tensor stored as IQ2_XS**.
The name describes the quantization budget of a mixed-precision model.

| Tensor role | Observed storage | Required operations and reference |
| --- | --- | --- |
| Routed expert gate and up | IQ2_S in 34 layers, IQ2_XXS in 11, IQ1_M in 3 | Per-format decode, activation quantization, selected rows/GEMV and batched GEMM; pinned ggml `to_float` and CPU FP64 math |
| Routed expert down | Q2_0 in all 48 layers | Native Q2_0 decode/dot and CPU AVX2 path; the same pinned ggml expert oracle |
| Quantized dense/shared expert weights | IQ3_S, IQ4_XS, IQ4_NL, Q6_K, Q8_0, Q2_0 | Per-type decode, dense GEMV/GEMM and SwiGLU; pinned ggml plus FP64 accumulation |
| Token embedding and output head | IQ4_XS | Row decode, projection, sampling; pinned ggml decode and host arithmetic |
| HC/router/SSM/indexer/PLE projections | BF16 | BF16 activation rounding and projection; existing BF16/GR/router/GDN/QSA parity |
| Norms, biases, recurrence parameters | F32; PLE convolution F16 | RMS, gate, convolution, recurrence, indexing; existing host references |
| PLE n-gram table, second shard | IQ4_NL | Integer n-gram hash, selected row reads/decode; existing PLE vectors and actual table rows |

The native expert oracle in `src/kernels/native_expert_parity.cpp` uses ggml dequantization and host FP64 dot products before comparing the CPU and GPU paths.
On the layers of the types above, it also checks that the CPU's rows and the GPU's are the same bits, the tokens at once and one at a time.
Its per-expert relative L1 bound of `3e-2` belongs to that comparison and is not a universal kernel tolerance.
The GPU's q8_1 activation contract and the CPU's activations (`canon_expert.cpp`'s form for the experts of the types above, ggml's `vec_dot_type` for the others) stay explicit.

## Validation inventory

| Tests and assets | Status and limits |
| --- | --- |
| `strata-device --selftest`, `elementwise_parity` | Ported; pass on the B70. The device selftest checks allocation, alignment and capacity, not numerical readback |
| `gguf_reader_test`, `suffix_drafter_test`, `controller_test`, `draft_policy_test`, `conv_cache_test` | Pass with the current toolchain; they do not establish full engine integration |
| `s_gemv_parity`, `s_gemv_q8k_parity` | Ported, registered in CTest, pass on the B70; `--bench` also checks `s_gemv_split`, `s2_gemv_quads` and `s2_gemv_fast` against the naive kernel |
| `dequant_s2_parity`, `s2_gemv_parity`, `s2_gemv_q8_parity` | Ported, registered in CTest, pass on the B70 |
| `rope_parity`, `router_top10_parity`, `bf16_gemv_parity`, `quantize_act_parity`, `qsa_parity`, `kv_q8_parity`, `kv_q4_parity`, `kv_stream_parity`, `gdn_parity`, `gr_parity`, `cvec_parity` | Ported, registered in CTest, pass on the B70 |
| `shared_expert_parity` | Ported, registered in CTest, passes on the B70, including the SYCL-graph replay of the native scalar gate |
| `sampler_parity` | Ported, registered in CTest, passes on the B70 |
| `qsa_prompt_attn_parity` | The existing FP64 reference and FP32 baseline, ported to SYCL (built, not a CTest case: it is also a benchmark). The XMX kernel passes it at int8 and FP16 KV, the tensor-core one at INT8, FP16 and K8V4 (Q4_0 skipped); without the matrix engines the caller keeps `qsa_decode_attn_batch` |
| `native_moe`, `native_gdn`, `native_gdn_preprocess`, `native_ple_postops`, `native_qsa`, `native_qsa_score`, `native_qsa_indexer`, `native_flash_attn`, `qsa_select`, `fused_gdn` (no in-tree parity) | Probes against ggml-cpu at the pinned commit pass on the B70; not registered in CTest |
| `native_expert_parity`, `dequant_bf16_test` | Ported, built with `STRATA_NATIVE_EXPERTS`, not registered (they need the model shards). Pass on the B70 with the shards of every model under [Models](#models-on-the-b70) |
| `iq_parity` | Ported and built, not registered. `tools/iq_fixture.py` writes `logs/iq_fixture` from ranged reads of the pinned model revision. All 10 formats pass. Q6_K/Q8_0 are not in its list |
| `ple_parity` | Has the checked-in `ple_oracle_vectors.inc`; full checks also need the actual table and reference artifacts |
| CPU expert/pool tests and `pool_stress` | Not in this inventory: the canonical expert tests need AVX-512 and/or a pack. The AVX-512 paths run under Intel SDE as AGENTS.md says |
| `serve.test_server`, `serve.test_detok`, `serve.test_mcp` | Pass with the mock engine |
| `tools/test_iq_pack.py`, `tools/test_shards.py`, `tools/test_calibrate.py` | Pass under `python -m unittest` (2026-10-04) |
| `src/ngram/ple_reader_test.cpp`, `src/platform/memory_test.cpp` | Built again; `ple_reader_selftest` passes, `platform_memory_test` fails under the 8 MiB memlock limit. Neither is evidence for the registered arena |

## Upstream settings not remeasured on Xe

These settings were chosen by upstream Strata on CUDA, mostly on a Ryzen 5 7600 with an RTX 5070.
XeStrata uses them as they are; only some have been tuned on Xe.

| Source | Setting | On Xe |
| --- | --- | --- |
| `src/program/generate.cpp` | Native missed-expert PCIe share 0.55 (canonical 0.2), scaled by min(1, link GB/s / 20), the link read as the best of four 256 MiB bursts | On the new development machine 0.15 was 2–4% faster and every larger share slower; the measurement does not support 0.55 |
| `tools/calibrate.py` | Baseline Ryzen 5 7600 + RTX 5070; PCIe candidates 0/0.2/0.35/0.55/0.75, draft floor 0.3/0.5/0.7, worker candidates; interleaved confirmation requires >3% gain | Candidates and baseline unchanged from upstream |
| `setup.py` | `--spec 4`, `--spec-min-p 0.5`, `--expert-cache auto`, `--prefill auto` | `--prefill auto` now picks chunks up to 32768 tokens; a chunk that does not fit with the default ring takes a 96-slot ring (upstream #583; IQ3_S on the B70 limited to 8 GB: 8K and 32K prompts 1.5-2x); one that still does not fit is planned again with the int8 products' group scratch (GU, H, Hq) sized for one group's rows instead of the chunk's (RTX 4070, IQ3_XXS 32K: the chunk 8192 -> 16384, prefill 1,194 -> 2,058 tok/s; a group's tile shape rounds its sums, so the answer can differ a little from the uncapped run's); the window's size comes from the output per ms at the measured window times and the corrected probabilities, not from the `--spec-min-p` floor (`--spec-window auto`, [DETAILS](DETAILS.md#tuning-for-your-pc---calibrate)); the rest as upstream |
| `src/program/generate.cpp` | 700 MiB VRAM reserve (down to 300 on a small card); prefill borrowing and chunk selection; cache adaptation every 4 rounds and up to 96 swaps | As upstream |
| `src/program/generate.cpp` and the CPU pool | Physical-core worker heuristic plus host worker; cache-hit pokes and graph/doorbell scheduling | The worker count was measured on the new machine: 7 to 19 workers decode equally fast, so the default stays |
| `src/kernels/cpu/iq_avx512.cpp` | Software prefetch distance and AVX-512-specific tuning | Not measured: the development machine has no AVX-512 |
