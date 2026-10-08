<!--
SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
SPDX-License-Identifier: LGPL-3.0-or-later
-->
# XeStrata on Intel GPUs

English | [日本語](XE.ja.md)

This document records how XeStrata's engine runs on Intel GPUs and how that was checked: what the engine requires of the GPU, and the ported arithmetic and its validation.
How to use it is in the [README](../README.md) and the [details](DETAILS.md); how to build it, what setup does and the packages it needs, in [BUILD](BUILD.md).
The Japanese version ([XE.ja.md](XE.ja.md)) is the original; this is its translation.

XeStrata xe0.1.39 runs Strata's engine on Intel GPUs through Level Zero and SYCL; its version follows the upstream version it has integrated.
It is ported from Strata 0.1.24 (`3ce2523c2823687de5372be3af58534f56cbf286`) and carries part of the changes up to 0.1.39 (`6f32ec07`)
([Integration through Strata 0.1.38](#integration-through-strata-0138)).
The same SYCL code is compiled with intel/llvm for NVIDIA GPUs as well (the contrib-llvm build, [Build and run](BUILD.md#build-and-run)).
Upstream's CUDA build is retired and its sources are removed.
Upstream Strata (`3ce2523`) keeps them, and each Xe source names the CUDA file it ports in a comment.

It is developed and measured on an Intel Arc Pro B70 (Xe2, 32 GB); the speeds in the records are that card's unless they say otherwise.
On NVIDIA it is checked on an RTX 4070 and an RTX 3070.
The engine chooses its paths from what the GPU reports (AGENTS.md, the first rule).
The prompt path's matrix products use a kernel of their own on Xe2's XMX (xmx_gemm), a joint_matrix product that takes the reported tile shape on the Arc A series' XMX and NVIDIA's tensor cores (mma_gemm), and the DP4a paths on a GPU without matrix engines.

A smaller card is checked by making the B70 look like one.
`STRATA_VRAM_LIMIT_MIB` caps the VRAM the engine sees, `STRATA_MAX_ALLOC_MIB` the largest allocation it assumes, and
`STRATA_NO_XMX=1` takes the paths without the matrix engines.
The processor's own graphics (a UHD 770 on the development machine) is a second, real configuration for checks.

## Integration through Strata 0.1.40

The changes of upstream Strata 0.1.39 to 0.1.40.2 (`e8ca9afd`) are carried into Xe, except the Windows and AMD (HIP) implementations.
Upstream's `sycl/` (another port to Intel GPUs) is not taken in: it is measured against on the same B70.
On NVIDIA GPUs the same kernels run in the contrib-llvm build ([BUILD](BUILD.md)).

The main things carried:

- Conversations saved to a file and restored (session files), `ckpt=0` for a request that takes no checkpoint, `pin=N` to keep a shared beginning, and `tools/research_run.py`.
- The PLE table is read in the format the GGUF has it: IQ4_NL (the default table), Q4_0, Q5_0, Q5_1, Q8_0, FP8 (E4M3 with a scale) or BF16.
  Upstream measured the mean per-row error against the BF16 table: Q8_0 0.53%, FP8 2.64%, Q5_1 3.78%, Q5_0 4.25%, IQ4_NL 7.60%, Q4_0 8.55%.
- GPU kernels for the gate/up of native Q4_0, Q4_1, Q5_0 and Q6_K experts; the CPU experts' AVX-2 i-quant and Q2_0 kernels.
- Fused kernels with fewer launches in the verify windows and the prompt path (#783 and others).
- `STRATA_PREFILL_CPU_SHARE=auto` (the default): a short chunk's least-routed experts are computed by the idle CPU pool.
- `--kv-grow`: the K/V takes VRAM only for the cells a request has reached, and the expert cache uses the rest.
  It is on by default where the device's virtual memory mapping granularity is 2 MiB or more (on for the RTX 4070; the B70's is 64 KiB, off).
- `--adapt-async` (the resident mode's default), `--batch-mtp` (the default with `--batch` and `--mtp`), `--pipeline-windows` (opt-in, [MULTIGPU](MULTIGPU.md)) and `--lookup-chain` (opt-in).
- Setup's additions (`--inspect`, `--source modelscope`, a 200K context and more) and the server's changes.

Of the paths meant to be faster, those faster on the B70 or the RTX 4070 without being slower on the other are carried.
One faster on only one of them is chosen at run time from what the device reports, or is opt-in ([record](../bench/results/2026-10-09-upstream-0140/README.md)).

The integration builds in the free (intel/llvm), contrib-icpx (icpx) and contrib-llvm (intel/llvm with CUDA) modes, and all 71 CTest cases pass in each (`expert_multi_test` skips on a CPU without AVX-512).
The AVX-512 path passes `expert_multi_test` under Intel SDE (`-icx`).
`expert_parity --selftest` and `pool_test --selftest` need the canonical pack and cannot run under SDE: `unverified` (they pass on AVX2).
On the B70 the icpx and free builds completed all eight answers (thinking on and off), six of them identical.
Prompt 1,148 tok/s, decode after it 98.6 tok/s, short chats 77.3 tok/s (icpx).
The configuration with 8 GiB, 4 GiB, no XMX and six CPU workers, and the RTX 4070, completed the answers as well.
On the UHD 770 the free and icpx builds started and gave a short answer.
`--pipeline-windows` on two Intel GPUs is `unverified`.

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

The paths meant to be faster are not carried: measured on the B70, none was faster ([record](../bench/results/2026-10-04-upstream-0138/README.md)).

| Path | On the B70 |
| --- | --- |
| Quantized expert products in the prompt path (`STRATA_PREFILL_MMQ`) | 0.34x of the default XMX FP16 path |
| Fused MoE (`STRATA_PF_FUSED`) | 0.25x of the default path |
| GDN key-head sharing (`STRATA_GDN_KEYHEAD`) | 0.54x of the default path |
| Split hyper-connection read (`STRATA_GR_V3`) | Decode about 18% slower |
| P-core / E-core placement of the CPU pool (`--pool-affinity`) | Decode 1–4% slower |
| AVX2 multi-token K-quant products (`STRATA_KQ256`) | Within the spread |
| Software prefetch in the AVX2 IQ kernels | Within the spread |
| Routing-aware prefetch of the file tier (`STRATA_LOOKAHEAD`) | Decode about 2% slower |

The quantized products and the fused MoE compute on the decode kernels' DP4a dots, without XMX.
A GPU with XMX therefore reads the prompt faster on the default path.

The integration builds in both modes, free (dpclang++) and nonfree (icpx), and both pass the 52 CTest cases.
The default configuration's output (IQ2_XS, nonfree) was compared with the pre-integration build (`bae99bd`).
32 tokens from an 18-token chat were the same in both of two runs.
16 tokens from a 4,095-token prompt were the same in all three pre-integration runs.
After the integration, two of three runs matched them and one differed from the first token ([Why outputs change from run to run](#why-outputs-change-from-run-to-run)).
Under Intel SDE (`-icx`), the AVX-512 paths pass `expert_multi_test` and `native_expert_parity --synthetic` (iq3_xxs/iq4_nl, q4_K/q5_1).
`expert_parity` and `pool_test` were skipped: there is no canonical pack.
The integration commit checked that, with the 8 GiB limit, no XMX and two CPU workers, the same 16 tokens come out with and without `--coupled-draft`.
On the UHD 770 the free build answered with a two-token MTP window.
The nonfree build's four-token window stopped inside the Intel runtime and is `unverified`.
Real-model checks of every quantization format, shared arenas across processes and the resident exchanges are also `unverified`.

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
  `registered_bytes` is the whole capacity, and `locked_bytes` counts explicit `mlock` bytes (zero).
  Neither `mlock` nor a hugetlbfs pool is used; transparent huge pages back the arena.
  [The B70 measurements](../bench/results/2026-09-30-b70/README.md) registered a 47.46 GiB arena under an 8 MiB memlock limit, with `VmLck` 0.
  Behavior under memory pressure and TLB-dominated CPU access are `unverified`.
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

The Xe library exports `embedding_gather`, `gdn_gate`, `scale_inplace`, `f32_to_f16_bulk`, `silu_inplace` and `rms_norm_weighted`.
The other declarations in `elementwise.hpp` are not ported and not provided by the library.
These kernels and the parity oracle are compiled without fast math and without multiply/add contraction.

Everything icpx compiles uses `-ffp-model=precise`: icpx defaults to fast, and upstream's host code was built with g++.
All device code is built with `-foffload-fp32-prec-div -foffload-fp32-prec-sqrt`.
CUDA divides and takes square roots with IEEE rounding by default; SYCL device code does not.
`quantize_act_parity` sees the one-ulp difference in Q8_K's `1/iscale`.

| Operation | Reference and unchanged gate |
| --- | --- |
| Packed embedding row | Independent scalar unpacker; separate multiply/add rounding; bitwise equality, signed zero, guards; 108 row cases plus captured executions |
| GDN gate | CPU double softplus with the `x > 20` branch; relative L1 ≤ `1e-6`, negative gates |
| SiLU | CPU double formula; relative L1 ≤ `1e-7`; Xe uses a stable FP32 formula |
| Scaling | CPU FP32 multiplication; bitwise equality |
| FP32 → FP16 | Shared integer conversion; bitwise agreement. Independence of this check is `unverified` because CPU and GPU share the converter |
| RMS norm | CPU FP64 sum/mean, epsilon, and optional weight; relative L1 < `1e-6`, finite output, original null-weight check |

No test cases were added; fixture inputs, CPU expected values and tolerances are unchanged.
`quantize_act_parity`, which also carries NumPy-derived FP16 boundary checks, passes on the B70.
Independent GPU boundary coverage and deliberate async failure/lifetime stress are `unverified`.
New cases require approval.

### i-quants and MMVQ

All of `iq_kernels.hpp` is ported (`src/kernels/xe/iq_kernels.cpp`):
the q8_1 quantizer, dequantization of IQ1_M, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, IQ4_NL, IQ4_XS, Q3_K and Q2_0, their MMVQ,
embedding rows, the prompt path's gate/up expansion, and the grouped native experts.
A warp is a sub-group of 32; its sums use CUDA's butterfly order, and CUDA's byte intrinsics are written out.
SwiGLU in the grouped experts uses the precise exponential where CUDA used `__expf`.

All of `native_mmvq.hpp` is ported too (`src/kernels/xe/native_mmvq.cpp`):
the llama.cpp MMVQ adapters for Q2_0, Q3_K, Q4_K, Q5_K, Q6_K, Q4_0, Q5_0, Q8_0, IQ4_NL and IQ4_XS (1–8 columns, in the exact and upstream layouts), and the dispatcher.
The dispatcher sends the other i-quant types to `iq_mmvq`, as the CUDA one does.
One column also runs the multi-column kernel in the CUDA single-column layout; there are no separate single-column kernels.
The CUDA file kept the two bitwise equal as well.
Calls require an explicit runtime stream and enqueue without waiting, as the CUDA contract required a non-null stream.

[The i-quant measurements](../bench/results/2026-09-30-xe-iq/README.md) show all ten formats dequantizing real model rows exactly as gguf-py does (Q2_0: the definition mirrored from ggml).
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
Their ggml-cuda oracles are not in the tree, so agreement with them is `unverified`.
[The decode-kernel checks](../bench/results/2026-09-30-xe-kernels/README.md) record `rope_parity`, `router_top10_parity` and `bf16_gemv_parity` passing; all are registered in CTest.

## Models on the B70

Each model went through the same checks ([the procedure](../bench/results/2026-09-30-xe-iq3xxs/README.md)):

- `native_expert_parity` on layers covering every expert type pairing, and `dequant_bf16_test`
- a short prompt, and 96 greedy tokens against llama.cpp at the pinned commit on the CPU
- the three draft modes with fixed residency
- the server (streaming, a cancel, a three-turn conversation)
- a 26,293-token prompt with and without KV streaming

[The tensor inventory](../bench/results/2026-09-30-model-inventory/README.md) pins the files.

| Model | Result | Against llama.cpp: tokens before the first split (the reference's top-two gap there) | Draft modes |
| --- | --- | --- | --- |
| Qwen3.8-Flash-Next IQ2_XS | [runs](../bench/results/2026-09-30-xe-iq2xs/README.md) | 96 of 96 without drafts | the same tokens |
| Qwen3.8-Flash-Next IQ3_XXS | [runs](../bench/results/2026-09-30-xe-iq3xxs/README.md) | 6 (0.090) | differ; see below |
| Qwen3.8-Flash-Next IQ3_S | [runs](../bench/results/2026-09-30-xe-iq3s/README.md) | 61 (0.051) | no drafts = suffix; MTP differs at the same near-tie |
| Qwen3.8-Flash-Next Q2_0 | [runs](../bench/results/2026-09-30-xe-q2_0/README.md) (the native pack; the canonical AVX-512 pack is not run on this CPU) | 47 (0.050) | the same tokens |
| Qwen3.8-Flash-Next Coder IQ1_M | [runs](../bench/results/2026-09-30-xe-coder-iq1m/README.md) | 20 (0.238) | the same tokens |
| Swift 1.5 IQ2_XS | [runs](../bench/results/2026-09-30-xe-swift-iq2xs/README.md) | 7 (0.151) | no drafts = suffix; MTP differs |
| Swift 1.5 IQ3_XXS | [runs](../bench/results/2026-09-30-xe-swift-iq3xxs/README.md) | 62 (0.058, three candidates) | differ (the cache size also differed between runs) |
| Swift 1.5 Q2_0 | [runs](../bench/results/2026-09-30-xe-swift-q2_0/README.md) (note) | 25 (0.273) | no drafts = suffix; MTP differs |
| OrcaRouter IQ3_XXS (compatibility procedure) | **not run**: the repository is gated and needs a Hugging Face login to download | — | — |

Note: in Swift 1.5 Q2_0, layer 13's expert tensors span both shards.
The pack index names a shard per tensor.
Packs from before 2026-10-02 say v4, newer ones XeStrata's own xs1.
xs1 is numbered apart from upstream's v&lt;N&gt;, so neither is taken for the other.

Every split from llama.cpp comes at a near-tie of the reference; after it the comparison says nothing.
The native path writes `--dump-logits` rows from the last prompt token on (its prompt is read batched, without the head).
The records above were taken before that and compare tokens only.

Where the draft modes differ, there are two causes:

- The MTP runs had fewer cached experts (the drafter takes VRAM), so some experts ran on the CPU instead of the GPU.
- On Qwen IQ3_XXS, the suffix run differs from the no-draft run even with the same cache.
  The CPU's multi-token expert kernels (IQ3_XXS gate/up and IQ4_NL down) sum in a different order from ggml's single-token dot.
  With the CPU held to ggml's dot (`STRATA_NO_IQ256=1 STRATA_NO_IQ512=1 STRATA_NO_IQ4NL=1`) the draft modes give bit-identical logits,
  so the GPU side of a multi-token window adds no difference
  ([record](../bench/results/2026-09-30-xe-iq3xxs/README.md#the-cause-with-the-logits-added-later-on-2026-09-30)).
  The difference is kept: it moves only near-ties.

### Why outputs change from run to run

Which experts run on the CPU follows the cache size, and that changes the output's last bits.
`auto` sizes the cache from the free VRAM, so free VRAM moving between starts moves it by a slot or two and can change the output
([record](../bench/results/2026-10-02-new-machine/README.md#outputs-that-change-from-run-to-run)).
Rounding it down to 64 slots gave every start on a machine the same cache, but it slowed the RTX 4070's decode by 3-4%, so it is not rounded.
When the same output is needed (comparisons across machines or settings too), fix it with `--expert-cache N`.

### The output head's VRAM refusal

Sometimes the upload of the output head is refused VRAM at start.
About 27 GiB is free then, and it is always the first device allocation after the arena is registered for device copies.
The same settings then start normally.
On the development machine it happened 3 times in about 120 starts
([record](../bench/results/2026-10-02-new-machine/README.md#the-output-heads-vram-refusal)).
The cause is `unverified`.

### The VRAM reserve on a small card

On a small card (upstream #496), the default 700 MiB reserve can leave the cache too few slots to lend the prompt path a 256-token chunk.
The engine then lowers the reserve to what leaves that many:
down to 300 MiB, and not below a reserve given with `--vram-reserve-mib`.
When even that is not enough, the start stops and says how many MiB are short and what makes room.
Checked with `STRATA_VRAM_LIMIT_MIB`:
at 4,608 MiB (IQ2_XS, 32K, MTP) the reserve became 482 MiB, leaving the 368 slots it needs;
at 4,096 MiB the start stopped 330 MiB short.

## Images

The image encoder (`tools/vision`: llama.cpp's mtmd with the model's mmproj) runs on the GPU by default (`./setup.sh --vision yes` or `gpu`).
When it cannot, the CPU encoder takes over; `--vision cpu` uses the CPU encoder alone.
The CPU encoder reads the pictures of [the image record](../bench/results/2026-09-30-xe-vision-cpu/README.md) as llama.cpp does, on grids of both shapes.
Every model that runs was checked with images ([record](../bench/results/2026-09-30-xe-vision-models/README.md)):
two grid shapes, two images in one request, and a conversation that sends the image again and reuses its cache.

The GPU encoder comes in two builds.
Both take the B70 by its PCI address (`strata-vision --gpu-pci`, from the config's `vision.gpu_pci`):
with Vulkan the first GPU can be the processor's graphics or another card.

- **Vulkan** (ggml-vulkan, `-DSTRATA_VISION_VULKAN=ON`): the free build's.
  Its driver is Mesa, free software.
  It encodes 7–9× faster than the CPU encoder (8 threads) and about half as fast as the SYCL one.
  Its embeddings differ from the CPU encoder's by 2.4–3.1% (relative L2), less than the SYCL encoder's, and repeat bit for bit
  ([record](../bench/results/2026-10-02-vision-vulkan/README.md)).
- **SYCL** (ggml-sycl, `-DSTRATA_VISION_SYCL=ON`): in the contrib-icpx mode (`--license contrib-icpx`), when oneMKL is installed.
  ggml-sycl links oneMKL, which is not free software.
  Every node runs on the GPU; it encodes 13–20× faster than the CPU encoder, with the same answers or a rewording at a near-tie.
  Its embeddings differ from the CPU encoder's by 3–6% (relative L2); it was adopted without a tolerance for that
  ([record](../bench/results/2026-09-30-xe-vision-sycl/README.md)).

ggml-sycl takes oneDNN (`intel-oneapi-dnnl-devel`) when it is installed at build time.
With it the SYCL encoder is 1.2–1.5× faster per image, but the same image gives slightly different embeddings from run to run.
So it is off unless chosen.
`./setup.sh --vision-onednn on` writes `"onednn": true` in the config, and the server's `--vision-onednn on|off` overrides it for one start.
The server sets `GGML_SYCL_ENABLE_DNN` and `GGML_SYCL_FA_ONEDNN` for the encoder; off gives the same bits as a build without oneDNN.

The server starts the encoders in this order until one is ready:

1. the GPU encoder (with oneDNN if chosen)
1. the GPU encoder without oneDNN
1. the CPU encoder (the config's `vision.fallback`)

The console and `/v1/status` (`vision.encoder`, `vision.fallback`) say which one serves and why the ones before it did not start.
When none starts, or the serving encoder stops later, the server keeps serving text and `vision.error` says why.

## The IQ2_XS file's tensors

The [frozen model inventory](../bench/results/2026-09-30-b70/iq2-xs-baseline.json) records the Hugging Face revision and the SHA-256 of each complete GGUF header.
The [tensor CSV](../bench/results/2026-09-30-b70/iq2-xs-tensors.csv) records every tensor's actual type, shape, offset and byte size.
Only headers were fetched for that inventory; a header prefix is not a usable model shard.
Generation with the complete model, the prompt path, MTP and prompt-lookup drafts, the server, and a 26k-token prompt with KV streaming are in
[the IQ2_XS results](../bench/results/2026-09-30-xe-iq2xs/README.md).
Greedy tokens were compared with llama.cpp on the CPU.
A logit-level comparison and pack hashes are `unverified`.

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
Its per-expert relative L1 bound of `3e-2` belongs to that comparison and is not a universal kernel tolerance.
The GPU's q8_1 activation contract and the CPU's ggml `vec_dot_type` stay explicit.

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
| `qsa_prompt_attn_parity` | The existing FP64 reference and FP32 baseline, ported to SYCL (built, not a CTest case: it is also a benchmark). The XMX kernel passes it at int8 and FP16 KV ([record](../bench/results/2026-10-02-prompt-attn-xmx/README.md)), the tensor-core one at INT8, FP16 and K8V4 (Q4_0 skipped); without the matrix engines the caller keeps `qsa_decode_attn_batch` |
| `native_moe`, `native_gdn`, `native_gdn_preprocess`, `native_ple_postops`, `native_qsa`, `native_qsa_score`, `native_qsa_indexer`, `native_flash_attn`, `qsa_select`, `fused_gdn` (no in-tree parity) | [Probes against ggml-cpu at the pinned commit](../bench/results/2026-09-30-xe-native/README.md) pass on the B70; not registered in CTest |
| `native_expert_parity`, `dequant_bf16_test` | Ported, built with `STRATA_NATIVE_EXPERTS`, not registered (they need the model shards). Pass on the B70 with the shards of every model under [Models](#models-on-the-b70) |
| `iq_parity` | Ported and built, not registered. `tools/iq_fixture.py` writes `logs/iq_fixture` from ranged reads of the pinned model revision. All 10 formats pass. Q6_K/Q8_0 are not in its list |
| `ple_parity` | Has the checked-in `ple_oracle_vectors.inc`; full checks also need the actual table and reference artifacts; `unverified` |
| CPU expert/pool tests and `pool_stress` | Not in this inventory: the canonical expert tests need AVX-512 and/or a pack. The AVX-512 paths run under Intel SDE as AGENTS.md says |
| `serve.test_server`, `serve.test_detok`, `serve.test_mcp` | Pass with the mock engine |
| `tools/test_iq_pack.py`, `tools/test_shards.py`, `tools/test_calibrate.py` | Pass under `python -m unittest` (2026-10-04) |
| `src/ngram/ple_reader_test.cpp`, `src/platform/memory_test.cpp` | Built again; `ple_reader_selftest` passes, `platform_memory_test` fails under the 8 MiB memlock limit. Neither is evidence for the registered arena |

## Upstream settings not remeasured on Xe

These settings were chosen by upstream Strata on CUDA, mostly on a Ryzen 5 7600 with an RTX 5070.
XeStrata uses them as they are; only some have been tuned on Xe.

| Source | Setting | On Xe |
| --- | --- | --- |
| `src/program/generate.cpp` | Native missed-expert PCIe share 0.55 (canonical 0.2), scaled by min(1, link GB/s / 20), the link read as the best of four 256 MiB bursts | On the new development machine 0.15 was 2–4% faster and every larger share slower; the measurement does not support 0.55 ([record](../bench/results/2026-10-02-new-machine/README.md#the-expert-arena-and-the-pcie-share)) |
| `tools/calibrate.py` | Baseline Ryzen 5 7600 + RTX 5070; PCIe candidates 0/0.2/0.35/0.55/0.75, draft floor 0.3/0.5/0.7, worker candidates; interleaved confirmation requires >3% gain | Candidates and baseline unchanged from upstream |
| `setup.py` | `--spec 4`, `--spec-min-p 0.5`, `--expert-cache auto`, `--prefill auto` | `--prefill auto` now picks chunks up to 32768 tokens ([record](../bench/results/2026-10-03-prompt-upstream/README.md)); a chunk that does not fit with the default ring takes a 96-slot ring (upstream #583; IQ3_S on the B70 limited to 8 GB: 8K and 32K prompts 1.5-2x); the rest as upstream |
| `src/program/generate.cpp` | 700 MiB VRAM reserve (down to 300 on a small card); prefill borrowing and chunk selection; cache adaptation every 4 rounds and up to 96 swaps | As upstream |
| `src/program/generate.cpp` and the CPU pool | Physical-core worker heuristic plus host worker; cache-hit pokes and graph/doorbell scheduling | The worker count was measured on the new machine: 7 to 19 workers decode equally fast, so the default stays ([record](../bench/results/2026-10-02-new-machine/README.md#cpu-worker-threads)) |
| `src/kernels/cpu/iq_avx512.cpp` | Software prefetch distance and AVX-512-specific tuning | Not measured: the development machine has no AVX-512 |
