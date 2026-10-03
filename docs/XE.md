<!--
SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
SPDX-License-Identifier: LGPL-3.0-or-later
-->
# XeStrata on Intel GPUs

XeStrata 0.1.0 runs Strata's engine on Intel GPUs through Level Zero and SYCL. It is developed and measured on an Intel Arc Pro B70 (Xe2, 32 GB); the speeds in the records are that card's. It chooses its paths from what the GPU reports (AGENTS.md, the first rule): a GPU without the matrix engines (XMX) takes the DP4a paths. The setup models listed under [Models](#models-on-the-b70), the prompt path and decode, MTP and prompt-lookup drafts, KV streaming, the server, and images through the image encoder.
The upstream reference is Strata 0.1.24 at `3ce2523c2823687de5372be3af58534f56cbf286`.
The CUDA build is retired and its sources are removed; upstream Strata (`3ce2523`) keeps them, and the Xe sources name the CUDA file each one ports.
A smaller card is checked by making the B70 look like one: `STRATA_VRAM_LIMIT_MIB` caps the memory the engine sees, `STRATA_MAX_ALLOC_MIB` the largest allocation it assumes, and `STRATA_NO_XMX=1` takes the paths without the matrix engines. The processor's own graphics (a UHD 770 here) is used as a second, real configuration for checks only.

## Build and run

The engine builds in two modes (AGENTS.md, "Free and non-free builds"), chosen by `STRATA_NONFREE`:

- free (the default): intel/llvm's DPC++, free software. Validated with Ubuntu 26.04's `dpclang++` 6.2.0 (package `dpclang-6`); no oneAPI environment is needed or used. Its SYCL runtime gives the Arc Pro B70 no matrix engines (XMX) before intel/llvm 7.0, so the prompt path's products then run through DP4a on int8 blocks, about 1.6 times slower and slightly less exact ([record](../bench/results/2026-10-02-dp4a/README.md)). `tools/intel_llvm_build.py` builds a newer one from source.
- nonfree (`-DSTRATA_NONFREE=ON`): also allows Intel oneAPI's icpx (validated: 2026.1.1), which is not free software. Source oneAPI's environment before building with it.

The engine asks the device at start whether it has the matrix combinations its XMX kernels need (FP16 and BF16 8 x 16 x 16) and takes the path without them otherwise, saying so once; `STRATA_NO_XMX=1` forces that path. `tools/xmx_probe.cpp` asks the same of every GPU without building the engine.

The foundation build, free:

```bash
cmake -S . -B build/free \
  -DCMAKE_CXX_COMPILER=dpclang++ -DCMAKE_C_COMPILER=dpclang \
  -DSTRATA_ENABLE_XE=ON \
  -DSTRATA_NATIVE_EXPERTS=OFF \
  -DSTRATA_BUILD_TESTS=ON
cmake --build build/free --target strata-device strata-load elementwise_parity \
  gguf_reader_test suffix_drafter_test controller_test draft_policy_test conv_cache_test \
  platform_memory_test ple_reader_test -j4
```

The same with icpx (nonfree):

```bash
source /opt/intel/oneapi/setvars.sh
cmake -S . -B build/xe \
  -DCMAKE_CXX_COMPILER=icpx -DSTRATA_NONFREE=ON \
  -DSTRATA_ENABLE_XE=ON \
  -DSTRATA_NATIVE_EXPERTS=OFF \
  -DSTRATA_BUILD_TESTS=ON
cmake --build build/xe --target strata-device strata-load elementwise_parity \
  gguf_reader_test suffix_drafter_test controller_test draft_policy_test conv_cache_test \
  platform_memory_test ple_reader_test -j4
build/xe/strata-device --version
ctest --test-dir build/xe \
  -R '^(device_selftest|elementwise_parity|gguf_reader_test|suffix_drafter_test|controller_test|draft_policy_test|conv_cache_test|ple_reader_selftest|platform_memory_test)$' \
  --output-on-failure
```

`STRATA_NATIVE_EXPERTS=OFF` keeps this foundation build independent of downloading llama.cpp.
The unchanged native CPU expert integration remains pinned to llama.cpp `3cf03257f219afbe7334045ff7c6a06ac68c627d`.
It has not been validated with this build.
`STRATA_PORTABLE` (default ON) builds ggml-cpu for any AVX2 CPU; `-DSTRATA_PORTABLE=OFF` builds it with `-march=native` for this machine, which may stop running when the CPU is changed. On the i7-14700 both decode equally fast. setup always passes ON (an existing build folder keeps the cached value otherwise). AVX-512 translation units remain separately compiled and dispatched.
`platform_memory_test` mlocks 256 MiB and fails under the default 8 MiB `ulimit -l`; it passes only where the limit allows that lock.
The host platform library (`memory.cpp`, `direct_file.cpp`) and the PLE n-gram reader build without the GPU toolchain.
The CPU canonical expert tests require AVX-512 or model artifacts and are not covered by the command above.
CPU-only tools can be configured with `STRATA_ENABLE_XE=OFF`.

The generated `build/xe/BUILD.json` records the backend, version, upstream reference, compiler, and build scope.
Device code uses SPIR-V JIT; the driver compiles each kernel on first use and keeps it in `~/.cache/neo_compiler_cache`. AOT and distribution packaging are not used.
The first run of a model that uses kernels not compiled before takes longer: on IQ3_XXS the first token came after 30 s instead of 0.6 s ([record](../bench/results/2026-09-30-xe-iq3xxs/README.md)). setup therefore ends by starting the model once and asking it a long question and a picture, so that the first real request does not wait ([record](../bench/results/2026-09-30-xe-setup/README.md#compiling-the-gpu-code-at-setup-added-later-on-2026-09-30)); `--no-warmup` skips it.

The engine (the `strata` executable) with the native i-quant experts needs llama.cpp at the pinned commit in `third_party/main/llama.cpp` (setup.py fetches it; the image encoder also needs its `tools/mtmd`):

```bash
cmake -S . -B build/free -DCMAKE_CXX_COMPILER=dpclang++ -DCMAKE_C_COMPILER=dpclang -DSTRATA_ENABLE_XE=ON \
  -DSTRATA_NATIVE_EXPERTS=ON -DSTRATA_GGML_DIR=$PWD/third_party/main/llama.cpp
cmake --build build/free --target strata -j6
```

With icpx: `source /opt/intel/oneapi/setvars.sh`, then `-DCMAKE_CXX_COMPILER=icpx -DSTRATA_NONFREE=ON` instead.

A SYCL translation unit takes several GB to compile: a build with 16 jobs was killed for lack of memory on the validated machine (92 GiB), one with 6 was not. setup uses at most one job per 12 GB of RAM.

## Setup

`./setup.sh` checks the GPU (sysfs and the kernel driver's memory query: an Intel GPU on the xe or i915 driver, its VRAM, the render node's permissions, Resizable BAR, the PCIe link), RAM and CPU, asks the questions, chooses the SYCL compiler (below) and compiles the engine in `build-xe`. The engine goes to `engine/` with `BUILD.json` (`"backend": "xe"`, and the compiler it was built with); an engine without that entry is upstream's CUDA build and is compiled over, and an engine whose compiler changed is compiled again. The config it writes lists the runtime's library folders under `lib_dirs` (oneAPI's for icpx, the intel/llvm built here for that one), so the server starts the engine without any toolkit environment. There is no ready-made Xe engine to download, and only one GPU is used (`--gpus` is refused).

### The SYCL compiler

setup builds free unless told otherwise; `--nonfree on` also allows icpx, `--nonfree off` goes back. The choice, a `--intel-llvm DIR` and an accepted build without XMX are kept in the settings for later runs.

1. `--intel-llvm DIR` (an intel/llvm installed in DIR): that one
1. otherwise the distribution's intel/llvm (`dpclang++`, or the newest `dpclang++-N`)
1. with `--intel-llvm-build`, when there is none or it gives the GPU no XMX: intel/llvm built here from source by `tools/intel_llvm_build.py` (below)
1. otherwise it stops and says how to get one: the distribution's package, `--intel-llvm-build`, and with `--nonfree on` also icpx (the commands for Intel's apt repository)

The compiler builds `tools/xmx_probe.cpp` and runs it on the chosen GPU. When its runtime does not list the GPU at all, setup stops and names the GPU's Level Zero driver (`libze-intel-gpu1`) as missing or too old for it: openSUSE Leap 16's compute-runtime 25.18 lists no Arc Pro B70. When the GPU reports no XMX to it, `--nonfree on` takes icpx if it is installed and does better; otherwise setup asks: build without XMX (prompt processing about 1.6 times slower), build intel/llvm 7 or later here, or stop (and with `--nonfree on` and no icpx, install icpx first); building intel/llvm is not offered when the compiler is already the one built here. Without a terminal, or with `--yes`, it stops, unless `--allow-no-xmx` accepts the slower path. When the engine is compiled again after an update, it uses the compiler recorded in `BUILD.json` and asks nothing; an engine from before the free build (no record) was built with icpx and stays so, as `--nonfree on`.

`tools/intel_llvm_build.py` (run by `--intel-llvm-build`, or by hand) clones intel/llvm's release tag it names into `.tools/intel-llvm/src`, builds it (its configuration downloads the sources the release pins, Level Zero's loader and headers among them even when the distribution has older ones), and installs it into `.tools/intel-llvm/install` with a record (`XESTRATA.json`: the tag, its commit, the GPUs' XMX as the probe saw them). Run again, it uses a finished build of the same tag, asks before building a newer tag over an older one, and continues an interrupted build. The build tree is deleted when it is done unless `--intel-llvm-keep-build`; `--intel-llvm-rebuild` builds again. v7.1.1 built in 13 minutes on the development machine (28 threads) and keeps 3.5 GB; [DEVTOOLS.md](DEVTOOLS.md#intel-llvm-from-source) has the packages it needs.

### Packages

setup.py does not install system packages and does not run apt or sudo; setup.sh only installs Python 3 with venv (`python3`, `python3-venv`, through `sudo apt-get`) when there is none. The packages on the validated machine (Ubuntu 26.04.1). The free column is also what a clean Ubuntu 26.04 needs: setup ran there from start to end with only these and `python3-venv` ([record](../bench/results/2026-10-03-clean-setup/README.md)); `intel-opencl-icd` is not needed to run the engine:

| For | Free | Nonfree (`--nonfree on`) |
| --- | --- | --- |
| Building the engine | `dpclang-6` 6.2.0, `libze-dev` 1.28.2 (or intel/llvm built here, see [DEVTOOLS.md](DEVTOOLS.md#intel-llvm-from-source)) | also `intel-oneapi-compiler-dpcpp-cpp` 2026.1.1 (Intel's apt repository), `intel-ocloc` 26.05.37020.3 |
| Running it | `libze1` 1.28.2, `libze-intel-gpu1` 26.05.37020.3, `intel-opencl-icd` 26.05.37020.3 (`libze-intel-gpu-legacy1-1` 24.35 is also installed; the B70 uses the new runtime) | the same |
| The CPU image encoder | `build-essential` | the same |
| The GPU image encoder (see [Images](#images)) | Vulkan: `libvulkan-dev` 1.4.341, `glslc` 2026.1, `spirv-headers` 1.6.1, `mesa-vulkan-drivers` 26.0.8 | SYCL: `intel-oneapi-mkl-sycl-devel` 2026.1.0 |
| oneDNN for the SYCL image encoder (optional; off unless chosen) | — | `intel-oneapi-dnnl-devel` 2026.0.2 |

[The setup record](../bench/results/2026-09-30-xe-setup/README.md) runs it end to end with both image encoders (before the free build).

## Runtime contract

- A discrete card needs Above 4G Decoding and Re-Size BAR enabled and CSM disabled; otherwise the B70's BARs stayed unassigned and the xe driver did not bind.
- The runtime drives one Intel GPU through Level Zero, chosen by what it reports, never by its device ID: it needs 16-wide sub-groups, FP16 and device and host USM.
  `STRATA_GPU_PCI` (setup writes it) names the card by PCI address; without it a discrete card is taken before the processor's own graphics (Level Zero's integrated flag), then the one with the most compute units.
  It has no CPU/OpenCL fallback; a GPU without what it needs is refused with the reason.
- One context owns an in-order compute queue and an in-order transfer queue.
  `copy_async` returns an event; `compute_after` inserts the cross-queue dependency.
  For readback, pass the compute completion event to `copy_async` and wait before reading the host buffer.
  The existing parity harness executes this path using separate host USM staging allocations.
- Device allocations are 4096-byte aligned; bump allocations validate capacity without wrapping.
  `PinnedArena` is one 2 MB-aligned anonymous mapping with `MADV_HUGEPAGE`, registered with `prepare_for_device_copy` before the loader fills it; a failed mapping or registration throws.
  Host USM is not used for it: Level Zero refuses a single USM allocation above `max_mem_alloc_size` (32.5 GB on the B70), below every model's arena except the Coder's.
  Its legacy slice arguments describe loader geometry. The registration is split into pieces no larger than the device's largest allocation, ending at layer starts: past that size the UHD 770 (4 GiB) registered without an error and copied wrong bytes.
  `registered_bytes` is the whole capacity, and `locked_bytes` counts explicit `mlock` bytes (zero).
- Kernels take what the device reports: a work-group of 1024 work-items only where the device takes one (the UHD 770 takes 512), FP64 only where the device has it (the UHD 770 and the Arc A series do not: those kernels then sum in FP32).
- The verify window's GPU waits for the CPU's experts by spinning on a flag in host memory, which needs a running kernel to see the host's writes. The verifier checks that at start (`doorbell_visible`); where it fails (the UHD 770) each window runs as segments the host launches one after another, with no waiting kernel: the same results, slower. `STRATA_VERIFY_SEGMENTED=1` / `0` forces either.
- Every source, destination, and reader must remain alive until its final consumer completes.
  A copy-complete event alone does not authorize reuse while a kernel still reads the destination.
  Arena destruction conservatively drains every owned queue before freeing USM or releasing the registration.
- `create_stream` makes further in-order queues on the same context, for the work CUDA put on its own streams; `destroy_stream` drains and releases one, and `finish()` drains them all.
  A kernel's `void* stream` is resolved by `Runtime::stream`: an engine stream, the compute queue, or null (the compute queue, with synchronous completion where CUDA synchronized).
  Any other pointer is refused. Work on different streams is ordered only through events, as with CUDA's non-blocking streams.
- Synchronous SYCL failures propagate to the CLI; asynchronous errors are retained and checked at waits.
  Waits block in the driver; a watchdog thread reports the named wait and terminates the process when one lasts 120 s.
  The xe driver resets a job after `job_timeout_ms` (5 s here), so a healthy wait returns well inside that limit.
  A hang or unsafe teardown failure terminates the process without unwinding through live GPU allocations.
  Driver job timeouts are not modified; CTest uses an outer 60-second timeout.
- `mlock` and a hugetlbfs pool are not used; transparent huge pages back the arena.
  The [B70 measurements](../bench/results/2026-09-30-b70/README.md) establish a 47.46 GiB arena with an 8 MiB memlock limit and `VmLck` 0.
  Behavior under memory pressure and TLB-dominated CPU access remain `unverified`.

USM pointers must be used in their allocation context and their dependencies remain the application's responsibility.
See the [Khronos USM reference](https://github.com/KhronosGroup/SYCL_Reference/blob/main/source/iface/usm_basic_concept.rst).
The parity harness also preserves its small embedding-plus-scale capture/replay check using the experimental [oneAPI graph extension](https://github.com/intel/llvm/blob/sycl/sycl/doc/extensions/experimental/sycl_ext_oneapi_graph.asciidoc).
This does not validate a whole-token graph or a CPU/GPU doorbell protocol.

## Implemented arithmetic

The Xe library exports `embedding_gather`, `gdn_gate`, `scale_inplace`, `f32_to_f16_bulk`, `silu_inplace`, and `rms_norm_weighted`.
Other declarations in `elementwise.hpp` have not been ported and are not provided by the library.
Compilation disables fast math and multiply/add contraction for these kernels and the parity oracle.
Everything icpx compiles uses `-ffp-model=precise` (icpx defaults to fast; upstream's host code was built with g++), and all device code is built with `-foffload-fp32-prec-div -foffload-fp32-prec-sqrt`: CUDA divides and takes square roots with IEEE rounding by default, SYCL device code does not, and `quantize_act_parity` sees the one-ulp difference in Q8_K's `1/iscale`.

All of `iq_kernels.hpp` is ported (`src/kernels/xe/iq_kernels.cpp`): the q8_1 quantizer, dequantization of IQ1_M, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, IQ4_NL, IQ4_XS, Q3_K and Q2_0, their MMVQ, embedding rows, the prompt path's gate/up expansion, and the grouped native experts.
A warp is a sub-group of 32; its sums use CUDA's butterfly order, and CUDA's byte intrinsics are written out.
SwiGLU in the grouped experts uses the precise exponential where CUDA used `__expf`.
All of `native_mmvq.hpp` is ported (`src/kernels/xe/native_mmvq.cpp`): the llama.cpp MMVQ adapters for Q2_0, Q3_K, Q4_K, Q5_K, Q6_K, Q4_0, Q5_0, Q8_0, IQ4_NL and IQ4_XS, 1–8 columns in the exact and upstream layouts, and the dispatcher, which sends the other i-quant types to `iq_mmvq` as the CUDA one does.
The multi-column kernel with one column and the CUDA single-column layout replaces the separate single-column kernels, which the CUDA file kept bitwise equal to it.
Calls require an explicit runtime stream and enqueue without waiting, as the CUDA contract requires a non-null stream.

[The i-quant measurements](../bench/results/2026-09-30-xe-iq/README.md) show all ten formats dequantizing real model rows exactly as gguf-py (Q2_0: the definition mirrored from ggml) and their MMVQ within `iq_parity`'s `2e-2` bound (4.2e-3–5.8e-3); `iq_parity` reports 0 failures.
Q6_K, Q8_0, Q4_K, Q5_K, Q4_0 and Q5_0 MMVQ are within `2e-2` of FP64 on random blocks, and ncols 2–8 are bitwise equal to single-column calls in the exact layout.
The grouped expert path remains `unverified`.

`src/kernels/xe/rope.cpp` ports `rope.hpp`, `native_rope.hpp` and the M-RoPE table of `mrope.hpp` (one pointer: one Xe device); `src/kernels/xe/router.cpp` ports `router_top10.hpp` and `native_router.hpp`; `src/kernels/xe/bf16_gemv.cpp` ports `bf16_gemv.hpp`, including the ggml mmvf kernels.
`src/kernels/xe/quantize_act.cpp`, `qsa.cpp`, `kv_q8.cpp`, `kv_q4.cpp`, `qsa_decode_attn.cpp` and `kv_stream.cpp` port the headers of the same names; the QSA entry points throw `DeviceError` where CUDA exited.
The KV host pools (`KvHostPools`) are read and written by kernels, so on Xe they must be host USM; memory registered only for device copies is not kernel-addressable.
`native_rope` and `native_router` use precise `pow`/`cos`/`sin`/`exp` where CUDA used fast math, and their ggml-cuda oracles are not in the tree, so they are `unverified` against it.
[The decode-kernel checks](../bench/results/2026-09-30-xe-kernels/README.md) record `rope_parity`, `router_top10_parity` and `bf16_gemv_parity` passing, now registered in CTest.

| Operation | Reference and unchanged gate |
| --- | --- |
| Packed embedding row | Independent scalar unpacker; separate multiply/add rounding; bitwise equality, signed zero, guards; 108 row cases plus captured executions |
| GDN gate | CPU double softplus with the `x > 20` branch; relative L1 ≤ `1e-6`, negative gates |
| SiLU | CPU double formula; relative L1 ≤ `1e-7`; Xe uses a stable FP32 formula |
| Scaling | CPU FP32 multiplication; bitwise equality |
| FP32 → FP16 | Shared integer conversion; bitwise agreement. Independence of this check is **unverified** because CPU and GPU share the converter |
| RMS norm | CPU FP64 sum/mean, epsilon, and optional weight; relative L1 < `1e-6`, finite output, original null-weight check |

No new test cases were added, and fixture inputs, CPU expected values, and tolerances were retained.
`quantize_act_parity`, which also carries NumPy-derived FP16 boundary checks, passes on the B70.
Independent GPU boundary coverage and deliberate async failure/lifetime stress remain `unverified`; new cases require approval.

## Models on the B70

Each model went through the same checks ([the procedure](../bench/results/2026-09-30-xe-iq3xxs/README.md)): `native_expert_parity` on layers covering every expert type pairing, `dequant_bf16_test`, a short prompt, 96 greedy tokens against llama.cpp at the pinned commit on the CPU, the three draft modes with fixed residency, the server (streaming, a cancel, a three-turn conversation), and a 26,293-token prompt with and without KV streaming. [The tensor inventory](../bench/results/2026-09-30-model-inventory/README.md) pins the files.

| Model | Result | Against llama.cpp: tokens before the first split (the reference's top-two gap there) | Draft modes |
| --- | --- | --- | --- |
| Qwen3.8-Flash-Next IQ2_XS | [runs](../bench/results/2026-09-30-xe-iq2xs/README.md) | 96 of 96 without drafts | the same tokens |
| Qwen3.8-Flash-Next IQ3_XXS | [runs](../bench/results/2026-09-30-xe-iq3xxs/README.md) | 6 (0.090) | differ; see below |
| Qwen3.8-Flash-Next IQ3_S | [runs](../bench/results/2026-09-30-xe-iq3s/README.md) | 61 (0.051) | no drafts = suffix; MTP differs at the same near-tie |
| Qwen3.8-Flash-Next Q2_0 | [runs](../bench/results/2026-09-30-xe-q2_0/README.md) (the native pack; the canonical AVX-512 pack is not run on this CPU) | 47 (0.050) | the same tokens |
| Qwen3.8-Flash-Next Coder IQ1_M | [runs](../bench/results/2026-09-30-xe-coder-iq1m/README.md) | 20 (0.238) | the same tokens |
| Swift 1.5 IQ2_XS | [runs](../bench/results/2026-09-30-xe-swift-iq2xs/README.md) | 7 (0.151) | no drafts = suffix; MTP differs |
| Swift 1.5 IQ3_XXS | [runs](../bench/results/2026-09-30-xe-swift-iq3xxs/README.md) | 62 (0.058, three candidates) | differ (the cache size also differed between runs) |
| Swift 1.5 Q2_0 | [runs](../bench/results/2026-09-30-xe-swift-q2_0/README.md) (layer 13's expert tensors span both shards: the pack index names a shard per tensor; packs from before 2026-10-02 say v4, newer ones XeStrata's own xs1 - numbered apart from upstream's v<N>, so neither is taken for the other) | 25 (0.273) | no drafts = suffix; MTP differs |
| OrcaRouter IQ3_XXS (compatibility procedure) | **not run**: the repository is gated and needs a Hugging Face login to download | — | — |

Every split from llama.cpp comes at a near-tie of the reference; after it the comparison says nothing. The native path writes `--dump-logits` rows from the last prompt token on (its prompt is read batched, without the head); the records above were taken before that and compare tokens only.
Where the draft modes differ, the MTP runs had fewer cached experts (the drafter's VRAM), so some experts ran on the CPU instead of the GPU. On Qwen IQ3_XXS, the suffix run differs from the no-draft run with the same cache: the CPU's multi-token expert kernels (IQ3_XXS gate/up and IQ4_NL down) sum in a different order from ggml's single-token dot. With the CPU held to ggml's dot (`STRATA_NO_IQ256=1 STRATA_NO_IQ512=1 STRATA_NO_IQ4NL=1`) the draft modes give bit-identical logits, so the GPU side of a multi-token window adds no difference ([record](../bench/results/2026-09-30-xe-iq3xxs/README.md#the-cause-with-the-logits-added-later-on-2026-09-30)). The difference is kept: it moves only near-ties.
Sometimes the engine's upload of the output head is refused VRAM at start, with about 27 GiB free, always as the first device allocation after the expert arena is registered for device copies; the same run then starts normally (twice on the old machine, 3 times in about 120 starts on the new one: [record](../bench/results/2026-10-02-new-machine/README.md#the-output-heads-vram-refusal); cause `unverified`).
Which experts run on the CPU follows the cache size, and that changes the output's last bits. `auto` sizes the cache from the free VRAM rounded down (64 slots, or 256 MiB when slots are sized per expert), so starts on the same machine get the same cache; before the rounding, free VRAM moved it by a slot or two ([record](../bench/results/2026-10-02-new-machine/README.md#outputs-that-change-from-run-to-run)). To compare outputs across machines or settings, fix it with `--expert-cache N`.
On a small card (upstream #496), when the default 700 MiB reserve leaves the cache too few slots to lend the prompt path a 256-token chunk, the engine lowers the reserve (down to 300 MiB, not below a reserve given with `--vram-reserve-mib`) to what leaves that many; when even that is not enough, the start stops and says how many MiB are short and what makes room. Checked with `STRATA_VRAM_LIMIT_MIB`: at 4,608 MiB (IQ2_XS, 32K, MTP) the reserve became 482 MiB and the 368 slots it needs; at 4,096 the start stopped 330 MiB short.

## Images

The image encoder (`tools/vision`, llama.cpp's mtmd with the model's mmproj) runs on the GPU by default (`./setup.sh --vision yes` or `gpu`), with the CPU encoder as its fallback; `--vision cpu` uses the CPU encoder alone. The CPU encoder reads the pictures of [the image record](../bench/results/2026-09-30-xe-vision-cpu/README.md) as llama.cpp does, on grids of both shapes. Every model that runs was checked with images ([record](../bench/results/2026-09-30-xe-vision-models/README.md)): two grid shapes, two images in one request, and a conversation that sends the image again and reuses its cache.

The GPU encoder is one of two builds, and takes the B70 by its PCI address (`strata-vision --gpu-pci`, from the config's `vision.gpu_pci`): with Vulkan the first GPU can be the CPU's integrated graphics or another card.

- Vulkan (ggml-vulkan, `-DSTRATA_VISION_VULKAN=ON`), the free build's: Mesa's driver, free software. It encodes 7–9× faster than the CPU encoder (8 threads) and about half as fast as the SYCL one; its embeddings differ from the CPU encoder's by 2.4–3.1% (relative L2), less than the SYCL encoder's, and repeat bit for bit ([record](../bench/results/2026-10-02-vision-vulkan/README.md)).
- SYCL (ggml-sycl, `-DSTRATA_VISION_SYCL=ON`), with `--nonfree on` when icpx and oneMKL are installed (ggml-sycl links oneMKL, not free software): every node runs on the GPU and it encodes 13–20× faster than the CPU encoder, with the same answers or a rewording at a near-tie. Its embeddings differ from the CPU encoder's by 3–6% (relative L2); it was adopted without a tolerance for that ([record](../bench/results/2026-09-30-xe-vision-sycl/README.md)).

ggml-sycl takes oneDNN when it is installed at build time (`intel-oneapi-dnnl-devel`). With it the SYCL encoder is 1.2–1.5× faster per image, but the same image gives slightly different embeddings from run to run, so it is off unless chosen: `./setup.sh --vision-onednn on` writes `"onednn": true` in the config, and the server's `--vision-onednn on|off` overrides it for one start (the server sets `GGML_SYCL_ENABLE_DNN` and `GGML_SYCL_FA_ONEDNN` for the encoder; off gives the same bits as a build without oneDNN).
The server starts the encoders in order until one is ready: the GPU encoder (with oneDNN if chosen), the GPU encoder without oneDNN, then the CPU encoder (the config's `vision.fallback`). The console and `/v1/status` (`vision.encoder`, `vision.fallback`) say which one serves and why the ones before it did not start. When none starts, or the serving encoder stops later, the server keeps serving text and `vision.error` says why.

## IQ2_XS baseline and next kernels

The [frozen model inventory](../bench/results/2026-09-30-b70/iq2-xs-baseline.json) records the Hugging Face revision and SHA-256 of each complete GGUF header.
The [tensor CSV](../bench/results/2026-09-30-b70/iq2-xs-tensors.csv) records every tensor's actual type, shape, offset, and byte size.
Only headers were fetched for that inventory; a header prefix is not a usable model shard.
The engine has since run the complete model: generation, the prompt path, MTP and prompt-lookup drafts, the server, and a 26k-token prompt with KV streaming are recorded in [the IQ2_XS results](../bench/results/2026-09-30-xe-iq2xs/README.md).
Greedy tokens were compared with llama.cpp on the CPU; a logit-level comparison and pack hashes remain `unverified`.

The `IQ2_XS` distribution contains 1,224 tensors across 12 storage types, with **zero tensors stored as IQ2_XS** at the recorded revision.
The name describes the quantization budget of a mixed-precision model.

| Tensor role | Observed storage | Required operations / next reference |
| --- | --- | --- |
| Routed expert gate and up | IQ2_S in 34 layers, IQ2_XXS in 11, IQ1_M in 3 | Per-format decode, activation quantization, selected rows/GEMV and batched GEMM; pinned ggml `to_float` and CPU FP64 math |
| Routed expert down | Q2_0 in all 48 layers | Native Q2_0 decode/dot and CPU AVX2 path; same pinned ggml expert oracle |
| Quantized dense/shared expert weights | IQ3_S, IQ4_XS, IQ4_NL, Q6_K, Q8_0, Q2_0 | Per-type decode, dense GEMV/GEMM and SwiGLU; pinned ggml plus FP64 accumulation |
| Token embedding and output head | IQ4_XS | Row decode, projection, sampling; pinned ggml decode and host arithmetic |
| HC/router/SSM/indexer/PLE projections | BF16 | BF16 activation rounding and projection; existing BF16/GR/router/GDN/QSA parity |
| Norms, biases, recurrence parameters | F32; PLE convolution F16 | RMS, gate, convolution, recurrence, indexing; existing host references |
| PLE n-gram table, second shard | IQ4_NL | Integer n-gram hash, selected row reads/decode; existing PLE vectors and actual table rows |

The native expert oracle in `src/kernels/native_expert_parity.cpp` uses ggml dequantization and host FP64 dot products before comparing CPU and GPU paths.
Its existing per-expert relative L1 bound is `3e-2`, not a universal kernel tolerance.
The GPU's q8_1 activation contract and the CPU's ggml `vec_dot_type` must remain explicit.
The former prefill MMQ build also depended on ggml-cuda; retaining ggml-cpu alone does not replace it.
Choosing ggml-sycl versus custom Xe MMQ and introducing oneMKL dense baselines belong to the following kernel phase.

## Existing validation inventory

| Existing assets | Current status and limits |
| --- | --- |
| `strata-device --selftest`, `elementwise_parity` | Ported; pass on B70. The device selftest checks allocation/alignment/capacity, not numerical poison readback |
| `gguf_reader_test`, `suffix_drafter_test`, `controller_test`, `draft_policy_test`, `conv_cache_test` | Pass with the current toolchain; they do not establish full engine integration |
| `s_gemv_parity`, `s_gemv_q8k_parity` | Ported, registered in CTest, pass on B70; `--bench` also checks `s_gemv_split`, `s2_gemv_quads` and `s2_gemv_fast` against the naive kernel |
| `dequant_s2_parity`, `s2_gemv_parity`, `s2_gemv_q8_parity` | Ported, registered in CTest, pass on B70 |
| `rope_parity`, `router_top10_parity`, `bf16_gemv_parity`, `quantize_act_parity`, `qsa_parity`, `kv_q8_parity`, `kv_q4_parity`, `kv_stream_parity`, `gdn_parity`, `gr_parity`, `cvec_parity` | Ported, registered in CTest, pass on B70 |
| `shared_expert_parity` | Ported, registered in CTest, passes on B70, including the SYCL-graph replay of the native scalar gate |
| `sampler_parity` | Ported, registered in CTest, passes on B70 |
| `qsa_prompt_attn_parity` | Existing FP64 reference and FP32 baseline, ported to SYCL (built, not a CTest case: it is also a benchmark). The XMX kernel passes it at int8 and FP16 KV ([record](../bench/results/2026-10-02-prompt-attn-xmx/README.md)); without the matrix engines the caller keeps `qsa_decode_attn_batch` |
| `native_moe`, `native_gdn`, `native_gdn_preprocess`, `native_ple_postops`, `native_qsa`, `native_qsa_score`, `native_qsa_indexer`, `native_flash_attn`, `qsa_select`, `fused_gdn` (no in-tree parity) | [Probes against ggml-cpu at the pinned commit](../bench/results/2026-09-30-xe-native/README.md) pass on B70; not registered in CTest |
| `native_expert_parity`, `dequant_bf16_test` | Ported, built with `STRATA_NATIVE_EXPERTS`, not registered (they need the model shards). Pass on B70 with the shards of every model under [Models](#models-on-the-b70) |
| `iq_parity` | Ported and built, not registered. `tools/iq_fixture.py` writes `logs/iq_fixture` from ranged reads of the pinned model revision. All 10 formats pass. Q6_K/Q8_0 are not in its list |
| `ple_parity` | Has checked-in `ple_oracle_vectors.inc`; full checks also require the actual table and reference artifacts; `unverified` |
| CPU expert/pool tests and `pool_stress` | Not executed in this foundation gate: canonical expert tests require AVX-512 and/or a pack; native IQ oracle remains pending |
| `serve.test_server`, `serve.test_detok`, `serve.test_mcp` | Pass (65 run, 3 skipped) with the mock engine |
| `tools/test_iq_pack.py`, `tools/test_shards.py`, `tools/test_calibrate.py` | Not run |
| `src/ngram/ple_reader_test.cpp`, `src/platform/memory_test.cpp` | Built again; `ple_reader_selftest` passes, `platform_memory_test` fails under the 8 MiB memlock limit. Neither is evidence for the registered arena |
| `tests/`, `bench/micro/`, Python references other than `ref/load.py` | Missing from the published tree; not counted as passing or available oracles |

## NVIDIA tuning inventory

These are upstream settings to remeasure during model integration, not calibrated Xe defaults.
The current foundation does not use them.

| Source | Setting / assumption |
| --- | --- |
| `src/program/generate.cpp` | Native missed-expert PCIe share 0.55, canonical share 0.2; native share scaled by min(1, link GB/s / 20), the link read as the best of four 256 MiB bursts |
| `tools/calibrate.py` | Baseline Ryzen 5 7600 + RTX 5070; PCIe candidates 0/0.2/0.35/0.55/0.75, draft floor 0.3/0.5/0.7, worker candidates; interleaved confirmation requires >3% gain |
| `setup.py` | `--spec 4`, `--spec-min-p 0.5`, `--expert-cache auto`, `--prefill auto`; these must be revisited with the Xe engine version checks |
| `src/program/generate.cpp` | 700 MiB VRAM reserve (down to 300 on a small card); prefill borrowing/chunk selection; cache adaptation every 4 rounds and up to 96 swaps |
| `src/program/generate.cpp` and CPU pool | Physical-core worker heuristic plus host worker; cache-hit pokes and graph/doorbell scheduling |
| `src/kernels/cpu/iq_avx512.cpp` | Software prefetch distance and AVX-512-specific tuning; cannot be assumed beneficial on this AVX2 CPU |
| Current PLE source | No `--no-ple-prefetch` switch exists in this checkout. Historical Windows timings are not a Linux/Xe prefetch baseline |

Next: use the frozen mixed-format inventory to port dequantization and native expert/dense operations, retaining independent oracles and per-operation tolerances.
Model-level reference outputs require obtaining the full pinned model and pack; that validation remains `unverified`.
