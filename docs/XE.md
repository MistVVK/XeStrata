<!--
SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
SPDX-License-Identifier: LGPL-3.0-or-later
-->
# XeStrata on Intel GPUs

English | [日本語](XE.ja.md)

This document records how XeStrata's engine runs on Intel GPUs and how that was checked: how to build it, what setup does, the packages it needs, what the engine requires of the GPU, and the ported arithmetic and its validation.
How to use it is in the [README](../README.md) and the [details](DETAILS.md).
The Japanese version ([XE.ja.md](XE.ja.md)) is the original; this is its translation.

XeStrata 0.1.0 runs Strata's engine on Intel GPUs through Level Zero and SYCL.
It is ported from Strata 0.1.24 (`3ce2523c2823687de5372be3af58534f56cbf286`).
The CUDA build is retired and its sources are removed.
Upstream Strata (`3ce2523`) keeps them, and each Xe source names the CUDA file it ports in a comment.

It is developed and measured on an Intel Arc Pro B70 (Xe2, 32 GB); the speeds in the records are that card's.
The engine chooses its paths from what the GPU reports (AGENTS.md, the first rule).
A GPU without the matrix engines (XMX) takes the DP4a paths.

A smaller card is checked by making the B70 look like one.
`STRATA_VRAM_LIMIT_MIB` caps the VRAM the engine sees, `STRATA_MAX_ALLOC_MIB` the largest allocation it assumes, and
`STRATA_NO_XMX=1` takes the paths without the matrix engines.
The processor's own graphics (a UHD 770 on the development machine) is a second, real configuration for checks.

## Build and run

The engine builds in two modes (AGENTS.md, "Free and non-free builds"), chosen by the CMake option `STRATA_NONFREE`.

- **free** (the default): built with intel/llvm's DPC++.
  Free software only; no oneAPI environment is used.
  Validated with Ubuntu 26.04's `dpclang++` 6.2.0 (package `dpclang-6`).
  A SYCL runtime before intel/llvm 7.0, though, reports no XMX for the Arc Pro B70.
  The prompt path's products then run through DP4a on int8 blocks, take about 1.6 times as long and are slightly less exact
  ([record](../bench/results/2026-10-02-dp4a/README.md)).
  `tools/intel_llvm_build.py` builds a newer intel/llvm from source (see [Setup](#setup)).
- **nonfree** (`-DSTRATA_NONFREE=ON`): also allows Intel oneAPI's icpx, which is not free software (validated: 2026.1.1).
  Source oneAPI's environment before building.

At start the engine asks the GPU whether it has the matrix combinations its XMX kernels need (FP16 and BF16 8 x 16 x 16).
Without them it takes the path without XMX and says so once.
`STRATA_NO_XMX=1` takes that path even when the GPU has XMX.
`tools/xmx_probe.cpp` asks every GPU the same without building the engine.

### The engine's parts and tests

The parts and tests that need no model build without llama.cpp (`STRATA_NATIVE_EXPERTS=OFF`).
The free mode:

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

With icpx (nonfree):

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

- `platform_memory_test` mlocks 256 MiB and fails under the default 8 MiB `ulimit -l`.
  It passes only where that limit is raised.
- The host platform library (`memory.cpp`, `direct_file.cpp`) and the PLE n-gram reader build without the GPU toolchain.
  CPU-only tools can be configured with `STRATA_ENABLE_XE=OFF`.
- The CPU canonical expert tests need AVX-512 or model files and are not in the commands above.
- `build/xe/BUILD.json` records the backend, version, upstream reference, compiler and build scope.

### The engine

The engine (the `strata` executable) uses llama.cpp for the native i-quant experts.
llama.cpp goes in `third_party/main/llama.cpp` at the pinned commit (`3cf03257f219afbe7334045ff7c6a06ac68c627d`).
setup.py fetches it; the image encoder also uses its `tools/mtmd`.

```bash
cmake -S . -B build/free -DCMAKE_CXX_COMPILER=dpclang++ -DCMAKE_C_COMPILER=dpclang -DSTRATA_ENABLE_XE=ON \
  -DSTRATA_NATIVE_EXPERTS=ON -DSTRATA_GGML_DIR=$PWD/third_party/main/llama.cpp
cmake --build build/free --target strata -j6
```

With icpx: `source /opt/intel/oneapi/setvars.sh`, then `-DCMAKE_CXX_COMPILER=icpx -DSTRATA_NONFREE=ON` instead.

- **Jobs**: one SYCL translation unit takes several GB to compile.
  On the validated machine (92 GiB) a build with 16 jobs was killed for lack of memory, one with 6 was not.
  setup uses at most one job per 12 GB of RAM.
- **`STRATA_PORTABLE`** (default ON): builds ggml-cpu for any CPU with AVX2.
  `-DSTRATA_PORTABLE=OFF` builds it with `-march=native` for this machine, which may stop running when the CPU is changed.
  On the i7-14700 both decode equally fast.
  setup always passes ON (an existing build folder keeps its cached value otherwise).
  The AVX-512 translation units are compiled separately and chosen at run time.
- **GPU code**: device code is SPIR-V, compiled by the driver (JIT) when each kernel is first used and kept in `~/.cache/neo_compiler_cache`.
  No AOT and no distribution packaging.
  So the first run of a model whose kernels were not compiled before takes longer:
  on IQ3_XXS the first token came after 30 s instead of 0.6 s ([record](../bench/results/2026-09-30-xe-iq3xxs/README.md)).
  setup therefore ends by starting the model once and asking it a long question and a picture
  ([record](../bench/results/2026-09-30-xe-setup/README.md#compiling-the-gpu-code-at-setup-added-later-on-2026-09-30));
  `--no-warmup` skips it.

## Setup

`./setup.sh` goes through these steps:

1. It checks the GPU: through sysfs and the kernel driver's memory query, an Intel GPU on the xe or i915 driver,
   its VRAM, the render node's permissions, Resizable BAR and the PCIe link.
1. It checks RAM and CPU and asks its questions.
1. It chooses the SYCL compiler ([below](#the-sycl-compiler)) and compiles the engine in `build-xe`.

The engine goes to `engine/` with `BUILD.json` (`"backend": "xe"` and the compiler it was built with).
An engine without that record is taken for upstream's CUDA build and compiled over.
An engine whose compiler changed is compiled again.
The config setup writes lists the runtime's library folders under `lib_dirs`:
oneAPI's for icpx, the intel/llvm built here for that one.
So the server starts the engine without any toolkit environment.

There is no ready-made Xe engine to download.
Only one GPU is used; `--gpus` is refused.

### The SYCL compiler

setup builds free unless told otherwise.
`--nonfree on` also allows icpx; `--nonfree off` goes back.
That choice, a `--intel-llvm DIR` and an accepted build without XMX are kept in the settings for later runs.

The compiler is chosen in this order:

1. With `--intel-llvm DIR`: the intel/llvm installed in DIR.
1. Otherwise the distribution's intel/llvm (`dpclang++`, or the newest `dpclang++-N`).
1. With `--intel-llvm-build`, when neither exists or the GPU reports no XMX to it: intel/llvm built here from source.
1. Otherwise setup stops and says how to get one:
   the distribution's package, `--intel-llvm-build`, and with `--nonfree on` icpx (the commands for Intel's apt repository).

The chosen compiler builds `tools/xmx_probe.cpp` and runs it on the chosen GPU.
What happens next depends on the result:

- **The GPU is not listed at all**: setup stops.
  It names the GPU's Level Zero driver (`libze-intel-gpu1`, Intel's compute-runtime) as missing or too old for the GPU.
  openSUSE Leap 16's compute-runtime 25.18 lists no Arc Pro B70.
- **The GPU is listed without XMX**: with `--nonfree on`, an installed icpx that reports XMX is taken.
  Otherwise setup asks: build without XMX, build intel/llvm 7 or later here, or stop.
  Building intel/llvm is not offered when the compiler already is the one built here.
  With `--nonfree on` and no icpx, it also offers to install icpx.
  Without a terminal, or with `--yes`, it stops;
  `--allow-no-xmx` accepts the slower path.

When the engine is compiled again after an update, setup uses the compiler recorded in `BUILD.json` and asks nothing.
An engine from before the free build (no record) was built with icpx and stays so, as `--nonfree on`.

`tools/intel_llvm_build.py` is run by `--intel-llvm-build` or by hand.
It clones the intel/llvm release tag the script names into `.tools/intel-llvm/src`, builds it, and installs it into `.tools/intel-llvm/install`.
Its configuration downloads the sources the release pins.
Level Zero's loader and headers are among them, fetched even when the distribution has older ones.
The install keeps a record (`XESTRATA.json`: the tag, its commit, each GPU's XMX as the probe saw it).

- Run again, it uses a finished build of the same tag.
- It asks before building a newer tag over an older build.
- It continues an interrupted build.
- The build tree is deleted when it is done; `--intel-llvm-keep-build` keeps it, `--intel-llvm-rebuild` builds again.

v7.1.1 built in 13 minutes on the development machine (28 threads) and keeps 3.5 GB.
The packages it needs are in [the table below](#packages) and in [DEVTOOLS.md](DEVTOOLS.md#intelllvm-from-source).

### Packages

setup.py installs no system packages and runs neither apt nor sudo.
setup.sh only installs Python 3 with venv, through `sudo apt-get` (`sudo dnf` on Fedora), when there is none.
Install the other packages first.

#### Ubuntu 26.04

The packages on the development machine (Ubuntu 26.04.1).
The free column is also what a clean Ubuntu 26.04 needs:
in a container with only this column and `python3-venv`, setup ran from start to end ([record](../bench/results/2026-10-03-clean-setup/README.md)).
`intel-opencl-icd` is not needed to run the engine.

| For | Free | Nonfree (`--nonfree on`) |
| --- | --- | --- |
| Building the engine | `dpclang-6` 6.2.0, `libze-dev` 1.28.2 | also `intel-oneapi-compiler-dpcpp-cpp` 2026.1.1 (Intel's apt repository), `intel-ocloc` 26.05.37020.3 |
| XMX on the B70 (`--intel-llvm-build`) | `git`, `cmake`, `ninja-build`, `g++`, `libhwloc-dev` | (not needed: icpx gives XMX) |
| Running it | `libze1` 1.28.2, `libze-intel-gpu1` 26.05.37020.3, `intel-opencl-icd` 26.05.37020.3 (`libze-intel-gpu-legacy1-1` 24.35 is also installed; the B70 uses the new runtime) | the same |
| The CPU image encoder | `build-essential` | the same |
| The GPU image encoder (see [Images](#images)) | Vulkan: `libvulkan-dev` 1.4.341, `glslc` 2026.1, `spirv-headers` 1.6.1, `mesa-vulkan-drivers` 26.0.8 | SYCL: `intel-oneapi-mkl-sycl-devel` 2026.1.0 |
| oneDNN for the SYCL image encoder (optional; off unless chosen) | — | `intel-oneapi-dnnl-devel` 2026.0.2 |

Ubuntu 26.04's `dpclang-6` 6.2.0 gives the B70 no XMX.
For XMX, build intel/llvm with `--intel-llvm-build`, or use icpx in the nonfree mode.
To go on with `dpclang-6`, pass `--allow-no-xmx`.

[The setup record](../bench/results/2026-09-30-xe-setup/README.md) ran setup end to end with both image encoders, before the free build existed.

#### Fedora 44

Fedora 44 has no DPC++ package, so the free mode builds intel/llvm with `--intel-llvm-build`.
On 2026-10-04 setup ran from start to end in a Fedora 44 container on the development machine
(Ubuntu 26.04's kernel 7.0 and xe driver), with only these packages installed.
intel/llvm v7.1.1 built in 14 minutes, the B70 got XMX, and the model it started answered a question.

| For | Packages (versions checked) |
| --- | --- |
| Python | `python3` 3.14.7 (setup.sh can also install it through `dnf`) |
| Building intel/llvm | `git`, `cmake` 4.3.0, `ninja-build`, `gcc-c++` 16.2.1, `hwloc-devel` |
| Building the engine | `oneapi-level-zero-devel` 1.33.1 |
| Running it | `oneapi-level-zero` 1.33.1, `intel-level-zero` 26.35.39758.11 (the Level Zero driver for Intel GPUs) |

`intel-compute-runtime` (OpenCL) was installed too; as on Ubuntu, the engine should not need it (`unverified`).
The packages for the image encoders and running on Fedora's own kernel are not checked (`unverified`).

#### Other distributions

openSUSE Leap 16.0 was tried the same way on the same day.
It does not run as it is:
its compute-runtime (`libze_intel_gpu1`) is 25.18, too old to list the Arc Pro B70.
intel/llvm builds, but setup stops because it sees no GPU.

What any distribution needs:

- Python 3.10 or newer with venv
- a Level Zero driver that lists the GPU (for the B70, a compute-runtime of the 26 series, for example)
- the Level Zero headers (`/usr/include/level_zero/ze_api.h`)
- a SYCL compiler: the distribution's DPC++, or the build tools `--intel-llvm-build` needs

## Runtime contract

What the engine requires of the GPU and the host, and the rules it keeps.

- **BIOS**: a discrete card needs Above 4G Decoding and Re-Size BAR enabled and CSM disabled.
  Otherwise the B70's BARs stayed unassigned and the xe driver did not bind.
- **Choosing the GPU**: the runtime drives one Intel GPU through Level Zero, chosen by what it reports, never by its device ID.
  It needs 16-wide sub-groups, FP16, and device and host USM.
  `STRATA_GPU_PCI` (setup writes it) names the card by PCI address.
  Without it, a discrete card is taken before the processor's own graphics (Level Zero's integrated flag), then the one with the most compute units.
  There is no CPU or OpenCL fallback.
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
`auto` sizes the cache from the free VRAM rounded down (64 slots, or 256 MiB when slots are sized per expert),
so starts on the same machine get the same cache.
Before the rounding, free VRAM moved it by a slot or two
([record](../bench/results/2026-10-02-new-machine/README.md#outputs-that-change-from-run-to-run)).
To compare outputs across machines or settings, fix it with `--expert-cache N`.

### The output head's VRAM refusal

Sometimes the upload of the output head is refused VRAM at start.
About 27 GiB is free then, and it is always the first device allocation after the arena is registered for device copies.
The same settings then start normally.
It happened twice on the old machine and 3 times in about 120 starts on the new one
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
- **SYCL** (ggml-sycl, `-DSTRATA_VISION_SYCL=ON`): with `--nonfree on`, when icpx and oneMKL are installed.
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
| `qsa_prompt_attn_parity` | The existing FP64 reference and FP32 baseline, ported to SYCL (built, not a CTest case: it is also a benchmark). The XMX kernel passes it at int8 and FP16 KV ([record](../bench/results/2026-10-02-prompt-attn-xmx/README.md)); without the matrix engines the caller keeps `qsa_decode_attn_batch` |
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
| `setup.py` | `--spec 4`, `--spec-min-p 0.5`, `--expert-cache auto`, `--prefill auto` | `--prefill auto` now picks chunks up to 32768 tokens ([record](../bench/results/2026-10-03-prompt-upstream/README.md)); the rest as upstream |
| `src/program/generate.cpp` | 700 MiB VRAM reserve (down to 300 on a small card); prefill borrowing and chunk selection; cache adaptation every 4 rounds and up to 96 swaps | As upstream |
| `src/program/generate.cpp` and the CPU pool | Physical-core worker heuristic plus host worker; cache-hit pokes and graph/doorbell scheduling | The worker count was measured on the new machine: 7 to 19 workers decode equally fast, so the default stays ([record](../bench/results/2026-10-02-new-machine/README.md#cpu-worker-threads)) |
| `src/kernels/cpu/iq_avx512.cpp` | Software prefetch distance and AVX-512-specific tuning | Not measured: the development machine has no AVX-512 |
