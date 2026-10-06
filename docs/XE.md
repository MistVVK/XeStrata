<!--
SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
SPDX-License-Identifier: LGPL-3.0-or-later
-->
# XeStrata on Intel GPUs

English | [日本語](XE.ja.md)

This document records how XeStrata's engine runs on Intel GPUs and how that was checked: how to build it, what setup does, the packages it needs, what the engine requires of the GPU, and the ported arithmetic and its validation.
How to use it is in the [README](../README.md) and the [details](DETAILS.md).
The Japanese version ([XE.ja.md](XE.ja.md)) is the original; this is its translation.

XeStrata xe0.1.39 runs Strata's engine on Intel GPUs through Level Zero and SYCL; its version follows the upstream version it has integrated.
It is ported from Strata 0.1.24 (`3ce2523c2823687de5372be3af58534f56cbf286`) and carries part of the changes up to 0.1.39 (`6f32ec07`)
([Integration through Strata 0.1.38](#integration-through-strata-0138)).
The same SYCL code is compiled with intel/llvm for NVIDIA GPUs as well (the contrib build, [Build and run](#build-and-run)).
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

## Integration through Strata 0.1.38

Of upstream Strata's changes up to 0.1.38 (`99f3dbd0b21d1401b3769e0c0d963913607f380b`), the single-GPU Linux ones are carried into Xe.
The upstream history is kept as a merge.
The CUDA, HIP, Windows and multi-GPU implementations are not carried.

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

## Build and run

The engine builds in three modes (AGENTS.md, "Free and non-free builds"), chosen by the CMake option `STRATA_LICENSE`, after the Debian archive's main, contrib and non-free.

- **free** (`-DSTRATA_LICENSE=free`): built with intel/llvm's DPC++ 7 or later (its SYCL runtime is libsycl 9).
  Free software only; no oneAPI environment is used.
  Validated with intel/llvm v7.1.1 built from source by `tools/intel_llvm_build.py` (see [Setup](#setup)).
  A distribution's package of version 7 or later serves as well; an older one (Ubuntu 26.04's `dpclang++` 6.2) is refused by CMake and setup.
  6.2's SYCL runtime reported no XMX for the Arc Pro B70, and the prompt path took about 1.6 times as long
  ([record](../bench/results/2026-10-02-dp4a/README.md)).
- **contrib** (the default, `-DSTRATA_LICENSE=contrib`): intel/llvm built with its CUDA target (`tools/intel_llvm_build.py --contrib`), and with `STRATA_CUDA_ARCHS` (`sm_89`, for example) the code for NVIDIA GPUs as well.
  XeStrata's source is the free mode's and stays free software,
  but the build needs NVIDIA's CUDA toolkit and a run NVIDIA's driver, neither of them free software (XeStrata ships neither).
  Validated to build for sm_89 with Ubuntu 26.04's `nvidia-cuda-toolkit` 12.4 and intel/llvm v7.1.1.
  Validated to run on an RTX 4070 (sm_89).
- **contrib-icpx** (`-DSTRATA_LICENSE=contrib-icpx`): built with Intel oneAPI's icpx, which is not free software (validated: 2026.1.1).
  Source oneAPI's environment before building. Intel GPUs only:
  Codeplay's plugins that gave icpx NVIDIA and AMD targets ended with oneAPI 2025.2, and from 2025.3 the CUDA and HIP adapters are not released as binaries.

The contrib and contrib-icpx modes hand the prompt path's dense matrix products (`src/prefill/gemm.cpp`) to oneMath (Apache-2.0):
oneMKL on Intel GPUs, cuBLAS on NVIDIA ones.
Which backends are built follows the GPUs of the machine it is built on (every maker it has).
An Intel GPU brings the oneMKL backend (`STRATA_ONEMKL`), which needs oneMKL (oneAPI's `intel-oneapi-mkl-devel`, found at `MKL_ROOT`, else `MKLROOT`, else `/opt/intel/oneapi/mkl/latest`).
NVIDIA GPUs bring the code for each one's architecture as nvidia-smi reports it (`STRATA_CUDA_ARCHS`) and the cuBLAS backend, which need the CUDA toolkit.
With GPUs of different generations, each runs the newest code it can.
That needs intel/llvm built by `tools/intel_llvm_build.py --contrib` (with the fix `cuda-select-binary-2`); with another intel/llvm, `auto` builds the code for the oldest architecture only.
On a PC with several CUDA toolkits (a distribution's `/usr/lib/cuda` and NVIDIA's `/usr/local/cuda-X.Y`, say), clang takes `/usr/local/cuda` first and CMake's FindCUDA the `nvcc` on PATH, and the GPU code and cuBLAS came from different versions.
`STRATA_CUDA_PATH`'s default, `auto`, tries each toolkit found (`CUDA_PATH`, `CUDA_HOME`, `CUDA_ROOT`, `/usr/local/cuda-*`, `/opt/cuda`, `/usr/lib/cuda`, the `nvcc` on PATH) and takes the one that builds code for the most of this PC's NVIDIA GPUs, then the newest architectures for them, then has cuBLAS, then is the newest (`cmake/StrataCuda.cmake`).
The GPU code and oneMath's cuBLAS both use it; a toolkit without cuBLAS gets no cuBLAS backend, and the dense products run on XeStrata's own kernels.
Each GPU gets code for its architecture, or the newest older one the toolkit builds, and the driver compiles that PTX for the GPU.
intel/llvm 7.1.1's SYCL has no name above sm_90, so an architecture without one (an RTX 50's sm_120) is built through the generic NVIDIA target (`nvptx64-nvidia-cuda` with `--cuda-gpu-arch`); an executable takes it once, for the oldest GPU that needs it. A toolkit before CUDA 12.8 cannot build it, and the GPU gets sm_90 code.
oneMath's cuBLAS backend takes one architecture, the oldest (the products themselves run inside cuBLAS).
An executable built with CUDA 13.1 for sm_89 and the generic sm_120 took the sm_89 image on the 4070 and passed its 52 CTest tests (on an RTX 50 `unverified`).
An old GPU a toolkit dropped (Volta in CUDA 13, for one) makes it take a toolkit that has it, if there is one; otherwise it gets a warning and no code.
On this PC (CUDA 12.4 and 13.1) it took 13.1 for the RTX 4070 alone and 12.4 (code for all three) with a Volta, an Ada and an RTX 50 imitated; the 13.1 build passed its 52 CTest tests on the 4070.
The NVIDIA code is PTX of the toolkit's version by default (PTX 8.4 for CUDA 12.4).
A driver older than the toolkit cannot load it, so `STRATA_CUDA_PTX`'s default, `auto`, lowers it to the driver's version (libcuda's `cuDriverGetVersion`); a number sets it (`78`: CUDA 11.8's PTX 7.8).
A PTX 7.8 build passed its 52 CTest tests on an RTX 4070.
The driver needs CUDA 12.0 or later (525 or later) for the functions intel/llvm's CUDA adapter calls (SYCL graphs: `cuGraphAddKernelNode_v2`).
A CUDA 12.0-12.3 driver with the CUDA 12.4 toolkit has not been tried (`unverified`).
Both variables default to `auto`; a value given is used as it is (to build for another machine). contrib-icpx always builds the oneMKL backend.
CMake fetches oneMath v0.9 from GitHub when it configures (checking the archive's SHA-256) and applies XeStrata's changes (cuBLAS's BF16 product, `third_party/main/oneMath/patches/`).
On a machine without the network, give it the same archive with `STRATA_ONEMATH_SOURCE` (a local file or URL).
The patches are listed in [third_party/main/README.md](../third_party/main/README.md).
Where oneMath has no backend for the GPU the own kernels take the products.
coder-iq1_m's prefill of 997 tokens went from 1421 to 1350 ms on the B70 and stayed the same on the RTX 4070 (3845 ms).

At start the engine asks the GPU whether it has the matrix combinations its XMX kernels need (FP16 and BF16 8 x 16 x 16).
Without them it takes the path that uses another shape the GPU reports (the Arc A series' (Xe-HPG) 8 x 8 x 16, NVIDIA's tensor cores' 16 x 16 x 16) through joint_matrix's portable API (`src/kernels/xe/mma_gemm.cpp`),
and without that the DP4a path, and says once which.
The Arc A series takes `mma_gemm` because its SYCL runtime refuses the Intel extensions the XMX kernels use (joint_matrix's prefetches and checked loads and stores).
On the A380 it ran 17–38% faster than DP4a, and closer to FP64 (relative error 0.0001% against 0.53%; [record](../bench/results/2026-10-04-dg2-dp4a/README.md)).
`STRATA_NO_XMX=1` takes the DP4a path even when the GPU has matrix engines.
`STRATA_MMA=1` takes `mma_gemm` even where XMX is there, if the GPU reports a shape it carries (Xe2's 8 x 16 x 16 and others; to check that path).
`STRATA_NO_BF16_MMA=1` drops the BF16 matrix combinations from what the GPU reports, `STRATA_NO_INT8_MMA=1` the int8 ones (`src/kernels/xe/matrix_report.cpp`),
to imitate a GPU whose matrix engines lack the type (NVIDIA's before sm_80 have no BF16, before sm_72 no int8) on one that has it.
With `STRATA_NO_BF16_MMA=1` the contrib modes do not hand the BF16 products to oneMath either.
`STRATA_NO_BLAS=1` computes the dense matrix products with the own kernels in the contrib modes too (to compare).
On the RTX 4070 CTest gave the same results under each of them, and the top token of all 16 positions agreed.
Where oneMath fails a BF16 product on the GPU (a small one is tried at start), the BF16 products alone take the own kernels.
`tools/xmx_probe.cpp` asks every GPU the same without building the engine.

Where the GPU reports the int8 combination 16 x 16 x 16 on 32 lanes (NVIDIA's tensor cores), the prompt path's expert products run on the int8 matrix engines (`src/kernels/xe/iq_mmq.cpp`, llama.cpp's MMQ written anew on joint_matrix).
The weights are read in their GGUF blocks and the activations rounded to int8, a scale for each 32 values.
That reads and writes less than dequantizing to FP16 and multiplying, and the relative error against FP64 is about 0.4% (the activations' rounding).
It covers the i-quants but IQ1_M, and Q2_0; a layer of another type keeps the FP16 path.
Intel GPUs do not report that shape and keep the XMX FP16 path.
`STRATA_PREFILL_MMQ=0` takes the FP16 path.

### The engine's parts and tests

The parts and tests that need no model build without llama.cpp (`STRATA_NATIVE_EXPERTS=OFF`).
The free mode (intel/llvm in `.tools/intel-llvm/install`):

```bash
L=$PWD/.tools/intel-llvm/install
cmake -S . -B build/llvm7 \
  -DCMAKE_CXX_COMPILER=$L/bin/clang++ -DCMAKE_C_COMPILER=$L/bin/clang -DSTRATA_LICENSE=free \
  -DSTRATA_ENABLE_XE=ON \
  -DSTRATA_NATIVE_EXPERTS=OFF \
  -DSTRATA_BUILD_TESTS=ON
cmake --build build/llvm7 --target strata-device strata-load elementwise_parity \
  gguf_reader_test suffix_drafter_test controller_test draft_policy_test conv_cache_test \
  platform_memory_test ple_reader_test -j4
```

With icpx (contrib-icpx):

```bash
source /opt/intel/oneapi/setvars.sh
cmake -S . -B build/xe \
  -DCMAKE_CXX_COMPILER=icpx -DSTRATA_LICENSE=contrib-icpx \
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
- **Run a free build in a shell without oneAPI's environment (`setvars.sh`), with intel/llvm's `lib` on `LD_LIBRARY_PATH`.**
  With oneAPI's environment, oneAPI's adapters (`libur_adapter_level_zero` and others) and `libumf` come first on `LD_LIBRARY_PATH` and are loaded.
  With a runtime they did not match (dpclang 6.2's `libsycl.so.8`), the tests that use the GPU stopped with a segmentation fault as they started.
  setup puts no oneAPI on the library path of a free install, so the engine setup runs is not affected.

### The engine

The engine (the `strata` executable) uses llama.cpp for the native i-quant experts.
llama.cpp goes in `third_party/main/llama.cpp` at the pinned commit (`3cf03257f219afbe7334045ff7c6a06ac68c627d`).
setup.py fetches it; the image encoder also uses its `tools/mtmd`.

```bash
L=$PWD/.tools/intel-llvm/install
cmake -S . -B build/llvm7 -DCMAKE_CXX_COMPILER=$L/bin/clang++ -DCMAKE_C_COMPILER=$L/bin/clang -DSTRATA_ENABLE_XE=ON \
  -DSTRATA_LICENSE=free -DSTRATA_NATIVE_EXPERTS=ON -DSTRATA_GGML_DIR=$PWD/third_party/main/llama.cpp
cmake --build build/llvm7 --target strata -j6
```

With icpx: `source /opt/intel/oneapi/setvars.sh`, then `-DCMAKE_CXX_COMPILER=icpx -DSTRATA_LICENSE=contrib-icpx` instead.
For NVIDIA GPUs (contrib), with intel/llvm built by `tools/intel_llvm_build.py --contrib`:

```bash
C=$PWD/.tools/intel-llvm-contrib/install
cmake -S . -B build/contrib -DCMAKE_CXX_COMPILER=$C/bin/clang++ -DCMAKE_C_COMPILER=$C/bin/clang \
  -DSTRATA_LICENSE=contrib \
  -DSTRATA_ENABLE_XE=ON -DSTRATA_NATIVE_EXPERTS=ON -DSTRATA_GGML_DIR=$PWD/third_party/main/llama.cpp
cmake --build build/contrib --target strata -j6
LD_LIBRARY_PATH=$C/lib build/contrib/strata-device
```

For another machine, `STRATA_CUDA_ARCHS` lists the GPUs' architectures (`sm_89` for the RTX 40 series, `sm_86` for the RTX 30).
The same executable runs on Intel GPUs too.

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
  So the first run of a model whose kernels were not compiled before takes longer to its first token.
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

setup builds in the mode `--license` names ([above](#build-and-run)), contrib unless told otherwise.

- **free**: a free compiler, for Intel GPUs (chosen in the order below).
- **contrib**: for Intel and NVIDIA GPUs.
  With an NVIDIA GPU on the machine it uses intel/llvm with its CUDA target (`.tools/intel-llvm-contrib`; without one, setup asks and builds it here),
  without one a compiler chosen as for free.
  The NVIDIA GPUs' architectures come from what nvidia-smi reports (the compute capability), and then NVIDIA's CUDA toolkit (`nvcc`) is needed.
- **contrib-icpx**: Intel oneAPI's icpx, for Intel GPUs, with the SYCL image encoder.

contrib stops without oneMKL (`MKLROOT`, else `/opt/intel/oneapi/mkl/latest`) when the machine has an Intel GPU, and without the CUDA toolkit (`nvcc` and cuBLAS) when it has NVIDIA GPUs; with both, it wants both.
contrib-icpx stops without oneMKL.
NVIDIA GPUs can be used in the contrib mode only.
The mode, a `--intel-llvm DIR` and an accepted build without XMX are kept in the settings for later runs.
An older settings file's `nonfree` (icpx allowed) is read as contrib-icpx.

The free compiler is chosen in this order:

1. With `--intel-llvm DIR`: the intel/llvm installed in DIR (setup stops if it is older than 7).
1. With `--intel-llvm-build`: intel/llvm built here from source.
1. Otherwise the distribution's intel/llvm (`dpclang++`, or the newest `dpclang++-N`), when it is 7 or later.
1. Otherwise setup asks whether to build intel/llvm here; without it, it stops and says how to get one.

Whether a compiler is intel/llvm 7 or later is read from its SYCL runtime's version (`__LIBSYCL_MAJOR_VERSION` 9 or more).

On an Intel GPU the chosen compiler builds `tools/xmx_probe.cpp` and runs it on the chosen GPU.
What happens next depends on the result:

- **The GPU is not listed at all**: setup stops.
  It names the GPU's Level Zero driver (`libze-intel-gpu1`, Intel's compute-runtime) as missing or too old for the GPU.
  openSUSE Leap 16's compute-runtime 25.18 lists no Arc Pro B70.
- **The GPU is listed without XMX**: setup asks: build without XMX, or stop.
  Stopping also names `--license contrib-icpx` (icpx).
  Without a terminal, or with `--yes`, it stops;
  `--allow-no-xmx` accepts the slower path.

When the engine is compiled again after an update, setup uses the compiler recorded in `BUILD.json` and asks nothing.
An engine from before the free build (no record) was built with icpx and stays so, as `--license contrib-icpx`.

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
in a container with only this column and `python3-venv`, setup ran from start to end ([record](../bench/results/2026-10-03-clean-setup/README.md);
with `dpclang-6` then instead of intel/llvm: from a clean Ubuntu with today's intel/llvm build it is `unverified`).
`intel-opencl-icd` is not needed to run the engine.

| For | Free | contrib-icpx (`--license contrib-icpx`) |
| --- | --- | --- |
| Building the engine | `libze-dev` 1.28.2 | also `intel-oneapi-compiler-dpcpp-cpp` 2026.1.1 (Intel's apt repository), `intel-ocloc` 26.05.37020.3, `intel-oneapi-mkl-sycl-devel` 2026.1.0, `patch` (to apply XeStrata's changes to oneMath) |
| Building intel/llvm 7 or later (setup builds it here) | `git`, `cmake`, `ninja-build`, `g++`, `libhwloc-dev`, `libzstd-dev` | (not needed: icpx is used) |
| Running it | `libze1` 1.28.2, `libze-intel-gpu1` 26.05.37020.3, `intel-opencl-icd` 26.05.37020.3 (`libze-intel-gpu-legacy1-1` 24.35 is also installed; the B70 uses the new runtime) | the same |
| The CPU image encoder | `build-essential` | the same |
| The GPU image encoder (see [Images](#images)) | Vulkan: `libvulkan-dev` 1.4.341, `glslc` 2026.1, `spirv-headers` 1.6.1, `mesa-vulkan-drivers` 26.0.8 | SYCL: `intel-oneapi-mkl-sycl-devel` 2026.1.0 |
| oneDNN for the SYCL image encoder (optional; off unless chosen) | — | `intel-oneapi-dnnl-devel` 2026.0.2 |
| Compressing the saved conversations (optional; [`conversation_save_compress`](DETAILS.md#keeping-parked-conversations-across-restarts-opt-in)) | `libblosc2-dev` 2.23.0 (built in when the build finds it) | the same |

The contrib mode (`--license contrib`) needs the free column's packages for building intel/llvm,
`intel-oneapi-mkl-sycl-devel` 2026.1.0, `patch`, and for an NVIDIA GPU `nvidia-cuda-toolkit` 12.4 and NVIDIA's driver (`nvidia-driver-610-open`).

#### Fedora 44

Fedora 44 has no DPC++ package, so the free mode builds intel/llvm with `--intel-llvm-build`.
On 2026-10-04 setup ran from start to end in a Fedora 44 container on the development machine
(Ubuntu 26.04's kernel 7.0 and xe driver), with only these packages installed.
intel/llvm v7.1.1 built in 14 minutes, the B70 got XMX, and the model it started answered a question.

| For | Packages (versions checked) |
| --- | --- |
| Python | `python3` 3.14.7 (setup.sh can also install it through `dnf`) |
| Building intel/llvm | `git`, `cmake` 4.3.0, `ninja-build`, `gcc-c++` 16.2.1, `hwloc-devel`, `libzstd-devel` |
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
- **Choosing the GPU**: the runtime drives one GPU: an Intel GPU through Level Zero, an NVIDIA GPU through the CUDA backend (not OpenCL, which lists the same Intel GPUs again).
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
| `setup.py` | `--spec 4`, `--spec-min-p 0.5`, `--expert-cache auto`, `--prefill auto` | `--prefill auto` now picks chunks up to 32768 tokens ([record](../bench/results/2026-10-03-prompt-upstream/README.md)); a chunk that does not fit with the default ring takes a 96-slot ring (upstream #583; IQ3_S on the B70 limited to 8 GB: 8K and 32K prompts 1.5-2x); the rest as upstream |
| `src/program/generate.cpp` | 700 MiB VRAM reserve (down to 300 on a small card); prefill borrowing and chunk selection; cache adaptation every 4 rounds and up to 96 swaps | As upstream |
| `src/program/generate.cpp` and the CPU pool | Physical-core worker heuristic plus host worker; cache-hit pokes and graph/doorbell scheduling | The worker count was measured on the new machine: 7 to 19 workers decode equally fast, so the default stays ([record](../bench/results/2026-10-02-new-machine/README.md#cpu-worker-threads)) |
| `src/kernels/cpu/iq_avx512.cpp` | Software prefetch distance and AVX-512-specific tuning | Not measured: the development machine has no AVX-512 |
