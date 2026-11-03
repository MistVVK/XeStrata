<!--
SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
SPDX-License-Identifier: LGPL-3.0-or-later
-->
# Building XeStrata

English | [日本語](BUILD.ja.md)

How to install XeStrata from the source, the engine's build modes, what setup does and the packages it needs, how the deb and rpm packages are made, and the Podman container.
How to install the packages and use XeStrata is in the [README](../README.md).
The Japanese version ([BUILD.ja.md](BUILD.ja.md)) is the original; this is its translation.

## Installing from the source

On a distribution without the packages, or for development, install from the repository with `./setup.sh`.

1. [Download this project](https://github.com/MistVVK/XeStrata/archive/refs/heads/main.zip) and unpack it
   (`git clone` works too).
1. Install the packages in the [table](#packages): setup installs no system packages.
1. Run **`./setup.sh`** in a terminal and answer its questions, the same as in the [README](../README.md#install).

setup checks the GPU, RAM and CPU, chooses a SYCL compiler and builds the engine ([Setup](#setup)), downloads the model and starts it.

- **Compiler**: the default is the contrib-llvm build, with intel/llvm's DPC++ 7 or later, the one with the CUDA target when there is an NVIDIA GPU.
  When the distribution's `dpclang++` is older, setup asks whether to build intel/llvm here (about 13 minutes;
  [details](#the-sycl-compiler)).
  When the GPU's XMX cannot be used, it asks whether to build without XMX or stop.

  contrib-llvm uses parts that are not free software (oneMKL, and for NVIDIA GPUs the CUDA toolkit).
  `--license free` builds with free software only (Intel and AMD GPUs).
- **Time**: the first run takes a while for the download (60-110 GB) and the build. If it stops, the next run continues.
- **Where things go**: the model files in `XeStrata-data` next to the XeStrata folder, the model configs
  (`xestrata-<model>.json`) in `~/.config/xestrata` and the logs in `~/.local/state/xestrata` (as with the packages).

**Afterwards**, `./setup.sh` starts the model right away.
The README's `xestrata` options work the same with `./setup.sh`. The terminal chat is `.venv/bin/python chat.py`.

**Updating**: run `./update.sh`. In a git checkout it runs `git pull`, then updates the engine, the Python packages and
the model settings (it does not start the model). You can also download a new version, unpack it elsewhere and run
`./setup.sh` there: the new copy finds `XeStrata-data` and runs with the same settings.

## Build and run

The engine builds in two modes (AGENTS.md, "Free and non-free builds"), chosen by the CMake option `STRATA_LICENSE`, after the Debian archive's main, contrib and non-free.

- **free** (`-DSTRATA_LICENSE=free`): built with intel/llvm's DPC++ 7 or later (its SYCL runtime is libsycl 9).
  Free software only; no oneAPI environment is used.
  Validated with intel/llvm v7.2.0 built from source by `tools/intel_llvm_build.py` (see [Setup](#setup)).
  A distribution's package of version 7 or later serves as well; an older one (Ubuntu 26.04's `dpclang++` 6.2) is refused by CMake and setup.
  6.2's SYCL runtime reported no XMX for the Arc Pro B70, and the prefill path took about 1.6 times as long.
  With intel/llvm built with ROCm's HIP, `STRATA_HIP_ARCHS` (`gfx1200`, for example) makes the code for AMD GPUs (RDNA2 and later: gfx103x, gfx11xx, gfx12xx) as well.
  ROCm is free software, so this is the same in the free and contrib-llvm modes ([DEVTOOLS.md](DEVTOOLS.md#hip-amd-gpus)).
  `auto` (the default) takes this PC's AMD GPUs as the kernel's KFD reports them, and `rocblas` every RDNA2 or later GPU the distribution's rocBLAS has code for and the SYCL compiler has a target for (the packages are built so; intel/llvm 7.2.0 has none for gfx1152 and gfx1153).
  For ROCm's device libraries (`ockl.bc`), CMake builds a test kernel for the AMD GPU with AMD's in `/opt/rocm*`, the distribution's and the SYCL compiler's own (`tools/intel_llvm_build.py` builds them with it), in that order, and takes the first that builds.
  The installed ROCm's are taken where they build; the compiler's own where those are of a newer LLVM than the compiler's, which it cannot read (ROCm 7.14's, for example).
  When none does, it stops with each one's reason (bitcode of a newer LLVM than the compiler's, files the compiler asks for that are missing).
  `STRATA_ROCM_DEVICE_LIBS` gives their place; it stops with the reason when the test kernel does not build with them either.
  An AMD GPU's dense products run in oneMath's rocBLAS backend, which the fork's change makes hand those with 16-bit inputs to hipBLASLt.
  Without hipBLASLt, CMake warns (and stops with `STRATA_PACKAGE=ON`), and those products run in rocBLAS alone (at about a fifteenth of the speed on gfx12).
  Validated: CTest passes on an RX 9060 XT (gfx1200, Fedora 44, ROCm 7.1.1).
  Validated to build and run on a gfx1030 card (RX 6800 XT, CachyOS (Arch), ROCm 7.2.4): the setup unit tests pass, and IQ3_XXS decodes at ~64-68 tok/s (2000-token runs, 262144 context, ~65 tok/s with ~215K tokens loaded).
- **contrib-llvm** (the default, `-DSTRATA_LICENSE=contrib`): intel/llvm built with its CUDA target (`tools/intel_llvm_build.py --contrib`), and with `STRATA_CUDA_ARCHS` (`sm_89`, for example) the code for NVIDIA GPUs as well.
  XeStrata's source is the free mode's and stays free software,
  but the build needs NVIDIA's CUDA toolkit and a run NVIDIA's driver, neither of them free software (XeStrata ships neither).
  Validated to build for sm_89 with Ubuntu 26.04's `nvidia-cuda-toolkit` 12.4 and intel/llvm v7.2.0.
  Validated to run on an RTX 4070 (sm_89).
  The engine links no maker's driver library: the SYCL runtime and oneMath open the ones that are there at run time.
  So it runs without NVIDIA's driver and CUDA on a PC with only Intel GPUs, and without Level Zero on one with only NVIDIA GPUs
  (checked on the development machine with the other maker's libraries made unreadable).
Intel oneAPI's icpx is not taken (CMake refuses it): the third mode, contrib-icpx, which built with it, was removed on 2026-10-15.
`-DSTRATA_LICENSE=contrib-icpx` stops CMake with a pointer to `-DSTRATA_LICENSE=contrib` (contrib-llvm).

The contrib-llvm mode hands the prefill path's dense matrix products (`src/prefill/gemm.cpp`) to oneMath (Apache-2.0):
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
intel/llvm 7.2.0's SYCL has no name above sm_90, so an architecture without one (an RTX 50's sm_120) is built through the generic NVIDIA target (`nvptx64-nvidia-cuda` with `--cuda-gpu-arch`); an executable takes it once, for the oldest GPU that needs it. A toolkit before CUDA 12.8 cannot build it, and the GPU gets sm_90 code.
oneMath's cuBLAS backend takes one architecture, the oldest (the products themselves run inside cuBLAS).
An executable built with CUDA 13.1 for sm_89 and the generic sm_120 took the sm_89 image on the 4070 and passed its 52 CTest tests.
An old GPU a toolkit dropped (Volta in CUDA 13, for one) makes it take a toolkit that has it, if there is one; otherwise it gets a warning and no code.
On this PC (CUDA 12.4 and 13.1) it took 13.1 for the RTX 4070 alone and 12.4 (code for all three) with a Volta, an Ada and an RTX 50 imitated; the 13.1 build passed its 52 CTest tests on the 4070.
The NVIDIA code is PTX of the toolkit's version by default (PTX 8.4 for CUDA 12.4).
A driver older than the toolkit cannot load it, so `STRATA_CUDA_PTX`'s default, `auto`, lowers it to the driver's version (libcuda's `cuDriverGetVersion`); a number sets it (`78`: CUDA 11.8's PTX 7.8).
A PTX 7.8 build passed its 52 CTest tests on an RTX 4070.
The driver needs CUDA 12.0 or later (525 or later) for the functions intel/llvm's CUDA adapter calls (SYCL graphs: `cuGraphAddKernelNode_v2`).
Both variables default to `auto`; a value given is used as it is (to build for another machine).
CMake fetches oneMath from XeStrata's fork on GitHub when it configures ([MistVVK/oneMath](https://github.com/MistVVK/oneMath), tag `xestrata-12`, checking the archive's SHA-256): oneMath with XeStrata's changes (cuBLAS's BF16 product, hipBLASLt in the rocBLAS backend, the int8 products in the rocBLAS and cuBLAS backends, the cuBLAS and rocBLAS backends' targets kept apart, so that one build makes both, the rocBLAS backend without what HIP and SYCL deprecate, and Windows built by the GNU-style clang++ of the cross build, its cuBLAS and rocBLAS backends with Windows' import libraries, and oneMKL's SYCL link there without `-fsycl-allow-device-image-dependencies`), with oneMath's `develop` merged into it.
On a machine without the network, give it the same archive with `STRATA_ONEMATH_SOURCE` (a local file or URL).
The changes are listed in [third_party/main/README.md](../third_party/main/README.md).
Where oneMath has no backend for the GPU the own kernels take the products.
coder-iq1_m's prefill of 997 tokens went from 1421 to 1350 ms on the B70 and stayed the same on the RTX 4070 (3845 ms).

At start the engine asks the GPU whether it has the matrix combinations its XMX kernels need (FP16 and BF16 8 x 16 x 16).
Without them it takes the path that uses another shape the GPU reports (the Arc A series' (Xe-HPG) 8 x 8 x 16, NVIDIA's tensor cores' 16 x 16 x 16) through joint_matrix's portable API (`src/kernels/xe/mma_gemm.cpp`),
and without that the DP4a path, and says once which.
The Arc A series takes `mma_gemm` because its SYCL runtime refuses the Intel extensions the XMX kernels use (joint_matrix's prefetches and checked loads and stores).
On the A380 it ran 17–38% faster than DP4a, and closer to FP64 (relative error 0.0001% against 0.53%).
`STRATA_NO_XMX=1` takes the DP4a path even when the GPU has matrix engines.
`STRATA_MMA=1` takes `mma_gemm` even where XMX is there, if the GPU reports a shape it carries (Xe2's 8 x 16 x 16 and others; to check that path).
`STRATA_NO_BF16_MMA=1` drops the BF16 matrix combinations from what the GPU reports, `STRATA_NO_INT8_MMA=1` the int8 ones (`src/kernels/xe/matrix_report.cpp`),
to imitate a GPU whose matrix engines lack the type (NVIDIA's before sm_80 have no BF16, before sm_72 no int8) on one that has it.
With `STRATA_NO_BF16_MMA=1` the contrib-llvm mode does not hand the BF16 products to oneMath either; with FP16 matrix engines they take the path through FP16 below.
`STRATA_NO_BLAS=1` computes the dense matrix products with the own kernels in the contrib-llvm mode too (to compare).
On the RTX 4070 CTest gave the same results under each of them, and the top token of all 16 positions agreed.
Where oneMath fails a BF16 product on the GPU (a small one is tried at start), the BF16 products alone take the own kernels.
On a GPU whose matrix engines take FP16 and not BF16 (NVIDIA's before sm_80, Volta and Turing), the prefill path's BF16 products convert both operands to FP16 and run as FP16 products (`src/prefill/gemm.cpp`, upstream f2fb7c1 and ff6f9f1).
On such a GPU cuBLAS computes BF16 products without the tensor cores, and the own kernels fall to DP4a.
The conversion is exact within FP16's normal range; finite values past it saturate at ±65504. An accumulating product and one of a single output row stay BF16.
With sm_70 code on the RTX 4070 and `STRATA_NO_BF16_MMA=1` to imitate it, the relative error against FP64 was 2e-6 or less, 2.5–3.2 times as fast as the DP4a path (0.7%).
`STRATA_BF16_TC=0` turns it off.
`tools/xmx_probe.cpp` asks every GPU, without building the engine, whether it reports FP16 and BF16 matrix engines in the XMX kernels' shape or in `mma_gemm`'s Intel one (8 x 8 x 16).

Where the GPU reports the int8 combination 16 x 16 x 16 on 32 lanes (NVIDIA's tensor cores), the prefill path's expert products run on the int8 matrix engines (`src/kernels/xe/iq_mmq.cpp`, llama.cpp's MMQ written anew on joint_matrix).
The weights are read in their GGUF blocks and the activations rounded to int8, a scale for each 32 values.
That reads and writes less than dequantizing to FP16 and multiplying, and the relative error against FP64 is about 0.4% (the activations' rounding).
It covers the i-quants but IQ1_M, and Q2_0; a layer of another type keeps the FP16 path.

On a GPU without XMX that reports the FP16 combination 16 x 16 x 16 on 32 lanes (NVIDIA's tensor cores, from sm_70 on), prefill's attention runs a kernel written on joint_matrix's portable API (`src/kernels/xe/qsa_prompt_attn.cpp`).
It takes FP16, INT8 and K8V4 KV; Q4_0 keeps the FP32 kernel (the range of upstream's Volta kernel 06a90a2).
INT8's and Q4_0's codes are exact in FP16 and go to the matrix engines as they are, with the scales applied in FP32.
With sm_70 code on the RTX 4070, `qsa_prompt_attn_parity` was off FP64 by at most 3 times the FP32 kernel's error, and a chunk of 2,048 queries over 32K cells ran 1.6 times as fast at INT8 and 1.5 times at FP16 and K8V4.
Intel GPUs do not report that shape and keep the XMX FP16 path.
Qwen3.8's prefill takes the faster of the int8 and the FP16 expert products for each layer of a chunk, by the rows its experts average. Where one stops beating the other is measured at the start on each GPU, both timed at 16 to 256 rows an expert in the shape of the model's first layer, and logged (181 rows on the B70; int8 at every count on the RTX 4070). `STRATA_PREFILL_MMQ=0` takes the FP16 path, `1` the int8 products.

Pascal (sm_60, sm_61: P100, P40, the GTX 10 series) gets code too, from a CUDA 12 toolkit.
It has no tensor cores, so no matrix-engine path is chosen and it runs on the DP4a and FP32 paths.
sm_60 has no dp4a instruction, so that one product is computed a byte at a time there, to the same integer (`src/kernels/xe/cuda_intrinsics.hpp`, upstream e200e08f).
The kernels for the tensor cores compile an empty body in code for an architecture before sm_70 and are never launched on such a GPU.
Built with sm_60 code only and run on the RTX 4070 (the driver compiles the PTX), CTest passed its 59 tests, and with the matrix engines left out (`STRATA_NO_XMX=1`)
IQ2_XS gave the same answers as sm_89 code on all 8 questions (the expert cache size pinned; decode 3-5% slower).
On a GPU of compute capability 6.0 (GP100: the P100), `STRATA_DOWN_WS=1` copies the routed experts' IQ4_NL and Q2_0 down weight rows into the work-group's local memory before their products (upstream 6a6d58ad0, opt-in; the output is bitwise the same, and other GPUs ignore it).
Maxwell and older (compute capability below 6.0) do not run it: the SYCL runtime reports no FP16 for an NVIDIA GPU below 5.3, and the engine takes no GPU without FP16 (`xestrata --setup` says so).

On Volta (sm_70), `STRATA_SM70_TABLE=1` takes two decode kernels upstream measured on a V100 (opt-in until confirmed, as in upstream's c85b7c87).
One is the routed experts' "mode 8": gate/up, SwiGLU and the q8_1 quantization in one pass, with the codebook grid and the activations in the work-group's local memory (`src/kernels/xe/iq_kernels.cpp`, upstream f6330507: 227 -> 159 µs on a V100).
The other is the hyper-connection read's norm and up projection, with the weights loaded ahead to hide the latency (`src/kernels/xe/fused_gr.cpp`, upstream aaa323fe: 50.9 -> 46.7 µs a read on a V100).
Both give the default kernels' output bit for bit, and `STRATA_GR_FAST=1` takes the second on any GPU.
Mode 8 is the default on every GPU whose local memory holds it (B70 decode +0.8-1.1% at 1K and 32K, RTX 4070 equal; the idea of upstream f7eaf2de's STRATA_EXPDEC). `STRATA_EXP_MODE=0` keeps the former kernels.
Built with sm_70 code only and run on an RTX 3070, CTest passed its 71 tests with and without them, and the experts' and the read's outputs were the default's bit for bit (on an RX 9060 XT as well).

### The engine's parts and tests

The parts and tests that need no model build without llama.cpp (`STRATA_NATIVE_EXPERTS=OFF`).
The free mode (intel/llvm in `.tools/intel-llvm/install`):

```bash
L=$PWD/.tools/intel-llvm/install
cmake -S . -B build/free \
  -DCMAKE_CXX_COMPILER=$L/bin/clang++ -DCMAKE_C_COMPILER=$L/bin/clang -DSTRATA_LICENSE=free \
  -DSTRATA_ENABLE_XE=ON \
  -DSTRATA_NATIVE_EXPERTS=OFF \
  -DSTRATA_BUILD_TESTS=ON
cmake --build build/free --target strata-device strata-load elementwise_parity \
  gguf_reader_test suffix_drafter_test controller_test draft_policy_test conv_cache_test \
  platform_memory_test ple_reader_test -j4
```

- `platform_memory_test` mlocks 256 MiB and fails under the default 8 MiB `ulimit -l`.
  It passes only where that limit is raised.
- The host platform library (`memory.cpp`, `direct_file.cpp`) and the PLE n-gram reader build without the GPU toolchain.
  CPU-only tools can be configured with `STRATA_ENABLE_XE=OFF`.
- The CPU canonical expert tests need AVX-512 or model files and are not in the commands above.
- `build/free/BUILD.json` records the backend, version, upstream reference, compiler and build scope.
- **Run a free build in a shell without oneAPI's environment (`setvars.sh`), with intel/llvm's `lib` on `LD_LIBRARY_PATH`.**
  With oneAPI's environment, oneAPI's adapters (`libur_adapter_level_zero` and others) and `libumf` come first on `LD_LIBRARY_PATH` and are loaded.
  With a runtime they did not match (dpclang 6.2's `libsycl.so.8`), the tests that use the GPU stopped with a segmentation fault as they started.
  setup puts no oneAPI on the engine's library path, so the engine setup runs is not affected.

### The engine

The engine (the `strata` executable) uses llama.cpp for the native i-quant experts.
llama.cpp goes in `third_party/main/llama.cpp` at the pinned commit (`3cf03257f219afbe7334045ff7c6a06ac68c627d`).
setup.py fetches it; the image encoder also uses its `tools/mtmd`.

```bash
L=$PWD/.tools/intel-llvm/install
cmake -S . -B build/free -DCMAKE_CXX_COMPILER=$L/bin/clang++ -DCMAKE_C_COMPILER=$L/bin/clang -DSTRATA_ENABLE_XE=ON \
  -DSTRATA_LICENSE=free -DSTRATA_NATIVE_EXPERTS=ON -DSTRATA_GGML_DIR=$PWD/third_party/main/llama.cpp
cmake --build build/free --target strata -j6
```

For NVIDIA GPUs (contrib-llvm), with intel/llvm built by `tools/intel_llvm_build.py --contrib`:

```bash
C=$PWD/.tools/intel-llvm-contrib/install
cmake -S . -B build/contrib-llvm -DCMAKE_CXX_COMPILER=$C/bin/clang++ -DCMAKE_C_COMPILER=$C/bin/clang \
  -DSTRATA_LICENSE=contrib \
  -DSTRATA_ENABLE_XE=ON -DSTRATA_NATIVE_EXPERTS=ON -DSTRATA_GGML_DIR=$PWD/third_party/main/llama.cpp
cmake --build build/contrib-llvm --target strata -j6
LD_LIBRARY_PATH=$C/lib build/contrib-llvm/strata-device
```

For another machine, `STRATA_CUDA_ARCHS` lists the GPUs' architectures (`sm_89` for the RTX 40 series, `sm_86` for the RTX 30).
`STRATA_HIP_ARCHS` lists AMD GPUs' (`gfx1030` for the RX 6800, `gfx1100` for the RX 7900 XTX, `gfx1200` for the RX 9060 XT).
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
  No AOT code is built (the deb and rpm packages neither).
  So the first run of a model whose kernels were not compiled before takes longer to its first token.
  setup therefore ends by starting the model once and asking it a long question and a picture;
  `--no-warmup` skips it.

## Setup

`./setup.sh` goes through these steps:

1. It checks the GPU: through sysfs and the kernel driver's memory query, an Intel GPU on the xe or i915 driver,
   its VRAM, the render node's permissions, Resizable BAR and the PCIe link.
   An AMD GPU on amdgpu has its gfx (`gfx1200`, for example) read from the kernel's KFD topology, and `/dev/kfd`'s permissions checked too.
   A GPU older than RDNA2, or one the distribution's rocBLAS (in a package, its engine) has no code for, is shown as not usable ([README](../README.md#amd-gpus)).
1. It checks RAM and CPU and asks its questions.
1. It chooses the SYCL compiler ([below](#the-sycl-compiler)) and compiles the engine in `build-xe`.

The engine goes to `engine/` with `BUILD.json` (`"backend": "xe"` and the compiler it was built with).
An engine without that record is taken for upstream's CUDA build and compiled over.
An engine whose compiler changed is compiled again.
The config setup writes lists the runtime's library folders under `lib_dirs`:
the intel/llvm built here for that one.
So the server starts the engine without any toolkit environment.

Ready-made engines are in the deb and rpm packages ([below](#the-deb-and-rpm-packages)).
Several GPUs are chosen with setup's `--gpus` (a layer split, [MULTIGPU](MULTIGPU.md)).

### The SYCL compiler

setup builds in the mode `--license` names ([above](#build-and-run)), contrib-llvm unless told otherwise.

- **free**: a free compiler, for Intel GPUs (chosen in the order below).
- **contrib-llvm**: for Intel and NVIDIA GPUs.
  With an NVIDIA GPU on the machine it uses intel/llvm with its CUDA target (`.tools/intel-llvm-contrib`; without one, setup asks and builds it here),
  without one a compiler chosen as for free.
  The NVIDIA GPUs' architectures come from what nvidia-smi reports (the compute capability), and then NVIDIA's CUDA toolkit (`nvcc`) is needed.

With an AMD GPU on the machine, free and contrib-llvm make the code for its gfx as well (`STRATA_HIP_ARCHS`).
They then use intel/llvm built here, not the distribution's `dpclang++`.
Without ROCm's packages ([the table below](#packages)), setup names the missing ones and stops.
When the intel/llvm built here has no HIP target (built before ROCm was installed), setup asks whether to build it again.

contrib-llvm stops without oneMKL (`MKLROOT`, else `/opt/intel/oneapi/mkl/latest`) when the machine has an Intel GPU, and without the CUDA toolkit (`nvcc` and cuBLAS) when it has NVIDIA GPUs; with both, it wants both.
NVIDIA GPUs can be used in the contrib-llvm mode only.
The mode, a `--intel-llvm DIR` and an accepted build without XMX are kept in the settings for later runs.
`--license contrib-icpx` stops, saying the mode was removed and to use contrib-llvm (`--license contrib`).
An older settings file's contrib-icpx and `nonfree` (icpx allowed) are read as contrib-llvm, with a note saying so.

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
  Without a terminal, or with `--yes`, it stops;
  `--allow-no-xmx` accepts the slower path.

When the engine is compiled again after an update, setup uses the compiler recorded in `BUILD.json` and asks nothing.
An engine built with icpx (contrib-icpx's, and one from before the free build, with no record) stops setup with a pointer to `./setup.sh --setup`, which builds it again with intel/llvm.

`--cmake-arg KEY=VALUE` hands the engine's build a CMake option (`-DKEY=VALUE`), once for each.
They are kept in the settings for the builds after an update; `--cmake-arg ""` clears them.

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

setup.py installs no system packages and runs no package manager and no sudo.
setup.sh installs Python 3 with venv through `sudo apt-get` (`sudo dnf` on Fedora, `sudo pacman -S` on Arch) only when there is none.
Install the other packages first.

From the table, take the common row of the mode you build and the rows of the GPU makers the PC has (any number of them), and put them after `sudo apt install` (`sudo dnf install` on Fedora, `sudo pacman -S` on Arch).
`./setup.sh --packages` prints that line for this PC's distribution and GPUs and the mode of `--license` (contrib-llvm unless given).
It prints other choices too, as in `./setup.sh --packages fedora44/intel,amd --license free`.

| Choice | Ubuntu 26.04 | Fedora 44 | Arch |
| --- | --- | --- | --- |
| common (free, contrib-llvm) | `python3-venv git cmake ninja-build build-essential lld libhwloc-dev libzstd-dev libvulkan-dev glslc spirv-headers mesa-vulkan-drivers libblosc2-dev` | `python3 git cmake ninja-build gcc-c++ lld hwloc-devel libzstd-devel libzstd-static vulkan-loader-devel glslc spirv-headers-devel mesa-vulkan-drivers blosc2-devel` | `python base-devel git cmake ninja lld hwloc zstd vulkan-headers vulkan-icd-loader shaderc spirv-headers opencl-headers blosc2` |
| Intel GPUs | `libze1 libze-intel-gpu1 libigc2 libigdfcl2` | `oneapi-level-zero oneapi-level-zero-devel intel-level-zero` | `level-zero-loader intel-compute-runtime vulkan-intel` |
| Intel GPUs (contrib-llvm) | `intel-oneapi-mkl-sycl-devel` | `intel-oneapi-mkl-sycl-devel` | none (oneMKL is not in Arch's repositories) |
| AMD GPUs (free, contrib-llvm) | `libamdhip64-dev rocm-device-libs-21 clang-21 libclang-rt-21-dev librocblas-dev libhipblaslt-dev libhipblas-common-dev` | `rocm-hip-devel rocm-device-libs rocm-clang rocm-clang-runtime-devel rocblas-devel hipblaslt-devel hipblas-common-devel` | `hip-runtime-amd rocm-device-libs rocm-llvm rocblas hipblaslt hipblas-common vulkan-radeon` |
| NVIDIA GPUs (contrib-llvm) | `nvidia-cuda-toolkit` and NVIDIA's driver | `cuda-toolkit-13-4` (NVIDIA's CUDA repository) and NVIDIA's driver | `cuda` and NVIDIA's driver |

- The packages starting with `intel-oneapi-` are in Intel's oneAPI repository (Intel's guides show how to add it: [APT](https://www.intel.com/content/www/us/en/docs/oneapi-toolkit/installation-guide-linux/latest/install-oneapi-toolkit-with-apt.html), [DNF](https://www.intel.com/content/www/us/en/docs/oneapi-toolkit/installation-guide-linux/latest/install-oneapi-toolkit-with-yum-dnf.html)).
- Intel GPU drivers (`libze-intel-gpu1` and the rest) ship old on Ubuntu 24.04/26.04 (26.04 has compute-runtime 26.05 and IGC 2.28).
  Below compute-runtime 26.22 a driver state can refuse device allocations while the VRAM is free (Intel's [compute-runtime #916](https://github.com/intel/compute-runtime/issues/916); the driver works around an xe kernel bug fixed in Linux 7.1 — on XeStrata's side, [#34](https://github.com/MistVVK/XeStrata/issues/34)).
  On Ubuntu 24.04/26.04 with the HWE kernel, setup points to [Intel's graphics PPA](https://launchpad.net/~kobuk-team/+archive/ubuntu/intel-graphics) (a newer compute-runtime and IGC) and stops the start until it is updated (`--old-driver` starts anyway).
  On the GA kernel it recommends moving to the HWE kernel with the PPA, or Fedora 44 / Podman (whose IGC is 2.32 or later; [Troubleshooting](DETAILS.md#troubleshooting)).
- The common row's Vulkan packages (`libvulkan-dev` and the others) are for the GPU image encoder, and `libblosc2-dev` (`blosc2-devel`) for compressing the saved conversations (optional; [`conversation_save_compress`](DETAILS.md#keeping-parked-conversations-on-the-disk)).
- The AMD row's `clang-21` and `libclang-rt-21-dev` (on Fedora `rocm-clang` and `rocm-clang-runtime-devel`: ROCm's clang and its runtime) compile nothing (intel/llvm does the compiling).
  HIP's CMake package, which oneMath's rocBLAS backend reads, asks the compiler for clang's runtime builtins (`libclang_rt.builtins`) to add them to the link.
  intel/llvm has no such builtins, so ROCm's clang is made `HIP_CXX_COMPILER` to answer (`strata_hip_cxx` in `cmake/StrataHip.cmake`).
  They are needed to build only, not to run.
- On Arch the Vulkan ICD is the GPU maker's own package (`vulkan-intel`, `vulkan-radeon`), not the one `mesa-vulkan-drivers` carries for both on Ubuntu and Fedora.
- Arch's `zstd` has no `libzstd.a` (`zstd-static` is an AUR package).
  intel/llvm's `configure.py` asks for the static zstd (`LLVM_USE_STATIC_ZSTD=ON`), so `tools/intel_llvm_build.py` turns that off where there is none and links the shared one (intel/llvm handles it from v7.1.1: [intel/llvm#22294](https://github.com/intel/llvm/pull/22294)).
- Arch's ROCm clang is in `/opt/rocm/lib/llvm`, and its runtime builtins (`libclang_rt.builtins`) are in the same `rocm-llvm` package (one package where the AMD row's `clang-21` and `libclang-rt-21-dev` are two).
- Arch's ROCm lies under AMD's own tree, `/opt/rocm` (its CMake packages in `/opt/rocm/lib/cmake`, its libraries in `/opt/rocm/lib`).
  Neither CMake's default search nor ld.so's looks there, so `cmake/StrataHip.cmake` puts `/opt/rocm` on `CMAKE_PREFIX_PATH`, and setup records `/opt/rocm/lib` among the engine's library folders.
- On Arch `opencl-headers` is installed as well: intel/llvm's build installs the SYCL tools too, and `sycl-ls` includes `CL/cl.h` through `sycl/detail/cl.h` while its target gets no include for the OpenCL headers the configuration fetches (those belong to the runtime's own objects).
  Ubuntu's and Fedora's builds have that header (their compilers bring it in); Arch's clang does not.
- oneMKL is not in Arch's repositories: install Intel's offline installer for it if it is wanted.
  The contrib-llvm mode hands an Intel GPU's dense products to oneMKL, so it needs oneMKL for an Intel GPU.
  Without it, use the free mode: it builds intel/llvm from the source, and an Intel GPU's dense products are XeStrata's own kernels' work.

#### Other distributions

openSUSE Leap 16.0 was tried in a container on 2026-10-04.
It does not run as it is:
its compute-runtime (`libze_intel_gpu1`) is 25.18, too old to list the Arc Pro B70.
intel/llvm builds, but setup stops because it sees no GPU.

What any distribution needs:

- Python 3.10 or newer with venv
- a Level Zero driver that lists the GPU (for the B70, a compute-runtime of the 26 series, for example)
- the Level Zero headers (`/usr/include/level_zero/ze_api.h`)
- a SYCL compiler: the distribution's DPC++, or the build tools `--intel-llvm-build` needs

## The deb and rpm packages

`tools/package/build.sh` makes the packages.
Each time it starts a new container from the distribution's official image and builds in it.

```bash
tools/package/build.sh ubuntu26.04 free       # xestrata-free
tools/package/build.sh ubuntu26.04 cuda13.1   # xestrata-contrib-cuda13.1
tools/package/build.sh ubuntu26.04 cuda12.4   # xestrata-contrib-cuda12.4
tools/package/build.sh fedora44 free
tools/package/build.sh fedora44 cuda13.4
```

- **The input is a commit**: `git archive` passes the commit's files, nothing else, to the container.
  The default is HEAD, and then uncommitted changes in the work tree stop it; a third argument names another commit.
  Nothing of the work tree (`build-xe`, `.tools`, `.venv`) is used.
- **A clean container each time**: the images are pinned by digest in `build.sh`.
  In the container it installs the build tools and the distribution's ROCm and builds intel/llvm (`tools/intel_llvm_build.py`,
  with the HIP target, and the CUDA target for the cuda variants), the image encoders (CPU and Vulkan) and the engine, and CPack makes the packages
  (`tools/package/container.sh`, `cmake/packaging.cmake`).
  The container is removed at the end. intel/llvm is built every time, so a build takes a while.
  Kept between builds: the compiler cache (ccache, `build/pkg-ccache/<distro>/`, at most `XESTRATA_CCACHE_MAX`, default 30G),
  from which a second package of the same distribution takes most of intel/llvm's and the engine's objects.
  `build/pkg-cache/<distro>/` also keeps the packages apt and dnf fetched and the finished intel/llvm toolchains (used while the version, the patches, the base image and the CUDA and ROCm packages stay the same).
  `XESTRATA_PKG_CLEAN=1` leaves those two out: the build tools are fetched and intel/llvm is built again (a release's builds do this).
  Deleting a folder makes the next build start without it.
  The packages are compressed on every thread (the deb with zstd, the rpm with zstd 19).
  `XESTRATA_JOBS=16 tools/package/build.sh …` sets how many files compile at once (default: the threads; intel/llvm links as many at once as the free RAM allows).
- **No GPU needed**: the container gets no GPU and nothing of this PC's.
  CMake values that come from this PC's GPUs or CPU (`auto` for `STRATA_CUDA_ARCHS`, `STRATA_HIP_ARCHS`, `STRATA_ONEMKL`, `STRATA_CUDA_PATH` and
  `STRATA_CUDA_PTX`, and `STRATA_PORTABLE=OFF`) are refused with `STRATA_PACKAGE=ON`; `build.sh` sets them all.
  AMD GPUs get `STRATA_HIP_ARCHS=rocblas`: every RDNA2 or later GPU the distribution's rocBLAS has code for.
- **Network**: it fetches the distribution's packages, intel/llvm, llama.cpp, oneMath, and for the cuda variants NVIDIA's
  and Intel's repositories (the CUDA toolkit and oneMKL).
- **What comes out**: `build/pkg/<distro>-<variant>/` with the packages and `BUILDINFO` (the commit, the image, the intel/llvm
  and llama.cpp versions, the compiles taken from ccache and not, the versions of the packages the build had).

| Variant | Distribution | CUDA toolkit | NVIDIA code |
| --- | --- | --- | --- |
| free | Ubuntu 26.04, Fedora 44 | — | — |
| cuda13.1 | Ubuntu 26.04 | `cuda-toolkit-13-1` from multiverse | sm_75, sm_80, sm_86, sm_89, sm_90 |
| cuda13.4 | Fedora 44 | `cuda-toolkit-13-4` from NVIDIA's repository | sm_75, sm_80, sm_86, sm_89, sm_90 |
| cuda12.4 | Ubuntu 26.04 | `nvidia-cuda-toolkit` | sm_60, sm_61, sm_70, sm_75, sm_80, sm_86, sm_89, sm_90 |

Every variant has AMD code for the GPUs the distribution's rocBLAS (ROCm 7.1) has code for (but gfx1152 and gfx1153, which intel/llvm 7.2.0 has no target for).

| Distribution | AMD code |
| --- | --- |
| Ubuntu 26.04 | gfx1030, gfx1100, gfx1101, gfx1151, gfx1200, gfx1201 |
| Fedora 44 | gfx1030, gfx1031, gfx1035, gfx1036, gfx1100, gfx1101, gfx1102, gfx1103, gfx1150, gfx1151, gfx1200, gfx1201 |

- A cuda variant is named after the CUDA version it is built with.
  NVIDIA's repository for Fedora 44 has CUDA 13.3 and 13.4 only, so Fedora's is 13.4; it has no CUDA 12 either, so cuda12.4 is Ubuntu's only.
- CUDA 13 builds no Pascal (sm_60, sm_61) or Volta (sm_70) code: a P100, P40 or V100 takes cuda12.4.
- intel/llvm 7.2.0's SYCL has no name above sm_90: newer GPUs (an RTX 50) run sm_90's PTX, which their driver compiles.
- The PTX is the toolkit's version, so the driver must support that CUDA version.

### The Windows 7z

The Windows 7z is built with the same `tools/package/build.sh`, in a disposable Ubuntu 26.04 container (`tools/package/windows.sh`).

```bash
tools/package/build.sh windows cuda12.4      # XeStrata-<version>-windows-x64.7z
```

- All of it is cross-built on Linux for Windows: intel/llvm (the Linux contrib toolchain and the SYCL runtime for Windows), the engine (the contrib-llvm mode, the NVIDIA code as the cuda12.4 deb's), the image encoder on the CPU (`strata-vision-cpu.exe`; no Vulkan one, so pictures are read on the CPU), the .NET launchers (`dotnet-sdk-10.0`), and Python's embeddable distribution with the wheels; `tools/package/windows_bundle.py` makes the folder and the 7z.
- The MSVC CRT and the Windows SDK come from Microsoft through xwin, its license accepted (they are not free software and do not go into the 7z). Neither do the Windows CUDA headers and import libraries.
- It uses the Ubuntu 26.04 packages' ccache (`build/pkg-ccache/ubuntu26.04/`) and keeps xwin's and NuGet's downloads in `build/pkg-cache/windows/`.
- Every exe, DLL and Python package in the 7z must be one of `windows_bundle.py`'s components; a Visual C++ runtime or an NVIDIA library in it stops the build.
  `THIRD-PARTY-NOTICES.md` names each component, its version, license and source, and `licenses/` holds the license texts.
- What comes out: the 7z and `BUILDINFO` in `build/pkg/windows-cuda12.4/`.

### What the packages hold

Each variant (`xestrata-free`, `xestrata-contrib-cuda<version>`) is one package with the following.
Its version is, after Debian's, the upstream version integrated and the count of XeStrata's releases on it (from 1):
xe0.1.43.1-1 is `0.1.43.1-1` (for rpm, Version 0.1.43.1 and Release 1).

| What | Where |
| --- | --- |
| the `xestrata` command (setup.py), the server, the web app, the tools setup runs, llama.cpp's gguf-py, the data files, a systemd user unit | `/usr/share/xestrata`, `/usr/bin/xestrata`, `/usr/lib/systemd/user/xestrata.service` |
| the engine (`strata`), the image encoders, intel/llvm's SYCL runtime (`libsycl.so.9`, UR's loader, its Level Zero and HIP (AMD) adapters and in the cuda variants its CUDA adapter, `libumf`), oneMath (its rocBLAS backend, and in the cuda variants its oneMKL and cuBLAS backends) | `<libdir>/xestrata/engine` |

- The packages provide and conflict with `xestrata-engine`: one is installed at a time, and installing another replaces it.
- Neither Ubuntu 26.04 (`dpclang` up to 6) nor Fedora 44 has libsycl 9, so the packages carry it.
  intel/llvm's `libsycl-jit` (157 MB) is opened only to compile kernels from source, which the engine does not: it is left out.
- The dependencies are the distribution's.
  The Python packages are the distribution's versions, not pip's pinned ones (`requirements.txt`).
  The dependencies on the libraries the engine links come from `dpkg-shlibdeps` (deb) and rpmbuild (rpm).
  The bundled libraries are neither required nor provided.
- No package requires a GPU maker's driver or library.
  The free package recommends Intel's Level Zero driver and, for AMD GPUs, ROCm's libraries (the HIP runtime, rocBLAS,
  hipBLASLt), all free software.
  The contrib-llvm packages, which go on a PC with Intel, NVIDIA or AMD GPUs, only suggest the makers' ones (Level Zero, cuBLAS,
  oneMKL, ROCm's HIP runtime, rocBLAS and hipBLASLt): apt and dnf install recommendations by default, which would bring another maker's too.
  The UR adapters and oneMath's backends open them at run time, so the packages install without them, and a GPU without them is not used.
- Nothing that is not free software (oneMKL, cuBLAS, NVIDIA's driver) is bundled.
- Every variant's build makes intel/llvm's HIP target and adapter with the distribution's ROCm (7.1, in Ubuntu's universe
  and Fedora's own repositories), and carries the adapter. The AMD code the engine has is recorded in `BUILD.json`
  (`hip_archs`), from which setup decides whether a GPU can be used.
- The makers' packages the package recommends or suggests are recorded in `BUILD.json` too, by maker (`runtime`). setup asks apt or dnf which of those of the GPUs it uses are missing, and shows how to install them (adding a repository first where needed). Without the GPU's runtime it stops; without oneMKL or cuBLAS it only shows them and goes on.
- `third_party/nonfree/` is not packaged: the chat template is the one setup writes into the model's pack.
  The experimental speed projection's vector is downloaded by setup from the version's tag on GitHub when it is turned on.
- The Python files are compiled when the package is built, and `xestrata` runs with `PYTHONDONTWRITEBYTECODE=1`: nothing is written
  under `/usr` later and nothing goes in `/etc`, so removing the packages leaves nothing behind.

### Distributions not covered

- **Debian 13 (trixie)**: its archive has no Level Zero driver for Intel GPUs (`intel-compute-runtime`)
  (2026-10-06; forky and sid have 26.27).
- **openSUSE Leap 16.0**: its compute-runtime is 25.18, which does not list the Arc Pro B70, and openSUSE's OBS has no newer one
  for Leap 16 (2026-10-06).

## Podman

The Containerfile and the compose files in `container/` run XeStrata in a rootless Podman container (how to use it: [README](../README.md#podman)).
The image is built on the user's PC and not distributed: the contrib packages' image holds cuBLAS and oneMKL, which are not free software.

- **What it holds**: the Fedora 44 image `tools/package/build.sh` uses (pinned by digest), with the release's rpm, checked against `SHA256SUMS`.
  Then every maker's package its `BUILD.json` records in `runtime` (Level Zero, ROCm's HIP runtime, rocBLAS and hipBLASLt, cuBLAS, oneMKL)
  and Mesa's Vulkan drivers (`container/install.sh`).
  NVIDIA's driver libraries and `nvidia-smi` are not in it: CDI brings the host's.
- **Build arguments**: environment variables for `podman compose build`.

| Variable | Default | Meaning |
| --- | --- | --- |
| `XESTRATA_VERSION` | `latest` | the release's tag (`xe0.1.43.1`, for example) |
| `XESTRATA_VARIANT` | `contrib-cuda13.4` | the rpm's variant |
| `XESTRATA_NONFREE` | `1` | `0` leaves out oneMKL and NVIDIA's cuBLAS and cuDNN (Intel's and NVIDIA's dense products and NVIDIA's image attention then run on XeStrata's own kernels) |

- **A local rpm**: one rpm in `container/rpm/` is installed instead of the release's.
  `xestrata-free`, which the Releases do not have, is made with `tools/package/build.sh fedora44 free` and put there.
- **GPU**: `/dev/dri` is passed, and `userns_mode: keep-id` with `group_add: keep-groups` opens it as the host's user, with its uid and render group
  (`keep-groups` needs the crun runtime). For AMD, `compose.amd.yaml` adds `/dev/kfd`.
  For NVIDIA, `compose.nvidia.yaml` passes CDI's `nvidia.com/gpu=all`: install the NVIDIA Container Toolkit (`nvidia-container-toolkit`, in NVIDIA's repository) on the host and make its CDI specification once.

```bash
sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml
```

- **Network**: the host's. The server listens on 127.0.0.1:8095, as the packages' does;
  for other devices, `--host 0.0.0.0 --api-key <a secret>` as with the packages.
- **Where things go**: `~/.config/xestrata`, `~/.local/state/xestrata` and `~/.local/share/xestrata` are passed at the same paths as on the host, and `HOME` is the host's.
  The absolute paths in the settings are then the same in the host and in the container, so the host's package (Ubuntu's deb as well) shares the settings and the models.
  The engine's place in a model's settings is pointed on every start at the package that runs.
  The kernels the GPU drivers compile are kept in the named volume `cache` (`~/.cache`).
- **A data folder elsewhere** (`--data-dir`) is passed at its own path in `container/compose.override.yaml`.
  Run with `-f`, list `-f compose.override.yaml` as well.

```yaml
services:
  xestrata:
    volumes:
      - /mnt/models/XeStrata-data:/mnt/models/XeStrata-data
```

- **SELinux** (Fedora and others): the container runs with `label=disable`, so the host's folders are not relabeled; only the build labels `container/` to read it (`relabel=shared`).
- **memlock**: a rootless container cannot go above the host user's limit. When setup says the limit is too low, raise it on the host.
- **Stopping**: `podman compose down`. `init`'s PID 1 passes the signal to setup and the server, and the engine gets up to 60 seconds to stop.
