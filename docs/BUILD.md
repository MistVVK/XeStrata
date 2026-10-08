<!--
SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
SPDX-License-Identifier: LGPL-3.0-or-later
-->
# Building XeStrata

English | [日本語](BUILD.ja.md)

How to install XeStrata from the source, the engine's build modes, what setup does and the packages it needs, and how the deb and rpm packages are made.
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
  `--license free` builds with free software only (Intel GPUs), `--license contrib-icpx` with Intel oneAPI's icpx.
- **Time**: the first run takes a while for the download (60-110 GB) and the build. If it stops, the next run continues.
- **Where things go**: the model files in `XeStrata-data` next to the XeStrata folder, the model configs
  (`xestrata-<model>.json`) and logs in the XeStrata folder.

**Afterwards**, `./setup.sh` starts the model right away.
The README's `xestrata` options work the same with `./setup.sh`. The terminal chat is `.venv/bin/python chat.py`.

**Updating**: run `./update.sh`. In a git checkout it runs `git pull`, then updates the engine, the Python packages and
the model settings (it does not start the model). You can also download a new version, unpack it elsewhere and run
`./setup.sh` there: the new copy finds `XeStrata-data` and runs with the same settings.

## Build and run

The engine builds in three modes (AGENTS.md, "Free and non-free builds"), chosen by the CMake option `STRATA_LICENSE`, after the Debian archive's main, contrib and non-free.

- **free** (`-DSTRATA_LICENSE=free`): built with intel/llvm's DPC++ 7 or later (its SYCL runtime is libsycl 9).
  Free software only; no oneAPI environment is used.
  Validated with intel/llvm v7.1.1 built from source by `tools/intel_llvm_build.py` (see [Setup](#setup)).
  A distribution's package of version 7 or later serves as well; an older one (Ubuntu 26.04's `dpclang++` 6.2) is refused by CMake and setup.
  6.2's SYCL runtime reported no XMX for the Arc Pro B70, and the prompt path took about 1.6 times as long
  ([record](../bench/results/2026-10-02-dp4a/README.md)).
  With intel/llvm built with ROCm's HIP, `STRATA_HIP_ARCHS` (`gfx1200`, for example) makes the code for AMD GPUs (RDNA2 and later: gfx103x, gfx11xx, gfx12xx) as well.
  ROCm is free software, so this is the same in the free and contrib-llvm modes ([DEVTOOLS.md](DEVTOOLS.md#hip-amd-gpus)).
  `auto` (the default) takes this PC's AMD GPUs as the kernel's KFD reports them, and `rocblas` every RDNA2 or later GPU the distribution's rocBLAS has code for and the SYCL compiler has a target for (the packages are built so; intel/llvm 7.1.1 has none for gfx1152 and gfx1153).
  `STRATA_ROCM_DEVICE_LIBS` gives the place of ROCm's device libraries (`ockl.bc`).
  An AMD GPU's dense products run in oneMath's rocBLAS backend, which the fork's change makes hand those with 16-bit inputs to hipBLASLt.
  Without hipBLASLt, CMake warns (and stops with `STRATA_PACKAGE=ON`), and those products run in rocBLAS alone (at about a fifteenth of the speed on gfx12).
  Validated: CTest passes on an RX 9060 XT (gfx1200, Fedora 44, ROCm 7.1.1).
- **contrib-llvm** (the default, `-DSTRATA_LICENSE=contrib`): intel/llvm built with its CUDA target (`tools/intel_llvm_build.py --contrib`), and with `STRATA_CUDA_ARCHS` (`sm_89`, for example) the code for NVIDIA GPUs as well.
  XeStrata's source is the free mode's and stays free software,
  but the build needs NVIDIA's CUDA toolkit and a run NVIDIA's driver, neither of them free software (XeStrata ships neither).
  Validated to build for sm_89 with Ubuntu 26.04's `nvidia-cuda-toolkit` 12.4 and intel/llvm v7.1.1.
  Validated to run on an RTX 4070 (sm_89).
  The engine links no maker's driver library: the SYCL runtime and oneMath open the ones that are there at run time.
  So it runs without NVIDIA's driver and CUDA on a PC with only Intel GPUs, and without Level Zero on one with only NVIDIA GPUs
  (checked on the development machine with the other maker's libraries made unreadable; on such a PC itself `unverified`).
- **contrib-icpx** (`-DSTRATA_LICENSE=contrib-icpx`): built with Intel oneAPI's icpx, which is not free software (validated: 2026.1.1).
  Source oneAPI's environment before building. Intel GPUs only:
  Codeplay's plugins that gave icpx NVIDIA and AMD targets ended with oneAPI 2025.2, and from 2025.3 the CUDA and HIP adapters are not released as binaries.

The contrib-llvm and contrib-icpx modes hand the prompt path's dense matrix products (`src/prefill/gemm.cpp`) to oneMath (Apache-2.0):
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
CMake fetches oneMath from XeStrata's fork on GitHub when it configures ([MistVVK/oneMath](https://github.com/MistVVK/oneMath), tag `xestrata-2`, checking the archive's SHA-256): oneMath with XeStrata's changes (cuBLAS's BF16 product, hipBLASLt in the rocBLAS backend, and the cuBLAS and rocBLAS backends' targets kept apart, so that one build makes both).
On a machine without the network, give it the same archive with `STRATA_ONEMATH_SOURCE` (a local file or URL).
The changes are listed in [third_party/main/README.md](../third_party/main/README.md).
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
With `STRATA_NO_BF16_MMA=1` the contrib-llvm and contrib-icpx modes do not hand the BF16 products to oneMath either; with FP16 matrix engines they take the path through FP16 below.
`STRATA_NO_BLAS=1` computes the dense matrix products with the own kernels in the contrib-llvm and contrib-icpx modes too (to compare).
On the RTX 4070 CTest gave the same results under each of them, and the top token of all 16 positions agreed.
Where oneMath fails a BF16 product on the GPU (a small one is tried at start), the BF16 products alone take the own kernels.
On a GPU whose matrix engines take FP16 and not BF16 (NVIDIA's before sm_80, Volta and Turing), the prompt path's BF16 products convert both operands to FP16 and run as FP16 products (`src/prefill/gemm.cpp`, upstream f2fb7c1 and ff6f9f1).
On such a GPU cuBLAS computes BF16 products without the tensor cores, and the own kernels fall to DP4a.
The conversion is exact within FP16's normal range; finite values past it saturate at ±65504. An accumulating product and one of a single output row stay BF16.
With sm_70 code on the RTX 4070 and `STRATA_NO_BF16_MMA=1` to imitate it, the relative error against FP64 was 2e-6 or less, 2.5–3.2 times as fast as the DP4a path (0.7%; on Volta and Turing themselves `unverified`).
`STRATA_BF16_TC=0` turns it off.
`tools/xmx_probe.cpp` asks every GPU the same without building the engine.

Where the GPU reports the int8 combination 16 x 16 x 16 on 32 lanes (NVIDIA's tensor cores), the prompt path's expert products run on the int8 matrix engines (`src/kernels/xe/iq_mmq.cpp`, llama.cpp's MMQ written anew on joint_matrix).
The weights are read in their GGUF blocks and the activations rounded to int8, a scale for each 32 values.
That reads and writes less than dequantizing to FP16 and multiplying, and the relative error against FP64 is about 0.4% (the activations' rounding).
It covers the i-quants but IQ1_M, and Q2_0; a layer of another type keeps the FP16 path.

On a GPU without XMX that reports the FP16 combination 16 x 16 x 16 on 32 lanes (NVIDIA's tensor cores, from sm_70 on), the prompt attention runs a kernel written on joint_matrix's portable API (`src/kernels/xe/qsa_prompt_attn.cpp`).
It takes FP16, INT8 and K8V4 KV; Q4_0 keeps the FP32 kernel (the range of upstream's Volta kernel 06a90a2).
INT8's and Q4_0's codes are exact in FP16 and go to the matrix engines as they are, with the scales applied in FP32.
With sm_70 code on the RTX 4070, `qsa_prompt_attn_parity` was off FP64 by at most 3 times the FP32 kernel's error, and a chunk of 2,048 queries over 32K cells ran 1.6 times as fast at INT8 and 1.5 times at FP16 and K8V4.
Intel GPUs do not report that shape and keep the XMX FP16 path.
`STRATA_PREFILL_MMQ=0` takes the FP16 path.

Pascal (sm_60, sm_61: P100, P40, the GTX 10 series) gets code too, from a CUDA 12 toolkit.
It has no tensor cores, so no matrix-engine path is chosen and it runs on the DP4a and FP32 paths.
sm_60 has no dp4a instruction, so that one product is computed a byte at a time there, to the same integer (`src/kernels/xe/cuda_intrinsics.hpp`, upstream e200e08f).
The kernels for the tensor cores compile an empty body in code for an architecture before sm_70 and are never launched on such a GPU.
Built with sm_60 code only and run on the RTX 4070 (the driver compiles the PTX), CTest passed its 59 tests, and with the matrix engines left out (`STRATA_NO_XMX=1`)
IQ2_XS gave the same answers as sm_89 code on all 8 questions (the expert cache size pinned; decode 3-5% slower).
Pascal hardware itself is `unverified`.

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
For NVIDIA GPUs (contrib-llvm), with intel/llvm built by `tools/intel_llvm_build.py --contrib`:

```bash
C=$PWD/.tools/intel-llvm-contrib/install
cmake -S . -B build/contrib -DCMAKE_CXX_COMPILER=$C/bin/clang++ -DCMAKE_C_COMPILER=$C/bin/clang \
  -DSTRATA_LICENSE=contrib \
  -DSTRATA_ENABLE_XE=ON -DSTRATA_NATIVE_EXPERTS=ON -DSTRATA_GGML_DIR=$PWD/third_party/main/llama.cpp
cmake --build build/contrib --target strata -j6
LD_LIBRARY_PATH=$C/lib build/contrib/strata-device
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
  setup therefore ends by starting the model once and asking it a long question and a picture
  ([record](../bench/results/2026-09-30-xe-setup/README.md#compiling-the-gpu-code-at-setup-added-later-on-2026-09-30));
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
oneAPI's for icpx, the intel/llvm built here for that one.
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
- **contrib-icpx**: Intel oneAPI's icpx, for Intel GPUs, with the SYCL image encoder.

With an AMD GPU on the machine, free and contrib-llvm make the code for its gfx as well (`STRATA_HIP_ARCHS`).
They then use intel/llvm built here, not the distribution's `dpclang++`.
Without ROCm's packages ([the table below](#packages)), setup names the missing ones and stops.
When the intel/llvm built here has no HIP target (built before ROCm was installed), setup asks whether to build it again.
contrib-icpx cannot use AMD GPUs.

contrib-llvm stops without oneMKL (`MKLROOT`, else `/opt/intel/oneapi/mkl/latest`) when the machine has an Intel GPU, and without the CUDA toolkit (`nvcc` and cuBLAS) when it has NVIDIA GPUs; with both, it wants both.
contrib-icpx stops without oneMKL.
NVIDIA GPUs can be used in the contrib-llvm mode only.
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
setup.sh installs Python 3 with venv through `sudo apt-get` (`sudo dnf` on Fedora) only when there is none.
Install the other packages first.

From the table, take the common row of the mode you build and the rows of the GPU makers the PC has (any number of them), and put them after `sudo apt install` (`sudo dnf install` on Fedora).
`./setup.sh --packages` prints that line for this PC's distribution and GPUs and the mode of `--license` (contrib-llvm unless given).
It prints other choices too, as in `./setup.sh --packages fedora44/intel,amd --license free`.

| Choice | Ubuntu 26.04 | Fedora 44 |
| --- | --- | --- |
| common (free, contrib-llvm) | `python3-venv git cmake ninja-build build-essential libhwloc-dev libzstd-dev libvulkan-dev glslc spirv-headers mesa-vulkan-drivers libblosc2-dev` | `python3 git cmake ninja-build gcc-c++ hwloc-devel libzstd-devel libzstd-static vulkan-loader-devel glslc spirv-headers-devel mesa-vulkan-drivers blosc2-devel` |
| common (contrib-icpx) | `python3-venv build-essential libblosc2-dev intel-oneapi-compiler-dpcpp-cpp intel-ocloc intel-oneapi-mkl-sycl-devel` | `python3 gcc-c++ blosc2-devel intel-oneapi-compiler-dpcpp-cpp intel-ocloc intel-oneapi-mkl-sycl-devel` |
| Intel GPUs | `libze1 libze-intel-gpu1 libigc2 libigdfcl2` | `oneapi-level-zero oneapi-level-zero-devel intel-level-zero` |
| Intel GPUs (contrib-llvm) | `intel-oneapi-mkl-sycl-devel` | `intel-oneapi-mkl-sycl-devel` |
| AMD GPUs (free, contrib-llvm) | `libamdhip64-dev rocm-device-libs-21 clang-21 libclang-rt-21-dev librocblas-dev libhipblaslt-dev libhipblas-common-dev` | `rocm-hip-devel rocm-device-libs rocm-clang rocm-clang-runtime-devel rocblas-devel hipblaslt-devel hipblas-common-devel` |
| NVIDIA GPUs (contrib-llvm) | `nvidia-cuda-toolkit` and NVIDIA's driver | `cuda-toolkit-13-4` (NVIDIA's CUDA repository) and NVIDIA's driver |

- The packages starting with `intel-oneapi-` are in Intel's oneAPI repository (Intel's guides show how to add it: [APT](https://www.intel.com/content/www/us/en/docs/oneapi-toolkit/installation-guide-linux/latest/install-oneapi-toolkit-with-apt.html), [DNF](https://www.intel.com/content/www/us/en/docs/oneapi-toolkit/installation-guide-linux/latest/install-oneapi-toolkit-with-yum-dnf.html)).
- The common row's Vulkan packages (`libvulkan-dev` and the others) are for the GPU image encoder, and `libblosc2-dev` (`blosc2-devel`) for compressing the saved conversations (optional; [`conversation_save_compress`](DETAILS.md#keeping-parked-conversations-across-restarts-opt-in)).
  With contrib-icpx and oneDNN for the SYCL image encoder, install `intel-oneapi-dnnl-devel` too.
- The AMD row's `clang-21` and `libclang-rt-21-dev` (on Fedora `rocm-clang` and `rocm-clang-runtime-devel`: ROCm's clang and its runtime) compile nothing (intel/llvm does the compiling).
  HIP's CMake package, which oneMath's rocBLAS backend reads, asks the compiler for clang's runtime builtins (`libclang_rt.builtins`) to add them to the link.
  intel/llvm has no such builtins, so ROCm's clang is made `HIP_CXX_COMPILER` to answer (`strata_hip_cxx` in `cmake/StrataHip.cmake`).
  They are needed to build only, not to run.
- Not checked (`unverified`): running an AMD GPU on Ubuntu, and contrib-icpx on Fedora.

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
  `XESTRATA_JOBS=16 tools/package/build.sh …` sets how many files compile at once (default: the threads; intel/llvm links as many at once as the free RAM allows).
- **No GPU needed**: the container gets no GPU and nothing of this PC's.
  CMake values that come from this PC's GPUs or CPU (`auto` for `STRATA_CUDA_ARCHS`, `STRATA_HIP_ARCHS`, `STRATA_ONEMKL`, `STRATA_CUDA_PATH` and
  `STRATA_CUDA_PTX`, and `STRATA_PORTABLE=OFF`) are refused with `STRATA_PACKAGE=ON`; `build.sh` sets them all.
  AMD GPUs get `STRATA_HIP_ARCHS=rocblas`: every RDNA2 or later GPU the distribution's rocBLAS has code for.
- **Network**: it fetches the distribution's packages, intel/llvm, llama.cpp, oneMath, and for the cuda variants NVIDIA's
  and Intel's repositories (the CUDA toolkit and oneMKL).
- **What comes out**: `build/pkg/<distro>-<variant>/` with the packages and `BUILDINFO` (the commit, the image, the intel/llvm
  and llama.cpp versions, the versions of the packages the build had).

| Variant | Distribution | CUDA toolkit | NVIDIA code |
| --- | --- | --- | --- |
| free | Ubuntu 26.04, Fedora 44 | — | — |
| cuda13.1 | Ubuntu 26.04 | `cuda-toolkit-13-1` from multiverse | sm_75, sm_80, sm_86, sm_89, sm_90 |
| cuda13.4 | Fedora 44 | `cuda-toolkit-13-4` from NVIDIA's repository | sm_75, sm_80, sm_86, sm_89, sm_90 |
| cuda12.4 | Ubuntu 26.04 | `nvidia-cuda-toolkit` | sm_60, sm_61, sm_70, sm_75, sm_80, sm_86, sm_89, sm_90 |

Every variant has AMD code for the GPUs the distribution's rocBLAS (ROCm 7.1) has code for (but gfx1152 and gfx1153, which intel/llvm 7.1.1 has no target for).

| Distribution | AMD code |
| --- | --- |
| Ubuntu 26.04 | gfx1030, gfx1100, gfx1101, gfx1151, gfx1200, gfx1201 |
| Fedora 44 | gfx1030, gfx1031, gfx1035, gfx1036, gfx1100, gfx1101, gfx1102, gfx1103, gfx1150, gfx1151, gfx1200, gfx1201 |

- A cuda variant is named after the CUDA version it is built with.
  NVIDIA's repository for Fedora 44 has CUDA 13.3 and 13.4 only, so Fedora's is 13.4; it has no CUDA 12 either, so cuda12.4 is Ubuntu's only.
- CUDA 13 builds no Pascal (sm_60, sm_61) or Volta (sm_70) code: a P100, P40 or V100 takes cuda12.4.
- intel/llvm 7.1.1's SYCL has no name above sm_90: newer GPUs (an RTX 50) run sm_90's PTX, which their driver compiles.
- The PTX is the toolkit's version, so the driver must support that CUDA version.

### What the packages hold

Each variant (`xestrata-free`, `xestrata-contrib-cuda<version>`) is one package with:

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
- The Python files are compiled when the package is built, and `xestrata` runs with `PYTHONDONTWRITEBYTECODE=1`: nothing is written
  under `/usr` later and nothing goes in `/etc`, so removing the packages leaves nothing behind.

### Distributions not covered

- **Debian 13 (trixie)**: its archive has no Level Zero driver for Intel GPUs (`intel-compute-runtime`)
  (2026-10-06; forky and sid have 26.27).
- **openSUSE Leap 16.0**: its compute-runtime is 25.18, which does not list the Arc Pro B70, and openSUSE's OBS has no newer one
  for Leap 16 (2026-10-06).
