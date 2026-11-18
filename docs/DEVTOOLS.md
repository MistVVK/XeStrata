<!--
SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
SPDX-License-Identifier: LGPL-3.0-or-later
-->
# Developer tools

English | [日本語](DEVTOOLS.ja.md)

The tools a developer installs to lint, check and measure XeStrata.
What building and running it needs is in [BUILD.md](BUILD.md#setup).
Nothing here is needed to build, run or test the engine, and nothing here is bundled.
The tools come from the distribution, or from their own projects into folders git ignores.
The non-free ones (Intel SDE, VTune) come from Intel by hand.
The commands are for Ubuntu and Debian; the versions are those last used on the development machine.

| Folder (git ignores it) | What goes there |
| --- | --- |
| `.lint/` | lint tools that the distribution does not package |
| `.tools/` | other tools built from source (intel/llvm, Metrics Discovery), and what the Windows cross build uses |

## Lints

[AGENTS.md](../AGENTS.md#lints) says which lints to run and how (`tools/lint/run.sh`).

From the distribution:

```sh
sudo apt install gitleaks shellcheck clang-tidy cppcheck flake8 mypy python3-pyflakes codespell \
    markdownlint cmake-format tidy eslint
```

`tools/lint/run.sh` runs oneAPI's clang-tidy, which knows SYCL (found under `/opt/intel/oneapi/compiler`), not the distribution's.
clang-tidy reads `compile_commands.json` from the build folder (by default `build/free`; `XESTRATA_LINT_BUILD` changes it):

```sh
cmake -S . -B build/free -G Ninja -DCMAKE_EXPORT_COMPILE_COMMANDS=ON   # Ninja, as in BUILD.md (with Make the SYCL device code's dependencies do not connect)
```

The ones the distribution does not package, or packages too old, go into `.lint/`:

```sh
python3 -m venv .lint/venv && .lint/venv/bin/pip install ruff reuse        # ruff 0.16, reuse 6.2
npm install --prefix .lint stylelint stylelint-config-recommended         # stylelint 17
mkdir -p .lint/bin && curl -sSL https://github.com/lycheeverse/lychee/releases/latest/download/lychee-x86_64-unknown-linux-gnu.tar.gz \
    | tar -xz -C .lint/bin lychee                                          # lychee 0.24
```

`tools/lint/run.sh` uses the distribution's `reuse` when there is one and `.lint/venv/bin/reuse` otherwise.

## intel/llvm from source

The free build compiles the engine with intel/llvm's DPC++ ([BUILD.md](BUILD.md#build-and-run)).
Where the distribution has no DPC++, or one whose SYCL runtime reports no XMX for the GPU,
`tools/intel_llvm_build.py` builds a release from source into `.tools/intel-llvm/`.
An example of the latter is Ubuntu 26.04's 6.2 with the Arc Pro B70 (intel/llvm added the B70's XMX in v7.0.0).
setup runs it for `--intel-llvm-build`;
[BUILD.md](BUILD.md#the-sycl-compiler) describes how it reuses a finished build.
intel/llvm is free software (Apache-2.0 with LLVM exceptions) and builds with free software only.
It needs these packages (on Fedora 44: `git cmake ninja-build gcc-c++ lld hwloc-devel libzstd-devel libzstd-static python3`; on Arch: `git cmake ninja base-devel lld hwloc zstd opencl-headers python`):

```sh
sudo apt install git cmake ninja-build g++ lld python3 libhwloc-dev libzstd-dev
python3 tools/intel_llvm_build.py            # --keep-build keeps the build tree for a quicker update; --jobs N
```

Arch's `zstd` has no `libzstd.a` (`zstd-static` is an AUR package).
The script then passes `-DLLVM_USE_STATIC_ZSTD=OFF` and links the shared zstd (intel/llvm handles it from v7.1.1; [BUILD.md](BUILD.md#packages)).

Its configuration downloads what the release pins (Level Zero's headers and loader, emhash; all free software), so it needs the network.
The script makes it fetch Level Zero even when one is installed (`SYCL_UR_FORCE_FETCH_LEVEL_ZERO`):
without `pkg-config` the runtime adapter takes an installed loader without checking its version, and Debian 13's 1.20 failed to compile.

The toolchain is `.tools/intel-llvm/install/bin/clang++`, its runtime `.tools/intel-llvm/install/lib/libsycl.so`.
Configure the engine with `-DCMAKE_CXX_COMPILER=$PWD/.tools/intel-llvm/install/bin/clang++` and run it with that `lib/` on `LD_LIBRARY_PATH`.
On the development machine (28 threads, 91 GiB) v7.1.1 built in 13 minutes.
The clone (2.8 GB) and `install/` (0.7 GB) stay; the build tree is deleted.
This runtime gives the Arc Pro B70 XMX (FP16 and BF16); 6.2's does not.

### HIP (AMD GPUs)

Where ROCm's HIP is installed, the script builds the HIP target (AMD GPUs) as well, in the free mode and the contrib-llvm one.
HIP and ROCm are free software, in Ubuntu's universe and Fedora's repositories.

```sh
sudo apt install hipcc libamdhip64-dev libhsa-runtime-dev rocminfo   # Ubuntu
sudo dnf install hipcc rocm-hip-devel rocm-runtime-devel rocm-device-libs rocminfo   # Fedora 44
```

- ROCm is looked for in AMD's `/opt/rocm` (`/opt/rocm/core-7.14` with ROCm 7.14's packages from AMD) or the distribution's (headers in `/usr/include`, libraries in the multiarch folder or `/usr/lib64`).
  ROCm 7.14's packages from AMD install no ld.so.conf, so setup adds their `lib` to the engine's library path (the config's `lib_dirs`).
  Without it the build has no HIP.
- Code for AMD GPUs needs libclc's libspirv for AMD and ROCm's device libraries.
  A compiler reads bitcode of its own LLVM or an older one only (ROCm 10.1's device libraries are LLVM 24's, which intel/llvm 7.2.0's LLVM 23 cannot read),
  so the script fetches the device libraries of the ROCm release on intel/llvm's LLVM (ROCm 7.14.1: `amd/device-libs` of ROCm's llvm-project at the commit TheRock's tag `therock-7.14.1` pins, under the University of Illinois/NCSA license), builds them in the same tree and installs them into `install/amdgcn/bitcode`.
  It installs those a SYCL program's AMD code links (`ocml`, `ockl`, `oclc_*`) and `opencl`, whose presence the driver asks for.
  CMake takes the installed ROCm's device libraries where the compiler reads them, and these where it does not; when they fail to build, intel/llvm is still built.
- In a Fedora 44 container (ROCm 7.1.1), a SYCL kernel compiled with `-fsycl-targets=amd_gpu_gfx1200` ran correctly on an RX 9060 XT (gfx1200).
  The HIP adapter reports 1 GiB as the largest allocation, but 12 GiB could be allocated.

### The contrib-llvm build (CUDA)

The contrib-llvm build ([BUILD.md](BUILD.md#build-and-run)) also makes code for NVIDIA GPUs, with intel/llvm built with its CUDA target.
`--contrib` builds the same clone with the CUDA target (and HIP where ROCm is installed) into `.tools/intel-llvm-contrib/`.
The CUDA target needs NVIDIA's CUDA toolkit, which is not free software (Ubuntu: multiverse; Debian: non-free).

```sh
sudo apt install nvidia-cuda-toolkit           # 12.4 on Ubuntu 26.04
python3 tools/intel_llvm_build.py --contrib --keep-build
```

- On Ubuntu, ROCm's metapackages (`rocm`, `rocm-dev`) and `nvidia-cuda-toolkit` cannot be installed together.
  `librocthrust-dev` (pulled in by `rocm-dev`) and `libthrust-dev` (pulled in by `nvidia-cuda-dev`) both carry `/usr/include/thrust` and conflict, so installing one makes apt remove the other.
  HIP needs only the four packages above (not rocThrust).
  Packages that came in automatically with `rocm-dev` become `apt autoremove` candidates once it is gone; mark them with `sudo apt-mark manual hipcc libamdhip64-dev libhsa-runtime-dev rocminfo`.
- The script applies fixes the release lacks to the sources before building, as the patches in `third_party/main/intel-llvm/patches/` (`NN-<id>.patch`, in order; under intel/llvm's Apache-2.0 WITH LLVM-exception), and records their ids in `install/XESTRATA.json`.
  A build of the same release without them is offered a rebuild.
  There are twenty fixes now, for intel/llvm v7.2.0. Of the sixteen v7.1.1 took, libclc's AMD target name is no longer needed, libclc being built otherwise since v7.2.0, and the AMD FMA is fixed in v7.2.0 (2026-10-22).
  First, the CUDA and HIP adapters copied the table of sync points whole for every node they added to a command-buffer, so finalizing a SYCL graph took the square of its node count (a 2600-kernel graph: 90 ms on an RTX 4070, 4 ms fixed; Level Zero 2 ms).
  Second, the SYCL runtime gave every NVIDIA GPU the first NVIDIA image it found, whatever its architecture.
  With an executable carrying code for several architectures, a GPU older than that image did not run and a newer one ran older code.
  Fixed, the runtime hands the CUDA adapter the image itself, as it does the HIP one, and the adapter takes the PTX with the highest `.target` the GPU runs and refuses those for a newer one (`cuda-select-binary-2` and others; in a fatbin, whose PTX CUDA 13 compresses, it reads the entries' architecture field).
  Third, the CUDA adapter left host memory registration (`urUSMImportExp`, `urUSMReleaseExp`) out of its table, as functions doing nothing, so the loader refused it with `UR_RESULT_ERROR_UNINITIALIZED`.
  Fixed, it page-locks the memory with `cuMemHostRegister`, so copies from it are DMA transfers (`cuda-host-register` and another; about 4% faster decoding on an RTX 4070).
  Fourth, the SYCL runtime's table of NVIDIA architectures stops at sm_90, so a newer GPU (an RTX 50, compute capability 12.0) reported no matrix combinations and the engine took no tensor-core path.
  Fixed, a compute capability of 9.0 or more that the table lacks reports sm_90's combinations (the mma instructions of sm_80 on; `cuda-newer-matrix`, checked with an RTX 4070 made to report 12.0).
  Fifth, every link with an NVIDIA target left the driver's split bitcode, PTX and cubins in TMPDIR (22 GB of a tmpfs `/tmp` in a day).
  `sycl-post-link`'s table and `llvm-foreach`'s lists were not registered as the temporary-file types whose contents the driver removes; fixed, their registration takes those types (`cuda-temp-lists`; spir64 leaves nothing either way).
  Sixth, xptifw, built in the tree, includes LLVM's CMake settings (`HandleLLVMOptions`) a second time, which defined the link job pool twice.
  With the links limited (`LLVM_RAM_PER_LINK_JOB`), Ninja refused the build file and the configuration failed (`duplicate pool 'link_job_pool'`).
  Fixed, each pool is defined once (`build-job-pools`); a fix to the build only, it asks no finished toolchain to be built again.
  Seventh, the HIP adapter, like the CUDA one in the third, left host memory registration out of its table as functions doing nothing, so the loader refused it with `UR_RESULT_ERROR_UNINITIALIZED` (`pinned_shared_test` and `elementwise_parity` failed).
  Fixed, it page-locks the memory with `hipHostRegister` and has the functions in its table (`hip-host-register`; CTest passes on an RX 9060 XT).
  Eighth, the HIP adapter's queue-empty query took `hipErrorNotReady`, the answer for a stream still at work, as an error and logged it every time.
  Fixed, as in the CUDA adapter, it answers "not empty" without a log (`hip-queue-empty`).
  Ninth, at a process's exit the SYCL runtime unregisters its device images from the executable's finalizers after the HIP runtime has torn itself down, and glibc aborted with "double free or corruption" when the HIP adapter unloaded the module (every SYCL program that had used an AMD GPU ended with SIGABRT).
  Fixed, a program released after the exit began is not unloaded (`hip-exit-unload`).
  Tenth, a kernel bundle (`get_kernel_bundle`) of a program with code for several AMD architectures got the first architecture's image, which the GPU refused (`hipErrorInvalidImage`).
  AMD images carry no `compile_target`; the runtime asked the HIP adapter about each image alone without handing it the image, and the adapter falls back on the first AMD image when none matches.
  An engine for many AMD GPUs, as the packages build it, stopped with no usable GPU (kernels submitted to a queue were not affected: that path hands the adapter all the images and their bytes).
  Fixed, the runtime hands the HIP adapter the image too, as it does the CUDA one, and the adapter does not fall back on a clang offload bundle without the GPU's architecture (`hip-bundle-arch`).
  Eleventh, a program with code for several NVIDIA or AMD architectures had another architecture's device code under each one's name.
  SYCL's bundle IDs carry the GPU without an environment field before it (`sycl-amdgcn-amd-amdhsa-gfx1030`), and clang-offload-bundler read the GPU as the triple's environment.
  The target ID was empty and every architecture of a triple matched every other, so unbundling an object or an archive gave each architecture's link the triple's first unclaimed entry in hash order.
  The packages' engine had gfx1200's code under gfx1030, gfx1030's under gfx1101 and gfx1100's under gfx1201, which an RX 9070 XT refused as an illegal instruction.
  Fixed, a fifth component that names a GPU is read as the target ID (`offload-bundle-arch`).
  Twelfth, the HIP adapter answered a GPU's free memory (SYCL's `ext_intel_free_memory`) with `hipMemGetInfo` without making that GPU current, so every AMD GPU reported the free memory of the thread's current one.
  A layer split over two AMD GPUs saw the first stage's free memory, nearly all of it held by its expert cache, on the second stage, which found no room for its own cache and stopped.
  Fixed, the GPU is made current before the query, as the adapter's other calls do (`hip-mem-free-device`; the CUDA adapter makes the GPU's context current for every query).
  Thirteenth, where the system's Level Zero is older than 1.32 (Ubuntu 26.04's libze 1.28, where the packages are built), the Level Zero adapters link the loader statically, and the static loader hands the stable record and replay graph functions on to the system's loader at run time.
  The 1.28 loader lacks them, and the static one answered the first call UNSUPPORTED and called a null pointer on the next (the static loader's ze_libapi.cpp, 1.32 to 1.34).
  The adapter took the functions as there, the static loader's own entry points being not null, and with a driver that announces the stable graph extension (compute-runtime 26.35.39758.10) the query made at every submission to a queue (`urQueueIsGraphCaptureEnabledExp` in `handler::finalize`) went through.
  oneMKL's GEMM takes that path, so with oneMKL installed the engine stopped with SIGSEGV at its first product on an Arc Pro B70 (XeStrata issue #12).
  Fixed, with the static loader the graph functions are used only when the system's loader (`libze_loader.so.1`, `ze_loader.dll`) has them (`l0-static-graph`).
  Fourteenth, the device libraries (libdevice) depended on the compiler being built and all its tools (`sycl-compiler`) even when `LIBCLC_CUSTOM_LLVM_TOOLS_BINARY_DIR` gives the tools that compile them.
  The cross build for Windows (`--windows`) built a Windows `clang.exe` and tools nobody runs, most of its time.
  Fixed, with the tools given they depend on the SYCL headers alone (`build-libdevice-custom-tools`; a build without them is unchanged). A fix to the build only, it asks no finished toolchain to be built again.
  Fifteenth, v7.2.0's `configure.py` builds libclc's AMD libspirv as the LLVM runtime target `amdgcn-amd-amdhsa-llvm`, into `lib/clang/<N>/lib/amdgcn-amd-amdhsa-llvm`, but the driver compiles SYCL's AMD device code for `amdgcn-amd-amdhsa` and looks for `libspirv.l64.signed_char.bc` under that name.
  Every compile for an AMD GPU stopped ("cannot find 'libspirv.l64.signed_char.bc'"), and with the folder linked under the other name, every link warned that the two modules' targets differ.
  Fixed, the AMD libspirv is built for `amdgcn-amd-amdhsa` (`hip-libclc-triple`).
  Sixteenth, since v7.2.0 the SYCL build refuses the CUDA and HIP adapters unless libclc is built as an LLVM runtime.
  The cross build for Windows builds the runtime alone, with the contrib-llvm toolchain's device tools and its libspirv, and builds no libclc, so its configuration stopped.
  Fixed, with the tools given (`LIBCLC_CUSTOM_LLVM_TOOLS_BINARY_DIR`) the adapters are configured without libclc (`build-cross-adapters`; a build without them is unchanged). A fix to the build only, it asks no finished toolchain to be built again.
  Seventeenth, since v7.2.0 SYCL's sub-group shuffles (Shuffle, ShuffleXor, ShuffleUp and ShuffleDown in `sycl/detail/spirv.hpp`) call the SPIR-V GroupNonUniform forms on every target but NVIDIA's, and AMD's libspirv has only the SubgroupShuffle*INTEL ones.
  Every AMD program with a shuffle stopped at its link ("undefined symbol: __spirv_GroupNonUniformShuffleXor(int, float, unsigned int)").
  Fixed, AMD's libspirv defines the four GroupNonUniform shuffles for the scalar types with SubgroupShuffle*INTEL (`hip-group-shuffle`).
  Eighteenth, the HIP adapter answered every virtual memory call (`urVirtualMem*`, `urPhysicalMem*`) with UNSUPPORTED and reported no support, so SYCL's `ext_oneapi_virtual_mem` was not there on AMD GPUs, and `--vram-elastic` and `--kv-grow` fell back to one allocation.
  intel/llvm implemented them after the release (commit c13ad6c7e3, "[UR][HIP] Implement virtual and physical memory management", pull request 22671): with HIP's virtual memory (`hipMemAddressReserve`, `hipMemCreate`, `hipMemMap` and the others) as the CUDA adapter implements them with CUDA's, the support reported as HIP's `hipDeviceAttributeVirtualMemoryManagementSupported` says.
  The patch applies that commit's diff to v7.2.0 as it is (`hip-virtual-mem`; nothing of XeStrata's in it, and it goes when intel/llvm is raised to a release that has the commit).
  Nineteenth, a SYCL kernel that asks for a global id calls `__spirv_BuiltInGlobalOffset`, which the GlobalOffset pass replaces in codegen (with a parameter of the kernel's `_with_offset` clone, or 0).
  The AMDGPU attributor runs before that and took the call as an unknown callee, so every such kernel kept all the implicit inputs, the hostcall buffer among them (14,304 of XeStrata 0.2.2's 32,556 AMD kernels; 0 of 48,084 with v7.1.1).
  HIP launches no kernel with a hostcall buffer on a GPU without PCIe atomics (one passed through to a virtual machine, or in a slot behind a chipset), so the first kernel stopped with `hipErrorIllegalState` (XeStrata's issues #40 and #54).
  Fixed, the attributor takes the builtin as reading no implicit argument (`amdgpu-global-offset-hostcall`).
  Twentieth, linked without RDC (`-fno-sycl-rdc`), a static library's device code goes through `sycl-post-link` once per object in it, and the output is a list of file tables.
  The SPIR-V path extracts the code column of each table and merges the lists and the tables; the NVIDIA and AMD path extracted the code column from the list itself, so every link of a program with a static library of SYCL code for an NVIDIA or AMD GPU stopped with "column not found Code".
  Fixed, that path extracts and merges as the SPIR-V one does (`offload-nordc-archive`).
  Twenty-first, on Linux a range unmapped with `hipMemUnmap` and mapped again on other physical memory was still reached through its old translation by the GPU's engines that had reached it before.
  The copy engine wrote a copy into the range to the physical memory that had been given back, and a kernel read what was there before (on the RX 9060 XT with ROCm 7.1.1 and Linux 7.2; `--vram-elastic`, `--kv-grow` and the draft head parked during prefill run into it).
  ROCm unmaps virtual memory with the DRM driver's GEM_VA call, and for an address space of compute queues whose page tables amdgpu writes from the CPU (its default where all of the VRAM is visible to the CPU) that call clears the page table entries and flushes no TLB.
  The flush comes with the next mapping KFD itself makes.
  Fixed, right after the unmap `urVirtualMemUnmap` allocates and frees 4 MiB of device memory on each of the context's devices, which KFD maps (larger than the 2 MiB blocks ROCr serves smaller allocations from without calling KFD; where a device has no memory for it, pinned host memory of that size).
  It costs about 0.1 ms an unmap (`hip-virtual-mem-unmap`; `vmm_test` checks it).
  Twenty-second, with a ROCr that leaves its own mapping at an unmapped range, the physical memory of a range unmapped with `hipMemUnmap` stayed allocated after `hipMemRelease` (the device's free memory did not grow, and neither another program nor another allocation of this process could take it; seen with ROCm 7.1.1 on the RX 9060 XT).
  `--vram-elastic` could not lend VRAM to an image job (the engine sees it at start and refuses), and the draft head parked during prefill made no room.
  ROCr maps the physical memory to the CPU at the same range, inaccessible, when it maps it to the GPU (a mapping of the DRM device). ROCm 7.1.1's ROCr does not take it away at the unmap, and the physical memory stays until the range is mapped again or freed. ROCm 10.1's ROCr makes the range an anonymous mapping again.
  Fixed, `urVirtualMemUnmap` looks at the range in `/proc/self/maps` after the first unmap, and where the DRM device's mapping is still there it makes every unmapped range the anonymous inaccessible mapping it was reserved with from then on. The physical memory goes back when it is released (`hip-virtual-mem-release`).
  Where the range is anonymous again (or its line is not found, or is of another file) it does nothing. What the runtime does decides, not ROCm's version, and `UR_LOG_HIP=level:info` says which it was.
  Twenty-third, intel/llvm's `urPhysicalMemCreate` (the eighteenth's commit) calls `hipMemCreate` without making its device current.
  `hipMemCreate` takes the memory from the current device whatever device its properties name (ROCm 7.1.1 and 10.1), so on a PC with two AMD GPUs a thread whose current device is the other one gets that one's memory.
  Fixed, it makes its device current first (`hip-physical-mem-device`).
  The CUDA fixes ask only the contrib-llvm toolchain to be built again, the HIP ones and `amdgpu-global-offset-hostcall` only a toolchain with HIP (or one that would have it, ROCm being installed now).
  `offload-bundle-arch`, `l0-static-graph` and `offload-nordc-archive` ask every toolchain.
- The script configures the build tree again when it was configured with other values of its options (the install folder, libclc's targets and others).
- On the development machine (Ubuntu 26.04, CUDA 12.4, ROCm 7.1) the runtime's backends were cuda, hip, level_zero and opencl.
- Running on an NVIDIA GPU needs NVIDIA's driver (on Ubuntu 26.04, `nvidia-driver-610-open` or another).

## Faster builds (Ninja and ccache)

For the incremental builds of development, Ninja takes the place of Make and ccache goes in front of the compiler.
Both change the build folder's configuration only, not the sources or the package builds.

- Ninja: decides what to rebuild and fills the parallel jobs faster than Make, so a small change waits less.
- ccache: keeps the result of compiling the same source with the same settings and returns it the next time.
  It helps with a file changed and changed back, and with a build folder made again.
  A file whose content changed is compiled every time.

```sh
sudo apt install ninja-build ccache
```

Configure with `-G Ninja -DCMAKE_CXX_COMPILER_LAUNCHER=ccache` added.
An existing build folder cannot change its generator, so make it again (for `build/contrib-llvm`: delete it and configure with the same options).

```sh
cmake -S . -B build/contrib-llvm -G Ninja -DCMAKE_CXX_COMPILER_LAUNCHER=ccache <the other options>
cmake --build build/contrib-llvm -j 28
ccache -s          # the hits
```

## Building the packages (Podman)

The deb and rpm packages and the Windows 7z (`tools/package/build.sh`, [BUILD.md](BUILD.md#the-deb-and-rpm-packages)) are built in rootless Podman containers.

```sh
sudo apt install podman               # 5.7.0 (Fedora: sudo dnf install podman)
grep "^$USER:" /etc/subuid /etc/subgid   # if nothing: sudo usermod --add-subuids 100000-165535 --add-subgids 100000-165535 "$USER"
```

Rootless, it needs neither sudo nor a group with root's power, and the container's root is this user (what the container writes is this user's).
The containers get no GPU, so nothing extra for GPUs (such as nvidia-container-toolkit) is needed.
A build makes intel/llvm too; for Fedora 44's free packages the container grew to 6.8 GB at most (measured every minute; the cuda variants add the CUDA toolkit). The container is removed at the end. The images and containers are kept in `~/.local/share/containers`.

## Windows cross build

The Windows engine and tests are built on Linux, up to the link (`cmake/toolchain-windows-msvc.cmake`).
intel/llvm's SYCL runtime takes only the MSVC ABI on Windows, so the target is `x86_64-pc-windows-msvc`.
intel/llvm's clang++ compiles and lld-link links.
What it builds does not run here: running it needs a Windows PC.

It needs three things:

- lld-link (LLVM's `lld`): `sudo apt install lld-21` (21.1)
- the MSVC CRT and the Windows SDK: downloaded from Microsoft with [xwin](https://github.com/Jake-Shadle/xwin) (0.10.0).
  They are not free software; Microsoft's license is accepted with `--accept-license`. They do not go into the repository.
- intel/llvm's Windows SYCL tree (headers, import libraries, the runtime's DLLs): `tools/intel_llvm_build.py --windows` cross-builds the release XeStrata uses (v7.2.0), with XeStrata's patches, for Windows here ("intel/llvm for Windows" below).

```sh
mkdir -p .tools/xwin
curl -L https://github.com/Jake-Shadle/xwin/releases/download/0.10.0/xwin-0.10.0-x86_64-unknown-linux-musl.tar.gz |
    tar -xz -C .tools/xwin
.tools/xwin/xwin-0.10.0-x86_64-unknown-linux-musl/xwin --accept-license --cache-dir .tools/xwin/cache --arch x86_64 \
    splat --output .tools/xwin/sdk
python3 tools/intel_llvm_build.py --contrib && python3 tools/intel_llvm_build.py --windows
export CCACHE_DIR=$PWD/build/win-ccache CCACHE_COMPILERCHECK=content
cmake -S . -B build/win -G Ninja -DCMAKE_BUILD_TYPE=Release -DSTRATA_BUILD_TESTS=ON \
    -DCMAKE_TOOLCHAIN_FILE=cmake/toolchain-windows-msvc.cmake \
    -DCMAKE_CXX_COMPILER=.tools/intel-llvm/install/bin/clang++ -DSTRATA_LICENSE=free \
    -DSTRATA_XWIN=$PWD/.tools/xwin/sdk -DSTRATA_SYCL_WINDOWS=$PWD/.tools/intel-llvm-win/install \
    -DCMAKE_CXX_COMPILER_LAUNCHER=ccache -DCMAKE_C_COMPILER_LAUNCHER=ccache
ninja -C build/win
```

For an everyday change, only the objects of the sources it touches are compiled ([AGENTS.md](../AGENTS.md#operating-systems-linux-main-and-windows)).
ninja builds a source's object when given the source's path with `^` after it (a full path).

```sh
export CCACHE_DIR=$PWD/build/win-ccache CCACHE_COMPILERCHECK=content
ninja -C build/win $(git diff --name-only HEAD -- '*.cpp' | sed "s|^|$PWD/|; s|$|^|")
```

- ccache is the one of "Faster builds" above. The Windows objects never hit the Linux ones (their settings differ), so a `CCACHE_DIR` of their own keeps them from pushing the Linux cache out. `CCACHE_COMPILERCHECK=content` tells the compiler by its content when intel/llvm is built again.
- When only a header changed, `ninja -C build/win` builds everything (ccache returns the files that did not change).
- xwin's `cache` is not needed after the splat and may be deleted (`sdk` is 0.8 GB).
- lld-link warns `skipping imported DllMain symbol` about `sycl9.lib`: it comes from lld reading an import library MSVC made, and does not change the link.

### intel/llvm for Windows

`tools/intel_llvm_build.py --windows` builds the SYCL runtime for Windows from the same clone as the Linux one, into `.tools/intel-llvm-win/install`.
It builds the parts the engine uses only (`sycl9.dll` with UR's loader in it, the CUDA, HIP and Level Zero adapters, `xptifw.dll`, SYCL's device libraries (libdevice), the headers and the import libraries): no Debug DLLs, no `clang.exe` and no sycl-jit.

- The clang-cl of the toolchain `--contrib` built, of the same release, compiles and lld-link links. The tools that make the device libraries are that toolchain's too (`--contrib`'s build also copies `opt`, which it does not install, into its install folder). No compiler is built, for Windows or for this machine. Build `--contrib` first.
- The MSVC CRT and the Windows SDK are xwin's splat (`--xwin`, by default `.tools/xwin/sdk`). `llvm-lib`, `llvm-rc`, `llvm-mt` and `llvm-ml` are the distribution's LLVM's (`sudo apt install llvm`).
- `sycl9.dll` links zstd statically (BSD-3-Clause; its license text is `install/licenses/zstd-LICENSE`). oneMKL's SYCL DLLs carry zstd-compressed device images, and with a runtime without zstd their load fails with WinError 1114. The script fetches the zstd release it names, checks its SHA-256 and builds it for Windows with the same toolchain. The install also gets Level Zero's headers (MIT), for the engine's calls into the loader.
- The Windows CUDA headers and import libraries the CUDA adapter links come from NVIDIA's redistributable archives (CUDA 12.4.1's `cuda_cudart`, `cuda_nvml_dev` and `cuda_cccl`). They are not free software and do not go into the 7z.
- The HIP adapter (AMD GPUs) is built where this machine has ROCm's HIP (Ubuntu's `libamdhip64-dev`) and `--contrib` was built with the HIP target. Its headers are that HIP's (MIT; the license text in `install/licenses/hip-headers.txt`), and the `amdhip64.lib` it links is an import library of `amdhip64_7.dll`, made with `llvm-dlltool` from the C names Linux's `libamdhip64.so.7` exports. AMD's driver (Adrenalin 26.5.1 and later) puts `amdhip64_7.dll` into System32.
- A Windows engine's code for NVIDIA and AMD GPUs links the Windows l32 libspirv the Linux toolchain builds since v7.2.0 (`lib/clang/<version>/lib/<target>`) and that toolchain's ROCm device libraries: neither carries `wchar_size`, so neither is built again for Windows.
- It takes about three minutes on the development machine (28 threads).

For NVIDIA GPUs, configure the cross build with `-DSTRATA_LICENSE=contrib -DSTRATA_CUDA_ARCHS=<the GPU's architecture> -DSTRATA_CUDA_PATH=<a CUDA toolkit>`, a CUDA toolkit no newer than the version the target PC's driver supports.
For AMD GPUs add `-DSTRATA_HIP_ARCHS=<the gfx names>` (RDNA2 and later, as on Linux).
For oneMKL add `-DSTRATA_ONEMKL=ON -DMKL_ROOT=<oneMKL for Windows' Library/>`: oneMKL's Windows headers, import libraries and CMake package are in the `Library/` of Intel's PyPI wheels (`mkl-devel`, `mkl-include`, `mkl-devel-dpcpp` and `onemkl-sycl-include`, win_amd64). On Windows oneMKL's CMake package also checks that the DLLs are there, so the `mkl` and `onemkl-sycl-*` wheels' DLLs are unpacked into the same `Library/` (for the build only; the 7z does not carry them). It builds oneMath and its oneMKL backend (`onemath.dll`, `onemath_blas_mklgpu.dll`); on Windows the engine looks for oneMKL of the series it was built with first (the `Library\bin` pip makes in the bundled Python, oneAPI's `mkl\<series>\bin`), before the PATH, and then in oneAPI's `mkl` and `tbb` folders (`ONEAPI_ROOT`, else the installer's default folder) and `MKLROOT`.
`tools/package/build.sh windows cuda12.4` builds all of this, and the Windows 7z, in a disposable Ubuntu 26.04 container ([BUILD.md](BUILD.md)).

### Running on a Windows PC

Put `build/win`'s `xestrata*.exe` in one folder with `.tools/intel-llvm-win/install/bin`'s `sycl9.dll`, `ur_win_proxy_loader.dll`, the adapter of the GPU (`ur_adapter_cuda.dll` for NVIDIA, `ur_adapter_level_zero.dll` for Intel) and `xptifw.dll`.
When `xestrata-device.exe --list-devices` lists the GPU, the engine runs.

## Intel SDE (the AVX-512 paths)

The development machine has no AVX-512.
[AGENTS.md](../AGENTS.md#avx-512-code) says when and how to run the tests under SDE.
SDE is not free software.
Download it by hand from [Intel](https://www.intel.com/content/www/us/en/developer/articles/tool/software-development-emulator.html) (its license is accepted there),
unpack it outside the repository and put a small wrapper on the `PATH`:

```sh
mkdir -p ~/.local/opt ~/.local/bin
tar -xf sde-external-*-lin.tar.xz -C ~/.local/opt                          # 10.13.1 was used
printf '#!/bin/sh\nexec %s/sde64 "$@"\n' ~/.local/opt/sde-external-*-lin > ~/.local/bin/sde64
chmod +x ~/.local/bin/sde64
sde64 -version
```

## Intel Graphics Compiler (IGC): the GEP simplification of 2.27 to 2.30

IGC 2.27.10, 2.28.4 and 2.30.1 miscompile, for Xe2 and Xe3 GPUs (the targets `bmg`, `lnl` and `ptl`: the Arc B-series, Lunar Lake, Panther Lake), a kernel that reads from one pointer at a runtime index and at constant multiples of it.
With `const float* xt = x + base; xt[0] + xt[N] + xt[2 * N] + xt[3 * N]`, intel/llvm's optimizer writes `xt[2 * N]` and `xt[3 * N]` as byte offsets (8N and 12N bytes) from the pointer it already offset for `xt[N]`, and IGC's GEP simplification takes those byte offsets for element indices.
The kernel reads `xt[8 * N]` and `xt[9 * N]`: a fault past the buffer (`DEVICE_LOST`, `Engine memory CAT error` in the kernel's log) or wrong values inside it.
2.24.8 and earlier and 2.32.7 and later compile the reads right (each release's libigc from GitHub on the reproducer's SPIR-V, 2026-10-19); Alchemist (`dg2`) is not affected (IGC reads its buffers there another way).
Ubuntu 26.04, Debian and Arch ship 2.28.4; Fedora 44 ships 2.24.8 and updates to 2.41.9.
The reproducer and the checks are in the data folder's `records/igc-gep-check/`.

### The engine's workaround

- The runtime sets `IGC_EnableGEPSimplification=0` in its environment before the SYCL runtime lists the devices (`engine_devices` in `src/core/device.cpp`), unless the user set the variable; IGC reads it when it first compiles a kernel, which the runtime does as it lists the devices.
- Off, the pass costs the B70 5 to 10% of the decode of a 1K prompt and nothing at 32K (generate, 2026-10-19), so the variable is set only where the `libigc.so.2` the loader finds is 2.25 to 2.31 by its file name (`libigc.so.2.28.4+0`) or carries no version, and a GPU on the xe driver is Xe2 or later by its graphics IP (`xe_gpu_from_xe2`: the main GT's `ip_ver_major` of `DRM_XE_DEVICE_QUERY_GT_LIST` 20 or more; 0, a GPU without GMD_ID, is Alchemist or older; asked of the kernel driver before Level Zero loads).
- Linux only: Windows sets nothing and keeps its driver's compiler as it is.
- The start log says so (`IGC's GEP simplification is off`), the engine reports it as `INFO igc_workaround=<version>` (the web app's notice and its About page), and `xestrata --setup` (`check_igc` in `setup.py`, on the same two conditions) warns of it. They send the user to Fedora 44 (IGC 2.32 or later) or the Podman container (Fedora 44 inside); setup, the documents (BUILD's packages, DETAILS's troubleshooting) and the release notes name Intel's graphics PPA as well (compute-runtime 26.31 and IGC 2.40 for Ubuntu 24.04/26.04): with the HWE kernel straight to the PPA, from the GA kernel to the HWE kernel first (the PPA installs the driver with the IGC, and that driver is the one for the HWE kernel).
- The same `xestrata --setup` (`check_level_zero`) stops the start when the Level Zero driver predates the build that stops it refusing device memory while the VRAM is free (26.22.38646.4; intel/compute-runtime #916, xe's kernel bug fixed in Linux 7.1; on XeStrata's side #34 and #15) on Ubuntu 24.04/26.04 with the HWE kernel (`--old-driver` starts anyway); on the GA kernel and elsewhere it only says so and goes on.

### Writing and checking kernels

- Write kernels that compile right either way: step a pointer by N from the last read (`glm5_hc_mean`), or index the stream in one product (`x[(t * HC + s) * N + n]`), not `xt[k * N]` with a constant k of 2 or more beside `xt[N]`.
- To see what IGC makes of a kernel without running it, compile ahead for Xe2 with IGC's dumps on (`IGC_ShaderDumpEnable=1 IGC_DumpToCustomDir=dump clang++ -fsycl -fsycl-targets=intel_gpu_bmg_g31 ...`) and read `dump/*_entry_*.asm`: the miscompile shows as a scalar `mul ... 0x1C:uw` (28N: the 32N bytes of the misread `xt[8 * N]`, less the 4N of `xt[N]`) ahead of the loads; `*_beforeUnification.ll` is the IR IGC received.
- `IGC_EnableGEPSimplification=1 ./igc_gep_probe` runs the pattern on the GPU with room for the wrong reads and fails on an affected IGC; CTest runs it through the runtime's workaround, where it passes on every GPU.

## Measuring the GPU

### What works without extra tools

- The engine's own timings: `XESTRATA_PREFILL_TIMING=1` for the prefill path.
  The decode's `verify window` line (the host's wait for the GPU, the CPU pool) is always printed.
- `XESTRATA_VERIFY_NODES=1` prints how many kernels a decode window's graph holds.
- Scratch probes that time single kernels with SYCL event profiling.
- `XESTRATA_VERIFY_PROFILE=1` prints the stages of a decode window's layers (device clock ticks) on the `verify profile` line of `--stats`.
  Level Zero does not report the device clock aspect on the B70, but its compiler runs SPIR-V's OpReadClockKHR, so the engine builds the stamp kernel once at start and uses it where that works (NVIDIA: %globaltimer; AMD: no profile).
  On the B70 the tick is the GPU's core clock (2.8 GHz against a bandwidth kernel of known length, its `max_freq`), slower when the clock drops, so read the stages as shares.

### The xe driver's kernel log

The xe driver reports its errors in the kernel log.
`journalctl -k` reads it without root (`dmesg` needs root):

```sh
journalctl -k --since "-1h" --no-pager | grep "xe 0000"
```

`VM worker error: -16` there came about 8 s before each start that had the output head's VRAM refused.

### VTune and the GPU's hardware counters

VTune is not free software; it comes from oneAPI's package repository (`sudo apt install intel-oneapi-vtune`; 2026.4 was used).
Reading the GPU's hardware counters (memory bandwidth per kernel) needs three things:

1. Membership in the `render` group, and both drivers' observation setting open.
   The B70 runs on xe; VTune checks i915's setting as well.
   They last until the next boot:

   ```sh
   sudo sysctl dev.xe.observation_paranoid=0 dev.i915.perf_stream_paranoid=0
   ```

   For every boot, put the same two lines (`dev.xe.observation_paranoid = 0` ...) in `/etc/sysctl.d/60-gpu-observation.conf` and add a udev rule that applies them again when a GPU device appears.
   The boot-time sysctl pass runs before xe and i915 are ready, so the file alone finds no such keys and is ignored (on the development machine the drivers initialized a second after `systemd-sysctl` reported `No such file or directory`).

   ```sh
   printf 'dev.xe.observation_paranoid = 0\ndev.i915.perf_stream_paranoid = 0\n' | sudo tee /etc/sysctl.d/60-gpu-observation.conf
   echo 'ACTION=="add", SUBSYSTEM=="drm", KERNEL=="card*", RUN+="/usr/lib/systemd/systemd-sysctl --prefix=/dev/xe --prefix=/dev/i915"' | sudo tee /etc/udev/rules.d/60-gpu-observation.rules
   ```

1. Intel's Metrics Discovery library (MIT; Ubuntu does not package it), built into `.tools/`:

   ```sh
   sudo apt install libdrm-dev
   git clone --depth 1 https://github.com/intel/metrics-discovery.git .tools/src/metrics-discovery
   cmake -S .tools/src/metrics-discovery -B .tools/src/metrics-discovery/build -DCMAKE_BUILD_TYPE=Release
   cmake --build .tools/src/metrics-discovery/build -j
   ```

   It builds `libigdmd.so` into `.tools/src/metrics-discovery/dump/linux64/release/metrics_discovery/`, which goes on the library path when VTune runs:

   ```sh
   export LD_LIBRARY_PATH=$PWD/.tools/src/metrics-discovery/dump/linux64/release/metrics_discovery:$LD_LIBRARY_PATH
   ```

1. `vtune -collect gpu-hotspots -r <result folder> -- <program>` collects;
   `vtune -report hotspots -r <result folder> -group-by computing-task -format csv` reports each kernel's time, its XVE pipelines
   (ALU0 for floating point, ALU1 for integer and extended math), its stall reasons and its memory bandwidth.
   Result folders (`r000hs/` and so on) are ignored by git.

Metrics Discovery read the B70's counters through the xe driver (Linux 7.0, 2026-10-02).
VTune's `xpu-offload` collection and unitrace both left the engine stalled in prefill for minutes.
They are for small programs and probes, not the engine.

### Measuring NVIDIA GPUs (nsys and ncu)

Nsight Systems (`nsys`, time per kernel) and Nsight Compute (`ncu`, memory bandwidth and stall reasons inside a kernel) come with `nvidia-cuda-toolkit` (2023.4 and 2024.1 on Ubuntu 26.04).
Neither is free software.

- `ncu` reads the GPU's hardware counters, which need root by default (`RmProfilingAdminOnly: 1` in `/proc/driver/nvidia/params`).
  To open them to users, set the driver option, rebuild the initramfs and reboot.
  Ubuntu puts the NVIDIA modules in the initramfs and loads them early, so the option file alone has no effect.

  ```sh
  echo 'options nvidia NVreg_RestrictProfilingToAdminUsers=0' | sudo tee /etc/modprobe.d/nvidia-profiling.conf
  sudo update-initramfs -u
  ```

- Ubuntu's `nsys` 2023.4 fails to convert a SYCL program's recording to `.nsys-rep`.
  Recorded with `--cuda-graph-trace=node` and converted by hand, it reads.
  The converter does not overwrite an existing `.nsys-rep`, so delete it first.

  ```sh
  nsys profile -t cuda --cuda-graph-trace=node -o out -f true <program>
  /usr/lib/nsight-systems/host-linux-x64/QdstrmImporter -i out.qdstrm
  nsys stats -r cuda_gpu_kern_sum out.nsys-rep
  ```

- Under `ncu` the engine stopped once it had captured its first verify window.
  The kernel in which the GPU waits for the host (`wait_flag_ge`) presumably cannot finish while `ncu` measures one kernel at a time.
  Use `ncu` on the parity tests and probes, not on the engine.
