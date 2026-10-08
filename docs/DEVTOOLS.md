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
| `.tools/` | other tools built from source (intel/llvm, Metrics Discovery) |

## Lints

[AGENTS.md](../AGENTS.md#lints) says which lints to run and how (`tools/lint/run.sh`).

From the distribution:

```sh
sudo apt install gitleaks shellcheck clang-tidy cppcheck flake8 mypy python3-pyflakes codespell \
    markdownlint cmake-format tidy eslint
```

`tools/lint/run.sh` runs oneAPI's clang-tidy, which knows SYCL (found under `/opt/intel/oneapi/compiler`), not the distribution's.
clang-tidy reads `compile_commands.json` from the build folder:

```sh
cmake -S . -B build/xe -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
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
It needs these packages (on Fedora 44: `git cmake ninja-build gcc-c++ hwloc-devel libzstd-devel libzstd-static python3`):

```sh
sudo apt install git cmake ninja-build g++ python3 libhwloc-dev libzstd-dev
python3 tools/intel_llvm_build.py            # --keep-build keeps the build tree for a quicker update; --jobs N
```

Its configuration downloads what the release pins (Level Zero's headers and loader, emhash; all free software), so it needs the network.
The script makes it fetch Level Zero even when one is installed (`SYCL_UR_FORCE_FETCH_LEVEL_ZERO`):
without `pkg-config` the runtime adapter takes an installed loader without checking its version, and Debian 13's 1.20 failed to compile.

The toolchain is `.tools/intel-llvm/install/bin/clang++`, its runtime `.tools/intel-llvm/install/lib/libsycl.so`.
Configure the engine with `-DCMAKE_CXX_COMPILER=$PWD/.tools/intel-llvm/install/bin/clang++` and run it with that `lib/` on `LD_LIBRARY_PATH`.
On the development machine (28 threads, 91 GiB) v7.1.1 built in 13 minutes.
The clone (2.8 GB) and `install/` (0.7 GB) stay; the build tree is deleted (its size was not measured).
This runtime gives the Arc Pro B70 XMX (FP16 and BF16); 6.2's does not.

### HIP (AMD GPUs)

Where ROCm's HIP is installed, the script builds the HIP target (AMD GPUs) as well, in the free mode and the contrib one.
HIP and ROCm are free software, in Ubuntu's universe and Fedora's repositories.

```sh
sudo apt install hipcc libamdhip64-dev libhsa-runtime-dev rocminfo   # Ubuntu
sudo dnf install hipcc rocm-hip-devel rocm-runtime-devel rocm-device-libs rocminfo   # Fedora 44
```

- ROCm is looked for in AMD's `/opt/rocm` or the distribution's (headers in `/usr/include`, libraries in the multiarch folder or `/usr/lib64`).
  Without it the build has no HIP.
- Code for AMD GPUs needs libclc's libspirv for AMD and ROCm's device libraries.
  Fedora's device libraries are in `/usr/lib64/rocm/llvm/lib/clang/20/lib/amdgcn/bitcode`; give them to the compiler with `--rocm-device-lib-path`.
- In a Fedora 44 container (ROCm 7.1.1), a SYCL kernel compiled with `-fsycl-targets=amd_gpu_gfx1200` ran correctly on an RX 9060 XT (gfx1200).
  The HIP adapter reports 1 GiB as the largest allocation, but 12 GiB could be allocated.

### The contrib build (CUDA)

The contrib build ([BUILD.md](BUILD.md#build-and-run)) also makes code for NVIDIA GPUs, with intel/llvm built with its CUDA target.
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
  There are twelve fixes now, none fixed in intel/llvm's `sycl` branch either (2026-10-08).
  First, the CUDA and HIP adapters copied the table of sync points whole for every node they added to a command-buffer, so finalizing a SYCL graph took the square of its node count (a 2600-kernel graph: 90 ms on an RTX 4070, 4 ms fixed; Level Zero 2 ms).
  Second, the SYCL runtime gave every NVIDIA GPU the first NVIDIA image it found, whatever its architecture.
  With an executable carrying code for several architectures, a GPU older than that image did not run and a newer one ran older code.
  Fixed, the runtime hands the CUDA adapter the image itself, as it does the HIP one, and the adapter takes the PTX with the highest `.target` the GPU runs and refuses those for a newer one (`cuda-select-binary-2` and others; in a fatbin, whose PTX CUDA 13 compresses, it reads the entries' architecture field).
  Third, the CUDA adapter left host memory registration (`urUSMImportExp`, `urUSMReleaseExp`) out of its table, as functions doing nothing, so the loader refused it with `UR_RESULT_ERROR_UNINITIALIZED`.
  Fixed, it page-locks the memory with `cuMemHostRegister`, so copies from it are DMA transfers (`cuda-host-register` and another; about 4% faster decoding on an RTX 4070).
  Fourth, the SYCL runtime's table of NVIDIA architectures stops at sm_90, so a newer GPU (an RTX 50, compute capability 12.0) reported no matrix combinations and the engine took no tensor-core path.
  Fixed, a compute capability of 9.0 or more that the table lacks reports sm_90's combinations (the mma instructions of sm_80 on; `cuda-newer-matrix`, checked with an RTX 4070 made to report 12.0; on such a GPU itself `unverified`).
  Fifth, every link with an NVIDIA target left the driver's split bitcode, PTX and cubins in TMPDIR (22 GB of a tmpfs `/tmp` in a day).
  `sycl-post-link`'s table and `llvm-foreach`'s lists were not registered as the temporary-file types whose contents the driver removes; fixed, their registration takes those types (`cuda-temp-lists`; spir64 leaves nothing either way).
  Sixth, xptifw, built in the tree, includes LLVM's CMake settings (`HandleLLVMOptions`) a second time, which defined the link job pool twice.
  With the links limited (`LLVM_RAM_PER_LINK_JOB`), Ninja refused the build file and the configuration failed (`duplicate pool 'link_job_pool'`).
  Fixed, each pool is defined once (`build-job-pools`); a fix to the build only, it asks no finished toolchain to be built again.
  Seventh, libclc builds the AMD libspirv only for the target name `amdgcn--amdhsa`, but left that name out of the targets it accepts.
  Configured with another name (`amdgcn-amd-amdhsa`), the build went through without an AMD libspirv, and no SYCL program for AMD GPUs compiled.
  Fixed, `amdgcn--amdhsa` is accepted (`hip-libclc-target`; the `sycl` branch has reorganized libclc's AMD target).
  Eighth, the HIP adapter, like the CUDA one in the third, left host memory registration out of its table as functions doing nothing, so the loader refused it with `UR_RESULT_ERROR_UNINITIALIZED` (`pinned_shared_test` and `elementwise_parity` failed).
  Fixed, it page-locks the memory with `hipHostRegister` and has the functions in its table (`hip-host-register`; CTest passes on an RX 9060 XT).
  Ninth, the HIP adapter's queue-empty query took `hipErrorNotReady`, the answer for a stream still at work, as an error and logged it every time.
  Fixed, as in the CUDA adapter, it answers "not empty" without a log (`hip-queue-empty`).
  Tenth, at a process's exit the SYCL runtime unregisters its device images from the executable's finalizers after the HIP runtime has torn itself down, and glibc aborted with "double free or corruption" when the HIP adapter unloaded the module (every SYCL program that had used an AMD GPU ended with SIGABRT).
  Fixed, a program released after the exit began is not unloaded (`hip-exit-unload`).
  Eleventh, libclc took an amdgcn GPU for one without FMA instructions unless the compiler defined `__HAS_FMAF__`.
  The AMD libspirv is built for no processor and lacks it, so `sycl::fma` and the math built on it ran a software FMA on every AMD GPU (the engine's DeltaNet recurrence about 95 µs a token on an RX 9060 XT).
  Every amdgcn GPU has `v_fma_f32`: fixed, only r600 keeps the software FMA (`hip-libclc-fma`).
  Twelfth, a kernel bundle (`get_kernel_bundle`) of a program with code for several AMD architectures got the first architecture's image, which the GPU refused (`hipErrorInvalidImage`).
  AMD images carry no `compile_target`; the runtime asked the HIP adapter about each image alone without handing it the image, and the adapter falls back on the first AMD image when none matches.
  An engine for many AMD GPUs, as the packages build it, stopped with no usable GPU (kernels submitted to a queue were not affected: that path hands the adapter all the images and their bytes).
  Fixed, the runtime hands the HIP adapter the image too, as it does the CUDA one, and the adapter does not fall back on a clang offload bundle without the GPU's architecture (`hip-bundle-arch`).
  The CUDA fixes ask only the contrib toolchain to be built again, the HIP ones only a toolchain with HIP (or one that would have it, ROCm being installed now).
- The script configures the build tree again when it was configured with other values of its options (the install folder, libclc's targets and others).
- On the development machine (Ubuntu 26.04, CUDA 12.4, ROCm 7.1) the runtime's backends were cuda, hip, level_zero and opencl.
  It ran beside other builds, so its time alone was not measured.
- Running on an NVIDIA GPU needs NVIDIA's driver (on Ubuntu 26.04, `nvidia-driver-610-open` or another).

## Building the packages (Docker)

The deb and rpm packages (`tools/package/build.sh`, [BUILD.md](BUILD.md#the-deb-and-rpm-packages)) are built in Docker containers.

```sh
sudo apt install docker.io            # 29.1.3
sudo usermod -aG docker "$USER"       # after logging in again it runs without sudo
```

The docker group has the same power as root.
The containers get no GPU, so nothing extra for GPUs (such as nvidia-container-toolkit) is needed.
A build makes intel/llvm too; for Fedora 44's free packages the container grew to 6.8 GB at most (measured every minute; the cuda variants add the CUDA toolkit). The container is removed at the end.

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

## Measuring the GPU

### What works without extra tools

- The engine's own timings: `STRATA_PREFILL_TIMING=1` for the prompt path.
  The decode's `verify window` line (the host's wait for the GPU, the CPU pool) is always printed.
- `STRATA_VERIFY_NODES=1` prints how many kernels a decode window's graph holds.
- Scratch probes that time single kernels with SYCL event profiling (the records in `bench/results/2026-10-02-xe-decode-gpu`).
- `STRATA_VERIFY_PROFILE=1` needs a device-scope clock, which the B70 does not have.

### The xe driver's kernel log

The xe driver reports its errors in the kernel log.
`journalctl -k` reads it without root (`dmesg` needs root):

```sh
journalctl -k --since "-1h" --no-pager | grep "xe 0000"
```

`VM worker error: -16` there came about 8 s before each start that had the output head's VRAM refused
(`bench/results/2026-10-02-xe-decode-gpu`).

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
VTune's `xpu-offload` collection and unitrace both left the engine stalled at its prompt for minutes.
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
  The kernel in which the GPU waits for the host (`wait_flag_ge`) presumably cannot finish while `ncu` measures one kernel at a time (unverified).
  Use `ncu` on the parity tests and probes, not on the engine.
