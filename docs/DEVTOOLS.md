<!--
SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
SPDX-License-Identifier: LGPL-3.0-or-later
-->
# Developer tools

English | [日本語](DEVTOOLS.ja.md)

The tools a developer installs to lint, check and measure XeStrata.
What building and running it needs is in [XE.md](XE.md#setup).
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

The free build compiles the engine with intel/llvm's DPC++ ([XE.md](XE.md#build-and-run)).
Where the distribution has no DPC++, or one whose SYCL runtime reports no XMX for the GPU,
`tools/intel_llvm_build.py` builds a release from source into `.tools/intel-llvm/`.
An example of the latter is Ubuntu 26.04's 6.2 with the Arc Pro B70 (intel/llvm added the B70's XMX in v7.0.0).
setup runs it for `--intel-llvm-build`;
[XE.md](XE.md#the-sycl-compiler) describes how it reuses a finished build.
intel/llvm is free software (Apache-2.0 with LLVM exceptions) and builds with free software only.
It needs these packages (on Fedora 44: `git cmake ninja-build gcc-c++ hwloc-devel python3`):

```sh
sudo apt install git cmake ninja-build g++ python3 libhwloc-dev
python3 tools/intel_llvm_build.py            # --keep-build keeps the build tree for a quicker update
```

Its configuration downloads what the release pins (Level Zero's headers and loader, emhash; all free software), so it needs the network.
The script makes it fetch Level Zero even when one is installed (`SYCL_UR_FORCE_FETCH_LEVEL_ZERO`):
without `pkg-config` the runtime adapter takes an installed loader without checking its version, and Debian 13's 1.20 failed to compile.

The toolchain is `.tools/intel-llvm/install/bin/clang++`, its runtime `.tools/intel-llvm/install/lib/libsycl.so`.
Configure the engine with `-DCMAKE_CXX_COMPILER=$PWD/.tools/intel-llvm/install/bin/clang++` and run it with that `lib/` on `LD_LIBRARY_PATH`.
On the development machine (28 threads, 91 GiB) v7.1.1 built in 13 minutes.
The clone (2.8 GB) and `install/` (0.7 GB) stay; the build tree is deleted (its size was not measured).
This runtime gives the Arc Pro B70 XMX (FP16 and BF16); 6.2's does not.

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
