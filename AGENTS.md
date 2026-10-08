# Notes for coding agents

## Hardware independence (the first rule)

XeStrata runs on any Intel Arc GPU and any x86-64 CPU with AVX2, not on the development machine's B70 and i7-14700.
A GPU with matrix engines (XMX) takes the XMX paths; one without (an older Arc, an integrated GPU) takes the DP4a or plain paths and still runs.
The contrib build (below) runs on NVIDIA GPUs as well, through the same kernels: their tensor cores take the matrix path that reads its tile shape from the device (`mma_gemm`), not Intel's.

- Choose code paths from what the hardware reports: the CPUID feature bits (as `cpu_avx512_ok` does); the device's matrix combinations, sub-group sizes, GRF modes, local memory, memory size and largest allocation. Never from a vendor string, a model number, a device ID or a product name.
- Sizes and thresholds measured on one machine (tile shapes, chunk sizes, ring slots, cache sizes, thread counts, memory reserves) are derived from those reports at run time, or have a fallback that works on a smaller device. A constant tuned on the B70 says so in a comment, with what it depends on.
- A feature some devices lack (XMX, large GRF mode, a SYCL extension) is checked before use, with a working fallback.
- Check a change on a smaller configuration as well: on the B70 with `STRATA_VRAM_LIMIT_MIB=8192 STRATA_MAX_ALLOC_MIB=4096 STRATA_NO_XMX=1` (the memory, the largest allocation and the matrix engines of a small card) and fewer CPU threads; at milestones also on the processor's own graphics (the UHD 770, a check only: start and a short answer). Mark what was not checked `unverified`.

## Free and non-free builds (Debian main)

XeStrata builds in three modes, chosen by the CMake option `STRATA_LICENSE`, after the Debian archive's areas:

- free (`-DSTRATA_LICENSE=free`): free software only, so the program could go into Debian main. The SYCL compiler is intel/llvm's DPC++ 7 or later (its SYCL runtime libsycl 9; one built from source, or a distribution's as new); icpx is refused. Besides XeStrata's own kernels it uses only kernels and libraries that are free software by the DFSG (what Debian main would take): oneMath with rocBLAS (ROCm) for AMD GPUs' dense matrix products, for example.
- contrib (the default, `STRATA_LICENSE=contrib`): XeStrata's free source built with intel/llvm's CUDA target, for NVIDIA GPUs as well (`STRATA_CUDA_ARCHS` and `STRATA_ONEMKL` default to the build machine's GPUs, of every maker it has). Its dense matrix products go through oneMath (Apache-2.0; XeStrata's fork, which CMake fetches): oneMKL on Intel GPUs, cuBLAS on NVIDIA ones, rocBLAS on AMD ones (the free mode's path). It needs NVIDIA's CUDA toolkit to build and NVIDIA's driver to run, and oneMKL, none of them free software: the program could go into Debian contrib.
- contrib-icpx (`-DSTRATA_LICENSE=contrib-icpx`): the same free source built with Intel oneAPI's icpx and the runtime libraries it links, with oneMKL for the dense matrix products (through oneMath) and the SYCL image encoder (ggml-sycl), for Intel GPUs. setup builds in this mode only when asked (`--license contrib-icpx`).

In reports, commit messages and documents, call the three modes free, contrib-llvm and contrib-icpx: "contrib" alone could be either of the last two.
contrib-llvm is the mode the code calls `contrib` (`STRATA_LICENSE=contrib`, `--license contrib`, the packages `xestrata-contrib-cuda<version>`); those names stay as they are.

Rules:

- XeStrata itself stays free software, and the free mode must build, run and pass its tests. Do not write anything that only the contrib or contrib-icpx mode can build or run, other than the paths for the GPUs only those modes reach (NVIDIA's, in the contrib mode).
- A non-free dependency goes only on a path the contrib or contrib-icpx mode switches on, never in the free mode, and is never bundled. The free mode may be slower without it (an older free compiler, for example), never broken.
- Keep one code path where the modes can share it: the engine's dense matrix products (`src/prefill/gemm.cpp`) go through oneMath in the contrib and contrib-icpx modes, and in the free mode where oneMath's backend for the GPU is free software (rocBLAS for AMD GPUs); elsewhere (and where oneMath has no backend for the GPU) through XeStrata's own kernels. The AMD path is the same in the free and contrib modes. Every other product is XeStrata's own kernels, or a DFSG-free library's, in every mode.
- Build and test a change to the SYCL code in the free and contrib-icpx modes: a free build (`build/llvm7`, intel/llvm built from source as [docs/DEVTOOLS.md](docs/DEVTOOLS.md#intelllvm-from-source) describes) and a contrib-icpx one (`build/xe`, icpx). A change that NVIDIA's device compile sees is also built in the contrib mode (`build/contrib`, `STRATA_CUDA_ARCHS`), and run on an NVIDIA GPU where one is there (else `unverified`). The compilers differ in version: an extension one of them lacks needs a fallback (as `sycl_ext_oneapi_clock` in `verify_kernels.cpp`), and a warning one of them gives counts as new.
- `third_party/nonfree/` (below) stays optional in every mode.
- The model itself (Qwen3.8-Flash-Next) is excluded from these rules.
- Non-free tools a developer runs by hand, such as Intel SDE below, are allowed as long as they are not bundled and neither the build nor the tests require them.

## Licenses

XeStrata is under the LGPL, version 3 or later (`COPYING.LESSER`, `COPYING`); copyright and MIT notices are in `NOTICE`.
Keep the MIT notice of the parts that come from Strata.
`NOTICE` also grants an additional permission (GPL version 3, section 7): XeStrata linked with a GPU or CPU maker's math or compute library and its runtime libraries (oneMKL, cuBLAS, the CUDA runtime) may be conveyed.
Material from other projects goes into `third_party/`, one folder per project with its license text, and is listed in that folder's README:

- `third_party/main/<project>/`: free software (what Debian main would take), kept under its own license. XeStrata may build and run with it.
- Code transcribed from such a project into XeStrata's own sources (the kernels in `src/` that follow ggml, for example) is part of XeStrata under the LGPL, with the original notice kept in a comment.
- `third_party/nonfree/<project>/`: anything that is not free software.
- XeStrata's changes to oneMath are commits in its fork ([MistVVK/oneMath](https://github.com/MistVVK/oneMath), branch `xestrata`: oneMath's commit 6ff3a43 with XeStrata's changes on it), under oneMath's license (Apache-2.0), not the LGPL, even when MistVVK writes them; CMake fetches a tag of it. In each file a change touches, it keeps the Apache-2.0 header and adds a line saying who changed what (`Modified 2026 by MistVVK and the XeStrata contributors: ...`); a new file carries XeStrata's copyright and the Apache-2.0 header. The changes are listed in `third_party/main/README.md`.

XeStrata must build, run and pass its tests without `third_party/nonfree/`: whatever is there stays optional and can be removed by deleting the folder.

Every file under the LGPL starts with its copyright holders and license in SPDX form (the [REUSE](https://reuse.software) specification), in the file's comment syntax:

```cpp
// SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
```

- A file that came from Strata has both lines; a file new in XeStrata has only the XeStrata line.
- A file with code transcribed from ggml / llama.cpp adds `SPDX-FileCopyrightText: 2023-2026 The ggml authors` and keeps its MIT notice comment.
- Files that cannot carry a comment (data, images, JSON) are covered in `REUSE.toml`; the license texts are in `LICENSES/`.
- `reuse lint` must pass.

## The README and the documents

The README and the documents in `docs/` are written in Japanese first: `README.ja.md` and `docs/<NAME>.ja.md` are the originals, and `README.md` and `docs/<NAME>.md` their English translations.
A change goes into the Japanese file first and then into the English one in the same commit, so the two keep the same sections and content.
The English files keep the headings that code, records and links point to (`docs/BUILD.md#packages`, for example); a Japanese file's anchors come from its own headings.

## Lints

Before committing, run `tools/lint/run.sh` from the repository root: without arguments it lints the files that differ from HEAD (changed or new), or the files given.
It picks the lints by file kind, with the configurations in `tools/lint/`:

| Files | Lints |
| --- | --- |
| C++ (and SYCL) | the compiler's warnings (`-Wall -Wextra` through `strata_warnings` in `CMakeLists.txt`; `-DSTRATA_WERROR=ON` makes them errors), clang-tidy (oneAPI's, which knows SYCL), cppcheck |
| Python | flake8, ruff, mypy |
| Shell | shellcheck |
| CMake | cmake-lint |
| Markdown | mdl, lychee (`--offline`: links to files in the repository) |
| CSS / JavaScript / HTML | stylelint, eslint, tidy |
| all text files | codespell |
| every file | gitleaks (built-in rules, no configuration of its own) |
| the whole repository | `reuse lint` (see Licenses above) |

The files a change touches must get no new findings; findings that were there before may stay.
The records that came from upstream's `bench/results` (other people's measurements, listed in `tools/lint/records.txt`) and other projects' files in `third_party/<area>/<project>/` (kept as they are; XeStrata's `patches/` there excepted) get only gitleaks and `reuse lint`.
clang-tidy reads `compile_commands.json` from the build folder (`build/xe`, or `STRATA_LINT_BUILD`): configure it with `-DCMAKE_EXPORT_COMPILE_COMMANDS=ON`.

Then run the existing tests the change can affect (CTest in the build folder, `python -m unittest` for `serve/` and `tools/`).
For memory and undefined-behavior bugs, build in a folder of its own with `-DSTRATA_SANITIZE=address,undefined` (for example `build/xe-asan`) and run CTest there.
It instruments the project's own host code only, not ggml or the SYCL device code; do not run its binaries under Intel SDE.

Do not reformat code with clang-format or a similar tool: the code is formatted by hand, a reformat would rewrite about a quarter of the C++ lines and make Strata's changes hard to carry over. Match the style of the surrounding code.

## Developer tools

Install the lint tools, Intel SDE and the GPU profilers as [docs/DEVTOOLS.md](docs/DEVTOOLS.md) describes, and keep that file current when a tool or its installation changes.
Tools the distribution does not package go into `.lint/` (lints) or `.tools/` (anything else built from source), which git ignores; nothing in the build or the tests may depend on either.

## Packages

Build the deb and rpm packages with `tools/package/build.sh` ([docs/BUILD.md](docs/BUILD.md#the-deb-and-rpm-packages)) and leave `XESTRATA_JOBS` unset: its default compiles as many files at once as there are threads, and intel/llvm's links limit themselves to the free RAM.
A lower value applies to every build in the container, intel/llvm's included, and makes a package take much longer.

## AVX-512 code

The development machine has no AVX-512, so the AVX-512 paths never run natively here.
When a change touches them, run the affected tests under Intel SDE (installation: [docs/DEVTOOLS.md](docs/DEVTOOLS.md#intel-sde-the-avx-512-paths)) emulating an Ice Lake CPU (`-icx`: AVX-512 F/BW/VL/VNNI/VBMI, the set `cpu_avx512_ok` requires).

This covers:

- the files compiled with AVX-512 flags: `src/kernels/cpu/expert.cpp` and `src/kernels/cpu/iq_avx512.cpp` (see `CMakeLists.txt`)
- the run-time choice between the AVX-512 and AVX-2 kernels: `cpu_avx512_ok` and its callers (`src/kernels/cpu/expert_layout.cpp`, `src/kernels/cpu/native_expert.cpp`, `src/program/generate.cpp`)

```sh
sde64 -icx -- ./build/xe/expert_multi_test
sde64 -icx -- ./build/xe/expert_parity --selftest   # from the repository root
sde64 -icx -- ./build/xe/pool_test --selftest       # from the repository root
```

The same tests must also still pass or skip natively, so the AVX-2 path is checked as well.
SDE measures correctness only: its timings say nothing about the speed on real hardware.
Whether SDE runs the tests that also use the GPU (`native_expert_parity`) is `unverified`.
