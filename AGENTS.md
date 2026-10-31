<!--
SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
-->
# Notes for coding agents

## LOCAL.md: this machine's facts (read it first, keep it current)

`LOCAL.md` in the repository root holds what is true of this machine and the machines it works with, and nothing else holds it; git ignores it.

- **Read it before any work that runs something** (a build, a test, a measurement, a container, another machine).
- **Facts of a machine go there and only there:** host names, users, IP addresses, containers and how to run in them, paths (the data folder, the model files, the toolchains, the venvs), the GPUs' PCI addresses, what a machine lacks or needs set up, and its quirks (naming a GPU or CPU model, as everywhere, is fine).
  Never write them into AGENTS.md, a skill, a script, a document or a commit message; these name the fact and send the reader to `LOCAL.md` (`<data>: LOCAL.md`), and scripts take such values from the environment or their arguments.
- **Keep it current in the same turn:** a fact learned or found changed (a new container, a moved folder, something a machine lacks) goes into `LOCAL.md` before the work moves on. A fact that is missing there is looked up, and asked of the user only when it cannot be found, then written down.
- It is written in Japanese, by machine, facts only, as it is now.

## Hardware independence (the first rule)

XeStrata runs on any Intel Arc GPU and any x86-64 CPU with AVX2, not on the development machine's B70 and i7-14700.
A GPU with matrix engines (XMX) takes the XMX paths; one without (an older Arc, an integrated GPU) takes the DP4a or plain paths and still runs.
The contrib-llvm build (below) runs on NVIDIA GPUs as well, through the same kernels: their tensor cores take the matrix path that reads its tile shape from the device (`mma_gemm`), not Intel's.

- Choose code paths from what the hardware reports: the CPUID feature bits (as `cpu_avx512_ok` does); the device's matrix combinations, sub-group sizes, GRF modes, local memory, memory size and largest allocation. Never from a vendor string, a model number, a device ID or a product name.
- Sizes and thresholds (tile shapes, chunk sizes, ring slots, cache sizes, thread counts, memory reserves, spin bounds) are not hard-coded, as far as they can be: they are derived from those reports at run time. A value that has to be measured is measured on the three makers' GPUs the maintainer has, the Arc Pro B70, the RTX 4070 and the RX 9060 XT, not on the B70 alone; it has a fallback that works on a smaller device, and a comment says where it was measured and what it depends on.
- A feature some devices lack (XMX, large GRF mode, a SYCL extension) is checked before use, with a working fallback.
- Check a change on a smaller configuration as well: on the B70 with `STRATA_VRAM_LIMIT_MIB=8192 STRATA_MAX_ALLOC_MIB=4096 STRATA_NO_XMX=1` (the memory, the largest allocation and the matrix engines of a small card) and fewer CPU threads; at milestones also on the processor's own graphics (the UHD 770, a check only: start and a short answer). Mark what was not checked `unverified`.

## Operating systems: Linux (main) and Windows

Linux is the main platform: development, tests, measurements and the packages are on Linux.
Windows is a target as well, built natively for the MSVC ABI with intel/llvm's clang.

- Host code that differs between the two keeps a `#if defined(_WIN32)` branch beside the Linux one, as upstream does, or goes through `src/platform/`.
- A change to the C++ code is also compiled for Windows on Linux with the cross build ([docs/DEVTOOLS.md](docs/DEVTOOLS.md#windows-cross-build)): at least the files it touches, and the whole build, linked, when dev is integrated. It must compile and link with no new warnings.
- Running on Windows is checked only at milestones; until then a report marks it `unverified`.

## Free and non-free builds (Debian main)

XeStrata builds in two modes, chosen by the CMake option `STRATA_LICENSE`, after the Debian archive's areas:

- free (`-DSTRATA_LICENSE=free`): free software only, so the program could go into Debian main. The SYCL compiler is intel/llvm's DPC++ 7 or later (its SYCL runtime libsycl 9; one built from source, or a distribution's as new). Besides XeStrata's own kernels it uses only kernels and libraries that are free software by the DFSG (what Debian main would take): oneMath with rocBLAS (ROCm) for AMD GPUs' dense matrix products, for example.
- contrib-llvm (the default, `STRATA_LICENSE=contrib`): XeStrata's free source built with intel/llvm's CUDA target, for NVIDIA GPUs as well (`STRATA_CUDA_ARCHS` and `STRATA_ONEMKL` default to the build machine's GPUs, of every maker it has). Its dense matrix products go through oneMath (Apache-2.0; XeStrata's fork, which CMake fetches): oneMKL on Intel GPUs, cuBLAS on NVIDIA ones, rocBLAS on AMD ones (the free mode's path). It needs NVIDIA's CUDA toolkit to build and NVIDIA's driver to run, and oneMKL, none of them free software: the program could go into Debian contrib.

Intel oneAPI's icpx is not taken in either mode: CMake refuses it. A third mode built with it, contrib-icpx, was removed on 2026-10-15 (the user's decision): intel/llvm reaches every maker's GPUs.

The dense matrix products, by mode and GPU maker:

| Mode | Intel Arc | AMD | NVIDIA |
| --- | --- | --- | --- |
| free | XeStrata's own kernels | AMD's runtime (rocBLAS) | — |
| contrib-llvm | Intel's runtime (oneMKL) | AMD's runtime (rocBLAS) | NVIDIA's runtime (cuBLAS) |

In reports, commit messages and documents, call the two modes free and contrib-llvm.
Name a build by its mode, not by its compiler or a former folder name ("llvm7", "contrib", "xe"): "built and tested in the free and contrib-llvm modes". The build folders carry the modes' names: `build/free` and `build/contrib-llvm`.
contrib-llvm is the mode the code calls `contrib` (`STRATA_LICENSE=contrib`, `--license contrib`, the packages `xestrata-contrib-cuda<version>`); those names stay as they are.

Rules:

- XeStrata itself stays free software, and the free mode must build, run and pass its tests. Do not write anything that only the contrib-llvm mode can build or run, other than the paths for the GPUs only that mode reaches (NVIDIA's).
- A non-free dependency goes only on a path the contrib-llvm mode switches on, never in the free mode, and is never bundled. The free mode may be slower without it (an older free compiler, for example), never broken.
- Keep one code path where the modes can share it: the engine's dense matrix products (`src/prefill/gemm.cpp`) go through oneMath in the contrib-llvm mode, and in the free mode where oneMath's backend for the GPU is free software (rocBLAS for AMD GPUs); elsewhere (and where oneMath has no backend for the GPU) through XeStrata's own kernels. The AMD path is the same in the free and contrib-llvm modes. Every other product is XeStrata's own kernels, or a DFSG-free library's, in every mode.
- Build and test a change to the SYCL code with intel/llvm only, in the free and contrib-llvm modes: a free build (`build/free`, intel/llvm built from source as [docs/DEVTOOLS.md](docs/DEVTOOLS.md#intelllvm-from-source) describes) and a contrib-llvm one (`build/contrib-llvm`, `STRATA_CUDA_ARCHS`). A change that NVIDIA's device compile sees is also run on an NVIDIA GPU where one is there (else `unverified`). The compilers differ in version: an extension one of them lacks needs a fallback (as `sycl_ext_oneapi_clock` in `verify_kernels.cpp`), and a warning one of them gives counts as new.
- `third_party/nonfree/` (below) stays optional in every mode.
- The model itself (Qwen3.8-Flash-Next) is excluded from these rules.
- Non-free tools a developer runs by hand, such as Intel SDE below, are allowed as long as they are not bundled and neither the build nor the tests require them.

## Two engines, one set of capabilities

Qwen3.8-Flash-Next (and the Gyro packs) run on `src/program/generate.cpp`; the other models (Ornith-1.5, llm-jp-4.1, Qwen3.6-35B-A3B, Qwen3.5-9B, GLM-5.3-Flash) on the plain engine, `src/program/plain_generate.cpp` and `src/core/plain_model.cpp`.
The two do not differ in what they can do (the user's decision of 2026-10-31): a feature, a fix, a speed path or an engine option goes into both.

- Write it once, as a shared part both engines call (`src/core/`, `src/kernels/`, `src/prefill/`, `serve/`), not as a second copy in each engine.
- upstream's changes land in `generate.cpp`: whatever of them is taken goes into the plain engine too, in the same work ([the upstream-sync skill](.claude/skills/upstream-sync/SKILL.md)).
- Only what rests on Qwen3.8-Flash-Next's own architecture stays with it: what `kQwen38Only` in `src/program/option_scope.cpp` lists (PLE, QSA, the separate MTP pack, their kernels and switches).
  An engine option `generate.cpp` gains goes into the plain engine or into `kQwen38Only`; `kNotYet` lists only what the plain engine still lacks, and it only shrinks.
- A speed path taken into the plain engine is measured there as well, on a plain model (Ornith-1.5 or Qwen3.6-35B-A3B).

## What goes in

Every change goes in only on these two conditions, whoever writes it: the maintainer, a coding agent, a pull request, a port from upstream or another project.

- It was built and tested with intel/llvm, in the free and contrib-llvm modes (above). intel/llvm is the compiler that reaches every GPU maker's devices: a change built only with another compiler (Intel oneAPI's icpx, for example), or not built, does not go in.
- It depends on the hardware it was written on as little as it can (the first rule, above): not only no CUDA or other maker's API outside the paths only the contrib modes reach, but no value measured on one card (the B70's spin bounds or tile shapes, for example) hard-coded where it can be derived from what the device reports.

A change that does not meet them, a pull request among others, does not go in as it is: it is a reference, and what it does is written again in XeStrata's code.

## Licenses

XeStrata's own code is under the LGPL, version 3 or later (`COPYING.LESSER`, `COPYING`), and XeStrata as a combined work is distributed under it; the code from other projects keeps its own license (below). Copyright and license notices are in `NOTICE`.
`NOTICE` also grants an additional permission (GPL version 3, section 7): XeStrata linked with a GPU or CPU maker's math or compute library and its runtime libraries (oneMKL, cuBLAS, the CUDA runtime), or with the C and C++ runtime libraries of the operating system or its standard compiler (Microsoft's Universal C Runtime and Visual C++ runtime), may be conveyed.

Each file carries the license of the code in it, by where that code comes from:

- XeStrata's own: a file written for XeStrata is under the LGPL, with XeStrata's copyright only.
- Mixed: a file from a project under a permissive license (Strata, ggml / llama.cpp: MIT) that XeStrata has changed, or a file of XeStrata's with code transcribed from such a project (the kernels in `src/` that follow ggml, for example).
  The original code keeps its copyright and license; XeStrata's changes and the file as a whole are under the LGPL. The file names both copyright holders and both licenses, and keeps the original notice comment.
- Unchanged: a file from such a project that XeStrata has not changed (Strata's files as upstream has them, upstream's data files) keeps its own license and copyright only, without XeStrata's line.
- A file is changed when its content, apart from the SPDX header XeStrata adds, differs from the upstream version XeStrata has integrated (`STRATA_UPSTREAM_VERSION`): from upstream's file at the same path, or from a file of the same content elsewhere (a rename); a file of Strata's that upstream has removed, from the content it last had there. One that XeStrata renamed and changed is mixed; one that has none of upstream's lines (apart from blank and short ones) is XeStrata's own, written anew. An upstream sync that brings a file back to upstream's content makes it unchanged again; a change of XeStrata's to an unchanged file makes it mixed.
- `tools/license_class.py` checks each file's header (or its `REUSE.toml` annotation) against its class, and `--fix` rewrites the headers that differ; `tools/lint/run.sh` runs the check.
- Exceptions: XeStrata's changes to oneMath (below) and its patches to intel/llvm (`third_party/main/intel-llvm/patches/`, Apache-2.0 WITH LLVM-exception) stay under those projects' licenses, even when MistVVK writes them.

Material from other projects goes into `third_party/`, one folder per project with its license text, and is listed in that folder's README:

- `third_party/main/<project>/`: free software (what Debian main would take), kept under its own license. XeStrata may build and run with it.
- `third_party/nonfree/<project>/`: anything that is not free software.
- XeStrata's changes to oneMath are commits in its fork ([MistVVK/oneMath](https://github.com/MistVVK/oneMath), branch `xestrata`: oneMath's commit 6ff3a43 with XeStrata's changes on it, and oneMath's `develop` merged into it, up to 1308c5e), under oneMath's license (Apache-2.0), not the LGPL, even when MistVVK writes them; CMake fetches a tag of it. In each file a change touches, it keeps the Apache-2.0 header and adds a line saying who changed what (`Modified 2026 by MistVVK and the XeStrata contributors: ...`); a new file carries XeStrata's copyright and the Apache-2.0 header. The changes are listed in `third_party/main/README.md`.

XeStrata must build, run and pass its tests without `third_party/nonfree/`: whatever is there stays optional and can be removed by deleting the folder.

Every file starts with its copyright holders and license in SPDX form (the [REUSE](https://reuse.software) specification), in the file's comment syntax.
XeStrata's own file:

<!-- REUSE-IgnoreStart: examples, not this file's own header -->

```cpp
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
```

A mixed file (one from Strata that XeStrata has changed; a file with code transcribed from ggml / llama.cpp adds `SPDX-FileCopyrightText: 2023-2026 The ggml authors` instead of or besides Strata's line):

```cpp
// SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
```

An unchanged file from Strata:

```cpp
// SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
// SPDX-License-Identifier: MIT
```

<!-- REUSE-IgnoreEnd -->

- Files that cannot carry a comment (data, images, JSON) are covered in `REUSE.toml`; the license texts are in `LICENSES/`.
- `reuse lint` must pass.

## The README and the documents

The README and the documents in `docs/` are written in Japanese first: `README.ja.md` and `docs/<NAME>.ja.md` are the originals, and `README.md` and `docs/<NAME>.md` their English translations.
A change goes into the Japanese file first and then into the English one in the same commit, so the two keep the same sections and content.
The English files keep the headings that code, records and links point to (`docs/BUILD.md#packages`, for example); a Japanese file's anchors come from its own headings.
Where the README or a document states something that was not checked, do not add that it is unverified (`unverified`, 未検証); that mark is for reports and commit messages.
The two stages of generation are prefill and decode, the LLM terms, in Japanese as in English (the README, the documents, the release notes): not プロンプト・出力・デコード or prompt and output speed. 推測デコード (speculative decoding) and a prompt as the text that is read stay as they are.

## Model names: written out, never abbreviated

A model's name is written in full everywhere, in code and in text: `HunyuanImage 3.0` in prose (the README, the documents, commit messages, release notes, comments, the engine's messages) and `hunyuanimage3` where an identifier needs it - `hunyuanimage3_dit.cpp`, `HunyuanImage3Dit`, `hunyuanimage3_layer_run`, `HUNYUANIMAGE3_VAE_DUMP`. Its abbreviations (`hy3`, `Hy3`, `HY3`) are not used: a reader of a log line or a commit message should not have to know a short form, and the short form of a model's name is how two models' files start to look alike. The same holds of every model the image engine takes.

What already exists keeps its name: Qwen-Image-2.1's `qi_*` and Mage-Flow's `mf_*` parts (`qi_dit.cpp`, `QiPipeline`) stay as they are - renaming them would churn the whole image engine for no reading gain, and this rule is about names given from now on. So does everything outside the repository: the data folder's `hy3/`, the records' scripts (`hy3_conform.py`, `hy3_vae_ref.py`) and any path quoted from them; a comment that cites one of those says it is a record's or the data folder's path, so it does not read as a name for the model.

## Measurement records

XeStrata's own measurements (benchmarks, A/B runs, probes, the upstream-sync and release records) are not shared: they go
into the data folder's `records/` (`<data>: LOCAL.md`) and never into git. `bench/XeStrataBenchResults` keeps only the
measurements of XeStrata that others sent (pull requests). upstream's records (its `bench/results`, its community's
included) are not brought in. Commit messages and documents give the numbers themselves, not a link to a record.

Do not run a measurement whose result is already known. A run that gives the same result each time (greedy, the same
binary, settings and text, on the same machine in the same state) and has been measured is read from its record, not
run again. Measure again only what can differ: a changed binary or setting, a run at temperature, or a cell suspected
of noise (once).

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
| every file | `tools/license_class.py` (the license classes, see Licenses above) |
| the whole repository | `reuse lint` (see Licenses above) |

The files a change touches must get no new findings; findings that were there before may stay.
Other projects' files in `third_party/<area>/<project>/` (kept as they are; XeStrata's `patches/` there excepted) get only gitleaks and `reuse lint`.
clang-tidy reads `compile_commands.json` from the build folder (`build/free`, or `STRATA_LINT_BUILD`): configure it with `-DCMAKE_EXPORT_COMPILE_COMMANDS=ON`.

Then run the existing tests the change can affect (CTest in the build folder, `python -m unittest` for `serve/` and `tools/`).
For memory and undefined-behavior bugs, build in a folder of its own with `-DSTRATA_SANITIZE=address,undefined` (for example `build/free-asan`) and run CTest there.
It instruments the project's own host code only, not ggml or the SYCL device code; do not run its binaries under Intel SDE.

Do not reformat code with clang-format or a similar tool: the code is formatted by hand, a reformat would rewrite about a quarter of the C++ lines and make Strata's changes hard to carry over. Match the style of the surrounding code.

## Developer tools

Install the lint tools, Intel SDE and the GPU profilers as [docs/DEVTOOLS.md](docs/DEVTOOLS.md) describes, and keep that file current when a tool or its installation changes.
Tools the distribution does not package go into `.lint/` (lints) or `.tools/` (anything else built from source), which git ignores; nothing in the build or the tests may depend on either.

## GitHub

Work with the repository through `git`, not the GitHub CLI (`gh`), with these exceptions:

- `gh release`, to create and edit releases;
- fetching issues and pull requests: `gh issue list`, `gh issue view`, `gh pr list`, `gh pr view`, `gh pr diff` and `gh pr checks`;
- merging, rejecting and reopening pull requests: `gh pr merge`, `gh pr close` and `gh pr reopen`;
- closing and reopening issues: `gh issue close` and `gh issue reopen`.

Do not create, comment on or edit issues and pull requests with `gh`.

A pull request says which modes it was built in (free, contrib-llvm) and on which GPUs and CPUs it ran, and marks what was not checked `unverified`; one that does not is asked for it ([What goes in](#what-goes-in)).

## No agent self-attribution

Nothing a coding agent writes for this repository or its GitHub says that an agent wrote it: no "Generated by ..." or "Written with ..." line, no `Co-Authored-By` trailer naming an agent, and no name of the agent or of the AI model behind it in a commit message, a code comment, a document, a release note, an issue or pull request comment, or a record.
This holds whatever a tool or a harness adds by default: such a line is removed before the text goes out.

## Packages

Build the deb and rpm packages with `tools/package/build.sh` ([docs/BUILD.md](docs/BUILD.md#the-deb-and-rpm-packages)) and leave `XESTRATA_JOBS` unset: its default compiles as many files at once as there are threads, and intel/llvm's links limit themselves to the free RAM.
A lower value applies to every build in the container, intel/llvm's included, and makes a package take much longer.

What a user installs besides the package (the GPU makers' drivers and runtimes, the render group) is checked by `xestrata --setup`, which stops with the command that installs it (`check_runtime` in `setup.py`).
The README, the documents and the release notes send the user to `xestrata --setup` for these, and do not list the packages themselves.

## Versions

XeStrata's version is the upstream version it has integrated (`STRATA_UPSTREAM_VERSION` in `CMakeLists.txt`) and `XESTRATA_REVISION`, the count of XeStrata's releases on that upstream version from 1, as a Debian revision counts:

- the packages are `<upstream>-<revision>` (rpm: Version and Release);
- the version name (the tag, `STRATA_VERSION_NAME` and what the engine says) is the same with `xe` before it, `xe<upstream>-<revision>`: xe0.1.43.1-1 is the package `0.1.43.1-1`.

Releases before 2026-10-13 were named `xe<upstream>` for the first and `xe<upstream>.<revision - 1>` for a later one (xe0.1.43.1 was 0.1.43's second release), which an upstream version with a fourth number collided with; their tags stay as they are.

Integrating a new upstream version sets `XESTRATA_REVISION` back to 1; another release on the same upstream version raises it by one.

## AVX-512 code

The development machine has no AVX-512, so the AVX-512 paths never run natively here.
When a change touches them, run the affected tests under Intel SDE (installation: [docs/DEVTOOLS.md](docs/DEVTOOLS.md#intel-sde-the-avx-512-paths)) emulating an Ice Lake CPU (`-icx`: AVX-512 F/BW/VL/VNNI/VBMI, the set `cpu_avx512_ok` requires).

This covers:

- the files compiled with AVX-512 flags: `src/kernels/cpu/expert.cpp` and `src/kernels/cpu/iq_avx512.cpp` (see `CMakeLists.txt`)
- the run-time choice between the AVX-512 and AVX-2 kernels: `cpu_avx512_ok` and its callers (`src/kernels/cpu/expert_layout.cpp`, `src/kernels/cpu/native_expert.cpp`, `src/program/generate.cpp`)

```sh
sde64 -icx -- ./build/free/expert_multi_test
sde64 -icx -- ./build/free/expert_parity --selftest   # from the repository root
sde64 -icx -- ./build/free/pool_test --selftest       # from the repository root
```

The same tests must also still pass or skip natively, so the AVX-2 path is checked as well.
SDE measures correctness only: its timings say nothing about the speed on real hardware.
Whether SDE runs the tests that also use the GPU (`native_expert_parity`) is `unverified`.

## Intel Graphics Compiler (IGC): the GEP simplification of 2.27 to 2.30

IGC 2.27.10, 2.28.4 and 2.30.1 miscompile, for Xe2 and Xe3 GPUs (the targets `bmg`, `lnl` and `ptl`: the Arc B-series, Lunar Lake, Panther Lake), a kernel that reads from one pointer at a runtime index and at constant multiples of it.
With `const float* xt = x + base; xt[0] + xt[N] + xt[2 * N] + xt[3 * N]`, intel/llvm's optimizer writes `xt[2 * N]` and `xt[3 * N]` as byte offsets (8N and 12N bytes) from the pointer it already offset for `xt[N]`, and IGC's GEP simplification takes those byte offsets for element indices: the kernel reads `xt[8 * N]` and `xt[9 * N]`, a fault past the buffer (`DEVICE_LOST`, `Engine memory CAT error` in the kernel's log) or wrong values inside it.
2.24.8 and earlier and 2.32.7 and later compile the reads right (each release's libigc from GitHub on the reproducer's SPIR-V, 2026-10-19); Alchemist (`dg2`) is not affected (IGC reads its buffers there another way).
Ubuntu 26.04, Debian and Arch ship 2.28.4; Fedora 44 ships 2.24.8 and updates to 2.41.9.
The reproducer and the checks are in the data folder's `records/igc-gep-check/` (`<data>: LOCAL.md`).

- The runtime sets `IGC_EnableGEPSimplification=0` in its environment before the SYCL runtime lists the devices (`engine_devices` in `src/core/device.cpp`), unless the user set the variable; IGC reads it when it first compiles a kernel, which the runtime does as it lists the devices. Off, the pass costs the B70 5 to 10% of the decode of a 1K prompt and nothing at 32K (generate, 2026-10-19), so the variable is set only where both are true: the `libigc.so.2` the loader finds is 2.25 to 2.31 by its file name (`libigc.so.2.28.4+0`), or carries no version; and a GPU on the xe driver is Xe2 or later by its graphics IP (`xe_gpu_from_xe2`: the main GT's `ip_ver_major` of `DRM_XE_DEVICE_QUERY_GT_LIST` 20 or more; 0, a GPU without GMD_ID, is Alchemist or older), asked of the kernel driver before Level Zero loads, since a variable set after the devices are listed is not read. Without such a GPU nothing is set. Linux only: Windows sets nothing and keeps its driver's compiler as it is. The start log says so (`IGC's GEP simplification is off`), the engine reports it as `INFO igc_workaround=<version>` (the web app's notice and its About page), and `xestrata --setup` (`check_igc` in `setup.py`, on the same two conditions) warns of it. The start log and the engine's report send the user to Fedora 44, whose IGC is 2.32 or later, or to the Podman container (`container/`, Fedora 44 inside: IGC is the container's, the host gives only the kernel driver); setup (`check_igc`), the documents (BUILD's packages, DETAILS's troubleshooting) and the release notes name one more way, Intel's graphics PPA (launchpad.net/~kobuk-team/+archive/ubuntu/intel-graphics, the compute-runtime project's repackaging: compute-runtime 26.31 and IGC 2.40 for Ubuntu 24.04/26.04) - with the HWE kernel straight to the PPA, from the GA kernel to the HWE kernel first (the PPA installs the driver with the IGC, and that driver is the one for the HWE kernel).  The same `xestrata --setup` (`check_level_zero`) stops the start when the Level Zero driver predates the build that stops it refusing device memory while the VRAM is free (26.22.38646.4; intel/compute-runtime #916, xe's kernel bug fixed in Linux 7.1; on XeStrata's side #34 and #15) on Ubuntu 24.04/26.04 with the HWE kernel - `--old-driver` is the explicit consent to start anyway - and, on the GA kernel and elsewhere, only says it and goes on: the GA kernel is sent to the HWE kernel and the PPA, or to Fedora 44 / Podman, the rest to a driver of 26.22 or later, or to Fedora 44 / Podman.
- Write kernels that compile right either way: step a pointer by N from the last read (`glm5_hc_mean`), or index the stream in one product (`x[(t * HC + s) * N + n]`), not `xt[k * N]` with a constant k of 2 or more beside `xt[N]`.
- To see what IGC makes of a kernel without running it, compile ahead for Xe2 with IGC's dumps on, `IGC_ShaderDumpEnable=1 IGC_DumpToCustomDir=dump clang++ -fsycl -fsycl-targets=intel_gpu_bmg_g31 ...`, and read `dump/*_entry_*.asm`: the miscompile shows as a scalar `mul ... 0x1C:uw` (28N: the 32N bytes of the misread `xt[8 * N]`, less the 4N of `xt[N]`) ahead of the loads; `*_beforeUnification.ll` is the IR IGC received. `IGC_EnableGEPSimplification=1 ./igc_gep_probe` runs the pattern on the GPU with room for the wrong reads and fails on an affected IGC; CTest runs it through the runtime's workaround, where it passes on every GPU.
