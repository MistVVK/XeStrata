<!--
SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
-->
# Notes for coding agents

The rules every change follows. The facts and procedures behind them are in the documents each section names.

## LOCAL.md: this machine's facts (read it first, keep it current)

`LOCAL.md` in the repository root (ignored by git) holds what is true of this machine and the machines it works with, and nothing else holds it.

- Read it before any work that runs something (a build, a test, a measurement, a container, another machine).
- Facts of a machine go there only: host names, users, IP addresses, containers, paths (the data folder, models, toolchains, venvs), PCI addresses, what a machine lacks, its quirks (naming a GPU or CPU model is fine everywhere). AGENTS.md, skills, scripts, documents and commit messages name the fact and send the reader there (`<data>: LOCAL.md`); scripts take such values from the environment or their arguments.
- Keep it current in the same turn: a fact learned or found changed goes in before the work moves on; a missing fact is looked up, asked of the user only when it cannot be found, then written down.
- Japanese, facts only, as they are now.

## One computing engine for every model (the most important rule)

Every model computes through one set of shared parts: the image engine's DiTs, VAEs, text encoders and upscalers, and the LLMs (Qwen3.8's engine and `PlainModel`) alike. The shared parts are the attention (`flash_attn`), the products (`prefill::Gemm`, `iq_mmq`, `i8_gemm`), the row operations (`dit_rows.hpp`, `rms_norm_rows`), the QK norm and RoPE (`qk_norm_rope.hpp`), the KV appends, the activations' quantization, the tiles, the VRAM budget and the warm-up, so that a speed-up reaches every model at once.

- A speed-up goes into the shared part, on by default on every device that has what it needs: not an opt-in flag, not one model's private copy.
- Each device takes its fastest path automatically, from its capabilities (the first rule below), never from the model's name or family.
- A new model is built on the shared parts; a role they lack is added to them. A separate implementation stays only where the shared one was measured slower for it, with the numbers in a comment.
- An opt-in (an environment variable, off by default) is only for a path that changes the answers beyond the models' checks, or one measured not faster on the three makers' GPUs; its comment says which.
- Check a change to a shared part on every model that uses it: images by byte match or by eye, LLMs by KL, speed on the three makers' GPUs.

### Two engines, one set of capabilities

Qwen3.8-Flash-Next (and the Gyro packs) run on `src/program/generate.cpp`, the other LLMs on the plain engine (`src/program/plain_generate.cpp`, `src/core/plain_model.cpp`); the two can do the same things (2026-10-31).

- A feature, fix, speed path or engine option goes into both, written once as a shared part (`src/core/`, `src/kernels/`, `src/prefill/`, `serve/`).
- What upstream changes in `generate.cpp` and XeStrata takes goes into the plain engine in the same work.
- Only what rests on Qwen3.8-Flash-Next's own architecture stays with it: `kQwen38Only` in `src/program/option_scope.cpp` (PLE, QSA, the separate MTP pack). An engine option `generate.cpp` gains goes into the plain engine or into that list.
- A speed path taken into the plain engine is measured there too, on a plain model (Ornith-1.5 or Qwen3.6-35B-A3B).

## Hardware independence (the first rule)

XeStrata runs on any Intel Arc GPU (with XMX or without: the DP4a and plain paths) and any x86-64 CPU with AVX2, not only on the development machine's B70 and i7-14700. The contrib-llvm build runs NVIDIA GPUs through the same kernels (`mma_gemm` reads its tile shape from the device).

- Choose code paths from what the hardware reports: the CPUID bits (as `cpu_avx512_ok`), the device's matrix combinations, sub-group sizes, GRF modes, local memory, memory size and largest allocation. Never from a vendor string, model number, device ID or product name.
- Derive sizes and thresholds (tiles, chunks, ring slots, caches, threads, reserves, spin bounds) from those reports at run time. A value that must be measured is measured on the three makers' GPUs (Arc Pro B70, RTX 4070, RX 9060 XT), has a fallback for a smaller device, and its comment says where it was measured and what it depends on.
- Check a feature some devices lack (XMX, large GRF mode, a SYCL extension) before use, with a working fallback.
- Check a change on a small configuration too: the B70 with `XESTRATA_VRAM_LIMIT_MIB=8192 XESTRATA_MAX_ALLOC_MIB=4096 XESTRATA_NO_XMX=1` and fewer CPU threads; at milestones the processor's own graphics (start and a short answer). Mark what was not checked `unverified`.

## Operating systems: Linux (main) and Windows

Development, tests, measurements and packages are on Linux; Windows is a target too, built for the MSVC ABI with intel/llvm's clang.

- Host code that differs keeps a `#if defined(_WIN32)` branch beside the Linux one, or goes through `src/platform/`.
- A C++ change is compiled for Windows with the cross build ([docs/DEVTOOLS.md](docs/DEVTOOLS.md#windows-cross-build)): the files it touches, and the whole build linked when dev is integrated, with no new warnings.
- Running on Windows is checked at milestones; until then a report marks it `unverified`.

### setup and its entries: the same features

`setup.py` does the work (with `chat.py` and `tools/image/install.py`); `setup.sh`, the packages' `xestrata`, Windows' `xestrata-cli.exe` and the launcher's window (`launcher/`: `xestrata.exe` on Windows, `xestrata-gui` on Linux) are its entries and do none of it themselves.

- What one entry can do, a user reaches through each of the others, on Linux as on Windows (2026-11-09). A feature added to one entry alone, or to setup without its way through each entry, does not go in.
- The command-line entries take setup's command line as it is, with the same subcommands (`chat`, `images`, `gui`).
- The window is setup's front end for everything a 7z user can run (not building from source or `--original-strata`): setup's questions are fields of the install page, the options it takes without asking are fields under Advanced, its actions are buttons, and a question asked during one (`--json`'s ask objects) is asked on the page.
- Each option a user sets is one entry of `OPTIONS` in `setup.py`; setup's command line, its questions and `--describe`'s `"options"` are made from it and the window draws its fields from it (`launcher/Gui/Fields.fs`). A default lives in setup alone. Add an option there, not in the window; its translations go into the launcher's catalogs (`setup.<name>.*`, checked by `tools/launcher_i18n.py`). A change to setup's questions that `OPTIONS` does not carry changes the launcher in the same commit.
- On Linux the window is an extra: everything works without it and nothing else needs .NET. It is its own package, `xestrata-gui`, the only one depending on the distribution's .NET 10 runtime (never bundled); the main package only suggests it, setup's checks never ask for .NET, and `xestrata gui` says what to install when it is missing. Linux has no `xestrata-cli`.

## Free and non-free builds (Debian main)

The CMake option `STRATA_LICENSE` picks one of two modes, after the Debian archive's areas. Call them free and contrib-llvm in reports, commit messages and documents, and name a build by its mode (`build/free`, `build/contrib-llvm`), not by its compiler or an old folder name; the code's name `contrib` (`STRATA_LICENSE=contrib`, `--license contrib`, the packages `xestrata-contrib-cuda<version>`) stays.

- free: free software only (what Debian main would take), built with intel/llvm's DPC++ 7 or later (libsycl 9). Besides XeStrata's own kernels it uses only DFSG-free kernels and libraries (oneMath with rocBLAS for AMD GPUs, for example).
- contrib-llvm (the default): the same free source with intel/llvm's CUDA target, for NVIDIA GPUs too (`STRATA_CUDA_ARCHS` and `STRATA_ONEMKL` default to the build machine's GPUs). It needs the CUDA toolkit, NVIDIA's driver and oneMKL, none of them free: it could go into Debian contrib.
- Intel oneAPI's icpx is refused in both modes.

The dense matrix products (`src/prefill/gemm.cpp`) go through oneMath (XeStrata's fork) where its backend is allowed, else through XeStrata's own kernels; every other product is XeStrata's own or a DFSG-free library's in every mode:

| Mode | Intel Arc | AMD | NVIDIA |
| --- | --- | --- | --- |
| free | XeStrata's own kernels | rocBLAS | — |
| contrib-llvm | oneMKL | rocBLAS | cuBLAS |

- The free mode must build, run and pass its tests. Nothing may be buildable or runnable only in contrib-llvm except the NVIDIA paths.
- A non-free dependency goes only on a path contrib-llvm switches on, never bundled; the free mode may be slower without it, never broken.
- Build and test a SYCL change with intel/llvm only, in both modes (intel/llvm from source: [docs/DEVTOOLS.md](docs/DEVTOOLS.md#intelllvm-from-source)); a change NVIDIA's device compile sees is run on an NVIDIA GPU where there is one (else `unverified`). An extension one compiler lacks needs a fallback (as `sycl_ext_oneapi_clock` in `verify_kernels.cpp`), and a warning either gives counts as new.
- `third_party/nonfree/` stays optional; the model (Qwen3.8-Flash-Next) is outside these rules; non-free tools a developer runs by hand (Intel SDE) are fine as long as nothing bundles or requires them.

## What goes in

Whoever writes a change (the maintainer, an agent, a pull request, a port), it goes in only when it was built and tested with intel/llvm in the free and contrib-llvm modes, and depends on the hardware it was written on as little as it can (no maker's API outside the contrib-only paths, no value measured on one card where the device's reports give it). One that does not is a reference, written again in XeStrata's code.

## Licenses

XeStrata's own code is under the LGPL 3.0 or later (`COPYING.LESSER`, `COPYING`), and XeStrata as a combined work is distributed under it; other projects' code keeps its license. `NOTICE` holds the notices and an additional permission (GPL 3, section 7) to link with GPU and CPU makers' math libraries and the systems' C and C++ runtimes.

Each file carries the license of the code in it:

- XeStrata's own (written for XeStrata, or with none of upstream's lines left): LGPL, XeStrata's copyright only.
- Mixed (a file of a permissive project, Strata or ggml / llama.cpp, that XeStrata changed, or XeStrata's file with code transcribed from one): both copyright holders and both licenses, the original notice kept; the file as a whole is LGPL.
- Unchanged (upstream's file as upstream has it, at the integrated `STRATA_UPSTREAM_VERSION`, renames included): its own license and copyright only. A sync that brings a file back to upstream's content makes it unchanged; a change of XeStrata's makes it mixed.
- `tools/license_class.py` checks each header (or `REUSE.toml` annotation) against its class (`--fix` rewrites); `reuse lint` must pass. Files that cannot carry a comment are covered in `REUSE.toml`; the license texts are in `LICENSES/`.
- Exceptions: XeStrata's changes to oneMath (commits in [MistVVK/oneMath](https://github.com/MistVVK/oneMath), branch `xestrata`) its patches to intel/llvm (`third_party/main/intel-llvm/patches/`) and to oneDNN (`third_party/main/oneDNN/patches/`, its pull requests backported to the tag CMake fetches) stay under those projects' licenses (Apache-2.0; Apache-2.0 WITH LLVM-exception; Apache-2.0), even when MistVVK writes them. A oneMath file a change touches keeps its header and adds `Modified 2026 by MistVVK and the XeStrata contributors: ...`; the changes are listed in `third_party/main/README.md`.
- Other projects' material goes into `third_party/main/<project>/` (free software) or `third_party/nonfree/<project>/` (anything else), with its license text, listed in that folder's README. XeStrata builds, runs and passes its tests without `third_party/nonfree/`.

Every file starts with its SPDX header (the [REUSE](https://reuse.software) specification) in its comment syntax: XeStrata's own, mixed (code from ggml / llama.cpp adds `SPDX-FileCopyrightText: 2023-2026 The ggml authors` instead of or besides Strata's line), unchanged:

<!-- REUSE-IgnoreStart: examples, not this file's own header -->

```cpp
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

// SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later AND MIT

// SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
// SPDX-License-Identifier: MIT
```

<!-- REUSE-IgnoreEnd -->

## Names

### XeStrata's name, not upstream's, where a user reads it

<!-- upstream-name: off -->
What a user reads says `xestrata` (`XeStrata` in prose): messages and log heads, usage texts, environment variables, programs, XeStrata's own API fields, documents, the web app, setup's lines. `strata` alone is upstream's name and is not shown there. What a user does not read keeps upstream's name, so upstream's changes carry over: the namespace `strata`, `include/strata/`, file names, CMake targets and options (`-DSTRATA_*`), and the API fields and markers shared with upstream (`strata_mcp`, `/health`'s `service: "strata"` and the like).

- A new environment variable is `XESTRATA_<NAME>`, read through the one reader: `strata::env_get("<NAME>")` in C++ (`include/strata/platform/env.hpp`), `envvars.get("<NAME>")` in Python (`serve/envvars.py`), `launcher/Core/Platform.fs` in F#; a shell script reads `$XESTRATA_<NAME>`. Never `std::getenv` or `os.environ` directly.
- The old `STRATA_<NAME>` is read only for a variable that existed before the rename (the readers do it; a script writes `${XESTRATA_<NAME>:-${STRATA_<NAME>:-}}`); a new variable never gets a `STRATA_` name.
- An upstream sync reads upstream's names into XeStrata's as it carries a change over (the upstream-sync skill).
- `tools/lint/upstream_name.py` watches the strings of C++, Python and F#, the shell scripts' code, the documents and the web app.

<!-- upstream-name: on -->

### Model names: written out, never abbreviated

A model's name is written in full in code and text: `HunyuanImage 3.0` in prose, `hunyuanimage3` in identifiers (`hunyuanimage3_dit.cpp`, `HUNYUANIMAGE3_VAE_DUMP`), never `hy3`. Names that already exist stay (Qwen-Image-2.1's `qi_*`, Mage-Flow's `mf_*`), and so do paths outside the repository (the data folder's `hy3/`, the records' scripts); a comment citing one says it is such a path.

## The README and the documents

- `README.ja.md` and `docs/<NAME>.ja.md` are the originals; `README.md` and `docs/<NAME>.md` their English translations. A change goes into the Japanese file first and into the English one in the same commit, keeping the same sections. The English files keep the headings that code, records and links point to.
- Documents do not mark what was not checked (`unverified`, 未検証): that mark is for reports and commit messages.
- The two stages of generation are prefill and decode, in Japanese as in English (not プロンプト・出力・デコード, not prompt and output speed).

## Measurement records

- XeStrata's own measurements (benchmarks, A/B runs, probes, upstream-sync and release records) go into the data folder's `records/` (`<data>: LOCAL.md`), never into git. `bench/XeStrataBenchResults` holds only measurements others sent; upstream's records are not brought in. Commit messages and documents give the numbers, not a link to a record.
- Do not measure what is already known: a run that gives the same result each time (greedy, same binary, settings, text and machine state) is read from its record. Measure again only what can differ: a changed binary or setting, a run at temperature, a cell suspected of noise (once).

## Lints

Before committing, run `tools/lint/run.sh` from the repository root (no arguments: the files that differ from HEAD). The files a change touches get no new findings; earlier findings may stay. Its configurations are in `tools/lint/`:

| Files | Lints |
| --- | --- |
| C++ (and SYCL) | the compiler's warnings (`strata_warnings`; `-DSTRATA_WERROR=ON` makes them errors), clang-tidy (reads `compile_commands.json` from `build/free` or `XESTRATA_LINT_BUILD`), cppcheck |
| Python | flake8, ruff, mypy |
| Shell / CMake | shellcheck / cmake-lint |
| Markdown | mdl, lychee (`--offline`) |
| CSS / JavaScript / HTML | stylelint, eslint, tidy |
| text / every file | codespell / gitleaks, `tools/license_class.py` |
| C++, Python, F#, shell, documents, web | `tools/lint/upstream_name.py` |
| the repository | `reuse lint` |

- Other projects' files in `third_party/<area>/<project>/` (not XeStrata's `patches/`) get only gitleaks and `reuse lint`.
- Then run the existing tests the change can affect: CTest in the build folder, `python -m unittest` for `serve/` and `tools/` (the data folder's venv), `dotnet test launcher/Tests` and `python tools/launcher_i18n.py` for the launchers (the window headless; on a screen only at milestones).
- For memory and undefined-behavior bugs, build in a folder of its own with `-DSTRATA_SANITIZE=address,undefined` and run CTest there (host code only; not under SDE).
- Do not reformat code with clang-format or the like: the code is formatted by hand, and a reformat would make upstream's changes hard to carry over. Match the surrounding style.

## Developer tools

Install the lints, Intel SDE and the GPU profilers as [docs/DEVTOOLS.md](docs/DEVTOOLS.md) describes, and keep it current. Tools the distribution does not package go into `.lint/` or `.tools/` (ignored by git); nothing in the build or the tests depends on either.

## GitHub

Work through `git`, not `gh`, except: `gh release`; `gh issue list/view`, `gh pr list/view/diff/checks`; `gh pr merge/close/reopen`; `gh issue close/reopen`. Do not create, comment on or edit issues and pull requests with `gh`. A pull request says which modes it was built in and on which GPUs and CPUs it ran, marking what was not checked `unverified`; one that does not is asked for it.

## No agent self-attribution

Nothing written for this repository or its GitHub says an agent wrote it: no "Generated by" or "Written with" line, no `Co-Authored-By` naming an agent, no name of the agent or its model in a commit message, comment, document, release note, issue or pull request comment, or record, whatever a tool adds by default.

## Packages and versions

- Build the deb and rpm packages with `tools/package/build.sh` ([docs/BUILD.md](docs/BUILD.md#the-deb-and-rpm-packages)) and leave `XESTRATA_JOBS` unset (a lower value slows every build in the container, intel/llvm's included).
- What a user installs besides the package (drivers, runtimes, the render group) is checked by `xestrata --setup` (`check_runtime` in `setup.py`), which gives the command; the README, the documents and the release notes send the user there and do not list the packages.
- The version is the integrated upstream version (`STRATA_UPSTREAM_VERSION`) and `XESTRATA_REVISION` (XeStrata's releases on it, from 1, as a Debian revision): packages `<upstream>-<revision>`, the version name and tag `xe<upstream>-<revision>`. A new upstream version sets the revision back to 1. Tags from before 2026-10-13 keep their old form.

## AVX-512 code

The development machine has no AVX-512. A change to the AVX-512 paths runs the affected tests under Intel SDE emulating Ice Lake (`-icx`, the set `cpu_avx512_ok` requires; installation: [docs/DEVTOOLS.md](docs/DEVTOOLS.md#intel-sde-the-avx-512-paths)), and the same tests natively (the AVX2 path). SDE checks correctness, not speed.

- The paths: the files built with AVX-512 flags (`src/kernels/cpu/expert.cpp`, `iq_avx512.cpp`), the run-time choice (`cpu_avx512_ok` and its callers: `expert_layout.cpp`, `native_expert.cpp`, `generate.cpp`), and the int8 product's AVX-512 VNNI copy (`i8_gemm_avx512.cpp`, chosen by `cpu_avx512vnni_ok`; the one AVX-512 kernel written for XeStrata, 2026-11-08 — CPU kernels are otherwise AVX2 and AVX2 + AVX-VNNI).
- The tests: `sde64 -icx -- ./build/free/expert_multi_test`, `expert_parity --selftest`, `pool_test --selftest` (from the repository root), `cpu_i8_gemm_parity`.

## Intel Graphics Compiler (IGC): the GEP simplification of 2.27 to 2.30

IGC 2.27 to 2.30 miscompile, for Xe2 and Xe3 GPUs, a kernel that reads one pointer at `xt[N]` and at `xt[k * N]` with a constant k of 2 or more; the Linux runtime turns the pass off where such an IGC and GPU meet. The details, the workaround's conditions and how to read IGC's output: [docs/DEVTOOLS.md](docs/DEVTOOLS.md#intel-graphics-compiler-igc-the-gep-simplification-of-227-to-230).

- Write kernels that compile right either way: step a pointer by N from the last read (`glm5_hc_mean`), or index in one product (`x[(t * HC + s) * N + n]`).
