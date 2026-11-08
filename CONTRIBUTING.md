<!--
SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
SPDX-License-Identifier: LGPL-3.0-or-later
-->
# Contributing

English | [日本語](CONTRIBUTING.ja.md)

Reports and pull requests are welcome. Before sending one, read the sections below.

## The first rule: no dependence on the model number

XeStrata runs not only on the development machine's GPU and CPU but on any Intel Arc GPU (those without XMX and integrated
GPUs included), any x86-64 CPU with AVX2, and in the contrib-llvm mode on NVIDIA and AMD GPUs as well. So:

- **Never choose a code path by a model number, a product name, a device ID or a vendor name.** Choose it from what
  the hardware reports: the CPUID feature bits for the CPU (as `cpu_avx512_ok` does); for the GPU, its matrix
  combinations, sub-group sizes, GRF modes, local memory, memory size and largest allocation.
- **Do not hard-code sizes and thresholds.** Tile shapes, chunk sizes, cache sizes, thread counts, spin bounds and the
  like are derived from those reports at run time.
- **A value that has to be measured** is not settled on one card. Measure it on Intel, NVIDIA and AMD GPUs, give it a
  fallback that works on a smaller device, and say in a comment where it was measured and what it depends on.
- **A feature some devices lack** (XMX, large GRF mode, a SYCL extension) is checked before use, with a path that works
  without it.
- **Check a smaller configuration as well**, for example `XESTRATA_VRAM_LIMIT_MIB=8192 XESTRATA_MAX_ALLOC_MIB=4096
  XESTRATA_NO_XMX=1` with fewer CPU threads. Mark what was not checked `unverified` in the pull request.

A change that breaks this rule does not go in, however fast it is ([AGENTS.md](AGENTS.md#hardware-independence-the-first-rule)).

## Licenses

XeStrata's own code is under the LGPL, version 3 or later (LGPL-3.0-or-later; `COPYING.LESSER`, `COPYING`), and XeStrata as a combined work is distributed under it; the code from other projects keeps its own license, file by file as below.
The original Strata is under MIT; XeStrata has been under the LGPL since xe0.1.39.

Each file carries the license of the code in it, by where that code comes from.

| File | License | SPDX header |
| --- | --- | --- |
| Written for XeStrata | LGPL | XeStrata's copyright only, `LGPL-3.0-or-later` |
| Strata's, changed by XeStrata, or with code transcribed from ggml / llama.cpp | the original code stays under MIT; XeStrata's changes and the file as a whole are under the LGPL | both copyrights, `LGPL-3.0-or-later AND MIT` |
| Strata's, unchanged by XeStrata | MIT | Strata's copyright only, `MIT` |

- A new file carries XeStrata's header.
- A change to an unchanged Strata file makes it mixed.
  `python tools/license_class.py --fix` rewrites its header.
- This classification dates from 2026-10-13 and goes into the releases after xe0.1.43.1-1. Before it, the code from
  Strata and ggml was said to be under the LGPL along with the rest.
- Exceptions: XeStrata's changes to oneMath (the fork [MistVVK/oneMath](https://github.com/MistVVK/oneMath)) are under
  Apache-2.0, and its patches to intel/llvm (`third_party/main/intel-llvm/patches/`) under Apache-2.0 WITH LLVM-exception.
- Material from other projects goes into `third_party/`, one folder per project with its license text.
  What is not free software goes into `third_party/nonfree/`, and the build and the tests pass without it.

The details are in [AGENTS.md](AGENTS.md#licenses) and `NOTICE`.

## The additional permission under GPL section 7

`NOTICE` grants an additional permission under section 7 of the GPL, version 3:
XeStrata linked with a math or compute library a GPU or CPU maker publishes (oneMKL, cuBLAS and the like) and its
runtime libraries (the CUDA runtime and driver, the runtime libraries of oneAPI's compilers and the like) may be conveyed.

The contrib-llvm build and its packages link these libraries, which are not free software.
Without this permission, the LGPL alone would not allow conveying them.

A change goes into XeStrata under these terms, the additional permission included.
Whoever sends a change also agrees to the following:

- When MistVVK judges that an additional permission is needed to convey XeStrata linked with another library, MistVVK
  may add an additional permission under GPL section 7 to XeStrata.
- Such a permission covers the code that person sent as well.

If you do not agree, or the code you send cannot carry these permissions (code copied from another project, for
example), the change is, as a rule, not accepted.

The measurement records added to `bench/XeStrataBenchResults/` link with nothing, so this section does not apply to
them: they go in under LGPL-3.0-or-later without the additional permission, and need no such agreement.

## What goes in

Whoever wrote it (the maintainer, a pull request, a port from upstream), a change goes in only on these two conditions.

1. **It was built and tested with intel/llvm, in both the free and contrib-llvm modes.**
   A change built only with another compiler (Intel oneAPI's icpx, for example), or not built, does not go in.
   How to build: [docs/BUILD.md](docs/BUILD.md) and [docs/DEVTOOLS.md](docs/DEVTOOLS.md).
1. **It keeps [the first rule](#the-first-rule-no-dependence-on-the-model-number)**: no dependence on the model number.
   The free mode relies on nothing that is not free software.

A change that does not meet them does not go in as it is: it is a reference, written again as XeStrata's code.

A pull request says:

- which modes it was built in (free, contrib-llvm);
- on which GPUs and CPUs it ran;
- what was not checked, marked `unverified`.

Before committing, run `tools/lint/run.sh` and the existing tests the change can affect ([AGENTS.md](AGENTS.md#lints)).

## Reports

Open an [issue](https://github.com/MistVVK/XeStrata/issues).
Every report names the XeStrata version, how it was installed (from the source or the deb/rpm package), the build mode,
and the GPUs and CPU.
Remove user names, personal paths, host names, IP addresses and keys before pasting anything.

### A bug

- If you have an LLM agent that can work in a terminal, give it the [recovery and diagnosis instructions](docs/DIAGNOSE.md)
  and the symptom, and attach its report in the [report format](docs/DIAGNOSE.md#report-format).
- Otherwise, attach what happened, the command and settings, and the end of the engine's log.
  The log's path is the `log` in the settings of the model in use.

### The speed

- Run `xestrata --bench` and paste `records/bench-<date and time>/summary.md` from the data folder.
  Nothing is sent anywhere, so check what it says before pasting ([docs/BENCH.md](docs/BENCH.md#xestrata---bench)).
- To keep the record itself, send a pull request that adds a folder to `bench/XeStrataBenchResults/` (it goes in under
  LGPL-3.0-or-later without the additional permission).
