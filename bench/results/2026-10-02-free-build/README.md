<!--
SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
SPDX-License-Identifier: LGPL-3.0-or-later
-->
# The free build with intel/llvm — 2026-10-02

The engine builds in two modes (`STRATA_NONFREE`, AGENTS.md): free, with intel/llvm's DPC++ only, and nonfree, which allows Intel oneAPI's icpx.
This record builds the same source both ways on the B70 machine and compares them.

- free: `dpclang++` 6.2.0 from Ubuntu 26.04 (`dpclang-6`, source package `intel-dpcpp`), run without oneAPI's environment. The binary links `libsycl.so.8` from Ubuntu and nothing from oneAPI
- nonfree: icpx 2026.1.1 (`-DSTRATA_NONFREE=ON`)
- Both `STRATA_PORTABLE=ON`, the source of `cd0e784` before its lint fixes (`c39a0c7`: wider integer arithmetic; the 32-token runs after them gave the same tokens); runs as in [the new machine's record](../2026-10-02-new-machine/README.md), `--expert-cache 10000`, through [free_ab.sh](free_ab.sh) (two alternating rounds; logs in [runs/](runs/))
- Both builds pass CTest (31 tests, `expert_multi_test` skipped without AVX-512); the free build also passes `iq_parity` and `native_expert_parity` on the IQ3_S file

## intel/llvm 6.2 and the B70's XMX

intel/llvm 6.2.0's SYCL runtime knows the B70 (`intel_gpu_bmg_g31`, IP 0x05008000) but lists no matrix combination for it (oneAPI 2026.1's lists 53), so it refuses every kernel that uses `joint_matrix` ("no matrix hardware on the target device").
intel/llvm added the B70 to its table in `fce4f1245` ("Add support for bmg31 for matrix aspect", #20552, 2025-11-05); the first release with it is v7.0.0 (2026-07-13), and v6.3.0 (2025-12-23) is still without it.

The engine now asks the device at start: without an FP16 / BF16 8 x 16 x 16 combination (FP32 accumulators) and 16-wide sub-groups, the prompt path's products run on the vector engines (`src/kernels/xe/simt_gemm.cpp`), and it says so once.
`STRATA_NO_XMX=1` forces that path with any compiler.
That path was later replaced by DP4a, 1.6 times slower than XMX instead of 3.1 ([record](../2026-10-02-dp4a/README.md)).

## Result

IQ3_S; MTP decode of the 20-token prompt (128 tokens); the 26,292-token prompt (prefill auto, int8 KV, 24 tokens):

| Build | Decode (tok/s) | MTP acceptance | Prompt (tok/s) |
| --- | --- | --- | --- |
| icpx, XMX | 58.4 / 59.1 | 77 of 155 | 1017 / 1020 |
| icpx, `STRATA_NO_XMX=1` | 63.9 / 64.0 | 81 of 143 | 324 / 324 |
| free (dpclang 6.2, no XMX) | 63.0 / 63.2 | 80 of 143 | 338 / 339 |

- Without XMX the prompt path is 3.1 times slower; the free build's vector-engine path is as fast as icpx's
- Decode does not use these products. Its rates differ with the drafts accepted: the 20-token prompt goes through the products too, their different rounding changes the generation from token 77 of 128, and the acceptance with it. Each build repeats its own tokens exactly
- After the long prompt, all three generate the same 24 tokens; their logits differ from the XMX build's by at most 2.74
- The first run of a new free build spends about 20 s in the JIT (decode 1.9 tok/s in that run); later runs use the driver's cache

## intel/llvm 7.1.1 built from source

`tools/intel_llvm_build.py` built v7.1.1 (`504366f4b`) in 13 minutes on the development machine (28 threads); it keeps the clone (2.8 GB) and `install/` (0.7 GB) and deletes the build tree.
Its runtime reports the B70's FP16 and BF16 matrix combinations (`tools/xmx_probe.cpp`: fp16=1 bf16=1; the integrated UHD 770 none), so the engine built with it is free and takes the XMX path.
It passes CTest; with its newer clang, `src/core/weights.cpp` warned about `%llu` for `uint64_t` (fixed with `SCNu64`).

[llvm7_ab.sh](llvm7_ab.sh), the same runs as above, two alternating rounds:

| Build | Decode (tok/s) | MTP acceptance | Prompt (tok/s) |
| --- | --- | --- | --- |
| icpx, XMX | 59.1 / 58.9 | 77 of 155 | 1003 / 1019 |
| intel/llvm 7.1.1, XMX (free) | 47.5 / 58.7 | 79 of 155 | 1017 / 1018 |

- The prompt path is as fast as icpx's. The first decode run compiled the MTP kernels (the warm-up ran without MTP); the second is as fast as icpx's
- After the long prompt the 24 tokens are the same as icpx's; the logits differ by at most 2.42 (another compiler's rounding). With the 20-token prompt the drafts accepted differ (79 against 77 of 155); each build repeats its own tokens

## intel/llvm in the distributions (2026-10-02)

| Distribution | Package | Has the B70's matrix table |
| --- | --- | --- |
| Ubuntu 26.04 LTS | `dpclang-6` 6.2.0 (none in backports) | no |
| Ubuntu, next release (stonking) | `dpclang-7` 7.0.1 (`intel-dpcpp-7`) | yes by version (v7.1.1 built from source has it) |
| NixOS unstable | `intel-llvm` 7.1.1 | yes by version, `unverified` |
| NixOS 26.05 | `intel-llvm` unstable-2025-11-14 | yes by date, `unverified` |
| Fedora 43, 44, rawhide; Debian (sid, forky); Gentoo | none | — |
| openSUSE Tumbleweed | none (AdaptiveCpp 25.10, another SYCL implementation) | — |
| Arch | Intel oneAPI's binaries only (not free) | — |
