<!--
SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
SPDX-License-Identifier: LGPL-3.0-or-later
-->
# Free third-party material

Code and files from other projects that are free software, the kind Debian main takes. XeStrata may build and run
with them. Each project's folder keeps its own license text.

| Folder | Project | License | Used for |
| --- | --- | --- | --- |
| `ggml/` | [ggml / llama.cpp](https://github.com/ggml-org/llama.cpp) | MIT (`ggml/LICENSE`) | `ggml-common.h`, an unmodified copy at the commit in `ggml/VERSION.txt` (it stays MIT): the block layouts and codebook grids of the quantized formats. Needed to build the engine. Update it only together with the pinned llama.cpp (the engine also links ggml-cpu from it) |
| `intel-llvm/` | [intel/llvm](https://github.com/intel/llvm) (DPC++) | Apache-2.0 with LLVM exceptions (`intel-llvm/LICENSE.TXT`) | Not the compiler itself, which `tools/intel_llvm_build.py` clones from intel/llvm at the release it pins (v7.1.1) and builds into `.tools/`: XeStrata's fixes to it, `patches/NN-<id>.patch`, which the script applies in order before building. They are under intel/llvm's license, as changes to its files; each says what it fixes. The deb and rpm packages carry the SYCL runtime of the intel/llvm built so (`libsycl`, UR's loader and adapters, `libumf`), with this license text |
| `llama.cpp/` | [llama.cpp](https://github.com/ggml-org/llama.cpp) | MIT (its own `LICENSE`) | Not in the repository: setup fetches it at the pinned commit. ggml-cpu for the CPU experts, the image encoder's `mtmd`, and `gguf-py` for the tools. The deb and rpm packages carry `gguf-py` with llama.cpp's `LICENSE` |
| `oneMath/` | [oneMath](https://github.com/uxlfoundation/oneMath) | Apache-2.0 (`oneMath/LICENSE`) | Not oneMath itself, which `CMakeLists.txt` fetches from XeStrata's fork ([MistVVK/oneMath](https://github.com/MistVVK/oneMath), branch `xestrata`) at the tag it pins (`xestrata-1`, its archive's SHA-256 checked), or takes from `STRATA_ONEMATH_SOURCE` (a local archive or URL); in the contrib and contrib-icpx modes, and in the free mode for AMD GPUs. The fork is oneMath's commit 6ff3a43 with XeStrata's changes as commits, under oneMath's license, each marking the files it changes in their headers: the cuBLAS backend's BF16 product (oneMath has none), and the rocBLAS backend's 16-bit products through hipBLASLt where it has kernels for the GPU (ROCm 7.1's rocBLAS has no tuned ones for RDNA4). oneMath runs the dense matrix products through its oneMKL (Intel), cuBLAS (NVIDIA) and rocBLAS (AMD) backends. The deb and rpm packages carry the oneMath built so (its rocBLAS backend; the contrib ones its oneMKL and cuBLAS backends too), with this license text |
| `outfit/` | [Outfit](https://github.com/Outfitio/Outfit-Fonts) | SIL Open Font License 1.1 (`outfit/OFL.txt`) | The web app's font (WOFF2, Latin and Latin Extended). Without it the web app uses the system font |
