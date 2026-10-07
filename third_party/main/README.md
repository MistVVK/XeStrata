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
| `oneMath/` | [oneMath](https://github.com/uxlfoundation/oneMath) | Apache-2.0 (`oneMath/LICENSE`) | Not oneMath itself, which `CMakeLists.txt` fetches in the contrib and contrib-icpx modes (the free mode does not use it) from GitHub at the release it pins (v0.9, 6ff3a43e555dbb20357017d48f0f6c6263259895, its archive's SHA-256 checked), or takes from `STRATA_ONEMATH_SOURCE` (a local archive or URL): XeStrata's changes to it, `patches/NN-<id>.patch`, which CMake applies in order. They are under oneMath's license, as changes to its files, and each marks the files it changes in their headers: `01-cublas-bf16-gemm`, the cuBLAS backend's BF16 product (v0.9 has none). oneMath runs the dense matrix products through its oneMKL (Intel) and cuBLAS (NVIDIA) backends. The contrib packages carry the oneMath built so, with this license text |
| `quantscope/` | [quantscope](https://github.com/1872183316/quantscope) | MIT or Apache-2.0, at your option (`quantscope/LICENSE-MIT`, `quantscope/LICENSE-APACHE`) | `quantscope.py`, unmodified at commit bfdf205 but for a line saying where it comes from (as upstream Strata took it, #911): it reads a GGUF's headers only (a file, a folder, a URL, a ModelScope or Hugging Face repository). `tools/strata_inspect.py` and `setup.py --inspect` use it to say what a GGUF really is and whether XeStrata runs it. Without it only that check is gone |
| `outfit/` | [Outfit](https://github.com/Outfitio/Outfit-Fonts) | SIL Open Font License 1.1 (`outfit/OFL.txt`) | The web app's font (WOFF2, Latin and Latin Extended). Without it the web app uses the system font |
