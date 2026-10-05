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
| `llama.cpp/` | [llama.cpp](https://github.com/ggml-org/llama.cpp) | MIT (its own `LICENSE`) | Not in the repository: setup fetches it at the pinned commit. ggml-cpu for the CPU experts, the image encoder's `mtmd`, and `gguf-py` for the tools |
| `oneMath/` | [oneMath](https://github.com/uxlfoundation/oneMath) | Apache-2.0 (`oneMath/LICENSE`) | The contrib and contrib-icpx modes' dense matrix products, through its oneMKL (Intel) and cuBLAS (NVIDIA) backends; the free mode does not use it. v0.9 (6ff3a43e555dbb20357017d48f0f6c6263259895) as released, with XeStrata's changes, under Apache-2.0 like the rest of it and each marked in its file's header: the cuBLAS backend's BF16 product (`src/blas/backends/cublas/cublas_helper.hpp`, `cublas_level3.cpp`; v0.9 has none) |
| `outfit/` | [Outfit](https://github.com/Outfitio/Outfit-Fonts) | SIL Open Font License 1.1 (`outfit/OFL.txt`) | The web app's font (WOFF2, Latin and Latin Extended). Without it the web app uses the system font |
