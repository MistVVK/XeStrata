#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
"""tools/package/engine_notices.py - the parts from other projects a build of XeStrata carries or is compiled with,
their licenses and texts: one list (ROWS) for the deb and rpm packages, the Windows 7z (windows_bundle.py takes its
rows from here) and a build from source (the server reads it for About > Licenses).

    engine_notices.py --src DIR --build DIR [--rocm DIR] --out FILE

writes FILE (NOTICES.json, the packages' <engine>/NOTICES.json) for the engine built in DIR: the parts its PARTS.json
(CMakeLists.txt writes it) names, each with its license texts read in, so the server shows them without the source
tree.  NOTICES.json is a list of {"name", "version", "license", "source", "free", "bundled", "texts": [{"name",
"text"}]}; "free": a free software license (DFSG), "bundled": carried in the package, or else what the engine opens
where the user installed it (the contrib mode's oneMKL and NVIDIA's libraries)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# A row: name, version (a key of the versions given, or the text), license, source, the texts (ROOT:glob; roots:
# src the source tree, llama llama.cpp's, onemath oneMath's, sycl intel/llvm's install, rocm the ROCm BLAS packages'
# texts (tools/package/win_rocm.sh's on Windows), or a URL), the Windows 7z's files it covers (globs), the build's part
# it goes with (PARTS.json's "parts"; None: every build), the texts on Linux where they differ, and whether it is free
# software and carried (False: opened where the user installed it, never in a package).
ROWS: list[dict[str, Any]] = [
    {"name": "ggml, mtmd and gguf-py (llama.cpp: compiled into the engine and the image encoders; the tools' GGUF "
             "reader)", "version": "llama.cpp", "license": "MIT", "source": "https://github.com/ggml-org/llama.cpp",
     "texts": ["llama:LICENSE"]},
    {"name": "oneMath (XeStrata's fork)", "version": "onemath", "license": "Apache-2.0",
     "source": "https://github.com/MistVVK/oneMath", "texts": ["onemath:LICENSE"],
     "linux_texts": ["src:third_party/main/oneMath/LICENSE"], "part": "onemath",
     "win": ["app/engine/onemath.dll", "app/engine/onemath_blas_mklgpu.dll", "app/engine/onemath_blas_cublas.dll",
             "app/engine/onemath_blas_rocblas.dll"]},
    {"name": "ROCm BLAS headers (rocBLAS, hipBLASLt and hipblas-common; compiled into oneMath's rocBLAS backend)",
     "version": "rocm", "license": "MIT", "source": "https://github.com/ROCm/rocm-libraries",
     "texts": ["rocm:LICENSE-*.txt"], "part": "onemath_rocblas"},
    {"name": "intel/llvm (DPC++) SYCL runtime, with Unified Runtime, UMF, hwloc and zstd", "version": "intel-llvm",
     "license": "Apache-2.0 WITH LLVM-exception; UMF Apache-2.0 WITH LLVM-exception; hwloc BSD-3-Clause; zstd "
                "BSD-3-Clause; Level Zero loader headers MIT; AMD's HIP headers MIT",
     "source": "https://github.com/intel/llvm", "texts": ["sycl:licenses/*"],
     "linux_texts": ["src:third_party/main/intel-llvm/LICENSE.TXT"]},
    {"name": "ROCm Device Libraries (in the engine's code for AMD GPUs)", "version": "f30ae3e (ROCm 7.14.1)",
     "license": "NCSA", "source": "https://github.com/ROCm/llvm-project",
     "texts": ["src:third_party/main/rocm-device-libs/LICENSE.TXT"], "part": "hip"},
    {"name": "Outfit (the web app's font)", "version": "third_party/main/outfit", "license": "OFL-1.1",
     "source": "https://github.com/Outfitio/Outfit-Fonts", "texts": ["src:third_party/main/outfit/OFL.txt"]},
    {"name": "KaTeX (the web chat's math)", "version": "0.19.1", "license": "MIT", "source": "https://katex.org/",
     "texts": ["src:third_party/main/katex/LICENSE"]},
    {"name": "mermaid (the web chat's flowcharts)", "version": "12.3.0",
     "license": "MIT; the packages it bundles: ISC, BSD-3-Clause, Apache-2.0, EPL-2.0, MIT "
                "(third_party/main/mermaid/BUNDLED.md)", "source": "https://mermaid.js.org/",
     "texts": ["src:third_party/main/mermaid/LICENSE", "src:third_party/main/mermaid/BUNDLED.md"]},
    {"name": "XGrammar (compiled into the engine: grammar-constrained decoding)", "version": "0.2.8",
     "license": "Apache-2.0", "source": "https://github.com/mlc-ai/xgrammar",
     "texts": ["src:third_party/main/xgrammar/LICENSE", "src:third_party/main/xgrammar/NOTICE"], "part": "xgrammar"},
    {"name": "picojson (in XGrammar, compiled into the engine)", "version": "XGrammar 0.2.8's copy",
     "license": "BSD-2-Clause", "source": "https://github.com/kazuho/picojson",
     "texts": ["src:third_party/main/picojson/LICENSE"], "part": "xgrammar"},
    {"name": "DLPack (a header XGrammar reads)", "version": "bbd2f4d", "license": "Apache-2.0",
     "source": "https://github.com/dmlc/dlpack", "texts": ["src:third_party/main/dlpack/LICENSE"], "part": "xgrammar"},
    {"name": "AOTriton (its library and kernels' images)", "version": "0.14.2b",
     "license": "MIT; the images can carry parts of aiter (MIT), FlyDSL's kernels (Apache-2.0) and Triton (MIT)",
     "source": "https://github.com/ROCm/aotriton",
     "texts": ["src:third_party/main/aotriton/LICENSE", "src:third_party/main/aotriton/LICENSE.aiter",
               "src:third_party/main/aotriton/LICENSE.FlyDSL", "src:third_party/main/aotriton/LICENSE.triton"],
     "part": "aotriton", "win": ["app/engine/aotriton_v2.dll", "app/engine/aotriton.images/*"]},
    {"name": "liblzma of XZ Utils (linked into AOTriton's library)", "version": "5.8.4", "license": "0BSD",
     "source": "https://tukaani.org/xz/", "texts": ["src:third_party/main/xz/COPYING.0BSD"], "part": "aotriton"},
    {"name": "oneDNN", "version": "3.14.2",
     "license": "Apache-2.0 (parts of it under other licenses, listed in THIRD-PARTY-PROGRAMS)",
     "source": "https://github.com/uxlfoundation/oneDNN",
     "texts": ["src:third_party/main/oneDNN/LICENSE", "src:third_party/main/oneDNN/THIRD-PARTY-PROGRAMS"],
     "part": "onednn", "win": ["app/engine/dnnl.dll"]},
    {"name": "cuDNN Frontend (compiled into the image engine)", "version": "1.32.2",
     "license": "Apache-2.0 (parts MIT)",
     "source": "https://github.com/NVIDIA/cudnn-frontend", "texts": ["src:third_party/main/cudnn-frontend/LICENSE.txt"],
     "part": "cudnn_frontend"},
    {"name": "CUTLASS (compiled into the engine)", "version": "4.8.1", "license": "BSD-3-Clause",
     "source": "https://github.com/NVIDIA/cutlass", "texts": ["src:third_party/main/cutlass/LICENSE.txt"],
     "part": "cutlass"},
    {"name": "comfy-kitchen's HIP kernels (the engine's module for AMD GPUs)", "version": "v0.2.41",
     "license": "Apache-2.0", "source": "https://github.com/Comfy-Org/comfy-kitchen",
     "texts": ["src:third_party/main/comfy-kitchen/LICENSE", "src:third_party/main/comfy-kitchen/NOTICE"],
     "part": "kitchen"},
    {"name": "stb_image (compiled into the image engine)", "version": "2.30", "license": "MIT OR Unlicense",
     "source": "https://github.com/nothings/stb", "texts": ["src:third_party/main/stb/LICENSE"]},
    {"name": "quantscope", "version": "bfdf205", "license": "MIT OR Apache-2.0",
     "source": "https://github.com/1872183316/quantscope",
     "texts": ["src:third_party/main/quantscope/LICENSE-MIT", "src:third_party/main/quantscope/LICENSE-APACHE"]},
    {"name": "py3langid (the translation APIs' language detection)", "version": "0.4.0", "license": "BSD-3-Clause",
     "source": "https://github.com/adbar/py3langid", "texts": ["src:third_party/main/py3langid/LICENSE"]},
    # what the contrib mode's engine opens where the user installed it: never carried, not free software
    {"name": "Intel oneMKL (Intel GPUs' dense products)", "version": "the one installed",
     "license": "Intel Simplified Software License", "source": "https://www.intel.com/content/www/us/en/developer/"
     "tools/oneapi/onemkl.html", "texts": [], "part": "onemkl", "free": False, "bundled": False},
    {"name": "NVIDIA cuBLAS and the CUDA runtime (NVIDIA GPUs' dense products)", "version": "the one installed",
     "license": "NVIDIA CUDA Toolkit End User License Agreement",
     "source": "https://docs.nvidia.com/cuda/eula/index.html", "texts": [], "part": "cuda", "free": False,
     "bundled": False},
    {"name": "NVIDIA cuDNN (the image engine's attention on NVIDIA GPUs)", "version": "the one installed",
     "license": "NVIDIA cuDNN Software License Agreement",
     "source": "https://docs.nvidia.com/deeplearning/cudnn/latest/reference/eula.html", "texts": [],
     "part": "cudnn_frontend", "free": False, "bundled": False},
]


def rows_for(parts: set, linux: bool) -> list[dict]:
    """ROWS of a build with these parts, with their texts on Linux or Windows."""
    out = []
    for r in ROWS:
        if r.get("part") and r["part"] not in parts:
            continue
        out.append({**r, "texts": r.get("linux_texts", r["texts"]) if linux else r["texts"]})
    return out


def read_texts(row: dict, roots: dict) -> list[dict]:
    """The row's license texts: [{"name", "text"}]; a root not given, or a glob that finds nothing, stops."""
    got = []
    for t in row["texts"]:
        root, rel = t.split(":", 1)
        if root not in roots or roots[root] is None:
            raise SystemExit(f"engine_notices: {row['name']}: no folder for {t}")
        found = sorted(Path(roots[root]).glob(rel))
        if not found:
            raise SystemExit(f"engine_notices: {row['name']}: no license text {t}")
        got += [{"name": f.name, "text": f.read_text(encoding="utf-8", errors="replace")} for f in found]
    return got


def entry(row: dict, version: str, texts: list[dict]) -> dict:
    return {"name": row["name"], "version": version, "license": row["license"], "source": row["source"],
            "free": row.get("free", True), "bundled": row.get("bundled", True), "texts": texts}


def build_notices(src: Path, build: Path, rocm: Path | None = None) -> list[dict]:
    """NOTICES.json's list for the engine built in `build` (its PARTS.json), the texts read from the source tree and
    the build's own folders."""
    rec = json.loads((build / "PARTS.json").read_text(encoding="utf-8"))
    parts = set(rec.get("parts", []))
    roots = {"src": src, "llama": rec.get("llama_cpp_dir") or None, "onemath": rec.get("onemath_dir") or None,
             "rocm": rocm}
    versions = {"llama.cpp": rec.get("llama_cpp") or "", "onemath": rec.get("onemath") or "",
                "intel-llvm": rec.get("intel_llvm") or "", "rocm": rec.get("rocm") or "the build's"}
    if rocm is None and "onemath_rocblas" in parts:
        parts.discard("onemath_rocblas")   # a build from source: the distribution's ROCm packages carry their texts
    return [entry(r, versions.get(r["version"], r["version"]), read_texts(r, roots))
            for r in rows_for(parts, linux=True)]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--build", type=Path, required=True)
    ap.add_argument("--rocm", type=Path, help="the ROCm BLAS packages' license texts (LICENSE-<package>.txt)")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(build_notices(a.src, a.build, a.rocm), indent=1, ensure_ascii=False) + "\n",
                     encoding="utf-8")
    print(f"engine_notices: {a.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
