"""tools/_paths.py - where the tools find a GGUF library.

The preparation tools use Strata's own reader/writer and quantizers (tools/strata_gguf.py, tools/strata_quants.py): gguf_lib() returns it.
$STRATA_GGUF_PY (a folder that holds llama.cpp's `gguf` package) still selects that package instead, for two releases (the byte-identical
A/B of the pack tools runs both); llama.cpp's HF converter (NVFP4-local) imports gguf-py itself and always uses the vendored copy.

Where gguf-py is looked for (gguf_py(), add_gguf_py()), in order: $STRATA_GGUF_PY (a folder that holds the `gguf` package, e.g. <llama.cpp>/gguf-py), then the vendored copy
<engine>/third_party/strata-gguf-py (frozen at the pinned llama.cpp commit, no download), then the development tree's
../../.ref/llama.cpp/gguf-py, then (#3269) the install folders: in the Control package setup.py's per-user install folder
($STRATA_INSTALL_DIR, else the app's folder, the Terminal's remembered installs), so a tool run by hand from the package finds it there.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ENGINE = Path(__file__).resolve().parents[1]
VENDORED = ENGINE / "third_party" / "strata-gguf-py"       # holds gguf/ (the package) and converter/ (llama.cpp's HF converter)


def _install_dirs() -> list[Path]:
    """The per-user install folders a Control-mode setup.py works in. Best effort: never raises."""
    out: list[Path] = []
    raw = os.environ.get("STRATA_INSTALL_DIR")
    if raw:
        out.append(Path(raw).expanduser())
    try:
        if str(ENGINE) not in sys.path:
            sys.path.append(str(ENGINE))
        from strata_control import discovery
        out.append(discovery.app_install_dir())
        out += [pl.path for pl in discovery.places(here=ENGINE)]
    except Exception:
        pass
    return out


def gguf_py() -> str:
    cands = [os.environ.get("STRATA_GGUF_PY"), ENGINE / "third_party" / "strata-gguf-py",
             ENGINE.parents[min(1, len(ENGINE.parents) - 1)] / ".ref" / "llama.cpp" / "gguf-py"]   # the dev tree; a shallow install (G:\Strata) has no grandparent
    cands += [d / "third_party" / "strata-gguf-py" for d in _install_dirs()]
    for c in cands:
        if c and (Path(c) / "gguf").is_dir():
            return str(c)
    sys.exit("gguf-py was not found. It ships with Strata in third_party/strata-gguf-py; set STRATA_GGUF_PY to the folder that holds the "
             "`gguf` package (for example <llama.cpp>/gguf-py) to use another copy")


def gguf_lib():
    """The GGUF library the preparation tools use, with gguf-py's names (GGMLQuantizationType, GGUFReader, GGUFWriter, GGUFValueType, quants):
    llama.cpp's gguf-py when STRATA_GGUF_PY names a folder that holds it (kept for two releases), else tools/strata_gguf.py."""
    own = os.environ.get("STRATA_GGUF_PY")
    if own:
        if not (Path(own) / "gguf").is_dir():
            sys.exit("STRATA_GGUF_PY=%s does not hold a `gguf` package" % own)
        if own not in sys.path:
            sys.path.insert(0, own)
        import gguf
        return gguf
    here = str(Path(__file__).resolve().parent)
    if here not in sys.path:
        sys.path.insert(0, here)
    import strata_gguf
    return strata_gguf


def add_gguf_py() -> None:
    p = gguf_py()
    if p not in sys.path:
        sys.path.insert(0, p)


def converter_dir() -> str:
    """llama.cpp's Hugging Face converter (convert_hf_to_gguf.py and conversion/), vendored beside gguf-py (the NVFP4-local preparation)."""
    for c in (ENGINE / "third_party" / "strata-gguf-py" / "converter", *[d / "third_party" / "strata-gguf-py" / "converter" for d in _install_dirs()]):
        if (c / "convert_hf_to_gguf.py").is_file():
            return str(c)
    sys.exit("llama.cpp's converter was not found (third_party/strata-gguf-py/converter)")
