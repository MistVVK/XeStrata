"""The Inkling-2-Mini family: exact verified sizes & SHA-256 hashes, shape engine component, defaults, and sources.

Pure data and small helpers; the hardware auto-selection and placement rules live in inkling_fit.py.
Exact byte sizes and SHA-256 digests are pinned from the verified R2 `.strata` v1.1 checkpoints (`SHA256SUMS.txt`).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from strata_control import inkling_fit

FAMILY = "inkling-2-mini"
TAG = "inkling-2-mini-"               # run config file: strata-inkling-2-mini-sq-20.json
ARCH = "inkling2_loop"                # run config's "arch"; serve & Control route by it
SHAPE = "inkling2"                    # shape engine component: component:strata-inkling (strata_control/shape_engine.py)
COMPONENT = "strata-inkling"
EXE = "strata-inkling.exe" if os.name == "nt" else "strata-inkling"
TITLE = "Inkling-2-Mini"
BY = "Strata's native .strata v1.1 build of Thinking Machines' Inkling-2-Mini"
VISION_FILE = "Inkling-2-Mini-Vision.strata"
TOKENIZER_FILE = "tokenizer.json"
CHAT_TEMPLATE_FILE = "chat_template.jinja"
DEFAULT_CONTEXT = inkling_fit.DEFAULT_CONTEXT
CONTEXTS = inkling_fit.CONTEXTS
DEFAULT_DRAFTER = "dspark"
DRAFTERS = ("off", "dspark")
DEFAULT_VISION = True
DEFAULT_PARALLEL = inkling_fit.DEFAULT_PARALLEL
PARALLEL_CHOICES = inkling_fit.PARALLEL_CHOICES
MAX_PARALLEL = inkling_fit.MAX_PARALLEL
REASONING_EFFORTS = ("none",)
DEFAULT_REASONING_EFFORT = "none"
HAS_THINKING = False

#: file name -> (bytes, sha256 or None)
FILES = {
    "Inkling-2-Mini-SQ-20.strata": (19_346_685_952, "0bbfd52344f0dd4779a7807b5f73554a812f24c035e6877b63cd0d499f7e1b97"),
    "Inkling-2-Mini-SQ-15.strata": (14_348_316_672, "2ea32eed533afba257e2cbee7b073dc2c3da80fe55d1d2d794779b5498e30737"),
    "Inkling-2-Mini-SQ-10.strata": (10_471_526_400, "ac4d452ad03e4930abd9d46f9c5eec22b2f75a1da45a1e49bd24ccd58b065934"),
    VISION_FILE: (636_526_592, "6589eff47f9c20bc3a032faa3d449f10bf4f04a84cc6b2ee8a44d51d2b6fc677"),
}
#: Auxiliary tokenizer files hosted beside the `.strata` weights (Section 0 embeds chat_template + config; tokenizer.json holds the 200K o200k BPE vocab/merges).
TOKENIZER_FILES = {
    TOKENIZER_FILE: (26_178_678, None),
    CHAT_TEMPLATE_FILE: (6_868, None),
}
BUILD_ABOUT = {
    "SQ-20": "bit-exact lossless native SF4+MX8 LLM + DSpark drafter: 100% GPU on 24 GiB cards and 32+ GiB unified memory",
    "SQ-15": "flagship quality (beats UD-Q4_K_XL): 100% GPU on 16 GiB cards / 24 GiB Macs, or fast 3x loop-pinned hybrid on 12 GiB cards",
    "SQ-10": "compact high-speed build: 100% GPU on 12 GiB cards, 16 GiB Macs, 8 GiB cards, and Radeon 780M / Strix Point 890M iGPUs",
}


def build_file(build: str) -> str:
    return inkling_fit.BUILDS[build]["file"]


def files_of(build: str, vision: bool = DEFAULT_VISION) -> list[str]:
    """The .strata files for one build: the model checkpoint plus the Vision shard when vision is enabled."""
    out = [build_file(build)]
    if vision:
        out.append(VISION_FILE)
    return out


def size_of(name: str) -> int | None:
    got = FILES.get(name) or TOKENIZER_FILES.get(name)
    return got[0] if got else None


def sha256_of(name: str) -> str | None:
    got = FILES.get(name) or TOKENIZER_FILES.get(name)
    return got[1] if got else None


def model_key(build: str) -> str:
    """"SQ-20" -> "inkling-2-mini-sq-20"."""
    return TAG + build.lower()


SOURCES_FILE = Path(__file__).resolve().parents[1] / "data" / "inkling-sources.json"
SOURCE_ENV = "STRATA_INKLING_SOURCE"


def sources(env: dict | None = None, cfg: dict | None = None, shipped: Path | None = None) -> list[dict]:
    """Download sources in priority order: config's `inkling_source`, env `STRATA_INKLING_SOURCE`, then shipped `inkling-sources.json`."""
    env = os.environ if env is None else env
    out: list[dict] = []
    for item in ((cfg or {}).get("inkling_source"),):
        out += _normalise(item)
    if env.get(SOURCE_ENV):
        out += _normalise(env[SOURCE_ENV])
    try:
        data = json.loads((shipped or SOURCES_FILE).read_text(encoding="utf-8"))
        out += _normalise(data.get("sources") if isinstance(data, dict) else data)
    except (OSError, ValueError):
        pass
    return out


def _normalise(item) -> list[dict]:
    if isinstance(item, str):
        return [{"kind": "http", "base": item.rstrip("/")}] if item.startswith(("http://", "https://")) else []
    if isinstance(item, dict):
        base = str(item.get("base") or "").rstrip("/")
        kind = item.get("kind") if item.get("kind") in ("http", "http-parts") else "http"
        files = item.get("files") if isinstance(item.get("files"), dict) else {}
        if base.startswith(("http://", "https://")) or files:
            return [{"kind": kind, "base": base, "files": files}]
        return []
    if isinstance(item, list):
        return [x for sub in item for x in _normalise(sub)]
    return []


def urls_for(name: str, source: dict) -> list[str]:
    listed = (source.get("files") or {}).get(name)
    if listed:
        return [listed] if isinstance(listed, str) else list(listed)
    base = source.get("base") or ""
    if not base:
        return []
    return [f"{base}/{name}"]


def local_files(folder: str | os.PathLike, build: str | None = None) -> dict[str, Path]:
    """Inkling-2-Mini .strata files already present in `folder` with exact pinned sizes."""
    found: dict[str, Path] = {}
    base = Path(folder)
    for name, (size, _sha) in FILES.items():
        if build is not None and name.startswith("Inkling-2-Mini-SQ-") and name != build_file(build):
            continue
        p = base / name
        try:
            if p.is_file() and p.stat().st_size == size:
                found[name] = p
        except OSError:
            continue
    return found
