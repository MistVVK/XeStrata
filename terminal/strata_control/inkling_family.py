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
DSPARK_GGUF_FILE = "dspark-Inkling-2-Mini-Q8_0.gguf"
MMPROJ_BF16_FILE = "mmproj-BF16.gguf"
MMPROJ_F16_FILE = "mmproj-F16.gguf"
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

#: Unsloth Dynamic GGUF files hosted on huggingface.co/unsloth/Inkling-2-Mini-GGUF (exact byte sizes & LFS SHA-256 digests).
UNSLOTH_GGUF_FILES = {
    "Inkling-2-Mini-UD-Q8_K_XL.gguf": (22_867_451_616, "dfb04bf37bee457f98800989dedf93ec442ede8636dbea9433992f1e527e9a41"),
    "Inkling-2-Mini-UD-Q4_K_XL.gguf": (17_629_759_456, "e62d463c135ffe283df8d94b279a96ca145ac128b52ce854478613045bf4de15"),
    "Inkling-2-Mini-UD-Q4_K_M.gguf": (16_610_543_328, "f90c5a4df44d335ac9f9eea2a15c31d6c5a377b1b3cd9ff7fc2cf123f3cdf7e1"),
    "Inkling-2-Mini-UD-IQ4_XS.gguf": (15_733_755_232, "a861d21e7c6d7bc69d8cdcdfb9eb195fd37ef1cd2ab31499db61cd7d30d9b3a1"),
    "Inkling-2-Mini-UD-Q3_K_XL.gguf": (13_631_778_016, "f72f396d1b479b14d76346df80baa449c8b1f399653bff38fa4764b359a40048"),
    "Inkling-2-Mini-UD-IQ3_XXS.gguf": (11_663_309_504, "bd7ef0865adf119b5183e45e741c629bafa649769b33b5ebf689bb4bcc4690d6"),
    "Inkling-2-Mini-UD-Q2_K_XL.gguf": (9_138_715_328, "20292e8142b69535ca88a5d04439d0fdeb4dc698d44c002dca706dcf92901441"),
    "Inkling-2-Mini-UD-IQ2_XXS.gguf": (8_486_236_352, "4fe958f73a667d8082f911a4347e52bc34696a7816cf3d5b9f7c8a81d5c5ceaa"),
    "Inkling-2-Mini-UD-IQ1_M.gguf": (8_024_441_024, "544e0b2412ba2392272b0510b0c8f53a1ab872b73c2ba4f7488de0d5d61467d9"),
    "Inkling-2-Mini-UD-IQ1_S.gguf": (7_764_291_776, "c09c028b050c108624682baeb28ac6e0cdb936af021ac637c8c3609f923ea4b8"),
    DSPARK_GGUF_FILE: (402_278_560, "2ec78510da837e8fc3be54a14757a0c5fbd2d8d9e85a0d92853fafdf674f3773"),
    MMPROJ_BF16_FILE: (641_805_760, "2c165c78790e78a3f45a2d1e9e931083e52ee69d45ff2a0dc9c281fdf1a666e7"),
    MMPROJ_F16_FILE: (639_348_160, "3bff75ffeb2d6e0371ba7bd42e30d26a7d98aea681fa44b03b5a20891cfa2c5f"),
}

#: file name -> (bytes, sha256 or None)
FILES = {
    "Inkling-2-Mini-SQ-20.strata": (19_346_685_952, "0bbfd52344f0dd4779a7807b5f73554a812f24c035e6877b63cd0d499f7e1b97"),
    "Inkling-2-Mini-SQ-15.strata": (14_348_316_672, "2ea32eed533afba257e2cbee7b073dc2c3da80fe55d1d2d794779b5498e30737"),
    "Inkling-2-Mini-SQ-10.strata": (10_471_526_400, "ac4d452ad03e4930abd9d46f9c5eec22b2f75a1da45a1e49bd24ccd58b065934"),
    VISION_FILE: (636_526_592, "6589eff47f9c20bc3a032faa3d449f10bf4f04a84cc6b2ee8a44d51d2b6fc677"),
    **UNSLOTH_GGUF_FILES,
}
#: Auxiliary tokenizer files hosted beside the `.strata` weights (Section 0 embeds chat_template + config; tokenizer.json holds the 200K o200k BPE vocab/merges).
TOKENIZER_FILES = {
    TOKENIZER_FILE: (26_178_678, None),
    CHAT_TEMPLATE_FILE: (6_868, None),
}
BUILD_ABOUT = {
    "SQ-20": "bit-exact lossless native SF4+MX8 LLM + DSpark drafter: 100% GPU on 24 GiB cards and 32+ GiB unified memory",
    "SQ-15": "flagship quality (beats UD-Q4_K_XL): 100% GPU on 20+ GiB cards / 24 GiB Macs, or fast 3x loop-pinned hybrid on 12–16 GiB cards",
    "SQ-10": "compact high-speed build: 100% GPU on 12 GiB cards, 16 GiB Macs, 4–8 GiB hybrid cards, and Radeon 780M / Strix Point 890M iGPUs",
    "UD-Q8_K_XL": "Unsloth Dynamic ~8-bit GGUF + DSpark drafter: near-lossless GGUF for 24+ GiB GPUs or 32+ GiB RAM",
    "UD-Q4_K_XL": "Unsloth Dynamic ~4.5-bit GGUF + DSpark drafter: high-quality GGUF for 16–24 GiB GPUs or hybrid RAM",
    "UD-Q4_K_M": "Unsloth Dynamic 4-bit medium GGUF + DSpark drafter: balanced GGUF for 16–24 GiB GPUs or hybrid RAM",
    "UD-IQ4_XS": "Unsloth Dynamic 4-bit i-quant GGUF + DSpark drafter: compact 4-bit GGUF for 16 GiB GPUs or hybrid RAM",
    "UD-Q3_K_XL": "Unsloth Dynamic ~3.5-bit GGUF + DSpark drafter: strong 3-bit GGUF for 12–16 GiB GPUs or hybrid RAM",
    "UD-IQ3_XXS": "Unsloth Dynamic 3-bit i-quant GGUF + DSpark drafter: compact 3-bit GGUF for 8–12 GiB GPUs or 4–6 GiB hybrid",
    "UD-Q2_K_XL": "Unsloth Dynamic ~2.5-bit GGUF + DSpark drafter: small GGUF for 4–12 GiB GPUs and 16 GiB RAM",
    "UD-IQ2_XXS": "Unsloth Dynamic 2-bit i-quant GGUF + DSpark drafter: ultra-compact GGUF for 4–8 GiB GPUs and 16 GiB RAM",
    "UD-IQ1_M": "Unsloth Dynamic ~1.7-bit GGUF + DSpark drafter: tiny GGUF for 4–8 GiB GPUs and tight RAM",
    "UD-IQ1_S": "Unsloth Dynamic ~1.5-bit GGUF + DSpark drafter: smallest Inkling-2-Mini download for 4–8 GiB GPUs",
}


def is_gguf_build(build: str | None) -> bool:
    return inkling_fit.is_gguf_build(build)


def build_file(build: str) -> str:
    return inkling_fit.BUILDS[build]["file"]


def files_of(build: str, vision: bool = DEFAULT_VISION, dspark: bool = True) -> list[str]:
    """The files for one build: model checkpoint (+ companion DSpark GGUF for Unsloth GGUFs) + Vision shard when enabled."""
    out = [build_file(build)]
    if is_gguf_build(build) and dspark:
        out.append(DSPARK_GGUF_FILE)
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
    """"SQ-20" -> "inkling-2-mini-sq-20", "UD-Q2_K_XL" -> "inkling-2-mini-ud-q2_k_xl"."""
    return TAG + build.lower()


SOURCES_FILE = Path(__file__).resolve().parents[1] / "data" / "inkling-sources.json"
SOURCE_ENV = "STRATA_INKLING_SOURCE"
GGUF_SOURCE_ENV = "STRATA_INKLING_GGUF_SOURCE"


def sources(env: dict | None = None, cfg: dict | None = None, shipped: Path | None = None) -> list[dict]:
    """Download sources for native `.strata` / tokenizer files in priority order: config `inkling_source`, env `STRATA_INKLING_SOURCE`, then shipped `inkling-sources.json`."""
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


def gguf_sources(env: dict | None = None, cfg: dict | None = None, shipped: Path | None = None) -> list[dict]:
    """Download sources for Unsloth GGUF files in priority order: config/env overrides first, then shipped `gguf_sources`."""
    env = os.environ if env is None else env
    out: list[dict] = []
    for item in ((cfg or {}).get("inkling_gguf_source"), (cfg or {}).get("inkling_source")):
        out += _normalise(item)
    if env.get(GGUF_SOURCE_ENV):
        out += _normalise(env[GGUF_SOURCE_ENV])
    if env.get(SOURCE_ENV):
        out += _normalise(env[SOURCE_ENV])
    try:
        data = json.loads((shipped or SOURCES_FILE).read_text(encoding="utf-8"))
        if isinstance(data, dict) and data.get("gguf_sources"):
            out += _normalise(data["gguf_sources"])
    except (OSError, ValueError):
        pass
    return out


def sources_for(name: str, env: dict | None = None, cfg: dict | None = None, shipped: Path | None = None) -> list[dict]:
    """Return the prioritized download sources for `name` (Unsloth GGUF repo for GGUF files, Strata repo for .strata/tokenizer)."""
    if name in UNSLOTH_GGUF_FILES or str(name).lower().endswith(".gguf"):
        return gguf_sources(env=env, cfg=cfg, shipped=shipped)
    return sources(env=env, cfg=cfg, shipped=shipped)


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
    """Inkling-2-Mini .strata / .gguf files already present in `folder` with exact pinned sizes."""
    found: dict[str, Path] = {}
    base = Path(folder)
    wanted_model = build_file(build) if build is not None and build in inkling_fit.BUILDS else None
    for name, (size, _sha) in FILES.items():
        if wanted_model is not None and name.startswith("Inkling-2-Mini-") and name != VISION_FILE and name != wanted_model:
            continue
        p = base / name
        try:
            if p.is_file() and p.stat().st_size == size:
                found[name] = p
        except OSError:
            continue
    return found
