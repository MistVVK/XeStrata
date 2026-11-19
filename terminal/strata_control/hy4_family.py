"""The Hy4-Flash family: its files (exact sizes and SHA-256), its engine component, its defaults and its sources.  Pure data and small helpers; the fit rule is hy4_fit.py.

Sizes are from the owner's R2 listing; SHA-256 of the four builds from the owner's SHA256SUMS.txt (they match the model card).  The Engram shard's and the Vision shard's
whole-file SHA-256 are not known yet (None: the download is checked by size only until HY4-CUDA's hash of both is entered; the card's Section 4 hash of the Engram shard,
176944953bf2b601645d206171964f7fa51b77522ef5dc14ae61b0eb1c803d74, is not a whole-file hash and is checked by the engine against the build's header).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from strata_control import hy4_fit

FAMILY = "hy4-flash"
TAG = "hy4-flash-"                    # run config file: strata-hy4-flash-sq-64.json (the usual strata-<tag><model>.json)
ARCH = "hy4flash"                     # the run config's "arch"; serve and Control decide the engine kind by it
SHAPE = "hy4"                         # the engine component is shape `hy4`: component:strata-hy4 (strata_control/shape_engine.py)
COMPONENT = "strata-hy4"
EXE = "strata-hy4.exe" if os.name == "nt" else "strata-hy4"
TITLE = "Tencent Hy4-Flash 220B-A12B"
BY = "Strata's own quantization of Tencent's model"
ENGRAM_FILE = "Hy4-Flash-Engram-Q8.strata"
VISION_FILE = "Hy4-Flash-Vision.strata"
DEFAULT_CONTEXT = 32768               # owner, 2026-11-19 (provisional): until 32K / 128K runs are measured
CONTEXTS = [8192, 32768, 65536, 131072]
#: The drafter (DFly2) the engine starts with when the person has not chosen: "off" or "dfly2".  One constant, open until the owner decides (the engine has no DFly2 yet).
DEFAULT_DRAFTER = "off"
DRAFTERS = ("off", "dfly2")

#: file name -> (bytes, sha256 or None)
FILES = {
    "Hy4-Flash-SQ-32.strata": (38_086_242_304, "6a21cf2a9f17732e39507e2ba476a7bab17edda9018b2a469618597ed629b1b8"),
    "Hy4-Flash-SQ-64.strata": (62_030_467_072, "4287bbbd42dbf0dc8bba3fd94142f0051dbc79be380452540726ab22a5d45e93"),
    "Hy4-Flash-SQ-96.strata": (89_577_742_336, "ba9d501350330ebe230ecc815d6e2127c43eaa1741ce9dca7576bf69ef78ba64"),
    "Hy4-Flash-SQ-128.strata": (113_654_603_776, "9bfd65640402114ac7162374c9799a444eb41b7add2a42e10457811a0d4474da"),
    ENGRAM_FILE: (67_584_905_216, None),
    VISION_FILE: (934_092_800, "ef09bceff8f01b8d226d2dc56aef09cec719b21b45783ee9b5d64e7241b72a09"),
}
BUILD_ABOUT = {
    "SQ-32": "the smallest: for 32 GB of RAM and a 12 to 16 GB card",
    "SQ-64": "the flagship: for 64 GB of RAM and an 8 to 16 GB card",
    "SQ-96": "close to the full model: for 96 GB of RAM, or a 128 GB shared-memory PC",
    "SQ-128": "the closest to the full model: for 128 GB of RAM, or a 128 GB shared-memory PC",
}


def build_file(build: str) -> str:
    return hy4_fit.BUILDS[build]["file"]


def files_of(build: str, vision: bool = False, engram_present: bool = False) -> list[str]:
    """The files one choice needs, in download order: the build, the Engram shard (unless it is there), the Vision shard when wanted."""
    out = [build_file(build)]
    if not engram_present:
        out.append(ENGRAM_FILE)
    if vision:
        out.append(VISION_FILE)
    return out


def size_of(name: str) -> int | None:
    got = FILES.get(name)
    return got[0] if got else None


def sha256_of(name: str) -> str | None:
    got = FILES.get(name)
    return got[1] if got else None


def model_key(build: str) -> str:
    """"SQ-64" -> the lower-case model id in a config's model_name: hy4-flash-sq-64."""
    return TAG + build.lower()


# ---- download sources (the owner decides which one a release ships: phaseA-tests/hy4/HY4-INTEGRATION-DESIGN.md section 4) ----

SOURCES_FILE = Path(__file__).resolve().parents[1] / "data" / "hy4-sources.json"
SOURCE_ENV = "STRATA_HY4_SOURCE"


def sources(env: dict | None = None, cfg: dict | None = None, shipped: Path | None = None) -> list[dict]:
    """The download sources in the order they are tried: the config's "hy4_source", the environment's STRATA_HY4_SOURCE (a base URL), then the shipped list.  Each is
    {"kind": "http" | "http-parts", "base": url, "files": {name: [urls]}?}.  A file nobody lists falls back to base + "/" + name (and, for "http-parts", name + ".part-aa", ...).
    Empty when nothing is configured: the family is then offered only for files already on the disk.  Never raises on a bad file: a source that cannot be read is left out."""
    env = os.environ if env is None else env
    out: list[dict] = []
    for item in ((cfg or {}).get("hy4_source"),):
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
    """The URLs of one file at one source (several for parts, joined in order by the downloader)."""
    listed = (source.get("files") or {}).get(name)
    if listed:
        return [listed] if isinstance(listed, str) else list(listed)
    base = source.get("base") or ""
    if not base:
        return []
    if source.get("kind") == "http-parts":
        return [f"{base}/{name}.part-{chr(97 + i // 26)}{chr(97 + i % 26)}" for i in range(_parts(name))]
    return [f"{base}/{name}"]


PART_BYTES = 42 * 10 ** 9              # doc 03: `split -b 42G`


def _parts(name: str) -> int:
    size = size_of(name) or 0
    return max(1, -(-size // (42 * 2 ** 30)))     # 42G in `split` is GiB


def join_parts(parts, dst, *, keep_parts: bool = False, chunk: int = 16 << 20) -> Path:
    """Join the downloaded parts of one file (in the order given) into `dst`: each part is appended and then removed, so the disk holds the file plus one part.  The result is
    written under a temporary name and moved into place when every part is in, so a stop in the middle never leaves a short `dst` that looks whole."""
    dst = Path(dst)
    tmp = dst.with_name(dst.name + ".joining")
    with open(tmp, "wb") as out:
        for part in parts:
            with open(part, "rb") as src:
                while True:
                    block = src.read(chunk)
                    if not block:
                        break
                    out.write(block)
            out.flush()
            if not keep_parts:
                try:
                    Path(part).unlink()
                    Path(str(part) + ".done").unlink(missing_ok=True)
                except OSError:
                    pass
        out.flush()
        os.fsync(out.fileno())
    os.replace(tmp, dst)
    return dst


def local_files(folder: str | os.PathLike, build: str | None = None) -> dict[str, Path]:
    """The Hy4-Flash files that are already in `folder` (by name; a size that is not the pinned one is left out): the family is offered for these even with no source."""
    found: dict[str, Path] = {}
    base = Path(folder)
    for name, (size, _sha) in FILES.items():
        if build is not None and name.startswith("Hy4-Flash-SQ-") and name != build_file(build):
            continue
        p = base / name
        try:
            if p.is_file() and p.stat().st_size == size:
                found[name] = p
        except OSError:
            continue
    return found
