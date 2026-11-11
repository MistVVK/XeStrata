"""Getting the picture files (the mmproj, about 0.9 GB) onto this computer for a model that is already installed, and nothing else.

A model whose run config has no ``vision`` section reads no pictures. The person turns "Read pictures" on (it is on by default), presses
Load, and this module does the one thing that is missing: it finds the picture file (or downloads it once, pinned, size and SHA-256
checked) and works out the run config's ``vision`` section for the engine in use. It never looks at the model's own files, never prepares
the pack or the draft layer again, never changes the data folder, and never writes the run config: the caller does (apps/host).

    pictures.fetch(config_path, ...)        the whole step; returns {"mmproj", "vision", "fetched", "note"}

The file lives in ``<data folder>/models/`` beside the model files of that data folder (the place setup uses), shared by every model of
the folder. The data folder is the one the model's own run config names (``--pack <data>/packs/<tag>``, strata_control/discovery.py
``data_dir_of``), not the app's models folder. A copy that is already on this computer is used where it is: in that data folder, in the
model's own folder, or one folder up from it (the layout of the Hugging Face repository).

Only an explicit command reaches the network (strata_control/acquisition.py). With no internet the answer is the plain sentence of
``PicturesError`` (code ``offline``) after one short probe, never a stalled bar. The apps/host caller then loads the model without
pictures and says so.

``python -c "...; pictures.main(sys.argv[1:])"`` is how the app runs it: ``fetch --config FILE --engine-dir DIR --vision yes ...``
with ``--output jsonl`` (one JSON line per progress step, a last line ``{"ok": true, "event": "result", ...}`` or ``{"ok": false, "error": ...}``).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from strata_control import acquisition, discovery, transfer, vision_reader

Progress = Callable[[str, int, int], None]    # (step in plain words, bytes done, bytes total)

HF_DEFAULT = "https://huggingface.co"
#: What a card of this kind needs for the picture reader (setup.py VISION): the images one picture becomes, and the CPU reader's thread cap.
MAX_TOKENS = {"gpu": 1024, "cpu": 300}
THREADS_CAP = 16
#: Room kept beyond the file itself: the download is written next to its final place and then renamed.
SPARE_BYTES = 256 * 1024 * 1024


class PicturesError(RuntimeError):
    """The picture files could not be had. ``code`` is for the caller, the message is plain words for the person."""

    def __init__(self, code: str, message: str, details: str = ""):
        super().__init__(message)
        self.code = code
        self.details = details                        # the technical cause, for a "Details" area; never the message


@dataclass(frozen=True)
class Projector:
    """One pinned picture-reader file: the repository revision it comes from, its exact size and SHA-256."""

    name: str
    size: int
    sha256: str
    repo: str
    revision: str

    def url(self, endpoint: str | None = None) -> str:
        base = (endpoint if endpoint is not None else (os.environ.get("HF_ENDPOINT") or "").strip().rstrip("/") or HF_DEFAULT)
        return f"{base}/{self.repo}/resolve/{self.revision}/{self.name}"


#: The same files and revisions as setup.py (ISTA_FILES, SWIFT_FILES, HF_REVISIONS); tests/terminal/test_pictures.py keeps the two the same.
QWEN = Projector("mmproj-Qwen3.8-Flash-Next-BF16.gguf", 907543008, "b1a82259702816a5330d7bd7607cd9676b11780e79ff7348c21103ff3ce49bd0",
                 "ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF", "ed59f92082b1e93c0e96d60a8b11aab089b52f09")
SWIFT = Projector("mmproj-Swift-Qwen3.8-Flash-Next-BF16.gguf", 907543520, "cd1140f4abba943ce8d9fa92fed8407e389954d26907efde7b72153c75bd8a60",
                  "ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF", "99bb8f7f95c7aa7b24a36a7786a4f657b30f5d3d")
#: The families whose weights are a GGUF shard set read by the Flash-Next engine: the original, its Coder, Unsloth's and Gyro's files all use the
#: original's picture reader. NVFP4 ready-made packs, Qwen3.6 (its own engine path) and DeepSeek (its own picture tower) are not handled here.
PROJECTOR_BY_FAMILY = {"qwen": QWEN, "coder": QWEN, "unsloth": QWEN, "gyro": QWEN, "swift": SWIFT}
FAMILIES = tuple(PROJECTOR_BY_FAMILY)

WORDS = {
    "offline": "There is no internet connection, so the picture files can't be downloaded now.",
    "disk": "There is not enough free disk space for the picture files (about 1 GB).",
    "failed": "The picture files could not be downloaded.",
    "checksum": "The downloaded picture files did not pass the safety check, so they were deleted.",
    "unsupported": "This model's picture files are not downloaded by this step.",
    "no_reader": "This engine has no picture reader.",
    "no_folder": "Strata could not tell which folder this model's files are in.",
    "no_config": "The model's settings file could not be read.",
}


# -------------------------------------------------------------------------------------------------------- what and where

def family_of(cfg: dict, gguf: Path | None, asked: str | None = None) -> str | None:
    """The family whose picture file this model uses; None for one this step does not handle. ``asked`` is the catalog's family (the app knows it);
    without it the model's own file name and name decide (Swift's files are called Swift-..., Qwen3.6's start with Qwen3.6)."""
    if asked:
        return asked.lower() if asked.lower() in PROJECTOR_BY_FAMILY else None
    arch = str(cfg.get("arch") or "").lower()
    names = [str(cfg.get("model_name") or "").lower(), (gguf.name.lower() if gguf else "")]
    if arch == "deepseek41" or any("deepseek" in n for n in names) or "strata-dsv41" in str(cfg.get("exe") or "").lower():
        return None
    if any(n.startswith("qwen3.6") or "qwen3.6-35b" in n for n in names):
        return None
    if any(n.startswith("swift") for n in names):
        return "swift"
    return "qwen"


def run_folder(foot: "discovery.Footprint") -> Path | None:
    """The data folder the model really lives in, from its own run config: the folder that holds its ``packs/<tag>`` (or its ``mtp/rt``, or the
    ``models/<tag>`` its files are in). A pack or draft layer in a folder of another shape says nothing about it."""
    def above(path: Path | None, child: str, name: str | None = None) -> Path | None:
        if path is None:
            return None
        if name is not None and path.name.lower() != name:
            return None
        return path.parent.parent if path.parent.name.lower() == child else None
    found = above(foot.pack, "packs") or above(foot.mtp, "mtp", "rt")
    if found is not None:
        return found
    for w in foot.weights:
        for parent in w.parents:
            if parent.name.lower() == "models":
                return parent.parent
    return None


def models_root(data: Path) -> Path:
    """``<data>/models`` (``Models`` where that is what the folder has, as on Linux with an older install)."""
    for name in ("models", "Models"):
        if (data / name).is_dir():
            return data / name
    return data / "models"


def places(data: Path | None, gguf: Path | None, named: str | None, projector: Projector) -> list[Path]:
    """Where a copy of the file may already be: the path the run config names, the data folder's models folder, the model's own folder and the
    folder above it (the Hugging Face repository layout keeps the picture file beside the size folders)."""
    out: list[Path] = []
    if named:
        out.append(Path(named))
    if data is not None:
        out.append(models_root(data) / projector.name)
    if gguf is not None:
        out += [gguf.parent / projector.name, gguf.parent.parent / projector.name]
    seen: set[str] = set()
    unique = []
    for p in out:
        key = os.path.normcase(str(p))
        if key not in seen:
            seen.add(key)
            unique.append(p)
    return unique


def _mark(path: Path) -> Path:
    return path.with_name(path.name + ".done")


def _marked(path: Path, projector: Projector) -> bool:
    try:
        return f"sha256 {projector.sha256}" in _mark(path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return False


def whole(path: Path, projector: Projector, report: Progress | None = None) -> bool:
    """Is this the pinned file, whole? Size first, then the finish mark setup leaves, else the hash (a few seconds; the file is not changed)."""
    try:
        if not path.is_file() or path.stat().st_size != projector.size:
            return False
        if _marked(path, projector):
            return True
        if report:
            report("Checking the picture files…", 0, projector.size)
        return transfer.sha256_file(path) == projector.sha256
    except OSError:
        return False


def find(candidates: list[Path], projector: Projector, report: Progress | None = None) -> Path | None:
    for path in candidates:
        if whole(path, projector, report):
            return path
    return None


def target_for(data: Path | None, named: str | None, projector: Projector) -> Path:
    """Where a download goes: the place the run config already names when its folder is there, else the data folder's models folder."""
    if named:
        p = Path(named)
        if p.parent.is_dir():
            return p
    if data is None:
        raise PicturesError("no_folder", WORDS["no_folder"])
    return models_root(data) / projector.name


def _room(folder: Path, need: int) -> None:
    above = folder
    while not above.exists() and above.parent != above:
        above = above.parent
    try:
        free = shutil.disk_usage(above).free
    except OSError:
        return                                            # cannot tell: go on, the download reports a real failure
    if free < need + SPARE_BYTES:
        raise PicturesError("disk", WORDS["disk"], f"{free / 1e9:.1f} GB are free in {above}")


# ----------------------------------------------------------------------------------------------------------- the reader

def _numbers(value) -> list[int]:
    if value is None or value == "":
        return []
    parts = list(value) if isinstance(value, (list, tuple, set)) else re.split(r"[;,\s]+", str(value))
    out = []
    for part in parts:
        m = re.search(r"\d+", str(part))
        if m:
            out.append(int(m.group(0)))
    return sorted(set(out))


def _build(engine_dir: Path) -> dict:
    try:
        data = json.loads((engine_dir / "BUILD.json").read_text(encoding="utf-8-sig"))
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def _reader_archs(engine_dir: Path) -> list[str]:
    for name, keys in (("BUILD.json", ("vision_archs", "archs")), ("engine-component.json", ("archs",))):
        try:
            data = json.loads((engine_dir / name).read_text(encoding="utf-8-sig"))
        except (OSError, ValueError):
            continue
        for key in keys:
            if isinstance(data, dict) and data.get(key):
                value = data[key]
                parts = list(value) if isinstance(value, (list, tuple, set)) else re.split(r"[;,\s]+", str(value))
                names = [str(p) for p in parts if str(p)]
                if names:
                    return names
    return []


def threads(logical: int | None = None) -> int:
    return max(1, min((logical if logical is not None else (os.cpu_count() or 8)) // 2, THREADS_CAP))


def reader_section(engine_dir: Path, mmproj: Path, gguf: Path, asked: str, hardware: dict | None = None, previous: dict | None = None) -> tuple[dict | None, str]:
    """The run config's ``vision`` section for this engine folder and card, and one plain note when it is not what was asked for.
    The same rule as setup (strata_control/vision_reader.py is the one place): the CPU reader under 24 GB of VRAM, the GPU reader from 24 GB; AMD and
    Intel Arc get their GPU reader with the CPU one behind it; a Mac has one program for both. (None, note) when the folder has no reader at all."""
    hw = hardware or {}
    make = str(hw.get("make") or "").lower()
    mac = make == "apple" or sys.platform == "darwin"
    windows = os.name == "nt"
    wanted = vision_reader.wanted(asked, hw.get("vram_gb"), make, unified=mac)
    if wanted == "none":
        return None, ""
    have = vision_reader.find(engine_dir, windows, mac=mac)
    arch = hw.get("arch")
    if mac:
        reader = vision_reader.choose(engine_dir, wanted, windows=False, mac=True)
    elif have["gpu"] is not None:
        reader = vision_reader.choose(engine_dir, wanted, windows=windows, gpu_arch=str(arch or ""), archs=_reader_archs(engine_dir))
    elif have["cpu"] is not None:
        mode, note = wanted, ""
        if mode == "gpu":                                        # the ready-made reader may cover fewer cards than the engine (setup.py prebuilt_vision)
            build = _build(engine_dir)
            built = _numbers(build.get("vision_archs", build.get("archs")))
            try:
                sm = int(arch)
            except (TypeError, ValueError):
                sm = None
            if built and sm is not None and sm not in built and not (build.get("ptx") and sm > max(built)):
                mode, note = "cpu", f"the picture reader has no code for your card (sm_{sm}): pictures are read on the CPU (a few seconds each)"
        reader = vision_reader.Reader(mode, have["cpu"], note=note)
    else:
        reader = vision_reader.Reader("none", note="this engine has no picture reader: pictures off")
    if reader.mode == "none" or reader.exe is None:
        return None, reader.note or WORDS["no_reader"]
    earlier = (previous or {}).get("max_tokens")
    same_device = bool((previous or {}).get("gpu")) == (reader.mode == "gpu")
    tokens = int(earlier) if isinstance(earlier, int) and earlier > 0 and same_device else MAX_TOKENS[reader.mode]
    base = {"exe": str(reader.exe), "mmproj": str(mmproj), "model": str(gguf), "gpu": reader.mode == "gpu", "max_tokens": tokens}
    section = vision_reader.vision_section(reader, base)
    if reader.mode == "cpu":
        section["threads"] = threads()
    return section, reader.note


# ------------------------------------------------------------------------------------------------------------- the step

def fetch(config_path: str | Path, *, engine_dir: str | Path, asked: str = "yes", data_dir: str | Path | None = None, family: str | None = None,
          hardware: dict | None = None, progress: Progress | None = None, cancel: threading.Event | None = None,
          endpoint: str | None = None, online: Callable[[str], bool] | None = None) -> dict:
    """Make sure this model has its picture file and say what its ``vision`` section is.

    Returns {"mmproj": path, "vision": section | None, "fetched": bool, "note": str, "data_dir": path, "family": name}. Raises PicturesError. Nothing of the
    model changes; the only file written is the picture file (and its finish mark) when it was downloaded."""
    report = progress or (lambda *_: None)
    config = Path(config_path)
    try:
        cfg = json.loads(config.read_text(encoding="utf-8-sig"))
        if not isinstance(cfg, dict):
            raise ValueError("not an object")
    except (OSError, ValueError):
        raise PicturesError("no_config", WORDS["no_config"]) from None
    foot = discovery.footprint(cfg, config)
    gguf = foot.weights[0] if foot.weights else None
    name = family_of(cfg, gguf, family)
    projector = PROJECTOR_BY_FAMILY.get(name) if name else None
    if projector is None or gguf is None:
        raise PicturesError("unsupported", WORDS["unsupported"])
    data = run_folder(foot) or (Path(data_dir) if data_dir else None)
    if data is None and gguf is not None:
        data = discovery.prepared_beside(gguf).get("data_dir")
    vision_now = cfg.get("vision") if isinstance(cfg.get("vision"), dict) else None
    named = str(vision_now["mmproj"]) if vision_now and vision_now.get("mmproj") else None
    found = find(places(data, gguf, named, projector), projector, report)
    fetched = False
    if found is None:
        out = target_for(data, named, projector)
        acquisition.require("the picture files", "Load in the app, or strata install")
        url = projector.url(endpoint)
        _room(out.parent, projector.size)
        if not (online or _online)(url):
            raise PicturesError("offline", WORDS["offline"])
        out.parent.mkdir(parents=True, exist_ok=True)
        partial = out.with_name(out.name + ".part")
        try:
            transfer.download(url, partial, size=projector.size, sha256=projector.sha256, cancel=cancel, what="the picture reader",
                              progress=lambda d, s: report("Downloading the picture files…", d, s))
        except transfer.ChecksumMismatch:
            raise PicturesError("checksum", f"{WORDS['checksum']} Try again.") from None
        except transfer.Cancelled:
            raise
        except (transfer.TransferError, OSError) as exc:
            raise PicturesError("failed", f"{WORDS['failed']} Try again; it continues where it stopped.", str(exc)) from None
        os.replace(partial, out)
        try:
            _mark(out).write_text(f"sha256 {projector.sha256}", encoding="utf-8")
        except OSError:
            pass                                                  # the mark only saves a second hashing
        found, fetched = out, True
    section, note = reader_section(Path(engine_dir), found, gguf, asked, hardware, vision_now)
    if section is None:
        raise PicturesError("no_reader", note or WORDS["no_reader"])
    report("The picture files are ready.", projector.size, projector.size)
    return {"mmproj": str(found), "vision": section, "fetched": fetched, "note": note, "data_dir": str(data) if data else None, "family": name}


def _online(url: str) -> bool:
    from strata_control import drafting_fetch                    # here: it pulls in the network code
    return drafting_fetch.online(url)


# ------------------------------------------------------------------------------------------------------------ the command

def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="pictures")
    sub = ap.add_subparsers(dest="command", required=True)
    one = sub.add_parser("fetch")
    one.add_argument("--config", required=True)
    one.add_argument("--engine-dir", required=True)
    one.add_argument("--vision", default="yes", choices=("yes", "gpu", "cpu"))
    one.add_argument("--data-dir")
    one.add_argument("--family")
    one.add_argument("--vram-gb", type=float)
    one.add_argument("--make")
    one.add_argument("--arch")
    a = ap.parse_args(argv)
    cancel = threading.Event()

    def emit(obj: dict) -> None:
        sys.stdout.write(json.dumps(obj) + "\n")
        sys.stdout.flush()

    last = {"step": "", "pct": -1}

    def show(step: str, done: int, total: int) -> None:
        pct = int(100 * done / total) if total else 0
        if step == last["step"] and pct == last["pct"]:
            return
        last["step"], last["pct"] = step, pct
        emit({"event": "progress", "step": step, "done": done, "total": total})

    hardware = {"vram_gb": a.vram_gb, "make": a.make, "arch": a.arch}
    try:
        result = fetch(a.config, engine_dir=a.engine_dir, asked=a.vision, data_dir=a.data_dir, family=a.family, hardware=hardware, progress=show, cancel=cancel)
    except PicturesError as exc:
        emit({"ok": False, "error": {"code": exc.code, "message": str(exc), "details": exc.details}})
        return 5
    except acquisition.AcquisitionRefused as exc:
        emit({"ok": False, "error": {"code": "acquisition_refused", "message": str(exc)}})
        return 5
    except transfer.Cancelled:
        emit({"ok": False, "error": {"code": "cancelled", "message": "Stopped. The part that was downloaded is kept."}})
        return 130
    emit({"ok": True, "event": "result", **result})
    return 0


if __name__ == "__main__":                                       # pragma: no cover
    sys.exit(main(sys.argv[1:]))
