"""Paths in a saved run config that name a version folder of Strata's own components, and what to do when an update has removed that folder.

Setup writes absolute paths into the run config: ``cwd`` and ``--expert-profile`` into the Control bundle that ran it
(``<components>/control/<version>/...``), the picture reader and library folders into an engine package
(``<components>/engine/<version>-<platform>-<backend>/...``).  An update puts the new version
beside the old one and removes the versions older than the one before, so the second update after a model was set up deleted the folders its config
names: the model showed as "not installed" and a start failed (#4891; "Qwen breaks each time Strata is updated").

One resolver, used by both sides:

* ``resolve_config(cfg)`` (pure; ``profiles.read_run_config`` calls it, so the doctor, the model list, the launch plan and the runtime host see the
  current folders without a file being written);
* ``repair(configs)`` (runtime.start calls it for the profile being started and every saved profile): the same, written once into the file, one plain
  line for each file it changed.

Only paths in Strata's own folders are touched (``components/control``, ``components/engine``), and only when the version folder they
name is GONE; a path the person typed themselves, a model or pack folder, or a version folder that still exists is left exactly as it is.  A path whose
file is nowhere in the current version either is not guessed: ``unresolved`` names it.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

_SEP = r"[\\/]"
_COMPONENT = re.compile(rf"^(?P<head>.*?{_SEP}components{_SEP})(?P<kind>control|engine){_SEP}(?P<ver>[^\\/]+)(?P<rest>(?:{_SEP}.*)?)$", re.S)
_VERSION = re.compile(r"^\d+(?:\.\d+)+")
SKIP_KEYS = ("log",)                      # paths the engine writes, not reads
SKIP_FLAGS = ("--expert-profile-save",)


def _key(name: str) -> tuple:
    head = str(name).split("-")[0]
    return tuple(int(x) for x in head.split(".") if x.isdigit()), str(name)


def _kind_of(name: str) -> str:
    return str(name).split("-", 1)[1] if "-" in str(name) else ""


def _newest(folder: Path, kind: str | None = None) -> Path | None:
    try:
        found = [d for d in folder.iterdir() if d.is_dir() and _VERSION.match(d.name) and (kind is None or _kind_of(d.name) == kind)]
    except OSError:
        return None
    return max(found, key=lambda d: _key(d.name), default=None)


def _active_control(control: Path) -> Path | None:
    try:
        data = json.loads((control / "active.json").read_text(encoding="utf-8-sig"))
        folder = control / str(data["folder"])
        if folder.is_dir():
            return folder
    except (OSError, ValueError, KeyError, TypeError):
        pass
    return _newest(control)


def _components() -> Path:
    from strata_control import engine_component as ec
    return ec.components_dir()


def resolve_path(value: str) -> tuple[str, bool] | None:
    """For a string that names a file in a version folder of Strata's own that is gone: (the same file in the current version, True) when it is there,
    (value, False) when it is nowhere; None when the string is not such a path or its version folder still exists (nothing to do)."""
    if not isinstance(value, str) or len(value) < 8 or "\n" in value:
        return None
    m = _COMPONENT.match(value)
    if not m:
        return None
    old = Path(m["head"] + m["kind"] + os.sep + m["ver"])
    if old.exists() or not _VERSION.match(m["ver"]):
        return None
    root = _components() / m["kind"]
    current = _active_control(root) if m["kind"] == "control" else _newest(root, _kind_of(m["ver"]))
    rest = m["rest"].lstrip("\\/")
    if current is None:
        return value, False
    new = current / rest if rest else current
    return (str(new), True) if new.exists() else (value, False)


def _walk(obj, found: list, lost: list, key: str = ""):
    if isinstance(obj, str):
        if key in SKIP_KEYS:
            return obj
        res = resolve_path(obj)
        if res is None:
            return obj
        (found if res[1] else lost).append((obj, res[0]))
        return res[0]
    if isinstance(obj, list):
        out, skip = [], False
        for item in obj:
            out.append(item if skip else _walk(item, found, lost, key))
            skip = isinstance(item, str) and item in SKIP_FLAGS
        return out
    if isinstance(obj, dict):
        return {k: _walk(v, found, lost, str(k)) for k, v in obj.items()}
    return obj


def resolve_config(cfg: dict) -> tuple[dict, list, list]:
    """(the config with every path of a removed version folder pointed at the current one, [(old, new)], [(old, "") that exist nowhere])."""
    found: list = []
    lost: list = []
    return _walk(cfg, found, lost), found, [(old, "") for old, _ in lost]


def repair(configs) -> tuple[list[str], list[str]]:
    """For a START: every run config in `configs` is resolved and, when a path changed, written once (atomically, nothing else changed).
    Returns (one plain line for each file changed, one plain line for each path that exists nowhere).  Never raises."""
    notes: list[str] = []
    problems: list[str] = []
    seen = set()
    for c in configs:
        c = Path(c)
        if c in seen:
            continue
        seen.add(c)
        try:
            cfg = json.loads(c.read_text(encoding="utf-8-sig"))
        except (OSError, ValueError):
            continue
        if not isinstance(cfg, dict):
            continue
        from strata_control import retired
        if retired.config_reason(cfg) or retired.config_file_reason(c):
            continue                                     # a DeepSeek config is never written: this Strata does not run it (retired.py)
        new, found, lost = resolve_config(cfg)
        for old, _ in lost:
            problems.append(f"{c.name}: {old} was removed by an update and no file of that name is in the current version; run setup again for this model")
        if not found:
            continue
        try:
            from strata_control import profiles
            profiles._atomic(c, json.dumps(new, indent=2).encode("utf-8"))
        except (OSError, ValueError):
            continue
        notes.append(f"{c.name}: {len(found)} path{'s' if len(found) != 1 else ''} named folders of an earlier Strata version that an update removed ("
                     + ", ".join(sorted({Path(o).name or o for o, _ in found})[:3]) + "); they now name the current version.")
    return notes, problems
