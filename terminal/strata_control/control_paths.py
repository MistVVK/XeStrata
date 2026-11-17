"""Paths a run config names inside a Control folder that no longer exists, pointed at this Control.

Setup writes absolute paths into a run config (``cwd``, ``--expert-profile``) that lie inside the Control it ran from. That folder can go away:
the Linux AppImage mounts its bundled Control at ``/tmp/.mount_<random>/usr/lib/Strata/control``, a new name at every start, and an app
reinstall or a Control update moves it too. A path that ends up inside a folder called ``control`` which no longer exists is pointed at the
same file in this Control when the file is there; any other value is left as it is. The app host does the same for the saved files
(apps/host/server/native-control.mjs ``repairControlPaths``); this is for a start that reads a run config itself (the update's test start).
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

_INSIDE = re.compile(r"^(.+?[/\\]control)(?=[/\\]|$)(.*)$", re.IGNORECASE)


def this_control() -> Path:
    """The Control folder this code runs from (the folder that holds ``strata_control``)."""
    return Path(__file__).resolve().parent.parent


def _same(a: str, b: Path) -> bool:
    try:
        return os.path.normcase(os.path.abspath(a)) == os.path.normcase(str(b.resolve()))
    except OSError:
        return False


def repoint_value(value: Any, here: Path | None = None) -> Any:
    """``value`` pointed at ``here`` (default: this Control) when it names a missing path inside another ``control`` folder; else unchanged."""
    if not isinstance(value, str):
        return value
    m = _INSIDE.match(value)
    if not m:
        return value
    here = here or this_control()
    try:
        if os.path.exists(value) or _same(m.group(1), here):
            return value
    except OSError:
        return value
    rest = m.group(2).lstrip("/\\")
    candidate = here.joinpath(*re.split(r"[/\\]", rest)) if rest else here
    return str(candidate) if candidate.exists() else value


def repoint(cfg: dict, here: Path | None = None) -> dict:
    """A copy of the run config with ``cwd`` and every ``args`` value repointed (see repoint_value)."""
    out = dict(cfg)
    if "cwd" in out:
        out["cwd"] = repoint_value(out["cwd"], here)
    if isinstance(out.get("args"), list):
        out["args"] = [repoint_value(a, here) for a in out["args"]]
    return out
