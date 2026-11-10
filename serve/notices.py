# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
"""serve/notices.py - the software a server runs and its licenses (About > Licenses, GET /v1/licenses/software).

XeStrata itself (the LGPL, version 3 or later, and NOTICE's additional permission), then the parts from other projects
this build carries or is compiled with, as the build wrote them down: a package's or the Windows 7z's NOTICES.json
beside the engine (tools/package/engine_notices.py, windows_bundle.py), the launcher window's notices.json where it is
installed (tools/package/launcher_notices.py), or for a build from source the build folder's PARTS.json read with the
same list; and the Python packages, read where Python has them (requirements.txt's).  Each part says whether it is
free software and whether it is carried, or opened where the user installed it (the contrib mode's oneMKL and NVIDIA's
libraries); its license texts are shown whole."""
from __future__ import annotations

import importlib.metadata
import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _read(p: Path) -> str | None:
    try:
        return p.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None


def _build_record(engine_dir: Path) -> dict:
    try:
        return json.loads((engine_dir / "BUILD.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def xestrata(engine_dir: Path | None, version: str) -> dict:
    """XeStrata's own row: COPYING.LESSER, COPYING and NOTICE from the program's folder, a package's documents or the
    7z's top."""
    rec = _build_record(engine_dir) if engine_dir else {}
    places = [ROOT, ROOT.parent]
    if rec.get("package"):
        places.append(Path("/usr/share/doc") / rec["package"])
    texts = []
    for name in ("NOTICE", "COPYING.LESSER", "COPYING"):
        got = next((t for d in places if (t := _read(d / name)) is not None), None)
        if got is not None:
            texts.append({"name": name, "text": got})
    return {"name": "XeStrata", "version": version, "license": "LGPL-3.0-or-later, with NOTICE's additional permission",
            "source": "https://github.com/MistVVK/XeStrata", "free": True, "bundled": True, "texts": texts}


def engine_parts(engine_dir: Path | None) -> list[dict]:
    """The build's parts: NOTICES.json beside the engine, else (a build from source) its PARTS.json read with
    tools/package/engine_notices.py; [] when neither is there."""
    if engine_dir is None:
        return []
    try:
        return json.loads((engine_dir / "NOTICES.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        pass
    if not (engine_dir / "PARTS.json").is_file() or not (ROOT / "tools" / "package" / "engine_notices.py").is_file():
        return []
    sys.path.insert(0, str(ROOT / "tools" / "package"))
    try:
        import engine_notices
        return engine_notices.build_notices(ROOT, engine_dir)
    except SystemExit:          # a text the build should have and has not: shown without the parts' list
        return []
    finally:
        sys.path.remove(str(ROOT / "tools" / "package"))


def launcher_parts() -> list[dict]:
    """The launcher window's parts where its package is installed (Linux; the 7z's are in its NOTICES.json)."""
    for p in (Path("/usr/share/doc/xestrata-gui/notices.json"),):
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
    return []


def _requirements() -> list[str]:
    names = []
    for line in (_read(ROOT / "requirements.txt") or "").splitlines():
        m = re.match(r"\s*([A-Za-z0-9_.-]+)\s*==", line)
        if m and m.group(1).lower() not in names:
            names.append(m.group(1).lower())
    return names


def python_parts() -> list[dict]:
    """The Python packages the program uses (requirements.txt), as this Python has them: their license field and
    the license files their distribution carries."""
    rows = []
    for name in _requirements():
        try:
            dist = importlib.metadata.distribution(name)
        except importlib.metadata.PackageNotFoundError:
            continue
        meta = dist.metadata
        texts = []
        for f in dist.files or ():
            parts = Path(str(f)).parts
            named = parts and parts[-1].upper().startswith(("LICEN", "COPYING", "NOTICE", "AUTHORS"))
            if named or "licenses" in parts:
                if (t := _read(Path(str(dist.locate_file(f))))) is not None:
                    texts.append({"name": "/".join(parts[1:]) if len(parts) > 1 else parts[-1], "text": t})
        lic = meta.get("License-Expression") or meta.get("License") or ""
        if len(lic) > 200 or "\n" in lic:   # a whole text in the field: it is among the texts then
            texts.append({"name": "License (the package's metadata)", "text": lic})
            lic = ""
        classifiers = [c.split(" :: ")[-1] for c in meta.get_all("Classifier") or () if c.startswith("License ::")]
        rows.append({"name": f"{meta['Name']} (Python package)", "version": dist.version,
                     "license": lic or ", ".join(classifiers) or "its license text",
                     "source": meta.get("Home-page") or f"https://pypi.org/project/{meta['Name']}/",
                     "free": True, "bundled": True, "python": True, "texts": texts})
    return rows


def software(engine_exe: str | None, version: str) -> dict:
    """GET /v1/licenses/software's answer: XeStrata, the free parts and the non-free ones, each with its texts."""
    engine_dir = Path(engine_exe).resolve().parent if engine_exe and os.path.exists(engine_exe) else None
    parts = [*engine_parts(engine_dir), *launcher_parts(), *python_parts()]
    rec = _build_record(engine_dir) if engine_dir else {}
    return {"xestrata": xestrata(engine_dir, version),
            "license_mode": rec.get("license") or (_parts_mode(engine_dir) if engine_dir else None),
            "free": [p for p in parts if p.get("free", True)],
            "nonfree": [p for p in parts if not p.get("free", True)]}


def _parts_mode(engine_dir: Path) -> str | None:
    try:
        return json.loads((engine_dir / "PARTS.json").read_text(encoding="utf-8")).get("license")
    except (OSError, ValueError):
        return None
