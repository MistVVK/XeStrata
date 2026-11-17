"""Normalization: the golden files must be byte-identical on Windows, Linux and macOS and on every developer's machine.

Done here: the scratch folder, the repository folder and the home folder become <ROOT>, <REPO>, <HOME>; every backslash in a path becomes a slash; the engine
stub's executable suffix is dropped; random API keys, run ids and times become fixed words.  Nothing else is touched, so a real difference stays visible.
"""

from __future__ import annotations

import re
from pathlib import Path

KEY = re.compile(r"\b(sk-[A-Za-z0-9_-]{8,}|strata-[A-Za-z0-9]{24,})\b")


def _roots(world) -> list[tuple[str, str]]:
    from .world import ROOT
    pairs = []
    for path, word in ((world.root, "<ROOT>"), (ROOT, "<REPO>")):
        for form in {str(path), path.as_posix(), str(path).replace("\\", "/")}:
            pairs.append((form, word))
    pairs.sort(key=lambda p: -len(p[0]))                 # the longest first: <ROOT> sits inside no other, but be safe
    return pairs


def normalize(value, world):
    pairs = _roots(world)

    def text(s: str) -> str:
        for form, word in pairs:
            s = s.replace(form, word)
            s = s.replace(form.replace("/", "\\"), word)
        if "<" in s or "\\" in s:
            s = re.sub(r"(<ROOT>|<REPO>)([^\s\"',;]*)", lambda m: m.group(1) + m.group(2).replace("\\", "/"), s)
        s = re.sub(r"<ROOT>/(local/Strata|state/strata)/", "<ROOT>/<STATE>/", s)   # the per-user state folder is named by the host OS
        s = KEY.sub("<KEY>", s)
        s = re.sub(r"/strata-engine\.exe\b", "/strata-engine", s)
        return s

    def walk(v, key=None):
        if isinstance(v, dict):
            out = {}
            for k, x in v.items():
                if k in ("api_key",) and x:
                    out[k] = "<KEY>"
                elif k in ("created", "updated", "at", "stamp", "saved_at", "written_at", "ts", "time") and isinstance(x, (int, float)):
                    out[k] = "<TIME>"
                else:
                    out[k] = walk(x, k)
            return out
        if isinstance(v, list):
            return [walk(x) for x in v]
        if isinstance(v, str):
            return text(v)
        return v

    out = walk(value)
    # the run config carries the argument list three times (args, and the stamps of what the launch derived and wrote); an identical copy is written as "=args"
    cfg = out.get("config_file") if isinstance(out, dict) else None
    if isinstance(cfg, dict):
        for k in ("args_derived", "args_written"):
            if k in cfg and cfg[k] == cfg.get("args"):
                cfg[k] = "=args"
    return out
