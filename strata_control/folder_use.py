"""Is a program of one of Strata's own folders running (an engine package, a Control bundle, a shape engine)?  Local only.

A folder in use is never removed: on Windows the removal would take its libraries from under the running engine.  (This was
dsv41_paths.in_use; it never depended on DeepSeek's engine, and every package folder kind uses it.)
"""
from __future__ import annotations

import os
from pathlib import Path


def in_use(folder) -> bool:
    """True when any running process's program or working folder lies in `folder` (psutil; without psutil: False)."""
    folder = Path(folder)
    try:
        import psutil
    except ImportError:
        return False
    here = os.path.normcase(os.path.abspath(folder)) + os.sep
    for p in psutil.process_iter(["exe", "cwd"]):
        try:
            for where in (p.info.get("exe"), p.info.get("cwd")):          # the program, or the folder a Python server runs in (a Control)
                if where and os.path.normcase(os.path.abspath(where) + os.sep).startswith(here):
                    return True
        except (psutil.Error, OSError, ValueError):
            continue
    return False
