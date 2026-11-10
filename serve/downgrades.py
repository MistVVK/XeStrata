# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
"""serve/downgrades.py - the slower paths the engines fell back to (src/platform/downgrade.cpp: a library or module the
build and the GPU promise that could not be taken), as the engines report them (`INFO downgrade=KEY`, once a key each):
the server says each in its log once and lists them in GET /v1/status ("downgrades"), so a slow fallback is not silent.
"""
from __future__ import annotations

import threading

_lock = threading.Lock()
_seen: dict[str, str] = {}   # key -> the engine that reported it first


def note(key: str, engine: str) -> None:
    """A downgrade an engine reported: said in the log the first time."""
    with _lock:
        if not key or key in _seen:
            return
        _seen[key] = engine
    print(f"[xestrata] WARNING: the {engine} fell back to a slower path ({key}): its log says why", flush=True)


def from_info_line(line: str, engine: str) -> bool:
    """`INFO downgrade=KEY` noted; whether the line was one."""
    if not line.startswith("INFO downgrade="):
        return False
    note(line.split("=", 1)[1].strip(), engine)
    return True


def snapshot() -> dict[str, str]:
    with _lock:
        return dict(_seen)
