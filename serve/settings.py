# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Read and atomically write the user's settings, with the caller's read policy."""
import json
import os
from pathlib import Path


def load(path: str | Path, *, strict: bool = True):
    path = Path(path)
    try:
        text = path.read_text(encoding="utf-8-sig")   # a BOM is valid JSON input for both readers (Windows Notepad)
    except FileNotFoundError:
        return {}
    data = json.loads(text)
    if strict and not isinstance(data, dict):
        raise ValueError(f"{path.name} is not a JSON object")
    return data


def save(path: str | Path, data: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with os.fdopen(os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600), "w", encoding="utf-8") as f:
        f.write(json.dumps(data, indent=1))
    os.chmod(tmp, 0o600)
    os.replace(tmp, path)
