# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Shared streaming and SHA-256 primitives for downloads."""
import hashlib
import os
import re
from pathlib import Path

HF_DEFAULT = "https://huggingface.co"


def hf_endpoint() -> str:
    return (os.environ.get("HF_ENDPOINT") or "").strip().rstrip("/") or HF_DEFAULT


def range_matches(value: str | None, start: int, end: int | None = None, total: int | None = None) -> bool:
    """Check a Content-Range; an unspecified total permits the MTP reader's unknown total."""
    raw = (value or "").strip()
    if total is None and end is not None:
        return raw.startswith(f"bytes {start}-{end}/")
    match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", raw)
    return bool(match and int(match[1]) == start and (end is None or int(match[2]) == end)
                and (total is None or int(match[3]) == total))


def resumed_at(response, have: int, total: int = 0, *, strict: bool = True) -> int:
    """The response's write offset: a full response restarts; strict callers reject an incorrect range."""
    if have:
        matches = not strict or range_matches(response.headers.get("Content-Range"), have, total=total or None)
        if response.status == 206 and matches:
            return have
        if response.status != 200 and strict:
            raise OSError(f"the server answered another range than bytes {have}- (HTTP {response.status})")
        return 0
    if strict and response.status not in (200, 206):
        raise OSError(f"HTTP {response.status}")
    return 0


def write_done(path, text: str) -> None:
    path = Path(path)
    path.with_name(path.name + ".done").write_text(text, encoding="utf-8")


def sha256_file(path, chunk: int = 16 << 20, progress=None) -> str:
    digest = hashlib.sha256()
    seen = 0
    with open(path, "rb") as f:
        while data := f.read(chunk):
            digest.update(data)
            seen += len(data)
            if progress is not None:
                progress(seen)
    return digest.hexdigest()


def copy_stream(source, out, *, have: int = 0, chunk: int = 1 << 20, limit: int = 0, progress=None, stop=None) -> int:
    while stop is None or not stop():
        data = source.read(chunk)
        if not data:
            break
        if limit and have + len(data) > limit:
            raise OSError("the server sent more bytes than the file has")
        out.write(data)
        have += len(data)
        if progress is not None:
            progress(have)
    return have
