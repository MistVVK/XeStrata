# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Shared streaming and SHA-256 primitives for downloads."""
import hashlib
import os
from pathlib import Path

HF_DEFAULT = "https://huggingface.co"


def hf_endpoint() -> str:
    return (os.environ.get("HF_ENDPOINT") or "").strip().rstrip("/") or HF_DEFAULT


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
