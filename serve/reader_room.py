# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
"""serve/reader_room.py - the picture reader and a graphics card that has no room for it: what its output says, and
the words.

The GPU picture reader asks the card for its weights (the mmproj) and its work buffers when it starts.  A card that
another program has filled answers out of device memory, and the reader cannot print READY: it prints ERR on its
output, or dies, with the runtime's text in its log.  `find` reads those lines; `on_processor` and
`not_started_on_processor` are the sentences the person reads when the server reads the pictures on the processor
instead.

The figures are what can be known: the card (by the PCI address the config names it by), the reader's weights (the
mmproj file's size, what it loads) and the card's free memory read after the reader has gone.  What cannot be known is
left out.  Pure functions but for the file's size.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass

# ggml's lines (upstream's reader, and any llama.cpp-based one): the card as the reader numbers it, and what it asked
_ALLOC = re.compile(r"allocating (\d+(?:\.\d+)?) MiB on device (\d+): (?:cuda|hip)Malloc failed: out of memory", re.I)
_BUFFER = re.compile(r"failed to allocate (?:CUDA|ROCm|HIP|MUSA)(\d+) buffer of size (\d+)", re.I)
# XeStrata's own reader: the Unified Runtime / Level Zero names, and a plain "out of memory"
_GENERIC = re.compile(r"out of device memory|out_of_device_memory|out of memory|out_of_resources|out of resources|"
                      r"(?:cuda|hip)Malloc failed|failed to allocate|cannot allocate|memory allocation failed", re.I)


@dataclass(frozen=True)
class NoRoom:
    """`device`: the card's number as the reader counts it (None when the lines do not say); `need_mib`: what it asked
    for (None when the lines do not say)."""
    device: int | None = None
    need_mib: float | None = None


def find(text_or_lines) -> NoRoom | None:
    """The reader's out-of-memory lines in `text_or_lines` (a string, or a list of lines), or None.  The last
    allocation that failed is the one reported."""
    lines = text_or_lines.splitlines() if isinstance(text_or_lines, str) else [str(x) for x in text_or_lines or []]
    alloc = buffer = None
    generic = False
    for line in lines:
        m = _ALLOC.search(line)
        if m:
            alloc = NoRoom(int(m.group(2)), float(m.group(1)))
            continue
        m = _BUFFER.search(line)
        if m:
            buffer = NoRoom(int(m.group(1)), int(m.group(2)) / 1048576.0)
            continue
        if _GENERIC.search(line):
            generic = True
    return alloc or buffer or (NoRoom() if generic else None)


def weights_mib(mmproj) -> float | None:
    """The reader's weights as far as they can be known here: its mmproj file's size in MiB (it loads that file's
    tensors).  None when the file cannot be read."""
    try:
        return os.path.getsize(mmproj) / 1048576.0
    except (OSError, TypeError, ValueError):
        return None


def _gb(mib: float) -> str:
    gb = mib / 1024.0
    return f"{gb:.1f} GB" if gb >= 0.05 else "less than 0.1 GB"


def _facts(need_mib: float | None, free_mib: float | None) -> str:
    """The figures in brackets: what the reader needs and what the card has; only what is known, nothing when
    neither."""
    parts = []
    if need_mib:
        parts.append(f"it needs about {_gb(need_mib)}")
    if free_mib is not None:
        parts.append(f"{_gb(free_mib)} are free")
    return " (" + "; ".join(parts) + ")" if parts else ""


def _card(card) -> str:
    return f"card {card}" if card else "the graphics card"


def on_processor(card=None, need_mib: float | None = None, free_mib: float | None = None) -> str:
    """The server goes on and the pictures are read on the processor because the card had no room for the reader."""
    return (f"Pictures are read on the processor, which is slower: {_card(card)} has no room for the picture "
            f"reader{_facts(need_mib, free_mib)}. To read them on the card, close the other programs that use it and "
            "start again.")


def not_started_on_processor(card=None, log_path: str | None = None) -> str:
    """The server goes on and the pictures are read on the processor because the GPU reader did not start for a reason
    that is not a full card (the reason is in the engine log, which is the reader's stderr).  Without the card, or the
    log's path, those words are left out."""
    where = f" on {_card(card)}" if card else ""
    why = f" The reason is in the engine's log: {log_path}." if log_path else ""
    return f"Pictures are read on the processor, which is slower: the picture reader could not start{where}.{why}"


def start_error(room: NoRoom, said) -> str:
    """The error when the reader could not start because the card had no room for it: what it asked for, and the
    reader's own line that says so."""
    asked = (f" (it asked for {room.need_mib:.0f} MiB" + (f" on card {room.device}" if room.device is not None
                                                          else "") + ")") if room.need_mib else ""
    lines = [x.strip() for x in said if x and x.strip()]
    evidence = (next((x for x in reversed(lines) if "malloc failed" in x.lower()), None)
                or next((x for x in reversed(lines) if "allocat" in x.lower()), None)  # codespell:ignore
                or (lines[-1] if lines else ""))
    return ("the vision encoder did not start: the graphics card has no room for it" + asked +
            (". From the encoder: " + evidence[:300] if evidence else ""))
