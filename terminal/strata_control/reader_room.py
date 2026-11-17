"""The picture reader and a graphics card that has no room for it: what its output says, and the words.

The GPU picture reader (strata-vision, llama.cpp's mtmd) asks the card for its weights and work buffers when it starts.  A card that another program
has filled answers "cudaMalloc failed: out of memory" (the same text on AMD cards, whose runtime ggml names the same way) and ggml aborts: the reader cannot
print an answer of its own.  Its lines end up in the engine log (its stderr is the log):

    ggml_backend_cuda_buffer_type_alloc_buffer: allocating 865.48 MiB on device 0: cudaMalloc failed: out of memory
    alloc_tensor_range: failed to allocate CUDA0 buffer of size 907524736
    ggml-backend.cpp:188: GGML_ASSERT(buffer) failed

`find` reads them (card number inside the reader's own numbering, and how much it asked for); the two sentences below are what the person reads, in the
Terminal's launch summary (the reader moved to the processor) and in the error (the launch ended because of it).  One wording for the server and the
Terminal.  Pure functions, no I/O.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

_ALLOC = re.compile(r"allocating (\d+(?:\.\d+)?) MiB on device (\d+): (?:cuda|hip)Malloc failed: out of memory", re.I)
_BUFFER = re.compile(r"failed to allocate (?:CUDA|ROCm|HIP|MUSA)(\d+) buffer of size (\d+)", re.I)
_GENERIC = re.compile(r"(?:cuda|hip)Malloc failed: out of memory", re.I)


#: What the GPU picture reader takes from the card in all, at the default image limit of 1,024 tokens: 1,342 MiB measured on an RTX A6000 and an R9700, 1,536 the
#: figure the server plans with (serve/server.py READER_NEEDS_MIB).  Above 1,024 tokens the picture is bigger and so are the reader's work buffers: at 16,384
#: tokens the largest peak measured for mtmd's GPU reader was 3,674 MiB (P100, Qwen3.8), 3,264-3,656 MiB on the other cards; between the two the need is taken to
#: grow in a straight line with the limit (an estimate, said as "about").  A limit of 0 means no limit: the largest the Terminal offers, 16,384.
READER_NEEDS_MIB = 1536
READER_NEEDS_AT_LIMIT_MIB = 3674
READER_LIMIT_DEFAULT = 1024
READER_LIMIT_LARGEST = 16384


def reader_needs_mib(max_tokens: int | None = None) -> float:
    """The reader's whole need of the card, in MiB, at an image limit of max_tokens tokens (None: the default limit; 0: no limit)."""
    limit = READER_LIMIT_DEFAULT if max_tokens is None else int(max_tokens)
    if limit == 0:
        limit = READER_LIMIT_LARGEST
    limit = min(max(limit, READER_LIMIT_DEFAULT), READER_LIMIT_LARGEST)
    return READER_NEEDS_MIB + (limit - READER_LIMIT_DEFAULT) * (READER_NEEDS_AT_LIMIT_MIB - READER_NEEDS_MIB) / (READER_LIMIT_LARGEST - READER_LIMIT_DEFAULT)


def whole_need_mib(failed_mib: float | None, max_tokens: int | None = None) -> float:
    """What to tell the person the reader needs: the larger of the one allocation that failed (the line says it) and the reader's known need at its limit.  The
    failed allocation alone is not the need: a person who frees that much and nothing more still fails on the next one."""
    return max(float(failed_mib or 0.0), reader_needs_mib(max_tokens))


@dataclass(frozen=True)
class NoRoom:
    """`device`: the card's number as the reader counts (the cards its environment shows it), None when the lines do not say; `need_mib`: what it asked for."""
    device: int | None = None
    need_mib: float | None = None


def find(text_or_lines) -> NoRoom | None:
    """The reader's out-of-memory lines in `text_or_lines` (a string, or a list of lines), or None.  The last allocation that failed is the one reported."""
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


def physical_card(device: int | None, visible) -> int | None:
    """The card's number as nvidia-smi counts, from the reader's number and the list of cards its environment shows it (CUDA_VISIBLE_DEVICES, the engine's
    card order); the reader's own number when it sees them all.  None when it cannot be said."""
    if device is None:
        return None
    if isinstance(visible, str):
        visible = [x for x in re.split(r"[,\s]+", visible.strip()) if x]
    if visible:
        try:
            return int(str(visible[device]).strip())
        except (IndexError, ValueError):
            return None
    return device


def _gb(mib: float) -> str:
    gb = mib / 1024.0
    return f"{gb:.1f} GB" if gb >= 0.05 else "less than 0.1 GB"


def _facts(need_mib: float | None, free_mib: float | None) -> str:
    """The figures in brackets: what the reader needs and what the card has; only what is known, nothing when neither is."""
    parts = []
    if need_mib:
        parts.append(f"it needs about {_gb(need_mib)}")
    if free_mib is not None:
        parts.append(f"{_gb(free_mib)} are free")
    return " (" + "; ".join(parts) + ")" if parts else ""


def _card(card: int | None) -> str:
    return f"graphics card {card}" if card is not None else "the graphics card"


def on_processor(card: int | None = None, need_mib: float | None = None, free_mib: float | None = None) -> str:
    """The launch goes on and the pictures are read on the processor (the Terminal's summary line)."""
    return (f"Pictures are read on the processor, which is slower: {_card(card)} has no room for the picture reader{_facts(need_mib, free_mib)}. "
            "To read them on the card, close the other programs that use it and start again.")


def not_started_on_processor(card: int | None = None, log_path: str | None = None) -> str:
    """The launch goes on and the pictures are read on the processor because the GPU reader did not start for a reason that is not a full card (the summary
    line; the reason is in the engine log, which is the reader's stderr).  Without the card's number, or the log's path, those words are left out."""
    where = f" on graphics card {card}" if card is not None else ""
    why = f" The reason is in the engine's log: {log_path}." if log_path else ""
    return f"Pictures are read on the processor, which is slower: the picture reader could not start{where}.{why}"


def could_not_start(card: int | None = None, need_mib: float | None = None, free_mib: float | None = None) -> tuple[str, str]:
    """(what happened, the next step) when the launch ends because the reader could not start."""
    return (f"The picture reader could not start: {_card(card)} has no room for it{_facts(need_mib, free_mib)}.",
            "Close the other programs that use the card and start again, or add --vision off to start without pictures.")
