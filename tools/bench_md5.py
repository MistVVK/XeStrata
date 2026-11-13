# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
"""The answers' md5: the watch of a change that must not move an engine's answer, one fixed text and setting a cell.

A speed idea is measured by ab_pair.py; a change that has to leave the answers as they are is measured here, by the
md5 of what each build answers.  A cell is one model at one length (an image model at one size): the fixed prompt of
that length, the settings that decide the answer (greedy, --adapt-every 0, --pcie-frac 0), and the engine's answer:
an LLM's `output  : ...` / `tokens: ...` line, an image model's PNG file.  md5_watch.py runs the cells and keeps a
record; md5_judge.py reads two records and lists the cells whose md5 differs, and for those alone gives the KL of the
logits (an LLM's) or the pixels' difference (an image's), against the floor two runs of one build give.

The md5 is of the answer's line as the engine printed it, with its newline, and of the image file's bytes: the tables
the unify2 records published (V1's and S's `llm/*.out`, their s-sum.py) are these md5s, so a new record is compared
with them as it is.

A record folder holds md5.json (the rows: kind, model, tier, round, md5, bits, file, rows, note), meta.json (what
made them) and the runs' files: answers/<cell>.out and logs/<cell>.err (an LLM's), logits/<cell>.f32 (the logits its
answer was chosen from: 32 rows of the vocabulary, ~32 MB a cell), images/<cell>.png (an image's).  A row's `bits` is
the logits' md5: two runs whose answers are the same gave the same arithmetic, bit for bit, only when it is equal
(the engine's own --window-hashes rule), so a cell that answers the same from different bits is one to read too.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

NEW = 32                    # tokens a cell follows (the logits' rows: the KL's)
KINDS = ("llm", "image")
_OUTPUT = re.compile(rb"^output\s*:")     # the generate engine's answer line
_TOKENS = re.compile(rb"^tokens:")        # the plain engine's


# ------------------------------------------------------------------------------------------------ reading an answer
def answer_lines(out: bytes) -> list[bytes]:
    """The lines of an engine's stdout that carry its answer, each with its newline: the generate engine's
    (`output  : 40 3172 ...`) when it printed one, else the plain engine's (`tokens: 40,3172,...`)."""
    lines = [ln + b"\n" for ln in out.split(b"\n") if ln]
    for pattern in (_OUTPUT, _TOKENS):
        found = [ln for ln in lines if pattern.match(ln)]
        if found:
            return found
    return []


def answer_md5(out: bytes) -> str | None:
    """The md5 of an engine's answer, as its record publishes it; None when the run printed no answer."""
    lines = answer_lines(out)
    return hashlib.md5(b"".join(lines)).hexdigest() if lines else None


def answer_tokens(out: bytes) -> list[int] | None:
    """The answer's token ids, from its first line (a space or a comma between them, as each engine prints them)."""
    lines = answer_lines(out)
    if not lines:
        return None
    body = lines[0].decode("utf-8", "replace").split(":", 1)[1]
    return [int(x) for x in re.split(r"[,\s]+", body.strip()) if x]


def answer_file(path: Path) -> tuple[str | None, list[int] | None]:
    """A run's stdout file -> (its answer's md5, its ids)."""
    try:
        out = path.read_bytes()
    except OSError:
        return None, None
    return answer_md5(out), answer_tokens(out)


# ------------------------------------------------------------------------------------------------ the record
def write(out: Path, rows: list[dict], meta: dict) -> None:
    """OUT/md5.json (the rows), OUT/md5.txt (a cell a line, for reading) and OUT/meta.json (what made them)."""
    out.mkdir(parents=True, exist_ok=True)
    (out / "md5.json").write_text(json.dumps(rows, indent=1) + "\n", encoding="utf-8")
    (out / "meta.json").write_text(json.dumps(meta, indent=1) + "\n", encoding="utf-8")
    (out / "md5.txt").write_text(table(rows) + "\n", encoding="utf-8")


def read(folder: Path) -> tuple[list[dict], dict]:
    """A record's (rows, meta)."""
    try:
        rows = json.loads((folder / "md5.json").read_text(encoding="utf-8"))
        meta = json.loads((folder / "meta.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as err:
        raise SystemExit(f"{folder}: not a record of the answers' md5 ({err})") from None
    return rows, meta


def table(rows: Iterable[dict]) -> str:
    """A record's cells, one line a cell: its kind, model and tier, and each round's md5 (8 characters) in order."""
    lines = [f"{'kind':5s} {'model':20s} {'tier':5s} md5, one a round"]
    for key, cell in cells(list(rows)).items():
        md5s = " ".join((r.get("md5") or "none")[:8] for r in cell)
        lines.append(f"{key[0]:5s} {key[1]:20s} {key[2]:5s} {md5s}")
    return "\n".join(lines)


def cells(rows: list[dict]) -> dict[tuple[str, str, str], list[dict]]:
    """(kind, model, tier) -> the cell's rows, its rounds in order."""
    out: dict[tuple[str, str, str], list[dict]] = {}
    for r in rows:
        out.setdefault((r["kind"], r["model"], r["tier"]), []).append(r)
    for cell in out.values():
        cell.sort(key=lambda r: r.get("round") or 0)
    return out


def majority(cell: list[dict]) -> dict:
    """The row of the answer most of the cell's rounds gave (the first of them): the answer one build gives, when it
    answers differently from run to run."""
    counts: dict[str, int] = {}
    for r in cell:
        if r.get("md5"):
            counts[r["md5"]] = counts.get(r["md5"], 0) + 1
    if not counts:
        return cell[0]
    best = max(counts.values())
    return next(r for r in cell if r.get("md5") and counts[r["md5"]] == best)


def verdict(old: list[dict] | None, new: list[dict] | None) -> str:
    """How a cell's answer moved from the previous record to this one: "same" (the same md5s), "mixed" (they meet and
    each record has one the other lacks: the answer moves from run to run), "changed" (no md5 in common), "failed"
    (this record's rounds printed no answer), "new" (the previous record's did not) or "not compared" (a record has no
    such cell)."""
    if old is None or new is None:
        return "not compared"
    o = {r["md5"] for r in old if r.get("md5")}
    n = {r["md5"] for r in new if r.get("md5")}
    if not n:
        return "failed" if o else "not compared"
    if not o:
        return "new"
    if o == n:
        return "same"
    return "mixed" if o & n else "changed"


def bits_moved(old: list[dict], new: list[dict]) -> bool:
    """Whether two records' answers were chosen from other logits, bit for bit: the answers most rounds gave have other
    `bits` (False when a record kept no dumps, and for a cell whose answers differ: `verdict` says that)."""
    one, other = majority(new), majority(old)
    return bool(one.get("bits") and other.get("bits") and one["bits"] != other["bits"])


def shared_rows(one: list[int] | None, other: list[int] | None, rows: int) -> int:
    """The rows two answers read the same text at: up to their first differing token, that token's row included (the
    one its pick was made from); every row when the answers are the same or one of them is unknown."""
    if not one or not other:
        return rows
    for i, (a, b) in enumerate(zip(one, other)):
        if a != b:
            return i + 1
    return rows


# ------------------------------------------------------------------------------------------------ the measures
def strip_header(v: np.ndarray) -> tuple[np.ndarray, int]:
    """A dump without its header, and the rows it says it has (0: the dump has none): the generate engine writes
    [vocab, rows] in 8 bytes, the plain engine writes the rows alone."""
    import numpy as np
    if v.size >= 2:
        head = v[:2].view(np.int32)
        if 0 < int(head[0]) <= (1 << 20) and int(head[1]) > 0 and v.size - 2 == int(head[0]) * int(head[1]):
            return v[2:], int(head[1])
    return v, 0


def kl(one: Path, other: Path, rows_one: int = NEW, rows_other: int = NEW, rows: int = NEW) -> dict:
    """Two --dump-logits files compared over their first `rows` rows: the mean and largest KL divergence (one ||
    other), the share of rows whose most likely token is the same, and whether every row is equal bit for bit.
    `rows_one`/`rows_other`: the tokens each run followed; a dump's header, when it has one, says its own rows."""
    import numpy as np
    va, na = strip_header(np.fromfile(one, dtype=np.float32))
    vb, nb = strip_header(np.fromfile(other, dtype=np.float32))
    rows_one, rows_other = na or rows_one, nb or rows_other
    if rows_one < 1 or rows_other < 1 or va.size % rows_one or vb.size % rows_other:
        return {"error": f"{one.name} and {other.name}: not whole rows ({va.size} and {vb.size} floats)"}
    vocab_one, vocab_other = va.size // rows_one, vb.size // rows_other
    if vocab_one != vocab_other:
        return {"error": f"{one.name} and {other.name}: vocabularies {vocab_one} and {vocab_other}"}
    rows = max(1, min(rows, rows_one, rows_other))
    x = va[:rows * vocab_one].reshape(rows, vocab_one).astype(np.float64)
    y = vb[:rows * vocab_other].reshape(rows, vocab_other).astype(np.float64)
    lx = x - x.max(axis=1, keepdims=True)
    lx -= np.log(np.exp(lx).sum(axis=1, keepdims=True))
    ly = y - y.max(axis=1, keepdims=True)
    ly -= np.log(np.exp(ly).sum(axis=1, keepdims=True))
    k = (np.exp(lx) * (lx - ly)).sum(axis=1)
    return {"rows": rows, "vocab": vocab_one, "mean": float(k.mean()), "max": float(k.max()),
            "top1": float((x.argmax(axis=1) == y.argmax(axis=1)).mean()), "bitwise": bool((x == y).all())}


def pixels(one: Path, other: Path) -> dict:
    """The largest and mean absolute difference over the channels (0..255, four a pixel: RGBA), the share of the
    channels that differ at all, and the PSNR in dB (infinite when every value is the same)."""
    import numpy as np
    from PIL import Image
    try:
        a = np.asarray(Image.open(one).convert("RGBA"), dtype=np.int16)
        b = np.asarray(Image.open(other).convert("RGBA"), dtype=np.int16)
    except (OSError, ValueError) as err:
        return {"error": str(err)}
    if a.shape != b.shape:
        return {"error": f"{one.name} and {other.name}: {a.shape[1]}x{a.shape[0]} and {b.shape[1]}x{b.shape[0]}"}
    d = np.abs(a - b)
    mse = float((d.astype(np.float64) ** 2).mean())
    return {"size": f"{a.shape[1]}x{a.shape[0]}", "max": int(d.max()), "mean": float(d.mean()),
            "differ": float((d > 0).mean()), "psnr": math.inf if mse == 0 else 10 * math.log10(255.0 ** 2 / mse)}
