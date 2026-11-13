# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
"""The answers' md5 of one build against the previous record's: which cells moved, and how far (tools/bench_md5.py).

    python .claude/skills/upstream-sync/scripts/md5_judge.py OUT --prev DIR

A cell passes as `same` when its md5s are the same and the logits its answer was chosen from are the same bit for bit
(a row's `bits`).  `bits` is a cell that answered the same from other logits: the arithmetic moved without moving the
answer, which an exact change must not do either.  `mixed` is a cell whose answers meet but not in every round (the
engine answers differently from run to run: its own rounds are the floor such a cell is read against); `changed` is a
cell with no md5 in common; `failed` is one that printed no answer where the previous record's did; `new` and
`not compared` are the cells one record lacks.

For a cell that moved, and only for those: an LLM's KL divergence of the previous record's logits against this one's
(mean and largest, the share of rows whose most likely token is the same: the direction the records' kl.py compares a
before with an after), over the rows both read the same text at and with the floor its own rounds give; an image's
pixels' difference (the largest and mean difference, the share of the values that differ, the PSNR).  Out: a line a
cell and the moved ones' measures; the last line is "ALL SAME" (exit 0), "MIXED" (exit 0: the records' answers meet,
but some cell answers differently from run to run) or "CHANGED <n>" (exit 1, a cell whose answer or bits moved).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "tools"))
import bench_md5 as M  # noqa: E402

args = sys.argv[1:]
if len(args) != 3 or args[1] != "--prev":
    sys.exit(f"usage: {sys.argv[0]} OUT --prev DIR")
out, prev = Path(args[0]), Path(args[2])
rows, meta = M.read(out)
old_rows, old_meta = M.read(prev)


def md5s(cell: list[dict] | None) -> str:
    return " ".join((r.get("md5") or "none")[:8] for r in cell or []) or "-"


def floor_line(cell: list[dict], row: dict) -> str:
    """The KL between this record's two rounds that answered the same: what the cell's own noise is worth."""
    same = [r for r in cell if r.get("md5") == row.get("md5") and r.get("logits")]
    if len(same) < 2:
        return "      no floor: this record's rounds all answered differently"
    one, other = out / same[0]["logits"], out / same[1]["logits"]
    if not one.exists() or not other.exists():
        return f"      floor unknown: {one if not one.exists() else other} is not there"
    a, b = M.answer_file(out / same[0]["file"])[1] or [], M.answer_file(out / same[1]["file"])[1] or []
    f = M.kl(one, other, same[0].get("rows") or M.NEW, same[1].get("rows") or M.NEW,
             M.shared_rows(a, b, same[0].get("rows") or M.NEW))
    if "error" in f:
        return f"      floor: {f['error']}"
    return (f"      floor {f['mean']:.2e} mean, {f['max']:.2e} max: this record's rounds "
            f"{same[0].get('round')} and {same[1].get('round')}")


def detail(key: tuple[str, str, str], old_cell: list[dict], new_cell: list[dict]) -> list[str]:
    """The measures of a cell that moved: an LLM's KL against its floor, an image's pixels' difference."""
    one, other = M.majority(new_cell), M.majority(old_cell)
    if key[0] == "llm":
        if not one.get("logits") or not other.get("logits"):
            return ["      no logits in the record: the KL needs both builds' dumps (LOGITS=1)"]
        one_file, other_file = out / one["logits"], prev / other["logits"]
        if not one_file.exists() or not other_file.exists():
            return [f"      {one_file if not one_file.exists() else other_file}: the dump is not there"]
        one_ids = M.answer_file(out / one["file"])[1] or []
        other_ids = M.answer_file(prev / other["file"])[1] or []
        rows = M.shared_rows(one_ids, other_ids, one.get("rows") or M.NEW)
        k = M.kl(other_file, one_file, other.get("rows") or M.NEW, one.get("rows") or M.NEW, rows)
        if "error" in k:
            return [f"      KL: {k['error']}"]
        total = max(one.get("rows") or M.NEW, other.get("rows") or M.NEW)
        text = (f"      KL {k['mean']:.2e} mean, {k['max']:.2e} max, top-1 {k['top1']:.3f}, over {k['rows']} rows of "
                f"{total}")
        if k["rows"] < total:
            text += f" (the answers part after {k['rows']} tokens)"
        if k["bitwise"]:
            text += ", bitwise equal"
        return [text, floor_line(new_cell, one)]
    one_file, other_file = out / one["file"], prev / other["file"]
    if not one_file.exists() or not other_file.exists():
        return [f"      {one_file if not one_file.exists() else other_file}: the image is not there"]
    p = M.pixels(one_file, other_file)
    if "error" in p:
        return [f"      pixels: {p['error']}"]
    psnr = "infinite" if p["psnr"] == float("inf") else f"{p['psnr']:.1f} dB"
    lines = [f"      pixels {p['size']}: max {p['max']}, mean {p['mean']:.2f} of 255, {100 * p['differ']:.1f}% "
             f"differ, PSNR {psnr}"]
    if len({r.get("md5") for r in new_cell if r.get("md5")}) > 1:
        lines.append("      its own rounds differ: this record's image is not the same twice")
    return lines


new_cells, old_cells = M.cells(rows), M.cells(old_rows)
print(f"== {out} ({meta.get('commit', '')[:8]}, {meta.get('gpu', '')}, {meta.get('date', '')})")
print(f"== {prev} ({old_meta.get('commit', '')[:8]}, {old_meta.get('gpu', '')}, {old_meta.get('date', '')})")
counts = {"same": 0, "bits": 0, "mixed": 0, "changed": 0, "failed": 0, "new": 0, "not compared": 0}
moved: list[str] = []
for key in [*new_cells, *(k for k in old_cells if k not in new_cells)]:
    old_cell, new_cell = old_cells.get(key), new_cells.get(key)
    v = M.verdict(old_cell, new_cell)
    if v == "same" and old_cell and new_cell and M.bits_moved(old_cell, new_cell):
        v = "bits"
    counts[v] += 1
    print(f"{key[0]:5s} {key[1]:20s} {key[2]:7s} {v.upper():12s} {md5s(new_cell):20s} | {md5s(old_cell)}")
    if v in ("bits", "mixed", "changed"):
        moved.append(f"{key[1]} {key[2]}")
        for text in detail(key, old_cell or [], new_cell or []):
            print(text)
print(f"== {sum(counts.values())} cells: {counts['same']} same, {counts['bits']} same answer from other bits, "
      f"{counts['mixed']} mixed, {counts['changed']} changed, {counts['failed']} failed, {counts['new']} new, "
      f"{counts['not compared']} not compared")
if moved:
    print(f"== moved: {', '.join(moved)}")
fails = counts["bits"] + counts["changed"] + counts["failed"]
if fails:
    print(f"CHANGED {fails}")
    sys.exit(1)
print("MIXED" if counts["mixed"] else "ALL SAME" if counts["same"] else "CHANGED 0 (no cell compared)")
sys.exit(0 if counts["same"] or counts["mixed"] else 1)
