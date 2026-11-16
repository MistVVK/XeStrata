# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Tests for tools/bench_md5.py, without a model: answers, records, verdicts and the measures on small dumps.

The last class reads the answers the unify2 records kept (their `llm/*.out` and their `--dump-logits` files) and
checks the md5s against the tables their reports published, so the watch's reading of a past record is the record's
own.  DATA=<data folder> names them; without it (or without the records) those tests skip.

    python -m unittest tools.test_bench_md5
"""
from __future__ import annotations

import hashlib
import json
import os
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import bench_md5 as M  # noqa: E402

try:                                   # the measures of a changed cell: the data folder's venv has both
    import numpy  # noqa: F401
    HAVE_NUMPY = True
except ImportError:
    HAVE_NUMPY = False

JUDGE = ROOT / ".claude" / "skills" / "upstream-sync" / "scripts" / "md5_judge.py"
DATA = Path(os.environ.get("DATA", "")) if os.environ.get("DATA") else None
UNIFY2 = DATA / "records" / "2026-11-10-unify2" if DATA else None

GENERATE = b"""prompt  : 248045 8678 198
output  : 40 3172 1151 539
decode                   32 tokens in 434.2 ms  ->  73.69 tok/s
"""
PLAIN = b"""tokens: 40,1144,310,1423
decode 32 tokens in 434.2 ms -> 73.69 tok/s
"""


def dump(path: Path, rows: int, vocab: int, header: bool = False, shift: float = 0.0) -> Path:
    """A --dump-logits file of `rows` rows of `vocab` logits (a ramp, so every row's pick is the last one)."""
    values = []
    for r in range(rows):
        values += [float(r) / 1000 + float(v) / vocab / 10 + shift for v in range(vocab)]
    with open(path, "wb") as f:
        if header:
            f.write(struct.pack("<ii", vocab, rows))
        f.write(struct.pack(f"<{len(values)}f", *values))
    return path


class Answer(unittest.TestCase):
    def test_the_generate_engines_line(self):
        self.assertEqual(M.answer_lines(GENERATE), [b"output  : 40 3172 1151 539\n"])
        self.assertEqual(M.answer_tokens(GENERATE), [40, 3172, 1151, 539])

    def test_the_plain_engines_line(self):
        self.assertEqual(M.answer_lines(PLAIN), [b"tokens: 40,1144,310,1423\n"])
        self.assertEqual(M.answer_tokens(PLAIN), [40, 1144, 310, 1423])

    def test_the_md5_is_of_the_line_and_its_newline(self):
        want = hashlib.md5(b"output  : 40 3172 1151 539\n").hexdigest()
        self.assertEqual(M.answer_md5(GENERATE), want)
        self.assertEqual(M.answer_md5(b"output  : 40 3172 1151 539"), want)   # the last line, unterminated

    def test_no_answer(self):
        self.assertEqual(M.answer_lines(b"xestrata: no GPU\n"), [])
        self.assertIsNone(M.answer_md5(b"xestrata: no GPU\n"))
        self.assertIsNone(M.answer_tokens(b""))

    def test_the_prompt_line_is_not_the_answer(self):
        self.assertEqual(M.answer_tokens(b"prompt  : 1 2 3\noutput  : 4 5\n"), [4, 5])

    def test_a_file(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "a.out"
            path.write_bytes(GENERATE)
            self.assertEqual(M.answer_file(path)[0], M.answer_md5(GENERATE))
            self.assertEqual(M.answer_file(path)[1], [40, 3172, 1151, 539])
            self.assertEqual(M.answer_file(Path(d) / "none.out"), (None, None))


class Records(unittest.TestCase):
    def row(self, md5, rnd=1, bits=None, tier="1k", model="q38", kind="llm"):
        return {"kind": kind, "model": model, "tier": tier, "round": rnd, "md5": md5, "bits": bits}

    def test_round_trip(self):
        with tempfile.TemporaryDirectory() as d:
            rows = [self.row("aa"), self.row("aa", rnd=2), self.row("bb", model="q35")]
            rows[0]["file"] = "answers/q38-1k-r1.out"
            M.write(Path(d), rows, {"commit": "c"})
            back, meta = M.read(Path(d))
            self.assertEqual(back, rows)
            self.assertEqual(meta["commit"], "c")
            self.assertEqual((Path(d) / "md5.txt").read_text().splitlines()[1].split()[0], "llm")

    def test_cells_orders_the_rounds(self):
        rows = [self.row("b", rnd=3), self.row("a", rnd=1), self.row("a", rnd=2)]
        self.assertEqual([r["md5"] for r in M.cells(rows)[("llm", "q38", "1k")]], ["a", "a", "b"])

    def test_majority_is_the_most_given_answer(self):
        rows = [self.row("a", rnd=1), self.row("b", rnd=2), self.row("b", rnd=3)]
        self.assertEqual(M.majority(rows)["round"], 2)

    def test_verdicts(self):
        a, b = [self.row("a")], [self.row("a")]
        self.assertEqual(M.verdict(a, b), "same")
        self.assertEqual(M.verdict(a, [self.row("b")]), "changed")
        mixed = [self.row("a"), self.row("b", rnd=2)]
        self.assertEqual(M.verdict(a, mixed), "mixed")
        self.assertEqual(M.verdict(a, [self.row(None)]), "failed")
        self.assertEqual(M.verdict([self.row(None)], a), "new")
        self.assertEqual(M.verdict([self.row(None)], [self.row(None)]), "not compared")
        self.assertEqual(M.verdict(None, a), "not compared")

    def test_bits_moved(self):
        self.assertTrue(M.bits_moved([self.row("a", bits="1")], [self.row("a", bits="2")]))
        self.assertFalse(M.bits_moved([self.row("a", bits="1")], [self.row("a", bits="1")]))
        self.assertFalse(M.bits_moved([self.row("a")], [self.row("a")]))       # no dumps: unknown, not moved

    def test_shared_rows(self):
        self.assertEqual(M.shared_rows([1, 2, 3], [1, 2, 3], 32), 32)
        self.assertEqual(M.shared_rows([1, 2, 3], [1, 2, 9], 32), 3)
        self.assertEqual(M.shared_rows([9, 2, 3], [1, 2, 9], 32), 1)
        self.assertEqual(M.shared_rows(None, [1], 32), 32)


@unittest.skipUnless(HAVE_NUMPY, "numpy (the data folder's venv)")
class Measures(unittest.TestCase):
    def test_kl_of_a_dump_with_itself(self):
        with tempfile.TemporaryDirectory() as d:
            one = dump(Path(d) / "a.f32", 4, 64)
            k = M.kl(one, one, 4, 4, 4)
            self.assertEqual((k["rows"], k["vocab"]), (4, 64))
            self.assertTrue(k["bitwise"])
            self.assertEqual((k["mean"], k["max"], k["top1"]), (0.0, 0.0, 1.0))

    def test_kl_of_another_distribution(self):
        with tempfile.TemporaryDirectory() as d:
            one = dump(Path(d) / "a.f32", 4, 64)
            other = dump(Path(d) / "b.f32", 4, 64, shift=1.0)
            k = M.kl(one, other, 4, 4, 4)
            self.assertFalse(k["bitwise"])
            self.assertGreater(k["mean"], 0.0)
            self.assertGreaterEqual(k["max"], k["mean"])
            self.assertEqual(k["top1"], 1.0)      # a ramp's pick is its last value, moved or not

    def test_the_generate_engines_header(self):
        with tempfile.TemporaryDirectory() as d:
            one = dump(Path(d) / "a.f32", 4, 64, header=True)
            self.assertEqual(M.strip_header(numpy.fromfile(one, dtype=numpy.float32))[1], 4)
            k = M.kl(one, one, 32, 32, 32)        # the caller's rows are overridden by the header's
            self.assertEqual((k["rows"], k["vocab"]), (4, 64))

    def test_rows_are_the_smaller_count(self):
        with tempfile.TemporaryDirectory() as d:
            one = dump(Path(d) / "a.f32", 8, 64)
            other = dump(Path(d) / "b.f32", 4, 64)
            self.assertEqual(M.kl(one, other, 8, 4, 32)["rows"], 4)
            self.assertEqual(M.kl(one, other, 8, 4, 3)["rows"], 3)

    def test_other_vocabularies_are_an_error(self):
        with tempfile.TemporaryDirectory() as d:
            one = dump(Path(d) / "a.f32", 4, 64)
            other = dump(Path(d) / "b.f32", 4, 32)
            self.assertIn("vocabularies", M.kl(one, other, 4, 4, 4)["error"])

    def png(self, path: Path, value: int) -> Path:
        try:
            from PIL import Image
        except ImportError:
            self.skipTest("no PIL in this interpreter")
        Image.new("RGBA", (8, 8), (value, value, value, 255)).save(path)
        return path

    def test_pixels_of_the_same_image(self):
        with tempfile.TemporaryDirectory() as d:
            one = self.png(Path(d) / "a.png", 128)
            p = M.pixels(one, one)
            self.assertEqual((p["max"], p["mean"], p["differ"]), (0, 0.0, 0.0))
            self.assertEqual(p["psnr"], float("inf"))

    def test_pixels_of_another_image(self):
        with tempfile.TemporaryDirectory() as d:
            one, other = self.png(Path(d) / "a.png", 128), self.png(Path(d) / "b.png", 129)
            p = M.pixels(one, other)
            self.assertEqual(p["max"], 1)
            self.assertEqual(p["differ"], 0.75)   # RGB moved, the alpha did not
            self.assertGreater(p["psnr"], 40)

    def test_pixels_of_another_size(self):
        with tempfile.TemporaryDirectory() as d:
            try:
                from PIL import Image
            except ImportError:
                self.skipTest("no PIL in this interpreter")
            one = self.png(Path(d) / "a.png", 128)
            other = Path(d) / "b.png"
            Image.new("RGBA", (4, 4)).save(other)
            self.assertIn("8x8", M.pixels(one, other)["error"])


@unittest.skipUnless(UNIFY2 and UNIFY2.exists(), "DATA: the unify2 records (DATA=<data folder>)")
class ThePastRecords(unittest.TestCase):
    """The md5s of V1's and S's answers are the ones their reports published: the watch reads a record as its own."""

    def md5(self, path: Path) -> str:
        md5, _ = M.answer_file(path)
        if md5 is None:
            self.fail(f"{path}: no answer")
        return md5[:8]

    def need(self, what: str, *paths: Path) -> None:
        """Skip unless every file the test reads is there: a cleaned-up record keeps its folder, not its dumps."""
        missing = [p for p in paths if not p.exists()]
        if missing:
            self.skipTest(f"{what}: {len(missing)} of its files are gone ({missing[0]})")

    def test_v1_answers(self):
        llm = UNIFY2 / "V1" / "llm"
        names = ("q38-03-1k-after-r1", "q35-03-1k-after-r1", *(f"q35-03-1k-before-r{r}" for r in (1, 2, 3)))
        self.need("V1's answers", *(llm / f"v-{n}.out" for n in names))
        # V1's table: q38's answer was 6f374fe1 at 1k, q35's 3db5b39a before and 6b6f4b5a after
        self.assertEqual(self.md5(llm / "v-q38-03-1k-after-r1.out"), "6f374fe1")
        self.assertEqual(self.md5(llm / "v-q35-03-1k-before-r1.out"), "3db5b39a")
        self.assertEqual(self.md5(llm / "v-q35-03-1k-after-r1.out"), "6b6f4b5a")
        # one answer a side, three rounds: the cell's md5 set is one md5
        cell = {M.answer_file(llm / f"v-q35-03-1k-before-r{r}.out")[0] for r in (1, 2, 3)}
        self.assertEqual(len(cell), 1)

    def test_v1_before_against_after(self):
        """The judge lists the cells V1's own table says moved: q35 32k the same, q35 1k and ornith 1k moved."""
        llm = UNIFY2 / "V1" / "llm"
        cells = (("q35", "1k"), ("q35", "32k"), ("ornith", "1k"))
        self.need("V1's answers", *(llm / f"v-{model}-03-{tier}-{side}-r{r}.out" for side in ("before", "after")
                                    for r in (1, 2, 3) for model, tier in cells))
        with tempfile.TemporaryDirectory() as d:
            for side in ("before", "after"):
                rows = []
                for r in (1, 2, 3):
                    for model, tier in (("q35", "1k"), ("q35", "32k"), ("ornith", "1k")):
                        path = llm / f"v-{model}-03-{tier}-{side}-r{r}.out"
                        rows.append({"kind": "llm", "model": model, "tier": tier, "round": r,
                                     "md5": M.answer_file(path)[0], "file": str(path), "rows": 32})
                M.write(Path(d) / side, rows, {"commit": side})
            out = subprocess.run([sys.executable, str(JUDGE), str(Path(d) / "after"), "--prev",
                                  str(Path(d) / "before")], capture_output=True, text=True, check=False)
        text = out.stdout
        self.assertRegex(text, r"llm\s+q35\s+32k\s+SAME")
        self.assertRegex(text, r"llm\s+q35\s+1k\s+CHANGED")
        self.assertRegex(text, r"llm\s+ornith\s+1k\s+CHANGED")
        self.assertIn("CHANGED 2", text)
        self.assertEqual(out.returncode, 1)

    def test_s_answers_and_kl(self):
        """S's table: q38 32K's answer was d6a838ce, ornith 1K's bb6bcf6a; its KL of ornith 1K (before against head,
        over 64 rows) was 6.761e-02 mean and 1.016 max, its floor bitwise equal."""
        amd, dev = UNIFY2 / "S" / "amd" / "llm", UNIFY2 / "S" / "dev"
        self.need("S's answers and dumps", *(amd / f"{n}-head-r1.out" for n in ("q38-32k", "ornith-1k", "q35-1k")),
                  *(dev / "kl" / f"ornith-b70-1k-{n}.f32" for n in ("before-a", "before-b", "head")))
        self.assertEqual(self.md5(amd / "q38-32k-head-r1.out"), "d6a838ce")
        self.assertEqual(self.md5(amd / "ornith-1k-head-r1.out"), "bb6bcf6a")
        self.assertEqual(self.md5(amd / "q35-1k-head-r1.out"), "3db5b39a")
        kl = dev / "kl"
        head = M.kl(kl / "ornith-b70-1k-before-a.f32", kl / "ornith-b70-1k-head.f32", 64, 64, 64)
        self.assertAlmostEqual(head["mean"], 6.761e-02, places=4)
        self.assertAlmostEqual(head["max"], 1.016, places=3)
        self.assertAlmostEqual(head["top1"], 0.891, places=3)
        floor = M.kl(kl / "ornith-b70-1k-before-a.f32", kl / "ornith-b70-1k-before-b.f32", 64, 64, 64)
        self.assertTrue(floor["bitwise"])

    def test_s_kl_through_the_judge(self):
        """The same cell judged from two records of dumps: the answer the same, the bits moved, the KL printed."""
        dev = UNIFY2 / "S" / "dev"
        names = ("head", "before-a", "before-b")
        self.need("S's dumps", *(dev / "kl" / f"ornith-b70-1k-{n}.f32" for n in names),
                  *(dev / "llm" / f"kl-ornith-b70-1k-{n}.out" for n in names))
        with tempfile.TemporaryDirectory() as d:
            for side, names in (("head", ("head", "head")), ("before", ("before-a", "before-b"))):
                rows = []
                for rnd, name in enumerate(names, start=1):
                    path = dev / "kl" / f"ornith-b70-1k-{name}.f32"
                    answer = dev / "llm" / f"kl-ornith-b70-1k-{name}.out"
                    rows.append({"kind": "llm", "model": "ornith", "tier": "1k", "round": rnd,
                                 "md5": M.answer_file(answer)[0], "file": str(answer), "logits": str(path),
                                 "bits": hashlib.md5(path.read_bytes()).hexdigest(), "rows": 64})
                M.write(Path(d) / side, rows, {"commit": side})
            out = subprocess.run([sys.executable, str(JUDGE), str(Path(d) / "head"), "--prev",
                                  str(Path(d) / "before")], capture_output=True, text=True, check=False)
        self.assertIn("BITS", out.stdout)
        self.assertRegex(out.stdout, r"KL 6\.\d\de-02 mean")
        self.assertRegex(out.stdout, r"floor 0\.00e\+00 mean")
        self.assertEqual(out.returncode, 1)

    def test_the_records_agree_with_md5_json(self):
        """A record the watch wrote reads back as its table says: the rounds' md5s in order."""
        with tempfile.TemporaryDirectory() as d:
            rows = [{"kind": "llm", "model": "q38", "tier": "1k", "round": r,
                     "md5": hashlib.md5(f"{r}".encode()).hexdigest()} for r in (1, 2)]
            M.write(Path(d), rows, {})
            self.assertEqual(json.loads((Path(d) / "md5.json").read_text()), rows)
            self.assertEqual(M.cells(M.read(Path(d))[0])[("llm", "q38", "1k")], rows)
            self.assertIn("c4ca4238 c81e728d", (Path(d) / "md5.txt").read_text())


if __name__ == "__main__":
    unittest.main()
