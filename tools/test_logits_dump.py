"""#6030: tools/logits_dump.py reads STRATA_LOGITS_DUMP records with their 24-byte headers.

The engine's record is {int32 "LOGS" magic, int32 T, int32 n_vocab, int32 record, int64 position} and T * n_vocab float32.
Read as one flat float array, the magic becomes the "logit" 856028872704.0 at index 0 - #6030's report.  Model-free.

    python -m pytest tools/test_logits_dump.py -q
"""
import array
import os
import struct
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import logits_dump  # noqa: E402


def write_dump(path, recs):
    """recs: (T, n_vocab, position, flat list of floats) per record, as verify.cpp writes them."""
    with open(path, "wb") as f:
        for k, (t, n, pos, vals) in enumerate(recs):
            f.write(struct.pack("<iiiiq", 0x53474F4C, t, n, k, pos))
            array.array("f", vals).tofile(f)


class LogitsDump(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tmp.name, "dump.bin")

    def tearDown(self):
        self.tmp.cleanup()

    def test_records_and_rows_come_back_as_written(self):
        r0 = [0.0] * 8 + [1.5, -2.0, 7.25, 0.0, 0.0, 3.0, 0.0, -1.0]   # T=2: the first row all zero
        write_dump(self.path, [(2, 8, 100, r0), (1, 8, 102, [float(i) for i in range(8)])])
        got = list(logits_dump.records(self.path))
        self.assertEqual([(rec, t, n, pos) for rec, t, n, pos, _ in got], [(0, 2, 8, 100), (1, 1, 8, 102)])
        self.assertEqual([list(r) for r in got[0][4]], [r0[:8], r0[8:]])
        s = logits_dump.row_stats(got[0][4][1])
        self.assertEqual(s["top5"][0], (2, 7.25))
        self.assertEqual(s["nonzero"], 5)
        s0 = logits_dump.row_stats(got[0][4][0])
        self.assertEqual((s0["nonzero"], s0["max_abs"]), (0, 0.0))            # an all-zero head output stays visible as one

    def test_a_flat_float_read_takes_the_header_for_logits(self):
        write_dump(self.path, [(1, 8, 0, [0.0] * 8)])
        flat = array.array("f")
        with open(self.path, "rb") as f:
            flat.fromfile(f, os.path.getsize(self.path) // 4)
        self.assertEqual(flat[0], 856028872704.0)                             # the magic, as #6030 saw it
        rec = next(logits_dump.records(self.path))
        self.assertEqual(max(abs(v) for v in rec[4][0]), 0.0)                 # the real logits: all zero

    def test_a_cut_or_foreign_file_is_refused(self):
        write_dump(self.path, [(1, 8, 0, [1.0] * 8)])
        with open(self.path, "r+b") as f:
            f.truncate(os.path.getsize(self.path) - 4)
        with self.assertRaisesRegex(ValueError, "cut short"):
            list(logits_dump.records(self.path))
        with open(self.path, "wb") as f:
            f.write(struct.pack("<iiiiq", 0x12345678, 1, 8, 0, 0) + b"\0" * 32)
        with self.assertRaisesRegex(ValueError, "not the record magic"):
            list(logits_dump.records(self.path))

    def test_the_command_line(self):
        write_dump(self.path, [(1, 8, 5, [float(i) for i in range(8)])])
        self.assertEqual(logits_dump.main([self.path]), 0)
        self.assertEqual(logits_dump.main([self.path, "--rows"]), 0)


if __name__ == "__main__":
    unittest.main()
