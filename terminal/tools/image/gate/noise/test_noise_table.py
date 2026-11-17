"""tools/image/gate/noise/test_noise_table.py - TEST-ONLY: the packed glibc sin exception table (strata_sin_exc.bin).

Checks the file's sha256, that it unpacks to a sorted list of reachable v (positive floats below 6.3; repeats carry the same flag), and
re-derives a sample of entries: the correctly rounded sin from Python's decimal module and, where this Python's
math.sin is glibc on x86-64 (the reference platform), that glibc's double is that value one unit up or down as the
entry says.  Also checks the tables in strata_noise_tables.inc against make_tables.py.

    python -m pytest tools/image/gate/noise/test_noise_table.py"""
import hashlib
import math
import platform
import random
import struct
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
PRODUCT = HERE.parents[1] / "sdcpp_noise"       # tools/image/sdcpp_noise: the generator, its tables and the packed sin table
sys.path.insert(0, str(HERE))
import make_sin_exceptions as M  # noqa: E402

SHA256 = "3d61d50cca41c09e25b4acb0202aedd3c86bffdfbfed30df6066a5b5972429d6"
COUNT = 109290


class Table(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = (PRODUCT / "strata_sin_exc.bin").read_bytes()
        cls.entries = M.unpack(cls.data)

    def test_sha256_and_size(self):
        self.assertEqual(hashlib.sha256(self.data).hexdigest(), SHA256)
        self.assertEqual(len(self.entries), COUNT)
        self.assertLess(len(self.data), 200_000)

    def test_sorted_reachable_repeats_agree(self):
        # 109,290 entries over 92,529 distinct floats: 16,761 floats are listed twice, always with the same
        # flag (found by the second worker; the list comes from check_domain as it was).  The loader's lower-bound lookup is fine with that.
        keys = [k for k, _ in self.entries]
        self.assertEqual(keys, sorted(keys))
        self.assertEqual(len(set(keys)), 92529)
        flags = {}
        for k, down in self.entries:
            flags.setdefault(k, set()).add(down)
        self.assertTrue(all(len(s) == 1 for s in flags.values()))
        for k in (keys[0], keys[-1]):
            v = M.fval(k)
            self.assertTrue(0 < v < 6.3, v)

    def test_sample_rederived(self):
        glibc_x86 = sys.platform.startswith("linux") and platform.machine() == "x86_64" and \
            platform.libc_ver()[0] == "glibc"
        for k, down in random.Random(11).sample(self.entries, 300):
            v = M.fval(k)
            cr = M.dbits(M.cr_sin(v))
            if glibc_x86:
                self.assertEqual(M.dbits(math.sin(v)), cr + (-1 if down else 1), hex(k))
            self.assertIn(down, (0, 1))

    def test_pack_round_trip(self):
        self.assertEqual(M.pack(self.entries), self.data)


class Constants(unittest.TestCase):
    def test_tables_are_what_make_tables_writes(self):
        import subprocess
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            subprocess.run([sys.executable, "-I", "-B", str(HERE / "make_tables.py"),
                            str(Path(d) / "strata_noise_tables.inc")], check=True)
            self.assertEqual((Path(d) / "strata_noise_tables.inc").read_text().splitlines(),
                             (PRODUCT / "strata_noise_tables.inc").read_text().splitlines())


if __name__ == "__main__":
    unittest.main()
