"""0.2.17 retest (GB10 in a Vast container, IQ3_S): the container rule of make_room (5281c94b7) dropped the file cache of EVERY file in the engine's arguments before
each start, the PLE shard included. The engine then read the 26.8 GiB table cold from the disk (PLE table loaded in 48-52 s, against 0.0-1.8 s in 0.2.16, which
did not drop): +50 s per start. The shard named by --ple-gguf is left in the cache (unless --ple-io direct); the other files are dropped as before.

    python -m unittest tests.terminal.test_oomd_room_ple
"""
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from serve import oomd_pacer as op  # noqa: E402

GIB = 1 << 30


class PleShardStaysCached(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        d = Path(self.tmp.name)
        self.shard1 = d / "model-00001-of-00002.gguf"
        self.shard2 = d / "model-00002-of-00002.gguf"
        self.link = d / "elsewhere.gguf"
        for p in (self.shard1, self.shard2):
            p.write_bytes(b"x" * 1000)
        (d / "pack").mkdir()
        (d / "pack" / "t1.bin").write_bytes(b"y" * 500)
        self.dropped, self.said, self.cache = [], [], [40 * GIB]
        patch = mock.patch.object(op, "_DONTNEED", 4)
        patch.start()
        self.addCleanup(patch.stop)

    def run_room(self, args, free=78 * GIB):
        def drop(path):
            self.dropped.append(path.name)
            self.cache[0] = 0
            return True
        with mock.patch.object(op, "applicable", return_value=None), mock.patch.object(op, "in_container", return_value=True):
            return op.make_room(args, self.said.append, {}, drop=drop, cache_of=lambda: self.cache[0], free_of=lambda: free,
                                sizes=lambda p: 84 * GIB // 3)

    def args(self, *extra):
        return ["--pack", str(self.shard1.parent / "pack"), "--native", str(self.shard1), "--ple-gguf", str(self.shard2), *extra]

    def test_the_ple_shard_is_not_dropped_with_ple_io_ram(self):
        # the GB10 figures: 78 GB free is below 1.1 x the model's files, so the rule fires, as in 0.2.17
        freed = self.run_room(self.args("--ple-io", "ram"))
        self.assertGreater(freed, 0)
        self.assertEqual(sorted(self.dropped), ["model-00001-of-00002.gguf", "t1.bin"])
        self.assertNotIn("model-00002-of-00002.gguf", self.dropped)

    def test_the_ple_shard_is_not_dropped_with_ple_io_mmap_or_when_ple_io_is_not_named(self):
        for extra in (("--ple-io", "mmap"), ()):
            self.dropped.clear()
            self.cache[0] = 40 * GIB
            self.run_room(self.args(*extra))
            self.assertNotIn("model-00002-of-00002.gguf", self.dropped, extra)

    def test_ple_io_direct_reads_at_random_and_gains_nothing_from_the_cache_so_it_is_dropped_as_before(self):
        self.run_room(self.args("--ple-io", "direct"))
        self.assertIn("model-00002-of-00002.gguf", self.dropped)

    def test_a_link_to_the_shard_is_the_same_file(self):
        try:
            self.link.symlink_to(self.shard2)
        except (OSError, NotImplementedError):
            self.skipTest("no symlinks here")
        args = ["--native", str(self.shard1), "--native", str(self.link), "--ple-gguf", str(self.shard2), "--ple-io", "ram"]
        self.run_room(args)
        self.assertEqual(self.dropped, ["model-00001-of-00002.gguf"])

    def test_a_model_without_a_ple_table_drops_every_file_as_before(self):
        self.run_room(["--pack", str(self.shard1.parent / "pack"), "--native", str(self.shard1), "--native", str(self.shard2)])
        self.assertEqual(sorted(self.dropped), ["model-00001-of-00002.gguf", "model-00002-of-00002.gguf", "t1.bin"])

    def test_ple_table_file(self):
        self.assertEqual(op.ple_table_file(["--ple-gguf", str(self.shard2), "--ple-io", "ram"]), self.shard2)
        self.assertIsNone(op.ple_table_file(["--ple-gguf", str(self.shard2), "--ple-io", "direct"]))
        self.assertIsNone(op.ple_table_file(["--native", str(self.shard1)]))
        self.assertIsNone(op.ple_table_file(["--ple-gguf", "relative.gguf"]))


if __name__ == "__main__":
    unittest.main()
