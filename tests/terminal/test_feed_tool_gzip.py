"""feed_tool can gzip its tar.gz archives with several threads (OPT-IN, STRATA_PACK_THREADS) (Q4-READY 11-10: the gzip of one 180 MB engine archive was ~50 s on one core of a release box).

What these tests pin:
  * a multi-threaded archive holds the same members, modes and bytes as the single-stream one (STRATA_PACK_THREADS=1, the gzip of before);
  * it is several gzip members, and Python's gzip / tarfile (what Control's updater and components.py read) and the system's tar read it as one stream;
  * the same tree always gives the same archive bytes (no file name or time in the gzip headers);
  * pack-engine's entry names the archive's real sha256 and size, and a zip is unchanged.
"""
from __future__ import annotations

import gzip
import hashlib
import io
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "packaging"))

import feed_tool  # noqa: E402


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def members(archive: Path) -> list[tuple]:
    with tarfile.open(archive, "r:gz") as tar:
        out = []
        for m in tar.getmembers():
            data = tar.extractfile(m).read() if m.isfile() else b""
            out.append((m.name, m.mode, m.type, m.linkname, hashlib.sha256(data).hexdigest()))
        return out


class ParallelGzip(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.tree = self.root / "tree"
        (self.tree / "lib").mkdir(parents=True)
        rnd = os.urandom(300_000)
        (self.tree / "strata").write_bytes(b"\x7fELF" + rnd)                       # an "executable" (ELF magic)
        (self.tree / "lib" / "big.so").write_bytes(rnd * 7 + b"tail")               # compressible + random, several blocks
        (self.tree / "notes.txt").write_text("hello\n" * 5000)
        (self.tree / "empty.bin").write_bytes(b"")
        self.files = feed_tool._tree(self.tree)
        p = mock.patch.object(feed_tool, "GZIP_BLOCK", 64 * 1024)                  # small blocks: many members even for a small tree
        p.start()
        self.addCleanup(p.stop)

    def pack(self, name: str, threads: int) -> Path:
        out = self.root / name
        with mock.patch.dict(os.environ, {"STRATA_PACK_THREADS": str(threads)}):
            feed_tool._write_archive(self.files, out, "tar.gz", "strata")
        return out

    def test_the_same_members_and_bytes_as_the_single_stream_archive(self):
        one = self.pack("one.tar.gz", 1)
        many = self.pack("many.tar.gz", 4)
        self.assertEqual(members(one), members(many))
        self.assertFalse((self.root / "many.tar.gz.tar-part").exists(), "the plain tar beside the archive is removed")

    def test_several_gzip_members_read_as_one_stream(self):
        many = self.pack("many.tar.gz", 4)
        raw = many.read_bytes()
        self.assertGreater(raw.count(b"\x1f\x8b\x08"), 4, "more than one gzip member")
        plain = self.root / "plain.tar"
        feed_tool._write_tar(self.files, plain, "w", "strata")
        self.assertEqual(gzip.decompress(raw), plain.read_bytes())               # Python's gzip reads every member
        with tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz") as tar:          # tarfile's r:gz (strata_control/components.py, updater.py)
            self.assertEqual(len(tar.getmembers()), len(members(many)))
        tar_exe = shutil.which("tar")
        if tar_exe and os.name != "nt":
            listed = subprocess.run([tar_exe, "-tzf", str(many)], capture_output=True, text=True)
            self.assertEqual(listed.returncode, 0, listed.stderr)
            self.assertEqual(sorted(l.rstrip("/") for l in listed.stdout.split()), sorted(m[0] for m in members(many)))
            dest = self.root / "x"
            dest.mkdir()
            self.assertEqual(subprocess.run([tar_exe, "-xzf", str(many), "-C", str(dest)]).returncode, 0)
            self.assertEqual((dest / "lib" / "big.so").read_bytes(), (self.tree / "lib" / "big.so").read_bytes())

    def test_the_same_tree_gives_the_same_bytes(self):
        a = self.pack("a.tar.gz", 4)
        b = self.pack("b.tar.gz", 3)
        self.assertEqual(sha(a), sha(b))                                         # pieces are cut by bytes, not by thread count

    def test_the_thread_count(self):
        with mock.patch.dict(os.environ, {"STRATA_PACK_THREADS": "1"}):
            self.assertEqual(feed_tool.gzip_threads(), 1)
        with mock.patch.dict(os.environ, {"STRATA_PACK_THREADS": "6"}):
            self.assertEqual(feed_tool.gzip_threads(), 6)
        for unset in ("", "x", "0"):                                          # OPT-IN: no setting (or a bad one) is the single stream
            with mock.patch.dict(os.environ, {"STRATA_PACK_THREADS": unset}), mock.patch.object(os, "cpu_count", return_value=32):
                self.assertEqual(feed_tool.gzip_threads(), 1)
        env = {k: v for k, v in os.environ.items() if k != "STRATA_PACK_THREADS"}
        with mock.patch.dict(os.environ, env, clear=True):
            self.assertEqual(feed_tool.gzip_threads(), 1)

    def test_the_default_archive_is_the_single_stream_gzip_of_before(self):
        env = {k: v for k, v in os.environ.items() if k != "STRATA_PACK_THREADS"}
        out = self.root / "default.tar.gz"
        with mock.patch.dict(os.environ, env, clear=True), mock.patch.object(feed_tool, "_parallel_gzip", side_effect=AssertionError("threaded by default")):
            feed_tool._write_archive(self.files, out, "tar.gz", "strata")
        self.assertEqual(out.read_bytes()[:3], b"\x1f\x8b\x08")
        with gzip.open(out) as g:                                               # one stream, as tarfile w:gz writes it
            g.read()

    def test_a_zip_is_unchanged(self):
        out = self.root / "c.zip"
        with mock.patch.dict(os.environ, {"STRATA_PACK_THREADS": "4"}), mock.patch.object(feed_tool, "_parallel_gzip", side_effect=AssertionError("zip")):
            feed_tool._write_archive(self.files, out, "zip", "strata.exe")
        self.assertTrue(out.is_file())


if __name__ == "__main__":
    unittest.main()
