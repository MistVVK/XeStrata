"""Lane NET item 7: setup may use an Unsloth mmproj-F16.gguf that the person already has as the Flash-Next picture reader, and the plan says the same.

The rule (setup.ALT_READERS, setup.accepted_alt_reader, setup.resolve_reader; the plan asks the same function through modelstate.reader_present):
  * only a file named mmproj-F16.gguf with the pinned size AND SHA-256, in a place where setup already looks for the pinned reader;
  * only when no pinned BF16 file is found anywhere (the pinned file always wins); anything else with that name is ignored, as before;
  * never downloaded, never deleted, never given a mark beside it: the SHA-256 is remembered in Strata's marks/ (keyed by real path, size,
    modification time and change time, so a replaced file is hashed again, also one rewritten in place with size and time kept);
  * the plan hashes without writing a mark and says nothing.
The tests use a small stand-in file and patch the pin; one test holds the real pin."""

from __future__ import annotations

import contextlib
import hashlib
import io
import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

from strata_cli.launch import modelstate
from tests.terminal.launch_support import HW_12, Home, catalog

ALT = "mmproj-F16.gguf"
GOOD = b"unsloth-reader " * 100
BAD = b"another-reader " * 100                                 # the same size, another content
assert len(GOOD) == len(BAD)
NOTE = ("  note: {path} has the name of Unsloth's picture reader but is a different file, so it is not used. "
        "The picture reader is downloaded (0.9 GB).")


class Base(unittest.TestCase):
    def setUp(self):
        home = Home().__enter__()
        self.addCleanup(home.__exit__, None, None, None)
        t = tempfile.TemporaryDirectory(prefix="strata-altreader-")
        self.addCleanup(t.cleanup)
        self.base = Path(t.name)
        self.cat = catalog(HW_12)
        self.mod = self.cat.strata.setup_module()               # the module the plan asks too
        self.pinned = self.mod.MMPROJ
        for p in (mock.patch.dict(self.mod.ALT_READERS, {self.pinned: (ALT, len(GOOD), hashlib.sha256(GOOD).hexdigest())}),
                  mock.patch.object(self.mod, "ROOT", self.base / "strata-root")):
            p.start()
            self.addCleanup(p.stop)
        self.own = self.base / "mine" / "IQ3_XXS"               # a person's own folder of shards
        self.own.mkdir(parents=True)
        self.models = self.base / "data" / "models"             # Strata's own models folder (the pinned file would be here)
        self.models.mkdir(parents=True)

    def start(self, gguf_dir="own"):
        """What setup does for the reader: (kind, path, printed text)."""
        out = io.StringIO()
        gd = self.own if gguf_dir == "own" else gguf_dir
        with contextlib.redirect_stdout(out):
            kind, path = self.mod.resolve_reader(self.pinned, self.models / self.pinned, self.models, [], gd, 907543008)
        return kind, path, out.getvalue()

    def plan(self):
        """What the plan believes: does setup find a reader without a download (and what it printed)."""
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            present = modelstate.reader_present(self.cat, "qwen", [self.base / "data"], own=self.own)
        return present, out.getvalue()

    def marks(self):
        d = self.mod.ROOT / "marks"
        return sorted(d.iterdir()) if d.is_dir() else []

    def hashes(self):
        """Count the files setup hashes from here on."""
        real = self.mod._hash_file
        calls = []

        def counting(path, size):
            calls.append(Path(path).name)
            return real(path, size)
        p = mock.patch.object(self.mod, "_hash_file", counting)
        p.start()
        self.addCleanup(p.stop)
        return calls


class TheRealPin(unittest.TestCase):
    def test_the_pinned_values_are_the_owners(self):
        import setup
        self.assertEqual(setup.ALT_READERS[setup.MMPROJ], ("mmproj-F16.gguf", 904004000,
                                                            "1f7b7f0b984cf065c604360c29c8098362ed61b290db0ff12c6f360bb1a8a980"))
        self.assertEqual(setup.MMPROJ, "mmproj-Qwen3.8-Flash-Next-BF16.gguf")
        self.assertEqual(list(setup.ALT_READERS), [setup.MMPROJ], "Flash-Next's pinned file only, no other family")


class Accepted(Base):
    def test_a_file_with_the_pinned_size_and_hash_is_used_and_checked_once(self):
        (self.own / ALT).write_bytes(GOOD)
        hashed = self.hashes()
        present, said = self.plan()                              # the plan first: hashes, says nothing, writes nothing
        self.assertTrue(present)
        self.assertEqual(said, "")
        self.assertEqual(self.marks(), [], "the plan writes no mark")
        kind, path, said = self.start()
        self.assertEqual((kind, path), ("alt", self.own / ALT))
        self.assertEqual(said, f"  checking {ALT} (SHA-256, 0.0 GB) ...\n")
        self.assertEqual(len(self.marks()), 1, "one mark in Strata's own folder")
        self.assertEqual(sorted(p.name for p in self.own.iterdir()), [ALT], "nothing was written beside the person's file")
        before = len(hashed)
        kind, _path, said = self.start()                         # the next run: the mark answers
        self.assertEqual((kind, said), ("alt", ""))
        self.assertTrue(self.plan()[0])
        self.assertEqual(len(hashed), before, "no second hash")

    def test_the_folder_above_and_its_models_folder_and_the_data_models_folder_count_too(self):
        for where in (self.own.parent, self.own.parent / "models", self.models):
            where.mkdir(parents=True, exist_ok=True)
            f = where / ALT
            f.write_bytes(GOOD)
            self.assertEqual(self.start()[:2], ("alt", f), str(where))
            self.assertTrue(self.plan()[0], str(where))
            f.unlink()
        self.assertEqual(self.start()[0], "fetch")
        self.assertFalse(self.plan()[0])


class NotAccepted(Base):
    def test_a_wrong_size_is_ignored_exactly_as_before(self):
        (self.own / ALT).write_bytes(GOOD[:-1])
        hashed = self.hashes()
        kind, path, said = self.start()
        self.assertEqual((kind, path, said), ("fetch", self.models / self.pinned, ""))
        self.assertEqual(hashed, [], "not even hashed")
        self.assertEqual(self.plan(), (False, ""))
        self.assertEqual((self.own / ALT).read_bytes(), GOOD[:-1])

    def test_a_wrong_hash_is_ignored_with_the_note_and_the_pinned_file_is_downloaded(self):
        (self.own / ALT).write_bytes(BAD)
        hashed = self.hashes()
        kind, path, said = self.start()
        self.assertEqual((kind, path), ("fetch", self.models / self.pinned))
        self.assertEqual(said, f"  checking {ALT} (SHA-256, 0.0 GB) ...\n" + NOTE.format(path=self.own / ALT) + "\n")
        self.assertEqual(self.plan(), (False, ""), "the plan agrees with the start: the reader is downloaded")
        self.assertEqual((self.own / ALT).read_bytes(), BAD, "the person's file is never touched")
        self.assertEqual(sorted(p.name for p in self.own.iterdir()), [ALT])
        before = len(hashed)
        kind, _path, said = self.start()                         # remembered: not hashed again, the note says it again
        self.assertEqual(kind, "fetch")
        self.assertEqual(said, NOTE.format(path=self.own / ALT) + "\n")
        self.assertEqual(len(hashed), before)

    def test_another_name_is_never_looked_at(self):
        (self.own / "mmproj-BF16.gguf").write_bytes(GOOD)
        (self.own / "mmproj-Q8_0.gguf").write_bytes(GOOD)
        self.assertEqual(self.start()[0], "fetch")
        self.assertFalse(self.plan()[0])

    def test_the_pinned_file_always_wins(self):
        (self.own / ALT).write_bytes(GOOD)
        pinned = self.own / self.pinned
        pinned.write_bytes(b"x" * 10)                            # beside the files: by name, any size (setup's mmproj_beside)
        hashed = self.hashes()
        kind, path, said = self.start()
        self.assertEqual((kind, path, said), ("near", pinned, ""))
        self.assertEqual(hashed, [], "the alternative is not even hashed")
        self.assertTrue(self.plan()[0])
        pinned.unlink()
        (self.models / self.pinned).write_bytes(b"y")            # Strata's own models folder
        kind, path, said = self.start()
        self.assertEqual((kind, path, said), ("fetch", self.models / self.pinned, ""), "the pinned file is used (and verified later by setup)")
        self.assertTrue(self.plan()[0])
        self.assertEqual(hashed, [])


class Replaced(Base):
    def test_a_file_accepted_once_and_then_replaced_by_another_of_the_same_size_is_hashed_again_and_refused(self):
        f = self.own / ALT
        f.write_bytes(GOOD)
        self.assertEqual(self.start()[0], "alt")
        self.assertEqual(len(self.marks()), 1)
        hashed = self.hashes()
        stat = f.stat()
        f.write_bytes(BAD)                                       # the same size and place, other bytes
        os.utime(f, ns=(stat.st_atime_ns, stat.st_mtime_ns + 5_000_000_000))      # (a new modification time, as any real replacement has)
        kind, _path, said = self.start()
        self.assertEqual(kind, "fetch")
        self.assertIn("is a different file", said)
        self.assertEqual(hashed, [ALT], "hashed again")
        self.assertEqual(self.plan(), (False, ""))

    def test_a_rewrite_in_place_that_keeps_size_and_modification_time_is_hashed_again_and_refused(self):
        f = self.own / ALT
        f.write_bytes(GOOD)
        self.assertEqual(self.start()[0], "alt")
        self.assertEqual(len(self.marks()), 1)
        hashed = self.hashes()
        stat = f.stat()
        time.sleep(0.05)                                         # past the file system clock step (a kernel tick on Linux)
        with open(f, "r+b") as fh:                               # the same file, rewritten in place (no new file, no rename)
            fh.write(BAD)
        os.utime(f, ns=(stat.st_atime_ns, stat.st_mtime_ns))     # the old modification time put back
        self.assertEqual((f.stat().st_size, f.stat().st_mtime_ns), (stat.st_size, stat.st_mtime_ns), "size and modification time as before")
        kind, _path, said = self.start()
        self.assertEqual(kind, "fetch")
        self.assertIn("is a different file", said)
        self.assertEqual(hashed, [ALT], "hashed again")
        self.assertEqual(self.plan(), (False, ""))

    def test_the_change_time_moves_on_such_a_rewrite_on_this_system(self):
        f = self.own / ALT
        f.write_bytes(GOOD)
        st = f.stat()
        before = self.mod._change_time_ns(f, st)
        time.sleep(0.05)                                         # past the file system clock step (a kernel tick on Linux)
        with open(f, "r+b") as fh:
            fh.write(BAD)
        os.utime(f, ns=(st.st_atime_ns, st.st_mtime_ns))
        self.assertNotEqual(self.mod._change_time_ns(f, f.stat()), before)
        if os.name == "nt":                                      # Windows: the file system's ChangeTime, not os.stat's ctime (the creation time)
            self.assertNotEqual(self.mod._change_time_ns(f, f.stat()), f.stat().st_ctime_ns)

    def test_other_kept_marks_keep_their_key(self):
        f = self.own / "model-00001-of-00002.gguf"
        f.write_bytes(b"shard")
        st = f.stat()
        key = f"{f.resolve()}|{st.st_size}|{st.st_mtime_ns}"
        self.assertEqual(self.mod._kept_mark(f).name, hashlib.sha1(key.encode()).hexdigest()[:24] + ".done",
                         "the key of every other kept mark (an Engram's too) is unchanged, so no mark made before is lost")
        self.assertNotEqual(self.mod._alt_reader_mark(f), self.mod._kept_mark(f))

    def test_a_changed_size_has_no_mark_either(self):
        f = self.own / ALT
        f.write_bytes(GOOD)
        self.assertEqual(self.start()[0], "alt")
        f.write_bytes(GOOD + b"!")
        self.assertEqual(self.start(), ("fetch", self.models / self.pinned, ""))

    def test_the_mark_is_keyed_by_the_real_path_so_a_linked_folder_shares_it_and_cannot_hide_a_replacement(self):
        real = self.base / "real"
        real.mkdir()
        (real / ALT).write_bytes(GOOD)
        link = self.base / "link"
        try:
            os.symlink(real, link, target_is_directory=True)
        except (OSError, NotImplementedError, AttributeError):
            self.skipTest("this account cannot make symbolic links")
        self.assertEqual(self.start(gguf_dir=real)[0], "alt")
        hashed = self.hashes()
        self.assertEqual(self.start(gguf_dir=link)[0], "alt")
        self.assertEqual(hashed, [], "the same file through the link: the same mark")
        stat = (real / ALT).stat()
        (real / ALT).write_bytes(BAD)
        os.utime(real / ALT, ns=(stat.st_atime_ns, stat.st_mtime_ns + 5_000_000_000))
        self.assertEqual(self.start(gguf_dir=link)[0], "fetch", "replaced behind the link: hashed again, refused")
        self.assertEqual(hashed, [ALT])


if __name__ == "__main__":
    unittest.main()
