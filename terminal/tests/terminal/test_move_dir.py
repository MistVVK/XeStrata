"""#4853: a checked engine folder that Windows (a virus scanner, the indexer) will not let Strata rename is copied into place instead of
ending the install with "the update didn't finish". A rename that works is still a rename."""

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from strata_control import updater


def tree(root: Path) -> Path:
    (root / "lib").mkdir(parents=True)
    (root / "strata.exe").write_bytes(b"exe")
    (root / "lib" / "cublas.dll").write_bytes(b"dll" * 1000)
    return root


class MoveDir(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.source = tree(self.root / "stage" / "engine")
        self.destination = self.root / "live" / "engine"
        self.destination.parent.mkdir()
        self.clock = [0.0]

    def sleep(self, seconds):
        self.clock[0] += seconds

    def move(self, patience=5.0):
        updater._move_dir(self.source, self.destination, patience=patience, sleep=self.sleep, clock=lambda: self.clock[0])

    def test_a_rename_that_works_is_a_rename(self):
        self.move()
        self.assertTrue((self.destination / "strata.exe").is_file())
        self.assertFalse(self.source.exists())

    def test_a_rename_held_for_a_moment_is_waited_for(self):
        real, calls = os.replace, []

        def held(a, b):
            calls.append(1)
            if len(calls) < 4:
                raise PermissionError(13, "in use")
            return real(a, b)
        with mock.patch.object(updater.os, "replace", held):
            self.move()
        self.assertEqual(len(calls), 4)
        self.assertTrue((self.destination / "lib" / "cublas.dll").is_file())

    def test_a_rename_held_for_good_is_copied(self):
        with mock.patch.object(updater.os, "replace", side_effect=PermissionError(5, "Access is denied")):
            self.move(patience=3.0)
        self.assertGreaterEqual(self.clock[0], 3.0)                       # it waited the whole patience first
        self.assertEqual((self.destination / "lib" / "cublas.dll").read_bytes(), b"dll" * 1000)
        self.assertEqual((self.destination / "strata.exe").read_bytes(), b"exe")
        self.assertFalse(self.source.exists())

    def test_a_copy_that_fails_too_says_both_and_leaves_nothing_behind(self):
        with mock.patch.object(updater.os, "replace", side_effect=PermissionError(5, "Access is denied")), \
                mock.patch.object(updater.shutil, "copytree", side_effect=OSError(28, "No space left on device")):
            with self.assertRaises(updater.UpdateError) as caught:
                self.move(patience=1.0)
        self.assertEqual(caught.exception.exit_code, 6)
        self.assertIn("could not be moved into place", str(caught.exception))
        self.assertIn("copying it did not work either", str(caught.exception))
        self.assertFalse(self.destination.exists())
        self.assertTrue((self.source / "strata.exe").is_file())           # the staged copy is untouched

    @unittest.skipUnless(os.name == "nt", "a program holding a file inside the folder refuses the rename only on Windows")
    def test_a_real_open_file_blocks_the_rename_and_the_copy_still_works(self):
        with open(self.source / "lib" / "cublas.dll", "rb"):              # what a scanner does: it holds one file open
            self.move(patience=0.0)
        self.assertEqual((self.destination / "lib" / "cublas.dll").read_bytes(), b"dll" * 1000)


if __name__ == "__main__":
    unittest.main()
