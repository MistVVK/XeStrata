"""0.2.13: a data folder on an external drive is allowed (/run/media/$USER, /media/$USER, /mnt), real system folders are still refused,
and a refusal says "Nothing was downloaded or changed" once."""
from __future__ import annotations

import contextlib
import io
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import setup
from strata_control import folders

SAID = "Nothing was downloaded or changed"


def posix_rules():
    """The POSIX folder rules, whatever system the test runs on."""
    return mock.patch.object(folders, "_system_roots", lambda: list(folders._POSIX_SYSTEM))


class MountPoints(unittest.TestCase):
    def test_the_drives_a_desktop_mounts_are_not_system_folders(self):
        with posix_rules(), mock.patch.dict(os.environ, {"HOME": "/home/ann", "USERPROFILE": "/home/ann"}):
            for p in ("/run/media/ann/Backup/Strata", "/run/media/ann/My Drive/Strata-models", "/run/media/ann/Backup",
                      "/media/ann/Backup/Strata", "/media/ann/Backup", "/mnt/models/Strata", "/mnt/data"):
                with self.subTest(p=p):
                    self.assertFalse(folders._is_system(p), p)

    def test_real_system_folders_are_still_refused(self):
        with posix_rules(), mock.patch.dict(os.environ, {"HOME": "/home/ann", "USERPROFILE": "/home/ann"}):
            for p in ("/usr/share/strata", "/etc/strata", "/bin", "/boot/x", "/dev/shm/x", "/proc/1", "/sys/x", "/root/x", "/var/lib/strata",
                      "/lib/x", "/sbin", "/run", "/run/user/1000/strata", "/run/systemd/x", "/run/lock", "/run/media", "/run/mediax/Strata"):
                with self.subTest(p=p):
                    self.assertTrue(folders._is_system(p), p)

    def test_the_run_media_folder_of_the_mounts_is_not_a_place_to_save(self):
        with tempfile.TemporaryDirectory() as d:
            run = Path(d) / "run"
            (run / "media" / "ann" / "Backup").mkdir(parents=True)
            with mock.patch.object(folders, "_system_roots", lambda: [str(run)]), mock.patch.object(folders, "_POSIX_MOUNTS", (str(run / "media"),)), \
                    mock.patch.dict(os.environ, {"HOME": str(Path(d) / "home"), "USERPROFILE": str(Path(d) / "home")}):
                self.assertIsNone(folders.problem(str(run / "media" / "ann" / "Backup" / "Strata")))        # created on the drive: fine
                self.assertIn("system folder", folders.problem(str(run / "media"))[0])
                self.assertIn("system folder", folders.problem(str(run / "elsewhere"))[0])
                self.assertIn("system folder", folders.problem(str(run / "user" / "1000"))[0])

    def test_a_mount_the_user_cannot_write_to_says_so_and_is_not_called_a_system_folder(self):
        with tempfile.TemporaryDirectory() as d:
            run = Path(d) / "run"
            (run / "media" / "bob" / "Drive").mkdir(parents=True)                      # somebody else's drive
            with mock.patch.object(folders, "_system_roots", lambda: [str(run)]), mock.patch.object(folders, "_POSIX_MOUNTS", (str(run / "media"),)), \
                    mock.patch.object(folders, "_can_write", return_value=False), \
                    mock.patch.dict(os.environ, {"HOME": str(Path(d) / "home"), "USERPROFILE": str(Path(d) / "home")}):
                what, _ = folders.problem(str(run / "media" / "bob" / "Drive" / "Strata"))
            self.assertIn("cannot be created", what)
            self.assertNotIn("system folder", what)

    def test_a_writable_folder_outside_home_is_fine(self):                              # /mnt/... and the like, "outside /home"
        with tempfile.TemporaryDirectory() as d:
            with posix_rules(), mock.patch.dict(os.environ, {"HOME": "/home/nobody", "USERPROFILE": "/home/nobody"}):
                self.assertIsNone(folders.problem(str(Path(d) / "Strata")))


class OneSentence(unittest.TestCase):
    def test_the_closing_sentence_is_dropped_only_when_it_is_there(self):
        both = ("/etc/x is a system folder.", folders.NEXT_POSIX)
        self.assertEqual(folders.without_nothing_changed(both).count(SAID), 0)
        self.assertTrue(folders.without_nothing_changed(both).endswith("for example ~/strata-models."))
        self.assertEqual(folders.without_nothing_changed(("It is bad.", "Do that.")), "It is bad. Do that.")

    def test_a_refused_data_dir_says_nothing_changed_once(self):
        with tempfile.TemporaryDirectory() as d:
            f = Path(d) / "afile.txt"
            f.write_text("x")
            for target in (f, Path(d) / "afile.txt" / "inside"):
                out = io.StringIO()
                with self.subTest(target=target), contextlib.redirect_stdout(out), self.assertRaises(SystemExit):
                    setup.data_folder(str(target))
                self.assertEqual(out.getvalue().count(SAID), 1, out.getvalue())
                self.assertIn("was not saved as your data folder", out.getvalue())

    @unittest.skipIf(os.name == "nt", "POSIX paths")
    def test_a_system_folder_refusal_says_nothing_changed_once(self):
        out = io.StringIO()
        with posix_rules(), mock.patch.dict(os.environ, {"HOME": "/home/nobody", "USERPROFILE": "/home/nobody"}), \
                mock.patch.object(setup, "load_settings", return_value={}), contextlib.redirect_stdout(out), self.assertRaises(SystemExit):
            setup.data_folder("/usr/share/Strata")
        self.assertEqual(out.getvalue().count(SAID), 1, out.getvalue())
        self.assertIn("is a system folder", out.getvalue())


if __name__ == "__main__":
    unittest.main()
