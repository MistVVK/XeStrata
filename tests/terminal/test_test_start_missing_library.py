"""A test start that fails because the system lacks a shared library (boxia 11-15: the hip-rx6000 engine exited 127, "error while loading shared
libraries: libatomic.so.1: cannot open shared object file", and the update only said "The engine program did not run on this computer").

Proves: the message names the library in plain words with one next step (the package for this Linux family, or the names per family when it is
not known); the engine's output is kept in .updates/logs and the message says where; other failures keep their words; the full test start (a model
installed) names the library too."""

import os
import stat
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from strata_control import updater

LOADER = "./strata: error while loading shared libraries: libatomic.so.1: cannot open shared object file: No such file or directory"


def make_engine(folder: Path, body_posix: str, body_nt: str) -> Path:
    if os.name == "nt":
        path = folder / "strata.cmd"
        path.write_text("@echo off\r\n" + body_nt, encoding="utf-8")
    else:
        path = folder / "strata"
        path.write_text("#!/bin/sh\n" + body_posix, encoding="utf-8")
        path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


class Words(unittest.TestCase):
    def test_the_loader_line_names_the_library(self):
        self.assertEqual(updater.missing_library(LOADER), "libatomic.so.1")
        self.assertIsNone(updater.missing_library("No device of requested type available"))
        self.assertIsNone(updater.missing_library(None))

    def test_the_family_comes_from_os_release(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "os-release"
            for text, family in (('ID=ubuntu\nID_LIKE=debian\n', "debian"), ('ID=debian\n', "debian"), ('ID="fedora"\n', "fedora"),
                                 ('ID="rocky"\nID_LIKE="rhel centos fedora"\n', "fedora"), ('ID=arch\n', "arch"), ('ID=manjaro\nID_LIKE=arch\n', "arch"),
                                 ('ID=opensuse-tumbleweed\nID_LIKE="opensuse suse"\n', None)):
                path.write_text(text)
                self.assertEqual(updater.linux_family(str(path)), family, text)
            self.assertIsNone(updater.linux_family(str(Path(tmp) / "missing")))

    def test_one_next_step_with_the_package(self):
        self.assertEqual(updater.library_hint("libatomic.so.1", "debian"), "Install it with `sudo apt install libatomic1`, then run `strata update` again.")
        self.assertEqual(updater.library_hint("libatomic.so.1", "fedora"), "Install it with `sudo dnf install libatomic`, then run `strata update` again.")
        self.assertEqual(updater.library_hint("libatomic.so.1", "arch"), "Install it with `sudo pacman -S gcc-libs`, then run `strata update` again.")
        self.assertEqual(updater.library_hint("libatomic.so.1", None),
                         "Install it with your system's package manager (libatomic1 on Debian and Ubuntu, libatomic on Fedora, gcc-libs on Arch), "
                         "then run `strata update` again.")
        self.assertEqual(updater.library_hint("libfoo.so.3", "debian"),
                         "Install the package that has it with your system's package manager, then run `strata update` again.")

    def test_the_plain_message_names_the_library(self):
        text = str(updater.TestStartFailed("0.1.60", f"--version exited 127: {LOADER}", busy=False, hint="NEXT"))
        self.assertIn("The new engine 0.1.60 did not start correctly on this computer, so it was not installed.", text)
        self.assertIn("The engine needs the system library libatomic.so.1, and this computer does not have it.", text)
        self.assertNotIn("The engine program did not run on this computer.", text)
        self.assertTrue(text.endswith("NEXT"))

    def test_other_failures_keep_their_words(self):
        self.assertIn("The engine program did not run on this computer.", str(updater.TestStartFailed("0.1.48", "--version exited 3: dll missing")))


class BinarySmoke(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.engine = self.root / "engine"
        self.engine.mkdir()
        self.components = self.root / "components"
        self.ctx = SimpleNamespace(components=self.components)

    def component(self, exe):
        return SimpleNamespace(exe=exe, root=self.engine, version="0.1.60")

    def test_a_missing_library_is_named_with_the_package_and_the_log_is_kept(self):
        exe = make_engine(self.engine, f"echo '{LOADER}' >&2\nexit 127\n", f"echo {LOADER} 1>&2\r\nexit /b 127\r\n")
        with mock.patch.object(updater, "linux_family", return_value="debian"), self.assertRaises(updater.TestStartFailed) as caught:
            updater.binary_smoke(self.component(exe), ctx=self.ctx)
        log = self.components / updater.STATE_DIR / "logs" / "test-start-0.1.60.log"
        text = str(caught.exception)
        self.assertIn("The engine needs the system library libatomic.so.1, and this computer does not have it.", text)
        self.assertIn("Install it with `sudo apt install libatomic1`, then run `strata update` again.", text)
        self.assertIn(f"The engine's log from this test is in {log}.", text)
        self.assertFalse(caught.exception.busy)
        self.assertIn("libatomic.so.1", caught.exception.reason)
        kept = log.read_text()
        self.assertIn("exit code 127", kept)
        self.assertIn("libatomic.so.1: cannot open shared object file", kept)

    def test_another_failure_keeps_the_log_and_says_where(self):
        exe = make_engine(self.engine, "echo 'No device of requested type available' >&2\nexit 250\n",
                          "echo No device of requested type available 1>&2\r\nexit /b 250\r\n")
        with self.assertRaises(updater.TestStartFailed) as caught:
            updater.binary_smoke(self.component(exe), ctx=self.ctx)
        log = self.components / updater.STATE_DIR / "logs" / "test-start-0.1.60.log"
        self.assertIn("The engine program did not run on this computer.", str(caught.exception))
        self.assertIn(f"report it with the log: {log}", str(caught.exception))
        self.assertIn("No device of requested type", log.read_text())

    def test_without_a_context_nothing_is_written_and_the_words_are_as_before(self):
        exe = make_engine(self.engine, "echo 'bad' >&2\nexit 3\n", "echo bad 1>&2\r\nexit /b 3\r\n")
        with self.assertRaises(updater.TestStartFailed) as caught:
            updater.binary_smoke(self.component(exe))
        self.assertIn("Run `strata update --force` to try again; if it fails again, report it with the log.", str(caught.exception))
        self.assertFalse(self.components.exists())

    def test_the_default_test_start_with_no_model_keeps_the_log(self):
        exe = make_engine(self.engine, f"echo '{LOADER}' >&2\nexit 127\n", f"echo {LOADER} 1>&2\r\nexit /b 127\r\n")
        ctx = SimpleNamespace(components=self.components, run_config=None, environ={}, allow_no_model_smoke=True)
        with mock.patch.object(updater, "discover_run_config", return_value=None), mock.patch.object(updater, "linux_family", return_value=None), \
                self.assertRaises(updater.TestStartFailed) as caught:
            updater.default_test_start(self.root / "stage", self.component(exe), ctx)
        self.assertIn("libatomic1 on Debian and Ubuntu", str(caught.exception))
        self.assertTrue((self.components / updater.STATE_DIR / "logs" / "test-start-0.1.60.log").is_file())


class FullTestStart(unittest.TestCase):
    def test_a_loader_error_in_the_engine_log_is_named(self):
        with tempfile.TemporaryDirectory() as tmp:
            components = Path(tmp)
            log = components / updater.STATE_DIR / "logs" / "test-start-0.1.60.log"
            log.parent.mkdir(parents=True)
            log.write_text(LOADER + "\n")
            ctx = SimpleNamespace(components=components, test_timeout=60)
            with mock.patch.object(updater, "linux_family", return_value="fedora"):
                failure = updater._test_start_failure(SimpleNamespace(version="0.1.60"), {"ok": False, "error": "exited", "state": "failed"}, ctx, "")
            text = str(failure)
            self.assertIn("The engine needs the system library libatomic.so.1", text)
            self.assertIn("`sudo dnf install libatomic`", text)
            self.assertIn(str(log), text)


if __name__ == "__main__":
    unittest.main()
