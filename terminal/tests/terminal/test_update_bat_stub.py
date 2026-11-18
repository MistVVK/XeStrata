"""P5 (0.2.16): an old clone's UPDATE.bat ends cleanly after the pull that removed it.

A clone of 0.2.15 or earlier runs its own UPDATE.bat. The whole file is one ( ... ) block, so cmd has it in memory, but cmd opens a running
.bat file again after each program it calls. The pull inside it brought the 0.2.16 layout, which had no UPDATE.bat, so after
`call START-HERE.bat --update` cmd printed "The batch file cannot be found." twice and the exit code was 1, though the update had finished
(USERTEST-WIN K6). The 0.2.16 layout keeps a short UPDATE.bat, so the file is there again; its saved read position lies past the end of
the short stub and cmd just ends.
"""
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
STUB = REPO / "UPDATE.bat"
OLD_UPDATE_BAT_BYTES = 4000          # v0.2.15's UPDATE.bat with CRLF line ends: the read position cmd keeps is near its end


def old_shaped_update_bat() -> bytes:
    """The shape of v0.2.15's file: the whole body in one block, ending with the call of START-HERE.bat --update and an exit."""
    pad = "".join(f"  rem padding line {i:03d} so that this file is as long as the real one was\r\n" for i in range(38))
    text = ("@echo off\r\nsetlocal\r\ncd /d \"%~dp0\"\r\n(\r\n" + pad +
            "  call \"%~dp0START-HERE.bat\" --update %*\r\n  if not errorlevel 1 echo done-in-memory\r\n  exit /b\r\n)\r\n")
    return text.encode("ascii")


class Text(unittest.TestCase):
    def test_the_stub_is_there_short_and_calls_the_update(self):
        data = STUB.read_bytes()
        text = data.decode("ascii")
        self.assertLess(len(data), OLD_UPDATE_BAT_BYTES - 1000)           # shorter than the read position of the file it replaces
        self.assertIn('call "%~dp0START-HERE.bat" --update %*', text)
        self.assertIn("exit /b %errorlevel%", text)
        self.assertTrue(text.startswith("@echo off"))
        self.assertNotIn("\n\n", text.replace("\r\n", "\n"))

    def test_it_is_a_batch_file_with_cr_lf_line_ends(self):
        self.assertIn(b"\r\n", STUB.read_bytes())
        self.assertIn("*.bat text eol=crlf", (REPO / ".gitattributes").read_text(encoding="utf-8"))


@unittest.skipUnless(sys.platform == "win32" and shutil.which("cmd"), "needs Windows cmd")
class Run(unittest.TestCase):
    def run_update(self, replace_with):
        """Run an old-shaped UPDATE.bat whose START-HERE.bat (fake) does what the pull did: remove it, or put `replace_with` in its place."""
        with tempfile.TemporaryDirectory() as d:
            folder = Path(d)
            (folder / "UPDATE.bat").write_bytes(old_shaped_update_bat())
            if replace_with is None:
                swap = 'del "%~dp0UPDATE.bat"\r\n'
            else:
                (folder / "UPDATE.new").write_bytes(replace_with)
                swap = 'del "%~dp0UPDATE.bat"\r\ncopy /y "%~dp0UPDATE.new" "%~dp0UPDATE.bat" >nul\r\n'
            (folder / "START-HERE.bat").write_bytes(("@echo off\r\necho fake-start-here %*\r\n" + swap + "exit /b 0\r\n").encode("ascii"))
            env = {k: v for k, v in os.environ.items() if k != "NoDefaultCurrentDirectoryInExePath"}
            done = subprocess.run(["cmd", "/c", str(folder / "UPDATE.bat")], stdin=subprocess.DEVNULL, capture_output=True, text=True,
                                  cwd=d, env=env, timeout=60)
        return done

    def test_without_the_stub_the_old_file_ends_with_the_error_and_exit_1(self):
        done = self.run_update(None)                                    # the reproduction of K6
        self.assertEqual(done.returncode, 1)
        self.assertIn("The batch file cannot be found.", done.stderr)

    def test_with_the_stub_it_ends_cleanly(self):
        done = self.run_update(STUB.read_bytes())
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
        self.assertNotIn("cannot be found", done.stderr)
        self.assertIn("fake-start-here --update", done.stdout)

    def test_the_stub_run_by_hand_passes_its_arguments_and_the_exit_code(self):
        with tempfile.TemporaryDirectory() as d:
            folder = Path(d)
            shutil.copy(STUB, folder / "UPDATE.bat")
            (folder / "START-HERE.bat").write_bytes(b"@echo off\r\necho args=%*\r\nexit /b 3\r\n")
            env = {k: v for k, v in os.environ.items() if k != "NoDefaultCurrentDirectoryInExePath"}
            done = subprocess.run(["cmd", "/c", str(folder / "UPDATE.bat"), "--yes"], stdin=subprocess.DEVNULL, capture_output=True,
                                  text=True, cwd=d, env=env, timeout=60)
        self.assertEqual(done.returncode, 3)
        self.assertIn("args=--update --yes", done.stdout)


if __name__ == "__main__":
    unittest.main()
