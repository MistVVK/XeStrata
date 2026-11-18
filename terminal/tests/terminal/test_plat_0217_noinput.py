"""0.2.17 ("Also open"): the input that means "nobody is there" (strata_cli/noinput.py), used by `strata launch` so that `< /dev/null` shows no question."""

from __future__ import annotations

import io
import os
import tempfile
import unittest
from pathlib import Path

from strata_cli import noinput


class NoKeyboard(unittest.TestCase):
    def test_the_null_device_is_no_keyboard(self):
        with open(os.devnull, "r") as f:
            if os.name == "nt" and os.isatty(f.fileno()):
                self.skipTest("this Python reports NUL as a terminal")
            self.assertTrue(noinput.stdin_is_no_keyboard(f))

    def test_a_pipe_may_carry_answers_so_it_is_not(self):
        r, w = os.pipe()
        try:
            with os.fdopen(r, "r", closefd=False) as f:
                self.assertFalse(noinput.stdin_is_no_keyboard(f))
        finally:
            os.close(r)
            os.close(w)

    def test_a_file_with_answers_is_not(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "answers.txt"
            p.write_text("y\n")
            with p.open("r") as f:
                self.assertFalse(noinput.stdin_is_no_keyboard(f))

    def test_what_cannot_be_read_is_not_taken_for_no_keyboard(self):
        self.assertFalse(noinput.stdin_is_no_keyboard(io.StringIO("y\n")))        # a replaced stdin has no file number
        self.assertFalse(noinput.stdin_is_no_keyboard(object()))
        f = open(os.devnull)
        f.close()
        self.assertFalse(noinput.stdin_is_no_keyboard(f))


if __name__ == "__main__":
    unittest.main()
