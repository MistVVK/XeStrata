"""#5935: setup's menus colour the model family and size names only on a terminal that shows colour: never in a pipe or a log (the app's
installer reads setup's output), never with NO_COLOR or TERM=dumb, and on Windows only where the console takes escape codes."""
import io
import os
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import setup  # noqa: E402


class Tty(io.StringIO):
    def isatty(self):
        return True


class MenuColour(unittest.TestCase):
    def colour(self, out, env, win=False, vt=True):
        with mock.patch.object(sys, "stdout", out), mock.patch.dict(os.environ, env, clear=True), \
                mock.patch.object(setup, "WIN", win), mock.patch.object(setup, "_VT", [vt]):
            return setup.menu_colour("36")

    def test_a_terminal_gets_colour(self):
        self.assertEqual(self.colour(Tty(), {"TERM": "xterm-256color"}), ("\033[36m", "\033[0m"))

    def test_a_pipe_or_a_log_gets_none(self):
        self.assertEqual(self.colour(io.StringIO(), {"TERM": "xterm"}), ("", ""))

    def test_no_color_and_a_dumb_terminal_get_none(self):
        self.assertEqual(self.colour(Tty(), {"NO_COLOR": ""}), ("", ""))
        self.assertEqual(self.colour(Tty(), {"TERM": "dumb"}), ("", ""))

    def test_a_windows_console_without_escape_codes_gets_none(self):
        self.assertEqual(self.colour(Tty(), {}, win=True, vt=False), ("", ""))
        self.assertEqual(self.colour(Tty(), {}, win=True, vt=True), ("\033[36m", "\033[0m"))


if __name__ == "__main__":
    unittest.main()
