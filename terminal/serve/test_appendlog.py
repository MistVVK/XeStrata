"""#5175: the engine's, the image encoder's and the image generator's lines all stay in the one server log on Windows."""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from serve import appendlog


def child(handle, text: str):
    """Run a child that writes `text` to its stderr, the file `handle` (the way the engine and the encoder are started)."""
    subprocess.run([sys.executable, "-c", f"import sys; sys.stderr.write({text!r}); sys.stderr.flush()"], stderr=handle, check=True)


class SharedLog(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="strata-appendlog-")
        self.addCleanup(self.temp.cleanup)
        self.log = Path(self.temp.name) / "server.log"
        self.log.write_text("previous session\n", encoding="utf-8")

    def test_children_writing_through_separate_handles_keep_every_line_in_order(self):
        reader = appendlog.open_append(self.log)        # the image encoder's: opened when the server started
        engine = appendlog.open_append(self.log)        # the engine's: opened at each engine start, later
        try:
            engine.write("engine started\n")
            engine.flush()
            child(engine, "engine 1\n")
            child(reader, "reader A\n")                 # with plain "a" handles this landed right after "previous session"
            child(engine, "engine 2\n")
            child(reader, "reader B\n")
            engine.write("engine 3\n")
            engine.flush()
        finally:
            reader.close()
            engine.close()
        self.assertEqual(self.log.read_text(encoding="utf-8").splitlines(),
                         ["previous session", "engine started", "engine 1", "reader A", "engine 2", "reader B", "engine 3"])

    def test_the_file_is_created_when_it_is_not_there(self):
        fresh = Path(self.temp.name) / "new.log"
        handle = appendlog.open_append(fresh)
        handle.write("first\n")
        handle.close()
        self.assertEqual(fresh.read_text(encoding="utf-8"), "first\n")

    def test_text_is_utf8(self):
        handle = appendlog.open_append(self.log)
        handle.write("nächste Zeile ✓\n")
        handle.close()
        self.assertEqual(self.log.read_text(encoding="utf-8").splitlines()[-1], "nächste Zeile ✓")


class ServerUsesIt(unittest.TestCase):
    def test_the_server_opens_the_logs_its_children_write_to_with_it(self):
        text = (Path(__file__).with_name("server.py")).read_text(encoding="utf-8")
        self.assertNotIn('self.log = open(log, "a"', text)
        self.assertNotIn('log=open(cfg["log"], "a"', text)
        self.assertGreaterEqual(text.count("appendlog.open_append("), 3)


if __name__ == "__main__":
    unittest.main()
