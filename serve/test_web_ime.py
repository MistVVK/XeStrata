"""The page's text fields leave an IME's keys to the IME (#5071): the Enter that commits a Japanese, Chinese or Korean conversion
does not send the message (Safari sent it: its commit Enter says isComposing false), does not end a chat's rename, and an Escape
that cancels a conversion does not close the answer editor.

The rule runs in Node (serve/test_web_ime.js). These checks read the sources for where it must be used."""
from __future__ import annotations

import re
import shutil
import subprocess
import unittest
from pathlib import Path

WEB = Path(__file__).parent / "web"


@unittest.skipUnless(shutil.which("node"), "node is not installed")
class ImeInNode(unittest.TestCase):
    def test_each_browsers_order_of_events(self):
        r = subprocess.run([shutil.which("node"), str(Path(__file__).with_name("test_web_ime.js"))],
                           capture_output=True, text=True, timeout=60)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)


class TheFieldsAskTheGuardFirst(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.html = (WEB / "index.html").read_text(encoding="utf-8")
        cls.app = (WEB / "app.js").read_text(encoding="utf-8")
        cls.ui = (WEB / "chats-ui.js").read_text(encoding="utf-8")

    def test_the_guard_loads_before_the_scripts_that_use_it(self):
        scripts = re.findall(r'<script src="web/([^"]+)"></script>', self.html)
        self.assertIn("ime.js", scripts)
        self.assertLess(scripts.index("ime.js"), scripts.index("chats-ui.js"))
        self.assertLess(scripts.index("ime.js"), scripts.index("app.js"))

    def test_the_composer(self):
        self.assertIn('const composerIme = StrataIme.watch($("input"));', self.app)
        self.assertRegex(self.app, r'\$\("input"\)\.addEventListener\("keydown", \(e\) => \{\s*if \(composerIme\(e\)\) return;')

    def test_the_rename_field_and_the_answer_editor(self):
        self.assertEqual(self.ui.count("const imeKey = StrataIme.watch(input);"), 2)
        self.assertRegex(self.ui, r'input\.addEventListener\("keydown", \(e\) => \{\s*if \(imeKey\(e\)\) return;')
        self.assertRegex(self.ui, r'input\.onkeydown = \(e\) => \{\s*if \(imeKey\(e\)\) return;')

    def test_no_text_field_acts_on_enter_without_the_guard(self):
        # every keydown handler of the page that looks at Enter asks the guard first
        for name, src in (("app.js", self.app), ("chats-ui.js", self.ui)):
            for m in re.finditer(r'(?:addEventListener\("keydown", \(e\) => \{|onkeydown = \(e\) => \{)(.*?)\n\s*\}', src, re.S):
                if '"Enter"' in m.group(1):
                    with self.subTest(script=name, at=src[:m.start()].count("\n") + 1):
                        self.assertRegex(m.group(1), r'^\s*if \((?:composerIme|imeKey)\(e\)\) return;')


if __name__ == "__main__":
    unittest.main()
