"""#5948: a masked key ("sk-strata-********") copied as it is fails with "wrong or invalid API key", and a local launch that has a key on
said the key was "not needed on this computer" although the server checks it there too. `strata connect` now says how to print the key when
it hid it, and the Access step says that apps on this computer must send a key that is on (coordinator-approved wording, 11-16)."""

from __future__ import annotations

import io
import unittest
from contextlib import redirect_stdout
from unittest import mock

from strata_cli.cli import main
from tests.terminal.test_launch_tui import Base, press, text_of

KEY = "sk-strata-0123456789abcdef"
HINT = "The key is hidden here; add --show-key to print it."


class ConnectSaysHowToSeeTheKey(unittest.TestCase):
    def connect(self, *flags):
        rendered = {"harness": "opencode", "files": [{"path": "/tmp/x.json", "text": f'{{"apiKey": "{KEY}"}}', "merge": False, "kind": "json"}],
                    "env": {}, "limits": [], "run": None, "install": None, "tested": None, "title": "OpenCode", "speaks": "OpenAI", "status": "ok"}
        out = io.StringIO()
        with mock.patch("strata_cli.launch.harness.render", return_value=rendered), redirect_stdout(out):
            main(["connect", "opencode", "--url", "http://127.0.0.1:1", *flags])
        return out.getvalue()

    def test_a_hidden_key_says_how_to_print_it(self):
        text = self.connect("--key", KEY)
        self.assertNotIn(KEY, text)
        self.assertIn(HINT, text)

    def test_a_shown_key_has_no_hint(self):
        text = self.connect("--key", KEY, "--show-key")
        self.assertIn(KEY, text)
        self.assertNotIn(HINT, text)

    def test_no_key_no_hint(self):
        with mock.patch("strata_cli.launch.cmd.launch_info", return_value={}):
            self.assertNotIn(HINT, self.connect())


class AccessStepWords(Base):
    def reach_access(self):
        return ("enter",) * 7 + ("n",)

    def test_a_local_launch_with_a_key_says_apps_must_send_it(self):
        async def case(app, pilot, ctx):
            await press(pilot, *self.reach_access())
            self.assertEqual(app.current, "access")
            ctx.settings.access, ctx.settings.api_key = "local", KEY
            app.screen.paint()
            await press(pilot)
            text = text_of(app)
            self.assertIn("(apps on this computer must send it too; t = no key)", text)
            self.assertNotIn("(not needed on this computer)", text)
        self.run_app(case)

    def test_a_local_launch_without_a_key_keeps_its_words(self):
        async def case(app, pilot, ctx):
            await press(pilot, *self.reach_access())
            ctx.settings.access, ctx.settings.api_key = "local", None
            app.screen.paint()
            await press(pilot)
            self.assertIn("(not needed on this computer)", text_of(app))
        self.run_app(case)


if __name__ == "__main__":
    unittest.main()
