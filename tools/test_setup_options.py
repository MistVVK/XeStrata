# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Tests for setup.py's table of options (OPTIONS): every option is one of the command line's, and setup's question
for it offers what --describe offers.  No GPU, no downloads, no prompts.

    python -m unittest tools.test_setup_options
"""
from __future__ import annotations

import argparse
import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import setup  # noqa: E402


def parser() -> argparse.ArgumentParser:
    """The command line as setup's main() makes it: its parser, kept from main() before anything is parsed."""
    made: list = []

    def keep(self, *args, **kwargs):
        made.append(self)
        raise SystemExit(0)

    with mock.patch.object(argparse.ArgumentParser, "parse_args", keep), \
            mock.patch.object(setup, "wsl_version", return_value=0), mock.patch.object(sys, "argv", ["setup.py"]):
        try:
            setup.main()
        except SystemExit:
            pass
    return made[0]


class Options(unittest.TestCase):
    def test_every_option_is_on_the_command_line(self):
        actions = {a.option_strings[0]: a for a in parser()._actions if a.option_strings}
        for o in setup.OPTIONS:
            if o["kind"] == "password":               # given in the environment, never on a command line
                self.assertTrue(o.get("env"), o["name"])
                continue
            name = "--" + o["name"]
            self.assertIn(name, actions)
            a = actions[name]
            self.assertEqual(o["help"], a.help, name)
            free = "choices" in o.get("cli", {}) and o["cli"]["choices"] is None    # any value (--host's addresses)
            if o["kind"] == "choice" and not o.get("gpu_choices") and not o.get("path_choice") and not free:
                for v, _ in o["choices"]:
                    if v:
                        self.assertIn(v, a.choices, f"{name} {v}")
            if o["kind"] == "flag":
                self.assertEqual(o.get("values"), list(a.choices) if a.choices else None, name)

    def test_the_questions_offer_what_describe_offers(self):
        with mock.patch.object(setup, "load_settings", return_value={}), mock.patch.object(setup, "_qimage") as qi:
            qi.return_value.SAM3_LICENSE = "a license"
            described = {o["name"]: o for o in setup.options_described()}
            for o in setup.OPTIONS:
                if not o.get("question"):
                    continue
                with mock.patch.object(setup, "ask", side_effect=lambda q, choices, default, yes: default) as ask:
                    got = setup.ask_option(o["name"], True)
                offered = ask.call_args.args[2]
                if o["kind"] == "flag":
                    values = o.get("values") or [True, False]
                    self.assertEqual(values[0 if described[o["name"]]["default"] else 1], got, o["name"])
                else:
                    self.assertEqual(described[o["name"]]["default"], offered, o["name"])
                    self.assertIn(offered, ask.call_args.args[1], o["name"])

    def test_the_ranges_are_checked(self):
        self.assertIsNone(setup.option_problem("port", 8095))
        self.assertEqual("--port takes a port number, 1 to 65535", setup.option_problem("port", 70000))
        self.assertIsNone(setup.option_problem("layer-split", "16,32"))
        self.assertIsNotNone(setup.option_problem("layer-split", "16;32"))


if __name__ == "__main__":
    unittest.main()
