"""#5786 (LawiK974): strata launch --draft-vocab and the Docker image's DRAFT_VOCAB carry setup's draft vocabulary choice. One list: setup.py's
DRAFT_VOCABS (cjk, en, cyrillic, fr, latin); an unknown name is refused before anything starts; a Qwen helper family passes it to setup, the
preset keeps it, and without it nothing is passed (setup's own pick from the computer's languages stays)."""

from __future__ import annotations

import argparse
import ast
import contextlib
import io
import unittest
from pathlib import Path

from strata_cli.launch import cmd
from strata_cli.launch import settings as S
from strata_cli.launch.runner import Runner
from tests.terminal.launch_support import HW_12, Home, catalog

ROOT = Path(__file__).resolve().parents[2]


def setup_vocabs() -> list[str]:
    tree = ast.parse((ROOT / "setup.py").read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "DRAFT_VOCABS" for t in node.targets):
            return [k.value for k in node.value.keys]
    raise AssertionError("setup.py has no DRAFT_VOCABS")


def parse(*argv):
    ap = argparse.ArgumentParser()
    cmd.add_parsers(ap.add_subparsers(dest="command"))
    return ap.parse_args(["launch", *argv])


class DraftVocab(unittest.TestCase):
    def test_the_launch_list_is_setups_list(self):
        self.assertEqual(list(S.DRAFT_VOCAB_MODES), setup_vocabs())

    def test_every_name_reaches_setup_and_the_preset_keeps_it(self):
        for name in S.DRAFT_VOCAB_MODES:
            with self.subTest(name=name), Home():
                s, _ = cmd.settings_from_args(parse("--model", "IQ3_XXS", "--draft-vocab", name), catalog=catalog(HW_12))
                self.assertEqual(s.draft_vocab, name)
                again = S.LaunchSettings.from_dict(s.to_dict())
                self.assertEqual(again.draft_vocab, name)
                args = Runner(again.copy(family="qwen", context=32768, kv="q4_0", vision="off"), catalog=catalog(HW_12)).setup_args()
                self.assertEqual(args[args.index("--draft-vocab") + 1], name)
                self.assertIn(f"--draft-vocab {name}", S.flags_text(again))

    def test_without_it_nothing_is_passed(self):
        with Home():
            args = Runner(S.LaunchSettings(family="qwen", model="IQ3_XXS", context=32768, kv="q4_0", vision="off"), catalog=catalog(HW_12)).setup_args()
        self.assertNotIn("--draft-vocab", args)

    def test_an_unknown_name_is_refused(self):
        err = io.StringIO()
        with contextlib.redirect_stderr(err), self.assertRaises(SystemExit):
            parse("--draft-vocab", "klingon")
        self.assertIn("klingon", err.getvalue())

    def test_the_docker_entrypoint_passes_both_settings(self):
        text = (ROOT.parent / "tools" / "docker" / "entrypoint.sh").read_text(encoding="utf-8")
        self.assertIn('[ -n "${DRAFT_VOCAB:-}" ] && { set -- "$@" --draft-vocab "$DRAFT_VOCAB"; have=1; }', text)
        self.assertIn('[ -n "${DRAFTING:-}" ] && { set -- "$@" --drafting "$DRAFTING"; have=1; }', text)
        docs = (ROOT.parent / "docs" / "DOCKER.md").read_text(encoding="utf-8")
        for name in S.DRAFT_VOCAB_MODES:
            self.assertIn(f"`{name}`", docs.split("| `DRAFT_VOCAB` |", 1)[1].split("\n", 1)[0])


if __name__ == "__main__":
    unittest.main()
