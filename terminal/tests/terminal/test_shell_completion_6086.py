"""`strata complete` prints a tab-completion script for bash, zsh and fish, built from the CLI's own parser (#6086, PR #6087's idea
without the optional `shtab` package).  The scripts name every command the installed command has; the bash one is run in a real bash."""
from __future__ import annotations

import argparse
import contextlib
import io
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from strata_cli import cli, completion  # noqa: E402


def _top_level():
    sub = [a for a in cli.parser()._actions if isinstance(a, argparse._SubParsersAction)][0]
    return sorted(sub.choices)


def _run(shell: str) -> str:
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        code = cli.main(["complete", "--shell", shell])
    assert code == 0
    return out.getvalue()


def _bash_complete(script: str, words: list[str]) -> list[str]:
    """What bash offers after `words` (the last one is the word being completed)."""
    with tempfile.TemporaryDirectory() as tmp:
        f = Path(tmp) / "strata.bash"
        f.write_text(script, encoding="utf-8", newline="\n")
        quoted = " ".join("'" + w + "'" for w in words)
        cmd = f"source '{f.as_posix()}'; COMP_WORDS=({quoted}); COMP_CWORD={len(words) - 1}; _strata; printf '%s\\n' \"${{COMPREPLY[@]}}\""
        r = subprocess.run([shutil.which("bash"), "-c", cmd], capture_output=True, text=True, timeout=30)
        assert r.returncode == 0, r.stderr
        return r.stdout.split()


class ShellCompletion(unittest.TestCase):
    def test_every_script_names_every_command(self):
        for shell in completion.SHELLS:
            script = _run(shell)
            self.assertTrue(script.strip(), shell)
            for command in _top_level():
                self.assertIn(command, script, f"{shell}: {command}")

    def test_the_command_is_in_strata_help(self):
        self.assertIn("complete", _top_level())

    def test_the_scripts_are_plain_text_the_shell_can_read(self):
        for shell in completion.SHELLS:
            script = _run(shell)
            self.assertNotIn("\r", script)
            self.assertTrue(script.endswith("\n"))
        self.assertIn("complete -F _strata strata", _run("bash"))
        self.assertIn("bashcompinit", _run("zsh"))
        self.assertIn("complete -c strata", _run("fish"))

    def test_zsh_and_fish_scripts_parse_where_those_shells_exist(self):
        for shell in ("zsh", "fish"):
            binary = shutil.which(shell)
            if not binary:
                continue
            with tempfile.TemporaryDirectory() as tmp:
                f = Path(tmp) / f"strata.{shell}"
                f.write_text(_run(shell), encoding="utf-8", newline="\n")
                r = subprocess.run([binary, "-n", str(f)], capture_output=True, text=True, timeout=30)
            self.assertEqual(r.returncode, 0, f"{shell}: {r.stderr}")

    @unittest.skipUnless(shutil.which("bash"), "no bash on this machine")
    def test_bash_offers_commands_sub_commands_options_and_choices(self):
        script = _run("bash")
        with tempfile.TemporaryDirectory() as tmp:
            f = Path(tmp) / "strata.bash"
            f.write_text(script, encoding="utf-8", newline="\n")
            syntax = subprocess.run([shutil.which("bash"), "-n", f.as_posix()], capture_output=True, text=True)
        self.assertEqual(syntax.returncode, 0, syntax.stderr)
        top = _bash_complete(script, ["strata", ""])
        for command in ("launch", "models", "profile", "complete", "bench"):
            self.assertIn(command, top)
        self.assertEqual(_bash_complete(script, ["strata", "lau"]), ["launch"])
        self.assertIn("--port", _bash_complete(script, ["strata", "launch", "--po"]))
        self.assertIn("explain", _bash_complete(script, ["strata", "models", ""]))
        self.assertEqual(sorted(_bash_complete(script, ["strata", "complete", "--shell", ""])), ["bash", "fish", "zsh"])
        # an option inside a sub-command does not offer the top-level commands
        self.assertNotIn("launch", _bash_complete(script, ["strata", "models", "list", "--"]))


if __name__ == "__main__":
    unittest.main()
