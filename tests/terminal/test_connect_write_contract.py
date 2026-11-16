"""connect --write: one order and the same side effects in both output modes (#5955). --check runs first and a failed check writes
nothing (exit 1); a declined or ended prompt writes nothing (exit 0); a file that cannot be written ends the write there, the files
before it stay written, the result names each file, exit 1."""

import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from strata_cli.cli import main

FAILED_CHECK = [SimpleNamespace(name="prompt", ok=True, detail="ok", seconds=0.1),
                SimpleNamespace(name="tool call", ok=False, detail="no tool call came back", seconds=0.2)]


class ConnectWriteContract(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(prefix="strata-connect-contract-")
        self.addCleanup(temp.cleanup)
        self.dir = Path(temp.name)
        self.first = self.dir / "app.toml"
        blocker = self.dir / "blocker"
        blocker.write_text("a file where a folder should be\n")
        self.second = blocker / "nested" / "second.toml"          # its folder cannot be made: the write fails with an OSError
        self.third = self.dir / "third.toml"

    def rendered(self, *paths):
        return {"files": [{"path": str(p), "text": f"name = '{p.name}'\n", "merge": False, "kind": "toml"} for p in paths],
                "env": {}, "limits": []}

    def cli(self, rendered, *flags, json_out=True, stdin="", tty=False, checks=None):
        args = ["connect", "opencode", "--url", "http://127.0.0.1:1", *flags] + (["--output", "json"] if json_out else [])
        out, err = io.StringIO(), io.StringIO()
        fake_in = io.StringIO(stdin)
        fake_in.isatty = lambda: tty
        with mock.patch("strata_cli.launch.harness.render", return_value=rendered), \
                mock.patch("strata_cli.launch.check.run_checks", return_value=checks or []), \
                redirect_stdout(out), redirect_stderr(err), mock.patch("sys.stdin", fake_in):
            code = main(args)
        return code, (json.loads(out.getvalue()) if json_out else out.getvalue()), err.getvalue()

    # ---- a failed check writes nothing, in both modes
    def test_json_a_failed_check_writes_nothing(self):
        code, payload, _ = self.cli(self.rendered(self.first), "--check", "--write", "--yes", checks=FAILED_CHECK)
        self.assertEqual((code, payload["ok"], payload["error"]["code"]), (1, False, 1))
        self.assertEqual(payload["write"], {"written": []})
        self.assertFalse(self.first.exists())

    def test_words_a_failed_check_writes_nothing(self):
        code, _, _ = self.cli(self.rendered(self.first), "--check", "--write", "--yes", json_out=False, checks=FAILED_CHECK)
        self.assertEqual(code, 1)
        self.assertFalse(self.first.exists())

    def test_json_a_passed_check_then_writes(self):
        passed = [SimpleNamespace(name="prompt", ok=True, detail="ok", seconds=0.1)]
        code, payload, _ = self.cli(self.rendered(self.first), "--check", "--write", "--yes", checks=passed)
        self.assertEqual((code, payload["ok"]), (0, True))
        self.assertEqual([r["action"] for r in payload["write"]["written"]], ["created"])
        self.assertTrue(self.first.is_file())

    # ---- a declined or ended prompt writes nothing, in both modes
    def test_json_an_ended_prompt_is_a_no(self):
        code, payload, err = self.cli(self.rendered(self.first), "--write", stdin="", tty=True)
        self.assertEqual((code, payload["ok"], payload["write"]["written"]), (0, True, []))
        self.assertIn("Write these files now?", err)                    # the question goes to stderr: stdout stays one JSON object
        self.assertFalse(self.first.exists())

    def test_words_a_declined_prompt_writes_nothing(self):
        code, words, _ = self.cli(self.rendered(self.first), "--write", json_out=False, stdin="n\n", tty=True)
        self.assertEqual(code, 0)
        self.assertIn("Nothing was written.", words)
        self.assertFalse(self.first.exists())

    def test_both_modes_refuse_a_pipe_without_yes(self):
        code, payload, _ = self.cli(self.rendered(self.first), "--write")
        self.assertEqual((code, payload["error"]["code"]), (2, 2))
        code, _, err = self.cli(self.rendered(self.first), "--write", json_out=False)
        self.assertEqual(code, 2)
        self.assertIn("--write asks first", err)
        self.assertFalse(self.first.exists())

    # ---- a file that cannot be written
    def test_json_a_failed_second_file_is_reported_with_the_files_around_it(self):
        code, payload, _ = self.cli(self.rendered(self.first, self.second, self.third), "--write", "--yes")
        self.assertEqual((code, payload["ok"], payload["error"]["code"]), (1, False, 1))
        rows = payload["write"]["written"]
        self.assertEqual([r["action"] for r in rows], ["created", "failed", "not_written"])
        self.assertEqual([r["path"] for r in rows], [str(self.first), str(self.second), str(self.third)])
        self.assertTrue(rows[1]["error"])
        self.assertIn(str(self.second), payload["error"]["message"])
        self.assertTrue(self.first.is_file())
        self.assertFalse(self.third.exists())

    def test_words_a_failed_second_file_says_so_and_exits_1(self):
        code, words, err = self.cli(self.rendered(self.first, self.second, self.third), "--write", "--yes", json_out=False)
        self.assertEqual(code, 1)
        self.assertIn(f"Created {self.first}", words)
        self.assertIn(f"Could not write {self.second}", err)
        self.assertNotIn("Traceback", err)
        self.assertTrue(self.first.is_file())
        self.assertFalse(self.third.exists())


if __name__ == "__main__":
    unittest.main()
