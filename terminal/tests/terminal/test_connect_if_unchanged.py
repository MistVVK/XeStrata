"""connect --write --if-unchanged leaves a file that already holds the rendered content alone: no write, no backup."""

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

from strata_cli.cli import main


class ConnectIfUnchangedTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="strata-connect-unchanged-")
        self.addCleanup(self.temp.cleanup)
        self.dir = Path(self.temp.name)
        self.target = self.dir / "app.toml"

    def _rendered(self, text="keep = true\n", *, merge=False, kind="toml", path=None):
        return {"files": [{"path": str(path or self.target), "text": text, "merge": merge, "kind": kind}],
                "env": {}, "limits": []}

    def _cli(self, rendered, *flags):
        with mock.patch("strata_cli.launch.harness.render", return_value=rendered):
            out = io.StringIO()
            with redirect_stdout(out), mock.patch("sys.stdin", io.StringIO("")):
                code = main(["connect", "opencode", "--url", "http://127.0.0.1:1", "--write", "--yes", *flags])
        return code, out.getvalue()

    def _backups(self):
        return list(self.dir.glob("*.strata-bak-*"))

    def test_first_write_creates_even_with_the_flag(self):
        code, words = self._cli(self._rendered(), "--if-unchanged")
        self.assertEqual(code, 0)
        self.assertIn(f"Created {self.target}", words)
        self.assertEqual(self.target.read_text(), "keep = true\n")

    def test_identical_rerun_with_the_flag_writes_and_backs_up_nothing(self):
        self._cli(self._rendered())                          # a plain first write
        code, words = self._cli(self._rendered(), "--if-unchanged")
        self.assertEqual(code, 0)
        self.assertIn(f"Unchanged {self.target}", words)
        self.assertEqual(self._backups(), [])
        self.assertEqual(self.target.read_text(), "keep = true\n")

    def test_without_the_flag_an_identical_rerun_still_backs_up(self):
        self._cli(self._rendered())
        code, words = self._cli(self._rendered())           # the control: the old behaviour stands unasked
        self.assertEqual(code, 0)
        self.assertIn(f"Replaced {self.target}", words)
        self.assertEqual(len(self._backups()), 1)

    def test_changed_content_still_replaces_with_a_backup(self):
        self.target.write_text("old = true\n")
        code, words = self._cli(self._rendered(), "--if-unchanged")
        self.assertEqual(code, 0)
        self.assertIn(f"Replaced {self.target}", words)
        self.assertEqual(len(self._backups()), 1)
        self.assertEqual(self.target.read_text(), "keep = true\n")

    def test_a_merged_write_that_reproduces_the_file_is_unchanged(self):
        # the comparison runs on the FINAL text, after the merge: merging the same keys back into a file
        # that already has them changes nothing, and must not create a backup
        config = self.dir / "settings.json"
        rendered = json.dumps({"provider": {"strata": {"api": "http://x"}}}, indent=2)
        code, _ = self._cli(self._rendered(rendered, merge=True, kind="json", path=config))
        self.assertEqual(code, 0)
        self.assertTrue(config.is_file())
        code, words = self._cli(self._rendered(rendered, merge=True, kind="json", path=config), "--if-unchanged")
        self.assertEqual(code, 0)
        self.assertIn(f"Unchanged {config}", words)
        self.assertEqual(self._backups(), [])


if __name__ == "__main__":
    unittest.main()
