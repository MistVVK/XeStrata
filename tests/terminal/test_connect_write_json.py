"""connect --write --output json runs the write and reports each file's result (the JSON path used to return before the write block)."""

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

from strata_cli.cli import main


class ConnectWriteJsonTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="strata-connect-write-")
        self.addCleanup(self.temp.cleanup)
        self.dir = Path(self.temp.name)

    def _rendered(self, text="keep = true\n"):
        return {"files": [{"path": str(self.dir / "app.toml"), "text": text, "merge": False, "kind": "toml"}],
                "env": {}, "limits": []}

    def _cli(self, *flags):
        with mock.patch("strata_cli.launch.harness.render", return_value=self._rendered()):
            out = io.StringIO()
            with redirect_stdout(out), mock.patch("sys.stdin", io.StringIO("")):
                code = main(["connect", "opencode", "--url", "http://127.0.0.1:1", "--output", "json", *flags])
        return code, json.loads(out.getvalue())

    def test_write_with_json_creates_the_file_and_reports_it(self):
        code, payload = self._cli("--write", "--yes")
        self.assertEqual((code, payload["ok"], payload["write"]["written"]),
                         (0, True, [{"path": str(self.dir / "app.toml"), "backup": None, "action": "created"}]))
        self.assertEqual((self.dir / "app.toml").read_text(), "keep = true\n")

    def test_write_with_json_replaces_and_keeps_a_backup(self):
        target = self.dir / "app.toml"
        target.write_text("old = true\n")
        code, payload = self._cli("--write", "--yes")
        row = payload["write"]["written"][0]
        self.assertEqual((code, row["action"]), (0, "replaced"))
        self.assertEqual(Path(row["backup"]).read_text(), "old = true\n")
        self.assertEqual(target.read_text(), "keep = true\n")

    def test_write_with_json_without_yes_still_refuses_a_pipe(self):
        code, payload = self._cli("--write")
        self.assertEqual((code, payload["ok"], payload["error"]["code"]), (2, False, 2))

    def test_plain_words_output_does_not_write(self):
        with mock.patch("strata_cli.launch.harness.render", return_value=self._rendered()):
            out = io.StringIO()
            with redirect_stdout(out), mock.patch("sys.stdin", io.StringIO("")):
                code = main(["connect", "opencode", "--url", "http://127.0.0.1:1"])
        self.assertEqual(code, 0)
        self.assertTrue(out.getvalue().lstrip().startswith("OpenCode"))
        self.assertFalse((self.dir / "app.toml").exists())


if __name__ == "__main__":
    unittest.main()
