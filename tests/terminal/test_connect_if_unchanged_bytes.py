"""#5957 follow-up: --if-unchanged compares the bytes the write would produce (line ends and a BOM count), and a file it leaves alone is
still made private (it can hold the API key), like a written one. JSON mode honours the flag through the shared --write block."""

import io
import json
import os
import stat
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

from strata_cli.cli import main


class IfUnchangedBytes(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(prefix="strata-connect-bytes-")
        self.addCleanup(temp.cleanup)
        self.dir = Path(temp.name)
        self.target = self.dir / "app.toml"

    def cli(self, *flags, json_out=False):
        rendered = {"files": [{"path": str(self.target), "text": "keep = true\n", "merge": False, "kind": "toml"}], "env": {}, "limits": []}
        out = io.StringIO()
        with mock.patch("strata_cli.launch.harness.render", return_value=rendered), redirect_stdout(out), \
                mock.patch("sys.stdin", io.StringIO("")):
            code = main(["connect", "opencode", "--url", "http://127.0.0.1:1", "--write", "--yes", *flags]
                        + (["--output", "json"] if json_out else []))
        return code, out.getvalue()

    def backups(self):
        return list(self.dir.glob("*.strata-bak-*"))

    def same_bytes(self) -> bytes:
        return ("keep = true" + os.linesep).encode("utf-8")

    @unittest.skipIf(os.name == "nt", "POSIX modes")
    def test_an_unchanged_file_is_still_made_private(self):
        self.target.write_bytes(self.same_bytes())
        self.target.chmod(0o644)
        code, words = self.cli("--if-unchanged")
        self.assertEqual(code, 0)
        self.assertIn(f"Unchanged {self.target}", words)
        self.assertEqual(stat.S_IMODE(self.target.stat().st_mode), 0o600)
        self.assertEqual(self.backups(), [])

    def test_other_line_ends_are_not_unchanged(self):
        other = b"keep = true\n" if os.linesep == "\r\n" else b"keep = true\r\n"
        self.target.write_bytes(other)
        code, words = self.cli("--if-unchanged")
        self.assertEqual(code, 0)
        self.assertIn(f"Replaced {self.target}", words)
        self.assertEqual(self.target.read_bytes(), self.same_bytes())
        self.assertEqual(len(self.backups()), 1)

    def test_a_bom_is_not_unchanged(self):
        self.target.write_bytes(b"\xef\xbb\xbf" + self.same_bytes())
        code, words = self.cli("--if-unchanged")
        self.assertIn(f"Replaced {self.target}", words)
        self.assertEqual(self.target.read_bytes(), self.same_bytes())

    def test_json_mode_reports_unchanged_too(self):
        self.target.write_bytes(self.same_bytes())
        code, out = self.cli("--if-unchanged", json_out=True)
        payload = json.loads(out)
        self.assertEqual((code, payload["ok"]), (0, True))
        self.assertEqual(payload["write"]["written"], [{"path": str(self.target), "backup": None, "action": "unchanged"}])
        self.assertEqual(self.backups(), [])


if __name__ == "__main__":
    unittest.main()
