# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Tests for setup.py's warm-up against a server with an API key (#19): the requests carry the key
the server takes, and a refusal ends the warm-up at once instead of after 15 minutes.  No server is started.

    python -m unittest tools.test_setup_warmup
"""
from __future__ import annotations

import io
import json
import os
import sys
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import setup  # noqa: E402


class FakeProc:
    def poll(self):
        return None


class WarmUp(unittest.TestCase):
    def run_warm_up(self, cfg, env=None, refuse=False):
        """warm_up on a config; returns the Authorization header of each request and what it said."""
        seen: list[tuple[str, str | None]] = []

        def urlopen(req, timeout=None):
            url, given = (req, None) if isinstance(req, str) else (req.full_url, req.get_header("Authorization"))
            seen.append((url.split("/", 3)[3], given))
            if refuse:
                raise urllib.error.HTTPError(url, 401, "Unauthorized", None, None)  # type: ignore[arg-type]
            return io.BytesIO(json.dumps({"data": [], "choices": []}).encode())

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "xestrata-test.json"
            path.write_text(json.dumps(cfg), encoding="utf-8")
            with mock.patch.dict(os.environ, env or {}), \
                    mock.patch.object(setup, "STATE", Path(tmp)), \
                    mock.patch.object(setup.subprocess, "Popen", return_value=FakeProc()), \
                    mock.patch.object(setup, "end_process_group"), \
                    mock.patch.object(setup.urllib.request, "urlopen", side_effect=urlopen), \
                    mock.patch.object(setup, "ok") as ok, mock.patch.object(setup, "warn") as warn, \
                    mock.patch.object(setup, "say"):
                if not env:
                    os.environ.pop("STRATA_API_KEY", None)
                setup.warm_up(path)
        return seen, ok, warn

    def test_no_key(self):
        seen, ok, _ = self.run_warm_up({})
        self.assertEqual([a for _, a in seen], [None, None])
        ok.assert_called_once()

    def test_the_configs_first_key(self):
        seen, ok, _ = self.run_warm_up({"api_key": " k1 , k2"})
        self.assertEqual([u for u, _ in seen], ["v1/models", "v1/chat/completions"])
        self.assertEqual([a for _, a in seen], ["Bearer k1", "Bearer k1"])
        ok.assert_called_once()

    def test_a_list_of_keys(self):
        # a list is the way to give a key with a comma in it
        seen, _, _ = self.run_warm_up({"api_key": ["a,b", "c"]})
        self.assertEqual(seen[0][1], "Bearer a,b")

    def test_the_environment_before_the_config(self):
        seen, _, _ = self.run_warm_up({"api_key": "cfg"}, env={"STRATA_API_KEY": "env"})
        self.assertEqual(seen[0][1], "Bearer env")

    def test_a_refusal_ends_it_at_once(self):
        seen, ok, warn = self.run_warm_up({"api_key": "wrong"}, refuse=True)
        self.assertEqual(len(seen), 1)
        ok.assert_not_called()
        self.assertIn("401", warn.call_args.args[0])


if __name__ == "__main__":
    unittest.main()
