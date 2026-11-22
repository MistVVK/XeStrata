"""Regression tests for Track 5A MCP/Control fixes in `terminal/tools/strata_mcp.py`:
- Issue #6518: `files_here()` falls back to `S.file_pin(S.model_hf(fam, model) + path.name)` for families without a `sha256` table (`qwen`, `swift`, `coder`, `nvfp4`) while keeping `S.fam_pin()` first, exact-size checks, `.part` rejection, and unknown-pin rejection.
- Issue #6475: `Strata.ports()` stops endlessly probing `127.0.0.1:8080` when Strata models use other ports or once 8080 answered as non-Strata, and honors `STRATA_MODEL_PORT` and `STRATA_NO_DEFAULT_PORT`.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from unittest import mock

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import strata_mcp as M  # noqa: E402


def _make_sparse(path: Path, size: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as fh:
        if size > 0:
            fh.seek(size - 1)
            fh.write(b"\0")


class FilesHere6518(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.S = M.Strata(HERE.parent).setup_module()
        if cls.S is None:
            raise unittest.SkipTest("setup.py could not be imported")

    def _pinned_expected(self, family: str, model: str, models_dir: Path, vision: bool) -> list[tuple[Path, int]]:
        S = self.S
        fam = S.FAMILIES[family]
        out = []
        for f in M.expected_files(S, family, model, models_dir, vision):
            p = Path(f["path"])
            sz = f["bytes"]
            if sz is None:
                pin = S.fam_pin(fam, p) if "sha256" in fam else S.file_pin(S.model_hf(fam, model) + p.name)
                self.assertIsNotNone(pin, f"missing pin for {family}/{model}/{p.name}")
                sz = pin[0]
            out.append((p, sz))
        return out

    def test_qwen_swift_coder_and_gyro_complete_missing_truncated_and_part(self):
        for family, model, vision in (
            ("qwen", "IQ3_S", True),
            ("qwen", "IQ3_S", False),
            ("swift", "IQ3_S", True),
            ("swift", "IQ3_XXS", False),
            ("coder", "IQ1_M", True),
            ("gyro", "Gyro-S", True),
            ("unsloth", "UD-IQ4_XS", False),
        ):
            with self.subTest(family=family, model=model, vision=vision), tempfile.TemporaryDirectory() as tmp:
                models_dir = Path(tmp) / "models"
                expected = self._pinned_expected(family, model, models_dir, vision)
                total_bytes = sum(sz for _, sz in expected)

                # 1. Missing all files -> complete=False, left_gb == total_gb
                res_missing = M.files_here(self.S, family, model, models_dir, vision)
                self.assertIsNotNone(res_missing)
                self.assertFalse(res_missing["complete"])
                self.assertEqual(res_missing["left_gb"], res_missing["total_gb"])
                self.assertEqual(res_missing["total_gb"], round(total_bytes / 1e9, 1))

                # 2. All files present at exact pinned size -> complete=True, left_gb == 0.0
                for p, sz in expected:
                    _make_sparse(p, sz)
                res_complete = M.files_here(self.S, family, model, models_dir, vision)
                self.assertIsNotNone(res_complete)
                self.assertTrue(res_complete["complete"])
                self.assertEqual(res_complete["left_gb"], 0.0)

                # 3. .part file next to largest shard -> rejected (complete=False, left_gb > 0)
                target_path, target_size = max(expected, key=lambda x: x[1])
                part_path = target_path.with_name(target_path.name + ".part")
                part_path.write_bytes(b"partial")
                res_part = M.files_here(self.S, family, model, models_dir, vision)
                self.assertIsNotNone(res_part)
                self.assertFalse(res_part["complete"])
                self.assertGreater(res_part["left_gb"], 0.0)
                part_path.unlink()

                # 4. Truncated file (1 byte short) -> rejected (complete=False, left_gb > 0)
                _make_sparse(target_path, max(0, target_size - 1))
                res_trunc = M.files_here(self.S, family, model, models_dir, vision)
                self.assertIsNotNone(res_trunc)
                self.assertFalse(res_trunc["complete"])
                self.assertGreater(res_trunc["left_gb"], 0.0)

    def test_unknown_pin_still_returns_none(self):
        with tempfile.TemporaryDirectory() as tmp:
            models_dir = Path(tmp) / "models"
            with mock.patch.object(self.S, "file_pin", return_value=None):
                self.assertIsNone(M.files_here(self.S, "qwen", "IQ3_S", models_dir, False))

    def test_fam_pin_takes_precedence_when_sha256_table_exists(self):
        with tempfile.TemporaryDirectory() as tmp:
            models_dir = Path(tmp) / "models"
            with mock.patch.object(self.S, "file_pin", side_effect=AssertionError("must not fall back when fam_pin succeeds")):
                res = M.files_here(self.S, "gyro", "Gyro-S", models_dir, False)
            self.assertIsNotNone(res)
            self.assertFalse(res["complete"])


class PortsNo8080Spam6475(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.strata = M.Strata(self.root, src=HERE.parent)
        self.tools = M.Tools(self.strata)

    def _write_cfg(self, tag: str, port: int) -> None:
        cfg = {
            "exe": sys.executable,
            "args": ["--max-context", "32768"],
            "port": port,
            "model_name": f"qwen3.8-flash-next-{tag}",
            "log": str(self.root / f"strata-{tag}.log"),
        }
        (self.root / f"strata-{tag}.json").write_text(json.dumps(cfg), encoding="utf-8")

    def test_ports_does_not_append_8080_when_configs_use_other_ports(self):
        self._write_cfg("q2_0", port=18500)
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("STRATA_MODEL_PORT", None)
            os.environ.pop("STRATA_NO_DEFAULT_PORT", None)
            self.assertEqual(self.strata.ports(None), [18500])

    def test_non_strata_listener_on_fallback_port_is_probed_only_once(self):
        hits = {"count": 0}

        class SpringBootMock(BaseHTTPRequestHandler):
            def do_GET(self):
                hits["count"] += 1
                self.send_response(404)
                self.end_headers()

            def log_message(self, *a):
                pass

        httpd = HTTPServer(("127.0.0.1", 0), SpringBootMock)
        fake_default = httpd.server_address[1]
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        self.addCleanup(httpd.server_close)
        self.addCleanup(httpd.shutdown)

        with mock.patch.object(M, "DEFAULT_PORT", fake_default), \
                mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("STRATA_MODEL_PORT", None)
            os.environ.pop("STRATA_NO_DEFAULT_PORT", None)
            # First status poll sees the non-Strata listener once and caches it.
            st1 = self.tools.strata_status(hardware=False)
            self.assertEqual(st1["ports_used_by_other_programs"][0]["port"], fake_default)
            self.assertEqual(hits["count"], 1)
            # Subsequent status polls do NOT hit the non-Strata listener again.
            st2 = self.tools.strata_status(hardware=False)
            st3 = self.tools.strata_status(hardware=False)
            self.assertEqual(hits["count"], 1, "must stop re-probing non-Strata fallback port")
            self.assertNotIn("ports_used_by_other_programs", st2)
            self.assertNotIn("ports_used_by_other_programs", st3)
            # Explicit port=... still probes it.
            st4 = self.tools.strata_status(port=fake_default, hardware=False)
            self.assertEqual(hits["count"], 2)
            self.assertEqual(st4["ports_used_by_other_programs"][0]["port"], fake_default)

    def test_env_contract_strata_model_port_and_no_default_port(self):
        with mock.patch.dict(os.environ, {"STRATA_NO_DEFAULT_PORT": "1"}, clear=False):
            os.environ.pop("STRATA_MODEL_PORT", None)
            self.assertEqual(self.strata.ports(None), [])

        with mock.patch.dict(os.environ, {"STRATA_MODEL_PORT": "18501", "STRATA_NO_DEFAULT_PORT": "1"}, clear=False):
            self.assertEqual(self.strata.default_port(), 18501)
            self.assertEqual(self.strata.ports(None), [18501])
            info = self.tools.strata_connect_info()
            self.assertIn(":18501", info["openai_base_url"])


if __name__ == "__main__":
    unittest.main()
