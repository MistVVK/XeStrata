# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Two small setup paths: a loopback check must not send its API key to a proxy, and importing the person's Anima
file must not leave a cut one under a name the page shows.

    python -m unittest tools.test_setup_misc
"""
from __future__ import annotations

import contextlib
import io
import json
import os
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import setup  # noqa: E402


class ServerOnPort(unittest.TestCase):
    def test_a_local_server_is_asked_directly_even_with_a_proxy(self):
        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                body = json.dumps({"data": [{"id": "qwen", "owned_by": "xestrata"}]}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args):
                pass

        srv = HTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=srv.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(lambda: (srv.shutdown(), srv.server_close(), thread.join(2)))
        with mock.patch.dict(os.environ, {"http_proxy": "http://127.0.0.1:1", "HTTP_PROXY": "http://127.0.0.1:1",
                                          "no_proxy": "", "NO_PROXY": ""}):
            got = setup.server_on_port(srv.server_port, "secret")
        self.assertEqual(got, ("xestrata", ["qwen"]))


class ImportAnima(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = Path(tmp.name)
        self.models = self.dir / "models"
        self.models.mkdir()
        self.src = self.dir / "dit.safetensors"
        self.src.write_bytes(b"the whole file")
        qi = SimpleNamespace(ANIMA_DIR="anima", ANIMA_USER="anima-user-", IMAGE_MODELS={}, FILES={},
                             anima_header=lambda path: None,
                             local_path=lambda models_dir, k: Path(k),
                             anima_user_dits=lambda models_dir: ["anima-user-dit"])
        patch = mock.patch.object(setup, "_qimage", return_value=qi)
        patch.start()
        self.addCleanup(patch.stop)

    def import_it(self):
        with contextlib.redirect_stdout(io.StringIO()):
            return setup.import_anima(self.src, self.models)

    def test_a_complete_file_gets_its_final_name(self):
        self.assertEqual(self.import_it(), "anima-user-dit")
        self.assertEqual((self.models / "anima" / "dit.safetensors").read_bytes(), b"the whole file")
        self.assertFalse(list((self.models / "anima").glob("*.part")))

    def test_a_failed_copy_leaves_no_cut_file(self):
        def broken(src, dst):
            dst.write(b"cut")
            raise OSError("disk full")

        with mock.patch.object(setup.shutil, "copyfileobj", broken):
            with self.assertRaises(OSError):
                self.import_it()
        folder = self.models / "anima"
        self.assertEqual(list(folder.iterdir()), [])

    def test_an_existing_file_is_not_overwritten(self):
        folder = self.models / "anima"
        folder.mkdir()
        (folder / "dit.safetensors").write_bytes(b"the person's own file")
        with self.assertRaises(FileExistsError):
            self.import_it()
        self.assertEqual((folder / "dit.safetensors").read_bytes(), b"the person's own file")


if __name__ == "__main__":
    unittest.main()
