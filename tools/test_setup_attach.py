# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Tests for a start beside a server already on the model's port (setup.py's attach_running: the packages' systemd
service, or a start in another terminal): the same model is used, not started twice; another model of XeStrata's stops
the start with how to free the port; anything else is left to the server's own message.  No GPU, no engine.

    python -m unittest tools.test_setup_attach
"""
from __future__ import annotations

import http.server
import json
import sys
import threading
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import setup  # noqa: E402


class Server(http.server.BaseHTTPRequestHandler):
    """GET /v1/models as a server answers it: `body` (None: an error)."""
    body: dict | None = None

    def do_GET(self):
        if self.body is None:
            self.send_error(404)
            return
        data = json.dumps(self.body).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args):
        pass


class Attach(unittest.TestCase):
    def serve(self, body):
        handler = type("H", (Server,), {"body": body})
        httpd = http.server.HTTPServer(("127.0.0.1", 0), handler)
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        self.addCleanup(httpd.server_close)
        self.addCleanup(httpd.shutdown)
        return httpd.server_address[1]

    def setUp(self):
        for p in (mock.patch.object(setup, "say"), mock.patch.object(setup, "ok"),
                  mock.patch.object(setup, "systemd_service_active", return_value=True)):
            p.start()
            self.addCleanup(p.stop)

    def test_the_same_model_is_used(self):
        port = self.serve({"object": "list", "data": [{"id": "ornith-q4", "owned_by": "xestrata"}]})
        self.assertTrue(setup.attach_running({"model_name": "Ornith-Q4"}, port, False))

    def test_another_model_stops_the_start(self):
        port = self.serve({"object": "list", "data": [{"id": "other", "owned_by": "xestrata"}]})
        with mock.patch.object(setup, "fail", side_effect=SystemExit(1)) as fail:
            with self.assertRaises(SystemExit):
                setup.attach_running({"model_name": "ornith-q4"}, port, False)
        self.assertIn("systemctl --user stop xestrata", fail.call_args.args[1])

    def test_another_program_or_nothing_is_left_to_the_server(self):
        port = self.serve({"object": "list", "data": [{"id": "x", "owned_by": "someone"}]})
        self.assertFalse(setup.attach_running({"model_name": "x"}, port, False))
        port = self.serve(None)
        self.assertFalse(setup.attach_running({"model_name": "x"}, port, False))
        httpd = http.server.HTTPServer(("127.0.0.1", 0), Server)
        free = httpd.server_address[1]
        httpd.server_close()
        self.assertFalse(setup.attach_running({"model_name": "x"}, free, False))


if __name__ == "__main__":
    unittest.main()
