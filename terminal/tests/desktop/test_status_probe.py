"""A9 (0.2.16 retest): the app's first read of the PC waited about 2 s for every port nobody listens on (strata_status looks at the default
port and at the port of every prepared model, one after the other, each with the full HTTP timeout; a connect to a closed port on Windows
never answers "refused", it runs into the timeout). The status call now asks first whether anything accepts a connection (0.4 s), and looks at
the ports all at once. What it reports for a running Strata server is unchanged."""

from __future__ import annotations

import http.server
import importlib.util
import json
import socket
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def load_mcp():
    spec = importlib.util.spec_from_file_location("strata_mcp_probe_test", ROOT / "tools" / "strata_mcp.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


MCP = load_mcp()


class Health(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        body = json.dumps({"service": "strata", "model": "fake", "max_context": 4096, "images": False, "loaded": True}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass


def free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


class StatusProbe(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.strata = MCP.Strata(Path(self.tmp.name), ROOT)

    def serve_health(self):
        server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Health)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        return server.server_address[1]

    def test_a_listening_port_is_seen_and_a_closed_one_is_not(self):
        up = self.serve_health()
        self.assertTrue(MCP.port_accepts(up))
        self.assertFalse(MCP.port_accepts(free_port(), timeout=0.2))

    def test_probe_reports_a_running_server_as_before(self):
        info = self.strata.probe(self.serve_health(), deep=False)
        self.assertEqual((info or {}).get("model"), "fake")
        self.assertEqual(info.get("loaded"), True)

    def test_a_closed_port_never_reaches_the_http_call(self):
        with mock.patch.object(MCP, "http_json", side_effect=AssertionError("no HTTP to a closed port")):
            self.assertIsNone(self.strata.probe(free_port()))

    def test_probe_many_keeps_the_order_and_waits_once_not_per_port(self):
        up = self.serve_health()
        closed = [free_port() for _ in range(4)]
        ports = [closed[0], up, closed[1], closed[2], closed[3]]
        slow = mock.Mock(side_effect=lambda port, timeout=0.4: (time.sleep(0.5), port == up)[1])
        with mock.patch.object(MCP, "port_accepts", slow):
            started = time.perf_counter()
            found = self.strata.probe_many(ports, deep=False)
            took = time.perf_counter() - started
        self.assertEqual([i is not None for i in found], [False, True, False, False, False])
        self.assertLess(took, 1.4, f"five ports at 0.5 s each took {took:.2f} s: they were looked at one after the other")

    def test_strata_status_lists_the_running_server_and_ignores_closed_ports(self):
        up = self.serve_health()
        tools = MCP.Tools(self.strata)
        out = tools.strata_status(port=up, hardware=False)
        self.assertEqual([r["port"] for r in out["running"]], [up])
        self.assertTrue(out["summary"].startswith(f"Strata is running on port {up}"))
        out = tools.strata_status(port=free_port(), hardware=False)
        self.assertEqual(out["running"], [])
        self.assertTrue(out["summary"].startswith("Strata is not running"))


if __name__ == "__main__":
    unittest.main()
