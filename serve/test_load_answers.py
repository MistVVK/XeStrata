"""PR #4307's honest-answers part, on the routes as they are (the pre-save ladder it is stacked on is not part of this):
POST /load and POST /v1/load answer an engine that dies or cannot be ended while it starts with a 503 and a JSON error,
instead of an unhandled exception that closed the connection with no answer."""
from __future__ import annotations

import json
import unittest
import urllib.error
import urllib.request
from pathlib import Path

from serve.frontend import ChatTemplate
from serve.server import ByteTokenizer, EngineDied, EngineStuck, Service, serve
from serve.test_server import CTX, UnloadableEngine

ROOT = Path(__file__).resolve().parents[1]


class FailingStart(UnloadableEngine):
    error: Exception = EngineDied("the engine exited before it was ready (exit code 1)")

    def restart(self):
        raise self.error


class LoadAnswers(unittest.TestCase):
    def setUp(self):
        tok = ByteTokenizer()
        self.engine = FailingStart(tok, "</think>\n\nok", max_context=CTX)
        self.svc = Service(self.engine, tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
        self.svc.unload()
        self.httpd = serve(self.svc, port=0)
        self.base = f"http://127.0.0.1:{self.httpd.server_address[1]}"

    def tearDown(self):
        self.httpd.shutdown()
        self.httpd.server_close()

    def post(self, path, body=None):
        r = urllib.request.Request(self.base + path, data=json.dumps(body or {}).encode(),
                                   headers={"Content-Type": "application/json"}, method="POST")
        try:
            with urllib.request.urlopen(r, timeout=30) as resp:
                return resp.status, json.loads(resp.read())
        except urllib.error.HTTPError as e:
            with e:
                return e.code, json.loads(e.read())

    def test_an_engine_that_dies_while_it_starts_is_a_503_with_its_reason(self):
        for path in ("/load", "/v1/load"):
            with self.subTest(path=path):
                code, body = self.post(path)
                self.assertEqual(code, 503)
                self.assertEqual(body["error"]["type"], "server_error")
                self.assertIn("exited before it was ready", body["error"]["message"])

    def test_an_engine_that_cannot_be_ended_is_a_503_too(self):
        self.engine.error = EngineStuck("the old engine did not end")
        code, body = self.post("/load")
        self.assertEqual(code, 503)
        self.assertIn("did not end", body["error"]["message"])


if __name__ == "__main__":
    unittest.main()
