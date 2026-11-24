# SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
"""台帳 0804（upstream 25d88c4a5b）: /v1/messages のエラーは Anthropic の形で返す。"""
import json
import unittest
import urllib.error
import urllib.request
from pathlib import Path

from serve.frontend import ChatTemplate
from serve.server import ByteTokenizer, MockEngine, Service, api_error_body, serve

ROOT = Path(__file__).resolve().parents[1]


class AnthropicErrorEnvelope(unittest.TestCase):
    def test_body_types_and_request_id(self):
        for code, kind in ((401, "authentication_error"), (403, "permission_error"), (404, "not_found_error"),
                           (409, "conflict_error"), (413, "request_too_large"), (429, "rate_limit_error"),
                           (500, "api_error"), (400, "invalid_request_error")):
            body = api_error_body("/v1/messages", code, {"error": {"message": "x"}}, "req_abc")
            self.assertEqual(body["type"], "error")
            self.assertEqual(body["error"], {"message": "x", "type": kind})
            self.assertEqual(body["request_id"], "req_abc")
        # 502/503 keep their own type: Claude Code must not read a 503 as overloaded_error
        self.assertEqual(api_error_body("/v1/messages", 503, {"error": {"type": "server_error", "message": "x"}},
                                        "req_abc")["error"]["type"], "server_error")
        self.assertEqual(api_error_body("/v1/messages", 502, {"error": {"message": "x"}},
                                        "req_abc")["error"]["type"], "api_error")
        self.assertNotIn("request_id", api_error_body("/v1/chat/completions", 400, {"error": {"message": "x"}},
                                                      "req_abc"))

    @classmethod
    def setUpClass(cls):
        tok = ByteTokenizer()
        cls.svc = Service(MockEngine(tok, "ok", max_context=4096), tok,
                          ChatTemplate(ROOT / "serve/test_chat_template.jinja"))
        cls.httpd = serve(cls.svc, port=0)
        cls.base = f"http://127.0.0.1:{cls.httpd.server_address[1]}"

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()

    def test_http_error_has_top_level_type_and_id(self):
        req = urllib.request.Request(self.base + "/v1/messages", data=b"{}",
                                     headers={"Content-Type": "application/json"})
        with self.assertRaises(urllib.error.HTTPError) as caught:
            urllib.request.urlopen(req, timeout=10)
        with caught.exception as e:
            self.assertEqual(e.code, 400)
            self.assertTrue(e.headers.get("request-id", "").startswith("req_"))
            body = json.loads(e.read())
        self.assertEqual(body["type"], "error")
        self.assertEqual(body["error"]["type"], "invalid_request_error")
        self.assertEqual(body["request_id"], e.headers["request-id"])


if __name__ == "__main__":
    unittest.main()
