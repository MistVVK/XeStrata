"""serve/test_v1_origin.py - the /v1 foreign-Origin rule (owner 11-16), the same contract as the app host's /v1.

While no api_key is set, a /v1 request whose Origin names a page of another site is refused with 403: any site can make a visitor's
browser POST to this computer. A request without an Origin (curl, SDKs, editors, server-side apps), the server's own page and an
origin in trusted_origins pass. "null", an empty Origin and an unreadable one are another site. GET /v1 is checked like POST. With
an api_key set the key is the check: the right key passes from any page, a wrong one is 401.

    python -m pytest serve/test_v1_origin.py
"""
from __future__ import annotations

import http.client
import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve.frontend import ChatTemplate  # noqa: E402
from serve.server import ByteTokenizer, MockEngine, Service, serve  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


class V1ForeignOrigin(unittest.TestCase):
    def setUp(self):
        tok = ByteTokenizer()
        self.svc = Service(MockEngine(tok, "</think>\n\nok", max_context=4096), tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
        self.httpd = serve(self.svc, port=0)
        self.port = self.httpd.server_address[1]
        self.host = f"127.0.0.1:{self.port}"

    def tearDown(self):
        self.httpd.shutdown()
        self.httpd.server_close()

    def call(self, method, path, headers):
        """One request with the headers exactly as given (an empty Origin included). Returns (status, body text)."""
        c = http.client.HTTPConnection("127.0.0.1", self.port, timeout=30)
        try:
            body = None
            if method == "POST":
                body = json.dumps({"model": "m", "max_tokens": 4, "messages": [{"role": "user", "content": "hi"}]})
                headers = {"Content-Type": "application/json", **headers}
            c.putrequest(method, path, skip_host=True)
            c.putheader("Host", self.host)
            for k, v in headers.items():
                c.putheader(k, v)
            if body is not None:
                c.putheader("Content-Length", str(len(body)))
            c.endheaders(body.encode() if body is not None else None)
            r = c.getresponse()
            return r.status, r.read().decode("utf-8", "replace")
        finally:
            c.close()

    def test_no_key_another_site_is_refused_on_get_and_post(self):
        for method, path in (("GET", "/v1/models"), ("POST", "/v1/chat/completions")):
            for origin in ("https://evil.example", "null", "", "not a url"):
                with self.subTest(method=method, origin=origin):
                    status, body = self.call(method, path, {"Origin": origin})
                    self.assertEqual(status, 403, body)

    def test_no_key_no_origin_passes(self):
        self.assertEqual(self.call("GET", "/v1/models", {})[0], 200)
        self.assertEqual(self.call("POST", "/v1/chat/completions", {})[0], 200)

    def test_no_key_its_own_page_passes(self):
        for method, path in (("GET", "/v1/models"), ("POST", "/v1/chat/completions")):
            with self.subTest(method=method):
                self.assertEqual(self.call(method, path, {"Origin": "http://" + self.host})[0], 200)

    def test_no_key_a_trusted_origin_passes(self):
        self.svc.trusted_origins = ["https://app.example"]
        self.assertEqual(self.call("GET", "/v1/models", {"Origin": "https://app.example"})[0], 200)
        self.assertEqual(self.call("POST", "/v1/chat/completions", {"Origin": "https://app.example"})[0], 200)
        self.assertEqual(self.call("GET", "/v1/models", {"Origin": "https://other.example"})[0], 403)

    def test_key_set_the_key_is_the_check(self):
        self.svc.api_key = "sk-test-origin"
        right = {"Origin": "https://evil.example", "Authorization": "Bearer sk-test-origin"}
        wrong = {"Origin": "https://evil.example", "Authorization": "Bearer sk-wrong"}
        self.assertEqual(self.call("GET", "/v1/models", right)[0], 200)
        self.assertEqual(self.call("POST", "/v1/chat/completions", right)[0], 200)
        self.assertEqual(self.call("GET", "/v1/models", wrong)[0], 401)
        self.assertEqual(self.call("POST", "/v1/chat/completions", wrong)[0], 401)


if __name__ == "__main__":
    unittest.main()
