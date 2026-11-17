"""#4037: the reply's token limit is one whole number on every route.  A boolean, a fraction, a string or a list used to be
coerced by int() on the Chat and Messages routes (true became 1, 2.5 became 2) while Responses refused them; all three now answer
400 before anything is sent to the engine.  -1 and 0 (no cap), a positive whole number and an absent or null limit still work.

    python -m unittest serve.test_token_limit -v
"""
import http.client
import json
import unittest
from pathlib import Path
from unittest import mock

from serve.frontend import ChatTemplate
from serve.server import ByteTokenizer, MockEngine, Service, serve

ROOT = Path(__file__).resolve().parents[1]
ROUTES = (("/v1/chat/completions", "max_tokens"),
          ("/v1/chat/completions", "max_completion_tokens"),
          ("/v1/messages", "max_tokens"),
          ("/v1/responses", "max_output_tokens"))
BAD = (True, False, 2.5, "2.5", "abc", "", " 5", "5 ", "-2", [], {}, -2, float("inf"))
GOOD = (None, -1, 0, 7, 4096.0)   # (4096.0: a whole number a client wrote as a float)
# a whole number as a string ("1024": int() always took it, and clients rely on it): Chat and Messages keep taking it, as before;
# the Responses route never did
DIGITS = ("1024", "+7", "0", "-1")


class TokenLimit(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        tok = ByteTokenizer()
        cls.engine = MockEngine(tok, "ok", max_context=131072)
        cls.svc = Service(cls.engine, tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
        cls.httpd = serve(cls.svc, port=0)
        cls.port = cls.httpd.server_address[1]

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()

    def post(self, path, key, value, stream=False):
        body = {"input": "hi"} if path == "/v1/responses" else {"messages": [{"role": "user", "content": "hi"}]}
        body["stream"] = stream
        body[key] = value
        c = http.client.HTTPConnection("127.0.0.1", self.port, timeout=20)
        try:
            c.request("POST", path, json.dumps(body), {"Content-Type": "application/json"})
            r = c.getresponse()
            return r.status, r.read()
        finally:
            c.close()

    def test_a_limit_that_is_not_a_whole_number_is_a_400_before_the_engine_is_asked(self):
        for path, key in ROUTES:
            for stream in (False, True):
                for value in BAD:
                    with self.subTest(path=path, key=key, stream=stream, value=value):
                        with mock.patch.object(self.engine, "generate") as generate:
                            status, body = self.post(path, key, value, stream)
                        generate.assert_not_called()
                        self.assertEqual(status, 400, body[:300])
                        self.assertIn(key.encode(), body)

    def test_a_whole_number_limit_or_none_is_answered(self):
        for path, key in ROUTES:
            for value in GOOD + (DIGITS if path != "/v1/responses" else ()):
                with self.subTest(path=path, key=key, value=value):
                    status, body = self.post(path, key, value)
                    self.assertEqual(status, 200, body[:300])

    def test_the_responses_route_still_refuses_a_string(self):
        for value in DIGITS:
            with self.subTest(value=value):
                status, body = self.post("/v1/responses", "max_output_tokens", value)
                self.assertEqual(status, 400, body[:300])


if __name__ == "__main__":
    unittest.main()
