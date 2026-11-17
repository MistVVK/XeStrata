"""serve/test_anthropic_base_url.py - an Anthropic client reaches the server by the root address, and by the "/v1" one too.

The Anthropic SDKs (and Claude Code) add /v1/messages to the base URL themselves. The app used to show the OpenAI-style base
(.../v1) in its Anthropic tab, so those clients posted /v1/v1/messages and got a 404. The address is now shown without /v1, and
the server accepts /v1/v1/messages (and /v1/v1/messages/count_tokens) as the same request, for people who already pasted the old one.

Uses the real `anthropic` Python client against the mock-engine server (no network, no key). Skipped when the package is missing.

    python -m unittest serve.test_anthropic_base_url -v
"""
from __future__ import annotations

import http.client
import json
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve.frontend import ChatTemplate  # noqa: E402
from serve.server import BadBody, ByteTokenizer, GpuBusy, MockEngine, ModelBusy, Service, serve  # noqa: E402
from serve.structured import StructuredOutputError  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

try:
    import anthropic
except ImportError:                                           # pragma: no cover
    anthropic = None


REPLY = "<think>\nhm\n</think>\n\nhello from the mock"


class AnthropicBase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        tok = ByteTokenizer()
        cls.engine = MockEngine(tok, REPLY, max_context=4096)
        cls.svc = Service(cls.engine, tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
        cls.httpd = serve(cls.svc, port=0)
        cls.port = cls.httpd.server_address[1]
        cls.root = f"http://127.0.0.1:{cls.port}"

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()

    def raw(self, path, body):
        c = http.client.HTTPConnection("127.0.0.1", self.port, timeout=20)
        try:
            c.request("POST", path, body=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
            r = c.getresponse()
            return r.status, json.loads(r.read() or b"{}")
        finally:
            c.close()

    @unittest.skipIf(anthropic is None, "the anthropic package is not installed")
    def test_sdk_with_the_root_address_works(self):
        client = anthropic.Anthropic(base_url=self.root, api_key="not-needed", max_retries=0)
        msg = client.messages.create(model="strata", max_tokens=32, messages=[{"role": "user", "content": "hi"}])
        self.assertEqual(msg.type, "message")
        self.assertIn("hello", "".join(b.text for b in msg.content if b.type == "text"))

    @unittest.skipIf(anthropic is None, "the anthropic package is not installed")
    def test_sdk_with_the_old_v1_address_works_through_the_alias(self):
        client = anthropic.Anthropic(base_url=self.root + "/v1", api_key="not-needed", max_retries=0)
        msg = client.messages.create(model="strata", max_tokens=32, messages=[{"role": "user", "content": "hi"}])
        self.assertEqual(msg.type, "message")
        self.assertIn("hello", "".join(b.text for b in msg.content if b.type == "text"))

    @unittest.skipIf(anthropic is None, "the anthropic package is not installed")
    def test_sdk_streaming_through_the_alias(self):
        client = anthropic.Anthropic(base_url=self.root + "/v1", api_key="not-needed", max_retries=0)
        with client.messages.stream(model="strata", max_tokens=32, messages=[{"role": "user", "content": "hi"}]) as stream:
            text = "".join(stream.text_stream)
        self.assertIn("hello", text)

    def test_count_tokens_alias(self):
        body = {"model": "strata", "messages": [{"role": "user", "content": "hi"}]}
        code, real = self.raw("/v1/messages/count_tokens", body)
        code2, aliased = self.raw("/v1/v1/messages/count_tokens", body)
        self.assertEqual((code, code2), (200, 200))
        self.assertEqual(real, aliased)

    def test_only_the_two_messages_paths_are_aliased(self):
        body = {"model": "strata", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 8}
        for path in ("/v1/v1/chat/completions", "/v1/v1/responses", "/v1/v1/v1/messages", "/v1/v2/messages"):
            code, _ = self.raw(path, body)
            self.assertEqual(code, 404, path)


class ErrorEnvelope(unittest.TestCase):
    """platform.claude.com/docs/en/api/errors: {"type": "error", "error": {"type", "message"}, "request_id"} and a
    request-id header. @ai-sdk/anthropic requires the top-level type: without it the client shows only "Bad Request"."""

    @classmethod
    def setUpClass(cls):
        tok = ByteTokenizer()
        cls.engine = MockEngine(tok, REPLY, max_context=4096)
        cls.svc = Service(cls.engine, tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
        cls.httpd = serve(cls.svc, port=0)
        cls.port = cls.httpd.server_address[1]
        cls.root = f"http://127.0.0.1:{cls.port}"

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()

    def post(self, path, body, headers=None):
        c = http.client.HTTPConnection("127.0.0.1", self.port, timeout=20)
        try:
            c.request("POST", path, body=json.dumps(body).encode(),
                      headers={"Content-Type": "application/json", **(headers or {})})
            r = c.getresponse()
            return r.status, r.getheader("request-id"), json.loads(r.read() or b"{}")
        finally:
            c.close()

    def assert_envelope(self, request_id, body, error_type):
        self.assertEqual(body.get("type"), "error", body)
        self.assertEqual(body["error"]["type"], error_type, body)
        self.assertTrue(body["error"]["message"], body)
        self.assertRegex(body["request_id"], r"^req_[0-9a-f]{12}$")
        self.assertEqual(request_id, body["request_id"])

    def test_a_refused_request_on_each_path(self):
        for path in ("/v1/messages", "/v1/messages/", "/v1/messages?beta=true", "/v1/v1/messages",
                     "/v1/messages/count_tokens", "/v1/v1/messages/count_tokens"):
            with self.subTest(path=path), mock.patch("builtins.print"):
                status, request_id, body = self.post(path, {"messages": 5, "max_tokens": 4})
                self.assertEqual(status, 400, body)
                self.assert_envelope(request_id, body, "invalid_request_error")

    def test_the_id_is_the_request_monitor_s(self):
        with mock.patch("builtins.print"), mock.patch.object(self.svc, "api_monitor", True):
            status, request_id, body = self.post("/v1/messages", {"messages": 5, "max_tokens": 4})
        self.assertEqual(status, 400)
        self.assertEqual(request_id, "req_" + self.svc.request_records()[0]["id"])
        self.assertEqual(self.svc.request_records()[0]["http_status"], 400)

    def test_each_status_gets_the_type_on_the_errors_page(self):
        chat = {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 4}
        with mock.patch.object(self.svc, "api_key", "secret"):
            status, request_id, body = self.post("/v1/messages", chat)
        self.assertEqual(status, 401)
        self.assert_envelope(request_id, body, "authentication_error")
        # another site's page: a 403 "forbidden" until now
        with mock.patch("builtins.print"):
            status, request_id, body = self.post("/v1/messages", chat, {"Origin": "http://example.com"})
        self.assertEqual(status, 403)
        self.assert_envelope(request_id, body, "permission_error")
        # a body with no error.type until now; "other 4XX status codes not listed" are invalid_request_error
        status, request_id, body = self.post("/v1/messages", chat, {"Origin": self.root, "Content-Type": "text/plain"})
        self.assertEqual(status, 415)
        self.assert_envelope(request_id, body, "invalid_request_error")
        with mock.patch("serve.server.body_limit", return_value=4):
            status, request_id, body = self.post("/v1/messages", chat)
        self.assertEqual(status, 413)
        self.assert_envelope(request_id, body, "request_too_large")
        # the code that says why it was refused stays
        with mock.patch.object(self.svc, "reserve_inference", return_value=False):
            status, request_id, body = self.post("/v1/messages", chat)
        self.assertEqual(status, 429)
        self.assert_envelope(request_id, body, "rate_limit_error")
        self.assertEqual(body["error"]["code"], "admission_full")

    def test_a_503_is_not_called_overloaded(self):
        # 503 has no row on the page, and Claude Code treats "overloaded_error" in a message as an overload
        with mock.patch.object(self.svc, "load", side_effect=GpuBusy("the GPU is busy")):
            status, request_id, body = self.post("/v1/messages", {"messages": [{"role": "user", "content": "hi"}],
                                                                  "max_tokens": 4})
        self.assertEqual(status, 503, body)
        self.assert_envelope(request_id, body, "server_error")

    def test_the_rows_no_normal_request_reaches(self):
        # 409 and 500 do not happen on these paths in normal use, so the refusal is forced; 502 has no row and keeps its type
        chat = {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 4}
        for refusal, code, error_type in ((ModelBusy("busy"), 409, "conflict_error"),
                                          (BadBody(500, "broken"), 500, "api_error"),
                                          (StructuredOutputError("no JSON"), 502, "structured_output_failed")):
            with self.subTest(code=code), mock.patch.object(self.svc, "load", side_effect=refusal):
                status, request_id, body = self.post("/v1/messages", chat)
                self.assertEqual(status, code, body)
                self.assert_envelope(request_id, body, error_type)

    def test_a_get_on_these_paths_is_a_404(self):
        for path in ("/v1/messages", "/v1/messages/count_tokens"):
            with self.subTest(path=path):
                c = http.client.HTTPConnection("127.0.0.1", self.port, timeout=20)
                try:
                    c.request("GET", path)
                    r = c.getresponse()
                    status, request_id, body = r.status, r.getheader("request-id"), json.loads(r.read())
                finally:
                    c.close()
                self.assertEqual(status, 404, body)
                self.assert_envelope(request_id, body, "not_found_error")

    def test_retry_after_stays_beside_the_id(self):
        chat = {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 4}
        c = http.client.HTTPConnection("127.0.0.1", self.port, timeout=20)
        try:
            with mock.patch.object(self.svc, "load", side_effect=GpuBusy("the GPU is busy", retry_after=7)):
                c.request("POST", "/v1/messages", body=json.dumps(chat).encode(), headers={"Content-Type": "application/json"})
                r = c.getresponse()
            status, retry_after, request_id, body = r.status, r.getheader("Retry-After"), r.getheader("request-id"), json.loads(r.read())
        finally:
            c.close()
        self.assertEqual((status, retry_after), (503, "7"), body)
        self.assert_envelope(request_id, body, "server_error")

    def test_success_and_other_families_are_untouched(self):
        chat = {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 4}
        status, request_id, body = self.post("/v1/messages", chat)
        self.assertEqual((status, request_id, body["type"]), (200, None, "message"))
        self.assertNotIn("request_id", body)
        status, request_id, body = self.post("/v1/messages/count_tokens", chat)
        self.assertEqual((status, request_id), (200, None))
        self.assertEqual(list(body), ["input_tokens"])
        with mock.patch("builtins.print"):
            for path in ("/v1/chat/completions", "/v1/messages/batches", "/v1/v1/chat/completions"):
                status, request_id, body = self.post(path, {"messages": 5})
                self.assertIn(status, (400, 404), path)
                self.assertEqual(request_id, None, path)
                self.assertNotIn("type", body, path)
                self.assertNotIn("request_id", body, path)

    @unittest.skipIf(anthropic is None, "the anthropic package is not installed")
    def test_sdk_reads_the_error(self):
        client = anthropic.Anthropic(base_url=self.root, api_key="not-needed", max_retries=0)
        with mock.patch("builtins.print"), self.assertRaises(anthropic.BadRequestError) as caught:
            client.messages.create(model="strata", max_tokens=4, messages=[{"role": "user", "content": "x" * 8192}])
        self.assertEqual(caught.exception.body.get("type"), "error")
        self.assertIn("exceeds the context window", caught.exception.body["error"]["message"])
        self.assertEqual(caught.exception.request_id, caught.exception.body["request_id"])
        self.assertIsNotNone(caught.exception.request_id)


if __name__ == "__main__":
    unittest.main()
