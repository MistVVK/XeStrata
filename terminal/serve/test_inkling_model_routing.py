"""Tests for #6621: rejecting mismatched Strata catalog model IDs and --strict-model mode."""
from __future__ import annotations

import json
import sys
import unittest
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from serve.server import ChatTemplate, Service, serve
from serve.test_server import ByteTokenizer, CTX, MockEngine


class InklingModelRoutingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        tok = ByteTokenizer()
        cls.svc = Service(
            MockEngine(tok, "</think>\n\nok", max_context=CTX),
            tok,
            ChatTemplate(ROOT / "serve/chat_template.jinja"),
            model_name="iq3_xxs",
        )
        cls.svc.set_aliases(["qwen3.8-flash-next-iq3_xxs"])
        cls.httpd = serve(cls.svc, port=0)
        cls.base = f"http://127.0.0.1:{cls.httpd.server_address[1]}"

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()

    def _post(self, path: str, body: dict):
        req = urllib.request.Request(
            self.base + path,
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                return r.status, json.loads(r.read().decode())
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read().decode())

    def test_cross_family_strata_id_rejected(self):
        """#6621: requesting inkling-2-mini-sq-10 while Qwen iq3_xxs is loaded returns 404 model_not_found."""
        msgs = [{"role": "user", "content": "hi"}]
        code, body = self._post(
            "/v1/chat/completions",
            {"model": "inkling-2-mini-sq-10", "messages": msgs, "max_tokens": 8},
        )
        self.assertEqual(code, 404)
        self.assertEqual(body["error"]["code"], "model_not_found")
        self.assertEqual(body["error"]["param"], "model")
        self.assertIn("inkling-2-mini-sq-10", body["error"]["message"])
        self.assertIn("iq3_xxs", body["error"]["message"])

        # Same rejection on /v1/messages and /v1/responses
        code2, body2 = self._post(
            "/v1/messages",
            {"model": "hy4-flash-q4_k_xl", "messages": msgs, "max_tokens": 8},
        )
        self.assertEqual(code2, 404)
        self.assertEqual(body2["error"]["code"], "model_not_found")

        code3, body3 = self._post(
            "/v1/responses",
            {"model": "inkling-2-mini-sq-15", "input": "hi", "max_output_tokens": 8},
        )
        self.assertEqual(code3, 404)
        self.assertEqual(body3["error"]["code"], "model_not_found")

    def test_loaded_model_alias_and_generic_names_accepted(self):
        msgs = [{"role": "user", "content": "hi"}]
        for asked, want in (
            ("iq3_xxs", "iq3_xxs"),
            ("qwen3.8-flash-next-iq3_xxs", "qwen3.8-flash-next-iq3_xxs"),
            ("gpt-4", "iq3_xxs"),
            ("something-else", "iq3_xxs"),
        ):
            code, body = self._post(
                "/v1/chat/completions",
                {"model": asked, "messages": msgs, "max_tokens": 8},
            )
            self.assertEqual(code, 200, f"asked={asked} body={body}")
            self.assertEqual(body["model"], want)

    def test_strict_model_rejects_arbitrary_unknown_names(self):
        msgs = [{"role": "user", "content": "hi"}]
        prev = self.svc.strict_model
        self.svc.strict_model = True
        try:
            code, body = self._post(
                "/v1/chat/completions",
                {"model": "diagnostic-nonexistent-model-47291", "messages": msgs, "max_tokens": 8},
            )
            self.assertEqual(code, 404)
            self.assertEqual(body["error"]["code"], "model_not_found")

            code_ok, body_ok = self._post(
                "/v1/chat/completions",
                {"model": "iq3_xxs", "messages": msgs, "max_tokens": 8},
            )
            self.assertEqual(code_ok, 200)
            self.assertEqual(body_ok["model"], "iq3_xxs")
        finally:
            self.svc.strict_model = prev


if __name__ == "__main__":
    unittest.main()
