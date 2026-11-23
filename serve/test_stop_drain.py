# SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
"""台帳 0589（upstream 3cf22f5ef1、#6055）: 停止中に届いた要求は 503 server_draining。

XeStrata には upstream の `--shutdown-timeout` の drain 待ちが無い。停止の入口で
Service.begin_drain() が受付を閉じ、 engine.close() のあいだに届いた要求が
200・0 トークンで切られる代わりに、再試行できる 503 になることだけを写す。
"""
import json
import unittest
import urllib.error
import urllib.request
from pathlib import Path

from serve.frontend import ChatTemplate
from serve.server import ByteTokenizer, EngineClosing, MockEngine, Service, serve

ROOT = Path(__file__).resolve().parents[1]


class StopDrain(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        tok = ByteTokenizer()
        cls.svc = Service(MockEngine(tok, "ok", max_context=8192), tok,
                          ChatTemplate(ROOT / "serve/test_chat_template.jinja"))
        cls.httpd = serve(cls.svc, port=0)
        cls.base = f"http://127.0.0.1:{cls.httpd.server_address[1]}"

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()

    def post(self):
        body = {"model": "m", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 8}
        req = urllib.request.Request(self.base + "/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=10) as r:
                return r.status, r.read().decode(), r.headers.get("Retry-After")
        except urllib.error.HTTPError as e:
            return e.code, e.read().decode(), e.headers.get("Retry-After")

    def test_a_request_before_the_stop_is_a_reply(self):
        status, body, _ = self.post()
        self.assertEqual(status, 200, body)
        self.assertEqual(json.loads(body)["choices"][0]["finish_reason"], "stop")

    def test_begin_drain_closes_admission(self):
        self.assertFalse(self.svc.draining)
        self.svc.begin_drain()
        self.assertTrue(self.svc.draining)
        with self.assertRaises(EngineClosing):
            self.svc.prepare([{"role": "user", "content": "hi"}], None, {})
        status, body, retry = self.post()
        self.assertEqual(status, 503, body)
        self.assertEqual(json.loads(body)["error"]["code"], "server_draining")
        self.assertEqual(retry, "1")


if __name__ == "__main__":
    unittest.main()
