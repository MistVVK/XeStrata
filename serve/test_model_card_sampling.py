# SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
"""serve/test_model_card_sampling.py - a request that sends no sampling gets the model card's, not greedy (mock engine).

Qwen's "Best Practices": thinking mode temperature 1.0 / top_p 0.95 / top_k 20, non-thinking 0.7 / 0.8 / 20 with
presence_penalty 1.5.  The request's own fields win, then the Chat settings shared with apps, then the config's
`sampling` block, then the card; temperature 0 asked for is still greedy.

    python -m unittest serve.test_model_card_sampling -v
"""
from __future__ import annotations

import json
import sys
import unittest
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve.frontend import ChatTemplate  # noqa: E402
from serve.server import (ByteTokenizer, MockEngine, Service, StrataEngine, asks_greedy,  # noqa: E402
                          model_card_sampling, serve)

ROOT = Path(__file__).resolve().parents[1]


class Recording(MockEngine):
    def generate(self, ids, max_new, sampling, cancel, embeddings=None):
        self.sampling = dict(sampling or {})
        yield from super().generate(ids, max_new, sampling, cancel, embeddings)


class ModelCardSampling(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tok = ByteTokenizer()
        cls.engine = Recording(cls.tok, "<think>\nhm\n</think>\n\nok", max_context=8192)
        cls.svc = Service(cls.engine, cls.tok, ChatTemplate(ROOT / "serve/test_chat_template.jinja"))
        cls.httpd = serve(cls.svc, port=0)
        cls.base = f"http://127.0.0.1:{cls.httpd.server_address[1]}"

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()

    def setUp(self):
        self.svc.model_arch = "qwen4exp"
        self.svc.sampling_defaults = {}
        self.svc.shared = {}
        self.svc.config_effort = None

    def tearDown(self):
        self.svc.model_arch = None
        self.svc.mtp_greedy = False

    def keys(self, **body):
        body = {"model": "m", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 8, **body}
        self.engine.sampling = {}
        req = urllib.request.Request(self.base + "/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        urllib.request.urlopen(req, timeout=10).read()
        return StrataEngine.sampling_keys(self.engine.sampling)

    def test_tables(self):
        self.assertEqual(model_card_sampling("qwen4exp", True), {"temperature": 1.0, "top_p": 0.95, "top_k": 20})
        self.assertEqual(model_card_sampling("qwen4exp", False),
                         {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "presence_penalty": 1.5})
        self.assertEqual(model_card_sampling("qwen35moe", True)["presence_penalty"], 1.5)
        self.assertEqual(model_card_sampling(None, True), {})
        self.assertEqual(model_card_sampling("unknown", False), {})

    def test_mtp_greedy_opt_in(self):
        # "mtp_greedy": true (upstream d353539037, opt-in here): a request naming no sampling stays greedy; one that
        # names any keeps the card for the rest, as without it
        self.svc.mtp_greedy = True
        self.assertNotIn("temperature=1.0", self.keys())
        k = self.keys(top_p=0.9)
        self.assertIn("temperature=1.0", k)
        self.assertIn("top_p=0.9", k)
        self.svc.sampling_defaults = {"temperature": 0.6}
        self.assertIn("temperature=0.6", self.keys())

    def test_thinking_default(self):
        k = self.keys()
        self.assertIn("temperature=1.0", k)
        self.assertIn("top_p=0.95", k)
        self.assertIn("top_k=20", k)
        self.assertNotIn("penalty_present", k)

    def test_non_thinking_default(self):
        k = self.keys(reasoning_effort="none")
        self.assertIn("temperature=0.7", k)
        self.assertIn("top_p=0.8", k)
        self.assertIn("penalty_present=1.5", k)

    def test_request_wins_and_zero_is_greedy(self):
        k = self.keys(temperature=0)
        self.assertNotIn("temperature", k)
        self.assertIn("top_k=20", k)
        k = self.keys(temperature=0.3, top_p=0.5)
        self.assertIn("temperature=0.3", k)
        self.assertIn("top_p=0.5", k)

    def test_config_block_and_shared_settings_win_over_the_card(self):
        self.svc.sampling_defaults = {"temperature": 0.6, "top_k": 40}
        k = self.keys()
        self.assertIn("temperature=0.6", k)
        self.assertIn("top_k=40", k)
        self.assertIn("top_p=0.95", k)
        self.svc.shared = {"temperature": 0.2}
        self.assertIn("temperature=0.2", self.keys())
        self.svc.sampling_defaults = {"temperature": 0}
        self.svc.shared = {}
        self.assertNotIn("temperature", self.keys())     # a config that asks for greedy gets it

    def test_no_family_no_defaults(self):
        self.svc.model_arch = None
        self.assertEqual(self.keys(), "")


class GreedyCard(ModelCardSampling):
    """台帳 0791（upstream d7ce708baf、issue #69）: temperature 0 ではカードの penalty を落とす。"""

    def test_greedy_gets_no_card_penalty(self):
        k = self.keys(temperature=0, reasoning_effort="none")
        self.assertNotIn("penalty_present", k)
        self.svc.model_arch = "qwen35moe"
        self.assertNotIn("penalty_present", self.keys(temperature=0))
        self.assertNotIn("penalty_present", self.keys(temperature=0, reasoning_effort="none"))
        self.assertIn("penalty_present=1.2", self.keys(temperature=0, presence_penalty=1.2))
        self.assertIn("penalty_present=1.5", self.keys(temperature=0.3))
        self.svc.sampling_defaults = {"temperature": 0}
        self.assertNotIn("penalty_present", self.keys())
        self.assertIn("penalty_present=1.5", self.keys(temperature=0.7))
        self.svc.sampling_defaults = {}
        self.svc.shared = {"temperature": 0}
        self.assertNotIn("penalty_present", self.keys())

    def test_asks_greedy(self):
        self.assertTrue(asks_greedy({"temperature": 0}))
        self.assertTrue(asks_greedy({}, {"temperature": 0.0}))
        self.assertFalse(asks_greedy({"temperature": 0.5}, {"temperature": 0}))
        self.assertFalse(asks_greedy({"temperature": None}, {}))
        self.assertFalse(asks_greedy())
        self.assertEqual(model_card_sampling("qwen35moe", True, greedy=True),
                         {"temperature": 1.0, "top_p": 0.95, "top_k": 20})


if __name__ == "__main__":
    unittest.main()
