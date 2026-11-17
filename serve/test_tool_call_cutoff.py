"""serve/test_tool_call_cutoff.py - a reply cut by max tokens INSIDE a tool call carries strata_unread_tool_markup {"reason": "cut_off"}.

    python -m unittest serve.test_tool_call_cutoff -v

The parser announces a call as soon as it has the function name (tool_start, then argument pieces).  When the output then ends
inside it there is no tool_call and no markup in the answer's text, so the marker (computed from the text the client got) was missing:
a non-streamed answer came back with finish_reason "length", no content and no tool call (soak l620b2, two answers at max_tokens
4096).  The marker now says so on chat completions, messages and responses, streamed or not; chars is 0 (the markup is not in the
answer's text).  A cut by max tokens must not start the server's own retry (it would meet the same cap again): one engine pass only.
Against a scripted engine (no GPU, no pack).
"""
from __future__ import annotations

import json
import sys
import unittest
import urllib.request
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve.frontend import ChatTemplate  # noqa: E402
from serve.server import ByteTokenizer, Service, serve  # noqa: E402
from serve.test_tool_markup_retry import DECLARED, OPENAI_TOOLS, ROOT, Scripted, call  # noqa: E402

P = {"type": "object", "properties": {"command": {"type": "string"}, "description": {"type": "string"},
                                      "file_path": {"type": "string"}, "content": {"type": "string"}}}
WANT = {"reason": "cut_off", "chars": 0}
LONG_CALL = call("write", file_path="a.txt", content="x" * 600)        # cut at 80 tokens: inside the content value
ANTHROPIC_TOOLS = [{"name": n, "input_schema": P} for n in DECLARED]
RESPONSES_TOOLS = [{"type": "function", "name": n, "parameters": P} for n in DECLARED]


class CutInsideACall(unittest.TestCase):
    def start(self, texts):
        self.tok = ByteTokenizer()
        self.engine = Scripted(self.tok, texts)
        self.svc = Service(self.engine, self.tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
        self.httpd = serve(self.svc, port=0)
        self.base = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        self.addCleanup(self.httpd.server_close)
        self.addCleanup(self.httpd.shutdown)

    def post(self, path, body):
        req = urllib.request.Request(self.base + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=30) as r:
            raw = r.read().decode()
            return raw if body.get("stream") else json.loads(raw)

    @staticmethod
    def events(raw):
        return [json.loads(x[6:]) for x in raw.splitlines() if x.startswith("data: {")]

    def test_chat_completions(self):
        self.start(["</think>\n\n" + LONG_CALL])
        b = self.post("/v1/chat/completions", {"model": "m", "max_tokens": 80, "tools": OPENAI_TOOLS,
                                               "messages": [{"role": "user", "content": "write it"}]})
        c = b["choices"][0]
        self.assertEqual(c["finish_reason"], "length")
        self.assertFalse((c["message"].get("content") or "").strip())
        self.assertNotIn("tool_calls", c["message"])
        self.assertEqual(b.get("strata_unread_tool_markup"), WANT)
        self.assertEqual(len(self.engine.prompts), 1, "a cut by max tokens does not start the server's own retry")
        raw = self.post("/v1/chat/completions", {"model": "m", "max_tokens": 80, "tools": OPENAI_TOOLS, "stream": True,
                                                 "messages": [{"role": "user", "content": "write it"}]})
        chunks = self.events(raw)
        self.assertEqual([x["strata_unread_tool_markup"] for x in chunks if "strata_unread_tool_markup" in x], [WANT])
        self.assertEqual([x["choices"][0]["finish_reason"] for x in chunks if x["choices"] and x["choices"][0].get("finish_reason")], ["length"])
        self.assertEqual(len(self.engine.prompts), 2, "one engine pass for each of the two requests")

    def test_messages(self):
        self.start(["</think>\n\n" + LONG_CALL])
        body = {"model": "m", "max_tokens": 80, "messages": [{"role": "user", "content": "write it"}], "tools": ANTHROPIC_TOOLS}
        a = self.post("/v1/messages", body)
        self.assertEqual(a["stop_reason"], "max_tokens")
        self.assertEqual(a.get("strata_unread_tool_markup"), WANT)
        deltas = [x for x in self.events(self.post("/v1/messages", {**body, "stream": True})) if x.get("type") == "message_delta"]
        self.assertEqual(deltas[-1]["delta"]["stop_reason"], "max_tokens")
        self.assertEqual(deltas[-1].get("strata_unread_tool_markup"), WANT)
        self.assertEqual(len(self.engine.prompts), 2)

    def test_responses(self):
        self.start(["</think>\n\n" + LONG_CALL])
        body = {"model": "m", "max_output_tokens": 80, "input": "write it", "tools": RESPONSES_TOOLS}
        r = self.post("/v1/responses", body)
        self.assertEqual(r["status"], "incomplete")
        self.assertEqual(r.get("strata_unread_tool_markup"), WANT)
        final = [x["response"] for x in self.events(self.post("/v1/responses", {**body, "stream": True}))
                 if x.get("type") in ("response.completed", "response.incomplete")]
        self.assertEqual(final[-1]["status"], "incomplete")
        self.assertEqual(final[-1].get("strata_unread_tool_markup"), WANT)
        self.assertEqual(len(self.engine.prompts), 2)

    def test_a_cut_at_the_opener_is_marked_too_and_not_retried(self):
        """Cut before the function name is out: the opener is text (held, then sent when the reply ends): the existing marker, no retry."""
        self.start(["</think>\n\nI will write it now.\n<tool_call>\n<function=wri" + "x" * 200])
        b = self.post("/v1/chat/completions", {"model": "m", "max_tokens": 58, "tools": OPENAI_TOOLS,
                                               "messages": [{"role": "user", "content": "write it"}]})
        self.assertEqual(b["choices"][0]["finish_reason"], "length")
        self.assertEqual(b["strata_unread_tool_markup"]["reason"], "cut_off")
        self.assertEqual(len(self.engine.prompts), 1, "no server retry for a cut by max tokens")

    def test_whole_calls_and_plain_answers_carry_no_marker(self):
        self.start(["</think>\n\nReading it.\n" + call("read", file_path="a.txt")])
        b = self.post("/v1/chat/completions", {"model": "m", "max_tokens": 4000, "tools": OPENAI_TOOLS,
                                               "messages": [{"role": "user", "content": "read it"}]})
        self.assertEqual(b["choices"][0]["finish_reason"], "tool_calls")
        self.assertNotIn("strata_unread_tool_markup", b)
        self.start(["</think>\n\nNothing to call."])
        b = self.post("/v1/chat/completions", {"model": "m", "max_tokens": 4000, "tools": OPENAI_TOOLS,
                                               "messages": [{"role": "user", "content": "hi"}]})
        self.assertNotIn("strata_unread_tool_markup", b)
        # no tools in the request: no marker, whatever the text
        self.start(["</think>\n\n" + LONG_CALL])
        b = self.post("/v1/chat/completions", {"model": "m", "max_tokens": 80, "messages": [{"role": "user", "content": "hi"}]})
        self.assertNotIn("strata_unread_tool_markup", b)


if __name__ == "__main__":
    unittest.main()
