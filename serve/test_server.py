# SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
"""serve/test_server.py - the max tokens budget over both APIs, against the mock engine (no GPU, no pack).

    python -m unittest serve.test_server -v
"""
from __future__ import annotations

import contextlib
import io
import json
import os
import sys
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve.frontend import ChatTemplate  # noqa: E402
from serve.server import (CTX_SLACK, ByteTokenizer, EngineDied, MockEngine, Service, StrataEngine,  # noqa: E402
                          engine_args, prompt_tokens_seen, request_timings, serve, start_failure_hint)
from types import SimpleNamespace  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CTX = 4096
ANSWER = "xy" * 1000                             # longer than the old 1024 fallback: one token per byte (#606: not
#                                                one token repeated, which the server ends at 256)


class RecordingEngine(MockEngine):
    def generate(self, ids, max_new, sampling, cancel, embeddings=None):
        self.last_max_new = max_new
        yield from super().generate(ids, max_new, sampling, cancel, embeddings)


class MaxTokens(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        tok = ByteTokenizer()
        cls.engine = RecordingEngine(tok, "</think>\n\n" + ANSWER, max_context=CTX)
        cls.svc = Service(cls.engine, tok, ChatTemplate(ROOT / "serve/test_chat_template.jinja"))
        cls.httpd = serve(cls.svc, port=0)
        cls.base = f"http://127.0.0.1:{cls.httpd.server_address[1]}"

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()

    def post(self, path, body):
        req = urllib.request.Request(self.base + path, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as e:
            with e:
                return e.code, json.loads(e.read())

    def call(self, api, text="hi", **budget):
        """-> (status, body, prompt tokens, completion tokens); `budget` is merged into the request as given."""
        msgs = [{"role": "user", "content": text}]
        if api == "openai":
            s, b = self.post("/v1/chat/completions", {"model": "m", "messages": msgs, **budget})
            u = b.get("usage", {})
            return s, b, u.get("prompt_tokens"), u.get("completion_tokens")
        s, b = self.post("/v1/messages", {"model": "m", "messages": msgs, **budget})
        u = b.get("usage", {})
        return s, b, u.get("input_tokens"), u.get("output_tokens")

    def test_anthropic_thinks_only_when_asked(self):
        # #278: Anthropic's thinking is opt-in; "thinking", an effort or a reasoning_budget_tokens (#123) asks for it
        from serve.frontend import anthropic_to_messages
        msgs = [{"role": "user", "content": "u"}]
        kw = lambda **r: anthropic_to_messages({"messages": msgs, **r}, think_unasked=False)[2]   # noqa: E731
        # the default ("anthropic_thinking": "model") renders an unasked request as 0.1.31 did
        self.assertNotIn("enable_thinking", anthropic_to_messages({"messages": msgs})[2])
        self.assertEqual(kw(), {"enable_thinking": False})
        self.assertEqual(kw(thinking={"type": "disabled"}), {"enable_thinking": False})
        self.assertNotIn("enable_thinking", kw(thinking={"type": "enabled", "budget_tokens": 2048}))
        self.assertNotIn("enable_thinking", kw(output_config={"effort": "high"}))
        self.assertNotIn("enable_thinking", kw(reasoning_budget_tokens=30))

    def test_count_tokens_is_the_prompt_messages_reads(self):
        # /v1/messages/count_tokens renders and tokenizes the same prompt /v1/messages would read, without running it
        msgs = [{"role": "user", "content": "how many tokens is this?"}]
        s, b = self.post("/v1/messages/count_tokens", {"model": "m", "messages": msgs})
        self.assertEqual(s, 200)
        s2, _, n_in, _ = self.call("anthropic", "how many tokens is this?", max_tokens=8)
        self.assertEqual(s2, 200)
        self.assertEqual(b["input_tokens"], n_in)

    def test_request_line_parses_the_engine_summary(self):
        line = ("strata serve: prompt 1200 tokens = 1000 reused + 200 read in 50 ms (4000.0 tok/s), 30 generated in "
                "300 ms (100.0 tok/s), drafts accepted 20 of 28, 2 checkpoints")
        from serve.server import ENGINE_REQUEST
        m = ENGINE_REQUEST.search(line)
        self.assertIsNotNone(m)
        self.assertEqual((m["prompt"], m["reused"], m["gen"], m["tg"]), ("1200", "1000", "30", "100.0"))
        # #471: a request cancelled while its prompt was read says how far it got
        m = ENGINE_REQUEST.search("strata serve: prompt 98179 tokens = 0 reused + 12288 of 98179 read in 17565 ms "
                                  "(699.6 tok/s), 0 generated in 0 ms (0.0 tok/s), drafts accepted 0 of 0, "
                                  "0 checkpoints (cancelled)")
        self.assertIsNotNone(m)
        self.assertEqual((m["prompt"], m["reused"], m["read"], m["pp"], m["gen"]),
                         ("98179", "0", "17565", "699.6", "0"))

    def test_unset_budget_is_the_rest_of_the_context(self):
        cases = {"openai": [{"max_tokens": -1}, {"max_tokens": 0}, {}, {"max_tokens": None},
                            {"max_completion_tokens": -1}, {"max_completion_tokens": None, "max_tokens": None}],
                 "anthropic": [{"max_tokens": -1}, {"max_tokens": 0}, {}, {"max_tokens": None}]}
        for api, budgets in cases.items():
            for budget in budgets:
                with self.subTest(api=api, budget=budget):
                    s, b, pt, ct = self.call(api, **budget)
                    self.assertEqual(s, 200, b)
                    self.assertEqual(self.engine.last_max_new, CTX - CTX_SLACK - pt)
                    self.assertGreater(ct, 1024)          # the whole answer, not cut at the old 1024 fallback

    def test_explicit_budget_is_honoured(self):
        for api, budget in [("openai", {"max_tokens": 50}), ("openai", {"max_completion_tokens": 50}),
                            ("openai", {"max_completion_tokens": 50, "max_tokens": 9}),
                            ("anthropic", {"max_tokens": 50}), ("openai", {"max_tokens": 1500}),
                            ("anthropic", {"max_tokens": 1500})]:
            with self.subTest(api=api, budget=budget):
                want = budget.get("max_completion_tokens") or budget["max_tokens"]
                s, b, _, ct = self.call(api, **budget)
                self.assertEqual(s, 200, b)
                self.assertEqual(self.engine.last_max_new, want)
                self.assertEqual(ct, want)

    def test_explicit_budget_over_the_context_is_rejected(self):
        for api in ("openai", "anthropic"):
            with self.subTest(api=api):
                s, b, _, _ = self.call(api, max_tokens=CTX)
                self.assertEqual(s, 400)
                self.assertIn("exceeds the context", b["error"]["message"])
                self.assertIn("\"fit_max_tokens\": true", b["error"]["message"])     # #545: says how to get past it
                self.assertRegex(b["error"]["message"], r"at most \d+ here")

    def test_unset_budget_with_a_near_full_prompt(self):
        _, _, pt0, _ = self.call("openai", max_tokens=1)
        overhead = pt0 - len("hi")                  # the template's tokens around the user text
        for api in ("openai", "anthropic"):
            _, _, pa, _ = self.call(api, max_tokens=1)
            over = pa - pt0                          # the Anthropic template may differ slightly
            with self.subTest(api=api, room=5):     # a few tokens left: the budget is exactly those
                text = "y" * (CTX - CTX_SLACK - overhead - over - 5)
                s, b, pt, ct = self.call(api, text=text, max_tokens=-1)
                self.assertEqual(s, 200, b)
                self.assertEqual(self.engine.last_max_new, 5)
                self.assertEqual(ct, 5)
            with self.subTest(api=api, room=0):     # nothing left: rejected, not truncated
                text = "y" * (CTX - CTX_SLACK - overhead - over)
                s, b, _, _ = self.call(api, text=text)
                self.assertEqual(s, 400, b)
                self.assertIn("no room to answer", b["error"]["message"])

    def test_debug_log_shows_the_resolved_budget(self):
        import contextlib
        import io
        os.environ["STRATA_DEBUG"] = "1"
        try:
            for api in ("openai", "anthropic"):
                with self.subTest(api=api):
                    out = io.StringIO()
                    with contextlib.redirect_stdout(out):
                        _, _, pt, _ = self.call(api, max_tokens=-1)
                    self.assertIn(f"max_new={CTX - CTX_SLACK - pt} ", out.getvalue())
        finally:
            del os.environ["STRATA_DEBUG"]


class FitMaxTokens(unittest.TestCase):
    """PR #24: --fit-max-tokens clamps an explicit budget that overshoots the context instead of a 400."""

    @classmethod
    def setUpClass(cls):
        tok = ByteTokenizer()
        cls.engine = RecordingEngine(tok, "</think>\n\n" + ANSWER, max_context=CTX)
        cls.svc = Service(cls.engine, tok, ChatTemplate(ROOT / "serve/test_chat_template.jinja"), fit_max_tokens=True)
        cls.httpd = serve(cls.svc, port=0)
        cls.base = f"http://127.0.0.1:{cls.httpd.server_address[1]}"

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()

    post = MaxTokens.post
    call = MaxTokens.call

    def test_overshoot_is_clamped_to_the_room(self):
        for api in ("openai", "anthropic"):
            with self.subTest(api=api):
                s, b, pt, ct = self.call(api, max_tokens=CTX)
                self.assertEqual(s, 200, b)
                self.assertEqual(self.engine.last_max_new, CTX - CTX_SLACK - pt)

    def test_a_budget_that_fits_is_unchanged(self):
        s, b, _, ct = self.call("openai", max_tokens=50)
        self.assertEqual(s, 200, b)
        self.assertEqual(self.engine.last_max_new, 50)

    def test_no_room_is_still_a_400(self):
        _, _, pt0, _ = self.call("openai", max_tokens=1)
        overhead = pt0 - len("hi")
        s, b, _, _ = self.call("openai", text="y" * (CTX - CTX_SLACK - overhead), max_tokens=100)
        self.assertEqual(s, 400, b)
        self.assertIn("no room to answer", b["error"]["message"])


class ImageMarkers(unittest.TestCase):
    """#150: the text "<|image_pad|>" inside a message is text, not an image's place."""

    class FakeVision:
        def __init__(self, d):
            self.dir = Path(d)
            self.rows = self.dir / "img.sve"
            self.rows.write_bytes(b"rows")

        def encode(self, source):
            return self.rows, 3

    def test_literal_marker_with_an_image(self):
        import tempfile
        tok = ByteTokenizer()
        with tempfile.TemporaryDirectory() as d:
            svc = Service(MockEngine(tok, "ok", max_context=CTX), tok,
                          ChatTemplate(ROOT / "serve/test_chat_template.jinja"), vision=self.FakeVision(d))
            pad = tok.encode("<|image_pad|>", parse_special=True)[0]
            for text in ("the docs say <|image_pad|> marks an image", "plain"):
                with self.subTest(text=text):
                    msgs = [{"role": "user", "content": [{"type": "text", "text": text},
                                                         {"type": "image", "source": "x.png"}]}]
                    ids, _, _ = svc.prepare(msgs, None, {})
                    self.assertEqual(ids.count(pad), 3)          # the image's three rows, nothing else
                    self.assertIn("<|image_pad|> marks" if "docs" in text else "plain", tok.decode(ids))
            svc.embeddings.path.unlink(missing_ok=True)


class StatusNeedsTheKey(unittest.TestCase):
    """#212: /status shows the end of the answer being written, so it needs the key like /v1/*."""

    def test_status(self):
        tok = ByteTokenizer()
        svc = Service(MockEngine(tok, "ok", max_context=CTX), tok,
                      ChatTemplate(ROOT / "serve/test_chat_template.jinja"))
        svc.api_key = "k3y"
        httpd = serve(svc, port=0)
        base = f"http://127.0.0.1:{httpd.server_address[1]}/status"
        try:
            with self.assertRaises(urllib.error.HTTPError) as e:
                urllib.request.urlopen(base, timeout=10)
            self.assertEqual(e.exception.code, 401)
            e.exception.close()
            req = urllib.request.Request(base, headers={"Authorization": "Bearer k3y"})
            with urllib.request.urlopen(req, timeout=10) as r:
                self.assertEqual(r.status, 200)
                self.assertNotIn("tail", json.loads(r.read()))
        finally:
            httpd.shutdown()
            httpd.server_close()


class ToolCallTerminators(unittest.TestCase):
    """#210: a value that contains </parameter> or </tool_call> (a file documenting the call format) is kept whole."""
    CONTENT = ("Close each value with </parameter> and the call with </function></tool_call>.\n"
               "<parameter=x>\nnot a parameter\n</parameter>\nend")
    SCHEMA = [{"name": "write", "parameters": {"properties": {"path": {"type": "string"},
                                                              "content": {"type": "string"}}}}]

    def run_parser(self, stream_tools, step):
        from serve.frontend import OutputParser
        text = ("</think>\n\n<tool_call>\n<function=write>\n<parameter=path>\ndoc.md\n</parameter>\n"
                f"<parameter=content>\n{self.CONTENT}\n</parameter>\n</function>\n</tool_call>")
        p = OutputParser(thinking=True, tools=self.SCHEMA, stream_tools=stream_tools)
        evs = []
        for i in range(0, len(text), step):
            evs += p.feed(text[i:i + step])
        evs += p.finish()
        return evs

    def test_values_keep_the_terminators(self):
        for stream_tools in (False, True):
            for step in (1, 7, 10_000):
                with self.subTest(stream_tools=stream_tools, step=step):
                    evs = self.run_parser(stream_tools, step)
                    calls = [e.call for e in evs if e.kind == "tool_call"]
                    self.assertEqual(len(calls), 1)
                    self.assertEqual(calls[0].arguments, {"path": "doc.md", "content": self.CONTENT})
                    self.assertFalse([e for e in evs if e.kind == "content" and e.text.strip()])
                    if stream_tools:
                        streamed = "".join(e.text for e in evs if e.kind == "tool_args")
                        self.assertEqual(json.loads(streamed), {"path": "doc.md", "content": self.CONTENT})


class UnfinishedToolCall(unittest.TestCase):
    """#211: a call the output ends inside is not reported as a whole one - its streamed JSON is not closed and the
    finish reason is not "tool_calls" / "tool_use" - so a client can tell it from a call to run."""
    CALL = ("</think>\n\n<tool_call>\n<function=write>\n<parameter=path>\nnotes.txt\n</parameter>\n"
            "<parameter=content>\n")
    CUT = CALL + "first half of the fi"                            # the model's turn ends here
    WHOLE = CALL + "all of it\n</parameter>\n</function>\n</tool_call>"
    PROPS = {"path": {"type": "string"}, "content": {"type": "string"}}

    def test_parser(self):
        from serve.frontend import OutputParser
        schema = [{"name": "write", "parameters": {"properties": self.PROPS}}]
        for text, content in ((self.CUT, None), (self.WHOLE, "all of it"),
                              (self.WHOLE[:-len("</tool_call>")], "all of it")):   # only </tool_call> missing: whole
            for step in (1, 7, 10_000):
                with self.subTest(end=text[-12:], step=step):
                    p = OutputParser(thinking=True, tools=schema, stream_tools=True)
                    evs = []
                    for i in range(0, len(text), step):
                        evs += p.feed(text[i:i + step])
                    evs += p.finish()
                    streamed = "".join(e.text for e in evs if e.kind == "tool_args")
                    calls = [e for e in evs if e.kind == "tool_call"]
                    if content is None:
                        self.assertEqual((calls, streamed), ([], '{"path":"notes.txt","content":"first half of the fi'))
                    else:
                        self.assertEqual(len(calls), 1)
                        self.assertEqual(json.loads(streamed), {"path": "notes.txt", "content": content})

    def answers(self, script, max_tokens=500):
        """(finish reason, the call's arguments) from OpenAI and Anthropic, whole and streamed, for the model's
        `script`."""
        tok = ByteTokenizer()
        svc = Service(MockEngine(tok, script, max_context=CTX), tok,
                      ChatTemplate(ROOT / "serve/test_chat_template.jinja"))
        httpd = serve(svc, port=0)
        base = f"http://127.0.0.1:{httpd.server_address[1]}"
        tools = {"openai": [{"type": "function", "function": {"name": "write", "parameters": {
                     "type": "object", "properties": self.PROPS}}}],
                 "anthropic": [{"name": "write", "input_schema": {"type": "object", "properties": self.PROPS}}]}
        out = {}
        try:
            for api, path in (("openai", "/v1/chat/completions"), ("anthropic", "/v1/messages")):
                for stream in (False, True):
                    body = {"model": "x", "max_tokens": max_tokens, "stream": stream, "tools": tools[api],
                            "messages": [{"role": "user", "content": "save my notes"}]}
                    req = urllib.request.Request(base + path, data=json.dumps(body).encode(), headers={
                        "Content-Type": "application/json", "anthropic-version": "2023-06-01"})
                    with urllib.request.urlopen(req, timeout=30) as r:
                        raw = r.read().decode()
                    if stream:
                        evs = [json.loads(line[6:]) for line in raw.splitlines() if line.startswith("data: {")]
                        if api == "openai":
                            out[api, stream] = (evs[-1]["choices"][0]["finish_reason"], "".join(
                                (tc.get("function") or {}).get("arguments") or "" for e in evs
                                for tc in e["choices"][0]["delta"].get("tool_calls") or []))
                        else:
                            out[api, stream] = (evs[-2]["delta"]["stop_reason"], "".join(
                                e["delta"]["partial_json"] for e in evs if e["type"] == "content_block_delta"
                                and e["delta"]["type"] == "input_json_delta"))
                    elif api == "openai":
                        c = json.loads(raw)["choices"][0]
                        out[api, stream] = (c["finish_reason"], [tc["function"]["arguments"]
                                                                 for tc in c["message"].get("tool_calls") or []])
                    else:
                        m = json.loads(raw)
                        out[api, stream] = (m["stop_reason"],
                                            [b["input"] for b in m["content"] if b["type"] == "tool_use"])
        finally:
            httpd.shutdown()
            httpd.server_close()
        return out

    def test_a_cut_call(self):
        cut = '{"path":"notes.txt","content":"first half of the fi'
        self.assertEqual(self.answers(self.CUT), {
            ("openai", False): ("stop", []), ("openai", True): ("stop", cut),    # whole answers leave the cut
            ("anthropic", False): ("end_turn", []),                      # call out: it has no arguments to give
            ("anthropic", True): ("end_turn", cut)})

    def test_a_call_cut_at_the_token_limit(self):
        """The same cut by max_tokens: "length" / "max_tokens", and the whole (non-streamed) answers leave the call
        out in both APIs."""
        a = self.answers(self.CUT + "rest of the file, never reached" * 40, max_tokens=len(self.CUT))
        self.assertEqual((a["openai", False], a["anthropic", False]), (("length", []), ("max_tokens", [])))
        self.assertEqual((a["openai", True][0], a["anthropic", True][0]), ("length", "max_tokens"))

    def test_collect_keeps_calls_whose_arguments_parse(self):
        from serve.server import openai_collect

        def chunk(delta, finish=None):
            return {"id": "c", "created": 1, "model": "m", "usage": {},
                    "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
        whole = {"index": 0, "id": "a", "type": "function", "function": {"name": "f", "arguments": '{"x": 1}'}}
        cut = {"index": 1, "id": "b", "type": "function", "function": {"name": "g", "arguments": '{"y": "ha'}}
        for finish, want in (("stop", ["a"]), ("length", ["a"]), ("tool_calls", ["a", "b"])):
            with self.subTest(finish=finish):
                msg = openai_collect([chunk({"tool_calls": [whole]}), chunk({"tool_calls": [cut]}),
                                      chunk({}, finish)])["choices"][0]["message"]
                self.assertEqual([c["id"] for c in msg["tool_calls"]], want)
        msg = openai_collect([chunk({"tool_calls": [cut]}), chunk({}, "stop")])["choices"][0]["message"]
        self.assertNotIn("tool_calls", msg)

    def test_a_whole_call(self):
        whole = {"path": "notes.txt", "content": "all of it"}
        a = self.answers(self.WHOLE)
        self.assertEqual((a["openai", False][0], [json.loads(x) for x in a["openai", False][1]]),
                         ("tool_calls", [whole]))
        self.assertEqual((a["openai", True][0], json.loads(a["openai", True][1])), ("tool_calls", whole))
        self.assertEqual(a["anthropic", False], ("tool_use", [whole]))
        self.assertEqual((a["anthropic", True][0], json.loads(a["anthropic", True][1])), ("tool_use", whole))


class ClientShapes(unittest.TestCase):
    """What real clients send: Claude Code posts /v1/messages?beta=true (issue #55) and puts hook context into the
    conversation as a mid-conversation system message (issue #56); some OpenAI clients send a late developer message."""

    @classmethod
    def setUpClass(cls):
        tok = ByteTokenizer()
        cls.engine = RecordingPrompt(tok, "</think>\n\n2", max_context=CTX)
        cls.svc = Service(cls.engine, tok, ChatTemplate(ROOT / "serve/test_chat_template.jinja"))
        cls.httpd = serve(cls.svc, port=0)
        cls.base = f"http://127.0.0.1:{cls.httpd.server_address[1]}"

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()

    def post(self, path, body):
        req = urllib.request.Request(self.base + path, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json", "anthropic-version": "2023-06-01"})
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as e:
            with e:
                return e.code, json.loads(e.read())

    def prompt_text(self):
        return bytes(i for i in self.engine.last_ids if i < 256).decode("utf-8", "replace")

    def test_query_string(self):
        body = {"model": "x", "max_tokens": 20, "messages": [{"role": "user", "content": "hi"}]}
        for path in ("/v1/messages?beta=true", "/v1/chat/completions?api-version=1", "/v1/messages/?beta=true"):
            status, b = self.post(path, body)
            self.assertEqual(status, 200, (path, b))
        status, _ = self.post("/v1/nothing?beta=true", body)
        self.assertEqual(status, 404)

    def test_anthropic_mid_conversation_system(self):
        status, b = self.post("/v1/messages?beta=true", {
            "model": "x", "max_tokens": 50,
            "system": [{"type": "text", "text": "You are terse."}],
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": "1+1? digits only"}]},
                {"role": "system", "content": [{"type": "text", "text": "<system-reminder>answer in digits</system-reminder>"}]}]})
        self.assertEqual(status, 200, b)
        text = self.prompt_text()
        self.assertIn("You are terse.", text)
        self.assertIn("<system-reminder>answer in digits</system-reminder>", text)
        self.assertLess(text.index("You are terse."), text.index("1+1?"))          # the first system stays first
        self.assertLess(text.index("1+1?"), text.index("answer in digits"))        # the late one stays in place

    def test_openai_late_developer_and_system(self):
        status, b = self.post("/v1/chat/completions", {
            "model": "x", "max_tokens": 50,
            "messages": [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "hello"},
                         {"role": "assistant", "content": "hi"}, {"role": "developer", "content": "Now use digits."},
                         {"role": "system", "content": "Also this."}, {"role": "user", "content": "1+1?"}]})
        self.assertEqual(status, 200, b)
        text = self.prompt_text()
        for part in ("Be brief.", "Now use digits.", "Also this.", "1+1?"):
            self.assertIn(part, text)

    def test_no_user_turn_is_a_400(self):
        # #365: the template's own refusal (Qwen's "No user query found in messages." when no turn is a user's query)
        # answers 400, not a dropped connection
        with tempfile.TemporaryDirectory() as d:
            tpl = Path(d) / "chat_template.jinja"
            tpl.write_text("{% if messages[-1].role != 'user' %}"
                           "{{ raise_exception('No user query found in messages.') }}"
                           "{% endif %}{{ messages[-1].content }}", encoding="utf-8")
            template, self.svc.template = self.svc.template, ChatTemplate(tpl)
            try:
                status, b = self.post("/v1/chat/completions", {
                    "model": "x", "max_tokens": 20, "messages": [{"role": "system", "content": "Only a system."}]})
                self.assertEqual(status, 400, b)
                self.assertIn("No user query found", b["error"]["message"])
            finally:
                self.svc.template = template
        status, b = self.post("/v1/chat/completions", {"model": "x", "max_tokens": 20,
                                                       "messages": [{"role": "user", "content": "hi"}]})
        self.assertEqual(status, 200, b)                            # the server goes on

    def test_messages_as_a_json_string_over_http(self):
        # #460: a double-encoded "messages" is answered; one that is not a list of objects is a 400, not a 500
        encoded = json.dumps([{"role": "user", "content": "1+1?"}])
        for path in ("/v1/chat/completions", "/v1/messages"):
            with self.subTest(path=path):
                status, b = self.post(path, {"model": "x", "max_tokens": 20, "messages": encoded})
                self.assertEqual(status, 200, b)
                self.assertIn("1+1?", self.prompt_text())
                status, b = self.post(path, {"model": "x", "max_tokens": 20, "messages": ["hi"]})
                self.assertEqual(status, 400, b)
                self.assertIn("messages must be a list of objects", b["error"]["message"])

    def test_vision_temp_image_removed_when_the_pipe_fails(self):
        # #352: the temporary image goes even when the encoder's pipe raises
        from serve.server import Vision

        class Gone:
            def write(self, _):
                raise BrokenPipeError("the encoder is gone")

            def close(self):
                pass

        v = Vision.__new__(Vision)
        v.dir, v.lock, v.cache = Path(tempfile.mkdtemp(prefix="strata-vision-test-")), threading.Lock(), {}
        v.proc, v.dead = mock.Mock(stdin=Gone()), None
        with mock.patch.object(Vision, "load", return_value=b""), \
                mock.patch.object(Vision, "normalize", return_value=b"png"):
            with self.assertRaisesRegex(ValueError, "stopped"):   # the encoder is marked dead, with the reason
                v.encode("x")
        self.assertEqual(list(v.dir.iterdir()), [])
        self.assertIsNotNone(v.dead)
        v.dir.rmdir()

    def test_leading_system_unchanged(self):
        from serve.frontend import anthropic_to_messages, openai_to_messages
        msgs, _, _ = openai_to_messages({"messages": [{"role": "developer", "content": "D"}, {"role": "user", "content": "u"}]})
        self.assertEqual([m["role"] for m in msgs], ["system", "user"])
        msgs, _, _ = anthropic_to_messages({"system": "S", "messages": [{"role": "user", "content": "u"}]})
        self.assertEqual([m["role"] for m in msgs], ["system", "user"])

    def test_messages_sent_as_a_json_string(self):
        # #460: a client that double-encodes "messages" (and "tool_calls") as a JSON string gets them decoded; what is
        # still not a list of objects is a ValueError (the server's 400), not an AttributeError on m.get
        from serve.frontend import anthropic_to_messages, openai_to_messages
        call = [{"id": "c1", "type": "function", "function": {"name": "f", "arguments": "{\"x\": 1}"}}]
        listed = [{"role": "user", "content": "u"}, {"role": "assistant", "content": "", "tool_calls": call}]
        encoded = [listed[0], dict(listed[1], tool_calls=json.dumps(call))]
        want = openai_to_messages({"messages": listed})[0]
        self.assertEqual(want[1]["tool_calls"], [{"function": {"name": "f", "arguments": {"x": 1}}}])
        self.assertEqual(openai_to_messages({"messages": json.dumps(listed)})[0], want)
        self.assertEqual(openai_to_messages({"messages": json.dumps(encoded)})[0], want)
        anth = [{"role": "user", "content": "u"}]
        self.assertEqual(anthropic_to_messages({"messages": json.dumps(anth)})[0],
                         anthropic_to_messages({"messages": anth})[0])
        self.assertEqual(openai_to_messages({})[0], [])                     # no field: nothing, as before
        self.assertEqual(openai_to_messages({"messages": None})[0], [])
        for bad in ("not json", "\"a string\"", json.dumps({"role": "user"}), ["hi"], [{"role": "user"}, 3], 5,
                    {"role": "user", "content": "u"}):
            for fn in (openai_to_messages, anthropic_to_messages):
                with self.subTest(bad=bad, fn=fn.__name__):
                    with self.assertRaisesRegex(ValueError, "messages must be a list of objects"):
                        fn({"messages": bad})
        for calls in (["f"], "[1]", [{"function": "f"}], "{"):
            with self.subTest(calls=calls), self.assertRaisesRegex(ValueError, "tool_calls must be a list of objects"):
                openai_to_messages({"messages": [{"role": "assistant", "content": "", "tool_calls": calls}]})


class SamplingKeys(unittest.TestCase):
    """The GEN line's sampling keys: top_k 0 ("off") or wider than the engine's 64 get the widest list, 64 (they used
    to fall back to the engine default 20); a penalty always carries its window."""

    def keys(self, **sampling):
        return StrataEngine.sampling_keys(sampling).split()

    def test_top_k(self):
        self.assertIn("top_k=10", self.keys(temperature=0.7, top_k=10))
        self.assertIn("top_k=64", self.keys(temperature=0.7, top_k=64))
        self.assertIn("top_k=64", self.keys(temperature=0.7, top_k=0))
        self.assertIn("top_k=64", self.keys(temperature=0.7, top_k=100))
        for bad in (-1, True, 2.5, "20"):
            self.assertFalse([k for k in self.keys(temperature=0.7, top_k=bad) if k.startswith("top_k=")], bad)

    def test_tune_keys(self):
        k = self.keys(temperature=0, strata_tune={"pcie_frac": 0.2, "spec_min_p": 0.7})
        self.assertIn("pcie_frac=0.2", k)
        self.assertIn("spec_min_p=0.7", k)
        bad = self.keys(strata_tune={"pcie_frac": 3, "spec_min_p": True, "pool_workers": 2})
        self.assertFalse([x for x in bad if x.split("=")[0] in ("pcie_frac", "spec_min_p", "pool_workers")])

    def test_penalty_window(self):
        self.assertIn("penalty_last_n=64", self.keys(presence_penalty=1.5))
        self.assertIn("penalty_last_n=4096", self.keys(repetition_penalty=1.1, penalty_last_n=4096))
        self.assertFalse([k for k in self.keys(temperature=0.7) if k.startswith("penalty")])


class GpuChoice(unittest.TestCase):
    """Issue #51: the config's \"gpu\" reaches the engine as CUDA_VISIBLE_DEVICES, numbered like nvidia-smi."""

    def test_env(self):
        from serve.server import child_env
        env = child_env({"gpu": 1})
        self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "1")
        self.assertEqual(env["CUDA_DEVICE_ORDER"], "PCI_BUS_ID")
        plain = child_env({})                     # no choice: the environment as it was (existing installs)
        self.assertEqual(plain.get("CUDA_VISIBLE_DEVICES"), os.environ.get("CUDA_VISIBLE_DEVICES"))
        self.assertEqual(plain.get("CUDA_DEVICE_ORDER"), os.environ.get("CUDA_DEVICE_ORDER"))


class RecordingPrompt(MockEngine):
    def generate(self, ids, max_new, sampling, cancel, embeddings=None):
        self.last_ids = list(ids)
        yield from super().generate(ids, max_new, sampling, cancel, embeddings)


class DyingEngine(MockEngine):
    """Issue #27: an engine that dies after a few tokens of its first answer, and comes back when restarted."""

    def __init__(self, tok, script, max_context):
        super().__init__(tok, script, max_context=max_context)
        self.dead, self.restarts, self.die_after = False, 0, 5

    def alive(self):
        return not self.dead

    def restart(self):
        self.dead, self.die_after = False, None
        self.restarts += 1

    def generate(self, ids, max_new, sampling, cancel, embeddings=None):
        for i, t in enumerate(super().generate(ids, max_new, sampling, cancel, embeddings)):
            if self.die_after is not None and i == self.die_after:
                self.dead = True
                raise EngineDied("the engine stopped unexpectedly (exit code -9)")
            yield t


class SlowPromptEngine(MockEngine):
    """Reads a "long prompt" for up to 20 s without a token (the engine sends nothing then), stopping on cancel."""

    def generate(self, ids, max_new, sampling, cancel, embeddings=None):
        self.cancelled_after = None
        t0 = time.monotonic()
        while time.monotonic() - t0 < 20:
            if cancel.is_set():
                self.cancelled_after = time.monotonic() - t0
                return
            time.sleep(0.05)
        yield from super().generate(ids, max_new, sampling, cancel, embeddings)


class ClientHangUp(unittest.TestCase):
    """#430 #431: a client that hangs up during a long prompt read cancels the request within about a second -
    non-streamed (which writes nothing until the end) and streamed (one keep-alive per prompt chunk) alike."""

    @classmethod
    def setUpClass(cls):
        tok = ByteTokenizer()
        cls.engine = SlowPromptEngine(tok, "</think>\n\nOK", max_context=CTX)
        cls.svc = Service(cls.engine, tok, ChatTemplate(ROOT / "serve/test_chat_template.jinja"))
        cls.httpd = serve(cls.svc, port=0)
        cls.port = cls.httpd.server_address[1]

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()

    def hang_up(self, stream):
        import socket as so
        body = json.dumps({"model": "x", "max_tokens": 20, "stream": stream,
                           "messages": [{"role": "user", "content": "a long prompt"}]}).encode()
        c = so.create_connection(("127.0.0.1", self.port))
        c.sendall(b"POST /v1/chat/completions HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Type: application/json\r\n"
                  b"Content-Length: " + str(len(body)).encode() + b"\r\n\r\n" + body)
        time.sleep(1.0)
        c.close()
        t0 = time.monotonic()
        while self.engine.cancelled_after is None and time.monotonic() - t0 < 10:
            time.sleep(0.05)
        self.assertIsNotNone(self.engine.cancelled_after, "the request was not cancelled")
        self.assertLess(self.engine.cancelled_after, 3.0)

    def test_non_streamed(self):
        self.hang_up(False)

    def test_streamed(self):
        self.hang_up(True)


class EngineDeath(unittest.TestCase):
    """Issue #27: a dead engine is an error (not "length"), and the next request starts it again."""

    def test_error_then_restart(self):
        tok = ByteTokenizer()
        eng = DyingEngine(tok, "</think>\n\n" + ANSWER, max_context=CTX)
        svc = Service(eng, tok, ChatTemplate(ROOT / "serve/test_chat_template.jinja"))
        httpd = serve(svc, port=0)
        base = f"http://127.0.0.1:{httpd.server_address[1]}"
        try:
            def post(body):
                req = urllib.request.Request(base + "/v1/chat/completions", data=json.dumps(body).encode(),
                                             headers={"Content-Type": "application/json"})
                try:
                    with urllib.request.urlopen(req, timeout=30) as r:
                        return r.status, r.read().decode()
                except urllib.error.HTTPError as e:
                    with e:
                        return e.code, e.read().decode()
            msgs = [{"role": "user", "content": "hi"}]
            code, text = post({"model": "m", "messages": msgs, "max_tokens": 50, "stream": True})
            self.assertEqual(code, 200)
            self.assertIn('"error"', text)
            self.assertIn("stopped unexpectedly", text)
            self.assertTrue(text.rstrip().endswith("data: [DONE]"))
            self.assertEqual(svc.metrics()["requests"][0]["finish"], "error")
            code, text = post({"model": "m", "messages": msgs, "max_tokens": 50})
            self.assertEqual(code, 200, text)
            self.assertEqual(eng.restarts, 1)
            self.assertEqual(json.loads(text)["usage"]["completion_tokens"], 50)
        finally:
            httpd.shutdown()
            httpd.server_close()

    def test_engine_err_mid_stream(self):
        """The engine's ERR line after the stream started reaches the client as an error event (it used to be a
        400 written into the open stream, which clients read as an empty answer)."""
        class ErrEngine(MockEngine):
            def generate(self, ids, max_new, sampling, cancel, embeddings=None):
                yield None                                  # a prompt-progress heartbeat: the stream has started
                raise ValueError("verify: layer 31 never rang (an illegal memory access was encountered)")

        tok = ByteTokenizer()
        svc = Service(ErrEngine(tok, ANSWER, max_context=CTX), tok, ChatTemplate(ROOT / "serve/test_chat_template.jinja"))
        httpd = serve(svc, port=0)
        base = f"http://127.0.0.1:{httpd.server_address[1]}"
        try:
            for path, body in [("/v1/chat/completions", {"model": "m", "stream": True, "max_tokens": 20,
                                                          "messages": [{"role": "user", "content": "hi"}]}),
                               ("/v1/messages", {"model": "m", "stream": True, "max_tokens": 20,
                                                 "messages": [{"role": "user", "content": "hi"}]})]:
                req = urllib.request.Request(base + path, data=json.dumps(body).encode(),
                                             headers={"Content-Type": "application/json"})
                with urllib.request.urlopen(req, timeout=30) as r:
                    text = r.read().decode()
                self.assertIn("illegal memory access", text, path)
                self.assertNotIn("HTTP/1", text, path)
                self.assertEqual(svc.metrics()["requests"][0]["finish"], "error")
        finally:
            httpd.shutdown()
            httpd.server_close()


class DoneLineEngine(MockEngine):
    """The mock engine whose `last` comes from a DONE line, parsed as StrataEngine parses it."""

    def __init__(self, *a, done_lines=(), **kw):
        super().__init__(*a, **kw)
        self.done_lines = list(done_lines)

    def generate(self, ids, max_new, sampling, cancel, embeddings=None):
        try:
            yield from super().generate(ids, max_new, sampling, cancel, embeddings)
        finally:
            StrataEngine._parse_done(self, self.done_lines.pop(0))


class DraftCounts(unittest.TestCase):
    """#457: GET /metrics gives each request's speculative draft counts (offered / accepted, from the engine's DONE
    line; None when the line has no such fields) and their running sums in the totals."""

    def test_drafts_in_history_and_totals(self):
        tok = ByteTokenizer()
        engine = DoneLineEngine(tok, "</think>\n\nok", max_context=CTX, done_lines=[
            "DONE 4 20 40.0 30.0 stop 7 12 0",                  # 7 of 12 drafts accepted
            "DONE 4 20 40.0 30.0 stop",                          # an engine that reports no drafts
            "DONE 4 20 40.0 30.0 stop 3 5 0 9 10"])
        svc = Service(engine, tok, ChatTemplate(ROOT / "serve/test_chat_template.jinja"))
        self.assertEqual((svc.totals["drafts_offered"], svc.totals["drafts_accepted"]), (0, 0))
        httpd = serve(svc, port=0)
        base = f"http://127.0.0.1:{httpd.server_address[1]}"
        try:
            for _ in range(3):
                body = json.dumps({"model": "m", "max_tokens": 10, "messages": [{"role": "user", "content": "hi"}]})
                with urllib.request.urlopen(urllib.request.Request(base + "/v1/chat/completions", data=body.encode(),
                                                                   headers={"Content-Type": "application/json"}),
                                            timeout=30) as r:
                    self.assertEqual(r.status, 200)
            with urllib.request.urlopen(base + "/metrics", timeout=10) as r:
                m = json.loads(r.read())
        finally:
            httpd.shutdown()
            httpd.server_close()
        rows = m["requests"]                                      # newest first
        self.assertEqual([(r["drafts_offered"], r["drafts_accepted"]) for r in rows], [(5, 3), (None, None), (12, 7)])
        self.assertEqual((m["totals"]["drafts_offered"], m["totals"]["drafts_accepted"]), (17, 10))


class LearnedProfile(unittest.TestCase):
    """#477: "expert_profile_save" in the config: the engine saves its learned profile there, and the next start
    begins from it when it is a profile of the same model; without the key the arguments are unchanged."""

    def write(self, path, nl=48, ne=512, n=4):
        sys.path.insert(0, str(ROOT / "tools"))
        import make_profile
        old = make_profile.N_LAYER
        make_profile.N_LAYER = nl
        try:
            make_profile.write_profile(path, [(i % nl, i // nl) for i in range(n)], n_expert=ne)
        finally:
            make_profile.N_LAYER = old

    def test_without_the_key_nothing_changes(self):
        args = ["--native", "x", "--expert-profile", "data/expert-profile.bin", "--adapt-every", "4"]
        self.assertEqual(engine_args({"args": list(args)}), args)
        self.assertEqual(engine_args({"args": list(args), "expert_profile_save": ""}), args)

    def test_save_and_start_from_it(self):
        d = Path(tempfile.mkdtemp())
        self.write(d / "base.bin")
        cfg = {"args": ["--expert-profile", "base.bin"], "cwd": str(d), "expert_profile_save": "learned.bin",
               "expert_profile_save_every": 5}
        # nothing saved yet: the config's profile, and the engine is told where to save
        self.assertEqual(engine_args(cfg), ["--expert-profile", "base.bin", "--expert-profile-save", "learned.bin",
                                            "--expert-profile-save-every", "5"])
        self.write(d / "learned.bin")
        self.assertEqual(engine_args(cfg)[:2], ["--expert-profile", "learned.bin"])
        self.write(d / "learned.bin", ne=256)                    # another model's: not used
        self.assertEqual(engine_args(cfg)[:2], ["--expert-profile", "base.bin"])
        (d / "learned.bin").write_bytes(b"STRP" + bytes(20))     # not a whole profile
        self.assertEqual(engine_args(cfg)[:2], ["--expert-profile", "base.bin"])
        self.write(d / "learned.bin")
        (d / "learned.bin").write_bytes((d / "learned.bin").read_bytes()[:30])   # truncated
        self.assertEqual(engine_args(cfg)[:2], ["--expert-profile", "base.bin"])

    def test_no_profile_in_the_args(self):
        cfg = {"args": ["--native", "x"], "expert_profile_save": "learned.bin"}
        self.assertEqual(engine_args(cfg), ["--native", "x", "--expert-profile-save", "learned.bin"])


class DraftHeadHint(unittest.TestCase):
    """#474: a start that stopped at "the draft head does not fit" says what to change, from this start's log lines."""

    def log(self, text, before=""):
        d = tempfile.mkdtemp()
        p = Path(d) / "engine.log"
        p.write_text(before + text, encoding="utf-8")
        return str(p), len(before.encode())

    def test_the_engines_hint_is_relayed(self):
        p, off = self.log("strata mtp: the draft head over 106299 tokens needs 348 MiB of VRAM and 120 MiB is free.\n"
                          "strata mtp: hint: a smaller draft vocabulary needs less VRAM: --draft-vocab en (...)\n"
                          "strata serve: mtp: the draft head does not fit\n")
        h = start_failure_hint(p, off)
        self.assertIn("the draft head does not fit", h)
        self.assertIn("348 MiB", h)
        self.assertIn("--draft-vocab en", h)

    def test_an_older_engine_gets_the_advice_in_words(self):
        p, off = self.log("strata serve: mtp: the draft head does not fit\n")
        self.assertIn("--draft-vocab en", start_failure_hint(p, off))

    def test_other_failures_and_earlier_starts_add_nothing(self):
        p, off = self.log("strata serve: cannot open the pack\n")
        self.assertEqual(start_failure_hint(p, off), "")
        # an earlier start's failure (before this start's offset) is not this one's
        p, off = self.log("strata serve: cannot open the pack\n",
                          before="strata serve: mtp: the draft head does not fit\n")
        self.assertEqual(start_failure_hint(p, off), "")
        self.assertEqual(start_failure_hint(None, 0), "")
        self.assertEqual(start_failure_hint(str(Path(tempfile.mkdtemp()) / "missing.log"), 0), "")


class StartFailureLog(unittest.TestCase):
    """#496: whatever stopped the engine before READY, the error carries this start's last log lines."""

    def test_the_last_lines_of_this_start(self):
        from serve.server import start_log_tail
        d = tempfile.mkdtemp()
        p = Path(d) / "engine.log"
        before = "strata serve: an earlier start's line\n"
        p.write_text(before + "".join(f"strata serve: line {i}\n" for i in range(30)) + "\n"
                     "strata serve: cannot open the pack\n", encoding="utf-8")
        tail = start_log_tail(str(p), len(before.encode()))
        self.assertIn("the engine log's last lines:", tail)
        self.assertIn("cannot open the pack", tail)
        self.assertIn("line 29", tail)
        self.assertNotIn("line 10\n", tail + "\n")                 # 20 lines: 11..29 and the last one
        self.assertIn("line 11", tail)
        self.assertNotIn("earlier start", tail)
        short = start_log_tail(str(p), len(before.encode()), n=3)
        self.assertEqual(short.count("\n  "), 3)
        self.assertEqual(start_log_tail(str(p), p.stat().st_size), "")   # nothing from this start
        self.assertEqual(start_log_tail(None, 0), "")
        self.assertEqual(start_log_tail(str(Path(d) / "missing.log"), 0), "")

    def test_the_start_error_has_them(self):
        import serve.server as server
        fake = "import sys\nsys.stderr.write('strata serve: cannot open the pack packs/x\\n')\nsys.exit(2)\n"
        with tempfile.TemporaryDirectory() as d:
            script, log = Path(d) / "fake_strata.py", Path(d) / "strata.log"
            script.write_text(fake, encoding="utf-8")
            log.write_text("an earlier start\n", encoding="utf-8")
            real = server.subprocess.Popen
            with mock.patch.object(server.subprocess, "Popen",
                                   lambda cmd, **kw: real([sys.executable, str(script), *cmd[1:]], **kw)), \
                    mock.patch.object(server, "narrate_start", lambda *a, **k: None):
                with self.assertRaises(RuntimeError) as cm:
                    StrataEngine("strata", [], log=str(log))
            text = str(cm.exception)
            self.assertIn("exited before it was ready", text)
            self.assertIn("cannot open the pack packs/x", text)
            self.assertNotIn("an earlier start", text)


class StartNarrator(unittest.TestCase):
    """#505: the start's words say what happens to the experts - --mmap-experts loads nothing into RAM up front."""

    def test_the_words_follow_the_flags(self):
        from serve.server import experts_loading_words
        self.assertIn("loading the experts into RAM (about 38 GB)", experts_loading_words([], "about 38 GB"))
        mapped = experts_loading_words(["--mmap-experts"], "about 47 GB")
        self.assertIn("mapping the experts from the model files (about 47 GB", mapped)
        self.assertIn("not loaded into RAM", mapped)
        self.assertIn("the GPU does not hold", experts_loading_words(["--resident-experts"], "about 47 GB"))
        budget = experts_loading_words(["--mmap-experts", "--resident-budget-gib", "71"], "about 50 GB")
        self.assertIn("up to 71 GiB", budget)
        self.assertNotIn("about 50 GB", budget)

    def narrate(self, args, lines):
        import contextlib
        import io
        from serve.server import narrate_start
        d = tempfile.mkdtemp()
        log = Path(d) / "engine.log"
        log.write_text("".join(x + "\n" for x in lines), encoding="utf-8")
        done, out = threading.Event(), io.StringIO()
        with contextlib.redirect_stdout(out):
            t = threading.Thread(target=narrate_start, args=(str(log), 0, args, done))
            t.start()
            time.sleep(0.8)
            done.set()
            t.join(5)
        return out.getvalue()

    def test_a_mapped_start(self):
        said = self.narrate(["--mmap-experts"], ["strata generate: experts via mmap (--mmap-experts; the GGUF shards "
                                                 "in place, no experts.bin)"])
        self.assertIn("mapping the experts from the model files", said)
        self.assertNotIn("loading the experts into RAM", said)

    def test_an_arena_start(self):
        said = self.narrate([], ["strata generate: expert arena: resident, 31.64 GiB",
                                 "strata generate: loaded 31.64 GiB at 3.17 GiB/s"])
        self.assertIn("loading the experts into RAM (tens of GB)", said)
        self.assertIn("experts loaded: 31.64 GiB at 3.17 GiB/s", said)


class CancelledRead(unittest.TestCase):
    """#471: a request cancelled while its prompt was read is recorded with the tokens the engine read (the DONE
    line's 15th field), not the whole prompt; an older engine's line (no such field) keeps the whole prompt."""

    def test_parse_done_read_field(self):
        e = SimpleNamespace()
        StrataEngine._parse_done(e, "DONE 0 98179 17565.0 0.0 cancel 0 0 0 0 0 0 0 0.0 12288")
        self.assertEqual((e.last["prompt_tokens"], e.last["prompt_read"], e.last["finish"]), (98179, 12288, "cancel"))
        StrataEngine._parse_done(e, "DONE 0 98179 17565.0 0.0 cancel 0 0 0 0 0 0 0 0.0")
        self.assertNotIn("prompt_read", e.last)

    def test_prompt_tokens_seen(self):
        last = {"finish": "cancel", "reused": 1000, "prompt_read": 2000}
        self.assertEqual(prompt_tokens_seen(98179, last), 3000)
        self.assertEqual(prompt_tokens_seen(98179, {**last, "finish": "stop"}), 98179)   # read in full: all of it
        self.assertEqual(prompt_tokens_seen(98179, {"finish": "cancel", "reused": 0}), 98179)   # an older engine
        self.assertEqual(prompt_tokens_seen(98179, {}), 98179)                          # no DONE at all
        self.assertEqual(prompt_tokens_seen(10, {**last, "prompt_read": 50}), 10)       # never past the prompt

    def test_history_and_totals(self):
        tok = ByteTokenizer()
        engine = DoneLineEngine(tok, "</think>\n\nok", max_context=CTX, done_lines=[
            "DONE 0 20 400.0 0.0 cancel 0 0 0 0 0 0 0 0.0 8",      # stopped after 8 prompt tokens
            "DONE 4 20 40.0 30.0 stop 0 0 0 0 0 0 0 0.0 20",
            "DONE 0 20 400.0 0.0 cancel 0 0 0"])                   # an older engine: no read count
        svc = Service(engine, tok, ChatTemplate(ROOT / "serve/test_chat_template.jinja"))
        httpd = serve(svc, port=0)
        base = f"http://127.0.0.1:{httpd.server_address[1]}"
        try:
            for _ in range(3):
                body = json.dumps({"model": "m", "max_tokens": 10, "messages": [{"role": "user", "content": "hi"}]})
                with urllib.request.urlopen(urllib.request.Request(base + "/v1/chat/completions", data=body.encode(),
                                                                   headers={"Content-Type": "application/json"}),
                                            timeout=30) as r:
                    self.assertEqual(r.status, 200)
            with urllib.request.urlopen(base + "/metrics", timeout=10) as r:
                m = json.loads(r.read())
        finally:
            httpd.shutdown()
            httpd.server_close()
        rows = m["requests"][::-1]                                # oldest first
        total = rows[0]["prompt_total"]
        self.assertGreater(total, 8)
        self.assertEqual([r["prompt_total"] for r in rows], [total] * 3)
        self.assertEqual([(r["prompt_tokens"], r["prompt_read"]) for r in rows], [(8, 8), (total, 20), (total, None)])
        self.assertEqual(m["totals"]["prompt_tokens"], 8 + 2 * total)


class LiveRate(unittest.TestCase):
    """The Monitor's Speed readout: live.tok_s is a rate, and a request that never got a DONE keeps no counters.

    It used to be `generated / (now - first_token)` - the mean since the first token, whose first sample is
    1/elapsed.  Against a paced engine that reads five-digit numbers for the first instant of every answer and
    undershoots for the first second after that.  It is now the rate over the last RATE_WINDOW_S, with the mean
    still available as `live.tok_s_mean` for anyone who wants it."""

    PACE_S = 0.02                    # 50 tokens/s: a 30-token answer takes about 0.6 s
    TOKENS = 30

    def setUp(self):
        self.tok = ByteTokenizer()
        self.engine = MockEngine(self.tok, "x" * self.TOKENS, max_context=CTX, delay_s=self.PACE_S)
        self.svc = Service(self.engine, self.tok, ChatTemplate(ROOT / "serve/test_chat_template.jinja"))
        self.httpd = serve(self.svc, port=0)
        self.base = f"http://127.0.0.1:{self.httpd.server_address[1]}"

    def tearDown(self):
        self.httpd.shutdown()
        self.httpd.server_close()

    def metrics(self):
        with urllib.request.urlopen(self.base + "/metrics", timeout=10) as r:
            return json.loads(r.read())

    def test_prefill_rate_excludes_cached_tokens(self):
        import io
        import queue
        from types import SimpleNamespace
        engine = StrataEngine.__new__(StrataEngine)
        engine.proc = SimpleNamespace(stdin=io.StringIO(), poll=lambda: None)   # alive() asks it (#208)
        engine.lines = queue.Queue()
        engine.can_stop = False
        engine.max_context = 262144
        engine.prefill_tok_s_mean = 9999.0
        engine.lines.put("PP 10000 12000 2000 1000.0")  # 8000 cached, 2000 newly read in two seconds
        engine.lines.put("DONE 1 12000 4000 10 stop 0 0 8000")
        gen = engine.generate([1], 1, {}, threading.Event())
        self.assertIsNone(next(gen))
        self.assertEqual(engine.progress, (10000, 12000))
        self.assertEqual(engine.prefill_tok_s_mean, 1000.0)
        self.svc.engine = engine
        self.svc.status.update(busy=True, first_token=None)
        self.assertEqual(self.metrics()["live"]["prefill_tok_s_mean"], 1000.0)
        self.assertNotIn("prefill_tok_s", self.metrics()["live"])
        self.assertEqual(self.svc._prefill_tok_s_mean(), 1000.0)
        self.svc.status.update(first_token=time.time(), generated=1)
        self.assertEqual(self.svc._prefill_tok_s_mean(), 0.0)
        self.assertEqual(list(gen), [])
        timings = request_timings(12000, 1, engine.last)
        self.assertEqual(timings["prompt_per_second"], 1000.0)
        engine.lines.put("PP 8000 12000")
        engine.lines.put("DONE 0 12000 0 0 stop 0 0 12000")
        gen = engine.generate([1], 1, {}, threading.Event())
        next(gen)
        self.assertIsNone(engine.prefill_tok_s_mean)
        list(gen)
        self.svc.status["busy"] = False
        self.assertIsNone(self.metrics()["live"]["prefill_tok_s_mean"])

    def test_the_live_number_is_a_rate(self):
        live_samples, stop = [], threading.Event()

        def poll():                                   # what the Monitor polls, at 10 ms
            while not stop.is_set():
                live = self.metrics()["live"]
                if live["state"] == "generating" and live["tok_s"] is not None:
                    live_samples.append((live["generated"], live["tok_s"], live["tok_s_mean"]))
                time.sleep(0.01)

        body = json.dumps({"model": "m", "max_tokens": self.TOKENS, "temperature": 0,
                           "messages": [{"role": "user", "content": "hi"}]}).encode()
        watcher = threading.Thread(target=poll, daemon=True)
        watcher.start()
        t0 = time.time()
        try:
            with urllib.request.urlopen(urllib.request.Request(self.base + "/v1/chat/completions", data=body,
                                                               headers={"Content-Type": "application/json"}),
                                        timeout=30) as r:
                usage = json.loads(r.read())["usage"]
        finally:
            stop.set()
            watcher.join(2)
        true_rate = usage["completion_tokens"] / (time.time() - t0)
        self.assertGreaterEqual(len(live_samples), 3, "too few live readings to judge the readout")
        self.assertLess(max(s for _g, s, _m in live_samples), 4 * true_rate,
                        f"live.tok_s peaked at {max(s for _g, s, _m in live_samples):.1f} tok/s "
                        f"for a {true_rate:.1f} tok/s engine")
        self.assertEqual(self.metrics()["live"]["state"], "idle")
        self.assertIsNone(self.metrics()["live"]["tok_s"])

    def test_a_request_without_a_done_keeps_no_engine_counters(self):
        """An engine that dies mid-answer: the previous request's `last` must not become this row's decode rate."""
        class HalfDead(MockEngine):
            last = {"generated": 99, "prompt_tokens": 9, "prompt_ms": 10.0, "decode_ms": 100.0, "finish": "stop"}

            def generate(self, ids, max_new, sampling, cancel, embeddings=None):
                for i, t in enumerate(super().generate(ids, max_new, sampling, cancel, embeddings)):
                    if i == 3:
                        raise EngineDied("the engine stopped unexpectedly (exit code -9)")
                    yield t

        tok = ByteTokenizer()
        svc = Service(HalfDead(tok, "x" * self.TOKENS, max_context=CTX), tok,
                      ChatTemplate(ROOT / "serve/test_chat_template.jinja"))
        httpd = serve(svc, port=0)
        base = f"http://127.0.0.1:{httpd.server_address[1]}"
        try:
            body = json.dumps({"model": "m", "max_tokens": self.TOKENS, "stream": True,
                               "messages": [{"role": "user", "content": "hi"}]}).encode()
            with urllib.request.urlopen(urllib.request.Request(base + "/v1/chat/completions", data=body,
                                                               headers={"Content-Type": "application/json"}),
                                        timeout=30) as r:
                text = r.read().decode()
            self.assertIn('"error"', text)
            row = svc.metrics()["requests"][0]
            self.assertEqual(row["finish"], "error")
            self.assertEqual(row["output_tokens"], 3)
            self.assertIsNone(row["decode_tok_s"], "the previous request's counters were recorded as this one's")
            self.assertIsNone(row["engine_generated"])
        finally:
            httpd.shutdown()
            httpd.server_close()


class SharedSettings(unittest.TestCase):
    """The web app's "Use for other apps too": POST /settings makes its Chat settings every client's defaults."""

    @classmethod
    def setUpClass(cls):
        import tempfile

        class Sampled(RecordingEngine):
            def generate(self, ids, max_new, sampling, cancel, embeddings=None):
                self.last_sampling = dict(sampling or {})
                yield from super().generate(ids, max_new, sampling, cancel, embeddings)

        tok = ByteTokenizer()
        cls.engine = Sampled(tok, "</think>\n\nhello", max_context=CTX)
        cls.svc = Service(cls.engine, tok, ChatTemplate(ROOT / "serve/test_chat_template.jinja"))
        cls.tmp = tempfile.TemporaryDirectory()
        cls.svc.shared_path = os.path.join(cls.tmp.name, "strata-x.shared-settings.json")
        cls.httpd = serve(cls.svc, port=0)
        cls.base = f"http://127.0.0.1:{cls.httpd.server_address[1]}"

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()
        cls.tmp.cleanup()

    def req(self, path, body, headers=None, raw=None):
        h = {"Content-Type": "application/json", **(headers or {})}
        r = urllib.request.Request(self.base + path, data=raw if raw is not None else json.dumps(body).encode(), headers=h)
        try:
            with urllib.request.urlopen(r, timeout=30) as resp:
                return resp.status, json.loads(resp.read())
        except urllib.error.HTTPError as e:
            with e:
                return e.code, json.loads(e.read())

    def chat(self, **extra):
        return self.req("/v1/chat/completions", {"model": "m", "messages": [{"role": "user", "content": "hi"}], **extra})

    def tearDown(self):
        self.svc.set_shared(None)

    def test_other_apps_get_the_chat_settings(self):
        d = {"temperature": 0.3, "top_p": 0.9, "top_k": 10, "seed": 7, "max_tokens": 77,
             "reasoning_effort": "low", "experimental_speed_projection": False}
        code, b = self.req("/settings", {"defaults": d})
        self.assertEqual(code, 200, b)
        self.assertTrue(b["shared"])
        self.assertTrue(os.path.exists(self.svc.shared_path))
        code, _ = self.chat()                                        # a client that sets nothing
        self.assertEqual(code, 200)
        got = self.engine.last_sampling
        for k in ("temperature", "top_p", "top_k", "seed", "experimental_speed_projection"):
            self.assertEqual(got[k], d[k], k)
        self.assertEqual(self.engine.last_max_new, 77)
        code, _ = self.chat(temperature=0.9, max_tokens=5)          # its own values win
        self.assertEqual(self.engine.last_sampling["temperature"], 0.9)
        self.assertEqual(self.engine.last_max_new, 5)
        r = self.svc.with_shared({"messages": []}, "openai")
        self.assertEqual(r["reasoning_effort"], "low")
        self.assertEqual(self.svc.with_shared({"reasoning_effort": "high"}, "openai")["reasoning_effort"], "high")
        self.assertEqual(self.svc.with_shared({}, "anthropic")["output_config"], {"effort": "low"})

    def test_off_again(self):
        self.req("/settings", {"defaults": {"temperature": 0.3}})
        code, b = self.req("/settings", {"defaults": None})
        self.assertEqual((code, b["shared"]), (200, False))
        self.assertFalse(os.path.exists(self.svc.shared_path))
        self.chat()
        self.assertNotIn("temperature", self.engine.last_sampling)

    def test_only_strata_s_own_page_may_set_them(self):
        code, _ = self.req("/settings", None, {"Content-Type": "text/plain"}, raw=b'{"defaults": {"temperature": 1}}')
        self.assertEqual(code, 415)
        code, _ = self.req("/settings", {"defaults": {"temperature": 1}}, {"Origin": "http://evil.example"})
        self.assertEqual(code, 403)
        code, b = self.req("/settings", {"defaults": {"temperature": 9}})
        self.assertEqual(code, 400)
        self.assertIn("temperature", b["error"]["message"])
        self.assertEqual(self.svc.shared, {})
        host = self.base.split("://", 1)[1]
        code, _ = self.req("/settings", {"defaults": {"temperature": 1}}, {"Origin": "http://" + host})
        self.assertEqual(code, 200)

    def test_they_need_the_key_when_one_is_set(self):
        self.svc.api_key = "secret"
        try:
            self.assertEqual(self.req("/settings", {"defaults": {"temperature": 1}})[0], 401)
            self.assertEqual(self.req("/settings", {"defaults": {"temperature": 1}},
                                      {"Authorization": "Bearer secret"})[0], 200)
        finally:
            self.svc.api_key = ""

    def test_proxy_headers_do_not_make_a_page_strata_s_own(self):
        # #321: X-Forwarded-*, CF-Ray or CF-Connecting-IP say nothing about the page that sent the request - any web
        # page behind any proxy would otherwise change the settings (or run MCP tools)
        for extra in ({"X-Forwarded-Host": "proxy.example.com"}, {"CF-Ray": "1234567890"},
                      {"X-Forwarded-For": "203.0.113.9"}, {"CF-Connecting-IP": "203.0.113.9"}):
            code, _ = self.req("/settings", {"defaults": {"temperature": 1}},
                               {"Origin": "https://proxy.example.com", **extra})
            self.assertEqual(code, 403, extra)
        # another port on the same host is another site (a local dev server's page)
        code, _ = self.req("/settings", {"defaults": {"temperature": 1}}, {"Origin": "http://127.0.0.1:1"})
        self.assertEqual(code, 403)

    def test_a_trusted_origin_is_strata_s_own_page(self):
        # the web app behind a reverse proxy or tunnel: the config's trusted_origins
        self.svc.trusted_origins = ["https://strata.example.com"]
        try:
            code, _ = self.req("/settings", {"defaults": {}}, {"Origin": "https://strata.example.com"})
            self.assertEqual(code, 200)
            code, _ = self.req("/settings", {"defaults": {}}, {"Origin": "https://evil.example.com"})
            self.assertEqual(code, 403)
        finally:
            self.svc.trusted_origins = []


class WebApp(unittest.TestCase):
    """The web app (PR #22's dashboard idea, rebuilt): its page and files, and GET /metrics."""

    @classmethod
    def setUpClass(cls):
        tok = ByteTokenizer()
        cls.svc = Service(RecordingEngine(tok, "</think>\n\nhello", max_context=CTX), tok,
                          ChatTemplate(ROOT / "serve/test_chat_template.jinja"))
        cls.httpd = serve(cls.svc, port=0)
        cls.base = f"http://127.0.0.1:{cls.httpd.server_address[1]}"

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()

    def get(self, path, headers=None):
        req = urllib.request.Request(self.base + path, headers=headers or {})
        try:
            with urllib.request.urlopen(req, timeout=10) as r:
                return r.status, r.headers.get("Content-Type", ""), r.read()
        except urllib.error.HTTPError as e:
            return e.code, e.headers.get("Content-Type", ""), e.read()

    def test_page_and_files(self):
        code, ctype, body = self.get("/")
        self.assertEqual(code, 200)
        self.assertIn("text/html", ctype)
        self.assertIn(b"\"web/app.js\"", body)   # relative since #82 (works behind a path-prefixed proxy)
        for path, want in (("/web/app.js", "javascript"), ("/web/app.css", "text/css"), ("/web/tokens.css", "text/css"),
                           ("/web/components.css", "text/css"), ("/web/sprite.svg", "image/svg+xml")):
            with self.subTest(path=path):
                code, ctype, _ = self.get(path)
                self.assertEqual(code, 200)
                self.assertIn(want, ctype)

    def test_only_the_app_files_are_served(self):
        for path in ("/web/..%2Fserver.py", "/web/index.html", "/web/test.py", "/fonts/..%2F..%2Fsetup.py",
                     "/fonts/missing.woff2", "/fonts/x.ttf"):
            with self.subTest(path=path):
                self.assertEqual(self.get(path)[0], 404)

    def test_metrics(self):
        data = json.dumps({"model": "m", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 5}).encode()
        urllib.request.urlopen(urllib.request.Request(self.base + "/v1/chat/completions", data=data,
                                                      headers={"Content-Type": "application/json"}), timeout=10).read()
        code, ctype, body = self.get("/metrics")
        self.assertEqual(code, 200)
        m = json.loads(body)
        for key in ("engine", "live", "requests", "hardware", "hardware_static", "history"):
            self.assertIn(key, m)
        self.assertEqual(m["engine"]["max_context"], CTX)
        self.assertEqual(m["live"]["state"], "idle")
        self.assertEqual(m["requests"][0]["output_tokens"], 5)

    def test_model_discovery_and_props(self):
        svc = self.svc
        previous = svc.engine.max_context, svc.vision, svc.sampling_defaults, svc.shared
        try:
            svc.engine.max_context = 262144
            svc.sampling_defaults = {"temperature": 1.0, "repetition_penalty": 1.1}
            svc.shared = {"temperature": 0.7, "max_tokens": 4096}
            for vision in (None, object()):
                svc.vision = vision
                for path in ("/models", "/v1/models"):
                    code, _, body = self.get(path)
                    self.assertEqual(code, 200)
                    models = json.loads(body)["data"]
                    self.assertEqual(len(models), 1)
                    model = models[0]
                    self.assertEqual(model["id"], svc.model)
                    self.assertEqual(model["status"]["value"], "loaded")
                    self.assertEqual(model["meta"]["n_ctx"], 262144)
                    self.assertEqual(model["architecture"]["input_modalities"],
                                     ["text", "image"] if vision else ["text"])
                code, _, body = self.get("/props?model=" + svc.model + "&autoload=false")
                self.assertEqual(code, 200)
                props = json.loads(body)
                self.assertEqual(props["default_generation_settings"]["n_ctx"], 262144)
                self.assertEqual(props["default_generation_settings"]["params"],
                                 {"temperature": 0.7, "repeat_penalty": 1.1, "n_predict": 4096})
                self.assertEqual(props["chat_template"], (ROOT / "serve/test_chat_template.jinja").read_text(encoding="utf-8"))
                self.assertEqual(props["modalities"]["vision"], vision is not None)
                self.assertEqual(props["total_slots"], 1)
                self.assertFalse(props["models_autoload"])
            svc.shared = {}
            props = json.loads(self.get("/props")[2])
            self.assertEqual(props["default_generation_settings"]["params"]["n_predict"], -1)
            self.assertEqual(self.get("/props?model=not-loaded&autoload=true")[0], 404)
        finally:
            svc.engine.max_context, svc.vision, svc.sampling_defaults, svc.shared = previous

    def test_discovery_needs_the_api_key(self):
        self.svc.api_key = "secret"
        try:
            for path in ("/models", "/v1/models", "/props", "/slots"):
                self.assertEqual(self.get(path)[0], 401)
                self.assertEqual(self.get(path, {"Authorization": "Bearer secret"})[0], 200)
        finally:
            self.svc.api_key = ""

    def test_build_model_path_and_slot_status(self):
        engine = self.svc.engine
        engine.model_path = "models/example.gguf"
        engine.info = {"version": "0.1.21"}
        try:
            props = json.loads(self.get("/props")[2])
            self.assertEqual(props["model_path"], engine.model_path)
            self.assertEqual(props["build_info"], "Strata 0.1.21")
            for busy in (True, False):
                with self.svc.status_lock:
                    self.svc.status["busy"] = busy
                code, _, body = self.get("/slots")
                self.assertEqual(code, 200)
                self.assertEqual(json.loads(body), [{"id": 0, "n_ctx": CTX, "is_processing": busy}])
        finally:
            with self.svc.status_lock:
                self.svc.status["busy"] = False
            del engine.model_path, engine.info
        props = json.loads(self.get("/props")[2])
        self.assertNotIn("build_info", props)
        self.assertNotIn("model_path", props)

    def test_discovery_does_not_restart_a_dead_engine(self):
        self.svc.engine.alive = lambda: False
        try:
            for path in ("/models", "/v1/models"):
                code, _, body = self.get(path)
                self.assertEqual(code, 200)
                self.assertEqual(json.loads(body)["data"], [])
            self.assertEqual(self.get("/props")[0], 503)
            self.assertEqual(json.loads(self.get("/slots")[2]), [])
        finally:
            del self.svc.engine.alive

    def test_metrics_need_the_key_when_one_is_set(self):
        self.svc.api_key = "secret"
        try:
            self.assertEqual(self.get("/metrics")[0], 401)
            self.assertEqual(self.get("/metrics", {"Authorization": "Bearer secret"})[0], 200)
            self.assertEqual(self.get("/")[0], 200)                  # the page itself asks for the key
        finally:
            self.svc.api_key = ""


class ClockedEngine(MockEngine):
    """The mock engine with StrataEngine's clock: `last` as the engine's DONE line gives it, the conversation cache
    holding the first REUSED tokens of every prompt."""
    REUSED = 5

    def generate(self, ids, max_new, sampling, cancel, embeddings=None):
        n = 0
        try:
            for t in super().generate(ids, max_new, sampling, cancel, embeddings):
                n += 1
                yield t
        finally:          # as StrataEngine reads its DONE line: also when the server closes the request at a stop token
            self.last = {"generated": n, "prompt_tokens": len(ids), "prompt_ms": 40.0, "decode_ms": 20.0 * n,
                         "finish": "stop", "reused": min(self.REUSED, len(ids)), "hits": 9, "lookups": 10}


class UsageAndStatus(unittest.TestCase):
    """What clients read besides the text: the part of the prompt the conversation cache held (OpenAI's
    prompt_tokens_details.cached_tokens, Anthropic's cache_read_input_tokens), llama.cpp's timings, GET /v1/status."""

    @classmethod
    def setUpClass(cls):
        tok = ByteTokenizer()
        cls.engine = ClockedEngine(tok, "</think>\n\nok", max_context=CTX)
        cls.svc = Service(cls.engine, tok, ChatTemplate(ROOT / "serve/test_chat_template.jinja"))
        cls.httpd = serve(cls.svc, port=0)
        cls.base = f"http://127.0.0.1:{cls.httpd.server_address[1]}"

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()

    def request(self, path, body=None):
        req = urllib.request.Request(self.base + path, data=None if body is None else json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json", "anthropic-version": "2023-06-01"})
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status, r.read()

    def chat(self, path, stream=False):
        body = {"model": "x", "max_tokens": 20, "messages": [{"role": "user", "content": "hi"}], "stream": stream}
        status, raw = self.request(path, body)
        self.assertEqual(status, 200)
        if not stream:
            return json.loads(raw)
        return [json.loads(line[6:]) for line in raw.decode().splitlines()
                if line.startswith("data: {")]

    def preflight(self, path, origin="https://chat.example.com"):
        req = urllib.request.Request(self.base + path, method="OPTIONS",
                                     headers={"Origin": origin, "Access-Control-Request-Method": "POST",
                                              "Access-Control-Request-Headers": "authorization, content-type"})
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.status, r.headers

    def test_cors_is_off_by_default(self):
        # #321: OPTIONS is answered, but without cors_origins no page of another origin is let in
        status, h = self.preflight("/v1/chat/completions")
        self.assertEqual(status, 204)
        self.assertIsNone(h.get("Access-Control-Allow-Origin"))
        with urllib.request.urlopen(self.base + "/health", timeout=10) as r:
            self.assertIsNone(r.headers.get("Access-Control-Allow-Origin"))

    def test_cors_for_the_configured_origins_on_the_api_only(self):
        self.svc.cors_origins = ["https://chat.example.com"]
        try:
            status, h = self.preflight("/v1/chat/completions")
            self.assertEqual((status, h.get("Access-Control-Allow-Origin")), (204, "https://chat.example.com"))
            self.assertIn("OPTIONS", h.get("Access-Control-Allow-Methods", ""))
            self.assertIn("authorization", h.get("Access-Control-Allow-Headers", ""))
            self.assertIsNone(self.preflight("/v1/chat/completions", "https://evil.example.com")[1]
                              .get("Access-Control-Allow-Origin"))
            # never on the app's own endpoints (/settings, /unload, ...)
            self.assertIsNone(self.preflight("/settings")[1].get("Access-Control-Allow-Origin"))
            req = urllib.request.Request(self.base + "/v1/models", headers={"Origin": "https://chat.example.com"})
            with urllib.request.urlopen(req, timeout=10) as r:
                self.assertEqual(r.headers.get("Access-Control-Allow-Origin"), "https://chat.example.com")
            self.svc.cors_origins = ["*"]
            self.assertEqual(self.preflight("/v1/messages", "https://any.example.com")[1]
                             .get("Access-Control-Allow-Origin"), "*")
        finally:
            self.svc.cors_origins = []

    def test_sse_is_not_buffered_by_proxies(self):
        req = urllib.request.Request(self.base + "/v1/chat/completions", headers={"Content-Type": "application/json"},
                                     data=json.dumps({"model": "m", "stream": True,
                                                      "messages": [{"role": "user", "content": "hi"}]}).encode())
        with urllib.request.urlopen(req, timeout=30) as r:
            self.assertEqual(r.headers.get("X-Accel-Buffering"), "no")
            r.read()

    def test_origin_lists_are_checked(self):
        from serve.server import origins_of
        self.assertEqual(origins_of(None, "k", True), [])
        self.assertEqual(origins_of("https://a.example.com/", "k", False), ["https://a.example.com"])
        self.assertEqual(origins_of(["*", "http://localhost:3000"], "k", True), ["*", "http://localhost:3000"])
        for bad in ("*", "a.example.com", "https://a.example.com/path", "https://*.example.com", 5):
            with self.assertRaises(SystemExit):
                origins_of(bad, "k", False)

    def test_openai(self):
        b = self.chat("/v1/chat/completions")
        u, t = b["usage"], b["timings"]
        self.assertEqual(u["prompt_tokens_details"]["cached_tokens"], ClockedEngine.REUSED)
        self.assertEqual(t["cache_n"], ClockedEngine.REUSED)
        self.assertEqual(t["prompt_n"] + t["cache_n"], u["prompt_tokens"])
        self.assertEqual(t["predicted_n"], u["completion_tokens"])
        self.assertAlmostEqual(t["prompt_per_second"], t["prompt_n"] / 0.040, delta=0.1)
        self.assertAlmostEqual(t["predicted_per_second"], 50.0, delta=0.1)            # 20 ms a token

    def test_openai_stream(self):
        last = self.chat("/v1/chat/completions", stream=True)[-1]
        self.assertEqual(last["usage"]["prompt_tokens_details"]["cached_tokens"], ClockedEngine.REUSED)
        self.assertEqual(last["timings"]["cache_n"], ClockedEngine.REUSED)

    def test_anthropic(self):
        u = self.chat("/v1/messages")["usage"]
        self.assertEqual(u["cache_read_input_tokens"], ClockedEngine.REUSED)
        self.assertEqual(u["input_tokens"] + u["cache_read_input_tokens"], len(self.engine.last_prompt))
        self.assertGreater(u["output_tokens"], 0)

    def test_v1_status(self):
        self.chat("/v1/chat/completions")
        status, raw = self.request("/v1/status")
        self.assertEqual(status, 200)
        s = json.loads(raw)
        self.assertEqual(s["model"], self.svc.model)
        self.assertEqual(s["context"]["max_positions"], CTX)
        self.assertEqual(s["concurrency"]["serving"], 1)
        self.assertFalse(s["vision"]["available"])
        self.assertEqual(s["activity"]["in_flight"], 0)
        self.assertGreaterEqual(s["activity"]["requests"], 1)
        self.assertEqual(s["last_timings"]["cache_n"], ClockedEngine.REUSED)
        self.assertIn("at", s["last_timings"])

    def test_no_clock(self):
        """An engine without a clock (MockEngine): no timings, nothing cached."""
        tok = ByteTokenizer()
        svc = Service(MockEngine(tok, "</think>\n\nok", max_context=CTX), tok,
                      ChatTemplate(ROOT / "serve/test_chat_template.jinja"))
        httpd = serve(svc, port=0)
        try:
            data = json.dumps({"model": "x", "max_tokens": 5, "messages": [{"role": "user", "content": "hi"}]}).encode()
            req = urllib.request.Request(f"http://127.0.0.1:{httpd.server_address[1]}/v1/chat/completions", data=data,
                                         headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=30) as r:
                b = json.loads(r.read())
            self.assertNotIn("timings", b)
            self.assertEqual(b["usage"]["prompt_tokens_details"]["cached_tokens"], 0)
            self.assertIsNone(svc.v1_status()["last_timings"])
        finally:
            httpd.shutdown()
            httpd.server_close()


class TimingsDrafts(unittest.TestCase):
    """`timings` carries the speculative draft counts (PR #83's fields) only when the engine reported them."""

    def test_draft_fields(self):
        base = {"prompt_ms": 100.0, "decode_ms": 200.0, "generated": 20, "reused": 4}
        t = request_timings(24, 20, dict(base, drafts_offered=15, drafts_accepted=11))
        self.assertEqual((t["draft_n"], t["draft_n_accepted"]), (15, 11))
        self.assertEqual((t["prompt_n"], t["cache_n"]), (20, 4))
        self.assertNotIn("draft_n", request_timings(24, 20, base))
        self.assertIsNone(request_timings(24, 20, {}))


class UnloadableEngine(MockEngine):
    """A mock engine that can be stopped and started again like StrataEngine (alive / unload / restart)."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.running, self.unloaded, self.starts = True, False, 0

    def alive(self):
        return self.running

    def unload(self):
        self.running, self.unloaded = False, True

    def restart(self):
        self.running, self.unloaded = True, False
        self.starts += 1


class SharingTheGpu(unittest.TestCase):
    """Idle unload, POST /unload and /load, the free-VRAM guard and the before_load hook (all off by default)."""

    def setUp(self):
        tok = ByteTokenizer()
        self.engine = UnloadableEngine(tok, "</think>\n\nok", max_context=CTX)
        self.svc = Service(self.engine, tok, ChatTemplate(ROOT / "serve/test_chat_template.jinja"))
        self.httpd = serve(self.svc, port=0)
        self.base = f"http://127.0.0.1:{self.httpd.server_address[1]}"

    def tearDown(self):
        self.httpd.shutdown()
        self.httpd.server_close()

    def req(self, path, body=None):
        r = urllib.request.Request(self.base + path, data=None if body is None else json.dumps(body).encode(),
                                   headers={"Content-Type": "application/json"},
                                   method="GET" if body is None else "POST")
        try:
            with urllib.request.urlopen(r, timeout=30) as resp:
                return resp.status, json.loads(resp.read())
        except urllib.error.HTTPError as e:
            with e:
                return e.code, json.loads(e.read())

    def chat(self):
        return self.req("/v1/chat/completions", {"model": "m", "messages": [{"role": "user", "content": "hi"}]})

    def test_unload_then_the_next_request_loads(self):
        self.assertEqual(self.req("/unload", {}), (200, {"status": "unloaded"}))
        self.assertFalse(self.engine.alive())
        self.assertEqual(self.req("/health")[1]["loaded"], False)
        self.assertEqual(self.req("/v1/models")[1]["data"][0]["status"]["value"], "unloaded")
        self.assertEqual(self.req("/unload", {}), (200, {"status": "not loaded"}))
        s, b = self.chat()
        self.assertEqual(s, 200)
        self.assertEqual(b["choices"][0]["message"]["content"], "ok")
        self.assertEqual(self.engine.starts, 1)
        self.assertEqual(self.req("/health")[1]["loaded"], True)

    def test_load_endpoint(self):
        self.svc.unload()
        self.assertEqual(self.req("/load", {}), (200, {"status": "loaded"}))
        self.assertTrue(self.engine.alive())
        self.assertEqual(self.req("/load", {}), (200, {"status": "loaded"}))
        self.assertEqual(self.engine.starts, 1)

    def test_unload_refused_while_a_request_runs(self):
        with self.svc.fifo:
            self.assertEqual(self.svc.unload(), "busy")
        self.assertTrue(self.engine.alive())

    def test_idle_unload(self):
        self.svc.idle_unload_s = 1
        self.svc.last_request_at = time.time()
        self.assertEqual(self.svc.unload(idle_for=1), "busy")       # a request just now: not idle yet
        self.svc.start_idle_unload()
        deadline = time.time() + 10
        while self.engine.alive() and time.time() < deadline:
            time.sleep(0.1)
        self.assertFalse(self.engine.alive())
        self.assertEqual(self.chat()[0], 200)

    def test_min_free_vram_refuses_to_load(self):
        self.svc.min_free_vram_mib = 8000
        self.svc.free_vram_mib = lambda: 2000
        self.svc.unload()
        t0 = time.time()
        s, b = self.chat()
        self.assertEqual(s, 503)
        self.assertIn("in use by another program", b["error"]["message"])
        self.assertFalse(self.engine.alive())
        self.assertGreater(time.time() - t0, 10)                    # waited for memory being given back first
        self.svc.free_vram_mib = lambda: 9000
        self.assertEqual(self.chat()[0], 200)

    def test_min_free_vram_unreadable_loads(self):
        self.svc.min_free_vram_mib = 8000
        self.svc.free_vram_mib = lambda: None                       # unreadable: never refuse
        self.svc.unload()
        self.assertEqual(self.chat()[0], 200)

    def test_before_load_runs_first(self):
        mark = Path(tempfile.mkdtemp()) / "ran"
        self.svc.before_load = [sys.executable, "-c", f"open({str(mark)!r}, 'w').close()"]
        self.svc.unload()
        self.assertFalse(mark.exists())
        self.assertEqual(self.chat()[0], 200)
        self.assertTrue(mark.exists())

    def test_vision_encoder_unloads_and_starts_first(self):
        order = []

        class FakeVision:
            running = True

            def alive(self):
                return self.running

            def unload(self):
                self.running = False

            def restart(self):
                order.append("vision")
                self.running = True

        engine_restart = self.engine.restart
        self.engine.restart = lambda: (order.append("engine"), engine_restart())
        self.svc.vision = FakeVision()
        self.assertEqual(self.svc.unload(), "unloaded")
        self.assertFalse(self.svc.vision.alive())
        self.assertEqual(self.chat()[0], 200)
        self.assertEqual(order, ["vision", "engine"])             # the encoder first, as at a start
        self.assertTrue(self.svc.vision.alive())

    def test_off_by_default(self):
        self.assertEqual((self.svc.idle_unload_s, self.svc.min_free_vram_mib, self.svc.before_load), (0, 0, None))
        self.assertEqual(self.req("/health")[1]["loaded"], True)


class ThinkingEngine(MockEngine):
    """Thinks THOUGHT, then answers; a prompt that already ends its thinking (the budget's wrap-up) gets the answer
    at once, the way the model continues after </think>.  Records every prompt it is given."""
    THOUGHT = "Let me think step by step about two plus two. " * 4          # 184 reasoning tokens (one per byte)
    ANSWER = "The answer is 4."

    def __init__(self, tok):
        super().__init__(tok, "x", max_context=CTX)
        self.prompts = []

    def generate(self, ids, max_new, sampling, cancel, embeddings=None):
        self.prompts.append(list(ids))
        done = self.tok.decode(ids).endswith("</think>\n\n")
        text = self.ANSWER if done else self.THOUGHT + "</think>\n\n" + self.ANSWER
        for t in (self.tok.encode(text) + self.tok.encode("<|im_end|>", parse_special=True))[:max_new]:
            if cancel.is_set():
                return
            yield t


class ThinkingBudget(unittest.TestCase):
    """#123: reasoning_budget_tokens (opt-in): at the budget the thinking is wrapped up and the model answers,
    continuing from the prompt plus what it generated plus the wrap-up (a prefix the engine already holds)."""

    def setUp(self):
        self.tok = ByteTokenizer()
        self.engine = ThinkingEngine(self.tok)
        self.svc = Service(self.engine, self.tok, ChatTemplate(ROOT / "serve/test_chat_template.jinja"))
        self.httpd = serve(self.svc, port=0)
        self.base = f"http://127.0.0.1:{self.httpd.server_address[1]}"

    def tearDown(self):
        self.httpd.shutdown()
        self.httpd.server_close()

    def post(self, path, body):
        req = urllib.request.Request(self.base + path, data=json.dumps(body).encode(), headers={
            "Content-Type": "application/json", "anthropic-version": "2023-06-01"})
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                raw = r.read().decode()
                return r.status, (json.loads(raw) if not body.get("stream") else raw)
        except urllib.error.HTTPError as e:
            with e:
                return e.code, json.loads(e.read())

    def openai(self, **extra):
        return self.post("/v1/chat/completions", {"model": "m", "messages": [{"role": "user", "content": "2+2?"}],
                                                  "max_tokens": 400, **extra})

    def test_off_by_default(self):
        code, b = self.openai()
        self.assertEqual(code, 200, b)
        msg = b["choices"][0]["message"]
        self.assertEqual((msg["reasoning_content"], msg["content"]), (ThinkingEngine.THOUGHT, ThinkingEngine.ANSWER))
        self.assertEqual(len(self.engine.prompts), 1)

    def test_the_budget_wraps_up_the_thinking(self):
        from serve.server import REASONING_WRAP_UP
        code, b = self.openai(reasoning_budget_tokens=20)
        self.assertEqual(code, 200, b)
        msg = b["choices"][0]["message"]
        wrap = REASONING_WRAP_UP.split("</think>")[0]
        self.assertEqual(msg["reasoning_content"], ThinkingEngine.THOUGHT[:20] + wrap)
        self.assertEqual(msg["content"], ThinkingEngine.ANSWER)
        self.assertEqual(b["choices"][0]["finish_reason"], "stop")
        first, second = self.engine.prompts
        extra = self.tok.encode(REASONING_WRAP_UP, parse_special=True)
        self.assertEqual(second, first + self.tok.encode(ThinkingEngine.THOUGHT[:20]) + extra)   # a prefix + more
        self.assertEqual(b["usage"]["completion_tokens"], 20 + len(extra) + len(ThinkingEngine.ANSWER) + 1)
        self.assertEqual(b["usage"]["prompt_tokens"], len(first))

    def test_anthropic_stream(self):
        from serve.server import REASONING_WRAP_UP
        code, raw = self.post("/v1/messages", {"model": "m", "max_tokens": 400, "stream": True,
                                               "reasoning_budget_tokens": 30,
                                               "messages": [{"role": "user", "content": "2+2?"}]})
        self.assertEqual(code, 200)
        evs = [json.loads(line[6:]) for line in raw.splitlines() if line.startswith("data: {")]
        thinking = "".join(e["delta"].get("thinking", "") for e in evs if e["type"] == "content_block_delta")
        text = "".join(e["delta"].get("text", "") for e in evs if e["type"] == "content_block_delta")
        self.assertEqual(thinking, ThinkingEngine.THOUGHT[:30] + REASONING_WRAP_UP.split("</think>")[0])
        self.assertEqual(text, ThinkingEngine.ANSWER)
        self.assertEqual(evs[-2]["delta"]["stop_reason"], "end_turn")

    def test_a_budget_the_thinking_stays_under(self):
        code, b = self.openai(reasoning_budget_tokens=10_000)
        self.assertEqual(b["choices"][0]["message"]["reasoning_content"], ThinkingEngine.THOUGHT)
        self.assertEqual(len(self.engine.prompts), 1)

    def test_the_config_default_and_a_request_that_turns_it_off(self):
        self.svc.reasoning_budget_tokens = 20
        code, b = self.openai()
        self.assertEqual(len(self.engine.prompts), 2)
        self.assertTrue(b["choices"][0]["message"]["reasoning_content"].startswith(ThinkingEngine.THOUGHT[:20] + "\n"))
        code, b = self.openai(reasoning_budget_tokens=0)
        self.assertEqual(b["choices"][0]["message"]["reasoning_content"], ThinkingEngine.THOUGHT)
        self.assertEqual(len(self.engine.prompts), 3)

    def test_without_thinking_there_is_nothing_to_limit(self):
        self.engine.THOUGHT = ""
        code, b = self.openai(reasoning_budget_tokens=5, reasoning_effort="none")
        self.assertEqual(code, 200, b)
        self.assertEqual(len(self.engine.prompts), 1)

    def test_no_room_left_to_answer(self):
        code, b = self.openai(reasoning_budget_tokens=20, max_tokens=40)    # 20 thought + the wrap-up > 40
        self.assertEqual(b["choices"][0]["finish_reason"], "length")
        self.assertEqual(len(self.engine.prompts), 1)
        self.assertEqual(b["choices"][0]["message"]["reasoning_content"], ThinkingEngine.THOUGHT[:20])

    def test_a_reply_cut_while_thinking_is_named_in_the_log(self):
        """#530: max tokens reached inside the thinking gives an empty answer; the server log says what helps."""
        hint = "reached max tokens while still thinking"
        for extra, said in (({"max_tokens": 10}, True), ({}, False)):
            with self.subTest(extra=extra):
                out = io.StringIO()
                with contextlib.redirect_stdout(out):
                    code, b = self.openai(**extra)
                self.assertEqual(code, 200, b)
                self.assertEqual(b["choices"][0]["finish_reason"], "length" if said else "stop")
                self.assertEqual(hint in out.getvalue(), said, out.getvalue())
                if said:
                    self.assertIn("reasoning_budget_tokens", out.getvalue())

    def test_a_bad_value_is_a_400(self):
        for bad in ("lots", 2.5, True, [1]):
            with self.subTest(value=bad):
                code, b = self.openai(reasoning_budget_tokens=bad)
                self.assertEqual(code, 400)
                self.assertIn("reasoning_budget_tokens", b["error"]["message"])
        self.assertEqual(self.engine.prompts, [])


class StatusHandover(unittest.TestCase):
    """#266: a stream aborted mid-way and the next request, which was waiting for the fifo.  The aborted request's
    status/history block ran after the fifo was released, so the waiting request could start in that gap: the old
    request then recorded the NEW request's status as its own, set busy=False and popped `tail`, and the new request
    crashed in _note (KeyError 'tail').  The fifo below lets the waiting request run to its first token as soon as
    it is released, before the releasing thread goes on - the worst case of that gap, every time."""

    def test_abort_then_the_next_request(self):
        tok = ByteTokenizer()
        second_running = threading.Event()

        class Engine(MockEngine):
            calls = 0

            def generate(self, ids, max_new, sampling, cancel, embeddings=None):
                Engine.calls += 1
                me = Engine.calls
                for i, t in enumerate(super().generate(ids, max_new, sampling, cancel, embeddings)):
                    if me == 2 and i == 2:
                        second_running.set()        # the second request has its status and two tokens noted
                    yield t

        class SlowRelease:
            """A Lock whose release waits (briefly) until the thread it let in has started generating."""

            def __init__(self):
                self.lock, self.armed = threading.Lock(), False

            def acquire(self, blocking=True):
                return self.lock.acquire(blocking)

            def __enter__(self):
                self.lock.acquire()

            def __exit__(self, *exc):
                self.lock.release()
                if self.armed:
                    self.armed = False
                    second_running.wait(5)

            def release(self):
                self.lock.release()

        svc = Service(Engine(tok, "</think>\n\n" + "y" * 40, max_context=CTX), tok,
                      ChatTemplate(ROOT / "serve/test_chat_template.jinja"))
        svc.fifo = SlowRelease()
        ids = tok.encode("hi")
        first = svc.run(ids, True, None, 30, {}, threading.Event())
        for _ in range(5):
            next(first)                                 # mid-answer
        out, errors = [], []

        def second():
            try:
                out.extend(svc.run(ids, True, None, 20, {}, threading.Event()))
            except Exception as e:                      # noqa: BLE001 - the crash this test is about
                errors.append(e)

        waiter = threading.Thread(target=second)
        waiter.start()
        deadline = time.time() + 5
        while svc.status.get("queued") != 1 and time.time() < deadline:
            time.sleep(0.005)
        self.assertEqual(svc.status.get("queued"), 1, "the second request never queued")
        svc.fifo.armed = True
        first.close()                                   # the client went away: GeneratorExit in the first request
        waiter.join(10)
        self.assertFalse(waiter.is_alive())
        self.assertEqual(errors, [])
        self.assertEqual(out[-1][0], "done")
        self.assertEqual(out[-1][1]["completion_tokens"], 20)
        rows = list(svc.history)
        self.assertEqual([r["finish"] for r in rows], ["disconnect", "length"])
        self.assertTrue(0 < rows[0]["output_tokens"] < 30, rows)     # where the first one stopped, not the second's
        self.assertEqual(rows[1]["output_tokens"], 20)
        self.assertEqual(svc.totals["requests"], 2)
        self.assertEqual(svc.totals["output_tokens"], rows[0]["output_tokens"] + 20)
        self.assertFalse(svc.status["busy"])
        self.assertNotIn("tail", svc.status)


FAKE_STRATA = '''import pathlib, sys, time
gate = pathlib.Path(sys.argv[sys.argv.index("--gate") + 1])
print("INFO engine=0.0.0", flush=True)
while not gate.exists():                     # the test says when the engine is "ready"
    time.sleep(0.01)
print("READY 4096 stop", flush=True)
for line in sys.stdin:
    if line.startswith("QUIT"):
        break
'''


class RestartWindow(unittest.TestCase):
    """#344: while the engine restarts it is not alive (a request waits for the restart instead of reading
    max_context 0), and a request that still meets max_context 0 gets a 503 "starting", not a 400 about its prompt."""

    def test_not_alive_until_ready(self):
        from unittest import mock
        import serve.server as server
        with tempfile.TemporaryDirectory() as d:
            script, gate = Path(d) / "fake_strata.py", Path(d) / "ready"
            script.write_text(FAKE_STRATA, encoding="utf-8")
            real = server.subprocess.Popen
            with mock.patch.object(server.subprocess, "Popen",
                                   lambda cmd, **kw: real([sys.executable, str(script), *cmd[1:]], **kw)):
                gate.touch()
                eng = StrataEngine("strata", ["--gate", str(gate)])
                try:
                    self.assertEqual(eng.max_context, 4096)
                    self.assertTrue(eng.alive())
                    gate.unlink()
                    old = eng.proc
                    old.kill()
                    old.wait(10)
                    deadline = time.time() + 10
                    while not getattr(eng, "ended", False) and time.time() < deadline:
                        time.sleep(0.01)                  # the server has noticed: its output closed
                    self.assertFalse(eng.alive())
                    t = threading.Thread(target=eng.restart)
                    t.start()
                    deadline = time.time() + 10
                    while eng.proc is old and time.time() < deadline:
                        time.sleep(0.005)
                    self.assertIsNot(eng.proc, old, "restart never started the engine")
                    time.sleep(0.2)                       # the new engine is up, but has not said READY
                    self.assertEqual(eng.max_context, 0)
                    self.assertFalse(eng.alive(), "alive before READY: a request would plan with context 0")
                    gate.touch()
                    t.join(10)
                    self.assertFalse(t.is_alive())
                    self.assertTrue(eng.alive())
                    self.assertEqual(eng.max_context, 4096)
                    # restart() of a running engine kills it first: that engine's output thread ends after the new
                    # one is up, and must not mark it dead (it did: every request after restarted it again)
                    eng.restart()
                    time.sleep(0.5)
                    self.assertTrue(eng.alive())
                    self.assertEqual(eng.max_context, 4096)
                finally:
                    gate.touch()
                    eng.unload()

    def test_context_zero_is_503(self):
        tok = ByteTokenizer()
        svc = Service(MockEngine(tok, "</think>\n\nok", max_context=0), tok,
                      ChatTemplate(ROOT / "serve/test_chat_template.jinja"))
        httpd = serve(svc, port=0)
        base = f"http://127.0.0.1:{httpd.server_address[1]}"
        try:
            for path, body in (("/v1/chat/completions", {"model": "m", "messages": [{"role": "user", "content": "hi"}],
                                                         "max_tokens": 16}),
                               ("/v1/messages", {"model": "m", "max_tokens": 16,
                                                 "messages": [{"role": "user", "content": "hi"}]})):
                req = urllib.request.Request(base + path, data=json.dumps(body).encode(),
                                             headers={"Content-Type": "application/json"})
                with self.assertRaises(urllib.error.HTTPError) as cm:
                    urllib.request.urlopen(req, timeout=30)
                with cm.exception as e:
                    self.assertEqual(e.code, 503, path)
                    text = e.read().decode()
                self.assertIn("starting", text)
                self.assertNotIn("leaves no room", text)
        finally:
            httpd.shutdown()
            httpd.server_close()


class ModelAliases(unittest.TestCase):
    """#297: the config's `aliases` are listed by /v1/models and accepted as model names (answered under that name)."""

    @classmethod
    def setUpClass(cls):
        tok = ByteTokenizer()
        cls.svc = Service(MockEngine(tok, "</think>\n\nok", max_context=CTX), tok,
                          ChatTemplate(ROOT / "serve/test_chat_template.jinja"),
                          model_name="qwen3.8-flash-next-iq3_xxs")
        cls.svc.set_aliases(["qwen", "local-model", "qwen", " ", "qwen3.8-flash-next-iq3_xxs"])
        cls.httpd = serve(cls.svc, port=0)
        cls.base = f"http://127.0.0.1:{cls.httpd.server_address[1]}"

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()

    def get(self, path):
        with urllib.request.urlopen(self.base + path, timeout=30) as r:
            return json.loads(r.read())

    def post(self, path, body):
        req = urllib.request.Request(self.base + path, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.read().decode()

    def test_set_aliases(self):
        self.assertEqual(self.svc.aliases, ["qwen", "local-model"])        # duplicates, blanks and the name itself
        tok = ByteTokenizer()
        svc = Service(MockEngine(tok, "x", max_context=CTX), tok, ChatTemplate(ROOT / "serve/test_chat_template.jinja"))
        svc.set_aliases("a, b")
        self.assertEqual(svc.aliases, ["a", "b"])
        svc.set_aliases(None)
        self.assertEqual(svc.aliases, [])
        for bad in (3, ["a", 1], {"a": 1}):
            with self.assertRaises(ValueError):
                svc.set_aliases(bad)

    def test_models_lists_them(self):
        data = self.get("/v1/models")["data"]
        self.assertEqual([m["id"] for m in data], ["qwen3.8-flash-next-iq3_xxs", "qwen", "local-model"])
        self.assertEqual(data[0]["aliases"], ["qwen", "local-model"])
        self.assertEqual(data[1]["alias_of"], "qwen3.8-flash-next-iq3_xxs")
        self.assertEqual(self.get("/props?model=local-model")["model_alias"], "qwen3.8-flash-next-iq3_xxs")

    def test_requests_are_answered_under_the_alias(self):
        msgs = [{"role": "user", "content": "hi"}]
        for asked, want in (("qwen", "qwen"), ("local-model", "local-model"),
                            ("qwen3.8-flash-next-iq3_xxs", "qwen3.8-flash-next-iq3_xxs"),
                            ("something-else", "qwen3.8-flash-next-iq3_xxs")):    # still served, as before
            with self.subTest(asked=asked):
                out = json.loads(self.post("/v1/chat/completions", {"model": asked, "messages": msgs, "max_tokens": 8}))
                self.assertEqual(out["model"], want)
                text = self.post("/v1/chat/completions", {"model": asked, "messages": msgs, "max_tokens": 8,
                                                          "stream": True})
                first = json.loads(text.split("data: ", 2)[1].strip())
                self.assertEqual(first["model"], want)
                out = json.loads(self.post("/v1/messages", {"model": asked, "messages": msgs, "max_tokens": 8}))
                self.assertEqual(out["model"], want)

    def test_without_aliases_nothing_changes(self):
        tok = ByteTokenizer()
        svc = Service(MockEngine(tok, "x", max_context=CTX), tok, ChatTemplate(ROOT / "serve/test_chat_template.jinja"))
        httpd = serve(svc, port=0)
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{httpd.server_address[1]}/v1/models", timeout=30) as r:
                data = json.loads(r.read())["data"]
            self.assertEqual(len(data), 1)
            self.assertNotIn("aliases", data[0])
            self.assertEqual(svc.model_for({"model": "x"}), svc.model)
        finally:
            httpd.shutdown()
            httpd.server_close()


class SilentEngine(unittest.TestCase):
    """#481: an engine that prints nothing for engine_silence_s during a request (or never acknowledges a STOP) has
    lost step with the server: it is ended and the request fails with EngineDied, instead of waiting forever."""

    def bare(self, silence, can_stop=False):
        import io
        import queue
        engine = StrataEngine.__new__(StrataEngine)
        engine.proc = mock.Mock()
        engine.proc.stdin = io.StringIO()
        engine.proc.poll.return_value = None
        engine.lines, engine.can_stop, engine.max_context = queue.Queue(), can_stop, 4096
        engine.silence_s, engine.log_path = silence, None
        return engine

    def later(self, engine, delay, *lines):
        def put():
            time.sleep(delay)
            for x in lines:
                engine.lines.put(x)
        threading.Thread(target=put, daemon=True).start()

    def test_silence_mid_answer_ends_the_engine(self):
        from serve.server import EngineSilent
        engine = self.bare(0.3)
        engine.lines.put("T 5")
        gen = engine.generate([1], 10, {}, threading.Event())
        self.assertEqual(next(gen), 5)
        t0 = time.monotonic()
        with self.assertRaises(EngineSilent) as cm:
            next(gen)
        self.assertIsInstance(cm.exception, EngineDied)          # every EngineDied path handles it
        self.assertLess(time.monotonic() - t0, 5)
        engine.proc.kill.assert_called_once()
        self.assertFalse(engine.alive())                          # the next request restarts it
        self.assertIn("#481", engine.death_note())
        self.assertNotIn("STOP", engine.proc.stdin.getvalue())    # nothing is listening: no STOP, no drain

    def test_prompt_chunks_set_the_wait(self):
        # a PP line every second, at 100 tok/s: far over a 0.3 s silence, but each chunk is on time for its size
        engine = self.bare(0.3)
        engine.lines.put("RESUME 0")
        engine.lines.put("PP 100 300 1000 100.0")
        self.later(engine, 1.0, "PP 200 300 2000 100.0", "T 7", "DONE 1 300 2000 1 length")
        self.assertEqual(list(engine.generate([1], 10, {}, threading.Event())), [None, None, 7])
        engine.proc.kill.assert_not_called()

    def test_a_long_first_chunk_is_allowed(self):
        # 100 prompt tokens at the slowest prompt reading (50 tok/s): 2 s on top of the silence before the first PP
        engine = self.bare(0.3)
        self.later(engine, 1.0, "PP 100 100 1000 100.0", "DONE 0 100 1000 0 length")
        self.assertEqual(list(engine.generate([1] * 100, 10, {}, threading.Event())), [None])
        engine.proc.kill.assert_not_called()

    def test_a_stop_never_acknowledged(self):
        from serve.server import EngineSilent
        engine = self.bare(0.3, can_stop=True)
        engine.lines.put("T 5")
        gen = engine.generate([1], 10, {}, threading.Event())
        self.assertEqual(next(gen), 5)
        with self.assertRaises(EngineSilent):
            gen.close()                                           # the consumer stopped: STOP, then the drain
        self.assertIn("STOP", engine.proc.stdin.getvalue())
        engine.proc.kill.assert_called_once()
        self.assertFalse(engine.alive())

    def test_zero_waits_as_before(self):
        engine = self.bare(0)
        self.later(engine, 0.5, "T 5", "DONE 1 1 1 1 length")
        self.assertEqual(list(engine.generate([1], 10, {}, threading.Event())), [5])

    def test_config(self):
        from serve.server import ENGINE_SILENCE_S, engine_silence_s
        self.assertEqual(engine_silence_s({}), ENGINE_SILENCE_S)
        self.assertEqual(engine_silence_s({"engine_silence_s": 0}), 0.0)
        self.assertEqual(engine_silence_s({"engine_silence_s": 900}), 900.0)
        for bad in (-1, "300", True):
            with self.assertRaises(ValueError):
                engine_silence_s({"engine_silence_s": bad})


FAKE_LOST_STEP = '''import pathlib, sys, time
mark = pathlib.Path(sys.argv[sys.argv.index("--mark") + 1])
mode = sys.argv[sys.argv.index("--mode") + 1]
print("READY 4096 stop", flush=True)
for line in sys.stdin:
    if line.startswith("QUIT"):
        break
    if line.startswith("GEN"):
        if not mark.exists():                    # the first engine loses step (#481): it never says DONE
            mark.touch()
            print("T 104", flush=True)
            if mode == "stop":
                print("T 257", flush=True)       # <|im_end|>: the server STOPs and drains, and the engine is silent
            time.sleep(3600)
        print("T 111", flush=True)
        print("T 107", flush=True)
        print("DONE 2 5 1.0 1.0 length", flush=True)
'''


class LostStep(unittest.TestCase):
    """#481 over HTTP, with a real process: the request with the silent engine ends with an error, and the next one
    starts the engine again and is answered (the server used to wait forever, holding the request FIFO)."""

    def run_mode(self, mode, stream):
        import serve.server as server
        with tempfile.TemporaryDirectory() as d:
            script, mark = Path(d) / "fake_strata.py", Path(d) / "lost"
            script.write_text(FAKE_LOST_STEP, encoding="utf-8")
            real = server.subprocess.Popen
            with mock.patch.object(server.subprocess, "Popen",
                                   lambda cmd, **kw: real([sys.executable, str(script), *cmd[1:]], **kw)):
                eng = StrataEngine("strata", ["--mark", str(mark), "--mode", mode])
                eng.silence_s = 0.5
                tok = ByteTokenizer()
                svc = Service(eng, tok, ChatTemplate(ROOT / "serve/test_chat_template.jinja"))
                httpd = serve(svc, port=0)
                base = f"http://127.0.0.1:{httpd.server_address[1]}"
                body = {"model": "m", "max_tokens": 2, "reasoning_effort": "none", "stream": stream,
                        "messages": [{"role": "user", "content": "hi"}]}
                try:
                    first = eng.proc
                    req = urllib.request.Request(base + "/v1/chat/completions", data=json.dumps(body).encode(),
                                                 headers={"Content-Type": "application/json"})
                    try:
                        with urllib.request.urlopen(req, timeout=60) as r:
                            text = r.read().decode()
                    except urllib.error.HTTPError as e:
                        self.assertEqual(e.code, 503)
                        text = e.read().decode()
                    self.assertIn("the next request restarts it", text)
                    self.assertIsNotNone(first.poll(), "the silent engine still runs")
                    self.assertEqual(svc.history[-1]["finish"], "error")
                    with urllib.request.urlopen(req, timeout=60) as r:
                        text = r.read().decode()
                    self.assertIsNot(eng.proc, first)
                    if stream:
                        answer = "".join(json.loads(x[6:])["choices"][0]["delta"].get("content") or ""
                                         for x in text.splitlines() if x.startswith("data: {") and "choices" in x)
                    else:
                        answer = json.loads(text)["choices"][0]["message"]["content"]
                    self.assertEqual(answer, "ok")
                finally:
                    httpd.shutdown()
                    httpd.server_close()
                    eng.unload()

    def test_silent_mid_answer(self):
        self.run_mode("silent", stream=False)

    def test_silent_mid_stream(self):
        self.run_mode("silent", stream=True)

    def test_stop_never_acknowledged(self):
        self.run_mode("stop", stream=False)


if __name__ == "__main__":
    unittest.main()
