"""serve/test_tool_markup_retry.py - (a): a reply to a request with tools that ends in tool-call markup that is no call (a
name the request did not declare, a call cut before its end, a block that reads as no call) is written again once, inside
the same request; strata_unread_tool_markup marks a reply that still ends in such markup.  The 43 forms are the shapes of a
user's report of agent turns that stopped (499 turns, 63 sessions): its 30 undeclared names and 13 cut or unreadable blocks,
rewritten with neutral text.  Against a scripted engine (no GPU, no pack).

    python -m unittest serve.test_tool_markup_retry -v
"""
from __future__ import annotations

import json
import os
import sys
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve.frontend import ChatTemplate, OutputParser  # noqa: E402
from serve.server import (ByteTokenizer, MockEngine, Service, ToolMarkupHold, closest_declared,  # noqa: E402
                          dead_markup_tail, markup_names, serve)

ROOT = Path(__file__).resolve().parents[1]
DECLARED = ["pwsh", "read", "write", "edit", "glob", "grep", "background_jobs", "ask_user_question", "todo_write",
            "web_fetch", "web_search"]
P = {"type": "object", "properties": {"command": {"type": "string"}, "description": {"type": "string"},
                                      "file_path": {"type": "string"}, "content": {"type": "string"}}}
TOOLS = [{"name": n, "parameters": P} for n in DECLARED]


def call(name, **params):
    body = "".join(f"<parameter={k}>\n{v}\n</parameter>\n" for k, v in params.items())
    return f"<tool_call>\n<function={name}>\n{body}</function>\n</tool_call>"


# The report's 30 undeclared names (the name the model wrote, the declared one the hint should give, if any)
REJECTED = [("delete", None), ("delete", None), ("delete", None), ("psh", "pwsh"), ("psh:", "pwsh"),
            ("read_file", "read"), ("read_file", "read"), ("pash", "pwsh"), ("exec", None), ("exec", None),
            ("exec", None), ("exec", None), ("exec", None), ("Bash", None), ("PowerShell", None), ("exec", None),
            ("background_job", "background_jobs"), ("background_job", "background_jobs"), ("task", None),
            ("task", None), ("powrsh_placeholder", None), ("write_file", "write"), ("write_file", "write"),
            ("psn", None), ("psh", "pwsh"), ("pash", "pwsh"), ("ask_user_questions", "ask_user_question"),
            ("gread", "read"), ("run", None), ("pwash", "pwsh")]
# The report's cut and unreadable blocks, as shapes (the reason the server gives)
CUT_OR_GARBAGE = [
    ("Let me look at the folder first.\n<tool_call>", "cut_off"),                                    # bare opener
    ("<tool_call>\n<parameter name=\"command\">ls\n</parameter>\n<parameter=description>\nlist the", "cut_off"),
    ("<tool_call>\n<function=pw", "cut_off"),
    ("<tool_call>\n<=>\n</parameter>\n<parameter=command>\nls\n</parameter>\n</function>\n</tool_call>", "unreadable"),
    ("I will run it now.\n<tool_call>\n<=true>\n</invoke>\n</function>\n</tool_call>", "unreadable"),
    ("<tool_call>", "cut_off"),
    ("<tool_call>\n<=>\n</function>\n</tool_call>", "unreadable"),
    ("Next step.\n\n<tool_call>", "cut_off"),
    ("<tool_call>\n", "cut_off"),
    ("<tool_call>\n</function>\n</tool_call>", "unreadable"),
    ("<tool_call>\n<=command>\n</function>\n</tool_call>", "unreadable"),
    ("<tool_call>\n<function=", "cut_off"),
]
PROSE_ONLY = "I will read the file and then edit the line that sets the port."   # the report's 13th: no markup at all


def rejected_text(i, name):
    lead = "Checking the log.\n" if i % 4 == 1 else ""
    return lead + call(name, command="Get-ChildItem .\\logs", description="list the logs")


def run_hold(raw, step=0):
    """-> (events sent, the hold) after the parser and the hold read `raw` and the model stopped, as Service._run does."""
    p = OutputParser(thinking=False, tools=TOOLS, stream_tools=True, recover=True, declared_only=True)
    hold, out = ToolMarkupHold(True), []
    w = step or len(raw) or 1
    for i in range(0, len(raw), w):
        out += hold.push(p.feed(raw[i:i + w]))
    out += hold.push(p.finish("stop"))
    return out, hold


class DeadMarkupTail(unittest.TestCase):
    def test_ends(self):
        for text, want in [("<tool_call>", "cut_off"), ("ok\n<tool_call>\n<function=x>\n", "cut_off"),
                           ("ok\n" + call("x", a=1), "unreadable"), ("ok\n<function=x>\n</function>", "unreadable"),
                           ("ok\n<function=x>\n<parameter=a>", "cut_off"),
                           ("<tool_call>\n</function>\n</tool_call>\n\n", "unreadable")]:
            with self.subTest(text=text):
                got = dead_markup_tail(text)
                self.assertIsNotNone(got)
                self.assertEqual(got[1], want)
                self.assertTrue(text[got[0]:].startswith(("<tool_call>", "<function=")))

    def test_not_an_end(self):
        for text in ["plain answer", call("x", a=1) + "\n\nDone: the file is fixed.",
                     "Write `<tool_call>` to call a tool.", "```\n<tool_call>\n<function=x>\n```",
                     "```xml\n" + call("x", a=1), "see <function=x> in the doc"]:
            with self.subTest(text=text):
                self.assertIsNone(dead_markup_tail(text))

    def test_a_run_of_blocks_starts_at_the_first(self):
        text = "Two tries.\n" + call("a", q=1) + "\n" + call("b", q=2) + "\n"
        start, reason = dead_markup_tail(text)
        self.assertEqual((text[start:start + 11], reason), ("<tool_call>", "unreadable"))
        self.assertEqual(text[:start], "Two tries.\n")


class ReportShapes(unittest.TestCase):
    """The parser and the hold over the report's 43 stops: 42 end in markup the server writes again; prose does not."""

    def test_all_43(self):
        verdicts = []
        for step in (0, 1):
            seen = []
            for i, (name, near) in enumerate(REJECTED):
                raw = rejected_text(i, name)
                out, hold = run_hold(raw, step)
                dead = dead_markup_tail(hold.text) if hold.held and not hold.calls else None
                with self.subTest(case=i, step=step):
                    self.assertIsNotNone(dead, raw)
                    names = [x for x in markup_names(hold.text) if x not in DECLARED]
                    self.assertEqual(names, [name.strip()])
                    self.assertEqual(closest_declared(names[0], DECLARED), near)
                    sent = "".join(e.text for e in out if e.kind == "content")
                    self.assertEqual(sent, raw[:raw.index("<tool_call>")])      # the words before it went out
                seen.append("rejected")
            for raw, reason in CUT_OR_GARBAGE:
                out, hold = run_hold(raw, step)
                dead = dead_markup_tail(hold.text) if hold.held and not hold.calls else None
                with self.subTest(raw=raw, step=step):
                    self.assertIsNotNone(dead)
                    self.assertEqual(dead[1], reason)
                    self.assertFalse(any(e.kind in ("tool_call", "tool_start") for e in out))
                seen.append(reason)
            out, hold = run_hold(PROSE_ONLY, step)
            self.assertFalse(hold.held)
            self.assertEqual("".join(e.text for e in out), PROSE_ONLY)
            verdicts.append(seen)
        self.assertEqual(verdicts[0], verdicts[1])
        self.assertEqual(len(verdicts[0]), 42)

    def test_a_declared_call_is_never_held(self):
        raw = "Reading it.\n" + call("read", file_path="a.txt")
        out, hold = run_hold(raw, 1)
        self.assertTrue(hold.calls)
        self.assertEqual([e.call.name for e in out if e.kind == "tool_call"], ["read"])

    def test_prose_after_the_block_releases_it(self):
        raw = call("exec", command="ls") + "\n\nThat tool does not exist here, so here is the answer: 4."
        out, hold = run_hold(raw, 1)
        self.assertFalse(hold.held)
        self.assertEqual("".join(e.text for e in out if e.kind == "content"), raw)


class Scripted(MockEngine):
    """Each generate() call (a pass, the retry included) gets the next text; the last one repeats."""

    def __init__(self, tok, texts):
        super().__init__(tok, "x", max_context=32768)
        self.texts, self.prompts = list(texts), []

    def generate(self, ids, max_new, sampling, cancel, embeddings=None):
        self.prompts.append(list(ids))
        text = self.texts[min(len(self.prompts) - 1, len(self.texts) - 1)]
        yield from (self.tok.encode(text) + self.tok.encode("<|im_end|>", parse_special=True))[:max_new]


OPENAI_TOOLS = [{"type": "function", "function": {"name": n, "parameters": P}} for n in DECLARED]


class ServerRetry(unittest.TestCase):
    def start(self, texts, env=None):
        self.tok = ByteTokenizer()
        self.engine = Scripted(self.tok, texts)
        with mock.patch.dict(os.environ, env or {}):
            self.svc = Service(self.engine, self.tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
        self.httpd = serve(self.svc, port=0)
        self.base = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        self.addCleanup(self.httpd.server_close)
        self.addCleanup(self.httpd.shutdown)

    def post(self, path, body):
        req = urllib.request.Request(self.base + path, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=30) as r:
            raw = r.read().decode()
            return json.loads(raw) if not body.get("stream") else raw

    def chat(self, tools=OPENAI_TOOLS, **extra):
        return self.post("/v1/chat/completions", {"model": "m", "max_tokens": 4000, "tools": tools,
                                                  "messages": [{"role": "user", "content": "list the logs"}], **extra})

    def metric(self, name):
        with urllib.request.urlopen(self.base + "/metrics/prometheus", timeout=30) as r:
            for line in r.read().decode().splitlines():
                if line.startswith(name + " "):
                    return float(line.split()[1])
        return None

    def test_an_undeclared_name_is_written_again_as_a_declared_call(self):
        good = call("pwsh", command="Get-ChildItem .\\logs")
        self.start(["</think>\n\nChecking.\n" + call("exec", command="Get-ChildItem .\\logs"), good])
        b = self.chat()
        msg = b["choices"][0]["message"]
        self.assertEqual(b["choices"][0]["finish_reason"], "tool_calls")
        self.assertEqual([c["function"]["name"] for c in msg["tool_calls"]], ["pwsh"])
        self.assertNotIn("exec", msg.get("content") or "")
        self.assertEqual((msg.get("content") or "").strip(), "Checking.")
        self.assertNotIn("strata_rejected_tool_calls", b)
        self.assertNotIn("strata_unread_tool_markup", b)
        self.assertEqual(len(self.engine.prompts), 2)
        retry = self.tok.decode(self.engine.prompts[1])
        note = retry[len(self.tok.decode(self.engine.prompts[0])):]
        self.assertIn("`exec`, which is not one of the tools you can call", note)
        for n in DECLARED:
            self.assertIn(n, note)
        self.assertTrue(note.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n"))
        self.assertEqual(self.metric("strata:tool_markup_retries_total"), 1)

    def test_the_hint_names_the_closest_declared_tool(self):
        self.start(["</think>\n\n" + call("read_file", file_path="a.txt"), call("read", file_path="a.txt")])
        b = self.chat()
        self.assertEqual([c["function"]["name"] for c in b["choices"][0]["message"]["tool_calls"]], ["read"])
        self.assertIn("Did you mean `read`?", self.tok.decode(self.engine.prompts[1]))

    def test_a_forced_call_is_opened_again(self):
        """tool_choice "required": the retry's reply opens with the call's opening too, as the first one did."""
        self.start([call("exec", command="ls")[len("<tool_call>\n"):], call("pwsh", command="ls")[len("<tool_call>\n"):]])
        b = self.chat(tool_choice="required", chat_template_kwargs={"enable_thinking": False})
        self.assertEqual(b["choices"][0]["finish_reason"], "tool_calls")
        self.assertEqual([c["function"]["name"] for c in b["choices"][0]["message"]["tool_calls"]], ["pwsh"])
        self.assertTrue(self.tok.decode(self.engine.prompts[1]).endswith("</think>\n\n<tool_call>\n"))
        self.assertFalse((b["choices"][0]["message"].get("content") or "").strip())

    def test_one_retry_only_then_the_text_and_the_markers(self):
        for first, again, field, want in [
                (call("exec", command="ls"), call("exec", command="ls"), "strata_rejected_tool_calls",
                 [{"name": "exec", "reason": "undeclared"}]),
                ("<tool_call>", "<tool_call>\n<=>\n</function>\n</tool_call>", "strata_unread_tool_markup",
                 {"reason": "unreadable", "chars": len("<tool_call>\n<=>\n</function>\n</tool_call>")}),
                ("<tool_call>\n</function>\n</tool_call>", "Sure.\n<tool_call>", "strata_unread_tool_markup",
                 {"reason": "cut_off", "chars": len("<tool_call>")})]:
            with self.subTest(first=first):
                self.start(["</think>\n\n" + first, again])
                for stream in (False, True):
                    self.engine.prompts.clear()
                    b = self.chat(stream=stream)
                    self.assertEqual(len(self.engine.prompts), 2, "one retry, never a second")
                    if stream:
                        chunks = [json.loads(x[6:]) for x in b.splitlines() if x.startswith("data: {")]
                        text = "".join(c["choices"][0]["delta"].get("content") or "" for c in chunks if c["choices"])
                        got = [c[field] for c in chunks if field in c]
                        self.assertEqual(len(got), 1)
                        got = got[0]
                    else:
                        text = b["choices"][0]["message"]["content"]
                        got = b.get(field)
                    self.assertEqual(text.strip(), again.strip())
                    self.assertEqual(got, want)
                    if field == "strata_unread_tool_markup":
                        self.assertEqual(text[len(text) - got["chars"]:].strip(), text[text.index("<tool_call>"):].strip())

    def test_the_rollback_switch_sends_the_text_as_before(self):
        dead = "<tool_call>\n<=>\n</function>\n</tool_call>"
        self.start(["</think>\n\nOne moment.\n" + dead, call("pwsh", command="ls")],
                   env={"STRATA_TOOL_MARKUP_RETRY": "0"})
        b = self.chat()
        self.assertEqual(len(self.engine.prompts), 1)
        self.assertEqual(b["choices"][0]["message"]["content"], "One moment.\n" + dead)
        self.assertEqual(b["strata_unread_tool_markup"], {"reason": "unreadable", "chars": len(dead)})
        self.assertEqual(self.metric("strata:tool_markup_retries_total"), 0)

    def test_no_room_left_no_retry(self):
        """The note is prompt: with too few output tokens left after it, the reply goes out as written."""
        dead = "<tool_call>\n<=>\n</function>\n</tool_call>"
        self.start(["</think>\n\n" + dead, call("pwsh", command="ls")])
        b = self.chat(max_tokens=400)               # one token per byte here: the note alone is longer
        self.assertEqual(len(self.engine.prompts), 1)
        self.assertEqual(b["strata_unread_tool_markup"], {"reason": "unreadable", "chars": len(dead)})

    def test_no_tools_no_retry_and_no_marker(self):
        self.start(["</think>\n\nHere: <tool_call>", call("pwsh", command="ls")])
        b = self.post("/v1/chat/completions", {"model": "m", "max_tokens": 300,
                                               "messages": [{"role": "user", "content": "hi"}]})
        self.assertEqual(len(self.engine.prompts), 1)
        self.assertNotIn("strata_unread_tool_markup", b)

    def test_streamed_markup_never_reaches_the_client_when_the_retry_calls(self):
        self.start(["</think>\n\nChecking.\n" + call("Bash", command="ls"), call("pwsh", command="ls")])
        raw = self.chat(stream=True)
        self.assertNotIn("Bash", raw)
        self.assertIn('"pwsh"', raw)

    def test_messages_and_responses_carry_the_marker(self):
        dead = "<tool_call>\n<=>\n</function>\n</tool_call>"
        want = {"reason": "unreadable", "chars": len(dead)}
        self.start(["</think>\n\n" + dead])
        a = self.post("/v1/messages", {"model": "m", "max_tokens": 4000, "messages": [{"role": "user", "content": "x"}],
                                       "tools": [{"name": n, "input_schema": P} for n in DECLARED]})
        self.assertEqual(a.get("strata_unread_tool_markup"), want)
        s = self.post("/v1/messages", {"model": "m", "max_tokens": 4000, "stream": True,
                                       "messages": [{"role": "user", "content": "x"}],
                                       "tools": [{"name": n, "input_schema": P} for n in DECLARED]})
        deltas = [json.loads(x[6:]) for x in s.splitlines() if x.startswith("data: {") and "message_delta" in x]
        self.assertEqual(deltas[-1].get("strata_unread_tool_markup"), want)
        r = self.post("/v1/responses", {"model": "m", "max_output_tokens": 4000, "input": "x",
                                        "tools": [{"type": "function", "name": n, "parameters": P} for n in DECLARED]})
        self.assertEqual(r.get("strata_unread_tool_markup"), want)


if __name__ == "__main__":
    unittest.main()
