"""serve/test_stop_open_string.py - a tool call whose string value is still open when the model's output ends with a normal stop (EOS) is never
repaired into a complete call, on any of the three APIs, streamed or not (soak 11-17, s005 turn 36).

The mock engine replays a script and then the end-of-turn token (finish "stop"), like a model that stops in the middle of a value:
  A  the content value is open, `path` never written;
  B  content and path written, the function never closed;
  C  `path` first, then the content value open (the order in which a client would write a truncated file);
  D  a fully closed call: stays a call (control);
  E  like C, but cut by max_tokens (the existing cut handling, control);
  F  the MODEL closes the call itself right after a value that stops mid-sentence, no path: a closed call is a call, delivered as written, which a client
     that checks the required parameters rejects (the t36 case; the server cannot tell it from any other closed call).
What the client sees for A, B, C: no call at all (non-stream chat and messages), or the announced call's argument text ending unterminated (streamed chat and
messages), or a Responses item with status "incomplete" and unterminated arguments; always strata_unread_tool_markup {reason: cut_off}; never valid JSON.

    python -m pytest serve/test_stop_open_string.py -q
"""
import json
import sys
import unittest
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve.frontend import ChatTemplate  # noqa: E402
from serve.server import ByteTokenizer, MockEngine, Service, serve  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
BODY = "def audit(ctx):\n    return {'ok': True}\n" * 8
HEAD = "x</think>\n\n<tool_call>\n<function=write_file>\n"
SCRIPTS = {
    "A": HEAD + "<parameter=content>\n" + BODY[:150],
    "B": HEAD + "<parameter=content>\n" + BODY[:150] + "\n</parameter>\n<parameter=path>\ncore/a.py\n</parameter>\n",
    "C": HEAD + "<parameter=path>\ncore/a.py\n</parameter>\n<parameter=content>\n" + BODY[:150],
    "D": HEAD + "<parameter=path>\ncore/a.py\n</parameter>\n<parameter=content>\n" + BODY[:150] + "\n</parameter>\n</function>\n</tool_call>",
    "F": HEAD + "<parameter=content>\n" + BODY[:150] + "\n</parameter>\n</function>\n</tool_call>",
}
PARAMS = {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}
APIS = {
    "chat": ("/v1/chat/completions", lambda s, cap: {"model": "m", "messages": [{"role": "user", "content": "go"}], "stream": s, "max_tokens": cap,
                                                     "tools": [{"type": "function", "function": {"name": "write_file", "description": "w", "parameters": PARAMS}}]}),
    "messages": ("/v1/messages", lambda s, cap: {"model": "m", "messages": [{"role": "user", "content": "go"}], "stream": s, "max_tokens": cap,
                                                 "tools": [{"name": "write_file", "description": "w", "input_schema": PARAMS}]}),
    "responses": ("/v1/responses", lambda s, cap: {"model": "m", "input": "go", "stream": s, "max_output_tokens": cap,
                                                   "tools": [{"type": "function", "name": "write_file", "description": "w", "parameters": PARAMS}]}),
}


def sse(raw):
    out = []
    for block in raw.replace("\r\n", "\n").split("\n\n"):
        data = [ln[5:].strip() for ln in block.split("\n") if ln.startswith("data:")]
        if data and data[0] != "[DONE]":
            out.append(json.loads("\n".join(data)))
    return out


def seen(api, stream, body):
    """-> (finish, calls: [argument text], item statuses, markers) as the client sees them"""
    calls, finish, status, marks = {}, None, [], {}

    def marks_of(js):
        for k in ("strata_unread_tool_markup", "strata_rejected_tool_calls"):
            v = js.get(k) or (js.get("response") or {}).get(k)
            if v:
                marks[k] = v
    if not stream:
        js = json.loads(body)
        marks_of(js)
        if api == "chat":
            ch = js["choices"][0]
            finish = ch["finish_reason"]
            for i, tc in enumerate(ch["message"].get("tool_calls") or []):
                calls[i] = tc["function"]["arguments"]
        elif api == "messages":
            finish = js["stop_reason"]
            for i, b in enumerate(js["content"]):
                if b.get("type") == "tool_use":
                    calls[i] = json.dumps(b["input"])
        else:
            finish = js["status"] if js["status"] == "incomplete" else "completed"
            for i, it in enumerate(js["output"]):
                if it["type"] == "function_call":
                    calls[i] = it["arguments"]
                    status.append(it["status"])
        return finish, [calls[i] for i in sorted(calls)], status, marks
    for js in sse(body):
        t = js.get("type")
        marks_of(js)
        if api == "chat":
            for ch in js.get("choices") or []:
                for tc in (ch.get("delta") or {}).get("tool_calls") or []:
                    calls[tc.get("index", 0)] = calls.get(tc.get("index", 0), "") + ((tc.get("function") or {}).get("arguments") or "")
                finish = ch.get("finish_reason") or finish
        elif api == "messages":
            if t == "content_block_delta" and js["delta"].get("type") == "input_json_delta":
                calls[js["index"]] = calls.get(js["index"], "") + js["delta"].get("partial_json", "")
            if t == "message_delta":
                finish = js["delta"].get("stop_reason")
        elif t in ("response.completed", "response.incomplete"):
            finish = "incomplete" if t == "response.incomplete" else "completed"
            for i, it in enumerate(js["response"]["output"]):
                if it["type"] == "function_call":
                    calls[i] = it["arguments"]
                    status.append(it["status"])
    return finish, [calls[i] for i in sorted(calls)], status, marks


def valid_object(text):
    try:
        return isinstance(json.loads(text), dict)
    except ValueError:
        return False


class StopOpenString(unittest.TestCase):
    def run_case(self, script, cap=4000):
        tok = ByteTokenizer()
        svc = Service(MockEngine(tok, script, max_context=65536), tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
        httpd = serve(svc, port=0)
        base = "http://127.0.0.1:%d" % httpd.server_address[1]
        try:
            for api, (path, make) in APIS.items():
                for stream in (False, True):
                    req = urllib.request.Request(base + path, data=json.dumps(make(stream, cap)).encode(), headers={"Content-Type": "application/json"})
                    with urllib.request.urlopen(req, timeout=60) as r:
                        self.assertEqual(r.status, 200)
                        yield api, stream, seen(api, stream, r.read().decode("utf-8"))
        finally:
            httpd.shutdown()
            httpd.server_close()

    def test_an_open_string_at_stop_is_never_a_complete_call(self):
        for case in ("A", "B", "C"):
            for api, stream, (finish, calls, status, marks) in self.run_case(SCRIPTS[case]):
                with self.subTest(case=case, api=api, stream=stream):
                    self.assertEqual(marks.get("strata_unread_tool_markup", {}).get("reason"), "cut_off", marks)
                    self.assertFalse([c for c in calls if valid_object(c)], "a repaired call: %r" % calls)
                    if api == "responses":
                        self.assertEqual(len(calls), 1)
                        self.assertEqual(status, ["incomplete"])      # the item says it is cut; the response itself ends "completed" at a stop
                    elif not stream:
                        self.assertEqual(calls, [])                   # chat and messages, whole: no call at all
                    else:
                        self.assertEqual(len(calls), 1)               # streamed: the argument text ends unterminated
                    self.assertNotIn(finish, ("tool_calls", "tool_use"))

    def test_a_closed_call_stays_a_call(self):
        for api, stream, (finish, calls, status, marks) in self.run_case(SCRIPTS["D"]):
            with self.subTest(api=api, stream=stream):
                self.assertEqual(len(calls), 1)
                self.assertEqual(sorted(json.loads(calls[0])), ["content", "path"])
                self.assertEqual(json.loads(calls[0])["content"], BODY[:150])
                self.assertEqual(marks, {})
                self.assertIn(finish, ("tool_calls", "tool_use", "completed"))
                self.assertTrue(all(s == "completed" for s in status))

    def test_a_call_the_model_closed_itself_is_delivered_as_written(self):
        # t36: the closer was written by the model, the value stops mid-sentence and there is no path: valid JSON with `content` only, no marker.
        # The server cannot tell it from any closed call; a client that checks the required parameters rejects it.
        for api, stream, (finish, calls, status, marks) in self.run_case(SCRIPTS["F"]):
            with self.subTest(api=api, stream=stream):
                self.assertEqual(len(calls), 1)
                self.assertEqual(sorted(json.loads(calls[0])), ["content"])
                self.assertEqual(marks, {})

    def test_the_cut_by_max_tokens_is_the_same_cut(self):
        for api, stream, (finish, calls, status, marks) in self.run_case(SCRIPTS["C"], cap=40):
            with self.subTest(api=api, stream=stream):
                self.assertEqual(finish, {"chat": "length", "messages": "max_tokens", "responses": "incomplete"}[api])
                self.assertEqual(marks.get("strata_unread_tool_markup", {}).get("reason"), "cut_off", marks)
                self.assertFalse([c for c in calls if valid_object(c)])


if __name__ == "__main__":
    unittest.main()
