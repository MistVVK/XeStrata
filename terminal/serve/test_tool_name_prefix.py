"""serve/test_tool_name_prefix.py - #5409: `<function=tools actio>` and the bare `<function=tools>` are a declared tool's call.

Seen live (Qwen3.8-Flash-Next IQ3_S through Pi, after code-mode work that spells its tools `tools.actio(...)`): the model wrote
the prefix of #4751 with a space for the dot, or alone.  Neither named a declared tool, so the block came back as answer text and
the turn ended.  The space form is `tools.NAME` again; the bare prefix is the one declared tool that takes the call's parameters
(only with tool_call_recovery).  Anything else stays text as before, and an exactly declared name always wins.

    python -m unittest serve.test_tool_name_prefix -v
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from serve.frontend import OutputParser  # noqa: E402

ACTIO = {"name": "actio", "parameters": {"properties": {"command": {"type": "string"}}, "required": ["command"]}}
READ = {"name": "read", "parameters": {"properties": {"path": {"type": "string"}}, "required": ["path"]}}
BASH = {"name": "bash", "parameters": {"properties": {"command": {"type": "string"}, "timeout": {"type": "integer"}}}}


def call(name, param="command", value="ls"):
    return f"<tool_call>\n<function={name}>\n<parameter={param}>\n{value}\n</parameter>\n</function>\n</tool_call>"


def run(text, tools, stream_tools=False, recover=True, thinking=False, step=3):
    p = OutputParser(thinking=thinking, tools=tools, stream_tools=stream_tools, recover=recover)
    evs = []
    for i in range(0, len(text), step):
        evs += p.feed(text[i:i + step])
    evs += p.finish("stop")
    return evs


def calls(evs):
    return [(e.call.name, e.call.arguments) for e in evs if e.kind == "tool_call"]


def text(evs):
    return "".join(e.text or "" for e in evs if e.kind in ("content", "reasoning"))


class SpaceForm(unittest.TestCase):
    def test_every_prefix_and_mode(self):
        for prefix in ("tools", "functions", "default_api"):
            for stream in (False, True):
                for step in (1, 3, 1000):
                    evs = run(call(f"{prefix} actio"), [ACTIO, READ], stream_tools=stream, step=step)
                    self.assertEqual(calls(evs), [("actio", {"command": "ls"})], (prefix, stream, step))
                    self.assertEqual(text(evs).strip(), "")

    def test_without_recovery_too(self):
        evs = run(call("tools actio"), [ACTIO], recover=False)
        self.assertEqual(calls(evs), [("actio", {"command": "ls"})])

    def test_text_before_the_call_is_kept(self):
        evs = run("Let me look.\n" + call("tools actio"), [ACTIO])
        self.assertEqual(calls(evs), [("actio", {"command": "ls"})])
        self.assertEqual(text(evs).strip(), "Let me look.")

    def test_in_the_reasoning(self):
        evs = run("thinking...\n" + call("tools actio") + "\n</think>\n", [ACTIO], thinking=True)
        self.assertEqual(calls(evs), [("actio", {"command": "ls"})])

    def test_undeclared_name_stays_text(self):
        for nm in ("tools nothing", "tools read now"):
            evs = run(call(nm), [ACTIO, READ])
            self.assertEqual(calls(evs), [], nm)
            self.assertIn(f"<function={nm}>", text(evs))

    def test_cut_off_name_is_flushed_as_text(self):
        t = "<tool_call>\n<function=tools ac"
        evs = run(t, [ACTIO])
        self.assertEqual(calls(evs), [])
        self.assertEqual(text(evs), t)

    def test_two_calls(self):
        evs = run(call("tools actio") + "\n" + call("tools read", "path", "/x"), [ACTIO, READ], stream_tools=True)
        self.assertEqual(calls(evs), [("actio", {"command": "ls"}), ("read", {"path": "/x"})])


class BarePrefix(unittest.TestCase):
    def test_one_tool_fits(self):
        for prefix in ("tools", "functions"):
            for stream in (False, True):
                for step in (1, 3, 1000):
                    evs = run(call(prefix, "path", "/x"), [ACTIO, READ], stream_tools=stream, step=step)
                    self.assertEqual(calls(evs), [("read", {"path": "/x"})], (prefix, stream, step))

    def test_two_tools_fit_stays_text(self):
        both = {"name": "also", "parameters": {"properties": {"path": {"type": "string"}}}}
        evs = run(call("tools", "path", "/x"), [READ, both])
        self.assertEqual(calls(evs), [])
        self.assertIn("<function=tools>", text(evs))

    def test_none_fits_stays_text(self):
        evs = run(call("tools", "nope", "1"), [ACTIO, READ])
        self.assertEqual(calls(evs), [])
        self.assertIn("<function=tools>", text(evs))

    def test_required_parameter_missing(self):
        evs = run(call("tools", "timeout", "5"), [ACTIO, READ])    # `command` is required by actio
        self.assertEqual(calls(evs), [])

    def test_optional_only_call_fits_one(self):
        evs = run(call("tools", "timeout", "5"), [BASH, READ])
        self.assertEqual(calls(evs), [("bash", {"timeout": 5})])

    def test_off_without_recovery(self):
        evs = run(call("tools", "path", "/x"), [READ], recover=False)
        self.assertEqual(calls(evs), [])
        self.assertIn("<function=tools>", text(evs))

    def test_a_declared_tool_named_tools_wins(self):
        mine = {"name": "tools", "parameters": {"properties": {"x": {"type": "string"}}}}
        evs = run(call("tools", "x", "1"), [mine, READ])
        self.assertEqual(calls(evs), [("tools", {"x": "1"})])

    def test_unfinished_bare_call_is_flushed_as_text(self):
        t = "<tool_call>\n<function=tools>\n<parameter=path>\n/x"
        evs = run(t, [READ], stream_tools=True)
        self.assertEqual(calls(evs), [])
        self.assertEqual(text(evs), t)

    def test_no_tools_declared_is_unchanged(self):
        evs = run(call("tools actio"), [])
        self.assertEqual(calls(evs), [])


if __name__ == "__main__":
    unittest.main()
