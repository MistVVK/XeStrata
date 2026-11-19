"""The reporters' own texts of issues #4071, #4630, #4701, #4751 and #5282, fed to the parser as written (no model, no GPU): each reads as the call it is,
or, for a call that cannot be read, is caught by the one-time rewrite (dead_markup_tail).  Pinned so the fixes they got (4630: v0.2.8; 4751: v0.2.10;
4701: v0.2.11; the markup retry: v0.2.16) stay.

    python -m unittest serve.test_reported_shapes_0217
"""
import unittest

from serve.frontend import OutputParser
from serve.server import dead_markup_tail

S = {"type": "string"}


def tool(name, **props):
    return {"name": name, "parameters": {"type": "object", "properties": props}}


def qwen(name, **params):
    body = "".join(f"<parameter={k}>\n{v}\n</parameter>\n" for k, v in params.items())
    return f"<tool_call>\n<function={name}>\n{body}</function>\n</tool_call>"


def parse(text, tools, piece=0, recover=True, stream=False):
    p = OutputParser(thinking=False, tools=tools, stream_tools=stream, recover=recover, declared_only=True)
    evs = []
    if piece:
        for i in range(0, len(text), piece):
            evs += p.feed(text[i:i + piece])
        evs += p.finish("stop")
    else:
        evs = p.feed(text) + p.finish("stop")
    return [(e.call.name, e.call.arguments) for e in evs if e.kind == "tool_call"], "".join(e.text for e in evs if e.kind == "content")


class ReportedShapes(unittest.TestCase):
    def each(self, text, tools):
        for recover in (False, True):
            for piece in (0, 1, 7):
                for stream in (False, True):
                    yield parse(text, tools, piece, recover, stream)

    def test_4071_the_reported_block_is_a_call(self):
        t = ('<tool_call>\n<function=grep_code>\n<parameter=pattern>\nPOLICIES|POLICY_DEFAULT|function M\\.allowed|unsolicited\n</parameter>\n'
             '<parameter=path>\nH:\\somepath\\Comm.lua\n</parameter>\n</function>\n</tool_call>')
        for calls, answer in self.each(t, [tool("grep_code", pattern=S, path=S)]):
            self.assertEqual(calls, [("grep_code", {"pattern": "POLICIES|POLICY_DEFAULT|function M\\.allowed|unsolicited", "path": "H:\\somepath\\Comm.lua"})])
            self.assertEqual(answer, "")

    def test_4630_a_raw_newline_in_an_array_parameter(self):
        value = '[{"old_str": "a", "new_str": "line one\nline two"}]'
        tools = [tool("edit_file", edits={"type": "array", "items": {"type": "object"}})]
        for calls, _ in self.each(qwen("edit_file", edits=value), tools):
            self.assertEqual(calls, [("edit_file", {"edits": [{"old_str": "a", "new_str": "line one\nline two"}]})])

    def test_4701_a_file_that_lists_the_closing_tags(self):
        doc = ("# Tool calls\n\nWrite the call, then each parameter, then close every tag.\n\nThe closing tags, in order:\n\n"
               "</parameter>\n</function>\n</tool_call>\n\nNothing may follow the last one.")
        tools = [tool("read", path=S), tool("write_file", path=S, content=S)]
        for calls, answer in self.each("I will save the doc.\n\n" + qwen("write_file", path="TOOLS.md", content=doc), tools):
            self.assertEqual(calls, [("write_file", {"path": "TOOLS.md", "content": doc})])
            self.assertEqual(answer, "I will save the doc.")

    def test_4751_a_ptc_name_and_text_before_the_opener(self):
        t = ("        <tool_call>\n          <function=tools.write>\n            <parameter=file_path>/home/a.py</parameter>\n"
             "            <parameter=content> x=1 </parameter>\n          </function>\n        </tool_call>")
        for calls, _ in self.each(t, [tool("tools.write", file_path=S, content=S)]):
            self.assertEqual(calls, [("tools.write", {"file_path": "/home/a.py", "content": " x=1 "})])
        t = "Ich habe die Kernfaeden identifiziert.<tool_call>\n<function=run_command>\n<parameter=command>\nls -la\n</parameter>\n</function>\n</tool_call>"
        for calls, answer in self.each(t, [tool("run_command", command=S)]):
            self.assertEqual(calls, [("run_command", {"command": "ls -la"})])
            self.assertEqual(answer, "Ich habe die Kernfaeden identifiziert.")

    def test_5282_empty_envelopes_and_undeclared_names_are_caught_for_the_rewrite(self):
        for t in ("<tool_call>\n</tool_call>", "<tool_call>\n</tool_call>\n<tool_call>\n</tool_call>", qwen("AskUserQuestion", questions="[1]")):
            for calls, _ in self.each(t, [tool("get_weather", city=S)]):
                self.assertEqual(calls, [])
            self.assertEqual(dead_markup_tail(t), (0, "unreadable"))


if __name__ == "__main__":
    unittest.main()
