"""serve/test_tool_call_invoke.py - #5969's parser part (recovery on): a call written in Claude's
`<invoke name="NAME"><parameter name="K">V</parameter></invoke>` dialect inside the `<tool_call>` wrapper is the call
(cases from PR #5969's test_tool_call_envelope.py InvokeDialect, Sergey Zinchenko; the incident reply ended at
`</invoke>` with no `</tool_call>`, and leaked into the answer as raw XML with finish "stop").  A cut call, an
undeclared name, prose and code stay text; with recovery off nothing changes.

    python -m pytest serve/test_tool_call_invoke.py -q
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve import frontend  # noqa: E402

BASH = [{"name": "Bash", "parameters": {"properties": {"command": {"type": "string"},
                                                       "description": {"type": "string"},
                                                       "timeout": {"type": "integer"}},
                                         "required": ["command"]}},
        {"name": "Read", "parameters": {"properties": {"file_path": {"type": "string"}}, "required": ["file_path"]}}]
INVOKE_REPLY = ('Waiting for the bu<tool_call>\n<invoke name="Bash">\n'
                '<parameter name="command">sleep 1; tail -3 "build.log"</parameter>\n'
                '<parameter name="description">peek build progress</parameter>\n</invoke>')


def run(text, width=0, stream=True, recover=True, finish=None, thinking=False):
    p = frontend.OutputParser(thinking=thinking, stream_tools=stream, recover=recover, tools=BASH)
    evs = []
    for k in range(0, len(text), width or len(text) or 1):
        evs += p.feed(text[k:k + (width or len(text))])
    return evs + p.finish(finish)


def calls(evs):
    return [(e.call.name, e.call.arguments) for e in evs if e.kind == "tool_call"]


def text(evs):
    return "".join(e.text or "" for e in evs if e.kind == "content")


class InvokeDialect(unittest.TestCase):
    def test_the_incident_reply_is_a_call(self):
        for stream in (False, True):
            for width in (0, 1, 2, 3, 7, 16):
                with self.subTest(stream=stream, width=width):
                    evs = run(INVOKE_REPLY, width, stream)
                    self.assertEqual(calls(evs), [("Bash", {"command": 'sleep 1; tail -3 "build.log"',
                                                            "description": "peek build progress"})])
                    self.assertEqual(text(evs), "Waiting for the bu")

    def test_an_explicit_closer_and_a_following_call(self):
        raw = ('<tool_call>\n<invoke name="Read">\n<parameter name="file_path">/a</parameter>\n</invoke>\n</tool_call>\n'
               '<tool_call>\n<invoke name="Bash">\n<parameter name="command">ls</parameter>\n<parameter name="timeout">5</parameter>\n'
               '</invoke>\n</tool_call>')
        for width in (0, 1, 5, 13):
            with self.subTest(width=width):
                evs = run(raw, width)
                self.assertEqual(calls(evs), [("Read", {"file_path": "/a"}), ("Bash", {"command": "ls", "timeout": 5})])
                self.assertEqual(text(evs).strip(), "")

    def test_a_value_may_hold_the_dialect_tags(self):
        value = 'echo "</parameter> and </invoke> and <parameter name=x>"'
        raw = f'<tool_call>\n<invoke name="Bash">\n<parameter name="command">{value}</parameter>\n</invoke>\n</tool_call>'
        for width in (0, 1, 6):
            with self.subTest(width=width):
                self.assertEqual(calls(run(raw, width)), [("Bash", {"command": value})])

    def test_an_undeclared_name_stays_text(self):
        raw = '<tool_call>\n<invoke name="TaskCreate">\n<parameter name="subject">x</parameter>\n</invoke>\n</tool_call>'
        for width in (0, 1, 4):
            with self.subTest(width=width):
                evs = run(raw, width)
                self.assertEqual(calls(evs), [])
                self.assertEqual(text(evs), raw)

    def test_prose_and_code_are_not_a_call(self):
        for raw in ('Write <tool_call><invoke name="Bash"> to call it.\n```\n<tool_call>\n<invoke name="Bash">\n'
                    '<parameter name="command">ls</parameter>\n</invoke>\n```',
                    "The <invoke tag is Claude's."):
            with self.subTest(raw=raw[:30]):
                evs = run(raw, 3)
                self.assertEqual(calls(evs), [])
                self.assertEqual(text(evs), raw)

    def test_a_call_cut_by_the_token_limit_stays_text(self):
        raw = '<tool_call>\n<invoke name="Bash">\n<parameter name="command">sleep 1; tail'
        for width in (0, 1, 7):
            evs = run(raw, width, finish="length")
            self.assertEqual((calls(evs), text(evs)), ([], raw))

    def test_the_output_ending_inside_the_opener_loses_nothing(self):
        for raw in ('ok<tool_call>\n<invoke name="Bash">', 'ok<tool_call>\n<invoke name="Bash">\n\n',
                    'ok<tool_call>\n<invoke name="Ba', 'ok<tool_call>\n<inv', 'ok<tool_call>'):
            for width in (0, 1, 4):
                with self.subTest(raw=raw[-14:], width=width):
                    evs = run(raw, width)
                    self.assertEqual((calls(evs), text(evs)), ([], raw))

    def test_a_dialect_call_after_the_reasoning(self):
        evs = run("plan</think>\n\nGo.\n" + INVOKE_REPLY, 5, thinking=True)
        self.assertEqual([c[0] for c in calls(evs)], ["Bash"])
        self.assertEqual([e.text for e in evs if e.kind == "reasoning"], ["plan"])
        self.assertNotIn("invoke", text(evs))

    def test_off_without_recovery_as_before(self):
        evs = run(INVOKE_REPLY, 4, recover=False)
        self.assertEqual(calls(evs), [])


if __name__ == "__main__":
    unittest.main()
