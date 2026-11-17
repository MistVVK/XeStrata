"""serve/test_parser_compaction.py - a streamed tool call whose string value is longer than about 4 KB keeps all its parameters.

    python -m pytest serve/test_parser_compaction.py -q

The streamed scan moves consumed call text out of its buffer once the scan position passes 4096 (OutputParser._compact_scan_body, #4686).
An open string value is walked from its start each time a closer may have arrived, so the compaction must leave that text where it is
and rebase every index it shifts; with the value's start left behind, the value's closing tag was not seen and the parameters after the
value were lost (sometimes the whole call).  Here: values of 4,000 / 4,096 / 4,097 / 8,200 / 60,000 characters first, in the middle and
last in a call, a call with two long values, streamed in pieces of 1 / 7 / 64 / 4,096, with the recovery switch on and off.
"""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve.frontend import OutputParser  # noqa: E402

TOOLS = [{"name": "write_file", "parameters": {"type": "object", "properties": {
    "content": {"type": "string"}, "path": {"type": "string"}, "mode": {"type": "string"}, "append": {"type": "boolean"}},
    "required": ["path", "content"]}}]
SIZES = (4000, 4096, 4097, 8200, 60000)
PIECES = (1, 7, 64, 4096)


def value(n: int, tag: str = "a") -> str:
    """n characters of plain lines (no trailing newline: the call's own newline before the closing tag is dropped by the parser)."""
    line = "    %s_value = compute(ctx, 12345)    # padding to get a line of 60 characters....\n" % tag
    return ((line * (n // len(line) + 1))[:n - 1] + "z")


def call(params: list[tuple[str, str]]) -> str:
    body = "".join(f"<parameter={k}>\n{v}\n</parameter>\n" for k, v in params)
    return f"<tool_call>\n<function=write_file>\n{body}</function>\n</tool_call>"


def stream(text: str, step: int, recover: bool):
    p = OutputParser(thinking=False, tools=TOOLS, stream_tools=True, recover=recover)
    evs = []
    for k in range(0, len(text), step):
        evs += p.feed(text[k:k + step])
    evs += p.finish("stop")
    return evs


class CompactionKeepsEveryParameter(unittest.TestCase):
    def check(self, params, step, recover):
        want = dict(params)
        evs = stream(call(params), step, recover)
        where = f"{[(k, len(v)) for k, v in params]} pieces of {step}, recover={recover}"
        calls = [e for e in evs if e.kind == "tool_call"]
        self.assertEqual(len(calls), 1, where)
        self.assertEqual(calls[0].call.arguments, want, where)
        streamed = "".join(x.text for x in evs if x.kind == "tool_args" and x.call is not None)
        self.assertEqual(json.loads(streamed), want, where + " (the streamed arguments)")
        self.assertEqual("".join(x.text for x in evs if x.kind == "content").strip(), "", where + " (nothing leaks into the answer)")

    def test_a_long_value_first_then_more_parameters(self):
        for n in SIZES:
            for step in PIECES:
                for recover in (True, False):
                    self.check([("content", value(n)), ("path", "x/y.py"), ("mode", "w")], step, recover)

    def test_a_long_value_in_the_middle(self):
        for n in SIZES:
            for step in PIECES:
                for recover in (True, False):
                    self.check([("path", "x/y.py"), ("content", value(n)), ("mode", "w")], step, recover)

    def test_a_long_value_last(self):
        for n in SIZES:
            for step in PIECES:
                for recover in (True, False):
                    self.check([("path", "x/y.py"), ("mode", "w"), ("content", value(n))], step, recover)

    def test_two_long_values(self):
        for n in (4097, 8200):
            for step in PIECES:
                for recover in (True, False):
                    self.check([("content", value(n, "a")), ("path", value(n, "b")), ("mode", "w")], step, recover)

    def test_a_non_string_parameter_after_a_long_value(self):
        for step in PIECES:
            for recover in (True, False):
                evs = stream(call([("content", value(8200)), ("append", "true"), ("path", "p")]), step, recover)
                calls = [e for e in evs if e.kind == "tool_call"]
                self.assertEqual([c.call.arguments for c in calls], [{"content": value(8200), "append": True, "path": "p"}], (step, recover))


if __name__ == "__main__":
    unittest.main()
