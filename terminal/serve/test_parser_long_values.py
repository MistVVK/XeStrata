"""serve/test_parser_long_values.py - long streamed tool arguments: same calls and same streamed JSON whatever the pieces,
and parsing time that grows about linearly with the value (#4686).

    python -m pytest serve/test_parser_long_values.py -q

The tool corpus (test_tool_corpus.py) holds short specimens; the parser's handling of a call body over a few KB (a file
written through a tool call, a long array) is checked here: piece sizes 1, 4, 64, 4096 and whole, cuts around 4096,
literal closing tags inside a string value, Unicode, a rewritten non-string parameter, a second call, a call cut at
the end of the output, and the recovery switch on and off.
"""
from __future__ import annotations

import json
import sys
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve.frontend import OutputParser  # noqa: E402

TOOLS = [{"name": "write", "parameters": {"type": "object", "properties": {"path": {"type": "string"},
                                                                         "content": {"type": "string"}}}},
         {"name": "todo", "parameters": {"type": "object", "properties": {"items": {"type": "array"},
                                                                        "n": {"type": "integer"}}}}]


def call(name, **params):
    body = "".join(f"<parameter={k}>\n{v}\n</parameter>\n" for k, v in params.items())
    return f"<tool_call>\n<function={name}>\n{body}</function>\n</tool_call>"


LONG = ("line " + "x" * 70 + "\n") * 120                       # ~9 KB
TAGS = "a file that documents the format: </parameter> then </function> then </tool_call> end\n"
UNI = "naïve café 中文 \U0001F600 " * 200
ITEMS = json.dumps([{"content": "task %d " % i + "y" * 40} for i in range(150)])

SPECIMENS = {      # (a value's own last newline stays: the parser drops only the one call() writes before the close)
    "long string": (call("write", path="a.txt", content=LONG), [("write", {"path": "a.txt", "content": LONG})]),
    "long string with literal tags": (call("write", path="d.md", content=LONG + TAGS + LONG),
                                      [("write", {"path": "d.md", "content": LONG + TAGS + LONG})]),
    "unicode": (call("write", path="u.txt", content=UNI), [("write", {"path": "u.txt", "content": UNI})]),
    "long array": (call("todo", items=ITEMS, n="3"), [("todo", {"items": json.loads(ITEMS), "n": 3})]),
    "two calls": (call("write", path="a.txt", content=LONG) + "\n" + call("todo", items=ITEMS),
                  [("write", {"path": "a.txt", "content": LONG}), ("todo", {"items": json.loads(ITEMS)})]),
    "prose then a call": ("I will write it.\n\n" + call("write", path="p.txt", content=LONG),
                          [("write", {"path": "p.txt", "content": LONG})]),
}


def run(text, pieces, stream_tools, recover, finish="stop"):
    p = OutputParser(thinking=False, tools=TOOLS, stream_tools=stream_tools, recover=recover, declared_only=True)
    evs = []
    for piece in pieces:
        evs += p.feed(piece)
    evs += p.finish(finish)
    return evs


def cut(text, size):
    return [text[i:i + size] for i in range(0, len(text), size)]


class LongValues(unittest.TestCase):
    def check(self, name, text, want, pieces, stream_tools, recover):
        evs = run(text, pieces, stream_tools, recover)
        where = f"{name} [{len(pieces)} pieces, stream_tools={stream_tools}, recover={recover}]"
        self.assertEqual([(e.call.name, e.call.arguments) for e in evs if e.kind == "tool_call"], want, where)
        if stream_tools:
            for e in (x for x in evs if x.kind == "tool_call"):
                streamed = "".join(x.text for x in evs if x.kind == "tool_args" and x.call is not None
                                   and x.call.id == e.call.id)
                self.assertEqual(json.loads(streamed), e.call.arguments, where)
        return evs

    def test_every_specimen_in_every_piece_size(self):
        for name, (text, want) in SPECIMENS.items():
            for recover in (False, True):
                for stream_tools in (False, True):
                    for size in (1, 4, 64, 4096, len(text)):
                        self.check(name, text, want, cut(text, size), stream_tools, recover)

    def test_cuts_around_the_first_4_kb(self):
        for name, (text, want) in SPECIMENS.items():
            for at in list(range(4080, 4120)) + [8191, 8192, 8193]:
                if at < len(text):
                    for recover in (False, True):
                        self.check(name, text, want, [text[:at], text[at:]], True, recover)

    def test_a_call_cut_at_the_end_of_the_output(self):
        text = "<tool_call>\n<function=write>\n<parameter=path>\na.txt\n</parameter>\n<parameter=content>\n" + LONG
        for recover in (False, True):
            ref = None
            for size in (1, 64, len(text)):
                evs = run(text, cut(text, size), True, recover, finish="length")
                got = ([(e.call.name, e.call.arguments) for e in evs if e.kind == "tool_call"],
                       "".join(e.text for e in evs if e.kind == "content"))
                ref = got if ref is None else ref
                self.assertEqual(got, ref, f"size {size} recover={recover}")

    def test_parsing_time_grows_about_linearly(self):
        # #4686: the streamed body was searched and copied again for every token, so time grew with the square of
        # the value.  Doubling the value must not much more than double the time (4x would be quadratic).
        def seconds(n):
            text = call("write", path="a.txt", content="z" * n)
            best = None
            for _ in range(3):
                t0 = time.perf_counter()
                run(text, cut(text, 4), True, True)
                dt = time.perf_counter() - t0
                best = dt if best is None else min(best, dt)
            return best
        small, big = seconds(40000), seconds(160000)
        self.assertLess(big / small, 8.0, f"4x the value took {big / small:.1f}x the time ({small:.3f} s -> {big:.3f} s)")


if __name__ == "__main__":
    unittest.main()
