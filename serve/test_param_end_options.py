# SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
"""#4701 decision data: a file whose text lists </parameter>, </function>, </tool_call> on lines of their own, written
through a real call.

Option C: the closing sequence ends the value unless prose follows it AND a later closing sequence follows that prose,
with no new <tool_call> opener in between.  The cases below are what it must do; the first one fails today
(expectedFailure) and decides nothing by itself: the other two are what Option A (any prose after the sequence keeps
the value open) would break.

    python -m unittest serve.test_param_end_options
"""
import unittest

from serve.frontend import OutputParser


def tool(name, **props):
    return {"name": name, "parameters": {"type": "object", "properties": props}}


S = {"type": "string"}
TOOLS = [tool("read", path=S), tool("write_file", path=S, content=S)]
DOC = "# Tool calls\n\nThe closing tags, in order:\n\n</parameter>\n</function>\n</tool_call>\n\nNothing may follow them."
READ = "<tool_call>\n<function=read>\n<parameter=path>\na.txt\n</parameter>\n</function>\n</tool_call>"
WRITE = ("<tool_call>\n<function=write_file>\n<parameter=path>\nd.md\n</parameter>\n<parameter=content>\n" + DOC
         + "\n</parameter>\n</function>\n</tool_call>")


def parse(text):
    p = OutputParser(thinking=False, tools=TOOLS, stream_tools=False, recover=False, declared_only=True)
    evs = p.feed(text) + p.finish("stop")
    return [(e.call.name, e.call.arguments) for e in evs if e.kind == "tool_call"], \
        "".join(e.text for e in evs if e.kind == "content")


class ParamEndOptions(unittest.TestCase):
    @unittest.expectedFailure
    def test_a_file_that_lists_the_closing_tags_keeps_all_its_text(self):
        calls, answer = parse(WRITE)
        self.assertEqual(calls, [("write_file", {"path": "d.md", "content": DOC})])
        self.assertEqual(answer, "")

    def test_a_call_followed_by_a_sentence_is_still_a_call(self):
        calls, answer = parse(READ + "\n\nI will read that file now.")
        self.assertEqual(calls, [("read", {"path": "a.txt"})])
        self.assertIn("I will read", answer)

    def test_two_calls_with_a_sentence_between_are_two_calls(self):
        calls, _ = parse(READ + "\nOkay.\n" + READ)
        self.assertEqual(len(calls), 2)


if __name__ == "__main__":
    unittest.main()
