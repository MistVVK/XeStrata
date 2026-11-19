"""#4923: one unpaired backtick in a sentence must not hide the tool calls that follow it in the same paragraph.

The inline-code bookkeeping counted backticks across the paragraph.  A model that wrote a tool name as inline code and
never closed it ("Let me use `run_commands to check.") left the count odd for the rest of the paragraph, so every later
`<tool_call>` at the start of a line was read as quoted inline code and stayed reasoning text.

An opener with prose after it on its line is now a stray marker and is not carried into the next line.  A backtick at
the end of its line still opens a span that goes on (the shape #804/#1058 keep as text, `Use ` followed by the example
on the next line), and a call written in one line of inline code or in a fence is still text.

    python -m unittest serve.test_parser_stray_backtick
"""
import unittest

from serve.frontend import OutputParser

TOOLS = [{"name": "run_commands", "parameters": {"type": "object", "properties": {
    "commands": {"type": "array", "items": {"type": "string"}}}, "required": ["commands"]}}]

CALL = ('<tool_call>\n<function=run_commands>\n<parameter=commands>\n["ls"]\n</parameter>\n'
        '</function>\n</tool_call>')


def delivered(text, chunk=None, recover=False):
    """Whether the parser delivers a tool call for `text`, whole (`chunk=None`) or streamed in `chunk`-sized pieces."""
    parser = OutputParser(True, list(TOOLS), stream_tools=True, recover=recover)
    events = []
    if chunk:
        for i in range(0, len(text), chunk):
            events += parser.feed(text[i:i + chunk])
    else:
        events = parser.feed(text)
    return "tool_call" in [e.kind for e in events + parser.finish("stop")]


class StrayBacktickInReasoning(unittest.TestCase):
    def test_a_call_at_the_start_of_a_line_after_an_unpaired_backtick_is_a_call(self):
        # The report's three inputs: paired inline code (worked before the fix), one stray backtick, and a stray
        # backtick on an earlier line of the same paragraph (both lost the call).
        heads = {
            "paired inline code": "Let me use `run_commands` to check.\n",
            "stray backtick": "Let me use `run_commands to check.\n",
            "stray backtick on an earlier line": "I should check the ls` output first.\nthen run it.\n",
        }
        for label, head in heads.items():
            text = head + CALL + "\n"
            with self.subTest(label):
                self.assertTrue(delivered(text), "whole")
                for chunk in (1, 7):
                    self.assertTrue(delivered(text, chunk=chunk), f"streamed in {chunk}-character pieces")
                self.assertTrue(delivered(text, recover=True), "tool_call_recovery on")

    def test_a_backtick_at_the_end_of_a_line_still_opens_inline_code(self):
        # #804/#1058, kept: `Use ` and the example on the next line is a quoted call, not an act.
        self.assertFalse(delivered("Use `\n" + CALL + "\n"))

    def test_a_call_quoted_in_one_line_of_inline_code_stays_text(self):
        # #1058: a call the model quotes is an example, not an act.  The opening backtick is on the call's line.
        text = "The format is `" + CALL + "` and nothing else.\n"
        for chunk in (None, 1, 7):
            with self.subTest(chunk=chunk):
                self.assertFalse(delivered(text, chunk=chunk))

    def test_a_call_inside_a_fence_stays_text(self):
        text = "Example:\n```\n" + CALL + "\n```\n"
        for chunk in (None, 1, 7):
            with self.subTest(chunk=chunk):
                self.assertFalse(delivered(text, chunk=chunk))


if __name__ == "__main__":
    unittest.main()
