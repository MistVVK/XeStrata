"""#4923: one unpaired backtick in the reasoning hid every tool call after it in the same paragraph (the inline-code count was carried to the
paragraph's end), so the turn ended with thinking only and no call.  Inline code is a one-line matter: a quoted `<tool_call>` is in a fence
or on a line with its own backticks, and those stay text.

    python -m unittest serve.test_parser_stray_backtick_eng
"""
import unittest

from serve.frontend import OutputParser

TOOLS = [{"name": "run_commands", "parameters": {"type": "object", "properties": {
    "commands": {"type": "array", "items": {"type": "string"}}}, "required": ["commands"]}}]
CALL = ('<tool_call>\n<function=run_commands>\n<parameter=commands>\n["ls"]\n</parameter>\n'
        '</function>\n</tool_call>')


def delivered(text, chunk=0, recover=False, thinking=True):
    p = OutputParser(thinking, list(TOOLS), stream_tools=True, recover=recover)
    ev = []
    if chunk:
        for i in range(0, len(text), chunk):
            ev += p.feed(text[i:i + chunk])
    else:
        ev = p.feed(text)
    ev += p.finish("stop")
    return [e for e in ev if e.kind == "tool_call"]


class StrayBacktick(unittest.TestCase):
    def each(self, text, **kw):
        for recover in (False, True):
            for chunk in (0, 1, 7):
                yield delivered(text, chunk, recover, **kw)

    def test_paired_backticks_before_a_call(self):
        for calls in self.each("Let me use `run_commands` to check.\n" + CALL + "\n"):
            self.assertEqual(len(calls), 1)

    def test_one_stray_backtick_on_an_earlier_line(self):
        for calls in self.each("Let me use `run_commands to check.\n" + CALL + "\n"):
            self.assertEqual(len(calls), 1)

    def test_a_stray_backtick_two_lines_up(self):
        for calls in self.each("I should check the ls` output first.\nthen run it.\n" + CALL + "\n"):
            self.assertEqual(len(calls), 1)

    def test_an_opener_inside_inline_code_on_its_own_line_is_still_text(self):
        for calls in self.each("The format is `<tool_call>` and nothing else.\nDone thinking.\n</think>\n\nok"):
            self.assertEqual(calls, [])

    def test_a_call_inside_a_fence_is_still_text(self):
        for calls in self.each("Example:\n```xml\n" + CALL + "\n```\n"):
            self.assertEqual(calls, [])


if __name__ == "__main__":
    unittest.main()
