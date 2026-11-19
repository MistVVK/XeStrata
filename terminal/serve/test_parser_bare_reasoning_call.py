"""#5804: (1) a declared tool's `<function=NAME>` written without its `<tool_call>` wrapper INSIDE the reasoning was never read (only the answer's was), so
the turn ended with the call as reasoning text and no `tool_calls`; (2) a lone `<parameter=arguments>` whose value is a JSON object reached the client as
`{"arguments": {...}}` (also in the streamed argument deltas) when the tool has no parameter of that name.  Both with tool_call_recovery (recover) on.

    python -m unittest serve.test_parser_bare_reasoning_call
"""
import json
import unittest

from serve.frontend import OutputParser

TOOLS = [{"name": "exec", "parameters": {"properties": {"path": {"type": "string"}, "content": {"type": "string"}}}},
         {"name": "arguments_tool", "parameters": {"properties": {"arguments": {"type": "object"}}}}]
BODY = '<function=exec>\n<parameter=arguments>{"path": "a.py", "content": "x = 1"}</parameter>\n</function>'
PLAIN = '<function=exec>\n<parameter=path>\na.py\n</parameter>\n</function>'


def run(text, thinking, recover=True, piece=0, stream_tools=False):
    p = OutputParser(thinking=thinking, tools=TOOLS, recover=recover, stream_tools=stream_tools)
    evs = []
    if piece:
        for i in range(0, len(text), piece):
            evs += p.feed(text[i:i + piece])
        evs += p.finish("stop")
    else:
        evs = p.feed(text) + p.finish("stop")
    return evs


def calls(evs):
    return [(e.call.name, e.call.arguments) for e in evs if e.kind == "tool_call"]


def text(evs, kind):
    return "".join(e.text for e in evs if e.kind == kind)


class BareCallInReasoning(unittest.TestCase):
    def each(self, t, thinking=True, **kw):
        for piece in (0, 1, 7):
            for stream in (False, True):
                yield run(t, thinking, piece=piece, stream_tools=stream, **kw)

    def test_a_bare_call_in_the_reasoning_is_a_call(self):
        for evs in self.each("Let me write it.\n" + PLAIN + "\n</think>\n\nDone."):
            self.assertEqual(calls(evs), [("exec", {"path": "a.py"})])
            self.assertEqual(text(evs, "content"), "Done.")
            self.assertEqual(text(evs, "reasoning"), "Let me write it.\n")

    def test_the_call_at_the_very_end_of_the_output(self):
        for evs in self.each("Let me write it.\n" + PLAIN):
            self.assertEqual(calls(evs), [("exec", {"path": "a.py"})])

    def test_two_bare_calls(self):
        for evs in self.each("Plan.\n" + PLAIN + "\n" + PLAIN + "\n</think>\n\nok"):
            self.assertEqual(len(calls(evs)), 2)

    def test_a_call_followed_by_more_thinking_stays_reasoning(self):
        t = "Plan.\n" + PLAIN + "\nand then I think some more.\n</think>\n\nok"
        for evs in self.each(t):
            self.assertEqual(calls(evs), [])
            self.assertIn("<function=exec>", text(evs, "reasoning"))

    def test_the_quote_guards_hold(self):
        for t in ("I would write `" + PLAIN + "` like this.\n</think>\n\nok",
                  "Mid-sentence " + PLAIN + " and on.\n</think>\n\nok",
                  "Example:\n```\n" + PLAIN + "\n```\n</think>\n\nok",
                  "Plan.\n<function=undeclared>\n<parameter=path>\na\n</parameter>\n</function>\n</think>\n\nok"):
            for evs in self.each(t):
                self.assertEqual(calls(evs), [], t)

    def test_recovery_off_leaves_it_as_reasoning(self):
        for evs in self.each("Plan.\n" + PLAIN + "\n</think>\n\nok", recover=False):
            self.assertEqual(calls(evs), [])


class LoneArgumentsParameter(unittest.TestCase):
    def test_the_object_is_the_arguments_whole_and_streamed(self):
        for wrapped in (BODY, "<tool_call>\n" + BODY + "\n</tool_call>"):
            for piece in (0, 1, 7):
                evs = run(wrapped, False, piece=piece, stream_tools=True)
                self.assertEqual(calls(evs), [("exec", {"path": "a.py", "content": "x = 1"})])
                streamed = "".join(e.text for e in evs if e.kind == "tool_args")
                self.assertEqual(json.loads(streamed), {"path": "a.py", "content": "x = 1"})

    def test_a_tool_that_declares_arguments_keeps_it(self):
        t = '<tool_call>\n<function=arguments_tool>\n<parameter=arguments>{"k": 1}</parameter>\n</function>\n</tool_call>'
        for piece in (0, 7):
            evs = run(t, False, piece=piece, stream_tools=True)
            self.assertEqual(calls(evs), [("arguments_tool", {"arguments": {"k": 1}})])

    def test_arguments_beside_another_parameter_is_not_unwrapped(self):
        t = ('<tool_call>\n<function=exec>\n<parameter=path>\na.py\n</parameter>\n<parameter=arguments>{"k": 1}</parameter>\n'
             '</function>\n</tool_call>')
        evs = run(t, False)
        self.assertEqual(calls(evs), [("exec", {"path": "a.py", "arguments": {"k": 1}})])


if __name__ == "__main__":
    unittest.main()
