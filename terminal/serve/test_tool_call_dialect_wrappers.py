"""#6022 (recovery on): the spellings a model falls back to for a declared tool's call that leaked as text with finish "stop":
- `<invoke name="T">...</invoke>` alone, or inside `<function_calls>` (Claude's dialect without the template's `<tool_call>`);
- `<function=T>{"k": v}</function>`, a JSON object as the body (it was run with NO arguments).
An undeclared name, prose, code and recovery off stay as they were.

    python -m unittest serve.test_tool_call_dialect_wrappers
"""
import json
import unittest

from serve.frontend import OutputParser

TOOLS = [{"name": "bash", "parameters": {"properties": {"command": {"type": "string"}, "timeout": {"type": "integer"}}}},
         {"name": "ping", "parameters": {"properties": {}}}]
INVOKE = '<invoke name="bash">\n<parameter name="command">ls -la</parameter>\n</invoke>'


def run(text, recover=True, piece=0, stream=False, tools=TOOLS):
    p = OutputParser(thinking=False, tools=tools, recover=recover, stream_tools=stream, declared_only=True)
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


def answer(evs):
    return "".join(e.text for e in evs if e.kind == "content")


class Dialect(unittest.TestCase):
    def each(self, text, **kw):
        for piece in (0, 1, 7):
            for stream in (False, True):
                yield run(text, piece=piece, stream=stream, **kw)

    def test_function_calls_wrapper(self):
        for evs in self.each("I will list it.\n<function_calls>\n" + INVOKE + "\n</function_calls>"):
            self.assertEqual(calls(evs), [("bash", {"command": "ls -la"})])
            self.assertEqual(answer(evs), "I will list it.")

    def test_function_calls_wrapper_cut_before_its_closer(self):
        for evs in self.each("<function_calls>\n" + INVOKE):
            self.assertEqual(calls(evs), [("bash", {"command": "ls -la"})])
            self.assertEqual(answer(evs), "")

    def test_a_bare_invoke(self):
        for evs in self.each("I will list it.\n\n" + INVOKE):
            self.assertEqual(calls(evs), [("bash", {"command": "ls -la"})])
            self.assertEqual(answer(evs), "I will list it.")

    def test_the_wrapped_form_is_unchanged(self):
        for evs in self.each("<tool_call>\n" + INVOKE + "\n</tool_call>"):
            self.assertEqual(calls(evs), [("bash", {"command": "ls -la"})])
            self.assertEqual(answer(evs).strip(), "")

    def test_undeclared_prose_and_code_stay_text(self):
        for t in ('<function_calls>\n<invoke name="TaskCreate">\n<parameter name="subject">x</parameter>\n</invoke>\n</function_calls>',
                  "Write <invoke name=\"bash\"> to call it.",
                  "Use `<invoke name=\"bash\">\n<parameter name=\"command\">ls</parameter>\n</invoke>` like this.",
                  "```\n" + INVOKE + "\n```"):
            for evs in self.each(t):
                self.assertEqual(calls(evs), [], t)

    def test_recovery_off_is_unchanged(self):
        for t in ("<function_calls>\n" + INVOKE + "\n</function_calls>", INVOKE):
            evs = run(t, recover=False)
            self.assertEqual(calls(evs), [])
            self.assertIn("invoke", answer(evs))


class JsonBody(unittest.TestCase):
    BODY = '<function=bash>{"command": "ls", "timeout": 5}</function>'

    def test_the_object_is_the_arguments(self):
        for wrapped in (self.BODY, "<tool_call>\n" + self.BODY + "\n</tool_call>", "<tool_call>\n<function=bash>\n{\"command\": \"ls\", \"timeout\": 5}\n</function>\n</tool_call>"):
            for piece in (0, 1, 7):
                evs = run(wrapped, piece=piece, stream=True)
                self.assertEqual(calls(evs), [("bash", {"command": "ls", "timeout": 5})], (wrapped, piece))
                streamed = "".join(e.text for e in evs if e.kind == "tool_args")
                self.assertEqual(json.loads(streamed), {"command": "ls", "timeout": 5}, (wrapped, piece))

    def test_a_tool_without_parameters_is_still_a_call(self):
        evs = run("<tool_call>\n<function=ping>\n</function>\n</tool_call>", stream=True)
        self.assertEqual(calls(evs), [("ping", {})])

    def test_recovery_off_is_unchanged(self):
        evs = run("<tool_call>\n" + self.BODY + "\n</tool_call>", recover=False)
        self.assertEqual(calls(evs), [("bash", {})])


if __name__ == "__main__":
    unittest.main()
