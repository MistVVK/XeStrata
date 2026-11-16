"""#5979: a declared boolean parameter came back as the string "True" / "False" (the model's Python spelling), not as true / false."""

import unittest

from serve.frontend import OutputParser, coerce_value, parse_tool_call

TOOL = {"name": "write_note", "parameters": {"type": "object", "properties": {
    "formatted": {"type": "boolean"}, "title": {"type": "string"}, "pinned": {"type": ["boolean", "null"]},
    "label": {"type": ["boolean", "string"]}}}}


class BooleanArguments(unittest.TestCase):
    def test_the_python_spelling_of_a_declared_boolean_is_the_boolean(self):
        for text, want in (("True", True), ("False", False), ("true", True), ("false", False), (" TRUE ", True)):
            self.assertIs(coerce_value(text, "boolean"), want, text)
        self.assertIs(coerce_value("False", ["boolean", "null"]), False)

    def test_strings_and_other_words_are_left_as_they_were(self):
        self.assertEqual(coerce_value("True", "string"), "True")
        self.assertEqual(coerce_value("True", ["boolean", "string"]), "True", "a string is allowed: the text stays text")
        self.assertEqual(coerce_value("yes", "boolean"), "yes")
        self.assertEqual(coerce_value("True", None), "True", "no schema: as before")

    def test_a_whole_call_and_the_streamed_parser_give_booleans(self):
        body = "<function=write_note>\n<parameter=formatted>\nTrue\n</parameter>\n<parameter=title>\nTrue\n</parameter>\n</function>"
        self.assertEqual(parse_tool_call(body, TOOL).arguments, {"formatted": True, "title": "True"})
        parser = OutputParser(thinking=False, tools=[TOOL], stream_tools=True)
        events = parser.feed("<tool_call>\n" + body + "\n</tool_call>") + parser.finish()
        calls = [e for e in events if e.kind == "tool_call"]
        self.assertEqual(len(calls), 1, events)
        self.assertEqual(calls[0].call.arguments["formatted"], True)
        self.assertEqual(calls[0].call.arguments["title"], "True")


if __name__ == "__main__":
    unittest.main()
