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


# #2371: the type declared through anyOf / oneOf (what Pydantic and many MCP servers write for Optional[bool], unions)
UNION = {"name": "set_flags", "parameters": {"type": "object", "properties": {
    "any_bool": {"anyOf": [{"type": "boolean"}, {"type": "null"}]},
    "one_bool": {"oneOf": [{"type": "boolean"}, {"type": "null"}], "description": "Optional[bool]"},
    "nested": {"anyOf": [{"oneOf": [{"type": "integer"}, {"type": "boolean"}]}, {"type": "null"}]},
    "text_or_bool": {"anyOf": [{"type": "boolean"}, {"type": "string"}]},
    "by_ref": {"anyOf": [{"$ref": "#/$defs/Flag"}, {"type": "null"}]},
    "count": {"oneOf": [{"type": "integer"}, {"type": "null"}]}}}}
UNION_CALL = ("<function=set_flags>\n<parameter=any_bool>\nTrue\n</parameter>\n<parameter=one_bool>\nfalse\n</parameter>\n"
              "<parameter=nested>\nTrue\n</parameter>\n<parameter=text_or_bool>\nTrue\n</parameter>\n"
              "<parameter=by_ref>\nTrue\n</parameter>\n<parameter=count>\n42\n</parameter>\n</function>")
UNION_WANT = {"any_bool": True, "one_bool": False, "nested": True, "text_or_bool": "True", "by_ref": "True", "count": 42}


class UnionTypes(unittest.TestCase):
    def test_declared_type_reads_the_branches(self):
        from serve.frontend import declared_type
        props = UNION["parameters"]["properties"]
        self.assertEqual(declared_type(props["any_bool"]), ["boolean", "null"])
        self.assertEqual(declared_type(props["one_bool"]), ["boolean", "null"])
        self.assertEqual(declared_type(props["nested"]), ["integer", "boolean", "null"])
        self.assertEqual(declared_type(props["text_or_bool"]), ["boolean", "string"])
        self.assertIsNone(declared_type(props["by_ref"]), "a $ref branch: unknown, read as with no schema")
        self.assertEqual(declared_type({"type": "boolean", "anyOf": [{"type": "string"}]}), "boolean", "an explicit type wins")
        self.assertEqual(declared_type({"anyOf": [{"type": "boolean"}]}), "boolean", "one branch: that type")
        self.assertIsNone(declared_type({"anyOf": []}))
        self.assertIsNone(declared_type("boolean"))
        deep = {"type": "boolean"}
        for _ in range(20):
            deep = {"anyOf": [deep]}
        self.assertIsNone(declared_type(deep), "a schema nested past the bound is read as with no schema")

    def test_a_whole_call_reads_union_booleans(self):
        self.assertEqual(parse_tool_call(UNION_CALL, UNION).arguments, UNION_WANT)

    def test_the_streamed_parser_reads_union_booleans(self):
        parser = OutputParser(thinking=False, tools=[UNION], stream_tools=True)
        events = parser.feed("<tool_call>\n" + UNION_CALL + "\n</tool_call>") + parser.finish()
        calls = [e for e in events if e.kind == "tool_call"]
        self.assertEqual(len(calls), 1, events)
        self.assertEqual(calls[0].call.arguments, UNION_WANT)

    def test_the_streamed_parser_in_pieces(self):
        parser = OutputParser(thinking=False, tools=[UNION], stream_tools=True)
        events = []
        text = "<tool_call>\n" + UNION_CALL + "\n</tool_call>"
        for i in range(0, len(text), 7):
            events += parser.feed(text[i:i + 7])
        events += parser.finish()
        calls = [e for e in events if e.kind == "tool_call"]
        self.assertEqual([c.call.arguments for c in calls], [UNION_WANT])


if __name__ == "__main__":
    unittest.main()
