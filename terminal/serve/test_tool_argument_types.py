"""Tool argument conversion and incremental parser regressions; no engine or model required.

    python -m unittest serve.test_tool_argument_types -v
"""
import json
import unittest

from serve.frontend import OutputParser, parse_tool_call


class ToolArgumentTypes(unittest.TestCase):
    def check_value(self, declared, value, expected):
        prop = declared if isinstance(declared, dict) else ({} if declared is None else {"type": declared})
        schema = {"name": "f", "parameters": {"properties": {"p": prop}}}
        body = f"<function=f><parameter=p>\n{value}\n</parameter></function>"
        direct = parse_tool_call(body, schema).arguments
        self.assertEqual(direct, {"p": expected})
        self.assertIs(type(direct["p"]), type(expected))
        text = "before</think>\n<tool_call>" + body + "</tool_call>\nafter"
        # Whole, character-wise, and every possible two-chunk split exercise tag/value boundaries.
        chunks = [[text], list(text)] + [[text[:i], text[i:]] for i in range(1, len(text))]
        for stream in (False, True):
            for parts in chunks:
                with self.subTest(stream=stream, chunks=len(parts), first=len(parts[0])):
                    parser = OutputParser(tools=[schema], stream_tools=stream)
                    events = []
                    for part in parts:
                        events.extend(parser.feed(part))
                    events.extend(parser.finish())
                    calls = [e.call for e in events if e.kind == "tool_call"]
                    self.assertEqual(len(calls), 1)
                    self.assertEqual(calls[0].arguments, direct)
                    self.assertIs(type(calls[0].arguments["p"]), type(expected))
                    self.assertEqual("".join(e.text for e in events if e.kind == "reasoning"), "before")
                    self.assertEqual("".join(e.text for e in events if e.kind == "content"), "after")
                    if stream:
                        decoded = json.loads("".join(e.text for e in events if e.kind == "tool_args"))
                        self.assertEqual(decoded, direct)
                        self.assertIs(type(decoded["p"]), type(expected))
                        self.assertEqual([e.call.id for e in events if e.kind == "tool_start"], [calls[0].id])

    def test_booleans(self):
        for value, expected in (("False", False), ("false", False), ("FALSE", False), ("True", True),
                                ("TRUE", True), ("true", True), ("  False\t", False), ("maybe", "maybe"),
                                ("0", 0), ('"False"', "False")):
            with self.subTest(value=value):
                self.check_value("boolean", value, expected)

    def test_numbers(self):
        for declared in ("integer", "number"):
            for value, expected in (("10", 10), ("10.0", 10.0), ('"10"', "10"), ("1e2", 100.0),
                                    ("+10", 10 if declared == "integer" else 10.0),
                                    ("010", 10 if declared == "integer" else 10.0),
                                    ("ten", "ten"), ("True", "True"), ("1_000", "1_000"),
                                    ("inf", "inf"), ("nan", "nan"), ("+1e999", "+1e999")):
                with self.subTest(declared=declared, value=value):
                    self.check_value(declared, value, expected)
        self.check_value("number", ".5", 0.5)
        self.check_value("integer", ".5", ".5")

    def test_nullable_and_union_types(self):
        for declared in (["boolean", "null"], ["null", "boolean"]):
            for value, expected in (("None", None), ("NONE", None), ("null", None), ("False", False),
                                    ("maybe", "maybe")):
                with self.subTest(declared=declared, value=value):
                    self.check_value(declared, value, expected)
        self.check_value(["integer", "number"], "+10", 10)
        self.check_value("null", "None", None)

    def test_strings_stay_verbatim(self):
        for declared in ("string", ["string"]):
            for value in ("False", "True", "None", "10", '"10"', '{"a":1}', "  False  ", "", "\nFalse\n",
                          'quotes " and \\ tabs\t雪', "literal </parameter> and </function></tool_call> text"):
                with self.subTest(declared=declared, value=value):
                    self.check_value(declared, value, value)

    def test_untyped_and_structured(self):
        for declared in (None, {}):
            for value, expected in (('{"a":1}', {"a": 1}),
                                    ("[1,false,null]", [1, False, None]), ("+10", "+10"), ("maybe", "maybe")):
                with self.subTest(declared=declared, value=value):
                    self.check_value(declared, value, expected)
        self.check_value("object", '{"a":1}', {"a": 1})
        self.check_value("array", "[true,2]", [True, 2])
        self.check_value("object", "False", "False")
        self.check_value("array", "[True, None]", [True, None])          # upstream lenient_json (#4427)

    def test_schemas_without_top_level_type_preserve_python_literals(self):
        cases = [
            ({"enum": ["None", "Low", "High"]}, "None"),
            ({"enum": ["True", "False"]}, "False"),
            ({"const": "True"}, "True"),
            ({"anyOf": [{"type": "string"}, {"type": "null"}]}, "False"),
            ({"anyOf": [{"type": "string"}, {"type": "null"}]}, "None"),
            ({"oneOf": [{"type": "string"}, {"type": "boolean"}]}, "True"),
            ({"$ref": "#/$defs/Level"}, "None"),
            ({"description": "A level"}, "False"),
            ({"type": None}, "True"),
            ({"type": []}, "True"),
            ({"type": ""}, "True"),
        ]
        for prop, value in cases:
            with self.subTest(prop=prop, value=value):
                self.check_value(prop, value, value)
        # Preserve the existing JSON-first behavior even without a top-level type.
        self.check_value({"enum": [True, False]}, "false", False)
        self.check_value({"anyOf": [{"type": "object"}, {"type": "null"}]}, '{"a":1}', {"a": 1})

    def test_vehicle_call_and_multiple_calls(self):
        schema = {"name": "lockDoors", "parameters": {"properties": {
            "unlock": {"type": "boolean"}, "door": {"type": "array"}, "note": {"type": "string"}}}}
        body = ('<function=lockDoors><parameter=unlock>False</parameter><parameter=door>["driver"]</parameter>'
                '<parameter=note>True</parameter></function>')
        expected = {"unlock": False, "door": ["driver"], "note": "True"}
        self.assertEqual(parse_tool_call(body, schema).arguments, expected)
        text = ("<tool_call>" + body + "</tool_call>") * 2
        for step in (1, 7, len(text)):
            parser = OutputParser(thinking=False, tools=[schema], stream_tools=True)
            events = []
            for i in range(0, len(text), step):
                events.extend(parser.feed(text[i:i + step]))
            events.extend(parser.finish())
            calls = [e.call for e in events if e.kind == "tool_call"]
            self.assertEqual([c.arguments for c in calls], [expected, expected])
            for call in calls:
                streamed = "".join(e.text for e in events if e.kind == "tool_args" and e.call.id == call.id)
                self.assertEqual(json.loads(streamed), expected)


if __name__ == "__main__":
    unittest.main()