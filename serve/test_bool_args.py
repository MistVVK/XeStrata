# SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
"""台帳 0597（upstream 1aabfb1ce4、issue #71）と 0735/0843（a54f9dbbb7、5e965ec57d）。

宣言された真偽値の引数がモデルの Python 表記 True/False で届いた時は真偽値に直す。anyOf/oneOf の型も読む。
"""
import unittest

from serve.frontend import OutputParser, parse_tool_call, param_value

TOOL = {"name": "write_note", "parameters": {"type": "object", "properties": {
    "formatted": {"type": "boolean"}, "title": {"type": "string"}, "pinned": {"type": ["boolean", "null"]},
    "label": {"type": ["boolean", "string"]}}}}
UNION = {"name": "set_flags", "parameters": {"type": "object", "properties": {
    "any_bool": {"anyOf": [{"type": "boolean"}, {"type": "null"}]},
    "one_bool": {"oneOf": [{"type": "boolean"}, {"type": "null"}]},
    "nested": {"anyOf": [{"oneOf": [{"type": "integer"}, {"type": "boolean"}]}, {"type": "null"}]},
    "text_or_bool": {"anyOf": [{"type": "boolean"}, {"type": "string"}]},
    "count": {"oneOf": [{"type": "integer"}, {"type": "null"}]}}}}


class BooleanArguments(unittest.TestCase):
    def test_declared_boolean_accepts_the_python_spelling(self):
        for text, want in (("True", True), ("False", False), ("true", True), ("false", False), (" TRUE ", True)):
            self.assertIs(param_value(text, TOOL["parameters"]["properties"]["formatted"]), want, text)
        self.assertIs(param_value("False", TOOL["parameters"]["properties"]["pinned"]), False)

    def test_strings_and_other_words_are_left_as_they_were(self):
        self.assertEqual(param_value("True", TOOL["parameters"]["properties"]["title"]), "True")
        self.assertEqual(param_value("True", TOOL["parameters"]["properties"]["label"]), "True")
        self.assertEqual(param_value("True", TOOL["parameters"]["properties"]["formatted"] | {"enum": ["True"]}), "True")
        self.assertEqual(param_value("yes", TOOL["parameters"]["properties"]["formatted"]), "yes")
        self.assertEqual(param_value("True", None), "True")

    def test_whole_call_and_streamed_parser(self):
        body = "<function=write_note>\n<parameter=formatted>\nTrue\n</parameter>\n<parameter=title>\nTrue\n</parameter>\n</function>"
        self.assertEqual(parse_tool_call(body, TOOL).arguments, {"formatted": True, "title": "True"})
        parser = OutputParser(thinking=False, tools=[TOOL], stream_tools=True)
        events = parser.feed("<tool_call>\n" + body + "\n</tool_call>") + parser.finish()
        calls = [e for e in events if e.kind == "tool_call"]
        self.assertEqual(len(calls), 1, events)
        self.assertEqual(calls[0].call.arguments["formatted"], True)
        self.assertEqual(calls[0].call.arguments["title"], "True")


class UnionTypes(unittest.TestCase):
    def test_types_declared_through_anyof_oneof_are_read(self):
        props = UNION["parameters"]["properties"]
        self.assertIs(param_value("True", props["any_bool"]), True)
        self.assertIs(param_value("false", props["one_bool"]), False)
        self.assertIs(param_value("True", props["nested"]), True)
        self.assertEqual(param_value("True", props["text_or_bool"]), "True")
        self.assertEqual(param_value("42", props["count"]), 42)

    def test_a_whole_call_with_union_types(self):
        body = ("<function=set_flags>\n<parameter=any_bool>\nTrue\n</parameter>\n"
                "<parameter=one_bool>\nfalse\n</parameter>\n<parameter=count>\n42\n</parameter>\n</function>")
        self.assertEqual(parse_tool_call(body, UNION).arguments,
                         {"any_bool": True, "one_bool": False, "count": 42})


if __name__ == "__main__":
    unittest.main()
