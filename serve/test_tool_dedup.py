# SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
"""台帳 0863（upstream 8426249f01、#5493）: 同じ呼び出しで non-string 引数を書き直した時は後勝ち。

引数を空にした・null/None と書いた時の扱いも固定する（null を許す型なら null、それ以外は欠落）。
"""
import json
import unittest

from serve.frontend import OutputParser, parse_tool_call

TOOL = {"name": "fit", "parameters": {"type": "object", "properties": {
    "path": {"type": "string"}, "bounds": {"type": "array"}, "dry_run": {"type": "boolean"},
    "timeout": {"type": "integer"}, "limit": {"anyOf": [{"type": "integer"}, {"type": "null"}]},
    "count": {"type": "integer"}}}}


class RepeatedParameters(unittest.TestCase):
    def body(self, *pairs):
        return ("<function=fit>\n" + "".join(f"<parameter={k}>\n{v}\n</parameter>\n" for k, v in pairs)
                + "</function>")

    def test_a_rewrite_further_down_the_call_uses_the_later_value(self):
        body = self.body(("bounds", "[1, 2]"), ("path", "curve.dat"), ("bounds", "[10, 20]"),
                         ("timeout", "5"), ("timeout", "7"))
        want = {"path": "curve.dat", "bounds": [10, 20], "timeout": 7}
        self.assertEqual(parse_tool_call(body, TOOL).arguments, want)
        for step in (1, 7, 10_000):
            events = self.events(body, step)
            calls = [e.call for e in events if e.kind == "tool_call"]
            self.assertEqual(len(calls), 1, events)
            self.assertEqual(calls[0].arguments, want)
            streamed = "".join(e.text for e in events if e.kind == "tool_args")
            self.assertEqual(streamed.count('"bounds"'), 1, streamed)
            self.assertEqual(streamed.count('"timeout"'), 1, streamed)
            self.assertEqual(json.loads(streamed), want)

    def test_null_empty_and_false(self):
        body = self.body(("limit", "None"), ("dry_run", "false"), ("count", "0"), ("bounds", ""))
        want = {"limit": None, "dry_run": False, "count": 0}
        self.assertEqual(parse_tool_call(body, TOOL).arguments, want)
        for step in (1, 5, 10_000):
            events = self.events(body, step)
            streamed = "".join(e.text for e in events if e.kind == "tool_args")
            self.assertEqual(json.loads(streamed), want, streamed)

    def test_a_string_parameter_keeps_its_repeated_value_as_text(self):
        body = self.body(("path", "one"), ("path", "two"))
        self.assertEqual(parse_tool_call(body, TOOL).arguments, {"path": "one\n</parameter>\n<parameter=path>\ntwo"})

    def test_without_a_schema_none_stays_text(self):
        call = parse_tool_call("<function=other>\n<parameter=x>\nNone\n</parameter>\n</function>", None)
        self.assertEqual(call.arguments, {"x": "None"})

    def events(self, body: str, step: int):
        text = "</think>\n\n<tool_call>\n" + body + "\n</tool_call>"
        parser = OutputParser(thinking=True, tools=[TOOL], stream_tools=True, recover=True)
        events = []
        for i in range(0, len(text), step):
            events += parser.feed(text[i:i + step])
        return events + parser.finish()


if __name__ == "__main__":
    unittest.main()
