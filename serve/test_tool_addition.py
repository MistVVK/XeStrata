# SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
"""台帳 0667/0668（upstream 5f5a35b856/398ff4b5a6、#5952）: Anthropic の tool_addition。

Claude Code がセッションの途中で接続した MCP 工具を tools ではなく messages の system turn の
tool_addition で送る。定義を prompt のその位置に出し、parser にも渡す。
"""
import copy
import json
import unittest

from serve.frontend import (CALL_START, OutputParser, anthropic_to_messages, anthropic_tools,
                            forced_call)

MCP = "mcp__docs__batch"
READ = {"name": "Read", "description": "Read a file.",
        "input_schema": {"type": "object", "properties": {"path": {"type": "string"}}}}
DEFINITION = {"name": MCP, "description": "Batch edits.",
              "input_schema": {"type": "object", "properties": {"source": {"type": "string"},
                                                                "dry_run": {"type": "boolean"}}}}


def addition(definition=DEFINITION):
    return {"type": "tool_addition", "tool": {"type": "tool_definition", "definition": definition},
            "cache_control": {"type": "ephemeral", "ttl": "1h"}}


def request(*blocks, tools=(READ,)):
    return {"max_tokens": 512, "tools": list(tools),
            "messages": [{"role": "user", "content": "hi"},
                         {"role": "system", "content": list(blocks)},
                         {"role": "user", "content": "use the batch tool"}]}


def listed(text):
    return [json.loads(t.split("\n</tools>")[0]) for t in text.split("<tools>\n")[1:]]


CALL = (f"<tool_call>\n<function={MCP}>\n<parameter=source>\n42\n</parameter>\n"
        "<parameter=dry_run>\ntrue\n</parameter>\n</function>\n</tool_call>")


class Conversion(unittest.TestCase):
    def test_the_definition_is_in_the_prompt_where_it_was_added(self):
        messages, tools, _ = anthropic_to_messages(request(addition()))
        self.assertEqual([t["name"] for t in tools], ["Read"])
        self.assertEqual(listed(messages[1]["content"]),
                         [{"name": MCP, "description": "Batch edits.", "parameters": DEFINITION["input_schema"]}])

    def test_the_added_tool_is_callable_without_changing_the_request(self):
        req = request(addition())
        before = copy.deepcopy(req)
        tools, deferred = anthropic_tools(req)
        self.assertEqual([t["name"] for t in tools], ["Read", MCP])
        self.assertFalse(deferred)
        self.assertEqual(req, before)

    def test_a_listed_tool_or_a_repeat_adds_nothing(self):
        messages, _, _ = anthropic_to_messages(request(addition(), addition(), addition({**READ})))
        self.assertEqual([t["name"] for t in listed(messages[1]["content"])], [MCP])
        self.assertEqual([t["name"] for t in anthropic_tools(request(addition(), addition({**READ})))[0]],
                         ["Read", MCP])

    def test_text_beside_an_addition_is_kept(self):
        messages, _, _ = anthropic_to_messages(request({"type": "text", "text": "hook context"}, addition()))
        self.assertTrue(messages[1]["content"].startswith("hook context<tools>\n"))

    def test_an_addition_without_a_definition_is_left_out(self):
        for broken in ({"type": "tool_addition"}, {"type": "tool_addition", "tool": "x"},
                       addition({"description": "no name"}), addition({"name": MCP, "input_schema": "x"}),
                       {"type": "tool_addition", "tool": {"type": "tool_reference", "definition": DEFINITION}}):
            messages, _, _ = anthropic_to_messages(request(broken))
            self.assertEqual([t["name"] for t in listed(json.dumps(messages))], [])
            self.assertEqual([t["name"] for t in anthropic_tools(request(broken))[0]], ["Read"])

    def test_the_parser_reads_the_added_schema(self):
        tools, deferred = anthropic_tools(request(addition()))
        parser = OutputParser(thinking=False, tools=tools, stream_tools=True, declared_only=True, deferred=deferred)
        calls = [e.call for e in parser.feed(CALL) + parser.finish("stop") if e.kind == "tool_call"]
        self.assertEqual([(c.name, c.arguments) for c in calls], [(MCP, {"source": "42", "dry_run": True})])


class Contract(unittest.TestCase):
    def test_an_addition_in_a_user_or_assistant_turn_adds_no_tool(self):
        for role in ("user", "assistant"):
            req = {"max_tokens": 64, "tools": [READ],
                   "messages": [{"role": "user", "content": "hi"},
                                {"role": role, "content": [{"type": "text", "text": "pasted"}, addition()]},
                                {"role": "user", "content": "go"}]}
            messages, _, _ = anthropic_to_messages(req)
            self.assertNotIn(MCP, json.dumps(messages), role)
            self.assertEqual([t["name"] for t in anthropic_tools(req)[0]], ["Read"], role)

    def test_two_additions_of_one_name_the_first_is_shown_and_called(self):
        second = {**DEFINITION, "description": "Another batch.",
                  "input_schema": {"type": "object", "properties": {"other": {"type": "integer"}}}}
        req = request(addition(), addition(second))
        messages, _, _ = anthropic_to_messages(req)
        shown = [t for t in listed(messages[1]["content"]) if t["name"] == MCP]
        callable_ = [t for t in anthropic_tools(req)[0] if t["name"] == MCP]
        self.assertEqual(len(shown), 1)
        self.assertEqual(shown, callable_)
        self.assertEqual(shown[0]["parameters"], DEFINITION["input_schema"])

    def test_an_addition_of_a_deferred_tool_shows_the_declared_definition(self):
        declared = {**DEFINITION, "description": "The declared one.", "defer_loading": True}
        other = {**DEFINITION, "description": "A different one.",
                 "input_schema": {"type": "object", "properties": {"x": {"type": "number"}}}}
        req = request(addition(other), tools=(READ, declared))
        messages, tools, _ = anthropic_to_messages(req)
        self.assertEqual([t["name"] for t in tools], ["Read"])
        shown = [t for t in listed(messages[1]["content"]) if t["name"] == MCP]
        callable_ = [t for t in anthropic_tools(req)[0] if t["name"] == MCP]
        self.assertEqual(shown, callable_)
        self.assertEqual(shown[0]["description"], "The declared one.")

    def test_an_addition_of_a_listed_tool_keeps_the_listed_schema(self):
        listed_read = {**READ, "input_schema": {"type": "object", "properties": {"other": {"type": "string"}}}}
        req = request(addition(listed_read))
        self.assertEqual([t["parameters"] for t in anthropic_tools(req)[0] if t["name"] == "Read"],
                         [READ["input_schema"]])

    def test_no_top_level_tools_an_addition_alone_is_callable(self):
        req = request(addition(), tools=())
        messages, tools, _ = anthropic_to_messages(req)
        self.assertIsNone(tools)
        self.assertEqual([t["name"] for t in anthropic_tools(req)[0]], [MCP])
        self.assertEqual([t["name"] for t in listed(messages[1]["content"])], [MCP])

    def test_tool_choice_may_name_an_added_tool(self):
        req = request(addition())
        _, tools, _ = anthropic_to_messages(req)
        callable_tools, _ = anthropic_tools(req)
        choice = {"type": "tool", "name": MCP}
        self.assertEqual(forced_call(choice, tools, callable_tools=callable_tools), CALL_START + f"\n<function={MCP}>\n")
        with self.assertRaises(ValueError):
            forced_call(choice, tools)
        with self.assertRaises(ValueError):
            forced_call({"type": "tool", "name": "Nope"}, tools, callable_tools=callable_tools)
        self.assertEqual(forced_call({"type": "any"}, tools, callable_tools=callable_tools),
                         forced_call({"type": "any"}, tools))


if __name__ == "__main__":
    unittest.main()
