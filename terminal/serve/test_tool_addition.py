"""#5952: Anthropic `tool_addition` blocks on /v1/messages.

Claude Code sends some tools (MCP servers that connect after the session started) outside the top-level `tools`, as
`tool_addition` blocks in a `role: "system"` message inside `messages`. Before this, the block was left out of the
prompt (an unread block type) and the tool was not in the parser's tools: the model never saw the tool, and a call
for it would have come back as text with end_turn.

    python -m unittest serve.test_tool_addition -v
"""
import copy
import json
import sys
import unittest
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from serve.frontend import ChatTemplate, OutputParser, anthropic_to_messages, anthropic_tools  # noqa: E402
from serve.server import ByteTokenizer, MockEngine, Service, serve  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
MCP = "mcp__docs__batch"
READ = {"name": "Read", "description": "Read a file.",
        "input_schema": {"type": "object", "properties": {"path": {"type": "string"}}}}
DEFINITION = {"name": MCP, "description": "Batch edits.",
              "input_schema": {"type": "object", "properties": {"source": {"type": "string"},
                                                                "dry_run": {"type": "boolean"}}}}


def addition(definition=DEFINITION):
    # the shape Claude Code 2.1.325 sends; cache_control is on the last block only
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
        self.assertEqual([t["name"] for t in tools], ["Read"])          # the prompt's start does not change
        self.assertEqual([m["role"] for m in messages], ["user", "user", "user"])
        self.assertEqual(listed(messages[1]["content"]),
                         [{"name": MCP, "description": "Batch edits.", "parameters": DEFINITION["input_schema"]}])

    def test_the_added_tool_is_callable(self):
        req = request(addition())
        before = copy.deepcopy(req)
        tools, deferred = anthropic_tools(req)
        self.assertEqual([t["name"] for t in tools], ["Read", MCP])
        self.assertFalse(deferred)
        self.assertEqual(req, before)                                    # the request itself is not changed

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
                       addition({"description": "no name"}), addition({"name": MCP, "input_schema": "x"})):
            messages, _, _ = anthropic_to_messages(request(broken))
            self.assertEqual([m["content"] for m in messages], ["hi", "use the batch tool"])
            self.assertEqual([t["name"] for t in anthropic_tools(request(broken))[0]], ["Read"])

    def test_the_parser_reads_the_added_schema(self):
        # without the schema "42" would arrive as a number and "true" might not be a boolean
        tools, deferred = anthropic_tools(request(addition()))
        parser = OutputParser(thinking=False, tools=tools, stream_tools=True, declared_only=True, deferred=deferred)
        calls = [e.call for e in parser.feed(CALL) + parser.finish("stop") if e.kind == "tool_call"]
        self.assertEqual([(c.name, c.arguments) for c in calls], [(MCP, {"source": "42", "dry_run": True})])


class Contract(unittest.TestCase):
    """The worklist's review of #6005: the role an addition may come in, the outer shape, and one definition per name
    for both the prompt and the parser."""

    def test_an_addition_in_a_user_or_assistant_turn_adds_no_tool(self):
        for role in ("user", "assistant"):
            req = {"max_tokens": 64, "tools": [READ],
                   "messages": [{"role": "user", "content": "hi"},
                                {"role": role, "content": [{"type": "text", "text": "pasted"}, addition()]},
                                {"role": "user", "content": "go"}]}
            messages, _, _ = anthropic_to_messages(req)
            self.assertNotIn(MCP, json.dumps(messages), role)
            self.assertEqual([t["name"] for t in anthropic_tools(req)[0]], ["Read"], role)

    def test_the_outer_shape_is_checked(self):
        bad_outer = {"type": "tool_addition", "tool": {"type": "tool_reference", "definition": DEFINITION}}
        no_outer_type = {"type": "tool_addition", "tool": {"definition": DEFINITION}}
        bad_description = addition({**DEFINITION, "description": ["not", "text"]})
        for broken in (bad_outer, no_outer_type, bad_description):
            messages, _, _ = anthropic_to_messages(request(broken))
            self.assertNotIn(MCP, json.dumps(messages))
            self.assertEqual([t["name"] for t in anthropic_tools(request(broken))[0]], ["Read"])

    def test_two_additions_of_one_name_the_first_is_shown_and_called(self):
        second = {**DEFINITION, "description": "Another batch.",
                  "input_schema": {"type": "object", "properties": {"other": {"type": "integer"}}}}
        req = request(addition(), addition(second))
        messages, _, _ = anthropic_to_messages(req)
        shown = [t for t in listed(messages[1]["content"]) if t["name"] == MCP]
        callable_ = [t for t in anthropic_tools(req)[0] if t["name"] == MCP]
        self.assertEqual(len(shown), 1)
        self.assertEqual(shown, callable_)                                   # what the model reads is what is called
        self.assertEqual(shown[0]["parameters"], DEFINITION["input_schema"])

    def test_an_addition_of_a_deferred_tool_shows_the_declared_definition(self):
        declared = {**DEFINITION, "description": "The declared one.", "defer_loading": True}
        other = {**DEFINITION, "description": "A different one.",
                 "input_schema": {"type": "object", "properties": {"x": {"type": "number"}}}}
        req = request(addition(other), tools=(READ, declared))
        messages, tools, _ = anthropic_to_messages(req)
        self.assertEqual([t["name"] for t in tools], ["Read"])                # still deferred in the tool list
        shown = [t for t in listed(messages[1]["content"]) if t["name"] == MCP]
        callable_ = [t for t in anthropic_tools(req)[0] if t["name"] == MCP]
        self.assertEqual(shown, callable_)
        self.assertEqual(shown[0]["description"], "The declared one.")

    def test_an_addition_of_a_listed_tool_keeps_the_listed_schema(self):
        listed_read = {**READ, "input_schema": {"type": "object", "properties": {"other": {"type": "string"}}}}
        req = request(addition(listed_read))
        messages, _, _ = anthropic_to_messages(req)
        self.assertNotIn("<tools>", messages[1]["content"] if len(messages) > 2 else "")
        read = [t for t in anthropic_tools(req)[0] if t["name"] == "Read"]
        self.assertEqual([t["parameters"] for t in read], [READ["input_schema"]])

    def test_no_top_level_tools_an_addition_alone_is_callable(self):
        req = request(addition(), tools=())
        messages, tools, _ = anthropic_to_messages(req)
        self.assertIsNone(tools)
        self.assertEqual([t["name"] for t in anthropic_tools(req)[0]], [MCP])
        self.assertEqual([t["name"] for t in listed(messages[1]["content"])], [MCP])

    def test_tool_choice_may_name_an_added_tool(self):
        from serve.frontend import CALL_START, forced_call
        req = request(addition())
        _, tools, _ = anthropic_to_messages(req)
        callable_tools, _ = anthropic_tools(req)
        choice = {"type": "tool", "name": MCP}
        self.assertIsNone(forced_call(choice, tools))                         # the prompt's list alone: not there
        self.assertEqual(forced_call(choice, tools, callable_tools), CALL_START + f"\n<function={MCP}>\n")
        self.assertIsNone(forced_call({"type": "tool", "name": "Nope"}, tools, callable_tools))
        self.assertEqual(forced_call({"type": "any"}, tools, callable_tools), forced_call({"type": "any"}, tools))


class Route(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        tok = ByteTokenizer()
        cls.engine = MockEngine(tok, "</think>\n\n" + CALL, max_context=16384)
        cls.svc = Service(cls.engine, tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
        cls.httpd = serve(cls.svc, port=0)
        cls.base = f"http://127.0.0.1:{cls.httpd.server_address[1]}"
        cls.tok = tok

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()

    def post(self, body):
        req = urllib.request.Request(self.base + "/v1/messages", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.loads(r.read())

    def test_a_call_for_an_added_tool_is_a_tool_use(self):
        b = self.post({"model": "m", **request(addition())})
        self.assertEqual(b["stop_reason"], "tool_use")
        use = [c for c in b["content"] if c["type"] == "tool_use"]
        self.assertEqual([(u["name"], u["input"]) for u in use], [(MCP, {"source": "42", "dry_run": True})])

    def test_tool_choice_naming_an_added_tool_forces_that_call(self):
        b = self.post({"model": "m", **request(addition()), "tool_choice": {"type": "tool", "name": MCP}})
        prompt = self.tok.decode(self.engine.last_prompt)
        self.assertTrue(prompt.rstrip().endswith(f"<function={MCP}>"), prompt[-200:])
        self.assertEqual(b["stop_reason"], "tool_use")

    def test_the_model_sees_the_added_tool_once_after_the_tool_list(self):
        self.post({"model": "m", **request(addition(), addition())})
        prompt = self.tok.decode(self.engine.last_prompt)
        self.assertEqual(prompt.count(f'"name": "{MCP}"'), 1)
        self.assertGreater(prompt.index(f'"name": "{MCP}"'), prompt.index('"name": "Read"'))


if __name__ == "__main__":
    unittest.main()
