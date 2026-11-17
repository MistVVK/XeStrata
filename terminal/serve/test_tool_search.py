"""#4663: Anthropic's tool search on /v1/messages - `defer_loading` tools and `tool_reference` blocks.

Claude Code (ENABLE_TOOL_SEARCH, the default behind a gateway) declares ToolSearch and a `defer_loading`
DeferredToolPlaceholder, and names its MCP tools only in the conversation. ToolSearch's result is a tool_result of
tool_reference blocks; from then on the found tool is in `tools` with `defer_loading: true`. Claude Code runs a call
for a held-back tool even before it was loaded. Before this, a call for an undeclared name came back as text with
end_turn and the session stopped; the reference was dropped; and every deferred tool was in the prompt's tool list,
which changed the prompt's start (and so missed the prompt cache) each time a tool was loaded.

    python -m unittest serve.test_tool_search -v
"""
import json
import sys
import unittest
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from serve.frontend import ChatTemplate, OutputParser, anthropic_to_messages, anthropic_tools  # noqa: E402
from serve.server import ByteTokenizer, MockEngine, Service, serve  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
MCP = "mcp__plugin_ctx__ctx_search"
SEARCH = {"name": "ToolSearch", "description": "Load deferred tools.",
          "input_schema": {"type": "object", "properties": {"query": {"type": "string"}}}}
PLACEHOLDER = {"name": "DeferredToolPlaceholder", "input_schema": {"type": "object"}, "defer_loading": True}
LOADED = {"name": MCP, "description": "Search the index.", "defer_loading": True,
          "input_schema": {"type": "object", "properties": {"queries": {"type": "array", "items": {"type": "string"}},
                                                            "source": {"type": "string"}}}}
CALL = f"<tool_call>\n<function={MCP}>\n<parameter=queries>\n[\"a\", \"b\"]\n</parameter>\n</function>\n</tool_call>"
REFERENCE = [{"role": "user", "content": "find it"},
             {"role": "assistant", "content": [{"type": "tool_use", "id": "t1", "name": "ToolSearch",
                                                "input": {"query": "select:" + MCP}}]},
             {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1",
                                           "content": [{"type": "tool_reference", "tool_name": MCP}]}]}]


def events(text, tools, deferred, thinking=False):
    parser = OutputParser(thinking=thinking, tools=tools, stream_tools=True, declared_only=True, deferred=deferred)
    return parser.feed(text) + parser.finish("stop")


class Conversion(unittest.TestCase):
    def test_deferred_tools_stay_out_of_the_tool_list(self):
        _, tools, _ = anthropic_to_messages({"tools": [SEARCH, PLACEHOLDER, LOADED], "messages": REFERENCE})
        self.assertEqual([t["name"] for t in tools], ["ToolSearch"])

    def test_a_reference_loads_the_definition_where_it_is(self):
        messages, _, _ = anthropic_to_messages({"tools": [SEARCH, LOADED], "messages": REFERENCE})
        result = messages[-1]
        self.assertEqual(result["role"], "tool")
        self.assertEqual(json.loads(result["content"].removeprefix("<tools>\n").removesuffix("\n</tools>")),
                         {"name": MCP, "description": "Search the index.", "parameters": LOADED["input_schema"]})

    def test_a_reference_to_a_listed_tool_adds_nothing(self):
        req = {"tools": [SEARCH, {**LOADED, "defer_loading": False}], "messages": REFERENCE}
        self.assertEqual(anthropic_to_messages(req)[0][-1]["content"], "")

    def test_a_reference_to_an_undeclared_tool_is_left_out(self):
        # an old reference (its MCP server gone) must not fail the rest of the session (test_frontend pins the same)
        self.assertEqual(anthropic_to_messages({"tools": [SEARCH], "messages": REFERENCE})[0][-1]["content"], "")

    def test_the_parser_gets_every_declared_tool(self):
        tools, deferred = anthropic_tools({"tools": [SEARCH, PLACEHOLDER, LOADED]})
        self.assertEqual([t["name"] for t in tools], ["ToolSearch", "DeferredToolPlaceholder", MCP])
        self.assertTrue(deferred)
        self.assertFalse(anthropic_tools({"tools": [SEARCH]})[1])
        self.assertEqual(anthropic_tools({}), (None, False))


class Calls(unittest.TestCase):
    def call(self, evs):
        return [e.call for e in evs if e.kind == "tool_call"]

    def test_an_undeclared_call_is_a_call_with_tool_search(self):
        calls = self.call(events(CALL, [{"name": "ToolSearch"}], deferred=True))
        self.assertEqual([(c.name, c.arguments) for c in calls], [(MCP, {"queries": ["a", "b"]})])

    def test_without_tool_search_it_stays_text(self):
        evs = events(CALL, [{"name": "ToolSearch"}], deferred=False)
        self.assertEqual(self.call(evs), [])
        self.assertIn("<tool_call>", "".join(e.text for e in evs if e.kind == "content"))

    def test_a_loaded_deferred_tool_keeps_its_schema(self):
        # without its schema a string that reads as a number would arrive as one, and the client refuses it
        tools, deferred = anthropic_tools({"tools": [SEARCH, LOADED]})
        call = self.call(events(CALL.replace("queries>\n[\"a\", \"b\"]", "source>\n42"), tools, deferred))[0]
        self.assertEqual(call.arguments, {"source": "42"})

    def test_a_call_in_the_reasoning_is_rescued_too(self):
        calls = self.call(events("Searching.\n" + CALL, [{"name": "ToolSearch"}], deferred=True, thinking=True))
        self.assertEqual([c.name for c in calls], [MCP])


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

    def test_claude_code_tool_search(self):
        b = self.post({"model": "m", "max_tokens": 512, "tools": [SEARCH, PLACEHOLDER],
                       "messages": [{"role": "user", "content": "search"}]})
        self.assertEqual(b["stop_reason"], "tool_use")
        use = [c for c in b["content"] if c["type"] == "tool_use"]
        self.assertEqual([(u["name"], u["input"]) for u in use], [(MCP, {"queries": ["a", "b"]})])
        self.assertNotIn("DeferredToolPlaceholder", self.tok.decode(self.engine.last_prompt))

    def test_without_tool_search_the_rule_holds(self):
        b = self.post({"model": "m", "max_tokens": 512, "tools": [SEARCH],
                       "messages": [{"role": "user", "content": "search"}]})
        self.assertEqual(b["stop_reason"], "end_turn")
        self.assertEqual(b.get("strata_rejected_tool_calls"), [{"name": MCP, "reason": "undeclared"}])

    def test_a_tool_the_conversation_called_is_a_call_without_tool_search(self):
        # #4663 on /v1/messages: the history's tool_use blocks count as declared, as on the chat route
        history = [{"role": "user", "content": "search"},
                   {"role": "assistant", "content": [{"type": "tool_use", "id": "t0", "name": MCP, "input": {"queries": ["x"]}}]},
                   {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t0", "content": "ok"}]}]
        b = self.post({"model": "m", "max_tokens": 512, "tools": [SEARCH], "messages": history})
        self.assertEqual(b["stop_reason"], "tool_use")
        self.assertEqual([u["name"] for u in b["content"] if u["type"] == "tool_use"], [MCP])

    def test_the_loaded_tool_is_in_the_tool_result_not_the_tool_list(self):
        self.post({"model": "m", "max_tokens": 512, "tools": [SEARCH, PLACEHOLDER, LOADED], "messages": REFERENCE})
        prompt = self.tok.decode(self.engine.last_prompt)
        self.assertEqual(prompt.count(f'"name": "{MCP}"'), 1)
        self.assertGreater(prompt.index(f'"name": "{MCP}"'), prompt.index("<tool_response>"))


if __name__ == "__main__":
    unittest.main()
