"""serve/test_tool_name_mcp_member.py - #5281: a namespaced MCP tool called by its last segment is that tool.

Seen live (Qwen3.8-Flash-Next Q2_0 under DeepSeek Harness, stream): the request declared `mcp__assets__get_asset_info` and
`mcp__assets__get_maintenance_plan`; the model wrote `<function=get_asset_info>` and `<function=get_maintenance_plan>`.  Neither name was
declared, so each block went back in `content` as XML, the harness got `stop` and the lookups never ran.  The member after the last `__`
is the tool when exactly one declared tool ends that way; a declared name always wins, and two tools with the same ending are not guessed.

    python -m unittest serve.test_tool_name_mcp_member -v
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from serve.frontend import OutputParser  # noqa: E402

def tool(name, prop="id"):
    return {"name": name, "parameters": {"properties": {prop: {"type": "integer"}}}}

TOOLS = [tool("read_skill_resource", "path"), tool("mcp__assets__get_asset_info"), tool("mcp__assets__get_maintenance_plan")]


def call(name, arg="id", value="101"):
    return f"<tool_call>\n<function={name}>\n<parameter={arg}>\n{value}\n</parameter>\n</function>\n</tool_call>"


def run(text, tools=TOOLS, stream_tools=True, step=5, recover=True):
    p = OutputParser(thinking=False, tools=tools, stream_tools=stream_tools, recover=recover, declared_only=True)
    evs = []
    for i in range(0, len(text), step):
        evs += p.feed(text[i:i + step])
    evs += p.finish("stop")
    return evs


def calls(evs):
    return [(e.call.name, e.call.arguments) for e in evs if e.kind == "tool_call"]


class McpMember(unittest.TestCase):
    def test_the_last_segment_is_the_tool(self):
        for stream in (False, True):
            for step in (1, 5, 1000):
                evs = run(call("get_asset_info") + "\n" + call("get_maintenance_plan"), stream_tools=stream, step=step)
                self.assertEqual(calls(evs), [("mcp__assets__get_asset_info", {"id": 101}),
                                              ("mcp__assets__get_maintenance_plan", {"id": 101})], (stream, step))
                self.assertFalse([e for e in evs if e.kind == "content" and e.text.strip()])

    def test_the_full_name_is_unchanged(self):
        self.assertEqual(calls(run(call("mcp__assets__get_asset_info"))), [("mcp__assets__get_asset_info", {"id": 101})])

    def test_two_tools_with_the_same_ending_are_not_guessed(self):
        tools = [tool("mcp__a__search"), tool("mcp__b__search")]
        evs = run(call("search"), tools)
        self.assertEqual(calls(evs), [])
        self.assertIn("<function=search>", "".join(e.text or "" for e in evs if e.kind == "content"))

    def test_a_declared_name_wins(self):
        tools = [tool("get_asset_info"), tool("mcp__assets__get_asset_info")]
        self.assertEqual(calls(run(call("get_asset_info"), tools)), [("get_asset_info", {"id": 101})])

    def test_an_unrelated_name_stays_text(self):
        evs = run(call("get_asset"))
        self.assertEqual(calls(evs), [])

    def test_the_dotted_member_rule_is_unchanged(self):
        tools = [tool("mcp__fs.write_file", "path")]
        self.assertEqual(calls(run(call("write_file", "path", "x"), tools)), [("mcp__fs.write_file", {"path": "x"})])


if __name__ == "__main__":
    unittest.main()
