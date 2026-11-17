"""#4663: a call to a tool that this turn's `tools` no longer lists, but that the conversation already called, is a tool call (not raw text).

    python -m unittest serve.test_history_tools
"""
import unittest

import serve.server as server
from serve.frontend import OutputParser

TODO = {"name": "TodoWrite", "parameters": {"type": "object", "properties": {"todos": {"type": "array"}}}}
CALL = "<tool_call>\n<function=Bash>\n<parameter=command>\nls -la\n</parameter>\n</function>\n</tool_call>"


def history(*names):
    return {"messages": [{"role": "user", "content": "go"},
                         *[{"role": "assistant", "content": "", "tool_calls": [{"id": f"c{i}", "type": "function",
                           "function": {"name": n, "arguments": "{}"}}]} for i, n in enumerate(names)]]}


def run(tools, text=CALL):
    p = OutputParser(thinking=False, tools=tools, stream_tools=True, recover=True, declared_only=True)
    evs = p.feed(text) + p.finish("stop")
    return [e.call.name for e in evs if e.kind == "tool_call"], "".join(e.text for e in evs if e.kind == "content")


class HistoryTools(unittest.TestCase):
    def test_a_tool_the_conversation_called_is_a_tool_call_again(self):
        names, answer = run(server.with_history_tools([TODO], history("Bash")))
        self.assertEqual(names, ["Bash"])
        self.assertEqual(answer, "")

    def test_without_the_history_it_stays_text(self):
        names, answer = run(server.with_history_tools([TODO], history("Read")))
        self.assertEqual(names, [])
        self.assertIn("<tool_call>", answer)

    def test_declared_names_are_not_doubled_and_no_tools_stays_no_tools(self):
        tools = [TODO, {"name": "Bash", "parameters": {"type": "object", "properties": {"command": {"type": "string"}}}}]
        self.assertIs(server.with_history_tools(tools, history("Bash", "TodoWrite")), tools)
        self.assertEqual(server.with_history_tools([], history("Bash")), [])
        self.assertIsNone(server.with_history_tools(None, history("Bash")))

    def test_anthropic_tool_use_blocks_count(self):
        req = {"messages": [{"role": "user", "content": "go"},
                            {"role": "assistant", "content": [{"type": "text", "text": "ok"},
                                                              {"type": "tool_use", "id": "t1", "name": "Bash", "input": {}}]}]}
        names, answer = run(server.with_history_tools([TODO], req))
        self.assertEqual(names, ["Bash"])
        self.assertEqual(answer, "")

    def test_responses_function_call_items_count(self):
        req = {"input": [{"type": "message", "role": "user", "content": "go"},
                         {"type": "function_call", "name": "Bash", "call_id": "c1", "arguments": "{}"}]}
        names, answer = run(server.with_history_tools([TODO], req))
        self.assertEqual(names, ["Bash"])
        self.assertEqual(answer, "")

    def test_responses_namespace_is_part_of_the_name(self):
        req = {"input": [{"type": "function_call", "namespace": "mcp", "name": "Bash", "call_id": "c1", "arguments": "{}"}]}
        self.assertEqual([t["name"] for t in server.with_history_tools([TODO], req)], ["TodoWrite", "mcp.Bash"])

    def test_the_other_routes_without_history_stay_text(self):
        names, answer = run(server.with_history_tools([TODO], {"input": [{"type": "message", "role": "user", "content": "go"}]}))
        self.assertEqual(names, [])
        self.assertIn("<tool_call>", answer)

    def test_odd_requests_are_left_alone(self):
        self.assertEqual(server.with_history_tools([TODO], None), [TODO])
        self.assertEqual(server.with_history_tools([TODO], {"messages": [None, {"role": "assistant", "tool_calls": [None, {}]}]}), [TODO])



class PrefixedNames(unittest.TestCase):
    """#4751: `tools.write` for a declared `write` (a programmatic-tool-calling harness shows the model `tools.write(...)`)."""

    WRITE = {"name": "write", "parameters": {"type": "object", "properties": {"file_path": {"type": "string"}, "content": {"type": "string"}}}}
    TEXT = ("<tool_call>\n<function=tools.write>\n<parameter=file_path>/a/b.py</parameter>\n"
            "<parameter=content>x = 1\n</parameter>\n</function>\n</tool_call>")

    def calls(self, tools, text=None):
        p = OutputParser(thinking=False, tools=tools, stream_tools=True, recover=True, declared_only=True)
        evs = p.feed(text or self.TEXT) + p.finish("stop")
        return [e.call for e in evs if e.kind == "tool_call"], "".join(e.text for e in evs if e.kind == "content")

    def test_the_prefix_is_read_as_the_declared_tool(self):
        calls, answer = self.calls([self.WRITE])
        self.assertEqual([(c.name, c.arguments["file_path"]) for c in calls], [("write", "/a/b.py")])
        self.assertEqual(answer, "")

    def test_functions_prefix_too(self):
        calls, _ = self.calls([self.WRITE], self.TEXT.replace("tools.write", "functions.write"))
        self.assertEqual([c.name for c in calls], ["write"])

    def test_an_undeclared_name_stays_text(self):
        calls, answer = self.calls([self.WRITE], self.TEXT.replace("tools.write", "tools.delete"))
        self.assertEqual(calls, [])
        self.assertIn("<tool_call>", answer)

    def test_a_declared_dotted_name_wins(self):
        dotted = {"name": "tools.write", "parameters": self.WRITE["parameters"]}
        calls, _ = self.calls([self.WRITE, dotted])
        self.assertEqual([c.name for c in calls], ["tools.write"])


if __name__ == "__main__":
    unittest.main()
