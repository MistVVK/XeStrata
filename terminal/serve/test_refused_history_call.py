"""#6120: a call the client REFUSED is not a tool the client has.  `with_history_tools` (#4663) declares every name the conversation's assistant turns
called; a call whose result says the tool does not exist (`AI_NoSuchToolError: Model tried to call unavailable tool 'edit'`) never ran, so its name is
left out and the model's next try at it stays text (as the first one did).  A call the client ran keeps its name declared.

    python -m unittest serve.test_refused_history_call
"""
import unittest

import serve.server as server
from serve.frontend import OutputParser

EDITOR = {"name": "editor", "parameters": {"type": "object", "properties": {"path": {"type": "string"}}}}
CALL_EDIT = "<tool_call>\n<function=edit>\n<parameter=path>\n/a\n</parameter>\n</function>\n</tool_call>"
REFUSAL = ('{"error":"Tool call edit was rejected before execution: AI_NoSuchToolError: Model tried to call unavailable tool \'edit\'. '
           'Available tools: editor, run_commands"}')


def chat(result):
    msgs = [{"role": "user", "content": "go"},
            {"role": "assistant", "content": "", "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "edit", "arguments": "{}"}}]}]
    if result is not None:
        msgs.append({"role": "tool", "tool_call_id": "c1", "content": result})
    return {"messages": msgs}


def run(tools):
    p = OutputParser(thinking=False, tools=tools, stream_tools=True, recover=True, declared_only=True)
    evs = p.feed(CALL_EDIT) + p.finish("stop")
    return [e.call.name for e in evs if e.kind == "tool_call"], "".join(e.text for e in evs if e.kind == "content")


class RefusedHistoryCall(unittest.TestCase):
    def test_a_refused_call_is_not_declared_again(self):
        names, answer = run(server.with_history_tools([EDITOR], chat(REFUSAL)))
        self.assertEqual(names, [])
        self.assertIn("<tool_call>", answer)

    def test_the_refusal_in_other_words_and_in_part_lists(self):
        for result in ("Error: unknown tool edit", "No such tool: edit", [{"type": "text", "text": REFUSAL}]):
            names, _ = run(server.with_history_tools([EDITOR], chat(result)))
            self.assertEqual(names, [], result)

    def test_a_call_the_client_ran_stays_declared(self):
        for result in ("ok, 3 lines changed", '{"error": "file not found: /a"}', ""):
            names, answer = run(server.with_history_tools([EDITOR], chat(result)))
            self.assertEqual(names, ["edit"], result)
            self.assertEqual(answer, "")

    def test_a_name_called_twice_one_of_them_run_stays_declared(self):
        req = chat(REFUSAL)
        req["messages"] += [{"role": "assistant", "content": "", "tool_calls": [{"id": "c2", "type": "function", "function": {"name": "edit", "arguments": "{}"}}]},
                            {"role": "tool", "tool_call_id": "c2", "content": "done"}]
        names, _ = run(server.with_history_tools([EDITOR], req))
        self.assertEqual(names, ["edit"])

    def test_a_call_with_no_result_in_the_request_is_as_before(self):
        names, _ = run(server.with_history_tools([EDITOR], chat(None)))
        self.assertEqual(names, ["edit"])

    def test_anthropic_and_responses_refusals(self):
        anthropic = {"messages": [{"role": "user", "content": "go"},
                                  {"role": "assistant", "content": [{"type": "tool_use", "id": "t1", "name": "edit", "input": {}}]},
                                  {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "is_error": True, "content": REFUSAL}]}]}
        responses = {"input": [{"type": "function_call", "name": "edit", "call_id": "c1", "arguments": "{}"},
                               {"type": "function_call_output", "call_id": "c1", "output": REFUSAL}]}
        for req in (anthropic, responses):
            names, _ = run(server.with_history_tools([EDITOR], req))
            self.assertEqual(names, [])

    def test_declared_names_are_not_doubled(self):
        tools = [EDITOR, {"name": "edit", "parameters": {"type": "object", "properties": {}}}]
        self.assertIs(server.with_history_tools(tools, chat(REFUSAL)), tools)


if __name__ == "__main__":
    unittest.main()
