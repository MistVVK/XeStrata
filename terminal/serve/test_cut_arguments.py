"""serve/test_cut_arguments.py - a call max_tokens cut short, replayed in a client's history (soak 11-17, 0.2.16): its arguments
arrive as text that starts like a JSON object and is not valid JSON.  The prompt must show the model the call it wrote, with
its parameters, not `<parameter=arguments>` + the cut text: after a dozen cut calls in a history the model wrote
{"arguments": "<json>"} calls itself (and every one of them failed in the client).

  * the members that are complete come back, and a string value the text ends inside as far as it goes (cut_arguments);
  * `{` (or nothing readable) renders the call with no parameter blocks;
  * a tool that really declares a parameter "arguments" keeps the old form (#510);
  * valid histories render as they did: valid JSON objects, objects given parsed, prose, lists and numbers give the same
    mappings as the code before this change (OLD below is that code).

    python -m pytest serve/test_cut_arguments.py -q
"""
import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve import frontend  # noqa: E402
from serve.frontend import ChatTemplate, anthropic_to_messages, openai_to_messages, tool_arguments  # noqa: E402
from serve.responses import input_messages  # noqa: E402

TEMPLATE = ChatTemplate(Path(__file__).resolve().parent / "chat_template.jinja")
WRITE = {"type": "function", "function": {"name": "write_file", "description": "w", "parameters": {
    "type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}}}
# a tool that really has a parameter called "arguments"
WRAPPED = {"type": "function", "function": {"name": "invoke", "description": "i", "parameters": {
    "type": "object", "properties": {"arguments": {"type": "string"}}, "required": ["arguments"]}}}
CUT = '{"path":"core/lease_audit.py","content":"\\"\\"\\"core.lease_audit: audit leases.\\n\\nCovers:\\n    a   b.py:1\\nSecond line cut mid-sen'
CUT_TEXT = '"""core.lease_audit: audit leases.\n\nCovers:\n    a   b.py:1\nSecond line cut mid-sen'


def old_tool_arguments(raw) -> dict:
    """tool_arguments as it was before this change (f1b70b2a9)."""
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        return {}
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            value = json.loads(raw)
        except ValueError:
            return {"arguments": raw}
        return value if isinstance(value, dict) else {"arguments": raw}
    return {"arguments": json.dumps(raw, ensure_ascii=False)}


def chat_prompt(name, arguments, tools):
    req = {"model": "m", "tools": tools, "messages": [
        {"role": "user", "content": "Write the audit module."},
        {"role": "assistant", "content": None, "reasoning_content": "One call.",
         "tool_calls": [{"id": "c1", "type": "function", "function": {"name": name, "arguments": arguments}}]},
        {"role": "tool", "tool_call_id": "c1", "content": "cut off"},
        {"role": "user", "content": "Again."}]}
    messages, tool_defs, kwargs = openai_to_messages(req)
    return TEMPLATE.render(messages, tools=tool_defs, **kwargs)


def history_call(prompt):
    """the assistant turn of the history: from its <tool_call> to the end of that tool_call block"""
    i = prompt.rfind("<tool_call>\n<function=")
    return prompt[i:prompt.index("</tool_call>", i) + len("</tool_call>")]


class CutCalls(unittest.TestCase):
    def test_a_cut_call_shows_its_complete_members_and_the_partial_last_string(self):
        shown = history_call(chat_prompt("write_file", CUT, [WRITE]))
        self.assertEqual(shown, "<tool_call>\n<function=write_file>\n<parameter=path>\ncore/lease_audit.py\n</parameter>\n"
                                "<parameter=content>\n" + CUT_TEXT + "\n</parameter>\n</function>\n</tool_call>")
        self.assertNotIn("<parameter=arguments>", shown)

    def test_a_lone_brace_renders_the_call_with_no_parameters(self):
        for raw in ("{", " { ", '{"path":', '{"path"', '{"pa', '{"start_line": 12'):
            with self.subTest(raw):
                shown = history_call(chat_prompt("write_file", raw, [WRITE]))
                self.assertEqual(shown, "<tool_call>\n<function=write_file>\n</function>\n</tool_call>")

    def test_the_recovered_members(self):
        cases = {
            CUT: {"path": "core/lease_audit.py", "content": CUT_TEXT},
            '{"path":"a.py","content":"x"': {"path": "a.py", "content": "x"},                # every pair complete, no closing brace
            '{"path":"a.py","con': {"path": "a.py"},                                         # a cut key is no member
            '{"path":"a.py","content":"b\\': {"path": "a.py", "content": "b"},               # a half escape is not text
            '{"path":"a.py","content":"b\\u00': {"path": "a.py", "content": "b"},
            '{"c":"caf\\u00e9 \\u00e': {"c": "café "},
            '{"path":"a.py","start_line":12': {"path": "a.py"},                              # a number the text ends in may be cut
            '{"path":"a.py","n":12,"content":"c': {"path": "a.py", "n": 12, "content": "c"},
            '{"a":{"b":"c"},"d":"e': {"a": {"b": "c"}, "d": "e"},
            '{"a":{"b":"c': {},                                                               # a nested value cut short is left out
            "{'path': 'a.py', 'content': 'x": {"path": "a.py", "content": "x"},              # _Lenient's slips are forgiven
        }
        for raw, want in cases.items():
            with self.subTest(raw[-30:]):
                self.assertEqual(frontend.cut_arguments(raw), want)
                self.assertEqual(tool_arguments(raw), want)

    def test_a_tool_that_declares_arguments_keeps_the_old_form(self):
        shown = history_call(chat_prompt("invoke", '{"arguments": "ls -la", "x', [WRAPPED]))
        self.assertEqual(shown, '<tool_call>\n<function=invoke>\n<parameter=arguments>\n{"arguments": "ls -la", "x\n</parameter>\n'
                                "</function>\n</tool_call>")
        self.assertEqual(tool_arguments('{"a": "b', {"arguments"}), {"arguments": '{"a": "b'})

    def test_a_tool_the_request_does_not_declare_is_recovered_too(self):
        shown = history_call(chat_prompt("gone", '{"x":"1","y":"cut', [WRITE]))
        self.assertIn("<parameter=x>\n1\n</parameter>\n<parameter=y>\ncut\n</parameter>", shown)

    def test_declared_parameters(self):
        dp = frontend.declared_parameters
        self.assertEqual(dp([WRITE], "write_file"), {"path", "content"})
        self.assertEqual(dp([{"name": "t", "input_schema": {"properties": {"a": {}}}}], "t"), {"a"})          # Anthropic
        self.assertEqual(dp([{"type": "function", "name": "t", "parameters": {"properties": {"b": {}}}}], "t"), {"b"})   # Responses
        self.assertEqual(dp([{"type": "namespace", "name": "ns", "tools": [{"type": "function", "name": "t", "parameters": {
            "properties": {"c": {}}}}]}], "ns.t"), {"c"})
        self.assertEqual(dp([{"name": "t"}], "t"), set())
        self.assertIsNone(dp([WRITE], "other"))
        self.assertIsNone(dp(None, "t"))
        self.assertIsNone(dp("x", "t"))

    def test_the_anthropic_and_responses_histories_render_the_same_members(self):
        # /v1/messages: the tool_use input as text; /v1/responses: the function_call's arguments string
        areq = {"model": "m", "tools": [{"name": "write_file", "description": "w", "input_schema": WRITE["function"]["parameters"]}],
                "messages": [{"role": "user", "content": "go"},
                             {"role": "assistant", "content": [{"type": "tool_use", "id": "c1", "name": "write_file", "input": CUT}]},
                             {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "c1", "content": "cut off"}]}]}
        msgs, tools, kw = anthropic_to_messages(areq)
        call = next(c for m in msgs for c in m.get("tool_calls") or [])
        self.assertEqual(call["function"]["arguments"], {"path": "core/lease_audit.py", "content": CUT_TEXT})
        rreq = {"model": "m", "tools": [{"type": "function", "name": "write_file", "description": "w", "parameters": WRITE["function"]["parameters"]}],
                "input": [{"role": "user", "content": "go"},
                          {"type": "function_call", "call_id": "c1", "name": "write_file", "arguments": CUT},
                          {"type": "function_call_output", "call_id": "c1", "output": "cut off"}]}
        call = next(c for m in input_messages(rreq) for c in m.get("tool_calls") or [])
        self.assertEqual(call["function"]["arguments"], {"path": "core/lease_audit.py", "content": CUT_TEXT})
        rreq["tools"] = [{"type": "function", "name": "write_file", "parameters": {"properties": {"arguments": {"type": "string"}}}}]
        call = next(c for m in input_messages(rreq) for c in m.get("tool_calls") or [])
        self.assertEqual(call["function"]["arguments"], {"arguments": CUT})                 # declared "arguments": the old form


class ValidHistoriesRenderAsBefore(unittest.TestCase):
    VALID = [
        '{"path": "core/a.py", "content": "x\\ny"}', '{}', '{"a": 1, "b": [1, 2], "c": {"d": null}}', '{"content": "\\"\\"\\"doc\\"\\"\\"\\n"}',
        '  {"x": true}  ', json.dumps({"path": "p", "content": "line\n" * 2000}), '{"arguments": "{\\"path\\": \\"a\\"}"}',
    ]
    OTHER = ["path=core/x.py", "[1, 2]", "12", "null", '"text"', "true", "ls -la", "[", "plain", "{cut", "{malformed_json", "{ x: 1", "{1}"]

    def test_valid_json_objects_and_parsed_objects_are_unchanged(self):
        for raw in self.VALID:
            with self.subTest(raw[:40]):
                self.assertEqual(tool_arguments(raw), old_tool_arguments(raw))
                self.assertEqual(tool_arguments(json.loads(raw)), old_tool_arguments(json.loads(raw)))
                for declared in (None, set(), {"arguments"}, {"path"}):
                    self.assertEqual(tool_arguments(raw, declared), old_tool_arguments(raw))

    def test_empty_and_not_object_text_are_unchanged(self):
        for raw in [None, "", "   "] + self.OTHER:
            with self.subTest(repr(raw)):
                self.assertEqual(tool_arguments(raw), old_tool_arguments(raw))
        self.assertEqual(tool_arguments(5), old_tool_arguments(5))
        self.assertEqual(tool_arguments([1]), old_tool_arguments([1]))

    def test_rendered_prompts_of_valid_histories_are_byte_identical(self):
        # the same request rendered with the old mapping function must give the same bytes
        for raw in self.VALID + self.OTHER:
            new = chat_prompt("write_file", raw, [WRITE])
            real = frontend.tool_arguments
            try:
                frontend.tool_arguments = lambda r, declared=None: old_tool_arguments(r)
                old = chat_prompt("write_file", raw, [WRITE])
            finally:
                frontend.tool_arguments = real
            self.assertEqual(new, old, raw[:40])

    def test_a_normal_streamed_call_still_closes_where_it_did(self):
        # the cut reader is never asked about text json.loads accepted
        raw = json.dumps({"path": "a.py", "content": "b" * 6000})
        self.assertEqual(tool_arguments(raw), {"path": "a.py", "content": "b" * 6000})


if __name__ == "__main__":
    unittest.main()
