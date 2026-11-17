"""serve/test_parser_named_param.py - #5969's third form (recovery on): inside the call format's `<function=NAME>`, a
parameter opened as `<parameter name="K">` (the Anthropic XML form's opener; a user's report of stopped agent turns has
it, and the trailing-`</parameter>` replay of 11-16 lost the parameter) is parameter K when K is a declared parameter of
that tool not set yet.  Negatives: an undeclared K, the same opener inside a value, a parameter already set, recovery off.

    python -m pytest serve/test_parser_named_param.py -q

Every case whole and in pieces of 1, 3 and 7 bytes, with the arguments streamed and not; the streamed JSON of a call
is its final arguments.
"""
import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve.frontend import OutputParser  # noqa: E402

TOOLS = [
    {"name": "pwsh", "parameters": {"properties": {"command": {"type": "string"}, "description": {"type": "string"},
                                                   "timeout": {"type": "integer"}}}},
    {"name": "write", "parameters": {"properties": {"file_path": {"type": "string"}, "content": {"type": "string"}}}},
]
CASES = {
    "quoted_name": (
        '<tool_call>\n<function=pwsh>\n<parameter name="command">\nGet-ChildItem .\n</parameter>\n</function>\n</tool_call>',
        ("pwsh", {"command": "Get-ChildItem ."})),
    "mixed_with_the_template_form": (
        '<tool_call>\n<function=pwsh>\n<parameter=command>\nls\n</parameter>\n<parameter name="description">\nlist\n'
        '</parameter>\n<parameter name="timeout">\n30\n</parameter>\n</function>\n</tool_call>',
        ("pwsh", {"command": "ls", "description": "list", "timeout": 30})),
    "unquoted_name_value_on_the_same_line": (
        "<tool_call>\n<function=pwsh>\n<parameter name=command>ls -la</parameter>\n</function>\n</tool_call>",
        ("pwsh", {"command": "ls -la"})),
    "first_named_then_the_template_form": (
        '<tool_call>\n<function=write>\n<parameter name="file_path">\na.txt\n</parameter>\n<parameter=content>\nhi\n'
        "</parameter>\n</function>\n</tool_call>",
        ("write", {"file_path": "a.txt", "content": "hi"})),
    # negatives
    "an_undeclared_name_is_dropped_not_guessed": (
        '<tool_call>\n<function=pwsh>\n<parameter=command>\nls\n</parameter>\n<parameter name="cwd">\n/tmp\n</parameter>\n'
        "</function>\n</tool_call>",
        ("pwsh", {"command": "ls"})),
    "the_opener_inside_a_value_stays": (
        '<tool_call>\n<function=write>\n<parameter=file_path>\nx.xml\n</parameter>\n<parameter=content>\n'
        '<doc>\n<parameter name="file_path">\n<parameter name="content">\n</doc>\n</parameter>\n</function>\n</tool_call>',
        ("write", {"file_path": "x.xml",
                   "content": '<doc>\n<parameter name="file_path">\n<parameter name="content">\n</doc>'})),
    "a_parameter_already_set_is_not_opened_again": (
        '<tool_call>\n<function=pwsh>\n<parameter=command>\nls\n</parameter>\n<parameter name="command">\nrm -rf x\n'
        "</parameter>\n</function>\n</tool_call>",
        ("pwsh", {"command": "ls"})),
}


def run(text, stream, step, recover=True):
    p = OutputParser(thinking=False, tools=TOOLS, stream_tools=stream, recover=recover)
    evs = []
    for k in range(0, len(text), step):
        evs += p.feed(text[k:k + step])
    return evs + p.finish()


class NamedParam(unittest.TestCase):
    def test_cases(self):
        for label, (text, want) in CASES.items():
            for stream in (False, True):
                for step in (1, 3, 7, len(text)):
                    with self.subTest(label, stream=stream, step=step):
                        evs = run(text, stream, step)
                        self.assertEqual([(e.call.name, e.call.arguments) for e in evs if e.kind == "tool_call"], [want])
                        self.assertEqual("".join(e.text or "" for e in evs if e.kind == "content"), "")
                        if stream:
                            streamed = "".join(e.text for e in evs if e.kind == "tool_args")
                            self.assertEqual(json.loads(streamed), want[1], streamed)

    def test_recovery_off_reads_no_named_opener(self):
        text = CASES["quoted_name"][0]
        for stream in (False, True):
            with self.subTest(stream=stream):
                for e in run(text, stream, len(text), recover=False):
                    if e.kind == "tool_call":
                        self.assertNotIn("command", e.call.arguments)


if __name__ == "__main__":
    unittest.main()
