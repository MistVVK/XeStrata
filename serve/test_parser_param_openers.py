"""serve/test_parser_param_openers.py - #6032 R2 (recovery on): damaged parameter openers of a declared parameter not
set yet - `<parameter=NAME: value`, a bare `<NAME>` / `</NAME>`, `<=NAME>` - read as that parameter (cases from PR #6032's
test_parser_damaged_param_openers.py, test_parser_bare_param_opener.py and test_parser_followers.py, Gyro / Coder-L
recordings), with negatives: the same tags inside a value stay byte for byte, a tag of a parameter already set stays
text, and with recovery off nothing changes.

    python -m pytest serve/test_parser_param_openers.py -q

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
    {"name": "Grep", "parameters": {"properties": {
        "pattern": {"type": "string"}, "path": {"type": "string"}, "output_mode": {"type": "string"},
        "-n": {"type": "boolean"}, "head_limit": {"type": "integer"}, "context": {"type": "number"}}}},
    {"name": "Read", "parameters": {"properties": {
        "file_path": {"type": "string"}, "offset": {"type": "integer"}, "limit": {"type": "integer"}}}},
    {"name": "Bash", "parameters": {"properties": {"command": {"type": "string"}, "timeout": {"type": "number"}}}},
    {"name": "Write", "parameters": {"properties": {"file_path": {"type": "string"}, "content": {"type": "string"}}}},
]
REST = ("<parameter=path>\nsrc/router/limits.ts\n</parameter>\n<parameter=output_mode>\ncontent\n</parameter>\n"
        "<parameter=-n>\ntrue\n</parameter>\n</function>\n</tool_call>")
CASES = {
    "colon_value_quoted": (
        "<tool_call>\n<function=Grep>\n<parameter=pattern: \"const sh|function sh|import.*sh\"\n</parameter>\n" + REST,
        ("Grep", {"pattern": "const sh|function sh|import.*sh", "path": "src/router/limits.ts", "output_mode": "content",
                  "-n": True})),
    "colon_value_quote_left_open": (
        "<tool_call>\n<function=Grep>\n<parameter=pattern: \"const sh|async function sh\n</parameter>\n" + REST,
        ("Grep", {"pattern": "const sh|async function sh", "path": "src/router/limits.ts", "output_mode": "content",
                  "-n": True})),
    "bare_opener_after_a_closer": (
        "<tool_call>\n<function=Grep>\n<parameter=pattern>\nfoo\n</parameter>\n"
        "<parameter=output_mode>\ncontent\n</parameter>\n<head_limit>\n10\n</parameter>\n</function>\n</tool_call>",
        ("Grep", {"pattern": "foo", "output_mode": "content", "head_limit": 10})),
    "closing_tag_of_a_declared_parameter_opens_it": (
        "<tool_call>\n<function=Grep>\n<parameter=pattern>\nfoo\n</parameter>\n<parameter=-n>\ntrue\n</parameter>\n"
        "</context>\n2\n</parameter>\n</function>\n</tool_call>",
        ("Grep", {"pattern": "foo", "-n": True, "context": 2})),
    "damaged_opener_of_a_declared_parameter": (
        "<tool_call>\n<function=Read>\n<parameter=file_path>\n/d/f\n</parameter>\n<=offset>\n5\n</parameter>\n"
        "</function>\n</tool_call>",
        ("Read", {"file_path": "/d/f", "offset": 5})),
    "bare_timeout": (
        "<tool_call>\n<function=Bash>\n<parameter=command>\nls\n</parameter>\n<timeout>\n300\n</parameter>\n"
        "</function>\n</tool_call>",
        ("Bash", {"command": "ls", "timeout": 300})),
    # negatives
    "a_tag_of_a_parameter_already_set_stays": (
        "<tool_call>\n<function=Bash>\n<parameter=command>\nls\n</parameter>\n</command>\nx\n</parameter>\n"
        "</function>\n</tool_call>",
        ("Bash", {"command": "ls\n</parameter>\n</command>\nx"})),
    "the_forms_inside_a_value_stay": (
        "<tool_call>\n<function=Write>\n<parameter=file_path>\na.txt\n</parameter>\n<parameter=content>\n"
        "a\n<parameter=x: y\n<function=z>\n<file_path>\n</content>\n<parameter=nope>\n<parameter=file_path: q\nend\n"
        "</parameter>\n</function>\n</tool_call>",
        ("Write", {"file_path": "a.txt", "content": "a\n<parameter=x: y\n<function=z>\n<file_path>\n</content>\n"
                                                    "<parameter=nope>\n<parameter=file_path: q\nend"})),
    "html_with_declared_names_stays": (
        "<tool_call>\n<function=Write>\n<parameter=file_path>\na.html\n</parameter>\n<parameter=content>\n"
        "<html>\n<pattern>\nx\n</pattern>\n<div>\n</html>\n</parameter>\n</function>\n</tool_call>",
        ("Write", {"file_path": "a.html", "content": "<html>\n<pattern>\nx\n</pattern>\n<div>\n</html>"})),
}


def run(text, stream, step, recover=True):
    p = OutputParser(thinking=False, tools=TOOLS, stream_tools=stream, recover=recover)
    evs = []
    for k in range(0, len(text), step):
        evs += p.feed(text[k:k + step])
    return evs + p.finish()


class ParamOpeners(unittest.TestCase):
    def test_cases(self):
        for label, (text, want) in CASES.items():
            for stream in (False, True):
                for step in (1, 3, 7, len(text)):
                    with self.subTest(label, stream=stream, step=step):
                        evs = run(text, stream, step)
                        self.assertEqual([(e.call.name, e.call.arguments) for e in evs if e.kind == "tool_call"], [want])
                        if stream:
                            streamed = "".join(e.text for e in evs if e.kind == "tool_args")
                            self.assertEqual(json.loads(streamed), want[1], streamed)

    def test_recovery_off_reads_no_damaged_opener(self):
        for label, key in (("bare_opener_after_a_closer", "head_limit"), ("bare_timeout", "timeout"),
                           ("damaged_opener_of_a_declared_parameter", "offset")):
            text = CASES[label][0]
            for stream in (False, True):
                with self.subTest(label, stream=stream):
                    calls = [e.call for e in run(text, stream, len(text), recover=False) if e.kind == "tool_call"]
                    for call in calls:
                        self.assertNotIn(key, call.arguments)


if __name__ == "__main__":
    unittest.main()
