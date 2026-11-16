"""serve/test_parser_followers.py - #6032 R1: what ends a parameter value after its `</parameter>`, and the parameter
names a damaged opener writes (cases from PR #6032's test_parser_followers.py, Gyro / Coder-L recordings; the ones of
later repairs left out), plus negatives: text that only looks like a follower stays in the value.

    python -m pytest serve/test_parser_followers.py -q

Every case: whole and in pieces of 1 and 5 bytes, with the arguments streamed and not; the streamed JSON of a call is
its final arguments.
"""
import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve.frontend import OutputParser, param_name  # noqa: E402

LT, GT = "<", ">"
F, FE = LT + "function=", LT + "/function" + GT
P, PE = LT + "parameter=", LT + "/parameter" + GT
CS, CE = LT + "tool_call" + GT, LT + "/tool_call" + GT


def tool(name, **props):
    return {"name": name, "parameters": {"type": "object", "properties": {k: {"type": v} for k, v in props.items()}}}


SCHEMAS = [tool("Grep", pattern="string", path="string", head_limit="integer", context="number", **{"-n": "boolean"}),
           tool("Read", file_path="string", offset="integer"),
           tool("Bash", command="string", description="string", timeout="number", run_in_background="boolean")]


def run(text, stream, step):
    p = OutputParser(thinking=False, tools=SCHEMAS, stream_tools=stream, recover=True)
    evs = []
    for k in range(0, len(text), step):
        evs += p.feed(text[k:k + step])
    return evs + p.finish()


def body(name, inner, close=True):
    return CS + "\n" + F + name + GT + "\n" + inner + ("\n" + FE + "\n" + CE if close else "")


CASES = {
    # followers (the PR's)
    "damaged_opener_ends_the_value": (
        body("Grep", P + "pattern" + GT + "\nfoo\n" + PE + "\n" + P + "path" + GT + "\n/d/src\n" + PE + "\n"
             + LT + "=limit" + GT + "\n60\n" + PE),
        ("Grep", {"pattern": "foo", "path": "/d/src"})),
    "garbled_closer_follower": (
        body("Grep", P + "pattern" + GT + "\nfoo\n" + PE + "\n" + P + "path" + GT + "\n/d/src\n" + PE + "\n"
             + LT + "/" + LT + "novel" + GT),
        ("Grep", {"pattern": "foo", "path": "/d/src"})),
    "anthropic_invoke_closer": (
        CS + "\n" + F + "Read" + GT + "\n" + P + "file_path" + GT + "\n/d/f\n" + PE + "\n</invoke>\n" + CE,
        ("Read", {"file_path": "/d/f"})),
    "anthropic_parameter_name_follower": (
        body("Grep", P + "pattern" + GT + "\nfoo\n" + PE + "\n<parameter name=\"path\">/d</parameter>"),
        ("Grep", {"pattern": "foo"})),
    "stray_closer_before_the_wrapper_closer": (
        CS + "\n" + F + "Read" + GT + "\n" + P + "file_path" + GT + "\n/d/f\n" + PE + "\n</Read>\n" + CE,
        ("Read", {"file_path": "/d/f"})),
    # parameter names (the PR's)
    "name_runs_past_closer": (
        body("Bash", P + "command " + PE + "\necho hi\n" + PE + "\n" + P + "timeout" + GT + "\n2.5\n" + PE),
        ("Bash", {"command": "echo hi", "timeout": 2.5})),
    "name_null_dropped": (
        body("Bash", P + "command" + GT + "\nls\n" + PE + "\n" + P + " null " + PE + "\nx\n" + PE),
        ("Bash", {"command": "ls"})),
    "name_think_cut": (
        body("Bash", P + "command" + GT + "\nls\n" + PE + "\n" + P + "command " + LT + "/think" + GT + "\nx\n" + PE),
        ("Bash", {"command": "x"})),
    "function_is_no_parameter": (
        body("Bash", P + "command" + GT + "\nls\n" + PE + "\n" + P + "function" + GT + "\nBash\n" + PE),
        ("Bash", {"command": "ls"})),
    "coerced_after_cut": (
        body("Bash", P + "command" + GT + "\nls\n" + PE + "\n" + P + "run_in_background " + GT + "\ntrue\n" + PE),
        ("Bash", {"command": "ls", "run_in_background": True})),
    # negatives: these stay in the value (the PR's last two, and ours)
    "closing_tag_of_a_value_stays": (
        body("Bash", P + "command" + GT + "\nls\n" + PE + "\n" + LT + "/command" + GT + "\nx\n" + PE),
        ("Bash", {"command": "ls\n" + PE + "\n" + LT + "/command" + GT + "\nx"})),
    "plain_text_after_closer_stays": (
        body("Bash", P + "command" + GT + "\necho a\n" + PE + " then text\n" + PE),
        ("Bash", {"command": "echo a" + "\n" + PE + " then text"})),
    "a_tag_line_of_the_value_stays": (
        body("Bash", P + "command" + GT + "\ncat <<EOF\n" + PE + "\n<div>\nEOF\n" + PE),
        ("Bash", {"command": "cat <<EOF\n" + PE + "\n<div>\nEOF"})),
    "a_stray_closer_not_before_the_wrapper_stays": (
        body("Bash", P + "command" + GT + "\nprintf x\n" + PE + "\n</b> and more\n" + PE),
        ("Bash", {"command": "printf x\n" + PE + "\n</b> and more"})),
    "a_spaced_name_is_the_name": (
        body("Read", P + " file_path  " + GT + "\n/d/f\n" + PE),
        ("Read", {"file_path": "/d/f"})),
}


class Followers(unittest.TestCase):
    def test_cases(self):
        for label, (text, want) in CASES.items():
            for stream in (False, True):
                for step in (1, 5, len(text)):
                    with self.subTest(label, stream=stream, step=step):
                        evs = run(text, stream, step)
                        calls = [e for e in evs if e.kind == "tool_call"]
                        self.assertEqual([(e.call.name, e.call.arguments) for e in calls], [want])
                        if stream:
                            streamed = "".join(e.text for e in evs if e.kind == "tool_args")
                            self.assertEqual(json.loads(streamed), want[1], streamed)

    def test_prose_that_shows_the_forms_stays_text(self):
        text = "Write `</parameter>` then `</invoke>` or `<=limit>` in your notes; `<parameter name=\"x\">` too."
        for stream in (False, True):
            for step in (1, len(text)):
                evs = run(text, stream, step)
                self.assertFalse([e for e in evs if e.kind in ("tool_call", "tool_start")])
                self.assertEqual("".join(e.text for e in evs if e.kind == "content"), text)

    def test_param_name(self):
        self.assertEqual(param_name("path"), "path")
        self.assertEqual(param_name(" path "), "path")
        self.assertEqual(param_name("command </parameter"), "command")
        for dropped in ("", " ", " null ", "None", "function", "think", "parameter"):
            self.assertIsNone(param_name(dropped), dropped)


if __name__ == "__main__":
    unittest.main()
