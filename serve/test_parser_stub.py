"""serve/test_parser_stub.py - #6032 R5 (recovery on): the model's empty stub after a value - a lone `<` or `</` line, a
line of one tag, closed by a `</parameter>` or followed by the next opener, a closer written again - is skipped and the
value ends at its own closer; an opener cut before its `>` is dropped; a call whose every parameter was unreadable is
never run with no arguments (Gyro / Coder-L recordings; cases from PR #6032's test_parser_stub.py and
test_parser_xml_elems.py J), with negatives: a `<` line inside a value stays.

    python -m pytest serve/test_parser_stub.py -q
"""
import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve.frontend import OutputParser  # noqa: E402

PE = "</parameter>"
SCHEMAS = [
    {"name": "Bash", "parameters": {"properties": {"command": {"type": "string"}, "description": {"type": "string"},
                                                   "timeout": {"type": "number"}}}},
    {"name": "Grep", "parameters": {"properties": {"pattern": {"type": "string"}, "path": {"type": "string"},
                                                   "output_mode": {"type": "string"}}}},
    {"name": "Write", "parameters": {"properties": {"file_path": {"type": "string"}, "content": {"type": "string"}}}},
]
CMD = 'ls src/builds/ && grep -rn "defaultConfigReader\\|readConfig" src/builds/*.ts | head -20'
WRITE_BODY = 'import { x } from "y";\n\nit("z", () => {});\n'
GREP = {"pattern": "a|b", "path": "/x/src", "output_mode": "content"}
I_CALL = ("<tool_call>\n<function=Bash>\n<parameter=command>\ngrep -n x c.ts\n" + PE + "\n<parameter=description>\nInspect\n"
          + PE + "\n</function>\n</tool_call>")
CASES = {
    "A_lone_lt_then_closer": ("<tool_call>\n<function=Bash>\n<parameter=command>\n" + CMD + "\n" + PE + "\n<\n" + PE
                              + "\n</function>\n</tool_call>", [("Bash", {"command": CMD})]),
    "B_stub_then_more": ("<tool_call>\n<function=Bash>\n<parameter=command>\nls\n" + PE + "\n<\n" + PE
                         + "\n<parameter=description>\nd\n" + PE + "\n<parameter=timeout>\n120000\n" + PE
                         + "\n</function>\n</tool_call>", [("Bash", {"command": "ls", "description": "d", "timeout": 120000})]),
    "C_stub_after_the_opener_and_at_the_end": (
        "<tool_call>\n<function=Grep>\n<\n" + PE + "\n<parameter=pattern>\na|b\n" + PE + "\n<parameter=path>\n/x/src\n"
        + PE + "\n<parameter=output_mode>\ncontent\n" + PE + "\n<\n" + PE + "\n</function>\n</tool_call>", [("Grep", GREP)]),
    "E_lone_lt_left_open": ("<tool_call>\n<function=Write>\n<parameter=file_path>\n/t/p.ts\n" + PE + "\n<\n"
                            "<parameter=content>\n" + WRITE_BODY + "\n" + PE + "\n</function>\n</tool_call>",
                            [("Write", {"file_path": "/t/p.ts", "content": WRITE_BODY})]),
    "F_closer_written_again": ("<tool_call>\n<function=Bash>\n<parameter=command>\nps\n" + PE + "\n<parameter=timeout>\n60000\n"
                               + PE + "\n" + PE + "\n" + PE + "\n</function>\n</tool_call>",
                               [("Bash", {"command": "ps", "timeout": 60000})]),
    "G_opener_cut_before_its_gt": ("<tool_call>\n<function=Bash>\n<parameter=command\n" + PE + "\n<parameter=command>\n"
                                   "grep -rln x scripts/\n" + PE + "\n" + PE + "\n<parameter=description>\nList\n" + PE
                                   + "\n</function>\n</tool_call>",
                                   [("Bash", {"command": "grep -rln x scripts/", "description": "List"})]),
    "H_lone_slash_line": ("<tool_call>\n<function=Bash>\n<parameter=command>\nnpx vitest run\n" + PE + "\n<parameter=timeout>\n"
                          "120000\n" + PE + "\n</\n" + PE + "\n</function>\n</tool_call>",
                          [("Bash", {"command": "npx vitest run", "timeout": 120000})]),
    "I_tag_line_stub": ("<tool_call>\n<function=Grep>\n<parameter=pattern>\npruneDist|prune-dist\n" + PE + "\n<=output_mode>\n"
                        "files_with_matches\n" + PE + "\n</head_after>\n\n" + PE + "\n</function>\n</tool_call>",
                        [("Grep", {"pattern": "pruneDist|prune-dist", "output_mode": "files_with_matches"})]),
    "J_only_a_broken_opener_is_no_call": ("<tool_call>\n<function=Bash>\n<parameter=command\n" + PE + "\n</function>\n</tool_call>",
                                          []),
    "J2_then_a_whole_call": ("<tool_call>\n<function=Bash>\n<parameter=command\n" + PE + "\n</function>\n</tool_call>\n" + I_CALL,
                             [("Bash", {"command": "grep -n x c.ts", "description": "Inspect"})]),
    # negative: a `<` line and a tag line inside a value, not before a closer or opener, stay
    "a_lt_line_inside_a_value_stays": ("<tool_call>\n<function=Write>\n<parameter=file_path>\nf\n" + PE + "\n<parameter=content>\n"
                                       "a\n<\nb\n<div>\nc\n" + PE + "\n</function>\n</tool_call>",
                                       [("Write", {"file_path": "f", "content": "a\n<\nb\n<div>\nc"})]),
}


def run(text, step, stream=True, recover=True):
    p = OutputParser(thinking=False, tools=SCHEMAS, stream_tools=stream, recover=recover)
    evs = []
    for k in range(0, len(text), step):
        evs += p.feed(text[k:k + step])
    return evs + p.finish()


class Stubs(unittest.TestCase):
    def test_cases(self):
        for label, (text, want) in CASES.items():
            for stream in (False, True):
                for step in (1, 7, len(text)):
                    with self.subTest(label, stream=stream, step=step):
                        evs = run(text, step, stream)
                        calls = [e for e in evs if e.kind == "tool_call"]
                        self.assertEqual([(e.call.name, e.call.arguments) for e in calls], want)
                        if stream:
                            for e in calls:
                                streamed = "".join(x.text for x in evs if x.kind == "tool_args" and x.call is not None
                                                   and x.call.id == e.call.id)
                                self.assertEqual(json.loads(streamed), e.call.arguments, streamed)

    def test_no_call_ever_runs_with_no_arguments(self):
        for label, (text, _) in CASES.items():
            for stream in (False, True):
                for step in (1, len(text)):
                    for e in run(text, step, stream):
                        if e.kind == "tool_call":
                            self.assertTrue(e.call.arguments, (label, stream, step))


if __name__ == "__main__":
    unittest.main()
