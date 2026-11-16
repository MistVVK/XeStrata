"""serve/test_parser_function_names.py - #6032 R6 (recovery on): the tool's opener written damaged right after
`<tool_call>` - its `>` left out (`<function=Grep` + line break), the bare `<Grep>`, `<=Read>` - for a declared tool
(Gyro / Coder-L recordings; cases from PR #6032's test_parser_followers.py, test_parser_stub.py D,
test_parser_bare_param_opener.py A, test_parser_damaged_opener.py and test_parser_cut_name.py), with negatives: an
undeclared name stays text, a call cut inside its name is no call.

    python -m pytest serve/test_parser_function_names.py -q
"""
import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve.frontend import OutputParser  # noqa: E402

PE = "</parameter>"
SCHEMAS = [
    {"name": "Grep", "parameters": {"properties": {"pattern": {"type": "string"}, "path": {"type": "string"},
                                                   "output_mode": {"type": "string"}, "-n": {"type": "boolean"}}}},
    {"name": "Read", "parameters": {"properties": {"file_path": {"type": "string"}}}},
    {"name": "Bash", "parameters": {"properties": {"command": {"type": "string"}, "description": {"type": "string"}}}},
]
CASES = {
    "name_ends_at_the_line": ("<tool_call>\n<function=Grep\n<parameter=pattern>\nfoo\n" + PE + "\n</function>\n</tool_call>",
                              [("Grep", {"pattern": "foo"})]),
    "bare_tool_opener": ("<tool_call>\n<Grep>\n<parameter=pattern>\nloop\n" + PE + "\n<parameter=path>\ntest\n" + PE
                         + "\n<parameter=-n>\ntrue\n" + PE + "\n</function>\n</tool_call>",
                         [("Grep", {"pattern": "loop", "path": "test", "-n": True})]),
    "damaged_tool_opener": ("<tool_call>\n<=Read>\n<parameter=file_path>\nsrc/router/router.ts\n" + PE + "\n</function>\n</tool_call>",
                            [("Read", {"file_path": "src/router/router.ts"})]),
    # negatives
    "undeclared_bare_opener_stays_text": ("<tool_call>\n<Nope>\n<parameter=x>\n1\n" + PE + "\n</function>\n</tool_call>", []),
    "undeclared_damaged_opener_stays_text": ("<tool_call>\n<=Nope>\n<parameter=x>\n1\n" + PE + "\n</function>\n</tool_call>", []),
    "cut_inside_the_name": ("<tool_call>\n<function=Ba", []),
    "cut_inside_a_parameter_name": ("<tool_call>\n<function=Bash>\n<parameter=command>\necho ok\n" + PE + "\n<parameter=descr",
                                    []),
}


def run(text, step, stream=True, recover=True):
    p = OutputParser(thinking=False, tools=SCHEMAS, stream_tools=stream, recover=recover)
    evs = []
    for k in range(0, len(text), step):
        evs += p.feed(text[k:k + step])
    return evs + p.finish()


class FunctionNames(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
