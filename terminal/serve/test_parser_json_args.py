"""serve/test_parser_json_args.py - #6032 R3 (recovery on): a call whose body is `<arguments>` + a JSON object instead of
`<parameter=` tags (Gyro / Coder-L recording; cases from PR #6032's test_parser_json_args.py): the object is the call's
arguments, closed by `</arguments>`, by the stray `</parameter>` the model wrote, or by nothing; string values of
declared non-string parameters are read by their type; a call cut inside the object is never run.

    python -m pytest serve/test_parser_json_args.py -q
"""
import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve.frontend import OutputParser  # noqa: E402

SCHEMAS = [
    {"name": "Edit", "parameters": {"properties": {"file_path": {"type": "string"}, "old_string": {"type": "string"},
                                                   "new_string": {"type": "string"}}}},
    {"name": "Grep", "parameters": {"properties": {"pattern": {"type": "string"}, "-n": {"type": "boolean"},
                                                   "head_limit": {"type": "integer"}, "-A": {"type": "number"}}}},
    {"name": "Read", "parameters": {"properties": {"file_path": {"type": "string"}}}},
    {"name": "Write", "parameters": {"properties": {"file_path": {"type": "string"}, "content": {"type": "string"}}}},
]
OLD = ("/** Shipped defaults merged under the user's main config. */\nasync function defaultConfigReader() {\n"
       "  const { getConfigValueOp } = await import(\"../config/main-config-ops.js\");\n  return () => ({});\n}")
NEW = OLD.replace("getConfigValueOp", "effectiveMainConfig").replace("main-config-ops", "shipped-defaults")
ARGS = {"file_path": "/x/src/builds/mirror-sync.ts", "old_string": OLD, "new_string": NEW}
JSON = json.dumps(ARGS)
HEAD = "<tool_call>\n<function=Edit>\n<arguments>\n"
CASES = {
    "stray": HEAD + JSON + "\n</parameter>\n</function>\n</tool_call>",
    "closed": HEAD + JSON + "\n</arguments>\n</function>\n</tool_call>",
    "bare": HEAD + JSON + "\n</function>\n</tool_call>",
}
CUT = HEAD + JSON[:JSON.index("new_string") + 40]
TYPED = ('<tool_call>\n<function=Grep>\n<arguments>{"pattern": "p", "-n": "true", "head_limit": 20, "-A": "3"}'
         '</arguments>\n</function>\n</tool_call>')


def run(text, step, stream=True, finish="stop", recover=True):
    p = OutputParser(thinking=False, tools=SCHEMAS, stream_tools=stream, recover=recover)
    evs = []
    for k in range(0, len(text), step):
        evs += p.feed(text[k:k + step])
    return evs + p.finish(finish)


def check(case, evs, want, stream):
    case.assertEqual([(e.call.name, e.call.arguments) for e in evs if e.kind == "tool_call"], want)
    if stream:
        for e in (x for x in evs if x.kind == "tool_call"):
            streamed = "".join(x.text for x in evs if x.kind == "tool_args" and x.call is not None and x.call.id == e.call.id)
            case.assertEqual(json.loads(streamed), e.call.arguments)
    case.assertFalse([e for e in evs if e.kind in ("content", "reasoning") and e.text.strip()])


class JsonArgs(unittest.TestCase):
    def test_each_closer(self):
        for key, text in CASES.items():
            for stream in (False, True):
                for step in (1, 7, len(text)):
                    with self.subTest(key, stream=stream, step=step):
                        check(self, run(text, step, stream), [("Edit", ARGS)], stream)

    def test_types_are_read(self):
        want = {"pattern": "p", "-n": True, "head_limit": 20, "-A": 3}
        for stream in (False, True):
            for step in (1, len(TYPED)):
                with self.subTest(stream=stream, step=step):
                    check(self, run(TYPED, step, stream), [("Grep", want)], stream)

    def test_a_cut_object_is_never_a_call(self):
        for finish in ("stop", "length"):
            for stream in (False, True):
                for step in (1, 7, len(CUT)):
                    with self.subTest(finish=finish, stream=stream, step=step):
                        evs = run(CUT, step, stream, finish)
                        self.assertEqual([e for e in evs if e.kind == "tool_call"], [])

    def test_a_call_after_a_json_call(self):
        text = CASES["stray"] + "\n<tool_call>\n<function=Read>\n<parameter=file_path>\n/a\n</parameter>\n</function>\n</tool_call>"
        for stream in (False, True):
            for step in (1, len(text)):
                with self.subTest(stream=stream, step=step):
                    check(self, run(text, step, stream), [("Edit", ARGS), ("Read", {"file_path": "/a"})], stream)

    def test_arguments_text_inside_a_value_stays(self):
        content = "<arguments>\n{\"a\": 1}\n</arguments>"
        text = ("<tool_call>\n<function=Write>\n<parameter=file_path>\nx\n</parameter>\n<parameter=content>\n" + content
                + "\n</parameter>\n</function>\n</tool_call>")
        for stream in (False, True):
            for step in (1, len(text)):
                with self.subTest(stream=stream, step=step):
                    check(self, run(text, step, stream), [("Write", {"file_path": "x", "content": content})], stream)

    def test_recovery_off_reads_no_json_body(self):
        for stream in (False, True):
            evs = run(CASES["closed"], len(CASES["closed"]), stream, recover=False)
            self.assertNotIn(("Edit", ARGS), [(e.call.name, e.call.arguments) for e in evs if e.kind == "tool_call"])


if __name__ == "__main__":
    unittest.main()
