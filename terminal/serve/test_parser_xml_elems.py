"""serve/test_parser_xml_elems.py - #6032 R4 (recovery on): declared parameters written as XML elements `<NAME>value</NAME>`,
alone or mixed with `<parameter=NAME>` ones, and a call closed by a stray `</user>` (Gyro / Coder-L recordings; cases
from PR #6032's test_parser_xml_elems.py), with negatives: an undeclared tag and element-shaped text inside a value
stay byte for byte; with recovery off nothing changes.

    python -m pytest serve/test_parser_xml_elems.py -q
"""
import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve.frontend import OutputParser  # noqa: E402

SCHEMAS = [
    {"name": "Edit", "parameters": {"properties": {"file_path": {"type": "string"}, "old_string": {"type": "string"},
                                                   "new_string": {"type": "string"}, "replace_all": {"type": "boolean"}}}},
    {"name": "Bash", "parameters": {"properties": {"command": {"type": "string"}, "description": {"type": "string"},
                                                   "timeout": {"type": "number"}}}},
    {"name": "Write", "parameters": {"properties": {"file_path": {"type": "string"}, "content": {"type": "string"}}}},
]
PE = "</parameter>"
OLD = ("/** Shipped defaults merged under the user's main config, resolved at call time. */\n"
       "async function defaultConfigReader(): Promise<ConfigReader> {\n"
       "  const { getConfigValueOp } = await import(\"../config/main-config-ops.js\");\n"
       "  return () => ({ release: { forgeTokenEnv: v } });\n}")
NEW = OLD.replace("getConfigValueOp", "effectiveMainConfig").replace("main-config-ops", "shipped-defaults")
PATH = "/x/src/builds/mirror-sync.ts"
CASES = {
    "mixed": ("<tool_call>\n<function=Edit>\n<parameter=replace_all>\nfalse\n" + PE + "\n<parameter=file_path>\n" + PATH
              + "\n" + PE + "\n<old_string>\n" + OLD + "\n</old_string>\n<new_string>\n" + NEW
              + "\n</new_string>\n</function>\n</tool_call>",
              ("Edit", {"replace_all": False, "file_path": PATH, "old_string": OLD, "new_string": NEW})),
    "elements_only_and_a_stray_closer": (
        "<tool_call>\n<function=Bash>\n<command>\nls -la\n</command>\n<timeout>\n5000\n</timeout>\n</user>\n</tool_call>",
        ("Bash", {"command": "ls -la", "timeout": 5000})),
    "an_element_holding_tags": (
        "<tool_call>\n<function=Edit>\n<file_path>\n" + PATH + "\n</file_path>\n<old_string>\na <b>c</b>\n</old_string>\n"
        "<new_string>\nx\n</new_string>\n</function>\n</tool_call>",
        ("Edit", {"file_path": PATH, "old_string": "a <b>c</b>", "new_string": "x"})),
    "a_stray_user_closer": (
        "<tool_call>\n<function=Bash>\n<parameter=command>\ngrep -n x src/daemon/config.ts\n" + PE
        + "\n<parameter=description>\nInspect maskKey and loadConfig\n" + PE + "\n</user>\n</tool_call>",
        ("Bash", {"command": "grep -n x src/daemon/config.ts", "description": "Inspect maskKey and loadConfig"})),
    # negatives
    "an_undeclared_tag_stays_in_the_value": (
        "<tool_call>\n<function=Bash>\n<parameter=command>\necho '<b>x</b>' </user>\n" + PE + "\n</function>\n</tool_call>",
        ("Bash", {"command": "echo '<b>x</b>' </user>"})),
    "element_text_inside_a_value_stays": (
        "<tool_call>\n<function=Write>\n<parameter=file_path>\na.xml\n</parameter>\n<parameter=content>\n"
        "<file_path>\n/etc/x\n</file_path>\n<content>\ny\n</content>\n</parameter>\n</function>\n</tool_call>",
        ("Write", {"file_path": "a.xml", "content": "<file_path>\n/etc/x\n</file_path>\n<content>\ny\n</content>"})),
}


def run(text, step, stream=True, recover=True):
    p = OutputParser(thinking=False, tools=SCHEMAS, stream_tools=stream, recover=recover)
    evs = []
    for k in range(0, len(text), step):
        evs += p.feed(text[k:k + step])
    return evs + p.finish()


class XmlElements(unittest.TestCase):
    def test_cases(self):
        for label, (text, want) in CASES.items():
            for stream in (False, True):
                for step in (1, 7, len(text)):
                    with self.subTest(label, stream=stream, step=step):
                        evs = run(text, step, stream)
                        self.assertEqual([(e.call.name, e.call.arguments) for e in evs if e.kind == "tool_call"], [want])
                        if stream:
                            streamed = "".join(e.text for e in evs if e.kind == "tool_args")
                            self.assertEqual(json.loads(streamed), want[1], streamed)

    def test_recovery_off_reads_no_element(self):
        text = CASES["elements_only_and_a_stray_closer"][0]
        for stream in (False, True):
            evs = run(text, len(text), stream, recover=False)
            for e in (x for x in evs if x.kind == "tool_call"):
                self.assertNotIn("command", e.call.arguments)


if __name__ == "__main__":
    unittest.main()
