"""serve/test_tool_opener_misspelt.py - #5276: `<toolcall>` (no underscore) before `<function=NAME>` of a declared tool opens the call.

Seen live (Qwen3.8-Flash-Next UD-Q4_K_XL, OpenMinis on /v1/messages): the model wrote the opener as ordinary text, misspelled
`<toolcall>`, then a whole call of a declared tool and the real `</tool_call>`.  The block came back as answer text, the turn
ended and the agent waited for a result that never came.  The misspelling counts only where a `<tool_call>` would open a call
(start of a line, outside code) and only when a declared tool's `<function=NAME>` follows; prose never triggers it.

    python -m unittest serve.test_tool_opener_misspelt -v
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from serve.frontend import OutputParser  # noqa: E402

BASH = {"name": "bash", "parameters": {"properties": {"command": {"type": "string"}, "tooltitle": {"type": "string"}}}}
BODY = ("<function=bash>\n<parameter=command>\ncd proj && rg -o 'x' landing.html | sort | uniq -c\n</parameter>\n"
        "<parameter=tooltitle>\nCount\n</parameter>\n</function>\n")
ARGS = {"command": "cd proj && rg -o 'x' landing.html | sort | uniq -c", "tooltitle": "Count"}


def run(text, stream_tools=False, recover=True, thinking=False, step=3, tools=(BASH,)):
    p = OutputParser(thinking=thinking, tools=list(tools), stream_tools=stream_tools, recover=recover)
    evs = []
    for i in range(0, len(text), step):
        evs += p.feed(text[i:i + step])
    evs += p.finish("stop")
    return evs


def calls(evs):
    return [(e.call.name, e.call.arguments) for e in evs if e.kind == "tool_call"]


def text(evs):
    return "".join(e.text or "" for e in evs if e.kind in ("content", "reasoning"))


class Misspelt(unittest.TestCase):
    def test_issue_block(self):
        for opener in ("<toolcall>", "<tool-call>"):
            for stream in (False, True):
                for step in (1, 3, 1000):
                    for recover in (False, True):
                        evs = run(opener + "\n" + BODY + "</tool_call>", stream, recover, step=step)
                        self.assertEqual(calls(evs), [("bash", ARGS)], (opener, stream, step, recover))
                        self.assertEqual(text(evs).strip(), "")

    def test_without_the_closing_tag_as_the_real_opener(self):
        self.assertEqual(calls(run("<toolcall>\n" + BODY)), calls(run("<tool_call>\n" + BODY)))

    def test_mid_line_as_the_real_opener(self):
        t = "ok <toolcall>\n" + BODY + "</tool_call>"
        self.assertEqual(calls(run(t)), calls(run(t.replace("<toolcall>", "<tool_call>"))))
        self.assertEqual(calls(run(t)), [("bash", ARGS)])

    def test_in_the_reasoning_mid_line_stays_text(self):
        evs = run("hm <toolcall>\n" + BODY + "</tool_call>\n</think>\n", thinking=True)
        self.assertEqual(calls(evs), [])

    def test_text_before_is_kept(self):
        evs = run("I will count.\n<toolcall>\n" + BODY + "</tool_call>")
        self.assertEqual(calls(evs), [("bash", ARGS)])
        self.assertEqual(text(evs).strip(), "I will count.")

    def test_in_the_reasoning(self):
        evs = run("hm\n<toolcall>\n" + BODY + "</tool_call>\n</think>\n", thinking=True)
        self.assertEqual(calls(evs), [("bash", ARGS)])

    def test_prose_naming_the_tag_stays_text(self):
        for t in ("The model may write <toolcall> by mistake.",
                  "Write <toolcall> and then a call.\n",
                  "<toolcall>\nnot a call\n</tool_call>",
                  "<toolcall>",
                  "<toolcall>\n<function=nothing>\n<parameter=a>\n1\n</parameter>\n</function>\n</tool_call>"):
            for stream in (False, True):
                evs = run(t, stream)
                self.assertEqual(calls(evs), [], t)
                self.assertEqual(text(evs), t)

    def test_code_fence_stays_text(self):
        t = "Example:\n```\n<toolcall>\n" + BODY + "</tool_call>\n```\n"
        evs = run(t)
        self.assertEqual(calls(evs), [])
        self.assertEqual(text(evs), t)

    def test_indented_code_stays_text(self):
        t = "Example:\n\n    <toolcall>\n    " + BODY.replace("\n", "\n    ") + "</tool_call>\n"
        self.assertEqual(calls(run(t)), [])

    def test_no_tools_declared(self):
        t = "<toolcall>\n" + BODY + "</tool_call>"
        self.assertEqual(calls(run(t, tools=())), [])
        self.assertEqual(text(run(t, tools=())), t)

    def test_the_real_opener_is_unchanged(self):
        evs = run("<tool_call>\n" + BODY + "</tool_call>")
        self.assertEqual(calls(evs), [("bash", ARGS)])


if __name__ == "__main__":
    unittest.main()
