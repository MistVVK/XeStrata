"""serve/test_tool_format.py - the tool-call parser is chosen from the chat template (Qwen4 prep, P4).

  * serve/toolformat.py names the form a template documents: xml (every template Strata ships), json (Qwen3 / Hermes), dsml,
    none, other;
  * serve/frontend.py parser_for_template() maps that to a parser class: JSON templates get JsonOutputParser, everything else
    keeps OutputParser, so every model that runs today gets the class it always had;
  * serve/tool_parser_golden.json holds the events each parser gives for fixed model outputs, fed whole, in 5-character pieces and
    one character at a time, with and without streamed arguments.

    python -m unittest serve.test_tool_format
"""
from __future__ import annotations

import gzip
import json
import tempfile
import unittest
from pathlib import Path

from serve import toolformat
from serve.frontend import ChatTemplate, JsonOutputParser, OutputParser, parser_for_template

ROOT = Path(__file__).resolve().parents[1]
GOLDEN = json.loads((ROOT / "serve" / "tool_parser_golden.json").read_text(encoding="utf-8"))
FIXTURES = ROOT / "tools" / "qwen4_intake_fixtures"

HERMES = """{%- if tools %}<|im_start|>system
# Tools

You may call one or more functions to assist with the user query.

<tools>
{%- for tool in tools %}
{{ tool | tojson }}
{%- endfor %}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call><|im_end|>
{%- endif %}
{%- for m in messages %}
<|im_start|>{{ m.role }}
{{ m.content }}<|im_end|>
{%- endfor %}
{%- if add_generation_prompt %}
<|im_start|>assistant
{%- endif %}"""

NO_TOOLS = "{%- for m in messages %}<|im_start|>{{ m.role }}\n{{ m.content }}<|im_end|>\n{%- endfor %}"
ODD = "{%- if tools %}Call a tool by writing [TOOL name(args)] and nothing else.{%- endif %}{{ messages }}"


def fixture_template(name: str) -> str:
    with gzip.open(FIXTURES / (name + ".json.gz"), "rt", encoding="utf-8") as f:
        return json.load(f)["files"][0]["metadata"]["tokenizer.chat_template"]


def normalised(parser_cls, raw: str, thinking: bool, chunk: int, stream: bool) -> list:
    p = parser_cls(thinking=thinking, tools=GOLDEN["tools"], stream_tools=stream)
    events = []
    for i in range(0, len(raw), chunk):
        events += p.feed(raw[i:i + chunk])
    events += p.finish()
    out: list = []
    for e in events:
        if e.kind == "tool_call":
            out.append(["tool_call", "", e.call.name, e.call.arguments])
        elif e.kind in ("content", "reasoning"):
            if out and out[-1][0] == e.kind:
                out[-1][1] += e.text
            else:
                out.append([e.kind, e.text, None, None])
    return out


class Detection(unittest.TestCase):
    def test_every_template_strata_ships_is_xml(self):
        self.assertEqual(toolformat.tool_format((ROOT / "serve" / "chat_template.jinja").read_text(encoding="utf-8")), "xml")
        for name in ("fn_unsloth_ud_q4_k_xl", "fn_swift_iq3_s", "q36_ud_q4_k_xl"):
            with self.subTest(name):
                self.assertEqual(toolformat.tool_format(fixture_template(name)), "xml")

    def test_other_forms(self):
        self.assertEqual(toolformat.tool_format(fixture_template("ds41_ud_q8_k_xl")), "dsml")
        self.assertEqual(toolformat.tool_format(HERMES), "json")
        self.assertEqual(toolformat.tool_format(NO_TOOLS), "none")
        self.assertEqual(toolformat.tool_format(ODD), "other")
        self.assertEqual(toolformat.tool_format(""), "none")
        self.assertEqual(toolformat.tool_format(None), "none")

    def test_the_real_qwen3_template_is_json(self):
        # recorded from unsloth/Qwen3-0.6B-GGUF (Apache-2.0): the form this parser exists for
        src = (FIXTURES / "qwen3-0.6b.chat_template.jinja").read_text(encoding="utf-8")
        self.assertEqual(toolformat.tool_format(src), "json")
        self.assertIs(parser_for_template(src), JsonOutputParser)

    def test_thinking_tags(self):
        t = toolformat.thinking_tags((ROOT / "serve" / "chat_template.jinja").read_text(encoding="utf-8"))
        self.assertTrue(t["has_think_tags"])
        self.assertTrue(t["enable_thinking_switch"])
        self.assertFalse(toolformat.thinking_tags(HERMES)["has_think_tags"])


class Selection(unittest.TestCase):
    def test_shipped_templates_keep_the_class_they_had(self):
        for src in [(ROOT / "serve" / "chat_template.jinja").read_text(encoding="utf-8"),
                    *(fixture_template(n) for n in ("fn_unsloth_ud_q4_k_xl", "fn_swift_iq3_s", "q36_ud_q4_k_xl", "ds41_ud_q8_k_xl")),
                    NO_TOOLS, ODD, "", None]:
            self.assertIs(parser_for_template(src), OutputParser)

    def test_json_template_gets_the_json_parser(self):
        self.assertIs(parser_for_template(HERMES), JsonOutputParser)
        self.assertTrue(issubclass(JsonOutputParser, OutputParser))

    def test_service_picks_by_template(self):
        from serve.server import ByteTokenizer, MockEngine, Service
        with tempfile.TemporaryDirectory() as d:
            h = Path(d) / "hermes.jinja"
            h.write_text(HERMES, encoding="utf-8")
            tok = ByteTokenizer()
            self.assertIs(Service(MockEngine(tok, "ok"), tok, ChatTemplate(h)).parser_cls, JsonOutputParser)
            shipped = ChatTemplate(ROOT / "serve" / "chat_template.jinja")
            self.assertIs(Service(MockEngine(tok, "ok"), tok, shipped).parser_cls, OutputParser)


class UnknownPreTokenizerAtStart(unittest.TestCase):
    """The server stops at start on a tokenizer pack whose pre-tokenizer it has no rule for (one clear line, exit status 2), and still
    starts on the ones it has."""

    def pack(self, d: Path, pre: str) -> Path:
        t = d / "tokenizer"
        t.mkdir()
        chars = [chr(i) for i in range(32, 127)]
        (t / "vocab.json").write_text(json.dumps({c: i for i, c in enumerate(chars)}), encoding="utf-8")
        (t / "merges.txt").write_text("", encoding="utf-8")
        (t / "token_type.json").write_text(json.dumps([1] * len(chars)), encoding="utf-8")
        (t / "tokenizer.json").write_text(json.dumps({"pre": pre, "special_ids": {}}), encoding="utf-8")
        return t

    def test_unknown_pre_stops_with_a_line(self):
        import contextlib
        import io
        from serve import server
        with tempfile.TemporaryDirectory() as d:
            t = self.pack(Path(d), "qwen4")
            err = io.StringIO()
            with contextlib.redirect_stderr(err), self.assertRaises(SystemExit) as cm:
                server.main(["--engine", "mock", "--tokenizer", str(t), "--port", "0"])
            self.assertEqual(cm.exception.code, 2)
            self.assertIn("'qwen4'", err.getvalue())
            self.assertIn("so it isn't loaded", err.getvalue())
            self.assertIn("Strata doesn't have yet ('qwen4')", err.getvalue())
            self.assertIn("the model's tokenizer (", err.getvalue())
            self.assertNotIn("Traceback", err.getvalue())


class Golden(unittest.TestCase):
    def check(self, cls_for_case, cases):
        for c in cases:
            raw = c["input"]
            for chunk in (len(raw), 5, 1):
                for stream in (False, True):
                    with self.subTest(case=c["name"], chunk=chunk, stream_tools=stream):
                        self.assertEqual(normalised(cls_for_case, raw, c["thinking"], chunk, stream), c["events"])

    def test_xml_form_through_the_class_every_shipped_template_selects(self):
        for src in (fixture_template("fn_unsloth_ud_q4_k_xl"), fixture_template("q36_ud_q4_k_xl")):
            self.check(parser_for_template(src), GOLDEN["xml"])

    def test_json_form_through_the_class_a_json_template_selects(self):
        self.check(parser_for_template(HERMES), GOLDEN["json"])

    def test_json_parser_reads_json_without_the_recovery_switch(self):
        raw = GOLDEN["json"][0]["input"]
        p = JsonOutputParser(thinking=False, tools=GOLDEN["tools"], recover=False)
        calls = [e.call for e in p.feed(raw) + p.finish() if e.kind == "tool_call"]
        self.assertEqual([(c.name, c.arguments) for c in calls], [("get_weather", {"city": "Paris", "days": 3})])
        p = OutputParser(thinking=False, tools=GOLDEN["tools"], recover=False)
        calls = [e.call for e in p.feed(raw) + p.finish() if e.kind == "tool_call"]
        self.assertEqual(calls, [])                      # the XML parser alone leaves JSON as text, as it always did

    def test_xml_form_is_not_changed_by_the_json_parser_existing(self):
        self.check(OutputParser, GOLDEN["xml"])


if __name__ == "__main__":
    unittest.main()
