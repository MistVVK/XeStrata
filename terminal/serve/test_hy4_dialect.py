"""serve/hy4_dialect.py and Hy4OutputParser: Hy4-Flash's tagged reasoning and tool-call tokens read by the parser serve already has, the reasoning effort words, the forced call's
opening, and the real chat template's own rendering of a call parsed back to the same call.

    python -m pytest serve/test_hy4_dialect.py
"""
from __future__ import annotations

import sys
import unittest
from unittest import mock
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from serve import frontend, hy4_dialect as D, hy4_kind, toolformat  # noqa: E402
from serve.frontend import ChatTemplate, Hy4OutputParser, OutputParser, parser_for_template  # noqa: E402

TEMPLATE = ROOT / "tests" / "data" / "hy4" / "chat_template.jinja"
QWEN_TEMPLATE = ROOT / "serve" / "chat_template.jinja"

CALL = ("<tool_calls:opensource><tool_call:opensource>get_weather<arg_key:opensource>city</arg_key:opensource><arg_value:opensource>Tokyo</arg_value:opensource>"
        "<arg_key:opensource>days</arg_key:opensource><arg_value:opensource>3</arg_value:opensource></tool_call:opensource></tool_calls:opensource>")
QWEN_CALL = ("<tool_call>\n<function=get_weather>\n<parameter=city>\nTokyo\n</parameter>\n<parameter=days>\n3\n</parameter>\n</function>\n</tool_call>")
TOOLS = [{"name": "get_weather", "description": "d", "parameters": {"type": "object", "properties": {"city": {"type": "string"}, "days": {"type": "integer"}}}}]


def pieces(text: str, n: int):
    for i in range(0, len(text), n):
        yield text[i:i + n]


def stream(text: str, n: int) -> str:
    d = D.Hy4Dialect()
    out = "".join(d.feed(p) for p in pieces(text, n))
    return out + d.feed("", final=True)


def events(text: str, n: int = 7, thinking: bool = False, tools=TOOLS):
    p = Hy4OutputParser(thinking=thinking, tools=tools, recover=True)
    out = []
    for piece in pieces(text, n):
        out += p.feed(piece)
    out += p.finish("stop")
    return out


class Translate(unittest.TestCase):
    def test_a_call_becomes_the_qwen_form(self):
        self.assertEqual(D.translate(CALL), QWEN_CALL)

    def test_every_split_point_gives_the_same_text(self):
        whole = D.translate(CALL)
        for n in range(1, 20):
            self.assertEqual(stream(CALL, n), whole, n)
        text = "<think:opensource>hmm</think:opensource>Sure." + CALL
        want = "<think>hmm</think>Sure." + QWEN_CALL
        for n in (1, 2, 3, 5, 11, 40):
            self.assertEqual(stream(text, n), want, n)

    def test_think_tags(self):
        self.assertEqual(D.translate("a<think:opensource>b</think:opensource>c"), "a<think>b</think>c")

    def test_a_value_may_hold_angle_brackets_and_tag_like_text(self):
        v = "if a < b and x<think:opensource> or </arg_key:opensource> then <div>"
        text = f"<tool_calls:opensource><tool_call:opensource>write<arg_key:opensource>body</arg_key:opensource><arg_value:opensource>{v}</arg_value:opensource></tool_call:opensource></tool_calls:opensource>"
        self.assertEqual(D.translate(text), f"<tool_call>\n<function=write>\n<parameter=body>\n{v}\n</parameter>\n</function>\n</tool_call>")

    def test_a_lone_angle_bracket_in_text_is_kept(self):
        self.assertEqual(stream("1 < 2 and <b>bold</b> <tool", 3), "1 < 2 and <b>bold</b> <tool")

    def test_two_calls_and_a_call_without_arguments(self):
        text = ("<tool_calls:opensource><tool_call:opensource>now</tool_call:opensource><tool_call:opensource>f<arg_key:opensource>a</arg_key:opensource>"
                "<arg_value:opensource>1</arg_value:opensource></tool_call:opensource></tool_calls:opensource>")
        self.assertEqual(D.translate(text), "<tool_call>\n<function=now>\n</function>\n</tool_call>"
                                            "<tool_call>\n<function=f>\n<parameter=a>\n1\n</parameter>\n</function>\n</tool_call>")

    def test_a_cut_call_is_left_open_for_the_parser(self):
        out = D.translate("<tool_calls:opensource><tool_call:opensource>get_weather<arg_key:opensource>city</arg_key:opensource><arg_value:opensource>Tok")
        self.assertTrue(out.startswith("<tool_call>\n<function=get_weather>\n<parameter=city>\nTok"))
        self.assertNotIn("</tool_call>", out)

    def test_the_forced_opening(self):
        self.assertEqual(D.force_text("<tool_call>\n<function=get_weather>\n"), "<tool_calls:opensource><tool_call:opensource>get_weather")
        self.assertEqual(D.force_text("<tool_call>\n"), "<tool_calls:opensource><tool_call:opensource>")
        self.assertIsNone(D.force_text(None))
        self.assertEqual(D.translate(D.force_text("<tool_call>\n<function=f>\n")), "<tool_call>\n<function=f>")


class Parse(unittest.TestCase):
    def test_a_call_is_an_event_with_typed_arguments(self):
        for n in (1, 4, 1000):
            evs = [e for e in events("Let me check." + CALL, n) if e.kind == "tool_call"]
            self.assertEqual(len(evs), 1, n)
            self.assertEqual((evs[0].call.name, evs[0].call.arguments), ("get_weather", {"city": "Tokyo", "days": 3}))

    def test_reasoning_then_answer(self):
        evs = events("need a plan</think:opensource>The answer.", 5, thinking=True, tools=None)
        self.assertEqual("".join(e.text for e in evs if e.kind == "reasoning"), "need a plan")
        self.assertEqual("".join(e.text for e in evs if e.kind == "content").strip(), "The answer.")

    def test_reasoning_then_a_call(self):
        evs = events("check the tool</think:opensource>" + CALL, 9, thinking=True)
        self.assertEqual([e.kind for e in evs if e.kind == "tool_call"], ["tool_call"])
        self.assertEqual("".join(e.text for e in evs if e.kind == "reasoning"), "check the tool")

    def test_the_parser_is_picked_by_the_template(self):
        self.assertIs(parser_for_template(TEMPLATE.read_text(encoding="utf-8")), Hy4OutputParser)
        self.assertIs(parser_for_template(QWEN_TEMPLATE.read_text(encoding="utf-8")), OutputParser)
        self.assertEqual(toolformat.tool_format(TEMPLATE.read_text(encoding="utf-8")), "hy4")


class RoundTrip(unittest.TestCase):
    def test_the_templates_own_call_is_read_back(self):
        tpl = ChatTemplate(TEMPLATE)
        msgs = [{"role": "user", "content": "weather?"},
                {"role": "assistant", "content": "", "tool_calls": [{"id": "c1", "function": {"name": "get_weather", "arguments": {"city": "Tokyo", "days": 3}}}]}]
        text = tpl.render(msgs, tools=TOOLS, add_generation_prompt=False)
        written = text[text.rindex("assistant<\uff5chy_middle:opensource\uff5c>") + len("assistant<\uff5chy_middle:opensource\uff5c>"):]
        written = written.replace("<\uff5chy_end:opensource\uff5c>", "")
        evs = events(written, 6, thinking=False)
        calls = [e.call for e in evs if e.kind == "tool_call"]
        self.assertEqual([(c.name, c.arguments) for c in calls], [("get_weather", {"city": "Tokyo", "days": 3})])


class Caps(unittest.TestCase):
    def test_the_real_template_reports_what_it_can_do(self):
        caps = ChatTemplate(TEMPLATE).caps
        self.assertEqual(caps, {"supports_tools": True, "supports_tool_calls": True, "supports_system_role": True,
                                "supports_parallel_tool_calls": True, "supports_preserve_reasoning": True})

    def test_the_qwen_template_is_unchanged(self):
        self.assertTrue(ChatTemplate(QWEN_TEMPLATE).caps["supports_tool_calls"])


class Effort(unittest.TestCase):
    def tearDown(self):
        frontend.use_family(None)

    def test_the_words_of_the_family(self):
        frontend.use_family("hy4")
        self.assertEqual(frontend.effort_kwargs("none"), {"enable_thinking": False})
        self.assertEqual(frontend.effort_kwargs("low"), {"reasoning_effort": "high"})
        self.assertEqual(frontend.effort_kwargs("high"), {"reasoning_effort": "high"})
        self.assertEqual(frontend.effort_kwargs("max"), {"reasoning_effort": "max"})
        self.assertEqual(frontend.effort_kwargs("xhigh"), {"reasoning_effort": "max"})
        frontend.use_family(None)
        self.assertEqual(frontend.effort_kwargs("high"), {"reasoning_effort": "xhigh"})       # the Qwen table is back

    def test_template_kwargs(self):
        self.assertEqual(D.template_kwargs({"enable_thinking": False}), {"enable_thinking": False, "reasoning_effort": "no_think"})
        self.assertEqual(D.template_kwargs({"reasoning_effort": "max"}), {"reasoning_effort": "max"})
        self.assertEqual(D.template_kwargs({"reasoning_effort": "medium"}), {"reasoning_effort": "high"})
        self.assertEqual(D.template_kwargs({"reasoning_effort": "no_think"}), {"reasoning_effort": "no_think", "enable_thinking": False})
        self.assertEqual(D.template_kwargs({}), {})

    def test_no_think_renders_the_closed_block(self):
        tpl = ChatTemplate(TEMPLATE)
        out = tpl.render([{"role": "user", "content": "x"}], **D.template_kwargs({"enable_thinking": False}))
        self.assertTrue(out.endswith("assistant<\uff5chy_middle:opensource\uff5c><think:opensource></think:opensource>"))

    def test_external_clients_cannot_crash_hy4_template_with_unsupported_efforts(self):
        tpl = ChatTemplate(TEMPLATE)
        frontend.use_family("hy4")
        # OpenAI / Responses / Olares / OpenCode / Codex / omp / pi levels:
        for raw in ("minimal", "min", "low", "medium", "med", "high", "auto", "default", "custom_unknown"):
            msgs, tools, kw = frontend.openai_to_messages(
                {"messages": [{"role": "user", "content": "hi"}], "reasoning_effort": raw}
            )
            rendered = tpl.render(msgs, tools=tools, **kw)
            self.assertIn("<\uff5creasoning_mode:opensource\uff5c>reasoning_effort:high<\uff5chy_end:opensource\uff5c>", rendered, raw)
            self.assertTrue(rendered.endswith("<think:opensource>"), raw)
        for raw in ("xhigh", "max", "maximum"):
            msgs, tools, kw = frontend.openai_to_messages(
                {"messages": [{"role": "user", "content": "hi"}], "reasoning": {"effort": raw}}
            )
            rendered = tpl.render(msgs, tools=tools, **kw)
            self.assertIn("<\uff5creasoning_mode:opensource\uff5c>reasoning_effort:max<\uff5chy_end:opensource\uff5c>", rendered, raw)
            self.assertTrue(rendered.endswith("<think:opensource>"), raw)
        for raw in ("none", "off", "no_think", "disabled", "false", False):
            msgs, tools, kw = frontend.openai_to_messages(
                {"messages": [{"role": "user", "content": "hi"}], "reasoning_effort": raw}
            )
            rendered = tpl.render(msgs, tools=tools, **kw)
            self.assertTrue(rendered.endswith("<think:opensource></think:opensource>"), raw)
        # chat_template_kwargs (vLLM / llama.cpp / Olares Router style):
        for ctk, expected in (
            ({"reasoning_effort": "low"}, "reasoning_effort:high<\uff5chy_end:opensource\uff5c>"),
            ({"reasoning_effort": "medium"}, "reasoning_effort:high<\uff5chy_end:opensource\uff5c>"),
            ({"thinking_mode": "low"}, "reasoning_effort:high<\uff5chy_end:opensource\uff5c>"),
            ({"thinking_mode": "xhigh"}, "reasoning_effort:max<\uff5chy_end:opensource\uff5c>"),
            ({"thinking_mode": "no_think"}, "<think:opensource></think:opensource>"),
        ):
            msgs, tools, kw = frontend.openai_to_messages(
                {"messages": [{"role": "user", "content": "hi"}], "chat_template_kwargs": ctk}
            )
            rendered = tpl.render(msgs, tools=tools, **kw)
            self.assertIn(expected, rendered, ctk)
        # Anthropic Messages (Claude Code): effort + budget_tokens (<2K, <8K, >=8K) + disabled
        for budget in (512, 1024, 4096, 16384):
            msgs, tools, kw = frontend.anthropic_to_messages(
                {"messages": [{"role": "user", "content": "hi"}], "thinking": {"type": "enabled", "budget_tokens": budget}}
            )
            rendered = tpl.render(msgs, tools=tools, **kw)
            self.assertIn("reasoning_effort:high<\uff5chy_end:opensource\uff5c>", rendered, budget)
        for eff, mode in (("low", "high"), ("medium", "high"), ("minimal", "high"), ("high", "high"), ("max", "max")):
            msgs, tools, kw = frontend.anthropic_to_messages(
                {"messages": [{"role": "user", "content": "hi"}], "output_config": {"effort": eff}}
            )
            rendered = tpl.render(msgs, tools=tools, **kw)
            self.assertIn(f"reasoning_effort:{mode}<\uff5chy_end:opensource\uff5c>", rendered, eff)
        # Direct ChatTemplate.render defensive clamp (e.g. resolve_prefix or custom caller passing raw Qwen level):
        for raw_kw, mode in (
            ({"reasoning_effort": "low"}, "reasoning_effort:high<\uff5chy_end:opensource\uff5c>"),
            ({"reasoning_effort": "medium"}, "reasoning_effort:high<\uff5chy_end:opensource\uff5c>"),
            ({"reasoning_effort": "minimal"}, "reasoning_effort:high<\uff5chy_end:opensource\uff5c>"),
            ({"reasoning_effort": "xhigh"}, "reasoning_effort:max<\uff5chy_end:opensource\uff5c>"),
            ({"thinking_mode": "low"}, "reasoning_effort:high<\uff5chy_end:opensource\uff5c>"),
            ({"reasoning_effort": "no_think"}, "<think:opensource></think:opensource>"),
        ):
            rendered = tpl.render([{"role": "user", "content": "hi"}], **raw_kw)
            self.assertIn(mode, rendered, raw_kw)


class Kind(unittest.TestCase):
    def test_is_hy4(self):
        self.assertTrue(hy4_kind.is_hy4({"arch": "hy4flash"}))
        self.assertTrue(hy4_kind.is_hy4({"exe": "component:strata-hy4"}))
        self.assertTrue(hy4_kind.is_hy4({"exe": r"C:\x\strata-hy4.exe"}))
        self.assertFalse(hy4_kind.is_hy4({"exe": "component:strata-engine", "arch": "qwen4exp"}))
        self.assertFalse(hy4_kind.is_hy4(None))

    BASE = ["--native", "m.strata", "--engram", "e.strata", "--max-context", "131072"]

    def test_args_are_the_configs_own_when_the_engine_has_neither(self):
        cfg = {"arch": "hy4flash", "args": self.BASE}
        with mock.patch.object(hy4_kind, "probe", return_value=None):
            self.assertEqual(hy4_kind.engine_args(cfg, "x", announce=False), self.BASE)             # drafter asked (the default), the engine cannot say it has one
        with mock.patch.object(hy4_kind, "probe", return_value={"arch": "hy4flash", "drafter": "none", "images": "0"}):
            self.assertEqual(hy4_kind.engine_args({**cfg, "args": self.BASE + ["--vision", "v.strata"]}, "x", announce=False), self.BASE)   # no image reader: --vision <file> left out

    def test_the_drafter_and_the_pictures_are_on_by_default_where_the_engine_has_them(self):
        cfg = {"arch": "hy4flash", "args": self.BASE + ["--vision", "v.strata"]}
        have = {"arch": "hy4flash", "drafter": "dfly2", "images": "1"}
        with mock.patch.object(hy4_kind, "probe", return_value=have):
            self.assertEqual(hy4_kind.engine_args(cfg, "x", announce=False), self.BASE + ["--vision", "v.strata", "--draft", "dfly2"])
            self.assertEqual(hy4_kind.engine_args({**cfg, "drafting": "off"}, "x", announce=False), self.BASE + ["--vision", "v.strata"])
            for word in ("auto", "mtp", "dflash2", "dfly2"):                                          # Qwen's words mean "the default" here
                self.assertIn("--draft", hy4_kind.engine_args({**cfg, "drafting": word}, "x", announce=False), word)
            kept = {**cfg, "args": cfg["args"] + ["--draft", "dfly2"]}
            self.assertEqual(hy4_kind.engine_args(kept, "x", announce=False).count("--draft"), 1)

    def test_one_line_says_what_was_left_out(self):
        cfg = {"arch": "hy4flash", "args": self.BASE + ["--vision", "v.strata"]}
        with mock.patch.object(hy4_kind, "probe", return_value=None), mock.patch("builtins.print") as say:
            hy4_kind.engine_args(cfg, "x")
        said = " ".join(str(c.args[0]) for c in say.call_args_list)
        self.assertEqual(said.count(hy4_kind.NO_DRAFTER_NOTE), 1)
        self.assertEqual(said.count(hy4_kind.NO_IMAGES_NOTE), 1)

    def test_info_line_and_paths(self):
        self.assertEqual(hy4_kind.parse_info("noise\narch=hy4flash layers=42 drafter=dfly2 images=1\n"), {"arch": "hy4flash", "layers": "42", "drafter": "dfly2", "images": "1"})
        self.assertEqual(hy4_kind.parse_info("usage: strata-hy4"), {})
        p = "C:/Users/A B/100%/x.png"
        self.assertEqual(hy4_kind.decode_path(hy4_kind.encode_path(p)), p)
        self.assertNotIn(" ", hy4_kind.encode_path(p))

    def test_the_probe_says_none_for_a_program_that_cannot_answer(self):
        self.assertIsNone(hy4_kind.probe("does-not-exist", ["--native", "m.strata"]))
        self.assertIsNone(hy4_kind.probe(None, ["--native", "m.strata"]))

    def test_write_images_isolates_overlapping_requests_on_stop_or_retry(self):
        """Regression for real-human Stop + follow-up in a multi-turn image chat:
        Request A's cleanup must NEVER unlink Request B's temp file for the same image."""
        import tempfile
        png = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR"
        with tempfile.TemporaryDirectory() as td:
            req_a = hy4_kind.write_images([png, png], lambda s: s, lambda b: b, folder=td)
            req_b = hy4_kind.write_images([png], lambda s: s, lambda b: b, folder=td)
            self.assertEqual(len(set(req_a + req_b)), 3)
            # Simulate Request A finishing cancel/drop_embeddings AFTER Request B ran prepare():
            for p in req_a:
                Path(p).unlink(missing_ok=True)
            self.assertTrue(Path(req_b[0]).is_file(), "Request B's image file must survive Request A's cleanup")
            self.assertEqual(Path(req_b[0]).read_bytes(), png)
            Path(req_b[0]).unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
