"""Unit tests for serve/inkling_dialect.py, serve/inkling_kind.py, and frontend Inkling-2-Mini integration."""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

TERMINAL = Path(__file__).resolve().parents[1]
if str(TERMINAL) not in sys.path:
    sys.path.insert(0, str(TERMINAL))

from serve import frontend, inkling_dialect, inkling_kind, toolformat


class TestInklingDialect(unittest.TestCase):
    def test_streaming_thinking_discarded_and_text_byte_by_byte(self):
        raw = (
            "<|message_model|><|content_thinking|>Let me check 2 + 2 = 4.<|end_message|>"
            "<|message_model|><|content_text|>The answer is 4.<|end_message|>"
            "<|content_model_end_sampling|>"
        )
        # Feed 1 character at a time: raw <|content_thinking|> is silently discarded (no thinking mode), text streams cleanly
        d = inkling_dialect.InklingDialect()
        out = "".join(d.feed(ch) for ch in raw) + d.feed(final=True)
        self.assertEqual(out, "The answer is 4.")
        self.assertEqual(d.calls, 0)

    def test_tool_call_preserves_xml_chars_quotes_and_closing_tags(self):
        code = (
            'def check(a, b):\n'
            '    xml = "<tag attr=\\"x\\">a < b && c > d </parameter> </tool_call>"\n'
            "    return xml + 'single'\n"
        )
        payload = json.dumps({"name": "write_file", "args": {"path": "src/a&b.py", "content": code, "overwrite": True, "lines": [1, 2, 3]}})
        raw = (
            "<|message_model|><|content_thinking|>Writing patch with < and > and </tool_call>.<|end_message|>"
            f"<|message_model|>write_file<|content_invoke_tool_json|>{payload}<|end_message|>"
            "<|content_model_end_sampling|>"
        )
        # Feed raw Inkling-2-Mini tokens in awkward 3-char chunks directly into InklingOutputParser
        p = frontend.InklingOutputParser(tools=[{"name": "write_file", "parameters": {"type": "object"}}])
        reasoning_parts, content_parts, tool_calls = [], [], []
        for i in range(0, len(raw), 3):
            for ev in p.feed(raw[i:i + 3]):
                if ev.kind == "reasoning":
                    reasoning_parts.append(ev.text)
                elif ev.kind == "content":
                    content_parts.append(ev.text)
                elif ev.kind == "tool_call" and ev.call is not None:
                    tool_calls.append({"name": ev.call.name, "arguments": ev.call.arguments})
        for ev in p.finish():
            if ev.kind == "reasoning":
                reasoning_parts.append(ev.text)
            elif ev.kind == "content":
                content_parts.append(ev.text)
            elif ev.kind == "tool_call" and ev.call is not None:
                tool_calls.append({"name": ev.call.name, "arguments": ev.call.arguments})

        self.assertEqual("".join(reasoning_parts), "")
        self.assertEqual("".join(content_parts), "")
        self.assertEqual(len(tool_calls), 1)
        self.assertEqual(tool_calls[0]["name"], "write_file")
        self.assertEqual(tool_calls[0]["arguments"]["path"], "src/a&b.py")
        self.assertEqual(tool_calls[0]["arguments"]["content"], code)
        self.assertIs(tool_calls[0]["arguments"]["overwrite"], True)
        self.assertEqual(tool_calls[0]["arguments"]["lines"], [1, 2, 3])

    def test_multiple_parallel_tool_calls_in_one_turn(self):
        raw = (
            '<|message_model|>read_file<|content_invoke_tool_json|>{"name":"read_file","args":{"path":"a.py"}}<|end_message|>'
            '<|message_model|>read_file<|content_invoke_tool_json|>{"name":"read_file","args":{"path":"b.py"}}<|end_message|>'
            '<|content_model_end_sampling|>'
        )
        p = frontend.InklingOutputParser(tools=[{"name": "read_file", "parameters": {"type": "object"}}])
        events = p.feed(raw) + p.finish()
        calls = [ev.call for ev in events if ev.kind == "tool_call" and ev.call is not None]
        self.assertEqual([c.arguments["path"] for c in calls], ["a.py", "b.py"])

    def test_no_thinking_mode_always_off(self):
        # Inkling-2-Mini has no thinking mode: default is off ("none") and every input normalizes to "none"
        for level in (None, "", "none", "off", "minimal", "low", "medium", "high", "max", "xhigh", 0.0, 0.9):
            self.assertEqual(inkling_dialect.normalize_effort(level), "none")
        self.assertEqual(inkling_dialect.template_kwargs({}), {"enable_thinking": False, "reasoning_effort": "none"})
        for level in ("none", "minimal", "low", "medium", "high", "max"):
            got = inkling_dialect.template_kwargs({"reasoning_effort": level})
            self.assertEqual(got, {"enable_thinking": False, "reasoning_effort": "none"})
        self.assertEqual(inkling_dialect.template_kwargs({"enable_thinking": True, "thinking_mode": "high"}), {"enable_thinking": False, "reasoning_effort": "none"})

    def test_chat_template_empty_kwargs_forces_effort_zero_and_caps_no_reasoning(self):
        import tempfile
        tpl_src = (
            "{%- set effort_map = {'none': 0.0, 'high': 0.9} -%}"
            "{%- set eff = reasoning_effort if reasoning_effort is defined and reasoning_effort is not none else 0.9 -%}"
            "{%- set num = effort_map[eff] if eff is string else eff -%}"
            "<|message_system|><|content_text|>Thinking effort level: {% if num == 0.0 %}0{% else %}{{ num }}{% endif %}<|end_message|>"
            "{%- if tools -%}<|message_system|>tool_declare<|content_xml|>{{ tools | tojson }}<|end_message|>{%- endif -%}"
            "{%- for m in messages -%}"
            "{%- if m.role == 'system' -%}<|message_system|><|content_text|>{{ m.content }}<|end_message|>"
            "{%- elif m.role == 'user' -%}<|message_user|><|content_text|>{{ m.content }}<|end_message|>"
            "{%- elif m.role == 'assistant' -%}"
            "{%- if m.tool_calls -%}{%- for tc in m.tool_calls -%}<|message_model|>{{ tc.function.name }}<|content_invoke_tool_json|>{{ tc.function.arguments | tojson }}<|end_message|>{%- endfor -%}{%- endif -%}"
            "{%- if m.content -%}<|message_model|><|content_text|>{{ m.content }}<|end_message|>{%- endif -%}"
            "{%- elif m.role == 'tool' -%}<|message_tool|><|content_text|>{{ m.content }}<|end_message|>{%- endif -%}"
            "{%- endfor -%}"
        )
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "chat_template.jinja"
            p.write_text(tpl_src, encoding="utf-8")
            tpl = frontend.ChatTemplate(p)
            rendered = tpl.render([{"role": "user", "content": "hi"}])
            self.assertIn("Thinking effort level: 0<|end_message|>", rendered)
            self.assertFalse(tpl.caps["supports_preserve_reasoning"])
            self.assertTrue(tpl.caps["supports_tools"])

    def test_prepare_messages_converts_json_string_tool_args_to_dicts(self):
        msgs = [
            {"role": "user", "content": "check file"},
            {
                "role": "assistant",
                "content": None,
                "reasoning_content": "plan",
                "tool_calls": [
                    {"id": "call_1", "type": "function", "function": {"name": "read_file", "arguments": '{"path": "a.py"}'}}
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "name": "read_file", "content": "ok"},
        ]
        out = inkling_dialect.prepare_messages(msgs)
        self.assertEqual(out[1]["tool_calls"][0]["function"]["arguments"], {"path": "a.py"})
        # Original input is not mutated
        self.assertIsInstance(msgs[1]["tool_calls"][0]["function"]["arguments"], str)

    def test_toolformat_detects_inkling2(self):
        sample = "<|message_system|>tool_declare<|content_xml|>...<|content_invoke_tool_json|>"
        self.assertEqual(toolformat.tool_format(sample), "inkling2")

    def test_inkling_kind_detection_and_arg_validation(self):
        self.assertTrue(inkling_kind.is_inkling({"arch": "inkling2_loop"}))
        self.assertTrue(inkling_kind.is_inkling({"exe": "/opt/strata/strata-inkling"}))
        self.assertTrue(inkling_kind.is_inkling({"exe": r"C:\strata\strata-inkling.exe"}))
        self.assertFalse(inkling_kind.is_inkling({"exe": "/opt/strata/strata-hy4"}))

        info = inkling_kind.parse_info("arch=inkling2_loop layers=24 visits=48 eos=200006 image_pad=200054 parallel=64")
        self.assertEqual(info["arch"], "inkling2_loop")
        self.assertEqual(int(info["image_pad"]), 200054)
        self.assertEqual(int(info["parallel"]), 64)

        class _FakeEngine:
            inkling_engine = True

        self.assertIsNone(inkling_kind.refuse(_FakeEngine(), {"parallel": 8}))
        with self.assertRaises(ValueError):
            inkling_kind.refuse(_FakeEngine(), {"logprobs": True})

        built = inkling_kind.engine_args(
            {"args": ["--native", "Inkling-2-Mini-SQ-20.strata", "--max-context", "32768", "--vision", "Inkling-2-Mini-Vision.strata"], "parallel": 32, "gpu": [0, 1]},
            announce=False,
        )
        self.assertIn("--parallel", built)
        self.assertEqual(built[built.index("--parallel") + 1], "32")
        self.assertIn("--gpus", built)
        self.assertEqual(built[built.index("--gpus") + 1], "1,0")
        self.assertIn("--layer-split", built)
        self.assertEqual(built[built.index("--layer-split") + 1], "auto")

    def test_multigpu_main_first_ordinals_and_stale_vram_strip_6631(self):
        base = ["--native", "Inkling-2-Mini-SQ-15.strata", "--max-context", "32768"]
        # 5080 (GPU 0) = Main, 5070 Ti (GPU 1) = Helper -> child_env sets CUDA_VISIBLE_DEVICES=1,0 ->
        # local ordinal 1 is Main (5080) and local 0 is Helper (5070 Ti), so --gpus MUST be 1,0!
        cfg_roles = {
            "arch": "inkling2_loop",
            "backend": "cuda",
            "gpu_roles": {"0": "main", "1": "helper"},
            "args": base + ["--vram-experts-gb", "10.50", "--ram-experts-gb", "1.78"],
        }
        got = inkling_kind.engine_args(cfg_roles, announce=False)
        self.assertEqual(got[got.index("--gpus") + 1], "1,0")
        self.assertEqual(got[got.index("--layer-split") + 1], "auto")
        self.assertNotIn("--vram-experts-gb", got)
        self.assertNotIn("--ram-experts-gb", got)

        # Bare count --gpus 2 written by setup is rewritten to Main-first 1,0 unless explicitly in user_args
        cfg_setup = {
            "arch": "inkling2_loop",
            "backend": "cuda",
            "gpu_roles": {"0": "main", "1": "helper"},
            "args": base + ["--gpus", "2"],
        }
        got_setup = inkling_kind.engine_args(cfg_setup, announce=False)
        self.assertEqual(got_setup[got_setup.index("--gpus") + 1], "1,0")

        # Explicit user_args --gpus / --vram-experts-gb are respected verbatim
        cfg_user = {
            "arch": "inkling2_loop",
            "backend": "cuda",
            "gpu": [0, 1],
            "args": list(base),
            "user_args": ["--gpus", "0,1", "--vram-experts-gb", "9.00"],
        }
        got_user = inkling_kind.engine_args(cfg_user, announce=False)
        self.assertEqual(got_user[got_user.index("--gpus") + 1], "0,1")
        self.assertIn("--vram-experts-gb", got_user)

        # 3 & 4 CUDA GPUs emit Main-first local ordinals + --layer-split auto; HIP/SYCL cap at 2 GPUs (Main + last Helper)
        for gpu_sel, bkd, want in (
            ([0, 1, 2], "cuda", "2,0,1"),
            ([0, 1, 2, 3], "cuda", "3,0,1,2"),
            ([0, 1], "hip", "1,0"),
            ([0, 1, 2], "hip", "2,1"),
            ([0, 1], "sycl", "1,0"),
        ):
            c = {"arch": "inkling2_loop", "backend": bkd, "gpu": gpu_sel, "args": list(base)}
            a = inkling_kind.engine_args(c, announce=False)
            self.assertEqual(a[a.index("--gpus") + 1], want, (gpu_sel, bkd, a))
            if bkd == "cuda":
                self.assertEqual(a[a.index("--layer-split") + 1], "auto")
            else:
                self.assertNotIn("--layer-split", a)

        # Explicit --ep-split keeps EP2 without injecting --layer-split auto
        cfg_ep = {"arch": "inkling2_loop", "backend": "cuda", "gpu": [0, 1], "args": base + ["--ep-split", "0.5"]}
        got_ep = inkling_kind.engine_args(cfg_ep, announce=False)
        self.assertEqual(got_ep[got_ep.index("--gpus") + 1], "1,0")
        self.assertNotIn("--layer-split", got_ep)

    def test_dspark_drafting_off_flag_and_drafting_info_6633(self):
        base = ["--native", "Inkling-2-Mini-SQ-20.strata", "--max-context", "32768"]
        # drafting: "off" MUST emit --draft off so strata-inkling does not default to auto/dspark
        got_off = inkling_kind.engine_args({"arch": "inkling2_loop", "drafting": "off", "args": list(base)}, announce=False)
        self.assertIn("--draft", got_off)
        self.assertEqual(got_off[got_off.index("--draft") + 1], "off")
        self.assertEqual(inkling_kind.drafting_info(got_off, {"drafter": "dspark"}), {"drafting": "off", "drafting_note": ""})

        # --draft dspark or INFO drafter=dspark reports drafting: "dspark"
        got_on = base + ["--draft", "dspark"]
        self.assertEqual(inkling_kind.drafting_info(got_on, {"drafter": "dspark"}), {"drafting": "dspark", "drafting_note": ""})
        self.assertEqual(inkling_kind.drafting_info(base, {"drafter": "dspark"}), {"drafting": "dspark", "drafting_note": ""})
        self.assertEqual(inkling_kind.drafting_info(base, {"drafter": "none"}), {"drafting": "off", "drafting_note": ""})

    def test_finish_prompt_skips_hidden_thinking_for_plain_and_tool_turns_6612_6627(self):
        base_prompt = (
            "<|message_system|><|content_text|>Thinking effort level: 0<|end_message|>"
            "<|message_user|><|content_text|>Say hello in five words.<|end_message|>"
        )
        plain = inkling_dialect.finish_prompt(base_prompt, has_tools=False)
        self.assertTrue(plain.endswith("<|message_model|><|content_text|>"))
        # Idempotent
        self.assertEqual(inkling_dialect.finish_prompt(plain, has_tools=False), plain)

        # With tools: closes an empty thinking turn so model immediately chooses text or tool invocation
        with_tools = inkling_dialect.finish_prompt(base_prompt, has_tools=True)
        self.assertTrue(with_tools.endswith("<|message_model|><|content_thinking|><|end_message|>"))
        self.assertEqual(inkling_dialect.finish_prompt(with_tools, has_tools=True), with_tools)

        # Already ending with <|message_model|>
        self.assertEqual(
            inkling_dialect.finish_prompt(base_prompt + "<|message_model|>", has_tools=False),
            base_prompt + "<|message_model|><|content_text|>",
        )
        self.assertEqual(
            inkling_dialect.finish_prompt(base_prompt + "<|message_model|>", has_tools=True),
            base_prompt + "<|message_model|><|content_thinking|><|end_message|>",
        )
        # Forced tool call leaves prompt untouched
        self.assertEqual(inkling_dialect.finish_prompt(base_prompt, has_tools=True, forced=True), base_prompt)

    def test_dialect_hardening_prevents_empty_replies_on_missing_tags_and_yaml_6620(self):
        # 1. Model emits <|message_model|> followed directly by YAML/prose (omitting <|content_text|>):
        # header state flushes on newline / close instead of discarding the entire reply!
        yaml_reply = (
            "<|message_model|>Her er problemerne i din docker-compose-fil:\n"
            "```yaml\nservices:\n  db:\n    image: postgres:16\n```\n"
            "<|end_message|><|content_model_end_sampling|>"
        )
        p = frontend.InklingOutputParser(tools=[
            {"name": "search_web", "parameters": {"type": "object"}},
            {"name": "fetch_article", "parameters": {"type": "object"}},
        ])
        evs = []
        for i in range(0, len(yaml_reply), 4):
            evs.extend(p.feed(yaml_reply[i:i + 4]))
        evs.extend(p.finish())
        content = "".join(ev.text for ev in evs if ev.kind == "content")
        self.assertIn("postgres:16", content)
        self.assertIn("Her er problemerne", content)

        # 2. Short single-line reply with <|message_model|>...<|end_message|> (no <|content_text|>) still flushes
        short_missing_tag = "<|message_model|>Hello world!<|end_message|>"
        self.assertEqual(inkling_dialect.translate(short_missing_tag), "Hello world!")

        # 3. Direct <|content_text|> transition out of <|content_thinking|> (omitting <|end_message|><|message_model|>)
        direct_exit = (
            "<|message_model|><|content_thinking|>internal plan"
            "<|content_text|>Visible YAML answer.<|end_message|>"
        )
        self.assertEqual(inkling_dialect.translate(direct_exit), "Visible YAML answer.")

        # 4. Truncated non-JSON inside <|content_invoke_tool_json|> on final=True releases text instead of bogus raw call
        truncated_prose = "<|message_model|>search_web<|content_invoke_tool_json|>services:\n  db:\n    image: postgres:16"
        d = inkling_dialect.InklingDialect()
        out = d.feed(truncated_prose, final=True)
        self.assertIn("postgres:16", out)
        self.assertNotIn("<tool_call>", out)

        # 5. Model continuing directly from finish_prompt's <|message_model|><|content_text|> prefill
        prefilled_out = "Red<|end_message|><|content_model_end_sampling|>"
        self.assertEqual(inkling_dialect.translate(prefilled_out), "Red")


if __name__ == "__main__":
    unittest.main()
