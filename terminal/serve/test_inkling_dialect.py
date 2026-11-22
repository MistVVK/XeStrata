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
        self.assertEqual(built[built.index("--gpus") + 1], "2")


if __name__ == "__main__":
    unittest.main()
