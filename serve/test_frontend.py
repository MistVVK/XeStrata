# SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
"""Chat history normalization before rendering the model's prompt."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from serve.frontend import (HARMONY_NO_THINKING, HARMONY_TURN, ChatTemplate, anthropic_to_messages, effort_kwargs,
                            legacy_functions, openai_to_messages, require_messages)
from serve.server import ByteTokenizer, MockEngine, Service


class EmptyAssistantHistory(unittest.TestCase):
    def setUp(self):
        self.template = ChatTemplate(Path(__file__).with_name("test_chat_template.jinja"))
        tok = ByteTokenizer()
        self.service = Service(MockEngine(tok, "ok"), tok, self.template)

    def test_openai_empty_replies_are_not_rendered(self):
        for content in (None, "", " \n\t", [], [{"type": "text", "text": ""}]):
            for preserve in (True, False):
                with self.subTest(content=content, preserve_thinking=preserve):
                    history = []
                    for _ in range(3):
                        history.extend([{"role": "user", "content": "haz un ls"},
                                        {"role": "assistant", "content": content,
                                         "reasoning_content": "Let me run ls."}])
                    history.append({"role": "user", "content": "haz un ls"})
                    messages, tools, kwargs = openai_to_messages({"messages": history})
                    kwargs["preserve_thinking"] = preserve
                    clean = [m for m in messages if m["role"] == "user"]
                    self.assertEqual(self.service.encode_prompt(messages, tools, kwargs),
                                     self.service.encode_prompt(clean, tools, kwargs))

    def test_anthropic_thinking_only_replies_are_not_rendered(self):
        messages, tools, kwargs = anthropic_to_messages({"messages": [
            {"role": "user", "content": "haz un ls"},
            {"role": "assistant", "content": [{"type": "thinking", "thinking": "Let me run ls."}]},
            {"role": "user", "content": "haz un ls"},
        ]})
        self.assertEqual(self.service.encode_prompt(messages, tools, kwargs),
                         self.service.encode_prompt([messages[0], messages[2]], tools, kwargs))

    def test_text_and_tool_calls_are_kept(self):
        messages, tools, kwargs = openai_to_messages({"messages": [
            {"role": "user", "content": "haz un ls"},
            {"role": "assistant", "content": [{"type": "text", "text": "Running ls."}]},
            {"role": "assistant", "content": None, "tool_calls": [
                {"type": "function", "function": {"name": "run", "arguments": '{"command":"ls"}'}},
            ]},
            {"role": "tool", "content": "file.txt"},
            {"role": "user", "content": "thanks"},
        ]})
        before = copy.deepcopy(messages)
        prompt = self.template.render(messages, tools=tools, **kwargs)
        self.assertIn("Running ls.", prompt)
        self.assertIn("<function=run>", prompt)
        self.assertIn("file.txt", prompt)
        self.assertEqual(messages, before)

    def test_image_only_assistant_turn_is_kept(self):
        messages, tools, kwargs = openai_to_messages({"messages": [
            {"role": "user", "content": "describe the image"},
            {"role": "assistant", "content": [{"type": "image_url", "image_url": "image.png"}]},
            {"role": "user", "content": "thanks"},
        ]})
        self.assertIn("<|vision_start|><|image_pad|><|vision_end|>",
                      self.template.render(messages, tools=tools, **kwargs))

    def test_final_empty_assistant_turn_is_kept(self):
        prompt = self.template.render([{"role": "user", "content": "hi"},
                                       {"role": "assistant", "content": ""}], add_generation_prompt=False)
        self.assertIn("<|im_start|>assistant\n", prompt)

    def test_empty_user_and_tool_turns_are_kept(self):
        prompt = self.template.render([{"role": "user", "content": ""},
                                       {"role": "tool", "content": ""},
                                       {"role": "user", "content": "hi"}])
        self.assertEqual(prompt.count("<|im_start|>user\n"), 3)
        self.assertIn("<tool_response>\n\n</tool_response>", prompt)


class UnreadContentParts(unittest.TestCase):
    """A content part the server does not read is refused naming it and where it sits, as /v1/responses does; chat
    and messages used to drop it without a word, and the model answered about a file it had never seen."""

    def setUp(self):
        self.template = ChatTemplate(Path(__file__).with_name("test_chat_template.jinja"))
        tok = ByteTokenizer()
        self.service = Service(MockEngine(tok, "ok"), tok, self.template)

    def test_chat_file_part_is_kept_and_audio_refused(self):
        audio = {"type": "input_audio", "input_audio": {"data": "AA", "format": "wav"}}
        with self.assertRaisesRegex(ValueError, "input_audio"):   # the model reads no audio: a 400, as OpenAI's
            openai_to_messages({"messages": [{"role": "user", "content": [{"type": "text", "text": "this"}, audio]}]})
        # a file stays in the content as a "file" item, which Service.expand_files reads into text and pages
        part = {"type": "file", "file": {"filename": "a.pdf", "file_data": "data:application/pdf;base64,AA"}}
        got = openai_to_messages({"messages": [{"role": "user", "content": [{"type": "text", "text": "this"}, part]}]})
        self.assertEqual(got[0][0]["content"][1], {"type": "file", "data": "data:application/pdf;base64,AA",
                                                   "url": None, "name": "a.pdf", "mime": "application/pdf"})
        with self.assertRaisesRegex(ValueError, "file_id"):       # no Files API here
            only_id = {"type": "file", "file": {"file_id": "f"}}
            openai_to_messages({"messages": [{"role": "user", "content": [only_id]}]})

    def test_messages_blocks_real_clients_send_are_left_out(self):
        # tool_reference (Claude Code's tool search), server tool blocks, signed thinking (documents and search
        # results are read: _anthropic_documents)
        extra = [{"type": "server_tool_use", "id": "s1", "name": "web_search", "input": {}},
                 {"type": "web_search_tool_result", "tool_use_id": "s1", "content": []},
                 {"type": "container_upload", "file_id": "f"}]
        for block in extra:
            with self.subTest(kind=block["type"]):
                got = anthropic_to_messages({"messages": [
                    {"role": "user", "content": "hi"},
                    {"role": "assistant", "content": [{"type": "text", "text": "ok"}, block]},
                    {"role": "user", "content": [{"type": "text", "text": "go"}, block]}]})
                want = anthropic_to_messages({"messages": [
                    {"role": "user", "content": "hi"},
                    {"role": "assistant", "content": [{"type": "text", "text": "ok"}]},
                    {"role": "user", "content": [{"type": "text", "text": "go"}]}]})
                self.assertEqual(got, want)

    def test_parameter_types_that_allow_a_string(self):
        # a parameter whose schema allows a string keeps what the model wrote when reading it as JSON gives a type the
        # schema does not allow: ["string", "null"]'s and anyOf's "1" stays "1", as do a string enum's and const's
        from serve.frontend import OutputParser, parse_tool_call
        body = "<function=f>\n<parameter=x>\n1\n</parameter>\n</function>"
        nullable = {"anyOf": [{"type": "string"}, {"type": "null"}]}
        for schema, want in (({"type": ["string", "null"]}, "1"), (nullable, "1"), ({"enum": ["1", "2"]}, "1"),
                             ({"const": "1"}, "1"), ({"type": "string"}, "1"), ({"type": ["integer", "null"]}, 1),
                             ({"type": ["string", "integer"]}, 1), (True, 1)):
            tool = {"name": "f", "parameters": {"type": "object", "properties": {"x": schema}}}
            with self.subTest(schema=schema):
                self.assertEqual(parse_tool_call(body, tool).arguments, {"x": want})
                p = OutputParser(thinking=False, tools=[tool], stream_tools=True)
                evs = p.feed("<tool_call>\n" + body + "\n</tool_call>") + p.finish()
                self.assertEqual([e.call.arguments for e in evs if e.kind == "tool_call"], [{"x": want}])
                streamed = "".join(e.text for e in evs if e.kind == "tool_args")    # the arguments as they stream
                self.assertEqual(json.loads(streamed), {"x": want})

    def test_responses_results_follow_calls_split_by_reasoning(self):
        # Responses: a reasoning or a message between two calls makes two assistant messages; the results, sent in the
        # order they finished, still follow the calls (the template pairs them by position)
        from serve.responses import input_messages
        call = [{"type": "function_call", "call_id": f"c{n}", "name": n, "arguments": "{}"} for n in "ab"]
        between = ({"type": "reasoning", "summary": [], "content": [{"type": "reasoning_text", "text": "then b"}]},
                   {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "and b"}]})
        results = [{"type": "function_call_output", "call_id": f"c{n}", "output": f"RESULT-{n}"} for n in "ba"]
        for item in between:
            with self.subTest(item["type"]):
                got = input_messages({"input": [{"role": "user", "content": "go"}, call[0], item, call[1], *results]})
                self.assertEqual([m["content"] for m in got if m["role"] == "tool"], ["RESULT-a", "RESULT-b"])

    def test_function_call_none_or_null_without_functions(self):
        # a null is no field, and function_call "none" without functions is OpenAI's default: an ordinary request
        for req in ({"function_call": "none"}, {"functions": None, "function_call": None}, {"functions": None}):
            self.assertIsNone(legacy_functions({"messages": [], **req}), req)
        with self.assertRaisesRegex(ValueError, "functions must be"):
            legacy_functions({"function_call": "auto"})
        # "none" beside the new tools is the old and the new shape mixed: a 400, as any other mix
        tool = {"type": "function", "function": {"name": "f", "parameters": {"type": "object"}}}
        for req in ({"function_call": "none", "tools": [tool]}, {"function_call": "none", "tool_choice": "auto"}):
            with self.assertRaisesRegex(ValueError, "cannot be used with tools"):
                legacy_functions(req)
        self.assertIsNone(legacy_functions({"functions": None, "function_call": None, "tools": [tool]}))

    def test_messages_document_images_are_read(self):
        # a "content" document's images reach the model as the message's own images do, after its text
        img = {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "AA"}}
        doc = {"type": "document", "title": "T",
               "source": {"type": "content", "content": [img, {"type": "text", "text": "body"}]}}
        got = anthropic_to_messages({"messages": [{"role": "user", "content": [doc]}]})[0]
        self.assertEqual(got[0]["content"], [{"type": "text", "text": "Document: T\n\nbody"},
                                             {"type": "image", "source": "data:image/png;base64,AA"}])

    def test_messages_tool_reference_in_a_tool_result_is_left_out(self):
        def run(extra):
            return anthropic_to_messages({"messages": [
                {"role": "user", "content": "find a tool"},
                {"role": "assistant", "content": [{"type": "tool_use", "id": "t1", "name": "ToolSearch", "input": {}}]},
                {"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": "t1",
                     "content": [{"type": "text", "text": "found"}] + extra}]}]})
        self.assertEqual(run([{"type": "tool_reference", "tool_name": "mcp__x"}]), run([]))

    def test_a_system_prompt_with_an_image_part_is_read_as_text_as_before(self):
        got = openai_to_messages({"messages": [
            {"role": "system", "content": [{"type": "text", "text": "be brief"},
                                           {"type": "image_url", "image_url": "image.png"}]},
            {"role": "user", "content": "hi"}]})
        self.assertEqual(got[0][0]["content"], "be brief")

    def test_messages_redacted_thinking_is_dropped(self):
        msgs = {"messages": [
            {"role": "user", "content": "haz un ls"},
            {"role": "assistant", "content": [{"type": "redacted_thinking", "data": "ZW50cnk="},
                                              {"type": "text", "text": "Running ls."}]},
            {"role": "user", "content": "thanks"}]}
        without = copy.deepcopy(msgs)
        del without["messages"][1]["content"][0]
        with_block = anthropic_to_messages(msgs)
        without_block = anthropic_to_messages(without)
        self.assertEqual(with_block[0], without_block[0])
        self.assertEqual(self.service.encode_prompt(with_block[0], with_block[1], with_block[2]),
                         self.service.encode_prompt(without_block[0], without_block[1], without_block[2]))

    def test_chat_refusal_part_is_read_as_its_text(self):
        messages, tools, kwargs = openai_to_messages({"messages": [
            {"role": "user", "content": "do a thing"},
            {"role": "assistant", "content": [{"type": "refusal", "refusal": "I cannot do that."}]},
            {"role": "user", "content": "ok"}]})
        self.assertEqual(messages[1]["content"], "I cannot do that.")
        self.assertIn("I cannot do that.", self.template.render(messages, tools=tools, **kwargs))

    def test_text_and_image_requests_render_as_before(self):
        parts = [{"role": "user", "content": [{"type": "text", "text": "haz un ls"}]},
                 {"role": "assistant", "content": [{"type": "text", "text": "file.txt"}]}]
        plain = [{"role": "user", "content": "haz un ls"}, {"role": "assistant", "content": "file.txt"}]
        for convert in (openai_to_messages, anthropic_to_messages):
            with self.subTest(convert=convert.__name__):
                self.assertEqual(self.service.encode_prompt(*convert({"messages": parts})),
                                 self.service.encode_prompt(*convert({"messages": plain})))
        chat = openai_to_messages({"messages": [
            {"role": "user", "content": [{"type": "text", "text": "describe"},
                                         {"type": "image_url", "image_url": "image.png"}]}]})
        self.assertIn("<|vision_start|><|image_pad|><|vision_end|>",
                      self.template.render(chat[0], tools=chat[1], **chat[2]))
        msgs = anthropic_to_messages({"messages": [
            {"role": "user", "content": [{"type": "text", "text": "describe"},
                                         {"type": "image", "source": {"type": "url", "url": "image.png"}}]}]})
        self.assertIn("<|vision_start|><|image_pad|><|vision_end|>",
                      self.template.render(msgs[0], tools=msgs[1], **msgs[2]))


if __name__ == "__main__":
    unittest.main()


class EmptyTurnsSwitch(unittest.TestCase):
    def test_keep_empty_turns_restores_the_old_prompt(self):
        import os
        from unittest import mock
        template = ChatTemplate(Path(__file__).with_name("test_chat_template.jinja"))
        msgs = [{"role": "user", "content": "a"}, {"role": "assistant", "content": ""},
                {"role": "user", "content": "b"}]
        clean = template.render([msgs[0], msgs[2]])
        self.assertEqual(template.render(msgs), clean)
        with mock.patch.dict(os.environ, {"XESTRATA_KEEP_EMPTY_TURNS": "1"}):
            self.assertNotEqual(template.render(msgs), clean)


class ForcedCallOpening(unittest.TestCase):
    def test_required_with_one_tool_names_it(self):
        from serve.frontend import forced_call
        one, two = [{"name": "a"}], [{"name": "a"}, {"name": "b"}]
        self.assertTrue(forced_call("required", one).endswith("<function=a>\n"))
        self.assertTrue(forced_call("required", two).endswith("<function="))
        self.assertIsNone(forced_call("auto", one))


class QuotedToolCallCloser(unittest.TestCase):
    """#2026: a `</tool_call>` a user quotes is text, not the end-of-call token (type 4: matched even without
    parse_special); an assistant message replaying raw call text keeps the model's own tokens."""

    def setUp(self):
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
        import strata_tokenizer as ST
        b2u = ST.bytes_to_unicode()
        self.tok = ST.Tokenizer([b2u[b] for b in range(256)] + ["</tool_call>", "<|im_end|>"], [], [1] * 256 + [4, 3])
        self.tags = __import__("serve.frontend", fromlist=["literal_tags"]).literal_tags(self.tok.control_tokens)

    def encode(self, role, text):
        from serve.frontend import mark_think_literals, unmark_think_literals
        marked, _, changed = mark_think_literals([{"role": role, "content": text}], None, self.tags)
        prompt, plain = unmark_think_literals(marked[0]["content"], self.tags)
        return self.tok.encode(prompt, parse_special=True, plain=plain), changed

    def test_user_text_stays_text(self):
        text = "output verbatim: </tool_call>."
        ids, changed = self.encode("user", text)
        self.assertTrue(changed)
        self.assertEqual(ids, [*text.encode()])
        self.assertNotEqual(self.tok.encode(text, parse_special=True), ids)     # unmarked, it was the special token

    def test_assistant_replay_keeps_its_token(self):
        text = "<tool_call>x</tool_call>"
        ids, _ = self.encode("assistant", text)
        self.assertEqual(ids, self.tok.encode(text, parse_special=True))


class FencedJsonCallInWrapper(unittest.TestCase):
    """#2037: with recovery on, a declared tool's JSON call in a Markdown fence inside <tool_call> is the call."""
    TOOLS = [{"name": "write_file", "parameters": {"type": "object", "properties": {"content": {"type": "string"}}}}]
    PAYLOAD = '{"name": "write_file", "arguments": {"content": "hello `world`"}}'

    def run_parser(self, raw, chunk, stream_tools, recover=True, tools=None):
        from serve.frontend import OutputParser
        p = OutputParser(thinking=False, tools=self.TOOLS if tools is None else tools, recover=recover,
                         stream_tools=stream_tools)
        events = []
        for i in range(0, len(raw), chunk):
            events += p.feed(raw[i:i + chunk])
        events += p.finish()
        return ([e.call.arguments for e in events if e.kind == "tool_call"],
                "".join(e.text for e in events if e.kind == "content"))

    def test_fenced_and_bare_agree(self):
        for fence in ("```json\n", "```\n", "```JSON\r\n"):
            raw = f"<tool_call>\n{fence}{self.PAYLOAD}\n```\n</tool_call>"
            for chunk in (len(raw), 1, 7):
                for st in (False, True):
                    with self.subTest(fence=fence, chunk=chunk, stream_tools=st):
                        self.assertEqual(self.run_parser(raw, chunk, st), ([{"content": "hello `world`"}], ""))

    def test_other_fences_and_prose_stay_text(self):
        raw = "<tool_call>\n```python\nx = 1\n```\n</tool_call>"
        for chunk in (len(raw), 1):
            self.assertEqual(self.run_parser(raw, chunk, False), ([], raw))
        raw = f"Here:\n```json\n{self.PAYLOAD}\n```\n"                 # no wrapper: just text
        self.assertEqual(self.run_parser(raw, len(raw), False), ([], raw))

    def test_off_without_recovery_or_for_an_undeclared_tool(self):
        raw = f"<tool_call>\n```json\n{self.PAYLOAD}\n```\n</tool_call>"
        calls, text = self.run_parser(raw, len(raw), False, recover=False)
        self.assertEqual((calls, text), ([], raw))
        other = [{"name": "other", "parameters": {}}]
        calls, text = self.run_parser(raw, len(raw), False, tools=other)
        self.assertEqual((calls, text), ([], raw))


class EmptyConversation(unittest.TestCase):
    def test_no_message_is_an_error_a_message_is_not(self):
        with self.assertRaises(ValueError):
            require_messages(openai_to_messages({"messages": []})[0])
        with self.assertRaises(ValueError):
            require_messages(openai_to_messages({})[0])
        require_messages(openai_to_messages({"messages": [{"role": "user", "content": "hi"}]})[0])


# the lines of each model's template that take its thinking (a stand-in, not the model's own file)
GLM_TPL = ("{%- set e = reasoning_effort if reasoning_effort is defined and reasoning_effort in ['low', 'high'] "
           "else 'max' -%}Reasoning Effort: {{ e | capitalize }}<arg_key>{%- for m in messages %}{{ m.content }}"
           "{%- endfor %}{%- if add_generation_prompt %}<|assistant|><think>{%- endif %}")
HARMONY_TPL = ("Reasoning: {{ reasoning_effort | default('medium') }}<|channel|>{%- for m in messages %}{{ m.content }}"
               "{%- endfor %}{%- if add_generation_prompt %}<|start|>assistant{%- endif %}")
QWEN38_TPL = ("{{ reasoning_effort | default('xhigh') }}{%- for m in messages %}{{ m.content }}{%- endfor %}"
              "{%- if enable_thinking is defined and enable_thinking is false %}off{%- endif %}")
ONOFF_TPL = ("{%- for m in messages %}{{ m.content }}{%- endfor %}{%- if add_generation_prompt %}<|im_start|>"
             "assistant\n{%- if enable_thinking is defined and enable_thinking is false %}<think>\n\n</think>"
             "{%- else %}<think>{%- endif %}{%- endif %}")


class ThinkingLevels(unittest.TestCase):
    """Each template's thinking levels by their own names (/health's "thinking") and the efforts it is given."""

    def template(self, source: str) -> ChatTemplate:
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "chat_template.jinja"
            p.write_text(source, encoding="utf-8")
            return ChatTemplate(p)

    def render(self, t: ChatTemplate, **kw) -> str:
        return t.render([{"role": "user", "content": "hi"}], add_generation_prompt=True, **kw)

    def test_levels_by_template(self):
        for source, levels, default in ((GLM_TPL, ["none", "low", "high", "max"], "max"),
                                        (HARMONY_TPL, ["none", "low", "medium", "high"], "medium"),
                                        (QWEN38_TPL, ["none", "low", "medium", "high"], "high"),
                                        (ONOFF_TPL, ["none", "on"], "on")):
            self.assertEqual(self.template(source).thinking, {"levels": levels, "default": default})

    def test_glm_takes_its_efforts_by_name(self):
        t = self.template(GLM_TPL)
        for given, level in (("low", "Low"), ("medium", "High"), ("high", "High"), ("xhigh", "Max"),
                             ("max", "Max"), (None, "Max")):
            self.assertIn(f"Reasoning Effort: {level}", self.render(t, **effort_kwargs(given)), given)

    def test_harmony_high_and_no_thinking(self):
        t = self.template(HARMONY_TPL)
        for given, level in (("high", "high"), ("xhigh", "high"), ("max", "high"), ("low", "low"), (None, "medium")):
            self.assertIn(f"Reasoning: {level}", self.render(t, **effort_kwargs(given)), given)
        no_thinking = HARMONY_TURN + HARMONY_NO_THINKING + HARMONY_TURN
        self.assertTrue(self.render(t, **effort_kwargs("none")).endswith(no_thinking))
        self.assertTrue(self.render(t).endswith("hi" + HARMONY_TURN))
