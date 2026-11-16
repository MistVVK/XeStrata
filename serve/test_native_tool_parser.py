"""Native tool parsing regressions; no model or GPU required."""
import json
import unittest
from serve.frontend import OutputParser

CALL = "<tool_call>\n<function=mcp__files__read>\n<parameter=path>\n123\n</parameter>\n</function>\n</tool_call>"
TOOLS = [{"name": "mcp__files.read", "parameters": {"properties": {"path": {"type": "string"}}}}]

class NativeTools(unittest.TestCase):
    def test_a_large_streamed_string_does_not_keep_one_growing_buffer(self):
        tools = [{"name": "write", "parameters": {"properties": {"content": {"type": "string"}}}}]
        value = ("x" * 5000 + " literal </parameter> and </function> and </tool_call> " + "y" * 5000)
        parser = OutputParser(thinking=False, tools=tools, stream_tools=True, recover=True)
        opening = "<tool_call><function=write><parameter=content>\n"
        events = parser.feed(opening + value)
        self.assertTrue(parser._scan_parts)       # consumed text is chunked instead of recopied on every next token
        self.assertFalse(parser.clean)
        closing = "\n</parameter></function></tool_call>"
        for i in range(0, len(closing), 3):
            events += parser.feed(closing[i:i + 3])
        events += parser.finish()
        calls = [event.call for event in events if event.kind == "tool_call"]
        self.assertEqual([(call.name, call.arguments) for call in calls], [("write", {"content": value})])
        streamed = "".join(event.text for event in events if event.kind == "tool_args")
        self.assertEqual(json.loads(streamed), {"content": value})

    def test_compacted_call_keeps_closing_sequence_lookahead(self):
        tools = [{"name": "write", "parameters": {"properties": {"content": {"type": "string"}}}}]
        opening = "<tool_call><function=write><parameter=content>"
        closing = "</parameter></function></tool_call>"
        value = "x" * 5000
        for recover in (False, True):
            for tail in ("", " prose", " prose " * 220, opening + "second" + closing,
                         " more file text " + closing):
                with self.subTest(recover=recover, tail=tail[:40]):
                    parser = OutputParser(thinking=False, tools=tools, stream_tools=True, recover=recover)
                    events = parser.feed(opening + value)
                    events += parser.feed(closing)
                    self.assertFalse(any(event.kind == "tool_call" for event in events))
                    for i in range(0, len(tail), 7):
                        events += parser.feed(tail[i:i + 7])
                    if len(tail) > 1500 or tail.startswith(opening):
                        self.assertTrue(any(event.kind == "tool_call" for event in events))
                    events += parser.finish()
                    expected = [value + closing + " more file text " if tail.startswith(" more") else value]
                    if tail.startswith(opening):
                        expected.append("second")
                    self.assertEqual([event.call.arguments["content"] for event in events if event.kind == "tool_call"],
                                     expected)

    def test_compacted_incomplete_call_restores_body_after_final_feed(self):
        tools = [{"name": "write", "parameters": {"properties": {"content": {"type": "string"}}}}]
        parser = OutputParser(thinking=False, tools=tools, stream_tools=True)
        value = "x" * 5000
        events = parser.feed("<tool_call><function=write><parameter=content>" + value)
        events += parser.feed("</parameter></function>")
        events += parser.finish()
        self.assertEqual([event.call.arguments for event in events if event.kind == "tool_call"], [{"content": value}])
        self.assertEqual(json.loads("".join(event.text for event in events if event.kind == "tool_args")),
                         {"content": value})

    def test_unclosed_think_call_and_split_tags(self):
        for streaming in (False, True):
            for step in (1, 7, 4096):
                parser = OutputParser(thinking=True, tools=TOOLS, stream_tools=streaming)
                events = []
                text = "Inspect the file.\n" + CALL
                for i in range(0,len(text),step):
                    events += parser.feed(text[i:i+step])
                events += parser.finish()
                calls = [e.call for e in events if e.kind == "tool_call"]
                self.assertEqual(len(calls),1)
                self.assertEqual((calls[0].name, calls[0].arguments), ("mcp__files.read", {"path": "123"}))
                self.assertEqual("".join(e.text for e in events if e.kind == "reasoning"), "Inspect the file.\n")


    def test_whitespace_name_keeps_namespace_schema_and_stream_events(self):
        for thinking, prefix in ((True, "Inspect the file.\n"), (True, "Ready.</think>\n\n"), (False, "")):
            for whitespace in (" ", "\t", " \t"):
                text = prefix + CALL.replace("<function=mcp__files__read>",
                                             "<function=" + whitespace + "mcp__files__read" + whitespace + ">")
                for streaming in (False, True):
                    for step in (1, 7, 4096):
                        with self.subTest(thinking=thinking, prefix=prefix, whitespace=whitespace,
                                          streaming=streaming, step=step):
                            parser = OutputParser(thinking=thinking, tools=TOOLS, stream_tools=streaming)
                            events = []
                            for i in range(0, len(text), step):
                                events += parser.feed(text[i:i + step])
                            events += parser.finish()
                            calls = [event.call for event in events if event.kind == "tool_call"]
                            self.assertEqual([(c.name, c.arguments) for c in calls],
                                             [("mcp__files.read", {"path": "123"})])
                            self.assertTrue(all(event.call.name == "mcp__files.read"
                                                for event in events if event.kind == "tool_start"))


    def test_padded_unknown_wrapper_stays_text_and_unpadded_stays_native(self):
        for recover in (False, True):
            for streaming in (False, True):
                for step in (1, 7, 4096):
                    for name in (" missing ", "\tmissing\t", "mcp__ files", " "):
                        text = "<tool_call><function=" + name + "></function></tool_call>"
                        parser = OutputParser(thinking=False, tools=TOOLS, stream_tools=streaming, recover=recover)
                        events = []
                        for i in range(0, len(text), step):
                            events += parser.feed(text[i:i + step])
                        events += parser.finish()
                        self.assertFalse(any(event.kind.startswith("tool") for event in events))
                        self.assertEqual("".join(event.text for event in events if event.kind == "content"), text)
                    # an unpadded name nobody declared: text where tools are declared (the 0.2.2 rule, #3075); the Responses
                    # route of a request with no tools (Codex compaction, declared_only off) keeps it as a native call
                    text = "<tool_call><function=missing></function></tool_call>"
                    for tools, declared_only, expected in ((TOOLS, False, []), (TOOLS, True, []), (None, False, ["missing"])):
                        parser = OutputParser(thinking=False, tools=tools, stream_tools=streaming, recover=recover,
                                              declared_only=declared_only)
                        events = []
                        for i in range(0, len(text), step):
                            events += parser.feed(text[i:i + step])
                        events += parser.finish()
                        self.assertEqual([event.call.name for event in events if event.kind == "tool_call"], expected)


    def test_padded_unknown_second_call_is_not_announced(self):
        text = CALL[:-len("</tool_call>")] + "<function= missing ></function></tool_call>"
        for step in (1, 7, 4096):
            parser = OutputParser(thinking=False, tools=TOOLS, stream_tools=True, recover=True)
            events = []
            for i in range(0, len(text), step):
                events += parser.feed(text[i:i + step])
            events += parser.finish()
            self.assertEqual([event.call.name for event in events if event.kind == "tool_call"], ["mcp__files.read"])
            self.assertEqual([event.call.name for event in events if event.kind == "tool_start"], ["mcp__files.read"])
            self.assertIn("<function= missing >", "".join(event.text for event in events if event.kind == "content"))


    def test_padded_bare_alias_is_declared_and_schema_typed(self):
        text = CALL.replace("<tool_call>\n", "").replace("\n</tool_call>", "").replace(
            "<function=mcp__files__read>", "<function= mcp__files__read >")
        for step in (1, 7, 4096):
            parser = OutputParser(thinking=False, tools=TOOLS, stream_tools=True, recover=True)
            events = []
            for i in range(0, len(text), step):
                events += parser.feed(text[i:i + step])
            events += parser.finish()
            self.assertEqual([(event.call.name, event.call.arguments) for event in events if event.kind == "tool_call"],
                             [("mcp__files.read", {"path": "123"})])
            starts = [event.call.name for event in events if event.kind == "tool_start"]
            self.assertEqual(starts, ["mcp__files.read"])


    def test_literal_call_markers_remain_reasoning_prose(self):
        for prefix in ("Quote <tool_call> as prose.", "Use `<tool_call>` in docs.", "Literal marker:\n<tool_call>\nContinue thinking.", "Empty <tool_call><function=> is prose."):
            for step in (1, 7, 4096):
                parser=OutputParser(thinking=True,tools=TOOLS,stream_tools=True)
                text=prefix+"</think>\n\nVisible answer."
                events=[]
                for i in range(0,len(text),step):events+=parser.feed(text[i:i+step])
                events+=parser.finish()
                self.assertEqual(''.join(e.text for e in events if e.kind=='reasoning'),prefix)
                self.assertEqual(''.join(e.text for e in events if e.kind=='content'),'Visible answer.')
                self.assertFalse(any(e.kind.startswith('tool') for e in events))


    def test_literal_call_markers_in_content_before_genuine_call(self):
        prefix = "Use `<tool_call>` in docs. Empty <tool_call><function=> stays text."
        for streaming in (False, True):
            for step in (1, 7, 4096):
                parser = OutputParser(thinking=False, tools=TOOLS, stream_tools=streaming)
                text = prefix + "\n" + CALL
                events = []
                for i in range(0, len(text), step):
                    events += parser.feed(text[i:i+step])
                events += parser.finish()
                self.assertEqual(''.join(e.text for e in events if e.kind == 'content'), prefix)
                calls = [e.call for e in events if e.kind == 'tool_call']
                self.assertEqual(len(calls), 1)
                self.assertEqual(calls[0].arguments, {'path': '123'})


    def test_namespace_alias_schema_and_exact_name_precedence(self):
        for streaming in (False, True):
            parser = OutputParser(thinking=False, tools=TOOLS, stream_tools=streaming)
            events = parser.feed(CALL) + parser.finish()
            call = next(e.call for e in events if e.kind == "tool_call")
            self.assertEqual((call.name,call.arguments), ("mcp__files.read", {"path": "123"}))
            exact = {"name": "mcp__files__read", "parameters": {"properties": {"path": {"type": "integer"}}}}
            parser = OutputParser(thinking=False, tools=[*TOOLS,exact], stream_tools=streaming)
            call = next(e.call for e in parser.feed(CALL)+parser.finish() if e.kind == "tool_call")
            self.assertEqual((call.name,call.arguments), ("mcp__files__read", {"path": 123}))

        ambiguous = [{"name": "a.b__c"}, {"name": "a__b.c"}]
        parser = OutputParser(thinking=False, tools=ambiguous)
        text = CALL.replace("mcp__files__read", "a__b__c")
        events = parser.feed(text) + parser.finish()
        # an ambiguous alias is no declared tool: it does not route to the first namespace; it stays text
        self.assertFalse(any(e.kind == "tool_call" for e in events))
        self.assertEqual("".join(e.text for e in events if e.kind == "content"), text)
        self.assertEqual(parser.unread, 1)                           # and it is counted, so the server logs it


if __name__ == "__main__":
    unittest.main()
