"""#4701 decision data: a file whose text lists </parameter>, </function>, </tool_call> on lines of their own, written through a real call.

Option C (decided 10-30, phaseA-tests/urgent029/OPTIONS-4701-4704.md): the closing sequence ends the value unless prose follows it AND a later
closing sequence follows that prose, with no new <tool_call> opener in between.  While the turn is still streaming a call that might go on is
held, bounded by the end of the turn, a new <tool_call>, or CLOSE_LOOKAHEAD_CHARS of text.

    python -m unittest serve.test_param_end_options
"""
import unittest

from serve.frontend import OutputParser


def tool(name, **props):
    return {"name": name, "parameters": {"type": "object", "properties": props}}


S = {"type": "string"}
TOOLS = [tool("read", path=S), tool("write_file", path=S, content=S)]
DOC = "# Tool calls\n\nThe closing tags, in order:\n\n</parameter>\n</function>\n</tool_call>\n\nNothing may follow them."
READ = "<tool_call>\n<function=read>\n<parameter=path>\na.txt\n</parameter>\n</function>\n</tool_call>"
WRITE = ("<tool_call>\n<function=write_file>\n<parameter=path>\nd.md\n</parameter>\n<parameter=content>\n" + DOC
         + "\n</parameter>\n</function>\n</tool_call>")


def parse(text):
    p = OutputParser(thinking=False, tools=TOOLS, stream_tools=False, recover=False, declared_only=True)
    evs = p.feed(text) + p.finish("stop")
    return [(e.call.name, e.call.arguments) for e in evs if e.kind == "tool_call"], "".join(e.text for e in evs if e.kind == "content")


class ParamEndOptions(unittest.TestCase):
    def test_a_file_that_lists_the_closing_tags_keeps_all_its_text(self):
        calls, answer = parse(WRITE)
        self.assertEqual(calls, [("write_file", {"path": "d.md", "content": DOC})])
        self.assertEqual(answer, "")

    def test_a_call_followed_by_a_sentence_is_still_a_call(self):
        calls, answer = parse(READ + "\n\nI will read that file now.")
        self.assertEqual(calls, [("read", {"path": "a.txt"})])
        self.assertIn("I will read", answer)

    def test_two_calls_with_a_sentence_between_are_two_calls(self):
        calls, _ = parse(READ + "\nOkay.\n" + READ)
        self.assertEqual(len(calls), 2)


def stream(text, piece, finish=True):
    p = OutputParser(thinking=False, tools=TOOLS, stream_tools=False, recover=False, declared_only=True)
    seen = []                      # (characters fed so far, calls delivered so far)
    n = 0
    for i in range(0, len(text), piece):
        n += len(text[i:i + piece])
        got = [e for e in p.feed(text[i:i + piece]) if e.kind == "tool_call"]
        seen.append((n, len(got)))
    evs = p.finish("stop") if finish else []
    calls = [(e.call.name, e.call.arguments) for e in evs if e.kind == "tool_call"]
    return seen, calls, "".join(e.text for e in evs if e.kind == "content")


class Streaming(unittest.TestCase):
    def test_the_file_comes_whole_however_it_is_cut_into_pieces(self):
        for piece in (1, 3, 7, 50, len(WRITE)):
            p = OutputParser(thinking=False, tools=TOOLS, stream_tools=False, recover=False, declared_only=True)
            evs = []
            for i in range(0, len(WRITE), piece):
                evs += p.feed(WRITE[i:i + piece])
            evs += p.finish("stop")
            calls = [(e.call.name, e.call.arguments) for e in evs if e.kind == "tool_call"]
            self.assertEqual(calls, [("write_file", {"path": "d.md", "content": DOC})], piece)
            self.assertEqual("".join(e.text for e in evs if e.kind == "content"), "", piece)

    def test_an_ordinary_call_is_delivered_when_the_turn_ends_and_not_before_a_second_call_is_needed(self):
        for piece in (1, 7):
            p = OutputParser(thinking=False, tools=TOOLS, stream_tools=False, recover=False, declared_only=True)
            got = []
            for i in range(0, len(READ), piece):
                got += [e for e in p.feed(READ[i:i + piece]) if e.kind == "tool_call"]
            got += [e for e in p.finish("stop") if e.kind == "tool_call"]
            self.assertEqual(len(got), 1)

    def test_a_second_call_releases_the_first_at_once(self):
        p = OutputParser(thinking=False, tools=TOOLS, stream_tools=False, recover=False, declared_only=True)
        first = [e for e in p.feed(READ + "\n<tool_call>") if e.kind == "tool_call"]
        self.assertEqual(len(first), 1)

    def test_a_long_sentence_after_a_call_cannot_hold_it_for_ever(self):
        from serve.frontend import CLOSE_LOOKAHEAD_CHARS
        p = OutputParser(thinking=False, tools=TOOLS, stream_tools=False, recover=False, declared_only=True)
        got = [e for e in p.feed(READ + "\n\n" + "word " * 200) if e.kind == "tool_call"]
        self.assertEqual(len(got), 0)                       # still inside the bound
        got = [e for e in p.feed("more " * 200) if e.kind == "tool_call"]
        self.assertEqual(len(got), 1)                       # past it: the call goes out while the model is still talking


if __name__ == "__main__":
    unittest.main()
