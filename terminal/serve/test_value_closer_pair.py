"""A value whose text has `</parameter>` and then `</function>` on the next line, followed by more of the file (#4701's rule, one tag short).

Live (Pi on /v1/chat/completions, Qwen3.8-Flash-Next): a write of an XML file whose element `<parameter ...>...</parameter>` was the last
child of a `<function ...>` element.  The value ended at that `</parameter>`, the call went out with the file cut there and no error,
and the rest of the file was lost.  The sequence now ends the value only as #4701 decided for the three tags: unless a later closing
tag follows inside the same wrapped call.

    python -m unittest serve.test_value_closer_pair
"""
import unittest

from serve.frontend import OutputParser


def tool(name, **props):
    return {"name": name, "parameters": {"type": "object", "properties": props}}


S = {"type": "string"}
TOOLS = [tool("read", path=S), tool("write", path=S, content=S)]
XML = ('<catalog>\n  <function name="esc_case">\n    <parameter name="path" type="str">C:\\tmp\\a.xml</parameter>\n  </function>\n'
       '  <function name="next">\n    <call id="c-5">noop</call>\n  </function>\n</catalog>')
READ = "<tool_call>\n<function=read>\n<parameter=path>\na.txt\n</parameter>\n</function>\n</tool_call>"
READ_OPEN = "<tool_call>\n<function=read>\n<parameter=path>\na.txt\n</parameter>\n</function>\n"      # its </tool_call> left out


def write(first, second):
    return (f"<tool_call>\n<function=write>\n<parameter={first[0]}>\n{first[1]}\n</parameter>\n"
            f"<parameter={second[0]}>\n{second[1]}\n</parameter>\n</function>\n</tool_call>")


def run(text, piece=None, recover=False, stream_tools=False):
    p = OutputParser(thinking=False, tools=TOOLS, stream_tools=stream_tools, recover=recover, declared_only=True)
    evs = []
    piece = piece or len(text)
    for i in range(0, len(text), piece):
        evs += p.feed(text[i:i + piece])
    evs += p.finish("stop")
    return ([(e.call.name, e.call.arguments) for e in evs if e.kind == "tool_call"],
            "".join(e.text for e in evs if e.kind == "content"))


class ValueCloserPair(unittest.TestCase):
    def test_the_whole_file_stays_in_the_value(self):
        for order in (("content", XML), ("path", "s.xml")), (("path", "s.xml"), ("content", XML)):
            for recover in (False, True):
                for stream_tools in (False, True):
                    for piece in (1, 7, None):
                        with self.subTest(first=order[0][0], recover=recover, stream=stream_tools, piece=piece):
                            calls, answer = run(write(*order), piece, recover, stream_tools)
                            self.assertEqual(calls, [("write", {"path": "s.xml", "content": XML})])
                            self.assertEqual(answer.strip(), "")

    def test_a_call_whose_wrapper_close_is_left_out_still_ends_before_the_next_call(self):
        for piece in (1, 7, None):
            with self.subTest(piece=piece):
                calls, _ = run(READ_OPEN + READ, piece)
                self.assertEqual(calls, [("read", {"path": "a.txt"}), ("read", {"path": "a.txt"})])

    def test_a_file_that_lists_all_three_closing_tags_is_unchanged(self):
        doc = "The closing tags, in order:\n\n</parameter>\n</function>\n</tool_call>\n\nNothing may follow them."
        for piece in (1, 7, None):
            with self.subTest(piece=piece):
                calls, _ = run(write(("path", "d.md"), ("content", doc)), piece)
                self.assertEqual(calls, [("write", {"path": "d.md", "content": doc})])


if __name__ == "__main__":
    unittest.main()
