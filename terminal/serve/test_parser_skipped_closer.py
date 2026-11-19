"""#5756: a value that holds a nested `<parameter=X>...</parameter>` pair (a file that documents the call format), where the model leaves out the value's
own `</parameter>`: the value ended at the nested closer, so the call was delivered with a truncated value (`<parameter=limit>25`) and the rest was lost.

    python -m unittest serve.test_parser_skipped_closer
"""
import unittest

from serve.frontend import OutputParser, param_end

S = {"type": "string"}
TOOLS = [{"name": "edit_file", "parameters": {"type": "object", "properties": {"path": S, "new_string": S, "old_string": S}}}]


def parse(text, piece=0, recover=False, stream_tools=False):
    p = OutputParser(thinking=False, tools=TOOLS, stream_tools=stream_tools, recover=recover, declared_only=True)
    evs = []
    if piece:
        for i in range(0, len(text), piece):
            evs += p.feed(text[i:i + piece])
        evs += p.finish("stop")
    else:
        evs = p.feed(text) + p.finish("stop")
    calls = [(e.call.name, e.call.arguments) for e in evs if e.kind == "tool_call"]
    return calls, "".join(e.text for e in evs if e.kind == "content")


SKIPPED = ('<tool_call>\n<function=edit_file>\n<parameter=path>\ndocs/tools.md\n</parameter>\n<parameter=new_string>\n'
           '<parameter=limit>25</parameter>\n</function>\n</tool_call>')
SKIPPED_NEXT = ('<tool_call>\n<function=edit_file>\n<parameter=new_string>\n<parameter=limit>25</parameter>\n'
                '<parameter=old_string>\nx\n</parameter>\n</function>\n</tool_call>')
OWN_CLOSER = ('<tool_call>\n<function=edit_file>\n<parameter=new_string>\nuse\n<parameter=limit>25</parameter>\nhere\n</parameter>\n'
              '</function>\n</tool_call>')


class SkippedCloser(unittest.TestCase):
    def each(self, text):
        for recover in (False, True):
            for piece in (0, 1, 7):
                for stream in (False, True):
                    yield parse(text, piece, recover, stream)

    def test_param_end_cases_of_the_report(self):
        for t, want in (('<parameter=limit>25</parameter>\n</function>\n</tool_call>', '<parameter=limit>25</parameter>'),
                        ('<parameter=limit>25</parameter>\n<parameter=old_string>x</parameter>\n</function>', '<parameter=limit>25</parameter>'),
                        ('<parameter=limit>25</parameter>\n</parameter>\n</function>', '<parameter=limit>25</parameter>\n')):
            e = param_end(t, final=True, name="new_string")
            self.assertEqual(t[:e], want)

    def test_the_value_keeps_its_nested_pair_when_the_call_ends(self):
        for calls, answer in self.each(SKIPPED):
            self.assertEqual(calls, [("edit_file", {"path": "docs/tools.md", "new_string": "<parameter=limit>25</parameter>"})])
            self.assertEqual(answer, "")

    def test_the_value_keeps_its_nested_pair_before_the_next_parameter(self):
        for calls, answer in self.each(SKIPPED_NEXT):
            self.assertEqual(calls, [("edit_file", {"new_string": "<parameter=limit>25</parameter>", "old_string": "x"})])

    def test_a_value_with_its_own_closer_is_unchanged(self):
        for calls, answer in self.each(OWN_CLOSER):
            self.assertEqual(calls, [("edit_file", {"new_string": "use\n<parameter=limit>25</parameter>\nhere"})])

    def test_a_following_call_is_not_lost(self):
        for calls, answer in self.each(SKIPPED + "\n" + SKIPPED_NEXT):
            self.assertEqual(len(calls), 2)


if __name__ == "__main__":
    unittest.main()
