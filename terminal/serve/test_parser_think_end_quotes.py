"""#5605 / #6218: whether a `</think>` right after a quote ends the thinking or is the model quoting its own tag.
- #5605: a quote that CLOSES a sentence (`said "Reply with the number only."</think>2`) does not open a quotation the tag is in: the thinking ends there
  (the answer was kept as reasoning, so it came back empty).
- #6218: a tag on its own line inside a backtick span the model opened and closes after it (`the marker `, line break, tag, line break, backtick) is quoted.
- the pinned cases stay: `returns "</think>\\n\\n2"`, `` `</think>` ``, `'</think>'`, `("</think>...` are quotes; `plan</think>answer` and a tag followed by a line break end it.

    python -m unittest serve.test_parser_think_end_quotes
"""
import unittest

from serve.frontend import OutputParser

END = "</" + "think>"


def split(text, step):
    p = OutputParser(thinking=True)
    ev = []
    for k in range(0, max(len(text), 1), step):
        ev += p.feed(text[k:k + step])
    ev += p.finish()
    return {kind: "".join(e.text for e in ev if e.kind == kind) for kind in ("reasoning", "content")}


class ThinkEndQuotes(unittest.TestCase):
    def check(self, text, reasoning, content):
        for step in (1, 2, 7, 100000):
            self.assertEqual(split(text, step), {"reasoning": reasoning, "content": content}, (text, step))

    def test_a_closing_quote_before_the_end_5605(self):
        self.check('The user said "Reply with the number only."' + END + "2", 'The user said "Reply with the number only."', "2")
        self.check("It is `x`" + END + "answer", "It is `x`", "answer")
        self.check("she said 'go.'" + END + "ok", "she said 'go.'", "ok")

    def test_the_end_after_a_quote_and_a_line_break_still_ends(self):
        self.check('she said "yes"' + END + "\n\nanswer", 'she said "yes"', "answer")

    def test_an_opening_quote_before_the_tag_is_still_a_quote(self):
        for before in ('it returns "', "split it on '", "call `", 'text = ("'):
            text = "plan " + before + END + 'x" and on\n' + END + "\n\nanswer"
            self.check(text, "plan " + before + END + 'x" and on\n', "answer")

    def test_the_tag_on_its_own_line_inside_a_backtick_span_6218(self):
        self.check("the marker `\n" + END + "\n\n` appears once\n" + END + "\n\nanswer", "the marker `\n" + END + "\n\n` appears once\n", "answer")

    def test_a_backtick_answer_after_a_real_end_is_not_taken_for_a_quote(self):
        self.check("used `x`\n" + END + "\n\n`code` block", "used `x`\n", "`code` block")
        self.check("plan\n" + END + "\n\n`code` block", "plan\n", "`code` block")

    def test_no_quote_at_all(self):
        self.check("plan" + END + "answer", "plan", "answer")
        self.check("Simple.\n" + END + "\n\n2", "Simple.\n", "2")


if __name__ == "__main__":
    unittest.main()
