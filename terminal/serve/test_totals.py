"""The page's Model settings card: a help line that wraps, and a field that keeps its width.  Neither the style
nor the server's own words may let one long help line squeeze the card's fields down to a spinner's width again.

    python -m unittest serve.test_totals -v
"""
import re
import unittest
from pathlib import Path

from serve import runconfig
class WhatTheSettingsCardLooksLike(unittest.TestCase):
    """The page's Model settings card is a grid of a help line and a field (#564).  The help is the server's own
    words and may be long, so the card must give way in its label and never in its field: sizing the label column to
    max-content once let one long help line squeeze every field of the card, the four prices among them, down to the
    width of its own spinner arrows.  Neither the style nor the words may bring that back.
    """

    LONGEST = 140      # the longest help line the card was laid out beside before it broke (mtp: 137)

    @classmethod
    def setUpClass(cls):
        cls.css = (Path(__file__).resolve().parent / "web" / "app.css").read_text(encoding="utf-8")

    def test_the_field_column_keeps_a_floor_of_its_own(self):
        rule = re.search(r"\n\.cfg-form \{[^}]*grid-template-columns:([^;]+);", self.css)
        self.assertIsNotNone(rule, "the card's grid rule is gone")
        cols = rule.group(1)
        self.assertNotIn("max-content", cols,               # the label wraps rather than pushing the field out
                         f"the label column is sized to its content again: {cols}")
        self.assertRegex(cols, r"1fr\)\s*minmax\(\s*[\d.]+",  # and the field keeps a width whatever the label says
                         f"the field column has no floor: {cols}")

    def test_no_help_line_is_a_wall_of_text(self):
        """A help line is both the field's label and its tooltip, so it stays short enough to read in one row."""
        for key, _, help_text in runconfig.EDITABLE:
            self.assertLessEqual(len(help_text), self.LONGEST,
                                 f"{key}'s help is {len(help_text)} characters, over the {self.LONGEST} the card fits")


if __name__ == "__main__":
    unittest.main()
