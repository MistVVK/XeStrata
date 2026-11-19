"""0.2.17 lane GUI-TUI: what the hands-on QA of the Terminal screens found.  Screen cases run at 80 x 24 and 120 x 40 through Textual's pilot."""

from __future__ import annotations

import unittest

from tests.terminal.launch_support import HW_12, Home, catalog, text_of
from tests.terminal.test_fix2_tui_0216 import goto, small
from tests.terminal.test_launch_tui import make_ctx


class ModelStepExplanation(unittest.TestCase):
    """The Model step's explanation box (80 x 24: four lines) used to end in the middle of a sentence for a model that does not fit."""

    def text(self, size):
        with Home():
            ctx = make_ctx({**HW_12, "disk_free_gb": 57.0})

            async def coro(app, pilot, c):
                await goto(pilot, app, "model")
                await pilot.pause(0.3)
                return text_of(app)
            return small(self, coro, ctx, size)

    def test_80x24_does_not_end_the_box_in_the_middle_of_the_general_sentence(self):
        t = self.text((80, 24))
        self.assertIn("Not enough disk space", t)
        self.assertNotIn("The n-gram table", t)           # the sentence that was cut: "... stays on your SSD and is"
        self.assertNotRegex(t, r"stays on your SSD and is\s*\n")
        lines = [ln for ln in t.split("\n") if ln.strip()]
        # the last line above the key hints is a whole sentence
        box = lines[-2]
        self.assertTrue(box.rstrip().endswith((".", "!", ")")), box)

    def test_a_tall_window_still_shows_the_general_sentence_whole(self):
        t = " ".join(self.text((120, 40)).split())
        self.assertIn("The n-gram table stays on your SSD and is not counted above.", t)

    def test_a_model_that_fits_keeps_the_sentence_at_80x24(self):
        with Home():
            ctx = make_ctx({**HW_12, "disk_free_gb": 5000.0})

            async def coro(app, pilot, c):
                await goto(pilot, app, "model")
                await pilot.pause(0.3)
                return text_of(app)
            t = " ".join(small(self, coro, ctx, (80, 24)).split())
        self.assertIn("The n-gram table stays on your SSD and is not counted above.", t)


class AdvancedTunerNote(unittest.TestCase):
    """The Advanced step (80 columns) said "The tuner chose off." for a switch that is simply off by default, in a launch whose Tuning step said Skip."""

    class Row:
        def __init__(self, default, tuned):
            self.default, self.tuned = default, tuned

    def test_a_value_equal_to_the_default_is_not_called_a_choice_of_the_tuner(self):
        from strata_cli.launch.screens import tuned_sentence
        self.assertEqual(tuned_sentence(self.Row("off", "off")), "")
        self.assertEqual(tuned_sentence(self.Row("", "-")), "")
        self.assertEqual(tuned_sentence(self.Row("off", "")), "")

    def test_a_value_the_tuner_changed_is_named(self):
        from strata_cli.launch.screens import tuned_sentence
        self.assertEqual(tuned_sentence(self.Row("off", "on")), " The tuner chose on.")
        self.assertEqual(tuned_sentence(self.Row("", "8")), " The tuner chose 8.")


if __name__ == "__main__":
    unittest.main()
