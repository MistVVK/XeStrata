"""0.2.17 retest finding F1: the Model bar was drawn 8 cells wide while "Checking the files downloaded so far (...)" was the phase text
(0.2.16 drew 40 cells at the same width). The item-14 rule that lets the bar give way to the phase words must not cut it for a phrase that
cannot fit beside any bar."""

from __future__ import annotations

import unittest

from strata_cli.launch.runner import Event

from tests.terminal.launch_support import text_of
from tests.terminal.test_fix2_tui_0216 import goto, small
from tests.terminal.test_launch_tui import Base, make_ctx

CHECK = "Checking the files downloaded so far (54.8 of 84.5 GB); the download goes on after it"


def stage_bar_width(*a):
    from strata_cli.launch.screens import stage_bar_width as f
    return f(*a)


class Width(unittest.TestCase):
    def test_the_checking_phrase_keeps_the_long_bar_in_a_110_column_window(self):
        self.assertEqual(stage_bar_width(40, 110, len(CHECK)), 40)

    def test_the_checking_phrase_keeps_the_long_bar_at_80_columns_too(self):
        self.assertEqual(stage_bar_width(40, 80, len(CHECK)), 40)

    def test_a_short_phrase_gives_way_in_a_small_window_as_item_14_wanted(self):
        text = "Downloading the DFlash2 drafter… 0.26 of 1.21 GB"
        self.assertEqual(stage_bar_width(40, 80, len(text)), 80 - 14 - 7 - len(text))

    def test_a_short_phrase_leaves_the_bar_whole_in_a_wide_window(self):
        self.assertEqual(stage_bar_width(40, 110, len("Downloading 54.8 of 84.5 GB")), 40)

    def test_the_bar_is_never_cut_below_its_floor_for_a_phrase_that_fits_with_it(self):
        self.assertEqual(stage_bar_width(40, 80, 80 - 14 - 7 - 8), 8)


class OnScreen(Base):
    def bar_cells(self, size):
        out = {}

        async def case(app, pilot, ctx):
            await goto(pilot, app, "save")
            app.run_launch()
            await pilot.pause(0.3)
            scr = app.run_screen
            scr.handle_event(Event("model", "Downloading 54.8 of 84.5 GB", 64.0, "info", {}))
            await pilot.pause(0.3)
            out["download"] = self.cells(text_of(app), "Downloading 54.8")
            scr.handle_event(Event("model", CHECK, 64.0, "info", {}))
            await pilot.pause(0.3)
            out["check"] = self.cells(text_of(app), "64%")
            app.cancel.set()
        from tests.terminal.test_launch_tui import FakeRunner
        FakeRunner.behaviour = "wait"
        small(self, case, make_ctx(), size=size)
        return out

    @staticmethod
    def cells(text, needle):
        for ln in text.splitlines():
            if needle in ln:
                return sum(ln.count(c) for c in "█░")
        return None

    def test_the_bar_is_40_cells_while_the_files_are_checked_in_110_columns(self):
        got = self.bar_cells((110, 36))
        self.assertEqual(got["download"], 40)
        self.assertEqual(got["check"], 40)


if __name__ == "__main__":
    unittest.main()
