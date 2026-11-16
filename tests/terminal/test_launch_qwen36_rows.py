"""#5869: `strata launch`'s model list shows Qwen3.6-35B-A3B as a group of its own, with the sizes, titles and words of the shared table the
app uses; the Flash-Next rows are unchanged."""

from __future__ import annotations

import unittest

from tests.terminal.launch_support import HW_12, Home, catalog


class Qwen36Rows(unittest.TestCase):
    def rows(self):
        with Home():
            cat = catalog(HW_12)
            return cat, cat.rows(cat.computer().raw)

    def test_qwen36_is_a_group_of_its_own_with_the_shared_tables_sizes(self):
        cat, rows = self.rows()
        own = cat.strata.own_families()
        self.assertIn("qwen36", own)
        mine = [r for r in rows if r.family == "qwen36"]
        self.assertTrue(mine, "Qwen3.6 has rows")
        table = next(f for f in cat.strata.model_table(cat.hardware()) if f["family"] == "qwen36")
        self.assertEqual([r.model for r in mine], [s["model"] for s in table["sizes"]])
        self.assertEqual({r.family_title for r in mine}, {table["title"]})
        self.assertEqual([r.about for r in mine], [s["about"] for s in table["sizes"]])

    def test_the_flash_next_rows_are_unchanged(self):
        cat, rows = self.rows()
        own = set(cat.strata.own_families())
        flash = [(r.family, r.model) for r in rows if r.family not in own]
        want = [(f["family"], s["model"]) for f in cat.strata.model_table(cat.hardware()) if f["family"] not in own for s in f["sizes"]]
        self.assertEqual(flash, want)


if __name__ == "__main__":
    unittest.main()
