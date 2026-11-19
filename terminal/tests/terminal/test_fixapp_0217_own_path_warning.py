"""#6173: setup was said to accept `--context 262144` for Qwen3.6-35B-A3B UD-Q6_K_XL on a 12 GB card without a word, and the engine then refused to start.
Setup says it, in the sentence the server says at the start (own_path_context_warning, #5715), and keeps the choice (recommend, never force). This keeps it for the
Q6_K_XL file and the card sizes of the report, and checks that the sentence names a context that does fit."""
from __future__ import annotations

import re
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import setup  # noqa: E402


class ContextWarning(unittest.TestCase):
    fam = setup.OWN_PATH_FAMILIES["qwen36"]

    def test_256k_on_a_12_gb_card_is_said_for_both_files(self):
        for model in ("UD-Q4_K_XL", "UD-Q6_K_XL"):
            for vram in (11.5, 11.94, 12.0):
                text = setup.own_path_context_warning(self.fam, model, vram, 262144)
                self.assertIsNotNone(text, (model, vram))
                self.assertIn("At 256K", text)
                self.assertIn("will probably not start", text)
                self.assertRegex(text, r"\d+K fits on this card: choose \d+K or less\.")

    def test_the_context_it_names_is_one_that_fits(self):
        from strata_control import own_path_fit
        for model in ("UD-Q4_K_XL", "UD-Q6_K_XL"):
            fit = own_path_fit.inputs(self.fam.get("name"), model)
            text = setup.own_path_context_warning(self.fam, model, 11.94, 262144)
            named = int(re.search(r"(\d+)K fits", text).group(1)) * 1024
            self.assertTrue(own_path_fit.fits(fit, 11.94, named))
            self.assertFalse(own_path_fit.fits(fit, 11.94, 262144))

    def test_a_context_that_fits_is_not_warned_about(self):
        self.assertIsNone(setup.own_path_context_warning(self.fam, "UD-Q6_K_XL", 11.94, 32768))

    def test_a_big_card_is_not_warned_about(self):
        self.assertIsNone(setup.own_path_context_warning(self.fam, "UD-Q6_K_XL", 48.0, 262144))


if __name__ == "__main__":
    unittest.main()
