"""#5239: "fit to your card" planned 3 parallel slots for Qwen3.6-35B-A3B UD-Q4_K_XL at 128K on a 24 GB card; the engine needs 5.56 GiB
per slot there and refused to start (15.14 GiB free) after the download. For a model with an engine path of its own the number now comes
from setup's rule for that model and context (never more than the VRAM table's); Flash-Next keeps the table."""

from __future__ import annotations

import unittest

from strata_cli.launch import settings as S
from strata_cli.launch.runner import fit_parallel_for
from tests.terminal.launch_support import HW_12, Home, catalog


class FitToCard(unittest.TestCase):
    def fit(self, family, model, context, vram):
        with Home():
            cat = catalog(HW_12)
            return fit_parallel_for(cat, S.LaunchSettings(family=family, model=model, context=context), vram)

    def test_qwen36_at_128k_on_24_gb_is_not_three_slots(self):
        n = self.fit("qwen36", "UD-Q4_K_XL", 131072, 24.0)
        self.assertLessEqual(n, 2)                                          # 15.14 GiB free / 5.56 GiB per slot: at most 2
        self.assertEqual(S.fit_parallel(24.0), 3)                           # what the table alone said

    def test_qwen36_follows_setups_rule(self):
        with Home():
            cat = catalog(HW_12)
            setup = cat.strata.setup_module()
            fam = setup.OWN_PATH_FAMILIES["qwen36"]
            for ctx, vram in ((8192, 12.0), (32768, 16.0), (131072, 24.0), (131072, 48.0), (32768, 48.0)):
                rec = setup.own_path_parallel_recommend(fam, "UD-Q4_K_XL", vram, ctx)
                got = fit_parallel_for(cat, S.LaunchSettings(family="qwen36", model="UD-Q4_K_XL", context=ctx), vram)
                self.assertEqual(got, max(1, min(S.fit_parallel(vram), rec or 1)), (ctx, vram))

    def test_flash_next_keeps_the_table(self):
        for vram in (12.0, 20.0, 24.0, 32.0):
            self.assertEqual(self.fit("qwen", "IQ3_S", 131072, vram), S.fit_parallel(vram))


if __name__ == "__main__":
    unittest.main()
