"""#5324: a quick tune measured the long (~96K) prompt and saved that speed as the preset's pp4k; `presets recheck` then compared its 4K
prompt with it and asked for a new tune every time. The preset keeps the 4K speed as pp4k and the long one apart (pp_long_tok_s), the
recheck compares 4K with 4K, and a preset saved before the split (no pp_long_tok_s key, its pp4k possibly a long prompt's) is not judged on
prompt speed (answers and memory still are)."""

import unittest

from strata_control import autotune
from tests.tuning.stub_session import StubSession
from tests.tuning.test_autotune import BASE, make_ctx


def preset(measured: dict) -> dict:
    return {"settings": {"args": {"--host-core": "last"}, "env": {}, "per_request": {}},
            "measured": {"decode_tok_s": 60.0, "rss_peak_mib": 12000.0, "vram_peak_mib": 10000.0, **measured}}


class RecheckComparesLikeWithLike(unittest.TestCase):
    def recheck(self, measured: dict, pp: float) -> dict:
        s = StubSession(model=lambda a, e, tune=None: 60.0, pp=lambda a, e: pp, noise=0.0)
        return autotune.recheck(s, BASE, preset(measured), ctx=make_ctx())

    def test_a_4k_rate_against_the_saved_4k_rate_keeps_the_preset(self):
        out = self.recheck({"pp4k_tok_s": 3000.0, "pp_long_tok_s": 4968.0}, pp=2998.0)
        self.assertEqual(out["verdict"], "keep", out["reason"])

    def test_a_real_4k_slowdown_is_still_found(self):
        out = self.recheck({"pp4k_tok_s": 3000.0, "pp_long_tok_s": 4968.0}, pp=2000.0)
        self.assertEqual(out["verdict"], "retune")
        self.assertIn("long prompts are slower", out["reason"])

    def test_an_old_preset_whose_pp4k_may_be_the_long_prompts_is_not_judged_on_it(self):
        out = self.recheck({"pp4k_tok_s": 4968.0}, pp=2998.0)               # the report: 4K ~3,000 against a saved ~4,968
        self.assertEqual(out["verdict"], "keep", out["reason"])

    def test_an_old_preset_is_still_judged_on_answer_speed(self):
        s = StubSession(model=lambda a, e, tune=None: 45.0, pp=lambda a, e: 2998.0, noise=0.0)
        out = autotune.recheck(s, BASE, preset({"pp4k_tok_s": 4968.0}), ctx=make_ctx())
        self.assertEqual(out["verdict"], "retune")
        self.assertIn("answers are", out["reason"])


class PresetKeepsBothRates(unittest.TestCase):
    def test_measured_has_the_4k_and_the_long_rate_apart(self):
        import os
        import tempfile
        from pathlib import Path
        from strata_control import presets
        from tests.tuning import test_presets as tp
        with tempfile.TemporaryDirectory() as d:
            os.environ["STRATA_PRESET_DIR"] = str(Path(d) / "presets")
            try:
                result = {**tp.RESULT, "chosen": {**tp.RESULT["chosen"], "pp_tok_s": 3000.0, "pp_long_tok_s": 4968.0}}
                saved = presets.create(result, choices=presets.choices_from_config(tp.CFG), identity=tp.identity(), name="x")
            finally:
                os.environ.pop("STRATA_PRESET_DIR", None)
        self.assertEqual((saved["measured"]["pp4k_tok_s"], saved["measured"]["pp_long_tok_s"]), (3000.0, 4968.0))


if __name__ == "__main__":
    unittest.main()
