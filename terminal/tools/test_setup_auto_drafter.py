"""P14 (0.2.16): setup uses the Terminal's Auto drafter rule.

USERTEST-LIN L2 (p1004, a P100): `./setup.sh` fetched the 5 GB MTP draft layer, where the Terminal's Auto picks DFlash2 (1.2 GB) on the same
computer. setup.auto_drafter asks strata_control.drafting.choice_of (the rule `strata launch` and the app use) and, when it says DFlash2 and the
MTP layer is not on disk, setup fetches the DFlash2 drafter instead of the layer; if that fetch fails the MTP layer is fetched as before.

    python -m unittest tools.test_setup_auto_drafter
"""
from __future__ import annotations

import argparse
import contextlib
import io
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import setup  # noqa: E402
from strata_control import drafting, drafting_fetch  # noqa: E402

QWEN = {"name": "qwen3.8-flash-next"}
P100 = {"vram_gb": 16.0, "arch": 60}


def args(**kw):
    return argparse.Namespace(**{"drafter": None, "mtp": None, **kw})


class AutoDrafter(unittest.TestCase):
    def pick(self, a=None, *, family="qwen", model="IQ3_XXS", hip=False, gpu=P100, mtp_here=False, hardware=("cuda", ["sm_60"])):
        with mock.patch.object(drafting, "hardware_for", return_value=hardware):
            return setup.auto_drafter(a or args(), family, QWEN, model, hip, gpu, mtp_here)

    def test_a_p100_gets_dflash2_as_the_terminal_does(self):
        self.assertEqual(self.pick(), "dflash2")
        # the same answer the Terminal's rule gives for the same model name, backend and card
        self.assertEqual(drafting.choice_of("auto", model_name="qwen3.8-flash-next-iq3_xxs", family="qwen", backend="cuda", archs=["sm_60"]), "dflash2")

    def test_it_asks_the_terminals_rule_and_follows_its_answer(self):
        with mock.patch.object(drafting, "choice_of", return_value="mtp") as rule, mock.patch.object(drafting, "hardware_for", return_value=("cuda", ["sm_86"])):
            self.assertIsNone(setup.auto_drafter(args(), "qwen", QWEN, "IQ3_S", False, P100, False))
        kw = rule.call_args.kwargs
        self.assertEqual((rule.call_args.args, kw["model_name"], kw["family"], kw["backend"], kw["archs"]),
                         (("auto",), "qwen3.8-flash-next-iq3_s", "qwen", "cuda", ["sm_86"]))

    def test_the_cards_where_auto_keeps_the_mtp_keep_it_in_setup_too(self):
        self.assertIsNone(self.pick(model="IQ3_S", hip=True, hardware=("hip", ["gfx1201"])))       # AUTO_MTP: the R9700

    def test_a_model_dflash2_is_not_for_is_left_alone(self):
        self.assertIsNone(self.pick(family="swift", model="Q2_0"))

    def test_nothing_changes_when_the_person_named_a_drafter_or_the_layer_is_here(self):
        self.assertIsNone(self.pick(args(drafter="mtp")))
        self.assertIsNone(self.pick(args(drafter="dflash2")))      # already the explicit route
        self.assertIsNone(self.pick(args(mtp="off")))
        self.assertIsNone(self.pick(args(mtp="on")))
        self.assertIsNone(self.pick(mtp_here=True))                # the layer is on disk: nothing to save, no extra download

    def test_an_unreadable_rule_leaves_setup_as_it_was(self):
        with mock.patch.object(drafting, "hardware_for", side_effect=RuntimeError("no probe")):
            self.assertIsNone(setup.auto_drafter(args(), "qwen", QWEN, "IQ3_XXS", False, P100, False))


class FetchDrafter(unittest.TestCase):
    def setUp(self):
        for name in ("say", "ok"):
            p = mock.patch.object(setup, name, lambda *a, **k: None)
            p.start()
            self.addCleanup(p.stop)
        self.warned = []
        p = mock.patch.object(setup, "warn", self.warned.append)
        p.start()
        self.addCleanup(p.stop)
        p = mock.patch.object(drafting, "hardware_for", return_value=("cuda", ["sm_60"]))
        p.start()
        self.addCleanup(p.stop)

    def call(self, **effects):
        with tempfile.TemporaryDirectory() as d, mock.patch.object(drafting_fetch, "ensure_for", **effects) as ensure:
            got = setup.fetch_dflash2_drafter(Path(d), QWEN, "IQ3_XXS", False, P100)
        return got, ensure

    def test_the_drafter_is_fetched_with_the_terminals_function(self):
        got, ensure = self.call(return_value=Path("pack"))
        self.assertTrue(got)
        self.assertEqual(self.warned, [])
        kw = ensure.call_args.kwargs
        self.assertEqual((kw["model_name"], kw["backend"], kw["archs"]), ("qwen3.8-flash-next-iq3_xxs", "cuda", ["sm_60"]))
        self.assertEqual(kw["vram_mib"], 16.0 * 1024)

    def test_offline_falls_back_to_the_mtp_layer_with_a_warning(self):
        got, _ = self.call(side_effect=drafting_fetch.OfflineError("There is no internet connection, so the DFlash2 drafter can't be downloaded now."))
        self.assertFalse(got)
        self.assertIn("fetching the MTP draft layer instead", self.warned[0])

    def test_no_answer_and_a_failed_check_fall_back_too(self):
        got, _ = self.call(return_value=None)
        self.assertFalse(got)
        got, _ = self.call(side_effect=drafting_fetch.DraftingError("checksum differs"))
        self.assertFalse(got)
        self.assertEqual(len(self.warned), 2)

    def test_a_drafter_that_is_already_there_prints_no_sizes(self):
        # 0.2.16 retest F4: "The DFlash2 drafter is ready. 0.00 of 0.00 GB" (the ready line reports 1 of 1 byte).
        def present(data, progress=None, **_):
            progress("The DFlash2 drafter is ready.", 1, 1)
            return Path("pack")
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            got, _ = self.call(side_effect=present)
        self.assertTrue(got)
        self.assertNotIn("GB", out.getvalue())
        self.assertNotIn("0.00", out.getvalue())

    def test_a_shorter_line_leaves_no_tail_of_the_longer_one(self):
        # 0.2.17 item 18 (p1004): "The DFlash2 drafter is ready. 1.21 of 1.21 GB.21 of 1.21 GB": the \r rewrite of a longer line was not cleared.
        def steps(data, progress=None, **_):
            progress("Downloading the DFlash2 drafter (1.2 GB, once)", 1_210_000_000, 1_210_000_000)
            progress("Packing the DFlash2 drafter for the engine", 1_210_000_000, 1_210_000_000)
            progress("Ready.", 1_210_000_000, 1_210_000_000)
            return Path("pack")
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            got, _ = self.call(side_effect=steps)
        self.assertTrue(got)
        parts = out.getvalue().split("\r")[1:]
        self.assertEqual(len(parts), 3)
        self.assertEqual(len({len(p) for p in parts}), 1, "every rewrite is as long as the longest line so far")
        # what the screen shows (a terminal overwrites from the left and keeps the rest of the old line)
        screen = ""
        for p in parts:
            screen = p + screen[len(p):]
        self.assertEqual(screen.rstrip(), "  Ready. 1.21 of 1.21 GB")

    def test_a_real_download_still_shows_its_sizes(self):
        def download(data, progress=None, **_):
            progress("Downloading the DFlash2 drafter", 340_000_000, 522_000_000)
            return Path("pack")
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            got, _ = self.call(side_effect=download)
        self.assertTrue(got)
        self.assertIn("0.34 of 0.52 GB", out.getvalue())


class Wiring(unittest.TestCase):
    def test_step_6_drops_to_the_mtp_layer_when_the_drafter_cannot_be_had(self):
        text = (ROOT / "setup.py").read_text(encoding="utf-8")
        auto = text.index("auto_drafter(a, family, fam, model, hip, gpu, mtp_have)")
        self.assertLess(text.index("mtp_have = find_in(roots"), auto)
        self.assertLess(auto, text.index("mtp_fetch = 5.2 if"))                                      # before the disk-room estimate
        fetch = text.index("fetch_dflash2_drafter(data, fam, model, hip, gpu)")
        self.assertLess(fetch, text.index('elif getattr(a, "drafter", None) == "dflash2" and not (rt / "experts.bin").exists():'))
        self.assertIn("a.drafter = None", text[fetch:fetch + 400])


if __name__ == "__main__":
    unittest.main()
