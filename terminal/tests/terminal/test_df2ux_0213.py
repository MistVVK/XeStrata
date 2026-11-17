"""0.2.13 user-facing gaps of the tuned DFlash2 drafter (release 0.2.12 checks, RESULTS-DF2-DL): the plan names the drafter Auto will really fetch and its size, the
Terminal's plan sees a stock pack from 0.2.11 as "the tuned one is still to come", and a load nobody asked the drafter for can fetch with the 12 hour miss rule and no
words when the feed is not there."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from strata_cli.launch import settings as S
from strata_control import drafting, drafting_fetch
from tests.terminal.launch_support import HW_12, Home, catalog, run_cli

STRIX = ("hip", ["gfx1151"])
SPARK = ("cuda", ["sm_121"])
ARC = ("sycl", ["arc_a"])
Q4 = "qwen3.8-flash-next-unsloth-ud-q4_k_xl"


def hardware(pair):
    return mock.patch.object(drafting, "detect_hardware", return_value=pair)


def serving():
    return mock.patch.object(drafting, "ENGINE_SERVES_DFLASH2", True)


class PlanNamesTheRealDrafter(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.data = Path(self.tmp.name) / "data"
        (self.data / "dflash2").mkdir(parents=True)

    def plan(self, *argv):
        code, out, err = run_cli(["drafting", "plan", "--data-dir", str(self.data), "--output", "json", *argv])
        self.assertEqual(code, 0, err)
        return json.loads(out)

    def test_with_no_model_the_plan_is_the_old_generic_one(self):
        with serving(), hardware(STRIX):
            got = self.plan("--family", "unsloth", "--mode", "auto")
        self.assertEqual((got["gb"], got["tuned"]), (1.2, False))

    def test_a_model_not_installed_yet_gets_the_tuned_size(self):
        with serving(), hardware(STRIX):
            for mode in ("auto", "dflash2"):
                got = self.plan("--family", "unsloth", "--model", "UD-Q4_K_XL", "--mode", mode)
                self.assertEqual((got["needed"], got["tuned_needed"], got["tuned"], got["gb"], got["drafter_gb"]), (True, True, True, 0.34, 0.34), mode)
                self.assertEqual(got["file"], drafting.TUNED_NAME)
                self.assertIn("0.34 GB", got["text"])
                self.assertNotIn("1.2 GB", got["text"])

    def test_the_spark_and_arc_and_a_full_model_name(self):
        with serving(), hardware(SPARK):
            self.assertEqual(self.plan("--family", "qwen", "--model", "IQ3_S", "--mode", "auto")["gb"], 0.34)
        with serving(), hardware(ARC):
            got = self.plan("--family", "qwen", "--model", "IQ3_XXS", "--mode", "auto")
            self.assertEqual((got["gb"], got["file"]), (0.52, drafting.TUNED_BF16_NAME))
            self.assertEqual(self.plan("--model", "qwen3.8-flash-next-iq3_xxs", "--mode", "auto")["gb"], 0.52, "a run config's own name is taken as it is")

    def test_a_set_the_tuned_pack_is_not_for_keeps_the_stock_size(self):
        with serving(), hardware(STRIX):
            got = self.plan("--family", "qwen", "--model", "Q2_0", "--mode", "auto")
        self.assertEqual((got["tuned"], got["gb"], got["needed"]), (False, 1.2, True))

    def test_mtp_says_nothing_is_needed_whatever_the_model(self):
        with serving(), hardware(STRIX):
            got = self.plan("--family", "unsloth", "--model", "UD-Q4_K_XL", "--mode", "mtp")
            self.assertEqual((got["needed"], got["gb"], got["text"]), (False, 0.0, "Nothing to download."))
            _code, out, _err = run_cli(["drafting", "plan", "--data-dir", str(self.data), "--family", "unsloth", "--model", "UD-Q4_K_XL", "--mode", "mtp"])
            self.assertEqual(out.strip(), "Nothing to download.")

    def test_a_stock_pack_from_0_2_11_does_not_hide_the_tuned_download(self):
        with serving(), hardware(STRIX), mock.patch.object(drafting, "pack_ready", return_value=True):
            got = self.plan("--family", "unsloth", "--model", "UD-Q4_K_XL", "--mode", "auto")
        self.assertEqual((got["tuned_needed"], got["needed"], got["gb"]), (True, True, 0.34))


class TheLaunchPlan(unittest.TestCase):
    """`strata launch --plan` for a model that is not installed yet: the number is the tuned drafter's."""

    def setUp(self):
        from tests.terminal.launch_support import not_intel
        not_intel(self)

    def test_the_download_list_and_the_plan_line_use_the_tuned_pack(self):
        from strata_cli.launch.runner import Runner
        with Home():
            with serving(), hardware(STRIX):
                runner = Runner(S.LaunchSettings(family="unsloth", model="UD-Q4_K_XL", context=32768, vision="off"), catalog=catalog(HW_12))
                part = next(p for p in runner.download_plan()["parts"] if p["kind"] == "dflash2")
                self.assertEqual(part["gb"], 0.34)
                self.assertIn("the DFlash2 drafter (0.34 GB)", runner.download_plan()["text"])
                self.assertEqual(runner.drafting_plan_line(), "Drafting: DFlash2 (its drafter is downloaded first)")
                stock = Runner(S.LaunchSettings(family="qwen", model="Q2_0", context=32768, vision="off"), catalog=catalog(HW_12))
                self.assertEqual(next(p for p in stock.download_plan()["parts"] if p["kind"] == "dflash2")["gb"], 1.2)
                mtp = Runner(S.LaunchSettings(family="unsloth", model="UD-Q4_K_XL", context=32768, vision="off", drafting="mtp"), catalog=catalog(HW_12))
                self.assertNotIn("dflash2", [p["kind"] for p in mtp.download_plan()["parts"]])

    def test_a_stock_pack_already_here_still_lists_the_tuned_one(self):
        from strata_cli.launch.runner import Runner
        with Home():
            with serving(), hardware(STRIX), mock.patch.object(drafting, "pack_ready", return_value=True):
                runner = Runner(S.LaunchSettings(family="unsloth", model="UD-Q4_K_XL", context=32768, vision="off"), catalog=catalog(HW_12))
                self.assertEqual(next(p for p in runner.download_plan()["parts"] if p["kind"] == "dflash2")["gb"], 0.34)


class AnAutomaticFetch(unittest.TestCase):
    """`strata drafting fetch --if-needed --throttle`: what the app's plain load, restart and tuning run call."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.data = Path(self.tmp.name) / "data"
        (self.data / "dflash2").mkdir(parents=True)
        self.cfg = Path(self.tmp.name) / "run.json"
        self.cfg.write_text(json.dumps({"model_name": Q4, "args": []}), encoding="utf-8")

    def fetch(self, *extra):
        return run_cli(["drafting", "fetch", "--config", str(self.cfg), "--data-dir", str(self.data), "--if-needed", "--output", "jsonl", *extra])

    def test_the_feed_not_answering_is_silent_with_throttle_and_an_error_without(self):
        with serving(), hardware(STRIX), mock.patch.object(drafting_fetch, "ensure_for", return_value=None) as ensure:
            code, out, _ = self.fetch("--throttle")
            self.assertEqual(code, 0)
            last = json.loads(out.strip().splitlines()[-1])
            self.assertEqual((last["ok"], last["skipped"], last["pack"]), (True, True, None))
            self.assertTrue(ensure.call_args.kwargs["throttle"])
            code, out, _ = self.fetch()
            self.assertEqual(code, 5, "the Models screen's own Load keeps saying so")
            self.assertFalse(ensure.call_args.kwargs["throttle"])

    def test_a_miss_less_than_twelve_hours_old_skips_the_feed(self):
        (self.data / "dflash2" / drafting_fetch.MISS_MARK).write_text("x")
        # After the skipped tuned lookup the fetch falls back to the STOCK drafter (ensure): on a box with internet that is a real 1.2 GB download from
        # Hugging Face, or a hash and pack of the person's own Hugging Face cache (17 minutes and killed on the owner PC, 11-10). This test is about the
        # feed being skipped, so the stock path is kept offline here: no network, no Hugging Face cache.
        with serving(), hardware(STRIX), mock.patch.object(drafting_fetch, "tuned_from_feed") as feed, mock.patch.object(drafting_fetch.acquisition, "require"), \
                mock.patch.object(drafting_fetch, "online", return_value=False) as reach, mock.patch.object(drafting_fetch, "_hub_snapshots", return_value=[]):
            code, _out, _ = self.fetch("--throttle")
        reach.assert_called()                               # (the stock path was reached and stayed offline: nothing was downloaded)
        self.assertEqual(code, 0)
        feed.assert_not_called()

    def test_a_real_fault_is_still_reported(self):
        fault = drafting_fetch.DraftingError("The tuned DFlash2 drafter did not pass the safety check, so it was deleted.")
        with serving(), hardware(STRIX), mock.patch.object(drafting_fetch, "ensure_for", side_effect=fault):
            code, out, _ = self.fetch("--throttle")
        self.assertEqual(code, 5)
        self.assertIn("safety check", json.loads(out.strip().splitlines()[-1])["error"]["message"])

    def test_mtp_chosen_downloads_nothing(self):
        with serving(), hardware(STRIX), mock.patch.object(drafting_fetch, "ensure_for") as ensure:
            code, _out, _ = run_cli(["drafting", "fetch", "--config", str(self.cfg), "--data-dir", str(self.data), "--mode", "mtp", "--if-needed", "--throttle", "--output", "jsonl"])
        self.assertEqual(code, 0)
        ensure.assert_not_called()


class ABareFetch(AnAutomaticFetch):
    """`strata drafting fetch` with nothing named: the one installed model's run config decides (coordinator 11-07)."""

    def installed(self, *names):
        rows = [{"path": str(self.cfg), "model_name": n} for n in names]
        return mock.patch("strata_cli.launch.catalog.Catalog.installed_configs", return_value=rows)

    def bare(self):
        return run_cli(["drafting", "fetch", "--data-dir", str(self.data), "--output", "jsonl"])

    def test_one_installed_model_the_tuned_pack_is_fetched(self):
        with serving(), hardware(STRIX), self.installed(Q4), mock.patch.object(drafting_fetch, "ensure_for", return_value=self.data / "x") as ensure:
            code, _out, _ = self.bare()
        self.assertEqual(code, 0)
        self.assertEqual(ensure.call_args.kwargs["model_name"], Q4)
        self.assertEqual(ensure.call_args.kwargs["backend"], "hip")

    def test_none_installed_keeps_the_generic_fetch(self):
        with serving(), hardware(STRIX), self.installed(), mock.patch.object(drafting_fetch, "ensure_for", return_value=self.data / "x") as ensure:
            self.bare()
        self.assertIsNone(ensure.call_args.kwargs["model_name"])

    def test_several_models_ask_which_one_and_download_nothing(self):
        """Owner-approved wording (11-07)."""
        with serving(), hardware(STRIX), self.installed(Q4, "qwen3.8-flash-next-iq3_s"), mock.patch.object(drafting_fetch, "ensure_for") as ensure:
            code, out, _ = self.bare()
            code_text, _o, err = run_cli(["drafting", "fetch", "--data-dir", str(self.data)])
        self.assertEqual(code, 2)
        ensure.assert_not_called()
        said = json.loads(out.strip().splitlines()[-1])
        self.assertFalse(said["ok"])
        self.assertEqual(said["error"]["message"], "Several models are set up here. Say which one: strata drafting fetch --model <name>, or --config <file>.")
        self.assertEqual(code_text, 2)
        self.assertIn("Several models are set up here. Say which one: strata drafting fetch --model <name>, or --config <file>.", err)

    def test_naming_the_model_still_fetches_with_several_set_up(self):
        with serving(), hardware(STRIX), self.installed(Q4, "qwen3.8-flash-next-iq3_s"), mock.patch.object(drafting_fetch, "ensure_for", return_value=self.data / "x") as ensure:
            code, _out, _ = run_cli(["drafting", "fetch", "--config", str(self.cfg), "--data-dir", str(self.data), "--output", "jsonl"])
        self.assertEqual(code, 0)
        ensure.assert_called_once()

    def test_a_model_dflash2_is_not_for_is_not_picked(self):
        with serving(), hardware(STRIX), self.installed("qwen3.8-flash-next-coder-iq1_m"), mock.patch.object(drafting_fetch, "ensure_for", return_value=self.data / "x") as ensure:
            self.bare()
        self.assertIsNone(ensure.call_args.kwargs["model_name"])


if __name__ == "__main__":
    unittest.main()
