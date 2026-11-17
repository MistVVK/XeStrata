"""A fixed expert cache beside a RAM budget on a Strix Halo (STRIX-2Q, 2026-11-01): UD-Q4_K_XL with --expert-cache 10000 or 5000 ended with
exit code 1 after 41 s and no error text.  The estimate (strata_control/uma_pool.py) is the measured one of the 128 GB box; setup lowers its
own budget with the person's cache and only warns about a budget the person set; a launch whose engine stopped says what to change.
No GPU, no network.

    python -m unittest tools.test_setup_uma_pool
"""
from __future__ import annotations

import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import setup  # noqa: E402
from strata_control import uma_pool  # noqa: E402

Q4_GB = setup.MODELS["UD-Q4_K_XL"]["arena_gb"]
STRIX = {"uma": True, "vram_gb": 96.0, "dedicated_gb": 64.0, "shared_gb": 32.0, "name": "Radeon 8060S"}   # the measured box: 92,058 MiB of pool


class Estimate(unittest.TestCase):
    def test_it_reproduces_the_measured_free_memory(self):
        # MiB free at the first window (STRIX-2Q-RESULTS): cache 14007 + budget 12 -> 8.6 GiB, 12735 + 12 -> 12.4, 12735 + 10 -> 16.5
        pool = 92058 / 1024
        for cache, budget, free in ((14007, 12, 8.6), (12735, 12, 12.4), (12735, 10, 16.5), (15283, 12, 4.8)):
            self.assertAlmostEqual(pool - uma_pool.demand_gib(cache, budget, Q4_GB), free, delta=0.7, msg=(cache, budget))

    def test_the_failing_settings_are_too_much_and_the_measured_ones_are_not(self):
        for cache in (10000, 5000):
            self.assertTrue(uma_pool.too_much(cache, 40, Q4_GB, 96.0), cache)         # exit code 1 after 41 s
        for cache, budget in ((10000, 12), (12735, 12), (14007, 12), (16223, 12), (12735, 10)):
            self.assertFalse(uma_pool.too_much(cache, budget, Q4_GB, 96.0), (cache, budget))   # started, no fault in 10+ starts each

    def test_the_budget_that_fits_is_below_the_failing_one_and_above_the_floor(self):
        for cache in (10000, 5000, 2000):
            fit = uma_pool.safe_budget_gib(cache, Q4_GB, 96.0)
            self.assertLess(fit, 40)
            self.assertGreaterEqual(fit, uma_pool.MIN_BUDGET_GIB)
            self.assertFalse(uma_pool.too_much(cache, fit, Q4_GB, 96.0))
        self.assertEqual(uma_pool.safe_budget_gib(24000, Q4_GB, 40.0), uma_pool.MIN_BUDGET_GIB)    # a tiny pool: the floor, never zero or negative

    def test_only_a_number_beside_a_budget_counts(self):
        self.assertEqual(uma_pool.fixed_cache_and_budget(["--expert-cache", "10000", "--resident-budget-gib", "40"]), (10000, 40.0))
        self.assertIsNone(uma_pool.fixed_cache_and_budget(["--expert-cache", "auto", "--resident-budget-gib", "40"]))
        self.assertIsNone(uma_pool.fixed_cache_and_budget(["--expert-cache", "10000"]))
        self.assertIsNone(uma_pool.stop_advice(["--expert-cache", "auto", "--resident-budget-gib", "40"]))


def run_guard(args, user=None, gpu=None):
    cfg = {"args": list(args)}
    if user is not None:
        cfg[setup.user_args.USER_KEY] = user
    out = io.StringIO()
    with redirect_stdout(out):
        setup.uma_cache_guard(cfg, "UD-Q4_K_XL", gpu or STRIX)
    return cfg, out.getvalue()


class SetupGuard(unittest.TestCase):
    def test_a_smaller_cache_lowers_setups_own_budget_and_says_so(self):
        cfg, text = run_guard(["--expert-cache", "10000", "--resident-budget-gib", "40", "--vision"])
        budget = float(cfg["args"][cfg["args"].index("--resident-budget-gib") + 1])
        self.assertLess(budget, 40)
        self.assertEqual(cfg["args"][cfg["args"].index("--expert-cache") + 1], "10000", "the person's cache is kept")
        self.assertEqual(cfg["args"][-1], "--vision", "nothing else moves")
        self.assertIn("10000 slots", text)
        self.assertIn("instead of 40", text)
        self.assertIn("--expert-cache auto", text)

    def test_a_budget_the_person_set_is_only_warned_about(self):
        cfg, text = run_guard(["--expert-cache", "10000", "--resident-budget-gib", "40"], user=["--expert-cache", "10000", "--resident-budget-gib", "40"])
        self.assertEqual(cfg["args"][cfg["args"].index("--resident-budget-gib") + 1], "40")
        self.assertIn("kept as you chose", text)

    def test_what_already_fits_and_other_pcs_are_left_alone(self):
        for args, gpu in ((["--expert-cache", "10000", "--resident-budget-gib", "12"], None),
                          (["--expert-cache", "auto", "--resident-budget-gib", "40"], None),
                          (["--expert-cache", "10000", "--resident-budget-gib", "40"], {**STRIX, "uma": False})):
            cfg, text = run_guard(args, gpu=gpu)
            self.assertEqual(cfg["args"], args)
            self.assertEqual(text, "")


class LaunchExplains(unittest.TestCase):
    def config(self, d, args, user=None):
        log = Path(d) / "engine.log"
        log.write_text("prefill: copy stream\n", encoding="utf-8")
        cfg = Path(d) / "run.json"
        cfg.write_text(json.dumps({"log": str(log), "args": args, **({"user_args": user} if user else {})}), encoding="utf-8")
        return cfg

    def test_a_silent_exit_code_1_gets_a_plain_next_step_on_a_shared_memory_pc(self):
        from strata_cli.launch import runner as R
        result = {"state": "failed", "error": "server_exited", "engine_exit_code": 1}
        with tempfile.TemporaryDirectory() as d:
            cfg = self.config(d, ["--expert-cache", "4000", "--resident-budget-gib", "40"], user=["--expert-cache", "10000"])
            text, nxt = R.explain_engine_exit(result, cfg, shared_memory=True)
            self.assertNotIn("copy stream", text, "a progress line is not a reason")
            self.assertIn("exit code 1", text)
            self.assertIn("10000 slots", nxt, "the person's own value is the one that counts")
            self.assertIn("--resident-budget-gib", nxt)
            self.assertIn("--expert-cache auto", nxt)
            plain_text, plain_next = R.explain_engine_exit(result, cfg, shared_memory=False)    # another PC: the old answer
            self.assertIn("report the log", plain_next)
            auto = self.config(d, ["--expert-cache", "auto", "--resident-budget-gib", "40"])
            self.assertIn("report the log", R.explain_engine_exit(result, auto, shared_memory=True)[1])


if __name__ == "__main__":
    unittest.main()
