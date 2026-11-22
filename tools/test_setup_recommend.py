# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Tests for setup.py's recommendations by quality and its lists' levels (the user's decisions of 2026-11-09): the
size of each kind of family a PC of some RAM and VRAM is offered, the model a first install is offered, and the
model, version and quantization that name a family.  Pure functions - no GPU, no downloads; the questions answered
with their defaults.

    python -m unittest tools.test_setup_recommend
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from typing import Any
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import setup  # noqa: E402


def room(ram: float, vram: float | None) -> setup.Room:
    """A PC of `ram` GB and one GPU of `vram` GB (None: no GPU)"""
    return setup.room_of(ram, [] if vram is None else [{"vram_gb": vram, "integrated": False}])


class SizeByQuality(unittest.TestCase):
    def test_gsq_rco_is_iq3_s_where_it_runs_else_iq3_xxs(self):
        self.assertEqual(setup.recommend_size("qwen", room(64, 16)), "IQ3_S")
        self.assertEqual(setup.recommend_size("qwen", room(48, 8)), "IQ3_XXS")     # in the low-RAM mode
        self.assertEqual(setup.recommend_size("sc117-swift", room(128, 24)), "IQ3_S")

    def test_gsq_rco_without_iq3_xxs_running_takes_the_best_that_runs(self):
        self.assertEqual(setup.recommend_size("qwen", room(48, 0)), "IQ2_XS")
        self.assertEqual(setup.recommend_size("qwen", room(36, 4)), "Q2_0")        # none runs: the smallest

    def test_a_ram_budget_size_needs_all_its_experts_in_ram(self):
        self.assertEqual(setup.recommend_size("unsloth", room(64, 16)), "UD-Q2_0")
        self.assertEqual(setup.recommend_size("unsloth", room(128, 24)), "UD-IQ4_XS")   # never the experimental
        self.assertEqual(setup.recommend_size("gyro", room(48, 8)), "Gyro-M")
        self.assertEqual(setup.recommend_size("atomicchat", room(64, 0)), "AD-5.00bpw")  # the same experts, 8-bit table

    def test_sync_does_not_change_other_families_entry_gate(self):
        self.assertEqual(setup.minimum_setup_ram("sync"), 24)
        self.assertEqual(setup.minimum_setup_ram("gyro"), 26)
        self.assertEqual(setup.minimum_setup_ram(None), 26)

    def test_sync_thresholds_and_recommendation(self):
        cases = [(23, 16, "Sync-XS"), (24, 16, "Sync-XS"), (31.99, 16, "Sync-XS"),
                 (32, 11.99, "Sync-XS"), (32, 12, "Sync-S"), (47.99, 11.99, "Sync-XS"),
                 (48, 0, "Sync-M"), (48, 12, "Sync-M")]
        for ram, vram, expected in cases:
            with self.subTest(ram=ram, vram=vram):
                self.assertEqual(setup.recommend_size("sync", room(ram, vram)), expected)
        self.assertFalse(setup.size_fits("sync", "Sync-XS", room(23.99, 16)))
        self.assertTrue(setup.size_fits("sync", "Sync-XS", room(24, 16)))
        self.assertFalse(setup.size_fits("sync", "Sync-S", room(32, 11.99)))
        self.assertTrue(setup.size_fits("sync", "Sync-S", room(48, 0)))
        self.assertEqual(setup.recommend_family(FirstModel.fams, room(64, 16)), "sc117-swift")

    def test_a_plain_moe_takes_the_best_that_ram_and_vram_hold_tested_or_not(self):
        self.assertEqual(setup.recommend_size("ornith", room(36, 4)), "Q5_K_M")    # untested, as the user chose
        self.assertEqual(setup.recommend_size("ornith", room(128, 24)), "Q8_0")    # never above Q8_0
        self.assertEqual(setup.recommend_size("llmjp", room(128, 24)), "Q4_K_M")   # nor its BF16 weights
        self.assertEqual(setup.recommend_size("qwen36-uncensored", room(36, 4)), "APEX-I-Quality")   # by its rank
        self.assertEqual(setup.recommend_size("glm53", room(36, 4)), "UD-IQ1_S")   # none fits: the smallest

    def test_a_dense_model_stops_at_q8_0(self):
        self.assertEqual(setup.recommend_size("qwen35", room(64, 24)), "Q8_0")
        self.assertEqual(setup.recommend_size("qwen35", room(64, 8)), "Q4_K_M")
        self.assertEqual(setup.recommend_size("qwen35-0.8b", room(16, 8)), "Q8_0")

    def test_on_the_cpu_the_familys_own_choice(self):
        self.assertEqual(setup.recommend_size("qwen35-2b", room(16, None)), "UD-Q5_K_XL")

    def test_one_quantizations_sizes(self):
        gsq = [m for m in setup.family_sizes("ornith") if setup.size_quantizer("ornith", m) == "gsq-rco-pfeifferj"]
        self.assertEqual(setup.recommend_size("ornith", room(36, 4), gsq), "GSQ-RCO-3.5BIT")


# Signal-2's and SenseNova-Loop-9B (QSA)'s entries as their branches register them (signal-2-setup 4dcba88997,
# sensenova-loop-qsa 966ff45a3e), for the steps of FIRST_CHOICE that name them
OTHER_BRANCHES: dict[str, dict[str, Any]] = {
    "MODELS": {"S": {"about": "Signal-2", "download_gb": 59.5, "ram_gb": 35.8, "arena_gb": 25.8,
                     "families": ("signal2",), "file": "Signal-2-Flash-Next-Gyro-S-TQ1_0.gguf", "shards": 1}},
    "FAMILIES": {"signal2": {"title": "Signal-2-Flash-Next (Gyro)", "by": "", "about": "", "file": "", "tag": "",
                             "name": "qwen3.8-flash-next-signal-2"},
                 "sensenova-loop-qsa": {"title": "SenseNova-Loop-9B (QSA)", "by": "", "about": "", "tag": "",
                                        "name": "sensenova-loop-9b-qsa", "max_context": 524288, "plain": True,
                                        "dense": True, "kv_bytes": 57344,
                                        "models": {"Q4_K_M": {"size": 5965458112, "about": ""},
                                                   "Q8_0": {"size": 9945475776, "about": ""},
                                                   "BF16": {"size": 18566736576, "about": ""},
                                                   "Q5_K_M": {"size": 6815328960, "about": ""}}}},
    "TIERS": {"signal2": ("qwen3.8-flash-next", "signal-2", "gyro"),
              "sensenova-loop-qsa": ("sensenova-loop-9b", "original", "qsa")},
}


def with_other_branches():
    """Signal-2 and SenseNova-Loop-9B (QSA) registered, as when their branches are in"""
    return mock.patch.multiple(setup, **{name: {**getattr(setup, name), **add} for name, add in OTHER_BRANCHES.items()})


class FirstModel(unittest.TestCase):
    # edge0, like CUSTOM, is outside the recommendation ladder (--family edge0, setup_edge0; setup's list excludes it)
    fams = [f for f in setup.FAMILIES if f not in (setup.CUSTOM, "edge0")]

    def first(self, ram: float, vram: float | None, fams=None) -> tuple:
        r = room(ram, vram)
        f = setup.recommend_family(fams or self.fams + list(OTHER_BRANCHES["FAMILIES"]), r)
        return f, setup.recommend_size(f, r)

    def test_the_first_this_pc_runs_in_the_users_order(self):
        # the user's order of 2026-11-22: SC117's Swift 1.5, Signal-2 where the RAM is plenty else Sync, SenseNova,
        # Qwen3.5-2B.  Every PC with the RAM for Signal-2 runs SC117's IQ3_XXS, so the order never reaches Signal-2
        # (the user is asked whether Signal-2 goes first there: then FIRST_CHOICE and the first case change)
        cases = [((123.2, 32), ("sc117-swift", "IQ3_S")),        # 128 GB installed: SC117 before Signal-2
                 ((62.7, 16), ("sc117-swift", "IQ3_S")),         # 64 GB installed
                 ((52, 0), ("sc117-swift", "IQ3_XXS")),          # IQ3_XXS in the low-RAM mode, IQ3_S not
                 ((48, 0), ("sync", "Sync-M")),                  # SC117's IQ2_XS only: not SC117
                 ((32, 12), ("sync", "Sync-S")),
                 ((24, 8), ("sync", "Sync-XS")),
                 ((20, 16), ("sensenova-loop-qsa", "Q8_0")),
                 ((20, 10), ("sensenova-loop-qsa", "Q5_K_M")),
                 ((20, 9), ("sensenova-loop-qsa", "Q4_K_M")),
                 ((20, 8), ("qwen35-2b", "Q8_0")),
                 ((2, 1), ("qwen35-2b", "Q4_K_M"))]               # nothing runs: the last choice
        with with_other_branches():
            for (ram, vram), expected in cases:
                with self.subTest(ram=ram, vram=vram):
                    self.assertEqual(self.first(ram, vram), expected)

    def test_signal2_where_the_ram_is_plenty_else_sync(self):
        # the step itself, SC117 aside: Signal-2's peak, 64.4 GiB, with LOW_RAM_HEADROOM_GB beside it
        fams = [f for f in self.fams if f != "sc117-swift"] + list(OTHER_BRANCHES["FAMILIES"])
        with with_other_branches():
            self.assertEqual(self.first(123.2, 32, fams), ("signal2", "S"))    # 128 GB installed
            self.assertEqual(self.first(92, 16, fams), ("signal2", "S"))       # 96 GB installed
            self.assertEqual(self.first(74.4, 16, fams), ("signal2", "S"))
            self.assertEqual(self.first(74.3, 16, fams), ("sync", "Sync-M"))
            self.assertEqual(self.first(62.7, 16, fams), ("sync", "Sync-M"))   # 64 GB installed

    def test_a_family_not_registered_is_passed_over(self):
        # this tree without Signal-2's and SenseNova's entries: SC117, Sync, Qwen3.5-2B
        self.assertEqual(self.first(123.2, 32, self.fams), ("sc117-swift", "IQ3_S"))
        self.assertEqual(self.first(48, 0, self.fams), ("sync", "Sync-M"))
        self.assertEqual(self.first(20, 16, self.fams), ("qwen35-2b", "Q8_0"))

    def test_without_a_gpu_the_2b(self):
        self.assertEqual(setup.recommend_family(["qwen35-2b", "qwen35-0.8b"], room(32, None)), "qwen35-2b")
        self.assertEqual(setup.recommend_family(self.fams, room(128, None)), "qwen35-2b")


class Levels(unittest.TestCase):
    def test_every_family_has_its_place(self):
        self.assertEqual(set(setup.TIERS), {f for f in setup.FAMILIES if f not in (setup.CUSTOM, "edge0")})
        self.assertEqual(len(set(setup.TIERS.values())), len(setup.TIERS))

    def test_the_levels_name_the_family(self):
        self.assertEqual(setup.tier_family("qwen3.8-flash-next", None, None), "qwen")
        self.assertEqual(setup.tier_family("qwen3.8-flash-next", None, "gyro"), "gyro")
        self.assertEqual(setup.tier_family("qwen3.8-flash-next", "huihui-ai", None), "huihui-unsloth")
        self.assertEqual(setup.tier_family("qwen3.8-flash-next", "swift-sc117", None), "sc117-swift")
        self.assertEqual(setup.tier_family("qwen3.8-flash-next", "coder", "gyro"), "gyro-coder")
        self.assertEqual(setup.tier_family("ornith-1.5", None, "gsq-rco-pfeifferj"), "ornith")

    def test_a_level_that_does_not_match_stops_with_the_choices(self):
        with mock.patch.object(setup, "fail", side_effect=SystemExit) as fail:
            with self.assertRaises(SystemExit):
                setup.tier_family("qwen3.8-flash-next", "swift", "unsloth")
        self.assertIn("gsq-rco", fail.call_args.args[1])

    def test_the_questions_default_to_the_recommendation(self):
        fams = [f for f in setup.FAMILIES if f != setup.CUSTOM]
        r = room(36, 4)
        with mock.patch.object(setup, "say"), \
                mock.patch.object(setup, "ask", side_effect=lambda q, choices, default, yes: default) as ask:
            family, quant = setup.ask_tiers(fams, setup.recommend_family(fams, r), r, None, True)
        self.assertEqual((family, quant), ("sync", None))
        self.assertEqual([c.args[0] for c in ask.call_args_list], ["Which model?", "Which version?",
                                                                   "Which quantization?"])

    def test_a_level_of_one_choice_is_not_asked(self):
        r = room(64, 16)
        with mock.patch.object(setup, "say"), \
                mock.patch.object(setup, "ask", side_effect=lambda q, choices, default, yes: default) as ask:
            family, quant = setup.ask_tiers(["llmjp"], "llmjp", r, None, True)
        self.assertEqual((family, quant), ("llmjp", None))
        ask.assert_not_called()


if __name__ == "__main__":
    unittest.main()
