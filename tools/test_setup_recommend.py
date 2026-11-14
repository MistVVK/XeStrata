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


class FirstModel(unittest.TestCase):
    # edge0, like CUSTOM, is outside the recommendation ladder (--family edge0, setup_edge0; setup's list excludes it)
    fams = [f for f in setup.FAMILIES if f not in (setup.CUSTOM, "edge0")]

    def test_the_first_this_pc_runs_in_the_users_order(self):
        self.assertEqual(setup.recommend_family(self.fams, room(64, 16)), "sc117-swift")
        self.assertEqual(setup.recommend_family(self.fams, room(48, 0)), "gyro-coder")    # IQ2_XS only: not Swift
        self.assertEqual(setup.recommend_family(self.fams, room(36, 4)), "gyro-coder")    # Gyro-Coder-L in RAM
        self.assertEqual(setup.recommend_family(self.fams, room(32, 8)), "ornith")        # Gyro-Coder alone is not
        self.assertEqual(setup.recommend_size("gyro-coder", room(36, 4)), "Gyro-Coder-L")
        self.assertEqual(setup.recommend_family(self.fams, room(16, 12)), "ornith")    # its GSQ-RCO 3-bit
        self.assertEqual(setup.recommend_family(self.fams, room(12, 8)), "qwen35")
        self.assertEqual(setup.recommend_family(self.fams, room(8, 4)), "qwen35-2b")

    def test_without_a_gpu_the_2b(self):
        self.assertEqual(setup.recommend_family(["qwen35-2b", "qwen35-0.8b"], room(32, None)), "qwen35-2b")


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
        self.assertEqual((family, quant), ("gyro-coder", None))
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
