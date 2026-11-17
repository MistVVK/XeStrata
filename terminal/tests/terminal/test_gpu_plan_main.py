"""The plan's Main card is the run's Main card: `strata launch --plan` (strata_cli/launch/cmd.py roles_line -> gpu_roles.from_legacy) and the start
(runner.apply_cards -> serve/server.py ordered_gpus) take the card order from ONE function, gpu_roles.speed_stage_order (5% rule #1760, faster card last
#1352, VRAM guard #1576).  Faked cards, nothing is started or probed.

Card sets (speed = multiprocessors x max clock):
  control  four Tesla V100 (32, 32, 16, 16 GB), equal speed
  A        card 0 = 16 GB, speed 171 (sm_120); card 1 = 24 GB, speed 150 (sm_89): the FASTER card has LESS memory
  B        card 0 = 16 GB, speed 150; card 1 = 24 GB, speed 171: the faster card has MORE memory
  C        card 0 = 12 GB, card 1 = 16 GB, both speed 150: equal speed, different memory
Before the fix the plan named card 0 in A typed 0,1 while the start ended on card 1 (the #1576 guard lives only in the start).
A typed --gpu after an earlier launch with roles (a saved "gpu_order": "as_given") means the same as on a fresh home: the engine decides
(owner 11-11, "Ok for --gpu list"; before, apply_cards kept the old key and the start kept the typed order where the plan re-sorted).
"""

from __future__ import annotations

import contextlib
import io
import os
import re
import unittest
from unittest import mock

from serve import server
from strata_cli.launch import cmd as launch_cmd
from strata_cli.launch import runner
from strata_cli.launch import settings as S
from strata_control import gpu_roles
from tests.terminal.launch_support import HW_12, Home, catalog


def card(i, name, gb, speed):
    return {"index": i, "name": name, "vram_gb": float(gb), "speed": speed, "usable": True, "vendor": "nvidia"}


SETS = {
    "control": [card(0, "Tesla V100-SXM2-32GB", 32, 122.4), card(1, "Tesla V100-SXM2-32GB", 32, 122.4),
                card(2, "Tesla V100-SXM2-16GB", 16, 122.4), card(3, "Tesla V100-SXM2-16GB", 16, 122.4)],
    "A": [card(0, "NVIDIA GeForce RTX 5070 Ti", 16, 171.0), card(1, "NVIDIA RTX 4500 Ada Generation", 24, 150.0)],
    "B": [card(0, "NVIDIA RTX 4000 Ada Generation", 16, 150.0), card(1, "NVIDIA GeForce RTX 5090 D", 24, 171.0)],
    "C": [card(0, "NVIDIA GeForce RTX 4070", 12, 150.0), card(1, "NVIDIA GeForce RTX 4070 Ti SUPER", 16, 150.0)],
}
# (set, typed --gpu order) -> the card (nvidia-smi number) the run makes Main, pinned
START_MAIN = {
    ("control", (0, 1, 2, 3)): 3,
    ("A", (0, 1)): 1,       # the faster 16 GB card would go last, the guard keeps the typed order: the 24 GB card is Main
    ("A", (1, 0)): 0,
    ("B", (0, 1)): 1,
    ("B", (1, 0)): 1,       # reordered: the faster card last
    ("C", (0, 1)): 1,
    ("C", (1, 0)): 0,
}


def plan_main(cards, typed):
    s = S.LaunchSettings(family="qwen", model="UD-Q4_K_XL", gpus=list(typed))
    c = catalog({**HW_12, "engine_vendor": "nvidia", "gpus": cards})
    with mock.patch.dict(os.environ, {}):
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        line = launch_cmd.roles_line(s, c)
    m = re.search(r"Main: GPU \d+ \(--gpu (\d+)\)", line or "")
    assert m, line
    return int(m.group(1)), line


def start_main(cards, typed, base=None):
    """The saved config of a typed --gpu on a fresh home (no earlier gpu_order) or on `base` (a config an earlier launch saved), then the
    order the engine stages the cards in."""
    cfg = dict(base) if base else {"layer_split": "auto"}
    runner.apply_cards(cfg, list(typed))
    scores = {c["index"]: c["speed"] for c in cards}
    vrams = {c["index"]: int(c["vram_gb"] * 1024) for c in cards}
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        order = server.ordered_gpus(cfg, scores=scores, vrams=vrams)
    return order[-1], order, buf.getvalue().strip()


class PlanNamesTheRunsMain(unittest.TestCase):
    def setUp(self):
        self.home = Home().__enter__()                  # a private install / data / home folder (STRATA_INSTALL_DIR ...), as the other launch tests
        self.addCleanup(self.home.__exit__, None, None, None)

    def test_every_row_the_plan_equals_the_start(self):
        for (name, typed), want in START_MAIN.items():
            with self.subTest(cards=name, typed=typed):
                p, line = plan_main(SETS[name], typed)
                s, order, _ = start_main(SETS[name], typed)
                self.assertEqual(s, want, f"the start's Main changed ({order})")
                self.assertEqual(p, s, f"the plan says '{line}', the start ends on card {s} ({order})")

    def test_a_typed_list_after_a_roles_launch_means_the_same_as_on_a_fresh_home(self):
        for (name, typed), want in START_MAIN.items():
            n = len(SETS[name])
            roles = {"gpu": list(range(n)), "gpu_order": "as_given", "layer_split": "auto",
                     "gpu_roles": {str(i): ("main" if i == n - 1 else "helper") for i in range(n)}}
            with self.subTest(cards=name, typed=typed):
                s, order, _ = start_main(SETS[name], typed, base=roles)
                p, line = plan_main(SETS[name], typed)
                self.assertEqual(s, want, f"the start's Main after a roles launch ({order})")
                self.assertEqual(p, s, f"the plan says '{line}', the start ends on card {s} ({order})")
        cfg = {"gpu": [0, 1], "gpu_order": "as_given", "gpu_roles": {"0": "helper", "1": "main"}}
        runner.apply_cards(cfg, [1, 0])
        self.assertNotIn("gpu_order", cfg)
        self.assertNotIn("gpu_roles", cfg)

    def test_roles_still_keep_their_order(self):
        cfg = {"gpu": [1, 0]}
        runner.apply_cards(cfg, {"gpu": [1, 0], "gpu_order": "as_given", "gpu_roles": {"0": "main", "1": "helper"}})
        self.assertEqual(cfg.get("gpu_order"), "as_given")

    def test_the_help_line_says_who_picks_the_main_card(self):
        import argparse
        parsers = launch_cmd.add_parsers(argparse.ArgumentParser().add_subparsers())
        helps = [a.help for p in parsers for a in p._actions if "--gpu" in a.option_strings]
        self.assertTrue(helps)
        text = helps[0]
        self.assertEqual(text, "graphics card numbers to use, e.g. 0 or 0,1 (default: the engine's choice). With more than one card the engine "
                               "decides which is the Main card; --gpu-roles (instead of --gpu) sets the roles yourself.")

    def test_the_vram_guard_message_is_unchanged(self):
        _, order, said = start_main(SETS["A"], (0, 1))
        self.assertEqual(order, [0, 1])
        self.assertEqual(said, "[strata] layer split: keeping card order 0,1 - the faster card has less VRAM than the one the split ends on (16384 vs 24576 MiB, #1576)")

    def test_the_reorder_message_is_unchanged(self):
        _, order, said = start_main(SETS["B"], (1, 0))
        self.assertEqual(order, [0, 1])
        self.assertEqual(said, '[strata] layer split: card order 0,1 (the faster card last; "gpu_order": "as_given" keeps 1,0, #1352)')

    def test_equal_cards_print_nothing_and_keep_the_typed_order(self):
        for name, typed in (("control", (0, 1, 2, 3)), ("C", (0, 1)), ("C", (1, 0))):
            _, order, said = start_main(SETS[name], typed)
            self.assertEqual((order, said), (list(typed), ""))


class SpeedStageOrder(unittest.TestCase):
    def f(self, *a):
        return gpu_roles.speed_stage_order(*a)

    def test_why(self):
        self.assertEqual(self.f([0, 1], {0: 100.0, 1: 104.9}), ([0, 1], "same", None))                    # within 5%
        self.assertEqual(self.f([0, 1], {0: 100.0, 1: 105.0}), ([0, 1], "kept", None))                    # already faster last
        self.assertEqual(self.f([1, 0], {0: 100.0, 1: 105.0}, {0: 8192, 1: 8192}), ([0, 1], "faster-last", None))
        self.assertEqual(self.f([0, 1], {0: 105.0, 1: 100.0}, {0: 8192, 1: 16384}), ([0, 1], "kept-vram", (8192, 16384)))
        self.assertEqual(self.f([0, 1], {0: 105.0, 1: 100.0}, {0: 16384, 1: 8192}), ([1, 0], "faster-last", None))
        self.assertEqual(self.f([0, 1], {0: 105.0, 1: 100.0}, {0: 8192}), ([1, 0], "faster-last", None))   # a card without a VRAM figure: no guard
        self.assertEqual(self.f([3], {3: 1.0}), ([3], "same", None))

    def test_the_vram_figures_are_asked_only_when_a_reorder_is_on_the_table(self):
        asked = []

        def vram():
            asked.append(1)
            return {0: 8192, 1: 8192}
        self.f([0, 1], {0: 100.0, 1: 101.0}, vram)                  # same speed: no question
        self.f([0, 1], {0: 100.0, 1: 150.0}, vram)                  # already faster last: no question
        self.assertEqual(asked, [])
        self.f([1, 0], {0: 100.0, 1: 150.0}, vram)                  # a reorder: asked once
        self.assertEqual(asked, [1])


if __name__ == "__main__":
    unittest.main()
