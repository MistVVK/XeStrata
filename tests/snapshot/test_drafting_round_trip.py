"""Drafting Off (or MTP), then Auto again, in one scratch world: Auto starts the engine exactly as Auto did before (the same drafter, --spec, --draft-conf and
batch), on every machine class and model set.  Found in the shipped 0.2.13.1 by the real-use rehearsal on an RTX PRO 5000 (REAL-USE-1, block (b)): after one
`strata launch --drafting off`, `strata launch --drafting auto` started with no drafter (the Off line, --spec 4 --batch 4) and stayed so.  There the launch named
--model-path, so setup ran at every launch: the Off launch's "--mtp off" left "mtp": false in the installed config, and Auto read that as "the draft layer is off".

The Auto line compared with is the second Auto launch of the world: on the UD sets the first launch fetches the plain DFlash2 pack and the next one converts it
for the selector (the same with Auto, Auto and nothing else), which is not what this test is about.  An explicit Off or MTP stays what it is, and a start
without a --drafting word after Off stays Off.  Nothing real is used (tests/snapshot/README.md: scratch HOME, data folder, install folder).

  python -m pytest tests/snapshot/test_drafting_round_trip.py -q -p no:cacheprovider
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tests.snapshot.driver import drive_launch  # noqa: E402
from tests.snapshot.matrix import MACHINES, MODEL_SETS  # noqa: E402
from tests.snapshot.normalize import normalize  # noqa: E402
from tests.snapshot.world import World  # noqa: E402


def launches(machine, mset, steps):
    """One world, one `strata launch` per (choice, setup_runs) step (choice None: no --drafting word), in order; the normalized records."""
    world = World(machine, mset).build()
    try:
        out = []
        for choice, runs in steps:
            with world.patched():
                rec = drive_launch(world, () if choice is None else ("--drafting", choice), setup_runs=runs)
            assert "failed" not in rec, rec.get("failed")
            out.append(normalize(rec, world))
        return out
    finally:
        world.close()


def argv(rec: dict) -> list[str]:
    return rec["engine"]["argv"]


def drafter(rec: dict) -> str:
    a = argv(rec)
    return "dflash2" if "--draft" in a else "mtp" if "--mtp" in a else "off"


class AutoAgainIsAuto(unittest.TestCase):
    maxDiff = None

    def cells(self, setup_runs=(False, True)):
        for runs in setup_runs:
            for m in MACHINES:
                for s in MODEL_SETS:
                    yield m, s, runs

    def round_trip(self, middle: str, back: str = "auto"):
        for m, s, runs in self.cells():
            with self.subTest(machine=m.key, set=s.key, setup_runs=runs):
                recs = launches(m, s, [("auto", runs), (back, runs), (middle, runs), (back, runs)])
                self.assertEqual(drafter(recs[2]), middle, f"the explicit {middle} stays {middle}")
                self.assertEqual(argv(recs[3]), argv(recs[1]), f"{back} after {middle} starts as {back} did before")
                self.assertEqual(recs[3]["config_file"].get("drafting"), back)

    def test_off_then_auto_is_auto(self):
        self.round_trip("off")

    def test_mtp_then_auto_is_auto(self):
        self.round_trip("mtp")

    def test_off_then_mtp_is_mtp(self):
        self.round_trip("off", back="mtp")

    def test_auto_after_off_runs_setup_again_when_the_install_would_be_reused(self):
        # setup ran at the Off launch only (a change of context, a --model-path), the Auto launch after it would reuse the installed copy as it is
        for m, s, _runs in self.cells(setup_runs=(True,)):
            with self.subTest(machine=m.key, set=s.key):
                recs = launches(m, s, [("auto", False), ("auto", False), ("off", True), ("auto", "when_needed")])
                self.assertEqual(argv(recs[3]), argv(recs[1]), "Auto after Off starts as Auto did before")

    def test_off_stays_off_at_a_start_without_a_drafting_word(self):
        for m, s, runs in self.cells(setup_runs=(True,)):
            if m.os == "macos":
                continue                                     # a Mac's setup takes no --mtp word (runner.setup_args): Off is not written there, as before
            with self.subTest(machine=m.key, set=s.key):
                recs = launches(m, s, [("auto", runs), ("off", runs), (None, runs)])
                self.assertEqual(argv(recs[2]), argv(recs[1]), "the Off of the launch before is kept (setup's \"mtp\": false)")


if __name__ == "__main__":
    unittest.main()
