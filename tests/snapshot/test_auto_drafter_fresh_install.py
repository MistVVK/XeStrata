"""A first `strata launch` with Auto (no Drafting word) on a computer without the draft layer: setup is told the drafter the start will run, so the start has it.
Found in the 0.2.14 real-use run on an RTX 5070 12 GB with IQ3_S (present since 0.2.13): Auto's start-time rule for that card class is the MTP
(drafting.CLASS_DEFAULTS, the "streaming" class, read from the card's memory), but setup was told "--drafter dflash2" (Auto resolved without the model's set and
the card's memory), skipped the draft layer, and the engine started with no drafter ("Drafting: off (no drafter is downloaded yet)"), at the next Auto launch too.
The plan said "Nothing to download".  Nothing real is used (tests/snapshot/README.md: scratch HOME, data folder, install folder).

  python -m pytest tests/snapshot/test_auto_drafter_fresh_install.py -q -p no:cacheprovider
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tests.snapshot.driver import drive_launch  # noqa: E402
from tests.snapshot.matrix import MACHINES, MODEL_SETS, machine as get_machine, model_set as get_set  # noqa: E402
from tests.snapshot.normalize import normalize  # noqa: E402
from tests.snapshot.world import World  # noqa: E402


def fresh_world(m, s) -> World:
    """A new install: the model files are there (setup's download is not what this is about), no run config and no draft layer yet."""
    world = World(m, s).build()
    world.config_path.unlink()
    for f in (world.mtp / "experts.bin", world.mtp / "draft_vocab.bin", world.data / "mtp" / "mtp-q2_0.gguf"):
        f.unlink()
    world.mtp.rmdir()
    return world


def first_launches(m, s, n=2):
    """`n` Auto launches without a Drafting word on a fresh install: the first runs setup, the later ones run it when ensure_model would (its own test)."""
    world = fresh_world(m, s)
    try:
        out = []
        for i in range(n):
            with world.patched():
                rec = drive_launch(world, (), setup_runs=True if i == 0 else "when_needed", fresh=True)
            assert "failed" not in rec, rec.get("failed")
            out.append(normalize(rec, world))
        return out
    finally:
        world.close()


def drafter(rec: dict) -> str:
    a = rec["engine"]["argv"]
    return "dflash2" if "--draft" in a else "mtp" if "--mtp" in a else "off"


def setup_drafter(rec: dict) -> str | None:
    a = rec["setup_args"]
    return a[a.index("--drafter") + 1] if "--drafter" in a else None


class AutoOnAFreshInstall(unittest.TestCase):
    maxDiff = None

    def test_12gb_iq3_s_auto_fetches_and_starts_the_mtp(self):
        first, second = first_launches(get_machine("nv_12gb_streaming"), get_set("iq3_s"))
        self.assertIsNone(setup_drafter(first), f"setup is told the MTP (no --drafter dflash2): {first['setup_args']}")
        self.assertIn("mtp", first["plan_parts"], "the plan lists the draft layer's download")
        self.assertIn({"call": "setup_mtp_layer"}, first["fetches"], "setup fetched the draft layer")
        self.assertEqual(drafter(first), "mtp", f"the first Auto start runs the MTP: {first['engine']['argv']}")
        self.assertEqual(drafter(second), "mtp", "the next Auto start too")
        self.assertEqual(second["engine"]["argv"], first["engine"]["argv"])

    def test_every_class_setup_is_told_the_drafter_the_start_runs(self):
        for m in MACHINES:
            if m.os == "macos":
                continue                                     # a Mac's setup takes no --drafter or --mtp word (runner.setup_args)
            for s in MODEL_SETS:
                with self.subTest(machine=m.key, set=s.key):
                    first, _second = first_launches(m, s)
                    self.assertNotEqual(drafter(first), "off", f"a first Auto start has a drafter: {first['engine']['argv']}")
                    self.assertEqual(setup_drafter(first) == "dflash2", drafter(first) == "dflash2",
                                     f"setup {first['setup_args']} / start {first['engine']['argv']}")
                    self.assertEqual("mtp" in first["plan_parts"], drafter(first) == "mtp", f"plan {first['plan_parts']}")


if __name__ == "__main__":
    unittest.main()
