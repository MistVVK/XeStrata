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


def offline_launch(m, s, with_pack=False):
    """A first Auto launch on a fresh install of a computer without internet: setup fetches no draft layer (setup.fetch_mtp offline goes on), the DFlash2
    drafter cannot be downloaded either. `with_pack`: the DFlash2 drafter is on disk already."""
    from unittest import mock
    from strata_control import drafting_fetch
    from tests.snapshot import driver
    world = fresh_world(m, s)
    try:
        with world.patched():
            if with_pack:
                drafting_fetch.ensure(world.data)
                world.fetches.clear()
            with mock.patch.object(driver, "setup_fetches_layer", lambda *a, **k: None), mock.patch.object(drafting_fetch, "online", lambda *a, **k: False):
                rec = drive_launch(world, (), setup_runs=True, fresh=True)
        return rec if "failed" in rec else normalize(rec, world)
    finally:
        world.close()


class PickedDrafterMissing(unittest.TestCase):
    maxDiff = None

    def test_12gb_layer_missing_with_the_dflash2_drafter_on_disk_starts_dflash2(self):
        rec = offline_launch(get_machine("nv_12gb_streaming"), get_set("iq3_s"), with_pack=True)
        self.assertNotIn("failed", rec)
        self.assertEqual(drafter(rec), "dflash2", f"the other drafter stands in: {rec['engine']['argv']}")

    def test_offline_with_nothing_on_disk_starts_without_a_drafter(self):
        for key in ("nv_12gb_streaming", "r9700_linux"):
            with self.subTest(machine=key):
                rec = offline_launch(get_machine(key), get_set("iq3_s"))
                self.assertNotIn("failed", rec, "an offline first launch never stops")
                self.assertEqual(drafter(rec), "off")

    def test_r9700_layer_missing_does_not_fall_back_to_dflash2(self):
        # Auto keeps the MTP on gfx1201 because of the DFlash2 prompt stall (drafting.AUTO_MTP): DFlash2 is no stand-in there
        rec = offline_launch(get_machine("r9700_linux"), get_set("iq3_s"), with_pack=True)
        self.assertNotIn("failed", rec)
        self.assertEqual(drafter(rec), "off")


class MacSetupFetchesThePickedDrafter(unittest.TestCase):
    # 0.2.14 SMOKE-1 (M5 Max, Q2_0, offline): the plan named DFlash2, setup fetched the MTP layer anyway (setup_args left --drafter out on a Mac) and stopped offline
    def test_metal_sets(self):
        from tests.snapshot.matrix import ModelSet
        q2 = ModelSet("q2_0", "qwen", "Q2_0", "qwen3.8-flash-next-q2_0", ("Qwen3.8-Flash-Next-Q2_0-00001-of-00002.gguf", "Qwen3.8-Flash-Next-Q2_0-00002-of-00002.gguf"))
        for mkey in ("mac_small", "mac_large"):
            for s, want in ((get_set("iq3_s"), None), (q2, "dflash2"), (get_set("ud_q4_k_xl"), "dflash2"), (get_set("ud_iq4_xs_v31"), "dflash2")):
                with self.subTest(machine=mkey, set=s.key):
                    first, = first_launches(get_machine(mkey), s, n=1)
                    self.assertEqual(setup_drafter(first), want, f"setup is told the drafter the start runs: {first['setup_args']}")
                    self.assertEqual("mtp" in first["plan_parts"], want is None, f"plan {first['plan_parts']}")


class SavedOffHolds(unittest.TestCase):
    def test_a_saved_off_survives_a_start_without_a_drafting_word_when_setup_does_not_run(self):
        # 0.2.14 real-use run, Strix T6: the installed copy was reused, the Off was only in the launch's own run config, the next start ran DFlash2
        from tests.snapshot.test_drafting_round_trip import launches
        for m in MACHINES:
            for s in MODEL_SETS:
                with self.subTest(machine=m.key, set=s.key):
                    recs = launches(m, s, [("auto", False), ("off", False), (None, False)])
                    self.assertEqual(drafter(recs[1]), "off")
                    self.assertEqual(recs[2]["engine"]["argv"], recs[1]["engine"]["argv"], "the Off of the launch before holds")


if __name__ == "__main__":
    unittest.main()
