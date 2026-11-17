"""A drafter the person chose stays chosen (owner 11-15, item 7 of lane FIX-0215A): MTP or DFlash2 picked by hand (`--drafting`, the TUI's Drafting step, the app's
Drafting setting) is kept by a later restart or launch with no Drafting word, the way Off is; only a new explicit choice changes it, Auto included.  The drafter
rule holds: a chosen drafter whose file is not here and cannot be fetched (offline) starts the other drafter when that one is on disk and safe on the card, with
the existing note; the saved choice stays what the person set, never the fallback; on gfx1201 DFlash2 never stands in.  Nothing real is used
(tests/snapshot/README.md).

  python -m pytest tests/snapshot/test_drafting_chosen_holds.py -q -p no:cacheprovider
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tests.snapshot.driver import drive_launch  # noqa: E402
from tests.snapshot.matrix import MACHINES, MODEL_SETS, machine as get_machine, model_set as get_set  # noqa: E402
from tests.snapshot.normalize import normalize  # noqa: E402
from tests.snapshot.test_drafting_round_trip import argv, drafter, launches  # noqa: E402
from tests.snapshot.world import World, context_for  # noqa: E402

FLASH = [s for s in MODEL_SETS if s.family in ("qwen", "unsloth", "gyro")]


def setup_drafter(rec: dict) -> str | None:
    a = rec.get("setup_args") or []
    return a[a.index("--drafter") + 1] if "--drafter" in a else None


class ChosenHolds(unittest.TestCase):
    maxDiff = None

    def test_a_chosen_mtp_or_dflash2_holds_over_restarts_with_no_word(self):
        for m in MACHINES:
            for s in FLASH:
                for word in ("mtp", "dflash2"):
                    with self.subTest(machine=m.key, set=s.key, word=word):
                        recs = launches(m, s, [(word, False), (word, False), (None, False), (None, False)])
                        for later in recs[2:]:
                            self.assertEqual(later["config_file"].get("drafting"), word, "the saved choice stays as the person set it")
                            self.assertEqual(argv(later), argv(recs[1]), "a restart with no word starts as one with the word")

    def test_only_a_new_explicit_choice_changes_it_auto_included(self):
        for mkey, skey in (("nv_resident", "iq3_s"), ("nv_12gb_streaming", "ud_q4_k_xl"), ("strix_halo_win", "ud_q4_k_xl"), ("mac_large", "ud_q4_k_xl")):
            with self.subTest(machine=mkey, set=skey):
                m, s = get_machine(mkey), get_set(skey)
                recs = launches(m, s, [("auto", False), ("auto", False), ("mtp", False), ("auto", False), (None, False)])
                self.assertEqual(drafter(recs[2]), "mtp")
                self.assertEqual(recs[4]["config_file"].get("drafting"), "auto", "Auto chosen again replaces the MTP")
                self.assertEqual(argv(recs[4]), argv(recs[1]), "and starts as Auto did")
                recs = launches(m, s, [("dflash2", False), ("off", False), (None, False)])
                self.assertEqual(drafter(recs[2]), "off", "Off chosen after a drafter holds")

    def test_setup_is_told_the_chosen_drafter_at_a_start_with_no_word(self):
        # setup runs again (a --model-path, a change): it is told the drafter the person chose, not Auto's
        for mkey, skey in (("nv_resident", "iq3_s"), ("nv_resident", "ud_q4_k_xl"), ("strix_halo_win", "ud_q4_k_xl")):
            with self.subTest(machine=mkey, set=skey):
                recs = launches(get_machine(mkey), get_set(skey), [("mtp", True), (None, True)])
                self.assertEqual(setup_drafter(recs[1]), setup_drafter(recs[0]))
                self.assertEqual(drafter(recs[1]), "mtp")

    def test_the_models_screen_choice_still_decides_and_is_not_overwritten(self):
        def app_says_dflash2(world):
            cfg = json.loads(world.config_path.read_text(encoding="utf-8"))
            cfg["drafting"] = "dflash2"
            world.config_path.write_text(json.dumps(cfg, indent=1), encoding="utf-8")
        recs = launches(get_machine("nv_resident"), get_set("ud_q4_k_xl"), [("mtp", False), (None, False)], prepare=app_says_dflash2)
        self.assertEqual(recs[0]["config_file"].get("drafting"), "mtp", "the word given wins at its own launch")
        self.assertEqual(recs[1]["config_file"].get("drafting"), "dflash2", "no word: the installed config's own choice (a later one) decides")


def chosen_then_offline(m, s, word, *, other_on_disk: bool):
    """`--drafting <word>` online, then the chosen drafter's file is gone and this computer is offline; a restart with no word.  `other_on_disk`: the other
    drafter is on this computer."""
    from strata_control import drafting_fetch
    from tests.snapshot import driver
    world = World(m, s).build()
    try:
        with world.patched():
            first = drive_launch(world, ("--drafting", word), setup_runs=False)
            assert "failed" not in first, first.get("failed")
            if word == "mtp":
                for f in (world.mtp / "experts.bin", world.data / "mtp" / "mtp-q2_0.gguf"):
                    f.unlink(missing_ok=True)
                if other_on_disk:
                    drafting_fetch.ensure(world.data)
            else:
                for f in sorted((world.data / "dflash2").glob("*.pack*")):
                    f.unlink()
                if not other_on_disk:
                    (world.mtp / "experts.bin").unlink(missing_ok=True)
            world.fetches.clear()
            # offline: nothing can be fetched (the world's own fakes of the drafter download write the file whatever `online` says)
            with mock.patch.object(driver, "setup_fetches_layer", lambda *a, **k: None), mock.patch.object(drafting_fetch, "online", lambda *a, **k: False), \
                    mock.patch.object(drafting_fetch, "ensure", lambda *a, **k: None), mock.patch.object(drafting_fetch, "ensure_tuned", lambda *a, **k: None), \
                    mock.patch.object(drafting_fetch, "convert_selector", lambda *a, **k: None):
                rec = drive_launch(world, (), setup_runs=False)
        assert "failed" not in rec, rec.get("failed")
        return normalize(rec, world)
    finally:
        world.close()


class ChosenOffline(unittest.TestCase):
    maxDiff = None

    def test_a_chosen_mtp_without_its_layer_starts_dflash2_from_disk_and_stays_mtp(self):
        for mkey in ("nv_resident", "nv_12gb_streaming", "strix_halo_win"):
            with self.subTest(machine=mkey):
                rec = chosen_then_offline(get_machine(mkey), get_set("iq3_s"), "mtp", other_on_disk=True)
                self.assertEqual(drafter(rec), "dflash2", rec["engine"]["argv"])
                self.assertEqual(rec["config_file"].get("drafting"), "mtp", "the saved choice is the person's, not the fallback")

    def test_on_gfx1201_dflash2_never_stands_in(self):
        rec = chosen_then_offline(get_machine("r9700_linux"), get_set("iq3_s"), "mtp", other_on_disk=True)
        self.assertEqual(drafter(rec), "off", "no safe drafter on disk there: no DFlash2 stand-in")
        self.assertEqual(rec["config_file"].get("drafting"), "mtp")

    def test_a_chosen_dflash2_without_its_drafter_starts_the_mtp_and_stays_dflash2(self):
        for mkey in ("nv_resident", "strix_halo_win"):
            with self.subTest(machine=mkey):
                rec = chosen_then_offline(get_machine(mkey), get_set("ud_q4_k_xl"), "dflash2", other_on_disk=True)
                self.assertEqual(drafter(rec), "mtp", rec["engine"]["argv"])
                self.assertEqual(rec["config_file"].get("drafting"), "dflash2")


class TuiStep(unittest.TestCase):
    """The TUI's Drafting step after a launch with a chosen drafter: it shows the choice that holds, and an Auto picked there is written, so it replaces it."""

    def step(self, world, pick):
        from strata_cli.launch import settings as S
        from strata_cli.launch.screens import DraftingScreen
        base = json.loads(world.config_path.read_text(encoding="utf-8"))
        s = S.LaunchSettings(family=world.mset.family, model=world.mset.model, context=context_for(world.machine, world.mset))
        menu = SimpleNamespace(columns=None, sel=None, current=None)
        menu.set_items = lambda items, sel: setattr(menu, "sel", sel)
        fake = SimpleNamespace(ctx=SimpleNamespace(settings=s, base_config=lambda: base), CHOICES=DraftingScreen.CHOICES,
                               query_one=lambda _cls: menu, paint=lambda: None, menu=lambda: menu)
        DraftingScreen.on_mount(fake)
        shown = DraftingScreen.CHOICES[menu.sel][0]
        menu.current = SimpleNamespace(key=pick or shown)
        DraftingScreen.commit(fake)
        return shown, s.drafting

    def test_the_step_shows_the_chosen_drafter_and_auto_replaces_it(self):
        m, s = get_machine("nv_resident"), get_set("ud_q4_k_xl")
        world = World(m, s).build()
        try:
            with world.patched():
                rec = drive_launch(world, ("--drafting", "mtp"), setup_runs=False)
                self.assertNotIn("failed", rec)
                self.assertEqual(self.step(world, None), ("mtp", "mtp"), "the step shows the MTP that holds; left as it is it stays")
                self.assertEqual(self.step(world, "auto"), ("mtp", "auto"), "Auto picked is written, so it replaces the MTP")
        finally:
            world.close()


if __name__ == "__main__":
    unittest.main()
