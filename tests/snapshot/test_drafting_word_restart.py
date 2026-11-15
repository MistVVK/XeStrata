"""A restart with no Drafting word keeps the saved word (boxia d3d and l620 C2, 0.2.14.1: after `--drafting auto` and a restart with no word the launch's
records said null where they said "auto"; the start still ran DFlash2, because null is Auto).  Nothing real is used (tests/snapshot/README.md).

Proves on every machine and model set of the matrix: "auto" stays "auto" over two restarts with no word and the engine line does not change; a saved Off
still holds (keep_saved_off); a choice the model's installed config holds (the app's Models screen) still decides a start with no word, so no "auto" is
written over it; a first launch with no word still writes nothing (null = Auto, as before).

  python -m pytest tests/snapshot/test_drafting_word_restart.py -q -p no:cacheprovider
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tests.snapshot.matrix import MACHINES, MODEL_SETS, machine as get_machine, model_set as get_set  # noqa: E402
from tests.snapshot.test_drafting_round_trip import argv, drafter, launches  # noqa: E402


class AutoStaysAuto(unittest.TestCase):
    maxDiff = None

    def test_a_restart_without_a_word_keeps_auto(self):
        for m in MACHINES:
            for s in MODEL_SETS:
                with self.subTest(machine=m.key, set=s.key):
                    recs = launches(m, s, [("auto", False), (None, False), (None, False)])
                    for later in recs[1:]:
                        self.assertEqual(later["config_file"].get("drafting"), "auto", "the saved word stays as it was")
                        self.assertEqual(drafter(later), drafter(recs[0]), "the same drafter as with the word")
                    # (the first Auto start can fetch the drafter's pack; the restarts after it start alike)
                    self.assertEqual(argv(recs[2]), argv(recs[1]), "the restarts start alike")
                    same = launches(m, s, [("auto", False), ("auto", False)])
                    self.assertEqual(argv(recs[1]), argv(same[1]), "a restart with no word starts as one with --drafting auto")

    def test_a_saved_off_still_holds(self):
        for mkey, skey in (("nv_12gb_streaming", "iq3_s"), ("nv_resident", "iq3_s"), ("strix_halo_win", "ud_q4_k_xl")):
            with self.subTest(machine=mkey, set=skey):
                recs = launches(get_machine(mkey), get_set(skey), [("auto", False), ("off", False), (None, False)])
                self.assertEqual(recs[2]["config_file"].get("drafting"), "off")
                self.assertEqual(argv(recs[2]), argv(recs[1]))

    def test_null_is_auto_where_the_word_is_read(self):
        from strata_control import drafting
        self.assertEqual(drafting.normalize(None), "auto")
        self.assertEqual(drafting.DEFAULT_MODE, "auto")
        for backend, archs in (("cuda", ["sm_120"]), ("hip", ["gfx1201"]), ("metal", []), ("sycl", ["arc_a"])):
            self.assertEqual(drafting.choice_of(None, family="qwen", backend=backend, archs=archs),
                             drafting.choice_of("auto", family="qwen", backend=backend, archs=archs), backend)
        host = (ROOT / "apps" / "host" / "server" / "native-control.mjs").read_text(encoding="utf-8")
        self.assertIn("drafting:cfg.drafting??'auto'", host, "the app reads a missing word as Auto")

    def test_a_first_launch_without_a_word_writes_nothing(self):
        recs = launches(get_machine("nv_resident"), get_set("iq3_s"), [(None, False), (None, False)])
        for rec in recs:
            self.assertIsNone(rec["config_file"].get("drafting"), "no word, nothing chosen: null is Auto")
        self.assertEqual(argv(recs[1]), argv(recs[0]))

    def test_the_models_screen_choice_still_decides_a_start_without_a_word(self):
        # the app saves its Drafting choice in the model's installed config; a later `strata launch` with no word runs that choice, as before
        def models_screen_says_mtp(world):
            cfg = json.loads(world.config_path.read_text(encoding="utf-8"))
            cfg["drafting"] = "mtp"
            world.config_path.write_text(json.dumps(cfg, indent=1), encoding="utf-8")
        recs = launches(get_machine("nv_resident"), get_set("iq3_s"), [("auto", False), (None, False)], prepare=models_screen_says_mtp)
        self.assertEqual(recs[0]["config_file"].get("drafting"), "auto", "the word given wins over the installed choice")
        self.assertEqual(recs[1]["config_file"].get("drafting"), "mtp", "no word: the installed config's own choice, not an auto written over it")


if __name__ == "__main__":
    unittest.main()
