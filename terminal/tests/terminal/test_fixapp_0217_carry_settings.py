"""#6244: a setup run again for an installed model (the desktop app does it when the context memory option changes) rewrote the run config and
turned the person's STRATA_RESIDENT_PIN=1 into "0" and dropped the DFlash2 drafting they had set. Setup's own environment entries are refreshed on
every run, but an entry whose value setup never writes is the person's: it is kept. The drafting mode and its arguments stay as they were.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import setup  # noqa: E402

DFLASH = ["--draft", "dflash2", "--draft-model", "D:/x/qwen3.8-flash-next-dflash2.q4_0-selq4.pack", "--draft-fallback", "none",
          "--mtp-draft-vocab", "D:/x/draft_vocab_latin.bin"]


def old_config():
    args = ["--max-context", "262144", *DFLASH]
    return {"model_name": "gyro-gyro-s", "backend": "hip", "drafting": "dflash2", "args": list(args), "args_written": ["--max-context", "262144", "--mtp", "D:/x/mtp/rt", "--spec-min-p", "0.5"],
            "env": {"STRATA_RESIDENT_PIN": "1", "STRATA_PARTIAL_PIN": "1", "STRATA_HIP_WMMA": "1"}}


def new_config():
    return {"model_name": "gyro-gyro-s", "backend": "hip", "args": ["--max-context", "131072", "--mtp", "D:/x/mtp/rt", "--spec-min-p", "0.5"],
            "env": {"STRATA_RESIDENT_PIN": "0"}}


class SetupRunAgain(unittest.TestCase):
    def test_the_persons_resident_pin_survives_a_setup_run(self):
        cfg = new_config()
        kept = setup.carry_over(old_config(), cfg)
        self.assertEqual(cfg["env"]["STRATA_RESIDENT_PIN"], "1")
        self.assertIn("env STRATA_RESIDENT_PIN", kept)
        self.assertEqual(cfg["env"]["STRATA_PARTIAL_PIN"], "1")

    def test_the_value_setup_writes_itself_is_still_refreshed(self):
        old = old_config()
        old["env"]["STRATA_RESIDENT_PIN"] = "0"                # what an earlier setup run wrote
        cfg = new_config()
        cfg["env"] = {}                                         # and this run no longer writes it
        setup.carry_over(old, cfg)
        self.assertNotIn("STRATA_RESIDENT_PIN", cfg["env"])

    def test_the_drafting_mode_survives_a_setup_run(self):
        cfg = new_config()
        setup.carry_over(old_config(), cfg)
        self.assertEqual(cfg.get("drafting"), "dflash2")

    def test_the_dflash2_arguments_the_person_applied_survive(self):
        cfg = new_config()
        setup.carry_over(old_config(), cfg)
        self.assertEqual(cfg["args"][cfg["args"].index("--draft"):cfg["args"].index("--draft") + len(DFLASH)], DFLASH)


if __name__ == "__main__":
    unittest.main()
