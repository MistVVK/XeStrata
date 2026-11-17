"""Auto on HIP gfx1201 (RX 9070 / R9700 class) starts the MTP until PE-RDNA4's fix for the open DFlash2 prompt stall lands (owner 11-06 ~10:30; drafting.AUTO_MTP, ONE entry).

Only Auto changes. DFlash2 chosen by name still starts DFlash2; gfx1151, other AMD cards, CUDA, Metal and SYCL are what they were; the confidence tables are untouched.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from strata_control import drafting

ROOT = Path(__file__).resolve().parents[2]
IQ3S = "qwen3.8-flash-next-iq3_s"
Q4 = "qwen3.8-flash-next-unsloth-ud-q4_k_xl"
V31 = "qwen3.8-flash-next-unsloth-ud-iq4_xs-v31"
GYRO = "qwen3.8-flash-next-gyro-gyro-s"
R9700 = ("hip", ["gfx1201"], 32768)


def flag(args, name):
    return args[args.index(name) + 1] if name in args and args.index(name) + 1 < len(args) else None


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.data = self.root / "data"
        for sub in ("dflash2", "mtp/rt", "packs/tag"):
            (self.data / sub).mkdir(parents=True)
        (self.data / "dflash2" / drafting.PACK_NAME).write_bytes(b"s" * (2 << 20))
        (self.data / "dflash2" / drafting.VOCAB_NAME).write_bytes(b"v" * 100)
        (self.data / "mtp" / "rt" / "experts.bin").write_bytes(b"e" * 100)
        (self.data / "mtp" / "rt" / drafting.VOCAB_NAME).write_bytes(b"v" * 100)
        self.exe = self.root / "strata"
        self.exe.write_bytes(b"--mtp-draft-vocab FILE (or --mtp DIR ... --draft-fallback ... selector codebooks: Q4_0 rows")
        self.args = ["--native", "m.gguf", "--pack", str(self.data / "packs" / "tag"), "--mtp", str(self.data / "mtp" / "rt"), "--spec", "4", "--max-context", "65536"]
        patcher = mock.patch.object(drafting, "ENGINE_SERVES_DFLASH2", True)
        patcher.start()
        self.addCleanup(patcher.stop)

    def apply(self, name, hardware, **extra):
        backend, archs, vram = hardware
        cfg = {"model_name": name, "args": list(self.args), **extra}
        decision = drafting.apply(cfg, data_dir=self.data, backend=backend, archs=archs, engine=self.exe, vram_mib=vram)
        return decision, cfg["args"]


class Gfx1201(Base):
    def test_auto_is_the_mtp_for_every_set(self):
        for name in (IQ3S, Q4, V31, GYRO):
            for extra in ({}, {"drafting": "auto"}):
                before = list(self.args)
                d, args = self.apply(name, R9700, **extra)
                self.assertEqual((d.drafter, d.note), ("mtp", ""), (name, extra))
                self.assertFalse([a for a in args if a.startswith("--draft")], (name, extra))
                self.assertEqual(args, before, "the MTP arguments exactly as saved")

    def test_dflash2_chosen_by_name_still_wins(self):
        d, args = self.apply(IQ3S, R9700, drafting="dflash2")
        self.assertEqual((d.drafter, d.note), ("dflash2", ""))
        self.assertEqual(flag(args, "--draft"), "dflash2")
        self.assertEqual(flag(args, "--draft-conf"), "0.35", "the R9700's own conf (12,570 slots)")
        self.assertNotIn("--mtp", args)
        mine = ["--draft-conf", "0.5"]
        d, args = self.apply(IQ3S, R9700, drafting="dflash2", user_args=mine)
        self.assertEqual(flag(args, "--draft-conf"), "0.5", "and the person's own conf over it")

    def test_the_other_choices_are_as_they_were(self):
        for mode, drafter in (("mtp", "mtp"), ("off", "off")):
            d, args = self.apply(IQ3S, R9700, drafting=mode)
            self.assertEqual(d.drafter, drafter, mode)

    def test_with_no_mtp_layer_on_disk_auto_is_not_dflash2_either(self):
        (self.data / "mtp" / "rt" / "experts.bin").unlink()
        d, args = self.apply(IQ3S, R9700)
        self.assertEqual((d.drafter, d.note), ("off", drafting.NONE_HERE))
        self.assertFalse([a for a in args if a.startswith("--draft")])
        d, args = self.apply(IQ3S, R9700, drafting="dflash2")
        self.assertEqual(d.drafter, "dflash2", "named: DFlash2 as before")

    def test_what_gets_fetched(self):
        kw = dict(model_name=IQ3S, backend="hip", archs=["gfx1201"], data_dir=self.root / "empty")
        self.assertFalse(drafting.needs_pack("auto", **kw), "no DFlash2 download for Auto")
        self.assertTrue(drafting.needs_pack("dflash2", **kw), "but one for DFlash2 named")
        self.assertFalse(drafting.needs_tuned("auto", **kw))
        self.assertTrue(drafting.needs_mtp("auto", mtp_here=False, **{k: v for k, v in kw.items() if k != "data_dir"}))
        self.assertFalse(drafting.wants_one_request("auto", model_name=IQ3S, backend="hip", archs=["gfx1201"]))
        self.assertTrue(drafting.wants_one_request("dflash2", model_name=IQ3S, backend="hip", archs=["gfx1201"]))

    def test_the_support_line_says_it_in_plain_words(self):
        support = drafting.hardware_support("hip", ["gfx1201"], IQ3S)
        self.assertEqual((support.can_run, support.level, support.line), (True, "tested", drafting.LINE_AUTO_MTP))
        self.assertEqual(drafting.LINE_AUTO_MTP, "DFlash2 is off by default on this graphics card for now; choose it in Drafting to use it.")
        for word in ("recommend", "best", "better", "should", "avoid"):
            self.assertNotIn(word, drafting.LINE_AUTO_MTP.lower())
        en = json.loads((ROOT.parent / "app" / "host" / "src" / "locales" / "df2-en.json").read_text(encoding="utf-8"))
        self.assertEqual(en["df22"], drafting.LINE_AUTO_MTP)

    def test_it_is_one_entry_and_removing_it_puts_auto_back(self):
        self.assertEqual(list(drafting.AUTO_MTP), [("hip", "gfx1201")])
        self.assertIn("stall", drafting.AUTO_MTP[("hip", "gfx1201")])
        with mock.patch.dict(drafting.AUTO_MTP, clear=True):
            d, args = self.apply(IQ3S, R9700)
            self.assertEqual((d.drafter, flag(args, "--draft")), ("dflash2", "dflash2"))
            self.assertEqual(drafting.hardware_support("hip", ["gfx1201"], IQ3S).line, drafting.LINE_TESTED)


class EverythingElseIsUnchanged(Base):
    def test_gfx1151_cuda_other_amd_metal_and_sycl_still_start_dflash2_under_auto(self):
        for hardware in (("hip", ["gfx1151"], None), ("hip", ["gfx1100"], 24576), ("hip", ["gfx1200"], 16384), ("cuda", ["sm_120"], 73728), ("cuda", ["sm_86"], 49152),      # (a 12 GB card has its own entry: test_drafting_streaming_class.py)
                         
                         ("cuda", ["sm_121"], None), ("sycl", ["arc_a"], None)):
            d, args = self.apply(IQ3S, hardware)
            self.assertEqual((d.drafter, flag(args, "--draft")), ("dflash2", "dflash2"), hardware)
        d, args = self.apply(Q4, ("metal", [], None))
        self.assertEqual((d.drafter, flag(args, "--draft")), ("dflash2", "dflash2"))

    def test_the_confidence_tables_and_the_other_entries_are_what_they_were(self):
        self.assertEqual(drafting.CONF_STEPS, ((12000, 0.35), (7500, 0.5), (3000, 0.7), (0, 0.8)))
        self.assertEqual(drafting.BACKEND_CONF, {"sycl": 0.6, "metal": 0.5})
        self.assertEqual((drafting.DRAFT_CONF, drafting.SPEC_WINDOW), (0.35, 6))
        self.assertEqual(drafting.conf_for("hip", drafting.slots_of(self.args, 32768)), 0.35)
        self.assertEqual(drafting.choice_of("auto", model_name=IQ3S, backend="hip", archs=["gfx1151"]), "dflash2")
        self.assertEqual(drafting.choice_of("auto", model_name=IQ3S, backend="hip", archs=["gfx1201"]), "mtp")
        self.assertEqual(drafting.choice_of("dflash2", model_name=IQ3S, backend="hip", archs=["gfx1201"]), "dflash2")
        self.assertEqual(drafting.choice_of("auto", family="qwen", backend="hip"), "dflash2", "no architecture known: as before")
        self.assertEqual(drafting.choice_of("auto", model_name=IQ3S, backend="cuda", archs=["gfx1201"]), "dflash2", "the family matters: only hip")
        self.assertEqual(drafting.hardware_support("hip", ["gfx1151"], IQ3S).line, drafting.LINE_TESTED)
        self.assertEqual(drafting.hardware_support("cuda", ["sm_120"], IQ3S).line, drafting.LINE_TESTED)


if __name__ == "__main__":
    unittest.main()
