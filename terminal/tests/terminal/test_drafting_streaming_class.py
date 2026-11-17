"""Auto defaults for the 12 GB / streaming CUDA class (owner 11-06 ~15:30 "1 yes"; drafting.CLASS_DEFAULTS, one removable entry per set).

The class is the one the confidence already reads: a CUDA card whose slots_of() is under the last-but-one CONF_STEPS step, so conf_for() gives 0.8 (P2003, RTX 3060 12 GB).
IQ3_S -> Auto is the MTP. UD-IQ4_XS-v31 -> Auto is DFlash2 with the tuned selq4 pack at the class's current 0.8 (it depends on RTX2's file-tier fix in the same release).
Anything chosen by name wins; the resident-CUDA, Strix, gfx1201, Arc and Metal entries and the conf tables are unchanged.
"""

from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from strata_control import drafting, drafting_fetch

IQ3S = "qwen3.8-flash-next-iq3_s"
Q4 = "qwen3.8-flash-next-unsloth-ud-q4_k_xl"
V31 = "qwen3.8-flash-next-unsloth-ud-iq4_xs-v31"
OLD_XS = "qwen3.8-flash-next-unsloth-ud-iq4_xs"
GYRO = "qwen3.8-flash-next-gyro-gyro-s"
IQ3XXS = "qwen3.8-flash-next-iq3_xxs"
Q2 = "qwen3.8-flash-next-q2_0"
TUNED = b"tuned-pack-" * 200_000
P2003 = ("cuda", ["sm_86"], 12288)             # RTX 3060 12 GB
CARD16 = ("cuda", ["sm_89"], 16384)            # conf 0.7 step: not the class
CARD24 = ("cuda", ["sm_89"], 24576)
L620 = ("cuda", ["sm_120"], 73728)
SPARK = ("cuda", ["sm_121"], None)
STRIX = ("hip", ["gfx1151"], None)
R9700 = ("hip", ["gfx1201"], 32768)
ARC = ("sycl", ["arc_a"], None)


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
        self.args = ["--native", "m.gguf", "--pack", str(self.data / "packs" / "tag"), "--mtp", str(self.data / "mtp" / "rt"), "--spec", "4", "--max-context", "8192"]
        for patcher in (mock.patch.object(drafting, "ENGINE_SERVES_DFLASH2", True),
                        mock.patch.multiple(drafting, TUNED_BYTES=len(TUNED), TUNED_SHA256=hashlib.sha256(TUNED).hexdigest())):
            patcher.start()
            self.addCleanup(patcher.stop)

    def put_tuned(self):
        (self.data / "dflash2" / drafting.TUNED_NAME).write_bytes(TUNED)

    def apply(self, name, hardware, **extra):
        backend, archs, vram = hardware
        cfg = {"model_name": name, "args": list(self.args), **extra}
        decision = drafting.apply(cfg, data_dir=self.data, backend=backend, archs=archs, engine=self.exe, vram_mib=vram)
        return decision, cfg["args"]


class TheClass(Base):
    def test_it_is_the_card_class_the_confidence_calls_0_8(self):
        for hardware in (P2003, ("cuda", ["sm_86"], 8192), ("cuda", ["sm_75"], 6144)):
            slots = drafting.slots_of(self.args, hardware[2])
            self.assertEqual(drafting.conf_for("cuda", slots), 0.8, hardware)
            self.assertTrue(drafting.streaming_class("cuda", hardware[1], args=self.args, vram_mib=hardware[2]), hardware)
        for hardware in (CARD16, CARD24, L620, SPARK, ("cuda", ["sm_120"], None), ("cuda", ["sm_86"], None)):
            self.assertFalse(drafting.streaming_class(hardware[0], hardware[1], args=self.args, vram_mib=hardware[2]), hardware)
        for backend, archs in (("hip", ["gfx1100"]), ("hip", ["gfx1201"]), ("sycl", ["arc_a"]), ("metal", [])):
            self.assertFalse(drafting.streaming_class(backend, archs, args=self.args, vram_mib=12288), backend)
        # the boundary is the 0.8 step itself, and a person's own --expert-cache counts as the confidence counts it
        for given, inside in (("2300", True), ("2400", False), ("64", True)):
            args = self.args + ["--expert-cache", given]
            self.assertEqual(drafting.streaming_class("cuda", ["sm_120"], args=args, vram_mib=73728), inside, given)
            self.assertEqual(drafting.conf_for("cuda", drafting.slots_of(args, 73728)) == 0.8, inside, given)


class IQ3SOnTheClass(Base):
    def test_auto_is_the_mtp(self):
        self.put_tuned()
        for extra in ({}, {"drafting": "auto"}):
            before = list(self.args)
            d, args = self.apply(IQ3S, P2003, **extra)
            self.assertEqual((d.drafter, d.note), ("mtp", ""), extra)
            self.assertFalse([a for a in args if a.startswith("--draft")], extra)
            self.assertEqual(args, before, "the MTP arguments exactly as saved")

    def test_dflash2_by_name_wins_with_the_class_conf(self):
        d, args = self.apply(IQ3S, P2003, drafting="dflash2")
        self.assertEqual((d.drafter, flag(args, "--draft")), ("dflash2", "dflash2"))
        self.assertEqual(flag(args, "--draft-conf"), "0.8")
        self.assertEqual(flag(args, "--draft-model"), str(self.data / "dflash2" / drafting.PACK_NAME), "the stock pack: IQ3_S has no tuned entry on this class")
        d, args = self.apply(IQ3S, P2003, drafting="dflash2", user_args=["--draft-conf", "0.5"])
        self.assertEqual(flag(args, "--draft-conf"), "0.5")
        for mode, drafter in (("mtp", "mtp"), ("off", "off")):
            self.assertEqual(self.apply(IQ3S, P2003, drafting=mode)[0].drafter, drafter)

    def test_without_the_mtp_layer_auto_starts_the_dflash2_drafter_on_disk(self):
        # one drafter is picked; when its file is missing the other one stands in where it is on disk (never no drafter unless Off was chosen; 0.2.14)
        (self.data / "mtp" / "rt" / "experts.bin").unlink()
        d, args = self.apply(IQ3S, P2003)
        self.assertEqual((d.drafter, d.note), ("dflash2", drafting.NO_MTP))
        self.assertEqual(self.apply(IQ3S, P2003, drafting="dflash2")[0].drafter, "dflash2")

    def test_the_fetch(self):
        kw = dict(model_name=IQ3S, backend="cuda", archs=["sm_86"], data_dir=self.root / "empty", args=self.args, vram_mib=12288)
        self.assertFalse(drafting.needs_pack("auto", **kw))
        self.assertFalse(drafting.needs_tuned("auto", **kw))
        self.assertTrue(drafting.needs_pack("dflash2", **kw))
        self.assertTrue(drafting.needs_mtp("auto", mtp_here=False, **{k: v for k, v in kw.items() if k != "data_dir"}))
        self.assertFalse(drafting.wants_one_request("auto", model_name=IQ3S, backend="cuda", archs=["sm_86"], args=self.args, vram_mib=12288))
        self.assertEqual(drafting.choice_of("auto", model_name=IQ3S, backend="cuda", archs=["sm_86"], args=self.args, vram_mib=12288), "mtp")

    def test_every_other_card_keeps_dflash2_for_iq3_s(self):
        for hardware in (CARD16, CARD24, L620, SPARK, STRIX, ARC):
            d, args = self.apply(IQ3S, hardware)
            self.assertEqual((d.drafter, flag(args, "--draft")), ("dflash2", "dflash2"), hardware)


class V31OnTheClass(Base):
    def test_auto_is_dflash2_with_the_tuned_pack_at_the_class_conf(self):
        self.put_tuned()
        for extra in ({}, {"drafting": "auto"}):
            d, args = self.apply(V31, P2003, **extra)
            self.assertEqual((d.drafter, d.note), ("dflash2", ""), extra)
            self.assertEqual(flag(args, "--draft-model"), str(drafting.tuned_path(self.data)))
            self.assertEqual(flag(args, "--draft-conf"), "0.8", "the class's current conf, unchanged")
            self.assertEqual(flag(args, "--spec"), "6")
            self.assertNotIn("--mtp", args)
        # only the pack changes against the same start without the tuned file
        (self.data / "dflash2" / drafting.TUNED_NAME).unlink()
        (self.data / "dflash2" / (drafting.TUNED_NAME + ".ok")).unlink()
        d2, args2 = self.apply(V31, P2003)
        self.assertEqual(flag(args2, "--draft-conf"), "0.8")
        self.assertEqual(flag(args2, "--draft-model"), str(drafting.pack_path(self.data, self.exe)))

    def test_by_name_and_the_users_own_conf_win(self):
        self.put_tuned()
        d, args = self.apply(V31, P2003, drafting="dflash2", user_args=["--draft-conf", "0.6"])
        self.assertEqual((d.drafter, flag(args, "--draft-conf")), ("dflash2", "0.6"))
        self.assertEqual(self.apply(V31, P2003, drafting="mtp")[0].drafter, "mtp")
        self.assertEqual(self.apply(V31, P2003, drafting="off")[0].drafter, "off")

    def test_nothing_fetched_is_the_mtp_as_before(self):
        d, args = self.apply(V31, P2003)
        self.assertEqual(d.drafter, "dflash2", "the stock pack is on this disk")
        (self.data / "dflash2" / drafting.PACK_NAME).unlink()
        before = list(self.args)
        d, args = self.apply(V31, P2003)
        self.assertEqual((d.drafter, d.note, args), ("mtp", drafting.NOT_DOWNLOADED, before))

    def test_it_is_fetched_for_this_class_only(self):
        kw = dict(model_name=V31, backend="cuda", data_dir=self.data, args=self.args)
        self.assertTrue(drafting.needs_tuned("auto", archs=["sm_86"], vram_mib=12288, **kw))
        self.assertTrue(drafting.tuned_applies("cuda", V31, ["sm_86"], args=self.args, vram_mib=12288))
        self.assertTrue(drafting_fetch.plan(self.data, model_name=V31, backend="cuda", archs=["sm_86"], args=self.args, vram_mib=12288)["gb"] == drafting.TUNED_GB)
        # the resident rule of the same set is untouched
        self.assertTrue(drafting.tuned_applies("cuda", V31, ["sm_120"], args=self.args, vram_mib=73728))
        self.assertTrue(drafting.tuned_applies("cuda", V31, ["sm_121"], args=[], vram_mib=None))
        self.assertFalse(drafting.tuned_applies("cuda", V31, ["sm_89"], args=self.args, vram_mib=16384), "the 0.7 step is neither class")
        self.assertFalse(drafting.tuned_applies("cuda", V31, ["sm_120", "sm_120"], args=self.args, vram_mib=None), "several cards")

    def test_the_support_line_for_sm_86_reads_the_set(self):
        h = drafting.hardware_support
        self.assertEqual(h("cuda", ["sm_86"], IQ3S).line, drafting.LINE_SLOWER, "IQ3_S on sm_86 keeps the line it had")
        self.assertEqual(h("cuda", ["sm_86"]).line, drafting.LINE_SLOWER)
        self.assertEqual(h("cuda", ["sm_86"], V31).line, drafting.LINE_TESTED, "v31 is DFlash2 by default there: faster on some text, slower on other")
        self.assertEqual(h("cuda", ["sm_120"], V31).line, drafting.LINE_TESTED)


class OtherSetsAndEntriesAreUnchanged(Base):
    def test_other_sets_on_the_class_follow_the_general_rule(self):
        self.put_tuned()
        for name in (Q4, OLD_XS, GYRO, IQ3XXS, Q2):
            d, args = self.apply(name, P2003)
            self.assertEqual((d.drafter, flag(args, "--draft")), ("dflash2", "dflash2"), name)
            self.assertNotEqual(flag(args, "--draft-model"), str(drafting.tuned_path(self.data)), name)
            self.assertEqual(flag(args, "--draft-conf"), "0.8", name)

    def test_the_other_entries_and_the_conf_tables(self):
        self.assertEqual(drafting.CONF_STEPS, ((12000, 0.35), (7500, 0.5), (3000, 0.7), (0, 0.8)))
        self.assertEqual(drafting.BACKEND_CONF, {"sycl": 0.6, "metal": 0.5})
        self.assertEqual((drafting.DRAFT_CONF, drafting.SPEC_WINDOW), (0.35, 6))
        self.assertEqual(set(drafting.CLASS_DEFAULTS), {("cuda", "iq3_s", "streaming"), ("cuda", "unsloth-ud-iq4_xs-v31", "streaming")})
        self.assertEqual(set(drafting.SET_DEFAULTS), {
            ("metal", "unsloth-ud-q4_k_xl"), ("metal", "unsloth-ud-iq4_xs-v31"), ("metal", "iq3_s"),
            ("cuda", "iq3_s"), ("cuda", "unsloth-ud-iq4_xs-v31"),
            ("hip", "iq3_s"), ("hip", "unsloth-ud-q4_k_xl"), ("hip", "unsloth-ud-iq4_xs"), ("hip", "unsloth-ud-iq4_xs-v31"),
            ("sycl", "iq3_xxs")})
        self.assertEqual(list(drafting.AUTO_MTP), [("hip", "gfx1201")])
        self.assertEqual(drafting.choice_of("auto", model_name=IQ3S, backend="metal"), "mtp")
        self.assertEqual(drafting.choice_of("auto", model_name=IQ3S, backend="hip", archs=["gfx1201"]), "mtp")
        self.assertEqual(drafting.choice_of("auto", model_name=IQ3S, backend="hip", archs=["gfx1151"]), "dflash2")
        self.assertEqual(drafting.choice_of("auto", model_name=IQ3S, backend="sycl", archs=["arc_a"]), "dflash2")
        self.assertEqual(drafting.choice_of("auto", model_name=IQ3S, backend="cuda", archs=["sm_86"]), "dflash2", "no card signal given: the general rule, as before")

    def test_the_resident_strix_arc_and_metal_starts_are_what_they_were(self):
        self.put_tuned()
        for hardware, name, conf in ((L620, IQ3S, "0.55"), (SPARK, V31, "0.35"), (STRIX, IQ3S, "0.35")):      # l620: 0.55 since 11-08
            d, args = self.apply(name, hardware)
            self.assertEqual((d.drafter, flag(args, "--draft-model"), flag(args, "--draft-conf")), ("dflash2", str(drafting.tuned_path(self.data)), conf), (hardware, name))
        d, args = self.apply(Q4, ("metal", [], None))
        self.assertEqual((flag(args, "--draft-model"), flag(args, "--draft-conf")), (str(drafting.tuned_path(self.data)), "0.7"))
        self.assertEqual(self.apply(IQ3S, R9700)[0].drafter, "mtp")
        self.assertEqual(self.apply(IQ3S, R9700, drafting="dflash2")[0].drafter, "dflash2")

    def test_one_removable_entry_per_set(self):
        self.put_tuned()
        with mock.patch.dict(drafting.CLASS_DEFAULTS, {k: v for k, v in drafting.CLASS_DEFAULTS.items() if k[1] != "iq3_s"}, clear=True):
            self.assertEqual(self.apply(IQ3S, P2003)[0].drafter, "dflash2")
            self.assertEqual(flag(self.apply(V31, P2003)[1], "--draft-model"), str(drafting.tuned_path(self.data)))
        with mock.patch.dict(drafting.CLASS_DEFAULTS, {k: v for k, v in drafting.CLASS_DEFAULTS.items() if k[1] == "iq3_s"}, clear=True):
            self.assertEqual(self.apply(IQ3S, P2003)[0].drafter, "mtp")
            d, args = self.apply(V31, P2003)
            self.assertEqual(flag(args, "--draft-conf"), "0.8")
            self.assertNotEqual(flag(args, "--draft-model"), str(drafting.tuned_path(self.data)), "without its entry the 12 GB card gets the stock pack")


if __name__ == "__main__":
    unittest.main()
