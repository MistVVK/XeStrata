"""The per-card DFlash2 window and confidence of 0.2.13 (owner 11-08, RESULTS-CONF.md): Strix Halo window 8 (conf 0.35), Blackwell cc 12.0 with >= 48 GB window 8 and conf 0.55.

Every other card keeps conf and --spec 6, so its EFFECTIVE window (--spec lowered to the engine's cap, include/strata/core/draft_cap.hpp) is what it was.
"""

from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from strata_control import drafting

IQ3S = "qwen3.8-flash-next-iq3_s"
Q4 = "qwen3.8-flash-next-unsloth-ud-q4_k_xl"
V31 = "qwen3.8-flash-next-unsloth-ud-iq4_xs-v31"

GIB = 1 << 30
TUNED = b"tuned-pack-" * 200_000


def old_cap(uma, major):       # include/strata/core/draft_cap.hpp before 11-08
    return 5 if uma else 6 if major == 12 else 0


def new_cap(uma, major, minor, vram_bytes):      # the same file after (mirrored here; src/core/draft_cap_test.cpp tests the C++)
    if uma:
        return 8
    if major == 12:
        return 8 if (minor == 0 and vram_bytes >= 48_000_000_000) else 6
    return 0


def effective(spec, cap):
    return min(spec, cap) if cap > 0 else spec


class CardWindow(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.data = Path(self.tmp.name)
        (self.data / "dflash2").mkdir()
        (self.data / "dflash2" / drafting.PACK_NAME).write_bytes(b"s" * (2 << 20))
        (self.data / "dflash2" / drafting.VOCAB_NAME).write_bytes(b"v" * 100)
        self.exe = self.data / "strata"
        self.exe.write_bytes(b"--mtp-draft-vocab FILE (or --mtp DIR ... --draft-fallback ... selector codebooks: Q4_0 rows")
        for patcher in (mock.patch.object(drafting, "ENGINE_SERVES_DFLASH2", True),
                        mock.patch.multiple(drafting, TUNED_BYTES=len(TUNED), TUNED_SHA256=hashlib.sha256(TUNED).hexdigest())):
            patcher.start()
            self.addCleanup(patcher.stop)

    def flags(self, name, backend, archs, vram, extra=(), user_args=None, tuned=False):
        tuned_file = self.data / "dflash2" / drafting.TUNED_NAME
        if tuned:
            tuned_file.write_bytes(TUNED)
        else:
            tuned_file.unlink(missing_ok=True)
        cfg = {"model_name": name, "args": ["--native", "m.gguf", *extra], "drafting": "dflash2"}
        if user_args:
            cfg["user_args"] = list(user_args)
        with mock.patch.object(drafting, "detect_hardware", return_value=(backend, list(archs))):
            d = drafting.apply(cfg, data_dir=self.data, backend=backend, archs=list(archs), engine=self.exe, vram_mib=vram)
        self.assertEqual(d.drafter, "dflash2")
        a = cfg["args"]
        return a[a.index("--spec") + 1], a[a.index("--draft-conf") + 1]

    def test_the_constants(self):
        self.assertEqual((drafting.SPEC_WINDOW, drafting.WIDE_WINDOW, drafting.DRAFT_CONF), (6, 8, 0.35))
        self.assertEqual(drafting.BIG_CARD_BYTES, 48_000_000_000)

    def test_strix_halo_window_8_conf_unchanged(self):
        for name in (IQ3S, Q4, V31):
            self.assertEqual(self.flags(name, "hip", ["gfx1151"], None), ("8", "0.35"), name)

    def test_blackwell_cc_12_0_with_48_gb_or_more(self):
        """Window 8 for every set; conf 0.55 for IQ3_S only (the set measured there) and only with the tuned pack."""
        for mib in (73728, 73415, 49152, 49140, 98304, 45777):         # 72 GB, the l620 reading, 48 GiB, an RTX A6000-class 48 GB, 96 GB, the 48e9 byte line
            self.assertEqual(self.flags(IQ3S, "cuda", ["sm_120"], mib, tuned=True), ("8", "0.55"), mib)
            self.assertEqual(self.flags(IQ3S, "cuda", ["sm_120"], mib), ("8", "0.35"), "the stock pack was not measured at 0.55")
            for name in (V31, Q4):
                self.assertEqual(self.flags(name, "cuda", ["sm_120"], mib, tuned=True), ("8", "0.35"), (name, mib))
                self.assertEqual(self.flags(name, "cuda", ["sm_120"], mib), ("8", "0.35"), (name, mib))

    def test_blackwell_cc_12_0_under_48_gb_keeps_today(self):
        for mib in (32768, 24576, 16384, 12288, 45776, None):           # RTX 5090 32 GB .. 5070 12 GB, just under the line, memory unknown (several cards)
            self.assertEqual(self.flags(IQ3S, "cuda", ["sm_120"], mib, tuned=True), ("6", f"{drafting.conf_for('cuda', drafting.slots_of([], mib)):g}"), mib)     # the slot rule, as before
        self.assertEqual(self.flags(IQ3S, "cuda", ["sm_120"], 32768)[1], "0.35")

    def test_the_spark_is_unchanged(self):
        for tuned in (False, True):
            self.assertEqual(self.flags(IQ3S, "cuda", ["sm_121"], None, tuned=tuned), ("6", "0.35"))
            self.assertEqual(self.flags(V31, "cuda", ["sm_121"], None, tuned=tuned), ("6", "0.35"))
            self.assertEqual(self.flags(IQ3S, "cuda", ["sm_121"], 131072, tuned=tuned), ("6", "0.35"), "even when a reading of its pool is passed")

    def test_the_big_card_with_a_small_expert_cache_keeps_the_slot_rule(self):
        self.assertEqual(self.flags(IQ3S, "cuda", ["sm_120"], 73728, extra=["--expert-cache", "1000"], tuned=True), ("6", "0.8"))

    def test_other_cards_are_unchanged(self):
        for backend, archs, mib in (("cuda", ["sm_86"], 12288), ("cuda", ["sm_89"], 24576), ("cuda", ["sm_90"], 81920), ("cuda", ["sm_86"], 49152), ("hip", ["gfx1201"], 32768),
                                    ("hip", ["gfx1100"], 24576), ("hip", ["gfx1103"], None), ("sycl", ["arc_a"], None), ("metal", [], None)):
            spec, conf = self.flags(Q4 if backend == "hip" else V31, backend, archs, mib)
            self.assertEqual(spec, "6", (backend, archs))
            self.assertEqual(conf, f"{drafting.conf_for(backend, drafting.slots_of([], mib)):g}", (backend, archs))      # the stock pack here: the backend's own rule

    def test_the_persons_own_flags_win(self):
        self.assertEqual(self.flags(IQ3S, "cuda", ["sm_120"], 73728, user_args=["--draft-conf", "0.4", "--spec", "5"], extra=["--draft-conf", "0.4", "--spec", "5"], tuned=True)[1], "0.4")
        cfg = {"model_name": IQ3S, "args": ["--native", "m.gguf", "--spec", "5"], "user_args": ["--spec", "5"], "drafting": "dflash2"}
        with mock.patch.object(drafting, "detect_hardware", return_value=("hip", ["gfx1151"])):
            drafting.apply(cfg, data_dir=self.data, backend="hip", archs=["gfx1151"], engine=self.exe)
        self.assertEqual(cfg["args"].count("--spec"), 1)
        self.assertEqual(cfg["args"][cfg["args"].index("--spec") + 1], "5")

    def test_no_card_but_the_two_classes_changes_its_effective_window(self):
        """--spec as written, lowered to the engine's cap: the same number as before for every card but Strix Halo (5 -> 8) and the large Blackwell class (6 -> 8)."""
        cards = [("hip", ["gfx1151"], None, True, 0, 0, 0)]
        for mib, major, minor in ((73728, 12, 0), (98304, 12, 0), (49152, 12, 0), (32768, 12, 0), (16384, 12, 0), (12288, 12, 0), (None, 12, 0), (131072, 12, 1), (None, 12, 1),
                                  (12288, 8, 6), (24576, 8, 9), (81920, 9, 0), (49152, 8, 6), (73728, 10, 0)):
            cards.append(("cuda", [f"sm_{major}{minor}"], mib, False, major, minor, (mib or 0) * (1 << 20)))
        for backend, archs, mib, uma, major, minor, vram in (*cards, ("hip", ["gfx1201"], 32768, False, 0, 0, 0), ("hip", ["gfx1100"], 24576, False, 0, 0, 0), ("sycl", ["arc_a"], None, False, 0, 0, 0), ("metal", [], None, False, 0, 0, 0)):
            spec = int(self.flags(IQ3S if backend != "hip" else Q4, backend, archs, mib)[0])
            old = effective(6, old_cap(uma, major))
            new = effective(spec, new_cap(uma, major, minor, vram))
            big = backend == "cuda" and major == 12 and minor == 0 and vram >= 48_000_000_000
            want = 8 if (uma or big) else old
            self.assertEqual(new, want, (backend, archs, mib, "old", old))


if __name__ == "__main__":
    unittest.main()
