"""Setup's recommendations from the community bench data (0.1.40): text only, never a changed default.

    python -m unittest tools.test_setup_bench_tips
"""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
import setup  # noqa: E402
from test_setup_hybrid import fake_sys  # noqa: E402

BASE = ["--expert-cache", "auto", "--prefill", "auto", "--max-context", "65536"]


def tips(args=BASE, env=None, ram=32.0, model=20.0, vram=12.0, vision="none", win=False, gpu=None):
    return setup.bench_tips(list(args), env, ram, model, vram, vision, win, gpu)


class BenchTips(unittest.TestCase):
    def test_nothing_to_say_on_a_plain_pc(self):
        self.assertEqual(tips(), [])

    def test_explicit_cache_warns_on_windows_only(self):
        a = ["--expert-cache", "2400", "--prefill", "auto"]
        self.assertTrue(any("--expert-cache 2400" in t and "7x" in t for t in tips(a, win=True)))
        self.assertEqual(tips(a, win=False), [])
        self.assertEqual(tips(BASE, win=True), [])                      # auto: nothing to warn about

    def test_the_chunk_tip_is_not_said_on_an_apu(self):
        tip = next(t for t in tips(ram=128) if "auto:32768" in t)
        self.assertTrue(setup.apu_hides_chunk_tip({"arch": "gfx1151", "uma": True}, tip))     # Strix Halo
        self.assertFalse(setup.apu_hides_chunk_tip({"arch": "gfx1201"}, tip))                 # a discrete card keeps it
        self.assertFalse(setup.apu_hides_chunk_tip({"arch": "gfx1151", "uma": True}, "warning: --prefill auto:32768 on 32 GB of RAM"))

    def test_prefill_auto_32768_follows_the_ram(self):
        self.assertTrue(any("auto:32768" in t and t.startswith("tip") for t in tips(ram=128)))
        self.assertEqual([t for t in tips(ram=64) if "auto:32768" in t], [])
        big = ["--prefill", "auto:32768"]
        self.assertTrue(any(t.startswith("warning") and "auto:32768" in t for t in tips(big, ram=32)))
        self.assertEqual([t for t in tips(big, ram=96) if "auto:32768" in t], [])

    def test_the_32768_tip_is_left_out_where_a_bare_auto_already_takes_it(self):
        # round 0212: one NVIDIA card of compute capability 8.0+ (arch "86", "120", "121") gets a 32768 ceiling from a bare auto; so does 7.0 (Volta, "70"),
        # the engine's own rule (nv_chunk_card_ok in include/strata/core/nv_chunk.hpp)
        tip = lambda **k: [t for t in tips(ram=128, **k) if "auto:32768" in t]
        for arch in ("70", "80", "86", "89", "90", "120", "121"):
            self.assertEqual(tip(gpu={"arch": arch}), [], arch)
        for arch in ("75", "60", "61", "62", "52", "gfx1201", ""):      # Turing, Pascal, older cards, AMD, unknown: the tip stays
            self.assertTrue(tip(gpu={"arch": arch}), arch)
        self.assertTrue(tip())                                           # no GPU information: as before
        self.assertTrue(tip(gpu={"arch": "86"}, env={"STRATA_NV_CHUNK": "0"}))
        self.assertTrue(tip(gpu={"arch": "86"}, env={"STRATA_DEFAULTS": "026"}))
        self.assertTrue(tip(args=[*BASE, "--gpus", "0,1"], gpu={"arch": "86"}))      # a layer split reads 8192-token chunks
        self.assertTrue(tip(args=[*BASE, "--layer-split", "auto"], gpu={"arch": "86"}))
        self.assertTrue(tip(gpu={"arch": "70"}, env={"STRATA_NV_CHUNK": "0"}))      # Volta follows the same switches and the same layer-split rule
        self.assertTrue(tip(args=[*BASE, "--gpus", "0,1"], gpu={"arch": "70"}))

    def test_resident_headroom_on_small_ram(self):
        res = [*BASE, "--resident-experts"]
        self.assertTrue(any("STRATA_RESIDENT_HEADROOM_GIB" in t for t in tips(res, ram=32)))
        self.assertTrue(any("STRATA_RESIDENT_HEADROOM_GIB" in t for t in tips([*BASE, "--resident-budget-gib", "40"], ram=48)))
        self.assertEqual([t for t in tips(res, ram=64) if "HEADROOM" in t], [])
        self.assertEqual([t for t in tips(res, {"STRATA_RESIDENT_HEADROOM_GIB": "8"}, ram=32) if "HEADROOM" in t], [])
        self.assertEqual([t for t in tips(BASE, ram=32) if "HEADROOM" in t], [])   # not a resident config

    def test_no_conversation_cache_tip_the_engine_sizes_it_itself(self):
        # 0.2.3 (#3493): the engine sizes the conversation cache from the free RAM when no value is given, so setup no longer tips about it
        for ram, model in ((64, 40), (64, 41), (128, 40)):
            self.assertEqual([t for t in tips(ram=ram, model=model) if "conversation" in t], [])
        self.assertEqual([t for t in tips([*BASE, "--conversation-cache-mib", "4096"], ram=128) if "conversation" in t], [])

    def test_vision_cpu_on_small_cards(self):
        self.assertTrue(any("--vision cpu" in t for t in tips(vision="gpu", vram=12)))
        self.assertEqual([t for t in tips(vision="gpu", vram=24) if "vision" in t], [])
        self.assertEqual([t for t in tips(vision="cpu", vram=12) if "vision" in t], [])

    def test_tips_never_touch_the_arguments(self):
        a = ["--expert-cache", "2400", "--prefill", "auto", "--resident-experts"]
        before = list(a)
        env = {}
        tips(a, env, ram=128, vision="gpu", win=True)
        self.assertEqual(a, before)
        self.assertEqual(env, {})


class Sockets(unittest.TestCase):
    def fake(self, d, sockets, cores):
        n = 0
        for s in range(sockets):
            for c in range(cores):
                t = Path(d) / "devices" / "system" / "cpu" / f"cpu{n}" / "topology"
                t.mkdir(parents=True)
                (t / "physical_package_id").write_text(str(s))
                (t / "core_id").write_text(str(c))
                n += 1

    def test_linux_sockets(self):
        with tempfile.TemporaryDirectory() as d:
            self.fake(d, 2, 18)
            self.assertEqual(setup.linux_sockets(d), (2, 18))
        with tempfile.TemporaryDirectory() as d:
            self.fake(d, 1, 8)
            self.assertEqual(setup.linux_sockets(d), (1, 8))
        with tempfile.TemporaryDirectory() as d:
            self.assertIsNone(setup.linux_sockets(d))

    def test_two_socket_tip(self):
        t = setup.two_socket_note((2, 18))
        self.assertEqual(len(t), 1)
        self.assertIn("--pool-workers 17", t[0])
        self.assertEqual(setup.two_socket_note((1, 18)), [])
        self.assertEqual(setup.two_socket_note(None), [])


if __name__ == "__main__":
    unittest.main()
