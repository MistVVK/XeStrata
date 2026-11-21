"""strata_control/vram_fit.py: the fit math with mock card sizes, and its use in setup (sycl/setup_intel.py) and strata launch."""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from strata_control import vram_fit as V  # noqa: E402


class Measured(unittest.TestCase):
    """The estimate against the 12 starts on the Arc A750 (8 GB): engine slots / MiB short, from /data/022-arcgyro/out/fa_*.txt."""

    def slack(self, ctx, reserve, prefill):
        return V.slack_mib("Gyro-M", 8.0, ctx, "int8", reserve, prefill, "intel")

    def test_the_session_matches_the_engines_three_numbers(self):
        for ctx, mib in ((8192, 234), (16384, 347), (32768, 573)):
            self.assertAlmostEqual(V.session_mib(ctx), mib, delta=8)

    def test_the_default_does_not_start_and_the_estimate_says_how_short(self):
        self.assertAlmostEqual(self.slack(32768, 300, "auto"), -747, delta=40)     # engine: about 748 MiB more
        self.assertAlmostEqual(self.slack(16384, 300, "auto"), -497, delta=40)     # 497
        self.assertAlmostEqual(self.slack(8192, 100, "auto"), -171, delta=40)      # 171

    def test_the_known_good_start_has_room_for_its_slots(self):
        self.assertAlmostEqual(self.slack(8192, 100, 1024), 412 * V.SLOT_MIB, delta=60)    # 412 slots
        self.assertAlmostEqual(self.slack(32768, 100, 1024), 107 * V.SLOT_MIB, delta=60)   # 107 slots
        self.assertAlmostEqual(self.slack(8192, 100, 512), 683 * V.SLOT_MIB, delta=60)     # 683 slots
        self.assertAlmostEqual(self.slack(8192, 100, 256), 820 * V.SLOT_MIB, delta=60)     # 820 slots

    def test_gyro_s_default_fits_as_it_did(self):
        self.assertTrue(V.fits("Gyro-S", 8.0, 32768, "int8", 300, "auto", "intel"))        # 410 slots measured


class Plan(unittest.TestCase):
    """The rule (owner, 2026-11-02): a fit never changes the context or the number of requests; it moves work off the GPU, and says so."""

    def test_gyro_m_on_8gb_reads_prompts_in_smaller_chunks_and_keeps_its_context(self):
        f = V.plan("Gyro-M", 8.0, 32768, "int8", 300, "auto", "intel", title="Gyro-M")
        self.assertTrue(f.changed and f.fits)
        self.assertEqual((f.ctx, f.reserve_mib, f.prefill), (32768, 300, "512"))      # 32K stays; the chunk is the cheap knob, the reserve is not touched
        self.assertTrue(f.line.startswith("Gyro-M on this 8 GB card: " + V.NOTE_SLOW), f.line)
        self.assertGreaterEqual(f.slack_mib, V.needed_mib())

    def test_the_context_is_never_changed_for_any_card(self):
        for model in ("Gyro-M", "Gyro-S", "Qwen3.8-Flash-Next-IQ3_S"):
            for vendor in ("intel", "cuda", "other"):
                for gb in (5.0, 6.0, 8.0, 9.0, 10.0, 12.0, 16.0, 24.0, 48.0):
                    for ctx in (8192, 32768, 131072, 262144):
                        for slots in (0, 2, 4):
                            f = V.plan(model, gb, ctx, "int8", 700, "auto", vendor, slots=slots, kv_resident=32768 if ctx >= 65536 else 0)
                            self.assertEqual(f.ctx, ctx, (model, vendor, gb, ctx, slots))
                            self.assertEqual(f.slots, slots, (model, vendor, gb, ctx, slots))

    def test_a_chosen_reserve_is_kept(self):
        f = V.plan("Gyro-M", 8.0, 32768, "int8", 500, "auto", "intel", reserve_fixed=True)
        self.assertEqual(f.reserve_mib, 500)

    def test_a_chosen_chunk_is_kept(self):
        f = V.plan("Gyro-M", 8.0, 32768, "int8", 300, "2048", "intel", prefill_fixed=True)
        self.assertIsNone(f.prefill)

    def test_when_nothing_more_can_move_the_start_goes_as_far_as_it_can_and_says_so(self):
        f = V.plan("Gyro-M", 6.0, 32768, "int8", 300, "auto", "intel")
        self.assertEqual(f.ctx, 32768)
        self.assertEqual((f.prefill, f.reserve_mib), ("256", 100))        # every knob to its end
        self.assertFalse(f.fits)
        self.assertIn(V.NOTE_SLOW, f.line)

    def test_a_config_that_fits_comes_back_unchanged(self):
        for model, gb, ctx in (("Gyro-M", 12.0, 32768), ("Gyro-M", 16.0, 65536), ("Gyro-S", 8.0, 32768), ("Gyro-S", 12.0, 131072),
                               ("Qwen3.8-Flash-Next-IQ3_XXS", 8.0, 32768), ("Qwen3.8-Flash-Next-IQ3_XXS", 12.0, 131072),
                               ("Qwen3.8-Flash-Next-IQ3_XXS", 16.0, 262144), ("Qwen3.8-Flash-Next-IQ2_XS", 24.0, 262144)):
            f = V.plan(model, gb, ctx, "int8", 300, "auto", "nvidia")
            self.assertEqual((f.ctx, f.reserve_mib, f.prefill, f.changed, f.line), (ctx, 300, None, False, ""), (model, gb, ctx))

    def test_unknown_card_size_and_tiny_cards_are_left_alone(self):
        for gb in (0, None, 4.0):
            f = V.plan("Gyro-M", gb, 32768, "int8", 300, "auto", "intel")
            self.assertFalse(f.changed)

    def test_fit_context_is_the_context_asked_for(self):
        for ctx in (8192, 32768, 131072):
            self.assertEqual(V.fit_context("Gyro-M", 8.0, ctx), ctx)

    def test_generic_model_is_found_short_only_when_far_short(self):
        self.assertFalse(V.plan("Qwen3.8-Flash-Next-IQ3_XXS", 8.0, 32768, "int8", 300, "auto", "intel").changed)
        self.assertTrue(V.plan("Qwen3.8-Flash-Next-IQ3_XXS", 6.0, 131072, "int8", 300, "auto", "intel").changed)


class SetupIntel(unittest.TestCase):
    @staticmethod
    def mod():
        spec = importlib.util.spec_from_file_location("setup_intel_fit", ROOT.parent / "engine" / "sycl" / "setup_intel.py")
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        return m

    def cfg(self, m, native="Gyro-M-TQ2_0.gguf", extra=()):
        base = m.MOUNT
        return {"exe": "x", "cwd": "y", "args": ["--pack", str(base / "d" / "pack"), "--native", str(base / "d" / native),
                                                 "--expert-cache", "auto", "--prefill", "auto", "--max-context", "32768",
                                                 "--kv", "int8", "--vram-reserve-mib", "300", *extra]}

    def val(self, c, flag):
        return c["args"][c["args"].index(flag) + 1]

    def run_it(self, m, cfg, vram, drv="i915", **kw):
        notes = []
        c = m.to_sycl(cfg, m.ROOT / "build-sycl" / "strata", 64.0, {}, vram, drv, "56a1", fit_notes=notes, **kw)
        return c, notes

    def test_gyro_m_on_the_a750_gets_a_config_that_starts_and_one_line(self):
        m = self.mod()
        c, notes = self.run_it(m, self.cfg(m), 8.0, auto_reserve=True)
        self.assertEqual(self.val(c, "--max-context"), "32768")           # the context is the person's: it stays
        self.assertEqual(self.val(c, "--vram-reserve-mib"), "300")
        self.assertEqual(self.val(c, "--prefill"), "512")
        self.assertEqual(c["args"].count("--vram-reserve-mib"), 1)
        self.assertEqual(len(notes), 1)
        self.assertIn("Gyro-M on this 8 GB card: short of GPU memory", notes[0])

    def test_a_reserve_the_user_asked_for_is_kept(self):
        m = self.mod()
        c, notes = self.run_it(m, self.cfg(m), 8.0, auto_reserve=False)
        self.assertEqual(self.val(c, "--vram-reserve-mib"), "300")
        self.assertEqual(self.val(c, "--max-context"), "32768")

    def test_a_context_the_user_asked_for_is_kept(self):
        m = self.mod()
        c, notes = self.run_it(m, self.cfg(m), 8.0, auto_reserve=True, ctx_asked=True)
        self.assertEqual(self.val(c, "--max-context"), "32768")

    def test_configs_that_fit_are_byte_identical(self):
        m = self.mod()
        for vram, drv in ((8.0, "i915"), (12.0, "xe"), (16.0, "i915"), (24.0, "xe"), (32.0, "xe")):
            for native in ("a.gguf", "Gyro-S-TQ1_0.gguf"):
                off, _ = self.run_it(m, self.cfg(m, native), vram, drv, fit=False)
                on, notes = self.run_it(m, self.cfg(m, native), vram, drv, auto_reserve=True)
                self.assertEqual(off, on, (vram, drv, native))
                self.assertEqual(notes, [])


FN = "qwen3.8-flash-next-iq3_xxs"


class SlotSessions(unittest.TestCase):
    """Request slots ("Concurrent requests" 2 or more, the engine's --batch N): each slot is a session of its own, out of the expert cache's room.
    The 8 GB Arc A750 figures are the engine's: --batch 2 at 8192 tokens with the MTP layer and a 1024 MiB reserve left 2519 MiB free and the cache needed 99 MiB
    more (0 slots, a refused start; a SIGSEGV before ab7baa5b1)."""

    def test_the_slot_session_is_the_main_sessions_size(self):
        self.assertAlmostEqual(V.session_mib(8192), 234, delta=8)       # the engine: "2 slot sessions (0.23 GiB each)"

    def test_kv_streaming_caps_what_a_session_keeps_in_vram(self):
        full = V.session_mib(131072)
        streamed = V.session_mib(131072, "int8", 32768)
        self.assertAlmostEqual(full, 1929, delta=15)
        self.assertAlmostEqual(streamed, 748, delta=15)                  # measured 0.73 GiB on the CUDA engine (DF2-MEM): the 32K pool (573 MiB with its fixed part) plus what scales with the whole context
        self.assertEqual(V.session_mib(8192, "int8", 32768), V.session_mib(8192))     # a resident window larger than the context changes nothing
        self.assertAlmostEqual(V.session_mib(131072, "int8", 4096), V.session_mib(131072, "int8", V.KV_RESIDENT_MIN), delta=0.01)   # the engine raises a smaller window to its minimum (4096)

    def test_without_slots_nothing_changes(self):
        a = V.slack_mib(FN, 8.0, 8192, "int8", 700, "auto", "intel")
        self.assertEqual(a, V.slack_mib(FN, 8.0, 8192, "int8", 700, "auto", "intel", 0))
        self.assertEqual(a, V.slack_mib(FN, 8.0, 8192, "int8", 700, "auto", "intel", 1))
        self.assertEqual(V.needed_mib(), V.needed_mib(FN, 1))

    def test_the_engines_own_minimum_is_the_threshold(self):
        self.assertAlmostEqual(V.needed_mib(FN, 2, "intel"), V.SLOT_RUN_MIN * V.SLOT_SLOTTED_MIB + V.MARGIN_MIB, delta=0.01)    # one slot and the margin, not 128 slots
        self.assertGreater(V.needed_mib(FN, 0), 128 * 1.2)                                                                          # a single request keeps its old rule

    def test_an_8gb_arc_with_two_requests_and_the_refused_reserve_reads_prompts_in_smaller_chunks(self):
        # 8192 with a 1024 MiB reserve is the start the engine refused (the estimate says 137 MiB short; the engine 99)
        self.assertLess(V.slack_mib(FN, 8.0, 8192, "int8", 1024, "auto", "intel", 2), 0)
        f = V.plan(FN, 8.0, 8192, "int8", 1024, "auto", "intel", title=FN, reserve_fixed=True, slots=2)
        self.assertTrue(f.changed and f.fits)
        self.assertEqual((f.ctx, f.slots, f.reserve_mib, f.prefill), (8192, 2, 1024, "1024"))      # the context and the 2 requests stay
        self.assertGreaterEqual(f.slack_mib, V.needed_mib(FN, 2, "intel"))
        self.assertIn(V.NOTE_SLOW, f.line)

    def test_an_8gb_arc_whose_start_the_engine_runs_is_not_touched(self):
        # a 700 MiB reserve at 8192 leaves the cache about 84 slots: the engine runs it
        f = V.plan(FN, 8.0, 8192, "int8", 700, "auto", "intel", title=FN, reserve_fixed=True, prefill_fixed=True, slots=2)
        self.assertFalse(f.changed)
        self.assertEqual((f.ctx, f.slots), (8192, 2))

    def test_not_even_every_knob_fits_then_the_start_goes_as_far_as_it_can_with_the_same_requests(self):
        f = V.plan(FN, 6.0, 8192, "int8", 1500, "auto", "intel", title=FN, reserve_fixed=True, slots=2)
        self.assertEqual((f.ctx, f.slots), (8192, 2))
        self.assertEqual(f.prefill, "256")
        self.assertFalse(f.fits)

    def test_a_12gb_card_keeps_128k_with_streamed_kv(self):
        # setup writes --kv-resident 32768 at 128K: each of the 3 sessions keeps about 750 MiB (measured), not 1.9 GB
        for vendor in ("intel", "cuda"):
            f = V.plan(FN, 12.0, 131072, "int8", 700, "auto", vendor, title=FN, reserve_fixed=True, prefill_fixed=True, slots=2, kv_resident=32768)
            self.assertFalse(f.changed, vendor)
            self.assertEqual((f.ctx, f.slots), (131072, 2))
        self.assertGreater(V.slack_mib(FN, 12.0, 131072, "int8", 700, "auto", "intel", 2, 32768), 2000)
        # charging each slot the whole 128K would have lowered it (the mistake of 2a1042877)
        self.assertLess(V.slack_mib(FN, 12.0, 131072, "int8", 700, "auto", "intel", 2, 0), 0)

    def test_a_card_with_no_measured_figure_is_left_to_the_engine(self):
        f = V.plan(FN, 8.0, 8192, "int8", 1024, "auto", "other", title=FN, reserve_fixed=True, prefill_fixed=True, slots=2)
        self.assertFalse(f.changed)                # HIP: no measured base yet, so no claim; the engine sizes its cache and says what is short
        self.assertEqual(f.ctx, 8192)

    def test_a_24gb_cuda_card_is_unchanged(self):
        f = V.plan(FN, 24.0, 131072, "int8", 700, "auto", "cuda", title=FN, reserve_fixed=True, prefill_fixed=True, slots=4, kv_resident=32768)
        self.assertFalse(f.changed)
        self.assertEqual((f.ctx, f.slots, f.line), (131072, 4, ""))


class FitStart(unittest.TestCase):
    """fit_start: the engine arguments of one start (serve/server.py engine_args calls it).  It never changes --max-context and never the requests at once."""

    ARGS = ["--native", "/m/Qwen3.8-Flash-Next-IQ3_XXS-00001-of-00002.gguf", "--expert-cache", "auto", "--prefill", "auto", "--max-context", "8192",
            "--kv", "int8", "--vram-reserve-mib", "1024", "--mtp", "/m/rt", "--spec", "4"]

    def cfg(self, **kw):
        c = {"model_name": FN, "parallel": 2}
        c.update(kw)
        return c

    def test_an_8gb_arc_reads_prompts_in_smaller_chunks_and_keeps_context_and_requests(self):
        args, par, note = V.fit_start(self.cfg(), list(self.ARGS), 8192, "sycl")
        self.assertEqual(args[args.index("--max-context") + 1], "8192")
        self.assertEqual(args[args.index("--prefill") + 1], "1024")
        self.assertIsNone(par)                                 # still 2 requests at once
        self.assertIn(V.NOTE_SLOW, note)
        self.assertEqual(self.ARGS[self.ARGS.index("--prefill") + 1], "auto")           # the input is not touched

    def test_the_persons_own_reserve_and_chunk_are_kept(self):
        args, par, note = V.fit_start(self.cfg(user_args=["--vram-reserve-mib", "1024"]), list(self.ARGS), 8192, "sycl")
        self.assertEqual(args[args.index("--vram-reserve-mib") + 1], "1024")
        args2 = [("2048" if a == "auto" else a) for a in self.ARGS]
        self.assertEqual(V.fit_start(self.cfg(), list(args2), 8192, "sycl")[0][args2.index("--prefill") + 1], "2048")

    def test_a_12gb_card_at_128k_with_streamed_kv_is_unchanged(self):
        a = ["--native", "/m/Qwen3.8-Flash-Next-IQ3_S-00001-of-00002.gguf", "--max-context", "131072", "--kv", "int8", "--kv-resident", "32768",
             "--vram-reserve-mib", "700"]
        for backend in ("cuda", "sycl"):
            self.assertEqual(V.fit_start(self.cfg(model_name="qwen3.8-flash-next-iq3_s"), list(a), 12288, backend), (a, None, ""), backend)

    def test_a_24gb_cuda_card_and_one_request_are_unchanged(self):
        big = ["--native", "/m/q.gguf", "--max-context", "131072", "--kv", "int8", "--vram-reserve-mib", "700"]
        self.assertEqual(V.fit_start(self.cfg(parallel=4), list(big), 24576, "cuda"), (big, None, ""))
        self.assertEqual(V.fit_start(self.cfg(parallel=1), list(self.ARGS), 8192, "sycl"), (self.ARGS, None, ""))
        self.assertEqual(V.fit_start({"model_name": FN}, list(self.ARGS), 8192, "sycl"), (self.ARGS, None, ""))

    def test_nothing_without_one_cards_memory_or_on_a_split_or_metal(self):
        self.assertEqual(V.fit_start(self.cfg(), list(self.ARGS), None, "sycl"), (self.ARGS, None, ""))
        self.assertEqual(V.fit_start(self.cfg(), list(self.ARGS), 8192, "metal"), (self.ARGS, None, ""))
        self.assertEqual(V.fit_start(self.cfg(layer_split="auto"), list(self.ARGS), 8192, "sycl"), (self.ARGS, None, ""))

    def test_a_start_that_cannot_fit_keeps_its_requests_and_its_context(self):
        tight = [a if a != "1024" else "1500" for a in self.ARGS]
        args, par, note = V.fit_start(self.cfg(), tight, 8192, "sycl")
        self.assertIsNone(par)
        self.assertEqual(args[args.index("--max-context") + 1], "8192")
        self.assertIn(V.NOTE_SLOW, note)

    def test_no_context_in_the_args_is_never_added(self):
        no_ctx = list(self.ARGS)
        i = no_ctx.index("--max-context")
        del no_ctx[i:i + 2]
        args, par, note = V.fit_start(self.cfg(), no_ctx, 8192, "sycl")
        self.assertNotIn("--max-context", args)


class CudaSpill(unittest.TestCase):
    """The CUDA engine, measured on the RTX PRO 5000 held to 7942 / 8966 / 9990 / 12000 MiB by a memory hog (DF2-MEM, RESULTS-DF2MEM.md): an 8 GB NVIDIA card does not start
    the default 128K with the KV window setup writes (32768), and starts with the engine's minimum window (20480).  The knob is the window; the context and the requests stay."""

    M = "qwen3.8-flash-next-iq3_s"
    A = ["--native", "/m/Qwen3.8-Flash-Next-IQ3_S-00001-of-00002.gguf", "--max-context", "131072", "--kv", "int8", "--kv-resident", "32768",
         "--vram-reserve-mib", "700", "--prefill", "auto"]
    DF2 = ["--draft", "dflash2", "--draft-model", "/p/q4_0.pack", "--draft-fallback", "none", "--spec", "6", "--draft-conf", "0.8"]
    MTP = ["--mtp", "/m/rt", "--spec", "4", "--spec-min-p", "0.5"]

    def start(self, vram_mib, n=None, drafter=None, user=None, backend="cuda"):
        cfg = {"model_name": self.M}
        if n:
            cfg["parallel"] = n
        if user:
            cfg["user_args"] = user
        return V.fit_start(cfg, self.A + (drafter if drafter is not None else self.DF2), vram_mib, backend)

    @staticmethod
    def flag(args, name):
        return args[args.index(name) + 1] if name in args else None

    def test_the_base_reproduces_the_engines_cache_room(self):
        g = lambda free: (free + V.GAP_OTHER_MIB) / 1024.0
        sl = lambda free, n: V.slack_mib(self.M, g(free), 131072, "int8", 700, "auto", "cuda", n, 32768)
        self.assertAlmostEqual(sl(12000, 2), 3615 - 700 - 218 - 256, delta=60)     # "expert cache auto: 3.53 GiB free ... -> 961 slots"
        self.assertAlmostEqual(sl(12000, 4), 2120 - 700 - 218 - 256, delta=60)     # "2.07 GiB free ... -> 371 slots"
        self.assertAlmostEqual(sl(7800, 2), 850 - 835 - 598 - 700 - 218 - 256, delta=60)     # 0.83 GiB after the sessions, then the MTP layer and the head: the OOM

    def test_an_8gb_card_with_dflash2_alone_gets_the_small_kv_window_and_keeps_128k(self):
        for mib, window in ((8192, "20480"), (8050, None)):
            args, par, note = self.start(mib)
            self.assertEqual(self.flag(args, "--max-context"), "131072", mib)
            if window:
                self.assertEqual(self.flag(args, "--kv-resident"), window, mib)         # measured: starts and decodes at 69-73 tok/s; 32768 does not start
            else:
                self.assertIn(self.flag(args, "--kv-resident"), ("20480", "8192", "4096"), mib)   # a card a little smaller walks further down
            self.assertEqual(self.flag(args, "--prefill"), "auto")                       # the engine steps its own chunk down
            self.assertEqual(self.flag(args, "--vram-reserve-mib"), "700")               # a lower reserve starves the DFlash2 prompt staging (measured)
            self.assertIsNone(par)
            self.assertIn(V.NOTE_SLOW, note)

    def test_an_8gb_card_with_requests_at_once_keeps_the_requests_and_the_context(self):
        for n in (2, 4):
            args, par, note = self.start(8192, n, self.MTP)
            self.assertEqual(self.flag(args, "--max-context"), "131072", n)
            self.assertIsNone(par, n)                                                     # the number of requests is the person's
            self.assertEqual(self.flag(args, "--kv-resident"), "4096", n)               # measured: 2 requests start at 4096 (277 MiB free, 46 tok/s per user) and 8192; 32768 and 20480 do not
            self.assertIn(V.NOTE_SLOW, note)

    def test_the_window_is_walked_down_only_as_far_as_the_card_needs(self):
        win = lambda mib, n, drafter: self.flag(self.start(mib, n, drafter)[0], "--kv-resident")
        self.assertEqual(win(8192, None, self.DF2), "20480")       # DFlash2 alone fits at 20480
        self.assertEqual(win(8192, None, self.MTP), "8192")        # the MTP layer needs a smaller one
        self.assertEqual(win(10240, 2, self.MTP), "20480")
        self.assertEqual(win(10240, 4, self.MTP), "4096")
        self.assertEqual(win(12288, 8, self.MTP), "8192")
        self.assertEqual(win(12288, 4, self.MTP), "32768")         # nothing moves where it fits

    def test_the_window_floor_and_steps_match_the_engine(self):
        self.assertEqual(V.KV_RESIDENT_MIN, 4096)                  # qsa_kv_resident_min()
        self.assertEqual(V.KV_RESIDENT_STEPS, (20480, 8192, 4096))
        self.assertTrue(all(w >= V.KV_RESIDENT_MIN for w in V.KV_RESIDENT_STEPS))

    def test_12_16_and_24gb_keep_everything_for_1_2_and_4_requests(self):
        for mib in (12288, 12282, 16384, 24576):
            for n, drafter in ((None, self.DF2), (None, self.MTP), (2, self.MTP), (4, self.MTP)):
                a = self.A + drafter
                self.assertEqual(V.fit_start({"model_name": self.M, **({"parallel": n} if n else {})}, list(a), mib, "cuda"), (a, None, ""), (mib, n))

    def test_the_8gb_figures_behind_it(self):
        gb = (7942 + V.GAP_OTHER_MIB) / 1024.0
        one = V.SLOT_RUN_MIN * V.SLOT_SLOTTED_MIB + V.MARGIN_MIB
        sl = lambda kvr, drafter: V.slack_mib(self.M, gb, 131072, "int8", 700, "auto", "cuda", 0, kvr, drafter)
        self.assertLess(sl(32768, "dflash2"), one)             # did not start: "the device arena (75 MiB) does not fit"
        self.assertGreaterEqual(sl(20480, "dflash2"), one)     # started
        self.assertLess(sl(32768, "mtp"), one)

    def test_a_window_the_person_set_is_kept(self):
        args, par, note = self.start(8192, user=["--kv-resident", "32768"])
        self.assertEqual(self.flag(args, "--kv-resident"), "32768")

    def test_metal_and_unknown_backends_are_not_claimed(self):
        for backend in ("metal", "vulkan", None, ""):
            a = self.A + self.DF2
            self.assertEqual(V.fit_start({"model_name": self.M}, list(a), 8192, backend), (a, None, ""), backend)

    def test_the_hip_base_is_its_own_measurement(self):
        hip, cuda = V.BASE_SLOTTED_MIB[("hip", "flash-next")], V.BASE_SLOTTED_MIB[("cuda", "flash-next")]
        self.assertNotEqual(hip, cuda)
        self.assertAlmostEqual(cuda - hip, 1090, delta=200)           # the R9700 start holds about 1.1 GiB less before the cache than the CUDA one (FREE - 5045 against FREE - 6324, less the context)
        self.assertTrue(V.HIP_BASE_NOTE.startswith("measured on an R9700"))

    def test_an_8gb_discrete_amd_card_starts_as_set_up_and_a_6gb_one_walks_the_window(self):
        # measured on the R9700 held to 7942 MiB: DFlash2 alone starts at 128K with setup's window (787 slots, 864 MiB free); at 6400 MiB it needs the loan that keeps 16
        for mib in (8192, 8176):
            a = self.A + self.DF2
            self.assertEqual(V.fit_start({"model_name": self.M}, list(a), mib, "hip"), (a, None, ""), mib)
        args, par, note = self.start(6144, backend="hip")
        self.assertEqual(self.flag(args, "--max-context"), "131072")
        self.assertIn(self.flag(args, "--kv-resident"), ("20480", "8192", "4096"))
        self.assertIsNone(par)
        self.assertIn(V.NOTE_SLOW, note)
        for mib in (12288, 16384, 24576, 32768):
            a = self.A + self.DF2
            self.assertEqual(V.fit_start({"model_name": self.M}, list(a), mib, "hip"), (a, None, ""), mib)
        for n in (2, 4):
            args, par, note = self.start(8192, n, self.MTP, backend="hip")
            self.assertEqual(self.flag(args, "--max-context"), "131072", n)
            self.assertIsNone(par, n)                                        # the number of requests is never touched

    def test_an_integrated_amd_chip_is_left_alone(self):
        # serve/server.py drafting_vram_mib() returns None for Strix Halo and the other APUs (one pool with the host's RAM): fit_start does nothing without a card's memory
        a = self.A + self.DF2
        self.assertEqual(V.fit_start({"model_name": self.M}, list(a), None, "hip"), (a, None, ""))

    def test_the_drafter_is_read_from_the_arguments(self):
        self.assertEqual(V.drafter_of(self.A + self.DF2), "dflash2")
        self.assertEqual(V.drafter_of(self.A + ["--mtp", "/m", "--draft", "dflash2", "--draft-model", "/p"]), "dflash2+mtp")
        self.assertEqual(V.drafter_of(self.A + self.MTP), "mtp")
        self.assertEqual(V.drafter_of(self.A), "none")



class Reserve3750(unittest.TestCase):
    """#3750: 256K context + Q4 KV on a small card.  The app's reserve for a card of 16 GB or less is 1536 MiB (app/host native-resources.mjs), no longer a fifth of the
    card (3256 on the reporter's RX 9070 XT).  The fit arithmetic prices the expert cache that is left, and it is positive at 12 GiB with the app's reserve on both
    discrete backends; the old 3256 left the CUDA engine's measured base at its threshold (a start on the edge, which the reporter's 83 GB model fell over)."""

    M = "Qwen3.8-Flash-Next-IQ3_S"

    def slack(self, gb, reserve, vendor, kv_resident=0, ctx=262144):
        return V.slack_mib(self.M, gb, ctx, "q4_0", reserve, "auto", vendor, 0, kv_resident, "mtp")

    def test_a_12_gib_card_at_256k_q4_has_a_positive_cache_with_the_apps_reserve(self):
        for vendor in ("cuda", "hip"):
            s = self.slack(12.0, 1536, vendor)
            self.assertGreater(s, V.needed_mib(self.M, 0, vendor, "mtp") + 1000, (vendor, s))
            self.assertTrue(V.fits(self.M, 12.0, 262144, "q4_0", 1536, "auto", vendor, 0, 0, "mtp"), vendor)

    def test_the_old_fifth_of_the_card_takes_the_room_the_reserve_cap_gives_back(self):
        for gb, old in ((12.0, 2457), (15.9, 3256)):
            for vendor in ("cuda", "hip"):
                self.assertAlmostEqual(self.slack(gb, 1536, vendor) - self.slack(gb, old, vendor), old - 1536, delta=1)
        # on a 12 GiB CUDA card the old reserve left the cache nothing beyond the working threshold
        self.assertLess(self.slack(12.0, 3256, "cuda") - V.needed_mib(self.M, 0, "cuda", "mtp"), 100)

    def test_a_fit_never_changes_the_context_and_a_fitting_start_is_returned_byte_identical(self):
        args = ["--native", "m.gguf", "--max-context", "262144", "--kv", "q4_0", "--vram-reserve-mib", "1536", "--mtp", "mtp.gguf"]
        for mib in (12288, 16282, 24576):
            out = V.fit_start({"model_name": self.M}, list(args), mib, "cuda")
            self.assertEqual(out[0], args, mib)
            self.assertEqual(out[0][out[0].index("--max-context") + 1], "262144")
        # a start that would be short keeps its context and its reserve and moves KV to RAM instead
        out = V.fit_start({"model_name": self.M}, list(args), 6144, "cuda")
        self.assertEqual(out[0][out[0].index("--max-context") + 1], "262144")
        self.assertEqual(out[0][out[0].index("--vram-reserve-mib") + 1], "1536")

if __name__ == "__main__":
    unittest.main()
