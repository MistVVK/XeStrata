"""Auto with 2 requests at once on one large Blackwell card that holds every expert starts DFlash2 slot drafts (``--draft dflash2 --batch N``, no MTP layer);
every other class and every other number of requests keeps the rule it had (the MTP loaded, slots with no drafts). strata_control/drafting.py slot_drafts_by_default."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from strata_control import drafting

GYRO = "qwen3.8-flash-next-gyro-gyro-s"
IQ3S = "qwen3.8-flash-next-iq3_s"
PRO5000 = 73728.0          # MiB, the 72 GB card the rule was measured on (RTX PRO 5000)
BASE = dict(mode="auto", backend="cuda", archs=["sm_120"], vram_mib=PRO5000, args=["--expert-cache", "auto"], slots=2)


def rule(**over) -> bool:
    kw = {**BASE, **over}
    return drafting.slot_drafts_by_default(kw.pop("mode"), **kw)


_HARDWARE = mock.patch.object(drafting, "detect_hardware", return_value=("cuda", ["sm_120"]))


def setUpModule():
    _HARDWARE.start()


def tearDownModule():
    _HARDWARE.stop()


class TheRule(unittest.TestCase):
    def setUp(self):
        env = {k: v for k, v in os.environ.items() if k not in ("STRATA_BATCH_DF2", "STRATA_BATCH_MTP")}
        patch = mock.patch.dict(os.environ, env, clear=True)
        patch.start()
        self.addCleanup(patch.stop)

    def test_two_requests_on_the_measured_class_get_slot_drafts(self):
        self.assertTrue(rule(slots=2))
        self.assertFalse(rule(slots=3), "3 clients: DFlash2 slot drafts were level with plain slots (73.5 / 74.5 against 74.1 / 73.3 requests per minute)")

    def test_one_request_and_three_or_more_keep_their_rule(self):
        for n in (0, 1, 3, 4, 5, 8):
            self.assertFalse(rule(slots=n), n)

    def test_the_number_can_be_in_the_arguments(self):
        for flag in ("--batch", "--slots", "--parallel"):
            self.assertTrue(rule(slots=0, args=["--expert-cache", "auto", flag, "2"]), flag)
            self.assertFalse(rule(slots=0, args=["--expert-cache", "auto", flag, "3"]), flag)
        self.assertEqual(drafting.request_count(["--batch", "2"], 3), 3, "the larger of the two")
        self.assertEqual(drafting.request_count([], None), 0)
        self.assertEqual(drafting.request_count(["--batch", "x"], True), 0)

    def test_the_other_classes_stay_as_they_are(self):
        self.assertFalse(rule(archs=["sm_121"], vram_mib=None), "the DGX Spark: one pool of memory")
        self.assertFalse(rule(vram_mib=None), "several cards or a card that does not answer")
        self.assertFalse(rule(archs=["sm_86"], vram_mib=49152.0), "Ampere: DFlash2 was slower there")
        self.assertFalse(rule(archs=["sm_89"], vram_mib=49152.0))
        self.assertFalse(rule(vram_mib=32768.0), "a consumer Blackwell card under 48 GB")
        self.assertFalse(rule(args=["--expert-cache", "1000"]), "an expert cache that does not hold every expert (the streaming class)")
        for backend, archs in (("hip", ["gfx1151"]), ("hip", ["gfx1100"]), ("hip", ["gfx1201"]), ("metal", []), ("sycl", ["arc_a"]), ("cpu", []), (None, None)):
            self.assertFalse(rule(backend=backend, archs=archs, vram_mib=PRO5000 if backend == "hip" else None), (backend, archs))

    def test_only_auto_is_decided_here(self):
        for mode in ("mtp", "off", "dflash2"):
            self.assertFalse(rule(mode=mode), mode)       # (DFlash2 by name already runs its slots: blocked_by named)
        self.assertTrue(rule(mode=None), "no choice saved is Auto")

    def test_a_layer_split_the_person_choices_and_the_way_back_stay_as_before(self):
        self.assertFalse(rule(split=True))
        for extra in (["--layer-split", "auto"], ["--batch-mtp"], ["--pipeline-windows", "2"]):
            self.assertFalse(rule(args=["--expert-cache", "auto", *extra]), extra)
        with mock.patch.dict(os.environ, {"STRATA_BATCH_MTP": "1"}):
            self.assertFalse(rule(), "the person asked for the MTP's slot drafts")
        with mock.patch.dict(os.environ, {"STRATA_BATCH_MTP": "0"}):
            self.assertTrue(rule())
        with mock.patch.dict(os.environ, {"STRATA_BATCH_DF2": "0"}):
            self.assertFalse(rule(), "STRATA_BATCH_DF2=0 is the way back")
        with mock.patch.dict(os.environ, {"STRATA_BATCH_DF2": "1"}):
            self.assertTrue(rule())

    def test_no_drafter_on_this_computer_changes_nothing(self):
        self.assertFalse(rule(drafter_here=False), "no pack: the MTP stays, with its usual line")

    def test_blocked_by_lets_it_through_like_a_named_dflash2(self):
        self.assertTrue(drafting.blocked_by([], slots=2, backend="cuda"))
        self.assertFalse(drafting.blocked_by([], slots=2, backend="cuda", slot_drafts=True))
        self.assertTrue(drafting.blocked_by(["--pipeline-windows", "2"], slots=2, backend="cuda", slot_drafts=True))
        self.assertTrue(drafting.blocked_by([], slots=2, backend="metal", slot_drafts=True), "Metal has no per-request drafters")


class TheStart(unittest.TestCase):
    ARGS = ["--pack", "/d/packs/gyro", "--native", "/d/m.gguf", "--mtp", "", "--spec", "4", "--spec-min-p", "0.5", "--expert-cache", "auto"]

    def setUp(self):
        env = {k: v for k, v in os.environ.items() if k not in ("STRATA_BATCH_DF2", "STRATA_BATCH_MTP")}
        patch = mock.patch.dict(os.environ, env, clear=True)
        patch.start()
        self.addCleanup(patch.stop)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.data = Path(self.tmp.name)
        self.mtp = self.data / "mtp" / "rt"
        self.args = list(self.ARGS)
        self.args[self.args.index("--mtp") + 1] = str(self.mtp)
        self.mtp.mkdir(parents=True)
        (self.mtp / "experts.bin").write_bytes(b"e")
        (self.mtp / "draft_vocab.bin").write_bytes(b"v")

    def pack(self):
        pack = drafting.pack_path(self.data)
        pack.parent.mkdir(parents=True)
        pack.write_bytes(b"x" * (2 << 20))
        return pack

    def cfg(self, **extra):
        return {"model_name": GYRO, "args": list(self.args), **extra}

    def apply(self, cfg, **kw):
        return drafting.apply(cfg, data_dir=self.data, backend="cuda", archs=["sm_120"], vram_mib=PRO5000, **kw)

    def test_auto_with_two_requests_is_dflash2_with_a_drafter_per_request(self):
        pack = self.pack()
        for n in (2,):
            c = self.cfg()
            d = self.apply(c, slots=n)
            self.assertEqual((d.drafter, d.note), ("dflash2", ""), n)
            a = c["args"]
            self.assertEqual((a[a.index("--draft") + 1], a[a.index("--draft-model") + 1], a[a.index("--draft-fallback") + 1]), ("dflash2", str(pack), "none"), n)
            for flag in ("--mtp", "--spec-min-p", "--batch-mtp"):
                self.assertNotIn(flag, a, (n, flag, "never the MTP layer beside DFlash2"))

    def test_auto_with_other_numbers_of_requests_is_the_mtp_as_before(self):
        self.pack()
        for n in (3, 4, 8):
            c = self.cfg()
            d = self.apply(c, slots=n)
            self.assertEqual((d.drafter, d.note), ("mtp", drafting.NONE_IN_SLOTS), n)
            self.assertEqual(c["args"], self.args, n)

    def test_no_pack_is_the_mtp_with_the_usual_line(self):
        c = self.cfg()
        d = self.apply(c, slots=2)
        self.assertEqual((d.drafter, d.note), ("mtp", drafting.NONE_IN_SLOTS))
        self.assertEqual(c["args"], self.args)

    def test_the_ways_back(self):
        self.pack()
        for kw, want in (({"drafting": "mtp"}, ("mtp", "")), ({"drafting": "off"}, ("off", ""))):
            c = self.cfg(**kw)
            d = self.apply(c, slots=2)
            self.assertEqual((d.drafter, d.note), want, kw)
            self.assertNotIn("--draft", c["args"], kw)
        c = self.cfg()
        with mock.patch.dict(os.environ, {"STRATA_BATCH_DF2": "0"}):
            d = self.apply(c, slots=2)
        self.assertEqual((d.drafter, d.note, c["args"]), ("mtp", drafting.NONE_IN_SLOTS, self.args))

    def test_the_server_adds_the_slots_and_says_nothing_new(self):
        from serve.server import engine_args
        self.pack()
        with mock.patch("serve.server.drafting_vram_mib", return_value=PRO5000), mock.patch.object(drafting, "data_dir_of", return_value=self.data):
            for n in (2,):
                got = engine_args(self.cfg(parallel=n))
                self.assertEqual((got[got.index("--draft") + 1], got[got.index("--draft-fallback") + 1], got[got.index("--batch") + 1]), ("dflash2", "none", str(n)))
                self.assertNotIn("--mtp", got)
                self.assertEqual(drafting.take_note(), "", "no note: DFlash2 runs")
            for n in (3, 4):
                got = engine_args(self.cfg(parallel=n))
                self.assertEqual((got[:len(self.args)], got[-2:]), (self.args, ["--batch", str(n)]), "%d requests: the MTP loaded, slots with no drafts, as today" % n)
                self.assertEqual(drafting.take_note(), drafting.NONE_IN_SLOTS)
            # one request at a time: the same arguments with the rule on or off, byte for byte
            for parallel in (None, 1):
                cfg = self.cfg() if parallel is None else self.cfg(parallel=parallel)
                on = engine_args(cfg)
                with mock.patch.dict(os.environ, {"STRATA_BATCH_DF2": "0"}):
                    off = engine_args(cfg)
                self.assertEqual(on, off, parallel)
                self.assertNotIn("--batch", on)
            # a card that does not say its memory (several cards): as before
            with mock.patch("serve.server.drafting_vram_mib", return_value=None):
                got = engine_args(self.cfg(parallel=2))
                self.assertEqual(got[:len(self.args)], self.args)
            drafting.take_note()

    def lines(self, cfg, **env):
        """The `[strata]` lines the server prints while it builds the engine arguments for this config (nothing is started)."""
        import contextlib
        import io
        from serve.server import engine_args
        out = io.StringIO()
        with mock.patch("serve.server.drafting_vram_mib", return_value=PRO5000), mock.patch.object(drafting, "data_dir_of", return_value=self.data), \
                mock.patch.dict(os.environ, env), contextlib.redirect_stdout(out):
            engine_args(cfg)
        drafting.take_note()
        return [x for x in out.getvalue().splitlines() if "rafting" in x]

    def test_the_server_says_what_each_kind_of_start_drafts(self):
        self.pack()
        old_slots, old_none = "[strata] Drafting: MTP (DFlash2 isn't available in this mode)", "[strata] drafting: DFlash2 alone"
        # request slots that draft with DFlash2 (Auto on the measured class: 2 requests; or DFlash2 by name with any number of requests)
        want = ["[strata] Drafting: DFlash2, one guess window per request slot (spare rows only)"]
        self.assertEqual([x for x in self.lines(self.cfg(parallel=2)) if "vocab" not in x], want)
        self.assertEqual([x for x in self.lines(self.cfg(parallel=4, drafting="dflash2")) if "vocab" not in x], want)
        # request slots with no drafts: Auto with 3 or more requests, and the way back (STRATA_BATCH_DF2=0) with 2
        none = ["[strata] Drafting: none in parallel slots (one token per request and step); a single request still drafts"]
        self.assertEqual(self.lines(self.cfg(parallel=3)), none)
        self.assertEqual(self.lines(self.cfg(parallel=2), STRATA_BATCH_DF2="0"), none)
        # request slots whose slots carry MTP drafts (--batch-mtp, or STRATA_BATCH_MTP): the line that was always printed there stays, and it is true (the MTP drafts, DFlash2 is not the drafter)
        self.assertEqual(self.lines(self.cfg(parallel=3, args=[*self.args, "--batch-mtp"])), [old_slots])
        self.assertEqual(self.lines(self.cfg(parallel=3), STRATA_BATCH_MTP="1"), [old_slots])
        # one request at a time: today's DFlash2 line, word for word (the conf is the card's)
        for cfg in (self.cfg(), self.cfg(parallel=1), self.cfg(drafting="dflash2")):
            got = [x for x in self.lines(cfg) if "vocab" not in x]
            self.assertEqual(len(got), 1, cfg)
            self.assertRegex(got[0], r"^\[strata\] drafting: DFlash2 alone \(--draft dflash2, --draft-conf [0-9.]+\)$")
            self.assertNotIn("slot", got[0])
        self.assertNotIn(old_none, "".join(self.lines(self.cfg(parallel=2))), "slots with drafts no longer say 'DFlash2 alone'")

    def test_the_words_come_from_pure_functions(self):
        flag = ["--batch-mtp"]
        self.assertTrue(drafting.slots_draft_nothing([], 2) and drafting.slots_draft_nothing(["--batch", "4"], 0))
        self.assertFalse(drafting.slots_draft_nothing([], 1) or drafting.slots_draft_nothing([], 0), "one request at a time is not slots")
        self.assertFalse(drafting.slots_draft_nothing(flag, 3), "--batch-mtp: the slots carry MTP drafts")
        self.assertFalse(drafting.slots_draft_nothing(["--pipeline-windows", "2"], 3))
        with mock.patch.dict(os.environ, {"STRATA_BATCH_MTP": "1"}):
            self.assertFalse(drafting.slots_draft_nothing([], 3))
        with mock.patch.dict(os.environ, {"STRATA_BATCH_MTP": "0"}):
            self.assertTrue(drafting.slots_draft_nothing([], 3))
        self.assertTrue(drafting.slot_drafts_run([], 2) and drafting.slot_drafts_run(["--batch", "8"], 0))
        self.assertFalse(drafting.slot_drafts_run([], 1) or drafting.slot_drafts_run([], 0))
        self.assertFalse(drafting.slot_drafts_run([], 2, split=True) or drafting.slot_drafts_run(["--layer-split", "auto"], 2) or drafting.slot_drafts_run(["--pipeline-windows", "2"], 2))
        with mock.patch.dict(os.environ, {"STRATA_BATCH_DF2": "0"}):
            self.assertFalse(drafting.slot_drafts_run([], 2))
        self.assertEqual(drafting.line("mtp", drafting.NONE_IN_SLOTS), "Drafting: none in parallel slots (one token per request and step); a single request still drafts")
        self.assertEqual(drafting.line("mtp", drafting.NOT_HERE), "Drafting: MTP (DFlash2 isn't available in this mode)", "the other reasons keep their form")
        self.assertEqual(drafting.engine_info(["--mtp", "/m", "--batch", "3"], note=drafting.NONE_IN_SLOTS)["drafting_note"], drafting.line("mtp", drafting.NONE_IN_SLOTS), "/metrics and the app")

    def test_mtp_slot_drafts_keep_their_line_through_apply(self):
        self.pack()
        c = self.cfg(args=[*self.args, "--batch-mtp"])
        d = self.apply(c, slots=3)
        self.assertEqual((d.drafter, d.note), ("mtp", drafting.NOT_HERE))
        self.assertIn("--batch-mtp", c["args"])

    def plan_by_command(self, parallel, args=None, **env) -> str:
        """The plan line `strata drafting plan` shows for this config (strata_cli/drafting_cmd.py: _inputs, then decide)."""
        from strata_cli import drafting_cmd
        cfg = {**self.cfg(), **({} if parallel is None else {"parallel": parallel}), "args": list(args if args is not None else self.args)}
        path = self.data / "plan.json"
        path.write_text(__import__("json").dumps(cfg), encoding="utf-8")
        a = SimpleNamespace(config=str(path), data_dir=str(self.data), mode=None, family="qwen", model=None, no_split=True, engines_unloading=False, drafting_command="plan")
        with mock.patch.object(drafting, "card_vram_mib", return_value=PRO5000), mock.patch.object(drafting_cmd, "_hardware", return_value=("cuda", ["sm_120"])), \
                mock.patch.object(drafting_cmd, "_split_expected", return_value=False), mock.patch.dict(os.environ, env):
            want = drafting_cmd._inputs(a)
            sig = drafting_cmd._signals(a, want)
            return drafting.summary_line(drafting.decide(**want, **sig))

    def plan_by_launch(self, parallel, args=None, **env) -> str:
        """The plan line the Terminal's `strata launch --plan` shows (runner.drafting_plan_line) for an installed config like this one."""
        from strata_cli.launch import settings as S
        from strata_cli.launch.runner import Runner
        from tests.terminal.launch_support import HW_12, Home, catalog
        cfg = {**self.cfg(), "args": list(args if args is not None else self.args)}
        with Home():
            runner = Runner(S.LaunchSettings(family="qwen", model="IQ3_S", context=32768, vision="off", parallel=parallel), catalog=catalog(HW_12))
            runner._split_expected = False
            with mock.patch.object(Runner, "installed_state", return_value=(None, cfg)), mock.patch.object(drafting, "detect_hardware", return_value=("cuda", ["sm_120"])), \
                    mock.patch.object(drafting, "card_vram_mib", return_value=PRO5000), mock.patch.object(drafting, "data_dir_of", return_value=self.data), mock.patch.dict(os.environ, env):
                return runner.drafting_plan_line()

    def test_the_plan_line_says_what_the_server_says(self):
        self.pack()
        none = "Drafting: none in parallel slots (one token per request and step); a single request still drafts"
        mtp_slots = "Drafting: MTP (DFlash2 isn't available in this mode)"
        for how in (self.plan_by_command, self.plan_by_launch):
            name = how.__name__
            self.assertEqual(how(2), "Drafting: DFlash2", (name, "2 requests on the measured class: DFlash2 slot drafts"))
            self.assertEqual(how(3), none, (name, "slots with no drafts"))
            self.assertEqual(how(2, STRATA_BATCH_DF2="0"), none, (name, "the way back from the slot drafts"))
            self.assertEqual(how(3, args=[*self.args, "--batch-mtp"]), mtp_slots, (name, "MTP slot drafts keep their line"))
            self.assertEqual(how(3, STRATA_BATCH_MTP="1"), mtp_slots, (name, "MTP slot drafts through the environment"))
            self.assertEqual(how(1), "Drafting: DFlash2", (name, "one request at a time"))
            self.assertEqual(how(None), "Drafting: DFlash2", (name, "the number of requests is not known: today's line"))
        # the plan and the server print the same words in the same case
        self.assertEqual("[strata] " + self.plan_by_command(3), self.lines(self.cfg(parallel=3))[0])

    def test_the_plan_agrees_with_the_start(self):
        from strata_cli import drafting_cmd
        self.pack()
        cfg_path = self.data / "run.json"
        for parallel, expect_blocked in ((2, False), (3, True), (4, True), (1, False)):
            cfg_path.write_text(__import__("json").dumps({**self.cfg(), "parallel": parallel}), encoding="utf-8")
            a = SimpleNamespace(config=str(cfg_path), data_dir=str(self.data), mode=None, family="qwen", model=None, no_split=True, engines_unloading=False, drafting_command="plan")
            with mock.patch.object(drafting, "card_vram_mib", return_value=PRO5000), mock.patch.object(drafting_cmd, "_hardware", return_value=("cuda", ["sm_120"])), \
                    mock.patch.object(drafting_cmd, "_split_expected", return_value=False):
                want = drafting_cmd._inputs(a)
            self.assertEqual(want["blocked"], expect_blocked, parallel)


if __name__ == "__main__":
    unittest.main()
