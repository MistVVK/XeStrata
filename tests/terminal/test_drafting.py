"""Which drafter a model uses (MTP, DFlash2 or none): the suggestion table, the Terminal flag, the engine arguments, the download and the
fallback words. strata_control/drafting.py is the one place; the Terminal and the app read it."""

from __future__ import annotations

import http.server
import importlib.util
import json
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

from strata_cli.launch import settings as S
from strata_cli.launch.runner import derive_config
from strata_control import acquisition, drafting, drafting_fetch
from tests.terminal.launch_support import run_cli

ROOT = Path(__file__).resolve().parents[2]
GYRO = "qwen3.8-flash-next-gyro-gyro-s"
_HARDWARE = mock.patch.object(drafting, "detect_hardware", return_value=("cuda", ["sm_120"]))


def setUpModule():
    """These tests are about the table, the flags and the download, on a computer whose engine can run DFlash2 (the box that runs them may have an Intel or no card;
    tests/terminal/test_drafting_hardware.py has the hardware matrix)."""
    _HARDWARE.start()


def tearDownModule():
    _HARDWARE.stop()


def serving_dflash2():
    """The tests turn the switch on explicitly (drafting.ENGINE_SERVES_DFLASH2)."""
    return mock.patch.object(drafting, "ENGINE_SERVES_DFLASH2", True)


class AutoAndTheBackends(unittest.TestCase):
    def test_auto_is_dflash2_where_the_engine_has_it(self):
        """Owner 2026-11-02: DFlash2 is the primary drafter on every system whose engine has it (CUDA, HIP, Metal, Intel SYCL since the DF2-SYCL engine)."""
        self.assertEqual(drafting.DFLASH2_BACKENDS, ("cuda", "hip", "metal", "sycl"))
        for backend in ("cuda", "hip", "metal", "sycl"):
            for family in ("qwen", "gyro", "unsloth"):
                self.assertEqual(drafting.choice_of("auto", family=family, backend=backend), "dflash2", (backend, family))
        self.assertEqual(drafting.choice_of("auto", family="qwen", backend=None), "mtp", "an unknown computer is the MTP")
        self.assertEqual(drafting.choice_of("auto", family="qwen", backend="cpu"), "mtp", "an engine without the code is the MTP")
        for family in ("coder", "swift", "nvfp4", "deepseek41"):
            self.assertEqual(drafting.choice_of("auto", family=family, backend="cuda"), "mtp", family)

    def test_intel_runs_dflash2_and_a_build_without_the_code_does_not(self):
        self.assertTrue(drafting.hardware_support("sycl", ["8086"]).can_run)
        with mock.patch.object(drafting, "DFLASH2_BACKENDS", ("cuda", "hip", "metal")):
            self.assertEqual(drafting.choice_of("auto", family="qwen", backend="sycl"), "mtp")
            self.assertFalse(drafting.hardware_support("sycl").can_run)

    def test_the_choice_is_the_choice(self):
        for backend in ("cuda", "sycl"):
            self.assertEqual(drafting.choice_of("mtp", family="qwen", backend=backend), "mtp")
            self.assertEqual(drafting.choice_of("off", family="qwen", backend=backend), "off")
        self.assertEqual(drafting.choice_of("dflash2", family="qwen", backend="cuda"), "dflash2")
        self.assertEqual(drafting.choice_of("dflash2", family="qwen", backend="sycl"), "dflash2")
        self.assertEqual(drafting.choice_of("dflash2", family="qwen", backend="cpu"), "mtp", "named, but the engine has none")
        self.assertEqual(drafting.choice_of("dflash2", family="coder", backend="cuda"), "dflash2", "#5240: the Coder runs DFlash2 when it is chosen by name")
        self.assertEqual(drafting.choice_of("dflash2", family="swift", backend="cuda"), "mtp")

    def test_the_two_drafters_have_their_own_memory_figure(self):
        self.assertEqual((drafting.DRAFTER_GB["dflash2"], drafting.DRAFTER_GB["mtp"], drafting.DRAFTER_GB["off"]), (0.89, 1.08, 0.0))
        self.assertEqual(drafting.VRAM_GB, 0.89)

    def test_cards_are_classified_by_their_architecture(self):
        self.assertEqual(drafting.card_class(["sm_120"]), "blackwell")
        self.assertEqual(drafting.card_class(["gfx1151"]), "strix_halo")
        self.assertEqual(drafting.card_class(["gfx1201"]), "rdna4")
        self.assertEqual(drafting.card_class(["sm_86"]), "ampere")
        self.assertEqual(drafting.card_class([]), "other")
        self.assertEqual(drafting.card_class("SM_120"), "blackwell")

    def test_both_sides_know_the_same_modes(self):
        self.assertEqual(drafting.MODES, S.DRAFTING_MODES)
        self.assertEqual(drafting.normalize(None), "auto")
        self.assertEqual(drafting.normalize("DFlash2"), "dflash2")
        self.assertEqual(drafting.normalize("fastest"), "auto", "an unknown value is Auto, never an error")


class Decisions(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.data = Path(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)

    def ready(self):
        pack = drafting.pack_path(self.data)
        pack.parent.mkdir(parents=True)
        pack.write_bytes(b"x" * (2 << 20))

    def decide(self, mode, **kw):
        return drafting.decide(mode, model_name=GYRO, data_dir=self.data, backend="cuda", archs=["sm_120"], **kw)

    def test_auto_and_dflash2_run_dflash2_alone_when_the_pack_is_here(self):
        self.ready()
        for mode in ("auto", "dflash2", None):
            d = self.decide(mode)
            self.assertEqual((d.drafter, d.note), ("dflash2", ""), mode)
            self.assertEqual(d.flags, ("--draft", "dflash2", "--draft-model", str(drafting.pack_path(self.data)), "--draft-fallback", "none",
                                       "--spec", "6", "--draft-conf", "0.35"), "no --mtp, and the engine is told there is no fallback")
        self.assertNotIn("--mtp", self.decide("auto").flags)

    def test_mtp_is_the_mtp_with_no_dflash2_flag(self):
        d = self.decide("mtp")
        self.assertEqual((d.drafter, d.flags, d.note), ("mtp", (), ""))

    def test_off_is_off(self):
        self.assertEqual(self.decide("off").drafter, "off")

    def test_no_pack_and_the_mtp_layer_here_runs_the_mtp_and_says_so(self):
        d = self.decide("auto", mtp_here=True)
        self.assertEqual((d.drafter, d.note), ("mtp", drafting.NOT_DOWNLOADED))
        self.assertTrue(drafting.needs_pack("auto", model_name=GYRO, data_dir=self.data, backend="cuda"))

    def test_no_pack_and_no_mtp_layer_runs_none_and_says_so(self):
        d = self.decide("dflash2", mtp_here=False)
        self.assertEqual((d.drafter, d.note, d.flags), ("off", drafting.NONE_HERE, ()))
        self.assertEqual(drafting.line(d.drafter, d.note), "Drafting: off (no drafter is downloaded yet)")

    def test_mtp_chosen_without_its_layer_runs_dflash2_if_it_is_here_else_none(self):
        d = self.decide("mtp", mtp_here=False)
        self.assertEqual((d.drafter, d.note), ("off", drafting.NONE_HERE))
        self.ready()
        d = self.decide("mtp", mtp_here=False)
        self.assertEqual((d.drafter, d.note), ("dflash2", drafting.NO_MTP))
        self.assertEqual(drafting.line(d.drafter, d.note), "Drafting: DFlash2 (the MTP drafter isn't downloaded yet)")
        self.assertTrue(drafting.needs_mtp("mtp", model_name=GYRO, backend="cuda", mtp_here=False), "the switch fetches the layer once")
        self.assertFalse(drafting.needs_mtp("mtp", model_name=GYRO, backend="cuda", mtp_here=True))
        self.assertFalse(drafting.needs_mtp("auto", model_name=GYRO, backend="cuda", mtp_here=False), "Auto is DFlash2: it needs no layer")

    def test_several_requests_at_once_load_only_the_mtp_and_say_so(self):
        self.ready()
        d = self.decide("dflash2", mtp_here=True, blocked=True)
        self.assertEqual((d.drafter, d.note, d.flags), ("mtp", drafting.NOT_HERE, ()))
        d = self.decide("auto", mtp_here=True, blocked=True)
        self.assertEqual((d.drafter, d.note), ("mtp", drafting.NOT_HERE), "Auto says it too: nothing changes silently")
        self.assertTrue(drafting.blocked_by([], slots=4))
        self.assertTrue(drafting.blocked_by(["--batch", "4"]) and drafting.blocked_by(["--pipeline-windows", "2"]))
        # DFlash2 chosen by name (owner 2026-11-03: the person's own choice wins): several requests are their word too, only --pipeline-windows still blocks
        for args, kw in (([], {"slots": 4}), (["--batch", "4"], {}), (["--slots", "2"], {}), (["--parallel", "3"], {}), (["--batch", "4"], {"slots": 4, "split": True})):
            self.assertFalse(drafting.blocked_by(args, named=True, **kw), (args, kw))
            self.assertTrue(drafting.blocked_by(args, **kw), (args, kw, "Auto and the MTP keep their rule"))
        self.assertTrue(drafting.blocked_by(["--pipeline-windows", "2"], named=True), "the MTP's speculative next window has no DFlash2 counterpart")
        self.assertFalse(drafting.blocked_by([], split=True), "a layer split over several cards runs DFlash2 (owner 2026-11-02; the 0.1.56 engine does)")
        self.assertFalse(drafting.blocked_by(["--layer-split", "auto"]))
        self.assertFalse(drafting.blocked_by(["--spec", "4"], slots=1))

    def test_dflash2_is_for_qwen38_flash_next_only(self):
        self.ready()
        for name in ("qwen3.6-35b-a3b-q4", "qwen3.8-flash-next-swift-iq2_xs", None):
            d = drafting.decide("dflash2", model_name=name, data_dir=self.data, backend="cuda")
            self.assertEqual((d.drafter, d.note), ("mtp", drafting.NOT_SUPPORTED), name)
            self.assertFalse(drafting.needs_pack("dflash2", model_name=name, data_dir=self.data, backend="cuda"))
            self.assertEqual(drafting.decide("auto", model_name=name, data_dir=self.data, backend="cuda").note, "", "Auto never nags about a model DFlash2 is not for")
        for name in (GYRO, "qwen3.8-flash-next-q2_0", "qwen3.8-flash-next-iq3_xxs"):
            self.assertEqual(drafting.decide("dflash2", model_name=name, data_dir=self.data, backend="cuda").drafter, "dflash2", name)
        self.assertTrue(drafting.supported_model(family="qwen") and drafting.supported_model(family="gyro"))
        self.assertFalse(drafting.supported_model(family="coder") or drafting.supported_model(family="swift") or drafting.supported_model(family="nvfp4"))
        coder = "qwen3.8-flash-next-coder-iq3_xxs"                 # #5240: the Coder runs DFlash2 when it is chosen by name; Auto keeps the MTP
        self.assertEqual(drafting.decide("dflash2", model_name=coder, data_dir=self.data, backend="cuda").drafter, "dflash2")
        self.assertEqual(drafting.decide("auto", model_name=coder, data_dir=self.data, backend="cuda").drafter, "mtp")

    def test_the_engine_of_this_branch_serves_dflash2(self):
        self.assertTrue(drafting.ENGINE_SERVES_DFLASH2, "src/program/generate.cpp: --draft dflash2 runs with --serve (one card, one request at a time)")

    def test_while_the_engine_cannot_serve_it_the_choice_says_so_and_nothing_is_downloaded(self):
        with mock.patch.object(drafting, "ENGINE_SERVES_DFLASH2", False):
            d = self.decide("dflash2")
            self.assertEqual((d.drafter, d.note), ("mtp", drafting.NOT_SERVED))
            self.assertFalse(drafting.needs_pack("dflash2", model_name=GYRO, data_dir=self.data, backend="cuda"))
            self.assertEqual(self.decide("auto").note, "")
            self.assertEqual(self.decide("auto").drafter, "mtp")

    def test_intel_is_dflash2_and_fetches_its_pack(self):
        self.assertTrue(drafting.needs_pack("auto", model_name=GYRO, data_dir=self.data, backend="sycl"))
        d = drafting.decide("auto", model_name=GYRO, data_dir=self.data, backend="sycl")
        self.assertEqual(d.drafter, "mtp", "the pack is not downloaded yet: the MTP until it is")
        d = drafting.decide("auto", model_name=GYRO, data_dir=self.data, backend="cpu")
        self.assertEqual((d.drafter, d.note), ("mtp", ""))
        self.assertFalse(drafting.needs_pack("auto", model_name=GYRO, data_dir=self.data, backend="cpu"))


class EngineArguments(unittest.TestCase):
    ARGS = ["--pack", "/d/packs/gyro", "--native", "/d/m.gguf", "--mtp", "/d/mtp/rt", "--spec", "4", "--spec-min-p", "0.5"]

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.data = Path(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)
        self.vocab = self.data / "draft_vocab.bin"
        self.vocab.write_bytes(b"v")
        self.mtp = self.data / "mtp" / "rt"
        self.args = ["--pack", "/d/packs/gyro", "--native", "/d/m.gguf", "--mtp", str(self.mtp), "--spec", "4", "--spec-min-p", "0.5"]

    def layer(self):
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
        return drafting.apply(cfg, data_dir=self.data, backend="cuda", archs=["sm_120"], **kw)

    def test_dflash2_alone_has_no_mtp_option_and_says_there_is_no_fallback(self):
        pack = self.pack()
        self.layer()
        for drafting_mode in ("dflash2", "auto", None):
            c = self.cfg(args=[*self.args, "--batch-mtp", "--mtp-q4", "all"], **({} if drafting_mode is None else {"drafting": drafting_mode}))
            d = self.apply(c)
            self.assertEqual(d.drafter, "dflash2", drafting_mode)
            a = c["args"]
            for flag in ("--mtp", "--spec-min-p", "--batch-mtp", "--mtp-q4", "--draft-mtp-spec"):
                self.assertNotIn(flag, a, (drafting_mode, flag))
            self.assertEqual(a[a.index("--draft") + 1], "dflash2")
            self.assertEqual(a[a.index("--draft-model") + 1], str(pack))
            self.assertEqual(a[a.index("--draft-fallback") + 1], "none")
            self.assertEqual(a[a.index("--mtp-draft-vocab") + 1], str(self.mtp / "draft_vocab.bin"), "the subset the person chose in setup")
            self.assertEqual((a[a.index("--spec") + 1], a[a.index("--draft-conf") + 1]), ("6", "0.35"))
            self.assertEqual(a.count("--spec"), 1)

    def test_dflash2_alone_works_when_no_mtp_layer_was_ever_downloaded(self):
        self.pack()
        with mock.patch.object(drafting, "vocab_path", return_value=self.vocab):
            c = self.cfg(drafting="dflash2")
            d = self.apply(c)
        self.assertEqual((d.drafter, d.note), ("dflash2", ""))
        self.assertEqual(c["args"][c["args"].index("--mtp-draft-vocab") + 1], str(self.vocab), "Strata's own default subset")
        self.assertNotIn("--mtp", c["args"])

    def test_mtp_has_no_dflash2_option(self):
        self.pack()
        self.layer()
        c = self.cfg(drafting="mtp", args=[*self.args, "--draft", "dflash2", "--draft-model", "/p", "--draft-fallback", "mtp", "--draft-conf", "0.3", "--draft-mtp-spec", "4",
                                          "--draft-cap", "5", "--draft-rule", "df2"])
        d = self.apply(c)
        self.assertEqual((d.drafter, c["args"]), ("mtp", self.args), "exactly the MTP's own arguments")

    def test_a_config_that_names_both_starts_with_the_one_chosen(self):
        pack = self.pack()
        self.layer()
        both = [*self.args, "--draft", "dflash2", "--draft-model", str(pack), "--draft-fallback", "mtp", "--spec", "6"]
        mtp = self.cfg(drafting="mtp", args=list(both))
        self.apply(mtp)
        self.assertEqual(mtp["args"].count("--mtp"), 1)
        self.assertNotIn("--draft", mtp["args"])
        two = self.cfg(drafting="dflash2", args=list(both))
        self.apply(two)
        self.assertNotIn("--mtp", two["args"])
        self.assertEqual(two["args"][two["args"].index("--draft-fallback") + 1], "none")
        self.assertEqual(two["args"].count("--draft-fallback"), 1)
        # Auto picks DFlash2 for it
        auto = self.cfg(args=list(both))
        self.apply(auto)
        self.assertNotIn("--mtp", auto["args"])
        self.assertEqual(auto["args"][auto["args"].index("--draft") + 1], "dflash2")

    def test_the_persons_own_spec_and_conf_stay(self):
        self.pack()
        self.layer()
        c = self.cfg(drafting="dflash2", args=[*self.args[:self.args.index("--spec")], "--spec", "5"], user_args=["--spec", "5", "--draft-conf", "0.2"])
        self.apply(c)
        self.assertEqual((c["args"][c["args"].index("--spec") + 1], c["args"].count("--spec"), c["args"][c["args"].index("--draft-conf") + 1]), ("5", 1, "0.2"))

    def test_parallel_with_auto_loads_only_the_mtp_and_says_so(self):
        self.pack()
        self.layer()
        for kw in ({"slots": 4},):
            c = self.cfg()                                   # Auto: unchanged, the MTP for several requests
            d = self.apply(c, **kw)
            self.assertEqual((d.drafter, d.note), ("mtp", drafting.NONE_IN_SLOTS), kw)
            self.assertEqual(c["args"], self.args, kw)
            c = self.cfg(drafting="mtp")                     # the MTP chosen: unchanged
            d = self.apply(c, **kw)
            self.assertEqual((d.drafter, d.note), ("mtp", ""), kw)
            self.assertEqual(c["args"], self.args, kw)

    def test_dflash2_chosen_by_name_keeps_dflash2_for_the_requests_the_person_chose(self):
        """Owner 2026-11-03: DFlash2 chosen + a number of requests chosen = DFlash2 with that many (the engine runs a drafter per request)."""
        self.pack()
        self.layer()
        for kw in ({"slots": 2}, {"slots": 4}, {"slots": 8}):
            c = self.cfg(drafting="dflash2")
            d = self.apply(c, **kw)
            self.assertEqual((d.drafter, d.note), ("dflash2", ""), kw)
            a = c["args"]
            self.assertEqual((a[a.index("--draft") + 1], a[a.index("--draft-fallback") + 1]), ("dflash2", "none"), kw)
            self.assertNotIn("--mtp", a)
            self.assertNotIn("--batch-mtp", a, "the MTP's slot switch is the MTP's; the slot drafts are the engine's default for --draft dflash2 --batch")
        # the number in the arguments themselves (Expert mode / the saved --batch) is the same word
        c = self.cfg(drafting="dflash2", args=[*self.args, "--batch", "4", "--batch-mtp"])
        d = self.apply(c)
        self.assertEqual((d.drafter, d.note), ("dflash2", ""))
        self.assertEqual(c["args"][c["args"].index("--batch") + 1], "4")
        self.assertNotIn("--batch-mtp", c["args"])
        # on a layer split too
        c = self.cfg(drafting="dflash2")
        self.assertEqual(self.apply(c, split=True, slots=4).drafter, "dflash2")
        # one request (1, or none given) was always DFlash2
        for kw in ({"slots": 0}, {"slots": 1}):
            self.assertEqual(self.apply(self.cfg(drafting="dflash2"), **kw).drafter, "dflash2", kw)
        # --pipeline-windows is the one thing that still keeps it out
        c = self.cfg(drafting="dflash2", args=[*self.args, "--pipeline-windows", "2"])
        d = self.apply(c, slots=4)
        self.assertEqual((d.drafter, d.note), ("mtp", drafting.NOT_HERE))

    def test_no_pack_no_internet_runs_the_mtp_when_it_is_here_else_none(self):
        self.layer()
        c = self.cfg(drafting="dflash2")
        d = self.apply(c)
        self.assertEqual((d.drafter, d.note, c["args"]), ("mtp", drafting.NOT_DOWNLOADED, self.args))
        self.mtp.joinpath("experts.bin").unlink()
        c = self.cfg(drafting="dflash2")
        d = self.apply(c)
        self.assertEqual((d.drafter, d.note), ("off", drafting.NONE_HERE))
        for flag in ("--mtp", "--draft", "--spec-min-p"):
            self.assertNotIn(flag, c["args"])

    def test_the_mtp_switch_off_with_no_drafter_chosen_since_means_no_drafter(self):
        self.pack()
        self.layer()
        c = self.cfg(mtp=False)
        self.assertEqual(self.apply(c).drafter, "off")
        c = self.cfg(mtp=False, drafting="dflash2")
        self.assertEqual(self.apply(c).drafter, "dflash2", "a drafter chosen by name is the person's later word")

    def test_another_model_is_left_alone(self):
        c = self.cfg(model_name="qwen3.8-flash-next-coder-iq3_xxs")
        d = self.apply(c)
        self.assertEqual((d.drafter, d.note, c["args"]), ("mtp", "", self.args))
        c = self.cfg(model_name="qwen3.8-flash-next-swift-iq2_xs", drafting="dflash2", args=[*self.args, "--draft", "dflash2", "--draft-model", "/p"])   # (the Coder named: test_drafting_coder_named.py)
        d = self.apply(c)
        self.assertEqual((d.drafter, d.note, c["args"]), ("mtp", drafting.NOT_SUPPORTED, self.args))

    def test_intel_gets_its_own_confidence_not_the_cuda_slot_steps(self):
        self.layer()
        self.pack()
        for vram in (None, 8000, 24000, 96000):
            c = self.cfg()
            d = drafting.apply(c, data_dir=self.data, backend="sycl", archs=[], vram_mib=vram)
            a = c["args"]
            self.assertEqual((d.drafter, a[a.index("--draft-conf") + 1], a[a.index("--spec") + 1], "--mtp" in a, a[a.index("--draft-fallback") + 1]), ("dflash2", "0.6", "6", False, "none"), vram)
        c = self.cfg(user_args=["--draft-conf", "0.5"], args=[*self.args, "--draft-conf", "0.5"])
        drafting.apply(c, data_dir=self.data, backend="sycl", archs=[], vram_mib=8000)
        self.assertEqual(c["args"][c["args"].index("--draft-conf") + 1], "0.5", "the person's own value stays")
        c = self.cfg()
        drafting.apply(c, data_dir=self.data, backend="cuda", archs=["sm_120"], vram_mib=8000)
        self.assertEqual(c["args"][c["args"].index("--draft-conf") + 1], "0.8", "CUDA keeps its slot steps")

    def test_a_mac_gets_its_own_confidence_not_the_cuda_slot_steps(self):
        """Metal maps every expert: the --expert-cache 64 the Mac setup writes is a slot count of 83 to the CUDA steps (0.8), which was not measured on a Mac; M5 Max measured 0.5."""
        self.layer()
        self.pack()
        for vram in (None, 8000, 96000):
            c = self.cfg()
            d = drafting.apply(c, data_dir=self.data, backend="metal", archs=[], vram_mib=vram)
            a = c["args"]
            self.assertEqual((d.drafter, a[a.index("--draft-conf") + 1], a[a.index("--spec") + 1], "--mtp" in a, a[a.index("--draft-fallback") + 1]), ("dflash2", "0.5", "6", False, "none"), vram)
        c = self.cfg(user_args=["--draft-conf", "0.4"], args=[*self.args, "--draft-conf", "0.4"])
        drafting.apply(c, data_dir=self.data, backend="metal", archs=[])
        self.assertEqual(c["args"][c["args"].index("--draft-conf") + 1], "0.4", "the person's own value stays")

    def test_off_is_the_draft_layers_own_switch(self):
        c = self.cfg(drafting="off")
        d = self.apply(c)
        self.assertEqual(d.drafter, "off")
        self.assertIs(c["mtp"], False)
        for flag in ("--mtp", "--spec-min-p", "--batch-mtp", "--draft"):
            self.assertNotIn(flag, c["args"])
        self.assertEqual(c["args"][c["args"].index("--spec") + 1], "4", "prompt lookup keeps its windows")

    def test_the_server_applies_it_at_every_start(self):
        from serve.server import engine_args
        self.pack()
        self.layer()
        with mock.patch("serve.server.drafting_vram_mib", return_value=None), mock.patch.object(drafting, "data_dir_of", return_value=self.data):
            for kw in ({"drafting": "dflash2"}, {}):          # chosen, and Auto
                args = engine_args(self.cfg(**kw))
                self.assertNotIn("--mtp", args)
                self.assertEqual(args[args.index("--draft") + 1], "dflash2")
                self.assertEqual(args[args.index("--draft-fallback") + 1], "none")
            args = engine_args(self.cfg(drafting="mtp"))
            self.assertEqual(args, self.args)
            self.assertEqual(engine_args(self.cfg(parallel=4))[:len(self.args)], self.args, "Auto with several requests: the MTP alone")
            self.assertEqual(drafting.take_note(), drafting.NONE_IN_SLOTS, "the engine that starts next is told why")
            self.assertEqual(engine_args(self.cfg(drafting="mtp", parallel=4))[:len(self.args)], self.args, "the MTP chosen: unchanged")
            self.assertEqual(drafting.take_note(), "")
            # DFlash2 chosen by name with a number of requests the person chose: DFlash2 with that many, no note
            named = engine_args(self.cfg(drafting="dflash2", parallel=4))
            self.assertNotIn("--mtp", named)
            self.assertEqual((named[named.index("--draft") + 1], named[named.index("--draft-fallback") + 1], named[named.index("--batch") + 1]), ("dflash2", "none", "4"))
            self.assertEqual(drafting.take_note(), "")
            one = engine_args(self.cfg(drafting="dflash2", parallel=1))
            self.assertNotIn("--batch", one)
            self.assertEqual(one[one.index("--draft") + 1], "dflash2")
        # a 16 GB card holds about 3,700 slots with the drafter on: the server passes its memory and the confidence follows (drafting.CONF_STEPS)
        with mock.patch("serve.server.drafting_vram_mib", return_value=16384.0), mock.patch.object(drafting, "data_dir_of", return_value=self.data):
            args = engine_args(self.cfg(drafting="dflash2"))
            self.assertIn(args[args.index("--draft-conf") + 1], ("0.7", "0.35"))
            args = engine_args(self.cfg(drafting="dflash2", args=[*self.args, "--expert-cache", "1000"]))
            self.assertEqual(args[args.index("--draft-conf") + 1], "0.8", "the cache the person asked for decides, not the card")
        with mock.patch.object(drafting, "ENGINE_SERVES_DFLASH2", False):
            self.assertEqual(engine_args(self.cfg()), self.args, "the engine cannot serve it: nothing changes")
            off = engine_args(self.cfg(drafting="off"))
            self.assertNotIn("--mtp", off)
            drafting.take_note()
            engine_args(self.cfg(drafting="dflash2"))
            self.assertEqual(drafting.take_note(), drafting.NOT_SERVED)

    def test_the_intel_launcher_script_reaches_the_check_as_its_binary(self):
        """#5138 Bug 1: on an Intel install `exe` is sycl/serve/strata-sycl.sh. Reading that shell script for the
        engine's option strings finds nothing (FORM_NONE) and Auto lost DFlash2 on every such start. The drafting
        check must follow the launcher to the binary it runs (serve.server.engine_binary), as effort_position does."""
        from serve.server import engine_args
        self.pack()
        self.layer()
        script = self.data / "sycl" / "serve" / "strata-sycl.sh"
        script.parent.mkdir(parents=True)
        script.write_text("#!/bin/sh\nexec \"$STRATA_SYCL_BIN\" \"$@\"\n", encoding="utf-8")
        binary = self.data / "build-sycl" / "strata"
        binary.parent.mkdir(parents=True)
        binary.write_bytes(b"MZ\x00\x00 strata generate: --draft dflash2 drafts without the MTP layer and needs "
                           b"--draft-model PACK, --spec 2 or more and --mtp-draft-vocab FILE (or --mtp DIR) "
                           b"\x00--draft-fallback\x00")
        self.assertEqual(drafting.engine_form(script), drafting.FORM_NONE, "the script itself carries no option strings")
        c = self.cfg(env={"STRATA_SYCL_BIN": "build-sycl/strata"})
        with mock.patch("serve.server.drafting_vram_mib", return_value=None), mock.patch.object(drafting, "data_dir_of", return_value=self.data):
            args = engine_args(c, exe=str(script))
        self.assertIn("--draft", args, "Auto on an Intel launcher still gets a drafter decision")
        self.assertEqual(args[args.index("--draft") + 1], "dflash2", "the check read the binary, not the launcher script")

    def test_data_folder_of_an_argument_list(self):
        self.assertEqual(drafting.data_dir_of(["--pack", "/d/packs/x"]), Path("/d"))
        self.assertEqual(drafting.data_dir_of(["--mtp", "/d/mtp/rt"]), Path("/d"))
        self.assertIsNone(drafting.data_dir_of(["--mtp", "4"]))
        self.assertIsNone(drafting.data_dir_of([]))

    def test_the_engine_takes_the_vocabulary_file_alone(self):
        """src/program/generate.cpp: --draft dflash2 needs --mtp-draft-vocab FILE or --mtp DIR, and loads no MTP layer either way."""
        src = (ROOT / "src" / "program" / "generate.cpp").read_text(encoding="utf-8")
        self.assertIn("(o.mtp.empty() && o.mtp_draft_vocab.empty())", src)
        self.assertIn('o.draft_fallback != "mtp"', src)


class UnslothAndTheOtherFlashNextFiles(unittest.TestCase):
    """DF2-METAL found that Auto never picked DFlash2 for Unsloth's quantizations: they are the same Qwen3.8-Flash-Next weights and the DFlash2 numbers (Strix, R9700,
    M5 Max) were measured on UD-Q4_K_XL. Swift (UkisAI's fine-tune) and Coder (ISTA-DASLab's coding version) are fine-tunes with no acceptance numbers: the MTP.
    The names are the ones setup.py and metal/setup_mac.py write: f"{family name}-{model.lower()}"."""

    UNSLOTH = ("qwen3.8-flash-next-unsloth-ud-q4_k_xl", "qwen3.8-flash-next-unsloth-ud-iq4_xs", "qwen3.8-flash-next-unsloth-ud-q5_k_xl",
               "qwen3.8-flash-next-unsloth-ud-q6_k_xl", "qwen3.8-flash-next-unsloth-q8_0")
    FLASH_NEXT = ("qwen3.8-flash-next", "qwen3.8-flash-next-q2_0", "qwen3.8-flash-next-iq2_xs", "qwen3.8-flash-next-iq3_xxs", "qwen3.8-flash-next-iq3_s",
                  "qwen3.8-flash-next-gyro-gyro-s", "qwen3.8-flash-next-gyro-gyro-m")
    OTHERS = ("qwen3.8-flash-next-coder-iq1_m", "qwen3.8-flash-next-nvfp4-nvfp4", "qwen3.8-flash-next-nvfp4-nvfp4-nvidia", "qwen3.8-flash-next-nvfp4-nvfp4-qad",
              "swift-1.5-iq2_xs", "swift-1.5-q4_0", "qwen3.6-35b-a3b-ud-q4_k_xl", "deepseek-v4.1-flash", "qwen3.8-flash-next-unsloth", "qwen3.8-flash-next-unslothx-q8_0")
    HARDWARE = (("cuda", ["sm_120"]), ("hip", ["gfx1151"]), ("metal", None), ("sycl", ["8086"]))

    def test_the_names_the_setups_write(self):
        for name in self.UNSLOTH + self.FLASH_NEXT:
            self.assertTrue(drafting.supported_model(name), name)
            self.assertTrue(drafting.supported_model(name.upper()), name.upper())
        for name in self.OTHERS:
            self.assertFalse(drafting.supported_model(name), name)
        self.assertTrue(drafting.supported_model(None, "unsloth"))
        for family in ("swift", "coder", "nvfp4", "qwen36", "deepseek41"):
            self.assertFalse(drafting.supported_model(None, family), family)
        self.assertEqual(drafting.SUPPORTED_FAMILIES, ("qwen", "gyro", "unsloth"))

    def test_the_names_are_the_ones_setup_writes(self):
        """Read from the code, not from memory: the family names and the way setup.py and metal/setup_mac.py make the run config's model_name."""
        setup = (ROOT / "setup.py").read_text(encoding="utf-8") + (ROOT / "strata_control" / "setup_config.py").read_text(encoding="utf-8")    # (the run config's construction moved to the setup seam)
        mac = (ROOT / "metal" / "setup_mac.py").read_text(encoding="utf-8")
        self.assertIn('"name": "qwen3.8-flash-next-unsloth"', setup)
        self.assertIn('"name": "qwen3.8-flash-next-coder"', setup)
        self.assertIn('"name": "qwen3.8-flash-next-nvfp4"', setup)
        self.assertGreaterEqual(setup.count("\"model_name\": f\"{fam['name']}-{model.lower()}\""), 3)
        self.assertIn("f\"{fam['name']}-{model.lower()}\"", mac)
        for model in ("UD-IQ4_XS", "UD-Q4_K_XL", "UD-Q5_K_XL", "UD-Q6_K_XL", "Q8_0"):
            self.assertIn(f'"{model}": {{"about"', setup)
            self.assertTrue(drafting.supported_model(f"qwen3.8-flash-next-unsloth-{model.lower()}"), model)

    def run_config(self, name, tmp, mode=None):
        data = Path(tmp)
        mtp = data / "mtp" / "rt"
        mtp.mkdir(parents=True)
        (mtp / "experts.bin").write_bytes(b"e")
        (mtp / "draft_vocab.bin").write_bytes(b"v")
        pack = drafting.pack_path(data)
        pack.parent.mkdir(parents=True)
        pack.write_bytes(b"x" * (2 << 20))
        args = ["--pack", str(data / "packs" / "x"), "--native", "/d/m.gguf", "--mtp", str(mtp), "--spec", "4", "--spec-min-p", "0.5"]
        cfg = {"model_name": name, "args": args, **({} if mode is None else {"drafting": mode})}
        return cfg, data, mtp

    def test_unsloth_resolves_to_dflash2_alone_with_the_vocabulary_on_every_backend(self):
        for backend, archs in self.HARDWARE:
            for name in self.UNSLOTH:
                with tempfile.TemporaryDirectory() as tmp:
                    cfg, data, mtp = self.run_config(name, tmp)
                    decision = drafting.apply(cfg, data_dir=data, backend=backend, archs=archs)
                    args = cfg["args"]
                    self.assertEqual((decision.drafter, decision.note), ("dflash2", ""), (backend, name))
                    self.assertEqual(args[args.index("--draft") + 1], "dflash2")
                    self.assertEqual(args[args.index("--draft-fallback") + 1], "none")
                    self.assertEqual(args[args.index("--mtp-draft-vocab") + 1], str(mtp / "draft_vocab.bin"), "the vocabulary the engine needs")
                    for flag in ("--mtp", "--spec-min-p"):
                        self.assertNotIn(flag, args, (backend, name, flag))
                    self.assertEqual(args[args.index("--spec") + 1], "8" if (backend, archs) == ("hip", ["gfx1151"]) else "6", "Strix Halo 8 (RESULTS-CONF.md 11-08), every other card 6")

    def test_swift_coder_and_nvfp4_stay_on_the_mtp(self):
        for backend, archs in self.HARDWARE:
            for name in ("qwen3.8-flash-next-coder-iq1_m", "qwen3.8-flash-next-nvfp4-nvfp4", "swift-1.5-iq2_xs"):
                for mode in (None, "auto", "dflash2"):
                    if mode == "dflash2" and "-coder-" in name:
                        continue                                    # #5240: the Coder named DFlash2 runs it (test_drafting_coder_named.py)
                    with tempfile.TemporaryDirectory() as tmp:
                        cfg, data, mtp = self.run_config(name, tmp, mode)
                        original = list(cfg["args"])
                        decision = drafting.apply(cfg, data_dir=data, backend=backend, archs=archs)
                        self.assertEqual(decision.drafter, "mtp", (backend, name, mode))
                        self.assertEqual(cfg["args"], original, "its own MTP arguments, untouched")
                        self.assertNotIn("--draft", cfg["args"])
                        if mode == "dflash2":
                            self.assertEqual(decision.note, drafting.NOT_SUPPORTED)

    def test_by_family_the_terminal_and_the_app_agree(self):
        for backend, _ in self.HARDWARE:
            self.assertEqual(drafting.choice_of("auto", family="unsloth", backend=backend), "dflash2", backend)
            for family in ("swift", "coder", "nvfp4"):
                self.assertEqual(drafting.choice_of("auto", family=family, backend=backend), "mtp", (backend, family))
        # the app's own copies of the rule (the host and the page) name the same families and the same names
        host = (ROOT / "apps" / "host" / "server" / "native-settings.mjs").read_text(encoding="utf-8")
        page = (ROOT / "apps" / "host" / "src" / "lib" / "drafting.ts").read_text(encoding="utf-8")
        generated = (ROOT / "apps" / "host" / "server" / "policy-generated.mjs").read_text(encoding="utf-8")        # the host's copy is generated from drafting.py (strata_control/app_policy.py)
        self.assertIn("DRAFTING_FAMILIES=GENERATED_DRAFTING_FAMILIES", host)
        self.assertIn('DRAFTING_FAMILIES = ["qwen", "gyro", "unsloth"]', generated)
        self.assertIn("DRAFTING_FAMILIES = ['qwen', 'gyro', 'unsloth']", page)
        self.assertIn(r"-unsloth-[a-z0-9._-]+", generated)


class TheInstalledEngineDecidesTheFlags(unittest.TestCase):
    """GUI-FB3 item 2. Gyro-S with Auto sent `--mtp-draft-vocab FILE` and no `--mtp` to the engine 0.1.55.2 (f-0291-r5070-df2), which only takes `--mtp DIR` for the
    vocabulary and ended with exit code 2. The app reads what the engine program takes (it is never started for this) and sends only that."""

    OLD = b"MZ\x00\x00 strata generate: --draft dflash2 drafts without the MTP layer (--draft-fallback mtp loads it too) and needs --draft-model PACK, --spec 2 or more and --mtp DIR (for DIR/draft_vocab.bin) \x00--draft-fallback\x00"
    NEW = b"MZ\x00\x00 strata generate: --draft dflash2 drafts without the MTP layer and needs --draft-model PACK, --spec 2 or more and --mtp-draft-vocab FILE (or --mtp DIR, for DIR/draft_vocab.bin) \x00--draft-fallback\x00"
    UNKNOWN = b"MZ\x00\x00 a program that has the flag and a message nobody knows \x00--draft-fallback\x00"
    NONE = b"MZ\x00\x00 an engine from before DFlash2 ran alone: --draft dflash2 --draft-model \x00"

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.data = self.root / "data"
        self.mtp = self.data / "mtp" / "rt"
        self.mtp.mkdir(parents=True)
        (self.mtp / "experts.bin").write_bytes(b"e")
        (self.mtp / "draft_vocab.bin").write_bytes(b"v")
        self.pack = drafting.pack_path(self.data)
        self.pack.parent.mkdir(parents=True)
        self.pack.write_bytes(b"x" * (2 << 20))
        self.args = ["--pack", str(self.data / "packs" / "gyro"), "--native", "/d/m.gguf", "--mtp", str(self.mtp), "--spec", "4", "--spec-min-p", "0.5"]

    def exe(self, name: str, body: bytes) -> Path:
        path = self.root / name
        path.write_bytes(body)
        return path

    def apply(self, engine, **extra):
        cfg = {"model_name": GYRO, "args": list(self.args), **extra}
        decision = drafting.apply(cfg, data_dir=self.data, backend="cuda", archs=["sm_120"], engine=engine)
        return decision, cfg["args"]

    def test_the_program_is_read_and_the_answer_kept_per_file(self):
        self.assertEqual(drafting.engine_form(self.exe("old.exe", self.OLD)), drafting.FORM_DIR)
        self.assertEqual(drafting.engine_form(self.exe("new.exe", self.NEW)), drafting.FORM_FILE)
        self.assertIsNone(drafting.engine_form(self.exe("odd.exe", self.UNKNOWN)), "the flag is there but the message is not known: the usual form")
        self.assertEqual(drafting.engine_form(self.exe("none.exe", self.NONE)), drafting.FORM_NONE, "no --draft-fallback: no DFlash2 on its own")
        self.assertIsNone(drafting.engine_form(None))
        self.assertIsNone(drafting.engine_form(self.root / "missing.exe"))
        self.assertIsNone(drafting.engine_form(self.exe("empty.exe", b"")), "an empty file cannot be read: the usual form")
        # a newer program in the same place is read again (size and time are part of the key)
        path = self.exe("swap.exe", self.OLD)
        self.assertEqual(drafting.engine_form(path), drafting.FORM_DIR)
        path.write_bytes(self.NEW + b"more")
        self.assertEqual(drafting.engine_form(path), drafting.FORM_FILE)

    def test_the_vocabulary_flag_follows_the_engine(self):
        vocab = self.mtp / "draft_vocab.bin"
        usual = drafting.engine_flags("/p", vocab)
        self.assertEqual(usual, drafting.engine_flags("/p", vocab, drafting.FORM_FILE))
        self.assertIn("--mtp-draft-vocab", usual)
        self.assertNotIn("--mtp", usual)
        old = drafting.engine_flags("/p", vocab, drafting.FORM_DIR)
        self.assertEqual(old[old.index("--mtp") + 1], str(self.mtp), "the folder that holds draft_vocab.bin")
        self.assertNotIn("--mtp-draft-vocab", old)
        self.assertEqual(old[:6], ("--draft", "dflash2", "--draft-model", "/p", "--draft-fallback", "none"), "still no MTP layer: the fallback is none")
        self.assertEqual(old[-4:], ("--spec", "6", "--draft-conf", "0.35"))
        self.assertNotIn("--mtp", drafting.engine_flags("/p", None, drafting.FORM_DIR), "no vocabulary, no flag")

    def test_an_engine_that_takes_mtp_dir_gets_dflash2_alone_that_way(self):
        decision, args = self.apply(self.exe("old.exe", self.OLD))
        self.assertEqual((decision.drafter, decision.note), ("dflash2", ""))
        self.assertNotIn("--mtp-draft-vocab", args)
        self.assertEqual(args[args.index("--mtp") + 1], str(self.mtp))
        self.assertEqual(args[args.index("--draft-fallback") + 1], "none", "the MTP layer is still not loaded")
        for flag in ("--spec-min-p", "--batch-mtp", "--draft-mtp-spec"):
            self.assertNotIn(flag, args)
        self.assertEqual(args.count("--mtp"), 1)
        self.assertEqual(args.count("--spec"), 1)
        # the same start for the newer engine and for one nobody can tell: the vocabulary file alone
        for body in (self.NEW, self.UNKNOWN):
            decision, args = self.apply(self.exe("x.exe", body))
            self.assertEqual(decision.drafter, "dflash2")
            self.assertEqual(args[args.index("--mtp-draft-vocab") + 1], str(self.mtp / "draft_vocab.bin"))
            self.assertNotIn("--mtp", args)
        decision, args = self.apply(None)
        self.assertIn("--mtp-draft-vocab", args)

    def test_with_no_mtp_folder_the_old_form_names_the_dflash2_folder_that_holds_the_vocabulary(self):
        own = drafting.vocab_target(self.data)
        own.parent.mkdir(parents=True, exist_ok=True)
        own.write_bytes(b"v")
        args = ["--pack", str(self.data / "packs" / "gyro"), "--native", "/d/m.gguf"]
        cfg = {"model_name": GYRO, "args": args}
        decision = drafting.apply(cfg, data_dir=self.data, backend="cuda", archs=["sm_120"], engine=self.exe("old.exe", self.OLD))
        self.assertEqual(decision.drafter, "dflash2")
        self.assertEqual(cfg["args"][cfg["args"].index("--mtp") + 1], str(own.parent))

    def test_a_vocabulary_that_is_not_called_draft_vocab_cannot_go_by_mtp_dir(self):
        custom = self.root / "my-subset.bin"
        custom.write_bytes(b"v")
        args = [*self.args, "--mtp-draft-vocab", str(custom)]
        cfg = {"model_name": GYRO, "args": args}
        decision = drafting.apply(cfg, data_dir=self.data, backend="cuda", archs=["sm_120"], engine=self.exe("old.exe", self.OLD))
        self.assertEqual((decision.drafter, decision.note), ("mtp", drafting.NOT_HERE), "said in one line; the MTP runs")
        self.assertNotIn("--draft", cfg["args"])
        cfg = {"model_name": GYRO, "args": list(args)}
        decision = drafting.apply(cfg, data_dir=self.data, backend="cuda", archs=["sm_120"], engine=self.exe("new.exe", self.NEW))
        self.assertEqual(decision.drafter, "dflash2")
        self.assertEqual(cfg["args"][cfg["args"].index("--mtp-draft-vocab") + 1], str(custom), "an engine that takes the file keeps the person's own subset")

    def test_an_engine_without_the_fallback_flag_never_gets_a_dflash2_flag(self):
        for mode in ("auto", "dflash2"):
            decision, args = self.apply(self.exe("none.exe", self.NONE), drafting=mode)
            self.assertEqual((decision.drafter, decision.note), ("mtp", drafting.NOT_HERE), mode)
            for flag in ("--draft", "--draft-model", "--draft-fallback", "--draft-conf", "--mtp-draft-vocab"):
                self.assertNotIn(flag, args, (mode, flag))
            self.assertEqual(args[args.index("--mtp") + 1], str(self.mtp), "the MTP, as the config had it")

    def test_the_server_gives_the_engine_program_to_the_drafting_rule(self):
        from serve.server import engine_args
        old, new = self.exe("old.exe", self.OLD), self.exe("new.exe", self.NEW)
        cfg = {"model_name": GYRO, "args": list(self.args), "drafting": "auto"}
        with mock.patch("serve.server.drafting_vram_mib", return_value=None), mock.patch.object(drafting, "data_dir_of", return_value=self.data):
            a_old, a_new, a_none = engine_args(cfg, str(old)), engine_args(cfg, str(new)), engine_args(cfg)
        self.assertEqual(a_old[a_old.index("--mtp") + 1], str(self.mtp))
        self.assertNotIn("--mtp-draft-vocab", a_old)
        self.assertIn("--mtp-draft-vocab", a_new)
        self.assertIn("--mtp-draft-vocab", a_none, "no program given (a test, a tool): the usual form")
        # "mtp": false does not take the vocabulary folder out of a DFlash2 start
        off = {"model_name": GYRO, "args": list(self.args), "drafting": "dflash2", "mtp": False}
        with mock.patch("serve.server.drafting_vram_mib", return_value=None), mock.patch.object(drafting, "data_dir_of", return_value=self.data):
            a = engine_args(off, str(old))
        self.assertEqual(a[a.index("--mtp") + 1], str(self.mtp))

    def test_the_retry_arguments_exist_only_for_an_auto_start_that_picked_dflash2(self):
        from serve.server import mtp_retry_args
        old = str(self.exe("old.exe", self.OLD))
        base = {"model_name": GYRO, "args": list(self.args)}
        with mock.patch("serve.server.drafting_vram_mib", return_value=None), mock.patch.object(drafting, "data_dir_of", return_value=self.data):
            alt = mtp_retry_args({**base, "drafting": "auto"}, old, ["--effort", "x"])
            self.assertIsNotNone(alt)
            self.assertEqual(alt[:len(self.args)], self.args, "the MTP, as the config had it")
            self.assertNotIn("--draft", alt)
            self.assertEqual(alt[-2:], ["--effort", "x"], "what the start adds goes along")
            self.assertIsNotNone(mtp_retry_args(dict(base), old), "no choice saved means Auto")
            self.assertIsNone(mtp_retry_args({**base, "drafting": "dflash2"}, old), "named by the person: it fails with its reason")
            self.assertIsNone(mtp_retry_args({**base, "drafting": "mtp"}, old), "nothing to fall back from")
            self.assertIsNone(mtp_retry_args({**base, "drafting": "off"}, old))
            self.assertIsNone(mtp_retry_args({"model_name": "qwen3.6-35b-a3b", "args": list(self.args), "drafting": "auto"}, old), "another model has no DFlash2")
            # no MTP layer on this computer: the other start would be DFlash2 again, so there is nothing to retry with
            for name in ("experts.bin", "draft_vocab.bin"):
                (self.mtp / name).unlink()
            self.assertIsNone(mtp_retry_args({**base, "drafting": "auto"}, old))
        # the lines and the note of the real start are not touched by working the retry out
        drafting.take_note()
        with mock.patch("serve.server.drafting_vram_mib", return_value=None), mock.patch.object(drafting, "data_dir_of", return_value=self.data):
            mtp_retry_args({**base, "drafting": "auto"}, old)
        self.assertEqual(drafting.take_note(), "")

    def test_the_no_drafter_retry_exists_for_an_auto_start_with_a_drafter_and_never_swaps_in_the_mtp(self):
        """A start that ended for lack of graphics memory is repeated with no drafter (multi-card walk): DFlash2 -> MTP would take MORE memory from a card that is short."""
        from serve.server import nodraft_retry_args
        old = str(self.exe("old.exe", self.OLD))
        base = {"model_name": GYRO, "args": list(self.args)}
        with mock.patch("serve.server.drafting_vram_mib", return_value=None), mock.patch.object(drafting, "data_dir_of", return_value=self.data):
            alt = nodraft_retry_args({**base, "drafting": "auto"}, old, ["--effort", "x"])
            self.assertIsNotNone(alt)
            for flag in ("--draft", "--draft-model", "--draft-fallback", "--mtp", "--mtp-draft-vocab"):
                self.assertNotIn(flag, alt, flag)
            self.assertEqual(alt[-2:], ["--effort", "x"], "what the start adds goes along")
            self.assertIn("--native", alt, "everything else is the same start")
            self.assertIsNotNone(nodraft_retry_args(dict(base), old), "no choice saved means Auto")
            self.assertIsNone(nodraft_retry_args({**base, "drafting": "dflash2"}, old), "named by the person: it fails with its reason")
            self.assertIsNone(nodraft_retry_args({**base, "drafting": "mtp"}, old), "named by the person: it fails with its reason")
            self.assertIsNone(nodraft_retry_args({**base, "drafting": "off"}, old), "no drafter to drop")
        drafting.take_note()
        with mock.patch("serve.server.drafting_vram_mib", return_value=None), mock.patch.object(drafting, "data_dir_of", return_value=self.data):
            nodraft_retry_args({**base, "drafting": "auto"}, old)
        self.assertEqual(drafting.take_note(), "", "working the retry out leaves the note of the real start alone")

    def test_the_engine_sources_carry_the_words_the_program_is_read_for(self):
        src = (ROOT / "src" / "program" / "generate.cpp").read_text(encoding="utf-8")
        self.assertIn(drafting._MARK_FILE.decode(), src, "rc1's start message is what marks an engine that takes the vocabulary file alone")
        self.assertIn(drafting._MARK_FALLBACK.decode(), src)


class WhatTheEngineSaid(unittest.TestCase):
    ARGS = ["--mtp", "/m", "--draft", "dflash2", "--draft-model", "/p"]

    def test_no_room_in_graphics_memory(self):
        log = ("strata generate: --draft dflash2: could not allocate the drafter's buffers. DFlash2 needs about 1.2 GiB of VRAM (the pack 0.76 GiB, "
               "its buffers and K/V ring 0.25 GiB, a draft head 0.2 GiB); start with --vram-reserve-mib 1900 to keep it. The MTP drafts instead.\n")
        info = drafting.engine_info(self.ARGS, log)
        self.assertEqual(info, {"drafting": "mtp", "drafting_note": "Drafting: MTP (DFlash2 didn't fit)"})

    def test_a_mode_this_build_does_not_run(self):
        log = "strata generate: --draft dflash2 is off: it is for one card and the CLI run (not --serve, --batch or a layer split) in this build; the MTP drafts instead\n"
        self.assertEqual(drafting.engine_info(self.ARGS, log)["drafting_note"], "Drafting: MTP (DFlash2 isn't available in this mode)")

    def test_it_ran(self):
        log = "strata generate: DFlash2 drafter on (/p, graphs captured in 3 ms), the MTP layer stays as the fallback\n"
        self.assertEqual(drafting.engine_info(self.ARGS, log), {"drafting": "dflash2", "drafting_note": ""})
        self.assertEqual(drafting.fallback_note(log), "")

    def test_choices_that_never_reached_the_engine_keep_their_own_words(self):
        self.assertEqual(drafting.engine_info(["--mtp", "/m"], "", drafting.NOT_SUPPORTED)["drafting_note"],
                         "Drafting: MTP (DFlash2 is for Qwen3.8-Flash-Next only)")
        self.assertEqual(drafting.engine_info(["--mtp", "/m"]), {"drafting": "mtp", "drafting_note": ""})
        self.assertEqual(drafting.engine_info(["--spec", "4"]), {"drafting": "off", "drafting_note": ""})

    def test_summary_line(self):
        self.assertEqual(drafting.summary_line(drafting.Decision("auto", "mtp")), "Drafting: MTP")
        self.assertEqual(drafting.summary_line(drafting.Decision("dflash2", "dflash2")), "Drafting: DFlash2")
        self.assertEqual(drafting.summary_line(drafting.Decision("off", "off")), "Drafting: Off")
        self.assertEqual(drafting.summary_line(drafting.Decision("dflash2", "mtp", drafting.NOT_DOWNLOADED)),
                         "Drafting: MTP (the DFlash2 drafter isn't downloaded yet)")


class WordsMatchTheApp(unittest.TestCase):
    def test_every_line_the_engine_start_can_say_is_in_the_apps_catalog_in_all_nine_languages(self):
        folder = ROOT / "apps" / "host" / "src" / "locales"
        catalogs = {l: {**json.loads((folder / f"{l}.json").read_text(encoding="utf-8")), **json.loads((folder / f"df2-{l}.json").read_text(encoding="utf-8"))}
                    for l in ("en", "de", "es", "fr", "ja", "ko", "pt", "ru", "zh")}
        ids = {v: k for k, v in catalogs["en"].items()}
        self.assertEqual(len(drafting.REASONS), 10)
        lines = [drafting.mtp_line(r) for r in (drafting.DIDNT_FIT, drafting.NOT_HERE, drafting.NOT_SUPPORTED, drafting.NOT_DOWNLOADED, drafting.NOT_THIS_CARD, drafting.DIDNT_START)]
        lines += [drafting.line("dflash2", drafting.NO_MTP), drafting.line("off", drafting.NONE_HERE), drafting.line("off", drafting.SHORT_OF_VRAM), drafting.line("mtp", drafting.NONE_IN_SLOTS)]
        for line in lines:
            self.assertIn(line, ids, line)
            for lang, catalog in catalogs.items():
                self.assertTrue(catalog[ids[line]].strip(), f"{line} in {lang}")
        # and what the engine's log makes of them
        self.assertEqual(drafting.fallback_note("--draft dflash2: out of memory"), drafting.mtp_line(drafting.DIDNT_FIT))
        self.assertEqual(drafting.fallback_note("--draft dflash2 is off: it is for one card and the CLI run"), drafting.mtp_line(drafting.NOT_HERE))


class TerminalFlag(unittest.TestCase):
    BASE = {"exe": "component:strata-engine", "args": ["--pack", "/d/packs/x", "--mtp", "/d/mtp/rt"], "port": 8080, "model_name": GYRO}

    def test_the_flag_is_listed_planned_copyable_and_saved_in_the_launch_config(self):
        code, out, _ = run_cli(["launch", "--help"])
        self.assertIn("--drafting", out)
        self.assertIn("auto|mtp|dflash2|off", out)
        cfg = derive_config(self.BASE, S.LaunchSettings(drafting="dflash2"), bool_flags=frozenset())
        self.assertEqual(cfg["drafting"], "dflash2")
        self.assertNotIn("drafting", derive_config(self.BASE, S.LaunchSettings(), bool_flags=frozenset()), "Auto writes nothing: nobody chose")
        self.assertIn("--drafting dflash2", S.flags_text(S.LaunchSettings(drafting="dflash2")))
        self.assertNotIn("--drafting", S.flags_text(S.LaunchSettings()))
        code, out, err = run_cli(["launch", "--plan", "--drafting", "fastest"])
        self.assertNotEqual(code, 0)

    def test_a_saved_preset_keeps_the_choice(self):
        s = S.LaunchSettings.from_dict(S.LaunchSettings(drafting="mtp").to_dict())
        self.assertEqual(s.drafting, "mtp")
        self.assertIsNone(S.LaunchSettings.from_dict({}).drafting)


class _Handler(http.server.BaseHTTPRequestHandler):
    files: dict = {}
    hits: list = []

    def do_GET(self):
        body = self.files.get(self.path.rsplit("/", 1)[-1])
        self.hits.append(self.path)
        if body is None:
            self.send_error(404)
            return
        start = 0
        rng = self.headers.get("Range")
        if rng and rng.startswith("bytes="):
            start = int(rng[6:].split("-")[0])
        self.send_response(206 if start else 200)
        if start:
            self.send_header("Content-Range", f"bytes {start}-{len(body) - 1}/{len(body)}")
        self.send_header("Content-Length", str(len(body) - start))
        self.end_headers()
        self.wfile.write(body[start:])

    def log_message(self, *_):
        pass


class FetchAndPack(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(ROOT / "tools"))
        spec = importlib.util.spec_from_file_location("t_pack", ROOT / "tools" / "test_dflash2_pack.py")
        cls.synth = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.synth)

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name)
        self.synth.write_safetensors(self.dir / "m.safetensors", self.synth.tensors())
        (self.dir / "c.json").write_text(json.dumps(self.synth.CFG))
        import hashlib
        self.blobs = {"config.json": (self.dir / "c.json").read_bytes(), "model.safetensors": (self.dir / "m.safetensors").read_bytes()}
        self.files = tuple((n, len(b), hashlib.sha256(b).hexdigest()) for n, b in self.blobs.items())
        _Handler.files, _Handler.hits = dict(self.blobs), []
        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.addCleanup(self.server.shutdown)
        self.base = f"http://127.0.0.1:{self.server.server_address[1]}/"
        self.data = self.dir / "data"
        self.steps: list = []
        # the packed test file is a few hundred kB, not the real 0.76 GB: "ready" asks for more than 1 MiB by default
        patcher = mock.patch.object(drafting, "pack_ready", lambda d: bool(d) and drafting.pack_path(d).is_file())
        patcher.start()
        self.addCleanup(patcher.stop)

    def fetch(self, **kw):
        with mock.patch.dict("os.environ", {"STRATA_UPDATE_ALLOW_HTTP": "1"}):
            return drafting_fetch.ensure(self.data, progress=lambda *a: self.steps.append(a), base_url=self.base, files=self.files, **kw)

    def test_the_pinned_files_are_the_repositorys_own(self):
        self.assertEqual(drafting.REVISION, "a22a1744e85c17364dcc54a6a420af7135d4c7f6")
        self.assertTrue(drafting.BASE_URL.endswith(f"/{drafting.REVISION}/"), "a revision, never a branch name")
        names = {n: (size, sha) for n, size, sha in drafting.FILES}
        self.assertEqual(names["model.safetensors"], (1206140008, "d6723f2a359eaf33ec0c95f22ae8edda896774020c552296a85e1913b4d3186f"))
        self.assertEqual(len(names["config.json"][1]), 64)
        self.assertEqual(drafting.LICENSE, "Apache-2.0")

    def test_it_downloads_checks_packs_and_removes_the_source(self):
        out = self.fetch()
        self.assertEqual(out, drafting.pack_path(self.data))
        self.assertEqual(out.read_bytes()[:4], b"DF2P")
        self.assertFalse((out.parent / "src").exists(), "the 1.2 GB source file does not stay on disk")
        words = [s[0] for s in self.steps]
        self.assertIn("Downloading the DFlash2 drafter…", words)
        self.assertIn("Packing the DFlash2 drafter for the engine…", words)
        self.assertEqual(words[-1], "The DFlash2 drafter is ready.")
        self.assertEqual(max(s[1] for s in self.steps if s[0].startswith("Downloading")), sum(f[1] for f in self.files))

    def test_a_drafter_that_is_here_is_not_fetched_again(self):
        self.fetch()
        _Handler.hits.clear()
        self.steps.clear()
        self.fetch()
        self.assertEqual(_Handler.hits, [])
        self.assertEqual(self.steps[0][0], "The DFlash2 drafter is ready.")

    def test_a_fresh_install_that_selected_only_dflash2_has_everything_the_engine_needs(self):
        """No MTP layer was ever downloaded (no mtp folder at all): the pack comes with the draft vocabulary, and the engine arguments carry
        --mtp-draft-vocab for it and no --mtp."""
        self.assertFalse((self.data / "mtp").exists())
        self.assertFalse(drafting.vocab_target(self.data).exists())
        self.fetch()
        vocab = drafting.vocab_target(self.data)
        self.assertEqual(vocab, self.data / drafting.FOLDER / "draft_vocab.bin")
        self.assertTrue(vocab.is_file() and vocab.stat().st_size > 0)
        self.assertEqual(vocab.read_bytes(), drafting.shipped_vocab().read_bytes(), "the subset Strata ships")
        args = ["--pack", str(self.data / "packs" / "gyro"), "--native", "/m.gguf", "--mtp", str(self.data / "mtp" / "rt"), "--spec", "4", "--spec-min-p", "0.5"]
        # the Strata folder's own copy is not needed any more: the install keeps its own
        with mock.patch.object(drafting, "shipped_vocab", return_value=self.dir / "gone.bin"):
            cfg = {"model_name": GYRO, "args": list(args)}
            d = drafting.apply(cfg, data_dir=self.data, backend="cuda", archs=["sm_120"])
        self.assertEqual((d.drafter, d.note), ("dflash2", ""))
        a = cfg["args"]
        self.assertNotIn("--mtp", a)
        self.assertEqual(a[a.index("--mtp-draft-vocab") + 1], str(vocab))
        self.assertEqual(a[a.index("--draft-model") + 1], str(drafting.pack_path(self.data)))
        self.assertEqual(a[a.index("--draft-fallback") + 1], "none")
        # and through the server, as a start builds it
        from serve.server import engine_args
        with mock.patch("serve.server.drafting_vram_mib", return_value=None), mock.patch.object(drafting, "data_dir_of", return_value=self.data):
            served = engine_args({"model_name": GYRO, "args": list(args)})
        self.assertNotIn("--mtp", served)
        self.assertTrue(Path(served[served.index("--mtp-draft-vocab") + 1]).is_file())

    def test_the_vocabulary_is_provided_once_and_never_overwritten(self):
        out = drafting.provide_vocab(self.data)
        self.assertEqual(out, drafting.vocab_target(self.data))
        out.write_bytes(b"mine")
        self.assertEqual(drafting.provide_vocab(self.data).read_bytes(), b"mine")
        self.assertIsNone(drafting.provide_vocab(None))
        with mock.patch.object(drafting, "shipped_vocab", return_value=self.dir / "gone.bin"):
            self.assertIsNone(drafting.provide_vocab(self.dir / "other"))

    def test_without_any_vocabulary_dflash2_does_not_start_and_says_so(self):
        """The engine refuses a missing vocabulary file; the app never gets there: no file anywhere means the MTP (or none), in one line."""
        self.fetch()
        args = ["--pack", str(self.data / "packs" / "gyro"), "--mtp", str(self.data / "mtp" / "rt")]
        drafting.vocab_target(self.data).unlink()
        with mock.patch.object(drafting, "shipped_vocab", return_value=self.dir / "gone.bin"):
            cfg = {"model_name": GYRO, "args": list(args)}
            d = drafting.apply(cfg, data_dir=self.data, backend="cuda", archs=["sm_120"])
        self.assertEqual((d.drafter, d.note), ("off", drafting.NONE_HERE))
        self.assertNotIn("--draft", cfg["args"])

    def test_a_file_that_does_not_match_its_hash_is_deleted_and_the_words_are_plain(self):
        bad = tuple((n, size, "0" * 64) for n, size, _h in self.files)
        with mock.patch.dict("os.environ", {"STRATA_UPDATE_ALLOW_HTTP": "1"}):
            with self.assertRaises(drafting_fetch.DraftingError) as ctx:
                drafting_fetch.ensure(self.data, base_url=self.base, files=bad)
        self.assertIn("did not pass the safety check", str(ctx.exception))
        self.assertFalse(drafting.pack_path(self.data).exists())
        self.assertEqual(list((drafting.pack_path(self.data).parent / "src").glob("*.part")), [])

    def test_an_interrupted_download_is_continued(self):
        name, size, _sha = self.files[1]
        src = drafting.pack_path(self.data).parent / "src"
        src.mkdir(parents=True)
        (src / (name + ".part")).write_bytes(self.blobs[name][: size // 2])
        self.fetch()
        ranges = [h for h in _Handler.hits if h.endswith(name)]
        self.assertEqual(len(ranges), 1)
        self.assertTrue(drafting.pack_path(self.data).is_file())

    def test_a_start_never_downloads(self):
        with acquisition.forbidden():
            with self.assertRaises(acquisition.AcquisitionRefused):
                drafting_fetch.ensure(self.data, base_url=self.base, files=self.files)
        self.assertFalse(drafting.pack_path(self.data).exists())

    def test_the_plan_names_the_size_and_where_it_comes_from(self):
        with mock.patch.object(drafting, "pack_ready", return_value=False):
            plan = drafting_fetch.plan(self.data)
        self.assertEqual(plan["gb"], 1.2)
        self.assertIn("1.2 GB to download", plan["text"])
        self.assertIn("0.52 GB", plan["text"])
        self.assertIn("incoai/Qwen3.8-Flash-Next-DFlash2", plan["text"])
        self.assertIn("Apache-2.0", plan["text"])
        self.assertIn("It needs about 0.9 GB of graphics memory", plan["text"], "the figure of DFlash2 alone")
        with mock.patch.object(drafting, "pack_ready", return_value=True):
            self.assertEqual(drafting_fetch.plan(self.data)["gb"], 0.0)

    def test_the_command_line_says_the_same(self):
        code, out, err = run_cli(["drafting", "plan", "--data-dir", str(self.data), "--output", "json"])
        self.assertEqual(code, 0, err)
        payload = json.loads(out)
        self.assertEqual((payload["ok"], payload["revision"]), (True, drafting.REVISION))
        self.assertEqual(payload["gb"], 1.2)


class TheAppAsksControl(unittest.TestCase):
    """The app calls `strata drafting plan|fetch --config <run config> --mode <choice>` before a load; Control decides, the app never reads the table."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.data = Path(self.tmp.name) / "data"
        self.cfg = Path(self.tmp.name) / "strata-gyro.json"
        self.cfg.write_text(json.dumps({"model_name": GYRO, "args": ["--pack", str(self.data / "packs" / "gyro"), "--mtp", str(self.data / "mtp" / "rt")]}))

    def plan(self, *extra):
        code, out, err = run_cli(["drafting", "plan", "--config", str(self.cfg), "--output", "json", *extra])
        self.assertEqual(code, 0, err)
        return json.loads(out)

    def test_nothing_is_needed_for_mtp_or_off_and_the_pack_for_auto_and_dflash2(self):
        for mode in ("mtp", "off"):
            plan = self.plan("--mode", mode)
            self.assertFalse(plan["needed"], mode)
            self.assertEqual(plan["gb"], 0.0)
        for mode in ("auto", "dflash2"):
            plan = self.plan("--mode", mode)
            self.assertTrue(plan["needed"], mode)
            self.assertEqual((plan["gb"], plan["choice"]), (1.2, "dflash2"))
        self.assertEqual(self.plan("--mode", "off")["drafter"], "off")

    def test_choosing_the_mtp_without_its_layer_says_the_layer_is_needed(self):
        plan = self.plan("--mode", "mtp")
        self.assertTrue(plan["mtp_needed"])
        self.assertEqual((plan["mtp_gb"], plan["choice"]), (5.2, "mtp"))
        self.assertFalse(self.plan("--mode", "auto")["mtp_needed"])

    def test_dflash2_is_needed_once_the_engine_can_serve_it_and_not_before(self):
        (self.data / "mtp" / "rt").mkdir(parents=True)
        (self.data / "mtp" / "rt" / "experts.bin").write_bytes(b"e")           # the MTP layer is on this computer
        with mock.patch.object(drafting, "ENGINE_SERVES_DFLASH2", False):
            before = self.plan("--mode", "dflash2")
        self.assertFalse(before["needed"], "while the engine cannot serve it nothing is downloaded")
        self.assertEqual(before["line"], "Drafting: MTP (DFlash2 isn't available in this mode)")
        with serving_dflash2():
            plan = self.plan("--mode", "dflash2")
            self.assertTrue(plan["needed"])
            self.assertEqual(plan["gb"], 1.2)
            self.assertEqual(Path(plan["pack"]), drafting.pack_path(self.data), "the data folder comes from the run config's own arguments")
            self.cfg.write_text(json.dumps({"model_name": "qwen3.8-flash-next-swift-iq2_xs", "drafting": "dflash2", "args": ["--pack", str(self.data / "packs" / "c")]}))
            other = self.plan()
            self.assertFalse(other["needed"])
            self.assertEqual(other["note"], drafting.NOT_SUPPORTED)

    def test_a_model_that_is_not_installed_yet_is_asked_about_by_its_family(self):
        with serving_dflash2():
            for family, needed in (("gyro", True), ("qwen", True), ("coder", True), ("swift", False)):       # (#5240: the Coder named)
                code, out, err = run_cli(["drafting", "plan", "--family", family, "--mode", "dflash2", "--data-dir", str(self.data), "--output", "json"])
                self.assertEqual(code, 0, err)
                self.assertEqual(json.loads(out)["needed"], needed, family)

    def test_fetch_if_needed_does_nothing_when_nothing_is_needed(self):   # (the MTP is chosen: no drafter file is needed)
        code, out, err = run_cli(["drafting", "fetch", "--config", str(self.cfg), "--mode", "mtp", "--if-needed", "--output", "jsonl"])
        self.assertEqual(code, 0, err)
        last = json.loads(out.strip().splitlines()[-1])
        self.assertEqual((last["ok"], last["skipped"], last["pack"]), (True, True, None))
        self.assertFalse(drafting.pack_path(self.data).exists())


class TheDownloadPlanListsIt(unittest.TestCase):
    """`strata launch --plan --drafting dflash2` adds the drafter to the download when it would be used."""

    def setUp(self):
        from tests.terminal.launch_support import not_intel
        not_intel(self)

    def test_the_plan_line_and_the_part(self):
        from tests.terminal.launch_support import HW_12, Home, catalog
        from strata_cli.launch.runner import Runner
        with Home() as home:
            runner = Runner(S.LaunchSettings(family="qwen", model="Q2_0", context=32768, vision="off", drafting="dflash2"), catalog=catalog(HW_12))
            with mock.patch.object(drafting, "ENGINE_SERVES_DFLASH2", False):
                line = runner.drafting_plan_line()
                self.assertEqual(line, "Drafting: MTP (DFlash2 isn't available in this mode)")
                self.assertNotIn("dflash2", [p["kind"] for p in runner.download_plan()["parts"]], "nothing to fetch while the engine cannot serve it")
            with serving_dflash2():
                kinds = {p["kind"]: p for p in runner.download_plan()["parts"]}
                self.assertEqual(kinds["dflash2"]["gb"], 1.2)
                self.assertIn("the DFlash2 drafter (1.2 GB)", runner.download_plan()["text"])
                self.assertIn("--drafting mtp", runner.download_plan()["text"])
                auto = Runner(S.LaunchSettings(family="qwen", model="Q2_0", context=32768, vision="off"), catalog=catalog(HW_12))
                self.assertIn("dflash2", [p["kind"] for p in auto.download_plan()["parts"]], "Auto is DFlash2: its drafter is the download")
                mtp = Runner(S.LaunchSettings(family="qwen", model="Q2_0", context=32768, vision="off", drafting="mtp"), catalog=catalog(HW_12))
                kinds = [p["kind"] for p in mtp.download_plan()["parts"]]
                self.assertNotIn("dflash2", kinds)
                self.assertIn("mtp", kinds, "the MTP is chosen: its layer is the download, DFlash2's is not")
                self.assertNotIn("mtp", [p["kind"] for p in auto.download_plan()["parts"]], "one drafter per run: a new install fetches only the one it will use")
                other = Runner(S.LaunchSettings(family="coder", model="IQ1_M", context=32768, vision="off"), catalog=catalog(HW_12))
                self.assertNotIn("dflash2", [p["kind"] for p in other.download_plan()["parts"]], "Auto on the Coder is the MTP: not offered")


if __name__ == "__main__":
    unittest.main()


class ControlReportsWhetherTheEngineServesIt(unittest.TestCase):
    def test_the_plan_says_it_so_the_app_can_hide_the_choice(self):
        for served in (False, True):
            with mock.patch.object(drafting, "ENGINE_SERVES_DFLASH2", served):
                code, out, err = run_cli(["drafting", "plan", "--mode", "auto", "--family", "gyro", "--data-dir", tempfile.gettempdir(), "--output", "json"])
                self.assertEqual(code, 0, err)
                self.assertIs(json.loads(out)["serves"], served)
