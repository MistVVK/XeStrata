"""The all-cards split engine and the parallel setting: no collision, and the number means the split's lanes.

The split's run config (strata_control/split_engine.split_config) always carries `--batch N`.  The launcher used to write a launch-profile override
`parallel=<card fit>` beside it even when nobody asked, and `strata serve --profile` (strata_cli/cli.py _effective) refuses a parallel override next to
--batch: every 2-card NVIDIA start with the split on Auto stopped with "parallel override conflicts with --batch/--slots in run config args".
Now: the launcher writes no parallel override when the run config holds the slots; `--parallel N` is the split's lanes; the app's Concurrent requests
(a profile override on the saved run config) is accepted for a split config and sets --batch in the engine's arguments.  Nothing is started.
"""

from __future__ import annotations

import contextlib
import copy
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from strata_cli import cli
from strata_cli.launch import settings as S
from strata_cli.launch.runner import Runner
from strata_control import profiles, split_engine as SE
from tests.terminal.launch_support import HW_12, Home, catalog
import serve.server  # noqa: F401  imported before any test patches sys.platform: serve/lifeline.py runs Linux-only libc code when it is imported as Linux

A6000 = {"name": "NVIDIA RTX A6000", "vram_gb": 48.0, "usable": True, "vendor": "nvidia"}
HW_2XA6000 = {**HW_12, "ram_gb": 125.0, "disk_free_gb": 900.0, "engine_vendor": "nvidia", "gpus": [{**A6000, "index": i} for i in range(2)]}
R9700 = {"name": "AMD Radeon AI PRO R9700", "vram_gb": 32.0, "usable": True, "vendor": "amd", "arch": "gfx1201"}
HW_4XR9700 = {**HW_12, "ram_gb": 251.0, "disk_free_gb": 900.0, "engine_vendor": "amd", "gpus": [{**R9700, "index": i} for i in range(4)]}

SPLIT_ARGS = ["--native", "/m/x-00001.gguf", "--mtp", "/m/mtp/rt", "--draft-vocab", "/e/draft_vocab_en.bin", "--max-context", "131072",
              "--spec", "6", "--lookup", "15", "--conversation-cache-mib", "4096", "--batch", "6"]
LAYER_ARGS = ["--pack", "/m/packs/x", "--native", "/m/x-00001.gguf", "--spec", "4", "--mtp", "/m/mtp/rt"]


def split_cfg(lanes: int = 6, **more) -> dict:
    args = list(SPLIT_ARGS)
    args[args.index("--batch") + 1] = str(lanes)
    return {"exe": "/e/strata-split-2-sm120", "args": args, "tokenizer": "/m/tokenizer", "model_name": "m", "port": 8080,
            "split_engine_note": "Using the all-cards split engine on 2 cards.", "drafting": "mtp", **more}


def layer_cfg(**more) -> dict:
    return {"exe": "/e/strata", "args": list(LAYER_ARGS), "tokenizer": "/m/tokenizer", "model_name": "m", "port": 8080, **more}


class Base(unittest.TestCase):
    hw = HW_2XA6000

    def setUp(self):
        self.home = Home().__enter__()
        self.addCleanup(self.home.__exit__, None, None, None)
        # the all-cards split rule asks the host OS and is Linux only (strata_control/split_engine.expected): these fake cards are a Linux PC
        plat = mock.patch("sys.platform", "linux")
        plat.start()
        self.addCleanup(plat.stop)
        # and NVIDIA: an Intel Arc in the box that runs the test would make the launch an Intel one (Runner.intel_chosen reads the real machine)
        intel = mock.patch.object(Runner, "intel_chosen", lambda self: False)
        intel.start()
        self.addCleanup(intel.stop)

    def profile_for(self, cfg: dict, *, parallel=None, hw=None, four_card=None) -> dict:
        """What Runner.ensure_profile saves as the launch profile's overrides for this written run config."""
        cfgfile = Path(self.home.base) / "cfg.json"
        cfgfile.write_text(json.dumps(cfg), encoding="utf-8")
        s = S.LaunchSettings(family="qwen", model="UD-Q4_K_XL", context=131072, tune="skip", parallel=parallel)
        r = Runner(s, catalog=catalog(hw or self.hw))
        if four_card is not None:
            r.four_card_parallel = four_card
        r.ensure_profile(cfgfile)
        prof, _rev = profiles.load(r.profile)
        return dict(prof.get("overrides") or {})


class TerminalPath(Base):
    def test_the_split_run_config_gets_no_parallel_override_by_default(self):
        """The 2-card Auto start: before, the card-fit default (4) was written beside the split's --batch and the server refused to start."""
        ov = self.profile_for(split_cfg())
        self.assertNotIn("parallel", ov)
        values, _o = cli._effective(split_cfg(), {"overrides": ov}, None)      # the check that refused the start
        self.assertEqual(values["parallel"], None)

    def test_a_layer_split_run_config_keeps_the_fitted_default(self):
        ov = self.profile_for(layer_cfg())
        self.assertEqual(ov.get("parallel"), 4, "48 GB card: the fitted default is unchanged on the layer split")
        cli._effective(layer_cfg(), {"overrides": ov}, None)

    def test_parallel_n_on_the_split_is_not_a_profile_override_either(self):
        ov = self.profile_for(split_cfg(lanes=3), parallel=3)
        self.assertNotIn("parallel", ov)
        cli._effective(split_cfg(lanes=3), {"overrides": ov}, None)

    def test_the_amd_four_card_default_does_not_collide_with_the_split(self):
        """AMD gfx1201 x4 starts the split by itself too; amd_four_card.apply put parallel=8 in the run config and the launcher mirrored it into the profile."""
        cfg = split_cfg(lanes=8, parallel=8, exe="/e/strata-split-4-amd")
        ov = self.profile_for(cfg, hw=HW_4XR9700, four_card=8)
        self.assertNotIn("parallel", ov)
        values, origins = cli._effective(cfg, {"overrides": ov}, None)
        self.assertEqual((values["parallel"], origins["parallel"]), (8, "run_config"))

    def test_the_conflict_check_still_stops_a_plain_batch_config(self):
        """Only the split's run config is exempt: another config with its own --batch and a parallel override is still refused."""
        cfg = layer_cfg(args=LAYER_ARGS + ["--batch", "6"])
        with self.assertRaises(profiles.ProfileError):
            cli._effective(cfg, {"overrides": {"parallel": 4}}, None)


class LanesFromTheFlag(Base):
    def run_apply(self, parallel, use_split):
        """Runner.apply_split_engine with the rule stubbed: what the rule sees in the config, and what is left of it afterwards."""
        seen = {}

        def fake(cfg, **_kw):
            seen["parallel"] = cfg.get("parallel")
            if use_split:
                new = split_cfg(lanes=int(cfg.get("parallel") or 6), parallel=cfg.get("parallel"))
                cfg.clear()
                cfg.update(new)
            return SE.Decision(use_split, ["Using the all-cards split engine on 2 cards."] if use_split else ["no"], 2, quiet=not use_split)

        s = S.LaunchSettings(family="qwen", model="UD-Q4_K_XL", context=131072, tune="skip", parallel=parallel)
        r = Runner(s, catalog=catalog(self.hw))
        mtp = Path(self.home.base) / "mtp" / "rt"                     # the draft layer is on disk
        mtp.mkdir(parents=True, exist_ok=True)
        (mtp / "dense.bin").write_bytes(b"x")
        cfg = layer_cfg(args=LAYER_ARGS[:-1] + [str(mtp)])
        with mock.patch.object(SE, "apply_to_config", fake), contextlib.redirect_stdout(io.StringIO()):
            r.apply_split_engine(cfg)
        self.layer_args = LAYER_ARGS[:-1] + [str(mtp)]
        return seen["parallel"], cfg

    def test_parallel_n_reaches_the_rule_and_becomes_the_lane_count(self):
        seen, cfg = self.run_apply(3, True)
        self.assertEqual(seen, 3)
        self.assertEqual(cfg["args"][cfg["args"].index("--batch") + 1], "3")

    def test_without_the_flag_the_rule_sees_nothing_and_the_default_lanes_stay(self):
        seen, cfg = self.run_apply(None, True)
        self.assertIsNone(seen)
        self.assertEqual(cfg["args"][cfg["args"].index("--batch") + 1], "6")

    def test_a_refusal_leaves_the_layer_split_config_without_the_extra_key(self):
        seen, cfg = self.run_apply(3, False)
        self.assertEqual(seen, 3)
        self.assertNotIn("parallel", cfg)
        self.assertEqual(cfg["args"], self.layer_args)


class DraftLayerMissing(Base):
    """The plan for Auto downloads DFlash2's drafter and not the draft layer; the split needs the draft layer (--mtp) and exits at start without it
    ("sfnx: --native and --mtp are required"), so the server fell back to the layer split after a failed start and a misleading line.  Now the launcher
    does not pick the split when the draft layer's folder is not on disk."""

    def run_apply(self, mode, mtp_exists):
        called = []
        mtp = Path(self.home.base) / "mtp" / "rt"
        if mtp_exists:
            mtp.mkdir(parents=True)
            (mtp / "dense.bin").write_bytes(b"x")
        cfg = layer_cfg(args=LAYER_ARGS[:-1] + [str(mtp)], **({"split_engine": mode} if mode else {}))

        def fake(c, **_kw):
            called.append(True)
            return SE.Decision(False, ["no"], 2, quiet=True)

        r = Runner(S.LaunchSettings(family="qwen", model="UD-Q4_K_XL", context=131072, tune="skip"), catalog=catalog(self.hw))
        with mock.patch.object(SE, "apply_to_config", fake), contextlib.redirect_stdout(io.StringIO()):
            r.apply_split_engine(cfg)
        return bool(called), r.notices

    def test_without_the_folder_auto_stays_on_the_layer_split_without_a_word(self):
        called, notices = self.run_apply(None, False)
        self.assertEqual((called, notices), (False, []))

    def test_without_the_folder_on_says_how_to_get_it(self):
        called, notices = self.run_apply("on", False)
        self.assertFalse(called)
        self.assertEqual(len(notices), 1)
        self.assertIn("--drafting mtp", notices[0])

    def test_with_the_folder_the_rule_is_asked_as_before(self):
        self.assertEqual(self.run_apply(None, True), (True, []))

    def test_off_never_asks(self):
        self.assertEqual(self.run_apply("off", True), (False, []))


def cards(n, cc=120, vendor="nvidia", arch=""):
    return [SE.Card(i, "RTX PRO 6000" if vendor == "nvidia" else "R9700", 98304 if vendor == "nvidia" else 32768, cc if vendor == "nvidia" else 0, vendor, arch)
            for i in range(n)]


class AutoFetchesTheDraftLayer(Base):
    """The owner's default: the split engine on 2 cards (sm_86 / sm_120) and on 4 x gfx1201.  It runs the MTP draft layer and nothing else, so where Auto will pick it the plan and
    the fetch are for the MTP (5.2 GB), not DFlash2's drafter; a person's own DFlash2 / off keeps the layer split; one card, other cards and --split-engine off keep DFlash2."""

    def settings(self, **kw):
        return S.LaunchSettings(**{"family": "qwen", "model": "UD-Q4_K_XL", "context": 131072, "tune": "skip", "vision": "off", **kw})

    def runner(self, probe, **kw):
        r = Runner(self.settings(**kw), catalog=catalog(self.hw))
        self.patches = [mock.patch.object(SE, "probe_cards", lambda idx=None: probe), mock.patch.object(SE, "ram_gib", lambda: (500.0, 400.0)),
                        mock.patch("strata_control.drafting.detect_hardware", lambda: ("cuda", ["sm_120"]))]
        for p in self.patches:
            p.start()
            self.addCleanup(p.stop)
        return r

    def test_two_cards_on_auto_resolve_to_the_mtp_and_the_plan_says_why(self):
        r = self.runner(cards(2), gpus=[0, 1])
        self.assertEqual(r.resolved_drafting("qwen"), "mtp")
        self.assertEqual(r.drafting_plan_line(), "Drafting: MTP (the all-cards split engine uses the MTP draft layer, 5.2 GB)")
        kinds = [p["kind"] for p in r.download_plan()["parts"]]
        self.assertIn("mtp", kinds)
        self.assertNotIn("dflash2", kinds)

    def test_the_size_in_the_plan_is_the_size_setup_fetches(self):
        """tools/mtp_fetch.py reads the same 31 pinned tensors of the checkpoint for every set (5.214 GB) and packs them into the 0.9 GB layer the engine loads; the plan, the
        plan line, `strata drafting plan` and setup's own disk check all use that one number."""
        from strata_control.discovery import MTP_FETCH_GB
        r = self.runner(cards(2), gpus=[0, 1])
        part = next(p for p in r.download_plan()["parts"] if p["kind"] == "mtp")
        self.assertEqual(part["gb"], MTP_FETCH_GB)
        self.assertIn(f"{MTP_FETCH_GB:g} GB", r.drafting_plan_line())
        setup_text = (Path(__file__).resolve().parents[2] / "setup.py").read_text(encoding="utf-8")
        self.assertIn(f"mtp_fetch = {MTP_FETCH_GB} if", setup_text, "setup.py counts the layer with the same number")
        self.assertEqual(MTP_FETCH_GB, 5.2)

    def test_the_inputs_of_the_fetch_are_the_mtp_so_the_dflash2_drafter_is_not_fetched(self):
        from strata_control import drafting
        r = self.runner(cards(2), gpus=[0, 1])
        want = r.drafting_inputs({"model_name": "qwen3.8-flash-next-unsloth-ud-q4_k_xl", "args": []})
        self.assertEqual(want["mode"], "mtp")
        self.assertFalse(drafting.needs_pack(**want))
        self.assertFalse(drafting.wants_one_request(**want), "the split has its lanes: no 'one request at a time'")

    def test_a_persons_own_choice_off_or_one_card_or_an_other_card_keeps_dflash2(self):
        for name, probe, kw in (("their own DFlash2", cards(2), {"gpus": [0, 1], "drafting": "dflash2"}),
                                ("split-engine off", cards(2), {"gpus": [0, 1], "split_engine": "off"}),
                                ("one card", cards(1), {"gpus": [0]}),
                                ("sm_89 cards", cards(2, cc=89), {"gpus": [0, 1]}),
                                ("three cards", cards(3), {"gpus": [0, 1, 2]})):
            with self.subTest(name):
                r = self.runner(probe, **kw)
                self.assertEqual(r.resolved_drafting("qwen"), "dflash2")
                self.assertNotIn("mtp", [p["kind"] for p in r.download_plan()["parts"]])

    def test_dflash2_by_name_with_the_split_engine_on_fetches_the_dflash2_drafter_not_the_layer(self):
        """Owner 2026-11-03: both choices are the person's own (--drafting dflash2 --split-engine on): the split engine runs, with DFlash2's drafter."""
        r = self.runner(cards(2), gpus=[0, 1], drafting="dflash2", split_engine="on")
        self.assertTrue(r.split_expected())
        self.assertEqual(r.resolved_drafting("qwen"), "dflash2")
        kinds = [p["kind"] for p in r.download_plan()["parts"]]
        self.assertIn("dflash2", kinds)
        self.assertNotIn("mtp", kinds)
        want = r.drafting_inputs({"model_name": "qwen3.8-flash-next-unsloth-ud-q4_k_xl", "args": []})
        self.assertEqual(want["mode"], "dflash2")
        self.assertNotIn("MTP", r.drafting_plan_line())
        # the same name with the split engine on Auto stays the layer split (and a number of requests changes nothing about which drafter)
        r = self.runner(cards(2), gpus=[0, 1], drafting="dflash2")
        self.assertFalse(r.split_expected())
        r = self.runner(cards(2), gpus=[0, 1], drafting="dflash2", split_engine="on", parallel=4)
        want = r.drafting_inputs({"model_name": "qwen3.8-flash-next-unsloth-ud-q4_k_xl", "args": []})
        self.assertEqual((want["mode"], want["blocked"]), ("dflash2", False), "the person's own number of requests no longer blocks a DFlash2 they named")

    def test_amd_four_gfx1201_is_the_mtp_two_are_not(self):
        self.hw = HW_4XR9700
        r = self.runner(cards(4, vendor="amd", arch="gfx1201"), gpus=[0, 1, 2, 3])
        self.assertEqual(r.resolved_drafting("qwen"), "mtp")
        r2 = self.runner(cards(2, vendor="amd", arch="gfx1201"), gpus=[0, 1])
        self.assertEqual(r2.resolved_drafting("qwen"), "dflash2")

    def test_the_rule_answers_without_files_and_says_no_to_what_it_would_refuse(self):
        self.assertTrue(SE.expected({}, indexes=[0, 1], probe=lambda i: cards(2), ram=lambda: (500.0, 400.0), system="linux"))
        self.assertFalse(SE.expected({"split_engine": "off"}, probe=lambda i: cards(2), ram=lambda: (500.0, 400.0), system="linux"))
        self.assertFalse(SE.expected({"drafting": "dflash2"}, probe=lambda i: cards(2), ram=lambda: (500.0, 400.0), system="linux"))
        self.assertTrue(SE.expected({"drafting": "dflash2", "split_engine": "on"}, probe=lambda i: cards(2), ram=lambda: (500.0, 400.0), system="linux"), "named and asked for: both are the person's own")
        self.assertFalse(SE.expected({}, probe=lambda i: cards(2), ram=lambda: (500.0, 400.0), system="darwin"))
        self.assertFalse(SE.expected({}, probe=lambda i: cards(2), ram=lambda: (500.0, 4.0), system="linux"), "not enough RAM for the locked tables")

    def test_the_apps_drafting_plan_follows_the_same_rule(self):
        from strata_cli import drafting_cmd
        cfg = {"gpu": [0, 1], "args": ["--native", "/nowhere/x-00001.gguf", "--max-context", "131072"]}
        self.runner(cards(2))
        self.assertTrue(drafting_cmd._split_expected(cfg, cfg["args"]))
        self.assertFalse(drafting_cmd._split_expected({**cfg, "split_engine": "off"}, cfg["args"]))
        self.assertFalse(drafting_cmd._split_expected({**cfg, "drafting": "dflash2"}, cfg["args"]))


class SummaryTellsWhatRuns(Base):
    """The server falls back to the normal layer split when the split engine exits at start (serve/server.py start_engine_or_fallback; metrics engine.split_fallback).
    The launcher's summary used to keep the line written before the start, "Using the all-cards split engine on 2 cards"."""

    def runner(self, log_name="e.log"):
        cfgfile = Path(self.home.base) / "cfg.json"
        log = Path(self.home.base) / log_name
        cfgfile.write_text(json.dumps({"log": str(log), "args": []}), encoding="utf-8")
        r = Runner(S.LaunchSettings(family="qwen", model="UD-Q4_K_XL", context=131072, tune="skip", port=18999), catalog=catalog(self.hw))
        r.config_path = cfgfile
        r.notices.append("Using the all-cards split engine on 2 cards. Turn it off with: strata launch --split-engine off.")
        return r, log

    def metrics(self, engine):
        resp = mock.MagicMock()
        resp.read.return_value = json.dumps({"engine": engine}).encode()
        resp.__enter__.return_value = resp
        return mock.patch("urllib.request.urlopen", lambda *a, **k: resp)

    def test_a_fallback_replaces_the_line_and_gives_the_engines_reason(self):
        r, log = self.runner()
        Path(str(log) + ".split-engine").write_text("[strata] engine started: x\nsfnx: --native and --mtp are required\n", encoding="utf-8")
        with self.metrics({"split_fallback": True}), contextlib.redirect_stdout(io.StringIO()):
            r.report_drafting()
        self.assertEqual(r.notices, ["The all-cards split engine could not start (sfnx: --native and --mtp are required); running the normal layer split."])

    def test_a_fallback_without_a_reason_in_the_log_still_says_it(self):
        r, _log = self.runner()
        with self.metrics({"split_fallback": True}), contextlib.redirect_stdout(io.StringIO()):
            r.report_drafting()
        self.assertEqual(r.notices, ["The all-cards split engine could not start; running the normal layer split."])

    def test_when_the_split_runs_the_line_stays(self):
        r, _log = self.runner()
        with self.metrics({}), contextlib.redirect_stdout(io.StringIO()):
            r.report_drafting()
        self.assertEqual(len(r.notices), 1)
        self.assertTrue(r.notices[0].startswith("Using the all-cards split engine"))


class PlanTellsTheApp(Base):
    """`strata drafting plan` is what the app reads (native-control.mjs): with the split taking the start it says choice mtp and split true."""

    def plan(self, probe, *extra):
        from tests.terminal.launch_support import run_cli
        data = Path(self.home.base) / "d"
        data.mkdir(exist_ok=True)
        with mock.patch.object(SE, "probe_cards", lambda idx=None: probe), mock.patch.object(SE, "ram_gib", lambda: (500.0, 400.0)),                 mock.patch("strata_control.drafting.detect_hardware", lambda: ("cuda", ["sm_120"])):
            code, out, err = run_cli(["drafting", "plan", "--family", "qwen", "--data-dir", str(data), "--output", "json", *extra])
        self.assertEqual(code, 0, err)
        return json.loads(out)

    def test_two_cards_on_auto_say_split_and_the_mtp(self):
        p = self.plan(cards(2), "--mode", "auto")
        self.assertEqual((p["split"], p["choice"], p["needed"]), (True, "mtp", False))

    def test_one_card_or_a_chosen_dflash2_do_not(self):
        for probe, mode in ((cards(1), "auto"), (cards(2), "dflash2")):
            p = self.plan(probe, "--mode", mode)
            self.assertFalse(p["split"], (len(probe), mode))
            self.assertEqual(p["choice"], "dflash2")


@contextlib.contextmanager
def tempdir():
    with tempfile.TemporaryDirectory() as tmp:
        yield tmp


def app_cfg(tmp: str, **more) -> dict:
    """The run config the app hands to Control at a Load (native-control.mjs buildConfig): the installed model's layer-split config with the app's own keys
    (host, port, api_key, the card list, parallel 4 where nobody chose a number) and the draft layer on disk."""
    root = Path(tmp)
    (root / "data").mkdir(exist_ok=True)
    (root / "data" / "draft_vocab_en.bin").write_bytes(b"x")
    mtp = root / "mtp" / "rt"
    mtp.mkdir(parents=True, exist_ok=True)
    (mtp / "dense.bin").write_bytes(b"x")
    cfg = {"exe": "/e/strata", "args": ["--pack", "/m/packs/x", "--native", "/m/x-00001.gguf", "--spec", "4", "--mtp", str(mtp), "--max-context", "131072",
                                        "--kv", "int8", "--vram-reserve-mib", "1536"],
           "cwd": tmp, "tokenizer": "/m/tokenizer", "model_name": "qwen3.8-flash-next-unsloth-ud-q4_k_xl", "host": "127.0.0.1", "port": 18500, "api_key": "k", "gpu": [0, 1], "layer_split": "auto",
           "open_browser": False, "parallel": 4, "fit_max_tokens": True, "env": {}}
    cfg.update(more)
    return cfg


def set_arg(cfg: dict, flag: str, value: str) -> None:
    cfg["args"][cfg["args"].index(flag) + 1] = value


def flag_of(cfg: dict, flag: str) -> str:
    return cfg["args"][cfg["args"].index(flag) + 1]


class AppPlanEqualsTerminalPlan(Base):
    """0.2.10.1: the app asks `strata drafting plan` / `fetch` WITHOUT --no-split, so its plan is the Terminal's: where the split rule says yes, Auto is the split with the MTP draft
    layer (fetched through the app's existing MTP step), and a person's DFlash2 or Off keeps the layer split."""

    def plan(self, probe, *extra, config=None):
        from tests.terminal.launch_support import run_cli
        data = Path(self.home.base) / "d"
        data.mkdir(exist_ok=True)
        argv = ["drafting", "plan", "--data-dir", str(data), "--output", "json", *extra]
        if config is not None:
            path = Path(self.home.base) / "cfg-plan.json"
            path.write_text(json.dumps(config), encoding="utf-8")
            argv += ["--config", str(path)]
        else:
            argv += ["--family", "qwen"]
        with mock.patch.object(SE, "probe_cards", lambda idx=None: probe), mock.patch.object(SE, "ram_gib", lambda: (500.0, 400.0)), \
                mock.patch("strata_control.drafting.detect_hardware", lambda: ("cuda", ["sm_120"])):
            code, out, err = run_cli(argv)
        self.assertEqual(code, 0, err)
        return json.loads(out)

    def terminal(self, probe, **kw):
        """What the Terminal's launch decides for the same machine and choice: the drafter, the split word and the parts of its download plan."""
        s = S.LaunchSettings(**{"family": "qwen", "model": "UD-Q4_K_XL", "context": 131072, "tune": "skip", "vision": "off", **kw})
        r = Runner(s, catalog=catalog(self.hw))
        with mock.patch.object(SE, "probe_cards", lambda idx=None: probe), mock.patch.object(SE, "ram_gib", lambda: (500.0, 400.0)), \
                mock.patch("strata_control.drafting.detect_hardware", lambda: ("cuda", ["sm_120"])):
            return r.resolved_drafting("qwen"), r.split_expected(), [p["kind"] for p in r.download_plan()["parts"]]

    def test_the_apps_plan_on_two_cards_is_the_terminals_split_and_the_mtp(self):
        p = self.plan(cards(2), "--mode", "auto")
        drafter, split, kinds = self.terminal(cards(2), gpus=[0, 1])
        self.assertEqual((p["split"], p["choice"], p["needed"]), (True, "mtp", False))
        self.assertEqual((drafter, split), ("mtp", True))
        self.assertIn("mtp", kinds)
        self.assertNotIn("dflash2", kinds)

    def test_without_the_mtp_layer_the_plan_asks_for_it(self):
        """The draft layer is not on disk (a DFlash2 install leaves it out): mtp_needed with its size, and nothing for DFlash2."""
        from strata_control.discovery import MTP_FETCH_GB
        with tempdir() as tmp:
            cfg = app_cfg(tmp)
            set_arg(cfg, "--mtp", str(Path(tmp) / "nowhere" / "rt"))
            p = self.plan(cards(2), "--mode", "auto", config=cfg)
        self.assertEqual((p["split"], p["choice"], p["needed"], p["mtp_needed"], p["mtp_gb"]), (True, "mtp", False, True, MTP_FETCH_GB))
        self.assertIn("all-cards split engine", p["line"])
        self.assertIn(f"{MTP_FETCH_GB:g} GB", p["line"])

    def test_the_plan_line_is_the_terminals_line(self):
        from strata_control.discovery import MTP_FETCH_GB
        self.assertEqual(SE.split_plan_line(MTP_FETCH_GB), f"Drafting: MTP (the all-cards split engine uses the MTP draft layer, {MTP_FETCH_GB:g} GB)")
        self.assertEqual(SE.split_plan_line(), "Drafting: MTP (the all-cards split engine uses the MTP draft layer)")

    def test_a_persons_dflash2_or_off_or_one_card_is_the_same_in_the_app_and_the_terminal(self):
        for name, probe, mode, config, kw in (("their DFlash2", cards(2), "dflash2", None, {"gpus": [0, 1], "drafting": "dflash2"}),
                                              ("split-engine off", cards(2), "auto", {"split_engine": "off", "model_name": "qwen3.8-flash-next-unsloth-ud-q4_k_xl", "args": []}, {"gpus": [0, 1], "split_engine": "off"}),
                                              ("one card", cards(1), "auto", None, {"gpus": [0]})):
            with self.subTest(name):
                p = self.plan(probe, "--mode", mode, config=config)
                drafter, split, kinds = self.terminal(probe, **kw)
                self.assertFalse(p["split"])
                self.assertEqual(p["choice"], "dflash2")
                self.assertEqual((drafter, split), ("dflash2", False))
                self.assertNotIn("mtp", kinds)

    def test_a_saved_dflash2_is_overridden_by_the_auto_the_screen_carries(self):
        """The Models screen's Load with Auto on a model whose settings file still says dflash2: the choice in hand is Auto, so the rule is asked with Auto."""
        with tempdir() as tmp:
            cfg = app_cfg(tmp, drafting="dflash2")
            p_auto = self.plan(cards(2), "--mode", "auto", config=cfg)
            p_saved = self.plan(cards(2), config=cfg)
        self.assertTrue(p_auto["split"])
        self.assertFalse(p_saved["split"])
        self.assertEqual(p_saved["choice"], "dflash2")

    def test_an_explicit_mtp_is_the_split_too_as_in_the_terminal(self):
        p = self.plan(cards(2), "--mode", "mtp")
        _, split, _ = self.terminal(cards(2), gpus=[0, 1], drafting="mtp")
        self.assertEqual((p["split"], p["choice"]), (True, "mtp"))
        self.assertTrue(split)

    def test_four_r9700_in_the_app_are_the_split_and_the_mtp_too(self):
        p = self.plan(cards(4, vendor="amd", arch="gfx1201"), "--mode", "auto")
        self.assertEqual((p["split"], p["choice"]), (True, "mtp"))
        p2 = self.plan(cards(2, vendor="amd", arch="gfx1201"), "--mode", "auto")
        self.assertEqual((p2["split"], p2["choice"]), (False, "dflash2"))

    def test_a_conversation_memory_other_than_int8_is_not_the_split_in_the_plan_either(self):
        with tempdir() as tmp:
            cfg = app_cfg(tmp)
            set_arg(cfg, "--kv", "q4_0")
            p = self.plan(cards(2), "--mode", "auto", config=cfg)
        self.assertFalse(p["split"])
        self.assertEqual(p["choice"], "dflash2")

    def test_no_split_stays_for_a_person_and_for_the_failed_layer_fetch(self):
        from tests.terminal.launch_support import run_cli
        p = self.plan(cards(2), "--mode", "auto", "--no-split")
        self.assertEqual((p["split"], p["choice"], p["mtp_needed"]), (False, "dflash2", False))
        self.assertTrue(p["needed"], "DFlash2's drafter is what the plan fetches then")
        for sub in ("plan", "fetch"):
            code, out, _ = run_cli(["drafting", sub, "--help"])
            self.assertIn("--no-split", out)

    def test_the_hosts_calls_only_carry_no_split_when_it_is_told_to(self):
        """The app (apps/host/server/control-runtime.mjs) adds --no-split only after the split's draft layer could not be downloaded."""
        text = (Path(__file__).resolve().parents[2].parent / "app" / "host" / "server" / "control-runtime.mjs").read_text(encoding="utf-8")
        self.assertNotIn("'--no-split', '--output', 'json'", text)
        self.assertNotIn("'--no-split', '--if-needed'", text)
        self.assertEqual(text.count("...(noSplit ? ['--no-split'] : [])"), 2)


class AppStartsTheSplit(Base):
    """The app's engine start (apps/host: buildConfig, then `profile` + `runtime start`) never went through the Terminal's apply_split_engine; Control now answers it
    (`strata split-engine apply`, strata_control/split_engine.apply_run_config) from the same rule, engine package and draft layer."""

    @staticmethod
    def model():
        return SE.ModelFacts(routed_types={"Q4_K": 97, "Q5_K": 2, "Q5_1": 43, "Q8_0": 5}, routed_bytes=int(71.7 * (1 << 30)), embd_type="Q8_0", shexp_types={"Q8_0": 144},
                             embd_ram_bytes=int(27.5 * (1 << 30)), layers=48, experts=512)

    def apply(self, cfg, *, probe=None, avail=400.0, exe="/e/strata-split-2-sm120", system="linux"):
        probe = probe or cards(2)
        return SE.apply_run_config(cfg, exe_for_cards=lambda n, cc=None, arch=None: exe, system=system, probe=lambda idx: probe, ram=lambda: (500.0, avail),
                                   scan=lambda path: self.model())

    def test_the_apps_default_of_four_slots_is_the_splits_four_lanes(self):
        with tempdir() as tmp:
            cfg = app_cfg(tmp)
            d, line, warn = self.apply(cfg)
            self.assertTrue(d.use_split, d.reasons)
            self.assertEqual(flag_of(cfg, "--batch"), "4")
            self.assertEqual(cfg["exe"], "/e/strata-split-2-sm120")
            self.assertEqual(cfg["drafting"], "mtp")
            self.assertIn("all-cards split engine on 2 cards", line)
            self.assertFalse(warn)
            # the app's own keys ride along, and the normal layer split is kept whole for the server's fallback
            for key in ("host", "port", "api_key", "tokenizer", "model_name", "fit_max_tokens", "parallel"):
                self.assertIn(key, cfg, key)
            self.assertEqual(cfg["split_fallback"]["exe"], "/e/strata")
            self.assertIn("--vram-reserve-mib", cfg["split_fallback"]["args"])
            self.assertNotIn("--vram-reserve-mib", cfg["args"], "the layer split's own flags are not given to the split engine")

    def test_a_number_the_person_chose_is_the_lanes_and_never_more_than_eight(self):
        for chosen, want in ((1, "1"), (3, "3"), (8, "8"), (12, "8")):
            with self.subTest(chosen), tempdir() as tmp:
                cfg = app_cfg(tmp, parallel=chosen)
                d, _, _ = self.apply(cfg)
                self.assertTrue(d.use_split)
                self.assertEqual(flag_of(cfg, "--batch"), want)

    def test_no_number_at_all_gives_the_splits_own_default_lanes(self):
        with tempdir() as tmp:
            cfg = app_cfg(tmp)
            del cfg["parallel"]
            self.apply(cfg)
            self.assertEqual(flag_of(cfg, "--batch"), "6")

    def test_vision_is_passed_when_the_app_reads_pictures(self):
        with tempdir() as tmp:
            cfg = app_cfg(tmp, vision={"exe": "/e/strata-vision", "mmproj": "/m/mm.gguf"})
            self.apply(cfg)
            self.assertIn("--vision", cfg["args"])
        with tempdir() as tmp:
            cfg = app_cfg(tmp)
            self.apply(cfg)
            self.assertNotIn("--vision", cfg["args"])

    def test_every_no_leaves_the_apps_run_config_alone_and_auto_says_nothing(self):
        cases = {"a person's DFlash2": ({"drafting": "dflash2"}, {}), "drafting off": ({"drafting": "off"}, {}), "the switch off": ({"split_engine": "off"}, {}),
                 "engine arguments of their own": ({"user_args": ["--x", "1"]}, {}), "one card": ({"gpu": 0}, {"probe": cards(1)}),
                 "two other cards": ({}, {"probe": cards(2, cc=89)}), "too little RAM": ({}, {"avail": 10.0}), "no split engine in the package": ({}, {"exe": None}),
                 "Windows": ({}, {"system": "windows"})}
        for name, (more, kw) in cases.items():
            with self.subTest(name), tempdir() as tmp:
                cfg = app_cfg(tmp, **more)
                before = copy.deepcopy(cfg)
                d, line, _ = self.apply(cfg, **kw)
                self.assertFalse(d.use_split, name)
                self.assertEqual(cfg, before, name)
                self.assertEqual(line, "", f"{name}: Auto says nothing")
        with tempdir() as tmp:                                # a conversation memory other than int8 is refused by the rule itself
            cfg = app_cfg(tmp)
            set_arg(cfg, "--kv", "q4_0")
            before = copy.deepcopy(cfg)
            d, line, _ = self.apply(cfg)
            self.assertFalse(d.use_split)
            self.assertEqual((cfg, line), (before, ""))

    def test_a_missing_draft_layer_is_a_quiet_layer_split_and_on_says_how_to_get_it(self):
        with tempdir() as tmp:
            cfg = app_cfg(tmp)
            set_arg(cfg, "--mtp", str(Path(tmp) / "nowhere" / "rt"))
            d, line, _ = self.apply(cfg)
            self.assertFalse(d.use_split)
            self.assertEqual(line, "")
            self.assertNotIn("split_fallback", cfg)
            cfg2 = app_cfg(tmp, split_engine="on")
            set_arg(cfg2, "--mtp", str(Path(tmp) / "nowhere" / "rt"))
            d2, line2, warn2 = self.apply(cfg2)
            self.assertFalse(d2.use_split)
            self.assertIn("draft layer", line2)
            self.assertTrue(warn2)

    def test_the_command_writes_the_split_config_only_on_a_yes_and_never_touches_the_input(self):
        from tests.terminal.launch_support import run_cli
        with tempdir() as tmp:
            src = Path(tmp) / "layer.json"
            out = Path(tmp) / "split.json"
            cfg = app_cfg(tmp)
            src.write_text(json.dumps(cfg), encoding="utf-8")
            before = src.read_text(encoding="utf-8")
            with mock.patch.object(SE, "probe_cards", lambda idx=None: cards(2)), mock.patch.object(SE, "ram_gib", lambda: (500.0, 400.0)), \
                    mock.patch.object(SE, "scan_model", lambda path: self.model()), \
                    mock.patch.object(SE, "exe_finder", lambda c: (lambda n, cc=None, arch=None: "/e/strata-split-2-sm120")), \
                    mock.patch("sys.platform", "linux"):
                code, text, err = run_cli(["split-engine", "apply", "--config", str(src), "--out", str(out), "--output", "json"])
                self.assertEqual(code, 0, err)
                got = json.loads(text)
                self.assertTrue(got["ok"] and got["split"], got)
                self.assertEqual(got["cards"], 2)
                written = json.loads(out.read_text(encoding="utf-8"))
                self.assertEqual(flag_of(written, "--batch"), "4")
                self.assertEqual(src.read_text(encoding="utf-8"), before)
                out.unlink()
                src.write_text(json.dumps({**cfg, "drafting": "dflash2"}), encoding="utf-8")
                code, text, err = run_cli(["split-engine", "apply", "--config", str(src), "--out", str(out), "--output", "json"])
                got = json.loads(text)
                self.assertEqual((code, got["split"], got["written"]), (0, False, None))
                self.assertFalse(out.exists())
                src.write_text("not json", encoding="utf-8")
                code, text, _ = run_cli(["split-engine", "apply", "--config", str(src), "--out", str(out), "--output", "json"])
                self.assertNotEqual(code, 0)
                self.assertFalse(json.loads(text)["ok"])

    def test_the_fallback_layer_split_keeps_the_apps_slots_and_its_own_drafting_choice(self):
        """If the split engine will not come up, the server starts the layer split from split_fallback (serve/server.py start_engine_or_fallback) with the app's parallel."""
        from serve.server import SPLIT_KEYS
        with tempdir() as tmp:
            cfg = app_cfg(tmp)
            self.apply(cfg)
            cfg2 = {k: v for k, v in cfg.items() if k not in SPLIT_KEYS + ("exe", "args", "env", "layer_split", "replicas", "user_args", "drafting")}
            cfg2.update(cfg["split_fallback"])
            self.assertEqual(cfg2["parallel"], 4)
            self.assertEqual(cfg2["exe"], "/e/strata")
            self.assertNotIn("drafting", cfg2)


class LaneClamp(Base):
    """The engine runs 1..8 lanes ("sfnx: --batch must be 1..8"): the AMD four-card default PARALLEL 32, a layer config's own --batch and an app profile's parallel
    are cut to 8 in the split's run config and in the server's --batch."""

    def split_args(self, lanes):
        d = SE.Decision(True, [], 2)
        cfg = SE.split_config({"tokenizer": "/t", "port": 1}, d, exe="/e/strata-split-2-sm86", native="/m/x-00001.gguf", mtp_dir="/m/mtp/rt", draft_vocab="/e/v.bin",
                              context=131072, lanes=lanes)
        return cfg["args"]

    def test_split_config_cuts_the_lanes_to_1_through_8(self):
        for lanes, want in ((32, "8"), (9, "8"), (8, "8"), (5, "5"), (1, "1"), (0, "1")):
            a = self.split_args(lanes)
            self.assertEqual(a[a.index("--batch") + 1], want, lanes)

    def stubbed(self, cfg, probe, mode_="on", exe="/e/strata-split-2-sm86"):
        data = Path(self.home.base) / "vocab"
        data.mkdir(exist_ok=True)
        (data / "draft_vocab_en.bin").write_bytes(b"v")
        return SE.apply_to_config(cfg, exe_for_cards=lambda n, cc=None, arch=None: exe, system="linux", probe=lambda i: probe, ram=lambda: (500.0, 400.0),
                                  scan=lambda p: SE.ASSUMED_MODEL, data_dir=str(data), mode_=mode_)

    def test_iq3_s_on_two_cards_with_a_layer_config_batch_32_starts_with_8(self):
        native = Path(self.home.base) / "x-00001.gguf"
        native.write_bytes(b"g")
        cfg = {"gpu": [0, 1], "split_engine": "on", "args": ["--pack", "/p", "--native", str(native), "--mtp", "/m/mtp/rt", "--max-context", "131072", "--batch", "32"]}
        d = self.stubbed(cfg, cards(2, cc=86))
        self.assertTrue(d.use_split, d.reasons)
        self.assertEqual(cfg["args"][cfg["args"].index("--batch") + 1], "8")

    def test_the_amd_four_card_default_parallel_32_starts_with_8(self):
        native = Path(self.home.base) / "x-00001.gguf"
        native.write_bytes(b"g")
        cfg = {"gpu": [0, 1, 2, 3], "parallel": 32, "args": ["--pack", "/p", "--native", str(native), "--mtp", "/m/mtp/rt", "--max-context", "131072"]}
        d = self.stubbed(cfg, cards(4, vendor="amd", arch="gfx1201"), mode_="auto", exe="/e/strata-split-4-amd")
        self.assertTrue(d.use_split, d.reasons)
        self.assertEqual(cfg["args"][cfg["args"].index("--batch") + 1], "8")

    def test_a_profile_parallel_of_16_gives_the_servers_batch_8(self):
        from serve.server import parallel_args
        for n, want in ((16, "8"), (32, "8"), (3, "3"), (8, "8")):
            args = list(SPLIT_ARGS)
            parallel_args(split_cfg(parallel=n), args)
            self.assertEqual(args[args.index("--batch") + 1], want, n)


class AppPath(Base):
    """The app's Settings 'Concurrent requests = N' is a profile override on the saved run config (strata_control/profiles); `strata serve --profile` reads it."""

    def engine_argv(self, cfg: dict, overrides: dict) -> list[str]:
        from serve.server import engine_args
        values, _o = cli._effective(cfg, {"overrides": overrides}, None)
        eff = {**cfg, "args": list(cfg["args"])}
        if values["parallel"] is not None:
            eff["parallel"] = values["parallel"]
        with contextlib.redirect_stdout(io.StringIO()):
            return engine_args(eff)

    def test_concurrent_requests_on_a_split_config_sets_its_lanes(self):
        argv = self.engine_argv(split_cfg(lanes=6), {"host": "127.0.0.1", "port": 8080, "parallel": 3})
        self.assertEqual(argv.count("--batch"), 1)
        self.assertEqual(argv[argv.index("--batch") + 1], "3")

    def test_no_override_keeps_the_lanes_the_launcher_wrote(self):
        argv = self.engine_argv(split_cfg(lanes=5), {"host": "127.0.0.1", "port": 8080})
        self.assertEqual(argv[argv.index("--batch") + 1], "5")

    def test_the_servers_parallel_flag_is_accepted_for_the_split_and_still_refused_beside_another_batch(self):
        """`strata serve --profile` hands the profile's parallel to the server as --parallel N; the server refused it next to any --batch in the saved args."""
        from serve.server import parallel_flag_conflicts
        self.assertFalse(parallel_flag_conflicts(split_cfg()))
        self.assertFalse(parallel_flag_conflicts(layer_cfg()))
        self.assertTrue(parallel_flag_conflicts(layer_cfg(args=LAYER_ARGS + ["--batch", "6"])))
        self.assertTrue(parallel_flag_conflicts(layer_cfg(args=LAYER_ARGS + ["--slots", "6"])))

    def test_the_layer_split_still_takes_its_batch_from_parallel(self):
        argv = self.engine_argv(layer_cfg(), {"host": "127.0.0.1", "port": 8080, "parallel": 3})
        self.assertEqual(argv[argv.index("--batch") + 1], "3")


if __name__ == "__main__":
    unittest.main()
