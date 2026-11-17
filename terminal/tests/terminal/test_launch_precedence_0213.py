"""Round 0213 TUICFG, step 3: one precedence for a launch's engine options (#5178, #5132, #4985).

Explicit launch values, then persisted overrides, then calibrated values, then Auto (strata_control/arg_layers.py). Each case builds the run
config like a launch does (derive_config, stamp, write, read back), then asks serve/server.py engine_args for the final native argv, and
launches a second time with no new flags. The Control and the engine are doubles; nothing starts."""
from __future__ import annotations

import json
import unittest
from unittest import mock

from strata_cli.launch import settings as S
from strata_cli.launch.runner import derive_config
from strata_control import arg_layers, drafting, tune_knobs
from tests.terminal.test_drafting_single import Base, flag


class Launches(Base):
    def server_args(self, cfg, backend="cuda", archs=("sm_120",), vram=8000.0):
        from serve.server import engine_args
        with mock.patch.object(drafting, "detect_hardware", return_value=(backend, list(archs))), mock.patch("serve.server.drafting_vram_mib", return_value=vram), \
                mock.patch.object(drafting, "installed_engine_exe", return_value=self.exe):
            return engine_args(cfg, str(self.exe), announce=False)

    def base(self, **extra):
        return {"exe": "component:strata-engine", "model_name": "qwen3.8-flash-next-iq3_s", "port": 8080, "args": list(self.args), **extra}

    def launch(self, base=None, extra=None, prev=None, **settings):
        """What `Runner.build_config` writes: derive, stamp, and a round trip through the file."""
        s = S.LaunchSettings(extra=dict(extra or {}), **settings)
        with mock.patch.object(drafting, "detect_hardware", return_value=("cuda", ["sm_120"])):
            cfg = derive_config(base or self.base(), s, bool_flags=tune_knobs.BOOL_FLAGS, prev=prev)
        arg_layers.stamp(cfg)
        return json.loads(json.dumps(cfg))

    # ---- #5178
    def test_an_explicit_draft_conf_reaches_the_engine_and_survives_a_second_launch(self):
        for mode in ("dflash2", "auto"):
            with self.subTest(mode=mode):
                first = self.launch(extra={"--draft-conf": "0.35"}, drafting=mode)
                argv = self.server_args(first, vram=24000.0)           # (an 8 GB card is the 12 GB class, where Auto is the MTP for IQ3_S: drafting.CLASS_DEFAULTS)
                self.assertEqual((argv.count("--draft-conf"), flag(argv, "--draft-conf")), (1, "0.35"))
                self.assertEqual(flag(first["user_args"], "--draft-conf"), "0.35", "the saved config carries it where the drafter policy keeps it")
                second = self.launch(extra={"--draft-conf": "0.35"}, prev=first, drafting=mode)
                self.assertEqual(self.server_args(second, vram=24000.0), argv, "a second launch with the same flags is the same argv")
                self.assertEqual(second["args"].count("--draft-conf"), 1)

    def test_an_explicit_value_is_not_remembered_when_the_next_launch_leaves_it_out(self):
        first = self.launch(extra={"--draft-conf": "0.35"}, drafting="dflash2")
        second = self.launch(prev=first, drafting="dflash2")
        argv = self.server_args(second)
        self.assertNotEqual(flag(argv, "--draft-conf"), "0.35", "back to Auto")
        self.assertNotIn("--draft-conf", second.get("user_args") or [])
        self.assertNotIn("launch_set", second)

    def test_the_new_explicit_value_beats_the_earlier_one(self):
        first = self.launch(extra={"--draft-conf": "0.35"}, drafting="dflash2")
        second = self.launch(extra={"--draft-conf": "0.6"}, prev=first, drafting="dflash2")
        argv = self.server_args(second)
        self.assertEqual((argv.count("--draft-conf"), flag(argv, "--draft-conf")), (1, "0.6"))

    def test_an_explicit_spec_survives_the_drafter_policy_too(self):
        cfg = self.launch(extra={"--spec": "3"}, drafting="dflash2")
        argv = self.server_args(cfg)
        self.assertEqual((argv.count("--spec"), flag(argv, "--spec")), (1, "3"))

    # ---- explicit off
    def test_an_explicit_off_removes_the_option_and_stays_off_for_the_same_command(self):
        base = self.base(args=[*self.args, "--mmap-experts"])
        first = self.launch(base, extra={"--mmap-experts": "off"})
        self.assertNotIn("--mmap-experts", first["args"])
        self.assertNotIn("--mmap-experts", self.server_args(first))
        second = self.launch(base, extra={"--mmap-experts": "off"}, prev=first)
        self.assertNotIn("--mmap-experts", self.server_args(second))
        third = self.launch(base, prev=second)                 # the flag left out: the installed config's own value is back
        self.assertIn("--mmap-experts", self.server_args(third))

    def test_an_explicit_off_beats_a_persisted_override(self):
        base = self.base(user_args=["--pcie-frac", "0.3"], args=[*self.args, "--pcie-frac", "0.3"])
        cfg = self.launch(base, extra={"--pcie-frac": "off"})
        argv = self.server_args(cfg)
        self.assertNotIn("--pcie-frac", argv)
        self.assertNotIn("--pcie-frac", cfg.get("user_args") or [])

    # ---- #5132
    def test_a_hand_edit_of_the_launch_config_survives_the_next_launch(self):
        first = self.launch()
        self.assertNotIn("--pcie-frac", first["args"])
        first["args"] += ["--pcie-frac", "0.2", "--pool-workers", "6"]       # written by hand (or by a tool that does not stamp)
        second = self.launch(prev=first)
        argv = self.server_args(second)
        self.assertEqual((flag(argv, "--pcie-frac"), flag(argv, "--pool-workers")), ("0.2", "6"))
        self.assertEqual((argv.count("--pcie-frac"), argv.count("--pool-workers")), (1, 1))
        third = self.launch(prev=second)                                     # and again with no new flags
        self.assertEqual(self.server_args(third), argv)
        self.assertEqual(flag(third["user_args"], "--pcie-frac"), "0.2")

    def test_the_launch_flags_do_not_come_back_as_hand_edits(self):
        first = self.launch(extra={"--pcie-frac": "0.4"})
        again = self.launch(prev=first)
        self.assertNotIn("--pcie-frac", again["args"], "given once, not remembered")
        again2 = self.launch(extra={"--pcie-frac": "0.4"}, prev=first)
        self.assertEqual(flag(again2["args"], "--pcie-frac"), "0.4")

    def test_a_config_from_before_the_stamp_is_not_mistaken_for_hand_edits(self):
        old = self.launch()
        for key in ("args_derived",):
            old.pop(key)
        old["args_written"] = ["--unrelated"]                # the installed config's copy, as older launches left it
        old["args"] += ["--cards", "0"]
        new = self.launch(prev=old)
        self.assertNotIn("--cards", new["args"])

    def test_calibrated_values_are_kept_by_the_next_launch(self):
        from strata_control import calibration
        cfg0 = self.launch()                                 # the launch config the profile points at
        applied = json.loads(json.dumps(cfg0))
        applied["args"] = tune_knobs.apply_to_argv(applied["args"], {"--pcie-frac": "0.2", "--pool-workers": "6"})
        arg_layers.record_calibration(applied, cfg0, {"--pcie-frac": "0.2", "--pool-workers": "6"})
        self.assertEqual(applied["calibrated"]["settings"], {"--pcie-frac": "0.2", "--pool-workers": "6"})
        self.assertEqual(applied["args_derived"], applied["args"], "its own values are not hand edits")
        second = self.launch(prev=applied)
        argv = self.server_args(second)
        self.assertEqual((flag(argv, "--pcie-frac"), flag(argv, "--pool-workers")), ("0.2", "6"))
        self.assertIn("calibrated", second)
        self.assertNotIn("--pcie-frac", second.get("user_args") or [], "calibrated is not an override: provenance kept")
        third = self.launch(prev=second)
        self.assertEqual(self.server_args(third), argv)
        self.assertTrue(hasattr(calibration, "apply_result"))

    def test_calibrated_values_lose_to_persisted_overrides_and_explicit_values(self):
        cfg0 = self.launch()
        applied = json.loads(json.dumps(cfg0))
        applied["args"] = tune_knobs.apply_to_argv(applied["args"], {"--pcie-frac": "0.2"})
        arg_layers.record_calibration(applied, cfg0, {"--pcie-frac": "0.2"})
        base = self.base(user_args=["--pcie-frac", "0.3"])
        self.assertEqual(flag(self.server_args(self.launch(base, prev=applied)), "--pcie-frac"), "0.3", "persisted over calibrated")
        self.assertEqual(flag(self.server_args(self.launch(base, extra={"--pcie-frac": "0.5"}, prev=applied)), "--pcie-frac"), "0.5", "explicit over both")
        self.assertEqual(flag(self.server_args(self.launch(prev=applied)), "--pcie-frac"), "0.2", "calibrated over Auto")

    def test_a_calibration_for_another_context_or_model_is_not_applied(self):
        cfg0 = self.launch()
        applied = json.loads(json.dumps(cfg0))
        applied["args"] = tune_knobs.apply_to_argv(applied["args"], {"--pcie-frac": "0.2"})
        arg_layers.record_calibration(applied, cfg0, {"--pcie-frac": "0.2"})
        other_ctx = self.base(args=[a if a != "131072" else "65536" for a in self.args])
        self.assertNotIn("--pcie-frac", self.launch(other_ctx, prev=applied)["args"])
        other_model = self.base(model_name="qwen3.8-flash-next-gyro-gyro-s")
        self.assertNotIn("--pcie-frac", self.launch(other_model, prev=applied)["args"])

    # ---- #4985 and the shape of the file
    def test_one_effective_value_per_option_in_every_case(self):
        cases = [dict(), dict(extra={"--draft-conf": "0.35"}), dict(extra={"--pcie-frac": "0.4", "--spec": "3"}), dict(extra={"--pool-workers": "off"})]
        prev = None
        for kw in cases:
            cfg = self.launch(prev=prev, drafting="dflash2", **kw)
            argv = self.server_args(cfg)
            for name in set(a for a in argv if a.startswith("--") and a not in tune_knobs.BOOL_FLAGS):
                self.assertEqual(argv.count(name), 1, (kw, name))
            prev = cfg


if __name__ == "__main__":
    unittest.main()
