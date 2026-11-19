"""0.2.17 RAMMODE (owner 11-19): setup writes two RAM budgets for UD-Q4_K_XL / UD-IQ4_XS.  `--resident-budget-gib` in "args" is the new rule (the RAM less 17 GB: 47 on
64 GB), used by "Full performance" and by the Terminal; "ui_budget_work_gib" is the old rule (the RAM less 24 GB: 40 on 64 GB) that the app's "Leave 20% for my work"
swaps in (app/host/server/native-resources.mjs, tests/resource-rammode.test.mjs).  The rule lives in one place, setup.resident_budget_gib(work=...).  The person's own
budget wins in every mode: the key is left out when setup was asked for a budget, when the config keeps one of the person's, and where both rules give the same number.
Every outside effect is mocked (tools/test_setup_golden.py's install()).

    python -m unittest tools.test_setup_rammode
"""
from __future__ import annotations

import contextlib
import io
import json
import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
import setup  # noqa: E402
from serve.server import engine_args  # noqa: E402
from test_setup_golden import PROFILES, install  # noqa: E402
from test_setup_unsloth import Base  # noqa: E402

M = "UD-Q4_K_XL"
KEY = "ui_budget_work_gib"
NAME = "strata-unsloth-ud-q4_k_xl.json"


def value(args, flag):
    return args[args.index(flag) + 1] if flag in args[:-1] else None


class Rule(unittest.TestCase):
    def test_the_two_numbers_for_every_ram_size(self):
        # ram_gb() reads a little under the label.  new = the RAM less 17 (Terminal, Full performance), work = the RAM less 24 (Leave 20% for my work)
        for label, new, work in ((32, 15, 8), (48, 31, 24), (64, 47, 40), (96, 71, 71), (128, 71, 71)):
            ram = label - 0.3
            self.assertEqual(setup.resident_budget_gib(M, ram), new, label)
            self.assertEqual(setup.resident_budget_gib(M, ram, work=True), work, label)
            self.assertEqual(setup.resident_budget_gib(M, ram, uma=True), work, label)     # a unified-memory GPU: 24 in both
            self.assertEqual(setup.resident_budget_gib(M, ram, uma=True, work=True), work, label)
        self.assertEqual(setup.resident_budget_gib(M, 63.7, 1.7), 45)                      # a streamed KV cache comes out of both
        self.assertEqual(setup.resident_budget_gib(M, 63.7, 1.7, work=True), 38)
        self.assertEqual(setup.UNSLOTH_RAM_LEFT_GB, 24)
        self.assertEqual(setup.UNSLOTH_BUDGET_LEFT_GB, 17)
        # the other set that has a budget: UD-IQ4_XS (its cap is 55 GiB)
        self.assertEqual((setup.resident_budget_gib("UD-IQ4_XS", 63.7), setup.resident_budget_gib("UD-IQ4_XS", 63.7, work=True)), (47, 40))

    def test_the_key_helper(self):
        def cfg(*extra):
            return {"args": ["--max-context", "8192", *extra], KEY: 99}
        c = cfg("--resident-budget-gib", "47")
        setup.work_budget_key(c, 40)
        self.assertEqual(c[KEY], 40)
        for label, c, work, asked in (("the person asked for a budget", cfg("--resident-budget-gib", "52"), 40, 52.0),
                                      ("both rules give the same number", cfg("--resident-budget-gib", "40"), 40, None),
                                      ("no budget in the args", cfg(), 40, None),
                                      ("no number", cfg("--resident-budget-gib", "47"), None, None),
                                      ("another mode", cfg("--mmap-experts"), 40, None)):
            setup.work_budget_key(c, work, asked)
            self.assertNotIn(KEY, c, label)
        self.assertIn(KEY, setup.SETUP_KEYS)          # setup's own: a re-run refreshes it and takes it out again, never carries an old one over


class Nominal(unittest.TestCase):
    """The reading is rounded UP to the size the PC is sold with, within 3% under it (Windows reads 63.1 on a 64 GB PC, and gave 46 where Linux gave 47)."""

    def test_rounds_up_to_the_size(self):
        for reading, size in ((63.1, 64), (63.11, 64), (63.7, 64), (62.5, 64), (62.1, 64), (64.0, 64), (31.6, 32), (47.6, 48), (47.0, 48), (95.7, 96), (127.8, 128),
                              (15.7, 16), (23.5, 24), (191.0, 192), (254.0, 256), (300.0, 304), (303.0, 304)):
            self.assertEqual(setup.nominal_ram_gb(reading), size, reading)

    def test_a_pc_with_less_stays_as_it_reads(self):
        for reading in (60.0, 62.0, 61.0, 44.0, 30.0, 40.0, 80.0, 0.0, 20.0, 280.0):
            self.assertEqual(setup.nominal_ram_gb(reading), reading, reading)

    def test_the_budget_rule_uses_it(self):
        for reading, new, work in ((63.1, 47, 40), (63.7, 47, 40), (62.5, 47, 40), (31.6, 15, 8), (47.6, 31, 24), (60.0, 43, 36)):
            self.assertEqual(setup.resident_budget_gib(M, reading), new, reading)
            self.assertEqual(setup.resident_budget_gib(M, reading, work=True), work, reading)
        self.assertEqual(setup.resident_budget_gib(M, 63.1, uma=True), 40)            # a unified-memory PC: the same reading, 24 left


def old_config(budget, **extra):
    """A run config as 0.2.16 setup wrote it (no second number), with its copy of what it wrote."""
    args = ["--native", "/models/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf", "--max-context", "8192", "--kv", "int8", "--resident-budget-gib", str(budget)]
    return {"exe": "component:strata-engine", "args": args, "args_written": list(args), "model_name": "qwen3.8-flash-next-unsloth-ud-q4_k_xl", "gpu": 0, **extra}


class Move(unittest.TestCase):
    """0.2.17 (owner 11-19): an installed 0.2.16 config moves to the new rule without a setup run; what the person set stays."""

    def move(self, cfg, ram=63.7, uma=False):
        return setup.migrate_ram_budget(cfg, ram=ram, uma=uma)

    def test_a_0216_config_moves_and_work_mode_keeps_the_old_number(self):
        for ram, old, new in ((31.7, 8, 15), (47.7, 24, 31), (63.7, 40, 47), (62.6, 40, 47)):
            cfg = old_config(old)
            before = json.dumps(cfg)
            moved = self.move(cfg, ram)
            self.assertEqual(value(moved["args"], "--resident-budget-gib"), str(new), ram)
            self.assertEqual(moved[KEY], old, ram)                                    # "Leave 20% for my work" keeps the old amount
            self.assertEqual(moved["args_written"], moved["args"], ram)               # setup's own change, not the person's
            self.assertEqual(json.dumps(cfg), before, "the input is not changed")
            self.assertEqual({k: v for k, v in moved.items() if k not in ("args", "args_written", KEY)}, {k: v for k, v in cfg.items() if k not in ("args", "args_written")})

    def test_the_owner_pc_on_windows(self):
        moved = self.move(old_config(39), ram=63.11)                                  # 0.2.16 read 63.11 and wrote 39
        self.assertEqual((value(moved["args"], "--resident-budget-gib"), moved[KEY]), ("47", 40))

    def test_a_second_run_changes_nothing(self):
        moved = self.move(old_config(40))
        self.assertIsNone(self.move(moved))
        self.assertIsNone(self.move(old_config(47)))                                  # already the new number
        self.assertIsNone(self.move({**old_config(40), KEY: 40}))                     # the second number is there

    def test_every_expert_in_ram_and_no_budget(self):
        cfg = old_config(40)
        for flag in ("--resident-budget-gib", "40"):
            cfg["args"].remove(flag)
        cfg["args_written"] = list(cfg["args"])
        self.assertIsNone(self.move(cfg))
        self.assertIsNone(self.move(old_config(71), ram=95.7))                        # 96 GB: 71 in both rules

    def test_what_the_person_set_is_never_touched(self):
        mine = old_config(40)
        mine["user_args"] = ["--resident-budget-gib", "40"]
        edited = old_config(40)
        edited["args"][-1] = "36"                                                     # a hand edit of args: differs from what setup wrote
        edited_same = old_config(40)
        edited_same["args_written"][-1] = "36"                                        # (a budget the person edited back to setup's number is theirs too)
        keep = old_config(40, env={"STRATA_RAM_KEEP_GIB": "12"})
        headroom = old_config(40, env={"STRATA_RESIDENT_HEADROOM_GIB": "20"})
        flag = old_config(40)
        flag["args"] += ["--ram-keep-gib", "12"]
        flag["args_written"] += ["--ram-keep-gib", "12"]
        no_copy = old_config(40)
        del no_copy["args_written"]                                                   # an older config: setup's number cannot be told from the person's
        not_the_rule = old_config(36)                                                 # not what the old rule gives for this PC
        for label, cfg in (("user_args", mine), ("args edited", edited), ("written differs", edited_same), ("STRATA_RAM_KEEP_GIB", keep),
                           ("STRATA_RESIDENT_HEADROOM_GIB", headroom), ("--ram-keep-gib", flag), ("no args_written", no_copy), ("a number of its own", not_the_rule)):
            self.assertIsNone(self.move(cfg), label)

    def test_an_env_setting_the_app_wrote_is_not_the_persons(self):
        cfg = old_config(40, env={"STRATA_RAM_KEEP_GIB": "8"}, ui_resource_baseline={"workers": 6, "ramKeep": True})
        self.assertEqual(value(self.move(cfg)["args"], "--resident-budget-gib"), "47")
        app_budget = old_config(40, ui_resource_baseline={"workers": 6, "ramBudget": True})      # the 0.2.9 app's 80% rule wrote it
        self.assertIsNone(self.move(app_budget))

    def test_unified_memory_and_the_layer_split_stay_at_24(self):
        self.assertIsNone(self.move(old_config(40), uma=True))
        self.assertIsNone(self.move({**old_config(40), "gpu": [0, 1], "layer_split": "auto"}))

    def test_a_streamed_kv_cache_comes_out_of_both(self):
        cfg = old_config(38)
        cfg["args"][cfg["args"].index("--max-context") + 1] = "131072"
        cfg["args"] += ["--kv-resident", "32768"]
        cfg["args_written"] = list(cfg["args"])
        moved = self.move(cfg)
        self.assertEqual((value(moved["args"], "--resident-budget-gib"), moved[KEY]), ("45", 38))

    def test_iq4_xs(self):
        cfg = old_config(40)
        cfg["args"][1] = "/models/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf"
        cfg["args_written"] = list(cfg["args"])
        self.assertEqual(setup.budget_model(cfg), "UD-IQ4_XS")
        moved = self.move(cfg)
        self.assertEqual((value(moved["args"], "--resident-budget-gib"), moved[KEY]), ("47", 40))


class MoveFiles(unittest.TestCase):
    """strata_cli/ram_budget_move.py: once per folder, byte-identical where nothing moves, the earlier file kept."""

    def setUp(self):
        import tempfile
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.patches = [mock.patch.object(setup, "ram_gb", lambda: 63.7), mock.patch.object(setup, "_unified_memory_pc", lambda: False)]
        for p in self.patches:
            p.start()
        from strata_cli import ram_budget_move
        self.mod = ram_budget_move

    def tearDown(self):
        for p in self.patches:
            p.stop()
        self.tmp.cleanup()

    def write(self, name, cfg):
        (self.dir / name).write_text(json.dumps(cfg, indent=1), encoding="utf-8")

    def test_moves_once_and_leaves_the_rest_alone(self):
        self.write("strata-unsloth-ud-q4_k_xl.json", old_config(40))
        mine = {**old_config(40), "user_args": ["--resident-budget-gib", "40"]}
        self.write("strata-mine.json", mine)
        self.write("strata-iq3_s.json", {"exe": "x", "args": ["--max-context", "8192"], "args_written": ["--max-context", "8192"]})
        self.write("strata-unsloth-ud-q4_k_xl.shared-settings.json", {"temperature": 0.7})
        before = {p.name: p.read_bytes() for p in self.dir.glob("*.json")}
        done = self.mod.run([self.dir], setup_module=setup, say=lambda t: None)
        self.assertEqual(done, {str(self.dir): ["strata-unsloth-ud-q4_k_xl.json"]})
        moved = json.loads((self.dir / "strata-unsloth-ud-q4_k_xl.json").read_text())
        self.assertEqual((value(moved["args"], "--resident-budget-gib"), moved[KEY]), ("47", 40))
        self.assertEqual((self.dir / "strata-unsloth-ud-q4_k_xl.json.bak").read_bytes(), before["strata-unsloth-ud-q4_k_xl.json"])
        for name in ("strata-mine.json", "strata-iq3_s.json", "strata-unsloth-ud-q4_k_xl.shared-settings.json"):
            self.assertEqual((self.dir / name).read_bytes(), before[name], name)                # byte-identical
        self.assertTrue((self.dir / self.mod.MARK).is_file())
        after = (self.dir / "strata-unsloth-ud-q4_k_xl.json").read_bytes()
        self.assertEqual(self.mod.run([self.dir], setup_module=setup, say=lambda t: None), {})   # a second run looks at nothing
        self.assertEqual((self.dir / "strata-unsloth-ud-q4_k_xl.json").read_bytes(), after)

    def test_even_without_the_mark_a_moved_config_is_not_moved_again(self):
        self.write("strata-unsloth-ud-q4_k_xl.json", old_config(40))
        self.mod.run([self.dir], setup_module=setup, say=lambda t: None)
        (self.dir / self.mod.MARK).unlink()
        after = (self.dir / "strata-unsloth-ud-q4_k_xl.json").read_bytes()
        self.assertEqual(self.mod.run([self.dir], setup_module=setup, say=lambda t: None), {str(self.dir): []})
        self.assertEqual((self.dir / "strata-unsloth-ud-q4_k_xl.json").read_bytes(), after)

    def test_a_folder_that_is_not_there_is_skipped(self):
        self.assertEqual(self.mod.run([self.dir / "nothing"], setup_module=setup), {})


class TerminalStart(unittest.TestCase):
    """A Terminal start (strata launch, strata serve, Docker) of a config the app has loaded in "Leave 20% for my work": the app saves setup's number, not its own swap."""

    def test_the_engine_gets_setup_number_whatever_the_app_saved(self):
        saved = old_config(47, ui_budget_work_gib=40, ui_resource_profile="work", ui_resource_baseline={"workers": 6, "reserve": 0, "ramBudget": False, "ramHeadroom": False, "ramKeep": False})
        saved["exe"] = "strata"
        with contextlib.redirect_stdout(io.StringIO()):
            args = engine_args(saved)
        self.assertEqual(value(args, "--resident-budget-gib"), "47")
        self.assertEqual(args.count("--resident-budget-gib"), 1)


class Written(unittest.TestCase):
    """The run config setup writes (setup.main() on a mocked PC)."""
    found = PROFILES["64GB-1x32GB"][1]

    def run_setup(self, ram, *extra, configs=(), model=M):
        return install(ram, self.found, ["--family", "unsloth", "--model", model, "--no-start", "--context", "8192", *extra], configs=configs)

    def test_each_ram_size(self):
        for label, new, work in ((32, 15, 8), (48, 31, 24), (64, 47, 40)):
            code, out, cfg, _ = self.run_setup(label - 0.3)
            self.assertEqual(code, 0, out[-2000:])
            self.assertEqual(value(cfg["args"], "--resident-budget-gib"), str(new), label)
            self.assertEqual(cfg[KEY], work, label)
        for label in (96, 128):       # every expert in RAM: no budget to swap
            code, out, cfg, _ = self.run_setup(label - 0.3)
            self.assertEqual(code, 0, out[-2000:])
            self.assertNotIn("--resident-budget-gib", cfg["args"], label)
            self.assertNotIn(KEY, cfg, label)

    def test_the_budget_mode_at_the_cap_has_one_number(self):
        with mock.patch.object(setup, "unsloth_budget", lambda *a, **k: True):          # the budget mode on a 96 GB PC: 71 either way
            code, out, cfg, _ = self.run_setup(95.7)
        self.assertEqual(value(cfg["args"], "--resident-budget-gib"), "71")
        self.assertNotIn(KEY, cfg)

    def test_a_streamed_kv_cache_comes_out_of_both(self):
        code, out, cfg, _ = install(63.7, self.found, ["--family", "unsloth", "--model", M, "--no-start", "--context", "131072"])
        self.assertEqual(code, 0, out[-2000:])
        self.assertIn("--kv-resident", cfg["args"])
        self.assertEqual((value(cfg["args"], "--resident-budget-gib"), cfg[KEY]), ("45", 38))

    def test_a_budget_the_person_asked_for_has_no_second_number(self):
        for asked in ("52", "40", "30"):
            code, out, cfg, _ = self.run_setup(63.7, "--resident-budget-gib", asked)
            self.assertEqual(code, 0, out[-2000:])
            self.assertEqual(value(cfg["args"], "--resident-budget-gib"), asked)
            self.assertNotIn(KEY, cfg, asked)

    def test_a_setup_run_again_over_a_0216_config(self):
        code, out, first, _ = self.run_setup(63.7)
        old = [*first["args"]]
        old[old.index("--resident-budget-gib") + 1] = "40"                              # what 0.2.16 wrote: setup's 40, its copy, no second number
        old_cfg = {k: v for k, v in first.items() if k != KEY}
        code, out, cfg, _ = self.run_setup(63.7, configs=[(NAME, {**old_cfg, "args": old, "args_written": old})])
        self.assertEqual(code, 0, out[-2000:])
        self.assertEqual((value(cfg["args"], "--resident-budget-gib"), cfg[KEY]), ("47", 40))
        code, out, again, _ = self.run_setup(63.7, configs=[(NAME, cfg)])               # and once more: the same
        self.assertEqual((value(again["args"], "--resident-budget-gib"), again[KEY]), ("47", 40))

    def test_a_setup_run_again_over_a_config_the_app_put_in_work_mode(self):
        code, out, first, _ = self.run_setup(63.7)
        swapped = [*first["args"]]
        swapped[swapped.index("--resident-budget-gib") + 1] = "40"                       # the app wrote its work budget and recorded it as written
        code, out, cfg, _ = self.run_setup(63.7, configs=[(NAME, {**first, "args": swapped, "args_written": swapped, "ui_resource_profile": "work",
                                                                   "ui_resource_baseline": {"workers": 6, "reserve": 0, "setupBudget": 47}})])
        self.assertEqual(code, 0, out[-2000:])
        self.assertEqual((value(cfg["args"], "--resident-budget-gib"), cfg[KEY]), ("47", 40))
        self.assertNotIn("user_args", cfg)                                                # the app's swap is not taken for the person's

    def test_a_budget_the_person_set_stays_and_leaves_no_second_number(self):
        code, out, first, _ = self.run_setup(63.7)
        written = [*first["args"]]
        written[written.index("--resident-budget-gib") + 1] = "40"
        mine = [*written]
        mine[mine.index("--resident-budget-gib") + 1] = "36"
        for label, old in (("in user_args", {**first, "args": mine, "args_written": written, "user_args": ["--resident-budget-gib", "36"], KEY: 40}),
                           ("a hand edit of args", {**first, "args": mine, "args_written": written, KEY: 40})):
            code, out, cfg, _ = self.run_setup(63.7, configs=[(NAME, old)])
            self.assertEqual(code, 0, out[-2000:])
            self.assertEqual(value(cfg["args"], "--resident-budget-gib"), "36", label)
            self.assertNotIn(KEY, cfg, label)

    def test_a_layer_split_has_no_budget_and_no_second_number(self):
        cfg = {"args": ["--pack", "p", "--resident-budget-gib", "47", "--kv", "int8"], KEY: 40, "gpu": [0, 1]}
        with mock.patch.object(setup, "resident_split", lambda: False), mock.patch.object(setup, "ram_gb", lambda: 165.0):
            self.assertTrue(setup.split_budget(cfg, yes=True, explicit=True))
        self.assertNotIn("--resident-budget-gib", cfg["args"])
        self.assertNotIn(KEY, cfg)


class UnifiedMemory(Base):
    """A unified-memory GPU keeps 24 for the budget in every mode: there is no second number."""
    HALO = [{"index": 0, "name": "AMD Radeon 8060S", "vram_gb": 114.0, "arch": "gfx1151", "driver": "amdgpu",
             "uma": True, "dedicated_gb": 2.0, "shared_gb": 112.0}]

    def test_one_number(self):
        with mock.patch.object(setup, "get_prebuilt_hip", lambda *a, **k: self.t / "engine"), mock.patch.object(setup, "WIN", False), \
                mock.patch.object(setup, "build_engine_hip", lambda *a, **k: self.t / "engine"):
            code, out, cfg = self.main(["--context", "8192", "--backend", "hip"], version="0.1.41", amd=self.HALO, ram=63.7, m="UD-IQ4_XS")
        self.assertEqual(code, 0, out)
        self.assertEqual(value(cfg["args"], "--resident-budget-gib"), "40")
        self.assertNotIn(KEY, cfg)


if __name__ == "__main__":
    unittest.main()
