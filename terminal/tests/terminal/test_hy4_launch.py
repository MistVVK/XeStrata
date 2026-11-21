"""`strata launch` for Hy4-Flash (strata_cli/launch/runner.py): its engine is the component strata-hy4 (shape "hy4"), and the Flash-Next steps (drafter fetch, tuning, slots, the
split engine, KV format, picture reader) leave it alone."""

from __future__ import annotations

import unittest
from pathlib import Path
from unittest import mock

from strata_cli.launch import runner as R
from strata_cli.launch import settings as S


def runner_for(family: str | None, **kw) -> R.Runner:
    r = object.__new__(R.Runner)
    r.s = S.LaunchSettings(family=family, **kw)
    return r


BASE = {"exe": "component:strata-hy4", "arch": "hy4flash", "model_name": "hy4-flash-sq-64", "port": 8095, "drafting": "off",
        "args": ["--native", "C:/m/Hy4-Flash-SQ-64.strata", "--engram", "C:/m/Hy4-Flash-Engram-Q8.strata", "--max-context", "32768"]}


class Which(unittest.TestCase):
    def test_the_shape_is_hy4_only_for_hy4_flash(self):
        self.assertEqual(runner_for("hy4-flash").model_shape(), "hy4")
        self.assertTrue(runner_for("hy4-flash").is_hy4())
        self.assertFalse(runner_for("qwen").is_hy4())
        with mock.patch.object(R.Runner, "catalog", create=True) as cat:
            cat.strata.setup_module.return_value = mock.Mock(FAMILIES={"qwen": {}}, HY4_FAMILIES={"hy4-flash": {"title": "Tencent Hy4-Flash"}})
            self.assertIsNone(runner_for("qwen").model_shape())
            self.assertEqual(runner_for("hy4-flash").model_title(), "Tencent Hy4-Flash")


class Steps(unittest.TestCase):
    def test_the_flash_next_steps_are_skipped(self):
        r = runner_for("hy4-flash")
        self.assertIsNone(r.ensure_drafter(Path("x.json")))
        self.assertEqual(r.fetch_missing_drafter(Path("x.json")), Path("x.json"))
        with mock.patch.object(R.Runner, "stop_own_server_for_tuning") as stop, mock.patch.object(R.Runner, "emit"):
            r.tune()
            stop.assert_not_called()
        cfg = {"args": []}
        r.apply_four_card(cfg)
        self.assertIsNone(r.four_card_parallel)

    def test_the_setup_arguments_carry_no_flash_next_flags(self):
        r = runner_for("hy4-flash", model="SQ-64", context=32768, kv="int8", vision="on")
        r._catalog = mock.Mock()
        r._catalog.resolve.side_effect = lambda model, fam: (fam, model)
        r.feed = None
        r.move_data = False
        r.existing_data_dir = lambda: None
        r.installed_card_args = lambda: []
        args = r.setup_args()
        self.assertEqual(args[args.index("--family") + 1], "hy4-flash")
        self.assertNotIn(args[args.index("--vision") + 1], ("no", "none"))        # images are on by default (owner, 2026-11-19): setup downloads the Vision shard
        for flag in ("--kv", "--mtp", "--drafter", "--draft-vocab"):
            self.assertNotIn(flag, args)


class Config(unittest.TestCase):
    def test_apply_choices_changes_only_the_context(self):
        s = S.LaunchSettings(family="hy4-flash", context=65536, kv="int8", vision="on")
        out = R.apply_choices(BASE, s)
        self.assertEqual(out["args"][out["args"].index("--max-context") + 1], "65536")
        self.assertNotIn("--kv", out["args"])
        self.assertNotIn("--vision", out["args"])
        self.assertEqual(BASE["args"][-1], "32768")                          # the input is not touched

    def test_vision_off_takes_the_vision_file_out_of_the_command(self):
        cfg = {**BASE, "args": BASE["args"] + ["--vision", "C:/m/Hy4-Flash-Vision.strata"]}
        on = R.apply_choices(cfg, S.LaunchSettings(family="hy4-flash", vision="on"))
        self.assertIn("--vision", on["args"])
        off = R.apply_choices(cfg, S.LaunchSettings(family="hy4-flash", vision="off"))
        self.assertNotIn("--vision", off["args"])
        self.assertNotIn("C:/m/Hy4-Flash-Vision.strata", off["args"])

    def test_missing_strata_files_are_named(self):
        cfg = {"args": ["--native", "C:/does/not/exist/a.strata", "--engram", "C:/does/not/exist/e.strata", "--max-context", "4096"]}
        self.assertEqual(R.files_present(cfg), ["C:/does/not/exist/a.strata", "C:/does/not/exist/e.strata"])

    def test_derive_config_keeps_the_engine_and_its_arguments(self):
        s = S.LaunchSettings(family="hy4-flash", port=8123)
        from strata_control import tune_knobs
        cfg = R.derive_config(BASE, s, bool_flags=tune_knobs.BOOL_FLAGS, log="x.log")
        self.assertEqual(cfg["exe"], "component:strata-hy4")
        self.assertEqual(cfg["arch"], "hy4flash")
        self.assertEqual(cfg["port"], 8123)
        for flag in ("--native", "--engram", "--max-context"):
            self.assertIn(flag, cfg["args"])
        for flag in ("--vision", "--mtp", "--batch", "--draft", "--vram-reserve-mib"):
            self.assertNotIn(flag, cfg["args"])


class CatalogAndHarness(unittest.TestCase):
    def test_catalog_resolve_and_names_include_hy4_flash(self):
        # P0 #2 regression: strata launch --plan --family hy4-flash --model SQ-64 must resolve SQ-64 cleanly
        import tempfile
        from strata_cli.launch.catalog import Catalog
        with tempfile.TemporaryDirectory() as d:
            cat = Catalog(src=Path(__file__).resolve().parents[2], root=Path(d))
            self.assertEqual(cat.resolve("SQ-64", "hy4-flash"), ("hy4-flash", "SQ-64"))
            self.assertEqual(cat.resolve("sq-64"), ("hy4-flash", "SQ-64"))
            self.assertEqual(cat.model_name_of("hy4-flash", "SQ-64"), "hy4-flash-sq-64")
            self.assertEqual(cat.tag_of("hy4-flash", "SQ-64"), "hy4-flash-sq-64")
            self.assertEqual(cat.model_folder(Path(d), "hy4-flash", "SQ-64"), Path(d) / "models" / "hy4-flash")

    def test_harness_display_name_uses_hy4_flash_title(self):
        from strata_cli.launch import harness
        t_hy4 = harness.Target(url="http://127.0.0.1:8095", key="k", model="hy4-flash-sq-64", context=131072)
        self.assertIn("name: Tencent Hy4-Flash (Strata, local)", harness.omp_models(t_hy4))
        self.assertIn("Tencent Hy4-Flash (Strata, local)", harness.pi_models(t_hy4))
        t_qwen = harness.Target(url="http://127.0.0.1:8095", key="k", model="qwen3.8-flash-next-iq3_s", context=32768)
        self.assertIn("name: Qwen3.8-Flash-Next (Strata, local)", harness.omp_models(t_qwen))


if __name__ == "__main__":
    unittest.main()
