"""Human TUI, CLI and harness UX tests for Tencent Hy4-Flash (0.2.18):
- 6-step launch flow (Model, Folder, Context, Pictures, Access, Save) at 110x40 and 80x24 across SQ-64/SQ-96/SQ-128,
  including simulated Linux Intel Arc (sycl) and macOS (metal) without Qwen's strata-vision sidecar,
- Qwen3.8-Flash-Next regression walk at 110x40 and 80x24,
- Partial-folder & missing Hy4-Flash-Vision.strata detection (setup.hy4_have, modelstate, runner, describe_config),
- Friendly 1-line CLI answers for `strata tune auto` / `strata engine calibrate` / `strata drafting plan|fetch`,
- `strata connect` configs for OpenCode, omp and pi with Hy4-Flash display name and [high, max] thinking efforts.
"""

from __future__ import annotations

import asyncio
import json
import sys
import unittest
from pathlib import Path
from unittest import mock

from strata_cli.launch import access, harness, modelstate, runner as R, screens, settings as S, store
from strata_cli.launch.app import LaunchApp
from strata_control import hy4_family
from tests.terminal.launch_support import HW_12, HW_24, HW_2X16, Home, catalog, run_cli, text_of
from tests.terminal.test_launch_tui import FakeRunner, flat, make_ctx

HW_ARC_LINUX = {
    "ram_gb": 128.0,
    "os": "Linux",
    "disk_free_gb": 5000.0,
    "cpu": {"avx2": True, "name": "test cpu"},
    "gpus": [{"index": 0, "name": "Intel Arc A770", "vram_gb": 16.0, "usable": True, "vendor": "intel"}],
}
HW_MAC_METAL = {
    "ram_gb": 128.0,
    "mac": True,
    "os": "macOS",
    "disk_free_gb": 5000.0,
    "cpu": {"avx2": False, "name": "Apple M3 Max"},
    "gpus": [{"index": 0, "name": "Apple M3 Max", "vram_gb": 128.0, "unified_gb": 128.0, "usable": True, "vendor": "apple"}],
}


async def settle(pilot, n: int = 3) -> None:
    await pilot.pause(0.05 * n)


class Hy4TuiWalkTests(unittest.TestCase):
    def setUp(self) -> None:
        self.home = Home().__enter__()
        self.addCleanup(self.home.__exit__, None, None, None)
        FakeRunner.instances.clear()
        FakeRunner.behaviour = "ok"
        FakeRunner.download = 0.0
        for p in (
            mock.patch("strata_cli.launch.app.Runner", FakeRunner),
            mock.patch.object(access, "lan_ips", return_value=["192.168.1.5"]),
        ):
            p.start()
            self.addCleanup(p.stop)

    def drive(self, coro, hw=HW_24, size=(110, 40)):
        ctx = make_ctx(hw)
        app = LaunchApp(ctx)

        async def go():
            async with app.run_test(size=size) as pilot:
                await settle(pilot, 3)
                return await coro(app, pilot, ctx)

        return asyncio.run(go())

    def test_flow_for_omits_qwen_only_screens_for_hy4_and_keeps_qwen_intact(self) -> None:
        for hw in (HW_12, HW_24, HW_2X16, HW_ARC_LINUX, HW_MAC_METAL):
            ctx = make_ctx(hw)
            self.assertEqual(
                screens.flow_for(ctx, "hy4-flash"),
                ["model", "folder", "context", "pictures", "access", "save"],
            )
            qwen_flow = screens.flow_for(ctx, "qwen")
            for must_have in ("parallel", "tuning", "advanced"):
                self.assertIn(must_have, qwen_flow)
            if hw is HW_2X16:
                self.assertIn("cards", qwen_flow)

    def test_keystroke_walk_hy4_builds_at_110x40_and_80x24(self) -> None:
        for build in ("SQ-64", "SQ-96", "SQ-128"):
            for size in ((110, 40), (80, 24)):
                with self.subTest(build=build, size=size):
                    async def case(app, pilot, ctx):
                        app.screen.menu().select(("new", None))
                        await pilot.press("enter")
                        await settle(pilot)
                        self.assertEqual(app.current, "model")
                        app.screen.menu().select(("hy4-flash", build))
                        app.screen.paint()
                        await settle(pilot, 2)
                        head = flat(app)
                        self.assertIn("Tencent Hy4-Flash", head)
                        self.assertIn("Step 1 of 6: Model", head)

                        # Step 1 -> Step 2 (Folder)
                        await pilot.press("enter")
                        await settle(pilot)
                        self.assertEqual(app.current, "folder")
                        self.assertIn("Step 2 of 6: Folder", flat(app))

                        # Step 2 -> Step 3 (Context)
                        await pilot.press("enter")
                        await settle(pilot)
                        self.assertEqual(app.current, "context")
                        ctx_text = flat(app)
                        self.assertIn("Step 3 of 6: Context", ctx_text)
                        self.assertIn("128K", ctx_text)
                        self.assertIn("managed automatically by Hy4-Flash", ctx_text)
                        self.assertNotIn("press m to change", ctx_text)
                        # Pressing 'm' does not inject Qwen KV formats
                        await pilot.press("m")
                        await settle(pilot, 1)
                        self.assertIsNone(ctx.settings.kv)

                        # Step 3 -> Step 4 (Pictures)
                        await pilot.press("enter")
                        await settle(pilot)
                        self.assertEqual(app.current, "pictures")
                        pic_text = flat(app)
                        self.assertIn("Step 4 of 6: Pictures", pic_text)
                        self.assertIn("Yes, read pictures", pic_text)
                        self.assertNotIn("no picture reader", pic_text)
                        self.assertNotIn("not available on Intel Arc", pic_text)
                        self.assertEqual(app.screen.menu().current.key, "on")
                        if size[1] >= 30:
                            self.assertIn("Hy4-Flash-Vision.strata", pic_text)

                        # Step 4 -> Step 5 (Access) - skips Drafting, Parallel, Tuning, Advanced!
                        await pilot.press("enter")
                        await settle(pilot)
                        self.assertEqual(app.current, "access")
                        self.assertEqual(ctx.settings.vision, "on")
                        self.assertIn("Step 5 of 6: Access", flat(app))

                        # Step 5 -> Step 6 (Save)
                        await pilot.press("enter")
                        await settle(pilot)
                        self.assertEqual(app.current, "save")
                        save_text = flat(app)
                        self.assertIn("Step 6 of 6: Save", save_text)
                        self.assertIn(f"Tencent Hy4-Flash {build}", save_text)
                        self.assertIn("Pictures on", save_text)
                        self.assertIn("DFly2", save_text)
                        self.assertNotIn("Parallel", save_text)
                        self.assertNotIn("Tuning", save_text)
                        self.assertNotIn("memory int8", save_text)
                        self.assertNotIn("It tunes first.", save_text)

                        # Escape all the way back to Start cleanly
                        for expected in ("access", "pictures", "context", "folder", "model", "start"):
                            await pilot.press("escape")
                            await settle(pilot)
                            self.assertEqual(app.current, expected)

                    self.drive(case, hw=HW_2X16, size=size)

    def test_simulated_intel_arc_and_macos_without_strata_vision_keep_hy4_pictures_on(self) -> None:
        # Simulate an install where Qwen's strata-vision is missing and Linux backend is SYCL (Intel Arc) or Metal (macOS)
        for label, hw, plat, backend in (
            ("arc_sycl", HW_ARC_LINUX, "linux", "sycl"),
            ("mac_metal", HW_MAC_METAL, "darwin", "metal"),
        ):
            for size in ((110, 40), (80, 24)):
                with self.subTest(env=label, size=size):
                    with (
                        mock.patch.object(sys, "platform", plat),
                        mock.patch("strata_cli.launch.catalog.Catalog.engine_backend", return_value=backend),
                        mock.patch(
                            "strata_cli.launch.catalog.Catalog.reader_state",
                            autospec=True,
                            side_effect=lambda self_cat, family=None: "present" if family == "hy4-flash" else "missing",
                        ),
                    ):
                        cat = catalog(hw)
                        # Qwen disables pictures on SYCL without reader, while Hy4-Flash keeps native pictures ON!
                        if backend == "sycl":
                            self.assertTrue(cat.pictures_not_on_card("qwen"))
                        self.assertFalse(cat.pictures_not_on_card("hy4-flash"))
                        self.assertEqual(cat.picture_default(True, "hy4-flash"), "on")

                        async def case(app, pilot, ctx):
                            app.screen.menu().select(("new", None))
                            await pilot.press("enter")
                            await settle(pilot)
                            app.screen.menu().select(("hy4-flash", "SQ-64"))
                            await pilot.press("enter")  # -> folder
                            await settle(pilot)
                            await pilot.press("enter")  # -> context
                            await settle(pilot)
                            await pilot.press("enter")  # -> pictures
                            await settle(pilot)
                            self.assertEqual(app.current, "pictures")
                            pic_text = flat(app)
                            self.assertEqual(app.screen.menu().current.key, "on")
                            self.assertTrue(app.screen.menu().items[1].enabled)
                            self.assertNotIn("not available on Intel Arc", pic_text)
                            self.assertNotIn("does not carry the picture reader", pic_text)
                            self.assertNotIn("no picture reader", pic_text)
                            await pilot.press("enter")  # -> access
                            await settle(pilot)
                            self.assertEqual(ctx.settings.vision, "on")

                        self.drive(case, hw=hw, size=size)

    def test_qwen_walk_at_110x40_and_80x24_keeps_parallel_and_tuning(self) -> None:
        for size in ((110, 40), (80, 24)):
            with self.subTest(size=size):
                async def case(app, pilot, ctx):
                    app.screen.menu().select(("new", None))
                    await pilot.press("enter")
                    await settle(pilot)
                    app.screen.menu().select(("qwen", "IQ3_XXS"))
                    for _ in range(15):
                        if app.current == "save":
                            break
                        await pilot.press("n" if app.current == "advanced" else "enter")
                        await settle(pilot)
                    self.assertEqual(app.current, "save")
                    save_text = flat(app)
                    self.assertIn("Parallel", save_text)
                    self.assertIn("Tuning", save_text)
                    self.assertIn("memory int8", save_text)

                self.drive(case, hw=HW_12, size=size)

    def test_saved_hy4_preset_on_start_screen_and_cli_list(self) -> None:
        store.save(S.LaunchSettings(family="hy4-flash", model="SQ-64", context=131072, vision="on"), "Hy4 Daily")

        async def case(app, pilot, ctx):
            start_text = flat(app)
            self.assertIn("Hy4 Daily", start_text)
            self.assertIn("1 request", start_text)
            self.assertIn("built-in", start_text)
            self.assertNotIn("fit to card", start_text)
            self.assertNotIn("not tuned", start_text)

        self.drive(case, hw=HW_24, size=(110, 40))
        code, out, err = run_cli(["launch", "--list-presets"])
        self.assertEqual(code, 0, err)
        self.assertIn("1 request", out)
        self.assertIn("built-in", out)
        self.assertNotIn("fit to card", out)


class PartialFolderAndVisionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.home = Home().__enter__()
        self.addCleanup(self.home.__exit__, None, None, None)

    def test_partial_hy4_folder_without_done_marks_only_requests_missing_vision_shard(self) -> None:
        cat = catalog(HW_24)
        mod = cat.strata.setup_module()
        data_dir = self.home.base / "data"
        hy4_dir = data_dir / "models" / "hy4-flash"
        hy4_dir.mkdir(parents=True)

        sq_path = hy4_dir / "Hy4-Flash-SQ-64.strata"
        eng_path = hy4_dir / "Hy4-Flash-Engram-Q8.strata"
        vis_path = hy4_dir / "Hy4-Flash-Vision.strata"

        # Create files of exact pinned sizes WITHOUT .done markers; leave Vision missing
        fake_files = {
            "Hy4-Flash-SQ-64.strata": (1024, hy4_family.FILES["Hy4-Flash-SQ-64.strata"][1]),
            "Hy4-Flash-Engram-Q8.strata": (512, hy4_family.FILES["Hy4-Flash-Engram-Q8.strata"][1]),
            "Hy4-Flash-Vision.strata": hy4_family.FILES["Hy4-Flash-Vision.strata"],
        }
        sq_path.write_bytes(b"\x00" * 1024)
        eng_path.write_bytes(b"\x00" * 512)

        ns = type("Args", (), {"gguf_dir": None})()
        with mock.patch.dict(hy4_family.FILES, fake_files):
            self.assertTrue(mod.hy4_have(ns, sq_path))
            self.assertTrue(mod.hy4_have(ns, eng_path))
            self.assertFalse(mod.hy4_have(ns, vis_path))

            st = modelstate.assess(cat, "hy4-flash", "SQ-64", vision="on", data_dir=data_dir, use_installed=False)
            self.assertTrue(st.only_helpers)
            self.assertEqual([(p.kind, p.label, p.gb) for p in st.parts], [("mmproj", "the picture reader", 0.93)])

            st_off = modelstate.assess(cat, "hy4-flash", "SQ-64", vision="off", data_dir=data_dir, use_installed=False)
            self.assertEqual(st_off.parts, [])

    def test_runner_vision_word_and_describe_config_with_vision_shard(self) -> None:
        cat = catalog(HW_24)
        for v in ("on", "yes", "gpu", "cpu"):
            r = R.Runner(S.LaunchSettings(family="hy4-flash", model="SQ-64", vision=v), catalog=cat)
            self.assertEqual(r.vision_word(), "yes", v)
        for v in ("off", "no"):
            r = R.Runner(S.LaunchSettings(family="hy4-flash", model="SQ-64", vision=v), catalog=cat)
            self.assertEqual(r.vision_word(), "no", v)

        # Installed config without --vision -> desc_matches is False when user turns Pictures ON
        cfg_dir = Path(self.home.env["STRATA_INSTALL_DIR"])
        cfg_dir.mkdir(parents=True, exist_ok=True)
        m_dir = self.home.base / "m"
        m_dir.mkdir(parents=True, exist_ok=True)
        for n in ("Hy4-Flash-SQ-64.strata", "Hy4-Flash-Engram-Q8.strata", "Hy4-Flash-Vision.strata"):
            (m_dir / n).write_bytes(b"x")
        cfg_path = cfg_dir / "strata-hy4-flash-sq-64.json"
        base_cfg = {
            "exe": "component:strata-hy4",
            "arch": "hy4flash",
            "model_name": "hy4-flash-sq-64",
            "args": ["--native", str(m_dir / "Hy4-Flash-SQ-64.strata"), "--engram", str(m_dir / "Hy4-Flash-Engram-Q8.strata"), "--max-context", "131072"],
        }
        cfg_path.write_text(json.dumps(base_cfg), encoding="utf-8")
        desc_no_vis = cat.strata.describe_config(cfg_path)
        self.assertEqual(desc_no_vis["images"], "off")
        self.assertFalse(R.desc_matches(desc_no_vis, S.LaunchSettings(family="hy4-flash", model="SQ-64", context=131072, vision="on")))

        # Installed config WITH --vision -> describe_config says "on", desc_matches is True, and check_pictures emits NO false warning
        vis_cfg = {**base_cfg, "args": [*base_cfg["args"], "--vision", str(m_dir / "Hy4-Flash-Vision.strata")]}
        cfg_path.write_text(json.dumps(vis_cfg), encoding="utf-8")
        desc_vis = cat.strata.describe_config(cfg_path)
        self.assertEqual(desc_vis["images"], "on")
        for v in ("on", "yes", "gpu", "cpu"):
            self.assertTrue(R.desc_matches(desc_vis, S.LaunchSettings(family="hy4-flash", model="SQ-64", context=131072, vision=v)), v)
        r = R.Runner(S.LaunchSettings(family="hy4-flash", model="SQ-64", context=131072, vision="on"), catalog=cat)
        r.check_pictures(cfg_path)
        self.assertEqual(r.notices, [])


class CliAndHarnessUxTests(unittest.TestCase):
    def setUp(self) -> None:
        self.home = Home().__enter__()
        self.addCleanup(self.home.__exit__, None, None, None)

    def test_strata_tune_and_calibrate_give_clean_one_line_hy4_explanation(self) -> None:
        # Uninstalled Hy4 target name
        code, out, err = run_cli(["tune", "auto", "hy4-flash-sq-64"])
        self.assertEqual(code, 2)
        self.assertEqual(len(err.strip().splitlines()), 1)
        self.assertIn("Tencent Hy4-Flash needs no tuning", err)
        self.assertNotIn("Traceback", err)

        # Installed Hy4 config file
        cfg = self.home.base / "strata-hy4-flash-sq-64.json"
        cfg.write_text(json.dumps({"exe": "component:strata-hy4", "arch": "hy4flash", "model_name": "hy4-flash-sq-64", "args": []}), encoding="utf-8")
        code, out, err = run_cli(["tune", "auto", str(cfg)])
        self.assertEqual(code, 2)
        self.assertEqual(len(err.strip().splitlines()), 1)
        self.assertIn("Tencent Hy4-Flash needs no tuning", err)

        code_j, out_j, _ = run_cli(["tune", "auto", str(cfg), "--output", "json"])
        self.assertEqual(code_j, 2)
        self.assertFalse(json.loads(out_j)["ok"])

    def test_strata_drafting_gives_clean_one_line_hy4_explanation_and_fetches_nothing(self) -> None:
        cfg = self.home.base / "strata-hy4-flash-sq-64.json"
        cfg.write_text(json.dumps({"exe": "component:strata-hy4", "arch": "hy4flash", "model_name": "hy4-flash-sq-64", "args": []}), encoding="utf-8")
        with mock.patch("strata_control.drafting_fetch.ensure_for", side_effect=AssertionError("must never download Qwen DFlash2 for Hy4")):
            for argv in (
                ["drafting", "plan", "--model", "SQ-64"],
                ["drafting", "fetch", "--model", "SQ-64"],
                ["drafting", "plan", "--family", "hy4-flash", "--model", "SQ-96"],
                ["drafting", "fetch", "--config", str(cfg)],
            ):
                code, out, err = run_cli(argv)
                self.assertEqual(code, 0, (argv, err))
                self.assertEqual(len(out.strip().splitlines()), 1, (argv, out))
                self.assertIn("DFly2", out)

            code_jl, out_jl, err_jl = run_cli(["drafting", "plan", "--config", str(cfg), "--output", "jsonl"])
            self.assertEqual(code_jl, 0, err_jl)
            doc = json.loads(out_jl)
            self.assertTrue(doc["ok"])
            self.assertFalse(doc["needed"])
            self.assertFalse(doc["serves"])
            self.assertEqual(doc["gb"], 0.0)

            code_f, out_f, err_f = run_cli(["drafting", "fetch", "--config", str(cfg), "--if-needed", "--output", "jsonl"])
            self.assertEqual(code_f, 0, err_f)
            self.assertTrue(json.loads(out_f)["skipped"])

    def test_strata_connect_configs_for_hy4_and_qwen(self) -> None:
        t_hy4 = harness.Target(url="http://127.0.0.1:8095", key="k", model="hy4-flash-sq-64", context=131072)
        oc_hy4 = json.loads(harness.opencode_file(t_hy4))
        self.assertEqual(
            oc_hy4["provider"]["strata"]["models"]["hy4-flash-sq-64"]["name"],
            "Tencent Hy4-Flash (Strata, local)",
        )
        omp_m = harness.omp_models(t_hy4)
        self.assertIn("efforts: [high, max]", omp_m)
        self.assertIn("defaultLevel: high", omp_m)
        self.assertNotIn("minimal", omp_m)
        self.assertIn("defaultThinkingLevel: high", harness.omp_config(t_hy4))
        self.assertEqual(json.loads(harness.pi_settings(t_hy4))["defaultThinkingLevel"], "high")

        # Qwen keeps its tested levels
        t_qwen = harness.Target(url="http://127.0.0.1:8095", key="k", model="qwen3.8-flash-next-iq3_s", context=32768)
        oc_qwen = json.loads(harness.opencode_file(t_qwen))
        self.assertEqual(
            oc_qwen["provider"]["strata"]["models"]["qwen3.8-flash-next-iq3_s"]["name"],
            "Qwen3.8-Flash-Next (Strata, local)",
        )
        self.assertIn("efforts: [minimal, low, medium, high]", harness.omp_models(t_qwen))
        self.assertIn("defaultThinkingLevel: medium", harness.omp_config(t_qwen))
        self.assertEqual(json.loads(harness.pi_settings(t_qwen))["defaultThinkingLevel"], "medium")


if __name__ == "__main__":
    unittest.main()
