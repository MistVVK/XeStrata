"""Regression tests for Track 5A TUI/CLI fixes:
- PR #6451 extra path: Swift shard 1 `--ple-gguf` on re-launch without `--model-path` and on Mac (`mac_fit.engine_args`).
- PR #6512 + `strata connect` (all 6 harnesses: claude-code, codex, opencode, cline, pi, openclaw + `--dry-run`) + Issues #6459, #6460, #6463.
- Issue #6519: `[TUI] can't reset a setting with [Del]` on `AdvancedScreen`.
"""

from __future__ import annotations

import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from strata_cli.launch import app as launch_app
from strata_cli.launch import adopt, harness, params as P, runner, settings as S
from strata_control import mac_fit
from tests.terminal.launch_support import Home, run_cli


class SwiftPleShardPaths(unittest.TestCase):
    def test_mac_fit_engine_args_uses_shard_1_for_swift(self):
        shards = [Path("/m/Swift-IQ3_XXS-00001-of-00002.gguf"), Path("/m/Swift-IQ3_XXS-00002-of-00002.gguf")]
        swift_args = mac_fit.engine_args(Path("/p"), shards, Path("/rt"), 32768, family="swift")
        qwen_args = mac_fit.engine_args(Path("/p"), shards, Path("/rt"), 32768, family="qwen")
        self.assertEqual(swift_args[swift_args.index("--ple-gguf") + 1], str(shards[0]))
        self.assertEqual(qwen_args[qwen_args.index("--ple-gguf") + 1], str(shards[1]))

    def test_adopt_prepared_files_without_model_path_keeps_swift_shard_1_and_reuses_saved_folder(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src = root / "src"
            (src / "data").mkdir(parents=True)
            (src / "data" / "expert-profile.bin").write_bytes(b"p")
            data = root / "data"
            pack = data / "packs" / "swift-iq3_xxs"
            (pack / "tokenizer").mkdir(parents=True)
            (pack / "experts.bin").write_bytes(b"e")
            custom_dir = root / "custom-swift"
            custom_dir.mkdir()
            s1 = custom_dir / "Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf"
            s2 = custom_dir / "Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00002-of-00002.gguf"
            s1.write_bytes(b"1")
            s2.write_bytes(b"2")

            install_root = root / "install"
            install_root.mkdir()
            installed_json = install_root / "strata-swift-iq3_xxs.json"
            installed_json.write_text(json.dumps({"args": ["--native", str(s1), "--ple-gguf", str(s1)]}), encoding="utf-8")

            cat = mock.Mock()
            cat.src = src
            cat.root = install_root
            cat.data_dir.return_value = data
            cat.tag_of.return_value = "swift-iq3_xxs"
            cat.installed_config.return_value = installed_json
            cat.tables.return_value = ({}, {"swift": {"tag": "swift-", "name": "swift-1.5"}}, [], "test")
            cat.computer.return_value = mock.Mock(main_vram_gb=24.0)

            s = S.LaunchSettings(family="swift", model="IQ3_XXS", context=65536, model_path=None)
            r = runner.Runner(s, catalog=cat)
            with mock.patch("strata_control.engine_component.detect_backend", return_value=("sycl", "detected")), \
                    mock.patch.object(adopt, "intel_driver", return_value="xe"):
                out_path = r.adopt_prepared_files("", chosen=True)

            self.assertEqual(out_path, installed_json)
            cfg = json.loads(installed_json.read_text(encoding="utf-8"))
            args = cfg["args"]
            self.assertEqual(args[args.index("--native") + 1], str(s1))
            self.assertEqual(args[args.index("--ple-gguf") + 1], str(s1))
            self.assertEqual(args[args.index("--max-context") + 1], "65536")


class HarnessAndConnectTrack5A(unittest.TestCase):
    URL = "http://127.0.0.1:18900"

    def test_all_six_core_harnesses_present_and_render(self):
        for hid in ("claude-code", "codex", "opencode", "cline", "pi", "openclaw"):
            h = harness.find(hid)
            self.assertIsNotNone(h, hid)
            out = harness.render(h, harness.target(self.URL, "sk-test", "qwen3.8-flash-next-iq3_s", 131072))
            self.assertTrue(out["files"], hid)
            self.assertTrue(out["limits"], hid)

    def test_codex_disables_remote_compaction_6459(self):
        out = harness.render(harness.find("codex"), harness.target(self.URL, None, "strata", 131072))
        text = out["files"][0]["text"]
        self.assertIn("[model_providers.strata.capabilities]", text)
        self.assertIn('remote_compaction = "unsupported"', text)

    def test_claude_code_limits_cover_6460_and_6463(self):
        out = harness.render(harness.find("claude-code"), harness.target(self.URL, None, "strata", 131072))
        joined = "\n".join(out["limits"])
        self.assertIn('"anthropic_thinking": "on_request"', joined)
        self.assertIn("--conversation-cache-min-tokens", joined)

    def test_cline_renders_openai_compatible_json(self):
        out = harness.render(harness.find("cline"), harness.target(self.URL, "sk-cline", "strata", 131072))
        cfg = json.loads(out["files"][0]["text"])
        self.assertEqual(cfg["apiProvider"], "openai")
        self.assertEqual(cfg["openAiBaseUrl"], self.URL + "/v1")
        self.assertEqual(cfg["openAiModelId"], "strata")
        self.assertEqual(cfg["openAiModelInfo"]["contextWindow"], 131072)
        self.assertEqual(cfg["openAiModelInfo"]["maxTokens"], 16384)
        self.assertTrue(cfg["openAiModelInfo"]["supportsImages"])

    def test_connect_dry_run_writes_nothing_and_reports_planned_action(self):
        with Home() as home:
            code, out, err = run_cli(["connect", "cline", "--url", self.URL, "--context", "131072",
                                      "--write", "--dry-run", "--output", "json"])
            self.assertEqual(code, 0, err)
            data = json.loads(out)
            self.assertTrue(data["ok"])
            self.assertTrue(data["write"]["dry_run"])
            written = data["write"]["written"]
            self.assertEqual(len(written), 1)
            self.assertEqual(written[0]["action"], "would_create")
            self.assertFalse(Path(written[0]["path"]).exists(), "dry-run must not create any file")

            # Existing file: dry-run reports would_merge and leaves existing file untouched.
            target_path = Path(written[0]["path"])
            target_path.parent.mkdir(parents=True, exist_ok=True)
            original = '{"custom": true}\n'
            target_path.write_text(original, encoding="utf-8")
            code2, out2, err2 = run_cli(["connect", "cline", "--url", self.URL, "--context", "131072",
                                         "--write", "--dry-run", "--output", "json"])
            self.assertEqual(code2, 0, err2)
            data2 = json.loads(out2)
            self.assertEqual(data2["write"]["written"][0]["action"], "would_merge")
            self.assertIsNotNone(data2["write"]["written"][0]["backup"])
            self.assertEqual(target_path.read_text(encoding="utf-8"), original)
            self.assertFalse(Path(data2["write"]["written"][0]["backup"]).exists())

    def test_details_md_opencode_uses_providers_key(self):
        details = Path(__file__).resolve().parents[3] / "docs" / "DETAILS.md"
        text = details.read_text(encoding="utf-8")
        marker = "- **OpenCode:** a starting point for `opencode.json`."
        self.assertIn(marker, text)
        after = text.split(marker, 1)[1]
        block = after.split("```json\n", 1)[1].split("\n  ```", 1)[0]
        parsed = json.loads(block)
        self.assertIn("providers", parsed)
        self.assertNotIn("provider", parsed)


class AdvancedScreenDeleteReset6519(unittest.TestCase):
    def _make_ctx(self, extra: dict[str, str], base_args: list[str]):
        class FakeComp:
            usable_gpus = [{"index": 0, "name": "RTX 5090", "vram_gb": 32.0, "vendor": "nvidia", "usable": True}]
            gpus = usable_gpus
            main_vram_gb = 32.0
            ram_gb = 64.0
            raw = {}
            engine_vendor = "nvidia"
            other_vendor_note = None

            def line(self, width=None):
                return "RTX 5090"

        class FakeCat:
            def installed_config(self, tag):
                return None

            def tag_of(self, f, m):
                return "q2_0"

            def installed_backend(self):
                return "cuda"

            def data_dir(self):
                return Path(".")

            def disk_free(self, p):
                return 100.0

        s = S.LaunchSettings(family="qwen", model="Q2_0", context=32768, extra=dict(extra))
        ctx = launch_app.Ctx(catalog=FakeCat(), computer=FakeComp(), rows=[], settings=s)
        ctx._help = "  --prefill N      prefill tokens (default: auto)\n  --kv FMT         kv format (default: int8)\n"
        ctx._help_exe = "fake-exe"
        ctx.engine_exe = lambda: "fake-exe"
        ctx.base_config = lambda: {"args": list(base_args)}
        return ctx

    def test_del_resets_user_override_and_toggles_red_unknown_config_flag(self):
        ctx = self._make_ctx(
            extra={"--prefill": "65536", "--bogus-extra": "1"},
            base_args=["--kv", "int8", "--legacy-red-flag", "42"],
        )

        async def go():
            app = launch_app.LaunchApp(ctx)
            async with app.run_test(size=(120, 35)) as pilot:
                app.goto("advanced")
                await pilot.pause(0.1)
                scr = app.screen

                def select_key(key: str):
                    m = scr.menu()
                    for i, it in enumerate(m.items):
                        if it.key == key:
                            m.index = i
                            m._refresh()
                            return
                    raise AssertionError(f"{key} not found in menu")

                # 1. Red-marked extra override -> Del removes it from s.extra.
                select_key("--bogus-extra")
                await pilot.press("delete")
                await pilot.pause(0.05)
                self.assertNotIn("--bogus-extra", ctx.settings.extra)
                self.assertIn("back to the tuned value", scr.message)

                # 2. Known user override -> Del resets it back to tuned/default.
                select_key("--prefill")
                await pilot.press("delete")
                await pilot.pause(0.05)
                self.assertNotIn("--prefill", ctx.settings.extra)

                # 3. Red-marked unknown flag coming from base_args (not yet in s.extra):
                # Del turns it off in s.extra so it is removed from the launch argv; Del again restores it.
                select_key("--legacy-red-flag")
                row = next(r for r in scr.rows if r.name == "--legacy-red-flag")
                self.assertFalse(row.known)
                await pilot.press("delete")
                await pilot.pause(0.05)
                self.assertEqual(ctx.settings.extra.get("--legacy-red-flag"), "off")
                self.assertIn("turned off", scr.message)

                # Second Del puts it back.
                select_key("--legacy-red-flag")
                await pilot.press("delete")
                await pilot.pause(0.05)
                self.assertNotIn("--legacy-red-flag", ctx.settings.extra)

                # 4. Known setting already at default -> Del gives explicit feedback instead of silence.
                select_key("--kv")
                await pilot.press("delete")
                await pilot.pause(0.05)
                self.assertIn("already at its", scr.message)

        asyncio.run(go())

    def test_boolean_extra_flag_displays_on_in_yours_column(self):
        rows = P.build_rows("  --mmap-experts   mmap experts\n", [], None, None, None, {"--mmap-experts": ""})
        r = next(x for x in rows if x.name == "--mmap-experts")
        self.assertEqual(r.yours, "on")


if __name__ == "__main__":
    unittest.main()
