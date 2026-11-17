"""0.2.3 RC stress test, #3 (H01): the version bump is complete and consistent, and a 0.2.2 install sees the update."""

import json
import os
import re
import subprocess
import sys
import unittest
from pathlib import Path

import strata_cli
from strata_control import feed as feedmod, updater
from tests.terminal import test_updater

ROOT = Path(__file__).resolve().parents[2]


def first(path, pattern):
    return re.search(pattern, (ROOT / path).read_text(encoding="utf-8"), re.MULTILINE).group(1)


class VersionBump(unittest.TestCase):
    def test_the_terminal_is_0_2_4_2_and_control_is_0_1_49_2_while_the_app_stays_0_2_3(self):
        # 0.2.13 (owner 2026-11-08, release of main before Qwen4): Terminal and app are 0.2.13; Control and every engine package (Qwen, DeepSeek, Metal, SYCL, aarch64) are 0.1.59 (the CMake project version).
        # 0.2.12 (owner 2026-11-06, release of rounds 0215/0216/0217): Terminal and app are 0.2.12; Control and every engine package (Qwen, DeepSeek, Metal, SYCL, aarch64) are 0.1.58 (the CMake project version).
        # 0.2.11 (coordinator 2026-11-04; the owner can still rename it 0.2.10.1): Terminal and app are 0.2.11; Control and every engine package (Qwen, DeepSeek, Metal, SYCL, aarch64) are 0.1.57 (the CMake project version).
        # 0.2.10 (the 0.2.9 update, owner 2026-11-02): Terminal and app are 0.2.10; Control and every engine package (Qwen, DeepSeek, Metal, SYCL, aarch64) are 0.1.56 (the CMake project version).
        # 0.2.9.1 (a label prepared on 10-31 for a DeepSeek-only Windows AMD hotfix and never published: no tag, no 0.1.55.1 assets on v0.2.9) (DeepSeek engine only, the Windows AMD package): the DeepSeek engine is 0.1.55.1 (the CMake project version); the Terminal, the app and Control (0.1.55) and every other engine package stay as 0.2.9 shipped them.
        # 0.2.9: Terminal and app are 0.2.9; Control and every engine package (Qwen, DeepSeek, Metal, SYCL, aarch64) are 0.1.55.
        # 0.2.8.1 (Control-only): the Terminal is 0.2.8.1 and Control 0.1.55 (the CMake project version); the app stays 0.2.8 and every engine package 0.1.54 (as 0.2.3.1 did for the Terminal).
        # 0.2.8: Terminal and app are 0.2.8; Control and every engine package (Qwen, DeepSeek, Metal, SYCL, aarch64) are 0.1.54 (both engines changed since 0.2.7.1).
        # 0.2.7.1 (DGX Spark): Control and the linux-aarch64 engine packages are 0.1.53 (the CMake project version); the app and the Terminal stay 0.2.7, the Metal engine 0.1.52.
        # 0.2.7: Terminal and app are 0.2.7; Control and every engine package (Qwen, DeepSeek, Metal) are 0.1.52 (the CMake project version; the Qwen and DeepSeek C++ both changed since 0.2.6)
        # 0.2.13.1 (Control-only, Linux x86_64: the app on an Intel Arc card, ARCFIX): the Terminal is 0.2.13.1 and Control 0.1.59.1 (the CMake project version); the app stays 0.2.13 and every engine package 0.1.59 (as 0.2.8.1 and 0.2.3.1 did).
        # 0.2.14.1 (Control-only: the drafter fixes of the Terminal and Control): the Terminal is 0.2.14.1 and Control 0.1.60.1; the app stays 0.2.14 and every engine package 0.1.60.
        # 0.2.15 (an app + Control fix release): the app and the Terminal are 0.2.15 and Control 0.1.61; every engine package stays 0.1.60 (the v0.2.14 files).
        # 0.2.16 (the layout, own-stack and GUI / TUI release): the app and the Terminal are 0.2.16, Control 0.1.62 (0.1.61 went out with 0.2.15) and the engine packages 0.1.61.
        self.assertEqual(strata_cli.PRODUCT_VERSION, "0.2.16")
        self.assertEqual(strata_cli.version(), "0.1.62")              # the CMake project version: Control
        self.assertEqual(first("../engine/CMakeLists.txt", r"^project\(strata VERSION ([0-9.]+) "), "0.1.62")
        self.assertIn('STRATA_VERSION="0.1.61-metal"', (ROOT.parent / "engine" / "metal" / "CMakeLists.txt").read_text(encoding="utf-8"))

    def test_every_place_the_app_version_lives_says_0_2_3(self):
        self.assertEqual(first("../app/desktop/src-tauri/Cargo.toml", r'^version = "([^"]+)"'), "0.2.16")
        lock = (ROOT.parent / "app" / "desktop" / "src-tauri" / "Cargo.lock").read_text(encoding="utf-8")
        self.assertRegex(lock, r'name = "strata-desktop"\r?\nversion = "0.2.16"')
        self.assertEqual(json.loads((ROOT.parent / "app" / "desktop" / "package.json").read_text(encoding="utf-8"))["version"], "0.2.16")
        host = json.loads((ROOT.parent / "app" / "host" / "package.json").read_text(encoding="utf-8"))
        self.assertEqual(host["version"], "0.2.16")
        locked = json.loads((ROOT.parent / "app" / "host" / "package-lock.json").read_text(encoding="utf-8"))
        self.assertEqual((locked["version"], locked["packages"][""]["version"]), ("0.2.16", "0.2.16"))

    def test_the_documents_that_state_the_current_version_agree(self):
        self.assertIn("Version 0.2.16, engine 0.1.61.", (ROOT.parent / "AGENTS.md").read_text(encoding="utf-8"))
        for name in ("README.md", "docs/lang/README.de.md", "docs/lang/README.es.md", "docs/lang/README.fr.md", "docs/lang/README.ja.md", "docs/lang/README.pl.md", "docs/lang/README.pt-BR.md", "docs/lang/README.zh-CN.md"):
            text = (ROOT.parent / name).read_text(encoding="utf-8")
            self.assertRegex(text, r"Strata-0[.]2[.][0-9]+-", name)       # (the restyled READMEs state the app version through the download buttons; their owner moves them to the release)
            self.assertNotIn("Strata-0.2.2-", text, name)

    def test_version_prints_the_product_and_the_build(self):
        done = subprocess.run([sys.executable, str(ROOT / "strata.py"), "--version"], capture_output=True, text=True, cwd=str(ROOT),
                              env={**os.environ, "STRATA_NO_HANDOVER": "1"})
        self.assertTrue(done.stdout.startswith("strata 0.2.16 (build 0.1.62;"), done.stdout)


class UpdateFromTheLastRelease(test_updater.UpdaterCase):
    def feed_with_control(self, version):
        bundle = self.root / "control-bundle"
        (bundle / "strata_control").mkdir(parents=True)
        (bundle / "strata.py").write_text(f"print('strata 0.2.3 (build {version})')\n")
        (bundle / "strata_control" / "__init__.py").write_text("")
        from feed_tool import build_feed, pack_control
        (self.release / "control").mkdir()
        pack_control(bundle, self.release / "control", version=version, platform=test_updater.PLATFORM)
        build_feed(self.release, key=self.key, sequence=1, control_dir=self.release / "control")

    def test_a_0_2_2_install_is_offered_control_0_1_48_and_a_0_2_3_install_is_current(self):
        self.feed_with_control("0.1.48")
        (self.components / "control").mkdir(parents=True)
        (self.components / "control" / "active.json").write_text(json.dumps({"version": "0.1.47", "folder": "0.1.47"}))
        report = updater.check(self.ctx())
        self.assertEqual((report["control"]["status"], report["control"]["installed"], report["control"]["offered"]),
                         ("available", "0.1.47", "0.1.48"))
        out = updater.run_update(self.ctx(), engine=False, control=True)
        self.assertEqual((out["exit_code"], out["results"][0]["status"]), (0, "updated"), out)
        self.assertEqual(updater.active_control(self.components)["version"], "0.1.48")
        self.assertEqual(updater.check(self.ctx())["control"]["status"], "current")

    def test_the_bumped_numbers_sort_above_the_last_release(self):
        self.assertGreater(feedmod.version_key(strata_cli.version()), feedmod.version_key("0.1.47"))
        self.assertGreater(feedmod.version_key("0.1.48"), feedmod.version_key("0.1.46.3"))
        self.assertGreater(feedmod.version_key("0.1.48"), feedmod.version_key("0.1.45.3"))
