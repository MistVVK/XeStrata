"""ARCFIX (0.2.13): the app's model setup on an Intel Arc card (Linux, the SYCL engine).

Found by QAGUI-LIN on an Arc A750: the app's install job ran setup.py with no `--backend`, so setup's NVIDIA / AMD path stopped with "no NVIDIA GPU found
(nvidia-smi did not answer)". tools/strata_mcp.py now picks `sycl` for an Intel PC and passes `--backend sycl`; setup.py's CONTROL mode answers that with the
Terminal's own two steps (the pack, then the run config of strata_cli/launch/adopt.py). NVIDIA and AMD are what they were. Offline: nothing runs but Python."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
import setup  # noqa: E402
import strata_mcp as M  # noqa: E402
from strata_control import intel_app  # noqa: E402

A750 = {"vendor": "intel", "name": "Intel Arc A750", "vram_gb": 8.0, "usable": True, "index": 0}
NVIDIA = {"vendor": "nvidia", "name": "NVIDIA GeForce RTX 5070", "vram_gb": 11.9, "usable": True, "index": 0, "arch": "120"}
AMD = {"vendor": "amd", "name": "AMD Radeon RX 9070 XT", "vram_gb": 15.9, "usable": True, "index": 0, "arch": "gfx1201"}
AMD_IGPU = {"vendor": "amd", "name": "AMD Radeon Graphics", "vram_gb": 0.5, "usable": False, "index": 0, "arch": "gfx1036", "uma": True}


def shards(folder: Path, model: str = "Q2_0", family: str = "qwen") -> list[Path]:
    fam = setup.FAMILIES[family]
    folder.mkdir(parents=True, exist_ok=True)
    out = [folder / setup.model_file(fam, model, i) for i in range(1, setup.model_shards(fam, model) + 1)]
    for p in out:
        p.write_bytes(b"x")
    return out


class CardList(unittest.TestCase):
    def add(self, nvidia, amd, *, platform="linux", cards=(A750,), intel=True):
        return intel_app.cards_to_add(nvidia, amd, platform=platform, cards=lambda: [dict(c) for c in cards], intel_engine=lambda: intel)

    def test_an_arc_only_pc_gets_its_card(self):
        self.assertEqual([c["name"] for c in self.add([], [])], ["Intel Arc A750"])
        self.assertEqual([c["name"] for c in self.add([], [AMD_IGPU])], ["Intel Arc A750"], "an integrated Radeon that is not usable does not hide the Arc")

    def test_nvidia_and_amd_pcs_are_what_they_were(self):
        self.assertEqual(self.add([NVIDIA], []), [])
        self.assertEqual(self.add([], [AMD]), [])
        self.assertEqual(self.add([NVIDIA], [AMD]), [])

    def test_only_linux_and_only_when_the_engine_is_the_intel_one(self):
        self.assertEqual(self.add([], [], platform="win32"), [])           # no Windows Intel engine in the feed
        self.assertEqual(self.add([], [], platform="darwin"), [])
        self.assertEqual(self.add([], [], intel=False), [])                # the engine Control would pick is not SYCL (STRATA_ENGINE_BACKEND, a saved choice)
        self.assertEqual(self.add([], [], cards=()), [])

    def test_a_probe_that_fails_adds_nothing(self):
        def boom():
            raise OSError("no sysfs")
        self.assertEqual(intel_app.cards_to_add([], [], platform="linux", cards=boom, intel_engine=lambda: True), [])
        self.assertEqual(intel_app.cards_to_add([], [], platform="linux", cards=lambda: [A750], intel_engine=boom), [])

    def test_the_hardware_report_carries_the_arc_with_its_own_id(self):
        s = M.Strata(ROOT)
        S = s.setup_module()
        with mock.patch.object(M.sys, "platform", "linux"), mock.patch.object(M, "WIN", False), \
                mock.patch.object(S, "gpus", lambda: []), mock.patch.object(S, "amd_gpus", lambda: []), \
                mock.patch.object(M, "rocm_smi_gpus", lambda: []), mock.patch.object(M, "kfd_gpus", lambda: []), \
                mock.patch.object(intel_app, "arc_cards", lambda root="": [dict(A750)]), mock.patch.object(intel_app, "engine_is_intel", lambda: True), \
                mock.patch("strata_control.engine_component.status", lambda: {"usable": False}):
            hw = s._hardware()
        self.assertEqual([(g["vendor"], g["name"], g["vram_gb"], g["id"]) for g in hw["gpus"]], [("intel", "Intel Arc A750", 8.0, "intel:0")])
        self.assertEqual(hw["engine_vendor"], "intel")
        self.assertEqual([g["name"] for g in s.engine_gpus(hw)], ["Intel Arc A750"])


class Recommendation(unittest.TestCase):
    def test_an_arc_is_a_sycl_pc_and_says_there_is_no_download(self):
        s = M.Strata(ROOT)
        rec = s.recommend({"ram_gb": 64.0, "gpus": [dict(A750)], "engine_vendor": "intel", "cpu": {"avx2": True}})
        self.assertEqual(rec["backend"], "sycl")
        self.assertIn(intel_app.NO_DOWNLOAD, rec["notes"])
        self.assertTrue(rec["setup_command"].endswith(" --backend sycl"))

    def test_nvidia_and_amd_recommendations_are_unchanged(self):
        s = M.Strata(ROOT)
        rec = s.recommend({"ram_gb": 64.0, "gpus": [dict(NVIDIA)], "cpu": {"avx2": True}})
        self.assertEqual(rec["backend"], "cuda")
        self.assertNotIn("--backend", rec["setup_command"])
        self.assertNotIn(intel_app.NO_DOWNLOAD, rec["notes"])
        rec = s.recommend({"ram_gb": 64.0, "gpus": [dict(AMD)], "cpu": {"avx2": True}})
        self.assertEqual(rec["backend"], "hip")
        self.assertTrue(rec["setup_command"].endswith(" --backend hip"))


class InstallPlan(unittest.TestCase):
    def plan(self, gpus, *, backend=None, family="qwen", model="Q2_0", gguf_dir=None, data_dir=None, drafter="dflash2", vision="no", engine_vendor=None):
        s = M.Strata(ROOT)
        hw = {"ram_gb": 64.0, "gpus": [dict(g) for g in gpus], "cpu": {"avx2": True}, **({"engine_vendor": engine_vendor} if engine_vendor else {})}
        s.hardware = lambda fresh=False: hw
        s.disk_free = lambda p: 1000.0
        with mock.patch.object(M.sys, "platform", "linux"), mock.patch.object(M, "WIN", False):
            return M.Tools(s).install_plan(family, model, 32768, vision, "int8", None, None, backend, None, None, data_dir, gguf_dir, drafter)

    def test_the_apps_request_on_an_arc_passes_backend_sycl(self):
        """The exact request QAGUI-LIN saw the app make (own folder, 32K, int8, DFlash2): now with --backend sycl and without the drafter."""
        with tempfile.TemporaryDirectory() as t:
            folder = Path(t) / "Q2_0"
            shards(folder)
            plan, args = self.plan([A750], gguf_dir=str(folder), engine_vendor="intel")
        self.assertEqual(args, ["--yes", "--no-start", "--family", "qwen", "--model", "Q2_0", "--context", "32768", "--vision", "no",
                                "--experimental-speed-projection", "off", "--kv", "int8", "--gguf-dir", str(folder), "--backend", "sycl"])
        self.assertTrue(plan["partly_downloaded"])

    def test_pictures_are_off_on_an_arc_even_when_asked_for(self):
        with tempfile.TemporaryDirectory() as t:
            folder = Path(t) / "Q2_0"
            shards(folder)
            _plan, args = self.plan([A750], gguf_dir=str(folder), vision="yes")
        self.assertEqual(args[args.index("--vision") + 1], "no")

    def test_the_backend_can_be_named(self):
        with tempfile.TemporaryDirectory() as t:
            folder = Path(t) / "Q2_0"
            shards(folder)
            _plan, args = self.plan([A750], backend="sycl", gguf_dir=str(folder))
        self.assertEqual(args[-2:], ["--backend", "sycl"])
        with self.assertRaises(M.ToolError) as raised:                       # sycl on a PC with no Arc card
            self.plan([NVIDIA], backend="sycl")
        self.assertIn("no Intel Arc card", str(raised.exception))

    @unittest.skipIf(sys.platform == "darwin", "the test's temporary folder is under /var/folders, a macOS system folder the data-folder check refuses; and setup hands a Mac to metal/setup_mac.py")
    def test_without_the_files_there_is_no_download_and_the_words_say_so(self):
        with tempfile.TemporaryDirectory() as t:
            data = Path(t) / "Strata-data"
            data.mkdir()
            with self.assertRaises(M.ToolError) as raised:
                self.plan([A750], data_dir=str(data))
            self.assertEqual(str(raised.exception), intel_app.NO_DOWNLOAD)
            shards(data / "models" / "Q2_0")                                   # the files are where the Terminal looks (<data>/models/<tag>)
            plan, args = self.plan([A750], data_dir=str(data))
        self.assertEqual(args[-2:], ["--backend", "sycl"])
        self.assertNotIn("--gguf-dir", args)
        self.assertTrue(plan["partly_downloaded"])

    def test_nvidia_and_amd_requests_are_byte_identical_to_before(self):
        with tempfile.TemporaryDirectory() as t:
            folder = Path(t) / "Q2_0"
            shards(folder)
            _plan, nvidia = self.plan([NVIDIA], gguf_dir=str(folder))
            _plan, amd = self.plan([AMD], gguf_dir=str(folder))
            _plan, amd_named = self.plan([AMD], backend="hip", gguf_dir=str(folder))
            _plan, cuda_named = self.plan([NVIDIA], backend="cuda", gguf_dir=str(folder))
        base = ["--yes", "--no-start", "--family", "qwen", "--model", "Q2_0", "--context", "32768", "--vision", "no",
                "--experimental-speed-projection", "off", "--kv", "int8", "--gguf-dir", str(folder)]
        self.assertEqual(nvidia, base + ["--drafter", "dflash2"])
        self.assertEqual(amd, base + ["--drafter", "dflash2"])
        self.assertEqual(amd_named, base + ["--backend", "hip", "--drafter", "dflash2"])
        self.assertEqual(cuda_named, base + ["--backend", "cuda", "--drafter", "dflash2"])

    def test_an_arc_beside_an_nvidia_card_is_an_nvidia_install(self):
        """The card list only holds an Arc when no usable NVIDIA / AMD card is there, so a mixed PC takes the old path."""
        with tempfile.TemporaryDirectory() as t:
            folder = Path(t) / "Q2_0"
            shards(folder)
            _plan, args = self.plan([NVIDIA], gguf_dir=str(folder))
        self.assertNotIn("sycl", args)

    def test_the_tool_schema_offers_sycl(self):
        s = M.Strata(ROOT)
        tools = {t["name"]: t for t in M.Tools(s).definitions()} if hasattr(M.Tools, "definitions") else {}
        if tools:
            self.assertEqual(tools["strata_install"]["inputSchema"]["properties"]["backend"]["enum"], ["auto", "cuda", "hip", "sycl"])
        else:
            self.assertIn('"enum": ["auto", "cuda", "hip", "sycl"]', (ROOT / "tools" / "strata_mcp.py").read_text(encoding="utf-8"))


class SetupControlMode(unittest.TestCase):
    def args(self, folder, **kw):
        base = dict(family="qwen", model="Q2_0", gguf_dir=str(folder), models_dir=None, data_dir=None, context=32768, kv="int8", port=None, host=None,
                    api_key=None, browser=None, check=False, yes=True, move_data=False, set_default_data_dir=False, prepare_pack=None)
        base.update(kw)
        return mock.Mock(**base)

    def run_setup(self, a, data, home, *, driver="xe", prepared=True):
        def fake_pack(args):
            pack = Path(args.prepare_pack)
            (pack / "tokenizer").mkdir(parents=True, exist_ok=True)
            (pack / "tokenizer" / "vocab.json").write_text("{}")
            (pack / "index.txt").write_text("x")
            self.packed = args.prepare_pack
            return 0
        self.packed = None
        with mock.patch.object(setup, "CONTROL", True), mock.patch.object(setup, "_CONTROL_HOME", home), \
                mock.patch.object(setup, "data_folder_for", lambda a: (data, [])), \
                mock.patch.object(setup, "prepare_pack_only", fake_pack), \
                mock.patch.object(intel_app, "arc_cards", lambda root="": [dict(A750)]), \
                mock.patch("strata_cli.launch.adopt.intel_driver", lambda: driver), \
                mock.patch.object(intel_app, "supported_here", lambda platform=None: True):
            return setup.intel_control_setup(a)

    def test_the_pack_and_the_run_config_are_written_as_the_terminal_writes_them(self):
        with tempfile.TemporaryDirectory() as t:
            t = Path(t)
            folder, data, home = t / "Q2_0", t / "Strata-data", t / "install"
            files = shards(folder)
            home.mkdir()
            data.mkdir()
            self.assertEqual(self.run_setup(self.args(folder), data, home, driver="i915"), 0)
            self.assertEqual(self.packed, str(data / "packs" / "q2_0"))
            cfg = json.loads((home / "strata-q2_0.json").read_text(encoding="utf-8"))
        self.assertEqual((cfg["exe"], cfg["backend"], cfg["port"]), ("component:strata-engine", "sycl", 8080))
        a = cfg["args"]
        self.assertEqual(a[a.index("--pack") + 1], str(data / "packs" / "q2_0"))
        self.assertEqual(a[a.index("--native") + 1], str(files[0]))
        self.assertEqual(a[a.index("--ple-gguf") + 1], str(files[-1]))
        self.assertEqual(a[a.index("--ple-io") + 1], "ram")                    # the A750 runs on i915
        self.assertEqual(a[a.index("--kv") + 1], "int8")
        self.assertNotIn("--vision", a)
        self.assertEqual(cfg["tokenizer"], str(data / "packs" / "q2_0" / "tokenizer"))

    def test_a_b_series_card_streams_its_experts(self):
        with tempfile.TemporaryDirectory() as t:
            t = Path(t)
            folder, data, home = t / "Q2_0", t / "Strata-data", t / "install"
            shards(folder)
            home.mkdir()
            data.mkdir()
            self.run_setup(self.args(folder, port=8123), data, home, driver="xe")
            cfg = json.loads((home / "strata-q2_0.json").read_text(encoding="utf-8"))
        self.assertIn("--stream-experts", cfg["args"])
        self.assertEqual(cfg["port"], 8123)

    def test_no_files_no_download(self):
        with tempfile.TemporaryDirectory() as t:
            t = Path(t)
            (t / "empty").mkdir()
            home = t / "install"
            home.mkdir()
            with self.assertRaises(SystemExit):
                self.run_setup(self.args(t / "empty"), t / "Strata-data", home)
            self.assertIsNone(self.packed)
            self.assertEqual(list(home.glob("strata-*.json")), [])

    @unittest.skipIf(sys.platform == "darwin", "the test's temporary folder is under /var/folders, a macOS system folder the data-folder check refuses; and setup hands a Mac to metal/setup_mac.py")
    def test_the_dispatch_in_main_is_by_mode(self):
        """CONTROL mode (the app, the Terminal's installer mode) prepares and adopts; a checkout keeps the from-source Intel setup."""
        for control, expect in ((True, "intel_control_setup"), (False, "sycl_setup")):
            with mock.patch.object(setup, "CONTROL", control), mock.patch.object(setup, "intel_control_setup", return_value=11) as ctl, \
                    mock.patch.object(setup, "sycl_setup", return_value=22) as old, mock.patch.object(sys, "argv", ["setup.py", "--backend", "sycl", "--yes"]):
                self.assertEqual(setup.main(), 11 if control else 22)
                self.assertEqual((ctl.called, old.called), (control, not control), expect)


if __name__ == "__main__":
    unittest.main()
