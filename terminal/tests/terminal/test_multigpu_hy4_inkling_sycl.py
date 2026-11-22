"""Multi-GPU support tests for Hy4-Flash, Inkling-2-Mini, AMD HIP (Windows + Linux),
Intel SYCL (ZE_AFFINITY_MASK / ONEAPI_DEVICE_SELECTOR), and Issue #6473 stall guard.
"""

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
from serve import hy4_kind, inkling_kind, server  # noqa: E402
from strata_cli.launch import runner as R  # noqa: E402
from strata_control import gpu_roles, hy4_fit, intel_app, split_engine  # noqa: E402


class Hy4KindMultiGpuTests(unittest.TestCase):
    BASE = ["--native", "m.strata", "--engram", "e.strata", "--max-context", "131072"]

    def test_single_gpu_does_not_pass_gpus_flag(self):
        with mock.patch.object(hy4_kind, "probe", return_value=None):
            for gpu in (None, 0, [0], "0"):
                cfg = {"arch": "hy4flash", "args": list(self.BASE)}
                if gpu is not None:
                    cfg["gpu"] = gpu
                args = hy4_kind.engine_args(cfg, "x", announce=False)
                self.assertNotIn("--gpus", args, gpu)

    def test_multi_gpu_emits_main_first_local_ordinals(self):
        """child_env() puts Helpers first and Main last in *_VISIBLE_DEVICES, so inside the child
        process local device N-1 is Main and 0..N-2 are Helpers. strata-hy4 --gpus expects primary first."""
        with mock.patch.object(hy4_kind, "probe", return_value=None):
            for gpu, want in (
                ([0, 1], "1,0"),
                ([2, 0, 1], "2,0,1"),
                ([0, 1, 2, 3], "3,0,1,2"),
            ):
                cfg = {"arch": "hy4flash", "args": list(self.BASE), "gpu": gpu}
                args = hy4_kind.engine_args(cfg, "x", announce=False)
                self.assertIn("--gpus", args)
                self.assertEqual(args[args.index("--gpus") + 1], want, gpu)

    def test_gpu_roles_emits_main_first_and_respects_explicit_gpus_arg(self):
        with mock.patch.object(hy4_kind, "probe", return_value=None):
            cfg = {"arch": "hy4flash", "args": list(self.BASE), "gpu_roles": {"0": "main", "1": "helper"}}
            args = hy4_kind.engine_args(cfg, "x", announce=False)
            self.assertEqual(args[args.index("--gpus") + 1], "1,0")

            explicit = {"arch": "hy4flash", "args": self.BASE + ["--gpus", "0,1"], "gpu": [0, 1]}
            args2 = hy4_kind.engine_args(explicit, "x", announce=False)
            self.assertEqual(args2.count("--gpus"), 1)
            self.assertEqual(args2[args2.index("--gpus") + 1], "0,1")


class InklingKindMultiGpuTests(unittest.TestCase):
    BASE = ["--native", "i.strata", "--max-context", "32768"]

    def test_multi_gpu_via_list_csv_or_gpu_roles_across_backends(self):
        with mock.patch.object(inkling_kind, "probe", return_value=None):
            for backend in ("cuda", "hip", "sycl"):
                for extra in ({"gpu": [0, 1]}, {"gpu": "0,1"}, {"gpu_roles": {"0": "main", "1": "helper"}}):
                    cfg = {"arch": "inkling2_loop", "backend": backend, "args": list(self.BASE), **extra}
                    args = inkling_kind.engine_args(cfg, "x", announce=False)
                    self.assertIn("--gpus", args, (backend, extra))
                    self.assertEqual(args[args.index("--gpus") + 1], "1,0", (backend, extra))

            single = {"arch": "inkling2_loop", "backend": "hip", "gpu": 0, "args": list(self.BASE)}
            self.assertNotIn("--gpus", inkling_kind.engine_args(single, "x", announce=False))


class SyclEnvAndRolesTests(unittest.TestCase):
    def test_ordered_gpus_sycl_skips_nvidia_speed_scores(self):
        cfg = {"backend": "sycl", "gpu": [1, 0]}
        with mock.patch.object(server, "gpu_speed_scores", side_effect=AssertionError("must not probe NVIDIA for SYCL")):
            self.assertEqual(server.ordered_gpus(cfg), [1, 0])

    def test_child_env_sycl_sets_ze_affinity_mask_and_selector(self):
        with mock.patch.dict(server.os.environ, {}, clear=True):
            env = server.child_env({"backend": "sycl", "gpu": [1, 0], "gpu_order": "as_given"})
            self.assertEqual(env["ZE_AFFINITY_MASK"], "1,0")
            self.assertEqual(env["ONEAPI_DEVICE_SELECTOR"], "level_zero:gpu")

        with mock.patch.dict(server.os.environ, {"ONEAPI_DEVICE_SELECTOR": "level_zero:0,1"}, clear=True):
            env = server.child_env({"backend": "sycl", "gpu": [0, 1], "gpu_order": "as_given"})
            self.assertEqual(env["ZE_AFFINITY_MASK"], "0,1")
            self.assertEqual(env["ONEAPI_DEVICE_SELECTOR"], "level_zero:0,1")

        with mock.patch.dict(server.os.environ, {}, clear=True):
            env_off = server.child_env({"backend": "sycl", "gpu_roles": {"0": "off", "1": "off"}, "gpu_none": True})
            self.assertEqual(env_off["ZE_AFFINITY_MASK"], "")
            self.assertEqual(env_off["ONEAPI_DEVICE_SELECTOR"], "")

    def test_gpu_roles_env_visible_intel(self):
        self.assertEqual(gpu_roles.env_visible([0, 1], "intel", {"ZE_AFFINITY_MASK": "1,0"}), [1, 0])
        self.assertEqual(gpu_roles.env_visible([0, 1], "intel", {"ZE_AFFINITY_MASK": ""}), [])
        self.assertIsNone(gpu_roles.env_visible([0, 1], "intel", {}))


class StallGuardMultiGpuTests(unittest.TestCase):
    """Issue #6473: averaged gpu_util across N cards can dip below 10% while one stage card is active."""

    def test_any_single_active_card_keeps_gpu_busy_true(self):
        svc = server.Service.__new__(server.Service)
        svc.telemetry = None
        svc.gpu_index = 0
        svc.gpu_indices = [0, 1]
        svc.backend = "cuda"
        svc._tok_s = lambda: 0.0
        svc._tok_s_mean = lambda: 0.0
        svc._prefill_tok_s_mean = lambda: 0.0

        class DummyEngine:
            gpu_busy = None

        svc.engine = DummyEngine()

        class FakeTelemetry:
            def __init__(self, **kw):
                self.snap = {"now": {"gpu_util": 5, "gpus": [{"index": 0, "util": 0}, {"index": 1, "util": 12}]}}

            def snapshot(self):
                return self.snap

            def start(self):
                pass

        with mock.patch("serve.telemetry.Telemetry", FakeTelemetry):
            svc.start_telemetry()

        self.assertTrue(svc.engine.gpu_busy())
        svc.telemetry.snap = {"now": {"gpu_util": 4, "gpus": [{"index": 0, "util": 4}, {"index": 1, "util": 5}]}}
        self.assertFalse(svc.engine.gpu_busy())
        svc.telemetry.snap = {"now": {"gpu_util": 15, "gpus": []}}
        self.assertTrue(svc.engine.gpu_busy())


class Hy4FitMultiGpuTests(unittest.TestCase):
    def test_extra_vram_increases_vram_and_preserves_single_gpu(self):
        single = hy4_fit.place("SQ-64", 16.0, 64.0, False, 131072, True)
        single_zero = hy4_fit.place("SQ-64", 16.0, 64.0, False, 131072, True, extra_vram_gib=0.0)
        self.assertEqual(single, single_zero)

        dual = hy4_fit.place("SQ-64", 16.0, 64.0, False, 131072, True, extra_vram_gib=16.0)
        self.assertGreater(dual["vram_gb"], single["vram_gb"] + 14.0)
        self.assertLess(dual["ram_gb"], single["ram_gb"])
        self.assertEqual(dual["ssd_gb"], 0.0)

        rec_single = hy4_fit.recommend(24.0, 64.0, False, 131072, True)
        rec_quad = hy4_fit.recommend(24.0, 64.0, False, 131072, True, extra_vram_gib=72.0)
        self.assertEqual(rec_quad["build"], "SQ-128")
        self.assertTrue(rec_quad["fits"])
        self.assertNotEqual(rec_single["build"], "SQ-128")


class SplitEngineWindowsAmdTests(unittest.TestCase):
    R9700 = dict(name="AMD Radeon AI PRO R9700", vram_mib=32624, cc=0, vendor="amd", arch="gfx1201")
    A6000 = dict(name="NVIDIA RTX A6000", vram_mib=49140, cc=86, vendor="nvidia")

    def _model(self):
        gib = 1 << 30
        return split_engine.ModelFacts(
            routed_types={"Q4_K": 97, "Q5_K": 2, "Q5_1": 43, "Q8_0": 5},
            routed_bytes=int(71.7 * gib),
            embd_type="Q8_0",
            shexp_types={"Q8_0": 144},
            embd_ram_bytes=int(27.5 * gib),
            layers=48,
            experts=512,
        )

    def test_windows_amd_is_allowed_while_windows_nvidia_remains_refused(self):
        amd_cards = [split_engine.Card(index=i, **self.R9700) for i in range(4)]
        nv_cards = [split_engine.Card(index=i, **self.A6000) for i in range(2)]
        d_amd = split_engine.decide(enabled=True, system="windows", cards=amd_cards, ram_total_gib=128, ram_available_gib=80, model=self._model())
        self.assertTrue(d_amd.use_split, d_amd.reasons)
        d_nv = split_engine.decide(enabled=True, system="windows", cards=nv_cards, ram_total_gib=128, ram_available_gib=80, model=self._model())
        self.assertFalse(d_nv.use_split)

    def test_exe_finder_resolves_windows_amd_exe(self):
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            base = d / "strata-engine.exe"
            split_amd = d / "strata-split-4-amd.exe"
            base.write_bytes(b"x")
            split_amd.write_bytes(b"y")
            finder = split_engine.exe_finder({"exe": str(base)})
            self.assertEqual(finder(4, 0, "gfx1201"), str(split_amd))


class RunnerApplyCardsNativeEpTests(unittest.TestCase):
    def test_apply_cards_omits_layer_split_for_native_ep_and_keeps_for_qwen(self):
        for cfg in (
            {"exe": "component:strata-hy4", "arch": "hy4flash", "layer_split": "auto"},
            {"exe": "component:strata-inkling", "arch": "inkling2_loop", "layer_split": "auto"},
        ):
            R.apply_cards(cfg, {"gpu": [1, 0], "gpu_roles": {"0": "main", "1": "helper"}, "gpu_order": "as_given"})
            self.assertNotIn("layer_split", cfg)
            self.assertEqual(cfg["gpu"], [1, 0])

        qwen_cfg = {"exe": "component:strata-engine", "arch": "qwen4exp"}
        R.apply_cards(qwen_cfg, {"gpu": [1, 0], "gpu_roles": {"0": "main", "1": "helper"}, "gpu_order": "as_given"})
        self.assertEqual(qwen_cfg["layer_split"], "auto")


class SetupMultiGpuTests(unittest.TestCase):
    def test_multi_gpu_cards_resolves_ints_and_dicts(self):
        amd = [
            {"index": 0, "name": "AMD Radeon RX 9070 XT", "vram_gb": 16.0, "usable": True, "vendor": "amd"},
            {"index": 1, "name": "AMD Radeon RX 9070 XT", "vram_gb": 16.0, "usable": True, "vendor": "amd"},
        ]
        got_ints = setup._multi_gpu_cards([1, 0], amd[0], True, amd)
        self.assertEqual([c["index"] for c in got_ints], [1, 0])
        got_dicts = setup._multi_gpu_cards(amd, amd[0], True, amd)
        self.assertEqual([c["index"] for c in got_dicts], [0, 1])

    def test_intel_app_allows_hy4_and_inkling(self):
        self.assertTrue(intel_app.can_set_up("hy4-flash"))
        self.assertTrue(intel_app.can_set_up("inkling-2-mini"))
        self.assertTrue(intel_app.can_set_up("custom", {"hy4": True}))
        self.assertTrue(intel_app.can_set_up("custom", {"inkling": True}))

    def test_install_hy4_and_inkling_multi_gpu_write_configs(self):
        with tempfile.TemporaryDirectory() as td:
            t = Path(td)
            data = t / "data"
            home = t / "install"
            data.mkdir()
            home.mkdir()
            hy4_dir = data / "models" / "hy4-flash"
            ink_dir = data / "models" / "inkling-2-mini"
            hy4_dir.mkdir(parents=True)
            ink_dir.mkdir(parents=True)
            for p in (
                hy4_dir / "Hy4-Flash-SQ-64.strata",
                hy4_dir / "Hy4-Flash-Engram-Q8.strata",
                hy4_dir / "Hy4-Flash-Vision.strata",
                ink_dir / "Inkling-2-Mini-SQ-20.strata",
                ink_dir / "Inkling-2-Mini-Vision.strata",
            ):
                p.write_bytes(b"x")

            amd_cards = [
                {"index": 0, "name": "AMD Radeon RX 9070 XT", "vram_gb": 16.0, "usable": True, "vendor": "amd", "arch": "gfx1201"},
                {"index": 1, "name": "AMD Radeon RX 9070 XT", "vram_gb": 16.0, "usable": True, "vendor": "amd", "arch": "gfx1201"},
            ]
            arc_cards = [
                {"index": 0, "name": "Intel Arc B580", "vram_gb": 12.0, "usable": True, "vendor": "intel"},
                {"index": 1, "name": "Intel Arc B580", "vram_gb": 12.0, "usable": True, "vendor": "intel"},
            ]

            def make_args(family, model, backend):
                return type(
                    "Args",
                    (),
                    {
                        "family": family,
                        "model": model,
                        "backend": backend,
                        "context": None,
                        "vision": "yes",
                        "port": 8095,
                        "host": "127.0.0.1",
                        "api_key": None,
                        "browser": False,
                        "check": False,
                        "yes": True,
                        "no_start": True,
                        "gguf_dir": None,
                        "models_dir": None,
                        "data_dir": str(data),
                        "parallel": None,
                        "reasoning_effort": None,
                        "rope_scaling": None,
                        "rope_scale": None,
                        "kv": None,
                        "resident_budget_gib": None,
                        "low_ram": None,
                        "gpu": None,
                        "gpus": None,
                        "gpu_roles": None,
                        "mtp": None,
                        "drafter": None,
                        "experimental_speed_projection": None,
                        "layer_split": None,
                        "move_data": False,
                        "set_default_data_dir": False,
                        "prepare_pack": None,
                        "_roles": {"0": "main", "1": "helper"},
                        "_roles_cards": amd_cards if backend == "hip" else arc_cards,
                    },
                )()

            with (
                mock.patch.object(setup, "CONTROL", True),
                mock.patch.object(setup, "_CONTROL_HOME", home),
                mock.patch.object(setup, "data_folder_for", lambda a: (data, [])),
                mock.patch.object(setup, "hy4_have", lambda a, p: True),
                mock.patch.object(setup, "inkling_have", lambda a, p: True),
                mock.patch.object(setup, "hy4_fetch", lambda a, n, folder, own: folder / n),
                mock.patch.object(setup, "inkling_fetch", lambda a, n, folder, own: folder / n),
                mock.patch.object(setup, "hy4_tokenizer_dir", lambda *a, **kw: data / "tok"),
                mock.patch.object(setup, "inkling_tokenizer_dir", lambda *a, **kw: data / "tok"),
                mock.patch.object(setup, "engine_step", lambda *a, **kw: (None, str(t / "eng"), {"component": "ok"}, [], a[1], None)),
                mock.patch.object(setup, "write_run_script", lambda *a, **kw: home / "run.bat"),
                mock.patch.object(setup, "ram_gb", lambda: 64.0),
                mock.patch.object(setup, "free_gb", lambda p: 500.0),
                mock.patch("builtins.print") as mock_print,
            ):
                rc_hy4 = setup.install_hy4_flash(make_args("hy4-flash", "SQ-64", "sycl"), "hy4-flash", arc_cards[0], False, [], None, arc_cards, 8095, 64.0, None)
                self.assertEqual(rc_hy4, 0)
                printed = "\n".join(str(c) for c in mock_print.call_args_list)
                self.assertNotIn("runs on one GPU so far", printed)
                hy4_cfg = json.loads((home / "strata-hy4-flash-sq-64.json").read_text(encoding="utf-8"))
                self.assertEqual(hy4_cfg["backend"], "sycl")
                self.assertEqual(hy4_cfg["gpu"], [1, 0])
                self.assertEqual(hy4_cfg["gpu_order"], "as_given")
                self.assertEqual(hy4_cfg["gpu_roles"], {"0": "main", "1": "helper"})

                # Inkling-2-Mini on AMD HIP with integer multi list (as passed from main())
                rc_ink = setup.install_inkling2(make_args("inkling-2-mini", "SQ-20", "hip"), "inkling-2-mini", amd_cards[0], True, amd_cards, None, [1, 0], 8096, 64.0, None)
                self.assertEqual(rc_ink, 0)
                ink_cfg = json.loads((home / "strata-inkling-2-mini-sq-20.json").read_text(encoding="utf-8"))
                self.assertEqual(ink_cfg["gpu"], [1, 0])
                self.assertIn("--gpus", ink_cfg["args"])
                self.assertEqual(ink_cfg["args"][ink_cfg["args"].index("--gpus") + 1], "2")


if __name__ == "__main__":
    unittest.main()
