"""#5109: on a computer with an idle NVIDIA card beside the AMD card the engine serves from (an RTX 3060 next to an RX 7900 XTX; the lent
RTX 3080 + RX 6700 XT test box), detect_hardware() answers "cuda" with the NVIDIA card first.  The drafter's support line, card class and
confidence must come from the engine's own family (the config's "backend"), not from the idle card.  No GPU needed."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from strata_control import drafting  # noqa: E402

MIXED = ("cuda", ["sm_86", "gfx1100"])


class HardwareFor(unittest.TestCase):
    def test_the_config_family_wins_with_its_own_cards(self):
        with mock.patch.object(drafting, "detect_hardware", return_value=MIXED):
            self.assertEqual(drafting.hardware_for("hip"), ("hip", ["gfx1100"]))
            self.assertEqual(drafting.hardware_for("cuda"), ("cuda", ["sm_86"]))
            self.assertEqual(drafting.hardware_for("HIP"), ("hip", ["gfx1100"]))

    def test_no_family_or_another_family_keeps_detect_hardware(self):
        with mock.patch.object(drafting, "detect_hardware", return_value=MIXED):
            self.assertEqual(drafting.hardware_for(None), MIXED)
            self.assertEqual(drafting.hardware_for(""), MIXED)
            self.assertEqual(drafting.hardware_for("metal"), MIXED)
            self.assertEqual(drafting.hardware_for("sycl"), MIXED)

    def test_one_vendor_computers_are_unchanged(self):
        for hw, fam in ((("cuda", ["sm_120"]), "cuda"), (("hip", ["gfx1201"]), "hip"), (("hip", ["gfx1151"]), "hip")):
            with mock.patch.object(drafting, "detect_hardware", return_value=hw):
                self.assertEqual(drafting.hardware_for(fam), hw)

    def test_class_and_support_follow_the_serving_card(self):
        with mock.patch.object(drafting, "detect_hardware", return_value=MIXED):
            backend, archs = drafting.hardware_for("hip")
        self.assertEqual(drafting.card_class(archs), drafting.card_class(["gfx1100"]))
        self.assertNotEqual(drafting.card_class(archs), drafting.card_class(["sm_86"]))
        self.assertEqual(drafting.hardware_support(backend, archs), drafting.hardware_support("hip", ["gfx1100"]))


class ServerStart(unittest.TestCase):
    """serve/server.py engine_args hands drafting.apply the serving card's family, cards and class."""

    def test_engine_args_on_a_mixed_computer(self):
        from serve import server
        seen = {}

        def apply(shaped, **kw):
            seen.update(kw)
            return mock.Mock(note="", flags=[], vocab="", drafter="off")
        with mock.patch.object(drafting, "detect_hardware", return_value=MIXED), mock.patch.object(drafting, "apply", side_effect=apply), \
                mock.patch.object(server, "drafting_vram_mib", return_value=24560.0):
            try:
                server.engine_args({"args": ["--max-context", "4096"], "backend": "hip"}, announce=False)
            except Exception:                                 # noqa: BLE001 - only the drafting call is under test here
                pass
        self.assertEqual(seen.get("backend"), "hip")
        self.assertEqual(seen.get("archs"), ["gfx1100"])
        self.assertEqual(seen.get("klass"), drafting.card_class(["gfx1100"]))


class VramFitOfTheServingCard(unittest.TestCase):
    """#5322: a HIP config on a Strix Halo with two NVIDIA cards beside it (detect_hardware: cuda, sm_120, sm_120, gfx1151) was sized by the start-time VRAM fit as a
    16 GB NVIDIA card and got --kv-resident 4096.  The fit takes the config's family now (the same backend as the drafter, 188aaf7bf): the Strix Halo is a
    shared-memory chip, so no single-card figure, and nothing is added."""

    def test_no_kv_window_from_the_nvidia_card(self):
        from serve import server
        cfg = {"backend": "hip", "gpu": 0, "parallel": 2, "model_name": "Qwen3.8-Flash-Next",
               "args": ["--expert-cache", "24576", "--resident-experts", "--prefill", "16384", "--spec", "4", "--max-context", "262144", "--kv", "int8"]}
        seen = []

        def vram(gl, hip=False):
            seen.append(hip)
            return {0: 16311} if not hip else None              # an RTX 5060 Ti answers through NVML; the iGPU is never asked for one card's memory
        with mock.patch.object(drafting, "detect_hardware", return_value=("cuda", ["sm_120", "sm_120", "gfx1151"])), \
                mock.patch.object(server, "gpu_vram_mib", side_effect=vram), \
                mock.patch.object(drafting, "apply", return_value=mock.Mock(note="", flags=[], vocab="", drafter="off")):
            args = server.engine_args(cfg, announce=False)
        self.assertNotIn("--kv-resident", args)
        self.assertNotIn(False, seen)                            # the NVIDIA card's memory was never read for this start


class OtherCallers(unittest.TestCase):
    """#5109-rest: every other place that reads the computer's cards for the drafter takes the config's family on a mixed computer.  Each test spies on
    drafting.hardware_for with detect_hardware answering the mixed computer, and checks what the caller passes on."""

    def spy(self):
        seen = []
        real = drafting.hardware_for

        def spy(fam):
            seen.append(fam)
            return real(fam)
        return seen, spy

    def test_doctor(self):
        from strata_control import doctor
        seen, spy = self.spy()
        got = {}
        with mock.patch.object(drafting, "detect_hardware", return_value=MIXED), mock.patch.object(drafting, "hardware_for", side_effect=spy), \
                mock.patch.object(drafting, "choice_of", side_effect=lambda *a, **k: got.update(k) or "mtp"):
            doctor._drafter_is_dflash2({"backend": "hip", "drafting": "auto"})
        self.assertEqual(seen, ["hip"])
        self.assertEqual((got["backend"], got["archs"]), ("hip", ["gfx1100"]))

    def test_modelstate(self):
        from strata_cli.launch import modelstate
        got = {}
        with mock.patch.object(drafting, "detect_hardware", return_value=MIXED), \
                mock.patch.object(drafting, "choice_of", side_effect=lambda *a, **k: got.update(k) or "mtp"):
            modelstate.resolved_drafting("auto", "qwen", engine_backend="hip")
            self.assertEqual((got["backend"], got["archs"]), ("hip", ["gfx1100"]))
            modelstate.resolved_drafting("auto", "qwen")                       # no config family known: as before
            self.assertEqual((got["backend"], got["archs"]), MIXED)

    def test_screens(self):
        from strata_cli.launch import screens
        ctx = mock.Mock()
        ctx.settings.family = sorted(drafting.SUPPORTED_FAMILIES)[0]
        ctx.catalog.installed_backend.return_value = "hip"
        got = []
        with mock.patch.object(drafting, "detect_hardware", return_value=MIXED), mock.patch.object(drafting, "ENGINE_SERVES_DFLASH2", True), \
                mock.patch.object(drafting, "hardware_support", side_effect=lambda b, a=None, m=None: got.append((b, a)) or drafting.Support(True, "x", "")):
            screens.drafting_choice_available(ctx)
        self.assertEqual(got, [("hip", ["gfx1100"])])

    def test_runner_auto_signals_and_drafting_inputs(self):
        from strata_cli.launch import runner
        catalog = mock.Mock()
        catalog.installed_config.return_value = "cfg.json"
        catalog.model_name_of.return_value = "Qwen3.8-Flash-Next"
        cfg = {"backend": "hip", "args": ["--max-context", "4096"], "model_name": "Qwen3.8-Flash-Next"}
        want = []
        with mock.patch.object(drafting, "detect_hardware", return_value=MIXED), mock.patch.object(runner, "read_config", return_value=dict(cfg)), \
                mock.patch.object(runner, "apply_choices", side_effect=lambda c, s: c), \
                mock.patch.object(runner, "card_signals", side_effect=lambda c, g, w: want.append(w) or {"args": [], "vram_mib": None}):
            sig = runner.auto_signals(catalog, mock.Mock(gpus=None, family="qwen", model="iq3_s"), "qwen")
        self.assertEqual(want, [{"backend": "hip", "archs": ["gfx1100"]}])
        self.assertEqual(sig["engine_backend"], "hip")
        fake = mock.Mock()
        fake.s = mock.Mock(drafting=None, gpus=None, parallel=0, family="qwen", data_dir=None)
        fake.split_expected.return_value = False
        with mock.patch.object(drafting, "detect_hardware", return_value=MIXED):
            got = runner.Runner.drafting_inputs(fake, dict(cfg))
        self.assertEqual((got["backend"], got["archs"], got["klass"]), ("hip", ["gfx1100"], drafting.card_class(["gfx1100"])))

    def test_drafting_cmd_and_installed_engine(self):
        from strata_cli import drafting_cmd
        with mock.patch.object(drafting, "detect_hardware", return_value=MIXED):
            self.assertEqual(drafting_cmd._hardware("hip"), ("hip", ["gfx1100"]))
            self.assertEqual(drafting_cmd._hardware(), MIXED)
        from strata_control import engine_component as ec

        class C:
            def __init__(self, backend, version):
                self.backend, self.version = backend, version
                self.exe = mock.Mock(is_file=lambda: True, name=backend)
        found = [C("cuda", "0.1.60"), C("hip:rx7900", "0.1.60")]
        with mock.patch.object(drafting, "detect_hardware", return_value=MIXED), mock.patch.object(ec, "scan", return_value=(found, None)):
            self.assertIs(drafting.installed_engine_exe("hip"), found[1].exe)
            self.assertIs(drafting.installed_engine_exe(), found[0].exe)      # no family given: detect_hardware's, as before


if __name__ == "__main__":
    unittest.main()
