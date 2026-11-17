"""A PC with an NVIDIA card and an AMD card (the lent boxia: RTX 3080 + RX 6700 XT gfx1031, 11-15).

Before: `strata update --backend amd` said "The feed has no engine package for this PC" (the word amd was never turned into the hip family, and the
hardware the choice read stayed the NVIDIA card's), `strata components engine --backend amd` exited 5 ("this PC needs amd (requested)"), and
`strata components engines` marked the installed hip-rx6000 engine "not for this computer" although the PC has the card it is for.

Proves, with a faked two-card detection: amd / hip / hip-rx6000 / nvidia name their family; --backend amd reads the AMD card (gfx1031) and picks
the rx6000 package from a feed that also has the CUDA, rx9070, rx7900 and halo builds; resolve / status accept the make's word; the engines list marks
the AMD engine as one this computer can run (for its other card) and keeps "newest" on the engine of the make in use; switching to it keeps the
AMD choice the way `strata launch --backend amd` does."""
import argparse
import io
import json
import shutil
import unittest
from contextlib import redirect_stdout
from unittest import mock

from strata_cli import update_cmd
from strata_control import engine_component as ec, engine_library as lib, feed as feedmod, updater
from tests.terminal import update_fixture as fx
from tests.terminal.test_updater import UpdaterCase


def _detect_backend(environ=None):
    forced = ec.split_backend((environ or {}).get(ec.ENV_BACKEND, ""))[0]
    if forced:
        return forced, "environment"
    return "cuda", "nvidia-smi"                       # the NVIDIA card answers first, as on boxia


def _pkg(backend, variant, archs):
    return feedmod.EnginePackage(version="0.1.60", platform="linux-x86_64", backend=backend, variant=variant, archs=tuple(archs), min_driver={},
                                 min_runtime=None, build_id="x", url=f"strata-engine-0.1.60-linux-x86_64-{backend}-{variant}.tar.gz", format="tar.gz",
                                 size=1, sha256="0" * 64, manifest_sha256="0" * 64, installed_bytes=1)


FEED = [_pkg("cuda", "cuda13", ("sm_86", "sm_120")), _pkg("hip", "rx6000", ("gfx1030", "gfx1031", "gfx1032")),
        _pkg("hip", "rx9070", ("gfx1200", "gfx1201")), _pkg("hip", "rx7900", ("gfx1100", "gfx1101", "gfx1102")),
        _pkg("hip", "halo", ("gfx1151",))]


class MixedPcHardware(unittest.TestCase):
    def setUp(self):
        for target, name, value in ((ec, "detect_backend", None), (ec, "gpu_makes_present", {"cuda", "hip"}), (updater, "host_platform", "linux-x86_64"),
                                    (updater, "_gfx_names", ["gfx1031"]), (updater, "_amd_card_names", []), (ec, "runtime_version", None),
                                    (ec, "nvidia_driver_version", "580.95"), (ec, "cuda_archs_present", ["sm_86"])):
            patch = mock.patch.object(target, name, _detect_backend) if value is None else mock.patch.object(target, name, return_value=value)
            patch.start()
            self.addCleanup(patch.stop)

    def test_the_make_words_name_their_family(self):
        self.assertEqual(ec.split_backend("amd"), ("hip", ""))
        self.assertEqual(ec.split_backend(" AMD "), ("hip", ""))
        self.assertEqual(ec.split_backend("nvidia"), ("cuda", ""))
        self.assertEqual(ec.split_backend("intel"), ("sycl", ""))
        self.assertEqual(ec.split_backend("hip-rx6000"), ("hip", "rx6000"))
        self.assertEqual(ec.split_backend("cuda"), ("cuda", ""))

    def test_without_a_choice_the_nvidia_card_decides_as_before(self):
        hw = updater.detect_hardware({})
        self.assertEqual((hw["backend"], hw["gfx"], hw["cuda_archs"]), ("cuda", [], ["sm_86"]))
        pick, _ = feedmod.select_engine(FEED, hw)
        self.assertEqual((pick.backend, pick.variant), ("cuda", "cuda13"))

    def test_backend_amd_reads_the_amd_card_and_picks_the_rx6000_package(self):
        hw = update_cmd.requested_hardware("amd", environ={})
        self.assertEqual((hw["backend"], hw["gfx"], hw["backend_source"]), ("hip", ["gfx1031"], "requested"))
        pick, reasons = feedmod.select_engine(FEED, hw)
        self.assertIsNotNone(pick, reasons)
        self.assertEqual((pick.backend, pick.variant), ("hip", "rx6000"))

    def test_backend_hip_and_the_package_name_pick_the_same(self):
        for word in ("hip", "hip-rx6000", "HIP"):
            hw = update_cmd.requested_hardware(word, environ={})
            pick, reasons = feedmod.select_engine(FEED, hw)
            self.assertEqual((pick.backend, pick.variant), ("hip", "rx6000"), (word, reasons))
        self.assertEqual(update_cmd.requested_hardware("hip-rx6000", environ={})["variant_wanted"], "rx6000")

    def test_backend_nvidia_stays_on_the_nvidia_card(self):
        hw = update_cmd.requested_hardware("nvidia", environ={})
        self.assertEqual((hw["backend"], hw["cuda_archs"], hw["gfx"]), ("cuda", ["sm_86"], []))

    def test_a_word_that_names_no_family_is_still_answered_by_the_feed(self):
        hw = update_cmd.requested_hardware("vulkan", environ={})
        pick, reasons = feedmod.select_engine(FEED, hw)
        self.assertIsNone(pick)
        self.assertTrue(any("needs vulkan" in r for r in reasons))

    def test_the_make_guard_lets_the_named_family_through(self):
        ctx = mock.Mock(hardware=update_cmd.requested_hardware("amd", environ={}))
        self.assertIsNone(updater.make_guard(ctx))


class MixedPcEngines(UpdaterCase):
    """The installed engines on the two-card PC: the CUDA engine the PC picked first, and the hip-rx6000 one installed with --backend amd."""

    def setUp(self):
        if fx.local_family()["backend"] != "cuda":
            fx.cuda_pc(self)                                # (#4138) these cases are about CUDA (and AMD) packages: on a Mac they speak about a Linux PC
        super().setUp()
        cuda = self.install_current("0.1.60")
        # the hip-rx6000 package: the stub CUDA package with an AMD manifest (the packager wants a real AMD stage, its hipBLASLt version)
        self.amd = cuda.parent / f"0.1.60-{self.platform}-hip-rx6000"
        shutil.copytree(cuda, self.amd)
        manifest = json.loads((self.amd / ec.MANIFEST).read_text())
        manifest.update(backend="hip", variant="rx6000", archs=["gfx1030", "gfx1031", "gfx1032"], min_driver={})
        (self.amd / ec.MANIFEST).write_text(json.dumps(manifest))
        for target, name, value in ((lib, "_process_uses", False), (ec, "hip_archs_present", ["gfx1031"]), (ec, "gpu_makes_present", {"cuda", "hip"})):
            patch = mock.patch.object(target, name, return_value=value)
            patch.start()
            self.addCleanup(patch.stop)
        self.settings = self.root / "settings"
        patch = mock.patch("strata_control.discovery.own_settings_file", return_value=self.settings / "settings.json")
        patch.start()
        self.addCleanup(patch.stop)

    def rows(self, hardware):
        return {e["folder"]: e for e in lib.rows_of(lib.list_engines(self.ctx(hardware=hardware)), lib.QWEN)}

    def test_the_amd_engine_is_one_this_computer_can_run(self):
        rows = self.rows({**self.hardware, "makes": ["cuda", "hip"]})
        amd = rows[self.amd.name]
        self.assertTrue(amd["fits"] and amd["other_card"] and amd["why_not"] is None, amd)
        self.assertFalse(amd["active"] or amd["newest"], "the newest is the one of the make a start uses")
        cuda = next(r for r in rows.values() if r["backend"] == "cuda")
        self.assertTrue(cuda["active"] and cuda["newest"] and not cuda["other_card"])

    def test_on_a_pc_without_an_amd_card_it_stays_for_a_different_kind_of_card(self):
        rows = self.rows({**self.hardware, "makes": ["cuda"]})
        amd = rows[self.amd.name]
        self.assertFalse(amd["fits"])
        self.assertEqual(amd["why_code"], "gpu_make")

    def test_with_backend_amd_the_amd_engine_is_the_one_in_use(self):
        rows = self.rows({**self.hardware, "backend": "hip", "gfx": ["gfx1031"], "makes": ["cuda", "hip"]})
        self.assertTrue(rows[self.amd.name]["active"] and rows[self.amd.name]["newest"] and not rows[self.amd.name]["other_card"])

    def test_resolve_and_status_take_the_make_word(self):
        self.assertEqual(ec.resolve(self.components, backend="amd").root.name, self.amd.name)
        self.assertEqual(ec.resolve(self.components, backend="hip-rx6000").root.name, self.amd.name)
        report = ec.status(self.components, backend="amd")
        self.assertTrue(report["usable"], report["notice"])
        self.assertEqual((report["backend"], report["engine"]["backend"]), ("hip", "hip"))

    def test_switching_to_the_amd_engine_keeps_the_amd_choice(self):
        hardware = {**self.hardware, "makes": ["cuda", "hip"]}
        with mock.patch.object(updater, "detect_hardware", side_effect=lambda environ=None: {**hardware, "backend": ec.saved_backend() or "cuda"}):
            out = lib.use_engine(self.ctx(hardware=hardware), "qwen", self.amd.name)
        self.assertEqual(out.get("backend_saved"), "hip")
        self.assertEqual(json.loads((self.settings / "settings.json").read_text())[ec.SAVED_KEY], "hip")
        self.assertEqual(ec.resolve(self.components, backend="hip").root.name, self.amd.name)

    def test_the_engines_listing_with_backend_amd_from_the_terminal(self):
        args = argparse.Namespace(components_dir=str(self.components), backend="amd", wait=None, force=False, model_config=None, output="json",
                                  components_command="engines")
        out = io.StringIO()
        with mock.patch.object(updater, "detect_hardware", side_effect=lambda environ=None: {**self.hardware, "makes": ["cuda", "hip"],
                                                                                             "backend": ec.split_backend((environ or {}).get(ec.ENV_BACKEND, "cuda"))[0]}), \
                redirect_stdout(out):
            code = update_cmd.run(args)
        self.assertEqual(code, 0)
        rows = {e["folder"]: e for f in json.loads(out.getvalue())["families"] for e in f["engines"]}
        self.assertTrue(rows[self.amd.name]["active"] and rows[self.amd.name]["fits"])
        shutil.rmtree(self.settings, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
