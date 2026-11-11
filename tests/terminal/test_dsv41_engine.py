"""DeepSeek's engine as its own signed package (feed member ``dsv41``): fetched with resume, verified, unpacked beside the engine, and the
engine's own files are not touched.  A local signed feed on a loopback server, a stub strata-dsv41, a throwaway key."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "packaging"))
import feed_tool  # noqa: E402
import package_engine  # noqa: E402
from strata_control import components, dsv41_engine, engine_component as ec, signing  # noqa: E402
from strata_control.targets import host_platform  # noqa: E402
from tests.terminal import update_fixture as fx  # noqa: E402

EXE = dsv41_engine.exe_name()


def build(parent: Path, version: str, key: dict, padding: int = 0) -> Path:
    src = parent / f"src-{version}"
    src.mkdir(parents=True)
    (src / EXE).write_bytes(b"stub strata-dsv41 " + (os.urandom(padding) if padding else b"x" * 100))
    library = "cudart64_13.dll" if os.name == "nt" else "libcudart.so.13"
    (src / library).write_bytes(b"a runtime library " * 50)
    out = parent / "built"
    out.mkdir(exist_ok=True)
    folder = package_engine.build(src, "cuda", version, components.host_platform(), out, entry=EXE, archs=("sm_86",),
                                  min_driver={"nvidia": "580.65"}, build_id=f"dsv41-{version}", variant="cuda13")
    feed_tool.sign_package(folder, key)
    return folder


class Dsv41Engine(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="strata-dsv41-eng-")
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.key = fx.make_key()
        trust = self.root / "trust.json"
        trust.write_text(json.dumps({"keys": [signing.public_part(self.key)]}))
        for p in (mock.patch.dict(os.environ, {"STRATA_UPDATE_TRUSTED_KEYS": str(trust), "STRATA_COMPONENTS_DIR": str(self.root / "c")}),
                  mock.patch.object(signing, "BUILTIN_KEYS", self.root / "none.json"),
                  mock.patch.object(ec, "cuda_archs_present", return_value=["sm_86"]),
                  mock.patch.object(ec, "nvidia_driver_version", return_value="610.47")):
            p.start()
            self.addCleanup(p.stop)
        self.addCleanup(ec.register_signature_verifier, None)
        self.hw = {"platform": host_platform(), "backend": "cuda", "runtime": None, "nvidia_driver": "610.47", "cuda_archs": ["sm_86"], "gfx": []}
        self.release = self.root / "release"
        self.release.mkdir()
        self.server = fx.FeedServer(self.release)
        self.server.__enter__()
        self.addCleanup(self.server.__exit__)
        self.eng = self.root / "data" / "engine"                    # a stand-in Qwen engine folder: its hash must not change
        self.eng.mkdir(parents=True)
        for n in ("strata", "BUILD.json"):
            (self.eng / n).write_bytes(os.urandom(500))
        self.before = self.tree(self.eng)

    @staticmethod
    def tree(d: Path) -> str:
        h = hashlib.sha256()
        for p in sorted(Path(d).rglob("*")):
            if p.is_file():
                h.update(p.name.encode() + p.read_bytes())
        return h.hexdigest()

    def publish(self, version="0.1.49", padding=0):
        shutil.rmtree(self.release, ignore_errors=True)
        (self.release / "dsv41").mkdir(parents=True)
        folder = build(self.root / f"pub-{version}", version, self.key, padding)
        feed_tool.pack_engine(folder, self.release / "dsv41", url_prefix="dsv41/")
        feed_tool.build_feed(self.release, key=self.key, sequence=1, dsv41_dir=self.release / "dsv41")

    def install(self, lines=None):
        out = [] if lines is None else lines
        return dsv41_engine.install(self.root / "data" / "engine-dsv41", self.server.url + "/strata-feed.json", hardware=self.hw, emit=out.append)

    def test_downloads_verifies_installs_beside_the_engine(self):
        self.publish()
        lines = []
        exe = self.install(lines)
        self.assertIsNotNone(exe, lines)
        self.assertEqual(exe.parent.parent, self.root / "data" / "engine-dsv41")
        self.assertTrue(exe.is_file())
        self.assertEqual(dsv41_engine.find(self.root / "data" / "engine-dsv41"), exe)
        self.assertEqual(self.tree(self.eng), self.before, "the Qwen engine's files must not change")
        self.assertFalse((self.root / "data" / "engine-dsv41.work").exists())

    def test_a_cut_download_resumes(self):
        self.publish(padding=3_000_000)
        name = next((self.release / "dsv41").glob("*.tar.gz")).name
        self.server.cut[name] = 800_000
        lines = []
        exe = self.install(lines)                                    # the connection broke part-way: the same run continues from the byte it reached
        self.assertIsNotNone(exe, lines)                             # (the network rule: resumed while bytes come, strata_control/netpolicy.py)
        ranges = [r[2] for r in self.server.requests if r[1].endswith(".tar.gz")]
        self.assertEqual(ranges, [None, "bytes=800000-"], "exactly one more request, from the byte the cut reached: no byte twice")
        self.assertEqual(self.tree(self.eng), self.before)

    def test_no_package_for_this_pc_is_a_plain_line_not_a_crash(self):
        self.publish()
        self.hw = {**self.hw, "cuda_archs": ["sm_61"], "gfx": []}
        lines = []
        with mock.patch.object(ec, "cuda_archs_present", return_value=["sm_61"]):
            self.assertIsNone(self.install(lines))
        self.assertTrue(lines)
        self.assertEqual(self.tree(self.eng), self.before)

    def test_a_feed_signed_with_another_key_is_refused(self):
        self.publish()
        self.key = fx.make_key()                                     # (the trust file still names the first key)
        shutil.rmtree(self.release / "dsv41")
        self.publish_other = None
        folder = build(self.root / "other", "0.1.50", self.key)
        (self.release / "dsv41").mkdir()
        feed_tool.pack_engine(folder, self.release / "dsv41", url_prefix="dsv41/")
        feed_tool.build_feed(self.release, key=self.key, sequence=2, dsv41_dir=self.release / "dsv41")
        lines = []
        self.assertIsNone(self.install(lines))
        self.assertIn("could not be fetched", " ".join(lines))
        self.assertFalse((self.root / "data" / "engine-dsv41").exists() and any((self.root / "data" / "engine-dsv41").iterdir()))


class ManualPackage(Dsv41Engine):
    """A package the person downloaded by hand: a .tar.gz / .zip or the unpacked folder, checked before it is installed."""

    def archive(self, version="0.1.49"):
        folder = build(self.root / f"man-{version}", version, self.key)
        out = self.root / "dl"
        out.mkdir(exist_ok=True)
        entry = feed_tool.pack_engine(folder, out, prefix="strata-dsv41")
        return folder, out / Path(entry["url"]).name

    def test_a_downloaded_archive_is_checked_and_installed_beside_the_engine(self):
        _folder, archive = self.archive()
        lines = []
        exe = dsv41_engine.install_package(self.root / "data" / "engine-dsv41", archive, emit=lines.append)
        self.assertIsNotNone(exe, lines)
        self.assertTrue(exe.is_file())
        self.assertIn("signed, every file matches", " ".join(lines))
        self.assertEqual(self.tree(self.eng), self.before)

    def test_an_unpacked_folder_works_too(self):
        folder, _ = self.archive()
        exe = dsv41_engine.install_package(self.root / "data" / "engine-dsv41", folder)
        self.assertIsNotNone(exe)

    def test_a_damaged_file_is_refused_and_nothing_is_installed(self):
        folder, _ = self.archive()
        (folder / EXE).write_bytes(b"tampered")
        lines = []
        self.assertIsNone(dsv41_engine.install_package(self.root / "data" / "engine-dsv41", folder, emit=lines.append))
        self.assertIn("did not pass its check", " ".join(lines))
        dest = self.root / "data" / "engine-dsv41"
        self.assertFalse(dest.exists() and any(p.is_dir() for p in dest.iterdir()))

    def test_a_package_signed_by_an_unknown_key_is_refused(self):
        other = fx.make_key()
        folder = build(self.root / "foreign", "0.1.49", other)
        lines = []
        self.assertIsNone(dsv41_engine.install_package(self.root / "data" / "engine-dsv41", folder, emit=lines.append))
        self.assertIn("did not pass its check", " ".join(lines))

    def test_the_strata_engine_package_is_not_taken_for_deepseeks(self):
        src = self.root / "qsrc"
        src.mkdir()
        (src / "strata").write_bytes(b"qwen engine")
        folder = package_engine.build(src, "cuda", "0.1.49", components.host_platform(), self.root / "qout", entry="strata", archs=("sm_86",),
                                      min_driver={"nvidia": "580.65"}, build_id="q", variant="cuda13")
        feed_tool.sign_package(folder, self.key)
        lines = []
        self.assertIsNone(dsv41_engine.install_package(self.root / "data" / "engine-dsv41", folder, emit=lines.append))
        self.assertIn("not the DeepSeek engine", " ".join(lines))

    def test_a_missing_or_wrong_file_is_said_plainly(self):
        lines = []
        self.assertIsNone(dsv41_engine.install_package(self.root / "x", self.root / "nope.tar.gz", emit=lines.append))
        self.assertIn("is not there", " ".join(lines))
        junk = self.root / "notes.txt"
        junk.write_text("x")
        self.assertIsNone(dsv41_engine.install_package(self.root / "x", junk, emit=lines.append))
        self.assertIn("not a package file", " ".join(lines))

    def test_files_put_in_the_folder_are_found(self):
        _f, archive = self.archive()
        dest = self.root / "data" / "engine-dsv41"
        dest.mkdir(parents=True)
        shutil.copy(archive, dest / archive.name)
        (dest / "readme.txt").write_text("x")
        self.assertEqual([p.name for p in dsv41_engine.pending(dest)], [archive.name])


class HipPackageLibraryPath(unittest.TestCase):
    """A Windows HIP DeepSeek package keeps the ROCm DLLs in rocm/bin (hipBLAS / rocBLAS), exactly like the Qwen rdna3 engine package,
    so the engine's folder list must carry it or a clean PC starts with 0xC0000135 (found on strix, 2026-10-25)."""

    def package(self, build_json=None) -> Path:
        tmp = tempfile.TemporaryDirectory(prefix="strata-dsv41-hip-")
        self.addCleanup(tmp.cleanup)
        folder = Path(tmp.name) / "0.1.51-windows-x86_64-hip-rdna3"
        (folder / "rocm" / "bin").mkdir(parents=True)
        (folder / "rocm" / "bin" / "rocblas.dll").write_bytes(b"x")
        (folder / EXE).write_bytes(b"stub")
        if build_json is not None:
            (folder / "BUILD.json").write_text(json.dumps(build_json), encoding="utf-8")
        return folder

    def test_rocm_bin_comes_after_the_package_folder(self):
        folder = self.package({"lib_dirs": ["rocm/bin"]})
        self.assertEqual(dsv41_engine.lib_dirs(folder / EXE), [str(folder), str(folder / "rocm" / "bin")])
        env = dsv41_engine.start_env(folder / EXE, {dsv41_engine.LIB_VAR: "C:\\Windows"})      # PATH on Windows, LD_LIBRARY_PATH elsewhere
        first = env[dsv41_engine.LIB_VAR].split(os.pathsep)[:2]
        self.assertEqual(first, [str(folder), str(folder / "rocm" / "bin")])

    def test_a_cuda_package_needs_only_its_folder(self):
        folder = self.package(None)
        self.assertEqual(dsv41_engine.lib_dirs(folder / EXE), [str(folder)])

    def test_a_package_folder_missing_from_disk_is_skipped(self):
        folder = self.package({"lib_dirs": ["rocm/bin", "nope"]})
        self.assertEqual(dsv41_engine.lib_dirs(folder / EXE), [str(folder), str(folder / "rocm" / "bin")])

    def test_the_feed_offers_a_windows_hip_package_to_strix_halo_only(self):
        from strata_control import feed as feedmod
        p = {"version": "0.1.51", "platform": "windows-x86_64", "backend": "hip", "variant": "rdna3", "archs": ["gfx1151"],
             "min_driver": {}, "min_runtime": None, "build_id": "dsv41-hip", "format": "tar.gz", "url": "strata-dsv41-0.1.51-windows-x86_64-hip-rdna3.tar.gz",
             "size": 1, "sha256": "a" * 64, "manifest_sha256": "b" * 64, "installed_bytes": 2, "notes": ""}
        packages = [feedmod._engine_package(p)]
        halo = {"platform": "windows-x86_64", "backend": "hip", "runtime": None, "nvidia_driver": None, "cuda_archs": [], "gfx": ["gfx1151"]}
        best, _why = feedmod.select_engine(packages, halo)
        self.assertIsNotNone(best)
        self.assertEqual((best.platform, best.backend, best.variant), ("windows-x86_64", "hip", "rdna3"))
        for gfx in ("gfx1100", "gfx1201", "gfx1030"):
            other = {**halo, "gfx": [gfx]}
            self.assertIsNone(feedmod.select_engine(packages, other)[0], gfx)


class HipCards(unittest.TestCase):
    """0.2.9 shipped one Linux AMD DeepSeek package, for gfx1200/gfx1201 (RX 9070, R9700).  A 7900 XTX (gfx1100) is not covered; a PC whose card could
    not be named used to get the package without a word (feed.gfx_covers lets an unknown card through)."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="strata-dsv41-hip-cards-")
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.key = fx.make_key()
        trust = self.root / "trust.json"
        trust.write_text(json.dumps({"keys": [signing.public_part(self.key)]}))
        for p in (mock.patch.dict(os.environ, {"STRATA_UPDATE_TRUSTED_KEYS": str(trust), "STRATA_COMPONENTS_DIR": str(self.root / "c")}),
                  mock.patch.object(signing, "BUILTIN_KEYS", self.root / "none.json")):
            p.start()
            self.addCleanup(p.stop)
        self.addCleanup(ec.register_signature_verifier, None)
        self.release = self.root / "release"
        self.release.mkdir()
        self.server = fx.FeedServer(self.release)
        self.server.__enter__()
        self.addCleanup(self.server.__exit__)
        src = self.root / "src"
        src.mkdir()
        (src / EXE).write_bytes(b"stub strata-dsv41 hip")
        (src / "libamdhip64.so.7").write_bytes(b"a runtime library " * 50)
        self.src = src

    def package(self):
        with mock.patch("strata_control.hipblaslt_tuning.check_stage", return_value=[]):
            folder = package_engine.build(self.src, "hip", "0.1.55", components.host_platform(), self.root / "built", entry=EXE,
                                          archs=("gfx1200", "gfx1201"), build_id="dsv41-hip", variant="rx9070")
        feed_tool.sign_package(folder, self.key)
        return folder

    def hw(self, gfx):
        return {"platform": host_platform(), "backend": "hip", "runtime": None, "nvidia_driver": None, "cuda_archs": [], "gfx": gfx}

    def publish(self):
        (self.release / "dsv41").mkdir(parents=True)
        feed_tool.pack_engine(self.package(), self.release / "dsv41", url_prefix="dsv41/")
        feed_tool.build_feed(self.release, key=self.key, sequence=1, dsv41_dir=self.release / "dsv41")

    def install(self, gfx, lines):
        return dsv41_engine.install(self.root / "data" / "engine-dsv41", self.server.url + "/strata-feed.json", hardware=self.hw(gfx), emit=lines.append)

    def test_a_card_nothing_could_name_gets_the_package_and_one_plain_line(self):
        self.publish()
        lines = []
        exe = self.install([], lines)
        self.assertIsNotNone(exe, lines)
        text = " ".join(lines)
        self.assertIn("RX 9070 and R9700 (gfx1200/1201)", text)
        self.assertIn("your graphics card wasn't recognised", text)

    def test_a_7900_xtx_is_told_which_cards_the_package_is_for_and_nothing_is_installed(self):
        self.publish()
        lines = []
        self.assertIsNone(self.install(["gfx1100"], lines))
        text = " ".join(lines)
        self.assertIn("RX 9070 and R9700", text)
        self.assertIn("RX 7000", text)
        self.assertIn("gfx1100", text)
        self.assertFalse((self.root / "data" / "engine-dsv41").exists() and any((self.root / "data" / "engine-dsv41").iterdir()))

    def test_a_card_the_package_holds_code_for_hears_nothing_about_cards(self):
        self.publish()
        lines = []
        self.assertIsNotNone(self.install(["gfx1201"], lines), lines)
        self.assertNotIn("recognised", " ".join(lines))
        self.assertNotIn("RX 7000", " ".join(lines))

    def test_a_hand_installed_package_is_checked_against_the_cards_when_they_are_known(self):
        folder = self.package()
        from strata_control import updater
        for gfx, installed, words in ((["gfx1100"], False, "Nothing was installed"), ([], True, "wasn't recognised"), (["gfx1201"], True, "")):
            lines = []
            dest = self.root / f"data-{len(gfx)}-{gfx[0] if gfx else 'none'}" / "engine-dsv41"
            with mock.patch.object(updater, "detect_hardware", return_value=self.hw(gfx)):
                exe = dsv41_engine.install_package(dest, folder, emit=lines.append)
            self.assertEqual(exe is not None, installed, (gfx, lines))
            self.assertIn(words, " ".join(lines), (gfx, lines))
            if not installed:
                self.assertIn("RX 7000", " ".join(lines))

    def test_the_package_names_its_cards(self):
        folder = self.package()
        self.assertEqual(dsv41_engine.package_archs(folder / EXE), ("gfx1200", "gfx1201"))


class HipTwoFamilies(HipCards):
    """The feed holds a DeepSeek package per card family on Linux AMD, as the Qwen engine does: rx9070 (gfx1200, gfx1201) and rx7900 (gfx1100, gfx1101,
    gfx1102; built, not run on a real RX 7000 card).  Each card gets its own; a card nothing could name gets neither by guess."""
    # (HipCards' own tests are not repeated)
    test_a_card_nothing_could_name_gets_the_package_and_one_plain_line = None
    test_a_7900_xtx_is_told_which_cards_the_package_is_for_and_nothing_is_installed = None
    test_a_card_the_package_holds_code_for_hears_nothing_about_cards = None
    test_a_hand_installed_package_is_checked_against_the_cards_when_they_are_known = None
    test_the_package_names_its_cards = None

    def package_of(self, variant, archs):
        src = self.root / f"src-{variant}"
        src.mkdir()
        (src / EXE).write_bytes(b"stub strata-dsv41 hip " + variant.encode())
        (src / "libamdhip64.so.7").write_bytes(b"a runtime library " * 50)
        with mock.patch("strata_control.hipblaslt_tuning.check_stage", return_value=[]):
            folder = package_engine.build(src, "hip", "0.1.55", components.host_platform(), self.root / "built", entry=EXE, archs=archs,
                                          build_id=f"dsv41-hip-{variant}", variant=variant)
        feed_tool.sign_package(folder, self.key)
        return folder

    def publish_both(self):
        (self.release / "dsv41").mkdir(parents=True)
        for variant, archs in (("rx9070", ("gfx1200", "gfx1201")), ("rx7900", ("gfx1100", "gfx1101", "gfx1102"))):
            feed_tool.pack_engine(self.package_of(variant, archs), self.release / "dsv41", url_prefix="dsv41/")
        feed_tool.build_feed(self.release, key=self.key, sequence=1, dsv41_dir=self.release / "dsv41")

    def test_each_card_gets_the_package_of_its_family(self):
        self.publish_both()
        for gfx, variant in (("gfx1100", "rx7900"), ("gfx1101", "rx7900"), ("gfx1102", "rx7900"), ("gfx1200", "rx9070"), ("gfx1201", "rx9070")):
            lines = []
            dest = self.root / f"data-{gfx}"
            exe = dsv41_engine.install(dest / "engine-dsv41", self.server.url + "/strata-feed.json", hardware=self.hw([gfx]), emit=lines.append)
            self.assertIsNotNone(exe, (gfx, lines))
            self.assertTrue(exe.parent.name.endswith("-hip-" + variant), (gfx, exe))
            self.assertEqual(dsv41_engine.package_archs(exe), {"rx7900": ("gfx1100", "gfx1101", "gfx1102"), "rx9070": ("gfx1200", "gfx1201")}[variant])
            self.assertNotIn("recognised", " ".join(lines))

    def test_a_pc_with_a_card_of_each_family_gets_one_of_them_without_a_refusal(self):
        self.publish_both()
        lines = []
        self.assertIsNotNone(dsv41_engine.install(self.root / "data" / "engine-dsv41", self.server.url + "/strata-feed.json",
                                                  hardware=self.hw(["gfx1100", "gfx1201"]), emit=lines.append), lines)

    def test_a_card_nothing_could_name_is_not_guessed_when_two_families_fit(self):
        self.publish_both()
        lines = []
        self.assertIsNone(self.install([], lines))
        self.assertIn("cannot tell which AMD build", " ".join(lines))
        self.assertFalse((self.root / "data" / "engine-dsv41").exists() and any((self.root / "data" / "engine-dsv41").iterdir()))

    def test_a_card_neither_family_covers_is_told_both(self):
        self.publish_both()
        lines = []
        self.assertIsNone(self.install(["gfx1030"], lines))
        text = " ".join(lines)
        self.assertIn("RX 7000 and RX 9000", text)
        self.assertIn("gfx1030", text)


if __name__ == "__main__":
    unittest.main()
