"""The picture files of a model that is already installed (strata_control/pictures.py): found or downloaded once, the data folder and the model's
other files left exactly as they are, no internet answered in a sentence, and the run config's ``vision`` section worked out for the engine in use.

The file is pinned (size and SHA-256), so the tests swap in a tiny pinned file of their own and a folder on disk for the download address."""

from __future__ import annotations

import hashlib
import io
import json
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

import setup
from strata_control import acquisition, pictures

ROOT = Path(__file__).resolve().parents[2]
BODY = b"mmproj-bytes-" * 5000
TINY = pictures.Projector("mmproj-Tiny.gguf", len(BODY), hashlib.sha256(BODY).hexdigest(), "owner/repo", "rev1")
NAME = TINY.name


def tree(root: Path) -> dict:
    """Every file under root with its size and modified time: what must not change."""
    out = {}
    for p in sorted(root.rglob("*")):
        if p.is_file():
            st = p.stat()
            out[p.relative_to(root).as_posix()] = (st.st_size, st.st_mtime_ns)
    return out


class Layout:
    """A model in its own folder (not <data>/models/<tag>) with its pack and draft layer in a data folder that is not the models folder."""

    def __init__(self, base: Path, *, ordinary: bool = False, with_vision: bool = False):
        self.base = base
        self.data = base / ("Strata-data" if ordinary else "own-data")
        self.model_dir = (self.data / "models" / "IQ3_S") if ordinary else (base / "downloads" / "IQ3_S")
        self.model_dir.mkdir(parents=True)
        self.gguf = self.model_dir / "Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf"
        self.gguf.write_bytes(b"GGUF")
        (self.model_dir / "Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf").write_bytes(b"GGUF")
        self.pack = self.data / "packs" / "iq3_s"
        (self.pack / "tokenizer").mkdir(parents=True)
        (self.pack / "native_experts.txt").write_text("x")
        self.mtp = self.data / "mtp" / "rt"
        self.mtp.mkdir(parents=True)
        (self.mtp / "experts.bin").write_bytes(b"e")
        (self.data / "mtp" / "mtp-q2_0.gguf").write_bytes(b"m")
        self.engine = base / "engine"
        self.engine.mkdir()
        exe = "strata-vision.exe" if os.name == "nt" else "strata-vision"
        (self.engine / exe).write_bytes(b"x")
        self.reader = self.engine / exe
        self.config = base / "install" / "strata-iq3_s.json"
        self.config.parent.mkdir()
        cfg = {"exe": "component:strata-engine", "model_name": "qwen3.8-flash-next-iq3_s",
               "args": ["--pack", str(self.pack), "--native", str(self.gguf), "--mtp", str(self.mtp), "--max-context", "8192"]}
        if with_vision:
            cfg["vision"] = {"exe": str(self.reader), "mmproj": str(self.data / "models" / NAME), "model": str(self.gguf), "gpu": False, "max_tokens": 450}
        self.config.write_text(json.dumps(cfg))
        self.mirror = base / "mirror"
        served = self.mirror / "owner" / "repo" / "resolve" / "rev1"
        served.mkdir(parents=True)
        (served / NAME).write_bytes(BODY)

    def fetch(self, **kw):
        kw.setdefault("engine_dir", self.engine)
        kw.setdefault("hardware", {"vram_gb": 12, "make": "nvidia", "arch": 120})
        kw.setdefault("endpoint", self.mirror.as_uri())
        kw.setdefault("online", lambda url: True)
        with mock.patch.dict(pictures.PROJECTOR_BY_FAMILY, {"qwen": TINY}), mock.patch.dict(os.environ, {"STRATA_NO_ACQUIRE": ""}):
            return pictures.fetch(self.config, **kw)


class FetchInOwnFolder(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.base = Path(self._tmp.name)

    def test_the_file_goes_to_the_data_folder_of_the_run_config_and_nothing_else_changes(self):
        lay = Layout(self.base)
        before = tree(lay.data) | {f"gguf/{k}": v for k, v in tree(lay.model_dir).items()}
        cfg_before = lay.config.read_bytes()
        seen = []
        result = lay.fetch(progress=lambda step, d, t: seen.append((step, d, t)))
        self.assertTrue(result["fetched"])
        self.assertEqual(Path(result["mmproj"]), lay.data / "models" / NAME, "beside the pack and draft layer the model uses, not the app's models folder")
        self.assertEqual(result["data_dir"], str(lay.data))
        self.assertEqual((lay.data / "models" / NAME).read_bytes(), BODY)
        self.assertFalse((lay.data / "models" / (NAME + ".part")).exists())
        self.assertIn("Downloading the picture files…", [s for s, _d, _t in seen])
        after = tree(lay.data) | {f"gguf/{k}": v for k, v in tree(lay.model_dir).items()}
        added = sorted(set(after) - set(before))
        self.assertEqual(added, [f"models/{NAME}", f"models/{NAME}.done"], "only the picture file and its finish mark")
        for key, value in before.items():
            self.assertEqual(after[key], value, f"{key} was left alone")
        self.assertEqual(lay.config.read_bytes(), cfg_before, "the run config is the caller's to write")
        self.assertFalse((self.base / "Strata-data").exists(), "no second data folder")

    @unittest.skipIf(sys.platform == "darwin", 'NVIDIA / AMD picture-reader rules, asked on a Mac host')
    def test_the_section_is_the_one_setup_writes_for_this_card(self):
        lay = Layout(self.base)
        small = lay.fetch(hardware={"vram_gb": 12, "make": "nvidia", "arch": 120})["vision"]
        self.assertEqual((small["gpu"], small["max_tokens"]), (False, 300), "under 24 GB the CPU reader")
        self.assertEqual(small["exe"], str(lay.reader))
        self.assertEqual(small["model"], str(lay.gguf))
        self.assertEqual(small["mmproj"], str(lay.data / "models" / NAME))
        self.assertGreaterEqual(small["threads"], 1)
        big = lay.fetch(hardware={"vram_gb": 48, "make": "nvidia", "arch": 86})["vision"]
        self.assertEqual((big["gpu"], big["max_tokens"]), (True, 1024), "from 24 GB the GPU reader")
        self.assertNotIn("threads", big)
        cpu = lay.fetch(asked="cpu", hardware={"vram_gb": 48, "make": "nvidia", "arch": 86})["vision"]
        self.assertFalse(cpu["gpu"], "what the person asked for is kept")

    def test_a_copy_already_on_the_computer_is_used_where_it_is(self):
        lay = Layout(self.base)
        lay.model_dir.parent.joinpath(NAME).write_bytes(BODY)       # one folder up from the model files: the repository layout
        result = lay.fetch(online=lambda url: self.fail("no network for a file that is here"))
        self.assertFalse(result["fetched"])
        self.assertEqual(Path(result["mmproj"]), lay.model_dir.parent / NAME)
        self.assertFalse((lay.data / "models").exists(), "nothing is written")

    def test_a_copy_with_the_wrong_bytes_is_not_used_and_not_deleted(self):
        lay = Layout(self.base)
        wrong = lay.data / "models" / NAME
        wrong.parent.mkdir()
        wrong.write_bytes(b"x" * len(BODY))                          # right size, wrong content
        result = lay.fetch()
        self.assertTrue(result["fetched"], "downloaded because the copy there is not the pinned file")
        self.assertEqual(wrong.read_bytes(), BODY, "the bad copy was replaced by the checked download")

    @unittest.skipIf(sys.platform == "darwin", 'NVIDIA / AMD picture-reader rules, asked on a Mac host')
    def test_the_path_the_run_config_names_is_kept_when_its_folder_is_there(self):
        lay = Layout(self.base, with_vision=True)
        named = Path(json.loads(lay.config.read_text())["vision"]["mmproj"])
        named.parent.mkdir(parents=True, exist_ok=True)
        result = lay.fetch()
        self.assertEqual(Path(result["mmproj"]), named)
        self.assertEqual(result["vision"]["max_tokens"], 450, "what the person set for the same reader is kept")
        cfg = json.loads(lay.config.read_text())
        cfg["vision"]["gpu"] = True                                  # an earlier value for the other reader does not carry over
        lay.config.write_text(json.dumps(cfg))
        self.assertEqual(lay.fetch()["vision"]["max_tokens"], 300)

    def test_an_ordinary_install_keeps_the_file_in_its_own_models_folder(self):
        lay = Layout(self.base, ordinary=True)
        result = lay.fetch()
        self.assertEqual(Path(result["mmproj"]), lay.data / "models" / NAME)
        self.assertEqual(result["data_dir"], str(lay.data))

    def test_the_data_folder_from_the_caller_is_used_when_the_config_names_none(self):
        lay = Layout(self.base)
        cfg = json.loads(lay.config.read_text())
        cfg["args"] = ["--native", str(lay.gguf)]                    # no --pack, no --mtp: nothing says where the data folder is
        lay.config.write_text(json.dumps(cfg))
        elsewhere = self.base / "chosen"
        elsewhere.mkdir()
        result = lay.fetch(data_dir=elsewhere)
        self.assertEqual(Path(result["mmproj"]), elsewhere / "models" / NAME)

    def test_no_internet_is_a_sentence_after_one_short_probe_and_nothing_is_written(self):
        lay = Layout(self.base)
        before = tree(lay.base)
        with self.assertRaises(pictures.PicturesError) as caught:
            lay.fetch(online=lambda url: False)
        self.assertEqual(caught.exception.code, "offline")
        self.assertEqual(str(caught.exception), "There is no internet connection, so the picture files can't be downloaded now.")
        self.assertEqual(tree(lay.base), before, "not even an empty file or folder")

    def test_a_download_that_does_not_match_is_deleted_and_said(self):
        lay = Layout(self.base)
        (lay.mirror / "owner" / "repo" / "resolve" / "rev1" / NAME).write_bytes(b"y" * len(BODY))
        with self.assertRaises(pictures.PicturesError) as caught:
            lay.fetch()
        self.assertEqual(caught.exception.code, "checksum")
        self.assertFalse((lay.data / "models" / NAME).exists())
        self.assertFalse((lay.data / "models" / (NAME + ".part")).exists())

    def test_a_half_download_is_continued(self):
        lay = Layout(self.base)
        partial = lay.data / "models" / (NAME + ".part")
        partial.parent.mkdir()
        partial.write_bytes(BODY[:1000])
        lay.fetch()
        self.assertEqual((lay.data / "models" / NAME).read_bytes(), BODY)

    def test_not_enough_disk_is_said_before_anything_is_fetched(self):
        lay = Layout(self.base)
        full = mock.Mock(free=10)
        with mock.patch.object(pictures.shutil, "disk_usage", return_value=full), self.assertRaises(pictures.PicturesError) as caught:
            lay.fetch(online=lambda url: self.fail("the space is checked first"))
        self.assertEqual(caught.exception.code, "disk")
        self.assertIn("1 GB", str(caught.exception))

    def test_a_start_never_downloads(self):
        lay = Layout(self.base)
        with mock.patch.dict(pictures.PROJECTOR_BY_FAMILY, {"qwen": TINY}), mock.patch.dict(os.environ, {"STRATA_NO_ACQUIRE": "1"}):
            with self.assertRaises(acquisition.AcquisitionRefused):
                pictures.fetch(lay.config, engine_dir=lay.engine, endpoint=lay.mirror.as_uri(), online=lambda url: True)
        self.assertFalse((lay.data / "models").exists())

    def test_a_model_this_step_does_not_know_is_left_alone(self):
        lay = Layout(self.base)
        cfg = json.loads(lay.config.read_text())
        cfg["model_name"] = "qwen3.6-35b-a3b-ud-q4_k_xl"
        lay.config.write_text(json.dumps(cfg))
        with self.assertRaises(pictures.PicturesError) as caught:
            lay.fetch()
        self.assertEqual(caught.exception.code, "unsupported")
        cfg.update({"model_name": "x", "arch": "deepseek41"})
        lay.config.write_text(json.dumps(cfg))
        with self.assertRaises(pictures.PicturesError) as caught:
            lay.fetch()
        self.assertEqual(caught.exception.code, "unsupported")

    def test_an_engine_without_a_picture_reader_is_a_clear_answer(self):
        lay = Layout(self.base)
        lay.reader.unlink()
        with self.assertRaises(pictures.PicturesError) as caught:
            lay.fetch()
        self.assertEqual(caught.exception.code, "no_reader")


class Families(unittest.TestCase):
    def test_the_family_comes_from_the_catalog_or_from_the_names(self):
        g = Path("Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf")
        self.assertEqual(pictures.family_of({}, g, "swift"), "swift")
        self.assertEqual(pictures.family_of({}, g, "qwen36"), None)
        self.assertEqual(pictures.family_of({}, g, "nvfp4"), None)
        self.assertEqual(pictures.family_of({"model_name": "iq3_s"}, g), "qwen", "a hand-made name still gets the original's file")
        self.assertEqual(pictures.family_of({"model_name": "swift-1.5-iq3_s"}, g), "swift")
        self.assertEqual(pictures.family_of({}, Path("Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf")), "swift")
        self.assertIsNone(pictures.family_of({"model_name": "qwen3.6-35b-a3b-q8_0"}, Path("Qwen3.6-35B-A3B-Q8_0.gguf")))
        self.assertIsNone(pictures.family_of({"arch": "deepseek41"}, None))

    def test_the_pins_are_the_ones_setup_downloads(self):
        for name, projector, repo in (("qwen", pictures.QWEN, "ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF"), ("swift", pictures.SWIFT, "ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF")):
            fam = setup.FAMILIES[name]
            self.assertEqual(projector.name, fam["mmproj"])
            self.assertEqual(projector.repo, repo)
            self.assertEqual(projector.revision, setup.HF_REVISIONS[repo])
            self.assertEqual(fam["mmproj_hf"], f"{setup.hf_endpoint()}/{repo}/resolve/{projector.revision}/")
            pins = {**setup.ISTA_FILES, **setup.SWIFT_FILES}[projector.name]
            self.assertEqual((projector.size, projector.sha256), tuple(pins))
        for name in ("coder", "unsloth", "gyro"):
            self.assertEqual(setup.FAMILIES[name]["mmproj"], pictures.PROJECTOR_BY_FAMILY[name].name, name)
            self.assertEqual(tuple(setup.CODER_FILES[pictures.QWEN.name]), (pictures.QWEN.size, pictures.QWEN.sha256), "the Coder's file is the original's")

    def test_the_address_follows_hf_endpoint_like_setup(self):
        with mock.patch.dict(os.environ, {"HF_ENDPOINT": "http://127.0.0.1:9/"}):
            self.assertEqual(pictures.QWEN.url(), f"http://127.0.0.1:9/{pictures.QWEN.repo}/resolve/{pictures.QWEN.revision}/{pictures.QWEN.name}")
        with mock.patch.dict(os.environ, {"HF_ENDPOINT": ""}):
            self.assertTrue(pictures.QWEN.url().startswith("https://huggingface.co/"))


class RunFolder(unittest.TestCase):
    def foot(self, *args):
        from strata_control import discovery
        return discovery.footprint({"args": list(args)})

    def test_the_folder_comes_from_where_the_pack_and_the_draft_layer_are(self):
        self.assertEqual(pictures.run_folder(self.foot("--pack", "/d/packs/iq3_s", "--native", "/other/IQ3_S/a.gguf")), Path("/d"))
        self.assertEqual(pictures.run_folder(self.foot("--mtp", "/x/mtp/rt", "--native", "/other/a.gguf")), Path("/x"))
        self.assertEqual(pictures.run_folder(self.foot("--native", "/data/models/IQ3_S/a.gguf")), Path("/data"))

    def test_a_pack_in_a_folder_of_another_shape_says_nothing_about_the_data_folder(self):
        self.assertIsNone(pictures.run_folder(self.foot("--pack", "/d/my-pack", "--native", "/other/IQ3_S/a.gguf")))
        self.assertIsNone(pictures.run_folder(self.foot("--mtp", "/x/draft", "--native", "/other/a.gguf")))
        self.assertIsNone(pictures.run_folder(self.foot("--mtp", "7")), "a draft depth is not a folder")


class ReaderRule(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.engine = Path(self._tmp.name)
        self.cpu = "strata-vision.exe" if os.name == "nt" else "strata-vision"
        self.hip = "strata-vision-hip.exe" if os.name == "nt" else "strata-vision-hip"

    def make(self, *names, build=None):
        for n in names:
            (self.engine / n).write_bytes(b"x")
        if build is not None:
            (self.engine / "BUILD.json").write_text(json.dumps(build))

    def section(self, asked="yes", **hw):
        return pictures.reader_section(self.engine, Path("/m/mm.gguf"), Path("/m/a.gguf"), asked, hw)

    @unittest.skipIf(sys.platform == "darwin", 'NVIDIA / AMD picture-reader rules, asked on a Mac host')
    def test_amd_gets_the_gpu_reader_with_the_cpu_one_behind_it_from_24_gb(self):
        self.make(self.cpu, self.hip, build={"vision_archs": "gfx1201;gfx1100"})
        s, _ = self.section(vram_gb=32, make="amd", arch="gfx1201")
        self.assertEqual((s["gpu"], Path(s["exe"]).name, Path(s["cpu_exe"]).name), (True, self.hip, self.cpu))
        s, _ = self.section(vram_gb=16, make="amd", arch="gfx1201")
        self.assertEqual((s["gpu"], Path(s["exe"]).name), (False, self.cpu), "a 16 GB card reads on the CPU")
        s, note = self.section("gpu", vram_gb=32, make="amd", arch="gfx1030")
        self.assertFalse(s["gpu"])
        self.assertIn("gfx1030", note)

    @unittest.skipIf(sys.platform == "darwin", 'NVIDIA / AMD picture-reader rules, asked on a Mac host')
    def test_nvidia_reader_the_build_has_no_code_for_runs_on_the_cpu(self):
        self.make(self.cpu, build={"vision_archs": "86;89", "ptx": False})
        s, note = self.section("gpu", vram_gb=48, make="nvidia", arch=75)
        self.assertFalse(s["gpu"])
        self.assertIn("sm_75", note)
        s, _ = self.section("gpu", vram_gb=48, make="nvidia", arch=86)
        self.assertTrue(s["gpu"])

    def test_off_and_no_reader(self):
        self.make()
        self.assertEqual(self.section("yes", vram_gb=12, make="nvidia")[0], None)
        self.make(self.cpu)
        self.assertIsNone(pictures.reader_section(self.engine, Path("m"), Path("g"), "no", {})[0])


class Command(unittest.TestCase):
    def test_the_command_speaks_json_lines_and_exits_with_the_sentence(self):
        with tempfile.TemporaryDirectory() as tmp:
            lay = Layout(Path(tmp))
            out = io.StringIO()
            with mock.patch.dict(pictures.PROJECTOR_BY_FAMILY, {"qwen": TINY}), mock.patch.dict(os.environ, {"STRATA_NO_ACQUIRE": "", "HF_ENDPOINT": ""}), \
                    mock.patch.object(pictures, "_online", return_value=False), redirect_stdout(out):
                code = pictures.main(["fetch", "--config", str(lay.config), "--engine-dir", str(lay.engine), "--vision", "yes", "--vram-gb", "12", "--make", "nvidia", "--arch", "120"])
            self.assertEqual(code, 5)
            last = json.loads(out.getvalue().strip().splitlines()[-1])
            self.assertEqual(last["ok"], False)
            self.assertEqual(last["error"]["code"], "offline")
            out = io.StringIO()
            with mock.patch.dict(pictures.PROJECTOR_BY_FAMILY, {"qwen": TINY}), mock.patch.dict(os.environ, {"STRATA_NO_ACQUIRE": "", "HF_ENDPOINT": lay.mirror.as_uri()}), \
                    mock.patch.object(pictures, "_online", return_value=True), redirect_stdout(out):
                code = pictures.main(["fetch", "--config", str(lay.config), "--engine-dir", str(lay.engine), "--vram-gb", "12", "--make", "nvidia", "--arch", "120"])
            lines = [json.loads(line) for line in out.getvalue().strip().splitlines()]
            self.assertEqual(code, 0, lines[-1])
            self.assertEqual([l["event"] for l in lines[:-1]], ["progress"] * (len(lines) - 1))
            self.assertTrue(lines[-1]["ok"] and lines[-1]["event"] == "result" and lines[-1]["fetched"])


if __name__ == "__main__":
    unittest.main()
