"""Lane NET item 4: the plan says what the start does.

4a. A copy of the model installed in another data folder is not what a launch with --data-dir uses (installed_serves_request): the start runs setup
    with the given folder, and setup takes model files only from there and from the folders it reuses (the remembered data folder, Strata's own). The plan
    counted the other copy as installed ("About 1.2 GB": the start then fetched 112 GB, lane 10 on the V100 box).
4b. The plan took any mmproj*.gguf as the picture reader; setup takes only the family's pinned file name (beside a person's own files by name, in the folder
    above it or its models/ only at the pinned size) and downloads 0.9 GB otherwise. Which file setup accepts is NOT changed here.
Nothing is started, probed or downloaded."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from strata_cli.launch import modelstate
from strata_cli.launch import settings as S
from strata_cli.launch.catalog import Catalog
from strata_cli.launch.runner import Runner
from tests.terminal.launch_support import HW_12, Home, catalog, not_intel
from tests.terminal.test_own_files import IQ3XXS, gguf_bytes

PINNED = "mmproj-Qwen3.8-Flash-Next-BF16.gguf"
PINNED_SIZE = 907543008


class Base(unittest.TestCase):
    def setUp(self):
        self.home = Home().__enter__()
        self.addCleanup(self.home.__exit__, None, None, None)
        p = mock.patch.object(Catalog, "hardware", lambda self, fresh=False: dict(HW_12, disk_free_gb=500.0))
        p.start()
        self.addCleanup(p.stop)
        not_intel(self)                  # the plan asks the machine that runs the test whether the engine is the Intel one (p1006 has an Arc card: its plan drops the picture reader)
        t = tempfile.TemporaryDirectory(prefix="strata-nettruth-")
        self.addCleanup(t.cleanup)
        self.base = Path(t.name)

    def model_files(self, folder: Path) -> Path:
        folder.mkdir(parents=True, exist_ok=True)
        for n in IQ3XXS:
            (folder / n).write_bytes(gguf_bytes(["IQ3_XXS"]))
        return folder

    def installed_in(self, data: Path) -> Path:
        """An installed IQ3_XXS whose files are all in `data` (models/, packs/), its run config written there."""
        folder = self.model_files(data / "models" / "IQ3_XXS")
        pack = data / "packs" / "iq3_xxs"
        pack.mkdir(parents=True, exist_ok=True)
        cfg = data / "strata-iq3_xxs.json"
        cfg.write_text(json.dumps({"exe": "x", "args": ["--native", str(folder / IQ3XXS[0]), "--pack", str(pack)]}), encoding="utf-8")
        return cfg

    def runner(self, cfg: Path | None, remembered: Path, **kw) -> Runner:
        r = Runner(S.LaunchSettings(family="qwen", model="IQ3_XXS", vision="off", mtp="off", drafting="off", context=32768, **kw),
                   catalog=catalog(HW_12))
        for p in (mock.patch.object(Runner, "installed_state", lambda self: (cfg, {"images": "off"}) if cfg else (None, None)),
                  mock.patch.object(Catalog, "data_dir", lambda self: remembered)):
            p.start()
            self.addCleanup(p.stop)
        return r


class PlanUsesTheGivenDataDir(Base):
    def test_a_copy_in_another_folder_is_not_counted(self):
        other, new = self.base / "other", self.base / "new"
        cfg = self.installed_in(other)
        r = self.runner(cfg, self.base / "remembered", data_dir=str(new))
        self.assertFalse(r.installed_serves_request(cfg))
        self.assertEqual(r.other_copy(), other)
        st = r.model_state()
        self.assertFalse(st.installed)
        self.assertEqual([p.kind for p in st.parts], ["weights"])
        self.assertGreater(st.remaining_gb, 10)
        plan = r.download_plan()
        self.assertTrue(plan["text"].startswith(f"About {plan['gb']:.1f} GB to download into {new}: the model ("), plan["text"])
        self.assertIn(f"The copy in {other} is not used with this --data-dir.", plan["text"])

    def test_a_copy_in_the_remembered_folder_is_what_setup_reuses(self):
        remembered, new = self.base / "remembered", self.base / "new"
        cfg = self.installed_in(remembered)
        r = self.runner(cfg, remembered, data_dir=str(new))
        self.assertFalse(r.installed_serves_request(cfg))
        self.assertIsNone(r.other_copy())                    # setup.data_folder_for keeps it as a source: nothing is fetched again
        self.assertTrue(r.model_state().installed)
        self.assertNotIn("is not used with this --data-dir", r.download_plan()["text"])

    def test_the_same_folder_is_the_installed_copy_as_before(self):
        other = self.base / "other"
        cfg = self.installed_in(other)
        r = self.runner(cfg, self.base / "remembered", data_dir=str(other))
        self.assertTrue(r.installed_serves_request(cfg))
        self.assertIsNone(r.other_copy())
        self.assertTrue(r.model_state().installed)

    def test_no_data_dir_is_the_installed_copy_as_before(self):
        cfg = self.installed_in(self.base / "other")
        r = self.runner(cfg, self.base / "remembered")
        self.assertIsNone(r.other_copy())
        self.assertTrue(r.model_state().installed)


class PictureReaderBySetupsRule(Base):
    def sparse(self, path: Path, size: int) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            f.truncate(size)
        return path

    def test_reader_present_follows_setup(self):
        c = catalog(HW_12)
        own = self.model_files(self.base / "mine" / "IQ3_XXS")
        self.assertEqual(modelstate.mmproj_size(c, "qwen", PINNED), PINNED_SIZE)
        (own / "mmproj-F16.gguf").write_bytes(b"x")
        self.assertFalse(modelstate.reader_present(c, "qwen", [], own=own), "another picture file is not the one setup takes")
        self.sparse(self.base / "mine" / PINNED, 1000)
        self.assertFalse(modelstate.reader_present(c, "qwen", [], own=own), "the folder above: only at the pinned size")
        self.sparse(self.base / "mine" / PINNED, PINNED_SIZE)
        self.assertTrue(modelstate.reader_present(c, "qwen", [], own=own))
        (self.base / "mine" / PINNED).unlink()
        (own / PINNED).write_bytes(b"x")
        self.assertTrue(modelstate.reader_present(c, "qwen", [], own=own), "beside the files: by name, as setup's mmproj_beside")
        data = self.base / "data"
        self.assertFalse(modelstate.reader_present(c, "qwen", [data]))
        (data / "models").mkdir(parents=True)
        (data / "models" / "mmproj-F16.gguf").write_bytes(b"x")
        self.assertFalse(modelstate.reader_present(c, "qwen", [data]))
        (data / "models" / PINNED).write_bytes(b"x")
        self.assertTrue(modelstate.reader_present(c, "qwen", [data]))

    def test_the_plan_counts_the_reader_when_only_another_picture_file_is_there(self):
        own = self.model_files(self.base / "mine" / "IQ3_XXS")
        (own / "mmproj-F16.gguf").write_bytes(b"x")                 # Unsloth's file beside the shards (the V100 box's layout)
        r = Runner(S.LaunchSettings(family="qwen", model="IQ3_XXS", vision="gpu", mtp="off", drafting="off", context=32768,
                                    model_path=str(own), data_dir=str(self.base / "data")), catalog=catalog(HW_12))
        remembered = self.base / "remembered"
        with mock.patch.object(Catalog, "data_dir", lambda _cat: remembered):
            kinds = [p["kind"] for p in r.download_plan()["parts"]]
            self.assertIn("mmproj", kinds, "setup downloads the pinned reader: the plan says so")
            (own / PINNED).write_bytes(b"x")
            kinds = [p["kind"] for p in r.download_plan()["parts"]]
        self.assertNotIn("mmproj", kinds)


if __name__ == "__main__":
    unittest.main()
