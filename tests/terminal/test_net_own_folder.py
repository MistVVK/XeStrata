"""Lane NET item 5: --model-path and the data folder Strata guesses from it (the folder above a `models` folder). Strata keeps its own files (the
prepared data, the picture reader) there only when it can write there and it is not a Hugging Face cache; otherwise in the data folder every
command agrees on, with one line saying so, instead of stopping with "Strata cannot write to ...". The finish marks of the person's own files go
to Strata's own marks/ folder, never beside the files."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import setup
from strata_cli.launch import settings as S
from strata_cli.launch.catalog import Catalog
from strata_cli.launch.runner import Runner
from strata_control import discovery
from tests.terminal.launch_support import HW_12, Home, catalog
from tests.terminal.test_own_files import IQ3XXS, gguf_bytes


class Base(unittest.TestCase):
    def setUp(self):
        self.home = Home().__enter__()
        self.addCleanup(self.home.__exit__, None, None, None)
        p = mock.patch.object(Catalog, "hardware", lambda self, fresh=False: dict(HW_12, disk_free_gb=500.0))
        p.start()
        self.addCleanup(p.stop)
        t = tempfile.TemporaryDirectory(prefix="strata-netown-")
        self.addCleanup(t.cleanup)
        self.base = Path(t.name)

    def own(self, folder: Path) -> Path:
        folder.mkdir(parents=True, exist_ok=True)
        for n in IQ3XXS:
            (folder / n).write_bytes(gguf_bytes(["IQ3_XXS"]))
        return folder


class HuggingFaceCache(Base):
    def test_cache_layouts(self):
        snap = self.base / "hub" / "models--unsloth--Qwen3.8-Flash-Next-GGUF" / "snapshots" / "38bb39ee" / "UD-Q4_K_XL"
        snap.mkdir(parents=True)
        env = {"HOME": str(self.base / "h")}
        self.assertTrue(discovery.in_hf_cache(snap, env))
        self.assertTrue(discovery.in_hf_cache(self.base / "hub", env), "a folder that holds models--* folders is a cache")
        self.assertTrue(discovery.in_hf_cache(self.base / "elsewhere" / "x", {"HF_HOME": str(self.base / "elsewhere")}))
        self.assertFalse(discovery.in_hf_cache(self.base / "data", env))

    def test_a_guessed_folder_inside_a_cache_is_not_used(self):
        own = self.own(self.base / "hfhome" / "models" / "IQ3_XXS")
        first = own / IQ3XXS[0]
        self.assertEqual(discovery.prepared_beside(first)["data_dir"], self.base / "hfhome")
        with mock.patch.dict(os.environ, {"HF_HOME": str(self.base / "hfhome")}):
            self.assertIsNone(discovery.writable_beside(first))
        self.assertEqual(discovery.writable_beside(first), self.base / "hfhome")


@unittest.skipIf(os.name == "nt" or (hasattr(os, "geteuid") and os.geteuid() == 0), "a read-only folder needs POSIX permissions and no root")
class ReadOnlyFolder(Base):
    def test_launch_uses_the_data_folder_and_says_so(self):
        top = self.base / "lender"
        own = self.own(top / "models" / "IQ3_XXS")
        os.chmod(top, 0o555)
        self.addCleanup(os.chmod, top, 0o755)
        r = Runner(S.LaunchSettings(family="qwen", model="IQ3_XXS", vision="off", mtp="off", drafting="off", context=32768, model_path=str(own)),
                   catalog=catalog(HW_12))
        self.assertIsNone(r.existing_data_dir(), "not the folder Strata cannot write to: setup's own data folder")
        self.assertEqual(r.beside_refused(), top)
        self.assertNotIn("--data-dir", r.setup_args())
        self.assertIn("--gguf-dir", r.setup_args())

    def test_a_writable_folder_is_still_used(self):
        top = self.base / "mine"
        own = self.own(top / "models" / "IQ3_XXS")
        r = Runner(S.LaunchSettings(family="qwen", model="IQ3_XXS", vision="off", mtp="off", drafting="off", context=32768, model_path=str(own)),
                   catalog=catalog(HW_12))
        self.assertEqual(r.existing_data_dir(), top)
        self.assertIsNone(r.beside_refused())


class FinishMarks(Base):
    def test_own_files_are_marked_in_stratas_folder(self):
        own = self.own(self.base / "mine" / "IQ3_XXS")
        root = self.base / "strata"
        root.mkdir()
        shard = own / IQ3XXS[0]
        with mock.patch.object(setup, "ROOT", root), mock.patch.object(setup, "_OWN_DIRS", []):
            setup.own_files_in(own)
            setup.mark(shard, "whole")
            self.assertFalse(shard.with_name(shard.name + ".done").exists(), "nothing is written beside the person's files")
            self.assertTrue(setup.done(shard))
            self.assertEqual(len(list((root / "marks").glob("*.done"))), 1)
            other = self.base / "data" / "x.gguf"
            other.parent.mkdir()
            other.write_bytes(b"x")
            setup.mark(other)                                # a download of Strata's own: beside it, as before
            self.assertTrue(other.with_name("x.gguf.done").exists())


if __name__ == "__main__":
    unittest.main()
