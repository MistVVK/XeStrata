"""#4892: "Download and prepare" asked for the whole 128 GB on an NVFP4 pack that was already on the disk. The disk check looked in models/<family tag><model>
(models/nvfp4-NVFP4), but setup keeps a ready-made pack under models/<owner>--<repo> (setup.ready_where). The check now also looks there."""
from __future__ import annotations

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

RTX5090 = {"index": 0, "name": "NVIDIA GeForce RTX 5090", "vram_gb": 32.0, "arch": "120", "vendor": "nvidia", "usable": True}


class ReadyPackOnDisk(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.data = Path(tmp.name) / "Strata-data"
        (self.data / "models").mkdir(parents=True)

    def plan(self, free_gb=122.0):
        s = M.Strata(ROOT)
        s.hardware = lambda fresh=False: {"ram_gb": 128.0, "gpus": [dict(RTX5090)], "cpu": {"avx2": True}, "engine_vendor": "nvidia"}
        s.disk_free = lambda p: free_gb
        s.data_dir = lambda: self.data
        with mock.patch.object(M, "WIN", False):
            try:
                return M.Tools(s).install_plan("nvfp4", "NVFP4", 32768, "no", "int8", None, None, None, None, None, None, None, "auto")
            except M.ToolError as e:                       # the disk check is reported as the plan's refusal
                return str(e), None

    def test_the_pack_folder_helper_names_the_repository_layout(self):
        got = M.ready_pack_dir(self.data, setup.MODELS["NVFP4"])
        self.assertEqual(got, self.data / "models" / setup.MODELS["NVFP4"]["repo"].replace("/", "--"))
        self.assertIsNone(M.ready_pack_dir(self.data, {"download_gb": 3.0}))

    def test_an_empty_data_folder_still_needs_the_whole_download(self):
        plan, _ = self.plan(122.0)
        text = plan if isinstance(plan, str) else str(plan)
        self.assertIn("not enough free disk space", text)

    def test_a_pack_that_is_already_there_asks_for_working_room_only(self):
        folder = M.ready_pack_dir(self.data, setup.MODELS["NVFP4"])
        folder.mkdir(parents=True)
        for i in range(1, 7):
            (folder / f"shard-{i:05d}-of-00006.gguf").write_bytes(b"x")
        plan, _ = self.plan(122.0)
        text = plan if isinstance(plan, str) else str(plan)
        self.assertNotIn("not enough free disk space", text)


if __name__ == "__main__":
    unittest.main()
