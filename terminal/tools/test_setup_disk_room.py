"""The disk-room check of setup's own-files path (QA 0213): the low-RAM mode's copy of the experts is measured where it is written (the data
folder's packs folder), and the refusal names that folder, not the model folder of files that are already there.

    python -m unittest tools.test_setup_disk_room
"""
from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import setup  # noqa: E402

APPROVED_HINT = ("Free some space, move Strata's data folder to a bigger drive (strata launch --data-dir), "
                 "or set Low-RAM mode to Automatic to run without the copy.")


class DiskRoom(unittest.TestCase):
    def folders(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        models = Path(tmp.name) / "own-model-folder"
        pack = Path(tmp.name) / "data" / "packs" / "iq3_s"
        models.mkdir(parents=True)
        pack.parent.mkdir(parents=True)
        return models, pack

    def test_the_copy_of_the_experts_is_refused_in_the_words_the_owner_approved_and_names_the_packs_folder(self):
        models, pack = self.folders()
        problem = setup.disk_room_problem(models, pack, 1.0, 52.0, free=lambda _path: 22.0)   # the QA case: 22 GB free, the model files are already there
        self.assertIsNotNone(problem)
        message, hint = problem
        self.assertEqual(message, f"not enough free disk space for the low-RAM copy of the experts in {pack}: it needs about 53 GB and 22 GB are free there.")
        self.assertEqual(hint, APPROVED_HINT)
        self.assertNotIn(str(models), message)            # the refusal no longer names the folder of the person's own model files

    def test_the_copy_is_measured_on_the_packs_drive_not_on_the_model_drive(self):
        models, pack = self.folders()
        free = {models: 500.0, pack.parent: 22.0}
        real_stat = os.stat

        def other_device(path, *a, **k):                   # the packs folder sits on another drive than the model folder
            result = real_stat(path, *a, **k)
            if Path(path) == pack.parent:
                return os.stat_result((result.st_mode, result.st_ino, result.st_dev + 1, *tuple(result)[3:]))
            return result
        from unittest import mock
        with mock.patch.object(setup.os, "stat", other_device):
            problem = setup.disk_room_problem(models, pack, 1.0, 52.0, free=lambda path: free[path])
        self.assertIsNotNone(problem)
        self.assertIn("low-RAM copy of the experts", problem[0])
        self.assertIn("it needs about 52 GB and 22 GB are free there", problem[0])      # no other files on that drive: just the copy
        # and the other way round: a roomy packs drive, a full model drive: the model folder's refusal
        free = {models: 0.5, pack.parent: 500.0}
        with mock.patch.object(setup.os, "stat", other_device):
            problem = setup.disk_room_problem(models, pack, 1.0, 52.0, free=lambda path: free[path])
        self.assertIsNotNone(problem)
        self.assertIn(f"not enough free disk space in {models}", problem[0])

    def test_enough_room_and_no_copy(self):
        models, pack = self.folders()
        self.assertIsNone(setup.disk_room_problem(models, pack, 1.0, 52.0, free=lambda _path: 200.0))
        self.assertIsNone(setup.disk_room_problem(models, pack, 5.0, 0, free=lambda _path: 6.0))
        message, _hint = setup.disk_room_problem(models, pack, 5.0, 0, free=lambda _path: 4.0)
        self.assertTrue(message.startswith(f"not enough free disk space in {models}: need ~5 GB, 4 GB free"))     # a download that does not fit: the old sentence

    def test_the_old_refusal_for_the_copy_is_gone_from_the_run_path(self):
        source = (ROOT / "setup.py").read_text(encoding="utf-8")
        run = source[source.index("copy_gb = MODELS[model]"):]
        run = run[:run.index("if free_now < need:")]
        self.assertIn("disk_room_problem(", run)
        self.assertNotIn("not enough free disk space in {models_dir}", run)


if __name__ == "__main__":
    unittest.main()
