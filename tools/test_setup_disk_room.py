# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
"""The low-RAM mode's copy of the experts (the Q2_0 AVX-512 pack) is checked for room where it is written - the data
folder's packs folder - and a refusal names that folder (upstream 16904848b).

    python -m unittest tools.test_setup_disk_room
"""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import setup  # noqa: E402


class DiskRoom(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="xestrata-room-"))
        self.addCleanup(lambda: __import__("shutil").rmtree(self.tmp, ignore_errors=True))
        self.models = self.tmp / "models"
        self.pack_dir = self.tmp / "data" / "packs" / "iq3_xxs"
        self.pack_dir.mkdir(parents=True)
        self.models.mkdir()

    def problem(self, free_models, free_packs, other, copy, note=""):
        space = {self.models: free_models, self.pack_dir.parent: free_packs}
        return setup.disk_room_problem(self.models, self.pack_dir, other, copy, note,
                                       free=lambda p: space[Path(p)])

    def test_the_model_folder_alone_can_refuse(self):
        msg, hint = self.problem(10, 500, other=20, copy=40, note=" (5 GB of the model is already there)")
        self.assertIn(str(self.models), msg)
        self.assertIn("5 GB of the model is already there", msg)
        self.assertIn("--models-dir", hint)

    def test_the_copy_is_measured_where_it_is_written(self):
        # the model folder has its room; the pack's folder has not (another drive: the two do not add up)
        space = {self.models: 25, self.pack_dir.parent: 30}
        other_drive = lambda p: SimpleNamespace(st_dev=1 if Path(p) == self.models else 2)   # noqa: E731
        with mock.patch.object(setup.os, "stat", other_drive):
            report = setup.disk_room_problem(self.models, self.pack_dir, 20, 40, free=lambda p: space[Path(p)])
        self.assertIsNotNone(report)
        self.assertIn(str(self.pack_dir), report[0])
        self.assertIn("40", report[0])
        space[self.pack_dir.parent] = 40
        with mock.patch.object(setup.os, "stat", other_drive):
            self.assertIsNone(setup.disk_room_problem(self.models, self.pack_dir, 20, 40,
                                                      free=lambda p: space[Path(p)]))

    def test_on_one_drive_both_add_up(self):
        with mock.patch.object(setup.os, "stat", lambda p: SimpleNamespace(st_dev=7)):
            self.assertIsNotNone(setup.disk_room_problem(self.models, self.pack_dir, 20, 40,
                                                         free=lambda p: 55))   # 20 + 40 > 55 there
            self.assertIsNone(setup.disk_room_problem(self.models, self.pack_dir, 20, 40, free=lambda p: 65))

    def test_without_a_copy_only_the_model_folder_counts(self):
        self.assertIsNone(self.problem(25, 0, other=20, copy=0))
        self.assertIsNotNone(self.problem(10, 1000, other=20, copy=0))


if __name__ == "__main__":
    unittest.main()
