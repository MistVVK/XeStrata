# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
"""SenseNova-Loop-9B's QSA family offers the four files pinned at the revision.

    python -m unittest tools.test_setup_sensenova_qsa
"""
from __future__ import annotations

import unittest

import setup


class SenseNovaQSAFiles(unittest.TestCase):
    def test_four_pinned_files(self):
        names = {"Q4_K_M", "Q8_0", "BF16", "Q5_K_M"}
        self.assertEqual(set(setup.SENSENOVA_QSA_FILES), names)
        models = setup.FAMILIES["sensenova-loop-qsa"]["models"]
        self.assertEqual(set(models), names)
        for name, d in models.items():
            pin = setup.REPO_PINS[d["repo"]][d["file"]]
            self.assertEqual(pin, (d["size"], d["sha256"]))
            self.assertEqual(len(d["sha256"]), 64)
            self.assertIn("/resolve/" + setup.HF_REVISIONS[d["repo"]] + "/", setup.hf(d["repo"]))

    def test_picture_reader_and_terms(self):
        fam = setup.FAMILIES["sensenova-loop-qsa"]
        reader = fam["mmproj"]
        self.assertEqual(setup.REPO_PINS[reader["repo"]][reader["path"]], (reader["size"], reader["sha256"]))
        self.assertEqual(setup.hf(reader["repo"]),
                         f"https://huggingface.co/{reader['repo']}/resolve/{setup.HF_REVISIONS[reader['repo']]}/")
        repos = dict(setup.family_repos("sensenova-loop-qsa"))
        self.assertEqual(repos["model"], setup.SENSENOVA_QSA_REPO)
        self.assertEqual(repos["mmproj"], reader["repo"])
        self.assertIn("Apache", setup.family_license("sensenova-loop-qsa"))


if __name__ == "__main__":
    unittest.main()
