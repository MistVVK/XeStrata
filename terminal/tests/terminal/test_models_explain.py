"""#4887: `strata models explain NAME` shows the folder lookup strata launch makes (Catalog.model_folder_trace, whose pick IS model_folder) and changes nothing.
Throw-away data folders only."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

from strata_cli import models_explain as ME
from tests.terminal.launch_support import HW_12, Home, catalog, run_cli


def snapshot(root: Path) -> list[tuple[str, int]]:
    return sorted((str(p.relative_to(root)), p.stat().st_size if p.is_file() else -1) for p in root.rglob("*"))


class Explain(unittest.TestCase):
    def setUp(self):
        home = Home()
        home.__enter__()
        self.addCleanup(home.__exit__, None, None, None)
        temp = tempfile.TemporaryDirectory(prefix="strata-explain-")
        self.addCleanup(temp.cleanup)
        self.data = Path(temp.name)
        self.cat = catalog(HW_12)

    def run_it(self, name, **kw):
        before = snapshot(self.data)
        r = ME.explain(name, data_dir=str(self.data), catalog=self.cat, **kw)
        self.assertEqual(snapshot(self.data), before, "explain writes nothing")
        fam, size = r["family"], r["model"]
        self.assertEqual(Path(r["folder"]), self.cat.model_folder(self.data, fam, size), "the explanation is the launch's own pick")
        return r

    def test_setups_spelling_is_found_with_its_files(self):
        folder = self.data / "models" / "IQ3_S"
        folder.mkdir(parents=True)
        (folder / "Qwen-IQ3_S-00001-of-00002.gguf").write_bytes(b"x" * 1000)
        (folder / "Qwen-IQ3_S-00002-of-00002.gguf").write_bytes(b"x" * 500)
        (folder / "mmproj-Qwen-BF16.gguf").write_bytes(b"x" * 10)
        for name in ("IQ3_S", "iq3_s", "qwen/IQ3_S"):
            r = self.run_it(name)
            self.assertEqual((r["family"], r["model"], r["rule"], r["found"], r["weights_bytes"]), ("qwen", "IQ3_S", "exact", True, 1500), name)
            self.assertEqual(sorted(f["picture_reader"] for f in r["gguf_files"]), [False, False, True])
        text = ME.words(r)
        self.assertIn("Uses: " + str(folder), text)
        self.assertIn("Model files: 2, 0.0 GB", text)

    def test_the_lower_case_tag_and_another_letter_case(self):
        (self.data / "models" / "iq3_s").mkdir(parents=True)
        r = self.run_it("IQ3_S")
        sensitive = not (self.data / "models" / "IQ3_S").is_dir()       # a case-sensitive disk (Linux): the two spellings are two folders
        self.assertEqual(r["rule"], "lower" if sensitive else "exact")
        if sensitive:
            (self.data / "models" / "iq3_s").rename(self.data / "models" / "Iq3_s")
            r = self.run_it("IQ3_S")
            self.assertEqual((r["rule"], Path(r["folder"]).name), ("case", "Iq3_s"))
            self.assertEqual([t["exists"] for t in r["tried"]], [False, False, True])

    def test_nothing_found_names_the_folders_that_are_there(self):
        for name in ("IQ3_XXS", "q2_0", "notes"):
            (self.data / "models" / name).mkdir(parents=True)
        r = self.run_it("IQ3_S")
        self.assertEqual((r["rule"], r["found"]), ("fallback", False))
        self.assertEqual(Path(r["folder"]), self.data / "models" / "IQ3_S")
        self.assertEqual(sorted(r["other_folders"]), ["IQ3_XXS", "notes", "q2_0"])
        self.assertIn("Not found. strata launch would download the model into", ME.words(r))

    def test_no_models_folder(self):
        r = self.run_it("IQ3_S")
        self.assertFalse(r["models_dir_exists"])
        self.assertIn("There is no models folder in it", ME.words(r))

    def test_an_unknown_name_is_refused_with_the_closest(self):
        with self.assertRaises(ValueError) as cm:
            ME.explain("IQ3_SS", data_dir=str(self.data), catalog=self.cat)
        self.assertIn("Did you mean", str(cm.exception))

    def test_the_command_and_its_json(self):
        (self.data / "models" / "IQ3_S").mkdir(parents=True)
        code, out, err = run_cli(["models", "explain", "IQ3_S", "--data-dir", str(self.data), "--output", "json"])
        self.assertEqual(code, 0, err)
        doc = json.loads(out)
        payload = doc.get("result", doc)
        self.assertEqual((payload["rule"], payload["family"]), ("exact", "qwen"))
        code, out, err = run_cli(["models", "explain", "nope", "--data-dir", str(self.data)])
        self.assertEqual(code, 2)


if __name__ == "__main__":
    unittest.main()
