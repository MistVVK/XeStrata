"""0.2.17 GH-LIST #6240: the model list's "(installed)" tag looks at the model's files, not only at the name of a leftover run config."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from tests.terminal.launch_support import Home, catalog


def write_config(install: Path, tag: str, args: list) -> Path:
    install.mkdir(parents=True, exist_ok=True)
    p = install / f"strata-{tag}.json"
    p.write_text(json.dumps({"exe": "engine.exe", "tokenizer": "tokenizer.model", "model_name": tag, "args": args}), encoding="utf-8")
    return p


class InstalledNeedsFiles(unittest.TestCase):
    def test_a_run_config_whose_weights_are_gone_is_not_installed(self):
        with Home() as h:
            install = Path(h.env["STRATA_INSTALL_DIR"])
            weights = h.base / "data" / "Qwen3.8-Flash-Next-IQ3_S-00001-of-00002.gguf"
            write_config(install, "iq3_s", ["--native", str(weights), "--max-context", "65536"])
            cat = catalog()
            self.assertIsNotNone(cat.installed_config("iq3_s"), "the config file is there")
            self.assertFalse(cat.installed_with_files("iq3_s"))
            row = next(r for r in cat.rows() if r.id == "iq3_s")
            self.assertFalse(row.installed, "the row said (installed) from the file name alone")

    def test_with_its_weights_on_the_disk_it_is(self):
        with Home() as h:
            install = Path(h.env["STRATA_INSTALL_DIR"])
            weights = h.base / "data" / "Qwen3.8-Flash-Next-IQ3_S-00001-of-00002.gguf"
            weights.parent.mkdir(parents=True)
            weights.write_bytes(b"x")
            write_config(install, "iq3_s", ["--native", str(weights)])
            cat = catalog()
            self.assertTrue(cat.installed_with_files("iq3_s"))
            self.assertTrue(next(r for r in cat.rows() if r.id == "iq3_s").installed)

    def test_a_missing_pack_counts_but_a_missing_draft_layer_does_not(self):
        with Home() as h:
            install = Path(h.env["STRATA_INSTALL_DIR"])
            weights = h.base / "data" / "m.gguf"
            weights.parent.mkdir(parents=True)
            weights.write_bytes(b"x")
            write_config(install, "iq3_s", ["--native", str(weights), "--pack", str(h.base / "data" / "packs" / "gone"), "--mtp", str(h.base / "data" / "mtp" / "rt")])
            self.assertFalse(catalog().installed_with_files("iq3_s"), "its prepared pack is gone")
            (h.base / "data" / "packs" / "gone").mkdir(parents=True)
            self.assertTrue(catalog().installed_with_files("iq3_s"), "the draft layer is fetched on its own")

    def test_a_config_that_names_no_files_or_cannot_be_read_still_counts(self):
        with Home() as h:
            install = Path(h.env["STRATA_INSTALL_DIR"])
            write_config(install, "iq3_s", ["--max-context", "65536"])
            (install / "strata-iq3_xxs.json").write_text("{ not json", encoding="utf-8")
            cat = catalog()
            self.assertTrue(cat.installed_with_files("iq3_s"))
            self.assertTrue(cat.installed_with_files("iq3_xxs"))
            self.assertFalse(cat.installed_with_files("iq1_m"), "no config at all")


if __name__ == "__main__":
    unittest.main()
