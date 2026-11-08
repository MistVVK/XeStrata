"""0.2.10: for an Intel Arc card Strata builds the pack itself (setup.py --prepare-pack)."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

class IntelPack(unittest.TestCase):
    def test_prepare_pack_uses_the_vendored_gguf_py_and_runs_the_pack_tool(self):
        import setup
        calls = {}
        with tempfile.TemporaryDirectory() as d:
            gdir = Path(d) / "models"
            gdir.mkdir()
            fam = setup.FAMILIES["qwen"]
            shards = [gdir / setup.model_file(fam, "IQ3_XXS", i) for i in range(1, setup.model_shards(fam, "IQ3_XXS") + 1)]
            for s in shards:
                s.write_bytes(b"x")
            pack = Path(d) / "packs" / "p"

            def fake_run(cmd, cwd=None, env=None, check=True, quiet=False):
                calls["cmd"], calls["env"] = cmd, env
                (pack / "tokenizer").mkdir(parents=True)
                (pack / "tokenizer" / "vocab.json").write_text("{}")

            args = mock.Mock(gguf_dir=str(gdir), family="qwen", model="IQ3_XXS", prepare_pack=str(pack))
            fetched = []
            with mock.patch.object(setup, "get_llama_cpp", lambda: fetched.append(1)), mock.patch.object(setup, "run", fake_run), \
                    mock.patch.object(setup, "gguf_dir_problem", lambda *a: None), \
                    mock.patch.dict("os.environ", {k: v for k, v in __import__("os").environ.items() if k != "STRATA_GGUF_PY"}, clear=True):
                self.assertEqual(setup.prepare_pack_only(args), 0)
        self.assertIn("iq_pack.py", " ".join(map(str, calls["cmd"])))
        self.assertEqual(calls["env"]["STRATA_GGUF_PY"], str(setup.ROOT / "third_party" / "strata-gguf-py"))
        self.assertEqual(fetched, [], "the vendored gguf-py needs no download")


if __name__ == "__main__":
    unittest.main()
