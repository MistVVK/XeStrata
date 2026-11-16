"""0.2.10: for an Intel Arc card Strata builds the pack itself (setup.py --prepare-pack)."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

class IntelPack(unittest.TestCase):
    def test_prepare_pack_runs_the_pack_tool_with_its_own_gguf_code(self):
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
            fetched, said = [], []
            with mock.patch.object(setup, "get_llama_cpp", lambda: fetched.append(1)), mock.patch.object(setup, "run", fake_run), \
                    mock.patch.object(setup, "ok", said.append), \
                    mock.patch.object(setup, "gguf_dir_problem", lambda *a: None), \
                    mock.patch.dict("os.environ", {k: v for k, v in __import__("os").environ.items() if k != "STRATA_GGUF_PY"}, clear=True):
                self.assertEqual(setup.prepare_pack_only(args), 0)
        self.assertIn("iq_pack.py", " ".join(map(str, calls["cmd"])))
        self.assertNotIn("STRATA_GGUF_PY", calls["env"], "the tool uses tools/strata_gguf.py unless the user named a gguf-py")
        self.assertEqual(fetched, [], "nothing is downloaded")
        self.assertIn("model file reader ready", said)             # the step line (owner-approved 11-16): no llama.cpp / gguf-py named
        self.assertFalse([line for line in said if "llama.cpp" in line or "gguf-py" in line])


if __name__ == "__main__":
    unittest.main()
